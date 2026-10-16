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
    uint4 _S1 = uint4((kernelContext_1->islands_0+kernelContext_1->params_0->halt_index_0)->info_0) ;
    bool _S2;
    if(((_S1.z) & 1U) != 0U)
    {
        _S2 = true;
    }
    else
    {
        uint _S3 = _S1.y;
        if(_S3 != 0U)
        {
            _S2 = _S3 <= (isl_0->info_0.w);
        }
        else
        {
            _S2 = false;
        }
    }
    return _S2;
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
    float3 _S4 = float3(0.0f) ;
    (&rg_0)->torque_0 = _S4;
    (&rg_0)->a_0 = _S4;
    (&rg_0)->alpha_0 = _S4;
    return rg_0;
}

float time_since_0(float4 origin_0, uint k_0, float dt_1, KernelContext_0 thread* kernelContext_2)
{
    return kernelContext_2->params_0->t_hi_0 - origin_0.x + (kernelContext_2->params_0->t_lo_0 - origin_0.y) + float(k_0) * dt_1;
}

float table_eval_0(uint offset_0, uint count_0, float tau_0, KernelContext_0 thread* kernelContext_3)
{
    float4 _S5 = float4(*(kernelContext_3->loads_0+offset_0)) ;
    if(tau_0 <= (_S5.x))
    {
        return _S5.y;
    }
    uint i_0 = 1U;
    for(;;)
    {
        if(i_0 < count_0)
        {
        }
        else
        {
            break;
        }
        uint _S6 = offset_0 + i_0;
        float4 _S7 = float4(*(kernelContext_3->loads_0+_S6)) ;
        float _S8 = _S7.x;
        if(tau_0 <= _S8)
        {
            float4 _S9 = float4(*(kernelContext_3->loads_0+(_S6 - 1U))) ;
            float _S10 = _S9.x;
            float _S11 = _S9.y;
            return _S11 + (tau_0 - _S10) / max(_S8 - _S10, 1.00000000317107685e-30f) * (_S7.y - _S11);
        }
        i_0 = i_0 + 1U;
    }
    return (float4(*(kernelContext_3->loads_0+(offset_0 + count_0 - 1U))) ).y;
}

float eval_function_0(uint term_0, uint k_1, float dt_2, float shift_0, KernelContext_0 thread* kernelContext_4)
{
    uint _S12 = 5U * term_0;
    uint4 info_2 = (as_type<uint4>((float4(*(kernelContext_4->loads_0+_S12)) )));
    float4 _S13 = float4(*(kernelContext_4->loads_0+(_S12 + 3U))) ;
    float4 _S14 = float4(*(kernelContext_4->loads_0+(_S12 + 4U))) ;
    uint kind_0 = info_2.z;
    if(kind_0 == 0U)
    {
        return _S13.z;
    }
    float _S15 = time_since_0(_S13, k_1, dt_2, kernelContext_4);
    float tau_1 = _S15 + shift_0;
    float shape_1;
    if(kind_0 == 1U)
    {
        if(tau_1 <= 0.0f)
        {
            shape_1 = 0.0f;
        }
        else
        {
            float _S16 = _S14.x;
            if(tau_1 >= _S16)
            {
                shape_1 = _S14.y;
            }
            else
            {
                shape_1 = _S14.y * tau_1 / _S16;
            }
        }
        return shape_1;
    }
    bool _S17;
    if(kind_0 == 2U)
    {
        if(tau_1 < 0.0f)
        {
            _S17 = true;
        }
        else
        {
            _S17 = tau_1 > (_S14.x);
        }
        if(_S17)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S14.y * sin(3.14159274101257324f * tau_1 / _S14.x);
        }
        return shape_1;
    }
    if(kind_0 == 3U)
    {
        float sn_0 = tau_1 / _S14.y;
        if(sn_0 < 0.0f)
        {
            _S17 = true;
        }
        else
        {
            _S17 = sn_0 > 1.0f;
        }
        if(_S17)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S14.x * (1.0f - sn_0) * exp(- _S14.z * sn_0);
        }
        return shape_1;
    }
    if(kind_0 == 4U)
    {
        float _S18 = table_eval_0(info_2.w, (as_type<uint>((_S14.x))), tau_1, kernelContext_4);
        return _S18;
    }
    if(kind_0 == 5U)
    {
        (*kernelContext_4->g_seen_0)[1U] = 1U;
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        float sn_1 = tau_1 / _S14.x;
        if(sn_1 < 0.0f)
        {
            _S17 = true;
        }
        else
        {
            _S17 = sn_1 > 1.0f;
        }
        if(_S17)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = (1.0f - sn_1) * exp(- _S14.y * sn_1);
        }
        float clearing_0 = _S13.w;
        float relax_0;
        if(clearing_0 > 0.0f)
        {
            relax_0 = max(1.0f - tau_1 / clearing_0, 0.0f);
        }
        else
        {
            relax_0 = 0.0f;
        }
        float _S19 = _S14.w;
        return (_S19 + (_S14.z - _S19) * relax_0) * shape_1;
    }
    if(kind_0 == 7U)
    {
        (*kernelContext_4->g_seen_0)[2U] = 1U;
        if(tau_1 < 0.0f)
        {
            return 0.0f;
        }
        float _S20 = _S14.y;
        if(tau_1 < _S20)
        {
            return _S14.x;
        }
        float s_0 = tau_1 - _S20;
        float _S21 = _S14.w;
        if(s_0 > _S21)
        {
            shape_1 = 0.0f;
        }
        else
        {
            shape_1 = _S14.z * sin(3.14159274101257324f * s_0 / _S21);
        }
        return shape_1;
    }
    float _S22 = _S14.x;
    if(_S22 <= 0.0f)
    {
        return 0.0f;
    }
    return clamp(1.0f - tau_1 / _S22, 0.0f, 1.0f);
}

float3 rotate_0(const Quat_0 thread* q_1, float3 v_0)
{
    float3 qv_0 = float3(q_1->x_0, q_1->y_0, q_1->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_1->w_0)  + cross(qv_0, t_0);
}

void chunk_loads_0(uint _S23, uint _S24, const Quat_0 thread* _S25, uint _S26, float _S27, bool _S28, float _S29, float3 thread* _S30, float3 thread* _S31, float3 thread* _S32, float3 thread* _S33, KernelContext_0 thread* kernelContext_5)
{
    bool _S34;
    ChunkStatic_natural_0 device* _S35 = kernelContext_5->chunks_0+_S24;
    float3 _S36 = float3(0.0f) ;
    *_S30 = _S36;
    *_S31 = _S36;
    *_S32 = _S36;
    *_S33 = _S36;
    uint4 _S37 = uint4(_S35->load_range_0) ;
    uint term_1 = _S37.x;
    for(;;)
    {
        if(term_1 < (_S37.y))
        {
        }
        else
        {
            break;
        }
        (*kernelContext_5->g_seen_0)[0U] = 1U;
        uint _S38 = 5U * term_1;
        uint _S39 = (as_type<uint4>((float4(*(kernelContext_5->loads_0+_S38)) ))).y;
        bool replacement_0 = _S39 == 2U;
        float4 _S40 = float4(*(kernelContext_5->loads_0+(_S38 + 1U))) ;
        float4 _S41 = float4(*(kernelContext_5->loads_0+(_S38 + 2U))) ;
        float _S42;
        if(replacement_0)
        {
            _S42 = _S29;
        }
        else
        {
            _S42 = 0.0f;
        }
        float _S43 = eval_function_0(term_1, _S26, _S27, _S42, kernelContext_5);
        if(replacement_0)
        {
            float3 _S44 = float3(_S43) ;
            *_S32 = *_S32 + _S40.xyz * _S44;
            *_S33 = *_S33 + _S41.xyz * _S44;
            term_1 = term_1 + 1U;
            continue;
        }
        if(_S39 == 0U)
        {
            _S34 = true;
        }
        else
        {
            _S34 = _S39 == 3U;
        }
        float3 fw_0;
        if(_S34)
        {
            fw_0 = _S40.xyz * float3(_S43) ;
        }
        else
        {
            float3 _S45 = rotate_0(_S25, _S40.xyz);
            fw_0 = _S45 * float3((- _S43 * _S40.w)) ;
        }
        float3 lever_0;
        if(_S39 == 3U)
        {
            lever_0 = _S41.xyz - (float4(*(kernelContext_5->state_0+4U * _S23)) ).xyz;
        }
        else
        {
            lever_0 = _S41.xyz;
        }
        *_S30 = *_S30 + fw_0;
        float3 _S46 = rotate_0(_S25, lever_0);
        *_S31 = *_S31 + cross(_S46, fw_0);
        term_1 = term_1 + 1U;
    }
    if(_S28)
    {
        _S34 = ((uint4(_S35->cinfo_0) ).z) != 0U;
    }
    else
    {
        _S34 = false;
    }
    if(_S34)
    {
        uint4 _S47 = uint4(_S35->cinfo_0) ;
        uint g_0 = _S47.x;
        for(;;)
        {
            if(g_0 < (_S47.y))
            {
            }
            else
            {
                break;
            }
            uint _S48 = 2U * g_0;
            *_S30 = *_S30 + (float4(*(kernelContext_5->scratch_0+(kernelContext_5->params_0->seg_base_0 + _S48))) ).xyz;
            *_S31 = *_S31 + (float4(*(kernelContext_5->scratch_0+(kernelContext_5->params_0->seg_base_0 + _S48 + 1U))) ).xyz;
            g_0 = g_0 + 1U;
        }
    }
    return;
}

void chunk_external_0(uint _S49, uint _S50, const Quat_0 thread* _S51, uint _S52, float _S53, bool _S54, float3 thread* _S55, float3 thread* _S56, KernelContext_0 thread* kernelContext_6)
{
    thread float3 rf_0;
    thread float3 rm_0;
    chunk_loads_0(_S49, _S50, _S51, _S52, _S53, _S54, 0.0f, _S55, _S56, &rf_0, &rm_0, kernelContext_6);
    return;
}

void record_chunk_load_0(uint c_0, float3 f_0, float3 t_1, KernelContext_0 thread* kernelContext_7)
{
    if((kernelContext_7->params_0->solve_mode_0) == 0U)
    {
        return;
    }
    uint _S57 = 2U * c_0;
    *(kernelContext_7->scratch_0+(kernelContext_7->params_0->cload_base_0 + _S57)) = packed_float4(float4(f_0, 0.0f)) ;
    *(kernelContext_7->scratch_0+(kernelContext_7->params_0->cload_base_0 + _S57 + 1U)) = packed_float4(float4(t_1, 0.0f)) ;
    *(kernelContext_7->scratch_0+(kernelContext_7->params_0->cframe_base_0 + _S57)) = packed_float4(float4((float4(*(kernelContext_7->scratch_0+(kernelContext_7->params_0->cframe_base_0 + _S57))) ).xyz + f_0, 0.0f)) ;
    *(kernelContext_7->scratch_0+(kernelContext_7->params_0->cframe_base_0 + _S57 + 1U)) = packed_float4(float4((float4(*(kernelContext_7->scratch_0+(kernelContext_7->params_0->cframe_base_0 + _S57 + 1U))) ).xyz + t_1, 0.0f)) ;
    return;
}

float settled_chunk_load_0(uint c_1, const Quat_0 thread* rot_1, uint k_2, float dt_3, bool contact_0, KernelContext_0 thread* kernelContext_8)
{
    thread float3 f_1;
    thread float3 t_2;
    chunk_external_0(c_1, c_1, rot_1, k_2, dt_3, contact_0, &f_1, &t_2, kernelContext_8);
    record_chunk_load_0(c_1, f_1, t_2, kernelContext_8);
    return length(f_1);
}

void group_sum2_0(uint tid_1, float4 thread* a_1, float4 thread* b_0, KernelContext_0 thread* kernelContext_9)
{
    float4 wa_0 = simd_sum(*a_1);
    float4 wb_0 = simd_sum(*b_0);
    if((tid_1 % 32U) == 0U)
    {
        (*kernelContext_9->g_part_a_0)[tid_1 / 32U] = wa_0;
        (*kernelContext_9->g_part_b_0)[tid_1 / 32U] = wb_0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 _S58 = (*kernelContext_9->g_part_a_0)[int(0)];
    float4 _S59 = (*kernelContext_9->g_part_b_0)[int(0)];
    uint w_2 = 1U;
    float4 sa_0 = _S58;
    float4 sb_0 = _S59;
    for(;;)
    {
        if(w_2 < 8U)
        {
        }
        else
        {
            break;
        }
        float4 sa_1 = sa_0 + (*kernelContext_9->g_part_a_0)[w_2];
        float4 sb_1 = sb_0 + (*kernelContext_9->g_part_b_0)[w_2];
        w_2 = w_2 + 1U;
        sa_0 = sa_1;
        sb_0 = sb_1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *a_1 = sa_0;
    *b_0 = sb_0;
    return;
}

void group_sum3_0(uint tid_2, float3 thread* a_2, float3 thread* b_1, KernelContext_0 thread* kernelContext_10)
{
    thread float4 x_1 = float4(*a_2, 0.0f);
    thread float4 y_1 = float4(*b_1, 0.0f);
    group_sum2_0(tid_2, &x_1, &y_1, kernelContext_10);
    *a_2 = x_1.xyz;
    *b_1 = y_1.xyz;
    return;
}

void net_load_0(uint c_2, const Island_0 thread* isl_2, const Rigid_0 thread* rg_1, uint k_3, float dt_4, bool contact_1, float3 thread* f_2, float3 thread* t_3, KernelContext_0 thread* kernelContext_11)
{
    ChunkStatic_natural_0 device* _S60 = kernelContext_11->chunks_0+c_2;
    thread float3 fl_0;
    thread float3 tl_0;
    thread float3 rep_f_0;
    thread float3 rep_m_0;
    chunk_loads_0(c_2, c_2, &rg_1->rot_0, k_3, dt_4, contact_1, 0.0f, &fl_0, &tl_0, &rep_f_0, &rep_m_0, kernelContext_11);
    float4 _S61 = float4(_S60->center_0) ;
    float3 fc_0 = fl_0 + kernelContext_11->params_0->gravity_0.xyz * float3(_S61.w) ;
    float3 _S62 = _S61.xyz;
    float3 _S63 = isl_2->com_0.xyz;
    float3 _S64 = rotate_0(&rg_1->rot_0, _S62 + (float4(*(kernelContext_11->state_0+4U * c_2)) ).xyz - _S63);
    *f_2 = *f_2 + fc_0;
    *t_3 = *t_3 + (cross(_S64, fc_0) + tl_0);
    uint4 _S65 = uint4(_S60->load_range_0) ;
    if((_S65.y) > (_S65.x))
    {
        float3 _S66 = rotate_0(&rg_1->rot_0, rep_f_0);
        *f_2 = *f_2 + _S66;
        float3 _S67 = rotate_0(&rg_1->rot_0, _S62 - _S63);
        float3 _S68 = cross(_S67, _S66);
        float3 _S69 = rotate_0(&rg_1->rot_0, rep_m_0);
        *t_3 = *t_3 + (_S68 + _S69);
    }
    return;
}

float3 inverse_rotate_0(const Quat_0 thread* q_2, float3 v_1)
{
    thread Quat_0 c_3;
    (&c_3)->w_0 = q_2->w_0;
    (&c_3)->x_0 = - q_2->x_0;
    (&c_3)->y_0 = - q_2->y_0;
    (&c_3)->z_0 = - q_2->z_0;
    thread Quat_0 _S70 = c_3;
    float3 _S71 = rotate_0(&_S70, v_1);
    return _S71;
}

float3 rows_mul_0(float4 r0_0, float4 r1_0, float4 r2_0, float3 v_2)
{
    return float3(dot(r0_0.xyz, v_2), dot(r1_0.xyz, v_2), dot(r2_0.xyz, v_2));
}

float3 rotate_1(const Quat_0 thread* q_3, float3 v_3)
{
    float3 qv_1 = float3(q_3->x_0, q_3->y_0, q_3->z_0);
    float3 t_4 = cross(qv_1, v_3) * float3(2.0f) ;
    return v_3 + t_4 * float3(q_3->w_0)  + cross(qv_1, t_4);
}

float3 world_mul_0(const Quat_0 thread* q_4, float4 r0_1, float4 r1_1, float4 r2_1, float3 v_4)
{
    float3 _S72 = inverse_rotate_0(q_4, v_4);
    float3 _S73 = rotate_1(q_4, rows_mul_0(r0_1, r1_1, r2_1, _S72));
    return _S73;
}

void rigid_acceleration_0(const Island_0 thread* isl_3, Rigid_0 thread* rg_2, float3 f_3, float3 t_5)
{
    thread Quat_0 _S74 = rg_2->rot_0;
    float3 _S75 = world_mul_0(&_S74, isl_3->inertia0_0, isl_3->inertia1_0, isl_3->inertia2_0, rg_2->w_1);
    rg_2->a_0 = f_3 / float3(isl_3->com_0.w) ;
    float3 _S76 = t_5 - cross(rg_2->w_1, _S75);
    thread Quat_0 _S77 = rg_2->rot_0;
    float3 _S78 = world_mul_0(&_S77, isl_3->inv0_0, isl_3->inv1_0, isl_3->inv2_0, _S76);
    rg_2->alpha_0 = _S78;
    rg_2->torque_0 = t_5;
    return;
}

float3 inverse_rotate_1(const Quat_0 thread* q_5, float3 v_5)
{
    thread Quat_0 c_4;
    (&c_4)->w_0 = q_5->w_0;
    (&c_4)->x_0 = - q_5->x_0;
    (&c_4)->y_0 = - q_5->y_0;
    (&c_4)->z_0 = - q_5->z_0;
    thread Quat_0 _S79 = c_4;
    float3 _S80 = rotate_0(&_S79, v_5);
    return _S80;
}

void comp_add1_0(float thread* sum_0, float thread* err_0, float x_2)
{
    float t_6 = *sum_0 + x_2;
    if((abs(*sum_0)) >= (abs(x_2)))
    {
        *err_0 = *err_0 + (*sum_0 - t_6 + x_2);
    }
    else
    {
        *err_0 = *err_0 + (x_2 - t_6 + *sum_0);
    }
    *sum_0 = t_6;
    return;
}

void chunk_update_0(uint c_5, const Island_0 thread* isl_4, const Rigid_0 thread* rg_3, float dt_5, bool rml_0, uint step_0, bool contact_2, float thread* work_0, float thread* work_err_0, KernelContext_0 thread* kernelContext_12)
{
    ChunkStatic_natural_0 device* _S81 = kernelContext_12->chunks_0+c_5;
    float3 _S82 = float3(0.0f) ;
    uint _S83 = kernelContext_12->index_0[c_5];
    float peak_0 = 0.0f;
    uint e_0 = _S83;
    float3 fi_0 = _S82;
    float3 mi_0 = _S82;
    for(;;)
    {
        if(e_0 < (kernelContext_12->index_0)[c_5 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_0 = kernelContext_12->index_0[e_0];
        uint _S84 = 3U * (entry_0 >> 1U);
        float4 _S85 = float4(*(kernelContext_12->scratch_0+_S84)) ;
        if((entry_0 & 1U) == 0U)
        {
            float3 mi_1 = mi_0 + (float4(*(kernelContext_12->scratch_0+(_S84 + 1U))) ).xyz;
            fi_0 = fi_0 + _S85.xyz;
            mi_0 = mi_1;
        }
        else
        {
            float3 mi_2 = mi_0 + (float4(*(kernelContext_12->scratch_0+(_S84 + 2U))) ).xyz;
            fi_0 = fi_0 + - _S85.xyz;
            mi_0 = mi_2;
        }
        float _S86 = max(peak_0, _S85.w);
        uint _S87 = e_0 + 1U;
        peak_0 = _S86;
        e_0 = _S87;
    }
    uint _S88 = 4U * c_5;
    float3 u_0 = (float4(*(kernelContext_12->state_0+_S88)) ).xyz;
    uint _S89 = _S88 + 1U;
    float3 th_0 = (float4(*(kernelContext_12->state_0+_S89)) ).xyz;
    uint _S90 = _S88 + 2U;
    float3 v_6 = (float4(*(kernelContext_12->state_0+_S90)) ).xyz;
    uint _S91 = _S88 + 3U;
    float3 w_3 = (float4(*(kernelContext_12->state_0+_S91)) ).xyz;
    float4 _S92 = float4(_S81->center_0) ;
    float mass_0 = _S92.w;
    float3 _S93 = _S92.xyz;
    float3 _S94 = isl_4->com_0.xyz;
    float3 _S95 = rotate_0(&rg_3->rot_0, _S93 + u_0 - _S94);
    thread float3 f_load_0;
    thread float3 t_load_0;
    thread float3 rep_f_1;
    thread float3 rep_m_1;
    chunk_loads_0(c_5, c_5, &rg_3->rot_0, step_0, dt_5, contact_2, dt_5, &f_load_0, &t_load_0, &rep_f_1, &rep_m_1, kernelContext_12);
    record_chunk_load_0(c_5, f_load_0, t_load_0, kernelContext_12);
    float3 _S96 = float3(mass_0) ;
    float3 f_world_0 = f_load_0 + kernelContext_12->params_0->gravity_0.xyz * _S96;
    float3 t_world_0 = t_load_0;
    float3 f_world_1;
    float3 t_world_1;
    if(rml_0)
    {
        float3 f_world_2 = f_world_0 - (rg_3->a_0 + cross(rg_3->alpha_0, _S95) + cross(rg_3->w_1, cross(rg_3->w_1, _S95))) * _S96;
        float4 _S97 = float4(_S81->inertia0_1) ;
        float4 _S98 = float4(_S81->inertia1_1) ;
        float4 _S99 = float4(_S81->inertia2_1) ;
        float3 _S100 = world_mul_0(&rg_3->rot_0, _S97, _S98, _S99, rg_3->alpha_0);
        float3 _S101 = world_mul_0(&rg_3->rot_0, _S97, _S98, _S99, rg_3->w_1);
        float3 t_world_2 = t_world_0 - (_S100 + cross(rg_3->w_1, _S101));
        f_world_1 = f_world_2;
        t_world_1 = t_world_2;
    }
    else
    {
        f_world_1 = f_world_0;
        t_world_1 = t_world_0;
    }
    float3 _S102 = inverse_rotate_1(&rg_3->rot_0, f_world_1);
    float3 _S103 = inverse_rotate_1(&rg_3->rot_0, t_world_1);
    float3 f_ext_0;
    float3 m_ext_0;
    if(rml_0)
    {
        float3 _S104 = inverse_rotate_1(&rg_3->rot_0, rg_3->w_1);
        float4 _S105 = float4(_S81->inertia0_1) ;
        float4 _S106 = float4(_S81->inertia1_1) ;
        float4 _S107 = float4(_S81->inertia2_1) ;
        float3 i_w_0 = rows_mul_0(_S105, _S106, _S107, w_3);
        float3 m_ext_1 = _S103 - (cross(_S104, i_w_0) + cross(w_3, rows_mul_0(_S105, _S106, _S107, _S104)) + cross(w_3, i_w_0));
        f_ext_0 = _S102 - cross(_S104, v_6) * float3((2.0f * mass_0)) ;
        m_ext_0 = m_ext_1;
    }
    else
    {
        f_ext_0 = _S102;
        m_ext_0 = _S103;
    }
    float3 f_4 = f_ext_0 + rep_f_1 + fi_0;
    float3 m_1 = m_ext_0 + rep_m_1 + mi_0;
    uint support_0 = (uint4(_S81->info_1) ).x;
    float3 _S108 = float3((float4(*(kernelContext_12->state_0+_S89)) ).w, (float4(*(kernelContext_12->state_0+_S90)) ).w, (float4(*(kernelContext_12->state_0+_S91)) ).w);
    float3 reaction_0;
    float3 u_1;
    float3 th_1;
    float3 v_7;
    float3 w_4;
    if(support_0 == 1U)
    {
        reaction_0 = - f_4;
        u_1 = u_0;
        th_1 = th_0;
        v_7 = _S82;
        w_4 = _S82;
    }
    else
    {
        float4 _S109 = float4(_S81->scale_0) ;
        float3 w_5 = w_3 + rows_mul_0(float4(_S81->inv0_1) , float4(_S81->inv1_1) , float4(_S81->inv2_1) , m_1) * float3((dt_5 * _S109.z)) ;
        float3 _S110 = float3(dt_5) ;
        float3 th_2 = th_0 + w_5 * _S110;
        if(support_0 == 2U)
        {
            reaction_0 = - f_4;
            u_1 = u_0;
            th_1 = _S82;
        }
        else
        {
            float3 v_8 = v_6 + f_4 * float3((dt_5 * _S109.y)) ;
            float3 u_2 = u_0 + v_8 * _S110;
            reaction_0 = _S108;
            u_1 = u_2;
            th_1 = v_8;
        }
        float3 _S111 = th_1;
        th_1 = th_2;
        v_7 = _S111;
        w_4 = w_5;
    }
    *(kernelContext_12->state_0+_S88) = packed_float4(float4(u_1, peak_0)) ;
    *(kernelContext_12->state_0+_S89) = packed_float4(float4(th_1, reaction_0.x)) ;
    *(kernelContext_12->state_0+_S90) = packed_float4(float4(v_7, reaction_0.y)) ;
    *(kernelContext_12->state_0+_S91) = packed_float4(float4(w_4, reaction_0.z)) ;
    float3 _S112 = rotate_0(&rg_3->rot_0, _S93 + u_1 - _S94);
    float3 _S113 = rg_3->vel_0 + rg_3->vel_err_0 + cross(rg_3->w_1, _S112);
    float3 _S114 = rotate_0(&rg_3->rot_0, v_7);
    float3 v_world_0 = _S113 + _S114;
    float3 _S115 = rotate_0(&rg_3->rot_0, w_4);
    comp_add1_0(work_0, work_err_0, (dot(f_load_0, v_world_0) + dot(t_load_0, rg_3->w_1 + _S115)) * dt_5);
    return;
}

void comp_add_0(float3 thread* sum_1, float3 thread* err_1, float3 x_3)
{
    float3 t_7 = *sum_1 + x_3;
    float3 _S116 = abs(x_3);
    *err_1 = *err_1 + (select(x_3, *sum_1, (abs(*sum_1)) >= _S116) - t_7 + select(*sum_1, x_3, (abs(*sum_1)) >= _S116));
    *sum_1 = t_7;
    return;
}

float3 turn_difference_0(float3 omega_0, float dt_6, float3 v_9)
{
    float _S117 = length(omega_0);
    float angle_0 = _S117 * dt_6;
    if(angle_0 < 1.00000000317107685e-30f)
    {
        return float3(0.0f) ;
    }
    float3 a_3 = omega_0 / float3(_S117) ;
    float s_1 = sin(0.5f * angle_0);
    float3 av_0 = cross(a_3, v_9);
    return av_0 * float3(sin(angle_0))  + cross(a_3, av_0) * float3((2.0f * s_1 * s_1)) ;
}

float4 turn_minus_one_0(float3 axis_0, float angle_1)
{
    float s_2 = sin(0.25f * angle_1);
    return float4(axis_0 * float3(sin(0.5f * angle_1)) , -2.0f * s_2 * s_2);
}

Quat_0 quat_mul_0(const Quat_0 thread* a_4, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_4->w_0 * o_0->w_0 - a_4->x_0 * o_0->x_0 - a_4->y_0 * o_0->y_0 - a_4->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_4->w_0 * o_0->x_0 + a_4->x_0 * o_0->w_0 + a_4->y_0 * o_0->z_0 - a_4->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_4->w_0 * o_0->y_0 - a_4->x_0 * o_0->z_0 + a_4->y_0 * o_0->w_0 + a_4->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_4->w_0 * o_0->z_0 + a_4->x_0 * o_0->y_0 - a_4->y_0 * o_0->x_0 + a_4->z_0 * o_0->w_0;
    return r_1;
}

float4 quat_vec_0(const Quat_0 thread* q_6)
{
    return float4(q_6->x_0, q_6->y_0, q_6->z_0, q_6->w_0);
}

void comp_add4_0(float4 thread* sum_2, float4 thread* err_2, float4 x_4)
{
    float4 t_8 = *sum_2 + x_4;
    float4 _S118 = abs(x_4);
    *err_2 = *err_2 + (select(x_4, *sum_2, (abs(*sum_2)) >= _S118) - t_8 + select(*sum_2, x_4, (abs(*sum_2)) >= _S118));
    *sum_2 = t_8;
    return;
}

void quat_accumulate_0(Quat_0 thread* hi_0, float4 thread* lo_0, float4 x_5)
{
    thread Quat_0 _S119 = *hi_0;
    float4 _S120 = quat_vec_0(&_S119);
    thread float4 sum_3 = _S120;
    comp_add4_0(&sum_3, lo_0, x_5);
    float4 t_9 = sum_3 + *lo_0;
    *lo_0 = *lo_0 - (t_9 - sum_3);
    *hi_0 = quat_of_0(t_9);
    return;
}

void turn_left_0(Quat_0 thread* hi_1, float4 thread* lo_1, float3 omega_1, float dt_7)
{
    float _S121 = length(omega_1);
    float angle_2 = _S121 * dt_7;
    if(angle_2 < 1.00000000317107685e-30f)
    {
        return;
    }
    float4 d_0 = turn_minus_one_0(omega_1 / float3(_S121) , angle_2);
    thread Quat_0 dq_0;
    (&dq_0)->x_0 = d_0.x;
    (&dq_0)->y_0 = d_0.y;
    (&dq_0)->z_0 = d_0.z;
    (&dq_0)->w_0 = d_0.w;
    thread Quat_0 _S122 = dq_0;
    thread Quat_0 _S123 = *hi_1;
    Quat_0 _S124 = quat_mul_0(&_S122, &_S123);
    thread Quat_0 _S125 = _S124;
    float4 _S126 = quat_vec_0(&_S125);
    quat_accumulate_0(hi_1, lo_1, _S126);
    return;
}

void integrate_rigid_0(const Island_0 thread* isl_5, Rigid_0 thread* rg_4, float dt_8)
{
    float3 _S127 = float3(dt_8) ;
    comp_add_0(&rg_4->l_0, &rg_4->l_err_0, rg_4->torque_0 * _S127);
    float3 l_1 = rg_4->l_0 + rg_4->l_err_0;
    comp_add_0(&rg_4->vel_0, &rg_4->vel_err_0, rg_4->a_0 * _S127);
    float3 vel_1 = rg_4->vel_0 + rg_4->vel_err_0;
    float4 _S128 = isl_5->inv0_0;
    float4 _S129 = isl_5->inv1_0;
    float4 _S130 = isl_5->inv2_0;
    thread Quat_0 _S131 = rg_4->rot_0;
    float3 _S132 = world_mul_0(&_S131, isl_5->inv0_0, isl_5->inv1_0, isl_5->inv2_0, l_1);
    float3 _S133 = vel_1 * _S127;
    float3 _S134 = isl_5->com_0.xyz;
    thread Quat_0 _S135 = rg_4->rot_0;
    float3 _S136 = rotate_0(&_S135, _S134);
    comp_add_0(&rg_4->pos_0, &rg_4->pos_err_0, _S133 - turn_difference_0(_S132, dt_8, _S136));
    turn_left_0(&rg_4->rot_0, &rg_4->rot_err_0, _S132, dt_8);
    thread Quat_0 _S137 = rg_4->rot_0;
    float3 _S138 = world_mul_0(&_S137, _S128, _S129, _S130, l_1);
    rg_4->w_1 = _S138;
    return;
}

void drift_moments_0(uint c_6, float3 thread* tu_0, float3 thread* pv_0, KernelContext_0 thread* kernelContext_13)
{
    ChunkStatic_natural_0 device* _S139 = kernelContext_13->chunks_0+c_6;
    uint _S140 = 4U * c_6;
    float3 _S141 = float3(((float4(_S139->center_0) ).w * (float4(_S139->scale_0) ).x)) ;
    *tu_0 = *tu_0 + (float4(*(kernelContext_13->state_0+_S140)) ).xyz * _S141;
    *pv_0 = *pv_0 + (float4(*(kernelContext_13->state_0+(_S140 + 2U))) ).xyz * _S141;
    return;
}

void drift_angular_0(uint c_7, float3 wcom_1, float3 tr_0, float3 dv_0, float3 thread* lu_0, float3 thread* lv_0, KernelContext_0 thread* kernelContext_14)
{
    ChunkStatic_natural_0 device* _S142 = kernelContext_14->chunks_0+c_7;
    float4 _S143 = float4(_S142->center_0) ;
    float3 r_2 = _S143.xyz - wcom_1;
    uint _S144 = 4U * c_7;
    float3 _S145 = float3(_S143.w) ;
    float4 _S146 = float4(_S142->inertia0_1) ;
    float4 _S147 = float4(_S142->inertia1_1) ;
    float4 _S148 = float4(_S142->inertia2_1) ;
    float3 _S149 = float3((float4(_S142->scale_0) ).x) ;
    *lu_0 = *lu_0 + (cross(r_2, (float4(*(kernelContext_14->state_0+_S144)) ).xyz - tr_0) * _S145 + rows_mul_0(_S146, _S147, _S148, (float4(*(kernelContext_14->state_0+(_S144 + 1U))) ).xyz)) * _S149;
    *lv_0 = *lv_0 + (cross(r_2, (float4(*(kernelContext_14->state_0+(_S144 + 2U))) ).xyz - dv_0) * _S145 + rows_mul_0(_S146, _S147, _S148, (float4(*(kernelContext_14->state_0+(_S144 + 3U))) ).xyz)) * _S149;
    return;
}

void drift_apply_0(uint c_8, float3 wcom_2, float3 tr_1, float3 phi_0, float3 dv_1, float3 dw_0, KernelContext_0 thread* kernelContext_15)
{
    float3 r_3 = (float4((kernelContext_15->chunks_0+c_8)->center_0) ).xyz - wcom_2;
    uint _S150 = 4U * c_8;
    *(kernelContext_15->state_0+_S150) = packed_float4(float4((float4(*(kernelContext_15->state_0+_S150)) ).xyz - (tr_1 + cross(phi_0, r_3)), (float4(*(kernelContext_15->state_0+_S150)) ).w)) ;
    uint _S151 = _S150 + 1U;
    *(kernelContext_15->state_0+_S151) = packed_float4(float4((float4(*(kernelContext_15->state_0+_S151)) ).xyz - phi_0, (float4(*(kernelContext_15->state_0+_S151)) ).w)) ;
    uint _S152 = _S150 + 2U;
    *(kernelContext_15->state_0+_S152) = packed_float4(float4((float4(*(kernelContext_15->state_0+_S152)) ).xyz - (dv_1 + cross(dw_0, r_3)), (float4(*(kernelContext_15->state_0+_S152)) ).w)) ;
    uint _S153 = _S150 + 3U;
    *(kernelContext_15->state_0+_S153) = packed_float4(float4((float4(*(kernelContext_15->state_0+_S153)) ).xyz - dw_0, (float4(*(kernelContext_15->state_0+_S153)) ).w)) ;
    return;
}

void turn_right_0(Quat_0 thread* hi_2, float4 thread* lo_2, float3 phi_1)
{
    float angle_3 = length(phi_1);
    if(angle_3 < 1.00000000317107685e-30f)
    {
        return;
    }
    float4 d_1 = turn_minus_one_0(phi_1 / float3(angle_3) , angle_3);
    thread Quat_0 dq_1;
    (&dq_1)->x_0 = d_1.x;
    (&dq_1)->y_0 = d_1.y;
    (&dq_1)->z_0 = d_1.z;
    (&dq_1)->w_0 = d_1.w;
    thread Quat_0 _S154 = *hi_2;
    thread Quat_0 _S155 = dq_1;
    Quat_0 _S156 = quat_mul_0(&_S154, &_S155);
    thread Quat_0 _S157 = _S156;
    float4 _S158 = quat_vec_0(&_S157);
    quat_accumulate_0(hi_2, lo_2, _S158);
    return;
}

void drift_rigid_0(const Island_0 thread* isl_6, Rigid_0 thread* rg_5, float3 tr_2, float3 phi_2, float3 dv_2, float3 dw_1)
{
    float3 wcom_3 = isl_6->wcom_0.xyz;
    Quat_0 rot_2 = rg_5->rot_0;
    float3 _S159 = tr_2 - turn_difference_0(phi_2, 1.0f, wcom_3);
    thread Quat_0 _S160 = rg_5->rot_0;
    float3 _S161 = rotate_0(&_S160, _S159);
    comp_add_0(&rg_5->pos_0, &rg_5->pos_err_0, _S161);
    thread Quat_0 _S162 = rg_5->rot_0;
    float3 _S163 = inverse_rotate_1(&_S162, rg_5->w_1);
    float4 _S164 = isl_6->inertia0_0;
    float4 _S165 = isl_6->inertia1_0;
    float4 _S166 = isl_6->inertia2_0;
    float3 _S167 = cross(phi_2, rows_mul_0(isl_6->inertia0_0, isl_6->inertia1_0, isl_6->inertia2_0, _S163)) - rows_mul_0(isl_6->inertia0_0, isl_6->inertia1_0, isl_6->inertia2_0, cross(phi_2, _S163));
    thread Quat_0 _S168 = rg_5->rot_0;
    float3 _S169 = rotate_0(&_S168, _S167);
    turn_right_0(&rg_5->rot_0, &rg_5->rot_err_0, phi_2);
    float3 _S170 = dv_2 + cross(dw_1, isl_6->com_0.xyz - wcom_3);
    thread Quat_0 _S171 = rot_2;
    float3 _S172 = rotate_0(&_S171, _S170);
    comp_add_0(&rg_5->vel_0, &rg_5->vel_err_0, _S172);
    thread Quat_0 _S173 = rot_2;
    float3 _S174 = rotate_0(&_S173, dw_1);
    rg_5->w_1 = rg_5->w_1 + _S174;
    thread Quat_0 _S175 = rg_5->rot_0;
    float3 _S176 = world_mul_0(&_S175, _S164, _S165, _S166, _S174);
    comp_add_0(&rg_5->l_0, &rg_5->l_err_0, _S169 + _S176);
    return;
}

void write_probe_0(uint slot_0, uint k_4, float value_0, KernelContext_0 thread* kernelContext_16)
{
    uint at_0 = kernelContext_16->params_0->probe_base_0 * 4U + slot_0 * kernelContext_16->params_0->probe_stride_0 + k_4;
    thread float4 v_10 = float4(*(kernelContext_16->scratch_0+at_0 / 4U)) ;
    v_10[at_0 % 4U] = value_0;
    *(kernelContext_16->scratch_0+at_0 / 4U) = packed_float4(v_10) ;
    return;
}

void record_probes_0(const Island_0 thread* isl_7, const Rigid_0 thread* rg_6, uint k_5, KernelContext_0 thread* kernelContext_17)
{
    uint4 _S177 = isl_7->probes_0;
    uint at_1 = isl_7->probes_0.x;
    for(;;)
    {
        if(at_1 < (_S177.y))
        {
        }
        else
        {
            break;
        }
        uint4 info_3 = (as_type<uint4>((float4(*(kernelContext_17->loads_0+at_1)) )));
        float4 _S178 = float4(*(kernelContext_17->loads_0+(at_1 + 1U))) ;
        float4 _S179 = float4(*(kernelContext_17->loads_0+(at_1 + 2U))) ;
        float4 _S180 = float4(*(kernelContext_17->loads_0+(at_1 + 3U))) ;
        uint kind_1 = info_3.x;
        uint i_1 = info_3.y;
        float value_1;
        if(kind_1 == 0U)
        {
            float3 small_0 = (float4(*(kernelContext_17->state_0+4U * i_1)) ).xyz + float3(_S178.w, _S179.w, _S180.w);
            float3 _S181 = rg_6->pos_0 - _S179.xyz;
            float3 _S182 = rotate_0(&rg_6->rot_0, (float4((kernelContext_17->chunks_0+i_1)->center_0) ).xyz);
            float3 _S183 = _S181 + _S182;
            float3 _S184 = rg_6->pos_err_0 - _S180.xyz;
            float3 _S185 = rotate_0(&rg_6->rot_0, small_0);
            value_1 = dot(_S183 + (_S184 + _S185), _S178.xyz);
        }
        else
        {
            if(kind_1 == 1U)
            {
                uint _S186 = 4U * i_1;
                float3 _S187 = rotate_0(&rg_6->rot_0, (float4((kernelContext_17->chunks_0+i_1)->center_0) ).xyz + (float4(*(kernelContext_17->state_0+_S186)) ).xyz - isl_7->com_0.xyz);
                float3 _S188 = rg_6->vel_0 + rg_6->vel_err_0 + cross(rg_6->w_1, _S187);
                float3 _S189 = rotate_0(&rg_6->rot_0, (float4(*(kernelContext_17->state_0+(_S186 + 2U))) ).xyz);
                value_1 = dot(_S188 + _S189, _S178.xyz);
            }
            else
            {
                if(kind_1 == 2U)
                {
                    uint _S190 = 3U * i_1;
                    float3 f_5 = (float4(*(kernelContext_17->scratch_0+_S190)) ).xyz;
                    bool _S191 = (info_3.z) == 0U;
                    float3 mc_0;
                    if(_S191)
                    {
                        mc_0 = (float4(*(kernelContext_17->scratch_0+(_S190 + 1U))) ).xyz;
                    }
                    else
                    {
                        mc_0 = (float4(*(kernelContext_17->scratch_0+(_S190 + 2U))) ).xyz;
                    }
                    float3 fc_1;
                    if(_S191)
                    {
                        fc_1 = f_5;
                    }
                    else
                    {
                        fc_1 = - f_5;
                    }
                    value_1 = dot(fc_1, _S178.xyz) + dot(mc_0, _S179.xyz);
                }
                else
                {
                    uint _S192 = 4U * i_1;
                    float3 _S193 = rotate_0(&rg_6->rot_0, float3((float4(*(kernelContext_17->state_0+(_S192 + 1U))) ).w, (float4(*(kernelContext_17->state_0+(_S192 + 2U))) ).w, (float4(*(kernelContext_17->state_0+(_S192 + 3U))) ).w));
                    value_1 = dot(_S193, _S178.xyz);
                }
            }
        }
        write_probe_0(info_3.w, k_5, value_1, kernelContext_17);
        at_1 = at_1 + 4U;
    }
    return;
}

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S194;
    if((st_0->damage_0) < 1.0f)
    {
        _S194 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S194 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S194 = false;
        }
    }
    return _S194;
}

bool island_connected_narrow_0(uint tid_3, const Island_0 thread* isl_8, KernelContext_0 thread* kernelContext_18)
{
    uint _S195 = isl_8->range_0.x;
    uint n_0 = isl_8->range_0.y - _S195;
    bool converged_0;
    if(n_0 > 512U)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (kernelContext_18->params_0->device_connect_0) == 0U;
    }
    if(converged_0)
    {
        return false;
    }
    uint c_9 = tid_3;
    for(;;)
    {
        if(c_9 < n_0)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_18->g_label_narrow_0)[c_9] = c_9;
        c_9 = c_9 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    converged_0 = false;
    uint pass_0 = 0U;
    for(;;)
    {
        if(pass_0 <= n_0)
        {
        }
        else
        {
            break;
        }
        if(tid_3 == 0U)
        {
            *kernelContext_18->g_changed_narrow_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_9 = tid_3;
        for(;;)
        {
            if(c_9 < n_0)
            {
            }
            else
            {
                break;
            }
            uint gc_0 = _S195 + c_9;
            uint _S196 = kernelContext_18->index_0[gc_0];
            uint l_2 = (*kernelContext_18->g_label_narrow_0)[c_9];
            uint e_1 = _S196;
            for(;;)
            {
                if(e_1 < (kernelContext_18->index_0)[gc_0 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint entry_1 = kernelContext_18->index_0[e_1];
                uint bond_0 = entry_1 >> 1U;
                bool _S197 = ((float4((kernelContext_18->bonds_0+bond_0)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S198 = (kernelContext_18->bond_dyn_0+bond_0)->js_0;
                bool _S199 = connected_0(&_S198, _S197);
                if(!_S199)
                {
                    e_1 = e_1 + 1U;
                    continue;
                }
                uint other_0;
                if((entry_1 & 1U) == 0U)
                {
                    other_0 = (uint4((kernelContext_18->bonds_0+bond_0)->law_0.ids_0) ).z;
                }
                else
                {
                    other_0 = (uint4((kernelContext_18->bonds_0+bond_0)->law_0.ids_0) ).y;
                }
                l_2 = min(l_2, (*kernelContext_18->g_label_narrow_0)[other_0 - _S195]);
                e_1 = e_1 + 1U;
            }
            uint _S200 = min(l_2, (*kernelContext_18->g_label_narrow_0)[l_2]);
            if(_S200 < (*kernelContext_18->g_label_narrow_0)[c_9])
            {
                (*kernelContext_18->g_label_narrow_0)[c_9] = _S200;
                *kernelContext_18->g_changed_narrow_0 = 1U;
            }
            c_9 = c_9 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool converged_1 = (*kernelContext_18->g_changed_narrow_0) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(converged_1)
        {
            converged_0 = converged_1;
            break;
        }
        uint _S201 = pass_0 + 1U;
        converged_0 = converged_1;
        pass_0 = _S201;
    }
    if(!converged_0)
    {
        return false;
    }
    if(tid_3 == 0U)
    {
        *kernelContext_18->g_changed_narrow_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c_9 = tid_3;
    for(;;)
    {
        if(c_9 < n_0)
        {
        }
        else
        {
            break;
        }
        if(((*kernelContext_18->g_label_narrow_0)[c_9]) != 0U)
        {
            *kernelContext_18->g_changed_narrow_0 = 1U;
        }
        c_9 = c_9 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_18->g_changed_narrow_0) == 0U;
}

void contact_split_at_0(uint at_2, KernelContext_0 thread* kernelContext_19)
{
    uint previous_0 = (uint4((kernelContext_19->islands_0+kernelContext_19->params_0->halt_index_0)->info_0) ).y;
    uint _S202;
    if(previous_0 == 0U)
    {
        _S202 = at_2;
    }
    else
    {
        _S202 = min(previous_0, at_2);
    }
    (kernelContext_19->islands_0+kernelContext_19->params_0->halt_index_0)->info_0[int(1)] = _S202;
    return;
}

uint seen_bits_0(KernelContext_0 thread* kernelContext_20)
{
    uint _S203;
    if(((*kernelContext_20->g_seen_0)[0U]) != 0U)
    {
        _S203 = 32768U;
    }
    else
    {
        _S203 = 0U;
    }
    uint _S204;
    if(((*kernelContext_20->g_seen_0)[1U]) != 0U)
    {
        _S204 = 65536U;
    }
    else
    {
        _S204 = 0U;
    }
    uint _S205 = _S203 | _S204;
    if(((*kernelContext_20->g_seen_0)[2U]) != 0U)
    {
        _S203 = 131072U;
    }
    else
    {
        _S203 = 0U;
    }
    uint _S206 = _S205 | _S203;
    if(((*kernelContext_20->g_seen_0)[3U]) != 0U)
    {
        _S203 = 32U;
    }
    else
    {
        _S203 = 0U;
    }
    uint _S207 = _S206 | _S203;
    if(((*kernelContext_20->g_seen_0)[4U]) != 0U)
    {
        _S203 = 64U;
    }
    else
    {
        _S203 = 0U;
    }
    return _S207 | _S203;
}

void island_frame_at_0(uint gi_0, uint tid_4, KernelContext_0 thread* kernelContext_21)
{
    bool woke_0;
    Island_natural_0 device* _S208 = kernelContext_21->islands_0+gi_0;
    uint4 _S209 = uint4((*_S208).info_0) ;
    float4 _S210 = float4((*_S208).com_0) ;
    float4 _S211 = float4((*_S208).inertia0_0) ;
    float4 _S212 = float4((*_S208).inertia1_0) ;
    float4 _S213 = float4((*_S208).inertia2_0) ;
    float4 _S214 = float4((*_S208).inv0_0) ;
    float4 _S215 = float4((*_S208).inv1_0) ;
    float4 _S216 = float4((*_S208).inv2_0) ;
    float4 _S217 = float4((*_S208).wcom_0) ;
    float4 _S218 = float4((*_S208).winv0_0) ;
    float4 _S219 = float4((*_S208).winv1_0) ;
    float4 _S220 = float4((*_S208).winv2_0) ;
    float4 _S221 = float4((*_S208).rotation_0) ;
    float4 _S222 = float4((*_S208).position_0) ;
    float4 _S223 = float4((*_S208).position_err_0) ;
    float4 _S224 = float4((*_S208).velocity_0) ;
    float4 _S225 = float4((*_S208).velocity_err_0) ;
    float4 _S226 = float4((*_S208).angular_velocity_0) ;
    uint4 _S227 = uint4((*_S208).probes_0) ;
    float4 _S228 = float4((*_S208).energy_0) ;
    float4 _S229 = float4((*_S208).rotation_err_0) ;
    float4 _S230 = float4((*_S208).momentum_0) ;
    float4 _S231 = float4((*_S208).momentum_err_0) ;
    thread Island_0 isl_9;
    (&isl_9)->range_0 = uint4((*_S208).range_0) ;
    (&isl_9)->info_0 = _S209;
    (&isl_9)->com_0 = _S210;
    (&isl_9)->inertia0_0 = _S211;
    (&isl_9)->inertia1_0 = _S212;
    (&isl_9)->inertia2_0 = _S213;
    (&isl_9)->inv0_0 = _S214;
    (&isl_9)->inv1_0 = _S215;
    (&isl_9)->inv2_0 = _S216;
    (&isl_9)->wcom_0 = _S217;
    (&isl_9)->winv0_0 = _S218;
    (&isl_9)->winv1_0 = _S219;
    (&isl_9)->winv2_0 = _S220;
    (&isl_9)->rotation_0 = _S221;
    (&isl_9)->position_0 = _S222;
    (&isl_9)->position_err_0 = _S223;
    (&isl_9)->velocity_0 = _S224;
    (&isl_9)->velocity_err_0 = _S225;
    (&isl_9)->angular_velocity_0 = _S226;
    (&isl_9)->done_0 = (*_S208).done_0;
    (&isl_9)->coverage_0 = (*_S208).coverage_0;
    (&isl_9)->wide_groups_0 = (*_S208).wide_groups_0;
    (&isl_9)->done_pad_0 = (*_S208).done_pad_0;
    (&isl_9)->probes_0 = _S227;
    (&isl_9)->energy_0 = _S228;
    (&isl_9)->rotation_err_0 = _S229;
    (&isl_9)->momentum_0 = _S230;
    (&isl_9)->momentum_err_0 = _S231;
    bool driven_0 = (((&isl_9)->info_0.x) & 2U) != 0U;
    bool _S232 = !((((&isl_9)->info_0.x) & 1U) != 0U);
    bool _S233;
    if(_S232)
    {
        _S233 = !driven_0;
    }
    else
    {
        _S233 = false;
    }
    bool contact_island_0 = (((&isl_9)->info_0.x) & 4U) != 0U;
    bool _S234 = (((&isl_9)->info_0.x) & 16U) != 0U;
    seen_clear_0(tid_4, kernelContext_21);
    bool _S235 = tid_4 == 0U;
    bool settled_0;
    uint cov_0;
    if(_S235)
    {
        if(contact_island_0 != ((kernelContext_21->params_0->contact_mode_0) == 1U))
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
            thread Island_0 _S236 = isl_9;
            bool _S237 = contact_stopped_0(&_S236, kernelContext_21);
            settled_0 = _S237;
        }
        else
        {
            settled_0 = false;
        }
        if(settled_0)
        {
            cov_0 = 0U;
        }
        *kernelContext_21->g_run_0 = cov_0;
        *kernelContext_21->g_halt_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((((&isl_9)->info_0.z) & 1U) != 0U)
    {
        settled_0 = true;
    }
    else
    {
        settled_0 = (*kernelContext_21->g_run_0) == 0U;
    }
    if(settled_0)
    {
        cov_0 = 0U;
    }
    else
    {
        cov_0 = min((&isl_9)->info_0.y, kernelContext_21->params_0->max_steps_0);
    }
    float _S238 = kernelContext_21->params_0->dt_0;
    bool _S239 = (kernelContext_21->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S240 = isl_9;
    Rigid_0 _S241 = rigid_of_0(&_S240);
    thread Rigid_0 rg_7 = _S241;
    thread float work_1 = 0.0f;
    thread float work_err_1 = 0.0f;
    settled_0 = _S234;
    uint done_1 = 0U;
    bool woke_1 = false;
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
        uint k_6 = (&isl_9)->info_0.w + done_1 + 1U - 1U - kernelContext_21->params_0->step_start_0;
        bool quiet_0 = (((&isl_9)->info_0.x) & 32U) != 0U;
        bool _S242;
        if(settled_0)
        {
            _S242 = !quiet_0;
        }
        else
        {
            _S242 = false;
        }
        bool _S243;
        bool explicit_step_0;
        uint c_10;
        if(_S242)
        {
            float3 _S244 = float3(0.0f) ;
            thread float3 norm_0 = _S244;
            thread float3 unused0_0 = _S244;
            c_10 = (&isl_9)->range_0.x + tid_4;
            for(;;)
            {
                if(c_10 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S245 = (&rg_7)->rot_0;
                float _S246 = settled_chunk_load_0(c_10, &_S245, k_6, _S238, contact_island_0, kernelContext_21);
                norm_0.x = norm_0.x + _S246;
                c_10 = c_10 + 256U;
            }
            group_sum3_0(tid_4, &norm_0, &unused0_0, kernelContext_21);
            if((kernelContext_21->params_0->solve_mode_0) == 1U)
            {
                _S243 = (abs(norm_0.x - (&isl_9)->energy_0.z)) > ((&isl_9)->energy_0.w);
            }
            else
            {
                _S243 = false;
            }
            if(_S243)
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
        if(_S232)
        {
            _S243 = !quiet_0;
        }
        else
        {
            _S243 = false;
        }
        if(_S243)
        {
            float3 _S247 = float3(0.0f) ;
            thread float3 f_6 = _S247;
            thread float3 t_10 = _S247;
            c_10 = (&isl_9)->range_0.x + tid_4;
            for(;;)
            {
                if(c_10 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S248 = isl_9;
                thread Rigid_0 _S249 = rg_7;
                net_load_0(c_10, &_S248, &_S249, k_6, _S238, contact_island_0, &f_6, &t_10, kernelContext_21);
                c_10 = c_10 + 256U;
            }
            group_sum3_0(tid_4, &f_6, &t_10, kernelContext_21);
            thread Island_0 _S250 = isl_9;
            rigid_acceleration_0(&_S250, &rg_7, f_6, t_10);
        }
        if(!settled_0)
        {
            explicit_step_0 = !quiet_0;
        }
        else
        {
            explicit_step_0 = false;
        }
        bool _S251;
        if(explicit_step_0)
        {
            if(_S235)
            {
                _S251 = (((uint4((kernelContext_21->islands_0+gi_0)->info_0) ).z) & 2U) != 0U;
            }
            else
            {
                _S251 = false;
            }
            if(_S251)
            {
                *kernelContext_21->g_halt_0 = 1U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_10 = (&isl_9)->range_0.x + tid_4;
            for(;;)
            {
                if(c_10 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S252 = isl_9;
                thread Rigid_0 _S253 = rg_7;
                chunk_update_0(c_10, &_S252, &_S253, _S238, _S239, k_6, contact_island_0, &work_1, &work_err_1, kernelContext_21);
                c_10 = c_10 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        if(_S233)
        {
            _S251 = !quiet_0;
        }
        else
        {
            _S251 = false;
        }
        if(_S251)
        {
            thread Island_0 _S254 = isl_9;
            integrate_rigid_0(&_S254, &rg_7, _S238);
        }
        bool _S255;
        if(_S232)
        {
            _S255 = explicit_step_0;
        }
        else
        {
            _S255 = false;
        }
        if(_S255)
        {
            float3 _S256 = (&isl_9)->wcom_0.xyz;
            float3 _S257 = float3(0.0f) ;
            thread float3 tu_1 = _S257;
            thread float3 pv_1 = _S257;
            c_10 = (&isl_9)->range_0.x + tid_4;
            for(;;)
            {
                if(c_10 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_10, &tu_1, &pv_1, kernelContext_21);
                c_10 = c_10 + 256U;
            }
            group_sum3_0(tid_4, &tu_1, &pv_1, kernelContext_21);
            float3 tr_3 = tu_1 / float3((&isl_9)->wcom_0.w) ;
            float3 dv_3 = pv_1 / float3((&isl_9)->wcom_0.w) ;
            thread float3 lu_1 = _S257;
            thread float3 lv_1 = _S257;
            uint c_11 = (&isl_9)->range_0.x + tid_4;
            for(;;)
            {
                if(c_11 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_11, _S256, tr_3, dv_3, &lu_1, &lv_1, kernelContext_21);
                c_11 = c_11 + 256U;
            }
            group_sum3_0(tid_4, &lu_1, &lv_1, kernelContext_21);
            float3 phi_3 = rows_mul_0((&isl_9)->winv0_0, (&isl_9)->winv1_0, (&isl_9)->winv2_0, lu_1);
            float3 dw_2 = rows_mul_0((&isl_9)->winv0_0, (&isl_9)->winv1_0, (&isl_9)->winv2_0, lv_1);
            uint c_12 = (&isl_9)->range_0.x + tid_4;
            for(;;)
            {
                if(c_12 < ((&isl_9)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_12, _S256, tr_3, phi_3, dv_3, dw_2, kernelContext_21);
                c_12 = c_12 + 256U;
            }
            if(!driven_0)
            {
                thread Island_0 _S258 = isl_9;
                drift_rigid_0(&_S258, &rg_7, tr_3, phi_3, dv_3, dw_2);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S259;
        if(_S235)
        {
            _S259 = ((&isl_9)->probes_0.y) > ((&isl_9)->probes_0.x);
        }
        else
        {
            _S259 = false;
        }
        if(_S259)
        {
            thread Island_0 _S260 = isl_9;
            thread Rigid_0 _S261 = rg_7;
            record_probes_0(&_S260, &_S261, k_6, kernelContext_21);
        }
        uint _S262 = done_1 + 1U;
        bool _S263;
        if((*kernelContext_21->g_halt_0) != 0U)
        {
            thread Island_0 _S264 = isl_9;
            bool _S265 = island_connected_narrow_0(tid_4, &_S264, kernelContext_21);
            _S263 = _S265;
        }
        else
        {
            _S263 = false;
        }
        bool resumed_1;
        if(_S263)
        {
            if(_S235)
            {
                *kernelContext_21->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_1 = true;
        }
        else
        {
            resumed_1 = resumed_0;
        }
        if((*kernelContext_21->g_halt_0) != 0U)
        {
            done_1 = _S262;
            resumed_0 = resumed_1;
            break;
        }
        done_1 = _S262;
        woke_1 = woke_0;
        resumed_0 = resumed_1;
    }
    thread float3 wsum_0 = float3(work_1, work_err_1, 0.0f);
    thread float3 unused_0 = float3(0.0f) ;
    group_sum3_0(tid_4, &wsum_0, &unused_0, kernelContext_21);
    if(_S235)
    {
        thread Quat_0 _S266 = (&rg_7)->rot_0;
        float4 _S267 = quat_vec_0(&_S266);
        (&isl_9)->rotation_0 = _S267;
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
        if((*kernelContext_21->g_halt_0) != 0U)
        {
            _S233 = contact_island_0;
        }
        else
        {
            _S233 = false;
        }
        if(_S233)
        {
            contact_split_at_0((&isl_9)->info_0.w + done_1, kernelContext_21);
        }
        float _S268 = wsum_0.x;
        thread float _S269 = (&isl_9)->energy_0.x;
        thread float _S270 = (&isl_9)->energy_0.y;
        comp_add1_0(&_S269, &_S270, _S268);
        (&isl_9)->energy_0.x = _S269;
        (&isl_9)->energy_0.y = _S270 + wsum_0.y;
        (&isl_9)->info_0.w = (&isl_9)->info_0.w + done_1;
        if((((&isl_9)->info_0.x) & 8U) == 0U)
        {
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_21->g_halt_0) != 0U)
        {
            (&isl_9)->info_0.z = ((&isl_9)->info_0.z) | 1U;
        }
        uint _S271 = seen_bits_0(kernelContext_21);
        if((*kernelContext_21->g_halt_0) != 0U)
        {
            cov_0 = 128U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint _S272 = _S271 | cov_0;
        if(woke_0)
        {
            cov_0 = 2048U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint _S273 = _S272 | cov_0;
        if(resumed_0)
        {
            cov_0 = 2097280U;
        }
        else
        {
            cov_0 = 0U;
        }
        uint cov_1 = _S273 | cov_0;
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
                _S233 = !quiet_1;
            }
            else
            {
                _S233 = false;
            }
            if(_S233)
            {
                cov_0 = 512U;
            }
            else
            {
                cov_0 = 0U;
            }
            uint cov_3 = cov_2 | cov_0;
            if((kernelContext_21->params_0->contact_mode_0) == 1U)
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
        Island_natural_0 device* _S274 = kernelContext_21->islands_0+gi_0;
        _S274->range_0 = packed_uint4(isl_9.range_0) ;
        _S274->info_0 = packed_uint4(isl_9.info_0) ;
        _S274->com_0 = packed_float4(isl_9.com_0) ;
        _S274->inertia0_0 = packed_float4(isl_9.inertia0_0) ;
        _S274->inertia1_0 = packed_float4(isl_9.inertia1_0) ;
        _S274->inertia2_0 = packed_float4(isl_9.inertia2_0) ;
        _S274->inv0_0 = packed_float4(isl_9.inv0_0) ;
        _S274->inv1_0 = packed_float4(isl_9.inv1_0) ;
        _S274->inv2_0 = packed_float4(isl_9.inv2_0) ;
        _S274->wcom_0 = packed_float4(isl_9.wcom_0) ;
        _S274->winv0_0 = packed_float4(isl_9.winv0_0) ;
        _S274->winv1_0 = packed_float4(isl_9.winv1_0) ;
        _S274->winv2_0 = packed_float4(isl_9.winv2_0) ;
        _S274->rotation_0 = packed_float4(isl_9.rotation_0) ;
        _S274->position_0 = packed_float4(isl_9.position_0) ;
        _S274->position_err_0 = packed_float4(isl_9.position_err_0) ;
        _S274->velocity_0 = packed_float4(isl_9.velocity_0) ;
        _S274->velocity_err_0 = packed_float4(isl_9.velocity_err_0) ;
        _S274->angular_velocity_0 = packed_float4(isl_9.angular_velocity_0) ;
        _S274->done_0 = isl_9.done_0;
        _S274->coverage_0 = isl_9.coverage_0;
        _S274->wide_groups_0 = isl_9.wide_groups_0;
        _S274->done_pad_0 = isl_9.done_pad_0;
        _S274->probes_0 = packed_uint4(isl_9.probes_0) ;
        _S274->energy_0 = packed_float4(isl_9.energy_0) ;
        _S274->rotation_err_0 = packed_float4(isl_9.rotation_err_0) ;
        _S274->momentum_0 = packed_float4(isl_9.momentum_0) ;
        _S274->momentum_err_0 = packed_float4(isl_9.momentum_err_0) ;
    }
    return;
}

float3 to_local_0(uint _S275, float3 _S276, KernelContext_0 thread* kernelContext_22)
{
    BondStatic_natural_0 device* _S277 = kernelContext_22->bonds_0+_S275;
    return float3(dot(_S276, (float4(_S277->t1_0) ).xyz), dot(_S276, (float4(_S277->t2_0) ).xyz), dot(_S276, (float4(_S277->normal_0) ).xyz));
}

bool is_damaged_0(const JointState_0 thread* st_1)
{
    bool _S278;
    if((st_1->damage_0) > 0.0f)
    {
        _S278 = true;
    }
    else
    {
        _S278 = (st_1->crush_1) > 0.0f;
    }
    return _S278;
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
    float dissipated_1;
    float overshoot_0;
    float stored_0;
    bool disconnected_0;
    float fatigue_before_0;
    Measures_0 measures_0;
};

Measures_0 stress_measures_0(const JointBond_natural_0 thread* b_2, float3 q_lin_0, float3 q_ang_0)
{
    float4 _S279 = float4(b_2->geom0_0) ;
    float area_0 = _S279.x;
    float _S280 = q_lin_0.z;
    float axial_0 = (metal::fast::divide((_S280), (area_0)));
    float4 _S281 = float4(b_2->geom1_0) ;
    float _S282 = (metal::fast::divide((abs(q_ang_0.x)), (_S281.x)));
    float _S283 = (metal::fast::divide((abs(q_ang_0.y)), (_S281.y)));
    float bending_0 = _S282 + _S283;
    float _S284 = q_lin_0.x;
    float _S285 = q_lin_0.y;
    float _S286 = (metal::fast::sqrt((_S284 * _S284 + _S285 * _S285)));
    float _S287 = (metal::fast::divide((_S286), (area_0)));
    float _S288 = (metal::fast::divide((abs(q_ang_0.z)), (_S279.w)));
    float shear_1 = _S287 + _S288;
    thread Measures_0 m_2;
    (&m_2)->tension_0 = axial_0 + bending_0;
    (&m_2)->shear_0 = shear_1;
    float _S289 = - axial_0;
    (&m_2)->normal_compression_0 = max(_S289, 0.0f);
    (&m_2)->compression_0 = _S289 + bending_0;
    (&m_2)->compressive_force_0 = max(- _S280, 0.0f);
    return m_2;
}

float expm1_accurate_0(float x_6)
{
    if((abs(x_6)) < 0.00100000004749745f)
    {
        return x_6 * (1.0f + x_6 * (0.5f + x_6 * 0.1666666716337204f));
    }
    return exp(x_6) - 1.0f;
}

float dif_factor_0(const JointMaterial_0 constant* mat_1, float strain_rate_1)
{
    float r_4 = abs(strain_rate_1);
    float4 _S290 = mat_1->dif_0;
    float ref_0 = mat_1->dif_0.x;
    if(r_4 <= ref_0)
    {
        return 1.0f;
    }
    float _S291 = _S290.z;
    float f_7;
    if(r_4 <= _S291)
    {
        float _S292 = (metal::fast::divide((r_4), (ref_0)));
        float _S293 = (metal::fast::pow((_S292), (_S290.y)));
        f_7 = _S293;
    }
    else
    {
        float _S294 = (metal::fast::divide((_S291), (ref_0)));
        float _S295 = (metal::fast::pow((_S294), (_S290.y)));
        float _S296 = (metal::fast::divide((r_4), (_S291)));
        float _S297 = (metal::fast::pow((_S296), (_S290.w)));
        f_7 = _S295 * _S297;
    }
    return clamp(f_7, 1.0f, mat_1->misc_0.x);
}

float fatigue_factor_0(const JointMaterial_0 constant* mat_2, float life_1)
{
    if(((mat_2->kind_flags_0.y) & 64U) == 0U)
    {
        return 1.0f;
    }
    float _S298 = clamp(life_1, 0.0f, 1.0f);
    float _S299 = (metal::fast::divide((1.0f), (mat_2->misc_0.y - 2.0f)));
    float _S300 = (metal::fast::pow((_S298), (_S299)));
    return _S300;
}

float infinity_0()
{
    return (as_type<float>((2139095040U)));
}

float4 failure_indices_0(const JointMaterial_0 constant* mat_3, const JointBond_natural_0 thread* b_3, const Measures_0 thread* m_3, float multiplier_0)
{
    float fc_2 = mat_3->strength_0.y * multiplier_0;
    float _S301 = min(mat_3->strength_0.z * multiplier_0 + mat_3->strength_0.w * m_3->normal_compression_0, mat_3->energy_1.x * multiplier_0);
    thread float4 idx_0;
    float _S302 = (metal::fast::divide((m_3->tension_0), (mat_3->strength_0.x * multiplier_0)));
    idx_0.x = max(_S302, 0.0f);
    float _S303;
    if(_S301 > 0.0f)
    {
        float _S304 = (metal::fast::divide((m_3->shear_0), (_S301)));
        _S303 = _S304;
    }
    else
    {
        _S303 = infinity_0();
    }
    idx_0.y = _S303;
    float _S305 = (metal::fast::divide((m_3->compression_0), (fc_2)));
    idx_0.z = max(_S305, 0.0f);
    float _S306 = (float4(b_3->stiff1_0) ).y;
    if(_S306 > 0.0f)
    {
        float _S307 = (metal::fast::divide((m_3->compressive_force_0), (_S306)));
        _S303 = _S307;
    }
    else
    {
        _S303 = 0.0f;
    }
    idx_0.w = _S303;
    return idx_0;
}

float sq_0(float x_7)
{
    return x_7 * x_7;
}

float damage_law_0(uint kind_2, float kappa_1, float r_5)
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_2 == 0U)
    {
        if(r_5 <= 1.0f)
        {
            return 1.0f;
        }
        float _S308 = (metal::fast::divide((r_5 * (kappa_1 - 1.0f)), (kappa_1 * (r_5 - 1.0f))));
        return min(_S308, 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_5 + 1.0f)))
    {
        return 1.0f;
    }
    float _S309 = (metal::fast::divide((1.0f), (kappa_1)));
    return 1.0f - _S309;
}

__attribute__((noinline))
float2 damage_increment_0(uint kind_3, float kappa_old_0, float lambda_0, float r_6, float d_old_0, float psi_0)
{
    float _S310 = damage_law_0(kind_3, lambda_0, r_6);
    float _S311 = max(_S310, d_old_0);
    bool _S312;
    if(_S311 <= d_old_0)
    {
        _S312 = true;
    }
    else
    {
        _S312 = d_old_0 >= 1.0f;
    }
    if(_S312)
    {
        return float2(d_old_0, 0.0f);
    }
    float u0_0 = (metal::fast::divide((psi_0), (lambda_0 * lambda_0)));
    float _S313 = max(kappa_old_0, 1.0f);
    if(kind_3 == 0U)
    {
        if(r_6 > 1.0f)
        {
            float _S314 = (metal::fast::divide((u0_0 * r_6), (r_6 - 1.0f)));
            return float2(_S311, _S314 * max(min(lambda_0, r_6) - min(_S313, r_6), 0.0f));
        }
        return float2(_S311, (1.0f - d_old_0) * psi_0);
    }
    float ku_0 = 0.5f * (r_6 + 1.0f);
    float plateau_0 = u0_0 * max(min(lambda_0, ku_0) - min(_S313, ku_0), 0.0f);
    float snap_0;
    if(_S311 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return float2(_S311, plateau_0 + snap_0);
}

struct Contact_0
{
    float3 q_lin_1;
    float3 q_ang_1;
    float energy_2;
    float diss_0;
    float3 plastic_0;
};

void compressed_region_0(float w0_0, float w1_0, float dz_0, float ax_0, float ay_0, array<float, int(6)> thread* region_0)
{
    uint count_1;
    float h0_0 = 0.5f * w0_0;
    float h1_0 = 0.5f * w1_0;
    float _S315 = - h0_0;
    float _S316 = - h1_0;
    array<float2, int(4)> _S317 = { { float2(_S315, _S316), float2(h0_0, _S316), float2(h0_0, h1_0), float2(_S315, h1_0) } };
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
        uint _S318 = i_2;
        uint _S319 = i_2 + 1U;
        uint _S320 = _S319 % 4U;
        float _S321 = _S317[i_2].y;
        float _S322 = _S317[i_2].x;
        float fp_0 = dz_0 + ax_0 * _S321 - ay_0 * _S322;
        float _S323 = _S317[_S320].y;
        float _S324 = _S317[_S320].x;
        float fq_0 = dz_0 + ax_0 * _S323 - ay_0 * _S324;
        bool _S325 = fp_0 < 0.0f;
        if(_S325)
        {
            uint _S326 = count_2 + 1U;
            poly_0[count_2] = _S317[_S318];
            count_1 = _S326;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S325 != (fq_0 < 0.0f))
        {
            float t_11 = fp_0 / (fp_0 - fq_0);
            uint _S327 = count_1 + 1U;
            poly_0[count_1] = float2(_S322 + t_11 * (_S324 - _S322), _S321 + t_11 * (_S323 - _S321));
            count_2 = _S327;
        }
        else
        {
            count_2 = count_1;
        }
        i_2 = _S319;
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
    float2 o_1 = poly_0[int(0)];
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
        float _S328 = o_1.x;
        float x0_0 = poly_0[i_2].x - _S328;
        float _S329 = o_1.y;
        float y0_0 = poly_0[i_2].y - _S329;
        uint _S330 = i_2 + 1U;
        uint _S331 = _S330 % count_2;
        float x1_0 = poly_0[_S331].x - _S328;
        float y1_0 = poly_0[_S331].y - _S329;
        float _S332 = x0_0 * y1_0;
        float _S333 = x1_0 * y0_0;
        float cr_0 = _S332 - _S333;
        float a_6 = a_5 + cr_0 / 2.0f;
        float sx_1 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        float sy_1 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        float ixx_1 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        float iyy_1 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        float ixy_1 = ixy_0 + (_S332 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S333) * cr_0 / 24.0f;
        i_2 = _S330;
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
    (*region_0)[int(1)] = o_1.x + cx_0;
    (*region_0)[int(2)] = o_1.y + cy_0;
    float _S334 = a_5 * cx_0;
    (*region_0)[int(3)] = ixx_0 - _S334 * cx_0;
    (*region_0)[int(4)] = iyy_0 - a_5 * cy_0 * cy_0;
    (*region_0)[int(5)] = ixy_0 - _S334 * cy_0;
    return;
}

__attribute__((noinline))
float4 no_tension_patch_0(float kn_0, float w0_1, float w1_1, float dz_1, float ax_1, float ay_1)
{
    thread array<float, int(6)> r_7;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &r_7);
    float a_7 = r_7[int(0)];
    if((r_7[int(0)]) == 0.0f)
    {
        return float4(0.0f) ;
    }
    float k_7 = kn_0 / (w0_1 * w1_1);
    float fc_3 = dz_1 + ax_1 * r_7[int(2)] - ay_1 * r_7[int(1)];
    float _S335 = a_7 * fc_3;
    float _S336 = - ay_1;
    return float4(k_7 * a_7 * fc_3, k_7 * (_S335 * r_7[int(2)] + (_S336 * r_7[int(5)] + ax_1 * r_7[int(4)])), - k_7 * (_S335 * r_7[int(1)] + (_S336 * r_7[int(3)] + ax_1 * r_7[int(5)])), 0.5f * k_7 * (_S335 * fc_3 + ay_1 * ay_1 * r_7[int(3)] + ax_1 * ax_1 * r_7[int(4)] - 2.0f * ax_1 * ay_1 * r_7[int(5)]));
}

constant array<float, int(6)> SPRING_AT_0 = { { -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f } };
float signum_0(float x_8)
{
    float _S337;
    if(((as_type<uint>((x_8))) & 2147483648U) != 0U)
    {
        _S337 = -1.0f;
    }
    else
    {
        _S337 = 1.0f;
    }
    return _S337;
}

float2 return_map_0(float k_8, float total_0, float plastic_1, float cap_0)
{
    float trial_0 = k_8 * (total_0 - plastic_1);
    if((abs(trial_0)) <= cap_0)
    {
        return float2(trial_0, 0.0f);
    }
    float f_8 = cap_0 * signum_0(trial_0);
    float _S338 = (metal::fast::divide((trial_0 - f_8), (k_8)));
    return float2(f_8, _S338);
}

__attribute__((noinline))
Contact_0 contact_part_0(const JointMaterial_0 constant* mat_4, const JointBond_natural_0 thread* b_4, float crush_2, float3 plastic_2, float3 d_lin_0, float3 d_ang_0)
{
    thread Contact_0 c_13;
    float3 _S339 = float3(0.0f) ;
    (&c_13)->q_lin_1 = _S339;
    (&c_13)->q_ang_1 = _S339;
    (&c_13)->energy_2 = 0.0f;
    (&c_13)->diss_0 = 0.0f;
    (&c_13)->plastic_0 = plastic_2;
    uint _S340 = mat_4->kind_flags_0.y;
    if((_S340 & 2U) == 0U)
    {
        return c_13;
    }
    float4 _S341 = float4(b_4->stiff0_0) ;
    float kn_1 = _S341.x;
    float ks_0 = _S341.y;
    float kt_0 = (float4(b_4->stiff1_0) ).x;
    float4 _S342 = float4(b_4->geom0_0) ;
    float w0_2 = _S342.y;
    float w1_2 = _S342.z;
    float diss_1;
    float nc_sum_0;
    float m1_0;
    float m2_0;
    float energy_3;
    if((_S340 & 4U) != 0U)
    {
        float4 p_0 = no_tension_patch_0(kn_1 * (1.0f - crush_2), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        float _S343 = p_0.y;
        float _S344 = p_0.z;
        float _S345 = p_0.w;
        nc_sum_0 = p_0.x;
        m1_0 = _S343;
        m2_0 = _S344;
        energy_3 = _S345;
    }
    else
    {
        float ki_0 = kn_1 * (1.0f - crush_2) / 36.0f;
        float _S346 = d_ang_0.x;
        float _S347 = d_ang_0.y;
        float spread_0 = abs(_S346) * 0.4166666567325592f * w1_2 + abs(_S347) * 0.4166666567325592f * w0_2;
        float _S348 = d_lin_0.z;
        float slack_0 = 9.99999997475242708e-07f * (abs(_S348) + spread_0);
        if((_S348 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_3 = 0.0f;
        }
        else
        {
            if((_S348 + spread_0) < (- slack_0))
            {
                float i1_0 = 2.91666650772094727f * w0_2 * w0_2;
                float i2_0 = 2.91666650772094727f * w1_2 * w1_2;
                float _S349 = ki_0 * _S346 * i2_0;
                float _S350 = ki_0 * _S347 * i1_0;
                float _S351 = 0.5f * ki_0 * (36.0f * _S348 * _S348 + _S346 * _S346 * i2_0 + _S347 * _S347 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S348;
                m1_0 = _S349;
                m2_0 = _S350;
                energy_3 = _S351;
            }
            else
            {
                uint i_3 = 0U;
                diss_1 = 0.0f;
                float m1_1 = 0.0f;
                float m2_1 = 0.0f;
                float energy_4 = 0.0f;
                for(;;)
                {
                    if(i_3 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S352 = SPRING_AT_0[i_3] * w0_2;
                    uint j_0 = 0U;
                    nc_sum_0 = diss_1;
                    m1_0 = m1_1;
                    m2_0 = m2_1;
                    energy_3 = energy_4;
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
                        float di_0 = _S348 + _S346 * s2_0 - _S347 * _S352;
                        if(di_0 < 0.0f)
                        {
                            float f_9 = ki_0 * di_0;
                            float m1_2 = m1_0 + f_9 * s2_0;
                            float m2_2 = m2_0 - f_9 * _S352;
                            float energy_5 = energy_3 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_9;
                            m1_0 = m1_2;
                            m2_0 = m2_2;
                            energy_3 = energy_5;
                        }
                        j_0 = j_0 + 1U;
                    }
                    i_3 = i_3 + 1U;
                    diss_1 = nc_sum_0;
                    m1_1 = m1_0;
                    m2_1 = m2_0;
                    energy_4 = energy_3;
                }
                nc_sum_0 = diss_1;
                m1_0 = m1_1;
                m2_0 = m2_1;
                energy_3 = energy_4;
            }
        }
    }
    float nc_0 = - nc_sum_0;
    thread float3 p_1 = plastic_2;
    (&c_13)->q_lin_1 = float3(0.0f, 0.0f, nc_sum_0);
    (&c_13)->q_ang_1 = float3(m1_0, m2_0, 0.0f);
    float slide_cap_0 = mat_4->strength_0.w * nc_0;
    float _S353 = ks_0 * (d_lin_0.x - plastic_2.x);
    float _S354 = ks_0 * (d_lin_0.y - plastic_2.y);
    float tn_0 = (metal::fast::sqrt((_S353 * _S353 + _S354 * _S354)));
    bool _S355;
    if(tn_0 > slide_cap_0)
    {
        _S355 = tn_0 > 0.0f;
    }
    else
    {
        _S355 = false;
    }
    if(_S355)
    {
        float _S356 = (metal::fast::divide((_S353), (tn_0)));
        float _S357 = (metal::fast::divide((_S354), (tn_0)));
        float dslip_0 = (metal::fast::divide((tn_0 - slide_cap_0), (ks_0)));
        p_1.x = p_1.x + _S356 * dslip_0;
        p_1.y = p_1.y + _S357 * dslip_0;
        (&c_13)->q_lin_1.x = _S356 * slide_cap_0;
        (&c_13)->q_lin_1.y = _S357 * slide_cap_0;
        diss_1 = slide_cap_0 * dslip_0;
    }
    else
    {
        (&c_13)->q_lin_1.x = _S353;
        (&c_13)->q_lin_1.y = _S354;
        diss_1 = 0.0f;
    }
    float2 tq_0 = return_map_0(kt_0, d_ang_0.z, p_1.z, slide_cap_0 * (float4(b_4->geom1_0) ).z);
    float _S358 = tq_0.x;
    float _S359 = tq_0.y;
    float diss_2 = diss_1 + abs(_S358) * abs(_S359);
    p_1.z = p_1.z + _S359;
    (&c_13)->q_ang_1.z = _S358;
    float _S360 = (metal::fast::divide((sq_0((&c_13)->q_lin_1.x)), (ks_0)));
    float _S361 = (metal::fast::divide((sq_0((&c_13)->q_lin_1.y)), (ks_0)));
    float _S362 = _S360 + _S361;
    float _S363 = (metal::fast::divide((sq_0(_S358)), (kt_0)));
    (&c_13)->energy_2 = energy_3 + 0.5f * (_S362 + _S363);
    (&c_13)->diss_0 = diss_2;
    (&c_13)->plastic_0 = p_1;
    return c_13;
}

float3 contact_offsets_0(const JointMaterial_0 constant* mat_5, const JointBond_natural_0 thread* b_5, float crush_3, float3 plastic_3, float3 d_lin_1, float3 d_ang_1)
{
    uint _S364 = mat_5->kind_flags_0.y;
    if((_S364 & 2U) == 0U)
    {
        return plastic_3;
    }
    float4 _S365 = float4(b_5->stiff0_0) ;
    float kn_2 = _S365.x;
    float ks_1 = _S365.y;
    float kt_1 = (float4(b_5->stiff1_0) ).x;
    float4 _S366 = float4(b_5->geom0_0) ;
    float w0_3 = _S366.y;
    float w1_3 = _S366.z;
    float nc_sum_1;
    if((_S364 & 4U) != 0U)
    {
        float4 _S367 = no_tension_patch_0(kn_2 * (1.0f - crush_3), w0_3, w1_3, d_lin_1.z, d_ang_1.x, d_ang_1.y);
        nc_sum_1 = _S367.x;
    }
    else
    {
        float ki_1 = kn_2 * (1.0f - crush_3) / 36.0f;
        float _S368 = d_ang_1.x;
        float _S369 = d_ang_1.y;
        float spread_1 = abs(_S368) * 0.4166666567325592f * w1_3 + abs(_S369) * 0.4166666567325592f * w0_3;
        float _S370 = d_lin_1.z;
        float slack_1 = 9.99999997475242708e-07f * (abs(_S370) + spread_1);
        if((_S370 - spread_1) > slack_1)
        {
            nc_sum_1 = 0.0f;
        }
        else
        {
            if((_S370 + spread_1) < (- slack_1))
            {
                nc_sum_1 = ki_1 * 36.0f * _S370;
            }
            else
            {
                uint i_4 = 0U;
                float nc_sum_2 = 0.0f;
                for(;;)
                {
                    if(i_4 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S371 = SPRING_AT_0[i_4] * w0_3;
                    uint j_1 = 0U;
                    nc_sum_1 = nc_sum_2;
                    for(;;)
                    {
                        if(j_1 < 6U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float di_1 = _S370 + _S368 * (SPRING_AT_0[j_1] * w1_3) - _S369 * _S371;
                        if(di_1 < 0.0f)
                        {
                            nc_sum_1 = nc_sum_1 + ki_1 * di_1;
                        }
                        j_1 = j_1 + 1U;
                    }
                    i_4 = i_4 + 1U;
                    nc_sum_2 = nc_sum_1;
                }
                nc_sum_1 = nc_sum_2;
            }
        }
    }
    float nc_1 = - nc_sum_1;
    thread float3 p_2 = plastic_3;
    float slide_cap_1 = mat_5->strength_0.w * nc_1;
    float _S372 = ks_1 * (d_lin_1.x - plastic_3.x);
    float _S373 = ks_1 * (d_lin_1.y - plastic_3.y);
    float tn_1 = (metal::fast::sqrt((_S372 * _S372 + _S373 * _S373)));
    bool _S374;
    if(tn_1 > slide_cap_1)
    {
        _S374 = tn_1 > 0.0f;
    }
    else
    {
        _S374 = false;
    }
    if(_S374)
    {
        float _S375 = (metal::fast::divide((_S372), (tn_1)));
        float _S376 = (metal::fast::divide((_S373), (tn_1)));
        float dslip_1 = (metal::fast::divide((tn_1 - slide_cap_1), (ks_1)));
        p_2.x = p_2.x + _S375 * dslip_1;
        p_2.y = p_2.y + _S376 * dslip_1;
    }
    float2 tq_1 = return_map_0(kt_1, d_ang_1.z, p_2.z, slide_cap_1 * (float4(b_5->geom1_0) ).z);
    p_2.z = p_2.z + tq_1.y;
    return p_2;
}

float life_rate_0(const JointMaterial_0 constant* mat_6, float s_3)
{
    if(s_3 <= 0.0f)
    {
        return 0.0f;
    }
    float _S377 = mat_6->misc_0.y;
    float _S378 = _S377 + 1.0f;
    float _S379 = (metal::fast::pow((s_3), (_S377)));
    float _S380 = (metal::fast::divide((_S378 * _S379), (mat_6->misc_0.z)));
    return _S380;
}

JointResponse_0 joint_evaluate_0(const JointMaterial_0 constant* mat_7, const JointBond_natural_0 thread* b_6, const JointState_0 thread* state_2, float3 d_lin_2, float3 d_ang_2, float dt_9, bool fracture_1)
{
    float4 _S381 = float4(b_6->stiff0_0) ;
    float kn_3 = _S381.x;
    float ks_2 = _S381.y;
    float kb1_0 = _S381.z;
    float kb2_0 = _S381.w;
    float4 _S382 = float4(b_6->stiff1_0) ;
    float kt_2 = _S382.x;
    bool has_rebar_1 = (_S382.w) != 0.0f;
    uint kind_4 = mat_7->kind_flags_0.x;
    uint flags_0 = mat_7->kind_flags_0.y;
    bool softening_0 = (flags_0 & 1U) != 0U;
    thread JointState_0 st_2 = *state_2;
    bool _S383 = connected_0(state_2, has_rebar_1);
    float3 qe_lin_0 = d_lin_2 * float3(ks_2, ks_2, kn_3);
    float3 qe_ang_0 = d_ang_2 * float3(kb1_0, kb2_0, kt_2);
    Measures_0 _S384 = stress_measures_0(b_6, qe_lin_0, qe_ang_0);
    float _S385 = max(max(_S384.tension_0, _S384.shear_0), _S384.compression_0);
    bool _S386 = dt_9 > 0.0f;
    float dif_1;
    if(_S386)
    {
        float _S387 = (metal::fast::divide((_S385 - (&st_2)->governing_stress_0), (dt_9)));
        float raw_0 = (metal::fast::divide((max(_S387, 0.0f)), (mat_7->misc_0.w)));
        float tau_2 = _S382.z;
        if((flags_0 & 16U) != 0U)
        {
            float _S388 = (metal::fast::divide((dt_9), (tau_2)));
            dif_1 = - expm1_accurate_0(- _S388);
        }
        else
        {
            float _S389 = (metal::fast::divide((dt_9), (tau_2)));
            dif_1 = min(_S389, 1.0f);
        }
        (&st_2)->strain_rate_0 = (&st_2)->strain_rate_0 + (raw_0 - (&st_2)->strain_rate_0) * dif_1;
        (&st_2)->governing_stress_0 = _S385;
    }
    if((flags_0 & 32U) != 0U)
    {
        float _S390 = dif_factor_0(mat_7, (&st_2)->strain_rate_0);
        dif_1 = _S390;
    }
    else
    {
        dif_1 = 1.0f;
    }
    float weibull_0 = (float4(b_6->geom1_0) ).w;
    float _S391 = fatigue_factor_0(mat_7, (&st_2)->life_0);
    float multiplier_1 = weibull_0 * dif_1 * _S391;
    thread Measures_0 _S392 = _S384;
    float4 _S393 = failure_indices_0(mat_7, b_6, &_S392, multiplier_1);
    float _S394 = _S393.x;
    float _S395 = _S393.y;
    (&st_2)->utilization_0 = max(max(_S394, _S395), max(_S393.z, _S393.w));
    float _S396 = d_lin_2.x;
    float _S397 = d_lin_2.y;
    float _S398 = ks_2 * (sq_0(_S396) + sq_0(_S397)) + kb1_0 * sq_0(d_ang_2.x) + kb2_0 * sq_0(d_ang_2.y) + kt_2 * sq_0(d_ang_2.z);
    float _S399 = d_lin_2.z;
    bool _S400 = _S399 > 0.0f;
    if(_S400)
    {
        dif_1 = kn_3 * sq_0(_S399);
    }
    else
    {
        dif_1 = 0.0f;
    }
    float psi_ts_0 = 0.5f * (_S398 + dif_1);
    float psi_c_0;
    if(_S399 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S399);
    }
    else
    {
        psi_c_0 = 0.0f;
    }
    float3 plastic_4 = float3((&st_2)->plastic_x_0, (&st_2)->plastic_y_0, (&st_2)->plastic_t_0);
    float diss_contact_0;
    float psi_contact_0;
    float intact_normal_0;
    float dissipated_2;
    float overshoot_1;
    bool _S401;
    float3 qc_lin_0;
    if(fracture_1)
    {
        bool _S402 = _S394 >= _S395;
        if(_S402)
        {
            diss_contact_0 = _S394;
        }
        else
        {
            diss_contact_0 = _S395;
        }
        uint mode_ts_0;
        if(_S402)
        {
            mode_ts_0 = 1U;
        }
        else
        {
            mode_ts_0 = 2U;
        }
        if(diss_contact_0 > ((&st_2)->kappa_0))
        {
            _S401 = diss_contact_0 > 1.0f;
        }
        else
        {
            _S401 = false;
        }
        if(_S401)
        {
            _S401 = psi_ts_0 > 0.0f;
        }
        else
        {
            _S401 = false;
        }
        uint mode_c_0;
        if(_S401)
        {
            if(mode_ts_0 == 1U)
            {
                psi_contact_0 = mat_7->energy_1.y;
            }
            else
            {
                psi_contact_0 = mat_7->energy_1.z;
            }
            if(softening_0)
            {
                float _S403 = (metal::fast::divide((psi_contact_0 * (float4(b_6->geom0_0) ).x * diss_contact_0 * diss_contact_0), (psi_ts_0)));
                intact_normal_0 = _S403;
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            (&st_2)->ductility_0 = intact_normal_0;
            if(softening_0)
            {
                mode_c_0 = kind_4;
            }
            else
            {
                mode_c_0 = 0U;
            }
            float2 inc_0 = damage_increment_0(mode_c_0, (&st_2)->kappa_0, diss_contact_0, intact_normal_0, (&st_2)->damage_0, psi_ts_0);
            float _S404 = inc_0.x;
            if(_S404 > ((&st_2)->damage_0))
            {
                Contact_0 _S405 = contact_part_0(mat_7, b_6, (&st_2)->crush_1, plastic_4, d_lin_2, d_ang_2);
                float _S406 = max(_S405.energy_2 - (1.0f - (&st_2)->crush_1) * psi_c_0, 0.0f);
                float _S407 = max(inc_0.y - _S406 * (_S404 - (&st_2)->damage_0), 0.0f);
                float _S408 = max((psi_ts_0 - _S406) * (_S404 - (&st_2)->damage_0) - _S407, 0.0f);
                (&st_2)->damage_0 = _S404;
                (&st_2)->mode_0 = mode_ts_0;
                dissipated_2 = _S407;
                overshoot_1 = _S408;
            }
            else
            {
                dissipated_2 = 0.0f;
                overshoot_1 = 0.0f;
            }
        }
        else
        {
            dissipated_2 = 0.0f;
            overshoot_1 = 0.0f;
        }
        (&st_2)->kappa_0 = max((&st_2)->kappa_0, diss_contact_0);
        float _S409 = state_2->damage_0;
        if((state_2->damage_0) > 0.0f)
        {
            Contact_0 _S410 = contact_part_0(mat_7, b_6, state_2->crush_1, float3(state_2->plastic_x_0, state_2->plastic_y_0, state_2->plastic_t_0), d_lin_2, d_ang_2);
            qc_lin_0 = qe_ang_0 * float3((1.0f - _S409))  + _S410.q_ang_1 * float3(_S409) ;
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        Measures_0 _S411 = stress_measures_0(b_6, float3(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        thread Measures_0 _S412 = _S411;
        float4 _S413 = failure_indices_0(mat_7, b_6, &_S412, multiplier_1);
        float _S414 = _S413.z;
        float _S415 = _S413.w;
        bool _S416 = _S414 >= _S415;
        if(_S416)
        {
            psi_contact_0 = _S414;
        }
        else
        {
            psi_contact_0 = _S415;
        }
        if(_S416)
        {
            mode_c_0 = 3U;
        }
        else
        {
            mode_c_0 = 4U;
        }
        if(psi_contact_0 > ((&st_2)->kappa_c_0))
        {
            _S401 = psi_contact_0 > 1.0f;
        }
        else
        {
            _S401 = false;
        }
        if(_S401)
        {
            _S401 = psi_c_0 > 0.0f;
        }
        else
        {
            _S401 = false;
        }
        if(_S401)
        {
            if(softening_0)
            {
                float _S417 = (metal::fast::divide((mat_7->energy_1.w * (float4(b_6->geom0_0) ).x * psi_contact_0 * psi_contact_0), (psi_c_0)));
                intact_normal_0 = _S417;
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
                    mode_ts_0 = kind_4;
                }
                law_1 = mode_ts_0;
            }
            float2 inc_1 = damage_increment_0(law_1, (&st_2)->kappa_c_0, psi_contact_0, intact_normal_0, (&st_2)->crush_1, psi_c_0);
            float _S418 = inc_1.x;
            if(_S418 > ((&st_2)->crush_1))
            {
                float _S419 = inc_1.y;
                float dissipated_3 = dissipated_2 + _S419;
                float overshoot_2 = overshoot_1 + max(psi_c_0 * (_S418 - (&st_2)->crush_1) - _S419, 0.0f);
                (&st_2)->crush_1 = _S418;
                (&st_2)->mode_0 = mode_c_0;
                if(_S418 >= 1.0f)
                {
                    _S401 = ((&st_2)->damage_0) < 1.0f;
                }
                else
                {
                    _S401 = false;
                }
                if(_S401)
                {
                    float dissipated_4 = dissipated_3 + psi_ts_0 * (1.0f - (&st_2)->damage_0);
                    (&st_2)->damage_0 = 1.0f;
                    dissipated_2 = dissipated_4;
                }
                else
                {
                    dissipated_2 = dissipated_3;
                }
                overshoot_1 = overshoot_2;
            }
        }
        (&st_2)->kappa_c_0 = max((&st_2)->kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_2 = 0.0f;
        overshoot_1 = 0.0f;
    }
    float dmg_0 = (&st_2)->damage_0;
    float3 _S420 = float3(0.0f) ;
    float3 qc_ang_0;
    if(((&st_2)->damage_0) == 0.0f)
    {
        if((flags_0 & 8U) == 0U)
        {
            float3 _S421 = contact_offsets_0(mat_7, b_6, (&st_2)->crush_1, plastic_4, d_lin_2, d_ang_2);
            (&st_2)->plastic_x_0 = _S421.x;
            (&st_2)->plastic_y_0 = _S421.y;
            (&st_2)->plastic_t_0 = _S421.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S420;
        qc_ang_0 = _S420;
        psi_contact_0 = 0.0f;
    }
    else
    {
        Contact_0 _S422 = contact_part_0(mat_7, b_6, (&st_2)->crush_1, plastic_4, d_lin_2, d_ang_2);
        (&st_2)->plastic_x_0 = _S422.plastic_0.x;
        (&st_2)->plastic_y_0 = _S422.plastic_0.y;
        (&st_2)->plastic_t_0 = _S422.plastic_0.z;
        diss_contact_0 = _S422.diss_0;
        qc_lin_0 = _S422.q_lin_1;
        qc_ang_0 = _S422.q_ang_1;
        psi_contact_0 = _S422.energy_2;
    }
    float dissipated_5 = dissipated_2 + dmg_0 * diss_contact_0;
    if(_S400)
    {
        intact_normal_0 = kn_3 * _S399;
    }
    else
    {
        intact_normal_0 = (1.0f - (&st_2)->crush_1) * kn_3 * _S399;
    }
    float _S423 = 1.0f - dmg_0;
    float3 force_lin_2 = float3(_S423 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S423 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S423 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    float3 force_ang_2 = qe_ang_0 * float3(_S423)  + qc_ang_0 * float3(dmg_0) ;
    float stored_1 = _S423 * (psi_ts_0 + (1.0f - (&st_2)->crush_1) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_1)
    {
        _S401 = ((&st_2)->rebar_broken_0) == 0.0f;
    }
    else
    {
        _S401 = false;
    }
    float stored_2;
    float3 force_lin_3;
    if(_S401)
    {
        float4 _S424 = float4(b_6->rebar0_0) ;
        float k_axial_0 = _S424.x;
        float k_dowel_0 = _S424.y;
        float yield_force_0 = _S424.z;
        float dowel_capacity_0 = _S424.w;
        float2 nr_0 = return_map_0(k_axial_0, _S399, (&st_2)->rebar_plastic_0, yield_force_0);
        float2 v1_0 = return_map_0(k_dowel_0, _S396, (&st_2)->rebar_slip0_0, dowel_capacity_0);
        float2 v2_0 = return_map_0(k_dowel_0, _S397, (&st_2)->rebar_slip1_0, dowel_capacity_0);
        float _S425 = nr_0.y;
        float _S426 = v1_0.y;
        float _S427 = v2_0.y;
        float work_2 = yield_force_0 * abs(_S425) + dowel_capacity_0 * (abs(_S426) + abs(_S427));
        (&st_2)->rebar_plastic_0 = (&st_2)->rebar_plastic_0 + _S425;
        (&st_2)->rebar_slip0_0 = (&st_2)->rebar_slip0_0 + _S426;
        (&st_2)->rebar_slip1_0 = (&st_2)->rebar_slip1_0 + _S427;
        (&st_2)->rebar_work_0 = (&st_2)->rebar_work_0 + work_2;
        float dissipated_6 = dissipated_5 + work_2;
        float _S428 = nr_0.x;
        float _S429 = (metal::fast::divide((sq_0(_S428)), (k_axial_0)));
        float _S430 = v1_0.x;
        float _S431 = v2_0.x;
        float _S432 = (metal::fast::divide((sq_0(_S430) + sq_0(_S431)), (k_dowel_0)));
        float elastic_0 = 0.5f * (_S429 + _S432);
        if(fracture_1)
        {
            _S401 = ((&st_2)->rebar_work_0) >= ((float4(b_6->rebar1_0) ).x);
        }
        else
        {
            _S401 = false;
        }
        if(_S401)
        {
            (&st_2)->rebar_broken_0 = 1.0f;
            float dissipated_7 = dissipated_6 + elastic_0;
            force_lin_3 = force_lin_2;
            dissipated_2 = dissipated_7;
            stored_2 = stored_1;
        }
        else
        {
            float stored_3 = stored_1 + elastic_0;
            force_lin_3 = force_lin_2 + float3(_S430, _S431, _S428);
            dissipated_2 = dissipated_6;
            stored_2 = stored_3;
        }
    }
    else
    {
        force_lin_3 = force_lin_2;
        dissipated_2 = dissipated_5;
        stored_2 = stored_1;
    }
    if(fracture_1)
    {
        _S401 = _S386;
    }
    else
    {
        _S401 = false;
    }
    if(_S401)
    {
        _S401 = (flags_0 & 64U) != 0U;
    }
    else
    {
        _S401 = false;
    }
    if(_S401)
    {
        Measures_0 _S433 = stress_measures_0(b_6, force_lin_3, force_ang_2);
        thread Measures_0 _S434 = _S433;
        float4 _S435 = failure_indices_0(mat_7, b_6, &_S434, weibull_0);
        float _S436 = life_rate_0(mat_7, max(max(_S435.x, _S435.y), _S435.z));
        (&st_2)->life_0 = max((&st_2)->life_0 - _S436 * dt_9, 0.0f);
    }
    (&st_2)->dissipated_0 = (&st_2)->dissipated_0 + dissipated_2;
    thread JointResponse_0 resp_0;
    (&resp_0)->force_lin_1 = force_lin_3;
    (&resp_0)->force_ang_1 = force_ang_2;
    (&resp_0)->state_1 = st_2;
    (&resp_0)->dissipated_1 = dissipated_2;
    (&resp_0)->overshoot_0 = overshoot_1;
    (&resp_0)->stored_0 = stored_2;
    if(_S383)
    {
        thread JointState_0 _S437 = st_2;
        bool _S438 = connected_0(&_S437, has_rebar_1);
        _S401 = !_S438;
    }
    else
    {
        _S401 = false;
    }
    (&resp_0)->disconnected_0 = _S401;
    (&resp_0)->fatigue_before_0 = _S391;
    (&resp_0)->measures_0 = _S384;
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

void secant_factors_0(const JointBond_natural_0 thread* b_7, const JointState_0 thread* st_3, float3 d_lin_3, float3 thread* f_lin_0, float3 thread* f_ang_0)
{
    float _S439 = st_3->damage_0;
    bool compressed_0 = (d_lin_3.z) < 0.0f;
    float contact_3;
    if(compressed_0)
    {
        contact_3 = _S439;
    }
    else
    {
        contact_3 = 0.0f;
    }
    float _S440 = 1.0f - _S439;
    float _S441 = max(_S440 + contact_3, 9.99999997475242708e-07f);
    float normal_1;
    if(compressed_0)
    {
        normal_1 = max(1.0f - st_3->crush_1, 9.99999997475242708e-07f);
    }
    else
    {
        normal_1 = max(_S440, 9.99999997475242708e-07f);
    }
    *f_lin_0 = float3(_S441, _S441, normal_1);
    *f_ang_0 = float3(_S441) ;
    bool _S442;
    if(((float4(b_7->stiff1_0) ).w) != 0.0f)
    {
        _S442 = (st_3->rebar_broken_0) == 0.0f;
    }
    else
    {
        _S442 = false;
    }
    if(_S442)
    {
        float4 _S443 = float4(b_7->rebar0_0) ;
        float4 _S444 = float4(b_7->stiff0_0) ;
        float _S445 = (metal::fast::divide((_S443.x), (_S444.x)));
        (*f_lin_0).z = (*f_lin_0).z + _S445;
        float _S446 = _S443.y;
        float _S447 = _S444.y;
        float _S448 = (metal::fast::divide((_S446), (_S447)));
        (*f_lin_0).x = (*f_lin_0).x + _S448;
        float _S449 = (metal::fast::divide((_S446), (_S447)));
        (*f_lin_0).y = (*f_lin_0).y + _S449;
    }
    return;
}

float3 to_body_0(uint _S450, float3 _S451, KernelContext_0 thread* kernelContext_23)
{
    BondStatic_natural_0 device* _S452 = kernelContext_23->bonds_0+_S450;
    return (float4(_S452->t1_0) ).xyz * float3(_S451.x)  + (float4(_S452->t2_0) ).xyz * float3(_S451.y)  + (float4(_S452->normal_0) ).xyz * float3(_S451.z) ;
}

void cov_or_0(uint island_0, uint bits_0, KernelContext_0 thread* kernelContext_24)
{
    uint _S453 = atomic_fetch_or_explicit(((atomic_uint device*)(&(kernelContext_24->islands_0+island_0)->coverage_0)), bits_0, memory_order_relaxed);
    return;
}

bool bond_update_0(uint i_5, float dt_10, bool fracture_2, uint abs_step_0, bool wide_0, KernelContext_0 thread* kernelContext_25)
{
    BondStatic_natural_0 device* _S454 = kernelContext_25->bonds_0+i_5;
    uint4 _S455 = uint4(_S454->law_0.ids_0) ;
    float3 ra_1 = (float4(_S454->ra_0) ).xyz;
    float3 rb_1 = (float4(_S454->rb_0) ).xyz;
    uint _S456 = 4U * _S455.y;
    float3 ta_0 = (float4(*(kernelContext_25->state_0+(_S456 + 1U))) ).xyz;
    float3 va_0 = (float4(*(kernelContext_25->state_0+(_S456 + 2U))) ).xyz;
    float3 wa_1 = (float4(*(kernelContext_25->state_0+(_S456 + 3U))) ).xyz;
    uint _S457 = 4U * _S455.z;
    float3 tb_0 = (float4(*(kernelContext_25->state_0+(_S457 + 1U))) ).xyz;
    float3 vb_0 = (float4(*(kernelContext_25->state_0+(_S457 + 2U))) ).xyz;
    float3 wb_1 = (float4(*(kernelContext_25->state_0+(_S457 + 3U))) ).xyz;
    float3 _S458 = to_local_0(i_5, (float4(*(kernelContext_25->state_0+_S457)) ).xyz + cross(tb_0, rb_1) - ((float4(*(kernelContext_25->state_0+_S456)) ).xyz + cross(ta_0, ra_1)), kernelContext_25);
    float3 _S459 = to_local_0(i_5, tb_0 - ta_0, kernelContext_25);
    float3 _S460 = to_local_0(i_5, vb_0 + cross(wb_1, rb_1) - (va_0 + cross(wa_1, ra_1)), kernelContext_25);
    float3 _S461 = to_local_0(i_5, wb_1 - wa_1, kernelContext_25);
    JointState_0 previous_1 = (kernelContext_25->bond_dyn_0+i_5)->js_0;
    thread JointState_0 _S462 = (kernelContext_25->bond_dyn_0+i_5)->js_0;
    bool _S463 = is_damaged_0(&_S462);
    thread JointBond_natural_0 _S464 = (kernelContext_25->bonds_0+i_5)->law_0;
    thread JointState_0 _S465 = (kernelContext_25->bond_dyn_0+i_5)->js_0;
    JointResponse_0 _S466 = joint_evaluate_0(&kernelContext_25->materials_0->m_0[(uint4((kernelContext_25->bonds_0+i_5)->law_0.ids_0) ).x], &_S464, &_S465, _S458, _S459, dt_10, fracture_2);
    BondStatic_natural_0 device* _S467 = kernelContext_25->bonds_0+i_5;
    BondDyn_natural_0 device* _S468 = kernelContext_25->bond_dyn_0+i_5;
    float4 _S469 = float4((*_S468).force_lin_0) ;
    float4 _S470 = float4((*_S468).force_ang_0) ;
    float4 _S471 = float4((*_S468).sums_0) ;
    float4 _S472 = float4((*_S468).comps_0) ;
    uint4 _S473 = uint4((*_S468).events_0) ;
    thread BondDyn_0 bd_0;
    (&bd_0)->js_0 = (*_S468).js_0;
    (&bd_0)->force_lin_0 = _S469;
    (&bd_0)->force_ang_0 = _S470;
    (&bd_0)->sums_0 = _S471;
    (&bd_0)->comps_0 = _S472;
    (&bd_0)->events_0 = _S473;
    float3 ra_2 = (float4(_S467->ra_0) ).xyz;
    float3 rb_2 = (float4(_S467->rb_0) ).xyz;
    thread JointBond_natural_0 _S474 = _S467->law_0;
    thread JointState_0 _S475 = _S466.state_1;
    thread float3 f_lin_1;
    thread float3 f_ang_1;
    secant_factors_0(&_S474, &_S475, _S458, &f_lin_1, &f_ang_1);
    float3 qd_lin_0 = _S460 * (float4(_S467->c_lin_0) ).xyz * f_lin_1;
    float3 qd_ang_0 = _S461 * (float4(_S467->c_ang_0) ).xyz * f_ang_1;
    float3 q_lin_2 = _S466.force_lin_1 + qd_lin_0;
    float3 q_ang_2 = _S466.force_ang_1 + qd_ang_0;
    float damped_0 = (dot(qd_lin_0, _S460) + dot(qd_ang_0, _S461)) * dt_10;
    float3 _S476 = to_body_0(i_5, q_lin_2, kernelContext_25);
    float3 _S477 = to_body_0(i_5, q_ang_2, kernelContext_25);
    uint _S478 = 3U * i_5;
    *(kernelContext_25->scratch_0+_S478) = packed_float4(float4(_S476, max(_S466.measures_0.tension_0, _S466.measures_0.compression_0))) ;
    *(kernelContext_25->scratch_0+(_S478 + 1U)) = packed_float4(float4(_S477 + cross(ra_2, _S476), 0.0f)) ;
    *(kernelContext_25->scratch_0+(_S478 + 2U)) = packed_float4(float4(- _S477 + cross(rb_2, - _S476), 0.0f)) ;
    thread float _S479 = (&bd_0)->sums_0.x;
    thread float _S480 = (&bd_0)->comps_0.x;
    comp_add1_0(&_S479, &_S480, _S466.dissipated_1);
    (&bd_0)->comps_0.x = _S480;
    (&bd_0)->sums_0.x = _S479;
    thread float _S481 = (&bd_0)->sums_0.y;
    thread float _S482 = (&bd_0)->comps_0.y;
    comp_add1_0(&_S481, &_S482, _S466.overshoot_0);
    (&bd_0)->comps_0.y = _S482;
    (&bd_0)->sums_0.y = _S481;
    thread float _S483 = (&bd_0)->sums_0.z;
    thread float _S484 = (&bd_0)->comps_0.z;
    comp_add1_0(&_S483, &_S484, damped_0);
    (&bd_0)->comps_0.z = _S484;
    (&bd_0)->sums_0.z = _S483;
    (&bd_0)->force_lin_0 = float4(q_lin_2, _S466.stored_0);
    (&bd_0)->force_ang_0 = float4(q_ang_2, max((&bd_0)->force_ang_0.w, _S466.state_1.utilization_0));
    bool _S485;
    if(!_S463)
    {
        thread JointState_0 _S486 = _S466.state_1;
        bool _S487 = is_damaged_0(&_S486);
        _S485 = _S487;
    }
    else
    {
        _S485 = false;
    }
    if(_S485)
    {
        _S485 = ((&bd_0)->events_0.x) == 0U;
    }
    else
    {
        _S485 = false;
    }
    if(_S485)
    {
        (&bd_0)->events_0.x = abs_step_0;
        (&bd_0)->events_0.w = _S466.state_1.mode_0;
        if(wide_0)
        {
            cov_or_0((uint4((&_S474)->ids_0) ).w, 32U, kernelContext_25);
        }
        else
        {
            (*kernelContext_25->g_seen_0)[3U] = 1U;
        }
    }
    if(((&bd_0)->events_0.y) == 0U)
    {
        _S485 = (_S466.state_1.life_0) != (previous_1.life_0);
    }
    else
    {
        _S485 = false;
    }
    if(_S485)
    {
        _S485 = (_S466.fatigue_before_0) > 0.99000000953674316f;
    }
    else
    {
        _S485 = false;
    }
    if(_S485)
    {
        float _S488 = fatigue_factor_0(&kernelContext_25->materials_0->m_0[(uint4((kernelContext_25->bonds_0+i_5)->law_0.ids_0) ).x], _S466.state_1.life_0);
        _S485 = _S488 <= 0.99000000953674316f;
    }
    else
    {
        _S485 = false;
    }
    if(_S485)
    {
        (&bd_0)->events_0.y = abs_step_0;
        if(wide_0)
        {
            cov_or_0((uint4((&_S474)->ids_0) ).w, 64U, kernelContext_25);
        }
        else
        {
            (*kernelContext_25->g_seen_0)[4U] = 1U;
        }
    }
    if(_S466.disconnected_0)
    {
        (&bd_0)->events_0.z = abs_step_0;
    }
    (&bd_0)->js_0 = _S466.state_1;
    BondDyn_natural_0 device* _S489 = kernelContext_25->bond_dyn_0+i_5;
    _S489->js_0 = bd_0.js_0;
    _S489->force_lin_0 = packed_float4(bd_0.force_lin_0) ;
    _S489->force_ang_0 = packed_float4(bd_0.force_ang_0) ;
    _S489->sums_0 = packed_float4(bd_0.sums_0) ;
    _S489->comps_0 = packed_float4(bd_0.comps_0) ;
    _S489->events_0 = packed_uint4(bd_0.events_0) ;
    return _S466.disconnected_0;
}

void island_frame_at_1(uint gi_1, uint tid_5, KernelContext_0 thread* kernelContext_26)
{
    bool woke_2;
    Island_natural_0 device* _S490 = kernelContext_26->islands_0+gi_1;
    uint4 _S491 = uint4((*_S490).info_0) ;
    float4 _S492 = float4((*_S490).com_0) ;
    float4 _S493 = float4((*_S490).inertia0_0) ;
    float4 _S494 = float4((*_S490).inertia1_0) ;
    float4 _S495 = float4((*_S490).inertia2_0) ;
    float4 _S496 = float4((*_S490).inv0_0) ;
    float4 _S497 = float4((*_S490).inv1_0) ;
    float4 _S498 = float4((*_S490).inv2_0) ;
    float4 _S499 = float4((*_S490).wcom_0) ;
    float4 _S500 = float4((*_S490).winv0_0) ;
    float4 _S501 = float4((*_S490).winv1_0) ;
    float4 _S502 = float4((*_S490).winv2_0) ;
    float4 _S503 = float4((*_S490).rotation_0) ;
    float4 _S504 = float4((*_S490).position_0) ;
    float4 _S505 = float4((*_S490).position_err_0) ;
    float4 _S506 = float4((*_S490).velocity_0) ;
    float4 _S507 = float4((*_S490).velocity_err_0) ;
    float4 _S508 = float4((*_S490).angular_velocity_0) ;
    uint4 _S509 = uint4((*_S490).probes_0) ;
    float4 _S510 = float4((*_S490).energy_0) ;
    float4 _S511 = float4((*_S490).rotation_err_0) ;
    float4 _S512 = float4((*_S490).momentum_0) ;
    float4 _S513 = float4((*_S490).momentum_err_0) ;
    thread Island_0 isl_10;
    (&isl_10)->range_0 = uint4((*_S490).range_0) ;
    (&isl_10)->info_0 = _S491;
    (&isl_10)->com_0 = _S492;
    (&isl_10)->inertia0_0 = _S493;
    (&isl_10)->inertia1_0 = _S494;
    (&isl_10)->inertia2_0 = _S495;
    (&isl_10)->inv0_0 = _S496;
    (&isl_10)->inv1_0 = _S497;
    (&isl_10)->inv2_0 = _S498;
    (&isl_10)->wcom_0 = _S499;
    (&isl_10)->winv0_0 = _S500;
    (&isl_10)->winv1_0 = _S501;
    (&isl_10)->winv2_0 = _S502;
    (&isl_10)->rotation_0 = _S503;
    (&isl_10)->position_0 = _S504;
    (&isl_10)->position_err_0 = _S505;
    (&isl_10)->velocity_0 = _S506;
    (&isl_10)->velocity_err_0 = _S507;
    (&isl_10)->angular_velocity_0 = _S508;
    (&isl_10)->done_0 = (*_S490).done_0;
    (&isl_10)->coverage_0 = (*_S490).coverage_0;
    (&isl_10)->wide_groups_0 = (*_S490).wide_groups_0;
    (&isl_10)->done_pad_0 = (*_S490).done_pad_0;
    (&isl_10)->probes_0 = _S509;
    (&isl_10)->energy_0 = _S510;
    (&isl_10)->rotation_err_0 = _S511;
    (&isl_10)->momentum_0 = _S512;
    (&isl_10)->momentum_err_0 = _S513;
    bool driven_1 = (((&isl_10)->info_0.x) & 2U) != 0U;
    bool _S514 = !((((&isl_10)->info_0.x) & 1U) != 0U);
    bool _S515;
    if(_S514)
    {
        _S515 = !driven_1;
    }
    else
    {
        _S515 = false;
    }
    bool contact_island_1 = (((&isl_10)->info_0.x) & 4U) != 0U;
    bool _S516 = (((&isl_10)->info_0.x) & 16U) != 0U;
    seen_clear_0(tid_5, kernelContext_26);
    bool _S517 = tid_5 == 0U;
    bool settled_1;
    uint cov_5;
    if(_S517)
    {
        if(contact_island_1 != ((kernelContext_26->params_0->contact_mode_0) == 1U))
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
            thread Island_0 _S518 = isl_10;
            bool _S519 = contact_stopped_0(&_S518, kernelContext_26);
            settled_1 = _S519;
        }
        else
        {
            settled_1 = false;
        }
        if(settled_1)
        {
            cov_5 = 0U;
        }
        *kernelContext_26->g_run_0 = cov_5;
        *kernelContext_26->g_halt_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if((((&isl_10)->info_0.z) & 1U) != 0U)
    {
        settled_1 = true;
    }
    else
    {
        settled_1 = (*kernelContext_26->g_run_0) == 0U;
    }
    if(settled_1)
    {
        cov_5 = 0U;
    }
    else
    {
        cov_5 = min((&isl_10)->info_0.y, kernelContext_26->params_0->max_steps_0);
    }
    float _S520 = kernelContext_26->params_0->dt_0;
    bool _S521 = (kernelContext_26->params_0->fracture_0) != 0U;
    bool _S522 = (kernelContext_26->params_0->rigid_motion_loads_0) != 0U;
    thread Island_0 _S523 = isl_10;
    Rigid_0 _S524 = rigid_of_0(&_S523);
    thread Rigid_0 rg_8 = _S524;
    thread float work_3 = 0.0f;
    thread float work_err_2 = 0.0f;
    settled_1 = _S516;
    uint done_2 = 0U;
    bool woke_3 = false;
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
        uint abs_step_1 = (&isl_10)->info_0.w + done_2 + 1U;
        uint k_9 = abs_step_1 - 1U - kernelContext_26->params_0->step_start_0;
        bool quiet_2 = (((&isl_10)->info_0.x) & 32U) != 0U;
        bool _S525;
        if(settled_1)
        {
            _S525 = !quiet_2;
        }
        else
        {
            _S525 = false;
        }
        bool _S526;
        bool explicit_step_1;
        uint c_14;
        if(_S525)
        {
            float3 _S527 = float3(0.0f) ;
            thread float3 norm_1 = _S527;
            thread float3 unused0_1 = _S527;
            c_14 = (&isl_10)->range_0.x + tid_5;
            for(;;)
            {
                if(c_14 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Quat_0 _S528 = (&rg_8)->rot_0;
                float _S529 = settled_chunk_load_0(c_14, &_S528, k_9, _S520, contact_island_1, kernelContext_26);
                norm_1.x = norm_1.x + _S529;
                c_14 = c_14 + 256U;
            }
            group_sum3_0(tid_5, &norm_1, &unused0_1, kernelContext_26);
            if((kernelContext_26->params_0->solve_mode_0) == 1U)
            {
                _S526 = (abs(norm_1.x - (&isl_10)->energy_0.z)) > ((&isl_10)->energy_0.w);
            }
            else
            {
                _S526 = false;
            }
            if(_S526)
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
        if(_S514)
        {
            _S526 = !quiet_2;
        }
        else
        {
            _S526 = false;
        }
        if(_S526)
        {
            float3 _S530 = float3(0.0f) ;
            thread float3 f_10 = _S530;
            thread float3 t_12 = _S530;
            c_14 = (&isl_10)->range_0.x + tid_5;
            for(;;)
            {
                if(c_14 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S531 = isl_10;
                thread Rigid_0 _S532 = rg_8;
                net_load_0(c_14, &_S531, &_S532, k_9, _S520, contact_island_1, &f_10, &t_12, kernelContext_26);
                c_14 = c_14 + 256U;
            }
            group_sum3_0(tid_5, &f_10, &t_12, kernelContext_26);
            thread Island_0 _S533 = isl_10;
            rigid_acceleration_0(&_S533, &rg_8, f_10, t_12);
        }
        if(!settled_1)
        {
            explicit_step_1 = !quiet_2;
        }
        else
        {
            explicit_step_1 = false;
        }
        uint c_15;
        if(explicit_step_1)
        {
            c_14 = (&isl_10)->range_0.z + tid_5;
            for(;;)
            {
                if(c_14 < ((&isl_10)->range_0.w))
                {
                }
                else
                {
                    break;
                }
                bool _S534 = bond_update_0(c_14, _S520, _S521, abs_step_1, false, kernelContext_26);
                if(_S534)
                {
                    *kernelContext_26->g_halt_0 = 1U;
                }
                c_14 = c_14 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_15 = (&isl_10)->range_0.x + tid_5;
            for(;;)
            {
                if(c_15 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                thread Island_0 _S535 = isl_10;
                thread Rigid_0 _S536 = rg_8;
                chunk_update_0(c_15, &_S535, &_S536, _S520, _S522, k_9, contact_island_1, &work_3, &work_err_2, kernelContext_26);
                c_15 = c_15 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S537;
        if(_S515)
        {
            _S537 = !quiet_2;
        }
        else
        {
            _S537 = false;
        }
        if(_S537)
        {
            thread Island_0 _S538 = isl_10;
            integrate_rigid_0(&_S538, &rg_8, _S520);
        }
        bool _S539;
        if(_S514)
        {
            _S539 = explicit_step_1;
        }
        else
        {
            _S539 = false;
        }
        if(_S539)
        {
            float3 _S540 = (&isl_10)->wcom_0.xyz;
            float3 _S541 = float3(0.0f) ;
            thread float3 tu_2 = _S541;
            thread float3 pv_2 = _S541;
            c_14 = (&isl_10)->range_0.x + tid_5;
            for(;;)
            {
                if(c_14 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_moments_0(c_14, &tu_2, &pv_2, kernelContext_26);
                c_14 = c_14 + 256U;
            }
            group_sum3_0(tid_5, &tu_2, &pv_2, kernelContext_26);
            float3 tr_4 = tu_2 / float3((&isl_10)->wcom_0.w) ;
            float3 dv_4 = pv_2 / float3((&isl_10)->wcom_0.w) ;
            thread float3 lu_2 = _S541;
            thread float3 lv_2 = _S541;
            c_15 = (&isl_10)->range_0.x + tid_5;
            for(;;)
            {
                if(c_15 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_angular_0(c_15, _S540, tr_4, dv_4, &lu_2, &lv_2, kernelContext_26);
                c_15 = c_15 + 256U;
            }
            group_sum3_0(tid_5, &lu_2, &lv_2, kernelContext_26);
            float3 phi_4 = rows_mul_0((&isl_10)->winv0_0, (&isl_10)->winv1_0, (&isl_10)->winv2_0, lu_2);
            float3 dw_3 = rows_mul_0((&isl_10)->winv0_0, (&isl_10)->winv1_0, (&isl_10)->winv2_0, lv_2);
            uint c_16 = (&isl_10)->range_0.x + tid_5;
            for(;;)
            {
                if(c_16 < ((&isl_10)->range_0.y))
                {
                }
                else
                {
                    break;
                }
                drift_apply_0(c_16, _S540, tr_4, phi_4, dv_4, dw_3, kernelContext_26);
                c_16 = c_16 + 256U;
            }
            if(!driven_1)
            {
                thread Island_0 _S542 = isl_10;
                drift_rigid_0(&_S542, &rg_8, tr_4, phi_4, dv_4, dw_3);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        }
        bool _S543;
        if(_S517)
        {
            _S543 = ((&isl_10)->probes_0.y) > ((&isl_10)->probes_0.x);
        }
        else
        {
            _S543 = false;
        }
        if(_S543)
        {
            thread Island_0 _S544 = isl_10;
            thread Rigid_0 _S545 = rg_8;
            record_probes_0(&_S544, &_S545, k_9, kernelContext_26);
        }
        uint _S546 = done_2 + 1U;
        bool _S547;
        if((*kernelContext_26->g_halt_0) != 0U)
        {
            thread Island_0 _S548 = isl_10;
            bool _S549 = island_connected_narrow_0(tid_5, &_S548, kernelContext_26);
            _S547 = _S549;
        }
        else
        {
            _S547 = false;
        }
        bool resumed_3;
        if(_S547)
        {
            if(_S517)
            {
                *kernelContext_26->g_halt_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            resumed_3 = true;
        }
        else
        {
            resumed_3 = resumed_2;
        }
        if((*kernelContext_26->g_halt_0) != 0U)
        {
            done_2 = _S546;
            resumed_2 = resumed_3;
            break;
        }
        done_2 = _S546;
        woke_3 = woke_2;
        resumed_2 = resumed_3;
    }
    thread float3 wsum_1 = float3(work_3, work_err_2, 0.0f);
    thread float3 unused_1 = float3(0.0f) ;
    group_sum3_0(tid_5, &wsum_1, &unused_1, kernelContext_26);
    if(_S517)
    {
        thread Quat_0 _S550 = (&rg_8)->rot_0;
        float4 _S551 = quat_vec_0(&_S550);
        (&isl_10)->rotation_0 = _S551;
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
        if((*kernelContext_26->g_halt_0) != 0U)
        {
            _S515 = contact_island_1;
        }
        else
        {
            _S515 = false;
        }
        if(_S515)
        {
            contact_split_at_0((&isl_10)->info_0.w + done_2, kernelContext_26);
        }
        float _S552 = wsum_1.x;
        thread float _S553 = (&isl_10)->energy_0.x;
        thread float _S554 = (&isl_10)->energy_0.y;
        comp_add1_0(&_S553, &_S554, _S552);
        (&isl_10)->energy_0.x = _S553;
        (&isl_10)->energy_0.y = _S554 + wsum_1.y;
        (&isl_10)->info_0.w = (&isl_10)->info_0.w + done_2;
        if((((&isl_10)->info_0.x) & 8U) == 0U)
        {
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) & 4294967293U;
        }
        if((*kernelContext_26->g_halt_0) != 0U)
        {
            (&isl_10)->info_0.z = ((&isl_10)->info_0.z) | 1U;
        }
        uint _S555 = seen_bits_0(kernelContext_26);
        if((*kernelContext_26->g_halt_0) != 0U)
        {
            cov_5 = 128U;
        }
        else
        {
            cov_5 = 0U;
        }
        uint _S556 = _S555 | cov_5;
        if(woke_2)
        {
            cov_5 = 2048U;
        }
        else
        {
            cov_5 = 0U;
        }
        uint _S557 = _S556 | cov_5;
        if(resumed_2)
        {
            cov_5 = 2097280U;
        }
        else
        {
            cov_5 = 0U;
        }
        uint cov_6 = _S557 | cov_5;
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
                _S515 = !quiet_3;
            }
            else
            {
                _S515 = false;
            }
            if(_S515)
            {
                cov_5 = 512U;
            }
            else
            {
                cov_5 = 0U;
            }
            uint cov_8 = cov_7 | cov_5;
            if((kernelContext_26->params_0->contact_mode_0) == 1U)
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
        Island_natural_0 device* _S558 = kernelContext_26->islands_0+gi_1;
        _S558->range_0 = packed_uint4(isl_10.range_0) ;
        _S558->info_0 = packed_uint4(isl_10.info_0) ;
        _S558->com_0 = packed_float4(isl_10.com_0) ;
        _S558->inertia0_0 = packed_float4(isl_10.inertia0_0) ;
        _S558->inertia1_0 = packed_float4(isl_10.inertia1_0) ;
        _S558->inertia2_0 = packed_float4(isl_10.inertia2_0) ;
        _S558->inv0_0 = packed_float4(isl_10.inv0_0) ;
        _S558->inv1_0 = packed_float4(isl_10.inv1_0) ;
        _S558->inv2_0 = packed_float4(isl_10.inv2_0) ;
        _S558->wcom_0 = packed_float4(isl_10.wcom_0) ;
        _S558->winv0_0 = packed_float4(isl_10.winv0_0) ;
        _S558->winv1_0 = packed_float4(isl_10.winv1_0) ;
        _S558->winv2_0 = packed_float4(isl_10.winv2_0) ;
        _S558->rotation_0 = packed_float4(isl_10.rotation_0) ;
        _S558->position_0 = packed_float4(isl_10.position_0) ;
        _S558->position_err_0 = packed_float4(isl_10.position_err_0) ;
        _S558->velocity_0 = packed_float4(isl_10.velocity_0) ;
        _S558->velocity_err_0 = packed_float4(isl_10.velocity_err_0) ;
        _S558->angular_velocity_0 = packed_float4(isl_10.angular_velocity_0) ;
        _S558->done_0 = isl_10.done_0;
        _S558->coverage_0 = isl_10.coverage_0;
        _S558->wide_groups_0 = isl_10.wide_groups_0;
        _S558->done_pad_0 = isl_10.done_pad_0;
        _S558->probes_0 = packed_uint4(isl_10.probes_0) ;
        _S558->energy_0 = packed_float4(isl_10.energy_0) ;
        _S558->rotation_err_0 = packed_float4(isl_10.rotation_err_0) ;
        _S558->momentum_0 = packed_float4(isl_10.momentum_0) ;
        _S558->momentum_err_0 = packed_float4(isl_10.momentum_err_0) ;
    }
    return;
}

uint pair_lane_count_0(KernelContext_0 thread* kernelContext_27)
{
    return max(kernelContext_27->params_0->pair_lanes_0, 1U);
}

bool stopped_0(KernelContext_0 thread* kernelContext_28)
{
    uint4 _S559 = uint4((kernelContext_28->islands_0+kernelContext_28->params_0->halt_index_0)->info_0) ;
    bool _S560;
    if(((_S559.z) & 1U) != 0U)
    {
        _S560 = true;
    }
    else
    {
        _S560 = (_S559.y) != 0U;
    }
    return _S560;
}

bool pair_any_0(bool x_9, uint lanes_0)
{
    bool _S561;
    if(lanes_0 > 1U)
    {
        bool _S562 = simd_any(x_9);
        _S561 = _S562;
    }
    else
    {
        _S561 = x_9;
    }
    return _S561;
}

struct WorldPoint_0
{
    float3 hi_3;
    float3 lo_3;
    float3 rel_0;
};

WorldPoint_0 chunk_world_0(uint c_17, KernelContext_0 thread* kernelContext_29)
{
    ChunkStatic_natural_0 device* _S563 = kernelContext_29->chunks_0+c_17;
    Island_natural_0 device* _S564 = kernelContext_29->islands_0+(uint4(_S563->info_1) ).y;
    Quat_0 q_7 = quat_of_0(float4(_S564->rotation_0) );
    thread WorldPoint_0 w_6;
    (&w_6)->hi_3 = (float4(_S564->position_0) ).xyz;
    (&w_6)->lo_3 = (float4(_S564->position_err_0) ).xyz;
    float3 _S565 = (float4(_S563->center_0) ).xyz + (float4(*(kernelContext_29->state_0+4U * c_17)) ).xyz;
    thread Quat_0 _S566 = q_7;
    float3 _S567 = rotate_0(&_S566, _S565);
    (&w_6)->rel_0 = _S567;
    return w_6;
}

float3 world_diff_0(const WorldPoint_0 thread* a_8, const WorldPoint_0 thread* b_8)
{
    return a_8->hi_3 - b_8->hi_3 + (a_8->lo_3 - b_8->lo_3) + (a_8->rel_0 - b_8->rel_0);
}

float3 safe_normalize_0(float3 v_11)
{
    float n_1 = length(v_11);
    float3 _S568;
    if(n_1 > 1.00000000317107685e-30f)
    {
        _S568 = v_11 / float3(n_1) ;
    }
    else
    {
        _S568 = float3(0.0f) ;
    }
    return _S568;
}

Quat_0 from_axis_angle_0(float3 axis_1, float angle_4)
{
    float3 a_9 = safe_normalize_0(axis_1);
    float _S569 = 0.5f * angle_4;
    float s_4 = sin(_S569);
    thread Quat_0 q_8;
    (&q_8)->w_0 = cos(_S569);
    (&q_8)->x_0 = a_9.x * s_4;
    (&q_8)->y_0 = a_9.y * s_4;
    (&q_8)->z_0 = a_9.z * s_4;
    return q_8;
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

Box_0 chunk_box_0(uint c_18, float3 center_2, KernelContext_0 thread* kernelContext_30)
{
    ChunkStatic_natural_0 device* _S570 = kernelContext_30->chunks_0+c_18;
    Quat_0 q_9 = quat_of_0(float4((kernelContext_30->islands_0+(uint4(_S570->info_1) ).y)->rotation_0) );
    float3 th_3 = (float4(*(kernelContext_30->state_0+(4U * c_18 + 1U))) ).xyz;
    Quat_0 hidden_0 = from_axis_angle_0(th_3, length(th_3));
    thread Box_0 b_9;
    (&b_9)->center_1 = center_2;
    float4 _S571 = float4(_S570->crot0_0) ;
    float4 _S572 = float4(_S570->crot1_0) ;
    float4 _S573 = float4(_S570->crot2_0) ;
    float3 _S574 = float3(_S571.x, _S572.x, _S573.x);
    thread Quat_0 _S575 = hidden_0;
    float3 _S576 = rotate_0(&_S575, _S574);
    thread Quat_0 _S577 = q_9;
    float3 _S578 = rotate_0(&_S577, _S576);
    (&b_9)->axis0_0 = _S578;
    float3 _S579 = float3(_S571.y, _S572.y, _S573.y);
    thread Quat_0 _S580 = hidden_0;
    float3 _S581 = rotate_0(&_S580, _S579);
    thread Quat_0 _S582 = q_9;
    float3 _S583 = rotate_0(&_S582, _S581);
    (&b_9)->axis1_0 = _S583;
    float3 _S584 = float3(_S571.z, _S572.z, _S573.z);
    thread Quat_0 _S585 = hidden_0;
    float3 _S586 = rotate_0(&_S585, _S584);
    thread Quat_0 _S587 = q_9;
    float3 _S588 = rotate_0(&_S587, _S586);
    (&b_9)->axis2_0 = _S588;
    (&b_9)->half_2 = (float4(_S570->half_0) ).xyz;
    float4 _S589 = float4(_S570->cmat_0) ;
    (&b_9)->hull_at_0 = (as_type<uint>((_S589.z)));
    uint _S590 = (as_type<uint>((_S589.w)));
    (&b_9)->hull_v_0 = _S590 & 255U;
    (&b_9)->hull_f_0 = _S590 >> 8U;
    return b_9;
}

uint sample_count_0(const Box_0 thread* b_10)
{
    uint _S591 = b_10->hull_v_0;
    uint _S592;
    if((b_10->hull_v_0) == 0U)
    {
        _S592 = 14U;
    }
    else
    {
        _S592 = _S591 + b_10->hull_f_0;
    }
    return _S592;
}

bool may_overlap_lanes_0(const Box_0 thread* a_10, const Box_0 thread* b_11, uint lane_0, uint lanes_1)
{
    bool separated_0 = false;
    uint i_6 = lane_0;
    for(;;)
    {
        if(i_6 < 15U)
        {
        }
        else
        {
            break;
        }
        float3 d_2 = b_11->center_1 - a_10->center_1;
        float3 _S593 = a_10->half_2;
        float3 _S594 = b_11->half_2;
        float margin_0 = 0.00000999999974738f * (length(a_10->half_2) + length(b_11->half_2));
        float3 _S595 = a_10->axis0_0;
        float3 _S596 = a_10->axis1_0;
        float3 _S597 = a_10->axis2_0;
        float3 _S598 = b_11->axis0_0;
        float3 _S599 = b_11->axis1_0;
        float3 _S600 = b_11->axis2_0;
        array<float3, int(6)> axes_0 = { { a_10->axis0_0, a_10->axis1_0, a_10->axis2_0, b_11->axis0_0, b_11->axis1_0, b_11->axis2_0 } };
        float3 l_3;
        if(i_6 < 6U)
        {
            l_3 = axes_0[i_6];
        }
        else
        {
            uint _S601 = i_6 - 6U;
            l_3 = cross(axes_0[_S601 / 3U], axes_0[3U + _S601 % 3U]);
        }
        float len_0 = length(l_3);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_6 = i_6 + lanes_1;
            continue;
        }
        bool separated_1;
        if((abs(dot(d_2, l_3))) > (_S593.x * abs(dot(_S595, l_3)) + _S593.y * abs(dot(_S596, l_3)) + _S593.z * abs(dot(_S597, l_3)) + (_S594.x * abs(dot(_S598, l_3)) + _S594.y * abs(dot(_S599, l_3)) + _S594.z * abs(dot(_S600, l_3))) + margin_0 * len_0))
        {
            separated_1 = true;
        }
        else
        {
            separated_1 = separated_0;
        }
        separated_0 = separated_1;
        i_6 = i_6 + lanes_1;
    }
    bool _S602 = pair_any_0(separated_0, lanes_1);
    return !_S602;
}

bool may_overlap_0(const Box_0 thread* a_11, const Box_0 thread* b_12)
{
    float3 _S603 = b_12->center_1 - a_11->center_1;
    float3 _S604 = a_11->half_2;
    float3 _S605 = b_12->half_2;
    float _S606 = 0.00000999999974738f * (length(a_11->half_2) + length(b_12->half_2));
    float3 _S607 = a_11->axis0_0;
    float3 _S608 = a_11->axis1_0;
    float3 _S609 = a_11->axis2_0;
    float3 _S610 = b_12->axis0_0;
    float3 _S611 = b_12->axis1_0;
    float3 _S612 = b_12->axis2_0;
    array<float3, int(6)> _S613 = { { a_11->axis0_0, a_11->axis1_0, a_11->axis2_0, b_12->axis0_0, b_12->axis1_0, b_12->axis2_0 } };
    uint i_7 = 0U;
    for(;;)
    {
        if(i_7 < 15U)
        {
        }
        else
        {
            break;
        }
        float3 l_4;
        if(i_7 < 6U)
        {
            l_4 = _S613[i_7];
        }
        else
        {
            uint _S614 = i_7 - 6U;
            l_4 = cross(_S613[_S614 / 3U], _S613[3U + _S614 % 3U]);
        }
        float len_1 = length(l_4);
        if(len_1 <= 9.99999997475242708e-07f)
        {
            i_7 = i_7 + 1U;
            continue;
        }
        if((abs(dot(_S603, l_4))) > (_S604.x * abs(dot(_S607, l_4)) + _S604.y * abs(dot(_S608, l_4)) + _S604.z * abs(dot(_S609, l_4)) + (_S605.x * abs(dot(_S610, l_4)) + _S605.y * abs(dot(_S611, l_4)) + _S605.z * abs(dot(_S612, l_4))) + _S606 * len_1))
        {
            return false;
        }
        i_7 = i_7 + 1U;
    }
    return true;
}

void chunk_velocity_0(uint c_19, float3 thread* v_12, float3 thread* w_7, KernelContext_0 thread* kernelContext_31)
{
    ChunkStatic_natural_0 device* _S615 = kernelContext_31->chunks_0+c_19;
    Island_natural_0 device* _S616 = kernelContext_31->islands_0+(uint4(_S615->info_1) ).y;
    Quat_0 q_10 = quat_of_0(float4(_S616->rotation_0) );
    uint _S617 = 4U * c_19;
    float3 _S618 = (float4(_S615->center_0) ).xyz + (float4(*(kernelContext_31->state_0+_S617)) ).xyz - (float4(_S616->com_0) ).xyz;
    thread Quat_0 _S619 = q_10;
    float3 _S620 = rotate_0(&_S619, _S618);
    float3 _S621 = (float4(_S616->angular_velocity_0) ).xyz;
    float3 _S622 = (float4(_S616->velocity_0) ).xyz + (float4(_S616->velocity_err_0) ).xyz + cross(_S621, _S620);
    float3 _S623 = (float4(*(kernelContext_31->state_0+(_S617 + 2U))) ).xyz;
    thread Quat_0 _S624 = q_10;
    float3 _S625 = rotate_0(&_S624, _S623);
    *v_12 = _S622 + _S625;
    float3 _S626 = (float4(*(kernelContext_31->state_0+(_S617 + 3U))) ).xyz;
    thread Quat_0 _S627 = q_10;
    float3 _S628 = rotate_0(&_S627, _S626);
    *w_7 = _S621 + _S628;
    return;
}

float3 box_to_world_0(const Box_0 thread* b_13, float3 local_0)
{
    return b_13->axis0_0 * float3(local_0.x)  + b_13->axis1_0 * float3(local_0.y)  + b_13->axis2_0 * float3(local_0.z) ;
}

float3 box_axis_0(const Box_0 thread* b_14, uint k_10)
{
    float3 _S629;
    if(k_10 == 0U)
    {
        _S629 = b_14->axis0_0;
    }
    else
    {
        if(k_10 == 1U)
        {
            _S629 = b_14->axis1_0;
        }
        else
        {
            _S629 = b_14->axis2_0;
        }
    }
    return _S629;
}

float comp3_0(float3 v_13, uint k_11)
{
    float _S630;
    if(k_11 == 0U)
    {
        _S630 = v_13.x;
    }
    else
    {
        if(k_11 == 1U)
        {
            _S630 = v_13.y;
        }
        else
        {
            _S630 = v_13.z;
        }
    }
    return _S630;
}

float3 sample_point_0(const Box_0 thread* b_15, uint i_8, KernelContext_0 thread* kernelContext_32)
{
    uint _S631 = b_15->hull_v_0;
    if((b_15->hull_v_0) != 0U)
    {
        float3 local_1;
        if(i_8 < _S631)
        {
            local_1 = (float4(*(kernelContext_32->loads_0+(b_15->hull_at_0 + i_8))) ).xyz * float3(0.89999997615814209f) ;
        }
        else
        {
            local_1 = (float4(*(kernelContext_32->loads_0+(b_15->hull_at_0 + _S631 + b_15->hull_f_0 + (i_8 - _S631)))) ).xyz;
        }
        float3 _S632 = b_15->center_1;
        float3 _S633 = box_to_world_0(b_15, local_1);
        return _S632 + _S633;
    }
    float sign_0;
    if(i_8 < 8U)
    {
        float3 h_0 = b_15->half_2 * float3(0.89999997615814209f) ;
        if((i_8 & 1U) == 0U)
        {
            sign_0 = - h_0.x;
        }
        else
        {
            sign_0 = h_0.x;
        }
        float _S634;
        if((i_8 & 2U) == 0U)
        {
            _S634 = - h_0.y;
        }
        else
        {
            _S634 = h_0.y;
        }
        float _S635;
        if((i_8 & 4U) == 0U)
        {
            _S635 = - h_0.z;
        }
        else
        {
            _S635 = h_0.z;
        }
        return b_15->center_1 + b_15->axis0_0 * float3(sign_0)  + b_15->axis1_0 * float3(_S634)  + b_15->axis2_0 * float3(_S635) ;
    }
    uint _S636 = i_8 - 8U;
    uint axis_2 = _S636 / 2U;
    if((_S636 % 2U) == 0U)
    {
        sign_0 = -1.0f;
    }
    else
    {
        sign_0 = 1.0f;
    }
    float3 _S637 = b_15->center_1;
    float3 _S638 = box_axis_0(b_15, axis_2);
    return _S637 + _S638 * float3((sign_0 * comp3_0(b_15->half_2, axis_2))) ;
}

float3 sample_point_1(const Box_0 thread* b_16, uint i_9, KernelContext_0 thread* kernelContext_33)
{
    uint _S639 = b_16->hull_v_0;
    if((b_16->hull_v_0) != 0U)
    {
        float3 local_2;
        if(i_9 < _S639)
        {
            local_2 = (float4(*(kernelContext_33->loads_0+(b_16->hull_at_0 + i_9))) ).xyz * float3(0.89999997615814209f) ;
        }
        else
        {
            local_2 = (float4(*(kernelContext_33->loads_0+(b_16->hull_at_0 + _S639 + b_16->hull_f_0 + (i_9 - _S639)))) ).xyz;
        }
        float3 _S640 = b_16->center_1;
        float3 _S641 = box_to_world_0(b_16, local_2);
        return _S640 + _S641;
    }
    float sign_1;
    if(i_9 < 8U)
    {
        float3 h_1 = b_16->half_2 * float3(0.89999997615814209f) ;
        if((i_9 & 1U) == 0U)
        {
            sign_1 = - h_1.x;
        }
        else
        {
            sign_1 = h_1.x;
        }
        float _S642;
        if((i_9 & 2U) == 0U)
        {
            _S642 = - h_1.y;
        }
        else
        {
            _S642 = h_1.y;
        }
        float _S643;
        if((i_9 & 4U) == 0U)
        {
            _S643 = - h_1.z;
        }
        else
        {
            _S643 = h_1.z;
        }
        return b_16->center_1 + b_16->axis0_0 * float3(sign_1)  + b_16->axis1_0 * float3(_S642)  + b_16->axis2_0 * float3(_S643) ;
    }
    uint _S644 = i_9 - 8U;
    uint axis_3 = _S644 / 2U;
    if((_S644 % 2U) == 0U)
    {
        sign_1 = -1.0f;
    }
    else
    {
        sign_1 = 1.0f;
    }
    float3 _S645 = b_16->center_1;
    float3 _S646 = box_axis_0(b_16, axis_3);
    return _S645 + _S646 * float3((sign_1 * comp3_0(b_16->half_2, axis_3))) ;
}

float3 box_to_local_0(const Box_0 thread* b_17, float3 r_8)
{
    return float3(dot(r_8, b_17->axis0_0), dot(r_8, b_17->axis1_0), dot(r_8, b_17->axis2_0));
}

float hull_signed_distance_0(const Box_0 thread* b_18, float3 local_3, uint thread* face_0, KernelContext_0 thread* kernelContext_34)
{
    *face_0 = 0U;
    float best_0 = -1.00000001504746622e+30f;
    uint f_11 = 0U;
    for(;;)
    {
        if(f_11 < (b_18->hull_f_0))
        {
        }
        else
        {
            break;
        }
        float4 _S647 = float4(*(kernelContext_34->loads_0+(b_18->hull_at_0 + b_18->hull_v_0 + f_11))) ;
        float d_3 = dot(_S647.xyz, local_3) - _S647.w;
        if(d_3 > best_0)
        {
            *face_0 = f_11;
            best_0 = d_3;
        }
        f_11 = f_11 + 1U;
    }
    return best_0;
}

bool penetration_0(const Box_0 thread* b_19, float3 p_3, float thread* depth_0, float3 thread* normal_2, KernelContext_0 thread* kernelContext_35)
{
    *depth_0 = 0.0f;
    *normal_2 = float3(0.0f) ;
    float3 r_9 = p_3 - b_19->center_1;
    float3 _S648 = b_19->half_2;
    if((dot(r_9, r_9)) > (dot(b_19->half_2, b_19->half_2) * 1.00001001358032227f))
    {
        return false;
    }
    uint _S649 = b_19->hull_v_0;
    if((b_19->hull_v_0) != 0U)
    {
        float3 _S650 = box_to_local_0(b_19, r_9);
        thread uint face_1;
        float _S651 = hull_signed_distance_0(b_19, _S650, &face_1, kernelContext_35);
        if(!(_S651 < 0.0f))
        {
            return false;
        }
        *depth_0 = - _S651;
        float3 _S652 = box_to_world_0(b_19, (float4(*(kernelContext_35->loads_0+(b_19->hull_at_0 + _S649 + face_1))) ).xyz);
        *normal_2 = _S652;
        return true;
    }
    float best_1 = 1.00000001504746622e+30f;
    uint axis_4 = 0U;
    float side_0 = 1.0f;
    uint k_12 = 0U;
    for(;;)
    {
        if(k_12 < 3U)
        {
        }
        else
        {
            break;
        }
        float3 _S653 = box_axis_0(b_19, k_12);
        float local_4 = dot(r_9, _S653);
        float d_4 = comp3_0(_S648, k_12) - abs(local_4);
        if(d_4 <= 0.0f)
        {
            return false;
        }
        if(d_4 < best_1)
        {
            float _S654;
            if(local_4 >= 0.0f)
            {
                _S654 = 1.0f;
            }
            else
            {
                _S654 = -1.0f;
            }
            best_1 = d_4;
            axis_4 = k_12;
            side_0 = _S654;
        }
        k_12 = k_12 + 1U;
    }
    *depth_0 = best_1;
    float3 _S655 = box_axis_0(b_19, axis_4);
    *normal_2 = _S655 * float3(side_0) ;
    return true;
}

bool penetration_1(const Box_0 thread* b_20, float3 p_4, float thread* depth_1, float3 thread* normal_3, KernelContext_0 thread* kernelContext_36)
{
    *depth_1 = 0.0f;
    *normal_3 = float3(0.0f) ;
    float3 r_10 = p_4 - b_20->center_1;
    float3 _S656 = b_20->half_2;
    if((dot(r_10, r_10)) > (dot(b_20->half_2, b_20->half_2) * 1.00001001358032227f))
    {
        return false;
    }
    uint _S657 = b_20->hull_v_0;
    if((b_20->hull_v_0) != 0U)
    {
        float3 _S658 = box_to_local_0(b_20, r_10);
        thread uint face_2;
        float _S659 = hull_signed_distance_0(b_20, _S658, &face_2, kernelContext_36);
        if(!(_S659 < 0.0f))
        {
            return false;
        }
        *depth_1 = - _S659;
        float3 _S660 = box_to_world_0(b_20, (float4(*(kernelContext_36->loads_0+(b_20->hull_at_0 + _S657 + face_2))) ).xyz);
        *normal_3 = _S660;
        return true;
    }
    float best_2 = 1.00000001504746622e+30f;
    uint axis_5 = 0U;
    float side_1 = 1.0f;
    uint k_13 = 0U;
    for(;;)
    {
        if(k_13 < 3U)
        {
        }
        else
        {
            break;
        }
        float3 _S661 = box_axis_0(b_20, k_13);
        float local_5 = dot(r_10, _S661);
        float d_5 = comp3_0(_S656, k_13) - abs(local_5);
        if(d_5 <= 0.0f)
        {
            return false;
        }
        if(d_5 < best_2)
        {
            float _S662;
            if(local_5 >= 0.0f)
            {
                _S662 = 1.0f;
            }
            else
            {
                _S662 = -1.0f;
            }
            best_2 = d_5;
            axis_5 = k_13;
            side_1 = _S662;
        }
        k_13 = k_13 + 1U;
    }
    *depth_1 = best_2;
    float3 _S663 = box_axis_0(b_20, axis_5);
    *normal_3 = _S663 * float3(side_1) ;
    return true;
}

bool pair_point_0(const Box_0 thread* ba_0, const Box_0 thread* bb_0, uint na_0, uint e_2, float3 thread* p_5, float3 thread* n_2, float thread* d_6, KernelContext_0 thread* kernelContext_37)
{
    if(e_2 < na_0)
    {
        float3 _S664 = sample_point_1(ba_0, e_2, kernelContext_37);
        *p_5 = _S664;
        bool _S665 = penetration_1(bb_0, _S664, d_6, n_2, kernelContext_37);
        return _S665;
    }
    float3 _S666 = sample_point_1(bb_0, e_2 - na_0, kernelContext_37);
    *p_5 = _S666;
    bool _S667 = penetration_1(ba_0, _S666, d_6, n_2, kernelContext_37);
    if(!_S667)
    {
        return false;
    }
    *n_2 = - *n_2;
    return true;
}

bool is_nan_0(float x_10)
{
    return ((as_type<uint>((x_10))) & 2147483647U) > 2139095040U;
}

uint pair_count_sum_0(uint x_11, uint lanes_2)
{
    uint _S668;
    if(lanes_2 > 1U)
    {
        uint _S669 = simd_sum(x_11);
        _S668 = _S669;
    }
    else
    {
        _S668 = x_11;
    }
    return _S668;
}

float2 half_thickness_and_area_0(const Box_0 thread* b_21, float3 d_7)
{
    uint k_14 = 0U;
    float h_2 = 0.0f;
    float area_1 = 0.0f;
    for(;;)
    {
        if(k_14 < 3U)
        {
        }
        else
        {
            break;
        }
        float3 _S670 = box_axis_0(b_21, k_14);
        float c_20 = abs(dot(d_7, _S670));
        float h_3 = h_2 + c_20 * comp3_0(b_21->half_2, k_14);
        uint _S671 = k_14 + 1U;
        float area_2 = area_1 + c_20 * 4.0f * comp3_0(b_21->half_2, _S671 % 3U) * comp3_0(b_21->half_2, (k_14 + 2U) % 3U);
        k_14 = _S671;
        h_2 = h_3;
        area_1 = area_2;
    }
    return float2(h_2, area_1);
}

float contact_stiffness_0(float ea_0, const Box_0 thread* a_12, float eb_0, const Box_0 thread* b_22, float3 dir_0)
{
    float3 d_8 = safe_normalize_0(dir_0);
    float2 _S672 = half_thickness_and_area_0(a_12, d_8);
    float2 _S673 = half_thickness_and_area_0(b_22, d_8);
    return min(_S672.y, _S673.y) / (_S672.x / ea_0 + _S673.x / eb_0);
}

float3 penalty_force_0(float k_15, float m_red_0, float friction_0, float depth_2, float3 normal_4, float3 rel_velocity_0, float dt_11, uint points_0, float thread* stored_4, float thread* dissipated_8, KernelContext_0 thread* kernelContext_38)
{
    float c_max_0 = 1.0f / max(float(points_0), 10.0f) * m_red_0 / dt_11;
    float vn_0 = dot(rel_velocity_0, normal_4);
    float _S674 = k_15 * depth_2;
    float _S675 = min(2.0f * kernelContext_38->params_0->zeta_0 * sqrt(k_15 * m_red_0), c_max_0) * vn_0;
    float _S676 = _S674 - _S675;
    float _S677 = max(_S676, 0.0f);
    float3 vt_0 = rel_velocity_0 - normal_4 * float3(vn_0) ;
    float vt_mag_0 = length(vt_0);
    float _S678 = friction_0 * _S677;
    float _S679 = min(_S678, min(c_max_0, _S678 / 0.00100000004749745f) * vt_mag_0);
    float3 ft_0;
    if(vt_mag_0 > 0.0f)
    {
        ft_0 = - vt_0 * float3((_S679 / vt_mag_0)) ;
    }
    else
    {
        ft_0 = float3(0.0f) ;
    }
    *stored_4 = 0.5f * k_15 * depth_2 * depth_2;
    float damping_power_0;
    if(_S676 > 0.0f)
    {
        damping_power_0 = _S675 * vn_0;
    }
    else
    {
        damping_power_0 = _S674 * max(vn_0, 0.0f);
    }
    *dissipated_8 = (damping_power_0 + length(ft_0) * vt_mag_0) * dt_11;
    return normal_4 * float3(_S677)  + ft_0;
}

float3 pair_sum3_0(float3 x_12, uint lanes_3)
{
    float3 _S680;
    if(lanes_3 > 1U)
    {
        float3 _S681 = simd_sum(x_12);
        _S680 = _S681;
    }
    else
    {
        _S680 = x_12;
    }
    return _S680;
}

float pair_sum_0(float x_13, uint lanes_4)
{
    float _S682;
    if(lanes_4 > 1U)
    {
        float _S683 = simd_sum(x_13);
        _S682 = _S683;
    }
    else
    {
        _S682 = x_13;
    }
    return _S682;
}

void cov_or_1(uint island_1, uint bits_1, KernelContext_0 thread* kernelContext_39)
{
    uint _S684 = atomic_fetch_or_explicit(((atomic_uint device*)(&(kernelContext_39->islands_0+island_1)->coverage_0)), bits_1, memory_order_relaxed);
    return;
}

void cov_mark_0(uint island_2, uint seen_0, uint bits_2, KernelContext_0 thread* kernelContext_40)
{
    if((bits_2 & (~seen_0)) != 0U)
    {
        cov_or_1(island_2, bits_2, kernelContext_40);
    }
    return;
}

void sentinel_mark_0(uint bits_3, KernelContext_0 thread* kernelContext_41)
{
    cov_mark_0(kernelContext_41->params_0->halt_index_0, (kernelContext_41->islands_0+kernelContext_41->params_0->halt_index_0)->coverage_0, bits_3, kernelContext_41);
    return;
}

void pair_contact_0(uint i_10, uint lane_1, uint lanes_5, KernelContext_0 thread* kernelContext_42)
{
    uint at_3 = kernelContext_42->params_0->pair_index_0 + 6U * i_10;
    uint ca_0 = kernelContext_42->index_0[at_3];
    uint cb_0 = kernelContext_42->index_0[at_3 + 1U];
    uint _S685 = kernelContext_42->index_0[at_3 + 3U];
    float _S686 = (as_type<float>((kernelContext_42->index_0[at_3 + 4U])));
    float _S687 = (as_type<float>((kernelContext_42->index_0[at_3 + 5U])));
    float _S688 = kernelContext_42->params_0->dt_0;
    uint out_0 = kernelContext_42->params_0->slot_base_0 + 2U * kernelContext_42->index_0[at_3 + 2U];
    WorldPoint_0 _S689 = chunk_world_0(ca_0, kernelContext_42);
    WorldPoint_0 _S690 = chunk_world_0(cb_0, kernelContext_42);
    thread WorldPoint_0 _S691 = _S690;
    thread WorldPoint_0 _S692 = _S689;
    float3 _S693 = world_diff_0(&_S691, &_S692);
    bool touching_0 = !((length(_S693)) > ((float4((kernelContext_42->chunks_0+ca_0)->half_0) ).w + (float4((kernelContext_42->chunks_0+cb_0)->half_0) ).w));
    float4 _S694 = float4(*(kernelContext_42->scratch_0+(kernelContext_42->params_0->ledger_base_0 + i_10))) ;
    thread float4 ledger_1 = _S694;
    uint flags_1 = (as_type<uint>((_S694.w)));
    float3 _S695 = float3(0.0f) ;
    Box_0 _S696 = chunk_box_0(ca_0, _S695, kernelContext_42);
    Box_0 _S697 = chunk_box_0(cb_0, _S693, kernelContext_42);
    thread Box_0 _S698 = _S696;
    uint _S699 = sample_count_0(&_S698);
    thread Box_0 _S700 = _S697;
    uint _S701 = sample_count_0(&_S700);
    uint _S702 = _S699 + _S701;
    bool has_state_0;
    uint e_3;
    if(!touching_0)
    {
        if((flags_1 & 1U) != 0U)
        {
            e_3 = lane_1;
            for(;;)
            {
                if(e_3 < _S702)
                {
                }
                else
                {
                    break;
                }
                *(kernelContext_42->contact_state_0+(_S685 + e_3)) = packed_float4(float4((as_type<float>((2143289344U))), 0.0f, 0.0f, 0.0f)) ;
                e_3 = e_3 + lanes_5;
            }
        }
        if(lane_1 == 0U)
        {
            if((flags_1 & 2U) != 0U)
            {
                e_3 = 0U;
                for(;;)
                {
                    if(e_3 < 4U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    *(kernelContext_42->scratch_0+(out_0 + e_3)) = packed_float4(float4(0.0f) ) ;
                    e_3 = e_3 + 1U;
                }
            }
            if(flags_1 != 0U)
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (ledger_1.x) != 0.0f;
            }
            if(has_state_0)
            {
                ledger_1.x = 0.0f;
                ledger_1.w = (as_type<float>((0U)));
                *(kernelContext_42->scratch_0+(kernelContext_42->params_0->ledger_base_0 + i_10)) = packed_float4(ledger_1) ;
            }
        }
        return;
    }
    if(lanes_5 > 1U)
    {
        thread Box_0 _S703 = _S696;
        thread Box_0 _S704 = _S697;
        bool _S705 = may_overlap_lanes_0(&_S703, &_S704, lane_1, lanes_5);
        has_state_0 = _S705;
    }
    else
    {
        thread Box_0 _S706 = _S696;
        thread Box_0 _S707 = _S697;
        bool _S708 = may_overlap_0(&_S706, &_S707);
        has_state_0 = _S708;
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
        chunk_velocity_0(ca_0, &va0_0, &wa0_0, kernelContext_42);
        thread float3 vb0_0;
        thread float3 wb0_0;
        chunk_velocity_0(cb_0, &vb0_0, &wb0_0, kernelContext_42);
        e_3 = lane_1;
        count_3 = 0U;
        uint engaged_0 = 0U;
        for(;;)
        {
            if(e_3 < _S702)
            {
            }
            else
            {
                break;
            }
            thread Box_0 _S709 = _S696;
            thread Box_0 _S710 = _S697;
            thread float3 p_6;
            thread float3 n_3;
            thread float d_9;
            bool _S711 = pair_point_0(&_S709, &_S710, _S699, e_3, &p_6, &n_3, &d_9, kernelContext_42);
            if(!_S711)
            {
                uint _S712 = _S685 + e_3;
                if(!is_nan_0((float4(*(kernelContext_42->contact_state_0+_S712)) ).x))
                {
                    *(kernelContext_42->contact_state_0+_S712) = packed_float4(float4((as_type<float>((2143289344U))), 0.0f, 0.0f, 0.0f)) ;
                }
                e_3 = e_3 + lanes_5;
                continue;
            }
            uint _S713 = count_3 + 1U;
            uint _S714 = _S685 + e_3;
            float4 _S715 = float4(*(kernelContext_42->contact_state_0+_S714)) ;
            thread float4 entry_2 = _S715;
            if(is_nan_0(_S715.x))
            {
                has_state_0 = true;
            }
            else
            {
                has_state_0 = (dot(entry_2.yzw, n_3)) < 0.99000000953674316f;
            }
            if(has_state_0)
            {
                if(d_9 > (2.0f * abs(dot(va0_0 + cross(wa0_0, p_6) - (vb0_0 + cross(wb0_0, p_6 - _S697.center_1)), n_3)) * _S688 + 9.99999971718068537e-10f))
                {
                    stored_sum_0 = d_9;
                }
                else
                {
                    stored_sum_0 = 0.0f;
                }
                entry_2 = float4(stored_sum_0, n_3);
            }
            entry_2.x = min(entry_2.x, d_9);
            *(kernelContext_42->contact_state_0+_S714) = packed_float4(entry_2) ;
            uint engaged_1;
            if((d_9 - entry_2.x) > 0.0f)
            {
                engaged_1 = engaged_0 + 1U;
            }
            else
            {
                engaged_1 = engaged_0;
            }
            count_3 = _S713;
            engaged_0 = engaged_1;
            e_3 = e_3 + lanes_5;
        }
        uint count_4 = pair_count_sum_0(count_3, lanes_5);
        uint engaged_2 = pair_count_sum_0(engaged_0, lanes_5);
        if(count_4 > 0U)
        {
            float _S716 = (float4((kernelContext_42->chunks_0+ca_0)->cmat_0) ).x;
            float _S717 = (float4((kernelContext_42->chunks_0+cb_0)->cmat_0) ).x;
            float3 _S718 = _S697.center_1 - _S696.center_1;
            thread Box_0 _S719 = _S696;
            thread Box_0 _S720 = _S697;
            float _S721 = contact_stiffness_0(_S716, &_S719, _S717, &_S720, _S718);
            float _S722 = _S721 / max(float(engaged_2), 10.0f);
            e_3 = lane_1;
            fa_0 = _S695;
            ta_1 = _S695;
            fb_0 = _S695;
            tb_1 = _S695;
            stored_sum_0 = 0.0f;
            diss_sum_0 = 0.0f;
            for(;;)
            {
                if(e_3 < _S702)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S723 = _S696;
                thread Box_0 _S724 = _S697;
                thread float3 p_7;
                thread float3 n_4;
                thread float d_10;
                bool _S725 = pair_point_0(&_S723, &_S724, _S699, e_3, &p_7, &n_4, &d_10, kernelContext_42);
                if(!_S725)
                {
                    e_3 = e_3 + lanes_5;
                    continue;
                }
                float eff_0 = d_10 - (float4(*(kernelContext_42->contact_state_0+(_S685 + e_3))) ).x;
                if(eff_0 <= 0.0f)
                {
                    e_3 = e_3 + lanes_5;
                    continue;
                }
                thread float stored_5;
                thread float diss_3;
                float3 _S726 = penalty_force_0(_S722, _S686, _S687, eff_0, n_4, va0_0 + cross(wa0_0, p_7) - (vb0_0 + cross(wb0_0, p_7 - _S697.center_1)), _S688, engaged_2, &stored_5, &diss_3, kernelContext_42);
                float3 ta_2 = ta_1 + cross(p_7, _S726);
                float3 _S727 = - _S726;
                float3 fb_1 = fb_0 + _S727;
                float3 tb_2 = tb_1 + cross(p_7 - _S697.center_1, _S727);
                float stored_sum_1 = stored_sum_0 + stored_5;
                float diss_sum_1 = diss_sum_0 + diss_3;
                fa_0 = fa_0 + _S726;
                ta_1 = ta_2;
                fb_0 = fb_1;
                tb_1 = tb_2;
                stored_sum_0 = stored_sum_1;
                diss_sum_0 = diss_sum_1;
                e_3 = e_3 + lanes_5;
            }
            float3 _S728 = pair_sum3_0(fa_0, lanes_5);
            float3 _S729 = pair_sum3_0(ta_1, lanes_5);
            float3 _S730 = pair_sum3_0(fb_0, lanes_5);
            float3 _S731 = pair_sum3_0(tb_1, lanes_5);
            float _S732 = pair_sum_0(stored_sum_0, lanes_5);
            float _S733 = pair_sum_0(diss_sum_0, lanes_5);
            has_state_0 = true;
            diss_sum_0 = _S733;
            fa_0 = _S728;
            ta_1 = _S729;
            fb_0 = _S730;
            tb_1 = _S731;
            stored_sum_0 = _S732;
        }
        else
        {
            has_state_0 = false;
            diss_sum_0 = 0.0f;
            fa_0 = _S695;
            ta_1 = _S695;
            fb_0 = _S695;
            tb_1 = _S695;
            stored_sum_0 = 0.0f;
        }
    }
    else
    {
        has_state_0 = false;
        diss_sum_0 = 0.0f;
        fa_0 = _S695;
        ta_1 = _S695;
        fb_0 = _S695;
        tb_1 = _S695;
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
        e_3 = lane_1;
        for(;;)
        {
            if(e_3 < _S702)
            {
            }
            else
            {
                break;
            }
            *(kernelContext_42->contact_state_0+(_S685 + e_3)) = packed_float4(float4((as_type<float>((2143289344U))), 0.0f, 0.0f, 0.0f)) ;
            e_3 = e_3 + lanes_5;
        }
    }
    if(lane_1 != 0U)
    {
        return;
    }
    if(has_state_0)
    {
        thread float _S734 = ledger_1.y;
        thread float _S735 = ledger_1.z;
        comp_add1_0(&_S734, &_S735, diss_sum_0);
        ledger_1.z = _S735;
        ledger_1.y = _S734;
    }
    float3 _S736 = float3(0.0f) ;
    if(any(fa_0 != _S736))
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(ta_1 != _S736);
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(fb_0 != _S736);
    }
    if(loaded_0)
    {
        loaded_0 = true;
    }
    else
    {
        loaded_0 = any(tb_1 != _S736);
    }
    bool _S737;
    if(loaded_0)
    {
        _S737 = true;
    }
    else
    {
        _S737 = (flags_1 & 2U) != 0U;
    }
    if(_S737)
    {
        *(kernelContext_42->scratch_0+out_0) = packed_float4(float4(fa_0, 0.0f)) ;
        *(kernelContext_42->scratch_0+(out_0 + 1U)) = packed_float4(float4(ta_1, 0.0f)) ;
        *(kernelContext_42->scratch_0+(out_0 + 2U)) = packed_float4(float4(fb_0, 0.0f)) ;
        *(kernelContext_42->scratch_0+(out_0 + 3U)) = packed_float4(float4(tb_1, 0.0f)) ;
    }
    ledger_1.x = stored_sum_0;
    if(has_state_0)
    {
        e_3 = 1U;
    }
    else
    {
        e_3 = 0U;
    }
    if(loaded_0)
    {
        count_3 = 2U;
    }
    else
    {
        count_3 = 0U;
    }
    ledger_1.w = (as_type<float>((e_3 | count_3)));
    *(kernelContext_42->scratch_0+(kernelContext_42->params_0->ledger_base_0 + i_10)) = packed_float4(ledger_1) ;
    if(loaded_0)
    {
        sentinel_mark_0(1U, kernelContext_42);
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
    thread Box_0 b_23;
    (&b_23)->center_1 = center_3;
    float3 _S738 = float3(1.0f, 0.0f, 0.0f);
    thread Quat_0 _S739 = q_11;
    float3 _S740 = rotate_0(&_S739, _S738);
    (&b_23)->axis0_0 = _S740;
    float3 _S741 = float3(0.0f, 1.0f, 0.0f);
    thread Quat_0 _S742 = q_11;
    float3 _S743 = rotate_0(&_S742, _S741);
    (&b_23)->axis1_0 = _S743;
    float3 _S744 = float3(0.0f, 0.0f, 1.0f);
    thread Quat_0 _S745 = q_11;
    float3 _S746 = rotate_0(&_S745, _S744);
    (&b_23)->axis2_0 = _S746;
    (&b_23)->half_2 = half_3;
    (&b_23)->hull_at_0 = 0U;
    (&b_23)->hull_v_0 = 0U;
    (&b_23)->hull_f_0 = 0U;
    return b_23;
}

uint impactor_slots_0(const Impactor_natural_0 thread* imp_1, const Box_0 thread* b_24)
{
    uint _S747;
    if(((float4(imp_1->shape_0) ).x) == 0.0f)
    {
        _S747 = 1U;
    }
    else
    {
        uint _S748 = sample_count_0(b_24);
        _S747 = _S748 + 14U;
    }
    return _S747;
}

uint impactor_slots_1(const Impactor_natural_0 thread* imp_2, const Box_0 thread* b_25)
{
    uint _S749;
    if(((float4(imp_2->shape_0) ).x) == 0.0f)
    {
        _S749 = 1U;
    }
    else
    {
        uint _S750 = sample_count_0(b_25);
        _S749 = _S750 + 14U;
    }
    return _S749;
}

bool sphere_contact_0(const Box_0 thread* b_26, float3 center_4, float radius_0, float3 thread* point_0, float3 thread* normal_5, float thread* depth_3, KernelContext_0 thread* kernelContext_43)
{
    float3 _S751 = float3(0.0f) ;
    *point_0 = _S751;
    *normal_5 = _S751;
    *depth_3 = 0.0f;
    float3 _S752 = b_26->center_1;
    float3 r_11 = center_4 - b_26->center_1;
    float3 _S753 = b_26->axis0_0;
    float3 _S754 = b_26->axis1_0;
    float3 _S755 = b_26->axis2_0;
    float3 local_6 = float3(dot(r_11, b_26->axis0_0), dot(r_11, b_26->axis1_0), dot(r_11, b_26->axis2_0));
    uint _S756 = b_26->hull_v_0;
    if((b_26->hull_v_0) != 0U)
    {
        thread uint face_3;
        float _S757 = hull_signed_distance_0(b_26, local_6, &face_3, kernelContext_43);
        if(_S757 >= radius_0)
        {
            return false;
        }
        float3 _S758 = box_to_world_0(b_26, (float4(*(kernelContext_43->loads_0+(b_26->hull_at_0 + _S756 + face_3))) ).xyz);
        *normal_5 = _S758;
        *point_0 = center_4 - _S758 * float3(max(_S757, 0.0f)) ;
        *depth_3 = radius_0 - _S757;
        return true;
    }
    float3 q_12 = clamp(local_6, - b_26->half_2, b_26->half_2);
    float3 d_11 = local_6 - q_12;
    float dist_0 = length(d_11);
    if(dist_0 > 9.999999960041972e-13f)
    {
        if(dist_0 >= radius_0)
        {
            return false;
        }
        float3 dn_0 = d_11 / float3(dist_0) ;
        *normal_5 = _S753 * float3(dn_0.x)  + _S754 * float3(dn_0.y)  + _S755 * float3(dn_0.z) ;
        *point_0 = _S752 + _S753 * float3(q_12.x)  + _S754 * float3(q_12.y)  + _S755 * float3(q_12.z) ;
        *depth_3 = radius_0 - dist_0;
        return true;
    }
    thread float inside_0;
    thread float3 n_5;
    bool _S759 = penetration_0(b_26, center_4, &inside_0, &n_5, kernelContext_43);
    if(!_S759)
    {
        return false;
    }
    *normal_5 = n_5;
    *point_0 = center_4 - n_5 * float3(min(radius_0, inside_0)) ;
    *depth_3 = radius_0 + inside_0;
    return true;
}

bool impactor_contact_0(const Impactor_natural_0 thread* imp_3, float crush_depth_0, const Box_0 thread* shrunk_0, const Box_0 thread* b_27, uint j_2, float3 thread* p_8, float3 thread* n_6, float thread* d_12, KernelContext_0 thread* kernelContext_44)
{
    float3 _S760 = float3(0.0f) ;
    *p_8 = _S760;
    *n_6 = _S760;
    *d_12 = 0.0f;
    float4 _S761 = float4(imp_3->shape_0) ;
    if((_S761.x) == 0.0f)
    {
        bool _S762 = sphere_contact_0(b_27, _S760, _S761.y - crush_depth_0, p_8, n_6, d_12, kernelContext_44);
        if(!_S762)
        {
            return false;
        }
        *n_6 = - *n_6;
        return true;
    }
    uint _S763 = sample_count_0(b_27);
    if(j_2 < _S763)
    {
        float3 _S764 = sample_point_0(b_27, j_2, kernelContext_44);
        *p_8 = _S764;
        bool _S765 = penetration_0(shrunk_0, _S764, d_12, n_6, kernelContext_44);
        return _S765;
    }
    float3 _S766 = sample_point_0(shrunk_0, j_2 - _S763, kernelContext_44);
    *p_8 = _S766;
    bool _S767 = penetration_0(b_27, _S766, d_12, n_6, kernelContext_44);
    if(!_S767)
    {
        return false;
    }
    *n_6 = - *n_6;
    return true;
}

bool impactor_contact_1(const Impactor_natural_0 thread* imp_4, float crush_depth_1, const Box_0 thread* shrunk_1, const Box_0 thread* b_28, uint j_3, float3 thread* p_9, float3 thread* n_7, float thread* d_13, KernelContext_0 thread* kernelContext_45)
{
    float3 _S768 = float3(0.0f) ;
    *p_9 = _S768;
    *n_7 = _S768;
    *d_13 = 0.0f;
    float4 _S769 = float4(imp_4->shape_0) ;
    if((_S769.x) == 0.0f)
    {
        bool _S770 = sphere_contact_0(b_28, _S768, _S769.y - crush_depth_1, p_9, n_7, d_13, kernelContext_45);
        if(!_S770)
        {
            return false;
        }
        *n_7 = - *n_7;
        return true;
    }
    uint _S771 = sample_count_0(b_28);
    if(j_3 < _S771)
    {
        float3 _S772 = sample_point_0(b_28, j_3, kernelContext_45);
        *p_9 = _S772;
        bool _S773 = penetration_0(shrunk_1, _S772, d_13, n_7, kernelContext_45);
        return _S773;
    }
    float3 _S774 = sample_point_0(shrunk_1, j_3 - _S771, kernelContext_45);
    *p_9 = _S774;
    bool _S775 = penetration_0(b_28, _S774, d_13, n_7, kernelContext_45);
    if(!_S775)
    {
        return false;
    }
    *n_7 = - *n_7;
    return true;
}

WorldPoint_0 impactor_point_0(uint _S776, KernelContext_0 thread* kernelContext_46)
{
    Impactor_natural_0 device* _S777 = kernelContext_46->impactors_0+_S776;
    thread WorldPoint_0 wi_0;
    (&wi_0)->hi_3 = (float4(_S777->position_1) ).xyz;
    (&wi_0)->lo_3 = (float4(_S777->position_err_1) ).xyz;
    (&wi_0)->rel_0 = float3(0.0f) ;
    return wi_0;
}

Box_0 impactor_box_1(uint _S778, float3 _S779, float3 _S780, KernelContext_0 thread* kernelContext_47)
{
    Quat_0 q_13 = quat_of_0(float4((kernelContext_47->impactors_0+_S778)->rotation_1) );
    thread Box_0 b_29;
    (&b_29)->center_1 = _S779;
    float3 _S781 = float3(1.0f, 0.0f, 0.0f);
    thread Quat_0 _S782 = q_13;
    float3 _S783 = rotate_0(&_S782, _S781);
    (&b_29)->axis0_0 = _S783;
    float3 _S784 = float3(0.0f, 1.0f, 0.0f);
    thread Quat_0 _S785 = q_13;
    float3 _S786 = rotate_0(&_S785, _S784);
    (&b_29)->axis1_0 = _S786;
    float3 _S787 = float3(0.0f, 0.0f, 1.0f);
    thread Quat_0 _S788 = q_13;
    float3 _S789 = rotate_0(&_S788, _S787);
    (&b_29)->axis2_0 = _S789;
    (&b_29)->half_2 = _S780;
    (&b_29)->hull_at_0 = 0U;
    (&b_29)->hull_v_0 = 0U;
    (&b_29)->hull_f_0 = 0U;
    return b_29;
}

Box_0 impactor_shrunk_0(uint _S790, float _S791, const Box_0 thread* _S792)
{
    thread Box_0 shrunk_2 = *_S792;
    (&shrunk_2)->half_2 = _S792->half_2 - min(float3(_S791) , _S792->half_2 * float3(0.5f) );
    return shrunk_2;
}

bool impactor_contact_2(uint _S793, float _S794, const Box_0 thread* _S795, const Box_0 thread* _S796, uint _S797, float3 thread* _S798, float3 thread* _S799, float thread* _S800, KernelContext_0 thread* kernelContext_48)
{
    Impactor_natural_0 _S801 = *(kernelContext_48->impactors_0+_S793);
    float3 _S802 = float3(0.0f) ;
    *_S798 = _S802;
    *_S799 = _S802;
    *_S800 = 0.0f;
    float4 _S803 = float4(_S801.shape_0) ;
    if((_S803.x) == 0.0f)
    {
        bool _S804 = sphere_contact_0(_S796, _S802, _S803.y - _S794, _S798, _S799, _S800, kernelContext_48);
        if(!_S804)
        {
            return false;
        }
        *_S799 = - *_S799;
        return true;
    }
    uint _S805 = sample_count_0(_S796);
    if(_S797 < _S805)
    {
        float3 _S806 = sample_point_0(_S796, _S797, kernelContext_48);
        *_S798 = _S806;
        bool _S807 = penetration_0(_S795, _S806, _S800, _S799, kernelContext_48);
        return _S807;
    }
    float3 _S808 = sample_point_0(_S795, _S797 - _S805, kernelContext_48);
    *_S798 = _S808;
    bool _S809 = penetration_0(_S796, _S808, _S800, _S799, kernelContext_48);
    if(!_S809)
    {
        return false;
    }
    *_S799 = - *_S799;
    return true;
}

void impactor_candidate_forces_0(uint k_16, uint lane_2, uint lanes_6, KernelContext_0 thread* kernelContext_49)
{
    uint _S810;
    bool _S811;
    uint _S812 = 3U * k_16;
    uint at_4 = kernelContext_49->params_0->cand_index_0 + _S812;
    uint c_21 = kernelContext_49->index_0[at_4];
    uint slot_1 = kernelContext_49->index_0[at_4 + 1U];
    uint _S813 = kernelContext_49->index_0[at_4 + 2U];
    thread Impactor_natural_0 _S814 = *(kernelContext_49->impactors_0+_S813);
    float _S815 = kernelContext_49->params_0->dt_0;
    float3 _S816 = float3(0.0f) ;
    thread float4 data_0 = float4(*(kernelContext_49->scratch_0+(kernelContext_49->params_0->cand_base_0 + _S812))) ;
    bool hit_0;
    float3 f_sum_0;
    float3 t_sum_0;
    float3 imp_f_0;
    float3 imp_t_0;
    if(((uint4((&_S814)->cand_0) ).z) == 0U)
    {
        WorldPoint_0 _S817 = chunk_world_0(c_21, kernelContext_49);
        WorldPoint_0 _S818 = impactor_point_0(_S813, kernelContext_49);
        thread WorldPoint_0 _S819 = _S817;
        thread WorldPoint_0 _S820 = _S818;
        float3 _S821 = world_diff_0(&_S819, &_S820);
        float4 _S822 = float4((&_S814)->half_1) ;
        if(!((length(_S821)) > (_S822.w + (float4((kernelContext_49->chunks_0+c_21)->half_0) ).w)))
        {
            Box_0 _S823 = impactor_box_1(_S813, _S816, _S822.xyz, kernelContext_49);
            Box_0 _S824 = chunk_box_0(c_21, _S821, kernelContext_49);
            float4 _S825 = float4((&_S814)->mat_0) ;
            float _S826 = _S825.x;
            float _S827 = (float4((kernelContext_49->chunks_0+c_21)->cmat_0) ).x;
            float3 _S828 = _S824.center_1 - _S823.center_1;
            thread Box_0 _S829 = _S823;
            thread Box_0 _S830 = _S824;
            float _S831 = contact_stiffness_0(_S826, &_S829, _S827, &_S830, _S828);
            float4 _S832 = float4((&_S814)->geom_0) ;
            float _S833 = _S832.y;
            thread Box_0 _S834 = _S823;
            Box_0 _S835 = impactor_shrunk_0(_S813, _S833, &_S834);
            uint j_4 = lane_2;
            uint n_8 = 0U;
            for(;;)
            {
                thread Box_0 _S836 = _S824;
                uint _S837 = impactor_slots_0(&_S814, &_S836);
                _S810 = _S837;
                if(j_4 < _S837)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S838 = _S835;
                thread Box_0 _S839 = _S824;
                thread float3 p_10;
                thread float3 nrm_0;
                thread float dep_0;
                bool _S840 = impactor_contact_2(_S813, _S833, &_S838, &_S839, j_4, &p_10, &nrm_0, &dep_0, kernelContext_49);
                if(_S840)
                {
                    n_8 = n_8 + 1U;
                }
                j_4 = j_4 + lanes_6;
            }
            uint n_9 = pair_count_sum_0(n_8, lanes_6);
            bool _S841 = ((float4((&_S814)->shape_0) ).x) == 0.0f;
            float _S842;
            if(_S841)
            {
                _S842 = _S831;
            }
            else
            {
                _S842 = _S831 / max(float(n_9), 10.0f);
            }
            if(_S841)
            {
                j_4 = 1U;
            }
            else
            {
                j_4 = n_9;
            }
            float m_4 = (float4((kernelContext_49->chunks_0+c_21)->center_0) ).w;
            float _S843 = _S825.z;
            float _S844 = m_4 * _S843 / (m_4 + _S843);
            float _S845;
            if((kernelContext_49->params_0->pair_friction_0) >= 0.0f)
            {
                _S845 = kernelContext_49->params_0->pair_friction_0;
            }
            else
            {
                _S845 = min(_S825.y, (float4((kernelContext_49->chunks_0+c_21)->cmat_0) ).y);
            }
            float3 _S846 = (float4((&_S814)->velocity_1) ).xyz + (float4((&_S814)->velocity_err_1) ).xyz;
            thread float3 vc_0;
            thread float3 wc_0;
            chunk_velocity_0(c_21, &vc_0, &wc_0, kernelContext_49);
            hit_0 = false;
            uint j_5 = lane_2;
            f_sum_0 = _S816;
            t_sum_0 = _S816;
            imp_f_0 = _S816;
            imp_t_0 = _S816;
            float diss_sum_2 = 0.0f;
            for(;;)
            {
                bool _S847 = n_9 > 0U;
                _S811 = _S847;
                bool _S848;
                if(_S847)
                {
                    _S848 = j_5 < _S810;
                }
                else
                {
                    _S848 = false;
                }
                if(_S848)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S849 = _S835;
                thread Box_0 _S850 = _S824;
                thread float3 p_11;
                thread float3 nrm_1;
                thread float dep_1;
                bool _S851 = impactor_contact_0(&_S814, _S833, &_S849, &_S850, j_5, &p_11, &nrm_1, &dep_1, kernelContext_49);
                if(!_S851)
                {
                    j_5 = j_5 + lanes_6;
                    continue;
                }
                thread float stored_6;
                thread float diss_4;
                float3 _S852 = penalty_force_0(_S842, _S844, _S845, dep_1 * _S832.x, nrm_1, vc_0 + cross(wc_0, p_11 - _S824.center_1) - (_S846 + cross((float4((&_S814)->angular_velocity_1) ).xyz, p_11)), _S815, j_4, &stored_6, &diss_4, kernelContext_49);
                float3 f_sum_1 = f_sum_0 + _S852;
                float3 t_sum_1 = t_sum_0 + cross(p_11 - _S824.center_1, _S852);
                float3 imp_f_1 = imp_f_0 - _S852;
                float3 imp_t_1 = imp_t_0 - cross(p_11, _S852);
                float diss_sum_3 = diss_sum_2 + diss_4;
                hit_0 = true;
                f_sum_0 = f_sum_1;
                t_sum_0 = t_sum_1;
                imp_f_0 = imp_f_1;
                imp_t_0 = imp_t_1;
                diss_sum_2 = diss_sum_3;
                j_5 = j_5 + lanes_6;
            }
            float3 _S853 = pair_sum3_0(f_sum_0, lanes_6);
            float3 _S854 = pair_sum3_0(t_sum_0, lanes_6);
            float3 _S855 = pair_sum3_0(imp_f_0, lanes_6);
            float3 _S856 = pair_sum3_0(imp_t_0, lanes_6);
            float diss_sum_4 = pair_sum_0(diss_sum_2, lanes_6);
            bool _S857 = pair_any_0(hit_0, lanes_6);
            if(_S811)
            {
                thread float _S858 = data_0.z;
                thread float _S859 = data_0.w;
                comp_add1_0(&_S858, &_S859, diss_sum_4);
                data_0.w = _S859;
                data_0.z = _S858;
            }
            hit_0 = _S857;
            f_sum_0 = _S853;
            t_sum_0 = _S854;
            imp_f_0 = _S855;
            imp_t_0 = _S856;
        }
        else
        {
            hit_0 = false;
            f_sum_0 = _S816;
            t_sum_0 = _S816;
            imp_f_0 = _S816;
            imp_t_0 = _S816;
        }
    }
    else
    {
        hit_0 = false;
        f_sum_0 = _S816;
        t_sum_0 = _S816;
        imp_f_0 = _S816;
        imp_t_0 = _S816;
    }
    if(lane_2 != 0U)
    {
        return;
    }
    uint _S860 = 2U * slot_1;
    *(kernelContext_49->scratch_0+(kernelContext_49->params_0->slot_base_0 + _S860)) = packed_float4(float4(f_sum_0, 0.0f)) ;
    *(kernelContext_49->scratch_0+(kernelContext_49->params_0->slot_base_0 + _S860 + 1U)) = packed_float4(float4(t_sum_0, 0.0f)) ;
    *(kernelContext_49->scratch_0+(kernelContext_49->params_0->cand_base_0 + _S812)) = packed_float4(data_0) ;
    *(kernelContext_49->scratch_0+(kernelContext_49->params_0->cand_base_0 + _S812 + 1U)) = packed_float4(float4(imp_f_0, 0.0f)) ;
    *(kernelContext_49->scratch_0+(kernelContext_49->params_0->cand_base_0 + _S812 + 2U)) = packed_float4(float4(imp_t_0, 0.0f)) ;
    if(hit_0)
    {
        sentinel_mark_0(2U, kernelContext_49);
    }
    return;
}

void travel_check_0(uint c_22, KernelContext_0 thread* kernelContext_50)
{
    ChunkStatic_natural_0 device* _S861 = kernelContext_50->chunks_0+c_22;
    uint4 _S862 = uint4(_S861->cinfo_0) ;
    if((_S862.z) == 0U)
    {
        return;
    }
    uint _S863 = 4U * c_22;
    uint at_5 = kernelContext_50->params_0->cplan_base_0 + _S863;
    float4 _S864 = float4(*(kernelContext_50->scratch_0+at_5)) ;
    float4 _S865 = float4(*(kernelContext_50->scratch_0+(at_5 + 1U))) ;
    float4 _S866 = float4(*(kernelContext_50->scratch_0+(at_5 + 2U))) ;
    float4 _S867 = float4(*(kernelContext_50->scratch_0+(at_5 + 3U))) ;
    WorldPoint_0 _S868 = chunk_world_0(c_22, kernelContext_50);
    float3 moved_0 = _S868.hi_3 - _S864.xyz + (_S868.lo_3 - _S865.xyz) + _S868.rel_0;
    uint _S869 = (uint4(_S861->info_1) ).y;
    float3 _S870 = float3((kernelContext_50->params_0->t_hi_0 - _S867.w + (kernelContext_50->params_0->t_lo_0 - _S865.w) + float((uint4((kernelContext_50->islands_0+_S869)->info_0) ).w - kernelContext_50->params_0->step_start_0) * kernelContext_50->params_0->dt_0)) ;
    float3 _S871 = _S867.xyz * _S870;
    uint _S872 = _S862.w;
    float3 _S873;
    if((_S872 & 2U) != 0U)
    {
        _S873 = float3(0.5f)  * kernelContext_50->params_0->gravity_0.xyz * _S870 * _S870;
    }
    else
    {
        _S873 = float3(0.0f) ;
    }
    float travel_0 = length(moved_0 - (_S871 + _S873));
    float travel_1;
    if((_S872 & 1U) != 0U)
    {
        float3 th_4 = (float4(*(kernelContext_50->state_0+(_S863 + 1U))) ).xyz;
        Quat_0 _S874 = from_axis_angle_0(th_4, length(th_4));
        thread Quat_0 _S875 = quat_of_0(float4((kernelContext_50->islands_0+_S869)->rotation_0) );
        thread Quat_0 _S876 = _S874;
        Quat_0 _S877 = quat_mul_0(&_S875, &_S876);
        thread Quat_0 _S878 = quat_of_0(float4(- _S866.xyz, _S866.w));
        thread Quat_0 _S879 = _S877;
        Quat_0 _S880 = quat_mul_0(&_S878, &_S879);
        travel_1 = travel_0 + 2.0f * atan2(length(float3(_S880.x_0, _S880.y_0, _S880.z_0)), abs(_S880.w_0)) * (float4(_S861->half_0) ).w;
    }
    else
    {
        travel_1 = travel_0;
    }
    if(travel_1 > (_S864.w))
    {
        (kernelContext_50->islands_0+kernelContext_50->params_0->halt_index_0)->info_0[int(2)] = ((uint4((kernelContext_50->islands_0+kernelContext_50->params_0->halt_index_0)->info_0) ).z) | 1U;
        sentinel_mark_0(1048576U, kernelContext_50);
    }
    return;
}

void impactor_travel_check_0(uint ii_0, KernelContext_0 thread* kernelContext_51)
{
    Impactor_natural_0 device* _S881 = kernelContext_51->impactors_0+ii_0;
    uint at_6 = kernelContext_51->params_0->iplan_base_0 + 4U * ii_0;
    float4 _S882 = float4(*(kernelContext_51->scratch_0+at_6)) ;
    uint4 _S883 = uint4(_S881->cand_0) ;
    bool _S884;
    if((_S883.z) != 0U)
    {
        _S884 = true;
    }
    else
    {
        _S884 = (_S882.w) <= 0.0f;
    }
    if(_S884)
    {
        return;
    }
    float4 _S885 = float4(*(kernelContext_51->scratch_0+(at_6 + 1U))) ;
    float4 _S886 = float4(*(kernelContext_51->scratch_0+(at_6 + 2U))) ;
    float4 _S887 = float4(*(kernelContext_51->scratch_0+(at_6 + 3U))) ;
    float3 moved_1 = (float4(_S881->position_1) ).xyz - _S882.xyz + ((float4(_S881->position_err_1) ).xyz - _S885.xyz);
    float3 _S888 = float3((kernelContext_51->params_0->t_hi_0 - _S887.w + (kernelContext_51->params_0->t_lo_0 - _S885.w) + float(_S883.w - kernelContext_51->params_0->step_start_0) * kernelContext_51->params_0->dt_0)) ;
    float3 expected_0 = _S887.xyz * _S888 + float3(0.5f)  * kernelContext_51->params_0->gravity_0.xyz * _S888 * _S888;
    Quat_0 _S889 = quat_of_0(float4(_S881->rotation_1) );
    thread Quat_0 _S890 = quat_of_0(float4(- _S886.xyz, _S886.w));
    thread Quat_0 _S891 = _S889;
    Quat_0 _S892 = quat_mul_0(&_S890, &_S891);
    if((length(moved_1 - expected_0) + 2.0f * atan2(length(float3(_S892.x_0, _S892.y_0, _S892.z_0)), abs(_S892.w_0)) * (float4(_S881->half_1) ).w) > (_S882.w))
    {
        (kernelContext_51->islands_0+kernelContext_51->params_0->halt_index_0)->info_0[int(2)] = ((uint4((kernelContext_51->islands_0+kernelContext_51->params_0->halt_index_0)->info_0) ).z) | 1U;
        sentinel_mark_0(1048576U, kernelContext_51);
    }
    return;
}

[[kernel]] void contact_forces(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* params_1 [[buffer(0)]], Island_natural_0 device* islands_1 [[buffer(9)]], uint device* index_1 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(3)]], packed_float4 device* state_3 [[buffer(6)]], packed_float4 device* scratch_1 [[buffer(8)]], packed_float4 device* contact_state_1 [[buffer(11)]], packed_float4 device* loads_1 [[buffer(5)]], Impactor_natural_0 device* impactors_1 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(7)]], BondStatic_natural_0 device* bonds_1 [[buffer(2)]], MaterialTable_0 constant* materials_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_52;
    (&kernelContext_52)->params_0 = params_1;
    (&kernelContext_52)->islands_0 = islands_1;
    (&kernelContext_52)->index_0 = index_1;
    (&kernelContext_52)->chunks_0 = chunks_1;
    (&kernelContext_52)->state_0 = state_3;
    (&kernelContext_52)->scratch_0 = scratch_1;
    (&kernelContext_52)->contact_state_0 = contact_state_1;
    (&kernelContext_52)->loads_0 = loads_1;
    (&kernelContext_52)->impactors_0 = impactors_1;
    (&kernelContext_52)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_52)->bonds_0 = bonds_1;
    (&kernelContext_52)->materials_0 = materials_1;
    threadgroup array<float4, int(32)> g_part_a_1;
    (&kernelContext_52)->g_part_a_0 = &g_part_a_1;
    threadgroup array<float4, int(32)> g_part_b_1;
    (&kernelContext_52)->g_part_b_0 = &g_part_b_1;
    threadgroup array<uint, int(5)> g_seen_1;
    (&kernelContext_52)->g_seen_0 = &g_seen_1;
    threadgroup uint g_wide_run_1;
    (&kernelContext_52)->g_wide_run_0 = &g_wide_run_1;
    threadgroup array<uint, int(4096)> g_label_1;
    (&kernelContext_52)->g_label_0 = &g_label_1;
    threadgroup uint g_changed_1;
    (&kernelContext_52)->g_changed_0 = &g_changed_1;
    threadgroup uint g_halt_1;
    (&kernelContext_52)->g_halt_0 = &g_halt_1;
    threadgroup uint g_run_1;
    (&kernelContext_52)->g_run_0 = &g_run_1;
    threadgroup array<uint, int(512)> g_label_narrow_1;
    (&kernelContext_52)->g_label_narrow_0 = &g_label_narrow_1;
    threadgroup uint g_changed_narrow_1;
    (&kernelContext_52)->g_changed_narrow_0 = &g_changed_narrow_1;
    uint i_11 = id_0.x;
    uint _S893 = pair_lane_count_0(&kernelContext_52);
    uint pair_threads_0 = (&kernelContext_52)->params_0->pair_count_0 * _S893;
    if(i_11 < pair_threads_0)
    {
        bool _S894 = stopped_0(&kernelContext_52);
        bool _S895 = pair_any_0(_S894, _S893);
        if(!_S895)
        {
            uint _S896 = i_11 / _S893;
            uint _S897 = i_11 % _S893;
            pair_contact_0(_S896, _S897, _S893, &kernelContext_52);
        }
        return;
    }
    uint j_6 = i_11 - pair_threads_0;
    uint cand_threads_0 = (&kernelContext_52)->params_0->cand_count_0 * _S893;
    if(j_6 < cand_threads_0)
    {
        bool _S898 = stopped_0(&kernelContext_52);
        bool _S899 = pair_any_0(_S898, _S893);
        if(!_S899)
        {
            uint _S900 = j_6 / _S893;
            uint _S901 = j_6 % _S893;
            impactor_candidate_forces_0(_S900, _S901, _S893, &kernelContext_52);
        }
        return;
    }
    uint _S902 = j_6 - cand_threads_0;
    bool _S903;
    if(_S902 < ((&kernelContext_52)->params_0->chunk_count_0))
    {
        bool _S904 = stopped_0(&kernelContext_52);
        _S903 = !_S904;
    }
    else
    {
        _S903 = false;
    }
    if(_S903)
    {
        travel_check_0(_S902, &kernelContext_52);
    }
    else
    {
        if((_S902 - (&kernelContext_52)->params_0->chunk_count_0) < ((&kernelContext_52)->params_0->impactor_count_0))
        {
            bool _S905 = stopped_0(&kernelContext_52);
            _S903 = !_S905;
        }
        else
        {
            _S903 = false;
        }
        if(_S903)
        {
            impactor_travel_check_0(_S902 - (&kernelContext_52)->params_0->chunk_count_0, &kernelContext_52);
        }
    }
    return;
}

uint impactor_contact_count_0(uint _S906, float _S907, const Box_0 thread* _S908, const Box_0 thread* _S909, KernelContext_0 thread* kernelContext_53)
{
    thread Impactor_natural_0 _S910 = *(kernelContext_53->impactors_0+_S906);
    uint j_7 = 0U;
    uint count_5 = 0U;
    for(;;)
    {
        uint _S911 = impactor_slots_1(&_S910, _S909);
        if(j_7 < _S911)
        {
        }
        else
        {
            break;
        }
        thread float3 p_12;
        thread float3 n_10;
        thread float d_14;
        bool _S912 = impactor_contact_2(_S906, _S907, _S908, _S909, j_7, &p_12, &n_10, &d_14, kernelContext_53);
        if(_S912)
        {
            count_5 = count_5 + 1U;
        }
        j_7 = j_7 + 1U;
    }
    return count_5;
}

[[kernel]] void impactor_shares(uint3 id_1 [[thread_position_in_grid]], Params_0 constant* params_2 [[buffer(0)]], Island_natural_0 device* islands_2 [[buffer(9)]], uint device* index_2 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(3)]], packed_float4 device* state_4 [[buffer(6)]], packed_float4 device* scratch_2 [[buffer(8)]], packed_float4 device* contact_state_2 [[buffer(11)]], packed_float4 device* loads_2 [[buffer(5)]], Impactor_natural_0 device* impactors_2 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(7)]], BondStatic_natural_0 device* bonds_2 [[buffer(2)]], MaterialTable_0 constant* materials_2 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_54;
    (&kernelContext_54)->params_0 = params_2;
    (&kernelContext_54)->islands_0 = islands_2;
    (&kernelContext_54)->index_0 = index_2;
    (&kernelContext_54)->chunks_0 = chunks_2;
    (&kernelContext_54)->state_0 = state_4;
    (&kernelContext_54)->scratch_0 = scratch_2;
    (&kernelContext_54)->contact_state_0 = contact_state_2;
    (&kernelContext_54)->loads_0 = loads_2;
    (&kernelContext_54)->impactors_0 = impactors_2;
    (&kernelContext_54)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_54)->bonds_0 = bonds_2;
    (&kernelContext_54)->materials_0 = materials_2;
    threadgroup array<float4, int(32)> g_part_a_2;
    (&kernelContext_54)->g_part_a_0 = &g_part_a_2;
    threadgroup array<float4, int(32)> g_part_b_2;
    (&kernelContext_54)->g_part_b_0 = &g_part_b_2;
    threadgroup array<uint, int(5)> g_seen_2;
    (&kernelContext_54)->g_seen_0 = &g_seen_2;
    threadgroup uint g_wide_run_2;
    (&kernelContext_54)->g_wide_run_0 = &g_wide_run_2;
    threadgroup array<uint, int(4096)> g_label_2;
    (&kernelContext_54)->g_label_0 = &g_label_2;
    threadgroup uint g_changed_2;
    (&kernelContext_54)->g_changed_0 = &g_changed_2;
    threadgroup uint g_halt_2;
    (&kernelContext_54)->g_halt_0 = &g_halt_2;
    threadgroup uint g_run_2;
    (&kernelContext_54)->g_run_0 = &g_run_2;
    threadgroup array<uint, int(512)> g_label_narrow_2;
    (&kernelContext_54)->g_label_narrow_0 = &g_label_narrow_2;
    threadgroup uint g_changed_narrow_2;
    (&kernelContext_54)->g_changed_narrow_0 = &g_changed_narrow_2;
    uint k_17 = id_1.x;
    bool _S913;
    if(k_17 >= (params_2->cand_count_0))
    {
        _S913 = true;
    }
    else
    {
        bool _S914 = stopped_0(&kernelContext_54);
        _S913 = _S914;
    }
    if(_S913)
    {
        return;
    }
    uint _S915 = 3U * k_17;
    uint at_7 = (&kernelContext_54)->params_0->cand_index_0 + _S915;
    uint c_23 = (&kernelContext_54)->index_0[at_7];
    uint _S916 = (&kernelContext_54)->index_0[at_7 + 2U];
    thread Impactor_natural_0 _S917 = *((&kernelContext_54)->impactors_0+_S916);
    if(((uint4((&_S917)->cand_0) ).z) == 0U)
    {
        _S913 = ((float4((&_S917)->crush_0) ).x) > 0.0f;
    }
    else
    {
        _S913 = false;
    }
    float total_1;
    float ksum_0;
    if(_S913)
    {
        WorldPoint_0 _S918 = chunk_world_0(c_23, &kernelContext_54);
        WorldPoint_0 _S919 = impactor_point_0(_S916, &kernelContext_54);
        thread WorldPoint_0 _S920 = _S918;
        thread WorldPoint_0 _S921 = _S919;
        float3 _S922 = world_diff_0(&_S920, &_S921);
        float4 _S923 = float4((&_S917)->half_1) ;
        if(!((length(_S922)) > (_S923.w + (float4(((&kernelContext_54)->chunks_0+c_23)->half_0) ).w)))
        {
            Box_0 _S924 = impactor_box_1(_S916, float3(0.0f) , _S923.xyz, &kernelContext_54);
            Box_0 _S925 = chunk_box_0(c_23, _S922, &kernelContext_54);
            float _S926 = (float4((&_S917)->mat_0) ).x;
            float _S927 = (float4(((&kernelContext_54)->chunks_0+c_23)->cmat_0) ).x;
            float3 _S928 = _S925.center_1 - _S924.center_1;
            thread Box_0 _S929 = _S924;
            thread Box_0 _S930 = _S925;
            float _S931 = contact_stiffness_0(_S926, &_S929, _S927, &_S930, _S928);
            float _S932 = (float4((&_S917)->crush_0) ).w;
            thread Box_0 _S933 = _S924;
            Box_0 _S934 = impactor_shrunk_0(_S916, _S932, &_S933);
            thread Box_0 _S935 = _S934;
            thread Box_0 _S936 = _S925;
            uint _S937 = impactor_contact_count_0(_S916, _S932, &_S935, &_S936, &kernelContext_54);
            float _S938;
            if(((float4((&_S917)->shape_0) ).x) == 0.0f)
            {
                _S938 = _S931;
            }
            else
            {
                _S938 = _S931 / max(float(_S937), 10.0f);
            }
            uint j_8 = 0U;
            total_1 = 0.0f;
            ksum_0 = 0.0f;
            for(;;)
            {
                if(_S937 > 0U)
                {
                    thread Box_0 _S939 = _S925;
                    uint _S940 = impactor_slots_0(&_S917, &_S939);
                    _S913 = j_8 < _S940;
                }
                else
                {
                    _S913 = false;
                }
                if(_S913)
                {
                }
                else
                {
                    break;
                }
                thread Box_0 _S941 = _S934;
                thread Box_0 _S942 = _S925;
                thread float3 p_13;
                thread float3 nrm_2;
                thread float dep_2;
                bool _S943 = impactor_contact_1(&_S917, _S932, &_S941, &_S942, j_8, &p_13, &nrm_2, &dep_2, &kernelContext_54);
                if(!_S943)
                {
                    j_8 = j_8 + 1U;
                    continue;
                }
                float ksum_1 = ksum_0 + _S938;
                total_1 = total_1 + _S938 * dep_2;
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
    float4 _S944 = float4(*((&kernelContext_54)->scratch_0+((&kernelContext_54)->params_0->cand_base_0 + _S915))) ;
    *((&kernelContext_54)->scratch_0+((&kernelContext_54)->params_0->cand_base_0 + _S915)) = packed_float4(float4(total_1, ksum_0, _S944.z, _S944.w)) ;
    return;
}

[[kernel]] void impactor_crush(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], Params_0 constant* params_3 [[buffer(0)]], Island_natural_0 device* islands_3 [[buffer(9)]], uint device* index_3 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(3)]], packed_float4 device* state_5 [[buffer(6)]], packed_float4 device* scratch_3 [[buffer(8)]], packed_float4 device* contact_state_3 [[buffer(11)]], packed_float4 device* loads_3 [[buffer(5)]], Impactor_natural_0 device* impactors_3 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(7)]], BondStatic_natural_0 device* bonds_3 [[buffer(2)]], MaterialTable_0 constant* materials_3 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_55;
    (&kernelContext_55)->params_0 = params_3;
    (&kernelContext_55)->islands_0 = islands_3;
    (&kernelContext_55)->index_0 = index_3;
    (&kernelContext_55)->chunks_0 = chunks_3;
    (&kernelContext_55)->state_0 = state_5;
    (&kernelContext_55)->scratch_0 = scratch_3;
    (&kernelContext_55)->contact_state_0 = contact_state_3;
    (&kernelContext_55)->loads_0 = loads_3;
    (&kernelContext_55)->impactors_0 = impactors_3;
    (&kernelContext_55)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_55)->bonds_0 = bonds_3;
    (&kernelContext_55)->materials_0 = materials_3;
    threadgroup array<float4, int(32)> g_part_a_3;
    (&kernelContext_55)->g_part_a_0 = &g_part_a_3;
    threadgroup array<float4, int(32)> g_part_b_3;
    (&kernelContext_55)->g_part_b_0 = &g_part_b_3;
    threadgroup array<uint, int(5)> g_seen_3;
    (&kernelContext_55)->g_seen_0 = &g_seen_3;
    threadgroup uint g_wide_run_3;
    (&kernelContext_55)->g_wide_run_0 = &g_wide_run_3;
    threadgroup array<uint, int(4096)> g_label_3;
    (&kernelContext_55)->g_label_0 = &g_label_3;
    threadgroup uint g_changed_3;
    (&kernelContext_55)->g_changed_0 = &g_changed_3;
    threadgroup uint g_halt_3;
    (&kernelContext_55)->g_halt_0 = &g_halt_3;
    threadgroup uint g_run_3;
    (&kernelContext_55)->g_run_0 = &g_run_3;
    threadgroup array<uint, int(512)> g_label_narrow_3;
    (&kernelContext_55)->g_label_narrow_0 = &g_label_narrow_3;
    threadgroup uint g_changed_narrow_3;
    (&kernelContext_55)->g_changed_narrow_0 = &g_changed_narrow_3;
    uint ii_1 = group_0.x;
    uint tid_6 = thread_0.x;
    bool _S945;
    if(ii_1 >= (params_3->impactor_count_0))
    {
        _S945 = true;
    }
    else
    {
        bool _S946 = stopped_0(&kernelContext_55);
        _S945 = _S946;
    }
    if(_S945)
    {
        return;
    }
    Impactor_natural_0 device* _S947 = (&kernelContext_55)->impactors_0+ii_1;
    float4 _S948 = float4((*_S947).position_err_1) ;
    float4 _S949 = float4((*_S947).velocity_1) ;
    float4 _S950 = float4((*_S947).velocity_err_1) ;
    float4 _S951 = float4((*_S947).angular_velocity_1) ;
    float4 _S952 = float4((*_S947).rotation_1) ;
    float4 _S953 = float4((*_S947).inertia0_2) ;
    float4 _S954 = float4((*_S947).inertia1_2) ;
    float4 _S955 = float4((*_S947).inertia2_2) ;
    float4 _S956 = float4((*_S947).inv0_2) ;
    float4 _S957 = float4((*_S947).inv1_2) ;
    float4 _S958 = float4((*_S947).inv2_2) ;
    float4 _S959 = float4((*_S947).shape_0) ;
    float4 _S960 = float4((*_S947).half_1) ;
    float4 _S961 = float4((*_S947).mat_0) ;
    float4 _S962 = float4((*_S947).crush_0) ;
    float4 _S963 = float4((*_S947).geom_0) ;
    float4 _S964 = float4((*_S947).rotation_err_1) ;
    float4 _S965 = float4((*_S947).ledger_0) ;
    uint4 _S966 = uint4((*_S947).cand_0) ;
    float4 _S967 = float4((*_S947).momentum_1) ;
    float4 _S968 = float4((*_S947).momentum_err_1) ;
    thread Impactor_0 imp_5;
    (&imp_5)->position_1 = float4((*_S947).position_1) ;
    (&imp_5)->position_err_1 = _S948;
    (&imp_5)->velocity_1 = _S949;
    (&imp_5)->velocity_err_1 = _S950;
    (&imp_5)->angular_velocity_1 = _S951;
    (&imp_5)->rotation_1 = _S952;
    (&imp_5)->inertia0_2 = _S953;
    (&imp_5)->inertia1_2 = _S954;
    (&imp_5)->inertia2_2 = _S955;
    (&imp_5)->inv0_2 = _S956;
    (&imp_5)->inv1_2 = _S957;
    (&imp_5)->inv2_2 = _S958;
    (&imp_5)->shape_0 = _S959;
    (&imp_5)->half_1 = _S960;
    (&imp_5)->mat_0 = _S961;
    (&imp_5)->crush_0 = _S962;
    (&imp_5)->geom_0 = _S963;
    (&imp_5)->rotation_err_1 = _S964;
    (&imp_5)->ledger_0 = _S965;
    (&imp_5)->cand_0 = _S966;
    (&imp_5)->momentum_1 = _S967;
    (&imp_5)->momentum_err_1 = _S968;
    float4 _S969 = float4(0.0f) ;
    thread float4 shares_0 = _S969;
    thread float4 unused_2 = _S969;
    uint k_18 = (&imp_5)->cand_0.x + tid_6;
    for(;;)
    {
        if(k_18 < ((&imp_5)->cand_0.y))
        {
        }
        else
        {
            break;
        }
        shares_0 = shares_0 + float4(*((&kernelContext_55)->scratch_0+((&kernelContext_55)->params_0->cand_base_0 + 3U * k_18))) ;
        k_18 = k_18 + 256U;
    }
    group_sum2_0(tid_6, &shares_0, &unused_2, &kernelContext_55);
    if(tid_6 != 0U)
    {
        _S945 = true;
    }
    else
    {
        _S945 = ((&imp_5)->cand_0.z) != 0U;
    }
    if(_S945)
    {
        return;
    }
    (&imp_5)->geom_0 = float4(1.0f, (&imp_5)->crush_0.w, 0.0f, 0.0f);
    float total_2 = shares_0.x;
    if(((&imp_5)->crush_0.x) > 0.0f)
    {
        _S945 = ((&imp_5)->crush_0.z) < ((&imp_5)->crush_0.y);
    }
    else
    {
        _S945 = false;
    }
    if(_S945)
    {
        _S945 = total_2 > ((&imp_5)->crush_0.x);
    }
    else
    {
        _S945 = false;
    }
    if(_S945)
    {
        float extra_0 = (total_2 - (&imp_5)->crush_0.x) / shares_0.y;
        (&imp_5)->crush_0.w = (&imp_5)->crush_0.w + extra_0;
        (&imp_5)->crush_0.z = (&imp_5)->crush_0.z + (&imp_5)->crush_0.x * extra_0;
        float _S970 = (&imp_5)->crush_0.x * extra_0;
        thread float _S971 = (&imp_5)->ledger_0.z;
        thread float _S972 = (&imp_5)->ledger_0.w;
        comp_add1_0(&_S971, &_S972, _S970);
        (&imp_5)->ledger_0.w = _S972;
        (&imp_5)->ledger_0.z = _S971;
        float _S973 = (&imp_5)->crush_0.x * extra_0;
        thread float _S974 = (&imp_5)->ledger_0.x;
        thread float _S975 = (&imp_5)->ledger_0.y;
        comp_add1_0(&_S974, &_S975, _S973);
        (&imp_5)->ledger_0.y = _S975;
        (&imp_5)->ledger_0.x = _S974;
        (&imp_5)->geom_0.x = (&imp_5)->crush_0.x / total_2;
    }
    Impactor_natural_0 device* _S976 = (&kernelContext_55)->impactors_0+ii_1;
    _S976->position_1 = packed_float4(imp_5.position_1) ;
    _S976->position_err_1 = packed_float4(imp_5.position_err_1) ;
    _S976->velocity_1 = packed_float4(imp_5.velocity_1) ;
    _S976->velocity_err_1 = packed_float4(imp_5.velocity_err_1) ;
    _S976->angular_velocity_1 = packed_float4(imp_5.angular_velocity_1) ;
    _S976->rotation_1 = packed_float4(imp_5.rotation_1) ;
    _S976->inertia0_2 = packed_float4(imp_5.inertia0_2) ;
    _S976->inertia1_2 = packed_float4(imp_5.inertia1_2) ;
    _S976->inertia2_2 = packed_float4(imp_5.inertia2_2) ;
    _S976->inv0_2 = packed_float4(imp_5.inv0_2) ;
    _S976->inv1_2 = packed_float4(imp_5.inv1_2) ;
    _S976->inv2_2 = packed_float4(imp_5.inv2_2) ;
    _S976->shape_0 = packed_float4(imp_5.shape_0) ;
    _S976->half_1 = packed_float4(imp_5.half_1) ;
    _S976->mat_0 = packed_float4(imp_5.mat_0) ;
    _S976->crush_0 = packed_float4(imp_5.crush_0) ;
    _S976->geom_0 = packed_float4(imp_5.geom_0) ;
    _S976->rotation_err_1 = packed_float4(imp_5.rotation_err_1) ;
    _S976->ledger_0 = packed_float4(imp_5.ledger_0) ;
    _S976->cand_0 = packed_uint4(imp_5.cand_0) ;
    _S976->momentum_1 = packed_float4(imp_5.momentum_1) ;
    _S976->momentum_err_1 = packed_float4(imp_5.momentum_err_1) ;
    return;
}

struct WideGroup_0
{
    uint island_3;
    uint begin_0;
    uint end_0;
    uint first_0;
};

WideGroup_0 wide_group_0(uint table_0, uint g_1, KernelContext_0 thread* kernelContext_56)
{
    thread WideGroup_0 w_8;
    uint _S977 = table_0 + 4U * g_1;
    (&w_8)->island_3 = kernelContext_56->index_0[_S977];
    (&w_8)->begin_0 = kernelContext_56->index_0[_S977 + 1U];
    (&w_8)->end_0 = kernelContext_56->index_0[_S977 + 2U];
    (&w_8)->first_0 = kernelContext_56->index_0[_S977 + 3U];
    return w_8;
}

bool contact_stopped_1(const Island_natural_0 thread* isl_11, KernelContext_0 thread* kernelContext_57)
{
    uint4 _S978 = uint4((kernelContext_57->islands_0+kernelContext_57->params_0->halt_index_0)->info_0) ;
    bool _S979;
    if(((_S978.z) & 1U) != 0U)
    {
        _S979 = true;
    }
    else
    {
        uint _S980 = _S978.y;
        if(_S980 != 0U)
        {
            _S979 = _S980 <= ((uint4(isl_11->info_0) ).w);
        }
        else
        {
            _S979 = false;
        }
    }
    return _S979;
}

bool wide_runs_0(const Island_natural_0 thread* isl_12, KernelContext_0 thread* kernelContext_58)
{
    uint4 _S981 = uint4(isl_12->info_0) ;
    bool _S982;
    if(((_S981.z) & 1U) != 0U)
    {
        _S982 = true;
    }
    else
    {
        _S982 = (_S981.y) == 0U;
    }
    if(_S982)
    {
        return false;
    }
    if(((_S981.x) & 4U) == 0U)
    {
        _S982 = true;
    }
    else
    {
        bool _S983 = contact_stopped_1(isl_12, kernelContext_58);
        _S982 = !_S983;
    }
    return _S982;
}

bool wide_enter_0(uint tid_7, const Island_natural_0 thread* isl_13, KernelContext_0 thread* kernelContext_59)
{
    seen_clear_0(tid_7, kernelContext_59);
    if(tid_7 == 0U)
    {
        bool _S984 = wide_runs_0(isl_13, kernelContext_59);
        int _S985;
        if(_S984)
        {
            _S985 = int(1);
        }
        else
        {
            _S985 = int(0);
        }
        *kernelContext_59->g_wide_run_0 = uint(_S985);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_59->g_wide_run_0) != 0U;
}

uint wide_step_0(const Island_natural_0 thread* isl_14, KernelContext_0 thread* kernelContext_60)
{
    return (uint4(isl_14->info_0) ).w - kernelContext_60->params_0->step_start_0;
}

uint wide_step_1(const Island_natural_0 thread* isl_15, KernelContext_0 thread* kernelContext_61)
{
    return (uint4(isl_15->info_0) ).w - kernelContext_61->params_0->step_start_0;
}

uint wide_step_2(const Island_0 thread* isl_16, KernelContext_0 thread* kernelContext_62)
{
    return isl_16->info_0.w - kernelContext_62->params_0->step_start_0;
}

bool wide_enter_1(uint _S986, uint _S987, KernelContext_0 thread* kernelContext_63)
{
    thread Island_natural_0 _S988 = *(kernelContext_63->islands_0+_S987);
    seen_clear_0(_S986, kernelContext_63);
    if(_S986 == 0U)
    {
        bool _S989 = wide_runs_0(&_S988, kernelContext_63);
        int _S990;
        if(_S989)
        {
            _S990 = int(1);
        }
        else
        {
            _S990 = int(0);
        }
        *kernelContext_63->g_wide_run_0 = uint(_S990);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_63->g_wide_run_0) != 0U;
}

[[kernel]] void wide_wake(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], Params_0 constant* params_4 [[buffer(0)]], Island_natural_0 device* islands_4 [[buffer(9)]], uint device* index_4 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(3)]], packed_float4 device* state_6 [[buffer(6)]], packed_float4 device* scratch_4 [[buffer(8)]], packed_float4 device* contact_state_4 [[buffer(11)]], packed_float4 device* loads_4 [[buffer(5)]], Impactor_natural_0 device* impactors_4 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(7)]], BondStatic_natural_0 device* bonds_4 [[buffer(2)]], MaterialTable_0 constant* materials_4 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_64;
    (&kernelContext_64)->params_0 = params_4;
    (&kernelContext_64)->islands_0 = islands_4;
    (&kernelContext_64)->index_0 = index_4;
    (&kernelContext_64)->chunks_0 = chunks_4;
    (&kernelContext_64)->state_0 = state_6;
    (&kernelContext_64)->scratch_0 = scratch_4;
    (&kernelContext_64)->contact_state_0 = contact_state_4;
    (&kernelContext_64)->loads_0 = loads_4;
    (&kernelContext_64)->impactors_0 = impactors_4;
    (&kernelContext_64)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_64)->bonds_0 = bonds_4;
    (&kernelContext_64)->materials_0 = materials_4;
    threadgroup array<float4, int(32)> g_part_a_4;
    (&kernelContext_64)->g_part_a_0 = &g_part_a_4;
    threadgroup array<float4, int(32)> g_part_b_4;
    (&kernelContext_64)->g_part_b_0 = &g_part_b_4;
    threadgroup array<uint, int(5)> g_seen_4;
    (&kernelContext_64)->g_seen_0 = &g_seen_4;
    threadgroup uint g_wide_run_4;
    (&kernelContext_64)->g_wide_run_0 = &g_wide_run_4;
    threadgroup array<uint, int(4096)> g_label_4;
    (&kernelContext_64)->g_label_0 = &g_label_4;
    threadgroup uint g_changed_4;
    (&kernelContext_64)->g_changed_0 = &g_changed_4;
    threadgroup uint g_halt_4;
    (&kernelContext_64)->g_halt_0 = &g_halt_4;
    threadgroup uint g_run_4;
    (&kernelContext_64)->g_run_0 = &g_run_4;
    threadgroup array<uint, int(512)> g_label_narrow_4;
    (&kernelContext_64)->g_label_narrow_0 = &g_label_narrow_4;
    threadgroup uint g_changed_narrow_4;
    (&kernelContext_64)->g_changed_narrow_0 = &g_changed_narrow_4;
    uint tid_8 = thread_1.x;
    uint _S991 = group_1.x;
    WideGroup_0 _S992 = wide_group_0(params_4->wide_chunk_table_0, _S991, &kernelContext_64);
    if(_S991 != (_S992.first_0))
    {
        return;
    }
    thread Island_natural_0 _S993 = *((&kernelContext_64)->islands_0+_S992.island_3);
    uint4 _S994 = uint4((&_S993)->info_0) ;
    uint _S995 = _S994.x;
    bool _S996;
    if((_S995 & 16U) == 0U)
    {
        _S996 = true;
    }
    else
    {
        bool _S997 = wide_enter_1(tid_8, _S992.island_3, &kernelContext_64);
        _S996 = !_S997;
    }
    if(_S996)
    {
        return;
    }
    Quat_0 _S998 = quat_of_0(float4((&_S993)->rotation_0) );
    bool _S999 = (_S995 & 4U) != 0U;
    float3 _S1000 = float3(0.0f) ;
    thread float3 norm_2 = _S1000;
    thread float3 unused_3 = _S1000;
    uint4 _S1001 = uint4((&_S993)->range_0) ;
    uint c_24 = _S1001.x + tid_8;
    for(;;)
    {
        if(c_24 < (_S1001.y))
        {
        }
        else
        {
            break;
        }
        uint _S1002 = wide_step_0(&_S993, &kernelContext_64);
        float _S1003 = (&kernelContext_64)->params_0->dt_0;
        thread Quat_0 _S1004 = _S998;
        float _S1005 = settled_chunk_load_0(c_24, &_S1004, _S1002, _S1003, _S999, &kernelContext_64);
        norm_2.x = norm_2.x + _S1005;
        c_24 = c_24 + 256U;
    }
    group_sum3_0(tid_8, &norm_2, &unused_3, &kernelContext_64);
    if(tid_8 != 0U)
    {
        return;
    }
    uint _S1006 = seen_bits_0(&kernelContext_64);
    uint cov_10 = 1024U | _S1006;
    if(((&kernelContext_64)->params_0->solve_mode_0) == 1U)
    {
        float4 _S1007 = float4((&_S993)->energy_0) ;
        _S996 = (abs(norm_2.x - _S1007.z)) > (_S1007.w);
    }
    else
    {
        _S996 = false;
    }
    uint cov_11;
    if(_S996)
    {
        ((&kernelContext_64)->islands_0+_S992.island_3)->info_0[int(0)] = _S995 & 4294967279U;
        ((&kernelContext_64)->islands_0+_S992.island_3)->info_0[int(2)] = (_S994.z) | 4U;
        cov_11 = cov_10 | 2048U;
    }
    else
    {
        cov_11 = cov_10;
    }
    cov_mark_0(_S992.island_3, (&_S993)->coverage_0, cov_11, &kernelContext_64);
    return;
}

void wide_partials_0(uint tid_9, uint first_1, uint count_6, uint p_14, float3 thread* a_13, float3 thread* b_30, KernelContext_0 thread* kernelContext_65)
{
    float4 _S1008 = float4(0.0f) ;
    thread float4 x_14 = _S1008;
    thread float4 y_2 = _S1008;
    uint s_5 = tid_9;
    for(;;)
    {
        if(s_5 < count_6)
        {
        }
        else
        {
            break;
        }
        uint _S1009 = 8U * (first_1 + s_5);
        x_14 = x_14 + float4(*(kernelContext_65->scratch_0+(kernelContext_65->params_0->wide_base_0 + _S1009 + p_14))) ;
        y_2 = y_2 + float4(*(kernelContext_65->scratch_0+(kernelContext_65->params_0->wide_base_0 + _S1009 + p_14 + 1U))) ;
        s_5 = s_5 + 256U;
    }
    group_sum2_0(tid_9, &x_14, &y_2, kernelContext_65);
    *a_13 = x_14.xyz;
    *b_30 = y_2.xyz;
    return;
}

void wide_store_0(uint slot_2, uint p_15, float3 a_14, float3 b_31, KernelContext_0 thread* kernelContext_66)
{
    uint _S1010 = 8U * slot_2;
    *(kernelContext_66->scratch_0+(kernelContext_66->params_0->wide_base_0 + _S1010 + p_15)) = packed_float4(float4(a_14, 0.0f)) ;
    *(kernelContext_66->scratch_0+(kernelContext_66->params_0->wide_base_0 + _S1010 + p_15 + 1U)) = packed_float4(float4(b_31, 0.0f)) ;
    return;
}

[[kernel]] void wide_drift(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], Params_0 constant* params_5 [[buffer(0)]], Island_natural_0 device* islands_5 [[buffer(9)]], uint device* index_5 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(3)]], packed_float4 device* state_7 [[buffer(6)]], packed_float4 device* scratch_5 [[buffer(8)]], packed_float4 device* contact_state_5 [[buffer(11)]], packed_float4 device* loads_5 [[buffer(5)]], Impactor_natural_0 device* impactors_5 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(7)]], BondStatic_natural_0 device* bonds_5 [[buffer(2)]], MaterialTable_0 constant* materials_5 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_67;
    (&kernelContext_67)->params_0 = params_5;
    (&kernelContext_67)->islands_0 = islands_5;
    (&kernelContext_67)->index_0 = index_5;
    (&kernelContext_67)->chunks_0 = chunks_5;
    (&kernelContext_67)->state_0 = state_7;
    (&kernelContext_67)->scratch_0 = scratch_5;
    (&kernelContext_67)->contact_state_0 = contact_state_5;
    (&kernelContext_67)->loads_0 = loads_5;
    (&kernelContext_67)->impactors_0 = impactors_5;
    (&kernelContext_67)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_67)->bonds_0 = bonds_5;
    (&kernelContext_67)->materials_0 = materials_5;
    threadgroup array<float4, int(32)> g_part_a_5;
    (&kernelContext_67)->g_part_a_0 = &g_part_a_5;
    threadgroup array<float4, int(32)> g_part_b_5;
    (&kernelContext_67)->g_part_b_0 = &g_part_b_5;
    threadgroup array<uint, int(5)> g_seen_5;
    (&kernelContext_67)->g_seen_0 = &g_seen_5;
    threadgroup uint g_wide_run_5;
    (&kernelContext_67)->g_wide_run_0 = &g_wide_run_5;
    threadgroup array<uint, int(4096)> g_label_5;
    (&kernelContext_67)->g_label_0 = &g_label_5;
    threadgroup uint g_changed_5;
    (&kernelContext_67)->g_changed_0 = &g_changed_5;
    threadgroup uint g_halt_5;
    (&kernelContext_67)->g_halt_0 = &g_halt_5;
    threadgroup uint g_run_5;
    (&kernelContext_67)->g_run_0 = &g_run_5;
    threadgroup array<uint, int(512)> g_label_narrow_5;
    (&kernelContext_67)->g_label_narrow_0 = &g_label_narrow_5;
    threadgroup uint g_changed_narrow_5;
    (&kernelContext_67)->g_changed_narrow_0 = &g_changed_narrow_5;
    uint tid_10 = thread_2.x;
    uint _S1011 = group_2.x;
    WideGroup_0 _S1012 = wide_group_0(params_5->wide_chunk_table_0, _S1011, &kernelContext_67);
    Island_natural_0 device* _S1013 = (&kernelContext_67)->islands_0+_S1012.island_3;
    Island_natural_0 isl_17 = *_S1013;
    bool _S1014;
    if((((uint4((*_S1013).info_0) ).x) & 17U) != 0U)
    {
        _S1014 = true;
    }
    else
    {
        bool _S1015 = wide_enter_1(tid_10, _S1012.island_3, &kernelContext_67);
        _S1014 = !_S1015;
    }
    if(_S1014)
    {
        return;
    }
    thread float3 tu_3;
    thread float3 pv_3;
    wide_partials_0(tid_10, _S1012.first_0, isl_17.wide_groups_0, 2U, &tu_3, &pv_3, &kernelContext_67);
    float4 _S1016 = float4(isl_17.wcom_0) ;
    float3 _S1017 = float3(_S1016.w) ;
    float3 tr_5 = tu_3 / _S1017;
    float3 dv_5 = pv_3 / _S1017;
    float3 _S1018 = float3(0.0f) ;
    thread float3 lu_3 = _S1018;
    thread float3 lv_3 = _S1018;
    uint c_25 = _S1012.begin_0 + tid_10;
    if(c_25 < (_S1012.end_0))
    {
        drift_angular_0(c_25, _S1016.xyz, tr_5, dv_5, &lu_3, &lv_3, &kernelContext_67);
    }
    group_sum3_0(tid_10, &lu_3, &lv_3, &kernelContext_67);
    if(tid_10 == 0U)
    {
        wide_store_0(_S1011, 4U, lu_3, lv_3, &kernelContext_67);
    }
    return;
}

Rigid_0 rigid_of_1(const Island_natural_0 thread* isl_18)
{
    thread Rigid_0 rg_9;
    (&rg_9)->rot_0 = quat_of_0(float4(isl_18->rotation_0) );
    (&rg_9)->rot_err_0 = float4(isl_18->rotation_err_0) ;
    (&rg_9)->pos_0 = (float4(isl_18->position_0) ).xyz;
    (&rg_9)->pos_err_0 = (float4(isl_18->position_err_0) ).xyz;
    (&rg_9)->vel_0 = (float4(isl_18->velocity_0) ).xyz;
    (&rg_9)->vel_err_0 = (float4(isl_18->velocity_err_0) ).xyz;
    (&rg_9)->w_1 = (float4(isl_18->angular_velocity_0) ).xyz;
    (&rg_9)->l_0 = (float4(isl_18->momentum_0) ).xyz;
    (&rg_9)->l_err_0 = (float4(isl_18->momentum_err_0) ).xyz;
    float3 _S1019 = float3(0.0f) ;
    (&rg_9)->torque_0 = _S1019;
    (&rg_9)->a_0 = _S1019;
    (&rg_9)->alpha_0 = _S1019;
    return rg_9;
}

void rigid_acceleration_1(const Island_natural_0 thread* isl_19, Rigid_0 thread* rg_10, float3 f_12, float3 t_13)
{
    float4 _S1020 = float4(isl_19->inertia0_0) ;
    float4 _S1021 = float4(isl_19->inertia1_0) ;
    float4 _S1022 = float4(isl_19->inertia2_0) ;
    thread Quat_0 _S1023 = rg_10->rot_0;
    float3 _S1024 = world_mul_0(&_S1023, _S1020, _S1021, _S1022, rg_10->w_1);
    rg_10->a_0 = f_12 / float3((float4(isl_19->com_0) ).w) ;
    float4 _S1025 = float4(isl_19->inv0_0) ;
    float4 _S1026 = float4(isl_19->inv1_0) ;
    float4 _S1027 = float4(isl_19->inv2_0) ;
    float3 _S1028 = t_13 - cross(rg_10->w_1, _S1024);
    thread Quat_0 _S1029 = rg_10->rot_0;
    float3 _S1030 = world_mul_0(&_S1029, _S1025, _S1026, _S1027, _S1028);
    rg_10->alpha_0 = _S1030;
    rg_10->torque_0 = t_13;
    return;
}

Rigid_0 wide_rigid_frame_0(uint tid_11, const Island_natural_0 thread* isl_20, const WideGroup_0 thread* wg_0, KernelContext_0 thread* kernelContext_68)
{
    Rigid_0 _S1031 = rigid_of_1(isl_20);
    thread Rigid_0 rg_11 = _S1031;
    if((((uint4(isl_20->info_0) ).x) & 1U) == 0U)
    {
        thread float3 f_13;
        thread float3 t_14;
        wide_partials_0(tid_11, wg_0->first_0, isl_20->wide_groups_0, 0U, &f_13, &t_14, kernelContext_68);
        rigid_acceleration_1(isl_20, &rg_11, f_13, t_14);
    }
    return rg_11;
}

Rigid_0 wide_rigid_frame_1(uint tid_12, const Island_natural_0 thread* isl_21, const WideGroup_0 thread* wg_1, KernelContext_0 thread* kernelContext_69)
{
    Rigid_0 _S1032 = rigid_of_1(isl_21);
    thread Rigid_0 rg_12 = _S1032;
    if((((uint4(isl_21->info_0) ).x) & 1U) == 0U)
    {
        thread float3 f_14;
        thread float3 t_15;
        wide_partials_0(tid_12, wg_1->first_0, isl_21->wide_groups_0, 0U, &f_14, &t_15, kernelContext_69);
        rigid_acceleration_1(isl_21, &rg_12, f_14, t_15);
    }
    return rg_12;
}

void integrate_rigid_1(const Island_natural_0 thread* isl_22, Rigid_0 thread* rg_13, float dt_12)
{
    float3 _S1033 = float3(dt_12) ;
    comp_add_0(&rg_13->l_0, &rg_13->l_err_0, rg_13->torque_0 * _S1033);
    float3 l_5 = rg_13->l_0 + rg_13->l_err_0;
    comp_add_0(&rg_13->vel_0, &rg_13->vel_err_0, rg_13->a_0 * _S1033);
    float3 vel_2 = rg_13->vel_0 + rg_13->vel_err_0;
    float4 _S1034 = float4(isl_22->inv0_0) ;
    float4 _S1035 = float4(isl_22->inv1_0) ;
    float4 _S1036 = float4(isl_22->inv2_0) ;
    thread Quat_0 _S1037 = rg_13->rot_0;
    float3 _S1038 = world_mul_0(&_S1037, _S1034, _S1035, _S1036, l_5);
    float3 _S1039 = vel_2 * _S1033;
    float3 _S1040 = (float4(isl_22->com_0) ).xyz;
    thread Quat_0 _S1041 = rg_13->rot_0;
    float3 _S1042 = rotate_0(&_S1041, _S1040);
    comp_add_0(&rg_13->pos_0, &rg_13->pos_err_0, _S1039 - turn_difference_0(_S1038, dt_12, _S1042));
    turn_left_0(&rg_13->rot_0, &rg_13->rot_err_0, _S1038, dt_12);
    thread Quat_0 _S1043 = rg_13->rot_0;
    float3 _S1044 = world_mul_0(&_S1043, _S1034, _S1035, _S1036, l_5);
    rg_13->w_1 = _S1044;
    return;
}

void drift_rigid_1(const Island_natural_0 thread* isl_23, Rigid_0 thread* rg_14, float3 tr_6, float3 phi_5, float3 dv_6, float3 dw_4)
{
    float3 wcom_4 = (float4(isl_23->wcom_0) ).xyz;
    Quat_0 rot_3 = rg_14->rot_0;
    float3 _S1045 = tr_6 - turn_difference_0(phi_5, 1.0f, wcom_4);
    thread Quat_0 _S1046 = rg_14->rot_0;
    float3 _S1047 = rotate_0(&_S1046, _S1045);
    comp_add_0(&rg_14->pos_0, &rg_14->pos_err_0, _S1047);
    thread Quat_0 _S1048 = rg_14->rot_0;
    float3 _S1049 = inverse_rotate_1(&_S1048, rg_14->w_1);
    float4 _S1050 = float4(isl_23->inertia0_0) ;
    float4 _S1051 = float4(isl_23->inertia1_0) ;
    float4 _S1052 = float4(isl_23->inertia2_0) ;
    float3 _S1053 = cross(phi_5, rows_mul_0(_S1050, _S1051, _S1052, _S1049)) - rows_mul_0(_S1050, _S1051, _S1052, cross(phi_5, _S1049));
    thread Quat_0 _S1054 = rg_14->rot_0;
    float3 _S1055 = rotate_0(&_S1054, _S1053);
    turn_right_0(&rg_14->rot_0, &rg_14->rot_err_0, phi_5);
    float3 _S1056 = dv_6 + cross(dw_4, (float4(isl_23->com_0) ).xyz - wcom_4);
    thread Quat_0 _S1057 = rot_3;
    float3 _S1058 = rotate_0(&_S1057, _S1056);
    comp_add_0(&rg_14->vel_0, &rg_14->vel_err_0, _S1058);
    thread Quat_0 _S1059 = rot_3;
    float3 _S1060 = rotate_0(&_S1059, dw_4);
    rg_14->w_1 = rg_14->w_1 + _S1060;
    thread Quat_0 _S1061 = rg_14->rot_0;
    float3 _S1062 = world_mul_0(&_S1061, _S1050, _S1051, _S1052, _S1060);
    comp_add_0(&rg_14->l_0, &rg_14->l_err_0, _S1055 + _S1062);
    return;
}

[[kernel]] void wide_rigid(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], Params_0 constant* params_6 [[buffer(0)]], Island_natural_0 device* islands_6 [[buffer(9)]], uint device* index_6 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(3)]], packed_float4 device* state_8 [[buffer(6)]], packed_float4 device* scratch_6 [[buffer(8)]], packed_float4 device* contact_state_6 [[buffer(11)]], packed_float4 device* loads_6 [[buffer(5)]], Impactor_natural_0 device* impactors_6 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(7)]], BondStatic_natural_0 device* bonds_6 [[buffer(2)]], MaterialTable_0 constant* materials_6 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_70;
    (&kernelContext_70)->params_0 = params_6;
    (&kernelContext_70)->islands_0 = islands_6;
    (&kernelContext_70)->index_0 = index_6;
    (&kernelContext_70)->chunks_0 = chunks_6;
    (&kernelContext_70)->state_0 = state_8;
    (&kernelContext_70)->scratch_0 = scratch_6;
    (&kernelContext_70)->contact_state_0 = contact_state_6;
    (&kernelContext_70)->loads_0 = loads_6;
    (&kernelContext_70)->impactors_0 = impactors_6;
    (&kernelContext_70)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_70)->bonds_0 = bonds_6;
    (&kernelContext_70)->materials_0 = materials_6;
    threadgroup array<float4, int(32)> g_part_a_6;
    (&kernelContext_70)->g_part_a_0 = &g_part_a_6;
    threadgroup array<float4, int(32)> g_part_b_6;
    (&kernelContext_70)->g_part_b_0 = &g_part_b_6;
    threadgroup array<uint, int(5)> g_seen_6;
    (&kernelContext_70)->g_seen_0 = &g_seen_6;
    threadgroup uint g_wide_run_6;
    (&kernelContext_70)->g_wide_run_0 = &g_wide_run_6;
    threadgroup array<uint, int(4096)> g_label_6;
    (&kernelContext_70)->g_label_0 = &g_label_6;
    threadgroup uint g_changed_6;
    (&kernelContext_70)->g_changed_0 = &g_changed_6;
    threadgroup uint g_halt_6;
    (&kernelContext_70)->g_halt_0 = &g_halt_6;
    threadgroup uint g_run_6;
    (&kernelContext_70)->g_run_0 = &g_run_6;
    threadgroup array<uint, int(512)> g_label_narrow_6;
    (&kernelContext_70)->g_label_narrow_0 = &g_label_narrow_6;
    threadgroup uint g_changed_narrow_6;
    (&kernelContext_70)->g_changed_narrow_0 = &g_changed_narrow_6;
    uint tid_13 = thread_3.x;
    uint _S1063 = group_3.x;
    WideGroup_0 _S1064 = wide_group_0(params_6->wide_chunk_table_0, _S1063, &kernelContext_70);
    thread Island_natural_0 _S1065 = *((&kernelContext_70)->islands_0+_S1064.island_3);
    uint _S1066 = (uint4((&_S1065)->info_0) ).x;
    bool _S1067;
    if((_S1066 & 1U) != 0U)
    {
        _S1067 = true;
    }
    else
    {
        bool _S1068 = wide_enter_1(tid_13, _S1064.island_3, &kernelContext_70);
        _S1067 = !_S1068;
    }
    if(_S1067)
    {
        return;
    }
    if((_S1066 & 16U) != 0U)
    {
        if(_S1063 != (_S1064.first_0))
        {
            return;
        }
        thread WideGroup_0 _S1069 = _S1064;
        Rigid_0 _S1070 = wide_rigid_frame_0(tid_13, &_S1065, &_S1069, &kernelContext_70);
        thread Rigid_0 rs_0 = _S1070;
        if(tid_13 != 0U)
        {
            _S1067 = true;
        }
        else
        {
            _S1067 = (_S1066 & 2U) != 0U;
        }
        if(_S1067)
        {
            return;
        }
        integrate_rigid_1(&_S1065, &rs_0, (&kernelContext_70)->params_0->dt_0);
        thread Quat_0 _S1071 = (&rs_0)->rot_0;
        float4 _S1072 = quat_vec_0(&_S1071);
        ((&kernelContext_70)->islands_0+_S1064.island_3)->rotation_0 = packed_float4(_S1072) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->rotation_err_0 = packed_float4((&rs_0)->rot_err_0) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->position_0 = packed_float4(float4((&rs_0)->pos_0, 0.0f)) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->position_err_0 = packed_float4(float4((&rs_0)->pos_err_0, 0.0f)) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->velocity_0 = packed_float4(float4((&rs_0)->vel_0, 0.0f)) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->velocity_err_0 = packed_float4(float4((&rs_0)->vel_err_0, 0.0f)) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->angular_velocity_0 = packed_float4(float4((&rs_0)->w_1, 0.0f)) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->momentum_0 = packed_float4(float4((&rs_0)->l_0, 0.0f)) ;
        ((&kernelContext_70)->islands_0+_S1064.island_3)->momentum_err_0 = packed_float4(float4((&rs_0)->l_err_0, 0.0f)) ;
        return;
    }
    thread float3 tu_4;
    thread float3 pv_4;
    wide_partials_0(tid_13, _S1064.first_0, (&_S1065)->wide_groups_0, 2U, &tu_4, &pv_4, &kernelContext_70);
    thread float3 lu_4;
    thread float3 lv_4;
    wide_partials_0(tid_13, _S1064.first_0, (&_S1065)->wide_groups_0, 4U, &lu_4, &lv_4, &kernelContext_70);
    float4 _S1073 = float4((&_S1065)->wcom_0) ;
    float3 _S1074 = float3(_S1073.w) ;
    float3 tr_7 = tu_4 / _S1074;
    float3 dv_7 = pv_4 / _S1074;
    float4 _S1075 = float4((&_S1065)->winv0_0) ;
    float4 _S1076 = float4((&_S1065)->winv1_0) ;
    float4 _S1077 = float4((&_S1065)->winv2_0) ;
    float3 phi_6 = rows_mul_0(_S1075, _S1076, _S1077, lu_4);
    float3 dw_5 = rows_mul_0(_S1075, _S1076, _S1077, lv_4);
    uint c_26 = _S1064.begin_0 + tid_13;
    if(c_26 < (_S1064.end_0))
    {
        drift_apply_0(c_26, _S1073.xyz, tr_7, phi_6, dv_7, dw_5, &kernelContext_70);
    }
    if(_S1063 != (_S1064.first_0))
    {
        return;
    }
    thread WideGroup_0 _S1078 = _S1064;
    Rigid_0 _S1079 = wide_rigid_frame_0(tid_13, &_S1065, &_S1078, &kernelContext_70);
    thread Rigid_0 rg_15 = _S1079;
    if(tid_13 != 0U)
    {
        return;
    }
    if(!((_S1066 & 2U) != 0U))
    {
        integrate_rigid_1(&_S1065, &rg_15, (&kernelContext_70)->params_0->dt_0);
        drift_rigid_1(&_S1065, &rg_15, tr_7, phi_6, dv_7, dw_5);
    }
    thread Quat_0 _S1080 = (&rg_15)->rot_0;
    float4 _S1081 = quat_vec_0(&_S1080);
    ((&kernelContext_70)->islands_0+_S1064.island_3)->rotation_0 = packed_float4(_S1081) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->rotation_err_0 = packed_float4((&rg_15)->rot_err_0) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->position_0 = packed_float4(float4((&rg_15)->pos_0, 0.0f)) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->position_err_0 = packed_float4(float4((&rg_15)->pos_err_0, 0.0f)) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->velocity_0 = packed_float4(float4((&rg_15)->vel_0, 0.0f)) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->velocity_err_0 = packed_float4(float4((&rg_15)->vel_err_0, 0.0f)) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->angular_velocity_0 = packed_float4(float4((&rg_15)->w_1, 0.0f)) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->momentum_0 = packed_float4(float4((&rg_15)->l_0, 0.0f)) ;
    ((&kernelContext_70)->islands_0+_S1064.island_3)->momentum_err_0 = packed_float4(float4((&rg_15)->l_err_0, 0.0f)) ;
    return;
}

bool island_connected_0(uint tid_14, const Island_0 thread* isl_24, KernelContext_0 thread* kernelContext_71)
{
    uint _S1082 = isl_24->range_0.x;
    uint n_11 = isl_24->range_0.y - _S1082;
    bool converged_2;
    if(n_11 > 4096U)
    {
        converged_2 = true;
    }
    else
    {
        converged_2 = (kernelContext_71->params_0->device_connect_0) == 0U;
    }
    if(converged_2)
    {
        return false;
    }
    uint c_27 = tid_14;
    for(;;)
    {
        if(c_27 < n_11)
        {
        }
        else
        {
            break;
        }
        (*kernelContext_71->g_label_0)[c_27] = c_27;
        c_27 = c_27 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    converged_2 = false;
    uint pass_1 = 0U;
    for(;;)
    {
        if(pass_1 <= n_11)
        {
        }
        else
        {
            break;
        }
        if(tid_14 == 0U)
        {
            *kernelContext_71->g_changed_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c_27 = tid_14;
        for(;;)
        {
            if(c_27 < n_11)
            {
            }
            else
            {
                break;
            }
            uint gc_1 = _S1082 + c_27;
            uint _S1083 = kernelContext_71->index_0[gc_1];
            uint l_6 = (*kernelContext_71->g_label_0)[c_27];
            uint e_4 = _S1083;
            for(;;)
            {
                if(e_4 < (kernelContext_71->index_0)[gc_1 + 1U])
                {
                }
                else
                {
                    break;
                }
                uint entry_3 = kernelContext_71->index_0[e_4];
                uint bond_1 = entry_3 >> 1U;
                bool _S1084 = ((float4((kernelContext_71->bonds_0+bond_1)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S1085 = (kernelContext_71->bond_dyn_0+bond_1)->js_0;
                bool _S1086 = connected_0(&_S1085, _S1084);
                if(!_S1086)
                {
                    e_4 = e_4 + 1U;
                    continue;
                }
                uint other_1;
                if((entry_3 & 1U) == 0U)
                {
                    other_1 = (uint4((kernelContext_71->bonds_0+bond_1)->law_0.ids_0) ).z;
                }
                else
                {
                    other_1 = (uint4((kernelContext_71->bonds_0+bond_1)->law_0.ids_0) ).y;
                }
                l_6 = min(l_6, (*kernelContext_71->g_label_0)[other_1 - _S1082]);
                e_4 = e_4 + 1U;
            }
            uint _S1087 = min(l_6, (*kernelContext_71->g_label_0)[l_6]);
            if(_S1087 < (*kernelContext_71->g_label_0)[c_27])
            {
                (*kernelContext_71->g_label_0)[c_27] = _S1087;
                *kernelContext_71->g_changed_0 = 1U;
            }
            c_27 = c_27 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool converged_3 = (*kernelContext_71->g_changed_0) == 0U;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(converged_3)
        {
            converged_2 = converged_3;
            break;
        }
        uint _S1088 = pass_1 + 1U;
        converged_2 = converged_3;
        pass_1 = _S1088;
    }
    if(!converged_2)
    {
        return false;
    }
    if(tid_14 == 0U)
    {
        *kernelContext_71->g_changed_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    c_27 = tid_14;
    for(;;)
    {
        if(c_27 < n_11)
        {
        }
        else
        {
            break;
        }
        if(((*kernelContext_71->g_label_0)[c_27]) != 0U)
        {
            *kernelContext_71->g_changed_0 = 1U;
        }
        c_27 = c_27 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_71->g_changed_0) == 0U;
}

[[kernel]] void wide_end(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], Params_0 constant* params_7 [[buffer(0)]], Island_natural_0 device* islands_7 [[buffer(9)]], uint device* index_7 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(3)]], packed_float4 device* state_9 [[buffer(6)]], packed_float4 device* scratch_7 [[buffer(8)]], packed_float4 device* contact_state_7 [[buffer(11)]], packed_float4 device* loads_7 [[buffer(5)]], Impactor_natural_0 device* impactors_7 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(7)]], BondStatic_natural_0 device* bonds_7 [[buffer(2)]], MaterialTable_0 constant* materials_7 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_72;
    (&kernelContext_72)->params_0 = params_7;
    (&kernelContext_72)->islands_0 = islands_7;
    (&kernelContext_72)->index_0 = index_7;
    (&kernelContext_72)->chunks_0 = chunks_7;
    (&kernelContext_72)->state_0 = state_9;
    (&kernelContext_72)->scratch_0 = scratch_7;
    (&kernelContext_72)->contact_state_0 = contact_state_7;
    (&kernelContext_72)->loads_0 = loads_7;
    (&kernelContext_72)->impactors_0 = impactors_7;
    (&kernelContext_72)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_72)->bonds_0 = bonds_7;
    (&kernelContext_72)->materials_0 = materials_7;
    threadgroup array<float4, int(32)> g_part_a_7;
    (&kernelContext_72)->g_part_a_0 = &g_part_a_7;
    threadgroup array<float4, int(32)> g_part_b_7;
    (&kernelContext_72)->g_part_b_0 = &g_part_b_7;
    threadgroup array<uint, int(5)> g_seen_7;
    (&kernelContext_72)->g_seen_0 = &g_seen_7;
    threadgroup uint g_wide_run_7;
    (&kernelContext_72)->g_wide_run_0 = &g_wide_run_7;
    threadgroup array<uint, int(4096)> g_label_7;
    (&kernelContext_72)->g_label_0 = &g_label_7;
    threadgroup uint g_changed_7;
    (&kernelContext_72)->g_changed_0 = &g_changed_7;
    threadgroup uint g_halt_7;
    (&kernelContext_72)->g_halt_0 = &g_halt_7;
    threadgroup uint g_run_7;
    (&kernelContext_72)->g_run_0 = &g_run_7;
    threadgroup array<uint, int(512)> g_label_narrow_7;
    (&kernelContext_72)->g_label_narrow_0 = &g_label_narrow_7;
    threadgroup uint g_changed_narrow_7;
    (&kernelContext_72)->g_changed_narrow_0 = &g_changed_narrow_7;
    uint tid_15 = thread_4.x;
    uint _S1089 = group_4.x;
    WideGroup_0 _S1090 = wide_group_0(params_7->wide_chunk_table_0, _S1089, &kernelContext_72);
    if(_S1089 != (_S1090.first_0))
    {
        return;
    }
    Island_natural_0 device* _S1091 = (&kernelContext_72)->islands_0+_S1090.island_3;
    thread Island_natural_0 _S1092 = *_S1091;
    uint4 _S1093 = uint4((&_S1092)->info_0) ;
    float4 _S1094 = float4((&_S1092)->com_0) ;
    float4 _S1095 = float4((&_S1092)->inertia0_0) ;
    float4 _S1096 = float4((&_S1092)->inertia1_0) ;
    float4 _S1097 = float4((&_S1092)->inertia2_0) ;
    float4 _S1098 = float4((&_S1092)->inv0_0) ;
    float4 _S1099 = float4((&_S1092)->inv1_0) ;
    float4 _S1100 = float4((&_S1092)->inv2_0) ;
    float4 _S1101 = float4((&_S1092)->wcom_0) ;
    float4 _S1102 = float4((&_S1092)->winv0_0) ;
    float4 _S1103 = float4((&_S1092)->winv1_0) ;
    float4 _S1104 = float4((&_S1092)->winv2_0) ;
    float4 _S1105 = float4((&_S1092)->rotation_0) ;
    float4 _S1106 = float4((&_S1092)->position_0) ;
    float4 _S1107 = float4((&_S1092)->position_err_0) ;
    float4 _S1108 = float4((&_S1092)->velocity_0) ;
    float4 _S1109 = float4((&_S1092)->velocity_err_0) ;
    float4 _S1110 = float4((&_S1092)->angular_velocity_0) ;
    uint4 _S1111 = uint4((&_S1092)->probes_0) ;
    float4 _S1112 = float4((&_S1092)->energy_0) ;
    float4 _S1113 = float4((&_S1092)->rotation_err_0) ;
    float4 _S1114 = float4((&_S1092)->momentum_0) ;
    float4 _S1115 = float4((&_S1092)->momentum_err_0) ;
    thread Island_0 isl_25;
    (&isl_25)->range_0 = uint4((&_S1092)->range_0) ;
    (&isl_25)->info_0 = _S1093;
    (&isl_25)->com_0 = _S1094;
    (&isl_25)->inertia0_0 = _S1095;
    (&isl_25)->inertia1_0 = _S1096;
    (&isl_25)->inertia2_0 = _S1097;
    (&isl_25)->inv0_0 = _S1098;
    (&isl_25)->inv1_0 = _S1099;
    (&isl_25)->inv2_0 = _S1100;
    (&isl_25)->wcom_0 = _S1101;
    (&isl_25)->winv0_0 = _S1102;
    (&isl_25)->winv1_0 = _S1103;
    (&isl_25)->winv2_0 = _S1104;
    (&isl_25)->rotation_0 = _S1105;
    (&isl_25)->position_0 = _S1106;
    (&isl_25)->position_err_0 = _S1107;
    (&isl_25)->velocity_0 = _S1108;
    (&isl_25)->velocity_err_0 = _S1109;
    (&isl_25)->angular_velocity_0 = _S1110;
    (&isl_25)->done_0 = (&_S1092)->done_0;
    (&isl_25)->coverage_0 = (&_S1092)->coverage_0;
    (&isl_25)->wide_groups_0 = (&_S1092)->wide_groups_0;
    (&isl_25)->done_pad_0 = (&_S1092)->done_pad_0;
    (&isl_25)->probes_0 = _S1111;
    (&isl_25)->energy_0 = _S1112;
    (&isl_25)->rotation_err_0 = _S1113;
    (&isl_25)->momentum_0 = _S1114;
    (&isl_25)->momentum_err_0 = _S1115;
    _S1092 = *_S1091;
    bool _S1116 = wide_enter_0(tid_15, &_S1092, &kernelContext_72);
    if(!_S1116)
    {
        return;
    }
    thread float3 work_4;
    thread float3 flags_2;
    wide_partials_0(tid_15, _S1090.first_0, (&isl_25)->wide_groups_0, 6U, &work_4, &flags_2, &kernelContext_72);
    bool halt_0 = (((&isl_25)->info_0.z) & 2U) != 0U;
    bool whole_0;
    if(halt_0)
    {
        thread Island_0 _S1117 = isl_25;
        bool _S1118 = island_connected_0(tid_15, &_S1117, &kernelContext_72);
        whole_0 = _S1118;
    }
    else
    {
        whole_0 = false;
    }
    if(tid_15 != 0U)
    {
        return;
    }
    bool halt_1;
    if(whole_0)
    {
        (&isl_25)->info_0.z = ((&isl_25)->info_0.z) & 4294967293U;
        (&isl_25)->coverage_0 = ((&isl_25)->coverage_0) | 2097280U;
        halt_1 = false;
    }
    else
    {
        halt_1 = halt_0;
    }
    thread Island_0 _S1119 = isl_25;
    uint _S1120 = wide_step_2(&_S1119, &kernelContext_72);
    if(((&isl_25)->probes_0.y) > ((&isl_25)->probes_0.x))
    {
        thread Island_0 _S1121 = isl_25;
        Rigid_0 _S1122 = rigid_of_0(&_S1121);
        thread Island_0 _S1123 = isl_25;
        thread Rigid_0 _S1124 = _S1122;
        record_probes_0(&_S1123, &_S1124, _S1120, &kernelContext_72);
    }
    if(halt_1)
    {
        whole_0 = (((&isl_25)->info_0.x) & 4U) != 0U;
    }
    else
    {
        whole_0 = false;
    }
    if(whole_0)
    {
        contact_split_at_0((&isl_25)->info_0.w + 1U, &kernelContext_72);
    }
    float _S1125 = work_4.x;
    thread float _S1126 = (&isl_25)->energy_0.x;
    thread float _S1127 = (&isl_25)->energy_0.y;
    comp_add1_0(&_S1126, &_S1127, _S1125);
    (&isl_25)->energy_0.x = _S1126;
    (&isl_25)->energy_0.y = _S1127 + work_4.y;
    (&isl_25)->done_0 = (&isl_25)->done_0 + 1U;
    (&isl_25)->info_0.y = (&isl_25)->info_0.y - 1U;
    (&isl_25)->info_0.w = (&isl_25)->info_0.w + 1U;
    if(halt_1)
    {
        (&isl_25)->info_0.z = (((&isl_25)->info_0.z) & 4294967293U) | 1U;
    }
    uint _S1128;
    if(halt_1)
    {
        _S1128 = 128U;
    }
    else
    {
        _S1128 = 0U;
    }
    uint _S1129 = 256U | _S1128;
    if(((&isl_25)->probes_0.y) > ((&isl_25)->probes_0.x))
    {
        _S1128 = 262144U;
    }
    else
    {
        _S1128 = 0U;
    }
    uint _S1130 = _S1129 | _S1128;
    if((((&isl_25)->info_0.x) & 4U) != 0U)
    {
        _S1128 = 524288U;
    }
    else
    {
        _S1128 = 0U;
    }
    uint _S1131 = _S1130 | _S1128;
    if((flags_2.x) > 0.0f)
    {
        _S1128 = 32768U;
    }
    else
    {
        _S1128 = 0U;
    }
    uint _S1132 = _S1131 | _S1128;
    if((flags_2.y) > 0.0f)
    {
        _S1128 = 65536U;
    }
    else
    {
        _S1128 = 0U;
    }
    uint _S1133 = _S1132 | _S1128;
    if((flags_2.z) > 0.0f)
    {
        _S1128 = 131072U;
    }
    else
    {
        _S1128 = 0U;
    }
    (&isl_25)->coverage_0 = ((&isl_25)->coverage_0) | (_S1133 | _S1128);
    Island_natural_0 device* _S1134 = (&kernelContext_72)->islands_0+_S1090.island_3;
    _S1134->range_0 = packed_uint4(isl_25.range_0) ;
    _S1134->info_0 = packed_uint4(isl_25.info_0) ;
    _S1134->com_0 = packed_float4(isl_25.com_0) ;
    _S1134->inertia0_0 = packed_float4(isl_25.inertia0_0) ;
    _S1134->inertia1_0 = packed_float4(isl_25.inertia1_0) ;
    _S1134->inertia2_0 = packed_float4(isl_25.inertia2_0) ;
    _S1134->inv0_0 = packed_float4(isl_25.inv0_0) ;
    _S1134->inv1_0 = packed_float4(isl_25.inv1_0) ;
    _S1134->inv2_0 = packed_float4(isl_25.inv2_0) ;
    _S1134->wcom_0 = packed_float4(isl_25.wcom_0) ;
    _S1134->winv0_0 = packed_float4(isl_25.winv0_0) ;
    _S1134->winv1_0 = packed_float4(isl_25.winv1_0) ;
    _S1134->winv2_0 = packed_float4(isl_25.winv2_0) ;
    _S1134->rotation_0 = packed_float4(isl_25.rotation_0) ;
    _S1134->position_0 = packed_float4(isl_25.position_0) ;
    _S1134->position_err_0 = packed_float4(isl_25.position_err_0) ;
    _S1134->velocity_0 = packed_float4(isl_25.velocity_0) ;
    _S1134->velocity_err_0 = packed_float4(isl_25.velocity_err_0) ;
    _S1134->angular_velocity_0 = packed_float4(isl_25.angular_velocity_0) ;
    _S1134->done_0 = isl_25.done_0;
    _S1134->coverage_0 = isl_25.coverage_0;
    _S1134->wide_groups_0 = isl_25.wide_groups_0;
    _S1134->done_pad_0 = isl_25.done_pad_0;
    _S1134->probes_0 = packed_uint4(isl_25.probes_0) ;
    _S1134->energy_0 = packed_float4(isl_25.energy_0) ;
    _S1134->rotation_err_0 = packed_float4(isl_25.rotation_err_0) ;
    _S1134->momentum_0 = packed_float4(isl_25.momentum_0) ;
    _S1134->momentum_err_0 = packed_float4(isl_25.momentum_err_0) ;
    return;
}

uint sv_0(uint c_28, uint slot_3, KernelContext_0 thread* kernelContext_73)
{
    return kernelContext_73->params_0->statics_base_0 + 23U * c_28 + slot_3;
}

void project_load_slot_0(uint tid_16, const Island_natural_0 thread* isl_26, uint slot_4, KernelContext_0 thread* kernelContext_74)
{
    uint _S1135;
    float3 _S1136 = float3(0.0f) ;
    thread float3 net_f_0 = _S1136;
    thread float3 net_m_0 = _S1136;
    uint4 _S1137 = uint4(isl_26->range_0) ;
    uint _S1138 = _S1137.x + tid_16;
    uint c_29 = _S1138;
    for(;;)
    {
        uint _S1139 = _S1137.y;
        _S1135 = _S1139;
        if(c_29 < _S1139)
        {
        }
        else
        {
            break;
        }
        uint _S1140 = sv_0(c_29, slot_4, kernelContext_74);
        float3 fi_1 = (float4(*(kernelContext_74->scratch_0+_S1140)) ).xyz;
        net_f_0 = net_f_0 + fi_1;
        float3 _S1141 = cross((float4((kernelContext_74->chunks_0+c_29)->center_0) ).xyz - (float4(isl_26->com_0) ).xyz, fi_1);
        uint _S1142 = sv_0(c_29, slot_4 + 1U, kernelContext_74);
        net_m_0 = net_m_0 + (_S1141 + (float4(*(kernelContext_74->scratch_0+_S1142)) ).xyz);
        c_29 = c_29 + 256U;
    }
    group_sum3_0(tid_16, &net_f_0, &net_m_0, kernelContext_74);
    float4 _S1143 = float4(isl_26->com_0) ;
    float3 _S1144 = net_f_0 / float3(_S1143.w) ;
    float3 _S1145 = rows_mul_0(float4(isl_26->inv0_0) , float4(isl_26->inv1_0) , float4(isl_26->inv2_0) , net_m_0);
    c_29 = _S1138;
    for(;;)
    {
        if(c_29 < _S1135)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S1146 = kernelContext_74->chunks_0+c_29;
        float4 _S1147 = float4(_S1146->center_0) ;
        float3 r_12 = _S1147.xyz - _S1143.xyz;
        uint _S1148 = sv_0(c_29, slot_4, kernelContext_74);
        *(kernelContext_74->scratch_0+_S1148) = packed_float4(float4((float4(*(kernelContext_74->scratch_0+_S1148)) ).xyz - (_S1144 + cross(_S1145, r_12)) * float3(_S1147.w) , 0.0f)) ;
        uint _S1149 = sv_0(c_29, slot_4 + 1U, kernelContext_74);
        *(kernelContext_74->scratch_0+_S1149) = packed_float4(float4((float4(*(kernelContext_74->scratch_0+_S1149)) ).xyz - rows_mul_0(float4(_S1146->inertia0_1) , float4(_S1146->inertia1_1) , float4(_S1146->inertia2_1) , _S1145), 0.0f)) ;
        c_29 = c_29 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    return;
}

float island_dot_0(uint tid_17, uint c0_0, uint c1_0, uint sa_2, uint sb_2, KernelContext_0 thread* kernelContext_75)
{
    float4 _S1150 = float4(0.0f) ;
    thread float4 acc_0 = _S1150;
    thread float4 unused_4 = _S1150;
    uint c_30 = c0_0 + tid_17;
    for(;;)
    {
        if(c_30 < c1_0)
        {
        }
        else
        {
            break;
        }
        uint _S1151 = sv_0(c_30, sa_2, kernelContext_75);
        float3 _S1152 = (float4(*(kernelContext_75->scratch_0+_S1151)) ).xyz;
        uint _S1153 = sv_0(c_30, sb_2, kernelContext_75);
        float _S1154 = dot(_S1152, (float4(*(kernelContext_75->scratch_0+_S1153)) ).xyz);
        uint _S1155 = sv_0(c_30, sa_2 + 1U, kernelContext_75);
        float3 _S1156 = (float4(*(kernelContext_75->scratch_0+_S1155)) ).xyz;
        uint _S1157 = sv_0(c_30, sb_2 + 1U, kernelContext_75);
        acc_0.x = acc_0.x + (_S1154 + dot(_S1156, (float4(*(kernelContext_75->scratch_0+_S1157)) ).xyz));
        c_30 = c_30 + 256U;
    }
    group_sum2_0(tid_17, &acc_0, &unused_4, kernelContext_75);
    return acc_0.x;
}

void static_kinematics_0(uint _S1158, float3 thread* _S1159, float3 thread* _S1160, KernelContext_0 thread* kernelContext_76)
{
    BondStatic_natural_0 device* _S1161 = kernelContext_76->bonds_0+_S1158;
    uint4 _S1162 = uint4(_S1161->law_0.ids_0) ;
    uint ca_1 = _S1162.y;
    uint cb_1 = _S1162.z;
    uint _S1163 = 4U * cb_1;
    uint _S1164 = 4U * ca_1;
    float3 _S1165 = (float4(*(kernelContext_76->state_0+_S1163)) ).xyz - (float4(*(kernelContext_76->state_0+_S1164)) ).xyz;
    uint _S1166 = sv_0(cb_1, 21U, kernelContext_76);
    float3 _S1167 = (float4(*(kernelContext_76->scratch_0+_S1166)) ).xyz;
    uint _S1168 = sv_0(ca_1, 21U, kernelContext_76);
    float3 du_0 = _S1165 + (_S1167 - (float4(*(kernelContext_76->scratch_0+_S1168)) ).xyz);
    uint _S1169 = _S1163 + 1U;
    uint _S1170 = _S1164 + 1U;
    float3 _S1171 = (float4(*(kernelContext_76->state_0+_S1169)) ).xyz - (float4(*(kernelContext_76->state_0+_S1170)) ).xyz;
    uint _S1172 = sv_0(cb_1, 22U, kernelContext_76);
    float3 _S1173 = (float4(*(kernelContext_76->scratch_0+_S1172)) ).xyz;
    uint _S1174 = sv_0(ca_1, 22U, kernelContext_76);
    float3 dth_0 = _S1171 + (_S1173 - (float4(*(kernelContext_76->scratch_0+_S1174)) ).xyz);
    float3 _S1175 = to_local_0(_S1158, du_0 + (cross((float4(*(kernelContext_76->state_0+_S1169)) ).xyz + (float4(*(kernelContext_76->scratch_0+_S1172)) ).xyz, (float4(_S1161->rb_0) ).xyz) - cross((float4(*(kernelContext_76->state_0+_S1170)) ).xyz + (float4(*(kernelContext_76->scratch_0+_S1174)) ).xyz, (float4(_S1161->ra_0) ).xyz)), kernelContext_76);
    *_S1159 = _S1175;
    float3 _S1176 = to_local_0(_S1158, dth_0, kernelContext_76);
    *_S1160 = _S1176;
    return;
}

JointResponse_0 static_response_0(uint i_12, KernelContext_0 thread* kernelContext_77)
{
    BondStatic_natural_0 device* _S1177 = kernelContext_77->bonds_0+i_12;
    thread float3 d_lin_4;
    thread float3 d_ang_3;
    static_kinematics_0(i_12, &d_lin_4, &d_ang_3, kernelContext_77);
    thread JointBond_natural_0 _S1178 = _S1177->law_0;
    _S1178 = _S1177->law_0;
    thread JointState_0 _S1179 = (kernelContext_77->bond_dyn_0+i_12)->js_0;
    JointResponse_0 _S1180 = joint_evaluate_0(&kernelContext_77->materials_0->m_0[(uint4((&_S1178)->ids_0) ).x], &_S1178, &_S1179, d_lin_4, d_ang_3, 0.0f, false);
    return _S1180;
}

void gather_loads_0(uint c_31, float3 thread* fi_2, float3 thread* mi_3, KernelContext_0 thread* kernelContext_78)
{
    float3 _S1181 = float3(0.0f) ;
    *fi_2 = _S1181;
    *mi_3 = _S1181;
    uint e_5 = kernelContext_78->index_0[c_31];
    for(;;)
    {
        if(e_5 < (kernelContext_78->index_0)[c_31 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_4 = kernelContext_78->index_0[e_5];
        uint bond_2 = entry_4 >> 1U;
        if((entry_4 & 1U) == 0U)
        {
            uint _S1182 = 3U * bond_2;
            *fi_2 = *fi_2 + (float4(*(kernelContext_78->scratch_0+_S1182)) ).xyz;
            *mi_3 = *mi_3 + (float4(*(kernelContext_78->scratch_0+(_S1182 + 1U))) ).xyz;
        }
        else
        {
            uint _S1183 = 3U * bond_2;
            *fi_2 = *fi_2 - (float4(*(kernelContext_78->scratch_0+_S1183)) ).xyz;
            *mi_3 = *mi_3 + (float4(*(kernelContext_78->scratch_0+(_S1183 + 2U))) ).xyz;
        }
        e_5 = e_5 + 1U;
    }
    return;
}

float bond_load_magnitude2_0(uint c_32, KernelContext_0 thread* kernelContext_79)
{
    uint e_6 = kernelContext_79->index_0[c_32];
    float m_5 = 0.0f;
    for(;;)
    {
        if(e_6 < (kernelContext_79->index_0)[c_32 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_5 = kernelContext_79->index_0[e_6];
        uint _S1184 = 3U * (entry_5 >> 1U);
        float3 f_15 = (float4(*(kernelContext_79->scratch_0+_S1184)) ).xyz;
        float3 t_16;
        if((entry_5 & 1U) == 0U)
        {
            t_16 = (float4(*(kernelContext_79->scratch_0+(_S1184 + 1U))) ).xyz;
        }
        else
        {
            t_16 = (float4(*(kernelContext_79->scratch_0+(_S1184 + 2U))) ).xyz;
        }
        float m_6 = m_5 + (dot(f_15, f_15) + dot(t_16, t_16));
        e_6 = e_6 + 1U;
        m_5 = m_6;
    }
    return m_5;
}

uint fixed_mask_0(uint c_33, KernelContext_0 thread* kernelContext_80)
{
    uint support_1 = (uint4((kernelContext_80->chunks_0+c_33)->info_1) ).x;
    uint _S1185;
    if(support_1 == 1U)
    {
        _S1185 = 63U;
    }
    else
    {
        if(support_1 == 2U)
        {
            _S1185 = 7U;
        }
        else
        {
            _S1185 = 0U;
        }
    }
    return _S1185;
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

uint statics_bond_slot_0(uint i_13, KernelContext_0 thread* kernelContext_81)
{
    return kernelContext_81->params_0->statics_base_0 + 23U * kernelContext_81->params_0->chunk_count_0 + 2U * i_13;
}

void store_inverse_0(uint c_34, const array<float, int(36)> thread* a_15, KernelContext_0 thread* kernelContext_82)
{
    uint j_9;
    float sum_4;
    thread array<float, int(36)> l_7;
    uint k_19 = 0U;
    for(;;)
    {
        if(k_19 < 36U)
        {
        }
        else
        {
            break;
        }
        l_7[k_19] = 0.0f;
        k_19 = k_19 + 1U;
    }
    bool spd_0 = true;
    uint i_14 = 0U;
    for(;;)
    {
        bool _S1186;
        if(i_14 < 6U)
        {
            _S1186 = spd_0;
        }
        else
        {
            _S1186 = false;
        }
        if(_S1186)
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
            uint _S1187 = i_14 * 6U;
            uint _S1188 = _S1187 + j_9;
            k_19 = 0U;
            sum_4 = (*a_15)[_S1188];
            for(;;)
            {
                if(k_19 < j_9)
                {
                }
                else
                {
                    break;
                }
                float sum_5 = sum_4 - l_7[_S1187 + k_19] * l_7[j_9 * 6U + k_19];
                k_19 = k_19 + 1U;
                sum_4 = sum_5;
            }
            if(i_14 == j_9)
            {
                if(sum_4 <= 0.0f)
                {
                    spd_0 = false;
                    break;
                }
                l_7[_S1187 + i_14] = sqrt(sum_4);
            }
            else
            {
                l_7[_S1188] = sum_4 / l_7[j_9 * 6U + j_9];
            }
            j_9 = j_9 + 1U;
        }
        i_14 = i_14 + 1U;
    }
    thread array<float, int(36)> inv_0;
    if(!spd_0)
    {
        k_19 = 0U;
        for(;;)
        {
            if(k_19 < 36U)
            {
            }
            else
            {
                break;
            }
            inv_0[k_19] = 0.0f;
            k_19 = k_19 + 1U;
        }
        k_19 = 0U;
        for(;;)
        {
            if(k_19 < 6U)
            {
            }
            else
            {
                break;
            }
            uint _S1189 = k_19 * 6U + k_19;
            float _S1190 = (*a_15)[_S1189];
            if(((*a_15)[_S1189]) > 0.0f)
            {
                sum_4 = 1.0f / _S1190;
            }
            else
            {
                sum_4 = 0.0f;
            }
            inv_0[_S1189] = sum_4;
            k_19 = k_19 + 1U;
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
                k_19 = 0U;
                float s_6 = sum_4;
                for(;;)
                {
                    if(k_19 < i_14)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_7 = s_6 - l_7[i_14 * 6U + k_19] * y_3[k_19];
                    k_19 = k_19 + 1U;
                    s_6 = s_7;
                }
                y_3[i_14] = s_6 / l_7[i_14 * 6U + i_14];
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
                k_19 = i_15 + 1U;
                sum_4 = y_3[i_15];
                for(;;)
                {
                    if(k_19 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float s_8 = sum_4 - l_7[k_19 * 6U + i_15] * x_16[k_19];
                    k_19 = k_19 + 1U;
                    sum_4 = s_8;
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
        uint _S1191 = sv_0(c_34, 12U + j_9, kernelContext_82);
        uint _S1192 = 4U * j_9;
        *(kernelContext_82->scratch_0+_S1191) = packed_float4(float4(inv_0[_S1192], inv_0[_S1192 + 1U], inv_0[_S1192 + 2U], inv_0[_S1192 + 3U])) ;
        j_9 = j_9 + 1U;
    }
    return;
}

void assemble_block_0(uint c_35, KernelContext_0 thread* kernelContext_83)
{
    uint p_16;
    uint r_13;
    thread array<float, int(36)> a_16;
    uint k_20 = 0U;
    for(;;)
    {
        if(k_20 < 36U)
        {
        }
        else
        {
            break;
        }
        a_16[k_20] = 0.0f;
        k_20 = k_20 + 1U;
    }
    uint e_7 = kernelContext_83->index_0[c_35];
    for(;;)
    {
        if(e_7 < (kernelContext_83->index_0)[c_35 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_6 = kernelContext_83->index_0[e_7];
        uint i_17 = entry_6 >> 1U;
        bool _S1193 = (entry_6 & 1U) != 0U;
        BondStatic_natural_0 device* _S1194 = kernelContext_83->bonds_0+i_17;
        uint _S1195 = statics_bond_slot_0(i_17, kernelContext_83);
        float4 _S1196 = float4(*(kernelContext_83->scratch_0+_S1195)) ;
        float4 _S1197 = float4(*(kernelContext_83->scratch_0+(_S1195 + 1U))) ;
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
            uint _S1198 = p_16 % 3U;
            float3 t_17;
            if(_S1198 == 0U)
            {
                t_17 = (float4(_S1194->t1_0) ).xyz;
            }
            else
            {
                if(_S1198 == 1U)
                {
                    t_17 = (float4(_S1194->t2_0) ).xyz;
                }
                else
                {
                    t_17 = (float4(_S1194->normal_0) ).xyz;
                }
            }
            bool _S1199 = p_16 < 3U;
            float3 row_u_0;
            float3 row_t_0;
            if(_S1199)
            {
                if(_S1193)
                {
                    row_u_0 = t_17;
                }
                else
                {
                    row_u_0 = - t_17;
                }
                if(_S1193)
                {
                    row_t_0 = cross((float4(_S1194->rb_0) ).xyz, t_17);
                }
                else
                {
                    row_t_0 = - cross((float4(_S1194->ra_0) ).xyz, t_17);
                }
            }
            else
            {
                float3 _S1200 = float3(0.0f) ;
                if(_S1193)
                {
                    row_u_0 = t_17;
                }
                else
                {
                    row_u_0 = - t_17;
                }
                float3 _S1201 = row_u_0;
                row_u_0 = _S1200;
                row_t_0 = _S1201;
            }
            float kp_0;
            if(_S1199)
            {
                kp_0 = _S1196[p_16];
            }
            else
            {
                kp_0 = _S1197[p_16 - 3U];
            }
            if(kp_0 == 0.0f)
            {
                p_16 = p_16 + 1U;
                continue;
            }
            array<float, int(6)> _S1202 = { { row_u_0.x, row_u_0.y, row_u_0.z, row_t_0.x, row_t_0.y, row_t_0.z } };
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
                    a_16[r_13 * 6U + q_14] = a_16[r_13 * 6U + q_14] + kp_0 * _S1202[r_13] * _S1202[q_14];
                    q_14 = q_14 + 1U;
                }
                r_13 = r_13 + 1U;
            }
            p_16 = p_16 + 1U;
        }
        e_7 = e_7 + 1U;
    }
    uint _S1203 = fixed_mask_0(c_35, kernelContext_83);
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
        if((_S1203 & (1U << p_16)) != 0U)
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
    thread array<float, int(36)> _S1204 = a_16;
    store_inverse_0(c_35, &_S1204, kernelContext_83);
    return;
}

float block_get_0(uint c_36, uint i_18, uint j_10, KernelContext_0 thread* kernelContext_84)
{
    uint k_21 = i_18 * 6U + j_10;
    uint _S1205 = sv_0(c_36, 12U + k_21 / 4U, kernelContext_84);
    return (*(kernelContext_84->scratch_0+_S1205))[k_21 % 4U];
}

void precondition_0(uint c_37, KernelContext_0 thread* kernelContext_85)
{
    uint _S1206 = sv_0(c_37, 4U, kernelContext_85);
    float4 _S1207 = float4(*(kernelContext_85->scratch_0+_S1206)) ;
    uint _S1208 = sv_0(c_37, 5U, kernelContext_85);
    float4 _S1209 = float4(*(kernelContext_85->scratch_0+_S1208)) ;
    array<float, int(6)> _S1210 = { { _S1207.x, _S1207.y, _S1207.z, _S1209.x, _S1209.y, _S1209.z } };
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
        float s_9 = 0.0f;
        for(;;)
        {
            if(j_11 < 6U)
            {
            }
            else
            {
                break;
            }
            float _S1211 = block_get_0(c_37, i_19, j_11, kernelContext_85);
            float s_10 = s_9 + _S1211 * _S1210[j_11];
            j_11 = j_11 + 1U;
            s_9 = s_10;
        }
        z_1[i_19] = s_9;
        i_19 = i_19 + 1U;
    }
    uint _S1212 = sv_0(c_37, 6U, kernelContext_85);
    *(kernelContext_85->scratch_0+_S1212) = packed_float4(float4(z_1[int(0)], z_1[int(1)], z_1[int(2)], 0.0f)) ;
    uint _S1213 = sv_0(c_37, 7U, kernelContext_85);
    *(kernelContext_85->scratch_0+_S1213) = packed_float4(float4(z_1[int(3)], z_1[int(4)], z_1[int(5)], 0.0f)) ;
    return;
}

void project_displacement_slot_0(uint tid_18, const Island_natural_0 thread* isl_27, uint slot_5, KernelContext_0 thread* kernelContext_86)
{
    uint _S1214;
    float3 _S1215 = float3(0.0f) ;
    thread float3 p_17 = _S1215;
    thread float3 l_8 = _S1215;
    uint4 _S1216 = uint4(isl_27->range_0) ;
    uint _S1217 = _S1216.x + tid_18;
    uint c_38 = _S1217;
    for(;;)
    {
        uint _S1218 = _S1216.y;
        _S1214 = _S1218;
        if(c_38 < _S1218)
        {
        }
        else
        {
            break;
        }
        ChunkStatic_natural_0 device* _S1219 = kernelContext_86->chunks_0+c_38;
        uint _S1220 = sv_0(c_38, slot_5, kernelContext_86);
        float3 u_3 = (float4(*(kernelContext_86->scratch_0+_S1220)) ).xyz;
        uint _S1221 = sv_0(c_38, slot_5 + 1U, kernelContext_86);
        float3 th_5 = (float4(*(kernelContext_86->scratch_0+_S1221)) ).xyz;
        float4 _S1222 = float4(_S1219->center_0) ;
        float3 r_14 = _S1222.xyz - (float4(isl_27->com_0) ).xyz;
        float3 _S1223 = float3(_S1222.w) ;
        p_17 = p_17 + u_3 * _S1223;
        l_8 = l_8 + (cross(r_14, u_3) * _S1223 + rows_mul_0(float4(_S1219->inertia0_1) , float4(_S1219->inertia1_1) , float4(_S1219->inertia2_1) , th_5));
        c_38 = c_38 + 256U;
    }
    group_sum3_0(tid_18, &p_17, &l_8, kernelContext_86);
    float4 _S1224 = float4(isl_27->com_0) ;
    float3 _S1225 = p_17 / float3(_S1224.w) ;
    float3 _S1226 = rows_mul_0(float4(isl_27->inv0_0) , float4(isl_27->inv1_0) , float4(isl_27->inv2_0) , l_8);
    c_38 = _S1217;
    for(;;)
    {
        if(c_38 < _S1214)
        {
        }
        else
        {
            break;
        }
        float3 r_15 = (float4((kernelContext_86->chunks_0+c_38)->center_0) ).xyz - _S1224.xyz;
        uint _S1227 = sv_0(c_38, slot_5, kernelContext_86);
        *(kernelContext_86->scratch_0+_S1227) = packed_float4(float4((float4(*(kernelContext_86->scratch_0+_S1227)) ).xyz - _S1225 - cross(_S1226, r_15), 0.0f)) ;
        uint _S1228 = sv_0(c_38, slot_5 + 1U, kernelContext_86);
        *(kernelContext_86->scratch_0+_S1228) = packed_float4(float4((float4(*(kernelContext_86->scratch_0+_S1228)) ).xyz - _S1226, 0.0f)) ;
        c_38 = c_38 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    return;
}

uint statics_result_slot_0(uint island_4, KernelContext_0 thread* kernelContext_87)
{
    return kernelContext_87->params_0->statics_base_0 + 23U * kernelContext_87->params_0->chunk_count_0 + 2U * kernelContext_87->params_0->statics_bonds_0 + island_4;
}

void write_bond_loads_0(uint _S1229, uint _S1230, float3 _S1231, float3 _S1232, float _S1233, KernelContext_0 thread* kernelContext_88)
{
    BondStatic_natural_0 device* _S1234 = kernelContext_88->bonds_0+_S1230;
    float3 _S1235 = to_body_0(_S1230, _S1231, kernelContext_88);
    float3 _S1236 = to_body_0(_S1230, _S1232, kernelContext_88);
    uint _S1237 = 3U * _S1229;
    *(kernelContext_88->scratch_0+_S1237) = packed_float4(float4(_S1235, _S1233)) ;
    *(kernelContext_88->scratch_0+(_S1237 + 1U)) = packed_float4(float4(_S1236 + cross((float4(_S1234->ra_0) ).xyz, _S1235), 0.0f)) ;
    *(kernelContext_88->scratch_0+(_S1237 + 2U)) = packed_float4(float4(- _S1236 + cross((float4(_S1234->rb_0) ).xyz, - _S1235), 0.0f)) ;
    return;
}

void bond_kinematics_0(uint _S1238, float3 _S1239, float3 _S1240, float3 _S1241, float3 _S1242, float3 thread* _S1243, float3 thread* _S1244, KernelContext_0 thread* kernelContext_89)
{
    BondStatic_natural_0 device* _S1245 = kernelContext_89->bonds_0+_S1238;
    float3 _S1246 = to_local_0(_S1238, _S1241 + cross(_S1242, (float4(_S1245->rb_0) ).xyz) - (_S1239 + cross(_S1240, (float4(_S1245->ra_0) ).xyz)), kernelContext_89);
    *_S1243 = _S1246;
    float3 _S1247 = to_local_0(_S1238, _S1242 - _S1240, kernelContext_89);
    *_S1244 = _S1247;
    return;
}

[[kernel]] void island_statics(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], Params_0 constant* params_8 [[buffer(0)]], Island_natural_0 device* islands_8 [[buffer(9)]], uint device* index_8 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(3)]], packed_float4 device* state_10 [[buffer(6)]], packed_float4 device* scratch_8 [[buffer(8)]], packed_float4 device* contact_state_8 [[buffer(11)]], packed_float4 device* loads_8 [[buffer(5)]], Impactor_natural_0 device* impactors_8 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(7)]], BondStatic_natural_0 device* bonds_8 [[buffer(2)]], MaterialTable_0 constant* materials_8 [[buffer(1)]])
{
    uint i_20;
    bool converged_4;
    uint c_39;
    thread KernelContext_0 kernelContext_90;
    (&kernelContext_90)->params_0 = params_8;
    (&kernelContext_90)->islands_0 = islands_8;
    (&kernelContext_90)->index_0 = index_8;
    (&kernelContext_90)->chunks_0 = chunks_8;
    (&kernelContext_90)->state_0 = state_10;
    (&kernelContext_90)->scratch_0 = scratch_8;
    (&kernelContext_90)->contact_state_0 = contact_state_8;
    (&kernelContext_90)->loads_0 = loads_8;
    (&kernelContext_90)->impactors_0 = impactors_8;
    (&kernelContext_90)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_90)->bonds_0 = bonds_8;
    (&kernelContext_90)->materials_0 = materials_8;
    threadgroup array<float4, int(32)> g_part_a_8;
    (&kernelContext_90)->g_part_a_0 = &g_part_a_8;
    threadgroup array<float4, int(32)> g_part_b_8;
    (&kernelContext_90)->g_part_b_0 = &g_part_b_8;
    threadgroup array<uint, int(5)> g_seen_8;
    (&kernelContext_90)->g_seen_0 = &g_seen_8;
    threadgroup uint g_wide_run_8;
    (&kernelContext_90)->g_wide_run_0 = &g_wide_run_8;
    threadgroup array<uint, int(4096)> g_label_8;
    (&kernelContext_90)->g_label_0 = &g_label_8;
    threadgroup uint g_changed_8;
    (&kernelContext_90)->g_changed_0 = &g_changed_8;
    threadgroup uint g_halt_8;
    (&kernelContext_90)->g_halt_0 = &g_halt_8;
    threadgroup uint g_run_8;
    (&kernelContext_90)->g_run_0 = &g_run_8;
    threadgroup array<uint, int(512)> g_label_narrow_8;
    (&kernelContext_90)->g_label_narrow_0 = &g_label_narrow_8;
    threadgroup uint g_changed_narrow_8;
    (&kernelContext_90)->g_changed_narrow_0 = &g_changed_narrow_8;
    uint tid_19 = thread_5.x;
    uint _S1248 = group_5.x;
    thread Island_natural_0 _S1249 = *(islands_8+_S1248);
    uint4 _S1250 = uint4((&_S1249)->info_0) ;
    uint _S1251 = _S1250.z;
    if((_S1251 & 8U) == 0U)
    {
        return;
    }
    bool free_0 = ((_S1250.x) & 1U) == 0U;
    uint4 _S1252 = uint4((&_S1249)->range_0) ;
    uint c0_1 = _S1252.x;
    uint c1_1 = _S1252.y;
    uint b0_0 = _S1252.z;
    uint _S1253 = _S1252.w;
    uint _S1254 = c0_1 + tid_19;
    uint c_40 = _S1254;
    for(;;)
    {
        if(c_40 < c1_1)
        {
        }
        else
        {
            break;
        }
        uint _S1255 = sv_0(c_40, 21U, &kernelContext_90);
        packed_float4 _S1256 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_90)->scratch_0+_S1255) = _S1256;
        uint _S1257 = sv_0(c_40, 22U, &kernelContext_90);
        *((&kernelContext_90)->scratch_0+_S1257) = _S1256;
        c_40 = c_40 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    if(free_0)
    {
        project_load_slot_0(tid_19, &_S1249, 0U, &kernelContext_90);
    }
    float _S1258 = island_dot_0(tid_19, c0_1, c1_1, 0U, 0U, &kernelContext_90);
    float _S1259 = max(sqrt(_S1258), 1.00000000317107685e-30f);
    float _S1260 = (&kernelContext_90)->params_0->statics_tol_0;
    uint _S1261 = min((&kernelContext_90)->params_0->statics_cg_0, 20U * (c1_1 - c0_1) * 6U + 200U);
    float previous_2 = 1.00000001504746622e+30f;
    float start_0 = 1.00000001504746622e+30f;
    float residual_0 = 0.0f;
    uint newton_0 = 0U;
    uint cg_total_0 = 0U;
    for(;;)
    {
        if(newton_0 < ((&kernelContext_90)->params_0->statics_newton_0))
        {
        }
        else
        {
            converged_4 = false;
            break;
        }
        uint _S1262 = b0_0 + tid_19;
        i_20 = _S1262;
        for(;;)
        {
            if(i_20 < _S1253)
            {
            }
            else
            {
                break;
            }
            JointResponse_0 _S1263 = static_response_0(i_20, &kernelContext_90);
            write_bond_loads_0(i_20, i_20, _S1263.force_lin_1, _S1263.force_ang_1, 0.0f, &kernelContext_90);
            i_20 = i_20 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float4 _S1264 = float4(0.0f) ;
        thread float4 magnitude_0 = _S1264;
        thread float4 unused_m_0 = _S1264;
        c_40 = _S1254;
        for(;;)
        {
            if(c_40 < c1_1)
            {
            }
            else
            {
                break;
            }
            thread float3 fi_3;
            thread float3 mi_4;
            gather_loads_0(c_40, &fi_3, &mi_4, &kernelContext_90);
            float _S1265 = bond_load_magnitude2_0(c_40, &kernelContext_90);
            magnitude_0.x = magnitude_0.x + _S1265;
            uint _S1266 = sv_0(c_40, 0U, &kernelContext_90);
            thread float4 r_lin_0 = float4((float4(*((&kernelContext_90)->scratch_0+_S1266)) ).xyz + fi_3, 0.0f);
            uint _S1267 = sv_0(c_40, 1U, &kernelContext_90);
            thread float4 r_ang_0 = float4((float4(*((&kernelContext_90)->scratch_0+_S1267)) ).xyz + mi_4, 0.0f);
            uint _S1268 = fixed_mask_0(c_40, &kernelContext_90);
            hold_0(_S1268, &r_lin_0, &r_ang_0, _S1264, _S1264);
            uint _S1269 = sv_0(c_40, 4U, &kernelContext_90);
            *((&kernelContext_90)->scratch_0+_S1269) = packed_float4(r_lin_0) ;
            uint _S1270 = sv_0(c_40, 5U, &kernelContext_90);
            *((&kernelContext_90)->scratch_0+_S1270) = packed_float4(r_ang_0) ;
            c_40 = c_40 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        group_sum2_0(tid_19, &magnitude_0, &unused_m_0, &kernelContext_90);
        if(free_0)
        {
            project_load_slot_0(tid_19, &_S1249, 4U, &kernelContext_90);
        }
        float _S1271 = island_dot_0(tid_19, c0_1, c1_1, 4U, 4U, &kernelContext_90);
        float residual_1 = sqrt(_S1271) / _S1259;
        float _S1272 = max(0.00100000004749745f, 3.83999986297567375e-06f * sqrt(magnitude_0.x) / _S1259);
        if(newton_0 == 0U)
        {
            start_0 = residual_1;
        }
        bool _S1273 = is_finite_0(residual_1);
        bool _S1274;
        if(_S1273)
        {
            if(residual_1 <= _S1260)
            {
                converged_4 = true;
            }
            else
            {
                if(residual_1 <= _S1272)
                {
                    converged_4 = residual_1 > (0.5f * previous_2);
                }
                else
                {
                    converged_4 = false;
                }
                if(converged_4)
                {
                    _S1274 = residual_1 <= start_0;
                }
                else
                {
                    _S1274 = false;
                }
                converged_4 = _S1274;
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
        if(!_S1273)
        {
            residual_0 = residual_1;
            converged_4 = false;
            break;
        }
        uint i_21 = _S1262;
        for(;;)
        {
            if(i_21 < _S1253)
            {
            }
            else
            {
                break;
            }
            BondStatic_natural_0 device* _S1275 = (&kernelContext_90)->bonds_0+i_21;
            thread float3 d_lin_5;
            thread float3 d_ang_4;
            static_kinematics_0(i_21, &d_lin_5, &d_ang_4, &kernelContext_90);
            thread JointBond_natural_0 _S1276 = _S1275->law_0;
            thread JointState_0 _S1277 = ((&kernelContext_90)->bond_dyn_0+i_21)->js_0;
            thread float3 f_lin_2;
            thread float3 f_ang_2;
            secant_factors_0(&_S1276, &_S1277, d_lin_5, &f_lin_2, &f_ang_2);
            float4 _S1278 = float4((&_S1276)->stiff0_0) ;
            uint _S1279 = statics_bond_slot_0(i_21, &kernelContext_90);
            float _S1280 = _S1278.y;
            *((&kernelContext_90)->scratch_0+_S1279) = packed_float4(float4(_S1280 * f_lin_2.x, _S1280 * f_lin_2.y, _S1278.x * f_lin_2.z, 0.0f)) ;
            *((&kernelContext_90)->scratch_0+(_S1279 + 1U)) = packed_float4(float4(_S1278.z * f_ang_2.x, _S1278.w * f_ang_2.y, (float4((&_S1276)->stiff1_0) ).x * f_ang_2.z, 0.0f)) ;
            i_21 = i_21 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint c_41 = _S1254;
        for(;;)
        {
            if(c_41 < c1_1)
            {
            }
            else
            {
                break;
            }
            assemble_block_0(c_41, &kernelContext_90);
            uint _S1281 = sv_0(c_41, 2U, &kernelContext_90);
            packed_float4 _S1282 = packed_float4(_S1264) ;
            *((&kernelContext_90)->scratch_0+_S1281) = _S1282;
            uint _S1283 = sv_0(c_41, 3U, &kernelContext_90);
            *((&kernelContext_90)->scratch_0+_S1283) = _S1282;
            c_41 = c_41 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint c_42 = _S1254;
        for(;;)
        {
            if(c_42 < c1_1)
            {
            }
            else
            {
                break;
            }
            precondition_0(c_42, &kernelContext_90);
            c_42 = c_42 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if(free_0)
        {
            project_displacement_slot_0(tid_19, &_S1249, 6U, &kernelContext_90);
        }
        uint c_43 = _S1254;
        for(;;)
        {
            if(c_43 < c1_1)
            {
            }
            else
            {
                break;
            }
            uint _S1284 = sv_0(c_43, 8U, &kernelContext_90);
            packed_float4 device* _S1285 = (&kernelContext_90)->scratch_0+_S1284;
            uint _S1286 = sv_0(c_43, 6U, &kernelContext_90);
            *_S1285 = packed_float4(float4(*((&kernelContext_90)->scratch_0+_S1286)) ) ;
            uint _S1287 = sv_0(c_43, 9U, &kernelContext_90);
            packed_float4 device* _S1288 = (&kernelContext_90)->scratch_0+_S1287;
            uint _S1289 = sv_0(c_43, 7U, &kernelContext_90);
            *_S1288 = packed_float4(float4(*((&kernelContext_90)->scratch_0+_S1289)) ) ;
            c_43 = c_43 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float _S1290 = island_dot_0(tid_19, c0_1, c1_1, 4U, 6U, &kernelContext_90);
        float _S1291 = island_dot_0(tid_19, c0_1, c1_1, 4U, 4U, &kernelContext_90);
        float _S1292 = sqrt(_S1291);
        float rz_0 = _S1290;
        uint k_22 = 0U;
        uint cg_total_1 = cg_total_0;
        for(;;)
        {
            if(k_22 < _S1261)
            {
                _S1274 = _S1292 > 0.0f;
            }
            else
            {
                _S1274 = false;
            }
            if(_S1274)
            {
            }
            else
            {
                cg_total_0 = cg_total_1;
                break;
            }
            uint i_22 = _S1262;
            for(;;)
            {
                if(i_22 < _S1253)
                {
                }
                else
                {
                    break;
                }
                uint4 _S1293 = uint4(((&kernelContext_90)->bonds_0+i_22)->law_0.ids_0) ;
                uint ca_2 = _S1293.y;
                uint cb_2 = _S1293.z;
                uint _S1294 = sv_0(ca_2, 8U, &kernelContext_90);
                float3 _S1295 = (float4(*((&kernelContext_90)->scratch_0+_S1294)) ).xyz;
                uint _S1296 = sv_0(ca_2, 9U, &kernelContext_90);
                float3 _S1297 = (float4(*((&kernelContext_90)->scratch_0+_S1296)) ).xyz;
                uint _S1298 = sv_0(cb_2, 8U, &kernelContext_90);
                float3 _S1299 = (float4(*((&kernelContext_90)->scratch_0+_S1298)) ).xyz;
                uint _S1300 = sv_0(cb_2, 9U, &kernelContext_90);
                thread float3 d_lin_6;
                thread float3 d_ang_5;
                bond_kinematics_0(i_22, _S1295, _S1297, _S1299, (float4(*((&kernelContext_90)->scratch_0+_S1300)) ).xyz, &d_lin_6, &d_ang_5, &kernelContext_90);
                uint _S1301 = statics_bond_slot_0(i_22, &kernelContext_90);
                write_bond_loads_0(i_22, i_22, d_lin_6 * (float4(*((&kernelContext_90)->scratch_0+_S1301)) ).xyz, d_ang_5 * (float4(*((&kernelContext_90)->scratch_0+(_S1301 + 1U))) ).xyz, 0.0f, &kernelContext_90);
                i_22 = i_22 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            c_39 = _S1254;
            for(;;)
            {
                if(c_39 < c1_1)
                {
                }
                else
                {
                    break;
                }
                thread float3 fi_4;
                thread float3 mi_5;
                gather_loads_0(c_39, &fi_4, &mi_5, &kernelContext_90);
                thread float4 ap_lin_0 = float4(- fi_4, 0.0f);
                thread float4 ap_ang_0 = float4(- mi_5, 0.0f);
                uint _S1302 = fixed_mask_0(c_39, &kernelContext_90);
                uint _S1303 = sv_0(c_39, 8U, &kernelContext_90);
                float4 _S1304 = float4(*((&kernelContext_90)->scratch_0+_S1303)) ;
                uint _S1305 = sv_0(c_39, 9U, &kernelContext_90);
                hold_0(_S1302, &ap_lin_0, &ap_ang_0, _S1304, float4(*((&kernelContext_90)->scratch_0+_S1305)) );
                uint _S1306 = sv_0(c_39, 10U, &kernelContext_90);
                *((&kernelContext_90)->scratch_0+_S1306) = packed_float4(ap_lin_0) ;
                uint _S1307 = sv_0(c_39, 11U, &kernelContext_90);
                *((&kernelContext_90)->scratch_0+_S1307) = packed_float4(ap_ang_0) ;
                c_39 = c_39 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            float _S1308 = island_dot_0(tid_19, c0_1, c1_1, 8U, 10U, &kernelContext_90);
            uint _S1309 = cg_total_1 + 1U;
            if(_S1308 <= 0.0f)
            {
                cg_total_0 = _S1309;
                break;
            }
            float _S1310 = rz_0 / _S1308;
            uint c_44 = _S1254;
            for(;;)
            {
                if(c_44 < c1_1)
                {
                }
                else
                {
                    break;
                }
                uint _S1311 = sv_0(c_44, 2U, &kernelContext_90);
                packed_float4 device* _S1312 = (&kernelContext_90)->scratch_0+_S1311;
                float3 _S1313 = (float4(*((&kernelContext_90)->scratch_0+_S1311)) ).xyz;
                uint _S1314 = sv_0(c_44, 8U, &kernelContext_90);
                float3 _S1315 = float3(_S1310) ;
                *_S1312 = packed_float4(float4(_S1313 + _S1315 * (float4(*((&kernelContext_90)->scratch_0+_S1314)) ).xyz, 0.0f)) ;
                uint _S1316 = sv_0(c_44, 3U, &kernelContext_90);
                packed_float4 device* _S1317 = (&kernelContext_90)->scratch_0+_S1316;
                float3 _S1318 = (float4(*((&kernelContext_90)->scratch_0+_S1316)) ).xyz;
                uint _S1319 = sv_0(c_44, 9U, &kernelContext_90);
                *_S1317 = packed_float4(float4(_S1318 + _S1315 * (float4(*((&kernelContext_90)->scratch_0+_S1319)) ).xyz, 0.0f)) ;
                uint _S1320 = sv_0(c_44, 4U, &kernelContext_90);
                packed_float4 device* _S1321 = (&kernelContext_90)->scratch_0+_S1320;
                float3 _S1322 = (float4(*((&kernelContext_90)->scratch_0+_S1320)) ).xyz;
                uint _S1323 = sv_0(c_44, 10U, &kernelContext_90);
                *_S1321 = packed_float4(float4(_S1322 - _S1315 * (float4(*((&kernelContext_90)->scratch_0+_S1323)) ).xyz, 0.0f)) ;
                uint _S1324 = sv_0(c_44, 5U, &kernelContext_90);
                packed_float4 device* _S1325 = (&kernelContext_90)->scratch_0+_S1324;
                float3 _S1326 = (float4(*((&kernelContext_90)->scratch_0+_S1324)) ).xyz;
                uint _S1327 = sv_0(c_44, 11U, &kernelContext_90);
                *_S1325 = packed_float4(float4(_S1326 - _S1315 * (float4(*((&kernelContext_90)->scratch_0+_S1327)) ).xyz, 0.0f)) ;
                c_44 = c_44 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            float _S1328 = island_dot_0(tid_19, c0_1, c1_1, 4U, 4U, &kernelContext_90);
            if((sqrt(_S1328)) <= (0.00009999999747379f * _S1292))
            {
                cg_total_0 = _S1309;
                break;
            }
            uint c_45 = _S1254;
            for(;;)
            {
                if(c_45 < c1_1)
                {
                }
                else
                {
                    break;
                }
                precondition_0(c_45, &kernelContext_90);
                c_45 = c_45 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            if(free_0)
            {
                project_displacement_slot_0(tid_19, &_S1249, 6U, &kernelContext_90);
            }
            float _S1329 = island_dot_0(tid_19, c0_1, c1_1, 4U, 6U, &kernelContext_90);
            float _S1330 = _S1329 / rz_0;
            uint c_46 = _S1254;
            for(;;)
            {
                if(c_46 < c1_1)
                {
                }
                else
                {
                    break;
                }
                uint _S1331 = sv_0(c_46, 8U, &kernelContext_90);
                packed_float4 device* _S1332 = (&kernelContext_90)->scratch_0+_S1331;
                uint _S1333 = sv_0(c_46, 6U, &kernelContext_90);
                float3 _S1334 = float3(_S1330) ;
                *_S1332 = packed_float4(float4((float4(*((&kernelContext_90)->scratch_0+_S1333)) ).xyz + _S1334 * (float4(*((&kernelContext_90)->scratch_0+_S1331)) ).xyz, 0.0f)) ;
                uint _S1335 = sv_0(c_46, 9U, &kernelContext_90);
                packed_float4 device* _S1336 = (&kernelContext_90)->scratch_0+_S1335;
                uint _S1337 = sv_0(c_46, 7U, &kernelContext_90);
                *_S1336 = packed_float4(float4((float4(*((&kernelContext_90)->scratch_0+_S1337)) ).xyz + _S1334 * (float4(*((&kernelContext_90)->scratch_0+_S1335)) ).xyz, 0.0f)) ;
                c_46 = c_46 + 256U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            uint _S1338 = k_22 + 1U;
            rz_0 = _S1329;
            k_22 = _S1338;
            cg_total_1 = _S1309;
        }
        c_39 = _S1254;
        for(;;)
        {
            if(c_39 < c1_1)
            {
            }
            else
            {
                break;
            }
            uint _S1339 = 4U * c_39;
            thread float3 u_4 = (float4(*((&kernelContext_90)->state_0+_S1339)) ).xyz;
            uint _S1340 = sv_0(c_39, 21U, &kernelContext_90);
            thread float3 u_lo_0 = (float4(*((&kernelContext_90)->scratch_0+_S1340)) ).xyz;
            uint _S1341 = _S1339 + 1U;
            thread float3 th_6 = (float4(*((&kernelContext_90)->state_0+_S1341)) ).xyz;
            uint _S1342 = sv_0(c_39, 22U, &kernelContext_90);
            thread float3 th_lo_0 = (float4(*((&kernelContext_90)->scratch_0+_S1342)) ).xyz;
            uint _S1343 = sv_0(c_39, 2U, &kernelContext_90);
            comp_add_0(&u_4, &u_lo_0, (float4(*((&kernelContext_90)->scratch_0+_S1343)) ).xyz);
            uint _S1344 = sv_0(c_39, 3U, &kernelContext_90);
            comp_add_0(&th_6, &th_lo_0, (float4(*((&kernelContext_90)->scratch_0+_S1344)) ).xyz);
            *((&kernelContext_90)->state_0+_S1339) = packed_float4(float4(u_4, (float4(*((&kernelContext_90)->state_0+_S1339)) ).w)) ;
            *((&kernelContext_90)->state_0+_S1341) = packed_float4(float4(th_6, (float4(*((&kernelContext_90)->state_0+_S1341)) ).w)) ;
            *((&kernelContext_90)->scratch_0+_S1340) = packed_float4(float4(u_lo_0, 0.0f)) ;
            *((&kernelContext_90)->scratch_0+_S1342) = packed_float4(float4(th_lo_0, 0.0f)) ;
            c_39 = c_39 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        uint _S1345 = newton_0 + 1U;
        previous_2 = residual_1;
        residual_0 = residual_1;
        newton_0 = _S1345;
    }
    i_20 = b0_0 + tid_19;
    for(;;)
    {
        if(i_20 < _S1253)
        {
        }
        else
        {
            break;
        }
        JointResponse_0 _S1346 = static_response_0(i_20, &kernelContext_90);
        BondDyn_natural_0 device* _S1347 = (&kernelContext_90)->bond_dyn_0+i_20;
        float4 _S1348 = float4((*_S1347).force_lin_0) ;
        float4 _S1349 = float4((*_S1347).force_ang_0) ;
        float4 _S1350 = float4((*_S1347).sums_0) ;
        float4 _S1351 = float4((*_S1347).comps_0) ;
        uint4 _S1352 = uint4((*_S1347).events_0) ;
        thread BondDyn_0 bd_1;
        (&bd_1)->js_0 = (*_S1347).js_0;
        (&bd_1)->force_lin_0 = _S1348;
        (&bd_1)->force_ang_0 = _S1349;
        (&bd_1)->sums_0 = _S1350;
        (&bd_1)->comps_0 = _S1351;
        (&bd_1)->events_0 = _S1352;
        (&bd_1)->force_lin_0 = float4(_S1346.force_lin_1, _S1346.stored_0);
        (&bd_1)->force_ang_0 = float4(_S1346.force_ang_1, (&bd_1)->force_ang_0.w);
        BondDyn_natural_0 device* _S1353 = (&kernelContext_90)->bond_dyn_0+i_20;
        _S1353->js_0 = bd_1.js_0;
        _S1353->force_lin_0 = packed_float4(bd_1.force_lin_0) ;
        _S1353->force_ang_0 = packed_float4(bd_1.force_ang_0) ;
        _S1353->sums_0 = packed_float4(bd_1.sums_0) ;
        _S1353->comps_0 = packed_float4(bd_1.comps_0) ;
        _S1353->events_0 = packed_uint4(bd_1.events_0) ;
        write_bond_loads_0(i_20, i_20, _S1346.force_lin_1, _S1346.force_ang_1, max(_S1346.measures_0.tension_0, _S1346.measures_0.compression_0), &kernelContext_90);
        i_20 = i_20 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    c_40 = _S1254;
    for(;;)
    {
        if(c_40 < c1_1)
        {
        }
        else
        {
            break;
        }
        thread float3 fi_5;
        thread float3 mi_6;
        gather_loads_0(c_40, &fi_5, &mi_6, &kernelContext_90);
        uint _S1354 = fixed_mask_0(c_40, &kernelContext_90);
        float3 reaction_1;
        if(_S1354 != 0U)
        {
            uint _S1355 = sv_0(c_40, 0U, &kernelContext_90);
            reaction_1 = - ((float4(*((&kernelContext_90)->scratch_0+_S1355)) ).xyz + fi_5);
        }
        else
        {
            uint _S1356 = 4U * c_40;
            reaction_1 = float3((float4(*((&kernelContext_90)->state_0+(_S1356 + 1U))) ).w, (float4(*((&kernelContext_90)->state_0+(_S1356 + 2U))) ).w, (float4(*((&kernelContext_90)->state_0+(_S1356 + 3U))) ).w);
        }
        uint _S1357 = 4U * c_40;
        uint _S1358 = _S1357 + 1U;
        *((&kernelContext_90)->state_0+_S1358) = packed_float4(float4((float4(*((&kernelContext_90)->state_0+_S1358)) ).xyz, reaction_1.x)) ;
        *((&kernelContext_90)->state_0+(_S1357 + 2U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, reaction_1.y)) ;
        *((&kernelContext_90)->state_0+(_S1357 + 3U)) = packed_float4(float4(0.0f, 0.0f, 0.0f, reaction_1.z)) ;
        c_40 = c_40 + 256U;
    }
    if(tid_19 == 0U)
    {
        uint _S1359 = statics_result_slot_0(_S1248, &kernelContext_90);
        packed_float4 device* _S1360 = (&kernelContext_90)->scratch_0+_S1359;
        float _S1361 = (as_type<float>((newton_0)));
        float _S1362 = (as_type<float>((cg_total_0)));
        if(converged_4)
        {
            previous_2 = 1.0f;
        }
        else
        {
            previous_2 = 0.0f;
        }
        *_S1360 = packed_float4(float4(residual_0, _S1361, _S1362, previous_2)) ;
        ((&kernelContext_90)->islands_0+_S1248)->info_0[int(2)] = _S1251 & 4294967287U;
    }
    return;
}

[[kernel]] void settled_fatigue(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], Params_0 constant* params_9 [[buffer(0)]], Island_natural_0 device* islands_9 [[buffer(9)]], uint device* index_9 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(3)]], packed_float4 device* state_11 [[buffer(6)]], packed_float4 device* scratch_9 [[buffer(8)]], packed_float4 device* contact_state_9 [[buffer(11)]], packed_float4 device* loads_9 [[buffer(5)]], Impactor_natural_0 device* impactors_9 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_9 [[buffer(7)]], BondStatic_natural_0 device* bonds_9 [[buffer(2)]], MaterialTable_0 constant* materials_9 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_91;
    (&kernelContext_91)->params_0 = params_9;
    (&kernelContext_91)->islands_0 = islands_9;
    (&kernelContext_91)->index_0 = index_9;
    (&kernelContext_91)->chunks_0 = chunks_9;
    (&kernelContext_91)->state_0 = state_11;
    (&kernelContext_91)->scratch_0 = scratch_9;
    (&kernelContext_91)->contact_state_0 = contact_state_9;
    (&kernelContext_91)->loads_0 = loads_9;
    (&kernelContext_91)->impactors_0 = impactors_9;
    (&kernelContext_91)->bond_dyn_0 = bond_dyn_9;
    (&kernelContext_91)->bonds_0 = bonds_9;
    (&kernelContext_91)->materials_0 = materials_9;
    threadgroup array<float4, int(32)> g_part_a_9;
    (&kernelContext_91)->g_part_a_0 = &g_part_a_9;
    threadgroup array<float4, int(32)> g_part_b_9;
    (&kernelContext_91)->g_part_b_0 = &g_part_b_9;
    threadgroup array<uint, int(5)> g_seen_9;
    (&kernelContext_91)->g_seen_0 = &g_seen_9;
    threadgroup uint g_wide_run_9;
    (&kernelContext_91)->g_wide_run_0 = &g_wide_run_9;
    threadgroup array<uint, int(4096)> g_label_9;
    (&kernelContext_91)->g_label_0 = &g_label_9;
    threadgroup uint g_changed_9;
    (&kernelContext_91)->g_changed_0 = &g_changed_9;
    threadgroup uint g_halt_9;
    (&kernelContext_91)->g_halt_0 = &g_halt_9;
    threadgroup uint g_run_9;
    (&kernelContext_91)->g_run_0 = &g_run_9;
    threadgroup array<uint, int(512)> g_label_narrow_9;
    (&kernelContext_91)->g_label_narrow_0 = &g_label_narrow_9;
    threadgroup uint g_changed_narrow_9;
    (&kernelContext_91)->g_changed_narrow_0 = &g_changed_narrow_9;
    uint tid_20 = thread_6.x;
    uint _S1363 = group_6.x;
    Island_natural_0 device* _S1364 = islands_9+_S1363;
    Island_natural_0 isl_28 = *_S1364;
    uint4 _S1365 = uint4((*_S1364).info_0) ;
    bool _S1366;
    if(((_S1365.x) & 16U) == 0U)
    {
        _S1366 = true;
    }
    else
    {
        uint4 _S1367 = uint4(isl_28.range_0) ;
        _S1366 = (_S1367.w) == (_S1367.z);
    }
    if(_S1366)
    {
        return;
    }
    seen_clear_0(tid_20, &kernelContext_91);
    if(tid_20 == 0U)
    {
        *(&kernelContext_91)->g_halt_0 = 0U;
        *(&kernelContext_91)->g_run_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _S1368 = _S1365.w;
    uint4 _S1369 = uint4(isl_28.range_0) ;
    uint i_23 = _S1369.z + tid_20;
    for(;;)
    {
        if(i_23 < (_S1369.w))
        {
        }
        else
        {
            break;
        }
        BondStatic_natural_0 device* _S1370 = (&kernelContext_91)->bonds_0+i_23;
        BondDyn_natural_0 device* _S1371 = (&kernelContext_91)->bond_dyn_0+i_23;
        float4 _S1372 = float4((*_S1371).force_lin_0) ;
        float4 _S1373 = float4((*_S1371).force_ang_0) ;
        float4 _S1374 = float4((*_S1371).sums_0) ;
        float4 _S1375 = float4((*_S1371).comps_0) ;
        uint4 _S1376 = uint4((*_S1371).events_0) ;
        thread BondDyn_0 bd_2;
        (&bd_2)->js_0 = (*_S1371).js_0;
        (&bd_2)->force_lin_0 = _S1372;
        (&bd_2)->force_ang_0 = _S1373;
        (&bd_2)->sums_0 = _S1374;
        (&bd_2)->comps_0 = _S1375;
        (&bd_2)->events_0 = _S1376;
        thread JointBond_natural_0 _S1377 = _S1370->law_0;
        uint4 _S1378 = uint4((&_S1377)->ids_0) ;
        uint _S1379 = 4U * _S1378.y;
        uint _S1380 = 4U * _S1378.z;
        thread float3 d_lin_7;
        thread float3 d_ang_6;
        bond_kinematics_0(i_23, (float4(*((&kernelContext_91)->state_0+_S1379)) ).xyz, (float4(*((&kernelContext_91)->state_0+(_S1379 + 1U))) ).xyz, (float4(*((&kernelContext_91)->state_0+_S1380)) ).xyz, (float4(*((&kernelContext_91)->state_0+(_S1380 + 1U))) ).xyz, &d_lin_7, &d_ang_6, &kernelContext_91);
        JointState_0 previous_3 = (&bd_2)->js_0;
        float _S1381 = (&kernelContext_91)->params_0->dt_0;
        bool _S1382 = ((&kernelContext_91)->params_0->fracture_0) != 0U;
        _S1377 = _S1370->law_0;
        thread JointState_0 _S1383 = (&bd_2)->js_0;
        JointResponse_0 _S1384 = joint_evaluate_0(&(&kernelContext_91)->materials_0->m_0[_S1378.x], &_S1377, &_S1383, d_lin_7, d_ang_6, _S1381, _S1382);
        if((_S1384.state_1.damage_0) > (previous_3.damage_0 + 9.99999971718068537e-10f))
        {
            _S1366 = true;
        }
        else
        {
            _S1366 = (_S1384.state_1.crush_1) > (previous_3.crush_1 + 9.99999971718068537e-10f);
        }
        uint flags_3;
        if(_S1366)
        {
            flags_3 = 16U;
        }
        else
        {
            flags_3 = 0U;
        }
        thread float _S1385 = (&bd_2)->sums_0.x;
        thread float _S1386 = (&bd_2)->comps_0.x;
        comp_add1_0(&_S1385, &_S1386, _S1384.dissipated_1);
        (&bd_2)->comps_0.x = _S1386;
        (&bd_2)->sums_0.x = _S1385;
        thread float _S1387 = (&bd_2)->sums_0.y;
        thread float _S1388 = (&bd_2)->comps_0.y;
        comp_add1_0(&_S1387, &_S1388, _S1384.overshoot_0);
        (&bd_2)->comps_0.y = _S1388;
        (&bd_2)->sums_0.y = _S1387;
        (&bd_2)->force_lin_0 = float4(_S1384.force_lin_1, _S1384.stored_0);
        (&bd_2)->force_ang_0 = float4(_S1384.force_ang_1, max((&bd_2)->force_ang_0.w, _S1384.state_1.utilization_0));
        thread JointState_0 _S1389 = previous_3;
        bool _S1390 = is_damaged_0(&_S1389);
        bool _S1391;
        if(!_S1390)
        {
            thread JointState_0 _S1392 = _S1384.state_1;
            bool _S1393 = is_damaged_0(&_S1392);
            _S1391 = _S1393;
        }
        else
        {
            _S1391 = false;
        }
        bool _S1394;
        if(_S1391)
        {
            _S1394 = ((&bd_2)->events_0.x) == 0U;
        }
        else
        {
            _S1394 = false;
        }
        if(_S1394)
        {
            (&bd_2)->events_0.x = _S1368;
            (&bd_2)->events_0.w = _S1384.state_1.mode_0;
            (*(&kernelContext_91)->g_seen_0)[3U] = 1U;
        }
        bool _S1395;
        if(((&bd_2)->events_0.y) == 0U)
        {
            _S1395 = (_S1384.state_1.life_0) != (previous_3.life_0);
        }
        else
        {
            _S1395 = false;
        }
        bool _S1396;
        if(_S1395)
        {
            _S1396 = (_S1384.fatigue_before_0) > 0.99000000953674316f;
        }
        else
        {
            _S1396 = false;
        }
        bool _S1397;
        if(_S1396)
        {
            float _S1398 = fatigue_factor_0(&(&kernelContext_91)->materials_0->m_0[_S1378.x], _S1384.state_1.life_0);
            _S1397 = _S1398 <= 0.99000000953674316f;
        }
        else
        {
            _S1397 = false;
        }
        if(_S1397)
        {
            (&bd_2)->events_0.y = _S1368;
            (*(&kernelContext_91)->g_seen_0)[4U] = 1U;
        }
        uint flags_4;
        if(_S1384.disconnected_0)
        {
            (&bd_2)->events_0.z = _S1368;
            flags_4 = flags_3 | 32U;
        }
        else
        {
            flags_4 = flags_3;
        }
        (&bd_2)->js_0 = _S1384.state_1;
        BondDyn_natural_0 device* _S1399 = (&kernelContext_91)->bond_dyn_0+i_23;
        _S1399->js_0 = bd_2.js_0;
        _S1399->force_lin_0 = packed_float4(bd_2.force_lin_0) ;
        _S1399->force_ang_0 = packed_float4(bd_2.force_ang_0) ;
        _S1399->sums_0 = packed_float4(bd_2.sums_0) ;
        _S1399->comps_0 = packed_float4(bd_2.comps_0) ;
        _S1399->events_0 = packed_uint4(bd_2.events_0) ;
        write_bond_loads_0(i_23, i_23, _S1384.force_lin_1, _S1384.force_ang_1, max(_S1384.measures_0.tension_0, _S1384.measures_0.compression_0), &kernelContext_91);
        if((flags_4 & 16U) != 0U)
        {
            *(&kernelContext_91)->g_halt_0 = 1U;
        }
        if((flags_4 & 32U) != 0U)
        {
            *(&kernelContext_91)->g_run_0 = 1U;
        }
        i_23 = i_23 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid_20 != 0U)
    {
        return;
    }
    if(((*(&kernelContext_91)->g_halt_0) | (*(&kernelContext_91)->g_run_0)) != 0U)
    {
        uint _S1400 = _S1365.z;
        if((*(&kernelContext_91)->g_halt_0) != 0U)
        {
            i_23 = 16U;
        }
        else
        {
            i_23 = 0U;
        }
        uint _S1401 = _S1400 | i_23;
        if((*(&kernelContext_91)->g_run_0) != 0U)
        {
            i_23 = 32U;
        }
        else
        {
            i_23 = 0U;
        }
        ((&kernelContext_91)->islands_0+_S1363)->info_0[int(2)] = _S1401 | i_23;
    }
    uint _S1402 = (isl_28.coverage_0) | 16384U;
    uint _S1403 = seen_bits_0(&kernelContext_91);
    uint _S1404 = _S1402 | _S1403;
    if((*(&kernelContext_91)->g_run_0) != 0U)
    {
        i_23 = 128U;
    }
    else
    {
        i_23 = 0U;
    }
    ((&kernelContext_91)->islands_0+_S1363)->coverage_0 = _S1404 | i_23;
    return;
}

void impactor_integrate_at_0(uint ii_3, uint tid_21, KernelContext_0 thread* kernelContext_92)
{
    float3 p_18;
    if(ii_3 >= (kernelContext_92->params_0->impactor_count_0))
    {
        return;
    }
    Island_natural_0 device* _S1405 = kernelContext_92->islands_0+kernelContext_92->params_0->halt_index_0;
    Impactor_natural_0 device* _S1406 = kernelContext_92->impactors_0+ii_3;
    float4 _S1407 = float4((*_S1406).position_err_1) ;
    float4 _S1408 = float4((*_S1406).velocity_1) ;
    float4 _S1409 = float4((*_S1406).velocity_err_1) ;
    float4 _S1410 = float4((*_S1406).angular_velocity_1) ;
    float4 _S1411 = float4((*_S1406).rotation_1) ;
    float4 _S1412 = float4((*_S1406).inertia0_2) ;
    float4 _S1413 = float4((*_S1406).inertia1_2) ;
    float4 _S1414 = float4((*_S1406).inertia2_2) ;
    float4 _S1415 = float4((*_S1406).inv0_2) ;
    float4 _S1416 = float4((*_S1406).inv1_2) ;
    float4 _S1417 = float4((*_S1406).inv2_2) ;
    float4 _S1418 = float4((*_S1406).shape_0) ;
    float4 _S1419 = float4((*_S1406).half_1) ;
    float4 _S1420 = float4((*_S1406).mat_0) ;
    float4 _S1421 = float4((*_S1406).crush_0) ;
    float4 _S1422 = float4((*_S1406).geom_0) ;
    float4 _S1423 = float4((*_S1406).rotation_err_1) ;
    float4 _S1424 = float4((*_S1406).ledger_0) ;
    uint4 _S1425 = uint4((*_S1406).cand_0) ;
    float4 _S1426 = float4((*_S1406).momentum_1) ;
    float4 _S1427 = float4((*_S1406).momentum_err_1) ;
    thread Impactor_0 imp_6;
    (&imp_6)->position_1 = float4((*_S1406).position_1) ;
    (&imp_6)->position_err_1 = _S1407;
    (&imp_6)->velocity_1 = _S1408;
    (&imp_6)->velocity_err_1 = _S1409;
    (&imp_6)->angular_velocity_1 = _S1410;
    (&imp_6)->rotation_1 = _S1411;
    (&imp_6)->inertia0_2 = _S1412;
    (&imp_6)->inertia1_2 = _S1413;
    (&imp_6)->inertia2_2 = _S1414;
    (&imp_6)->inv0_2 = _S1415;
    (&imp_6)->inv1_2 = _S1416;
    (&imp_6)->inv2_2 = _S1417;
    (&imp_6)->shape_0 = _S1418;
    (&imp_6)->half_1 = _S1419;
    (&imp_6)->mat_0 = _S1420;
    (&imp_6)->crush_0 = _S1421;
    (&imp_6)->geom_0 = _S1422;
    (&imp_6)->rotation_err_1 = _S1423;
    (&imp_6)->ledger_0 = _S1424;
    (&imp_6)->cand_0 = _S1425;
    (&imp_6)->momentum_1 = _S1426;
    (&imp_6)->momentum_err_1 = _S1427;
    uint4 _S1428 = uint4(_S1405->info_0) ;
    bool _S1429;
    if(((_S1428.z) & 1U) != 0U)
    {
        _S1429 = true;
    }
    else
    {
        uint _S1430 = _S1428.y;
        if(_S1430 != 0U)
        {
            _S1429 = ((&imp_6)->cand_0.w) >= _S1430;
        }
        else
        {
            _S1429 = false;
        }
    }
    if(_S1429)
    {
        return;
    }
    if(((&imp_6)->cand_0.z) != 0U)
    {
        if(tid_21 == 0U)
        {
            (kernelContext_92->impactors_0+ii_3)->cand_0[int(3)] = (&imp_6)->cand_0.w + 1U;
        }
        return;
    }
    float4 _S1431 = float4(0.0f) ;
    thread float4 rf_1 = _S1431;
    thread float4 rt_0 = _S1431;
    uint k_23 = (&imp_6)->cand_0.x + tid_21;
    for(;;)
    {
        if(k_23 < ((&imp_6)->cand_0.y))
        {
        }
        else
        {
            break;
        }
        uint _S1432 = 3U * k_23;
        rf_1 = rf_1 + float4(*(kernelContext_92->scratch_0+(kernelContext_92->params_0->cand_base_0 + _S1432 + 1U))) ;
        rt_0 = rt_0 + float4(*(kernelContext_92->scratch_0+(kernelContext_92->params_0->cand_base_0 + _S1432 + 2U))) ;
        k_23 = k_23 + 256U;
    }
    group_sum2_0(tid_21, &rf_1, &rt_0, kernelContext_92);
    if(tid_21 != 0U)
    {
        return;
    }
    float dt_13 = kernelContext_92->params_0->dt_0;
    float3 _S1433 = float3(0.0f) ;
    float3 load_f_0;
    float3 load_t_0;
    if((kernelContext_92->params_0->has_ground_0) != 0U)
    {
        float3 _S1434 = (&imp_6)->half_1.xyz;
        thread Impactor_0 _S1435 = imp_6;
        Box_0 _S1436 = impactor_box_0(&_S1435, _S1433, _S1434);
        float3 _S1437 = (&imp_6)->velocity_1.xyz + (&imp_6)->velocity_err_1.xyz;
        float _S1438 = kernelContext_92->params_0->ground_modulus_0;
        float _S1439 = (&imp_6)->mat_0.x;
        float3 _S1440 = float3(0.0f, 0.0f, 1.0f);
        thread Box_0 _S1441 = _S1436;
        thread Box_0 _S1442 = _S1436;
        float _S1443 = contact_stiffness_0(_S1438, &_S1441, _S1439, &_S1442, _S1440);
        float _S1444 = (&imp_6)->position_1.z - kernelContext_92->params_0->ground_hi_0 + ((&imp_6)->position_err_1.z - kernelContext_92->params_0->ground_lo_0);
        uint total_points_0;
        if(((&imp_6)->shape_0.x) == 0.0f)
        {
            total_points_0 = 1U;
        }
        else
        {
            total_points_0 = 14U;
        }
        float _S1445 = _S1443 / float(min(total_points_0, 5U));
        uint s_11 = 0U;
        uint below_0 = 0U;
        for(;;)
        {
            if(s_11 < total_points_0)
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
                thread Box_0 _S1446 = _S1436;
                float3 _S1447 = sample_point_0(&_S1446, s_11, kernelContext_92);
                p_18 = _S1447;
            }
            if((_S1444 + p_18.z) < 0.0f)
            {
                below_0 = below_0 + 1U;
            }
            s_11 = s_11 + 1U;
        }
        if(below_0 > 0U)
        {
            sentinel_mark_0(8U, kernelContext_92);
        }
        s_11 = 0U;
        load_f_0 = _S1433;
        load_t_0 = _S1433;
        for(;;)
        {
            if(s_11 < total_points_0)
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
                thread Box_0 _S1448 = _S1436;
                float3 _S1449 = sample_point_0(&_S1448, s_11, kernelContext_92);
                p_18 = _S1449;
            }
            float depth_4 = - (_S1444 + p_18.z);
            if(depth_4 <= 0.0f)
            {
                s_11 = s_11 + 1U;
                continue;
            }
            thread float stored_7;
            thread float diss_5;
            float3 _S1450 = penalty_force_0(_S1445, (&imp_6)->mat_0.z, kernelContext_92->params_0->ground_friction_0, depth_4, _S1440, _S1437 + cross((&imp_6)->angular_velocity_1.xyz, p_18), dt_13, below_0, &stored_7, &diss_5, kernelContext_92);
            float3 load_f_1 = load_f_0 + _S1450;
            float3 load_t_1 = load_t_0 + cross(p_18, _S1450);
            thread float _S1451 = (&imp_6)->ledger_0.x;
            thread float _S1452 = (&imp_6)->ledger_0.y;
            comp_add1_0(&_S1451, &_S1452, diss_5);
            (&imp_6)->ledger_0.y = _S1452;
            (&imp_6)->ledger_0.x = _S1451;
            load_f_0 = load_f_1;
            load_t_0 = load_t_1;
            s_11 = s_11 + 1U;
        }
    }
    else
    {
        load_f_0 = _S1433;
        load_t_0 = _S1433;
    }
    float3 load_f_2 = rf_1.xyz + load_f_0;
    float3 load_t_2 = rt_0.xyz + load_t_0;
    float m_7 = (&imp_6)->mat_0.z;
    thread float3 vel_3 = (&imp_6)->velocity_1.xyz;
    thread float3 vel_err_1 = (&imp_6)->velocity_err_1.xyz;
    float3 _S1453 = float3(dt_13) ;
    comp_add_0(&vel_3, &vel_err_1, (load_f_2 / float3(m_7)  + kernelContext_92->params_0->gravity_0.xyz) * _S1453);
    Quat_0 _S1454 = quat_of_0((&imp_6)->rotation_1);
    thread Quat_0 q_15 = _S1454;
    thread float4 q_err_0 = (&imp_6)->rotation_err_1;
    thread float3 l_hi_0 = (&imp_6)->momentum_1.xyz;
    thread float3 l_err_1 = (&imp_6)->momentum_err_1.xyz;
    comp_add_0(&l_hi_0, &l_err_1, load_t_2 * _S1453);
    float3 l_9 = l_hi_0 + l_err_1;
    thread Quat_0 _S1455 = _S1454;
    float3 _S1456 = world_mul_0(&_S1455, (&imp_6)->inv0_2, (&imp_6)->inv1_2, (&imp_6)->inv2_2, l_9);
    thread float3 pos_1 = (&imp_6)->position_1.xyz;
    thread float3 pos_err_1 = (&imp_6)->position_err_1.xyz;
    comp_add_0(&pos_1, &pos_err_1, (vel_3 + vel_err_1) * _S1453);
    turn_left_0(&q_15, &q_err_0, _S1456, dt_13);
    thread Quat_0 _S1457 = q_15;
    float3 _S1458 = world_mul_0(&_S1457, (&imp_6)->inv0_2, (&imp_6)->inv1_2, (&imp_6)->inv2_2, l_9);
    (&imp_6)->angular_velocity_1 = float4(_S1458, 0.0f);
    thread Quat_0 _S1459 = q_15;
    float4 _S1460 = quat_vec_0(&_S1459);
    (&imp_6)->rotation_1 = _S1460;
    (&imp_6)->rotation_err_1 = q_err_0;
    (&imp_6)->momentum_1 = float4(l_hi_0, 0.0f);
    (&imp_6)->momentum_err_1 = float4(l_err_1, 0.0f);
    (&imp_6)->position_1 = float4(pos_1, 0.0f);
    (&imp_6)->position_err_1 = float4(pos_err_1, 0.0f);
    (&imp_6)->velocity_1 = float4(vel_3, 0.0f);
    (&imp_6)->velocity_err_1 = float4(vel_err_1, 0.0f);
    (&imp_6)->cand_0.w = (&imp_6)->cand_0.w + 1U;
    (&imp_6)->geom_0 = float4(1.0f, (&imp_6)->crush_0.w, 0.0f, 0.0f);
    Impactor_natural_0 device* _S1461 = kernelContext_92->impactors_0+ii_3;
    _S1461->position_1 = packed_float4(imp_6.position_1) ;
    _S1461->position_err_1 = packed_float4(imp_6.position_err_1) ;
    _S1461->velocity_1 = packed_float4(imp_6.velocity_1) ;
    _S1461->velocity_err_1 = packed_float4(imp_6.velocity_err_1) ;
    _S1461->angular_velocity_1 = packed_float4(imp_6.angular_velocity_1) ;
    _S1461->rotation_1 = packed_float4(imp_6.rotation_1) ;
    _S1461->inertia0_2 = packed_float4(imp_6.inertia0_2) ;
    _S1461->inertia1_2 = packed_float4(imp_6.inertia1_2) ;
    _S1461->inertia2_2 = packed_float4(imp_6.inertia2_2) ;
    _S1461->inv0_2 = packed_float4(imp_6.inv0_2) ;
    _S1461->inv1_2 = packed_float4(imp_6.inv1_2) ;
    _S1461->inv2_2 = packed_float4(imp_6.inv2_2) ;
    _S1461->shape_0 = packed_float4(imp_6.shape_0) ;
    _S1461->half_1 = packed_float4(imp_6.half_1) ;
    _S1461->mat_0 = packed_float4(imp_6.mat_0) ;
    _S1461->crush_0 = packed_float4(imp_6.crush_0) ;
    _S1461->geom_0 = packed_float4(imp_6.geom_0) ;
    _S1461->rotation_err_1 = packed_float4(imp_6.rotation_err_1) ;
    _S1461->ledger_0 = packed_float4(imp_6.ledger_0) ;
    _S1461->cand_0 = packed_uint4(imp_6.cand_0) ;
    _S1461->momentum_1 = packed_float4(imp_6.momentum_1) ;
    _S1461->momentum_err_1 = packed_float4(imp_6.momentum_err_1) ;
    uint k_24 = (&imp_6)->cand_0.w - 1U - kernelContext_92->params_0->step_start_0;
    if(k_24 < (kernelContext_92->params_0->record_stride_0))
    {
        uint at_8 = kernelContext_92->params_0->record_base_0 + 2U * (ii_3 * kernelContext_92->params_0->record_stride_0 + k_24);
        *(kernelContext_92->scratch_0+at_8) = packed_float4(float4(vel_3 + vel_err_1, 0.0f)) ;
        *(kernelContext_92->scratch_0+(at_8 + 1U)) = packed_float4(float4(pos_1 + pos_err_1, 0.0f)) ;
    }
    return;
}

[[kernel]] void impactor_integrate(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], Params_0 constant* params_10 [[buffer(0)]], Island_natural_0 device* islands_10 [[buffer(9)]], uint device* index_10 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(3)]], packed_float4 device* state_12 [[buffer(6)]], packed_float4 device* scratch_10 [[buffer(8)]], packed_float4 device* contact_state_10 [[buffer(11)]], packed_float4 device* loads_10 [[buffer(5)]], Impactor_natural_0 device* impactors_10 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_10 [[buffer(7)]], BondStatic_natural_0 device* bonds_10 [[buffer(2)]], MaterialTable_0 constant* materials_10 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_93;
    (&kernelContext_93)->params_0 = params_10;
    (&kernelContext_93)->islands_0 = islands_10;
    (&kernelContext_93)->index_0 = index_10;
    (&kernelContext_93)->chunks_0 = chunks_10;
    (&kernelContext_93)->state_0 = state_12;
    (&kernelContext_93)->scratch_0 = scratch_10;
    (&kernelContext_93)->contact_state_0 = contact_state_10;
    (&kernelContext_93)->loads_0 = loads_10;
    (&kernelContext_93)->impactors_0 = impactors_10;
    (&kernelContext_93)->bond_dyn_0 = bond_dyn_10;
    (&kernelContext_93)->bonds_0 = bonds_10;
    (&kernelContext_93)->materials_0 = materials_10;
    threadgroup array<float4, int(32)> g_part_a_10;
    (&kernelContext_93)->g_part_a_0 = &g_part_a_10;
    threadgroup array<float4, int(32)> g_part_b_10;
    (&kernelContext_93)->g_part_b_0 = &g_part_b_10;
    threadgroup array<uint, int(5)> g_seen_10;
    (&kernelContext_93)->g_seen_0 = &g_seen_10;
    threadgroup uint g_wide_run_10;
    (&kernelContext_93)->g_wide_run_0 = &g_wide_run_10;
    threadgroup array<uint, int(4096)> g_label_10;
    (&kernelContext_93)->g_label_0 = &g_label_10;
    threadgroup uint g_changed_10;
    (&kernelContext_93)->g_changed_0 = &g_changed_10;
    threadgroup uint g_halt_10;
    (&kernelContext_93)->g_halt_0 = &g_halt_10;
    threadgroup uint g_run_10;
    (&kernelContext_93)->g_run_0 = &g_run_10;
    threadgroup array<uint, int(512)> g_label_narrow_10;
    (&kernelContext_93)->g_label_narrow_0 = &g_label_narrow_10;
    threadgroup uint g_changed_narrow_10;
    (&kernelContext_93)->g_changed_narrow_0 = &g_changed_narrow_10;
    impactor_integrate_at_0(group_7.x, thread_7.x, &kernelContext_93);
    return;
}

void ground_contact_0(uint c_47, bool account_0, float3 thread* f_16, float3 thread* t_18, KernelContext_0 thread* kernelContext_94)
{
    ChunkStatic_natural_0 device* _S1462 = kernelContext_94->chunks_0+c_47;
    WorldPoint_0 _S1463 = chunk_world_0(c_47, kernelContext_94);
    float above_0 = _S1463.hi_3.z - kernelContext_94->params_0->ground_hi_0 + (_S1463.lo_3.z - kernelContext_94->params_0->ground_lo_0) + _S1463.rel_0.z;
    if((above_0 - (float4(_S1462->half_0) ).w) > 0.0f)
    {
        return;
    }
    Box_0 _S1464 = chunk_box_0(c_47, float3(0.0f) , kernelContext_94);
    float _S1465 = kernelContext_94->params_0->ground_modulus_0;
    float _S1466 = (float4(_S1462->cmat_0) ).x;
    float3 _S1467 = float3(0.0f, 0.0f, 1.0f);
    thread Box_0 _S1468 = _S1464;
    thread Box_0 _S1469 = _S1464;
    float _S1470 = contact_stiffness_0(_S1465, &_S1468, _S1466, &_S1469, _S1467);
    thread Box_0 _S1471 = _S1464;
    uint _S1472 = sample_count_0(&_S1471);
    uint s_12 = 0U;
    uint n_12 = 0U;
    for(;;)
    {
        if(s_12 < _S1472)
        {
        }
        else
        {
            break;
        }
        thread Box_0 _S1473 = _S1464;
        float3 _S1474 = sample_point_0(&_S1473, s_12, kernelContext_94);
        if((above_0 + _S1474.z) < 0.0f)
        {
            n_12 = n_12 + 1U;
        }
        s_12 = s_12 + 1U;
    }
    if(n_12 == 0U)
    {
        return;
    }
    sentinel_mark_0(4U, kernelContext_94);
    thread float3 vc_1;
    thread float3 wc_1;
    chunk_velocity_0(c_47, &vc_1, &wc_1, kernelContext_94);
    thread float4 ledger_2 = float4(*(kernelContext_94->scratch_0+(kernelContext_94->params_0->ledger_base_0 + kernelContext_94->params_0->pair_count_0 + c_47))) ;
    s_12 = 0U;
    for(;;)
    {
        if(s_12 < _S1472)
        {
        }
        else
        {
            break;
        }
        thread Box_0 _S1475 = _S1464;
        float3 _S1476 = sample_point_0(&_S1475, s_12, kernelContext_94);
        float _S1477 = above_0 + _S1476.z;
        if(!(_S1477 < 0.0f))
        {
            s_12 = s_12 + 1U;
            continue;
        }
        thread float stored_8;
        thread float diss_6;
        float3 _S1478 = penalty_force_0(_S1470 / float(max(n_12, 5U)), (float4(_S1462->center_0) ).w, kernelContext_94->params_0->ground_friction_0, - _S1477, _S1467, vc_1 + cross(wc_1, _S1476), kernelContext_94->params_0->dt_0, n_12, &stored_8, &diss_6, kernelContext_94);
        *f_16 = *f_16 + _S1478;
        *t_18 = *t_18 + cross(_S1476, _S1478);
        thread float _S1479 = ledger_2.y;
        thread float _S1480 = ledger_2.z;
        comp_add1_0(&_S1479, &_S1480, diss_6);
        ledger_2.z = _S1480;
        ledger_2.y = _S1479;
        s_12 = s_12 + 1U;
    }
    if(account_0)
    {
        *(kernelContext_94->scratch_0+(kernelContext_94->params_0->ledger_base_0 + kernelContext_94->params_0->pair_count_0 + c_47)) = packed_float4(ledger_2) ;
    }
    return;
}

void contact_sum_at_0(uint g_2, KernelContext_0 thread* kernelContext_95)
{
    bool _S1481;
    if(g_2 >= (kernelContext_95->params_0->seg_count_0))
    {
        _S1481 = true;
    }
    else
    {
        bool _S1482 = stopped_0(kernelContext_95);
        _S1481 = _S1482;
    }
    if(_S1481)
    {
        return;
    }
    uint _S1483 = 3U * g_2;
    uint _S1484 = kernelContext_95->index_0[kernelContext_95->params_0->seg_index_0 + _S1483];
    uint begin_1 = kernelContext_95->index_0[kernelContext_95->params_0->seg_index_0 + _S1483 + 1U];
    uint _S1485 = kernelContext_95->index_0[kernelContext_95->params_0->seg_index_0 + _S1483 + 2U];
    float3 _S1486 = float3(0.0f) ;
    thread float3 f_17 = _S1486;
    thread float3 t_19 = _S1486;
    uint e_8 = begin_1;
    for(;;)
    {
        if(e_8 < _S1485)
        {
        }
        else
        {
            break;
        }
        uint entry_7 = kernelContext_95->index_0[e_8];
        if(entry_7 == 2147483648U)
        {
            ground_contact_0(_S1484, true, &f_17, &t_19, kernelContext_95);
            e_8 = e_8 + 1U;
            continue;
        }
        uint _S1487 = 2U * entry_7;
        f_17 = f_17 + (float4(*(kernelContext_95->scratch_0+(kernelContext_95->params_0->slot_base_0 + _S1487))) ).xyz;
        t_19 = t_19 + (float4(*(kernelContext_95->scratch_0+(kernelContext_95->params_0->slot_base_0 + _S1487 + 1U))) ).xyz;
        e_8 = e_8 + 1U;
    }
    uint _S1488 = 2U * g_2;
    *(kernelContext_95->scratch_0+(kernelContext_95->params_0->seg_base_0 + _S1488)) = packed_float4(float4(f_17, 0.0f)) ;
    *(kernelContext_95->scratch_0+(kernelContext_95->params_0->seg_base_0 + _S1488 + 1U)) = packed_float4(float4(t_19, 0.0f)) ;
    return;
}

[[kernel]] void contact_sums(uint3 id_2 [[thread_position_in_grid]], Params_0 constant* params_11 [[buffer(0)]], Island_natural_0 device* islands_11 [[buffer(9)]], uint device* index_11 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(3)]], packed_float4 device* state_13 [[buffer(6)]], packed_float4 device* scratch_11 [[buffer(8)]], packed_float4 device* contact_state_11 [[buffer(11)]], packed_float4 device* loads_11 [[buffer(5)]], Impactor_natural_0 device* impactors_11 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_11 [[buffer(7)]], BondStatic_natural_0 device* bonds_11 [[buffer(2)]], MaterialTable_0 constant* materials_11 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_96;
    (&kernelContext_96)->params_0 = params_11;
    (&kernelContext_96)->islands_0 = islands_11;
    (&kernelContext_96)->index_0 = index_11;
    (&kernelContext_96)->chunks_0 = chunks_11;
    (&kernelContext_96)->state_0 = state_13;
    (&kernelContext_96)->scratch_0 = scratch_11;
    (&kernelContext_96)->contact_state_0 = contact_state_11;
    (&kernelContext_96)->loads_0 = loads_11;
    (&kernelContext_96)->impactors_0 = impactors_11;
    (&kernelContext_96)->bond_dyn_0 = bond_dyn_11;
    (&kernelContext_96)->bonds_0 = bonds_11;
    (&kernelContext_96)->materials_0 = materials_11;
    threadgroup array<float4, int(32)> g_part_a_11;
    (&kernelContext_96)->g_part_a_0 = &g_part_a_11;
    threadgroup array<float4, int(32)> g_part_b_11;
    (&kernelContext_96)->g_part_b_0 = &g_part_b_11;
    threadgroup array<uint, int(5)> g_seen_11;
    (&kernelContext_96)->g_seen_0 = &g_seen_11;
    threadgroup uint g_wide_run_11;
    (&kernelContext_96)->g_wide_run_0 = &g_wide_run_11;
    threadgroup array<uint, int(4096)> g_label_11;
    (&kernelContext_96)->g_label_0 = &g_label_11;
    threadgroup uint g_changed_11;
    (&kernelContext_96)->g_changed_0 = &g_changed_11;
    threadgroup uint g_halt_11;
    (&kernelContext_96)->g_halt_0 = &g_halt_11;
    threadgroup uint g_run_11;
    (&kernelContext_96)->g_run_0 = &g_run_11;
    threadgroup array<uint, int(512)> g_label_narrow_11;
    (&kernelContext_96)->g_label_narrow_0 = &g_label_narrow_11;
    threadgroup uint g_changed_narrow_11;
    (&kernelContext_96)->g_changed_narrow_0 = &g_changed_narrow_11;
    contact_sum_at_0(id_2.x, &kernelContext_96);
    return;
}

void net_load_1(uint c_48, const Island_natural_0 thread* isl_29, const Rigid_0 thread* rg_16, uint k_25, float dt_14, bool contact_4, float3 thread* f_18, float3 thread* t_20, KernelContext_0 thread* kernelContext_97)
{
    ChunkStatic_natural_0 device* _S1489 = kernelContext_97->chunks_0+c_48;
    thread float3 fl_1;
    thread float3 tl_1;
    thread float3 rep_f_2;
    thread float3 rep_m_2;
    chunk_loads_0(c_48, c_48, &rg_16->rot_0, k_25, dt_14, contact_4, 0.0f, &fl_1, &tl_1, &rep_f_2, &rep_m_2, kernelContext_97);
    float4 _S1490 = float4(_S1489->center_0) ;
    float3 fc_4 = fl_1 + kernelContext_97->params_0->gravity_0.xyz * float3(_S1490.w) ;
    float3 _S1491 = _S1490.xyz;
    float3 _S1492 = (float4(isl_29->com_0) ).xyz;
    float3 _S1493 = rotate_0(&rg_16->rot_0, _S1491 + (float4(*(kernelContext_97->state_0+4U * c_48)) ).xyz - _S1492);
    *f_18 = *f_18 + fc_4;
    *t_20 = *t_20 + (cross(_S1493, fc_4) + tl_1);
    uint4 _S1494 = uint4(_S1489->load_range_0) ;
    if((_S1494.y) > (_S1494.x))
    {
        float3 _S1495 = rotate_0(&rg_16->rot_0, rep_f_2);
        *f_18 = *f_18 + _S1495;
        float3 _S1496 = rotate_0(&rg_16->rot_0, _S1491 - _S1492);
        float3 _S1497 = cross(_S1496, _S1495);
        float3 _S1498 = rotate_0(&rg_16->rot_0, rep_m_2);
        *t_20 = *t_20 + (_S1497 + _S1498);
    }
    return;
}

void chunk_update_1(uint c_49, const Island_natural_0 thread* isl_30, const Rigid_0 thread* rg_17, float dt_15, bool rml_1, uint step_1, bool contact_5, float thread* work_5, float thread* work_err_3, KernelContext_0 thread* kernelContext_98)
{
    ChunkStatic_natural_0 device* _S1499 = kernelContext_98->chunks_0+c_49;
    float3 _S1500 = float3(0.0f) ;
    uint _S1501 = kernelContext_98->index_0[c_49];
    float peak_1 = 0.0f;
    uint e_9 = _S1501;
    float3 fi_6 = _S1500;
    float3 mi_7 = _S1500;
    for(;;)
    {
        if(e_9 < (kernelContext_98->index_0)[c_49 + 1U])
        {
        }
        else
        {
            break;
        }
        uint entry_8 = kernelContext_98->index_0[e_9];
        uint _S1502 = 3U * (entry_8 >> 1U);
        float4 _S1503 = float4(*(kernelContext_98->scratch_0+_S1502)) ;
        if((entry_8 & 1U) == 0U)
        {
            float3 mi_8 = mi_7 + (float4(*(kernelContext_98->scratch_0+(_S1502 + 1U))) ).xyz;
            fi_6 = fi_6 + _S1503.xyz;
            mi_7 = mi_8;
        }
        else
        {
            float3 mi_9 = mi_7 + (float4(*(kernelContext_98->scratch_0+(_S1502 + 2U))) ).xyz;
            fi_6 = fi_6 + - _S1503.xyz;
            mi_7 = mi_9;
        }
        float _S1504 = max(peak_1, _S1503.w);
        uint _S1505 = e_9 + 1U;
        peak_1 = _S1504;
        e_9 = _S1505;
    }
    uint _S1506 = 4U * c_49;
    float3 u_5 = (float4(*(kernelContext_98->state_0+_S1506)) ).xyz;
    uint _S1507 = _S1506 + 1U;
    float3 th_7 = (float4(*(kernelContext_98->state_0+_S1507)) ).xyz;
    uint _S1508 = _S1506 + 2U;
    float3 v_14 = (float4(*(kernelContext_98->state_0+_S1508)) ).xyz;
    uint _S1509 = _S1506 + 3U;
    float3 w_9 = (float4(*(kernelContext_98->state_0+_S1509)) ).xyz;
    float4 _S1510 = float4(_S1499->center_0) ;
    float mass_1 = _S1510.w;
    float3 _S1511 = _S1510.xyz;
    float3 _S1512 = (float4(isl_30->com_0) ).xyz;
    float3 _S1513 = rotate_0(&rg_17->rot_0, _S1511 + u_5 - _S1512);
    thread float3 f_load_1;
    thread float3 t_load_1;
    thread float3 rep_f_3;
    thread float3 rep_m_3;
    chunk_loads_0(c_49, c_49, &rg_17->rot_0, step_1, dt_15, contact_5, dt_15, &f_load_1, &t_load_1, &rep_f_3, &rep_m_3, kernelContext_98);
    record_chunk_load_0(c_49, f_load_1, t_load_1, kernelContext_98);
    float3 _S1514 = float3(mass_1) ;
    float3 f_world_3 = f_load_1 + kernelContext_98->params_0->gravity_0.xyz * _S1514;
    float3 t_world_3 = t_load_1;
    float3 f_world_4;
    float3 t_world_4;
    if(rml_1)
    {
        float3 f_world_5 = f_world_3 - (rg_17->a_0 + cross(rg_17->alpha_0, _S1513) + cross(rg_17->w_1, cross(rg_17->w_1, _S1513))) * _S1514;
        float4 _S1515 = float4(_S1499->inertia0_1) ;
        float4 _S1516 = float4(_S1499->inertia1_1) ;
        float4 _S1517 = float4(_S1499->inertia2_1) ;
        float3 _S1518 = world_mul_0(&rg_17->rot_0, _S1515, _S1516, _S1517, rg_17->alpha_0);
        float3 _S1519 = world_mul_0(&rg_17->rot_0, _S1515, _S1516, _S1517, rg_17->w_1);
        float3 t_world_5 = t_world_3 - (_S1518 + cross(rg_17->w_1, _S1519));
        f_world_4 = f_world_5;
        t_world_4 = t_world_5;
    }
    else
    {
        f_world_4 = f_world_3;
        t_world_4 = t_world_3;
    }
    float3 _S1520 = inverse_rotate_1(&rg_17->rot_0, f_world_4);
    float3 _S1521 = inverse_rotate_1(&rg_17->rot_0, t_world_4);
    float3 f_ext_1;
    float3 m_ext_2;
    if(rml_1)
    {
        float3 _S1522 = inverse_rotate_1(&rg_17->rot_0, rg_17->w_1);
        float4 _S1523 = float4(_S1499->inertia0_1) ;
        float4 _S1524 = float4(_S1499->inertia1_1) ;
        float4 _S1525 = float4(_S1499->inertia2_1) ;
        float3 i_w_1 = rows_mul_0(_S1523, _S1524, _S1525, w_9);
        float3 m_ext_3 = _S1521 - (cross(_S1522, i_w_1) + cross(w_9, rows_mul_0(_S1523, _S1524, _S1525, _S1522)) + cross(w_9, i_w_1));
        f_ext_1 = _S1520 - cross(_S1522, v_14) * float3((2.0f * mass_1)) ;
        m_ext_2 = m_ext_3;
    }
    else
    {
        f_ext_1 = _S1520;
        m_ext_2 = _S1521;
    }
    float3 f_19 = f_ext_1 + rep_f_3 + fi_6;
    float3 m_8 = m_ext_2 + rep_m_3 + mi_7;
    uint support_2 = (uint4(_S1499->info_1) ).x;
    float3 _S1526 = float3((float4(*(kernelContext_98->state_0+_S1507)) ).w, (float4(*(kernelContext_98->state_0+_S1508)) ).w, (float4(*(kernelContext_98->state_0+_S1509)) ).w);
    float3 reaction_2;
    float3 u_6;
    float3 th_8;
    float3 v_15;
    float3 w_10;
    if(support_2 == 1U)
    {
        reaction_2 = - f_19;
        u_6 = u_5;
        th_8 = th_7;
        v_15 = _S1500;
        w_10 = _S1500;
    }
    else
    {
        float4 _S1527 = float4(_S1499->scale_0) ;
        float3 w_11 = w_9 + rows_mul_0(float4(_S1499->inv0_1) , float4(_S1499->inv1_1) , float4(_S1499->inv2_1) , m_8) * float3((dt_15 * _S1527.z)) ;
        float3 _S1528 = float3(dt_15) ;
        float3 th_9 = th_7 + w_11 * _S1528;
        if(support_2 == 2U)
        {
            reaction_2 = - f_19;
            u_6 = u_5;
            th_8 = _S1500;
        }
        else
        {
            float3 v_16 = v_14 + f_19 * float3((dt_15 * _S1527.y)) ;
            float3 u_7 = u_5 + v_16 * _S1528;
            reaction_2 = _S1526;
            u_6 = u_7;
            th_8 = v_16;
        }
        float3 _S1529 = th_8;
        th_8 = th_9;
        v_15 = _S1529;
        w_10 = w_11;
    }
    *(kernelContext_98->state_0+_S1506) = packed_float4(float4(u_6, peak_1)) ;
    *(kernelContext_98->state_0+_S1507) = packed_float4(float4(th_8, reaction_2.x)) ;
    *(kernelContext_98->state_0+_S1508) = packed_float4(float4(v_15, reaction_2.y)) ;
    *(kernelContext_98->state_0+_S1509) = packed_float4(float4(w_10, reaction_2.z)) ;
    float3 _S1530 = rotate_0(&rg_17->rot_0, _S1511 + u_6 - _S1512);
    float3 _S1531 = rg_17->vel_0 + rg_17->vel_err_0 + cross(rg_17->w_1, _S1530);
    float3 _S1532 = rotate_0(&rg_17->rot_0, v_15);
    float3 v_world_1 = _S1531 + _S1532;
    float3 _S1533 = rotate_0(&rg_17->rot_0, w_10);
    comp_add1_0(work_5, work_err_3, (dot(f_load_1, v_world_1) + dot(t_load_1, rg_17->w_1 + _S1533)) * dt_15);
    return;
}

[[kernel]] void island_frame(uint3 group_8 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], Params_0 constant* params_12 [[buffer(0)]], Island_natural_0 device* islands_12 [[buffer(9)]], uint device* index_12 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(3)]], packed_float4 device* state_14 [[buffer(6)]], packed_float4 device* scratch_12 [[buffer(8)]], packed_float4 device* contact_state_12 [[buffer(11)]], packed_float4 device* loads_12 [[buffer(5)]], Impactor_natural_0 device* impactors_12 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_12 [[buffer(7)]], BondStatic_natural_0 device* bonds_12 [[buffer(2)]], MaterialTable_0 constant* materials_12 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_99;
    (&kernelContext_99)->params_0 = params_12;
    (&kernelContext_99)->islands_0 = islands_12;
    (&kernelContext_99)->index_0 = index_12;
    (&kernelContext_99)->chunks_0 = chunks_12;
    (&kernelContext_99)->state_0 = state_14;
    (&kernelContext_99)->scratch_0 = scratch_12;
    (&kernelContext_99)->contact_state_0 = contact_state_12;
    (&kernelContext_99)->loads_0 = loads_12;
    (&kernelContext_99)->impactors_0 = impactors_12;
    (&kernelContext_99)->bond_dyn_0 = bond_dyn_12;
    (&kernelContext_99)->bonds_0 = bonds_12;
    (&kernelContext_99)->materials_0 = materials_12;
    threadgroup array<float4, int(32)> g_part_a_12;
    (&kernelContext_99)->g_part_a_0 = &g_part_a_12;
    threadgroup array<float4, int(32)> g_part_b_12;
    (&kernelContext_99)->g_part_b_0 = &g_part_b_12;
    threadgroup array<uint, int(5)> g_seen_12;
    (&kernelContext_99)->g_seen_0 = &g_seen_12;
    threadgroup uint g_wide_run_12;
    (&kernelContext_99)->g_wide_run_0 = &g_wide_run_12;
    threadgroup array<uint, int(4096)> g_label_12;
    (&kernelContext_99)->g_label_0 = &g_label_12;
    threadgroup uint g_changed_12;
    (&kernelContext_99)->g_changed_0 = &g_changed_12;
    threadgroup uint g_halt_12;
    (&kernelContext_99)->g_halt_0 = &g_halt_12;
    threadgroup uint g_run_12;
    (&kernelContext_99)->g_run_0 = &g_run_12;
    threadgroup array<uint, int(512)> g_label_narrow_12;
    (&kernelContext_99)->g_label_narrow_0 = &g_label_narrow_12;
    threadgroup uint g_changed_narrow_12;
    (&kernelContext_99)->g_changed_narrow_0 = &g_changed_narrow_12;
    island_frame_at_1(group_8.x, thread_8.x, &kernelContext_99);
    return;
}

[[kernel]] void contact_islands(uint3 group_9 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], Params_0 constant* params_13 [[buffer(0)]], Island_natural_0 device* islands_13 [[buffer(9)]], uint device* index_13 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(3)]], packed_float4 device* state_15 [[buffer(6)]], packed_float4 device* scratch_13 [[buffer(8)]], packed_float4 device* contact_state_13 [[buffer(11)]], packed_float4 device* loads_13 [[buffer(5)]], Impactor_natural_0 device* impactors_13 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_13 [[buffer(7)]], BondStatic_natural_0 device* bonds_13 [[buffer(2)]], MaterialTable_0 constant* materials_13 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_100;
    (&kernelContext_100)->params_0 = params_13;
    (&kernelContext_100)->islands_0 = islands_13;
    (&kernelContext_100)->index_0 = index_13;
    (&kernelContext_100)->chunks_0 = chunks_13;
    (&kernelContext_100)->state_0 = state_15;
    (&kernelContext_100)->scratch_0 = scratch_13;
    (&kernelContext_100)->contact_state_0 = contact_state_13;
    (&kernelContext_100)->loads_0 = loads_13;
    (&kernelContext_100)->impactors_0 = impactors_13;
    (&kernelContext_100)->bond_dyn_0 = bond_dyn_13;
    (&kernelContext_100)->bonds_0 = bonds_13;
    (&kernelContext_100)->materials_0 = materials_13;
    threadgroup array<float4, int(32)> g_part_a_13;
    (&kernelContext_100)->g_part_a_0 = &g_part_a_13;
    threadgroup array<float4, int(32)> g_part_b_13;
    (&kernelContext_100)->g_part_b_0 = &g_part_b_13;
    threadgroup array<uint, int(5)> g_seen_13;
    (&kernelContext_100)->g_seen_0 = &g_seen_13;
    threadgroup uint g_wide_run_13;
    (&kernelContext_100)->g_wide_run_0 = &g_wide_run_13;
    threadgroup array<uint, int(4096)> g_label_13;
    (&kernelContext_100)->g_label_0 = &g_label_13;
    threadgroup uint g_changed_13;
    (&kernelContext_100)->g_changed_0 = &g_changed_13;
    threadgroup uint g_halt_13;
    (&kernelContext_100)->g_halt_0 = &g_halt_13;
    threadgroup uint g_run_13;
    (&kernelContext_100)->g_run_0 = &g_run_13;
    threadgroup array<uint, int(512)> g_label_narrow_13;
    (&kernelContext_100)->g_label_narrow_0 = &g_label_narrow_13;
    threadgroup uint g_changed_narrow_13;
    (&kernelContext_100)->g_changed_narrow_0 = &g_changed_narrow_13;
    island_frame_at_0(group_9.x, thread_9.x, &kernelContext_100);
    return;
}

void wide_chunks_at_0(uint gx_0, uint tid_22, KernelContext_0 thread* kernelContext_101)
{
    WideGroup_0 _S1534 = wide_group_0(kernelContext_101->params_0->wide_chunk_table_0, gx_0, kernelContext_101);
    thread Island_natural_0 _S1535 = *(kernelContext_101->islands_0+_S1534.island_3);
    bool _S1536 = wide_enter_1(tid_22, _S1534.island_3, kernelContext_101);
    if(!_S1536)
    {
        return;
    }
    uint _S1537 = (uint4((&_S1535)->info_0) ).x;
    bool anchored_0 = (_S1537 & 1U) != 0U;
    if((_S1537 & 16U) != 0U)
    {
        if(tid_22 == 0U)
        {
            *(kernelContext_101->scratch_0+(kernelContext_101->params_0->wide_base_0 + 8U * gx_0 + 6U)) = packed_float4(float4(0.0f) ) ;
        }
        return;
    }
    thread WideGroup_0 _S1538 = _S1534;
    Rigid_0 _S1539 = wide_rigid_frame_1(tid_22, &_S1535, &_S1538, kernelContext_101);
    thread float work_6 = 0.0f;
    thread float work_err_4 = 0.0f;
    float3 _S1540 = float3(0.0f) ;
    thread float3 tu_5 = _S1540;
    thread float3 pv_5 = _S1540;
    uint c_50 = _S1534.begin_0 + tid_22;
    if(c_50 < (_S1534.end_0))
    {
        float _S1541 = kernelContext_101->params_0->dt_0;
        bool _S1542 = (kernelContext_101->params_0->rigid_motion_loads_0) != 0U;
        uint _S1543 = wide_step_1(&_S1535, kernelContext_101);
        bool _S1544 = (_S1537 & 4U) != 0U;
        thread Rigid_0 _S1545 = _S1539;
        chunk_update_1(c_50, &_S1535, &_S1545, _S1541, _S1542, _S1543, _S1544, &work_6, &work_err_4, kernelContext_101);
        if(!anchored_0)
        {
            drift_moments_0(c_50, &tu_5, &pv_5, kernelContext_101);
        }
    }
    thread float3 wsum_2 = float3(work_6, work_err_4, 0.0f);
    thread float3 unused_5 = _S1540;
    group_sum3_0(tid_22, &wsum_2, &unused_5, kernelContext_101);
    bool _S1546 = !anchored_0;
    if(_S1546)
    {
        group_sum3_0(tid_22, &tu_5, &pv_5, kernelContext_101);
    }
    if(tid_22 == 0U)
    {
        uint _S1547 = 8U * gx_0;
        *(kernelContext_101->scratch_0+(kernelContext_101->params_0->wide_base_0 + _S1547 + 6U)) = packed_float4(float4(wsum_2, 0.0f)) ;
        if(_S1546)
        {
            wide_store_0(gx_0, 2U, tu_5, pv_5, kernelContext_101);
        }
        *(kernelContext_101->scratch_0+(kernelContext_101->params_0->wide_base_0 + _S1547 + 7U)) = packed_float4(float4(float((*kernelContext_101->g_seen_0)[0U]), float((*kernelContext_101->g_seen_0)[1U]), float((*kernelContext_101->g_seen_0)[2U]), 0.0f)) ;
    }
    return;
}

[[kernel]] void wide_chunks(uint3 group_10 [[threadgroup_position_in_grid]], uint3 thread_10 [[thread_position_in_threadgroup]], Params_0 constant* params_14 [[buffer(0)]], Island_natural_0 device* islands_14 [[buffer(9)]], uint device* index_14 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_14 [[buffer(3)]], packed_float4 device* state_16 [[buffer(6)]], packed_float4 device* scratch_14 [[buffer(8)]], packed_float4 device* contact_state_14 [[buffer(11)]], packed_float4 device* loads_14 [[buffer(5)]], Impactor_natural_0 device* impactors_14 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_14 [[buffer(7)]], BondStatic_natural_0 device* bonds_14 [[buffer(2)]], MaterialTable_0 constant* materials_14 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_102;
    (&kernelContext_102)->params_0 = params_14;
    (&kernelContext_102)->islands_0 = islands_14;
    (&kernelContext_102)->index_0 = index_14;
    (&kernelContext_102)->chunks_0 = chunks_14;
    (&kernelContext_102)->state_0 = state_16;
    (&kernelContext_102)->scratch_0 = scratch_14;
    (&kernelContext_102)->contact_state_0 = contact_state_14;
    (&kernelContext_102)->loads_0 = loads_14;
    (&kernelContext_102)->impactors_0 = impactors_14;
    (&kernelContext_102)->bond_dyn_0 = bond_dyn_14;
    (&kernelContext_102)->bonds_0 = bonds_14;
    (&kernelContext_102)->materials_0 = materials_14;
    threadgroup array<float4, int(32)> g_part_a_14;
    (&kernelContext_102)->g_part_a_0 = &g_part_a_14;
    threadgroup array<float4, int(32)> g_part_b_14;
    (&kernelContext_102)->g_part_b_0 = &g_part_b_14;
    threadgroup array<uint, int(5)> g_seen_14;
    (&kernelContext_102)->g_seen_0 = &g_seen_14;
    threadgroup uint g_wide_run_14;
    (&kernelContext_102)->g_wide_run_0 = &g_wide_run_14;
    threadgroup array<uint, int(4096)> g_label_14;
    (&kernelContext_102)->g_label_0 = &g_label_14;
    threadgroup uint g_changed_14;
    (&kernelContext_102)->g_changed_0 = &g_changed_14;
    threadgroup uint g_halt_14;
    (&kernelContext_102)->g_halt_0 = &g_halt_14;
    threadgroup uint g_run_14;
    (&kernelContext_102)->g_run_0 = &g_run_14;
    threadgroup array<uint, int(512)> g_label_narrow_14;
    (&kernelContext_102)->g_label_narrow_0 = &g_label_narrow_14;
    threadgroup uint g_changed_narrow_14;
    (&kernelContext_102)->g_changed_narrow_0 = &g_changed_narrow_14;
    wide_chunks_at_0(group_10.x, thread_10.x, &kernelContext_102);
    return;
}

[[kernel]] void sums_and_impactors(uint3 group_11 [[threadgroup_position_in_grid]], uint3 thread_11 [[thread_position_in_threadgroup]], Params_0 constant* params_15 [[buffer(0)]], Island_natural_0 device* islands_15 [[buffer(9)]], uint device* index_15 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_15 [[buffer(3)]], packed_float4 device* state_17 [[buffer(6)]], packed_float4 device* scratch_15 [[buffer(8)]], packed_float4 device* contact_state_15 [[buffer(11)]], packed_float4 device* loads_15 [[buffer(5)]], Impactor_natural_0 device* impactors_15 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_15 [[buffer(7)]], BondStatic_natural_0 device* bonds_15 [[buffer(2)]], MaterialTable_0 constant* materials_15 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_103;
    (&kernelContext_103)->params_0 = params_15;
    (&kernelContext_103)->islands_0 = islands_15;
    (&kernelContext_103)->index_0 = index_15;
    (&kernelContext_103)->chunks_0 = chunks_15;
    (&kernelContext_103)->state_0 = state_17;
    (&kernelContext_103)->scratch_0 = scratch_15;
    (&kernelContext_103)->contact_state_0 = contact_state_15;
    (&kernelContext_103)->loads_0 = loads_15;
    (&kernelContext_103)->impactors_0 = impactors_15;
    (&kernelContext_103)->bond_dyn_0 = bond_dyn_15;
    (&kernelContext_103)->bonds_0 = bonds_15;
    (&kernelContext_103)->materials_0 = materials_15;
    threadgroup array<float4, int(32)> g_part_a_15;
    (&kernelContext_103)->g_part_a_0 = &g_part_a_15;
    threadgroup array<float4, int(32)> g_part_b_15;
    (&kernelContext_103)->g_part_b_0 = &g_part_b_15;
    threadgroup array<uint, int(5)> g_seen_15;
    (&kernelContext_103)->g_seen_0 = &g_seen_15;
    threadgroup uint g_wide_run_15;
    (&kernelContext_103)->g_wide_run_0 = &g_wide_run_15;
    threadgroup array<uint, int(4096)> g_label_15;
    (&kernelContext_103)->g_label_0 = &g_label_15;
    threadgroup uint g_changed_15;
    (&kernelContext_103)->g_changed_0 = &g_changed_15;
    threadgroup uint g_halt_15;
    (&kernelContext_103)->g_halt_0 = &g_halt_15;
    threadgroup uint g_run_15;
    (&kernelContext_103)->g_run_0 = &g_run_15;
    threadgroup array<uint, int(512)> g_label_narrow_15;
    (&kernelContext_103)->g_label_narrow_0 = &g_label_narrow_15;
    threadgroup uint g_changed_narrow_15;
    (&kernelContext_103)->g_changed_narrow_0 = &g_changed_narrow_15;
    uint sums_groups_0 = (params_15->seg_count_0 + 255U) / 256U;
    uint _S1548 = group_11.x;
    if(_S1548 < sums_groups_0)
    {
        contact_sum_at_0(_S1548 * 256U + thread_11.x, &kernelContext_103);
    }
    else
    {
        impactor_integrate_at_0(_S1548 - sums_groups_0, thread_11.x, &kernelContext_103);
    }
    return;
}

[[kernel]] void islands_and_wide_chunks(uint3 group_12 [[threadgroup_position_in_grid]], uint3 thread_12 [[thread_position_in_threadgroup]], Params_0 constant* params_16 [[buffer(0)]], Island_natural_0 device* islands_16 [[buffer(9)]], uint device* index_16 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_16 [[buffer(3)]], packed_float4 device* state_18 [[buffer(6)]], packed_float4 device* scratch_16 [[buffer(8)]], packed_float4 device* contact_state_16 [[buffer(11)]], packed_float4 device* loads_16 [[buffer(5)]], Impactor_natural_0 device* impactors_16 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_16 [[buffer(7)]], BondStatic_natural_0 device* bonds_16 [[buffer(2)]], MaterialTable_0 constant* materials_16 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_104;
    (&kernelContext_104)->params_0 = params_16;
    (&kernelContext_104)->islands_0 = islands_16;
    (&kernelContext_104)->index_0 = index_16;
    (&kernelContext_104)->chunks_0 = chunks_16;
    (&kernelContext_104)->state_0 = state_18;
    (&kernelContext_104)->scratch_0 = scratch_16;
    (&kernelContext_104)->contact_state_0 = contact_state_16;
    (&kernelContext_104)->loads_0 = loads_16;
    (&kernelContext_104)->impactors_0 = impactors_16;
    (&kernelContext_104)->bond_dyn_0 = bond_dyn_16;
    (&kernelContext_104)->bonds_0 = bonds_16;
    (&kernelContext_104)->materials_0 = materials_16;
    threadgroup array<float4, int(32)> g_part_a_16;
    (&kernelContext_104)->g_part_a_0 = &g_part_a_16;
    threadgroup array<float4, int(32)> g_part_b_16;
    (&kernelContext_104)->g_part_b_0 = &g_part_b_16;
    threadgroup array<uint, int(5)> g_seen_16;
    (&kernelContext_104)->g_seen_0 = &g_seen_16;
    threadgroup uint g_wide_run_16;
    (&kernelContext_104)->g_wide_run_0 = &g_wide_run_16;
    threadgroup array<uint, int(4096)> g_label_16;
    (&kernelContext_104)->g_label_0 = &g_label_16;
    threadgroup uint g_changed_16;
    (&kernelContext_104)->g_changed_0 = &g_changed_16;
    threadgroup uint g_halt_16;
    (&kernelContext_104)->g_halt_0 = &g_halt_16;
    threadgroup uint g_run_16;
    (&kernelContext_104)->g_run_0 = &g_run_16;
    threadgroup array<uint, int(512)> g_label_narrow_16;
    (&kernelContext_104)->g_label_narrow_0 = &g_label_narrow_16;
    threadgroup uint g_changed_narrow_16;
    (&kernelContext_104)->g_changed_narrow_0 = &g_changed_narrow_16;
    uint _S1549 = group_12.x;
    if(_S1549 < (params_16->halt_index_0))
    {
        island_frame_at_1(_S1549, thread_12.x, &kernelContext_104);
    }
    else
    {
        wide_chunks_at_0(_S1549 - params_16->halt_index_0, thread_12.x, &kernelContext_104);
    }
    return;
}

[[kernel]] void contact_islands_and_wide_chunks(uint3 group_13 [[threadgroup_position_in_grid]], uint3 thread_13 [[thread_position_in_threadgroup]], Params_0 constant* params_17 [[buffer(0)]], Island_natural_0 device* islands_17 [[buffer(9)]], uint device* index_17 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_17 [[buffer(3)]], packed_float4 device* state_19 [[buffer(6)]], packed_float4 device* scratch_17 [[buffer(8)]], packed_float4 device* contact_state_17 [[buffer(11)]], packed_float4 device* loads_17 [[buffer(5)]], Impactor_natural_0 device* impactors_17 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_17 [[buffer(7)]], BondStatic_natural_0 device* bonds_17 [[buffer(2)]], MaterialTable_0 constant* materials_17 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_105;
    (&kernelContext_105)->params_0 = params_17;
    (&kernelContext_105)->islands_0 = islands_17;
    (&kernelContext_105)->index_0 = index_17;
    (&kernelContext_105)->chunks_0 = chunks_17;
    (&kernelContext_105)->state_0 = state_19;
    (&kernelContext_105)->scratch_0 = scratch_17;
    (&kernelContext_105)->contact_state_0 = contact_state_17;
    (&kernelContext_105)->loads_0 = loads_17;
    (&kernelContext_105)->impactors_0 = impactors_17;
    (&kernelContext_105)->bond_dyn_0 = bond_dyn_17;
    (&kernelContext_105)->bonds_0 = bonds_17;
    (&kernelContext_105)->materials_0 = materials_17;
    threadgroup array<float4, int(32)> g_part_a_17;
    (&kernelContext_105)->g_part_a_0 = &g_part_a_17;
    threadgroup array<float4, int(32)> g_part_b_17;
    (&kernelContext_105)->g_part_b_0 = &g_part_b_17;
    threadgroup array<uint, int(5)> g_seen_17;
    (&kernelContext_105)->g_seen_0 = &g_seen_17;
    threadgroup uint g_wide_run_17;
    (&kernelContext_105)->g_wide_run_0 = &g_wide_run_17;
    threadgroup array<uint, int(4096)> g_label_17;
    (&kernelContext_105)->g_label_0 = &g_label_17;
    threadgroup uint g_changed_17;
    (&kernelContext_105)->g_changed_0 = &g_changed_17;
    threadgroup uint g_halt_17;
    (&kernelContext_105)->g_halt_0 = &g_halt_17;
    threadgroup uint g_run_17;
    (&kernelContext_105)->g_run_0 = &g_run_17;
    threadgroup array<uint, int(512)> g_label_narrow_17;
    (&kernelContext_105)->g_label_narrow_0 = &g_label_narrow_17;
    threadgroup uint g_changed_narrow_17;
    (&kernelContext_105)->g_changed_narrow_0 = &g_changed_narrow_17;
    uint _S1550 = group_13.x;
    if(_S1550 < (params_17->halt_index_0))
    {
        island_frame_at_0(_S1550, thread_13.x, &kernelContext_105);
    }
    else
    {
        wide_chunks_at_0(_S1550 - params_17->halt_index_0, thread_13.x, &kernelContext_105);
    }
    return;
}

void wide_bonds_at_0(uint gx_1, uint tid_23, KernelContext_0 thread* kernelContext_106)
{
    bool bond_group_0 = gx_1 < (kernelContext_106->params_0->wide_bond_groups_0);
    WideGroup_0 wg_2;
    if(bond_group_0)
    {
        WideGroup_0 _S1551 = wide_group_0(kernelContext_106->params_0->wide_bond_table_0, gx_1, kernelContext_106);
        wg_2 = _S1551;
    }
    else
    {
        WideGroup_0 _S1552 = wide_group_0(kernelContext_106->params_0->wide_chunk_table_0, gx_1 - kernelContext_106->params_0->wide_bond_groups_0, kernelContext_106);
        wg_2 = _S1552;
    }
    WideGroup_0 _S1553 = wg_2;
    thread Island_natural_0 _S1554 = *(kernelContext_106->islands_0+wg_2.island_3);
    bool _S1555 = wide_enter_1(tid_23, wg_2.island_3, kernelContext_106);
    if(!_S1555)
    {
        return;
    }
    uint _S1556 = wide_step_1(&_S1554, kernelContext_106);
    if(bond_group_0)
    {
        uint4 _S1557 = uint4((&_S1554)->info_0) ;
        if(((_S1557.x) & 16U) != 0U)
        {
            return;
        }
        uint i_24 = wg_2.begin_0 + tid_23;
        bool _S1558;
        if(i_24 < (wg_2.end_0))
        {
            bool _S1559 = bond_update_0(i_24, kernelContext_106->params_0->dt_0, (kernelContext_106->params_0->fracture_0) != 0U, _S1557.w + 1U, true, kernelContext_106);
            _S1558 = _S1559;
        }
        else
        {
            _S1558 = false;
        }
        if(_S1558)
        {
            (kernelContext_106->islands_0+_S1553.island_3)->info_0[int(2)] = (_S1557.z) | 2U;
        }
        return;
    }
    uint _S1560 = (uint4((&_S1554)->info_0) ).x;
    if((_S1560 & 1U) != 0U)
    {
        return;
    }
    float3 _S1561 = float3(0.0f) ;
    thread float3 f_20 = _S1561;
    thread float3 t_21 = _S1561;
    uint c_51 = wg_2.begin_0 + tid_23;
    if(c_51 < (wg_2.end_0))
    {
        Rigid_0 _S1562 = rigid_of_1(&_S1554);
        float _S1563 = kernelContext_106->params_0->dt_0;
        bool _S1564 = (_S1560 & 4U) != 0U;
        thread Rigid_0 _S1565 = _S1562;
        net_load_1(c_51, &_S1554, &_S1565, _S1556, _S1563, _S1564, &f_20, &t_21, kernelContext_106);
    }
    group_sum3_0(tid_23, &f_20, &t_21, kernelContext_106);
    if(tid_23 == 0U)
    {
        wide_store_0(gx_1 - kernelContext_106->params_0->wide_bond_groups_0, 0U, f_20, t_21, kernelContext_106);
    }
    return;
}

[[kernel]] void wide_bonds(uint3 group_14 [[threadgroup_position_in_grid]], uint3 thread_14 [[thread_position_in_threadgroup]], Params_0 constant* params_18 [[buffer(0)]], Island_natural_0 device* islands_18 [[buffer(9)]], uint device* index_18 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_18 [[buffer(3)]], packed_float4 device* state_20 [[buffer(6)]], packed_float4 device* scratch_18 [[buffer(8)]], packed_float4 device* contact_state_18 [[buffer(11)]], packed_float4 device* loads_18 [[buffer(5)]], Impactor_natural_0 device* impactors_18 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_18 [[buffer(7)]], BondStatic_natural_0 device* bonds_18 [[buffer(2)]], MaterialTable_0 constant* materials_18 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_107;
    (&kernelContext_107)->params_0 = params_18;
    (&kernelContext_107)->islands_0 = islands_18;
    (&kernelContext_107)->index_0 = index_18;
    (&kernelContext_107)->chunks_0 = chunks_18;
    (&kernelContext_107)->state_0 = state_20;
    (&kernelContext_107)->scratch_0 = scratch_18;
    (&kernelContext_107)->contact_state_0 = contact_state_18;
    (&kernelContext_107)->loads_0 = loads_18;
    (&kernelContext_107)->impactors_0 = impactors_18;
    (&kernelContext_107)->bond_dyn_0 = bond_dyn_18;
    (&kernelContext_107)->bonds_0 = bonds_18;
    (&kernelContext_107)->materials_0 = materials_18;
    threadgroup array<float4, int(32)> g_part_a_18;
    (&kernelContext_107)->g_part_a_0 = &g_part_a_18;
    threadgroup array<float4, int(32)> g_part_b_18;
    (&kernelContext_107)->g_part_b_0 = &g_part_b_18;
    threadgroup array<uint, int(5)> g_seen_18;
    (&kernelContext_107)->g_seen_0 = &g_seen_18;
    threadgroup uint g_wide_run_18;
    (&kernelContext_107)->g_wide_run_0 = &g_wide_run_18;
    threadgroup array<uint, int(4096)> g_label_18;
    (&kernelContext_107)->g_label_0 = &g_label_18;
    threadgroup uint g_changed_18;
    (&kernelContext_107)->g_changed_0 = &g_changed_18;
    threadgroup uint g_halt_18;
    (&kernelContext_107)->g_halt_0 = &g_halt_18;
    threadgroup uint g_run_18;
    (&kernelContext_107)->g_run_0 = &g_run_18;
    threadgroup array<uint, int(512)> g_label_narrow_18;
    (&kernelContext_107)->g_label_narrow_0 = &g_label_narrow_18;
    threadgroup uint g_changed_narrow_18;
    (&kernelContext_107)->g_changed_narrow_0 = &g_changed_narrow_18;
    wide_bonds_at_0(group_14.x, thread_14.x, &kernelContext_107);
    return;
}

void bond_group_at_0(uint table_1, uint gx_2, uint tid_24, KernelContext_0 thread* kernelContext_108)
{
    WideGroup_0 _S1566 = wide_group_0(table_1, gx_2, kernelContext_108);
    Island_natural_0 isl_31 = *(kernelContext_108->islands_0+_S1566.island_3);
    bool _S1567 = wide_enter_1(tid_24, _S1566.island_3, kernelContext_108);
    bool _S1568;
    if(!_S1567)
    {
        _S1568 = true;
    }
    else
    {
        _S1568 = (((uint4(isl_31.info_0) ).x) & 16U) != 0U;
    }
    if(_S1568)
    {
        return;
    }
    uint i_25 = _S1566.begin_0 + tid_24;
    if(i_25 < (_S1566.end_0))
    {
        bool _S1569 = bond_update_0(i_25, kernelContext_108->params_0->dt_0, (kernelContext_108->params_0->fracture_0) != 0U, (uint4(isl_31.info_0) ).w + 1U, true, kernelContext_108);
        _S1568 = _S1569;
    }
    else
    {
        _S1568 = false;
    }
    if(_S1568)
    {
        (kernelContext_108->islands_0+_S1566.island_3)->info_0[int(2)] = ((uint4(isl_31.info_0) ).z) | 2U;
    }
    return;
}

[[kernel]] void sums_impactors_bonds(uint3 group_15 [[threadgroup_position_in_grid]], uint3 thread_15 [[thread_position_in_threadgroup]], Params_0 constant* params_19 [[buffer(0)]], Island_natural_0 device* islands_19 [[buffer(9)]], uint device* index_19 [[buffer(4)]], ChunkStatic_natural_0 device* chunks_19 [[buffer(3)]], packed_float4 device* state_21 [[buffer(6)]], packed_float4 device* scratch_19 [[buffer(8)]], packed_float4 device* contact_state_19 [[buffer(11)]], packed_float4 device* loads_19 [[buffer(5)]], Impactor_natural_0 device* impactors_19 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_19 [[buffer(7)]], BondStatic_natural_0 device* bonds_19 [[buffer(2)]], MaterialTable_0 constant* materials_19 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_109;
    (&kernelContext_109)->params_0 = params_19;
    (&kernelContext_109)->islands_0 = islands_19;
    (&kernelContext_109)->index_0 = index_19;
    (&kernelContext_109)->chunks_0 = chunks_19;
    (&kernelContext_109)->state_0 = state_21;
    (&kernelContext_109)->scratch_0 = scratch_19;
    (&kernelContext_109)->contact_state_0 = contact_state_19;
    (&kernelContext_109)->loads_0 = loads_19;
    (&kernelContext_109)->impactors_0 = impactors_19;
    (&kernelContext_109)->bond_dyn_0 = bond_dyn_19;
    (&kernelContext_109)->bonds_0 = bonds_19;
    (&kernelContext_109)->materials_0 = materials_19;
    threadgroup array<float4, int(32)> g_part_a_19;
    (&kernelContext_109)->g_part_a_0 = &g_part_a_19;
    threadgroup array<float4, int(32)> g_part_b_19;
    (&kernelContext_109)->g_part_b_0 = &g_part_b_19;
    threadgroup array<uint, int(5)> g_seen_19;
    (&kernelContext_109)->g_seen_0 = &g_seen_19;
    threadgroup uint g_wide_run_19;
    (&kernelContext_109)->g_wide_run_0 = &g_wide_run_19;
    threadgroup array<uint, int(4096)> g_label_19;
    (&kernelContext_109)->g_label_0 = &g_label_19;
    threadgroup uint g_changed_19;
    (&kernelContext_109)->g_changed_0 = &g_changed_19;
    threadgroup uint g_halt_19;
    (&kernelContext_109)->g_halt_0 = &g_halt_19;
    threadgroup uint g_run_19;
    (&kernelContext_109)->g_run_0 = &g_run_19;
    threadgroup array<uint, int(512)> g_label_narrow_19;
    (&kernelContext_109)->g_label_narrow_0 = &g_label_narrow_19;
    threadgroup uint g_changed_narrow_19;
    (&kernelContext_109)->g_changed_narrow_0 = &g_changed_narrow_19;
    uint sums_groups_1 = (params_19->seg_count_0 + 255U) / 256U;
    uint _S1570 = max(params_19->impactor_count_0, 1U);
    uint _S1571 = group_15.x;
    if(_S1571 < sums_groups_1)
    {
        contact_sum_at_0(_S1571 * 256U + thread_15.x, &kernelContext_109);
    }
    else
    {
        if(_S1571 < (sums_groups_1 + _S1570))
        {
            impactor_integrate_at_0(_S1571 - sums_groups_1, thread_15.x, &kernelContext_109);
        }
        else
        {
            uint gb_0 = _S1571 - sums_groups_1 - _S1570;
            bool narrow_0 = gb_0 < ((&kernelContext_109)->params_0->contact_bond_groups_0);
            uint _S1572;
            if(narrow_0)
            {
                _S1572 = (&kernelContext_109)->params_0->contact_bond_table_0;
            }
            else
            {
                _S1572 = (&kernelContext_109)->params_0->wide_bond_table_0;
            }
            uint _S1573;
            if(narrow_0)
            {
                _S1573 = gb_0;
            }
            else
            {
                _S1573 = gb_0 - (&kernelContext_109)->params_0->contact_bond_groups_0;
            }
            bond_group_at_0(_S1572, _S1573, thread_15.x, &kernelContext_109);
        }
    }
    return;
}

