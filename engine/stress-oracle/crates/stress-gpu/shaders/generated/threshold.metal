#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

struct JointState_0
{
    float damage_0;
    float crush_0;
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

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S1;
    if((st_0->damage_0) < 1.0f)
    {
        _S1 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S1 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S1 = false;
        }
    }
    return _S1;
}

struct ThresholdParams_0
{
    uint island_count_0;
    uint eload_base_0;
    uint pending_0;
    uint impactor_count_0;
    uint engine_base_0;
    uint pad0_0;
    float dt_0;
    float speed_0;
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

struct BondDyn_natural_0
{
    JointState_0 js_0;
    packed_float4 force_lin_0;
    packed_float4 force_ang_0;
    packed_float4 sums_0;
    packed_float4 comps_0;
    packed_uint4 events_0;
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

struct EngineBody_natural_0
{
    packed_float4 pos_inv_mass_0;
    packed_float4 rot_0;
    packed_float4 inv_inertia_flags_0;
    packed_float4 vel_sleep_0;
    packed_float4 ang_vel_gen_0;
    packed_float4 initial_pos_0;
    packed_float4 initial_rot_0;
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

struct KernelContext_0
{
    ThresholdParams_0 constant* params_0;
    uint device* status_0;
    Island_natural_0 device* islands_0;
    BondStatic_natural_0 device* bond_static_0;
    BondDyn_natural_0 device* bond_dyn_0;
    MaterialTable_0 constant* materials_0;
    packed_float4 device* ctab_0;
    packed_float4 device* fast_0;
    EngineBody_natural_0 device* bodies_0;
    ChunkStatic_natural_0 device* chunks_0;
    packed_float4 device* state_0;
    packed_float4 device* impactors_0;
    array<uint, int(256)> threadgroup* g_broke_0;
    array<float3, int(256)> threadgroup* g_lo_0;
    array<float3, int(256)> threadgroup* g_hi_0;
    array<float, int(256)> threadgroup* g_a_0;
    array<uint, int(256)> threadgroup* g_idx_0;
};

float contact_force_0(uint chunk_0, KernelContext_0 thread* kernelContext_0)
{
    return length((float4(*(kernelContext_0->ctab_0+(kernelContext_0->params_0->eload_base_0 + 2U * chunk_0))) ).xyz);
}

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

struct BondDyn_0
{
    JointState_0 js_0;
    float4 force_lin_0;
    float4 force_ang_0;
    float4 sums_0;
    float4 comps_0;
    uint4 events_0;
};

[[kernel]] void threshold_breaks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], ThresholdParams_0 constant* params_1 [[buffer(0)]], uint device* status_1 [[buffer(6)]], Island_natural_0 device* islands_1 [[buffer(1)]], BondStatic_natural_0 device* bond_static_1 [[buffer(2)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(3)]], MaterialTable_0 constant* materials_1 [[buffer(5)]], packed_float4 device* ctab_1 [[buffer(4)]], packed_float4 device* fast_1 [[buffer(11)]], EngineBody_natural_0 device* bodies_1 [[buffer(9)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(7)]], packed_float4 device* state_1 [[buffer(8)]], packed_float4 device* impactors_1 [[buffer(10)]])
{
    uint broke_0;
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->status_0 = status_1;
    (&kernelContext_1)->islands_0 = islands_1;
    (&kernelContext_1)->bond_static_0 = bond_static_1;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_1)->materials_0 = materials_1;
    (&kernelContext_1)->ctab_0 = ctab_1;
    (&kernelContext_1)->fast_0 = fast_1;
    (&kernelContext_1)->bodies_0 = bodies_1;
    (&kernelContext_1)->chunks_0 = chunks_1;
    (&kernelContext_1)->state_0 = state_1;
    (&kernelContext_1)->impactors_0 = impactors_1;
    threadgroup array<uint, int(256)> g_broke_1;
    (&kernelContext_1)->g_broke_0 = &g_broke_1;
    threadgroup array<float3, int(256)> g_lo_1;
    (&kernelContext_1)->g_lo_0 = &g_lo_1;
    threadgroup array<float3, int(256)> g_hi_1;
    (&kernelContext_1)->g_hi_0 = &g_hi_1;
    threadgroup array<float, int(256)> g_a_1;
    (&kernelContext_1)->g_a_0 = &g_a_1;
    threadgroup array<uint, int(256)> g_idx_1;
    (&kernelContext_1)->g_idx_0 = &g_idx_1;
    uint i_0 = group_0.x;
    uint t_0 = thread_1.x;
    bool _S2;
    if(i_0 >= (params_1->island_count_0))
    {
        _S2 = true;
    }
    else
    {
        _S2 = (*((&kernelContext_1)->status_0+0U)) == 0U;
    }
    if(_S2)
    {
        _S2 = true;
    }
    else
    {
        _S2 = ((*((&kernelContext_1)->status_0+(16U + i_0))) & 1U) == 0U;
    }
    uint broke_1;
    if(_S2)
    {
        broke_1 = 1U;
    }
    else
    {
        broke_1 = 0U;
    }
    if((group_uniform_0(broke_1, t_0)) != 0U)
    {
        return;
    }
    Island_natural_0 device* _S3 = (&kernelContext_1)->islands_0+i_0;
    uint4 _S4 = uint4((*_S3).info_0) ;
    float4 _S5 = float4((*_S3).com_0) ;
    float4 _S6 = float4((*_S3).inertia0_0) ;
    float4 _S7 = float4((*_S3).inertia1_0) ;
    float4 _S8 = float4((*_S3).inertia2_0) ;
    float4 _S9 = float4((*_S3).inv0_0) ;
    float4 _S10 = float4((*_S3).inv1_0) ;
    float4 _S11 = float4((*_S3).inv2_0) ;
    float4 _S12 = float4((*_S3).wcom_0) ;
    float4 _S13 = float4((*_S3).winv0_0) ;
    float4 _S14 = float4((*_S3).winv1_0) ;
    float4 _S15 = float4((*_S3).winv2_0) ;
    float4 _S16 = float4((*_S3).rotation_0) ;
    float4 _S17 = float4((*_S3).position_0) ;
    float4 _S18 = float4((*_S3).position_err_0) ;
    float4 _S19 = float4((*_S3).velocity_0) ;
    float4 _S20 = float4((*_S3).velocity_err_0) ;
    float4 _S21 = float4((*_S3).angular_velocity_0) ;
    uint4 _S22 = uint4((*_S3).probes_0) ;
    float4 _S23 = float4((*_S3).energy_0) ;
    float4 _S24 = float4((*_S3).rotation_err_0) ;
    float4 _S25 = float4((*_S3).momentum_0) ;
    float4 _S26 = float4((*_S3).momentum_err_0) ;
    thread Island_0 isl_0;
    (&isl_0)->range_0 = uint4((*_S3).range_0) ;
    (&isl_0)->info_0 = _S4;
    (&isl_0)->com_0 = _S5;
    (&isl_0)->inertia0_0 = _S6;
    (&isl_0)->inertia1_0 = _S7;
    (&isl_0)->inertia2_0 = _S8;
    (&isl_0)->inv0_0 = _S9;
    (&isl_0)->inv1_0 = _S10;
    (&isl_0)->inv2_0 = _S11;
    (&isl_0)->wcom_0 = _S12;
    (&isl_0)->winv0_0 = _S13;
    (&isl_0)->winv1_0 = _S14;
    (&isl_0)->winv2_0 = _S15;
    (&isl_0)->rotation_0 = _S16;
    (&isl_0)->position_0 = _S17;
    (&isl_0)->position_err_0 = _S18;
    (&isl_0)->velocity_0 = _S19;
    (&isl_0)->velocity_err_0 = _S20;
    (&isl_0)->angular_velocity_0 = _S21;
    (&isl_0)->done_0 = (*_S3).done_0;
    (&isl_0)->coverage_0 = (*_S3).coverage_0;
    (&isl_0)->wide_groups_0 = (*_S3).wide_groups_0;
    (&isl_0)->done_pad_0 = (*_S3).done_pad_0;
    (&isl_0)->probes_0 = _S22;
    (&isl_0)->energy_0 = _S23;
    (&isl_0)->rotation_err_0 = _S24;
    (&isl_0)->momentum_0 = _S25;
    (&isl_0)->momentum_err_0 = _S26;
    uint _S27 = (&isl_0)->range_0.z + t_0;
    broke_1 = 0U;
    uint bi_0 = _S27;
    for(;;)
    {
        if(bi_0 < ((&isl_0)->range_0.w))
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_1 = ((&kernelContext_1)->bond_static_0+bi_0)->law_0;
        BondDyn_natural_0 device* _S28 = (&kernelContext_1)->bond_dyn_0+bi_0;
        float4 _S29 = float4((*_S28).force_lin_0) ;
        float4 _S30 = float4((*_S28).force_ang_0) ;
        float4 _S31 = float4((*_S28).sums_0) ;
        float4 _S32 = float4((*_S28).comps_0) ;
        uint4 _S33 = uint4((*_S28).events_0) ;
        thread BondDyn_0 dyn_0;
        (&dyn_0)->js_0 = (*_S28).js_0;
        (&dyn_0)->force_lin_0 = _S29;
        (&dyn_0)->force_ang_0 = _S30;
        (&dyn_0)->sums_0 = _S31;
        (&dyn_0)->comps_0 = _S32;
        (&dyn_0)->events_0 = _S33;
        bool _S34 = ((float4(((&kernelContext_1)->bond_static_0+bi_0)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S35 = (&dyn_0)->js_0;
        bool _S36 = connected_0(&_S35, _S34);
        if(!_S36)
        {
            if(((&kernelContext_1)->params_0->pending_0) != 0U)
            {
                broke_0 = 1U;
            }
            else
            {
                broke_0 = broke_1;
            }
            uint bi_1 = bi_0 + 256U;
            broke_1 = broke_0;
            bi_0 = bi_1;
            continue;
        }
        if(((&kernelContext_1)->params_0->pending_0) != 0U)
        {
            broke_0 = broke_1;
            uint bi_1 = bi_0 + 256U;
            broke_1 = broke_0;
            bi_0 = bi_1;
            continue;
        }
        uint4 _S37 = uint4(law_1.ids_0) ;
        float capacity_0 = (&(&kernelContext_1)->materials_0->m_0[_S37.x])->strength_0.x * (float4(law_1.geom0_0) ).x * (float4(law_1.geom1_0) ).w;
        float _S38 = contact_force_0(_S37.y, &kernelContext_1);
        float _S39 = contact_force_0(_S37.z, &kernelContext_1);
        if((max(_S38, _S39)) > capacity_0)
        {
            (&(&dyn_0)->js_0)->damage_0 = 1.0f;
            (&(&dyn_0)->js_0)->rebar_broken_0 = 1.0f;
            (&dyn_0)->events_0.z = max((&dyn_0)->events_0.z, 1U);
            BondDyn_natural_0 device* _S40 = (&kernelContext_1)->bond_dyn_0+bi_0;
            _S40->js_0 = dyn_0.js_0;
            _S40->force_lin_0 = packed_float4(dyn_0.force_lin_0) ;
            _S40->force_ang_0 = packed_float4(dyn_0.force_ang_0) ;
            _S40->sums_0 = packed_float4(dyn_0.sums_0) ;
            _S40->comps_0 = packed_float4(dyn_0.comps_0) ;
            _S40->events_0 = packed_uint4(dyn_0.events_0) ;
            broke_0 = 1U;
        }
        else
        {
            broke_0 = broke_1;
        }
        uint bi_1 = bi_0 + 256U;
        broke_1 = broke_0;
        bi_0 = bi_1;
    }
    (*(&kernelContext_1)->g_broke_0)[t_0] = broke_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    broke_0 = 128U;
    for(;;)
    {
        if(broke_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_0 < broke_0)
        {
            (*(&kernelContext_1)->g_broke_0)[t_0] = ((*(&kernelContext_1)->g_broke_0)[t_0]) | (*(&kernelContext_1)->g_broke_0)[t_0 + broke_0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        broke_0 = broke_0 >> 1U;
    }
    if(t_0 != 0U)
    {
        return;
    }
    uint device* _S41 = (&kernelContext_1)->status_0+(16U + i_0);
    *_S41 = (*_S41) | 2U;
    if(((*(&kernelContext_1)->g_broke_0)[int(0)]) != 0U)
    {
        (&isl_0)->info_0.z = ((&isl_0)->info_0.z) | 48U;
        Island_natural_0 device* _S42 = (&kernelContext_1)->islands_0+i_0;
        _S42->range_0 = packed_uint4(isl_0.range_0) ;
        _S42->info_0 = packed_uint4(isl_0.info_0) ;
        _S42->com_0 = packed_float4(isl_0.com_0) ;
        _S42->inertia0_0 = packed_float4(isl_0.inertia0_0) ;
        _S42->inertia1_0 = packed_float4(isl_0.inertia1_0) ;
        _S42->inertia2_0 = packed_float4(isl_0.inertia2_0) ;
        _S42->inv0_0 = packed_float4(isl_0.inv0_0) ;
        _S42->inv1_0 = packed_float4(isl_0.inv1_0) ;
        _S42->inv2_0 = packed_float4(isl_0.inv2_0) ;
        _S42->wcom_0 = packed_float4(isl_0.wcom_0) ;
        _S42->winv0_0 = packed_float4(isl_0.winv0_0) ;
        _S42->winv1_0 = packed_float4(isl_0.winv1_0) ;
        _S42->winv2_0 = packed_float4(isl_0.winv2_0) ;
        _S42->rotation_0 = packed_float4(isl_0.rotation_0) ;
        _S42->position_0 = packed_float4(isl_0.position_0) ;
        _S42->position_err_0 = packed_float4(isl_0.position_err_0) ;
        _S42->velocity_0 = packed_float4(isl_0.velocity_0) ;
        _S42->velocity_err_0 = packed_float4(isl_0.velocity_err_0) ;
        _S42->angular_velocity_0 = packed_float4(isl_0.angular_velocity_0) ;
        _S42->done_0 = isl_0.done_0;
        _S42->coverage_0 = isl_0.coverage_0;
        _S42->wide_groups_0 = isl_0.wide_groups_0;
        _S42->done_pad_0 = isl_0.done_pad_0;
        _S42->probes_0 = packed_uint4(isl_0.probes_0) ;
        _S42->energy_0 = packed_float4(isl_0.energy_0) ;
        _S42->rotation_err_0 = packed_float4(isl_0.rotation_err_0) ;
        _S42->momentum_0 = packed_float4(isl_0.momentum_0) ;
        _S42->momentum_err_0 = packed_float4(isl_0.momentum_err_0) ;
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

Quat_0 quat_of_0(float4 q_0)
{
    thread Quat_0 r_0;
    (&r_0)->x_0 = q_0.x;
    (&r_0)->y_0 = q_0.y;
    (&r_0)->z_0 = q_0.z;
    (&r_0)->w_0 = q_0.w;
    return r_0;
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

Quat_0 normalized_0(const Quat_0 thread* q_1)
{
    float n_0 = sqrt(q_1->w_0 * q_1->w_0 + q_1->x_0 * q_1->x_0 + q_1->y_0 * q_1->y_0 + q_1->z_0 * q_1->z_0);
    thread Quat_0 r_2;
    (&r_2)->w_0 = q_1->w_0 / n_0;
    (&r_2)->x_0 = q_1->x_0 / n_0;
    (&r_2)->y_0 = q_1->y_0 / n_0;
    (&r_2)->z_0 = q_1->z_0 / n_0;
    return r_2;
}

float3 to_zup_0(float3 v_0)
{
    return float3(v_0.x, - v_0.z, v_0.y);
}

float3 rotate_0(const Quat_0 thread* q_2, float3 v_1)
{
    float3 qv_0 = float3(q_2->x_0, q_2->y_0, q_2->z_0);
    float3 t_1 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_1 * float3(q_2->w_0)  + cross(qv_0, t_1);
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

struct Strike_0
{
    Island_0 isl_1;
    uint n_1;
    float3 origin_0;
    Quat_0 q_3;
    float3 hit_0;
    float3 d_0;
    float speed_1;
    float face_0;
    float m_imp_0;
    float e_imp_0;
    uint entry_0;
    uint entry_k_0;
    float entry_depth_0;
    float3 from_0;
    float3 rel_0;
    float radius_0;
};

bool find_strike_0(uint i_1, uint t_2, uint max_chunks_0, Strike_0 thread* st_1, KernelContext_0 thread* kernelContext_2)
{
    uint ax_0;
    float r_3;
    float t0_0;
    Island_natural_0 device* _S43 = kernelContext_2->islands_0+i_1;
    Island_natural_0 _S44 = *_S43;
    uint4 _S45 = uint4((*_S43).info_0) ;
    float4 _S46 = float4((*_S43).com_0) ;
    float4 _S47 = float4((*_S43).inertia0_0) ;
    float4 _S48 = float4((*_S43).inertia1_0) ;
    float4 _S49 = float4((*_S43).inertia2_0) ;
    float4 _S50 = float4((*_S43).inv0_0) ;
    float4 _S51 = float4((*_S43).inv1_0) ;
    float4 _S52 = float4((*_S43).inv2_0) ;
    float4 _S53 = float4((*_S43).wcom_0) ;
    float4 _S54 = float4((*_S43).winv0_0) ;
    float4 _S55 = float4((*_S43).winv1_0) ;
    float4 _S56 = float4((*_S43).winv2_0) ;
    float4 _S57 = float4((*_S43).rotation_0) ;
    float4 _S58 = float4((*_S43).position_0) ;
    float4 _S59 = float4((*_S43).position_err_0) ;
    float4 _S60 = float4((*_S43).velocity_0) ;
    float4 _S61 = float4((*_S43).velocity_err_0) ;
    float4 _S62 = float4((*_S43).angular_velocity_0) ;
    uint4 _S63 = uint4((*_S43).probes_0) ;
    float4 _S64 = float4((*_S43).energy_0) ;
    float4 _S65 = float4((*_S43).rotation_err_0) ;
    float4 _S66 = float4((*_S43).momentum_0) ;
    float4 _S67 = float4((*_S43).momentum_err_0) ;
    (&st_1->isl_1)->range_0 = uint4((*_S43).range_0) ;
    (&st_1->isl_1)->info_0 = _S45;
    (&st_1->isl_1)->com_0 = _S46;
    (&st_1->isl_1)->inertia0_0 = _S47;
    (&st_1->isl_1)->inertia1_0 = _S48;
    (&st_1->isl_1)->inertia2_0 = _S49;
    (&st_1->isl_1)->inv0_0 = _S50;
    (&st_1->isl_1)->inv1_0 = _S51;
    (&st_1->isl_1)->inv2_0 = _S52;
    (&st_1->isl_1)->wcom_0 = _S53;
    (&st_1->isl_1)->winv0_0 = _S54;
    (&st_1->isl_1)->winv1_0 = _S55;
    (&st_1->isl_1)->winv2_0 = _S56;
    (&st_1->isl_1)->rotation_0 = _S57;
    (&st_1->isl_1)->position_0 = _S58;
    (&st_1->isl_1)->position_err_0 = _S59;
    (&st_1->isl_1)->velocity_0 = _S60;
    (&st_1->isl_1)->velocity_err_0 = _S61;
    (&st_1->isl_1)->angular_velocity_0 = _S62;
    (&st_1->isl_1)->done_0 = _S44.done_0;
    (&st_1->isl_1)->coverage_0 = _S44.coverage_0;
    (&st_1->isl_1)->wide_groups_0 = _S44.wide_groups_0;
    (&st_1->isl_1)->done_pad_0 = _S44.done_pad_0;
    (&st_1->isl_1)->probes_0 = _S63;
    (&st_1->isl_1)->energy_0 = _S64;
    (&st_1->isl_1)->rotation_err_0 = _S65;
    (&st_1->isl_1)->momentum_0 = _S66;
    (&st_1->isl_1)->momentum_err_0 = _S67;
    st_1->n_1 = 0U;
    float3 _S68 = float3(0.0f) ;
    st_1->origin_0 = _S68;
    st_1->q_3 = quat_of_0(float4(0.0f, 0.0f, 0.0f, 1.0f));
    st_1->hit_0 = _S68;
    st_1->d_0 = _S68;
    st_1->speed_1 = 0.0f;
    st_1->face_0 = 0.0f;
    st_1->m_imp_0 = 0.0f;
    st_1->e_imp_0 = 0.0f;
    st_1->entry_0 = 4294967295U;
    st_1->entry_k_0 = 4294967295U;
    st_1->entry_depth_0 = 0.0f;
    st_1->from_0 = _S68;
    st_1->rel_0 = _S68;
    st_1->radius_0 = 0.0f;
    Island_natural_0 device* _S69 = kernelContext_2->islands_0+i_1;
    Island_natural_0 isl_2 = *_S69;
    uint4 _S70 = uint4((*_S69).range_0) ;
    uint4 _S71 = uint4((*_S69).info_0) ;
    float4 _S72 = float4((*_S69).com_0) ;
    float4 _S73 = float4((*_S69).inertia0_0) ;
    float4 _S74 = float4((*_S69).inertia1_0) ;
    float4 _S75 = float4((*_S69).inertia2_0) ;
    float4 _S76 = float4((*_S69).inv0_0) ;
    float4 _S77 = float4((*_S69).inv1_0) ;
    float4 _S78 = float4((*_S69).inv2_0) ;
    float4 _S79 = float4((*_S69).wcom_0) ;
    float4 _S80 = float4((*_S69).winv0_0) ;
    float4 _S81 = float4((*_S69).winv1_0) ;
    float4 _S82 = float4((*_S69).winv2_0) ;
    float4 _S83 = float4((*_S69).rotation_0) ;
    float4 _S84 = float4((*_S69).position_0) ;
    float4 _S85 = float4((*_S69).position_err_0) ;
    float4 _S86 = float4((*_S69).velocity_0) ;
    float4 _S87 = float4((*_S69).velocity_err_0) ;
    float4 _S88 = float4((*_S69).angular_velocity_0) ;
    uint4 _S89 = uint4((*_S69).probes_0) ;
    float4 _S90 = float4((*_S69).energy_0) ;
    float4 _S91 = float4((*_S69).rotation_err_0) ;
    float4 _S92 = float4((*_S69).momentum_0) ;
    float4 _S93 = float4((*_S69).momentum_err_0) ;
    uint _S94 = _S70.x;
    uint n_2 = group_uniform_0(_S70.y - _S94, t_2);
    bool _S95;
    if((_S70.z) >= (_S70.w))
    {
        _S95 = true;
    }
    else
    {
        _S95 = n_2 > max_chunks_0;
    }
    if(_S95)
    {
        _S95 = true;
    }
    else
    {
        _S95 = (kernelContext_2->params_0->impactor_count_0) == 0U;
    }
    uint c_0;
    if(_S95)
    {
        c_0 = 1U;
    }
    else
    {
        c_0 = 0U;
    }
    if((group_uniform_0(c_0, t_2)) != 0U)
    {
        return false;
    }
    uint _S96 = 3U * i_1;
    uint body_0 = (as_type<uint>(((float4(*(kernelContext_2->ctab_0+(kernelContext_2->params_0->engine_base_0 + _S96))) ).x)));
    Quat_0 _S97 = quat_of_0(_S83);
    float3 _S98 = _S84.xyz;
    float3 _S99 = _S86.xyz;
    float3 origin_1;
    float3 v_2;
    Quat_0 q_4;
    if(body_0 != 4294967295U)
    {
        Quat_0 p_inv_0 = quat_of_0(float4(*(kernelContext_2->ctab_0+(kernelContext_2->params_0->engine_base_0 + _S96 + 1U))) );
        float3 com_1 = (float4(*(kernelContext_2->ctab_0+(kernelContext_2->params_0->engine_base_0 + _S96 + 2U))) ).xyz;
        EngineBody_natural_0 b_0 = kernelContext_2->bodies_0[body_0];
        thread Quat_0 a_inv_0;
        (&a_inv_0)->w_0 = 0.70710676908493042f;
        (&a_inv_0)->x_0 = 0.70710676908493042f;
        (&a_inv_0)->y_0 = 0.0f;
        (&a_inv_0)->z_0 = 0.0f;
        Quat_0 _S100 = quat_of_0(float4(b_0.rot_0) );
        thread Quat_0 _S101 = a_inv_0;
        thread Quat_0 _S102 = _S100;
        Quat_0 _S103 = quat_mul_0(&_S101, &_S102);
        thread Quat_0 _S104 = _S103;
        thread Quat_0 _S105 = p_inv_0;
        Quat_0 _S106 = quat_mul_0(&_S104, &_S105);
        thread Quat_0 _S107 = _S106;
        Quat_0 _S108 = normalized_0(&_S107);
        float3 _S109 = to_zup_0((float4(b_0.pos_inv_mass_0) ).xyz);
        thread Quat_0 _S110 = _S108;
        float3 _S111 = rotate_0(&_S110, com_1);
        float3 _S112 = to_zup_0((float4(b_0.vel_sleep_0) ).xyz);
        origin_1 = _S109 - _S111;
        q_4 = _S108;
        v_2 = _S112;
    }
    else
    {
        origin_1 = _S98;
        q_4 = _S97;
        v_2 = _S99;
    }
    float inf_0 = float_bits_0(2139095040U);
    float3 _S113 = float3(- inf_0) ;
    float3 lo_0 = float3(inf_0) ;
    float3 hi_0 = _S113;
    c_0 = t_2;
    for(;;)
    {
        if(c_0 < n_2)
        {
        }
        else
        {
            break;
        }
        uint k_0 = _S94 + c_0;
        float3 _S114 = (float4((kernelContext_2->chunks_0+k_0)->center_0) ).xyz + (float4(*(kernelContext_2->state_0+4U * k_0)) ).xyz;
        thread Quat_0 _S115 = q_4;
        float3 _S116 = rotate_0(&_S115, _S114);
        float3 p_0 = origin_1 + _S116;
        float3 _S117 = min(lo_0, p_0 - float3((float4((kernelContext_2->chunks_0+k_0)->half_0) ).w) );
        float3 _S118 = max(hi_0, p_0 + float3((float4((kernelContext_2->chunks_0+k_0)->half_0) ).w) );
        uint c_1 = c_0 + 256U;
        lo_0 = _S117;
        hi_0 = _S118;
        c_0 = c_1;
    }
    (*kernelContext_2->g_lo_0)[t_2] = lo_0;
    (*kernelContext_2->g_hi_0)[t_2] = hi_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = 128U;
    for(;;)
    {
        if(s_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_2 < s_0)
        {
            uint _S119 = t_2 + s_0;
            (*kernelContext_2->g_lo_0)[t_2] = min((*kernelContext_2->g_lo_0)[t_2], (*kernelContext_2->g_lo_0)[_S119]);
            (*kernelContext_2->g_hi_0)[t_2] = max((*kernelContext_2->g_hi_0)[t_2], (*kernelContext_2->g_hi_0)[_S119]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    float3 _S120 = (*kernelContext_2->g_lo_0)[int(0)];
    float3 _S121 = (*kernelContext_2->g_hi_0)[int(0)];
    uint _S122 = (as_type<uint>(((float4(*(kernelContext_2->fast_0+int(0))) ).x)));
    uint hit_k_0 = 4294967295U;
    float3 hit_1 = _S68;
    float3 d_1 = _S68;
    float speed_2 = 0.0f;
    float face_1 = 0.0f;
    float m_imp_1 = 0.0f;
    float e_imp_1 = 0.0f;
    float3 hit_c_0 = _S68;
    float3 hit_rel_0 = _S68;
    float p_radius_0 = 0.0f;
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < (kernelContext_2->params_0->impactor_count_0 + _S122))
        {
            _S95 = hit_k_0 == 4294967295U;
        }
        else
        {
            _S95 = false;
        }
        if(_S95)
        {
        }
        else
        {
            break;
        }
        float3 rel_1;
        float3 c_2;
        float4 info_2;
        if(k_1 < (kernelContext_2->params_0->impactor_count_0))
        {
            uint _S123 = 2U * k_1;
            float4 _S124 = float4(*(kernelContext_2->impactors_0+_S123)) ;
            EngineBody_natural_0 b_1 = kernelContext_2->bodies_0[(as_type<uint>((_S124.x)))];
            float3 _S125 = to_zup_0((float4(b_1.pos_inv_mass_0) ).xyz);
            float _S126 = _S124.y;
            float4 _S127 = float4(_S124.z, _S124.w, (float4(*(kernelContext_2->impactors_0+(_S123 + 1U))) ).x, 0.0f);
            rel_1 = to_zup_0((float4(b_1.vel_sleep_0) ).xyz) - v_2;
            r_3 = _S126;
            c_2 = _S125;
            info_2 = _S127;
        }
        else
        {
            uint f_0 = 1U + 3U * (k_1 - kernelContext_2->params_0->impactor_count_0);
            uint _S128 = f_0 + 1U;
            if((as_type<uint>(((float4(*(kernelContext_2->fast_0+_S128)) ).w))) == i_1)
            {
                k_1 = k_1 + 1U;
                continue;
            }
            float3 _S129 = (float4(*(kernelContext_2->fast_0+f_0)) ).xyz;
            float _S130 = (float4(*(kernelContext_2->fast_0+f_0)) ).w;
            float4 _S131 = float4(*(kernelContext_2->fast_0+(f_0 + 2U))) ;
            rel_1 = (float4(*(kernelContext_2->fast_0+_S128)) ).xyz - v_2;
            r_3 = _S130;
            c_2 = _S129;
            info_2 = _S131;
        }
        float sp_0 = length(rel_1);
        if(!(sp_0 > (kernelContext_2->params_0->speed_0)))
        {
            k_1 = k_1 + 1U;
            continue;
        }
        float3 _S132 = float3(r_3) ;
        float3 _S133 = _S120 - _S132;
        float3 _S134 = _S121 + _S132;
        float _S135 = kernelContext_2->params_0->dt_0;
        float t0_1 = 0.0f;
        float t1_1 = _S135;
        ax_0 = 0U;
        for(;;)
        {
            if(ax_0 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 _S136 = rel_1;
            uint _S137 = ax_0;
            if((abs(rel_1[ax_0])) < 9.999999960041972e-13f)
            {
                float3 _S138 = c_2;
                uint _S139 = ax_0;
                bool _S140;
                if((c_2[ax_0]) < (_S133[ax_0]))
                {
                    _S140 = true;
                }
                else
                {
                    _S140 = (_S138[_S139]) > (_S134[ax_0]);
                }
                if(_S140)
                {
                    t0_0 = inf_0;
                }
                else
                {
                    t0_0 = t0_1;
                }
                t0_1 = t0_0;
                ax_0 = ax_0 + 1U;
                continue;
            }
            float ta_0 = (_S133[ax_0] - c_2[ax_0]) / _S136[_S137];
            float tb_0 = (_S134[ax_0] - c_2[ax_0]) / _S136[_S137];
            float _S141 = min(t1_1, max(ta_0, tb_0));
            t0_1 = max(t0_1, min(ta_0, tb_0));
            t1_1 = _S141;
            ax_0 = ax_0 + 1U;
        }
        uint hit_k_1;
        float3 hit_2;
        float3 d_2;
        float3 hit_c_1;
        float3 hit_rel_1;
        float face_2;
        float m_imp_2;
        float e_imp_2;
        float p_radius_1;
        if(t0_1 <= t1_1)
        {
            float3 _S142 = c_2 + rel_1 * float3(t0_1) ;
            float3 _S143 = rel_1 / float3(sp_0) ;
            float _S144 = info_2.x;
            float _S145 = info_2.y;
            float _S146 = info_2.z;
            hit_k_1 = k_1;
            hit_2 = _S142;
            d_2 = _S143;
            t0_0 = sp_0;
            face_2 = _S144;
            m_imp_2 = _S145;
            e_imp_2 = _S146;
            hit_c_1 = c_2;
            hit_rel_1 = rel_1;
            p_radius_1 = r_3;
        }
        else
        {
            hit_k_1 = hit_k_0;
            hit_2 = hit_1;
            d_2 = d_1;
            t0_0 = speed_2;
            face_2 = face_1;
            m_imp_2 = m_imp_1;
            e_imp_2 = e_imp_1;
            hit_c_1 = hit_c_0;
            hit_rel_1 = hit_rel_0;
            p_radius_1 = p_radius_0;
        }
        hit_k_0 = hit_k_1;
        hit_1 = hit_2;
        d_1 = d_2;
        speed_2 = t0_0;
        face_1 = face_2;
        m_imp_1 = m_imp_2;
        e_imp_1 = e_imp_2;
        hit_c_0 = hit_c_1;
        hit_rel_0 = hit_rel_1;
        p_radius_0 = p_radius_1;
        k_1 = k_1 + 1U;
    }
    if(hit_k_0 == 4294967295U)
    {
        c_0 = 1U;
    }
    else
    {
        c_0 = 0U;
    }
    if((group_uniform_0(c_0, t_2)) != 0U)
    {
        return false;
    }
    r_3 = inf_0;
    ax_0 = 4294967295U;
    c_0 = t_2;
    for(;;)
    {
        if(c_0 < n_2)
        {
        }
        else
        {
            break;
        }
        uint k_2 = _S94 + c_0;
        float3 _S147 = (float4((kernelContext_2->chunks_0+k_2)->center_0) ).xyz + (float4(*(kernelContext_2->state_0+4U * k_2)) ).xyz;
        thread Quat_0 _S148 = q_4;
        float3 _S149 = rotate_0(&_S148, _S147);
        float3 r_4 = origin_1 + _S149 - hit_1;
        float along_0 = dot(r_4, d_1);
        float dist_0 = length(r_4 - d_1 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < r_3)
        {
            r_3 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + 256U;
    }
    (*kernelContext_2->g_a_0)[t_2] = r_3;
    (*kernelContext_2->g_idx_0)[t_2] = ax_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s_0 = 128U;
    for(;;)
    {
        if(s_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_2 < s_0)
        {
            uint _S150 = t_2 + s_0;
            if(((*kernelContext_2->g_a_0)[_S150]) < (*kernelContext_2->g_a_0)[t_2])
            {
                _S95 = true;
            }
            else
            {
                if(((*kernelContext_2->g_a_0)[_S150]) == (*kernelContext_2->g_a_0)[t_2])
                {
                    _S95 = ((*kernelContext_2->g_idx_0)[_S150]) < (*kernelContext_2->g_idx_0)[t_2];
                }
                else
                {
                    _S95 = false;
                }
            }
        }
        else
        {
            _S95 = false;
        }
        if(_S95)
        {
            uint _S151 = t_2 + s_0;
            (*kernelContext_2->g_a_0)[t_2] = (*kernelContext_2->g_a_0)[_S151];
            (*kernelContext_2->g_idx_0)[t_2] = (*kernelContext_2->g_idx_0)[_S151];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    uint entry_1 = (*kernelContext_2->g_idx_0)[int(0)];
    uint entry_k_1 = _S94 + (*kernelContext_2->g_idx_0)[int(0)];
    float3 _S152 = (float4((kernelContext_2->chunks_0+entry_k_1)->center_0) ).xyz + (float4(*(kernelContext_2->state_0+4U * entry_k_1)) ).xyz;
    thread Quat_0 _S153 = q_4;
    float3 _S154 = rotate_0(&_S153, _S152);
    float entry_depth_1 = dot(origin_1 + _S154 - hit_1, d_1);
    (&st_1->isl_1)->range_0 = _S70;
    (&st_1->isl_1)->info_0 = _S71;
    (&st_1->isl_1)->com_0 = _S72;
    (&st_1->isl_1)->inertia0_0 = _S73;
    (&st_1->isl_1)->inertia1_0 = _S74;
    (&st_1->isl_1)->inertia2_0 = _S75;
    (&st_1->isl_1)->inv0_0 = _S76;
    (&st_1->isl_1)->inv1_0 = _S77;
    (&st_1->isl_1)->inv2_0 = _S78;
    (&st_1->isl_1)->wcom_0 = _S79;
    (&st_1->isl_1)->winv0_0 = _S80;
    (&st_1->isl_1)->winv1_0 = _S81;
    (&st_1->isl_1)->winv2_0 = _S82;
    (&st_1->isl_1)->rotation_0 = _S83;
    (&st_1->isl_1)->position_0 = _S84;
    (&st_1->isl_1)->position_err_0 = _S85;
    (&st_1->isl_1)->velocity_0 = _S86;
    (&st_1->isl_1)->velocity_err_0 = _S87;
    (&st_1->isl_1)->angular_velocity_0 = _S88;
    (&st_1->isl_1)->done_0 = isl_2.done_0;
    (&st_1->isl_1)->coverage_0 = isl_2.coverage_0;
    (&st_1->isl_1)->wide_groups_0 = isl_2.wide_groups_0;
    (&st_1->isl_1)->done_pad_0 = isl_2.done_pad_0;
    (&st_1->isl_1)->probes_0 = _S89;
    (&st_1->isl_1)->energy_0 = _S90;
    (&st_1->isl_1)->rotation_err_0 = _S91;
    (&st_1->isl_1)->momentum_0 = _S92;
    (&st_1->isl_1)->momentum_err_0 = _S93;
    st_1->n_1 = n_2;
    st_1->origin_0 = origin_1;
    st_1->q_3 = q_4;
    st_1->hit_0 = hit_1;
    st_1->d_0 = d_1;
    st_1->speed_1 = speed_2;
    st_1->face_0 = face_1;
    st_1->m_imp_0 = m_imp_1;
    st_1->e_imp_0 = e_imp_1;
    st_1->entry_0 = entry_1;
    st_1->entry_k_0 = entry_k_1;
    st_1->entry_depth_0 = entry_depth_1;
    st_1->from_0 = hit_c_0;
    st_1->rel_0 = hit_rel_0;
    st_1->radius_0 = p_radius_0;
    return true;
}

[[kernel]] void threshold_predict(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], ThresholdParams_0 constant* params_2 [[buffer(0)]], uint device* status_2 [[buffer(6)]], Island_natural_0 device* islands_2 [[buffer(1)]], BondStatic_natural_0 device* bond_static_2 [[buffer(2)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(3)]], MaterialTable_0 constant* materials_2 [[buffer(5)]], packed_float4 device* ctab_2 [[buffer(4)]], packed_float4 device* fast_2 [[buffer(11)]], EngineBody_natural_0 device* bodies_2 [[buffer(9)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(7)]], packed_float4 device* state_2 [[buffer(8)]], packed_float4 device* impactors_2 [[buffer(10)]])
{
    uint ax_1;
    float t_in_0;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_2;
    (&kernelContext_3)->status_0 = status_2;
    (&kernelContext_3)->islands_0 = islands_2;
    (&kernelContext_3)->bond_static_0 = bond_static_2;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_3)->materials_0 = materials_2;
    (&kernelContext_3)->ctab_0 = ctab_2;
    (&kernelContext_3)->fast_0 = fast_2;
    (&kernelContext_3)->bodies_0 = bodies_2;
    (&kernelContext_3)->chunks_0 = chunks_2;
    (&kernelContext_3)->state_0 = state_2;
    (&kernelContext_3)->impactors_0 = impactors_2;
    threadgroup array<uint, int(256)> g_broke_2;
    (&kernelContext_3)->g_broke_0 = &g_broke_2;
    threadgroup array<float3, int(256)> g_lo_2;
    (&kernelContext_3)->g_lo_0 = &g_lo_2;
    threadgroup array<float3, int(256)> g_hi_2;
    (&kernelContext_3)->g_hi_0 = &g_hi_2;
    threadgroup array<float, int(256)> g_a_2;
    (&kernelContext_3)->g_a_0 = &g_a_2;
    threadgroup array<uint, int(256)> g_idx_2;
    (&kernelContext_3)->g_idx_0 = &g_idx_2;
    uint i_2 = group_1.x;
    uint t_3 = thread_2.x;
    if(i_2 >= (params_2->island_count_0))
    {
        return;
    }
    if((group_uniform_0((as_type<uint>(((float4(*((&kernelContext_3)->fast_0+int(0))) ).y))), t_3)) == 0U)
    {
        if(t_3 == 0U)
        {
            uint device* _S155 = (&kernelContext_3)->status_0+(16U + i_2);
            *_S155 = (*_S155) & 4294967287U;
        }
        return;
    }
    thread Strike_0 st_2;
    bool _S156 = find_strike_0(i_2, t_3, 4294967295U, &st_2, &kernelContext_3);
    uint _S157 = group_uniform_0((&st_2)->n_1, t_3);
    uint c_3;
    if(_S156)
    {
        c_3 = 1U;
    }
    else
    {
        c_3 = 0U;
    }
    if((group_uniform_0(c_3, t_3)) == 0U)
    {
        if(t_3 == 0U)
        {
            uint device* _S158 = (&kernelContext_3)->status_0+(16U + i_2);
            *_S158 = (*_S158) & 4294967287U;
        }
        return;
    }
    float inf_1 = float_bits_0(2139095040U);
    float first_0 = inf_1;
    c_3 = t_3;
    for(;;)
    {
        if(c_3 < _S157)
        {
        }
        else
        {
            break;
        }
        uint k_3 = (&(&st_2)->isl_1)->range_0.x + c_3;
        float3 _S159 = (float4(((&kernelContext_3)->chunks_0+k_3)->center_0) ).xyz + (float4(*((&kernelContext_3)->state_0+4U * k_3)) ).xyz;
        thread Quat_0 _S160 = (&st_2)->q_3;
        float3 _S161 = rotate_0(&_S160, _S159);
        float3 _S162 = (&st_2)->origin_0 + _S161;
        float _S163 = (&kernelContext_3)->params_0->dt_0;
        t_in_0 = 0.0f;
        float t_out_0 = _S163;
        ax_1 = 0U;
        for(;;)
        {
            if(ax_1 < 3U)
            {
            }
            else
            {
                break;
            }
            float3 _S164 = float3((float4(((&kernelContext_3)->chunks_0+k_3)->crot0_0) )[ax_1], (float4(((&kernelContext_3)->chunks_0+k_3)->crot1_0) )[ax_1], (float4(((&kernelContext_3)->chunks_0+k_3)->crot2_0) )[ax_1]);
            thread Quat_0 _S165 = (&st_2)->q_3;
            float3 _S166 = rotate_0(&_S165, _S164);
            float o_1 = dot((&st_2)->from_0 - _S162, _S166);
            float dv_0 = dot((&st_2)->rel_0, _S166);
            float ext_0 = (float4(((&kernelContext_3)->chunks_0+k_3)->half_0) )[ax_1] + (&st_2)->radius_0;
            if((abs(dv_0)) < 9.999999960041972e-13f)
            {
                float t_in_1;
                if((abs(o_1)) > ext_0)
                {
                    t_in_1 = inf_1;
                }
                else
                {
                    t_in_1 = t_in_0;
                }
                t_in_0 = t_in_1;
                ax_1 = ax_1 + 1U;
                continue;
            }
            float ta_1 = (- ext_0 - o_1) / dv_0;
            float tb_1 = (ext_0 - o_1) / dv_0;
            float _S167 = min(t_out_0, max(ta_1, tb_1));
            t_in_0 = max(t_in_0, min(ta_1, tb_1));
            t_out_0 = _S167;
            ax_1 = ax_1 + 1U;
        }
        if(t_in_0 <= t_out_0)
        {
            first_0 = min(first_0, t_in_0);
        }
        c_3 = c_3 + 256U;
    }
    (*(&kernelContext_3)->g_a_0)[t_3] = first_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ax_1 = 128U;
    for(;;)
    {
        if(ax_1 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_3 < ax_1)
        {
            (*(&kernelContext_3)->g_a_0)[t_3] = min((*(&kernelContext_3)->g_a_0)[t_3], (*(&kernelContext_3)->g_a_0)[t_3 + ax_1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ax_1 = ax_1 >> 1U;
    }
    float t_in_2 = (*(&kernelContext_3)->g_a_0)[int(0)];
    if(((*(&kernelContext_3)->g_a_0)[int(0)]) < inf_1)
    {
        c_3 = 0U;
    }
    else
    {
        c_3 = 1U;
    }
    if((group_uniform_0(c_3, t_3)) != 0U)
    {
        if(t_3 == 0U)
        {
            uint device* _S168 = (&kernelContext_3)->status_0+(16U + i_2);
            *_S168 = (*_S168) & 4294967287U;
        }
        return;
    }
    float3 _S169 = (&st_2)->from_0 + (&st_2)->rel_0 * float3(t_in_2)  + (&st_2)->d_0 * float3((&st_2)->radius_0) ;
    float m_isl_0 = (&(&st_2)->isl_1)->com_0.w;
    if((((&(&st_2)->isl_1)->info_0.x) & 1U) != 0U)
    {
        t_in_0 = (&st_2)->m_imp_0;
    }
    else
    {
        t_in_0 = (&st_2)->m_imp_0 * m_isl_0 / ((&st_2)->m_imp_0 + m_isl_0);
    }
    float modulus_0 = 1.0f / (1.0f / (&st_2)->e_imp_0 + 1.0f / (float4(((&kernelContext_3)->chunks_0+(&st_2)->entry_k_0)->cmat_0) ).x);
    float _S170 = t_in_0 * (&st_2)->speed_1 / (2.86999988555908203f * pow(t_in_0 * t_in_0 / (max((&st_2)->face_0, 0.00100000004749745f) * modulus_0 * modulus_0 * max((&st_2)->speed_1, 0.00100000004749745f)), 0.20000000298023224f));
    uint _S171 = (&(&st_2)->isl_1)->range_0.z + t_3;
    uint broke_2 = 0U;
    uint bi_2 = _S171;
    for(;;)
    {
        if(bi_2 < ((&(&st_2)->isl_1)->range_0.w))
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_2 = ((&kernelContext_3)->bond_static_0+bi_2)->law_0;
        BondDyn_natural_0 device* _S172 = (&kernelContext_3)->bond_dyn_0+bi_2;
        float4 _S173 = float4((*_S172).force_lin_0) ;
        float4 _S174 = float4((*_S172).force_ang_0) ;
        float4 _S175 = float4((*_S172).sums_0) ;
        float4 _S176 = float4((*_S172).comps_0) ;
        uint4 _S177 = uint4((*_S172).events_0) ;
        thread BondDyn_0 dyn_1;
        (&dyn_1)->js_0 = (*_S172).js_0;
        (&dyn_1)->force_lin_0 = _S173;
        (&dyn_1)->force_ang_0 = _S174;
        (&dyn_1)->sums_0 = _S175;
        (&dyn_1)->comps_0 = _S176;
        (&dyn_1)->events_0 = _S177;
        bool _S178 = ((float4(((&kernelContext_3)->bond_static_0+bi_2)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S179 = (&dyn_1)->js_0;
        bool _S180 = connected_0(&_S179, _S178);
        if(!_S180)
        {
            bi_2 = bi_2 + 256U;
            continue;
        }
        uint4 _S181 = uint4(law_2.ids_0) ;
        uint _S182 = _S181.y;
        float3 _S183 = (float4(((&kernelContext_3)->chunks_0+_S182)->center_0) ).xyz + (float4(*((&kernelContext_3)->state_0+4U * _S182)) ).xyz;
        thread Quat_0 _S184 = (&st_2)->q_3;
        float3 _S185 = rotate_0(&_S184, _S183);
        float3 pa_0 = (&st_2)->origin_0 + _S185;
        uint _S186 = _S181.z;
        float3 _S187 = (float4(((&kernelContext_3)->chunks_0+_S186)->center_0) ).xyz + (float4(*((&kernelContext_3)->state_0+4U * _S186)) ).xyz;
        thread Quat_0 _S188 = (&st_2)->q_3;
        float3 _S189 = rotate_0(&_S188, _S187);
        float dist_1 = length(float3(0.5f)  * (pa_0 + ((&st_2)->origin_0 + _S189)) - _S169);
        float _S190 = (float4(law_2.geom0_0) ).x;
        uint broke_3;
        if((_S170 * min(1.0f, _S190 / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > ((&(&kernelContext_3)->materials_0->m_0[_S181.x])->strength_0.x * _S190 * (float4(law_2.geom1_0) ).w))
        {
            (&(&dyn_1)->js_0)->damage_0 = 1.0f;
            (&(&dyn_1)->js_0)->rebar_broken_0 = 1.0f;
            (&dyn_1)->events_0.z = max((&dyn_1)->events_0.z, 1U);
            BondDyn_natural_0 device* _S191 = (&kernelContext_3)->bond_dyn_0+bi_2;
            _S191->js_0 = dyn_1.js_0;
            _S191->force_lin_0 = packed_float4(dyn_1.force_lin_0) ;
            _S191->force_ang_0 = packed_float4(dyn_1.force_ang_0) ;
            _S191->sums_0 = packed_float4(dyn_1.sums_0) ;
            _S191->comps_0 = packed_float4(dyn_1.comps_0) ;
            _S191->events_0 = packed_uint4(dyn_1.events_0) ;
            broke_3 = 1U;
        }
        else
        {
            broke_3 = broke_2;
        }
        broke_2 = broke_3;
        bi_2 = bi_2 + 256U;
    }
    (*(&kernelContext_3)->g_idx_0)[t_3] = broke_2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ax_1 = 128U;
    for(;;)
    {
        if(ax_1 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_3 < ax_1)
        {
            (*(&kernelContext_3)->g_idx_0)[t_3] = ((*(&kernelContext_3)->g_idx_0)[t_3]) | (*(&kernelContext_3)->g_idx_0)[t_3 + ax_1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ax_1 = ax_1 >> 1U;
    }
    if(t_3 != 0U)
    {
        return;
    }
    if(((*(&kernelContext_3)->g_idx_0)[int(0)]) != 0U)
    {
        uint device* _S192 = (&kernelContext_3)->status_0+(16U + i_2);
        *_S192 = (*_S192) | 8U;
        *((&kernelContext_3)->status_0+1U) = 1U;
    }
    else
    {
        uint device* _S193 = (&kernelContext_3)->status_0+(16U + i_2);
        *_S193 = (*_S193) & 4294967287U;
    }
    return;
}

