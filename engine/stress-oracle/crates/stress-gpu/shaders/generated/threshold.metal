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
    uint pad0_0;
    uint pad1_0;
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

struct KernelContext_0
{
    ThresholdParams_0 constant* params_0;
    uint device* status_0;
    Island_natural_0 device* islands_0;
    BondStatic_natural_0 device* bond_static_0;
    BondDyn_natural_0 device* bond_dyn_0;
    MaterialTable_0 constant* materials_0;
    packed_float4 device* ctab_0;
    array<uint, int(256)> threadgroup* g_broke_0;
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

[[kernel]] void threshold_breaks(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], ThresholdParams_0 constant* params_1 [[buffer(0)]], uint device* status_1 [[buffer(6)]], Island_natural_0 device* islands_1 [[buffer(1)]], BondStatic_natural_0 device* bond_static_1 [[buffer(2)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(3)]], MaterialTable_0 constant* materials_1 [[buffer(5)]], packed_float4 device* ctab_1 [[buffer(4)]])
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
    threadgroup array<uint, int(256)> g_broke_1;
    (&kernelContext_1)->g_broke_0 = &g_broke_1;
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
            bi_0 = bi_0 + 256U;
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
        broke_1 = broke_0;
        bi_0 = bi_0 + 256U;
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

