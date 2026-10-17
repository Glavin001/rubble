#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct CascadeParams_0
{
    uint island_count_0;
    uint result_base_0;
    uint pass_index_0;
    uint pad_0;
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

struct KernelContext_0
{
    CascadeParams_0 constant* params_0;
    uint device* status_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* scratch_0;
    packed_float4 device* state_0;
    packed_float4 device* backup_0;
    array<uint, int(256)> threadgroup* g_damaged_0;
    array<uint, int(256)> threadgroup* g_disconnected_0;
    array<uint, int(256)> threadgroup* g_unconverged_0;
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

[[kernel]] void cascade_begin(uint3 id_0 [[thread_position_in_grid]], CascadeParams_0 constant* params_1 [[buffer(0)]], uint device* status_1 [[buffer(5)]], Island_natural_0 device* islands_1 [[buffer(1)]], packed_float4 device* scratch_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* backup_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->status_0 = status_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->scratch_0 = scratch_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->backup_0 = backup_1;
    threadgroup array<uint, int(256)> g_damaged_1;
    (&kernelContext_0)->g_damaged_0 = &g_damaged_1;
    threadgroup array<uint, int(256)> g_disconnected_1;
    (&kernelContext_0)->g_disconnected_0 = &g_disconnected_1;
    threadgroup array<uint, int(256)> g_unconverged_1;
    (&kernelContext_0)->g_unconverged_0 = &g_unconverged_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_1->island_count_0))
    {
        return;
    }
    bool run_0 = (*((&kernelContext_0)->status_0+0U)) != 0U;
    Island_natural_0 device* _S1 = (&kernelContext_0)->islands_0+i_0;
    uint4 _S2 = uint4((*_S1).info_0) ;
    float4 _S3 = float4((*_S1).com_0) ;
    float4 _S4 = float4((*_S1).inertia0_0) ;
    float4 _S5 = float4((*_S1).inertia1_0) ;
    float4 _S6 = float4((*_S1).inertia2_0) ;
    float4 _S7 = float4((*_S1).inv0_0) ;
    float4 _S8 = float4((*_S1).inv1_0) ;
    float4 _S9 = float4((*_S1).inv2_0) ;
    float4 _S10 = float4((*_S1).wcom_0) ;
    float4 _S11 = float4((*_S1).winv0_0) ;
    float4 _S12 = float4((*_S1).winv1_0) ;
    float4 _S13 = float4((*_S1).winv2_0) ;
    float4 _S14 = float4((*_S1).rotation_0) ;
    float4 _S15 = float4((*_S1).position_0) ;
    float4 _S16 = float4((*_S1).position_err_0) ;
    float4 _S17 = float4((*_S1).velocity_0) ;
    float4 _S18 = float4((*_S1).velocity_err_0) ;
    float4 _S19 = float4((*_S1).angular_velocity_0) ;
    uint4 _S20 = uint4((*_S1).probes_0) ;
    float4 _S21 = float4((*_S1).energy_0) ;
    float4 _S22 = float4((*_S1).rotation_err_0) ;
    float4 _S23 = float4((*_S1).momentum_0) ;
    float4 _S24 = float4((*_S1).momentum_err_0) ;
    thread Island_0 isl_0;
    (&isl_0)->range_0 = uint4((*_S1).range_0) ;
    (&isl_0)->info_0 = _S2;
    (&isl_0)->com_0 = _S3;
    (&isl_0)->inertia0_0 = _S4;
    (&isl_0)->inertia1_0 = _S5;
    (&isl_0)->inertia2_0 = _S6;
    (&isl_0)->inv0_0 = _S7;
    (&isl_0)->inv1_0 = _S8;
    (&isl_0)->inv2_0 = _S9;
    (&isl_0)->wcom_0 = _S10;
    (&isl_0)->winv0_0 = _S11;
    (&isl_0)->winv1_0 = _S12;
    (&isl_0)->winv2_0 = _S13;
    (&isl_0)->rotation_0 = _S14;
    (&isl_0)->position_0 = _S15;
    (&isl_0)->position_err_0 = _S16;
    (&isl_0)->velocity_0 = _S17;
    (&isl_0)->velocity_err_0 = _S18;
    (&isl_0)->angular_velocity_0 = _S19;
    (&isl_0)->done_0 = (*_S1).done_0;
    (&isl_0)->coverage_0 = (*_S1).coverage_0;
    (&isl_0)->wide_groups_0 = (*_S1).wide_groups_0;
    (&isl_0)->done_pad_0 = (*_S1).done_pad_0;
    (&isl_0)->probes_0 = _S20;
    (&isl_0)->energy_0 = _S21;
    (&isl_0)->rotation_err_0 = _S22;
    (&isl_0)->momentum_0 = _S23;
    (&isl_0)->momentum_err_0 = _S24;
    bool solve_0;
    if(run_0)
    {
        solve_0 = ((&isl_0)->range_0.w) > ((&isl_0)->range_0.z);
    }
    else
    {
        solve_0 = false;
    }
    uint _S25 = ((&isl_0)->info_0.z) & 4294967287U;
    uint _S26;
    if(solve_0)
    {
        _S26 = 8U;
    }
    else
    {
        _S26 = 0U;
    }
    (&isl_0)->info_0.z = _S25 | _S26;
    Island_natural_0 device* _S27 = (&kernelContext_0)->islands_0+i_0;
    _S27->range_0 = packed_uint4(isl_0.range_0) ;
    _S27->info_0 = packed_uint4(isl_0.info_0) ;
    _S27->com_0 = packed_float4(isl_0.com_0) ;
    _S27->inertia0_0 = packed_float4(isl_0.inertia0_0) ;
    _S27->inertia1_0 = packed_float4(isl_0.inertia1_0) ;
    _S27->inertia2_0 = packed_float4(isl_0.inertia2_0) ;
    _S27->inv0_0 = packed_float4(isl_0.inv0_0) ;
    _S27->inv1_0 = packed_float4(isl_0.inv1_0) ;
    _S27->inv2_0 = packed_float4(isl_0.inv2_0) ;
    _S27->wcom_0 = packed_float4(isl_0.wcom_0) ;
    _S27->winv0_0 = packed_float4(isl_0.winv0_0) ;
    _S27->winv1_0 = packed_float4(isl_0.winv1_0) ;
    _S27->winv2_0 = packed_float4(isl_0.winv2_0) ;
    _S27->rotation_0 = packed_float4(isl_0.rotation_0) ;
    _S27->position_0 = packed_float4(isl_0.position_0) ;
    _S27->position_err_0 = packed_float4(isl_0.position_err_0) ;
    _S27->velocity_0 = packed_float4(isl_0.velocity_0) ;
    _S27->velocity_err_0 = packed_float4(isl_0.velocity_err_0) ;
    _S27->angular_velocity_0 = packed_float4(isl_0.angular_velocity_0) ;
    _S27->done_0 = isl_0.done_0;
    _S27->coverage_0 = isl_0.coverage_0;
    _S27->wide_groups_0 = isl_0.wide_groups_0;
    _S27->done_pad_0 = isl_0.done_pad_0;
    _S27->probes_0 = packed_uint4(isl_0.probes_0) ;
    _S27->energy_0 = packed_float4(isl_0.energy_0) ;
    _S27->rotation_err_0 = packed_float4(isl_0.rotation_err_0) ;
    _S27->momentum_0 = packed_float4(isl_0.momentum_0) ;
    _S27->momentum_err_0 = packed_float4(isl_0.momentum_err_0) ;
    if(run_0)
    {
        uint device* _S28 = (&kernelContext_0)->status_0+(16U + i_0);
        if(solve_0)
        {
            _S26 = 1U;
        }
        else
        {
            _S26 = 0U;
        }
        *_S28 = _S26;
    }
    return;
}

[[kernel]] void cascade_restore(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], CascadeParams_0 constant* params_2 [[buffer(0)]], uint device* status_2 [[buffer(5)]], Island_natural_0 device* islands_2 [[buffer(1)]], packed_float4 device* scratch_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], packed_float4 device* backup_2 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->status_0 = status_2;
    (&kernelContext_1)->islands_0 = islands_2;
    (&kernelContext_1)->scratch_0 = scratch_2;
    (&kernelContext_1)->state_0 = state_2;
    (&kernelContext_1)->backup_0 = backup_2;
    threadgroup array<uint, int(256)> g_damaged_2;
    (&kernelContext_1)->g_damaged_0 = &g_damaged_2;
    threadgroup array<uint, int(256)> g_disconnected_2;
    (&kernelContext_1)->g_disconnected_0 = &g_disconnected_2;
    threadgroup array<uint, int(256)> g_unconverged_2;
    (&kernelContext_1)->g_unconverged_0 = &g_unconverged_2;
    uint i_1 = group_0.x;
    if(i_1 >= (params_2->island_count_0))
    {
        return;
    }
    uint _S29 = 16U + i_1;
    uint s_0 = *((&kernelContext_1)->status_0+_S29);
    bool _S30;
    if((*((&kernelContext_1)->status_0+0U)) == 0U)
    {
        _S30 = true;
    }
    else
    {
        _S30 = (s_0 & 1U) == 0U;
    }
    if(_S30)
    {
        return;
    }
    if(((float4(*((&kernelContext_1)->scratch_0+((&kernelContext_1)->params_0->result_base_0 + i_1))) ).w) != 0.0f)
    {
        if((thread_0.x) == 0U)
        {
            *((&kernelContext_1)->status_0+_S29) = s_0 | 2U;
        }
        return;
    }
    uint4 _S31 = uint4(((&kernelContext_1)->islands_0+i_1)->range_0) ;
    uint k_0 = 4U * _S31.x + thread_0.x;
    for(;;)
    {
        if(k_0 < (4U * _S31.y))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_1)->state_0+k_0) = packed_float4(float4(*((&kernelContext_1)->backup_0+k_0)) ) ;
        k_0 = k_0 + 64U;
    }
    return;
}

[[kernel]] void cascade_settle(uint3 id_1 [[thread_position_in_grid]], CascadeParams_0 constant* params_3 [[buffer(0)]], uint device* status_3 [[buffer(5)]], Island_natural_0 device* islands_3 [[buffer(1)]], packed_float4 device* scratch_3 [[buffer(2)]], packed_float4 device* state_3 [[buffer(3)]], packed_float4 device* backup_3 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_3;
    (&kernelContext_2)->status_0 = status_3;
    (&kernelContext_2)->islands_0 = islands_3;
    (&kernelContext_2)->scratch_0 = scratch_3;
    (&kernelContext_2)->state_0 = state_3;
    (&kernelContext_2)->backup_0 = backup_3;
    threadgroup array<uint, int(256)> g_damaged_3;
    (&kernelContext_2)->g_damaged_0 = &g_damaged_3;
    threadgroup array<uint, int(256)> g_disconnected_3;
    (&kernelContext_2)->g_disconnected_0 = &g_disconnected_3;
    threadgroup array<uint, int(256)> g_unconverged_3;
    (&kernelContext_2)->g_unconverged_0 = &g_unconverged_3;
    uint i_2 = id_1.x;
    if(i_2 >= (params_3->island_count_0))
    {
        return;
    }
    uint s_1 = *((&kernelContext_2)->status_0+(16U + i_2));
    Island_natural_0 device* _S32 = (&kernelContext_2)->islands_0+i_2;
    uint4 _S33 = uint4((*_S32).info_0) ;
    float4 _S34 = float4((*_S32).com_0) ;
    float4 _S35 = float4((*_S32).inertia0_0) ;
    float4 _S36 = float4((*_S32).inertia1_0) ;
    float4 _S37 = float4((*_S32).inertia2_0) ;
    float4 _S38 = float4((*_S32).inv0_0) ;
    float4 _S39 = float4((*_S32).inv1_0) ;
    float4 _S40 = float4((*_S32).inv2_0) ;
    float4 _S41 = float4((*_S32).wcom_0) ;
    float4 _S42 = float4((*_S32).winv0_0) ;
    float4 _S43 = float4((*_S32).winv1_0) ;
    float4 _S44 = float4((*_S32).winv2_0) ;
    float4 _S45 = float4((*_S32).rotation_0) ;
    float4 _S46 = float4((*_S32).position_0) ;
    float4 _S47 = float4((*_S32).position_err_0) ;
    float4 _S48 = float4((*_S32).velocity_0) ;
    float4 _S49 = float4((*_S32).velocity_err_0) ;
    float4 _S50 = float4((*_S32).angular_velocity_0) ;
    uint4 _S51 = uint4((*_S32).probes_0) ;
    float4 _S52 = float4((*_S32).energy_0) ;
    float4 _S53 = float4((*_S32).rotation_err_0) ;
    float4 _S54 = float4((*_S32).momentum_0) ;
    float4 _S55 = float4((*_S32).momentum_err_0) ;
    thread Island_0 isl_1;
    (&isl_1)->range_0 = uint4((*_S32).range_0) ;
    (&isl_1)->info_0 = _S33;
    (&isl_1)->com_0 = _S34;
    (&isl_1)->inertia0_0 = _S35;
    (&isl_1)->inertia1_0 = _S36;
    (&isl_1)->inertia2_0 = _S37;
    (&isl_1)->inv0_0 = _S38;
    (&isl_1)->inv1_0 = _S39;
    (&isl_1)->inv2_0 = _S40;
    (&isl_1)->wcom_0 = _S41;
    (&isl_1)->winv0_0 = _S42;
    (&isl_1)->winv1_0 = _S43;
    (&isl_1)->winv2_0 = _S44;
    (&isl_1)->rotation_0 = _S45;
    (&isl_1)->position_0 = _S46;
    (&isl_1)->position_err_0 = _S47;
    (&isl_1)->velocity_0 = _S48;
    (&isl_1)->velocity_err_0 = _S49;
    (&isl_1)->angular_velocity_0 = _S50;
    (&isl_1)->done_0 = (*_S32).done_0;
    (&isl_1)->coverage_0 = (*_S32).coverage_0;
    (&isl_1)->wide_groups_0 = (*_S32).wide_groups_0;
    (&isl_1)->done_pad_0 = (*_S32).done_pad_0;
    (&isl_1)->probes_0 = _S51;
    (&isl_1)->energy_0 = _S52;
    (&isl_1)->rotation_err_0 = _S53;
    (&isl_1)->momentum_0 = _S54;
    (&isl_1)->momentum_err_0 = _S55;
    (&isl_1)->info_0.x = ((&isl_1)->info_0.x) & 4294967279U;
    (&isl_1)->info_0.z = ((&isl_1)->info_0.z) & 4294967247U;
    bool _S56;
    if((*((&kernelContext_2)->status_0+0U)) != 0U)
    {
        _S56 = (s_1 & 3U) == 3U;
    }
    else
    {
        _S56 = false;
    }
    if(_S56)
    {
        (&isl_1)->info_0.x = ((&isl_1)->info_0.x) | 16U;
    }
    Island_natural_0 device* _S57 = (&kernelContext_2)->islands_0+i_2;
    _S57->range_0 = packed_uint4(isl_1.range_0) ;
    _S57->info_0 = packed_uint4(isl_1.info_0) ;
    _S57->com_0 = packed_float4(isl_1.com_0) ;
    _S57->inertia0_0 = packed_float4(isl_1.inertia0_0) ;
    _S57->inertia1_0 = packed_float4(isl_1.inertia1_0) ;
    _S57->inertia2_0 = packed_float4(isl_1.inertia2_0) ;
    _S57->inv0_0 = packed_float4(isl_1.inv0_0) ;
    _S57->inv1_0 = packed_float4(isl_1.inv1_0) ;
    _S57->inv2_0 = packed_float4(isl_1.inv2_0) ;
    _S57->wcom_0 = packed_float4(isl_1.wcom_0) ;
    _S57->winv0_0 = packed_float4(isl_1.winv0_0) ;
    _S57->winv1_0 = packed_float4(isl_1.winv1_0) ;
    _S57->winv2_0 = packed_float4(isl_1.winv2_0) ;
    _S57->rotation_0 = packed_float4(isl_1.rotation_0) ;
    _S57->position_0 = packed_float4(isl_1.position_0) ;
    _S57->position_err_0 = packed_float4(isl_1.position_err_0) ;
    _S57->velocity_0 = packed_float4(isl_1.velocity_0) ;
    _S57->velocity_err_0 = packed_float4(isl_1.velocity_err_0) ;
    _S57->angular_velocity_0 = packed_float4(isl_1.angular_velocity_0) ;
    _S57->done_0 = isl_1.done_0;
    _S57->coverage_0 = isl_1.coverage_0;
    _S57->wide_groups_0 = isl_1.wide_groups_0;
    _S57->done_pad_0 = isl_1.done_pad_0;
    _S57->probes_0 = packed_uint4(isl_1.probes_0) ;
    _S57->energy_0 = packed_float4(isl_1.energy_0) ;
    _S57->rotation_err_0 = packed_float4(isl_1.rotation_err_0) ;
    _S57->momentum_0 = packed_float4(isl_1.momentum_0) ;
    _S57->momentum_err_0 = packed_float4(isl_1.momentum_err_0) ;
    return;
}

[[kernel]] void cascade_end(uint3 thread_1 [[thread_position_in_threadgroup]], CascadeParams_0 constant* params_4 [[buffer(0)]], uint device* status_4 [[buffer(5)]], Island_natural_0 device* islands_4 [[buffer(1)]], packed_float4 device* scratch_4 [[buffer(2)]], packed_float4 device* state_4 [[buffer(3)]], packed_float4 device* backup_4 [[buffer(4)]])
{
    uint damaged_0;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_4;
    (&kernelContext_3)->status_0 = status_4;
    (&kernelContext_3)->islands_0 = islands_4;
    (&kernelContext_3)->scratch_0 = scratch_4;
    (&kernelContext_3)->state_0 = state_4;
    (&kernelContext_3)->backup_0 = backup_4;
    threadgroup array<uint, int(256)> g_damaged_4;
    (&kernelContext_3)->g_damaged_0 = &g_damaged_4;
    threadgroup array<uint, int(256)> g_disconnected_4;
    (&kernelContext_3)->g_disconnected_0 = &g_disconnected_4;
    threadgroup array<uint, int(256)> g_unconverged_4;
    (&kernelContext_3)->g_unconverged_0 = &g_unconverged_4;
    uint tid_0 = thread_1.x;
    if((*(status_4+0U)) == 0U)
    {
        return;
    }
    uint damaged_1 = 0U;
    uint disconnected_0 = 0U;
    uint i_3 = tid_0;
    uint unconverged_0 = 0U;
    for(;;)
    {
        if(i_3 < ((&kernelContext_3)->params_0->island_count_0))
        {
        }
        else
        {
            break;
        }
        uint _S58 = 16U + i_3;
        uint device* _S59 = (&kernelContext_3)->status_0+_S58;
        uint s_2 = *_S59;
        if(((*_S59) & 1U) == 0U)
        {
            i_3 = i_3 + 256U;
            continue;
        }
        if((s_2 & 2U) == 0U)
        {
            unconverged_0 = unconverged_0 + 1U;
            i_3 = i_3 + 256U;
            continue;
        }
        uint z_0 = (uint4(((&kernelContext_3)->islands_0+i_3)->info_0) ).z;
        uint add_0;
        if((z_0 & 16U) != 0U)
        {
            damaged_0 = 1U;
            add_0 = 4U;
        }
        else
        {
            damaged_0 = damaged_1;
            add_0 = 0U;
        }
        uint disconnected_1;
        uint add_1;
        if((z_0 & 32U) != 0U)
        {
            uint add_2 = add_0 | 8U;
            disconnected_1 = 1U;
            add_1 = add_2;
        }
        else
        {
            disconnected_1 = disconnected_0;
            add_1 = add_0;
        }
        *((&kernelContext_3)->status_0+_S58) = s_2 | add_1;
        damaged_1 = damaged_0;
        disconnected_0 = disconnected_1;
        i_3 = i_3 + 256U;
    }
    (*(&kernelContext_3)->g_damaged_0)[tid_0] = damaged_1;
    (*(&kernelContext_3)->g_disconnected_0)[tid_0] = disconnected_0;
    (*(&kernelContext_3)->g_unconverged_0)[tid_0] = unconverged_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    damaged_0 = 128U;
    for(;;)
    {
        if(damaged_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(tid_0 < damaged_0)
        {
            uint _S60 = tid_0 + damaged_0;
            (*(&kernelContext_3)->g_damaged_0)[tid_0] = ((*(&kernelContext_3)->g_damaged_0)[tid_0]) | (*(&kernelContext_3)->g_damaged_0)[_S60];
            (*(&kernelContext_3)->g_disconnected_0)[tid_0] = ((*(&kernelContext_3)->g_disconnected_0)[tid_0]) | (*(&kernelContext_3)->g_disconnected_0)[_S60];
            (*(&kernelContext_3)->g_unconverged_0)[tid_0] = (*(&kernelContext_3)->g_unconverged_0)[tid_0] + (*(&kernelContext_3)->g_unconverged_0)[_S60];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        damaged_0 = damaged_0 >> 1U;
    }
    if(tid_0 != 0U)
    {
        return;
    }
    *((&kernelContext_3)->status_0+2U) = *((&kernelContext_3)->status_0+2U) + 1U;
    *((&kernelContext_3)->status_0+3U) = *((&kernelContext_3)->status_0+3U) + (*(&kernelContext_3)->g_unconverged_0)[int(0)];
    *((&kernelContext_3)->status_0+4U) = (*(&kernelContext_3)->g_damaged_0)[int(0)];
    if(((*(&kernelContext_3)->g_disconnected_0)[int(0)]) != 0U)
    {
        *((&kernelContext_3)->status_0+1U) = (&kernelContext_3)->params_0->pass_index_0 + 1U;
    }
    uint device* _S61 = (&kernelContext_3)->status_0+0U;
    bool _S62;
    if(((*(&kernelContext_3)->g_damaged_0)[int(0)]) != 0U)
    {
        _S62 = ((*(&kernelContext_3)->g_disconnected_0)[int(0)]) == 0U;
    }
    else
    {
        _S62 = false;
    }
    if(_S62)
    {
        damaged_1 = 1U;
    }
    else
    {
        damaged_1 = 0U;
    }
    *_S61 = damaged_1;
    return;
}

