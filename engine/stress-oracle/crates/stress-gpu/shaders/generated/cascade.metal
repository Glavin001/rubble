#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct CascadeParams_0
{
    uint island_count_0;
    uint result_base_0;
    uint pass_index_0;
    uint sload_base_0;
    uint gate_flag_base_0;
    uint bond_words_0;
    uint pad1_0;
    uint pad2_0;
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
    packed_float4 device* gate_0;
    packed_float4 device* state_0;
    packed_float4 device* backup_0;
    packed_float4 device* bond_dyn_0;
    packed_float4 device* bond_backup_0;
    uint threadgroup* g_moved_0;
    uint threadgroup* g_differs_0;
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

[[kernel]] void cascade_begin(uint3 id_0 [[thread_position_in_grid]], CascadeParams_0 constant* params_1 [[buffer(0)]], uint device* status_1 [[buffer(5)]], Island_natural_0 device* islands_1 [[buffer(1)]], packed_float4 device* scratch_1 [[buffer(2)]], packed_float4 device* gate_1 [[buffer(6)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* backup_1 [[buffer(4)]], packed_float4 device* bond_dyn_1 [[buffer(7)]], packed_float4 device* bond_backup_1 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->status_0 = status_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->scratch_0 = scratch_1;
    (&kernelContext_0)->gate_0 = gate_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->backup_0 = backup_1;
    (&kernelContext_0)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_0)->bond_backup_0 = bond_backup_1;
    threadgroup uint g_moved_1;
    (&kernelContext_0)->g_moved_0 = &g_moved_1;
    threadgroup uint g_differs_1;
    (&kernelContext_0)->g_differs_0 = &g_differs_1;
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

[[kernel]] void cascade_restore(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], CascadeParams_0 constant* params_2 [[buffer(0)]], uint device* status_2 [[buffer(5)]], Island_natural_0 device* islands_2 [[buffer(1)]], packed_float4 device* scratch_2 [[buffer(2)]], packed_float4 device* gate_2 [[buffer(6)]], packed_float4 device* state_2 [[buffer(3)]], packed_float4 device* backup_2 [[buffer(4)]], packed_float4 device* bond_dyn_2 [[buffer(7)]], packed_float4 device* bond_backup_2 [[buffer(8)]])
{
    uint _S29;
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->status_0 = status_2;
    (&kernelContext_1)->islands_0 = islands_2;
    (&kernelContext_1)->scratch_0 = scratch_2;
    (&kernelContext_1)->gate_0 = gate_2;
    (&kernelContext_1)->state_0 = state_2;
    (&kernelContext_1)->backup_0 = backup_2;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_1)->bond_backup_0 = bond_backup_2;
    threadgroup uint g_moved_2;
    (&kernelContext_1)->g_moved_0 = &g_moved_2;
    threadgroup uint g_differs_2;
    (&kernelContext_1)->g_differs_0 = &g_differs_2;
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
    uint _S30 = 16U + i_1;
    uint s_0 = *((&kernelContext_1)->status_0+_S30);
    bool _S31;
    if((*((&kernelContext_1)->status_0+0U)) == 0U)
    {
        _S31 = true;
    }
    else
    {
        _S31 = (s_0 & 1U) == 0U;
    }
    if(_S31)
    {
        return;
    }
    bool converged_0 = ((float4(*((&kernelContext_1)->scratch_0+((&kernelContext_1)->params_0->result_base_0 + i_1))) ).w) != 0.0f;
    uint4 _S32 = uint4(((&kernelContext_1)->islands_0+i_1)->range_0) ;
    uint _S33 = _S32.x;
    uint _S34 = thread_0.x;
    uint k_0 = 2U * _S33 + _S34;
    for(;;)
    {
        uint _S35 = _S32.y;
        _S29 = _S35;
        if(k_0 < (2U * _S35))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_1)->gate_0+k_0) = packed_float4(float4(*((&kernelContext_1)->scratch_0+((&kernelContext_1)->params_0->sload_base_0 + k_0))) ) ;
        k_0 = k_0 + 64U;
    }
    if(converged_0)
    {
        if(_S34 == 0U)
        {
            *((&kernelContext_1)->status_0+_S30) = s_0 | 2U;
            *((&kernelContext_1)->gate_0+((&kernelContext_1)->params_0->gate_flag_base_0 + i_1)) = packed_float4(float4(0.0f) ) ;
        }
        return;
    }
    k_0 = 4U * _S33 + _S34;
    for(;;)
    {
        if(k_0 < (4U * _S29))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_1)->state_0+k_0) = packed_float4(float4(*((&kernelContext_1)->backup_0+k_0)) ) ;
        k_0 = k_0 + 64U;
    }
    if(_S34 == 0U)
    {
        *((&kernelContext_1)->gate_0+((&kernelContext_1)->params_0->gate_flag_base_0 + i_1)) = packed_float4(float4(1.0f, 0.0f, 0.0f, 0.0f)) ;
    }
    return;
}

[[kernel]] void cascade_fixed(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], CascadeParams_0 constant* params_3 [[buffer(0)]], uint device* status_3 [[buffer(5)]], Island_natural_0 device* islands_3 [[buffer(1)]], packed_float4 device* scratch_3 [[buffer(2)]], packed_float4 device* gate_3 [[buffer(6)]], packed_float4 device* state_3 [[buffer(3)]], packed_float4 device* backup_3 [[buffer(4)]], packed_float4 device* bond_dyn_3 [[buffer(7)]], packed_float4 device* bond_backup_3 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_3;
    (&kernelContext_2)->status_0 = status_3;
    (&kernelContext_2)->islands_0 = islands_3;
    (&kernelContext_2)->scratch_0 = scratch_3;
    (&kernelContext_2)->gate_0 = gate_3;
    (&kernelContext_2)->state_0 = state_3;
    (&kernelContext_2)->backup_0 = backup_3;
    (&kernelContext_2)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_2)->bond_backup_0 = bond_backup_3;
    threadgroup uint g_moved_3;
    (&kernelContext_2)->g_moved_0 = &g_moved_3;
    threadgroup uint g_differs_3;
    (&kernelContext_2)->g_differs_0 = &g_differs_3;
    threadgroup array<uint, int(256)> g_damaged_3;
    (&kernelContext_2)->g_damaged_0 = &g_damaged_3;
    threadgroup array<uint, int(256)> g_disconnected_3;
    (&kernelContext_2)->g_disconnected_0 = &g_disconnected_3;
    threadgroup array<uint, int(256)> g_unconverged_3;
    (&kernelContext_2)->g_unconverged_0 = &g_unconverged_3;
    uint i_2 = group_1.x;
    if(i_2 >= (params_3->island_count_0))
    {
        return;
    }
    uint s_1 = *((&kernelContext_2)->status_0+(16U + i_2));
    bool _S36;
    if((*((&kernelContext_2)->status_0+0U)) == 0U)
    {
        _S36 = true;
    }
    else
    {
        _S36 = (s_1 & 3U) != 3U;
    }
    if(_S36)
    {
        _S36 = true;
    }
    else
    {
        _S36 = ((&kernelContext_2)->params_0->pass_index_0) != 0U;
    }
    if(_S36)
    {
        return;
    }
    uint _S37 = thread_1.x;
    bool _S38 = _S37 == 0U;
    if(_S38)
    {
        *(&kernelContext_2)->g_moved_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 _S39 = uint4(((&kernelContext_2)->islands_0+i_2)->range_0) ;
    uint _S40 = 4U * _S39.x + _S37;
    uint moved_0 = 0U;
    uint k_1 = _S40;
    for(;;)
    {
        if(k_1 < (4U * _S39.y))
        {
        }
        else
        {
            break;
        }
        if(any((as_type<uint4>((float4(*((&kernelContext_2)->state_0+k_1)) ))) != (as_type<uint4>((float4(*((&kernelContext_2)->backup_0+k_1)) )))))
        {
            moved_0 = 1U;
        }
        k_1 = k_1 + 64U;
    }
    k_1 = (&kernelContext_2)->params_0->bond_words_0 * _S39.z + _S37;
    for(;;)
    {
        if(k_1 < ((&kernelContext_2)->params_0->bond_words_0 * _S39.w))
        {
        }
        else
        {
            break;
        }
        if(any((as_type<uint4>((float4(*((&kernelContext_2)->bond_dyn_0+k_1)) ))) != (as_type<uint4>((float4(*((&kernelContext_2)->bond_backup_0+k_1)) )))))
        {
            moved_0 = 1U;
        }
        k_1 = k_1 + 64U;
    }
    if(moved_0 != 0U)
    {
        *(&kernelContext_2)->g_moved_0 = 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S38)
    {
        _S36 = (*(&kernelContext_2)->g_moved_0) == 0U;
    }
    else
    {
        _S36 = false;
    }
    if(_S36)
    {
        *((&kernelContext_2)->gate_0+((&kernelContext_2)->params_0->gate_flag_base_0 + i_2)) = packed_float4(float4(2.0f, 0.0f, 0.0f, 0.0f)) ;
    }
    return;
}

[[kernel]] void cascade_gate(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], CascadeParams_0 constant* params_4 [[buffer(0)]], uint device* status_4 [[buffer(5)]], Island_natural_0 device* islands_4 [[buffer(1)]], packed_float4 device* scratch_4 [[buffer(2)]], packed_float4 device* gate_4 [[buffer(6)]], packed_float4 device* state_4 [[buffer(3)]], packed_float4 device* backup_4 [[buffer(4)]], packed_float4 device* bond_dyn_4 [[buffer(7)]], packed_float4 device* bond_backup_4 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_4;
    (&kernelContext_3)->status_0 = status_4;
    (&kernelContext_3)->islands_0 = islands_4;
    (&kernelContext_3)->scratch_0 = scratch_4;
    (&kernelContext_3)->gate_0 = gate_4;
    (&kernelContext_3)->state_0 = state_4;
    (&kernelContext_3)->backup_0 = backup_4;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_3)->bond_backup_0 = bond_backup_4;
    threadgroup uint g_moved_4;
    (&kernelContext_3)->g_moved_0 = &g_moved_4;
    threadgroup uint g_differs_4;
    (&kernelContext_3)->g_differs_0 = &g_differs_4;
    threadgroup array<uint, int(256)> g_damaged_4;
    (&kernelContext_3)->g_damaged_0 = &g_damaged_4;
    threadgroup array<uint, int(256)> g_disconnected_4;
    (&kernelContext_3)->g_disconnected_0 = &g_disconnected_4;
    threadgroup array<uint, int(256)> g_unconverged_4;
    (&kernelContext_3)->g_unconverged_0 = &g_unconverged_4;
    uint i_3 = group_2.x;
    if(i_3 >= (params_4->island_count_0))
    {
        return;
    }
    uint _S41 = 16U + i_3;
    uint s_2 = *((&kernelContext_3)->status_0+_S41);
    float flag_0 = (float4(*((&kernelContext_3)->gate_0+((&kernelContext_3)->params_0->gate_flag_base_0 + i_3))) ).x;
    bool _S42;
    if((*((&kernelContext_3)->status_0+0U)) == 0U)
    {
        _S42 = true;
    }
    else
    {
        _S42 = (s_2 & 1U) == 0U;
    }
    if(_S42)
    {
        _S42 = true;
    }
    else
    {
        _S42 = flag_0 == 0.0f;
    }
    if(_S42)
    {
        _S42 = true;
    }
    else
    {
        if(flag_0 == 2.0f)
        {
            _S42 = ((&kernelContext_3)->params_0->pass_index_0) != 0U;
        }
        else
        {
            _S42 = false;
        }
    }
    if(_S42)
    {
        return;
    }
    uint _S43 = thread_2.x;
    if(_S43 == 0U)
    {
        *(&kernelContext_3)->g_differs_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Island_natural_0 device* _S44 = (&kernelContext_3)->islands_0+i_3;
    Island_natural_0 isl_1 = *_S44;
    uint4 _S45 = uint4((*_S44).range_0) ;
    uint _S46 = 2U * _S45.x + _S43;
    uint differs_0 = 0U;
    uint k_2 = _S46;
    for(;;)
    {
        if(k_2 < (2U * _S45.y))
        {
        }
        else
        {
            break;
        }
        if(any((as_type<uint4>((float4(*((&kernelContext_3)->gate_0+k_2)) ))) != (as_type<uint4>((float4(*((&kernelContext_3)->scratch_0+((&kernelContext_3)->params_0->sload_base_0 + k_2))) )))))
        {
            differs_0 = 1U;
        }
        k_2 = k_2 + 64U;
    }
    if(differs_0 != 0U)
    {
        *(&kernelContext_3)->g_differs_0 = 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(_S43 != 0U)
    {
        _S42 = true;
    }
    else
    {
        _S42 = (*(&kernelContext_3)->g_differs_0) != 0U;
    }
    if(_S42)
    {
        return;
    }
    ((&kernelContext_3)->islands_0+i_3)->info_0[int(2)] = ((uint4(isl_1.info_0) ).z) & 4294967287U;
    uint device* _S47 = (&kernelContext_3)->status_0+_S41;
    if(flag_0 == 2.0f)
    {
        differs_0 = 32U;
    }
    else
    {
        differs_0 = 16U;
    }
    *_S47 = differs_0;
    return;
}

[[kernel]] void cascade_settle(uint3 id_1 [[thread_position_in_grid]], CascadeParams_0 constant* params_5 [[buffer(0)]], uint device* status_5 [[buffer(5)]], Island_natural_0 device* islands_5 [[buffer(1)]], packed_float4 device* scratch_5 [[buffer(2)]], packed_float4 device* gate_5 [[buffer(6)]], packed_float4 device* state_5 [[buffer(3)]], packed_float4 device* backup_5 [[buffer(4)]], packed_float4 device* bond_dyn_5 [[buffer(7)]], packed_float4 device* bond_backup_5 [[buffer(8)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_0 = params_5;
    (&kernelContext_4)->status_0 = status_5;
    (&kernelContext_4)->islands_0 = islands_5;
    (&kernelContext_4)->scratch_0 = scratch_5;
    (&kernelContext_4)->gate_0 = gate_5;
    (&kernelContext_4)->state_0 = state_5;
    (&kernelContext_4)->backup_0 = backup_5;
    (&kernelContext_4)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_4)->bond_backup_0 = bond_backup_5;
    threadgroup uint g_moved_5;
    (&kernelContext_4)->g_moved_0 = &g_moved_5;
    threadgroup uint g_differs_5;
    (&kernelContext_4)->g_differs_0 = &g_differs_5;
    threadgroup array<uint, int(256)> g_damaged_5;
    (&kernelContext_4)->g_damaged_0 = &g_damaged_5;
    threadgroup array<uint, int(256)> g_disconnected_5;
    (&kernelContext_4)->g_disconnected_0 = &g_disconnected_5;
    threadgroup array<uint, int(256)> g_unconverged_5;
    (&kernelContext_4)->g_unconverged_0 = &g_unconverged_5;
    uint i_4 = id_1.x;
    if(i_4 >= (params_5->island_count_0))
    {
        return;
    }
    uint s_3 = *((&kernelContext_4)->status_0+(16U + i_4));
    Island_natural_0 device* _S48 = (&kernelContext_4)->islands_0+i_4;
    uint4 _S49 = uint4((*_S48).info_0) ;
    float4 _S50 = float4((*_S48).com_0) ;
    float4 _S51 = float4((*_S48).inertia0_0) ;
    float4 _S52 = float4((*_S48).inertia1_0) ;
    float4 _S53 = float4((*_S48).inertia2_0) ;
    float4 _S54 = float4((*_S48).inv0_0) ;
    float4 _S55 = float4((*_S48).inv1_0) ;
    float4 _S56 = float4((*_S48).inv2_0) ;
    float4 _S57 = float4((*_S48).wcom_0) ;
    float4 _S58 = float4((*_S48).winv0_0) ;
    float4 _S59 = float4((*_S48).winv1_0) ;
    float4 _S60 = float4((*_S48).winv2_0) ;
    float4 _S61 = float4((*_S48).rotation_0) ;
    float4 _S62 = float4((*_S48).position_0) ;
    float4 _S63 = float4((*_S48).position_err_0) ;
    float4 _S64 = float4((*_S48).velocity_0) ;
    float4 _S65 = float4((*_S48).velocity_err_0) ;
    float4 _S66 = float4((*_S48).angular_velocity_0) ;
    uint4 _S67 = uint4((*_S48).probes_0) ;
    float4 _S68 = float4((*_S48).energy_0) ;
    float4 _S69 = float4((*_S48).rotation_err_0) ;
    float4 _S70 = float4((*_S48).momentum_0) ;
    float4 _S71 = float4((*_S48).momentum_err_0) ;
    thread Island_0 isl_2;
    (&isl_2)->range_0 = uint4((*_S48).range_0) ;
    (&isl_2)->info_0 = _S49;
    (&isl_2)->com_0 = _S50;
    (&isl_2)->inertia0_0 = _S51;
    (&isl_2)->inertia1_0 = _S52;
    (&isl_2)->inertia2_0 = _S53;
    (&isl_2)->inv0_0 = _S54;
    (&isl_2)->inv1_0 = _S55;
    (&isl_2)->inv2_0 = _S56;
    (&isl_2)->wcom_0 = _S57;
    (&isl_2)->winv0_0 = _S58;
    (&isl_2)->winv1_0 = _S59;
    (&isl_2)->winv2_0 = _S60;
    (&isl_2)->rotation_0 = _S61;
    (&isl_2)->position_0 = _S62;
    (&isl_2)->position_err_0 = _S63;
    (&isl_2)->velocity_0 = _S64;
    (&isl_2)->velocity_err_0 = _S65;
    (&isl_2)->angular_velocity_0 = _S66;
    (&isl_2)->done_0 = (*_S48).done_0;
    (&isl_2)->coverage_0 = (*_S48).coverage_0;
    (&isl_2)->wide_groups_0 = (*_S48).wide_groups_0;
    (&isl_2)->done_pad_0 = (*_S48).done_pad_0;
    (&isl_2)->probes_0 = _S67;
    (&isl_2)->energy_0 = _S68;
    (&isl_2)->rotation_err_0 = _S69;
    (&isl_2)->momentum_0 = _S70;
    (&isl_2)->momentum_err_0 = _S71;
    (&isl_2)->info_0.x = ((&isl_2)->info_0.x) & 4294967279U;
    (&isl_2)->info_0.z = ((&isl_2)->info_0.z) & 4294967247U;
    bool _S72;
    if((*((&kernelContext_4)->status_0+0U)) != 0U)
    {
        _S72 = (s_3 & 3U) == 3U;
    }
    else
    {
        _S72 = false;
    }
    if(_S72)
    {
        (&isl_2)->info_0.x = ((&isl_2)->info_0.x) | 16U;
    }
    Island_natural_0 device* _S73 = (&kernelContext_4)->islands_0+i_4;
    _S73->range_0 = packed_uint4(isl_2.range_0) ;
    _S73->info_0 = packed_uint4(isl_2.info_0) ;
    _S73->com_0 = packed_float4(isl_2.com_0) ;
    _S73->inertia0_0 = packed_float4(isl_2.inertia0_0) ;
    _S73->inertia1_0 = packed_float4(isl_2.inertia1_0) ;
    _S73->inertia2_0 = packed_float4(isl_2.inertia2_0) ;
    _S73->inv0_0 = packed_float4(isl_2.inv0_0) ;
    _S73->inv1_0 = packed_float4(isl_2.inv1_0) ;
    _S73->inv2_0 = packed_float4(isl_2.inv2_0) ;
    _S73->wcom_0 = packed_float4(isl_2.wcom_0) ;
    _S73->winv0_0 = packed_float4(isl_2.winv0_0) ;
    _S73->winv1_0 = packed_float4(isl_2.winv1_0) ;
    _S73->winv2_0 = packed_float4(isl_2.winv2_0) ;
    _S73->rotation_0 = packed_float4(isl_2.rotation_0) ;
    _S73->position_0 = packed_float4(isl_2.position_0) ;
    _S73->position_err_0 = packed_float4(isl_2.position_err_0) ;
    _S73->velocity_0 = packed_float4(isl_2.velocity_0) ;
    _S73->velocity_err_0 = packed_float4(isl_2.velocity_err_0) ;
    _S73->angular_velocity_0 = packed_float4(isl_2.angular_velocity_0) ;
    _S73->done_0 = isl_2.done_0;
    _S73->coverage_0 = isl_2.coverage_0;
    _S73->wide_groups_0 = isl_2.wide_groups_0;
    _S73->done_pad_0 = isl_2.done_pad_0;
    _S73->probes_0 = packed_uint4(isl_2.probes_0) ;
    _S73->energy_0 = packed_float4(isl_2.energy_0) ;
    _S73->rotation_err_0 = packed_float4(isl_2.rotation_err_0) ;
    _S73->momentum_0 = packed_float4(isl_2.momentum_0) ;
    _S73->momentum_err_0 = packed_float4(isl_2.momentum_err_0) ;
    return;
}

[[kernel]] void cascade_end(uint3 thread_3 [[thread_position_in_threadgroup]], CascadeParams_0 constant* params_6 [[buffer(0)]], uint device* status_6 [[buffer(5)]], Island_natural_0 device* islands_6 [[buffer(1)]], packed_float4 device* scratch_6 [[buffer(2)]], packed_float4 device* gate_6 [[buffer(6)]], packed_float4 device* state_6 [[buffer(3)]], packed_float4 device* backup_6 [[buffer(4)]], packed_float4 device* bond_dyn_6 [[buffer(7)]], packed_float4 device* bond_backup_6 [[buffer(8)]])
{
    uint damaged_0;
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_0 = params_6;
    (&kernelContext_5)->status_0 = status_6;
    (&kernelContext_5)->islands_0 = islands_6;
    (&kernelContext_5)->scratch_0 = scratch_6;
    (&kernelContext_5)->gate_0 = gate_6;
    (&kernelContext_5)->state_0 = state_6;
    (&kernelContext_5)->backup_0 = backup_6;
    (&kernelContext_5)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_5)->bond_backup_0 = bond_backup_6;
    threadgroup uint g_moved_6;
    (&kernelContext_5)->g_moved_0 = &g_moved_6;
    threadgroup uint g_differs_6;
    (&kernelContext_5)->g_differs_0 = &g_differs_6;
    threadgroup array<uint, int(256)> g_damaged_6;
    (&kernelContext_5)->g_damaged_0 = &g_damaged_6;
    threadgroup array<uint, int(256)> g_disconnected_6;
    (&kernelContext_5)->g_disconnected_0 = &g_disconnected_6;
    threadgroup array<uint, int(256)> g_unconverged_6;
    (&kernelContext_5)->g_unconverged_0 = &g_unconverged_6;
    uint tid_0 = thread_3.x;
    if((*(status_6+0U)) == 0U)
    {
        return;
    }
    uint damaged_1 = 0U;
    uint disconnected_0 = 0U;
    uint i_5 = tid_0;
    uint unconverged_0 = 0U;
    for(;;)
    {
        if(i_5 < ((&kernelContext_5)->params_0->island_count_0))
        {
        }
        else
        {
            break;
        }
        uint _S74 = 16U + i_5;
        uint device* _S75 = (&kernelContext_5)->status_0+_S74;
        uint s_4 = *_S75;
        if(((*_S75) & 16U) != 0U)
        {
            unconverged_0 = unconverged_0 + 1U;
            i_5 = i_5 + 256U;
            continue;
        }
        if((s_4 & 1U) == 0U)
        {
            i_5 = i_5 + 256U;
            continue;
        }
        if((s_4 & 2U) == 0U)
        {
            unconverged_0 = unconverged_0 + 1U;
            i_5 = i_5 + 256U;
            continue;
        }
        uint z_0 = (uint4(((&kernelContext_5)->islands_0+i_5)->info_0) ).z;
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
        *((&kernelContext_5)->status_0+_S74) = s_4 | add_1;
        damaged_1 = damaged_0;
        disconnected_0 = disconnected_1;
        i_5 = i_5 + 256U;
    }
    (*(&kernelContext_5)->g_damaged_0)[tid_0] = damaged_1;
    (*(&kernelContext_5)->g_disconnected_0)[tid_0] = disconnected_0;
    (*(&kernelContext_5)->g_unconverged_0)[tid_0] = unconverged_0;
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
            uint _S76 = tid_0 + damaged_0;
            (*(&kernelContext_5)->g_damaged_0)[tid_0] = ((*(&kernelContext_5)->g_damaged_0)[tid_0]) | (*(&kernelContext_5)->g_damaged_0)[_S76];
            (*(&kernelContext_5)->g_disconnected_0)[tid_0] = ((*(&kernelContext_5)->g_disconnected_0)[tid_0]) | (*(&kernelContext_5)->g_disconnected_0)[_S76];
            (*(&kernelContext_5)->g_unconverged_0)[tid_0] = (*(&kernelContext_5)->g_unconverged_0)[tid_0] + (*(&kernelContext_5)->g_unconverged_0)[_S76];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        damaged_0 = damaged_0 >> 1U;
    }
    if(tid_0 != 0U)
    {
        return;
    }
    *((&kernelContext_5)->status_0+2U) = *((&kernelContext_5)->status_0+2U) + 1U;
    *((&kernelContext_5)->status_0+3U) = *((&kernelContext_5)->status_0+3U) + (*(&kernelContext_5)->g_unconverged_0)[int(0)];
    *((&kernelContext_5)->status_0+4U) = (*(&kernelContext_5)->g_damaged_0)[int(0)];
    if(((*(&kernelContext_5)->g_disconnected_0)[int(0)]) != 0U)
    {
        *((&kernelContext_5)->status_0+1U) = (&kernelContext_5)->params_0->pass_index_0 + 1U;
    }
    uint device* _S77 = (&kernelContext_5)->status_0+0U;
    bool _S78;
    if(((*(&kernelContext_5)->g_damaged_0)[int(0)]) != 0U)
    {
        _S78 = ((*(&kernelContext_5)->g_disconnected_0)[int(0)]) == 0U;
    }
    else
    {
        _S78 = false;
    }
    if(_S78)
    {
        damaged_1 = 1U;
    }
    else
    {
        damaged_1 = 0U;
    }
    *_S77 = damaged_1;
    return;
}

