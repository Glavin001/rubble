struct CascadeParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) result_base_0 : u32,
    @align(8) pass_index_0 : u32,
    @align(4) sload_base_0 : u32,
    @align(16) gate_flag_base_0 : u32,
    @align(4) bond_words_0 : u32,
    @align(8) certify_0 : u32,
    @align(4) unconverged_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : CascadeParams_std140_0;
@binding(5) @group(0) var<storage, read_write> status_0 : array<u32>;

struct Island_std430_0
{
    @align(16) range_0 : vec4<u32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) com_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
    @align(16) wcom_0 : vec4<f32>,
    @align(16) winv0_0 : vec4<f32>,
    @align(16) winv1_0 : vec4<f32>,
    @align(16) winv2_0 : vec4<f32>,
    @align(16) rotation_0 : vec4<f32>,
    @align(16) position_0 : vec4<f32>,
    @align(16) position_err_0 : vec4<f32>,
    @align(16) velocity_0 : vec4<f32>,
    @align(16) velocity_err_0 : vec4<f32>,
    @align(16) angular_velocity_0 : vec4<f32>,
    @align(16) done_0 : u32,
    @align(4) coverage_0 : u32,
    @align(8) wide_groups_0 : u32,
    @align(4) done_pad_0 : u32,
    @align(16) probes_0 : vec4<u32>,
    @align(16) energy_0 : vec4<f32>,
    @align(16) rotation_err_0 : vec4<f32>,
    @align(16) momentum_0 : vec4<f32>,
    @align(16) momentum_err_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read_write> islands_0 : array<Island_std430_0>;

@binding(2) @group(0) var<storage, read> scratch_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> gate_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read_write> state_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> backup_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read> bond_dyn_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> bond_backup_0 : array<vec4<f32>>;

struct Island_0
{
     range_0 : vec4<u32>,
     info_0 : vec4<u32>,
     com_0 : vec4<f32>,
     inertia0_0 : vec4<f32>,
     inertia1_0 : vec4<f32>,
     inertia2_0 : vec4<f32>,
     inv0_0 : vec4<f32>,
     inv1_0 : vec4<f32>,
     inv2_0 : vec4<f32>,
     wcom_0 : vec4<f32>,
     winv0_0 : vec4<f32>,
     winv1_0 : vec4<f32>,
     winv2_0 : vec4<f32>,
     rotation_0 : vec4<f32>,
     position_0 : vec4<f32>,
     position_err_0 : vec4<f32>,
     velocity_0 : vec4<f32>,
     velocity_err_0 : vec4<f32>,
     angular_velocity_0 : vec4<f32>,
     done_0 : u32,
     coverage_0 : u32,
     wide_groups_0 : u32,
     done_pad_0 : u32,
     probes_0 : vec4<u32>,
     energy_0 : vec4<f32>,
     rotation_err_0 : vec4<f32>,
     momentum_0 : vec4<f32>,
     momentum_err_0 : vec4<f32>,
};

@compute
@workgroup_size(64, 1, 1)
fn cascade_begin(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_0 : u32 = id_0.x;
    if(i_0 >= (params_0.island_count_0))
    {
        return;
    }
    var run_0 : bool = status_0[u32(0)] != u32(0);
    var isl_0 : Island_0;
    isl_0.range_0 = islands_0[i_0].range_0;
    isl_0.info_0 = islands_0[i_0].info_0;
    isl_0.com_0 = islands_0[i_0].com_0;
    isl_0.inertia0_0 = islands_0[i_0].inertia0_0;
    isl_0.inertia1_0 = islands_0[i_0].inertia1_0;
    isl_0.inertia2_0 = islands_0[i_0].inertia2_0;
    isl_0.inv0_0 = islands_0[i_0].inv0_0;
    isl_0.inv1_0 = islands_0[i_0].inv1_0;
    isl_0.inv2_0 = islands_0[i_0].inv2_0;
    isl_0.wcom_0 = islands_0[i_0].wcom_0;
    isl_0.winv0_0 = islands_0[i_0].winv0_0;
    isl_0.winv1_0 = islands_0[i_0].winv1_0;
    isl_0.winv2_0 = islands_0[i_0].winv2_0;
    isl_0.rotation_0 = islands_0[i_0].rotation_0;
    isl_0.position_0 = islands_0[i_0].position_0;
    isl_0.position_err_0 = islands_0[i_0].position_err_0;
    isl_0.velocity_0 = islands_0[i_0].velocity_0;
    isl_0.velocity_err_0 = islands_0[i_0].velocity_err_0;
    isl_0.angular_velocity_0 = islands_0[i_0].angular_velocity_0;
    isl_0.done_0 = islands_0[i_0].done_0;
    isl_0.coverage_0 = islands_0[i_0].coverage_0;
    isl_0.wide_groups_0 = islands_0[i_0].wide_groups_0;
    isl_0.done_pad_0 = islands_0[i_0].done_pad_0;
    isl_0.probes_0 = islands_0[i_0].probes_0;
    isl_0.energy_0 = islands_0[i_0].energy_0;
    isl_0.rotation_err_0 = islands_0[i_0].rotation_err_0;
    isl_0.momentum_0 = islands_0[i_0].momentum_0;
    isl_0.momentum_err_0 = islands_0[i_0].momentum_err_0;
    var solve_0 : bool;
    if(run_0)
    {
        solve_0 = (isl_0.range_0.w) > (isl_0.range_0.z);
    }
    else
    {
        solve_0 = false;
    }
    var _S1 : u32 = ((isl_0.info_0.z) & (u32(4294967287)));
    var _S2 : u32;
    if(solve_0)
    {
        _S2 = u32(8);
    }
    else
    {
        _S2 = u32(0);
    }
    isl_0.info_0[i32(2)] = (_S1 | (_S2));
    islands_0[i_0].range_0 = isl_0.range_0;
    islands_0[i_0].info_0 = isl_0.info_0;
    islands_0[i_0].com_0 = isl_0.com_0;
    islands_0[i_0].inertia0_0 = isl_0.inertia0_0;
    islands_0[i_0].inertia1_0 = isl_0.inertia1_0;
    islands_0[i_0].inertia2_0 = isl_0.inertia2_0;
    islands_0[i_0].inv0_0 = isl_0.inv0_0;
    islands_0[i_0].inv1_0 = isl_0.inv1_0;
    islands_0[i_0].inv2_0 = isl_0.inv2_0;
    islands_0[i_0].wcom_0 = isl_0.wcom_0;
    islands_0[i_0].winv0_0 = isl_0.winv0_0;
    islands_0[i_0].winv1_0 = isl_0.winv1_0;
    islands_0[i_0].winv2_0 = isl_0.winv2_0;
    islands_0[i_0].rotation_0 = isl_0.rotation_0;
    islands_0[i_0].position_0 = isl_0.position_0;
    islands_0[i_0].position_err_0 = isl_0.position_err_0;
    islands_0[i_0].velocity_0 = isl_0.velocity_0;
    islands_0[i_0].velocity_err_0 = isl_0.velocity_err_0;
    islands_0[i_0].angular_velocity_0 = isl_0.angular_velocity_0;
    islands_0[i_0].done_0 = isl_0.done_0;
    islands_0[i_0].coverage_0 = isl_0.coverage_0;
    islands_0[i_0].wide_groups_0 = isl_0.wide_groups_0;
    islands_0[i_0].done_pad_0 = isl_0.done_pad_0;
    islands_0[i_0].probes_0 = isl_0.probes_0;
    islands_0[i_0].energy_0 = isl_0.energy_0;
    islands_0[i_0].rotation_err_0 = isl_0.rotation_err_0;
    islands_0[i_0].momentum_0 = isl_0.momentum_0;
    islands_0[i_0].momentum_err_0 = isl_0.momentum_err_0;
    if(run_0)
    {
        if(solve_0)
        {
            _S2 = u32(1);
        }
        else
        {
            _S2 = u32(0);
        }
        status_0[u32(16) + i_0] = _S2;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn cascade_restore(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var _S3 : u32;
    var i_1 : u32 = group_0.x;
    if(i_1 >= (params_0.island_count_0))
    {
        return;
    }
    var _S4 : u32 = u32(16) + i_1;
    var s_0 : u32 = status_0[_S4];
    var converged_0 : bool;
    if(status_0[u32(0)] == u32(0))
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = ((s_0 & (u32(1)))) == u32(0);
    }
    if(converged_0)
    {
        return;
    }
    if(((s_0 & (u32(64)))) != u32(0))
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (scratch_0[params_0.result_base_0 + i_1].w) != 0.0f;
    }
    var _S5 : vec4<u32> = islands_0[i_1].range_0;
    var _S6 : u32 = islands_0[i_1].range_0.x;
    var _S7 : u32 = thread_0.x;
    var k_0 : u32 = u32(2) * _S6 + _S7;
    loop
    {
        var _S8 : u32 = _S5.y;
        _S3 = _S8;
        if(k_0 < (u32(2) * _S8))
        {
        }
        else
        {
            break;
        }
        gate_0[k_0] = scratch_0[params_0.sload_base_0 + k_0];
        k_0 = k_0 + u32(64);
    }
    if(converged_0)
    {
        if(_S7 == u32(0))
        {
            status_0[_S4] = (s_0 | (u32(2)));
            var _S9 : f32;
            if((params_0.certify_0) != u32(0))
            {
                _S9 = 3.0f;
            }
            else
            {
                _S9 = 0.0f;
            }
            gate_0[params_0.gate_flag_base_0 + i_1] = vec4<f32>(_S9, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    if((params_0.unconverged_0) != u32(0))
    {
        if(_S7 == u32(0))
        {
            gate_0[params_0.gate_flag_base_0 + i_1] = vec4<f32>(0.0f);
        }
        return;
    }
    k_0 = u32(4) * _S6 + _S7;
    loop
    {
        if(k_0 < (u32(4) * _S3))
        {
        }
        else
        {
            break;
        }
        state_0[k_0] = backup_0[k_0];
        k_0 = k_0 + u32(64);
    }
    if(_S7 == u32(0))
    {
        gate_0[params_0.gate_flag_base_0 + i_1] = vec4<f32>(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return;
}

var<workgroup> g_moved_0 : u32;

@compute
@workgroup_size(64, 1, 1)
fn cascade_fixed(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var i_2 : u32 = group_1.x;
    if(i_2 >= (params_0.island_count_0))
    {
        return;
    }
    var s_1 : u32 = status_0[u32(16) + i_2];
    var _S10 : bool;
    if(status_0[u32(0)] == u32(0))
    {
        _S10 = true;
    }
    else
    {
        _S10 = ((s_1 & (u32(3)))) != u32(3);
    }
    if(_S10)
    {
        _S10 = true;
    }
    else
    {
        _S10 = (params_0.pass_index_0) != u32(0);
    }
    if(_S10)
    {
        return;
    }
    var _S11 : u32 = thread_1.x;
    var _S12 : bool = _S11 == u32(0);
    if(_S12)
    {
        g_moved_0 = u32(0);
    }
    workgroupBarrier();
    var _S13 : vec4<u32> = islands_0[i_2].range_0;
    var _S14 : u32 = u32(4) * islands_0[i_2].range_0.x + _S11;
    var moved_0 : u32 = u32(0);
    var k_1 : u32 = _S14;
    loop
    {
        if(k_1 < (u32(4) * _S13.y))
        {
        }
        else
        {
            break;
        }
        if((any(((bitcast<vec4<u32>>((state_0[k_1]))) != (bitcast<vec4<u32>>((backup_0[k_1])))))))
        {
            moved_0 = u32(1);
        }
        k_1 = k_1 + u32(64);
    }
    k_1 = params_0.bond_words_0 * _S13.z + _S11;
    loop
    {
        if(k_1 < (params_0.bond_words_0 * _S13.w))
        {
        }
        else
        {
            break;
        }
        if((any(((bitcast<vec4<u32>>((bond_dyn_0[k_1]))) != (bitcast<vec4<u32>>((bond_backup_0[k_1])))))))
        {
            moved_0 = u32(1);
        }
        k_1 = k_1 + u32(64);
    }
    if(moved_0 != u32(0))
    {
        g_moved_0 = u32(1);
    }
    workgroupBarrier();
    if(_S12)
    {
        _S10 = g_moved_0 == u32(0);
    }
    else
    {
        _S10 = false;
    }
    if(_S10)
    {
        gate_0[params_0.gate_flag_base_0 + i_2] = vec4<f32>(2.0f, 0.0f, 0.0f, 0.0f);
    }
    return;
}

var<workgroup> g_differs_0 : u32;

@compute
@workgroup_size(64, 1, 1)
fn cascade_gate(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var i_3 : u32 = group_2.x;
    if(i_3 >= (params_0.island_count_0))
    {
        return;
    }
    var _S15 : u32 = u32(16) + i_3;
    var s_2 : u32 = status_0[_S15];
    var flag_0 : f32 = gate_0[params_0.gate_flag_base_0 + i_3].x;
    var _S16 : bool;
    if(status_0[u32(0)] == u32(0))
    {
        _S16 = true;
    }
    else
    {
        _S16 = ((s_2 & (u32(1)))) == u32(0);
    }
    if(_S16)
    {
        _S16 = true;
    }
    else
    {
        _S16 = flag_0 == 0.0f;
    }
    if(_S16)
    {
        _S16 = true;
    }
    else
    {
        if(flag_0 == 2.0f)
        {
            _S16 = (params_0.pass_index_0) != u32(0);
        }
        else
        {
            _S16 = false;
        }
    }
    if(_S16)
    {
        return;
    }
    var _S17 : u32 = thread_2.x;
    if(_S17 == u32(0))
    {
        g_differs_0 = u32(0);
    }
    workgroupBarrier();
    var isl_1 : Island_std430_0 = islands_0[i_3];
    var _S18 : u32 = u32(2) * islands_0[i_3].range_0.x + _S17;
    var differs_0 : u32 = u32(0);
    var k_2 : u32 = _S18;
    loop
    {
        if(k_2 < (u32(2) * isl_1.range_0.y))
        {
        }
        else
        {
            break;
        }
        if((any(((bitcast<vec4<u32>>((gate_0[k_2]))) != (bitcast<vec4<u32>>((scratch_0[params_0.sload_base_0 + k_2])))))))
        {
            differs_0 = u32(1);
        }
        k_2 = k_2 + u32(64);
    }
    if(differs_0 != u32(0))
    {
        g_differs_0 = u32(1);
    }
    workgroupBarrier();
    if(_S17 != u32(0))
    {
        _S16 = true;
    }
    else
    {
        _S16 = g_differs_0 != u32(0);
    }
    if(_S16)
    {
        return;
    }
    islands_0[i_3].info_0[i32(2)] = ((isl_1.info_0.z) & (u32(4294967287)));
    if(flag_0 == 2.0f)
    {
        differs_0 = u32(32);
    }
    else
    {
        if(flag_0 == 3.0f)
        {
            differs_0 = u32(65);
        }
        else
        {
            differs_0 = u32(16);
        }
    }
    status_0[_S15] = differs_0;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn cascade_settle(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_4 : u32 = id_1.x;
    if(i_4 >= (params_0.island_count_0))
    {
        return;
    }
    var s_3 : u32 = status_0[u32(16) + i_4];
    var isl_2 : Island_0;
    isl_2.range_0 = islands_0[i_4].range_0;
    isl_2.info_0 = islands_0[i_4].info_0;
    isl_2.com_0 = islands_0[i_4].com_0;
    isl_2.inertia0_0 = islands_0[i_4].inertia0_0;
    isl_2.inertia1_0 = islands_0[i_4].inertia1_0;
    isl_2.inertia2_0 = islands_0[i_4].inertia2_0;
    isl_2.inv0_0 = islands_0[i_4].inv0_0;
    isl_2.inv1_0 = islands_0[i_4].inv1_0;
    isl_2.inv2_0 = islands_0[i_4].inv2_0;
    isl_2.wcom_0 = islands_0[i_4].wcom_0;
    isl_2.winv0_0 = islands_0[i_4].winv0_0;
    isl_2.winv1_0 = islands_0[i_4].winv1_0;
    isl_2.winv2_0 = islands_0[i_4].winv2_0;
    isl_2.rotation_0 = islands_0[i_4].rotation_0;
    isl_2.position_0 = islands_0[i_4].position_0;
    isl_2.position_err_0 = islands_0[i_4].position_err_0;
    isl_2.velocity_0 = islands_0[i_4].velocity_0;
    isl_2.velocity_err_0 = islands_0[i_4].velocity_err_0;
    isl_2.angular_velocity_0 = islands_0[i_4].angular_velocity_0;
    isl_2.done_0 = islands_0[i_4].done_0;
    isl_2.coverage_0 = islands_0[i_4].coverage_0;
    isl_2.wide_groups_0 = islands_0[i_4].wide_groups_0;
    isl_2.done_pad_0 = islands_0[i_4].done_pad_0;
    isl_2.probes_0 = islands_0[i_4].probes_0;
    isl_2.energy_0 = islands_0[i_4].energy_0;
    isl_2.rotation_err_0 = islands_0[i_4].rotation_err_0;
    isl_2.momentum_0 = islands_0[i_4].momentum_0;
    isl_2.momentum_err_0 = islands_0[i_4].momentum_err_0;
    isl_2.info_0[i32(0)] = ((isl_2.info_0[i32(0)]) & (u32(4294967279)));
    isl_2.info_0[i32(2)] = ((isl_2.info_0[i32(2)]) & (u32(4294967247)));
    var _S19 : bool;
    if(status_0[u32(0)] != u32(0))
    {
        if(((s_3 & (u32(3)))) == u32(3))
        {
            _S19 = true;
        }
        else
        {
            if((params_0.unconverged_0) == u32(2))
            {
                _S19 = ((s_3 & (u32(1)))) != u32(0);
            }
            else
            {
                _S19 = false;
            }
        }
    }
    else
    {
        _S19 = false;
    }
    if(_S19)
    {
        isl_2.info_0[i32(0)] = ((isl_2.info_0[i32(0)]) | (u32(16)));
    }
    islands_0[i_4].range_0 = isl_2.range_0;
    islands_0[i_4].info_0 = isl_2.info_0;
    islands_0[i_4].com_0 = isl_2.com_0;
    islands_0[i_4].inertia0_0 = isl_2.inertia0_0;
    islands_0[i_4].inertia1_0 = isl_2.inertia1_0;
    islands_0[i_4].inertia2_0 = isl_2.inertia2_0;
    islands_0[i_4].inv0_0 = isl_2.inv0_0;
    islands_0[i_4].inv1_0 = isl_2.inv1_0;
    islands_0[i_4].inv2_0 = isl_2.inv2_0;
    islands_0[i_4].wcom_0 = isl_2.wcom_0;
    islands_0[i_4].winv0_0 = isl_2.winv0_0;
    islands_0[i_4].winv1_0 = isl_2.winv1_0;
    islands_0[i_4].winv2_0 = isl_2.winv2_0;
    islands_0[i_4].rotation_0 = isl_2.rotation_0;
    islands_0[i_4].position_0 = isl_2.position_0;
    islands_0[i_4].position_err_0 = isl_2.position_err_0;
    islands_0[i_4].velocity_0 = isl_2.velocity_0;
    islands_0[i_4].velocity_err_0 = isl_2.velocity_err_0;
    islands_0[i_4].angular_velocity_0 = isl_2.angular_velocity_0;
    islands_0[i_4].done_0 = isl_2.done_0;
    islands_0[i_4].coverage_0 = isl_2.coverage_0;
    islands_0[i_4].wide_groups_0 = isl_2.wide_groups_0;
    islands_0[i_4].done_pad_0 = isl_2.done_pad_0;
    islands_0[i_4].probes_0 = isl_2.probes_0;
    islands_0[i_4].energy_0 = isl_2.energy_0;
    islands_0[i_4].rotation_err_0 = isl_2.rotation_err_0;
    islands_0[i_4].momentum_0 = isl_2.momentum_0;
    islands_0[i_4].momentum_err_0 = isl_2.momentum_err_0;
    return;
}

var<workgroup> g_damaged_0 : array<u32, i32(256)>;

var<workgroup> g_disconnected_0 : array<u32, i32(256)>;

var<workgroup> g_unconverged_0 : array<u32, i32(256)>;

@compute
@workgroup_size(256, 1, 1)
fn cascade_end(@builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var unconverged_1 : u32;
    var tid_0 : u32 = thread_3.x;
    if(status_0[u32(0)] == u32(0))
    {
        return;
    }
    var damaged_0 : u32 = u32(0);
    var disconnected_0 : u32 = u32(0);
    var i_5 : u32 = tid_0;
    var unconverged_2 : u32 = u32(0);
    loop
    {
        if(i_5 < (params_0.island_count_0))
        {
        }
        else
        {
            break;
        }
        var _S20 : u32 = u32(16) + i_5;
        var s_4 : u32 = status_0[_S20];
        if(((status_0[_S20] & (u32(16)))) != u32(0))
        {
            unconverged_2 = unconverged_2 + u32(1);
            i_5 = i_5 + u32(256);
            continue;
        }
        if(((s_4 & (u32(1)))) == u32(0))
        {
            i_5 = i_5 + u32(256);
            continue;
        }
        if(((s_4 & (u32(2)))) == u32(0))
        {
            var _S21 : u32 = unconverged_2 + u32(1);
            if((params_0.unconverged_0) != u32(2))
            {
                unconverged_2 = _S21;
                i_5 = i_5 + u32(256);
                continue;
            }
            unconverged_1 = _S21;
        }
        else
        {
            unconverged_1 = unconverged_2;
        }
        var z_0 : u32 = islands_0[i_5].info_0.z;
        var damaged_1 : u32;
        var add_0 : u32;
        if(((z_0 & (u32(16)))) != u32(0))
        {
            damaged_1 = u32(1);
            add_0 = u32(4);
        }
        else
        {
            damaged_1 = damaged_0;
            add_0 = u32(0);
        }
        var disconnected_1 : u32;
        var add_1 : u32;
        if(((z_0 & (u32(32)))) != u32(0))
        {
            var add_2 : u32 = (add_0 | (u32(8)));
            disconnected_1 = u32(1);
            add_1 = add_2;
        }
        else
        {
            disconnected_1 = disconnected_0;
            add_1 = add_0;
        }
        if(add_1 != u32(0))
        {
            gate_0[params_0.gate_flag_base_0 + i_5] = vec4<f32>(0.0f);
        }
        status_0[_S20] = (s_4 | (add_1));
        damaged_0 = damaged_1;
        disconnected_0 = disconnected_1;
        unconverged_2 = unconverged_1;
        i_5 = i_5 + u32(256);
    }
    g_damaged_0[tid_0] = damaged_0;
    g_disconnected_0[tid_0] = disconnected_0;
    g_unconverged_0[tid_0] = unconverged_2;
    workgroupBarrier();
    unconverged_1 = u32(128);
    loop
    {
        if(unconverged_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < unconverged_1)
        {
            var _S22 : u32 = tid_0 + unconverged_1;
            g_damaged_0[tid_0] = ((g_damaged_0[tid_0]) | ((g_damaged_0[_S22])));
            g_disconnected_0[tid_0] = ((g_disconnected_0[tid_0]) | ((g_disconnected_0[_S22])));
            g_unconverged_0[tid_0] = g_unconverged_0[tid_0] + g_unconverged_0[_S22];
        }
        workgroupBarrier();
        unconverged_1 = (unconverged_1 >> (u32(1)));
    }
    if(tid_0 != u32(0))
    {
        return;
    }
    status_0[u32(2)] = status_0[u32(2)] + u32(1);
    status_0[u32(3)] = status_0[u32(3)] + g_unconverged_0[i32(0)];
    status_0[u32(4)] = g_damaged_0[i32(0)];
    if((g_disconnected_0[i32(0)]) != u32(0))
    {
        status_0[u32(1)] = params_0.pass_index_0 + u32(1);
    }
    var _S23 : bool;
    if((g_damaged_0[i32(0)]) != u32(0))
    {
        _S23 = (g_disconnected_0[i32(0)]) == u32(0);
    }
    else
    {
        _S23 = false;
    }
    if(_S23)
    {
        damaged_0 = u32(1);
    }
    else
    {
        damaged_0 = u32(0);
    }
    status_0[u32(0)] = damaged_0;
    return;
}

