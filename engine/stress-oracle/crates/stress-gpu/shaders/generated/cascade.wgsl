struct CascadeParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) result_base_0 : u32,
    @align(8) pass_index_0 : u32,
    @align(4) pad_0 : u32,
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

@binding(3) @group(0) var<storage, read_write> state_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> backup_0 : array<vec4<f32>>;

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
    var i_1 : u32 = group_0.x;
    if(i_1 >= (params_0.island_count_0))
    {
        return;
    }
    var _S3 : u32 = u32(16) + i_1;
    var s_0 : u32 = status_0[_S3];
    var _S4 : bool;
    if(status_0[u32(0)] == u32(0))
    {
        _S4 = true;
    }
    else
    {
        _S4 = ((s_0 & (u32(1)))) == u32(0);
    }
    if(_S4)
    {
        return;
    }
    if((scratch_0[params_0.result_base_0 + i_1].w) != 0.0f)
    {
        if((thread_0.x) == u32(0))
        {
            status_0[_S3] = (s_0 | (u32(2)));
        }
        return;
    }
    var _S5 : vec4<u32> = islands_0[i_1].range_0;
    var k_0 : u32 = u32(4) * islands_0[i_1].range_0.x + thread_0.x;
    loop
    {
        if(k_0 < (u32(4) * _S5.y))
        {
        }
        else
        {
            break;
        }
        state_0[k_0] = backup_0[k_0];
        k_0 = k_0 + u32(64);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn cascade_settle(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_2 : u32 = id_1.x;
    if(i_2 >= (params_0.island_count_0))
    {
        return;
    }
    var s_1 : u32 = status_0[u32(16) + i_2];
    var isl_1 : Island_0;
    isl_1.range_0 = islands_0[i_2].range_0;
    isl_1.info_0 = islands_0[i_2].info_0;
    isl_1.com_0 = islands_0[i_2].com_0;
    isl_1.inertia0_0 = islands_0[i_2].inertia0_0;
    isl_1.inertia1_0 = islands_0[i_2].inertia1_0;
    isl_1.inertia2_0 = islands_0[i_2].inertia2_0;
    isl_1.inv0_0 = islands_0[i_2].inv0_0;
    isl_1.inv1_0 = islands_0[i_2].inv1_0;
    isl_1.inv2_0 = islands_0[i_2].inv2_0;
    isl_1.wcom_0 = islands_0[i_2].wcom_0;
    isl_1.winv0_0 = islands_0[i_2].winv0_0;
    isl_1.winv1_0 = islands_0[i_2].winv1_0;
    isl_1.winv2_0 = islands_0[i_2].winv2_0;
    isl_1.rotation_0 = islands_0[i_2].rotation_0;
    isl_1.position_0 = islands_0[i_2].position_0;
    isl_1.position_err_0 = islands_0[i_2].position_err_0;
    isl_1.velocity_0 = islands_0[i_2].velocity_0;
    isl_1.velocity_err_0 = islands_0[i_2].velocity_err_0;
    isl_1.angular_velocity_0 = islands_0[i_2].angular_velocity_0;
    isl_1.done_0 = islands_0[i_2].done_0;
    isl_1.coverage_0 = islands_0[i_2].coverage_0;
    isl_1.wide_groups_0 = islands_0[i_2].wide_groups_0;
    isl_1.done_pad_0 = islands_0[i_2].done_pad_0;
    isl_1.probes_0 = islands_0[i_2].probes_0;
    isl_1.energy_0 = islands_0[i_2].energy_0;
    isl_1.rotation_err_0 = islands_0[i_2].rotation_err_0;
    isl_1.momentum_0 = islands_0[i_2].momentum_0;
    isl_1.momentum_err_0 = islands_0[i_2].momentum_err_0;
    isl_1.info_0[i32(0)] = ((isl_1.info_0[i32(0)]) & (u32(4294967279)));
    isl_1.info_0[i32(2)] = ((isl_1.info_0[i32(2)]) & (u32(4294967247)));
    var _S6 : bool;
    if(status_0[u32(0)] != u32(0))
    {
        _S6 = ((s_1 & (u32(3)))) == u32(3);
    }
    else
    {
        _S6 = false;
    }
    if(_S6)
    {
        isl_1.info_0[i32(0)] = ((isl_1.info_0[i32(0)]) | (u32(16)));
    }
    islands_0[i_2].range_0 = isl_1.range_0;
    islands_0[i_2].info_0 = isl_1.info_0;
    islands_0[i_2].com_0 = isl_1.com_0;
    islands_0[i_2].inertia0_0 = isl_1.inertia0_0;
    islands_0[i_2].inertia1_0 = isl_1.inertia1_0;
    islands_0[i_2].inertia2_0 = isl_1.inertia2_0;
    islands_0[i_2].inv0_0 = isl_1.inv0_0;
    islands_0[i_2].inv1_0 = isl_1.inv1_0;
    islands_0[i_2].inv2_0 = isl_1.inv2_0;
    islands_0[i_2].wcom_0 = isl_1.wcom_0;
    islands_0[i_2].winv0_0 = isl_1.winv0_0;
    islands_0[i_2].winv1_0 = isl_1.winv1_0;
    islands_0[i_2].winv2_0 = isl_1.winv2_0;
    islands_0[i_2].rotation_0 = isl_1.rotation_0;
    islands_0[i_2].position_0 = isl_1.position_0;
    islands_0[i_2].position_err_0 = isl_1.position_err_0;
    islands_0[i_2].velocity_0 = isl_1.velocity_0;
    islands_0[i_2].velocity_err_0 = isl_1.velocity_err_0;
    islands_0[i_2].angular_velocity_0 = isl_1.angular_velocity_0;
    islands_0[i_2].done_0 = isl_1.done_0;
    islands_0[i_2].coverage_0 = isl_1.coverage_0;
    islands_0[i_2].wide_groups_0 = isl_1.wide_groups_0;
    islands_0[i_2].done_pad_0 = isl_1.done_pad_0;
    islands_0[i_2].probes_0 = isl_1.probes_0;
    islands_0[i_2].energy_0 = isl_1.energy_0;
    islands_0[i_2].rotation_err_0 = isl_1.rotation_err_0;
    islands_0[i_2].momentum_0 = isl_1.momentum_0;
    islands_0[i_2].momentum_err_0 = isl_1.momentum_err_0;
    return;
}

var<workgroup> g_damaged_0 : array<u32, i32(256)>;

var<workgroup> g_disconnected_0 : array<u32, i32(256)>;

var<workgroup> g_unconverged_0 : array<u32, i32(256)>;

@compute
@workgroup_size(256, 1, 1)
fn cascade_end(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var damaged_0 : u32;
    var tid_0 : u32 = thread_1.x;
    if(status_0[u32(0)] == u32(0))
    {
        return;
    }
    var damaged_1 : u32 = u32(0);
    var disconnected_0 : u32 = u32(0);
    var i_3 : u32 = tid_0;
    var unconverged_0 : u32 = u32(0);
    loop
    {
        if(i_3 < (params_0.island_count_0))
        {
        }
        else
        {
            break;
        }
        var _S7 : u32 = u32(16) + i_3;
        var s_2 : u32 = status_0[_S7];
        if(((status_0[_S7] & (u32(1)))) == u32(0))
        {
            i_3 = i_3 + u32(256);
            continue;
        }
        if(((s_2 & (u32(2)))) == u32(0))
        {
            unconverged_0 = unconverged_0 + u32(1);
            i_3 = i_3 + u32(256);
            continue;
        }
        var z_0 : u32 = islands_0[i_3].info_0.z;
        var add_0 : u32;
        if(((z_0 & (u32(16)))) != u32(0))
        {
            damaged_0 = u32(1);
            add_0 = u32(4);
        }
        else
        {
            damaged_0 = damaged_1;
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
        status_0[_S7] = (s_2 | (add_1));
        damaged_1 = damaged_0;
        disconnected_0 = disconnected_1;
        i_3 = i_3 + u32(256);
    }
    g_damaged_0[tid_0] = damaged_1;
    g_disconnected_0[tid_0] = disconnected_0;
    g_unconverged_0[tid_0] = unconverged_0;
    workgroupBarrier();
    damaged_0 = u32(128);
    loop
    {
        if(damaged_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < damaged_0)
        {
            var _S8 : u32 = tid_0 + damaged_0;
            g_damaged_0[tid_0] = ((g_damaged_0[tid_0]) | ((g_damaged_0[_S8])));
            g_disconnected_0[tid_0] = ((g_disconnected_0[tid_0]) | ((g_disconnected_0[_S8])));
            g_unconverged_0[tid_0] = g_unconverged_0[tid_0] + g_unconverged_0[_S8];
        }
        workgroupBarrier();
        damaged_0 = (damaged_0 >> (u32(1)));
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
    var _S9 : bool;
    if((g_damaged_0[i32(0)]) != u32(0))
    {
        _S9 = (g_disconnected_0[i32(0)]) == u32(0);
    }
    else
    {
        _S9 = false;
    }
    if(_S9)
    {
        damaged_1 = u32(1);
    }
    else
    {
        damaged_1 = u32(0);
    }
    status_0[u32(0)] = damaged_1;
    return;
}

