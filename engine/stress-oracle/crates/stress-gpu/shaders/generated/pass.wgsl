@binding(9) @group(0) var<storage, read_write> counters_0 : array<u32>;

@binding(1) @group(0) var<storage, read_write> status_0 : array<u32>;

struct PassParams_std140_0
{
    @align(16) chunk_words_0 : u32,
    @align(4) bond_words_0 : u32,
    @align(8) dyn_words_0 : u32,
    @align(4) island_words_0 : u32,
    @align(16) props_base_0 : u32,
    @align(4) eload_base_0 : u32,
    @align(8) pulse_base_0 : u32,
    @align(4) pulse_count_base_0 : u32,
    @align(16) pulse_cap_0 : u32,
    @align(4) engine_base_0 : u32,
    @align(8) resident_0 : u32,
    @align(4) resume_0 : u32,
    @align(16) island_float4s_0 : u32,
    @align(4) index_words_0 : u32,
    @align(8) engine_float4s_0 : u32,
    @align(4) copy_float4s_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : PassParams_std140_0;
@binding(2) @group(0) var<storage, read> islands_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read_write> parents_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> ctab_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> islands_of_0 : array<vec4<u32>>;

@binding(7) @group(0) var<storage, read_write> parent_bodies_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read_write> control_0 : array<u32>;

@binding(8) @group(0) var<storage, read_write> args_0 : array<vec4<u32>>;

@binding(22) @group(0) var<storage, read_write> old_islands_0 : array<vec4<f32>>;

@binding(21) @group(0) var<storage, read_write> old_index_0 : array<u32>;

@binding(20) @group(0) var<storage, read> index_0 : array<u32>;

@binding(18) @group(0) var<storage, read_write> old_ctab_0 : array<vec4<f32>>;

@binding(24) @group(0) var<storage, read_write> old_copy_0 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read> island_copy_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read_write> old_chunks_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> chunks_0 : array<vec4<f32>>;

@binding(13) @group(0) var<storage, read_write> old_state_0 : array<vec4<f32>>;

@binding(12) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(15) @group(0) var<storage, read_write> old_bonds_0 : array<vec4<f32>>;

@binding(14) @group(0) var<storage, read> bonds_0 : array<vec4<f32>>;

@binding(17) @group(0) var<storage, read_write> old_dyn_0 : array<vec4<f32>>;

@binding(16) @group(0) var<storage, read> bond_dyn_0 : array<vec4<f32>>;

@binding(19) @group(0) var<storage, read_write> island_args_0 : array<vec4<u32>>;

fn island_range_0( i_0 : u32) -> vec4<u32>
{
    return (bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * i_0])));
}

var<workgroup> g_parents_0 : array<u32, i32(1024)>;

var<workgroup> g_chunks_0 : array<u32, i32(1024)>;

@compute
@workgroup_size(1024, 1, 1)
fn pass_parents(@builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var tid_0 : u32 = thread_0.x;
    var n_0 : u32 = counters_0[i32(0)];
    var _S1 : bool = status_0[u32(1)] != u32(0);
    var base_0 : u32 = u32(0);
    var carry_p_0 : u32 = u32(0);
    var carry_c_0 : u32 = u32(0);
    loop
    {
        if(base_0 < n_0)
        {
        }
        else
        {
            break;
        }
        var i_1 : u32 = base_0 + tid_0;
        var _S2 : bool;
        if(_S1)
        {
            _S2 = i_1 < n_0;
        }
        else
        {
            _S2 = false;
        }
        var parent_0 : bool;
        if(_S2)
        {
            parent_0 = ((status_0[u32(16) + i_1] & (u32(8)))) != u32(0);
        }
        else
        {
            parent_0 = false;
        }
        var r_0 : vec4<u32>;
        if(parent_0)
        {
            r_0 = island_range_0(i_1);
        }
        else
        {
            r_0 = vec4<u32>(vec4<i32>(i32(0)));
        }
        var fp_0 : u32;
        if(parent_0)
        {
            fp_0 = u32(1);
        }
        else
        {
            fp_0 = u32(0);
        }
        var fc_0 : u32;
        if(parent_0)
        {
            fc_0 = r_0.y - r_0.x;
        }
        else
        {
            fc_0 = u32(0);
        }
        g_parents_0[tid_0] = fp_0;
        g_chunks_0[tid_0] = fc_0;
        workgroupBarrier();
        var step_0 : u32 = u32(1);
        loop
        {
            if(step_0 < u32(1024))
            {
            }
            else
            {
                break;
            }
            var _S3 : bool = tid_0 >= step_0;
            var ap_0 : u32;
            if(_S3)
            {
                ap_0 = g_parents_0[tid_0 - step_0];
            }
            else
            {
                ap_0 = u32(0);
            }
            var ac_0 : u32;
            if(_S3)
            {
                ac_0 = g_chunks_0[tid_0 - step_0];
            }
            else
            {
                ac_0 = u32(0);
            }
            workgroupBarrier();
            g_parents_0[tid_0] = g_parents_0[tid_0] + ap_0;
            g_chunks_0[tid_0] = g_chunks_0[tid_0] + ac_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        if(parent_0)
        {
            var p_0 : u32 = carry_p_0 + g_parents_0[tid_0] - fp_0;
            parents_0[p_0] = vec4<u32>(i_1, carry_c_0 + g_chunks_0[tid_0] - fc_0, u32(0), u32(0));
            var _S4 : u32 = u32(3) * i_1;
            var e0_0 : vec4<f32> = ctab_0[params_0.engine_base_0 + _S4];
            var e1_0 : vec4<f32> = ctab_0[params_0.engine_base_0 + _S4 + u32(1)];
            var e2_0 : vec4<f32> = ctab_0[params_0.engine_base_0 + _S4 + u32(2)];
            var body_0 : vec4<u32> = (bitcast<vec4<u32>>((e0_0)));
            islands_of_0[p_0] = vec4<u32>(r_0.x, body_0.y, body_0.z, body_0.w);
            var _S5 : u32 = u32(2) * p_0;
            parent_bodies_0[_S5] = vec4<f32>(e0_0.x, e2_0.xyz);
            parent_bodies_0[_S5 + u32(1)] = vec4<f32>(- e1_0.x, - e1_0.y, - e1_0.z, e1_0.w);
        }
        var tp_0 : u32 = g_parents_0[i32(1023)];
        var tc_0 : u32 = g_chunks_0[i32(1023)];
        workgroupBarrier();
        var carry_p_1 : u32 = carry_p_0 + tp_0;
        var carry_c_1 : u32 = carry_c_0 + tc_0;
        base_0 = base_0 + u32(1024);
        carry_p_0 = carry_p_1;
        carry_c_0 = carry_c_1;
    }
    if(tid_0 != u32(0))
    {
        return;
    }
    parents_0[carry_p_0] = vec4<u32>(u32(0), carry_c_0, u32(0), u32(0));
    control_0[u32(0)] = u32(0);
    control_0[u32(1)] = carry_p_0;
    control_0[u32(2)] = n_0;
    control_0[u32(3)] = counters_0[i32(3)];
    control_0[u32(4)] = counters_0[i32(1)];
    control_0[u32(5)] = counters_0[i32(2)];
    control_0[u32(6)] = carry_c_0;
    args_0[i32(0)] = vec4<u32>(carry_p_0, u32(1), u32(1), u32(0));
    args_0[i32(1)] = vec4<u32>((carry_c_0 + u32(63)) / u32(64), u32(1), u32(1), u32(0));
    args_0[i32(2)] = vec4<u32>((carry_p_0 + u32(63)) / u32(64), u32(1), u32(1), u32(0));
    if(carry_p_0 > u32(0))
    {
        base_0 = u32(256);
    }
    else
    {
        base_0 = u32(0);
    }
    args_0[i32(3)] = vec4<u32>(base_0, u32(1), u32(1), u32(0));
    args_0[i32(4)] = vec4<u32>(carry_c_0, u32(1), u32(1), u32(0));
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn pass_whole(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S6 : u32 = id_0.x;
    var k_0 : u32 = _S6;
    loop
    {
        if(k_0 < (params_0.island_float4s_0))
        {
        }
        else
        {
            break;
        }
        old_islands_0[k_0] = islands_0[k_0];
        k_0 = k_0 + u32(65536);
    }
    k_0 = _S6;
    loop
    {
        if(k_0 < (params_0.index_words_0))
        {
        }
        else
        {
            break;
        }
        old_index_0[k_0] = index_0[k_0];
        k_0 = k_0 + u32(65536);
    }
    if((params_0.resident_0) != u32(0))
    {
        k_0 = _S6;
        loop
        {
            if(k_0 < (params_0.engine_float4s_0))
            {
            }
            else
            {
                break;
            }
            old_ctab_0[params_0.engine_base_0 + k_0] = ctab_0[params_0.engine_base_0 + k_0];
            k_0 = k_0 + u32(65536);
        }
    }
    k_0 = _S6;
    loop
    {
        if(k_0 < (params_0.copy_float4s_0))
        {
        }
        else
        {
            break;
        }
        old_copy_0[k_0] = island_copy_0[k_0];
        k_0 = k_0 + u32(65536);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn pass_snapshot(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var _S7 : u32;
    var _S8 : u32;
    var p_1 : u32 = group_0.x;
    if(p_1 >= control_0[u32(1)])
    {
        return;
    }
    var r_1 : vec4<u32> = island_range_0(parents_0[p_1].x);
    var t_0 : u32 = thread_1.x;
    var _S9 : u32 = r_1.x;
    var k_1 : u32 = params_0.chunk_words_0 * _S9 + t_0;
    loop
    {
        var _S10 : u32 = r_1.y;
        _S7 = _S10;
        if(k_1 < (params_0.chunk_words_0 * _S10))
        {
        }
        else
        {
            break;
        }
        old_chunks_0[k_1] = chunks_0[k_1];
        k_1 = k_1 + u32(256);
    }
    k_1 = u32(4) * _S9 + t_0;
    loop
    {
        if(k_1 < (u32(4) * _S7))
        {
        }
        else
        {
            break;
        }
        old_state_0[k_1] = state_0[k_1];
        k_1 = k_1 + u32(256);
    }
    var _S11 : u32 = r_1.z;
    k_1 = params_0.bond_words_0 * _S11 + t_0;
    loop
    {
        var _S12 : u32 = r_1.w;
        _S8 = _S12;
        if(k_1 < (params_0.bond_words_0 * _S12))
        {
        }
        else
        {
            break;
        }
        old_bonds_0[k_1] = bonds_0[k_1];
        k_1 = k_1 + u32(256);
    }
    k_1 = params_0.dyn_words_0 * _S11 + t_0;
    loop
    {
        if(k_1 < (params_0.dyn_words_0 * _S8))
        {
        }
        else
        {
            break;
        }
        old_dyn_0[k_1] = bond_dyn_0[k_1];
        k_1 = k_1 + u32(256);
    }
    if((params_0.resident_0) == u32(0))
    {
        return;
    }
    var c_0 : u32 = _S9 + t_0;
    loop
    {
        if(c_0 < _S7)
        {
        }
        else
        {
            break;
        }
        old_ctab_0[params_0.props_base_0 + c_0] = ctab_0[params_0.props_base_0 + c_0];
        var _S13 : u32 = u32(2) * c_0;
        old_ctab_0[params_0.eload_base_0 + _S13] = ctab_0[params_0.eload_base_0 + _S13];
        old_ctab_0[params_0.eload_base_0 + _S13 + u32(1)] = ctab_0[params_0.eload_base_0 + _S13 + u32(1)];
        old_ctab_0[params_0.pulse_count_base_0 + c_0] = ctab_0[params_0.pulse_count_base_0 + c_0];
        c_0 = c_0 + u32(256);
    }
    k_1 = u32(3) * params_0.pulse_cap_0 * _S9 + t_0;
    loop
    {
        if(k_1 < (u32(3) * params_0.pulse_cap_0 * _S7))
        {
        }
        else
        {
            break;
        }
        old_ctab_0[params_0.pulse_base_0 + k_1] = ctab_0[params_0.pulse_base_0 + k_1];
        k_1 = k_1 + u32(256);
    }
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn pass_finish(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S14 : bool;
    if(control_0[u32(1)] == u32(0))
    {
        _S14 = true;
    }
    else
    {
        _S14 = control_0[u32(0)] != u32(0);
    }
    if(_S14)
    {
        return;
    }
    var live_0 : u32 = control_0[u32(7)];
    counters_0[i32(0)] = control_0[u32(7)];
    counters_0[i32(1)] = control_0[u32(8)];
    counters_0[i32(2)] = control_0[u32(9)];
    counters_0[i32(3)] = control_0[u32(10)];
    var _S15 : u32 = max(live_0, u32(1));
    island_args_0[i32(0)] = vec4<u32>(_S15, u32(1), u32(1), u32(0));
    island_args_0[i32(1)] = vec4<u32>((_S15 + u32(63)) / u32(64), u32(1), u32(1), u32(0));
    if((params_0.resume_0) != u32(0))
    {
        status_0[u32(0)] = u32(1);
        status_0[u32(1)] = u32(0);
    }
    return;
}

