struct BlastParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) max_iterations_0 : u32,
    @align(8) warm_0 : u32,
    @align(4) skip_settled_0 : u32,
    @align(16) tolerance_0 : f32,
    @align(4) recip_length_0 : f32,
    @align(8) node_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(16) linear_scale_0 : f32,
    @align(4) angular_scale_0 : f32,
    @align(8) bend_gain_max_0 : f32,
    @align(4) fibres_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) damage_rate_0 : f32,
    @align(8) apply_damage_0 : u32,
    @align(4) force_tolerance_0 : f32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) w_alive_0 : u32,
    @align(4) a_dirty_0 : u32,
    @align(8) a_counters_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(1) @group(0) var<storage, read> islands_0 : array<vec4<u32>>;

@binding(8) @group(0) var<storage, read_write> report_0 : array<vec4<u32>>;

@binding(9) @group(0) var<storage, read> lists_0 : array<u32>;

@binding(5) @group(0) var<storage, read> accel_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

struct BlastNode_std430_0
{
    @align(4) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(4) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
};

@binding(2) @group(0) var<storage, read> nodes_0 : array<BlastNode_std430_0>;

@binding(6) @group(0) var<storage, read_write> x_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> refs_0 : array<u32>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(4) @group(0) var<storage, read_write> bonds_0 : array<BlastBond_std430_0>;

fn isnan_0( x_1 : f32) -> bool
{
    var _S1 : u32 = (bitcast<u32>((x_1)));
    var _S2 : u32 = (_S1 & (u32(8388607)));
    var _S3 : bool;
    if(((((_S1 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S3 = _S2 != u32(0);
    }
    else
    {
        _S3 = false;
    }
    return _S3;
}

fn isinf_0( x_2 : f32) -> bool
{
    var _S4 : u32 = (bitcast<u32>((x_2)));
    var _S5 : u32 = (_S4 & (u32(8388607)));
    var _S6 : bool;
    if(((((_S4 >> (u32(23)))) & (u32(255)))) == u32(255))
    {
        _S6 = _S5 == u32(0);
    }
    else
    {
        _S6 = false;
    }
    return _S6;
}

fn isfinite_0( x_3 : f32) -> bool
{
    var _S7 : bool;
    if(isinf_0(x_3))
    {
        _S7 = true;
    }
    else
    {
        _S7 = isnan_0(x_3);
    }
    return !_S7;
}

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_0 : u32,  thread_0 : u32) -> u32
{
    if(thread_0 == u32(0))
    {
        g_group_uniform_0 = value_0;
    }
    var _S8 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S8;
}

fn node_slot_0( node_0 : u32,  vector_0 : u32) -> u32
{
    return node_0 * u32(10) + vector_0;
}

var<workgroup> g_blast_red_0 : array<vec2<f32>, i32(1024)>;

var<workgroup> g_blast_sum_0 : vec2<f32>;

fn blast_group_sum_0( tid_0 : u32,  v_0 : vec2<f32>) -> vec2<f32>
{
    g_blast_red_0[tid_0] = v_0;
    workgroupBarrier();
    var k_0 : u32 = u32(512);
    loop
    {
        if(k_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < k_0)
        {
            g_blast_red_0[tid_0] = g_blast_red_0[tid_0] + g_blast_red_0[tid_0 + k_0];
        }
        workgroupBarrier();
        k_0 = (k_0 >> (u32(1)));
    }
    if(tid_0 == u32(0))
    {
        g_blast_sum_0 = g_blast_red_0[i32(0)];
    }
    var s_1 : vec2<f32> = (workgroupUniformLoad(&((g_blast_sum_0))));
    workgroupBarrier();
    return s_1;
}

fn norm2_0( a_0 : vec3<f32>,  b_0 : vec3<f32>) -> f32
{
    return dot(a_0, a_0) + dot(b_0, b_0);
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn bond_slot_0( node_count_1 : u32,  bond_0 : u32) -> u32
{
    return node_count_1 * u32(10) + bond_0 * u32(2);
}

fn gather_0( node_1 : u32,  from_x_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_0 : BlastNode_std430_0 = nodes_0[node_1];
    var _S9 : vec3<f32> = vec3<f32>(0.0f);
    var r_0 : u32 = n_0.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S9;
    var acc_l_0 : vec3<f32> = _S9;
    loop
    {
        if(r_0 < (n_0.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_0 : u32 = refs_0[r_0];
        var j_0 : u32 = (ref_0 & (u32(2147483647)));
        var b_1 : BlastBond_std430_0 = bonds_0[j_0];
        if(!((bonds_0[j_0].health_0) > 0.0f))
        {
            r_0 = r_0 + u32(1);
            continue;
        }
        var yl_0 : vec3<f32>;
        var ya_0 : vec3<f32>;
        if(from_x_0)
        {
            var _S10 : u32 = j_0 * u32(2);
            var _S11 : vec3<f32> = vec3<f32>(b_1.s_0);
            var _S12 : vec3<f32> = x_0[_S10].xyz * _S11;
            yl_0 = x_0[_S10 + u32(1)].xyz * _S11;
            ya_0 = _S12;
        }
        else
        {
            var t_0 : u32 = bond_slot_0(params_0.node_count_0, j_0);
            var _S13 : vec3<f32> = vec3<f32>(b_1.s_0);
            var _S14 : vec3<f32> = scratch_0[t_0].xyz * _S13;
            yl_0 = scratch_0[t_0 + u32(1)].xyz * _S13;
            ya_0 = _S14;
        }
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_0 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_0;
            acc_a_1 = acc_a_0 + (cross(b_1.o1_0.xyz, yl_0) - ya_0);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_0;
            acc_a_1 = acc_a_0 + (ya_0 - cross(b_1.o0_0.xyz, yl_0));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_0 = r_0 + u32(1);
    }
    (*ang_0) = acc_a_0 * vec3<f32>(n_0.w_ang_0);
    (*lin_0) = acc_l_0 * vec3<f32>(n_0.w_lin_0);
    return;
}

fn bond_term_0( b_2 : ptr<function, BlastBond_std430_0>,  vec_0 : u32,  ang_1 : ptr<function, vec3<f32>>,  lin_1 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_2).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_2).n1_0];
    var _S15 : u32 = node_slot_0((*b_2).n0_0, vec_0);
    var a0_0 : vec3<f32> = scratch_0[_S15].xyz * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = scratch_0[_S15 + u32(1)].xyz * vec3<f32>(m0_0.w_lin_0);
    var _S16 : u32 = node_slot_0((*b_2).n1_0, vec_0);
    var a1_0 : vec3<f32> = scratch_0[_S16].xyz * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = scratch_0[_S16 + u32(1)].xyz * vec3<f32>(m1_0.w_lin_0);
    var _S17 : vec3<f32> = vec3<f32>((*b_2).s_0);
    (*ang_1) = (a0_0 - a1_0) * _S17;
    (*lin_1) = (l0_0 - l1_0 + (cross((*b_2).o0_0.xyz, a0_0) - cross((*b_2).o1_0.xyz, a1_0))) * _S17;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn blast_solve(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var ia_0 : f32;
    var il_0 : f32;
    var v_1 : u32;
    var q_0 : u32;
    var tid_1 : u32 = thread_1.x;
    var k_1 : u32 = group_0.x;
    if(k_1 >= (params_0.island_count_0))
    {
        return;
    }
    var range_0 : vec4<u32> = islands_0[k_1];
    var n0_1 : u32 = range_0.x;
    var _S18 : u32 = range_0.y;
    var b0_0 : u32 = range_0.z;
    var _S19 : u32 = range_0.w;
    var last_0 : vec4<u32> = report_0[k_1];
    var last_flags_0 : u32 = group_uniform_0(report_0[k_1].y, tid_1);
    var warm_1 : bool = (params_0.warm_0) != u32(0);
    var converged_0 : bool;
    if((params_0.skip_settled_0) != u32(0))
    {
        converged_0 = warm_1;
    }
    else
    {
        converged_0 = false;
    }
    if(converged_0)
    {
        converged_0 = ((last_flags_0 & (u32(4)))) != u32(0);
    }
    else
    {
        converged_0 = false;
    }
    var target_0 : f32;
    var p_0 : u32;
    if(converged_0)
    {
        var _S20 : u32 = n0_1 + tid_1;
        target_0 = 0.0f;
        p_0 = _S20;
        loop
        {
            if(p_0 < _S18)
            {
            }
            else
            {
                break;
            }
            var i_0 : u32 = lists_0[p_0];
            var _S21 : u32 = i_0 * u32(2);
            var _S22 : u32 = node_slot_0(i_0, u32(8));
            if((any((accel_0[_S21] != scratch_0[_S22]))))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = (any((accel_0[_S21 + u32(1)] != scratch_0[_S22 + u32(1)])));
            }
            if(converged_0)
            {
                target_0 = 1.0f;
            }
            p_0 = p_0 + u32(1024);
        }
        var _S23 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(target_0, 0.0f));
        if((_S23.x) == 0.0f)
        {
            if(tid_1 == u32(0))
            {
                report_0[k_1] = vec4<u32>(u32(0), u32(6), last_0.z, last_0.w);
            }
            return;
        }
    }
    var _S24 : u32 = n0_1 + tid_1;
    p_0 = _S24;
    target_0 = 0.0f;
    loop
    {
        if(p_0 < _S18)
        {
        }
        else
        {
            break;
        }
        var i_1 : u32 = lists_0[p_0];
        var n_1 : BlastNode_std430_0 = nodes_0[i_1];
        var _S25 : u32 = i_1 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S25];
        var a_lin_0 : vec4<f32> = accel_0[_S25 + u32(1)];
        if((n_1.w_ang_0) > 0.0f)
        {
            ia_0 = 1.0f / n_1.w_ang_0;
        }
        else
        {
            ia_0 = 1.0f;
        }
        if((n_1.w_lin_0) > 0.0f)
        {
            il_0 = params_0.recip_length_0 / n_1.w_lin_0;
        }
        else
        {
            il_0 = params_0.recip_length_0;
        }
        var b_ang_0 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz) * vec3<f32>(ia_0);
        var b_lin_0 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz) * vec3<f32>(il_0);
        var target_1 : f32 = target_0 + norm2_0(b_ang_0, b_lin_0);
        var _S26 : u32 = node_slot_0(i_1, u32(0));
        scratch_0[_S26] = vec4<f32>(b_ang_0, 0.0f);
        scratch_0[_S26 + u32(1)] = vec4<f32>(b_lin_0, 0.0f);
        var _S27 : u32 = node_slot_0(i_1, u32(8));
        scratch_0[_S27] = a_ang_0;
        scratch_0[_S27 + u32(1)] = a_lin_0;
        v_1 = u32(2);
        loop
        {
            if(v_1 < u32(8))
            {
            }
            else
            {
                break;
            }
            scratch_0[node_slot_0(i_1, v_1)] = vec4<f32>(0.0f);
            v_1 = v_1 + u32(1);
        }
        p_0 = p_0 + u32(1024);
        target_0 = target_1;
    }
    if(!warm_1)
    {
        v_1 = b0_0 + tid_1;
        loop
        {
            if(v_1 < _S19)
            {
            }
            else
            {
                break;
            }
            var _S28 : u32 = lists_0[v_1] * u32(2);
            var _S29 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S28] = _S29;
            x_0[_S28 + u32(1)] = _S29;
            v_1 = v_1 + u32(1024);
        }
    }
    device_group_barrier_0();
    if(warm_1)
    {
        p_0 = _S24;
        loop
        {
            if(p_0 < _S18)
            {
            }
            else
            {
                break;
            }
            var i_2 : u32 = lists_0[p_0];
            var ya_1 : vec3<f32>;
            var yl_1 : vec3<f32>;
            gather_0(i_2, true, &(ya_1), &(yl_1));
            var _S30 : u32 = node_slot_0(i_2, u32(0));
            scratch_0[_S30] = vec4<f32>(scratch_0[_S30].xyz - ya_1, 0.0f);
            var _S31 : u32 = _S30 + u32(1);
            scratch_0[_S31] = vec4<f32>(scratch_0[_S31].xyz - yl_1, 0.0f);
            p_0 = p_0 + u32(1024);
        }
    }
    var _S32 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(target_0, 0.0f));
    var target2_0 : f32 = _S32.x * params_0.tolerance_0 * params_0.tolerance_0;
    device_group_barrier_0();
    ia_0 = 0.0f;
    v_1 = u32(0);
    var iterations_0 : u32 = u32(0);
    loop
    {
        if(v_1 < (params_0.max_iterations_0))
        {
        }
        else
        {
            converged_0 = false;
            il_0 = ia_0;
            break;
        }
        q_0 = b0_0 + tid_1;
        il_0 = 0.0f;
        loop
        {
            if(q_0 < _S19)
            {
            }
            else
            {
                break;
            }
            var j_1 : u32 = lists_0[q_0];
            var _S33 : BlastBond_std430_0 = bonds_0[j_1];
            var _S34 : vec3<f32> = vec3<f32>(0.0f);
            var ta_0 : vec3<f32> = _S34;
            var tl_0 : vec3<f32> = _S34;
            if((_S33.health_0) > 0.0f)
            {
                bond_term_0(&(_S33), u32(0), &(ta_0), &(tl_0));
            }
            var zt_0 : f32 = il_0 + norm2_0(ta_0, tl_0);
            var t_1 : u32 = bond_slot_0(params_0.node_count_0, j_1);
            scratch_0[t_1] = vec4<f32>(ta_0, 0.0f);
            scratch_0[t_1 + u32(1)] = vec4<f32>(tl_0, 0.0f);
            q_0 = q_0 + u32(1024);
            il_0 = zt_0;
        }
        var _S35 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(il_0, 0.0f));
        var z_0 : f32 = _S35.x;
        if(z_0 <= target2_0)
        {
            converged_0 = true;
            il_0 = z_0;
            break;
        }
        device_group_barrier_0();
        if(v_1 != u32(0))
        {
            converged_0 = ia_0 > 0.0f;
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
            target_0 = z_0 / ia_0;
        }
        else
        {
            target_0 = 0.0f;
        }
        p_0 = _S24;
        var qq_0 : f32 = 0.0f;
        loop
        {
            if(p_0 < _S18)
            {
            }
            else
            {
                break;
            }
            var i_3 : u32 = lists_0[p_0];
            var wa_0 : vec3<f32>;
            var wl_0 : vec3<f32>;
            gather_0(i_3, false, &(wa_0), &(wl_0));
            var _S36 : u32 = node_slot_0(i_3, u32(0));
            var _S37 : u32 = node_slot_0(i_3, u32(2));
            var _S38 : u32 = _S37 + u32(1);
            var pl_0 : vec3<f32> = scratch_0[_S36 + u32(1)].xyz + scratch_0[_S38].xyz * vec3<f32>(target_0);
            var _S39 : u32 = node_slot_0(i_3, u32(4));
            var qa_0 : vec3<f32> = wa_0 + scratch_0[_S39].xyz * vec3<f32>(target_0);
            var _S40 : u32 = _S39 + u32(1);
            var ql_0 : vec3<f32> = wl_0 + scratch_0[_S40].xyz * vec3<f32>(target_0);
            scratch_0[_S37] = vec4<f32>(scratch_0[_S36].xyz + scratch_0[_S37].xyz * vec3<f32>(target_0), 0.0f);
            scratch_0[_S38] = vec4<f32>(pl_0, 0.0f);
            scratch_0[_S39] = vec4<f32>(qa_0, 0.0f);
            scratch_0[_S40] = vec4<f32>(ql_0, 0.0f);
            var qq_1 : f32 = qq_0 + norm2_0(qa_0, ql_0);
            p_0 = p_0 + u32(1024);
            qq_0 = qq_1;
        }
        var _S41 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(qq_0, 0.0f));
        var q2_0 : f32 = _S41.x;
        var _S42 : bool;
        if(!(q2_0 > 0.0f))
        {
            _S42 = true;
        }
        else
        {
            _S42 = !isfinite_0(q2_0);
        }
        if(_S42)
        {
            converged_0 = false;
            il_0 = z_0;
            break;
        }
        var _S43 : f32 = z_0 / q2_0;
        var p_1 : u32 = _S24;
        loop
        {
            if(p_1 < _S18)
            {
            }
            else
            {
                break;
            }
            var i_4 : u32 = lists_0[p_1];
            var _S44 : u32 = node_slot_0(i_4, u32(6));
            var _S45 : u32 = node_slot_0(i_4, u32(2));
            var _S46 : vec3<f32> = vec3<f32>(_S43);
            scratch_0[_S44] = vec4<f32>(scratch_0[_S44].xyz + scratch_0[_S45].xyz * _S46, 0.0f);
            var _S47 : u32 = _S44 + u32(1);
            scratch_0[_S47] = vec4<f32>(scratch_0[_S47].xyz + scratch_0[_S45 + u32(1)].xyz * _S46, 0.0f);
            var _S48 : u32 = node_slot_0(i_4, u32(0));
            var _S49 : u32 = node_slot_0(i_4, u32(4));
            scratch_0[_S48] = vec4<f32>(scratch_0[_S48].xyz - scratch_0[_S49].xyz * _S46, 0.0f);
            var _S50 : u32 = _S48 + u32(1);
            scratch_0[_S50] = vec4<f32>(scratch_0[_S50].xyz - scratch_0[_S49 + u32(1)].xyz * _S46, 0.0f);
            p_1 = p_1 + u32(1024);
        }
        var _S51 : u32 = iterations_0 + u32(1);
        device_group_barrier_0();
        var _S52 : u32 = v_1 + u32(1);
        ia_0 = z_0;
        v_1 = _S52;
        iterations_0 = _S51;
    }
    device_group_barrier_0();
    q_0 = b0_0 + tid_1;
    loop
    {
        if(q_0 < _S19)
        {
        }
        else
        {
            break;
        }
        var j_2 : u32 = lists_0[q_0];
        var _S53 : BlastBond_std430_0 = bonds_0[j_2];
        if(!((_S53.health_0) > 0.0f))
        {
            var _S54 : u32 = j_2 * u32(2);
            var _S55 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S54] = _S55;
            x_0[_S54 + u32(1)] = _S55;
            q_0 = q_0 + u32(1024);
            continue;
        }
        var ta_1 : vec3<f32>;
        var tl_1 : vec3<f32>;
        bond_term_0(&(_S53), u32(6), &(ta_1), &(tl_1));
        var _S56 : u32 = j_2 * u32(2);
        x_0[_S56] = vec4<f32>(x_0[_S56].xyz + ta_1, 0.0f);
        var _S57 : u32 = _S56 + u32(1);
        x_0[_S57] = vec4<f32>(x_0[_S57].xyz + tl_1, 0.0f);
        q_0 = q_0 + u32(1024);
    }
    if(tid_1 == u32(0))
    {
        if(converged_0)
        {
            p_0 = u32(5);
        }
        else
        {
            p_0 = u32(0);
        }
        report_0[k_1] = vec4<u32>(iterations_0, p_0, (bitcast<u32>((il_0))), (bitcast<u32>((target2_0))));
    }
    return;
}

