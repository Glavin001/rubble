struct PlanParams_std140_0
{
    @align(16) chunk_count_0 : u32,
    @align(4) island_count_0 : u32,
    @align(8) job_count_0 : u32,
    @align(4) impactor_count_0 : u32,
    @align(16) duration_0 : f32,
    @align(4) g_0 : f32,
    @align(8) gz_0 : f32,
    @align(4) ground_height_0 : f32,
    @align(16) has_ground_0 : u32,
    @align(4) reach_base_0 : u32,
    @align(8) cp_base_0 : u32,
    @align(4) job_base_0 : u32,
    @align(16) count_base_0 : u32,
    @align(4) pair_base_0 : u32,
    @align(8) flag_base_0 : u32,
    @align(4) owned_base_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : PlanParams_std140_0;
struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(1) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

struct Island_std430_0
{
    @align(16) range_0 : vec4<u32>,
    @align(16) info_1 : vec4<u32>,
    @align(16) com_0 : vec4<f32>,
    @align(16) inertia0_1 : vec4<f32>,
    @align(16) inertia1_1 : vec4<f32>,
    @align(16) inertia2_1 : vec4<f32>,
    @align(16) inv0_1 : vec4<f32>,
    @align(16) inv1_1 : vec4<f32>,
    @align(16) inv2_1 : vec4<f32>,
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

@binding(2) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

@binding(3) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read_write> pm_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> pu_0 : array<u32>;

struct Impactor_std430_0
{
    @align(16) position_1 : vec4<f32>,
    @align(16) position_err_1 : vec4<f32>,
    @align(16) velocity_1 : vec4<f32>,
    @align(16) velocity_err_1 : vec4<f32>,
    @align(16) angular_velocity_1 : vec4<f32>,
    @align(16) rotation_1 : vec4<f32>,
    @align(16) inertia0_2 : vec4<f32>,
    @align(16) inertia1_2 : vec4<f32>,
    @align(16) inertia2_2 : vec4<f32>,
    @align(16) inv0_2 : vec4<f32>,
    @align(16) inv1_2 : vec4<f32>,
    @align(16) inv2_2 : vec4<f32>,
    @align(16) shape_0 : vec4<f32>,
    @align(16) half_1 : vec4<f32>,
    @align(16) mat_0 : vec4<f32>,
    @align(16) crush_0 : vec4<f32>,
    @align(16) geom_0 : vec4<f32>,
    @align(16) rotation_err_1 : vec4<f32>,
    @align(16) ledger_0 : vec4<f32>,
    @align(16) cand_0 : vec4<u32>,
    @align(16) momentum_1 : vec4<f32>,
    @align(16) momentum_err_1 : vec4<f32>,
};

@binding(4) @group(0) var<storage, read> impactors_0 : array<Impactor_std430_0>;

struct Quat_0
{
     w_0 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_0 : Quat_0;
    r_0.x_0 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn rotate_0( q_1 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_1.x_0, q_1.y_0, q_1.z_0);
    var t_0 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_0 * vec3<f32>(q_1.w_0) + cross(qv_0, t_0);
}

@compute
@workgroup_size(64, 1, 1)
fn plan_motion(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var k_0 : u32 = id_0.x;
    if(k_0 >= (params_0.chunk_count_0))
    {
        return;
    }
    var q_2 : Quat_0 = quat_of_0(islands_0[chunks_0[k_0].info_0.y].rotation_0);
    var _S1 : u32 = u32(4) * k_0;
    var local_0 : vec3<f32> = chunks_0[k_0].center_0.xyz + state_0[_S1].xyz;
    var _S2 : vec3<f32> = islands_0[chunks_0[k_0].info_0.y].position_0.xyz + islands_0[chunks_0[k_0].info_0.y].position_err_0.xyz;
    var pos_0 : vec3<f32> = _S2 + rotate_0(q_2, local_0);
    var _S3 : vec3<f32> = islands_0[chunks_0[k_0].info_0.y].com_0.xyz;
    var _S4 : vec3<f32> = islands_0[chunks_0[k_0].info_0.y].angular_velocity_0.xyz;
    var v_1 : vec3<f32> = islands_0[chunks_0[k_0].info_0.y].velocity_0.xyz + islands_0[chunks_0[k_0].info_0.y].velocity_err_0.xyz + cross(_S4, rotate_0(q_2, local_0 - _S3)) + rotate_0(q_2, state_0[_S1 + u32(2)].xyz);
    var w_1 : vec3<f32> = _S4 + rotate_0(q_2, state_0[_S1 + u32(3)].xyz);
    var radius_0 : f32 = length(chunks_0[k_0].half_0.xyz);
    var budget_0 : f32 = 2.0f * length(w_1) * (length(pos_0 - (_S2 + rotate_0(q_2, _S3))) + radius_0) * params_0.duration_0 + params_0.g_0 * params_0.duration_0 * params_0.duration_0 + 0.05000000074505806f * radius_0 + 0.00100000004749745f;
    var falls_0 : bool = (((islands_0[chunks_0[k_0].info_0.y].info_1.x) & (u32(1)))) == u32(0);
    var _S5 : u32 = u32(3) * k_0;
    pm_0[_S5] = vec4<f32>(pos_0, radius_0);
    pm_0[_S5 + u32(1)] = vec4<f32>(v_1, budget_0);
    var _S6 : f32;
    if(falls_0)
    {
        _S6 = 1.0f;
    }
    else
    {
        _S6 = 0.0f;
    }
    pm_0[_S5 + u32(2)] = vec4<f32>(w_1, _S6);
    return;
}

var<workgroup> g_spread_0 : array<f32, i32(64)>;

var<workgroup> g_rb_0 : array<f32, i32(64)>;

var<workgroup> g_v_0 : array<f32, i32(64)>;

@compute
@workgroup_size(64, 1, 1)
fn plan_reach(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var i_0 : u32 = group_0.x;
    var tid_0 : u32 = thread_0.x;
    if(i_0 >= (params_0.island_count_0))
    {
        return;
    }
    var _S7 : u32 = islands_0[i_0].range_0.y;
    var _S8 : u32 = islands_0[i_0].range_0.x;
    var center_1 : vec3<f32>;
    if(_S7 > _S8)
    {
        center_1 = pm_0[u32(3) * _S8].xyz;
    }
    else
    {
        center_1 = vec3<f32>(0.0f);
    }
    var _S9 : u32 = _S8 + tid_0;
    var spread_0 : f32 = 0.0f;
    var rb_0 : f32 = 0.0f;
    var vmax_0 : f32 = 0.0f;
    var k_1 : u32 = _S9;
    loop
    {
        if(k_1 < _S7)
        {
        }
        else
        {
            break;
        }
        var _S10 : u32 = u32(3) * k_1;
        var _S11 : f32 = max(spread_0, length(pm_0[_S10].xyz - center_1));
        var _S12 : f32 = max(rb_0, pm_0[_S10].w + pm_0[_S10 + u32(1)].w);
        var _S13 : f32 = max(vmax_0, length(pm_0[_S10 + u32(1)].xyz));
        var k_2 : u32 = k_1 + u32(64);
        spread_0 = _S11;
        rb_0 = _S12;
        vmax_0 = _S13;
        k_1 = k_2;
    }
    g_spread_0[tid_0] = spread_0;
    g_rb_0[tid_0] = rb_0;
    g_v_0[tid_0] = vmax_0;
    workgroupBarrier();
    var s_0 : u32 = u32(32);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < s_0)
        {
            var _S14 : u32 = tid_0 + s_0;
            g_spread_0[tid_0] = max(g_spread_0[tid_0], g_spread_0[_S14]);
            g_rb_0[tid_0] = max(g_rb_0[tid_0], g_rb_0[_S14]);
            g_v_0[tid_0] = max(g_v_0[tid_0], g_v_0[_S14]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    if(tid_0 == u32(0))
    {
        var _S15 : u32 = u32(2) * i_0;
        pm_0[params_0.reach_base_0 + _S15] = vec4<f32>(center_1, g_spread_0[i32(0)]);
        pm_0[params_0.reach_base_0 + _S15 + u32(1)] = vec4<f32>(g_rb_0[i32(0)], g_v_0[i32(0)], 0.0f, 0.0f);
    }
    return;
}

fn clusters_may_touch_0( ia_0 : u32,  ib_0 : u32) -> bool
{
    var _S16 : u32 = u32(2) * ia_0;
    var a0_0 : vec4<f32> = pm_0[params_0.reach_base_0 + _S16];
    var a1_0 : vec4<f32> = pm_0[params_0.reach_base_0 + _S16 + u32(1)];
    var _S17 : u32 = u32(2) * ib_0;
    var b0_0 : vec4<f32> = pm_0[params_0.reach_base_0 + _S17];
    var b1_0 : vec4<f32> = pm_0[params_0.reach_base_0 + _S17 + u32(1)];
    var d_0 : f32 = params_0.duration_0;
    var dg_0 : f32;
    if(((((islands_0[ia_0].info_1.x) & (u32(1)))) != u32(0)) == ((((islands_0[ib_0].info_1.x) & (u32(1)))) != u32(0)))
    {
        dg_0 = 0.0f;
    }
    else
    {
        dg_0 = params_0.g_0;
    }
    var _S18 : f32 = a0_0.w;
    var _S19 : f32 = b0_0.w;
    var _S20 : f32 = 0.5f * ((a1_0.y + b1_0.y) * d_0);
    var near_0 : f32 = length(b0_0.xyz - a0_0.xyz) - _S18 - _S19 - _S20;
    var limit_0 : f32 = a1_0.x + b1_0.x + _S20 + 0.5f * dg_0 * d_0 * d_0;
    return near_0 <= (limit_0 + 0.00000999999974738f * (abs(near_0) + limit_0 + _S18 + _S19 + 1.0f));
}

fn safe_normalize_0( v_2 : vec3<f32>) -> vec3<f32>
{
    var n_0 : f32 = length(v_2);
    var _S21 : vec3<f32>;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S21 = v_2 / vec3<f32>(n_0);
    }
    else
    {
        _S21 = vec3<f32>(0.0f);
    }
    return _S21;
}

fn from_axis_angle_0( axis_0 : vec3<f32>,  angle_0 : f32) -> Quat_0
{
    var a_0 : vec3<f32> = safe_normalize_0(axis_0);
    var _S22 : f32 = 0.5f * angle_0;
    var s_1 : f32 = sin(_S22);
    var q_3 : Quat_0;
    q_3.w_0 = cos(_S22);
    q_3.x_0 = a_0.x * s_1;
    q_3.y_0 = a_0.y * s_1;
    q_3.z_0 = a_0.z * s_1;
    return q_3;
}

struct Box_0
{
     center_2 : vec3<f32>,
     axis0_0 : vec3<f32>,
     axis1_0 : vec3<f32>,
     axis2_0 : vec3<f32>,
     half_2 : vec3<f32>,
};

fn grown_box_0( c_0 : u32,  center_3 : vec3<f32>,  grow_0 : f32) -> Box_0
{
    var q_4 : Quat_0 = quat_of_0(islands_0[chunks_0[c_0].info_0.y].rotation_0);
    var th_0 : vec3<f32> = state_0[u32(4) * c_0 + u32(1)].xyz;
    var hidden_0 : Quat_0 = from_axis_angle_0(th_0, length(th_0));
    var b_0 : Box_0;
    b_0.center_2 = center_3;
    b_0.axis0_0 = rotate_0(q_4, rotate_0(hidden_0, vec3<f32>(chunks_0[c_0].crot0_0.x, chunks_0[c_0].crot1_0.x, chunks_0[c_0].crot2_0.x)));
    b_0.axis1_0 = rotate_0(q_4, rotate_0(hidden_0, vec3<f32>(chunks_0[c_0].crot0_0.y, chunks_0[c_0].crot1_0.y, chunks_0[c_0].crot2_0.y)));
    b_0.axis2_0 = rotate_0(q_4, rotate_0(hidden_0, vec3<f32>(chunks_0[c_0].crot0_0.z, chunks_0[c_0].crot1_0.z, chunks_0[c_0].crot2_0.z)));
    b_0.half_2 = chunks_0[c_0].half_0.xyz + vec3<f32>(grow_0);
    return b_0;
}

fn may_overlap_0( a_1 : Box_0,  b_1 : Box_0) -> bool
{
    var _S23 : vec3<f32> = b_1.center_2 - a_1.center_2;
    var _S24 : f32 = 0.00000999999974738f * (length(a_1.half_2) + length(b_1.half_2));
    var _S25 : array<vec3<f32>, i32(6)> = array<vec3<f32>, i32(6)>( a_1.axis0_0, a_1.axis1_0, a_1.axis2_0, b_1.axis0_0, b_1.axis1_0, b_1.axis2_0 );
    var i_1 : u32 = u32(0);
    loop
    {
        if(i_1 < u32(15))
        {
        }
        else
        {
            break;
        }
        var l_0 : vec3<f32>;
        if(i_1 < u32(6))
        {
            l_0 = _S25[i_1];
        }
        else
        {
            var _S26 : u32 = i_1 - u32(6);
            l_0 = cross(_S25[_S26 / u32(3)], _S25[u32(3) + _S26 % u32(3)]);
        }
        var len_0 : f32 = length(l_0);
        if(len_0 <= 9.99999997475242708e-07f)
        {
            i_1 = i_1 + u32(1);
            continue;
        }
        if((abs(dot(_S23, l_0))) > (a_1.half_2.x * abs(dot(a_1.axis0_0, l_0)) + a_1.half_2.y * abs(dot(a_1.axis1_0, l_0)) + a_1.half_2.z * abs(dot(a_1.axis2_0, l_0)) + (b_1.half_2.x * abs(dot(b_1.axis0_0, l_0)) + b_1.half_2.y * abs(dot(b_1.axis1_0, l_0)) + b_1.half_2.z * abs(dot(b_1.axis2_0, l_0))) + _S24 * len_0))
        {
            return false;
        }
        i_1 = i_1 + u32(1);
    }
    return true;
}

fn pair_passes_0( a_2 : u32,  b_2 : u32,  swept_0 : ptr<function, bool>) -> bool
{
    var _S27 : u32 = u32(3) * a_2;
    var pa_0 : vec4<f32> = pm_0[_S27];
    var va_0 : vec4<f32> = pm_0[_S27 + u32(1)];
    var _S28 : u32 = u32(3) * b_2;
    var pb_0 : vec4<f32> = pm_0[_S28];
    var vb_0 : vec4<f32> = pm_0[_S28 + u32(1)];
    var d_1 : f32 = params_0.duration_0;
    var dv_0 : vec3<f32> = pm_0[_S28 + u32(1)].xyz - pm_0[_S27 + u32(1)].xyz;
    var dg_1 : f32;
    if((pm_0[_S27 + u32(2)].w) == (pm_0[_S28 + u32(2)].w))
    {
        dg_1 = 0.0f;
    }
    else
    {
        dg_1 = params_0.g_0;
    }
    var _S29 : vec3<f32> = pa_0.xyz;
    var middle_0 : vec3<f32> = pb_0.xyz - _S29 + dv_0 * vec3<f32>((0.5f * d_1));
    var sweep_0 : f32 = 0.5f * length(dv_0) * d_1 + 0.5f * dg_1 * d_1 * d_1;
    var _S30 : f32 = va_0.w;
    var _S31 : f32 = vb_0.w;
    var _S32 : bool = (length(middle_0)) <= (pa_0.w + pb_0.w + _S30 + _S31 + sweep_0);
    (*swept_0) = _S32;
    if(!_S32)
    {
        return false;
    }
    return may_overlap_0(grown_box_0(a_2, _S29, _S30), grown_box_0(b_2, _S29 + middle_0, _S31 + sweep_0));
}

@compute
@workgroup_size(64, 1, 1)
fn plan_pair_count(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var j_0 : u32 = id_1.x;
    if(j_0 >= (params_0.job_count_0))
    {
        return;
    }
    var _S33 : u32 = u32(2) * j_0;
    var _S34 : u32 = pu_0[params_0.job_base_0 + _S33 + u32(1)];
    var _S35 : u32 = u32(2) * pu_0[params_0.job_base_0 + _S33];
    var ib_1 : u32 = pu_0[params_0.cp_base_0 + _S35 + u32(1)];
    var count_0 : u32;
    if(clusters_may_touch_0(pu_0[params_0.cp_base_0 + _S35], pu_0[params_0.cp_base_0 + _S35 + u32(1)]))
    {
        var _S36 : vec4<u32> = islands_0[ib_1].range_0;
        var b_3 : u32 = islands_0[ib_1].range_0.x;
        count_0 = u32(0);
        loop
        {
            if(b_3 < (_S36.y))
            {
            }
            else
            {
                break;
            }
            var swept_1 : bool;
            var kept_0 : bool = pair_passes_0(_S34, b_3, &(swept_1));
            if(swept_1)
            {
                pu_0[params_0.flag_base_0 + _S34] = (pu_0[params_0.flag_base_0 + _S34] | (u32(1)));
                pu_0[params_0.flag_base_0 + b_3] = (pu_0[params_0.flag_base_0 + b_3] | (u32(1)));
            }
            if(kept_0)
            {
                count_0 = count_0 + u32(1);
            }
            b_3 = b_3 + u32(1);
        }
    }
    else
    {
        count_0 = u32(0);
    }
    pu_0[params_0.count_base_0 + j_0] = count_0;
    return;
}

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

@compute
@workgroup_size(1024, 1, 1)
fn plan_scan(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_1 : u32 = thread_1.x;
    var n_1 : u32 = params_0.job_count_0;
    var base_0 : u32 = u32(0);
    var carry_0 : u32 = u32(0);
    loop
    {
        if(base_0 < n_1)
        {
        }
        else
        {
            break;
        }
        var k_3 : u32 = base_0 + tid_1;
        var _S37 : bool = k_3 < n_1;
        var v_3 : u32;
        if(_S37)
        {
            v_3 = pu_0[params_0.count_base_0 + k_3];
        }
        else
        {
            v_3 = u32(0);
        }
        g_scan_0[tid_1] = v_3;
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
            var add_0 : u32;
            if(tid_1 >= step_0)
            {
                add_0 = g_scan_0[tid_1 - step_0];
            }
            else
            {
                add_0 = u32(0);
            }
            workgroupBarrier();
            g_scan_0[tid_1] = g_scan_0[tid_1] + add_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        if(_S37)
        {
            pu_0[params_0.count_base_0 + n_1 + u32(1) + k_3] = carry_0 + g_scan_0[tid_1] - v_3;
        }
        var total_0 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var carry_1 : u32 = carry_0 + total_0;
        base_0 = base_0 + u32(1024);
        carry_0 = carry_1;
    }
    if(tid_1 == u32(0))
    {
        pu_0[params_0.count_base_0 + n_1] = carry_0;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn plan_pair_write(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var j_1 : u32 = id_2.x;
    var n_2 : u32 = params_0.job_count_0;
    var _S38 : bool;
    if(j_1 >= (params_0.job_count_0))
    {
        _S38 = true;
    }
    else
    {
        _S38 = pu_0[params_0.count_base_0 + j_1] == u32(0);
    }
    if(_S38)
    {
        return;
    }
    var _S39 : u32 = u32(2) * j_1;
    var _S40 : u32 = pu_0[params_0.job_base_0 + _S39 + u32(1)];
    var _S41 : vec4<u32> = islands_0[pu_0[params_0.cp_base_0 + u32(2) * pu_0[params_0.job_base_0 + _S39] + u32(1)]].range_0;
    var b_4 : u32 = islands_0[pu_0[params_0.cp_base_0 + u32(2) * pu_0[params_0.job_base_0 + _S39] + u32(1)]].range_0.x;
    var at_0 : u32 = pu_0[params_0.count_base_0 + n_2 + u32(1) + j_1];
    loop
    {
        if(b_4 < (_S41.y))
        {
        }
        else
        {
            break;
        }
        var swept_2 : bool;
        if(pair_passes_0(_S40, b_4, &(swept_2)))
        {
            var _S42 : u32 = u32(2) * at_0;
            pu_0[params_0.pair_base_0 + _S42] = _S40;
            pu_0[params_0.pair_base_0 + _S42 + u32(1)] = b_4;
            at_0 = at_0 + u32(1);
        }
        b_4 = b_4 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn plan_side(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var k_4 : u32 = id_3.x;
    if(k_4 >= (params_0.chunk_count_0))
    {
        return;
    }
    var _S43 : u32 = chunks_0[k_4].info_0.y;
    var anchored_0 : bool = (((islands_0[_S43].info_1.x) & (u32(1)))) != u32(0);
    if(pu_0[params_0.owned_base_0 + _S43] == u32(0))
    {
        return;
    }
    var _S44 : u32 = u32(3) * k_4;
    var p_0 : vec4<f32> = pm_0[_S44];
    var v_4 : vec4<f32> = pm_0[_S44 + u32(1)];
    var falls_1 : bool = (pm_0[_S44 + u32(2)].w) != 0.0f;
    var d_2 : f32 = params_0.duration_0;
    var _S45 : bool;
    if((params_0.has_ground_0) != u32(0))
    {
        _S45 = !anchored_0;
    }
    else
    {
        _S45 = false;
    }
    var reach_0 : f32;
    var flags_0 : u32;
    if(_S45)
    {
        var z0_0 : f32 = p_0.z;
        var vz_0 : f32 = v_4.z;
        if(falls_1)
        {
            reach_0 = params_0.gz_0;
        }
        else
        {
            reach_0 = 0.0f;
        }
        var t_low_0 : f32;
        if(reach_0 != 0.0f)
        {
            t_low_0 = clamp(- vz_0 / reach_0, 0.0f, d_2);
        }
        else
        {
            t_low_0 = 0.0f;
        }
        var _S46 : f32 = 0.5f * reach_0;
        if((min(min(z0_0, z0_0 + vz_0 * d_2 + _S46 * d_2 * d_2), z0_0 + vz_0 * t_low_0 + _S46 * t_low_0 * t_low_0) - p_0.w - v_4.w) <= (params_0.ground_height_0))
        {
            flags_0 = u32(2);
        }
        else
        {
            flags_0 = u32(0);
        }
    }
    else
    {
        flags_0 = u32(0);
    }
    var _S47 : f32 = length(v_4.xyz) * d_2;
    if(falls_1)
    {
        reach_0 = 0.5f * params_0.g_0 * d_2 * d_2;
    }
    else
    {
        reach_0 = 0.0f;
    }
    var _S48 : f32 = _S47 + reach_0 + v_4.w;
    var i_2 : u32 = u32(0);
    loop
    {
        if(i_2 < (params_0.impactor_count_0))
        {
            _S45 = i_2 < u32(30);
        }
        else
        {
            _S45 = false;
        }
        if(_S45)
        {
        }
        else
        {
            break;
        }
        if((impactors_0[i_2].cand_0.z) != u32(0))
        {
            i_2 = i_2 + u32(1);
            continue;
        }
        var _S49 : vec4<f32> = impactors_0[i_2].shape_0;
        if((impactors_0[i_2].shape_0.x) == 0.0f)
        {
            reach_0 = _S49.y * 1.73205077648162842f;
        }
        else
        {
            reach_0 = length(impactors_0[i_2].half_1.xyz);
        }
        var flags_1 : u32;
        if((length(p_0.xyz - (impactors_0[i_2].position_1.xyz + impactors_0[i_2].position_err_1.xyz))) <= (reach_0 + p_0.w + _S48 + (2.0f * (length(impactors_0[i_2].velocity_1.xyz) + length(impactors_0[i_2].angular_velocity_1.xyz) * reach_0) * d_2 + params_0.g_0 * d_2 * d_2 + 0.00100000004749745f)))
        {
            flags_1 = (flags_0 | (((u32(4) << (i_2)))));
        }
        else
        {
            flags_1 = flags_0;
        }
        flags_0 = flags_1;
        i_2 = i_2 + u32(1);
    }
    pu_0[params_0.flag_base_0 + k_4] = (pu_0[params_0.flag_base_0 + k_4] | (flags_0));
    return;
}

