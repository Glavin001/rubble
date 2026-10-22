@binding(1) @group(0) var<storage, read_write> ctl_0 : array<u32>;

@binding(2) @group(0) var<storage, read_write> calls_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read_write> nodes_0 : array<vec4<f32>>;

struct LearnedParams_std140_0
{
    @align(16) impactor_count_0 : u32,
    @align(4) engine_base_0 : u32,
    @align(8) island_cap_0 : u32,
    @align(4) stub_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) speed_0 : f32,
    @align(8) log_tau_0 : f32,
    @align(4) tolerance_0 : f32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) max_calls_0 : u32,
    @align(4) max_nodes_0 : u32,
    @align(8) max_bonds_0 : u32,
    @align(4) max_call_chunks_0 : u32,
    @align(16) debris_0 : u32,
    @align(4) max_cisl_0 : u32,
    @align(8) chunk_count_0 : u32,
    @align(4) pad0_0 : u32,
    @align(16) out_bases_0 : vec4<u32>,
    @align(16) out_bases2_0 : vec4<u32>,
};

@binding(0) @group(0) var<uniform> params_0 : LearnedParams_std140_0;
@binding(4) @group(0) var<storage, read_write> bonds_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read_write> chunk_node_0 : array<u32>;

@binding(19) @group(0) var<storage, read> counters_0 : array<u32>;

@binding(25) @group(0) var<storage, read> route_0 : array<u32>;

@binding(5) @group(0) var<storage, read_write> isl_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read_write> islf_0 : array<vec4<f32>>;

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

@binding(11) @group(0) var<storage, read> islands_0 : array<Island_std430_0>;

@binding(14) @group(0) var<storage, read> ctab_0 : array<vec4<f32>>;

@binding(15) @group(0) var<storage, read_write> bodies_0 : array<vec4<f32>>;

struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_1 : vec4<f32>,
    @align(16) inertia1_1 : vec4<f32>,
    @align(16) inertia2_1 : vec4<f32>,
    @align(16) inv0_1 : vec4<f32>,
    @align(16) inv1_1 : vec4<f32>,
    @align(16) inv2_1 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_1 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(12) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(13) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> imps_0 : array<vec4<f32>>;

@binding(18) @group(0) var<storage, read_write> status_0 : array<u32>;

@binding(27) @group(0) var<storage, read_write> largs_0 : array<vec4<u32>>;

@binding(7) @group(0) var<storage, read_write> cisl_0 : array<vec4<u32>>;

struct JointState_std430_0
{
    @align(4) damage_0 : f32,
    @align(4) crush_0 : f32,
    @align(4) kappa_0 : f32,
    @align(4) kappa_c_0 : f32,
    @align(4) ductility_0 : f32,
    @align(4) ductility_c_0 : f32,
    @align(4) life_0 : f32,
    @align(4) plastic_x_0 : f32,
    @align(4) plastic_y_0 : f32,
    @align(4) plastic_t_0 : f32,
    @align(4) rebar_plastic_0 : f32,
    @align(4) rebar_slip0_0 : f32,
    @align(4) rebar_slip1_0 : f32,
    @align(4) rebar_work_0 : f32,
    @align(4) rebar_broken_0 : f32,
    @align(4) strain_rate_0 : f32,
    @align(4) governing_stress_0 : f32,
    @align(4) dissipated_0 : f32,
    @align(4) utilization_0 : f32,
    @align(4) mode_0 : u32,
};

struct BondDyn_std430_0
{
    @align(16) js_0 : JointState_std430_0,
    @align(16) force_lin_0 : vec4<f32>,
    @align(16) force_ang_0 : vec4<f32>,
    @align(16) sums_0 : vec4<f32>,
    @align(16) comps_0 : vec4<f32>,
    @align(16) events_0 : vec4<u32>,
};

@binding(17) @group(0) var<storage, read_write> bond_dyn_0 : array<BondDyn_std430_0>;

struct JointBond_std430_0
{
    @align(16) geom0_0 : vec4<f32>,
    @align(16) geom1_0 : vec4<f32>,
    @align(16) stiff0_0 : vec4<f32>,
    @align(16) stiff1_0 : vec4<f32>,
    @align(16) rebar0_0 : vec4<f32>,
    @align(16) rebar1_0 : vec4<f32>,
    @align(16) ids_0 : vec4<u32>,
};

struct BondStatic_std430_0
{
    @align(16) t1_0 : vec4<f32>,
    @align(16) t2_0 : vec4<f32>,
    @align(16) normal_0 : vec4<f32>,
    @align(16) ra_0 : vec4<f32>,
    @align(16) rb_0 : vec4<f32>,
    @align(16) centroid_0 : vec4<f32>,
    @align(16) c_lin_0 : vec4<f32>,
    @align(16) c_ang_0 : vec4<f32>,
    @align(16) law_0 : JointBond_std430_0,
};

@binding(16) @group(0) var<storage, read> bond_static_0 : array<BondStatic_std430_0>;

@binding(9) @group(0) var<storage, read_write> frozen_0 : array<vec4<f32>>;

@binding(21) @group(0) var<storage, read> split_work_0 : array<u32>;

@binding(23) @group(0) var<storage, read> split_control_0 : array<u32>;

@binding(20) @group(0) var<storage, read> split_parents_0 : array<vec4<u32>>;

@binding(22) @group(0) var<storage, read> split_slots_0 : array<vec4<u32>>;

@binding(26) @group(0) var<storage, read> split_out_0 : array<vec4<f32>>;

@binding(24) @group(0) var<storage, read> split_firsts_0 : array<vec4<u32>>;

@binding(28) @group(0) var<storage, read> lout_0 : array<f32>;

fn firstbitlow_0( value_0 : u32) -> u32
{
    var _S1 : u32 = firstTrailingBit(value_0);
    return _S1;
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_1 : vec4<u32>,  thread_0 : u32) -> vec4<u32>
{
    if(thread_0 == u32(0))
    {
        g_group_uniform4_0 = value_1;
    }
    var _S2 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S2;
}

fn node_0( n_0 : u32,  k_0 : u32) -> vec4<f32>
{
    return nodes_0[u32(14) * n_0 + k_0];
}

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

var<workgroup> g_red_0 : array<array<vec4<f32>, i32(256)>, i32(4)>;

fn reduce_store_0( t_1 : u32,  a_0 : vec4<f32>,  b_0 : vec4<f32>,  c_0 : vec4<f32>,  d_0 : vec4<f32>)
{
    g_red_0[i32(0)][t_1] = a_0;
    g_red_0[i32(1)][t_1] = b_0;
    g_red_0[i32(2)][t_1] = c_0;
    g_red_0[i32(3)][t_1] = d_0;
    workgroupBarrier();
    var s_0 : u32 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_0)
        {
            var _S3 : u32 = t_1 + s_0;
            g_red_0[i32(0)][t_1] = g_red_0[i32(0)][t_1] + g_red_0[i32(0)][_S3];
            g_red_0[i32(1)][t_1] = g_red_0[i32(1)][t_1] + g_red_0[i32(1)][_S3];
            g_red_0[i32(2)][t_1] = g_red_0[i32(2)][t_1] + g_red_0[i32(2)][_S3];
            g_red_0[i32(3)][t_1] = g_red_0[i32(3)][t_1] + g_red_0[i32(3)][_S3];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    return;
}

fn reduce_read_0( a_1 : ptr<function, vec4<f32>>,  b_1 : ptr<function, vec4<f32>>,  c_1 : ptr<function, vec4<f32>>,  d_1 : ptr<function, vec4<f32>>)
{
    (*a_1) = g_red_0[i32(0)][i32(0)];
    (*b_1) = g_red_0[i32(1)][i32(0)];
    (*c_1) = g_red_0[i32(2)][i32(0)];
    (*d_1) = g_red_0[i32(3)][i32(0)];
    workgroupBarrier();
    return;
}

fn quat_matrix_0( q_2 : vec4<f32>) -> mat3x3<f32>
{
    var w_1 : f32 = q_2.w;
    var x_1 : f32 = q_2.x;
    var y_1 : f32 = q_2.y;
    var z_1 : f32 = q_2.z;
    var _S4 : f32 = y_1 * y_1;
    var _S5 : f32 = z_1 * z_1;
    var _S6 : f32 = x_1 * y_1;
    var _S7 : f32 = w_1 * z_1;
    var _S8 : f32 = x_1 * z_1;
    var _S9 : f32 = w_1 * y_1;
    var _S10 : f32 = x_1 * x_1;
    var _S11 : f32 = y_1 * z_1;
    var _S12 : f32 = w_1 * x_1;
    return mat3x3<f32>(vec3<f32>(1.0f - 2.0f * (_S4 + _S5), 2.0f * (_S6 - _S7), 2.0f * (_S8 + _S9)), vec3<f32>(2.0f * (_S6 + _S7), 1.0f - 2.0f * (_S10 + _S5), 2.0f * (_S11 - _S12)), vec3<f32>(2.0f * (_S8 - _S9), 2.0f * (_S11 + _S12), 1.0f - 2.0f * (_S10 + _S4)));
}

fn diag3_0( d_2 : vec3<f32>) -> mat3x3<f32>
{
    return mat3x3<f32>(vec3<f32>(d_2.x, 0.0f, 0.0f), vec3<f32>(0.0f, d_2.y, 0.0f), vec3<f32>(0.0f, 0.0f, d_2.z));
}

fn world_inertia_0( q_3 : vec4<f32>,  principal_0 : vec3<f32>) -> mat3x3<f32>
{
    var r_1 : mat3x3<f32> = quat_matrix_0(q_3);
    return (((transpose(r_1)) * ((((diag3_0(principal_0)) * (r_1))))));
}

fn skew_0( r_2 : vec3<f32>) -> mat3x3<f32>
{
    var _S13 : f32 = r_2.z;
    var _S14 : f32 = r_2.y;
    var _S15 : f32 = r_2.x;
    return mat3x3<f32>(vec3<f32>(0.0f, - _S13, _S14), vec3<f32>(_S13, 0.0f, - _S15), vec3<f32>(- _S14, _S15, 0.0f));
}

fn rotvec_matrix_0( r_3 : vec3<f32>) -> mat3x3<f32>
{
    var th_0 : f32 = length(r_3);
    var eye_0 : mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f));
    if(th_0 < 9.99999997475242708e-07f)
    {
        return eye_0 + skew_0(r_3);
    }
    var k_1 : mat3x3<f32> = skew_0(r_3 / vec3<f32>(th_0));
    var _S16 : mat3x3<f32> = mat3x3<f32>(sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0));
    var _S17 : mat3x3<f32> = (((k_1) * (k_1)));
    var _S18 : mat3x3<f32> = mat3x3<f32>(1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0));
    return eye_0 + mat3x3<f32>(_S16[0] * k_1[0], _S16[1] * k_1[1], _S16[2] * k_1[2]) + mat3x3<f32>(_S18[0] * _S17[0], _S18[1] * _S17[1], _S18[2] * _S17[2]);
}

fn outer3_0( a_2 : vec3<f32>,  b_2 : vec3<f32>) -> mat3x3<f32>
{
    return mat3x3<f32>(a_2 * vec3<f32>(b_2.x), a_2 * vec3<f32>(b_2.y), a_2 * vec3<f32>(b_2.z));
}

var<workgroup> g_lambda_0 : array<f32, i32(6)>;

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_2 : u32,  thread_1 : u32) -> u32
{
    if(thread_1 == u32(0))
    {
        g_group_uniform_0 = value_2;
    }
    var _S19 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S19;
}

fn solve3_0( a_3 : mat3x3<f32>,  b_3 : vec3<f32>) -> vec3<f32>
{
    var c0_0 : vec3<f32> = vec3<f32>(a_3[i32(0)][i32(0)], a_3[i32(1)][i32(0)], a_3[i32(2)][i32(0)]);
    var c1_0 : vec3<f32> = vec3<f32>(a_3[i32(0)][i32(1)], a_3[i32(1)][i32(1)], a_3[i32(2)][i32(1)]);
    var c2_0 : vec3<f32> = vec3<f32>(a_3[i32(0)][i32(2)], a_3[i32(1)][i32(2)], a_3[i32(2)][i32(2)]);
    var _S20 : vec3<f32> = cross(c1_0, c2_0);
    return vec3<f32>(dot(b_3, _S20), dot(c0_0, cross(b_3, c2_0)), dot(c0_0, cross(c1_0, b_3))) / vec3<f32>(dot(c0_0, _S20));
}

@compute
@workgroup_size(256, 1, 1)
fn learned_decode(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var v_end_0 : vec3<f32>;
    var w_end_0 : vec3<f32>;
    var q_4 : u32;
    var g_0 : u32 = group_0.x;
    var t_2 : u32 = thread_2.x;
    if(g_0 >= ctl_0[u32(1)])
    {
        return;
    }
    var _S21 : u32 = u32(8) * g_0;
    var _S22 : u32 = _S21 + u32(4);
    var c4_0 : vec4<f32> = calls_0[_S22];
    var _S23 : f32 = calls_0[_S21].x;
    var _S24 : f32 = calls_0[_S21].z;
    var dt_1 : f32 = calls_0[_S21 + u32(1)].w;
    var e_in_0 : f32 = calls_0[_S21 + u32(2)].x;
    var _S25 : f32 = calls_0[_S22].y;
    var u_0 : vec4<u32> = group_uniform4_0(vec4<u32>((bitcast<u32>((calls_0[_S21 + u32(2)].y))), (bitcast<u32>((calls_0[_S21 + u32(2)].z))), (bitcast<u32>((calls_0[_S21 + u32(2)].w))), (bitcast<u32>((_S25)))), t_2);
    var flags_0 : u32 = u_0.x;
    var _S26 : u32 = u_0.y;
    var _S27 : u32 = u_0.z;
    var _S28 : u32 = u_0.w;
    var _S29 : f32 = calls_0[_S22].x;
    var _S30 : u32 = (bitcast<u32>((_S29)));
    var grav_0 : vec3<f32> = calls_0[_S21 + u32(3)].xyz;
    var free_0 : bool = ((flags_0 & (u32(1)))) != u32(0);
    var j_0 : u32 = t_2;
    loop
    {
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        var k_2 : u32 = _S26 + j_0;
        var v_1 : vec3<f32> = node_0(k_2, u32(3)).xyz;
        var w_2 : vec3<f32> = node_0(k_2, u32(4)).xyz;
        var o0_0 : vec4<f32> = node_0(k_2, u32(5));
        var o1_0 : vec4<f32> = node_0(k_2, u32(6));
        var o2_0 : vec4<f32> = node_0(k_2, u32(7));
        var qq_0 : Quat_0 = quat_of_0(node_0(k_2, u32(1)));
        var _S31 : vec3<f32> = vec3<f32>(_S24);
        var _S32 : vec3<f32> = v_1 + rotate_0(qq_0, o0_0.xyz) * _S31;
        var _S33 : vec3<f32> = vec3<f32>(_S23);
        var _S34 : vec3<f32> = w_2 + rotate_0(qq_0, vec3<f32>(o0_0.w, o1_0.x, o1_0.y)) * _S31 / _S33;
        var _S35 : vec3<f32> = vec3<f32>(dt_1);
        var _S36 : vec3<f32> = v_1 * _S35 + vec3<f32>(0.5f) * grav_0 * _S35 * _S35 + rotate_0(qq_0, vec3<f32>(o1_0.z, o1_0.w, o2_0.x)) * _S31 * _S35;
        var _S37 : vec3<f32> = w_2 * _S35 + rotate_0(qq_0, o2_0.yzw) * _S31 * _S35 / _S33;
        var dx_0 : vec3<f32>;
        var dr_0 : vec3<f32>;
        if((node_0(k_2, u32(2)).w) != 0.0f)
        {
            var _S38 : vec3<f32> = vec3<f32>(0.0f);
            v_end_0 = _S38;
            w_end_0 = _S38;
            dx_0 = _S38;
            dr_0 = _S38;
        }
        else
        {
            v_end_0 = _S32;
            w_end_0 = _S34;
            dx_0 = _S36;
            dr_0 = _S37;
        }
        var _S39 : u32 = u32(14) * k_2;
        nodes_0[_S39 + u32(8)] = vec4<f32>(v_end_0, 0.0f);
        nodes_0[_S39 + u32(9)] = vec4<f32>(w_end_0, 0.0f);
        nodes_0[_S39 + u32(10)] = vec4<f32>(dx_0, 0.0f);
        nodes_0[_S39 + u32(11)] = vec4<f32>(dr_0, 0.0f);
        j_0 = j_0 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    var a_4 : vec4<f32>;
    var b_4 : vec4<f32>;
    var c_2 : vec4<f32>;
    var d_3 : vec4<f32>;
    var _S40 : vec4<f32> = vec4<f32>(0.0f);
    j_0 = t_2;
    var sa_0 : vec4<f32> = _S40;
    var sb_0 : vec4<f32> = _S40;
    var sc_0 : vec4<f32> = _S40;
    loop
    {
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        var k_3 : u32 = _S26 + j_0;
        var p_0 : vec4<f32> = node_0(k_3, u32(0));
        var m_0 : f32 = p_0.w;
        var _S41 : vec3<f32> = vec3<f32>(m_0);
        var sa_1 : vec4<f32> = sa_0 + vec4<f32>(p_0.xyz * _S41, m_0);
        var sb_1 : vec4<f32> = sb_0 + vec4<f32>(node_0(k_3, u32(3)).xyz * _S41, 0.0f);
        var sc_1 : vec4<f32> = sc_0 + vec4<f32>(node_0(k_3, u32(10)).xyz * _S41, 0.0f);
        j_0 = j_0 + u32(256);
        sa_0 = sa_1;
        sb_0 = sb_1;
        sc_0 = sc_1;
    }
    reduce_store_0(t_2, sa_0, sb_0, sc_0, _S40);
    reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
    var mt_0 : f32 = a_4.w;
    var _S42 : vec3<f32> = vec3<f32>(mt_0);
    var _S43 : vec3<f32> = a_4.xyz / _S42;
    var p0_0 : vec3<f32> = b_4.xyz;
    var have_0 : vec3<f32> = c_2.xyz;
    var _S44 : vec3<f32> = vec3<f32>(0.0f);
    if(free_0)
    {
        var _S45 : vec3<f32> = vec3<f32>(dt_1);
        v_end_0 = (p0_0 * _S45 + vec3<f32>((0.5f * mt_0)) * grav_0 * _S45 * _S45 - have_0) / _S42;
    }
    else
    {
        v_end_0 = _S44;
    }
    j_0 = t_2;
    sa_0 = _S40;
    sb_0 = _S40;
    loop
    {
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        var k_4 : u32 = _S26 + j_0;
        var p_1 : vec4<f32> = node_0(k_4, u32(0));
        var m_1 : f32 = p_1.w;
        var dx_1 : vec3<f32> = node_0(k_4, u32(10)).xyz;
        if(free_0)
        {
            w_end_0 = dx_1 + v_end_0;
        }
        else
        {
            w_end_0 = dx_1;
        }
        var _S46 : vec3<f32> = p_1.xyz;
        var _S47 : vec3<f32> = vec3<f32>(m_1);
        var sa_2 : vec4<f32> = sa_0 + vec4<f32>((_S46 + w_end_0) * _S47, 0.0f);
        var sb_2 : vec4<f32> = sb_0 + vec4<f32>(cross(_S46 - _S43, _S47 * node_0(k_4, u32(3)).xyz) + (((node_0(k_4, u32(4)).xyz) * (world_inertia_0(node_0(k_4, u32(1)), node_0(k_4, u32(2)).xyz)))), 0.0f);
        j_0 = j_0 + u32(256);
        sa_0 = sa_2;
        sb_0 = sb_2;
    }
    reduce_store_0(t_2, sa_0, sb_0, _S40, _S40);
    reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
    var _S48 : vec3<f32> = a_4.xyz / _S42;
    var l0_0 : vec3<f32> = b_4.xyz;
    var dec_0 : u32;
    var bad_0 : u32;
    var ta_0 : vec4<f32>;
    var tb_0 : vec4<f32>;
    var tc_0 : vec4<f32>;
    if(free_0)
    {
        j_0 = t_2;
        loop
        {
            if(j_0 < _S27)
            {
            }
            else
            {
                break;
            }
            var k_5 : u32 = _S26 + j_0;
            nodes_0[u32(14) * k_5 + u32(10)] = vec4<f32>(node_0(k_5, u32(10)).xyz + v_end_0, 0.0f);
            j_0 = j_0 + u32(256);
        }
        j_0 = t_2;
        sa_0 = _S40;
        sb_0 = _S40;
        sc_0 = _S40;
        ta_0 = _S40;
        tb_0 = _S40;
        tc_0 = _S40;
        loop
        {
            if(j_0 < _S27)
            {
            }
            else
            {
                break;
            }
            var k_6 : u32 = _S26 + j_0;
            var p_2 : vec4<f32> = node_0(k_6, u32(0));
            var m_2 : f32 = p_2.w;
            var r_4 : vec3<f32> = p_2.xyz + node_0(k_6, u32(10)).xyz - _S48;
            var rd_0 : mat3x3<f32> = rotvec_matrix_0(node_0(k_6, u32(11)).xyz);
            var i1_0 : mat3x3<f32> = (((transpose(rd_0)) * ((((world_inertia_0(node_0(k_6, u32(1)), node_0(k_6, u32(2)).xyz)) * (rd_0))))));
            var _S49 : vec3<f32> = vec3<f32>(m_2);
            var _S50 : vec3<f32> = _S49 * node_0(k_6, u32(8)).xyz;
            var sa_3 : vec4<f32> = sa_0 + vec4<f32>(_S50, 0.0f);
            var sb_3 : vec4<f32> = sb_0 + vec4<f32>(cross(r_4, _S50) + (((node_0(k_6, u32(9)).xyz) * (i1_0))), 0.0f);
            var sc_2 : vec4<f32> = sc_0 + vec4<f32>(_S49 * r_4, 0.0f);
            var _S51 : mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f));
            var _S52 : mat3x3<f32> = mat3x3<f32>(dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4));
            var _S53 : mat3x3<f32> = mat3x3<f32>(_S52[0] * _S51[0], _S52[1] * _S51[1], _S52[2] * _S51[2]) - outer3_0(r_4, r_4);
            var _S54 : mat3x3<f32> = mat3x3<f32>(m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2);
            var jj_0 : mat3x3<f32> = mat3x3<f32>(_S54[0] * _S53[0], _S54[1] * _S53[1], _S54[2] * _S53[2]) + i1_0;
            var sd_0 : vec4<f32> = ta_0 + vec4<f32>(jj_0[i32(0)], 0.0f);
            var se_0 : vec4<f32> = tb_0 + vec4<f32>(jj_0[i32(1)], 0.0f);
            var sf_0 : vec4<f32> = tc_0 + vec4<f32>(jj_0[i32(2)], 0.0f);
            j_0 = j_0 + u32(256);
            sa_0 = sa_3;
            sb_0 = sb_3;
            sc_0 = sc_2;
            ta_0 = sd_0;
            tb_0 = se_0;
            tc_0 = sf_0;
        }
        reduce_store_0(t_2, sa_0, sb_0, sc_0, ta_0);
        var ax_p_0 : vec4<f32>;
        var ax_l_0 : vec4<f32>;
        var smr_0 : vec4<f32>;
        var j0_0 : vec4<f32>;
        reduce_read_0(&(ax_p_0), &(ax_l_0), &(smr_0), &(j0_0));
        reduce_store_0(t_2, tb_0, tc_0, _S40, _S40);
        var j1_0 : vec4<f32>;
        var j2_0 : vec4<f32>;
        var unused0_0 : vec4<f32>;
        var unused1_0 : vec4<f32>;
        reduce_read_0(&(j1_0), &(j2_0), &(unused0_0), &(unused1_0));
        if(t_2 == u32(0))
        {
            var _S55 : vec3<f32> = p0_0 + _S42 * grav_0 * vec3<f32>(dt_1) - ax_p_0.xyz;
            var _S56 : vec3<f32> = l0_0 - ax_l_0.xyz;
            var _S57 : mat3x3<f32> = skew_0(smr_0.xyz);
            var m6_0 : array<array<f32, i32(7)>, i32(6)>;
            dec_0 = u32(0);
            loop
            {
                if(dec_0 < u32(6))
                {
                }
                else
                {
                    break;
                }
                q_4 = u32(0);
                loop
                {
                    if(q_4 < u32(7))
                    {
                    }
                    else
                    {
                        break;
                    }
                    m6_0[dec_0][q_4] = 0.0f;
                    q_4 = q_4 + u32(1);
                }
                dec_0 = dec_0 + u32(1);
            }
            var _S58 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( j0_0.xyz, j1_0.xyz, j2_0.xyz );
            dec_0 = u32(0);
            loop
            {
                if(dec_0 < u32(3))
                {
                }
                else
                {
                    break;
                }
                m6_0[dec_0][dec_0] = mt_0;
                q_4 = u32(0);
                loop
                {
                    if(q_4 < u32(3))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S59 : u32 = u32(3) + q_4;
                    m6_0[dec_0][_S59] = - _S57[dec_0][q_4];
                    m6_0[u32(3) + dec_0][q_4] = _S57[dec_0][q_4];
                    m6_0[u32(3) + dec_0][_S59] = _S58[dec_0][q_4];
                    q_4 = q_4 + u32(1);
                }
                m6_0[dec_0][i32(6)] = _S55[dec_0];
                m6_0[u32(3) + dec_0][i32(6)] = _S56[dec_0];
                dec_0 = dec_0 + u32(1);
            }
            bad_0 = u32(0);
            loop
            {
                if(bad_0 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var _S60 : u32 = bad_0 + u32(1);
                q_4 = bad_0;
                dec_0 = _S60;
                loop
                {
                    if(dec_0 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    if((abs(m6_0[dec_0][bad_0])) > (abs(m6_0[q_4][bad_0])))
                    {
                        q_4 = dec_0;
                    }
                    dec_0 = dec_0 + u32(1);
                }
                var q_5 : u32 = u32(0);
                loop
                {
                    if(q_5 < u32(7))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var tmp_0 : f32 = m6_0[bad_0][q_5];
                    m6_0[bad_0][q_5] = m6_0[q_4][q_5];
                    m6_0[q_4][q_5] = tmp_0;
                    q_5 = q_5 + u32(1);
                }
                var r_5 : u32 = _S60;
                loop
                {
                    if(r_5 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S61 : f32 = m6_0[r_5][bad_0] / m6_0[bad_0][bad_0];
                    var q_6 : u32 = bad_0;
                    loop
                    {
                        if(q_6 < u32(7))
                        {
                        }
                        else
                        {
                            break;
                        }
                        m6_0[r_5][q_6] = m6_0[r_5][q_6] - _S61 * m6_0[bad_0][q_6];
                        q_6 = q_6 + u32(1);
                    }
                    r_5 = r_5 + u32(1);
                }
                bad_0 = _S60;
            }
            var x6_0 : array<f32, i32(6)>;
            x6_0[i32(0)] = 0.0f;
            x6_0[i32(1)] = 0.0f;
            x6_0[i32(2)] = 0.0f;
            x6_0[i32(3)] = 0.0f;
            x6_0[i32(4)] = 0.0f;
            x6_0[i32(5)] = 0.0f;
            var r_6 : i32 = i32(5);
            loop
            {
                if(r_6 >= i32(0))
                {
                }
                else
                {
                    break;
                }
                q_4 = u32(r_6) + u32(1);
                var acc_0 : f32 = m6_0[r_6][i32(6)];
                loop
                {
                    if(q_4 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var acc_1 : f32 = acc_0 - m6_0[r_6][q_4] * x6_0[q_4];
                    q_4 = q_4 + u32(1);
                    acc_0 = acc_1;
                }
                x6_0[r_6] = acc_0 / m6_0[r_6][r_6];
                r_6 = r_6 - i32(1);
            }
            dec_0 = u32(0);
            loop
            {
                if(dec_0 < u32(6))
                {
                }
                else
                {
                    break;
                }
                g_lambda_0[dec_0] = x6_0[dec_0];
                dec_0 = dec_0 + u32(1);
            }
        }
        workgroupBarrier();
        var _S62 : vec3<f32> = vec3<f32>(g_lambda_0[i32(0)], g_lambda_0[i32(1)], g_lambda_0[i32(2)]);
        var _S63 : vec3<f32> = vec3<f32>(g_lambda_0[i32(3)], g_lambda_0[i32(4)], g_lambda_0[i32(5)]);
        j_0 = t_2;
        loop
        {
            if(j_0 < _S27)
            {
            }
            else
            {
                break;
            }
            var k_7 : u32 = _S26 + j_0;
            var _S64 : u32 = u32(14) * k_7;
            nodes_0[_S64 + u32(8)] = vec4<f32>(node_0(k_7, u32(8)).xyz + _S62 + cross(_S63, node_0(k_7, u32(0)).xyz + node_0(k_7, u32(10)).xyz - _S48), 0.0f);
            nodes_0[_S64 + u32(9)] = vec4<f32>(node_0(k_7, u32(9)).xyz + _S63, 0.0f);
            j_0 = j_0 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
    }
    if(((flags_0 & (u32(2)))) == u32(0))
    {
        j_0 = t_2;
        sa_0 = _S40;
        sb_0 = _S40;
        loop
        {
            if(j_0 < _S27)
            {
            }
            else
            {
                break;
            }
            var k_8 : u32 = _S26 + j_0;
            var m_3 : f32 = node_0(k_8, u32(0)).w;
            var i0_0 : mat3x3<f32> = world_inertia_0(node_0(k_8, u32(1)), node_0(k_8, u32(2)).xyz);
            var rd_1 : mat3x3<f32> = rotvec_matrix_0(node_0(k_8, u32(11)).xyz);
            var v0_0 : vec3<f32> = node_0(k_8, u32(3)).xyz;
            var w0_0 : vec3<f32> = node_0(k_8, u32(4)).xyz;
            var v_2 : vec3<f32> = node_0(k_8, u32(8)).xyz;
            var w_3 : vec3<f32> = node_0(k_8, u32(9)).xyz;
            var sa_4 : vec4<f32> = sa_0 + vec4<f32>(0.5f * (m_3 * dot(v0_0, v0_0) + dot(w0_0, (((w0_0) * (i0_0))))), m_3 * dot(grav_0, node_0(k_8, u32(10)).xyz), 0.5f * (m_3 * dot(v_2, v_2) + dot(w_3, (((w_3) * ((((transpose(rd_1)) * ((((i0_0) * (rd_1))))))))))), 0.0f);
            var sb_4 : vec4<f32> = sb_0 + vec4<f32>(vec3<f32>(m_3) * v_2, 0.0f);
            j_0 = j_0 + u32(256);
            sa_0 = sa_4;
            sb_0 = sb_4;
        }
        reduce_store_0(t_2, sa_0, sb_0, _S40, _S40);
        reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
        var bound_0 : f32 = a_4.x + a_4.y + params_0.tolerance_0 * e_in_0;
        var ke_0 : f32 = a_4.z;
        if(ke_0 > bound_0)
        {
            j_0 = u32(1);
        }
        else
        {
            j_0 = u32(0);
        }
        var _S65 : u32 = group_uniform_0(j_0, t_2);
        if(_S65 != u32(0))
        {
            if(free_0)
            {
                var vc_0 : vec3<f32> = b_4.xyz / _S42;
                j_0 = t_2;
                ta_0 = _S40;
                tb_0 = _S40;
                tc_0 = _S40;
                var td_0 : vec4<f32> = _S40;
                loop
                {
                    if(j_0 < _S27)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_9 : u32 = _S26 + j_0;
                    var m_4 : f32 = node_0(k_9, u32(0)).w;
                    var r_7 : vec3<f32> = node_0(k_9, u32(0)).xyz + node_0(k_9, u32(10)).xyz - _S48;
                    var rd_2 : mat3x3<f32> = rotvec_matrix_0(node_0(k_9, u32(11)).xyz);
                    var i1_1 : mat3x3<f32> = (((transpose(rd_2)) * ((((world_inertia_0(node_0(k_9, u32(1)), node_0(k_9, u32(2)).xyz)) * (rd_2))))));
                    var _S66 : mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f));
                    var _S67 : mat3x3<f32> = mat3x3<f32>(dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7));
                    var _S68 : mat3x3<f32> = mat3x3<f32>(_S67[0] * _S66[0], _S67[1] * _S66[1], _S67[2] * _S66[2]) - outer3_0(r_7, r_7);
                    var _S69 : mat3x3<f32> = mat3x3<f32>(m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4);
                    var jj_1 : mat3x3<f32> = mat3x3<f32>(_S69[0] * _S68[0], _S69[1] * _S68[1], _S69[2] * _S68[2]) + i1_1;
                    var ta_1 : vec4<f32> = ta_0 + vec4<f32>(jj_1[i32(0)], 0.0f);
                    var tb_1 : vec4<f32> = tb_0 + vec4<f32>(jj_1[i32(1)], 0.0f);
                    var tc_1 : vec4<f32> = tc_0 + vec4<f32>(jj_1[i32(2)], 0.0f);
                    var td_1 : vec4<f32> = td_0 + vec4<f32>(cross(r_7, vec3<f32>(m_4) * (node_0(k_9, u32(8)).xyz - vc_0)) + (((node_0(k_9, u32(9)).xyz) * (i1_1))), 0.0f);
                    j_0 = j_0 + u32(256);
                    ta_0 = ta_1;
                    tb_0 = tb_1;
                    tc_0 = tc_1;
                    td_0 = td_1;
                }
                reduce_store_0(t_2, ta_0, tb_0, tc_0, td_0);
                reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
                var lrel_0 : vec3<f32> = d_3.xyz;
                var omega_0 : vec3<f32> = solve3_0(mat3x3<f32>(a_4.xyz, b_4.xyz, c_2.xyz), lrel_0);
                var ke_base_0 : f32 = 0.5f * (mt_0 * dot(vc_0, vc_0) + dot(omega_0, lrel_0));
                var ra_1 : vec4<f32> = _S40;
                j_0 = t_2;
                loop
                {
                    if(j_0 < _S27)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_10 : u32 = _S26 + j_0;
                    var rd_3 : mat3x3<f32> = rotvec_matrix_0(node_0(k_10, u32(11)).xyz);
                    var uu_0 : vec3<f32> = node_0(k_10, u32(8)).xyz - vc_0 - cross(omega_0, node_0(k_10, u32(0)).xyz + node_0(k_10, u32(10)).xyz - _S48);
                    var wr_0 : vec3<f32> = node_0(k_10, u32(9)).xyz - omega_0;
                    ra_1[i32(0)] = ra_1[i32(0)] + 0.5f * (node_0(k_10, u32(0)).w * dot(uu_0, uu_0) + dot(wr_0, (((wr_0) * ((((transpose(rd_3)) * ((((world_inertia_0(node_0(k_10, u32(1)), node_0(k_10, u32(2)).xyz)) * (rd_3)))))))))));
                    j_0 = j_0 + u32(256);
                }
                reduce_store_0(t_2, ra_1, _S40, _S40, _S40);
                reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
                var _S70 : f32 = sqrt(clamp((bound_0 - ke_base_0) / max(a_4.x, 1.00000000317107685e-30f), 0.0f, 1.0f));
                if(ke_base_0 > bound_0)
                {
                    dec_0 = u32(3);
                }
                else
                {
                    dec_0 = u32(1);
                }
                j_0 = t_2;
                loop
                {
                    if(j_0 < _S27)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_11 : u32 = _S26 + j_0;
                    var _S71 : vec3<f32> = cross(omega_0, node_0(k_11, u32(0)).xyz + node_0(k_11, u32(10)).xyz - _S48);
                    var wr_1 : vec3<f32> = node_0(k_11, u32(9)).xyz - omega_0;
                    var _S72 : u32 = u32(14) * k_11;
                    var _S73 : vec3<f32> = vec3<f32>(_S70);
                    nodes_0[_S72 + u32(8)] = vec4<f32>(vc_0 + _S71 + _S73 * (node_0(k_11, u32(8)).xyz - vc_0 - _S71), 0.0f);
                    nodes_0[_S72 + u32(9)] = vec4<f32>(omega_0 + _S73 * wr_1, 0.0f);
                    j_0 = j_0 + u32(256);
                }
            }
            else
            {
                var _S74 : f32 = sqrt(min(max(bound_0, 0.0f) / max(ke_0, 1.00000000317107685e-30f), 1.0f));
                j_0 = t_2;
                loop
                {
                    if(j_0 < _S27)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_12 : u32 = _S26 + j_0;
                    var _S75 : u32 = u32(14) * k_12;
                    var _S76 : vec3<f32> = vec3<f32>(_S74);
                    nodes_0[_S75 + u32(8)] = vec4<f32>(node_0(k_12, u32(8)).xyz * _S76, 0.0f);
                    nodes_0[_S75 + u32(9)] = vec4<f32>(node_0(k_12, u32(9)).xyz * _S76, 0.0f);
                    j_0 = j_0 + u32(256);
                }
                dec_0 = u32(1);
            }
        }
        else
        {
            dec_0 = u32(0);
        }
    }
    else
    {
        dec_0 = u32(0);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    j_0 = t_2;
    loop
    {
        if(j_0 < _S28)
        {
        }
        else
        {
            break;
        }
        var _S77 : u32 = u32(2) * (_S30 + j_0);
        var _S78 : u32 = _S77 + u32(1);
        var b1_0 : vec4<f32> = bonds_0[_S78];
        var _S79 : f32 = min(bonds_0[_S77].x + bonds_0[_S77].z, 0.99999898672103882f);
        if((bonds_0[_S77].y) >= (params_0.log_tau_0))
        {
            bad_0 = u32(1);
        }
        else
        {
            bad_0 = u32(0);
        }
        bonds_0[_S78] = vec4<f32>(_S79, (bitcast<f32>((bad_0))), b1_0.z, b1_0.w);
        j_0 = j_0 + u32(256);
    }
    bad_0 = u32(0);
    j_0 = t_2;
    loop
    {
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        var k_13 : u32 = _S26 + j_0;
        var _S80 : vec4<f32> = vec4<f32>(1.0f);
        if(!((abs(dot(node_0(k_13, u32(8)), _S80) + dot(node_0(k_13, u32(9)), _S80) + dot(node_0(k_13, u32(10)), _S80) + dot(node_0(k_13, u32(11)), _S80))) < 3.00000000549775576e+38f))
        {
            bad_0 = u32(1);
        }
        j_0 = j_0 + u32(256);
    }
    reduce_store_0(t_2, vec4<f32>(f32(bad_0), 0.0f, 0.0f, 0.0f), _S40, _S40, _S40);
    reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
    if((a_4.x) > 0.0f)
    {
        dec_0 = (dec_0 | (u32(4)));
    }
    if(t_2 == u32(0))
    {
        calls_0[_S22] = vec4<f32>(_S29, _S25, (bitcast<f32>((dec_0))), c4_0.w);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn learned_clear(@builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var t_3 : u32 = thread_3.x;
    var _S81 : u32 = ctl_0[u32(2)];
    var j_1 : u32 = t_3;
    loop
    {
        if(j_1 < _S81)
        {
        }
        else
        {
            break;
        }
        var _S82 : u32 = u32(14) * j_1;
        var w_4 : vec4<f32> = nodes_0[_S82 + u32(12)];
        var _S83 : bool;
        if((bitcast<u32>((nodes_0[_S82 + u32(4)].w))) == u32(0))
        {
            _S83 = (bitcast<u32>((w_4.x))) < (params_0.chunk_count_0);
        }
        else
        {
            _S83 = false;
        }
        if(_S83)
        {
            chunk_node_0[(bitcast<u32>((w_4.x)))] = u32(0);
        }
        j_1 = j_1 + u32(256);
    }
    if(t_3 == u32(0))
    {
        ctl_0[u32(0)] = u32(0);
        ctl_0[u32(18)] = u32(0);
    }
    return;
}

fn quat_mul_0( a_5 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_8 : Quat_0;
    r_8.w_0 = a_5.w_0 * o_0.w_0 - a_5.x_0 * o_0.x_0 - a_5.y_0 * o_0.y_0 - a_5.z_0 * o_0.z_0;
    r_8.x_0 = a_5.w_0 * o_0.x_0 + a_5.x_0 * o_0.w_0 + a_5.y_0 * o_0.z_0 - a_5.z_0 * o_0.y_0;
    r_8.y_0 = a_5.w_0 * o_0.y_0 - a_5.x_0 * o_0.z_0 + a_5.y_0 * o_0.w_0 + a_5.z_0 * o_0.x_0;
    r_8.z_0 = a_5.w_0 * o_0.z_0 + a_5.x_0 * o_0.y_0 - a_5.y_0 * o_0.x_0 + a_5.z_0 * o_0.w_0;
    return r_8;
}

fn quat_vec_0( q_7 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_7.x_0, q_7.y_0, q_7.z_0, q_7.w_0);
}

fn qmul_0( a_6 : vec4<f32>,  b_5 : vec4<f32>) -> vec4<f32>
{
    return quat_vec_0(quat_mul_0(quat_of_0(a_6), quat_of_0(b_5)));
}

fn body_rotation_zup_0( q_rubble_0 : vec4<f32>,  p_inv_0 : vec4<f32>) -> vec4<f32>
{
    return normalize(qmul_0(qmul_0(vec4<f32>(0.70710676908493042f, 0.0f, 0.0f, 0.70710676908493042f), q_rubble_0), p_inv_0));
}

fn to_zup_0( v_3 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_3.x, - v_3.z, v_3.y);
}

struct Frame_0
{
     q_8 : vec4<f32>,
     origin_0 : vec3<f32>,
     com_1 : vec3<f32>,
     v_4 : vec3<f32>,
     w_5 : vec3<f32>,
     body_0 : u32,
};

fn island_frame_0( i_0 : u32) -> Frame_0
{
    var f_0 : Frame_0;
    var _S84 : u32 = u32(3) * i_0;
    var _S85 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S84].x)));
    f_0.body_0 = _S85;
    f_0.q_8 = islands_0[i_0].rotation_0;
    f_0.origin_0 = islands_0[i_0].position_0.xyz;
    f_0.v_4 = islands_0[i_0].velocity_0.xyz;
    f_0.w_5 = islands_0[i_0].angular_velocity_0.xyz;
    if(_S85 != u32(4294967295))
    {
        var com_2 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S84 + u32(2)].xyz;
        var _S86 : vec4<f32> = body_rotation_zup_0(bodies_0[u32(7) * f_0.body_0 + u32(1)], ctab_0[params_0.engine_base_0 + _S84 + u32(1)]);
        f_0.q_8 = _S86;
        var _S87 : vec3<f32> = to_zup_0(bodies_0[u32(7) * f_0.body_0].xyz);
        f_0.com_1 = _S87;
        f_0.origin_0 = _S87 - rotate_0(quat_of_0(_S86), com_2);
        f_0.v_4 = to_zup_0(bodies_0[u32(7) * f_0.body_0 + u32(3)].xyz);
        f_0.w_5 = to_zup_0(bodies_0[u32(7) * f_0.body_0 + u32(4)].xyz);
    }
    else
    {
        f_0.com_1 = f_0.origin_0 + rotate_0(quat_of_0(f_0.q_8), islands_0[i_0].com_0.xyz);
    }
    return f_0;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn chunk_world_0( f_1 : Frame_0,  k_14 : u32) -> vec3<f32>
{
    return f_1.origin_0 + rotate_0(quat_of_0(f_1.q_8), chunks_0[k_14].center_0.xyz + state_0[u32(4) * k_14].xyz);
}

var<workgroup> g_lo_0 : array<vec3<f32>, i32(256)>;

var<workgroup> g_hi_0 : array<vec3<f32>, i32(256)>;

@compute
@workgroup_size(256, 1, 1)
fn learned_select(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var t0_0 : f32;
    var _S88 : bool;
    var i_1 : u32 = group_1.x;
    var t_4 : u32 = thread_4.x;
    var live_0 : u32 = group_uniform_0(counters_0[i32(0)], t_4);
    if(i_1 >= live_0)
    {
        return;
    }
    var _S89 : u32 = group_uniform_0(route_0[i32(0)], t_4);
    if(_S89 == u32(0))
    {
        if(t_4 == u32(0))
        {
            isl_0[i_1] = vec4<u32>(u32(0), u32(4294967295), u32(0), u32(0));
            islf_0[u32(3) * i_1 + u32(2)] = vec4<f32>(0.0f);
        }
        return;
    }
    var _S90 : u32 = islands_0[i_1].range_0.x;
    var _S91 : u32 = group_uniform_0(islands_0[i_1].range_0.y - _S90, t_4);
    var _S92 : Frame_0 = island_frame_0(i_1);
    var inf_0 : f32 = float_bits_0(u32(2139095040));
    var _S93 : vec3<f32> = vec3<f32>(- inf_0);
    var lo_0 : vec3<f32> = vec3<f32>(inf_0);
    var hi_0 : vec3<f32> = _S93;
    var c_3 : u32 = t_4;
    loop
    {
        if(c_3 < _S91)
        {
        }
        else
        {
            break;
        }
        var k_15 : u32 = _S90 + c_3;
        var p_3 : vec3<f32> = chunk_world_0(_S92, k_15);
        var _S94 : vec3<f32> = min(lo_0, p_3 - vec3<f32>(chunks_0[k_15].half_0.w));
        var _S95 : vec3<f32> = max(hi_0, p_3 + vec3<f32>(chunks_0[k_15].half_0.w));
        var c_4 : u32 = c_3 + u32(256);
        lo_0 = _S94;
        hi_0 = _S95;
        c_3 = c_4;
    }
    g_lo_0[t_4] = lo_0;
    g_hi_0[t_4] = hi_0;
    workgroupBarrier();
    var s_1 : u32 = u32(128);
    loop
    {
        if(s_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_4 < s_1)
        {
            var _S96 : u32 = t_4 + s_1;
            g_lo_0[t_4] = min(g_lo_0[t_4], g_lo_0[_S96]);
            g_hi_0[t_4] = max(g_hi_0[t_4], g_hi_0[_S96]);
        }
        workgroupBarrier();
        s_1 = (s_1 >> (u32(1)));
    }
    if(t_4 != u32(0))
    {
        return;
    }
    var lo_1 : vec3<f32> = g_lo_0[i32(0)];
    var hi_1 : vec3<f32> = g_hi_0[i32(0)];
    var _S97 : vec3<f32> = vec3<f32>(0.0f);
    var launched_0 : bool = false;
    var launch_0 : vec3<f32> = _S97;
    var k_16 : u32 = u32(0);
    var mask_0 : u32 = u32(0);
    loop
    {
        if(k_16 < (min(params_0.impactor_count_0, u32(32))))
        {
        }
        else
        {
            break;
        }
        var a_7 : vec4<f32> = imps_0[u32(3) * k_16];
        var _S98 : u32 = u32(7) * (bitcast<u32>((a_7.x)));
        var _S99 : vec3<f32> = to_zup_0(bodies_0[_S98].xyz);
        var rel_0 : vec3<f32> = to_zup_0(bodies_0[_S98 + u32(3)].xyz) - _S92.v_4;
        var r_9 : f32 = a_7.y;
        var _S100 : bool;
        if(!((length(rel_0)) > (params_0.speed_0)))
        {
            _S100 = true;
        }
        else
        {
            _S100 = _S91 == u32(0);
        }
        var launch_1 : vec3<f32>;
        var launched_1 : bool;
        if(_S100)
        {
            launched_1 = launched_0;
            launch_1 = launch_0;
            var _S101 : u32 = k_16 + u32(1);
            launched_0 = launched_1;
            launch_0 = launch_1;
            k_16 = _S101;
            continue;
        }
        var _S102 : vec3<f32> = vec3<f32>(r_9);
        var _S103 : vec3<f32> = lo_1 - _S102;
        var _S104 : vec3<f32> = hi_1 + _S102;
        t0_0 = 0.0f;
        var t1_1 : f32 = params_0.dt_0;
        var ax_0 : u32 = u32(0);
        loop
        {
            if(ax_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S105 : u32 = ax_0;
            if((abs(rel_0[ax_0])) < 9.999999960041972e-13f)
            {
                var _S106 : u32 = ax_0;
                if((_S99[ax_0]) < (_S103[ax_0]))
                {
                    _S88 = true;
                }
                else
                {
                    _S88 = (_S99[_S106]) > (_S104[ax_0]);
                }
                var t0_1 : f32;
                if(_S88)
                {
                    t0_1 = inf_0;
                }
                else
                {
                    t0_1 = t0_0;
                }
                t0_0 = t0_1;
                ax_0 = ax_0 + u32(1);
                continue;
            }
            var ta_2 : f32 = (_S103[ax_0] - _S99[ax_0]) / rel_0[_S105];
            var tb_2 : f32 = (_S104[ax_0] - _S99[ax_0]) / rel_0[_S105];
            var _S107 : f32 = min(t1_1, max(ta_2, tb_2));
            t0_0 = max(t0_0, min(ta_2, tb_2));
            t1_1 = _S107;
            ax_0 = ax_0 + u32(1);
        }
        var mask_1 : u32;
        if(t0_0 <= t1_1)
        {
            var mask_2 : u32 = (mask_0 | (((u32(1) << (k_16)))));
            if(!launched_0)
            {
                _S88 = (((islands_0[i_1].info_0.x) & (u32(1)))) == u32(0);
            }
            else
            {
                _S88 = false;
            }
            if(_S88)
            {
                var _S108 : vec3<f32> = vec3<f32>(2.0f) * rel_0 * vec3<f32>(params_0.dt_0);
                launched_1 = true;
                launch_1 = _S108;
            }
            else
            {
                launched_1 = launched_0;
                launch_1 = launch_0;
            }
            mask_1 = mask_2;
        }
        else
        {
            launched_1 = launched_0;
            launch_1 = launch_0;
            mask_1 = mask_0;
        }
        mask_0 = mask_1;
        var _S101 : u32 = k_16 + u32(1);
        launched_0 = launched_1;
        launch_0 = launch_1;
        k_16 = _S101;
    }
    isl_0[i_1] = vec4<u32>(mask_0, u32(4294967295), u32(0), u32(0));
    var _S109 : u32 = u32(3) * i_1;
    islf_0[_S109] = vec4<f32>(lo_1, 0.0f);
    islf_0[_S109 + u32(1)] = vec4<f32>(hi_1, 0.0f);
    if(launched_0)
    {
        t0_0 = 1.0f;
    }
    else
    {
        t0_0 = 0.0f;
    }
    islf_0[_S109 + u32(2)] = vec4<f32>(launch_0, t0_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn learned_debris(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var j_2 : u32 = id_0.x;
    var live_1 : u32 = counters_0[i32(0)];
    var _S110 : bool;
    if((params_0.debris_0) == u32(0))
    {
        _S110 = true;
    }
    else
    {
        _S110 = j_2 >= live_1;
    }
    if(_S110)
    {
        _S110 = true;
    }
    else
    {
        _S110 = route_0[i32(0)] == u32(0);
    }
    if(_S110)
    {
        _S110 = true;
    }
    else
    {
        _S110 = (isl_0[j_2].x) != u32(0);
    }
    if(_S110)
    {
        return;
    }
    var _S111 : u32 = u32(3) * j_2;
    var lo_2 : vec3<f32> = islf_0[_S111].xyz;
    var hi_2 : vec3<f32> = islf_0[_S111 + u32(1)].xyz;
    if(!((lo_2.x) <= (hi_2.x)))
    {
        return;
    }
    var i_2 : u32 = u32(0);
    var mask_3 : u32 = u32(0);
    loop
    {
        if(i_2 < live_1)
        {
        }
        else
        {
            break;
        }
        var _S112 : u32 = u32(3) * i_2;
        var l_0 : vec4<f32> = islf_0[_S112 + u32(2)];
        if((islf_0[_S112 + u32(2)].w) == 0.0f)
        {
            _S110 = true;
        }
        else
        {
            _S110 = i_2 == j_2;
        }
        if(_S110)
        {
            i_2 = i_2 + u32(1);
            continue;
        }
        var a_8 : vec3<f32> = islf_0[_S112].xyz;
        var b_6 : vec3<f32> = islf_0[_S112 + u32(1)].xyz;
        var _S113 : vec3<f32> = l_0.xyz;
        var _S114 : vec3<f32> = vec3<f32>(0.00100000004749745f);
        var shi_0 : vec3<f32> = max(b_6, b_6 + _S113) + _S114;
        var _S115 : bool;
        if((all(((min(a_8, a_8 + _S113) - _S114) <= hi_2))))
        {
            _S115 = (all((lo_2 <= shi_0)));
        }
        else
        {
            _S115 = false;
        }
        var mask_4 : u32;
        if(_S115)
        {
            mask_4 = (mask_3 | ((isl_0[i_2].x)));
        }
        else
        {
            mask_4 = mask_3;
        }
        mask_3 = mask_4;
        i_2 = i_2 + u32(1);
    }
    if(mask_3 != u32(0))
    {
        isl_0[j_2] = vec4<u32>(mask_3, u32(4294967295), u32(0), u32(0));
    }
    return;
}

var<workgroup> g_label_0 : array<u32, i32(32)>;

var<workgroup> g_used_0 : u32;

fn lowest_bit_0( m_5 : u32) -> u32
{
    return firstbitlow_0(m_5);
}

var<workgroup> g_call_of_0 : array<u32, i32(32)>;

var<workgroup> g_mask_of_0 : array<u32, i32(64)>;

var<workgroup> g_cnt_isl_0 : array<u32, i32(64)>;

var<workgroup> g_cnt_chunk_0 : array<u32, i32(64)>;

var<workgroup> g_cnt_bond_0 : array<u32, i32(64)>;

var<workgroup> g_cnt_imp_0 : array<u32, i32(64)>;

@compute
@workgroup_size(1024, 1, 1)
fn learned_compact(@builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var _S116 : u32;
    var t_5 : u32 = thread_5.x;
    var _S117 : u32 = group_uniform_0(counters_0[i32(0)], t_5);
    var _S118 : u32 = min(params_0.impactor_count_0, u32(32));
    var i_3 : u32 = t_5;
    loop
    {
        if(i_3 < _S117)
        {
        }
        else
        {
            break;
        }
        status_0[u32(16) + i_3] = (status_0[u32(16) + i_3] & (u32(4294963199)));
        i_3 = i_3 + u32(1024);
    }
    if(t_5 < u32(32))
    {
        g_label_0[t_5] = t_5;
    }
    var _S119 : bool = t_5 == u32(0);
    if(_S119)
    {
        g_used_0 = u32(0);
        ctl_0[u32(1)] = u32(0);
        ctl_0[u32(2)] = u32(0);
        ctl_0[u32(3)] = u32(0);
        ctl_0[u32(9)] = u32(0);
        ctl_0[u32(8)] = u32(0);
        ctl_0[u32(5)] = u32(0);
        ctl_0[u32(6)] = u32(0);
        ctl_0[u32(7)] = u32(0);
        ctl_0[u32(14)] = u32(0);
        ctl_0[u32(15)] = u32(0);
        ctl_0[u32(16)] = u32(0);
        ctl_0[u32(17)] = u32(0);
    }
    if(_S119)
    {
        const _S120 : vec4<u32> = vec4<u32>(u32(0), u32(1), u32(1), u32(0));
        largs_0[i32(0)] = _S120;
        largs_0[i32(1)] = _S120;
        largs_0[i32(2)] = _S120;
    }
    workgroupBarrier();
    var _S121 : u32 = group_uniform_0(route_0[i32(0)], t_5);
    if(_S121 == u32(0))
    {
        return;
    }
    var over_0 : u32;
    var routed_0 : u32;
    var g_1 : u32;
    var node_at_0 : u32;
    var bond_at_0 : u32;
    var changed_0 : bool;
    if(_S119)
    {
        i_3 = u32(0);
        over_0 = u32(0);
        loop
        {
            if(i_3 < _S117)
            {
            }
            else
            {
                break;
            }
            var used_0 : u32 = (over_0 | ((isl_0[i_3].x)));
            i_3 = i_3 + u32(1);
            over_0 = used_0;
        }
        g_used_0 = over_0;
        routed_0 = u32(0);
        loop
        {
            if(routed_0 < u32(32))
            {
            }
            else
            {
                break;
            }
            changed_0 = false;
            i_3 = u32(0);
            loop
            {
                if(i_3 < _S117)
                {
                }
                else
                {
                    break;
                }
                var m_6 : u32 = isl_0[i_3].x;
                var changed_1 : bool;
                if((countOneBits(m_6)) < u32(2))
                {
                    changed_1 = changed_0;
                    var _S122 : u32 = i_3 + u32(1);
                    changed_0 = changed_1;
                    i_3 = _S122;
                    continue;
                }
                g_1 = u32(32);
                node_at_0 = m_6;
                loop
                {
                    if(node_at_0 != u32(0))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S123 : u32 = min(g_1, g_label_0[lowest_bit_0(node_at_0)]);
                    var mm_0 : u32 = (node_at_0 & ((node_at_0 - u32(1))));
                    g_1 = _S123;
                    node_at_0 = mm_0;
                }
                changed_1 = changed_0;
                bond_at_0 = m_6;
                loop
                {
                    if(bond_at_0 != u32(0))
                    {
                    }
                    else
                    {
                        break;
                    }
                    if((g_label_0[lowest_bit_0(bond_at_0)]) > g_1)
                    {
                        g_label_0[lowest_bit_0(bond_at_0)] = g_1;
                        changed_1 = true;
                    }
                    bond_at_0 = (bond_at_0 & ((bond_at_0 - u32(1))));
                }
                var _S122 : u32 = i_3 + u32(1);
                changed_0 = changed_1;
                i_3 = _S122;
            }
            if(!changed_0)
            {
                break;
            }
            routed_0 = routed_0 + u32(1);
        }
    }
    workgroupBarrier();
    if(_S119)
    {
        var _S124 : u32 = g_used_0;
        over_0 = u32(0);
        routed_0 = u32(0);
        loop
        {
            if(over_0 < _S118)
            {
            }
            else
            {
                break;
            }
            g_call_of_0[over_0] = u32(4294967295);
            if((g_label_0[over_0]) != over_0)
            {
                over_0 = over_0 + u32(1);
                continue;
            }
            g_1 = u32(0);
            node_at_0 = u32(0);
            loop
            {
                if(g_1 < _S118)
                {
                }
                else
                {
                    break;
                }
                if((g_label_0[g_1]) == over_0)
                {
                    node_at_0 = (node_at_0 | (((u32(1) << (g_1)))));
                }
                g_1 = g_1 + u32(1);
            }
            var _S125 : u32 = (node_at_0 & (_S124));
            if(_S125 == u32(0))
            {
                over_0 = over_0 + u32(1);
                continue;
            }
            if(routed_0 < u32(64))
            {
                g_mask_of_0[routed_0] = _S125;
                g_cnt_isl_0[routed_0] = u32(0);
                g_cnt_chunk_0[routed_0] = u32(0);
                g_cnt_bond_0[routed_0] = u32(0);
                g_cnt_imp_0[routed_0] = countOneBits(_S125);
            }
            g_call_of_0[over_0] = routed_0;
            routed_0 = routed_0 + u32(1);
            over_0 = over_0 + u32(1);
        }
        ctl_0[u32(1)] = routed_0;
    }
    workgroupBarrier();
    var ncalls_0 : u32 = group_uniform_0(ctl_0[u32(1)], t_5);
    if(_S119)
    {
        i_3 = u32(0);
        loop
        {
            if(i_3 < _S117)
            {
            }
            else
            {
                break;
            }
            var e_0 : vec4<u32> = isl_0[i_3];
            var _S126 : u32 = isl_0[i_3].x;
            if(_S126 != u32(0))
            {
                isl_0[i_3] = vec4<u32>(_S126, g_call_of_0[g_label_0[lowest_bit_0(_S126)]], e_0.z, u32(0));
            }
            i_3 = i_3 + u32(1);
        }
    }
    if(t_5 != u32(0))
    {
        return;
    }
    if(ncalls_0 > (min(params_0.max_calls_0, u32(64))))
    {
        over_0 = u32(1);
    }
    else
    {
        over_0 = u32(0);
    }
    i_3 = u32(0);
    routed_0 = u32(0);
    loop
    {
        if(i_3 < _S117)
        {
        }
        else
        {
            break;
        }
        var e_1 : vec4<u32> = isl_0[i_3];
        if((isl_0[i_3].x) == u32(0))
        {
            changed_0 = true;
        }
        else
        {
            changed_0 = (e_1.y) >= u32(64);
        }
        if(changed_0)
        {
            i_3 = i_3 + u32(1);
            continue;
        }
        var _S127 : u32 = e_1.y;
        g_cnt_isl_0[_S127] = g_cnt_isl_0[_S127] + u32(1);
        g_cnt_chunk_0[_S127] = g_cnt_chunk_0[_S127] + (islands_0[i_3].range_0.y - islands_0[i_3].range_0.x);
        g_cnt_bond_0[_S127] = g_cnt_bond_0[_S127] + e_1.z;
        routed_0 = routed_0 + u32(1);
        i_3 = i_3 + u32(1);
    }
    g_1 = u32(0);
    node_at_0 = u32(0);
    loop
    {
        var _S128 : u32 = min(ncalls_0, u32(64));
        _S116 = _S128;
        if(g_1 < _S128)
        {
        }
        else
        {
            break;
        }
        var chunks_n_0 : u32 = g_cnt_chunk_0[g_1];
        var nodes_n_0 : u32 = g_cnt_chunk_0[g_1] + g_cnt_imp_0[g_1];
        if((g_cnt_chunk_0[g_1]) > (params_0.max_call_chunks_0))
        {
            over_0 = (over_0 | (u32(16)));
        }
        var _S129 : u32 = u32(8) * g_1;
        calls_0[_S129 + u32(2)] = vec4<f32>(0.0f, 0.0f, (bitcast<f32>((node_at_0))), (bitcast<f32>((nodes_n_0))));
        calls_0[_S129 + u32(3)] = vec4<f32>(params_0.gravity_0.xyz, (bitcast<f32>((chunks_n_0))));
        var _S130 : f32 = (bitcast<f32>((u32(0))));
        calls_0[_S129 + u32(4)] = vec4<f32>(_S130, (bitcast<f32>((g_cnt_bond_0[g_1]))), 0.0f, 0.0f);
        calls_0[_S129 + u32(5)] = vec4<f32>(0.0f, 0.0f, 0.0f, (bitcast<f32>((g_mask_of_0[g_1]))));
        calls_0[_S129 + u32(6)] = vec4<f32>(_S130, (bitcast<f32>((g_cnt_isl_0[g_1]))), 0.0f, 0.0f);
        g_cnt_chunk_0[g_1] = node_at_0;
        g_cnt_bond_0[g_1] = u32(0);
        g_cnt_isl_0[g_1] = u32(0);
        var node_at_1 : u32 = node_at_0 + nodes_n_0;
        g_1 = g_1 + u32(1);
        node_at_0 = node_at_1;
    }
    g_1 = u32(0);
    bond_at_0 = u32(0);
    var cisl_at_0 : u32 = u32(0);
    var chunk_base_0 : u32 = u32(0);
    var imp_base_0 : u32 = u32(0);
    loop
    {
        if(g_1 < _S116)
        {
        }
        else
        {
            break;
        }
        var _S131 : u32 = u32(8) * g_1;
        var _S132 : u32 = _S131 + u32(4);
        calls_0[_S132][i32(0)] = (bitcast<f32>((bond_at_0)));
        var _S133 : u32 = _S131 + u32(6);
        calls_0[_S133] = vec4<f32>((bitcast<f32>((cisl_at_0))), calls_0[_S133].y, (bitcast<f32>((chunk_base_0))), (bitcast<f32>((imp_base_0))));
        var chunk_base_1 : u32 = chunk_base_0 + (bitcast<u32>((calls_0[_S131 + u32(3)].w)));
        var imp_base_1 : u32 = imp_base_0 + countOneBits((bitcast<u32>((calls_0[_S131 + u32(5)].w))));
        g_cnt_bond_0[g_1] = bond_at_0;
        g_cnt_isl_0[g_1] = cisl_at_0;
        var bond_at_1 : u32 = bond_at_0 + (bitcast<u32>((calls_0[_S132].y)));
        var cisl_at_1 : u32 = cisl_at_0 + (bitcast<u32>((calls_0[_S133].y)));
        g_1 = g_1 + u32(1);
        bond_at_0 = bond_at_1;
        cisl_at_0 = cisl_at_1;
        chunk_base_0 = chunk_base_1;
        imp_base_0 = imp_base_1;
    }
    if(node_at_0 > (params_0.max_nodes_0))
    {
        over_0 = (over_0 | (u32(2)));
    }
    if(bond_at_0 > (params_0.max_bonds_0))
    {
        over_0 = (over_0 | (u32(4)));
    }
    if(cisl_at_0 > (params_0.max_cisl_0))
    {
        over_0 = (over_0 | (u32(8)));
    }
    if(over_0 != u32(0))
    {
        ctl_0[u32(8)] = over_0;
        ctl_0[u32(1)] = u32(0);
        return;
    }
    i_3 = u32(0);
    loop
    {
        if(i_3 < _S117)
        {
        }
        else
        {
            break;
        }
        var e_2 : vec4<u32> = isl_0[i_3];
        if((isl_0[i_3].x) == u32(0))
        {
            changed_0 = true;
        }
        else
        {
            changed_0 = (e_2.y) >= u32(64);
        }
        if(changed_0)
        {
            i_3 = i_3 + u32(1);
            continue;
        }
        var g_2 : u32 = e_2.y;
        var at_0 : u32 = g_cnt_isl_0[g_2];
        var nchunks_0 : u32 = islands_0[i_3].range_0.y - islands_0[i_3].range_0.x;
        var _S134 : u32 = u32(2) * g_cnt_isl_0[g_2];
        cisl_0[_S134] = vec4<u32>(i_3, g_cnt_chunk_0[g_2], nchunks_0, g_2);
        var _S135 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + u32(3) * i_3].x)));
        var _S136 : u32 = g_cnt_bond_0[g_2];
        var _S137 : u32 = e_2.z;
        if((((islands_0[i_3].info_0.x) & (u32(1)))) != u32(0))
        {
            over_0 = u32(1);
        }
        else
        {
            over_0 = u32(0);
        }
        cisl_0[_S134 + u32(1)] = vec4<u32>(_S135, _S136, _S137, over_0);
        g_cnt_isl_0[g_2] = at_0 + u32(1);
        g_cnt_chunk_0[g_2] = g_cnt_chunk_0[g_2] + nchunks_0;
        g_cnt_bond_0[g_2] = g_cnt_bond_0[g_2] + _S137;
        i_3 = i_3 + u32(1);
    }
    ctl_0[u32(2)] = node_at_0;
    ctl_0[u32(3)] = bond_at_0;
    ctl_0[u32(9)] = cisl_at_0;
    ctl_0[u32(17)] = routed_0;
    largs_0[i32(0)] = vec4<u32>(ncalls_0, u32(1), u32(1), u32(0));
    largs_0[i32(1)] = vec4<u32>(cisl_at_0, u32(1), u32(1), u32(0));
    largs_0[i32(2)] = vec4<u32>((cisl_at_0 + u32(63)) / u32(64), u32(1), u32(1), u32(0));
    return;
}

fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S138 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S138 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S138 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S138 = false;
        }
    }
    return _S138;
}

struct JointState_0
{
     damage_0 : f32,
     crush_0 : f32,
     kappa_0 : f32,
     kappa_c_0 : f32,
     ductility_0 : f32,
     ductility_c_0 : f32,
     life_0 : f32,
     plastic_x_0 : f32,
     plastic_y_0 : f32,
     plastic_t_0 : f32,
     rebar_plastic_0 : f32,
     rebar_slip0_0 : f32,
     rebar_slip1_0 : f32,
     rebar_work_0 : f32,
     rebar_broken_0 : f32,
     strain_rate_0 : f32,
     governing_stress_0 : f32,
     dissipated_0 : f32,
     utilization_0 : f32,
     mode_0 : u32,
};

fn connected_1( st_1 : JointState_0,  has_rebar_1 : bool) -> bool
{
    var _S139 : bool;
    if((st_1.damage_0) < 1.0f)
    {
        _S139 = true;
    }
    else
    {
        if(has_rebar_1)
        {
            _S139 = (st_1.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S139 = false;
        }
    }
    return _S139;
}

var<workgroup> g_scan_0 : array<u32, i32(256)>;

@compute
@workgroup_size(256, 1, 1)
fn learned_count_bonds(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var i_4 : u32 = group_2.x;
    var t_6 : u32 = thread_6.x;
    var live_2 : u32 = group_uniform_0(counters_0[i32(0)], t_6);
    if(i_4 >= live_2)
    {
        return;
    }
    var e_3 : vec4<u32> = isl_0[i_4];
    var _S140 : bool;
    if(route_0[i32(0)] == u32(0))
    {
        _S140 = true;
    }
    else
    {
        _S140 = (e_3.x) == u32(0);
    }
    var bi_0 : u32;
    if(_S140)
    {
        bi_0 = u32(1);
    }
    else
    {
        bi_0 = u32(0);
    }
    var _S141 : u32 = group_uniform_0(bi_0, t_6);
    if(_S141 != u32(0))
    {
        return;
    }
    var _S142 : vec4<u32> = islands_0[i_4].range_0;
    bi_0 = islands_0[i_4].range_0.z + t_6;
    var nb_0 : u32 = u32(0);
    loop
    {
        if(bi_0 < (_S142.w))
        {
        }
        else
        {
            break;
        }
        var _S143 : JointState_std430_0 = bond_dyn_0[bi_0].js_0;
        var _S144 : bool = connected_0(&(_S143), (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f);
        if(_S144)
        {
            nb_0 = nb_0 + u32(1);
        }
        bi_0 = bi_0 + u32(256);
    }
    g_scan_0[t_6] = nb_0;
    workgroupBarrier();
    var s_2 : u32 = u32(128);
    loop
    {
        if(s_2 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_6 < s_2)
        {
            g_scan_0[t_6] = g_scan_0[t_6] + g_scan_0[t_6 + s_2];
        }
        workgroupBarrier();
        s_2 = (s_2 >> (u32(1)));
    }
    if(t_6 == u32(0))
    {
        isl_0[i_4] = vec4<u32>(e_3.x, e_3.y, g_scan_0[i32(0)], e_3.w);
    }
    return;
}

fn matrix_quat_0( m_7 : mat3x3<f32>) -> vec4<f32>
{
    var tr_0 : f32 = m_7[i32(0)][i32(0)] + m_7[i32(1)][i32(1)] + m_7[i32(2)][i32(2)];
    var q_9 : vec4<f32>;
    if(tr_0 > 0.0f)
    {
        var s_3 : f32 = sqrt(tr_0 + 1.0f) * 2.0f;
        q_9 = vec4<f32>((m_7[i32(2)][i32(1)] - m_7[i32(1)][i32(2)]) / s_3, (m_7[i32(0)][i32(2)] - m_7[i32(2)][i32(0)]) / s_3, (m_7[i32(1)][i32(0)] - m_7[i32(0)][i32(1)]) / s_3, 0.25f * s_3);
    }
    else
    {
        var _S145 : bool;
        if((m_7[i32(0)][i32(0)]) > (m_7[i32(1)][i32(1)]))
        {
            _S145 = (m_7[i32(0)][i32(0)]) > (m_7[i32(2)][i32(2)]);
        }
        else
        {
            _S145 = false;
        }
        if(_S145)
        {
            var s_4 : f32 = sqrt(1.0f + m_7[i32(0)][i32(0)] - m_7[i32(1)][i32(1)] - m_7[i32(2)][i32(2)]) * 2.0f;
            q_9 = vec4<f32>(0.25f * s_4, (m_7[i32(0)][i32(1)] + m_7[i32(1)][i32(0)]) / s_4, (m_7[i32(0)][i32(2)] + m_7[i32(2)][i32(0)]) / s_4, (m_7[i32(2)][i32(1)] - m_7[i32(1)][i32(2)]) / s_4);
        }
        else
        {
            if((m_7[i32(1)][i32(1)]) > (m_7[i32(2)][i32(2)]))
            {
                var s_5 : f32 = sqrt(1.0f + m_7[i32(1)][i32(1)] - m_7[i32(0)][i32(0)] - m_7[i32(2)][i32(2)]) * 2.0f;
                q_9 = vec4<f32>((m_7[i32(0)][i32(1)] + m_7[i32(1)][i32(0)]) / s_5, 0.25f * s_5, (m_7[i32(1)][i32(2)] + m_7[i32(2)][i32(1)]) / s_5, (m_7[i32(0)][i32(2)] - m_7[i32(2)][i32(0)]) / s_5);
            }
            else
            {
                var s_6 : f32 = sqrt(1.0f + m_7[i32(2)][i32(2)] - m_7[i32(0)][i32(0)] - m_7[i32(1)][i32(1)]) * 2.0f;
                q_9 = vec4<f32>((m_7[i32(0)][i32(2)] + m_7[i32(2)][i32(0)]) / s_6, (m_7[i32(1)][i32(2)] + m_7[i32(2)][i32(1)]) / s_6, 0.25f * s_6, (m_7[i32(1)][i32(0)] - m_7[i32(0)][i32(1)]) / s_6);
            }
        }
    }
    return normalize(q_9);
}

@compute
@workgroup_size(256, 1, 1)
fn learned_gather_chunks(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    var e_4 : u32 = group_3.x;
    var t_7 : u32 = thread_7.x;
    if(e_4 >= ctl_0[u32(9)])
    {
        return;
    }
    var _S146 : u32 = u32(2) * e_4;
    var c0_1 : vec4<u32> = cisl_0[_S146];
    var _S147 : vec4<u32> = cisl_0[_S146 + u32(1)];
    var i_5 : u32 = cisl_0[_S146].x;
    var _S148 : Frame_0 = island_frame_0(i_5);
    var c_5 : u32 = t_7;
    loop
    {
        if(c_5 < (c0_1.z))
        {
        }
        else
        {
            break;
        }
        var k_17 : u32 = islands_0[i_5].range_0.x + c_5;
        var n_1 : u32 = c0_1.y + c_5;
        var x_2 : vec3<f32> = chunk_world_0(_S148, k_17);
        var rest_0 : mat3x3<f32> = transpose(mat3x3<f32>(chunks_0[k_17].crot0_0.xyz, chunks_0[k_17].crot1_0.xyz, chunks_0[k_17].crot2_0.xyz));
        var ic_0 : mat3x3<f32> = (((rest_0) * ((((mat3x3<f32>(chunks_0[k_17].inertia0_1.xyz, chunks_0[k_17].inertia1_1.xyz, chunks_0[k_17].inertia2_1.xyz)) * (transpose(rest_0)))))));
        var q_10 : vec4<f32> = normalize(qmul_0(_S148.q_8, matrix_quat_0(rest_0)));
        var v_5 : vec3<f32> = _S148.v_4 + cross(_S148.w_5, x_2 - _S148.com_1);
        var _S149 : vec4<u32> = chunks_0[k_17].info_1;
        var anchored_0 : bool = (chunks_0[k_17].info_1.x) != u32(0);
        var _S150 : u32 = u32(14) * n_1;
        nodes_0[_S150] = vec4<f32>(x_2, chunks_0[k_17].center_0.w);
        nodes_0[_S150 + u32(1)] = q_10;
        var _S151 : f32;
        if(anchored_0)
        {
            _S151 = 1.0f;
        }
        else
        {
            _S151 = 0.0f;
        }
        nodes_0[_S150 + u32(2)] = vec4<f32>(ic_0[i32(0)][i32(0)], ic_0[i32(1)][i32(1)], ic_0[i32(2)][i32(2)], _S151);
        nodes_0[_S150 + u32(3)] = vec4<f32>(v_5, (bitcast<f32>((c0_1.w))));
        nodes_0[_S150 + u32(4)] = vec4<f32>(_S148.w_5, (bitcast<f32>((u32(0)))));
        nodes_0[_S150 + u32(12)] = vec4<f32>((bitcast<f32>((k_17))), (bitcast<f32>((_S147.x))), (bitcast<f32>((_S149.z))), (bitcast<f32>((_S149.w))));
        chunk_node_0[k_17] = n_1 + u32(1);
        c_5 = c_5 + u32(256);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn learned_gather_bonds(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    var e_5 : u32 = group_4.x;
    var t_8 : u32 = thread_8.x;
    var base_0 : u32;
    if(e_5 >= ctl_0[u32(9)])
    {
        base_0 = u32(1);
    }
    else
    {
        base_0 = u32(0);
    }
    var _S152 : u32 = group_uniform_0(base_0, t_8);
    if(_S152 != u32(0))
    {
        return;
    }
    var _S153 : u32 = u32(2) * e_5;
    var c0_2 : vec4<u32> = cisl_0[_S153];
    var _S154 : vec4<u32> = cisl_0[_S153 + u32(1)];
    var z_2 : u32 = islands_0[cisl_0[_S153].x].range_0.z;
    var _S155 : u32 = group_uniform_0(islands_0[cisl_0[_S153].x].range_0.w, t_8);
    var _S156 : u32 = group_uniform_0(z_2, t_8);
    base_0 = _S156;
    var carry_0 : u32 = u32(0);
    loop
    {
        if(base_0 < _S155)
        {
        }
        else
        {
            break;
        }
        var bi_1 : u32 = base_0 + t_8;
        var live_3 : bool;
        if(bi_1 < _S155)
        {
            var _S157 : JointState_std430_0 = bond_dyn_0[bi_1].js_0;
            var _S158 : bool = connected_0(&(_S157), (bond_static_0[bi_1].law_0.stiff1_0.w) != 0.0f);
            live_3 = _S158;
        }
        else
        {
            live_3 = false;
        }
        var _S159 : u32;
        if(live_3)
        {
            _S159 = u32(1);
        }
        else
        {
            _S159 = u32(0);
        }
        g_scan_0[t_8] = _S159;
        workgroupBarrier();
        var step_0 : u32 = u32(1);
        loop
        {
            if(step_0 < u32(256))
            {
            }
            else
            {
                break;
            }
            var add_0 : u32;
            if(t_8 >= step_0)
            {
                add_0 = g_scan_0[t_8 - step_0];
            }
            else
            {
                add_0 = u32(0);
            }
            workgroupBarrier();
            g_scan_0[t_8] = g_scan_0[t_8] + add_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        if(live_3)
        {
            var _S160 : u32 = u32(2) * (_S154.y + carry_0 + g_scan_0[t_8] - u32(1));
            bonds_0[_S160] = vec4<f32>(bond_dyn_0[bi_1].js_0.damage_0, -30.0f, 0.0f, (bitcast<f32>((u32(1)))));
            bonds_0[_S160 + u32(1)] = vec4<f32>(0.0f, (bitcast<f32>((u32(0)))), (bitcast<f32>((bi_1))), (bitcast<f32>((c0_2.w))));
        }
        var total_0 : u32 = g_scan_0[u32(255)];
        workgroupBarrier();
        var carry_1 : u32 = carry_0 + total_0;
        base_0 = base_0 + u32(256);
        carry_0 = carry_1;
    }
    return;
}

@compute
@workgroup_size(32, 1, 1)
fn learned_gather_impactors(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    var g_3 : u32 = group_5.x;
    var k_18 : u32 = thread_9.x;
    if(g_3 >= ctl_0[u32(1)])
    {
        return;
    }
    var _S161 : u32 = u32(8) * g_3;
    var mask_5 : u32 = (bitcast<u32>((calls_0[_S161 + u32(5)].w)));
    var _S162 : u32 = (u32(1) << (k_18));
    if(((mask_5 & (_S162))) == u32(0))
    {
        return;
    }
    var _S163 : u32 = u32(3) * k_18;
    var a_9 : vec4<f32> = imps_0[_S163];
    var b_7 : vec4<f32> = imps_0[_S163 + u32(1)];
    var body_1 : u32 = (bitcast<u32>((a_9.x)));
    var _S164 : u32 = u32(7) * body_1;
    var q_11 : vec4<f32> = body_rotation_zup_0(bodies_0[_S164 + u32(1)], vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f));
    var _S165 : u32 = u32(14) * ((bitcast<u32>((calls_0[_S161 + u32(2)].z))) + (bitcast<u32>((calls_0[_S161 + u32(3)].w))) + countOneBits((mask_5 & ((_S162 - u32(1))))));
    nodes_0[_S165] = vec4<f32>(to_zup_0(bodies_0[_S164].xyz), a_9.w);
    nodes_0[_S165 + u32(1)] = q_11;
    nodes_0[_S165 + u32(2)] = vec4<f32>(b_7.xyz, 0.0f);
    nodes_0[_S165 + u32(3)] = vec4<f32>(to_zup_0(bodies_0[_S164 + u32(3)].xyz), (bitcast<f32>((g_3))));
    nodes_0[_S165 + u32(4)] = vec4<f32>(to_zup_0(bodies_0[_S164 + u32(4)].xyz), (bitcast<f32>((u32(1)))));
    var _S166 : f32 = (bitcast<f32>((u32(4294967295))));
    nodes_0[_S165 + u32(12)] = vec4<f32>((bitcast<f32>((k_18))), (bitcast<f32>((body_1))), _S166, _S166);
    return;
}

var<workgroup> g_sort_0 : array<f32, i32(4096)>;

fn sort_group_0( t_9 : u32,  n_2 : u32)
{
    var m_8 : u32 = u32(1);
    loop
    {
        if(m_8 < n_2)
        {
        }
        else
        {
            break;
        }
        m_8 = (m_8 << (u32(1)));
    }
    var _S167 : f32 = float_bits_0(u32(2139095040));
    var j_3 : u32 = n_2 + t_9;
    loop
    {
        if(j_3 < m_8)
        {
        }
        else
        {
            break;
        }
        g_sort_0[j_3] = _S167;
        j_3 = j_3 + u32(256);
    }
    workgroupBarrier();
    var size_0 : u32 = u32(2);
    loop
    {
        if(size_0 <= m_8)
        {
        }
        else
        {
            break;
        }
        var stride_0 : u32 = (size_0 >> (u32(1)));
        loop
        {
            if(stride_0 > u32(0))
            {
            }
            else
            {
                break;
            }
            j_3 = t_9;
            loop
            {
                if(j_3 < m_8)
                {
                }
                else
                {
                    break;
                }
                var p_4 : u32 = (j_3 ^ (stride_0));
                if(p_4 > j_3)
                {
                    var a_10 : f32 = g_sort_0[j_3];
                    var b_8 : f32 = g_sort_0[p_4];
                    if(((g_sort_0[j_3]) > (g_sort_0[p_4])) == (((j_3 & (size_0))) == u32(0)))
                    {
                        g_sort_0[j_3] = b_8;
                        g_sort_0[p_4] = a_10;
                    }
                }
                j_3 = j_3 + u32(256);
            }
            workgroupBarrier();
            stride_0 = (stride_0 >> (u32(1)));
        }
        size_0 = (size_0 << (u32(1)));
    }
    return;
}

fn median_sorted_0( n_3 : u32) -> f32
{
    var k_19 : u32 = n_3 / u32(2);
    var _S168 : f32;
    if((n_3 % u32(2)) == u32(1))
    {
        _S168 = g_sort_0[k_19];
    }
    else
    {
        _S168 = 0.5f * (g_sort_0[k_19 - u32(1)] + g_sort_0[k_19]);
    }
    return _S168;
}

@compute
@workgroup_size(256, 1, 1)
fn learned_finish(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_10 : vec3<u32>)
{
    var jc_0 : u32;
    var g_4 : u32 = group_6.x;
    var t_10 : u32 = thread_10.x;
    var j_4 : u32;
    if(g_4 >= ctl_0[u32(1)])
    {
        j_4 = u32(1);
    }
    else
    {
        j_4 = u32(0);
    }
    var _S169 : u32 = group_uniform_0(j_4, t_10);
    if(_S169 != u32(0))
    {
        return;
    }
    var _S170 : u32 = u32(8) * g_4;
    var _S171 : u32 = _S170 + u32(2);
    var _S172 : u32 = (bitcast<u32>((calls_0[_S171].z)));
    var _S173 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S171].w))), t_10);
    var nc_0 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S170 + u32(3)].w))), t_10);
    var a_11 : vec4<f32>;
    var b_9 : vec4<f32>;
    var c_6 : vec4<f32>;
    var d_4 : vec4<f32>;
    var _S174 : vec4<f32> = vec4<f32>(0.0f);
    j_4 = t_10;
    var sa_5 : vec4<f32> = _S174;
    var sb_5 : vec4<f32> = _S174;
    loop
    {
        if(j_4 < _S173)
        {
        }
        else
        {
            break;
        }
        var k_20 : u32 = _S172 + j_4;
        var p_5 : vec4<f32> = node_0(k_20, u32(0));
        var _S175 : f32 = p_5.w;
        var _S176 : vec3<f32> = vec3<f32>(_S175);
        var sa_6 : vec4<f32> = sa_5 + vec4<f32>(p_5.xyz * _S176, _S175);
        var sb_6 : vec4<f32> = sb_5 + vec4<f32>(node_0(k_20, u32(3)).xyz * _S176, node_0(k_20, u32(2)).w);
        j_4 = j_4 + u32(256);
        sa_5 = sa_6;
        sb_5 = sb_6;
    }
    reduce_store_0(t_10, sa_5, sb_5, _S174, _S174);
    reduce_read_0(&(a_11), &(b_9), &(c_6), &(d_4));
    var _S177 : vec3<f32> = vec3<f32>(a_11.w);
    var com_3 : vec3<f32> = a_11.xyz / _S177;
    var _S178 : vec3<f32> = b_9.xyz / _S177;
    var anchored_1 : bool = (b_9.w) > 0.0f;
    var vmax_0 : f32 = params_0.speed_0;
    j_4 = t_10;
    var ke_in_0 : f32 = 0.0f;
    var wg_0 : f32 = 0.0f;
    loop
    {
        if(j_4 < _S173)
        {
        }
        else
        {
            break;
        }
        var k_21 : u32 = _S172 + j_4;
        var p_6 : vec4<f32> = node_0(k_21, u32(0));
        var v_6 : vec3<f32> = node_0(k_21, u32(3)).xyz;
        var w_6 : vec3<f32> = node_0(k_21, u32(4)).xyz;
        var imp_0 : bool = (bitcast<u32>((node_0(k_21, u32(4)).w))) != u32(0);
        var id_1 : u32 = (bitcast<u32>((node_0(k_21, u32(12)).x)));
        var rim_0 : f32;
        if(imp_0)
        {
            rim_0 = imps_0[u32(3) * id_1 + u32(2)].x;
        }
        else
        {
            rim_0 = length(chunks_0[id_1].half_0.xyz);
        }
        var _S179 : f32 = max(vmax_0, max(length(v_6 - _S178), length(w_6) * rim_0));
        if(imp_0)
        {
            var vmax_1 : f32 = _S179;
            jc_0 = u32(0);
            loop
            {
                if(jc_0 < nc_0)
                {
                }
                else
                {
                    break;
                }
                var _S180 : u32 = _S172 + jc_0;
                var _S181 : f32 = max(vmax_1, dot(v_6 - node_0(_S180, u32(3)).xyz, normalize(node_0(_S180, u32(0)).xyz - p_6.xyz)));
                var _S182 : u32 = jc_0 + u32(1);
                vmax_1 = _S181;
                jc_0 = _S182;
            }
            vmax_0 = vmax_1;
        }
        else
        {
            vmax_0 = _S179;
        }
        var _S183 : f32 = p_6.w;
        var ke_in_1 : f32 = ke_in_0 + 0.5f * (_S183 * dot(v_6, v_6) + dot(w_6, (((w_6) * (world_inertia_0(node_0(k_21, u32(1)), node_0(k_21, u32(2)).xyz))))));
        var wg_1 : f32 = wg_0 + _S183 * dot(v_6 * vec3<f32>(params_0.dt_0) + params_0.gravity_0.xyz * vec3<f32>((0.5f * params_0.dt_0 * params_0.dt_0)), params_0.gravity_0.xyz);
        j_4 = j_4 + u32(256);
        ke_in_0 = ke_in_1;
        wg_0 = wg_1;
    }
    g_sort_0[t_10] = vmax_0;
    workgroupBarrier();
    jc_0 = u32(128);
    loop
    {
        if(jc_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_10 < jc_0)
        {
            g_sort_0[t_10] = max(g_sort_0[t_10], g_sort_0[t_10 + jc_0]);
        }
        workgroupBarrier();
        jc_0 = (jc_0 >> (u32(1)));
    }
    var vmax_2 : f32 = g_sort_0[i32(0)];
    workgroupBarrier();
    reduce_store_0(t_10, vec4<f32>(ke_in_0, wg_0, 0.0f, 0.0f), _S174, _S174, _S174);
    reduce_read_0(&(a_11), &(b_9), &(c_6), &(d_4));
    var ke_in_2 : f32 = a_11.x;
    var wg_2 : f32 = a_11.y;
    var _S184 : u32 = min(nc_0, u32(4096));
    j_4 = t_10;
    loop
    {
        if(j_4 < _S184)
        {
        }
        else
        {
            break;
        }
        var h_0 : vec3<f32> = chunks_0[(bitcast<u32>((node_0(_S172 + j_4, u32(12)).x)))].half_0.xyz;
        g_sort_0[j_4] = 8.0f * h_0.x * h_0.y * h_0.z;
        j_4 = j_4 + u32(256);
    }
    sort_group_0(t_10, _S184);
    var L_0 : f32 = pow(median_sorted_0(_S184), 0.3333333432674408f);
    workgroupBarrier();
    j_4 = t_10;
    loop
    {
        if(j_4 < _S184)
        {
        }
        else
        {
            break;
        }
        g_sort_0[j_4] = node_0(_S172 + j_4, u32(0)).w;
        j_4 = j_4 + u32(256);
    }
    sort_group_0(t_10, _S184);
    var mmed_0 : f32 = median_sorted_0(_S184);
    workgroupBarrier();
    j_4 = t_10;
    loop
    {
        if(j_4 < _S173)
        {
        }
        else
        {
            break;
        }
        var k_22 : u32 = _S172 + j_4;
        var p_7 : vec4<f32> = node_0(k_22, u32(0));
        var _S185 : u32 = u32(14) * k_22;
        var _S186 : vec3<f32> = p_7.xyz;
        nodes_0[_S185 + u32(13)] = vec4<f32>(_S186, 0.0f);
        nodes_0[_S185] = vec4<f32>(_S186 - com_3, p_7.w);
        j_4 = j_4 + u32(256);
    }
    if(t_10 == u32(0))
    {
        var _S187 : f32 = max(ke_in_2 + wg_2, 0.5f * mmed_0 * vmax_2 * vmax_2);
        var flags_1 : u32;
        if(anchored_1)
        {
            flags_1 = u32(4);
        }
        else
        {
            flags_1 = u32(0);
        }
        var _S188 : bool;
        if(!anchored_1)
        {
            _S188 = (params_0.gravity_0.w) == 0.0f;
        }
        else
        {
            _S188 = false;
        }
        if(_S188)
        {
            flags_1 = (flags_1 | (u32(1)));
        }
        var c2_1 : vec4<f32> = calls_0[_S171];
        calls_0[_S170] = vec4<f32>(L_0, mmed_0, vmax_2, 0.0f);
        calls_0[_S170 + u32(1)] = vec4<f32>(0.0f, 0.0f, 0.0f, params_0.dt_0);
        calls_0[_S171] = vec4<f32>(_S187, (bitcast<f32>((flags_1))), c2_1.z, c2_1.w);
        var _S189 : u32 = _S170 + u32(5);
        calls_0[_S189] = vec4<f32>(com_3, calls_0[_S189].w);
    }
    return;
}

fn inverse_rotate_0( q_12 : Quat_0,  v_7 : vec3<f32>) -> vec3<f32>
{
    var c_7 : Quat_0;
    c_7.w_0 = q_12.w_0;
    c_7.x_0 = - q_12.x_0;
    c_7.y_0 = - q_12.y_0;
    c_7.z_0 = - q_12.z_0;
    return rotate_0(c_7, v_7);
}

@compute
@workgroup_size(256, 1, 1)
fn learned_stub(@builtin(workgroup_id) group_7 : vec3<u32>, @builtin(local_invocation_id) thread_11 : vec3<u32>)
{
    var plug_0 : bool;
    var g_5 : u32 = group_7.x;
    var t_11 : u32 = thread_11.x;
    var pa_0 : bool;
    if(g_5 >= ctl_0[u32(1)])
    {
        pa_0 = true;
    }
    else
    {
        pa_0 = (params_0.stub_0) == u32(0);
    }
    var j_5 : u32;
    if(pa_0)
    {
        j_5 = u32(1);
    }
    else
    {
        j_5 = u32(0);
    }
    var _S190 : u32 = group_uniform_0(j_5, t_11);
    if(_S190 != u32(0))
    {
        return;
    }
    var _S191 : u32 = u32(8) * g_5;
    var _S192 : u32 = _S191 + u32(2);
    var first_0 : u32 = (bitcast<u32>((calls_0[_S192].z)));
    var n_4 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S192].w))), t_11);
    var nc_1 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S191 + u32(3)].w))), t_11);
    var _S193 : u32 = _S191 + u32(4);
    var _S194 : u32 = (bitcast<u32>((calls_0[_S193].x)));
    var _S195 : u32 = (bitcast<u32>((calls_0[_S193].y)));
    var vref_0 : f32 = calls_0[_S191].z;
    j_5 = t_11;
    loop
    {
        if(j_5 < n_4)
        {
        }
        else
        {
            break;
        }
        var _S196 : u32 = u32(14) * (first_0 + j_5);
        var _S197 : vec4<f32> = vec4<f32>(0.0f);
        nodes_0[_S196 + u32(5)] = _S197;
        nodes_0[_S196 + u32(6)] = _S197;
        nodes_0[_S196 + u32(7)] = _S197;
        j_5 = j_5 + u32(256);
    }
    j_5 = t_11;
    loop
    {
        if(j_5 < _S195)
        {
        }
        else
        {
            break;
        }
        var _S198 : u32 = u32(2) * (_S194 + j_5);
        bonds_0[_S198] = vec4<f32>(bonds_0[_S198].x, -30.0f, 0.0f, (bitcast<f32>((u32(1)))));
        j_5 = j_5 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    if((params_0.stub_0) == u32(3))
    {
        j_5 = u32(1);
    }
    else
    {
        j_5 = u32(0);
    }
    var _S199 : u32 = group_uniform_0(j_5, t_11);
    if(_S199 != u32(0))
    {
        var _S200 : f32 = float_bits_0(u32(2143289344));
        j_5 = t_11;
        loop
        {
            if(j_5 < n_4)
            {
            }
            else
            {
                break;
            }
            nodes_0[u32(14) * (first_0 + j_5) + u32(5)] = vec4<f32>(_S200);
            j_5 = j_5 + u32(256);
        }
        return;
    }
    if((params_0.stub_0) == u32(2))
    {
        j_5 = u32(1);
    }
    else
    {
        j_5 = u32(0);
    }
    var _S201 : u32 = group_uniform_0(j_5, t_11);
    if(_S201 == u32(0))
    {
        return;
    }
    var ki_0 : u32 = first_0 + nc_1;
    if(n_4 > nc_1)
    {
        j_5 = u32(1);
    }
    else
    {
        j_5 = u32(0);
    }
    var _S202 : u32 = group_uniform_0(j_5, t_11);
    if(_S202 == u32(0))
    {
        return;
    }
    var _S203 : vec3<f32> = node_0(ki_0, u32(0)).xyz;
    var vi_0 : vec3<f32> = node_0(ki_0, u32(3)).xyz;
    var mi_0 : f32 = node_0(ki_0, u32(0)).w;
    var _S204 : f32 = imps_0[u32(3) * (bitcast<u32>((node_0(ki_0, u32(12)).x)))].z;
    var _S205 : vec3<f32> = normalize(vi_0 - calls_0[_S191 + u32(5)].xyz * vec3<f32>(0.0f));
    var inf_1 : f32 = float_bits_0(u32(2139095040));
    var best_0 : f32 = inf_1;
    j_5 = t_11;
    loop
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        var k_23 : u32 = first_0 + j_5;
        var r_10 : vec3<f32> = node_0(k_23, u32(0)).xyz - _S203;
        var along_0 : f32 = dot(r_10, _S205);
        if((length(r_10 - _S205 * vec3<f32>(along_0))) <= (_S204 + chunks_0[(bitcast<u32>((node_0(k_23, u32(12)).x)))].half_0.w))
        {
            pa_0 = along_0 > 0.0f;
        }
        else
        {
            pa_0 = false;
        }
        if(pa_0)
        {
            best_0 = min(best_0, along_0);
        }
        j_5 = j_5 + u32(256);
    }
    g_sort_0[t_11] = best_0;
    workgroupBarrier();
    var s_7 : u32 = u32(128);
    loop
    {
        if(s_7 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_11 < s_7)
        {
            g_sort_0[t_11] = min(g_sort_0[t_11], g_sort_0[t_11 + s_7]);
        }
        workgroupBarrier();
        s_7 = (s_7 >> (u32(1)));
    }
    var _S206 : f32 = g_sort_0[i32(0)];
    workgroupBarrier();
    var _S207 : vec4<f32> = vec4<f32>(0.0f);
    j_5 = t_11;
    var sa_7 : vec4<f32> = _S207;
    loop
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        var k_24 : u32 = first_0 + j_5;
        var r_11 : vec3<f32> = node_0(k_24, u32(0)).xyz - _S203;
        var along_1 : f32 = dot(r_11, _S205);
        var h_1 : f32 = chunks_0[(bitcast<u32>((node_0(k_24, u32(12)).x)))].half_0.w;
        if(_S206 < inf_1)
        {
            pa_0 = along_1 >= (_S206 - h_1);
        }
        else
        {
            pa_0 = false;
        }
        if(pa_0)
        {
            plug_0 = (length(r_11 - _S205 * vec3<f32>(along_1))) <= _S204;
        }
        else
        {
            plug_0 = false;
        }
        var plug_1 : bool;
        if(plug_0)
        {
            plug_1 = (node_0(k_24, u32(2)).w) == 0.0f;
        }
        else
        {
            plug_1 = false;
        }
        nodes_0[u32(14) * k_24 + u32(12)][i32(1)] = node_0(k_24, u32(12)).y;
        if(plug_1)
        {
            sa_7 = sa_7 + vec4<f32>(node_0(k_24, u32(3)).xyz * vec3<f32>(node_0(k_24, u32(0)).w), node_0(k_24, u32(0)).w);
        }
        g_scan_0[t_11] = u32(0);
        j_5 = j_5 + u32(256);
    }
    reduce_store_0(t_11, sa_7, _S207, _S207, _S207);
    var a_12 : vec4<f32>;
    var b_10 : vec4<f32>;
    var c_8 : vec4<f32>;
    var d_5 : vec4<f32>;
    reduce_read_0(&(a_12), &(b_10), &(c_8), &(d_5));
    var mp_0 : f32 = a_12.w;
    if(mp_0 > 0.0f)
    {
        j_5 = u32(1);
    }
    else
    {
        j_5 = u32(0);
    }
    var _S208 : u32 = group_uniform_0(j_5, t_11);
    if(_S208 == u32(0))
    {
        return;
    }
    var vc_1 : vec3<f32> = (a_12.xyz + vec3<f32>(mi_0) * vi_0) / vec3<f32>((mp_0 + mi_0));
    j_5 = t_11;
    loop
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        var k_25 : u32 = first_0 + j_5;
        var r_12 : vec3<f32> = node_0(k_25, u32(0)).xyz - _S203;
        var along_2 : f32 = dot(r_12, _S205);
        if(along_2 >= (_S206 - chunks_0[(bitcast<u32>((node_0(k_25, u32(12)).x)))].half_0.w))
        {
            pa_0 = (length(r_12 - _S205 * vec3<f32>(along_2))) <= _S204;
        }
        else
        {
            pa_0 = false;
        }
        if(pa_0)
        {
            plug_0 = (node_0(k_25, u32(2)).w) == 0.0f;
        }
        else
        {
            plug_0 = false;
        }
        if(plug_0)
        {
            var o_1 : vec3<f32> = inverse_rotate_0(quat_of_0(node_0(k_25, u32(1))), vc_1 - node_0(k_25, u32(3)).xyz) / vec3<f32>(vref_0);
            var _S209 : u32 = u32(14) * k_25;
            nodes_0[_S209 + u32(5)] = vec4<f32>(o_1, 0.0f);
            nodes_0[_S209 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.5f * o_1.x, 0.5f * o_1.y);
            nodes_0[_S209 + u32(7)] = vec4<f32>(0.5f * o_1.z, 0.0f, 0.0f, 0.0f);
            nodes_0[_S209 + u32(4)] = vec4<f32>(node_0(k_25, u32(4)).xyz, (bitcast<f32>((u32(2)))));
        }
        j_5 = j_5 + u32(256);
    }
    if(t_11 == u32(0))
    {
        var o_2 : vec3<f32> = inverse_rotate_0(quat_of_0(node_0(ki_0, u32(1))), vc_1 - vi_0) / vec3<f32>(vref_0);
        var _S210 : u32 = u32(14) * ki_0;
        nodes_0[_S210 + u32(5)] = vec4<f32>(o_2, 0.0f);
        nodes_0[_S210 + u32(6)] = vec4<f32>(0.0f, 0.0f, 0.5f * o_2.x, 0.5f * o_2.y);
        nodes_0[_S210 + u32(7)] = vec4<f32>(0.5f * o_2.z, 0.0f, 0.0f, 0.0f);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    j_5 = t_11;
    loop
    {
        if(j_5 < _S195)
        {
        }
        else
        {
            break;
        }
        var _S211 : u32 = u32(2) * (_S194 + j_5);
        var na_0 : u32 = chunk_node_0[bond_static_0[(bitcast<u32>((bonds_0[_S211 + u32(1)].z)))].law_0.ids_0.y];
        var nb2_0 : u32 = chunk_node_0[bond_static_0[(bitcast<u32>((bonds_0[_S211 + u32(1)].z)))].law_0.ids_0.z];
        if(chunk_node_0[bond_static_0[(bitcast<u32>((bonds_0[_S211 + u32(1)].z)))].law_0.ids_0.y] != u32(0))
        {
            pa_0 = (((bitcast<u32>((nodes_0[u32(14) * (na_0 - u32(1)) + u32(4)].w))) & (u32(2)))) != u32(0);
        }
        else
        {
            pa_0 = false;
        }
        if(nb2_0 != u32(0))
        {
            plug_0 = (((bitcast<u32>((nodes_0[u32(14) * (nb2_0 - u32(1)) + u32(4)].w))) & (u32(2)))) != u32(0);
        }
        else
        {
            plug_0 = false;
        }
        if(pa_0 != plug_0)
        {
            bonds_0[_S211] = vec4<f32>(bonds_0[_S211].x, 1.0f, 0.0f, (bitcast<f32>((u32(1)))));
        }
        j_5 = j_5 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    j_5 = t_11;
    loop
    {
        if(j_5 < nc_1)
        {
        }
        else
        {
            break;
        }
        var k_26 : u32 = first_0 + j_5;
        nodes_0[u32(14) * k_26 + u32(4)] = vec4<f32>(node_0(k_26, u32(4)).xyz, (bitcast<f32>((u32(0)))));
        j_5 = j_5 + u32(256);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn learned_gate(@builtin(workgroup_id) group_8 : vec3<u32>, @builtin(local_invocation_id) thread_12 : vec3<u32>)
{
    var _S212 : bool;
    var g_6 : u32 = group_8.x;
    var t_12 : u32 = thread_12.x;
    if(g_6 >= ctl_0[u32(1)])
    {
        return;
    }
    var _S213 : u32 = u32(8) * g_6;
    var _S214 : u32 = _S213 + u32(4);
    var c4_1 : vec4<f32> = calls_0[_S214];
    var _S215 : f32 = calls_0[_S214].z;
    var dec_1 : u32 = (bitcast<u32>((_S215)));
    var fb_0 : u32;
    if(((dec_1 & (u32(4)))) != u32(0))
    {
        fb_0 = u32(1);
    }
    else
    {
        fb_0 = u32(0);
    }
    if(((dec_1 & (u32(2)))) != u32(0))
    {
        fb_0 = (fb_0 | (u32(2)));
    }
    var _S216 : u32 = _S213 + u32(6);
    var e0_0 : u32 = (bitcast<u32>((calls_0[_S216].x)));
    var _S217 : u32 = (bitcast<u32>((calls_0[_S216].y)));
    var mask_6 : u32 = (bitcast<u32>((calls_0[_S213 + u32(5)].w)));
    if(t_12 == u32(0))
    {
        calls_0[_S214] = vec4<f32>(c4_1.x, c4_1.y, _S215, (bitcast<f32>((fb_0))));
    }
    var j_6 : u32;
    if(fb_0 != u32(0))
    {
        var _S218 : u32 = (bitcast<u32>((calls_0[_S213 + u32(2)].z)));
        var _S219 : u32 = (bitcast<u32>((calls_0[_S213 + u32(3)].w)));
        j_6 = t_12;
        loop
        {
            if(j_6 < _S219)
            {
            }
            else
            {
                break;
            }
            chunk_node_0[(bitcast<u32>((nodes_0[u32(14) * (_S218 + j_6) + u32(12)].x)))] = u32(0);
            j_6 = j_6 + u32(64);
        }
        return;
    }
    j_6 = e0_0 + t_12;
    loop
    {
        if(j_6 < (e0_0 + _S217))
        {
        }
        else
        {
            break;
        }
        var _S220 : u32 = u32(2) * j_6;
        var c1_1 : vec4<u32> = cisl_0[_S220 + u32(1)];
        status_0[u32(16) + cisl_0[_S220].x] = (status_0[u32(16) + cisl_0[_S220].x] | (u32(4096)));
        var body_2 : u32 = c1_1.x;
        if(body_2 == u32(4294967295))
        {
            _S212 = true;
        }
        else
        {
            _S212 = (c1_1.w) != u32(0);
        }
        if(_S212)
        {
            frozen_0[u32(4) * j_6] = vec4<f32>((bitcast<f32>((u32(4294967295)))), 0.0f, 0.0f, 0.0f);
            j_6 = j_6 + u32(64);
            continue;
        }
        var at_1 : u32 = u32(7) * body_2;
        var _S221 : u32 = u32(4) * j_6;
        frozen_0[_S221] = vec4<f32>((bitcast<f32>((body_2))), bodies_0[at_1].w, 0.0f, 0.0f);
        var _S222 : u32 = at_1 + u32(2);
        frozen_0[_S221 + u32(1)] = bodies_0[_S222];
        var _S223 : u32 = at_1 + u32(3);
        frozen_0[_S221 + u32(2)] = bodies_0[_S223];
        var _S224 : u32 = at_1 + u32(4);
        frozen_0[_S221 + u32(3)] = bodies_0[_S224];
        bodies_0[at_1] = vec4<f32>(bodies_0[at_1].xyz, 0.0f);
        bodies_0[_S222] = vec4<f32>(0.0f, 0.0f, 0.0f, bodies_0[_S222].w);
        bodies_0[_S223] = vec4<f32>(0.0f, 0.0f, 0.0f, bodies_0[_S223].w);
        bodies_0[_S224] = vec4<f32>(0.0f, 0.0f, 0.0f, bodies_0[_S224].w);
        j_6 = j_6 + u32(64);
    }
    if(t_12 < u32(32))
    {
        _S212 = ((mask_6 & (((u32(1) << (t_12)))))) != u32(0);
    }
    else
    {
        _S212 = false;
    }
    if(_S212)
    {
        var body_3 : u32 = (bitcast<u32>((imps_0[u32(3) * t_12].x)));
        var at_2 : u32 = u32(7) * body_3;
        var _S225 : u32 = u32(4) * (params_0.max_cisl_0 + t_12);
        frozen_0[_S225] = vec4<f32>((bitcast<f32>((body_3))), bodies_0[at_2].w, 0.0f, 0.0f);
        var _S226 : u32 = at_2 + u32(2);
        frozen_0[_S225 + u32(1)] = bodies_0[_S226];
        var _S227 : u32 = at_2 + u32(3);
        frozen_0[_S225 + u32(2)] = bodies_0[_S227];
        var _S228 : u32 = at_2 + u32(4);
        frozen_0[_S225 + u32(3)] = bodies_0[_S228];
        bodies_0[at_2] = vec4<f32>(bodies_0[at_2].xyz, 0.0f);
        bodies_0[_S226] = vec4<f32>(0.0f, 0.0f, 0.0f, bodies_0[_S226].w);
        bodies_0[_S227] = vec4<f32>(0.0f, 0.0f, 0.0f, bodies_0[_S227].w);
        bodies_0[_S228] = vec4<f32>(0.0f, 0.0f, 0.0f, bodies_0[_S228].w);
    }
    return;
}

var<workgroup> g_stale_0 : u32;

fn island_current_0( e_6 : u32) -> bool
{
    var _S229 : u32;
    var c0_3 : vec4<u32> = cisl_0[u32(2) * e_6];
    var _S230 : u32 = cisl_0[u32(2) * e_6].y;
    var island_0 : u32 = u32(4294967295);
    var j_7 : u32 = u32(0);
    loop
    {
        var _S231 : u32 = c0_3.z;
        _S229 = _S231;
        if(j_7 < _S231)
        {
        }
        else
        {
            break;
        }
        var w_7 : vec4<f32> = nodes_0[u32(14) * (_S230 + j_7) + u32(12)];
        var k_27 : u32 = (bitcast<u32>((nodes_0[u32(14) * (_S230 + j_7) + u32(12)].x)));
        if(k_27 >= (params_0.chunk_count_0))
        {
            return false;
        }
        var info_2 : vec4<u32> = chunks_0[k_27].info_1;
        var _S232 : bool;
        if((chunks_0[k_27].info_1.z) != (bitcast<u32>((w_7.z))))
        {
            _S232 = true;
        }
        else
        {
            _S232 = (info_2.w) != (bitcast<u32>((w_7.w)));
        }
        if(_S232)
        {
            return false;
        }
        if(j_7 == u32(0))
        {
            island_0 = info_2.y;
        }
        else
        {
            if((info_2.y) != island_0)
            {
                return false;
            }
        }
        j_7 = j_7 + u32(1);
    }
    if(_S229 == u32(0))
    {
        return false;
    }
    return (islands_0[island_0].range_0.y - islands_0[island_0].range_0.x) == _S229;
}

@compute
@workgroup_size(256, 1, 1)
fn learned_check(@builtin(workgroup_id) group_9 : vec3<u32>, @builtin(local_invocation_id) thread_13 : vec3<u32>)
{
    var g_7 : u32 = group_9.x;
    var t_13 : u32 = thread_13.x;
    var _S233 : bool;
    if(ctl_0[u32(0)] == u32(0))
    {
        _S233 = true;
    }
    else
    {
        _S233 = g_7 >= ctl_0[u32(1)];
    }
    var e_7 : u32;
    if(_S233)
    {
        e_7 = u32(1);
    }
    else
    {
        e_7 = u32(0);
    }
    var _S234 : u32 = group_uniform_0(e_7, t_13);
    if(_S234 != u32(0))
    {
        return;
    }
    var _S235 : u32 = u32(8) * g_7;
    var _S236 : u32 = _S235 + u32(4);
    var c4_2 : vec4<f32> = calls_0[_S236];
    var _S237 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S236].w))), t_13);
    if(_S237 != u32(0))
    {
        return;
    }
    var _S238 : u32 = _S235 + u32(6);
    var e0_1 : u32 = (bitcast<u32>((calls_0[_S238].x)));
    var _S239 : u32 = (bitcast<u32>((calls_0[_S238].y)));
    var _S240 : bool = t_13 == u32(0);
    if(_S240)
    {
        g_stale_0 = u32(0);
    }
    workgroupBarrier();
    e_7 = e0_1 + t_13;
    loop
    {
        if(e_7 < (e0_1 + _S239))
        {
        }
        else
        {
            break;
        }
        if(!island_current_0(e_7))
        {
            g_stale_0 = u32(1);
        }
        e_7 = e_7 + u32(256);
    }
    workgroupBarrier();
    var _S241 : u32 = group_uniform_0(g_stale_0, t_13);
    if(_S241 == u32(0))
    {
        return;
    }
    if(_S240)
    {
        calls_0[_S236] = vec4<f32>(c4_2.x, c4_2.y, c4_2.z, (bitcast<f32>((u32(8)))));
        var _S242 : f32 = (bitcast<f32>((u32(0))));
        calls_0[_S235 + u32(7)] = vec4<f32>((bitcast<f32>((u32(1)))), _S242, _S242, 0.0f);
    }
    var _S243 : u32 = (bitcast<u32>((calls_0[_S235 + u32(2)].z)));
    var _S244 : u32 = (bitcast<u32>((calls_0[_S235 + u32(3)].w)));
    var j_8 : u32 = t_13;
    loop
    {
        if(j_8 < _S244)
        {
        }
        else
        {
            break;
        }
        var k_28 : u32 = (bitcast<u32>((nodes_0[u32(14) * (_S243 + j_8) + u32(12)].x)));
        if(k_28 < (params_0.chunk_count_0))
        {
            chunk_node_0[k_28] = u32(0);
        }
        j_8 = j_8 + u32(256);
    }
    return;
}

var<workgroup> g_broke_0 : u32;

struct BondDyn_0
{
     js_0 : JointState_0,
     force_lin_0 : vec4<f32>,
     force_ang_0 : vec4<f32>,
     sums_0 : vec4<f32>,
     comps_0 : vec4<f32>,
     events_0 : vec4<u32>,
};

@compute
@workgroup_size(256, 1, 1)
fn learned_apply_bonds(@builtin(workgroup_id) group_10 : vec3<u32>, @builtin(local_invocation_id) thread_14 : vec3<u32>)
{
    var count_0 : u32;
    var g_8 : u32 = group_10.x;
    var t_14 : u32 = thread_14.x;
    var _S245 : bool;
    if(ctl_0[u32(0)] == u32(0))
    {
        _S245 = true;
    }
    else
    {
        _S245 = g_8 >= ctl_0[u32(1)];
    }
    var j_9 : u32;
    if(_S245)
    {
        j_9 = u32(1);
    }
    else
    {
        j_9 = u32(0);
    }
    var _S246 : u32 = group_uniform_0(j_9, t_14);
    if(_S246 != u32(0))
    {
        return;
    }
    var _S247 : u32 = u32(8) * g_8;
    var c4_3 : vec4<f32> = calls_0[_S247 + u32(4)];
    var _S248 : u32 = group_uniform_0((bitcast<u32>((calls_0[_S247 + u32(4)].w))), t_14);
    if(_S248 != u32(0))
    {
        return;
    }
    var _S249 : bool = t_14 == u32(0);
    if(_S249)
    {
        g_broke_0 = u32(0);
    }
    workgroupBarrier();
    var _S250 : u32 = (bitcast<u32>((c4_3.x)));
    var _S251 : u32 = (bitcast<u32>((c4_3.y)));
    j_9 = t_14;
    var count_1 : u32 = u32(0);
    loop
    {
        if(j_9 < _S251)
        {
        }
        else
        {
            break;
        }
        var b1_1 : vec4<f32> = bonds_0[u32(2) * (_S250 + j_9) + u32(1)];
        var bi_2 : u32 = (bitcast<u32>((bonds_0[u32(2) * (_S250 + j_9) + u32(1)].z)));
        var _S252 : JointState_0 = JointState_0( bond_dyn_0[bi_2].js_0.damage_0, bond_dyn_0[bi_2].js_0.crush_0, bond_dyn_0[bi_2].js_0.kappa_0, bond_dyn_0[bi_2].js_0.kappa_c_0, bond_dyn_0[bi_2].js_0.ductility_0, bond_dyn_0[bi_2].js_0.ductility_c_0, bond_dyn_0[bi_2].js_0.life_0, bond_dyn_0[bi_2].js_0.plastic_x_0, bond_dyn_0[bi_2].js_0.plastic_y_0, bond_dyn_0[bi_2].js_0.plastic_t_0, bond_dyn_0[bi_2].js_0.rebar_plastic_0, bond_dyn_0[bi_2].js_0.rebar_slip0_0, bond_dyn_0[bi_2].js_0.rebar_slip1_0, bond_dyn_0[bi_2].js_0.rebar_work_0, bond_dyn_0[bi_2].js_0.rebar_broken_0, bond_dyn_0[bi_2].js_0.strain_rate_0, bond_dyn_0[bi_2].js_0.governing_stress_0, bond_dyn_0[bi_2].js_0.dissipated_0, bond_dyn_0[bi_2].js_0.utilization_0, bond_dyn_0[bi_2].js_0.mode_0 );
        var dyn_0 : BondDyn_0;
        dyn_0.js_0 = _S252;
        dyn_0.force_lin_0 = bond_dyn_0[bi_2].force_lin_0;
        dyn_0.force_ang_0 = bond_dyn_0[bi_2].force_ang_0;
        dyn_0.sums_0 = bond_dyn_0[bi_2].sums_0;
        dyn_0.comps_0 = bond_dyn_0[bi_2].comps_0;
        dyn_0.events_0 = bond_dyn_0[bi_2].events_0;
        if((bitcast<u32>((bonds_0[u32(2) * (_S250 + j_9) + u32(1)].y))) != u32(0))
        {
            if(connected_1(dyn_0.js_0, (bond_static_0[bi_2].law_0.stiff1_0.w) != 0.0f))
            {
                dyn_0.js_0.damage_0 = 1.0f;
                dyn_0.js_0.rebar_broken_0 = 1.0f;
                dyn_0.events_0[i32(2)] = max(dyn_0.events_0.z, u32(1));
                bond_dyn_0[bi_2].js_0.damage_0 = dyn_0.js_0.damage_0;
                bond_dyn_0[bi_2].js_0.crush_0 = dyn_0.js_0.crush_0;
                bond_dyn_0[bi_2].js_0.kappa_0 = dyn_0.js_0.kappa_0;
                bond_dyn_0[bi_2].js_0.kappa_c_0 = dyn_0.js_0.kappa_c_0;
                bond_dyn_0[bi_2].js_0.ductility_0 = dyn_0.js_0.ductility_0;
                bond_dyn_0[bi_2].js_0.ductility_c_0 = dyn_0.js_0.ductility_c_0;
                bond_dyn_0[bi_2].js_0.life_0 = dyn_0.js_0.life_0;
                bond_dyn_0[bi_2].js_0.plastic_x_0 = dyn_0.js_0.plastic_x_0;
                bond_dyn_0[bi_2].js_0.plastic_y_0 = dyn_0.js_0.plastic_y_0;
                bond_dyn_0[bi_2].js_0.plastic_t_0 = dyn_0.js_0.plastic_t_0;
                bond_dyn_0[bi_2].js_0.rebar_plastic_0 = dyn_0.js_0.rebar_plastic_0;
                bond_dyn_0[bi_2].js_0.rebar_slip0_0 = dyn_0.js_0.rebar_slip0_0;
                bond_dyn_0[bi_2].js_0.rebar_slip1_0 = dyn_0.js_0.rebar_slip1_0;
                bond_dyn_0[bi_2].js_0.rebar_work_0 = dyn_0.js_0.rebar_work_0;
                bond_dyn_0[bi_2].js_0.rebar_broken_0 = dyn_0.js_0.rebar_broken_0;
                bond_dyn_0[bi_2].js_0.strain_rate_0 = dyn_0.js_0.strain_rate_0;
                bond_dyn_0[bi_2].js_0.governing_stress_0 = dyn_0.js_0.governing_stress_0;
                bond_dyn_0[bi_2].js_0.dissipated_0 = dyn_0.js_0.dissipated_0;
                bond_dyn_0[bi_2].js_0.utilization_0 = dyn_0.js_0.utilization_0;
                bond_dyn_0[bi_2].js_0.mode_0 = dyn_0.js_0.mode_0;
                bond_dyn_0[bi_2].force_lin_0 = dyn_0.force_lin_0;
                bond_dyn_0[bi_2].force_ang_0 = dyn_0.force_ang_0;
                bond_dyn_0[bi_2].sums_0 = dyn_0.sums_0;
                bond_dyn_0[bi_2].comps_0 = dyn_0.comps_0;
                bond_dyn_0[bi_2].events_0 = dyn_0.events_0;
                var _S253 : u32 = count_1 + u32(1);
                status_0[u32(16) + chunks_0[bond_static_0[bi_2].law_0.ids_0.y].info_1.y] = (status_0[u32(16) + chunks_0[bond_static_0[bi_2].law_0.ids_0.y].info_1.y] | (u32(8)));
                g_broke_0 = u32(1);
                count_0 = _S253;
            }
            else
            {
                count_0 = count_1;
            }
            count_1 = count_0;
        }
        else
        {
            dyn_0.js_0.damage_0 = max(dyn_0.js_0.damage_0, min(b1_1.x, 0.99999898672103882f));
            bond_dyn_0[bi_2].js_0.damage_0 = dyn_0.js_0.damage_0;
            bond_dyn_0[bi_2].js_0.crush_0 = dyn_0.js_0.crush_0;
            bond_dyn_0[bi_2].js_0.kappa_0 = dyn_0.js_0.kappa_0;
            bond_dyn_0[bi_2].js_0.kappa_c_0 = dyn_0.js_0.kappa_c_0;
            bond_dyn_0[bi_2].js_0.ductility_0 = dyn_0.js_0.ductility_0;
            bond_dyn_0[bi_2].js_0.ductility_c_0 = dyn_0.js_0.ductility_c_0;
            bond_dyn_0[bi_2].js_0.life_0 = dyn_0.js_0.life_0;
            bond_dyn_0[bi_2].js_0.plastic_x_0 = dyn_0.js_0.plastic_x_0;
            bond_dyn_0[bi_2].js_0.plastic_y_0 = dyn_0.js_0.plastic_y_0;
            bond_dyn_0[bi_2].js_0.plastic_t_0 = dyn_0.js_0.plastic_t_0;
            bond_dyn_0[bi_2].js_0.rebar_plastic_0 = dyn_0.js_0.rebar_plastic_0;
            bond_dyn_0[bi_2].js_0.rebar_slip0_0 = dyn_0.js_0.rebar_slip0_0;
            bond_dyn_0[bi_2].js_0.rebar_slip1_0 = dyn_0.js_0.rebar_slip1_0;
            bond_dyn_0[bi_2].js_0.rebar_work_0 = dyn_0.js_0.rebar_work_0;
            bond_dyn_0[bi_2].js_0.rebar_broken_0 = dyn_0.js_0.rebar_broken_0;
            bond_dyn_0[bi_2].js_0.strain_rate_0 = dyn_0.js_0.strain_rate_0;
            bond_dyn_0[bi_2].js_0.governing_stress_0 = dyn_0.js_0.governing_stress_0;
            bond_dyn_0[bi_2].js_0.dissipated_0 = dyn_0.js_0.dissipated_0;
            bond_dyn_0[bi_2].js_0.utilization_0 = dyn_0.js_0.utilization_0;
            bond_dyn_0[bi_2].js_0.mode_0 = dyn_0.js_0.mode_0;
            bond_dyn_0[bi_2].force_lin_0 = dyn_0.force_lin_0;
            bond_dyn_0[bi_2].force_ang_0 = dyn_0.force_ang_0;
            bond_dyn_0[bi_2].sums_0 = dyn_0.sums_0;
            bond_dyn_0[bi_2].comps_0 = dyn_0.comps_0;
            bond_dyn_0[bi_2].events_0 = dyn_0.events_0;
        }
        j_9 = j_9 + u32(256);
    }
    g_scan_0[t_14] = count_1;
    workgroupBarrier();
    count_0 = u32(128);
    loop
    {
        if(count_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_14 < count_0)
        {
            g_scan_0[t_14] = g_scan_0[t_14] + g_scan_0[t_14 + count_0];
        }
        workgroupBarrier();
        count_0 = (count_0 >> (u32(1)));
    }
    if(_S249)
    {
        calls_0[_S247 + u32(7)] = vec4<f32>((bitcast<f32>((u32(0)))), (bitcast<f32>((g_scan_0[i32(0)]))), (bitcast<f32>((u32(1)))), 0.0f);
        if(g_broke_0 != u32(0))
        {
            status_0[u32(1)] = u32(1);
        }
    }
    return;
}

fn rotvec_quat_0( r_13 : vec3<f32>) -> vec4<f32>
{
    var th_1 : f32 = length(r_13);
    if(th_1 < 9.999999960041972e-13f)
    {
        return normalize(vec4<f32>(vec3<f32>(0.5f) * r_13, 1.0f));
    }
    var _S254 : f32 = 0.5f * th_1;
    return vec4<f32>(r_13 / vec3<f32>(th_1) * vec3<f32>(sin(_S254)), cos(_S254));
}

struct Fit_0
{
     com0_0 : vec3<f32>,
     com1_0 : vec3<f32>,
     turn_0 : vec4<f32>,
     v_8 : vec3<f32>,
     w_8 : vec3<f32>,
};

fn fit_nodes_0( first_1 : u32,  n_5 : u32,  listed_0 : bool,  list_base_0 : u32,  list_first_0 : u32) -> Fit_0
{
    var k_29 : u32;
    var _S255 : vec3<f32> = vec3<f32>(0.0f);
    var j_10 : u32 = u32(0);
    var m_9 : f32 = 0.0f;
    var x0_0 : vec3<f32> = _S255;
    var x1_0 : vec3<f32> = _S255;
    var rot_0 : vec3<f32> = _S255;
    var p_8 : vec3<f32> = _S255;
    loop
    {
        if(j_10 < n_5)
        {
        }
        else
        {
            break;
        }
        if(listed_0)
        {
            k_29 = chunk_node_0[list_first_0 + split_work_0[list_base_0 + j_10]] - u32(1);
        }
        else
        {
            k_29 = first_1 + j_10;
        }
        var mc_0 : f32 = node_0(k_29, u32(0)).w;
        var xs_0 : vec3<f32> = node_0(k_29, u32(13)).xyz;
        var m_10 : f32 = m_9 + mc_0;
        var _S256 : vec3<f32> = vec3<f32>(mc_0);
        var x0_1 : vec3<f32> = x0_0 + xs_0 * _S256;
        var x1_1 : vec3<f32> = x1_0 + (xs_0 + node_0(k_29, u32(10)).xyz) * _S256;
        var rot_1 : vec3<f32> = rot_0 + node_0(k_29, u32(11)).xyz * _S256;
        var p_9 : vec3<f32> = p_8 + node_0(k_29, u32(8)).xyz * _S256;
        j_10 = j_10 + u32(1);
        m_9 = m_10;
        x0_0 = x0_1;
        x1_0 = x1_1;
        rot_0 = rot_1;
        p_8 = p_9;
    }
    var f_2 : Fit_0;
    f_2.com0_0 = x0_0 / vec3<f32>(m_9);
    f_2.com1_0 = x1_0 / vec3<f32>(m_9);
    var _S257 : vec4<f32> = rotvec_quat_0(rot_0 / vec3<f32>(m_9));
    f_2.turn_0 = _S257;
    f_2.v_8 = p_8 / vec3<f32>(m_9);
    var _S258 : Quat_0 = quat_of_0(_S257);
    var _S259 : mat3x3<f32> = quat_matrix_0(_S257);
    var _S260 : mat3x3<f32> = mat3x3<f32>(_S255, _S255, _S255);
    j_10 = u32(0);
    var l_1 : vec3<f32> = _S255;
    var itot_0 : mat3x3<f32> = _S260;
    loop
    {
        if(j_10 < n_5)
        {
        }
        else
        {
            break;
        }
        if(listed_0)
        {
            k_29 = chunk_node_0[list_first_0 + split_work_0[list_base_0 + j_10]] - u32(1);
        }
        else
        {
            k_29 = first_1 + j_10;
        }
        var mc_1 : f32 = node_0(k_29, u32(0)).w;
        var r_14 : vec3<f32> = rotate_0(_S258, node_0(k_29, u32(13)).xyz - f_2.com0_0);
        var ic_1 : mat3x3<f32> = (((transpose(_S259)) * ((((world_inertia_0(node_0(k_29, u32(1)), node_0(k_29, u32(2)).xyz)) * (_S259))))));
        var l_2 : vec3<f32> = l_1 + (cross(r_14, vec3<f32>(mc_1) * node_0(k_29, u32(8)).xyz) + (((node_0(k_29, u32(9)).xyz) * (ic_1))));
        var _S261 : mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f));
        var _S262 : mat3x3<f32> = mat3x3<f32>(dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14), dot(r_14, r_14));
        var _S263 : mat3x3<f32> = mat3x3<f32>(_S262[0] * _S261[0], _S262[1] * _S261[1], _S262[2] * _S261[2]) - outer3_0(r_14, r_14);
        var _S264 : mat3x3<f32> = mat3x3<f32>(mc_1, mc_1, mc_1, mc_1, mc_1, mc_1, mc_1, mc_1, mc_1);
        var _S265 : mat3x3<f32> = itot_0 + (ic_1 + mat3x3<f32>(_S264[0] * _S263[0], _S264[1] * _S263[1], _S264[2] * _S263[2]));
        j_10 = j_10 + u32(1);
        l_1 = l_2;
        itot_0 = _S265;
    }
    f_2.w_8 = solve3_0(itot_0, l_1);
    return f_2;
}

fn to_yup_0( v_9 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_9.x, v_9.z, - v_9.y);
}

fn turn_rubble_0( q_rubble_1 : vec4<f32>,  rotvec_zup_0 : vec3<f32>) -> vec4<f32>
{
    return normalize(qmul_0(rotvec_quat_0(to_yup_0(rotvec_zup_0)), q_rubble_1));
}

fn place_body_0( body_4 : u32,  f_3 : Fit_0,  held_mass_0 : vec4<f32>,  held_inertia_0 : vec4<f32>)
{
    var at_3 : u32 = u32(7) * body_4;
    bodies_0[at_3] = vec4<f32>(to_yup_0(f_3.com1_0 + rotate_0(quat_of_0(f_3.turn_0), to_zup_0(bodies_0[at_3].xyz) - f_3.com0_0)), held_mass_0.y);
    var _S266 : u32 = at_3 + u32(1);
    var _S267 : vec4<f32> = bodies_0[_S266];
    var _S268 : vec3<f32> = f_3.turn_0.xyz;
    var _S269 : f32 = length(_S268);
    var _S270 : vec3<f32>;
    if(_S269 > 0.0f)
    {
        _S270 = vec3<f32>((2.0f * atan2(_S269, f_3.turn_0.w))) * normalize(_S268);
    }
    else
    {
        _S270 = vec3<f32>(0.0f);
    }
    bodies_0[_S266] = turn_rubble_0(_S267, _S270);
    bodies_0[at_3 + u32(2)] = held_inertia_0;
    bodies_0[at_3 + u32(3)] = vec4<f32>(to_yup_0(f_3.v_8), 0.0f);
    var _S271 : u32 = at_3 + u32(4);
    bodies_0[_S271] = vec4<f32>(to_yup_0(f_3.w_8), bodies_0[_S271].w);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn learned_apply_bodies(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var e_8 : u32 = id_2.x;
    var _S272 : bool;
    if(ctl_0[u32(0)] == u32(0))
    {
        _S272 = true;
    }
    else
    {
        _S272 = e_8 >= ctl_0[u32(9)];
    }
    if(_S272)
    {
        return;
    }
    var c0_4 : vec4<u32> = cisl_0[u32(2) * e_8];
    var _S273 : u32 = u32(4) * e_8;
    var h0_0 : vec4<f32> = frozen_0[_S273];
    var body_5 : u32 = (bitcast<u32>((frozen_0[_S273].x)));
    var _S274 : u32 = u32(8) * cisl_0[u32(2) * e_8].w + u32(4);
    var stood_0 : bool = (bitcast<u32>((calls_0[_S274].w))) == u32(0);
    if(body_5 == u32(4294967295))
    {
        return;
    }
    var at_4 : u32 = u32(7) * body_5;
    if(!stood_0)
    {
        if((((bitcast<u32>((calls_0[_S274].w))) & (u32(8)))) != u32(0))
        {
            bodies_0[at_4] = vec4<f32>(bodies_0[at_4].xyz, h0_0.y);
            bodies_0[at_4 + u32(2)] = frozen_0[_S273 + u32(1)];
            bodies_0[at_4 + u32(3)] = frozen_0[_S273 + u32(2)];
            bodies_0[at_4 + u32(4)] = frozen_0[_S273 + u32(3)];
        }
        return;
    }
    var _S275 : u32 = c0_4.y;
    if(((status_0[u32(16) + chunks_0[(bitcast<u32>((nodes_0[u32(14) * _S275 + u32(12)].x)))].info_1.y] & (u32(8)))) != u32(0))
    {
        bodies_0[at_4] = vec4<f32>(bodies_0[at_4].xyz, h0_0.y);
        bodies_0[at_4 + u32(2)] = frozen_0[_S273 + u32(1)];
        bodies_0[at_4 + u32(3)] = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    place_body_0(body_5, fit_nodes_0(_S275, c0_4.z, false, u32(0), u32(0)), h0_0, frozen_0[_S273 + u32(1)]);
    return;
}

@compute
@workgroup_size(32, 1, 1)
fn learned_apply_impactors(@builtin(workgroup_id) group_11 : vec3<u32>, @builtin(local_invocation_id) thread_15 : vec3<u32>)
{
    var g_9 : u32 = group_11.x;
    var k_30 : u32 = thread_15.x;
    var _S276 : bool;
    if(ctl_0[u32(0)] == u32(0))
    {
        _S276 = true;
    }
    else
    {
        _S276 = g_9 >= ctl_0[u32(1)];
    }
    if(_S276)
    {
        return;
    }
    var _S277 : u32 = u32(8) * g_9;
    var mask_7 : u32 = (bitcast<u32>((calls_0[_S277 + u32(5)].w)));
    var fb_1 : u32 = (bitcast<u32>((calls_0[_S277 + u32(4)].w)));
    var _S278 : u32 = (u32(1) << (k_30));
    if(((mask_7 & (_S278))) == u32(0))
    {
        _S276 = true;
    }
    else
    {
        if(fb_1 != u32(0))
        {
            _S276 = ((fb_1 & (u32(8)))) == u32(0);
        }
        else
        {
            _S276 = false;
        }
    }
    if(_S276)
    {
        return;
    }
    var _S279 : u32 = u32(4) * (params_0.max_cisl_0 + k_30);
    var at_5 : u32 = u32(7) * (bitcast<u32>((frozen_0[_S279].x)));
    var _S280 : f32 = frozen_0[_S279].y;
    bodies_0[at_5] = vec4<f32>(bodies_0[at_5].xyz, _S280);
    bodies_0[at_5 + u32(2)] = frozen_0[_S279 + u32(1)];
    if(fb_1 != u32(0))
    {
        bodies_0[at_5 + u32(3)] = frozen_0[_S279 + u32(2)];
        bodies_0[at_5 + u32(4)] = frozen_0[_S279 + u32(3)];
        return;
    }
    var n_6 : u32 = (bitcast<u32>((calls_0[_S277 + u32(2)].z))) + (bitcast<u32>((calls_0[_S277 + u32(3)].w))) + countOneBits((mask_7 & ((_S278 - u32(1)))));
    var dr_1 : vec3<f32> = node_0(n_6, u32(11)).xyz;
    bodies_0[at_5] = vec4<f32>(bodies_0[at_5].xyz + to_yup_0(node_0(n_6, u32(10)).xyz), _S280);
    var _S281 : u32 = at_5 + u32(1);
    bodies_0[_S281] = turn_rubble_0(bodies_0[_S281], dr_1);
    bodies_0[at_5 + u32(3)] = vec4<f32>(to_yup_0(node_0(n_6, u32(8)).xyz), 0.0f);
    var _S282 : u32 = at_5 + u32(4);
    bodies_0[_S282] = vec4<f32>(to_yup_0(node_0(n_6, u32(9)).xyz), bodies_0[_S282].w);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn learned_place(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var slot_0 : u32 = id_3.x;
    var _S283 : bool;
    if(ctl_0[u32(18)] == u32(0))
    {
        _S283 = true;
    }
    else
    {
        _S283 = split_control_0[u32(0)] != u32(0);
    }
    if(_S283)
    {
        return;
    }
    var np_0 : u32 = split_control_0[u32(1)];
    if(np_0 == u32(0))
    {
        return;
    }
    var total_1 : u32 = split_parents_0[np_0].y;
    if(slot_0 >= total_1)
    {
        return;
    }
    var _S284 : u32 = np_0 - u32(1);
    var lo_3 : u32 = u32(0);
    var hi_3 : u32 = _S284;
    loop
    {
        if(lo_3 < hi_3)
        {
        }
        else
        {
            break;
        }
        var mid_0 : u32 = (lo_3 + hi_3 + u32(1)) / u32(2);
        if((split_parents_0[mid_0].y) <= slot_0)
        {
            lo_3 = mid_0;
        }
        else
        {
            hi_3 = mid_0 - u32(1);
        }
    }
    var parent_0 : vec4<u32> = split_parents_0[lo_3];
    var _S285 : u32 = parent_0.y;
    if((slot_0 - _S285) >= (parent_0.z))
    {
        return;
    }
    var s_8 : vec4<u32> = split_slots_0[slot_0];
    if((s_8.w) == u32(0))
    {
        return;
    }
    var head_0 : vec4<f32> = split_out_0[u32(8) * slot_0];
    if((bitcast<u32>((head_0.w))) != u32(0))
    {
        return;
    }
    var count_2 : u32 = (bitcast<u32>((head_0.z)));
    var base_1 : u32 = total_1 + _S285 + (bitcast<u32>((head_0.y)));
    var first_2 : u32 = split_firsts_0[lo_3].x;
    if(count_2 == u32(0))
    {
        return;
    }
    var j_11 : u32 = u32(0);
    loop
    {
        if(j_11 < count_2)
        {
        }
        else
        {
            break;
        }
        if(chunk_node_0[first_2 + split_work_0[base_1 + j_11]] == u32(0))
        {
            return;
        }
        j_11 = j_11 + u32(1);
    }
    var _S286 : u32 = s_8.x;
    var at_6 : u32 = u32(7) * _S286;
    place_body_0(_S286, fit_nodes_0(u32(0), count_2, true, base_1, first_2), vec4<f32>(0.0f, bodies_0[at_6].w, 0.0f, 0.0f), bodies_0[at_6 + u32(2)]);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn learned_tally_apply()
{
    if(ctl_0[u32(0)] == u32(0))
    {
        return;
    }
    var g_10 : u32 = u32(0);
    var stale_0 : u32 = u32(0);
    var broken_0 : u32 = u32(0);
    var applied_0 : u32 = u32(0);
    loop
    {
        if(g_10 < ctl_0[u32(1)])
        {
        }
        else
        {
            break;
        }
        var _S287 : u32 = u32(8) * g_10;
        var _S288 : u32 = _S287 + u32(4);
        var _S289 : bool;
        if((bitcast<u32>((calls_0[_S288].w))) != u32(0))
        {
            _S289 = (bitcast<u32>((calls_0[_S288].w))) != u32(8);
        }
        else
        {
            _S289 = false;
        }
        if(_S289)
        {
            g_10 = g_10 + u32(1);
            continue;
        }
        var broken_1 : u32 = broken_0 + (bitcast<u32>((calls_0[_S287 + u32(7)].y)));
        var applied_1 : u32 = applied_0 + (bitcast<u32>((calls_0[_S287 + u32(7)].z)));
        stale_0 = stale_0 + (bitcast<u32>((calls_0[_S287 + u32(7)].x)));
        broken_0 = broken_1;
        applied_0 = applied_1;
        g_10 = g_10 + u32(1);
    }
    ctl_0[u32(18)] = u32(1);
    ctl_0[u32(10)] = stale_0;
    ctl_0[u32(12)] = broken_0;
    ctl_0[u32(11)] = applied_0;
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn learned_tally()
{
    var _S290 : u32 = ctl_0[u32(1)];
    var g_11 : u32 = u32(0);
    var bound_1 : u32 = u32(0);
    var infeasible_0 : u32 = u32(0);
    var fallbacks_0 : u32 = u32(0);
    var nonfinite_0 : u32 = u32(0);
    var fb_inf_0 : u32 = u32(0);
    var range_1 : u32 = u32(0);
    var stood_1 : u32 = u32(0);
    loop
    {
        if(g_11 < _S290)
        {
        }
        else
        {
            break;
        }
        var _S291 : u32 = u32(8) * g_11;
        var _S292 : u32 = _S291 + u32(4);
        var dec_2 : u32 = (bitcast<u32>((calls_0[_S292].z)));
        var fb_2 : u32 = (bitcast<u32>((calls_0[_S292].w)));
        var _S293 : u32;
        if(((dec_2 & (u32(1)))) != u32(0))
        {
            _S293 = u32(1);
        }
        else
        {
            _S293 = u32(0);
        }
        var bound_2 : u32 = bound_1 + _S293;
        var _S294 : u32;
        if(((dec_2 & (u32(2)))) != u32(0))
        {
            _S294 = u32(1);
        }
        else
        {
            _S294 = u32(0);
        }
        var infeasible_1 : u32 = infeasible_0 + _S294;
        if(fb_2 != u32(0))
        {
            var _S295 : u32 = fallbacks_0 + u32(1);
            var _S296 : u32;
            if(((fb_2 & (u32(1)))) != u32(0))
            {
                _S296 = u32(1);
            }
            else
            {
                _S296 = u32(0);
            }
            var nonfinite_1 : u32 = nonfinite_0 + _S296;
            var _S297 : u32;
            if(((fb_2 & (u32(2)))) != u32(0))
            {
                _S297 = u32(1);
            }
            else
            {
                _S297 = u32(0);
            }
            var fb_inf_1 : u32 = fb_inf_0 + _S297;
            var _S298 : u32;
            if(((fb_2 & (u32(4)))) != u32(0))
            {
                _S298 = u32(1);
            }
            else
            {
                _S298 = u32(0);
            }
            var range_2 : u32 = range_1 + _S298;
            fallbacks_0 = _S295;
            nonfinite_0 = nonfinite_1;
            fb_inf_0 = fb_inf_1;
            range_1 = range_2;
        }
        else
        {
            stood_1 = stood_1 + u32(1);
        }
        calls_0[_S291 + u32(7)] = vec4<f32>(0.0f);
        g_11 = g_11 + u32(1);
        bound_1 = bound_2;
        infeasible_0 = infeasible_1;
    }
    ctl_0[u32(5)] = fallbacks_0;
    ctl_0[u32(6)] = bound_1;
    ctl_0[u32(7)] = infeasible_0;
    ctl_0[u32(14)] = nonfinite_0;
    ctl_0[u32(15)] = fb_inf_0;
    ctl_0[u32(16)] = range_1;
    if(stood_1 > u32(0))
    {
        g_11 = u32(1);
    }
    else
    {
        g_11 = u32(0);
    }
    ctl_0[u32(0)] = g_11;
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn learned_adopt(@builtin(workgroup_id) group_12 : vec3<u32>, @builtin(local_invocation_id) thread_16 : vec3<u32>)
{
    var at_7 : u32;
    var g_12 : u32 = group_12.x;
    var t_15 : u32 = thread_16.x;
    if(g_12 >= ctl_0[u32(1)])
    {
        return;
    }
    var _S299 : u32 = u32(8) * g_12;
    var _S300 : u32 = _S299 + u32(2);
    var _S301 : u32 = (bitcast<u32>((calls_0[_S300].z)));
    var _S302 : u32 = (bitcast<u32>((calls_0[_S300].w)));
    var _S303 : u32 = (bitcast<u32>((calls_0[_S299 + u32(3)].w)));
    var _S304 : u32 = _S299 + u32(4);
    var _S305 : u32 = (bitcast<u32>((calls_0[_S304].x)));
    var _S306 : u32 = (bitcast<u32>((calls_0[_S304].y)));
    var _S307 : u32 = _S299 + u32(6);
    var _S308 : u32 = (bitcast<u32>((calls_0[_S307].z)));
    var _S309 : u32 = (bitcast<u32>((calls_0[_S307].w)));
    var j_12 : u32 = t_15;
    loop
    {
        if(j_12 < _S302)
        {
        }
        else
        {
            break;
        }
        if(j_12 < _S303)
        {
            at_7 = params_0.out_bases_0.x + u32(12) * (_S308 + j_12);
        }
        else
        {
            at_7 = params_0.out_bases_0.y + u32(12) * (_S309 + j_12 - _S303);
        }
        var _S310 : u32 = u32(14) * (_S301 + j_12);
        nodes_0[_S310 + u32(5)] = vec4<f32>(lout_0[at_7], lout_0[at_7 + u32(1)], lout_0[at_7 + u32(2)], lout_0[at_7 + u32(3)]);
        nodes_0[_S310 + u32(6)] = vec4<f32>(lout_0[at_7 + u32(4)], lout_0[at_7 + u32(5)], lout_0[at_7 + u32(6)], lout_0[at_7 + u32(7)]);
        nodes_0[_S310 + u32(7)] = vec4<f32>(lout_0[at_7 + u32(8)], lout_0[at_7 + u32(9)], lout_0[at_7 + u32(10)], lout_0[at_7 + u32(11)]);
        j_12 = j_12 + u32(256);
    }
    j_12 = t_15;
    loop
    {
        if(j_12 < _S306)
        {
        }
        else
        {
            break;
        }
        var b_11 : u32 = _S305 + j_12;
        var _S311 : u32 = params_0.out_bases2_0.x + u32(8) * b_11;
        at_7 = u32(0);
        var i_6 : u32 = u32(1);
        loop
        {
            if(i_6 < u32(8))
            {
            }
            else
            {
                break;
            }
            if(lout_0[_S311 + i_6] > lout_0[_S311 + at_7])
            {
                at_7 = i_6;
            }
            i_6 = i_6 + u32(1);
        }
        var _S312 : u32 = u32(2) * b_11;
        bonds_0[_S312] = vec4<f32>(bonds_0[_S312].x, lout_0[params_0.out_bases_0.z + b_11], lout_0[params_0.out_bases_0.w + b_11], (bitcast<f32>((at_7 + u32(1)))));
        j_12 = j_12 + u32(256);
    }
    return;
}

