struct DecodeParams_std140_0
{
    @align(16) call_count_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) log_tau_0 : f32,
    @align(4) tolerance_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : DecodeParams_std140_0;
@binding(1) @group(0) var<storage, read_write> calls_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read_write> nodes_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read_write> bonds_0 : array<vec4<f32>>;

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_0 : vec4<u32>,  thread_0 : u32) -> vec4<u32>
{
    if(thread_0 == u32(0))
    {
        g_group_uniform4_0 = value_0;
    }
    var _S1 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S1;
}

fn node_0( n_0 : u32,  k_0 : u32) -> vec4<f32>
{
    return nodes_0[u32(12) * n_0 + k_0];
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
            var _S2 : u32 = t_1 + s_0;
            g_red_0[i32(0)][t_1] = g_red_0[i32(0)][t_1] + g_red_0[i32(0)][_S2];
            g_red_0[i32(1)][t_1] = g_red_0[i32(1)][t_1] + g_red_0[i32(1)][_S2];
            g_red_0[i32(2)][t_1] = g_red_0[i32(2)][t_1] + g_red_0[i32(2)][_S2];
            g_red_0[i32(3)][t_1] = g_red_0[i32(3)][t_1] + g_red_0[i32(3)][_S2];
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
    var _S3 : f32 = y_1 * y_1;
    var _S4 : f32 = z_1 * z_1;
    var _S5 : f32 = x_1 * y_1;
    var _S6 : f32 = w_1 * z_1;
    var _S7 : f32 = x_1 * z_1;
    var _S8 : f32 = w_1 * y_1;
    var _S9 : f32 = x_1 * x_1;
    var _S10 : f32 = y_1 * z_1;
    var _S11 : f32 = w_1 * x_1;
    return mat3x3<f32>(vec3<f32>(1.0f - 2.0f * (_S3 + _S4), 2.0f * (_S5 - _S6), 2.0f * (_S7 + _S8)), vec3<f32>(2.0f * (_S5 + _S6), 1.0f - 2.0f * (_S9 + _S4), 2.0f * (_S10 - _S11)), vec3<f32>(2.0f * (_S7 - _S8), 2.0f * (_S10 + _S11), 1.0f - 2.0f * (_S9 + _S3)));
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
    var _S12 : f32 = r_2.z;
    var _S13 : f32 = r_2.y;
    var _S14 : f32 = r_2.x;
    return mat3x3<f32>(vec3<f32>(0.0f, - _S12, _S13), vec3<f32>(_S12, 0.0f, - _S14), vec3<f32>(- _S13, _S14, 0.0f));
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
    var _S15 : mat3x3<f32> = mat3x3<f32>(sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0));
    var _S16 : mat3x3<f32> = (((k_1) * (k_1)));
    var _S17 : mat3x3<f32> = mat3x3<f32>(1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0));
    return eye_0 + mat3x3<f32>(_S15[0] * k_1[0], _S15[1] * k_1[1], _S15[2] * k_1[2]) + mat3x3<f32>(_S17[0] * _S16[0], _S17[1] * _S16[1], _S17[2] * _S16[2]);
}

fn outer3_0( a_2 : vec3<f32>,  b_2 : vec3<f32>) -> mat3x3<f32>
{
    return mat3x3<f32>(a_2 * vec3<f32>(b_2.x), a_2 * vec3<f32>(b_2.y), a_2 * vec3<f32>(b_2.z));
}

var<workgroup> g_lambda_0 : array<f32, i32(6)>;

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_1 : u32,  thread_1 : u32) -> u32
{
    if(thread_1 == u32(0))
    {
        g_group_uniform_0 = value_1;
    }
    var _S18 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S18;
}

fn solve3_0( a_3 : mat3x3<f32>,  b_3 : vec3<f32>) -> vec3<f32>
{
    var c0_0 : vec3<f32> = vec3<f32>(a_3[i32(0)][i32(0)], a_3[i32(1)][i32(0)], a_3[i32(2)][i32(0)]);
    var c1_0 : vec3<f32> = vec3<f32>(a_3[i32(0)][i32(1)], a_3[i32(1)][i32(1)], a_3[i32(2)][i32(1)]);
    var c2_0 : vec3<f32> = vec3<f32>(a_3[i32(0)][i32(2)], a_3[i32(1)][i32(2)], a_3[i32(2)][i32(2)]);
    var _S19 : vec3<f32> = cross(c1_0, c2_0);
    return vec3<f32>(dot(b_3, _S19), dot(c0_0, cross(b_3, c2_0)), dot(c0_0, cross(c1_0, b_3))) / vec3<f32>(dot(c0_0, _S19));
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
    if(g_0 >= (params_0.call_count_0))
    {
        return;
    }
    var _S20 : u32 = u32(8) * g_0;
    var _S21 : u32 = _S20 + u32(4);
    var c4_0 : vec4<f32> = calls_0[_S21];
    var _S22 : f32 = calls_0[_S20].x;
    var _S23 : f32 = calls_0[_S20].z;
    var dt_0 : f32 = calls_0[_S20 + u32(1)].w;
    var e_in_0 : f32 = calls_0[_S20 + u32(2)].x;
    var _S24 : f32 = calls_0[_S21].y;
    var u_0 : vec4<u32> = group_uniform4_0(vec4<u32>((bitcast<u32>((calls_0[_S20 + u32(2)].y))), (bitcast<u32>((calls_0[_S20 + u32(2)].z))), (bitcast<u32>((calls_0[_S20 + u32(2)].w))), (bitcast<u32>((_S24)))), t_2);
    var flags_0 : u32 = u_0.x;
    var _S25 : u32 = u_0.y;
    var _S26 : u32 = u_0.z;
    var _S27 : u32 = u_0.w;
    var _S28 : f32 = calls_0[_S21].x;
    var _S29 : u32 = (bitcast<u32>((_S28)));
    var grav_0 : vec3<f32> = calls_0[_S20 + u32(3)].xyz;
    var free_0 : bool = ((flags_0 & (u32(1)))) != u32(0);
    var j_0 : u32 = t_2;
    loop
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        var k_2 : u32 = _S25 + j_0;
        var v_1 : vec3<f32> = node_0(k_2, u32(3)).xyz;
        var w_2 : vec3<f32> = node_0(k_2, u32(4)).xyz;
        var o0_0 : vec4<f32> = node_0(k_2, u32(5));
        var o1_0 : vec4<f32> = node_0(k_2, u32(6));
        var o2_0 : vec4<f32> = node_0(k_2, u32(7));
        var qq_0 : Quat_0 = quat_of_0(node_0(k_2, u32(1)));
        var _S30 : vec3<f32> = vec3<f32>(_S23);
        var _S31 : vec3<f32> = v_1 + rotate_0(qq_0, o0_0.xyz) * _S30;
        var _S32 : vec3<f32> = vec3<f32>(_S22);
        var _S33 : vec3<f32> = w_2 + rotate_0(qq_0, vec3<f32>(o0_0.w, o1_0.x, o1_0.y)) * _S30 / _S32;
        var _S34 : vec3<f32> = vec3<f32>(dt_0);
        var _S35 : vec3<f32> = v_1 * _S34 + vec3<f32>(0.5f) * grav_0 * _S34 * _S34 + rotate_0(qq_0, vec3<f32>(o1_0.z, o1_0.w, o2_0.x)) * _S30 * _S34;
        var _S36 : vec3<f32> = w_2 * _S34 + rotate_0(qq_0, o2_0.yzw) * _S30 * _S34 / _S32;
        var dx_0 : vec3<f32>;
        var dr_0 : vec3<f32>;
        if((node_0(k_2, u32(2)).w) != 0.0f)
        {
            var _S37 : vec3<f32> = vec3<f32>(0.0f);
            v_end_0 = _S37;
            w_end_0 = _S37;
            dx_0 = _S37;
            dr_0 = _S37;
        }
        else
        {
            v_end_0 = _S31;
            w_end_0 = _S33;
            dx_0 = _S35;
            dr_0 = _S36;
        }
        var _S38 : u32 = u32(12) * k_2;
        nodes_0[_S38 + u32(8)] = vec4<f32>(v_end_0, 0.0f);
        nodes_0[_S38 + u32(9)] = vec4<f32>(w_end_0, 0.0f);
        nodes_0[_S38 + u32(10)] = vec4<f32>(dx_0, 0.0f);
        nodes_0[_S38 + u32(11)] = vec4<f32>(dr_0, 0.0f);
        j_0 = j_0 + u32(256);
    }
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    var a_4 : vec4<f32>;
    var b_4 : vec4<f32>;
    var c_2 : vec4<f32>;
    var d_3 : vec4<f32>;
    var _S39 : vec4<f32> = vec4<f32>(0.0f);
    j_0 = t_2;
    var sa_0 : vec4<f32> = _S39;
    var sb_0 : vec4<f32> = _S39;
    var sc_0 : vec4<f32> = _S39;
    loop
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        var k_3 : u32 = _S25 + j_0;
        var p_0 : vec4<f32> = node_0(k_3, u32(0));
        var m_0 : f32 = p_0.w;
        var _S40 : vec3<f32> = vec3<f32>(m_0);
        var sa_1 : vec4<f32> = sa_0 + vec4<f32>(p_0.xyz * _S40, m_0);
        var sb_1 : vec4<f32> = sb_0 + vec4<f32>(node_0(k_3, u32(3)).xyz * _S40, 0.0f);
        var sc_1 : vec4<f32> = sc_0 + vec4<f32>(node_0(k_3, u32(10)).xyz * _S40, 0.0f);
        j_0 = j_0 + u32(256);
        sa_0 = sa_1;
        sb_0 = sb_1;
        sc_0 = sc_1;
    }
    reduce_store_0(t_2, sa_0, sb_0, sc_0, _S39);
    reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
    var mt_0 : f32 = a_4.w;
    var _S41 : vec3<f32> = vec3<f32>(mt_0);
    var _S42 : vec3<f32> = a_4.xyz / _S41;
    var p0_0 : vec3<f32> = b_4.xyz;
    var have_0 : vec3<f32> = c_2.xyz;
    var _S43 : vec3<f32> = vec3<f32>(0.0f);
    if(free_0)
    {
        var _S44 : vec3<f32> = vec3<f32>(dt_0);
        v_end_0 = (p0_0 * _S44 + vec3<f32>((0.5f * mt_0)) * grav_0 * _S44 * _S44 - have_0) / _S41;
    }
    else
    {
        v_end_0 = _S43;
    }
    j_0 = t_2;
    sa_0 = _S39;
    sb_0 = _S39;
    loop
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        var k_4 : u32 = _S25 + j_0;
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
        var _S45 : vec3<f32> = p_1.xyz;
        var _S46 : vec3<f32> = vec3<f32>(m_1);
        var sa_2 : vec4<f32> = sa_0 + vec4<f32>((_S45 + w_end_0) * _S46, 0.0f);
        var sb_2 : vec4<f32> = sb_0 + vec4<f32>(cross(_S45 - _S42, _S46 * node_0(k_4, u32(3)).xyz) + (((node_0(k_4, u32(4)).xyz) * (world_inertia_0(node_0(k_4, u32(1)), node_0(k_4, u32(2)).xyz)))), 0.0f);
        j_0 = j_0 + u32(256);
        sa_0 = sa_2;
        sb_0 = sb_2;
    }
    reduce_store_0(t_2, sa_0, sb_0, _S39, _S39);
    reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
    var _S47 : vec3<f32> = a_4.xyz / _S41;
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
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            var k_5 : u32 = _S25 + j_0;
            nodes_0[u32(12) * k_5 + u32(10)] = vec4<f32>(node_0(k_5, u32(10)).xyz + v_end_0, 0.0f);
            j_0 = j_0 + u32(256);
        }
        j_0 = t_2;
        sa_0 = _S39;
        sb_0 = _S39;
        sc_0 = _S39;
        ta_0 = _S39;
        tb_0 = _S39;
        tc_0 = _S39;
        loop
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            var k_6 : u32 = _S25 + j_0;
            var p_2 : vec4<f32> = node_0(k_6, u32(0));
            var m_2 : f32 = p_2.w;
            var r_4 : vec3<f32> = p_2.xyz + node_0(k_6, u32(10)).xyz - _S47;
            var rd_0 : mat3x3<f32> = rotvec_matrix_0(node_0(k_6, u32(11)).xyz);
            var i1_0 : mat3x3<f32> = (((transpose(rd_0)) * ((((world_inertia_0(node_0(k_6, u32(1)), node_0(k_6, u32(2)).xyz)) * (rd_0))))));
            var _S48 : vec3<f32> = vec3<f32>(m_2);
            var _S49 : vec3<f32> = _S48 * node_0(k_6, u32(8)).xyz;
            var sa_3 : vec4<f32> = sa_0 + vec4<f32>(_S49, 0.0f);
            var sb_3 : vec4<f32> = sb_0 + vec4<f32>(cross(r_4, _S49) + (((node_0(k_6, u32(9)).xyz) * (i1_0))), 0.0f);
            var sc_2 : vec4<f32> = sc_0 + vec4<f32>(_S48 * r_4, 0.0f);
            var _S50 : mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f));
            var _S51 : mat3x3<f32> = mat3x3<f32>(dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4));
            var _S52 : mat3x3<f32> = mat3x3<f32>(_S51[0] * _S50[0], _S51[1] * _S50[1], _S51[2] * _S50[2]) - outer3_0(r_4, r_4);
            var _S53 : mat3x3<f32> = mat3x3<f32>(m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2);
            var jj_0 : mat3x3<f32> = mat3x3<f32>(_S53[0] * _S52[0], _S53[1] * _S52[1], _S53[2] * _S52[2]) + i1_0;
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
        reduce_store_0(t_2, tb_0, tc_0, _S39, _S39);
        var j1_0 : vec4<f32>;
        var j2_0 : vec4<f32>;
        var unused0_0 : vec4<f32>;
        var unused1_0 : vec4<f32>;
        reduce_read_0(&(j1_0), &(j2_0), &(unused0_0), &(unused1_0));
        if(t_2 == u32(0))
        {
            var _S54 : vec3<f32> = p0_0 + _S41 * grav_0 * vec3<f32>(dt_0) - ax_p_0.xyz;
            var _S55 : vec3<f32> = l0_0 - ax_l_0.xyz;
            var _S56 : mat3x3<f32> = skew_0(smr_0.xyz);
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
            var _S57 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( j0_0.xyz, j1_0.xyz, j2_0.xyz );
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
                    var _S58 : u32 = u32(3) + q_4;
                    m6_0[dec_0][_S58] = - _S56[dec_0][q_4];
                    m6_0[u32(3) + dec_0][q_4] = _S56[dec_0][q_4];
                    m6_0[u32(3) + dec_0][_S58] = _S57[dec_0][q_4];
                    q_4 = q_4 + u32(1);
                }
                m6_0[dec_0][i32(6)] = _S54[dec_0];
                m6_0[u32(3) + dec_0][i32(6)] = _S55[dec_0];
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
                var _S59 : u32 = bad_0 + u32(1);
                q_4 = bad_0;
                dec_0 = _S59;
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
                var r_5 : u32 = _S59;
                loop
                {
                    if(r_5 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S60 : f32 = m6_0[r_5][bad_0] / m6_0[bad_0][bad_0];
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
                        m6_0[r_5][q_6] = m6_0[r_5][q_6] - _S60 * m6_0[bad_0][q_6];
                        q_6 = q_6 + u32(1);
                    }
                    r_5 = r_5 + u32(1);
                }
                bad_0 = _S59;
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
        var _S61 : vec3<f32> = vec3<f32>(g_lambda_0[i32(0)], g_lambda_0[i32(1)], g_lambda_0[i32(2)]);
        var _S62 : vec3<f32> = vec3<f32>(g_lambda_0[i32(3)], g_lambda_0[i32(4)], g_lambda_0[i32(5)]);
        j_0 = t_2;
        loop
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            var k_7 : u32 = _S25 + j_0;
            var _S63 : u32 = u32(12) * k_7;
            nodes_0[_S63 + u32(8)] = vec4<f32>(node_0(k_7, u32(8)).xyz + _S61 + cross(_S62, node_0(k_7, u32(0)).xyz + node_0(k_7, u32(10)).xyz - _S47), 0.0f);
            nodes_0[_S63 + u32(9)] = vec4<f32>(node_0(k_7, u32(9)).xyz + _S62, 0.0f);
            j_0 = j_0 + u32(256);
        }
        storageBarrier(); textureBarrier(); workgroupBarrier();;
    }
    if(((flags_0 & (u32(2)))) == u32(0))
    {
        j_0 = t_2;
        sa_0 = _S39;
        sb_0 = _S39;
        loop
        {
            if(j_0 < _S26)
            {
            }
            else
            {
                break;
            }
            var k_8 : u32 = _S25 + j_0;
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
        reduce_store_0(t_2, sa_0, sb_0, _S39, _S39);
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
        var _S64 : u32 = group_uniform_0(j_0, t_2);
        if(_S64 != u32(0))
        {
            if(free_0)
            {
                var vc_0 : vec3<f32> = b_4.xyz / _S41;
                j_0 = t_2;
                ta_0 = _S39;
                tb_0 = _S39;
                tc_0 = _S39;
                var td_0 : vec4<f32> = _S39;
                loop
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_9 : u32 = _S25 + j_0;
                    var m_4 : f32 = node_0(k_9, u32(0)).w;
                    var r_7 : vec3<f32> = node_0(k_9, u32(0)).xyz + node_0(k_9, u32(10)).xyz - _S47;
                    var rd_2 : mat3x3<f32> = rotvec_matrix_0(node_0(k_9, u32(11)).xyz);
                    var i1_1 : mat3x3<f32> = (((transpose(rd_2)) * ((((world_inertia_0(node_0(k_9, u32(1)), node_0(k_9, u32(2)).xyz)) * (rd_2))))));
                    var _S65 : mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0f, 0.0f, 0.0f), vec3<f32>(0.0f, 1.0f, 0.0f), vec3<f32>(0.0f, 0.0f, 1.0f));
                    var _S66 : mat3x3<f32> = mat3x3<f32>(dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7), dot(r_7, r_7));
                    var _S67 : mat3x3<f32> = mat3x3<f32>(_S66[0] * _S65[0], _S66[1] * _S65[1], _S66[2] * _S65[2]) - outer3_0(r_7, r_7);
                    var _S68 : mat3x3<f32> = mat3x3<f32>(m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4);
                    var jj_1 : mat3x3<f32> = mat3x3<f32>(_S68[0] * _S67[0], _S68[1] * _S67[1], _S68[2] * _S67[2]) + i1_1;
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
                var ra_0 : vec4<f32> = _S39;
                j_0 = t_2;
                loop
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_10 : u32 = _S25 + j_0;
                    var rd_3 : mat3x3<f32> = rotvec_matrix_0(node_0(k_10, u32(11)).xyz);
                    var uu_0 : vec3<f32> = node_0(k_10, u32(8)).xyz - vc_0 - cross(omega_0, node_0(k_10, u32(0)).xyz + node_0(k_10, u32(10)).xyz - _S47);
                    var wr_0 : vec3<f32> = node_0(k_10, u32(9)).xyz - omega_0;
                    ra_0[i32(0)] = ra_0[i32(0)] + 0.5f * (node_0(k_10, u32(0)).w * dot(uu_0, uu_0) + dot(wr_0, (((wr_0) * ((((transpose(rd_3)) * ((((world_inertia_0(node_0(k_10, u32(1)), node_0(k_10, u32(2)).xyz)) * (rd_3)))))))))));
                    j_0 = j_0 + u32(256);
                }
                reduce_store_0(t_2, ra_0, _S39, _S39, _S39);
                reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
                var _S69 : f32 = sqrt(clamp((bound_0 - ke_base_0) / max(a_4.x, 1.00000000317107685e-30f), 0.0f, 1.0f));
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
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_11 : u32 = _S25 + j_0;
                    var _S70 : vec3<f32> = cross(omega_0, node_0(k_11, u32(0)).xyz + node_0(k_11, u32(10)).xyz - _S47);
                    var wr_1 : vec3<f32> = node_0(k_11, u32(9)).xyz - omega_0;
                    var _S71 : u32 = u32(12) * k_11;
                    var _S72 : vec3<f32> = vec3<f32>(_S69);
                    nodes_0[_S71 + u32(8)] = vec4<f32>(vc_0 + _S70 + _S72 * (node_0(k_11, u32(8)).xyz - vc_0 - _S70), 0.0f);
                    nodes_0[_S71 + u32(9)] = vec4<f32>(omega_0 + _S72 * wr_1, 0.0f);
                    j_0 = j_0 + u32(256);
                }
            }
            else
            {
                var _S73 : f32 = sqrt(min(max(bound_0, 0.0f) / max(ke_0, 1.00000000317107685e-30f), 1.0f));
                j_0 = t_2;
                loop
                {
                    if(j_0 < _S26)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var k_12 : u32 = _S25 + j_0;
                    var _S74 : u32 = u32(12) * k_12;
                    var _S75 : vec3<f32> = vec3<f32>(_S73);
                    nodes_0[_S74 + u32(8)] = vec4<f32>(node_0(k_12, u32(8)).xyz * _S75, 0.0f);
                    nodes_0[_S74 + u32(9)] = vec4<f32>(node_0(k_12, u32(9)).xyz * _S75, 0.0f);
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
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        var _S76 : u32 = u32(2) * (_S29 + j_0);
        var _S77 : u32 = _S76 + u32(1);
        var b1_0 : vec4<f32> = bonds_0[_S77];
        var _S78 : f32 = min(bonds_0[_S76].x + bonds_0[_S76].z, 0.99999898672103882f);
        if((bonds_0[_S76].y) >= (params_0.log_tau_0))
        {
            bad_0 = u32(1);
        }
        else
        {
            bad_0 = u32(0);
        }
        bonds_0[_S77] = vec4<f32>(_S78, (bitcast<f32>((bad_0))), b1_0.z, b1_0.w);
        j_0 = j_0 + u32(256);
    }
    bad_0 = u32(0);
    j_0 = t_2;
    loop
    {
        if(j_0 < _S26)
        {
        }
        else
        {
            break;
        }
        var k_13 : u32 = _S25 + j_0;
        var _S79 : vec4<f32> = vec4<f32>(1.0f);
        if(!((abs(dot(node_0(k_13, u32(8)), _S79) + dot(node_0(k_13, u32(9)), _S79) + dot(node_0(k_13, u32(10)), _S79) + dot(node_0(k_13, u32(11)), _S79))) < 3.00000000549775576e+38f))
        {
            bad_0 = u32(1);
        }
        j_0 = j_0 + u32(256);
    }
    reduce_store_0(t_2, vec4<f32>(f32(bad_0), 0.0f, 0.0f, 0.0f), _S39, _S39, _S39);
    reduce_read_0(&(a_4), &(b_4), &(c_2), &(d_3));
    if((a_4.x) > 0.0f)
    {
        dec_0 = (dec_0 | (u32(4)));
    }
    if(t_2 == u32(0))
    {
        calls_0[_S21] = vec4<f32>(_S28, _S24, (bitcast<f32>((dec_0))), c4_0.w);
    }
    return;
}

