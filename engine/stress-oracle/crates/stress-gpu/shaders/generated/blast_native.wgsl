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
    @align(4) wide_min_0 : u32,
    @align(16) wide_rows_0 : u32,
    @align(4) wide_blocks_0 : u32,
    @align(8) opaque_zero_0 : u32,
    @align(4) first_polynomial_0 : u32,
    @align(16) rotation_0 : u32,
    @align(4) node_vectors_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(1) @group(0) var<storage, read> islands_0 : array<vec4<u32>>;

@binding(8) @group(0) var<storage, read_write> report_0 : array<vec4<u32>>;

@binding(9) @group(0) var<storage, read> lists_0 : array<u32>;

@binding(5) @group(0) var<storage, read> accel_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

struct BlastNode_std430_0
{
    @align(16) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(8) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
    @align(16) position_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> nodes_0 : array<BlastNode_std430_0>;

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

@binding(10) @group(0) var<storage, read> stiffness_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> x_0 : array<vec4<f32>>;

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

fn ns_0( n_0 : u32,  v_0 : u32) -> u32
{
    return n_0 * params_0.node_vectors_0 + v_0;
}

var<workgroup> g_blast_red_0 : array<vec2<f32>, i32(1024)>;

var<workgroup> g_blast_sum_0 : vec2<f32>;

fn blast_group_sum_0( tid_0 : u32,  v_1 : vec2<f32>) -> vec2<f32>
{
    g_blast_red_0[tid_0] = v_1;
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

fn set_v_0( n_1 : u32,  v_2 : u32,  a_1 : vec3<f32>,  l_0 : vec3<f32>)
{
    var _S9 : u32 = ns_0(n_1, v_2);
    scratch_0[_S9] = vec4<f32>(a_1, 0.0f);
    scratch_0[_S9 + u32(1)] = vec4<f32>(l_0, 0.0f);
    return;
}

fn stiff_0( j_0 : u32,  which_0 : u32,  v_3 : vec3<f32>) -> vec3<f32>
{
    var _S10 : u32 = u32(6) * which_0;
    var m_0 : array<f32, i32(6)>;
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S11 : u32 = _S10 + k_1;
        m_0[k_1] = stiffness_0[j_0 * u32(6) + _S11 / u32(4)][_S11 % u32(4)];
        k_1 = k_1 + u32(1);
    }
    var _S12 : f32 = v_3.x;
    var _S13 : f32 = v_3.y;
    var _S14 : f32 = v_3.z;
    return vec3<f32>(m_0[i32(0)] * _S12 + m_0[i32(3)] * _S13 + m_0[i32(4)] * _S14, m_0[i32(3)] * _S12 + m_0[i32(1)] * _S13 + m_0[i32(5)] * _S14, m_0[i32(4)] * _S12 + m_0[i32(5)] * _S13 + m_0[i32(2)] * _S14);
}

fn sym3_0( j_1 : u32,  which_1 : u32) -> mat3x3<f32>
{
    var c0_0 : vec3<f32> = stiff_0(j_1, which_1, vec3<f32>(1.0f, 0.0f, 0.0f));
    var c1_0 : vec3<f32> = stiff_0(j_1, which_1, vec3<f32>(0.0f, 1.0f, 0.0f));
    var c2_0 : vec3<f32> = stiff_0(j_1, which_1, vec3<f32>(0.0f, 0.0f, 1.0f));
    return mat3x3<f32>(c0_0.x, c1_0.x, c2_0.x, c0_0.y, c1_0.y, c2_0.y, c0_0.z, c1_0.z, c2_0.z);
}

fn skew_0( o_0 : vec3<f32>) -> mat3x3<f32>
{
    var _S15 : f32 = o_0.z;
    var _S16 : f32 = o_0.y;
    var _S17 : f32 = o_0.x;
    return mat3x3<f32>(0.0f, - _S15, _S16, _S15, 0.0f, - _S17, - _S16, _S17, 0.0f);
}

fn build_full_block_0( node_0 : u32) -> bool
{
    var c_0 : u32;
    var k_2 : u32;
    var s_2 : f32;
    var n_2 : BlastNode_std430_0 = nodes_0[node_0];
    var aa_0 : mat3x3<f32> = mat3x3<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    var r_0 : u32 = n_2.ref_begin_0;
    var aa_1 : mat3x3<f32> = aa_0;
    var al_0 : mat3x3<f32> = aa_0;
    var ll_0 : mat3x3<f32> = aa_0;
    loop
    {
        if(r_0 < (n_2.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_0 : u32 = refs_0[r_0];
        var j_2 : u32 = (ref_0 & (u32(2147483647)));
        var b_1 : BlastBond_std430_0 = bonds_0[j_2];
        if(!((bonds_0[j_2].health_0) > 0.0f))
        {
            r_0 = r_0 + u32(1);
            continue;
        }
        var o_1 : vec3<f32>;
        if(((ref_0 & (u32(2147483648)))) != u32(0))
        {
            o_1 = b_1.o1_0.xyz;
        }
        else
        {
            o_1 = b_1.o0_0.xyz;
        }
        var _S18 : f32 = b_1.s_0;
        var s2_0 : f32 = _S18 * _S18;
        var wl_0 : mat3x3<f32> = sym3_0(j_2, u32(3));
        var k_3 : mat3x3<f32> = skew_0(o_1);
        var _S19 : mat3x3<f32> = (((wl_0) * (k_3)));
        var _S20 : mat3x3<f32> = sym3_0(j_2, u32(1)) + (((transpose(k_3)) * (_S19)));
        var _S21 : mat3x3<f32> = mat3x3<f32>(s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        var _S22 : mat3x3<f32> = mat3x3<f32>(- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0);
        var _S23 : mat3x3<f32> = al_0 + mat3x3<f32>(_S22[0] * _S19[0], _S22[1] * _S19[1], _S22[2] * _S19[2]);
        var _S24 : mat3x3<f32> = ll_0 + mat3x3<f32>(_S21[0] * wl_0[0], _S21[1] * wl_0[1], _S21[2] * wl_0[2]);
        aa_1 = aa_1 + mat3x3<f32>(_S21[0] * _S20[0], _S21[1] * _S20[1], _S21[2] * _S20[2]);
        al_0 = _S23;
        ll_0 = _S24;
        r_0 = r_0 + u32(1);
    }
    var m_1 : array<array<f32, i32(6)>, i32(6)>;
    r_0 = u32(0);
    loop
    {
        if(r_0 < u32(3))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            m_1[r_0][c_0] = aa_1[r_0][c_0] * n_2.w_ang_0 * n_2.w_ang_0;
            var _S25 : u32 = c_0 + u32(3);
            var _S26 : f32 = al_0[r_0][c_0] * n_2.w_ang_0 * n_2.w_lin_0;
            m_1[r_0][_S25] = _S26;
            m_1[_S25][r_0] = _S26;
            m_1[r_0 + u32(3)][_S25] = ll_0[r_0][c_0] * n_2.w_lin_0 * n_2.w_lin_0;
            c_0 = c_0 + u32(1);
        }
        r_0 = r_0 + u32(1);
    }
    var packed_0 : array<f32, i32(24)>;
    var z_0 : u32 = u32(0);
    loop
    {
        if(z_0 < u32(24))
        {
        }
        else
        {
            break;
        }
        packed_0[z_0] = 0.0f;
        z_0 = z_0 + u32(1);
    }
    r_0 = u32(0);
    loop
    {
        if(r_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + u32(1)) / u32(2) + c_0] = m_1[r_0][c_0];
            c_0 = c_0 + u32(1);
        }
        r_0 = r_0 + u32(1);
    }
    var q_0 : u32 = u32(0);
    loop
    {
        if(q_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S27 : u32 = u32(4) * q_0;
        scratch_0[ns_0(node_0, u32(18) + q_0)] = vec4<f32>(packed_0[_S27], packed_0[_S27 + u32(1)], packed_0[_S27 + u32(2)], packed_0[_S27 + u32(3)]);
        q_0 = q_0 + u32(1);
    }
    var l_1 : array<array<f32, i32(6)>, i32(6)>;
    r_0 = u32(0);
    loop
    {
        if(r_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 < u32(6))
            {
            }
            else
            {
                break;
            }
            l_1[r_0][c_0] = 0.0f;
            c_0 = c_0 + u32(1);
        }
        r_0 = r_0 + u32(1);
    }
    r_0 = u32(0);
    loop
    {
        if(r_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            k_2 = u32(0);
            s_2 = m_1[r_0][c_0];
            loop
            {
                if(k_2 < c_0)
                {
                }
                else
                {
                    break;
                }
                var s_3 : f32 = s_2 - l_1[r_0][k_2] * l_1[c_0][k_2];
                k_2 = k_2 + u32(1);
                s_2 = s_3;
            }
            if(r_0 == c_0)
            {
                var _S28 : bool;
                if(!(s_2 > 0.0f))
                {
                    _S28 = true;
                }
                else
                {
                    _S28 = !isfinite_0(s_2);
                }
                if(_S28)
                {
                    return false;
                }
                l_1[r_0][c_0] = sqrt(s_2);
            }
            else
            {
                l_1[r_0][c_0] = s_2 / l_1[c_0][c_0];
            }
            c_0 = c_0 + u32(1);
        }
        r_0 = r_0 + u32(1);
    }
    var inv_0 : array<array<f32, i32(6)>, i32(6)>;
    c_0 = u32(0);
    loop
    {
        if(c_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var y_0 : array<f32, i32(6)>;
        r_0 = u32(0);
        loop
        {
            if(r_0 < u32(6))
            {
            }
            else
            {
                break;
            }
            if(r_0 == c_0)
            {
                s_2 = 1.0f;
            }
            else
            {
                s_2 = 0.0f;
            }
            k_2 = u32(0);
            var s_4 : f32 = s_2;
            loop
            {
                if(k_2 < r_0)
                {
                }
                else
                {
                    break;
                }
                var s_5 : f32 = s_4 - l_1[r_0][k_2] * y_0[k_2];
                k_2 = k_2 + u32(1);
                s_4 = s_5;
            }
            y_0[r_0] = s_4 / l_1[r_0][r_0];
            r_0 = r_0 + u32(1);
        }
        var r_1 : i32 = i32(5);
        loop
        {
            if(r_1 >= i32(0))
            {
            }
            else
            {
                break;
            }
            k_2 = u32(r_1) + u32(1);
            s_2 = y_0[r_1];
            loop
            {
                if(k_2 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var s_6 : f32 = s_2 - l_1[k_2][r_1] * inv_0[k_2][c_0];
                k_2 = k_2 + u32(1);
                s_2 = s_6;
            }
            inv_0[r_1][c_0] = s_2 / l_1[r_1][r_1];
            r_1 = r_1 - i32(1);
        }
        c_0 = c_0 + u32(1);
    }
    r_0 = u32(0);
    loop
    {
        if(r_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + u32(1)) / u32(2) + c_0] = inv_0[r_0][c_0];
            c_0 = c_0 + u32(1);
        }
        r_0 = r_0 + u32(1);
    }
    q_0 = u32(0);
    loop
    {
        if(q_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S29 : u32 = u32(4) * q_0;
        scratch_0[ns_0(node_0, u32(24) + q_0)] = vec4<f32>(packed_0[_S29], packed_0[_S29 + u32(1)], packed_0[_S29 + u32(2)], packed_0[_S29 + u32(3)]);
        q_0 = q_0 + u32(1);
    }
    return true;
}

fn build_block_0( node_1 : u32) -> bool
{
    if((params_0.rotation_0) == u32(2))
    {
        var _S30 : bool = build_full_block_0(node_1);
        return _S30;
    }
    var n_3 : BlastNode_std430_0 = nodes_0[node_1];
    var _S31 : vec3<f32> = vec3<f32>(0.0f);
    var r_2 : u32 = n_3.ref_begin_0;
    var diag_0 : vec3<f32> = _S31;
    var offd_0 : vec3<f32> = _S31;
    var ks_0 : vec3<f32> = _S31;
    var cs_0 : f32 = 0.0f;
    loop
    {
        if(r_2 < (n_3.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_1 : u32 = refs_0[r_2];
        var j_3 : u32 = (ref_1 & (u32(2147483647)));
        var b_2 : BlastBond_std430_0 = bonds_0[j_3];
        if(!((bonds_0[j_3].health_0) > 0.0f))
        {
            r_2 = r_2 + u32(1);
            continue;
        }
        var o_2 : vec3<f32>;
        if(((ref_1 & (u32(2147483648)))) != u32(0))
        {
            o_2 = b_2.o1_0.xyz;
        }
        else
        {
            o_2 = b_2.o0_0.xyz;
        }
        var _S32 : f32 = b_2.s_0;
        var s2_1 : f32 = _S32 * _S32;
        var _S33 : vec3<f32> = vec3<f32>(1.0f);
        var wd_0 : vec3<f32>;
        var wo_0 : vec3<f32>;
        if((params_0.rotation_0) != u32(0))
        {
            var c1_1 : vec3<f32> = stiff_0(j_3, u32(1), vec3<f32>(0.0f, 1.0f, 0.0f));
            var c2_1 : vec3<f32> = stiff_0(j_3, u32(1), vec3<f32>(0.0f, 0.0f, 1.0f));
            var _S34 : vec3<f32> = vec3<f32>(c1_1.x, c2_1.x, c2_1.y);
            wd_0 = vec3<f32>(stiff_0(j_3, u32(1), vec3<f32>(1.0f, 0.0f, 0.0f)).x, c1_1.y, c2_1.z);
            wo_0 = _S34;
        }
        else
        {
            wd_0 = _S33;
            wo_0 = _S31;
        }
        var _S35 : vec3<f32> = vec3<f32>(s2_1);
        var _S36 : f32 = o_2.x;
        var _S37 : f32 = o_2.y;
        var _S38 : f32 = o_2.z;
        var offd_1 : vec3<f32> = offd_0 + _S35 * (wo_0 - vec3<f32>(_S36 * _S37, _S36 * _S38, _S37 * _S38));
        var ks_1 : vec3<f32> = ks_0 + _S35 * o_2;
        var cs_1 : f32 = cs_0 + s2_1;
        diag_0 = diag_0 + _S35 * (wd_0 + vec3<f32>(dot(o_2, o_2)) - o_2 * o_2);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_2 = r_2 + u32(1);
    }
    var _S39 : f32 = n_3.w_ang_0;
    var da_0 : f32 = _S39 * _S39;
    var c_1 : f32 = n_3.w_lin_0 * n_3.w_lin_0 * cs_0;
    var k_4 : vec3<f32> = ks_0 * vec3<f32>((n_3.w_ang_0 * n_3.w_lin_0));
    var a0_0 : vec4<f32> = vec4<f32>(diag_0 * vec3<f32>(da_0), offd_0.x * da_0);
    var _S40 : f32 = offd_0.y * da_0;
    var _S41 : f32 = offd_0.z * da_0;
    var a1_0 : vec4<f32> = vec4<f32>(_S40, _S41, c_1, 0.0f);
    var _S42 : u32 = ns_0(node_1, u32(18));
    scratch_0[_S42] = a0_0;
    scratch_0[_S42 + u32(1)] = a1_0;
    scratch_0[_S42 + u32(2)] = vec4<f32>(k_4, 0.0f);
    if(!(c_1 > 0.0f))
    {
        return false;
    }
    var ic_0 : f32 = 1.0f / c_1;
    var kk_0 : f32 = dot(k_4, k_4);
    var _S43 : f32 = k_4.x;
    var sxx_0 : f32 = a0_0.x + (_S43 * _S43 - kk_0) * ic_0;
    var _S44 : f32 = k_4.y;
    var syy_0 : f32 = a0_0.y + (_S44 * _S44 - kk_0) * ic_0;
    var _S45 : f32 = k_4.z;
    var szz_0 : f32 = a0_0.z + (_S45 * _S45 - kk_0) * ic_0;
    var sxy_0 : f32 = a0_0.w + _S43 * _S44 * ic_0;
    var sxz_0 : f32 = _S40 + _S43 * _S45 * ic_0;
    var syz_0 : f32 = _S41 + _S44 * _S45 * ic_0;
    var cxx_0 : f32 = syy_0 * szz_0 - syz_0 * syz_0;
    var cxy_0 : f32 = sxz_0 * syz_0 - sxy_0 * szz_0;
    var cxz_0 : f32 = sxy_0 * syz_0 - sxz_0 * syy_0;
    var det_0 : f32 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    var _S46 : bool;
    if(!(det_0 > 0.0f))
    {
        _S46 = true;
    }
    else
    {
        _S46 = !isfinite_0(det_0);
    }
    if(_S46)
    {
        return false;
    }
    var id_0 : f32 = 1.0f / det_0;
    var cyz_0 : f32 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    var _S47 : u32 = ns_0(node_1, u32(21));
    scratch_0[_S47] = vec4<f32>(cxx_0 * id_0, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_0, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_0, cxy_0 * id_0);
    scratch_0[_S47 + u32(1)] = vec4<f32>(cxz_0 * id_0, cyz_0 * id_0, ic_0, 0.0f);
    scratch_0[_S47 + u32(2)] = vec4<f32>(k_4 * vec3<f32>(ic_0), 0.0f);
    return true;
}

fn blast_frame_0( node_count_1 : u32,  k_5 : u32) -> u32
{
    return node_count_1 + u32(9) * k_5;
}

fn frame_word_0( k_6 : u32,  w_0 : u32) -> u32
{
    return islands_0[blast_frame_0(params_0.node_count_0, k_6) + w_0 / u32(4)][w_0 % u32(4)];
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_1 : vec4<u32>,  thread_1 : u32) -> vec4<u32>
{
    if(thread_1 == u32(0))
    {
        g_group_uniform4_0 = value_1;
    }
    var _S48 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S48;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn bs_0( j_4 : u32) -> u32
{
    return params_0.node_count_0 * params_0.node_vectors_0 + u32(2) * j_4;
}

fn va_0( n_4 : u32,  v_4 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_4, v_4)].xyz;
}

fn vl_0( n_5 : u32,  v_5 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_5, v_5) + u32(1)].xyz;
}

fn rot_a_0( j_5 : u32,  v_6 : vec3<f32>) -> vec3<f32>
{
    var _S49 : vec3<f32>;
    if((params_0.rotation_0) != u32(0))
    {
        _S49 = stiff_0(j_5, u32(0), v_6);
    }
    else
    {
        _S49 = v_6;
    }
    return _S49;
}

fn rot_l_0( j_6 : u32,  v_7 : vec3<f32>) -> vec3<f32>
{
    var _S50 : vec3<f32>;
    if((params_0.rotation_0) == u32(2))
    {
        _S50 = stiff_0(j_6, u32(2), v_7);
    }
    else
    {
        _S50 = v_7;
    }
    return _S50;
}

fn bond_term_0( b_3 : ptr<function, BlastBond_std430_0>,  j_7 : u32,  v_8 : u32,  ta_0 : ptr<function, vec3<f32>>,  tl_0 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_3).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_3).n1_0];
    var a0_1 : vec3<f32> = va_0((*b_3).n0_0, v_8) * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = vl_0((*b_3).n0_0, v_8) * vec3<f32>(m0_0.w_lin_0);
    var a1_1 : vec3<f32> = va_0((*b_3).n1_0, v_8) * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = vl_0((*b_3).n1_0, v_8) * vec3<f32>(m1_0.w_lin_0);
    var _S51 : vec3<f32> = vec3<f32>((*b_3).s_0);
    (*ta_0) = rot_a_0(j_7, a0_1 - a1_1) * _S51;
    (*tl_0) = rot_l_0(j_7, l0_0 - l1_0 + (cross((*b_3).o0_0.xyz, a0_1) - cross((*b_3).o1_0.xyz, a1_1))) * _S51;
    return;
}

fn bond_term_1( b_4 : ptr<function, BlastBond_std430_0>,  j_8 : u32,  v_9 : u32,  ta_1 : ptr<function, vec3<f32>>,  tl_1 : ptr<function, vec3<f32>>)
{
    var m0_1 : BlastNode_std430_0 = nodes_0[(*b_4).n0_0];
    var m1_1 : BlastNode_std430_0 = nodes_0[(*b_4).n1_0];
    var a0_2 : vec3<f32> = va_0((*b_4).n0_0, v_9) * vec3<f32>(m0_1.w_ang_0);
    var l0_1 : vec3<f32> = vl_0((*b_4).n0_0, v_9) * vec3<f32>(m0_1.w_lin_0);
    var a1_2 : vec3<f32> = va_0((*b_4).n1_0, v_9) * vec3<f32>(m1_1.w_ang_0);
    var l1_1 : vec3<f32> = vl_0((*b_4).n1_0, v_9) * vec3<f32>(m1_1.w_lin_0);
    var _S52 : vec3<f32> = vec3<f32>((*b_4).s_0);
    (*ta_1) = rot_a_0(j_8, a0_2 - a1_2) * _S52;
    (*tl_1) = rot_l_0(j_8, l0_1 - l1_1 + (cross((*b_4).o0_0.xyz, a0_2) - cross((*b_4).o1_0.xyz, a1_2))) * _S52;
    return;
}

fn gather_0( node_2 : u32,  from_t_0 : bool,  with_u_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_6 : BlastNode_std430_0 = nodes_0[node_2];
    var _S53 : vec3<f32> = vec3<f32>(0.0f);
    var r_3 : u32 = n_6.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S53;
    var acc_l_0 : vec3<f32> = _S53;
    loop
    {
        if(r_3 < (n_6.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_2 : u32 = refs_0[r_3];
        var j_9 : u32 = (ref_2 & (u32(2147483647)));
        var _S54 : BlastBond_std430_0 = bonds_0[j_9];
        if(!((_S54.health_0) > 0.0f))
        {
            r_3 = r_3 + u32(1);
            continue;
        }
        var ya_0 : vec3<f32>;
        var yl_0 : vec3<f32>;
        if(from_t_0)
        {
            var _S55 : u32 = bs_0(j_9);
            var _S56 : vec3<f32> = scratch_0[_S55 + u32(1)].xyz;
            ya_0 = scratch_0[_S55].xyz;
            yl_0 = _S56;
        }
        else
        {
            var _S57 : u32 = j_9 * u32(2);
            var ya_1 : vec3<f32> = x_0[_S57].xyz;
            var yl_1 : vec3<f32> = x_0[_S57 + u32(1)].xyz;
            if(with_u_0)
            {
                var ta_2 : vec3<f32>;
                var tl_2 : vec3<f32>;
                bond_term_1(&(_S54), j_9, u32(6), &(ta_2), &(tl_2));
                var yl_2 : vec3<f32> = yl_1 + tl_2;
                ya_0 = ya_1 + ta_2;
                yl_0 = yl_2;
            }
            else
            {
                ya_0 = ya_1;
                yl_0 = yl_1;
            }
        }
        var _S58 : vec3<f32> = vec3<f32>(_S54.s_0);
        var ya_2 : vec3<f32> = rot_a_0(j_9, ya_0) * _S58;
        var yl_3 : vec3<f32> = rot_l_0(j_9, yl_0) * _S58;
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_2 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross(_S54.o1_0.xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross(_S54.o0_0.xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_3 = r_3 + u32(1);
    }
    (*ang_0) = acc_a_0 * vec3<f32>(n_6.w_ang_0);
    (*lin_0) = acc_l_0 * vec3<f32>(n_6.w_lin_0);
    return;
}

fn rebuild_0( tid_1 : u32,  n0_1 : u32,  n1_1 : u32,  with_u_1 : bool) -> f32
{
    var p_0 : u32 = n0_1 + tid_1;
    var s_7 : f32 = 0.0f;
    loop
    {
        if(p_0 < n1_1)
        {
        }
        else
        {
            break;
        }
        var i_0 : u32 = lists_0[p_0];
        var a_2 : vec3<f32>;
        var l_2 : vec3<f32>;
        gather_0(i_0, false, with_u_1, &(a_2), &(l_2));
        var ra_0 : vec3<f32> = va_0(i_0, u32(14)) - a_2;
        var rl_0 : vec3<f32> = vl_0(i_0, u32(14)) - l_2;
        set_v_0(i_0, u32(0), ra_0, rl_0);
        var s_8 : f32 = s_7 + norm2_0(ra_0, rl_0);
        p_0 = p_0 + u32(1024);
        s_7 = s_8;
    }
    device_group_barrier_0();
    var _S59 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(s_7, 0.0f));
    return _S59.x;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn frame_float_0( k_7 : u32,  w_1 : u32) -> f32
{
    return (bitcast<f32>((frame_word_0(k_7, w_1))));
}

fn frame_axis_0( k_8 : u32,  rotations_0 : u32,  kk_1 : u32) -> vec3<f32>
{
    if(rotations_0 == u32(1))
    {
        return vec3<f32>(frame_float_0(k_8, u32(4)), frame_float_0(k_8, u32(5)), frame_float_0(k_8, u32(6)));
    }
    var _S60 : f32;
    if(kk_1 == u32(0))
    {
        _S60 = 1.0f;
    }
    else
    {
        _S60 = 0.0f;
    }
    var _S61 : f32;
    if(kk_1 == u32(1))
    {
        _S61 = 1.0f;
    }
    else
    {
        _S61 = 0.0f;
    }
    var _S62 : f32;
    if(kk_1 == u32(2))
    {
        _S62 = 1.0f;
    }
    else
    {
        _S62 = 0.0f;
    }
    return vec3<f32>(_S60, _S61, _S62);
}

fn frame_terms_0( k_9 : u32,  rotations_1 : u32,  i_1 : u32,  v_10 : u32,  s_9 : ptr<function, array<f32, i32(6)>>)
{
    var n_7 : BlastNode_std430_0 = nodes_0[i_1];
    var lin_1 : vec3<f32> = vl_0(i_1, v_10) / vec3<f32>(n_7.w_lin_0);
    var _S63 : vec3<f32> = va_0(i_1, v_10) / vec3<f32>(n_7.w_ang_0) - cross(n_7.position_0.xyz, lin_1);
    var kk_2 : u32 = u32(0);
    loop
    {
        if(kk_2 < rotations_1)
        {
        }
        else
        {
            break;
        }
        (*s_9)[kk_2] = (*s_9)[kk_2] + dot(frame_axis_0(k_9, rotations_1, kk_2), _S63);
        kk_2 = kk_2 + u32(1);
    }
    var j_10 : u32 = u32(0);
    loop
    {
        if(j_10 < u32(3))
        {
        }
        else
        {
            break;
        }
        (*s_9)[rotations_1 + j_10] = (*s_9)[rotations_1 + j_10] + lin_1[j_10];
        j_10 = j_10 + u32(1);
    }
    return;
}

fn frame_solve_0( k_10 : u32,  rotations_2 : u32,  sum_0 : array<f32, i32(6)>,  omega_0 : ptr<function, vec3<f32>>,  t_0 : ptr<function, vec3<f32>>)
{
    var j_11 : u32;
    var d_0 : u32 = rotations_2 + u32(3);
    var c_2 : array<f32, i32(6)>;
    var z_1 : u32 = u32(0);
    loop
    {
        if(z_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_2[z_1] = 0.0f;
        z_1 = z_1 + u32(1);
    }
    var row_0 : u32 = u32(0);
    loop
    {
        if(row_0 < d_0)
        {
        }
        else
        {
            break;
        }
        var _S64 : f32 = sum_0[row_0] * frame_float_0(k_10, u32(8) + row_0);
        j_11 = u32(0);
        var value_2 : f32 = _S64;
        loop
        {
            if(j_11 < row_0)
            {
            }
            else
            {
                break;
            }
            var value_3 : f32 = value_2 - frame_float_0(k_10, u32(14) + row_0 * (row_0 + u32(1)) / u32(2) + j_11) * c_2[j_11];
            j_11 = j_11 + u32(1);
            value_2 = value_3;
        }
        var _S65 : u32 = row_0 + u32(1);
        c_2[row_0] = value_2 / frame_float_0(k_10, u32(14) + row_0 * _S65 / u32(2) + row_0);
        row_0 = _S65;
    }
    var row_1 : i32 = i32(d_0) - i32(1);
    loop
    {
        if(row_1 >= i32(0))
        {
        }
        else
        {
            break;
        }
        var r_4 : u32 = u32(row_1);
        var _S66 : u32 = r_4 + u32(1);
        j_11 = _S66;
        loop
        {
            if(j_11 < d_0)
            {
            }
            else
            {
                break;
            }
            var _S67 : u32 = j_11 + u32(1);
            c_2[r_4] = c_2[r_4] - frame_float_0(k_10, u32(14) + j_11 * _S67 / u32(2) + r_4) * c_2[j_11];
            j_11 = _S67;
        }
        c_2[r_4] = c_2[r_4] / frame_float_0(k_10, u32(14) + r_4 * _S66 / u32(2) + r_4);
        row_1 = row_1 - i32(1);
    }
    z_1 = u32(0);
    loop
    {
        if(z_1 < d_0)
        {
        }
        else
        {
            break;
        }
        c_2[z_1] = c_2[z_1] * frame_float_0(k_10, u32(8) + z_1);
        z_1 = z_1 + u32(1);
    }
    (*omega_0) = vec3<f32>(0.0f);
    j_11 = u32(0);
    loop
    {
        if(j_11 < rotations_2)
        {
        }
        else
        {
            break;
        }
        (*omega_0) = (*omega_0) + frame_axis_0(k_10, rotations_2, j_11) * vec3<f32>(c_2[j_11]);
        j_11 = j_11 + u32(1);
    }
    (*t_0) = vec3<f32>(c_2[rotations_2], c_2[rotations_2 + u32(1)], c_2[rotations_2 + u32(2)]);
    return;
}

var<workgroup> g_omega_0 : vec3<f32>;

var<workgroup> g_t_0 : vec3<f32>;

fn frame_apply_0( i_2 : u32,  v_11 : u32,  omega_1 : vec3<f32>,  t_1 : vec3<f32>)
{
    var n_8 : BlastNode_std430_0 = nodes_0[i_2];
    set_v_0(i_2, v_11, va_0(i_2, v_11) - omega_1 / vec3<f32>(n_8.w_ang_0), vl_0(i_2, v_11) - (t_1 + cross(n_8.position_0.xyz, omega_1)) / vec3<f32>(n_8.w_lin_0));
    return;
}

fn project_0( tid_2 : u32,  k_11 : u32,  rotations_3 : u32,  n0_2 : u32,  n1_2 : u32,  v_12 : u32,  free_0 : bool)
{
    if(!free_0)
    {
        return;
    }
    var s_10 : array<f32, i32(6)>;
    var z_2 : u32 = u32(0);
    loop
    {
        if(z_2 < u32(6))
        {
        }
        else
        {
            break;
        }
        s_10[z_2] = 0.0f;
        z_2 = z_2 + u32(1);
    }
    var _S68 : u32 = n0_2 + tid_2;
    var p_1 : u32 = _S68;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_terms_0(k_11, rotations_3, lists_0[p_1], v_12, &(s_10));
        p_1 = p_1 + u32(1024);
    }
    var a_3 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_10[i32(0)], s_10[i32(1)]));
    var b_5 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_10[i32(2)], s_10[i32(3)]));
    var c_3 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_10[i32(4)], s_10[i32(5)]));
    if(tid_2 == u32(0))
    {
        var sum_1 : array<f32, i32(6)> = array<f32, i32(6)>( a_3.x, a_3.y, b_5.x, b_5.y, c_3.x, c_3.y );
        var omega_2 : vec3<f32>;
        var t_2 : vec3<f32>;
        frame_solve_0(k_11, rotations_3, sum_1, &(omega_2), &(t_2));
        g_omega_0 = omega_2;
        g_t_0 = t_2;
    }
    workgroupBarrier();
    var _S69 : vec3<f32> = g_omega_0;
    var _S70 : vec3<f32> = g_t_0;
    p_1 = _S68;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_apply_0(lists_0[p_1], v_12, _S69, _S70);
        p_1 = p_1 + u32(1024);
    }
    device_group_barrier_0();
    return;
}

fn bond_pass_0( tid_3 : u32,  b0_0 : u32,  b1_0 : u32,  v_13 : u32) -> f32
{
    var q_1 : u32 = b0_0 + tid_3;
    var z_3 : f32 = 0.0f;
    loop
    {
        if(q_1 < b1_0)
        {
        }
        else
        {
            break;
        }
        var j_12 : u32 = lists_0[q_1];
        var _S71 : BlastBond_std430_0 = bonds_0[j_12];
        var _S72 : vec3<f32> = vec3<f32>(0.0f);
        var ta_3 : vec3<f32> = _S72;
        var tl_3 : vec3<f32> = _S72;
        if((_S71.health_0) > 0.0f)
        {
            bond_term_0(&(_S71), j_12, v_13, &(ta_3), &(tl_3));
        }
        var z_4 : f32 = z_3 + norm2_0(ta_3, tl_3);
        var _S73 : u32 = bs_0(j_12);
        scratch_0[_S73] = vec4<f32>(ta_3, 0.0f);
        scratch_0[_S73 + u32(1)] = vec4<f32>(tl_3, 0.0f);
        q_1 = q_1 + u32(1024);
        z_3 = z_4;
    }
    var _S74 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(z_3, 0.0f));
    return _S74.x;
}

fn residual_norm_0( tid_4 : u32,  n0_3 : u32,  n1_3 : u32) -> f32
{
    var p_2 : u32 = n0_3 + tid_4;
    var s_11 : f32 = 0.0f;
    loop
    {
        if(p_2 < n1_3)
        {
        }
        else
        {
            break;
        }
        var i_3 : u32 = lists_0[p_2];
        var s_12 : f32 = s_11 + norm2_0(va_0(i_3, u32(0)), vl_0(i_3, u32(0)));
        p_2 = p_2 + u32(1024);
        s_11 = s_12;
    }
    var _S75 : vec2<f32> = blast_group_sum_0(tid_4, vec2<f32>(s_11, 0.0f));
    return _S75.x;
}

fn full_apply_0( n_9 : u32,  at_0 : u32,  a_4 : vec3<f32>,  l_3 : vec3<f32>,  oa_0 : ptr<function, vec3<f32>>,  ol_0 : ptr<function, vec3<f32>>)
{
    var m_2 : array<f32, i32(24)>;
    var q_2 : u32 = u32(0);
    loop
    {
        if(q_2 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S76 : u32 = u32(4) * q_2;
        m_2[_S76] = scratch_0[ns_0(n_9, at_0 + q_2)].x;
        m_2[_S76 + u32(1)] = scratch_0[ns_0(n_9, at_0 + q_2)].y;
        m_2[_S76 + u32(2)] = scratch_0[ns_0(n_9, at_0 + q_2)].z;
        m_2[_S76 + u32(3)] = scratch_0[ns_0(n_9, at_0 + q_2)].w;
        q_2 = q_2 + u32(1);
    }
    var _S77 : array<f32, i32(6)> = array<f32, i32(6)>( a_4.x, a_4.y, a_4.z, l_3.x, l_3.y, l_3.z );
    var o_3 : array<f32, i32(6)>;
    var r_5 : u32 = u32(0);
    loop
    {
        if(r_5 < u32(6))
        {
        }
        else
        {
            break;
        }
        var c_4 : u32 = u32(0);
        var s_13 : f32 = 0.0f;
        loop
        {
            if(c_4 < u32(6))
            {
            }
            else
            {
                break;
            }
            if(r_5 >= c_4)
            {
                q_2 = r_5 * (r_5 + u32(1)) / u32(2) + c_4;
            }
            else
            {
                q_2 = c_4 * (c_4 + u32(1)) / u32(2) + r_5;
            }
            var s_14 : f32 = s_13 + m_2[q_2] * _S77[c_4];
            c_4 = c_4 + u32(1);
            s_13 = s_14;
        }
        o_3[r_5] = s_13;
        r_5 = r_5 + u32(1);
    }
    (*oa_0) = vec3<f32>(o_3[i32(0)], o_3[i32(1)], o_3[i32(2)]);
    (*ol_0) = vec3<f32>(o_3[i32(3)], o_3[i32(4)], o_3[i32(5)]);
    return;
}

fn sym_mul_0( s0_0 : vec4<f32>,  s1_0 : vec4<f32>,  v_14 : vec3<f32>) -> vec3<f32>
{
    var _S78 : f32 = v_14.x;
    var _S79 : f32 = s0_0.w;
    var _S80 : f32 = v_14.y;
    var _S81 : f32 = s1_0.x;
    var _S82 : f32 = v_14.z;
    var _S83 : f32 = s1_0.y;
    return vec3<f32>(s0_0.x * _S78 + _S79 * _S80 + _S81 * _S82, _S79 * _S78 + s0_0.y * _S80 + _S83 * _S82, _S81 * _S78 + _S83 * _S80 + s0_0.z * _S82);
}

fn inverse_apply_0( n_10 : u32,  a_5 : vec3<f32>,  l_4 : vec3<f32>,  oa_1 : ptr<function, vec3<f32>>,  ol_1 : ptr<function, vec3<f32>>)
{
    if((params_0.rotation_0) == u32(2))
    {
        full_apply_0(n_10, u32(24), a_5, l_4, &((*oa_1)), &((*ol_1)));
        return;
    }
    var _S84 : u32 = ns_0(n_10, u32(21));
    var s1_1 : vec4<f32> = scratch_0[_S84 + u32(1)];
    var kc_0 : vec3<f32> = scratch_0[_S84 + u32(2)].xyz;
    var _S85 : vec3<f32> = sym_mul_0(scratch_0[_S84], scratch_0[_S84 + u32(1)], a_5 + cross(kc_0, l_4));
    (*oa_1) = _S85;
    (*ol_1) = l_4 * vec3<f32>(s1_1.z) - cross(kc_0, _S85);
    return;
}

fn node_pass_0( tid_5 : u32,  n0_4 : u32,  n1_4 : u32,  out_v_0 : u32) -> f32
{
    var p_3 : u32 = n0_4 + tid_5;
    var s_15 : f32 = 0.0f;
    loop
    {
        if(p_3 < n1_4)
        {
        }
        else
        {
            break;
        }
        var i_4 : u32 = lists_0[p_3];
        var a_6 : vec3<f32>;
        var l_5 : vec3<f32>;
        gather_0(i_4, true, false, &(a_6), &(l_5));
        set_v_0(i_4, out_v_0, a_6, l_5);
        var s_16 : f32 = s_15 + norm2_0(a_6, l_5);
        p_3 = p_3 + u32(1024);
        s_15 = s_16;
    }
    return s_15;
}

fn block_apply_0( n_11 : u32,  a_7 : vec3<f32>,  l_6 : vec3<f32>,  oa_2 : ptr<function, vec3<f32>>,  ol_2 : ptr<function, vec3<f32>>)
{
    if((params_0.rotation_0) == u32(2))
    {
        full_apply_0(n_11, u32(18), a_7, l_6, &((*oa_2)), &((*ol_2)));
        return;
    }
    var _S86 : u32 = ns_0(n_11, u32(18));
    var b1_1 : vec4<f32> = scratch_0[_S86 + u32(1)];
    var k_12 : vec3<f32> = scratch_0[_S86 + u32(2)].xyz;
    (*oa_2) = sym_mul_0(scratch_0[_S86], scratch_0[_S86 + u32(1)], a_7) - cross(k_12, l_6);
    (*ol_2) = cross(k_12, a_7) + vec3<f32>(b1_1.z) * l_6;
    return;
}

var<workgroup> g_max_0 : array<f32, i32(1024)>;

var<workgroup> g_max_out_0 : f32;

fn group_max_0( tid_6 : u32,  v_15 : f32) -> f32
{
    g_max_0[tid_6] = v_15;
    workgroupBarrier();
    var k_13 : u32 = u32(512);
    loop
    {
        if(k_13 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_6 < k_13)
        {
            g_max_0[tid_6] = max(g_max_0[tid_6], g_max_0[tid_6 + k_13]);
        }
        workgroupBarrier();
        k_13 = (k_13 >> (u32(1)));
    }
    if(tid_6 == u32(0))
    {
        g_max_out_0 = g_max_0[i32(0)];
    }
    var _S87 : f32 = (workgroupUniformLoad(&((g_max_out_0))));
    workgroupBarrier();
    return _S87;
}

@compute
@workgroup_size(1024, 1, 1)
fn blast_native(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var force_start_0 : f32;
    var best_0 : f32;
    var failed_0 : bool;
    var q_3 : u32;
    var ok_0 : bool;
    var tid_7 : u32 = thread_2.x;
    var k_14 : u32 = group_0.x;
    if(k_14 >= (params_0.island_count_0))
    {
        return;
    }
    var range_0 : vec4<u32> = islands_0[k_14];
    var n0_5 : u32 = range_0.x;
    var n1_5 : u32 = range_0.y;
    var b0_1 : u32 = range_0.z;
    var _S88 : u32 = range_0.w;
    var p_4 : u32;
    if((n1_5 - n0_5) > (params_0.wide_min_0))
    {
        p_4 = u32(1);
    }
    else
    {
        p_4 = u32(0);
    }
    var _S89 : u32 = group_uniform_0(p_4, tid_7);
    if(_S89 != u32(0))
    {
        return;
    }
    var last_0 : vec4<u32> = report_0[k_14];
    var last_flags_0 : u32 = group_uniform_0(report_0[k_14].y, tid_7);
    var warm_1 : bool = (params_0.warm_0) != u32(0);
    var converged_0 : bool;
    if(warm_1)
    {
        converged_0 = ((last_flags_0 & (u32(4)))) != u32(0);
    }
    else
    {
        converged_0 = false;
    }
    var singular_0 : f32;
    if(converged_0)
    {
        var _S90 : u32 = n0_5 + tid_7;
        singular_0 = 0.0f;
        p_4 = _S90;
        loop
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            var i_5 : u32 = lists_0[p_4];
            var _S91 : u32 = i_5 * u32(2);
            var _S92 : u32 = ns_0(i_5, u32(16));
            if((any((accel_0[_S91] != scratch_0[_S92]))))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = (any((accel_0[_S91 + u32(1)] != scratch_0[_S92 + u32(1)])));
            }
            if(converged_0)
            {
                singular_0 = 1.0f;
            }
            p_4 = p_4 + u32(1024);
        }
        var _S93 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(singular_0, 0.0f));
        if((_S93.x) == 0.0f)
        {
            if(tid_7 == u32(0))
            {
                report_0[k_14] = vec4<u32>(u32(0), u32(7), last_0.z, last_0.w);
            }
            return;
        }
    }
    var _S94 : u32 = n0_5 + tid_7;
    singular_0 = 0.0f;
    p_4 = _S94;
    var target_0 : f32 = 0.0f;
    loop
    {
        if(p_4 < n1_5)
        {
        }
        else
        {
            break;
        }
        var i_6 : u32 = lists_0[p_4];
        var n_12 : BlastNode_std430_0 = nodes_0[i_6];
        var _S95 : u32 = i_6 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S95];
        var a_lin_0 : vec4<f32> = accel_0[_S95 + u32(1)];
        var _S96 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz);
        if((n_12.w_ang_0) > 0.0f)
        {
            force_start_0 = 1.0f / n_12.w_ang_0;
        }
        else
        {
            force_start_0 = 1.0f;
        }
        var ba_0 : vec3<f32> = _S96 * vec3<f32>(force_start_0);
        var _S97 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz);
        if((n_12.w_lin_0) > 0.0f)
        {
            best_0 = params_0.recip_length_0 / n_12.w_lin_0;
        }
        else
        {
            best_0 = params_0.recip_length_0;
        }
        var bl_0 : vec3<f32> = _S97 * vec3<f32>(best_0);
        var target_1 : f32 = target_0 + norm2_0(ba_0, bl_0);
        set_v_0(i_6, u32(14), ba_0, bl_0);
        set_v_0(i_6, u32(0), ba_0, bl_0);
        var _S98 : u32 = ns_0(i_6, u32(16));
        scratch_0[_S98] = a_ang_0;
        scratch_0[_S98 + u32(1)] = a_lin_0;
        var _S99 : vec3<f32> = vec3<f32>(0.0f);
        set_v_0(i_6, u32(6), _S99, _S99);
        set_v_0(i_6, u32(2), _S99, _S99);
        var _S100 : bool = build_block_0(i_6);
        if(!_S100)
        {
            singular_0 = 1.0f;
        }
        p_4 = p_4 + u32(1024);
        target_0 = target_1;
    }
    var q_4 : u32;
    if(!warm_1)
    {
        q_4 = b0_1 + tid_7;
        loop
        {
            if(q_4 < _S88)
            {
            }
            else
            {
                break;
            }
            var _S101 : u32 = lists_0[q_4] * u32(2);
            var _S102 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S101] = _S102;
            x_0[_S101 + u32(1)] = _S102;
            q_4 = q_4 + u32(1024);
        }
    }
    var tsum_0 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(target_0, singular_0));
    var delta2_0 : f32 = tsum_0.x * params_0.tolerance_0 * params_0.tolerance_0;
    var _S103 : vec4<u32> = group_uniform4_0(vec4<u32>(frame_word_0(k_14, u32(0)), frame_word_0(k_14, u32(1)), frame_word_0(k_14, u32(2)), u32(0)), tid_7);
    var frame_0 : vec3<u32> = _S103.xyz;
    if((tsum_0.y) > 0.0f)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (frame_0.z) != u32(0);
    }
    var _S104 : bool = (frame_0.x) == u32(0);
    var _S105 : u32 = frame_0.y;
    device_group_barrier_0();
    if(!(delta2_0 > 0.0f))
    {
        q_4 = b0_1 + tid_7;
        loop
        {
            if(q_4 < _S88)
            {
            }
            else
            {
                break;
            }
            var _S106 : u32 = lists_0[q_4] * u32(2);
            var _S107 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S106] = _S107;
            x_0[_S106 + u32(1)] = _S107;
            q_4 = q_4 + u32(1024);
        }
        if(tid_7 == u32(0))
        {
            if(warm_1)
            {
                p_4 = u32(4);
            }
            else
            {
                p_4 = u32(0);
            }
            report_0[k_14] = vec4<u32>(u32(0), (u32(1) | (p_4)), (bitcast<u32>((0.0f))), (bitcast<u32>((delta2_0))));
        }
        return;
    }
    if(warm_1)
    {
        var _S108 : f32 = rebuild_0(tid_7, n0_5, n1_5, false);
    }
    var _S109 : f32 = float_bits_0(u32(2139095040));
    if((params_0.force_tolerance_0) > 0.0f)
    {
        q_4 = b0_1 + tid_7;
        force_start_0 = 0.0f;
        loop
        {
            if(q_4 < _S88)
            {
            }
            else
            {
                break;
            }
            var _S110 : u32 = lists_0[q_4] * u32(2);
            var f_0 : f32 = force_start_0 + norm2_0(x_0[_S110].xyz, x_0[_S110 + u32(1)].xyz);
            q_4 = q_4 + u32(1024);
            force_start_0 = f_0;
        }
        var _S111 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(force_start_0, 0.0f));
        force_start_0 = sqrt(_S111.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    best_0 = _S109;
    q_4 = u32(0);
    var step_0 : f32 = _S109;
    var previous_0 : f32 = 0.0f;
    var num_0 : f32 = 0.0f;
    var iteration_0 : u32 = u32(0);
    var travel_0 : f32 = 0.0f;
    loop
    {
        if(!converged_0)
        {
            failed_0 = iteration_0 < (params_0.max_iterations_0);
        }
        else
        {
            failed_0 = false;
        }
        if(failed_0)
        {
        }
        else
        {
            var _S112 : bool = converged_0;
            converged_0 = false;
            failed_0 = _S112;
            break;
        }
        project_0(tid_7, k_14, _S105, n0_5, n1_5, u32(0), _S104);
        var num_1 : f32 = bond_pass_0(tid_7, b0_1, _S88, u32(0));
        var _S113 : bool = num_1 <= delta2_0;
        if(_S113)
        {
            var _S114 : f32 = residual_norm_0(tid_7, n0_5, n1_5);
            ok_0 = !(_S114 <= delta2_0);
        }
        else
        {
            ok_0 = false;
        }
        var _S115 : bool = iteration_0 > u32(0);
        var _S116 : bool;
        if(_S115)
        {
            _S116 = true;
        }
        else
        {
            _S116 = warm_1;
        }
        var _S117 : bool;
        if(_S116)
        {
            _S117 = _S113;
        }
        else
        {
            _S117 = false;
        }
        var _S118 : bool;
        if(_S117)
        {
            _S118 = !ok_0;
        }
        else
        {
            _S118 = false;
        }
        var hold_0 : bool;
        var num_2 : f32;
        var previous_1 : f32;
        if(_S118)
        {
            var _S119 : f32 = rebuild_0(tid_7, n0_5, n1_5, true);
            project_0(tid_7, k_14, _S105, n0_5, n1_5, u32(0), _S104);
            var num_3 : f32 = bond_pass_0(tid_7, b0_1, _S88, u32(0));
            if(num_3 <= delta2_0)
            {
                var _S120 : f32 = residual_norm_0(tid_7, n0_5, n1_5);
                hold_0 = !(_S120 <= delta2_0);
            }
            else
            {
                hold_0 = false;
            }
            num_2 = num_3;
            previous_1 = 0.0f;
        }
        else
        {
            num_2 = num_1;
            hold_0 = ok_0;
            previous_1 = previous_0;
        }
        if(num_2 < (best_0 * 0.99000000953674316f))
        {
            best_0 = num_2;
            q_4 = iteration_0;
        }
        else
        {
            if((iteration_0 - q_4) >= u32(512))
            {
                var _S121 : bool = converged_0;
                converged_0 = false;
                failed_0 = _S121;
                num_0 = num_2;
                break;
            }
        }
        var _S122 : bool;
        if(num_2 <= delta2_0)
        {
            _S122 = !hold_0;
        }
        else
        {
            _S122 = false;
        }
        if(_S122)
        {
            var _S123 : bool = converged_0;
            converged_0 = true;
            failed_0 = _S123;
            num_0 = num_2;
            break;
        }
        var _S124 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            _S124 = _S115;
        }
        else
        {
            _S124 = false;
        }
        var _S125 : bool;
        if(_S124)
        {
            _S125 = step_0 <= (params_0.force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S125 = false;
        }
        var mloc_0 : f32;
        if(_S125)
        {
            q_3 = b0_1 + tid_7;
            mloc_0 = 0.0f;
            loop
            {
                if(q_3 < _S88)
                {
                }
                else
                {
                    break;
                }
                var j_13 : u32 = lists_0[q_3];
                var _S126 : BlastBond_std430_0 = bonds_0[j_13];
                var ta_4 : vec3<f32>;
                var tl_4 : vec3<f32>;
                bond_term_1(&(_S126), j_13, u32(6), &(ta_4), &(tl_4));
                var _S127 : u32 = j_13 * u32(2);
                var f_1 : f32 = mloc_0 + norm2_0(x_0[_S127].xyz + ta_4, x_0[_S127 + u32(1)].xyz + tl_4);
                q_3 = q_3 + u32(1024);
                mloc_0 = f_1;
            }
            var _S128 : f32 = params_0.force_tolerance_0;
            var _S129 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(mloc_0, 0.0f));
            if(step_0 <= (_S128 * sqrt(_S129.x)))
            {
                var _S130 : bool = converged_0;
                converged_0 = true;
                failed_0 = _S130;
                num_0 = num_2;
                break;
            }
        }
        device_group_barrier_0();
        var _S131 : bool;
        if(iteration_0 == u32(0))
        {
            _S131 = (params_0.first_polynomial_0) == u32(0);
        }
        else
        {
            _S131 = false;
        }
        if(_S131)
        {
            p_4 = _S94;
            loop
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                var i_7 : u32 = lists_0[p_4];
                set_v_0(i_7, u32(8), va_0(i_7, u32(0)), vl_0(i_7, u32(0)));
                p_4 = p_4 + u32(1024);
            }
        }
        else
        {
            p_4 = _S94;
            loop
            {
                if(p_4 < n1_5)
                {
                }
                else
                {
                    break;
                }
                var i_8 : u32 = lists_0[p_4];
                var a_8 : vec3<f32>;
                var l_7 : vec3<f32>;
                inverse_apply_0(i_8, va_0(i_8, u32(0)), vl_0(i_8, u32(0)), &(a_8), &(l_7));
                set_v_0(i_8, u32(10), a_8, l_7);
                p_4 = p_4 + u32(1024);
            }
            device_group_barrier_0();
            var _S132 : f32 = bond_pass_0(tid_7, b0_1, _S88, u32(10));
            device_group_barrier_0();
            var _S133 : f32 = node_pass_0(tid_7, n0_5, n1_5, u32(12));
            q_3 = _S94;
            loop
            {
                if(q_3 < n1_5)
                {
                }
                else
                {
                    break;
                }
                var i_9 : u32 = lists_0[q_3];
                var sa_0 : vec3<f32>;
                var sl_0 : vec3<f32>;
                block_apply_0(i_9, va_0(i_9, u32(10)), vl_0(i_9, u32(10)), &(sa_0), &(sl_0));
                var ma_0 : vec3<f32>;
                var ml_0 : vec3<f32>;
                inverse_apply_0(i_9, va_0(i_9, u32(12)) - sa_0, vl_0(i_9, u32(12)) - sl_0, &(ma_0), &(ml_0));
                var _S134 : vec3<f32> = vec3<f32>(1.68946564197540283f);
                var _S135 : vec3<f32> = vec3<f32>(1.5220409631729126f);
                set_v_0(i_9, u32(8), _S134 * va_0(i_9, u32(10)) - _S135 * ma_0, _S134 * vl_0(i_9, u32(10)) - _S135 * ml_0);
                q_3 = q_3 + u32(1024);
            }
        }
        device_group_barrier_0();
        project_0(tid_7, k_14, _S105, n0_5, n1_5, u32(8), _S104);
        mloc_0 = 0.0f;
        p_4 = _S94;
        loop
        {
            if(p_4 < n1_5)
            {
            }
            else
            {
                break;
            }
            var i_10 : u32 = lists_0[p_4];
            var a_9 : vec3<f32> = abs(va_0(i_10, u32(8)));
            var l_8 : vec3<f32> = abs(vl_0(i_10, u32(8)));
            var _S136 : f32 = max(mloc_0, max(max(max(a_9.x, a_9.y), max(a_9.z, l_8.x)), max(l_8.y, l_8.z)));
            var p_5 : u32 = p_4 + u32(1024);
            mloc_0 = _S136;
            p_4 = p_5;
        }
        var mag_0 : f32 = group_max_0(tid_7, mloc_0);
        var _S137 : bool;
        if(!(mag_0 > 0.0f))
        {
            _S137 = true;
        }
        else
        {
            _S137 = !isfinite_0(mag_0);
        }
        if(_S137)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            break;
        }
        var _S138 : f32 = 1.0f / mag_0;
        q_3 = _S94;
        var gl_0 : f32 = 0.0f;
        loop
        {
            if(q_3 < n1_5)
            {
            }
            else
            {
                break;
            }
            var i_11 : u32 = lists_0[q_3];
            var _S139 : vec3<f32> = vec3<f32>(_S138);
            var ga_0 : vec3<f32> = va_0(i_11, u32(8)) * _S139;
            var gn_0 : vec3<f32> = vl_0(i_11, u32(8)) * _S139;
            set_v_0(i_11, u32(8), ga_0, gn_0);
            var gl_1 : f32 = gl_0 + (dot(va_0(i_11, u32(0)), ga_0) + dot(vl_0(i_11, u32(0)), gn_0));
            q_3 = q_3 + u32(1024);
            gl_0 = gl_1;
        }
        var _S140 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(gl_0, 0.0f));
        var gamma_0 : f32 = _S140.x;
        var _S141 : bool = gamma_0 > 0.0f;
        var _S142 : bool;
        if(!_S141)
        {
            _S142 = true;
        }
        else
        {
            _S142 = !isfinite_0(gamma_0);
        }
        if(_S142)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            break;
        }
        var _S143 : bool;
        if(iteration_0 > u32(1))
        {
            _S143 = true;
        }
        else
        {
            if((params_0.first_polynomial_0) != u32(0))
            {
                _S143 = _S115;
            }
            else
            {
                _S143 = false;
            }
        }
        var _S144 : bool;
        if(_S143)
        {
            _S144 = previous_1 > 0.0f;
        }
        else
        {
            _S144 = false;
        }
        if(_S144)
        {
            singular_0 = gamma_0 / previous_1;
        }
        else
        {
            singular_0 = 0.0f;
        }
        var p_6 : u32 = _S94;
        loop
        {
            if(p_6 < n1_5)
            {
            }
            else
            {
                break;
            }
            var i_12 : u32 = lists_0[p_6];
            if(singular_0 == 0.0f)
            {
                set_v_0(i_12, u32(2), va_0(i_12, u32(8)), vl_0(i_12, u32(8)));
            }
            else
            {
                set_v_0(i_12, u32(2), va_0(i_12, u32(8)) + va_0(i_12, u32(2)) * vec3<f32>(singular_0), vl_0(i_12, u32(8)) + vl_0(i_12, u32(2)) * vec3<f32>(singular_0));
            }
            p_6 = p_6 + u32(1024);
        }
        device_group_barrier_0();
        var den_0 : f32 = bond_pass_0(tid_7, b0_1, _S88, u32(2));
        device_group_barrier_0();
        var _S145 : f32 = node_pass_0(tid_7, n0_5, n1_5, u32(4));
        var _S146 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            if(den_0 > 0.0f)
            {
                _S146 = _S141;
            }
            else
            {
                _S146 = false;
            }
            var step_1 : f32;
            if(_S146)
            {
                step_1 = gamma_0 / sqrt(den_0);
            }
            else
            {
                step_1 = _S109;
            }
            var travel_1 : f32;
            if(isfinite_0(step_1))
            {
                travel_1 = travel_0 + step_1;
            }
            else
            {
                travel_1 = travel_0;
            }
            var _S147 : bool;
            if(_S115)
            {
                _S147 = true;
            }
            else
            {
                _S147 = (params_0.first_polynomial_0) != u32(0);
            }
            var step_2 : f32;
            if(!_S147)
            {
                step_2 = _S109;
            }
            else
            {
                step_2 = step_1;
            }
            step_0 = step_2;
            travel_0 = travel_1;
        }
        var _S148 : u32 = iteration_0 + u32(1);
        if(!(den_0 > 0.0f))
        {
            _S146 = true;
        }
        else
        {
            _S146 = !isfinite_0(den_0);
        }
        if(_S146)
        {
            var _S149 : bool = converged_0;
            converged_0 = false;
            failed_0 = _S149;
            num_0 = num_2;
            iteration_0 = _S148;
            break;
        }
        var _S150 : f32 = gamma_0 / den_0;
        var p_7 : u32 = _S94;
        loop
        {
            if(p_7 < n1_5)
            {
            }
            else
            {
                break;
            }
            var i_13 : u32 = lists_0[p_7];
            var _S151 : vec3<f32> = vec3<f32>(_S150);
            set_v_0(i_13, u32(6), va_0(i_13, u32(6)) + va_0(i_13, u32(2)) * _S151, vl_0(i_13, u32(6)) + vl_0(i_13, u32(2)) * _S151);
            set_v_0(i_13, u32(0), va_0(i_13, u32(0)) - va_0(i_13, u32(4)) * _S151, vl_0(i_13, u32(0)) - vl_0(i_13, u32(4)) * _S151);
            p_7 = p_7 + u32(1024);
        }
        device_group_barrier_0();
        previous_0 = gamma_0;
        num_0 = num_2;
        iteration_0 = _S148;
    }
    device_group_barrier_0();
    q_3 = b0_1 + tid_7;
    loop
    {
        if(q_3 < _S88)
        {
        }
        else
        {
            break;
        }
        var j_14 : u32 = lists_0[q_3];
        var _S152 : BlastBond_std430_0 = bonds_0[j_14];
        if(!((_S152.health_0) > 0.0f))
        {
            var _S153 : u32 = j_14 * u32(2);
            var _S154 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S153] = _S154;
            x_0[_S153 + u32(1)] = _S154;
            q_3 = q_3 + u32(1024);
            continue;
        }
        var ta_5 : vec3<f32>;
        var tl_5 : vec3<f32>;
        bond_term_1(&(_S152), j_14, u32(6), &(ta_5), &(tl_5));
        var _S155 : u32 = j_14 * u32(2);
        x_0[_S155] = vec4<f32>(x_0[_S155].xyz + ta_5, 0.0f);
        var _S156 : u32 = _S155 + u32(1);
        x_0[_S156] = vec4<f32>(x_0[_S156].xyz + tl_5, 0.0f);
        q_3 = q_3 + u32(1024);
    }
    if(converged_0)
    {
        ok_0 = !failed_0;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        p_4 = u32(1);
    }
    else
    {
        p_4 = u32(0);
    }
    if(ok_0)
    {
        converged_0 = warm_1;
    }
    else
    {
        converged_0 = false;
    }
    if(converged_0)
    {
        converged_0 = iteration_0 == u32(0);
    }
    else
    {
        converged_0 = false;
    }
    if(converged_0)
    {
        q_4 = u32(4);
    }
    else
    {
        q_4 = u32(0);
    }
    var flags_0 : u32 = (p_4 | (q_4));
    if(tid_7 == u32(0))
    {
        report_0[k_14] = vec4<u32>(iteration_0, flags_0, (bitcast<u32>((num_0))), (bitcast<u32>((delta2_0))));
    }
    return;
}

