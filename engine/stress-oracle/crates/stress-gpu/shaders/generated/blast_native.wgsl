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
    @align(8) carry_0 : u32,
    @align(4) section_bending_0 : u32,
    @align(16) chunk_loads_0 : u32,
    @align(4) r_virial_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(1) @group(0) var<storage, read_write> islands_0 : array<vec4<u32>>;

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

fn blast_frame_0( node_count_1 : u32,  k_1 : u32) -> u32
{
    return node_count_1 + u32(10) * k_1;
}

fn frame_word_0( k_2 : u32,  w_0 : u32) -> u32
{
    return islands_0[blast_frame_0(params_0.node_count_0, k_2) + w_0 / u32(4)][w_0 % u32(4)];
}

fn carry_of_0( k_3 : u32) -> vec3<u32>
{
    return vec3<u32>(frame_word_0(k_3, u32(36)), frame_word_0(k_3, u32(37)), frame_word_0(k_3, u32(38)));
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_1 : vec4<u32>,  thread_1 : u32) -> vec4<u32>
{
    if(thread_1 == u32(0))
    {
        g_group_uniform4_0 = value_1;
    }
    var _S9 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S9;
}

fn carry_kept_0( c_0 : vec3<u32>,  count_0 : u32) -> bool
{
    var gamma_0 : f32 = (bitcast<f32>((c_0.z)));
    var _S10 : bool;
    if((params_0.carry_0) != u32(0))
    {
        _S10 = (c_0.x) == count_0;
    }
    else
    {
        _S10 = false;
    }
    if(_S10)
    {
        _S10 = (c_0.y) < (u32(6) * count_0);
    }
    else
    {
        _S10 = false;
    }
    if(_S10)
    {
        _S10 = gamma_0 > 0.0f;
    }
    else
    {
        _S10 = false;
    }
    if(_S10)
    {
        _S10 = isfinite_0(gamma_0);
    }
    else
    {
        _S10 = false;
    }
    return _S10;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn set_carry_0( k_4 : u32,  count_1 : u32,  iterations_0 : u32,  gamma_1 : f32)
{
    var at_0 : u32 = blast_frame_0(params_0.node_count_0, k_4) + u32(9);
    var v_2 : vec4<u32> = islands_0[at_0];
    v_2[i32(0)] = count_1;
    v_2[i32(1)] = iterations_0;
    v_2[i32(2)] = (bitcast<u32>((gamma_1)));
    islands_0[at_0] = v_2;
    return;
}

fn norm2_0( a_0 : vec3<f32>,  b_0 : vec3<f32>) -> f32
{
    return dot(a_0, a_0) + dot(b_0, b_0);
}

fn va_0( n_1 : u32,  v_3 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_1, v_3)].xyz;
}

fn vl_0( n_2 : u32,  v_4 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_2, v_4) + u32(1)].xyz;
}

fn set_v_0( n_3 : u32,  v_5 : u32,  a_1 : vec3<f32>,  l_0 : vec3<f32>)
{
    var _S11 : u32 = ns_0(n_3, v_5);
    scratch_0[_S11] = vec4<f32>(a_1, 0.0f);
    scratch_0[_S11 + u32(1)] = vec4<f32>(l_0, 0.0f);
    return;
}

fn stiff_0( j_0 : u32,  which_0 : u32,  v_6 : vec3<f32>) -> vec3<f32>
{
    var _S12 : u32 = u32(6) * which_0;
    var m_0 : array<f32, i32(6)>;
    var k_5 : u32 = u32(0);
    loop
    {
        if(k_5 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S13 : u32 = _S12 + k_5;
        m_0[k_5] = stiffness_0[j_0 * u32(6) + _S13 / u32(4)][_S13 % u32(4)];
        k_5 = k_5 + u32(1);
    }
    var _S14 : f32 = v_6.x;
    var _S15 : f32 = v_6.y;
    var _S16 : f32 = v_6.z;
    return vec3<f32>(m_0[i32(0)] * _S14 + m_0[i32(3)] * _S15 + m_0[i32(4)] * _S16, m_0[i32(3)] * _S14 + m_0[i32(1)] * _S15 + m_0[i32(5)] * _S16, m_0[i32(4)] * _S14 + m_0[i32(5)] * _S15 + m_0[i32(2)] * _S16);
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
    var _S17 : f32 = o_0.z;
    var _S18 : f32 = o_0.y;
    var _S19 : f32 = o_0.x;
    return mat3x3<f32>(0.0f, - _S17, _S18, _S17, 0.0f, - _S19, - _S18, _S19, 0.0f);
}

fn build_full_block_0( node_0 : u32) -> bool
{
    var c_1 : u32;
    var k_6 : u32;
    var s_2 : f32;
    var n_4 : BlastNode_std430_0 = nodes_0[node_0];
    var aa_0 : mat3x3<f32> = mat3x3<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    var r_0 : u32 = n_4.ref_begin_0;
    var aa_1 : mat3x3<f32> = aa_0;
    var al_0 : mat3x3<f32> = aa_0;
    var ll_0 : mat3x3<f32> = aa_0;
    loop
    {
        if(r_0 < (n_4.ref_end_0))
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
        var _S20 : f32 = b_1.s_0;
        var s2_0 : f32 = _S20 * _S20;
        var wl_0 : mat3x3<f32> = sym3_0(j_2, u32(3));
        var k_7 : mat3x3<f32> = skew_0(o_1);
        var _S21 : mat3x3<f32> = (((wl_0) * (k_7)));
        var _S22 : mat3x3<f32> = sym3_0(j_2, u32(1)) + (((transpose(k_7)) * (_S21)));
        var _S23 : mat3x3<f32> = mat3x3<f32>(s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        var _S24 : mat3x3<f32> = mat3x3<f32>(- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0);
        var _S25 : mat3x3<f32> = al_0 + mat3x3<f32>(_S24[0] * _S21[0], _S24[1] * _S21[1], _S24[2] * _S21[2]);
        var _S26 : mat3x3<f32> = ll_0 + mat3x3<f32>(_S23[0] * wl_0[0], _S23[1] * wl_0[1], _S23[2] * wl_0[2]);
        aa_1 = aa_1 + mat3x3<f32>(_S23[0] * _S22[0], _S23[1] * _S22[1], _S23[2] * _S22[2]);
        al_0 = _S25;
        ll_0 = _S26;
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
        c_1 = u32(0);
        loop
        {
            if(c_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            m_1[r_0][c_1] = aa_1[r_0][c_1] * n_4.w_ang_0 * n_4.w_ang_0;
            var _S27 : u32 = c_1 + u32(3);
            var _S28 : f32 = al_0[r_0][c_1] * n_4.w_ang_0 * n_4.w_lin_0;
            m_1[r_0][_S27] = _S28;
            m_1[_S27][r_0] = _S28;
            m_1[r_0 + u32(3)][_S27] = ll_0[r_0][c_1] * n_4.w_lin_0 * n_4.w_lin_0;
            c_1 = c_1 + u32(1);
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
        c_1 = u32(0);
        loop
        {
            if(c_1 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + u32(1)) / u32(2) + c_1] = m_1[r_0][c_1];
            c_1 = c_1 + u32(1);
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
        var _S29 : u32 = u32(4) * q_0;
        scratch_0[ns_0(node_0, u32(18) + q_0)] = vec4<f32>(packed_0[_S29], packed_0[_S29 + u32(1)], packed_0[_S29 + u32(2)], packed_0[_S29 + u32(3)]);
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
        c_1 = u32(0);
        loop
        {
            if(c_1 < u32(6))
            {
            }
            else
            {
                break;
            }
            l_1[r_0][c_1] = 0.0f;
            c_1 = c_1 + u32(1);
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
        c_1 = u32(0);
        loop
        {
            if(c_1 <= r_0)
            {
            }
            else
            {
                break;
            }
            k_6 = u32(0);
            s_2 = m_1[r_0][c_1];
            loop
            {
                if(k_6 < c_1)
                {
                }
                else
                {
                    break;
                }
                var s_3 : f32 = s_2 - l_1[r_0][k_6] * l_1[c_1][k_6];
                k_6 = k_6 + u32(1);
                s_2 = s_3;
            }
            if(r_0 == c_1)
            {
                var _S30 : bool;
                if(!(s_2 > 0.0f))
                {
                    _S30 = true;
                }
                else
                {
                    _S30 = !isfinite_0(s_2);
                }
                if(_S30)
                {
                    return false;
                }
                l_1[r_0][c_1] = sqrt(s_2);
            }
            else
            {
                l_1[r_0][c_1] = s_2 / l_1[c_1][c_1];
            }
            c_1 = c_1 + u32(1);
        }
        r_0 = r_0 + u32(1);
    }
    var inv_0 : array<array<f32, i32(6)>, i32(6)>;
    c_1 = u32(0);
    loop
    {
        if(c_1 < u32(6))
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
            if(r_0 == c_1)
            {
                s_2 = 1.0f;
            }
            else
            {
                s_2 = 0.0f;
            }
            k_6 = u32(0);
            var s_4 : f32 = s_2;
            loop
            {
                if(k_6 < r_0)
                {
                }
                else
                {
                    break;
                }
                var s_5 : f32 = s_4 - l_1[r_0][k_6] * y_0[k_6];
                k_6 = k_6 + u32(1);
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
            k_6 = u32(r_1) + u32(1);
            s_2 = y_0[r_1];
            loop
            {
                if(k_6 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var s_6 : f32 = s_2 - l_1[k_6][r_1] * inv_0[k_6][c_1];
                k_6 = k_6 + u32(1);
                s_2 = s_6;
            }
            inv_0[r_1][c_1] = s_2 / l_1[r_1][r_1];
            r_1 = r_1 - i32(1);
        }
        c_1 = c_1 + u32(1);
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
        c_1 = u32(0);
        loop
        {
            if(c_1 <= r_0)
            {
            }
            else
            {
                break;
            }
            packed_0[r_0 * (r_0 + u32(1)) / u32(2) + c_1] = inv_0[r_0][c_1];
            c_1 = c_1 + u32(1);
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
        var _S31 : u32 = u32(4) * q_0;
        scratch_0[ns_0(node_0, u32(24) + q_0)] = vec4<f32>(packed_0[_S31], packed_0[_S31 + u32(1)], packed_0[_S31 + u32(2)], packed_0[_S31 + u32(3)]);
        q_0 = q_0 + u32(1);
    }
    return true;
}

fn build_block_0( node_1 : u32) -> bool
{
    if((params_0.rotation_0) == u32(2))
    {
        var _S32 : bool = build_full_block_0(node_1);
        return _S32;
    }
    var n_5 : BlastNode_std430_0 = nodes_0[node_1];
    var _S33 : vec3<f32> = vec3<f32>(0.0f);
    var r_2 : u32 = n_5.ref_begin_0;
    var diag_0 : vec3<f32> = _S33;
    var offd_0 : vec3<f32> = _S33;
    var ks_0 : vec3<f32> = _S33;
    var cs_0 : f32 = 0.0f;
    loop
    {
        if(r_2 < (n_5.ref_end_0))
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
        var _S34 : f32 = b_2.s_0;
        var s2_1 : f32 = _S34 * _S34;
        var diag_1 : vec3<f32>;
        var offd_1 : vec3<f32>;
        if((params_0.rotation_0) != u32(0))
        {
            var c1_1 : vec3<f32> = stiff_0(j_3, u32(1), vec3<f32>(0.0f, 1.0f, 0.0f));
            var c2_1 : vec3<f32> = stiff_0(j_3, u32(1), vec3<f32>(0.0f, 0.0f, 1.0f));
            var _S35 : vec3<f32> = vec3<f32>(s2_1);
            var _S36 : f32 = o_2.x;
            var _S37 : f32 = o_2.y;
            var _S38 : f32 = o_2.z;
            var offd_2 : vec3<f32> = offd_0 + _S35 * (vec3<f32>(c1_1.x, c2_1.x, c2_1.y) - vec3<f32>(_S36 * _S37, _S36 * _S38, _S37 * _S38));
            diag_1 = diag_0 + _S35 * (vec3<f32>(stiff_0(j_3, u32(1), vec3<f32>(1.0f, 0.0f, 0.0f)).x, c1_1.y, c2_1.z) + vec3<f32>(dot(o_2, o_2)) - o_2 * o_2);
            offd_1 = offd_2;
        }
        else
        {
            var _S39 : vec3<f32> = vec3<f32>(s2_1);
            var _S40 : f32 = o_2.x;
            var _S41 : f32 = o_2.y;
            var _S42 : f32 = o_2.z;
            var offd_3 : vec3<f32> = offd_0 + _S39 * (vec3<f32>(0) - vec3<f32>(_S40 * _S41, _S40 * _S42, _S41 * _S42));
            diag_1 = diag_0 + _S39 * (vec3<f32>((1.0f + dot(o_2, o_2))) - o_2 * o_2);
            offd_1 = offd_3;
        }
        var ks_1 : vec3<f32> = ks_0 + vec3<f32>(s2_1) * o_2;
        var cs_1 : f32 = cs_0 + s2_1;
        diag_0 = diag_1;
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_2 = r_2 + u32(1);
    }
    var _S43 : f32 = n_5.w_ang_0;
    var da_0 : f32 = _S43 * _S43;
    var c_2 : f32 = n_5.w_lin_0 * n_5.w_lin_0 * cs_0;
    var k_8 : vec3<f32> = ks_0 * vec3<f32>((n_5.w_ang_0 * n_5.w_lin_0));
    var a0_0 : vec4<f32> = vec4<f32>(diag_0 * vec3<f32>(da_0), offd_0.x * da_0);
    var _S44 : f32 = offd_0.y * da_0;
    var _S45 : f32 = offd_0.z * da_0;
    var a1_0 : vec4<f32> = vec4<f32>(_S44, _S45, c_2, 0.0f);
    var _S46 : u32 = ns_0(node_1, u32(18));
    scratch_0[_S46] = a0_0;
    scratch_0[_S46 + u32(1)] = a1_0;
    scratch_0[_S46 + u32(2)] = vec4<f32>(k_8, 0.0f);
    if(!(c_2 > 0.0f))
    {
        return false;
    }
    var ic_0 : f32 = 1.0f / c_2;
    var kk_0 : f32 = dot(k_8, k_8);
    var _S47 : f32 = k_8.x;
    var sxx_0 : f32 = a0_0.x + (_S47 * _S47 - kk_0) * ic_0;
    var _S48 : f32 = k_8.y;
    var syy_0 : f32 = a0_0.y + (_S48 * _S48 - kk_0) * ic_0;
    var _S49 : f32 = k_8.z;
    var szz_0 : f32 = a0_0.z + (_S49 * _S49 - kk_0) * ic_0;
    var sxy_0 : f32 = a0_0.w + _S47 * _S48 * ic_0;
    var sxz_0 : f32 = _S44 + _S47 * _S49 * ic_0;
    var syz_0 : f32 = _S45 + _S48 * _S49 * ic_0;
    var cxx_0 : f32 = syy_0 * szz_0 - syz_0 * syz_0;
    var cxy_0 : f32 = sxz_0 * syz_0 - sxy_0 * szz_0;
    var cxz_0 : f32 = sxy_0 * syz_0 - sxz_0 * syy_0;
    var det_0 : f32 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    var _S50 : bool;
    if(!(det_0 > 0.0f))
    {
        _S50 = true;
    }
    else
    {
        _S50 = !isfinite_0(det_0);
    }
    if(_S50)
    {
        return false;
    }
    var id_0 : f32 = 1.0f / det_0;
    var cyz_0 : f32 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    var _S51 : u32 = ns_0(node_1, u32(21));
    scratch_0[_S51] = vec4<f32>(cxx_0 * id_0, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_0, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_0, cxy_0 * id_0);
    scratch_0[_S51 + u32(1)] = vec4<f32>(cxz_0 * id_0, cyz_0 * id_0, ic_0, 0.0f);
    scratch_0[_S51 + u32(2)] = vec4<f32>(k_8 * vec3<f32>(ic_0), 0.0f);
    return true;
}

fn bs_0( j_4 : u32) -> u32
{
    return params_0.node_count_0 * params_0.node_vectors_0 + u32(2) * j_4;
}

fn rot_a_0( j_5 : u32,  v_7 : vec3<f32>) -> vec3<f32>
{
    var _S52 : vec3<f32>;
    if((params_0.rotation_0) != u32(0))
    {
        _S52 = stiff_0(j_5, u32(0), v_7);
    }
    else
    {
        _S52 = v_7;
    }
    return _S52;
}

fn rot_l_0( j_6 : u32,  v_8 : vec3<f32>) -> vec3<f32>
{
    var _S53 : vec3<f32>;
    if((params_0.rotation_0) == u32(2))
    {
        _S53 = stiff_0(j_6, u32(2), v_8);
    }
    else
    {
        _S53 = v_8;
    }
    return _S53;
}

fn bond_term_0( b_3 : ptr<function, BlastBond_std430_0>,  j_7 : u32,  v_9 : u32,  ta_0 : ptr<function, vec3<f32>>,  tl_0 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_3).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_3).n1_0];
    var a0_1 : vec3<f32> = va_0((*b_3).n0_0, v_9) * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = vl_0((*b_3).n0_0, v_9) * vec3<f32>(m0_0.w_lin_0);
    var a1_1 : vec3<f32> = va_0((*b_3).n1_0, v_9) * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = vl_0((*b_3).n1_0, v_9) * vec3<f32>(m1_0.w_lin_0);
    if((params_0.rotation_0) == u32(0))
    {
        var _S54 : vec3<f32> = vec3<f32>((*b_3).s_0);
        (*ta_0) = (a0_1 - a1_1) * _S54;
        (*tl_0) = (l0_0 - l1_0 + (cross((*b_3).o0_0.xyz, a0_1) - cross((*b_3).o1_0.xyz, a1_1))) * _S54;
        return;
    }
    var _S55 : vec3<f32> = vec3<f32>((*b_3).s_0);
    (*ta_0) = rot_a_0(j_7, a0_1 - a1_1) * _S55;
    (*tl_0) = rot_l_0(j_7, l0_0 - l1_0 + (cross((*b_3).o0_0.xyz, a0_1) - cross((*b_3).o1_0.xyz, a1_1))) * _S55;
    return;
}

fn bond_term_1( b_4 : ptr<function, BlastBond_std430_0>,  j_8 : u32,  v_10 : u32,  ta_1 : ptr<function, vec3<f32>>,  tl_1 : ptr<function, vec3<f32>>)
{
    var m0_1 : BlastNode_std430_0 = nodes_0[(*b_4).n0_0];
    var m1_1 : BlastNode_std430_0 = nodes_0[(*b_4).n1_0];
    var a0_2 : vec3<f32> = va_0((*b_4).n0_0, v_10) * vec3<f32>(m0_1.w_ang_0);
    var l0_1 : vec3<f32> = vl_0((*b_4).n0_0, v_10) * vec3<f32>(m0_1.w_lin_0);
    var a1_2 : vec3<f32> = va_0((*b_4).n1_0, v_10) * vec3<f32>(m1_1.w_ang_0);
    var l1_1 : vec3<f32> = vl_0((*b_4).n1_0, v_10) * vec3<f32>(m1_1.w_lin_0);
    if((params_0.rotation_0) == u32(0))
    {
        var _S56 : vec3<f32> = vec3<f32>((*b_4).s_0);
        (*ta_1) = (a0_2 - a1_2) * _S56;
        (*tl_1) = (l0_1 - l1_1 + (cross((*b_4).o0_0.xyz, a0_2) - cross((*b_4).o1_0.xyz, a1_2))) * _S56;
        return;
    }
    var _S57 : vec3<f32> = vec3<f32>((*b_4).s_0);
    (*ta_1) = rot_a_0(j_8, a0_2 - a1_2) * _S57;
    (*tl_1) = rot_l_0(j_8, l0_1 - l1_1 + (cross((*b_4).o0_0.xyz, a0_2) - cross((*b_4).o1_0.xyz, a1_2))) * _S57;
    return;
}

fn gather_0( node_2 : u32,  from_t_0 : bool,  with_u_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_6 : BlastNode_std430_0 = nodes_0[node_2];
    var _S58 : vec3<f32> = vec3<f32>(0.0f);
    var r_3 : u32 = n_6.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S58;
    var acc_l_0 : vec3<f32> = _S58;
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
        var _S59 : BlastBond_std430_0 = bonds_0[j_9];
        if(!((_S59.health_0) > 0.0f))
        {
            r_3 = r_3 + u32(1);
            continue;
        }
        var ya_0 : vec3<f32>;
        var yl_0 : vec3<f32>;
        if(from_t_0)
        {
            var _S60 : u32 = bs_0(j_9);
            var _S61 : vec3<f32> = scratch_0[_S60 + u32(1)].xyz;
            ya_0 = scratch_0[_S60].xyz;
            yl_0 = _S61;
        }
        else
        {
            var _S62 : u32 = j_9 * u32(2);
            var ya_1 : vec3<f32> = x_0[_S62].xyz;
            var yl_1 : vec3<f32> = x_0[_S62 + u32(1)].xyz;
            if(with_u_0)
            {
                var ta_2 : vec3<f32>;
                var tl_2 : vec3<f32>;
                bond_term_1(&(_S59), j_9, u32(6), &(ta_2), &(tl_2));
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
        var yl_3 : vec3<f32>;
        var ya_2 : vec3<f32>;
        if((params_0.rotation_0) == u32(0))
        {
            var _S63 : vec3<f32> = vec3<f32>(_S59.s_0);
            var ya_3 : vec3<f32> = ya_0 * _S63;
            yl_3 = yl_0 * _S63;
            ya_2 = ya_3;
        }
        else
        {
            var _S64 : vec3<f32> = vec3<f32>(_S59.s_0);
            var _S65 : vec3<f32> = rot_a_0(j_9, ya_0) * _S64;
            yl_3 = rot_l_0(j_9, yl_0) * _S64;
            ya_2 = _S65;
        }
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_2 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross(_S59.o1_0.xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross(_S59.o0_0.xyz, yl_3));
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
    var _S66 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(s_7, 0.0f));
    return _S66.x;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn frame_float_0( k_9 : u32,  w_1 : u32) -> f32
{
    return (bitcast<f32>((frame_word_0(k_9, w_1))));
}

fn frame_axis_0( k_10 : u32,  rotations_0 : u32,  kk_1 : u32) -> vec3<f32>
{
    if(rotations_0 == u32(1))
    {
        return vec3<f32>(frame_float_0(k_10, u32(4)), frame_float_0(k_10, u32(5)), frame_float_0(k_10, u32(6)));
    }
    var _S67 : f32;
    if(kk_1 == u32(0))
    {
        _S67 = 1.0f;
    }
    else
    {
        _S67 = 0.0f;
    }
    var _S68 : f32;
    if(kk_1 == u32(1))
    {
        _S68 = 1.0f;
    }
    else
    {
        _S68 = 0.0f;
    }
    var _S69 : f32;
    if(kk_1 == u32(2))
    {
        _S69 = 1.0f;
    }
    else
    {
        _S69 = 0.0f;
    }
    return vec3<f32>(_S67, _S68, _S69);
}

fn frame_terms_0( k_11 : u32,  rotations_1 : u32,  i_1 : u32,  v_11 : u32,  s_9 : ptr<function, array<f32, i32(6)>>)
{
    var n_7 : BlastNode_std430_0 = nodes_0[i_1];
    var lin_1 : vec3<f32> = vl_0(i_1, v_11) / vec3<f32>(n_7.w_lin_0);
    var _S70 : vec3<f32> = va_0(i_1, v_11) / vec3<f32>(n_7.w_ang_0) - cross(n_7.position_0.xyz, lin_1);
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
        (*s_9)[kk_2] = (*s_9)[kk_2] + dot(frame_axis_0(k_11, rotations_1, kk_2), _S70);
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

fn frame_solve_0( k_12 : u32,  rotations_2 : u32,  sum_0 : array<f32, i32(6)>,  omega_0 : ptr<function, vec3<f32>>,  t_0 : ptr<function, vec3<f32>>)
{
    var j_11 : u32;
    var d_0 : u32 = rotations_2 + u32(3);
    var c_3 : array<f32, i32(6)>;
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
        c_3[z_1] = 0.0f;
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
        var _S71 : f32 = sum_0[row_0] * frame_float_0(k_12, u32(8) + row_0);
        j_11 = u32(0);
        var value_2 : f32 = _S71;
        loop
        {
            if(j_11 < row_0)
            {
            }
            else
            {
                break;
            }
            var value_3 : f32 = value_2 - frame_float_0(k_12, u32(14) + row_0 * (row_0 + u32(1)) / u32(2) + j_11) * c_3[j_11];
            j_11 = j_11 + u32(1);
            value_2 = value_3;
        }
        var _S72 : u32 = row_0 + u32(1);
        c_3[row_0] = value_2 / frame_float_0(k_12, u32(14) + row_0 * _S72 / u32(2) + row_0);
        row_0 = _S72;
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
        var _S73 : u32 = r_4 + u32(1);
        j_11 = _S73;
        loop
        {
            if(j_11 < d_0)
            {
            }
            else
            {
                break;
            }
            var _S74 : u32 = j_11 + u32(1);
            c_3[r_4] = c_3[r_4] - frame_float_0(k_12, u32(14) + j_11 * _S74 / u32(2) + r_4) * c_3[j_11];
            j_11 = _S74;
        }
        c_3[r_4] = c_3[r_4] / frame_float_0(k_12, u32(14) + r_4 * _S73 / u32(2) + r_4);
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
        c_3[z_1] = c_3[z_1] * frame_float_0(k_12, u32(8) + z_1);
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
        (*omega_0) = (*omega_0) + frame_axis_0(k_12, rotations_2, j_11) * vec3<f32>(c_3[j_11]);
        j_11 = j_11 + u32(1);
    }
    (*t_0) = vec3<f32>(c_3[rotations_2], c_3[rotations_2 + u32(1)], c_3[rotations_2 + u32(2)]);
    return;
}

var<workgroup> g_omega_0 : vec3<f32>;

var<workgroup> g_t_0 : vec3<f32>;

fn frame_apply_0( i_2 : u32,  v_12 : u32,  omega_1 : vec3<f32>,  t_1 : vec3<f32>)
{
    var n_8 : BlastNode_std430_0 = nodes_0[i_2];
    set_v_0(i_2, v_12, va_0(i_2, v_12) - omega_1 / vec3<f32>(n_8.w_ang_0), vl_0(i_2, v_12) - (t_1 + cross(n_8.position_0.xyz, omega_1)) / vec3<f32>(n_8.w_lin_0));
    return;
}

fn project_0( tid_2 : u32,  k_13 : u32,  rotations_3 : u32,  n0_2 : u32,  n1_2 : u32,  v_13 : u32,  free_0 : bool)
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
    var _S75 : u32 = n0_2 + tid_2;
    var p_1 : u32 = _S75;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_terms_0(k_13, rotations_3, lists_0[p_1], v_13, &(s_10));
        p_1 = p_1 + u32(1024);
    }
    var a_3 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_10[i32(0)], s_10[i32(1)]));
    var b_5 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_10[i32(2)], s_10[i32(3)]));
    var c_4 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_10[i32(4)], s_10[i32(5)]));
    if(tid_2 == u32(0))
    {
        var sum_1 : array<f32, i32(6)> = array<f32, i32(6)>( a_3.x, a_3.y, b_5.x, b_5.y, c_4.x, c_4.y );
        var omega_2 : vec3<f32>;
        var t_2 : vec3<f32>;
        frame_solve_0(k_13, rotations_3, sum_1, &(omega_2), &(t_2));
        g_omega_0 = omega_2;
        g_t_0 = t_2;
    }
    workgroupBarrier();
    var _S76 : vec3<f32> = g_omega_0;
    var _S77 : vec3<f32> = g_t_0;
    p_1 = _S75;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_apply_0(lists_0[p_1], v_13, _S76, _S77);
        p_1 = p_1 + u32(1024);
    }
    device_group_barrier_0();
    return;
}

fn bond_pass_0( tid_3 : u32,  b0_0 : u32,  b1_0 : u32,  v_14 : u32) -> f32
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
        var _S78 : BlastBond_std430_0 = bonds_0[j_12];
        var _S79 : vec3<f32> = vec3<f32>(0.0f);
        var ta_3 : vec3<f32> = _S79;
        var tl_3 : vec3<f32> = _S79;
        if((_S78.health_0) > 0.0f)
        {
            bond_term_0(&(_S78), j_12, v_14, &(ta_3), &(tl_3));
        }
        var z_4 : f32 = z_3 + norm2_0(ta_3, tl_3);
        var _S80 : u32 = bs_0(j_12);
        scratch_0[_S80] = vec4<f32>(ta_3, 0.0f);
        scratch_0[_S80 + u32(1)] = vec4<f32>(tl_3, 0.0f);
        q_1 = q_1 + u32(1024);
        z_3 = z_4;
    }
    var _S81 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(z_3, 0.0f));
    return _S81.x;
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
    var _S82 : vec2<f32> = blast_group_sum_0(tid_4, vec2<f32>(s_11, 0.0f));
    return _S82.x;
}

fn full_apply_0( n_9 : u32,  at_1 : u32,  a_4 : vec3<f32>,  l_3 : vec3<f32>,  oa_0 : ptr<function, vec3<f32>>,  ol_0 : ptr<function, vec3<f32>>)
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
        var _S83 : u32 = u32(4) * q_2;
        m_2[_S83] = scratch_0[ns_0(n_9, at_1 + q_2)].x;
        m_2[_S83 + u32(1)] = scratch_0[ns_0(n_9, at_1 + q_2)].y;
        m_2[_S83 + u32(2)] = scratch_0[ns_0(n_9, at_1 + q_2)].z;
        m_2[_S83 + u32(3)] = scratch_0[ns_0(n_9, at_1 + q_2)].w;
        q_2 = q_2 + u32(1);
    }
    var _S84 : array<f32, i32(6)> = array<f32, i32(6)>( a_4.x, a_4.y, a_4.z, l_3.x, l_3.y, l_3.z );
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
        var c_5 : u32 = u32(0);
        var s_13 : f32 = 0.0f;
        loop
        {
            if(c_5 < u32(6))
            {
            }
            else
            {
                break;
            }
            if(r_5 >= c_5)
            {
                q_2 = r_5 * (r_5 + u32(1)) / u32(2) + c_5;
            }
            else
            {
                q_2 = c_5 * (c_5 + u32(1)) / u32(2) + r_5;
            }
            var s_14 : f32 = s_13 + m_2[q_2] * _S84[c_5];
            c_5 = c_5 + u32(1);
            s_13 = s_14;
        }
        o_3[r_5] = s_13;
        r_5 = r_5 + u32(1);
    }
    (*oa_0) = vec3<f32>(o_3[i32(0)], o_3[i32(1)], o_3[i32(2)]);
    (*ol_0) = vec3<f32>(o_3[i32(3)], o_3[i32(4)], o_3[i32(5)]);
    return;
}

fn sym_mul_0( s0_0 : vec4<f32>,  s1_0 : vec4<f32>,  v_15 : vec3<f32>) -> vec3<f32>
{
    var _S85 : f32 = v_15.x;
    var _S86 : f32 = s0_0.w;
    var _S87 : f32 = v_15.y;
    var _S88 : f32 = s1_0.x;
    var _S89 : f32 = v_15.z;
    var _S90 : f32 = s1_0.y;
    return vec3<f32>(s0_0.x * _S85 + _S86 * _S87 + _S88 * _S89, _S86 * _S85 + s0_0.y * _S87 + _S90 * _S89, _S88 * _S85 + _S90 * _S87 + s0_0.z * _S89);
}

fn inverse_apply_0( n_10 : u32,  a_5 : vec3<f32>,  l_4 : vec3<f32>,  oa_1 : ptr<function, vec3<f32>>,  ol_1 : ptr<function, vec3<f32>>)
{
    if((params_0.rotation_0) == u32(2))
    {
        full_apply_0(n_10, u32(24), a_5, l_4, &((*oa_1)), &((*ol_1)));
        return;
    }
    var _S91 : u32 = ns_0(n_10, u32(21));
    var s1_1 : vec4<f32> = scratch_0[_S91 + u32(1)];
    var kc_0 : vec3<f32> = scratch_0[_S91 + u32(2)].xyz;
    var _S92 : vec3<f32> = sym_mul_0(scratch_0[_S91], scratch_0[_S91 + u32(1)], a_5 + cross(kc_0, l_4));
    (*oa_1) = _S92;
    (*ol_1) = l_4 * vec3<f32>(s1_1.z) - cross(kc_0, _S92);
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
    var _S93 : u32 = ns_0(n_11, u32(18));
    var b1_1 : vec4<f32> = scratch_0[_S93 + u32(1)];
    var k_14 : vec3<f32> = scratch_0[_S93 + u32(2)].xyz;
    (*oa_2) = sym_mul_0(scratch_0[_S93], scratch_0[_S93 + u32(1)], a_7) - cross(k_14, l_6);
    (*ol_2) = cross(k_14, a_7) + vec3<f32>(b1_1.z) * l_6;
    return;
}

var<workgroup> g_max_0 : array<f32, i32(1024)>;

var<workgroup> g_max_out_0 : f32;

fn group_max_0( tid_6 : u32,  v_16 : f32) -> f32
{
    g_max_0[tid_6] = v_16;
    workgroupBarrier();
    var k_15 : u32 = u32(512);
    loop
    {
        if(k_15 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_6 < k_15)
        {
            g_max_0[tid_6] = max(g_max_0[tid_6], g_max_0[tid_6 + k_15]);
        }
        workgroupBarrier();
        k_15 = (k_15 >> (u32(1)));
    }
    if(tid_6 == u32(0))
    {
        g_max_out_0 = g_max_0[i32(0)];
    }
    var _S94 : f32 = (workgroupUniformLoad(&((g_max_out_0))));
    workgroupBarrier();
    return _S94;
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
    var k_16 : u32 = group_0.x;
    if(k_16 >= (params_0.island_count_0))
    {
        return;
    }
    var n0_5 : u32 = islands_0[k_16].x;
    var n1_5 : u32 = islands_0[k_16].y;
    var b0_1 : u32 = islands_0[k_16].z;
    var _S95 : u32 = islands_0[k_16].w;
    var _S96 : u32 = n1_5 - n0_5;
    var p_4 : u32;
    if(_S96 > (params_0.wide_min_0))
    {
        p_4 = u32(1);
    }
    else
    {
        p_4 = u32(0);
    }
    var _S97 : u32 = group_uniform_0(p_4, tid_7);
    if(_S97 != u32(0))
    {
        return;
    }
    var last_0 : vec4<u32> = report_0[k_16];
    var last_flags_0 : u32 = group_uniform_0(report_0[k_16].y, tid_7);
    var warm_1 : bool = (params_0.warm_0) != u32(0);
    var carried_0 : bool;
    if(warm_1)
    {
        carried_0 = ((last_flags_0 & (u32(4)))) != u32(0);
    }
    else
    {
        carried_0 = false;
    }
    var singular_0 : f32;
    if(carried_0)
    {
        var _S98 : u32 = n0_5 + tid_7;
        singular_0 = 0.0f;
        p_4 = _S98;
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
            var _S99 : u32 = i_5 * u32(2);
            var _S100 : u32 = ns_0(i_5, u32(16));
            if((any((accel_0[_S99] != scratch_0[_S100]))))
            {
                carried_0 = true;
            }
            else
            {
                carried_0 = (any((accel_0[_S99 + u32(1)] != scratch_0[_S100 + u32(1)])));
            }
            if(carried_0)
            {
                singular_0 = 1.0f;
            }
            p_4 = p_4 + u32(1024);
        }
        var _S101 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(singular_0, 0.0f));
        if((_S101.x) == 0.0f)
        {
            if(tid_7 == u32(0))
            {
                report_0[k_16] = vec4<u32>(u32(0), u32(7), last_0.z, last_0.w);
            }
            return;
        }
    }
    var carry_1 : vec4<u32> = group_uniform4_0(vec4<u32>(carry_of_0(k_16), _S96), tid_7);
    var kept_0 : vec3<u32> = carry_1.xyz;
    var carried_1 : bool = carry_kept_0(kept_0, carry_1.w);
    device_group_barrier_0();
    var _S102 : bool = tid_7 == u32(0);
    if(_S102)
    {
        carried_0 = (params_0.carry_0) != u32(0);
    }
    else
    {
        carried_0 = false;
    }
    if(carried_0)
    {
        set_carry_0(k_16, u32(0), u32(0), 0.0f);
    }
    var _S103 : u32 = n0_5 + tid_7;
    singular_0 = 0.0f;
    p_4 = _S103;
    var target_0 : f32 = 0.0f;
    var change_0 : f32 = 0.0f;
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
        var _S104 : u32 = i_6 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S104];
        var a_lin_0 : vec4<f32> = accel_0[_S104 + u32(1)];
        var _S105 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz);
        if((n_12.w_ang_0) > 0.0f)
        {
            force_start_0 = 1.0f / n_12.w_ang_0;
        }
        else
        {
            force_start_0 = 1.0f;
        }
        var ba_0 : vec3<f32> = _S105 * vec3<f32>(force_start_0);
        var _S106 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz);
        if((n_12.w_lin_0) > 0.0f)
        {
            best_0 = params_0.recip_length_0 / n_12.w_lin_0;
        }
        else
        {
            best_0 = params_0.recip_length_0;
        }
        var bl_0 : vec3<f32> = _S106 * vec3<f32>(best_0);
        var target_1 : f32 = target_0 + norm2_0(ba_0, bl_0);
        if(carried_1)
        {
            change_0 = change_0 + norm2_0(ba_0 - va_0(i_6, u32(14)), bl_0 - vl_0(i_6, u32(14)));
        }
        set_v_0(i_6, u32(14), ba_0, bl_0);
        set_v_0(i_6, u32(0), ba_0, bl_0);
        var _S107 : u32 = ns_0(i_6, u32(16));
        scratch_0[_S107] = a_ang_0;
        scratch_0[_S107 + u32(1)] = a_lin_0;
        var _S108 : vec3<f32> = vec3<f32>(0.0f);
        set_v_0(i_6, u32(6), _S108, _S108);
        if((params_0.carry_0) == u32(0))
        {
            set_v_0(i_6, u32(2), _S108, _S108);
        }
        var _S109 : bool = build_block_0(i_6);
        if(!_S109)
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
            if(q_4 < _S95)
            {
            }
            else
            {
                break;
            }
            var _S110 : u32 = lists_0[q_4] * u32(2);
            var _S111 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S110] = _S111;
            x_0[_S110 + u32(1)] = _S111;
            q_4 = q_4 + u32(1024);
        }
    }
    var tsum_0 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(target_0, singular_0));
    var _S112 : f32 = tsum_0.x;
    var delta2_0 : f32 = _S112 * params_0.tolerance_0 * params_0.tolerance_0;
    var _S113 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(change_0, 0.0f));
    var moved_0 : f32 = _S113.x;
    if(carried_1)
    {
        carried_0 = moved_0 <= (params_0.tolerance_0 * params_0.tolerance_0 * _S112);
    }
    else
    {
        carried_0 = false;
    }
    var _S114 : vec4<u32> = group_uniform4_0(vec4<u32>(frame_word_0(k_16, u32(0)), frame_word_0(k_16, u32(1)), frame_word_0(k_16, u32(2)), u32(0)), tid_7);
    var frame_0 : vec3<u32> = _S114.xyz;
    var converged_0 : bool;
    if((tsum_0.y) > 0.0f)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (frame_0.z) != u32(0);
    }
    var _S115 : bool = (frame_0.x) == u32(0);
    var _S116 : u32 = frame_0.y;
    device_group_barrier_0();
    if(!(delta2_0 > 0.0f))
    {
        q_4 = b0_1 + tid_7;
        loop
        {
            if(q_4 < _S95)
            {
            }
            else
            {
                break;
            }
            var _S117 : u32 = lists_0[q_4] * u32(2);
            var _S118 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S117] = _S118;
            x_0[_S117 + u32(1)] = _S118;
            q_4 = q_4 + u32(1024);
        }
        if(_S102)
        {
            if(warm_1)
            {
                p_4 = u32(4);
            }
            else
            {
                p_4 = u32(0);
            }
            report_0[k_16] = vec4<u32>(u32(0), (u32(1) | (p_4)), (bitcast<u32>((0.0f))), (bitcast<u32>((delta2_0))));
        }
        return;
    }
    if(warm_1)
    {
        var _S119 : f32 = rebuild_0(tid_7, n0_5, n1_5, false);
    }
    if(carried_0)
    {
        singular_0 = (bitcast<f32>((kept_0.z)));
    }
    else
    {
        singular_0 = 0.0f;
    }
    var _S120 : f32 = float_bits_0(u32(2139095040));
    if((params_0.force_tolerance_0) > 0.0f)
    {
        q_4 = b0_1 + tid_7;
        force_start_0 = 0.0f;
        loop
        {
            if(q_4 < _S95)
            {
            }
            else
            {
                break;
            }
            var _S121 : u32 = lists_0[q_4] * u32(2);
            var f_0 : f32 = force_start_0 + norm2_0(x_0[_S121].xyz, x_0[_S121 + u32(1)].xyz);
            q_4 = q_4 + u32(1024);
            force_start_0 = f_0;
        }
        var _S122 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(force_start_0, 0.0f));
        force_start_0 = sqrt(_S122.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    best_0 = _S120;
    q_4 = u32(0);
    var step_0 : f32 = _S120;
    var previous_0 : f32 = singular_0;
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
            var _S123 : bool = converged_0;
            converged_0 = false;
            failed_0 = _S123;
            break;
        }
        project_0(tid_7, k_16, _S116, n0_5, n1_5, u32(0), _S115);
        var num_1 : f32 = bond_pass_0(tid_7, b0_1, _S95, u32(0));
        var _S124 : bool = num_1 <= delta2_0;
        if(_S124)
        {
            var _S125 : f32 = residual_norm_0(tid_7, n0_5, n1_5);
            ok_0 = !(_S125 <= delta2_0);
        }
        else
        {
            ok_0 = false;
        }
        var _S126 : bool = iteration_0 > u32(0);
        var _S127 : bool;
        if(_S126)
        {
            _S127 = true;
        }
        else
        {
            _S127 = warm_1;
        }
        var _S128 : bool;
        if(_S127)
        {
            _S128 = _S124;
        }
        else
        {
            _S128 = false;
        }
        var _S129 : bool;
        if(_S128)
        {
            _S129 = !ok_0;
        }
        else
        {
            _S129 = false;
        }
        var hold_0 : bool;
        var num_2 : f32;
        var previous_1 : f32;
        if(_S129)
        {
            var _S130 : f32 = rebuild_0(tid_7, n0_5, n1_5, true);
            project_0(tid_7, k_16, _S116, n0_5, n1_5, u32(0), _S115);
            var num_3 : f32 = bond_pass_0(tid_7, b0_1, _S95, u32(0));
            if(num_3 <= delta2_0)
            {
                var _S131 : f32 = residual_norm_0(tid_7, n0_5, n1_5);
                hold_0 = !(_S131 <= delta2_0);
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
                var _S132 : bool = converged_0;
                converged_0 = false;
                failed_0 = _S132;
                num_0 = num_2;
                previous_0 = previous_1;
                break;
            }
        }
        var _S133 : bool;
        if(num_2 <= delta2_0)
        {
            _S133 = !hold_0;
        }
        else
        {
            _S133 = false;
        }
        if(_S133)
        {
            var _S134 : bool = converged_0;
            converged_0 = true;
            failed_0 = _S134;
            num_0 = num_2;
            previous_0 = previous_1;
            break;
        }
        var _S135 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            _S135 = _S126;
        }
        else
        {
            _S135 = false;
        }
        var _S136 : bool;
        if(_S135)
        {
            _S136 = step_0 <= (params_0.force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S136 = false;
        }
        var mloc_0 : f32;
        if(_S136)
        {
            q_3 = b0_1 + tid_7;
            mloc_0 = 0.0f;
            loop
            {
                if(q_3 < _S95)
                {
                }
                else
                {
                    break;
                }
                var j_13 : u32 = lists_0[q_3];
                var _S137 : BlastBond_std430_0 = bonds_0[j_13];
                var ta_4 : vec3<f32>;
                var tl_4 : vec3<f32>;
                bond_term_1(&(_S137), j_13, u32(6), &(ta_4), &(tl_4));
                var _S138 : u32 = j_13 * u32(2);
                var f_1 : f32 = mloc_0 + norm2_0(x_0[_S138].xyz + ta_4, x_0[_S138 + u32(1)].xyz + tl_4);
                q_3 = q_3 + u32(1024);
                mloc_0 = f_1;
            }
            var _S139 : f32 = params_0.force_tolerance_0;
            var _S140 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(mloc_0, 0.0f));
            if(step_0 <= (_S139 * sqrt(_S140.x)))
            {
                var _S141 : bool = converged_0;
                converged_0 = true;
                failed_0 = _S141;
                num_0 = num_2;
                previous_0 = previous_1;
                break;
            }
        }
        device_group_barrier_0();
        var _S142 : bool = iteration_0 == u32(0);
        var _S143 : bool;
        if(_S142)
        {
            _S143 = (params_0.first_polynomial_0) == u32(0);
        }
        else
        {
            _S143 = false;
        }
        var _S144 : bool;
        if(_S143)
        {
            _S144 = !carried_0;
        }
        else
        {
            _S144 = false;
        }
        if(_S144)
        {
            p_4 = _S103;
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
            p_4 = _S103;
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
            var _S145 : f32 = bond_pass_0(tid_7, b0_1, _S95, u32(10));
            device_group_barrier_0();
            var _S146 : f32 = node_pass_0(tid_7, n0_5, n1_5, u32(12));
            q_3 = _S103;
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
                var _S147 : vec3<f32> = vec3<f32>(1.68946564197540283f);
                var _S148 : vec3<f32> = vec3<f32>(1.5220409631729126f);
                set_v_0(i_9, u32(8), _S147 * va_0(i_9, u32(10)) - _S148 * ma_0, _S147 * vl_0(i_9, u32(10)) - _S148 * ml_0);
                q_3 = q_3 + u32(1024);
            }
        }
        device_group_barrier_0();
        project_0(tid_7, k_16, _S116, n0_5, n1_5, u32(8), _S115);
        mloc_0 = 0.0f;
        p_4 = _S103;
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
            var _S149 : f32 = max(mloc_0, max(max(max(a_9.x, a_9.y), max(a_9.z, l_8.x)), max(l_8.y, l_8.z)));
            var p_5 : u32 = p_4 + u32(1024);
            mloc_0 = _S149;
            p_4 = p_5;
        }
        var mag_0 : f32 = group_max_0(tid_7, mloc_0);
        var _S150 : bool;
        if(!(mag_0 > 0.0f))
        {
            _S150 = true;
        }
        else
        {
            _S150 = !isfinite_0(mag_0);
        }
        if(_S150)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            previous_0 = previous_1;
            break;
        }
        var _S151 : f32 = 1.0f / mag_0;
        q_3 = _S103;
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
            var _S152 : vec3<f32> = vec3<f32>(_S151);
            var ga_0 : vec3<f32> = va_0(i_11, u32(8)) * _S152;
            var gn_0 : vec3<f32> = vl_0(i_11, u32(8)) * _S152;
            set_v_0(i_11, u32(8), ga_0, gn_0);
            var gl_1 : f32 = gl_0 + (dot(va_0(i_11, u32(0)), ga_0) + dot(vl_0(i_11, u32(0)), gn_0));
            q_3 = q_3 + u32(1024);
            gl_0 = gl_1;
        }
        var _S153 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(gl_0, 0.0f));
        var gamma_2 : f32 = _S153.x;
        var _S154 : bool = gamma_2 > 0.0f;
        var _S155 : bool;
        if(!_S154)
        {
            _S155 = true;
        }
        else
        {
            _S155 = !isfinite_0(gamma_2);
        }
        if(_S155)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            previous_0 = previous_1;
            break;
        }
        var _S156 : bool;
        if(iteration_0 > u32(1))
        {
            _S156 = true;
        }
        else
        {
            if((params_0.first_polynomial_0) != u32(0))
            {
                _S156 = _S126;
            }
            else
            {
                _S156 = false;
            }
        }
        var _S157 : bool;
        if(_S156)
        {
            _S157 = true;
        }
        else
        {
            _S157 = carried_0;
        }
        var _S158 : bool;
        if(_S157)
        {
            _S158 = previous_1 > 0.0f;
        }
        else
        {
            _S158 = false;
        }
        var beta_0 : f32;
        if(_S158)
        {
            beta_0 = gamma_2 / previous_1;
        }
        else
        {
            beta_0 = 0.0f;
        }
        var _S159 : bool;
        if(carried_0)
        {
            _S159 = _S142;
        }
        else
        {
            _S159 = false;
        }
        var _S160 : bool;
        if(_S159)
        {
            _S160 = !(beta_0 <= 16.0f);
        }
        else
        {
            _S160 = false;
        }
        var beta_1 : f32;
        if(_S160)
        {
            beta_1 = 0.0f;
        }
        else
        {
            beta_1 = beta_0;
        }
        var p_6 : u32 = _S103;
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
            if(beta_1 == 0.0f)
            {
                set_v_0(i_12, u32(2), va_0(i_12, u32(8)), vl_0(i_12, u32(8)));
            }
            else
            {
                set_v_0(i_12, u32(2), va_0(i_12, u32(8)) + va_0(i_12, u32(2)) * vec3<f32>(beta_1), vl_0(i_12, u32(8)) + vl_0(i_12, u32(2)) * vec3<f32>(beta_1));
            }
            p_6 = p_6 + u32(1024);
        }
        device_group_barrier_0();
        var den_0 : f32 = bond_pass_0(tid_7, b0_1, _S95, u32(2));
        device_group_barrier_0();
        var _S161 : f32 = node_pass_0(tid_7, n0_5, n1_5, u32(4));
        var _S162 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            if(den_0 > 0.0f)
            {
                _S162 = _S154;
            }
            else
            {
                _S162 = false;
            }
            var step_1 : f32;
            if(_S162)
            {
                step_1 = gamma_2 / sqrt(den_0);
            }
            else
            {
                step_1 = _S120;
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
            var _S163 : bool;
            if(_S126)
            {
                _S163 = true;
            }
            else
            {
                _S163 = (params_0.first_polynomial_0) != u32(0);
            }
            var step_2 : f32;
            if(!_S163)
            {
                step_2 = _S120;
            }
            else
            {
                step_2 = step_1;
            }
            step_0 = step_2;
            travel_0 = travel_1;
        }
        var _S164 : u32 = iteration_0 + u32(1);
        if(!(den_0 > 0.0f))
        {
            _S162 = true;
        }
        else
        {
            _S162 = !isfinite_0(den_0);
        }
        if(_S162)
        {
            var _S165 : bool = converged_0;
            converged_0 = false;
            failed_0 = _S165;
            num_0 = num_2;
            previous_0 = gamma_2;
            iteration_0 = _S164;
            break;
        }
        var _S166 : f32 = gamma_2 / den_0;
        var p_7 : u32 = _S103;
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
            var _S167 : vec3<f32> = vec3<f32>(_S166);
            set_v_0(i_13, u32(6), va_0(i_13, u32(6)) + va_0(i_13, u32(2)) * _S167, vl_0(i_13, u32(6)) + vl_0(i_13, u32(2)) * _S167);
            set_v_0(i_13, u32(0), va_0(i_13, u32(0)) - va_0(i_13, u32(4)) * _S167, vl_0(i_13, u32(0)) - vl_0(i_13, u32(4)) * _S167);
            p_7 = p_7 + u32(1024);
        }
        device_group_barrier_0();
        previous_0 = gamma_2;
        num_0 = num_2;
        iteration_0 = _S164;
    }
    device_group_barrier_0();
    q_3 = b0_1 + tid_7;
    loop
    {
        if(q_3 < _S95)
        {
        }
        else
        {
            break;
        }
        var j_14 : u32 = lists_0[q_3];
        var _S168 : BlastBond_std430_0 = bonds_0[j_14];
        if(!((_S168.health_0) > 0.0f))
        {
            var _S169 : u32 = j_14 * u32(2);
            var _S170 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S169] = _S170;
            x_0[_S169 + u32(1)] = _S170;
            q_3 = q_3 + u32(1024);
            continue;
        }
        var ta_5 : vec3<f32>;
        var tl_5 : vec3<f32>;
        bond_term_1(&(_S168), j_14, u32(6), &(ta_5), &(tl_5));
        var _S171 : u32 = j_14 * u32(2);
        x_0[_S171] = vec4<f32>(x_0[_S171].xyz + ta_5, 0.0f);
        var _S172 : u32 = _S171 + u32(1);
        x_0[_S172] = vec4<f32>(x_0[_S172].xyz + tl_5, 0.0f);
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
        ok_0 = warm_1;
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = iteration_0 == u32(0);
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        q_4 = u32(4);
    }
    else
    {
        q_4 = u32(0);
    }
    var flags_0 : u32 = (p_4 | (q_4));
    if(_S102)
    {
        report_0[k_16] = vec4<u32>(iteration_0, flags_0, (bitcast<u32>((num_0))), (bitcast<u32>((delta2_0))));
        if((params_0.carry_0) != u32(0))
        {
            failed_0 = !failed_0;
        }
        else
        {
            failed_0 = false;
        }
        if(failed_0)
        {
            converged_0 = !converged_0;
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
            if(carried_0)
            {
                p_4 = kept_0.y;
            }
            else
            {
                p_4 = u32(0);
            }
            set_carry_0(k_16, _S96, p_4 + iteration_0, previous_0);
        }
    }
    return;
}

