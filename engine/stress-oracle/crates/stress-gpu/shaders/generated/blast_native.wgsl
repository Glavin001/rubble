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

@binding(6) @group(0) var<storage, read_write> x_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> positions_0 : array<vec4<f32>>;

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
    return n_0 * u32(24) + v_0;
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

fn build_block_0( node_0 : u32) -> bool
{
    var n_2 : BlastNode_std430_0 = nodes_0[node_0];
    var _S10 : vec3<f32> = vec3<f32>(0.0f);
    var r_0 : u32 = n_2.ref_begin_0;
    var diag_0 : vec3<f32> = _S10;
    var offd_0 : vec3<f32> = _S10;
    var ks_0 : vec3<f32> = _S10;
    var cs_0 : f32 = 0.0f;
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
        var b_1 : BlastBond_std430_0 = bonds_0[(ref_0 & (u32(2147483647)))];
        if(!((bonds_0[(ref_0 & (u32(2147483647)))].health_0) > 0.0f))
        {
            r_0 = r_0 + u32(1);
            continue;
        }
        var o_0 : vec3<f32>;
        if(((ref_0 & (u32(2147483648)))) != u32(0))
        {
            o_0 = b_1.o1_0.xyz;
        }
        else
        {
            o_0 = b_1.o0_0.xyz;
        }
        var _S11 : f32 = b_1.s_0;
        var s2_0 : f32 = _S11 * _S11;
        var _S12 : vec3<f32> = vec3<f32>(s2_0);
        var _S13 : f32 = o_0.x;
        var _S14 : f32 = o_0.y;
        var _S15 : f32 = o_0.z;
        var offd_1 : vec3<f32> = offd_0 + _S12 * (vec3<f32>(0) - vec3<f32>(_S13 * _S14, _S13 * _S15, _S14 * _S15));
        var ks_1 : vec3<f32> = ks_0 + _S12 * o_0;
        var cs_1 : f32 = cs_0 + s2_0;
        diag_0 = diag_0 + _S12 * (vec3<f32>((1.0f + dot(o_0, o_0))) - o_0 * o_0);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_0 = r_0 + u32(1);
    }
    var _S16 : f32 = n_2.w_ang_0;
    var da_0 : f32 = _S16 * _S16;
    var c_0 : f32 = n_2.w_lin_0 * n_2.w_lin_0 * cs_0;
    var k_1 : vec3<f32> = ks_0 * vec3<f32>((n_2.w_ang_0 * n_2.w_lin_0));
    var a0_0 : vec4<f32> = vec4<f32>(diag_0 * vec3<f32>(da_0), offd_0.x * da_0);
    var _S17 : f32 = offd_0.y * da_0;
    var _S18 : f32 = offd_0.z * da_0;
    var a1_0 : vec4<f32> = vec4<f32>(_S17, _S18, c_0, 0.0f);
    var _S19 : u32 = ns_0(node_0, u32(18));
    scratch_0[_S19] = a0_0;
    scratch_0[_S19 + u32(1)] = a1_0;
    scratch_0[_S19 + u32(2)] = vec4<f32>(k_1, 0.0f);
    if(!(c_0 > 0.0f))
    {
        return false;
    }
    var ic_0 : f32 = 1.0f / c_0;
    var kk_0 : f32 = dot(k_1, k_1);
    var _S20 : f32 = k_1.x;
    var sxx_0 : f32 = a0_0.x + (_S20 * _S20 - kk_0) * ic_0;
    var _S21 : f32 = k_1.y;
    var syy_0 : f32 = a0_0.y + (_S21 * _S21 - kk_0) * ic_0;
    var _S22 : f32 = k_1.z;
    var szz_0 : f32 = a0_0.z + (_S22 * _S22 - kk_0) * ic_0;
    var sxy_0 : f32 = a0_0.w + _S20 * _S21 * ic_0;
    var sxz_0 : f32 = _S17 + _S20 * _S22 * ic_0;
    var syz_0 : f32 = _S18 + _S21 * _S22 * ic_0;
    var cxx_0 : f32 = syy_0 * szz_0 - syz_0 * syz_0;
    var cxy_0 : f32 = sxz_0 * syz_0 - sxy_0 * szz_0;
    var cxz_0 : f32 = sxy_0 * syz_0 - sxz_0 * syy_0;
    var det_0 : f32 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    var _S23 : bool;
    if(!(det_0 > 0.0f))
    {
        _S23 = true;
    }
    else
    {
        _S23 = !isfinite_0(det_0);
    }
    if(_S23)
    {
        return false;
    }
    var id_0 : f32 = 1.0f / det_0;
    var cyz_0 : f32 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    var _S24 : u32 = ns_0(node_0, u32(21));
    scratch_0[_S24] = vec4<f32>(cxx_0 * id_0, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_0, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_0, cxy_0 * id_0);
    scratch_0[_S24 + u32(1)] = vec4<f32>(cxz_0 * id_0, cyz_0 * id_0, ic_0, 0.0f);
    scratch_0[_S24 + u32(2)] = vec4<f32>(k_1 * vec3<f32>(ic_0), 0.0f);
    return true;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

var<workgroup> g_centre_0 : vec3<f32>;

fn rel_0( n_3 : u32) -> vec3<f32>
{
    return positions_0[n_3].xyz - g_centre_0;
}

var<workgroup> g_gram_0 : array<array<f32, i32(6)>, i32(6)>;

fn build_modes_0( tid_1 : u32,  n0_1 : u32,  n1_1 : u32)
{
    var kk_1 : u32;
    var l_1 : u32;
    var _S25 : vec4<f32> = vec4<f32>(0.0f);
    var _S26 : u32 = n0_1 + tid_1;
    var p_0 : u32 = _S26;
    var cw_0 : vec4<f32> = _S25;
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
        var w_0 : f32 = 1.0f / (nodes_0[i_0].w_lin_0 * nodes_0[i_0].w_lin_0);
        var cw_1 : vec4<f32> = cw_0 + vec4<f32>(positions_0[i_0].xyz * vec3<f32>(w_0), w_0);
        p_0 = p_0 + u32(1024);
        cw_0 = cw_1;
    }
    var c0_0 : vec2<f32> = blast_group_sum_0(tid_1, cw_0.xy);
    var c1_0 : vec2<f32> = blast_group_sum_0(tid_1, cw_0.zw);
    var _S27 : bool = tid_1 == u32(0);
    if(_S27)
    {
        g_centre_0 = vec3<f32>(c0_0.x, c0_0.y, c1_0.x) / vec3<f32>(c1_0.y);
    }
    workgroupBarrier();
    var rr_0 : array<f32, i32(6)>;
    var rt_0 : array<vec3<f32>, i32(3)>;
    var z_0 : u32 = u32(0);
    loop
    {
        if(z_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        rr_0[z_0] = 0.0f;
        z_0 = z_0 + u32(1);
    }
    z_0 = u32(0);
    loop
    {
        if(z_0 < u32(3))
        {
        }
        else
        {
            break;
        }
        rt_0[z_0] = vec3<f32>(0.0f);
        z_0 = z_0 + u32(1);
    }
    p_0 = _S26;
    var tt_0 : f32 = 0.0f;
    loop
    {
        if(p_0 < n1_1)
        {
        }
        else
        {
            break;
        }
        var i_1 : u32 = lists_0[p_0];
        var n_4 : BlastNode_std430_0 = nodes_0[i_1];
        var _S28 : f32 = n_4.w_ang_0;
        var wa_0 : f32 = 1.0f / (_S28 * _S28);
        var _S29 : f32 = n_4.w_lin_0;
        var wl_0 : f32 = 1.0f / (_S29 * _S29);
        var q_0 : vec3<f32> = rel_0(i_1);
        var _S30 : vec3<f32> = cross(q_0, vec3<f32>(1.0f, 0.0f, 0.0f));
        var _S31 : vec3<f32> = cross(q_0, vec3<f32>(0.0f, 1.0f, 0.0f));
        var _S32 : vec3<f32> = cross(q_0, vec3<f32>(0.0f, 0.0f, 1.0f));
        var c_1 : array<vec3<f32>, i32(3)> = array<vec3<f32>, i32(3)>( _S30, _S31, _S32 );
        rr_0[i32(0)] = rr_0[i32(0)] + (wa_0 + wl_0 * dot(_S30, _S30));
        rr_0[i32(1)] = rr_0[i32(1)] + (wa_0 + wl_0 * dot(_S31, _S31));
        rr_0[i32(2)] = rr_0[i32(2)] + (wa_0 + wl_0 * dot(_S32, _S32));
        rr_0[i32(3)] = rr_0[i32(3)] + wl_0 * dot(_S30, _S31);
        rr_0[i32(4)] = rr_0[i32(4)] + wl_0 * dot(_S30, _S32);
        rr_0[i32(5)] = rr_0[i32(5)] + wl_0 * dot(_S31, _S32);
        kk_1 = u32(0);
        loop
        {
            if(kk_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            rt_0[kk_1] = rt_0[kk_1] + vec3<f32>(wl_0) * c_1[kk_1];
            kk_1 = kk_1 + u32(1);
        }
        var tt_1 : f32 = tt_0 + wl_0;
        p_0 = p_0 + u32(1024);
        tt_0 = tt_1;
    }
    var s0_0 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(rr_0[i32(0)], rr_0[i32(1)]));
    var s1_0 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(rr_0[i32(2)], rr_0[i32(3)]));
    var s2_1 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(rr_0[i32(4)], rr_0[i32(5)]));
    var s3_0 : vec2<f32> = blast_group_sum_0(tid_1, rt_0[i32(0)].xy);
    var s4_0 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(rt_0[i32(0)].z, rt_0[i32(1)].x));
    var s5_0 : vec2<f32> = blast_group_sum_0(tid_1, rt_0[i32(1)].yz);
    var s6_0 : vec2<f32> = blast_group_sum_0(tid_1, rt_0[i32(2)].xy);
    var s7_0 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(rt_0[i32(2)].z, tt_0));
    if(_S27)
    {
        var m_0 : array<array<f32, i32(6)>, i32(6)>;
        m_0[i32(0)][i32(0)] = s0_0.x;
        m_0[i32(1)][i32(1)] = s0_0.y;
        m_0[i32(2)][i32(2)] = s1_0.x;
        m_0[i32(0)][i32(1)] = s1_0.y;
        var _S33 : f32 = s2_1.x;
        m_0[i32(0)][i32(2)] = _S33;
        m_0[i32(1)][i32(2)] = s2_1.y;
        m_0[i32(1)][i32(0)] = m_0[i32(0)][i32(1)];
        m_0[i32(2)][i32(0)] = _S33;
        m_0[i32(2)][i32(1)] = m_0[i32(1)][i32(2)];
        var _S34 : vec3<f32> = vec3<f32>(s3_0.x, s3_0.y, s4_0.x);
        var _S35 : vec3<f32> = vec3<f32>(s4_0.y, s5_0.x, s5_0.y);
        var _S36 : vec3<f32> = vec3<f32>(s6_0.x, s6_0.y, s7_0.x);
        kk_1 = u32(0);
        loop
        {
            if(kk_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S37 : u32 = u32(3) + kk_1;
            m_0[i32(0)][_S37] = _S34[kk_1];
            m_0[i32(1)][_S37] = _S35[kk_1];
            m_0[i32(2)][_S37] = _S36[kk_1];
            m_0[_S37][i32(0)] = _S34[kk_1];
            m_0[_S37][i32(1)] = _S35[kk_1];
            m_0[_S37][i32(2)] = _S36[kk_1];
            l_1 = u32(0);
            loop
            {
                if(l_1 < u32(3))
                {
                }
                else
                {
                    break;
                }
                if(kk_1 == l_1)
                {
                    tt_0 = s7_0.y;
                }
                else
                {
                    tt_0 = 0.0f;
                }
                m_0[_S37][u32(3) + l_1] = tt_0;
                l_1 = l_1 + u32(1);
            }
            kk_1 = kk_1 + u32(1);
        }
        l_1 = u32(0);
        loop
        {
            if(l_1 < u32(6))
            {
            }
            else
            {
                break;
            }
            var c2_0 : u32 = u32(0);
            loop
            {
                if(c2_0 < u32(6))
                {
                }
                else
                {
                    break;
                }
                g_gram_0[l_1][c2_0] = m_0[l_1][c2_0];
                c2_0 = c2_0 + u32(1);
            }
            l_1 = l_1 + u32(1);
        }
    }
    workgroupBarrier();
    return;
}

fn bs_0( j_0 : u32) -> u32
{
    return params_0.node_count_0 * u32(24) + u32(2) * j_0;
}

fn va_0( n_5 : u32,  v_3 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_5, v_3)].xyz;
}

fn vl_0( n_6 : u32,  v_4 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_6, v_4) + u32(1)].xyz;
}

fn bond_term_0( b_2 : ptr<function, BlastBond_std430_0>,  v_5 : u32,  ta_0 : ptr<function, vec3<f32>>,  tl_0 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_2).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_2).n1_0];
    var a0_1 : vec3<f32> = va_0((*b_2).n0_0, v_5) * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = vl_0((*b_2).n0_0, v_5) * vec3<f32>(m0_0.w_lin_0);
    var a1_1 : vec3<f32> = va_0((*b_2).n1_0, v_5) * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = vl_0((*b_2).n1_0, v_5) * vec3<f32>(m1_0.w_lin_0);
    var _S38 : vec3<f32> = vec3<f32>((*b_2).s_0);
    (*ta_0) = (a0_1 - a1_1) * _S38;
    (*tl_0) = (l0_0 - l1_0 + (cross((*b_2).o0_0.xyz, a0_1) - cross((*b_2).o1_0.xyz, a1_1))) * _S38;
    return;
}

fn bond_term_1( b_3 : ptr<function, BlastBond_std430_0>,  v_6 : u32,  ta_1 : ptr<function, vec3<f32>>,  tl_1 : ptr<function, vec3<f32>>)
{
    var m0_1 : BlastNode_std430_0 = nodes_0[(*b_3).n0_0];
    var m1_1 : BlastNode_std430_0 = nodes_0[(*b_3).n1_0];
    var a0_2 : vec3<f32> = va_0((*b_3).n0_0, v_6) * vec3<f32>(m0_1.w_ang_0);
    var l0_1 : vec3<f32> = vl_0((*b_3).n0_0, v_6) * vec3<f32>(m0_1.w_lin_0);
    var a1_2 : vec3<f32> = va_0((*b_3).n1_0, v_6) * vec3<f32>(m1_1.w_ang_0);
    var l1_1 : vec3<f32> = vl_0((*b_3).n1_0, v_6) * vec3<f32>(m1_1.w_lin_0);
    var _S39 : vec3<f32> = vec3<f32>((*b_3).s_0);
    (*ta_1) = (a0_2 - a1_2) * _S39;
    (*tl_1) = (l0_1 - l1_1 + (cross((*b_3).o0_0.xyz, a0_2) - cross((*b_3).o1_0.xyz, a1_2))) * _S39;
    return;
}

fn gather_0( node_1 : u32,  from_t_0 : bool,  with_u_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_7 : BlastNode_std430_0 = nodes_0[node_1];
    var _S40 : vec3<f32> = vec3<f32>(0.0f);
    var r_1 : u32 = n_7.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S40;
    var acc_l_0 : vec3<f32> = _S40;
    loop
    {
        if(r_1 < (n_7.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_1 : u32 = refs_0[r_1];
        var j_1 : u32 = (ref_1 & (u32(2147483647)));
        var _S41 : BlastBond_std430_0 = bonds_0[j_1];
        if(!((_S41.health_0) > 0.0f))
        {
            r_1 = r_1 + u32(1);
            continue;
        }
        var ya_0 : vec3<f32>;
        var yl_0 : vec3<f32>;
        if(from_t_0)
        {
            var _S42 : u32 = bs_0(j_1);
            var _S43 : vec3<f32> = scratch_0[_S42 + u32(1)].xyz;
            ya_0 = scratch_0[_S42].xyz;
            yl_0 = _S43;
        }
        else
        {
            var _S44 : u32 = j_1 * u32(2);
            var ya_1 : vec3<f32> = x_0[_S44].xyz;
            var yl_1 : vec3<f32> = x_0[_S44 + u32(1)].xyz;
            if(with_u_0)
            {
                var ta_2 : vec3<f32>;
                var tl_2 : vec3<f32>;
                bond_term_1(&(_S41), u32(6), &(ta_2), &(tl_2));
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
        var _S45 : vec3<f32> = vec3<f32>(_S41.s_0);
        var ya_2 : vec3<f32> = ya_0 * _S45;
        var yl_3 : vec3<f32> = yl_0 * _S45;
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_1 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross(_S41.o1_0.xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross(_S41.o0_0.xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_1 = r_1 + u32(1);
    }
    (*ang_0) = acc_a_0 * vec3<f32>(n_7.w_ang_0);
    (*lin_0) = acc_l_0 * vec3<f32>(n_7.w_lin_0);
    return;
}

fn rebuild_0( tid_2 : u32,  n0_2 : u32,  n1_2 : u32,  with_u_1 : bool) -> f32
{
    var p_1 : u32 = n0_2 + tid_2;
    var s_2 : f32 = 0.0f;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        var i_2 : u32 = lists_0[p_1];
        var a_2 : vec3<f32>;
        var l_2 : vec3<f32>;
        gather_0(i_2, false, with_u_1, &(a_2), &(l_2));
        var ra_0 : vec3<f32> = va_0(i_2, u32(14)) - a_2;
        var rl_0 : vec3<f32> = vl_0(i_2, u32(14)) - l_2;
        set_v_0(i_2, u32(0), ra_0, rl_0);
        var s_3 : f32 = s_2 + norm2_0(ra_0, rl_0);
        p_1 = p_1 + u32(1024);
        s_2 = s_3;
    }
    device_group_barrier_0();
    var _S46 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_2, 0.0f));
    return _S46.x;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

var<workgroup> g_rhs_0 : array<f32, i32(6)>;

var<workgroup> g_coef_0 : array<f32, i32(6)>;

fn solve_gram_0()
{
    var c_2 : u32;
    var p_2 : u32;
    var a_3 : array<array<f32, i32(6)>, i32(6)>;
    var b_4 : array<f32, i32(6)>;
    var r_2 : u32 = u32(0);
    loop
    {
        if(r_2 < u32(6))
        {
        }
        else
        {
            break;
        }
        b_4[r_2] = g_rhs_0[r_2];
        c_2 = u32(0);
        loop
        {
            if(c_2 < u32(6))
            {
            }
            else
            {
                break;
            }
            a_3[r_2][c_2] = g_gram_0[r_2][c_2];
            c_2 = c_2 + u32(1);
        }
        r_2 = r_2 + u32(1);
    }
    c_2 = u32(0);
    loop
    {
        if(c_2 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S47 : u32 = c_2 + u32(1);
        p_2 = c_2;
        r_2 = _S47;
        loop
        {
            if(r_2 < u32(6))
            {
            }
            else
            {
                break;
            }
            if((abs(a_3[r_2][c_2])) > (abs(a_3[p_2][c_2])))
            {
                p_2 = r_2;
            }
            r_2 = r_2 + u32(1);
        }
        var k_2 : u32 = u32(0);
        loop
        {
            if(k_2 < u32(6))
            {
            }
            else
            {
                break;
            }
            var t_0 : f32 = a_3[c_2][k_2];
            a_3[c_2][k_2] = a_3[p_2][k_2];
            a_3[p_2][k_2] = t_0;
            k_2 = k_2 + u32(1);
        }
        var tb_0 : f32 = b_4[c_2];
        b_4[c_2] = b_4[p_2];
        b_4[p_2] = tb_0;
        var piv_0 : f32 = a_3[c_2][c_2];
        var z_1 : u32;
        if(!((abs(a_3[c_2][c_2])) > 0.0f))
        {
            z_1 = u32(0);
            loop
            {
                if(z_1 < u32(6))
                {
                }
                else
                {
                    break;
                }
                g_coef_0[z_1] = 0.0f;
                z_1 = z_1 + u32(1);
            }
            return;
        }
        z_1 = _S47;
        loop
        {
            if(z_1 < u32(6))
            {
            }
            else
            {
                break;
            }
            var f_0 : f32 = a_3[z_1][c_2] / piv_0;
            var k_3 : u32 = c_2;
            loop
            {
                if(k_3 < u32(6))
                {
                }
                else
                {
                    break;
                }
                a_3[z_1][k_3] = a_3[z_1][k_3] - f_0 * a_3[c_2][k_3];
                k_3 = k_3 + u32(1);
            }
            b_4[z_1] = b_4[z_1] - f_0 * b_4[c_2];
            z_1 = z_1 + u32(1);
        }
        c_2 = _S47;
    }
    var r_3 : i32 = i32(5);
    loop
    {
        if(r_3 >= i32(0))
        {
        }
        else
        {
            break;
        }
        p_2 = u32(r_3) + u32(1);
        var s_4 : f32 = b_4[r_3];
        loop
        {
            if(p_2 < u32(6))
            {
            }
            else
            {
                break;
            }
            var s_5 : f32 = s_4 - a_3[r_3][p_2] * g_coef_0[p_2];
            p_2 = p_2 + u32(1);
            s_4 = s_5;
        }
        g_coef_0[r_3] = s_4 / a_3[r_3][r_3];
        r_3 = r_3 - i32(1);
    }
    return;
}

fn project_0( tid_3 : u32,  n0_3 : u32,  n1_3 : u32,  v_7 : u32,  free_0 : bool)
{
    if(!free_0)
    {
        return;
    }
    var _S48 : vec3<f32> = vec3<f32>(0.0f);
    var _S49 : u32 = n0_3 + tid_3;
    var p_3 : u32 = _S49;
    var sr_0 : vec3<f32> = _S48;
    var st_0 : vec3<f32> = _S48;
    loop
    {
        if(p_3 < n1_3)
        {
        }
        else
        {
            break;
        }
        var i_3 : u32 = lists_0[p_3];
        var n_8 : BlastNode_std430_0 = nodes_0[i_3];
        var lin_1 : vec3<f32> = vl_0(i_3, v_7) / vec3<f32>(n_8.w_lin_0);
        var sr_1 : vec3<f32> = sr_0 + (va_0(i_3, v_7) / vec3<f32>(n_8.w_ang_0) - cross(rel_0(i_3), lin_1));
        var st_1 : vec3<f32> = st_0 + lin_1;
        p_3 = p_3 + u32(1024);
        sr_0 = sr_1;
        st_0 = st_1;
    }
    var a_4 : vec2<f32> = blast_group_sum_0(tid_3, sr_0.xy);
    var b_5 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(sr_0.z, st_0.x));
    var c_3 : vec2<f32> = blast_group_sum_0(tid_3, st_0.yz);
    if(tid_3 == u32(0))
    {
        g_rhs_0[i32(0)] = a_4.x;
        g_rhs_0[i32(1)] = a_4.y;
        g_rhs_0[i32(2)] = b_5.x;
        g_rhs_0[i32(3)] = b_5.y;
        g_rhs_0[i32(4)] = c_3.x;
        g_rhs_0[i32(5)] = c_3.y;
        solve_gram_0();
    }
    workgroupBarrier();
    var _S50 : vec3<f32> = vec3<f32>(g_coef_0[i32(0)], g_coef_0[i32(1)], g_coef_0[i32(2)]);
    var _S51 : vec3<f32> = vec3<f32>(g_coef_0[i32(3)], g_coef_0[i32(4)], g_coef_0[i32(5)]);
    p_3 = _S49;
    loop
    {
        if(p_3 < n1_3)
        {
        }
        else
        {
            break;
        }
        var i_4 : u32 = lists_0[p_3];
        var n_9 : BlastNode_std430_0 = nodes_0[i_4];
        set_v_0(i_4, v_7, va_0(i_4, v_7) - _S50 / vec3<f32>(n_9.w_ang_0), vl_0(i_4, v_7) - (_S51 + cross(rel_0(i_4), _S50)) / vec3<f32>(n_9.w_lin_0));
        p_3 = p_3 + u32(1024);
    }
    device_group_barrier_0();
    return;
}

fn bond_pass_0( tid_4 : u32,  b0_0 : u32,  b1_0 : u32,  v_8 : u32) -> f32
{
    var q_1 : u32 = b0_0 + tid_4;
    var z_2 : f32 = 0.0f;
    loop
    {
        if(q_1 < b1_0)
        {
        }
        else
        {
            break;
        }
        var j_2 : u32 = lists_0[q_1];
        var _S52 : BlastBond_std430_0 = bonds_0[j_2];
        var _S53 : vec3<f32> = vec3<f32>(0.0f);
        var ta_3 : vec3<f32> = _S53;
        var tl_3 : vec3<f32> = _S53;
        if((_S52.health_0) > 0.0f)
        {
            bond_term_0(&(_S52), v_8, &(ta_3), &(tl_3));
        }
        var z_3 : f32 = z_2 + norm2_0(ta_3, tl_3);
        var _S54 : u32 = bs_0(j_2);
        scratch_0[_S54] = vec4<f32>(ta_3, 0.0f);
        scratch_0[_S54 + u32(1)] = vec4<f32>(tl_3, 0.0f);
        q_1 = q_1 + u32(1024);
        z_2 = z_3;
    }
    var _S55 : vec2<f32> = blast_group_sum_0(tid_4, vec2<f32>(z_2, 0.0f));
    return _S55.x;
}

fn residual_norm_0( tid_5 : u32,  n0_4 : u32,  n1_4 : u32) -> f32
{
    var p_4 : u32 = n0_4 + tid_5;
    var s_6 : f32 = 0.0f;
    loop
    {
        if(p_4 < n1_4)
        {
        }
        else
        {
            break;
        }
        var i_5 : u32 = lists_0[p_4];
        var s_7 : f32 = s_6 + norm2_0(va_0(i_5, u32(0)), vl_0(i_5, u32(0)));
        p_4 = p_4 + u32(1024);
        s_6 = s_7;
    }
    var _S56 : vec2<f32> = blast_group_sum_0(tid_5, vec2<f32>(s_6, 0.0f));
    return _S56.x;
}

fn sym_mul_0( s0_1 : vec4<f32>,  s1_1 : vec4<f32>,  v_9 : vec3<f32>) -> vec3<f32>
{
    var _S57 : f32 = v_9.x;
    var _S58 : f32 = s0_1.w;
    var _S59 : f32 = v_9.y;
    var _S60 : f32 = s1_1.x;
    var _S61 : f32 = v_9.z;
    var _S62 : f32 = s1_1.y;
    return vec3<f32>(s0_1.x * _S57 + _S58 * _S59 + _S60 * _S61, _S58 * _S57 + s0_1.y * _S59 + _S62 * _S61, _S60 * _S57 + _S62 * _S59 + s0_1.z * _S61);
}

fn inverse_apply_0( n_10 : u32,  a_5 : vec3<f32>,  l_3 : vec3<f32>,  oa_0 : ptr<function, vec3<f32>>,  ol_0 : ptr<function, vec3<f32>>)
{
    var _S63 : u32 = ns_0(n_10, u32(21));
    var s1_2 : vec4<f32> = scratch_0[_S63 + u32(1)];
    var kc_0 : vec3<f32> = scratch_0[_S63 + u32(2)].xyz;
    var _S64 : vec3<f32> = sym_mul_0(scratch_0[_S63], scratch_0[_S63 + u32(1)], a_5 + cross(kc_0, l_3));
    (*oa_0) = _S64;
    (*ol_0) = l_3 * vec3<f32>(s1_2.z) - cross(kc_0, _S64);
    return;
}

fn node_pass_0( tid_6 : u32,  n0_5 : u32,  n1_5 : u32,  out_v_0 : u32) -> f32
{
    var p_5 : u32 = n0_5 + tid_6;
    var s_8 : f32 = 0.0f;
    loop
    {
        if(p_5 < n1_5)
        {
        }
        else
        {
            break;
        }
        var i_6 : u32 = lists_0[p_5];
        var a_6 : vec3<f32>;
        var l_4 : vec3<f32>;
        gather_0(i_6, true, false, &(a_6), &(l_4));
        set_v_0(i_6, out_v_0, a_6, l_4);
        var s_9 : f32 = s_8 + norm2_0(a_6, l_4);
        p_5 = p_5 + u32(1024);
        s_8 = s_9;
    }
    return s_8;
}

fn block_apply_0( n_11 : u32,  a_7 : vec3<f32>,  l_5 : vec3<f32>,  oa_1 : ptr<function, vec3<f32>>,  ol_1 : ptr<function, vec3<f32>>)
{
    var _S65 : u32 = ns_0(n_11, u32(18));
    var b1_1 : vec4<f32> = scratch_0[_S65 + u32(1)];
    var k_4 : vec3<f32> = scratch_0[_S65 + u32(2)].xyz;
    (*oa_1) = sym_mul_0(scratch_0[_S65], scratch_0[_S65 + u32(1)], a_7) - cross(k_4, l_5);
    (*ol_1) = cross(k_4, a_7) + vec3<f32>(b1_1.z) * l_5;
    return;
}

var<workgroup> g_max_0 : array<f32, i32(1024)>;

var<workgroup> g_max_out_0 : f32;

fn group_max_0( tid_7 : u32,  v_10 : f32) -> f32
{
    g_max_0[tid_7] = v_10;
    workgroupBarrier();
    var k_5 : u32 = u32(512);
    loop
    {
        if(k_5 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_7 < k_5)
        {
            g_max_0[tid_7] = max(g_max_0[tid_7], g_max_0[tid_7 + k_5]);
        }
        workgroupBarrier();
        k_5 = (k_5 >> (u32(1)));
    }
    if(tid_7 == u32(0))
    {
        g_max_out_0 = g_max_0[i32(0)];
    }
    var _S66 : f32 = (workgroupUniformLoad(&((g_max_out_0))));
    workgroupBarrier();
    return _S66;
}

@compute
@workgroup_size(1024, 1, 1)
fn blast_native(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var anchor_0 : f32;
    var force_start_0 : f32;
    var failed_0 : bool;
    var ok_0 : bool;
    var tid_8 : u32 = thread_1.x;
    var k_6 : u32 = group_0.x;
    if(k_6 >= (params_0.island_count_0))
    {
        return;
    }
    var range_0 : vec4<u32> = islands_0[k_6];
    var n0_6 : u32 = range_0.x;
    var n1_6 : u32 = range_0.y;
    var b0_1 : u32 = range_0.z;
    var _S67 : u32 = range_0.w;
    var last_0 : vec4<u32> = report_0[k_6];
    var last_flags_0 : u32 = group_uniform_0(report_0[k_6].y, tid_8);
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
    var p_6 : u32;
    if(converged_0)
    {
        var _S68 : u32 = n0_6 + tid_8;
        singular_0 = 0.0f;
        p_6 = _S68;
        loop
        {
            if(p_6 < n1_6)
            {
            }
            else
            {
                break;
            }
            var i_7 : u32 = lists_0[p_6];
            var _S69 : u32 = i_7 * u32(2);
            var _S70 : u32 = ns_0(i_7, u32(16));
            if((any((accel_0[_S69] != scratch_0[_S70]))))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = (any((accel_0[_S69 + u32(1)] != scratch_0[_S70 + u32(1)])));
            }
            if(converged_0)
            {
                singular_0 = 1.0f;
            }
            p_6 = p_6 + u32(1024);
        }
        var _S71 : vec2<f32> = blast_group_sum_0(tid_8, vec2<f32>(singular_0, 0.0f));
        if((_S71.x) == 0.0f)
        {
            if(tid_8 == u32(0))
            {
                report_0[k_6] = vec4<u32>(u32(0), u32(7), last_0.z, last_0.w);
            }
            return;
        }
    }
    var _S72 : u32 = n0_6 + tid_8;
    singular_0 = 0.0f;
    p_6 = _S72;
    var target_0 : f32 = 0.0f;
    loop
    {
        if(p_6 < n1_6)
        {
        }
        else
        {
            break;
        }
        var i_8 : u32 = lists_0[p_6];
        var n_12 : BlastNode_std430_0 = nodes_0[i_8];
        var _S73 : u32 = i_8 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S73];
        var a_lin_0 : vec4<f32> = accel_0[_S73 + u32(1)];
        var _S74 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz);
        if((n_12.w_ang_0) > 0.0f)
        {
            anchor_0 = 1.0f / n_12.w_ang_0;
        }
        else
        {
            anchor_0 = 1.0f;
        }
        var ba_0 : vec3<f32> = _S74 * vec3<f32>(anchor_0);
        var _S75 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz);
        if((n_12.w_lin_0) > 0.0f)
        {
            force_start_0 = params_0.recip_length_0 / n_12.w_lin_0;
        }
        else
        {
            force_start_0 = params_0.recip_length_0;
        }
        var bl_0 : vec3<f32> = _S75 * vec3<f32>(force_start_0);
        var target_1 : f32 = target_0 + norm2_0(ba_0, bl_0);
        set_v_0(i_8, u32(14), ba_0, bl_0);
        set_v_0(i_8, u32(0), ba_0, bl_0);
        var _S76 : u32 = ns_0(i_8, u32(16));
        scratch_0[_S76] = a_ang_0;
        scratch_0[_S76 + u32(1)] = a_lin_0;
        var _S77 : vec3<f32> = vec3<f32>(0.0f);
        set_v_0(i_8, u32(6), _S77, _S77);
        set_v_0(i_8, u32(2), _S77, _S77);
        var _S78 : bool = build_block_0(i_8);
        if(!_S78)
        {
            singular_0 = 1.0f;
        }
        p_6 = p_6 + u32(1024);
        target_0 = target_1;
    }
    var _S79 : u32 = b0_1 + tid_8;
    anchor_0 = 0.0f;
    var q_2 : u32 = _S79;
    loop
    {
        if(q_2 < _S67)
        {
        }
        else
        {
            break;
        }
        var b_6 : BlastBond_std430_0 = bonds_0[lists_0[q_2]];
        if((bonds_0[lists_0[q_2]].health_0) > 0.0f)
        {
            if(!((nodes_0[b_6.n0_0].w_lin_0) > 0.0f))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = !((nodes_0[b_6.n1_0].w_lin_0) > 0.0f);
            }
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
            anchor_0 = 1.0f;
        }
        q_2 = q_2 + u32(1024);
    }
    if(!warm_1)
    {
        q_2 = _S79;
        loop
        {
            if(q_2 < _S67)
            {
            }
            else
            {
                break;
            }
            var _S80 : u32 = lists_0[q_2] * u32(2);
            var _S81 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S80] = _S81;
            x_0[_S80 + u32(1)] = _S81;
            q_2 = q_2 + u32(1024);
        }
    }
    var tsum_0 : vec2<f32> = blast_group_sum_0(tid_8, vec2<f32>(target_0, singular_0));
    var delta2_0 : f32 = tsum_0.x * params_0.tolerance_0 * params_0.tolerance_0;
    var _S82 : bool = (tsum_0.y) > 0.0f;
    var _S83 : vec2<f32> = blast_group_sum_0(tid_8, vec2<f32>(anchor_0, 0.0f));
    var free_1 : bool = (_S83.x) == 0.0f;
    device_group_barrier_0();
    if(!(delta2_0 > 0.0f))
    {
        q_2 = _S79;
        loop
        {
            if(q_2 < _S67)
            {
            }
            else
            {
                break;
            }
            var _S84 : u32 = lists_0[q_2] * u32(2);
            var _S85 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S84] = _S85;
            x_0[_S84 + u32(1)] = _S85;
            q_2 = q_2 + u32(1024);
        }
        if(tid_8 == u32(0))
        {
            if(warm_1)
            {
                p_6 = u32(4);
            }
            else
            {
                p_6 = u32(0);
            }
            report_0[k_6] = vec4<u32>(u32(0), (u32(1) | (p_6)), (bitcast<u32>((0.0f))), (bitcast<u32>((delta2_0))));
        }
        return;
    }
    if(free_1)
    {
        build_modes_0(tid_8, n0_6, n1_6);
    }
    if(warm_1)
    {
        var _S86 : f32 = rebuild_0(tid_8, n0_6, n1_6, false);
    }
    var _S87 : f32 = float_bits_0(u32(2139095040));
    if((params_0.force_tolerance_0) > 0.0f)
    {
        q_2 = _S79;
        force_start_0 = 0.0f;
        loop
        {
            if(q_2 < _S67)
            {
            }
            else
            {
                break;
            }
            var _S88 : u32 = lists_0[q_2] * u32(2);
            var f_1 : f32 = force_start_0 + norm2_0(x_0[_S88].xyz, x_0[_S88 + u32(1)].xyz);
            q_2 = q_2 + u32(1024);
            force_start_0 = f_1;
        }
        var _S89 : vec2<f32> = blast_group_sum_0(tid_8, vec2<f32>(force_start_0, 0.0f));
        force_start_0 = sqrt(_S89.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    var best_0 : f32 = _S87;
    var best_iteration_0 : u32 = u32(0);
    var step_0 : f32 = _S87;
    var previous_0 : f32 = 0.0f;
    var num_0 : f32 = 0.0f;
    var iteration_0 : u32 = u32(0);
    var travel_0 : f32 = 0.0f;
    loop
    {
        if(!_S82)
        {
            converged_0 = iteration_0 < (params_0.max_iterations_0);
        }
        else
        {
            converged_0 = false;
        }
        if(converged_0)
        {
        }
        else
        {
            converged_0 = false;
            failed_0 = _S82;
            break;
        }
        project_0(tid_8, n0_6, n1_6, u32(0), free_1);
        var num_1 : f32 = bond_pass_0(tid_8, b0_1, _S67, u32(0));
        var _S90 : bool = num_1 <= delta2_0;
        if(_S90)
        {
            var _S91 : f32 = residual_norm_0(tid_8, n0_6, n1_6);
            failed_0 = !(_S91 <= delta2_0);
        }
        else
        {
            failed_0 = false;
        }
        var _S92 : bool = iteration_0 > u32(0);
        if(_S92)
        {
            ok_0 = true;
        }
        else
        {
            ok_0 = warm_1;
        }
        var _S93 : bool;
        if(ok_0)
        {
            _S93 = _S90;
        }
        else
        {
            _S93 = false;
        }
        var _S94 : bool;
        if(_S93)
        {
            _S94 = !failed_0;
        }
        else
        {
            _S94 = false;
        }
        var hold_0 : bool;
        var num_2 : f32;
        var previous_1 : f32;
        if(_S94)
        {
            var _S95 : f32 = rebuild_0(tid_8, n0_6, n1_6, true);
            project_0(tid_8, n0_6, n1_6, u32(0), free_1);
            var num_3 : f32 = bond_pass_0(tid_8, b0_1, _S67, u32(0));
            if(num_3 <= delta2_0)
            {
                var _S96 : f32 = residual_norm_0(tid_8, n0_6, n1_6);
                hold_0 = !(_S96 <= delta2_0);
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
            hold_0 = failed_0;
            previous_1 = previous_0;
        }
        if(num_2 < (best_0 * 0.99000000953674316f))
        {
            best_0 = num_2;
            best_iteration_0 = iteration_0;
        }
        else
        {
            if((iteration_0 - best_iteration_0) >= u32(512))
            {
                converged_0 = false;
                failed_0 = _S82;
                num_0 = num_2;
                break;
            }
        }
        var _S97 : bool;
        if(num_2 <= delta2_0)
        {
            _S97 = !hold_0;
        }
        else
        {
            _S97 = false;
        }
        if(_S97)
        {
            converged_0 = true;
            failed_0 = _S82;
            num_0 = num_2;
            break;
        }
        var _S98 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            _S98 = _S92;
        }
        else
        {
            _S98 = false;
        }
        var _S99 : bool;
        if(_S98)
        {
            _S99 = step_0 <= (params_0.force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S99 = false;
        }
        var mloc_0 : f32;
        if(_S99)
        {
            q_2 = _S79;
            mloc_0 = 0.0f;
            loop
            {
                if(q_2 < _S67)
                {
                }
                else
                {
                    break;
                }
                var j_3 : u32 = lists_0[q_2];
                var _S100 : BlastBond_std430_0 = bonds_0[j_3];
                var ta_4 : vec3<f32>;
                var tl_4 : vec3<f32>;
                bond_term_1(&(_S100), u32(6), &(ta_4), &(tl_4));
                var _S101 : u32 = j_3 * u32(2);
                var f_2 : f32 = mloc_0 + norm2_0(x_0[_S101].xyz + ta_4, x_0[_S101 + u32(1)].xyz + tl_4);
                q_2 = q_2 + u32(1024);
                mloc_0 = f_2;
            }
            var _S102 : f32 = params_0.force_tolerance_0;
            var _S103 : vec2<f32> = blast_group_sum_0(tid_8, vec2<f32>(mloc_0, 0.0f));
            if(step_0 <= (_S102 * sqrt(_S103.x)))
            {
                converged_0 = true;
                failed_0 = _S82;
                num_0 = num_2;
                break;
            }
        }
        device_group_barrier_0();
        var p_7 : u32;
        if(iteration_0 == u32(0))
        {
            p_6 = _S72;
            loop
            {
                if(p_6 < n1_6)
                {
                }
                else
                {
                    break;
                }
                var i_9 : u32 = lists_0[p_6];
                set_v_0(i_9, u32(8), va_0(i_9, u32(0)), vl_0(i_9, u32(0)));
                p_6 = p_6 + u32(1024);
            }
        }
        else
        {
            p_6 = _S72;
            loop
            {
                if(p_6 < n1_6)
                {
                }
                else
                {
                    break;
                }
                var i_10 : u32 = lists_0[p_6];
                var a_8 : vec3<f32>;
                var l_6 : vec3<f32>;
                inverse_apply_0(i_10, va_0(i_10, u32(0)), vl_0(i_10, u32(0)), &(a_8), &(l_6));
                set_v_0(i_10, u32(10), a_8, l_6);
                p_6 = p_6 + u32(1024);
            }
            device_group_barrier_0();
            var _S104 : f32 = bond_pass_0(tid_8, b0_1, _S67, u32(10));
            device_group_barrier_0();
            var _S105 : f32 = node_pass_0(tid_8, n0_6, n1_6, u32(12));
            p_7 = _S72;
            loop
            {
                if(p_7 < n1_6)
                {
                }
                else
                {
                    break;
                }
                var i_11 : u32 = lists_0[p_7];
                var sa_0 : vec3<f32>;
                var sl_0 : vec3<f32>;
                block_apply_0(i_11, va_0(i_11, u32(10)), vl_0(i_11, u32(10)), &(sa_0), &(sl_0));
                var ma_0 : vec3<f32>;
                var ml_0 : vec3<f32>;
                inverse_apply_0(i_11, va_0(i_11, u32(12)) - sa_0, vl_0(i_11, u32(12)) - sl_0, &(ma_0), &(ml_0));
                var _S106 : vec3<f32> = vec3<f32>(1.68946564197540283f);
                var _S107 : vec3<f32> = vec3<f32>(1.5220409631729126f);
                set_v_0(i_11, u32(8), _S106 * va_0(i_11, u32(10)) - _S107 * ma_0, _S106 * vl_0(i_11, u32(10)) - _S107 * ml_0);
                p_7 = p_7 + u32(1024);
            }
        }
        device_group_barrier_0();
        project_0(tid_8, n0_6, n1_6, u32(8), free_1);
        mloc_0 = 0.0f;
        p_6 = _S72;
        loop
        {
            if(p_6 < n1_6)
            {
            }
            else
            {
                break;
            }
            var i_12 : u32 = lists_0[p_6];
            var a_9 : vec3<f32> = abs(va_0(i_12, u32(8)));
            var l_7 : vec3<f32> = abs(vl_0(i_12, u32(8)));
            var _S108 : f32 = max(mloc_0, max(max(max(a_9.x, a_9.y), max(a_9.z, l_7.x)), max(l_7.y, l_7.z)));
            var p_8 : u32 = p_6 + u32(1024);
            mloc_0 = _S108;
            p_6 = p_8;
        }
        var mag_0 : f32 = group_max_0(tid_8, mloc_0);
        var _S109 : bool;
        if(!(mag_0 > 0.0f))
        {
            _S109 = true;
        }
        else
        {
            _S109 = !isfinite_0(mag_0);
        }
        if(_S109)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            break;
        }
        var _S110 : f32 = 1.0f / mag_0;
        p_7 = _S72;
        var gl_0 : f32 = 0.0f;
        loop
        {
            if(p_7 < n1_6)
            {
            }
            else
            {
                break;
            }
            var i_13 : u32 = lists_0[p_7];
            var _S111 : vec3<f32> = vec3<f32>(_S110);
            var ga_0 : vec3<f32> = va_0(i_13, u32(8)) * _S111;
            var gn_0 : vec3<f32> = vl_0(i_13, u32(8)) * _S111;
            set_v_0(i_13, u32(8), ga_0, gn_0);
            var gl_1 : f32 = gl_0 + (dot(va_0(i_13, u32(0)), ga_0) + dot(vl_0(i_13, u32(0)), gn_0));
            p_7 = p_7 + u32(1024);
            gl_0 = gl_1;
        }
        var _S112 : vec2<f32> = blast_group_sum_0(tid_8, vec2<f32>(gl_0, 0.0f));
        var gamma_0 : f32 = _S112.x;
        var _S113 : bool = gamma_0 > 0.0f;
        var _S114 : bool;
        if(!_S113)
        {
            _S114 = true;
        }
        else
        {
            _S114 = !isfinite_0(gamma_0);
        }
        if(_S114)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            break;
        }
        var _S115 : bool;
        if(iteration_0 > u32(1))
        {
            _S115 = previous_1 > 0.0f;
        }
        else
        {
            _S115 = false;
        }
        if(_S115)
        {
            singular_0 = gamma_0 / previous_1;
        }
        else
        {
            singular_0 = 0.0f;
        }
        var p_9 : u32 = _S72;
        loop
        {
            if(p_9 < n1_6)
            {
            }
            else
            {
                break;
            }
            var i_14 : u32 = lists_0[p_9];
            if(singular_0 == 0.0f)
            {
                set_v_0(i_14, u32(2), va_0(i_14, u32(8)), vl_0(i_14, u32(8)));
            }
            else
            {
                set_v_0(i_14, u32(2), va_0(i_14, u32(8)) + va_0(i_14, u32(2)) * vec3<f32>(singular_0), vl_0(i_14, u32(8)) + vl_0(i_14, u32(2)) * vec3<f32>(singular_0));
            }
            p_9 = p_9 + u32(1024);
        }
        device_group_barrier_0();
        var den_0 : f32 = bond_pass_0(tid_8, b0_1, _S67, u32(2));
        device_group_barrier_0();
        var _S116 : f32 = node_pass_0(tid_8, n0_6, n1_6, u32(4));
        var _S117 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            if(den_0 > 0.0f)
            {
                _S117 = _S113;
            }
            else
            {
                _S117 = false;
            }
            var step_1 : f32;
            if(_S117)
            {
                step_1 = gamma_0 / sqrt(den_0);
            }
            else
            {
                step_1 = _S87;
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
            step_0 = step_1;
            travel_0 = travel_1;
        }
        var _S118 : u32 = iteration_0 + u32(1);
        if(!(den_0 > 0.0f))
        {
            _S117 = true;
        }
        else
        {
            _S117 = !isfinite_0(den_0);
        }
        if(_S117)
        {
            converged_0 = false;
            failed_0 = _S82;
            num_0 = num_2;
            iteration_0 = _S118;
            break;
        }
        var _S119 : f32 = gamma_0 / den_0;
        var p_10 : u32 = _S72;
        loop
        {
            if(p_10 < n1_6)
            {
            }
            else
            {
                break;
            }
            var i_15 : u32 = lists_0[p_10];
            var _S120 : vec3<f32> = vec3<f32>(_S119);
            set_v_0(i_15, u32(6), va_0(i_15, u32(6)) + va_0(i_15, u32(2)) * _S120, vl_0(i_15, u32(6)) + vl_0(i_15, u32(2)) * _S120);
            set_v_0(i_15, u32(0), va_0(i_15, u32(0)) - va_0(i_15, u32(4)) * _S120, vl_0(i_15, u32(0)) - vl_0(i_15, u32(4)) * _S120);
            p_10 = p_10 + u32(1024);
        }
        device_group_barrier_0();
        previous_0 = gamma_0;
        num_0 = num_2;
        iteration_0 = _S118;
    }
    device_group_barrier_0();
    q_2 = _S79;
    loop
    {
        if(q_2 < _S67)
        {
        }
        else
        {
            break;
        }
        var j_4 : u32 = lists_0[q_2];
        var _S121 : BlastBond_std430_0 = bonds_0[j_4];
        if(!((_S121.health_0) > 0.0f))
        {
            var _S122 : u32 = j_4 * u32(2);
            var _S123 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S122] = _S123;
            x_0[_S122 + u32(1)] = _S123;
            q_2 = q_2 + u32(1024);
            continue;
        }
        var ta_5 : vec3<f32>;
        var tl_5 : vec3<f32>;
        bond_term_1(&(_S121), u32(6), &(ta_5), &(tl_5));
        var _S124 : u32 = j_4 * u32(2);
        x_0[_S124] = vec4<f32>(x_0[_S124].xyz + ta_5, 0.0f);
        var _S125 : u32 = _S124 + u32(1);
        x_0[_S125] = vec4<f32>(x_0[_S125].xyz + tl_5, 0.0f);
        q_2 = q_2 + u32(1024);
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
        p_6 = u32(1);
    }
    else
    {
        p_6 = u32(0);
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
        q_2 = u32(4);
    }
    else
    {
        q_2 = u32(0);
    }
    var flags_0 : u32 = (p_6 | (q_2));
    if(tid_8 == u32(0))
    {
        report_0[k_6] = vec4<u32>(iteration_0, flags_0, (bitcast<u32>((num_0))), (bitcast<u32>((delta2_0))));
    }
    return;
}

