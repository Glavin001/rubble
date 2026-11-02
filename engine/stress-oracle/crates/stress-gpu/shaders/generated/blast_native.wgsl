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

fn blast_frame_0( node_count_1 : u32,  k_2 : u32) -> u32
{
    return node_count_1 + u32(9) * k_2;
}

fn frame_word_0( k_3 : u32,  w_0 : u32) -> u32
{
    return islands_0[blast_frame_0(params_0.node_count_0, k_3) + w_0 / u32(4)][w_0 % u32(4)];
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_1 : vec4<u32>,  thread_1 : u32) -> vec4<u32>
{
    if(thread_1 == u32(0))
    {
        g_group_uniform4_0 = value_1;
    }
    var _S25 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S25;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn bs_0( j_0 : u32) -> u32
{
    return params_0.node_count_0 * u32(24) + u32(2) * j_0;
}

fn va_0( n_3 : u32,  v_3 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_3, v_3)].xyz;
}

fn vl_0( n_4 : u32,  v_4 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_4, v_4) + u32(1)].xyz;
}

fn bond_term_0( b_2 : ptr<function, BlastBond_std430_0>,  v_5 : u32,  ta_0 : ptr<function, vec3<f32>>,  tl_0 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_2).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_2).n1_0];
    var a0_1 : vec3<f32> = va_0((*b_2).n0_0, v_5) * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = vl_0((*b_2).n0_0, v_5) * vec3<f32>(m0_0.w_lin_0);
    var a1_1 : vec3<f32> = va_0((*b_2).n1_0, v_5) * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = vl_0((*b_2).n1_0, v_5) * vec3<f32>(m1_0.w_lin_0);
    var _S26 : vec3<f32> = vec3<f32>((*b_2).s_0);
    (*ta_0) = (a0_1 - a1_1) * _S26;
    (*tl_0) = (l0_0 - l1_0 + (cross((*b_2).o0_0.xyz, a0_1) - cross((*b_2).o1_0.xyz, a1_1))) * _S26;
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
    var _S27 : vec3<f32> = vec3<f32>((*b_3).s_0);
    (*ta_1) = (a0_2 - a1_2) * _S27;
    (*tl_1) = (l0_1 - l1_1 + (cross((*b_3).o0_0.xyz, a0_2) - cross((*b_3).o1_0.xyz, a1_2))) * _S27;
    return;
}

fn gather_0( node_1 : u32,  from_t_0 : bool,  with_u_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_5 : BlastNode_std430_0 = nodes_0[node_1];
    var _S28 : vec3<f32> = vec3<f32>(0.0f);
    var r_1 : u32 = n_5.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S28;
    var acc_l_0 : vec3<f32> = _S28;
    loop
    {
        if(r_1 < (n_5.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_1 : u32 = refs_0[r_1];
        var j_1 : u32 = (ref_1 & (u32(2147483647)));
        var _S29 : BlastBond_std430_0 = bonds_0[j_1];
        if(!((_S29.health_0) > 0.0f))
        {
            r_1 = r_1 + u32(1);
            continue;
        }
        var ya_0 : vec3<f32>;
        var yl_0 : vec3<f32>;
        if(from_t_0)
        {
            var _S30 : u32 = bs_0(j_1);
            var _S31 : vec3<f32> = scratch_0[_S30 + u32(1)].xyz;
            ya_0 = scratch_0[_S30].xyz;
            yl_0 = _S31;
        }
        else
        {
            var _S32 : u32 = j_1 * u32(2);
            var ya_1 : vec3<f32> = x_0[_S32].xyz;
            var yl_1 : vec3<f32> = x_0[_S32 + u32(1)].xyz;
            if(with_u_0)
            {
                var ta_2 : vec3<f32>;
                var tl_2 : vec3<f32>;
                bond_term_1(&(_S29), u32(6), &(ta_2), &(tl_2));
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
        var _S33 : vec3<f32> = vec3<f32>(_S29.s_0);
        var ya_2 : vec3<f32> = ya_0 * _S33;
        var yl_3 : vec3<f32> = yl_0 * _S33;
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_1 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross(_S29.o1_0.xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross(_S29.o0_0.xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_1 = r_1 + u32(1);
    }
    (*ang_0) = acc_a_0 * vec3<f32>(n_5.w_ang_0);
    (*lin_0) = acc_l_0 * vec3<f32>(n_5.w_lin_0);
    return;
}

fn rebuild_0( tid_1 : u32,  n0_1 : u32,  n1_1 : u32,  with_u_1 : bool) -> f32
{
    var p_0 : u32 = n0_1 + tid_1;
    var s_2 : f32 = 0.0f;
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
        var l_1 : vec3<f32>;
        gather_0(i_0, false, with_u_1, &(a_2), &(l_1));
        var ra_0 : vec3<f32> = va_0(i_0, u32(14)) - a_2;
        var rl_0 : vec3<f32> = vl_0(i_0, u32(14)) - l_1;
        set_v_0(i_0, u32(0), ra_0, rl_0);
        var s_3 : f32 = s_2 + norm2_0(ra_0, rl_0);
        p_0 = p_0 + u32(1024);
        s_2 = s_3;
    }
    device_group_barrier_0();
    var _S34 : vec2<f32> = blast_group_sum_0(tid_1, vec2<f32>(s_2, 0.0f));
    return _S34.x;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn frame_float_0( k_4 : u32,  w_1 : u32) -> f32
{
    return (bitcast<f32>((frame_word_0(k_4, w_1))));
}

fn frame_axis_0( k_5 : u32,  rotations_0 : u32,  kk_1 : u32) -> vec3<f32>
{
    if(rotations_0 == u32(1))
    {
        return vec3<f32>(frame_float_0(k_5, u32(4)), frame_float_0(k_5, u32(5)), frame_float_0(k_5, u32(6)));
    }
    var _S35 : f32;
    if(kk_1 == u32(0))
    {
        _S35 = 1.0f;
    }
    else
    {
        _S35 = 0.0f;
    }
    var _S36 : f32;
    if(kk_1 == u32(1))
    {
        _S36 = 1.0f;
    }
    else
    {
        _S36 = 0.0f;
    }
    var _S37 : f32;
    if(kk_1 == u32(2))
    {
        _S37 = 1.0f;
    }
    else
    {
        _S37 = 0.0f;
    }
    return vec3<f32>(_S35, _S36, _S37);
}

fn frame_terms_0( k_6 : u32,  rotations_1 : u32,  i_1 : u32,  v_7 : u32,  s_4 : ptr<function, array<f32, i32(6)>>)
{
    var n_6 : BlastNode_std430_0 = nodes_0[i_1];
    var lin_1 : vec3<f32> = vl_0(i_1, v_7) / vec3<f32>(n_6.w_lin_0);
    var _S38 : vec3<f32> = va_0(i_1, v_7) / vec3<f32>(n_6.w_ang_0) - cross(n_6.position_0.xyz, lin_1);
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
        (*s_4)[kk_2] = (*s_4)[kk_2] + dot(frame_axis_0(k_6, rotations_1, kk_2), _S38);
        kk_2 = kk_2 + u32(1);
    }
    var j_2 : u32 = u32(0);
    loop
    {
        if(j_2 < u32(3))
        {
        }
        else
        {
            break;
        }
        (*s_4)[rotations_1 + j_2] = (*s_4)[rotations_1 + j_2] + lin_1[j_2];
        j_2 = j_2 + u32(1);
    }
    return;
}

fn frame_solve_0( k_7 : u32,  rotations_2 : u32,  sum_0 : array<f32, i32(6)>,  omega_0 : ptr<function, vec3<f32>>,  t_0 : ptr<function, vec3<f32>>)
{
    var j_3 : u32;
    var d_0 : u32 = rotations_2 + u32(3);
    var c_1 : array<f32, i32(6)>;
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
        c_1[z_0] = 0.0f;
        z_0 = z_0 + u32(1);
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
        var _S39 : f32 = sum_0[row_0] * frame_float_0(k_7, u32(8) + row_0);
        j_3 = u32(0);
        var value_2 : f32 = _S39;
        loop
        {
            if(j_3 < row_0)
            {
            }
            else
            {
                break;
            }
            var value_3 : f32 = value_2 - frame_float_0(k_7, u32(14) + row_0 * (row_0 + u32(1)) / u32(2) + j_3) * c_1[j_3];
            j_3 = j_3 + u32(1);
            value_2 = value_3;
        }
        var _S40 : u32 = row_0 + u32(1);
        c_1[row_0] = value_2 / frame_float_0(k_7, u32(14) + row_0 * _S40 / u32(2) + row_0);
        row_0 = _S40;
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
        var r_2 : u32 = u32(row_1);
        var _S41 : u32 = r_2 + u32(1);
        j_3 = _S41;
        loop
        {
            if(j_3 < d_0)
            {
            }
            else
            {
                break;
            }
            var _S42 : u32 = j_3 + u32(1);
            c_1[r_2] = c_1[r_2] - frame_float_0(k_7, u32(14) + j_3 * _S42 / u32(2) + r_2) * c_1[j_3];
            j_3 = _S42;
        }
        c_1[r_2] = c_1[r_2] / frame_float_0(k_7, u32(14) + r_2 * _S41 / u32(2) + r_2);
        row_1 = row_1 - i32(1);
    }
    z_0 = u32(0);
    loop
    {
        if(z_0 < d_0)
        {
        }
        else
        {
            break;
        }
        c_1[z_0] = c_1[z_0] * frame_float_0(k_7, u32(8) + z_0);
        z_0 = z_0 + u32(1);
    }
    (*omega_0) = vec3<f32>(0.0f);
    j_3 = u32(0);
    loop
    {
        if(j_3 < rotations_2)
        {
        }
        else
        {
            break;
        }
        (*omega_0) = (*omega_0) + frame_axis_0(k_7, rotations_2, j_3) * vec3<f32>(c_1[j_3]);
        j_3 = j_3 + u32(1);
    }
    (*t_0) = vec3<f32>(c_1[rotations_2], c_1[rotations_2 + u32(1)], c_1[rotations_2 + u32(2)]);
    return;
}

var<workgroup> g_omega_0 : vec3<f32>;

var<workgroup> g_t_0 : vec3<f32>;

fn frame_apply_0( i_2 : u32,  v_8 : u32,  omega_1 : vec3<f32>,  t_1 : vec3<f32>)
{
    var n_7 : BlastNode_std430_0 = nodes_0[i_2];
    set_v_0(i_2, v_8, va_0(i_2, v_8) - omega_1 / vec3<f32>(n_7.w_ang_0), vl_0(i_2, v_8) - (t_1 + cross(n_7.position_0.xyz, omega_1)) / vec3<f32>(n_7.w_lin_0));
    return;
}

fn project_0( tid_2 : u32,  k_8 : u32,  rotations_3 : u32,  n0_2 : u32,  n1_2 : u32,  v_9 : u32,  free_0 : bool)
{
    if(!free_0)
    {
        return;
    }
    var s_5 : array<f32, i32(6)>;
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
        s_5[z_1] = 0.0f;
        z_1 = z_1 + u32(1);
    }
    var _S43 : u32 = n0_2 + tid_2;
    var p_1 : u32 = _S43;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_terms_0(k_8, rotations_3, lists_0[p_1], v_9, &(s_5));
        p_1 = p_1 + u32(1024);
    }
    var a_3 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_5[i32(0)], s_5[i32(1)]));
    var b_4 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_5[i32(2)], s_5[i32(3)]));
    var c_2 : vec2<f32> = blast_group_sum_0(tid_2, vec2<f32>(s_5[i32(4)], s_5[i32(5)]));
    if(tid_2 == u32(0))
    {
        var sum_1 : array<f32, i32(6)> = array<f32, i32(6)>( a_3.x, a_3.y, b_4.x, b_4.y, c_2.x, c_2.y );
        var omega_2 : vec3<f32>;
        var t_2 : vec3<f32>;
        frame_solve_0(k_8, rotations_3, sum_1, &(omega_2), &(t_2));
        g_omega_0 = omega_2;
        g_t_0 = t_2;
    }
    workgroupBarrier();
    var _S44 : vec3<f32> = g_omega_0;
    var _S45 : vec3<f32> = g_t_0;
    p_1 = _S43;
    loop
    {
        if(p_1 < n1_2)
        {
        }
        else
        {
            break;
        }
        frame_apply_0(lists_0[p_1], v_9, _S44, _S45);
        p_1 = p_1 + u32(1024);
    }
    device_group_barrier_0();
    return;
}

fn bond_pass_0( tid_3 : u32,  b0_0 : u32,  b1_0 : u32,  v_10 : u32) -> f32
{
    var q_0 : u32 = b0_0 + tid_3;
    var z_2 : f32 = 0.0f;
    loop
    {
        if(q_0 < b1_0)
        {
        }
        else
        {
            break;
        }
        var j_4 : u32 = lists_0[q_0];
        var _S46 : BlastBond_std430_0 = bonds_0[j_4];
        var _S47 : vec3<f32> = vec3<f32>(0.0f);
        var ta_3 : vec3<f32> = _S47;
        var tl_3 : vec3<f32> = _S47;
        if((_S46.health_0) > 0.0f)
        {
            bond_term_0(&(_S46), v_10, &(ta_3), &(tl_3));
        }
        var z_3 : f32 = z_2 + norm2_0(ta_3, tl_3);
        var _S48 : u32 = bs_0(j_4);
        scratch_0[_S48] = vec4<f32>(ta_3, 0.0f);
        scratch_0[_S48 + u32(1)] = vec4<f32>(tl_3, 0.0f);
        q_0 = q_0 + u32(1024);
        z_2 = z_3;
    }
    var _S49 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(z_2, 0.0f));
    return _S49.x;
}

fn residual_norm_0( tid_4 : u32,  n0_3 : u32,  n1_3 : u32) -> f32
{
    var p_2 : u32 = n0_3 + tid_4;
    var s_6 : f32 = 0.0f;
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
        var s_7 : f32 = s_6 + norm2_0(va_0(i_3, u32(0)), vl_0(i_3, u32(0)));
        p_2 = p_2 + u32(1024);
        s_6 = s_7;
    }
    var _S50 : vec2<f32> = blast_group_sum_0(tid_4, vec2<f32>(s_6, 0.0f));
    return _S50.x;
}

fn sym_mul_0( s0_0 : vec4<f32>,  s1_0 : vec4<f32>,  v_11 : vec3<f32>) -> vec3<f32>
{
    var _S51 : f32 = v_11.x;
    var _S52 : f32 = s0_0.w;
    var _S53 : f32 = v_11.y;
    var _S54 : f32 = s1_0.x;
    var _S55 : f32 = v_11.z;
    var _S56 : f32 = s1_0.y;
    return vec3<f32>(s0_0.x * _S51 + _S52 * _S53 + _S54 * _S55, _S52 * _S51 + s0_0.y * _S53 + _S56 * _S55, _S54 * _S51 + _S56 * _S53 + s0_0.z * _S55);
}

fn inverse_apply_0( n_8 : u32,  a_4 : vec3<f32>,  l_2 : vec3<f32>,  oa_0 : ptr<function, vec3<f32>>,  ol_0 : ptr<function, vec3<f32>>)
{
    var _S57 : u32 = ns_0(n_8, u32(21));
    var s1_1 : vec4<f32> = scratch_0[_S57 + u32(1)];
    var kc_0 : vec3<f32> = scratch_0[_S57 + u32(2)].xyz;
    var _S58 : vec3<f32> = sym_mul_0(scratch_0[_S57], scratch_0[_S57 + u32(1)], a_4 + cross(kc_0, l_2));
    (*oa_0) = _S58;
    (*ol_0) = l_2 * vec3<f32>(s1_1.z) - cross(kc_0, _S58);
    return;
}

fn node_pass_0( tid_5 : u32,  n0_4 : u32,  n1_4 : u32,  out_v_0 : u32) -> f32
{
    var p_3 : u32 = n0_4 + tid_5;
    var s_8 : f32 = 0.0f;
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
        var a_5 : vec3<f32>;
        var l_3 : vec3<f32>;
        gather_0(i_4, true, false, &(a_5), &(l_3));
        set_v_0(i_4, out_v_0, a_5, l_3);
        var s_9 : f32 = s_8 + norm2_0(a_5, l_3);
        p_3 = p_3 + u32(1024);
        s_8 = s_9;
    }
    return s_8;
}

fn block_apply_0( n_9 : u32,  a_6 : vec3<f32>,  l_4 : vec3<f32>,  oa_1 : ptr<function, vec3<f32>>,  ol_1 : ptr<function, vec3<f32>>)
{
    var _S59 : u32 = ns_0(n_9, u32(18));
    var b1_1 : vec4<f32> = scratch_0[_S59 + u32(1)];
    var k_9 : vec3<f32> = scratch_0[_S59 + u32(2)].xyz;
    (*oa_1) = sym_mul_0(scratch_0[_S59], scratch_0[_S59 + u32(1)], a_6) - cross(k_9, l_4);
    (*ol_1) = cross(k_9, a_6) + vec3<f32>(b1_1.z) * l_4;
    return;
}

var<workgroup> g_max_0 : array<f32, i32(1024)>;

var<workgroup> g_max_out_0 : f32;

fn group_max_0( tid_6 : u32,  v_12 : f32) -> f32
{
    g_max_0[tid_6] = v_12;
    workgroupBarrier();
    var k_10 : u32 = u32(512);
    loop
    {
        if(k_10 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_6 < k_10)
        {
            g_max_0[tid_6] = max(g_max_0[tid_6], g_max_0[tid_6 + k_10]);
        }
        workgroupBarrier();
        k_10 = (k_10 >> (u32(1)));
    }
    if(tid_6 == u32(0))
    {
        g_max_out_0 = g_max_0[i32(0)];
    }
    var _S60 : f32 = (workgroupUniformLoad(&((g_max_out_0))));
    workgroupBarrier();
    return _S60;
}

@compute
@workgroup_size(1024, 1, 1)
fn blast_native(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var force_start_0 : f32;
    var best_0 : f32;
    var failed_0 : bool;
    var q_1 : u32;
    var ok_0 : bool;
    var tid_7 : u32 = thread_2.x;
    var k_11 : u32 = group_0.x;
    if(k_11 >= (params_0.island_count_0))
    {
        return;
    }
    var range_0 : vec4<u32> = islands_0[k_11];
    var n0_5 : u32 = range_0.x;
    var n1_5 : u32 = range_0.y;
    var b0_1 : u32 = range_0.z;
    var _S61 : u32 = range_0.w;
    var p_4 : u32;
    if((n1_5 - n0_5) > (params_0.wide_min_0))
    {
        p_4 = u32(1);
    }
    else
    {
        p_4 = u32(0);
    }
    var _S62 : u32 = group_uniform_0(p_4, tid_7);
    if(_S62 != u32(0))
    {
        return;
    }
    var last_0 : vec4<u32> = report_0[k_11];
    var last_flags_0 : u32 = group_uniform_0(report_0[k_11].y, tid_7);
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
        var _S63 : u32 = n0_5 + tid_7;
        singular_0 = 0.0f;
        p_4 = _S63;
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
            var _S64 : u32 = i_5 * u32(2);
            var _S65 : u32 = ns_0(i_5, u32(16));
            if((any((accel_0[_S64] != scratch_0[_S65]))))
            {
                converged_0 = true;
            }
            else
            {
                converged_0 = (any((accel_0[_S64 + u32(1)] != scratch_0[_S65 + u32(1)])));
            }
            if(converged_0)
            {
                singular_0 = 1.0f;
            }
            p_4 = p_4 + u32(1024);
        }
        var _S66 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(singular_0, 0.0f));
        if((_S66.x) == 0.0f)
        {
            if(tid_7 == u32(0))
            {
                report_0[k_11] = vec4<u32>(u32(0), u32(7), last_0.z, last_0.w);
            }
            return;
        }
    }
    var _S67 : u32 = n0_5 + tid_7;
    singular_0 = 0.0f;
    p_4 = _S67;
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
        var n_10 : BlastNode_std430_0 = nodes_0[i_6];
        var _S68 : u32 = i_6 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S68];
        var a_lin_0 : vec4<f32> = accel_0[_S68 + u32(1)];
        var _S69 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz);
        if((n_10.w_ang_0) > 0.0f)
        {
            force_start_0 = 1.0f / n_10.w_ang_0;
        }
        else
        {
            force_start_0 = 1.0f;
        }
        var ba_0 : vec3<f32> = _S69 * vec3<f32>(force_start_0);
        var _S70 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz);
        if((n_10.w_lin_0) > 0.0f)
        {
            best_0 = params_0.recip_length_0 / n_10.w_lin_0;
        }
        else
        {
            best_0 = params_0.recip_length_0;
        }
        var bl_0 : vec3<f32> = _S70 * vec3<f32>(best_0);
        var target_1 : f32 = target_0 + norm2_0(ba_0, bl_0);
        set_v_0(i_6, u32(14), ba_0, bl_0);
        set_v_0(i_6, u32(0), ba_0, bl_0);
        var _S71 : u32 = ns_0(i_6, u32(16));
        scratch_0[_S71] = a_ang_0;
        scratch_0[_S71 + u32(1)] = a_lin_0;
        var _S72 : vec3<f32> = vec3<f32>(0.0f);
        set_v_0(i_6, u32(6), _S72, _S72);
        set_v_0(i_6, u32(2), _S72, _S72);
        var _S73 : bool = build_block_0(i_6);
        if(!_S73)
        {
            singular_0 = 1.0f;
        }
        p_4 = p_4 + u32(1024);
        target_0 = target_1;
    }
    var q_2 : u32;
    if(!warm_1)
    {
        q_2 = b0_1 + tid_7;
        loop
        {
            if(q_2 < _S61)
            {
            }
            else
            {
                break;
            }
            var _S74 : u32 = lists_0[q_2] * u32(2);
            var _S75 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S74] = _S75;
            x_0[_S74 + u32(1)] = _S75;
            q_2 = q_2 + u32(1024);
        }
    }
    var tsum_0 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(target_0, singular_0));
    var delta2_0 : f32 = tsum_0.x * params_0.tolerance_0 * params_0.tolerance_0;
    var _S76 : vec4<u32> = group_uniform4_0(vec4<u32>(frame_word_0(k_11, u32(0)), frame_word_0(k_11, u32(1)), frame_word_0(k_11, u32(2)), u32(0)), tid_7);
    var frame_0 : vec3<u32> = _S76.xyz;
    if((tsum_0.y) > 0.0f)
    {
        converged_0 = true;
    }
    else
    {
        converged_0 = (frame_0.z) != u32(0);
    }
    var _S77 : bool = (frame_0.x) == u32(0);
    var _S78 : u32 = frame_0.y;
    device_group_barrier_0();
    if(!(delta2_0 > 0.0f))
    {
        q_2 = b0_1 + tid_7;
        loop
        {
            if(q_2 < _S61)
            {
            }
            else
            {
                break;
            }
            var _S79 : u32 = lists_0[q_2] * u32(2);
            var _S80 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S79] = _S80;
            x_0[_S79 + u32(1)] = _S80;
            q_2 = q_2 + u32(1024);
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
            report_0[k_11] = vec4<u32>(u32(0), (u32(1) | (p_4)), (bitcast<u32>((0.0f))), (bitcast<u32>((delta2_0))));
        }
        return;
    }
    if(warm_1)
    {
        var _S81 : f32 = rebuild_0(tid_7, n0_5, n1_5, false);
    }
    var _S82 : f32 = float_bits_0(u32(2139095040));
    if((params_0.force_tolerance_0) > 0.0f)
    {
        q_2 = b0_1 + tid_7;
        force_start_0 = 0.0f;
        loop
        {
            if(q_2 < _S61)
            {
            }
            else
            {
                break;
            }
            var _S83 : u32 = lists_0[q_2] * u32(2);
            var f_0 : f32 = force_start_0 + norm2_0(x_0[_S83].xyz, x_0[_S83 + u32(1)].xyz);
            q_2 = q_2 + u32(1024);
            force_start_0 = f_0;
        }
        var _S84 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(force_start_0, 0.0f));
        force_start_0 = sqrt(_S84.x);
    }
    else
    {
        force_start_0 = 0.0f;
    }
    best_0 = _S82;
    q_2 = u32(0);
    var step_0 : f32 = _S82;
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
            var _S85 : bool = converged_0;
            converged_0 = false;
            failed_0 = _S85;
            break;
        }
        project_0(tid_7, k_11, _S78, n0_5, n1_5, u32(0), _S77);
        var num_1 : f32 = bond_pass_0(tid_7, b0_1, _S61, u32(0));
        var _S86 : bool = num_1 <= delta2_0;
        if(_S86)
        {
            var _S87 : f32 = residual_norm_0(tid_7, n0_5, n1_5);
            ok_0 = !(_S87 <= delta2_0);
        }
        else
        {
            ok_0 = false;
        }
        var _S88 : bool = iteration_0 > u32(0);
        var _S89 : bool;
        if(_S88)
        {
            _S89 = true;
        }
        else
        {
            _S89 = warm_1;
        }
        var _S90 : bool;
        if(_S89)
        {
            _S90 = _S86;
        }
        else
        {
            _S90 = false;
        }
        var _S91 : bool;
        if(_S90)
        {
            _S91 = !ok_0;
        }
        else
        {
            _S91 = false;
        }
        var hold_0 : bool;
        var num_2 : f32;
        var previous_1 : f32;
        if(_S91)
        {
            var _S92 : f32 = rebuild_0(tid_7, n0_5, n1_5, true);
            project_0(tid_7, k_11, _S78, n0_5, n1_5, u32(0), _S77);
            var num_3 : f32 = bond_pass_0(tid_7, b0_1, _S61, u32(0));
            if(num_3 <= delta2_0)
            {
                var _S93 : f32 = residual_norm_0(tid_7, n0_5, n1_5);
                hold_0 = !(_S93 <= delta2_0);
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
            q_2 = iteration_0;
        }
        else
        {
            if((iteration_0 - q_2) >= u32(512))
            {
                var _S94 : bool = converged_0;
                converged_0 = false;
                failed_0 = _S94;
                num_0 = num_2;
                break;
            }
        }
        var _S95 : bool;
        if(num_2 <= delta2_0)
        {
            _S95 = !hold_0;
        }
        else
        {
            _S95 = false;
        }
        if(_S95)
        {
            var _S96 : bool = converged_0;
            converged_0 = true;
            failed_0 = _S96;
            num_0 = num_2;
            break;
        }
        var _S97 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            _S97 = _S88;
        }
        else
        {
            _S97 = false;
        }
        var _S98 : bool;
        if(_S97)
        {
            _S98 = step_0 <= (params_0.force_tolerance_0 * (force_start_0 + travel_0));
        }
        else
        {
            _S98 = false;
        }
        var mloc_0 : f32;
        if(_S98)
        {
            q_1 = b0_1 + tid_7;
            mloc_0 = 0.0f;
            loop
            {
                if(q_1 < _S61)
                {
                }
                else
                {
                    break;
                }
                var j_5 : u32 = lists_0[q_1];
                var _S99 : BlastBond_std430_0 = bonds_0[j_5];
                var ta_4 : vec3<f32>;
                var tl_4 : vec3<f32>;
                bond_term_1(&(_S99), u32(6), &(ta_4), &(tl_4));
                var _S100 : u32 = j_5 * u32(2);
                var f_1 : f32 = mloc_0 + norm2_0(x_0[_S100].xyz + ta_4, x_0[_S100 + u32(1)].xyz + tl_4);
                q_1 = q_1 + u32(1024);
                mloc_0 = f_1;
            }
            var _S101 : f32 = params_0.force_tolerance_0;
            var _S102 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(mloc_0, 0.0f));
            if(step_0 <= (_S101 * sqrt(_S102.x)))
            {
                var _S103 : bool = converged_0;
                converged_0 = true;
                failed_0 = _S103;
                num_0 = num_2;
                break;
            }
        }
        device_group_barrier_0();
        var _S104 : bool;
        if(iteration_0 == u32(0))
        {
            _S104 = (params_0.first_polynomial_0) == u32(0);
        }
        else
        {
            _S104 = false;
        }
        if(_S104)
        {
            p_4 = _S67;
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
            p_4 = _S67;
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
                var a_7 : vec3<f32>;
                var l_5 : vec3<f32>;
                inverse_apply_0(i_8, va_0(i_8, u32(0)), vl_0(i_8, u32(0)), &(a_7), &(l_5));
                set_v_0(i_8, u32(10), a_7, l_5);
                p_4 = p_4 + u32(1024);
            }
            device_group_barrier_0();
            var _S105 : f32 = bond_pass_0(tid_7, b0_1, _S61, u32(10));
            device_group_barrier_0();
            var _S106 : f32 = node_pass_0(tid_7, n0_5, n1_5, u32(12));
            q_1 = _S67;
            loop
            {
                if(q_1 < n1_5)
                {
                }
                else
                {
                    break;
                }
                var i_9 : u32 = lists_0[q_1];
                var sa_0 : vec3<f32>;
                var sl_0 : vec3<f32>;
                block_apply_0(i_9, va_0(i_9, u32(10)), vl_0(i_9, u32(10)), &(sa_0), &(sl_0));
                var ma_0 : vec3<f32>;
                var ml_0 : vec3<f32>;
                inverse_apply_0(i_9, va_0(i_9, u32(12)) - sa_0, vl_0(i_9, u32(12)) - sl_0, &(ma_0), &(ml_0));
                var _S107 : vec3<f32> = vec3<f32>(1.68946564197540283f);
                var _S108 : vec3<f32> = vec3<f32>(1.5220409631729126f);
                set_v_0(i_9, u32(8), _S107 * va_0(i_9, u32(10)) - _S108 * ma_0, _S107 * vl_0(i_9, u32(10)) - _S108 * ml_0);
                q_1 = q_1 + u32(1024);
            }
        }
        device_group_barrier_0();
        project_0(tid_7, k_11, _S78, n0_5, n1_5, u32(8), _S77);
        mloc_0 = 0.0f;
        p_4 = _S67;
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
            var a_8 : vec3<f32> = abs(va_0(i_10, u32(8)));
            var l_6 : vec3<f32> = abs(vl_0(i_10, u32(8)));
            var _S109 : f32 = max(mloc_0, max(max(max(a_8.x, a_8.y), max(a_8.z, l_6.x)), max(l_6.y, l_6.z)));
            var p_5 : u32 = p_4 + u32(1024);
            mloc_0 = _S109;
            p_4 = p_5;
        }
        var mag_0 : f32 = group_max_0(tid_7, mloc_0);
        var _S110 : bool;
        if(!(mag_0 > 0.0f))
        {
            _S110 = true;
        }
        else
        {
            _S110 = !isfinite_0(mag_0);
        }
        if(_S110)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            break;
        }
        var _S111 : f32 = 1.0f / mag_0;
        q_1 = _S67;
        var gl_0 : f32 = 0.0f;
        loop
        {
            if(q_1 < n1_5)
            {
            }
            else
            {
                break;
            }
            var i_11 : u32 = lists_0[q_1];
            var _S112 : vec3<f32> = vec3<f32>(_S111);
            var ga_0 : vec3<f32> = va_0(i_11, u32(8)) * _S112;
            var gn_0 : vec3<f32> = vl_0(i_11, u32(8)) * _S112;
            set_v_0(i_11, u32(8), ga_0, gn_0);
            var gl_1 : f32 = gl_0 + (dot(va_0(i_11, u32(0)), ga_0) + dot(vl_0(i_11, u32(0)), gn_0));
            q_1 = q_1 + u32(1024);
            gl_0 = gl_1;
        }
        var _S113 : vec2<f32> = blast_group_sum_0(tid_7, vec2<f32>(gl_0, 0.0f));
        var gamma_0 : f32 = _S113.x;
        var _S114 : bool = gamma_0 > 0.0f;
        var _S115 : bool;
        if(!_S114)
        {
            _S115 = true;
        }
        else
        {
            _S115 = !isfinite_0(gamma_0);
        }
        if(_S115)
        {
            converged_0 = false;
            failed_0 = true;
            num_0 = num_2;
            break;
        }
        var _S116 : bool;
        if(iteration_0 > u32(1))
        {
            _S116 = true;
        }
        else
        {
            if((params_0.first_polynomial_0) != u32(0))
            {
                _S116 = _S88;
            }
            else
            {
                _S116 = false;
            }
        }
        var _S117 : bool;
        if(_S116)
        {
            _S117 = previous_1 > 0.0f;
        }
        else
        {
            _S117 = false;
        }
        if(_S117)
        {
            singular_0 = gamma_0 / previous_1;
        }
        else
        {
            singular_0 = 0.0f;
        }
        var p_6 : u32 = _S67;
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
        var den_0 : f32 = bond_pass_0(tid_7, b0_1, _S61, u32(2));
        device_group_barrier_0();
        var _S118 : f32 = node_pass_0(tid_7, n0_5, n1_5, u32(4));
        var _S119 : bool;
        if((params_0.force_tolerance_0) > 0.0f)
        {
            if(den_0 > 0.0f)
            {
                _S119 = _S114;
            }
            else
            {
                _S119 = false;
            }
            var step_1 : f32;
            if(_S119)
            {
                step_1 = gamma_0 / sqrt(den_0);
            }
            else
            {
                step_1 = _S82;
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
        var _S120 : u32 = iteration_0 + u32(1);
        if(!(den_0 > 0.0f))
        {
            _S119 = true;
        }
        else
        {
            _S119 = !isfinite_0(den_0);
        }
        if(_S119)
        {
            var _S121 : bool = converged_0;
            converged_0 = false;
            failed_0 = _S121;
            num_0 = num_2;
            iteration_0 = _S120;
            break;
        }
        var _S122 : f32 = gamma_0 / den_0;
        var p_7 : u32 = _S67;
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
            var _S123 : vec3<f32> = vec3<f32>(_S122);
            set_v_0(i_13, u32(6), va_0(i_13, u32(6)) + va_0(i_13, u32(2)) * _S123, vl_0(i_13, u32(6)) + vl_0(i_13, u32(2)) * _S123);
            set_v_0(i_13, u32(0), va_0(i_13, u32(0)) - va_0(i_13, u32(4)) * _S123, vl_0(i_13, u32(0)) - vl_0(i_13, u32(4)) * _S123);
            p_7 = p_7 + u32(1024);
        }
        device_group_barrier_0();
        previous_0 = gamma_0;
        num_0 = num_2;
        iteration_0 = _S120;
    }
    device_group_barrier_0();
    q_1 = b0_1 + tid_7;
    loop
    {
        if(q_1 < _S61)
        {
        }
        else
        {
            break;
        }
        var j_6 : u32 = lists_0[q_1];
        var _S124 : BlastBond_std430_0 = bonds_0[j_6];
        if(!((_S124.health_0) > 0.0f))
        {
            var _S125 : u32 = j_6 * u32(2);
            var _S126 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S125] = _S126;
            x_0[_S125 + u32(1)] = _S126;
            q_1 = q_1 + u32(1024);
            continue;
        }
        var ta_5 : vec3<f32>;
        var tl_5 : vec3<f32>;
        bond_term_1(&(_S124), u32(6), &(ta_5), &(tl_5));
        var _S127 : u32 = j_6 * u32(2);
        x_0[_S127] = vec4<f32>(x_0[_S127].xyz + ta_5, 0.0f);
        var _S128 : u32 = _S127 + u32(1);
        x_0[_S128] = vec4<f32>(x_0[_S128].xyz + tl_5, 0.0f);
        q_1 = q_1 + u32(1024);
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
        q_2 = u32(4);
    }
    else
    {
        q_2 = u32(0);
    }
    var flags_0 : u32 = (p_4 | (q_2));
    if(tid_7 == u32(0))
    {
        report_0[k_11] = vec4<u32>(iteration_0, flags_0, (bitcast<u32>((num_0))), (bitcast<u32>((delta2_0))));
    }
    return;
}

