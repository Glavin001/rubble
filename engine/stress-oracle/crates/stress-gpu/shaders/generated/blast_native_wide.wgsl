@binding(12) @group(0) var<storage, read_write> meta_0 : array<atomic<u32>>;

@binding(13) @group(0) var<storage, read> topo_args_0 : array<u32>;

@binding(1) @group(0) var<storage, read> islands_0 : array<vec4<u32>>;

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
@binding(11) @group(0) var<storage, read_write> wide_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read> lists_0 : array<u32>;

struct BlastNode_std430_0
{
    @align(16) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(8) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
    @align(16) position_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read> nodes_0 : array<BlastNode_std430_0>;

@binding(5) @group(0) var<storage, read> accel_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

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

@binding(8) @group(0) var<storage, read_write> report_0 : array<vec4<u32>>;

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

@compute
@workgroup_size(1, 1, 1)
fn w_reset(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    atomicStore(&(meta_0[u32(0)]), u32(0));
    atomicStore(&(meta_0[u32(1)]), u32(0));
    atomicStore(&(meta_0[u32(2)]), u32(0));
    return;
}

fn blocks_0( n_0 : u32) -> u32
{
    return (n_0 + u32(1024) - u32(1)) / u32(1024);
}

@compute
@workgroup_size(64, 1, 1)
fn w_classify(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var k_0 : u32 = id_1.x;
    if(k_0 >= topo_args_0[i32(3)])
    {
        return;
    }
    var r_0 : vec4<u32> = islands_0[k_0];
    var _S8 : u32 = r_0.y - r_0.x;
    if(_S8 <= (params_0.wide_min_0))
    {
        return;
    }
    var row_0 : u32 = atomicAdd(&(meta_0[u32(0)]), u32(1));
    if(row_0 < (params_0.wide_rows_0))
    {
        atomicStore(&(meta_0[u32(16) + row_0]), k_0);
    }
    var _S9 : u32 = atomicMax(&(meta_0[u32(1)]), blocks_0(_S8));
    var _S10 : u32 = atomicMax(&(meta_0[u32(2)]), blocks_0(r_0.w - r_0.z));
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn w_args(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S11 : u32 = atomicLoad(&(meta_0[u32(0)]));
    var _S12 : u32 = min(_S11, params_0.wide_rows_0);
    var _S13 : bool = _S12 > u32(0);
    var nb_0 : u32;
    if(_S13)
    {
        var _S14 : u32 = atomicLoad(&(meta_0[u32(1)]));
        nb_0 = _S14;
    }
    else
    {
        nb_0 = u32(0);
    }
    var bb_0 : u32;
    if(_S13)
    {
        var _S15 : u32 = atomicLoad(&(meta_0[u32(2)]));
        bb_0 = _S15;
    }
    else
    {
        bb_0 = u32(0);
    }
    atomicStore(&(meta_0[u32(4)]), nb_0);
    atomicStore(&(meta_0[u32(5)]), _S12);
    atomicStore(&(meta_0[u32(6)]), u32(1));
    atomicStore(&(meta_0[u32(8)]), bb_0);
    atomicStore(&(meta_0[u32(9)]), _S12);
    atomicStore(&(meta_0[u32(10)]), u32(1));
    atomicStore(&(meta_0[u32(12)]), _S12);
    atomicStore(&(meta_0[u32(13)]), u32(1));
    atomicStore(&(meta_0[u32(14)]), u32(1));
    return;
}

fn row_island_0( row_1 : u32) -> u32
{
    var _S16 : u32 = atomicLoad(&(meta_0[u32(16) + row_1]));
    return _S16;
}

fn st_0( row_2 : u32,  slot_0 : u32) -> u32
{
    return row_2 * u32(16) + slot_0;
}

fn flags_of_0( row_3 : u32) -> u32
{
    return (bitcast<u32>((wide_0[st_0(row_3, u32(2))].w)));
}

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_0 : vec4<u32>,  thread_0 : u32) -> vec4<u32>
{
    if(thread_0 == u32(0))
    {
        g_group_uniform4_0 = value_0;
    }
    var _S17 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S17;
}

struct Block_0
{
     row_4 : u32,
     island_0 : u32,
     begin_0 : u32,
     end_0 : u32,
     flags_0 : u32,
};

fn node_block_0( group_0 : vec3<u32>,  tid_0 : u32) -> Block_0
{
    var b_0 : Block_0;
    var _S18 : u32 = group_0.y;
    b_0.row_4 = _S18;
    var _S19 : u32 = row_island_0(_S18);
    var u_0 : vec4<u32> = group_uniform4_0(vec4<u32>(_S19, u32(0), u32(0), flags_of_0(_S18)), tid_0);
    var _S20 : u32 = u_0.x;
    b_0.island_0 = _S20;
    b_0.flags_0 = u_0.w;
    var g_0 : vec4<u32> = group_uniform4_0(islands_0[_S20], tid_0);
    var _S21 : u32 = g_0.x + group_0.x * u32(1024);
    b_0.begin_0 = _S21;
    b_0.end_0 = min(_S21 + u32(1024), g_0.y);
    return b_0;
}

fn ns_0( n_1 : u32,  v_0 : u32) -> u32
{
    return n_1 * params_0.node_vectors_0 + v_0;
}

fn norm2_0( a_0 : vec3<f32>,  b_1 : vec3<f32>) -> f32
{
    return dot(a_0, a_0) + dot(b_1, b_1);
}

fn set_v_0( n_2 : u32,  v_1 : u32,  a_1 : vec3<f32>,  l_0 : vec3<f32>)
{
    var _S22 : u32 = ns_0(n_2, v_1);
    scratch_0[_S22] = vec4<f32>(a_1, 0.0f);
    scratch_0[_S22 + u32(1)] = vec4<f32>(l_0, 0.0f);
    return;
}

fn stiff_0( j_0 : u32,  which_0 : u32,  v_2 : vec3<f32>) -> vec3<f32>
{
    var _S23 : u32 = u32(6) * which_0;
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
        var _S24 : u32 = _S23 + k_1;
        m_0[k_1] = stiffness_0[j_0 * u32(6) + _S24 / u32(4)][_S24 % u32(4)];
        k_1 = k_1 + u32(1);
    }
    var _S25 : f32 = v_2.x;
    var _S26 : f32 = v_2.y;
    var _S27 : f32 = v_2.z;
    return vec3<f32>(m_0[i32(0)] * _S25 + m_0[i32(3)] * _S26 + m_0[i32(4)] * _S27, m_0[i32(3)] * _S25 + m_0[i32(1)] * _S26 + m_0[i32(5)] * _S27, m_0[i32(4)] * _S25 + m_0[i32(5)] * _S26 + m_0[i32(2)] * _S27);
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
    var _S28 : f32 = o_0.z;
    var _S29 : f32 = o_0.y;
    var _S30 : f32 = o_0.x;
    return mat3x3<f32>(0.0f, - _S28, _S29, _S28, 0.0f, - _S30, - _S29, _S30, 0.0f);
}

fn build_full_block_0( node_0 : u32) -> bool
{
    var c_0 : u32;
    var k_2 : u32;
    var s_1 : f32;
    var n_3 : BlastNode_std430_0 = nodes_0[node_0];
    var aa_0 : mat3x3<f32> = mat3x3<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    var r_1 : u32 = n_3.ref_begin_0;
    var aa_1 : mat3x3<f32> = aa_0;
    var al_0 : mat3x3<f32> = aa_0;
    var ll_0 : mat3x3<f32> = aa_0;
    loop
    {
        if(r_1 < (n_3.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_0 : u32 = refs_0[r_1];
        var j_2 : u32 = (ref_0 & (u32(2147483647)));
        var b_2 : BlastBond_std430_0 = bonds_0[j_2];
        if(!((bonds_0[j_2].health_0) > 0.0f))
        {
            r_1 = r_1 + u32(1);
            continue;
        }
        var o_1 : vec3<f32>;
        if(((ref_0 & (u32(2147483648)))) != u32(0))
        {
            o_1 = b_2.o1_0.xyz;
        }
        else
        {
            o_1 = b_2.o0_0.xyz;
        }
        var _S31 : f32 = b_2.s_0;
        var s2_0 : f32 = _S31 * _S31;
        var wl_0 : mat3x3<f32> = sym3_0(j_2, u32(3));
        var k_3 : mat3x3<f32> = skew_0(o_1);
        var _S32 : mat3x3<f32> = (((wl_0) * (k_3)));
        var _S33 : mat3x3<f32> = sym3_0(j_2, u32(1)) + (((transpose(k_3)) * (_S32)));
        var _S34 : mat3x3<f32> = mat3x3<f32>(s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0, s2_0);
        var _S35 : mat3x3<f32> = mat3x3<f32>(- s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0, - s2_0);
        var _S36 : mat3x3<f32> = al_0 + mat3x3<f32>(_S35[0] * _S32[0], _S35[1] * _S32[1], _S35[2] * _S32[2]);
        var _S37 : mat3x3<f32> = ll_0 + mat3x3<f32>(_S34[0] * wl_0[0], _S34[1] * wl_0[1], _S34[2] * wl_0[2]);
        aa_1 = aa_1 + mat3x3<f32>(_S34[0] * _S33[0], _S34[1] * _S33[1], _S34[2] * _S33[2]);
        al_0 = _S36;
        ll_0 = _S37;
        r_1 = r_1 + u32(1);
    }
    var m_1 : array<array<f32, i32(6)>, i32(6)>;
    r_1 = u32(0);
    loop
    {
        if(r_1 < u32(3))
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
            m_1[r_1][c_0] = aa_1[r_1][c_0] * n_3.w_ang_0 * n_3.w_ang_0;
            var _S38 : u32 = c_0 + u32(3);
            var _S39 : f32 = al_0[r_1][c_0] * n_3.w_ang_0 * n_3.w_lin_0;
            m_1[r_1][_S38] = _S39;
            m_1[_S38][r_1] = _S39;
            m_1[r_1 + u32(3)][_S38] = ll_0[r_1][c_0] * n_3.w_lin_0 * n_3.w_lin_0;
            c_0 = c_0 + u32(1);
        }
        r_1 = r_1 + u32(1);
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
    r_1 = u32(0);
    loop
    {
        if(r_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 <= r_1)
            {
            }
            else
            {
                break;
            }
            packed_0[r_1 * (r_1 + u32(1)) / u32(2) + c_0] = m_1[r_1][c_0];
            c_0 = c_0 + u32(1);
        }
        r_1 = r_1 + u32(1);
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
        var _S40 : u32 = u32(4) * q_0;
        scratch_0[ns_0(node_0, u32(18) + q_0)] = vec4<f32>(packed_0[_S40], packed_0[_S40 + u32(1)], packed_0[_S40 + u32(2)], packed_0[_S40 + u32(3)]);
        q_0 = q_0 + u32(1);
    }
    var l_1 : array<array<f32, i32(6)>, i32(6)>;
    r_1 = u32(0);
    loop
    {
        if(r_1 < u32(6))
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
            l_1[r_1][c_0] = 0.0f;
            c_0 = c_0 + u32(1);
        }
        r_1 = r_1 + u32(1);
    }
    r_1 = u32(0);
    loop
    {
        if(r_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 <= r_1)
            {
            }
            else
            {
                break;
            }
            k_2 = u32(0);
            s_1 = m_1[r_1][c_0];
            loop
            {
                if(k_2 < c_0)
                {
                }
                else
                {
                    break;
                }
                var s_2 : f32 = s_1 - l_1[r_1][k_2] * l_1[c_0][k_2];
                k_2 = k_2 + u32(1);
                s_1 = s_2;
            }
            if(r_1 == c_0)
            {
                var _S41 : bool;
                if(!(s_1 > 0.0f))
                {
                    _S41 = true;
                }
                else
                {
                    _S41 = !isfinite_0(s_1);
                }
                if(_S41)
                {
                    return false;
                }
                l_1[r_1][c_0] = sqrt(s_1);
            }
            else
            {
                l_1[r_1][c_0] = s_1 / l_1[c_0][c_0];
            }
            c_0 = c_0 + u32(1);
        }
        r_1 = r_1 + u32(1);
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
        r_1 = u32(0);
        loop
        {
            if(r_1 < u32(6))
            {
            }
            else
            {
                break;
            }
            if(r_1 == c_0)
            {
                s_1 = 1.0f;
            }
            else
            {
                s_1 = 0.0f;
            }
            k_2 = u32(0);
            var s_3 : f32 = s_1;
            loop
            {
                if(k_2 < r_1)
                {
                }
                else
                {
                    break;
                }
                var s_4 : f32 = s_3 - l_1[r_1][k_2] * y_0[k_2];
                k_2 = k_2 + u32(1);
                s_3 = s_4;
            }
            y_0[r_1] = s_3 / l_1[r_1][r_1];
            r_1 = r_1 + u32(1);
        }
        var r_2 : i32 = i32(5);
        loop
        {
            if(r_2 >= i32(0))
            {
            }
            else
            {
                break;
            }
            k_2 = u32(r_2) + u32(1);
            s_1 = y_0[r_2];
            loop
            {
                if(k_2 < u32(6))
                {
                }
                else
                {
                    break;
                }
                var s_5 : f32 = s_1 - l_1[k_2][r_2] * inv_0[k_2][c_0];
                k_2 = k_2 + u32(1);
                s_1 = s_5;
            }
            inv_0[r_2][c_0] = s_1 / l_1[r_2][r_2];
            r_2 = r_2 - i32(1);
        }
        c_0 = c_0 + u32(1);
    }
    r_1 = u32(0);
    loop
    {
        if(r_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        c_0 = u32(0);
        loop
        {
            if(c_0 <= r_1)
            {
            }
            else
            {
                break;
            }
            packed_0[r_1 * (r_1 + u32(1)) / u32(2) + c_0] = inv_0[r_1][c_0];
            c_0 = c_0 + u32(1);
        }
        r_1 = r_1 + u32(1);
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
        var _S42 : u32 = u32(4) * q_0;
        scratch_0[ns_0(node_0, u32(24) + q_0)] = vec4<f32>(packed_0[_S42], packed_0[_S42 + u32(1)], packed_0[_S42 + u32(2)], packed_0[_S42 + u32(3)]);
        q_0 = q_0 + u32(1);
    }
    return true;
}

fn build_block_0( node_1 : u32) -> bool
{
    if((params_0.rotation_0) == u32(2))
    {
        var _S43 : bool = build_full_block_0(node_1);
        return _S43;
    }
    var n_4 : BlastNode_std430_0 = nodes_0[node_1];
    var _S44 : vec3<f32> = vec3<f32>(0.0f);
    var r_3 : u32 = n_4.ref_begin_0;
    var diag_0 : vec3<f32> = _S44;
    var offd_0 : vec3<f32> = _S44;
    var ks_0 : vec3<f32> = _S44;
    var cs_0 : f32 = 0.0f;
    loop
    {
        if(r_3 < (n_4.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_1 : u32 = refs_0[r_3];
        var j_3 : u32 = (ref_1 & (u32(2147483647)));
        var b_3 : BlastBond_std430_0 = bonds_0[j_3];
        if(!((bonds_0[j_3].health_0) > 0.0f))
        {
            r_3 = r_3 + u32(1);
            continue;
        }
        var o_2 : vec3<f32>;
        if(((ref_1 & (u32(2147483648)))) != u32(0))
        {
            o_2 = b_3.o1_0.xyz;
        }
        else
        {
            o_2 = b_3.o0_0.xyz;
        }
        var _S45 : f32 = b_3.s_0;
        var s2_1 : f32 = _S45 * _S45;
        var _S46 : vec3<f32> = vec3<f32>(1.0f);
        var wd_0 : vec3<f32>;
        var wo_0 : vec3<f32>;
        if((params_0.rotation_0) != u32(0))
        {
            var c1_1 : vec3<f32> = stiff_0(j_3, u32(1), vec3<f32>(0.0f, 1.0f, 0.0f));
            var c2_1 : vec3<f32> = stiff_0(j_3, u32(1), vec3<f32>(0.0f, 0.0f, 1.0f));
            var _S47 : vec3<f32> = vec3<f32>(c1_1.x, c2_1.x, c2_1.y);
            wd_0 = vec3<f32>(stiff_0(j_3, u32(1), vec3<f32>(1.0f, 0.0f, 0.0f)).x, c1_1.y, c2_1.z);
            wo_0 = _S47;
        }
        else
        {
            wd_0 = _S46;
            wo_0 = _S44;
        }
        var _S48 : vec3<f32> = vec3<f32>(s2_1);
        var _S49 : f32 = o_2.x;
        var _S50 : f32 = o_2.y;
        var _S51 : f32 = o_2.z;
        var offd_1 : vec3<f32> = offd_0 + _S48 * (wo_0 - vec3<f32>(_S49 * _S50, _S49 * _S51, _S50 * _S51));
        var ks_1 : vec3<f32> = ks_0 + _S48 * o_2;
        var cs_1 : f32 = cs_0 + s2_1;
        diag_0 = diag_0 + _S48 * (wd_0 + vec3<f32>(dot(o_2, o_2)) - o_2 * o_2);
        offd_0 = offd_1;
        ks_0 = ks_1;
        cs_0 = cs_1;
        r_3 = r_3 + u32(1);
    }
    var _S52 : f32 = n_4.w_ang_0;
    var da_0 : f32 = _S52 * _S52;
    var c_1 : f32 = n_4.w_lin_0 * n_4.w_lin_0 * cs_0;
    var k_4 : vec3<f32> = ks_0 * vec3<f32>((n_4.w_ang_0 * n_4.w_lin_0));
    var a0_0 : vec4<f32> = vec4<f32>(diag_0 * vec3<f32>(da_0), offd_0.x * da_0);
    var _S53 : f32 = offd_0.y * da_0;
    var _S54 : f32 = offd_0.z * da_0;
    var a1_0 : vec4<f32> = vec4<f32>(_S53, _S54, c_1, 0.0f);
    var _S55 : u32 = ns_0(node_1, u32(18));
    scratch_0[_S55] = a0_0;
    scratch_0[_S55 + u32(1)] = a1_0;
    scratch_0[_S55 + u32(2)] = vec4<f32>(k_4, 0.0f);
    if(!(c_1 > 0.0f))
    {
        return false;
    }
    var ic_0 : f32 = 1.0f / c_1;
    var kk_0 : f32 = dot(k_4, k_4);
    var _S56 : f32 = k_4.x;
    var sxx_0 : f32 = a0_0.x + (_S56 * _S56 - kk_0) * ic_0;
    var _S57 : f32 = k_4.y;
    var syy_0 : f32 = a0_0.y + (_S57 * _S57 - kk_0) * ic_0;
    var _S58 : f32 = k_4.z;
    var szz_0 : f32 = a0_0.z + (_S58 * _S58 - kk_0) * ic_0;
    var sxy_0 : f32 = a0_0.w + _S56 * _S57 * ic_0;
    var sxz_0 : f32 = _S53 + _S56 * _S58 * ic_0;
    var syz_0 : f32 = _S54 + _S57 * _S58 * ic_0;
    var cxx_0 : f32 = syy_0 * szz_0 - syz_0 * syz_0;
    var cxy_0 : f32 = sxz_0 * syz_0 - sxy_0 * szz_0;
    var cxz_0 : f32 = sxy_0 * syz_0 - sxz_0 * syy_0;
    var det_0 : f32 = sxx_0 * cxx_0 + sxy_0 * cxy_0 + sxz_0 * cxz_0;
    var _S59 : bool;
    if(!(det_0 > 0.0f))
    {
        _S59 = true;
    }
    else
    {
        _S59 = !isfinite_0(det_0);
    }
    if(_S59)
    {
        return false;
    }
    var id_3 : f32 = 1.0f / det_0;
    var cyz_0 : f32 = sxy_0 * sxz_0 - sxx_0 * syz_0;
    var _S60 : u32 = ns_0(node_1, u32(21));
    scratch_0[_S60] = vec4<f32>(cxx_0 * id_3, (sxx_0 * szz_0 - sxz_0 * sxz_0) * id_3, (sxx_0 * syy_0 - sxy_0 * sxy_0) * id_3, cxy_0 * id_3);
    scratch_0[_S60 + u32(1)] = vec4<f32>(cxz_0 * id_3, cyz_0 * id_3, ic_0, 0.0f);
    scratch_0[_S60 + u32(2)] = vec4<f32>(k_4 * vec3<f32>(ic_0), 0.0f);
    return true;
}

var<workgroup> g_blast_red_0 : array<vec2<f32>, i32(1024)>;

var<workgroup> g_blast_sum_0 : vec2<f32>;

fn blast_group_sum_0( tid_1 : u32,  v_3 : vec2<f32>) -> vec2<f32>
{
    g_blast_red_0[tid_1] = v_3;
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
        if(tid_1 < k_5)
        {
            g_blast_red_0[tid_1] = g_blast_red_0[tid_1] + g_blast_red_0[tid_1 + k_5];
        }
        workgroupBarrier();
        k_5 = (k_5 >> (u32(1)));
    }
    if(tid_1 == u32(0))
    {
        g_blast_sum_0 = g_blast_red_0[i32(0)];
    }
    var s_6 : vec2<f32> = (workgroupUniformLoad(&((g_blast_sum_0))));
    workgroupBarrier();
    return s_6;
}

fn pt_0( row_5 : u32,  block_0 : u32,  k_6 : u32) -> u32
{
    return params_0.wide_rows_0 * u32(16) + (row_5 * params_0.wide_blocks_0 + block_0) * u32(6) + k_6;
}

fn write_partial_0( tid_2 : u32,  row_6 : u32,  block_1 : u32,  k_7 : u32,  v_4 : vec4<f32>)
{
    var a_2 : vec2<f32> = blast_group_sum_0(tid_2, v_4.xy);
    var b_4 : vec2<f32> = blast_group_sum_0(tid_2, v_4.zw);
    if(tid_2 == u32(0))
    {
        wide_0[pt_0(row_6, block_1, k_7)] = vec4<f32>(a_2, b_4);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_init_nodes(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_3 : u32 = thread_1.x;
    var b_5 : Block_0 = node_block_0(group_1, tid_3);
    if((b_5.begin_0) >= (b_5.end_0))
    {
        return;
    }
    var p0_0 : vec4<f32> = vec4<f32>(0.0f);
    var p_0 : u32 = b_5.begin_0 + tid_3;
    if(p_0 < (b_5.end_0))
    {
        var i_0 : u32 = lists_0[p_0];
        var n_5 : BlastNode_std430_0 = nodes_0[i_0];
        var _S61 : u32 = i_0 * u32(2);
        var a_ang_0 : vec4<f32> = accel_0[_S61];
        var a_lin_0 : vec4<f32> = accel_0[_S61 + u32(1)];
        var _S62 : u32 = ns_0(i_0, u32(16));
        var _S63 : bool;
        if((any((a_ang_0 != scratch_0[_S62]))))
        {
            _S63 = true;
        }
        else
        {
            _S63 = (any((a_lin_0 != scratch_0[_S62 + u32(1)])));
        }
        if(_S63)
        {
            p0_0[i32(2)] = 1.0f;
        }
        var _S64 : vec3<f32> = (vec3<f32>(0) - a_ang_0.xyz);
        var _S65 : f32;
        if((n_5.w_ang_0) > 0.0f)
        {
            _S65 = 1.0f / n_5.w_ang_0;
        }
        else
        {
            _S65 = 1.0f;
        }
        var ba_0 : vec3<f32> = _S64 * vec3<f32>(_S65);
        var _S66 : vec3<f32> = (vec3<f32>(0) - a_lin_0.xyz);
        if((n_5.w_lin_0) > 0.0f)
        {
            _S65 = params_0.recip_length_0 / n_5.w_lin_0;
        }
        else
        {
            _S65 = params_0.recip_length_0;
        }
        var bl_0 : vec3<f32> = _S66 * vec3<f32>(_S65);
        p0_0[i32(0)] = norm2_0(ba_0, bl_0);
        set_v_0(i_0, u32(14), ba_0, bl_0);
        set_v_0(i_0, u32(0), ba_0, bl_0);
        scratch_0[_S62] = a_ang_0;
        scratch_0[_S62 + u32(1)] = a_lin_0;
        var _S67 : vec3<f32> = vec3<f32>(0.0f);
        set_v_0(i_0, u32(6), _S67, _S67);
        set_v_0(i_0, u32(2), _S67, _S67);
        var _S68 : bool = build_block_0(i_0);
        if(!_S68)
        {
            p0_0[i32(1)] = 1.0f;
        }
    }
    write_partial_0(tid_3, b_5.row_4, group_1.x, u32(0), p0_0);
    return;
}

fn bond_block_0( group_2 : vec3<u32>,  tid_4 : u32) -> Block_0
{
    var _S69 : Block_0 = node_block_0(group_2, tid_4);
    var b_6 : Block_0 = _S69;
    var g_1 : vec4<u32> = group_uniform4_0(islands_0[b_6.island_0], tid_4);
    var _S70 : u32 = g_1.z + group_2.x * u32(1024);
    b_6.begin_0 = _S70;
    b_6.end_0 = min(_S70 + u32(1024), g_1.w);
    return b_6;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_init_bonds(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var tid_5 : u32 = thread_2.x;
    var b_7 : Block_0 = bond_block_0(group_3, tid_5);
    if((b_7.begin_0) >= (b_7.end_0))
    {
        return;
    }
    var p_1 : u32 = b_7.begin_0 + tid_5;
    var _S71 : bool;
    if(p_1 < (b_7.end_0))
    {
        _S71 = (params_0.warm_0) == u32(0);
    }
    else
    {
        _S71 = false;
    }
    if(_S71)
    {
        var _S72 : u32 = lists_0[p_1] * u32(2);
        var _S73 : vec4<f32> = vec4<f32>(0.0f);
        x_0[_S72] = _S73;
        x_0[_S72 + u32(1)] = _S73;
    }
    return;
}

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_1 : u32,  thread_3 : u32) -> u32
{
    if(thread_3 == u32(0))
    {
        g_group_uniform_0 = value_1;
    }
    var _S74 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S74;
}

fn row_sum_0( tid_6 : u32,  row_7 : u32,  count_0 : u32,  k_8 : u32) -> vec4<f32>
{
    var _S75 : vec4<f32> = vec4<f32>(0.0f);
    var b_8 : u32 = tid_6;
    var s_7 : vec4<f32> = _S75;
    loop
    {
        if(b_8 < count_0)
        {
        }
        else
        {
            break;
        }
        var s_8 : vec4<f32> = s_7 + wide_0[pt_0(row_7, b_8, k_8)];
        b_8 = b_8 + u32(1024);
        s_7 = s_8;
    }
    var x_4 : vec2<f32> = blast_group_sum_0(tid_6, s_7.xy);
    var y_1 : vec2<f32> = blast_group_sum_0(tid_6, s_7.zw);
    return vec4<f32>(x_4, y_1);
}

fn blast_frame_0( node_count_1 : u32,  k_9 : u32) -> u32
{
    return node_count_1 + u32(9) * k_9;
}

fn frame_word_0( k_10 : u32,  w_0 : u32) -> u32
{
    return islands_0[blast_frame_0(params_0.node_count_0, k_10) + w_0 / u32(4)][w_0 % u32(4)];
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

@compute
@workgroup_size(1024, 1, 1)
fn w_setup(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var tid_7 : u32 = thread_4.x;
    var row_8 : u32 = group_4.x;
    var _S76 : u32 = row_island_0(row_8);
    var island_1 : u32 = group_uniform_0(_S76, tid_7);
    var r_4 : vec4<u32> = group_uniform4_0(islands_0[island_1], tid_7);
    var s0_0 : vec4<f32> = row_sum_0(tid_7, row_8, blocks_0(r_4.y - r_4.x), u32(0));
    if(tid_7 != u32(0))
    {
        return;
    }
    var last_0 : vec4<u32> = report_0[island_1];
    var _S77 : bool;
    if((params_0.warm_0) != u32(0))
    {
        _S77 = (((last_0.y) & (u32(4)))) != u32(0);
    }
    else
    {
        _S77 = false;
    }
    if(_S77)
    {
        _S77 = (s0_0.z) == 0.0f;
    }
    else
    {
        _S77 = false;
    }
    var f_0 : u32;
    if(_S77)
    {
        f_0 = u32(128);
    }
    else
    {
        f_0 = u32(1);
    }
    var delta2_0 : f32 = s0_0.x * params_0.tolerance_0 * params_0.tolerance_0;
    if(f_0 == u32(1))
    {
        if((s0_0.y) > 0.0f)
        {
            _S77 = true;
        }
        else
        {
            _S77 = (frame_word_0(island_1, u32(2))) != u32(0);
        }
        if(_S77)
        {
            f_0 = u32(4);
        }
        else
        {
            if(!(delta2_0 > 0.0f))
            {
                f_0 = u32(18);
            }
        }
    }
    if((frame_word_0(island_1, u32(0))) == u32(0))
    {
        f_0 = (f_0 | (u32(8)));
    }
    var _S78 : f32 = float_bits_0(u32(2139095040));
    wide_0[st_0(row_8, u32(0))] = vec4<f32>(delta2_0, 0.0f, 0.0f, _S78);
    wide_0[st_0(row_8, u32(1))] = vec4<f32>(0.0f, 0.0f, 0.0f, _S78);
    var _S79 : f32 = (bitcast<f32>((u32(0))));
    wide_0[st_0(row_8, u32(2))] = vec4<f32>(0.0f, 0.0f, _S79, (bitcast<f32>((f_0))));
    wide_0[st_0(row_8, u32(3))] = vec4<f32>(0.0f, 0.0f, 0.0f, _S79);
    return;
}

fn bs_0( j_4 : u32) -> u32
{
    return params_0.node_count_0 * params_0.node_vectors_0 + u32(2) * j_4;
}

fn va_0( n_6 : u32,  v_5 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_6, v_5)].xyz;
}

fn vl_0( n_7 : u32,  v_6 : u32) -> vec3<f32>
{
    return scratch_0[ns_0(n_7, v_6) + u32(1)].xyz;
}

fn rot_a_0( j_5 : u32,  v_7 : vec3<f32>) -> vec3<f32>
{
    var _S80 : vec3<f32>;
    if((params_0.rotation_0) != u32(0))
    {
        _S80 = stiff_0(j_5, u32(0), v_7);
    }
    else
    {
        _S80 = v_7;
    }
    return _S80;
}

fn rot_l_0( j_6 : u32,  v_8 : vec3<f32>) -> vec3<f32>
{
    var _S81 : vec3<f32>;
    if((params_0.rotation_0) == u32(2))
    {
        _S81 = stiff_0(j_6, u32(2), v_8);
    }
    else
    {
        _S81 = v_8;
    }
    return _S81;
}

fn bond_term_0( b_9 : ptr<function, BlastBond_std430_0>,  j_7 : u32,  v_9 : u32,  ta_0 : ptr<function, vec3<f32>>,  tl_0 : ptr<function, vec3<f32>>)
{
    var m0_0 : BlastNode_std430_0 = nodes_0[(*b_9).n0_0];
    var m1_0 : BlastNode_std430_0 = nodes_0[(*b_9).n1_0];
    var a0_1 : vec3<f32> = va_0((*b_9).n0_0, v_9) * vec3<f32>(m0_0.w_ang_0);
    var l0_0 : vec3<f32> = vl_0((*b_9).n0_0, v_9) * vec3<f32>(m0_0.w_lin_0);
    var a1_1 : vec3<f32> = va_0((*b_9).n1_0, v_9) * vec3<f32>(m1_0.w_ang_0);
    var l1_0 : vec3<f32> = vl_0((*b_9).n1_0, v_9) * vec3<f32>(m1_0.w_lin_0);
    var _S82 : vec3<f32> = vec3<f32>((*b_9).s_0);
    (*ta_0) = rot_a_0(j_7, a0_1 - a1_1) * _S82;
    (*tl_0) = rot_l_0(j_7, l0_0 - l1_0 + (cross((*b_9).o0_0.xyz, a0_1) - cross((*b_9).o1_0.xyz, a1_1))) * _S82;
    return;
}

fn bond_term_1( b_10 : ptr<function, BlastBond_std430_0>,  j_8 : u32,  v_10 : u32,  ta_1 : ptr<function, vec3<f32>>,  tl_1 : ptr<function, vec3<f32>>)
{
    var m0_1 : BlastNode_std430_0 = nodes_0[(*b_10).n0_0];
    var m1_1 : BlastNode_std430_0 = nodes_0[(*b_10).n1_0];
    var a0_2 : vec3<f32> = va_0((*b_10).n0_0, v_10) * vec3<f32>(m0_1.w_ang_0);
    var l0_1 : vec3<f32> = vl_0((*b_10).n0_0, v_10) * vec3<f32>(m0_1.w_lin_0);
    var a1_2 : vec3<f32> = va_0((*b_10).n1_0, v_10) * vec3<f32>(m1_1.w_ang_0);
    var l1_1 : vec3<f32> = vl_0((*b_10).n1_0, v_10) * vec3<f32>(m1_1.w_lin_0);
    var _S83 : vec3<f32> = vec3<f32>((*b_10).s_0);
    (*ta_1) = rot_a_0(j_8, a0_2 - a1_2) * _S83;
    (*tl_1) = rot_l_0(j_8, l0_1 - l1_1 + (cross((*b_10).o0_0.xyz, a0_2) - cross((*b_10).o1_0.xyz, a1_2))) * _S83;
    return;
}

fn gather_0( node_2 : u32,  from_t_0 : bool,  with_u_0 : bool,  ang_0 : ptr<function, vec3<f32>>,  lin_0 : ptr<function, vec3<f32>>)
{
    var n_8 : BlastNode_std430_0 = nodes_0[node_2];
    var _S84 : vec3<f32> = vec3<f32>(0.0f);
    var r_5 : u32 = n_8.ref_begin_0;
    var acc_a_0 : vec3<f32> = _S84;
    var acc_l_0 : vec3<f32> = _S84;
    loop
    {
        if(r_5 < (n_8.ref_end_0))
        {
        }
        else
        {
            break;
        }
        var ref_2 : u32 = refs_0[r_5];
        var j_9 : u32 = (ref_2 & (u32(2147483647)));
        var _S85 : BlastBond_std430_0 = bonds_0[j_9];
        if(!((_S85.health_0) > 0.0f))
        {
            r_5 = r_5 + u32(1);
            continue;
        }
        var ya_0 : vec3<f32>;
        var yl_0 : vec3<f32>;
        if(from_t_0)
        {
            var _S86 : u32 = bs_0(j_9);
            var _S87 : vec3<f32> = scratch_0[_S86 + u32(1)].xyz;
            ya_0 = scratch_0[_S86].xyz;
            yl_0 = _S87;
        }
        else
        {
            var _S88 : u32 = j_9 * u32(2);
            var ya_1 : vec3<f32> = x_0[_S88].xyz;
            var yl_1 : vec3<f32> = x_0[_S88 + u32(1)].xyz;
            if(with_u_0)
            {
                var ta_2 : vec3<f32>;
                var tl_2 : vec3<f32>;
                bond_term_1(&(_S85), j_9, u32(6), &(ta_2), &(tl_2));
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
        var _S89 : vec3<f32> = vec3<f32>(_S85.s_0);
        var ya_2 : vec3<f32> = rot_a_0(j_9, ya_0) * _S89;
        var yl_3 : vec3<f32> = rot_l_0(j_9, yl_0) * _S89;
        var acc_a_1 : vec3<f32>;
        var acc_l_1 : vec3<f32>;
        if(((ref_2 & (u32(2147483648)))) != u32(0))
        {
            var acc_l_2 : vec3<f32> = acc_l_0 - yl_3;
            acc_a_1 = acc_a_0 + (cross(_S85.o1_0.xyz, yl_3) - ya_2);
            acc_l_1 = acc_l_2;
        }
        else
        {
            var acc_l_3 : vec3<f32> = acc_l_0 + yl_3;
            acc_a_1 = acc_a_0 + (ya_2 - cross(_S85.o0_0.xyz, yl_3));
            acc_l_1 = acc_l_3;
        }
        acc_a_0 = acc_a_1;
        acc_l_0 = acc_l_1;
        r_5 = r_5 + u32(1);
    }
    (*ang_0) = acc_a_0 * vec3<f32>(n_8.w_ang_0);
    (*lin_0) = acc_l_0 * vec3<f32>(n_8.w_lin_0);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_warm(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var tid_8 : u32 = thread_5.x;
    var b_11 : Block_0 = node_block_0(group_5, tid_8);
    var _S90 : bool;
    if((b_11.begin_0) >= (b_11.end_0))
    {
        _S90 = true;
    }
    else
    {
        _S90 = (params_0.warm_0) == u32(0);
    }
    if(_S90)
    {
        _S90 = true;
    }
    else
    {
        _S90 = (((b_11.flags_0) & (u32(1)))) == u32(0);
    }
    if(_S90)
    {
        return;
    }
    var p_2 : u32 = b_11.begin_0 + tid_8;
    if(p_2 < (b_11.end_0))
    {
        var i_1 : u32 = lists_0[p_2];
        var a_3 : vec3<f32>;
        var l_2 : vec3<f32>;
        gather_0(i_1, false, false, &(a_3), &(l_2));
        set_v_0(i_1, u32(0), va_0(i_1, u32(14)) - a_3, vl_0(i_1, u32(14)) - l_2);
    }
    return;
}

fn runs_0( flags_1 : u32,  only_rebuild_0 : bool) -> bool
{
    var _S91 : bool;
    if(((flags_1 & (u32(1)))) != u32(0))
    {
        if(!only_rebuild_0)
        {
            _S91 = true;
        }
        else
        {
            _S91 = ((flags_1 & (u32(64)))) != u32(0);
        }
    }
    else
    {
        _S91 = false;
    }
    return _S91;
}

fn frame_float_0( k_11 : u32,  w_1 : u32) -> f32
{
    return (bitcast<f32>((frame_word_0(k_11, w_1))));
}

fn frame_axis_0( k_12 : u32,  rotations_0 : u32,  kk_1 : u32) -> vec3<f32>
{
    if(rotations_0 == u32(1))
    {
        return vec3<f32>(frame_float_0(k_12, u32(4)), frame_float_0(k_12, u32(5)), frame_float_0(k_12, u32(6)));
    }
    var _S92 : f32;
    if(kk_1 == u32(0))
    {
        _S92 = 1.0f;
    }
    else
    {
        _S92 = 0.0f;
    }
    var _S93 : f32;
    if(kk_1 == u32(1))
    {
        _S93 = 1.0f;
    }
    else
    {
        _S93 = 0.0f;
    }
    var _S94 : f32;
    if(kk_1 == u32(2))
    {
        _S94 = 1.0f;
    }
    else
    {
        _S94 = 0.0f;
    }
    return vec3<f32>(_S92, _S93, _S94);
}

fn frame_terms_0( k_13 : u32,  rotations_1 : u32,  i_2 : u32,  v_11 : u32,  s_9 : ptr<function, array<f32, i32(6)>>)
{
    var n_9 : BlastNode_std430_0 = nodes_0[i_2];
    var lin_1 : vec3<f32> = vl_0(i_2, v_11) / vec3<f32>(n_9.w_lin_0);
    var _S95 : vec3<f32> = va_0(i_2, v_11) / vec3<f32>(n_9.w_ang_0) - cross(n_9.position_0.xyz, lin_1);
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
        (*s_9)[kk_2] = (*s_9)[kk_2] + dot(frame_axis_0(k_13, rotations_1, kk_2), _S95);
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

fn proj_part_0( group_6 : vec3<u32>,  tid_9 : u32,  v_12 : u32,  only_rebuild_1 : bool)
{
    var b_12 : Block_0 = node_block_0(group_6, tid_9);
    var _S96 : bool;
    if((b_12.begin_0) >= (b_12.end_0))
    {
        _S96 = true;
    }
    else
    {
        _S96 = !runs_0(b_12.flags_0, only_rebuild_1);
    }
    if(_S96)
    {
        _S96 = true;
    }
    else
    {
        _S96 = (((b_12.flags_0) & (u32(8)))) == u32(0);
    }
    if(_S96)
    {
        return;
    }
    var s_10 : array<f32, i32(6)>;
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
        s_10[z_1] = 0.0f;
        z_1 = z_1 + u32(1);
    }
    var p_3 : u32 = b_12.begin_0 + tid_9;
    if(p_3 < (b_12.end_0))
    {
        var k_14 : u32 = row_island_0(b_12.row_4);
        frame_terms_0(k_14, frame_word_0(k_14, u32(1)), lists_0[p_3], v_12, &(s_10));
    }
    var _S97 : u32 = group_6.x;
    write_partial_0(tid_9, b_12.row_4, _S97, u32(0), vec4<f32>(s_10[i32(0)], s_10[i32(1)], s_10[i32(2)], s_10[i32(3)]));
    write_partial_0(tid_9, b_12.row_4, _S97, u32(1), vec4<f32>(s_10[i32(4)], s_10[i32(5)], 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_part_r(@builtin(workgroup_id) g_2 : vec3<u32>, @builtin(local_invocation_id) t_0 : vec3<u32>)
{
    proj_part_0(g_2, t_0.x, u32(0), false);
    return;
}

fn row_range_0( tid_10 : u32,  row_9 : u32) -> vec4<u32>
{
    var _S98 : u32 = row_island_0(row_9);
    var _S99 : u32 = group_uniform_0(_S98, tid_10);
    var _S100 : vec4<u32> = group_uniform4_0(islands_0[_S99], tid_10);
    return _S100;
}

fn frame_solve_0( k_15 : u32,  rotations_2 : u32,  sum_0 : array<f32, i32(6)>,  omega_0 : ptr<function, vec3<f32>>,  t_1 : ptr<function, vec3<f32>>)
{
    var j_11 : u32;
    var d_0 : u32 = rotations_2 + u32(3);
    var c_2 : array<f32, i32(6)>;
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
        c_2[z_2] = 0.0f;
        z_2 = z_2 + u32(1);
    }
    var row_10 : u32 = u32(0);
    loop
    {
        if(row_10 < d_0)
        {
        }
        else
        {
            break;
        }
        var _S101 : f32 = sum_0[row_10] * frame_float_0(k_15, u32(8) + row_10);
        j_11 = u32(0);
        var value_2 : f32 = _S101;
        loop
        {
            if(j_11 < row_10)
            {
            }
            else
            {
                break;
            }
            var value_3 : f32 = value_2 - frame_float_0(k_15, u32(14) + row_10 * (row_10 + u32(1)) / u32(2) + j_11) * c_2[j_11];
            j_11 = j_11 + u32(1);
            value_2 = value_3;
        }
        var _S102 : u32 = row_10 + u32(1);
        c_2[row_10] = value_2 / frame_float_0(k_15, u32(14) + row_10 * _S102 / u32(2) + row_10);
        row_10 = _S102;
    }
    var row_11 : i32 = i32(d_0) - i32(1);
    loop
    {
        if(row_11 >= i32(0))
        {
        }
        else
        {
            break;
        }
        var r_6 : u32 = u32(row_11);
        var _S103 : u32 = r_6 + u32(1);
        j_11 = _S103;
        loop
        {
            if(j_11 < d_0)
            {
            }
            else
            {
                break;
            }
            var _S104 : u32 = j_11 + u32(1);
            c_2[r_6] = c_2[r_6] - frame_float_0(k_15, u32(14) + j_11 * _S104 / u32(2) + r_6) * c_2[j_11];
            j_11 = _S104;
        }
        c_2[r_6] = c_2[r_6] / frame_float_0(k_15, u32(14) + r_6 * _S103 / u32(2) + r_6);
        row_11 = row_11 - i32(1);
    }
    z_2 = u32(0);
    loop
    {
        if(z_2 < d_0)
        {
        }
        else
        {
            break;
        }
        c_2[z_2] = c_2[z_2] * frame_float_0(k_15, u32(8) + z_2);
        z_2 = z_2 + u32(1);
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
        (*omega_0) = (*omega_0) + frame_axis_0(k_15, rotations_2, j_11) * vec3<f32>(c_2[j_11]);
        j_11 = j_11 + u32(1);
    }
    (*t_1) = vec3<f32>(c_2[rotations_2], c_2[rotations_2 + u32(1)], c_2[rotations_2 + u32(2)]);
    return;
}

fn proj_solve_0( group_7 : vec3<u32>,  tid_11 : u32,  only_rebuild_2 : bool)
{
    var row_12 : u32 = group_7.x;
    var f_1 : u32 = group_uniform_0(flags_of_0(row_12), tid_11);
    var _S105 : bool;
    if(!runs_0(f_1, only_rebuild_2))
    {
        _S105 = true;
    }
    else
    {
        _S105 = ((f_1 & (u32(8)))) == u32(0);
    }
    if(_S105)
    {
        return;
    }
    var r_7 : vec4<u32> = row_range_0(tid_11, row_12);
    var nb_1 : u32 = blocks_0(r_7.y - r_7.x);
    var a_4 : vec4<f32> = row_sum_0(tid_11, row_12, nb_1, u32(0));
    var b_13 : vec4<f32> = row_sum_0(tid_11, row_12, nb_1, u32(1));
    if(tid_11 != u32(0))
    {
        return;
    }
    var k_16 : u32 = row_island_0(row_12);
    var sum_1 : array<f32, i32(6)> = array<f32, i32(6)>( a_4.x, a_4.y, a_4.z, a_4.w, b_13.x, b_13.y );
    var omega_1 : vec3<f32>;
    var t_2 : vec3<f32>;
    frame_solve_0(k_16, frame_word_0(k_16, u32(1)), sum_1, &(omega_1), &(t_2));
    wide_0[st_0(row_12, u32(4))] = vec4<f32>(omega_1, 0.0f);
    wide_0[st_0(row_12, u32(5))] = vec4<f32>(t_2, 0.0f);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_solve(@builtin(workgroup_id) g_3 : vec3<u32>, @builtin(local_invocation_id) t_3 : vec3<u32>)
{
    proj_solve_0(g_3, t_3.x, false);
    return;
}

fn frame_apply_0( i_3 : u32,  v_13 : u32,  omega_2 : vec3<f32>,  t_4 : vec3<f32>)
{
    var n_10 : BlastNode_std430_0 = nodes_0[i_3];
    set_v_0(i_3, v_13, va_0(i_3, v_13) - omega_2 / vec3<f32>(n_10.w_ang_0), vl_0(i_3, v_13) - (t_4 + cross(n_10.position_0.xyz, omega_2)) / vec3<f32>(n_10.w_lin_0));
    return;
}

fn proj_apply_0( group_8 : vec3<u32>,  tid_12 : u32,  v_14 : u32,  only_rebuild_3 : bool)
{
    var b_14 : Block_0 = node_block_0(group_8, tid_12);
    var _S106 : bool;
    if((b_14.begin_0) >= (b_14.end_0))
    {
        _S106 = true;
    }
    else
    {
        _S106 = !runs_0(b_14.flags_0, only_rebuild_3);
    }
    if(_S106)
    {
        _S106 = true;
    }
    else
    {
        _S106 = (((b_14.flags_0) & (u32(8)))) == u32(0);
    }
    if(_S106)
    {
        return;
    }
    var p_4 : u32 = b_14.begin_0 + tid_12;
    if(p_4 >= (b_14.end_0))
    {
        return;
    }
    frame_apply_0(lists_0[p_4], v_14, wide_0[st_0(b_14.row_4, u32(4))].xyz, wide_0[st_0(b_14.row_4, u32(5))].xyz);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_apply_r(@builtin(workgroup_id) g_4 : vec3<u32>, @builtin(local_invocation_id) t_5 : vec3<u32>)
{
    proj_apply_0(g_4, t_5.x, u32(0), false);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_part_rb(@builtin(workgroup_id) g_5 : vec3<u32>, @builtin(local_invocation_id) t_6 : vec3<u32>)
{
    proj_part_0(g_5, t_6.x, u32(0), true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_solve_rb(@builtin(workgroup_id) g_6 : vec3<u32>, @builtin(local_invocation_id) t_7 : vec3<u32>)
{
    proj_solve_0(g_6, t_7.x, true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_apply_rb(@builtin(workgroup_id) g_7 : vec3<u32>, @builtin(local_invocation_id) t_8 : vec3<u32>)
{
    proj_apply_0(g_7, t_8.x, u32(0), true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_part_g(@builtin(workgroup_id) g_8 : vec3<u32>, @builtin(local_invocation_id) t_9 : vec3<u32>)
{
    proj_part_0(g_8, t_9.x, u32(8), false);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_proj_apply_g(@builtin(workgroup_id) g_9 : vec3<u32>, @builtin(local_invocation_id) t_10 : vec3<u32>)
{
    proj_apply_0(g_9, t_10.x, u32(8), false);
    return;
}

fn bond_r_0( group_9 : vec3<u32>,  tid_13 : u32,  only_rebuild_4 : bool)
{
    var b_15 : Block_0 = bond_block_0(group_9, tid_13);
    var _S107 : bool;
    if((b_15.begin_0) >= (b_15.end_0))
    {
        _S107 = true;
    }
    else
    {
        _S107 = !runs_0(b_15.flags_0, only_rebuild_4);
    }
    if(_S107)
    {
        return;
    }
    var p_5 : u32 = b_15.begin_0 + tid_13;
    var z_3 : f32;
    var f_2 : f32;
    if(p_5 < (b_15.end_0))
    {
        var j_12 : u32 = lists_0[p_5];
        var _S108 : BlastBond_std430_0 = bonds_0[j_12];
        var _S109 : vec3<f32> = vec3<f32>(0.0f);
        var ta_3 : vec3<f32> = _S109;
        var tl_3 : vec3<f32> = _S109;
        if((_S108.health_0) > 0.0f)
        {
            bond_term_0(&(_S108), j_12, u32(0), &(ta_3), &(tl_3));
            if((params_0.force_tolerance_0) > 0.0f)
            {
                var ua_0 : vec3<f32>;
                var ul_0 : vec3<f32>;
                bond_term_0(&(_S108), j_12, u32(6), &(ua_0), &(ul_0));
                var _S110 : u32 = j_12 * u32(2);
                z_3 = norm2_0(x_0[_S110].xyz + ua_0, x_0[_S110 + u32(1)].xyz + ul_0);
            }
            else
            {
                z_3 = 0.0f;
            }
        }
        else
        {
            z_3 = 0.0f;
        }
        var _S111 : f32 = norm2_0(ta_3, tl_3);
        var _S112 : u32 = bs_0(j_12);
        scratch_0[_S112] = vec4<f32>(ta_3, 0.0f);
        scratch_0[_S112 + u32(1)] = vec4<f32>(tl_3, 0.0f);
        var _S113 : f32 = z_3;
        z_3 = _S111;
        f_2 = _S113;
    }
    else
    {
        z_3 = 0.0f;
        f_2 = 0.0f;
    }
    write_partial_0(tid_13, b_15.row_4, group_9.x, u32(2), vec4<f32>(z_3, f_2, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_r(@builtin(workgroup_id) g_10 : vec3<u32>, @builtin(local_invocation_id) t_11 : vec3<u32>)
{
    bond_r_0(g_10, t_11.x, false);
    return;
}

fn node_rn_0( group_10 : vec3<u32>,  tid_14 : u32,  only_rebuild_5 : bool)
{
    var b_16 : Block_0 = node_block_0(group_10, tid_14);
    var _S114 : bool;
    if((b_16.begin_0) >= (b_16.end_0))
    {
        _S114 = true;
    }
    else
    {
        _S114 = !runs_0(b_16.flags_0, only_rebuild_5);
    }
    if(_S114)
    {
        return;
    }
    var p_6 : u32 = b_16.begin_0 + tid_14;
    var s_11 : f32;
    if(p_6 < (b_16.end_0))
    {
        var i_4 : u32 = lists_0[p_6];
        s_11 = norm2_0(va_0(i_4, u32(0)), vl_0(i_4, u32(0)));
    }
    else
    {
        s_11 = 0.0f;
    }
    write_partial_0(tid_14, b_16.row_4, group_10.x, u32(3), vec4<f32>(s_11, 0.0f, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_node_rn(@builtin(workgroup_id) g_11 : vec3<u32>, @builtin(local_invocation_id) t_12 : vec3<u32>)
{
    node_rn_0(g_11, t_12.x, false);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_rb(@builtin(workgroup_id) g_12 : vec3<u32>, @builtin(local_invocation_id) t_13 : vec3<u32>)
{
    bond_r_0(g_12, t_13.x, true);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_node_rnb(@builtin(workgroup_id) g_13 : vec3<u32>, @builtin(local_invocation_id) t_14 : vec3<u32>)
{
    node_rn_0(g_13, t_14.x, true);
    return;
}

fn iteration_of_0( row_13 : u32) -> u32
{
    return (bitcast<u32>((wide_0[st_0(row_13, u32(2))].z)));
}

fn set_flags_0( row_14 : u32,  f_3 : u32)
{
    var _S115 : u32 = st_0(row_14, u32(2));
    var c_3 : vec4<f32> = wide_0[_S115];
    c_3[i32(3)] = (bitcast<f32>((f_3)));
    wide_0[_S115] = c_3;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_test(@builtin(workgroup_id) group_11 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var tid_15 : u32 = thread_6.x;
    var row_15 : u32 = group_11.x;
    var f_4 : u32 = group_uniform_0(flags_of_0(row_15), tid_15);
    if(((f_4 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_8 : vec4<u32> = row_range_0(tid_15, row_15);
    var z_4 : vec4<f32> = row_sum_0(tid_15, row_15, blocks_0(r_8.w - r_8.z), u32(2));
    var rn_0 : vec4<f32> = row_sum_0(tid_15, row_15, blocks_0(r_8.y - r_8.x), u32(3));
    if(tid_15 != u32(0))
    {
        return;
    }
    var _S116 : u32 = st_0(row_15, u32(0));
    var s0_1 : vec4<f32> = wide_0[_S116];
    var delta2_1 : f32 = wide_0[_S116].x;
    var _S117 : f32 = z_4.x;
    var _S118 : bool = _S117 <= delta2_1;
    var hold_0 : bool;
    if(_S118)
    {
        hold_0 = !((rn_0.x) <= delta2_1);
    }
    else
    {
        hold_0 = false;
    }
    var it_0 : u32 = iteration_of_0(row_15);
    var f_5 : u32 = (f_4 & (u32(4294967199)));
    var f_6 : u32;
    if(hold_0)
    {
        f_6 = (f_5 | (u32(32)));
    }
    else
    {
        f_6 = f_5;
    }
    var _S119 : bool;
    if(it_0 > u32(0))
    {
        _S119 = true;
    }
    else
    {
        _S119 = (params_0.warm_0) != u32(0);
    }
    if(_S119)
    {
        _S119 = _S118;
    }
    else
    {
        _S119 = false;
    }
    if(_S119)
    {
        hold_0 = !hold_0;
    }
    else
    {
        hold_0 = false;
    }
    if(hold_0)
    {
        f_6 = (f_6 | (u32(64)));
    }
    s0_1[i32(1)] = _S117;
    wide_0[_S116] = s0_1;
    var _S120 : u32 = st_0(row_15, u32(1));
    wide_0[_S120] = wide_0[_S120];
    set_flags_0(row_15, f_6);
    var _S121 : u32 = st_0(row_15, u32(2));
    var cnt_0 : vec4<f32> = wide_0[_S121];
    var _S122 : u32 = st_0(row_15, u32(3));
    wide_0[_S122] = vec4<f32>(wide_0[_S122].xyz, wide_0[_S122].w);
    wide_0[pt_0(row_15, u32(0), u32(5))] = vec4<f32>(z_4.y, 0.0f, 0.0f, 0.0f);
    wide_0[_S121] = cnt_0;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_rebuild(@builtin(workgroup_id) group_12 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    var tid_16 : u32 = thread_7.x;
    var b_17 : Block_0 = node_block_0(group_12, tid_16);
    var _S123 : bool;
    if((b_17.begin_0) >= (b_17.end_0))
    {
        _S123 = true;
    }
    else
    {
        _S123 = !runs_0(b_17.flags_0, true);
    }
    if(_S123)
    {
        return;
    }
    var p_7 : u32 = b_17.begin_0 + tid_16;
    if(p_7 < (b_17.end_0))
    {
        var i_5 : u32 = lists_0[p_7];
        var a_5 : vec3<f32>;
        var l_3 : vec3<f32>;
        gather_0(i_5, false, true, &(a_5), &(l_3));
        set_v_0(i_5, u32(0), va_0(i_5, u32(14)) - a_5, vl_0(i_5, u32(14)) - l_3);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_decide(@builtin(workgroup_id) group_13 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    var tid_17 : u32 = thread_8.x;
    var row_16 : u32 = group_13.x;
    var f_7 : u32 = group_uniform_0(flags_of_0(row_16), tid_17);
    if(((f_7 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_9 : vec4<u32> = row_range_0(tid_17, row_16);
    var _S124 : vec4<f32> = vec4<f32>(0.0f);
    var _S125 : bool = ((f_7 & (u32(64)))) != u32(0);
    var z_5 : vec4<f32>;
    var rn_1 : vec4<f32>;
    if(_S125)
    {
        var _S126 : vec4<f32> = row_sum_0(tid_17, row_16, blocks_0(r_9.w - r_9.z), u32(2));
        var _S127 : vec4<f32> = row_sum_0(tid_17, row_16, blocks_0(r_9.y - r_9.x), u32(3));
        z_5 = _S126;
        rn_1 = _S127;
    }
    else
    {
        z_5 = _S124;
        rn_1 = _S124;
    }
    if(tid_17 != u32(0))
    {
        return;
    }
    var _S128 : u32 = st_0(row_16, u32(0));
    var s0_2 : vec4<f32> = wide_0[_S128];
    var s1_0 : vec4<f32> = wide_0[st_0(row_16, u32(1))];
    var cnt_1 : vec4<f32> = wide_0[st_0(row_16, u32(2))];
    var _S129 : u32 = st_0(row_16, u32(3));
    var cen_0 : vec4<f32> = wide_0[_S129];
    var delta2_2 : f32 = wide_0[_S128].x;
    var _S130 : f32 = wide_0[_S128].y;
    var _S131 : bool = ((f_7 & (u32(32)))) != u32(0);
    var hold_1 : bool;
    var num_0 : f32;
    if(_S125)
    {
        var num_1 : f32 = z_5.x;
        if(num_1 <= delta2_2)
        {
            hold_1 = !((rn_1.x) <= delta2_2);
        }
        else
        {
            hold_1 = false;
        }
        s0_2[i32(2)] = 0.0f;
        num_0 = num_1;
    }
    else
    {
        num_0 = _S130;
        hold_1 = _S131;
    }
    var f_8 : u32 = (f_7 & (u32(4294967199)));
    var it_1 : u32 = (bitcast<u32>((cnt_1.z)));
    var best_it_0 : u32 = (bitcast<u32>((cen_0.w)));
    var stop_0 : bool;
    var best_it_1 : u32;
    if(num_0 < (s0_2.w * 0.99000000953674316f))
    {
        s0_2[i32(3)] = num_0;
        stop_0 = false;
        best_it_1 = it_1;
    }
    else
    {
        stop_0 = (it_1 - best_it_0) >= u32(512);
        best_it_1 = best_it_0;
    }
    var _S132 : bool;
    if(!stop_0)
    {
        _S132 = num_0 <= delta2_2;
    }
    else
    {
        _S132 = false;
    }
    if(_S132)
    {
        hold_1 = !hold_1;
    }
    else
    {
        hold_1 = false;
    }
    var f_9 : u32;
    if(hold_1)
    {
        var f_10 : u32 = (f_8 | (u32(2)));
        stop_0 = true;
        f_9 = f_10;
    }
    else
    {
        f_9 = f_8;
    }
    if(!stop_0)
    {
        hold_1 = (params_0.force_tolerance_0) > 0.0f;
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        hold_1 = it_1 > u32(0);
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        hold_1 = (s1_0.w) <= (params_0.force_tolerance_0 * (cnt_1.x + cnt_1.y));
    }
    else
    {
        hold_1 = false;
    }
    if(hold_1)
    {
        if((s1_0.w) <= (params_0.force_tolerance_0 * sqrt(wide_0[pt_0(row_16, u32(0), u32(5))].x)))
        {
            var f_11 : u32 = (f_9 | (u32(2)));
            stop_0 = true;
            f_9 = f_11;
        }
    }
    if(stop_0)
    {
        hold_1 = true;
    }
    else
    {
        hold_1 = it_1 >= (params_0.max_iterations_0);
    }
    if(hold_1)
    {
        f_9 = (f_9 & (u32(4294967294)));
    }
    s0_2[i32(1)] = num_0;
    wide_0[_S128] = s0_2;
    wide_0[_S129] = vec4<f32>(cen_0.xyz, (bitcast<f32>((best_it_1))));
    set_flags_0(row_16, f_9);
    return;
}

fn full_apply_0( n_11 : u32,  at_0 : u32,  a_6 : vec3<f32>,  l_4 : vec3<f32>,  oa_0 : ptr<function, vec3<f32>>,  ol_0 : ptr<function, vec3<f32>>)
{
    var m_2 : array<f32, i32(24)>;
    var q_1 : u32 = u32(0);
    loop
    {
        if(q_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S133 : u32 = u32(4) * q_1;
        m_2[_S133] = scratch_0[ns_0(n_11, at_0 + q_1)].x;
        m_2[_S133 + u32(1)] = scratch_0[ns_0(n_11, at_0 + q_1)].y;
        m_2[_S133 + u32(2)] = scratch_0[ns_0(n_11, at_0 + q_1)].z;
        m_2[_S133 + u32(3)] = scratch_0[ns_0(n_11, at_0 + q_1)].w;
        q_1 = q_1 + u32(1);
    }
    var _S134 : array<f32, i32(6)> = array<f32, i32(6)>( a_6.x, a_6.y, a_6.z, l_4.x, l_4.y, l_4.z );
    var o_3 : array<f32, i32(6)>;
    var r_10 : u32 = u32(0);
    loop
    {
        if(r_10 < u32(6))
        {
        }
        else
        {
            break;
        }
        var c_4 : u32 = u32(0);
        var s_12 : f32 = 0.0f;
        loop
        {
            if(c_4 < u32(6))
            {
            }
            else
            {
                break;
            }
            if(r_10 >= c_4)
            {
                q_1 = r_10 * (r_10 + u32(1)) / u32(2) + c_4;
            }
            else
            {
                q_1 = c_4 * (c_4 + u32(1)) / u32(2) + r_10;
            }
            var s_13 : f32 = s_12 + m_2[q_1] * _S134[c_4];
            c_4 = c_4 + u32(1);
            s_12 = s_13;
        }
        o_3[r_10] = s_12;
        r_10 = r_10 + u32(1);
    }
    (*oa_0) = vec3<f32>(o_3[i32(0)], o_3[i32(1)], o_3[i32(2)]);
    (*ol_0) = vec3<f32>(o_3[i32(3)], o_3[i32(4)], o_3[i32(5)]);
    return;
}

fn sym_mul_0( s0_3 : vec4<f32>,  s1_1 : vec4<f32>,  v_15 : vec3<f32>) -> vec3<f32>
{
    var _S135 : f32 = v_15.x;
    var _S136 : f32 = s0_3.w;
    var _S137 : f32 = v_15.y;
    var _S138 : f32 = s1_1.x;
    var _S139 : f32 = v_15.z;
    var _S140 : f32 = s1_1.y;
    return vec3<f32>(s0_3.x * _S135 + _S136 * _S137 + _S138 * _S139, _S136 * _S135 + s0_3.y * _S137 + _S140 * _S139, _S138 * _S135 + _S140 * _S137 + s0_3.z * _S139);
}

fn inverse_apply_0( n_12 : u32,  a_7 : vec3<f32>,  l_5 : vec3<f32>,  oa_1 : ptr<function, vec3<f32>>,  ol_1 : ptr<function, vec3<f32>>)
{
    if((params_0.rotation_0) == u32(2))
    {
        full_apply_0(n_12, u32(24), a_7, l_5, &((*oa_1)), &((*ol_1)));
        return;
    }
    var _S141 : u32 = ns_0(n_12, u32(21));
    var s1_2 : vec4<f32> = scratch_0[_S141 + u32(1)];
    var kc_0 : vec3<f32> = scratch_0[_S141 + u32(2)].xyz;
    var _S142 : vec3<f32> = sym_mul_0(scratch_0[_S141], scratch_0[_S141 + u32(1)], a_7 + cross(kc_0, l_5));
    (*oa_1) = _S142;
    (*ol_1) = l_5 * vec3<f32>(s1_2.z) - cross(kc_0, _S142);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_pre1(@builtin(workgroup_id) group_14 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    var tid_18 : u32 = thread_9.x;
    var b_18 : Block_0 = node_block_0(group_14, tid_18);
    var _S143 : bool;
    if((b_18.begin_0) >= (b_18.end_0))
    {
        _S143 = true;
    }
    else
    {
        _S143 = !runs_0(b_18.flags_0, false);
    }
    if(_S143)
    {
        return;
    }
    var p_8 : u32 = b_18.begin_0 + tid_18;
    if(p_8 >= (b_18.end_0))
    {
        return;
    }
    var i_6 : u32 = lists_0[p_8];
    if((iteration_of_0(b_18.row_4)) == u32(0))
    {
        _S143 = (params_0.first_polynomial_0) == u32(0);
    }
    else
    {
        _S143 = false;
    }
    if(_S143)
    {
        set_v_0(i_6, u32(8), va_0(i_6, u32(0)), vl_0(i_6, u32(0)));
        return;
    }
    var a_8 : vec3<f32>;
    var l_6 : vec3<f32>;
    inverse_apply_0(i_6, va_0(i_6, u32(0)), vl_0(i_6, u32(0)), &(a_8), &(l_6));
    set_v_0(i_6, u32(10), a_8, l_6);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_loc(@builtin(workgroup_id) group_15 : vec3<u32>, @builtin(local_invocation_id) thread_10 : vec3<u32>)
{
    var tid_19 : u32 = thread_10.x;
    var b_19 : Block_0 = bond_block_0(group_15, tid_19);
    var _S144 : bool;
    if((b_19.begin_0) >= (b_19.end_0))
    {
        _S144 = true;
    }
    else
    {
        _S144 = !runs_0(b_19.flags_0, false);
    }
    if(_S144)
    {
        _S144 = true;
    }
    else
    {
        if((iteration_of_0(b_19.row_4)) == u32(0))
        {
            _S144 = (params_0.first_polynomial_0) == u32(0);
        }
        else
        {
            _S144 = false;
        }
    }
    if(_S144)
    {
        return;
    }
    var p_9 : u32 = b_19.begin_0 + tid_19;
    if(p_9 >= (b_19.end_0))
    {
        return;
    }
    var j_13 : u32 = lists_0[p_9];
    var _S145 : BlastBond_std430_0 = bonds_0[j_13];
    var _S146 : vec3<f32> = vec3<f32>(0.0f);
    var ta_4 : vec3<f32> = _S146;
    var tl_4 : vec3<f32> = _S146;
    if((_S145.health_0) > 0.0f)
    {
        bond_term_1(&(_S145), j_13, u32(10), &(ta_4), &(tl_4));
    }
    var _S147 : u32 = bs_0(j_13);
    scratch_0[_S147] = vec4<f32>(ta_4, 0.0f);
    scratch_0[_S147 + u32(1)] = vec4<f32>(tl_4, 0.0f);
    return;
}

fn block_apply_0( n_13 : u32,  a_9 : vec3<f32>,  l_7 : vec3<f32>,  oa_2 : ptr<function, vec3<f32>>,  ol_2 : ptr<function, vec3<f32>>)
{
    if((params_0.rotation_0) == u32(2))
    {
        full_apply_0(n_13, u32(18), a_9, l_7, &((*oa_2)), &((*ol_2)));
        return;
    }
    var _S148 : u32 = ns_0(n_13, u32(18));
    var b1_0 : vec4<f32> = scratch_0[_S148 + u32(1)];
    var k_17 : vec3<f32> = scratch_0[_S148 + u32(2)].xyz;
    (*oa_2) = sym_mul_0(scratch_0[_S148], scratch_0[_S148 + u32(1)], a_9) - cross(k_17, l_7);
    (*ol_2) = cross(k_17, a_9) + vec3<f32>(b1_0.z) * l_7;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_pre2(@builtin(workgroup_id) group_16 : vec3<u32>, @builtin(local_invocation_id) thread_11 : vec3<u32>)
{
    var tid_20 : u32 = thread_11.x;
    var b_20 : Block_0 = node_block_0(group_16, tid_20);
    var _S149 : bool;
    if((b_20.begin_0) >= (b_20.end_0))
    {
        _S149 = true;
    }
    else
    {
        _S149 = !runs_0(b_20.flags_0, false);
    }
    if(_S149)
    {
        _S149 = true;
    }
    else
    {
        if((iteration_of_0(b_20.row_4)) == u32(0))
        {
            _S149 = (params_0.first_polynomial_0) == u32(0);
        }
        else
        {
            _S149 = false;
        }
    }
    if(_S149)
    {
        return;
    }
    var p_10 : u32 = b_20.begin_0 + tid_20;
    if(p_10 >= (b_20.end_0))
    {
        return;
    }
    var i_7 : u32 = lists_0[p_10];
    var oa_3 : vec3<f32>;
    var ol_3 : vec3<f32>;
    gather_0(i_7, true, false, &(oa_3), &(ol_3));
    var sa_0 : vec3<f32>;
    var sl_0 : vec3<f32>;
    block_apply_0(i_7, va_0(i_7, u32(10)), vl_0(i_7, u32(10)), &(sa_0), &(sl_0));
    var ma_0 : vec3<f32>;
    var ml_0 : vec3<f32>;
    inverse_apply_0(i_7, oa_3 - sa_0, ol_3 - sl_0, &(ma_0), &(ml_0));
    var _S150 : vec3<f32> = vec3<f32>(1.68946564197540283f);
    var _S151 : vec3<f32> = vec3<f32>(1.5220409631729126f);
    set_v_0(i_7, u32(8), _S150 * va_0(i_7, u32(10)) - _S151 * ma_0, _S150 * vl_0(i_7, u32(10)) - _S151 * ml_0);
    return;
}

var<workgroup> g_wide_max_0 : array<f32, i32(1024)>;

var<workgroup> g_wide_max_out_0 : f32;

fn wide_group_max_0( tid_21 : u32,  v_16 : f32) -> f32
{
    g_wide_max_0[tid_21] = v_16;
    workgroupBarrier();
    var k_18 : u32 = u32(512);
    loop
    {
        if(k_18 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_21 < k_18)
        {
            g_wide_max_0[tid_21] = max(g_wide_max_0[tid_21], g_wide_max_0[tid_21 + k_18]);
        }
        workgroupBarrier();
        k_18 = (k_18 >> (u32(1)));
    }
    if(tid_21 == u32(0))
    {
        g_wide_max_out_0 = g_wide_max_0[i32(0)];
    }
    var _S152 : f32 = (workgroupUniformLoad(&((g_wide_max_out_0))));
    workgroupBarrier();
    return _S152;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_max(@builtin(workgroup_id) group_17 : vec3<u32>, @builtin(local_invocation_id) thread_12 : vec3<u32>)
{
    var tid_22 : u32 = thread_12.x;
    var b_21 : Block_0 = node_block_0(group_17, tid_22);
    var _S153 : bool;
    if((b_21.begin_0) >= (b_21.end_0))
    {
        _S153 = true;
    }
    else
    {
        _S153 = !runs_0(b_21.flags_0, false);
    }
    if(_S153)
    {
        return;
    }
    var p_11 : u32 = b_21.begin_0 + tid_22;
    var m_3 : f32;
    if(p_11 < (b_21.end_0))
    {
        var i_8 : u32 = lists_0[p_11];
        var a_10 : vec3<f32> = abs(va_0(i_8, u32(8)));
        var l_8 : vec3<f32> = abs(vl_0(i_8, u32(8)));
        m_3 = max(max(max(a_10.x, a_10.y), max(a_10.z, l_8.x)), max(l_8.y, l_8.z));
    }
    else
    {
        m_3 = 0.0f;
    }
    var mm_0 : f32 = wide_group_max_0(tid_22, m_3);
    if(tid_22 == u32(0))
    {
        wide_0[pt_0(b_21.row_4, group_17.x, u32(4))] = vec4<f32>(mm_0, 0.0f, 0.0f, 0.0f);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_mag(@builtin(workgroup_id) group_18 : vec3<u32>, @builtin(local_invocation_id) thread_13 : vec3<u32>)
{
    var tid_23 : u32 = thread_13.x;
    var row_17 : u32 = group_18.x;
    var f_12 : u32 = group_uniform_0(flags_of_0(row_17), tid_23);
    if(((f_12 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_11 : vec4<u32> = row_range_0(tid_23, row_17);
    var _S154 : u32 = blocks_0(r_11.y - r_11.x);
    var m_4 : f32 = 0.0f;
    var blk_0 : u32 = tid_23;
    loop
    {
        if(blk_0 < _S154)
        {
        }
        else
        {
            break;
        }
        var _S155 : f32 = max(m_4, wide_0[pt_0(row_17, blk_0, u32(4))].x);
        var blk_1 : u32 = blk_0 + u32(1024);
        m_4 = _S155;
        blk_0 = blk_1;
    }
    var mag_0 : f32 = wide_group_max_0(tid_23, m_4);
    if(tid_23 != u32(0))
    {
        return;
    }
    var _S156 : bool;
    if(!(mag_0 > 0.0f))
    {
        _S156 = true;
    }
    else
    {
        _S156 = !isfinite_0(mag_0);
    }
    if(_S156)
    {
        set_flags_0(row_17, (((f_12 & (u32(4294967294)))) | (u32(4))));
        return;
    }
    var _S157 : u32 = st_0(row_17, u32(1));
    var s1_3 : vec4<f32> = wide_0[_S157];
    s1_3[i32(1)] = 1.0f / mag_0;
    wide_0[_S157] = s1_3;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_scale(@builtin(workgroup_id) group_19 : vec3<u32>, @builtin(local_invocation_id) thread_14 : vec3<u32>)
{
    var tid_24 : u32 = thread_14.x;
    var b_22 : Block_0 = node_block_0(group_19, tid_24);
    var _S158 : bool;
    if((b_22.begin_0) >= (b_22.end_0))
    {
        _S158 = true;
    }
    else
    {
        _S158 = !runs_0(b_22.flags_0, false);
    }
    if(_S158)
    {
        return;
    }
    var p_12 : u32 = b_22.begin_0 + tid_24;
    var gl_0 : f32;
    if(p_12 < (b_22.end_0))
    {
        var i_9 : u32 = lists_0[p_12];
        var _S159 : vec3<f32> = vec3<f32>(wide_0[st_0(b_22.row_4, u32(1))].y);
        var ga_0 : vec3<f32> = va_0(i_9, u32(8)) * _S159;
        var gn_0 : vec3<f32> = vl_0(i_9, u32(8)) * _S159;
        set_v_0(i_9, u32(8), ga_0, gn_0);
        gl_0 = dot(va_0(i_9, u32(0)), ga_0) + dot(vl_0(i_9, u32(0)), gn_0);
    }
    else
    {
        gl_0 = 0.0f;
    }
    write_partial_0(tid_24, b_22.row_4, group_19.x, u32(0), vec4<f32>(gl_0, 0.0f, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_gamma(@builtin(workgroup_id) group_20 : vec3<u32>, @builtin(local_invocation_id) thread_15 : vec3<u32>)
{
    var tid_25 : u32 = thread_15.x;
    var row_18 : u32 = group_20.x;
    var f_13 : u32 = group_uniform_0(flags_of_0(row_18), tid_25);
    if(((f_13 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_12 : vec4<u32> = row_range_0(tid_25, row_18);
    var _S160 : vec4<f32> = row_sum_0(tid_25, row_18, blocks_0(r_12.y - r_12.x), u32(0));
    var gamma_0 : f32 = _S160.x;
    if(tid_25 != u32(0))
    {
        return;
    }
    var _S161 : bool;
    if(!(gamma_0 > 0.0f))
    {
        _S161 = true;
    }
    else
    {
        _S161 = !isfinite_0(gamma_0);
    }
    if(_S161)
    {
        set_flags_0(row_18, (((f_13 & (u32(4294967294)))) | (u32(4))));
        return;
    }
    var s0_4 : vec4<f32> = wide_0[st_0(row_18, u32(0))];
    var it_2 : u32 = iteration_of_0(row_18);
    if(it_2 > u32(1))
    {
        _S161 = true;
    }
    else
    {
        if((params_0.first_polynomial_0) != u32(0))
        {
            _S161 = it_2 > u32(0);
        }
        else
        {
            _S161 = false;
        }
    }
    if(_S161)
    {
        _S161 = (s0_4.z) > 0.0f;
    }
    else
    {
        _S161 = false;
    }
    var beta_0 : f32;
    if(_S161)
    {
        beta_0 = gamma_0 / s0_4.z;
    }
    else
    {
        beta_0 = 0.0f;
    }
    var _S162 : u32 = st_0(row_18, u32(1));
    var s1_4 : vec4<f32> = wide_0[_S162];
    s1_4[i32(0)] = gamma_0;
    s1_4[i32(2)] = beta_0;
    wide_0[_S162] = s1_4;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_dir(@builtin(workgroup_id) group_21 : vec3<u32>, @builtin(local_invocation_id) thread_16 : vec3<u32>)
{
    var tid_26 : u32 = thread_16.x;
    var b_23 : Block_0 = node_block_0(group_21, tid_26);
    var _S163 : bool;
    if((b_23.begin_0) >= (b_23.end_0))
    {
        _S163 = true;
    }
    else
    {
        _S163 = !runs_0(b_23.flags_0, false);
    }
    if(_S163)
    {
        return;
    }
    var p_13 : u32 = b_23.begin_0 + tid_26;
    if(p_13 >= (b_23.end_0))
    {
        return;
    }
    var i_10 : u32 = lists_0[p_13];
    var beta_1 : f32 = wide_0[st_0(b_23.row_4, u32(1))].z;
    if(beta_1 == 0.0f)
    {
        set_v_0(i_10, u32(2), va_0(i_10, u32(8)), vl_0(i_10, u32(8)));
    }
    else
    {
        var _S164 : vec3<f32> = vec3<f32>(beta_1);
        set_v_0(i_10, u32(2), va_0(i_10, u32(8)) + va_0(i_10, u32(2)) * _S164, vl_0(i_10, u32(8)) + vl_0(i_10, u32(2)) * _S164);
    }
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_bond_p(@builtin(workgroup_id) group_22 : vec3<u32>, @builtin(local_invocation_id) thread_17 : vec3<u32>)
{
    var tid_27 : u32 = thread_17.x;
    var b_24 : Block_0 = bond_block_0(group_22, tid_27);
    var _S165 : bool;
    if((b_24.begin_0) >= (b_24.end_0))
    {
        _S165 = true;
    }
    else
    {
        _S165 = !runs_0(b_24.flags_0, false);
    }
    if(_S165)
    {
        return;
    }
    var p_14 : u32 = b_24.begin_0 + tid_27;
    var z_6 : f32;
    if(p_14 < (b_24.end_0))
    {
        var j_14 : u32 = lists_0[p_14];
        var _S166 : BlastBond_std430_0 = bonds_0[j_14];
        var _S167 : vec3<f32> = vec3<f32>(0.0f);
        var ta_5 : vec3<f32> = _S167;
        var tl_5 : vec3<f32> = _S167;
        if((_S166.health_0) > 0.0f)
        {
            bond_term_1(&(_S166), j_14, u32(2), &(ta_5), &(tl_5));
        }
        var _S168 : f32 = norm2_0(ta_5, tl_5);
        var _S169 : u32 = bs_0(j_14);
        scratch_0[_S169] = vec4<f32>(ta_5, 0.0f);
        scratch_0[_S169 + u32(1)] = vec4<f32>(tl_5, 0.0f);
        z_6 = _S168;
    }
    else
    {
        z_6 = 0.0f;
    }
    write_partial_0(tid_27, b_24.row_4, group_22.x, u32(1), vec4<f32>(z_6, 0.0f, 0.0f, 0.0f));
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_node_q(@builtin(workgroup_id) group_23 : vec3<u32>, @builtin(local_invocation_id) thread_18 : vec3<u32>)
{
    var tid_28 : u32 = thread_18.x;
    var b_25 : Block_0 = node_block_0(group_23, tid_28);
    var _S170 : bool;
    if((b_25.begin_0) >= (b_25.end_0))
    {
        _S170 = true;
    }
    else
    {
        _S170 = !runs_0(b_25.flags_0, false);
    }
    if(_S170)
    {
        return;
    }
    var p_15 : u32 = b_25.begin_0 + tid_28;
    if(p_15 >= (b_25.end_0))
    {
        return;
    }
    var i_11 : u32 = lists_0[p_15];
    var a_11 : vec3<f32>;
    var l_9 : vec3<f32>;
    gather_0(i_11, true, false, &(a_11), &(l_9));
    set_v_0(i_11, u32(4), a_11, l_9);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_den(@builtin(workgroup_id) group_24 : vec3<u32>, @builtin(local_invocation_id) thread_19 : vec3<u32>)
{
    var tid_29 : u32 = thread_19.x;
    var row_19 : u32 = group_24.x;
    var f_14 : u32 = group_uniform_0(flags_of_0(row_19), tid_29);
    if(((f_14 & (u32(1)))) == u32(0))
    {
        return;
    }
    var r_13 : vec4<u32> = row_range_0(tid_29, row_19);
    var _S171 : vec4<f32> = row_sum_0(tid_29, row_19, blocks_0(r_13.w - r_13.z), u32(1));
    var den_0 : f32 = _S171.x;
    if(tid_29 != u32(0))
    {
        return;
    }
    var _S172 : u32 = st_0(row_19, u32(0));
    var s0_5 : vec4<f32> = wide_0[_S172];
    var _S173 : u32 = st_0(row_19, u32(1));
    var s1_5 : vec4<f32> = wide_0[_S173];
    var _S174 : u32 = st_0(row_19, u32(2));
    var cnt_2 : vec4<f32> = wide_0[_S174];
    var gamma_1 : f32 = wide_0[_S173].x;
    var _S175 : bool;
    if((params_0.force_tolerance_0) > 0.0f)
    {
        if(den_0 > 0.0f)
        {
            _S175 = gamma_1 > 0.0f;
        }
        else
        {
            _S175 = false;
        }
        var _S176 : f32;
        if(_S175)
        {
            _S176 = gamma_1 / sqrt(den_0);
        }
        else
        {
            _S176 = float_bits_0(u32(2139095040));
        }
        s1_5[i32(3)] = _S176;
        if(isfinite_0(s1_5.w))
        {
            cnt_2[i32(1)] = cnt_2[i32(1)] + s1_5.w;
        }
        if((bitcast<u32>((cnt_2.z))) > u32(0))
        {
            _S175 = true;
        }
        else
        {
            _S175 = (params_0.first_polynomial_0) != u32(0);
        }
        if(!_S175)
        {
            s1_5[i32(3)] = float_bits_0(u32(2139095040));
        }
    }
    s0_5[i32(2)] = gamma_1;
    cnt_2[i32(2)] = (bitcast<f32>(((bitcast<u32>((cnt_2.z))) + u32(1))));
    if(!(den_0 > 0.0f))
    {
        _S175 = true;
    }
    else
    {
        _S175 = !isfinite_0(den_0);
    }
    var f_15 : u32;
    if(_S175)
    {
        var f_16 : u32 = (f_14 & (u32(4294967294)));
        s1_5[i32(1)] = 0.0f;
        f_15 = f_16;
    }
    else
    {
        s1_5[i32(1)] = gamma_1 / den_0;
        f_15 = f_14;
    }
    wide_0[_S172] = s0_5;
    wide_0[_S173] = s1_5;
    cnt_2[i32(3)] = (bitcast<f32>((f_15)));
    wide_0[_S174] = cnt_2;
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_update(@builtin(workgroup_id) group_25 : vec3<u32>, @builtin(local_invocation_id) thread_20 : vec3<u32>)
{
    var tid_30 : u32 = thread_20.x;
    var b_26 : Block_0 = node_block_0(group_25, tid_30);
    var _S177 : bool;
    if((b_26.begin_0) >= (b_26.end_0))
    {
        _S177 = true;
    }
    else
    {
        _S177 = !runs_0(b_26.flags_0, false);
    }
    if(_S177)
    {
        return;
    }
    var p_16 : u32 = b_26.begin_0 + tid_30;
    if(p_16 >= (b_26.end_0))
    {
        return;
    }
    var i_12 : u32 = lists_0[p_16];
    var _S178 : vec3<f32> = vec3<f32>(wide_0[st_0(b_26.row_4, u32(1))].y);
    set_v_0(i_12, u32(6), va_0(i_12, u32(6)) + va_0(i_12, u32(2)) * _S178, vl_0(i_12, u32(6)) + vl_0(i_12, u32(2)) * _S178);
    set_v_0(i_12, u32(0), va_0(i_12, u32(0)) - va_0(i_12, u32(4)) * _S178, vl_0(i_12, u32(0)) - vl_0(i_12, u32(4)) * _S178);
    return;
}

@compute
@workgroup_size(1024, 1, 1)
fn w_final(@builtin(workgroup_id) group_26 : vec3<u32>, @builtin(local_invocation_id) thread_21 : vec3<u32>)
{
    var tid_31 : u32 = thread_21.x;
    var b_27 : Block_0 = bond_block_0(group_26, tid_31);
    var _S179 : bool;
    if((b_27.begin_0) >= (b_27.end_0))
    {
        _S179 = true;
    }
    else
    {
        _S179 = (((b_27.flags_0) & (u32(128)))) != u32(0);
    }
    if(_S179)
    {
        return;
    }
    var p_17 : u32 = b_27.begin_0 + tid_31;
    if(p_17 >= (b_27.end_0))
    {
        return;
    }
    var j_15 : u32 = lists_0[p_17];
    var _S180 : BlastBond_std430_0 = bonds_0[j_15];
    if(!((_S180.health_0) > 0.0f))
    {
        _S179 = true;
    }
    else
    {
        _S179 = (((b_27.flags_0) & (u32(16)))) != u32(0);
    }
    if(_S179)
    {
        var _S181 : u32 = j_15 * u32(2);
        var _S182 : vec4<f32> = vec4<f32>(0.0f);
        x_0[_S181] = _S182;
        x_0[_S181 + u32(1)] = _S182;
        return;
    }
    var ta_6 : vec3<f32>;
    var tl_6 : vec3<f32>;
    bond_term_1(&(_S180), j_15, u32(6), &(ta_6), &(tl_6));
    var _S183 : u32 = j_15 * u32(2);
    x_0[_S183] = vec4<f32>(x_0[_S183].xyz + ta_6, 0.0f);
    var _S184 : u32 = _S183 + u32(1);
    x_0[_S184] = vec4<f32>(x_0[_S184].xyz + tl_6, 0.0f);
    return;
}

@compute
@workgroup_size(1, 1, 1)
fn w_report(@builtin(workgroup_id) group_27 : vec3<u32>)
{
    var row_20 : u32 = group_27.x;
    var island_2 : u32 = row_island_0(row_20);
    var f_17 : u32 = flags_of_0(row_20);
    var last_1 : vec4<u32> = report_0[island_2];
    if(((f_17 & (u32(128)))) != u32(0))
    {
        report_0[island_2] = vec4<u32>(u32(0), u32(7), last_1.z, last_1.w);
        return;
    }
    var ok_0 : bool;
    if(((f_17 & (u32(2)))) != u32(0))
    {
        ok_0 = ((f_17 & (u32(4)))) == u32(0);
    }
    else
    {
        ok_0 = false;
    }
    var it_3 : u32 = iteration_of_0(row_20);
    var _S185 : u32;
    if(ok_0)
    {
        _S185 = u32(1);
    }
    else
    {
        _S185 = u32(0);
    }
    if(ok_0)
    {
        ok_0 = (params_0.warm_0) != u32(0);
    }
    else
    {
        ok_0 = false;
    }
    if(ok_0)
    {
        ok_0 = it_3 == u32(0);
    }
    else
    {
        ok_0 = false;
    }
    var _S186 : u32;
    if(ok_0)
    {
        _S186 = u32(4);
    }
    else
    {
        _S186 = u32(0);
    }
    report_0[island_2] = vec4<u32>(it_3, (_S185 | (_S186)), (bitcast<u32>((wide_0[st_0(row_20, u32(0))].y))), (bitcast<u32>((wide_0[st_0(row_20, u32(0))].x))));
    return;
}

