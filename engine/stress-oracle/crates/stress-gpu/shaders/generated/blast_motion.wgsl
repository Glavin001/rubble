@binding(1) @group(0) var<storage, read_write> islands_0 : array<vec4<u32>>;

@binding(5) @group(0) var<storage, read> lists_0 : array<u32>;

struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(4) @group(0) var<storage, read> bonds_0 : array<BlastBond_std430_0>;

struct BlastNode_std430_0
{
    @align(16) w_ang_0 : f32,
    @align(4) w_lin_0 : f32,
    @align(8) ref_begin_0 : u32,
    @align(4) ref_end_0 : u32,
    @align(16) position_0 : vec4<f32>,
};

@binding(2) @group(0) var<storage, read_write> nodes_0 : array<BlastNode_std430_0>;

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
    @align(8) a_events_0 : u32,
    @align(4) ev_broken_cap_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
@binding(6) @group(0) var<storage, read_write> motion_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> refs_0 : array<u32>;

fn isnan_0( x_0 : f32) -> bool
{
    var _S1 : u32 = (bitcast<u32>((x_0)));
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

fn isinf_0( x_1 : f32) -> bool
{
    var _S4 : u32 = (bitcast<u32>((x_1)));
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

var<workgroup> g_group_uniform4_0 : vec4<u32>;

fn group_uniform4_0( value_0 : vec4<u32>,  thread_0 : u32) -> vec4<u32>
{
    if(thread_0 == u32(0))
    {
        g_group_uniform4_0 = value_0;
    }
    var _S7 : vec4<u32> = (workgroupUniformLoad(&((g_group_uniform4_0))));
    return _S7;
}

fn blast_frame_0( node_count_1 : u32,  k_0 : u32) -> u32
{
    return node_count_1 + u32(10) * k_0;
}

fn set_frame_0( k_1 : u32,  w_0 : u32,  value_1 : u32)
{
    var at_0 : u32 = blast_frame_0(params_0.node_count_0, k_1) + w_0 / u32(4);
    var v_0 : vec4<u32> = islands_0[at_0];
    v_0[w_0 % u32(4)] = value_1;
    islands_0[at_0] = v_0;
    return;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

var<workgroup> g_blast_red_0 : array<vec2<f32>, i32(1024)>;

var<workgroup> g_blast_sum_0 : vec2<f32>;

fn blast_group_sum_0( tid_0 : u32,  v_1 : vec2<f32>) -> vec2<f32>
{
    g_blast_red_0[tid_0] = v_1;
    workgroupBarrier();
    var k_2 : u32 = u32(512);
    loop
    {
        if(k_2 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < k_2)
        {
            g_blast_red_0[tid_0] = g_blast_red_0[tid_0] + g_blast_red_0[tid_0 + k_2];
        }
        workgroupBarrier();
        k_2 = (k_2 >> (u32(1)));
    }
    if(tid_0 == u32(0))
    {
        g_blast_sum_0 = g_blast_red_0[i32(0)];
    }
    var s_1 : vec2<f32> = (workgroupUniformLoad(&((g_blast_sum_0))));
    workgroupBarrier();
    return s_1;
}

struct Ex_0
{
     t_0 : array<f32, i32(8)>,
};

fn ex_zero_0() -> Ex_0
{
    var r_0 : Ex_0;
    var i_0 : u32 = u32(0);
    loop
    {
        if(i_0 < u32(8))
        {
        }
        else
        {
            break;
        }
        r_0.t_0[i_0] = 0.0f;
        i_0 = i_0 + u32(1);
    }
    return r_0;
}

fn store_ex_0( base_0 : u32,  e_0 : Ex_0)
{
    motion_0[base_0] = vec4<f32>(e_0.t_0[i32(0)], e_0.t_0[i32(1)], e_0.t_0[i32(2)], e_0.t_0[i32(3)]);
    motion_0[base_0 + u32(1)] = vec4<f32>(e_0.t_0[i32(4)], e_0.t_0[i32(5)], e_0.t_0[i32(6)], e_0.t_0[i32(7)]);
    return;
}

struct Ex3_0
{
     x_2 : Ex_0,
     y_0 : Ex_0,
     z_0 : Ex_0,
};

fn set_position_0( n_0 : u32,  p_0 : Ex3_0)
{
    var _S8 : u32 = n_0 * u32(7);
    store_ex_0(_S8, p_0.x_2);
    store_ex_0(_S8 + u32(2), p_0.y_0);
    store_ex_0(_S8 + u32(4), p_0.z_0);
    return;
}

fn set_level_0( n_1 : u32,  l_0 : u32)
{
    motion_0[n_1 * u32(7) + u32(6)] = vec4<f32>((bitcast<f32>((l_0))), 0.0f, 0.0f, 0.0f);
    return;
}

fn level_0( n_2 : u32) -> u32
{
    return (bitcast<u32>((motion_0[n_2 * u32(7) + u32(6)].x)));
}

fn opaque_0( x_3 : f32) -> f32
{
    return (bitcast<f32>((((bitcast<u32>((x_3))) ^ ((params_0.opaque_zero_0))))));
}

fn two_sum_0( a_0 : f32,  b_0 : f32,  s_2 : ptr<function, f32>,  e_1 : ptr<function, f32>)
{
    var _S9 : f32 = opaque_0(a_0 + b_0);
    (*s_2) = _S9;
    var v_2 : f32 = opaque_0(_S9 - a_0);
    (*e_1) = opaque_0(a_0 - opaque_0(_S9 - v_2)) + opaque_0(b_0 - v_2);
    return;
}

fn two_sum_1( a_1 : f32,  b_1 : f32,  s_3 : ptr<function, f32>,  e_2 : ptr<function, f32>)
{
    var _S10 : f32 = opaque_0(a_1 + b_1);
    (*s_3) = _S10;
    var v_3 : f32 = opaque_0(_S10 - a_1);
    (*e_2) = opaque_0(a_1 - opaque_0(_S10 - v_3)) + opaque_0(b_1 - v_3);
    return;
}

fn two_sum_2( a_2 : f32,  b_2 : f32,  s_4 : ptr<function, f32>,  e_3 : ptr<function, f32>)
{
    var _S11 : f32 = opaque_0(a_2 + b_2);
    (*s_4) = _S11;
    var v_4 : f32 = opaque_0(_S11 - a_2);
    (*e_3) = opaque_0(a_2 - opaque_0(_S11 - v_4)) + opaque_0(b_2 - v_4);
    return;
}

fn two_sum_3( a_3 : f32,  b_3 : f32,  s_5 : ptr<function, f32>,  e_4 : ptr<function, f32>)
{
    var _S12 : f32 = opaque_0(a_3 + b_3);
    (*s_5) = _S12;
    var v_5 : f32 = opaque_0(_S12 - a_3);
    (*e_4) = opaque_0(a_3 - opaque_0(_S12 - v_5)) + opaque_0(b_3 - v_5);
    return;
}

fn ex_representable_0( a_4 : Ex_0) -> bool
{
    var i_1 : u32 = u32(0);
    loop
    {
        if(i_1 < u32(8))
        {
        }
        else
        {
            break;
        }
        var _S13 : u32 = i_1;
        if((a_4.t_0[i_1]) != 0.0f)
        {
            var m_0 : f32 = abs(a_4.t_0[_S13]);
            var _S14 : bool;
            if(!(m_0 <= 1.15292150460684698e+18f))
            {
                _S14 = true;
            }
            else
            {
                _S14 = m_0 < 7.88860905221011805e-31f;
            }
            if(_S14)
            {
                return false;
            }
        }
        i_1 = i_1 + u32(1);
    }
    return true;
}

fn ex_difference_0( a_5 : f32,  b_4 : f32,  exact_0 : ptr<function, bool>) -> Ex_0
{
    var _S15 : bool;
    if(isinf_0(a_5))
    {
        _S15 = true;
    }
    else
    {
        _S15 = isnan_0(a_5);
    }
    if(_S15)
    {
        _S15 = true;
    }
    else
    {
        _S15 = isinf_0(b_4);
    }
    if(_S15)
    {
        _S15 = true;
    }
    else
    {
        _S15 = isnan_0(b_4);
    }
    if(_S15)
    {
        (*exact_0) = false;
        return ex_zero_0();
    }
    var r_1 : Ex_0 = ex_zero_0();
    var _S16 : f32 = - b_4;
    var _S17 : f32 = r_1.t_0[i32(0)];
    var _S18 : f32 = r_1.t_0[i32(1)];
    two_sum_3(a_5, _S16, &(_S17), &(_S18));
    r_1.t_0[i32(0)] = _S17;
    r_1.t_0[i32(1)] = _S18;
    if(ex_representable_0(r_1))
    {
        _S15 = (*exact_0);
    }
    else
    {
        _S15 = false;
    }
    (*exact_0) = _S15;
    return r_1;
}

fn offset_0( b_5 : ptr<function, BlastBond_std430_0>,  exact_1 : ptr<function, bool>) -> Ex3_0
{
    var r_2 : Ex3_0;
    var _S19 : vec4<f32> = (*b_5).o0_0;
    var _S20 : vec4<f32> = (*b_5).o1_0;
    var _S21 : Ex_0 = ex_difference_0((*b_5).o0_0.x, (*b_5).o1_0.x, &((*exact_1)));
    r_2.x_2 = _S21;
    var _S22 : Ex_0 = ex_difference_0(_S19.y, _S20.y, &((*exact_1)));
    r_2.y_0 = _S22;
    var _S23 : Ex_0 = ex_difference_0(_S19.z, _S20.z, &((*exact_1)));
    r_2.z_0 = _S23;
    return r_2;
}

fn offset_1( b_6 : ptr<function, BlastBond_std430_0>,  exact_2 : ptr<function, bool>) -> Ex3_0
{
    var r_3 : Ex3_0;
    var _S24 : vec4<f32> = (*b_6).o0_0;
    var _S25 : vec4<f32> = (*b_6).o1_0;
    var _S26 : Ex_0 = ex_difference_0((*b_6).o0_0.x, (*b_6).o1_0.x, &((*exact_2)));
    r_3.x_2 = _S26;
    var _S27 : Ex_0 = ex_difference_0(_S24.y, _S25.y, &((*exact_2)));
    r_3.y_0 = _S27;
    var _S28 : Ex_0 = ex_difference_0(_S24.z, _S25.z, &((*exact_2)));
    r_3.z_0 = _S28;
    return r_3;
}

fn load_ex_0( base_1 : u32) -> Ex_0
{
    var r_4 : Ex_0;
    r_4.t_0[i32(0)] = motion_0[base_1].x;
    r_4.t_0[i32(1)] = motion_0[base_1].y;
    r_4.t_0[i32(2)] = motion_0[base_1].z;
    r_4.t_0[i32(3)] = motion_0[base_1].w;
    r_4.t_0[i32(4)] = motion_0[base_1 + u32(1)].x;
    r_4.t_0[i32(5)] = motion_0[base_1 + u32(1)].y;
    r_4.t_0[i32(6)] = motion_0[base_1 + u32(1)].z;
    r_4.t_0[i32(7)] = motion_0[base_1 + u32(1)].w;
    return r_4;
}

fn position_1( n_3 : u32) -> Ex3_0
{
    var r_5 : Ex3_0;
    var _S29 : u32 = n_3 * u32(7);
    r_5.x_2 = load_ex_0(_S29);
    r_5.y_0 = load_ex_0(_S29 + u32(2));
    r_5.z_0 = load_ex_0(_S29 + u32(4));
    return r_5;
}

fn ex_distill_0( v_6 : ptr<function, array<f32, i32(16)>>,  count_0 : u32,  out_ex_0 : ptr<function, Ex_0>) -> bool
{
    var pass_0 : u32;
    var i_2 : u32;
    (*out_ex_0) = ex_zero_0();
    var exact_3 : bool = true;
    var term_0 : u32 = u32(0);
    loop
    {
        if(term_0 < count_0)
        {
        }
        else
        {
            break;
        }
        pass_0 = term_0;
        loop
        {
            var _S30 : u32 = pass_0 + u32(1);
            if(_S30 < count_0)
            {
            }
            else
            {
                break;
            }
            i_2 = count_0 - u32(1);
            loop
            {
                if(i_2 > term_0)
                {
                }
                else
                {
                    break;
                }
                var _S31 : u32 = i_2 - u32(1);
                var s_6 : f32;
                var e_5 : f32;
                two_sum_3((*v_6)[_S31], (*v_6)[i_2], &(s_6), &(e_5));
                (*v_6)[_S31] = s_6;
                (*v_6)[i_2] = e_5;
                i_2 = _S31;
            }
            pass_0 = _S30;
        }
        if(term_0 < u32(8))
        {
            (*out_ex_0).t_0[term_0] = (*v_6)[term_0];
        }
        else
        {
            var exact_4 : bool;
            if(((*v_6)[term_0]) != 0.0f)
            {
                exact_4 = false;
            }
            else
            {
                exact_4 = exact_3;
            }
            exact_3 = exact_4;
        }
        term_0 = term_0 + u32(1);
    }
    pass_0 = u32(0);
    loop
    {
        var _S32 : u32 = pass_0 + u32(1);
        if(_S32 < u32(8))
        {
        }
        else
        {
            break;
        }
        i_2 = u32(0);
        loop
        {
            var _S33 : u32 = i_2 + u32(1);
            if(_S33 < u32(8))
            {
            }
            else
            {
                break;
            }
            if(((*out_ex_0).t_0[i_2]) == 0.0f)
            {
                (*out_ex_0).t_0[i_2] = (*out_ex_0).t_0[_S33];
                (*out_ex_0).t_0[_S33] = 0.0f;
            }
            i_2 = _S33;
        }
        pass_0 = _S32;
    }
    if(exact_3)
    {
        exact_3 = ex_representable_0((*out_ex_0));
    }
    else
    {
        exact_3 = false;
    }
    return exact_3;
}

fn ex_add_0( a_6 : Ex_0,  b_7 : Ex_0,  exact_5 : ptr<function, bool>) -> Ex_0
{
    var v_7 : array<f32, i32(16)>;
    var i_3 : u32 = u32(0);
    loop
    {
        if(i_3 < u32(16))
        {
        }
        else
        {
            break;
        }
        v_7[i_3] = 0.0f;
        i_3 = i_3 + u32(1);
    }
    var wide_0 : bool = false;
    i_3 = u32(3);
    loop
    {
        if(i_3 < u32(8))
        {
        }
        else
        {
            break;
        }
        var _S34 : bool;
        if((a_6.t_0[i_3]) != 0.0f)
        {
            _S34 = true;
        }
        else
        {
            _S34 = (b_7.t_0[i_3]) != 0.0f;
        }
        if(_S34)
        {
            wide_0 = true;
        }
        i_3 = i_3 + u32(1);
    }
    var count_1 : u32;
    if(wide_0)
    {
        i_3 = u32(0);
        loop
        {
            if(i_3 < u32(8))
            {
            }
            else
            {
                break;
            }
            var _S35 : u32 = u32(2) * i_3;
            v_7[_S35] = a_6.t_0[i_3];
            v_7[_S35 + u32(1)] = b_7.t_0[i_3];
            i_3 = i_3 + u32(1);
        }
        count_1 = u32(16);
    }
    else
    {
        if((a_6.t_0[i32(2)]) != 0.0f)
        {
            wide_0 = true;
        }
        else
        {
            wide_0 = (b_7.t_0[i32(2)]) != 0.0f;
        }
        if(wide_0)
        {
            v_7[i32(0)] = a_6.t_0[i32(0)];
            v_7[i32(1)] = b_7.t_0[i32(0)];
            v_7[i32(2)] = a_6.t_0[i32(1)];
            v_7[i32(3)] = b_7.t_0[i32(1)];
            v_7[i32(4)] = a_6.t_0[i32(2)];
            v_7[i32(5)] = b_7.t_0[i32(2)];
            count_1 = u32(6);
        }
        else
        {
            v_7[i32(0)] = a_6.t_0[i32(0)];
            v_7[i32(1)] = b_7.t_0[i32(0)];
            v_7[i32(2)] = a_6.t_0[i32(1)];
            v_7[i32(3)] = b_7.t_0[i32(1)];
            count_1 = u32(4);
        }
    }
    var r_6 : Ex_0;
    var _S36 : bool = ex_distill_0(&(v_7), count_1, &(r_6));
    if(_S36)
    {
        wide_0 = (*exact_5);
    }
    else
    {
        wide_0 = false;
    }
    (*exact_5) = wide_0;
    return r_6;
}

fn ex3_add_0( a_7 : Ex3_0,  b_8 : Ex3_0,  exact_6 : ptr<function, bool>) -> Ex3_0
{
    var r_7 : Ex3_0;
    var _S37 : Ex_0 = ex_add_0(a_7.x_2, b_8.x_2, &((*exact_6)));
    r_7.x_2 = _S37;
    var _S38 : Ex_0 = ex_add_0(a_7.y_0, b_8.y_0, &((*exact_6)));
    r_7.y_0 = _S38;
    var _S39 : Ex_0 = ex_add_0(a_7.z_0, b_8.z_0, &((*exact_6)));
    r_7.z_0 = _S39;
    return r_7;
}

fn ex_neg_0( a_8 : Ex_0) -> Ex_0
{
    var _S40 : Ex_0 = a_8;
    var i_4 : u32 = u32(0);
    loop
    {
        if(i_4 < u32(8))
        {
        }
        else
        {
            break;
        }
        _S40.t_0[i_4] = - _S40.t_0[i_4];
        i_4 = i_4 + u32(1);
    }
    return _S40;
}

fn ex3_neg_0( a_9 : Ex3_0) -> Ex3_0
{
    var r_8 : Ex3_0;
    r_8.x_2 = ex_neg_0(a_9.x_2);
    r_8.y_0 = ex_neg_0(a_9.y_0);
    r_8.z_0 = ex_neg_0(a_9.z_0);
    return r_8;
}

fn closure_0( b_9 : ptr<function, BlastBond_std430_0>,  exact_7 : ptr<function, bool>) -> Ex3_0
{
    var _S41 : Ex3_0 = position_1((*b_9).n0_0);
    var _S42 : Ex3_0 = offset_1(&((*b_9)), &((*exact_7)));
    var _S43 : Ex3_0 = ex3_add_0(_S41, _S42, &((*exact_7)));
    var _S44 : Ex3_0 = ex3_add_0(_S43, ex3_neg_0(position_1((*b_9).n1_0)), &((*exact_7)));
    return _S44;
}

fn ex_nonzero_0( a_10 : Ex_0) -> bool
{
    return (a_10.t_0[i32(0)]) != 0.0f;
}

fn ex3_nonzero_0( a_11 : Ex3_0) -> bool
{
    var _S45 : bool;
    if(ex_nonzero_0(a_11.x_2))
    {
        _S45 = true;
    }
    else
    {
        _S45 = ex_nonzero_0(a_11.y_0);
    }
    if(_S45)
    {
        _S45 = true;
    }
    else
    {
        _S45 = ex_nonzero_0(a_11.z_0);
    }
    return _S45;
}

fn ex_magnitude_0( a_12 : Ex_0) -> Ex_0
{
    var _S46 : Ex_0;
    if((a_12.t_0[i32(0)]) < 0.0f)
    {
        _S46 = ex_neg_0(a_12);
    }
    else
    {
        _S46 = a_12;
    }
    return _S46;
}

fn ex_greater_0( a_13 : Ex_0,  b_10 : Ex_0) -> bool
{
    var v_8 : array<f32, i32(16)>;
    var i_5 : u32 = u32(0);
    loop
    {
        if(i_5 < u32(8))
        {
        }
        else
        {
            break;
        }
        var _S47 : u32 = u32(2) * i_5;
        v_8[_S47] = a_13.t_0[i_5];
        v_8[_S47 + u32(1)] = - b_10.t_0[i_5];
        i_5 = i_5 + u32(1);
    }
    var pass_1 : u32 = u32(0);
    loop
    {
        var _S48 : u32 = pass_1 + u32(1);
        if(_S48 < u32(16))
        {
        }
        else
        {
            break;
        }
        i_5 = u32(15);
        loop
        {
            if(i_5 > u32(0))
            {
            }
            else
            {
                break;
            }
            var _S49 : u32 = i_5 - u32(1);
            var s_7 : f32;
            var e_6 : f32;
            two_sum_1(v_8[_S49], v_8[i_5], &(s_7), &(e_6));
            v_8[_S49] = s_7;
            v_8[i_5] = e_6;
            i_5 = _S49;
        }
        pass_1 = _S48;
    }
    i_5 = u32(0);
    loop
    {
        if(i_5 < u32(16))
        {
        }
        else
        {
            break;
        }
        if((v_8[i_5]) != 0.0f)
        {
            return (v_8[i_5]) > 0.0f;
        }
        i_5 = i_5 + u32(1);
    }
    return false;
}

fn ex_max_0( a_14 : Ex_0,  b_11 : Ex_0) -> Ex_0
{
    var _S50 : Ex_0;
    if(ex_greater_0(b_11, a_14))
    {
        _S50 = b_11;
    }
    else
    {
        _S50 = a_14;
    }
    return _S50;
}

fn ex_of_0( f_0 : f32) -> Ex_0
{
    var r_9 : Ex_0 = ex_zero_0();
    r_9.t_0[i32(0)] = f_0;
    return r_9;
}

fn ex_scaled_0( a_15 : Ex_0,  power_0 : f32) -> Ex_0
{
    var _S51 : Ex_0 = a_15;
    var i_6 : u32 = u32(0);
    loop
    {
        if(i_6 < u32(8))
        {
        }
        else
        {
            break;
        }
        _S51.t_0[i_6] = _S51.t_0[i_6] * power_0;
        i_6 = i_6 + u32(1);
    }
    return _S51;
}

fn significant_0( b_12 : ptr<function, BlastBond_std430_0>,  c_0 : Ex3_0) -> bool
{
    var p_1 : Ex3_0 = position_1((*b_12).n0_0);
    var q_0 : Ex3_0 = position_1((*b_12).n1_0);
    var _S52 : f32 = max(max(max(abs((*b_12).o0_0.x), abs((*b_12).o0_0.y)), max(abs((*b_12).o0_0.z), abs((*b_12).o1_0.x))), max(abs((*b_12).o1_0.y), abs((*b_12).o1_0.z)));
    var _S53 : f32 = max(max(abs(c_0.x_2.t_0[i32(0)]), abs(c_0.y_0.t_0[i32(0)])), abs(c_0.z_0.t_0[i32(0)]));
    var threshold_0 : f32 = max(_S52, max(max(max(abs(p_1.x_2.t_0[i32(0)]), abs(p_1.y_0.t_0[i32(0)])), max(abs(p_1.z_0.t_0[i32(0)]), abs(q_0.x_2.t_0[i32(0)]))), max(abs(q_0.y_0.t_0[i32(0)]), abs(q_0.z_0.t_0[i32(0)])))) * 0.00000762939453125f;
    if(_S53 > (threshold_0 * 1.00000095367431641f))
    {
        return true;
    }
    if(_S53 < (threshold_0 * 0.99999904632568359f))
    {
        return false;
    }
    return ex_greater_0(ex_max_0(ex_max_0(ex_magnitude_0(c_0.x_2), ex_magnitude_0(c_0.y_0)), ex_magnitude_0(c_0.z_0)), ex_scaled_0(ex_max_0(ex_max_0(ex_max_0(ex_max_0(ex_magnitude_0(p_1.x_2), ex_magnitude_0(p_1.y_0)), ex_magnitude_0(p_1.z_0)), ex_max_0(ex_max_0(ex_magnitude_0(q_0.x_2), ex_magnitude_0(q_0.y_0)), ex_magnitude_0(q_0.z_0))), ex_of_0(_S52)), 0.00000762939453125f));
}

var<workgroup> g_word_0 : array<u32, i32(1024)>;

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_2 : u32,  thread_1 : u32) -> u32
{
    if(thread_1 == u32(0))
    {
        g_group_uniform_0 = value_2;
    }
    var _S54 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S54;
}

fn word_min_0( tid_1 : u32,  v_9 : u32) -> u32
{
    g_word_0[tid_1] = v_9;
    workgroupBarrier();
    var k_3 : u32 = u32(512);
    loop
    {
        if(k_3 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_1 < k_3)
        {
            g_word_0[tid_1] = min(g_word_0[tid_1], g_word_0[tid_1 + k_3]);
        }
        workgroupBarrier();
        k_3 = (k_3 >> (u32(1)));
    }
    var _S55 : u32 = group_uniform_0(g_word_0[i32(0)], tid_1);
    return _S55;
}

var<workgroup> g_rotations_0 : u32;

fn ex_pair_0( a_16 : Ex_0) -> vec2<f32>
{
    var i_7 : u32 = u32(7);
    var tail_0 : f32 = 0.0f;
    loop
    {
        if(i_7 > u32(0))
        {
        }
        else
        {
            break;
        }
        var tail_1 : f32 = tail_0 + a_16.t_0[i_7];
        i_7 = i_7 - u32(1);
        tail_0 = tail_1;
    }
    var s_8 : f32;
    var e_7 : f32;
    two_sum_2(a_16.t_0[i32(0)], tail_0, &(s_8), &(e_7));
    return vec2<f32>(s_8, e_7);
}

struct P3_0
{
     x_4 : vec2<f32>,
     y_1 : vec2<f32>,
     z_1 : vec2<f32>,
};

fn ex3_pair_0( a_17 : Ex3_0) -> P3_0
{
    var r_10 : P3_0;
    r_10.x_4 = ex_pair_0(a_17.x_2);
    r_10.y_1 = ex_pair_0(a_17.y_0);
    r_10.z_1 = ex_pair_0(a_17.z_0);
    return r_10;
}

fn p_abs_0( a_18 : vec2<f32>) -> vec2<f32>
{
    var _S56 : vec2<f32>;
    if((a_18.x) < 0.0f)
    {
        _S56 = (vec2<f32>(0) - a_18);
    }
    else
    {
        _S56 = a_18;
    }
    return _S56;
}

fn p_finite_0( a_19 : vec2<f32>) -> bool
{
    var _S57 : f32 = a_19.x;
    var _S58 : bool;
    if(!isinf_0(_S57))
    {
        _S58 = !isnan_0(_S57);
    }
    else
    {
        _S58 = false;
    }
    if(_S58)
    {
        _S58 = !isinf_0(a_19.y);
    }
    else
    {
        _S58 = false;
    }
    if(_S58)
    {
        _S58 = !isnan_0(a_19.y);
    }
    else
    {
        _S58 = false;
    }
    return _S58;
}

fn two_prod_0( a_20 : f32,  b_13 : f32,  p_2 : ptr<function, f32>,  e_8 : ptr<function, f32>)
{
    var _S59 : f32 = opaque_0(a_20 * b_13);
    (*p_2) = _S59;
    (*e_8) = fma(a_20, b_13, - _S59);
    return;
}

fn two_prod_1( a_21 : f32,  b_14 : f32,  p_3 : ptr<function, f32>,  e_9 : ptr<function, f32>)
{
    var _S60 : f32 = opaque_0(a_21 * b_14);
    (*p_3) = _S60;
    (*e_9) = fma(a_21, b_14, - _S60);
    return;
}

fn two_prod_2( a_22 : f32,  b_15 : f32,  p_4 : ptr<function, f32>,  e_10 : ptr<function, f32>)
{
    var _S61 : f32 = opaque_0(a_22 * b_15);
    (*p_4) = _S61;
    (*e_10) = fma(a_22, b_15, - _S61);
    return;
}

fn fast_two_sum_0( a_23 : f32,  b_16 : f32,  s_9 : ptr<function, f32>,  e_11 : ptr<function, f32>)
{
    var _S62 : f32 = opaque_0(a_23 + b_16);
    (*s_9) = _S62;
    (*e_11) = b_16 - opaque_0(_S62 - a_23);
    return;
}

fn fast_two_sum_1( a_24 : f32,  b_17 : f32,  s_10 : ptr<function, f32>,  e_12 : ptr<function, f32>)
{
    var _S63 : f32 = opaque_0(a_24 + b_17);
    (*s_10) = _S63;
    (*e_12) = b_17 - opaque_0(_S63 - a_24);
    return;
}

fn fast_two_sum_2( a_25 : f32,  b_18 : f32,  s_11 : ptr<function, f32>,  e_13 : ptr<function, f32>)
{
    var _S64 : f32 = opaque_0(a_25 + b_18);
    (*s_11) = _S64;
    (*e_13) = b_18 - opaque_0(_S64 - a_25);
    return;
}

fn fast_two_sum_3( a_26 : f32,  b_19 : f32,  s_12 : ptr<function, f32>,  e_14 : ptr<function, f32>)
{
    var _S65 : f32 = opaque_0(a_26 + b_19);
    (*s_12) = _S65;
    (*e_14) = b_19 - opaque_0(_S65 - a_26);
    return;
}

fn p_mul_0( a_27 : vec2<f32>,  b_20 : vec2<f32>) -> vec2<f32>
{
    var _S66 : f32 = a_27.x;
    var _S67 : f32 = b_20.x;
    var ch_0 : f32;
    var cl_0 : f32;
    two_prod_0(_S66, _S67, &(ch_0), &(cl_0));
    var _S68 : f32 = a_27.y;
    var _S69 : f32 = b_20.y;
    var zh_0 : f32;
    var zl_0 : f32;
    fast_two_sum_0(ch_0, cl_0 + fma(_S68, _S67, fma(_S66, _S69, _S68 * _S69)), &(zh_0), &(zl_0));
    return vec2<f32>(zh_0, zl_0);
}

fn p_div_0( a_28 : vec2<f32>,  b_21 : vec2<f32>) -> vec2<f32>
{
    var _S70 : f32 = a_28.x;
    var _S71 : f32 = b_21.x;
    var th_0 : f32 = _S70 / _S71;
    var r_11 : vec2<f32> = p_mul_0(b_21, vec2<f32>(th_0, 0.0f));
    var zh_1 : f32;
    var zl_1 : f32;
    fast_two_sum_2(th_0, (_S70 - r_11.x + (a_28.y - r_11.y)) / _S71, &(zh_1), &(zl_1));
    return vec2<f32>(zh_1, zl_1);
}

fn p3_scale_0( a_29 : P3_0,  k_4 : vec2<f32>) -> P3_0
{
    var r_12 : P3_0;
    r_12.x_4 = p_mul_0(a_29.x_4, k_4);
    r_12.y_1 = p_mul_0(a_29.y_1, k_4);
    r_12.z_1 = p_mul_0(a_29.z_1, k_4);
    return r_12;
}

fn p_add_0( a_30 : vec2<f32>,  b_22 : vec2<f32>) -> vec2<f32>
{
    var sh_0 : f32;
    var sl_0 : f32;
    two_sum_0(a_30.x, b_22.x, &(sh_0), &(sl_0));
    var th_1 : f32;
    var tl_0 : f32;
    two_sum_0(a_30.y, b_22.y, &(th_1), &(tl_0));
    var vh_0 : f32;
    var vl_0 : f32;
    fast_two_sum_1(sh_0, sl_0 + th_1, &(vh_0), &(vl_0));
    var zh_2 : f32;
    var zl_2 : f32;
    fast_two_sum_1(vh_0, tl_0 + vl_0, &(zh_2), &(zl_2));
    return vec2<f32>(zh_2, zl_2);
}

fn p3_dot_0( a_31 : P3_0,  b_23 : P3_0) -> vec2<f32>
{
    return p_add_0(p_add_0(p_mul_0(a_31.x_4, b_23.x_4), p_mul_0(a_31.y_1, b_23.y_1)), p_mul_0(a_31.z_1, b_23.z_1));
}

fn p_sqrt_0( a_32 : vec2<f32>) -> vec2<f32>
{
    var _S72 : f32 = a_32.x;
    if(!(_S72 > 0.0f))
    {
        return vec2<f32>(sqrt(_S72), 0.0f);
    }
    var s_13 : f32 = sqrt(_S72);
    var p_5 : f32;
    var e_15 : f32;
    two_prod_1(s_13, s_13, &(p_5), &(e_15));
    var zh_3 : f32;
    var zl_3 : f32;
    fast_two_sum_3(s_13, (_S72 - p_5 - e_15 + a_32.y) / (2.0f * s_13), &(zh_3), &(zl_3));
    return vec2<f32>(zh_3, zl_3);
}

var<workgroup> g_frame_0 : array<vec2<f32>, i32(24)>;

fn ex_product_equal_0( a_33 : Ex_0,  b_24 : Ex_0,  c_1 : Ex_0,  d_0 : Ex_0,  exact_8 : ptr<function, bool>) -> bool
{
    var e_16 : array<f32, i32(64)>;
    var z_2 : u32 = u32(0);
    loop
    {
        if(z_2 < u32(64))
        {
        }
        else
        {
            break;
        }
        e_16[z_2] = 0.0f;
        z_2 = z_2 + u32(1);
    }
    var count_2 : u32 = u32(0);
    var side_0 : u32 = u32(0);
    loop
    {
        if(side_0 < u32(2))
        {
        }
        else
        {
            break;
        }
        var i_8 : u32 = u32(0);
        loop
        {
            if(i_8 < u32(8))
            {
            }
            else
            {
                break;
            }
            var count_3 : u32 = count_2;
            var j_0 : u32 = u32(0);
            loop
            {
                if(j_0 < u32(8))
                {
                }
                else
                {
                    break;
                }
                var _S73 : bool = side_0 == u32(0);
                var x_5 : f32;
                if(_S73)
                {
                    x_5 = a_33.t_0[i_8];
                }
                else
                {
                    x_5 = c_1.t_0[i_8];
                }
                var y_2 : f32;
                if(_S73)
                {
                    y_2 = b_24.t_0[j_0];
                }
                else
                {
                    y_2 = - d_0.t_0[j_0];
                }
                var _S74 : bool;
                if(x_5 == 0.0f)
                {
                    _S74 = true;
                }
                else
                {
                    _S74 = y_2 == 0.0f;
                }
                var count_4 : u32;
                if(_S74)
                {
                    count_4 = count_3;
                    var _S75 : u32 = j_0 + u32(1);
                    count_3 = count_4;
                    j_0 = _S75;
                    continue;
                }
                var p_6 : f32;
                var pe_0 : f32;
                two_prod_2(x_5, y_2, &(p_6), &(pe_0));
                count_4 = count_3;
                var half_0 : u32 = u32(0);
                loop
                {
                    if(half_0 < u32(2))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var carry_1 : f32;
                    if(half_0 == u32(0))
                    {
                        carry_1 = pe_0;
                    }
                    else
                    {
                        carry_1 = p_6;
                    }
                    if(carry_1 == 0.0f)
                    {
                        half_0 = half_0 + u32(1);
                        continue;
                    }
                    var carry_2 : f32 = carry_1;
                    var k_5 : u32 = u32(0);
                    var next_0 : u32 = u32(0);
                    loop
                    {
                        if(k_5 < count_4)
                        {
                        }
                        else
                        {
                            break;
                        }
                        var s_14 : f32;
                        var h_0 : f32;
                        two_sum_3(carry_2, e_16[k_5], &(s_14), &(h_0));
                        if(h_0 != 0.0f)
                        {
                            var _S76 : u32 = next_0 + u32(1);
                            e_16[next_0] = h_0;
                            next_0 = _S76;
                        }
                        var _S77 : f32 = s_14;
                        var _S78 : u32 = k_5 + u32(1);
                        carry_2 = _S77;
                        k_5 = _S78;
                    }
                    var next_1 : u32;
                    if(carry_2 != 0.0f)
                    {
                        if(next_0 >= u32(64))
                        {
                            (*exact_8) = false;
                            return false;
                        }
                        var _S79 : u32 = next_0 + u32(1);
                        e_16[next_0] = carry_2;
                        next_1 = _S79;
                    }
                    else
                    {
                        next_1 = next_0;
                    }
                    count_4 = next_1;
                    half_0 = half_0 + u32(1);
                }
                var _S75 : u32 = j_0 + u32(1);
                count_3 = count_4;
                j_0 = _S75;
            }
            var _S80 : u32 = i_8 + u32(1);
            count_2 = count_3;
            i_8 = _S80;
        }
        side_0 = side_0 + u32(1);
    }
    return count_2 == u32(0);
}

fn ex3_collinear_0( a_34 : Ex3_0,  b_25 : Ex3_0,  exact_9 : ptr<function, bool>) -> bool
{
    var _S81 : bool = ex_product_equal_0(a_34.y_0, b_25.z_0, a_34.z_0, b_25.y_0, &((*exact_9)));
    var _S82 : bool;
    if(_S81)
    {
        var _S83 : bool = ex_product_equal_0(a_34.z_0, b_25.x_2, a_34.x_2, b_25.z_0, &((*exact_9)));
        _S82 = _S83;
    }
    else
    {
        _S82 = false;
    }
    if(_S82)
    {
        var _S84 : bool = ex_product_equal_0(a_34.x_2, b_25.y_0, a_34.y_0, b_25.x_2, &((*exact_9)));
        _S82 = _S84;
    }
    else
    {
        _S82 = false;
    }
    return _S82;
}

fn p_inverse_square_0( x_6 : f32) -> vec2<f32>
{
    var p_7 : f32;
    var e_17 : f32;
    two_prod_1(x_6, x_6, &(p_7), &(e_17));
    return p_div_0(vec2<f32>(1.0f, 0.0f), vec2<f32>(p_7, e_17));
}

var<workgroup> g_pair_0 : array<vec2<f32>, i32(1024)>;

fn pair_sum_0( tid_2 : u32,  v_10 : vec2<f32>) -> vec2<f32>
{
    g_pair_0[tid_2] = v_10;
    workgroupBarrier();
    var k_6 : u32 = u32(512);
    loop
    {
        if(k_6 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_2 < k_6)
        {
            g_pair_0[tid_2] = p_add_0(g_pair_0[tid_2], g_pair_0[tid_2 + k_6]);
        }
        workgroupBarrier();
        k_6 = (k_6 >> (u32(1)));
    }
    var s_15 : vec2<f32> = g_pair_0[i32(0)];
    workgroupBarrier();
    return s_15;
}

fn p_sub_0( a_35 : vec2<f32>,  b_26 : vec2<f32>) -> vec2<f32>
{
    return p_add_0(a_35, (vec2<f32>(0) - b_26));
}

fn p3_sub_0( a_36 : P3_0,  b_27 : P3_0) -> P3_0
{
    var r_13 : P3_0;
    r_13.x_4 = p_sub_0(a_36.x_4, b_27.x_4);
    r_13.y_1 = p_sub_0(a_36.y_1, b_27.y_1);
    r_13.z_1 = p_sub_0(a_36.z_1, b_27.z_1);
    return r_13;
}

fn p3_cross_0( a_37 : P3_0,  b_28 : P3_0) -> P3_0
{
    var r_14 : P3_0;
    r_14.x_4 = p_sub_0(p_mul_0(a_37.y_1, b_28.z_1), p_mul_0(a_37.z_1, b_28.y_1));
    r_14.y_1 = p_sub_0(p_mul_0(a_37.z_1, b_28.x_4), p_mul_0(a_37.x_4, b_28.z_1));
    r_14.z_1 = p_sub_0(p_mul_0(a_37.x_4, b_28.y_1), p_mul_0(a_37.y_1, b_28.x_4));
    return r_14;
}

fn p3_at_0( a_38 : P3_0,  i_9 : u32) -> vec2<f32>
{
    var _S85 : vec2<f32>;
    if(i_9 == u32(0))
    {
        _S85 = a_38.x_4;
    }
    else
    {
        if(i_9 == u32(1))
        {
            _S85 = a_38.y_1;
        }
        else
        {
            _S85 = a_38.z_1;
        }
    }
    return _S85;
}

struct BlastNode_0
{
     w_ang_0 : f32,
     w_lin_0 : f32,
     ref_begin_0 : u32,
     ref_end_0 : u32,
     position_0 : vec4<f32>,
};

@compute
@workgroup_size(1024, 1, 1)
fn blast_motion(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var second_0 : bool;
    var changed_0 : f32;
    var changed_1 : f32;
    var best_0 : u32;
    var other_0 : u32;
    var r_15 : u32;
    var _S86 : bool;
    var _S87 : bool;
    var row_0 : u32;
    var col_0 : u32;
    var value_3 : vec2<f32>;
    var tid_3 : u32 = thread_2.x;
    var k_7 : u32 = group_0.x;
    var range_0 : vec4<u32> = group_uniform4_0(islands_0[k_7], tid_3);
    var n0_1 : u32 = range_0.x;
    var _S88 : u32 = range_0.y;
    var _S89 : u32 = range_0.w;
    var _S90 : u32 = range_0.z + tid_3;
    var anchor_0 : f32 = 0.0f;
    var q_1 : u32 = _S90;
    loop
    {
        if(q_1 < _S89)
        {
        }
        else
        {
            break;
        }
        var b_29 : BlastBond_std430_0 = bonds_0[lists_0[q_1]];
        if((b_29.health_0) > 0.0f)
        {
            if(!((nodes_0[b_29.n0_0].w_lin_0) > 0.0f))
            {
                second_0 = true;
            }
            else
            {
                second_0 = !((nodes_0[b_29.n1_0].w_lin_0) > 0.0f);
            }
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            anchor_0 = 1.0f;
        }
        q_1 = q_1 + u32(1024);
    }
    var _S91 : bool = tid_3 == u32(0);
    var p_8 : u32;
    if(_S91)
    {
        p_8 = u32(0);
        loop
        {
            if(p_8 < u32(40))
            {
            }
            else
            {
                break;
            }
            set_frame_0(k_7, p_8, u32(0));
            p_8 = p_8 + u32(1);
        }
    }
    device_group_barrier_0();
    var _S92 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(anchor_0, 0.0f));
    if((_S92.x) > 0.0f)
    {
        if(_S91)
        {
            set_frame_0(k_7, u32(0), u32(1));
        }
        return;
    }
    var exact_10 : bool = true;
    var _S93 : u32 = n0_1 + tid_3;
    p_8 = _S93;
    loop
    {
        if(p_8 < _S88)
        {
        }
        else
        {
            break;
        }
        var i_10 : u32 = lists_0[p_8];
        var zero_0 : Ex3_0;
        var _S94 : Ex_0 = ex_zero_0();
        zero_0.x_2 = _S94;
        zero_0.y_0 = _S94;
        zero_0.z_0 = _S94;
        set_position_0(i_10, zero_0);
        if(p_8 == n0_1)
        {
            q_1 = u32(0);
        }
        else
        {
            q_1 = u32(4294967295);
        }
        set_level_0(i_10, q_1);
        p_8 = p_8 + u32(1024);
    }
    device_group_barrier_0();
    var l_1 : u32 = u32(0);
    loop
    {
        changed_0 = 0.0f;
        p_8 = _S93;
        loop
        {
            if(p_8 < _S88)
            {
            }
            else
            {
                break;
            }
            var i_11 : u32 = lists_0[p_8];
            if((level_0(i_11)) != u32(4294967295))
            {
                p_8 = p_8 + u32(1024);
                continue;
            }
            var n_4 : BlastNode_std430_0 = nodes_0[i_11];
            best_0 = u32(4294967295);
            second_0 = false;
            r_15 = nodes_0[i_11].ref_begin_0;
            loop
            {
                if(r_15 < (n_4.ref_end_0))
                {
                }
                else
                {
                    break;
                }
                var ref_0 : u32 = refs_0[r_15];
                var j_1 : u32 = (ref_0 & (u32(2147483647)));
                var b_30 : BlastBond_std430_0 = bonds_0[j_1];
                var _S95 : bool = ((ref_0 & (u32(2147483648)))) != u32(0);
                if(_S95)
                {
                    other_0 = b_30.n0_0;
                }
                else
                {
                    other_0 = b_30.n1_0;
                }
                if((b_30.health_0) > 0.0f)
                {
                    _S86 = (nodes_0[other_0].w_lin_0) > 0.0f;
                }
                else
                {
                    _S86 = false;
                }
                if(_S86)
                {
                    _S87 = (level_0(other_0)) == l_1;
                }
                else
                {
                    _S87 = false;
                }
                var _S96 : bool;
                if(_S87)
                {
                    _S96 = j_1 < best_0;
                }
                else
                {
                    _S96 = false;
                }
                if(_S96)
                {
                    best_0 = j_1;
                    second_0 = _S95;
                }
                r_15 = r_15 + u32(1);
            }
            if(best_0 != u32(4294967295))
            {
                var _S97 : BlastBond_std430_0 = bonds_0[best_0];
                var _S98 : Ex3_0 = offset_0(&(_S97), &(exact_10));
                var at_1 : Ex3_0;
                if(second_0)
                {
                    var _S99 : Ex3_0 = ex3_add_0(position_1(_S97.n0_0), _S98, &(exact_10));
                    at_1 = _S99;
                }
                else
                {
                    var _S100 : Ex3_0 = ex3_add_0(position_1(_S97.n1_0), ex3_neg_0(_S98), &(exact_10));
                    at_1 = _S100;
                }
                set_position_0(i_11, at_1);
                set_level_0(i_11, l_1 + u32(1));
                changed_1 = 1.0f;
            }
            else
            {
                changed_1 = changed_0;
            }
            changed_0 = changed_1;
            p_8 = p_8 + u32(1024);
        }
        device_group_barrier_0();
        var _S101 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(changed_0, 0.0f));
        if((_S101.x) == 0.0f)
        {
            break;
        }
        l_1 = l_1 + u32(1);
    }
    changed_1 = 0.0f;
    p_8 = _S93;
    loop
    {
        if(p_8 < _S88)
        {
        }
        else
        {
            break;
        }
        if((level_0(lists_0[p_8])) == u32(4294967295))
        {
            changed_1 = 1.0f;
        }
        p_8 = p_8 + u32(1024);
    }
    best_0 = u32(4294967295);
    q_1 = _S90;
    loop
    {
        if(q_1 < _S89)
        {
        }
        else
        {
            break;
        }
        var j_2 : u32 = lists_0[q_1];
        var _S102 : BlastBond_std430_0 = bonds_0[j_2];
        if(!((_S102.health_0) > 0.0f))
        {
            q_1 = q_1 + u32(1024);
            continue;
        }
        var _S103 : Ex3_0 = closure_0(&(_S102), &(exact_10));
        if(ex3_nonzero_0(_S103))
        {
            var _S104 : bool = significant_0(&(_S102), _S103);
            second_0 = _S104;
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            other_0 = min(best_0, j_2);
        }
        else
        {
            other_0 = best_0;
        }
        best_0 = other_0;
        q_1 = q_1 + u32(1024);
    }
    var seed_0 : u32 = word_min_0(tid_3, best_0);
    var value_4 : vec2<f32>;
    if(_S91)
    {
        if(seed_0 == u32(4294967295))
        {
            q_1 = u32(3);
        }
        else
        {
            q_1 = u32(1);
        }
        g_rotations_0 = q_1;
        if(seed_0 != u32(4294967295))
        {
            var e2_0 : bool = true;
            var _S105 : BlastBond_std430_0 = bonds_0[seed_0];
            var _S106 : Ex3_0 = closure_0(&(_S105), &(e2_0));
            var axis_0 : P3_0 = ex3_pair_0(_S106);
            var maximum_0 : vec2<f32> = p_abs_0(axis_0.x_4);
            if((abs(axis_0.y_1.x)) > (maximum_0.x))
            {
                value_4 = p_abs_0(axis_0.y_1);
            }
            else
            {
                value_4 = maximum_0;
            }
            if((abs(axis_0.z_1.x)) > (value_4.x))
            {
                value_4 = p_abs_0(axis_0.z_1);
            }
            if(!((value_4.x) > 0.0f))
            {
                second_0 = true;
            }
            else
            {
                second_0 = !p_finite_0(value_4);
            }
            if(second_0)
            {
                set_frame_0(k_7, u32(2), u32(4));
            }
            const _S107 : vec2<f32> = vec2<f32>(1.0f, 0.0f);
            var axis_1 : P3_0 = p3_scale_0(axis_0, p_div_0(_S107, value_4));
            var axis_2 : P3_0 = p3_scale_0(axis_1, p_div_0(_S107, p_sqrt_0(p3_dot_0(axis_1, axis_1))));
            g_frame_0[i32(21)] = axis_2.x_4;
            g_frame_0[i32(22)] = axis_2.y_1;
            g_frame_0[i32(23)] = axis_2.z_1;
        }
    }
    workgroupBarrier();
    var locked_0 : f32;
    if(seed_0 != u32(4294967295))
    {
        var e2_1 : bool = true;
        var _S108 : BlastBond_std430_0 = bonds_0[seed_0];
        var _S109 : Ex3_0 = closure_0(&(_S108), &(e2_1));
        locked_0 = 0.0f;
        q_1 = _S90;
        loop
        {
            if(q_1 < _S89)
            {
            }
            else
            {
                break;
            }
            var _S110 : BlastBond_std430_0 = bonds_0[lists_0[q_1]];
            if(!((_S110.health_0) > 0.0f))
            {
                q_1 = q_1 + u32(1024);
                continue;
            }
            var _S111 : Ex3_0 = closure_0(&(_S110), &(exact_10));
            if(!ex3_nonzero_0(_S111))
            {
                second_0 = true;
            }
            else
            {
                var _S112 : bool = significant_0(&(_S110), _S111);
                second_0 = !_S112;
            }
            if(second_0)
            {
                q_1 = q_1 + u32(1024);
                continue;
            }
            var _S113 : bool = ex3_collinear_0(_S109, _S111, &(exact_10));
            var locked_1 : f32;
            if(!_S113)
            {
                locked_1 = 1.0f;
            }
            else
            {
                locked_1 = locked_0;
            }
            locked_0 = locked_1;
            q_1 = q_1 + u32(1024);
        }
    }
    else
    {
        locked_0 = 0.0f;
    }
    if(exact_10)
    {
        anchor_0 = 0.0f;
    }
    else
    {
        anchor_0 = 1.0f;
    }
    var flags_0 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(locked_0, anchor_0));
    var _S114 : vec2<f32> = blast_group_sum_0(tid_3, vec2<f32>(changed_1, 0.0f));
    var unconnected_0 : bool = (_S114.x) > 0.0f;
    if((flags_0.x) > 0.0f)
    {
        q_1 = u32(0);
    }
    else
    {
        q_1 = g_rotations_0;
    }
    var rotations_0 : u32 = group_uniform_0(q_1, tid_3);
    var axis_3 : P3_0;
    axis_3.x_4 = g_frame_0[i32(21)];
    axis_3.y_1 = g_frame_0[i32(22)];
    axis_3.z_1 = g_frame_0[i32(23)];
    var sums_0 : array<vec2<f32>, i32(4)>;
    other_0 = u32(0);
    loop
    {
        if(other_0 < u32(4))
        {
        }
        else
        {
            break;
        }
        sums_0[other_0] = vec2<f32>(0.0f);
        other_0 = other_0 + u32(1);
    }
    p_8 = _S93;
    loop
    {
        if(p_8 < _S88)
        {
        }
        else
        {
            break;
        }
        var i_12 : u32 = lists_0[p_8];
        var w_1 : vec2<f32> = p_inverse_square_0(nodes_0[i_12].w_lin_0);
        var at_2 : P3_0 = ex3_pair_0(position_1(i_12));
        sums_0[i32(0)] = p_add_0(sums_0[i32(0)], w_1);
        sums_0[i32(1)] = p_add_0(sums_0[i32(1)], p_mul_0(at_2.x_4, w_1));
        sums_0[i32(2)] = p_add_0(sums_0[i32(2)], p_mul_0(at_2.y_1, w_1));
        sums_0[i32(3)] = p_add_0(sums_0[i32(3)], p_mul_0(at_2.z_1, w_1));
        p_8 = p_8 + u32(1024);
    }
    other_0 = u32(0);
    loop
    {
        if(other_0 < u32(4))
        {
        }
        else
        {
            break;
        }
        var s_16 : vec2<f32> = pair_sum_0(tid_3, sums_0[other_0]);
        if(_S91)
        {
            g_frame_0[other_0] = s_16;
        }
        other_0 = other_0 + u32(1);
    }
    workgroupBarrier();
    var centre_0 : P3_0;
    centre_0.x_4 = p_div_0(g_frame_0[i32(1)], g_frame_0[i32(0)]);
    centre_0.y_1 = p_div_0(g_frame_0[i32(2)], g_frame_0[i32(0)]);
    centre_0.z_1 = p_div_0(g_frame_0[i32(3)], g_frame_0[i32(0)]);
    if((g_frame_0[i32(0)].x) > 0.0f)
    {
        second_0 = p_finite_0(g_frame_0[i32(0)]);
    }
    else
    {
        second_0 = false;
    }
    workgroupBarrier();
    var _S115 : u32 = rotations_0 + u32(3);
    var _S116 : bool = rotations_0 == u32(1);
    if(_S116)
    {
        value_4 = p3_dot_0(axis_3, axis_3);
    }
    else
    {
        value_4 = vec2<f32>(1.0f, 0.0f);
    }
    var gram_0 : array<vec2<f32>, i32(21)>;
    other_0 = u32(0);
    loop
    {
        if(other_0 < u32(21))
        {
        }
        else
        {
            break;
        }
        gram_0[other_0] = vec2<f32>(0.0f);
        other_0 = other_0 + u32(1);
    }
    p_8 = _S93;
    loop
    {
        if(p_8 < _S88)
        {
        }
        else
        {
            break;
        }
        var i_13 : u32 = lists_0[p_8];
        var n_5 : BlastNode_0;
        n_5.w_ang_0 = nodes_0[i_13].w_ang_0;
        n_5.w_lin_0 = nodes_0[i_13].w_lin_0;
        n_5.ref_begin_0 = nodes_0[i_13].ref_begin_0;
        n_5.ref_end_0 = nodes_0[i_13].ref_end_0;
        n_5.position_0 = nodes_0[i_13].position_0;
        var q_2 : P3_0 = p3_sub_0(ex3_pair_0(position_1(i_13)), centre_0);
        var _S117 : vec2<f32> = p_inverse_square_0(n_5.w_lin_0);
        var _S118 : vec2<f32> = p_inverse_square_0(n_5.w_ang_0);
        n_5.position_0 = vec4<f32>(q_2.x_4.x, q_2.y_1.x, q_2.z_1.x, 0.0f);
        nodes_0[i_13].w_ang_0 = n_5.w_ang_0;
        nodes_0[i_13].w_lin_0 = n_5.w_lin_0;
        nodes_0[i_13].ref_begin_0 = n_5.ref_begin_0;
        nodes_0[i_13].ref_end_0 = n_5.ref_end_0;
        nodes_0[i_13].position_0 = n_5.position_0;
        var moment_0 : array<P3_0, i32(3)>;
        r_15 = u32(0);
        loop
        {
            if(r_15 < u32(3))
            {
            }
            else
            {
                break;
            }
            if(_S116)
            {
                moment_0[r_15] = p3_cross_0(q_2, axis_3);
            }
            else
            {
                var e_18 : P3_0;
                if(r_15 == u32(0))
                {
                    anchor_0 = 1.0f;
                }
                else
                {
                    anchor_0 = 0.0f;
                }
                e_18.x_4 = vec2<f32>(anchor_0, 0.0f);
                if(r_15 == u32(1))
                {
                    changed_0 = 1.0f;
                }
                else
                {
                    changed_0 = 0.0f;
                }
                e_18.y_1 = vec2<f32>(changed_0, 0.0f);
                if(r_15 == u32(2))
                {
                    changed_1 = 1.0f;
                }
                else
                {
                    changed_1 = 0.0f;
                }
                e_18.z_1 = vec2<f32>(changed_1, 0.0f);
                moment_0[r_15] = p3_cross_0(q_2, e_18);
            }
            r_15 = r_15 + u32(1);
        }
        row_0 = u32(0);
        loop
        {
            if(row_0 < _S115)
            {
            }
            else
            {
                break;
            }
            col_0 = u32(0);
            loop
            {
                if(col_0 <= row_0)
                {
                }
                else
                {
                    break;
                }
                var _S119 : vec2<f32> = vec2<f32>(0.0f);
                if(row_0 < rotations_0)
                {
                    if(row_0 == col_0)
                    {
                        value_3 = p_mul_0(_S118, value_4);
                    }
                    else
                    {
                        value_3 = _S119;
                    }
                    value_3 = p_add_0(value_3, p_mul_0(_S117, p3_dot_0(moment_0[row_0], moment_0[col_0])));
                }
                else
                {
                    if(col_0 < rotations_0)
                    {
                        value_3 = p_mul_0(_S117, p3_at_0(moment_0[col_0], row_0 - rotations_0));
                    }
                    else
                    {
                        if(row_0 == col_0)
                        {
                            value_3 = _S117;
                        }
                        else
                        {
                            value_3 = _S119;
                        }
                    }
                }
                gram_0[row_0 * (row_0 + u32(1)) / u32(2) + col_0] = p_add_0(gram_0[row_0 * (row_0 + u32(1)) / u32(2) + col_0], value_3);
                col_0 = col_0 + u32(1);
            }
            row_0 = row_0 + u32(1);
        }
        p_8 = p_8 + u32(1024);
    }
    other_0 = u32(0);
    loop
    {
        if(other_0 < u32(21))
        {
        }
        else
        {
            break;
        }
        var s_17 : vec2<f32> = pair_sum_0(tid_3, gram_0[other_0]);
        if(_S91)
        {
            gram_0[other_0] = s_17;
        }
        other_0 = other_0 + u32(1);
    }
    if(tid_3 != u32(0))
    {
        return;
    }
    if(unconnected_0)
    {
        q_1 = u32(2);
    }
    else
    {
        q_1 = u32(0);
    }
    if((flags_0.y) > 0.0f)
    {
        p_8 = u32(16);
    }
    else
    {
        p_8 = u32(0);
    }
    var error_0 : u32 = (q_1 | (p_8));
    if(!second_0)
    {
        r_15 = (error_0 | (u32(4)));
    }
    else
    {
        r_15 = error_0;
    }
    var scale_0 : array<vec2<f32>, i32(6)>;
    var factor_0 : array<vec2<f32>, i32(21)>;
    other_0 = u32(0);
    loop
    {
        if(other_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        scale_0[other_0] = vec2<f32>(0.0f);
        other_0 = other_0 + u32(1);
    }
    other_0 = u32(0);
    loop
    {
        if(other_0 < u32(21))
        {
        }
        else
        {
            break;
        }
        factor_0[other_0] = vec2<f32>(0.0f);
        other_0 = other_0 + u32(1);
    }
    var i_14 : u32 = u32(0);
    loop
    {
        if(i_14 < _S115)
        {
            second_0 = r_15 == u32(0);
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
        }
        else
        {
            break;
        }
        var _S120 : u32 = i_14 + u32(1);
        var diagonal_0 : vec2<f32> = gram_0[i_14 * _S120 / u32(2) + i_14];
        if(!((gram_0[i_14 * _S120 / u32(2) + i_14].x) > 0.0f))
        {
            _S86 = true;
        }
        else
        {
            _S86 = !p_finite_0(diagonal_0);
        }
        if(_S86)
        {
            r_15 = (r_15 | (u32(4)));
        }
        else
        {
            scale_0[i_14] = p_div_0(vec2<f32>(1.0f, 0.0f), p_sqrt_0(diagonal_0));
        }
        i_14 = _S120;
    }
    row_0 = u32(0);
    loop
    {
        if(row_0 < _S115)
        {
            second_0 = r_15 == u32(0);
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
        }
        else
        {
            break;
        }
        col_0 = u32(0);
        var error_1 : u32 = r_15;
        loop
        {
            if(col_0 <= row_0)
            {
                _S86 = error_1 == u32(0);
            }
            else
            {
                _S86 = false;
            }
            if(_S86)
            {
            }
            else
            {
                break;
            }
            var _S121 : u32 = row_0 * (row_0 + u32(1));
            value_4 = p_mul_0(p_mul_0(gram_0[_S121 / u32(2) + col_0], scale_0[row_0]), scale_0[col_0]);
            var m_1 : u32 = u32(0);
            loop
            {
                if(m_1 < col_0)
                {
                }
                else
                {
                    break;
                }
                var _S122 : vec2<f32> = p_sub_0(value_4, p_mul_0(factor_0[_S121 / u32(2) + m_1], factor_0[col_0 * (col_0 + u32(1)) / u32(2) + m_1]));
                var _S123 : u32 = m_1 + u32(1);
                value_4 = _S122;
                m_1 = _S123;
            }
            if(row_0 == col_0)
            {
                if(!((value_4.x) > 0.0f))
                {
                    _S87 = true;
                }
                else
                {
                    _S87 = !p_finite_0(value_4);
                }
                var error_2 : u32;
                if(_S87)
                {
                    error_2 = (error_1 | (u32(4)));
                }
                else
                {
                    error_2 = error_1;
                }
                value_3 = p_sqrt_0(value_4);
                error_1 = error_2;
            }
            else
            {
                value_3 = p_div_0(value_4, factor_0[col_0 * (col_0 + u32(1)) / u32(2) + col_0]);
            }
            factor_0[_S121 / u32(2) + col_0] = value_3;
            col_0 = col_0 + u32(1);
        }
        row_0 = row_0 + u32(1);
        r_15 = error_1;
    }
    set_frame_0(k_7, u32(1), rotations_0);
    set_frame_0(k_7, u32(2), (r_15 | ((islands_0[blast_frame_0(params_0.node_count_0, k_7)][u32(2)]))));
    set_frame_0(k_7, u32(4), (bitcast<u32>((axis_3.x_4.x))));
    set_frame_0(k_7, u32(5), (bitcast<u32>((axis_3.y_1.x))));
    set_frame_0(k_7, u32(6), (bitcast<u32>((axis_3.z_1.x))));
    i_14 = u32(0);
    loop
    {
        if(i_14 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S124 : u32 = u32(8) + i_14;
        if(i_14 < _S115)
        {
            second_0 = r_15 == u32(0);
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            anchor_0 = scale_0[i_14].x;
        }
        else
        {
            anchor_0 = 0.0f;
        }
        set_frame_0(k_7, _S124, (bitcast<u32>((anchor_0))));
        i_14 = i_14 + u32(1);
    }
    col_0 = u32(0);
    loop
    {
        if(col_0 < u32(21))
        {
        }
        else
        {
            break;
        }
        var _S125 : u32 = u32(14) + col_0;
        if(col_0 < (_S115 * (_S115 + u32(1)) / u32(2)))
        {
            second_0 = r_15 == u32(0);
        }
        else
        {
            second_0 = false;
        }
        if(second_0)
        {
            anchor_0 = factor_0[col_0].x;
        }
        else
        {
            anchor_0 = 0.0f;
        }
        set_frame_0(k_7, _S125, (bitcast<u32>((anchor_0))));
        col_0 = col_0 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn blast_motion_selftest(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_15 : u32 = id_0.x;
    if(i_15 >= (params_0.node_count_0))
    {
        return;
    }
    var _S126 : u32 = u32(2) * i_15;
    var s_18 : f32;
    var e_19 : f32;
    two_sum_3(motion_0[_S126].x, motion_0[_S126].y, &(s_18), &(e_19));
    var p_9 : f32;
    var pe_1 : f32;
    two_prod_2(motion_0[_S126].z, motion_0[_S126].w, &(p_9), &(pe_1));
    motion_0[_S126 + u32(1)] = vec4<f32>(s_18, e_19, p_9, pe_1);
    return;
}

