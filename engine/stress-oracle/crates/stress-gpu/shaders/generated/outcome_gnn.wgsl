struct Params_std140_0
{
    @align(16) p_0 : array<vec4<u32>, i32(16)>,
};

@binding(0) @group(0) var<uniform> params_0 : Params_std140_0;
@binding(2) @group(0) var<storage, read> table_0 : array<u32>;

@binding(3) @group(0) var<storage, read_write> arena_0 : array<f32>;

@binding(1) @group(0) var<storage, read> wts_0 : array<f32>;

fn P_0( i_0 : u32) -> u32
{
    return params_0.p_0[(i_0 >> (u32(2)))][(i_0 & (u32(3)))];
}

var<workgroup> xs_0 : array<array<f32, i32(384)>, i32(4)>;

var<workgroup> hid_0 : array<f32, i32(128)>;

fn silu_0( x_0 : f32) -> f32
{
    return x_0 / (1.0f + exp(- x_0));
}

var<workgroup> red_0 : array<f32, i32(128)>;

fn slot_sum_0( v_0 : f32,  tid_0 : u32,  lane_0 : u32,  lanes_0 : u32) -> f32
{
    red_0[tid_0] = v_0;
    workgroupBarrier();
    var s_0 : u32 = lanes_0 / u32(2);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(lane_0 < s_0)
        {
            red_0[tid_0] = red_0[tid_0] + red_0[tid_0 + s_0];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var r_0 : f32 = red_0[tid_0 - lane_0];
    workgroupBarrier();
    return r_0;
}

@compute
@workgroup_size(128, 1, 1)
fn gnn_mlp(@builtin(workgroup_id) gid_0 : vec3<u32>, @builtin(local_invocation_id) lid_0 : vec3<u32>)
{
    var r_1 : u32;
    var pre_0 : f32;
    var v_1 : f32;
    var _S1 : bool;
    var lanes_1 : u32 = P_0(u32(3));
    var tid_1 : u32 = lid_0.x;
    var slot_0 : u32 = tid_1 / lanes_1;
    var lane_1 : u32 = tid_1 % lanes_1;
    var _S2 : u32 = gid_0.y * P_0(u32(2)) + gid_0.x;
    var _S3 : u32 = u32(128) / lanes_1;
    var row_0 : u32 = _S2 * _S3 + slot_0;
    var live_0 : bool = row_0 < (P_0(u32(1)));
    var hidden_0 : u32 = P_0(u32(42));
    var out_0 : u32 = P_0(u32(43));
    var _S4 : u32 = P_0(u32(4));
    var s_1 : u32 = u32(0);
    var pre_1 : f32 = 0.0f;
    var k_0 : u32 = u32(0);
    loop
    {
        if(s_1 < _S4)
        {
        }
        else
        {
            break;
        }
        var o_0 : u32 = u32(8) + u32(5) * s_1;
        var base_0 : u32 = P_0(o_0);
        var idx_0 : u32 = P_0(o_0 + u32(1));
        var width_0 : u32 = P_0(o_0 + u32(2));
        var mode_0 : u32 = P_0(o_0 + u32(3));
        var _S5 : u32 = P_0(o_0 + u32(4));
        if(live_0)
        {
            if(mode_0 == u32(1))
            {
                _S1 = true;
            }
            else
            {
                _S1 = mode_0 == u32(4);
            }
            if(_S1)
            {
                r_1 = table_0[idx_0 + row_0];
            }
            else
            {
                if(mode_0 == u32(2))
                {
                    r_1 = idx_0;
                }
                else
                {
                    r_1 = row_0;
                }
            }
            if(mode_0 >= u32(4))
            {
                if(lane_1 < hidden_0)
                {
                    pre_0 = pre_1 + arena_0[base_0 + r_1 * width_0 + lane_1];
                }
                else
                {
                    pre_0 = pre_1;
                }
            }
            else
            {
                var c_0 : u32;
                if(mode_0 == u32(3))
                {
                    var _S6 : u32 = idx_0 + row_0;
                    var _S7 : u32 = table_0[_S6];
                    var _S8 : u32 = table_0[_S6 + u32(1)];
                    c_0 = lane_1;
                    loop
                    {
                        if(c_0 < width_0)
                        {
                        }
                        else
                        {
                            break;
                        }
                        var e_0 : u32 = _S7;
                        v_1 = 0.0f;
                        loop
                        {
                            if(e_0 < _S8)
                            {
                            }
                            else
                            {
                                break;
                            }
                            var _S9 : u32 = _S5 + u32(2) * e_0;
                            var v_2 : f32 = v_1 + arena_0[table_0[_S9] + c_0] * (bitcast<f32>((table_0[_S9 + u32(1)])));
                            e_0 = e_0 + u32(1);
                            v_1 = v_2;
                        }
                        xs_0[slot_0][k_0 + c_0] = v_1;
                        c_0 = c_0 + lanes_1;
                    }
                }
                else
                {
                    c_0 = lane_1;
                    loop
                    {
                        if(c_0 < width_0)
                        {
                        }
                        else
                        {
                            break;
                        }
                        var v_3 : f32 = arena_0[base_0 + r_1 * width_0 + c_0];
                        if(_S5 != u32(4294967295))
                        {
                            v_1 = (v_3 - wts_0[_S5 + c_0]) / wts_0[_S5 + width_0 + c_0];
                        }
                        else
                        {
                            v_1 = v_3;
                        }
                        xs_0[slot_0][k_0 + c_0] = v_1;
                        c_0 = c_0 + lanes_1;
                    }
                }
                pre_0 = pre_1;
            }
            pre_1 = pre_0;
        }
        if(mode_0 < u32(4))
        {
            k_0 = k_0 + width_0;
        }
        s_1 = s_1 + u32(1);
    }
    workgroupBarrier();
    var _S10 : u32 = P_0(u32(41));
    var _S11 : u32 = P_0(u32(40));
    if(live_0)
    {
        _S1 = lane_1 < hidden_0;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        var _S12 : u32 = P_0(u32(52));
        if(_S12 == u32(4294967295))
        {
            pre_0 = 0.0f;
        }
        else
        {
            pre_0 = wts_0[_S12 + lane_1];
        }
        r_1 = u32(0);
        loop
        {
            if(r_1 < _S10)
            {
            }
            else
            {
                break;
            }
            var a_0 : f32 = pre_0 + wts_0[_S11 + r_1 * hidden_0 + lane_1] * xs_0[slot_0][r_1];
            r_1 = r_1 + u32(1);
            pre_0 = a_0;
        }
        pre_1 = pre_1 + pre_0;
    }
    var _S13 : u32 = P_0(u32(44));
    if(_S13 == u32(4294967295))
    {
        pre_0 = pre_1;
    }
    else
    {
        if(live_0)
        {
            _S1 = lane_1 < hidden_0;
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            pre_1 = silu_0(pre_1);
        }
        else
        {
            pre_1 = 0.0f;
        }
        hid_0[tid_1] = pre_1;
        workgroupBarrier();
        if(live_0)
        {
            _S1 = lane_1 < out_0;
        }
        else
        {
            _S1 = false;
        }
        if(_S1)
        {
            var _S14 : f32 = wts_0[P_0(u32(53)) + lane_1];
            r_1 = u32(0);
            pre_0 = _S14;
            loop
            {
                if(r_1 < hidden_0)
                {
                }
                else
                {
                    break;
                }
                var a_1 : f32 = pre_0 + wts_0[_S13 + r_1 * out_0 + lane_1] * hid_0[tid_1 - lane_1 + r_1];
                r_1 = r_1 + u32(1);
                pre_0 = a_1;
            }
        }
        else
        {
            pre_0 = 0.0f;
        }
        var _S15 : u32 = P_0(u32(45));
        if(_S15 != u32(4294967295))
        {
            var _S16 : f32 = slot_sum_0(pre_0, tid_1, lane_1, lanes_1);
            var _S17 : f32 = f32(out_0);
            var d_0 : f32 = pre_0 - _S16 / _S17;
            var _S18 : f32 = slot_sum_0(d_0 * d_0, tid_1, lane_1, lanes_1);
            pre_0 = d_0 * (1.0f / sqrt(_S18 / _S17 + 0.00000999999974738f)) * wts_0[_S15 + lane_1] + wts_0[_S15 + out_0 + lane_1];
        }
    }
    if(live_0)
    {
        _S1 = lane_1 < out_0;
    }
    else
    {
        _S1 = false;
    }
    if(_S1)
    {
        var v_4 : f32 = (bitcast<f32>((P_0(u32(51))))) * pre_0;
        var _S19 : u32 = P_0(u32(48));
        if(_S19 != u32(4294967295))
        {
            v_1 = v_4 + arena_0[_S19 + row_0 * P_0(u32(49)) + lane_1];
        }
        else
        {
            v_1 = v_4;
        }
        var o_1 : u32 = P_0(u32(46)) + row_0 * P_0(u32(47)) + lane_1;
        if((((P_0(u32(50))) & (u32(1)))) != u32(0))
        {
            arena_0[o_1] = arena_0[o_1] + v_1;
        }
        else
        {
            arena_0[o_1] = v_1;
        }
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn gnn_csr(@builtin(workgroup_id) gid_1 : vec3<u32>, @builtin(local_invocation_id) lid_1 : vec3<u32>)
{
    var lanes_2 : u32 = P_0(u32(3));
    var _S20 : u32 = lid_1.x;
    var slot_1 : u32 = _S20 / lanes_2;
    var lane_2 : u32 = _S20 % lanes_2;
    var _S21 : u32 = gid_1.y * P_0(u32(2)) + gid_1.x;
    var _S22 : u32 = u32(128) / lanes_2;
    var row_1 : u32 = _S21 * _S22 + slot_1;
    var _S23 : bool;
    if(row_1 >= (P_0(u32(1))))
    {
        _S23 = true;
    }
    else
    {
        _S23 = lane_2 >= (P_0(u32(4)));
    }
    if(_S23)
    {
        return;
    }
    var _S24 : u32 = P_0(u32(5)) + row_1;
    var _S25 : u32 = table_0[_S24 + u32(1)];
    var _S26 : u32 = P_0(u32(6));
    var e_1 : u32 = table_0[_S24];
    var s_2 : f32 = 0.0f;
    loop
    {
        if(e_1 < _S25)
        {
        }
        else
        {
            break;
        }
        var _S27 : u32 = _S26 + u32(2) * e_1;
        var s_3 : f32 = s_2 + arena_0[table_0[_S27] + lane_2] * (bitcast<f32>((table_0[_S27 + u32(1)])));
        e_1 = e_1 + u32(1);
        s_2 = s_3;
    }
    var o_2 : u32 = P_0(u32(7)) + row_1 * P_0(u32(8)) + lane_2;
    if((((P_0(u32(9))) & (u32(1)))) != u32(0))
    {
        arena_0[o_2] = arena_0[o_2] + s_2;
    }
    else
    {
        arena_0[o_2] = s_2;
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn gnn_fibre(@builtin(workgroup_id) gid_2 : vec3<u32>, @builtin(local_invocation_id) lid_2 : vec3<u32>)
{
    var r_2 : u32 = (gid_2.y * P_0(u32(2)) + gid_2.x) * u32(128) + lid_2.x;
    if(r_2 >= (P_0(u32(1))))
    {
        return;
    }
    var _S28 : u32 = u32(4) * r_2;
    var t_0 : u32 = P_0(u32(5)) + _S28;
    var pb_0 : u32 = P_0(u32(4));
    var _S29 : u32 = pb_0 + u32(3) * table_0[t_0];
    var _S30 : u32 = pb_0 + u32(3) * table_0[t_0 + u32(1)];
    var d_1 : vec3<f32> = vec3<f32>(arena_0[_S29] - arena_0[_S30], arena_0[_S29 + u32(1)] - arena_0[_S30 + u32(1)], arena_0[_S29 + u32(2)] - arena_0[_S30 + u32(2)]);
    var qb_0 : u32 = P_0(u32(6)) + u32(4) * table_0[t_0 + u32(2)];
    var u_0 : vec3<f32> = (vec3<f32>(0) - vec3<f32>(arena_0[qb_0 + u32(1)], arena_0[qb_0 + u32(2)], arena_0[qb_0 + u32(3)]));
    var tt_0 : vec3<f32> = vec3<f32>(2.0f) * cross(u_0, d_1);
    var rr_0 : vec3<f32> = (d_1 + vec3<f32>(arena_0[qb_0]) * tt_0 + cross(u_0, tt_0)) / vec3<f32>((bitcast<f32>((table_0[t_0 + u32(3)]))));
    var o_3 : u32 = P_0(u32(7)) + _S28;
    var _S31 : f32 = rr_0.x;
    arena_0[o_3] = _S31;
    var _S32 : f32 = rr_0.y;
    arena_0[o_3 + u32(1)] = _S32;
    var _S33 : f32 = rr_0.z;
    arena_0[o_3 + u32(2)] = _S33;
    arena_0[o_3 + u32(3)] = sqrt(_S31 * _S31 + _S32 * _S32 + _S33 * _S33);
    return;
}

var<workgroup> tx_0 : array<array<f32, i32(64)>, i32(8)>;

fn seg_value_0( base_1 : u32,  idx_1 : u32,  width_1 : u32,  mode_1 : u32,  norm_0 : u32,  row_2 : u32,  c_1 : u32) -> f32
{
    if(c_1 >= width_1)
    {
        return 0.0f;
    }
    var e_2 : u32;
    var v_5 : f32;
    if(mode_1 == u32(3))
    {
        var _S34 : u32 = idx_1 + row_2;
        var _S35 : u32 = table_0[_S34 + u32(1)];
        e_2 = table_0[_S34];
        v_5 = 0.0f;
        loop
        {
            if(e_2 < _S35)
            {
            }
            else
            {
                break;
            }
            var _S36 : u32 = norm_0 + u32(2) * e_2;
            var v_6 : f32 = v_5 + arena_0[table_0[_S36] + c_1] * (bitcast<f32>((table_0[_S36 + u32(1)])));
            e_2 = e_2 + u32(1);
            v_5 = v_6;
        }
        return v_5;
    }
    if(mode_1 == u32(1))
    {
        e_2 = table_0[idx_1 + row_2];
    }
    else
    {
        if(mode_1 == u32(2))
        {
            e_2 = idx_1;
        }
        else
        {
            e_2 = row_2;
        }
    }
    var v_7 : f32 = arena_0[base_1 + e_2 * width_1 + c_1];
    if(norm_0 != u32(4294967295))
    {
        v_5 = (v_7 - wts_0[norm_0 + c_1]) / wts_0[norm_0 + width_1 + c_1];
    }
    else
    {
        v_5 = v_7;
    }
    return v_5;
}

var<workgroup> tw_0 : array<array<f32, i32(64)>, i32(8)>;

var<workgroup> th_0 : array<f32, i32(2048)>;

var<workgroup> tstat_0 : array<array<f32, i32(64)>, i32(2)>;

@compute
@workgroup_size(128, 1, 1)
fn gnn_mlp_tiled(@builtin(workgroup_id) gid_3 : vec3<u32>, @builtin(local_invocation_id) lid_3 : vec3<u32>)
{
    var r0_0 : vec4<f32>;
    var r1_0 : vec4<f32>;
    var r2_0 : vec4<f32>;
    var r3_0 : vec4<f32>;
    var k0_0 : u32;
    var q_0 : u32;
    var m_0 : f32;
    var lanes_3 : u32 = P_0(u32(3));
    var tr_0 : u32 = u32(2048) / lanes_3;
    var t_1 : u32 = lid_3.x;
    var cg_0 : u32 = lanes_3 / u32(4);
    var _S37 : u32 = t_1 / cg_0;
    var _S38 : u32 = _S37 * u32(4);
    var _S39 : u32 = t_1 % cg_0;
    var cb_0 : u32 = _S39 * u32(4);
    var _S40 : u32 = (gid_3.y * P_0(u32(2)) + gid_3.x) * tr_0;
    var _S41 : u32 = P_0(u32(1));
    var _S42 : u32 = P_0(u32(42));
    var out_1 : u32 = P_0(u32(43));
    var _S43 : u32 = P_0(u32(40));
    const _S44 : vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S45 : u32 = P_0(u32(52));
    var bias_0 : vec4<f32>;
    if(_S45 != u32(4294967295))
    {
        var _S46 : u32 = _S45 + cb_0;
        bias_0 = vec4<f32>(wts_0[_S46], wts_0[_S46 + u32(1)], wts_0[_S46 + u32(2)], wts_0[_S46 + u32(3)]);
    }
    else
    {
        bias_0 = _S44;
    }
    var _S47 : u32 = P_0(u32(4));
    var s_4 : u32 = u32(0);
    var koff_0 : u32 = u32(0);
    var r0_1 : vec4<f32> = bias_0;
    var r1_1 : vec4<f32> = bias_0;
    var r2_1 : vec4<f32> = bias_0;
    var r3_1 : vec4<f32> = bias_0;
    loop
    {
        if(s_4 < _S47)
        {
        }
        else
        {
            break;
        }
        var o_4 : u32 = u32(8) + u32(5) * s_4;
        var _S48 : u32 = P_0(o_4);
        var _S49 : u32 = P_0(o_4 + u32(1));
        var width_2 : u32 = P_0(o_4 + u32(2));
        var mode_2 : u32 = P_0(o_4 + u32(3));
        var _S50 : u32 = P_0(o_4 + u32(4));
        if(mode_2 >= u32(4))
        {
            s_4 = s_4 + u32(1);
            continue;
        }
        k0_0 = u32(0);
        r0_0 = r0_1;
        r1_0 = r1_1;
        r2_0 = r2_1;
        r3_0 = r3_1;
        loop
        {
            if(k0_0 < width_2)
            {
            }
            else
            {
                break;
            }
            q_0 = t_1;
            loop
            {
                if(q_0 < (u32(8) * tr_0))
                {
                }
                else
                {
                    break;
                }
                var kk_0 : u32 = q_0 / tr_0;
                var r_3 : u32 = q_0 - kk_0 * tr_0;
                var row_3 : u32 = _S40 + r_3;
                if(row_3 < _S41)
                {
                    m_0 = seg_value_0(_S48, _S49, width_2, mode_2, _S50, row_3, k0_0 + kk_0);
                }
                else
                {
                    m_0 = 0.0f;
                }
                tx_0[kk_0][r_3] = m_0;
                q_0 = q_0 + u32(128);
            }
            var q_1 : u32 = t_1;
            loop
            {
                if(q_1 < (u32(8) * lanes_3))
                {
                }
                else
                {
                    break;
                }
                var kk_1 : u32 = q_1 / lanes_3;
                var c_2 : u32 = q_1 - kk_1 * lanes_3;
                if((k0_0 + kk_1) < width_2)
                {
                    m_0 = wts_0[_S43 + (koff_0 + k0_0 + kk_1) * _S42 + c_2];
                }
                else
                {
                    m_0 = 0.0f;
                }
                tw_0[kk_1][c_2] = m_0;
                q_1 = q_1 + u32(128);
            }
            workgroupBarrier();
            var _S51 : u32 = cb_0 + u32(1);
            var _S52 : u32 = cb_0 + u32(2);
            var _S53 : u32 = cb_0 + u32(3);
            var b_0 : vec4<f32> = vec4<f32>(tw_0[u32(0)][cb_0], tw_0[u32(0)][_S51], tw_0[u32(0)][_S52], tw_0[u32(0)][_S53]);
            var _S54 : u32 = _S38 + u32(1);
            var _S55 : u32 = _S38 + u32(2);
            var _S56 : u32 = _S38 + u32(3);
            var b_1 : vec4<f32> = vec4<f32>(tw_0[u32(1)][cb_0], tw_0[u32(1)][_S51], tw_0[u32(1)][_S52], tw_0[u32(1)][_S53]);
            var b_2 : vec4<f32> = vec4<f32>(tw_0[u32(2)][cb_0], tw_0[u32(2)][_S51], tw_0[u32(2)][_S52], tw_0[u32(2)][_S53]);
            var b_3 : vec4<f32> = vec4<f32>(tw_0[u32(3)][cb_0], tw_0[u32(3)][_S51], tw_0[u32(3)][_S52], tw_0[u32(3)][_S53]);
            var b_4 : vec4<f32> = vec4<f32>(tw_0[u32(4)][cb_0], tw_0[u32(4)][_S51], tw_0[u32(4)][_S52], tw_0[u32(4)][_S53]);
            var b_5 : vec4<f32> = vec4<f32>(tw_0[u32(5)][cb_0], tw_0[u32(5)][_S51], tw_0[u32(5)][_S52], tw_0[u32(5)][_S53]);
            var b_6 : vec4<f32> = vec4<f32>(tw_0[u32(6)][cb_0], tw_0[u32(6)][_S51], tw_0[u32(6)][_S52], tw_0[u32(6)][_S53]);
            var b_7 : vec4<f32> = vec4<f32>(tw_0[u32(7)][cb_0], tw_0[u32(7)][_S51], tw_0[u32(7)][_S52], tw_0[u32(7)][_S53]);
            var r0_2 : vec4<f32> = r0_0 + vec4<f32>(tx_0[u32(0)][_S38]) * b_0 + vec4<f32>(tx_0[u32(1)][_S38]) * b_1 + vec4<f32>(tx_0[u32(2)][_S38]) * b_2 + vec4<f32>(tx_0[u32(3)][_S38]) * b_3 + vec4<f32>(tx_0[u32(4)][_S38]) * b_4 + vec4<f32>(tx_0[u32(5)][_S38]) * b_5 + vec4<f32>(tx_0[u32(6)][_S38]) * b_6 + vec4<f32>(tx_0[u32(7)][_S38]) * b_7;
            var r1_2 : vec4<f32> = r1_0 + vec4<f32>(tx_0[u32(0)][_S54]) * b_0 + vec4<f32>(tx_0[u32(1)][_S54]) * b_1 + vec4<f32>(tx_0[u32(2)][_S54]) * b_2 + vec4<f32>(tx_0[u32(3)][_S54]) * b_3 + vec4<f32>(tx_0[u32(4)][_S54]) * b_4 + vec4<f32>(tx_0[u32(5)][_S54]) * b_5 + vec4<f32>(tx_0[u32(6)][_S54]) * b_6 + vec4<f32>(tx_0[u32(7)][_S54]) * b_7;
            var r2_2 : vec4<f32> = r2_0 + vec4<f32>(tx_0[u32(0)][_S55]) * b_0 + vec4<f32>(tx_0[u32(1)][_S55]) * b_1 + vec4<f32>(tx_0[u32(2)][_S55]) * b_2 + vec4<f32>(tx_0[u32(3)][_S55]) * b_3 + vec4<f32>(tx_0[u32(4)][_S55]) * b_4 + vec4<f32>(tx_0[u32(5)][_S55]) * b_5 + vec4<f32>(tx_0[u32(6)][_S55]) * b_6 + vec4<f32>(tx_0[u32(7)][_S55]) * b_7;
            var r3_2 : vec4<f32> = r3_0 + vec4<f32>(tx_0[u32(0)][_S56]) * b_0 + vec4<f32>(tx_0[u32(1)][_S56]) * b_1 + vec4<f32>(tx_0[u32(2)][_S56]) * b_2 + vec4<f32>(tx_0[u32(3)][_S56]) * b_3 + vec4<f32>(tx_0[u32(4)][_S56]) * b_4 + vec4<f32>(tx_0[u32(5)][_S56]) * b_5 + vec4<f32>(tx_0[u32(6)][_S56]) * b_6 + vec4<f32>(tx_0[u32(7)][_S56]) * b_7;
            workgroupBarrier();
            k0_0 = k0_0 + u32(8);
            r0_0 = r0_2;
            r1_0 = r1_2;
            r2_0 = r2_2;
            r3_0 = r3_2;
        }
        koff_0 = koff_0 + width_2;
        r0_1 = r0_0;
        r1_1 = r1_0;
        r2_1 = r2_0;
        r3_1 = r3_0;
        s_4 = s_4 + u32(1);
    }
    s_4 = u32(0);
    loop
    {
        if(s_4 < _S47)
        {
        }
        else
        {
            break;
        }
        var o_5 : u32 = u32(8) + u32(5) * s_4;
        var _S57 : u32 = P_0(o_5);
        var _S58 : u32 = P_0(o_5 + u32(1));
        var mode_3 : u32 = P_0(o_5 + u32(3));
        if(mode_3 < u32(4))
        {
            s_4 = s_4 + u32(1);
            continue;
        }
        var _S59 : u32 = _S40 + _S38;
        if(_S59 < _S41)
        {
            if(mode_3 == u32(4))
            {
                k0_0 = table_0[_S58 + _S59];
            }
            else
            {
                k0_0 = _S59;
            }
            var _S60 : u32 = _S57 + k0_0 * _S42 + cb_0;
            var pv_0 : vec4<f32> = vec4<f32>(arena_0[_S60], arena_0[_S60 + u32(1)], arena_0[_S60 + u32(2)], arena_0[_S60 + u32(3)]);
            r0_0 = r0_1 + pv_0;
        }
        else
        {
            r0_0 = r0_1;
        }
        var row_4 : u32 = _S59 + u32(1);
        if(row_4 < _S41)
        {
            if(mode_3 == u32(4))
            {
                k0_0 = table_0[_S58 + row_4];
            }
            else
            {
                k0_0 = row_4;
            }
            var _S61 : u32 = _S57 + k0_0 * _S42 + cb_0;
            var pv_1 : vec4<f32> = vec4<f32>(arena_0[_S61], arena_0[_S61 + u32(1)], arena_0[_S61 + u32(2)], arena_0[_S61 + u32(3)]);
            r1_0 = r1_1 + pv_1;
        }
        else
        {
            r1_0 = r1_1;
        }
        var row_5 : u32 = _S59 + u32(2);
        if(row_5 < _S41)
        {
            if(mode_3 == u32(4))
            {
                k0_0 = table_0[_S58 + row_5];
            }
            else
            {
                k0_0 = row_5;
            }
            var _S62 : u32 = _S57 + k0_0 * _S42 + cb_0;
            var pv_2 : vec4<f32> = vec4<f32>(arena_0[_S62], arena_0[_S62 + u32(1)], arena_0[_S62 + u32(2)], arena_0[_S62 + u32(3)]);
            r2_0 = r2_1 + pv_2;
        }
        else
        {
            r2_0 = r2_1;
        }
        var row_6 : u32 = _S59 + u32(3);
        if(row_6 < _S41)
        {
            if(mode_3 == u32(4))
            {
                k0_0 = table_0[_S58 + row_6];
            }
            else
            {
                k0_0 = row_6;
            }
            var _S63 : u32 = _S57 + k0_0 * _S42 + cb_0;
            var pv_3 : vec4<f32> = vec4<f32>(arena_0[_S63], arena_0[_S63 + u32(1)], arena_0[_S63 + u32(2)], arena_0[_S63 + u32(3)]);
            r3_0 = r3_1 + pv_3;
        }
        else
        {
            r3_0 = r3_1;
        }
        r0_1 = r0_0;
        r1_1 = r1_0;
        r2_1 = r2_0;
        r3_1 = r3_0;
        s_4 = s_4 + u32(1);
    }
    var y0_0 : vec4<f32> = r0_1;
    var y1_0 : vec4<f32> = r1_1;
    var y2_0 : vec4<f32> = r2_1;
    var y3_0 : vec4<f32> = r3_1;
    var _S64 : u32 = P_0(u32(44));
    var v_8 : f32;
    if(_S64 != u32(4294967295))
    {
        var _S65 : u32 = _S38 * lanes_3;
        var _S66 : u32 = _S65 + cb_0;
        th_0[_S66] = silu_0(r0_1[u32(0)]);
        var _S67 : u32 = _S38 + u32(1);
        var _S68 : u32 = _S67 * lanes_3;
        var _S69 : u32 = _S68 + cb_0;
        th_0[_S69] = silu_0(r1_1[u32(0)]);
        var _S70 : u32 = _S38 + u32(2);
        var _S71 : u32 = _S70 * lanes_3;
        var _S72 : u32 = _S71 + cb_0;
        th_0[_S72] = silu_0(r2_1[u32(0)]);
        var _S73 : u32 = _S38 + u32(3);
        var _S74 : u32 = _S73 * lanes_3;
        var _S75 : u32 = _S74 + cb_0;
        th_0[_S75] = silu_0(r3_1[u32(0)]);
        th_0[_S66 + u32(1)] = silu_0(r0_1[u32(1)]);
        th_0[_S69 + u32(1)] = silu_0(r1_1[u32(1)]);
        th_0[_S72 + u32(1)] = silu_0(r2_1[u32(1)]);
        th_0[_S75 + u32(1)] = silu_0(r3_1[u32(1)]);
        th_0[_S66 + u32(2)] = silu_0(r0_1[u32(2)]);
        th_0[_S69 + u32(2)] = silu_0(r1_1[u32(2)]);
        th_0[_S72 + u32(2)] = silu_0(r2_1[u32(2)]);
        th_0[_S75 + u32(2)] = silu_0(r3_1[u32(2)]);
        th_0[_S66 + u32(3)] = silu_0(r0_1[u32(3)]);
        th_0[_S69 + u32(3)] = silu_0(r1_1[u32(3)]);
        th_0[_S72 + u32(3)] = silu_0(r2_1[u32(3)]);
        th_0[_S75 + u32(3)] = silu_0(r3_1[u32(3)]);
        workgroupBarrier();
        var b2_0 : vec4<f32> = _S44;
        if(cb_0 < out_1)
        {
            m_0 = wts_0[P_0(u32(53)) + cb_0];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0[u32(0)] = m_0;
        var _S76 : u32 = cb_0 + u32(1);
        if(_S76 < out_1)
        {
            m_0 = wts_0[P_0(u32(53)) + cb_0 + u32(1)];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0[u32(1)] = m_0;
        var _S77 : u32 = cb_0 + u32(2);
        if(_S77 < out_1)
        {
            m_0 = wts_0[P_0(u32(53)) + cb_0 + u32(2)];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0[u32(2)] = m_0;
        var _S78 : u32 = cb_0 + u32(3);
        if(_S78 < out_1)
        {
            m_0 = wts_0[P_0(u32(53)) + cb_0 + u32(3)];
        }
        else
        {
            m_0 = 0.0f;
        }
        b2_0[u32(3)] = m_0;
        y0_0 = b2_0;
        y1_0 = b2_0;
        y2_0 = b2_0;
        y3_0 = b2_0;
        k0_0 = u32(0);
        loop
        {
            if(k0_0 < _S42)
            {
            }
            else
            {
                break;
            }
            q_0 = t_1;
            loop
            {
                if(q_0 < (u32(8) * lanes_3))
                {
                }
                else
                {
                    break;
                }
                var kk_2 : u32 = q_0 / lanes_3;
                var c_3 : u32 = q_0 - kk_2 * lanes_3;
                if(c_3 < out_1)
                {
                    m_0 = wts_0[_S64 + (k0_0 + kk_2) * out_1 + c_3];
                }
                else
                {
                    m_0 = 0.0f;
                }
                tw_0[kk_2][c_3] = m_0;
                q_0 = q_0 + u32(128);
            }
            workgroupBarrier();
            var b_8 : vec4<f32> = vec4<f32>(tw_0[u32(0)][cb_0], tw_0[u32(0)][_S76], tw_0[u32(0)][_S77], tw_0[u32(0)][_S78]);
            var _S79 : u32 = _S65 + k0_0;
            var _S80 : u32 = _S68 + k0_0;
            var _S81 : u32 = _S71 + k0_0;
            var _S82 : u32 = _S74 + k0_0;
            var b_9 : vec4<f32> = vec4<f32>(tw_0[u32(1)][cb_0], tw_0[u32(1)][_S76], tw_0[u32(1)][_S77], tw_0[u32(1)][_S78]);
            var b_10 : vec4<f32> = vec4<f32>(tw_0[u32(2)][cb_0], tw_0[u32(2)][_S76], tw_0[u32(2)][_S77], tw_0[u32(2)][_S78]);
            var b_11 : vec4<f32> = vec4<f32>(tw_0[u32(3)][cb_0], tw_0[u32(3)][_S76], tw_0[u32(3)][_S77], tw_0[u32(3)][_S78]);
            var b_12 : vec4<f32> = vec4<f32>(tw_0[u32(4)][cb_0], tw_0[u32(4)][_S76], tw_0[u32(4)][_S77], tw_0[u32(4)][_S78]);
            var b_13 : vec4<f32> = vec4<f32>(tw_0[u32(5)][cb_0], tw_0[u32(5)][_S76], tw_0[u32(5)][_S77], tw_0[u32(5)][_S78]);
            var b_14 : vec4<f32> = vec4<f32>(tw_0[u32(6)][cb_0], tw_0[u32(6)][_S76], tw_0[u32(6)][_S77], tw_0[u32(6)][_S78]);
            var _S83 : vec4<f32> = y1_0 + vec4<f32>(th_0[_S80]) * b_8 + vec4<f32>(th_0[_S80 + u32(1)]) * b_9 + vec4<f32>(th_0[_S80 + u32(2)]) * b_10 + vec4<f32>(th_0[_S80 + u32(3)]) * b_11 + vec4<f32>(th_0[_S80 + u32(4)]) * b_12 + vec4<f32>(th_0[_S80 + u32(5)]) * b_13 + vec4<f32>(th_0[_S80 + u32(6)]) * b_14;
            var _S84 : vec4<f32> = y2_0 + vec4<f32>(th_0[_S81]) * b_8 + vec4<f32>(th_0[_S81 + u32(1)]) * b_9 + vec4<f32>(th_0[_S81 + u32(2)]) * b_10 + vec4<f32>(th_0[_S81 + u32(3)]) * b_11 + vec4<f32>(th_0[_S81 + u32(4)]) * b_12 + vec4<f32>(th_0[_S81 + u32(5)]) * b_13 + vec4<f32>(th_0[_S81 + u32(6)]) * b_14;
            var _S85 : vec4<f32> = y3_0 + vec4<f32>(th_0[_S82]) * b_8 + vec4<f32>(th_0[_S82 + u32(1)]) * b_9 + vec4<f32>(th_0[_S82 + u32(2)]) * b_10 + vec4<f32>(th_0[_S82 + u32(3)]) * b_11 + vec4<f32>(th_0[_S82 + u32(4)]) * b_12 + vec4<f32>(th_0[_S82 + u32(5)]) * b_13 + vec4<f32>(th_0[_S82 + u32(6)]) * b_14;
            var b_15 : vec4<f32> = vec4<f32>(tw_0[u32(7)][cb_0], tw_0[u32(7)][_S76], tw_0[u32(7)][_S77], tw_0[u32(7)][_S78]);
            y0_0 = y0_0 + vec4<f32>(th_0[_S79]) * b_8 + vec4<f32>(th_0[_S79 + u32(1)]) * b_9 + vec4<f32>(th_0[_S79 + u32(2)]) * b_10 + vec4<f32>(th_0[_S79 + u32(3)]) * b_11 + vec4<f32>(th_0[_S79 + u32(4)]) * b_12 + vec4<f32>(th_0[_S79 + u32(5)]) * b_13 + vec4<f32>(th_0[_S79 + u32(6)]) * b_14 + vec4<f32>(th_0[_S79 + u32(7)]) * b_15;
            y1_0 = _S83 + vec4<f32>(th_0[_S80 + u32(7)]) * b_15;
            y2_0 = _S84 + vec4<f32>(th_0[_S81 + u32(7)]) * b_15;
            y3_0 = _S85 + vec4<f32>(th_0[_S82 + u32(7)]) * b_15;
            workgroupBarrier();
            k0_0 = k0_0 + u32(8);
        }
        var _S86 : u32 = P_0(u32(45));
        if(_S86 != u32(4294967295))
        {
            th_0[_S66] = y0_0[u32(0)];
            th_0[_S69] = y1_0[u32(0)];
            th_0[_S72] = y2_0[u32(0)];
            th_0[_S75] = y3_0[u32(0)];
            th_0[_S66 + u32(1)] = y0_0[u32(1)];
            th_0[_S69 + u32(1)] = y1_0[u32(1)];
            th_0[_S72 + u32(1)] = y2_0[u32(1)];
            th_0[_S75 + u32(1)] = y3_0[u32(1)];
            th_0[_S66 + u32(2)] = y0_0[u32(2)];
            th_0[_S69 + u32(2)] = y1_0[u32(2)];
            th_0[_S72 + u32(2)] = y2_0[u32(2)];
            th_0[_S75 + u32(2)] = y3_0[u32(2)];
            th_0[_S66 + u32(3)] = y0_0[u32(3)];
            th_0[_S69 + u32(3)] = y1_0[u32(3)];
            th_0[_S72 + u32(3)] = y2_0[u32(3)];
            th_0[_S75 + u32(3)] = y3_0[u32(3)];
            workgroupBarrier();
            if(t_1 < tr_0)
            {
                q_0 = u32(0);
                m_0 = 0.0f;
                loop
                {
                    if(q_0 < out_1)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var m_1 : f32 = m_0 + th_0[t_1 * lanes_3 + q_0];
                    q_0 = q_0 + u32(1);
                    m_0 = m_1;
                }
                var _S87 : f32 = f32(out_1);
                var m_2 : f32 = m_0 / _S87;
                q_0 = u32(0);
                v_8 = 0.0f;
                loop
                {
                    if(q_0 < out_1)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var d_2 : f32 = th_0[t_1 * lanes_3 + q_0] - m_2;
                    var v_9 : f32 = v_8 + d_2 * d_2;
                    q_0 = q_0 + u32(1);
                    v_8 = v_9;
                }
                tstat_0[i32(0)][t_1] = m_2;
                tstat_0[i32(1)][t_1] = 1.0f / sqrt(v_8 / _S87 + 0.00000999999974738f);
            }
            workgroupBarrier();
            var _S88 : u32 = _S86 + cb_0;
            var _S89 : u32 = _S86 + out_1;
            var _S90 : u32 = _S89 + cb_0;
            y0_0[u32(0)] = (y0_0[u32(0)] - tstat_0[i32(0)][_S38]) * tstat_0[i32(1)][_S38] * wts_0[_S88] + wts_0[_S90];
            y1_0[u32(0)] = (y1_0[u32(0)] - tstat_0[i32(0)][_S67]) * tstat_0[i32(1)][_S67] * wts_0[_S88] + wts_0[_S90];
            y2_0[u32(0)] = (y2_0[u32(0)] - tstat_0[i32(0)][_S70]) * tstat_0[i32(1)][_S70] * wts_0[_S88] + wts_0[_S90];
            y3_0[u32(0)] = (y3_0[u32(0)] - tstat_0[i32(0)][_S73]) * tstat_0[i32(1)][_S73] * wts_0[_S88] + wts_0[_S90];
            var _S91 : u32 = _S86 + _S76;
            var _S92 : u32 = _S89 + _S76;
            y0_0[u32(1)] = (y0_0[u32(1)] - tstat_0[i32(0)][_S38]) * tstat_0[i32(1)][_S38] * wts_0[_S91] + wts_0[_S92];
            y1_0[u32(1)] = (y1_0[u32(1)] - tstat_0[i32(0)][_S67]) * tstat_0[i32(1)][_S67] * wts_0[_S91] + wts_0[_S92];
            y2_0[u32(1)] = (y2_0[u32(1)] - tstat_0[i32(0)][_S70]) * tstat_0[i32(1)][_S70] * wts_0[_S91] + wts_0[_S92];
            y3_0[u32(1)] = (y3_0[u32(1)] - tstat_0[i32(0)][_S73]) * tstat_0[i32(1)][_S73] * wts_0[_S91] + wts_0[_S92];
            var _S93 : u32 = _S86 + _S77;
            var _S94 : u32 = _S89 + _S77;
            y0_0[u32(2)] = (y0_0[u32(2)] - tstat_0[i32(0)][_S38]) * tstat_0[i32(1)][_S38] * wts_0[_S93] + wts_0[_S94];
            y1_0[u32(2)] = (y1_0[u32(2)] - tstat_0[i32(0)][_S67]) * tstat_0[i32(1)][_S67] * wts_0[_S93] + wts_0[_S94];
            y2_0[u32(2)] = (y2_0[u32(2)] - tstat_0[i32(0)][_S70]) * tstat_0[i32(1)][_S70] * wts_0[_S93] + wts_0[_S94];
            y3_0[u32(2)] = (y3_0[u32(2)] - tstat_0[i32(0)][_S73]) * tstat_0[i32(1)][_S73] * wts_0[_S93] + wts_0[_S94];
            var _S95 : u32 = _S86 + _S78;
            var _S96 : u32 = _S89 + _S78;
            y0_0[u32(3)] = (y0_0[u32(3)] - tstat_0[i32(0)][_S38]) * tstat_0[i32(1)][_S38] * wts_0[_S95] + wts_0[_S96];
            y1_0[u32(3)] = (y1_0[u32(3)] - tstat_0[i32(0)][_S67]) * tstat_0[i32(1)][_S67] * wts_0[_S95] + wts_0[_S96];
            y2_0[u32(3)] = (y2_0[u32(3)] - tstat_0[i32(0)][_S70]) * tstat_0[i32(1)][_S70] * wts_0[_S95] + wts_0[_S96];
            y3_0[u32(3)] = (y3_0[u32(3)] - tstat_0[i32(0)][_S73]) * tstat_0[i32(1)][_S73] * wts_0[_S95] + wts_0[_S96];
        }
    }
    var _S97 : f32 = (bitcast<f32>((P_0(u32(51)))));
    var _S98 : u32 = P_0(u32(48));
    var _S99 : u32 = _S40 + _S38;
    var _S100 : vec4<f32> = y0_0;
    if(_S99 < _S41)
    {
        if(cb_0 < out_1)
        {
            var v_10 : f32 = _S97 * _S100[u32(0)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_10 + arena_0[_S98 + _S99 * P_0(u32(49)) + cb_0];
            }
            else
            {
                v_8 = v_10;
            }
            var o_6 : u32 = P_0(u32(46)) + _S99 * P_0(u32(47)) + cb_0;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_6] = arena_0[o_6] + v_8;
            }
            else
            {
                arena_0[o_6] = v_8;
            }
        }
        var c_4 : u32 = cb_0 + u32(1);
        if(c_4 < out_1)
        {
            var v_11 : f32 = _S97 * _S100[u32(1)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_11 + arena_0[_S98 + _S99 * P_0(u32(49)) + c_4];
            }
            else
            {
                v_8 = v_11;
            }
            var o_7 : u32 = P_0(u32(46)) + _S99 * P_0(u32(47)) + c_4;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_7] = arena_0[o_7] + v_8;
            }
            else
            {
                arena_0[o_7] = v_8;
            }
        }
        var c_5 : u32 = cb_0 + u32(2);
        if(c_5 < out_1)
        {
            var v_12 : f32 = _S97 * _S100[u32(2)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_12 + arena_0[_S98 + _S99 * P_0(u32(49)) + c_5];
            }
            else
            {
                v_8 = v_12;
            }
            var o_8 : u32 = P_0(u32(46)) + _S99 * P_0(u32(47)) + c_5;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_8] = arena_0[o_8] + v_8;
            }
            else
            {
                arena_0[o_8] = v_8;
            }
        }
        var c_6 : u32 = cb_0 + u32(3);
        if(c_6 < out_1)
        {
            var v_13 : f32 = _S97 * _S100[u32(3)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_13 + arena_0[_S98 + _S99 * P_0(u32(49)) + c_6];
            }
            else
            {
                v_8 = v_13;
            }
            var o_9 : u32 = P_0(u32(46)) + _S99 * P_0(u32(47)) + c_6;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_9] = arena_0[o_9] + v_8;
            }
            else
            {
                arena_0[o_9] = v_8;
            }
        }
    }
    var row_7 : u32 = _S99 + u32(1);
    var _S101 : vec4<f32> = y1_0;
    if(row_7 < _S41)
    {
        if(cb_0 < out_1)
        {
            var v_14 : f32 = _S97 * _S101[u32(0)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_14 + arena_0[_S98 + row_7 * P_0(u32(49)) + cb_0];
            }
            else
            {
                v_8 = v_14;
            }
            var o_10 : u32 = P_0(u32(46)) + row_7 * P_0(u32(47)) + cb_0;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_10] = arena_0[o_10] + v_8;
            }
            else
            {
                arena_0[o_10] = v_8;
            }
        }
        var c_7 : u32 = cb_0 + u32(1);
        if(c_7 < out_1)
        {
            var v_15 : f32 = _S97 * _S101[u32(1)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_15 + arena_0[_S98 + row_7 * P_0(u32(49)) + c_7];
            }
            else
            {
                v_8 = v_15;
            }
            var o_11 : u32 = P_0(u32(46)) + row_7 * P_0(u32(47)) + c_7;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_11] = arena_0[o_11] + v_8;
            }
            else
            {
                arena_0[o_11] = v_8;
            }
        }
        var c_8 : u32 = cb_0 + u32(2);
        if(c_8 < out_1)
        {
            var v_16 : f32 = _S97 * _S101[u32(2)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_16 + arena_0[_S98 + row_7 * P_0(u32(49)) + c_8];
            }
            else
            {
                v_8 = v_16;
            }
            var o_12 : u32 = P_0(u32(46)) + row_7 * P_0(u32(47)) + c_8;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_12] = arena_0[o_12] + v_8;
            }
            else
            {
                arena_0[o_12] = v_8;
            }
        }
        var c_9 : u32 = cb_0 + u32(3);
        if(c_9 < out_1)
        {
            var v_17 : f32 = _S97 * _S101[u32(3)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_17 + arena_0[_S98 + row_7 * P_0(u32(49)) + c_9];
            }
            else
            {
                v_8 = v_17;
            }
            var o_13 : u32 = P_0(u32(46)) + row_7 * P_0(u32(47)) + c_9;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_13] = arena_0[o_13] + v_8;
            }
            else
            {
                arena_0[o_13] = v_8;
            }
        }
    }
    var row_8 : u32 = _S99 + u32(2);
    var _S102 : vec4<f32> = y2_0;
    if(row_8 < _S41)
    {
        if(cb_0 < out_1)
        {
            var v_18 : f32 = _S97 * _S102[u32(0)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_18 + arena_0[_S98 + row_8 * P_0(u32(49)) + cb_0];
            }
            else
            {
                v_8 = v_18;
            }
            var o_14 : u32 = P_0(u32(46)) + row_8 * P_0(u32(47)) + cb_0;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_14] = arena_0[o_14] + v_8;
            }
            else
            {
                arena_0[o_14] = v_8;
            }
        }
        var c_10 : u32 = cb_0 + u32(1);
        if(c_10 < out_1)
        {
            var v_19 : f32 = _S97 * _S102[u32(1)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_19 + arena_0[_S98 + row_8 * P_0(u32(49)) + c_10];
            }
            else
            {
                v_8 = v_19;
            }
            var o_15 : u32 = P_0(u32(46)) + row_8 * P_0(u32(47)) + c_10;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_15] = arena_0[o_15] + v_8;
            }
            else
            {
                arena_0[o_15] = v_8;
            }
        }
        var c_11 : u32 = cb_0 + u32(2);
        if(c_11 < out_1)
        {
            var v_20 : f32 = _S97 * _S102[u32(2)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_20 + arena_0[_S98 + row_8 * P_0(u32(49)) + c_11];
            }
            else
            {
                v_8 = v_20;
            }
            var o_16 : u32 = P_0(u32(46)) + row_8 * P_0(u32(47)) + c_11;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_16] = arena_0[o_16] + v_8;
            }
            else
            {
                arena_0[o_16] = v_8;
            }
        }
        var c_12 : u32 = cb_0 + u32(3);
        if(c_12 < out_1)
        {
            var v_21 : f32 = _S97 * _S102[u32(3)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_21 + arena_0[_S98 + row_8 * P_0(u32(49)) + c_12];
            }
            else
            {
                v_8 = v_21;
            }
            var o_17 : u32 = P_0(u32(46)) + row_8 * P_0(u32(47)) + c_12;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_17] = arena_0[o_17] + v_8;
            }
            else
            {
                arena_0[o_17] = v_8;
            }
        }
    }
    var row_9 : u32 = _S99 + u32(3);
    var _S103 : vec4<f32> = y3_0;
    if(row_9 < _S41)
    {
        if(cb_0 < out_1)
        {
            var v_22 : f32 = _S97 * _S103[u32(0)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_22 + arena_0[_S98 + row_9 * P_0(u32(49)) + cb_0];
            }
            else
            {
                v_8 = v_22;
            }
            var o_18 : u32 = P_0(u32(46)) + row_9 * P_0(u32(47)) + cb_0;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_18] = arena_0[o_18] + v_8;
            }
            else
            {
                arena_0[o_18] = v_8;
            }
        }
        var c_13 : u32 = cb_0 + u32(1);
        if(c_13 < out_1)
        {
            var v_23 : f32 = _S97 * _S103[u32(1)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_23 + arena_0[_S98 + row_9 * P_0(u32(49)) + c_13];
            }
            else
            {
                v_8 = v_23;
            }
            var o_19 : u32 = P_0(u32(46)) + row_9 * P_0(u32(47)) + c_13;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_19] = arena_0[o_19] + v_8;
            }
            else
            {
                arena_0[o_19] = v_8;
            }
        }
        var c_14 : u32 = cb_0 + u32(2);
        if(c_14 < out_1)
        {
            var v_24 : f32 = _S97 * _S103[u32(2)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_24 + arena_0[_S98 + row_9 * P_0(u32(49)) + c_14];
            }
            else
            {
                v_8 = v_24;
            }
            var o_20 : u32 = P_0(u32(46)) + row_9 * P_0(u32(47)) + c_14;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_20] = arena_0[o_20] + v_8;
            }
            else
            {
                arena_0[o_20] = v_8;
            }
        }
        var c_15 : u32 = cb_0 + u32(3);
        if(c_15 < out_1)
        {
            var v_25 : f32 = _S97 * _S103[u32(3)];
            if(_S98 != u32(4294967295))
            {
                v_8 = v_25 + arena_0[_S98 + row_9 * P_0(u32(49)) + c_15];
            }
            else
            {
                v_8 = v_25;
            }
            var o_21 : u32 = P_0(u32(46)) + row_9 * P_0(u32(47)) + c_15;
            if((((P_0(u32(50))) & (u32(1)))) != u32(0))
            {
                arena_0[o_21] = arena_0[o_21] + v_8;
            }
            else
            {
                arena_0[o_21] = v_8;
            }
        }
    }
    return;
}

