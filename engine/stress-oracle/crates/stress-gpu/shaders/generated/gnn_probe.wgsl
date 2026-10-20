struct Params_std140_0
{
    @align(16) width_0 : u32,
    @align(4) nodes_0 : u32,
    @align(8) grid_x_0 : u32,
    @align(4) weights_0 : u32,
    @align(16) edges_0 : u32,
    @align(4) mode_0 : u32,
    @align(8) in_width_0 : u32,
    @align(4) max_degree_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : Params_std140_0;
@binding(6) @group(0) var<storage, read> offsets_0 : array<u32>;

@binding(7) @group(0) var<storage, read> src_0 : array<u32>;

@binding(2) @group(0) var<storage, read> h_in_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> h_out_0 : array<f32>;

@binding(1) @group(0) var<storage, read> wts_0 : array<f32>;

@binding(4) @group(0) var<storage, read> e_in_0 : array<f32>;

@binding(5) @group(0) var<storage, read_write> e_out_0 : array<f32>;

fn rsqrt_0( x_0 : f32) -> f32
{
    return 1.0f / sqrt(x_0);
}

var<workgroup> xs_0 : array<array<f32, i32(384)>, i32(4)>;

fn silu_0( x_1 : f32) -> f32
{
    return x_1 / (1.0f + exp(- x_1));
}

var<workgroup> hid_0 : array<f32, i32(128)>;

var<workgroup> red_0 : array<f32, i32(128)>;

fn slot_sum_0( v_0 : f32,  tid_0 : u32,  lane_0 : u32,  w_0 : u32) -> f32
{
    red_0[tid_0] = v_0;
    workgroupBarrier();
    var s_0 : u32 = w_0 / u32(2);
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

fn mlp_0( base_0 : u32,  k_in_0 : u32,  tid_1 : u32,  slot_0 : u32,  lane_1 : u32,  w_1 : u32,  live_0 : bool) -> f32
{
    var k_0 : u32;
    var a_0 : f32;
    if(live_0)
    {
        var _S1 : f32 = wts_0[base_0 + k_in_0 * w_1 + lane_1];
        k_0 = u32(0);
        a_0 = _S1;
        loop
        {
            if(k_0 < k_in_0)
            {
            }
            else
            {
                break;
            }
            var a_1 : f32 = a_0 + wts_0[base_0 + k_0 * w_1 + lane_1] * xs_0[slot_0][k_0];
            k_0 = k_0 + u32(1);
            a_0 = a_1;
        }
        a_0 = silu_0(a_0);
    }
    else
    {
        a_0 = 0.0f;
    }
    hid_0[tid_1] = a_0;
    workgroupBarrier();
    var b2_0 : u32 = base_0 + k_in_0 * w_1 + w_1;
    var acc_0 : f32;
    if(live_0)
    {
        var _S2 : f32 = wts_0[b2_0 + w_1 * w_1 + lane_1];
        k_0 = u32(0);
        acc_0 = _S2;
        loop
        {
            if(k_0 < w_1)
            {
            }
            else
            {
                break;
            }
            var acc_1 : f32 = acc_0 + wts_0[b2_0 + k_0 * w_1 + lane_1] * hid_0[tid_1 - lane_1 + k_0];
            k_0 = k_0 + u32(1);
            acc_0 = acc_1;
        }
    }
    else
    {
        acc_0 = 0.0f;
    }
    workgroupBarrier();
    var _S3 : f32 = slot_sum_0(acc_0, tid_1, lane_1, w_1);
    var _S4 : f32 = f32(w_1);
    var d_0 : f32 = acc_0 - _S3 / _S4;
    var _S5 : f32 = slot_sum_0(d_0 * d_0, tid_1, lane_1, w_1);
    var g_0 : u32 = b2_0 + w_1 * w_1 + w_1;
    return wts_0[g_0 + lane_1] * d_0 * rsqrt_0(_S5 / _S4 + 0.00000999999974738f) + wts_0[g_0 + w_1 + lane_1];
}

fn mlp_size_0( k_in_1 : u32,  w_2 : u32) -> u32
{
    return k_in_1 * w_2 + w_2 + w_2 * w_2 + w_2 + u32(2) * w_2;
}

@compute
@workgroup_size(128, 1, 1)
fn gnn_round(@builtin(workgroup_id) gid_0 : vec3<u32>, @builtin(local_invocation_id) lid_0 : vec3<u32>)
{
    var w_3 : u32 = params_0.width_0;
    var tid_2 : u32 = lid_0.x;
    var slot_1 : u32 = tid_2 / params_0.width_0;
    var lane_2 : u32 = tid_2 % w_3;
    var group_0 : u32 = gid_0.y * params_0.grid_x_0 + gid_0.x;
    var _S6 : u32 = u32(128) / w_3;
    var node_0 : u32 = group_0 * _S6 + slot_1;
    var live_1 : bool = node_0 < (params_0.nodes_0);
    var on_0 : bool;
    if((params_0.mode_0) == u32(3))
    {
        on_0 = true;
    }
    else
    {
        on_0 = (params_0.mode_0) == u32(4);
    }
    var i_0 : u32;
    var s_1 : f32;
    if(on_0)
    {
        if(live_1)
        {
            on_0 = (params_0.mode_0) == u32(3);
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            var a_2 : u32 = offsets_0[node_0];
            var b_0 : u32 = offsets_0[node_0 + u32(1)];
            i_0 = a_2;
            s_1 = 0.0f;
            loop
            {
                if(i_0 < b_0)
                {
                }
                else
                {
                    break;
                }
                var s_2 : f32 = s_1 + h_in_0[src_0[i_0] * w_3 + lane_2];
                i_0 = i_0 + u32(1);
                s_1 = s_2;
            }
            h_out_0[node_0 * w_3 + lane_2] = s_1 / f32(max(b_0 - a_2, u32(1)));
        }
        else
        {
            if(live_1)
            {
                var _S7 : u32 = node_0 * w_3 + lane_2;
                h_out_0[_S7] = h_out_0[_S7] + h_in_0[src_0[node_0] * w_3 + lane_2];
            }
        }
        return;
    }
    if((params_0.mode_0) == u32(1))
    {
        on_0 = true;
    }
    else
    {
        on_0 = (params_0.mode_0) == u32(2);
    }
    if(on_0)
    {
        if((params_0.mode_0) == u32(1))
        {
            i_0 = params_0.in_width_0;
        }
        else
        {
            i_0 = w_3;
        }
        if(live_1)
        {
            on_0 = lane_2 < i_0;
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            xs_0[slot_1][lane_2] = h_in_0[node_0 * i_0 + lane_2];
        }
        workgroupBarrier();
        var y_0 : f32 = mlp_0(params_0.weights_0, i_0, tid_2, slot_1, lane_2, w_3, live_1);
        if(live_1)
        {
            if((params_0.mode_0) == u32(1))
            {
                on_0 = true;
            }
            else
            {
                on_0 = lane_2 < (params_0.in_width_0);
            }
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            if((params_0.mode_0) == u32(1))
            {
                i_0 = w_3;
            }
            else
            {
                i_0 = params_0.in_width_0;
            }
            h_out_0[node_0 * i_0 + lane_2] = y_0;
        }
        return;
    }
    if((params_0.mode_0) == u32(5))
    {
        var on_1 : bool = node_0 < (params_0.edges_0);
        if(on_1)
        {
            xs_0[slot_1][lane_2] = e_in_0[node_0 * w_3 + lane_2];
            xs_0[slot_1][w_3 + lane_2] = h_in_0[src_0[node_0] * w_3 + lane_2];
            xs_0[slot_1][u32(2) * w_3 + lane_2] = h_in_0[offsets_0[node_0] * w_3 + lane_2];
        }
        workgroupBarrier();
        var y_1 : f32 = mlp_0(params_0.weights_0, u32(3) * w_3, tid_2, slot_1, lane_2, w_3, on_1);
        if(on_1)
        {
            e_out_0[node_0 * w_3 + lane_2] = xs_0[slot_1][lane_2] + y_1;
        }
        return;
    }
    var agg_0 : f32;
    if((params_0.mode_0) == u32(6))
    {
        if(live_1)
        {
            var hd_0 : f32 = h_in_0[node_0 * w_3 + lane_2];
            i_0 = offsets_0[node_0];
            agg_0 = 0.0f;
            loop
            {
                if(i_0 < offsets_0[node_0 + u32(1)])
                {
                }
                else
                {
                    break;
                }
                var agg_1 : f32 = agg_0 + e_in_0[i_0 * w_3 + lane_2];
                i_0 = i_0 + u32(1);
                agg_0 = agg_1;
            }
            xs_0[slot_1][lane_2] = hd_0;
            xs_0[slot_1][w_3 + lane_2] = agg_0;
            s_1 = hd_0;
        }
        else
        {
            s_1 = 0.0f;
        }
        workgroupBarrier();
        var y_2 : f32 = mlp_0(params_0.weights_0 + mlp_size_0(u32(3) * w_3, w_3), u32(2) * w_3, tid_2, slot_1, lane_2, w_3, live_1);
        if(live_1)
        {
            h_out_0[node_0 * w_3 + lane_2] = s_1 + y_2;
        }
        return;
    }
    var edge_w_0 : u32 = params_0.weights_0;
    var _S8 : u32 = u32(3) * w_3;
    var node_w_0 : u32 = params_0.weights_0 + mlp_size_0(_S8, w_3);
    if(live_1)
    {
        i_0 = offsets_0[node_0];
    }
    else
    {
        i_0 = u32(0);
    }
    var _S9 : u32;
    if(live_1)
    {
        _S9 = offsets_0[node_0 + u32(1)];
    }
    else
    {
        _S9 = u32(0);
    }
    var _S10 : u32 = params_0.max_degree_0;
    if(live_1)
    {
        s_1 = h_in_0[node_0 * w_3 + lane_2];
    }
    else
    {
        s_1 = 0.0f;
    }
    var j_0 : u32 = u32(0);
    agg_0 = 0.0f;
    loop
    {
        if(j_0 < _S10)
        {
        }
        else
        {
            break;
        }
        var e_0 : u32 = i_0 + j_0;
        if(live_1)
        {
            on_0 = e_0 < _S9;
        }
        else
        {
            on_0 = false;
        }
        if(on_0)
        {
            var s_3 : u32 = src_0[e_0];
            xs_0[slot_1][lane_2] = e_in_0[e_0 * w_3 + lane_2];
            xs_0[slot_1][w_3 + lane_2] = h_in_0[s_3 * w_3 + lane_2];
            xs_0[slot_1][u32(2) * w_3 + lane_2] = s_1;
        }
        workgroupBarrier();
        var y_3 : f32 = mlp_0(edge_w_0, _S8, tid_2, slot_1, lane_2, w_3, on_0);
        if(on_0)
        {
            var en_0 : f32 = xs_0[slot_1][lane_2] + y_3;
            e_out_0[e_0 * w_3 + lane_2] = en_0;
            agg_0 = agg_0 + en_0;
        }
        workgroupBarrier();
        j_0 = j_0 + u32(1);
    }
    if(live_1)
    {
        xs_0[slot_1][lane_2] = s_1;
        xs_0[slot_1][w_3 + lane_2] = agg_0;
    }
    workgroupBarrier();
    var y_4 : f32 = mlp_0(node_w_0, u32(2) * w_3, tid_2, slot_1, lane_2, w_3, live_1);
    if(live_1)
    {
        h_out_0[node_0 * w_3 + lane_2] = s_1 + y_4;
    }
    return;
}

var<workgroup> tx_0 : array<array<f32, i32(64)>, i32(8)>;

fn gather_0( row_0 : u32,  k_1 : u32,  w_4 : u32,  mode_1 : u32) -> f32
{
    var seg_0 : u32 = k_1 / w_4;
    var c_0 : u32 = k_1 - seg_0 * w_4;
    if(seg_0 == u32(0))
    {
        return h_in_0[row_0 * w_4 + c_0];
    }
    if(mode_1 == u32(8))
    {
        return e_in_0[row_0 * w_4 + c_0];
    }
    var n_0 : u32;
    if(seg_0 == u32(1))
    {
        n_0 = src_0[row_0];
    }
    else
    {
        n_0 = offsets_0[row_0];
    }
    return e_in_0[n_0 * w_4 + c_0];
}

var<workgroup> tw_0 : array<array<f32, i32(128)>, i32(8)>;

var<workgroup> th_0 : array<f32, i32(2048)>;

var<workgroup> tstat_0 : array<array<f32, i32(64)>, i32(2)>;

@compute
@workgroup_size(128, 1, 1)
fn gnn_mlp_tiled(@builtin(workgroup_id) gid_1 : vec3<u32>, @builtin(local_invocation_id) lid_1 : vec3<u32>)
{
    var q_0 : u32;
    var m_0 : f32;
    var w_5 : u32 = params_0.width_0;
    var tr_0 : u32 = u32(2048) / params_0.width_0;
    var t_0 : u32 = lid_1.x;
    var cg_0 : u32 = w_5 / u32(4);
    var _S11 : u32 = t_0 / cg_0;
    var _S12 : u32 = _S11 * u32(4);
    var _S13 : u32 = t_0 % cg_0;
    var cb_0 : u32 = _S13 * u32(4);
    var _S14 : u32 = (gid_1.y * params_0.grid_x_0 + gid_1.x) * tr_0;
    var _S15 : u32 = params_0.nodes_0;
    var _S16 : i32;
    if((params_0.mode_0) == u32(7))
    {
        _S16 = i32(3);
    }
    else
    {
        _S16 = i32(2);
    }
    var k_in_2 : u32 = u32(_S16) * w_5;
    var _S17 : u32 = params_0.weights_0;
    var k0_0 : u32;
    if((params_0.mode_0) == u32(8))
    {
        k0_0 = mlp_size_0(u32(3) * w_5, w_5);
    }
    else
    {
        k0_0 = u32(0);
    }
    var base_1 : u32 = _S17 + k0_0;
    var _S18 : u32 = base_1 + k_in_2 * w_5;
    var _S19 : u32 = _S18 + cb_0;
    var bias_0 : vec4<f32> = vec4<f32>(wts_0[_S19], wts_0[_S19 + u32(1)], wts_0[_S19 + u32(2)], wts_0[_S19 + u32(3)]);
    k0_0 = u32(0);
    var r0_0 : vec4<f32> = bias_0;
    var r1_0 : vec4<f32> = bias_0;
    var r2_0 : vec4<f32> = bias_0;
    var r3_0 : vec4<f32> = bias_0;
    loop
    {
        if(k0_0 < k_in_2)
        {
        }
        else
        {
            break;
        }
        q_0 = t_0;
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
            var r_1 : u32 = q_0 - kk_0 * tr_0;
            var row_1 : u32 = _S14 + r_1;
            if(row_1 < _S15)
            {
                var _S20 : f32 = gather_0(row_1, k0_0 + kk_0, w_5, params_0.mode_0);
                m_0 = _S20;
            }
            else
            {
                m_0 = 0.0f;
            }
            tx_0[kk_0][r_1] = m_0;
            q_0 = q_0 + u32(128);
        }
        var q_1 : u32 = t_0;
        loop
        {
            if(q_1 < (u32(8) * w_5))
            {
            }
            else
            {
                break;
            }
            var kk_1 : u32 = q_1 / w_5;
            var c_1 : u32 = q_1 - kk_1 * w_5;
            tw_0[kk_1][c_1] = wts_0[base_1 + (k0_0 + kk_1) * w_5 + c_1];
            q_1 = q_1 + u32(128);
        }
        workgroupBarrier();
        var _S21 : u32 = cb_0 + u32(1);
        var _S22 : u32 = cb_0 + u32(2);
        var _S23 : u32 = cb_0 + u32(3);
        var b_1 : vec4<f32> = vec4<f32>(tw_0[u32(0)][cb_0], tw_0[u32(0)][_S21], tw_0[u32(0)][_S22], tw_0[u32(0)][_S23]);
        var _S24 : u32 = _S12 + u32(1);
        var _S25 : u32 = _S12 + u32(2);
        var _S26 : u32 = _S12 + u32(3);
        var b_2 : vec4<f32> = vec4<f32>(tw_0[u32(1)][cb_0], tw_0[u32(1)][_S21], tw_0[u32(1)][_S22], tw_0[u32(1)][_S23]);
        var b_3 : vec4<f32> = vec4<f32>(tw_0[u32(2)][cb_0], tw_0[u32(2)][_S21], tw_0[u32(2)][_S22], tw_0[u32(2)][_S23]);
        var b_4 : vec4<f32> = vec4<f32>(tw_0[u32(3)][cb_0], tw_0[u32(3)][_S21], tw_0[u32(3)][_S22], tw_0[u32(3)][_S23]);
        var b_5 : vec4<f32> = vec4<f32>(tw_0[u32(4)][cb_0], tw_0[u32(4)][_S21], tw_0[u32(4)][_S22], tw_0[u32(4)][_S23]);
        var b_6 : vec4<f32> = vec4<f32>(tw_0[u32(5)][cb_0], tw_0[u32(5)][_S21], tw_0[u32(5)][_S22], tw_0[u32(5)][_S23]);
        var b_7 : vec4<f32> = vec4<f32>(tw_0[u32(6)][cb_0], tw_0[u32(6)][_S21], tw_0[u32(6)][_S22], tw_0[u32(6)][_S23]);
        var b_8 : vec4<f32> = vec4<f32>(tw_0[u32(7)][cb_0], tw_0[u32(7)][_S21], tw_0[u32(7)][_S22], tw_0[u32(7)][_S23]);
        var r0_1 : vec4<f32> = r0_0 + vec4<f32>(tx_0[u32(0)][_S12]) * b_1 + vec4<f32>(tx_0[u32(1)][_S12]) * b_2 + vec4<f32>(tx_0[u32(2)][_S12]) * b_3 + vec4<f32>(tx_0[u32(3)][_S12]) * b_4 + vec4<f32>(tx_0[u32(4)][_S12]) * b_5 + vec4<f32>(tx_0[u32(5)][_S12]) * b_6 + vec4<f32>(tx_0[u32(6)][_S12]) * b_7 + vec4<f32>(tx_0[u32(7)][_S12]) * b_8;
        var r1_1 : vec4<f32> = r1_0 + vec4<f32>(tx_0[u32(0)][_S24]) * b_1 + vec4<f32>(tx_0[u32(1)][_S24]) * b_2 + vec4<f32>(tx_0[u32(2)][_S24]) * b_3 + vec4<f32>(tx_0[u32(3)][_S24]) * b_4 + vec4<f32>(tx_0[u32(4)][_S24]) * b_5 + vec4<f32>(tx_0[u32(5)][_S24]) * b_6 + vec4<f32>(tx_0[u32(6)][_S24]) * b_7 + vec4<f32>(tx_0[u32(7)][_S24]) * b_8;
        var r2_1 : vec4<f32> = r2_0 + vec4<f32>(tx_0[u32(0)][_S25]) * b_1 + vec4<f32>(tx_0[u32(1)][_S25]) * b_2 + vec4<f32>(tx_0[u32(2)][_S25]) * b_3 + vec4<f32>(tx_0[u32(3)][_S25]) * b_4 + vec4<f32>(tx_0[u32(4)][_S25]) * b_5 + vec4<f32>(tx_0[u32(5)][_S25]) * b_6 + vec4<f32>(tx_0[u32(6)][_S25]) * b_7 + vec4<f32>(tx_0[u32(7)][_S25]) * b_8;
        var r3_1 : vec4<f32> = r3_0 + vec4<f32>(tx_0[u32(0)][_S26]) * b_1 + vec4<f32>(tx_0[u32(1)][_S26]) * b_2 + vec4<f32>(tx_0[u32(2)][_S26]) * b_3 + vec4<f32>(tx_0[u32(3)][_S26]) * b_4 + vec4<f32>(tx_0[u32(4)][_S26]) * b_5 + vec4<f32>(tx_0[u32(5)][_S26]) * b_6 + vec4<f32>(tx_0[u32(6)][_S26]) * b_7 + vec4<f32>(tx_0[u32(7)][_S26]) * b_8;
        workgroupBarrier();
        k0_0 = k0_0 + u32(8);
        r0_0 = r0_1;
        r1_0 = r1_1;
        r2_0 = r2_1;
        r3_0 = r3_1;
    }
    var _S27 : f32 = silu_0(r0_0.x);
    var _S28 : f32 = silu_0(r0_0.y);
    var _S29 : f32 = silu_0(r0_0.z);
    var _S30 : f32 = silu_0(r0_0.w);
    var _S31 : f32 = silu_0(r1_0.x);
    var _S32 : f32 = silu_0(r1_0.y);
    var _S33 : f32 = silu_0(r1_0.z);
    var _S34 : f32 = silu_0(r1_0.w);
    var _S35 : f32 = silu_0(r2_0.x);
    var _S36 : f32 = silu_0(r2_0.y);
    var _S37 : f32 = silu_0(r2_0.z);
    var _S38 : f32 = silu_0(r2_0.w);
    var _S39 : f32 = silu_0(r3_0.x);
    var _S40 : f32 = silu_0(r3_0.y);
    var _S41 : f32 = silu_0(r3_0.z);
    var _S42 : f32 = silu_0(r3_0.w);
    var _S43 : u32 = _S12 * w_5;
    var _S44 : u32 = _S43 + cb_0;
    th_0[_S44] = _S27;
    var _S45 : u32 = _S12 + u32(1);
    var _S46 : u32 = _S45 * w_5;
    var _S47 : u32 = _S46 + cb_0;
    th_0[_S47] = _S31;
    var _S48 : u32 = _S12 + u32(2);
    var _S49 : u32 = _S48 * w_5;
    var _S50 : u32 = _S49 + cb_0;
    th_0[_S50] = _S35;
    var _S51 : u32 = _S12 + u32(3);
    var _S52 : u32 = _S51 * w_5;
    var _S53 : u32 = _S52 + cb_0;
    th_0[_S53] = _S39;
    th_0[_S44 + u32(1)] = _S28;
    th_0[_S47 + u32(1)] = _S32;
    th_0[_S50 + u32(1)] = _S36;
    th_0[_S53 + u32(1)] = _S40;
    th_0[_S44 + u32(2)] = _S29;
    th_0[_S47 + u32(2)] = _S33;
    th_0[_S50 + u32(2)] = _S37;
    th_0[_S53 + u32(2)] = _S41;
    th_0[_S44 + u32(3)] = _S30;
    th_0[_S47 + u32(3)] = _S34;
    th_0[_S50 + u32(3)] = _S38;
    th_0[_S53 + u32(3)] = _S42;
    workgroupBarrier();
    var b2_1 : u32 = _S18 + w_5;
    var _S54 : u32 = b2_1 + w_5 * w_5;
    var _S55 : u32 = _S54 + cb_0;
    var bias_1 : vec4<f32> = vec4<f32>(wts_0[_S55], wts_0[_S55 + u32(1)], wts_0[_S55 + u32(2)], wts_0[_S55 + u32(3)]);
    k0_0 = u32(0);
    r0_0 = bias_1;
    r1_0 = bias_1;
    r2_0 = bias_1;
    r3_0 = bias_1;
    loop
    {
        if(k0_0 < w_5)
        {
        }
        else
        {
            break;
        }
        q_0 = t_0;
        loop
        {
            if(q_0 < (u32(8) * w_5))
            {
            }
            else
            {
                break;
            }
            var kk_2 : u32 = q_0 / w_5;
            var c_2 : u32 = q_0 - kk_2 * w_5;
            tw_0[kk_2][c_2] = wts_0[b2_1 + (k0_0 + kk_2) * w_5 + c_2];
            q_0 = q_0 + u32(128);
        }
        workgroupBarrier();
        var _S56 : u32 = cb_0 + u32(1);
        var _S57 : u32 = cb_0 + u32(2);
        var _S58 : u32 = cb_0 + u32(3);
        var b_9 : vec4<f32> = vec4<f32>(tw_0[u32(0)][cb_0], tw_0[u32(0)][_S56], tw_0[u32(0)][_S57], tw_0[u32(0)][_S58]);
        var _S59 : u32 = _S43 + k0_0;
        var _S60 : u32 = _S46 + k0_0;
        var _S61 : u32 = _S49 + k0_0;
        var _S62 : u32 = _S52 + k0_0;
        var b_10 : vec4<f32> = vec4<f32>(tw_0[u32(1)][cb_0], tw_0[u32(1)][_S56], tw_0[u32(1)][_S57], tw_0[u32(1)][_S58]);
        var b_11 : vec4<f32> = vec4<f32>(tw_0[u32(2)][cb_0], tw_0[u32(2)][_S56], tw_0[u32(2)][_S57], tw_0[u32(2)][_S58]);
        var b_12 : vec4<f32> = vec4<f32>(tw_0[u32(3)][cb_0], tw_0[u32(3)][_S56], tw_0[u32(3)][_S57], tw_0[u32(3)][_S58]);
        var b_13 : vec4<f32> = vec4<f32>(tw_0[u32(4)][cb_0], tw_0[u32(4)][_S56], tw_0[u32(4)][_S57], tw_0[u32(4)][_S58]);
        var b_14 : vec4<f32> = vec4<f32>(tw_0[u32(5)][cb_0], tw_0[u32(5)][_S56], tw_0[u32(5)][_S57], tw_0[u32(5)][_S58]);
        var b_15 : vec4<f32> = vec4<f32>(tw_0[u32(6)][cb_0], tw_0[u32(6)][_S56], tw_0[u32(6)][_S57], tw_0[u32(6)][_S58]);
        var b_16 : vec4<f32> = vec4<f32>(tw_0[u32(7)][cb_0], tw_0[u32(7)][_S56], tw_0[u32(7)][_S57], tw_0[u32(7)][_S58]);
        var r0_2 : vec4<f32> = r0_0 + vec4<f32>(th_0[_S59]) * b_9 + vec4<f32>(th_0[_S59 + u32(1)]) * b_10 + vec4<f32>(th_0[_S59 + u32(2)]) * b_11 + vec4<f32>(th_0[_S59 + u32(3)]) * b_12 + vec4<f32>(th_0[_S59 + u32(4)]) * b_13 + vec4<f32>(th_0[_S59 + u32(5)]) * b_14 + vec4<f32>(th_0[_S59 + u32(6)]) * b_15 + vec4<f32>(th_0[_S59 + u32(7)]) * b_16;
        var r1_2 : vec4<f32> = r1_0 + vec4<f32>(th_0[_S60]) * b_9 + vec4<f32>(th_0[_S60 + u32(1)]) * b_10 + vec4<f32>(th_0[_S60 + u32(2)]) * b_11 + vec4<f32>(th_0[_S60 + u32(3)]) * b_12 + vec4<f32>(th_0[_S60 + u32(4)]) * b_13 + vec4<f32>(th_0[_S60 + u32(5)]) * b_14 + vec4<f32>(th_0[_S60 + u32(6)]) * b_15 + vec4<f32>(th_0[_S60 + u32(7)]) * b_16;
        var r2_2 : vec4<f32> = r2_0 + vec4<f32>(th_0[_S61]) * b_9 + vec4<f32>(th_0[_S61 + u32(1)]) * b_10 + vec4<f32>(th_0[_S61 + u32(2)]) * b_11 + vec4<f32>(th_0[_S61 + u32(3)]) * b_12 + vec4<f32>(th_0[_S61 + u32(4)]) * b_13 + vec4<f32>(th_0[_S61 + u32(5)]) * b_14 + vec4<f32>(th_0[_S61 + u32(6)]) * b_15 + vec4<f32>(th_0[_S61 + u32(7)]) * b_16;
        var r3_2 : vec4<f32> = r3_0 + vec4<f32>(th_0[_S62]) * b_9 + vec4<f32>(th_0[_S62 + u32(1)]) * b_10 + vec4<f32>(th_0[_S62 + u32(2)]) * b_11 + vec4<f32>(th_0[_S62 + u32(3)]) * b_12 + vec4<f32>(th_0[_S62 + u32(4)]) * b_13 + vec4<f32>(th_0[_S62 + u32(5)]) * b_14 + vec4<f32>(th_0[_S62 + u32(6)]) * b_15 + vec4<f32>(th_0[_S62 + u32(7)]) * b_16;
        workgroupBarrier();
        k0_0 = k0_0 + u32(8);
        r0_0 = r0_2;
        r1_0 = r1_2;
        r2_0 = r2_2;
        r3_0 = r3_2;
    }
    var _S63 : vec4<f32> = r0_0;
    th_0[_S44] = r0_0[u32(0)];
    var _S64 : vec4<f32> = r1_0;
    th_0[_S47] = r1_0[u32(0)];
    var _S65 : vec4<f32> = r2_0;
    th_0[_S50] = r2_0[u32(0)];
    var _S66 : vec4<f32> = r3_0;
    th_0[_S53] = r3_0[u32(0)];
    var _S67 : vec4<f32> = r0_0;
    th_0[_S44 + u32(1)] = r0_0[u32(1)];
    var _S68 : vec4<f32> = r1_0;
    th_0[_S47 + u32(1)] = r1_0[u32(1)];
    var _S69 : vec4<f32> = r2_0;
    th_0[_S50 + u32(1)] = r2_0[u32(1)];
    var _S70 : vec4<f32> = r3_0;
    th_0[_S53 + u32(1)] = r3_0[u32(1)];
    var _S71 : vec4<f32> = r0_0;
    th_0[_S44 + u32(2)] = r0_0[u32(2)];
    var _S72 : vec4<f32> = r1_0;
    th_0[_S47 + u32(2)] = r1_0[u32(2)];
    var _S73 : vec4<f32> = r2_0;
    th_0[_S50 + u32(2)] = r2_0[u32(2)];
    var _S74 : vec4<f32> = r3_0;
    th_0[_S53 + u32(2)] = r3_0[u32(2)];
    var _S75 : vec4<f32> = r0_0;
    th_0[_S44 + u32(3)] = r0_0[u32(3)];
    var _S76 : vec4<f32> = r1_0;
    th_0[_S47 + u32(3)] = r1_0[u32(3)];
    var _S77 : vec4<f32> = r2_0;
    th_0[_S50 + u32(3)] = r2_0[u32(3)];
    var _S78 : vec4<f32> = r3_0;
    th_0[_S53 + u32(3)] = r3_0[u32(3)];
    workgroupBarrier();
    if(t_0 < tr_0)
    {
        q_0 = u32(0);
        m_0 = 0.0f;
        loop
        {
            if(q_0 < w_5)
            {
            }
            else
            {
                break;
            }
            var m_1 : f32 = m_0 + th_0[t_0 * w_5 + q_0];
            q_0 = q_0 + u32(1);
            m_0 = m_1;
        }
        var _S79 : f32 = f32(w_5);
        var m_2 : f32 = m_0 / _S79;
        q_0 = u32(0);
        var v_1 : f32 = 0.0f;
        loop
        {
            if(q_0 < w_5)
            {
            }
            else
            {
                break;
            }
            var d_1 : f32 = th_0[t_0 * w_5 + q_0] - m_2;
            var v_2 : f32 = v_1 + d_1 * d_1;
            q_0 = q_0 + u32(1);
            v_1 = v_2;
        }
        tstat_0[i32(0)][t_0] = m_2;
        tstat_0[i32(1)][t_0] = rsqrt_0(v_1 / _S79 + 0.00000999999974738f);
    }
    workgroupBarrier();
    var _S80 : u32 = _S54 + w_5;
    var _S81 : u32 = _S14 + _S12;
    if(_S81 < _S15)
    {
        var _S82 : u32 = _S80 + w_5;
        var _S83 : u32 = _S81 * w_5;
        var _S84 : u32 = _S83 + cb_0;
        h_out_0[_S84] = h_in_0[_S84] + (wts_0[_S80 + cb_0] * (_S63[u32(0)] - tstat_0[i32(0)][_S12]) * tstat_0[i32(1)][_S12] + wts_0[_S82 + cb_0]);
        var c_3 : u32 = cb_0 + u32(1);
        var _S85 : u32 = _S83 + c_3;
        h_out_0[_S85] = h_in_0[_S85] + (wts_0[_S80 + c_3] * (_S67[u32(1)] - tstat_0[i32(0)][_S12]) * tstat_0[i32(1)][_S12] + wts_0[_S82 + c_3]);
        var c_4 : u32 = cb_0 + u32(2);
        var _S86 : u32 = _S83 + c_4;
        h_out_0[_S86] = h_in_0[_S86] + (wts_0[_S80 + c_4] * (_S71[u32(2)] - tstat_0[i32(0)][_S12]) * tstat_0[i32(1)][_S12] + wts_0[_S82 + c_4]);
        var c_5 : u32 = cb_0 + u32(3);
        var _S87 : u32 = _S83 + c_5;
        h_out_0[_S87] = h_in_0[_S87] + (wts_0[_S80 + c_5] * (_S75[u32(3)] - tstat_0[i32(0)][_S12]) * tstat_0[i32(1)][_S12] + wts_0[_S82 + c_5]);
    }
    var row_2 : u32 = _S81 + u32(1);
    if(row_2 < _S15)
    {
        var _S88 : u32 = _S80 + w_5;
        var _S89 : u32 = row_2 * w_5;
        var _S90 : u32 = _S89 + cb_0;
        h_out_0[_S90] = h_in_0[_S90] + (wts_0[_S80 + cb_0] * (_S64[u32(0)] - tstat_0[i32(0)][_S45]) * tstat_0[i32(1)][_S45] + wts_0[_S88 + cb_0]);
        var c_6 : u32 = cb_0 + u32(1);
        var _S91 : u32 = _S89 + c_6;
        h_out_0[_S91] = h_in_0[_S91] + (wts_0[_S80 + c_6] * (_S68[u32(1)] - tstat_0[i32(0)][_S45]) * tstat_0[i32(1)][_S45] + wts_0[_S88 + c_6]);
        var c_7 : u32 = cb_0 + u32(2);
        var _S92 : u32 = _S89 + c_7;
        h_out_0[_S92] = h_in_0[_S92] + (wts_0[_S80 + c_7] * (_S72[u32(2)] - tstat_0[i32(0)][_S45]) * tstat_0[i32(1)][_S45] + wts_0[_S88 + c_7]);
        var c_8 : u32 = cb_0 + u32(3);
        var _S93 : u32 = _S89 + c_8;
        h_out_0[_S93] = h_in_0[_S93] + (wts_0[_S80 + c_8] * (_S76[u32(3)] - tstat_0[i32(0)][_S45]) * tstat_0[i32(1)][_S45] + wts_0[_S88 + c_8]);
    }
    var row_3 : u32 = _S81 + u32(2);
    if(row_3 < _S15)
    {
        var _S94 : u32 = _S80 + w_5;
        var _S95 : u32 = row_3 * w_5;
        var _S96 : u32 = _S95 + cb_0;
        h_out_0[_S96] = h_in_0[_S96] + (wts_0[_S80 + cb_0] * (_S65[u32(0)] - tstat_0[i32(0)][_S48]) * tstat_0[i32(1)][_S48] + wts_0[_S94 + cb_0]);
        var c_9 : u32 = cb_0 + u32(1);
        var _S97 : u32 = _S95 + c_9;
        h_out_0[_S97] = h_in_0[_S97] + (wts_0[_S80 + c_9] * (_S69[u32(1)] - tstat_0[i32(0)][_S48]) * tstat_0[i32(1)][_S48] + wts_0[_S94 + c_9]);
        var c_10 : u32 = cb_0 + u32(2);
        var _S98 : u32 = _S95 + c_10;
        h_out_0[_S98] = h_in_0[_S98] + (wts_0[_S80 + c_10] * (_S73[u32(2)] - tstat_0[i32(0)][_S48]) * tstat_0[i32(1)][_S48] + wts_0[_S94 + c_10]);
        var c_11 : u32 = cb_0 + u32(3);
        var _S99 : u32 = _S95 + c_11;
        h_out_0[_S99] = h_in_0[_S99] + (wts_0[_S80 + c_11] * (_S77[u32(3)] - tstat_0[i32(0)][_S48]) * tstat_0[i32(1)][_S48] + wts_0[_S94 + c_11]);
    }
    var row_4 : u32 = _S81 + u32(3);
    if(row_4 < _S15)
    {
        var _S100 : u32 = _S80 + w_5;
        var _S101 : u32 = row_4 * w_5;
        var _S102 : u32 = _S101 + cb_0;
        h_out_0[_S102] = h_in_0[_S102] + (wts_0[_S80 + cb_0] * (_S66[u32(0)] - tstat_0[i32(0)][_S51]) * tstat_0[i32(1)][_S51] + wts_0[_S100 + cb_0]);
        var c_12 : u32 = cb_0 + u32(1);
        var _S103 : u32 = _S101 + c_12;
        h_out_0[_S103] = h_in_0[_S103] + (wts_0[_S80 + c_12] * (_S70[u32(1)] - tstat_0[i32(0)][_S51]) * tstat_0[i32(1)][_S51] + wts_0[_S100 + c_12]);
        var c_13 : u32 = cb_0 + u32(2);
        var _S104 : u32 = _S101 + c_13;
        h_out_0[_S104] = h_in_0[_S104] + (wts_0[_S80 + c_13] * (_S74[u32(2)] - tstat_0[i32(0)][_S51]) * tstat_0[i32(1)][_S51] + wts_0[_S100 + c_13]);
        var c_14 : u32 = cb_0 + u32(3);
        var _S105 : u32 = _S101 + c_14;
        h_out_0[_S105] = h_in_0[_S105] + (wts_0[_S80 + c_14] * (_S78[u32(3)] - tstat_0[i32(0)][_S51]) * tstat_0[i32(1)][_S51] + wts_0[_S100 + c_14]);
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn gnn_csr_sum(@builtin(workgroup_id) gid_2 : vec3<u32>, @builtin(local_invocation_id) lid_2 : vec3<u32>)
{
    var w_6 : u32 = params_0.width_0;
    var _S106 : u32 = lid_2.x;
    var slot_2 : u32 = _S106 / params_0.width_0;
    var lane_3 : u32 = _S106 % w_6;
    var _S107 : u32 = gid_2.y * params_0.grid_x_0 + gid_2.x;
    var _S108 : u32 = u32(128) / w_6;
    var node_1 : u32 = _S107 * _S108 + slot_2;
    if(node_1 >= (params_0.nodes_0))
    {
        return;
    }
    var i_1 : u32 = offsets_0[node_1];
    var s_4 : f32 = 0.0f;
    loop
    {
        if(i_1 < offsets_0[node_1 + u32(1)])
        {
        }
        else
        {
            break;
        }
        var s_5 : f32 = s_4 + e_in_0[i_1 * w_6 + lane_3];
        i_1 = i_1 + u32(1);
        s_4 = s_5;
    }
    h_out_0[node_1 * w_6 + lane_3] = s_4;
    return;
}

