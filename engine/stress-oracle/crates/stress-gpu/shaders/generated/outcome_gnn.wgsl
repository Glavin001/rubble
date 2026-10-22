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
                _S1 = mode_0 == u32(3);
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
            if(mode_0 >= u32(3))
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
                var c_0 : u32 = lane_1;
                loop
                {
                    if(c_0 < width_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    var v_2 : f32 = arena_0[base_0 + r_1 * width_0 + c_0];
                    if(_S5 != u32(4294967295))
                    {
                        v_1 = (v_2 - wts_0[_S5 + c_0]) / wts_0[_S5 + width_0 + c_0];
                    }
                    else
                    {
                        v_1 = v_2;
                    }
                    xs_0[slot_0][k_0 + c_0] = v_1;
                    c_0 = c_0 + lanes_1;
                }
                pre_0 = pre_1;
            }
            pre_1 = pre_0;
        }
        if(mode_0 < u32(3))
        {
            k_0 = k_0 + width_0;
        }
        s_1 = s_1 + u32(1);
    }
    workgroupBarrier();
    var _S6 : u32 = P_0(u32(41));
    var _S7 : u32 = P_0(u32(40));
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
        var _S8 : u32 = P_0(u32(52));
        if(_S8 == u32(4294967295))
        {
            pre_0 = 0.0f;
        }
        else
        {
            pre_0 = wts_0[_S8 + lane_1];
        }
        r_1 = u32(0);
        loop
        {
            if(r_1 < _S6)
            {
            }
            else
            {
                break;
            }
            var a_0 : f32 = pre_0 + wts_0[_S7 + r_1 * hidden_0 + lane_1] * xs_0[slot_0][r_1];
            r_1 = r_1 + u32(1);
            pre_0 = a_0;
        }
        pre_1 = pre_1 + pre_0;
    }
    var _S9 : u32 = P_0(u32(44));
    if(_S9 == u32(4294967295))
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
            var _S10 : f32 = wts_0[P_0(u32(53)) + lane_1];
            r_1 = u32(0);
            pre_0 = _S10;
            loop
            {
                if(r_1 < hidden_0)
                {
                }
                else
                {
                    break;
                }
                var a_1 : f32 = pre_0 + wts_0[_S9 + r_1 * out_0 + lane_1] * hid_0[tid_1 - lane_1 + r_1];
                r_1 = r_1 + u32(1);
                pre_0 = a_1;
            }
        }
        else
        {
            pre_0 = 0.0f;
        }
        var _S11 : u32 = P_0(u32(45));
        if(_S11 != u32(4294967295))
        {
            var _S12 : f32 = slot_sum_0(pre_0, tid_1, lane_1, lanes_1);
            var _S13 : f32 = f32(out_0);
            var d_0 : f32 = pre_0 - _S12 / _S13;
            var _S14 : f32 = slot_sum_0(d_0 * d_0, tid_1, lane_1, lanes_1);
            pre_0 = d_0 * (1.0f / sqrt(_S14 / _S13 + 0.00000999999974738f)) * wts_0[_S11 + lane_1] + wts_0[_S11 + out_0 + lane_1];
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
        var v_3 : f32 = (bitcast<f32>((P_0(u32(51))))) * pre_0;
        var _S15 : u32 = P_0(u32(48));
        if(_S15 != u32(4294967295))
        {
            v_1 = v_3 + arena_0[_S15 + row_0 * P_0(u32(49)) + lane_1];
        }
        else
        {
            v_1 = v_3;
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
    var _S16 : u32 = lid_1.x;
    var slot_1 : u32 = _S16 / lanes_2;
    var lane_2 : u32 = _S16 % lanes_2;
    var _S17 : u32 = gid_1.y * P_0(u32(2)) + gid_1.x;
    var _S18 : u32 = u32(128) / lanes_2;
    var row_1 : u32 = _S17 * _S18 + slot_1;
    var _S19 : bool;
    if(row_1 >= (P_0(u32(1))))
    {
        _S19 = true;
    }
    else
    {
        _S19 = lane_2 >= (P_0(u32(4)));
    }
    if(_S19)
    {
        return;
    }
    var _S20 : u32 = P_0(u32(5)) + row_1;
    var _S21 : u32 = table_0[_S20 + u32(1)];
    var _S22 : u32 = P_0(u32(6));
    var e_0 : u32 = table_0[_S20];
    var s_2 : f32 = 0.0f;
    loop
    {
        if(e_0 < _S21)
        {
        }
        else
        {
            break;
        }
        var _S23 : u32 = _S22 + u32(2) * e_0;
        var s_3 : f32 = s_2 + arena_0[table_0[_S23] + lane_2] * (bitcast<f32>((table_0[_S23 + u32(1)])));
        e_0 = e_0 + u32(1);
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
    var _S24 : u32 = u32(4) * r_2;
    var t_0 : u32 = P_0(u32(5)) + _S24;
    var pb_0 : u32 = P_0(u32(4));
    var _S25 : u32 = pb_0 + u32(3) * table_0[t_0];
    var _S26 : u32 = pb_0 + u32(3) * table_0[t_0 + u32(1)];
    var d_1 : vec3<f32> = vec3<f32>(arena_0[_S25] - arena_0[_S26], arena_0[_S25 + u32(1)] - arena_0[_S26 + u32(1)], arena_0[_S25 + u32(2)] - arena_0[_S26 + u32(2)]);
    var qb_0 : u32 = P_0(u32(6)) + u32(4) * table_0[t_0 + u32(2)];
    var u_0 : vec3<f32> = (vec3<f32>(0) - vec3<f32>(arena_0[qb_0 + u32(1)], arena_0[qb_0 + u32(2)], arena_0[qb_0 + u32(3)]));
    var tt_0 : vec3<f32> = vec3<f32>(2.0f) * cross(u_0, d_1);
    var rr_0 : vec3<f32> = (d_1 + vec3<f32>(arena_0[qb_0]) * tt_0 + cross(u_0, tt_0)) / vec3<f32>((bitcast<f32>((table_0[t_0 + u32(3)]))));
    var o_3 : u32 = P_0(u32(7)) + _S24;
    var _S27 : f32 = rr_0.x;
    arena_0[o_3] = _S27;
    var _S28 : f32 = rr_0.y;
    arena_0[o_3 + u32(1)] = _S28;
    var _S29 : f32 = rr_0.z;
    arena_0[o_3 + u32(2)] = _S29;
    arena_0[o_3 + u32(3)] = sqrt(_S27 * _S27 + _S28 * _S28 + _S29 * _S29);
    return;
}

