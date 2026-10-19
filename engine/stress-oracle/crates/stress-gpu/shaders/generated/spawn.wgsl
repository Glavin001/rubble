@binding(15) @group(0) var<storage, read_write> control_0 : array<u32>;

@binding(1) @group(0) var<storage, read> parents_0 : array<vec4<u32>>;

@binding(12) @group(0) var<storage, read> islands_of_0 : array<vec4<u32>>;

@binding(2) @group(0) var<storage, read> work_0 : array<u32>;

struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(7) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(14) @group(0) var<storage, read> structure_base_0 : array<u32>;

@binding(5) @group(0) var<storage, read> part_ranges_0 : array<vec4<u32>>;

@binding(16) @group(0) var<storage, read_write> member_offsets_0 : array<u32>;

struct SpawnParams_std140_0
{
    @align(16) parent_count_0 : u32,
    @align(4) first_body_0 : u32,
    @align(8) first_collider_0 : u32,
    @align(4) live_0 : u32,
    @align(16) max_bodies_0 : u32,
    @align(4) max_colliders_0 : u32,
    @align(8) island_cap_0 : u32,
    @align(4) pad2_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : SpawnParams_std140_0;
@binding(3) @group(0) var<storage, read> children_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read_write> slots_0 : array<vec4<u32>>;

@binding(6) @group(0) var<storage, read> parts_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> frames_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read_write> bodies_0 : array<vec4<f32>>;

@binding(13) @group(0) var<storage, read_write> extras_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read_write> colliders_0 : array<vec4<f32>>;

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_0 : u32,  thread_0 : u32) -> u32
{
    if(thread_0 == u32(0))
    {
        g_group_uniform_0 = value_0;
    }
    var _S1 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S1;
}

fn total_slots_0() -> u32
{
    return parents_0[control_0[u32(1)]].y;
}

fn parent_at_0( i_0 : u32) -> u32
{
    var _S2 : u32 = control_0[u32(1)] - u32(1);
    var lo_0 : u32 = u32(0);
    var hi_0 : u32 = _S2;
    loop
    {
        if(lo_0 < hi_0)
        {
        }
        else
        {
            break;
        }
        var mid_0 : u32 = (lo_0 + hi_0 + u32(1)) / u32(2);
        if((parents_0[mid_0].y) <= i_0)
        {
            lo_0 = mid_0;
        }
        else
        {
            hi_0 = mid_0 - u32(1);
        }
    }
    return lo_0;
}

fn part_range_0( layout_chunk_0 : u32) -> vec4<u32>
{
    return part_ranges_0[structure_base_0[chunks_0[layout_chunk_0].info_0.z] + chunks_0[layout_chunk_0].info_0.w];
}

fn member_parts_0( i_1 : u32,  total_0 : u32) -> u32
{
    var p_0 : u32 = parent_at_0(i_1);
    if((parents_0[p_0].z) < u32(2))
    {
        return u32(0);
    }
    return part_range_0(islands_of_0[p_0].x + work_0[total_0 + i_1]).y;
}

var<workgroup> g_scan_0 : array<u32, i32(1024)>;

fn block_scan_0( tid_0 : u32,  v_0 : u32) -> u32
{
    g_scan_0[tid_0] = v_0;
    workgroupBarrier();
    var step_0 : u32 = u32(1);
    loop
    {
        if(step_0 < u32(1024))
        {
        }
        else
        {
            break;
        }
        var add_0 : u32;
        if(tid_0 >= step_0)
        {
            add_0 = g_scan_0[tid_0 - step_0];
        }
        else
        {
            add_0 = u32(0);
        }
        workgroupBarrier();
        g_scan_0[tid_0] = g_scan_0[tid_0] + add_0;
        workgroupBarrier();
        step_0 = (step_0 << (u32(1)));
    }
    return g_scan_0[tid_0] - v_0;
}

fn slot_valid_0( s_0 : u32) -> bool
{
    var parent_0 : vec4<u32> = parents_0[parent_at_0(s_0)];
    var _S3 : u32 = parent_0.z;
    var _S4 : bool;
    if(_S3 >= u32(2))
    {
        _S4 = (s_0 - parent_0.y) < _S3;
    }
    else
    {
        _S4 = false;
    }
    return _S4;
}

var<workgroup> g_split_parents_0 : u32;

fn child_members_0( slot_0 : u32) -> vec2<u32>
{
    var head_0 : vec4<f32> = children_0[u32(8) * slot_0];
    return vec2<u32>((bitcast<u32>((head_0.y))), (bitcast<u32>((head_0.z))));
}

@compute
@workgroup_size(1024, 1, 1)
fn spawn_offsets(@builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var v_1 : u32;
    var fits_0 : bool;
    var tid_1 : u32 = thread_1.x;
    var body0_0 : u32 = control_0[u32(4)];
    var collider0_0 : u32 = control_0[u32(5)];
    var _S5 : u32 = group_uniform_0(control_0[u32(1)], tid_1);
    if(_S5 == u32(0))
    {
        if(tid_1 == u32(0))
        {
            control_0[u32(0)] = u32(0);
            control_0[u32(8)] = body0_0;
            control_0[u32(9)] = collider0_0;
        }
        return;
    }
    var _S6 : u32 = group_uniform_0(total_slots_0(), tid_1);
    var block_0 : u32 = u32(0);
    var carry_0 : u32 = u32(0);
    loop
    {
        if(block_0 < _S6)
        {
        }
        else
        {
            break;
        }
        var i_2 : u32 = block_0 + tid_1;
        var _S7 : bool = i_2 < _S6;
        if(_S7)
        {
            v_1 = member_parts_0(i_2, _S6);
        }
        else
        {
            v_1 = u32(0);
        }
        var at_0 : u32 = block_scan_0(tid_1, v_1);
        if(_S7)
        {
            member_offsets_0[i_2] = collider0_0 + carry_0 + at_0;
        }
        var t_0 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var carry_1 : u32 = carry_0 + t_0;
        block_0 = block_0 + u32(1024);
        carry_0 = carry_1;
    }
    block_0 = u32(0);
    var bodies_needed_0 : u32 = u32(0);
    loop
    {
        if(block_0 < _S6)
        {
        }
        else
        {
            break;
        }
        var s2_0 : u32 = block_0 + tid_1;
        if(s2_0 < _S6)
        {
            fits_0 = slot_valid_0(s2_0);
        }
        else
        {
            fits_0 = false;
        }
        if(fits_0)
        {
            v_1 = u32(1);
        }
        else
        {
            v_1 = u32(0);
        }
        var _S8 : u32 = block_scan_0(tid_1, v_1);
        var t_1 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var bodies_needed_1 : u32 = bodies_needed_0 + t_1;
        block_0 = block_0 + u32(1024);
        bodies_needed_0 = bodies_needed_1;
    }
    var _S9 : bool = tid_1 == u32(0);
    var c_end_0 : u32;
    if(_S9)
    {
        v_1 = u32(0);
        c_end_0 = u32(0);
        loop
        {
            if(v_1 < control_0[u32(1)])
            {
            }
            else
            {
                break;
            }
            if((parents_0[v_1].z) >= u32(2))
            {
                c_end_0 = c_end_0 + u32(1);
            }
            v_1 = v_1 + u32(1);
        }
        g_split_parents_0 = c_end_0;
    }
    workgroupBarrier();
    if((bodies_needed_0 - g_split_parents_0) <= (params_0.island_cap_0 - min(params_0.island_cap_0, control_0[u32(2)])))
    {
        if((params_0.live_0) == u32(0))
        {
            fits_0 = true;
        }
        else
        {
            if((body0_0 + bodies_needed_0) <= (params_0.max_bodies_0))
            {
                fits_0 = (collider0_0 + carry_0) <= (params_0.max_colliders_0);
            }
            else
            {
                fits_0 = false;
            }
        }
    }
    else
    {
        fits_0 = false;
    }
    if(_S9)
    {
        if(fits_0)
        {
            block_0 = u32(0);
        }
        else
        {
            block_0 = u32(1);
        }
        control_0[u32(0)] = block_0;
        if(fits_0)
        {
            block_0 = body0_0 + bodies_needed_0;
        }
        else
        {
            block_0 = body0_0;
        }
        control_0[u32(8)] = block_0;
        if(fits_0)
        {
            block_0 = collider0_0 + carry_0;
        }
        else
        {
            block_0 = collider0_0;
        }
        control_0[u32(9)] = block_0;
    }
    if(fits_0)
    {
        block_0 = u32(1);
    }
    else
    {
        block_0 = u32(0);
    }
    var _S10 : u32 = group_uniform_0(block_0, tid_1);
    if(_S10 == u32(0))
    {
        return;
    }
    block_0 = u32(0);
    v_1 = u32(0);
    loop
    {
        if(block_0 < _S6)
        {
        }
        else
        {
            break;
        }
        var s2_1 : u32 = block_0 + tid_1;
        var _S11 : bool = s2_1 < _S6;
        var valid_0 : bool;
        if(_S11)
        {
            valid_0 = slot_valid_0(s2_1);
        }
        else
        {
            valid_0 = false;
        }
        if(valid_0)
        {
            bodies_needed_0 = u32(1);
        }
        else
        {
            bodies_needed_0 = u32(0);
        }
        var at_1 : u32 = block_scan_0(tid_1, bodies_needed_0);
        if(_S11)
        {
            if(valid_0)
            {
                var m_0 : vec2<u32> = child_members_0(s2_1);
                var first_0 : u32 = parents_0[parent_at_0(s2_1)].y + m_0.x;
                var end_0 : u32 = first_0 + m_0.y;
                var c_first_0 : u32 = member_offsets_0[first_0];
                if(end_0 < _S6)
                {
                    c_end_0 = member_offsets_0[end_0];
                }
                else
                {
                    c_end_0 = collider0_0 + carry_0;
                }
                slots_0[s2_1] = vec4<u32>(body0_0 + v_1 + at_1, c_first_0, c_end_0 - c_first_0, u32(1));
            }
            else
            {
                slots_0[s2_1] = vec4<u32>(vec4<i32>(i32(0)));
            }
        }
        var t_2 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var bodies_1 : u32 = v_1 + t_2;
        block_0 = block_0 + u32(1024);
        v_1 = bodies_1;
    }
    return;
}

fn parent_of_0( slot_1 : u32) -> u32
{
    var p_1 : u32 = u32(0);
    loop
    {
        var _S12 : u32 = p_1 + u32(1);
        var _S13 : bool;
        if(_S12 < control_0[u32(1)])
        {
            _S13 = (parents_0[_S12].y) <= slot_1;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
        }
        else
        {
            break;
        }
        p_1 = _S12;
    }
    return p_1;
}

struct Quat_0
{
     w_0 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_from_columns_0( c0_0 : vec3<f32>,  c1_0 : vec3<f32>,  c2_0 : vec3<f32>) -> Quat_0
{
    var m00_0 : f32 = c0_0.x;
    var m01_0 : f32 = c0_0.y;
    var m02_0 : f32 = c0_0.z;
    var m10_0 : f32 = c1_0.x;
    var m11_0 : f32 = c1_0.y;
    var m12_0 : f32 = c1_0.z;
    var m20_0 : f32 = c2_0.x;
    var m21_0 : f32 = c2_0.y;
    var m22_0 : f32 = c2_0.z;
    var q_0 : Quat_0;
    if(m22_0 <= 0.0f)
    {
        var dif10_0 : f32 = m11_0 - m00_0;
        var omm22_0 : f32 = 1.0f - m22_0;
        if(dif10_0 <= 0.0f)
        {
            var f_0 : f32 = omm22_0 - dif10_0;
            var inv_0 : f32 = 0.5f / sqrt(f_0);
            q_0.x_0 = f_0 * inv_0;
            q_0.y_0 = (m01_0 + m10_0) * inv_0;
            q_0.z_0 = (m02_0 + m20_0) * inv_0;
            q_0.w_0 = (m12_0 - m21_0) * inv_0;
        }
        else
        {
            var f_1 : f32 = omm22_0 + dif10_0;
            var inv_1 : f32 = 0.5f / sqrt(f_1);
            q_0.x_0 = (m01_0 + m10_0) * inv_1;
            q_0.y_0 = f_1 * inv_1;
            q_0.z_0 = (m12_0 + m21_0) * inv_1;
            q_0.w_0 = (m20_0 - m02_0) * inv_1;
        }
    }
    else
    {
        var sum10_0 : f32 = m11_0 + m00_0;
        var opm22_0 : f32 = 1.0f + m22_0;
        if(sum10_0 <= 0.0f)
        {
            var f_2 : f32 = opm22_0 - sum10_0;
            var inv_2 : f32 = 0.5f / sqrt(f_2);
            q_0.x_0 = (m02_0 + m20_0) * inv_2;
            q_0.y_0 = (m12_0 + m21_0) * inv_2;
            q_0.z_0 = f_2 * inv_2;
            q_0.w_0 = (m01_0 - m10_0) * inv_2;
        }
        else
        {
            var f_3 : f32 = opm22_0 + sum10_0;
            var inv_3 : f32 = 0.5f / sqrt(f_3);
            q_0.x_0 = (m12_0 - m21_0) * inv_3;
            q_0.y_0 = (m20_0 - m02_0) * inv_3;
            q_0.z_0 = (m01_0 - m10_0) * inv_3;
            q_0.w_0 = f_3 * inv_3;
        }
    }
    return q_0;
}

fn normalized_0( q_1 : Quat_0) -> Quat_0
{
    var _S14 : f32 = q_1.w_0;
    var _S15 : f32 = q_1.x_0;
    var _S16 : f32 = q_1.y_0;
    var _S17 : f32 = q_1.z_0;
    var n_0 : f32 = sqrt(_S14 * _S14 + _S15 * _S15 + _S16 * _S16 + _S17 * _S17);
    var r_0 : Quat_0;
    r_0.w_0 = q_1.w_0 / n_0;
    r_0.x_0 = q_1.x_0 / n_0;
    r_0.y_0 = q_1.y_0 / n_0;
    r_0.z_0 = q_1.z_0 / n_0;
    return r_0;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S18 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S19 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S20 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_0 : array<array<f32, i32(3)>, i32(3)>;
    a_0[i32(0)] = _S18;
    a_0[i32(1)] = _S19;
    a_0[i32(2)] = _S20;
    var _S21 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S22 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S23 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_2 : array<array<f32, i32(3)>, i32(3)>;
    v_2[i32(0)] = _S21;
    v_2[i32(1)] = _S22;
    v_2[i32(2)] = _S23;
    var it_0 : u32 = u32(0);
    loop
    {
        if(it_0 < u32(32))
        {
        }
        else
        {
            break;
        }
        var q_2 : u32;
        if((abs(a_0[i32(0)][i32(2)])) > (abs(a_0[u32(0)][u32(1)])))
        {
            q_2 = u32(2);
        }
        else
        {
            q_2 = u32(1);
        }
        var p_2 : u32;
        var q_3 : u32;
        if((abs(a_0[i32(1)][i32(2)])) > (abs(a_0[u32(0)][q_2])))
        {
            p_2 = u32(1);
            q_3 = u32(2);
        }
        else
        {
            p_2 = u32(0);
            q_3 = q_2;
        }
        if((abs(a_0[p_2][q_3])) <= (9.99999982451670044e-15f * max(max(abs(a_0[i32(0)][i32(0)]), abs(a_0[i32(1)][i32(1)])), max(abs(a_0[i32(2)][i32(2)]), 1.00000000317107685e-30f))))
        {
            break;
        }
        var theta_0 : f32 = 0.5f * atan2(2.0f * a_0[p_2][q_3], a_0[q_3][q_3] - a_0[p_2][p_2]);
        var s_1 : f32 = sin(theta_0);
        var c_0 : f32 = cos(theta_0);
        var app_0 : f32 = a_0[p_2][p_2];
        var aqq_0 : f32 = a_0[q_3][q_3];
        var apq_0 : f32 = a_0[p_2][q_3];
        var k_0 : u32 = u32(0);
        loop
        {
            if(k_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S24 : bool;
            if(k_0 != p_2)
            {
                _S24 = k_0 != q_3;
            }
            else
            {
                _S24 = false;
            }
            if(_S24)
            {
                var kp_0 : f32 = a_0[k_0][p_2];
                var kq_0 : f32 = a_0[k_0][q_3];
                var _S25 : f32 = c_0 * a_0[k_0][p_2] - s_1 * a_0[k_0][q_3];
                a_0[k_0][p_2] = _S25;
                a_0[p_2][k_0] = _S25;
                var _S26 : f32 = s_1 * kp_0 + c_0 * kq_0;
                a_0[k_0][q_3] = _S26;
                a_0[q_3][k_0] = _S26;
            }
            k_0 = k_0 + u32(1);
        }
        var _S27 : f32 = c_0 * c_0;
        var _S28 : f32 = 2.0f * s_1 * c_0 * apq_0;
        var _S29 : f32 = s_1 * s_1;
        a_0[p_2][p_2] = _S27 * app_0 - _S28 + _S29 * aqq_0;
        a_0[q_3][q_3] = _S29 * app_0 + _S28 + _S27 * aqq_0;
        a_0[p_2][q_3] = 0.0f;
        a_0[q_3][p_2] = 0.0f;
        var row_0 : u32 = u32(0);
        loop
        {
            if(row_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            var kp_1 : f32 = v_2[row_0][p_2];
            var kq_1 : f32 = v_2[row_0][q_3];
            v_2[row_0][p_2] = c_0 * v_2[row_0][p_2] - s_1 * v_2[row_0][q_3];
            v_2[row_0][q_3] = s_1 * kp_1 + c_0 * kq_1;
            row_0 = row_0 + u32(1);
        }
        it_0 = it_0 + u32(1);
    }
    var x_1 : vec3<f32> = vec3<f32>(v_2[i32(0)][i32(0)], v_2[i32(1)][i32(0)], v_2[i32(2)][i32(0)]);
    var y_1 : vec3<f32> = vec3<f32>(v_2[i32(0)][i32(1)], v_2[i32(1)][i32(1)], v_2[i32(2)][i32(1)]);
    var z_1 : vec3<f32> = vec3<f32>(v_2[i32(0)][i32(2)], v_2[i32(1)][i32(2)], v_2[i32(2)][i32(2)]);
    var z_2 : vec3<f32>;
    if((dot(cross(x_1, y_1), z_1)) < 0.0f)
    {
        z_2 = (vec3<f32>(0) - z_1);
    }
    else
    {
        z_2 = z_1;
    }
    (*rot_0) = normalized_0(quat_from_columns_0(x_1, y_1, z_2));
    (*moments_0) = vec3<f32>(a_0[i32(0)][i32(0)], a_0[i32(1)][i32(1)], a_0[i32(2)][i32(2)]);
    return;
}

fn conj_0( q_4 : Quat_0) -> Quat_0
{
    var c_1 : Quat_0;
    c_1.w_0 = q_4.w_0;
    c_1.x_0 = - q_4.x_0;
    c_1.y_0 = - q_4.y_0;
    c_1.z_0 = - q_4.z_0;
    return c_1;
}

fn quat_of_0( q_5 : vec4<f32>) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.x_0 = q_5.x;
    r_1.y_0 = q_5.y;
    r_1.z_0 = q_5.z;
    r_1.w_0 = q_5.w;
    return r_1;
}

fn rotate_0( q_6 : Quat_0,  v_3 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_6.x_0, q_6.y_0, q_6.z_0);
    var t_3 : vec3<f32> = cross(qv_0, v_3) * vec3<f32>(2.0f);
    return v_3 + t_3 * vec3<f32>(q_6.w_0) + cross(qv_0, t_3);
}

fn quat_mul_0( a_1 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_2 : Quat_0;
    r_2.w_0 = a_1.w_0 * o_0.w_0 - a_1.x_0 * o_0.x_0 - a_1.y_0 * o_0.y_0 - a_1.z_0 * o_0.z_0;
    r_2.x_0 = a_1.w_0 * o_0.x_0 + a_1.x_0 * o_0.w_0 + a_1.y_0 * o_0.z_0 - a_1.z_0 * o_0.y_0;
    r_2.y_0 = a_1.w_0 * o_0.y_0 - a_1.x_0 * o_0.z_0 + a_1.y_0 * o_0.w_0 + a_1.z_0 * o_0.x_0;
    r_2.z_0 = a_1.w_0 * o_0.z_0 + a_1.x_0 * o_0.y_0 - a_1.y_0 * o_0.x_0 + a_1.z_0 * o_0.w_0;
    return r_2;
}

fn rot_to_rubble_0( q_7 : Quat_0) -> Quat_0
{
    var axes_0 : Quat_0;
    axes_0.w_0 = 0.70710676908493042f;
    axes_0.x_0 = -0.70710676908493042f;
    axes_0.y_0 = 0.0f;
    axes_0.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_0, q_7));
}

fn to_rubble_0( v_4 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_4.x, v_4.z, - v_4.y);
}

fn quat_vec_0( q_8 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_8.x_0, q_8.y_0, q_8.z_0, q_8.w_0);
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S30 : u32;
    var k_1 : u32;
    var slot_2 : u32 = id_0.x;
    var total_1 : u32 = total_slots_0();
    if(slot_2 >= total_1)
    {
        return;
    }
    var p_3 : u32 = parent_of_0(slot_2);
    var parent_1 : vec4<u32> = parents_0[p_3];
    var _S31 : u32 = parent_1.y;
    var _S32 : bool;
    if((slot_2 - _S31) >= (parent_1.z))
    {
        _S32 = true;
    }
    else
    {
        _S32 = (slots_0[slot_2].w) == u32(0);
    }
    if(_S32)
    {
        _S32 = true;
    }
    else
    {
        _S32 = control_0[u32(0)] != u32(0);
    }
    if(_S32)
    {
        return;
    }
    var _S33 : u32 = islands_of_0[p_3].x;
    var _S34 : vec2<u32> = child_members_0(slot_2);
    var _S35 : u32 = u32(8) * slot_2;
    var anchored_0 : bool = (bitcast<u32>((children_0[_S35].w))) != u32(0);
    var _S36 : vec3<f32> = vec3<f32>(0.0f);
    var j_0 : u32 = u32(0);
    var mass_0 : f32 = 0.0f;
    var moment_0 : vec3<f32> = _S36;
    loop
    {
        var _S37 : u32 = _S34.y;
        _S30 = _S37;
        if(j_0 < _S37)
        {
        }
        else
        {
            break;
        }
        var r_3 : vec4<u32> = part_range_0(_S33 + work_0[total_1 + _S31 + _S34.x + j_0]);
        var _S38 : u32 = r_3.x;
        k_1 = _S38;
        loop
        {
            if(k_1 < (_S38 + r_3.y))
            {
            }
            else
            {
                break;
            }
            var c_2 : vec4<f32> = parts_0[u32(8) * k_1];
            var _S39 : f32 = c_2.w;
            var mass_1 : f32 = mass_0 + _S39;
            var moment_1 : vec3<f32> = moment_0 + c_2.xyz * vec3<f32>(_S39);
            k_1 = k_1 + u32(1);
            mass_0 = mass_1;
            moment_0 = moment_1;
        }
        j_0 = j_0 + u32(1);
    }
    var com_0 : vec3<f32> = moment_0 / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    j_0 = u32(0);
    var i0_0 : vec3<f32> = _S36;
    var i1_0 : vec3<f32> = _S36;
    var i2_0 : vec3<f32> = _S36;
    loop
    {
        if(j_0 < _S30)
        {
        }
        else
        {
            break;
        }
        var r_4 : vec4<u32> = part_range_0(_S33 + work_0[total_1 + _S31 + _S34.x + j_0]);
        var _S40 : u32 = r_4.x;
        k_1 = _S40;
        loop
        {
            if(k_1 < (_S40 + r_4.y))
            {
            }
            else
            {
                break;
            }
            var _S41 : u32 = u32(8) * k_1;
            var c_3 : vec4<f32> = parts_0[_S41];
            var d_0 : vec3<f32> = c_3.xyz - com_0;
            var dd_0 : f32 = dot(d_0, d_0);
            var _S42 : vec3<f32> = vec3<f32>(c_3.w);
            var i0_1 : vec3<f32> = i0_0 + (parts_0[_S41 + u32(1)].xyz + (vec3<f32>(dd_0, 0.0f, 0.0f) - vec3<f32>(d_0.x) * d_0) * _S42);
            var i1_1 : vec3<f32> = i1_0 + (parts_0[_S41 + u32(2)].xyz + (vec3<f32>(0.0f, dd_0, 0.0f) - vec3<f32>(d_0.y) * d_0) * _S42);
            var i2_1 : vec3<f32> = i2_0 + (parts_0[_S41 + u32(3)].xyz + (vec3<f32>(0.0f, 0.0f, dd_0) - vec3<f32>(d_0.z) * d_0) * _S42);
            k_1 = k_1 + u32(1);
            i0_0 = i0_1;
            i1_0 = i1_1;
            i2_0 = i2_1;
        }
        j_0 = j_0 + u32(1);
    }
    var principal_0 : Quat_0;
    var moments_1 : vec3<f32>;
    principal_axes_0(i0_0, i1_0, i2_0, &(principal_0), &(moments_1));
    var _S43 : u32 = u32(5) * p_3;
    var pose_rot_0 : Quat_0 = quat_of_0(frames_0[_S43]);
    var pose_pos_0 : vec3<f32> = frames_0[_S43 + u32(4)].xyz;
    var _S44 : bool = !anchored_0;
    var inv_mass_0 : f32;
    var pose_pos_1 : vec3<f32>;
    if(_S44)
    {
        j_0 = u32(0);
        inv_mass_0 = 0.0f;
        pose_pos_1 = _S36;
        loop
        {
            if(j_0 < _S30)
            {
            }
            else
            {
                break;
            }
            var c_4 : u32 = _S33 + work_0[total_1 + _S31 + _S34.x + j_0];
            var cm_0 : f32 = chunks_0[c_4].center_0.w;
            var sm_0 : f32 = inv_mass_0 + cm_0;
            var su_0 : vec3<f32> = pose_pos_1 + state_0[u32(4) * c_4].xyz * vec3<f32>(cm_0);
            j_0 = j_0 + u32(1);
            inv_mass_0 = sm_0;
            pose_pos_1 = su_0;
        }
        pose_pos_1 = pose_pos_0 + rotate_0(pose_rot_0, pose_pos_1 / vec3<f32>(inv_mass_0));
    }
    else
    {
        pose_pos_1 = pose_pos_0;
    }
    var at_2 : vec4<u32> = slots_0[slot_2];
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(pose_rot_0, principal_0)));
    if(anchored_0)
    {
        _S32 = true;
    }
    else
    {
        _S32 = !(mass_0 > 0.0f);
    }
    if(_S32)
    {
        inv_mass_0 = 0.0f;
    }
    else
    {
        inv_mass_0 = 1.0f / mass_0;
    }
    var inv_inertia_0 : vec3<f32>;
    if(_S44)
    {
        var _S45 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S45 = 1.0f / moments_1.x;
        }
        else
        {
            _S45 = 0.0f;
        }
        var _S46 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S46 = 1.0f / moments_1.y;
        }
        else
        {
            _S46 = 0.0f;
        }
        var _S47 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S47 = 1.0f / moments_1.z;
        }
        else
        {
            _S47 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S45, _S46, _S47);
    }
    else
    {
        inv_inertia_0 = _S36;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S36;
    }
    else
    {
        velocity_0 = to_rubble_0(children_0[_S35 + u32(5)].xyz);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S36;
    }
    else
    {
        spin_0 = to_rubble_0(children_0[_S35 + u32(6)].xyz);
    }
    if(anchored_0)
    {
        k_1 = u32(0);
    }
    else
    {
        k_1 = u32(9);
    }
    var _S48 : array<vec4<f32>, i32(7)> = array<vec4<f32>, i32(7)>( vec4<f32>(to_rubble_0(pose_pos_1 + rotate_0(pose_rot_0, com_0)), inv_mass_0), quat_vec_0(orientation_0), vec4<f32>(inv_inertia_0, (bitcast<f32>((k_1)))), vec4<f32>(velocity_0, 0.0f), vec4<f32>(spin_0, 0.0f), vec4<f32>(0.0f), vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f) );
    if((params_0.live_0) != u32(0))
    {
        j_0 = at_2.x;
    }
    else
    {
        j_0 = at_2.x - control_0[u32(4)];
    }
    var _S49 : u32 = u32(7) * j_0;
    var w_1 : u32 = u32(0);
    loop
    {
        if(w_1 < u32(7))
        {
        }
        else
        {
            break;
        }
        bodies_0[_S49 + w_1] = _S48[w_1];
        w_1 = w_1 + u32(1);
    }
    if((params_0.live_0) != u32(0))
    {
        w_1 = u32(0);
        loop
        {
            if(w_1 < u32(7))
            {
            }
            else
            {
                break;
            }
            bodies_0[u32(7) * (params_0.max_bodies_0 + at_2.x) + w_1] = _S48[w_1];
            w_1 = w_1 + u32(1);
        }
    }
    var _S50 : u32 = u32(2) * slot_2;
    extras_0[_S50] = vec4<f32>(com_0, mass_0);
    extras_0[_S50 + u32(1)] = quat_vec_0(principal_0);
    return;
}

var<workgroup> g_s0_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_s1_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_s2_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_s3_0 : array<vec4<f32>, i32(64)>;

var<workgroup> g_s4_0 : array<vec4<f32>, i32(64)>;

fn spawn_tree_0( tid_2 : u32)
{
    workgroupBarrier();
    var s_2 : u32 = u32(32);
    loop
    {
        if(s_2 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_2 < s_2)
        {
            var _S51 : u32 = tid_2 + s_2;
            g_s0_0[tid_2] = g_s0_0[tid_2] + g_s0_0[_S51];
            g_s1_0[tid_2] = g_s1_0[tid_2] + g_s1_0[_S51];
            g_s2_0[tid_2] = g_s2_0[tid_2] + g_s2_0[_S51];
            g_s3_0[tid_2] = g_s3_0[tid_2] + g_s3_0[_S51];
            g_s4_0[tid_2] = g_s4_0[tid_2] + g_s4_0[_S51];
        }
        workgroupBarrier();
        s_2 = (s_2 >> (u32(1)));
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_group(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var _S52 : u32;
    var k_2 : u32;
    var slot_3 : u32 = group_0.x;
    var tid_3 : u32 = thread_2.x;
    var total_2 : u32 = total_slots_0();
    var _S53 : bool;
    if(slot_3 >= total_2)
    {
        _S53 = true;
    }
    else
    {
        _S53 = control_0[u32(1)] == u32(0);
    }
    if(_S53)
    {
        _S53 = true;
    }
    else
    {
        _S53 = control_0[u32(0)] != u32(0);
    }
    var j_1 : u32;
    if(_S53)
    {
        j_1 = u32(1);
    }
    else
    {
        j_1 = u32(0);
    }
    var _S54 : u32 = group_uniform_0(j_1, tid_3);
    if(_S54 != u32(0))
    {
        return;
    }
    var p_4 : u32 = parent_at_0(slot_3);
    var parent_2 : vec4<u32> = parents_0[p_4];
    var _S55 : u32 = parent_2.y;
    if((slot_3 - _S55) >= (parent_2.z))
    {
        _S53 = true;
    }
    else
    {
        _S53 = (slots_0[slot_3].w) == u32(0);
    }
    if(_S53)
    {
        j_1 = u32(1);
    }
    else
    {
        j_1 = u32(0);
    }
    var _S56 : u32 = group_uniform_0(j_1, tid_3);
    if(_S56 != u32(0))
    {
        return;
    }
    var _S57 : u32 = islands_of_0[p_4].x;
    var _S58 : vec2<u32> = child_members_0(slot_3);
    var _S59 : u32 = u32(8) * slot_3;
    var anchored_1 : bool = (bitcast<u32>((children_0[_S59].w))) != u32(0);
    var _S60 : vec4<f32> = vec4<f32>(0.0f);
    j_1 = tid_3;
    var mm_0 : vec4<f32> = _S60;
    var su_1 : vec4<f32> = _S60;
    loop
    {
        var _S61 : u32 = _S58.y;
        _S52 = _S61;
        if(j_1 < _S61)
        {
        }
        else
        {
            break;
        }
        var c_5 : u32 = _S57 + work_0[total_2 + _S55 + _S58.x + j_1];
        var r_5 : vec4<u32> = part_range_0(c_5);
        var _S62 : u32 = r_5.x;
        k_2 = _S62;
        loop
        {
            if(k_2 < (_S62 + r_5.y))
            {
            }
            else
            {
                break;
            }
            var pc_0 : vec4<f32> = parts_0[u32(8) * k_2];
            var _S63 : f32 = pc_0.w;
            var mm_1 : vec4<f32> = mm_0 + vec4<f32>(pc_0.xyz * vec3<f32>(_S63), _S63);
            k_2 = k_2 + u32(1);
            mm_0 = mm_1;
        }
        var cm_1 : f32 = chunks_0[c_5].center_0.w;
        var su_2 : vec4<f32> = su_1 + vec4<f32>(state_0[u32(4) * c_5].xyz * vec3<f32>(cm_1), cm_1);
        j_1 = j_1 + u32(64);
        su_1 = su_2;
    }
    g_s0_0[tid_3] = mm_0;
    g_s1_0[tid_3] = _S60;
    g_s2_0[tid_3] = _S60;
    g_s3_0[tid_3] = _S60;
    g_s4_0[tid_3] = su_1;
    spawn_tree_0(tid_3);
    var mass_2 : f32 = g_s0_0[i32(0)].w;
    var com_1 : vec3<f32> = g_s0_0[i32(0)].xyz / vec3<f32>(max(mass_2, 1.00000000317107685e-30f));
    var su_total_0 : vec4<f32> = g_s4_0[i32(0)];
    workgroupBarrier();
    var _S64 : vec3<f32> = vec3<f32>(0.0f);
    j_1 = tid_3;
    var i0_2 : vec3<f32> = _S64;
    var i1_2 : vec3<f32> = _S64;
    var i2_2 : vec3<f32> = _S64;
    loop
    {
        if(j_1 < _S52)
        {
        }
        else
        {
            break;
        }
        var r_6 : vec4<u32> = part_range_0(_S57 + work_0[total_2 + _S55 + _S58.x + j_1]);
        var _S65 : u32 = r_6.x;
        k_2 = _S65;
        loop
        {
            if(k_2 < (_S65 + r_6.y))
            {
            }
            else
            {
                break;
            }
            var _S66 : u32 = u32(8) * k_2;
            var c_6 : vec4<f32> = parts_0[_S66];
            var d_1 : vec3<f32> = c_6.xyz - com_1;
            var dd_1 : f32 = dot(d_1, d_1);
            var _S67 : vec3<f32> = vec3<f32>(c_6.w);
            var i0_3 : vec3<f32> = i0_2 + (parts_0[_S66 + u32(1)].xyz + (vec3<f32>(dd_1, 0.0f, 0.0f) - vec3<f32>(d_1.x) * d_1) * _S67);
            var i1_3 : vec3<f32> = i1_2 + (parts_0[_S66 + u32(2)].xyz + (vec3<f32>(0.0f, dd_1, 0.0f) - vec3<f32>(d_1.y) * d_1) * _S67);
            var i2_3 : vec3<f32> = i2_2 + (parts_0[_S66 + u32(3)].xyz + (vec3<f32>(0.0f, 0.0f, dd_1) - vec3<f32>(d_1.z) * d_1) * _S67);
            k_2 = k_2 + u32(1);
            i0_2 = i0_3;
            i1_2 = i1_3;
            i2_2 = i2_3;
        }
        j_1 = j_1 + u32(64);
    }
    g_s0_0[tid_3] = _S60;
    g_s1_0[tid_3] = vec4<f32>(i0_2, 0.0f);
    g_s2_0[tid_3] = vec4<f32>(i1_2, 0.0f);
    g_s3_0[tid_3] = vec4<f32>(i2_2, 0.0f);
    if(tid_3 == u32(0))
    {
        mm_0 = su_total_0;
    }
    else
    {
        mm_0 = _S60;
    }
    g_s4_0[tid_3] = mm_0;
    spawn_tree_0(tid_3);
    if(tid_3 != u32(0))
    {
        return;
    }
    var principal_1 : Quat_0;
    var moments_2 : vec3<f32>;
    principal_axes_0(g_s1_0[i32(0)].xyz, g_s2_0[i32(0)].xyz, g_s3_0[i32(0)].xyz, &(principal_1), &(moments_2));
    var _S68 : u32 = u32(5) * p_4;
    var pose_rot_1 : Quat_0 = quat_of_0(frames_0[_S68]);
    var pose_pos_2 : vec3<f32> = frames_0[_S68 + u32(4)].xyz;
    var _S69 : bool = !anchored_1;
    var pose_pos_3 : vec3<f32>;
    if(_S69)
    {
        pose_pos_3 = pose_pos_2 + rotate_0(pose_rot_1, g_s4_0[i32(0)].xyz / vec3<f32>(g_s4_0[i32(0)].w));
    }
    else
    {
        pose_pos_3 = pose_pos_2;
    }
    var at_3 : vec4<u32> = slots_0[slot_3];
    var orientation_1 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(pose_rot_1, principal_1)));
    if(anchored_1)
    {
        _S53 = true;
    }
    else
    {
        _S53 = !(mass_2 > 0.0f);
    }
    var inv_mass_1 : f32;
    if(_S53)
    {
        inv_mass_1 = 0.0f;
    }
    else
    {
        inv_mass_1 = 1.0f / mass_2;
    }
    var inv_inertia_1 : vec3<f32>;
    if(_S69)
    {
        var _S70 : f32;
        if((moments_2.x) > 0.0f)
        {
            _S70 = 1.0f / moments_2.x;
        }
        else
        {
            _S70 = 0.0f;
        }
        var _S71 : f32;
        if((moments_2.y) > 0.0f)
        {
            _S71 = 1.0f / moments_2.y;
        }
        else
        {
            _S71 = 0.0f;
        }
        var _S72 : f32;
        if((moments_2.z) > 0.0f)
        {
            _S72 = 1.0f / moments_2.z;
        }
        else
        {
            _S72 = 0.0f;
        }
        inv_inertia_1 = vec3<f32>(_S70, _S71, _S72);
    }
    else
    {
        inv_inertia_1 = _S64;
    }
    var velocity_1 : vec3<f32>;
    if(anchored_1)
    {
        velocity_1 = _S64;
    }
    else
    {
        velocity_1 = to_rubble_0(children_0[_S59 + u32(5)].xyz);
    }
    var spin_1 : vec3<f32>;
    if(anchored_1)
    {
        spin_1 = _S64;
    }
    else
    {
        spin_1 = to_rubble_0(children_0[_S59 + u32(6)].xyz);
    }
    if(anchored_1)
    {
        k_2 = u32(0);
    }
    else
    {
        k_2 = u32(9);
    }
    var _S73 : array<vec4<f32>, i32(7)> = array<vec4<f32>, i32(7)>( vec4<f32>(to_rubble_0(pose_pos_3 + rotate_0(pose_rot_1, com_1)), inv_mass_1), quat_vec_0(orientation_1), vec4<f32>(inv_inertia_1, (bitcast<f32>((k_2)))), vec4<f32>(velocity_1, 0.0f), vec4<f32>(spin_1, 0.0f), _S60, vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f) );
    if((params_0.live_0) != u32(0))
    {
        j_1 = at_3.x;
    }
    else
    {
        j_1 = at_3.x - control_0[u32(4)];
    }
    var _S74 : u32 = u32(7) * j_1;
    var w_2 : u32 = u32(0);
    loop
    {
        if(w_2 < u32(7))
        {
        }
        else
        {
            break;
        }
        bodies_0[_S74 + w_2] = _S73[w_2];
        w_2 = w_2 + u32(1);
    }
    if((params_0.live_0) != u32(0))
    {
        w_2 = u32(0);
        loop
        {
            if(w_2 < u32(7))
            {
            }
            else
            {
                break;
            }
            bodies_0[u32(7) * (params_0.max_bodies_0 + at_3.x) + w_2] = _S73[w_2];
            w_2 = w_2 + u32(1);
        }
    }
    var _S75 : u32 = u32(2) * slot_3;
    extras_0[_S75] = vec4<f32>(com_1, mass_2);
    extras_0[_S75 + u32(1)] = quat_vec_0(principal_1);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_colliders(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_3 : u32 = id_1.x;
    var total_3 : u32 = total_slots_0();
    var _S76 : bool;
    if(i_3 >= total_3)
    {
        _S76 = true;
    }
    else
    {
        _S76 = control_0[u32(1)] == u32(0);
    }
    if(_S76)
    {
        _S76 = true;
    }
    else
    {
        _S76 = control_0[u32(0)] != u32(0);
    }
    if(_S76)
    {
        return;
    }
    var p_5 : u32 = parent_at_0(i_3);
    var parent_3 : vec4<u32> = parents_0[p_5];
    var _S77 : u32 = parent_3.z;
    if(_S77 < u32(2))
    {
        return;
    }
    var _S78 : u32 = parent_3.y;
    var _S79 : u32 = i_3 - _S78;
    var _S80 : u32 = _S77 - u32(1);
    var lo_1 : u32 = u32(0);
    var hi_1 : u32 = _S80;
    loop
    {
        if(lo_1 < hi_1)
        {
        }
        else
        {
            break;
        }
        var mid_1 : u32 = (lo_1 + hi_1 + u32(1)) / u32(2);
        if((child_members_0(_S78 + mid_1).x) <= _S79)
        {
            lo_1 = mid_1;
        }
        else
        {
            hi_1 = mid_1 - u32(1);
        }
    }
    var slot_4 : u32 = _S78 + lo_1;
    var at_4 : vec4<u32> = slots_0[slot_4];
    if((slots_0[slot_4].w) == u32(0))
    {
        return;
    }
    var _S81 : u32 = u32(2) * slot_4;
    var _S82 : vec3<f32> = extras_0[_S81].xyz;
    var _S83 : Quat_0 = conj_0(quat_of_0(extras_0[_S81 + u32(1)]));
    var r_7 : vec4<u32> = part_range_0(islands_of_0[p_5].x + work_0[total_3 + i_3]);
    var _S84 : u32 = r_7.x;
    var k_3 : u32 = _S84;
    var col_0 : u32 = member_offsets_0[i_3];
    loop
    {
        if(k_3 < (_S84 + r_7.y))
        {
        }
        else
        {
            break;
        }
        var _S85 : u32 = u32(8) * k_3;
        var o_1 : vec4<f32> = parts_0[_S85 + u32(4)];
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S85 + u32(5)]);
        var ids_0 : vec4<f32> = parts_0[_S85 + u32(6)];
        var shape_params_0 : vec4<f32> = parts_0[_S85 + u32(7)];
        if((params_0.live_0) != u32(0))
        {
            lo_1 = col_0;
        }
        else
        {
            lo_1 = col_0 - control_0[u32(5)];
        }
        var cw_0 : u32 = u32(4) * lo_1;
        colliders_0[cw_0] = vec4<f32>((bitcast<f32>((at_4.x))), ids_0.x, ids_0.y, ids_0.z);
        colliders_0[cw_0 + u32(1)] = vec4<f32>(rotate_0(_S83, o_1.xyz - _S82), o_1.w);
        colliders_0[cw_0 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S83, rot_1)));
        colliders_0[cw_0 + u32(3)] = shape_params_0;
        var _S86 : u32 = col_0 + u32(1);
        k_3 = k_3 + u32(1);
        col_0 = _S86;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_retire(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var p_6 : u32 = group_1.x;
    var tid_4 : u32 = thread_3.x;
    var _S87 : bool;
    if(p_6 >= control_0[u32(1)])
    {
        _S87 = true;
    }
    else
    {
        _S87 = (params_0.live_0) == u32(0);
    }
    if(_S87)
    {
        _S87 = true;
    }
    else
    {
        _S87 = (parents_0[p_6].z) < u32(2);
    }
    if(_S87)
    {
        _S87 = true;
    }
    else
    {
        _S87 = control_0[i32(0)] != u32(0);
    }
    if(_S87)
    {
        return;
    }
    var e_0 : vec4<u32> = islands_of_0[p_6];
    var c_7 : u32;
    if(tid_4 == u32(0))
    {
        var _S88 : u32 = e_0.y;
        var _S89 : u32 = u32(7) * _S88;
        var record_0 : array<vec4<f32>, i32(7)>;
        c_7 = u32(0);
        loop
        {
            if(c_7 < u32(7))
            {
            }
            else
            {
                break;
            }
            record_0[c_7] = bodies_0[_S89 + c_7];
            c_7 = c_7 + u32(1);
        }
        record_0[i32(0)][i32(3)] = 0.0f;
        record_0[i32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, (bitcast<f32>((u32(0)))));
        var _S90 : vec3<f32> = vec3<f32>(0.0f);
        record_0[i32(3)].x = _S90.x;
        record_0[i32(3)].y = _S90.y;
        record_0[i32(3)].z = _S90.z;
        record_0[i32(4)].x = _S90.x;
        record_0[i32(4)].y = _S90.y;
        record_0[i32(4)].z = _S90.z;
        var region_0 : u32 = u32(0);
        loop
        {
            if(region_0 < u32(2))
            {
            }
            else
            {
                break;
            }
            c_7 = u32(0);
            loop
            {
                if(c_7 < u32(7))
                {
                }
                else
                {
                    break;
                }
                bodies_0[u32(7) * (_S88 + region_0 * params_0.max_bodies_0) + c_7] = record_0[c_7];
                c_7 = c_7 + u32(1);
            }
            region_0 = region_0 + u32(1);
        }
    }
    c_7 = e_0.z + tid_4;
    loop
    {
        if(c_7 < (e_0.w))
        {
        }
        else
        {
            break;
        }
        var _S91 : u32 = u32(4) * c_7;
        colliders_0[_S91] = vec4<f32>(colliders_0[_S91].x, colliders_0[_S91].y, (bitcast<f32>((((bitcast<u32>((colliders_0[_S91].z))) & (u32(65535)))))), (bitcast<f32>((u32(32768)))));
        c_7 = c_7 + u32(64);
    }
    return;
}

