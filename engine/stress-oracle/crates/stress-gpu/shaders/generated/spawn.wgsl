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

fn total_slots_0() -> u32
{
    return parents_0[control_0[u32(1)]].y;
}

fn parent_at_0( i_0 : u32) -> u32
{
    var _S1 : u32 = control_0[u32(1)] - u32(1);
    var lo_0 : u32 = u32(0);
    var hi_0 : u32 = _S1;
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
    var _S2 : u32 = parent_0.z;
    var _S3 : bool;
    if(_S2 >= u32(2))
    {
        _S3 = (s_0 - parent_0.y) < _S2;
    }
    else
    {
        _S3 = false;
    }
    return _S3;
}

var<workgroup> g_split_parents_0 : u32;

fn child_members_0( slot_0 : u32) -> vec2<u32>
{
    var head_0 : vec4<f32> = children_0[u32(8) * slot_0];
    return vec2<u32>((bitcast<u32>((head_0.y))), (bitcast<u32>((head_0.z))));
}

@compute
@workgroup_size(1024, 1, 1)
fn spawn_offsets(@builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var v_1 : u32;
    var fits_0 : bool;
    var tid_1 : u32 = thread_0.x;
    var body0_0 : u32 = control_0[u32(4)];
    var collider0_0 : u32 = control_0[u32(5)];
    if(control_0[u32(1)] == u32(0))
    {
        if(tid_1 == u32(0))
        {
            control_0[u32(0)] = u32(0);
            control_0[u32(8)] = body0_0;
            control_0[u32(9)] = collider0_0;
        }
        return;
    }
    var _S4 : u32 = total_slots_0();
    var block_0 : u32 = u32(0);
    var carry_0 : u32 = u32(0);
    loop
    {
        if(block_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var i_2 : u32 = block_0 + tid_1;
        var _S5 : bool = i_2 < _S4;
        if(_S5)
        {
            v_1 = member_parts_0(i_2, _S4);
        }
        else
        {
            v_1 = u32(0);
        }
        var at_0 : u32 = block_scan_0(tid_1, v_1);
        if(_S5)
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
        if(block_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var s2_0 : u32 = block_0 + tid_1;
        if(s2_0 < _S4)
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
        var _S6 : u32 = block_scan_0(tid_1, v_1);
        var t_1 : u32 = g_scan_0[i32(1023)];
        workgroupBarrier();
        var bodies_needed_1 : u32 = bodies_needed_0 + t_1;
        block_0 = block_0 + u32(1024);
        bodies_needed_0 = bodies_needed_1;
    }
    var _S7 : bool = tid_1 == u32(0);
    var c_end_0 : u32;
    if(_S7)
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
    if(_S7)
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
    if(!fits_0)
    {
        return;
    }
    block_0 = u32(0);
    v_1 = u32(0);
    loop
    {
        if(block_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var s2_1 : u32 = block_0 + tid_1;
        var _S8 : bool = s2_1 < _S4;
        var valid_0 : bool;
        if(_S8)
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
        if(_S8)
        {
            if(valid_0)
            {
                var m_0 : vec2<u32> = child_members_0(s2_1);
                var first_0 : u32 = parents_0[parent_at_0(s2_1)].y + m_0.x;
                var end_0 : u32 = first_0 + m_0.y;
                var c_first_0 : u32 = member_offsets_0[first_0];
                if(end_0 < _S4)
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
        var _S9 : u32 = p_1 + u32(1);
        var _S10 : bool;
        if(_S9 < control_0[u32(1)])
        {
            _S10 = (parents_0[_S9].y) <= slot_1;
        }
        else
        {
            _S10 = false;
        }
        if(_S10)
        {
        }
        else
        {
            break;
        }
        p_1 = _S9;
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
    var _S11 : f32 = q_1.w_0;
    var _S12 : f32 = q_1.x_0;
    var _S13 : f32 = q_1.y_0;
    var _S14 : f32 = q_1.z_0;
    var n_0 : f32 = sqrt(_S11 * _S11 + _S12 * _S12 + _S13 * _S13 + _S14 * _S14);
    var r_0 : Quat_0;
    r_0.w_0 = q_1.w_0 / n_0;
    r_0.x_0 = q_1.x_0 / n_0;
    r_0.y_0 = q_1.y_0 / n_0;
    r_0.z_0 = q_1.z_0 / n_0;
    return r_0;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S15 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S16 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S17 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_0 : array<array<f32, i32(3)>, i32(3)>;
    a_0[i32(0)] = _S15;
    a_0[i32(1)] = _S16;
    a_0[i32(2)] = _S17;
    var _S18 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S19 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S20 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_2 : array<array<f32, i32(3)>, i32(3)>;
    v_2[i32(0)] = _S18;
    v_2[i32(1)] = _S19;
    v_2[i32(2)] = _S20;
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
            var _S21 : bool;
            if(k_0 != p_2)
            {
                _S21 = k_0 != q_3;
            }
            else
            {
                _S21 = false;
            }
            if(_S21)
            {
                var kp_0 : f32 = a_0[k_0][p_2];
                var kq_0 : f32 = a_0[k_0][q_3];
                var _S22 : f32 = c_0 * a_0[k_0][p_2] - s_1 * a_0[k_0][q_3];
                a_0[k_0][p_2] = _S22;
                a_0[p_2][k_0] = _S22;
                var _S23 : f32 = s_1 * kp_0 + c_0 * kq_0;
                a_0[k_0][q_3] = _S23;
                a_0[q_3][k_0] = _S23;
            }
            k_0 = k_0 + u32(1);
        }
        var _S24 : f32 = c_0 * c_0;
        var _S25 : f32 = 2.0f * s_1 * c_0 * apq_0;
        var _S26 : f32 = s_1 * s_1;
        a_0[p_2][p_2] = _S24 * app_0 - _S25 + _S26 * aqq_0;
        a_0[q_3][q_3] = _S26 * app_0 + _S25 + _S24 * aqq_0;
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
    var _S27 : u32;
    var k_1 : u32;
    var slot_2 : u32 = id_0.x;
    var total_1 : u32 = total_slots_0();
    if(slot_2 >= total_1)
    {
        return;
    }
    var p_3 : u32 = parent_of_0(slot_2);
    var parent_1 : vec4<u32> = parents_0[p_3];
    var _S28 : u32 = parent_1.y;
    var _S29 : bool;
    if((slot_2 - _S28) >= (parent_1.z))
    {
        _S29 = true;
    }
    else
    {
        _S29 = (slots_0[slot_2].w) == u32(0);
    }
    if(_S29)
    {
        _S29 = true;
    }
    else
    {
        _S29 = control_0[u32(0)] != u32(0);
    }
    if(_S29)
    {
        return;
    }
    var _S30 : u32 = islands_of_0[p_3].x;
    var _S31 : vec2<u32> = child_members_0(slot_2);
    var _S32 : u32 = u32(8) * slot_2;
    var anchored_0 : bool = (bitcast<u32>((children_0[_S32].w))) != u32(0);
    var _S33 : vec3<f32> = vec3<f32>(0.0f);
    var j_0 : u32 = u32(0);
    var mass_0 : f32 = 0.0f;
    var moment_0 : vec3<f32> = _S33;
    loop
    {
        var _S34 : u32 = _S31.y;
        _S27 = _S34;
        if(j_0 < _S34)
        {
        }
        else
        {
            break;
        }
        var r_3 : vec4<u32> = part_range_0(_S30 + work_0[total_1 + _S28 + _S31.x + j_0]);
        var _S35 : u32 = r_3.x;
        k_1 = _S35;
        loop
        {
            if(k_1 < (_S35 + r_3.y))
            {
            }
            else
            {
                break;
            }
            var c_2 : vec4<f32> = parts_0[u32(8) * k_1];
            var _S36 : f32 = c_2.w;
            var mass_1 : f32 = mass_0 + _S36;
            var moment_1 : vec3<f32> = moment_0 + c_2.xyz * vec3<f32>(_S36);
            k_1 = k_1 + u32(1);
            mass_0 = mass_1;
            moment_0 = moment_1;
        }
        j_0 = j_0 + u32(1);
    }
    var com_0 : vec3<f32> = moment_0 / vec3<f32>(max(mass_0, 1.00000000317107685e-30f));
    j_0 = u32(0);
    var i0_0 : vec3<f32> = _S33;
    var i1_0 : vec3<f32> = _S33;
    var i2_0 : vec3<f32> = _S33;
    loop
    {
        if(j_0 < _S27)
        {
        }
        else
        {
            break;
        }
        var r_4 : vec4<u32> = part_range_0(_S30 + work_0[total_1 + _S28 + _S31.x + j_0]);
        var _S37 : u32 = r_4.x;
        k_1 = _S37;
        loop
        {
            if(k_1 < (_S37 + r_4.y))
            {
            }
            else
            {
                break;
            }
            var _S38 : u32 = u32(8) * k_1;
            var c_3 : vec4<f32> = parts_0[_S38];
            var d_0 : vec3<f32> = c_3.xyz - com_0;
            var dd_0 : f32 = dot(d_0, d_0);
            var _S39 : vec3<f32> = vec3<f32>(c_3.w);
            var i0_1 : vec3<f32> = i0_0 + (parts_0[_S38 + u32(1)].xyz + (vec3<f32>(dd_0, 0.0f, 0.0f) - vec3<f32>(d_0.x) * d_0) * _S39);
            var i1_1 : vec3<f32> = i1_0 + (parts_0[_S38 + u32(2)].xyz + (vec3<f32>(0.0f, dd_0, 0.0f) - vec3<f32>(d_0.y) * d_0) * _S39);
            var i2_1 : vec3<f32> = i2_0 + (parts_0[_S38 + u32(3)].xyz + (vec3<f32>(0.0f, 0.0f, dd_0) - vec3<f32>(d_0.z) * d_0) * _S39);
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
    var _S40 : u32 = u32(5) * p_3;
    var pose_rot_0 : Quat_0 = quat_of_0(frames_0[_S40]);
    var pose_pos_0 : vec3<f32> = frames_0[_S40 + u32(4)].xyz;
    var _S41 : bool = !anchored_0;
    var inv_mass_0 : f32;
    var pose_pos_1 : vec3<f32>;
    if(_S41)
    {
        j_0 = u32(0);
        inv_mass_0 = 0.0f;
        pose_pos_1 = _S33;
        loop
        {
            if(j_0 < _S27)
            {
            }
            else
            {
                break;
            }
            var c_4 : u32 = _S30 + work_0[total_1 + _S28 + _S31.x + j_0];
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
        _S29 = true;
    }
    else
    {
        _S29 = !(mass_0 > 0.0f);
    }
    if(_S29)
    {
        inv_mass_0 = 0.0f;
    }
    else
    {
        inv_mass_0 = 1.0f / mass_0;
    }
    var inv_inertia_0 : vec3<f32>;
    if(_S41)
    {
        var _S42 : f32;
        if((moments_1.x) > 0.0f)
        {
            _S42 = 1.0f / moments_1.x;
        }
        else
        {
            _S42 = 0.0f;
        }
        var _S43 : f32;
        if((moments_1.y) > 0.0f)
        {
            _S43 = 1.0f / moments_1.y;
        }
        else
        {
            _S43 = 0.0f;
        }
        var _S44 : f32;
        if((moments_1.z) > 0.0f)
        {
            _S44 = 1.0f / moments_1.z;
        }
        else
        {
            _S44 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S42, _S43, _S44);
    }
    else
    {
        inv_inertia_0 = _S33;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_0)
    {
        velocity_0 = _S33;
    }
    else
    {
        velocity_0 = to_rubble_0(children_0[_S32 + u32(5)].xyz);
    }
    var spin_0 : vec3<f32>;
    if(anchored_0)
    {
        spin_0 = _S33;
    }
    else
    {
        spin_0 = to_rubble_0(children_0[_S32 + u32(6)].xyz);
    }
    if(anchored_0)
    {
        k_1 = u32(0);
    }
    else
    {
        k_1 = u32(9);
    }
    var _S45 : array<vec4<f32>, i32(7)> = array<vec4<f32>, i32(7)>( vec4<f32>(to_rubble_0(pose_pos_1 + rotate_0(pose_rot_0, com_0)), inv_mass_0), quat_vec_0(orientation_0), vec4<f32>(inv_inertia_0, (bitcast<f32>((k_1)))), vec4<f32>(velocity_0, 0.0f), vec4<f32>(spin_0, 0.0f), vec4<f32>(0.0f), vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f) );
    if((params_0.live_0) != u32(0))
    {
        j_0 = at_2.x;
    }
    else
    {
        j_0 = at_2.x - control_0[u32(4)];
    }
    var _S46 : u32 = u32(7) * j_0;
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
        bodies_0[_S46 + w_1] = _S45[w_1];
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
            bodies_0[u32(7) * (params_0.max_bodies_0 + at_2.x) + w_1] = _S45[w_1];
            w_1 = w_1 + u32(1);
        }
    }
    var _S47 : u32 = u32(2) * slot_2;
    extras_0[_S47] = vec4<f32>(com_0, mass_0);
    extras_0[_S47 + u32(1)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_colliders(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var i_3 : u32 = id_1.x;
    var total_2 : u32 = total_slots_0();
    var _S48 : bool;
    if(i_3 >= total_2)
    {
        _S48 = true;
    }
    else
    {
        _S48 = control_0[u32(1)] == u32(0);
    }
    if(_S48)
    {
        _S48 = true;
    }
    else
    {
        _S48 = control_0[u32(0)] != u32(0);
    }
    if(_S48)
    {
        return;
    }
    var p_4 : u32 = parent_at_0(i_3);
    var parent_2 : vec4<u32> = parents_0[p_4];
    var _S49 : u32 = parent_2.z;
    if(_S49 < u32(2))
    {
        return;
    }
    var _S50 : u32 = parent_2.y;
    var _S51 : u32 = i_3 - _S50;
    var _S52 : u32 = _S49 - u32(1);
    var lo_1 : u32 = u32(0);
    var hi_1 : u32 = _S52;
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
        if((child_members_0(_S50 + mid_1).x) <= _S51)
        {
            lo_1 = mid_1;
        }
        else
        {
            hi_1 = mid_1 - u32(1);
        }
    }
    var slot_3 : u32 = _S50 + lo_1;
    var at_3 : vec4<u32> = slots_0[slot_3];
    if((slots_0[slot_3].w) == u32(0))
    {
        return;
    }
    var _S53 : u32 = u32(2) * slot_3;
    var _S54 : vec3<f32> = extras_0[_S53].xyz;
    var _S55 : Quat_0 = conj_0(quat_of_0(extras_0[_S53 + u32(1)]));
    var r_5 : vec4<u32> = part_range_0(islands_of_0[p_4].x + work_0[total_2 + i_3]);
    var _S56 : u32 = r_5.x;
    var k_2 : u32 = _S56;
    var col_0 : u32 = member_offsets_0[i_3];
    loop
    {
        if(k_2 < (_S56 + r_5.y))
        {
        }
        else
        {
            break;
        }
        var _S57 : u32 = u32(8) * k_2;
        var o_1 : vec4<f32> = parts_0[_S57 + u32(4)];
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S57 + u32(5)]);
        var ids_0 : vec4<f32> = parts_0[_S57 + u32(6)];
        var shape_params_0 : vec4<f32> = parts_0[_S57 + u32(7)];
        if((params_0.live_0) != u32(0))
        {
            lo_1 = col_0;
        }
        else
        {
            lo_1 = col_0 - control_0[u32(5)];
        }
        var cw_0 : u32 = u32(4) * lo_1;
        colliders_0[cw_0] = vec4<f32>((bitcast<f32>((at_3.x))), ids_0.x, ids_0.y, ids_0.z);
        colliders_0[cw_0 + u32(1)] = vec4<f32>(rotate_0(_S55, o_1.xyz - _S54), o_1.w);
        colliders_0[cw_0 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S55, rot_1)));
        colliders_0[cw_0 + u32(3)] = shape_params_0;
        var _S58 : u32 = col_0 + u32(1);
        k_2 = k_2 + u32(1);
        col_0 = _S58;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_retire(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var p_5 : u32 = group_0.x;
    var tid_2 : u32 = thread_1.x;
    var _S59 : bool;
    if(p_5 >= control_0[u32(1)])
    {
        _S59 = true;
    }
    else
    {
        _S59 = (params_0.live_0) == u32(0);
    }
    if(_S59)
    {
        _S59 = true;
    }
    else
    {
        _S59 = (parents_0[p_5].z) < u32(2);
    }
    if(_S59)
    {
        _S59 = true;
    }
    else
    {
        _S59 = control_0[i32(0)] != u32(0);
    }
    if(_S59)
    {
        return;
    }
    var e_0 : vec4<u32> = islands_of_0[p_5];
    var c_5 : u32;
    if(tid_2 == u32(0))
    {
        var _S60 : u32 = e_0.y;
        var _S61 : u32 = u32(7) * _S60;
        var record_0 : array<vec4<f32>, i32(7)>;
        c_5 = u32(0);
        loop
        {
            if(c_5 < u32(7))
            {
            }
            else
            {
                break;
            }
            record_0[c_5] = bodies_0[_S61 + c_5];
            c_5 = c_5 + u32(1);
        }
        record_0[i32(0)][i32(3)] = 0.0f;
        record_0[i32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, (bitcast<f32>((u32(0)))));
        var _S62 : vec3<f32> = vec3<f32>(0.0f);
        record_0[i32(3)].x = _S62.x;
        record_0[i32(3)].y = _S62.y;
        record_0[i32(3)].z = _S62.z;
        record_0[i32(4)].x = _S62.x;
        record_0[i32(4)].y = _S62.y;
        record_0[i32(4)].z = _S62.z;
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
            c_5 = u32(0);
            loop
            {
                if(c_5 < u32(7))
                {
                }
                else
                {
                    break;
                }
                bodies_0[u32(7) * (_S60 + region_0 * params_0.max_bodies_0) + c_5] = record_0[c_5];
                c_5 = c_5 + u32(1);
            }
            region_0 = region_0 + u32(1);
        }
    }
    c_5 = e_0.z + tid_2;
    loop
    {
        if(c_5 < (e_0.w))
        {
        }
        else
        {
            break;
        }
        var _S63 : u32 = u32(4) * c_5;
        colliders_0[_S63] = vec4<f32>(colliders_0[_S63].x, colliders_0[_S63].y, (bitcast<f32>((((bitcast<u32>((colliders_0[_S63].z))) & (u32(65535)))))), (bitcast<f32>((u32(32768)))));
        c_5 = c_5 + u32(64);
    }
    return;
}

