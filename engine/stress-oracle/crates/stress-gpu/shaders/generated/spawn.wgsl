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
    @align(4) moments_base_0 : u32,
    @align(16) info_base_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) pad1_0 : u32,
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

fn child_mass_0( first_1 : u32,  m_1 : vec2<u32>,  members_0 : u32,  mass_0 : ptr<function, f32>,  com_0 : ptr<function, vec3<f32>>,  i0_0 : ptr<function, vec3<f32>>,  i1_0 : ptr<function, vec3<f32>>,  i2_0 : ptr<function, vec3<f32>>)
{
    var _S14 : u32;
    var k_0 : u32;
    (*mass_0) = 0.0f;
    var _S15 : vec3<f32> = vec3<f32>(0.0f);
    var j_0 : u32 = u32(0);
    var moment_0 : vec3<f32> = _S15;
    loop
    {
        var _S16 : u32 = m_1.y;
        _S14 = _S16;
        if(j_0 < _S16)
        {
        }
        else
        {
            break;
        }
        var r_0 : vec4<u32> = part_range_0(first_1 + work_0[members_0 + j_0]);
        var _S17 : u32 = r_0.x;
        k_0 = _S17;
        loop
        {
            if(k_0 < (_S17 + r_0.y))
            {
            }
            else
            {
                break;
            }
            var c_0 : vec4<f32> = parts_0[u32(8) * k_0];
            var _S18 : f32 = c_0.w;
            (*mass_0) = (*mass_0) + _S18;
            var moment_1 : vec3<f32> = moment_0 + c_0.xyz * vec3<f32>(_S18);
            k_0 = k_0 + u32(1);
            moment_0 = moment_1;
        }
        j_0 = j_0 + u32(1);
    }
    (*com_0) = moment_0 / vec3<f32>(max((*mass_0), 1.00000000317107685e-30f));
    (*i0_0) = _S15;
    (*i1_0) = _S15;
    (*i2_0) = _S15;
    j_0 = u32(0);
    loop
    {
        if(j_0 < _S14)
        {
        }
        else
        {
            break;
        }
        var r_1 : vec4<u32> = part_range_0(first_1 + work_0[members_0 + j_0]);
        var _S19 : u32 = r_1.x;
        k_0 = _S19;
        loop
        {
            if(k_0 < (_S19 + r_1.y))
            {
            }
            else
            {
                break;
            }
            var _S20 : u32 = u32(8) * k_0;
            var c_1 : vec4<f32> = parts_0[_S20];
            var d_0 : vec3<f32> = c_1.xyz - (*com_0);
            var dd_0 : f32 = dot(d_0, d_0);
            var _S21 : vec3<f32> = vec3<f32>(c_1.w);
            (*i0_0) = (*i0_0) + (parts_0[_S20 + u32(1)].xyz + (vec3<f32>(dd_0, 0.0f, 0.0f) - vec3<f32>(d_0.x) * d_0) * _S21);
            (*i1_0) = (*i1_0) + (parts_0[_S20 + u32(2)].xyz + (vec3<f32>(0.0f, dd_0, 0.0f) - vec3<f32>(d_0.y) * d_0) * _S21);
            (*i2_0) = (*i2_0) + (parts_0[_S20 + u32(3)].xyz + (vec3<f32>(0.0f, 0.0f, dd_0) - vec3<f32>(d_0.z) * d_0) * _S21);
            k_0 = k_0 + u32(1);
        }
        j_0 = j_0 + u32(1);
    }
    return;
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
    var _S22 : f32 = q_1.w_0;
    var _S23 : f32 = q_1.x_0;
    var _S24 : f32 = q_1.y_0;
    var _S25 : f32 = q_1.z_0;
    var n_0 : f32 = sqrt(_S22 * _S22 + _S23 * _S23 + _S24 * _S24 + _S25 * _S25);
    var r_2 : Quat_0;
    r_2.w_0 = q_1.w_0 / n_0;
    r_2.x_0 = q_1.x_0 / n_0;
    r_2.y_0 = q_1.y_0 / n_0;
    r_2.z_0 = q_1.z_0 / n_0;
    return r_2;
}

fn principal_axes_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  rot_0 : ptr<function, Quat_0>,  moments_0 : ptr<function, vec3<f32>>)
{
    var _S26 : array<f32, i32(3)> = array<f32, i32(3)>( r0_0.x, r0_0.y, r0_0.z );
    var _S27 : array<f32, i32(3)> = array<f32, i32(3)>( r1_0.x, r1_0.y, r1_0.z );
    var _S28 : array<f32, i32(3)> = array<f32, i32(3)>( r2_0.x, r2_0.y, r2_0.z );
    var a_0 : array<array<f32, i32(3)>, i32(3)>;
    a_0[i32(0)] = _S26;
    a_0[i32(1)] = _S27;
    a_0[i32(2)] = _S28;
    var _S29 : array<f32, i32(3)> = array<f32, i32(3)>( 1.0f, 0.0f, 0.0f );
    var _S30 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 1.0f, 0.0f );
    var _S31 : array<f32, i32(3)> = array<f32, i32(3)>( 0.0f, 0.0f, 1.0f );
    var v_2 : array<array<f32, i32(3)>, i32(3)>;
    v_2[i32(0)] = _S29;
    v_2[i32(1)] = _S30;
    v_2[i32(2)] = _S31;
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
        var c_2 : f32 = cos(theta_0);
        var app_0 : f32 = a_0[p_2][p_2];
        var aqq_0 : f32 = a_0[q_3][q_3];
        var apq_0 : f32 = a_0[p_2][q_3];
        var k_1 : u32 = u32(0);
        loop
        {
            if(k_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S32 : bool;
            if(k_1 != p_2)
            {
                _S32 = k_1 != q_3;
            }
            else
            {
                _S32 = false;
            }
            if(_S32)
            {
                var kp_0 : f32 = a_0[k_1][p_2];
                var kq_0 : f32 = a_0[k_1][q_3];
                var _S33 : f32 = c_2 * a_0[k_1][p_2] - s_1 * a_0[k_1][q_3];
                a_0[k_1][p_2] = _S33;
                a_0[p_2][k_1] = _S33;
                var _S34 : f32 = s_1 * kp_0 + c_2 * kq_0;
                a_0[k_1][q_3] = _S34;
                a_0[q_3][k_1] = _S34;
            }
            k_1 = k_1 + u32(1);
        }
        var _S35 : f32 = c_2 * c_2;
        var _S36 : f32 = 2.0f * s_1 * c_2 * apq_0;
        var _S37 : f32 = s_1 * s_1;
        a_0[p_2][p_2] = _S35 * app_0 - _S36 + _S37 * aqq_0;
        a_0[q_3][q_3] = _S37 * app_0 + _S36 + _S35 * aqq_0;
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
            v_2[row_0][p_2] = c_2 * v_2[row_0][p_2] - s_1 * v_2[row_0][q_3];
            v_2[row_0][q_3] = s_1 * kp_1 + c_2 * kq_1;
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

fn quat_of_0( q_4 : vec4<f32>) -> Quat_0
{
    var r_3 : Quat_0;
    r_3.x_0 = q_4.x;
    r_3.y_0 = q_4.y;
    r_3.z_0 = q_4.z;
    r_3.w_0 = q_4.w;
    return r_3;
}

fn rotate_0( q_5 : Quat_0,  v_3 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_5.x_0, q_5.y_0, q_5.z_0);
    var t_3 : vec3<f32> = cross(qv_0, v_3) * vec3<f32>(2.0f);
    return v_3 + t_3 * vec3<f32>(q_5.w_0) + cross(qv_0, t_3);
}

fn child_pose_0( p_3 : u32,  first_2 : u32,  m_2 : vec2<u32>,  members_1 : u32,  anchored_0 : bool) -> vec3<f32>
{
    var _S38 : u32 = u32(5) * p_3;
    var pose_pos_0 : vec3<f32> = frames_0[_S38 + u32(4)].xyz;
    var pose_pos_1 : vec3<f32>;
    if(!anchored_0)
    {
        var pose_rot_0 : Quat_0 = quat_of_0(frames_0[_S38]);
        var _S39 : vec3<f32> = vec3<f32>(0.0f);
        var j_1 : u32 = u32(0);
        var sm_0 : f32 = 0.0f;
        pose_pos_1 = _S39;
        loop
        {
            if(j_1 < (m_2.y))
            {
            }
            else
            {
                break;
            }
            var c_3 : u32 = first_2 + work_0[members_1 + j_1];
            var cm_0 : f32 = chunks_0[c_3].center_0.w;
            var sm_1 : f32 = sm_0 + cm_0;
            var su_0 : vec3<f32> = pose_pos_1 + state_0[u32(4) * c_3].xyz * vec3<f32>(cm_0);
            j_1 = j_1 + u32(1);
            sm_0 = sm_1;
            pose_pos_1 = su_0;
        }
        pose_pos_1 = pose_pos_0 + rotate_0(pose_rot_0, pose_pos_1 / vec3<f32>(sm_0));
    }
    else
    {
        pose_pos_1 = pose_pos_0;
    }
    return pose_pos_1;
}

fn quat_mul_0( a_1 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_4 : Quat_0;
    r_4.w_0 = a_1.w_0 * o_0.w_0 - a_1.x_0 * o_0.x_0 - a_1.y_0 * o_0.y_0 - a_1.z_0 * o_0.z_0;
    r_4.x_0 = a_1.w_0 * o_0.x_0 + a_1.x_0 * o_0.w_0 + a_1.y_0 * o_0.z_0 - a_1.z_0 * o_0.y_0;
    r_4.y_0 = a_1.w_0 * o_0.y_0 - a_1.x_0 * o_0.z_0 + a_1.y_0 * o_0.w_0 + a_1.z_0 * o_0.x_0;
    r_4.z_0 = a_1.w_0 * o_0.z_0 + a_1.x_0 * o_0.y_0 - a_1.y_0 * o_0.x_0 + a_1.z_0 * o_0.w_0;
    return r_4;
}

fn rot_to_rubble_0( q_6 : Quat_0) -> Quat_0
{
    var axes_0 : Quat_0;
    axes_0.w_0 = 0.70710676908493042f;
    axes_0.x_0 = -0.70710676908493042f;
    axes_0.y_0 = 0.0f;
    axes_0.z_0 = 0.0f;
    return normalized_0(quat_mul_0(axes_0, q_6));
}

fn to_rubble_0( v_4 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_4.x, v_4.z, - v_4.y);
}

fn quat_vec_0( q_7 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_7.x_0, q_7.y_0, q_7.z_0, q_7.w_0);
}

fn child_record_0( slot_2 : u32,  p_4 : u32,  anchored_1 : bool,  mass_1 : f32,  com_1 : vec3<f32>,  principal_0 : Quat_0,  moments_1 : vec3<f32>,  pose_pos_2 : vec3<f32>)
{
    var pose_rot_1 : Quat_0 = quat_of_0(frames_0[u32(5) * p_4]);
    var at_2 : vec4<u32> = slots_0[slot_2];
    var orientation_0 : Quat_0 = rot_to_rubble_0(normalized_0(quat_mul_0(pose_rot_1, principal_0)));
    var _S40 : bool;
    if(anchored_1)
    {
        _S40 = true;
    }
    else
    {
        _S40 = !(mass_1 > 0.0f);
    }
    var inv_mass_0 : f32;
    if(_S40)
    {
        inv_mass_0 = 0.0f;
    }
    else
    {
        inv_mass_0 = 1.0f / mass_1;
    }
    var _S41 : vec3<f32> = vec3<f32>(0.0f);
    var inv_inertia_0 : vec3<f32>;
    if(!anchored_1)
    {
        var _S42 : f32 = moments_1.x;
        var _S43 : f32;
        if(_S42 > 0.0f)
        {
            _S43 = 1.0f / _S42;
        }
        else
        {
            _S43 = 0.0f;
        }
        var _S44 : f32 = moments_1.y;
        var _S45 : f32;
        if(_S44 > 0.0f)
        {
            _S45 = 1.0f / _S44;
        }
        else
        {
            _S45 = 0.0f;
        }
        var _S46 : f32 = moments_1.z;
        var _S47 : f32;
        if(_S46 > 0.0f)
        {
            _S47 = 1.0f / _S46;
        }
        else
        {
            _S47 = 0.0f;
        }
        inv_inertia_0 = vec3<f32>(_S43, _S45, _S47);
    }
    else
    {
        inv_inertia_0 = _S41;
    }
    var velocity_0 : vec3<f32>;
    if(anchored_1)
    {
        velocity_0 = _S41;
    }
    else
    {
        velocity_0 = to_rubble_0(children_0[u32(8) * slot_2 + u32(5)].xyz);
    }
    var spin_0 : vec3<f32>;
    if(anchored_1)
    {
        spin_0 = _S41;
    }
    else
    {
        spin_0 = to_rubble_0(children_0[u32(8) * slot_2 + u32(6)].xyz);
    }
    var flags_0 : u32;
    if(anchored_1)
    {
        flags_0 = u32(0);
    }
    else
    {
        flags_0 = u32(9);
    }
    var _S48 : array<vec4<f32>, i32(7)> = array<vec4<f32>, i32(7)>( vec4<f32>(to_rubble_0(pose_pos_2 + rotate_0(pose_rot_1, com_1)), inv_mass_0), quat_vec_0(orientation_0), vec4<f32>(inv_inertia_0, (bitcast<f32>((flags_0)))), vec4<f32>(velocity_0, 0.0f), vec4<f32>(spin_0, 0.0f), vec4<f32>(0.0f), vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f) );
    if((params_0.live_0) != u32(0))
    {
        flags_0 = at_2.x;
    }
    else
    {
        flags_0 = at_2.x - control_0[u32(4)];
    }
    var _S49 : u32 = u32(7) * flags_0;
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
    extras_0[_S50] = vec4<f32>(com_1, mass_1);
    extras_0[_S50 + u32(1)] = quat_vec_0(principal_0);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var slot_3 : u32 = id_0.x;
    var total_1 : u32 = total_slots_0();
    if(slot_3 >= total_1)
    {
        return;
    }
    var p_5 : u32 = parent_of_0(slot_3);
    var parent_1 : vec4<u32> = parents_0[p_5];
    var _S51 : u32 = parent_1.y;
    var _S52 : bool;
    if((slot_3 - _S51) >= (parent_1.z))
    {
        _S52 = true;
    }
    else
    {
        _S52 = (slots_0[slot_3].w) == u32(0);
    }
    if(_S52)
    {
        _S52 = true;
    }
    else
    {
        _S52 = control_0[u32(0)] != u32(0);
    }
    if(_S52)
    {
        return;
    }
    var first_3 : u32 = islands_of_0[p_5].x;
    var m_3 : vec2<u32> = child_members_0(slot_3);
    var anchored_2 : bool = (bitcast<u32>((children_0[u32(8) * slot_3].w))) != u32(0);
    var members_2 : u32 = total_1 + _S51 + m_3.x;
    var mass_2 : f32;
    var com_2 : vec3<f32>;
    var i0_1 : vec3<f32>;
    var i1_1 : vec3<f32>;
    var i2_1 : vec3<f32>;
    child_mass_0(first_3, m_3, members_2, &(mass_2), &(com_2), &(i0_1), &(i1_1), &(i2_1));
    var principal_1 : Quat_0;
    var moments_2 : vec3<f32>;
    principal_axes_0(i0_1, i1_1, i2_1, &(principal_1), &(moments_2));
    child_record_0(slot_3, p_5, anchored_2, mass_2, com_2, principal_1, moments_2, child_pose_0(p_5, first_3, m_3, members_2, anchored_2));
    return;
}

fn child_spawns_0( slot_4 : u32,  total_2 : u32,  p_6 : ptr<function, u32>,  parent_2 : ptr<function, vec4<u32>>) -> bool
{
    (*p_6) = u32(0);
    (*parent_2) = vec4<u32>(vec4<i32>(i32(0)));
    var _S53 : bool;
    if(slot_4 >= total_2)
    {
        _S53 = true;
    }
    else
    {
        _S53 = control_0[u32(0)] != u32(0);
    }
    if(_S53)
    {
        return false;
    }
    var _S54 : u32 = parent_of_0(slot_4);
    (*p_6) = _S54;
    var _S55 : vec4<u32> = parents_0[_S54];
    (*parent_2) = _S55;
    if((_S55.z) >= u32(2))
    {
        _S53 = (slot_4 - (*parent_2).y) < ((*parent_2).z);
    }
    else
    {
        _S53 = false;
    }
    return _S53;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_mass(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var slot_5 : u32 = id_1.x;
    var p_7 : u32;
    var parent_3 : vec4<u32>;
    var _S56 : bool = child_spawns_0(slot_5, total_slots_0(), &(p_7), &(parent_3));
    if(!_S56)
    {
        return;
    }
    var m_4 : vec2<u32> = child_members_0(slot_5);
    var mass_3 : f32;
    var com_3 : vec3<f32>;
    var i0_2 : vec3<f32>;
    var i1_2 : vec3<f32>;
    var i2_2 : vec3<f32>;
    child_mass_0(islands_of_0[p_7].x, m_4, total_slots_0() + parent_3.y + m_4.x, &(mass_3), &(com_3), &(i0_2), &(i1_2), &(i2_2));
    var principal_2 : Quat_0;
    var moments_3 : vec3<f32>;
    principal_axes_0(i0_2, i1_2, i2_2, &(principal_2), &(moments_3));
    var _S57 : u32 = u32(2) * slot_5;
    extras_0[_S57] = vec4<f32>(com_3, mass_3);
    extras_0[_S57 + u32(1)] = quat_vec_0(principal_2);
    extras_0[params_0.moments_base_0 + _S57] = vec4<f32>(moments_3, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_pose(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var slot_6 : u32 = id_2.x;
    var p_8 : u32;
    var parent_4 : vec4<u32>;
    var _S58 : bool = child_spawns_0(slot_6, total_slots_0(), &(p_8), &(parent_4));
    if(!_S58)
    {
        return;
    }
    var m_5 : vec2<u32> = child_members_0(slot_6);
    extras_0[params_0.moments_base_0 + u32(2) * slot_6 + u32(1)] = vec4<f32>(child_pose_0(p_8, islands_of_0[p_8].x, m_5, total_slots_0() + parent_4.y + m_5.x, (bitcast<u32>((children_0[u32(8) * slot_6].w))) != u32(0)), 0.0f);
    return;
}

fn phased_record_0( slot_7 : u32,  p_9 : u32)
{
    var _S59 : u32 = u32(2) * slot_7;
    child_record_0(slot_7, p_9, (bitcast<u32>((children_0[u32(8) * slot_7].w))) != u32(0), extras_0[_S59].w, extras_0[_S59].xyz, quat_of_0(extras_0[_S59 + u32(1)]), extras_0[params_0.moments_base_0 + _S59].xyz, extras_0[params_0.moments_base_0 + _S59 + u32(1)].xyz);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_record(@builtin(global_invocation_id) id_3 : vec3<u32>)
{
    var slot_8 : u32 = id_3.x;
    var p_10 : u32;
    var parent_5 : vec4<u32>;
    var _S60 : bool = child_spawns_0(slot_8, total_slots_0(), &(p_10), &(parent_5));
    var _S61 : bool;
    if(!_S60)
    {
        _S61 = true;
    }
    else
    {
        _S61 = (slots_0[slot_8].w) == u32(0);
    }
    if(_S61)
    {
        return;
    }
    phased_record_0(slot_8, p_10);
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
            var _S62 : u32 = tid_2 + s_2;
            g_s0_0[tid_2] = g_s0_0[tid_2] + g_s0_0[_S62];
            g_s1_0[tid_2] = g_s1_0[tid_2] + g_s1_0[_S62];
            g_s2_0[tid_2] = g_s2_0[tid_2] + g_s2_0[_S62];
            g_s3_0[tid_2] = g_s3_0[tid_2] + g_s3_0[_S62];
            g_s4_0[tid_2] = g_s4_0[tid_2] + g_s4_0[_S62];
        }
        workgroupBarrier();
        s_2 = (s_2 >> (u32(1)));
    }
    return;
}

fn tree_of_0( tid_3 : u32,  a_2 : vec4<f32>,  b_0 : vec4<f32>,  c_4 : vec4<f32>,  d_1 : vec4<f32>,  e_0 : vec4<f32>)
{
    g_s0_0[tid_3] = a_2;
    g_s1_0[tid_3] = b_0;
    g_s2_0[tid_3] = c_4;
    g_s3_0[tid_3] = d_1;
    g_s4_0[tid_3] = e_0;
    spawn_tree_0(tid_3);
    return;
}

fn group_first_tree_0( first_4 : u32,  m_6 : vec2<u32>,  members_3 : u32,  tid_4 : u32)
{
    var _S63 : vec4<f32> = vec4<f32>(0.0f);
    var j_2 : u32 = tid_4;
    var mm_0 : vec4<f32> = _S63;
    var su_1 : vec4<f32> = _S63;
    loop
    {
        if(j_2 < (m_6.y))
        {
        }
        else
        {
            break;
        }
        var c_5 : u32 = first_4 + work_0[members_3 + j_2];
        var r_5 : vec4<u32> = part_range_0(c_5);
        var _S64 : u32 = r_5.x;
        var k_2 : u32 = _S64;
        loop
        {
            if(k_2 < (_S64 + r_5.y))
            {
            }
            else
            {
                break;
            }
            var pc_0 : vec4<f32> = parts_0[u32(8) * k_2];
            var _S65 : f32 = pc_0.w;
            var mm_1 : vec4<f32> = mm_0 + vec4<f32>(pc_0.xyz * vec3<f32>(_S65), _S65);
            k_2 = k_2 + u32(1);
            mm_0 = mm_1;
        }
        var cm_1 : f32 = chunks_0[c_5].center_0.w;
        var su_2 : vec4<f32> = su_1 + vec4<f32>(state_0[u32(4) * c_5].xyz * vec3<f32>(cm_1), cm_1);
        j_2 = j_2 + u32(64);
        su_1 = su_2;
    }
    tree_of_0(tid_4, mm_0, _S63, _S63, _S63, su_1);
    return;
}

fn group_inertia_tree_0( first_5 : u32,  m_7 : vec2<u32>,  members_4 : u32,  tid_5 : u32,  com_4 : vec3<f32>,  su_total_0 : vec4<f32>)
{
    var _S66 : vec3<f32> = vec3<f32>(0.0f);
    var j_3 : u32 = tid_5;
    var i0_3 : vec3<f32> = _S66;
    var i1_3 : vec3<f32> = _S66;
    var i2_3 : vec3<f32> = _S66;
    loop
    {
        if(j_3 < (m_7.y))
        {
        }
        else
        {
            break;
        }
        var r_6 : vec4<u32> = part_range_0(first_5 + work_0[members_4 + j_3]);
        var _S67 : u32 = r_6.x;
        var k_3 : u32 = _S67;
        loop
        {
            if(k_3 < (_S67 + r_6.y))
            {
            }
            else
            {
                break;
            }
            var _S68 : u32 = u32(8) * k_3;
            var c_6 : vec4<f32> = parts_0[_S68];
            var d_2 : vec3<f32> = c_6.xyz - com_4;
            var dd_1 : f32 = dot(d_2, d_2);
            var _S69 : vec3<f32> = vec3<f32>(c_6.w);
            var i0_4 : vec3<f32> = i0_3 + (parts_0[_S68 + u32(1)].xyz + (vec3<f32>(dd_1, 0.0f, 0.0f) - vec3<f32>(d_2.x) * d_2) * _S69);
            var i1_4 : vec3<f32> = i1_3 + (parts_0[_S68 + u32(2)].xyz + (vec3<f32>(0.0f, dd_1, 0.0f) - vec3<f32>(d_2.y) * d_2) * _S69);
            var i2_4 : vec3<f32> = i2_3 + (parts_0[_S68 + u32(3)].xyz + (vec3<f32>(0.0f, 0.0f, dd_1) - vec3<f32>(d_2.z) * d_2) * _S69);
            k_3 = k_3 + u32(1);
            i0_3 = i0_4;
            i1_3 = i1_4;
            i2_3 = i2_4;
        }
        j_3 = j_3 + u32(64);
    }
    var _S70 : vec4<f32> = vec4<f32>(0.0f);
    var _S71 : vec4<f32> = vec4<f32>(i0_3, 0.0f);
    var _S72 : vec4<f32> = vec4<f32>(i1_3, 0.0f);
    var _S73 : vec4<f32> = vec4<f32>(i2_3, 0.0f);
    var _S74 : vec4<f32>;
    if(tid_5 == u32(0))
    {
        _S74 = su_total_0;
    }
    else
    {
        _S74 = _S70;
    }
    tree_of_0(tid_5, _S70, _S71, _S72, _S73, _S74);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_group(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var slot_9 : u32 = group_0.x;
    var tid_6 : u32 = thread_2.x;
    var total_3 : u32 = total_slots_0();
    var _S75 : bool;
    if(slot_9 >= total_3)
    {
        _S75 = true;
    }
    else
    {
        _S75 = control_0[u32(1)] == u32(0);
    }
    if(_S75)
    {
        _S75 = true;
    }
    else
    {
        _S75 = control_0[u32(0)] != u32(0);
    }
    var _S76 : u32;
    if(_S75)
    {
        _S76 = u32(1);
    }
    else
    {
        _S76 = u32(0);
    }
    var _S77 : u32 = group_uniform_0(_S76, tid_6);
    if(_S77 != u32(0))
    {
        return;
    }
    var p_11 : u32 = parent_at_0(slot_9);
    var parent_6 : vec4<u32> = parents_0[p_11];
    var _S78 : u32 = parent_6.y;
    if((slot_9 - _S78) >= (parent_6.z))
    {
        _S75 = true;
    }
    else
    {
        _S75 = (slots_0[slot_9].w) == u32(0);
    }
    if(_S75)
    {
        _S76 = u32(1);
    }
    else
    {
        _S76 = u32(0);
    }
    var _S79 : u32 = group_uniform_0(_S76, tid_6);
    if(_S79 != u32(0))
    {
        return;
    }
    var first_6 : u32 = islands_of_0[p_11].x;
    var m_8 : vec2<u32> = child_members_0(slot_9);
    var anchored_3 : bool = (bitcast<u32>((children_0[u32(8) * slot_9].w))) != u32(0);
    var members_5 : u32 = total_3 + _S78 + m_8.x;
    group_first_tree_0(first_6, m_8, members_5, tid_6);
    var mass_4 : f32 = g_s0_0[i32(0)].w;
    var com_5 : vec3<f32> = g_s0_0[i32(0)].xyz / vec3<f32>(max(mass_4, 1.00000000317107685e-30f));
    var su_total_1 : vec4<f32> = g_s4_0[i32(0)];
    workgroupBarrier();
    group_inertia_tree_0(first_6, m_8, members_5, tid_6, com_5, su_total_1);
    if(tid_6 != u32(0))
    {
        return;
    }
    var principal_3 : Quat_0;
    var moments_4 : vec3<f32>;
    principal_axes_0(g_s1_0[i32(0)].xyz, g_s2_0[i32(0)].xyz, g_s3_0[i32(0)].xyz, &(principal_3), &(moments_4));
    var _S80 : u32 = u32(5) * p_11;
    var pose_rot_2 : Quat_0 = quat_of_0(frames_0[_S80]);
    var pose_pos_3 : vec3<f32> = frames_0[_S80 + u32(4)].xyz;
    var pose_pos_4 : vec3<f32>;
    if(!anchored_3)
    {
        pose_pos_4 = pose_pos_3 + rotate_0(pose_rot_2, g_s4_0[i32(0)].xyz / vec3<f32>(g_s4_0[i32(0)].w));
    }
    else
    {
        pose_pos_4 = pose_pos_3;
    }
    child_record_0(slot_9, p_11, anchored_3, mass_4, com_5, principal_3, moments_4, pose_pos_4);
    return;
}

fn group_child_0( slot_10 : u32,  tid_7 : u32,  p_12 : ptr<function, u32>,  parent_7 : ptr<function, vec4<u32>>) -> bool
{
    (*p_12) = u32(0);
    (*parent_7) = vec4<u32>(vec4<i32>(i32(0)));
    var _S81 : bool;
    if(slot_10 >= (total_slots_0()))
    {
        _S81 = true;
    }
    else
    {
        _S81 = control_0[u32(1)] == u32(0);
    }
    if(_S81)
    {
        _S81 = true;
    }
    else
    {
        _S81 = control_0[u32(0)] != u32(0);
    }
    var _S82 : u32;
    if(_S81)
    {
        _S82 = u32(1);
    }
    else
    {
        _S82 = u32(0);
    }
    var _S83 : u32 = group_uniform_0(_S82, tid_7);
    if(_S83 != u32(0))
    {
        return false;
    }
    var _S84 : u32 = parent_at_0(slot_10);
    (*p_12) = _S84;
    var _S85 : vec4<u32> = parents_0[_S84];
    (*parent_7) = _S85;
    if((slot_10 - _S85.y) >= (_S85.z))
    {
        _S81 = true;
    }
    else
    {
        _S81 = ((*parent_7).z) < u32(2);
    }
    if(_S81)
    {
        _S82 = u32(1);
    }
    else
    {
        _S82 = u32(0);
    }
    var _S86 : u32 = group_uniform_0(_S82, tid_7);
    return _S86 == u32(0);
}

fn group_mass_tree_0( first_7 : u32,  m_9 : vec2<u32>,  members_6 : u32,  tid_8 : u32)
{
    var _S87 : vec4<f32> = vec4<f32>(0.0f);
    var j_4 : u32 = tid_8;
    var mm_2 : vec4<f32> = _S87;
    loop
    {
        if(j_4 < (m_9.y))
        {
        }
        else
        {
            break;
        }
        var r_7 : vec4<u32> = part_range_0(first_7 + work_0[members_6 + j_4]);
        var _S88 : u32 = r_7.x;
        var k_4 : u32 = _S88;
        loop
        {
            if(k_4 < (_S88 + r_7.y))
            {
            }
            else
            {
                break;
            }
            var pc_1 : vec4<f32> = parts_0[u32(8) * k_4];
            var _S89 : f32 = pc_1.w;
            var mm_3 : vec4<f32> = mm_2 + vec4<f32>(pc_1.xyz * vec3<f32>(_S89), _S89);
            k_4 = k_4 + u32(1);
            mm_2 = mm_3;
        }
        j_4 = j_4 + u32(64);
    }
    tree_of_0(tid_8, mm_2, _S87, _S87, _S87, _S87);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_group_mass(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var slot_11 : u32 = group_1.x;
    var tid_9 : u32 = thread_3.x;
    var p_13 : u32;
    var parent_8 : vec4<u32>;
    var _S90 : bool = group_child_0(slot_11, tid_9, &(p_13), &(parent_8));
    if(!_S90)
    {
        return;
    }
    var first_8 : u32 = islands_of_0[p_13].x;
    var m_10 : vec2<u32> = child_members_0(slot_11);
    var members_7 : u32 = total_slots_0() + parent_8.y + m_10.x;
    group_mass_tree_0(first_8, m_10, members_7, tid_9);
    var mass_5 : f32 = g_s0_0[i32(0)].w;
    var com_6 : vec3<f32> = g_s0_0[i32(0)].xyz / vec3<f32>(max(mass_5, 1.00000000317107685e-30f));
    workgroupBarrier();
    group_inertia_tree_0(first_8, m_10, members_7, tid_9, com_6, vec4<f32>(0.0f));
    if(tid_9 != u32(0))
    {
        return;
    }
    var principal_4 : Quat_0;
    var moments_5 : vec3<f32>;
    principal_axes_0(g_s1_0[i32(0)].xyz, g_s2_0[i32(0)].xyz, g_s3_0[i32(0)].xyz, &(principal_4), &(moments_5));
    var _S91 : u32 = u32(2) * slot_11;
    extras_0[_S91] = vec4<f32>(com_6, mass_5);
    extras_0[_S91 + u32(1)] = quat_vec_0(principal_4);
    extras_0[params_0.moments_base_0 + _S91] = vec4<f32>(moments_5, 0.0f);
    return;
}

fn group_su_tree_0( first_9 : u32,  m_11 : vec2<u32>,  members_8 : u32,  tid_10 : u32)
{
    var _S92 : vec4<f32> = vec4<f32>(0.0f);
    var j_5 : u32 = tid_10;
    var su_3 : vec4<f32> = _S92;
    loop
    {
        if(j_5 < (m_11.y))
        {
        }
        else
        {
            break;
        }
        var c_7 : u32 = first_9 + work_0[members_8 + j_5];
        var cm_2 : f32 = chunks_0[c_7].center_0.w;
        var su_4 : vec4<f32> = su_3 + vec4<f32>(state_0[u32(4) * c_7].xyz * vec3<f32>(cm_2), cm_2);
        j_5 = j_5 + u32(64);
        su_3 = su_4;
    }
    tree_of_0(tid_10, _S92, _S92, _S92, _S92, su_3);
    return;
}

fn group_carry_tree_0( tid_11 : u32,  su_total_2 : vec4<f32>)
{
    var _S93 : vec4<f32> = vec4<f32>(0.0f);
    var _S94 : vec4<f32>;
    if(tid_11 == u32(0))
    {
        _S94 = su_total_2;
    }
    else
    {
        _S94 = _S93;
    }
    tree_of_0(tid_11, _S93, _S93, _S93, _S93, _S94);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_group_pose(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var slot_12 : u32 = group_2.x;
    var tid_12 : u32 = thread_4.x;
    var p_14 : u32;
    var parent_9 : vec4<u32>;
    var _S95 : bool = group_child_0(slot_12, tid_12, &(p_14), &(parent_9));
    if(!_S95)
    {
        return;
    }
    var m_12 : vec2<u32> = child_members_0(slot_12);
    group_su_tree_0(islands_of_0[p_14].x, m_12, total_slots_0() + parent_9.y + m_12.x, tid_12);
    var su_total_3 : vec4<f32> = g_s4_0[i32(0)];
    workgroupBarrier();
    group_carry_tree_0(tid_12, su_total_3);
    if(tid_12 != u32(0))
    {
        return;
    }
    var pose_rot_3 : Quat_0 = quat_of_0(frames_0[u32(5) * p_14]);
    var pose_pos_5 : vec3<f32> = frames_0[u32(5) * p_14 + u32(4)].xyz;
    var pose_pos_6 : vec3<f32>;
    if(!((bitcast<u32>((children_0[u32(8) * slot_12].w))) != u32(0)))
    {
        pose_pos_6 = pose_pos_5 + rotate_0(pose_rot_3, g_s4_0[i32(0)].xyz / vec3<f32>(g_s4_0[i32(0)].w));
    }
    else
    {
        pose_pos_6 = pose_pos_5;
    }
    extras_0[params_0.moments_base_0 + u32(2) * slot_12 + u32(1)] = vec4<f32>(pose_pos_6, 0.0f);
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_bodies_group_record(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var slot_13 : u32 = group_3.x;
    var p_15 : u32;
    var parent_10 : vec4<u32>;
    var _S96 : bool;
    if((thread_5.x) != u32(0))
    {
        _S96 = true;
    }
    else
    {
        var _S97 : bool = child_spawns_0(slot_13, total_slots_0(), &(p_15), &(parent_10));
        _S96 = !_S97;
    }
    if(_S96)
    {
        _S96 = true;
    }
    else
    {
        _S96 = (slots_0[slot_13].w) == u32(0);
    }
    if(_S96)
    {
        return;
    }
    phased_record_0(slot_13, p_15);
    return;
}

fn member_info_0( i_3 : u32,  total_4 : u32,  col_0 : ptr<function, u32>) -> vec4<u32>
{
    (*col_0) = u32(0);
    var p_16 : u32 = parent_at_0(i_3);
    var parent_11 : vec4<u32> = parents_0[p_16];
    var _S98 : u32 = parent_11.z;
    if(_S98 < u32(2))
    {
        return vec4<u32>(u32(4294967295), u32(0), u32(0), u32(0));
    }
    var _S99 : u32 = parent_11.y;
    var _S100 : u32 = i_3 - _S99;
    var _S101 : u32 = _S98 - u32(1);
    var lo_1 : u32 = u32(0);
    var hi_1 : u32 = _S101;
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
        if((child_members_0(_S99 + mid_1).x) <= _S100)
        {
            lo_1 = mid_1;
        }
        else
        {
            hi_1 = mid_1 - u32(1);
        }
    }
    var slot_14 : u32 = _S99 + lo_1;
    var at_3 : vec4<u32> = slots_0[slot_14];
    if((slots_0[slot_14].w) == u32(0))
    {
        return vec4<u32>(u32(4294967295), u32(0), u32(0), u32(0));
    }
    var r_8 : vec4<u32> = part_range_0(islands_of_0[p_16].x + work_0[total_4 + i_3]);
    (*col_0) = member_offsets_0[i_3];
    return vec4<u32>(slot_14, at_3.x, r_8.x, r_8.y);
}

fn conj_0( q_8 : Quat_0) -> Quat_0
{
    var c_8 : Quat_0;
    c_8.w_0 = q_8.w_0;
    c_8.x_0 = - q_8.x_0;
    c_8.y_0 = - q_8.y_0;
    c_8.z_0 = - q_8.z_0;
    return c_8;
}

fn member_colliders_0( info_1 : vec4<u32>,  col_1 : u32)
{
    var _S102 : u32 = u32(2) * info_1.x;
    var _S103 : vec3<f32> = extras_0[_S102].xyz;
    var _S104 : Quat_0 = conj_0(quat_of_0(extras_0[_S102 + u32(1)]));
    var _S105 : u32 = info_1.z;
    var k_5 : u32 = _S105;
    var _S106 : u32 = col_1;
    loop
    {
        if(k_5 < (_S105 + info_1.w))
        {
        }
        else
        {
            break;
        }
        var _S107 : u32 = u32(8) * k_5;
        var o_1 : vec4<f32> = parts_0[_S107 + u32(4)];
        var rot_1 : Quat_0 = quat_of_0(parts_0[_S107 + u32(5)]);
        var ids_0 : vec4<f32> = parts_0[_S107 + u32(6)];
        var shape_params_0 : vec4<f32> = parts_0[_S107 + u32(7)];
        var _S108 : u32;
        if((params_0.live_0) != u32(0))
        {
            _S108 = _S106;
        }
        else
        {
            _S108 = _S106 - control_0[u32(5)];
        }
        var cw_0 : u32 = u32(4) * _S108;
        colliders_0[cw_0] = vec4<f32>((bitcast<f32>((info_1.y))), ids_0.x, ids_0.y, ids_0.z);
        colliders_0[cw_0 + u32(1)] = vec4<f32>(rotate_0(_S104, o_1.xyz - _S103), o_1.w);
        colliders_0[cw_0 + u32(2)] = quat_vec_0(normalized_0(quat_mul_0(_S104, rot_1)));
        colliders_0[cw_0 + u32(3)] = shape_params_0;
        var _S109 : u32 = _S106 + u32(1);
        k_5 = k_5 + u32(1);
        _S106 = _S109;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_colliders(@builtin(global_invocation_id) id_4 : vec3<u32>)
{
    var i_4 : u32 = id_4.x;
    var total_5 : u32 = total_slots_0();
    var _S110 : bool;
    if(i_4 >= total_5)
    {
        _S110 = true;
    }
    else
    {
        _S110 = control_0[u32(1)] == u32(0);
    }
    if(_S110)
    {
        _S110 = true;
    }
    else
    {
        _S110 = control_0[u32(0)] != u32(0);
    }
    if(_S110)
    {
        return;
    }
    var col_2 : u32;
    var info_2 : vec4<u32> = member_info_0(i_4, total_5, &(col_2));
    if((info_2.x) != u32(4294967295))
    {
        member_colliders_0(info_2, col_2);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_collider_members(@builtin(global_invocation_id) id_5 : vec3<u32>)
{
    var i_5 : u32 = id_5.x;
    var total_6 : u32 = total_slots_0();
    var _S111 : bool;
    if(i_5 >= total_6)
    {
        _S111 = true;
    }
    else
    {
        _S111 = control_0[u32(1)] == u32(0);
    }
    if(_S111)
    {
        _S111 = true;
    }
    else
    {
        _S111 = control_0[u32(0)] != u32(0);
    }
    if(_S111)
    {
        return;
    }
    var col_3 : u32;
    var _S112 : vec4<u32> = member_info_0(i_5, total_6, &(col_3));
    var w_2 : u32 = u32(0);
    loop
    {
        if(w_2 < u32(4))
        {
        }
        else
        {
            break;
        }
        member_offsets_0[params_0.info_base_0 + u32(4) * i_5 + w_2] = _S112[w_2];
        w_2 = w_2 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_colliders_phased(@builtin(global_invocation_id) id_6 : vec3<u32>)
{
    var i_6 : u32 = id_6.x;
    var _S113 : bool;
    if(i_6 >= (total_slots_0()))
    {
        _S113 = true;
    }
    else
    {
        _S113 = control_0[u32(1)] == u32(0);
    }
    if(_S113)
    {
        _S113 = true;
    }
    else
    {
        _S113 = control_0[u32(0)] != u32(0);
    }
    if(_S113)
    {
        return;
    }
    var _S114 : u32 = u32(4) * i_6;
    var info_3 : vec4<u32> = vec4<u32>(member_offsets_0[params_0.info_base_0 + _S114], member_offsets_0[params_0.info_base_0 + _S114 + u32(1)], member_offsets_0[params_0.info_base_0 + _S114 + u32(2)], member_offsets_0[params_0.info_base_0 + _S114 + u32(3)]);
    if(member_offsets_0[params_0.info_base_0 + _S114] != u32(4294967295))
    {
        member_colliders_0(info_3, member_offsets_0[i_6]);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn spawn_retire(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var p_17 : u32 = group_4.x;
    var tid_13 : u32 = thread_6.x;
    var _S115 : bool;
    if(p_17 >= control_0[u32(1)])
    {
        _S115 = true;
    }
    else
    {
        _S115 = (params_0.live_0) == u32(0);
    }
    if(_S115)
    {
        _S115 = true;
    }
    else
    {
        _S115 = (parents_0[p_17].z) < u32(2);
    }
    if(_S115)
    {
        _S115 = true;
    }
    else
    {
        _S115 = control_0[i32(0)] != u32(0);
    }
    if(_S115)
    {
        return;
    }
    var e_1 : vec4<u32> = islands_of_0[p_17];
    var c_9 : u32;
    if(tid_13 == u32(0))
    {
        var _S116 : u32 = e_1.y;
        var _S117 : u32 = u32(7) * _S116;
        var record_0 : array<vec4<f32>, i32(7)>;
        c_9 = u32(0);
        loop
        {
            if(c_9 < u32(7))
            {
            }
            else
            {
                break;
            }
            record_0[c_9] = bodies_0[_S117 + c_9];
            c_9 = c_9 + u32(1);
        }
        record_0[i32(0)][i32(3)] = 0.0f;
        record_0[i32(2)] = vec4<f32>(0.0f, 0.0f, 0.0f, (bitcast<f32>((u32(0)))));
        var _S118 : vec3<f32> = vec3<f32>(0.0f);
        record_0[i32(3)].x = _S118.x;
        record_0[i32(3)].y = _S118.y;
        record_0[i32(3)].z = _S118.z;
        record_0[i32(4)].x = _S118.x;
        record_0[i32(4)].y = _S118.y;
        record_0[i32(4)].z = _S118.z;
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
            c_9 = u32(0);
            loop
            {
                if(c_9 < u32(7))
                {
                }
                else
                {
                    break;
                }
                bodies_0[u32(7) * (_S116 + region_0 * params_0.max_bodies_0) + c_9] = record_0[c_9];
                c_9 = c_9 + u32(1);
            }
            region_0 = region_0 + u32(1);
        }
    }
    c_9 = e_1.z + tid_13;
    loop
    {
        if(c_9 < (e_1.w))
        {
        }
        else
        {
            break;
        }
        var _S119 : u32 = u32(4) * c_9;
        colliders_0[_S119] = vec4<f32>(colliders_0[_S119].x, colliders_0[_S119].y, (bitcast<f32>((((bitcast<u32>((colliders_0[_S119].z))) & (u32(65535)))))), (bitcast<f32>((u32(32768)))));
        c_9 = c_9 + u32(64);
    }
    return;
}

