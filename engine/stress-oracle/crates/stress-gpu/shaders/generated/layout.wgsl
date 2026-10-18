@binding(24) @group(0) var<storage, read_write> control_0 : array<u32>;

@binding(5) @group(0) var<storage, read_write> plan_0 : array<u32>;

@binding(1) @group(0) var<storage, read> parents_0 : array<vec4<u32>>;

struct LayoutParams_std140_0
{
    @align(16) parent_count_0 : u32,
    @align(4) island_count_0 : u32,
    @align(8) next_id_0 : u32,
    @align(4) substeps_0 : u32,
    @align(16) chunk_words_0 : u32,
    @align(4) bond_words_0 : u32,
    @align(8) dyn_words_0 : u32,
    @align(4) island_words_0 : u32,
    @align(16) chunk_count_0 : u32,
    @align(4) statics_base_0 : u32,
    @align(8) statics_bonds_0 : u32,
    @align(4) resident_0 : u32,
    @align(16) props_base_0 : u32,
    @align(4) eload_base_0 : u32,
    @align(8) pulse_base_0 : u32,
    @align(4) pulse_count_base_0 : u32,
    @align(16) pulse_cap_0 : u32,
    @align(4) engine_base_0 : u32,
    @align(8) total_0 : u32,
    @align(4) gate_flag_base_0 : u32,
    @align(16) bond_count_0 : u32,
    @align(4) driven_0 : u32,
    @align(8) pad2_0 : u32,
    @align(4) pad3_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : LayoutParams_std140_0;
@binding(16) @group(0) var<storage, read> old_islands_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> children_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> work_0 : array<u32>;

@binding(7) @group(0) var<storage, read_write> chunks_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read> old_chunks_0 : array<vec4<f32>>;

@binding(9) @group(0) var<storage, read_write> state_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> old_state_0 : array<vec4<f32>>;

@binding(18) @group(0) var<storage, read_write> scratch_0 : array<vec4<f32>>;

@binding(20) @group(0) var<storage, read_write> ctab_0 : array<vec4<f32>>;

@binding(19) @group(0) var<storage, read> old_ctab_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> old_bonds_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read_write> bonds_0 : array<vec4<f32>>;

@binding(13) @group(0) var<storage, read_write> dyn_0 : array<vec4<f32>>;

@binding(12) @group(0) var<storage, read> old_dyn_0 : array<vec4<f32>>;

@binding(14) @group(0) var<storage, read> old_index_0 : array<u32>;

@binding(15) @group(0) var<storage, read_write> index_0 : array<u32>;

@binding(4) @group(0) var<storage, read> frames_0 : array<vec4<f32>>;

@binding(17) @group(0) var<storage, read_write> islands_0 : array<vec4<f32>>;

@binding(22) @group(0) var<storage, read> spawn_slots_0 : array<vec4<u32>>;

@binding(23) @group(0) var<storage, read> spawn_extras_0 : array<vec4<f32>>;

@binding(21) @group(0) var<storage, read_write> gate_0 : array<vec4<f32>>;

@binding(26) @group(0) var<storage, read_write> island_copy_0 : array<vec4<f32>>;

@binding(25) @group(0) var<storage, read> old_copy_0 : array<vec4<f32>>;

fn order_at_0( i_0 : u32) -> u32
{
    return u32(16) + i_0;
}

fn cap_0() -> u32
{
    return control_0[u32(2)] + control_0[u32(6)];
}

fn child_id_at_0( slot_0 : u32) -> u32
{
    return u32(16) + cap_0() + control_0[u32(6)] + slot_0;
}

fn old_new_at_0( j_0 : u32) -> u32
{
    return u32(16) + cap_0() + u32(2) * control_0[u32(6)] + j_0;
}

fn child_island_at_0( slot_1 : u32) -> u32
{
    return u32(16) + cap_0() + slot_1;
}

@compute
@workgroup_size(1, 1, 1)
fn layout_plan(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var k_0 : u32;
    var _S1 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S1 = true;
    }
    else
    {
        _S1 = control_0[u32(1)] == u32(0);
    }
    if(_S1)
    {
        return;
    }
    var n_0 : u32 = control_0[u32(2)];
    var i_1 : u32 = u32(0);
    loop
    {
        if(i_1 < n_0)
        {
        }
        else
        {
            break;
        }
        plan_0[order_at_0(i_1)] = i_1;
        i_1 = i_1 + u32(1);
    }
    var q_0 : u32 = control_0[u32(1)];
    var len_0 : u32 = n_0;
    var next_0 : u32 = control_0[u32(3)];
    loop
    {
        if(q_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        var p_0 : u32 = q_0 - u32(1);
        var parent_0 : vec4<u32> = parents_0[p_0];
        var _S2 : u32 = parent_0.z;
        if(_S2 < u32(2))
        {
            q_0 = p_0;
            continue;
        }
        var len_1 : u32 = len_0 - u32(1);
        var _S3 : u32 = parent_0.x;
        if(_S3 < len_1)
        {
            plan_0[order_at_0(_S3)] = plan_0[order_at_0(len_1)];
        }
        k_0 = u32(0);
        var len_2 : u32 = len_1;
        var next_1 : u32 = next_0;
        loop
        {
            if(k_0 < _S2)
            {
            }
            else
            {
                break;
            }
            var _S4 : u32 = parent_0.y;
            plan_0[order_at_0(len_2)] = n_0 + _S4 + k_0;
            var _S5 : u32 = len_2 + u32(1);
            plan_0[child_id_at_0(_S4 + k_0)] = next_1;
            var _S6 : u32 = next_1 + u32(1);
            k_0 = k_0 + u32(1);
            len_2 = _S5;
            next_1 = _S6;
        }
        len_0 = len_2;
        next_0 = next_1;
        q_0 = p_0;
    }
    plan_0[i32(0)] = len_0;
    control_0[u32(7)] = len_0;
    control_0[u32(10)] = next_0;
    k_0 = u32(0);
    loop
    {
        if(k_0 < n_0)
        {
        }
        else
        {
            break;
        }
        plan_0[old_new_at_0(k_0)] = u32(4294967295);
        k_0 = k_0 + u32(1);
    }
    i_1 = u32(0);
    loop
    {
        if(i_1 < len_0)
        {
        }
        else
        {
            break;
        }
        var e_0 : u32 = plan_0[order_at_0(i_1)];
        if(plan_0[order_at_0(i_1)] < n_0)
        {
            plan_0[old_new_at_0(e_0)] = i_1;
        }
        else
        {
            plan_0[child_island_at_0(e_0 - n_0)] = i_1;
        }
        i_1 = i_1 + u32(1);
    }
    return;
}

fn island_word_0( i_2 : u32,  w_0 : u32) -> u32
{
    return params_0.island_words_0 * i_2 + w_0;
}

fn child_order_at_0( slot_2 : u32) -> u32
{
    return u32(16) + cap_0() + u32(5) * control_0[u32(6)] + control_0[u32(2)] + params_0.bond_count_0 + slot_2;
}

fn sort_children_0( parent_1 : vec4<u32>)
{
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < (parent_1.z))
        {
        }
        else
        {
            break;
        }
        var _S7 : u32 = parent_1.y;
        var _S8 : u32 = plan_0[child_island_at_0(_S7 + k_1)];
        var j_1 : u32 = k_1;
        loop
        {
            var _S9 : bool;
            if(j_1 > u32(0))
            {
                _S9 = plan_0[child_island_at_0(_S7 + plan_0[child_order_at_0(_S7 + j_1 - u32(1))])] > _S8;
            }
            else
            {
                _S9 = false;
            }
            if(_S9)
            {
            }
            else
            {
                break;
            }
            var _S10 : u32 = _S7 + j_1;
            plan_0[child_order_at_0(_S10)] = plan_0[child_order_at_0(_S10 - u32(1))];
            j_1 = j_1 - u32(1);
        }
        plan_0[child_order_at_0(_S7 + j_1)] = k_1;
        k_1 = k_1 + u32(1);
    }
    return;
}

fn child_by_island_0( parent_2 : vec4<u32>,  r_0 : u32) -> u32
{
    return plan_0[child_order_at_0(parent_2.y + r_0)];
}

fn child_start_at_0( slot_3 : u32) -> u32
{
    return u32(16) + cap_0() + u32(6) * control_0[u32(6)] + control_0[u32(2)] + params_0.bond_count_0 + slot_3;
}

fn device_group_barrier_0()
{
    storageBarrier(); textureBarrier(); workgroupBarrier();;
    return;
}

fn new_chunk_at_0( slot_4 : u32) -> u32
{
    return u32(16) + cap_0() + u32(2) * control_0[u32(6)] + control_0[u32(2)] + slot_4;
}

fn chunk_word_0( c_0 : u32,  w_1 : u32) -> u32
{
    return params_0.chunk_words_0 * c_0 + w_1;
}

fn set_chunk_island_0( c_1 : u32,  island_0 : u32)
{
    var _S11 : u32 = chunk_word_0(c_1, u32(8));
    var info_0 : vec4<f32> = chunks_0[_S11];
    info_0[i32(1)] = (bitcast<f32>((island_0)));
    chunks_0[_S11] = info_0;
    return;
}

fn child_bonds_at_0( slot_5 : u32) -> u32
{
    return u32(16) + cap_0() + u32(3) * control_0[u32(6)] + control_0[u32(2)] + u32(2) * slot_5;
}

fn bond_word_0( b_0 : u32,  w_2 : u32) -> u32
{
    return params_0.bond_words_0 * b_0 + w_2;
}

var<workgroup> g_class_0 : array<u32, i32(256)>;

var<workgroup> g_cut_next_0 : u32;

fn bond_new_at_0( b_1 : u32) -> u32
{
    return u32(16) + cap_0() + u32(5) * control_0[u32(6)] + control_0[u32(2)] + b_1;
}

fn bond_ranks_0( parent_3 : vec4<u32>,  base_0 : u32,  c0_0 : u32,  b0_0 : u32,  b1_0 : u32,  tid_0 : u32,  place_0 : bool)
{
    var _S12 : bool;
    var block_0 : u32 = b0_0;
    loop
    {
        if(block_0 < b1_0)
        {
        }
        else
        {
            break;
        }
        var b_2 : u32 = block_0 + tid_0;
        var cls_0 : u32;
        if(b_2 < b1_0)
        {
            var ids_0 : vec4<u32> = (bitcast<vec4<u32>>((old_bonds_0[bond_word_0(b_2, u32(14))])));
            var ka_0 : u32 = work_0[base_0 + ids_0.y - c0_0];
            if(ka_0 == work_0[base_0 + ids_0.z - c0_0])
            {
                cls_0 = ka_0;
            }
            else
            {
                cls_0 = u32(4294967294);
            }
        }
        else
        {
            cls_0 = u32(4294967295);
        }
        g_class_0[tid_0] = cls_0;
        workgroupBarrier();
        var _S13 : bool = cls_0 != u32(4294967295);
        var last_0 : bool = _S13;
        var t_0 : u32 = u32(0);
        var rank_0 : u32 = u32(0);
        loop
        {
            if(t_0 < u32(256))
            {
            }
            else
            {
                break;
            }
            if(t_0 < tid_0)
            {
                _S12 = (g_class_0[t_0]) == cls_0;
            }
            else
            {
                _S12 = false;
            }
            if(_S12)
            {
                rank_0 = rank_0 + u32(1);
            }
            var _S14 : bool;
            if(t_0 > tid_0)
            {
                _S14 = (g_class_0[t_0]) == cls_0;
            }
            else
            {
                _S14 = false;
            }
            if(_S14)
            {
                last_0 = false;
            }
            t_0 = t_0 + u32(1);
        }
        if(place_0)
        {
            _S12 = _S13;
        }
        else
        {
            _S12 = false;
        }
        if(_S12)
        {
            var first_0 : u32;
            if(cls_0 == u32(4294967294))
            {
                first_0 = g_cut_next_0;
            }
            else
            {
                first_0 = plan_0[child_bonds_at_0(base_0 + cls_0) + u32(1)];
            }
            plan_0[bond_new_at_0(b_2)] = first_0 + rank_0;
        }
        device_group_barrier_0();
        if(last_0)
        {
            if(place_0)
            {
                if(cls_0 == u32(4294967294))
                {
                    g_cut_next_0 = g_cut_next_0 + (rank_0 + u32(1));
                }
                else
                {
                    plan_0[child_bonds_at_0(base_0 + cls_0) + u32(1)] = plan_0[child_bonds_at_0(base_0 + cls_0) + u32(1)] + (rank_0 + u32(1));
                }
            }
            else
            {
                if(cls_0 != u32(4294967294))
                {
                    plan_0[child_bonds_at_0(base_0 + cls_0)] = plan_0[child_bonds_at_0(base_0 + cls_0)] + (rank_0 + u32(1));
                }
            }
        }
        device_group_barrier_0();
        block_0 = block_0 + u32(256);
    }
    return;
}

var<workgroup> g_cut_first_0 : u32;

fn degree_at_0( slot_6 : u32) -> u32
{
    return u32(16) + cap_0() + u32(7) * control_0[u32(6)] + control_0[u32(2)] + params_0.bond_count_0 + slot_6;
}

var<workgroup> g_scan64_0 : array<u32, i32(256)>;

fn inverse3_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  i0_0 : ptr<function, vec3<f32>>,  i1_0 : ptr<function, vec3<f32>>,  i2_0 : ptr<function, vec3<f32>>)
{
    var c0_1 : vec3<f32> = cross(r1_0, r2_0);
    var c1_0 : vec3<f32> = cross(r2_0, r0_0);
    var c2_0 : vec3<f32> = cross(r0_0, r1_0);
    var det_0 : f32 = dot(r0_0, c0_1);
    var inv_0 : f32;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    (*i0_0) = vec3<f32>(c0_1.x, c1_0.x, c2_0.x) * vec3<f32>(inv_0);
    (*i1_0) = vec3<f32>(c0_1.y, c1_0.y, c2_0.y) * vec3<f32>(inv_0);
    (*i2_0) = vec3<f32>(c0_1.z, c1_0.z, c2_0.z) * vec3<f32>(inv_0);
    return;
}

struct Quat_0
{
     w_3 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_1 : vec4<f32>) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.x_0 = q_1.x;
    r_1.y_0 = q_1.y;
    r_1.z_0 = q_1.z;
    r_1.w_3 = q_1.w;
    return r_1;
}

fn rotate_0( q_2 : Quat_0,  v_0 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_2.x_0, q_2.y_0, q_2.z_0);
    var t_1 : vec3<f32> = cross(qv_0, v_0) * vec3<f32>(2.0f);
    return v_0 + t_1 * vec3<f32>(q_2.w_3) + cross(qv_0, t_1);
}

fn inverse_rotate_0( q_3 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var c_2 : Quat_0;
    c_2.w_3 = q_3.w_3;
    c_2.x_0 = - q_3.x_0;
    c_2.y_0 = - q_3.y_0;
    c_2.z_0 = - q_3.z_0;
    return rotate_0(c_2, v_1);
}

fn quat_vec_0( q_4 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_4.x_0, q_4.y_0, q_4.z_0, q_4.w_3);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_0 : vec3<u32>)
{
    var hi_0 : u32;
    var w_4 : u32;
    var w_5 : u32;
    var w_6 : u32;
    var v_2 : u32;
    var step_0 : u32;
    var add_0 : u32;
    var tid_1 : u32 = thread_0.x;
    var p_1 : u32 = group_0.x;
    var scaled_0 : bool;
    if(p_1 >= control_0[u32(1)])
    {
        scaled_0 = true;
    }
    else
    {
        scaled_0 = control_0[i32(0)] != u32(0);
    }
    if(scaled_0)
    {
        return;
    }
    var parent_4 : vec4<u32> = parents_0[p_1];
    var _S15 : u32 = parent_4.z;
    if(_S15 < u32(2))
    {
        return;
    }
    var r_2 : vec4<f32> = old_islands_0[island_word_0(parent_4.x, u32(0))];
    var c0_2 : u32 = (bitcast<u32>((r_2.x)));
    var c1_1 : u32 = (bitcast<u32>((r_2.y)));
    var b0_1 : u32 = (bitcast<u32>((r_2.z)));
    var b1_1 : u32 = (bitcast<u32>((r_2.w)));
    var _S16 : u32 = c1_1 - c0_2;
    var base_1 : u32 = parent_4.y;
    var _S17 : u32 = control_0[u32(6)];
    var _S18 : bool = tid_1 == u32(0);
    var i_3 : u32;
    var lo_0 : u32;
    if(_S18)
    {
        sort_children_0(parent_4);
        i_3 = u32(0);
        lo_0 = u32(0);
        loop
        {
            if(i_3 < _S15)
            {
            }
            else
            {
                break;
            }
            var _S19 : u32 = base_1 + child_by_island_0(parent_4, i_3);
            plan_0[child_start_at_0(_S19)] = lo_0;
            var at_0 : u32 = lo_0 + (bitcast<u32>((children_0[u32(8) * _S19].z)));
            i_3 = i_3 + u32(1);
            lo_0 = at_0;
        }
    }
    device_group_barrier_0();
    i_3 = tid_1;
    loop
    {
        if(i_3 < _S16)
        {
        }
        else
        {
            break;
        }
        var _S20 : u32 = _S15 - u32(1);
        lo_0 = u32(0);
        hi_0 = _S20;
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
            if((bitcast<u32>((children_0[u32(8) * (base_1 + mid_0)].y))) <= i_3)
            {
                lo_0 = mid_0;
            }
            else
            {
                hi_0 = mid_0 - u32(1);
            }
        }
        var _S21 : u32 = base_1 + lo_0;
        plan_0[new_chunk_at_0(base_1 + work_0[_S17 + base_1 + i_3])] = plan_0[child_start_at_0(_S21)] + (i_3 - (bitcast<u32>((children_0[u32(8) * _S21].y))));
        i_3 = i_3 + u32(256);
    }
    device_group_barrier_0();
    lo_0 = tid_1;
    loop
    {
        if(lo_0 < _S16)
        {
        }
        else
        {
            break;
        }
        var _S22 : u32 = base_1 + lo_0;
        var nk_0 : u32 = plan_0[new_chunk_at_0(_S22)];
        var island_1 : u32 = plan_0[child_island_at_0(base_1 + work_0[_S22])];
        w_4 = u32(0);
        loop
        {
            if(w_4 < (params_0.chunk_words_0))
            {
            }
            else
            {
                break;
            }
            chunks_0[chunk_word_0(c0_2 + nk_0, w_4)] = old_chunks_0[chunk_word_0(c0_2 + lo_0, w_4)];
            w_4 = w_4 + u32(1);
        }
        var _S23 : u32 = c0_2 + nk_0;
        set_chunk_island_0(_S23, island_1);
        w_5 = u32(0);
        loop
        {
            if(w_5 < u32(4))
            {
            }
            else
            {
                break;
            }
            state_0[u32(4) * _S23 + w_5] = old_state_0[u32(4) * (c0_2 + lo_0) + w_5];
            w_5 = w_5 + u32(1);
        }
        w_6 = u32(0);
        loop
        {
            if(w_6 < u32(23))
            {
            }
            else
            {
                break;
            }
            scratch_0[params_0.statics_base_0 + u32(23) * _S23 + w_6] = vec4<f32>(0.0f);
            w_6 = w_6 + u32(1);
        }
        if((params_0.resident_0) != u32(0))
        {
            var o_0 : u32 = c0_2 + lo_0;
            ctab_0[params_0.props_base_0 + _S23] = old_ctab_0[params_0.props_base_0 + o_0];
            var _S24 : u32 = u32(2) * _S23;
            var _S25 : u32 = u32(2) * o_0;
            ctab_0[params_0.eload_base_0 + _S24] = old_ctab_0[params_0.eload_base_0 + _S25];
            ctab_0[params_0.eload_base_0 + _S24 + u32(1)] = old_ctab_0[params_0.eload_base_0 + _S25 + u32(1)];
            ctab_0[params_0.pulse_count_base_0 + _S23] = old_ctab_0[params_0.pulse_count_base_0 + o_0];
            hi_0 = u32(0);
            loop
            {
                if(hi_0 < (u32(3) * params_0.pulse_cap_0))
                {
                }
                else
                {
                    break;
                }
                ctab_0[params_0.pulse_base_0 + u32(3) * params_0.pulse_cap_0 * _S23 + hi_0] = old_ctab_0[params_0.pulse_base_0 + u32(3) * params_0.pulse_cap_0 * o_0 + hi_0];
                hi_0 = hi_0 + u32(1);
            }
        }
        lo_0 = lo_0 + u32(256);
    }
    device_group_barrier_0();
    hi_0 = tid_1;
    loop
    {
        if(hi_0 < _S15)
        {
        }
        else
        {
            break;
        }
        plan_0[child_bonds_at_0(base_1 + hi_0)] = u32(0);
        hi_0 = hi_0 + u32(256);
    }
    device_group_barrier_0();
    bond_ranks_0(parent_4, base_1, c0_2, b0_1, b1_1, tid_1, false);
    device_group_barrier_0();
    var b_3 : u32;
    if(_S18)
    {
        b_3 = u32(0);
        w_4 = u32(0);
        loop
        {
            if(b_3 < _S15)
            {
            }
            else
            {
                break;
            }
            var _S26 : u32 = base_1 + child_by_island_0(parent_4, b_3);
            var count_0 : u32 = plan_0[child_bonds_at_0(_S26)];
            var _S27 : u32 = b0_1 + w_4;
            plan_0[child_bonds_at_0(_S26)] = _S27;
            plan_0[child_bonds_at_0(_S26) + u32(1)] = _S27;
            var at_1 : u32 = w_4 + count_0;
            b_3 = b_3 + u32(1);
            w_4 = at_1;
        }
        var _S28 : u32 = b0_1 + w_4;
        g_cut_first_0 = _S28;
        g_cut_next_0 = _S28;
    }
    device_group_barrier_0();
    bond_ranks_0(parent_4, base_1, c0_2, b0_1, b1_1, tid_1, true);
    b_3 = b0_1 + tid_1;
    loop
    {
        if(b_3 < b1_1)
        {
        }
        else
        {
            break;
        }
        var ids_1 : vec4<u32> = (bitcast<vec4<u32>>((old_bonds_0[bond_word_0(b_3, u32(14))])));
        var _S29 : u32 = ids_1.y;
        var _S30 : u32 = base_1 + _S29 - c0_2;
        var ka_1 : u32 = work_0[_S30];
        var _S31 : u32 = ids_1.z;
        var _S32 : u32 = base_1 + _S31 - c0_2;
        var new_ids_0 : vec4<u32>;
        if(ka_1 == work_0[_S32])
        {
            var _S33 : vec4<u32> = vec4<u32>(ids_1.x, c0_2 + plan_0[new_chunk_at_0(_S30)], c0_2 + plan_0[new_chunk_at_0(_S32)], plan_0[child_island_at_0(base_1 + ka_1)]);
            w_4 = plan_0[bond_new_at_0(b_3)];
            new_ids_0 = _S33;
        }
        else
        {
            var nb_0 : u32 = plan_0[bond_new_at_0(b_3)];
            var _S34 : vec4<u32> = vec4<u32>(ids_1.x, _S29, _S31, u32(4294967295));
            w_4 = u32(0);
            loop
            {
                if(w_4 < u32(3))
                {
                }
                else
                {
                    break;
                }
                scratch_0[u32(3) * nb_0 + w_4] = vec4<f32>(0.0f);
                w_4 = w_4 + u32(1);
            }
            var _S35 : u32 = u32(2) * nb_0;
            var _S36 : vec4<f32> = vec4<f32>(0.0f);
            scratch_0[params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + _S35] = _S36;
            scratch_0[params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + _S35 + u32(1)] = _S36;
            w_4 = nb_0;
            new_ids_0 = _S34;
        }
        w_5 = u32(0);
        loop
        {
            if(w_5 < (params_0.bond_words_0))
            {
            }
            else
            {
                break;
            }
            bonds_0[bond_word_0(w_4, w_5)] = old_bonds_0[bond_word_0(b_3, w_5)];
            w_5 = w_5 + u32(1);
        }
        bonds_0[bond_word_0(w_4, u32(14))] = (bitcast<vec4<f32>>((new_ids_0)));
        w_6 = u32(0);
        loop
        {
            if(w_6 < (params_0.dyn_words_0))
            {
            }
            else
            {
                break;
            }
            dyn_0[params_0.dyn_words_0 * w_4 + w_6] = old_dyn_0[params_0.dyn_words_0 * b_3 + w_6];
            w_6 = w_6 + u32(1);
        }
        b_3 = b_3 + u32(256);
    }
    device_group_barrier_0();
    var _S37 : bool;
    if(g_cut_first_0 < b1_1)
    {
        w_4 = u32(2) * g_cut_first_0;
    }
    else
    {
        var _S38 : u32 = old_index_0[c0_2];
        w_4 = u32(4294967295);
        w_5 = _S38;
        loop
        {
            if(w_5 < old_index_0[c1_1])
            {
                scaled_0 = w_4 == u32(4294967295);
            }
            else
            {
                scaled_0 = false;
            }
            if(scaled_0)
            {
            }
            else
            {
                break;
            }
            var entry_0 : u32 = old_index_0[w_5];
            var bond_0 : u32 = (entry_0 >> (u32(1)));
            if(bond_0 < b0_1)
            {
                _S37 = true;
            }
            else
            {
                _S37 = bond_0 >= b1_1;
            }
            if(_S37)
            {
                w_4 = entry_0;
            }
            w_5 = w_5 + u32(1);
        }
    }
    lo_0 = tid_1;
    loop
    {
        if(lo_0 < _S16)
        {
        }
        else
        {
            break;
        }
        var _S39 : u32 = c0_2 + lo_0;
        plan_0[degree_at_0(base_1 + plan_0[new_chunk_at_0(base_1 + lo_0)])] = old_index_0[_S39 + u32(1)] - old_index_0[_S39];
        lo_0 = lo_0 + u32(256);
    }
    device_group_barrier_0();
    var _S40 : u32 = old_index_0[c0_2];
    w_5 = u32(0);
    w_6 = _S40;
    loop
    {
        if(w_5 < _S16)
        {
        }
        else
        {
            break;
        }
        var i_4 : u32 = w_5 + tid_1;
        var _S41 : bool = i_4 < _S16;
        if(_S41)
        {
            v_2 = plan_0[degree_at_0(base_1 + i_4)];
        }
        else
        {
            v_2 = u32(0);
        }
        g_scan64_0[tid_1] = v_2;
        workgroupBarrier();
        step_0 = u32(1);
        loop
        {
            if(step_0 < u32(256))
            {
            }
            else
            {
                break;
            }
            if(tid_1 >= step_0)
            {
                add_0 = g_scan64_0[tid_1 - step_0];
            }
            else
            {
                add_0 = u32(0);
            }
            workgroupBarrier();
            g_scan64_0[tid_1] = g_scan64_0[tid_1] + add_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        if(_S41)
        {
            index_0[c0_2 + i_4] = w_6 + g_scan64_0[tid_1] - v_2;
        }
        var block_total_0 : u32 = g_scan64_0[u32(255)];
        workgroupBarrier();
        var carry_0 : u32 = w_6 + block_total_0;
        w_5 = w_5 + u32(256);
        w_6 = carry_0;
    }
    device_group_barrier_0();
    lo_0 = tid_1;
    loop
    {
        if(lo_0 < _S16)
        {
        }
        else
        {
            break;
        }
        var _S42 : u32 = index_0[c0_2 + plan_0[new_chunk_at_0(base_1 + lo_0)]];
        var _S43 : u32 = c0_2 + lo_0;
        var _S44 : u32 = _S43 + u32(1);
        var _S45 : u32 = old_index_0[_S44] - old_index_0[_S43];
        v_2 = old_index_0[_S43];
        step_0 = u32(0);
        loop
        {
            if(v_2 < old_index_0[_S44])
            {
            }
            else
            {
                break;
            }
            var entry_1 : u32 = old_index_0[v_2];
            var bond_1 : u32 = (entry_1 >> (u32(1)));
            if(bond_1 < b0_1)
            {
                scaled_0 = true;
            }
            else
            {
                scaled_0 = bond_1 >= b1_1;
            }
            if(scaled_0)
            {
                v_2 = v_2 + u32(1);
                continue;
            }
            var nb_1 : u32 = plan_0[bond_new_at_0(bond_1)];
            if(plan_0[bond_new_at_0(bond_1)] >= g_cut_first_0)
            {
                v_2 = v_2 + u32(1);
                continue;
            }
            index_0[_S42 + step_0] = u32(2) * nb_1 + ((entry_1 & (u32(1))));
            step_0 = step_0 + u32(1);
            v_2 = v_2 + u32(1);
        }
        add_0 = step_0;
        loop
        {
            if(add_0 < _S45)
            {
            }
            else
            {
                break;
            }
            index_0[_S42 + add_0] = w_4;
            add_0 = add_0 + u32(1);
        }
        lo_0 = lo_0 + u32(256);
    }
    device_group_barrier_0();
    hi_0 = tid_1;
    loop
    {
        if(hi_0 < _S15)
        {
        }
        else
        {
            break;
        }
        var slot_7 : u32 = base_1 + hi_0;
        var island_2 : u32 = plan_0[child_island_at_0(slot_7)];
        var _S46 : u32 = u32(8) * slot_7;
        var head_0 : vec4<f32> = children_0[_S46];
        var _S47 : u32 = (bitcast<u32>((head_0.y)));
        var count_1 : u32 = (bitcast<u32>((head_0.z)));
        var anchored_0 : bool = (bitcast<u32>((head_0.w))) != u32(0);
        var com_0 : vec4<f32> = children_0[_S46 + u32(1)];
        var i0_1 : vec3<f32> = children_0[_S46 + u32(2)].xyz;
        var i1_1 : vec3<f32> = children_0[_S46 + u32(3)].xyz;
        var i2_1 : vec3<f32> = children_0[_S46 + u32(4)].xyz;
        var v_3 : vec3<f32> = children_0[_S46 + u32(5)].xyz;
        var w_7 : vec3<f32> = children_0[_S46 + u32(6)].xyz;
        var _S48 : vec3<f32> = vec3<f32>(0.0f);
        v_2 = u32(4294967295);
        scaled_0 = false;
        step_0 = u32(0);
        var wmass_0 : f32 = 0.0f;
        var wmoment_0 : vec3<f32> = _S48;
        var su_mass_0 : f32 = 0.0f;
        var su_0 : vec3<f32> = _S48;
        loop
        {
            if(step_0 < count_1)
            {
            }
            else
            {
                break;
            }
            var m_0 : u32 = work_0[_S17 + base_1 + _S47 + step_0];
            var _S49 : u32 = min(v_2, plan_0[new_chunk_at_0(base_1 + m_0)]);
            var _S50 : u32 = c0_2 + m_0;
            var ctr_0 : vec4<f32> = old_chunks_0[chunk_word_0(_S50, u32(0))];
            var mu_0 : f32 = old_chunks_0[chunk_word_0(_S50, u32(7))].x;
            if(scaled_0)
            {
                scaled_0 = true;
            }
            else
            {
                scaled_0 = mu_0 != 1.0f;
            }
            var _S51 : f32 = ctr_0.w;
            var wmass_1 : f32 = wmass_0 + _S51 * mu_0;
            var _S52 : vec3<f32> = vec3<f32>(_S51);
            var wmoment_1 : vec3<f32> = wmoment_0 + ctr_0.xyz * _S52 * vec3<f32>(mu_0);
            var su_mass_1 : f32 = su_mass_0 + _S51;
            var su_1 : vec3<f32> = su_0 + old_state_0[u32(4) * _S50].xyz * _S52;
            var _S53 : u32 = step_0 + u32(1);
            v_2 = _S49;
            step_0 = _S53;
            wmass_0 = wmass_1;
            wmoment_0 = wmoment_1;
            su_mass_0 = su_mass_1;
            su_0 = su_1;
        }
        var wcom_0 : vec3<f32>;
        if(scaled_0)
        {
            wcom_0 = wmoment_0 / vec3<f32>(wmass_0);
        }
        else
        {
            wcom_0 = com_0.xyz;
        }
        var w0_0 : vec3<f32>;
        var w1_0 : vec3<f32>;
        var w2_0 : vec3<f32>;
        if(scaled_0)
        {
            add_0 = u32(0);
            w0_0 = _S48;
            w1_0 = _S48;
            w2_0 = _S48;
            loop
            {
                if(add_0 < count_1)
                {
                }
                else
                {
                    break;
                }
                var _S54 : u32 = c0_2 + work_0[_S17 + base_1 + _S47 + add_0];
                var ctr_1 : vec4<f32> = old_chunks_0[chunk_word_0(_S54, u32(0))];
                var rr_0 : vec3<f32> = ctr_1.xyz - wcom_0;
                var dd_0 : f32 = dot(rr_0, rr_0);
                var _S55 : vec3<f32> = vec3<f32>(ctr_1.w);
                var _S56 : vec3<f32> = vec3<f32>(old_chunks_0[chunk_word_0(_S54, u32(7))].x);
                var w0_1 : vec3<f32> = w0_0 + (old_chunks_0[chunk_word_0(_S54, u32(1))].xyz + (vec3<f32>(dd_0, 0.0f, 0.0f) - vec3<f32>(rr_0.x) * rr_0) * _S55) * _S56;
                var w1_1 : vec3<f32> = w1_0 + (old_chunks_0[chunk_word_0(_S54, u32(2))].xyz + (vec3<f32>(0.0f, dd_0, 0.0f) - vec3<f32>(rr_0.y) * rr_0) * _S55) * _S56;
                var w2_1 : vec3<f32> = w2_0 + (old_chunks_0[chunk_word_0(_S54, u32(3))].xyz + (vec3<f32>(0.0f, 0.0f, dd_0) - vec3<f32>(rr_0.z) * rr_0) * _S55) * _S56;
                add_0 = add_0 + u32(1);
                w0_0 = w0_1;
                w1_0 = w1_1;
                w2_0 = w2_1;
            }
        }
        else
        {
            w0_0 = i0_1;
            w1_0 = i1_1;
            w2_0 = i2_1;
        }
        var v0_0 : vec3<f32>;
        var v1_0 : vec3<f32>;
        var v2_0 : vec3<f32>;
        inverse3_0(i0_1, i1_1, i2_1, &(v0_0), &(v1_0), &(v2_0));
        var x0_0 : vec3<f32>;
        var x1_0 : vec3<f32>;
        var x2_0 : vec3<f32>;
        inverse3_0(w0_0, w1_0, w2_0, &(x0_0), &(x1_0), &(x2_0));
        var _S57 : u32 = u32(5) * p_1;
        var q_5 : Quat_0 = quat_of_0(frames_0[_S57]);
        var pos_0 : vec3<f32> = frames_0[_S57 + u32(4)].xyz;
        var pos_1 : vec3<f32>;
        if(!anchored_0)
        {
            pos_1 = pos_0 + rotate_0(q_5, su_0 / vec3<f32>(su_mass_0));
        }
        else
        {
            pos_1 = pos_0;
        }
        var _S58 : vec3<f32> = inverse_rotate_0(q_5, w_7);
        var iw_0 : vec3<f32> = rotate_0(q_5, vec3<f32>(dot(i0_1, _S58), dot(i1_1, _S58), dot(i2_1, _S58)));
        if(anchored_0)
        {
            i_3 = u32(1);
        }
        else
        {
            i_3 = u32(0);
        }
        if((params_0.driven_0) != u32(0))
        {
            lo_0 = u32(2);
        }
        else
        {
            lo_0 = u32(0);
        }
        var flags_0 : u32 = (((i_3 | (lo_0))) | (u32(16)));
        var bfirst_0 : u32 = plan_0[child_bonds_at_0(slot_7)];
        var bend_0 : u32 = plan_0[child_bonds_at_0(slot_7) + u32(1)];
        var rec_0 : array<vec4<f32>, i32(25)>;
        add_0 = u32(0);
        loop
        {
            if(add_0 < u32(25))
            {
            }
            else
            {
                break;
            }
            rec_0[add_0] = vec4<f32>(0.0f);
            add_0 = add_0 + u32(1);
        }
        var _S59 : u32 = c0_2 + v_2;
        rec_0[i32(0)] = (bitcast<vec4<f32>>((vec4<u32>(_S59, _S59 + count_1, bfirst_0, bend_0))));
        rec_0[i32(1)] = (bitcast<vec4<f32>>((vec4<u32>(flags_0, u32(0), u32(0), params_0.substeps_0))));
        rec_0[i32(2)] = com_0;
        rec_0[i32(3)] = vec4<f32>(i0_1, 0.0f);
        rec_0[i32(4)] = vec4<f32>(i1_1, 0.0f);
        rec_0[i32(5)] = vec4<f32>(i2_1, 0.0f);
        rec_0[i32(6)] = vec4<f32>(v0_0, 0.0f);
        rec_0[i32(7)] = vec4<f32>(v1_0, 0.0f);
        rec_0[i32(8)] = vec4<f32>(v2_0, 0.0f);
        var _S60 : f32;
        if(scaled_0)
        {
            _S60 = wmass_0;
        }
        else
        {
            _S60 = com_0.w;
        }
        rec_0[i32(9)] = vec4<f32>(wcom_0, _S60);
        rec_0[i32(10)] = vec4<f32>(x0_0, 0.0f);
        rec_0[i32(11)] = vec4<f32>(x1_0, 0.0f);
        rec_0[i32(12)] = vec4<f32>(x2_0, 0.0f);
        rec_0[i32(13)] = quat_vec_0(q_5);
        rec_0[i32(14)] = vec4<f32>(pos_1, 0.0f);
        var _S61 : vec3<f32>;
        if(anchored_0)
        {
            _S61 = _S48;
        }
        else
        {
            _S61 = v_3;
        }
        rec_0[i32(16)] = vec4<f32>(_S61, 0.0f);
        var _S62 : vec3<f32>;
        if(anchored_0)
        {
            _S62 = _S48;
        }
        else
        {
            _S62 = w_7;
        }
        rec_0[i32(18)] = vec4<f32>(_S62, 0.0f);
        rec_0[i32(19)] = (bitcast<vec4<f32>>((vec4<u32>(u32(0), u32(0), u32(0), plan_0[child_id_at_0(slot_7)]))));
        rec_0[i32(21)] = vec4<f32>(0.0f, 0.0f, 0.0f, float_bits_0(u32(2139095040)));
        var _S63 : vec3<f32>;
        if(anchored_0)
        {
            _S63 = _S48;
        }
        else
        {
            _S63 = iw_0;
        }
        rec_0[i32(23)] = vec4<f32>(_S63, 0.0f);
        var t_2 : u32 = u32(0);
        loop
        {
            if(t_2 < u32(25))
            {
            }
            else
            {
                break;
            }
            islands_0[island_word_0(island_2, t_2)] = rec_0[t_2];
            t_2 = t_2 + u32(1);
        }
        if((params_0.resident_0) != u32(0))
        {
            var sl_0 : vec4<u32> = spawn_slots_0[slot_7];
            var _S64 : u32 = u32(2) * slot_7;
            var pr_0 : vec4<f32> = spawn_extras_0[_S64 + u32(1)];
            var at_2 : u32 = params_0.engine_base_0 + u32(3) * island_2;
            var _S65 : u32 = sl_0.w;
            var body_range_0 : vec3<f32>;
            if(_S65 != u32(0))
            {
                var _S66 : u32 = sl_0.y;
                body_range_0 = (bitcast<vec3<f32>>((vec3<u32>(sl_0.x, _S66, _S66 + sl_0.z))));
            }
            else
            {
                body_range_0 = vec3<f32>(float_bits_0(u32(4294967295)), 0.0f, 0.0f);
            }
            if(anchored_0)
            {
                _S37 = true;
            }
            else
            {
                _S37 = _S65 == u32(0);
            }
            if(_S37)
            {
                ctab_0[at_2] = vec4<f32>(float_bits_0(u32(4294967295)), body_range_0);
                var _S67 : vec4<f32> = vec4<f32>(0.0f);
                ctab_0[at_2 + u32(1)] = _S67;
                ctab_0[at_2 + u32(2)] = _S67;
            }
            else
            {
                ctab_0[at_2] = vec4<f32>((bitcast<f32>((sl_0.x))), body_range_0);
                ctab_0[at_2 + u32(1)] = vec4<f32>(- pr_0.x, - pr_0.y, - pr_0.z, pr_0.w);
                ctab_0[at_2 + u32(2)] = vec4<f32>(spawn_extras_0[_S64].xyz, 0.0f);
            }
        }
        gate_0[params_0.gate_flag_base_0 + island_2] = vec4<f32>(0.0f);
        hi_0 = hi_0 + u32(256);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn layout_moved(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S68 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S68 = true;
    }
    else
    {
        _S68 = control_0[u32(1)] == u32(0);
    }
    if(_S68)
    {
        return;
    }
    var i_5 : u32 = id_1.x;
    var t_3 : u32;
    if(i_5 == plan_0[i32(0)])
    {
        t_3 = u32(0);
        loop
        {
            if(t_3 < (params_0.island_words_0))
            {
            }
            else
            {
                break;
            }
            islands_0[island_word_0(i_5, t_3)] = vec4<f32>(0.0f);
            t_3 = t_3 + u32(1);
        }
        return;
    }
    if(i_5 >= plan_0[i32(0)])
    {
        return;
    }
    var e_1 : u32 = plan_0[order_at_0(i_5)];
    if(plan_0[order_at_0(i_5)] >= control_0[u32(2)])
    {
        _S68 = true;
    }
    else
    {
        _S68 = e_1 == i_5;
    }
    if(_S68)
    {
        return;
    }
    t_3 = u32(0);
    loop
    {
        if(t_3 < (params_0.island_words_0))
        {
        }
        else
        {
            break;
        }
        islands_0[island_word_0(i_5, t_3)] = old_islands_0[island_word_0(e_1, t_3)];
        t_3 = t_3 + u32(1);
    }
    var r_3 : vec4<f32> = old_islands_0[island_word_0(e_1, u32(0))];
    var _S69 : u32 = (bitcast<u32>((r_3.y)));
    var b0_2 : u32 = (bitcast<u32>((r_3.z)));
    var _S70 : u32 = (bitcast<u32>((r_3.w)));
    var c_3 : u32 = (bitcast<u32>((r_3.x)));
    loop
    {
        if(c_3 < _S69)
        {
        }
        else
        {
            break;
        }
        set_chunk_island_0(c_3, i_5);
        c_3 = c_3 + u32(1);
    }
    var b_4 : u32 = b0_2;
    loop
    {
        if(b_4 < _S70)
        {
        }
        else
        {
            break;
        }
        var _S71 : u32 = bond_word_0(b_4, u32(14));
        var ids_2 : vec4<f32> = bonds_0[_S71];
        ids_2[i32(3)] = (bitcast<f32>((i_5)));
        bonds_0[_S71] = ids_2;
        b_4 = b_4 + u32(1);
    }
    if((params_0.resident_0) != u32(0))
    {
        t_3 = u32(0);
        loop
        {
            if(t_3 < u32(3))
            {
            }
            else
            {
                break;
            }
            ctab_0[params_0.engine_base_0 + u32(3) * i_5 + t_3] = old_ctab_0[params_0.engine_base_0 + u32(3) * e_1 + t_3];
            t_3 = t_3 + u32(1);
        }
    }
    gate_0[params_0.gate_flag_base_0 + i_5] = gate_0[params_0.gate_flag_base_0 + e_1];
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn layout_copy(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var _S72 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S72 = true;
    }
    else
    {
        _S72 = control_0[u32(1)] == u32(0);
    }
    if(_S72)
    {
        return;
    }
    var i_6 : u32 = id_2.x;
    var n_1 : u32 = plan_0[i32(0)];
    if(i_6 > plan_0[i32(0)])
    {
        return;
    }
    var t_4 : u32;
    if(i_6 == n_1)
    {
        t_4 = u32(0);
        loop
        {
            if(t_4 < (params_0.island_words_0))
            {
            }
            else
            {
                break;
            }
            island_copy_0[island_word_0(i_6, t_4)] = vec4<f32>(0.0f);
            t_4 = t_4 + u32(1);
        }
        return;
    }
    var e_2 : u32 = plan_0[order_at_0(i_6)];
    if(plan_0[order_at_0(i_6)] < control_0[u32(2)])
    {
        t_4 = u32(0);
        loop
        {
            if(t_4 < (params_0.island_words_0))
            {
            }
            else
            {
                break;
            }
            island_copy_0[island_word_0(i_6, t_4)] = old_copy_0[island_word_0(e_2, t_4)];
            t_4 = t_4 + u32(1);
        }
    }
    else
    {
        t_4 = u32(0);
        loop
        {
            if(t_4 < (params_0.island_words_0))
            {
            }
            else
            {
                break;
            }
            var _S73 : u32 = island_word_0(i_6, t_4);
            island_copy_0[_S73] = islands_0[_S73];
            t_4 = t_4 + u32(1);
        }
    }
    return;
}

