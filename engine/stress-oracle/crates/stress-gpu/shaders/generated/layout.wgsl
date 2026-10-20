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

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_0 : u32,  thread_0 : u32) -> u32
{
    if(thread_0 == u32(0))
    {
        g_group_uniform_0 = value_0;
    }
    var _S7 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
    return _S7;
}

fn island_word_0( i_2 : u32,  w_0 : u32) -> u32
{
    return params_0.island_words_0 * i_2 + w_0;
}

fn layout_start_0( p_1 : u32,  tid_0 : u32,  parent_1 : ptr<function, vec4<u32>>,  c0_0 : ptr<function, u32>,  c1_0 : ptr<function, u32>,  b0_0 : ptr<function, u32>,  b1_0 : ptr<function, u32>) -> bool
{
    (*parent_1) = vec4<u32>(vec4<i32>(i32(0)));
    (*c0_0) = u32(0);
    (*c1_0) = u32(0);
    (*b0_0) = u32(0);
    (*b1_0) = u32(0);
    var _S8 : bool;
    if(p_1 >= control_0[u32(1)])
    {
        _S8 = true;
    }
    else
    {
        _S8 = control_0[i32(0)] != u32(0);
    }
    var _S9 : u32;
    if(_S8)
    {
        _S9 = u32(1);
    }
    else
    {
        _S9 = u32(0);
    }
    var _S10 : u32 = group_uniform_0(_S9, tid_0);
    if(_S10 != u32(0))
    {
        return false;
    }
    var _S11 : vec4<u32> = parents_0[p_1];
    (*parent_1) = _S11;
    if((_S11.z) < u32(2))
    {
        _S9 = u32(1);
    }
    else
    {
        _S9 = u32(0);
    }
    var _S12 : u32 = group_uniform_0(_S9, tid_0);
    if(_S12 != u32(0))
    {
        return false;
    }
    var r_0 : vec4<f32> = old_islands_0[island_word_0((*parent_1).x, u32(0))];
    var _S13 : u32 = group_uniform_0((bitcast<u32>((r_0.x))), tid_0);
    (*c0_0) = _S13;
    var _S14 : u32 = group_uniform_0((bitcast<u32>((r_0.y))), tid_0);
    (*c1_0) = _S14;
    var _S15 : u32 = group_uniform_0((bitcast<u32>((r_0.z))), tid_0);
    (*b0_0) = _S15;
    var _S16 : u32 = group_uniform_0((bitcast<u32>((r_0.w))), tid_0);
    (*b1_0) = _S16;
    return true;
}

fn child_order_at_0( slot_2 : u32) -> u32
{
    return u32(16) + cap_0() + u32(5) * control_0[u32(6)] + control_0[u32(2)] + params_0.bond_count_0 + slot_2;
}

fn sort_children_0( parent_2 : vec4<u32>)
{
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < (parent_2.z))
        {
        }
        else
        {
            break;
        }
        var _S17 : u32 = parent_2.y;
        var _S18 : u32 = plan_0[child_island_at_0(_S17 + k_1)];
        var j_1 : u32 = k_1;
        loop
        {
            var _S19 : bool;
            if(j_1 > u32(0))
            {
                _S19 = plan_0[child_island_at_0(_S17 + plan_0[child_order_at_0(_S17 + j_1 - u32(1))])] > _S18;
            }
            else
            {
                _S19 = false;
            }
            if(_S19)
            {
            }
            else
            {
                break;
            }
            var _S20 : u32 = _S17 + j_1;
            plan_0[child_order_at_0(_S20)] = plan_0[child_order_at_0(_S20 - u32(1))];
            j_1 = j_1 - u32(1);
        }
        plan_0[child_order_at_0(_S17 + j_1)] = k_1;
        k_1 = k_1 + u32(1);
    }
    return;
}

fn child_by_island_0( parent_3 : vec4<u32>,  r_1 : u32) -> u32
{
    return plan_0[child_order_at_0(parent_3.y + r_1)];
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

fn lay_positions_0( parent_4 : vec4<u32>,  n_1 : u32,  base_0 : u32,  total_1 : u32,  tid_1 : u32)
{
    var i_3 : u32;
    var lo_0 : u32;
    if(tid_1 == u32(0))
    {
        sort_children_0(parent_4);
        i_3 = u32(0);
        lo_0 = u32(0);
        loop
        {
            if(i_3 < (parent_4.z))
            {
            }
            else
            {
                break;
            }
            var _S21 : u32 = base_0 + child_by_island_0(parent_4, i_3);
            plan_0[child_start_at_0(_S21)] = lo_0;
            var at_0 : u32 = lo_0 + (bitcast<u32>((children_0[u32(8) * _S21].z)));
            i_3 = i_3 + u32(1);
            lo_0 = at_0;
        }
    }
    device_group_barrier_0();
    i_3 = tid_1;
    loop
    {
        if(i_3 < n_1)
        {
        }
        else
        {
            break;
        }
        var _S22 : u32 = parent_4.z - u32(1);
        lo_0 = u32(0);
        var hi_0 : u32 = _S22;
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
            if((bitcast<u32>((children_0[u32(8) * (base_0 + mid_0)].y))) <= i_3)
            {
                lo_0 = mid_0;
            }
            else
            {
                hi_0 = mid_0 - u32(1);
            }
        }
        var _S23 : u32 = base_0 + lo_0;
        plan_0[new_chunk_at_0(base_0 + work_0[total_1 + base_0 + i_3])] = plan_0[child_start_at_0(_S23)] + (i_3 - (bitcast<u32>((children_0[u32(8) * _S23].y))));
        i_3 = i_3 + u32(256);
    }
    return;
}

fn chunk_word_0( c_0 : u32,  w_1 : u32) -> u32
{
    return params_0.chunk_words_0 * c_0 + w_1;
}

fn set_chunk_island_0( c_1 : u32,  island_0 : u32)
{
    var _S24 : u32 = chunk_word_0(c_1, u32(8));
    var info_0 : vec4<f32> = chunks_0[_S24];
    info_0[i32(1)] = (bitcast<f32>((island_0)));
    chunks_0[_S24] = info_0;
    return;
}

fn lay_chunks_0( c0_1 : u32,  n_2 : u32,  base_1 : u32,  tid_2 : u32)
{
    var m_0 : u32 = tid_2;
    loop
    {
        if(m_0 < n_2)
        {
        }
        else
        {
            break;
        }
        var _S25 : u32 = base_1 + m_0;
        var nk_0 : u32 = plan_0[new_chunk_at_0(_S25)];
        var island_1 : u32 = plan_0[child_island_at_0(base_1 + work_0[_S25])];
        var w_2 : u32 = u32(0);
        loop
        {
            if(w_2 < (params_0.chunk_words_0))
            {
            }
            else
            {
                break;
            }
            chunks_0[chunk_word_0(c0_1 + nk_0, w_2)] = old_chunks_0[chunk_word_0(c0_1 + m_0, w_2)];
            w_2 = w_2 + u32(1);
        }
        var _S26 : u32 = c0_1 + nk_0;
        set_chunk_island_0(_S26, island_1);
        var w_3 : u32 = u32(0);
        loop
        {
            if(w_3 < u32(4))
            {
            }
            else
            {
                break;
            }
            state_0[u32(4) * _S26 + w_3] = old_state_0[u32(4) * (c0_1 + m_0) + w_3];
            w_3 = w_3 + u32(1);
        }
        var w_4 : u32 = u32(0);
        loop
        {
            if(w_4 < u32(23))
            {
            }
            else
            {
                break;
            }
            scratch_0[params_0.statics_base_0 + u32(23) * _S26 + w_4] = vec4<f32>(0.0f);
            w_4 = w_4 + u32(1);
        }
        m_0 = m_0 + u32(256);
    }
    return;
}

fn lay_ctab_chunks_0( c0_2 : u32,  n_3 : u32,  base_2 : u32,  tid_3 : u32)
{
    var m_1 : u32 = tid_3;
    loop
    {
        if(m_1 < n_3)
        {
        }
        else
        {
            break;
        }
        var o_0 : u32 = c0_2 + m_1;
        var d_0 : u32 = c0_2 + plan_0[new_chunk_at_0(base_2 + m_1)];
        ctab_0[params_0.props_base_0 + d_0] = old_ctab_0[params_0.props_base_0 + o_0];
        var _S27 : u32 = u32(2) * d_0;
        var _S28 : u32 = u32(2) * o_0;
        ctab_0[params_0.eload_base_0 + _S27] = old_ctab_0[params_0.eload_base_0 + _S28];
        ctab_0[params_0.eload_base_0 + _S27 + u32(1)] = old_ctab_0[params_0.eload_base_0 + _S28 + u32(1)];
        ctab_0[params_0.pulse_count_base_0 + d_0] = old_ctab_0[params_0.pulse_count_base_0 + o_0];
        var w_5 : u32 = u32(0);
        loop
        {
            if(w_5 < (u32(3) * params_0.pulse_cap_0))
            {
            }
            else
            {
                break;
            }
            ctab_0[params_0.pulse_base_0 + u32(3) * params_0.pulse_cap_0 * d_0 + w_5] = old_ctab_0[params_0.pulse_base_0 + u32(3) * params_0.pulse_cap_0 * o_0 + w_5];
            w_5 = w_5 + u32(1);
        }
        m_1 = m_1 + u32(256);
    }
    return;
}

fn child_bonds_at_0( slot_5 : u32) -> u32
{
    return u32(16) + cap_0() + u32(3) * control_0[u32(6)] + control_0[u32(2)] + u32(2) * slot_5;
}

fn bond_word_0( b_0 : u32,  w_6 : u32) -> u32
{
    return params_0.bond_words_0 * b_0 + w_6;
}

var<workgroup> g_class_0 : array<u32, i32(256)>;

var<workgroup> g_cut_next_0 : u32;

fn bond_new_at_0( b_1 : u32) -> u32
{
    return u32(16) + cap_0() + u32(5) * control_0[u32(6)] + control_0[u32(2)] + b_1;
}

fn bond_ranks_0( parent_5 : vec4<u32>,  base_3 : u32,  c0_3 : u32,  b0_1 : u32,  b1_1 : u32,  tid_4 : u32,  place_0 : bool)
{
    var _S29 : bool;
    var block_0 : u32 = b0_1;
    loop
    {
        if(block_0 < b1_1)
        {
        }
        else
        {
            break;
        }
        var b_2 : u32 = block_0 + tid_4;
        var cls_0 : u32;
        if(b_2 < b1_1)
        {
            var ids_0 : vec4<u32> = (bitcast<vec4<u32>>((old_bonds_0[bond_word_0(b_2, u32(14))])));
            var ka_0 : u32 = work_0[base_3 + ids_0.y - c0_3];
            if(ka_0 == work_0[base_3 + ids_0.z - c0_3])
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
        g_class_0[tid_4] = cls_0;
        workgroupBarrier();
        var _S30 : bool = cls_0 != u32(4294967295);
        var last_0 : bool = _S30;
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
            if(t_0 < tid_4)
            {
                _S29 = (g_class_0[t_0]) == cls_0;
            }
            else
            {
                _S29 = false;
            }
            if(_S29)
            {
                rank_0 = rank_0 + u32(1);
            }
            var _S31 : bool;
            if(t_0 > tid_4)
            {
                _S31 = (g_class_0[t_0]) == cls_0;
            }
            else
            {
                _S31 = false;
            }
            if(_S31)
            {
                last_0 = false;
            }
            t_0 = t_0 + u32(1);
        }
        if(place_0)
        {
            _S29 = _S30;
        }
        else
        {
            _S29 = false;
        }
        if(_S29)
        {
            var first_0 : u32;
            if(cls_0 == u32(4294967294))
            {
                first_0 = g_cut_next_0;
            }
            else
            {
                first_0 = plan_0[child_bonds_at_0(base_3 + cls_0) + u32(1)];
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
                    plan_0[child_bonds_at_0(base_3 + cls_0) + u32(1)] = plan_0[child_bonds_at_0(base_3 + cls_0) + u32(1)] + (rank_0 + u32(1));
                }
            }
            else
            {
                if(cls_0 != u32(4294967294))
                {
                    plan_0[child_bonds_at_0(base_3 + cls_0)] = plan_0[child_bonds_at_0(base_3 + cls_0)] + (rank_0 + u32(1));
                }
            }
        }
        device_group_barrier_0();
        block_0 = block_0 + u32(256);
    }
    return;
}

var<workgroup> g_cut_first_0 : u32;

fn lay_bond_ranks_0( parent_6 : vec4<u32>,  c0_4 : u32,  b0_2 : u32,  b1_2 : u32,  base_4 : u32,  tid_5 : u32)
{
    var _S32 : u32;
    var k_2 : u32 = tid_5;
    loop
    {
        var _S33 : u32 = parent_6.z;
        _S32 = _S33;
        if(k_2 < _S33)
        {
        }
        else
        {
            break;
        }
        plan_0[child_bonds_at_0(base_4 + k_2)] = u32(0);
        k_2 = k_2 + u32(256);
    }
    device_group_barrier_0();
    bond_ranks_0(parent_6, base_4, c0_4, b0_2, b1_2, tid_5, false);
    device_group_barrier_0();
    if(tid_5 == u32(0))
    {
        var r_2 : u32 = u32(0);
        var at_1 : u32 = u32(0);
        loop
        {
            if(r_2 < _S32)
            {
            }
            else
            {
                break;
            }
            var _S34 : u32 = base_4 + child_by_island_0(parent_6, r_2);
            var count_0 : u32 = plan_0[child_bonds_at_0(_S34)];
            var _S35 : u32 = b0_2 + at_1;
            plan_0[child_bonds_at_0(_S34)] = _S35;
            plan_0[child_bonds_at_0(_S34) + u32(1)] = _S35;
            var at_2 : u32 = at_1 + count_0;
            r_2 = r_2 + u32(1);
            at_1 = at_2;
        }
        var _S36 : u32 = b0_2 + at_1;
        g_cut_first_0 = _S36;
        g_cut_next_0 = _S36;
    }
    device_group_barrier_0();
    bond_ranks_0(parent_6, base_4, c0_4, b0_2, b1_2, tid_5, true);
    return;
}

fn lay_bonds_0( c0_5 : u32,  b0_3 : u32,  b1_3 : u32,  base_5 : u32,  tid_6 : u32)
{
    var b_3 : u32 = b0_3 + tid_6;
    loop
    {
        if(b_3 < b1_3)
        {
        }
        else
        {
            break;
        }
        var ids_1 : vec4<u32> = (bitcast<vec4<u32>>((old_bonds_0[bond_word_0(b_3, u32(14))])));
        var _S37 : u32 = ids_1.y;
        var _S38 : u32 = base_5 + _S37 - c0_5;
        var ka_1 : u32 = work_0[_S38];
        var _S39 : u32 = ids_1.z;
        var _S40 : u32 = base_5 + _S39 - c0_5;
        var nb_0 : u32;
        var new_ids_0 : vec4<u32>;
        if(ka_1 == work_0[_S40])
        {
            var _S41 : vec4<u32> = vec4<u32>(ids_1.x, c0_5 + plan_0[new_chunk_at_0(_S38)], c0_5 + plan_0[new_chunk_at_0(_S40)], plan_0[child_island_at_0(base_5 + ka_1)]);
            nb_0 = plan_0[bond_new_at_0(b_3)];
            new_ids_0 = _S41;
        }
        else
        {
            var nb_1 : u32 = plan_0[bond_new_at_0(b_3)];
            var _S42 : vec4<u32> = vec4<u32>(ids_1.x, _S37, _S39, u32(4294967295));
            nb_0 = u32(0);
            loop
            {
                if(nb_0 < u32(3))
                {
                }
                else
                {
                    break;
                }
                scratch_0[u32(3) * nb_1 + nb_0] = vec4<f32>(0.0f);
                nb_0 = nb_0 + u32(1);
            }
            var _S43 : u32 = u32(2) * nb_1;
            var _S44 : vec4<f32> = vec4<f32>(0.0f);
            scratch_0[params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + _S43] = _S44;
            scratch_0[params_0.statics_base_0 + u32(23) * params_0.chunk_count_0 + _S43 + u32(1)] = _S44;
            nb_0 = nb_1;
            new_ids_0 = _S42;
        }
        var w_7 : u32 = u32(0);
        loop
        {
            if(w_7 < (params_0.bond_words_0))
            {
            }
            else
            {
                break;
            }
            bonds_0[bond_word_0(nb_0, w_7)] = old_bonds_0[bond_word_0(b_3, w_7)];
            w_7 = w_7 + u32(1);
        }
        bonds_0[bond_word_0(nb_0, u32(14))] = (bitcast<vec4<f32>>((new_ids_0)));
        var w_8 : u32 = u32(0);
        loop
        {
            if(w_8 < (params_0.dyn_words_0))
            {
            }
            else
            {
                break;
            }
            dyn_0[params_0.dyn_words_0 * nb_0 + w_8] = old_dyn_0[params_0.dyn_words_0 * b_3 + w_8];
            w_8 = w_8 + u32(1);
        }
        b_3 = b_3 + u32(256);
    }
    return;
}

fn degree_at_0( slot_6 : u32) -> u32
{
    return u32(16) + cap_0() + u32(7) * control_0[u32(6)] + control_0[u32(2)] + params_0.bond_count_0 + slot_6;
}

var<workgroup> g_scan64_0 : array<u32, i32(256)>;

fn lay_incidence_0( c0_6 : u32,  c1_1 : u32,  b0_4 : u32,  b1_4 : u32,  n_4 : u32,  base_6 : u32,  cut_first_0 : u32,  tid_7 : u32)
{
    var v_0 : u32;
    var step_0 : u32;
    var add_0 : u32;
    var pad_0 : u32;
    var m_2 : u32;
    var _S45 : bool;
    if(cut_first_0 < b1_4)
    {
        pad_0 = u32(2) * cut_first_0;
    }
    else
    {
        var _S46 : u32 = old_index_0[c0_6];
        pad_0 = u32(4294967295);
        m_2 = _S46;
        loop
        {
            if(m_2 < old_index_0[c1_1])
            {
                _S45 = pad_0 == u32(4294967295);
            }
            else
            {
                _S45 = false;
            }
            if(_S45)
            {
            }
            else
            {
                break;
            }
            var entry_0 : u32 = old_index_0[m_2];
            var bond_0 : u32 = (entry_0 >> (u32(1)));
            var _S47 : bool;
            if(bond_0 < b0_4)
            {
                _S47 = true;
            }
            else
            {
                _S47 = bond_0 >= b1_4;
            }
            if(_S47)
            {
                pad_0 = entry_0;
            }
            m_2 = m_2 + u32(1);
        }
    }
    m_2 = tid_7;
    loop
    {
        if(m_2 < n_4)
        {
        }
        else
        {
            break;
        }
        var _S48 : u32 = c0_6 + m_2;
        plan_0[degree_at_0(base_6 + plan_0[new_chunk_at_0(base_6 + m_2)])] = old_index_0[_S48 + u32(1)] - old_index_0[_S48];
        m_2 = m_2 + u32(256);
    }
    device_group_barrier_0();
    var _S49 : u32 = old_index_0[c0_6];
    var block_1 : u32 = u32(0);
    var carry_0 : u32 = _S49;
    loop
    {
        if(block_1 < n_4)
        {
        }
        else
        {
            break;
        }
        var i_4 : u32 = block_1 + tid_7;
        var _S50 : bool = i_4 < n_4;
        if(_S50)
        {
            v_0 = plan_0[degree_at_0(base_6 + i_4)];
        }
        else
        {
            v_0 = u32(0);
        }
        g_scan64_0[tid_7] = v_0;
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
            if(tid_7 >= step_0)
            {
                add_0 = g_scan64_0[tid_7 - step_0];
            }
            else
            {
                add_0 = u32(0);
            }
            workgroupBarrier();
            g_scan64_0[tid_7] = g_scan64_0[tid_7] + add_0;
            workgroupBarrier();
            step_0 = (step_0 << (u32(1)));
        }
        if(_S50)
        {
            index_0[c0_6 + i_4] = carry_0 + g_scan64_0[tid_7] - v_0;
        }
        var block_total_0 : u32 = g_scan64_0[u32(255)];
        workgroupBarrier();
        var carry_1 : u32 = carry_0 + block_total_0;
        block_1 = block_1 + u32(256);
        carry_0 = carry_1;
    }
    device_group_barrier_0();
    m_2 = tid_7;
    loop
    {
        if(m_2 < n_4)
        {
        }
        else
        {
            break;
        }
        var _S51 : u32 = index_0[c0_6 + plan_0[new_chunk_at_0(base_6 + m_2)]];
        var _S52 : u32 = c0_6 + m_2;
        var _S53 : u32 = _S52 + u32(1);
        var _S54 : u32 = old_index_0[_S53] - old_index_0[_S52];
        v_0 = old_index_0[_S52];
        step_0 = u32(0);
        loop
        {
            if(v_0 < old_index_0[_S53])
            {
            }
            else
            {
                break;
            }
            var entry_1 : u32 = old_index_0[v_0];
            var bond_1 : u32 = (entry_1 >> (u32(1)));
            if(bond_1 < b0_4)
            {
                _S45 = true;
            }
            else
            {
                _S45 = bond_1 >= b1_4;
            }
            if(_S45)
            {
                v_0 = v_0 + u32(1);
                continue;
            }
            var nb_2 : u32 = plan_0[bond_new_at_0(bond_1)];
            if(plan_0[bond_new_at_0(bond_1)] >= cut_first_0)
            {
                v_0 = v_0 + u32(1);
                continue;
            }
            index_0[_S51 + step_0] = u32(2) * nb_2 + ((entry_1 & (u32(1))));
            step_0 = step_0 + u32(1);
            v_0 = v_0 + u32(1);
        }
        add_0 = step_0;
        loop
        {
            if(add_0 < _S54)
            {
            }
            else
            {
                break;
            }
            index_0[_S51 + add_0] = pad_0;
            add_0 = add_0 + u32(1);
        }
        m_2 = m_2 + u32(256);
    }
    return;
}

fn inverse3_0( r0_0 : vec3<f32>,  r1_0 : vec3<f32>,  r2_0 : vec3<f32>,  i0_0 : ptr<function, vec3<f32>>,  i1_0 : ptr<function, vec3<f32>>,  i2_0 : ptr<function, vec3<f32>>)
{
    var c0_7 : vec3<f32> = cross(r1_0, r2_0);
    var c1_2 : vec3<f32> = cross(r2_0, r0_0);
    var c2_0 : vec3<f32> = cross(r0_0, r1_0);
    var det_0 : f32 = dot(r0_0, c0_7);
    var inv_0 : f32;
    if(det_0 != 0.0f)
    {
        inv_0 = 1.0f / det_0;
    }
    else
    {
        inv_0 = 0.0f;
    }
    (*i0_0) = vec3<f32>(c0_7.x, c1_2.x, c2_0.x) * vec3<f32>(inv_0);
    (*i1_0) = vec3<f32>(c0_7.y, c1_2.y, c2_0.y) * vec3<f32>(inv_0);
    (*i2_0) = vec3<f32>(c0_7.z, c1_2.z, c2_0.z) * vec3<f32>(inv_0);
    return;
}

struct Quat_0
{
     w_9 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_1 : vec4<f32>) -> Quat_0
{
    var r_3 : Quat_0;
    r_3.x_0 = q_1.x;
    r_3.y_0 = q_1.y;
    r_3.z_0 = q_1.z;
    r_3.w_9 = q_1.w;
    return r_3;
}

fn rotate_0( q_2 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_2.x_0, q_2.y_0, q_2.z_0);
    var t_1 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_1 * vec3<f32>(q_2.w_9) + cross(qv_0, t_1);
}

fn inverse_rotate_0( q_3 : Quat_0,  v_2 : vec3<f32>) -> vec3<f32>
{
    var c_2 : Quat_0;
    c_2.w_9 = q_3.w_9;
    c_2.x_0 = - q_3.x_0;
    c_2.y_0 = - q_3.y_0;
    c_2.z_0 = - q_3.z_0;
    return rotate_0(c_2, v_2);
}

fn quat_vec_0( q_4 : Quat_0) -> vec4<f32>
{
    return vec4<f32>(q_4.x_0, q_4.y_0, q_4.z_0, q_4.w_9);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

fn lay_record_0( p_2 : u32,  c0_8 : u32,  base_7 : u32,  total_2 : u32,  slot_7 : u32)
{
    var _S55 : u32 = plan_0[child_island_at_0(slot_7)];
    var _S56 : u32 = u32(8) * slot_7;
    var head_0 : vec4<f32> = children_0[_S56];
    var _S57 : u32 = (bitcast<u32>((head_0.y)));
    var count_1 : u32 = (bitcast<u32>((head_0.z)));
    var anchored_0 : bool = (bitcast<u32>((head_0.w))) != u32(0);
    var com_0 : vec4<f32> = children_0[_S56 + u32(1)];
    var i0_1 : vec3<f32> = children_0[_S56 + u32(2)].xyz;
    var i1_1 : vec3<f32> = children_0[_S56 + u32(3)].xyz;
    var i2_1 : vec3<f32> = children_0[_S56 + u32(4)].xyz;
    var v_3 : vec3<f32> = children_0[_S56 + u32(5)].xyz;
    var w_10 : vec3<f32> = children_0[_S56 + u32(6)].xyz;
    var _S58 : vec3<f32> = vec3<f32>(0.0f);
    var first_new_0 : u32 = u32(4294967295);
    var scaled_0 : bool = false;
    var j_2 : u32 = u32(0);
    var wmass_0 : f32 = 0.0f;
    var wmoment_0 : vec3<f32> = _S58;
    var su_mass_0 : f32 = 0.0f;
    var su_0 : vec3<f32> = _S58;
    loop
    {
        if(j_2 < count_1)
        {
        }
        else
        {
            break;
        }
        var m_3 : u32 = work_0[total_2 + base_7 + _S57 + j_2];
        var _S59 : u32 = min(first_new_0, plan_0[new_chunk_at_0(base_7 + m_3)]);
        var _S60 : u32 = c0_8 + m_3;
        var ctr_0 : vec4<f32> = old_chunks_0[chunk_word_0(_S60, u32(0))];
        var mu_0 : f32 = old_chunks_0[chunk_word_0(_S60, u32(7))].x;
        if(scaled_0)
        {
            scaled_0 = true;
        }
        else
        {
            scaled_0 = mu_0 != 1.0f;
        }
        var _S61 : f32 = ctr_0.w;
        var wmass_1 : f32 = wmass_0 + _S61 * mu_0;
        var _S62 : vec3<f32> = vec3<f32>(_S61);
        var wmoment_1 : vec3<f32> = wmoment_0 + ctr_0.xyz * _S62 * vec3<f32>(mu_0);
        var su_mass_1 : f32 = su_mass_0 + _S61;
        var su_1 : vec3<f32> = su_0 + old_state_0[u32(4) * _S60].xyz * _S62;
        var _S63 : u32 = j_2 + u32(1);
        first_new_0 = _S59;
        j_2 = _S63;
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
        j_2 = u32(0);
        w0_0 = _S58;
        w1_0 = _S58;
        w2_0 = _S58;
        loop
        {
            if(j_2 < count_1)
            {
            }
            else
            {
                break;
            }
            var _S64 : u32 = c0_8 + work_0[total_2 + base_7 + _S57 + j_2];
            var ctr_1 : vec4<f32> = old_chunks_0[chunk_word_0(_S64, u32(0))];
            var rr_0 : vec3<f32> = ctr_1.xyz - wcom_0;
            var dd_0 : f32 = dot(rr_0, rr_0);
            var _S65 : vec3<f32> = vec3<f32>(ctr_1.w);
            var _S66 : vec3<f32> = vec3<f32>(old_chunks_0[chunk_word_0(_S64, u32(7))].x);
            var w0_1 : vec3<f32> = w0_0 + (old_chunks_0[chunk_word_0(_S64, u32(1))].xyz + (vec3<f32>(dd_0, 0.0f, 0.0f) - vec3<f32>(rr_0.x) * rr_0) * _S65) * _S66;
            var w1_1 : vec3<f32> = w1_0 + (old_chunks_0[chunk_word_0(_S64, u32(2))].xyz + (vec3<f32>(0.0f, dd_0, 0.0f) - vec3<f32>(rr_0.y) * rr_0) * _S65) * _S66;
            var w2_1 : vec3<f32> = w2_0 + (old_chunks_0[chunk_word_0(_S64, u32(3))].xyz + (vec3<f32>(0.0f, 0.0f, dd_0) - vec3<f32>(rr_0.z) * rr_0) * _S65) * _S66;
            j_2 = j_2 + u32(1);
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
    var _S67 : u32 = u32(5) * p_2;
    var q_5 : Quat_0 = quat_of_0(frames_0[_S67]);
    var pos_0 : vec3<f32> = frames_0[_S67 + u32(4)].xyz;
    var pos_1 : vec3<f32>;
    if(!anchored_0)
    {
        pos_1 = pos_0 + rotate_0(q_5, su_0 / vec3<f32>(su_mass_0));
    }
    else
    {
        pos_1 = pos_0;
    }
    var _S68 : vec3<f32> = inverse_rotate_0(q_5, w_10);
    var iw_0 : vec3<f32> = rotate_0(q_5, vec3<f32>(dot(i0_1, _S68), dot(i1_1, _S68), dot(i2_1, _S68)));
    if(anchored_0)
    {
        j_2 = u32(1);
    }
    else
    {
        j_2 = u32(0);
    }
    var t_2 : u32;
    if((params_0.driven_0) != u32(0))
    {
        t_2 = u32(2);
    }
    else
    {
        t_2 = u32(0);
    }
    var flags_0 : u32 = (((j_2 | (t_2))) | (u32(16)));
    var bfirst_0 : u32 = plan_0[child_bonds_at_0(slot_7)];
    var bend_0 : u32 = plan_0[child_bonds_at_0(slot_7) + u32(1)];
    var rec_0 : array<vec4<f32>, i32(25)>;
    t_2 = u32(0);
    loop
    {
        if(t_2 < u32(25))
        {
        }
        else
        {
            break;
        }
        rec_0[t_2] = vec4<f32>(0.0f);
        t_2 = t_2 + u32(1);
    }
    var _S69 : u32 = c0_8 + first_new_0;
    rec_0[i32(0)] = (bitcast<vec4<f32>>((vec4<u32>(_S69, _S69 + count_1, bfirst_0, bend_0))));
    rec_0[i32(1)] = (bitcast<vec4<f32>>((vec4<u32>(flags_0, u32(0), u32(0), params_0.substeps_0))));
    rec_0[i32(2)] = com_0;
    rec_0[i32(3)] = vec4<f32>(i0_1, 0.0f);
    rec_0[i32(4)] = vec4<f32>(i1_1, 0.0f);
    rec_0[i32(5)] = vec4<f32>(i2_1, 0.0f);
    rec_0[i32(6)] = vec4<f32>(v0_0, 0.0f);
    rec_0[i32(7)] = vec4<f32>(v1_0, 0.0f);
    rec_0[i32(8)] = vec4<f32>(v2_0, 0.0f);
    if(scaled_0)
    {
    }
    else
    {
        wmass_0 = com_0.w;
    }
    rec_0[i32(9)] = vec4<f32>(wcom_0, wmass_0);
    rec_0[i32(10)] = vec4<f32>(x0_0, 0.0f);
    rec_0[i32(11)] = vec4<f32>(x1_0, 0.0f);
    rec_0[i32(12)] = vec4<f32>(x2_0, 0.0f);
    rec_0[i32(13)] = quat_vec_0(q_5);
    rec_0[i32(14)] = vec4<f32>(pos_1, 0.0f);
    if(anchored_0)
    {
        wmoment_0 = _S58;
    }
    else
    {
        wmoment_0 = v_3;
    }
    rec_0[i32(16)] = vec4<f32>(wmoment_0, 0.0f);
    if(anchored_0)
    {
        wmoment_0 = _S58;
    }
    else
    {
        wmoment_0 = w_10;
    }
    rec_0[i32(18)] = vec4<f32>(wmoment_0, 0.0f);
    rec_0[i32(19)] = (bitcast<vec4<f32>>((vec4<u32>(u32(0), u32(0), u32(0), plan_0[child_id_at_0(slot_7)]))));
    rec_0[i32(21)] = vec4<f32>(0.0f, 0.0f, 0.0f, float_bits_0(u32(2139095040)));
    if(anchored_0)
    {
        wmoment_0 = _S58;
    }
    else
    {
        wmoment_0 = iw_0;
    }
    rec_0[i32(23)] = vec4<f32>(wmoment_0, 0.0f);
    t_2 = u32(0);
    loop
    {
        if(t_2 < u32(25))
        {
        }
        else
        {
            break;
        }
        islands_0[island_word_0(_S55, t_2)] = rec_0[t_2];
        t_2 = t_2 + u32(1);
    }
    return;
}

fn lay_engine_0( slot_8 : u32)
{
    var island_2 : u32 = plan_0[child_island_at_0(slot_8)];
    var anchored_1 : bool = (bitcast<u32>((children_0[u32(8) * slot_8].w))) != u32(0);
    if((params_0.resident_0) != u32(0))
    {
        var sl_0 : vec4<u32> = spawn_slots_0[slot_8];
        var _S70 : u32 = u32(2) * slot_8;
        var pr_0 : vec4<f32> = spawn_extras_0[_S70 + u32(1)];
        var at_3 : u32 = params_0.engine_base_0 + u32(3) * island_2;
        var _S71 : u32 = sl_0.w;
        var body_range_0 : vec3<f32>;
        if(_S71 != u32(0))
        {
            var _S72 : u32 = sl_0.y;
            body_range_0 = (bitcast<vec3<f32>>((vec3<u32>(sl_0.x, _S72, _S72 + sl_0.z))));
        }
        else
        {
            body_range_0 = vec3<f32>(float_bits_0(u32(4294967295)), 0.0f, 0.0f);
        }
        var _S73 : bool;
        if(anchored_1)
        {
            _S73 = true;
        }
        else
        {
            _S73 = _S71 == u32(0);
        }
        if(_S73)
        {
            ctab_0[at_3] = vec4<f32>(float_bits_0(u32(4294967295)), body_range_0);
            var _S74 : vec4<f32> = vec4<f32>(0.0f);
            ctab_0[at_3 + u32(1)] = _S74;
            ctab_0[at_3 + u32(2)] = _S74;
        }
        else
        {
            ctab_0[at_3] = vec4<f32>((bitcast<f32>((sl_0.x))), body_range_0);
            ctab_0[at_3 + u32(1)] = vec4<f32>(- pr_0.x, - pr_0.y, - pr_0.z, pr_0.w);
            ctab_0[at_3 + u32(2)] = vec4<f32>(spawn_extras_0[_S70].xyz, 0.0f);
        }
    }
    gate_0[params_0.gate_flag_base_0 + island_2] = vec4<f32>(0.0f);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var tid_8 : u32 = thread_1.x;
    var p_3 : u32 = group_0.x;
    var parent_7 : vec4<u32>;
    var c0_9 : u32;
    var c1_3 : u32;
    var b0_5 : u32;
    var b1_5 : u32;
    var _S75 : bool = layout_start_0(p_3, tid_8, &(parent_7), &(c0_9), &(c1_3), &(b0_5), &(b1_5));
    if(!_S75)
    {
        return;
    }
    var n_5 : u32 = c1_3 - c0_9;
    var base_8 : u32 = parent_7.y;
    var total_3 : u32 = control_0[u32(6)];
    lay_positions_0(parent_7, n_5, base_8, control_0[u32(6)], tid_8);
    device_group_barrier_0();
    lay_chunks_0(c0_9, n_5, base_8, tid_8);
    if((params_0.resident_0) != u32(0))
    {
        lay_ctab_chunks_0(c0_9, n_5, base_8, tid_8);
    }
    device_group_barrier_0();
    lay_bond_ranks_0(parent_7, c0_9, b0_5, b1_5, base_8, tid_8);
    lay_bonds_0(c0_9, b0_5, b1_5, base_8, tid_8);
    device_group_barrier_0();
    lay_incidence_0(c0_9, c1_3, b0_5, b1_5, n_5, base_8, g_cut_first_0, tid_8);
    device_group_barrier_0();
    var k_3 : u32 = tid_8;
    loop
    {
        if(k_3 < (parent_7.z))
        {
        }
        else
        {
            break;
        }
        var _S76 : u32 = base_8 + k_3;
        lay_record_0(p_3, c0_9, base_8, total_3, _S76);
        lay_engine_0(_S76);
        k_3 = k_3 + u32(256);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_positions(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var _S77 : u32 = thread_2.x;
    var parent_8 : vec4<u32>;
    var c0_10 : u32;
    var c1_4 : u32;
    var b0_6 : u32;
    var b1_6 : u32;
    var _S78 : bool = layout_start_0(group_1.x, _S77, &(parent_8), &(c0_10), &(c1_4), &(b0_6), &(b1_6));
    if(!_S78)
    {
        return;
    }
    lay_positions_0(parent_8, c1_4 - c0_10, parent_8.y, control_0[u32(6)], _S77);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_chunks(@builtin(workgroup_id) group_2 : vec3<u32>, @builtin(local_invocation_id) thread_3 : vec3<u32>)
{
    var _S79 : u32 = thread_3.x;
    var parent_9 : vec4<u32>;
    var c0_11 : u32;
    var c1_5 : u32;
    var b0_7 : u32;
    var b1_7 : u32;
    var _S80 : bool = layout_start_0(group_2.x, _S79, &(parent_9), &(c0_11), &(c1_5), &(b0_7), &(b1_7));
    if(!_S80)
    {
        return;
    }
    lay_chunks_0(c0_11, c1_5 - c0_11, parent_9.y, _S79);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_ctab(@builtin(workgroup_id) group_3 : vec3<u32>, @builtin(local_invocation_id) thread_4 : vec3<u32>)
{
    var _S81 : u32 = thread_4.x;
    var parent_10 : vec4<u32>;
    var c0_12 : u32;
    var c1_6 : u32;
    var b0_8 : u32;
    var b1_8 : u32;
    var _S82 : bool = layout_start_0(group_3.x, _S81, &(parent_10), &(c0_12), &(c1_6), &(b0_8), &(b1_8));
    var _S83 : bool;
    if(!_S82)
    {
        _S83 = true;
    }
    else
    {
        _S83 = (params_0.resident_0) == u32(0);
    }
    if(_S83)
    {
        return;
    }
    lay_ctab_chunks_0(c0_12, c1_6 - c0_12, parent_10.y, _S81);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_bond_ranks(@builtin(workgroup_id) group_4 : vec3<u32>, @builtin(local_invocation_id) thread_5 : vec3<u32>)
{
    var _S84 : u32 = thread_5.x;
    var parent_11 : vec4<u32>;
    var c0_13 : u32;
    var c1_7 : u32;
    var b0_9 : u32;
    var b1_9 : u32;
    var _S85 : bool = layout_start_0(group_4.x, _S84, &(parent_11), &(c0_13), &(c1_7), &(b0_9), &(b1_9));
    if(!_S85)
    {
        return;
    }
    lay_bond_ranks_0(parent_11, c0_13, b0_9, b1_9, parent_11.y, _S84);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_bonds(@builtin(workgroup_id) group_5 : vec3<u32>, @builtin(local_invocation_id) thread_6 : vec3<u32>)
{
    var _S86 : u32 = thread_6.x;
    var parent_12 : vec4<u32>;
    var c0_14 : u32;
    var c1_8 : u32;
    var b0_10 : u32;
    var b1_10 : u32;
    var _S87 : bool = layout_start_0(group_5.x, _S86, &(parent_12), &(c0_14), &(c1_8), &(b0_10), &(b1_10));
    if(!_S87)
    {
        return;
    }
    lay_bonds_0(c0_14, b0_10, b1_10, parent_12.y, _S86);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_incidence(@builtin(workgroup_id) group_6 : vec3<u32>, @builtin(local_invocation_id) thread_7 : vec3<u32>)
{
    var _S88 : u32 = thread_7.x;
    var parent_13 : vec4<u32>;
    var c0_15 : u32;
    var c1_9 : u32;
    var b0_11 : u32;
    var b1_11 : u32;
    var _S89 : bool = layout_start_0(group_6.x, _S88, &(parent_13), &(c0_15), &(c1_9), &(b0_11), &(b1_11));
    if(!_S89)
    {
        return;
    }
    lay_incidence_0(c0_15, c1_9, b0_11, b1_11, c1_9 - c0_15, parent_13.y, plan_0[child_bonds_at_0(parent_13.y + child_by_island_0(parent_13, parent_13.z - u32(1))) + u32(1)], _S88);
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_records(@builtin(workgroup_id) group_7 : vec3<u32>, @builtin(local_invocation_id) thread_8 : vec3<u32>)
{
    var _S90 : u32 = group_7.x;
    var _S91 : u32 = thread_8.x;
    var parent_14 : vec4<u32>;
    var c0_16 : u32;
    var c1_10 : u32;
    var b0_12 : u32;
    var b1_12 : u32;
    var _S92 : bool = layout_start_0(_S90, _S91, &(parent_14), &(c0_16), &(c1_10), &(b0_12), &(b1_12));
    if(!_S92)
    {
        return;
    }
    var k_4 : u32 = _S91;
    loop
    {
        if(k_4 < (parent_14.z))
        {
        }
        else
        {
            break;
        }
        lay_record_0(_S90, c0_16, parent_14.y, control_0[u32(6)], parent_14.y + k_4);
        k_4 = k_4 + u32(256);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn layout_parent_engine(@builtin(workgroup_id) group_8 : vec3<u32>, @builtin(local_invocation_id) thread_9 : vec3<u32>)
{
    var _S93 : u32 = thread_9.x;
    var parent_15 : vec4<u32>;
    var c0_17 : u32;
    var c1_11 : u32;
    var b0_13 : u32;
    var b1_13 : u32;
    var _S94 : bool = layout_start_0(group_8.x, _S93, &(parent_15), &(c0_17), &(c1_11), &(b0_13), &(b1_13));
    if(!_S94)
    {
        return;
    }
    var k_5 : u32 = _S93;
    loop
    {
        if(k_5 < (parent_15.z))
        {
        }
        else
        {
            break;
        }
        lay_engine_0(parent_15.y + k_5);
        k_5 = k_5 + u32(256);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn layout_moved(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S95 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S95 = true;
    }
    else
    {
        _S95 = control_0[u32(1)] == u32(0);
    }
    if(_S95)
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
        _S95 = true;
    }
    else
    {
        _S95 = e_1 == i_5;
    }
    if(_S95)
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
    var r_4 : vec4<f32> = old_islands_0[island_word_0(e_1, u32(0))];
    var _S96 : u32 = (bitcast<u32>((r_4.y)));
    var b0_14 : u32 = (bitcast<u32>((r_4.z)));
    var _S97 : u32 = (bitcast<u32>((r_4.w)));
    var c_3 : u32 = (bitcast<u32>((r_4.x)));
    loop
    {
        if(c_3 < _S96)
        {
        }
        else
        {
            break;
        }
        set_chunk_island_0(c_3, i_5);
        c_3 = c_3 + u32(1);
    }
    var b_4 : u32 = b0_14;
    loop
    {
        if(b_4 < _S97)
        {
        }
        else
        {
            break;
        }
        var _S98 : u32 = bond_word_0(b_4, u32(14));
        var ids_2 : vec4<f32> = bonds_0[_S98];
        ids_2[i32(3)] = (bitcast<f32>((i_5)));
        bonds_0[_S98] = ids_2;
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
    var _S99 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S99 = true;
    }
    else
    {
        _S99 = control_0[u32(1)] == u32(0);
    }
    if(_S99)
    {
        return;
    }
    var i_6 : u32 = id_2.x;
    var n_6 : u32 = plan_0[i32(0)];
    if(i_6 > plan_0[i32(0)])
    {
        return;
    }
    var t_4 : u32;
    if(i_6 == n_6)
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
            var _S100 : u32 = island_word_0(i_6, t_4);
            island_copy_0[_S100] = islands_0[_S100];
            t_4 = t_4 + u32(1);
        }
    }
    return;
}

