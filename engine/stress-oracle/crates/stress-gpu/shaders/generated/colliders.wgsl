@binding(11) @group(0) var<storage, read> control_0 : array<u32>;

struct ColliderParams_std140_0
{
    @align(16) parent_count_0 : u32,
    @align(4) chunk_count_0 : u32,
    @align(8) collider_count_0 : u32,
    @align(4) chunk_words_0 : u32,
    @align(16) island_words_0 : u32,
    @align(4) ground_modulus_0 : f32,
    @align(8) threads_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : ColliderParams_std140_0;
@binding(1) @group(0) var<storage, read> parents_0 : array<vec4<u32>>;

@binding(9) @group(0) var<storage, read> islands_of_0 : array<vec4<u32>>;

@binding(14) @group(0) var<storage, read_write> collider_key_0 : array<u32>;

@binding(2) @group(0) var<storage, read> work_0 : array<u32>;

@binding(6) @group(0) var<storage, read> old_chunks_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> structure_base_0 : array<u32>;

@binding(5) @group(0) var<storage, read> part_ranges_0 : array<vec4<u32>>;

@binding(16) @group(0) var<storage, read> member_offsets_0 : array<u32>;

@binding(7) @group(0) var<storage, read> chunks_0 : array<vec4<f32>>;

@binding(13) @group(0) var<storage, read_write> chunk_slot_0 : array<u32>;

@binding(15) @group(0) var<storage, read_write> collider_info_0 : array<vec4<u32>>;

@binding(8) @group(0) var<storage, read> islands_0 : array<vec4<f32>>;

@binding(12) @group(0) var<storage, read> chunk_props_0 : array<vec4<f32>>;

fn total_slots_0() -> u32
{
    return parents_0[control_0[u32(1)]].y;
}

fn key_of_0( info_0 : vec4<u32>) -> u32
{
    return structure_base_0[info_0.z] + info_0.w;
}

@compute
@workgroup_size(64, 1, 1)
fn collider_keys(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var c_0 : u32;
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
    var t_0 : u32 = id_0.x;
    var _S2 : u32 = params_0.threads_0;
    var p_0 : u32 = u32(0);
    loop
    {
        if(p_0 < control_0[u32(1)])
        {
        }
        else
        {
            break;
        }
        if((parents_0[p_0].z) < u32(2))
        {
            p_0 = p_0 + u32(1);
            continue;
        }
        var e_0 : vec4<u32> = islands_of_0[p_0];
        c_0 = e_0.z + t_0;
        loop
        {
            if(c_0 < (e_0.w))
            {
            }
            else
            {
                break;
            }
            collider_key_0[c_0] = u32(4294967294);
            c_0 = c_0 + _S2;
        }
        p_0 = p_0 + u32(1);
    }
    var total_0 : u32 = total_slots_0();
    if(t_0 >= total_0)
    {
        return;
    }
    var _S3 : u32 = control_0[u32(1)] - u32(1);
    p_0 = u32(0);
    c_0 = _S3;
    loop
    {
        if(p_0 < c_0)
        {
        }
        else
        {
            break;
        }
        var mid_0 : u32 = (p_0 + c_0 + u32(1)) / u32(2);
        if((parents_0[mid_0].y) <= t_0)
        {
            p_0 = mid_0;
        }
        else
        {
            c_0 = mid_0 - u32(1);
        }
    }
    if((parents_0[p_0].z) < u32(2))
    {
        return;
    }
    var key_0 : u32 = key_of_0((bitcast<vec4<u32>>((old_chunks_0[params_0.chunk_words_0 * (islands_of_0[p_0].x + work_0[total_0 + t_0]) + u32(8)]))));
    var _S4 : u32 = part_ranges_0[key_0].y;
    var _S5 : u32 = member_offsets_0[t_0];
    var k_0 : u32 = u32(0);
    loop
    {
        if(k_0 < _S4)
        {
        }
        else
        {
            break;
        }
        collider_key_0[_S5 + k_0] = key_0;
        k_0 = k_0 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn collider_slots(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S6 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S6 = true;
    }
    else
    {
        _S6 = control_0[u32(1)] == u32(0);
    }
    if(_S6)
    {
        _S6 = true;
    }
    else
    {
        _S6 = (id_1.x) >= (params_0.chunk_count_0);
    }
    if(_S6)
    {
        return;
    }
    var _S7 : u32 = id_1.x;
    chunk_slot_0[key_of_0((bitcast<vec4<u32>>((chunks_0[params_0.chunk_words_0 * _S7 + u32(8)]))))] = _S7;
    return;
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

@compute
@workgroup_size(64, 1, 1)
fn collider_refresh(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var c_1 : u32 = id_2.x;
    var _S8 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S8 = true;
    }
    else
    {
        _S8 = control_0[u32(1)] == u32(0);
    }
    if(_S8)
    {
        _S8 = true;
    }
    else
    {
        _S8 = c_1 >= (params_0.collider_count_0);
    }
    if(_S8)
    {
        return;
    }
    var key_1 : u32 = collider_key_0[c_1];
    if(collider_key_0[c_1] == u32(4294967295))
    {
        return;
    }
    var inf_0 : f32 = float_bits_0(u32(2139095040));
    if(key_1 == u32(4294967294))
    {
        var _S9 : u32 = u32(2) * c_1;
        collider_info_0[_S9] = vec4<u32>(u32(4294967295), u32(4294967295), u32(1), u32(0));
        collider_info_0[_S9 + u32(1)] = (bitcast<vec4<u32>>((vec4<f32>(inf_0, params_0.ground_modulus_0, 1.0f, 0.0f))));
        collider_key_0[c_1] = u32(4294967295);
        return;
    }
    var island_0 : u32 = (bitcast<vec4<u32>>((chunks_0[params_0.chunk_words_0 * chunk_slot_0[key_1] + u32(8)]))).y;
    var mass_0 : f32 = islands_0[params_0.island_words_0 * island_0 + u32(2)].w;
    var props_0 : vec4<f32> = chunk_props_0[key_of_0((bitcast<vec4<u32>>((chunks_0[params_0.chunk_words_0 * (bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * island_0]))).x + u32(8)]))))];
    var anchored_0 : bool = ((((bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * island_0 + u32(1)]))).x) & (u32(1)))) != u32(0);
    var _S10 : u32 = u32(2) * c_1;
    collider_info_0[_S10] = vec4<u32>(chunk_slot_0[key_1], (((bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * island_0 + u32(19)]))).w) << (u32(1))), u32(2), u32(1));
    var _S11 : f32;
    if(anchored_0)
    {
        _S11 = inf_0;
    }
    else
    {
        _S11 = mass_0;
    }
    collider_info_0[_S10 + u32(1)] = (bitcast<vec4<u32>>((vec4<f32>(_S11, props_0.x, props_0.y, 0.0f))));
    return;
}

