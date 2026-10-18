@binding(11) @group(0) var<storage, read> control_0 : array<u32>;

struct ColliderParams_std140_0
{
    @align(16) parent_count_0 : u32,
    @align(4) chunk_count_0 : u32,
    @align(8) collider_count_0 : u32,
    @align(4) chunk_words_0 : u32,
    @align(16) island_words_0 : u32,
    @align(4) ground_modulus_0 : f32,
    @align(8) pad0_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : ColliderParams_std140_0;
@binding(1) @group(0) var<storage, read> parents_0 : array<vec4<u32>>;

@binding(9) @group(0) var<storage, read> islands_of_0 : array<vec4<u32>>;

@binding(14) @group(0) var<storage, read_write> collider_key_0 : array<u32>;

@binding(4) @group(0) var<storage, read> slots_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> children_0 : array<vec4<f32>>;

@binding(2) @group(0) var<storage, read> work_0 : array<u32>;

@binding(6) @group(0) var<storage, read> old_chunks_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> structure_base_0 : array<u32>;

@binding(5) @group(0) var<storage, read> part_ranges_0 : array<vec4<u32>>;

@binding(7) @group(0) var<storage, read> chunks_0 : array<vec4<f32>>;

@binding(13) @group(0) var<storage, read_write> chunk_slot_0 : array<u32>;

@binding(15) @group(0) var<storage, read_write> collider_info_0 : array<vec4<u32>>;

@binding(8) @group(0) var<storage, read> islands_0 : array<vec4<f32>>;

@binding(12) @group(0) var<storage, read> chunk_props_0 : array<vec4<f32>>;

fn total_slots_0() -> u32
{
    return parents_0[params_0.parent_count_0].y;
}

fn parent_of_0( slot_0 : u32) -> u32
{
    var p_0 : u32 = u32(0);
    loop
    {
        var _S1 : u32 = p_0 + u32(1);
        var _S2 : bool;
        if(_S1 < (params_0.parent_count_0))
        {
            _S2 = (parents_0[_S1].y) <= slot_0;
        }
        else
        {
            _S2 = false;
        }
        if(_S2)
        {
        }
        else
        {
            break;
        }
        p_0 = _S1;
    }
    return p_0;
}

fn key_of_0( info_0 : vec4<u32>) -> u32
{
    return structure_base_0[info_0.z] + info_0.w;
}

@compute
@workgroup_size(64, 1, 1)
fn collider_keys(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    if(control_0[i32(0)] != u32(0))
    {
        return;
    }
    var t_0 : u32 = id_0.x;
    var _S3 : bool;
    if(t_0 < (params_0.parent_count_0))
    {
        _S3 = (parents_0[t_0].z) >= u32(2);
    }
    else
    {
        _S3 = false;
    }
    var j_0 : u32;
    if(_S3)
    {
        var e_0 : vec4<u32> = islands_of_0[t_0];
        j_0 = e_0.z;
        loop
        {
            if(j_0 < (e_0.w))
            {
            }
            else
            {
                break;
            }
            collider_key_0[j_0] = u32(4294967294);
            j_0 = j_0 + u32(1);
        }
    }
    var total_0 : u32 = total_slots_0();
    if(t_0 >= total_0)
    {
        return;
    }
    var p_1 : u32 = parent_of_0(t_0);
    var parent_0 : vec4<u32> = parents_0[p_1];
    var _S4 : u32 = parent_0.y;
    if((t_0 - _S4) >= (parent_0.z))
    {
        _S3 = true;
    }
    else
    {
        _S3 = (slots_0[t_0].w) == u32(0);
    }
    if(_S3)
    {
        return;
    }
    var _S5 : u32 = islands_of_0[p_1].x;
    var head_0 : vec4<f32> = children_0[u32(8) * t_0];
    var _S6 : u32 = (bitcast<u32>((head_0.y)));
    var _S7 : u32 = (bitcast<u32>((head_0.z)));
    var _S8 : u32 = slots_0[t_0].y;
    j_0 = u32(0);
    var col_0 : u32 = _S8;
    loop
    {
        if(j_0 < _S7)
        {
        }
        else
        {
            break;
        }
        var key_0 : u32 = key_of_0((bitcast<vec4<u32>>((old_chunks_0[params_0.chunk_words_0 * (_S5 + work_0[total_0 + _S4 + _S6 + j_0]) + u32(8)]))));
        var _S9 : u32 = part_ranges_0[key_0].y;
        var k_0 : u32 = u32(0);
        loop
        {
            if(k_0 < _S9)
            {
            }
            else
            {
                break;
            }
            collider_key_0[col_0] = key_0;
            var _S10 : u32 = col_0 + u32(1);
            k_0 = k_0 + u32(1);
            col_0 = _S10;
        }
        j_0 = j_0 + u32(1);
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn collider_slots(@builtin(global_invocation_id) id_1 : vec3<u32>)
{
    var _S11 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S11 = true;
    }
    else
    {
        _S11 = (id_1.x) >= (params_0.chunk_count_0);
    }
    if(_S11)
    {
        return;
    }
    var _S12 : u32 = id_1.x;
    chunk_slot_0[key_of_0((bitcast<vec4<u32>>((chunks_0[params_0.chunk_words_0 * _S12 + u32(8)]))))] = _S12;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn collider_refresh(@builtin(global_invocation_id) id_2 : vec3<u32>)
{
    var c_0 : u32 = id_2.x;
    var _S13 : bool;
    if(control_0[i32(0)] != u32(0))
    {
        _S13 = true;
    }
    else
    {
        _S13 = c_0 >= (params_0.collider_count_0);
    }
    if(_S13)
    {
        return;
    }
    var key_1 : u32 = collider_key_0[c_0];
    if(collider_key_0[c_0] == u32(4294967295))
    {
        return;
    }
    var inf_0 : f32 = (bitcast<f32>((u32(2139095040))));
    if(key_1 == u32(4294967294))
    {
        var _S14 : u32 = u32(2) * c_0;
        collider_info_0[_S14] = vec4<u32>(u32(4294967295), u32(4294967295), u32(1), u32(0));
        collider_info_0[_S14 + u32(1)] = (bitcast<vec4<u32>>((vec4<f32>(inf_0, params_0.ground_modulus_0, 1.0f, 0.0f))));
        collider_key_0[c_0] = u32(4294967295);
        return;
    }
    var island_0 : u32 = (bitcast<vec4<u32>>((chunks_0[params_0.chunk_words_0 * chunk_slot_0[key_1] + u32(8)]))).y;
    var mass_0 : f32 = islands_0[params_0.island_words_0 * island_0 + u32(2)].w;
    var props_0 : vec4<f32> = chunk_props_0[key_of_0((bitcast<vec4<u32>>((chunks_0[params_0.chunk_words_0 * (bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * island_0]))).x + u32(8)]))))];
    var anchored_0 : bool = ((((bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * island_0 + u32(1)]))).x) & (u32(1)))) != u32(0);
    var _S15 : u32 = u32(2) * c_0;
    collider_info_0[_S15] = vec4<u32>(chunk_slot_0[key_1], (((bitcast<vec4<u32>>((islands_0[params_0.island_words_0 * island_0 + u32(19)]))).w) << (u32(1))), u32(2), u32(1));
    var _S16 : f32;
    if(anchored_0)
    {
        _S16 = inf_0;
    }
    else
    {
        _S16 = mass_0;
    }
    collider_info_0[_S15 + u32(1)] = (bitcast<vec4<u32>>((vec4<f32>(_S16, props_0.x, props_0.y, 0.0f))));
    return;
}

