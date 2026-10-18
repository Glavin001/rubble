// The coupling's collider table (coupling.slang `ColliderInfo`) kept current on the GPU
// after a split the GPU laid out (engine/docs/stress-baseline-goal.md, M4), in place of the
// host's rebuild of it (stress-rubble `collider_table`): what the host's table says of a
// chunk's collider follows from its chunk, its island and its island's first chunk.
//   collider_keys     per child slot: the stable chunk key (`structure_base[s] + c`) of each
//                     collider spawn.slang made, in its order; per split parent: its retired
//                     colliders marked RETIRED
//   collider_slots    per layout chunk: its index, by its stable key (after the layout)
//   collider_refresh  per engine collider with a key: its table entry from its chunk's
//                     island (cluster id, mass, anchored) and that island's first chunk
//                     (modulus, radius); a retired one becomes static geometry
// Runs after layout.slang in the split pass; `old_chunks` is the layout before it.

import common;

struct ColliderParams
{
    uint parent_count;
    uint chunk_count;     // layout chunks
    uint collider_count;  // engine colliders to refresh (its capacity: keys past the count are NONE)
    uint chunk_words;     // float4 per chunk static (16)
    uint island_words;    // float4 per island (25)
    float ground_modulus; // static geometry's modulus (the host's `other(None)`)
    uint pad0;
    uint pad1;
};

[[vk::binding(0, 0)]] ConstantBuffer<ColliderParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> parents;     // split.slang (y: first child slot, z: children)
[[vk::binding(2, 0)]] StructuredBuffer<uint> work;         // split.slang (members)
[[vk::binding(3, 0)]] StructuredBuffer<float4> children;   // split.slang child records
[[vk::binding(4, 0)]] StructuredBuffer<uint4> slots;       // spawn.slang: body, first collider, colliders, valid
[[vk::binding(5, 0)]] StructuredBuffer<uint4> part_ranges; // per stable chunk: first part, count
[[vk::binding(6, 0)]] StructuredBuffer<float4> old_chunks; // chunk statics before the layout
[[vk::binding(7, 0)]] StructuredBuffer<float4> chunks;     // chunk statics after it
[[vk::binding(8, 0)]] StructuredBuffer<float4> islands;    // islands after it
[[vk::binding(9, 0)]] StructuredBuffer<uint4> islands_of;  // per parent: first chunk, engine body, first collider, collider end
[[vk::binding(10, 0)]] StructuredBuffer<uint> structure_base;
[[vk::binding(11, 0)]] StructuredBuffer<uint> control;     // spawn.slang: [0] 1 when nothing was made
[[vk::binding(12, 0)]] StructuredBuffer<float4> chunk_props; // per stable chunk: modulus, radius
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> chunk_slot;  // per stable chunk: its layout index
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> collider_key; // per engine collider: stable chunk, NONE or RETIRED
[[vk::binding(15, 0)]] RWStructuredBuffer<uint4> collider_info; // two uint4 per collider (`ColliderInfo` a, b)

static const uint NONE = 0xffffffffu;
static const uint RETIRED = 0xfffffffeu;
static const uint CHILD_WORDS = 8;
static const uint KIND_STATIC = 1;
static const uint KIND_BODY = 2;
static const uint ISLAND_FLAG_ANCHORED = 1;

uint total_slots()
{
    return parents[control[CTL_PARENTS]].y;
}

uint parent_of(uint slot)
{
    uint p = 0;
    while (p + 1 < control[CTL_PARENTS] && parents[p + 1].y <= slot)
        p++;
    return p;
}

uint key_of(uint4 info)
{
    return structure_base[info.z] + info.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collider_keys(uint3 id: SV_DispatchThreadID)
{
    if (control[0] != 0)
        return;
    uint t = id.x;
    // A split parent's colliders, retired with its body (spawn.slang `spawn_retire`).
    if (t < control[CTL_PARENTS] && parents[t].z >= 2)
    {
        uint4 e = islands_of[t];
        for (uint c = e.z; c < e.w; c++)
            collider_key[c] = RETIRED;
    }
    uint total = total_slots();
    if (t >= total)
        return;
    uint p = parent_of(t);
    uint4 parent = parents[p];
    if (t - parent.y >= parent.z || slots[t].w == 0)
        return;
    // The child's colliders as `spawn_bodies` made them: per member, its parts in order.
    uint first = islands_of[p].x;
    float4 head = children[CHILD_WORDS * t];
    uint2 m = uint2(asuint(head.y), asuint(head.z));
    uint col = slots[t].y;
    for (uint j = 0; j < m.y; j++)
    {
        uint chunk = first + work[total + parent.y + m.x + j];
        uint key = key_of(asuint(old_chunks[params.chunk_words * chunk + 8]));
        uint count = part_ranges[key].y;
        for (uint k = 0; k < count; k++)
        {
            collider_key[col] = key;
            col++;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collider_slots(uint3 id: SV_DispatchThreadID)
{
    if (control[0] != 0 || id.x >= params.chunk_count)
        return;
    chunk_slot[key_of(asuint(chunks[params.chunk_words * id.x + 8]))] = id.x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collider_refresh(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (control[0] != 0 || c >= params.collider_count)
        return;
    uint key = collider_key[c];
    if (key == NONE)
        return;
    float inf = asfloat(0x7f800000u);
    if (key == RETIRED)
    {
        // `Owner::Retired`: static geometry, as the other side sees it.
        collider_info[2 * c] = uint4(NONE, NONE, KIND_STATIC, 0);
        collider_info[2 * c + 1] = asuint(float4(inf, params.ground_modulus, 1.0, 0.0));
        collider_key[c] = NONE;
        return;
    }
    uint k = chunk_slot[key];
    uint island = asuint(chunks[params.chunk_words * k + 8]).y;
    uint4 range = asuint(islands[params.island_words * island]);
    uint4 info = asuint(islands[params.island_words * island + 1]);
    float mass = islands[params.island_words * island + 2].w;
    uint cluster = asuint(islands[params.island_words * island + 19]).w;
    uint first_key = key_of(asuint(chunks[params.chunk_words * range.x + 8]));
    float4 props = chunk_props[first_key];
    bool anchored = (info.x & ISLAND_FLAG_ANCHORED) != 0;
    collider_info[2 * c] = uint4(k, cluster << 1, KIND_BODY, 1);
    collider_info[2 * c + 1] = asuint(float4(anchored ? inf : mass, props.x, props.y, 0.0));
}
