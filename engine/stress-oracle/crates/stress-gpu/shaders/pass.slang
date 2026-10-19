// A split pass's inputs made on the GPU (engine/docs/stress-baseline-goal.md, M4): what the
// host otherwise gives `GpuSplitter::pass` from its layout and the engine's bookkeeping, from
// the cascade's status and the resident tables, so a split needs no host between the pass
// that disconnected an island and the next:
//   pass_parents   one group: the islands that disconnected (`cascade.slang` status bit 3,
//                  when a disconnection stopped the cascade), in ascending order, with their
//                  chunk bases (split.slang `parents`), each one's first chunk, engine body
//                  and collider range (spawn.slang `islands_of`) and engine frame data
//                  (split.slang `parent_bodies`, from the ctab engine entries); the control
//                  words (common.slang `CTL_*`) from the GPU's counters; the split kernels'
//                  indirect dispatch arguments
//   pass_snapshot  one group per parent: its chunk and bond ranges copied as they are (the
//                  layout reads them while it rewrites the live tables)
//   pass_finish    one thread: after the layout, the counters follow (live islands, engine
//                  bodies and colliders, the next cluster id) and the cascade's dispatch
//                  arguments over the live islands

import common;

struct PassParams
{
    uint chunk_words;      // float4 per chunk static (16)
    uint bond_words;       // float4 per bond static (15)
    uint dyn_words;        // float4 per bond record
    uint island_words;     // float4 per island (25)
    uint props_base;       // ctab: 1 float4 per chunk
    uint eload_base;       // ctab: 2 float4 per chunk
    uint pulse_base;       // ctab: 3 * pulse_cap float4 per chunk
    uint pulse_count_base; // ctab: 1 per chunk
    uint pulse_cap;
    uint engine_base;      // ctab: 3 float4 per island
    uint resident;         // 1: the ctab follows
    uint resume;           // 1: after a split, the cascade runs on (pass_finish)
    uint island_float4s;   // islands: (capacity + 1) records
    uint index_words;      // the incidence (uint)
    uint engine_float4s;   // ctab: the island entries (3 per island of the capacity)
    uint copy_float4s;     // the cascade's island copy
};

[[vk::binding(0, 0)]] ConstantBuffer<PassParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> status;       // cascade.slang status record
[[vk::binding(2, 0)]] StructuredBuffer<float4> islands;
[[vk::binding(3, 0)]] StructuredBuffer<float4> ctab;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> control;      // CTL_*
[[vk::binding(5, 0)]] RWStructuredBuffer<uint4> parents;     // split.slang `parents`
[[vk::binding(6, 0)]] RWStructuredBuffer<uint4> islands_of;  // spawn.slang
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> parent_bodies; // split.slang
// Indirect arguments (uint4 each): one group per parent; per child slot in groups of 64; per
// parent in groups of 64.
[[vk::binding(8, 0)]] RWStructuredBuffer<uint4> args;
// The GPU's counters across a frame: live islands, engine bodies, engine colliders, the
// next cluster id.
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> counters;
[[vk::binding(10, 0)]] StructuredBuffer<float4> chunks;
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> old_chunks;
[[vk::binding(12, 0)]] StructuredBuffer<float4> state;
[[vk::binding(13, 0)]] RWStructuredBuffer<float4> old_state;
[[vk::binding(14, 0)]] StructuredBuffer<float4> bonds;
[[vk::binding(15, 0)]] RWStructuredBuffer<float4> old_bonds;
[[vk::binding(16, 0)]] StructuredBuffer<float4> bond_dyn;
[[vk::binding(17, 0)]] RWStructuredBuffer<float4> old_dyn;
[[vk::binding(18, 0)]] RWStructuredBuffer<float4> old_ctab;
// The cascade's indirect arguments over the live islands (solver.rs `island_args`).
[[vk::binding(19, 0)]] RWStructuredBuffer<uint4> island_args;
[[vk::binding(20, 0)]] StructuredBuffer<uint> index;
[[vk::binding(21, 0)]] RWStructuredBuffer<uint> old_index;
[[vk::binding(22, 0)]] RWStructuredBuffer<float4> old_islands;
[[vk::binding(23, 0)]] StructuredBuffer<float4> island_copy;
[[vk::binding(24, 0)]] RWStructuredBuffer<float4> old_copy;

static const uint NONE = 0xffffffffu;
static const uint ST_RUN = 0;
static const uint ST_DISCONNECTED = 1;
static const uint ST_ISLANDS = 16;
static const uint DISCONNECTED_BIT = 8;

uint4 island_range(uint i)
{
    return asuint(islands[params.island_words * i]);
}

groupshared uint g_parents[1024];
groupshared uint g_chunks[1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void pass_parents(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    // The loop bound as a uniform value: its body has barriers.
    uint n = group_uniform(counters[0], tid);
    bool split = status[ST_DISCONNECTED] != 0;
    uint carry_p = 0;
    uint carry_c = 0;
    for (uint base = 0; base < n; base += 1024)
    {
        uint i = base + tid;
        bool parent = split && i < n && (status[ST_ISLANDS + i] & DISCONNECTED_BIT) != 0;
        uint4 r = parent ? island_range(i) : uint4(0);
        uint fp = parent ? 1u : 0u;
        uint fc = parent ? r.y - r.x : 0u;
        g_parents[tid] = fp;
        g_chunks[tid] = fc;
        GroupMemoryBarrierWithGroupSync();
        // Inclusive scans (a fixed tree: the same order on every run).
        for (uint step = 1; step < 1024; step <<= 1)
        {
            uint ap = tid >= step ? g_parents[tid - step] : 0;
            uint ac = tid >= step ? g_chunks[tid - step] : 0;
            GroupMemoryBarrierWithGroupSync();
            g_parents[tid] += ap;
            g_chunks[tid] += ac;
            GroupMemoryBarrierWithGroupSync();
        }
        if (parent)
        {
            uint p = carry_p + g_parents[tid] - fp;
            uint at = carry_c + g_chunks[tid] - fc;
            parents[p] = uint4(i, at, 0, 0);
            float4 e0 = ctab[params.engine_base + 3 * i];
            float4 e1 = ctab[params.engine_base + 3 * i + 1];
            float4 e2 = ctab[params.engine_base + 3 * i + 2];
            uint4 body = asuint(e0);
            islands_of[p] = uint4(r.x, body.y, body.z, body.w);
            // split.slang `parent_bodies`: (driven body as bits, centre of mass), principal
            // rotation (the entry holds its inverse).
            parent_bodies[2 * p] = float4(e0.x, e2.xyz);
            parent_bodies[2 * p + 1] = float4(-e1.x, -e1.y, -e1.z, e1.w);
        }
        uint tp = g_parents[1023];
        uint tc = g_chunks[1023];
        GroupMemoryBarrierWithGroupSync();
        carry_p += tp;
        carry_c += tc;
    }
    if (tid != 0)
        return;
    parents[carry_p] = uint4(0, carry_c, 0, 0);
    control[CTL_ABORT] = 0;
    control[CTL_PARENTS] = carry_p;
    control[CTL_ISLANDS] = n;
    control[CTL_NEXT_ID] = counters[3];
    control[CTL_FIRST_BODY] = counters[1];
    control[CTL_FIRST_COLLIDER] = counters[2];
    control[CTL_TOTAL] = carry_c;
    args[0] = uint4(carry_p, 1, 1, 0);
    args[1] = uint4((carry_c + 63) / 64, 1, 1, 0);
    args[2] = uint4((carry_p + 63) / 64, 1, 1, 0);
    // `pass_whole`: only when something splits.
    args[3] = uint4(carry_p > 0 ? WHOLE_GROUPS : 0u, 1, 1, 0);
    // One group per child slot (split.slang `split_children_group`).
    args[4] = uint4(carry_c, 1, 1, 0);
}

// The tables the layout reads whole (any island may move): the islands, the incidence,
// the ctab's island entries, the cascade's island copy; copied only when something splits.
static const uint WHOLE_GROUPS = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void pass_whole(uint3 id: SV_DispatchThreadID)
{
    uint stride = 256 * WHOLE_GROUPS;
    for (uint k = id.x; k < params.island_float4s; k += stride)
        old_islands[k] = islands[k];
    for (uint k = id.x; k < params.index_words; k += stride)
        old_index[k] = index[k];
    if (params.resident != 0)
        for (uint k = id.x; k < params.engine_float4s; k += stride)
            old_ctab[params.engine_base + k] = ctab[params.engine_base + k];
    for (uint k = id.x; k < params.copy_float4s; k += stride)
        old_copy[k] = island_copy[k];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pass_snapshot(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint p = group.x;
    if (p >= control[CTL_PARENTS])
        return;
    uint4 r = island_range(parents[p].x);
    uint t = thread.x;
    for (uint k = params.chunk_words * r.x + t; k < params.chunk_words * r.y; k += 256)
        old_chunks[k] = chunks[k];
    for (uint k = 4 * r.x + t; k < 4 * r.y; k += 256)
        old_state[k] = state[k];
    for (uint k = params.bond_words * r.z + t; k < params.bond_words * r.w; k += 256)
        old_bonds[k] = bonds[k];
    for (uint k = params.dyn_words * r.z + t; k < params.dyn_words * r.w; k += 256)
        old_dyn[k] = bond_dyn[k];
    if (params.resident == 0)
        return;
    for (uint c = r.x + t; c < r.y; c += 256)
    {
        old_ctab[params.props_base + c] = ctab[params.props_base + c];
        old_ctab[params.eload_base + 2 * c] = ctab[params.eload_base + 2 * c];
        old_ctab[params.eload_base + 2 * c + 1] = ctab[params.eload_base + 2 * c + 1];
        old_ctab[params.pulse_count_base + c] = ctab[params.pulse_count_base + c];
    }
    for (uint k = 3 * params.pulse_cap * r.x + t; k < 3 * params.pulse_cap * r.y; k += 256)
        old_ctab[params.pulse_base + k] = ctab[params.pulse_base + k];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pass_finish(uint3 id: SV_DispatchThreadID)
{
    if (control[CTL_PARENTS] == 0 || control[CTL_ABORT] != 0)
        return;
    uint live = control[CTL_NEW_ISLANDS];
    counters[0] = live;
    counters[1] = control[CTL_BODIES_AFTER];
    counters[2] = control[CTL_COLLIDERS_AFTER];
    counters[3] = control[CTL_NEXT_ID_AFTER];
    island_args[0] = uint4(max(live, 1u), 1, 1, 0);
    island_args[1] = uint4((max(live, 1u) + 63) / 64, 1, 1, 0);
    if (params.resume != 0)
    {
        status[ST_RUN] = 1;
        status[ST_DISCONNECTED] = 0;
    }
}
