// The resident frame's static cascade under GPU control (docs/stress-fracture-gpu-resident.md
// phase 5): world.rs `fast_static_tail`'s loop of solves and damage commits encoded for
// every pass at once, each pass's verdicts deciding on the GPU whether the next runs (a
// pass that does not run flags nothing, so its kernels return at once). The host reads
// one status record at the frame's end.
//
// A pass, as `equilibrate_resident` and `commit_resident` run it from the host:
//   (copy the layout's island records)  cascade_begin   island_statics
//   cascade_restore  (copy the layout's island records)  cascade_settle  settled_fatigue
//   cascade_end
// cascade_begin flags every island with bonds for a solve while the cascade runs;
// cascade_restore puts an unconverged island's state back; cascade_settle flags the
// converged ones for the commit; cascade_end reduces the commit's flags: damage grew and
// nothing disconnected, the next pass runs; a disconnection stops the cascade for the
// host to split.

import common;

struct CascadeParams
{
    uint island_count;
    uint result_base;   // scratch: the static solves' results (one float4 per island; w converged)
    uint pass_index;
    uint pad;
};

// Status record (uint words): 0 the next pass runs, 1 a bond disconnected (pass + 1),
// 2 passes run, 3 unconverged solves, 4 damage grew in the last pass run; per island
// from 16: bit 0 solved this pass, bit 1 converged, bit 2 damaged, bit 3 disconnected
// (of the last pass run).
static const uint ST_RUN = 0;
static const uint ST_DISCONNECTED = 1;
static const uint ST_PASSES = 2;
static const uint ST_UNCONVERGED = 3;
static const uint ST_DAMAGED = 4;
static const uint ST_ISLANDS = 16;

static const uint ISLAND_SOLVE = 8;

[[vk::binding(0, 0)]] ConstantBuffer<CascadeParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<Island> islands;
[[vk::binding(2, 0)]] StructuredBuffer<float4> scratch;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> state;
[[vk::binding(4, 0)]] StructuredBuffer<float4> backup;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> status;

[shader("compute")]
[numthreads(64, 1, 1)]
void cascade_begin(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.island_count)
        return;
    bool run = status[ST_RUN] != 0;
    Island isl = islands[i];
    bool solve = run && isl.range.w > isl.range.z;
    isl.info.z = (isl.info.z & ~ISLAND_SOLVE) | (solve ? ISLAND_SOLVE : 0u);
    islands[i] = isl;
    // A pass that does not run keeps the last pass's verdicts for the host.
    if (run)
        status[ST_ISLANDS + i] = solve ? 1u : 0u;
}

// One group per island: an unconverged solve's chunk states back from the backup.
[shader("compute")]
[numthreads(64, 1, 1)]
void cascade_restore(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    if (i >= params.island_count)
        return;
    uint s = status[ST_ISLANDS + i];
    if (status[ST_RUN] == 0 || (s & 1u) == 0)
        return;
    bool converged = scratch[params.result_base + i].w != 0.0;
    if (converged)
    {
        if (thread.x == 0)
            status[ST_ISLANDS + i] = s | 2u;
        return;
    }
    Island isl = islands[i];
    for (uint k = 4 * isl.range.x + thread.x; k < 4 * isl.range.y; k += 64)
        state[k] = backup[k];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cascade_settle(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.island_count)
        return;
    uint s = status[ST_ISLANDS + i];
    Island isl = islands[i];
    isl.info.x &= ~ISLAND_SETTLED;
    isl.info.z &= ~(ISLAND_DAMAGED | ISLAND_DISCONNECTED);
    if (status[ST_RUN] != 0 && (s & 3u) == 3u)
        isl.info.x |= ISLAND_SETTLED;
    islands[i] = isl;
}

// One group: the pass's verdict (sums by a tree over the group's threads).
groupshared uint g_damaged[256];
groupshared uint g_disconnected[256];
groupshared uint g_unconverged[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cascade_end(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    if (status[ST_RUN] == 0)
        return;
    uint damaged = 0;
    uint disconnected = 0;
    uint unconverged = 0;
    for (uint i = tid; i < params.island_count; i += 256)
    {
        uint s = status[ST_ISLANDS + i];
        if ((s & 1u) == 0)
            continue;
        if ((s & 2u) == 0)
        {
            unconverged++;
            continue;
        }
        uint z = islands[i].info.z;
        uint add = 0;
        if ((z & ISLAND_DAMAGED) != 0)
        {
            add |= 4u;
            damaged = 1;
        }
        if ((z & ISLAND_DISCONNECTED) != 0)
        {
            add |= 8u;
            disconnected = 1;
        }
        status[ST_ISLANDS + i] = s | add;
    }
    g_damaged[tid] = damaged;
    g_disconnected[tid] = disconnected;
    g_unconverged[tid] = unconverged;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 128; step > 0; step >>= 1)
    {
        if (tid < step)
        {
            g_damaged[tid] |= g_damaged[tid + step];
            g_disconnected[tid] |= g_disconnected[tid + step];
            g_unconverged[tid] += g_unconverged[tid + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid != 0)
        return;
    status[ST_PASSES] = status[ST_PASSES] + 1;
    status[ST_UNCONVERGED] = status[ST_UNCONVERGED] + g_unconverged[0];
    status[ST_DAMAGED] = g_damaged[0];
    if (g_disconnected[0] != 0)
        status[ST_DISCONNECTED] = params.pass_index + 1;
    status[ST_RUN] = (g_damaged[0] != 0 && g_disconnected[0] == 0) ? 1u : 0u;
}
