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
    uint sload_base;    // scratch: the static solves' loads (2 float4 per chunk)
    uint gate_flag_base; // gate: per island, x = 1 while its stored loads are an unconverged solve's, 2 a fixed point's
    uint bond_words;    // float4 per bond record (world.slang `BondDyn`)
    uint certify;       // 1: a converged solve's repeat counts as converged (bit 6)
    uint unconverged;   // an unconverged solve: 0 restored, 1 its iterate kept, 2 kept and committed (SolverConfig::static_unconverged)
};

// Status record (uint words): 0 the next pass runs, 1 a bond disconnected (pass + 1),
// 2 passes run, 3 unconverged solves, 4 damage grew in the last pass run; per island
// from 16: bit 0 solved this pass, bit 1 converged, bit 2 damaged, bit 3 disconnected,
// bit 4 skipped as a repeat of an unconverged solve, bit 5 skipped as a repeat at a fixed
// point (of the last pass run).
//
// The repeat gate (`cascade_gate`), lossless: the solve and the commit are deterministic
// functions of an island's loads, chunk states and bond records, so a repeat of their
// inputs, bit for bit, repeats their outcome.
// - An unconverged solve leaves its island as it found it (its state restored, no
//   commit): the same loads again give the same verdict on the same state.
// - A converged solve and its commit that left the state and the bond records as they
//   found them (`cascade_fixed`): the island is at a fixed point, and the same loads
//   again leave it there.
// Either repeat is skipped. And, your decision of 2026-10-17 (a change at f32's rounding
// level, not of the tolerance): a converged solve whose commit grew no damage leaves an
// equilibrium the next solve with the same loads would only polish (a floor-certified
// solve takes one more Newton step to see its residual stop falling). That island is
// certified converged without a solve (bit 6); its commit still runs, after which it
// may be a fixed point. `certify` off (STRESS_QS_CERTIFY=0): solved again.
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
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> backup;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> status;
// The loads of each island's last unconverged solve (2 float4 per chunk), then per island
// its flag (`gate_flag_base`).
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> gate;
[[vk::binding(7, 0)]] StructuredBuffer<float4> bond_dyn;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> bond_backup;

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
    bool converged = (s & 64u) != 0 || scratch[params.result_base + i].w != 0.0;
    Island isl = islands[i];
    // The loads this solve saw, for the repeat gate.
    for (uint k = 2 * isl.range.x + thread.x; k < 2 * isl.range.y; k += 64)
        gate[k] = scratch[params.sload_base + k];
    if (converged)
    {
        // A fixed point only once its commit is known to change nothing (`cascade_fixed`);
        // until then a candidate for certification (3), unless its commit grows damage
        // (`cascade_end`).
        if (thread.x == 0)
        {
            status[ST_ISLANDS + i] = s | 2u;
            gate[params.gate_flag_base + i] = float4(params.certify != 0 ? 3.0 : 0.0, 0.0, 0.0, 0.0);
        }
        return;
    }
    if (params.unconverged != 0)
    {
        // Its iterate kept (Blast's warm start): no repeat of it stands.
        if (thread.x == 0)
            gate[params.gate_flag_base + i] = float4(0.0);
        return;
    }
    for (uint k = 4 * isl.range.x + thread.x; k < 4 * isl.range.y; k += 64)
        state[k] = backup[k];
    if (thread.x == 0)
        gate[params.gate_flag_base + i] = float4(1.0, 0.0, 0.0, 0.0);
}

// One group per island solved and committed this pass: a fixed point when the solve and
// its commit left its chunk states and bond records bit for bit as they were.
groupshared uint g_moved;

[shader("compute")]
[numthreads(64, 1, 1)]
void cascade_fixed(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    if (i >= params.island_count)
        return;
    uint s = status[ST_ISLANDS + i];
    // Only the first pass commits with the frame's duration: the fixed points a later
    // frame's first pass can rely on.
    if (status[ST_RUN] == 0 || (s & 3u) != 3u || params.pass_index != 0)
        return;
    if (thread.x == 0)
        g_moved = 0;
    GroupMemoryBarrierWithGroupSync();
    Island isl = islands[i];
    uint moved = 0;
    for (uint k = 4 * isl.range.x + thread.x; k < 4 * isl.range.y; k += 64)
    {
        if (any(asuint(state[k]) != asuint(backup[k])))
            moved = 1;
    }
    for (uint k = params.bond_words * isl.range.z + thread.x; k < params.bond_words * isl.range.w; k += 64)
    {
        if (any(asuint(bond_dyn[k]) != asuint(bond_backup[k])))
            moved = 1;
    }
    if (moved != 0)
        g_moved = 1;
    GroupMemoryBarrierWithGroupSync();
    if (thread.x == 0 && g_moved == 0)
        gate[params.gate_flag_base + i] = float4(2.0, 0.0, 0.0, 0.0);
}

// One group per island flagged for a solve: a repeat of its last unconverged solve (the
// same loads, bit for bit) is skipped.
groupshared uint g_differs;

[shader("compute")]
[numthreads(64, 1, 1)]
void cascade_gate(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    if (i >= params.island_count)
        return;
    uint s = status[ST_ISLANDS + i];
    float flag = gate[params.gate_flag_base + i].x;
    if (status[ST_RUN] == 0 || (s & 1u) == 0 || flag == 0.0 || (flag == 2.0 && params.pass_index != 0))
        return;
    if (thread.x == 0)
        g_differs = 0;
    GroupMemoryBarrierWithGroupSync();
    Island isl = islands[i];
    uint differs = 0;
    for (uint k = 2 * isl.range.x + thread.x; k < 2 * isl.range.y; k += 64)
    {
        if (any(asuint(gate[k]) != asuint(scratch[params.sload_base + k])))
            differs = 1;
    }
    if (differs != 0)
        g_differs = 1;
    GroupMemoryBarrierWithGroupSync();
    if (thread.x != 0 || g_differs != 0)
        return;
    islands[i].info.z = isl.info.z & ~ISLAND_SOLVE;
    // A certified converged repeat stays solved and converged (its commit runs).
    status[ST_ISLANDS + i] = flag == 2.0 ? 32u : (flag == 3.0 ? (64u | 1u) : 16u);
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
    // Committed: the converged solves, and with `unconverged` 2 every solve (Blast's
    // unconverged answers graded).
    if (status[ST_RUN] != 0 && ((s & 3u) == 3u || (params.unconverged == 2 && (s & 1u) != 0)))
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
        if ((s & 16u) != 0)
        {
            unconverged++;
            continue;
        }
        if ((s & 1u) == 0)
            continue;
        if ((s & 2u) == 0)
        {
            unconverged++;
            if (params.unconverged != 2)
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
        // Damage grew: its equilibrium moved; no repeat stands for it.
        if (add != 0)
            gate[params.gate_flag_base + i] = float4(0.0);
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

// One group per island flagged for a solve this pass: its chunk states and bond records
// backed up (`cascade_restore` and `cascade_fixed` read the backup only for islands
// solved), in place of a copy of the whole tables every pass.
[shader("compute")]
[numthreads(64, 1, 1)]
void cascade_backup(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    if (i >= params.island_count || status[ST_RUN] == 0)
        return;
    Island isl = islands[i];
    if ((isl.info.z & ISLAND_SOLVE) == 0)
        return;
    for (uint k = 4 * isl.range.x + thread.x; k < 4 * isl.range.y; k += 64)
        backup[k] = state[k];
    for (uint k = params.bond_words * isl.range.z + thread.x; k < params.bond_words * isl.range.w; k += 64)
        bond_backup[k] = bond_dyn[k];
}
