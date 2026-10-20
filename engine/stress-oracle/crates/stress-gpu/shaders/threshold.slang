// A deliberately simple bond rule (`Profile::Threshold`, the v0 stress mode): no solve. A
// bond breaks when the contact force on either of its chunks this frame (resident.slang's
// engine loads: the sustained contacts and each impact's Hertz peak force, `impact_peak`,
// which the profile turns on; no gravity) exceeds its
// tensile capacity (strength x area x the bond's factor, as punch.slang counts it). It
// stands in a resident cascade pass for the static solve and the damage commit, and leaves
// what they leave for the rest of the pass: the bonds' records as a break leaves them, the
// island damaged and disconnected, its solve converged (cascade_end then stops the
// cascade for the split pass).
//
//   threshold_breaks  one group per live island (indirect, `island_args`), after
//                     cascade_settle

import common;
import joint;
import bonds;

struct ThresholdParams
{
    uint island_count;
    uint eload_base;   // ctab: 2 float4 per chunk (force, moment), world frame
    uint pad0;
    uint pad1;
};

[[vk::binding(0, 0)]] ConstantBuffer<ThresholdParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<Island> islands;
[[vk::binding(2, 0)]] StructuredBuffer<BondStatic> bond_static;
[[vk::binding(3, 0)]] RWStructuredBuffer<BondDyn> bond_dyn;
[[vk::binding(4, 0)]] StructuredBuffer<float4> ctab;
struct MaterialTable
{
    JointMaterial m[64];
};
[[vk::binding(5, 0)]] ConstantBuffer<MaterialTable> materials;
// cascade.slang status: [0] the cascade runs, [16 + i] island i's bits (0 solved, 1 converged).
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> status;

static const uint ST_RUN = 0;
static const uint ST_ISLANDS = 16;
static const uint G = 256;

groupshared uint g_broke[G];

float contact_force(uint chunk)
{
    return length(ctab[params.eload_base + 2 * chunk].xyz);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void threshold_breaks(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    // The exits as uniform values: a barrier follows.
    if (group_uniform(i >= params.island_count || status[ST_RUN] == 0 || (status[ST_ISLANDS + i] & 1u) == 0 ? 1u : 0u, t) != 0)
        return;
    Island isl = islands[i];
    uint broke = 0;
    for (uint bi = isl.range.z + t; bi < isl.range.w; bi += G)
    {
        JointBond law = bond_static[bi].law;
        BondDyn dyn = bond_dyn[bi];
        if (!connected(dyn.js, law.stiff1.w != 0.0))
            continue;
        float capacity = materials.m[law.ids.x].strength.x * law.geom0.x * law.geom1.w;
        float demand = max(contact_force(law.ids.y), contact_force(law.ids.z));
        if (demand > capacity)
        {
            dyn.js.damage = 1.0;
            dyn.js.rebar_broken = 1.0;
            dyn.events.z = max(dyn.events.z, 1u);
            bond_dyn[bi] = dyn;
            broke = 1;
        }
    }
    g_broke[t] = broke;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_broke[t] |= g_broke[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t != 0)
        return;
    // Solved and converged (no iterate to restore), as the static solve would leave it.
    status[ST_ISLANDS + i] |= 2u;
    if (g_broke[0] != 0)
    {
        isl.info.z |= ISLAND_DAMAGED | ISLAND_DISCONNECTED;
        islands[i] = isl;
    }
}
