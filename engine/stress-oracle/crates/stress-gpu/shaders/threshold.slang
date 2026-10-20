// The v0 stress mode's bond rules (`Profile::Threshold`): deliberately simple and fast, no
// stress solve, forward only (nothing is re-simulated).
//
//   threshold_predict  at the end of a resident frame, after routing.slang's check and
//                      punch.slang `punch_fast` (the fast free islands, the check's flag):
//                      a strike within the next frame (strike.slang), its force the impulse
//                      of an inelastic hit over the Hertz contact time, breaks every intact
//                      bond of the struck island whose tensile capacity is below the force
//                      reaching it; the island is marked disconnected for the split pass
//                      recorded after it, so the engine's next step meets the pieces;
//   threshold_breaks   in a resident cascade pass, in place of the static solve and the
//                      damage commit: a bond breaks when the contact force on either of its
//                      chunks this frame (resident.slang's engine loads: the sustained
//                      contacts and each impact's Hertz peak force, `impact_peak`, which the
//                      profile turns on; no gravity) exceeds its tensile capacity (debris,
//                      settling). It leaves what the solve and commit leave for the rest of
//                      the pass: the bonds' records as a break leaves them, the island
//                      damaged and disconnected, its solve converged (cascade_end then stops
//                      the cascade for the split pass).
// A bond's tensile capacity: strength x area x the bond's factor (as punch.slang counts it).
// One group per live island each (indirect, `island_args`).

import common;
import joint;
import bonds;

struct ThresholdParams
{
    uint island_count;
    uint eload_base;     // ctab: 2 float4 per chunk (force, moment), world frame
    // 1: decide nothing new; an island with a broken bond still in it (its break not yet
    // split) is disconnected (`STRESS_CORRECTION_LIMIT=1`'s split at the frame's start,
    // world.rs `correct_frame`; a measurement, not the profile's).
    uint pending;
    uint impactor_count; // threshold_predict: the impactors (strike.slang)
    uint engine_base;    // ctab: per island 3 float4 (resident.slang)
    uint pad0;
    float dt;            // the next frame
    float speed;         // the tiered impact speed (a slower closing is no strike)
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
// cascade.slang status: [0] the cascade runs, [1] the pass that disconnected, [16 + i]
// island i's bits (0 solved, 1 converged, 3 disconnected).
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> status;
// threshold_predict's (as punch.slang's): the chunk statics and states, the engine's bodies,
// the impactors (two float4 each: engine body as bits, bounding radius, face radius, mass;
// plane-strain modulus), punch.slang `punch_fast`'s list.
[[vk::binding(7, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(8, 0)]] StructuredBuffer<float4> state;
[[vk::binding(9, 0)]] StructuredBuffer<EngineBody> bodies;
[[vk::binding(10, 0)]] StructuredBuffer<float4> impactors;
[[vk::binding(11, 0)]] StructuredBuffer<float4> fast;

static const uint NONE = 0xffffffffu;
static const uint ST_DISCONNECTED = 1;
static const uint DISCONNECTED_BIT = 8;
static const uint ST_RUN = 0;
static const uint ST_ISLANDS = 16;
static const uint G = 256;

#include "strike.slang"

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
        {
            if (params.pending != 0)
                broke = 1;
            continue;
        }
        if (params.pending != 0)
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

[shader("compute")]
[numthreads(256, 1, 1)]
void threshold_predict(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    if (i >= params.island_count)
        return;
    // No impactor near anything this frame (routing.slang's check): nothing to predict.
    if (group_uniform(asuint(fast[0].y), t) == 0)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    // (Its result and the island's chunk count as uniform values: WGSL cannot see that a
    // function's results are; barriers follow.)
    Strike st;
    bool found = find_strike(i, t, NONE, st);
    uint n = group_uniform(st.n, t);
    if (group_uniform(found ? 1u : 0u, t) == 0)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    // The island's bounds are its chunks' bounding spheres (loose for a long panel): the
    // projectile must reach a chunk's box within the frame; the force spreads from where it
    // meets the first one it reaches.
    float inf = float_bits(0x7f800000u);
    float first = inf;
    for (uint c = t; c < n; c += G)
    {
        uint k = st.isl.range.x + c;
        float3 cc = st.origin + rotate(st.q, chunks[k].center.xyz + state[4 * k].xyz);
        float t_in = 0.0;
        float t_out = params.dt;
        for (uint ax = 0; ax < 3; ax++)
        {
            float3 axis = rotate(st.q, float3(chunks[k].crot0[ax], chunks[k].crot1[ax], chunks[k].crot2[ax]));
            float o = dot(st.from - cc, axis);
            float dv = dot(st.rel, axis);
            float ext = chunks[k].half[ax] + st.radius;
            if (abs(dv) < 1e-12)
            {
                if (abs(o) > ext)
                    t_in = inf;
                continue;
            }
            float ta = (-ext - o) / dv;
            float tb = (ext - o) / dv;
            t_in = max(t_in, min(ta, tb));
            t_out = min(t_out, max(ta, tb));
        }
        if (t_in <= t_out)
            first = min(first, t_in);
    }
    g_a[t] = first;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_a[t] = min(g_a[t], g_a[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float t_in = g_a[0];
    if (group_uniform(t_in < inf ? 0u : 1u, t) != 0)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    float3 contact = st.from + st.rel * t_in + st.d * st.radius;
    bool anchored = (st.isl.info.x & ISLAND_ANCHORED) != 0;
    float m_isl = st.isl.com.w;
    float m_eff = anchored ? st.m_imp : st.m_imp * m_isl / (st.m_imp + m_isl);
    float modulus = 1.0 / (1.0 / st.e_imp + 1.0 / chunks[st.entry_k].cmat.x);
    float face = max(st.face, 1e-3);
    // api.rs `hertz_duration`, as `punch_islands` takes it.
    float tau = 2.87 * pow(m_eff * m_eff / (face * modulus * modulus * max(st.speed, 1e-3)), 0.2);
    float force = m_eff * st.speed / tau;
    uint broke = 0;
    for (uint bi = st.isl.range.z + t; bi < st.isl.range.w; bi += G)
    {
        JointBond law = bond_static[bi].law;
        BondDyn dyn = bond_dyn[bi];
        if (!connected(dyn.js, law.stiff1.w != 0.0))
            continue;
        float3 pa = st.origin + rotate(st.q, chunks[law.ids.y].center.xyz + state[4 * law.ids.y].xyz);
        float3 pb = st.origin + rotate(st.q, chunks[law.ids.z].center.xyz + state[4 * law.ids.z].xyz);
        float dist = length(0.5 * (pa + pb) - contact);
        // A point load on a half-space: stress F / (2 pi d^2) at distance d, so the bond's
        // area carries that much of it (all of it close to the contact).
        float reach = force * min(1.0, law.geom0.x / (6.28318530718 * max(dist * dist, 1e-12)));
        float capacity = materials.m[law.ids.x].strength.x * law.geom0.x * law.geom1.w;
        if (reach > capacity)
        {
            dyn.js.damage = 1.0;
            dyn.js.rebar_broken = 1.0;
            dyn.events.z = max(dyn.events.z, 1u);
            bond_dyn[bi] = dyn;
            broke = 1;
        }
    }
    g_idx[t] = broke;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_idx[t] |= g_idx[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t != 0)
        return;
    if (g_idx[0] != 0)
    {
        status[ST_ISLANDS + i] |= DISCONNECTED_BIT;
        status[ST_DISCONNECTED] = 1;
    }
    else
    {
        status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
    }
}
