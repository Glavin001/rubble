// The predictive punch on the GPU (`STRESS_IMPACT_PUNCH=gpu`, a model under measurement;
// engine/docs/stress-baseline-goal.md, M5): at the end of a resident frame, in its
// submission, for each island an impactor reaches within the next frame closing faster
// than the tiered impact speed, the plug its face would punch out and whether it breaks out
// (world_coupling.rs `predict_punch`, in f32):
//   - the struck chunk: the island's nearest to the impactor's path;
//   - the plug: the chunks whose centres lie in a 45° cone from the face's edge, from the
//     struck chunk's depth on, connected to the struck chunk by intact bonds (labels
//     propagated in group memory);
//   - it breaks out when the force of an inelastic hit on it (its impulse over the Hertz
//     contact time) beats the summed tensile capacity of the intact bonds holding it in.
// A plug that breaks out has those bonds broken and its island marked disconnected in the
// cascade's status; the split pass recorded after this kernel splits it, in the same
// submission, so the next frame's engine step meets the free plug.
//
//   punch_islands  one group per live island (indirect, `island_args`)
//
// Frames: the engine is Y-up, the stress solver Z-up: (x, y, z) -> (x, -z, y).

import common;
import joint;
import bonds;

struct PunchParams
{
    uint island_count;   // dispatch bound
    uint impactor_count;
    uint engine_base;    // ctab: per island 3 float4 (resident.slang)
    uint pad0;
    float dt;            // the next frame
    float speed;         // the tiered impact speed
    float pad1;
    float pad2;
};


[[vk::binding(0, 0)]] ConstantBuffer<PunchParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(2, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(3, 0)]] StructuredBuffer<float4> state;
[[vk::binding(4, 0)]] StructuredBuffer<float4> ctab;
[[vk::binding(5, 0)]] StructuredBuffer<EngineBody> bodies;
// Per impactor two float4: (engine body as bits, bounding radius, face radius, mass),
// (plane-strain modulus as the solver scales it, -, -, -).
[[vk::binding(6, 0)]] StructuredBuffer<float4> impactors;
[[vk::binding(7, 0)]] StructuredBuffer<BondStatic> bond_static;
[[vk::binding(8, 0)]] RWStructuredBuffer<BondDyn> bond_dyn;
struct MaterialTable
{
    JointMaterial m[64];
};
[[vk::binding(9, 0)]] ConstantBuffer<MaterialTable> materials;
// cascade.slang status: [1] the pass that disconnected (0: none), [16 + i] island i's bits.
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> status;
// The fast free islands (`punch_fast`): [0].x their count (bits), [0].y the routing flag
// (bits, `route[0]`); per island three float4:
// (centre, bounding radius), (velocity, island as bits), (face radius, mass, modulus, -).
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> fast;
// routing.slang's flag of this frame: [0] 1 when an impactor may reach an island (a
// superset of a punch's conditions); 0: no island to look at.
[[vk::binding(12, 0)]] StructuredBuffer<uint> route;

static const uint NONE = 0xffffffffu;
static const uint ST_DISCONNECTED = 1;
static const uint ST_ISLANDS = 16;
static const uint DISCONNECTED_BIT = 8;
static const uint MAX_CHUNKS = 4096;
static const uint G = 256;

#include "strike.slang"


groupshared uint g_plug[MAX_CHUNKS];
groupshared float g_b[G];
groupshared uint g_changed;
groupshared uint g_go;

[shader("compute")]
[numthreads(256, 1, 1)]
void punch_islands(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    if (i >= params.island_count)
        return;
    // No impactor near anything this frame (the routing check, just before): no punch; an
    // earlier pass's split bits are left to the split pass (its parents are laid out).
    // (Exits, loop bounds and stops as uniform values below: barriers follow.)
    if (group_uniform(asuint(fast[0].y), t) == 0)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    // No strike (or nothing to punch: no bonds, more chunks than the group's labels hold):
    // the island's split bit cleared (any of an earlier pass's split is followed).
    // (Its result and the island's chunk count as uniform values: WGSL cannot see that a
    // function's results are; barriers follow.)
    Strike st;
    bool found = find_strike(i, t, MAX_CHUNKS, st);
    if (group_uniform(found ? 1u : 0u, t) == 0)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    Island isl = st.isl;
    uint n = group_uniform(st.n, t);
    float3 origin = st.origin;
    Quat q = st.q;
    float3 hit = st.hit;
    float3 d = st.d;
    float speed = st.speed;
    float face = st.face;
    float m_imp = st.m_imp;
    float e_imp = st.e_imp;
    uint entry = st.entry;
    uint entry_k = st.entry_k;
    float entry_depth = st.entry_depth;
    // The plug's labels: the struck chunk, then the cone's chunks bonded to a plug chunk.
    for (uint c = t; c < n; c += G)
        g_plug[c] = c == entry ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint iter = 0; iter < n; iter++)
    {
        if (t == 0)
            g_changed = 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint bi = isl.range.z + t; bi < isl.range.w; bi += G)
        {
            JointBond law = bond_static[bi].law;
            if (!connected(bond_dyn[bi].js, law.stiff1.w != 0.0))
                continue;
            uint ca = law.ids.y - isl.range.x;
            uint cb = law.ids.z - isl.range.x;
            if (g_plug[ca] == g_plug[cb])
                continue;
            uint o = g_plug[ca] != 0 ? cb : ca;
            uint ok = isl.range.x + o;
            float3 r = origin + rotate(q, chunks[ok].center.xyz + state[4 * ok].xyz) - hit;
            float along = dot(r, d);
            float lateral = length(r - d * along);
            float h = chunks[ok].half.w;
            if (along >= entry_depth - h && lateral <= face + max(along - entry_depth, 0.0))
            {
                g_plug[o] = 1;
                g_changed = 1;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (group_uniform(g_changed, t) == 0)
            break;
        GroupMemoryBarrierWithGroupSync();
    }
    // The plug's mass and chunks, and the capacity of the intact bonds holding it in.
    float mass = 0.0;
    float count = 0.0;
    for (uint c = t; c < n; c += G)
    {
        if (g_plug[c] != 0)
        {
            mass += chunks[isl.range.x + c].center.w;
            count += 1.0;
        }
    }
    float capacity = 0.0;
    for (uint bi = isl.range.z + t; bi < isl.range.w; bi += G)
    {
        JointBond law = bond_static[bi].law;
        if (!connected(bond_dyn[bi].js, law.stiff1.w != 0.0))
            continue;
        if (g_plug[law.ids.y - isl.range.x] != g_plug[law.ids.z - isl.range.x])
            capacity += materials.m[law.ids.x].strength.x * law.geom0.x * law.geom1.w;
    }
    g_a[t] = mass + 0.0 * count;
    g_b[t] = capacity;
    g_lo[t] = float3(count, 0.0, 0.0);
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            g_a[t] += g_a[t + s];
            g_b[t] += g_b[t + s];
            g_lo[t] += g_lo[t + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        float m_plug = g_a[0];
        float plugged = g_lo[0].x;
        float m_eff = m_plug * m_imp / (m_plug + m_imp);
        float impulse = m_eff * speed;
        float modulus = 1.0 / (1.0 / e_imp + 1.0 / chunks[entry_k].cmat.x);
        // api.rs `hertz_duration`.
        float tau = 2.87 * pow(m_eff * m_eff / (face * modulus * modulus * max(speed, 1e-3)), 0.2);
        // A plug held in by no intact bond is no break: nothing to split (a split that
        // makes one child would only remake the island's body).
        bool out = plugged < float(n) && g_b[0] > 0.0 && impulse / tau > g_b[0];
        g_go = out ? 1u : 0u;
        if (out)
        {
            status[ST_ISLANDS + i] |= DISCONNECTED_BIT;
            status[ST_DISCONNECTED] = 1;
        }
        else
        {
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (group_uniform(g_go, t) == 0)
        return;
    // The plug breaks out: its bonds broken (their records as a break leaves them).
    for (uint bi = isl.range.z + t; bi < isl.range.w; bi += G)
    {
        JointBond law = bond_static[bi].law;
        BondDyn dyn = bond_dyn[bi];
        if (!connected(dyn.js, law.stiff1.w != 0.0))
            continue;
        if (g_plug[law.ids.y - isl.range.x] != g_plug[law.ids.z - isl.range.x])
        {
            dyn.js.damage = 1.0;
            dyn.js.rebar_broken = 1.0;
            dyn.events.z = max(dyn.events.z, 1u);
            bond_dyn[bi] = dyn;
        }
    }
}

// The fast free islands as projectiles for `punch_islands`: one group lists, in island
// order, every free island (not anchored, an engine body, at most FAST_CHUNKS chunks)
// whose body moves faster than the tiered impact speed, up to MAX_FAST of them: its
// centre and bounding radius, velocity, its face radius (its cross-section across its
// motion), mass and its first chunk's modulus.
static const uint MAX_FAST = 128;
static const uint FAST_CHUNKS = 64;
groupshared uint g_scan[1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void punch_fast(uint3 thread: SV_GroupThreadID)
{
    uint t = thread.x;
    uint carry = 0;
    for (uint base = 0; base < params.island_count; base += 1024)
    {
        uint i = base + t;
        bool fast_i = false;
        float3 c = float3(0.0);
        float3 vel = float3(0.0);
        float radius = 0.0;
        float face_r = 0.0;
        Island isl;
        if (i < params.island_count && route[0] != 0)
        {
            isl = islands[i];
            uint body = asuint(ctab[params.engine_base + 3 * i].x);
            uint n = isl.range.y - isl.range.x;
            if (n > 0 && n <= FAST_CHUNKS && body != NONE && (isl.info.x & ISLAND_ANCHORED) == 0)
            {
                EngineBody b = bodies[body];
                vel = to_zup(b.vel_sleep.xyz);
                if (length(vel) > params.speed)
                {
                    fast_i = true;
                    c = to_zup(b.pos_inv_mass.xyz);
                    Quat p_inv = quat_of(ctab[params.engine_base + 3 * i + 1]);
                    float3 com = ctab[params.engine_base + 3 * i + 2].xyz;
                    float h = 0.70710678118;
                    Quat a_inv;
                    a_inv.w = h;
                    a_inv.x = h;
                    a_inv.y = 0.0;
                    a_inv.z = 0.0;
                    Quat q = normalized(quat_mul(quat_mul(a_inv, quat_of(b.rot)), p_inv));
                    float3 origin = c - rotate(q, com);
                    // Its bounding radius, and its face: the radius of its cross-section
                    // across its motion (chunk spheres projected on that plane).
                    float3 dir = normalize(vel);
                    for (uint k = isl.range.x; k < isl.range.y; k++)
                    {
                        float3 r = origin + rotate(q, chunks[k].center.xyz + state[4 * k].xyz) - c;
                        radius = max(radius, length(r) + chunks[k].half.w);
                        face_r = max(face_r, length(r - dir * dot(r, dir)) + chunks[k].half.w);
                    }
                }
            }
        }
        g_scan[t] = fast_i ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < 1024; step <<= 1)
        {
            uint add = t >= step ? g_scan[t - step] : 0;
            GroupMemoryBarrierWithGroupSync();
            g_scan[t] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        uint at = carry + g_scan[t] - (fast_i ? 1u : 0u);
        if (fast_i && at < MAX_FAST)
        {
            fast[1 + 3 * at] = float4(c, radius);
            fast[2 + 3 * at] = float4(vel, asfloat(i));
            fast[3 + 3 * at] = float4(face_r, isl.com.w, chunks[isl.range.x].cmat.x, 0.0);
        }
        uint total = g_scan[1023];
        GroupMemoryBarrierWithGroupSync();
        carry += total;
    }
    // y: the routing check's flag, for `punch_islands` (so it binds one buffer fewer: 10).
    if (t == 0)
        fast[0] = float4(asfloat(min(carry, MAX_FAST)), asfloat(route[0]), 0.0, 0.0);
}
