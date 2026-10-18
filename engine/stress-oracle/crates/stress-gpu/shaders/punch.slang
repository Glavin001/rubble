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

struct EngineBody
{
    float4 pos_inv_mass;
    float4 rot;
    float4 inv_inertia_flags;
    float4 vel_sleep;
    float4 ang_vel_gen;
    float4 initial_pos;
    float4 initial_rot;
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

static const uint NONE = 0xffffffffu;
static const uint ST_DISCONNECTED = 1;
static const uint ST_ISLANDS = 16;
static const uint DISCONNECTED_BIT = 8;
static const uint MAX_CHUNKS = 4096;
static const uint G = 256;

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

groupshared uint g_plug[MAX_CHUNKS];
groupshared float g_a[G];
groupshared float g_b[G];
groupshared uint g_idx[G];
groupshared float3 g_lo[G];
groupshared float3 g_hi[G];
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
    Island isl = islands[i];
    uint n = isl.range.y - isl.range.x;
    // No bonds (blank records too), or more chunks than the group's labels hold: nothing
    // to punch; the island's split bit cleared (any of an earlier pass's split is followed).
    if (isl.range.z >= isl.range.w || n > MAX_CHUNKS || params.impactor_count == 0)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    // Its pose and motion (routing.slang `route_islands`).
    uint body = asuint(ctab[params.engine_base + 3 * i].x);
    Quat q = quat_of(isl.rotation);
    float3 origin = isl.position.xyz;
    float3 v = isl.velocity.xyz;
    if (body != NONE)
    {
        Quat p_inv = quat_of(ctab[params.engine_base + 3 * i + 1]);
        float3 com = ctab[params.engine_base + 3 * i + 2].xyz;
        EngineBody b = bodies[body];
        float h = 0.70710678118;
        Quat a_inv;
        a_inv.w = h;
        a_inv.x = h;
        a_inv.y = 0.0;
        a_inv.z = 0.0;
        q = normalized(quat_mul(quat_mul(a_inv, quat_of(b.rot)), p_inv));
        origin = to_zup(b.pos_inv_mass.xyz) - rotate(q, com);
        v = to_zup(b.vel_sleep.xyz);
    }
    // The island's bounds (chunk centres +- bounding radius).
    float inf = asfloat(0x7f800000u);
    float3 lo = float3(inf);
    float3 hi = float3(-inf);
    for (uint c = t; c < n; c += G)
    {
        uint k = isl.range.x + c;
        float3 p = origin + rotate(q, chunks[k].center.xyz + state[4 * k].xyz);
        lo = min(lo, p - chunks[k].half.w);
        hi = max(hi, p + chunks[k].half.w);
    }
    g_lo[t] = lo;
    g_hi[t] = hi;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            g_lo[t] = min(g_lo[t], g_lo[t + s]);
            g_hi[t] = max(g_hi[t], g_hi[t + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    lo = g_lo[0];
    hi = g_hi[0];
    // The first impactor that reaches it within the frame, closing fast enough (the same
    // for every thread: uniform data).
    uint hit_k = NONE;
    float3 hit = float3(0.0);
    float3 d = float3(0.0);
    float speed = 0.0;
    for (uint k = 0; k < params.impactor_count && hit_k == NONE; k++)
    {
        float4 a = impactors[2 * k];
        EngineBody b = bodies[asuint(a.x)];
        float3 c = to_zup(b.pos_inv_mass.xyz);
        float3 rel = to_zup(b.vel_sleep.xyz) - v;
        float sp = length(rel);
        if (!(sp > params.speed))
            continue;
        float3 elo = lo - float3(a.y);
        float3 ehi = hi + float3(a.y);
        float t0 = 0.0;
        float t1 = params.dt;
        for (uint ax = 0; ax < 3; ax++)
        {
            if (abs(rel[ax]) < 1e-12)
            {
                if (c[ax] < elo[ax] || c[ax] > ehi[ax])
                    t0 = inf;
                continue;
            }
            float ta = (elo[ax] - c[ax]) / rel[ax];
            float tb = (ehi[ax] - c[ax]) / rel[ax];
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
        if (t0 <= t1)
        {
            hit_k = k;
            hit = c + rel * t0;
            d = rel / sp;
            speed = sp;
        }
    }
    if (hit_k == NONE)
    {
        if (t == 0)
            status[ST_ISLANDS + i] &= ~DISCONNECTED_BIT;
        return;
    }
    float4 imp_a = impactors[2 * hit_k];
    float4 imp_b = impactors[2 * hit_k + 1];
    float face = imp_a.z;
    // The struck chunk: the nearest to the impactor's path (ties: the lower index).
    float best = inf;
    uint best_c = NONE;
    for (uint c = t; c < n; c += G)
    {
        uint k = isl.range.x + c;
        float3 r = origin + rotate(q, chunks[k].center.xyz + state[4 * k].xyz) - hit;
        float along = dot(r, d);
        float dist = length(r - d * along) + abs(along) * 1e-3;
        if (dist < best)
        {
            best = dist;
            best_c = c;
        }
    }
    g_a[t] = best;
    g_idx[t] = best_c;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s && (g_a[t + s] < g_a[t] || (g_a[t + s] == g_a[t] && g_idx[t + s] < g_idx[t])))
        {
            g_a[t] = g_a[t + s];
            g_idx[t] = g_idx[t + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint entry = g_idx[0];
    uint entry_k = isl.range.x + entry;
    float entry_depth = dot(origin + rotate(q, chunks[entry_k].center.xyz + state[4 * entry_k].xyz) - hit, d);
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
        if (g_changed == 0)
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
        float m_imp = imp_a.w;
        float m_eff = m_plug * m_imp / (m_plug + m_imp);
        float impulse = m_eff * speed;
        float modulus = 1.0 / (1.0 / imp_b.x + 1.0 / chunks[entry_k].cmat.x);
        // api.rs `hertz_duration`.
        float tau = 2.87 * pow(m_eff * m_eff / (face * modulus * modulus * max(speed, 1e-3)), 0.2);
        bool out = plugged < float(n) && impulse / tau > g_b[0];
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
    if (g_go == 0)
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
