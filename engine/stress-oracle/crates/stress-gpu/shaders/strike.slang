// A projectile's strike on an island within the next frame, for the kernels that decide
// a strike's breaks before the engine steps (forward only): punch.slang `punch_islands`
// (the GPU punch) and threshold.slang `threshold_predict` (the threshold profile).
// Included (not imported): the including shader declares `params` (`impactor_count`,
// `engine_base`, `dt`, `speed`), `islands`, `chunks`, `state`, `ctab`, `bodies`,
// `impactors` and `fast` (punch.slang `punch_fast`'s list), as punch.slang documents them,
// and `NONE` and `G` (the group's threads).
//
// Frames: the engine is Y-up, the stress solver Z-up: (x, y, z) -> (x, -z, y).

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

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

groupshared float g_a[G];
groupshared uint g_idx[G];
groupshared float3 g_lo[G];
groupshared float3 g_hi[G];

// The strike on island `i` within the next frame (`punch_islands`, `threshold_predict`):
// the first projectile that reaches its bounds closing faster than the tiered impact
// speed (the impactors, then the fast free islands, each other island but this one), and
// the struck chunk (the nearest to the projectile's path). False: none (or nothing to
// break: no bonds, more than `max_chunks` chunks, no impactor). The whole group calls it.
struct Strike
{
    Island isl;
    uint n;
    float3 origin;
    Quat q;
    float3 hit;      // where the projectile meets the island's bounds
    float3 d;        // its direction relative to the island
    float speed;     // its closing speed
    float face;      // its face radius
    float m_imp;
    float e_imp;     // its plane-strain modulus
    uint entry;      // the struck chunk (in the island)
    uint entry_k;    // and in the layout
    float entry_depth;
    float3 from;     // the projectile's centre at the frame's start
    float3 rel;      // its velocity relative to the island
    float radius;    // its bounding radius
};

bool find_strike(uint i, uint t, uint max_chunks, out Strike st)
{
    st.isl = islands[i];
    st.n = 0;
    st.origin = float3(0.0);
    st.q = quat_of(float4(0.0, 0.0, 0.0, 1.0));
    st.hit = float3(0.0);
    st.d = float3(0.0);
    st.speed = 0.0;
    st.face = 0.0;
    st.m_imp = 0.0;
    st.e_imp = 0.0;
    st.entry = NONE;
    st.entry_k = NONE;
    st.entry_depth = 0.0;
    st.from = float3(0.0);
    st.rel = float3(0.0);
    st.radius = 0.0;
    Island isl = islands[i];
    uint n = group_uniform(isl.range.y - isl.range.x, t);
    // No bonds (blank records too), or more chunks than the group's labels hold: nothing
    // to punch; the island's split bit cleared (any of an earlier pass's split is followed).
    if (group_uniform(isl.range.z >= isl.range.w || n > max_chunks || params.impactor_count == 0 ? 1u : 0u, t) != 0)
        return false;
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
    float inf = float_bits(0x7f800000u);
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
    // The first projectile that reaches it within the frame, closing fast enough (the
    // same for every thread: uniform data): the impactors, then the fast free islands
    // (`punch_fast`'s list: debris, a plug punched out), each other island but this one.
    uint hit_k = NONE;
    float3 hit = float3(0.0);
    float3 d = float3(0.0);
    float speed = 0.0;
    float p_radius = 0.0;
    float face = 0.0;
    float m_imp = 0.0;
    float e_imp = 0.0;
    float3 hit_c = float3(0.0);
    float3 hit_rel = float3(0.0);
    uint fast_count = asuint(fast[0].x);
    for (uint k = 0; k < params.impactor_count + fast_count && hit_k == NONE; k++)
    {
        float3 c;
        float3 rel;
        float4 info;
        float r;
        if (k < params.impactor_count)
        {
            float4 a = impactors[2 * k];
            EngineBody b = bodies[asuint(a.x)];
            c = to_zup(b.pos_inv_mass.xyz);
            rel = to_zup(b.vel_sleep.xyz) - v;
            r = a.y;
            info = float4(a.z, a.w, impactors[2 * k + 1].x, 0.0);
        }
        else
        {
            uint f = 1 + 3 * (k - params.impactor_count);
            if (asuint(fast[f + 1].w) == i)
                continue;
            c = fast[f].xyz;
            r = fast[f].w;
            rel = fast[f + 1].xyz - v;
            info = fast[f + 2];
        }
        float sp = length(rel);
        if (!(sp > params.speed))
            continue;
        float3 elo = lo - float3(r);
        float3 ehi = hi + float3(r);
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
            hit_c = c;
            hit_rel = rel;
            hit = c + rel * t0;
            d = rel / sp;
            speed = sp;
            p_radius = r;
            face = info.x;
            m_imp = info.y;
            e_imp = info.z;
        }
    }
    if (group_uniform(hit_k == NONE ? 1u : 0u, t) != 0)
        return false;
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
    st.isl = isl;
    st.n = n;
    st.origin = origin;
    st.q = q;
    st.hit = hit;
    st.d = d;
    st.speed = speed;
    st.face = face;
    st.m_imp = m_imp;
    st.e_imp = e_imp;
    st.entry = entry;
    st.entry_k = entry_k;
    st.entry_depth = entry_depth;
    st.from = hit_c;
    st.rel = hit_rel;
    st.radius = p_radius;
    return true;
}

