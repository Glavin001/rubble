// ---------------------------------------------------------------------------

// Closest point on triangle abc to p (Ericson 5.1.5); w = barycentrics.
float3 closestOnTriangle(float3 p, float3 a, float3 b, float3 c, out float3 w)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { w = float3(1, 0, 0); return a; }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { w = float3(0, 1, 0); return b; }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float v = d1 / (d1 - d3);
        w = float3(1.0 - v, v, 0.0);
        return a + v * ab;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { w = float3(0, 0, 1); return c; }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float t = d2 / (d2 - d6);
        w = float3(1.0 - t, 0.0, t);
        return a + t * ac;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        w = float3(0.0, 1.0 - t, t);
        return b + t * (c - b);
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom, u = vc * denom;
    w = float3(1.0 - v - u, v, u);
    return a + ab * v + ac * u;
}

// Closest points of segments p1q1 and p2q2 (Ericson 5.1.9): s, t parameters.
void closestSegSeg(float3 p1, float3 q1, float3 p2, float3 q2, out float s, out float t)
{
    float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    if (a <= 1e-12 && e <= 1e-12) { s = 0.0; t = 0.0; return; }
    if (a <= 1e-12) { s = 0.0; t = clamp(f / e, 0.0, 1.0); return; }
    float c = dot(d1, r);
    if (e <= 1e-12) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); return; }
    float b = dot(d1, d2);
    float denom = a * e - b * b;
    s = denom > 1e-12 ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
    t = (b * s + f) / e;
    if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
    else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
}

// Closest points between segment p0p1 and triangle abc: q on the triangle
// (barycentrics w), cs on the segment.
void closestSegTriangle(float3 p0, float3 p1, float3 a, float3 b, float3 c, out float3 q, out float3 cs, out float3 w)
{
    float best = 1e30;
    float3 wq;
    float3 qq = closestOnTriangle(p0, a, b, c, wq);
    best = dot(qq - p0, qq - p0); q = qq; cs = p0; w = wq;
    qq = closestOnTriangle(p1, a, b, c, wq);
    float d = dot(qq - p1, qq - p1);
    if (d < best) { best = d; q = qq; cs = p1; w = wq; }
    float3 ea[3] = { a, b, c };
    float3 eb[3] = { b, c, a };
    [unroll]
    for (uint k = 0u; k < 3u; k++)
    {
        float s, t;
        closestSegSeg(p0, p1, ea[k], eb[k], s, t);
        float3 x = p0 + (p1 - p0) * s;
        float3 y = ea[k] + (eb[k] - ea[k]) * t;
        d = dot(x - y, x - y);
        if (d < best)
        {
            best = d; q = y; cs = x;
            w = k == 0u ? float3(1.0 - t, t, 0.0) : (k == 1u ? float3(0.0, 1.0 - t, t) : float3(t, 0.0, 1.0 - t));
        }
    }
    // The segment piercing the triangle.
    float3 n = cross(b - a, c - a);
    float d0 = dot(p0 - a, n), d1 = dot(p1 - a, n);
    if (d0 * d1 < 0.0)
    {
        float3 x = p0 + (p1 - p0) * (d0 / (d0 - d1));
        float3 wx;
        float3 qx = closestOnTriangle(x, a, b, c, wx);
        if (dot(qx - x, qx - x) < 1e-10) { q = qx; cs = x; w = wx; }
    }
}

uint triNode(uint t, uint k) { return SU(softTriBase() + t)[k]; }

// Per soft body: bounds of its particles at the initial guess, padded by this
// step's travel, for culling contact pairs.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_bounds(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint s = softGroupBody(gid), local = lid.x;
    bool live = s < softBodyCount();
    uint4 r = live ? softBodyRec(s) : uint4(0, 0, 0, 0);
    float3 mn = float3(1e30, 1e30, 1e30), mx = float3(-1e30, -1e30, -1e30);
    float travel = 0.0;
    for (uint k = local; k < r.y; k += 64u)
    {
        float3 x = gBodies[r.x + k].position;
        mn = min(mn, x);
        mx = max(mx, x);
        travel = max(travel, length(x - gBodies[r.x + k].initialPos));
    }
    gsBodyLo[local] = float4(mn, travel);
    gsBodyHi[local] = float4(mx, 0.0);
    GroupMemoryBarrierWithGroupSync();
    if (!live || local != 0u) return;
    for (uint k = 1u; k < 64u; k++)
    {
        mn = min(mn, gsBodyLo[k].xyz);
        mx = max(mx, gsBodyHi[k].xyz);
        travel = max(travel, gsBodyLo[k].w);
    }
    float pad = 2.0 * travel + 0.1;
    gSoftData[softAabbBase() + 2u * s] = float4(mn - pad, 0.0);
    gSoftData[softAabbBase() + 2u * s + 1u] = float4(mx + pad, 0.0);
}

// Per soft body, after soft_bounds: the soft bodies (itself included, ascending)
// whose bounds come within both skins + 0.3 m of its own. Every later test that
// walks other bodies (node contacts, a triangle's nodes, rigid-mode promotion)
// asks for less, so walking this list instead of every body finds the same.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_pairs(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    uint nb = softBodyCount();
    if (s >= nb) return;
    float m = asfloat(softBodyRec2(s).z) + 0.3;
    float3 lo = SD(softAabbBase() + 2u * s).xyz - m, hi = SD(softAabbBase() + 2u * s + 1u).xyz + m;
    uint base = softPairBase() + SOFT_PAIR_STRIDE * s;
    uint n = 0u;
    uint4 slot = uint4(0, 0, 0, 0);
    for (uint s2 = 0u; s2 < nb && n <= SOFT_PAIR_MAX; s2++)
    {
        float m2 = asfloat(softBodyRec2(s2).z);
        float3 lo2 = SD(softAabbBase() + 2u * s2).xyz - m2, hi2 = SD(softAabbBase() + 2u * s2 + 1u).xyz + m2;
        if (s2 != s && !(all(lo <= hi2) && all(lo2 <= hi))) continue;
        n++;
        if (n > SOFT_PAIR_MAX) break;
        slot[n & 3u] = s2;
        if ((n & 3u) == 3u) { gSoftData[base + (n >> 2)] = asfloat(slot); slot = uint4(0, 0, 0, 0); }
    }
    if ((n & 3u) != 3u && n <= SOFT_PAIR_MAX) gSoftData[base + (n >> 2)] = asfloat(slot);
    uint4 head = SU(base);
    head.x = n > SOFT_PAIR_MAX ? SOFT_PAIR_ALL : n;
    gSoftData[base] = asfloat(head);
}

bool softAabbHas(uint s, float3 p)
{
    float3 mn = SD(softAabbBase() + 2u * s).xyz, mx = SD(softAabbBase() + 2u * s + 1u).xyz;
    return all(p >= mn) && all(p <= mx);
}

// World AABB of a soft obstacle: sphere, capsule, oriented box (other shapes:
// the bounding radius in params.w).
void obstacleBounds(Collider col, BodyState ob, out float3 lo, out float3 hi)
{
    float3 c = transformPoint(ob.position, ob.orientation, col.localPos);
    float4 q = quatMul(ob.orientation, col.localRot);
    float3 e;
    if (col.shape == SHAPE_SPHERE) e = float3(col.params.x, col.params.x, col.params.x);
    else if (col.shape == SHAPE_CAPSULE) e = abs(quatRotate(q, float3(0.0, col.params.y, 0.0))) + col.params.x;
    else if (col.shape == SHAPE_BOX)
    {
        e = abs(quatRotate(q, float3(col.params.x, 0.0, 0.0))) + abs(quatRotate(q, float3(0.0, col.params.y, 0.0)))
            + abs(quatRotate(q, float3(0.0, 0.0, col.params.z)));
    }
    else e = float3(col.params.w, col.params.w, col.params.w);
    lo = c - e;
    hi = c + e;
}

bool softAabbOverlaps(uint s, float3 lo, float3 hi)
{
    float3 mn = SD(softAabbBase() + 2u * s).xyz, mx = SD(softAabbBase() + 2u * s + 1u).xyz;
    return all(lo <= mx) && all(hi >= mn);
}

// Anti-clip side cache (BeamNG's anti-clip): per particle, the last four
// triangles it took a contact with and the side it was on when it first did,
// as (triangle + 1) << 1 | back. A node pushed through a triangle it was in
// front of is pushed back to the side it came from, however deep, instead of
// being given the other side at the next step start.
static const uint PAIR_CACHE = 4u;
uint pairEntry(uint t, float side) { return ((t + 1u) << 1) | (side < 0.0 ? 1u : 0u); }
uint pairTri(uint e) { return (e >> 1) - 1u; }
float pairSide(uint e) { return (e & 1u) != 0u ? -1.0 : 1.0; }
// The side cached for triangle t, 0 when none.
float pairCachedSide(uint4 cache, uint t)
{
    for (uint k = 0u; k < PAIR_CACHE; k++)
        if (cache[k] != 0u && pairTri(cache[k]) == t) return pairSide(cache[k]);
    return 0.0;
}
// Whether a cached pair still matters: the triangle is alive, the node is over
// it (within 5 cm of its prism), not far in front on its side, and not beyond
// the walk's reach behind it.
bool pairKeep(uint e, float3 x)
{
    uint t = pairTri(e);
    float4 p0 = SD(softTriPlaneBase() + 2u * t);
    if (p0.w < 1e-8) return false;
    uint4 tri = SU(softTriBase() + t);
    float3 a = gBodies[tri.x].position, b = gBodies[tri.y].position, c = gBodies[tri.z].position;
    float sd = pairSide(e) * dot(x - a, p0.xyz);
    if (sd > triThickness(t, tri) + 0.1 || sd < -0.3) return false;
    float3 w;
    float3 foot = x - p0.xyz * dot(x - a, p0.xyz);
    float3 q = closestOnTriangle(foot, a, b, c, w);
    return dot(q - foot, q - foot) < 0.0025;
}

// Per solved particle: the triangle (of another vehicle, or its own outside the
// self-collision exclusion) it presses on most, on the side it came from.
// SOFT_NODE_LANES lanes per node: each walks every LANES-th candidate triangle and
// keeps its first best; the node's best is the lowest (score, triangle) over its
// lanes, the triangle the one-lane walk (bodies, then triangles, ascending) keeps.
static const uint SOFT_NODE_LANES = 8u;
groupshared float gsNodeScore[64];
groupshared uint gsNodeTri[64];
groupshared float gsNodeSide[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_collide_nodes(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint gi = dispatchIndex(tid);
    uint i = gi / SOFT_NODE_LANES, lane = gi % SOFT_NODE_LANES;
    bool live = i < softListCount();
    uint4 ent = live ? SU(softListBase() + i) : uint4(0, 0, 0, 0);
    uint body = ent.x;
    uint sb = ent.w & 0xFFFFu;
    uint ordSelf = particleOrdinal(sb, body);
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * ordSelf;
    if (live && lane < NODE_CONTACTS) gSoftData[ns + 3u * lane] = float4(0, 0, 0, asfloat(SOFT_NONE));
    bool ext = (ent.w & ENTRY_CONTACTER) != 0u;
    // Self-collision: a node against its own body's triangles too (a torn door, a
    // folding bonnet), except triangles it is a corner of and, with node groups
    // (BeamNG), triangles whose nodes share its group, or without, any it sat
    // over within SELF_REST_GAP in the undeformed body (its own panel).
    bool selfOn = !softIsRigid(sb) && (ent.w & ENTRY_SELF) != 0u;
    live = live && softContactsOn() && (ext || selfOn);
    BodyState b = gBodies[body];
    live = live && bodyIsDynamic(b) && !bodyIsSleeping(b);
    float3 x = b.position, x0 = b.initialPos;
    uint cacheAt = softPairCacheBase() + ordSelf;
    uint4 cache = live ? SU(cacheAt) : uint4(0, 0, 0, 0);
    float bestScore = 1e30;
    uint bestTri = SOFT_NONE;
    float3 bestN = float3(0, 0, 0), bestW = float3(0, 0, 0);
    float bestDelta = 0.0;
    float bestGap = 0.0;
    float bestSide = 1.0;
    float4 restS = SD(softRigidBase() + 2u * ordSelf + 1u);
    float3 restSelf = restS.xyz;
    uint groupWord = asuint(restS.w);
    bool grouped = (groupWord & NODE_GROUPED) != 0u;
    uint myGroup = groupWord & 0xFFFFu;
    uint selfOrdBase = asuint(softBodyRec1(sb).w) - softBodyRec(sb).x;
    uint np = live ? softPairCount(sb) : 0u;
    for (uint pk = 0u; pk < np; pk++)
    {
        uint s2 = softPairAt(sb, pk);
        bool self = s2 == sb;
        if (self ? !selfOn : (!ext || !softAabbHas(s2, x))) continue;
        uint4 r2 = softBodyRec2(s2);
        for (uint t = r2.x + lane; t < r2.x + r2.y; t += SOFT_NODE_LANES)
        {
            uint4 tri = SU(softTriBase() + t);
            if (self)
            {
                if (tri.x == body || tri.y == body || tri.z == body) continue;
                // BeamNG: a triangle whose nodes share the node's group.
                if (grouped && myGroup != 0u && (tri.w >> 16) == myGroup) continue;
            }
            float delta = triThickness(t, tri);
            float side = pairCachedSide(cache, t);
            // soft_tri_planes: first vertex and cull radius, then normal and area
            // (the same values as from the vertices, which load only past the cull).
            float reach = delta + length(x - x0) + 0.05;
            float4 p1 = SD(softTriPlaneBase() + 2u * t + 1u);
            float3 a = p1.xyz;
            // Inside the prism at |d| ≤ max(reach, 0.3) puts the node within the
            // triangle's reach of its first vertex.
            float far = p1.w + max(reach, 0.3) + 1e-3;
            if (dot(x - a, x - a) > far * far) continue;
            float4 p0 = SD(softTriPlaneBase() + 2u * t);
            if (p0.w < 1e-8) continue; // torn or degenerate
            float3 n = p0.xyz;
            float d = dot(x - a, n);
            if (abs(d) > max(reach, 0.3)) continue;
            float3 bpos = gBodies[tri.y].position, c = gBodies[tri.z].position;
            float3 a0 = gBodies[tri.x].initialPos, b0 = gBodies[tri.y].initialPos, c0 = gBodies[tri.z].initialPos;
            if (side == 0.0)
            {
                // A new pair takes the side the node came from at the step start.
                // Triangles face outward, so a node up to 5 cm behind the front face
                // (pressed in by an earlier contact) still counts as in front; only
                // one clearly behind starts on the back side (two-sided triangles).
                float3 n0 = cross(b0 - a0, c0 - a0);
                float l0 = length(n0);
                float d0 = l0 > 1e-8 ? dot(x0 - a0, n0 / l0) : d;
                side = d0 >= -0.05 ? 1.0 : -1.0;
            }
            float sd = side * d;
            if (sd > reach || sd < -0.3) continue;
            float3 w;
            float3 q = closestOnTriangle(x - n * d, a, bpos, c, w);
            // Prism test: the node projects inside the triangle.
            if (dot(q - (x - n * d), q - (x - n * d)) > 1e-6) continue;
            if (self && !grouped)
            {
                // Checked last (it costs a closest-point query): excluded only if the
                // node sat over this triangle's face, within SELF_REST_GAP, in the
                // rest body: its own panel, or the shell under a bolt-on part. A
                // neighbour beside it (a fender next to the bonnet) still collides
                // once the part comes loose.
                float3 wr;
                float3 ra = SD(softRigidBase() + 2u * (selfOrdBase + tri.x) + 1u).xyz;
                float3 rb = SD(softRigidBase() + 2u * (selfOrdBase + tri.y) + 1u).xyz;
                float3 rc = SD(softRigidBase() + 2u * (selfOrdBase + tri.z) + 1u).xyz;
                float3 rn = cross(rb - ra, rc - ra);
                float rl = length(rn);
                float dPlane = rl > 1e-8 ? abs(dot(restSelf - ra, rn / rl)) : 0.0;
                float dRest = length(closestOnTriangle(restSelf, ra, rb, rc, wr) - restSelf);
                if (dRest < SELF_REST_GAP && dRest - dPlane < 0.02) continue;
            }
            // The deepest on its own side first.
            float score = sd;
            if (score < bestScore || (score == bestScore && t < bestTri))
            {
                bestScore = score; bestTri = t; bestN = n * side; bestW = w; bestDelta = delta; bestSide = side;
                // Gap at the step start (frozen normal): the stabilised constraint
                // corrects only (1 − α) of an existing overlap per step.
                bestGap = dot(x0 - (w.x * a0 + w.y * b0 + w.z * c0), n * side) - delta;
            }
        }
    }
    gsNodeScore[lid.x] = bestScore;
    gsNodeTri[lid.x] = bestTri;
    gsNodeSide[lid.x] = bestSide;
    GroupMemoryBarrierWithGroupSync();
    if (!live || lane != 0u) return;
    // Lane 0 assembles the node's contacts: the deepest pair over the lanes, the
    // cached pairs that still matter, then the other lanes' nearest pairs (so a
    // pair is cached, with its side, as soon as the node comes within reach, not
    // only once it is the deepest). All of them are enforced.
    uint first = lid.x;
    uint top = SOFT_NONE;
    float topScore = 1e30;
    for (uint k = 0u; k < SOFT_NODE_LANES; k++)
    {
        float sk = gsNodeScore[first + k];
        uint tk = gsNodeTri[first + k];
        if (tk != SOFT_NONE && (sk < topScore || (sk == topScore && tk < top))) { topScore = sk; top = k; }
    }
    uint4 next = uint4(0, 0, 0, 0);
    uint m = 0u;
    if (top != SOFT_NONE) { next[0] = pairEntry(gsNodeTri[first + top], gsNodeSide[first + top]); m = 1u; }
    for (uint k = 0u; k < PAIR_CACHE && m < NODE_CONTACTS; k++)
    {
        uint e = cache[k];
        bool dup = false;
        for (uint j = 0u; j < m; j++) dup = dup || pairTri(next[j]) == pairTri(e);
        if (e != 0u && !dup && pairKeep(e, x)) { next[m] = e; m++; }
    }
    for (uint k = 0u; k < SOFT_NODE_LANES && m < NODE_CONTACTS; k++)
    {
        uint tk = gsNodeTri[first + k];
        bool dup = tk == SOFT_NONE;
        for (uint j = 0u; j < m; j++) dup = dup || pairTri(next[j]) == tk;
        if (!dup) { next[m] = pairEntry(tk, gsNodeSide[first + k]); m++; }
    }
    gSoftData[cacheAt] = asfloat(next);
    if (m == 0u) return;
    // Each pair's contact for this step: normal toward the node's side,
    // barycentrics of its foot (clamped onto the triangle), thickness and the
    // step-start gap.
    for (uint k = 0u; k < m; k++)
    {
        uint t = pairTri(next[k]);
        float side = pairSide(next[k]);
        uint4 tri = SU(softTriBase() + t);
        float4 p0 = SD(softTriPlaneBase() + 2u * t);
        float3 n = p0.xyz * side;
        float3 a = gBodies[tri.x].position, bp = gBodies[tri.y].position, c = gBodies[tri.z].position;
        float3 w;
        closestOnTriangle(x - p0.xyz * dot(x - a, p0.xyz), a, bp, c, w);
        float delta = triThickness(t, tri);
        float3 q0 = w.x * gBodies[tri.x].initialPos + w.y * gBodies[tri.y].initialPos + w.z * gBodies[tri.z].initialPos;
        // .y: the node's distance in front of the face on its side (0 when behind):
        // the displacement bound's room toward it.
        float sd = max(dot(x - a, n), 0.0);
        gSoftData[ns + 3u * k] = float4(n, asfloat(t));
        gSoftData[ns + 3u * k + 1u] = float4(w, delta);
        gSoftData[ns + 3u * k + 2u] = float4(dot(x0 - q0, n) - delta, sd, 0.0, 0.0);
    }
    gSoftData[softFlagBase() + ordSelf] = float4(1, 0, 0, 0);
}

// Per collision triangle, before node contacts: its unit normal and area (area 0
// when torn or degenerate) and its first vertex with the distance to the farthest
// other vertex, so soft_collide_nodes culls a triangle from two reads.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_tri_planes(uint3 tid : SV_DispatchThreadID)
{
    uint t = dispatchIndex(tid);
    if (t >= softTriCount()) return;
    uint4 tri = SU(softTriBase() + t);
    float3 a = gBodies[tri.x].position, b = gBodies[tri.y].position, c = gBodies[tri.z].position;
    float3 nr = cross(b - a, c - a);
    float area = length(nr);
    bool dead = triDead(t, a, b, c) || area < 1e-8;
    gSoftData[softTriPlaneBase() + 2u * t] = dead ? float4(0.0, 0.0, 0.0, 0.0) : float4(nr / area, area);
    gSoftData[softTriPlaneBase() + 2u * t + 1u] = float4(a, max(length(b - a), length(c - a)));
}

// Closest points between an (oriented) box and triangle abc: q on the triangle
// (barycentrics w), cs on the box surface. A triangle corner inside the box gives
// `inside`, with cs its exit point through the nearest face and n that face's
// normal. Otherwise the closest pair is a triangle corner against the box or a box
// edge against the triangle (which covers face–corner and edge–edge).
void boxTriangle(Collider col, BodyState ob, float3 a, float3 b, float3 c, out float3 q, out float3 cs, out float3 w,
    out bool inside, out float3 n)
{
    float4 rot = quatMul(ob.orientation, col.localRot);
    float3 centre = transformPoint(ob.position, ob.orientation, col.localPos);
    float3 h = col.params.xyz;
    inside = false;
    n = float3(0, 1, 0);
    float best = 1e30;
    float deepest = 0.0;
    float3 corners[3] = { a, b, c };
    float3 bary[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    [unroll]
    for (uint k = 0u; k < 3u; k++)
    {
        float3 l = quatRotate(quatConjugate(rot), corners[k] - centre);
        float3 clamped = clamp(l, -h, h);
        if (all(abs(l) < h))
        {
            // Inside: leave through the nearest face.
            float3 room = h - abs(l);
            uint ax = room.x < room.y ? (room.x < room.z ? 0u : 2u) : (room.y < room.z ? 1u : 2u);
            float3 out = l;
            out[ax] = h[ax] * (l[ax] >= 0.0 ? 1.0 : -1.0);
            if (room[ax] > deepest)
            {
                deepest = room[ax];
                inside = true;
                q = corners[k];
                w = bary[k];
                cs = centre + quatRotate(rot, out);
                float3 fn = float3(0, 0, 0);
                fn[ax] = l[ax] >= 0.0 ? 1.0 : -1.0;
                n = quatRotate(rot, fn);
            }
            continue;
        }
        float3 p = centre + quatRotate(rot, clamped);
        float d = dot(corners[k] - p, corners[k] - p);
        if (!inside && d < best) { best = d; q = corners[k]; w = bary[k]; cs = p; }
    }
    if (inside) return;
    // The twelve box edges against the triangle.
    [unroll]
    for (uint e = 0u; e < 12u; e++)
    {
        uint axis = e / 4u;
        uint u = (axis + 1u) % 3u, v = (axis + 2u) % 3u;
        float3 lo = float3(0, 0, 0), hi = float3(0, 0, 0);
        lo[axis] = -h[axis]; hi[axis] = h[axis];
        lo[u] = (e & 1u) != 0u ? h[u] : -h[u]; hi[u] = lo[u];
        lo[v] = (e & 2u) != 0u ? h[v] : -h[v]; hi[v] = lo[v];
        float3 qq, cc, ww;
        closestSegTriangle(centre + quatRotate(rot, lo), centre + quatRotate(rot, hi), a, b, c, qq, cc, ww);
        float d = dot(qq - cc, qq - cc);
        if (d < best) { best = d; q = qq; cs = cc; w = ww; }
    }
}

// Per collision triangle: its deepest thin obstacle, and the (up to four) nodes
// that chose it in soft_collide_nodes (the triangle side of their contacts).
// One 64-lane workgroup per triangle: lane 0 meets the obstacles; every lane
// scans a contiguous share of each nearby body's nodes for contacts naming the
// triangle, and lane 0 takes the first TRI_NODE_CONTACTS in node order (what a
// single thread walking the nodes in order would take).
static const uint SOFT_TRI_LANES = 64u;
groupshared uint gsTriHits[SOFT_TRI_LANES * 4u];
groupshared uint gsTriHitCount[SOFT_TRI_LANES];
groupshared uint gsTriFound;
groupshared uint gsTriMarked;

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_collide_tris(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // Every lane reaches every barrier (Chrome requires uniform control flow), so
    // the exits are a flag, and lane 0 publishes the walk's length through
    // gsUniformLanes (a uniform load).
    uint t = softGroupBody(gid), lane = lid.x;
    bool go = t < softTriCount();
    uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * t;
    if (go && lane == 0u)
    {
        gSoftData[tc] = float4(0, 0, 0, asfloat(SOFT_NONE));
        for (uint m = 0u; m < TRI_NODE_CONTACTS; m++) gSoftData[tc + 3u + 3u * m] = float4(0, 0, 0, asfloat(SOFT_NONE));
        gSoftData[tc + TRI_BOUND_SLOT] = float4(1e30, 1e30, 0, 0);
    }
    go = go && softContactsOn();
    uint4 tri = go ? SU(softTriBase() + t) : uint4(0, 0, 0, 0);
    // Anti-clip-only triangles (non-collidable) still keep their nodes' sides,
    // but meet no obstacles.
    bool collide = ((tri.w >> 8) & 1u) != 0u;
    uint sb = tri.w & 255u;
    BodyState ba = gBodies[tri.x], bb = gBodies[tri.y], bc = gBodies[tri.z];
    go = go && !(bodyIsSleeping(ba) && bodyIsSleeping(bb) && bodyIsSleeping(bc));
    float3 a = ba.position, b = bb.position, c = bc.position;
    go = go && !triDead(t, a, b, c);
    float delta = triThickness(t, tri);
    float travel = max(max(length(a - ba.initialPos), length(b - bb.initialPos)), length(c - bc.initialPos));
    float3 lo = min(min(a, b), c), hi = max(max(a, b), c);
    if (lane == 0u)
    {
        gsTriFound = 0u;
        gsTriMarked = 0u;
        gsUniformLanes = go ? softPairCount(sb) + 1u : 0u;
        // Thin obstacles.
        float bestGap = 1e30;
        uint nObs = go && collide ? softObstacleCount() : 0u;
        for (uint k = 0u; k < nObs; k++)
        {
            uint ci = SU(softObstacleBase() + k).x;
            Collider col = gColliders[ci];
            BodyState ob = gBodies[col.body];
            // Boxes apart by more than the contact reach on an axis cannot touch.
            float reach = delta + travel + 0.05;
            float3 olo, ohi;
            obstacleBounds(col, ob, olo, ohi);
            if (any(olo > hi + reach) || any(ohi < lo - reach)) continue;
            float r = col.params.x;
            float3 q, cs, w;
            bool inside = false;
            float3 insideN = float3(0, 1, 0);
            if (col.shape == SHAPE_SPHERE)
            {
                cs = transformPoint(ob.position, ob.orientation, col.localPos);
                q = closestOnTriangle(cs, a, b, c, w);
            }
            else if (col.shape == SHAPE_CAPSULE)
            {
                float3 ax = quatRotate(quatMul(ob.orientation, col.localRot), float3(0.0, col.params.y, 0.0));
                float3 mid = transformPoint(ob.position, ob.orientation, col.localPos);
                closestSegTriangle(mid - ax, mid + ax, a, b, c, q, cs, w);
            }
            else if (col.shape == SHAPE_BOX)
            {
                r = 0.0;
                boxTriangle(col, ob, a, b, c, q, cs, w, inside, insideN);
            }
            else continue;
            float3 dv = q - cs;
            float dist = length(dv);
            float gap = inside ? -dist : dist - r;
            if (gap > delta + travel + 0.05 || gap >= bestGap) continue;
            float3 n = inside ? insideN : (dist > 1e-5 ? dv / dist : normalize(cross(b - a, c - a)));
            // Touching or crossing (no direction from the gap): point away from the obstacle.
            if (!inside && dist <= 1e-5 && dot(n, (a + b + c) / 3.0 - cs) < 0.0) n = -n;
            bestGap = gap;
            // Gap at the step start (triangle at x⁻) for the stabilised constraint.
            float3 q0 = w.x * ba.initialPos + w.y * bb.initialPos + w.z * bc.initialPos;
            gSoftData[tc] = float4(n, asfloat(ci));
            gSoftData[tc + 1u] = float4(cs + n * r, dot(q0 - (cs + n * r), n) - delta);
            gSoftData[tc + 2u] = float4(w, delta);
            gsTriMarked = 1u;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Nodes (other bodies', and its own: self-collision, filtered on the node
    // side) whose contacts name this triangle.
    // The nearest node in front of the face, on its front and on its back.
    float2 nearest = float2(1e30, 1e30);
    float3 faceN = SD(softTriPlaneBase() + 2u * t).xyz;
    uint walk = gsUniformLanes;
    for (uint pk = 0u; pk + 1u < walk; pk++)
    {
        uint s2 = softPairAt(sb, pk);
        bool skip = gsTriFound >= TRI_NODE_CONTACTS
            || (s2 == sb ? softIsRigid(sb) : !softAabbOverlaps(s2, lo - 0.3, hi + 0.3));
        uint4 r2 = skip ? uint4(0, 0, 0, 0) : softBodyRec(s2);
        uint ordBase = skip ? 0u : asuint(softBodyRec1(s2).w);
        uint chunk = (r2.y + SOFT_TRI_LANES - 1u) / SOFT_TRI_LANES;
        uint k0 = lane * chunk, k1 = min(k0 + chunk, r2.y);
        uint hits = 0u;
        for (uint k = k0; k < k1 && hits < 4u; k++)
        {
            uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * (ordBase + k);
            for (uint j = 0u; j < NODE_CONTACTS && hits < 4u; j++)
            {
                uint tj = asuint(SD(ns + 3u * j).w);
                if (tj == SOFT_NONE) break;
                if (tj == t) { gsTriHits[4u * lane + hits] = (k << 2) | j; hits++; }
            }
        }
        gsTriHitCount[lane] = hits;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u && !skip)
        {
            uint found = gsTriFound;
            for (uint l = 0u; l < SOFT_TRI_LANES && found < TRI_NODE_CONTACTS; l++)
            {
                for (uint m = 0u; m < gsTriHitCount[l] && found < TRI_NODE_CONTACTS; m++)
                {
                    uint code = gsTriHits[4u * l + m];
                    uint k = code >> 2, j = code & 3u;
                    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * (ordBase + k);
                    float4 s0 = SD(ns + 3u * j);
                    float4 sg = SD(ns + 3u * j + 2u);
                    if (sg.y > 0.0)
                    {
                        if (dot(s0.xyz, faceN) >= 0.0) nearest.x = min(nearest.x, sg.y);
                        else nearest.y = min(nearest.y, sg.y);
                    }
                    gSoftData[tc + 3u + 3u * found] = float4(s0.xyz, asfloat(r2.x + k));
                    gSoftData[tc + 4u + 3u * found] = SD(ns + 3u * j + 1u);
                    gSoftData[tc + 5u + 3u * found] = sg;
                    found++;
                }
            }
            gsTriFound = found;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane != 0u || walk == 0u) return;
    gSoftData[tc + TRI_BOUND_SLOT] = float4(nearest, 0, 0);
    if (gsTriFound > 0u || gsTriMarked != 0u)
    {
        uint ob = asuint(softBodyRec1(sb).w) - softBodyRec(sb).x;
        gSoftData[softFlagBase() + ob + tri.x] = float4(1, 0, 0, 0);
        gSoftData[softFlagBase() + ob + tri.y] = float4(1, 0, 0, 0);
        gSoftData[softFlagBase() + ob + tri.z] = float4(1, 0, 0, 0);
    }
}

// Penalty contact on one particle: Jacobian coefficient s (1 for a node, ±w for a
// triangle vertex), gap C along n, gap C0 at the step start, tangential relative
// travel u this step. As rigid contacts (AVBD α), only (1 − α) of an overlap
// present at the step start is corrected per step, so a deep overlap (a node
// freed inside a crushed panel) is not fired out.
void softContactTerm(inout float3 g, inout float3x3 H, float s, float C, float C0, float3 n, float3 u, float delta)
{
    // The overlap left this step: α of the step-start one. Past the face itself
    // (deeper than the thickness δ), at least contact_push_out·h less, until the
    // node is back on the face (α = 0.99 alone held a node 2 cm behind a panel for
    // a third of a second). Within the thickness, α alone: parts that rest that
    // close are left be.
    float c0 = min(C0, 0.0);
    float keep = PARAMS.alpha * c0;
    if (c0 < -delta) keep = max(keep, min(c0 + softPushOut() * PARAMS.dt, -delta));
    C -= keep;
    if (C >= 0.0) return;
    float k = softContactStiffness();
    float3x3 nn = float3x3(n * n.x, n * n.y, n * n.z);
    g += n * (s * k * C);
    H += nn * (s * s * k);
    // Lagged Coulomb friction: a tangential spring capped at μ·|f_n| (secant stiffness).
    float3 ut = u - n * dot(u, n);
    float ul = length(ut);
    if (ul < 1e-9) { H += (float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) - nn) * (s * s * k); return; }
    float cap = softContactFriction() * k * (-C);
    float kt = min(k, cap / ul);
    g += ut * (s * kt);
    H += (float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) - nn) * (s * s * kt);
}

float3 triTravel(uint4 tri, float3 w)
{
    return w.x * (softPos(tri.x) - softPos0(tri.x))
        + w.y * (softPos(tri.y) - softPos0(tri.y))
        + w.z * (softPos(tri.z) - softPos0(tri.z));
}

float3 triPoint(uint4 tri, float3 w)
{
    return w.x * softPos(tri.x) + w.y * softPos(tri.y) + w.z * softPos(tri.z);
}

// Soft contact terms of one particle (node side and triangle-vertex side).
// Triangle-side contact terms of a particle at corner j of triangle tt: the
// triangle's thin obstacle (kind 0) or the node in its slot m (kind m + 1).
void softTriObstacleTerm(uint4 tri, uint tc, uint j, inout float3 g, inout float3x3 H)
{
    float4 o0 = SD(tc);
    float4 o1 = SD(tc + 1u);
    float4 o2 = SD(tc + 2u);
    float C = dot(o0.xyz, triPoint(tri, o2.xyz) - o1.xyz) - o2.w;
    softContactTerm(g, H, o2[j], C, o1.w, o0.xyz, triTravel(tri, o2.xyz), o2.w);
}
void softTriNodeTerm(uint4 tri, uint tc, uint j, uint m, inout float3 g, inout float3x3 H)
{
    float4 c0 = SD(tc + 3u + 3u * m);
    uint node = asuint(c0.w);
    float4 c1 = SD(tc + 4u + 3u * m);
    float3 np = softPos(node);
    float C = dot(c0.xyz, np - triPoint(tri, c1.xyz)) - c1.w;
    float3 u = (np - softPos0(node)) - triTravel(tri, c1.xyz);
    softContactTerm(g, H, -c1[j], C, SD(tc + 5u + 3u * m).x, c0.xyz, u, c1.w);
}

uint softContactListBase() { return SU(9u).y; }
static const uint CONTACT_LIST_STRIDE = 4u; // engine_softbody::CONTACT_LIST_STRIDE
static const uint CONTACT_LIST_MAX = 15u;
static const uint CONTACT_LIST_ALL = 0xFFFFFFFFu;

