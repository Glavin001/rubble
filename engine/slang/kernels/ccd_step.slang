// Continuous collision and step-start impulses (engine/docs/ccd.md).
// Included by world_step.slang, so every variant (canonical, compact,
// persistent, reuse-copy) gets these entries with its own contact storage.
//
// Step placement (engine_exec::rigid_plan::encode_step):
//   apply_impulses → … → broadphase_clear → ccd_register → broadphase_insert
//   → broadphase_query → broadphase_fast → … → ccd_toi → ccd_pick_key → ccd_pick
//   → narrowphase (skips each CCD body's winning pair) → convex → ccd_emit
//   → … → predict_bodies (CCD bodies start at their TOI pose)
//
// Row 1 (look-ahead that scales): CCD bodies have no reach cap, and a swept
// box longer than a grid cell goes to the fast list: it is not inserted in the
// grid, and broadphase_fast walks the cells along its sweep (every table and
// level), the plane and large lists and the other fast colliders, so it owns
// every pair it is part of.
// Rows 2-3 (TOI-placed speculative contacts): ccd_toi casts each CCD body's
// collider along its relative motion against each candidate (start rotation,
// as Jolt's LinearCast); the nearest time of impact per body is kept
// deterministically (min TOI bits, then the lowest candidate index). The
// winning pair's manifold is built at the TOI pose by ccd_emit and its rows
// hold the gap along the path, so AVBD solves it as an ordinary contact with
// real, two-way forces (visible to last_contacts and the stress solver).
// Row 4: GPU impulses (apply_impulses), summed per body in record order.

// ---------------------------------------------------------------------------
// CCD body registration (one thread per body, after broadphase_clear)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_register(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsCcd(b)) return;
    uint slot;
    AADD3(gScratchAtomics[CCD_BODY_COUNT], 1u, slot);
    if (slot >= CCD_BODY_CAP)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_CCD_BODIES);
        return;
    }
    uint r = CCD_BODY_BASE + 4u * slot;
    ASTORE(gScratchAtomics[r], i);
    ASTORE(gScratchAtomics[r + 1u], CCD_TOI_NONE);
    ASTORE(gScratchAtomics[r + 2u], CCD_NO_PAIR);
    ASTORE(gScratchAtomics[r + 3u], 0u);
    ASTORE(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 8u], CCD_TOI_NONE);
    ASTORE(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 10u], CCD_TOI_NONE);
    ASTORE(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 11u], 0u);
    // ccd_pick_key's (word 0 is ccd_mark's again once the pick is made).
    ASTORE(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot], 0u);
    gBodies[i].initialPad = asfloat(slot);
    AMAX2(gScratchAtomics[CCD_MAX_REACH], asuint(length(b.linearVel) * PARAMS.dt));
}

// ---------------------------------------------------------------------------
// Fast list query (row 1)
// ---------------------------------------------------------------------------
float segmentPointDistance(float3 a, float3 b, float3 p)
{
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-20), 0.0, 1.0);
    return length(a + ab * t - p);
}

// Fast collider i against grid entry j (any table, any level): the fast query
// owns this pair, whatever the entry's motion state.
void fastAccept(uint i, Collider a, bool aDyn, Aabb aa, uint j)
{
    if (j == i) return;
    Aabb ab = gAabbs[HOST_S(1) + j];
    if (!aabbOverlap(aa, ab)) return;
    Collider b = gColliders[j];
    if (b.body == a.body || b.shape == SHAPE_PLANE) return;
    if (!aDyn && !bodyIsDynamic(gBodies[b.body])) return;
    if (bodiesJointed(a.body, b.body)) return;
    emitCandidate(i, j);
}

static const uint FAST_MAX_CELLS = 262144u;


// Every cell of one grid level whose entries can overlap the swept box: the
// box padded by the entries' largest reach from their centre (E), then only
// cells within reach of the sweep segment (a tube, so a diagonal sweep does
// not visit its whole bounding box). Hash aliases are rejected by each
// entry's true cell, so every entry is accepted at most once.
void fastWalkLevel(uint i, Collider a, bool aDyn, Aabb aa, float3 s0, float3 s1, float R,
    float cs, float E, uint level, bool hier, uint lane)
{
    uint base = HOST_S(1);
    int3 lo = int3(floor((aa.mn - E) / cs));
    int3 hi = int3(floor((aa.mx + E) / cs));
    int3 span = hi - lo + int3(1, 1, 1);
    float cells = float(span.x) * float(span.y) * float(span.z);
    if (cells > float(FAST_MAX_CELLS))
    {
        if (lane == 0u) AOR2(gCounters[0].overflowFlags, OVERFLOW_FAST_LIST);
        return;
    }
    uint nCells = uint(span.x) * uint(span.y) * uint(span.z);
    float tube = R + 1.7320508 * E + 0.8660254 * cs;
    uint count = gCounters[0].colliderCount;
    for (uint k = lane; k < nCells; k += 64u)
    {
        int3 cell = lo + int3(int(k % uint(span.x)), int((k / uint(span.x)) % uint(span.y)),
            int(k / uint(span.x * span.y)));
        float3 cc = (float3(cell) + 0.5) * cs;
        if (segmentPointDistance(s0, s1, cc) > tube) continue;
        if (!hier)
        {
            // The uniform rest tables (world_step.slang restUniform): W then R, R without
            // its tombstones, through the same accept.
#if defined(ENGINE_WGSL)
            bool restU = restUniform();
#else
            bool restU = false;
#endif
            uint tables = restU || gridSplit() ? 2u : 1u;
            for (uint t = 0u; t < tables; t++)
            {
                bool rest = restU && t == 1u;
                uint heads = restU ? (rest ? restHeads() : awakeHeads()) : t * gridHashSize();
                uint link = ALOAD(gConstraintIds[heads + gridHash(cell.x, cell.y, cell.z)]);
                uint guard = 0u;
                while (link != 0xFFFFFFFFu && guard < 4096u)
                {
                    uint cur = link & GRID_COLLIDER_MASK;
                    Aabb ab = gAabbs[base + cur];
                    link = rest ? ALOAD(gConstraintIds[restLinks() + cur]) : (gridSplit() ? asuint(ab._pad1) : gColorLists[cur]);
                    guard++;
                    if (all(int3(floor((0.5 * (ab.mn + ab.mx)) / cs)) == cell) && !(rest && restTombstone(cur)))
                        fastAccept(i, a, aDyn, aa, cur);
                }
                if (link != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
            }
        }
        else
        {
            uint passes = restTables() ? 2u : 1u;
            for (uint t = 0u; t < passes; t++)
            {
                // Rest tables: W (movers, links in the AABB spare word) then R
                // (resting colliders; a mover left in R is a tombstone).
                bool rest = restTables() && t == 1u;
                uint heads = restTables() ? (rest ? restHeads() : awakeHeads()) : 0u;
                uint j = ALOAD(gConstraintIds[heads + hierHash(cell, level)]);
                uint guard = 0u;
                while (j != 0xFFFFFFFFu && guard < count)
                {
                    guard++;
                    Aabb ab = gAabbs[base + j];
                    uint next = rest ? ALOAD(gConstraintIds[restLinks() + j]) : asuint(ab._pad1);
                    if (hierLevel(ab._pad0) == level && all(hierCell(ab, level) == cell) &&
                        !(rest && bodyMovingOrKinematic(gBodies[gColliders[j].body])))
                        fastAccept(i, a, aDyn, aa, j);
                    j = next;
                }
                if (j != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
            }
        }
    }
}

void fastQuery(uint i, uint lane)
{
    uint base = HOST_S(1);
    Collider a = gColliders[i];
    BodyState ba = gBodies[a.body];
    bool aDyn = bodyIsDynamic(ba);
    Aabb aa = gAabbs[base + i];
    // build_aabbs swept the box by v·h: undo it for the start box, whose
    // centre and half diagonal bound the collider along the sweep segment.
    float3 sweep = ba.linearVel * PARAMS.dt;
    float3 umn = aa.mn - min(sweep, float3(0, 0, 0));
    float3 umx = aa.mx - max(sweep, float3(0, 0, 0));
    float3 s0 = 0.5 * (umn + umx);
    float R = length(0.5 * (umx - umn));
    if (HOST_S(18) == 0u)
    {
        // Uniform grid entries reach at most a cell from their centre (half a
        // cell of size and at most half a cell of sweep: longer ones are fast).
        float cs = max(PARAMS.gridCellSize, 0.5);
        fastWalkLevel(i, a, aDyn, aa, s0, s0 + sweep, R, cs, cs, 0u, false, lane);
    }
    else
    {
        // A level-L entry's swept half extent is at most half its cell (_pad0).
        uint levels = HOST_S(18);
        uint occupied = ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT]);
        for (uint level = 0u; level < levels; level++)
        {
            if (((occupied >> level) & 1u) == 0u) continue;
            float cs = hierCellSize(level);
            fastWalkLevel(i, a, aDyn, aa, s0, s0 + sweep, R, cs, 0.5 * cs, level, true, lane);
        }
    }
    // Lists: planes, oversized colliders, other fast colliders (lower index
    // owns a fast-fast pair).
    uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
    bool hier = HOST_S(18) != 0u;
    uint nLargeDyn = hier ? 0u : min(ALOAD(gScratchAtomics[25]), 16u);
    uint nLarge = hier ? min(ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT]), HIER_LARGE_MAX) : largeStaticCount();
    uint nFast = fastCount();
    uint total = nPlanes + nLargeDyn + nLarge + nFast;
    for (uint k = lane; k < total; k += 64u)
    {
        if (k < nPlanes)
        {
            uint j = ALOAD(gScratchAtomics[43u + k]);
            if (aDyn && !aabbAbovePlane(aa, gColliders[j].params)) emitCandidate(i, j);
        }
        else if (k < nPlanes + nLargeDyn)
            fastAccept(i, a, aDyn, aa, ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k - nPlanes]));
        else if (k < nPlanes + nLargeDyn + nLarge)
        {
            uint q = k - nPlanes - nLargeDyn;
            fastAccept(i, a, aDyn, aa, hier ? ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + q])
                                            : largeStaticCollider(q));
        }
        else
        {
            uint j = ALOAD(gScratchAtomics[FAST_BASE + k - nPlanes - nLargeDyn - nLarge]);
            if (j > i) fastAccept(i, a, aDyn, aa, j);
        }
    }
}

// One workgroup per fast collider: FAST_CAP groups (in rows of 32768), those
// past the list's count return at once, so a flood of fast debris after big
// impacts is walked in parallel, not a few groups striding the list.
// write_fast_args sizes it from the list's count instead (work slot 82, one
// workgroup per listed collider, the same group numbering), so a step with few
// fast colliders launches few groups, not FAST_CAP.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_fast_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(82u, fastCount() * 64u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_fast(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint k = gid.x + gid.y * 32768u;
    if (k >= fastCount()) return;
    fastQuery(ALOAD(gScratchAtomics[FAST_BASE + k]), lid.x);
}

// A moving kinematic finds the sleeping dynamics it may wake (uniform grid;
// the hierarchical grid treats it as a mover). Every cell under its swept box,
// padded by a cell, in every table, plus the large-dynamic list.
void kinematicQuery(uint i, Collider col)
{
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    float cs = max(PARAMS.gridCellSize, 0.5);
    int3 lo = int3(floor((box.mn - cs) / cs));
    int3 hi = int3(floor((box.mx + cs) / cs));
    int3 span = hi - lo + int3(1, 1, 1);
    if (float(span.x) * float(span.y) * float(span.z) > 65536.0)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    uint tables = gridSplit() ? 2u : 1u;
    for (int z = lo.z; z <= hi.z; z++)
    for (int y = lo.y; y <= hi.y; y++)
    for (int x = lo.x; x <= hi.x; x++)
    for (uint t = 0u; t < tables; t++)
    {
        uint link = ALOAD(gConstraintIds[t * gridHashSize() + gridHash(x, y, z)]);
        uint guard = 0u;
        while (link != 0xFFFFFFFFu && guard < 4096u)
        {
            uint cur = link & GRID_COLLIDER_MASK;
            Aabb ab = gAabbs[base + cur];
            link = gridSplit() ? asuint(ab._pad1) : gColorLists[cur];
            guard++;
            if (any(int3(floor((0.5 * (ab.mn + ab.mx)) / cs)) != int3(x, y, z))) continue;
            if (!aabbOverlap(box, ab)) continue;
            Collider other = gColliders[cur];
            BodyState ob = gBodies[other.body];
            if (other.body == col.body || !bodyIsDynamic(ob) || !bodyIsSleeping(ob)) continue;
            if (bodiesJointed(col.body, other.body)) continue;
            emitCandidate(i, cur);
        }
    }
    uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k]);
        Collider other = gColliders[j];
        BodyState ob = gBodies[other.body];
        if (other.body == col.body || !bodyIsSleeping(ob)) continue;
        if (!aabbOverlap(box, gAabbs[base + j])) continue;
        emitCandidate(i, j);
    }
}

// ---------------------------------------------------------------------------
// Shape casts: GJK ray cast against the Minkowski difference (van den Bergen,
// "Ray casting against general convex objects with application to continuous
// collision detection", 2004)
// ---------------------------------------------------------------------------

// World support point of a collider at pose (pos, rot) in direction dir.
float3 shapeSupport(Collider c, float3 pos, float4 rot, float3 dir)
{
    float len = length(dir);
    float3 u = len > 1e-20 ? dir / len : float3(0, 1, 0);
    if (c.shape == SHAPE_SPHERE) return pos + u * c.params.x;
    if (c.shape == SHAPE_CAPSULE)
    {
        float3 axis = quatRotate(rot, float3(0, 1, 0));
        float s = dot(u, axis) >= 0.0 ? 1.0 : -1.0;
        return pos + axis * (s * c.params.y) + u * c.params.x;
    }
    float3 local = quatRotate(quatConjugate(rot), u);
    if (c.shape == SHAPE_BOX)
    {
        float3 h = c.params.xyz;
        return pos + quatRotate(rot, float3(local.x >= 0.0 ? h.x : -h.x,
            local.y >= 0.0 ? h.y : -h.y, local.z >= 0.0 ? h.z : -h.z));
    }
    HullRef h = hullOf(c);
    return pos + quatRotate(rot, hullVertex(h, hullSupportIndex(h, local)));
}

// A sphere's or capsule's radius: the margin around its core (a point or a
// segment). Boxes and hulls have none.
float shapeMargin(Collider c)
{
    return (c.shape == SHAPE_SPHERE || c.shape == SHAPE_CAPSULE) ? c.params.x : 0.0;
}

// Support point of a collider's core (the shape less its margin).
float3 coreSupport(Collider c, float3 pos, float4 rot, float3 dir)
{
    if (c.shape == SHAPE_SPHERE) return pos;
    if (c.shape == SHAPE_CAPSULE)
    {
        float3 axis = quatRotate(rot, float3(0, 1, 0));
        return pos + axis * (dot(dir, axis) >= 0.0 ? c.params.y : -c.params.y);
    }
    return shapeSupport(c, pos, rot, dir);
}

// Closest point of triangle (a, b, c) to the origin (Ericson 5.1.5); `mask`
// names the vertices of the feature it lies on (bits 0, 1, 2).
float3 triangleClosest(float3 a, float3 b, float3 c, out uint mask)
{
    float3 ab = b - a, ac = c - a;
    float d1 = dot(ab, -a), d2 = dot(ac, -a);
    if (d1 <= 0.0 && d2 <= 0.0) { mask = 1u; return a; }
    float d3 = dot(ab, -b), d4 = dot(ac, -b);
    if (d3 >= 0.0 && d4 <= d3) { mask = 2u; return b; }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { mask = 3u; return a + ab * (d1 / max(d1 - d3, 1e-30)); }
    float d5 = dot(ab, -c), d6 = dot(ac, -c);
    if (d6 >= 0.0 && d5 <= d6) { mask = 4u; return c; }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { mask = 5u; return a + ac * (d2 / max(d2 - d6, 1e-30)); }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        mask = 6u;
        return b + (c - b) * ((d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-30));
    }
    float denom = 1.0 / max(va + vb + vc, 1e-30);
    mask = 7u;
    return a + ab * (vb * denom) + ac * (vc * denom);
}

// Origin and d on opposite sides of plane (a, b, c) (or a degenerate face).
bool originOutsideFace(float3 a, float3 b, float3 c, float3 d)
{
    float3 n = cross(b - a, c - a);
    float sp = dot(-a, n);
    float sd = dot(d - a, n);
    return sp * sd < 0.0 || abs(sd) < 1e-20;
}

// Closest point of the simplex Y[0..n) to the origin; Y and P (the support
// points behind Y) are reduced to the feature it lies on. n == 4 on return:
// the origin is inside the tetrahedron.
float3 simplexClosest(inout float3 Y[4], inout float3 P[4], inout uint n)
{
    uint mask = 1u;
    float3 v = Y[0];
    if (n == 2u)
    {
        float3 ab = Y[1] - Y[0];
        float t = -dot(Y[0], ab) / max(dot(ab, ab), 1e-30);
        if (t <= 0.0) { mask = 1u; v = Y[0]; }
        else if (t >= 1.0) { mask = 2u; v = Y[1]; }
        else { mask = 3u; v = Y[0] + ab * t; }
    }
    else if (n == 3u)
    {
        v = triangleClosest(Y[0], Y[1], Y[2], mask);
    }
    else if (n == 4u)
    {
        float best = 1e30;
        bool any = false;
        uint m;
        if (originOutsideFace(Y[0], Y[1], Y[2], Y[3]))
        {
            float3 q = triangleClosest(Y[0], Y[1], Y[2], m);
            if (dot(q, q) < best) { best = dot(q, q); v = q; mask = m; }
            any = true;
        }
        if (originOutsideFace(Y[0], Y[2], Y[3], Y[1]))
        {
            float3 q = triangleClosest(Y[0], Y[2], Y[3], m);
            if (dot(q, q) < best)
            {
                best = dot(q, q); v = q;
                mask = (m & 1u) | ((m & 2u) << 1) | ((m & 4u) << 1);
            }
            any = true;
        }
        if (originOutsideFace(Y[0], Y[3], Y[1], Y[2]))
        {
            float3 q = triangleClosest(Y[0], Y[3], Y[1], m);
            if (dot(q, q) < best)
            {
                best = dot(q, q); v = q;
                mask = (m & 1u) | ((m & 2u) << 2) | ((m & 4u) >> 1);
            }
            any = true;
        }
        if (originOutsideFace(Y[1], Y[3], Y[2], Y[0]))
        {
            float3 q = triangleClosest(Y[1], Y[3], Y[2], m);
            if (dot(q, q) < best)
            {
                best = dot(q, q); v = q;
                mask = ((m & 1u) << 1) | ((m & 2u) << 2) | (m & 4u);
            }
            any = true;
        }
        if (!any) return float3(0, 0, 0);
    }
    uint k = 0u;
    for (uint q = 0u; q < 4u; q++)
    {
        if (q < n && ((mask >> q) & 1u) != 0u)
        {
            Y[k] = Y[q];
            P[k] = P[q];
            k++;
        }
    }
    n = k;
    return v;
}

static const uint GJK_MAX_ITERS = 32u;
// Largest gap a stalled cast still reports as touching (m); under any skin or
// contact depth that uses it (character skin 2 cm).
static const float GJK_TOUCH = 0.005;
static const uint CAST_MISS = 0u;
static const uint CAST_HIT = 1u;
static const uint CAST_OVERLAP = 2u;
static const uint CAST_BUDGET = 3u;

// Collider A translated by mu·r from s0 (configuration space: x = s0 + mu·r)
// against B, both at their given poses, for mu in [0, muMax]. On a hit, mu is
// the first touch and n the contact normal from A to B; CAST_OVERLAP: they
// already overlap at mu = 0 (no normal).
//
// GJK runs on the shapes' cores (a sphere's centre, a capsule's segment) and
// the radii are a margin m: a touch is the cores `m` apart (Box2D
// b2ShapeCast, Jolt's convex radius). GJK on the rounded shapes themselves
// converges slowly against flat faces and stalls millimetres short, worse the
// larger the other shape. With no margin (boxes, hulls) every step is the
// plain cast's.
uint gjkCast(Collider ca, float3 pa, float4 qa, Collider cb, float3 pb, float4 qb,
    float3 s0, float3 r, float muMax, float eps, out float mu, out float3 n)
{
    mu = 0.0;
    n = float3(0, 0, 0);
    float m = shapeMargin(ca) + shapeMargin(cb);
    float3 x = s0;
    // C = B − A contains pb − pa (the reference points of both shapes).
    float3 v = x - (pb - pa);
    float3 Y[4];
    float3 P[4];
    Y[0] = float3(0, 0, 0); Y[1] = Y[0]; Y[2] = Y[0]; Y[3] = Y[0];
    P[0] = Y[0]; P[1] = Y[0]; P[2] = Y[0]; P[3] = Y[0];
    uint count = 0u;
    bool advanced = false;
    float prevVV = 1e30;
    for (uint it = 0u; it < GJK_MAX_ITERS; it++)
    {
        if (dot(v, v) <= (m + eps) * (m + eps)) return advanced ? CAST_HIT : CAST_OVERLAP;
        float3 p = coreSupport(cb, pb, qb, v) - coreSupport(ca, pa, qa, -v);
        float3 w = x - p;
        // Past the margin along v: advance until the cores are m apart there.
        float gapV = dot(v, w) - m * length(v);
        if (gapV > 0.0)
        {
            float vr = dot(v, r);
            if (vr >= 0.0) return CAST_MISS;
            mu -= gapV / vr;
            if (mu > muMax) return CAST_MISS;
            x = s0 + r * mu;
            n = -normalize(v);
            advanced = true;
            for (uint k = 0u; k < 4u; k++)
                if (k < count) Y[k] = x - P[k];
        }
        if (count < 4u)
        {
            Y[count] = x - p;
            P[count] = p;
            count++;
        }
        v = simplexClosest(Y, P, count);
        if (count == 4u) return advanced ? CAST_HIT : CAST_OVERLAP;
        // No progress: |v| is at f32 resolution of the configuration (Jolt's
        // GJK stops the same way). After an advance, x touches C to within |v|.
        // A stall is a touch within GJK_TOUCH: f32 GJK between a curved shape
        // and an edge can stop improving a millimetre or two away. Farther, the
        // ray grazes past C (it cannot get closer along r) and misses.
        float vv = dot(v, v);
        if (vv >= prevVV * (1.0 - 1.0e-5))
        {
            float touch = m + max(10.0 * eps, GJK_TOUCH);
            if (vv <= touch * touch) return advanced ? CAST_HIT : CAST_OVERLAP;
            return CAST_MISS;
        }
        prevVV = vv;
    }
    return CAST_BUDGET;
}

// World pose of a collider at its body's current pose.
void colliderPose(Collider c, BodyState b, out float3 pos, out float4 rot)
{
    pos = transformPoint(b.position, b.orientation, c.localPos);
    rot = quatMul(b.orientation, c.localRot);
}

// Time of impact of the fast collider (moving by r over the step, relative to
// the other) against the other one. The cast starts one step back along r, so
// a pair already touching and closing reports TOI 0 with a real normal; a
// pair overlapping even a step back (resting, or sliding along a surface) has
// no TOI and is left to the ordinary contacts. t ∈ [0, 1], n from fast to other.
uint ccdCast(Collider cf, BodyState bf, Collider co, BodyState bo, float3 r, out float t, out float3 n)
{
    t = 1.0;
    n = float3(0, 0, 0);
    float3 pf; float4 qf; colliderPose(cf, bf, pf, qf);
    if (co.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(co.params.xyz);
        float rate = dot(pn, r);
        if (rate >= -1e-9) return CAST_MISS;
        float dist = dot(pn, shapeSupport(cf, pf, qf, -pn)) - co.params.w;
        float lam = dist / -rate;
        if (lam <= -1.0) return CAST_OVERLAP;
        if (lam > 1.0) return CAST_MISS;
        t = max(lam, 0.0);
        n = -pn;
        return CAST_HIT;
    }
    if (co.shape == SHAPE_HEIGHTFIELD)
    {
        // The exact translational sweep (heightfield.slang), from one step
        // back along r over two steps, as the GJK cast below.
        float len = length(r);
        if (len < 1e-12) return CAST_MISS;
        float tc; float3 nc;
        if (!heightFieldShapeCast(heightFieldOf(co), colliderFrame(co, bo), cf, pf - r, qf, r / len, 2.0 * len, tc, nc))
            return CAST_MISS;
        float lam = tc / len - 1.0;
        if (tc <= 0.0) return CAST_OVERLAP;
        if (lam > 1.0) return CAST_MISS;
        t = max(lam, 0.0);
        n = -nc;
        return CAST_HIT;
    }
    float3 po; float4 qo; colliderPose(co, bo, po, qo);
    float mu;
    float eps = 1.0e-4 + 1.0e-5 * length(r);
    // In the fast collider's frame: f32 keeps its resolution far from the origin.
    uint status = gjkCast(cf, float3(0, 0, 0), qf, co, po - pf, qo, -r, r, 2.0, eps, mu, n);
    if (status != CAST_HIT) return status;
    t = clamp(mu - 1.0, 0.0, 1.0);
    return CAST_HIT;
}

// Which side of a candidate pair is the CCD body that casts: the CCD one, or
// the faster of two (ties: the lower body index).
bool ccdFastIsA(BodyState ba, uint bodyA, BodyState bb, uint bodyB)
{
    bool ccdA = bodyIsCcd(ba), ccdB = bodyIsCcd(bb);
    if (ccdA != ccdB) return ccdA;
    float va = dot(ba.linearVel, ba.linearVel), vb = dot(bb.linearVel, bb.linearVel);
    return va != vb ? va > vb : bodyA < bodyB;
}

// Relative motion of the fast body over the step (the other's only if it moves).
float3 ccdRelativeMotion(BodyState bf, BodyState bo)
{
    float3 vo = bodyMovingOrKinematic(bo) ? bo.linearVel : float3(0, 0, 0);
    return (bf.linearVel - vo) * PARAMS.dt;
}

// A TOI shapes the step only if moving the whole step would carry the fast
// body deeper than the contact depth past it; grazing and sliding contacts
// keep the ordinary prediction.
bool ccdTOIMatters(float t, float3 n, float3 r)
{
    return (1.0 - t) * dot(r, n) > PARAMS.contactOffset;
}

// gCandidates[i]._pad after ccd_toi: 0 none, else asuint(t) + 1 (bit 31 clear,
// so the convex defer flag never reads it as set).
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_toi(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= min(ALOAD(gCounters[0].candidateCount), HOST_S(2))) return;
    CandidatePair pair = gCandidates[i];
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    if (!bodyIsCcd(ba) && !bodyIsCcd(bb)) return;
    // Sensors have no response: no time of impact against them.
    if (((ca.material | cb.material) & COLLIDER_SENSOR) != 0u) return;
    bool fastIsA = ccdFastIsA(ba, ca.body, bb, cb.body);
    uint fastBody = fastIsA ? ca.body : cb.body;
    BodyState bf = fastIsA ? ba : bb;
    BodyState bo = fastIsA ? bb : ba;
    uint slot = ccdSlotOf(fastBody, bf);
    if (slot == CCD_NO_PAIR) return;
    float3 r = ccdRelativeMotion(bf, bo);
    if (dot(r, r) < 1e-12) return;
    float t;
    float3 n;
    uint status = ccdCast(fastIsA ? ca : cb, bf, fastIsA ? cb : ca, bo, r, t, n);
    if (status == CAST_BUDGET)
    {
        // Left to look-ahead (its contact still exists); counted, not failed.
        AADD2(gScratchAtomics[CCD_FALLBACKS], 1u);
        return;
    }
    if (status != CAST_HIT || !ccdTOIMatters(t, n, r)) return;
    gCandidates[i]._pad = asuint(t) + 1u;
    AMIN2(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 1u], asuint(t));
}

// The other side's collider of candidate `pair` at its CCD body's minimum TOI
// (slot `slot`), or false.
bool ccdAtMinimum(CandidatePair pair, out uint slot, out uint other)
{
    slot = CCD_NO_PAIR;
    other = 0u;
    if ((pair._pad & (CONVEX_DEFER | CCD_WIN)) != 0u || pair._pad == 0u) return false;
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    if (!bodyIsCcd(ba) && !bodyIsCcd(bb)) return false;
    bool fastIsA = ccdFastIsA(ba, ca.body, bb, cb.body);
    slot = ccdSlotOf(fastIsA ? ca.body : cb.body, fastIsA ? ba : bb);
    if (slot == CCD_NO_PAIR) return false;
    other = fastIsA ? pair.colliderB : pair.colliderA;
    return pair._pad - 1u == ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 1u]);
}

// Second pass: among pairs at their body's minimum TOI, the least other
// collider (its complement's maximum in carry word 0). The candidates are
// appended with atomics, so their order is no tie-break: coplanar pieces met
// at one TOI (a structure split in the same frame) gave a different winner run
// to run.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_pick_key(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= min(ALOAD(gCounters[0].candidateCount), HOST_S(2))) return;
    uint slot;
    uint other;
    if (ccdAtMinimum(gCandidates[i], slot, other))
        AMAX2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot], ~other);
}

// Third pass: of those, the pair with that collider wins; the lowest candidate
// index only between pairs of one CCD body's colliders with the same one (a
// compound CCD body).
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_pick(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= min(ALOAD(gCounters[0].candidateCount), HOST_S(2))) return;
    CandidatePair pair = gCandidates[i];
    if ((pair._pad & (CONVEX_DEFER | CCD_WIN)) != 0u) return;
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    if (!bodyIsCcd(ba) && !bodyIsCcd(bb)) return;
    // Nor a post-solve sweep check (validate, clamp, respond) against them.
    if (((ca.material | cb.material) & COLLIDER_SENSOR) != 0u) return;
    bool fastIsA = ccdFastIsA(ba, ca.body, bb, cb.body);
    uint slot = ccdSlotOf(fastIsA ? ca.body : cb.body, fastIsA ? ba : bb);
    if (slot == CCD_NO_PAIR) return;
    // For the post-solve sweep check (the candidate buffer is the constraint
    // graph's by then).
    uint at;
    AADD3(gScratchAtomics[CCD_PAIR_COUNT], 1u, at);
    if (at < CCD_PAIR_CAP)
    {
        ASTORE(gScratchAtomics[CCD_PAIR_BASE + 2u * at], pair.colliderA);
        ASTORE(gScratchAtomics[CCD_PAIR_BASE + 2u * at + 1u], pair.colliderB);
    }
    else if (at == CCD_PAIR_CAP) AOR2(gCounters[0].overflowFlags, OVERFLOW_CCD_PAIRS);
    if (pair._pad != 0u)
    {
        uint s2;
        uint other;
        if (ccdAtMinimum(pair, s2, other) && ~other == ALOAD(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot]))
            AMIN2(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 2u], i);
        return;
    }
    // Found only through the deflection pad (clear of the whole straight
    // sweep): the post-solve sweep check uses the pair, the solve needs no
    // contact row. Pairs along the path keep theirs: a striker that pushes its
    // first partner along meets the next ones in the same solve.
    if (ccdApartAlongSweep(ca, ba, cb, bb, 1.0)) gCandidates[i]._pad = CCD_BEYOND;
}

// Third pass, one thread per CCD body: tag its winning candidate in place, so
// the narrowphase and ccd_emit find it without the scratch buffer (the
// narrowphase entries are at WebGPU's ten storage buffers).
static const uint CCD_WIN = 0x40000000u;

#include "ccd_step_mark.slang"
#include "ccd_step_store.slang"
