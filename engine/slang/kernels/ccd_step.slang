// Continuous collision and step-start impulses (engine/docs/ccd.md).
// Included by world_step.slang, so every variant (canonical, compact,
// persistent, reuse-copy) gets these entries with its own contact storage.
//
// Step placement (engine_exec::rigid_plan::encode_step):
//   apply_impulses → … → broadphase_clear → ccd_register → broadphase_insert
//   → broadphase_query → broadphase_fast → … → ccd_toi → ccd_pick
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
            uint tables = gridSplit() ? 2u : 1u;
            for (uint t = 0u; t < tables; t++)
            {
                uint link = ALOAD(gConstraintIds[t * gridHashSize() + gridHash(cell.x, cell.y, cell.z)]);
                uint guard = 0u;
                while (link != 0xFFFFFFFFu && guard < 4096u)
                {
                    uint cur = link & GRID_COLLIDER_MASK;
                    Aabb ab = gAabbs[base + cur];
                    link = gridSplit() ? asuint(ab._pad1) : gColorLists[cur];
                    guard++;
                    if (all(int3(floor((0.5 * (ab.mn + ab.mx)) / cs)) == cell))
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
uint gjkCast(Collider ca, float3 pa, float4 qa, Collider cb, float3 pb, float4 qb,
    float3 s0, float3 r, float muMax, float eps, out float mu, out float3 n)
{
    mu = 0.0;
    n = float3(0, 0, 0);
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
        if (dot(v, v) <= eps * eps) return advanced ? CAST_HIT : CAST_OVERLAP;
        float3 p = shapeSupport(cb, pb, qb, v) - shapeSupport(ca, pa, qa, -v);
        float3 w = x - p;
        float vw = dot(v, w);
        if (vw > 0.0)
        {
            float vr = dot(v, r);
            if (vr >= 0.0) return CAST_MISS;
            mu -= vw / vr;
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
            float touch = max(10.0 * eps, GJK_TOUCH);
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

// Second pass: among pairs at their body's minimum TOI, the lowest candidate
// index wins (deterministic whatever the thread order).
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
    uint minBits = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 1u]);
    if (pair._pad != 0u)
    {
        if (pair._pad - 1u == minBits) AMIN2(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 2u], i);
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

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_mark(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (k >= ccdBodyCount()) return;
    uint win = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k + 2u]);
    if (win == CCD_NO_PAIR) return;
    gCandidates[win]._pad |= CCD_WIN;
    // The impact response (ccd_respond): both bodies' step-start velocities.
    CandidatePair pair = gCandidates[win];
    uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    uint other = gColliders[pair.colliderA].body == fast ? gColliders[pair.colliderB].body : gColliders[pair.colliderA].body;
    BodyState bo = gBodies[other];
    if (!bodyIsDynamic(bo)) return;
    float3 v = gBodies[fast].linearVel;
    float3 vo = bodyIsSleeping(bo) || wakePending(bo) ? float3(0, 0, 0) : bo.linearVel;
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    ASTORE(gScratchAtomics[r], asuint(v.x));
    ASTORE(gScratchAtomics[r + 1u], asuint(v.y));
    ASTORE(gScratchAtomics[r + 2u], asuint(v.z));
    bool fastA = gColliders[pair.colliderA].body == fast;
    ASTORE(gScratchAtomics[r + 3u], fastA ? pair.colliderA : pair.colliderB);
    ASTORE(gScratchAtomics[r + 9u], fastA ? pair.colliderB : pair.colliderA);
    ASTORE(gScratchAtomics[r + 4u], asuint(vo.x));
    ASTORE(gScratchAtomics[r + 5u], asuint(vo.y));
    ASTORE(gScratchAtomics[r + 6u], asuint(vo.z));
    ASTORE(gScratchAtomics[r + 7u], other);
    // Held still this step (asleep, or its wake deferred): the solve treated it
    // as immovable, so the response resolves the whole impact.
    ASTORE(gScratchAtomics[r + 15u], bodyIsSleeping(bo) || wakePending(bo) ? 1u : 0u);
    ASTORE(gScratchAtomics[CCD_BODY_BASE + 4u * k + 3u], CCD_RESPOND);
}

// ---------------------------------------------------------------------------
// Post-solve sweep check (after the solve, before extract_velocity_sleep). The
// TOI contact places the first hit and the look-ahead contacts hold the rest,
// but in dense rubble the solve can still push a CCD body into a piece it was
// paired with (wedged among rigid contacts, some against bodies held still
// this step). ccd_validate casts each CCD body's solved motion this step
// relative to each candidate partner, from both step-start poses; ccd_clamp
// pulls the body back along its own path to the first contact found. Its
// bounds are padded by the farthest the solve can deflect it, so every body
// it can reach is a candidate: it never ends a step past a surface.
// ---------------------------------------------------------------------------
// Whether the segment a + s·d (s in [0, 1]) enters collider c posed at
// (pos, rot) from outside, and where (s). Exact for boxes and hulls (the slab
// test over their faces) and spheres; other shapes fall back to a GJK point
// cast. A start inside is not an entry (false).
bool segmentEnters(Collider c, float3 pos, float4 rot, float3 a, float3 d, out float s)
{
    s = 1.0;
    if (c.shape == SHAPE_HEIGHTFIELD)
    {
        Frame f; f.p = pos; f.q = rot;
        HeightFieldRef hf = heightFieldOf(c);
        float3 lo = frameToLocal(f, a);
        float3 ld = quatRotate(quatConjugate(rot), d);
        float h; float3 nn;
        if (hfSurface(hf, lo.x, lo.z, h, nn) && lo.y < h) return false; // starts under it
        float len = length(ld);
        float th;
        if (len < 1e-12 || !heightFieldRayLocal(hf, lo, ld / len, len, th, nn)) return false;
        s = th / len;
        return s > 0.0 && s <= 1.0;
    }
    float4 qi = quatConjugate(rot);
    float3 lo = quatRotate(qi, a - pos);
    float3 ld = quatRotate(qi, d);
    float tNear = -1e30, tFar = 1e30;
    if (c.shape == SHAPE_BOX)
    {
        for (int k = 0; k < 3; k++)
        {
            float h = c.params[k];
            if (abs(ld[k]) < 1e-12)
            {
                if (abs(lo[k]) > h) return false;
                continue;
            }
            float t0 = (-h - lo[k]) / ld[k];
            float t1 = (h - lo[k]) / ld[k];
            tNear = max(tNear, min(t0, t1));
            tFar = min(tFar, max(t0, t1));
            if (tNear > tFar) return false;
        }
    }
    else if (c.shape == SHAPE_CONVEX)
    {
        HullRef hr = hullOf(c);
        for (uint f = 0u; f < hr.nf; f++)
        {
            float4 pl = hullPlane(hr, f);
            float denom = dot(pl.xyz, ld);
            float num = pl.w - dot(pl.xyz, lo);
            if (abs(denom) < 1e-12)
            {
                if (num < 0.0) return false;
                continue;
            }
            float tt = num / denom;
            if (denom < 0.0) tNear = max(tNear, tt);
            else tFar = min(tFar, tt);
            if (tNear > tFar) return false;
        }
    }
    else if (c.shape == SHAPE_SPHERE)
    {
        float r = c.params.x;
        float dd = dot(ld, ld);
        if (dd < 1e-20) return false;
        float b = dot(lo, ld);
        float cc = dot(lo, lo) - r * r;
        float disc = b * b - dd * cc;
        if (disc < 0.0) return false;
        tNear = (-b - sqrt(disc)) / dd;
        tFar = (-b + sqrt(disc)) / dd;
    }
    else
    {
        Collider probe = c;
        probe.shape = SHAPE_SPHERE;
        probe.params = float4(0.0, 0.0, 0.0, 0.0);
        float mu;
        float3 n;
        // The point (a) cast against c along d, in a's frame.
        uint st = gjkCast(probe, float3(0, 0, 0), float4(0, 0, 0, 1), c, pos - a, rot, float3(0, 0, 0), d, 1.0, 1.0e-4, mu, n);
        s = mu;
        return st == CAST_HIT;
    }
    // Entry from outside within the segment.
    if (tNear <= 0.0 || tNear > 1.0 || tFar < tNear) return false;
    s = tNear;
    return true;
}

// Smallest half extent of a collider (sphere and capsule: the radius; hull:
// half its bound radius, a conservative stand-in).
float colliderThickness(Collider c)
{
    if (c.shape == SHAPE_SPHERE || c.shape == SHAPE_CAPSULE) return c.params.x;
    if (c.shape == SHAPE_BOX) return min(c.params.x, min(c.params.y, c.params.z));
    if (c.shape == SHAPE_CONVEX) return 0.5 * c.params.w;
    return 1.0e30;
}

// Step-start pose of a body: x⁻ (predict saved it), or the pose itself for a
// body the step did not move (asleep, held, static).
void stepStartPose(BodyState b, out float3 x, out float4 q)
{
    bool moved = bodyMovingOrKinematic(b) && !wakePending(b);
    x = moved ? b.initialPos : b.position;
    q = moved ? b.initialRot : b.orientation;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_validate(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= min(ALOAD(gScratchAtomics[CCD_PAIR_COUNT]), CCD_PAIR_CAP)) return;
    uint colA = ALOAD(gScratchAtomics[CCD_PAIR_BASE + 2u * i]);
    uint colB = ALOAD(gScratchAtomics[CCD_PAIR_BASE + 2u * i + 1u]);
    for (uint side = 0u; side < 2u; side++)
    {
        Collider cf = gColliders[side == 0u ? colA : colB];
        Collider co = gColliders[side == 0u ? colB : colA];
        if (cf.body == co.body || co.shape == SHAPE_PLANE) continue;
        BodyState bf = gBodies[cf.body];
        if (!bodyIsCcd(bf)) continue;
        uint slot = ccdSlotOf(cf.body, bf);
        if (slot == CCD_NO_PAIR) continue;
        BodyState bo = gBodies[co.body];
        float3 xf0; float4 qf0; stepStartPose(bf, xf0, qf0);
        float3 xo0; float4 qo0; stepStartPose(bo, xo0, qo0);
        // A body that cannot move this step (static, kinematic, asleep or held,
        // its wake deferred): any real crossing.
        // Or one the solve did not move (no row moved it: a piece past the
        // first impact a 1,152 m/s ball reached within its step, woken by
        // ccd_wake but not yet pushed): it behaved as a fixed one did, so the
        // ball stops at its surface, not with its centre at it.
        float4 qoNow = dot(bo.orientation, bo.initialRot) < 0.0 ? -bo.orientation : bo.orientation;
        bool stayedStill = bodyMovingOrKinematic(bo) && !wakePending(bo)
            && length(bo.position - bo.initialPos) < 1e-3 && dot(qoNow, bo.initialRot) > 1.0 - 1e-6;
        bool fixedThisStep = !bodyIsDynamic(bo) || bodyIsSleeping(bo) || wakePending(bo) || stayedStill;
        // An awake body (moved or not): only the CCD body's centre entering it,
        // along their relative motion, counts. A centre inside a piece is a
        // pass-through in the making (it comes out the far side once the piece
        // stops); a heavy ball's overlap with bricks it pushes, or that the
        // solve has not pushed yet, is not.
        // Relative motion of the CCD body over the step, against the other at
        // its step-start pose.
        float3 d = (bf.position - xf0) - (bo.position - xo0);
        if (dot(d, d) < 1e-10) continue;
        BodyState sf = bf; sf.position = xf0; sf.orientation = qf0;
        BodyState so = bo; so.position = xo0; so.orientation = qo0;
        float3 pf; float4 qf; colliderPose(cf, sf, pf, qf);
        float3 po; float4 qo; colliderPose(co, so, po, qo);
        float mu = 1.0;
        float3 n = float3(0, 0, 0);
        float eps = 1.0e-4 + 1.0e-5 * length(d);
        // A body that cannot move: the whole shape, cast (GJK). Already
        // touching it (resting against it in rubble), or an awake body: its
        // centre must not go in (an exact segment test, not a cast: a GJK
        // point cast grazing a box edge can stall and miss).
        uint status = CAST_OVERLAP;
        if (fixedThisStep && co.shape == SHAPE_HEIGHTFIELD)
        {
            Frame fo; fo.p = po; fo.q = qo;
            float len = length(d);
            float tc; float3 nc;
            status = CAST_MISS;
            if (heightFieldShapeCast(heightFieldOf(co), fo, cf, pf, qf, d / len, len, tc, nc))
            {
                status = tc <= 0.0 ? CAST_OVERLAP : CAST_HIT;
                mu = tc / len;
                n = -nc;
            }
        }
        else if (fixedThisStep)
            status = gjkCast(cf, float3(0, 0, 0), qf, co, po - pf, qo, float3(0, 0, 0), d, 1.0, eps, mu, n);
        bool centreOnly = status == CAST_OVERLAP;
        // An awake body the CCD body is shoving along (it moved at least half as
        // far this step): the ball is pushing it, not passing it. Pulling the
        // ball back there froze it for a tick at each brick it ploughed (a
        // visible stutter) while its velocity carried on.
        if (!fixedThisStep && 2.0 * length(bo.position - xo0) >= length(bf.position - xf0)) continue;
        if (centreOnly)
        {
            if (!segmentEnters(co, po, qo, pf, d, mu)) continue;
            status = CAST_HIT;
        }
        if (status != CAST_HIT) continue;
        // Only a real crossing: deeper than a quarter of the thinner body's
        // smallest half extent (contact depth, rolling and resting sink stay
        // the solve's; a centre crossing is far past it).
        // Against a dynamic piece, a quarter of the CCD body's own thickness if
        // held, half if awake but still (its centre still never reaches it): the thinner body's let a 3 cm roof plate or
        // fence picket stop a 1.6 m ball at 4 mm of contact sink, a pull-back
        // nearly every tick in rubble (stop motion). Static geometry: as tight.
        float depth = bodyIsDynamic(bo)
            ? max(2.0 * PARAMS.contactOffset, (stayedStill ? 0.5 : 0.25) * colliderThickness(cf))
            : max(2.0 * PARAMS.contactOffset, 0.25 * min(colliderThickness(cf), colliderThickness(co)));
        if (!centreOnly && (1.0 - mu) * dot(d, n) <= depth) continue;
        AMIN2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 8u], asuint(mu));
        // First contact with a body that cannot move (static, kinematic).
        if (!bodyIsDynamic(bo)) AMIN2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 10u], asuint(mu));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_clamp(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (k >= ccdBodyCount()) return;
    uint bits = ALOAD(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * k + 8u]);
    if (bits == CCD_TOI_NONE) return;
    uint body = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    BodyState b = gBodies[body];
    float3 x0; float4 q0; stepStartPose(b, x0, q0);
    float3 d = b.position - x0;
    float len = length(d);
    if (len < 1e-8) return;
    // Back to the first contact, a contact offset short of it.
    float t = clamp(asfloat(bits) - PARAMS.contactOffset / len, 0.0, 1.0);
    float3 x = x0 + d * t;
    float4 q1 = dot(q0, b.orientation) < 0.0 ? -b.orientation : b.orientation;
    gBodies[body].position = x;
    gBodies[body].orientation = quatNormalize(q0 + (q1 - q0) * t);
    // Stopped by a dynamic body (held still this step, its wake deferred, or
    // pinned): it keeps the velocity of its solved motion and meets that body,
    // awake, next step. Stopped by static geometry: the clamped motion is its
    // velocity (it presses no further).
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    bool byStatic = ALOAD(gScratchAtomics[r + 10u]) <= bits;
    float3 v = (byStatic ? x - x0 : d) / max(PARAMS.dt, 1e-8);
    ASTORE(gScratchAtomics[r + 12u], asuint(v.x));
    ASTORE(gScratchAtomics[r + 13u], asuint(v.y));
    ASTORE(gScratchAtomics[r + 14u], asuint(v.z));
    ASTORE(gScratchAtomics[r + 11u], 1u);
    if (postStabilize()) gBodies[body].linearVel = v;
}

// The velocity ccd_clamp left for a clamped CCD body (extract_velocity_sleep
// takes it instead of the one its clamped motion gives).
bool ccdClampedVelocity(uint body, BodyState b, out float3 v)
{
    v = float3(0, 0, 0);
    if (!sceneHasCcd()) return false;
    uint slot = ccdSlotOf(body, b);
    if (slot == CCD_NO_PAIR) return false;
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * slot;
    if (ALOAD(gScratchAtomics[r + 11u]) == 0u) return false;
    v = asfloat(uint3(ALOAD(gScratchAtomics[r + 12u]), ALOAD(gScratchAtomics[r + 13u]), ALOAD(gScratchAtomics[r + 14u])));
    return true;
}

// Whether body `b` lies within `reach` × the CCD collider's radius, plus its own
// extent, of the swept segment this step of some CCD body moving more than four
// contact offsets a step (ccdUnholds).
bool ccdNearFastSweepAt(uint body, BodyState b, float reach, bool atStart)
{
    float own = 0.0;
    uint nc = gCounters[0].colliderCount;
    for (uint c = firstColliderOfBody(body); c < nc && gColliders[c].body == body; c++)
        own = max(own, colliderExtent(gColliders[c]));
    uint n = ccdBodyCount();
    for (uint k = 0u; k < n; k++)
    {
        uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
        BodyState bf = gBodies[fast];
        if (!ccdUnholds(bf)) continue;
        float r = 0.0;
        for (uint c = firstColliderOfBody(fast); c < nc && gColliders[c].body == fast; c++)
            r = max(r, colliderExtent(gColliders[c]));
        float3 a = atStart ? bf.position : bf.initialPos;
        float3 d = bf.linearVel * PARAMS.dt;
        float t = clamp(dot(b.position - a, d) / max(dot(d, d), 1e-20), 0.0, 1.0);
        if (length(a + d * t - b.position) <= reach * r + own) return true;
    }
    return false;
}

// Before predict_bodies a CCD body's pose is its step start; after it, its
// initialPos is (predict saved it and moved the body).
bool ccdNearFastSweepStart(uint body, BodyState b, float reach) { return ccdNearFastSweepAt(body, b, reach, true); }

// Sleepers in a fast CCD body's path wake before the narrowphase (after
// ccd_pick), so they get their contacts and are solved this step. Left asleep,
// a ball reaching a sleeping piece past its first impact had no contact with it,
// the post-solve check stopped the ball at it as at a wall, and the next tick
// at the next one: in rubble the ball stalled every tick (stop motion). Within
// 1.25 radii of the CCD collider of its swept segment, plus the piece's own size.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_wake(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    // Step start: initialPos is not yet this step's (predict saves it).
    if (!ccdNearFastSweepStart(i, b, 1.25)) return;
    gBodies[i].flags = (b.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
    gBodies[i].sleepTimer = 0.0;
}

// Woken near a fast CCD body: solved this step (before predict_bodies). A body
// the narrowphase wakes keeps still for the rest of its step (FLAG_DEFER_WAKE:
// solved without its supports, a woken brick in a pile started with a kick).
// But where a fast CCD body ploughs in, those bodies stood like a fixed wall
// for that step: the bricks it struck were backed by neighbours woken in the
// same step and held still, so a 200 t ball took its impulse off them (99% of
// the momentum it lost in a suburb of brick houses went into held bodies) and
// was deflected, often upward over the wall. So a body whose wake is deferred
// is solved this step when it is within reach of a CCD body moving more than
// four contact offsets a step: three of the CCD collider's radii, plus its own,
// from the CCD body's swept segment this step. A CCD slot whose impact partner
// is released no longer treats it as held (ccd_respond: the solve exchanges the
// momentum). Pure function of the step-start state: every thread agrees.
bool ccdReleasesHeld(uint body, BodyState b)
{
    if (!wakePending(b) || bodyIsSleeping(b) || !bodyIsDynamic(b)) return false;
    return ccdNearFastSweepStart(body, b, 3.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_unhold(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (ccdReleasesHeld(i, b)) gBodies[i].flags = b.flags & ~BODY_WAKE_PENDING;
    if (i < ccdBodyCount())
    {
        uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * i;
        if ((ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * i + 3u]) & CCD_RESPOND) != 0u)
        {
            uint other = ALOAD(gScratchAtomics[r + 7u]);
            // Read before any thread clears the partner's flag: its own thread
            // clears only the deferred-wake bit, which ccdReleasesHeld tests.
            BodyState bo = gBodies[other];
            if (ccdReleasesHeld(other, bo) || (!bodyIsSleeping(bo) && !wakePending(bo))) ASTORE(gScratchAtomics[r + 15u], 0u);
        }
    }
}

// Trust region of the bodies a CCD body can strike (after predict_bodies,
// before the solve). A free body struck by a fast one takes the impact through
// the solve, a position correction of up to the closing speed times h, but the
// primal trust region lets an ordinary body move only iterations · trustLin ·
// 0.5 (1.75 m) in a step: a 300 m/s ball into a free 2 kg box (inelastic
// 214 m/s, 3.6 m) kept 85% of the momentum. So for this step every awake
// dynamic body paired with a CCD body (ccd_pick's list) gets a trust region
// covering the largest CCD travel this step, read by primalApply from its
// inertial-target record's spare word (predict_bodies just zeroed it). Every
// write is the same value, so concurrent ones agree.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_trust(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= min(ALOAD(gScratchAtomics[CCD_PAIR_COUNT]), CCD_PAIR_CAP)) return;
    float reach = asfloat(ALOAD(gScratchAtomics[CCD_MAX_REACH]));
    for (uint side = 0u; side < 2u; side++)
    {
        uint body = gColliders[ALOAD(gScratchAtomics[CCD_PAIR_BASE + 2u * i + side])].body;
        BodyState b = gBodies[body];
        if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b) || bodyIsCcd(b)) continue;
        gAabbs[body]._pad1 = reach;
    }
}

// The primal step of body i under the trust region (trustLin · 0.5 per
// iteration). A CCD body's iterate starts at its time of impact: held to that
// trust it could advance only iterations · 0.175 m (1.75 m) in a step, so a
// fast ball pushing a free box lost the rest of its motion and with it the
// momentum. So along its velocity it may move as far as its step's travel
// over the iterations, its position along the sweep staying within the swept
// segment padded by the ordinary bound (ccdDeflectionBound), and across it the
// ordinary trust still holds: its broadphase pad stays a hard bound. A body a
// CCD body can strike gets that reach all round (ccd_trust).
float3 ccdTrustStep(uint i, BodyState b, float3 dlin)
{
    float base = PARAMS.trustLin * 0.5;
    float iters = max(float(PARAMS.iterations), 1.0);
    float sp = length(b.linearVel);
    if (sceneHasCcd() && bodyIsCcd(b) && sp > 1e-6)
    {
        float3 u = b.linearVel / sp;
        float along = dot(dlin, u);
        float3 perp = dlin - u * along;
        float pl = length(perp);
        if (pl > base) perp *= base / pl;
        float reach = sp * PARAMS.dt;
        along = clamp(along, -max(base, 2.0 * reach / iters), max(base, 2.0 * reach / iters));
        // Position along the sweep: within [x⁻, x⁻ + v·h], padded.
        float pad = ccdDeflectionBound(reach);
        float s0 = dot(b.position - b.initialPos, u);
        along = clamp(s0 + along, min(s0, -pad), max(s0, reach + pad)) - s0;
        return perp + u * along;
    }
    float maxLin = base;
    if (sceneHasCcd() && !bodyIsCcd(b))
        maxLin = max(maxLin, 2.0 * gAabbs[i]._pad1 / iters);
    float len = length(dlin);
    if (len > maxLin && len > 1e-12) dlin *= maxLin / len;
    return dlin;
}

// ccd slot flags word (+3): set by ccd_mark when the TOI partner is dynamic.
static const uint CCD_RESPOND = 1u;

// ---------------------------------------------------------------------------
// Impact response (after the sleep pass). A rigid TOI contact stops a CCD body
// at its partner's surface, but the solve barely passes momentum through a
// rigid contact in one step's iterations, and a partner held still this step
// (asleep, its wake deferred) or pinned by such neighbours takes none: the
// striker would stop dead. So each CCD body that struck a dynamic body gets a
// perfectly inelastic, momentum-conserving exchange along the TOI normal from
// both bodies' step-start velocities, with both bodies' rotational effective
// mass (the partner also spins, from the contact point), and a held partner
// wakes with its new velocity. Against a free partner the solve's result is
// already close, so this changes little; kinetic energy never grows.
//
// Two passes, one thread per CCD slot: ccd_respond computes each slot's
// velocity changes from the bodies as the solve left them (reads only);
// ccd_respond_apply has one owner thread per affected body (its own slot if it
// struck, else the lowest slot that struck it), which sums every change
// addressed to it in slot order. (One thread looping over the slots cost 3 ms
// at 512 strikers.)
// ---------------------------------------------------------------------------
static const uint CCD_RESP = 16u; // carry words: flag, striker dv, dw, partner dv, dw

void ccdStoreVec(uint at, float3 v)
{
    ASTORE(gScratchAtomics[at], asuint(v.x));
    ASTORE(gScratchAtomics[at + 1u], asuint(v.y));
    ASTORE(gScratchAtomics[at + 2u], asuint(v.z));
}
float3 ccdLoadVec(uint at)
{
    return asfloat(uint3(ALOAD(gScratchAtomics[at]), ALOAD(gScratchAtomics[at + 1u]), ALOAD(gScratchAtomics[at + 2u])));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_respond(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (!sceneHasCcd() || k >= ccdBodyCount()) return;
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    ASTORE(gScratchAtomics[r + CCD_RESP], 0u);
    if ((ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k + 3u]) & CCD_RESPOND) == 0u) return;
    uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    float3 v0 = ccdLoadVec(r);
    float3 vo0 = ccdLoadVec(r + 4u);
    uint other = ALOAD(gScratchAtomics[r + 7u]);
    // The winning pair's colliders (ccd_mark): the candidate buffer is the
    // constraint graph's by now.
    Collider cf = gColliders[ALOAD(gScratchAtomics[r + 3u])];
    Collider co = gColliders[ALOAD(gScratchAtomics[r + 9u])];
    BodyState bf = gBodies[fast];
    BodyState bo = gBodies[other];
    if (!bodyIsDynamic(bo) || !bodyIsDynamic(bf)) return;
    // The TOI normal again, from the step-start poses (a held partner did not
    // move) along the step's relative motion.
    BodyState bf0 = bf;
    bf0.position = bf.initialPos;
    bf0.orientation = bf.initialRot;
    float3 rel = v0 * PARAMS.dt;
    float t;
    float3 nrm;
    if (ccdCast(cf, bf0, co, bo, rel, t, nrm) != CAST_HIT || dot(nrm, nrm) < 0.5) nrm = normalize(rel);
    float closing = dot(v0 - vo0, nrm);
    if (closing <= 0.0) return;
    // The perfectly inelastic impulse along n at the contact point, with both
    // bodies' rotational effective mass (a linear-only impulse with the torque
    // added on top over-counts it off centre: struck bricks spun to 15,000 rad/s).
    float3 pf; float4 qf; colliderPose(cf, bf, pf, qf);
    float3 at = shapeSupport(cf, pf, qf, nrm);
    float3 rfn = cross(at - bf.position, nrm);
    float3 ron = cross(at - bo.position, nrm);
    float kf = bf.invMass + dot(rfn, applyInvInertia(bf, rfn));
    float ko = bo.invMass + dot(ron, applyInvInertia(bo, ron));
    float J = closing / max(kf + ko, 1e-20);
    // Each body's normal speed is only ever raised over the solve's result:
    // the striker to what the impulse leaves it, the partner to what it gives.
    // Only a partner held still this step needs it: the solve treated it as
    // immovable (the striker stopped against it as against a wall, its motion
    // across the normal too) and it took no momentum. A free partner's
    // contacts already exchange momentum in the solve; adjusting only one
    // side's speed there broke conservation (an edge hit that spun the box
    // into the ball again lost 57% of the momentum).
    if (ALOAD(gScratchAtomics[r + 15u]) == 0u) return;
    // The whole frictionless impact from the step-start velocities.
    ccdStoreVec(r + CCD_RESP + 1u, (v0 - nrm * (J * bf.invMass)) - bf.linearVel);
    ccdStoreVec(r + CCD_RESP + 4u, -applyInvInertia(bf, rfn * J));
    ccdStoreVec(r + CCD_RESP + 7u, (vo0 + nrm * (J * bo.invMass)) - bo.linearVel);
    ccdStoreVec(r + CCD_RESP + 10u, applyInvInertia(bo, ron * J));
    ASTORE(gScratchAtomics[r + CCD_RESP], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_respond_apply(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    uint n = ccdBodyCount();
    if (!sceneHasCcd() || k >= n) return;
    uint rk = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    bool struck = ALOAD(gScratchAtomics[rk + CCD_RESP]) != 0u;
    uint other = ALOAD(gScratchAtomics[rk + 7u]);
    // Owner of `other` (it struck nothing itself, so no slot owns it as a
    // striker): this slot, if no lower slot struck it too.
    bool ownsOther = struck;
    for (uint j = 0u; j < n && ownsOther; j++)
    {
        uint rj = CCD_CARRY_BASE + CCD_CARRY_WORDS * j;
        if (ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * j]) == other) ownsOther = false;
        else if (j < k && ALOAD(gScratchAtomics[rj + CCD_RESP]) != 0u && ALOAD(gScratchAtomics[rj + 7u]) == other) ownsOther = false;
    }
    // This slot's striker always belongs to this thread; `other` too if owned.
    for (uint pass = 0u; pass < 2u; pass++)
    {
        uint b = pass == 0u ? fast : other;
        if (pass == 1u && !ownsOther) break;
        float3 dv = float3(0, 0, 0), dw = float3(0, 0, 0);
        bool any = false;
        for (uint j = 0u; j < n; j++)
        {
            uint rj = CCD_CARRY_BASE + CCD_CARRY_WORDS * j;
            if (ALOAD(gScratchAtomics[rj + CCD_RESP]) == 0u) continue;
            if (ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * j]) == b)
            {
                dv += ccdLoadVec(rj + CCD_RESP + 1u);
                dw += ccdLoadVec(rj + CCD_RESP + 4u);
                any = true;
            }
            if (ALOAD(gScratchAtomics[rj + 7u]) == b)
            {
                dv += ccdLoadVec(rj + CCD_RESP + 7u);
                dw += ccdLoadVec(rj + CCD_RESP + 10u);
                any = true;
            }
        }
        if (!any) continue;
        // This thread owns body b: one load, one store.
        BodyState bb = gBodies[b];
        bb.linearVel += dv;
        // Under the per-step rotation limit as extract_velocity_sleep is (it ran
        // before this pass): an edge hit spun a box to 6 rad a step.
        float3 w = bb.angularVel + dw;
        if ((PARAMS.flags & FLAG_MAX_ROTATION) != 0u)
        {
            float wMax = MAX_ROTATION / max(PARAMS.dt, 1e-8);
            if (length(w) > wMax) w *= wMax / length(w);
        }
        bb.angularVel = w;
        if (pass == 1u)
        {
            bb.flags = (bb.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
            bb.sleepTimer = 0.0;
        }
        gBodies[b] = bb;
    }
}

// World inverse inertia times v (diagonal in the body frame).
float3 applyInvInertia(BodyState b, float3 v)
{
    float3 local = quatRotate(quatConjugate(b.orientation), v);
    return quatRotate(b.orientation, local * b.invInertia);
}

// The narrowphase leaves each CCD body's winning pair to ccd_emit.
bool ccdOwnsPair(CandidatePair pair)
{
    return (pair._pad & (CCD_WIN | CONVEX_DEFER)) == CCD_WIN;
}

// Fraction of its motion a CCD body is predicted forward (1: no TOI).
float ccdStartFraction(uint body, BodyState b)
{
    if (!sceneHasCcd()) return 1.0;
    uint slot = ccdSlotOf(body, b);
    if (slot == CCD_NO_PAIR) return 1.0;
    uint bits = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 1u]);
    return bits == CCD_TOI_NONE ? 1.0 : clamp(asfloat(bits), 0.0, 1.0);
}

// Support features of a collider at pose (pos, rot) along n: the points whose
// support value is within tol of the maximum (a face, an edge or a vertex),
// at most 8.
void supportFeature(Collider c, float3 pos, float4 rot, float3 n, float tol, inout float3 pts[8], inout uint count)
{
    count = 0u;
    if (c.shape == SHAPE_SPHERE)
    {
        pts[0] = pos + n * c.params.x;
        count = 1u;
        return;
    }
    if (c.shape == SHAPE_CAPSULE)
    {
        float3 axis = quatRotate(rot, float3(0, 1, 0));
        float3 e0 = pos - axis * c.params.y, e1 = pos + axis * c.params.y;
        float d0 = dot(e0, n), d1 = dot(e1, n);
        if (abs(d0 - d1) <= tol) { pts[0] = e0 + n * c.params.x; pts[1] = e1 + n * c.params.x; count = 2u; }
        else { pts[0] = (d0 > d1 ? e0 : e1) + n * c.params.x; count = 1u; }
        return;
    }
    float best = -1e30;
    if (c.shape == SHAPE_BOX)
    {
        float3 h = c.params.xyz;
        for (uint k = 0u; k < 8u; k++)
        {
            float3 l = float3((k & 1u) != 0u ? h.x : -h.x, (k & 2u) != 0u ? h.y : -h.y, (k & 4u) != 0u ? h.z : -h.z);
            best = max(best, dot(pos + quatRotate(rot, l), n));
        }
        for (uint k = 0u; k < 8u; k++)
        {
            float3 l = float3((k & 1u) != 0u ? h.x : -h.x, (k & 2u) != 0u ? h.y : -h.y, (k & 4u) != 0u ? h.z : -h.z);
            float3 w = pos + quatRotate(rot, l);
            if (dot(w, n) >= best - tol && count < 8u) { pts[count] = w; count++; }
        }
        return;
    }
    HullRef h = hullOf(c);
    for (uint k = 0u; k < h.nv; k++) best = max(best, dot(pos + quatRotate(rot, hullVertex(h, k)), n));
    for (uint k = 0u; k < h.nv; k++)
    {
        float3 w = pos + quatRotate(rot, hullVertex(h, k));
        if (dot(w, n) >= best - tol && count < 8u) { pts[count] = w; count++; }
    }
}

// Whether p lies within a collider's local bounds at pose (pos, rot), padded.
bool pointNearCollider(float3 p, Collider c, float3 pos, float4 rot, float pad)
{
    float3 l = quatRotate(quatConjugate(rot), p - pos);
    if (c.shape == SHAPE_SPHERE) return length(l) <= c.params.x + pad;
    if (c.shape == SHAPE_CAPSULE)
        return length(float3(l.x, max(abs(l.y) - c.params.y, 0.0), l.z)) <= c.params.x + pad;
    if (c.shape == SHAPE_BOX) return all(abs(l) <= c.params.xyz + pad);
    if (c.shape == SHAPE_HEIGHTFIELD)
    {
        float3 mn, mx;
        hfLocalBounds(heightFieldOf(c), mn, mx);
        return all(l >= mn - pad) && all(l <= mx + pad);
    }
    HullRef h = hullOf(c);
    return all(abs(l - hullLocalCenter(h)) <= hullLocalHalf(h) + pad);
}

// One thread per candidate (work slot 64); the CCD winners (ccd_mark) build
// their manifold at the TOI pose and emit it as a speculative contact whose
// rows hold the gap along the path.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_emit(uint3 tid : SV_DispatchThreadID)
{
    uint win = dispatchIndex(tid);
    if (win >= min(ALOAD(gCounters[0].candidateCount), HOST_S(2))) return;
    CandidatePair pair = gCandidates[win];
    if (!ccdOwnsPair(pair)) return;
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    bool fastIsA = ccdFastIsA(ba, ca.body, bb, cb.body);
    Collider cf = fastIsA ? ca : cb;
    Collider co = fastIsA ? cb : ca;
    BodyState bf = fastIsA ? ba : bb;
    BodyState bo = fastIsA ? bb : ba;
    float3 r = ccdRelativeMotion(bf, bo);
    float t;
    float3 n;
    // The narrowphase left this pair to us (ccd_mark): it must get a contact.
    // The cast that found it (ccd_toi) ran in another kernel; a touching pair
    // at TOI 0 is borderline in f32 and can come out differently here, so a
    // miss falls back to the TOI ccd_toi stored and a stopper normal along the
    // relative motion (the pair would otherwise have no contact at all).
    if (dot(r, r) < 1e-20 || ccdCast(cf, bf, co, bo, r, t, n) != CAST_HIT)
    {
        t = clamp(asfloat((pair._pad & ~CCD_WIN) - 1u), 0.0, 1.0);
        n = dot(r, r) > 1e-20 ? normalize(r) : float3(0, -1, 0);
    }
    // TOI pose: the fast body moved by t·r relative to the other (at x⁻).
    BodyState bfT = bf;
    bfT.position += r * t;
    float3 pf; float4 qf; colliderPose(cf, bfT, pf, qf);
    float tol = max(2.0 * PARAMS.contactOffset, 0.005);
    float3 fp[8];
    uint nf = 0u;
    supportFeature(cf, pf, qf, n, tol, fp, nf);
    BoxManifold mf;
    mf.normal = n; // fast → other
    mf.count = 0;
    if (co.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(co.params.xyz);
        for (uint q = 0u; q < nf; q++)
        {
            mf.pts[mf.count].pA = fp[q];
            mf.pts[mf.count].pB = fp[q] - pn * (dot(pn, fp[q]) - co.params.w);
            mf.pts[mf.count].feature = 0x0CCD0000u | q;
            mf.count++;
        }
    }
    else if (co.shape == SHAPE_HEIGHTFIELD)
    {
        // The ordinary heightfield manifold at the TOI pose (fast = A), tagged
        // as time-of-impact rows so they never warm-start the next step.
        BodyState boT = bo;
        mf = heightFieldPairManifold(cf, bfT, co, boT, tol);
        for (int q = 0; q < mf.count; q++) mf.pts[q].feature = 0x0CCD0000u | uint(q);
        if (mf.count == 0)
        {
            mf.normal = n;
            mf.pts[0].pA = shapeSupport(cf, pf, qf, n);
            mf.pts[0].pB = mf.pts[0].pA;
            mf.pts[0].feature = 0x0CCF0000u;
            mf.count = 1;
        }
    }
    else
    {
        float3 po; float4 qo; colliderPose(co, bo, po, qo);
        float3 sO = shapeSupport(co, po, qo, -n);
        // The fast body's support points that land on the other's support
        // plane inside its bounds; else the other's points on the fast body's;
        // else the single touching pair of support points.
        for (uint q = 0u; q < nf; q++)
        {
            float3 pB = fp[q] + n * dot(n, sO - fp[q]);
            if (pointNearCollider(pB, co, po, qo, tol))
            {
                mf.pts[mf.count].pA = fp[q];
                mf.pts[mf.count].pB = pB;
                mf.pts[mf.count].feature = 0x0CCD0000u | q;
                mf.count++;
            }
        }
        if (mf.count == 0)
        {
            float3 op[8];
            uint no = 0u;
            supportFeature(co, po, qo, -n, tol, op, no);
            float3 sF = shapeSupport(cf, pf, qf, n);
            for (uint q = 0u; q < no; q++)
            {
                float3 pA = op[q] - n * dot(n, op[q] - sF);
                if (pointNearCollider(pA, cf, pf, qf, tol) && mf.count < 8)
                {
                    mf.pts[mf.count].pA = pA;
                    mf.pts[mf.count].pB = op[q];
                    mf.pts[mf.count].feature = 0x0CCE0000u | q;
                    mf.count++;
                }
            }
        }
        if (mf.count == 0)
        {
            mf.pts[0].pA = shapeSupport(cf, pf, qf, n);
            mf.pts[0].pB = sO;
            mf.pts[0].feature = 0x0CCF0000u;
            mf.count = 1;
        }
    }
    // Diagnostics (ball traces): the TOI normal, fast → other, in the slot's
    // carry words 29-31 (nothing in the step reads them).
    uint slotF = ccdSlotOf(fastIsA ? ca.body : cb.body, bf);
    if (slotF != CCD_NO_PAIR) ccdStoreVec(CCD_CARRY_BASE + CCD_CARRY_WORDS * slotF + 29u, n);
    // Pair order: rows run A → B.
    if (!fastIsA) mf = flipManifold(mf);
    emitToiManifold(pair, ba, bb, ca, cb, mf, fastIsA ? r * t : float3(0, 0, 0), fastIsA ? float3(0, 0, 0) : r * t);
}

// ---------------------------------------------------------------------------
// GPU impulses (row 4): [body, mode, J.xyz, point.xyz] records. Mode 0 adds an
// impulse at a world point (dynamic bodies); mode 1 sets a kinematic body's
// velocity (J = linear, point = angular). One thread per record; the first
// record of each body sums its run in record order (deterministic, no float
// atomics) and wakes the body.
// ---------------------------------------------------------------------------
static const uint IMPULSE_AT_POINT = 0u;
static const uint IMPULSE_SET_VELOCITY = 1u;
static const uint IMPULSE_LINEAR_ANGULAR = 2u; // J through the centre, angular impulse in `point`

// Mode word: mode (bits 0..7) | key << 8. A body's records are summed in
// ascending key order (host records: their queue index; projectile hits:
// after every host record, by projectile then hit), whatever order the
// atomic appends left them in. PASS_U 0 applies [0, count) at the start of
// the step (host records); PASS_U 1, after projectile_step, applies the
// records past the host's (IMPULSE_FIRST).
uint impulseKey(uint r) { return ALOAD(gScratchAtomics[r + 1u]) >> 8; }

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_impulses(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    uint count = ALOAD(gScratchAtomics[IMPULSE_COUNT]);
    if (count > IMPULSE_CAP)
    {
        if (k == 0u) AOR2(gCounters[0].overflowFlags, OVERFLOW_IMPULSES);
        count = IMPULSE_CAP;
    }
    uint first = PASS_U == 1u ? min(ALOAD(gScratchAtomics[IMPULSE_FIRST]), count) : 0u;
    k += first;
    if (k >= count) return;
    uint rk = IMPULSE_BASE + IMPULSE_WORDS * k;
    uint body = ALOAD(gScratchAtomics[rk]);
    uint myKey = impulseKey(rk);
    // Only the body's lowest-key record sums the run.
    for (uint q = first; q < count; q++)
    {
        uint rq = IMPULSE_BASE + IMPULSE_WORDS * q;
        if (q != k && ALOAD(gScratchAtomics[rq]) == body && impulseKey(rq) < myKey) return;
    }
    if (body >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[body];
    float3 dv = float3(0, 0, 0);
    float3 dw = float3(0, 0, 0);
    bool setVel = false;
    float3 setLin = b.linearVel, setAng = b.angularVel;
    uint key = myKey;
    uint at = k;
    for (uint n = first; n < count; n++)
    {
        uint r = IMPULSE_BASE + IMPULSE_WORDS * at;
        uint mode = ALOAD(gScratchAtomics[r + 1u]) & 255u;
        float3 J = asfloat(uint3(ALOAD(gScratchAtomics[r + 2u]), ALOAD(gScratchAtomics[r + 3u]), ALOAD(gScratchAtomics[r + 4u])));
        float3 P = asfloat(uint3(ALOAD(gScratchAtomics[r + 5u]), ALOAD(gScratchAtomics[r + 6u]), ALOAD(gScratchAtomics[r + 7u])));
        if (mode == IMPULSE_SET_VELOCITY)
        {
            setVel = true;
            setLin = J;
            setAng = P;
        }
        else if (bodyIsDynamic(b) && mode == IMPULSE_LINEAR_ANGULAR)
        {
            dv += J * b.invMass;
            float3 local = quatRotate(quatConjugate(b.orientation), P);
            dw += quatRotate(b.orientation, local * b.invInertia);
        }
        else if (bodyIsDynamic(b))
        {
            dv += J * b.invMass;
            float3 torque = cross(P - b.position, J);
            float3 local = quatRotate(quatConjugate(b.orientation), torque);
            dw += quatRotate(b.orientation, local * b.invInertia);
        }
        // Next record of this body: the smallest key above the current one.
        uint nextKey = 0xFFFFFFFFu;
        uint next = count;
        for (uint q = first; q < count; q++)
        {
            uint rq = IMPULSE_BASE + IMPULSE_WORDS * q;
            if (ALOAD(gScratchAtomics[rq]) != body) continue;
            uint kq = impulseKey(rq);
            if (kq > key && kq < nextKey) { nextKey = kq; next = q; }
        }
        if (next == count) break;
        key = nextKey;
        at = next;
    }
    if (setVel && !bodyIsDynamic(b))
    {
        gBodies[body].linearVel = setLin;
        gBodies[body].angularVel = setAng;
        return;
    }
    if (setVel)
    {
        dv += setLin - b.linearVel;
        dw += setAng - b.angularVel;
    }
    if (!bodyIsDynamic(b)) return;
    gBodies[body].linearVel = b.linearVel + dv;
    gBodies[body].angularVel = b.angularVel + dw;
    gBodies[body].flags = (b.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
    gBodies[body].sleepTimer = 0.0;
}
