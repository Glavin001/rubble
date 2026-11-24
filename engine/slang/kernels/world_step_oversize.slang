// ---- Oversize swept boxes on the uniform grid ----
// The 27-cell stencil finds a pair only if both half extents add up to at most
// a cell, i.e. each box is at most half a cell (the cell is twice the largest
// small radius). Look-ahead sweeps and spin pads grow a box past that without
// making it fast. When the fast list is scheduled (look-ahead scenes), a box
// past half a cell (with the pad's tolerance) is oversize: it owns its pairs
// with awake neighbours of smaller (or equal, then by index) size, and walks
// every cell its box ± its own half extent can reach, so nothing is missed.
// Other scenes keep the 27-cell query exactly.
float boxHalfMax(Aabb a)
{
    float3 h = 0.5 * (a.mx - a.mn);
    return max(h.x, max(h.y, h.z));
}

bool boxOversize(Aabb a, float cellSize)
{
    return fastListScheduled() && boxHalfMax(a) > 0.5 * cellSize + 1.0e-3;
}

// Which of two awake colliders emits their pair: the lower index between two
// standard boxes (the 27-cell rule), else the larger box (ties: lower index).
bool ownsAwakePair(uint i, Aabb ai, uint j, Aabb aj, float cellSize)
{
    bool oi = boxOversize(ai, cellSize), oj = boxOversize(aj, cellSize);
    if (!oi && !oj) return i < j;
    float hi = boxHalfMax(ai), hj = boxHalfMax(aj);
    return hi > hj || (hi == hj && i < j);
}

static const int OVERSIZE_MAX_CELLS = 4096;

void oversizeQuery(uint i, Collider col, BodyState b, Aabb box, float3 cA, float rA,
    bool probeIsBox, BoxObb probeObb, float faceGap, float cellSize)
{
    uint base = HOST_S(1);
    float h = boxHalfMax(box);
    int3 lo = int3(floor((box.mn - h) / cellSize));
    int3 hi = int3(floor((box.mx + h) / cellSize));
    int3 span = hi - lo + int3(1, 1, 1);
    if (span.x * span.y * span.z > OVERSIZE_MAX_CELLS)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_FAST_LIST);
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
            bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
            Aabb ab = gAabbs[base + cur];
            link = gridSplit() ? asuint(ab._pad1) : gColorLists[cur];
            guard++;
            if (guard == 4096u) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
            if (cur == i) continue;
            if (any(int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize)) != int3(x, y, z))) continue;
            float rB = ab._pad0;
            if (rB > 0.0)
            {
                float3 d = cA - 0.5 * (ab.mn + ab.mx);
                float rr = rA + rB;
                if (dot(d, d) > rr * rr) continue;
            }
            if (!aabbOverlap(box, ab)) continue;
            if (awakeEntry && !ownsAwakePair(i, box, cur, ab, cellSize)) continue;
            Collider other = gColliders[cur];
            if (other.body == col.body || other.shape == SHAPE_PLANE || bodiesJointed(col.body, other.body)) continue;
            if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
            // Statics, kinematics and sleepers never query: this querier owns them.
            emitCandidate(i, cur);
        }
    }
}

// One cell of a split table. In the awake table every visited pair belongs to
// this querier (forward cell) or to the higher index (own cell); the other
// table keeps the legacy sleeper/static rule. Geometry tests match the full scan.
void gridScanCell(uint i, Collider col, BodyState b, Aabb box, float3 cA, float rA,
    bool probeIsBox, BoxObb probeObb, float faceGap, float cellSize, int3 cell,
    uint tableBase, bool ownCell)
{
    uint base = HOST_S(1);
    bool sleeper = bodyIsSleeping(b);
    uint link = ALOAD(gConstraintIds[tableBase + gridHash(cell.x, cell.y, cell.z)]);
    uint guard = 0;
    while (link != 0xFFFFFFFFu && guard < 4096u)
    {
        uint cur = link & GRID_COLLIDER_MASK;
        bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
        Aabb ab = gAabbs[base + cur];
        link = asuint(ab._pad1);
        guard++;
        if (guard == 4096u)
            AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        if (cur == i || (ownCell && awakeEntry && cur <= i)) continue;
        int3 candidateCell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
        if (any(candidateCell != cell)) continue;
        float rB = ab._pad0;
        if (rB > 0.0)
        {
            float3 d = cA - 0.5 * (ab.mn + ab.mx);
            float rr = rA + rB;
            if (dot(d, d) > rr * rr) continue;
        }
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
            box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
            box.mx.z < ab.mn.z || ab.mx.z < box.mn.z)
            continue;
        // An oversize awake neighbour owns the pair (oversizeQuery).
        if (awakeEntry && boxOversize(ab, cellSize)) continue;
        Collider other = gColliders[cur];
        bool keep;
        if (awakeEntry)
            keep = true;
        else
        {
            BodyState ob = gBodies[other.body];
            bool otherSleep = bodyIsDynamic(ob) && bodyIsSleeping(ob);
            // Statics and kinematics never query: owned at any index.
            keep = cur > i || (!sleeper && (otherSleep || !bodyIsDynamic(ob)));
        }
        if (!keep || other.body == col.body || bodiesJointed(col.body, other.body) ||
            other.shape == SHAPE_PLANE)
            continue;
        if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
        // Awake pairs keep the (lower, higher) order the full scan emitted.
        if (awakeEntry) emitCandidate(min(i, cur), max(i, cur));
        else emitCandidate(i, cur);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    if (HOST_S(18) != 0u)
    {
        hierQuery(i);
        return;
    }
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (col.shape == SHAPE_PLANE) return;
    // Fast colliders query along their sweep (broadphase_fast).
    if (colliderIsFast(i, col, b)) return;
    if (!bodyIsDynamic(b))
    {
        // A moving kinematic finds the sleepers it may wake; awake dynamics
        // find it like a static.
        if (kinematicMoving(b)) kinematicQuery(i, col);
        return;
    }
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    bool sleeper = bodyIsSleeping(b);
    // Probe OBB from build_aabbs cache (three-avbd findPairs loads querier once).
    bool probeIsBox = shapeHasHull(col.shape);
    BoxObb probeObb = probeIsBox ? worldObbFromCache(i) : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeIsBox ? cachedReach(i) : 0.0);

    // Small scenes: exact pairwise (correctness). Large: uniform-grid hash.
    // Awake dynamics test everyone; sleepers only keep static contacts (ground) so a
    // later impact can wake the island without the floor dropping out for a frame.
    if (gCounters[0].colliderCount <= 512u)
    {
        // while + nested tests, not for/continue: SPIR-V→naga turns `continue` into a
        // switch-break that never exits the loop (GPU hang on WebGPU).
        uint j = 0u;
        while (j < gCounters[0].colliderCount)
        {
            // Bounds first: a non-plane collider whose box misses this one's is
            // never kept, so skip it before loading its collider and body records
            // (most pairs; the same pairs, in the same order, as testing it last).
            Aabb pre = gAabbs[base + j];
            bool apart = box.mx.x < pre.mn.x || pre.mx.x < box.mn.x || box.mx.y < pre.mn.y || pre.mx.y < box.mn.y
                || box.mx.z < pre.mn.z || pre.mx.z < box.mn.z;
            if (j != i && !(apart && gColliders[j].shape != SHAPE_PLANE))
            {
                Collider other = gColliders[j];
                if (other.body != col.body)
                {
                    BodyState ob = gBodies[other.body];
                    bool otherStatic = !bodyIsDynamic(ob);
                    bool keep = true;
                    if (sleeper && (!otherStatic || carryRestPairs())) keep = false;
                    // A moving kinematic owns its pairs with sleepers (kinematicQuery).
                    if (sleeper && kinematicMoving(ob)) keep = false;
                    if (bodyIsSleeping(ob) && sleeper) keep = false;
                    // Lower-index dynamics own the pair, unless asleep: sleepers
                    // never query, so an awake body must emit its sleeping
                    // neighbours at any index (as the grid path does).
                    if (!otherStatic && j < i && !(bodyIsSleeping(ob) && !sleeper)) keep = false;
                    if (otherStatic && !bodyIsDynamic(b)) keep = false;
                    if (bodiesJointed(col.body, other.body)) keep = false;
                    if (keep && other.shape != SHAPE_PLANE)
                    {
                        Aabb ab = gAabbs[base + j];
                        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) keep = false;
                        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) keep = false;
                        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) keep = false;
                    }
                    if (keep && probeIsBox && boxesSeparatedOnFaceAxis(probeObb, j, faceGap)) keep = false;
                    if (keep) emitCandidate(i, j);
                }
            }
            j++;
        }
        return;
    }

    float cellSize = max(PARAMS.gridCellSize, 0.5);
    // Cell is 2·maxSmallRadius, so the centre cell ±1 (27) covers every small overlap.
    float3 center = 0.5 * (box.mn + box.mx);
    int3 cc = int3(floor(center / cellSize));

    // Sleepers emit nothing: frozen pose needs no ground manifold, and 32k sleeper–
    // plane candidates dominated settled-pile ms. Awake neighbours still insert into
    // the grid and wake sleepers on impact; a woken body re-acquires the plane next step.
    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    bool queryLarge = rA > cellSize;
    // The whole swept bound (_pad0), not the capped reach: the box is centred
    // mid-sweep and spans all of it.
    rA = max(rA + colliderReach(b, col), box._pad0);
    uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k]);
        if (j == i) continue;
        Collider other = gColliders[j];
        if (other.body == col.body || bodiesJointed(col.body, other.body)) continue;
        BodyState ob = gBodies[other.body];
        // Small→large owns this pair regardless of indices. Only large→large
        // needs index ordering; a large centre-cell query cannot cover its surface.
        if (queryLarge && bodyIsDynamic(ob) && j < i) continue;
        if (sleeper && carryRestPairs() && bodyIsSleeping(ob)) continue; // carried
        Aabb ab = gAabbs[base + j];
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        emitCandidate(i, j);
    }
    if (sleeper)
        return;
    uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
    for (uint pi = 0u; pi < nPlanes; pi++)
        emitCandidate(i, ALOAD(gScratchAtomics[43u + pi]));
    float3 cA = 0.5 * (box.mn + box.mx);
    if (!queryLarge && boxOversize(box, cellSize))
    {
        oversizeQuery(i, col, b, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize);
    }
    else if (!queryLarge && gridSplit())
    {
        // Awake table: own cell (higher index owns) + 13 forward cells (all).
        for (uint c = 0u; c < 14u; c++)
        {
            int3 off = gridForwardOffset(c);
            gridScanCell(i, col, b, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize,
                cc + off, 0u, c == 0u);
        }
        // Other table (sleepers, small statics): full stencil, skipped when empty.
        if (ALOAD(gScratchAtomics[52]) != 0u)
        {
            for (int z = cc.z - 1; z <= cc.z + 1; z++)
            for (int y = cc.y - 1; y <= cc.y + 1; y++)
            for (int x = cc.x - 1; x <= cc.x + 1; x++)
                gridScanCell(i, col, b, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize,
                    int3(x, y, z), gridHashSize(), false);
        }
    }
    else if (!queryLarge)
    {
    for (int z = cc.z - 1; z <= cc.z + 1; z++)
    for (int y = cc.y - 1; y <= cc.y + 1; y++)
    for (int x = cc.x - 1; x <= cc.x + 1; x++)
    {
        uint h = gridHash(x, y, z);
        uint link = ALOAD(gConstraintIds[h]);
        // Dense pile hash slots can exceed a few hundred links (32k in ~2 m cells).
        // Truncating at 256 silently dropped pairs → deep C0. Cap high; flag if hit.
        uint guard = 0;
        while (link != 0xFFFFFFFFu && guard < 4096u)
        {
            uint cur = link & GRID_COLLIDER_MASK;
            bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
            bool ownedElsewhere = awakeEntry && cur <= i;
            // Fetch the next dependent link before this candidate's geometry.
            // Lower-index awake dynamics own the reverse query; sleepers do not.
            link = gColorLists[cur];
            guard++;
            if (guard == 4096u)
                AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
            if (ownedElsewhere) continue;
            // Sphere test from the cached AABB (center + radius in _pad0) before
            // touching the collider or body. Same geometric radius as the old test.
            Aabb ab = gAabbs[base + cur];
            // Hash buckets are not cells: two of the 27 queried cells can alias.
            // Visiting only the candidate's true cell prevents duplicate constraints.
            int3 candidateCell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
            if (candidateCell.x != x || candidateCell.y != y || candidateCell.z != z)
            {
                continue;
            }
            float rB = ab._pad0;
            if (rB > 0.0)
            {
                float3 d = cA - 0.5 * (ab.mn + ab.mx);
                float rr = rA + rB;
                if (dot(d, d) > rr * rr)
                {
                    continue;
                }
            }
            if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                box.mx.z < ab.mn.z || ab.mx.z < box.mn.z)
            {
                continue;
            }
            // An oversize awake neighbour owns the pair (oversizeQuery).
            if (awakeEntry && boxOversize(ab, cellSize)) continue;
            // Dynamic-dynamic once (higher index), PLUS awake→sleeping at any index:
            // smash balls are last, so cur>i alone never pairs them with sleeping
            // bricks (sleepers only keep static contacts and would not emit).
            Collider other = gColliders[cur];
            BodyState ob = gBodies[other.body];
            bool otherSleep = bodyIsDynamic(ob) && bodyIsSleeping(ob);
            // Statics and kinematics never query the grid for awake bodies, so
            // the awake querier owns those pairs at any index (a lower-index
            // small static used to be dropped: bricks in a wall, debris shelves).
            bool pairOnce = cur > i || (!sleeper && (otherSleep || !bodyIsDynamic(ob)));
            if (pairOnce && cur != i)
            {
                if (other.body != col.body &&
                    !bodiesJointed(col.body, other.body) &&
                    other.shape != SHAPE_PLANE)
                {
                    if (!(probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)))
                        emitCandidate(i, cur);
                }
            }
        }
    }
    }
    if (queryLarge)
    {
        // Large dynamics (vehicle chassis, smash balls) do not use the 27-cell
        // stencil. Small dynamics and sleepers find them through the large list;
        // small statics never query, so walk the grid cells under this AABB for them.
        int3 lo = int3(floor(box.mn / cellSize));
        int3 hi = int3(floor(box.mx / cellSize));
        int3 span = hi - lo + int3(1, 1, 1);
        if (span.x * span.y * span.z <= 512)
        {
            uint tableBase = gridSplit() ? gridHashSize() : 0u;
            for (int z = lo.z; z <= hi.z; z++)
            for (int y = lo.y; y <= hi.y; y++)
            for (int x = lo.x; x <= hi.x; x++)
            {
                uint link = ALOAD(gConstraintIds[tableBase + gridHash(x, y, z)]);
                uint guard = 0u;
                while (link != 0xFFFFFFFFu && guard < 4096u)
                {
                    uint cur = link & GRID_COLLIDER_MASK;
                    Aabb ab = gAabbs[base + cur];
                    link = gridSplit() ? asuint(ab._pad1) : gColorLists[cur];
                    guard++;
                    int3 cell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
                    if (cell.x != x || cell.y != y || cell.z != z) continue;
                    if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                        box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                        box.mx.z < ab.mn.z || ab.mx.z < box.mn.z)
                        continue;
                    Collider other = gColliders[cur];
                    if (other.body == col.body || other.shape == SHAPE_PLANE) continue;
                    if (bodyIsDynamic(gBodies[other.body])) continue;
                    if (bodiesJointed(col.body, other.body)) continue;
                    if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
                    emitCandidate(i, cur);
                }
            }
        }
    }
    // Oversized statics are not in the grid. broadphase_insert lists up to
    // LARGE_STATIC_CAP of them; on overflow the legacy endpoint probe (ground at
    // collider 0 or last) applies. Large dynamics were handled exactly once above.
    uint nLargeStatic = ALOAD(gScratchAtomics[LARGE_STATIC_COUNT]);
    uint nListed = largeStaticCount();
    bool listOverflow = nLargeStatic > LARGE_STATIC_CAP;
    uint nProbe = nListed + (listOverflow ? 2u : 0u);
    for (uint t = 0u; t < nProbe; t++)
    {
        if (gCounters[0].colliderCount == 0u) break;
        bool endpoint = t >= nListed;
        uint j = !endpoint ? largeStaticCollider(t)
            : ((t == nListed) ? 0u : (gCounters[0].colliderCount - 1u));
        if (j == i || (endpoint && t > nListed && j == 0u)) continue;
        if (endpoint)
        {
            bool listed = false;
            for (uint u = 0u; u < nListed; u++)
                if (largeStaticCollider(u) == j) listed = true;
            if (listed) continue;
        }
        Collider other = gColliders[j];
        if (other.shape == SHAPE_PLANE || other.body == col.body) continue;
        Aabb ab = gAabbs[base + j];
        float3 ext = ab.mx - ab.mn;
        if (endpoint && ext.x < cellSize * 4.0 && ext.y < cellSize * 4.0 && ext.z < cellSize * 4.0)
            continue;
        // Dynamic-static always from the mover.
        BodyState ob = gBodies[other.body];
        if (bodyIsDynamic(ob)) continue;
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        if (probeIsBox && shapeHasHull(other.shape) && boxesSeparatedOnFaceAxis(probeObb, j, faceGap))
            continue;
        // Sphere large (smash ball): radius test.
        if (other.shape == SHAPE_SPHERE)
        {
            float3 d = b.position - ob.position;
            float rr = other.params.x + (col.params.w > 0.0 ? col.params.w : length(col.params.xyz))
                + colliderReach(b, col) + colliderReach(ob, other);
            if (dot(d, d) > rr * rr) continue;
        }
        emitCandidate(i, j);
    }
}

