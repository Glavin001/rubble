// ---- Uniform rest tables (World, uniform split grid; HOST_S(25) bits 0 and 4) ----
// The split grid's other table (sleepers and small statics) kept across steps as R
// (restHeads, a link per collider at restLinks), and its awake table filled each step
// as W (awakeHeads, links in the AABB spare word) from the moving colliders only
// (list_moving_colliders), which alone query (broadphase_query_awake_uniform). R is
// rebuilt as the hierarchical one is (write_moved_args), and between rebuilds holds
// every collider at rest at its current cell: no collider stops moving between them.
// A collider in W this step (its stamp is this step's index, HOST_S(17)) is a
// tombstone in R: every R walk skips it, and W has it, as the full pass's awake table
// had it and its other table did not. The stamp, not the body's flags, decides, so a
// body woken after the inserts (a projectile's late impulse) is still found in R by
// the walks that run later in the step, as the full pass's other table held it.
// Sleepers' pairs with the large-dynamic list come from broadphase_sleeper_large, a
// thread per collider dispatched only while that list is non-empty. Planes, oversized
// statics and resting large dynamics are listed in the header (oversized statics in
// the scratch list the full pass keeps) and published into the step's scratch lists.
// The pairs are the full pass's; only their append order differs. WGSL hosts only:
// Metal's HOST_S(25) is a scratch word.
#if defined(ENGINE_WGSL)
bool restUniform() { return (HOST_S(25) & 17u) == 17u; }
#else
bool restUniform() { return false; }
#endif
// Uniform R header words (restBase): 0 entries in R chains, 1 planes, 2..9 their
// colliders, 10 resting large dynamics, 11..26 their colliders, 28 oversized statics
// (their colliders at scratch LARGE_STATIC_BASE, as the full pass lists them).
static const uint REST_U_LARGE_MAX = 16u;
static const uint REST_U_STATICS = 28u;
// Header word 29: the largest AABB half extent of the entries pushed onto R's chains
// (asuint; half extents are not negative), so a walk can bound where they lie.
static const uint REST_U_HALF = 29u;
// Per-collider words past the moving-collider list: this step's W stamp, and the R
// chain a collider sits in (its bucket, REST_AT_LISTED in a header list, REST_AT_NONE).
static const uint REST_AT_NONE = 0xFFFFFFFFu;
static const uint REST_AT_LISTED = 0xFFFFFFFEu;
uint restStampBase() { return restLinks() + 2u * gCounters[0].colliderCount; }
uint restAtBase() { return restLinks() + 3u * gCounters[0].colliderCount; }
bool restTombstone(uint c) { return ALOAD(gConstraintIds[restStampBase() + c]) == HOST_S(17); }

// Uniform-grid cell of an AABB's centre (broadphase_insert's).
int3 uniformCell(Aabb box, float cellSize) { return int3(floor((0.5 * (box.mn + box.mx)) / cellSize)); }

// Rebuild 2 (uniform): every collider at rest into R by broadphase_insert's rules for
// colliders not awake (sleepers, statics; kinematics never take this path).
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_insert_uniform(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    ASTORE(gConstraintIds[restStampBase() + i], REST_AT_NONE);
    ASTORE(gConstraintIds[restAtBase() + i], REST_AT_NONE);
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (bodyMoving(b)) return;
    restInsertUniform(i, col, b);
}

void restInsertUniform(uint i, Collider col, BodyState b)
{
    uint hdr = restBase();
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gConstraintIds[hdr + 1u], 1u, slot);
        // Past 8: rest_publish_uniform fails each step with OVERFLOW_BP_CHAIN.
        if (slot < 8u) ASTORE(gConstraintIds[hdr + 2u + slot], i);
        ASTORE(gConstraintIds[restAtBase() + i], REST_AT_LISTED);
        return;
    }
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    Aabb box = gAabbs[HOST_S(1) + i];
    float3 ext = box.mx - box.mn;
    float boundR = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    if (!bodyIsDynamic(b) && boundR > 0.5 * cellSize)
    {
        uint slot;
        AADD3(gConstraintIds[hdr + REST_U_STATICS], 1u, slot);
        if (slot < LARGE_STATIC_CAP)
        {
            ASTORE(gScratchAtomics[LARGE_STATIC_BASE + slot], i);
            ASTORE(gConstraintIds[restAtBase() + i], REST_AT_LISTED);
            return;
        }
    }
    if (!bodyIsDynamic(b) &&
        (ext.x >= cellSize * 4.0 || ext.y >= cellSize * 4.0 || ext.z >= cellSize * 4.0))
        return;
    if (bodyIsDynamic(b) && boundR > cellSize)
    {
        uint slot;
        AADD3(gConstraintIds[hdr + 10u], 1u, slot);
        if (slot < REST_U_LARGE_MAX) ASTORE(gConstraintIds[hdr + 11u + slot], i);
        ASTORE(gConstraintIds[restAtBase() + i], REST_AT_LISTED);
        return;
    }
    int3 cc = uniformCell(box, cellSize);
    uint h = gridHash(cc.x, cc.y, cc.z);
    uint prev;
    AXCHG3(gConstraintIds[restHeads() + h], i, prev);
    ASTORE(gConstraintIds[restLinks() + i], prev);
    ASTORE(gConstraintIds[restAtBase() + i], h);
    AADD2(gConstraintIds[hdr], 1u);
    AMAX2(gConstraintIds[hdr + REST_U_HALF], asuint(boxHalfMax(box)));
}

// Each step, before the awake inserts: R's lists into the scratch words the full pass
// fills (planes 42.., large dynamics 25 / 64.., oversized statics 80, the other
// table's occupancy 52), leaving out colliders that move this step (W lists them).
// Overflows are flagged every step, as the full pass flags them.
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_publish_uniform(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint hdr = restBase();
    ASTORE(gScratchAtomics[CCD_BODY_COUNT], 0u);
    ASTORE(gScratchAtomics[FAST_COUNT], 0u);
    ASTORE(gScratchAtomics[CCD_FALLBACKS], 0u);
    ASTORE(gScratchAtomics[CCD_PAIR_COUNT], 0u);
    ASTORE(gScratchAtomics[CCD_MAX_REACH], 0u);
    ASTORE(gScratchAtomics[52], ALOAD(gConstraintIds[hdr]));
    uint planes = ALOAD(gConstraintIds[hdr + 1u]), np = 0u;
    for (uint k = 0u; k < min(planes, 8u); k++)
    {
        uint c = ALOAD(gConstraintIds[hdr + 2u + k]);
        if (bodyMoving(gBodies[gColliders[c].body])) continue;
        ASTORE(gScratchAtomics[43u + np], c);
        np++;
    }
    ASTORE(gScratchAtomics[42], planes > 8u ? planes : np);
    if (planes > 8u) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
    uint large = ALOAD(gConstraintIds[hdr + 10u]), nl = 0u;
    for (uint k = 0u; k < min(large, REST_U_LARGE_MAX); k++)
    {
        uint c = ALOAD(gConstraintIds[hdr + 11u + k]);
        if (bodyMoving(gBodies[gColliders[c].body])) continue;
        ASTORE(gScratchAtomics[LEGACY_LARGE_BASE + nl], c);
        nl++;
    }
    if (large > REST_U_LARGE_MAX)
    {
        nl = large;
        AOR2(gCounters[0].overflowFlags, OVERFLOW_LARGE_DYNAMICS);
    }
    ASTORE(gScratchAtomics[25], nl);
    uint statics = ALOAD(gConstraintIds[hdr + REST_U_STATICS]);
    ASTORE(gScratchAtomics[LARGE_STATIC_COUNT], statics);
    if (statics > LARGE_STATIC_CAP) AOR2(gCounters[0].overflowFlags, OVERFLOW_LARGE_STATICS);
}

// Listed moving colliders into W (one thread each, slot 81), by broadphase_insert's
// rules for awake dynamics; each is stamped (a tombstone in R this step).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert_awake_uniform(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    ASTORE(gConstraintIds[restStampBase() + c], HOST_S(17));
    Collider col = gColliders[c];
    BodyState b = gBodies[col.body];
    if (colliderIsFast(c, col, b))
    {
        fastListAppend(c);
        return;
    }
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gScratchAtomics[42], 1, slot);
        if (slot < 8u) ASTORE(gScratchAtomics[43u + slot], c);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    uint base = HOST_S(1);
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    Aabb box = gAabbs[base + c];
    float radius = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    if (radius > cellSize)
    {
        uint slot;
        AADD3(gScratchAtomics[25], 1, slot);
        if (slot < 16u) ASTORE(gScratchAtomics[LEGACY_LARGE_BASE + slot], c);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_LARGE_DYNAMICS);
        return;
    }
    int3 cc = uniformCell(box, cellSize);
    uint prev;
    AXCHG3(gConstraintIds[awakeHeads() + gridHash(cc.x, cc.y, cc.z)], c | GRID_AWAKE_DYNAMIC, prev);
    gAabbs[base + c]._pad1 = asfloat(prev);
}

// One cell of W (awake entries) or R (resting entries, tombstones skipped) as an awake
// querier sees it: gridScanCell's tests on the same values.
void restScanCellUniform(uint i, Collider col, Aabb box, float3 cA, float rA,
    bool probeIsBox, BoxObb probeObb, float faceGap, float cellSize, int3 cell,
    bool rest, bool ownCell)
{
    uint base = HOST_S(1);
    uint link = ALOAD(gConstraintIds[(rest ? restHeads() : awakeHeads()) + gridHash(cell.x, cell.y, cell.z)]);
    uint guard = 0;
    while (link != 0xFFFFFFFFu && guard < 4096u)
    {
        uint cur = link & GRID_COLLIDER_MASK;
        bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
        Aabb ab = gAabbs[base + cur];
        link = rest ? ALOAD(gConstraintIds[restLinks() + cur]) : asuint(ab._pad1);
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
        // Tombstones last: the test only filters, and most entries fail the cheaper ones.
        if (rest && restTombstone(cur)) continue;
        if (awakeEntry && boxOversize(ab, cellSize)) continue;
        Collider other = gColliders[cur];
        bool keep;
        if (awakeEntry)
            keep = true;
        else
        {
            BodyState ob = gBodies[other.body];
            bool otherSleep = bodyIsDynamic(ob) && bodyIsSleeping(ob);
            keep = cur > i || otherSleep || !bodyIsDynamic(ob);
        }
        if (!keep || other.body == col.body || bodiesJointed(col.body, other.body) ||
            other.shape == SHAPE_PLANE)
            continue;
        if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
        if (awakeEntry) emitCandidate(min(i, cur), max(i, cur));
        else emitCandidate(i, cur);
    }
}

// oversizeQuery over W and R (tombstones skipped).
void restOversizeQueryUniform(uint i, Collider col, Aabb box, float3 cA, float rA,
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
    for (int z = lo.z; z <= hi.z; z++)
    for (int y = lo.y; y <= hi.y; y++)
    for (int x = lo.x; x <= hi.x; x++)
    for (uint t = 0u; t < 2u; t++)
    {
        bool rest = t == 1u;
        uint link = ALOAD(gConstraintIds[(rest ? restHeads() : awakeHeads()) + gridHash(x, y, z)]);
        uint guard = 0u;
        while (link != 0xFFFFFFFFu && guard < 4096u)
        {
            uint cur = link & GRID_COLLIDER_MASK;
            bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
            Aabb ab = gAabbs[base + cur];
            link = rest ? ALOAD(gConstraintIds[restLinks() + cur]) : asuint(ab._pad1);
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
            if (rest && restTombstone(cur)) continue;
            if (awakeEntry && !ownsAwakePair(i, box, cur, ab, cellSize)) continue;
            Collider other = gColliders[cur];
            if (other.body == col.body || other.shape == SHAPE_PLANE || bodiesJointed(col.body, other.body)) continue;
            if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
            emitCandidate(i, cur);
        }
    }
}

// broadphase_query for a listed (awake dynamic) collider on the uniform split grid,
// past the small-world pairwise path (hosts take this path above 512 colliders): the
// awake table is W, the other table R.
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query_awake_uniform(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    restQueryUniform(awakeCollider(li));
}

void restQueryUniform(uint i)
{
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (col.shape == SHAPE_PLANE) return;
    if (colliderIsFast(i, col, b)) return;
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    bool probeIsBox = shapeHasHull(col.shape);
    BoxObb probeObb = probeIsBox ? worldObbFromCache(i) : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeIsBox ? cachedReach(i) : 0.0);
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    float3 center = 0.5 * (box.mn + box.mx);
    int3 cc = int3(floor(center / cellSize));
    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    bool queryLarge = rA > cellSize;
    rA = max(rA + colliderReach(b, col), box._pad0);
    uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k]);
        if (j == i) continue;
        Collider other = gColliders[j];
        if (other.body == col.body || bodiesJointed(col.body, other.body)) continue;
        BodyState ob = gBodies[other.body];
        if (queryLarge && bodyIsDynamic(ob) && j < i) continue;
        Aabb ab = gAabbs[base + j];
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        emitCandidate(i, j);
    }
    uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
    for (uint pi = 0u; pi < nPlanes; pi++)
        emitCandidate(i, ALOAD(gScratchAtomics[43u + pi]));
    float3 cA = 0.5 * (box.mn + box.mx);
    if (!queryLarge && boxOversize(box, cellSize))
    {
        restOversizeQueryUniform(i, col, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize);
    }
    else if (!queryLarge)
    {
        for (uint c = 0u; c < 14u; c++)
        {
            int3 off = gridForwardOffset(c);
            restScanCellUniform(i, col, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize,
                cc + off, false, c == 0u);
        }
        if (ALOAD(gScratchAtomics[52]) != 0u)
        {
            for (int z = cc.z - 1; z <= cc.z + 1; z++)
            for (int y = cc.y - 1; y <= cc.y + 1; y++)
            for (int x = cc.x - 1; x <= cc.x + 1; x++)
                restScanCellUniform(i, col, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize,
                    int3(x, y, z), true, false);
        }
    }
    if (queryLarge)
    {
        int3 lo = int3(floor(box.mn / cellSize));
        int3 hi = int3(floor(box.mx / cellSize));
        int3 span = hi - lo + int3(1, 1, 1);
        if (span.x * span.y * span.z <= 512)
        {
            for (int z = lo.z; z <= hi.z; z++)
            for (int y = lo.y; y <= hi.y; y++)
            for (int x = lo.x; x <= hi.x; x++)
            {
                uint link = ALOAD(gConstraintIds[restHeads() + gridHash(x, y, z)]);
                uint guard = 0u;
                while (link != 0xFFFFFFFFu && guard < 4096u)
                {
                    uint cur = link & GRID_COLLIDER_MASK;
                    Aabb ab = gAabbs[base + cur];
                    link = ALOAD(gConstraintIds[restLinks() + cur]);
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
        BodyState ob = gBodies[other.body];
        if (bodyIsDynamic(ob)) continue;
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        if (probeIsBox && shapeHasHull(other.shape) && boxesSeparatedOnFaceAxis(probeObb, j, faceGap))
            continue;
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

// The sleepers' pairs with the step's large dynamics (after the awake inserts appended
// to the list), found from the large ones: work slot 88, a workgroup per listed large
// dynamic walking the R cells that can hold a collider overlapping it
// (broadphase_sleeper_large_walk). Should a large box span more than
// SLEEPER_WALK_MAX_CELLS of them, slot 85 instead: broadphase_sleeper_large, a thread per
// collider (the full pass's sleeper threads). Both empty without a large dynamic.
static const uint SLEEPER_WALK_MAX_CELLS = 32768u;

int3 sleeperWalkSpan(Aabb lb, float cs, float h, out int3 lo)
{
    // Padded past the rounding of a centre and a half extent: a superset of cells.
    h = h * 1.001 + 1.0e-4;
    lo = int3(floor((lb.mn - h) / cs));
    int3 hi = int3(floor((lb.mx + h) / cs));
    return hi - lo + int3(1, 1, 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void write_sleeper_large_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint n = min(ALOAD(gScratchAtomics[25]), 16u);
    bool walk = ALOAD(gScratchAtomics[25]) <= 16u;
    float cs = max(PARAMS.gridCellSize, 0.5);
    float h = asfloat(ALOAD(gConstraintIds[restBase() + REST_U_HALF]));
    for (uint k = 0u; k < n; k++)
    {
        int3 lo;
        int3 span = sleeperWalkSpan(gAabbs[HOST_S(1) + ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k])], cs, h, lo);
        if (float(span.x) * float(span.y) * float(span.z) > float(SLEEPER_WALK_MAX_CELLS)) walk = false;
    }
    writeDispatchArgs(85u, n != 0u && !walk ? gCounters[0].colliderCount : 0u);
    // Slot 88: x = chunks of 64 cells (the most any listed large one has), y = listed.
    uint chunks = 0u;
    for (uint k = 0u; k < n && walk; k++)
    {
        int3 lo;
        int3 span = sleeperWalkSpan(gAabbs[HOST_S(1) + ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k])], cs, h, lo);
        chunks = max(chunks, (uint(span.x) * uint(span.y) * uint(span.z) + 63u) / 64u);
    }
    writeGroupGrid(88u, walk ? max(chunks, 1u) : 0u, walk && n != 0u ? n : 1u);
}

// broadphase_sleeper_large's test for sleeper collider i (small or large) and listed
// large dynamic j: the pair as the sleeper's thread of the full pass emits it.
void sleeperLargePair(uint i, uint j)
{
    if (j == i) return;
    uint base = HOST_S(1);
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (col.shape == SHAPE_PLANE || colliderIsFast(i, col, b)) return;
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    bool queryLarge = rA > cellSize;
    Collider other = gColliders[j];
    if (other.body == col.body || bodiesJointed(col.body, other.body)) return;
    BodyState ob = gBodies[other.body];
    if (queryLarge && bodyIsDynamic(ob) && j < i) return;
    if (carryRestPairs() && bodyIsSleeping(ob)) return; // carried
    Aabb box = gAabbs[base + i];
    Aabb ab = gAabbs[base + j];
    if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) return;
    if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) return;
    if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) return;
    emitCandidate(i, j);
}

// Workgroups (chunk, k) for each listed large dynamic L (k): a lane per R cell that can
// hold an entry overlapping L (an entry's centre lies in its cell, within its half
// extent, at most header word 29, of any point of its box), each entry taken at its
// true cell only; chunk 0 also takes the listed large dynamics (sleeping large ones are
// listed, not in the chains). Every sleeper a pair with L can come from is in one of
// them, so the pairs are the full pass's.
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_sleeper_large_walk(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint k = gid.y;
    if (k >= min(ALOAD(gScratchAtomics[25]), 16u)) return;
    uint j = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k]);
    uint base = HOST_S(1);
    Aabb lb = gAabbs[base + j];
    float cs = max(PARAMS.gridCellSize, 0.5);
    float h = asfloat(ALOAD(gConstraintIds[restBase() + REST_U_HALF]));
    int3 lo;
    int3 span = sleeperWalkSpan(lb, cs, h, lo);
    uint cells = uint(span.x) * uint(span.y) * uint(span.z);
    uint t = gid.x * 64u + lid.x;
    if (t < cells)
    {
        int3 cell = lo + int3(int(t % uint(span.x)), int((t / uint(span.x)) % uint(span.y)), int(t / uint(span.x * span.y)));
        uint link = ALOAD(gConstraintIds[restHeads() + gridHash(cell.x, cell.y, cell.z)]);
        uint guard = 0u;
        while (link != 0xFFFFFFFFu && guard < 4096u)
        {
            uint cur = link & GRID_COLLIDER_MASK;
            Aabb ab = gAabbs[base + cur];
            link = ALOAD(gConstraintIds[restLinks() + cur]);
            guard++;
            if (any(int3(floor((0.5 * (ab.mn + ab.mx)) / cs)) != cell)) continue;
            if (!aabbOverlap(ab, lb) || restTombstone(cur)) continue;
            sleeperLargePair(cur, j);
        }
        if (link != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
    }
    uint n = min(ALOAD(gScratchAtomics[25]), 16u);
    if (gid.x == 0u && lid.x < n)
        sleeperLargePair(ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + lid.x]), j);
}

// A sleeper's thread of the full broadphase_query: its pairs with the large dynamics.
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_sleeper_large(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (col.shape == SHAPE_PLANE || colliderIsFast(i, col, b)) return;
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    bool queryLarge = rA > cellSize;
    uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k]);
        if (j == i) continue;
        Collider other = gColliders[j];
        if (other.body == col.body || bodiesJointed(col.body, other.body)) continue;
        BodyState ob = gBodies[other.body];
        if (queryLarge && bodyIsDynamic(ob) && j < i) continue;
        if (carryRestPairs() && bodyIsSleeping(ob)) continue; // carried
        Aabb ab = gAabbs[base + j];
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        emitCandidate(i, j);
    }
}

// After every walker of this step's tables: empty the W heads the listed colliders
// may have filled (slot 81; a listed collider not in W clears a head that is empty).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_clear_awake_uniform(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    int3 cc = uniformCell(gAabbs[HOST_S(1) + awakeCollider(li)], max(PARAMS.gridCellSize, 0.5));
    ASTORE(gConstraintIds[awakeHeads() + gridHash(cc.x, cc.y, cc.z)], 0xFFFFFFFFu);
}

