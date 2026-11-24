// ---- Awake-only queries (JS host, moved-only bounds: HOST_S(19) != 0) ----
// Every moving collider then belongs to a body on last step's awake list
// (HOST_S(6) entries; nothing wakes between that list and this broadphase
// without a host write, which turns moved-only bounds off), so only those
// query: hierQuery with an active querier emits exactly the pairs it always
// did. The pairs a resting (sleeping or static) collider found from its own
// thread, with a coarser mover, are found by that mover instead: it scans the
// finer cells for resting colliders (hierScanFiner), replaying the resting
// collider's own acceptance test. emitCandidate orders every pair (min, max),
// so the candidate set is the same; only its append order differs, and every
// consumer is order-independent (contacts are keyed and CSR rows sorted).
// A coarse mover's finer cells are split into chunks of 64 (one lane each):
// the awake query appends one work item (mover, chunk) per chunk to a list in
// gConstraintIds between the grid heads (HOST_S(16) words) and the CSR base
// (HOST_S(24)), unused until the CSR build; the item count is scratch word 5
// (host-zeroed; HOST_S(5) is unused). broadphase_query_finer runs one
// workgroup per item, so a large mover's scan spreads over many SMs.
static const uint FINER_COUNT_SLOT = 5u;
static const uint FINER_CHUNK = 64u;
static const uint FINER_SKIP = 0xFFFFFFFFu;

uint finerItemCapacity() { return (HOST_S(24) - HOST_S(16)) / 2u; }

// Cells of `level` that can hold the centre of a collider overlapping `ma`.
// Level-L colliders reach at most cs/2 (plus the AABB pad, which _pad0 leaves
// out) from their centre.
uint3 finerSpan(Aabb ma, uint level, out int3 lo)
{
    float cs = hierCellSize(level);
    float reach = 0.5 * cs + max(1.0e-4, speculativeMargin());
    lo = int3(floor((ma.mn - reach) / cs));
    int3 hi = int3(floor((ma.mx + reach) / cs));
    return uint3(hi - lo + 1);
}

uint finerLevelMask(uint own)
{
    return ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT]) & ((1u << own) - 1u);
}

uint hierFinerCellCount(Aabb ma, uint own)
{
    uint mask = finerLevelMask(own), total = 0u;
    for (uint level = 0u; level < own; level++)
    {
        if (((mask >> level) & 1u) == 0u) continue;
        int3 lo;
        uint3 span = finerSpan(ma, level, lo);
        total += span.x * span.y * span.z;
    }
    return total;
}

// Pairs of coarse mover m with resting colliders at finer levels, over the
// finer cells [first, end) (every level's cells in turn) with step `stride`.
// A resting collider j at level L emitted (j, m) from hierQuery iff they
// overlap and m's cell lies in j's range at m's level; the same test runs
// here on the same values, so the accepted pairs are identical. Every j that
// overlaps m has its centre in one of the cells finerSpan lists.
// The face-axis test runs as hierQuery runs it for j: j the probe, m the other
// (finerSeparatedOnFaceAxis). With m the probe the same axes are tested in
// another order of operations, and pieces that share a face plane (a wall's
// neighbours) sit on the test's threshold: the last bit decided differently.
bool finerSeparatedOnFaceAxis(uint j, Collider b, uint m)
{
    if (!shapeHasHull(b.shape)) return false;
    float faceGap = max(0.001, speculativeMargin()) + cachedReach(j);
    return boxesSeparatedOnFaceAxis(worldObbFromCache(j), m, faceGap);
}

void hierScanFiner(uint m, uint first, uint stride, uint end)
{
    uint base = HOST_S(1);
    Aabb ma = gAabbs[base + m];
    uint own = hierLevel(ma._pad0);
    Collider cm = gColliders[m];
    uint mask = finerLevelMask(own);
    uint count = gCounters[0].colliderCount;
    float csM = hierCellSize(own);
    int3 cellM = hierCell(ma, own);
    for (uint t = first; t < end; t += stride)
    {
        uint rem = t, level = 0u;
        int3 lo = int3(0, 0, 0);
        uint3 span = uint3(1, 1, 1);
        for (; level < own; level++)
        {
            if (((mask >> level) & 1u) == 0u) continue;
            span = finerSpan(ma, level, lo);
            uint n = span.x * span.y * span.z;
            if (rem < n) break;
            rem -= n;
        }
        if (level >= own) break;
        int3 cell = lo + int3(uint3(rem % span.x, (rem / span.x) % span.y, rem / (span.x * span.y)));
        // Resting colliders: the grid, or only R (W holds movers alone).
        bool rest = restTables();
        uint j = ALOAD(gConstraintIds[(rest ? restHeads() : 0u) + hierHash(cell, level)]);
        uint guard = 0u;
        while (j != 0xFFFFFFFFu && guard < count)
        {
            guard++;
            Aabb ab = gAabbs[base + j];
            uint next = rest ? ALOAD(gConstraintIds[restLinks() + j]) : asuint(ab._pad1);
            if (aabbOverlap(ma, ab) && hierLevel(ab._pad0) == level &&
                all(hierCell(ab, level) == cell))
            {
                // j's query range at m's level, as hierQuery computes it.
                int3 jc = hierCell(ab, own);
                int3 jlo = max(int3(floor((ab.mn - 0.5 * csM) / csM)), jc - 1);
                int3 jhi = min(int3(floor((ab.mx + 0.5 * csM) / csM)), jc + 1);
                if (all(cellM >= jlo) && all(cellM <= jhi))
                {
                    Collider b = gColliders[j];
                    if (b.body != cm.body && !bodyMoving(gBodies[b.body]) &&
                        !bodiesJointed(cm.body, b.body) &&
                        !finerSeparatedOnFaceAxis(j, b, m))
                        emitCandidate(j, m);
                }
            }
            j = next;
        }
        if (j != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
    }
}

// One thread per listed awake collider (indirect, work slot 81).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query_awake(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    uint levels = HOST_S(18);
    // Fast colliders are not in the grid: broadphase_fast owns their pairs.
    if (colliderIsFast(c, gColliders[c], gBodies[gColliders[c].body])) return;
    hierQuery(c);
    if (gColliders[c].shape == SHAPE_PLANE) return;
    Aabb aa = gAabbs[HOST_S(1) + c];
    uint own = hierLevel(aa._pad0);
    if (own == 0u || own >= levels) return;
    uint cells = hierFinerCellCount(aa, own);
    if (cells == 0u) return;
    uint chunks = (cells + FINER_CHUNK - 1u) / FINER_CHUNK;
    uint k;
    AADD3(gScratchAtomics[FINER_COUNT_SLOT], chunks, k);
    uint cap = finerItemCapacity();
    bool fits = k + chunks <= cap;
    for (uint q = 0u; q < chunks && k + q < cap; q++)
    {
        ASTORE(gConstraintIds[HOST_S(16) + 2u * (k + q)], fits ? c : FINER_SKIP);
        ASTORE(gConstraintIds[HOST_S(16) + 2u * (k + q) + 1u], q);
    }
    if (!fits) hierScanFiner(c, 0u, 1u, cells); // list full: scan in this thread
}

// Colliders of moving bodies on last step's awake list (one thread per entry,
// work slot 78), so the awake insert, query and clear run one thread per
// collider: a body with many colliders no longer serializes them in one thread.
// Listed in gConstraintIds after the resting links (awakeColliderBase); the
// count is scratch word 6 (host-zeroed; HOST_S(6) lives in the uniform).
static const uint AWAKE_COLLIDER_COUNT_SLOT = 6u;
uint awakeColliderBase() { return restLinks() + gCounters[0].colliderCount; }
uint awakeColliderCount() { return ALOAD(gScratchAtomics[AWAKE_COLLIDER_COUNT_SLOT]); }
uint awakeCollider(uint li) { return ALOAD(gConstraintIds[awakeColliderBase() + li]); }

[shader("compute")]
[numthreads(64, 1, 1)]
void list_awake_colliders(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= HOST_S(6)) return;
    uint body = gAwakeList[li];
    if (!bodyMoving(gBodies[body])) return;
    uint n = gCounters[0].colliderCount;
    uint first = firstColliderOfBody(body), end = first;
    while (end < n && gColliders[end].body == body) end++;
    if (end == first) return;
    uint base;
    AADD3(gScratchAtomics[AWAKE_COLLIDER_COUNT_SLOT], end - first, base);
    for (uint c = first; c < end; c++) ASTORE(gConstraintIds[awakeColliderBase() + base + (c - first)], c);
}

// Work slot 81: one thread per listed awake collider.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_awake_collider_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(81u, awakeColliderCount());
}

// Work slot 77: one workgroup per listed chunk.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_finer_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(77u, min(ALOAD(gScratchAtomics[FINER_COUNT_SLOT]), finerItemCapacity()) * 64u);
}

// Whether a re-framed body at rest was seen this step (list_moving_colliders, below).
bool reframedPending()
{
#if defined(ENGINE_WGSL)
    if ((HOST_S(25) & 64u) == 0u) return false;
    return ALOAD(gConstraintIds[restBase() + 27u]) != 0u;
#else
    return false;
#endif
}

// Work slot 78: last step's awake list (HOST_S(6) entries), for moved-only
// bounds and awake-only queries. Slot 67 is reused by the sleep pass.
// Slots 79/80: the resting-table rebuild (clear, insert), dispatched only
// when last step put bodies to sleep (HOST_S(5) = its sleep events) or the
// host asks (HOST_S(25) bit 1).
[shader("compute")]
[numthreads(64, 1, 1)]
void write_moved_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(78u, HOST_S(6));
    // With incremental inserts (HOST_S(25) bit 5) only when they could not place a new
    // sleeper (header word 31); otherwise after any sleep event (HOST_S(5)).
    bool due = restIncremental() ? ALOAD(gConstraintIds[restBase() + REST_REBUILD_WORD]) != 0u
                                 : HOST_S(5) != 0u;
    bool rebuild = restTables() && (due || (HOST_S(25) & 2u) != 0u || reframedPending());
    writeDispatchArgs(79u, rebuild ? REST_HEADER + 2u * gridHashSize() : 0u);
    writeDispatchArgs(80u, rebuild ? gCounters[0].colliderCount : 0u);
}

// Rebuild 1: empty both tables and the header.
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= REST_HEADER + 2u * gridHashSize()) return;
    ASTORE(gConstraintIds[restBase() + i], i < REST_HEADER ? 0u : 0xFFFFFFFFu);
}

// Rebuild 2: every collider not moving now goes into R (planes and oversized
// statics into the header lists, as hierInsert lists them each step).
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    // Where the collider sits in R, for the incremental inserts (rest_sleepers_mark).
    ASTORE(gConstraintIds[restAtBase() + i], REST_AT_NONE);
    Collider col = gColliders[i];
    if (bodyMoving(gBodies[col.body])) return;
    uint hdr = restBase();
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gConstraintIds[hdr + 1u], 1u, slot);
        if (slot < 8u) ASTORE(gConstraintIds[hdr + 2u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        ASTORE(gConstraintIds[restAtBase() + i], REST_AT_LISTED);
        return;
    }
    Aabb box = gAabbs[HOST_S(1) + i];
    uint level = hierLevel(box._pad0);
    if (level >= HOST_S(18))
    {
        uint slot;
        AADD3(gConstraintIds[hdr + 10u], 1u, slot);
        if (slot < HIER_LARGE_MAX) ASTORE(gConstraintIds[hdr + 11u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        ASTORE(gConstraintIds[restAtBase() + i], REST_AT_LISTED);
        return;
    }
    AOR2(gConstraintIds[hdr], 1u << level);
    uint h = hierHash(hierCell(box, level), level);
    uint prev;
    AXCHG3(gConstraintIds[restHeads() + h], i, prev);
    ASTORE(gConstraintIds[restLinks() + i], prev);
    ASTORE(gConstraintIds[restAtBase() + i], h);
}

// Each step, before the awake inserts: R's levels, planes and oversized
// statics into the scratch words hierQuery reads (hierInsert's lists).
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_publish(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint hdr = restBase();
    // Rest steps skip broadphase_clear: reset the GPU-owned CCD lists here.
    ASTORE(gScratchAtomics[CCD_BODY_COUNT], 0u);
    ASTORE(gScratchAtomics[FAST_COUNT], 0u);
    ASTORE(gScratchAtomics[CCD_FALLBACKS], 0u);
    ASTORE(gScratchAtomics[CCD_PAIR_COUNT], 0u);
    ASTORE(gScratchAtomics[CCD_MAX_REACH], 0u);
    AOR2(gScratchAtomics[HIER_LEVEL_MASK_SLOT], ALOAD(gConstraintIds[hdr]));
    uint planes = ALOAD(gConstraintIds[hdr + 1u]);
    ASTORE(gScratchAtomics[42], planes);
    for (uint k = 0u; k < min(planes, 8u); k++)
        ASTORE(gScratchAtomics[43u + k], ALOAD(gConstraintIds[hdr + 2u + k]));
    uint large = ALOAD(gConstraintIds[hdr + 10u]);
    ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT], large);
    // The table's overflows, every step (its inserts flag only the step they ran).
    if (planes > 8u || large > HIER_LARGE_MAX) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
    for (uint k = 0u; k < min(large, HIER_LARGE_MAX); k++)
        ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + k], ALOAD(gConstraintIds[hdr + 11u + k]));
}

// Listed awake colliders into W (one thread each, slot 81).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert_awake(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    if (gColliders[c].shape == SHAPE_PLANE) return;
    if (colliderIsFast(c, gColliders[c], gBodies[gColliders[c].body]))
    {
        fastListAppend(c);
        return;
    }
    Aabb box = gAabbs[HOST_S(1) + c];
    uint level = hierLevel(box._pad0);
    if (level >= HOST_S(18))
    {
        uint slot;
        AADD3(gScratchAtomics[HIER_LARGE_COUNT_SLOT], 1, slot);
        if (slot < HIER_LARGE_MAX) ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + slot], c);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    AOR2(gScratchAtomics[HIER_LEVEL_MASK_SLOT], 1u << level);
    uint prev;
    AXCHG3(gConstraintIds[awakeHeads() + hierHash(hierCell(box, level), level)], c, prev);
    gAabbs[HOST_S(1) + c]._pad1 = asfloat(prev);
}

// After the queries: empty the W heads the listed colliders filled (slot 81).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_clear_awake(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    if (gColliders[c].shape == SHAPE_PLANE) return;
    if (colliderIsFast(c, gColliders[c], gBodies[gColliders[c].body])) return;
    Aabb box = gAabbs[HOST_S(1) + c];
    uint level = hierLevel(box._pad0);
    if (level < HOST_S(18))
        ASTORE(gConstraintIds[awakeHeads() + hierHash(hierCell(box, level), level)], 0xFFFFFFFFu);
}

// ---- Awake-indexed broadphase (World's rigid_plan, HOST_S(25) bit 0) ----
// Static and sleeping colliders stay in the resting table R between steps; a step
// lists the colliders of its moving bodies (bodyMovingOrKinematic: exactly the
// colliders hierQuery treats as active) and only they are bounded, inserted (into W)
// and queried. The list is one cheap scan over the colliders (each reads its body's
// flags), counted per workgroup so a fully awake world does not serialize on one
// atomic; within a workgroup the order is the colliders' own.
groupshared uint gsMoverFlags[64];
groupshared uint gsMoverBase;

// A GPU fracture between steps (stress-gpu fracture.slang fr_commit, fr_reparent) re-frames
// bodies and re-points colliders with no host write. A piece that moves is listed and
// bounded like any mover. A static one (a struck building's anchored remainder) keeps its
// place, but its body's frame and its colliders' local poses are new, so bounds built from
// them differ in the last bits from the ones stored, and its cell can differ with them.
// The fracture stamps every body it writes with the engine's next step as its generation
// (the engine itself never stamps a static body: update_refs), so a body at rest whose
// generation is this step's index was re-framed since the last step. Seeing one asks for
// what a host write would: every collider bounded and the resting table rebuilt, this
// step (header word 27, which rest_clear empties). HOST_S(25) bit 6: the world has such a
// fracture. WGSL hosts only, as the rest tables are.
static const uint REST_REFRAMED_WORD = 27u;
#if defined(ENGINE_WGSL)
bool gpuReframes() { return (HOST_S(25) & 64u) != 0u; }
#else
bool gpuReframes() { return false; }
#endif
void noteReframed(BodyState b)
{
    if (b.generation != HOST_S(17)) return;
    ASTORE(gConstraintIds[restBase() + REST_REFRAMED_WORD], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void list_moving_colliders(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint c = dispatchIndex(tid);
    uint local = lid.x;
    bool moving = false;
    if (c < gCounters[0].colliderCount)
    {
        BodyState b = gBodies[gColliders[c].body];
        moving = bodyMovingOrKinematic(b);
        if (!moving && gpuReframes()) noteReframed(b);
    }
    gsMoverFlags[local] = moving ? 1u : 0u;
    GROUP_SYNC();
    if (local == 0u)
    {
        uint n = 0u;
        for (uint k = 0u; k < 64u; k++) n += gsMoverFlags[k];
        uint base = 0u;
        if (n > 0u) AADD3(gScratchAtomics[AWAKE_COLLIDER_COUNT_SLOT], n, base);
        gsMoverBase = base;
    }
    GROUP_SYNC();
    if (moving)
    {
        uint rank = 0u;
        for (uint k = 0u; k < local; k++) rank += gsMoverFlags[k];
        ASTORE(gConstraintIds[awakeColliderBase() + gsMoverBase + rank], c);
    }
}

// Bounds of the listed moving colliders (slot 81), as the full build_aabbs pass
// builds them. The pass keeps every other collider's bounds: a sleeping body's
// (as the full pass does after step 0), and a static's or a stopped kinematic's,
// which the full pass would rebuild to the same values from the same pose and
// parameters (hosts run the full pass after any write that could change them).
[shader("compute")]
[numthreads(64, 1, 1)]
void build_aabbs_listed(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    Collider col = gColliders[c];
    buildColliderBounds(c, col, gBodies[col.body]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query_finer(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint g = gid.x + gid.y * 32768u;
    if (g >= min(ALOAD(gScratchAtomics[FINER_COUNT_SLOT]), finerItemCapacity())) return;
    uint m = ALOAD(gConstraintIds[HOST_S(16) + 2u * g]);
    if (m == FINER_SKIP) return;
    uint t = ALOAD(gConstraintIds[HOST_S(16) + 2u * g + 1u]) * FINER_CHUNK + lid.x;
    hierScanFiner(m, t, FINER_CHUNK, t + 1u);
}

// Back-compat name used by CUDA/WebGPU build lists.
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_grid(uint3 tid : SV_DispatchThreadID)
{
    broadphase_query(tid);
}

