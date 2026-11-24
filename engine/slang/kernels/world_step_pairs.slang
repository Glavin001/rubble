uint jointPairHash(uint bodyA, uint bodyB)
{
    uint lo = min(bodyA, bodyB);
    uint hi = max(bodyA, bodyB);
    return (lo * 73856093u) ^ (hi * 19349663u) ^ (lo << 16) ^ (hi >> 3);
}


[shader("compute")]
[numthreads(64, 1, 1)]
void joint_hash_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= jointHashSize()) return;
    ASTORE(gEvents[i], JOINT_HASH_EMPTY);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void joint_hash_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint nJ = gCounters[0].jointCount;
    if (i >= nJ) return;
    Joint jj = gJoints[i];
    // Broken joints stop filtering; COLLIDE joints never filter (PhysX flag);
    // a wheel standing on a crate must still let the chassis collide with it.
    if ((jj.flags & (JOINT_FLAG_BROKEN | JOINT_FLAG_COLLIDE | JOINT_FLAG_TYRE)) != 0u) return;
    uint mask = jointHashSize() - 1u;
    uint h = jointPairHash(jj.bodyA, jj.bodyB) & mask;
    for (uint probe = 0; probe < JOINT_HASH_PROBES; probe++)
    {
        uint slot = (h + probe) & mask;
        // Retry the same slot while WGSL's weak CAS fails spuriously: advancing
        // would leave an EMPTY hole and break open-address lookups.
        uint owner = JOINT_HASH_EMPTY;
        for (uint spin = 0u; spin < 64u && owner == JOINT_HASH_EMPTY; spin++)
        {
            uint prev;
            ACAS4(gEvents[slot], JOINT_HASH_EMPTY, i, prev);
            owner = ALOAD(gEvents[slot]);
        }
        if (owner == i)
        {
            uint size = mask + 1u;
            ASTORE(gEvents[size + slot], min(jj.bodyA, jj.bodyB));
            ASTORE(gEvents[2u * size + slot], max(jj.bodyA, jj.bodyB));
            return;
        }
    }
    AOR2(gCounters[0].overflowFlags, OVERFLOW_JOINTS);
}

bool bodiesJointed(uint bodyA, uint bodyB)
{
    // Ring/Wall have zero joints — skip the open-address probe entirely (and do not
    // require a clear of gEvents each step).
    if (gCounters[0].jointCount == 0u) return false;
    uint lo = min(bodyA, bodyB);
    uint hi = max(bodyA, bodyB);
    uint mask = jointHashSize() - 1u;
    uint h = jointPairHash(lo, hi) & mask;
    for (uint probe = 0; probe < JOINT_HASH_PROBES; probe++)
    {
        uint slot = (h + probe) & mask;
        if (ALOAD(gEvents[slot]) == JOINT_HASH_EMPTY) return false;
        if (ALOAD(gEvents[mask + 1u + slot]) == lo && ALOAD(gEvents[2u * (mask + 1u) + slot]) == hi)
            return true;
    }
    return false;
}

// Collision filtering (per collider; HOST_S(30) != 0 when any filter is set).
// Collider.generation: categories (bits 0..15) | mask (bits 16..31); 0 means
// "every category, collides with everything". Collider.material bits 16..31:
// signed group; a shared positive group always collides and a shared negative
// group never does, overriding categories (Box2D-style), for e.g. ragdolls or
// the chunks of one fractured compound.
bool collidersFilteredOut(uint i, uint j)
{
    Collider a = gColliders[i];
    Collider b = gColliders[j];
    int ga = int(a.material) >> 16;
    int gb = int(b.material) >> 16;
    if (ga != 0 && ga == gb) return ga < 0;
    uint fa = a.generation == 0u ? 0xFFFFFFFFu : a.generation;
    uint fb = b.generation == 0u ? 0xFFFFFFFFu : b.generation;
    return ((fa & 0xFFFFu) & (fb >> 16)) == 0u || ((fb & 0xFFFFu) & (fa >> 16)) == 0u;
}

// Bits 12..15 are shape flags (COLLIDER_CONTACT_EVENTS and friends).
uint materialIndex(Collider c) { return c.material & 0x0FFFu; }

// Look-ahead pairs are tested with their relative sweep: each box is swept by
// its own v·h for the broadphase, so pieces carried along together (debris in
// front of the ball) overlap their neighbours' boxes although they cannot
// close on them. Undo both sweeps and sweep one box by (vi − vj)·h instead.
bool lookAheadSeparated(uint i, uint j)
{
    if (!lookAheadOn()) return false;
    Collider ci = gColliders[i];
    Collider cj = gColliders[j];
    BodyState bi = gBodies[ci.body];
    BodyState bj = gBodies[cj.body];
    float3 si = colliderReach(bi, ci) > 0.0 ? bi.linearVel * PARAMS.dt : float3(0, 0, 0);
    float3 sj = colliderReach(bj, cj) > 0.0 ? bj.linearVel * PARAMS.dt : float3(0, 0, 0);
    if (all(si == float3(0, 0, 0)) && all(sj == float3(0, 0, 0))) return false;
    uint base = HOST_S(1);
    Aabb ai = gAabbs[base + i];
    Aabb aj = gAabbs[base + j];
    float3 imn = ai.mn - min(si, float3(0, 0, 0)), imx = ai.mx - max(si, float3(0, 0, 0));
    float3 jmn = aj.mn - min(sj, float3(0, 0, 0)), jmx = aj.mx - max(sj, float3(0, 0, 0));
    float3 rel = si - sj;
    imn = min(imn, imn + rel);
    imx = max(imx, imx + rel);
    return any(imx < jmn) || any(jmx < imn);
}

// Vehicle sever grace (HOST_S(30) bit 1, set by vehicle hosts when the scene has
// no category filters): Collider.generation = ignored body + 1 (0 = none), so a
// severed wheel or part does not explode out of the chassis it was attached
// to. The vehicle kernels write it on the GPU and clear it once they separate.
bool collidersIgnoreBody(uint i, uint j)
{
    Collider a = gColliders[i];
    Collider b = gColliders[j];
    return (a.generation != 0u && a.generation - 1u == b.body) ||
           (b.generation != 0u && b.generation - 1u == a.body);
}

void emitCandidateCore(uint i, uint j);
void archiveTouch(uint i, uint j);

void emitCandidate(uint i, uint j)
{
#if defined(ENGINE_SLEEP_ARCHIVE)
    if (sleepArchive() && i != j)
        archiveTouch(i, j);
#endif
    emitCandidateCore(i, j);
}

void emitCandidateCore(uint i, uint j)
{
    if (i == j) return;
    uint filterMode = HOST_S(30);
    if ((filterMode & 1u) != 0u ? collidersFilteredOut(i, j)
        : (filterMode & 2u) != 0u && collidersIgnoreBody(i, j))
        return;
    if (lookAheadSeparated(i, j)) return;
    uint a = min(i, j);
    uint b = max(i, j);
    uint slot;
    AADD3(gCounters[0].candidateCount, 1, slot);
    uint maxCand = HOST_S(2);
    if (slot >= maxCand)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_CANDIDATES);
        return;
    }
    CandidatePair p;
    p.colliderA = a;
    p.colliderB = b;
    p.key = (a << 16) ^ b;
    p._pad = 0;
    gCandidates[slot] = p;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // Split grids (HOST_S(20) != 0): awake heads [0, n), other heads [n, 2n).
    if (i >= gridHashSize() * (gridSplit() ? 2u : 1u)) return;
    ASTORE(gConstraintIds[i], 0xFFFFFFFFu);
    if (i == 0u && gridSplit()) ASTORE(gScratchAtomics[52], 0u);
    // Metal hosts do not rewrite GPU-owned scratch words each step: reset the
    // large-dynamic (25), plane (42) and large-static (53) list counts here.
    if (i == 0u)
    {
        ASTORE(gScratchAtomics[25], 0u);
        ASTORE(gScratchAtomics[42], 0u);
        ASTORE(gScratchAtomics[LARGE_STATIC_COUNT], 0u);
        ASTORE(gScratchAtomics[CCD_BODY_COUNT], 0u);
        ASTORE(gScratchAtomics[FAST_COUNT], 0u);
        ASTORE(gScratchAtomics[CCD_FALLBACKS], 0u);
        ASTORE(gScratchAtomics[CCD_PAIR_COUNT], 0u);
        ASTORE(gScratchAtomics[CCD_MAX_REACH], 0u);
    }
}

BoxObb worldObbFromCache(uint colliderIdx)
{
    WorldObbGpu w = gWorldObbs[colliderIdx];
    BoxObb b;
    b.center = w.center.xyz;
    b.half = w.half.xyz;
    b.axis0 = w.axis0.xyz;
    b.axis1 = w.axis1.xyz;
    b.axis2 = w.axis2.xyz;
    return b;
}

// three-avbd Stage 8h: reject AABB-overlapping box pairs separated on a face axis
// by more than `gap`. `probe` OBB is built once per broadphase thread.
bool boxesSeparatedOnFaceAxis(BoxObb probe, uint otherCollider, float gap)
{
    Collider cb = gColliders[otherCollider];
    if (!shapeHasHull(cb.shape)) return false;
    BoxObb other = worldObbFromCache(otherCollider);
    gap += cachedReach(otherCollider);
    float3 d = other.center - probe.center;
    float3 hA = probe.half;
    float3 hB = other.half;
    float3 a0 = probe.axis0;
    float3 a1 = probe.axis1;
    float3 a2 = probe.axis2;
    float3 b0 = other.axis0;
    float3 b1 = other.axis1;
    float3 b2 = other.axis2;
    {
        float rB = hB.x * abs(dot(a0, b0)) + hB.y * abs(dot(a0, b1)) + hB.z * abs(dot(a0, b2));
        if (abs(dot(d, a0)) > hA.x + rB + gap) return true;
        rB = hB.x * abs(dot(a1, b0)) + hB.y * abs(dot(a1, b1)) + hB.z * abs(dot(a1, b2));
        if (abs(dot(d, a1)) > hA.y + rB + gap) return true;
        rB = hB.x * abs(dot(a2, b0)) + hB.y * abs(dot(a2, b1)) + hB.z * abs(dot(a2, b2));
        if (abs(dot(d, a2)) > hA.z + rB + gap) return true;
    }
    {
        float rA = hA.x * abs(dot(b0, a0)) + hA.y * abs(dot(b0, a1)) + hA.z * abs(dot(b0, a2));
        if (abs(dot(d, b0)) > hB.x + rA + gap) return true;
        rA = hA.x * abs(dot(b1, a0)) + hA.y * abs(dot(b1, a1)) + hA.z * abs(dot(b1, a2));
        if (abs(dot(d, b1)) > hB.y + rA + gap) return true;
        rA = hA.x * abs(dot(b2, a0)) + hA.y * abs(dot(b2, a1)) + hA.z * abs(dot(b2, a2));
        if (abs(dot(d, b2)) > hB.z + rA + gap) return true;
    }
    return false;
}

// ---- Hierarchical size-level grid (HOST_S(18) = level count; 0 = uniform grid) ----
// Geometric levels in the spirit of three-avbd's fine/coarse ownership. Each
// collider is inserted once, at the level whose cell (base·2^level) covers its
// bound diameter, so compound children and mixed hull sizes never inflate the
// cell of the whole scene. A collider queries its own and every coarser level's
// 27 cells; fixed/sleeping fine colliders still query coarse movers (coarse
// movers never visit fine cells). Colliders beyond the top level (oversized
// statics) go to the explicit large list, tested by every mover.
static const uint HIER_LEVEL_MASK_SLOT = 51u;
// Uniform-grid large dynamics: count at 25, up to 16 IDs at 64..79. Metal reads
// HOST_S(k) from these same scratch words, so the IDs must stay clear of every
// host word (0..31): at 26..41 they overwrote HOST_S(26..30) mid-step.
static const uint LEGACY_LARGE_BASE = 64u;
// Levels holding at least one moving collider (scratch 31, zeroed by every host
// each step). Sleeping/static queriers only pair with movers, so they skip
// every level without one, and return at once when none is coarser than them.
static const uint HIER_MOVING_MASK_SLOT = 31u;
// Oversized statics: count at 53, up to 10 collider IDs at 54..63.
static const uint HIER_LARGE_COUNT_SLOT = 53u;
static const uint HIER_LARGE_MAX = 10u;
// Moving colliders above the finest level (count at 32, up to 9 IDs at
// 33..41; host-zeroed each step). Sleeping/static colliders only pair with
// coarser movers, so while the list is complete a sleeper that overlaps none
// of them skips its query: the same pairs, without 27 hash probes per level.
static const uint HIER_COARSE_COUNT_SLOT = 32u;
static const uint HIER_COARSE_MAX = 9u;

uint hierLevel(float radius)
{
    float diameter = 2.0 * radius + 0.0004;
    return uint(max(0.0, ceil(log2(max(diameter / PARAMS.gridCellSize, 1.0)))));
}

float hierCellSize(uint level) { return PARAMS.gridCellSize * exp2(float(level)); }

int3 hierCell(Aabb a, uint level)
{
    return int3(floor((a.mn + a.mx) * (0.5 / hierCellSize(level))));
}

uint hierHash(int3 c, uint level)
{
    return (gridHash(c.x, c.y, c.z) ^ (level * 2654435769u)) & (gridHashSize() - 1u);
}

bool aabbOverlap(Aabb a, Aabb b)
{
    return all(a.mn <= b.mx) && all(b.mn <= a.mx);
}

bool bodyMoving(BodyState b) { return bodyIsDynamic(b) && !bodyIsSleeping(b); }
// Movers for pair bookkeeping: awake dynamics and kinematics whose target moves.
bool bodyMovingOrKinematic(BodyState b) { return bodyMoving(b) || kinematicMoving(b); }

// AABB and plane n·x = w: true when the box lies entirely above the plane.
bool aabbAbovePlane(Aabb a, float4 plane)
{
    float3 n = normalize(plane.xyz);
    float3 c = 0.5 * (a.mn + a.mx);
    float3 e = 0.5 * (a.mx - a.mn);
    return dot(n, c) - dot(abs(n), e) - plane.w > 0.0;
}

void hierInsert(uint i)
{
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gScratchAtomics[42], 1, slot);
        if (slot < 8u) ASTORE(gScratchAtomics[43u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    Aabb box = gAabbs[HOST_S(1) + i];
    uint level = hierLevel(box._pad0);
    if (level >= HOST_S(18))
    {
        uint slot;
        AADD3(gScratchAtomics[HIER_LARGE_COUNT_SLOT], 1, slot);
        if (slot < HIER_LARGE_MAX) ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    // Occupied-level mask (scratch 51, zeroed by the host each step): queries
    // skip empty levels, so the host can keep slack levels for f32 rounding.
    AOR2(gScratchAtomics[HIER_LEVEL_MASK_SLOT], 1u << level);
    if (bodyMovingOrKinematic(gBodies[col.body]))
    {
        AOR2(gScratchAtomics[HIER_MOVING_MASK_SLOT], 1u << level);
        if (level > 0u)
        {
            uint k;
            AADD3(gScratchAtomics[HIER_COARSE_COUNT_SLOT], 1u, k);
            if (k < HIER_COARSE_MAX) ASTORE(gScratchAtomics[HIER_COARSE_COUNT_SLOT + 1u + k], i);
        }
    }
    uint prev;
    AXCHG3(gConstraintIds[hierHash(hierCell(box, level), level)], i, prev);
    // The chain link lives in the AABB's spare word, as in the split grid:
    // a query hop then fetches link and bounds with one load.
    gAabbs[HOST_S(1) + i]._pad1 = asfloat(prev);
}

// ---- Persistent resting table (JS host; HOST_S(25) bit 0) ----
// Static and sleeping colliders live in a resting table (R) that persists
// across steps; moving colliders go into an awake table (W) each step, and
// only they query (broadphase_query_awake). Both tables sit past the CSR
// index in gConstraintIds (restBase): a 32-word header (occupied-level mask,
// planes, oversized statics), R heads, W heads, then R chain links (one word
// per collider; W links stay in the AABB spare word). R is rebuilt from
// scratch on the step after any body fell asleep (or when the host asks):
// otherwise no collider stops moving, so every resting collider is already
// in R at its current cell. A woken collider stays in R as a tombstone: it
// moves, so R visits skip it, and it is found in W instead. W is emptied after
// the queries by the same moving colliders, so no step clears whole tables.
static const uint HIER_LINK_SPARE = 0xFFFFFFFFu;
static const uint REST_HEADER = 32u;
// Header words: 0 level mask, 1 plane count, 2..9 planes, 10 large count,
// 11..20 oversized statics.

bool restTables() { return (HOST_S(25) & 1u) != 0u; }
bool warmTagged();
// Past the tagged warm hash when there is one (HOST_S(9) varies step to step with the
// contact count, so World's untagged tables sit right past the CSR index instead): the
// awake bitmap (a word per 32 body slots, awakeBit), then the tables, which the host
// allocates only once a step takes the awake-indexed broadphase (WgpuSim::ensure_rest_tables).
uint awakeBitsBase() { return HOST_S(24) + HOST_S(1) + 1u + (warmTagged() ? HOST_S(9) : 0u); }
uint restBase() { return awakeBitsBase() + ((HOST_S(1) + 31u) >> 5); }
uint restHeads() { return restBase() + REST_HEADER; }
uint awakeHeads() { return restHeads() + gridHashSize(); }
uint restLinks() { return awakeHeads() + gridHashSize(); }

// One cell's chain (heads at headBase; links in the AABB spare word, or at
// linkBase + j for R) as seen by querier i: the pair rules of hierQuery.
void hierVisitCell(uint i, Collider a, bool active, uint own, Aabb aa, bool probeHasObb,
    BoxObb probe, float faceGap, int3 cell, uint level, uint headBase, uint linkBase, uint count)
{
    uint base = HOST_S(1);
    bool rest = linkBase != HIER_LINK_SPARE;
    uint j = ALOAD(gConstraintIds[headBase + hierHash(cell, level)]);
    uint guard = 0u;
    while (j != 0xFFFFFFFFu && guard < count)
    {
        guard++;
        Aabb ab = gAabbs[base + j];
        uint next = rest ? ALOAD(gConstraintIds[linkBase + j]) : asuint(ab._pad1);
        // Hash buckets alias: accept only colliders whose true cell/level
        // is this one, so no pair is visited twice. The overlap test runs
        // first: it rejects most entries before the log2/floor checks.
        if (j != i && aabbOverlap(aa, ab) && hierLevel(ab._pad0) == level &&
            all(hierCell(ab, level) == cell))
        {
            Collider b = gColliders[j];
            BodyState ob = gBodies[b.body];
            // Kinematic or static pairs need no contact (neither body is solved).
            if (b.body != a.body && (bodyIsDynamic(ob) || bodyIsDynamic(gBodies[a.body])))
            {
                bool other = bodyMovingOrKinematic(ob);
                bool owner = level > own || (active && (!other || i < j));
                // A moving collider in R is a tombstone (found in W).
                if ((active || other) && owner && !(rest && other) &&
                    !bodiesJointed(a.body, b.body) &&
                    !(probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)))
                    emitCandidate(i, j);
            }
        }
        j = next;
    }
    if (j != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
}

void hierQuery(uint i)
{
    Collider a = gColliders[i];
    if (a.shape == SHAPE_PLANE) return;
    BodyState ba = gBodies[a.body];
    bool active = bodyMovingOrKinematic(ba);
    // Fast colliders query along their sweep (broadphase_fast).
    if (colliderIsFast(i, a, ba)) return;
    uint base = HOST_S(1);
    Aabb aa = gAabbs[base + i];
    uint own = hierLevel(aa._pad0);
    uint levels = HOST_S(18);
    // Oversized colliders are static by host contract; movers find them.
    if (own >= levels) return;
    bool probeHasObb = shapeHasHull(a.shape);
    BoxObb probe = probeHasObb ? worldObbFromCache(i)
                               : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeHasObb ? cachedReach(i) : 0.0);
    if (active && bodyIsDynamic(ba))
    {
        uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
        for (uint pi = 0u; pi < nPlanes; pi++)
        {
            uint j = ALOAD(gScratchAtomics[43u + pi]);
            if (!aabbAbovePlane(aa, gColliders[j].params)) emitCandidate(i, j);
        }
        uint nLarge = min(ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT]), HIER_LARGE_MAX);
        for (uint k = 0u; k < nLarge; k++)
        {
            uint j = ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + k]);
            Collider b = gColliders[j];
            if (b.body == a.body || bodiesJointed(a.body, b.body)) continue;
            if (!aabbOverlap(aa, gAabbs[base + j])) continue;
            if (probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)) continue;
            emitCandidate(i, j);
        }
    }
    uint count = gCounters[0].colliderCount;
    // Movers may pair with anything; others only with movers at coarser levels.
    uint occupied = active ? ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT])
                           : ALOAD(gScratchAtomics[HIER_MOVING_MASK_SLOT]) & ~((2u << own) - 1u);
    if ((occupied >> own) == 0u) return;
    // Params flag 512 (diagnostic) disables the prefilter, to compare outputs.
    if (!active && (PARAMS.flags & 512u) == 0u)
    {
        uint nCoarse = ALOAD(gScratchAtomics[HIER_COARSE_COUNT_SLOT]);
        if (nCoarse <= HIER_COARSE_MAX)
        {
            bool near = false;
            for (uint k = 0u; k < nCoarse; k++)
            {
                Aabb mb = gAabbs[base + ALOAD(gScratchAtomics[HIER_COARSE_COUNT_SLOT + 1u + k])];
                if (hierLevel(mb._pad0) > own && aabbOverlap(aa, mb)) near = true;
            }
            if (!near) return;
        }
    }
    for (uint level = own; level < levels; level++)
    {
        if (((occupied >> level) & 1u) == 0u) continue;
        // A level-L collider's AABB centre lies in its cell and its half extent
        // is at most half a cell (the same bound the 27-cell scan relies on), so
        // only centres within aa ± cell/2 can overlap. That range is a subset of
        // the 27 cells: about 8 at the own level and 1-8 at coarser levels.
        float cs = hierCellSize(level);
        int3 lo = int3(floor((aa.mn - 0.5 * cs) / cs));
        int3 hi = int3(floor((aa.mx + 0.5 * cs) / cs));
        int3 center = hierCell(aa, level);
        lo = max(lo, center - 1);
        hi = min(hi, center + 1);
        for (int z = lo.z; z <= hi.z; z++)
        for (int y = lo.y; y <= hi.y; y++)
        for (int x = lo.x; x <= hi.x; x++)
        {
            int3 cell = int3(x, y, z);
            if (restTables())
            {
                hierVisitCell(i, a, active, own, aa, probeHasObb, probe, faceGap, cell, level,
                    awakeHeads(), HIER_LINK_SPARE, count);
                hierVisitCell(i, a, active, own, aa, probeHasObb, probe, faceGap, cell, level,
                    restHeads(), restLinks(), count);
            }
            else
                hierVisitCell(i, a, active, own, aa, probeHasObb, probe, faceGap, cell, level,
                    0u, HIER_LINK_SPARE, count);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    // Long sweeps query along their path instead of sitting in the grid.
    if (colliderIsFast(i, col, gBodies[col.body]))
    {
        fastListAppend(i);
        return;
    }
    if (HOST_S(18) != 0u)
    {
        hierInsert(i);
        return;
    }
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gScratchAtomics[42], 1, slot);
        if (slot < 8u) ASTORE(gScratchAtomics[43u + slot], i);
        // The 9th plane is never queried: fail the step as hierInsert does.
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    // Insert sleeping too — awake movers must find them (island wake on impact).
    uint base = HOST_S(1);
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    Aabb box = gAabbs[base + i];
    float3 ext = box.mx - box.mn;
    // Oversized statics are probed separately; large dynamics use the explicit list.
    BodyState bIns = gBodies[col.body];
    // Statics wider than a cell (ramps, walls, ground) cannot be found from the
    // 27-cell stencil: list them. Past LARGE_STATIC_CAP
    // flag the overflow and keep the legacy behaviour (grid, or endpoint probe).
    float boundR = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    if (!bodyIsDynamic(bIns) && boundR > 0.5 * cellSize)
    {
        uint slot;
        AADD3(gScratchAtomics[LARGE_STATIC_COUNT], 1, slot);
        if (slot < LARGE_STATIC_CAP)
        {
            ASTORE(gScratchAtomics[LARGE_STATIC_BASE + slot], i);
            return;
        }
        AOR2(gCounters[0].overflowFlags, OVERFLOW_LARGE_STATICS);
    }
    if (!bodyIsDynamic(bIns) &&
        (ext.x >= cellSize * 4.0 || ext.y >= cellSize * 4.0 || ext.z >= cellSize * 4.0))
        return;
    float radius = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    // Bigger than the cell (smash ball vs bricks): test from every dynamic, not one hash slot.
    if (bodyIsDynamic(bIns) && radius > cellSize)
    {
        uint slot;
        AADD3(gScratchAtomics[25], 1, slot);
        if (slot < 16u) ASTORE(gScratchAtomics[LEGACY_LARGE_BASE + slot], i);
        // The 17th is never found: fail the step (AGENTS.md) instead of
        // silently dropping its pairs.
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_LARGE_DYNAMICS);
        return;
    }
    float3 center = 0.5 * (box.mn + box.mx);
    int3 cc = int3(floor(center / cellSize));
    uint h = gridHash(cc.x, cc.y, cc.z);
    uint prev;
    bool awake = bodyIsDynamic(bIns) && !bodyIsSleeping(bIns);
    uint entry = i | (awake ? GRID_AWAKE_DYNAMIC : 0u);
    if (gridSplit() && !awake)
    {
        h += gridHashSize();
        AADD2(gScratchAtomics[52], 1u); // non-empty other table
    }
    AXCHG3(gConstraintIds[h], entry, prev);
    // Split grids keep the next link in the AABB's spare word (8eo): queries
    // then fetch link and bounds with one load. The legacy grid keeps gColorLists.
    if (gridSplit()) gAabbs[base + i]._pad1 = asfloat(prev);
    else gColorLists[i] = prev;
}

// Own cell (index 0), then the 13 neighbours lexicographically after it.
int3 gridForwardOffset(uint c)
{
    if (c == 0u) return int3(0, 0, 0);
    if (c <= 9u)
    {
        uint k = c - 1u; // dz = +1: all nine (dx, dy)
        return int3(int(k % 3u) - 1, int(k / 3u) - 1, 1);
    }
    if (c <= 12u) return int3(int(c - 10u) - 1, 1, 0); // dz = 0, dy = +1
    return int3(1, 0, 0);                              // dz = 0, dy = 0, dx = +1
}

