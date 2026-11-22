// --- Self-collision's listed pairs across workgroups (EB_SELF_SPLIT) ---
// engine/docs/measurements/selfcol-design.md. A solo frame a step a dispatch
// (EB_SOLO_STEP, the pair list's state saved between steps), cut at its self-collision
// tests: the step dispatch (explicit_beams_self_step, a workgroup a vehicle) runs step
// s - 1's rest (selfApply of its hits, the slides, the closing barrier) and step s up to
// its tests: the grid, the list's decision and build, the volatile triangles' tests and,
// past the list's room, the walk with its tests (all as pairReuseTests runs them); then
// the pairs dispatch (explicit_beams_self_pairs) tests every vehicle's listed pairs, a
// thread a pair over the whole GPU (pairReuseTest, as pairReuseTests' lanes test them).
// A last step dispatch (step = steps) runs the last step's rest and the frame's end.
// The same functions on the same state: tryContact's effects are order-free (fixed-point
// sums, clipRecord's storage atomics, idempotent flags) or ordered before use (selfApply
// applies the hit list in the specification's order), and every value the cut crosses
// is in storage: the hit flag (gsHit, soloStepSave; the pairs' hits from the hit count),
// the listed count, the hit list. So the results are the in-workgroup frame's bit for bit.
//
// gAcc in the vehicle's solo-step record (soloStepAt, a free word past REUSE_WORDS): the
// step's listed pairs for the pairs dispatch (0: none; written every step). The pairs
// dispatch's indirect arguments in their own buffer (bound by the step dispatch alone: a
// buffer cannot be storage and indirect in one dispatch), three slots of (x = the most
// chunks of SELF_PAIRS_WG pairs any vehicle listed, y = the vehicles, z = 1): step 0's (slot 2),
// then odd and even steps'. Step dispatch s clears step s + 1's slot (read last by pairs
// dispatch s - 1), the frame's last clears step 0's.
static const uint SELF_LISTED = REUSE_WORDS;
// The pairs and walk dispatches' workgroups: 64 threads (no barrier in either), so a
// list's threads spread over every core of the GPU (512-thread workgroups put ~10k
// pairs on ~20 cores, each core's threads a long dependent chain).
#define SELF_PAIRS_WG 64
#define SELF_WALK_WG 64
[[vk::binding(10, 0)]] RWStructuredBuffer<Atomic<int>> gSelfArgs;
uint selfListedAt(uint v) { return soloStepAt(v) + SELF_LISTED; }
uint selfArgsAt(uint s) { return 3u * (s == 0u ? 2u : (s & 1u)); }
// The walk (a building step's list, from the walk dispatch): per vehicle the pairs it
// listed (a storage atomic, in place of gsPairCount) and whether it walks this step; its
// indirect arguments' slots after the pairs' (x = the most chunks of SELF_WALK_WG (triangle,
// share) threads, y = the vehicles, z = 1). A triangle's cells are spread over WALK_SUBS
// threads (its cells sub, sub + WALK_SUBS, ...: the list's order differs from the
// workgroup's walk, its pairs do not; nothing depends on the list's order).
static const uint SELF_WALK = REUSE_WORDS + 1u;
static const uint WALK_SUBS = 32u;
static const uint WALK_X = 1024u;
// selfListedAt on a building step: the pairs dispatch tests the walk's list (its count
// from the walk, at most TRI_PAIR_CAP; the walk tests the pairs past the room itself).
static const uint SELF_FROM_WALK = 0xFFFFFFFFu;
uint selfWalkAt(uint v) { return soloStepAt(v) + SELF_WALK; }
uint selfWalkArgsAt(uint s) { return 9u + selfArgsAt(s); }
// The walk's and pairs' dispatches' rows (y): the vehicles, or with vehicle sleep's
// awake-indexed dispatch the awake ones (row y's vehicle: selfSplitRow).
bool sleepListedOn();
uint sleepRows();
uint sleepRow(uint y);
uint selfSplitRows() { return sleepListedOn() ? sleepRows() : gP.vehicles; }
uint selfSplitRow(uint y) { return sleepListedOn() ? sleepRow(y) : y; }

// pairReuseTests in the step dispatch (uniform): the listed pairs left to the pairs
// dispatch, everything else as pairReuseTests runs it.
// True (and nothing tested): the listed pairs are past the list's room, so the step walks
// the grid itself (the caller's triangleGrid loop, selfSplitTests's below): one call site
// of the walk where the member step has its own (the compilers inline every call).
bool selfSplitTestsOrWalk(uint v, uint lane, uint mode, uint triOff, uint triCount, float3 origin, float r, float cell,
                          uint candOff, uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    sync();
    // A building step (mode 1): the walk dispatch lists and tests the pairs
    // (selfSplitBuild); the volatile triangles here as pairReuseTests tests them. Its
    // count, the list's validity and statistics after the walk (selfSplitWalkEnd).
    bool build = mode == 1u;
    uint listed = gsCount[GRID_BUCKETS + RU_COUNT].load();
    if (!build && listed > TRI_PAIR_CAP) return true;
    if (!build && lane == 0u)
    {
        gAcc[selfListedAt(v)].store(int(listed));
        uint at = selfArgsAt(gP.pad2);
        // (listed <= TRI_PAIR_CAP: at most TRI_PAIR_CAP / SELF_PAIRS_WG workgroups a vehicle.)
        gSelfArgs[at].max(int(min((listed + SELF_PAIRS_WG - 1u) / SELF_PAIRS_WG, (TRI_PAIR_CAP + SELF_PAIRS_WG - 1u) / SELF_PAIRS_WG)));
        gSelfArgs[at + 1u].store(int(selfSplitRows()));
        gSelfArgs[at + 2u].store(1);
    }
    // (Both the building step and a listed one test the pieces' triangles: one call site,
    // as the compilers inline every call.)
    piecesTests(v, lane, triOff, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount, parity);
    if (build && lane == 0u && gP.pieceBase != 0u) piecesClearBuild();
    return false;
}
void selfSplitTests(uint v, uint lane, uint mode, uint triOff, uint triCount, float3 origin, float r, float cell,
                    uint candOff, uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (!selfSplitTestsOrWalk(v, lane, mode, triOff, triCount, origin, r, cell, candOff, candCount, beamOff, groupOff,
                              groupCount, parity))
        return;
    uint tested = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
    for (uint p = lane, pStop = EB_END(477, p, tested * GRID_SUBS, LANES); p < pStop; p += LANES)
    {
        uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
        triangleGrid(triOff + t, p % GRID_SUBS, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount,
                     parity, 0u);
    }
}

// The next step dispatch's start (lane 0, the pair list's state loaded): a walk's list
// as pairReuseTests records a build's.
void selfSplitWalkEnd(uint v)
{
    if (gAcc[selfWalkAt(v) + 1u].load() == 0) return;
    uint listed = uint(gAcc[selfWalkAt(v)].load());
    gsCount[GRID_BUCKETS + RU_VALID].store(listed <= TRI_PAIR_CAP ? 1u : 0u);
    gsCount[GRID_BUCKETS + RU_COUNT].store(listed);
    gsCount[GRID_BUCKETS + RU_STATS].add(1u);
    gsCount[GRID_BUCKETS + RU_STATS + 2u].add(listed);
    gsCount[GRID_BUCKETS + RU_STATS + 3u].max(listed);
}

// pairReuseBuild in the step dispatch (uniform, after the grid's scan): every node's
// build record and (labels in use) the volatile triangles' list and the contacters'
// separations, as pairReuseBuild / piecesBuild; the triangles' pairs are the walk
// dispatch's (the grid's offsets to storage for it, its arguments).
void selfSplitBuild(uint v, uint lane, uint triOff, uint triCount, uint nodeOff, uint nodeCount, float3 origin,
                    float r, float cell, uint candOff, uint candCount, uint parity)
{
    uint offBase = gP.gridOffBase + v * (GRID_BUCKETS + 1u);
    for (uint b = lane, bStop = EB_END(478, b, GRID_BUCKETS + 1u, LANES); b < bStop; b += LANES) gNext[offBase + b] = gsOff[b];
    if (lane == 0u)
    {
        gAcc[selfWalkAt(v)].store(0);
        gAcc[selfWalkAt(v) + 1u].store(1);
        uint at = selfWalkArgsAt(gP.pad2);
        // (At most WALK_X workgroups: past them each thread takes every WALK_X · SELF_WALK_WG-th.)
        gSelfArgs[at].max(int(min((triCount * WALK_SUBS + SELF_WALK_WG - 1u) / SELF_WALK_WG, WALK_X)));
        gSelfArgs[at + 1u].store(int(selfSplitRows()));
        gSelfArgs[at + 2u].store(1);
        // The walk's list is tested by the pairs dispatch after it (its count unknown
        // here: the list's room, threads past the count return at once).
        gAcc[selfListedAt(v)].store(int(SELF_FROM_WALK));
        uint pa = selfArgsAt(gP.pad2);
        gSelfArgs[pa].max(int((TRI_PAIR_CAP + SELF_PAIRS_WG - 1u) / SELF_PAIRS_WG));
        gSelfArgs[pa + 1u].store(int(selfSplitRows()));
        gSelfArgs[pa + 2u].store(1);
    }
    bool anti = (gStatic[v * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u;
    if (piecesOn(v))
    {
        // piecesBuild without its triangles' lists.
        float margin = asfloat(gP.pairReuse);
        for (uint p = lane, pStop = EB_END(479, p, triCount * LIST_SUBS, LANES); p < pStop; p += LANES)
        {
            uint t = triOff + p / LIST_SUBS;
            if (!pieceVolatile(t)) continue;
            bool gone = ((gMisc[gP.vehicles + t].z & TRI_BREAKABLE) != 0u && triangleDead(t))
                || (gP.retireBase != 0u && pieceTriRetired(t));
            if (p % LIST_SUBS == 0u && !gone) gNext[pieceVolAt(triOff + gsCount[GRID_BUCKETS + RU_VOL].add(1u))] = t;
        }
        float vall = gsCount[GRID_BUCKETS + RU_VALL].load() != 0u ? pieceKey(RU_VALL) : 3.4e38;
        float pb = anti ? r + (vall + clipVmax(v, parity)) * STEP + CLIP_DEPTH : r;
        pb = pb + 1.0e-4 + 1.0e-5 * pb;
        for (uint c = lane, cStop = EB_END(480, c, candCount, LANES); c < cStop; c += LANES)
        {
            uint h = gLists[candOff + c];
            uint g = pieceLabel(h);
            float3 p = gNodes[h * 3u].xyz;
            for (uint a = 0u; a < PIECE_LOOSE; a++)
            {
                if (a == g) continue;
                float s = pieceGap(p, a);
                s = s < 3.0e38 ? max(s - pb, margin) : 3.4e38;
                if (g == 0u)
                {
                    if (s < 3.0e38) pieceKeyMax(RU_S + a, -s);
                }
                else
                    gNext[pieceSigmaAt(h, a)] = asuint(s);
            }
        }
    }
    for (uint i = lane, iStop = EB_END(481, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        float3 P = gNodes[(nodeOff + i) * 3u].xyz;
        float3 V = gNodes[(nodeOff + i) * 3u + 1u].xyz;
        gBeams[gP.pairBuild + nodeOff + i] = float4(P, ssqrt(sdot(V, V)));
    }
    if (lane == 0u)
        gsCount[GRID_BUCKETS + RU_VMAX].store(asuint(anti ? clipVmax(v, parity) : 0.0));
}

// pairReuseList for the walk dispatch: share `sub` of WALK_SUBS of triangle gt's cells,
// the grid's offsets from storage (offBase), the count a storage atomic; the pairs
// dispatch after it tests the listed pairs (pairReuseTest, a thread a pair), and a pair
// past the list's room is tested here, so every pair is tested once (the workgroup walked
// with its tests there instead).
void selfWalkList(uint v, uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                  uint parity, uint listAt, float margin, uint offBase, uint beamOff, uint groupOff, uint groupCount)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    uint flags = gStatic[tn.w * 4u + 3u].x;
    bool anti = (flags & FLAG_ANTI_CLIP) != 0u;
    bool own = anti && (flags & FLAG_OWN_TRAVEL) != 0u;
    float pad = (anti ? triReach(tn, r, parity) : r) + margin;
    float3 mn = min(min(p0, p1), p2);
    float3 mx = max(max(p0, p1), p2);
    float3 lo = mn - float3(pad, pad, pad);
    float3 hi = mx + float3(pad, pad, pad);
    float tv = own ? clipTriSpeed(tn) : 0.0;
    float vm = own ? clipVmax(tn.w, parity) : 0.0;
    float3 f0 = floor((lo - origin) / cell);
    float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    bool every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(1, 1, 1);
    uint yz = 1u;
    uint cells = 0u;
    if (!every)
    {
        c0 = int3(f0);
        span = int3(fs);
        yz = uint(span.y * span.z);
        cells = uint(span.x) * yz;
    }
    uint outer = every ? candCount : cells;
    for (uint i = sub, iStop = EB_END(482, i, outer, WALK_SUBS); i < iStop; i += WALK_SUBS)
    {
        uint e0 = 0u;
        uint e1 = 1u;
        if (!every)
        {
            uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
            bool dup = false;
            int3 cj = int3(0, 0, 0);
            for (uint j = 0u, jStop = EB_END(483, j, i, 1u); j < jStop; j++)
            {
                if (gridBucket(c0 + cj) == b)
                {
                    dup = true;
                    break;
                }
                cj.z += 1;
                if (cj.z == span.z)
                {
                    cj.z = 0;
                    cj.y += 1;
                    if (cj.y == span.y)
                    {
                        cj.y = 0;
                        cj.x += 1;
                    }
                }
            }
            if (dup) continue;
            e0 = gNext[offBase + b];
            e1 = gNext[offBase + b + 1u];
        }
        for (uint e = e0, eGuard = 0u; e < e1; e++)
        {
            EB_PASS(484, eGuard);
            uint h;
            float3 p;
            if (every)
            {
                h = gLists[candOff + i];
                p = nf3(origin + gNodes[h * 3u].xyz);
            }
            else
            {
                float4 E = gBeams[gP.gridBase + candOff + e];
                p = origin + E.xyz;
                h = asuint(E.w);
            }
            if (h == tn.x || h == tn.y || h == tn.z) continue;
            if (any(p < lo) || any(p > hi)) continue;
            if (own)
            {
                float3 V = gNodes[h * 3u + 1u].xyz;
                float vh = ssqrt(sdot(V, V));
                float sp = vh < vm ? vh : vm;
                float q = nf(nf(r + smul(nf(tv + sp), STEP)) + CLIP_DEPTH) + margin;
                if (any(p < mn - float3(q, q, q)) || any(p > mx + float3(q, q, q))) continue;
            }
            uint n = uint(gAcc[selfWalkAt(v)].add(1));
            if (n < TRI_PAIR_CAP)
            {
                // Listed: the pairs dispatch after this one tests it.
                gNext[listAt + 2u * n] = gt;
                gNext[listAt + 2u * n + 1u] = h;
            }
            else
                pairReuseTest(gt, h, origin, r, cell, beamOff, groupOff, groupCount, parity);
        }
    }
}

// Whether vehicle v takes the split's walk and pairs dispatches this frame: every
// vehicle, except with vehicle contact (EB_VV_SOLO) a contact group's member whose step
// runs its tests in its own dispatch (PERF_MEMBER_LIST off: its solo-step words, which
// these dispatches read, are then whatever its last solo frame left). With
// PERF_MEMBER_LIST a member's step is cut at its tests as a solo vehicle's
// (contactMemberStep), so a member takes them too.
// Never a sleeping vehicle (vehicle sleep): its step dispatches do not run, and its
// listed and walk words are what its last awake step left.
bool sleepAsleep(uint v);
uint sleepPick(uint g, uint lane);
bool selfSplitVehicle(uint v)
{
    if (sleepAsleep(v)) return false;
#ifdef EB_VV_SOLO
    return ((gP.perfFlags & PERF_MEMBER_LIST) != 0u && pairReuseOn()) || gMisc[vvRec(v, VV_FLAGS)].y == 0u;
#else
    return true;
#endif
}

// The step dispatch's start (lane 0, before the step's clears): no pairs for the pairs
// dispatch unless selfSplitTests lists them.
void selfSplitClear(uint v)
{
    gAcc[selfListedAt(v)].store(0);
    gAcc[selfWalkAt(v) + 1u].store(0);
}

// The step dispatch's start (vehicle 0's lane 0): the next pairs dispatch's slot cleared
// (step s + 1's; after the last step, step 0's).
void selfSplitArgsClear(uint v, uint lane)
{
    uint s = gP.pad2;
    if (v != 0u || lane != 0u || s + 1u == gP.steps) return;
    uint next = s < gP.steps ? s + 1u : 0u;
    gSelfArgs[selfArgsAt(next)].store(0);
    gSelfArgs[selfWalkArgsAt(next)].store(0);
}

// Step s - 1's hit flag as the in-workgroup frame has it before selfApply (lane 0):
// the flag the step dispatch saved (gsHit, loaded at this dispatch's start) or a hit of
// the pairs dispatch (every hit counts in the hit list's count).
void selfSplitWalkEnd(uint v);
void selfSplitHits(uint v)
{
    selfSplitWalkEnd(v);
    if (gAcc[hitAcc(v)].load() != 0) gsHit = 1u;
    gsUniformHits = gsHit;
    gsUniformOpen = 0u;
}

void explicitBeamsFrameStep(uint3 gid, uint3 lid);
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_self_step(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    selfSplitArgsClear(gid.x, lid.x);
    // Vehicle sleep's awake-indexed dispatch: the workgroup's awake vehicle.
    uint3 vid = gid;
    if (sleepListedOn()) vid.x = sleepPick(gid.x, lid.x);
    explicitBeamsFrameStep(vid, lid);
}

// The pairs dispatch of step gP.pad2: workgroup (x, v) tests vehicle v's listed pairs
// x · SELF_PAIRS_WG .. (x + 1) · SELF_PAIRS_WG - 1, a pair a thread. No barrier.
[shader("compute")]
[numthreads(SELF_PAIRS_WG, 1, 1)]
void explicit_beams_self_pairs(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = selfSplitRow(gid.y);
    uint lane = lid.x;
    uint s = gP.pad2;
    uint listed = v < gP.vehicles && selfSplitVehicle(v) ? uint(gAcc[selfListedAt(v)].load()) : 0u;
    // A building step: the walk's list (its pairs past the room the walk tested).
    if (listed == SELF_FROM_WALK) listed = min(uint(gAcc[selfWalkAt(v)].load()), TRI_PAIR_CAP);
    uint i = gid.x * SELF_PAIRS_WG + lane;
    if (i < listed)
    {
        uint4 r0 = gStatic[v * 4u];
        uint4 r2 = gStatic[v * 4u + 2u];
        uint4 r3 = gStatic[v * 4u + 3u];
        float range = asfloat(r3.z);
        float cell = asfloat(r3.w);
        uint4 misc = gMisc[v];
        float3 origin = asfloat(misc.xyz);
        uint parity = (misc.w + s) & 1u;
        uint at = triPairsAt(v);
        pairReuseTest(gNext[at + 2u * i], gNext[at + 2u * i + 1u], origin, range, cell, r0.z, r2.z, r2.w, parity);
    }
    ebLoopCapFlush();
}

// The walk dispatch of step gP.pad2: workgroup (x, v) takes vehicle v's (triangle, share)
// threads x · SELF_WALK_WG .. (x + 1) · SELF_WALK_WG - 1 when v builds its list this step (every
// triangle, waiting or not, as pairReuseBuild lists them; with labels, not the volatile
// ones). No barrier.
[shader("compute")]
[numthreads(SELF_WALK_WG, 1, 1)]
void explicit_beams_self_walk(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = selfSplitRow(gid.y);
    uint walk = v < gP.vehicles && selfSplitVehicle(v) ? uint(gAcc[selfWalkAt(v) + 1u].load()) : 0u;
    uint4 r1 = gStatic[min(v, gP.vehicles - 1u) * 4u + 1u];
    if (walk != 0u)
    {
        uint4 r0 = gStatic[v * 4u];
        uint4 r2 = gStatic[v * 4u + 2u];
        uint4 r3 = gStatic[v * 4u + 3u];
        float range = asfloat(r3.z);
        float cell = asfloat(r3.w);
        uint4 misc = gMisc[v];
        float3 origin = asfloat(misc.xyz);
        uint parity = (misc.w + gP.pad2) & 1u;
        // The dispatch has WALK_X workgroups whenever some vehicle needs more (the args'
        // clamp): then this stride covers every share.
        for (uint p = gid.x * SELF_WALK_WG + lid.x, pStop = EB_END(485, p, r1.y * WALK_SUBS, WALK_X * SELF_WALK_WG); p < pStop;
             p += WALK_X * SELF_WALK_WG)
        {
            uint t = r1.x + p / WALK_SUBS;
            if (!(piecesOn(v) && pieceVolatile(t)))
                selfWalkList(v, t, p % WALK_SUBS, origin, range, cell, r2.x, r2.y, parity, triPairsAt(v),
                             asfloat(gP.pairReuse), gP.gridOffBase + v * (GRID_BUCKETS + 1u), r0.z, r2.z, r2.w);
        }
    }
    ebLoopCapFlush();
}
