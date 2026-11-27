// --- The contact groups: explicit_beams_contact ---

// Group member m (sorted by vehicle index) and its partner / reset flags, its
// contact points' first run index; per group at its member offset in regions of
// one word per vehicle.
uint vvMember(uint off, uint m) { return gNext[gP.vvMembers + gP.vehicles + off + m]; }
uint vvPrefixAt(uint off, uint m) { return gP.vvMembers + 2u * gP.vehicles + off + m; }
uint vvPartnerAt(uint off, uint m) { return gP.vvMembers + 3u * gP.vehicles + off + m; }
uint vvResetAt(uint off, uint m) { return gP.vvMembers + 4u * gP.vehicles + off + m; }
uint vvTestedAt(uint i) { return gP.vvMembers + 5u * gP.vehicles + i; }

// A member's step: explicit_beams_frame's step for vehicle v (every barrier at the
// top level), with the self-collision accumulators always read (the contact pass
// adds into them), the tested-triangle list restarted per member, and a pending
// reset of the self-collision throttles (a fast approach in the last contact pass)
// applied after their update.
// A contact group member's box of this step (world positions after integration) into
// this parity's keys, and the frame bounds check: `lanes` lanes from `lane`, each over
// its own nodes (no lane: nothing). Its own pass after the node phase, each lane
// reading back the positions it wrote. The box was once kept as the node loop's
// running min and max: Chrome's Metal compile (Tint's bounds-checked MSL) returned a
// wrong box from that loop for the lanes past the last full stride (every member's
// local node 208 of the page's 463-node car, 256 lanes a member: the position of
// global node 0), so vehicles reported frame-bounds errors native never shows. With
// centred damping (§39) it runs again after the correction (a superset of the box).
float contactOffMax(uint v);
void memberBox(uint v, uint nodeOff, uint nodeCount, float3 origin, uint parity, uint lane, uint lanes)
{
    float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 hi = -lo;
    bool has = false;
    for (uint i = lane, iStop = EB_END(138, i, nodeCount, lanes); i < iStop; i += lanes)
    {
        float3 x = nf3(origin + gNodes[(nodeOff + i) * 3u].xyz);
        lo = min(lo, x);
        hi = max(hi, x);
        has = true;
    }
    if (!has) return;
    // (Contact offsets: the triangles reach out past the nodes by up to the largest.)
    float om = contactOffMax(v);
    lo = lo - float3(om, om, om);
    hi = hi + float3(om, om, om);
    uint key = vvBoxKey(v, parity);
    gAcc[key].min(vvKey(lo.x));
    gAcc[key + 1u].min(vvKey(lo.y));
    gAcc[key + 2u].min(vvKey(lo.z));
    gAcc[key + 3u].max(vvKey(hi.x));
    gAcc[key + 4u].max(vvKey(hi.y));
    gAcc[key + 5u].max(vvKey(hi.z));
    if (any(lo < asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz)) || any(hi > asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz)))
        vvReportOutside(v);
}

// A member's pair list over the frame's steps (PERF_MEMBER_LIST; the split group path's
// members, a workgroup each and a dispatch a step): the solo step's reused pair list
// (pairReuse*), its state (gsCount past the buckets) in the member's solo-step record
// between the steps' dispatches (soloStepAt: the member's own grid count words, which
// vehicle contact's scratch no longer uses), its pairs in the member's own pair region
// (triPairsAt) and its build records (gBeams at pairBuild): nothing else in a group's
// frame writes them. The tests are the walk's, bit for bit, by the list's own argument
// (above pairReuseBegin): it reads only the member's node state of the step and its
// triangles' throttles of the step (a pending reset zeroes them before the tests, in
// gMisc, where the listed pairs' tests read them).
#ifdef EB_SOLO_STEP
void memberListLoad(uint v, uint lane, uint s)
{
    if (s == 0u)
    {
        if (lane == 0u) pairReuseBegin();
    }
    else
        soloStepLoad(v, lane);
}
// The state for the member's next dispatch.
void memberListKeep(uint v, uint lane)
{
    for (uint k = lane; k < REUSE_WORDS; k += LANES) gAcc[soloStepAt(v) + k].store(int(gsCount[GRID_BUCKETS + k].load()));
}
// After the step's closing barrier: the state for the next step's dispatch, or at the
// frame's last step the list's statistics (pairReuseEnd).
void memberListSave(uint v, uint lane, uint s)
{
    if (s + 1u < gP.steps)
        memberListKeep(v, lane);
    else
        pairReuseEnd(v, lane);
}
// A grid step with the list (uniform; mode 1 build, 2 reuse): the solo step's build and
// tests in the member's workgroup.
void memberListTests(uint v, uint lane, uint mode, uint triOff, uint triCount, uint nodeOff, uint nodeCount,
                     float3 origin, float r, float cell, uint candOff, uint candCount, uint beamOff, uint groupOff,
                     uint groupCount, uint parity)
{
    if (mode == 1u) pairReuseBuild(v, lane, triOff, triCount, nodeOff, nodeCount, origin, r, cell, candOff, candCount, parity);
    pairReuseTests(v, lane, mode, triOff, triCount, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount,
                   parity);
}
#endif

#include "explicit_beams_member_tail.slang"

// Member v's step s up to its tail (memberTail, which t and the MT_* bits returned
// describe; 0: no tail here).
// `phase` (continuous contact's seam, docs/continuous-contact-design.md): 0 the whole step;
// 1 up to the node update and its box (returns 0: no tail, not stepped); 2 the rest after
// them (the node update not run again).
uint contactMemberStepHead(uint v, uint lane, uint s, bool first, out MemberTail t)
{
    return contactMemberStepHeadPhase(v, lane, s, first, 0u, t);
}

uint contactMemberStepHeadPhase(uint v, uint lane, uint s, bool first, uint phase, out MemberTail t)
{
    t.v = v;
    t.s = s;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = r0.y;
    uint beamOff = r0.z;
    uint beamCount = r0.w;
    uint triOff = r1.x;
    uint triCount = r1.y;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = noPositionSteps(v) ? 0u : r2.y;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = selfRangeOf(v, asfloat(r3.z));
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
#endif
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint step = misc.w + s;
    uint parity = step & 1u;
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
#ifdef EB_SOLO_STEP
    // The pair list kept over the frame's steps (PERF_MEMBER_LIST): none at the frame's
    // first step, else its state as the member's last step left it.
    bool keep = (gP.perfFlags & PERF_MEMBER_LIST) != 0u && pairReuseOn();
    if (keep) memberListLoad(v, lane, s);
#else
    bool keep = false;
#endif
    if (lane == 0u)
    {
        gsTestedCount[parity].store(0u);
#ifndef EB_SOLO_STEP
        // (The split path's members' module: the groups' pass clears them, in its own
        // dispatch; Metal binds at most 31 workgroup variables a kernel.)
        if (first) vvClearCounters();
#endif
    }
    sync();
    // Continuous contact: the step's start positions (relative to the origin), for its pass.
    if (phase == 1u && gP.cc != 0u)
    {
        for (uint i = lane, iStop = EB_END(892, i, nodeCount, LANES); i < iStop; i += LANES)
            gBeams[gStatic[gP.cc].x + 6u * (nodeOff + i)] = gNodes[(nodeOff + i) * 3u];
        // (The contact offsets at the step's start, which the pass reads: every step's
        // member computes them at its end; before the run's first step, here.)
        // (Uniform for Chrome: the step count is read from storage.)
        sync();
        if (lane == 0u) gsUniformLanes = gP.offsetBase != 0u && misc.w + s == 0u ? 1u : 0u;
        sync();
        uint first = gsUniformLanes;
        sync();
        if (first != 0u)
        {
            contactOffsets(v, lane, LANES);
            sync();
        }
    }
    if (phase != 2u)
    {
        bool centredV = gP.centredDv != 0u && (flagsV & FLAG_CENTRED) != 0u;
        for (uint i = lane, iStop = EB_END(139, i, nodeCount, LANES); i < iStop; i += LANES)
        {
            if (centredV && (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u) centredKeep(nodeOff + i);
            float3 v0 = centredV ? gNodes[(nodeOff + i) * 3u + 1u].xyz : float3(0.0, 0.0, 0.0);
            nodeUpdate(nodeOff + i, gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground,
                       boxOff, boxCount, beamOff, groupOff, groupCount, true);
            if (centredV) centredDelta(nodeOff + i, v0);
        }
        // The contact pass's box of this step (the other parity's keys cleared for the
        // next step), and the frame bounds check.
        if (lane < 6u) gAcc[vvBoxKey(v, parity ^ 1u) + lane].store(lane < 3u ? VV_INT_MAX : VV_INT_MIN);
        memberBox(v, nodeOff, nodeCount, origin, parity, lane, LANES);
    }
    if (gP.centredDv != 0u && phase != 2u)
    {
        centredPhase(flagsV, nodeOff, nodeCount, beamOff, beamCount, groupOff, groupCount, step, lane, LANES);
        if ((flagsV & FLAG_CENTRED) != 0u) memberBox(v, nodeOff, nodeCount, origin, parity, lane, LANES);
    }
    if (phase == 1u) return 0u;
#ifdef EB_SOLO_STEP
    if (keep) pairReuseTrack(nodeOff, nodeCount, lane);
#endif
    for (uint g = lane, gStop = EB_END(140, g, groupCount, LANES); g < gStop; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = lane, tStop = EB_END(141, t, triCount, LANES); t < tStop; t += LANES)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (s > 0u && pending)
            {
                w = 0;
                gMisc[gP.vehicles + triOff + t].x = 0u;
            }
            if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
        }
    }
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    if (lane == 0u)
    {
        if (keep) gsPairCount.store(0u);
#ifdef EB_SELF_SPLIT
        // (No pairs for the pairs dispatch, no walk, unless this step lists them.)
        if (keep) selfSplitClear(v);
#endif
        gAcc[hitAcc(v)].store(0);
        gAcc[hitAcc(v) + 1u].store(0);
        clipVmaxClear(v, parity);
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    sync();
    if (lane == 0u) grabApply(nodeOff, nodeCount, origin);
#ifdef EB_AERO
    if (lane == 0u) brakeApply(aero2.z, aero2.w);
#endif
    if (lane == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    if (lane == 0u)
    {
        // Bit 0: the grid this step; bit 1: self-collision hits or pending beams
        // possible (selfApply's barriers).
        // Bits 2..3 (PERF_MEMBER_LIST): the grid's pair list this step (pairReuseDecide:
        // 0 the walk, 1 build, 2 reuse).
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        uint grid = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u;
        uint mode = 0u;
#ifdef EB_SOLO_STEP
        if (keep) mode = pairReuseDecide(v, grid, parity, origin);
#endif
        gsUniformLanes = grid | (triCount > 0u ? 2u : 0u) | (mode << 2);
    }
    if (cell > 0.0)
    {
        for (uint c = lane, cStop = EB_END(142, c, candCount, LANES); c < cStop; c += LANES)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = lane, wStop = EB_END(143, w, aero0.y, LANES); w < wStop; w += LANES) wingForces(aero0.x + w, origin);
    if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = lane, cStop = EB_END(144, c, aero0.w, LANES); c < cStop; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = lane, cStop = EB_END(145, c, aero0.w, LANES); c < cStop; c += LANES) commandUpdate(aero0.z + c);
    }
    sync();
    if (lane == 0u) wingApply(aero0.x, aero0.y);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
    }
#endif
    for (uint j = lane, jStop = EB_END(146, j, beamCount, LANES); j < jStop; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    // The grid this step (lane 0's decision above, from the same counts): each tested
    // triangle's geometry of this step for its walk, which loads it (triangleGridWalk,
    // triGeoLoad). Without this fill the walk read whatever the last solo or wide frame
    // left there: stale triangles for every member of a larger group.
    if (gP.triGeo != 0u && cell > 0.0 && candCount > 0u)
    {
        uint listedG = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        if (listedG * candCount > BRUTE_PAIRS_PER_LANE * LANES)
        {
            for (uint q = LANES - 1u - lane, qStop = EB_END(147, q, listedG, LANES); q < qStop; q += LANES)
                triGeoFill(triOff + (triCount <= TRI_LIST ? gsTested[q] : q), origin, range, parity);
        }
    }
    sync();
    uint flagsU = gsUniformLanes;
    uint gridOn = flagsU & 1u;
#ifdef EB_SOLO_STEP
    uint reuse = flagsU >> 2;
#else
    const uint reuse = 0u;
#endif
    if (lane == 0u) resolvePending(v, parity);
    if (gridOn != 0u && reuse != 2u)
    {
        uint per = GRID_BUCKETS / LANES;
        uint block = LANES / GRID_BLOCKS;
        uint sum = 0u;
        for (uint k = 0u, kStop = EB_END(148, k, per, 1u); k < kStop; k++) sum += gsCount[lane * per + k].load();
        gsPart[lane] = sum;
        sync();
        if (lane < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint k = 0u, kStop = EB_END(149, k, block, 1u); k < kStop; k++)
            {
                uint x = gsPart[lane * block + k];
                gsPart[lane * block + k] = run;
                run += x;
            }
            gsBlk[lane] = run;
        }
        sync();
        if (lane == 0u)
        {
            uint run = 0u;
            for (uint k = 0u; k < GRID_BLOCKS; k++)
            {
                uint x = gsBlk[k];
                gsBlk[k] = run;
                run += x;
            }
            gsOff[GRID_BUCKETS] = run;
        }
        sync();
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint k = 0u, kStop = EB_END(150, k, per, 1u); k < kStop; k++)
        {
            gsOff[lane * per + k] = run;
            run += gsCount[lane * per + k].load();
        }
        sync();
        if (cell > 0.0)
        {
            for (uint c = lane, cStop = EB_END(151, c, candCount, LANES); c < cStop; c += LANES)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
    // One of four tests a step (the walk, its own or past the self split's list; the pieces;
    // the brute pairs; every candidate), from one loop of triangleCandidates (one tryContact).
    bool walk = candCount > 0u && cell > 0.0 && gridOn != 0u;
    bool listedTests = false;
    bool pieces = false;
#ifdef EB_SOLO_STEP
    if (gridOn != 0u && reuse != 0u)
    {
        listedTests = true;
#ifdef EB_SELF_SPLIT
        // The solo step's cut: the build's walk and the listed pairs' tests in the walk
        // and pairs dispatches after this one.
        if (reuse == 1u)
            selfSplitBuild(v, lane, triOff, triCount, nodeOff, nodeCount, origin, range, cell, candOff, candCount, parity);
        uint how = selfSplitTestsHow(v, lane, reuse);
        walk = how == 1u;
        pieces = how == 2u;
#else
        memberListTests(v, lane, reuse, triOff, triCount, nodeOff, nodeCount, origin, range, cell, candOff, candCount,
                        beamOff, groupOff, groupCount, parity);
        walk = false;
#endif
    }
#endif
    // (cmode: 3 the walk, 2 the pieces', 1 the brute pairs', 0 every candidate's.)
    uint cmode = 4u, cN = 0u, ck = 1u, cstride = LANES;
    if (walk)
    {
        cN = (triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount) * GRID_SUBS;
        ck = GRID_SUBS;
        cmode = 3u;
    }
    else if (pieces)
    {
        cN = piecesBegin(v, lane, ck);
        cN *= ck;
        cmode = 2u;
    }
    else if (!walk && !listedTests && candCount > 0u)
    {
        bool brute = cell > 0.0;
        uint listed = brute && triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        ck = clamp(LANES / max(listed, 1u), 1u, candCount);
        cstride = (LANES / ck) * ck;
        cN = listed * ck;
        cmode = brute ? 1u : 0u;
    }
    for (uint p = lane, pStop = EB_END(153, p, cN, cstride); p < pStop && lane < cstride; p += cstride)
    {
        uint q = p / ck;
        uint gt = cmode == 2u ? gNext[pieceVolAt(triOff + q)] : triOff + (cmode % 2u == 1u && triCount <= TRI_LIST ? gsTested[q] : q);
        triangleCandidates(gt, p % ck, ck, origin, range, cell, candOff, candCount, beamOff, groupOff, groupCount, parity,
                           cmode);
    }
    if (pieces && reuse == 1u && lane == 0u && gP.pieceBase != 0u) piecesClearBuild();
#ifdef EB_SELF_SPLIT
    if (keep)
    {
        // The step's rest (its hits applied, the slide nodes) after the walk and pairs
        // dispatches: memberRest.
        sync();
        memberListKeep(v, lane);
        return 0u;
    }
#endif
    t.origin = origin;
    t.range = range;
    t.beamOff = beamOff;
    t.groupOff = groupOff;
    t.groupCount = groupCount;
    t.parity = parity;
    t.flagsV = flagsV;
    return MT_TAIL | ((flagsU & 2u) != 0u ? MT_SYNC | MT_APPLY : 0u) | (keep ? MT_SAVE : 0u);
}
void contactMemberStep(uint v, uint lane, uint s, bool first)
{
    MemberTail t;
    uint how = contactMemberStepHead(v, lane, s, first, t);
    if ((how & MT_TAIL) != 0u) memberTail(t, how, lane);
}

