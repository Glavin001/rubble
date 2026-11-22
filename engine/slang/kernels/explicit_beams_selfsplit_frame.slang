// Self-collision split's part of the frame (EB_SELF_SPLIT; selfcol-design.md), moved out of
// explicit_beams.slang as it was, included at the same place under the same define: a
// module built without EB_SELF_SPLIT does not depend on this file (the per-compile cache
// skips an include in an #ifdef whose name the module does not define).
// perfFlags bit 131072 (explicit_plan::PERF_AHEAD_APART, with PERF_WIDE_FUSED): step
// s + 1's coupling and wheels (what the list kernel runs beside its own phases: the
// coupling's step or its fold, the wheels' rays and bars) on a workgroup of their own,
// vehicle v's at workgroup vehicles + v of the list's dispatch, in place of on the list
// workgroup's lanes. The same calls on the same lanes with the same barriers between
// them: what they read (the nodes after this step's node phase, the coupling's records
// and sums, the wheels' records, the static shapes) and write (the coupling's bodies,
// spheres and counts, the wheels' scratch records) the list's phases neither write nor
// read, so they ran beside them already; on a core of their own the list's longest
// phases no longer carry them. (Its own function behind an early return: the list's
// code is unchanged.)
// perfFlags bit 262144 (explicit_plan::PERF_TYRE_AHEAD, with PERF_AHEAD_APART): the
// tyres split at their hubs' forces (explicit_beams_drive.slang drvTyrePre,
// drvTyrePost): the first part here, the second in the hits kernel after selfApply.
void wideAheadApart(uint v, uint lane)
{
    uint s = gP.pad2;
    uint4 misc = gMisc[v];
    if (lane == 0u)
    {
        coupleBegin();
        driveBegin();
        cplGsClear();
        cplGsAdd(0u, v, 0u, 0u);
        cplFoldMark(v);
    }
    sync();
    if (s + 1u >= gP.steps || (gP.perfFlags & PERF_WIDE_FUSED) == 0u) return;
    uint coupleOn = gsUniformCouple;
    uint driveOn = gsUniformDrive;
    bool aheadFold = (coupleOn & CPL_FOLD_FRAME) != 0u;
    if (!aheadFold) coupleStep(v, lane, s + 1u, coupleOn);
    bool aheadDrive = driveOn != 0u;
    bool cBody = false;
    float4 cC0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 cQ = float4(0.0, 0.0, 0.0, 1.0);
    float4 cC2 = float4(0.0, 0.0, 0.0, 0.0);
    float3 cW = float3(0.0, 0.0, 0.0);
    DrvCtx drvC = (DrvCtx)0;
    bool drvHas = false;
    bool drvApart = false;
    if (aheadFold) cplFoldB(v, lane, LANES, s + 1u, cBody, cC0, cQ, cC2, cW);
    if (aheadDrive)
    {
        driveAheadCtx(v, LANES - 1u - lane, drvC, drvHas, drvApart);
        driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 0u, misc.w + s + 1u);
    }
    sync();
    if (aheadFold) cplFoldC(v, lane, LANES, s + 1u, cBody, cC0, cQ, cC2, cW);
    if (aheadDrive) driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 1u, misc.w + s + 1u);
    // PERF_TYRE_AHEAD: the tyres' part before their hubs' forces (after the bars), and
    // the end; the hits kernel runs the rest (driveTyresPost).
    if ((gP.perfFlags & PERF_TYRE_AHEAD) == 0u) return;
    sync();
    if (aheadDrive) driveTyresPre(drvC, drvHas, drvApart, LANES - 1u - lane);
}

// perfFlags bit 4194304 (explicit_plan::PERF_WHEELS_APART, with PERF_AHEAD_APART): the
// coupling's and the wheels' work of wideAheadApart on two workgroups, the coupling's at
// vehicles + v, the wheels' at 2 · vehicles + v. In wideAheadApart every lane runs its
// share of the coupling's phases (its nodes' counts, its shapes and bodies) and lanes
// 255..252 also their wheels', so a phase there costs the sum of the two chains on those
// lanes. The same calls, on the same lanes, between the same barriers; neither part reads
// what the other writes (the wheels read the nodes, their records and the static boxes; the
// coupling writes its bodies, spheres, lists, counts and workgroup copies), as they already
// ran beside each other.
void wideCoupleApart(uint v, uint lane)
{
    uint s = gP.pad2;
    if (lane == 0u)
    {
        coupleBegin();
        cplGsClear();
        cplGsAdd(0u, v, 0u, 0u);
        cplFoldMark(v);
    }
    sync();
    if (s + 1u >= gP.steps || (gP.perfFlags & PERF_WIDE_FUSED) == 0u) return;
    uint coupleOn = gsUniformCouple;
    bool aheadFold = (coupleOn & CPL_FOLD_FRAME) != 0u;
    if (!aheadFold) coupleStep(v, lane, s + 1u, coupleOn);
    bool cBody = false;
    float4 cC0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 cQ = float4(0.0, 0.0, 0.0, 1.0);
    float4 cC2 = float4(0.0, 0.0, 0.0, 0.0);
    float3 cW = float3(0.0, 0.0, 0.0);
    if (aheadFold) cplFoldB(v, lane, LANES, s + 1u, cBody, cC0, cQ, cC2, cW);
    sync();
    if (aheadFold) cplFoldC(v, lane, LANES, s + 1u, cBody, cC0, cQ, cC2, cW);
}
void wideWheelsApart(uint v, uint lane)
{
    uint s = gP.pad2;
    uint4 misc = gMisc[v];
    if (lane == 0u) driveBegin();
    sync();
    if (s + 1u >= gP.steps || (gP.perfFlags & PERF_WIDE_FUSED) == 0u) return;
    uint driveOn = gsUniformDrive;
    bool aheadDrive = driveOn != 0u;
    DrvCtx drvC = (DrvCtx)0;
    bool drvHas = false;
    bool drvApart = false;
    if (aheadDrive)
    {
        driveAheadCtx(v, LANES - 1u - lane, drvC, drvHas, drvApart);
        driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 0u, misc.w + s + 1u);
    }
    sync();
    if (aheadDrive) driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 1u, misc.w + s + 1u);
    if ((gP.perfFlags & PERF_TYRE_AHEAD) == 0u) return;
    sync();
    if (aheadDrive) driveTyresPre(drvC, drvHas, drvApart, LANES - 1u - lane);
}

// The reused pair list's part of a step in the wide passes (PERF_WIDE_LIST), a workgroup a
// vehicle after the step's wide_beams: what explicit_beams_self_step runs of a step
// between its beams and its pairs dispatch, and nothing else (a small kernel: a dispatch
// of the frame kernel costs four times this one's). The list's tracking, the triangles'
// throttles and tested list, the grid's counts, the decision, the grid's scan and the
// build's records on a building step, the volatile triangles' tests; the list's state
// loaded at its start and saved at its end (soloStepLoad, soloStepSave). The walk and
// pairs dispatches follow, then wide_hits (the deferred breaks, the walk's end, the
// hits). The parameter copy `steps`: the list's end (pairReuseEnd).
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_wide_list(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    if ((gP.perfFlags & PERF_AHEAD_APART) != 0u && gid.x >= gP.vehicles && gP.sleepRec != 0u)
    {
        // Vehicle sleep: a sleeping vehicle's coupling and wheels are not stepped either
        // (as on the list workgroup's lanes, below).
        uint va = gid.x - gP.vehicles;
        if (va >= gP.vehicles) va -= gP.vehicles;
        if (sleepSkip(va, lid.x)) return ebLoopCapFlush();
    }
    if ((gP.perfFlags & (PERF_AHEAD_APART | PERF_WHEELS_APART)) == (PERF_AHEAD_APART | PERF_WHEELS_APART)
        && gid.x >= gP.vehicles)
    {
        if (gid.x >= 2u * gP.vehicles) wideWheelsApart(gid.x - 2u * gP.vehicles, lid.x);
        else wideCoupleApart(gid.x - gP.vehicles, lid.x);
        return ebLoopCapFlush();
    }
    if ((gP.perfFlags & PERF_AHEAD_APART) != 0u && gid.x >= gP.vehicles)
    {
        wideAheadApart(gid.x - gP.vehicles, lid.x);
        return ebLoopCapFlush();
    }
    selfSplitArgsClear(gid.x, lid.x);
    uint v = gid.x;
    uint lane = lid.x;
    uint s = gP.pad2;
    // Vehicle sleep: a sleeping vehicle is not stepped.
    if (gP.sleepRec != 0u)
    {
        if (sleepSkip(v, lane)) return ebLoopCapFlush();
    }
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = r0.y;
    uint beamOff = r0.z;
    uint triOff = r1.x;
    uint triCount = r1.y;
    uint candOff = r2.x;
    uint candCount = r2.y;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint parity = (misc.w + s) & 1u;
    if (lane == 0u)
    {
        gsHit = 0u;
        gsTestedCount[0].store(0u);
        gsTestedCount[1].store(0u);
        gsUniformLanes = 0u;
        if (s == 0u) pairReuseBegin();
        // (The next step's coupling and wheels here: PERF_WIDE_FUSED.)
        coupleBegin();
        driveBegin();
        cplGsClear();
        cplGsAdd(0u, v, 0u, 0u);
        cplFoldMark(v);
    }
    if (s > 0u) soloStepLoad(v, lane);
    sync();
    if (s >= gP.steps)
    {
        pairReuseEnd(v, lane);
        return ebLoopCapFlush();
    }
    // Step s + 1's coupling and wheels beside this step's list phases (uniform).
    // (PERF_AHEAD_APART: on a workgroup of their own, wideAheadApart.)
    bool ahead = (gP.perfFlags & (PERF_WIDE_FUSED | PERF_AHEAD_APART)) == PERF_WIDE_FUSED && s + 1u < gP.steps;
    uint coupleOn = gsUniformCouple;
    uint driveOn = gsUniformDrive;
    bool aheadFold = ahead && (coupleOn & CPL_FOLD_FRAME) != 0u;
    // (A coupling that cannot fold: its four phases here, before the list's.)
    if (ahead && !aheadFold) coupleStep(v, lane, s + 1u, coupleOn);
    bool aheadDrive = ahead && driveOn != 0u;
    bool cBody = false;
    float4 cC0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 cQ = float4(0.0, 0.0, 0.0, 1.0);
    float4 cC2 = float4(0.0, 0.0, 0.0, 0.0);
    float3 cW = float3(0.0, 0.0, 0.0);
    DrvCtx drvC = (DrvCtx)0;
    bool drvHas = false;
    bool drvApart = false;
    if (aheadFold) cplFoldB(v, lane, LANES, s + 1u, cBody, cC0, cQ, cC2, cW);
    if (aheadDrive)
    {
        driveAheadCtx(v, LANES - 1u - lane, drvC, drvHas, drvApart);
        driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 0u, misc.w + s + 1u);
    }
    pairReuseTrack(nodeOff, nodeCount, lane);
    if (candCount > 0u)
    {
        for (uint t = lane, tStop = EB_END(101, t, triCount, LANES); t < tStop; t += LANES)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
        }
    }
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    if (lane == 0u)
    {
        selfSplitClear(v);
        gsPairCount.store(0u);
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    sync();
    if (aheadFold) cplFoldC(v, lane, LANES, s + 1u, cBody, cC0, cQ, cC2, cW);
    if (aheadDrive) driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 1u, misc.w + s + 1u);
    if (lane == 0u)
    {
        gsTestedCount[parity ^ 1u].store(0u);
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        uint grid = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u;
        gsUniformLanes = grid | (pairReuseDecide(v, grid, parity, origin) << 1);
    }
    // (The grid's bucket counts: only a step that scans the grid reads them, below.)
    uint listedB = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
    bool early = candCount > 0u && !(cell > 0.0 && listedB * candCount > BRUTE_PAIRS_PER_LANE * LANES);
    if (early)
    {
        uint lanes = clamp(LANES / max(listedB, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        uint rl = LANES - 1u - lane;
        for (uint p = rl, pStop = EB_END(102, p, listedB * lanes, stride); p < pStop && rl < stride; p += stride)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
            trianglePairs(t, triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, parity);
        }
    }
    else if (candCount > 0u && gP.triGeo != 0u && (gP.perfFlags & PERF_WIDE_GEO) == 0u)
    {
        for (uint q = LANES - 1u - lane, qStop = EB_END(103, q, listedB, LANES); q < qStop; q += LANES)
            triGeoFill(triOff + (triCount <= TRI_LIST ? gsTested[q] : q), origin, range, parity);
    }
    sync();
    uint gridOn = gsUniformLanes;
    uint reuse = gridOn >> 1;
    if (gridOn != 0u && reuse != 2u)
    {
        // The candidates counted into their buckets here, not on every step: a step that
        // reuses its list (reuse 2) or has no grid never reads the counts or the ranks.
        if (cell > 0.0)
        {
            for (uint c = lane, cStop = EB_END(104, c, candCount, LANES); c < cStop; c += LANES)
            {
                float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
                uint b = gridBucket(int3(floor(p / cell)));
                gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
            }
        }
        sync();
        uint per = GRID_BUCKETS / LANES;
        uint block = LANES / GRID_BLOCKS;
        uint sum = 0u;
        for (uint k = 0u, kStop = EB_END(105, k, per, 1u); k < kStop; k++) sum += gsCount[lane * per + k].load();
        gsPart[lane] = sum;
        sync();
        if (lane < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint k = 0u, kStop = EB_END(106, k, block, 1u); k < kStop; k++)
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
        for (uint k = 0u, kStop = EB_END(107, k, per, 1u); k < kStop; k++)
        {
            gsOff[lane * per + k] = run;
            run += gsCount[lane * per + k].load();
        }
        sync();
        if (cell > 0.0)
        {
            for (uint c = lane, cStop = EB_END(108, c, candCount, LANES); c < cStop; c += LANES)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
    uint pairs = gsPairCount.load();
    if (early && pairs <= PAIR_LIST)
    {
        for (uint i = lane, iStop = EB_END(109, i, pairs, LANES); i < iStop; i += LANES)
        {
            uint e = gsPairs[i];
            tryContact(triangleGeo(triOff + (e >> 16), origin, range, parity), gLists[candOff + (e & 0xFFFFu)], origin,
                       range, beamOff, groupOff, groupCount, parity);
        }
    }
    else if (gridOn != 0u && reuse != 0u)
    {
        if (reuse == 1u)
            selfSplitBuild(v, lane, triOff, triCount, nodeOff, nodeCount, origin, range, cell, candOff, candCount,
                           parity);
    }
    else if (candCount > 0u && cell > 0.0 && gridOn != 0u)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        uint subs = gP.triPairs != 0u ? LIST_SUBS : GRID_SUBS;
        for (uint p = lane, pStop = EB_END(110, p, listed * subs, LANES); p < pStop; p += LANES)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / subs] : p / subs;
            triangleGrid(triOff + t, p % subs, origin, range, cell, candOff, candCount, beamOff,
                         groupOff, groupCount, parity, gP.triPairs != 0u ? triPairsAt(v) : 0u);
        }
    }
    else if (candCount > 0u && cell > 0.0)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        uint lanes = clamp(LANES / max(listed, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        for (uint p = lane, pStop = EB_END(111, p, listed * lanes, stride); p < pStop && lane < stride; p += stride)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
            triangleGridBrute(triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, beamOff,
                              groupOff, groupCount, parity);
        }
    }
    else if (candCount > 0u)
    {
        uint lanes = clamp(LANES / max(triCount, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        for (uint p = lane, pStop = EB_END(112, p, triCount * lanes, stride); p < pStop && lane < stride; p += stride)
            triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff,
                         groupOff, groupCount, parity);
    }
    if (gridOn != 0u && reuse != 0u)
        selfSplitTests(v, lane, reuse, triOff, triCount, origin, range, cell, candOff, candCount, beamOff, groupOff,
                       groupCount, parity);
    else if (gridOn != 0u && gP.triPairs != 0u)
    {
        sync();
        uint listedPairs = gsPairCount.load();
        if (listedPairs <= TRI_PAIR_CAP)
        {
            uint at = triPairsAt(v);
            for (uint i = lane, iStop = EB_END(113, i, listedPairs, LANES); i < iStop; i += LANES)
                tryContact(triGeoLoad(gNext[at + 2u * i]), gNext[at + 2u * i + 1u], origin, range, beamOff,
                           groupOff, groupCount, parity);
        }
        else
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            for (uint p = lane, pStop = EB_END(114, p, listed * GRID_SUBS, LANES); p < pStop; p += LANES)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
                triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff,
                             groupOff, groupCount, parity, 0u);
            }
        }
    }
    sync();
    soloStepSave(v, lane);
    ebLoopCapFlush();
}

// Surface contact on the self split paths, the surface pass as its own pipelines
// (surfPiped; explicit_beams_surface_pipes.slang): a solo vehicle's step s (gP.pad2)
// after its walk and pairs dispatches and before the pipelines: the walk's list recorded
// and the step's self-collision hits applied, as the next step dispatch's start runs them
// without surface contact (explicitBeamsFrameStep's rest, which then skips them). The
// same calls on the same state: the hit flag and the pair list's state loaded as that
// dispatch loads them and saved as the step's end saves them (soloStepSave: the flag in
// the vehicle's segment record, where the pipelines' prep and the next step read it).
// A workgroup a vehicle, as the step's (its awake-indexed pick, its sleep rule); with
// vehicle contact a group's member returns at once (its hits: contactMemberStep or
// memberRest).
#ifdef EB_SOLO_STEP
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_solo_rest(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint v = gid.x;
    bool sleepListed = gP.sleepRec != 0u && gP.sleepPad0 != 0u;
    if (sleepListed) v = sleepPick(gid.x, lane);
    if (v >= gP.vehicles || gP.pad2 >= gP.steps) return ebLoopCapFlush();
#ifdef EB_VV_SOLO
    if (lane == 0u) gsUniformLanes = gMisc[vvRec(v, VV_FLAGS)].y;
    sync();
    if (gsUniformLanes != 0u) return ebLoopCapFlush();
#endif
    if (gP.sleepRec != 0u && !sleepListed)
    {
        if (sleepSkip(v, lane)) return ebLoopCapFlush();
    }
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint parity = (misc.w + gP.pad2) & 1u;
    if (lane == 0u)
    {
        gsHit = gMisc[segRec(v)].x;
        gsTestedCount[0].store(0u);
        gsTestedCount[1].store(0u);
        gsUniformLanes = r1.y > 0u ? 1u : 0u;
    }
    soloStepLoad(v, lane);
    sync();
    uint selfOn = gsUniformLanes;
    if (lane == 0u)
    {
        selfSplitHits(v);
        if (selfHitsLocalOn(v)) gsUniformHits = 2u;
    }
    if (selfOn != 0u)
    {
        sync();
        uint hitsOn = gsUniformHits;
        if (hitsOn == 2u)
            selfApplyLocalKeep(v, lane, r0.z, r2.z, r2.w, parity);
        else
            selfApply(v, true, lane, LANES, lane, origin, asfloat(r3.z), r0.z, r2.z, r2.w, parity);
    }
    sync();
    soloStepSave(v, lane);
    ebLoopCapFlush();
}
#endif
