#ifdef EB_SOLO_STEP
// The split group path's solo vehicles (explicit_beams_split.slang): step gP.pad2 of the
// frame alone, a function of the members' entry (the frame's start in step 0, its end in
// the last; the state the frame keeps across steps in workgroup memory saved between).
void explicitBeamsFrameStep(uint3 gid, uint3 lid)
#else
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_frame(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
#endif
{
    uint v = gid.x;
    uint lane = lid.x;
#ifdef EB_ABLATE
    // Measurement only (time): 4096 the dispatch alone.
    if (EB_A2(4096u)) return ebLoopCapFlush();
#endif
    // Vehicle sleep, awake-indexed: the workgroup's awake vehicle (or contact group).
    bool sleepListed = gP.sleepRec != 0u && gP.sleepPad0 != 0u;
#if defined(EB_VV_SOLO) && defined(EB_VV_GROUPS) && !defined(EB_VV_SPLIT) && !defined(EB_SOLO_STEP)
    // The contact groups' module: the workgroups past the vehicles step the groups
    // (vvArgs' words 4..6, the same as with the solo frame here). Every vehicle is a
    // group's member, one without partners a group of its own, so the workgroups of the
    // vehicles have nothing to do; the vehicles of a group past VV_MAX_GROUP members run
    // alone in the solo module's dispatch (words 0..2). So this module holds no copy of
    // the solo frame (its first-load compile: kernel-compile-time.md).
    if (sleepListed) v = sleepPick(gid.x, lane);
    if (v >= gP.vehicles) contactGroup(v - gP.vehicles, lane);
#else
#ifdef EB_SOLO_STEP
    // (A step a dispatch: the entry picked the workgroup's vehicle, sleepPick, and gave
    // it as gid.x; past the awake vehicles, the one workgroup a dispatch keeps when none
    // is awake: nothing to step.)
    if (sleepListed && v >= gP.vehicles) return ebLoopCapFlush();
#else
    if (sleepListed) v = sleepPick(gid.x, lane);
#endif
#ifdef EB_VV_SOLO
#ifdef EB_VV_GROUPS
    // Workgroups past the vehicles step the contact groups (one dispatch: groups
    // and solo vehicles run side by side).
    if (v >= gP.vehicles)
    {
        contactGroup(v - gP.vehicles, lane);
        return ebLoopCapFlush();
    }
#endif
    // A vehicle of a contact group this frame is stepped by its group's workgroup.
    if (lane == 0u) gsUniformLanes = gMisc[vvRec(v, VV_FLAGS)].y;
    sync();
    if (gsUniformLanes != 0u) return ebLoopCapFlush();
#endif
    // Vehicle sleep: a sleeping vehicle is not stepped.
    bool sleepOn = gP.sleepRec != 0u;
    if (sleepOn && !sleepListed)
    {
        if (sleepSkip(v, lane)) return ebLoopCapFlush();
    }
    bool sleepMove = false;
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
    uint candCount = r2.y;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
    // The wing break check, once per frame on the positions before the rebase.
#if defined(EB_FRAME_SEG) || defined(EB_SOLO_STEP)
    if (gP.pad2 == 0u)
#endif
    {
        float3 o = asfloat(gMisc[v].xyz);
        for (uint w = lane, wStop = EB_END(61, w, aero0.y, LANES); w < wStop; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif

    // Origin rebase: every lane reads node 0 before any lane moves it.
    float3 first = gNodes[nodeOff * 3u].xyz;
    sync();
#if defined(EB_FRAME_SEG) || defined(EB_SOLO_STEP)
    // A frame in segments (EB_FRAME_SEG, EB_SOLO_STEP): the rebase in the first only.
    if (gP.pad2 == 0u && length(first) > REBASE_RADIUS)
#else
    if (length(first) > REBASE_RADIUS)
#endif
    {
        for (uint i = lane, iStop = EB_END(62, i, nodeCount, LANES); i < iStop; i += LANES)
        {
            float4 P = gNodes[(nodeOff + i) * 3u];
            gNodes[(nodeOff + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
        cplRebase(v, lane, first);
    }
    sync();
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint stepBase = misc.w;
    if (lane == 0u)
    {
#if defined(EB_FRAME_SEG) || defined(EB_SOLO_STEP)
        // A later segment: the hit flag the last one and the surface pipelines left.
        gsHit = gP.pad2 == 0u ? 1u : gMisc[segRec(v)].x;
#else
        gsHit = 1u;
#endif
        gsTestedCount[0].store(0u);
        gsTestedCount[1].store(0u);
        // Self-collision hits or pending beams possible (selfApply's barriers).
        gsUniformLanes = triCount > 0u ? 1u : 0u;
#ifdef EB_SOLO_STEP
        if (gP.pad2 == 0u) pairReuseBegin();
#else
        pairReuseBegin();
#endif
        coupleBegin();
        driveBegin();
        // The coupling's spheres in workgroup memory (cplGsRange).
        cplGsClear();
        cplGsAdd(0u, v, 0u, 0u);
#ifdef EB_SOLO_STEP
        // A step a dispatch: the coupling list's frame maxima (in storage) cleared at the
        // frame's start only, as the whole frame clears them (cplListFirst: step 0).
        if (gP.pad2 == 0u) cplListBegin(v);
#else
        cplListBegin(v);
#endif
        // The fold's uniform mark lives in workgroup memory: set at every dispatch.
        cplFoldMark(v);
    }
#ifdef EB_SOLO_STEP
    // A later step: the pair list's state as the last step left it (soloStepSave).
    if (gP.pad2 > 0u) soloStepLoad(v, lane);
#endif
    // The beam phase's sorted entries' header (beamPhaseSorted).
    uint4 beamHead = gP.beamOrder != 0u ? gStatic[gP.beamOrder + v] : uint4(0u, 0u, 0u, 0u);
    // Static records the lane reads every step, kept in registers: its first node's.
    uint4 myInfo = gStatic[gP.nodeInfoBase + nodeOff + min(lane, max(nodeCount, 1u) - 1u)];
#ifdef EB_VV_SOLO
    if (lane == 0u)
    {
        // Centre and half extents (relative).
        float3 lo = asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz) - origin;
        float3 hi = asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz) - origin;
        gsVvBounds[0] = 0.5 * (lo + hi);
        gsVvBounds[1] = 0.5 * (hi - lo);
    }
#endif

    sync();
#ifdef EB_ABLATE
    // Measurement only (time): 8192 the dispatch and the frame's start alone.
    if (EB_A2(8192u)) return ebLoopCapFlush();
#endif
    uint selfOn = gsUniformLanes;
    uint coupleOn = gsUniformCouple;
    uint driveOn = gsUniformDrive;
    uint surfOn = surfaceOn();
    // The wheels' parts beside other phases (driveAheadPart; uniform).
    bool driveAhead = (gP.perfFlags & PERF_DRIVE_AHEAD) != 0u && driveOn != 0u;
    // The lane's drive context for the next step's parts (driveAheadCtx), kept from its
    // rays to its tyres within one invocation.
    DrvCtx drvC = (DrvCtx)0;
    bool drvHas = false;
    bool drvApart = false;
    bool drvCarried = false;

#ifdef EB_SELF_SPLIT
    // Step k - 1's rest (k = gP.pad2), after its tests' dispatches: its hits applied,
    // its surface pass, its slide nodes and its closing barrier, as the frame runs them.
    if (gP.pad2 > 0u)
    {
        uint parityR = (stepBase + gP.pad2 - 1u) & 1u;
        if (lane == 0u && !surfPiped())
        {
            selfSplitHits(v);
            // (PERF_SELF_HITS_LOCAL: 2, the hits from workgroup memory; uniform after the
            // barrier below.)
            if (selfHitsLocalOn(v)) gsUniformHits = 2u;
        }
#ifdef EB_ABLATE
        if (selfOn != 0u && !surfPiped() && !EB_A2(64u))
#else
        if (selfOn != 0u && !surfPiped())
#endif
        {
            sync();
            uint hitsOn = gsUniformHits;
            if (hitsOn == 2u)
                selfApplyLocalKeep(v, lane, beamOff, groupOff, groupCount, parityR);
            else
                selfApply(v, true, lane, LANES, lane, origin, range, beamOff, groupOff, groupCount, parityR);
        }
        surfaceStep(v, lane, surfOn, parityR);
#ifdef EB_AERO
        if ((flagsV & FLAG_SLIDES) != 0u)
        {
            uint count = min(aero2.y, gP.slideTotal - min(aero2.x, gP.slideTotal));
            for (uint k = lane, kStop = EB_END(63, k, count, LANES); k < kStop; k += LANES) slideNode(aero2.x + k, SLIDE_FRAME, groupOff, parityR);
        }
#endif
        sync();
    }
#endif
#ifdef EB_FRAME_SEG
    // Segment k (gP.pad2) of a frame whose surface pass runs as its own pipelines
    // between segments: step k-1's rest after its surface pass (the slide nodes and the
    // step's closing barrier), then step k up to its surface pass. The last segment
    // (k = steps) runs the frame's end.
    if (gP.pad2 > 0u)
    {
#ifdef EB_AERO
        if ((flagsV & FLAG_SLIDES) != 0u)
        {
            uint parity = (stepBase + gP.pad2 - 1u) & 1u;
            uint count = min(aero2.y, gP.slideTotal - min(aero2.x, gP.slideTotal));
            for (uint k = lane, kStop = EB_END(64, k, count, LANES); k < kStop; k += LANES) slideNode(aero2.x + k, origin, groupOff, parity);
        }
#endif
        sync();
    }
    for (uint s = gP.pad2, sStop = EB_END_UNIFORM(65, s, min(gP.pad2 + 1u, gP.steps), 1u, 1u); s < sStop; s++)
#elif defined(EB_SOLO_STEP)
    for (uint s = gP.pad2, sStop = EB_END_UNIFORM(66, s, min(gP.pad2 + 1u, gP.steps), 1u, 1u); s < sStop; s++)
#else
    for (uint s = 0u, sStop = EB_END_UNIFORM(67, s, gP.steps, 1u, EB_MAX_STEPS); s < sStop; s++)
#endif
    {
#ifdef EB_ABLATE
        if (EB_A2(1024u))
        {
            sync();
            continue;
        }
        if (!EB_A2(512u))
#endif
        {
            if (driveAhead)
            {
                // The frame's first step: its rays and bars here (later steps': beside
                // the step before's beam phase).
                if (!drvCarried) driveAheadCtx(v, LANES - 1u - lane, drvC, drvHas, drvApart);
                if (s == 0u)
                {
                    driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 0u, stepBase + s);
                    sync();
                    driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 1u, stepBase + s);
                    sync();
                }
                driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 2u, stepBase + s);
                if (lane == 0u) gsHit = 1u;
                if (coupleOn == 0u) sync();
            }
            coupleStep(v, lane, s, coupleOn);
        }
#ifdef EB_ABLATE
        if (!EB_A2(1u))
#endif
        if (!driveAhead) driveStep(v, lane, s, driveOn);
        uint step = stepBase + s;
        uint parity = step & 1u;
        bool hits = gsHit != 0u;
#ifdef EB_VV_SOLO
        bool centredV = gP.centredDv != 0u && (flagsV & FLAG_CENTRED) != 0u;
        for (uint i = lane, iStop = EB_END(68, i, nodeCount, LANES); i < iStop; i += LANES)
        {
            if (centredV && (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u) centredKeep(nodeOff + i);
            float3 v0 = centredV ? gNodes[(nodeOff + i) * 3u + 1u].xyz : float3(0.0, 0.0, 0.0);
            nodeUpdateSolo(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
            if (centredV) centredDelta(nodeOff + i, v0);
        }
#else
#ifdef EB_ABLATE
        uint nodeEnd = (gP.ablate & 2u) != 0u ? 0u : ((gP.ablate & 4u) != 0u ? min(nodeCount, LANES) : nodeCount);
#else
        uint nodeEnd = nodeCount;
#endif
        bool centredV = gP.centredDv != 0u && (flagsV & FLAG_CENTRED) != 0u;
        for (uint i = lane, iStop = EB_END(69, i, nodeEnd, LANES); i < iStop; i += LANES)
        {
            if (centredV && (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u) centredKeep(nodeOff + i);
            float3 v0 = centredV ? gNodes[(nodeOff + i) * 3u + 1u].xyz : float3(0.0, 0.0, 0.0);
            nodeUpdate(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
            if (centredV) centredDelta(nodeOff + i, v0);
        }
#endif
        if (gP.centredDv != 0u) centredPhase(flagsV, nodeOff, nodeCount, beamOff, beamCount, groupOff, groupCount, step, lane, LANES);
        if (sleepOn) sleepMove = sleepTrack(nodeOff, nodeCount, lane, sleepMove);
#ifdef EB_ABLATE
        if (!EB_A2(8u))
#endif
        pairReuseTrack(nodeOff, nodeCount, lane);

#ifdef EB_ABLATE
        if (!EB_A2(256u))
#endif
        for (uint g = lane, gStop = EB_END(70, g, groupCount, LANES); g < gStop; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
#ifdef EB_ABLATE
        if (candCount > 0u && !EB_A2(2u))
#else
        if (candCount > 0u)
#endif
        {
            for (uint t = lane, tStop = EB_END(71, t, triCount, LANES); t < tStop; t += LANES)
            {
                int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
                if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
            }
        }
#ifdef EB_ABLATE
        if (!EB_A2(4u))
#endif
        for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
        if (lane == 0u)
        {
#ifdef EB_SELF_SPLIT
            selfSplitClear(v);
#endif
            gsPairCount.store(0u);
            gAcc[hitAcc(v)].store(0);
            gAcc[hitAcc(v) + 1u].store(0);
            clipVmaxClear(v, parity);
            gsUniformHits = 0u;
            gsUniformOpen = 0u;
        }
        sync();
#ifdef EB_ABLATE
        if (!EB_A2(2048u))
#endif
        if (lane == 0u) grabApply(nodeOff, nodeCount, origin);
#ifdef EB_AERO
#ifdef EB_ABLATE
        if (!EB_A2(2048u))
#endif
        if (lane == 0u) brakeApply(aero2.z, aero2.w);
#endif
        if (lane == 0u)
        {
            gsHit = 0u;
            gsTestedCount[parity ^ 1u].store(0u);
            // This step's self-collision: the grid, or (few pairs) every candidate.
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            uint grid = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u;
            // With the grid, its pair list built, reused or not used (pairReuseDecide).
            gsUniformLanes = grid | (pairReuseDecide(v, grid, parity, origin) << 1);
        }
#ifdef EB_ABLATE
        if (cell > 0.0 && !EB_A2(4u))
#else
        if (cell > 0.0)
#endif
        {
            for (uint c = lane, cStop = EB_END(72, c, candCount, LANES); c < cStop; c += LANES)
            {
                float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
                uint b = gridBucket(int3(floor(p / cell)));
                gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
            }
        }
#ifdef EB_AERO
#ifdef EB_ABLATE
        if ((gP.ablate & 16u) == 0u) {
#endif
        // Wings (forces from this step's integrated state), the fuselage drag for
        // the next step, and the commands' rest lengths; then one lane adds the
        // wings' forces in file order while the others start the beams.
        for (uint w = lane, wStop = EB_END(73, w, aero0.y, LANES); w < wStop; w += LANES) wingForces(aero0.x + w, origin);
        if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
        if ((flagsV & FLAG_KEYS) != 0u)
        {
            for (uint c = lane, cStop = EB_END(74, c, aero0.w, LANES); c < cStop; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
        }
        else
        {
            for (uint c = lane, cStop = EB_END(75, c, aero0.w, LANES); c < cStop; c += LANES) commandUpdate(aero0.z + c);
        }
        sync();
        {
            uint4 ext = gStatic[gP.vehExtBase + v];
#ifdef EB_ABLATE
            if ((gP.ablate & 8u) != 0u) ext.w = 0u;
#endif
            for (uint k = lane, kStop = EB_END(76, k, ext.w, LANES); k < kStop; k += LANES) wingGather(ext.z + k);
        }
        // The last step's trigger inputs, read by this step's commands: cleared for
        // the next step's beam phase.
        if ((flagsV & FLAG_KEYS) != 0u)
        {
            for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
        }
#ifdef EB_ABLATE
        }
#endif
#endif
#ifdef EB_ABLATE
        uint beamEnd = (gP.ablate & 32u) != 0u ? 0u : beamCount;
#else
        uint beamEnd = beamCount;
#endif
        // (Sorted records or, gP.beamOrder 0, file order: one call site of beamUpdate.)
        beamPhaseSorted(v, lane, LANES, beamHead, beamEnd != 0u, nodeOff, beamOff, fast, groupOff, parity, flagsV, beamEnd);
        // The next step's rays (driveAheadPart), on the lanes with the fewest beams.
        if (driveAhead && s + 1u < gP.steps)
        {
            driveAheadCtx(v, LANES - 1u - lane, drvC, drvHas, drvApart);
            drvCarried = true;
            driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 0u, stepBase + s + 1u);
        }
        // Without the grid this step: list the candidate pairs now (lanes in
        // reverse, so those with one beam fewer start first).
        uint listedB = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        bool early = candCount > 0u && !(cell > 0.0 && listedB * candCount > BRUTE_PAIRS_PER_LANE * LANES);
        if (early)
        {
            uint lanes = clamp(LANES / max(listedB, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            uint rl = LANES - 1u - lane;
            for (uint p = rl, pStop = EB_END(77, p, listedB * lanes, stride); p < pStop && rl < stride; p += stride)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
                trianglePairs(t, triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, parity);
            }
        }
#ifdef EB_ABLATE
        else if (candCount > 0u && gP.triGeo != 0u && !EB_A2(32u))
#else
        else if (candCount > 0u && gP.triGeo != 0u)
#endif
        {
            // The grid this step: each tested triangle's geometry once, for its walk
            // (triGeoFill; lanes in reverse).
            for (uint q = LANES - 1u - lane, qStop = EB_END(78, q, listedB, LANES); q < qStop; q += LANES)
                triGeoFill(triOff + (triCount <= TRI_LIST ? gsTested[q] : q), origin, range, parity);
        }
        sync();
        uint gridOn = gsUniformLanes;
        // The next step's bars, after its rays' barrier (the step's closing barrier
        // follows before its tyres).
        if (driveAhead && s + 1u < gP.steps) driveAheadPart(drvC, drvHas, drvApart, LANES - 1u - lane, 1u, stepBase + s + 1u);
        // The grid's pair list this step (pairReuseDecide; nonzero only with the grid): 2
        // reused, no grid to build.
        uint reuse = gridOn >> 1;
        // This step's deferred breaks, in spawn order (before selfApply's barrier).
#ifdef EB_ABLATE
        if (!EB_A2(2048u))
#endif
        if (lane == 0u) resolvePending(v, parity);
#ifdef EB_ABLATE
        if (gridOn != 0u && reuse != 2u && (gP.ablate & 1u) == 0u)
#else
        if (gridOn != 0u && reuse != 2u)
#endif
        {
            // Offsets: per lane its buckets' sum, a scan over blocks of lanes, then
            // the blocks' totals, then each lane's buckets.
            uint per = GRID_BUCKETS / LANES;
            uint block = LANES / GRID_BLOCKS;
            uint sum = 0u;
            for (uint k = 0u, kStop = EB_END(79, k, per, 1u); k < kStop; k++) sum += gsCount[lane * per + k].load();
            gsPart[lane] = sum;
            sync();
            if (lane < GRID_BLOCKS)
            {
                uint run = 0u;
                for (uint k = 0u, kStop = EB_END(80, k, block, 1u); k < kStop; k++)
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
            for (uint k = 0u, kStop = EB_END(81, k, per, 1u); k < kStop; k++)
            {
                gsOff[lane * per + k] = run;
                run += gsCount[lane * per + k].load();
            }
            sync();
            if (cell > 0.0)
            {
                for (uint c = lane, cStop = EB_END(82, c, candCount, LANES); c < cStop; c += LANES)
                {
                    uint br = gNext[candOff + c];
                    uint h = gLists[candOff + c];
                    gBeams[gP.gridBase + candOff + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] =
                        float4(gNodes[h * 3u].xyz, asfloat(h));
                }
            }
            sync();
        }
        // The step's candidate tests (one of five) at one call site, after the second phase.
        uint cmode = 4u, cN = 0u, ck = 1u, cstride = LANES;
        uint earlyN = 0u; // the early pairs, first in the pair loop below
        uint pairs = gsPairCount.load();
#ifdef EB_ABLATE
        if ((gP.ablate & 256u) != 0u) {} else
#endif
        if (early && pairs <= PAIR_LIST)
        {
            // The early pairs: the first of the pair loop below (a reused list's tests may
            // follow them in the same loop).
            earlyN = pairs;
        }
        else if (gridOn != 0u && reuse != 0u)
        {
            // The pair list kept over steps: built this step, or reused.
            if (reuse == 1u)
#ifdef EB_SELF_SPLIT
                selfSplitBuild(v, lane, triOff, triCount, nodeOff, nodeCount, origin, range, cell, candOff, candCount,
                               parity);
#else
                pairReuseBuild(v, lane, triOff, triCount, nodeOff, nodeCount, origin, range, cell, candOff, candCount,
                               parity);
#endif
        }
        else if (candCount > 0u && cell > 0.0 && gridOn != 0u)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            if (gP.triPairs != 0u)
            {
                for (uint p = lane, pStop = EB_END(84, p, listed * LIST_SUBS, LANES); p < pStop; p += LANES)
                {
                    uint t = triCount <= TRI_LIST ? gsTested[p / LIST_SUBS] : p / LIST_SUBS;
                    triangleGridList(triOff + t, p % LIST_SUBS, origin, range, cell, candOff, candCount, parity,
                                     triPairsAt(v));
                }
            }
            else
            {
                cN = listed * GRID_SUBS;
                ck = GRID_SUBS;
                cmode = 3u;
            }
        }
        else if (candCount > 0u)
        {
            // The brute pairs (no grid), or without a cell every candidate.
            bool brute = cell > 0.0;
            uint listed = brute && triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            ck = clamp(LANES / max(listed, 1u), 1u, candCount);
            cstride = (LANES / ck) * ck;
            cN = listed * ck;
            cmode = brute ? 1u : 0u;
        }
        // (The walk again past a list's room, the reused one's or the walk's: the loop below.)
        bool walk2 = false;
        uint reuseListed = 0xFFFFFFFFu; // (set: the reused list's tests ran, pairReuseTestsEnd)
        // (And one loop for a list's pairs, the reused list's with their box tests: one call
        // site of tryContact for both.)
        bool reusedList = false;
        uint pairN = 0u;
#ifdef EB_ABLATE
        if ((gP.ablate & 256u) != 0u) {} else
#endif
        if (gridOn != 0u && reuse != 0u)
        {
#ifdef EB_SELF_SPLIT
            walk2 = selfSplitTestsOrWalk(v, lane, reuse, triOff, triCount, origin, range, cell, candOff, candCount, beamOff, groupOff, groupCount, parity);
#else
            sync();
            reuseListed = pairReuseListed(reuse);
            walk2 = reuseListed > TRI_PAIR_CAP;
            reusedList = !walk2;
            pairN = walk2 ? 0u : reuseListed;
#endif
        }
        else if (gridOn != 0u && gP.triPairs != 0u)
        {
            // The walk's listed pairs, a pair a lane (triPairsAt); past the list's
            // room, the walk again with its tests.
            sync();
            uint listedPairs = gsPairCount.load();
            walk2 = listedPairs > TRI_PAIR_CAP;
            pairN = walk2 ? 0u : listedPairs;
        }
        uint pairAt = triPairsAt(v);
        for (uint i = lane, iStop = EB_END(87, i, earlyN + pairN, LANES); i < iStop; i += LANES)
        {
            uint gt;
            uint h;
            TriGeo tg;
            if (i < earlyN)
            {
                uint e = gsPairs[i];
                gt = triOff + (e >> 16);
                h = gLists[candOff + (e & 0xFFFFu)];
                tg = triangleGeo(gt, origin, range, parity);
            }
            else
            {
                gt = gNext[pairAt + 2u * (i - earlyN)];
                h = gNext[pairAt + 2u * (i - earlyN) + 1u];
                if (reusedList && !pairReuseNear(gt, h, origin, cell)) continue;
                tg = triGeoLoad(gt);
            }
            tryContact(tg, h, origin, range, beamOff, groupOff, groupCount, parity);
        }
        // (The pieces' tests stay a loop of their own: through the candidate loop below
        // the frame hung on the GPU in explicit_pair_reuse_gpu natively, its logic the
        // member step's; kernel-compile-time.md.)
        if (reusedList) piecesTests(v, lane, triOff, origin, range, cell, candOff, candCount, beamOff, groupOff, groupCount, parity);
        if (walk2)
        {
            cN = (triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount) * GRID_SUBS;
            ck = GRID_SUBS;
            cmode = 3u;
        }
        for (uint p = lane, pStop = EB_END(88, p, cN, cstride); p < pStop && lane < cstride; p += cstride)
        {
            uint q = p / ck;
            uint gt = cmode == 2u ? gNext[pieceVolAt(triOff + q)] : triOff + (cmode % 2u == 1u && triCount <= TRI_LIST ? gsTested[q] : q);
            triangleCandidates(gt, p % ck, ck, origin, range, cell, candOff, candCount, beamOff, groupOff, groupCount, parity, cmode);
        }
#ifndef EB_SELF_SPLIT
        if (reuseListed != 0xFFFFFFFFu) pairReuseTestsEnd(lane, reuse, reuseListed);
#endif
#ifdef EB_SELF_SPLIT
        // The pairs dispatch next; this step's rest opens the next step dispatch.
        sync();
        continue;
#endif
#ifdef EB_ABLATE
        if (selfOn != 0u && !EB_A2(64u))
#else
        if (selfOn != 0u)
#endif
        {
            sync();
            selfApply(v, true, lane, LANES, lane, origin, range, beamOff, groupOff, groupCount, parity);
        }
        surfaceStep(v, lane, surfOn, parity);
#ifdef EB_FRAME_SEG
        // The surface pipelines next; the rest of this step opens the next segment.
        continue;
#endif
#ifdef EB_AERO
        // Slide nodes, after this step's beams and self-collision, a lane each.
        if ((flagsV & FLAG_SLIDES) != 0u)
        {
            uint count = min(aero2.y, gP.slideTotal - min(aero2.x, gP.slideTotal));
            for (uint k = lane, kStop = EB_END(89, k, count, LANES); k < kStop; k += LANES) slideNode(aero2.x + k, SLIDE_FRAME, groupOff, parity);
        }
#endif
        sync();
    }
#ifdef EB_FRAME_SEG
    if (gP.pad2 < gP.steps)
    {
        // Before the surface pipelines: the hit flag for them and the next segment,
        // and this step's motion for the frame end's sleep rule.
        sync();
        if (lane == 0u) gMisc[segRec(v)].x = gsHit;
        if (sleepOn) sleepMark(v, sleepMove);
        return ebLoopCapFlush();
    }
#endif
#ifdef EB_SOLO_STEP
#ifdef EB_SELF_SPLIT
    // Every step's dispatch (the last step's rest and the frame's end in its own).
    if (gP.pad2 < gP.steps)
#else
    if (gP.pad2 + 1u < gP.steps)
#endif
    {
        // Not the last step (its closing barrier passed): the hit flag and the pair
        // list's state for the next, and this step's motion for the frame end's sleep
        // rule.
        soloStepSave(v, lane);
        if (sleepOn) sleepMark(v, sleepMove);
        return ebLoopCapFlush();
    }
#endif
    coupleEnd(v, lane, coupleOn);
    pairReuseEnd(v, lane);
    if (gP.steps > 0u && candCount > 0u)
    {
        for (uint t = lane, tStop = EB_END(90, t, triCount, LANES); t < tStop; t += LANES) triangleFinish(triOff + t);
    }
#ifdef EB_AERO
    if ((flagsV & FLAG_SLIDES) != 0u)
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        for (uint k = lane, kStop = EB_END(91, k, ext.y, LANES); k < kStop; k += LANES) slideSettle(ext.x + k);
    }
#endif
    // The last step's group breaks, for a read-back (the next node phase applies
    // them to the forces).
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint last = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane, jStop = EB_END(92, j, beamCount, LANES); j < jStop; j += LANES) groupDrops(beamOff + j, beamOff, groupOff, last, true);
    }
    if (sleepOn)
    {
        sleepMark(v, sleepMove);
        sleepSettle(v, lane, gP.steps, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
#ifdef EB_VV_SOLO
    // The last step's positions (each earlier step's were checked at the start of
    // the next step's node phase).
    for (uint i = lane, iStop = EB_END(93, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        float3 p = gNodes[(nodeOff + i) * 3u].xyz;
        if (any(abs(p - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(v);
    }
#endif
#endif
    ebLoopCapFlush();
}

