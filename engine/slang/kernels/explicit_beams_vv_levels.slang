void vvPassSmallLevels(uint k, uint off, uint runOff, uint points, uint parity, uint lane, bool vvGrid)
{
    // (The kept beam sums' tag for this pass: set by the caller, used from the snapshot
    // on; off while the snapshot is taken, which sums afresh.)
    uint tag = gVvTag;
    gVvTag = 0u;
    sync();
    // The two-body pass reads its hits' nodes' forces as at the pass's start (the
    // triangle's nodes and the point: vvStagedHit, vvTry). Nothing since the start
    // wrote a force (the scans, the lists and the tests do not), so they are taken
    // here, before the first hit is applied, for the listed hits' nodes alone (the
    // same value from every lane that takes a node); every member's nodes when an
    // owner's list overflowed.
    // (Measurement only, time: the parameter word's 65536 skips this snapshot, 131072
    // level 1's throttle updates, 16384 the residual phases, 32768 the hits' buckets and
    // application. The results are then meaningless.)
    if (gP.vvSplit != 0u && (gP.ablate & 65536u) == 0u)
    {
        bool every = false;
        for (uint a = 0u, aStop = EB_END(265, a, k, 1u); a < aStop; a++)
        {
            if (gsVvHit4[a].load() > vvHitRoom(a, points)) every = true;
        }
        if (every) vvSnapshot(off, k, parity, lane);
        else
        {
            for (uint a = 0u, aStop = EB_END(266, a, k, 1u); a < aStop; a++)
            {
                uint base = vvHitSmall(runOff, points, a, 0u);
                for (uint i = lane, iStop = EB_END(267, i, gsVvHit4[a].load(), LANES); i < iStop; i += LANES)
                {
                    uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
                    uint h = gNext[base + 2u * i + 1u];
                    // (The four nodes at one call site, in the same order: tn.x, tn.y, tn.z, h. The
                    // compilers inline every call; opaqueCount keeps the loop from being unrolled back.)
                    for (uint q = 0u, qStop = EB_END(651, q, opaqueCount(4u), 1u); q < qStop; q++)
                        vvSnapNode(q == 0u ? tn.x : (q == 1u ? tn.y : (q == 2u ? tn.z : h)),
                                   q < 3u ? tn.w : gStatic[gP.nodeInfoBase + h].w, parity, tag);
                }
            }
        }
        sync();
        if (!every) gVvTag = tag;
    }
    // The owners' levels, once (vvLevelsSet), and the highest: the levels past it
    // hold no owner (level 1 runs whatever the hits: the throttle updates).
    vvLevelsSet(k, lane);
    sync();
    // (The loop's bound from the uniform load itself: a browser's uniformity analysis
    // does not follow it through the lane's own copy.)
    uint packed = gsUniformLanes;
    gVvLv = packed;
    sync();
    uint top = 1u;
    for (uint a = 0u; a < VV_PART; a++) top = max(top, (packed >> (4u * a)) & 15u);
    // The hits, level by level (at most k levels); with the first, every listed
    // triangle's throttle update (its marks are final) and a later owner's reset.
    for (uint l = 1u, lGuard = 0u; l <= top; l++)
    {
        EB_PASS_UNIFORM(268, lGuard, EB_MAX_MEMBERS + 1u);
#ifdef EB_ABLATE
        // Measurement only: 524288 the levels (the hits' application).
        if ((gP.ablate & 524288u) != 0u) break;
#endif
        if (l == 1u && (gP.ablate & 131072u) == 0u && !vvPostDone())
        {
            uint total = 0u;
            for (uint a = 0u, aStop = EB_END(269, a, k, 1u); a < aStop; a++) total += gsVvList4[a].load();
            for (uint p = lane, pStop = EB_END(270, p, total, LANES); p < pStop; p += LANES)
            {
                uint q = p;
                uint a = 0u;
                for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
                {
                    EB_PASS(271, cGuard);
                    uint n = gsVvList4[a].load();
                    if (q < n) break;
                    q -= n;
                    a++;
                }
                uint triOff = gStatic[vvMember(off, a) * 4u + 1u].x;
                uint gt = triOff + gNext[vvTestedAt(triOff + q)];
                vvThrottleEnd(gt);
                if (gsVvFix[a] != 0u) gMisc[gP.vvTri + gt].x = 0u;
            }
        }
        if (vvGrid && (gP.ablate & 32768u) == 0u) vvHitBuckets(k, l, off, runOff, points, lane);
        for (uint a = 0u, aStop = EB_END(272, a, k, 1u); a < aStop; a++)
        {
            if (vvLv(a) != l || (gP.ablate & 32768u) != 0u) continue;
            uint va = vvMember(off, a);
            float range = asfloat(gStatic[va * 4u + 3u].z);
            uint hits = gsVvHit4[a].load();
            if (hits <= vvHitRoom(a, points))
                vvApplyOrderedIn(va, vvHitSmall(runOff, points, a, 0u), hits, range, parity, lane, LANES, a, vvGrid, runOff,
                                 points);
            else
            {
                uint4 ra1 = gStatic[va * 4u + 1u];
                uint listed = gsVvList4[a].load();
                uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
                for (uint p = lane, pStop = EB_END(273, p, listed * subs, LANES); p < pStop; p += LANES)
                {
                    uint t = gNext[vvTestedAt(ra1.x + p / subs)];
                    vvTriangleCandsApply(va, ra1.x + t, p % subs, subs, vvRunSmall(runOff, points, a),
                                         gsVvCand4[a].load(), range, parity, a);
                }
            }
        }
        sync();
        // §37.1: this level's owners' residual pass and work guard (listed hits).
        if (gP.vvSplit != 0u && (gP.ablate & 16384u) == 0u)
        {
            for (uint ph = 0u; ph < 3u; ph++)
            {
                for (uint a = 0u, aStop = EB_END(274, a, k, 1u); a < aStop; a++)
                {
                    if (vvLv(a) != l) continue;
                    uint hits = gsVvHit4[a].load();
                    if (hits > vvHitRoom(a, points)) continue;
                    uint va = vvMember(off, a);
                    vvResidualPhase(va, vvHitSmall(runOff, points, a, 0u), hits, asfloat(gStatic[va * 4u + 3u].z),
                                    parity, lane, LANES, a, ph);
                }
                sync();
            }
        }
    }
    // The two-body counts of every owner's turn, once all levels are applied: the
    // counts a hit wrote (vvSplitCount: its triangle's nodes, its point in its owner's
    // slot), from the hit lists; every node's when an owner's list overflowed.
    uint anyHits = 0u;
    bool listedAll = true;
    for (uint a = 0u, aStop = EB_END(275, a, k, 1u); a < aStop; a++)
    {
        uint n = gsVvHit4[a].load();
        anyHits += n;
        if (n > vvHitRoom(a, points)) listedAll = false;
    }
    if (anyHits > 0u && !listedAll) vvSplitClear(off, k, vvSplitSlots(k), lane);
    else if (anyHits > 0u && gP.vvSplit != 0u)
    {
        for (uint a = 0u, aStop = EB_END(276, a, k, 1u); a < aStop; a++)
        {
            uint base = vvHitSmall(runOff, points, a, 0u);
            for (uint i = lane, iStop = EB_END(277, i, gsVvHit4[a].load(), LANES); i < iStop; i += LANES)
            {
                uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
                gAcc[vvSplitTri(tn.x)].store(0);
                gAcc[vvSplitTri(tn.y)].store(0);
                gAcc[vvSplitTri(tn.z)].store(0);
                gAcc[vvSplitPoint(a, gNext[base + 2u * i + 1u])].store(0);
            }
        }
    }
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_contact(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    contactGroup(gid.x, lid.x);
    ebLoopCapFlush();
}

// Contact group g: its members stepped together with the contact pass.
void contactGroup(uint g, uint lane)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    if (lane == 0u)
    {
        gsUniformLanes = G.y;
        coupleBegin();
        driveBegin();
    }
    sync();
    uint k = gsUniformLanes;
    uint coupleOn = gsUniformCouple;
    uint driveOn = gsUniformDrive;
    uint surfOn = surfaceOn();
    sync();
    // The groups over two modules (engine_exec::explicit_plan Kernel::ContactBig): the
    // frame with the groups of at most VV_PART members (EB_VV_SMALL_ONLY) and the
    // larger groups' own dispatch (EB_VV_BIG_ONLY, entry explicit_beams_contact); each
    // returns at once for the other's groups (k uniform), and holds only its own path.
    // Without either define, every group.
#ifdef EB_VV_SMALL_ONLY
    if (k > VV_SIDE) return;
#endif
#ifdef EB_VV_BIG_ONLY
    if (k <= VV_SIDE) return;
#endif
    // Vehicle sleep: a group with an awake member wakes the others; a group asleep is
    // not stepped.
    bool sleepOn = gP.sleepRec != 0u;
    if (sleepOn)
    {
        if (sleepGroupSkip(off, k, lane)) return;
    }
    uint2 sleepMask = uint2(0u, 0u);
    // Members in vehicle order (a rank sort), their run prefixes and member
    // indices.
    for (uint i = lane, iStop = EB_END(278, i, k, LANES); i < iStop; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u, jStop = EB_END(279, j, k, 1u); j < jStop; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u, mStop = EB_END(280, m, k, 1u); m < mStop; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
        vvClearCounters();
        // The coupling's spheres in workgroup memory (cplGsRange): a small group's
        // members one after another.
        cplGsClear();
        for (uint m = 0u, mStop = EB_END(281, m, k, 1u); m < mStop; m++) cplListBegin(vvMember(off, m));
        if (k <= CPL_GS_VEHICLES)
        {
            uint base = 0u;
            uint nodeBase = 0u;
            for (uint m = 0u, mStop = EB_END(282, m, k, 1u); m < mStop; m++)
            {
                cplGsAdd(m, vvMember(off, m), base, nodeBase);
                base += gsCpl[CPL_GS_TAB + 5u * m + 2u];
                nodeBase += gsCpl[CPL_GS_TAB + 5u * m + 4u];
            }
        }
    }
    sync();
    for (uint m = 0u, mStop = EB_END_UNIFORM(283, m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++) contactMemberBegin(vvMember(off, m), lane);
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    for (uint s = 0u, sStop = EB_END_UNIFORM(284, s, gP.steps, 1u, EB_MAX_STEPS); s < sStop; s++)
    {
        // Wheels on each other (drive header bit 2): the last step's reactions, this
        // step's rays (explicit_beams_drive.slang wheelsPartners).
        if ((driveOn & 2u) != 0u) wheelsPartners(off, k, lane, s, true);
        coupleStepMembers(off, k, lane, s, coupleOn);
        driveStepMembers(off, k, lane, s, driveOn);
#ifndef EB_VV_BIG_ONLY
        if (k <= VV_SIDE)
        {
            contactMembersStep(off, k, lane, s);
        }
#endif
#if !defined(EB_VV_BIG_ONLY) && !defined(EB_VV_SMALL_ONLY)
        else
#endif
#ifndef EB_VV_SMALL_ONLY
        {
            for (uint m = 0u, mStop = EB_END_UNIFORM(285, m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++) contactMemberStep(vvMember(off, m), lane, s, m == 0u);
        }
#endif
        if (surfOn != 0u)
        {
            for (uint m = 0u, mStop = EB_END_UNIFORM(286, m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++)
                surfaceStep(vvMember(off, m), lane, surfOn, (stepBase + s) & 1u);
            sync();
        }
        if (sleepOn) sleepMask = sleepTrackMembers(off, k, lane, sleepMask);
#ifdef EB_ABLATE
        if ((gP.ablate & 512u) != 0u) {} else
#endif
#ifndef EB_VV_BIG_ONLY
        if (k <= VV_SIDE)
        {
            vvPassSmall(k, off, runOff, points, stepBase + s, lane);
        }
#endif
#if !defined(EB_VV_BIG_ONLY) && !defined(EB_VV_SMALL_ONLY)
        else
#endif
#ifndef EB_VV_SMALL_ONLY
        {
            vvPass(k, off, runOff, points, stepBase + s, lane);
        }
#endif
    }
    coupleEndMembers(off, k, lane, coupleOn);
    if ((driveOn & 2u) != 0u) wheelsPartners(off, k, lane, gP.steps, false);
    if (sleepOn) sleepGroupEnd(off, k, lane, sleepMask, gP.steps);
    for (uint m = 0u, mStop = EB_END_UNIFORM(287, m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++) contactMemberEnd(vvMember(off, m), lane);
}

// The split group path (contact-groups-design.md): a contact group's substeps across
// workgroups, a member's step in its own.
#ifdef EB_VV_SPLIT
#include "explicit_beams_fleet_track.slang"
#include "explicit_beams_split.slang"
#endif
#ifdef EB_VV_WIDE
#include "explicit_beams_vvwide.slang"
#endif
