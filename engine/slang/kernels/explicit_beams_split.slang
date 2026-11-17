// --- The split group path (engine/docs/measurements/contact-groups-design.md) ---
// A contact group's frame (contactGroup) as dispatches: each substep is two, the vehicles'
// steps (explicit_beams_split_member: a workgroup per vehicle, each member's step exactly
// as its group's workgroup ran it, on the same lanes of its partition, and each solo
// vehicle's step of its frame, EB_SOLO_STEP) and the groups' contact passes
// (explicit_beams_split_group: a workgroup per group, the same pass). Without a group
// the plan's arguments give these no workgroups and the solo frame runs whole. The
// group's frame start and end run in the group dispatches too: copy `steps + 1` of the
// parameters (pad2 = SPLIT_BEGIN) the start; the last step's pass then the end.
// Results are contactGroup's bit for bit: the same functions on the same state in the
// same order; nothing lives in workgroup memory from one dispatch to the next (every
// value contactGroup kept there across substeps is recomputed: the coupling's header and
// sphere table from the buffers, the contact pass's counters cleared at its start).
// Vehicle sleep (explicit_beams_sleep.slang) as contactGroup has it: the groups' frame
// start wakes a group with an awake member (sleepGroupSkip), a sleeping group's and
// member's dispatches return at once (splitAwake), each member's step marks its motion,
// the groups' frame end settles the members together (sleepGroupEnd).

static const uint SPLIT_BEGIN = 0xFFFFFFFFu;

// Lane 0 to every lane, uniform (gsUniformLanes: the WGSL post-process reads it with
// workgroupUniformLoad). Barriers on both sides.
uint splitUniform(uint lane, uint x)
{
    sync();
    if (lane == 0u) gsUniformLanes = x;
    sync();
    uint u = gsUniformLanes;
    sync();
    return u;
}

// Group record G's members, 0 for a group asleep: after the groups' frame start (or the
// awake list's kernel) every member of a group is awake or every member asleep, so its
// first listed member says which (the unsorted list: a sleeping group's start never
// sorted it).
uint splitAwake(uint4 G)
{
    if (G.y == 0u || gP.sleepRec == 0u) return G.y;
    return sleepAsleep(gNext[gP.vvMembers + G.x]) ? 0u : G.y;
}

// The coupling's sphere and node-range table as contactGroup writes it for its k members
// (lane 0): each member from the bases the earlier ones left.
void splitCplTable(uint off, uint k)
{
    cplGsClear();
    if (k > CPL_GS_VEHICLES) return;
    uint base = 0u;
    uint nodeBase = 0u;
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
    {
        cplGsAdd(m, vvMember(off, m), base, nodeBase);
        base += gsCpl[CPL_GS_TAB + 5u * m + 2u];
        nodeBase += gsCpl[CPL_GS_TAB + 5u * m + 4u];
    }
}

// coupleStepMembers for member `rank` alone (uniform `on`, k the group's members): the
// same per-vehicle calls on the same lanes of its partition (the candidate list's
// shapes' maxima and its count pass included, as contactGroup runs them).
void splitCoupleStep(uint v, uint rank, uint k, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    if (k <= CPL_GS_VEHICLES && (on & 4u) == 0u)
    {
        uint P = vvPartLanes(k);
        bool act = lane < P;
        if (s > 0u)
        {
            if (act) cplReactBy(v, lane, P);
            sync();
            if (act) cplIntegrateBy(v, lane, P);
        }
        sync();
        if (act)
        {
            cplSpheresBy(v, lane, P);
            cplListShapes(v, lane, P, s);
        }
        sync();
        if (act) cplCountStep(v, lane, P, s);
        sync();
        return;
    }
    if (s > 0u) cplIntegrate(v, lane);
    sync();
    cplSpheres(v, lane);
    cplListShapes(v, lane, LANES, s);
    sync();
    cplCountStep(v, lane, LANES, s);
    sync();
}

// driveStepMembers for one member: its wheels on lanes 0 .. DRV_MAX_WHEELS - 1.
void splitDriveStep(uint v, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    driveWheelsSplit(v, lane % DRV_MAX_WHEELS, lane < DRV_MAX_WHEELS, gMisc[v].w + s);
    sync();
}

// Vehicle v's step s (parameter copy s) when it is a member of a contact group this
// frame; any other vehicle returns at once.
void splitMember(uint v, uint lane, uint s)
{
    // (With self-collision split the vehicles' dispatch runs once more after the last
    // step, s = steps: the solo vehicles' last rest and frame end; a member has none.)
    if (s >= gP.steps) return;
    uint4 fl = v < gP.vehicles ? gMisc[vvRec(v, VV_FLAGS)] : uint4(0u, 0u, 0u, 0u);
    uint4 G = fl.y != 0u ? gMisc[gP.vvGroups + fl.z] : uint4(0u, 0u, 0u, 0u);
    // (A sleeping member is not stepped: its group sleeps.)
    uint k = splitUniform(lane, v < gP.vehicles && sleepAsleep(v) ? 0u : G.y);
    if (k == 0u) return;
    uint off = G.x;
    uint rank = gMisc[vvRec(v, VV_MEMBER)].x;
    if (lane == 0u)
    {
        coupleBegin();
        driveBegin();
        splitCplTable(off, k);
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    sync();
    uint coupleOn = gsUniformCouple;
    uint driveOn = gsUniformDrive;
    uint surfOn = surfaceOn();
    sync();
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    splitCoupleStep(v, rank, k, lane, s, coupleOn);
    splitDriveStep(v, lane, s, driveOn);
    // Every member on the whole of its workgroup with the one member step
    // (contactMemberStep), whatever the group's size: the same results as a small
    // group's partitioned step in the groups' own workgroup (contactMembersStepAt, which
    // this module no longer holds), bit for bit (group_split_gpu, self_split_gpu).
    contactMemberStep(v, lane, s, rank == 0u);
    if (surfOn != 0u)
    {
        surfaceStep(v, lane, surfOn, (stepBase + s) & 1u);
        sync();
    }
    // Vehicle sleep: this step's motion (the velocities its node phase left; nothing
    // later in the step writes one), as contactGroup's sleepTrackMembers after its
    // members' steps.
    if (gP.sleepRec != 0u)
    {
        uint4 r0 = gStatic[v * 4u];
        sleepMark(v, sleepTrack(r0.x, r0.y, lane, false));
    }
}

// Group g's frame start (contactGroup's before its steps): the members in vehicle order,
// their run prefixes, each member's start.
void splitGroupBegin(uint g, uint lane)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint k = splitUniform(lane, G.y);
    if (k == 0u) return;
    // Vehicle sleep: a group with an awake member wakes the others; a group asleep is not
    // stepped (contactGroup's).
    if (gP.sleepRec != 0u)
    {
        if (sleepGroupSkip(off, k, lane)) return;
    }
    for (uint i = lane, iStop = EB_END(i, k, LANES); i < iStop; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u, jStop = EB_END(j, k, 1u); j < jStop; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    // (The owners' state by vehicle: each member's group's step count, the parity of
    // the group's pass for its boxes: explicit_beams_fleet.slang flParity.)
    if (gP.vvFleetNext != 0u)
    {
        uint base = gMisc[vvMember(off, 0u)].w;
        for (uint i = lane, iStop = EB_END(i, k, LANES); i < iStop; i += LANES) gNext[flVehAt(0u, vvMember(off, i))] = base;
    }
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
        // The coupling list's frame maxima cleared (contactGroup's frame start;
        // cplListBegin reads the header from workgroup memory: coupleBegin first).
        coupleBegin();
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplListBegin(vvMember(off, m));
    }
    sync();
    for (uint m = 0u, mStop = EB_END_UNIFORM(m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++) contactMemberBegin(vvMember(off, m), lane);
}

// The step word of a parameter copy (gP.pad2): the step, with the phase in its top bits.
// SPLIT_TESTS (the step's first group dispatch): the triangle tests run in their own
// dispatches across workgroups (explicit_beams_split_tests), the pass stops after its
// front; SPLIT_LEVELS (the step's second group dispatch, after the tests): the levels
// (and after the last step the group's frame end); for the tests' dispatches the second
// of them (the listed pairs).
static const uint SPLIT_TESTS = 0x20000000u;
static const uint SPLIT_LEVELS = 0x40000000u;
static const uint SPLIT_STEP = 0x0FFFFFFFu;
// Workgroups per group of the tests' dispatches (the host's SPLIT_TEST_WGS).
static const uint SPLIT_TEST_WGS = 64u;

// Group g's scratch for the tests across workgroups: GRID_BUCKETS words of gAcc and
// GRID_BUCKETS + 1 of gNext at its first member's place in vehicle contact's own scratch
// (gP.vvScratch, gP.vvScratchOff: not the member's self-collision grid words, which hold
// a solo step's saved pair-list state, soloStepAt).
// Words of the counts: per owner its pairs (0..3), hits (4..7), fastest candidate (8..11),
// listed triangles (12..15), candidates (16..19), offered (20..23), reset (24..27), fix
// (28..31); the partner bits (32..47); the front's flags (48: bit 0 open, bit 1 the grid).
uint splitVvAt(uint off) { return gP.vvScratch + vvMember(off, 0u) * GRID_BUCKETS; }
uint splitVvOffAt(uint off) { return gP.vvScratchOff + vvMember(off, 0u) * (GRID_BUCKETS + 1u); }

// The front's results out to the scratch (every lane; after the front's barriers).
void splitVvExport(uint off, uint front, uint lane)
{
    uint c = splitVvAt(off);
    if (lane < VV_PART)
    {
        uint a = lane;
        gAcc[c + a].store(0);
        gAcc[c + 1u * VV_PART + a].store(0);
        gAcc[c + 2u * VV_PART + a].store(int(gsCount[GRID_BUCKETS + a].load()));
        gAcc[c + 3u * VV_PART + a].store(int(gsVvList4[a].load()));
        gAcc[c + 4u * VV_PART + a].store(int(gsVvCand4[a].load()));
        gAcc[c + 5u * VV_PART + a].store(int(gsVvOff4[a].load()));
        gAcc[c + 6u * VV_PART + a].store(int(gsVvReset4[a]));
        gAcc[c + 7u * VV_PART + a].store(int(gsVvFix[a]));
    }
    if (lane < VV_PART * VV_PART) gAcc[c + 8u * VV_PART + lane].store(int(gsVvPair[lane]));
    if (lane == 0u) gAcc[c + 8u * VV_PART + VV_PART * VV_PART].store(int(front));
    if ((front & 2u) != 0u)
    {
        uint o = splitVvOffAt(off);
        for (uint b = lane; b <= GRID_BUCKETS; b += LANES) gNext[o + b] = gsOff[b];
    }
}

// The levels' inputs back from the scratch (lane 0, before a barrier).
void splitVvImport(uint off)
{
    uint c = splitVvAt(off);
    for (uint a = 0u; a < VV_PART; a++)
    {
        gsVvHit4[a].store(uint(gAcc[c + 1u * VV_PART + a].load()));
        gsVvList4[a].store(uint(gAcc[c + 3u * VV_PART + a].load()));
        gsVvCand4[a].store(uint(gAcc[c + 4u * VV_PART + a].load()));
        gsVvOff4[a].store(uint(gAcc[c + 5u * VV_PART + a].load()));
        gsVvReset4[a] = uint(gAcc[c + 6u * VV_PART + a].load());
        gsVvFix[a] = uint(gAcc[c + 7u * VV_PART + a].load());
    }
    for (uint a = 0u; a < VV_PART; a++)
    {
        for (uint b = 0u; b < VV_PART; b++) gsVvPair[a * VV_PART + b] = uint(gAcc[c + 8u * VV_PART + a * VV_PART + b].load());
    }
}

// Group g's contact pass of step s; after the last step, the group's frame end. Phase A
// (the step's first group dispatch): the whole pass, or with SPLIT_TESTS a small group's
// front only; phase B (SPLIT_LEVELS): a small group's levels when its tests were split,
// then (last step) the frame's end.
void splitGroupStep(uint g, uint lane, uint word)
{
    uint s = word & SPLIT_STEP;
    bool levels = (word & SPLIT_LEVELS) != 0u;
    bool tests = (word & SPLIT_TESTS) != 0u;
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    uint k = splitUniform(lane, splitAwake(G));
    if (k == 0u) return;
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    uint parity = (stepBase + s) & 1u;
    bool small = k <= VV_PART;
#ifdef EB_VV_FLEET
    // The owners' state by vehicle (PERF_FLEET_STATE, explicit_beams_fleet.slang; the
    // host sets it only with the tests split, the candidates' grid and the pair list).
    // (PERF_FLEET_ANY, which the host sets only with the front fleet-wide: a group of any
    // size on the fleet-wide passes, its levels fleetLevelsAny; the second measurement word's
    // bit 0x80000000, tests only (set_ablation2), takes those levels whatever the hit lists.)
    bool anySize = (gP.perfFlags & (PERF_FLEET_FRONT | PERF_FLEET_ANY)) == (PERF_FLEET_FRONT | PERF_FLEET_ANY);
    if ((small || anySize) && (tests || levels) && (gP.perfFlags & PERF_FLEET_STATE) != 0u && (gP.ablate & 512u) == 0u)
    {
        if (!levels)
        {
            // (With the front as fleet-wide dispatches this one runs only for a larger
            // group's pass: nothing here.)
            if ((gP.perfFlags & PERF_FLEET_FRONT) != 0u) return;
            if (lane == 0u)
            {
                gsUniformHits = 0u;
                gsUniformOpen = 0u;
            }
            sync();
            fleetGroupFront(g, k, off, runOff, parity, lane);
            return;
        }
        uint tag = (gP.perfFlags & PERF_VV_BEAM_SUM) != 0u ? ((stepBase + s + 1u) | 0x80000000u) : 0u;
        // With PERF_FLEET_ANY every group takes the levels for any number of owners (they
        // visit the owners with hits alone, on the lanes with a hit); a small group one of
        // whose owners lists more than FL_SCAN_MOST hits keeps the small groups' levels,
        // whose hits' order is by buckets, not by scan.
        uint longest = 0u;
        if (small && anySize && (gP.beamOrderPad0 & 0x80000000u) == 0u)
        {
            if (lane == 0u)
            {
                for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) longest = max(longest, flHits(vvMember(off, m)));
            }
            longest = splitUniform(lane, longest);
        }
        // (PERF_FLEET_LEVELS: the levels of a group whose hit lists fit across the GPU,
        // explicit_beams_fleet.slang fleetLevelsDefer; the pass's end then in the levels'
        // end dispatch.)
        if (flLevelsOn() && fleetLevelsDefer(k, off, lane, s, !(small && (!anySize || longest > FL_SCAN_MOST)))) return;
        if (small && (!anySize || longest > FL_SCAN_MOST))
            fleetGroupLevels(k, off, runOff, points, parity, lane, tag);
        else if (flHitLatOn())
            fleetLevelsLat(k, off, parity, lane, tag);
        else
            fleetLevelsAny(k, off, parity, lane, tag);
        splitGroupEnd(off, k, lane, s);
        return;
    }
#endif
    if (!levels)
    {
        if (lane == 0u)
        {
            // The counters contactMemberStep / contactMembersStep clear at a step's start.
            vvClearCounters();
            vvClearSmall();
            gsUniformHits = 0u;
            gsUniformOpen = 0u;
        }
        sync();
        // Measurement only (time; contact-groups-design.md): the ablation word's 512 skips
        // the pass here too (the split modules have no EB_ABLATE build).
        if ((gP.ablate & 512u) != 0u)
        {
            if (lane == 0u && small) gAcc[splitVvAt(off) + 8u * VV_PART + VV_PART * VV_PART].store(0);
        }
        else if (small && tests)
        {
            uint front = vvPassSmallFront(k, off, runOff, points, parity, lane);
            splitVvExport(off, front, lane);
        }
        else if (small)
            vvPassSmall(k, off, runOff, points, stepBase + s, lane);
        else
            vvPass(k, off, runOff, points, stepBase + s, lane);
        // Without the tests split the frame's end follows the last pass here.
        if (tests || s + 1u != gP.steps) return;
    }
    else if (small)
    {
        // The levels after the tests' dispatches.
        uint front = splitUniform(lane, uint(gAcc[splitVvAt(off) + 8u * VV_PART + VV_PART * VV_PART].load()));
        if (front != 0u)
        {
            if (lane == 0u)
            {
                vvClearCounters();
                vvClearSmall();
                splitVvImport(off);
                gsUniformHits = 0u;
                gsUniformOpen = 0u;
            }
            sync();
            gVvTag = (gP.perfFlags & PERF_VV_BEAM_SUM) != 0u ? ((stepBase + s + 1u) | 0x80000000u) : 0u;
            vvPassSmallLevels(k, off, runOff, points, parity, lane, (front & 2u) != 0u);
            gVvTag = 0u;
        }
    }
    splitGroupEnd(off, k, lane, s);
}

// After the last step's pass: the group's frame end (contactGroup's after its steps).
void splitGroupEnd(uint off, uint k, uint lane, uint s)
{
    if (s + 1u != gP.steps) return;
    if (lane == 0u)
    {
        coupleBegin();
        splitCplTable(off, k);
    }
    sync();
    uint coupleOn = gsUniformCouple;
    sync();
    coupleEndMembers(off, k, lane, coupleOn);
    // Vehicle sleep: each member's rule on the motion its steps marked; the members fall
    // asleep together (contactGroup's, its marks already in the flags).
    if (gP.sleepRec != 0u) sleepGroupEnd(off, k, lane, uint2(0u, 0u), gP.steps);
    for (uint m = 0u, mStop = EB_END_UNIFORM(m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++) contactMemberEnd(vvMember(off, m), lane);
}

#ifdef EB_VV_GLOBAL
// The tests of small groups across workgroups (SPLIT_TEST_WGS a group): part `part` of
// group g's walk (phase A: each listed triangle's cells, listing its pairs or, without
// the pair list, testing inline) or its listed pairs' tests (SPLIT_LEVELS), as
// vvPassSmallTests runs them in the group's workgroup, the lanes of the group's
// workgroups taking its items in turn. The counters in the group's scratch.
void splitTests(uint g, uint part, uint lane, uint word)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint k = splitAwake(G);
    if (k == 0u || k > VV_PART) return;
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
#ifdef EB_VV_FLEET
    if ((gP.perfFlags & PERF_FLEET_STATE) != 0u)
    {
        fleetTests(g, part, lane, word);
        return;
    }
#endif
    gVvC = splitVvAt(off);
    gVvO = splitVvOffAt(off);
    uint front = uint(gAcc[gVvC + 8u * VV_PART + VV_PART * VV_PART].load());
    if (front == 0u) return;
    bool vvGrid = (front & 2u) != 0u;
    bool vvList = vvGrid && gP.vvTriPairs != 0u;
    uint at0 = part * LANES + lane;
    uint stride = SPLIT_TEST_WGS * LANES;
    if ((word & SPLIT_LEVELS) == 0u)
    {
        uint total = 0u;
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) total += uint(gAcc[gVvC + 3u * VV_PART + a].load());
        // (Parts of a triangle's cells or candidates: as many as the group's threads
        // here hold, the same pairs listed whatever the count.)
        uint subs = clamp(stride / max(total, 1u), 1u, 64u);
        for (uint p = at0, pStop = EB_END(p, total * subs, stride); p < pStop; p += stride)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(cGuard);
                uint n = uint(gAcc[gVvC + 3u * VV_PART + a].load());
                if (q < n) break;
                q -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint t = gNext[vvTestedAt(ra1.x + q)];
            uint cands = uint(gAcc[gVvC + 4u * VV_PART + a].load());
            float r = asfloat(gStatic[va * 4u + 3u].z);
            if (vvList)
                vvTriangleCandsList(a, va, ra1.x + t, p % subs, subs, runOff, points, cands, r);
            else if (vvGrid)
                vvTriangleCandsGrid(a, va, ra1.x + t, p % subs, subs, runOff, points, cands, r);
            else
                vvTriangleCandsSmall(a, va, ra1.x + t, p % subs, subs, runOff, points, cands, r);
        }
        return;
    }
    if (!vvList) return;
    // The listed pairs, a lane each; an owner past the list's room tests inline.
    uint total = 0u;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
    {
        uint n = uint(gAcc[gVvC + a].load());
        total += n <= TRI_PAIR_CAP ? n : 0u;
    }
    for (uint p = at0, pStop = EB_END(p, total, stride); p < pStop; p += stride)
    {
        uint q = p;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(cGuard);
            uint n = uint(gAcc[gVvC + c].load());
            n = n <= TRI_PAIR_CAP ? n : 0u;
            if (q < n)
            {
                a = c;
                break;
            }
            q -= n;
        }
        uint va = vvMember(off, a);
        uint at = vvTriPairsAt(va);
        vvPairTest(a, va, gNext[at + 2u * q], gNext[at + 2u * q + 1u], runOff, points, asfloat(gStatic[va * 4u + 3u].z));
    }
    uint over = 0u;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
        over += uint(gAcc[gVvC + a].load()) > TRI_PAIR_CAP ? uint(gAcc[gVvC + 3u * VV_PART + a].load()) : 0u;
    uint subs = clamp(stride / max(over, 1u), 1u, 64u);
    for (uint p = at0, pStop = EB_END(p, over * subs, stride); p < pStop; p += stride)
    {
        uint q = p / subs;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(cGuard);
            uint n = uint(gAcc[gVvC + c].load()) > TRI_PAIR_CAP ? uint(gAcc[gVvC + 3u * VV_PART + c].load()) : 0u;
            if (q < n)
            {
                a = c;
                break;
            }
            q -= n;
        }
        uint va = vvMember(off, a);
        uint4 ra1 = gStatic[va * 4u + 1u];
        uint t = gNext[vvTestedAt(ra1.x + q)];
        vvTriangleCandsGrid(a, va, ra1.x + t, p % subs, subs, runOff, points, uint(gAcc[gVvC + 4u * VV_PART + a].load()),
                            asfloat(gStatic[va * 4u + 3u].z));
    }
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_tests(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
#ifdef EB_VV_FLEET
    // The tests over every owner (a thread an element, not a group's workgroups).
    if ((gP.perfFlags & PERF_FLEET_FRONT) != 0u)
    {
        fleetWideTests(gid.x, lid.x, gP.pad2);
        ebLoopCapFlush();
        return;
    }
#endif
    splitTests(gid.x / SPLIT_TEST_WGS, gid.x % SPLIT_TEST_WGS, lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif

#ifdef EB_SELF_SPLIT
// A member's step s after its tests' dispatches (PERF_MEMBER_LIST with self-collision
// split: contactMemberStep ends at its tests): the walk's list recorded in the pair
// list's state (selfSplitWalkEnd), the step's self-collision hits applied (every hit,
// of the step's dispatch, the walk's and the pairs', counts in the hit list's count), the
// slide nodes, as contactMemberStep runs them after its tests; then the list's state for
// the next step, or its statistics at the frame's end. Before the groups' pass of the
// step, which reads what the hits leave. A solo vehicle returns at once.
void memberRest(uint v, uint lane, uint s)
{
    uint4 fl = v < gP.vehicles ? gMisc[vvRec(v, VV_FLAGS)] : uint4(0u, 0u, 0u, 0u);
    bool keep = (gP.perfFlags & PERF_MEMBER_LIST) != 0u && pairReuseOn() && s < gP.steps;
    // (A sleeping member has no step to finish.)
    uint member = splitUniform(lane, keep && !(v < gP.vehicles && sleepAsleep(v)) ? fl.y : 0u);
    if (member == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint parity = (misc.w + s) & 1u;
    soloStepLoad(v, lane);
    sync();
    if (lane == 0u)
    {
        selfSplitWalkEnd(v);
        gsUniformHits = gAcc[hitAcc(v)].load() != 0 ? 1u : 0u;
        gsUniformOpen = 0u;
        gsUniformLanes = r1.y > 0u ? 1u : 0u;
    }
    sync();
    uint selfOn = gsUniformLanes;
    if (selfOn != 0u) selfApply(v, true, lane, LANES, lane, origin, asfloat(r3.z), r0.z, r2.z, r2.w, parity);
#ifdef EB_AERO
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
    if (lane == LANES - 1u && (r3.x & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, SLIDE_FRAME, r2.z, parity);
#endif
    sync();
    memberListSave(v, lane, s);
}
#endif

#ifdef EB_SOLO_STEP
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_member(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    // Vehicle sleep's awake-indexed dispatch: the workgroup's awake vehicle (gP.vehicles
    // or more past them: nothing to step).
    uint3 vid = gid;
    if (sleepListedOn()) vid.x = sleepPick(gid.x, lid.x);
#ifdef EB_SELF_SPLIT
    // The members' rest of step s (the step word's SPLIT_LEVELS: after the step's walk
    // and pairs dispatches, before the groups' pass).
    if ((gP.pad2 & SPLIT_LEVELS) != 0u)
    {
        memberRest(vid.x, lid.x, gP.pad2 & SPLIT_STEP);
        ebLoopCapFlush();
        return;
    }
    // Self-collision split (explicit_beams_selfsplit.slang): the solo vehicles' steps cut
    // at their tests, the walk and pairs dispatches after this one; the next pairs
    // dispatch's slot cleared as the self split's step dispatch does (the dispatch's
    // first workgroup, whatever its vehicle).
    selfSplitArgsClear(gid.x, lid.x);
#endif
    // A solo vehicle's step of its frame (it returns at once for a group's member), then
    // a member's (returns at once for a solo vehicle).
    explicitBeamsFrameStep(vid, lid);
    sync();
    splitMember(vid.x, lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_group(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
#ifdef EB_VV_FLEET
    // (The levels across the GPU's end dispatch: explicit_beams_fleet.slang.)
    if ((gP.pad2 & 0xF0000000u) == SPLIT_LV)
    {
        fleetLevelsEnd(gid.x, lid.x, gP.pad2);
        ebLoopCapFlush();
        return;
    }
#endif
    if (gP.pad2 == SPLIT_BEGIN)
        splitGroupBegin(gid.x, lid.x);
    else
        splitGroupStep(gid.x, lid.x, gP.pad2);
    ebLoopCapFlush();
}
