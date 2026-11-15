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
// Without vehicle sleep (the host keeps the group's frame when sleep is on).

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
// same per-vehicle calls on the same lanes of its partition.
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
        if (act) cplSpheresBy(v, lane, P);
        sync();
        if (act) cplCountBy(v, lane, P);
        sync();
        return;
    }
    if (s > 0u) cplIntegrate(v, lane);
    sync();
    cplSpheres(v, lane);
    sync();
    cplCount(v, lane);
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
    uint4 fl = v < gP.vehicles ? gMisc[vvRec(v, VV_FLAGS)] : uint4(0u, 0u, 0u, 0u);
    uint4 G = fl.y != 0u ? gMisc[gP.vvGroups + fl.z] : uint4(0u, 0u, 0u, 0u);
    uint k = splitUniform(lane, G.y);
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
    if (k <= VV_PART)
    {
        // The member's partition first: its lanes are lanes 0 .. P - 1 here.
        uint P = vvPartLanes(k);
        contactMembersStepAt(off, k, (lane + rank * P) % LANES, s, rank);
    }
    else
    {
        contactMemberStep(v, lane, s, rank == 0u);
    }
    if (surfOn != 0u)
    {
        surfaceStep(v, lane, surfOn, (stepBase + s) & 1u);
        sync();
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
    for (uint i = lane, iStop = EB_END(i, k, LANES); i < iStop; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u, jStop = EB_END(j, k, 1u); j < jStop; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
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
static const uint SPLIT_TEST_WGS = 16u;

// Group g's scratch for the tests across workgroups: its first member's wide-pass grid
// counts (gAcc, GRID_BUCKETS words) and grid offsets (gNext, GRID_BUCKETS + 1): that
// member's own self-collision grid, dead from its step's end to the next step's start.
// Words of the counts: per owner its pairs (0..3), hits (4..7), fastest candidate (8..11),
// listed triangles (12..15), candidates (16..19), offered (20..23), reset (24..27), fix
// (28..31); the partner bits (32..47); the front's flags (48: bit 0 open, bit 1 the grid).
uint splitVvAt(uint off) { return gP.gridCountBase + vvMember(off, 0u) * GRID_BUCKETS; }
uint splitVvOffAt(uint off) { return gP.gridOffBase + vvMember(off, 0u) * (GRID_BUCKETS + 1u); }

// The front's results out to the scratch (every lane; after the front's barriers).
void splitVvExport(uint off, uint front, uint lane)
{
    uint c = splitVvAt(off);
    if (lane < VV_PART)
    {
        uint a = lane;
        gAcc[c + a].store(0);
        gAcc[c + 4u + a].store(0);
        gAcc[c + 8u + a].store(int(gsCount[GRID_BUCKETS + a].load()));
        gAcc[c + 12u + a].store(int(gsVvList4[a].load()));
        gAcc[c + 16u + a].store(int(gsVvCand4[a].load()));
        gAcc[c + 20u + a].store(int(gsVvOff4[a].load()));
        gAcc[c + 24u + a].store(int(gsVvReset4[a]));
        gAcc[c + 28u + a].store(int(gsVvFix[a]));
    }
    if (lane < VV_PART * VV_PART) gAcc[c + 32u + lane].store(int(gsVvPair[lane]));
    if (lane == 0u) gAcc[c + 48u].store(int(front));
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
        gsVvHit4[a].store(uint(gAcc[c + 4u + a].load()));
        gsVvList4[a].store(uint(gAcc[c + 12u + a].load()));
        gsVvCand4[a].store(uint(gAcc[c + 16u + a].load()));
        gsVvOff4[a].store(uint(gAcc[c + 20u + a].load()));
        gsVvReset4[a] = uint(gAcc[c + 24u + a].load());
        gsVvFix[a] = uint(gAcc[c + 28u + a].load());
    }
    for (uint a = 0u; a < VV_PART; a++)
    {
        for (uint b = 0u; b < VV_PART; b++) gsVvPair[a * VV_PART + b] = uint(gAcc[c + 32u + a * VV_PART + b].load());
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
    uint k = splitUniform(lane, G.y);
    if (k == 0u) return;
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    uint parity = (stepBase + s) & 1u;
    bool small = k <= VV_PART;
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
            if (lane == 0u && small) gAcc[splitVvAt(off) + 48u].store(0);
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
        uint front = splitUniform(lane, uint(gAcc[splitVvAt(off) + 48u].load()));
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
            vvPassSmallLevels(k, off, runOff, points, parity, lane, (front & 2u) != 0u);
        }
    }
    if (s + 1u != gP.steps) return;
    // The frame's end (contactGroup's after its steps).
    if (lane == 0u)
    {
        coupleBegin();
        splitCplTable(off, k);
    }
    sync();
    uint coupleOn = gsUniformCouple;
    sync();
    coupleEndMembers(off, k, lane, coupleOn);
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
    uint k = G.y;
    if (k == 0u || k > VV_PART) return;
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    gVvC = splitVvAt(off);
    gVvO = splitVvOffAt(off);
    uint front = uint(gAcc[gVvC + 48u].load());
    if (front == 0u) return;
    bool vvGrid = (front & 2u) != 0u;
    bool vvList = vvGrid && gP.triPairs != 0u;
    uint at0 = part * LANES + lane;
    uint stride = SPLIT_TEST_WGS * LANES;
    if ((word & SPLIT_LEVELS) == 0u)
    {
        uint total = 0u;
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) total += uint(gAcc[gVvC + 12u + a].load());
        uint subs = clamp(LANES / max(total, 1u), 1u, 64u);
        for (uint p = at0, pStop = EB_END(p, total * subs, stride); p < pStop; p += stride)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(cGuard);
                uint n = uint(gAcc[gVvC + 12u + a].load());
                if (q < n) break;
                q -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint t = gNext[vvTestedAt(ra1.x + q)];
            uint cands = uint(gAcc[gVvC + 16u + a].load());
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
        uint at = triPairsAt(va);
        vvPairTest(a, va, gNext[at + 2u * q], gNext[at + 2u * q + 1u], runOff, points, asfloat(gStatic[va * 4u + 3u].z));
    }
    uint over = 0u;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
        over += uint(gAcc[gVvC + a].load()) > TRI_PAIR_CAP ? uint(gAcc[gVvC + 12u + a].load()) : 0u;
    uint subs = clamp(LANES / max(over, 1u), 1u, 64u);
    for (uint p = at0, pStop = EB_END(p, over * subs, stride); p < pStop; p += stride)
    {
        uint q = p / subs;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c < k; c++)
        {
            EB_PASS(cGuard);
            uint n = uint(gAcc[gVvC + c].load()) > TRI_PAIR_CAP ? uint(gAcc[gVvC + 12u + c].load()) : 0u;
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
        vvTriangleCandsGrid(a, va, ra1.x + t, p % subs, subs, runOff, points, uint(gAcc[gVvC + 16u + a].load()),
                            asfloat(gStatic[va * 4u + 3u].z));
    }
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_tests(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    splitTests(gid.x / SPLIT_TEST_WGS, gid.x % SPLIT_TEST_WGS, lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif

#ifdef EB_SOLO_STEP
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_member(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    // A solo vehicle's step of its frame (it returns at once for a group's member), then
    // a member's (returns at once for a solo vehicle).
    explicitBeamsFrameStep(gid, lid);
    sync();
    splitMember(gid.x, lid.x, gP.pad2);
    ebLoopCapFlush();
}
#endif

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_group(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    if (gP.pad2 == SPLIT_BEGIN)
        splitGroupBegin(gid.x, lid.x);
    else
        splitGroupStep(gid.x, lid.x, gP.pad2);
    ebLoopCapFlush();
}
