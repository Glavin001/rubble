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

// Group g's contact pass of step s; after the last step, the group's frame end.
void splitGroupStep(uint g, uint lane, uint s)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    uint k = splitUniform(lane, G.y);
    if (k == 0u) return;
    if (lane == 0u)
    {
        // The counters contactMemberStep / contactMembersStep clear at a step's start.
        vvClearCounters();
        vvClearSmall();
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    sync();
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    // Measurement only (time; contact-groups-design.md): the ablation word's 512 skips
    // the pass here too (the split modules have no EB_ABLATE build).
    if ((gP.ablate & 512u) != 0u) {} else
    if (k <= VV_PART)
        vvPassSmall(k, off, runOff, points, stepBase + s, lane);
    else
        vvPass(k, off, runOff, points, stepBase + s, lane);
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
