// Vehicle sleep (CPU reference engine_softbody::explicit_sleep, the rule and its
// physics; engine/docs/explicit-vehicles-in-world.md §10). Included by
// explicit_beams.slang.
//
// Off (gP.sleepRec 0, the default: no room was allocated) nothing here runs. On:
// gMisc[sleepRec] = (per-vehicle record base, dwell steps, speed bits, reach bits),
// gMisc[sleepRec + 1] = (gAcc base of 2 flags per vehicle, gBeams base of 1 anchor per
// node, 0, 0); per vehicle v its record gMisc[base + v] = (still steps, flags: 1 asleep,
// 2 anchored, 0, 0). A host wakes a vehicle by writing its record (0, 0, 0, 0); the
// World's coupling plan writes it too (vehicle_coupling.slang cpl_plan).
//
// Awake-indexed dispatch (gP.sleepPad0 = 1, "listed"): explicit_beams_sleep_list writes,
// every frame before it, the awake vehicles in vehicle order at gNext[sleepList()] (count
// first) and the frame's indirect arguments; the frame's workgroups past the count step
// the contact groups. Sleeping vehicles are not dispatched at all.
//
// Every barrier here is under gP.sleepRec != 0 (a uniform) and each function is called
// by every lane: workgroup-uniform control flow. The paths are their own functions,
// called behind that uniform test (Metal's codegen of the shipped paths stays as it
// was).

static const uint SLEEP_ASLEEP = 1u;
static const uint SLEEP_ANCHORED = 2u;

uint vvMember(uint off, uint m);

uint4 sleepHdr() { return gMisc[gP.sleepRec]; }
// gNext word of the awake list (count, then the vehicles).
uint sleepList() { return gMisc[gP.sleepRec + 1u].z; }
uint sleepVRec(uint v) { return gMisc[gP.sleepRec].x + v; }
uint sleepFlags(uint v) { return gMisc[gP.sleepRec + 1u].x + 2u * v; }
uint sleepAnchor(uint gi) { return gMisc[gP.sleepRec + 1u].y + gi; }

// A node faster than the speed bound along an axis (NaN: moving).
bool sleepMoving(uint gi, float speed)
{
    float3 u = abs(gNodes[gi * 3u + 1u].xyz);
    return !(u.x <= speed && u.y <= speed && u.z <= speed);
}

// At the frame's start: vehicle v asleep (every lane; its frame then returns).
bool sleepSkip(uint v, uint lane)
{
    sync();
    if (lane == 0u) gsUniformLanes = gMisc[sleepVRec(v)].y & SLEEP_ASLEEP;
    sync();
    return gsUniformLanes != 0u;
}

// Listed dispatch: workgroup `g`'s vehicle (or vehicles + its contact group past the
// awake count), every lane, uniform.
uint sleepPick(uint g, uint lane)
{
    sync();
    if (lane == 0u) gsUniformLanes = gNext[sleepList()];
    sync();
    uint n = gsUniformLanes;
    sync();
    if (lane == 0u) gsUniformLanes = g < n ? gNext[sleepList() + 1u + g] : gP.vehicles + (g - n);
    sync();
    return gsUniformLanes;
}

// After a step's node phase: whether one of this lane's nodes of vehicle v is moving
// (or `moving` already).
bool sleepTrack(uint nodeOff, uint nodeCount, uint lane, bool moving)
{
    float speed = asfloat(sleepHdr().z);
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop && !moving; i += LANES)
        moving = sleepMoving(nodeOff + i, speed);
    return moving;
}

// A lane found vehicle v moving this frame.
void sleepMark(uint v, bool moving)
{
    if (moving) gAcc[sleepFlags(v)].store(1);
}

// The frame-end rule for vehicle v (after every lane's sleepMark for it): its still count
// and anchor; returns (lane 0) whether the count reached the dwell. A solo vehicle
// (`solo`) then falls asleep; a group's members are decided together.
bool sleepSettle(uint v, uint lane, uint steps, bool solo)
{
    uint4 h = sleepHdr();
    float reach = asfloat(h.w);
    uint4 r0 = gStatic[v * 4u];
    uint acc = sleepFlags(v);
    sync();
    bool moving = gAcc[acc].load() != 0;
    uint4 rec = gMisc[h.x + v];
    bool anchored = !moving && (rec.y & SLEEP_ANCHORED) != 0u;
    if (anchored)
    {
        for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 d = abs(ssub3(gNodes[gi * 3u].xyz, gBeams[sleepAnchor(gi)].xyz));
            if (!(d.x <= reach && d.y <= reach && d.z <= reach)) gAcc[acc + 1u].store(1);
        }
    }
    sync();
    bool displaced = !moving && (!anchored || gAcc[acc + 1u].load() != 0);
    if (displaced)
    {
        for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            gBeams[sleepAnchor(gi)] = float4(gNodes[gi * 3u].xyz, 0.0);
        }
    }
    sync();
    bool reached = false;
    if (lane == 0u)
    {
        gAcc[acc].store(0);
        gAcc[acc + 1u].store(0);
        uint still = (moving || displaced) ? 0u : min(rec.x + steps, 0x7FFFFFFFu);
        reached = still >= h.y;
        uint flags = (moving ? 0u : SLEEP_ANCHORED) | ((solo && reached) ? SLEEP_ASLEEP : 0u);
        gMisc[h.x + v] = uint4(still, flags, 0u, 0u);
    }
    return reached;
}

// A contact group's frame start (its members unsorted at off): an awake member wakes
// the sleeping ones; true (every lane) when every member sleeps (the group returns).
bool sleepGroupSkip(uint off, uint k, uint lane)
{
    sync();
    if (lane == 0u)
    {
        uint base = sleepHdr().x;
        bool awake = false;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
            awake = awake || (gMisc[base + gNext[gP.vvMembers + off + m]].y & SLEEP_ASLEEP) == 0u;
        if (awake)
        {
            for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
            {
                uint r = base + gNext[gP.vvMembers + off + m];
                if ((gMisc[r].y & SLEEP_ASLEEP) != 0u) gMisc[r] = uint4(0u, 0u, 0u, 0u);
            }
        }
        gsUniformLanes = awake ? 0u : 1u;
    }
    sync();
    return gsUniformLanes != 0u;
}

// After a group step: this lane's moving members as bits (at most 64 members).
uint2 sleepTrackMembers(uint off, uint k, uint lane, uint2 mask)
{
    float speed = asfloat(sleepHdr().z);
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
    {
        uint bit = 1u << (m & 31u);
        if (((m < 32u ? mask.x : mask.y) & bit) != 0u) continue;
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        bool moving = false;
        for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop && !moving; i += LANES)
            moving = sleepMoving(r0.x + i, speed);
        if (moving)
        {
            if (m < 32u) mask.x |= bit;
            else mask.y |= bit;
        }
    }
    return mask;
}

// A contact group's frame end: each member's rule; the members fall asleep together
// when every one reached the dwell.
void sleepGroupEnd(uint off, uint k, uint lane, uint2 mask, uint steps)
{
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        sleepMark(vvMember(off, m), ((m < 32u ? mask.x : mask.y) & (1u << (m & 31u))) != 0u);
    bool all = true;
    for (uint m = 0u, mStop = EB_END_UNIFORM(m, k, 1u); m < mStop; m++)
    {
        bool reached = sleepSettle(vvMember(off, m), lane, steps, false);
        all = all && reached;
    }
    if (lane == 0u && all)
    {
        uint base = sleepHdr().x;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) gMisc[base + vvMember(off, m)].y |= SLEEP_ASLEEP;
    }
}

// --- The awake list (one workgroup, every frame before the frame's dispatch): the awake
// vehicles in vehicle order, then the indirect arguments: without vehicle contact the
// frame over the awake vehicles (words 8..10); with it, as vvArgs wrote them at the last
// plan, the awake vehicles in place of the fleet (word 3: the plan's groups).
#ifdef EB_SLEEP_LIST
// (gArgs: the vehicle-contact plan's indirect arguments, binding 10, declared there.)
groupshared uint gsSleepScan[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_sleep_list(uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint nv = gP.vehicles;
    uint base = sleepHdr().x;
    uint list = sleepList();
    uint run = 0u;
    for (uint c = 0u, cStop = EB_END_UNIFORM(c, (nv + 255u) / 256u, 1u); c < cStop; c++)
    {
        uint v = 256u * c + lane;
        bool awake = v < nv && (gMisc[base + v].y & SLEEP_ASLEEP) == 0u;
        gsSleepScan[lane] = awake ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            uint r = run;
            for (uint k = 0u; k < 256u; k++)
            {
                uint x = gsSleepScan[k];
                gsSleepScan[k] = r;
                r += x;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (awake) gNext[list + 1u + gsSleepScan[lane]] = v;
        run = gsSleepScan[255] + ((256u * c + 255u < nv && (gMisc[base + 256u * c + 255u].y & SLEEP_ASLEEP) == 0u) ? 1u : 0u);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        gNext[list] = run;
        gArgs[8] = run;
        gArgs[9] = 1u;
        gArgs[10] = 1u;
        if (gP.vvOn != 0u)
        {
            uint groups = gArgs[3];
            gArgs[0] = groups == 0u ? run : 0u;
            gArgs[1] = 1u;
            gArgs[2] = 1u;
            gArgs[4] = groups == 0u ? 0u : run + groups;
            gArgs[5] = 1u;
            gArgs[6] = 1u;
        }
    }
    ebLoopCapFlush();
}
#endif
