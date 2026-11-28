// Bucket slots of the wide passes' grid (a fuller bucket overflows to a list).
static const uint WIDE_SLOTS = 16u;

// --- Wide passes: one thread per element across every vehicle, a dispatch per
// pass, for large vehicles (the whole GPU on one vehicle). Per frame:
// wide_begin, wide_rebase, then per step wide_nodes, wide_beams, wide_tris
// (each step binding its own parameter copy, whose pad2 is the step), then
// wide_end, wide_close. Same per-element functions as
// explicit_beams_frame, in the same order within a step; the grid offsets and
// tested lists live in buffers instead of workgroup memory, and the grid is
// fixed bucket slots (no scan). Every vehicle in
// a wide dispatch has run the same number of steps (the host checks).

// The wide passes' vehicle table, in constant memory: per vehicle its 4 static
// records, then per vehicle (origin bits xyz, steps run) copied from gMisc after
// the frame's rebase.
static const uint WIDE_MAX_VEHICLES = 256u;
struct VehicleTable
{
    uint4 rec[WIDE_MAX_VEHICLES * 5u];
};
[[vk::binding(9, 0)]] ConstantBuffer<VehicleTable> gVt;

struct Vehicle
{
    uint nodeOff, nodeCount, beamOff, beamCount, triOff, triCount, boxOff, boxCount;
    uint candOff, candCount, groupOff, groupCount, flags;
    float ground, range, cell;
};

Vehicle vehicleAt(uint v)
{
    uint4 r0 = gVt.rec[v * 4u];
    uint4 r1 = gVt.rec[v * 4u + 1u];
    uint4 r2 = gVt.rec[v * 4u + 2u];
    uint4 r3 = gVt.rec[v * 4u + 3u];
    Vehicle x;
    x.nodeOff = r0.x; x.nodeCount = r0.y; x.beamOff = r0.z; x.beamCount = r0.w;
    x.triOff = r1.x; x.triCount = r1.y; x.boxOff = r1.z; x.boxCount = r1.w;
    x.candOff = r2.x; x.candCount = r2.y; x.groupOff = r2.z; x.groupCount = r2.w;
    x.flags = r3.x; x.ground = asfloat(r3.y); x.range = asfloat(r3.z); x.cell = asfloat(r3.w);
    return x;
}

// The step within the frame (each step's dispatches bind their own parameter
// copy) and a vehicle's step.
uint wideStep() { return gP.pad2; }
uint vehicleStep(uint v) { return gVt.rec[WIDE_MAX_VEHICLES * 4u + v].w + wideStep(); }
float3 vehicleOrigin(uint v) { return asfloat(gVt.rec[WIDE_MAX_VEHICLES * 4u + v].xyz); }

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_begin(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= gP.vehicles) return ebLoopCapFlush();
    // (Vehicle sleep: a sleeping vehicle is not stepped; no rebase.)
    if (sleepAsleep(v))
    {
        gMisc[gP.wideMisc + 1u + v] = uint4(0u, 0u, 0u, 0u);
        return ebLoopCapFlush();
    }
    Vehicle x = vehicleAt(v);
    float3 first = gNodes[x.nodeOff * 3u].xyz;
    float3 shift = length(first) > REBASE_RADIUS ? first : float3(0.0, 0.0, 0.0);
    gMisc[gP.wideMisc + 1u + v] = uint4(asuint(shift), 0u);
    // The coupling's proxies with the vehicle (cplRebase, as the frame's rebase: every
    // lane's share on this thread).
    if (length(first) > REBASE_RADIUS)
    {
        for (uint l = 0u; l < LANES; l++) cplRebase(v, l, first);
    }
    gAcc[gP.testedBase + v * 2u].store(0);
    gAcc[gP.testedBase + v * 2u + 1u].store(0);
    ebLoopCapFlush();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_rebase(uint3 id : SV_DispatchThreadID)
{
    uint gi = id.x;
    if (gi >= gP.totalNodes) return;
    uint v = gStatic[gP.nodeInfoBase + gi].w;
    float3 shift = asfloat(gMisc[gP.wideMisc + 1u + v].xyz);
    if (all(shift == float3(0.0, 0.0, 0.0))) return;
    float4 P = gNodes[gi * 3u];
    gNodes[gi * 3u] = float4(P.xyz - shift, P.w);
    if (gi == gStatic[v * 4u].x)
    {
        uint4 m = gMisc[v];
        gMisc[v] = uint4(asuint(asfloat(m.xyz) + shift), m.w);
    }
}

// Nodes, then group slots, grid counts (and overflow and this step's tested counts).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_nodes(uint3 id : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    // The coupling's header for the nodes' proxy contacts (cplContacts reads it from
    // workgroup memory; no sphere or node-range copies there: every read from the buffers).
    if (lid.x == 0u)
    {
        coupleBegin();
        cplGsClear();
    }
    sync();
    uint i = id.x;
    if (i < gP.totalNodes)
    {
        uint4 info = gStatic[gP.nodeInfoBase + i];
        uint v = info.w;
        if (sleepAsleep(v)) return ebLoopCapFlush();
        Vehicle x = vehicleAt(v);
        float3 origin = vehicleOrigin(v);
        bool centred = gP.centredDv != 0u && (x.flags & FLAG_CENTRED) != 0u;
        if (centred && (x.flags & FLAG_CENTRED_OBJECTIVE) != 0u) centredKeep(i);
        float3 v0 = centred ? gNodes[i * 3u + 1u].xyz : float3(0.0, 0.0, 0.0);
        nodeUpdate(i, info, v, i - x.nodeOff, vehicleStep(v), origin, x.flags, x.ground, x.boxOff, x.boxCount,
                   x.beamOff, x.groupOff, x.groupCount, true);
        if (centred) centredDelta(i, v0);
        // Vehicle sleep: this step's motion (with centred damping, after its correction:
        // wide_centred_nodes), as the frame's sleepTrack after its node phase.
        if (gP.sleepRec != 0u && !centred) sleepMarkNode(v, i);
        return ebLoopCapFlush();
    }
    i -= gP.totalNodes;
    // (Vehicle sleep, a lone vehicle: asleep, nothing of its step runs.)
    if (sleepAsleep(0u)) return ebLoopCapFlush();
    if (i < gP.groupTotal)
    {
        // Every vehicle has the same step parity.
        gGroups[groupSlot(vehicleStep(0u) & 1u, 0u, i)].store(NO_GROUP);
        return ebLoopCapFlush();
    }
    i -= gP.groupTotal;
    if (i < gP.vehicles * GRID_BUCKETS)
    {
        // (With the reused pair list, PERF_WIDE_LIST, these words hold its state between
        // dispatches: soloStepSave.)
        if ((gP.perfFlags & PERF_WIDE_LIST) == 0u) gAcc[gP.gridCountBase + i].store(0);
        if (i < gP.vehicles)
        {
            gAcc[gP.overflowBase + i].store(0);
            gAcc[gP.testedBase + i * 2u + (vehicleStep(i) & 1u)].store(0);
            gAcc[hitAcc(i)].store(0);
            gAcc[hitAcc(i) + 1u].store(0);
            clipVmaxClear(i, vehicleStep(i) & 1u);
        }
    }
    ebLoopCapFlush();
}

// Centred damping (§39) in the wide passes, after wide_nodes (which kept each
// node's Δv): each record's correction into its entries, then each node's sum.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_centred_beams(uint3 id : SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= gP.totalBeams) return;
    uint v = gStatic[gP.beamBase + j * 3u].z >> 16;
    if (sleepAsleep(v)) return;
    uint flagsV = vehicleAt(v).flags;
    if ((flagsV & FLAG_CENTRED) != 0u) centredBeam(j, (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_centred_nodes(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= gP.totalNodes) return ebLoopCapFlush();
    uint v = gStatic[gP.nodeInfoBase + i].w;
    if (sleepAsleep(v)) return ebLoopCapFlush();
    Vehicle x = vehicleAt(v);
    if ((x.flags & FLAG_CENTRED) != 0u)
    {
        centredNode(i, x.beamOff, x.groupOff, x.groupCount, (vehicleStep(v) + 1u) & 1u);
        // (Vehicle sleep: the node's motion after the correction.)
        if (gP.sleepRec != 0u) sleepMarkNode(v, i);
    }
    ebLoopCapFlush();
}

// perfFlags bit 2048 (explicit_plan::PERF_WIDE_GEO, with PERF_WIDE_LIST): the tested
// triangles' step geometry (triGeoFill) on threads of the beams' dispatch, a triangle a
// thread across the GPU, in place of the list kernel's lanes (one workgroup: 528
// triangles' records on one core, the kernel's longest phase). What triGeoFill reads
// (the nodes' positions and velocities, the fastest node's speed, the triangles' flags)
// is the node phase's and nothing writes it until the triangles' tests are over, so
// the records are the list kernel's bit for bit; they are filled for the triangles the
// list kernel would fill: those its throttle update will leave tested this step
// (triangleFinishPeek: triangleFinish's result from the words the step before left,
// which nothing writes between its hits and this step's list kernel), every triangle
// of a vehicle past TRI_LIST. (A step on the list kernel's few-pairs path reads no
// record: these are then written and not read.)
static const uint PERF_WIDE_GEO = 2048u;
int triangleFinishPeek(uint gt)
{
    uint4 th = gMisc[gP.vehicles + gt];
    int wait = asint(th.x);
    int idle = asint(th.y);
    if (wait > 0) return wait - 1;
    if (idle > 0) wait = min(idle, 12);
    return th.w != 0u ? -20000 : min(wait + 1, throttleCap(gt));
}
void wideGeoThread(uint gt)
{
    if (gt >= gP.totalTris || gP.triGeo == 0u) return;
    uint v = gStatic[gP.triBase + gt].w;
    if (sleepAsleep(v)) return;
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    if (r2.y == 0u) return;
    // (The list kernel's origin, parity and range: the same words.)
    uint4 misc = gMisc[v];
    uint s = gP.pad2;
    if (r1.y <= TRI_LIST)
    {
        int w = s > 0u ? triangleFinishPeek(gt) : asint(gMisc[gP.vehicles + gt].x);
        if (w > 0) return;
    }
    triGeoFill(gt, asfloat(misc.xyz), selfRangeOf(v, asfloat(gStatic[v * 4u + 3u].z)), (misc.w + s) & 1u);
}

// Beams, then grid filling (candidates), then triangles' throttle and tested list.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_beams(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    // The node grab, after this step's wide_nodes (one thread).
    if (i == 0u) grabApplyWide();
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        if (sleepAsleep(v)) return ebLoopCapFlush();
        Vehicle x = vehicleAt(v);
        beamUpdate(i, i - x.beamOff, (x.flags & FLAG_FAST_INV_SQRT) != 0u, x.groupOff, vehicleStep(v) & 1u);
        return ebLoopCapFlush();
    }
    // (PERF_WIDE_LIST: the grid and the triangles' throttles are the list kernel's; with
    // PERF_WIDE_GEO the tested triangles' geometry here, a thread a triangle.)
    if ((gP.perfFlags & PERF_WIDE_LIST) != 0u)
    {
        if ((gP.perfFlags & PERF_WIDE_GEO) != 0u) wideGeoThread(i - gP.totalBeams);
        return ebLoopCapFlush();
    }
    i -= gP.totalBeams;
    if (i < gP.totalCands)
    {
        uint pos = gLists[gP.candTable + i];
        uint h = gLists[pos];
        uint v = gStatic[gP.nodeInfoBase + h].w;
        float cell = asfloat(gVt.rec[v * 4u + 3u].w);
        if (cell > 0.0 && !sleepAsleep(v))
        {
            float3 p = gNodes[h * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            uint rank = uint(gAcc[gP.gridCountBase + v * GRID_BUCKETS + b].add(1));
            if (rank < WIDE_SLOTS)
                gBeams[gP.slotBase16 + (v * GRID_BUCKETS + b) * WIDE_SLOTS + rank] = float4(p, asfloat(h));
            else
            {
                uint o = uint(gAcc[gP.overflowBase + v].add(1));
                gNext[gVt.rec[v * 4u + 2u].x + o] = h;
            }
        }
        return ebLoopCapFlush();
    }
    i -= gP.totalCands;
    if (i < gP.totalTris)
    {
        // The last step's throttle, and this step's tested list (independent of the
        // beams).
        uint v = gStatic[gP.triBase + i].w;
        Vehicle x = vehicleAt(v);
        if (x.candCount == 0u || sleepAsleep(v)) return ebLoopCapFlush();
        int w = wideStep() > 0u ? triangleFinish(i) : asint(gMisc[gP.vehicles + i].x);
        if (w <= 0)
        {
            uint k = uint(gAcc[gP.testedBase + v * 2u + (vehicleStep(v) & 1u)].add(1));
            gNext[gP.testedListBase + x.triOff + k] = i - x.triOff;
        }
    }
    ebLoopCapFlush();
}

// The node grab (grabApply) in the wide passes: the grabbed node's vehicle from its
// node record.
void grabApplyWide()
{
    uint4 g = gMisc[grabRec()];
    if (g.x == 0u || g.y >= gP.totalNodes) return;
    uint v = gStatic[gP.nodeInfoBase + g.y].w;
    if (sleepAsleep(v)) return;
    Vehicle x = vehicleAt(v);
    grabApply(x.nodeOff, x.nodeCount, vehicleOrigin(v));
}

// triangleGrid with the offsets in gAcc.
void triangleGridWide(uint gt, uint sub, uint v, float3 origin, float r, float cell, uint candOff, uint candCount,
                      uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    TriGeo t = triangleGeo(gt, origin, r, parity);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub, kStop = EB_END(94, k, candCount, GRID_SUBS); k < kStop; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub, iStop = EB_END(95, i, cells, GRID_SUBS); i < iStop; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        for (uint j = 0u, jStop = EB_END(96, j, i, 1u); j < jStop; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        uint n = min(uint(gAcc[gP.gridCountBase + v * GRID_BUCKETS + b].load()), WIDE_SLOTS);
        uint base = gP.slotBase16 + (v * GRID_BUCKETS + b) * WIDE_SLOTS;
        for (uint e = 0u, eStop = EB_END(97, e, n, 1u); e < eStop; e++)
        {
            float4 E = gBeams[base + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
    // Candidates past a full bucket: every triangle tests them (its share of lanes).
    uint over = uint(gAcc[gP.overflowBase + v].load());
    for (uint k = sub, kStop = EB_END(98, k, over, GRID_SUBS); k < kStop; k += GRID_SUBS)
        tryContact(t, gNext[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
}

// GRID_SUBS lanes per listed triangle (a vehicle's k-th listed triangle on the
// lanes of its k-th triangle record).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_tris(uint3 id : SV_DispatchThreadID)
{
    uint gt = id.x / GRID_SUBS;
    uint sub = id.x % GRID_SUBS;
    if (gt >= gP.totalTris) return ebLoopCapFlush();
    uint v = gStatic[gP.triBase + gt].w;
    Vehicle x = vehicleAt(v);
    if (x.candCount == 0u || sleepAsleep(v)) return ebLoopCapFlush();
    uint step = vehicleStep(v);
    uint k = gt - x.triOff;
    if (k >= uint(gAcc[gP.testedBase + v * 2u + (step & 1u)].load())) return ebLoopCapFlush();
    uint t = x.triOff + gNext[gP.testedListBase + x.triOff + k];
    float3 origin = vehicleOrigin(v);
    if (x.cell > 0.0)
        triangleGridWide(t, sub, v, origin, x.range, x.cell, x.candOff, x.candCount, x.beamOff, x.groupOff,
                         x.groupCount, step & 1u);
    else
        triangleTest(t, sub, GRID_SUBS, origin, x.range, x.candOff, x.candCount, x.beamOff, x.groupOff,
                     x.groupCount, step & 1u);
    ebLoopCapFlush();
}

