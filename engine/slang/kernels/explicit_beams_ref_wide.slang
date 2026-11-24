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
    if (v >= gP.vehicles) return;
    Vehicle x = vehicleAt(v);
    float3 first = gNodes[x.nodeOff * 3u].xyz;
    float3 shift = length(first) > REBASE_RADIUS ? first : float3(0.0, 0.0, 0.0);
    gMisc[gP.wideMisc + 1u + v] = uint4(asuint(shift), 0u);
    gAcc[gP.testedBase + v * 2u].store(0);
    gAcc[gP.testedBase + v * 2u + 1u].store(0);
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
void explicit_beams_wide_nodes(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < gP.totalNodes)
    {
        uint4 info = gStatic[gP.nodeInfoBase + i];
        uint v = info.w;
        Vehicle x = vehicleAt(v);
        float3 origin = vehicleOrigin(v);
        nodeUpdate(i, info, v, i - x.nodeOff, vehicleStep(v), origin, x.flags, x.ground, x.boxOff, x.boxCount,
                   x.beamOff, x.groupOff, x.groupCount, true);
        return;
    }
    i -= gP.totalNodes;
    if (i < gP.groupTotal)
    {
        // Every vehicle has the same step parity.
        gGroups[groupSlot(vehicleStep(0u) & 1u, 0u, i)].store(NO_GROUP);
        return;
    }
    i -= gP.groupTotal;
    if (i < gP.vehicles * GRID_BUCKETS)
    {
        gAcc[gP.gridCountBase + i].store(0);
        if (i < gP.vehicles)
        {
            gAcc[gP.overflowBase + i].store(0);
            gAcc[gP.testedBase + i * 2u + (vehicleStep(i) & 1u)].store(0);
            gAcc[hitAcc(i)].store(0);
            gAcc[hitAcc(i) + 1u].store(0);
        }
    }
}

// Beams, then grid filling (candidates), then triangles' throttle and tested list.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_beams(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        Vehicle x = vehicleAt(v);
        beamUpdate(i, i - x.beamOff, (x.flags & FLAG_FAST_INV_SQRT) != 0u, x.groupOff, vehicleStep(v) & 1u);
        return;
    }
    i -= gP.totalBeams;
    if (i < gP.totalCands)
    {
        uint pos = gLists[gP.candTable + i];
        uint h = gLists[pos];
        uint v = gStatic[gP.nodeInfoBase + h].w;
        float cell = asfloat(gVt.rec[v * 4u + 3u].w);
        if (cell > 0.0)
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
        return;
    }
    i -= gP.totalCands;
    if (i < gP.totalTris)
    {
        // The last step's throttle, and this step's tested list (independent of the
        // beams).
        uint v = gStatic[gP.triBase + i].w;
        Vehicle x = vehicleAt(v);
        if (x.candCount == 0u) return;
        int w = wideStep() > 0u ? triangleFinish(i) : asint(gMisc[gP.vehicles + i].x);
        if (w <= 0)
        {
            uint k = uint(gAcc[gP.testedBase + v * 2u + (vehicleStep(v) & 1u)].add(1));
            gNext[gP.testedListBase + x.triOff + k] = i - x.triOff;
        }
    }
}

// triangleGrid with the offsets in gAcc.
void triangleGridWide(uint gt, uint sub, uint v, float3 origin, float r, float cell, uint candOff, uint candCount,
                      uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    TriGeo t = triangleGeo(gt, origin, r);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub; i < cells; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        for (uint j = 0u; j < i; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        uint n = min(uint(gAcc[gP.gridCountBase + v * GRID_BUCKETS + b].load()), WIDE_SLOTS);
        uint base = gP.slotBase16 + (v * GRID_BUCKETS + b) * WIDE_SLOTS;
        for (uint e = 0u; e < n; e++)
        {
            float4 E = gBeams[base + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
    // Candidates past a full bucket: every triangle tests them (its share of lanes).
    uint over = uint(gAcc[gP.overflowBase + v].load());
    for (uint k = sub; k < over; k += GRID_SUBS)
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
    if (gt >= gP.totalTris) return;
    uint v = gStatic[gP.triBase + gt].w;
    Vehicle x = vehicleAt(v);
    if (x.candCount == 0u) return;
    uint step = vehicleStep(v);
    uint k = gt - x.triOff;
    if (k >= uint(gAcc[gP.testedBase + v * 2u + (step & 1u)].load())) return;
    uint t = x.triOff + gNext[gP.testedListBase + x.triOff + k];
    float3 origin = vehicleOrigin(v);
    if (x.cell > 0.0)
        triangleGridWide(t, sub, v, origin, x.range, x.cell, x.candOff, x.candCount, x.beamOff, x.groupOff,
                         x.groupCount, step & 1u);
    else
        triangleTest(t, sub, GRID_SUBS, origin, x.range, x.candOff, x.candCount, x.beamOff, x.groupOff,
                     x.groupCount, step & 1u);
}

// After the triangles: a workgroup per vehicle resolves its deferred breaks and
// applies its self-collision hits in order (selfApply).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_hits(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    Vehicle x = vehicleAt(v);
    uint parity = vehicleStep(v) & 1u;
    if (lane == 0u)
    {
        resolvePending(v, parity);
        gsUniformHits = uint(gAcc[hitAcc(v)].load()) > 0u ? 1u : 0u;
        gsUniformOpen = 0u;
    }
    sync();
    uint on = gsUniformHits;
    if (on != 0u)
        selfApply(v, true, lane, 256u, lane, vehicleOrigin(v), x.range, x.beamOff, x.groupOff, x.groupCount, parity);
}

// After the last step: its triangles' throttle and its group breaks.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_end(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (gP.steps == 0u) return;
    if (i < gP.totalTris)
    {
        uint v = gStatic[gP.triBase + i].w;
        if (gStatic[v * 4u + 2u].y > 0u) triangleFinish(i);
        return;
    }
    i -= gP.totalTris;
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        Vehicle x = vehicleAt(v);
        if (x.groupCount > 0u) groupDrops(i, x.beamOff, x.groupOff, (vehicleStep(v) - 1u) & 1u, true);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_close(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= gP.vehicles) return;
    uint4 m = gMisc[v];
    gMisc[v] = uint4(m.xyz, m.w + wideStep());
}

// --- Vehicle–vehicle contact (engine/docs/explicit-beams-vehicle-contact-spec.md,
// CPU reference engine_softbody::explicit_contact) ---
//
// A frame with vehicle contact on (more than one vehicle):
//   explicit_beams_vv_bounds  one workgroup per vehicle: its box over the node
//                             world positions; its frame bounds (the box grown by
//                             the travel of its fastest node over the steps the plan
//                             covers, twice, plus a margin) and the padded frame box
//                             (+ 0.05 m);
//                             the box into a hashed vehicle grid (or the big list)
//   explicit_beams_vv_pairs   one thread per vehicle: pairs of overlapping padded
//                             frame boxes
//   explicit_beams_vv_group   one workgroup: connected components of the pairs
//                             (label propagation), the solo list, the groups'
//                             member lists, the indirect dispatch arguments
//   (the plan passes run when the last plan's steps are used up: it covers at
//   least VV_PLAN_STEPS on the host, so at 120 fps every other frame)
//   explicit_beams_frame      (indirect) the solo vehicles, a whole frame each; a
//                             node leaving its frame bounds is reported (loud)
//   explicit_beams_contact    (indirect) one workgroup per group: every step, each
//                             member's step in vehicle order, then the contact pass
//                             (owners in vehicle order)
// Two vehicles whose padded frame boxes do not overlap, with every node inside
// its frame bounds, cannot be partners in any step of the frame: the solo
// vehicles need no contact pass. A node outside its bounds is reported to the
// host, which fails (the frame may have missed a contact).

static const int VV_INT_MAX = 2147483647;
static const int VV_INT_MIN = -2147483647 - 1;
static const float VV_BOX_MARGIN = 0.05;
static const uint VV_SLOTS = 8u;
static const uint VV_MAX_SPAN = 3u;
static const uint VV_MAX_CELLS = 27u;
static const float VV_APPROACH_SQ = 16.0;
// Members a contact group may have (one workgroup steps them one after another);
// a larger group's vehicles run alone that frame and the host fails.
static const uint VV_MAX_GROUP = 64u;
static const int VV_MAX_WAIT = 12;

[[vk::binding(10, 0)]] RWStructuredBuffer<uint> gArgs;

groupshared Atomic<int> gsVvBox[12];
groupshared uint gsVvA[256];
groupshared uint gsVvB[256];
groupshared Atomic<uint> gsVvOffered[2];
groupshared Atomic<uint> gsVvListed[2];
groupshared Atomic<uint> gsVvCands[2];
groupshared Atomic<uint> gsVvHits[2];
groupshared uint gsVvReset[2];

// Box coordinates as integers ordered like the floats (atomic minimum, maximum).
int vvKey(float f)
{
    int i = asint(f);
    return i >= 0 ? i : (i ^ 0x7FFFFFFF);
}
float vvUnkey(int k) { return asfloat(k >= 0 ? k : (k ^ 0x7FFFFFFF)); }

uint vvBoxKey(uint v, uint parity) { return gP.vvAcc + (v * 2u + parity) * 6u; }
uint vvCpOff(uint v) { return gStatic[gP.vvStatic + v].x; }
uint vvCpCount(uint v) { return gStatic[gP.vvStatic + v].y; }

bool vvOverlap(float3 alo, float3 ahi, float3 blo, float3 bhi)
{
    return !(any(ahi < blo) || any(bhi < alo));
}

uint vvHash(int3 c)
{
    return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u) & gP.vvGridMask;
}

// The vehicle grid cells of a padded frame box: false when it spans too many (or
// is not finite).
bool vvCells(float3 lo, float3 hi, out int3 c0, out int3 span)
{
    float cell = asfloat(gP.vvCell);
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    c0 = int3(0, 0, 0);
    span = int3(0, 0, 0);
    if (!(all(fs <= float(VV_MAX_SPAN)) && all(fs >= 1.0) && fs.x * fs.y * fs.z <= float(VV_MAX_CELLS)))
        return false;
    c0 = int3(f0);
    span = int3(fs);
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_bounds(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    uint4 r0 = gStatic[v * 4u];
    float3 origin = asfloat(gMisc[v].xyz);
    // Keys: position lo xyz, hi xyz; velocity lo xyz, hi xyz.
    if (lane < 12u) gsVvBox[lane].store((lane % 6u) < 3u ? VV_INT_MAX : VV_INT_MIN);
    sync();
    float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 hi = -lo;
    float3 vlo = float3(0.0, 0.0, 0.0);
    float3 vhi = float3(0.0, 0.0, 0.0);
    bool has = false;
    for (uint i = lane; i < r0.y; i += 256u)
    {
        uint gi = r0.x + i;
        float3 x = nf3(origin + gNodes[gi * 3u].xyz);
        float3 vel = gNodes[gi * 3u + 1u].xyz;
        lo = min(lo, x);
        hi = max(hi, x);
        vlo = min(vlo, vel);
        vhi = max(vhi, vel);
        has = true;
    }
    if (has)
    {
        gsVvBox[0].min(vvKey(lo.x));
        gsVvBox[1].min(vvKey(lo.y));
        gsVvBox[2].min(vvKey(lo.z));
        gsVvBox[3].max(vvKey(hi.x));
        gsVvBox[4].max(vvKey(hi.y));
        gsVvBox[5].max(vvKey(hi.z));
        gsVvBox[6].min(vvKey(vlo.x));
        gsVvBox[7].min(vvKey(vlo.y));
        gsVvBox[8].min(vvKey(vlo.z));
        gsVvBox[9].max(vvKey(vhi.x));
        gsVvBox[10].max(vvKey(vhi.y));
        gsVvBox[11].max(vvKey(vhi.z));
    }
    sync();
    if (lane == 0u)
    {
        float3 blo = float3(vvUnkey(gsVvBox[0].load()), vvUnkey(gsVvBox[1].load()), vvUnkey(gsVvBox[2].load()));
        float3 bhi = float3(vvUnkey(gsVvBox[3].load()), vvUnkey(gsVvBox[4].load()), vvUnkey(gsVvBox[5].load()));
        // Each face moves out by twice the farthest a node moving that way at its
        // velocity covers over the plan, plus the margin (min/max start at 0).
        float3 vl = float3(vvUnkey(gsVvBox[6].load()), vvUnkey(gsVvBox[7].load()), vvUnkey(gsVvBox[8].load()));
        float3 vh = float3(vvUnkey(gsVvBox[9].load()), vvUnkey(gsVvBox[10].load()), vvUnkey(gsVvBox[11].load()));
        float time = 2.0 * float(gP.vvPlanSteps) * STEP;
        float margin = asfloat(gP.vvMargin);
        float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
        // The positions the last frame left, against the bounds that held for it
        // (a solo frame checks each step's positions at the next step's start).
        if ((gP.vvOn & 4u) == 0u
            && (any(blo < asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz)) || any(bhi > asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz))))
            vvReportOutside(v);
        float3 alo = blo - (-vl * time + float3(margin, margin, margin));
        float3 ahi = bhi + (vh * time + float3(margin, margin, margin));
        float3 plo = alo - m3;
        float3 phi = ahi + m3;
        gMisc[vvRec(v, VV_P_LO)] = uint4(asuint(plo), 0u);
        gMisc[vvRec(v, VV_P_HI)] = uint4(asuint(phi), 0u);
        gMisc[vvRec(v, VV_ALLOWED_LO)] = uint4(asuint(alo), 0u);
        gMisc[vvRec(v, VV_ALLOWED_HI)] = uint4(asuint(ahi), 0u);
        int3 c0;
        int3 span;
        // A box that is not finite (a node at infinity or NaN) partners no one
        // here (a GPU deviation: the specification's rule would partner it with
        // every vehicle and group the whole fleet).
        bool finite = all(abs(plo) < 3.0e38) && all(abs(phi) < 3.0e38);
        bool big = finite && !vvCells(plo, phi, c0, span);
        if (finite && !big)
        {
            for (int x = 0; x < span.x; x++)
                for (int y = 0; y < span.y; y++)
                    for (int z = 0; z < span.z; z++)
                    {
                        uint b = vvHash(c0 + int3(x, y, z));
                        uint slot = uint(gAcc[gP.vvGridCount + b].add(1));
                        if (slot < VV_SLOTS) gNext[gP.vvGrid + b * VV_SLOTS + slot] = v;
                        else big = true;
                    }
        }
        if (big)
        {
            uint at = uint(gAcc[vvCtl(VV_BIGS)].add(1));
            gNext[gP.vvBig + at] = v;
        }
        uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
        gMisc[vvRec(v, VV_FLAGS)] = uint4(finite ? (big ? 1u : 0u) : 2u, 0u, 0u, fl.w);
    }
}

// The frame's dispatch arguments: with no group the solo frame over the vehicles
// (words 0..2), else the frame with the groups over vehicles + groups (words 4..6).
void vvArgs(uint nv, uint groups)
{
    gArgs[0] = groups == 0u ? nv : 0u;
    gArgs[1] = 1u;
    gArgs[2] = 1u;
    gArgs[4] = groups == 0u ? 0u : nv + groups;
    gArgs[5] = 1u;
    gArgs[6] = 1u;
}

void vvEmitPair(uint a, uint b)
{
    uint at = uint(gAcc[vvCtl(VV_PAIRS)].add(1));
    if (at < gP.vvPairCap)
    {
        gNext[gP.vvPairs + 2u * at] = a;
        gNext[gP.vvPairs + 2u * at + 1u] = b;
    }
    else
    {
        gAcc[vvCtl(VV_OVERFLOW)].store(1);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_pairs(uint3 id : SV_DispatchThreadID)
{
    if (id.x < gP.vehicles) vvFindPairs(id.x);
}

// The overlapping pairs of vehicle v's padded frame box (each pair once).
void vvFindPairs(uint v)
{
    float3 plo = asfloat(gMisc[vvRec(v, VV_P_LO)].xyz);
    float3 phi = asfloat(gMisc[vvRec(v, VV_P_HI)].xyz);
    uint kind = gMisc[vvRec(v, VV_FLAGS)].x;
    if (kind == 2u) return;
    if (kind == 1u)
    {
        // Against every vehicle (two big ones once).
        for (uint w = 0u; w < gP.vehicles; w++)
        {
            if (w == v) continue;
            uint kw = gMisc[vvRec(w, VV_FLAGS)].x;
            if (kw == 2u || (kw == 1u && w < v)) continue;
            if (vvOverlap(plo, phi, asfloat(gMisc[vvRec(w, VV_P_LO)].xyz), asfloat(gMisc[vvRec(w, VV_P_HI)].xyz)))
                vvEmitPair(v, w);
        }
        return;
    }
    int3 c0;
    int3 span;
    vvCells(plo, phi, c0, span);
    for (int x = 0; x < span.x; x++)
        for (int y = 0; y < span.y; y++)
            for (int z = 0; z < span.z; z++)
            {
                int3 c = c0 + int3(x, y, z);
                uint b = vvHash(c);
                uint n = min(uint(gAcc[gP.vvGridCount + b].load()), VV_SLOTS);
                for (uint e = 0u; e < n; e++)
                {
                    uint w = gNext[gP.vvGrid + b * VV_SLOTS + e];
                    // Each pair once: the higher index in the lower's thread, in
                    // the first cell both cover; big vehicles find their own.
                    if (w <= v || gMisc[vvRec(w, VV_FLAGS)].x != 0u) continue;
                    float3 wlo = asfloat(gMisc[vvRec(w, VV_P_LO)].xyz);
                    float3 whi = asfloat(gMisc[vvRec(w, VV_P_HI)].xyz);
                    if (!vvOverlap(plo, phi, wlo, whi)) continue;
                    int3 cw;
                    int3 sw;
                    vvCells(wlo, whi, cw, sw);
                    if (all(c == max(c0, cw))) vvEmitPair(v, w);
                }
            }
}

uint vvLabel(uint v) { return uint(gAcc[gP.vvLabel + v].load()); }
uint vvSum(uint g, uint k) { return gP.vvLabel + gP.vehicles + g * 3u + k; }

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_group(uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint nv = gP.vehicles;
    if (gP.vvFusedPairs != 0u)
    {
        for (uint v = lane; v < nv; v += 256u) vvFindPairs(v);
        sync();
    }
    uint pairs = min(uint(gAcc[vvCtl(VV_PAIRS)].load()), gP.vvPairCap);
    // No overlapping frame boxes (the common case): every vehicle is solo.
    if (lane == 0u) gsUniformLanes = pairs;
    sync();
    uint anyPairs = gsUniformLanes;
    sync();
    if (anyPairs == 0u)
    {
        for (uint b = lane; b <= gP.vvGridMask; b += 256u) gAcc[gP.vvGridCount + b].store(0);
        if (lane == 0u)
        {
            vvArgs(nv, 0u);
            gAcc[vvCtl(VV_BIGS)].store(0);
        }
        return;
    }
    for (uint v = lane; v < nv; v += 256u)
    {
        gAcc[gP.vvLabel + v].store(int(v));
        gMisc[vvRec(v, VV_FLAGS)].y = 0u;
    }
    sync();
    for (uint p = lane; p < pairs; p += 256u)
    {
        gMisc[vvRec(gNext[gP.vvPairs + 2u * p], VV_FLAGS)].y = 1u;
        gMisc[vvRec(gNext[gP.vvPairs + 2u * p + 1u], VV_FLAGS)].y = 1u;
    }
    // Connected components: every vehicle of a group ends with its smallest index.
    uint go = 1u;
    for (uint it = 0u; it < 64u; it++)
    {
        if (lane == 0u) gsUniformLanes = 0u;
        sync();
        bool changed = false;
        for (uint p = lane; p < pairs; p += 256u)
        {
            uint a = gNext[gP.vvPairs + 2u * p];
            uint b = gNext[gP.vvPairs + 2u * p + 1u];
            uint la = vvLabel(a);
            uint lb = vvLabel(b);
            if (la != lb)
            {
                int m = int(min(la, lb));
                gAcc[gP.vvLabel + a].min(m);
                gAcc[gP.vvLabel + b].min(m);
                gAcc[gP.vvLabel + la].min(m);
                gAcc[gP.vvLabel + lb].min(m);
                changed = true;
            }
        }
        for (uint v = lane; v < nv; v += 256u)
        {
            uint l = vvLabel(v);
            uint ll = vvLabel(l);
            if (ll < l) gAcc[gP.vvLabel + v].min(int(ll));
        }
        if (changed) gsUniformLanes = 1u;
        sync();
        go = gsUniformLanes;
        sync();
        if (go == 0u) break;
    }
    if (lane == 0u && go != 0u) gAcc[vvCtl(VV_UNCONVERGED)].store(1);
    // Groups (numbered by their smallest vehicle, in no particular order) and the
    // solo list.
    for (uint v = lane; v < nv; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u && vvLabel(v) == v)
            gMisc[vvRec(v, VV_FLAGS)].z = uint(gAcc[vvCtl(VV_GROUPS)].add(1));
    }
    sync();
    for (uint v = lane; v < nv; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u)
        {
            uint g = gMisc[vvRec(vvLabel(v), VV_FLAGS)].z;
            gMisc[vvRec(v, VV_FLAGS)].z = g;
            gAcc[vvSum(g, 0u)].add(1);
            gAcc[vvSum(g, 2u)].add(int(vvCpCount(v)));
        }

    }
    sync();
    // Offsets of the groups' members and contact point runs.
    uint groups = uint(gAcc[vvCtl(VV_GROUPS)].load());
    uint chunk = (groups + 255u) / 256u;
    uint g0 = min(lane * chunk, groups);
    uint g1 = min(g0 + chunk, groups);
    uint sm = 0u;
    uint sp = 0u;
    for (uint g = g0; g < g1; g++)
    {
        sm += uint(gAcc[vvSum(g, 0u)].load());
        sp += uint(gAcc[vvSum(g, 2u)].load());
    }
    gsVvA[lane] = sm;
    gsVvB[lane] = sp;
    sync();
    if (lane == 0u)
    {
        uint ra = 0u;
        uint rb = 0u;
        for (uint k = 0u; k < 256u; k++)
        {
            uint a = gsVvA[k];
            uint b = gsVvB[k];
            gsVvA[k] = ra;
            gsVvB[k] = rb;
            ra += a;
            rb += b;
        }
    }
    sync();
    uint ra = gsVvA[lane];
    uint rb = gsVvB[lane];
    for (uint g = g0; g < g1; g++)
    {
        uint m = uint(gAcc[vvSum(g, 0u)].load());
        uint pts = uint(gAcc[vvSum(g, 2u)].load());
        // Too many members: none stepped as a group (they run alone; reported).
        gMisc[gP.vvGroups + g] = m <= VV_MAX_GROUP ? uint4(ra, m, rb, pts) : uint4(ra, 0u, rb, 0u);
        ra += m;
        rb += pts;
    }
    sync();
    for (uint v = lane; v < nv; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u)
        {
            uint g = gMisc[vvRec(v, VV_FLAGS)].z;
            if (gMisc[gP.vvGroups + g].y == 0u)
            {
                // Stepped alone (by the solo frame) this frame.
                gMisc[vvRec(v, VV_FLAGS)].y = 0u;
                gAcc[vvCtl(VV_TOO_BIG)].store(1);
                continue;
            }
            uint slot = uint(gAcc[vvSum(g, 1u)].add(1));
            gNext[gP.vvMembers + gMisc[gP.vvGroups + g].x + slot] = v;
        }
    }
    sync();
    if (lane == 0u)
    {
        vvArgs(nv, groups);
    }
    // Clear for the next frame.
    for (uint b = lane; b <= gP.vvGridMask; b += 256u) gAcc[gP.vvGridCount + b].store(0);
    for (uint g = lane; g < groups; g += 256u)
    {
        gAcc[vvSum(g, 0u)].store(0);
        gAcc[vvSum(g, 1u)].store(0);
        gAcc[vvSum(g, 2u)].store(0);
    }
    sync();
    if (lane == 0u)
    {
        gAcc[vvCtl(VV_PAIRS)].store(0);
        gAcc[vvCtl(VV_BIGS)].store(0);
        gAcc[vvCtl(VV_SOLOS)].store(0);
        gAcc[vvCtl(VV_GROUPS)].store(0);
    }
}

