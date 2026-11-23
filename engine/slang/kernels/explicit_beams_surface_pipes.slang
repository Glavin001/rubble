// The surface pass as its own pipelines (EB_SURF_PIPE), for a frame run in segments
// (EB_FRAME_SEG, the frame modules without the surface pass): after segment k of the
// frame (step k up to its surface pass), the prep, then each kind's find, then each
// kind's apply, one dispatch each, a workgroup per vehicle; segment k + 1 continues the
// step. The same calls, in the same order per vehicle, as surfaceStep in the frame
// (explicit_beams_surface.slang), whose barriers become dispatch boundaries: the
// results are the same bit for bit. Small modules compile fast in a browser; the frame
// with the pass in it did not. Included by explicit_beams.slang (after the sleep code).
//
// Each workgroup's step is gMisc[v].w (the frame's first, written at its end) + the
// segment (gP.pad2, the parameter copy). The prep keeps its decisions for the find and
// the apply in segRec(v): .x the hit flag (the next segment's gsHit), .y the pair path
// on, .z the pairs listed, .w the proxies the per-element loops meet.

#ifdef EB_SURF_PIPE

// --- The pipelines of the fast schedules (engine_exec explicit_plan
// FramePlanner::set_surface_fast): each kind's find and apply a thread an element over
// the whole GPU (EB_SURF_SPREAD), sized on the GPU by the prep (EB_SURF_ARGS). ---
// A workgroup a vehicle walks a vehicle's elements LANES at a time, each lane a chain
// (the sedan's 4,251 segments: 17 a lane), and 20 vehicles fill 20 workgroups of a GPU
// that runs hundreds. An element's find and apply read what the prep and the finds left
// and write order-free sums, counts and masks (explicit_beams_surface.slang), so which
// thread takes an element changes no result: here thread (x, lane) of row v takes
// vehicle v's element x · SURF_WG + lane (and, past SURF_X workgroups a vehicle, every
// SURF_X · SURF_WG-th after it: the arguments' clamp), the same call with the same
// arguments as the workgroup's lane made. No barrier and no workgroup memory: the
// coupling's header from the buffer (EB_CPL_DIRECT, explicit_beams_rigid.slang; the
// spheres were read from gNext already), vehicle sleep's flag from its record. A
// workgroup's start with barriers and the header's copies cost more than its elements
// on Metal (20 cars' triangles: 165 workgroups, 63 us a dispatch against 17 us for 2
// workgroups of 256 lanes).
// The dispatches' sizes are the prep's (gSurfArgs: per step slot and kind three words,
// x = the most chunks of SURF_WG elements any vehicle has for the kind, y = the
// vehicles, z = 1): a kind no vehicle has work for this step gets no workgroup. The
// slots turn as the self split's (step 0's, then odd and even steps'): the prep of step
// s clears step s + 1's (last read by step s - 1's dispatches; after the last step,
// step 0's), so no dispatch reads a slot another is writing. Kinds: 0 triangles, 1
// segments, 2 points (the per-element loops, when the vehicle has static slots or
// proxies for them), 3 the pair path (when it listed a pair), 4 the pair path's listing.
//
// The pair path's listing (surfPairCull: every element against every proxy shape's
// bounds, the prep's longest phase beside proxies) is a dispatch of its own here too, a
// thread an element (explicit_surface_cull, kind 4), and the decision that follows it
// (the list within its room or not, the proxies left to the per-element loops) another, a
// workgroup a vehicle (explicit_surface_plan): the prep ends at its barrier before the
// listing and leaves the vehicle's record provisional (the pair path wanted, every
// proxy), the plan writes it as the one prep did. The list's order follows the threads
// as it followed the lanes; nothing reads it (its count, and each pair's own find).
#if defined(EB_SURF_ARGS) || defined(EB_SURF_SPREAD)
#define SURF_WG 64
static const uint SURF_X = 1024u;
static const uint SURF_ARG_KINDS = 5u;
uint surfArgsAt(uint s, uint kind) { return 3u * (SURF_ARG_KINDS * (s == 0u ? 2u : (s & 1u)) + kind); }
uint surfChunks(uint n) { return min((n + SURF_WG - 1u) / SURF_WG, SURF_X); }
#endif
#ifdef EB_SURF_ARGS
[[vk::binding(10, 0)]] RWStructuredBuffer<Atomic<int>> gSurfArgs;
// The prep's start (vehicle 0's lane 0, whatever its state): the next step's slot
// cleared. (A frame of one step clears none: its slot keeps the largest sizes, more
// workgroups than needed, which return at once.)
void surfArgsClear(uint v, uint lane)
{
    if (v != 0u || lane != 0u) return;
    uint s = gP.pad2;
    if (s + 1u >= gP.steps && s == 0u) return;
    uint next = s + 1u < gP.steps ? s + 1u : 0u;
    for (uint kind = 0u; kind < SURF_ARG_KINDS; kind++) gSurfArgs[surfArgsAt(next, kind)].store(0);
}
// The prep's end (lane 0 of a vehicle that steps its surface): its work for the step's
// dispatches.
void surfArgsWant(uint kind, uint chunks)
{
    if (chunks == 0u) return;
    uint at = surfArgsAt(gP.pad2, kind);
    gSurfArgs[at].max(int(chunks));
    gSurfArgs[at + 1u].store(int(gP.vehicles));
    gSurfArgs[at + 2u].store(1);
}
#endif

#ifdef EB_SURF_SPREAD
// surfPipeSkip for a thread of the spread pipelines (no barrier).
bool surfSpreadSkip(uint v)
{
    return v >= gP.vehicles || surfaceOn() == 0u || sleepAsleep(v);
}
#endif

// Whether workgroup v steps no surface (uniform): past the vehicles, asleep (the frame
// did not step it), or no vehicle of the upload has surface contact.
bool surfPipeSkip(uint v, uint lane)
{
    if (v >= gP.vehicles) return true;
    if (surfaceOn() == 0u) return true;
    if (gP.sleepRec != 0u && sleepSkip(v, lane)) return true;
    // The coupling's workgroup copies, as the frame's start writes them (its header;
    // an empty sphere table: every sphere read from gNext, the same words the frame's
    // copies hold).
    if (lane == 0u)
    {
        coupleBegin();
        cplGsClear();
    }
    sync();
    return false;
}

#if defined(EB_SURF_ARGS) && defined(EB_SURF_KIND)
// The pair path's decision after its listing (the fast schedules; a workgroup a vehicle,
// lane 0): the prep's own, from the count the listing left, into the vehicle's record and
// the step's dispatches' sizes. A vehicle the prep gave no pair path keeps its record.
[shader("compute")]
[numthreads(SURF_WG, 1, 1)]
void explicit_surface_plan(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    if (surfPipeSkip(v, lane)) return ebLoopCapFlush();
    uint4 rec = gMisc[segRec(v)];
    if (lane != 0u || rec.y == 0u) return ebLoopCapFlush();
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 sr = gStatic[gP.surfSeg + v];
    uint proxies = rec.w;
    bool anyFast = proxies != 0u && surfAnyFast(v, proxies);
    uint listed = uint(gAcc[surfPairCount(v)].load());
    bool pairs = true;
    uint px = proxies;
    if (listed <= SURF_PAIR_CAP) px = anyFast ? (proxies | 0x80000000u) : 0u;
    else pairs = false;
    gMisc[segRec(v)] = uint4(rec.x, pairs ? 1u : 0u, listed, px);
    // (A vehicle with the pair path has the flag: its counts are its own.)
    if (r1.w != 0u || px != 0u)
    {
        surfArgsWant(0u, surfChunks(r1.y));
        surfArgsWant(1u, surfChunks(sr.y));
        surfArgsWant(2u, surfChunks(r0.y));
    }
    if (pairs && listed != 0u) surfArgsWant(3u, surfChunks(max(listed, SURF_HIT_CAP)));
    ebLoopCapFlush();
}
#elif !defined(EB_SURF_KIND)
// The pass's start: the snapshot, the casts' gathering (a fast proxy), the pair path's
// bounds and listing; its decisions into segRec(v).
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_surface_prep(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
#ifdef EB_SURF_ARGS
    surfArgsClear(v, lane);
#endif
    if (surfPipeSkip(v, lane)) return ebLoopCapFlush();
    uint parity = (gMisc[v].w + gP.pad2) & 1u;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    bool mine = (r3.x & FLAG_SURFACE) != 0u;
    uint nodeCount = mine ? r0.y : 0u;
    uint triCount = mine ? r1.y : 0u;
    float3 origin = asfloat(gMisc[v].xyz);
    float r = asfloat(r3.z);
    uint4 sr = gStatic[gP.surfSeg + v];
    uint segCount = mine ? sr.y : 0u;
    uint proxies = mine ? surfProxies(v) : 0u;
    uint bodies = proxies != 0u ? cplVeh(v).x : 0u;
    bool anyFast = proxies != 0u && surfAnyFast(v, proxies);
    bool pairs = proxies != 0u && surfPairsOn();
    if (lane == 0u) gsUniformSurf = pairs ? 1u : 0u;
    sync();
    pairs = gsUniformSurf != 0u;
    for (uint i = lane, iStop = EB_END(547, i, nodeCount, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + gi], r0.z, r2.z, r2.w, parity, false);
        gBeams[gP.surfSnap + gi] = float4(sadd3(f, accForce(gi)), 0.0);
        gAcc[gP.surfSplit + gi].store(0);
    }
    for (uint j = lane, jStop = EB_END(548, j, bodies, LANES); j < jStop; j += LANES)
    {
        gAcc[surfBodyBase() + v * cplB() + j].store(0);
        gAcc[surfTminAt(v, j, parity + 1u)].store(0);
    }
    if (anyFast)
    {
        for (uint t = lane, tStop = EB_END(549, t, triCount, LANES); t < tStop; t += LANES)
            surfGatherTriangle(v, r1.x + t, origin, r, proxies, parity);
        for (uint j = lane, jStop = EB_END(550, j, segCount, LANES); j < jStop; j += LANES)
            surfGatherSegment(v, gLists[sr.x + j], origin, r, proxies, r2.z, parity);
        for (uint i = lane, iStop = EB_END(551, i, nodeCount, LANES); i < iStop; i += LANES)
            surfGatherPoint(v, r0.x + i, origin, r, proxies, parity);
    }
    if (pairs) surfPairBounds(v, lane, proxies);
    sync();
#ifdef EB_SURF_ARGS
    // The fast schedules: the listing and the decision are the next two dispatches
    // (explicit_surface_cull, explicit_surface_plan). The record until then: the pair
    // path wanted, every proxy. Without the pair path it is final, and the per-element
    // loops' sizes with it.
    if (lane == 0u)
    {
        uint hit0 = mine ? 1u : gMisc[segRec(v)].x;
        gMisc[segRec(v)] = uint4(hit0, pairs ? 1u : 0u, 0u, proxies);
        if (pairs)
            surfArgsWant(4u, surfChunks(triCount + segCount + nodeCount));
        else if (r1.w != 0u || proxies != 0u)
        {
            surfArgsWant(0u, surfChunks(triCount));
            surfArgsWant(1u, surfChunks(segCount));
            surfArgsWant(2u, surfChunks(nodeCount));
        }
    }
    return ebLoopCapFlush();
#else
    uint px = proxies;
    uint listed = 0u;
    if (pairs)
    {
        surfPairCull(v, lane, origin, r, r0, r1, sr, r2.z, parity, triCount, segCount, nodeCount, proxies);
        sync();
        listed = uint(gAcc[surfPairCount(v)].load());
        if (listed <= SURF_PAIR_CAP) px = anyFast ? (proxies | 0x80000000u) : 0u;
        else pairs = false;
    }
    if (lane == 0u)
    {
        uint hit = mine ? 1u : gMisc[segRec(v)].x;
        gMisc[segRec(v)] = uint4(hit, pairs ? 1u : 0u, listed, px);
    }
    ebLoopCapFlush();
#endif
}
#elif defined(EB_SURF_SPREAD) && EB_SURF_KIND == 5
// The pair path's listing, a thread an element (the fast schedules): surfPairCull's
// three loops over one index, the triangles, then the segments, then the nodes; each
// element's test and listing as there.
[shader("compute")]
[numthreads(SURF_WG, 1, 1)]
void explicit_surface_cull(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.y;
    uint lane = lid.x;
    if (surfSpreadSkip(v)) return ebLoopCapFlush();
    uint4 rec = gMisc[segRec(v)];
    if (rec.y == 0u) return ebLoopCapFlush();
    uint proxies = rec.w;
    uint parity = (gMisc[v].w + gP.pad2) & 1u;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    float r = asfloat(r3.z);
    uint4 sr = gStatic[gP.surfSeg + v];
    uint triCount = r1.y;
    uint segCount = sr.y;
    uint nodeCount = r0.y;
    uint stride = SURF_X * SURF_WG;
    for (uint e = gid.x * SURF_WG + lane, eStop = EB_END(605, e, triCount + segCount + nodeCount, stride); e < eStop; e += stride)
    {
        if (e < triCount)
        {
            uint gt = r1.x + e;
            if (gP.retireBase != 0u && surfTriRetired(gt)) continue;
            if ((gP.triBreak != 0u && triangleDead(gt)) || (gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u) continue;
            uint4 tn = gStatic[gP.triBase + gt];
            float3 x0 = nf3(gNodes[tn.x * 3u].xyz);
            float3 x1 = nf3(gNodes[tn.y * 3u].xyz);
            float3 x2 = nf3(gNodes[tn.z * 3u].xyz);
            float3 tlo = ssub3(min(min(x0, x1), x2), float3(r, r, r));
            float3 thi = sadd3(max(max(x0, x1), x2), float3(r, r, r));
            surfPairList(v, SURF_TRI, e, tlo, thi, proxies, true, x0, x1, x2, r);
        }
        else if (e < triCount + segCount)
        {
            uint j = e - triCount;
            uint gj = gLists[sr.x + j];
            if (gP.retireBase != 0u && surfSegRetired(gj)) continue;
            if (surfBroken(gj, r2.z, parity)) continue;
            uint4 bn = gStatic[gP.beamBase + gj * 3u];
            float3 x0 = nf3(gNodes[bn.x * 3u].xyz);
            float3 x1 = nf3(gNodes[bn.y * 3u].xyz);
            float3 tlo = ssub3(min(x0, x1), float3(r, r, r));
            float3 thi = sadd3(max(x0, x1), float3(r, r, r));
            surfPairList(v, SURF_SEG, j, tlo, thi, proxies, true, x0, x1, x1, r);
        }
        else
        {
            uint i = e - triCount - segCount;
            uint gi = r0.x + i;
            if (gP.retireBase != 0u && nodeRetired(gi)) continue;
            if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u) continue;
            float3 x = nf3(gNodes[gi * 3u].xyz);
            float3 tlo = ssub3(min(x, x), float3(r, r, r));
            float3 thi = sadd3(max(x, x), float3(r, r, r));
            surfPairList(v, SURF_PT, i, tlo, thi, proxies, false, x, x, x, r);
        }
    }
    ebLoopCapFlush();
}
#elif defined(EB_SURF_SPREAD)
// One kind's find or apply, a thread an element (the pipelines of the fast schedules,
// above): surfPipeKind's calls for the elements first, first + stride, ...
void surfSpreadKind(uint v, uint first, bool apply)
{
    if (surfSpreadSkip(v)) return;
    uint stride = SURF_X * SURF_WG;
    uint parity = (gMisc[v].w + gP.pad2) & 1u;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    bool mine = (r3.x & FLAG_SURFACE) != 0u;
    uint nodeCount = mine ? r0.y : 0u;
    uint triCount = mine ? r1.y : 0u;
    float3 origin = asfloat(gMisc[v].xyz);
    float r = asfloat(r3.z);
    uint4 sr = gStatic[gP.surfSeg + v];
    uint segCount = mine ? sr.y : 0u;
    uint4 rec = gMisc[segRec(v)];
    bool pairs = rec.y != 0u;
    uint listed = rec.z;
    uint px = rec.w;
    bool loops = r1.w != 0u || px != 0u;
#if EB_SURF_KIND == 1
    if (loops)
    {
        for (uint t = first, tStop = EB_END(599, t, triCount, stride); t < tStop; t += stride)
            surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, px, parity, apply);
    }
#elif EB_SURF_KIND == 2
    if (loops)
    {
        for (uint j = first, jStop = EB_END(600, j, segCount, stride); j < jStop; j += stride)
            surfSegment(v, gLists[sr.x + j], sr.z + j, origin, r, r1.z, r1.w, px, r2.z, parity, apply);
    }
#elif EB_SURF_KIND == 3
    if (loops)
    {
        for (uint i = first, iStop = EB_END(601, i, nodeCount, stride); i < iStop; i += stride)
            surfPoint(v, r0.x + i, sr.w + i, origin, r, r1.z, r1.w, px, parity, apply);
    }
#else
    if (pairs && !apply)
    {
        for (uint p = first, pStop = EB_END(602, p, listed, stride); p < pStop; p += stride)
            surfPairFind(v, p, origin, r, r0, r1, sr);
    }
    if (pairs && apply)
    {
        uint found = uint(gAcc[surfHitCount(v)].load());
        if (found <= SURF_HIT_CAP)
        {
            for (uint h = first, hStop = EB_END(603, h, found, stride); h < hStop; h += stride)
                surfHitApply(v, h, r, r0, r1, sr);
        }
        else
        {
            for (uint p = first, pStop = EB_END(604, p, listed, stride); p < pStop; p += stride)
                surfPairApply(v, p, origin, r, r0, r1, sr);
        }
    }
#endif
}

#ifdef EB_SURF_POOL
#include "explicit_beams_surface_pool.slang"
[shader("compute")]
[numthreads(SURF_WG, 1, 1)]
void explicit_surface_find(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    surfPoolFind(gid.y, gid.x * SURF_WG + lid.x, lid.x);
    ebLoopCapFlush();
}
#else
[shader("compute")]
[numthreads(SURF_WG, 1, 1)]
void explicit_surface_find(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    surfSpreadKind(gid.y, gid.x * SURF_WG + lid.x, false);
    ebLoopCapFlush();
}
#endif

[shader("compute")]
[numthreads(SURF_WG, 1, 1)]
void explicit_surface_apply(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    surfSpreadKind(gid.y, gid.x * SURF_WG + lid.x, true);
    ebLoopCapFlush();
}
#else
// One kind's find or apply (EB_SURF_KIND: 1 triangles, 2 segments, 3 points, 4 the
// pair path), after every kind's find (the hit counts the applies split by).
void surfPipeKind(uint v, uint lane, bool apply)
{
    if (surfPipeSkip(v, lane)) return;
    uint parity = (gMisc[v].w + gP.pad2) & 1u;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    bool mine = (r3.x & FLAG_SURFACE) != 0u;
    uint nodeCount = mine ? r0.y : 0u;
    uint triCount = mine ? r1.y : 0u;
    float3 origin = asfloat(gMisc[v].xyz);
    float r = asfloat(r3.z);
    uint4 sr = gStatic[gP.surfSeg + v];
    uint segCount = mine ? sr.y : 0u;
    uint4 rec = gMisc[segRec(v)];
    bool pairs = rec.y != 0u;
    uint listed = rec.z;
    uint px = rec.w;
    bool loops = r1.w != 0u || px != 0u;
#if EB_SURF_KIND == 1
    if (loops)
    {
        for (uint t = lane, tStop = EB_END(552, t, triCount, LANES); t < tStop; t += LANES)
            surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, px, parity, apply);
    }
#elif EB_SURF_KIND == 2
    if (loops)
    {
        for (uint j = lane, jStop = EB_END(553, j, segCount, LANES); j < jStop; j += LANES)
            surfSegment(v, gLists[sr.x + j], sr.z + j, origin, r, r1.z, r1.w, px, r2.z, parity, apply);
    }
#elif EB_SURF_KIND == 3
    if (loops)
    {
        for (uint i = lane, iStop = EB_END(554, i, nodeCount, LANES); i < iStop; i += LANES)
            surfPoint(v, r0.x + i, sr.w + i, origin, r, r1.z, r1.w, px, parity, apply);
    }
#else
    if (pairs && !apply)
    {
        for (uint p = lane, pStop = EB_END(555, p, listed, LANES); p < pStop; p += LANES)
            surfPairFind(v, p, origin, r, r0, r1, sr);
    }
    if (pairs && apply)
    {
        uint found = uint(gAcc[surfHitCount(v)].load());
        if (found <= SURF_HIT_CAP)
        {
            for (uint h = lane, hStop = EB_END(556, h, found, LANES); h < hStop; h += LANES)
                surfHitApply(v, h, r, r0, r1, sr);
        }
        else
        {
            for (uint p = lane, pStop = EB_END(557, p, listed, LANES); p < pStop; p += LANES)
                surfPairApply(v, p, origin, r, r0, r1, sr);
        }
    }
#endif
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_surface_find(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    surfPipeKind(gid.x, lid.x, false);
    ebLoopCapFlush();
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_surface_apply(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    surfPipeKind(gid.x, lid.x, true);
    ebLoopCapFlush();
}
#endif
#endif
