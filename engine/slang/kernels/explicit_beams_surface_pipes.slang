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

#if !defined(EB_SURF_KIND)
// The pass's start: the snapshot, the casts' gathering (a fast proxy), the pair path's
// bounds and listing; its decisions into segRec(v).
[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_surface_prep(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
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
    uint bodies = proxies != 0u ? gMisc[cplBase(v)].x : 0u;
    bool anyFast = proxies != 0u && surfAnyFast(v, proxies);
    bool pairs = proxies != 0u && surfPairsOn();
    if (lane == 0u) gsUniformSurf = pairs ? 1u : 0u;
    sync();
    pairs = gsUniformSurf != 0u;
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + gi], r0.z, r2.z, r2.w, parity, false);
        gBeams[gP.surfSnap + gi] = float4(sadd3(f, accForce(gi)), 0.0);
        gAcc[gP.surfSplit + gi].store(0);
    }
    for (uint j = lane, jStop = EB_END(j, bodies, LANES); j < jStop; j += LANES)
    {
        gAcc[surfBodyBase() + v * cplB() + j].store(0);
        gAcc[surfTminAt(v, j, parity + 1u)].store(0);
    }
    if (anyFast)
    {
        for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
            surfGatherTriangle(v, r1.x + t, origin, r, proxies, parity);
        for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
            surfGatherSegment(v, gLists[sr.x + j], origin, r, proxies, r2.z, parity);
        for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
            surfGatherPoint(v, r0.x + i, origin, r, proxies, parity);
    }
    if (pairs) surfPairBounds(v, lane, proxies);
    sync();
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
        for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
            surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, px, parity, apply);
    }
#elif EB_SURF_KIND == 2
    if (loops)
    {
        for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
            surfSegment(v, gLists[sr.x + j], sr.z + j, origin, r, r1.z, r1.w, px, r2.z, parity, apply);
    }
#elif EB_SURF_KIND == 3
    if (loops)
    {
        for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
            surfPoint(v, r0.x + i, sr.w + i, origin, r, r1.z, r1.w, px, parity, apply);
    }
#else
    if (pairs && !apply)
    {
        for (uint p = lane, pStop = EB_END(p, listed, LANES); p < pStop; p += LANES)
            surfPairFind(v, p, origin, r, r0, r1, sr);
    }
    if (pairs && apply)
    {
        uint found = uint(gAcc[surfHitCount(v)].load());
        if (found <= SURF_HIT_CAP)
        {
            for (uint h = lane, hStop = EB_END(h, found, LANES); h < hStop; h += LANES)
                surfHitApply(v, h, r, r0, r1, sr);
        }
        else
        {
            for (uint p = lane, pStop = EB_END(p, listed, LANES); p < pStop; p += LANES)
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
