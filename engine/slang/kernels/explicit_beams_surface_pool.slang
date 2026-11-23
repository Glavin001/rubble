// The pair path's find with its hulls' records in workgroup memory (the fast schedules'
// explicit_surface_find_pair_x: EB_SURF_SPREAD, EB_SURF_KIND 4; vehicle-perf-first-
// principles.md §8.61). Included by explicit_beams_surface_pipes.slang.
//
// A pair against a hull shape reads the hull's planes, vertices and edges from gMisc on
// every use: a triangle's narrow phase (fqContact) walks every edge (two vertices each)
// up to four times, clips its edges by every plane and, for a deep contact, projects
// every vertex on each of 1 + faces + 3 x edges axes: 600 to 1,900 dependent loads a
// pair, each a trip to L2 on CUDA (loads are device-coherent there, `ld.cg`), and the
// dispatch waits for its longest thread. Here the workgroup copies each distinct hull its
// pairs meet once (a lane a record) into gsSurfPool and the pair's shape reads it there
// (fqHullPlane, fqHullVertex, fqHullEdge: SURF_POOL_FLAG in FqShape.hull). The same
// words, so the same arithmetic and the same hits bit for bit; a hull that does not fit
// the pool is read from gMisc as before. Which hulls are copied is a choice of schedule.

#ifdef EB_SURF_POOL

static const uint SURF_POOL_NONE = 0xFFFFFFFFu;

// Every lane: its pair's hull shape (k, its first record in gMisc, its records), the
// first lane with the same shape (the hull's owner), the owners' pool offsets.
groupshared uint gsSurfPoolK[SURF_WG];
groupshared uint gsSurfPoolAt[SURF_WG];
groupshared uint gsSurfPoolOwners[SURF_WG];
groupshared Atomic<uint> gsSurfPoolUsed;
groupshared Atomic<uint> gsSurfPoolCount;

// Vehicle v's listed pair `first` (the lane's; the dispatch holds every listed pair at
// first < SURF_X · SURF_WG, so the loop below runs once): its hull staged, then its find.
void surfPoolFind(uint v, uint first, uint lane)
{
    // The decision of surfSpreadKind (the vehicle skipped, no pair path) by lane 0, read
    // by every lane as uniform: the barriers below are in uniform control flow.
    if (lane == 0u)
    {
        bool go = !surfSpreadSkip(v) && gMisc[segRec(v)].y != 0u;
        gsUniformSurf = go ? 1u : 0u;
        gsSurfPoolUsed.store(0u);
        gsSurfPoolCount.store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (gsUniformSurf == 0u) return;
    uint listed = gMisc[segRec(v)].z;
    uint myK = SURF_POOL_NONE;
    uint src = 0u;
    uint len = 0u;
    if (first < listed)
    {
        uint k = (gNext[surfPairAt(v, first)] >> 20) & 1023u;
        uint at = cplShape(v, k);
        uint4 h = gMisc[at];
        if (h.x == CPL_HULL)
        {
            uint pw = gMisc[at + 3u].w;
            myK = k;
            src = cplPlane(v, h.w & 0xFFFFu);
            len = (h.w >> 16) + (pw & 255u) + (((pw >> 8) & 255u) + 3u) / 4u;
        }
    }
    gsSurfPoolK[lane] = myK;
    gsSurfPoolAt[lane] = SURF_POOL_NONE;
    GroupMemoryBarrierWithGroupSync();
    // The owner: the first lane with the same shape.
    uint owner = lane;
    if (myK != SURF_POOL_NONE)
    {
        for (uint j = 0u, jStop = EB_END(760, j, lane, 1u); j < jStop; j++)
        {
            if (gsSurfPoolK[j] == myK)
            {
                owner = j;
                break;
            }
        }
        if (owner == lane)
        {
            uint off = gsSurfPoolUsed.add(len);
            if (off + len <= SURF_POOL_RECS)
            {
                gsSurfPoolAt[lane] = off;
                gsSurfPoolOwners[gsSurfPoolCount.add(1u)] = lane;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Each staged hull's records, a lane a record.
    uint owners = gsSurfPoolCount.load();
    for (uint c = 0u, cStop = EB_END(761, c, owners, 1u); c < cStop; c++)
    {
        uint o = gsSurfPoolOwners[c];
        uint oat = gsSurfPoolAt[o];
        uint4 oh = gMisc[cplShape(v, gsSurfPoolK[o])];
        uint opw = gMisc[cplShape(v, gsSurfPoolK[o]) + 3u].w;
        uint osrc = cplPlane(v, oh.w & 0xFFFFu);
        uint olen = (oh.w >> 16) + (opw & 255u) + (((opw >> 8) & 255u) + 3u) / 4u;
        for (uint i = lane, iStop = EB_END(762, i, olen, SURF_WG); i < iStop; i += SURF_WG)
            gsSurfPool[oat + i] = gMisc[osrc + i];
    }
    GroupMemoryBarrierWithGroupSync();
    gSurfPoolK = SURF_POOL_NONE;
    if (myK != SURF_POOL_NONE && gsSurfPoolAt[owner] != SURF_POOL_NONE)
    {
        gSurfPoolK = myK;
        gSurfPoolAt = gsSurfPoolAt[owner];
    }
    float3 origin = asfloat(gMisc[v].xyz);
    float r = asfloat(gStatic[v * 4u + 3u].z);
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 sr = gStatic[gP.surfSeg + v];
    uint stride = SURF_X * SURF_WG;
    for (uint p = first, pStop = EB_END(763, p, listed, stride); p < pStop; p += stride)
        surfPairFind(v, p, origin, r, r0, r1, sr);
}

#endif
