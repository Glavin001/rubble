// The pair path's find on the fast schedules (explicit_surface_find_pair_x:
// EB_SURF_SPREAD, EB_SURF_KIND 4, EB_SURF_POOL; vehicle-perf-first-principles.md
// §8.61). Included by explicit_beams_surface_pipes.slang.
//
// The dispatch waits for its longest thread, and that is a triangle against a hull:
// fqContact walks every hull edge (fqSegTri's six candidates each), clips the
// triangle's edges by every plane, measures each corner against every plane and walks
// the edges again: thousands of dependent strict operations on one lane, 100 to 370 us
// a step on a T4 while the rest of the list takes about 120. So:
//
// - The listing (surfPairListSplit) puts a triangle's pairs against hulls at the end of
//   the vehicle's region, and the find gives each such pair a workgroup of its own
//   (surfBigFind): the lanes take the hull's edges and planes, the order-free parts
//   (the least gap, the corners' greatest plane distance, the separation test) are
//   reduced across them, the candidates within the tie of the least gap are kept with
//   their place in fqContact's walk, and lane 0 sums them in that order. The same
//   candidates through the same functions (fqSegTriCand, fqInsidePair), the same sums
//   in the same order: the same contact bit for bit. A deep contact (fqContact's
//   overlap axis) or more kept candidates than the room: lane 0 runs fqContact whole.
// - Every other workgroup takes SURF_WG listed pairs a lane each, as before, with each
//   distinct hull its pairs meet copied once into workgroup memory (gsSurfPool, a lane
//   a record) and read there (fqHullPlane, fqHullVertex, fqHullEdge: SURF_POOL_FLAG in
//   FqShape.hull): the same words, so the same arithmetic. A hull that does not fit
//   the pool is read from gMisc.
//
// The culls in the big pair's first walk prune by each lane's own least gap: the walk's
// culls are lossless (feature_query.slang, fqPrune), so its least gap is the same
// whichever lane walks which edge.

#ifdef EB_SURF_POOL

static const uint SURF_POOL_NONE = 0xFFFFFFFFu;
static const uint SURF_BIG_KEEP = 32u;

groupshared uint gsSurfPoolK[SURF_WG];
groupshared uint gsSurfPoolAt[SURF_WG];
groupshared uint gsSurfPoolOwners[SURF_WG];
groupshared Atomic<uint> gsSurfPoolUsed;
groupshared Atomic<uint> gsSurfPoolCount;
// The big pair's reductions (a lane each) and its kept candidates.
groupshared float gsBigGap[SURF_WG];
groupshared float gsBigDist[3u * SURF_WG];
groupshared uint gsBigSep;
groupshared Atomic<uint> gsBigKept;
groupshared uint gsBigKey[SURF_BIG_KEEP];
groupshared float3 gsBigW[SURF_BIG_KEEP];
groupshared float3 gsBigQ[SURF_BIG_KEEP];
groupshared float3 gsBigCs[SURF_BIG_KEEP];

// A hull's records in gMisc from shape record `at`: the first, and how many.
uint surfPoolLen(uint at)
{
    uint4 h = gMisc[at];
    uint pw = gMisc[at + 3u].w;
    return (h.w >> 16) + (pw & 255u) + (((pw >> 8) & 255u) + 3u) / 4u;
}

void surfBigKeep(uint key, FqPair p)
{
    uint at = gsBigKept.add(1u);
    if (at >= SURF_BIG_KEEP) return;
    gsBigKey[at] = key;
    gsBigW[at] = p.w;
    gsBigQ[at] = p.q;
    gsBigCs[at] = p.cs;
}

// Item t of the hull's walk for lane-parallel fqContact: hull edge t (t < ne: its six
// candidates, culled as fqCandidates culls), else triangle edge t - ne inside the hull
// (fqEdgesInside: two). pass 0 visits into a; pass 1 keeps those within the tie of gx
// with key 8 t + their index.
void surfBigItem(FqShape s, float3 x0, float3 x1, float3 x2, float3 tlo, float3 thi, uint ne, uint t, uint pass,
                 inout FqAcc a)
{
    float thr = nf(a.gx + FQ_TIE);
    if (t < ne)
    {
        float3 e0;
        float3 e1;
        fqWalkEdge(s, t, e0, e1);
        if (fqFar(e0, e1, tlo, thi, fqPrune(a))) return;
        for (uint i = 0u; i < 6u; i++)
        {
            FqPair p;
            float g;
            if (!fqSegTriCand(s, e0, e1, x0, x1, x2, i, p, g)) continue;
            if (pass == 0u) fqVisit(a, p, g);
            else if (g <= thr) surfBigKeep(8u * t + i, p);
        }
        return;
    }
    uint k = t - ne;
    float t0;
    float t1;
    if (!fqClip(s, fqX(x0, x1, x2, k), fqX(x0, x1, x2, (k + 1u) % 3u), t0, t1)) return;
    for (uint e = 0u; e < 2u; e++)
    {
        FqPair p = fqInsidePair(x0, x1, x2, k, e == 0u ? t0 : t1);
        if (pass == 0u) fqVisit(a, p, 0.0);
        else if (0.0 <= thr) surfBigKeep(8u * t + e, p);
    }
}

// Listed pair `slot` (a triangle against a hull) on the whole workgroup: surfPairFind's
// result, bit for bit.
void surfBigFind(uint v, uint slot, uint lane)
{
    uint code = gNext[surfPairAt(v, slot)];
    uint k = (code >> 20) & 1023u;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 sr = gStatic[gP.surfSeg + v];
    float r = asfloat(gStatic[v * 4u + 3u].z);
    float3 origin = asfloat(gMisc[v].xyz);
    uint4 tn = surfPairNodes(code, r0, r1, sr);
    uint at = cplShape(v, k);
    uint len = surfPoolLen(at);
    bool fits = len <= SURF_POOL_RECS;
    if (fits)
    {
        uint src = cplPlane(v, gMisc[at].w & 0xFFFFu);
        for (uint i = lane, iStop = EB_END(765, i, len, SURF_WG); i < iStop; i += SURF_WG)
            gsSurfPool[i] = gMisc[src + i];
    }
    if (lane == 0u)
    {
        gsBigSep = 0u;
        gsBigKept.store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    gSurfPoolK = fits ? k : SURF_POOL_NONE;
    gSurfPoolAt = 0u;
    uint j;
    FqShape s = surfProxyShape(v, k, j);
    float3 x0 = nf3(gNodes[tn.x * 3u].xyz);
    float3 x1 = nf3(gNodes[tn.y * 3u].xyz);
    float3 x2 = nf3(gNodes[tn.z * 3u].xyz);
    float3 tlo = min(min(x0, x1), x2);
    float3 thi = max(max(x0, x1), x2);
    uint ne = fqWalkEdges(s);
    uint faces = min(s.nf, FQ_HULL_FACES);
    // The order-free parts, a lane a plane and a lane an item: fqSeparated's planes,
    // fqContact's first walk (its least gap), fqPointDist's planes for the corners.
    float3 m = max(max(abs(x0), abs(x1)), abs(x2));
    float big = max(max(m.x, m.y), m.z);
    float d0 = -3.0e38;
    float d1 = -3.0e38;
    float d2 = -3.0e38;
    for (uint f = lane, fStop = EB_END(766, f, faces, SURF_WG); f < fStop; f += SURF_WG)
    {
        float4 pl = fqHullP(s, f);
        if (fqCullsOn() && fqOutside(pl, x0, x1, x2, r, big)) gsBigSep = 1u;
        d0 = max(d0, nf(sdot(pl.xyz, x0) - pl.w));
        d1 = max(d1, nf(sdot(pl.xyz, x1) - pl.w));
        d2 = max(d2, nf(sdot(pl.xyz, x2) - pl.w));
    }
    FqAcc a = fqAccNew();
    a.lim = r;
    for (uint t = lane, tStop = EB_END(767, t, ne + 3u, SURF_WG); t < tStop; t += SURF_WG)
        surfBigItem(s, x0, x1, x2, tlo, thi, ne, t, 0u, a);
    gsBigGap[lane] = a.gx;
    gsBigDist[lane] = d0;
    gsBigDist[SURF_WG + lane] = d1;
    gsBigDist[2u * SURF_WG + lane] = d2;
    GroupMemoryBarrierWithGroupSync();
    float gx = 1e30;
    d0 = -3.0e38;
    d1 = -3.0e38;
    d2 = -3.0e38;
    for (uint l = 0u; l < SURF_WG; l++)
    {
        gx = min(gx, gsBigGap[l]);
        d0 = max(d0, gsBigDist[l]);
        d1 = max(d1, gsBigDist[SURF_WG + l]);
        d2 = max(d2, gsBigDist[2u * SURF_WG + l]);
    }
    float3 nc = scross(ssub3(x1, x0), ssub3(x2, x0));
    float nl = fqLen(nc);
    float tie = nf(gx + FQ_TIE);
    bool alive = gsBigSep == 0u && nl > 1e-8 && gx <= r && !(d0 >= 0.0 && d0 <= tie) && !(d1 >= 0.0 && d1 <= tie)
                 && !(d2 >= 0.0 && d2 <= tie);
    // fqContact's second walk: the candidates within the tie, kept.
    if (alive)
    {
        FqAcc a1 = fqAccNew();
        a1.lim = r;
        a1.gx = gx;
        a1.pass = 1u;
        for (uint t = lane, tStop = EB_END(768, t, ne + 3u, SURF_WG); t < tStop; t += SURF_WG)
            surfBigItem(s, x0, x1, x2, tlo, thi, ne, t, 1u, a1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0u || !alive) return;
    uint kept = gsBigKept.load();
    float3 w = float3(0.0, 0.0, 0.0);
    float3 nn = float3(0.0, 0.0, 0.0);
    float g = gx;
    float3 q = float3(0.0, 0.0, 0.0);
    bool whole = kept > SURF_BIG_KEEP;
    if (!whole)
    {
        // fqVisit's pass-1 sums in the walk's order (keys ascending).
        FqAcc b = fqAccNew();
        uint last = 0u;
        for (uint c = 0u, cStop = EB_END(769, c, kept, 1u); c < cStop; c++)
        {
            uint pick = SURF_POOL_NONE;
            for (uint e = 0u, eStop = EB_END(770, e, kept, 1u); e < eStop; e++)
            {
                uint key = gsBigKey[e];
                if ((c == 0u || key > last) && (pick == SURF_POOL_NONE || key < gsBigKey[pick])) pick = e;
            }
            last = gsBigKey[pick];
            b.w = float3(nf(b.w.x + gsBigW[pick].x), nf(b.w.y + gsBigW[pick].y), nf(b.w.z + gsBigW[pick].z));
            b.q = sadd3(b.q, gsBigQ[pick]);
            b.cs = sadd3(b.cs, gsBigCs[pick]);
            b.k = b.k + 1u;
        }
        float kf = float(b.k);
        w = float3(sdiv(b.w.x, kf), sdiv(b.w.y, kf), sdiv(b.w.z, kf));
        q = sdiv3(b.q, kf);
        float3 cs = sdiv3(b.cs, kf);
        uint nz = (w.x > 0.0 ? 1u : 0u) + (w.y > 0.0 ? 1u : 0u) + (w.z > 0.0 ? 1u : 0u);
        if (nz <= 1u) return;
        float3 dv = ssub3(cs, q);
        float dl = fqLen(dv);
        if (gx > FQ_PIERCE && dl > FQ_PIERCE) nn = sdiv3(dv, dl);
        else whole = true;
    }
    if (whole)
    {
        // A deep contact, or more candidates than the room: fqContact whole.
        bool two;
        float3 w2;
        float3 q2;
        if (!surfPairContact(v, code, origin, r, tn, j, w, nn, g, q, two, w2, q2)) return;
    }
    surfPairKeep(v, code, tn, j, w, nn, g, q, false, float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0));
}

// Workgroup gid of vehicle v: SURF_WG listed pairs from the region's start a lane each,
// then (past those workgroups) one big pair each from its end.
void surfPoolFind(uint v, uint gid, uint lane)
{
    // surfSpreadKind's decision (the vehicle skipped, no pair path) and this workgroup's
    // part by lane 0, read by every lane as uniform: the barriers below are in uniform
    // control flow. 1: SURF_WG pairs; 2 | slot << 2: a big pair.
    if (lane == 0u)
    {
        uint u = 0u;
        if (!surfSpreadSkip(v))
        {
            uint4 rec = gMisc[segRec(v)];
            uint lo = rec.z & 0xFFFFu;
            uint chunks = (lo + SURF_WG - 1u) / SURF_WG;
            if (rec.y != 0u && gid < chunks) u = 1u;
            else if (rec.y != 0u && gid - chunks < (rec.z >> 16)) u = 2u | (surfPairSlot(lo + gid - chunks, lo) << 2);
        }
        gsUniformSurf = u;
        gsSurfPoolUsed.store(0u);
        gsSurfPoolCount.store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint part = gsUniformSurf;
    if (part == 0u) return;
    if (part != 1u)
    {
        surfBigFind(v, part >> 2, lane);
        return;
    }
    uint first = gid * SURF_WG + lane;
    uint listed = gMisc[segRec(v)].z & 0xFFFFu;
    uint myK = SURF_POOL_NONE;
    uint len = 0u;
    if (first < listed)
    {
        uint k = (gNext[surfPairAt(v, first)] >> 20) & 1023u;
        uint at = cplShape(v, k);
        if (gMisc[at].x == CPL_HULL)
        {
            myK = k;
            len = surfPoolLen(at);
        }
    }
    gsSurfPoolK[lane] = myK;
    gsSurfPoolAt[lane] = SURF_POOL_NONE;
    GroupMemoryBarrierWithGroupSync();
    // The owner: the first lane with the same shape; it takes the hull's room.
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
        uint shape = cplShape(v, gsSurfPoolK[o]);
        uint osrc = cplPlane(v, gMisc[shape].w & 0xFFFFu);
        uint olen = surfPoolLen(shape);
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
