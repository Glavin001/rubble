// Edge vehicle contact, continued from explicit_beams_vv_edges.slang (included at its end):
// the lists, the grid and work stages, the pair tests, the per-owner passes.

// Vehicle v's live edges in [lo, hi] into its edge list, its contact points (`points`) too.
void edgeList(uint v, float3 lo, float3 hi, bool points, uint lane)
{
    uint4 v0 = edgeV0(v);
    uint4 v1 = edgeV1(v);
    for (uint e = lane, eStop = EB_END(825, e, v0.y, LANES); e < eStop; e += LANES)
    {
        float4 s0 = gBeams[v0.z + EDGE_SCR * e];
        if ((asuint(s0.w) & 8u) == 0u) continue;
        float3 q1 = gBeams[v0.z + EDGE_SCR * e + 1u].xyz;
        if (any(max(s0.xyz, q1) < lo) || any(min(s0.xyz, q1) > hi)) continue;
        gNext[v1.w + uint(gAcc[v1.x + 1u].add(1))] = e;
    }
    uint cps = points ? vvCpCount(v) : 0u;
    for (uint i = lane, iStop = EB_END(826, i, cps, LANES); i < iStop; i += LANES)
    {
        uint h = gLists[vvCpOff(v) + i];
        if (gP.retireBase != 0u && nodeRetired(h)) continue;
        float3 x = edgeWorld(v, h);
        if (any(x < lo) || any(x > hi)) continue;
        gNext[v1.w + v0.y + uint(gAcc[v1.x + 2u].add(1))] = h;
    }
}

// --- The listed edges' and points' grid (lossless: the same pairs as testing all) ---
// Each vehicle's listed edges and points in a hash grid of cells of side 2·reach (reach: the
// contact range and the crossing depth, the farthest two features in contact can be): each
// entry in every cell its box padded by the reach meets (an edge over more than
// EDGE_CELLS_MAX cells instead in the vehicle's long list); an owner edge visits the cells of
// its own box and tests an entry only in the cell holding the low corner of the two boxes'
// overlap (each pair once, and every pair whose boxes meet: every pair in contact). gAcc per
// vehicle at +10 the long list's count, at +12 per kind (0 edges, 1 points) EDGE_BUCKETS
// cursors (the counts, then each bucket's end) and EDGE_BUCKETS starts, then per kind the
// blocks' sums (EDGE_BUCKETS / EDGE_BLOCK); gNext after the lists
// the edges' entries (EDGE_CELLS_MAX a listed edge), the points' (8 a point), the long list.

static const uint EDGE_CELLS_MAX = 64u; // explicit_plan.rs EDGE_CELLS_MAX
uint edgeGridAt(uint v, uint kind) { return edgeV1(v).x + 12u + kind * 2u * EDGE_BUCKETS; }
uint edgeBlocksAt(uint v, uint kind) { return edgeV1(v).x + 12u + 4u * EDGE_BUCKETS + kind * (EDGE_BUCKETS / EDGE_BLOCK); }
uint edgeListsLen(uint v) { return edgeV0(v).y + vvCpCount(v); }
uint edgeEntriesAt(uint v, uint kind) { return edgeV1(v).w + edgeListsLen(v) + (kind == 0u ? 0u : EDGE_CELLS_MAX * edgeV0(v).y); }
uint edgeLongAt(uint v) { return edgeEntriesAt(v, 1u) + 8u * vvCpCount(v); }
// The owner side's work: per listed edge its first (edge, cell) item, then the scan's block sums.
uint edgeWorkAt(uint v) { return edgeLongAt(v) + edgeV0(v).y; }
uint edgeWorkBlocksAt(uint v) { return edgeWorkAt(v) + edgeV0(v).y; }
// The frame-coherent candidates, kept per owner: gAcc per vehicle after the grid (count, 0,
// overflow, EDGE_KEEP_MAGIC while its list stands), gNext after the work list its keep record
// (EDGE_KEEP words), its partners' nodes at its build (three words a node, partner after
// partner), then the candidates (two words a pair, as a hit); gBeams after its edges' scratch
// its own nodes at its build. The keep record: 1 with the partners' nodes kept (they fit in
// half the room), the partners at the build (count, vehicles in the members' order), the
// nodes' words. (Word 1 of gAcc was the partners' signature; 0 now, so a build of the
// layout before builds again.)
static const uint EDGE_KEEP = 16u;
static const uint EDGE_KEEP_PARTS = 7u; // VV_PART - 1: a small group's partners
static const int EDGE_KEEP_MAGIC = 0x4B454550;
uint edgeCandAcc(uint v) { return edgeV1(v).x + 12u + 4u * EDGE_BUCKETS + 2u * (EDGE_BUCKETS / EDGE_BLOCK); }
uint edgeKeepAt(uint v) { return edgeWorkBlocksAt(v) + edgeV0(v).y / 32u + 1u; }
// The candidates' region: 2 EDGE_CAND words a listed item (a vehicle with no edge and no
// point has none, and keeps nothing).
uint edgeRegion(uint v) { return 2u * EDGE_CAND * edgeListsLen(v); }
bool edgeKeepable(uint v) { return edgeListsLen(v) != 0u; }
uint edgeCandAt(uint v) { uint s = edgeKeepAt(v); return s + EDGE_KEEP + gNext[s + 9u]; }
uint edgeCandRoom(uint v) { uint used = EDGE_KEEP + gNext[edgeKeepAt(v) + 9u]; return (max(edgeRegion(v), used) - used) / 2u; }
uint edgeBuiltAt(uint v) { return edgeV0(v).z + EDGE_SCR * edgeV0(v).y + 2u; }
// A cell's bucket: its coordinates modulo 16, 16 and 8 (EDGE_BUCKETS = 2048): the cells of
// one box (at most 4 a side, edgeCellCount) are in distinct buckets, so an entry is in a
// bucket once and an owner edge visits a bucket once.
uint edgeBucket(int3 c) { return (uint(c.x) & 15u) | ((uint(c.y) & 15u) << 4u) | ((uint(c.z) & 7u) << 8u); }
int3 edgeCellOf(float3 x, float cell) { return int3(floor(x / cell)); }
uint edgeListed(uint v, uint kind) { return uint(gAcc[edgeV1(v).x + 1u + kind].load()); }
uint edgeListAt(uint v, uint kind) { return edgeV1(v).w + (kind == 0u ? 0u : edgeV0(v).y); }
// An entry's box padded by the reach: an edge's ends, a point.
void edgeEntryBox(uint v, uint kind, uint x, float reach, out float3 lo, out float3 hi)
{
    float3 a;
    float3 b;
    if (kind == 0u)
    {
        uint at = edgeV0(v).z + EDGE_SCR * x;
        a = gBeams[at].xyz;
        b = gBeams[at + 1u].xyz;
    }
    else
    {
        a = edgeWorld(v, x);
        b = a;
    }
    float3 r3 = float3(reach, reach, reach);
    lo = min(a, b) - r3;
    hi = max(a, b) + r3;
}
uint edgeCellCount(int3 c0, int3 c1)
{
    int3 n = c1 - c0 + int3(1, 1, 1);
    return (n.x > 4 || n.y > 4 || n.z > 4) ? EDGE_CELLS_MAX + 1u : uint(n.x * n.y * n.z);
}
int3 edgeCellAt(int3 c0, int3 c1, uint k)
{
    int3 n = c1 - c0 + int3(1, 1, 1);
    int ki = int(k);
    return c0 + int3(ki % n.x, (ki / n.x) % n.y, ki / (n.x * n.y));
}

// Vehicle v's grid, one stage (0 zero, 1 count, 2 the blocks' sums, 3 the starts and the
// cursors, 4 scatter); a barrier after each.
void edgeGridStage(uint v, float cell, float reach, uint stage, uint lane)
{
    uint blocks = EDGE_BUCKETS / EDGE_BLOCK;
    if (stage == 0u)
    {
        for (uint x = lane, xStop = EB_END(833, x, 2u * EDGE_BUCKETS, LANES); x < xStop; x += LANES)
            gAcc[edgeGridAt(v, x / EDGE_BUCKETS) + x % EDGE_BUCKETS].store(0);
        if (lane == 0u) gAcc[edgeV1(v).x + 10u].store(0);
        return;
    }
    if (stage == 2u || stage == 3u)
    {
        // A lane a block of EDGE_BLOCK buckets: its sum; then its start (the blocks before it)
        // and its buckets' starts, each bucket's cursor at its start.
        for (uint x = lane, xStop = EB_END(843, x, 2u * blocks, LANES); x < xStop; x += LANES)
        {
            uint kind = x / blocks;
            uint blk = x % blocks;
            uint g = edgeGridAt(v, kind);
            uint sums = edgeBlocksAt(v, kind);
            if (stage == 2u)
            {
                int sum = 0;
                for (uint c = 0u; c < EDGE_BLOCK; c++) sum += gAcc[g + blk * EDGE_BLOCK + c].load();
                gAcc[sums + blk].store(sum);
                continue;
            }
            int run = 0;
            for (uint c = 0u, cStop = EB_END(847, c, blk, 1u); c < cStop; c++) run += gAcc[sums + c].load();
            for (uint c = 0u; c < EDGE_BLOCK; c++)
            {
                uint b = blk * EDGE_BLOCK + c;
                int n = gAcc[g + b].load();
                gAcc[g + EDGE_BUCKETS + b].store(run);
                gAcc[g + b].store(run);
                run += n;
            }
        }
        return;
    }
    for (uint kind = 0u; kind < 2u; kind++)
    {
        uint n = edgeListed(v, kind);
        uint list = edgeListAt(v, kind);
        uint g = edgeGridAt(v, kind);
        uint entries = edgeEntriesAt(v, kind);
        for (uint i = lane, iStop = EB_END(834, i, n, LANES); i < iStop; i += LANES)
        {
            uint x = gNext[list + i];
            float3 lo;
            float3 hi;
            edgeEntryBox(v, kind, x, reach, lo, hi);
            int3 c0 = edgeCellOf(lo, cell);
            int3 c1 = edgeCellOf(hi, cell);
            uint cells = edgeCellCount(c0, c1);
            if (cells > EDGE_CELLS_MAX)
            {
                if (stage == 1u) gNext[edgeLongAt(v) + uint(gAcc[edgeV1(v).x + 10u].add(1))] = x;
                continue;
            }
            for (uint k = 0u, kStop = EB_END(844, k, cells, 1u); k < kStop; k++)
            {
                uint b = edgeBucket(edgeCellAt(c0, c1, k));
                if (stage == 1u) gAcc[g + b].add(1);
                else gNext[entries + uint(gAcc[g + b].add(1))] = x;
            }
        }
    }
}

// Vehicle v as an owner, one stage of its work list (1 each listed edge's cells, 2 the
// blocks' sums, 3 each edge's first item and the total at gAcc +11): the grid's queries as
// (edge, cell) items, a lane an item, so a long edge's cells spread over lanes.
void edgeWorkStage(uint v, float cell, uint stage, uint lane)
{
    if (stage == 0u || stage == 4u) return;
    uint la = edgeListed(v, 0u);
    uint list = edgeListAt(v, 0u);
    uint work = edgeWorkAt(v);
    uint blocks = (la + EDGE_BLOCK - 1u) / EDGE_BLOCK;
    if (stage == 1u)
    {
        for (uint i = lane, iStop = EB_END(848, i, la, LANES); i < iStop; i += LANES)
        {
            uint at = edgeV0(v).z + EDGE_SCR * gNext[list + i];
            float3 p = gBeams[at].xyz;
            float3 q = gBeams[at + 1u].xyz;
            uint cells = edgeCellCount(edgeCellOf(min(p, q), cell), edgeCellOf(max(p, q), cell));
            gNext[work + i] = cells > EDGE_CELLS_MAX ? 0u : cells;
        }
        return;
    }
    for (uint blk = lane, bStop = EB_END(849, blk, blocks, LANES); blk < bStop; blk += LANES)
    {
        uint i0 = blk * EDGE_BLOCK;
        uint i1 = min(i0 + EDGE_BLOCK, la);
        if (stage == 2u)
        {
            uint sum = 0u;
            for (uint i = i0, iStop = EB_END(850, i, i1, 1u); i < iStop; i++) sum += gNext[work + i];
            gNext[edgeWorkBlocksAt(v) + blk] = sum;
            continue;
        }
        uint run = 0u;
        for (uint c = 0u, cStop = EB_END(851, c, blk, 1u); c < cStop; c++) run += gNext[edgeWorkBlocksAt(v) + c];
        for (uint i = i0, iStop = EB_END(852, i, i1, 1u); i < iStop; i++)
        {
            // (The sum before the store: Slang's CUDA output read the element after it.)
            uint next = run + gNext[work + i];
            gNext[work + i] = run;
            run = next;
        }
        if (blk + 1u == blocks) gAcc[edgeV1(v).x + 11u].store(int(run));
    }
    if (stage == 3u && blocks == 0u && lane == 0u) gAcc[edgeV1(v).x + 11u].store(0);
}

// One candidate pair's test into owner va's hits.
void edgeTry(uint va, uint vb, uint slot, uint kind, uint ea, SkinE e, uint x)
{
    if (gEdgeCollect != 0u)
    {
        // A candidate: the boxes within reach + 2 skins.
        float3 loB;
        float3 hiB;
        edgeEntryBox(vb, kind, x, gEdgeCollectReach, loB, hiB);
        if (any(max(e.p, e.q) < loB) || any(hiB < min(e.p, e.q))) return;
        uint n = uint(gAcc[edgeCandAcc(va)].add(1));
        if (n >= edgeCandRoom(va))
        {
            gAcc[edgeCandAcc(va) + 2u].store(1);
            return;
        }
        uint at = edgeCandAt(va) + 2u * n;
        gNext[at] = kind == 0u ? ea : (ea | 0x80000000u);
        gNext[at + 1u] = kind == 0u ? ((vb << 20u) | x) : x;
        return;
    }
    EdgeHitG H;
    if (kind == 0u)
    {
        if (edgeEdgeTest(va, vb, e, edgeLoad(vb, x), H)) edgeHitAdd(va, slot, ea, (vb << 20u) | x, H);
    }
    else if (edgeVertexTest(va, vb, e, x, edgeWorld(vb, x), H))
        edgeHitAdd(va, slot, ea | 0x80000000u, x, H);
}

// Owner va (two-body slot `slot`) against partner vb, its grid of side `cell`: edge against
// edge (va < vb) and vb's points against va's edges. The long edges (either side) a lane an
// owner edge; the grid a lane an (owner edge, cell) item of va's work list.
void edgePairHits(uint va, uint vb, uint slot, float cell, float reach, uint lane)
{
    if ((edgeAblate() & 1u) != 0u) return;
    uint la = edgeListed(va, 0u);
    uint alist = edgeListAt(va, 0u);
    uint longs = uint(gAcc[edgeV1(vb).x + 10u].load());
    for (uint i = lane, iStop = EB_END(835, i, la, LANES); i < iStop; i += LANES)
    {
        uint ea = gNext[alist + i];
        SkinE e = edgeLoad(va, ea);
        uint cells = edgeCellCount(edgeCellOf(min(e.p, e.q), cell), edgeCellOf(max(e.p, e.q), cell));
        for (uint kind = 0u; kind < 2u; kind++)
        {
            if (kind == 0u && !(va < vb)) continue;
            if (cells > EDGE_CELLS_MAX)
            {
                // A long owner edge: every listed entry of the partner.
                uint n = edgeListed(vb, kind);
                uint list = edgeListAt(vb, kind);
                for (uint j = 0u, jStop = EB_END(836, j, n, 1u); j < jStop; j++) edgeTry(va, vb, slot, kind, ea, e, gNext[list + j]);
            }
            else if (kind == 0u)
            {
                // The partner's long edges.
                for (uint j = 0u, jStop = EB_END(837, j, longs, 1u); j < jStop; j++) edgeTry(va, vb, slot, 0u, ea, e, gNext[edgeLongAt(vb) + j]);
            }
        }
    }
    uint total = uint(gAcc[edgeV1(va).x + 11u].load());
    uint work = edgeWorkAt(va);
    for (uint w = lane, wStop = EB_END(853, w, total, LANES); w < wStop; w += LANES)
    {
        // The item's edge: the last listed edge whose first item is at or before it.
        uint lo = 0u;
        uint hi = la;
        for (uint it = 0u; it < 16u; it++)
        {
            if (hi - lo <= 1u) break;
            uint mid = (lo + hi) / 2u;
            if (gNext[work + mid] <= w) lo = mid;
            else hi = mid;
        }
        uint ea = gNext[alist + lo];
        SkinE e = edgeLoad(va, ea);
        float3 loA = min(e.p, e.q);
        float3 hiA = max(e.p, e.q);
        int3 c = edgeCellAt(edgeCellOf(loA, cell), edgeCellOf(hiA, cell), w - gNext[work + lo]);
        uint b = edgeBucket(c);
        for (uint kind = 0u; kind < 2u; kind++)
        {
            if (kind == 0u && !(va < vb)) continue;
            uint g = edgeGridAt(vb, kind);
            uint entries = edgeEntriesAt(vb, kind);
            uint end = uint(gAcc[g + b].load());
            for (uint j = uint(gAcc[g + EDGE_BUCKETS + b].load()), jStop = EB_END(838, j, end, 1u); j < jStop; j++)
            {
                uint x = gNext[entries + j];
                float3 loB;
                float3 hiB;
                edgeEntryBox(vb, kind, x, reach, loB, hiB);
                if (any(hiA < loB) || any(hiB < loA)) continue;
                if (any(edgeCellOf(max(loA, loB), cell) != c)) continue;
                edgeTry(va, vb, slot, kind, ea, e, x);
            }
        }
    }
}

// The larger groups' pass (vvPass, owner by owner): owner a's turn, after the previous
// owner's split counts are cleared and before its own hits apply (gNext's vvPartnerAt its
// partners). The lists: the owner's edges near its partners' edges, each partner's edges
// and points near the owner's; then its hits. Three barriers.
void edgeOwnerBig(uint a, uint k, uint off, uint lane)
{
    uint va = vvMember(off, a);
    for (uint m = lane, mStop = EB_END(829, m, k, LANES); m < mStop; m += LANES)
    {
        uint v1x = edgeV1(vvMember(off, m)).x;
        gAcc[v1x + 1u].store(0);
        gAcc[v1x + 2u].store(0);
    }
    sync();
    float pad = 0.0;
    float3 ulo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 uhi = -ulo;
    for (uint m = 0u, mStop = EB_END(830, m, k, 1u); m < mStop; m++)
    {
        uint vb = vvMember(off, m);
        if (m == a || gNext[vvPartnerAt(off, m)] == 0u || !edgePairOn(va, vb)) continue;
        pad = max(pad, nf(edgeRange(va, vb) + EDGE_DEPTH_M));
        float3 lo;
        float3 hi;
        edgeBox(vb, lo, hi);
        ulo = min(ulo, lo);
        uhi = max(uhi, hi);
    }
    float3 p3 = float3(pad, pad, pad);
    edgeList(va, ulo - p3, uhi + p3, false, lane);
    float3 olo;
    float3 ohi;
    edgeBox(va, olo, ohi);
    for (uint m = 0u, mStop = EB_END(831, m, k, 1u); m < mStop; m++)
    {
        uint vb = vvMember(off, m);
        if (m == a || gNext[vvPartnerAt(off, m)] == 0u || !edgePairOn(va, vb)) continue;
        edgeList(vb, olo - p3, ohi + p3, true, lane);
    }
    for (uint stage = 0u; stage < 5u; stage++)
    {
        for (uint m = 0u, mStop = EB_END(840, m, k, 1u); m < mStop; m++)
        {
            uint vb = vvMember(off, m);
            if (m == a || gNext[vvPartnerAt(off, m)] == 0u || !edgePairOn(va, vb)) continue;
            edgeGridStage(vb, 2.0 * pad, pad, stage, lane);
        }
        edgeWorkStage(va, 2.0 * pad, stage, lane);
        sync();
    }
    for (uint m = 0u, mStop = EB_END(832, m, k, 1u); m < mStop; m++)
    {
        uint vb = vvMember(off, m);
        if (m == a || gNext[vvPartnerAt(off, m)] == 0u || !edgePairOn(va, vb)) continue;
        edgePairHits(va, vb, 0u, 2.0 * pad, pad, lane);
    }
    sync();
}

// --- The small groups' pass (the split path's edgePass: k ≤ VV_PART; gsVvPair the
// overlapping ordered pairs) ---

// Per member its partners this step (overlapping pairs with edge contact; a bit a member).
groupshared uint gsEdgePart[VV_PART];
bool edgePartnerOf(uint k, uint off, uint a, uint m) { return ((gsEdgePart[a] >> m) & 1u) != 0u; }

// Each member of `need` its edges and contact points within `reach` of a partner's edges'
// box (every list's counts cleared first, and the candidates' of the members `clear`). Two
// barriers.
void edgeListsSmall(uint k, uint off, float reach, uint need, uint clear, uint lane)
{
    for (uint a = lane, aStop = EB_END(866, a, k, LANES); a < aStop; a += LANES)
    {
        uint v1x = edgeV1(vvMember(off, a)).x;
        gAcc[v1x + 1u].store(0);
        gAcc[v1x + 2u].store(0);
        if (((clear >> a) & 1u) == 0u) continue;
        uint ca = edgeCandAcc(vvMember(off, a));
        gAcc[ca].store(0);
        gAcc[ca + 2u].store(0);
    }
    sync();
    for (uint a = 0u, aStop = EB_END(817, a, k, 1u); a < aStop; a++)
    {
        if (((need >> a) & 1u) == 0u) continue;
        uint va = vvMember(off, a);
        float3 ulo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 uhi = -ulo;
        for (uint m = 0u, mStop = EB_END(818, m, k, 1u); m < mStop; m++)
        {
            if (!edgePartnerOf(k, off, a, m)) continue;
            float3 lo;
            float3 hi;
            edgeBox(vvMember(off, m), lo, hi);
            ulo = min(ulo, lo);
            uhi = max(uhi, hi);
        }
        if (any(ulo > uhi)) continue;
        float3 r3 = float3(reach, reach, reach);
        edgeList(va, ulo - r3, uhi + r3, true, lane);
    }
    sync();
}

// Every owner of `owners` against each partner (the grids built). One barrier.
void edgePairsSmall(uint k, uint off, float cell, float reach, uint owners, uint lane)
{
    for (uint a = 0u, aStop = EB_END(819, a, k, 1u); a < aStop; a++)
    {
        if (((owners >> a) & 1u) == 0u) continue;
        uint va = vvMember(off, a);
        for (uint m = 0u, mStop = EB_END(820, m, k, 1u); m < mStop; m++)
        {
            if (!edgePartnerOf(k, off, a, m)) continue;
            edgePairHits(va, vvMember(off, m), a, cell, reach, lane);
        }
    }
    sync();
}

// --- The kept candidates, per owner (lossless: the same hits as a search every step) ---
// Owner a's candidates, built at reach + 2 skins, hold every pair in contact while no node
// pair (one of a, one of a partner) has moved more than two skins apart since a's build: its
// own nodes then (gBeams) and its partners' (its keep record) taken off, about one vector
// c_a (the mean motion of the first nodes), a's nodes within D_a and each partner's within
// D_b, the list stands while D_a + D_b ≤ 2 skins for every partner (the partners those of
// the build, in its order). Owners build alone: another's build moves nothing of a's.
// Per owner (a) and slot (0 its own nodes, 1 + p its partner p) the squared bound's bits.
groupshared Atomic<int> gsEdgeDm[VV_PART * VV_PART];

// A vehicle's node position as kept in a record (three words at `at`).
float3 edgeKeptNode(uint at) { return float3(asfloat(gNext[at]), asfloat(gNext[at + 1u]), asfloat(gNext[at + 2u])); }

// Every kept owner's D (gsEdgeDm). Two barriers. (Any c_a is valid, so plain arithmetic: the
// bound needs only the same c_a in each of a's D.)
void edgeMotion(uint k, uint off, uint lane)
{
    for (uint x = lane, xStop = EB_END(854, x, VV_PART * VV_PART, LANES); x < xStop; x += LANES) gsEdgeDm[x].store(0);
    sync();
    for (uint a = 0u, aStop = EB_END(889, a, k, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        if (!edgeKeepable(va) || gAcc[edgeCandAcc(va) + 3u].load() != EDGE_KEEP_MAGIC) continue;
        uint s = edgeKeepAt(va);
        uint n = min(gNext[s + 1u], EDGE_KEEP_PARTS);
        // Slot q: 0 the owner (its nodes at gBeams `built`), 1 + p partner p (at `at`).
        float3 c = float3(0.0, 0.0, 0.0);
        for (uint pass = 0u; pass < 2u; pass++)
        {
            uint at = s + EDGE_KEEP;
            for (uint q = 0u, qStop = EB_END(857, q, n + 1u, 1u); q < qStop; q++)
            {
                uint v = q == 0u ? va : gNext[s + 1u + q];
                uint4 r0 = gStatic[v * 4u];
                uint built = edgeBuiltAt(va);
                if (pass == 0u)
                {
                    c += edgeWorld(v, r0.x) - (q == 0u ? gBeams[built].xyz : edgeKeptNode(at));
                    at += q == 0u ? 0u : 3u * r0.y;
                    continue;
                }
                float dd = 0.0;
                for (uint i = lane, iStop = EB_END(855, i, r0.y, LANES); i < iStop; i += LANES)
                {
                    float3 d = edgeWorld(v, r0.x + i) - (q == 0u ? gBeams[built + i].xyz : edgeKeptNode(at + 3u * i)) - c;
                    dd = max(dd, dot(d, d));
                }
                gsEdgeDm[a * VV_PART + q].max(asint(dd));
                at += q == 0u ? 0u : 3u * r0.y;
            }
            c /= float(n + 1u);
        }
    }
    sync();
}

// Owner a's kept list stands (lane 0, after edgeMotion): kept with its partners' nodes, the
// same partners in the same order, each pair's bound in two skins.
bool edgeKeeps(uint k, uint off, uint a)
{
    uint va = vvMember(off, a);
    if (!edgeKeepable(va) || gAcc[edgeCandAcc(va) + 3u].load() != EDGE_KEEP_MAGIC) return false;
    uint s = edgeKeepAt(va);
    // (A margin for the bound's rounding: 0.1 %.)
    float limit = 2.0 * edgeSkin() * 0.999;
    float own = sqrt(asfloat(uint(gsEdgeDm[a * VV_PART].load())));
    uint n = 0u;
    bool keep = gNext[s] == 1u;
    for (uint m = 0u, mStop = EB_END(856, m, k, 1u); m < mStop; m++)
    {
        if (!edgePartnerOf(k, off, a, m)) continue;
        if (n >= EDGE_KEEP_PARTS) return false;
        n++;
        if (gNext[s + 1u + n] != vvMember(off, m)) keep = false;
        if (!(own + sqrt(asfloat(uint(gsEdgeDm[a * VV_PART + n].load()))) <= limit)) keep = false;
    }
    return keep && n == gNext[s + 1u];
}

// The members whose lists are built this step (lane 0, after edgeMotion; a bit a member, and
// in bits 8.. those and their partners), each built owner's keep record written: its partners
// now, its partners' nodes' words (kept when they fit in half its candidates' region).
uint edgeBuilds(uint k, uint off)
{
    uint builds = 0u;
    for (uint a = 0u, aStop = EB_END(894, a, k, 1u); a < aStop; a++)
    {
        if ((edgeAblate() & 16u) == 0u && edgeKeeps(k, off, a)) continue;
        builds |= ((1u << a) | gsEdgePart[a]) << 8u | (1u << a);
        uint va = vvMember(off, a);
        if (!edgeKeepable(va)) continue;
        uint s = edgeKeepAt(va);
        uint n = 0u;
        uint words = 0u;
        for (uint m = 0u, mStop = EB_END(895, m, k, 1u); m < mStop; m++)
        {
            if (!edgePartnerOf(k, off, a, m) || n >= EDGE_KEEP_PARTS) continue;
            n++;
            gNext[s + 1u + n] = vvMember(off, m);
            words += 3u * gStatic[vvMember(off, m) * 4u].y;
        }
        bool fits = n == uint(countbits(gsEdgePart[a])) && 2u * (EDGE_KEEP + words) <= edgeRegion(va);
        gNext[s] = fits ? 1u : 0u;
        gNext[s + 1u] = n;
        gNext[s + 9u] = fits ? words : 0u;
    }
    return builds;
}

// The built owners' nodes and their partners' into their records (after their pairs are
// collected), and their marks. One barrier.
void edgeKeepNote(uint k, uint off, uint builds, uint lane)
{
    for (uint a = 0u, aStop = EB_END(859, a, k, 1u); a < aStop; a++)
    {
        if (((builds >> a) & 1u) == 0u) continue;
        uint va = vvMember(off, a);
        uint s = edgeKeepAt(va);
        uint n = edgeKeepable(va) && gNext[s] == 1u ? gNext[s + 1u] : 0u;
        uint at = s + EDGE_KEEP;
        for (uint q = 0u, qStop = EB_END(896, q, n + 1u, 1u); q < qStop; q++)
        {
            uint v = q == 0u ? va : gNext[s + 1u + q];
            uint4 r0 = gStatic[v * 4u];
            uint built = edgeBuiltAt(va);
            for (uint i = lane, iStop = EB_END(860, i, r0.y, LANES); i < iStop; i += LANES)
            {
                float3 x = edgeWorld(v, r0.x + i);
                if (q == 0u)
                {
                    gBeams[built + i] = float4(x, 0.0);
                    continue;
                }
                gNext[at + 3u * i] = asuint(x.x);
                gNext[at + 3u * i + 1u] = asuint(x.y);
                gNext[at + 3u * i + 2u] = asuint(x.z);
            }
            at += q == 0u ? 0u : 3u * r0.y;
        }
        if (lane != 0u) continue;
        uint ca = edgeCandAcc(va);
        gAcc[ca + 1u].store(0);
        gAcc[ca + 3u].store((n == 0u && gsEdgePart[a] != 0u) || gAcc[ca + 2u].load() != 0 ? 0 : EDGE_KEEP_MAGIC);
    }
    sync();
}

// The search for the owners `owners` (their lists and their partners', `need`, at `reach`, the
// grids, the pairs: collected within `reach` with `collect`, else hits). Barriers: uniform masks.
void edgeSearch(uint k, uint off, uint owners, uint need, float reach, bool collect, uint lane)
{
    edgeListsSmall(k, off, reach, need, collect ? owners : 0u, lane);
    for (uint stage = 0u; stage < 5u; stage++)
    {
        for (uint a = 0u, aStop = EB_END(842, a, k, 1u); a < aStop; a++)
        {
            if (((need >> a) & 1u) == 0u) continue;
            edgeGridStage(vvMember(off, a), 2.0 * reach, reach, stage, lane);
            edgeWorkStage(vvMember(off, a), 2.0 * reach, stage, lane);
        }
        sync();
    }
    gEdgeCollect = collect ? 1u : 0u;
    gEdgeCollectReach = reach;
    edgePairsSmall(k, off, 2.0 * reach, reach, owners, lane);
    gEdgeCollect = 0u;
}

// Before the tests' end: every member's edges this step, their boxes, the lists of each
// member's edges and contact points near its partners, then each owner's hits.
void edgeTestsSmall(uint k, uint off, uint parity, uint lane)
{
    if (!edgeOn()) return;
    // (No early return on the pairs: they are workgroup memory, and a barrier after a
    // branch on it is not uniform control flow to a browser. edgePass calls this only with a
    // pair.)
    for (uint a = lane, aStop = EB_END(812, a, k, LANES); a < aStop; a += LANES)
    {
        uint part = 0u;
        for (uint m = 0u, mStop = EB_END(813, m, k, 1u); m < mStop; m++)
            if (m != a && (gsVvPair[a * VV_PART + m] & 1u) != 0u && edgePairOn(vvMember(off, a), vvMember(off, m))) part |= 1u << m;
        gsEdgePart[a] = part;
    }
    sync();
    float rmax = 0.0;
    for (uint a = 0u, aStop = EB_END(816, a, k, 1u); a < aStop; a++) rmax = max(rmax, asfloat(gStatic[vvMember(off, a) * 4u + 3u].z));
    float pad = nf(rmax + EDGE_DEPTH_M);
    gEdgeNearK = k;
    edgeNearBoxes(k, off, parity, pad + 2.0 * edgeSkin() + 0.01, lane);
    edgeResetPrepare(k, off, lane, true);
    gEdgeNearK = 0u;
    uint kk = k;
    edgeMotion(kk, off, lane);
    if (lane == 0u) gsUniformLanes = edgeBuilds(kk, off);
    sync();
    uint builds = gsUniformLanes & 0xFFu;
    uint need = gsUniformLanes >> 8u;
    // The builds: their pairs within reach + 2 skins collected; an owner whose list
    // overflowed finds its hits by the search itself this step (and builds again next).
    uint over = 0u;
    for (uint pass = 0u; pass < 2u; pass++)
    {
        bool collect = pass == 0u;
        if (!collect)
        {
            if (lane == 0u)
            {
                uint o = 0u;
                for (uint a = 0u, aStop = EB_END(862, a, kk, 1u); a < aStop; a++)
                    if (((builds >> a) & 1u) != 0u && gAcc[edgeCandAcc(vvMember(off, a)) + 2u].load() != 0) o |= ((1u << a) | gsEdgePart[a]) << 8u | (1u << a);
                gsUniformLanes = o;
            }
            sync();
            over = gsUniformLanes & 0xFFu;
            need = gsUniformLanes >> 8u;
        }
        uint owners = collect ? builds : over;
        if (owners == 0u) continue;
        edgeSearch(kk, off, owners, need, collect ? nf(pad + 2.0 * edgeSkin()) : pad, collect, lane);
        if (!collect) continue;
        // The builds' nodes.
        edgeKeepNote(kk, off, builds, lane);
    }
    // The kept and built candidates tested, each owner's on all lanes.
    for (uint a = 0u, aStop = EB_END(864, a, kk, 1u); a < aStop; a++)
    {
        if (((over >> a) & 1u) != 0u) continue;
        uint va = vvMember(off, a);
        uint n = min(uint(gAcc[edgeCandAcc(va)].load()), edgeCandRoom(va));
        uint base = edgeCandAt(va);
        for (uint i = lane, iStop = EB_END(865, i, n, LANES); i < iStop; i += LANES)
        {
            EdgeHitG H;
            if (edgeHitAt(va, base + 2u * i, H)) edgeHitAdd(va, a, gNext[base + 2u * i], gNext[base + 2u * i + 1u], H);
        }
    }
    sync();
}

// The snapshot of owner va's edge hits' nodes (the levels', before any hit applies).
void edgeSnapOwner(uint va, uint parity, uint tag, uint lane, uint lanes)
{
    uint n = edgeHits(va);
    uint base = edgeV0(va).w;
    for (uint i = lane, iStop = EB_END(808, i, n, lanes); i < iStop; i += lanes)
    {
        EdgeHitG H;
        if (!edgeHitAt(va, base + 2u * i, H)) continue;
        for (uint q = 0u, qStop = EB_END(809, q, opaqueCount(4u), 1u); q < qStop; q++)
            vvSnapNode(q == 0u ? H.a0 : (q == 1u ? H.a1 : (q == 2u ? H.h : H.h2)), q < 2u ? H.va : H.vb, parity, tag);
    }
}

// Owner va's edge hits applied (its level, after its node hits).
void edgeApplyOwner(uint va, uint slot, uint lane, uint lanes)
{
    uint n = edgeHits(va);
    uint base = edgeV0(va).w;
    for (uint i = lane, iStop = EB_END(810, i, n, lanes); i < iStop; i += lanes)
    {
        EdgeHitG H;
        if (edgeHitAt(va, base + 2u * i, H)) edgeApplyOne(H, slot);
    }
}

// Owner va's edge hits' split counts cleared (after every level).
void edgeClearOwner(uint va, uint slot, uint lane, uint lanes)
{
    uint n = edgeHits(va);
    uint base = edgeV0(va).w;
    for (uint i = lane, iStop = EB_END(811, i, n, lanes); i < iStop; i += lanes)
    {
        EdgeHitG H;
        if (!edgeHitAt(va, base + 2u * i, H)) continue;
        gAcc[vvSplitTri(H.a1)].store(0);
        gAcc[vvSplitTri(H.a0)].store(0);
        gAcc[vvSplitPoint(slot, H.h)].store(0);
        gAcc[vvSplitPoint(slot, H.h2)].store(0);
    }
}
