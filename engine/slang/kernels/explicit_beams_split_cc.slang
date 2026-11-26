// --- Continuous contact's pass (docs/continuous-contact-design.md, "The GPU port"): between a
// step's node update and its rest, a workgroup a contact group. The CPU reference:
// crates/engine-softbody/src/explicit_contact/dat.rs (dat_truncate, list_pairs, speculative)
// and ccd.rs (resolve, rigid_zone), operation for operation in strict float; the per-node sums
// in fixed point (order free). Tables: engine_exec::explicit_plan::continuous::cc_tables. ---

#include "explicit_beams_cc_math.slang"

static const uint CC_BEAMS_NODE = 6u;
static const uint CC_ACC_NODE = 9u;
static const uint CC_VEH_WORDS = 16u;
static const uint CC_PAIR_WORDS = 16u;
static const uint CC_GRID_BUCKETS = 4096u;
// The grids' cells (m): about a primitive's size; each primitive in the cells its own box
// touches (not grown: the queries are grown by their threshold).
static const float CC_CELL = 0.2;
// The CPU's Dat::new(0.08, 0.85) and its rounds: 8 speculative, 8 impulse, 16 zone rounds.
static const uint CC_SPEC_ROUNDS = 8u;
static const uint CC_CCD_ROUNDS = 8u;
static const uint CC_ZONE_ROUNDS = 16u;
static const float CC_MIN_GAP = 0.0005;
// Counters (gAcc at gP.ccCounts, after the vehicles' words): 0 overflow (pairs or grid),
// 1 impacts, 2 zones, 3 unresolved, 4 zones past 1 rad, 5 nodes past the listing's reach;
// 7 the workgroup's flag.
static const uint CC_C_OVERFLOW = 0u;
static const uint CC_C_HITS = 1u;
static const uint CC_C_ZONES = 2u;
static const uint CC_C_UNRESOLVED = 3u;
static const uint CC_C_SPIN = 4u;
static const uint CC_C_REACH = 5u;

uint4 ccRec(uint k) { return gStatic[gP.cc + k]; }
uint4 ccVeh(uint v, uint k) { return gStatic[gP.cc + 2u + 3u * v + k]; }
uint ccB(uint gi, uint k) { return ccRec(0).x + CC_BEAMS_NODE * gi + k; }
uint ccA(uint gi, uint k) { return ccRec(0).y + CC_ACC_NODE * gi + k; }
uint ccC(uint k) { return gP.ccCounts + CC_VEH_WORDS * ccRec(0).w + k; }
uint ccVW(uint v, uint k) { return gP.ccCounts + CC_VEH_WORDS * v + k; }
uint ccCount(uint v) { return ccVW(v, 0u); }
// The vehicle's surface reach this step and largest offset (ccMeasure).
float ccReachV(uint v) { return asfloat(gAcc[ccVW(v, 1u)].load()); }
float3 ccOrigin(uint v) { return asfloat(gMisc[v].xyz); }
// World positions: at the step's start (the snapshot), now (x1), before the pass.
float3 ccX0Snap(uint v, uint gi) { return nf3(ccOrigin(v) + gBeams[ccB(gi, 0u)].xyz); }
// (World x0, written once a pass in the scratch's slot 5.)
float3 ccX0(uint v, uint gi) { return gBeams[ccB(gi, 5u)].xyz; }
float3 ccX1(uint gi) { return gBeams[ccB(gi, 1u)].xyz; }
float3 ccVel(uint gi) { return gBeams[ccB(gi, 3u)].xyz; }
float ccIm(uint gi) { return gBeams[ccB(gi, 1u)].w; }
float ccImOf(uint gi)
{
    if ((gStatic[gP.nodeInfoBase + gi].x & NODE_IMMOVABLE) != 0u || (gP.retireBase != 0u && nodeRetired(gi))) return 0.0;
    return sdiv(1.0, gNodes[gi * 3u].w);
}
uint ccVehicleOf(uint gi, uint off, uint k)
{
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        if (gi >= r0.x && gi < r0.x + r0.y) return v;
    }
    return 0u;
}
// A live surface triangle (the CPU's surface(): not torn, not anti-clip only, no retired node).
bool ccTriLive(uint gt)
{
    if ((gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u || triangleDead(gt)) return false;
    uint4 tn = gStatic[gP.triBase + gt];
    return gP.retireBase == 0u || !(nodeRetired(tn.x) || nodeRetired(tn.y) || nodeRetired(tn.z));
}
bool ccEdgeLive(uint e)
{
    uint4 r = gStatic[ccRec(1).x + e];
    for (uint t = 0u; t < r.w; t++)
        if (ccTriLive(gLists[r.z + t])) return true;
    return false;
}

// A pair's four nodes and its first side's count (3: a triangle then a point; 2: two edges),
// its vehicles (a the second side's, b the first's) and whether it is a car's own.
struct CcPair
{
    uint k[4];
    uint nn;
    bool own;
    uint va;
    uint vb;
};
uint ccPairAt(uint v, uint p) { return ccVeh(v, 0u).z + CC_PAIR_WORDS * p; }
CcPair ccLoad(uint v, uint p)
{
    uint at = ccPairAt(v, p);
    CcPair q;
    for (uint j = 0u; j < 4u; j++) q.k[j] = gNext[at + j];
    uint f = gNext[at + 4u];
    q.nn = f & 0xFFu;
    q.own = (f & 0x100u) != 0u;
    q.va = gNext[at + 5u] & 0xFFFFu;
    q.vb = gNext[at + 5u] >> 16;
    return q;
}

// Rec::new: the pair's closest points at the start (cn on the first side, cp the second's),
// weights (unsigned), its gap, unit normal and thickness (the sides' offsets along the normal,
// or the own clearance).
struct CcRec
{
    float w[4];
    float3 n;
    float3 cn;
    float g;
    float d;
};
float ccSign(uint nn, uint j) { return j < nn ? -1.0 : 1.0; }
// The record beside its pair (words 6..14: weights, normal, gap, thickness), once at listing.
void ccStoreRec(uint at, CcRec r)
{
    for (uint j = 0u; j < 4u; j++) gNext[at + 6u + j] = asuint(r.w[j]);
    gNext[at + 10u] = asuint(r.n.x);
    gNext[at + 11u] = asuint(r.n.y);
    gNext[at + 12u] = asuint(r.n.z);
    gNext[at + 13u] = asuint(r.g);
    gNext[at + 14u] = asuint(r.d);
}
CcRec ccRecAt(uint v, uint p)
{
    uint at = ccPairAt(v, p);
    CcRec r;
    for (uint j = 0u; j < 4u; j++) r.w[j] = asfloat(gNext[at + 6u + j]);
    r.n = float3(asfloat(gNext[at + 10u]), asfloat(gNext[at + 11u]), asfloat(gNext[at + 12u]));
    r.cn = float3(0.0, 0.0, 0.0);
    r.g = asfloat(gNext[at + 13u]);
    r.d = asfloat(gNext[at + 14u]);
    return r;
}

void ccStore(uint v, CcPair q, CcRec r)
{
    uint p = asuint(gAcc[ccCount(v)].add(1));
    if (p >= ccVeh(v, 0u).w)
    {
        gAcc[ccC(CC_C_OVERFLOW)].store(1);
        return;
    }
    uint at = ccPairAt(v, p);
    for (uint j = 0u; j < 4u; j++) gNext[at + j] = q.k[j];
    gNext[at + 4u] = q.nn | (q.own ? 0x100u : 0u);
    gNext[at + 5u] = q.va | (q.vb << 16);
    ccStoreRec(at, r);
}
CcRec ccRecOf(CcPair q)
{
    CcRec r;
    float3 p[4];
    for (uint j = 0u; j < 4u; j++) p[j] = ccX0(j < q.nn ? q.vb : q.va, q.k[j]);
    float3 cn;
    float3 cp;
    if (q.nn == 3u)
    {
        float3 b;
        cn = fqClosestTri(p[3], p[0], p[1], p[2], b);
        cp = p[3];
        r.w[0] = b.x; r.w[1] = b.y; r.w[2] = b.z; r.w[3] = 1.0;
    }
    else
    {
        float2 st = fqSegSeg(p[0], p[1], p[2], p[3]);
        cn = sadd3(p[0], sscale(ssub3(p[1], p[0]), st.x));
        cp = sadd3(p[2], sscale(ssub3(p[3], p[2]), st.y));
        r.w[0] = nf(1.0 - st.x); r.w[1] = st.x; r.w[2] = nf(1.0 - st.y); r.w[3] = st.y;
    }
    r.cn = cn;
    float3 dd = ssub3(cp, cn);
    r.g = ccLen(dd);
    float gm = max(r.g, 1.17549435e-38);
    r.n = float3(sdiv(dd.x, gm), sdiv(dd.y, gm), sdiv(dd.z, gm));
    if (q.own) r.d = asfloat(ccRec(1).w);
    else
    {
        float d = 0.0;
        for (uint j = 0u; j < 4u; j++)
        {
            float o = j < q.nn ? sdot(contactOff(q.k[j]), r.n) : -sdot(contactOff(q.k[j]), r.n);
            d = nf(d + smul(r.w[j], max(o, 0.0)));
        }
        r.d = d;
    }
    return r;
}

// The grid of vehicle v's live primitives (triangles, kind 0; edges, kind 1), each in every
// cell its box at the start grown by `grow` touches, the cells `cell` wide.
uint ccHash(int3 c, uint kind) { return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u ^ kind * 2654435761u) % CC_GRID_BUCKETS; }
uint ccPack(int3 c) { return (uint(c.x) & 1023u) | ((uint(c.y) & 1023u) << 10) | ((uint(c.z) & 1023u) << 20); }
int3 ccCell(float3 p, float cell) { return int3(floor(p / cell)); }
void ccPrimBox(uint v, uint kind, uint id, out float3 lo, out float3 hi)
{
    if (kind == 0u)
    {
        uint4 tn = gStatic[gP.triBase + id];
        float3 a = ccX0(v, tn.x);
        float3 b = ccX0(v, tn.y);
        float3 c = ccX0(v, tn.z);
        lo = min(min(a, b), c);
        hi = max(max(a, b), c);
    }
    else
    {
        uint4 r = gStatic[ccRec(1).x + id];
        float3 a = ccX0(v, r.x);
        float3 b = ccX0(v, r.y);
        lo = min(a, b);
        hi = max(a, b);
    }
}
// A primitive's nodes' largest contact offset (m): a pair's thickness is at most its two
// primitives' (Rec::new's d, a weighted sum of offsets along the normal).
// (Bounds only, never a pair's own value: grown by 0.1 % and 10 µm against the thickness's
// rounding, so the candidates stay a superset of the pairs listed.)
float ccSlack(float o) { return o * 1.001 + 1e-5; }
float ccNodeOff(uint gi) { return gBeams[ccB(gi, 5u)].w; }
float ccPrimOff(uint kind, uint id)
{
    if (kind == 0u)
    {
        uint4 tn = gStatic[gP.triBase + id];
        return ccSlack(max(max(ccNodeOff(tn.x), ccNodeOff(tn.y)), ccNodeOff(tn.z)));
    }
    uint4 r = gStatic[ccRec(1).x + id];
    return ccSlack(max(ccNodeOff(r.x), ccNodeOff(r.y)));
}
// The primitive's box at the start grown by its offset: what the grid holds it by.
void ccPrimBoxOff(uint v, uint kind, uint id, out float3 lo, out float3 hi, out float o)
{
    ccPrimBox(v, kind, id, lo, hi);
    o = ccPrimOff(kind, id);
    lo = ssub3(lo, float3(o, o, o));
    hi = sadd3(hi, float3(o, o, o));
}
// The grids, built across the GPU in three dispatches (count, prefix, fill) after the prep
// cleared their counts. A primitive: triangle i < tris, else edge i − tris.
bool ccGridPrim(uint v, uint i, out uint kind, out uint id)
{
    uint4 r1 = gStatic[v * 4u + 1u];
    kind = i < r1.y ? 0u : 1u;
    id = kind == 0u ? r1.x + i : ccVeh(v, 0u).x + (i - r1.y);
    return kind == 0u ? ccTriLive(id) : ccEdgeLive(id);
}
uint ccGridPrims(uint v) { return gStatic[v * 4u + 1u].y + ccVeh(v, 0u).y; }
void ccGridClear(uint v, uint lane)
{
    uint ba = ccVeh(v, 2u).x;
    for (uint b = lane; b < 2u * CC_GRID_BUCKETS; b += LANES) gAcc[ba + b].store(0);
}
// Pass 0 counts each bucket's entries, pass 1 writes them at their cursors; workgroup slice
// `slice` of `slices` takes its share of v's primitives.
void ccGridPass(uint v, uint pass, uint slice, uint slices, uint lane)
{
    uint g0 = ccVeh(v, 1u).x;
    uint cap = ccVeh(v, 1u).y;
    uint ba = ccVeh(v, 2u).x;
    uint ws = LANES * slices;
    for (uint i = lane + LANES * slice, iStop = EB_END(893, i, ccGridPrims(v), ws); i < iStop; i += ws)
    {
        uint kind;
        uint id;
        if (!ccGridPrim(v, i, kind, id)) continue;
        float3 lo;
        float3 hi;
        float o;
        ccPrimBoxOff(v, kind, id, lo, hi, o);
        int3 c0 = ccCell(lo, CC_CELL);
        int3 c1 = ccCell(hi, CC_CELL);
        for (int x = c0.x; x <= c1.x; x++)
            for (int y = c0.y; y <= c1.y; y++)
                for (int z = c0.z; z <= c1.z; z++)
                {
                    uint h = ccHash(int3(x, y, z), kind);
                    if (pass == 0u) { gAcc[ba + h].add(1); continue; }
                    uint slot = asuint(gAcc[ba + CC_GRID_BUCKETS + h].add(1));
                    if (slot >= cap) { gAcc[ccC(CC_C_OVERFLOW)].store(1); continue; }
                    gNext[g0 + 2u * slot] = id | (kind << 31);
                    gNext[g0 + 2u * slot + 1u] = ccPack(int3(x, y, z));
                }
    }
}
// The buckets' starts (a workgroup a vehicle, two levels): the cursors set to them, and each
// bucket's range [start, start + count) in plain memory for the queries.
void ccGridPrefix(uint v, uint lane)
{
    uint g0 = ccVeh(v, 1u).x;
    uint cap = ccVeh(v, 1u).y;
    uint ba = ccVeh(v, 2u).x;
    uint per = CC_GRID_BUCKETS / LANES;
    int sum = 0;
    for (uint j = 0u; j < per; j++) sum += gAcc[ba + lane * per + j].load();
    // (The lanes' sums scanned in workgroup memory: lane 0's walk over them costs no device
    // round trips.)
    gsPart[lane] = uint(sum);
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint l = 0u; l < LANES; l++)
        {
            uint c = gsPart[l];
            gsPart[l] = run;
            run += c;
        }
    }
    sync();
    int run = int(gsPart[lane]);
    for (uint j = 0u; j < per; j++)
    {
        uint b = lane * per + j;
        int c = gAcc[ba + b].load();
        gAcc[ba + CC_GRID_BUCKETS + b].store(run);
        gNext[g0 + 2u * cap + 2u * b] = uint(run);
        gNext[g0 + 2u * cap + 2u * b + 1u] = min(uint(run + c), cap);
        run += c;
    }
}
// A bucket's entries: from its start to its cursor (where the fill left it).
uint2 ccBucket(uint v, uint h)
{
    uint at = ccVeh(v, 1u).x + 2u * ccVeh(v, 1u).y + 2u * h;
    return uint2(gNext[at], gNext[at + 1u]);
}
float ccMaxOffset(uint v) { return asfloat(gAcc[ccVW(v, 2u)].load()); }
// The vehicle's first node's step (its reach's reference).
float3 ccRefDx(uint v) { uint g0 = gStatic[v * 4u].x; return ssub3(ccX1(g0), ccX0(v, g0)); }
// A car's own pairs' apart: the largest of the members near enough to touch it (their boxes
// within r_q and both paints).
float ccApartNear(uint off, uint k, uint a, float rq);
// Two vehicles' first nodes' relative step or speed times h (0 for one vehicle).
float ccApart(uint a, uint b)
{
    if (a == b) return 0.0;
    uint ga = gStatic[a * 4u].x;
    uint gb = gStatic[b * 4u].x;
    return max(ccLen(ssub3(ccRefDx(a), ccRefDx(b))), smul(ccLen(ssub3(ccVel(ga), ccVel(gb))), STEP));
}
void ccBox(uint v, out float3 lo, out float3 hi)
{
    lo = float3(vvUnkey(gAcc[ccVW(v, 3u)].load()), vvUnkey(gAcc[ccVW(v, 4u)].load()), vvUnkey(gAcc[ccVW(v, 5u)].load()));
    hi = float3(vvUnkey(gAcc[ccVW(v, 6u)].load()), vvUnkey(gAcc[ccVW(v, 7u)].load()), vvUnkey(gAcc[ccVW(v, 8u)].load()));
}
// The vehicle's reach (its surface nodes' largest step or speed times h), largest offset
// and surface box at the step's start, each a reduction in gAcc (positive floats as ints;
// the box as ordered keys): the CPU's folds, exact in any order.
void ccMeasure(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
    // (Each lane's own fold first, then one atomic a word: the same max and min.)
    int off = 0;
    int reach = 0;
    int3 lo = int3(VV_INT_MAX, VV_INT_MAX, VV_INT_MAX);
    int3 hi = int3(VV_INT_MIN, VV_INT_MIN, VV_INT_MIN);
    bool any = false;
    for (uint i = lane, iStop = EB_END(916, i, r0.y, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        off = max(off, asint(ccLen(contactOff(gi))));
        if (!ccOnSurface(v, gi)) continue;
        any = true;
        float3 x0 = ccX0(v, gi);
        reach = max(reach, asint(max(ccLen(ssub3(ssub3(ccX1(gi), x0), ccRefDx(v))), smul(ccLen(ssub3(ccVel(gi), ccVel(r0.x))), STEP))));
        lo = min(lo, int3(vvKey(x0.x), vvKey(x0.y), vvKey(x0.z)));
        hi = max(hi, int3(vvKey(x0.x), vvKey(x0.y), vvKey(x0.z)));
    }
    gAcc[ccVW(v, 2u)].max(off);
    if (!any) return;
    gAcc[ccVW(v, 1u)].max(reach);
    gAcc[ccVW(v, 3u)].min(lo.x);
    gAcc[ccVW(v, 4u)].min(lo.y);
    gAcc[ccVW(v, 5u)].min(lo.z);
    gAcc[ccVW(v, 6u)].max(hi.x);
    gAcc[ccVW(v, 7u)].max(hi.y);
    gAcc[ccVW(v, 8u)].max(hi.z);
}
// (Marked once a pass in the node's count word, ccMarkSurface; cleared after the listing.)
// (Word 8: the node's vehicle << 1, bit 0 its surface mark.)
bool ccOnSurface(uint v, uint gi) { return (gAcc[ccA(gi, 8u)].load() & 1) != 0; }
void ccMarkSurface(uint v, uint lane)
{
    uint4 r1 = gStatic[v * 4u + 1u];
    for (uint t = lane, tStop = EB_END(897, t, r1.y, LANES); t < tStop; t += LANES)
    {
        if (!ccTriLive(r1.x + t)) continue;
        uint4 tn = gStatic[gP.triBase + r1.x + t];
        gAcc[ccA(tn.x, 8u)].store(int(v << 1) | 1);
        gAcc[ccA(tn.y, 8u)].store(int(v << 1) | 1);
        gAcc[ccA(tn.z, 8u)].store(int(v << 1) | 1);
    }
}

// list_pairs for vehicles a (points, first edges) and b (triangles, second edges), rq the
// listing reach: each query's box grown by its threshold walks the cells of b's grid it
// touches, a candidate taken in the first cell both boxes share (once).
void ccList(uint a, uint b, float rq, float moA, float moB, uint lane, uint slice, uint slices)
{
    uint w0 = lane + LANES * slice;
    uint ws = LANES * slices;
    bool own = a == b;
    float r = own ? rq : nf(nf(rq + moA) + moB);
    float r2 = smul(r, r);
    float3 r3 = float3(r, r, r);
    uint4 ra = gStatic[a * 4u];
    uint g0 = ccVeh(b, 1u).x;
    // (b's surface box grown by the threshold: a query outside it lists nothing.)
    float3 blo;
    float3 bhi;
    ccBox(b, blo, bhi);
    blo = ssub3(blo, r3);
    bhi = sadd3(bhi, r3);
    // Points against triangles.
    // (A thread a point and a cell class: the cells congruent to its class modulo 4 on each
    // axis, so a query's cells spread over 64 threads, every cell still walked once.)
    for (uint t = w0, tStop = EB_END(894, t, ra.y * 64u, ws); t < tStop; t += ws)
    {
        uint i = t >> 6;
        int3 cls = int3(int(t & 3u), int((t >> 2) & 3u), int((t >> 4) & 3u));
        uint gi = ra.x + i;
        if (!ccOnSurface(a, gi)) continue;
        float3 p = ccX0(a, gi);
        if (any(p < blo) || any(p > bhi)) continue;
        // (The query grown by the reach and, between cars, the point's own offset.)
        float oq = own ? 0.0 : ccSlack(ccNodeOff(gi));
        float rp = nf(rq + oq);
        float3 q3 = float3(rp, rp, rp);
        int3 c0 = ccCell(ssub3(p, q3), CC_CELL);
        int3 c1 = ccCell(sadd3(p, q3), CC_CELL);
        for (int x = c0.x + cls.x; x <= c1.x; x += 4)
            for (int y = c0.y + cls.y; y <= c1.y; y += 4)
                for (int z = c0.z + cls.z; z <= c1.z; z += 4)
                {
                    int3 c = int3(x, y, z);
                    uint key = ccPack(c);
                    uint2 rg = ccBucket(b, ccHash(c, 0u));
                    for (uint e = rg.x; e < rg.y; e++)
                    {
                        uint ent = gNext[g0 + 2u * e];
                        if ((ent >> 31) != 0u || gNext[g0 + 2u * e + 1u] != key) continue;
                        uint gt = ent & 0x7FFFFFFFu;
                        float3 tlo;
                        float3 thi;
                        float ot;
                        ccPrimBoxOff(b, 0u, gt, tlo, thi, ot);
                        if (any(c != max(c0, ccCell(tlo, CC_CELL)))) continue;
                        if (any(tlo > p + q3) || any(thi < p - q3)) continue;
                        uint4 tn = gStatic[gP.triBase + gt];
                        if (own && (tn.x == gi || tn.y == gi || tn.z == gi)) continue;
                        float3 w;
                        float3 cp = fqClosestTri(p, ccX0(b, tn.x), ccX0(b, tn.y), ccX0(b, tn.z), w);
                        float3 dv = ssub3(p, cp);
                        // (Between cars the pair's own bound, under the vehicles': the same pairs pass
                        // the thickness test below.)
                        float rt = own ? rq : nf(rp + ot);
                        if (!(sdot(dv, dv) < smul(rt, rt))) continue;
                        CcPair q;
                        q.k[0] = tn.x; q.k[1] = tn.y; q.k[2] = tn.z; q.k[3] = gi;
                        q.nn = 3u; q.own = own; q.va = a; q.vb = b;
                        CcRec rec = ccRecOf(q);
                        if (rec.g < nf(rec.d + rq)) ccStore(a, q, rec);
                    }
                }
    }
    // Edges against edges, once a vehicle pair (a ≤ b; a car's own, the second edge after the
    // first, sharing no node).
    if (a > b) return;
    uint4 ea = ccVeh(a, 0u);
    for (uint t = w0, tStop = EB_END(895, t, ea.y * 64u, ws); t < tStop; t += ws)
    {
        uint i = t >> 6;
        int3 cls = int3(int(t & 3u), int((t >> 2) & 3u), int((t >> 4) & 3u));
        uint ge = ea.x + i;
        uint4 er = gStatic[ccRec(1).x + ge];
        if (!ccEdgeLive(ge)) continue;
        float3 p1 = ccX0(a, er.x);
        float3 q1 = ccX0(a, er.y);
        float oe = own ? 0.0 : ccSlack(max(ccNodeOff(er.x), ccNodeOff(er.y)));
        float re = nf(rq + oe);
        float3 e3 = float3(re, re, re);
        float3 elo = ssub3(min(p1, q1), e3);
        float3 ehi = sadd3(max(p1, q1), e3);
        if (any(ehi < blo) || any(elo > bhi)) continue;
        int3 c0 = ccCell(elo, CC_CELL);
        int3 c1 = ccCell(ehi, CC_CELL);
        for (int x = c0.x + cls.x; x <= c1.x; x += 4)
            for (int y = c0.y + cls.y; y <= c1.y; y += 4)
                for (int z = c0.z + cls.z; z <= c1.z; z += 4)
                {
                    int3 c = int3(x, y, z);
                    uint key = ccPack(c);
                    uint2 rg = ccBucket(b, ccHash(c, 1u));
                    for (uint e = rg.x; e < rg.y; e++)
                    {
                        uint ent = gNext[g0 + 2u * e];
                        if ((ent >> 31) == 0u || gNext[g0 + 2u * e + 1u] != key) continue;
                        uint gf = ent & 0x7FFFFFFFu;
                        if (own && gf <= ge) continue;
                        float3 flo;
                        float3 fhi;
                        float of;
                        ccPrimBoxOff(b, 1u, gf, flo, fhi, of);
                        if (any(c != max(c0, ccCell(flo, CC_CELL)))) continue;
                        if (any(flo > ehi) || any(fhi < elo)) continue;
                        uint4 fr = gStatic[ccRec(1).x + gf];
                        if (own && (fr.x == er.x || fr.x == er.y || fr.y == er.x || fr.y == er.y)) continue;
                        float3 p2 = ccX0(b, fr.x);
                        float3 q2 = ccX0(b, fr.y);
                        float2 st = fqSegSeg(p1, q1, p2, q2);
                        float3 dv = ssub3(sadd3(p2, sscale(ssub3(q2, p2), st.y)), sadd3(p1, sscale(ssub3(q1, p1), st.x)));
                        float rt = own ? rq : nf(re + of);
                        if (!(sdot(dv, dv) < smul(rt, rt))) continue;
                        CcPair q;
                        q.k[0] = er.x; q.k[1] = er.y; q.k[2] = fr.x; q.k[3] = fr.y;
                        q.nn = 2u; q.own = own; q.va = b; q.vb = a;
                        CcRec rec = ccRecOf(q);
                        if (rec.g < nf(rec.d + rq)) ccStore(a, q, rec);
                    }
                }
    }
}

float ccApartNear(uint off, uint k, uint a, float rq)
{
    float3 alo;
    float3 ahi;
    ccBox(a, alo, ahi);
    float m = 0.0;
    for (uint n = 0u; n < k; n++)
    {
        uint c = vvMember(off, n);
        if (c == a) continue;
        float r = nf(nf(rq + ccMaxOffset(a)) + ccMaxOffset(c));
        float3 clo;
        float3 chi;
        ccBox(c, clo, chi);
        float3 r3 = float3(r, r, r);
        if (any(ssub3(alo, r3) > chi) || any(ssub3(clo, r3) > ahi)) continue;
        m = max(m, ccApart(a, c));
    }
    return m;
}

// Per-node fixed sums (ccd.rs FixedSums): add a velocity change, a position step.
void ccAddV(uint gi, float3 d)
{
    gAcc[ccA(gi, 0u)].add(ccToFixedVel(d.x));
    gAcc[ccA(gi, 1u)].add(ccToFixedVel(d.y));
    gAcc[ccA(gi, 2u)].add(ccToFixedVel(d.z));
}
void ccAddX(uint gi, float3 d)
{
    gAcc[ccA(gi, 3u)].add(toFixedShift(d.x));
    gAcc[ccA(gi, 4u)].add(toFixedShift(d.y));
    gAcc[ccA(gi, 5u)].add(toFixedShift(d.z));
}
// Applies node gi's sums to its velocity (and position, `pos`) and clears them and its count.
void ccApply(uint gi, bool pos, float stamp)
{
    int3 a = int3(gAcc[ccA(gi, 0u)].load(), gAcc[ccA(gi, 1u)].load(), gAcc[ccA(gi, 2u)].load());
    if (any(a != int3(0, 0, 0))) gBeams[ccB(gi, 3u)].xyz = sadd3(ccVel(gi), sscale(float3(a), 1.0 / CC_VEL_FIXED));
    int3 b = int3(gAcc[ccA(gi, 3u)].load(), gAcc[ccA(gi, 4u)].load(), gAcc[ccA(gi, 5u)].load());
    if (pos && any(b != int3(0, 0, 0)))
    {
        gBeams[ccB(gi, 1u)].xyz = sadd3(ccX1(gi), sscale(float3(b), 1.0 / SHIFT_FIXED));
        gBeams[ccB(gi, 3u)].w = stamp;
    }
    for (uint w = 0u; w < 7u; w++) gAcc[ccA(gi, w)].store(0);
}


// The pass's prep (dispatch A, a workgroup a group): every member's scratch, surface marks,
// measures and grid.
void ccPrep(uint off, uint k, uint lane)
{
    float rq = asfloat(ccRec(1).z);
    // Each member's scratch: x1 (world) and its copy, the velocity and its copy; sums cleared.
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        float3 o = ccOrigin(v);
        for (uint i = lane, iStop = EB_END(896, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 x = nf3(o + gNodes[gi * 3u].xyz);
            float3 vel = gNodes[gi * 3u + 1u].xyz;
            gBeams[ccB(gi, 1u)] = float4(x, ccImOf(gi));
            gBeams[ccB(gi, 5u)] = float4(ccX0Snap(v, gi), ccLen(contactOff(gi)));
            gBeams[ccB(gi, 2u)] = float4(x, 0.0);
            gBeams[ccB(gi, 3u)] = float4(vel, 0.0);
            gBeams[ccB(gi, 4u)] = float4(vel, 0.0);
            for (uint w = 0u; w < CC_ACC_NODE; w++) gAcc[ccA(gi, w)].store(w == 7u ? -1 : (w == 8u ? int(v << 1) : 0));
        }
        if (lane < CC_VEH_WORDS) gAcc[ccVW(v, lane)].store(lane >= 3u && lane < 6u ? VV_INT_MAX : (lane >= 6u && lane < 9u ? VV_INT_MIN : 0));
    }
    sync();
    for (uint m = 0u; m < k; m++) ccMarkSurface(vvMember(off, m), lane);
    sync();
    for (uint m = 0u; m < k; m++) ccMeasure(vvMember(off, m), lane);
    sync();
    // The pairs: every member's grid, then every ordered member pair whose boxes come near,
    // each within the smaller of r_q and four times the two sides' reach (dat.rs list_pairs).
    float moAll = 0.0;
    float reachAll = 0.0;
    for (uint m = 0u; m < k; m++)
    {
        moAll = max(moAll, ccMaxOffset(vvMember(off, m)));
        reachAll = max(reachAll, ccReachV(vvMember(off, m)));
    }
    float apartAll = 0.0;
    for (uint m = 0u; m < k; m++)
        for (uint n = 0u; n < k; n++) apartAll = max(apartAll, ccApart(vvMember(off, m), vvMember(off, n)));
    for (uint m = 0u; m < k; m++) ccGridClear(vvMember(off, m), lane);
    sync();
}


// The pass's listing (dispatch B): workgroup g lists vehicle g / CC_SLICES's points and edges
// (its slice g % CC_SLICES of them) against every member of its contact group. No barriers:
// the grids and measures are the prep's, the pairs appended atomically.
static const uint CC_SLICES = 32u;
// Whether vehicle v's contact group runs the pass this step (awake, in a group).
bool ccVehicleOn(uint v, out uint off, out uint k)
{
    off = 0u;
    k = 0u;
    if (v >= gP.vehicles) return false;
    uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
    if (fl.y == 0u) return false;
    uint4 G = gMisc[gP.vvGroups + fl.z];
    k = splitAwake(G);
    off = G.x;
    return k != 0u;
}
// (Uniform: the prefix's barriers; every lane reads the same records.)
void ccGridPrefixWide(uint v, uint lane)
{
    uint off;
    uint k;
    bool on = ccVehicleOn(v, off, k);
    if (splitUniform(lane, on ? 1u : 0u) != 0u) ccGridPrefix(v, lane);
}
void ccGridWide(uint g, uint pass, uint lane)
{
    uint off;
    uint k;
    if (ccVehicleOn(g / CC_SLICES, off, k)) ccGridPass(g / CC_SLICES, pass, g % CC_SLICES, CC_SLICES, lane);
}
void ccListWide(uint g, uint lane)
{
    uint a = g / CC_SLICES;
    uint slice = g % CC_SLICES;
    uint off;
    uint k;
    if (!ccVehicleOn(a, off, k)) return;
    float rq = asfloat(ccRec(1).z);
    float moA = ccMaxOffset(a);
    float3 alo;
    float3 ahi;
    ccBox(a, alo, ahi);
    for (uint mb = 0u; mb < k; mb++)
    {
        uint b = vvMember(off, mb);
        float moB = ccMaxOffset(b);
        bool own = a == b;
        float rr = min(rq, smul(4.0, nf(nf(ccReachV(a) + ccReachV(b)) + (own ? ccApartNear(off, k, a, rq) : ccApart(a, b)))));
        float r = own ? rr : nf(nf(rr + moA) + moB);
        float3 blo;
        float3 bhi;
        ccBox(b, blo, bhi);
        float3 r3 = float3(r, r, r);
        if (any(ssub3(alo, r3) > bhi) || any(ssub3(blo, r3) > ahi)) continue;
        ccList(a, b, rr, moA, moB, lane, slice, CC_SLICES);
    }
}

// The pass's response (dispatch C, a workgroup a group).
void ccRespondPass(uint off, uint k, uint lane) { ccRespond(off, k, asfloat(ccRec(1).z), lane); }

#include "explicit_beams_split_cc_respond.slang"
