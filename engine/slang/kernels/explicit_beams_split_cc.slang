// --- Continuous contact's pass (docs/continuous-contact-design.md, "The GPU port"): between a
// step's node update and its rest, a workgroup a contact group. The CPU reference:
// crates/engine-softbody/src/explicit_contact/dat.rs (dat_truncate, list_pairs, speculative)
// and ccd.rs (resolve, rigid_zone), operation for operation in strict float; the per-node sums
// in fixed point (order free). Tables: engine_exec::explicit_plan::continuous::cc_tables. ---

#include "explicit_beams_cc_math.slang"

static const uint CC_BEAMS_NODE = 6u;
static const uint CC_ACC_NODE = 9u;
static const uint CC_VEH_WORDS = 12u;
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
static const uint CC_C_FLAG = 7u;

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
void ccGridBuild(uint v, uint lane)
{
    uint g0 = ccVeh(v, 1u).x;
    uint cap = ccVeh(v, 1u).y;
    uint ba = ccVeh(v, 2u).x;
    uint4 r1 = gStatic[v * 4u + 1u];
    uint tris = r1.y;
    uint edges = ccVeh(v, 0u).y;
    uint e0 = ccVeh(v, 0u).x;
    for (uint b = lane; b < 2u * CC_GRID_BUCKETS; b += LANES) gAcc[ba + b].store(0);
    sync();
    // Counts, then the starts (lane 0, in bucket order), then the entries.
    for (uint pass = 0u; pass < 2u; pass++)
    {
        for (uint i = lane, iStop = EB_END(893, i, tris + edges, LANES); i < iStop; i += LANES)
        {
            uint kind = i < tris ? 0u : 1u;
            uint id = kind == 0u ? r1.x + i : e0 + (i - tris);
            if (kind == 0u ? !ccTriLive(id) : !ccEdgeLive(id)) continue;
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
        sync();
        if (pass == 0u)
        {
            // (Two levels: each lane's run of buckets, then the runs' starts by lane 0.)
            uint per = CC_GRID_BUCKETS / LANES;
            int sum = 0;
            for (uint j = 0u; j < per; j++) sum += gAcc[ba + lane * per + j].load();
            gAcc[ba + CC_GRID_BUCKETS + lane * per].store(sum);
            sync();
            if (lane == 0u)
            {
                int run = 0;
                for (uint l = 0u; l < LANES; l++)
                {
                    int c = gAcc[ba + CC_GRID_BUCKETS + l * per].load();
                    gAcc[ba + CC_GRID_BUCKETS + l * per].store(run);
                    run += c;
                }
            }
            sync();
            int run = gAcc[ba + CC_GRID_BUCKETS + lane * per].load();
            for (uint j = 0u; j < per; j++)
            {
                uint b = lane * per + j;
                int c = gAcc[ba + b].load();
                gAcc[ba + b].store(run);
                gAcc[ba + CC_GRID_BUCKETS + b].store(run);
                run += c;
            }
            sync();
        }
    }
    // The buckets' ranges into plain memory (after the entries: the queries read them often).
    for (uint b = lane; b < CC_GRID_BUCKETS; b += LANES)
    {
        gNext[g0 + 2u * cap + 2u * b] = asuint(gAcc[ba + b].load());
        gNext[g0 + 2u * cap + 2u * b + 1u] = min(asuint(gAcc[ba + CC_GRID_BUCKETS + b].load()), cap);
    }
    sync();
}
// A bucket's entries: from its start to its cursor (where the fill left it).
uint2 ccBucket(uint v, uint h)
{
    uint at = ccVeh(v, 1u).x + 2u * ccVeh(v, 1u).y + 2u * h;
    return uint2(gNext[at], gNext[at + 1u]);
}
bool ccExcluded(uint v, uint k0, uint k1, uint k2, uint k3)
{
    uint4 r = ccVeh(v, 1u);
    for (uint e = 0u; e < r.w; e++)
    {
        uint at = r.z + 4u * e;
        if (gLists[at] == k0 && gLists[at + 1u] == k1 && gLists[at + 2u] == k2 && gLists[at + 3u] == k3) return true;
    }
    return false;
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
    for (uint i = lane, iStop = EB_END(916, i, r0.y, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        gAcc[ccVW(v, 2u)].max(asint(ccLen(contactOff(gi))));
        if (!ccOnSurface(v, gi)) continue;
        float3 x0 = ccX0(v, gi);
        gAcc[ccVW(v, 1u)].max(asint(max(ccLen(ssub3(ssub3(ccX1(gi), x0), ccRefDx(v))), smul(ccLen(ssub3(ccVel(gi), ccVel(r0.x))), STEP))));
        gAcc[ccVW(v, 3u)].min(vvKey(x0.x));
        gAcc[ccVW(v, 4u)].min(vvKey(x0.y));
        gAcc[ccVW(v, 5u)].min(vvKey(x0.z));
        gAcc[ccVW(v, 6u)].max(vvKey(x0.x));
        gAcc[ccVW(v, 7u)].max(vvKey(x0.y));
        gAcc[ccVW(v, 8u)].max(vvKey(x0.z));
    }
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
    for (uint i = w0, iStop = EB_END(894, i, ra.y, ws); i < iStop; i += ws)
    {
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
        for (int x = c0.x; x <= c1.x; x++)
            for (int y = c0.y; y <= c1.y; y++)
                for (int z = c0.z; z <= c1.z; z++)
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
                        if (own && ccExcluded(a, gi, tn.x, tn.y, tn.z)) continue;
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
    for (uint i = w0, iStop = EB_END(895, i, ea.y, ws); i < iStop; i += ws)
    {
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
        for (int x = c0.x; x <= c1.x; x++)
            for (int y = c0.y; y <= c1.y; y++)
                for (int z = c0.z; z <= c1.z; z++)
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
                        if (own && ccExcluded(a, er.x, er.y, fr.x, fr.y)) continue;
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
void ccApply(uint gi, bool pos)
{
    int3 a = int3(gAcc[ccA(gi, 0u)].load(), gAcc[ccA(gi, 1u)].load(), gAcc[ccA(gi, 2u)].load());
    if (any(a != int3(0, 0, 0))) gBeams[ccB(gi, 3u)].xyz = sadd3(ccVel(gi), sscale(float3(a), 1.0 / CC_VEL_FIXED));
    int3 b = int3(gAcc[ccA(gi, 3u)].load(), gAcc[ccA(gi, 4u)].load(), gAcc[ccA(gi, 5u)].load());
    if (pos && any(b != int3(0, 0, 0))) gBeams[ccB(gi, 1u)].xyz = sadd3(ccX1(gi), sscale(float3(b), 1.0 / SHIFT_FIXED));
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
    for (uint m = 0u; m < k; m++) ccGridBuild(vvMember(off, m), lane);
    sync();
}


// --- The response (dat.rs dat_truncate after the listing; ccd.rs resolve) ---

static const uint CC_F_NEAR = 0x200u;
static const uint CC_F_CCD = 0x400u;
static const uint CC_F_ACT = 0x800u;
static const uint CC_F_ZONED = 0x1000u;
uint ccFlags(uint v, uint p) { return gNext[ccPairAt(v, p) + 4u]; }
void ccFlagSet(uint v, uint p, uint f) { gNext[ccPairAt(v, p) + 4u] = ccFlags(v, p) | f; }
void ccFlagClear(uint v, uint p, uint f) { gNext[ccPairAt(v, p) + 4u] = ccFlags(v, p) & ~f; }
uint ccPairs(uint v) { return min(asuint(gAcc[ccCount(v)].load()), ccVeh(v, 0u).w); }
// The compact lists of the speculative pass's pairs (list 0) and the CCD's (list 1): pair
// indices after the vehicle's pairs, their counts in its words 9 and 10.
uint ccListAt(uint v, uint which) { return ccVeh(v, 0u).z + CC_PAIR_WORDS * ccVeh(v, 0u).w + which * ccVeh(v, 0u).w; }
uint ccListN(uint v, uint which) { return min(asuint(gAcc[ccVW(v, 9u + which)].load()), ccVeh(v, 0u).w); }
uint ccListP(uint v, uint which, uint i) { return gNext[ccListAt(v, which) + i]; }
void ccListAdd(uint v, uint which, uint p) { gNext[ccListAt(v, which) + asuint(gAcc[ccVW(v, 9u + which)].add(1))] = p; }

// How far node gi can close this step (its step, or its speed times h: reach).
float ccReach(uint gi, uint v, bool speed)
{
    float d = ccLen(ssub3(ccX1(gi), ccX0(v, gi)));
    return speed ? max(d, smul(ccLen(ccVel(gi)), STEP)) : max(d, 0.0);
}
// A side's largest reach (side: the first nn nodes when first).
float ccSide(CcPair q, bool first, bool speed)
{
    float m = 0.0;
    for (uint j = 0u; j < 4u; j++)
        if ((j < q.nn) == first) m = max(m, ccReach(q.k[j], j < q.nn ? q.vb : q.va, speed));
    return m;
}

// The pair's closest points at path time t (ccd.rs closest): signed weights and gap vector.
float3 ccAt(CcPair q, float t, out float4 w, out float3 dp[4])
{
    float3 p[4];
    for (uint j = 0u; j < 4u; j++)
    {
        float3 a = ccX0(j < q.nn ? q.vb : q.va, q.k[j]);
        dp[j] = ssub3(ccX1(q.k[j]), a);
        p[j] = sadd3(a, sscale(dp[j], t));
    }
    return ccClosest(q.nn, p, w);
}
// collide: the time of impact of pair q over its paths (−1: none, or not closing there).
float ccCollide(CcPair q, float xi)
{
    float3 a[4];
    float3 dp[4];
    for (uint j = 0u; j < 4u; j++)
    {
        a[j] = ccX0(j < q.nn ? q.vb : q.va, q.k[j]);
        dp[j] = ssub3(ccX1(q.k[j]), a[j]);
    }
    float t = ccAccd(q.nn, a, dp, xi);
    if (t < 0.0) return -1.0;
    float4 w;
    float3 d2[4];
    float3 n = contactNormOrZero(ccAt(q, t, w, d2));
    float c = 0.0;
    for (uint j = 0u; j < 4u; j++) c = nf(c + smul(w[j], sdot(dp[j], n)));
    return c < 0.0 ? t : -1.0;
}
float ccXi(CcRec r) { return max(smul(0.5, r.d), CC_MIN_GAP); }

// Applies every group node's sums (velocity; position too with `pos`) and clears them.
// The nodes a round's pairs touched (their count went from 0 to 1), a list a vehicle after
// its two pair lists (its word 11 the count): only those have sums to apply.
uint ccTouchedAt(uint v) { return ccListAt(v, 2u); }
void ccCountNode(uint gi)
{
    if (gAcc[ccA(gi, 6u)].add(1) == 0)
    {
        uint v = asuint(gAcc[ccA(gi, 8u)].load()) >> 1;
        gNext[ccTouchedAt(v) + asuint(gAcc[ccVW(v, 11u)].add(1))] = gi;
    }
}
void ccApplyAll(uint off, uint k, bool pos, uint lane)
{
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint n = asuint(gAcc[ccVW(v, 11u)].load());
        for (uint i = lane, iStop = EB_END(899, i, n, LANES); i < iStop; i += LANES) ccApply(gNext[ccTouchedAt(v) + i], pos);
    }
    sync();
    if (lane < k) gAcc[ccVW(vvMember(off, lane), 11u)].store(0);
    sync();
}

// The zone label's root: the smallest node of the component (label propagation converged).
int ccLabel(uint gi) { return gAcc[ccA(gi, 7u)].load(); }
// (The converged labels copied into the scratch's slot 2 .w for the zones' sums.)
int ccZoneOf(uint gi) { return asint(gBeams[ccB(gi, 2u)].w); }

// One impact zone, rooted at node `root`, moved rigidly (ccd.rs rigid_zone): its nodes in
// ascending order over the group's; false when it would turn more than 1 rad.
bool ccRigidZone(uint off, uint k, int root)
{
    bool pinned = false;
    float mass = 0.0;
    float3 cm = float3(0.0, 0.0, 0.0);
    float3 p = float3(0.0, 0.0, 0.0);
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        for (uint i = 0u; i < r0.y; i++)
        {
            uint gi = r0.x + i;
            if (ccZoneOf(gi) != root) continue;
            if (ccIm(gi) == 0.0) pinned = true;
            float ms = gNodes[gi * 3u].w;
            mass = nf(mass + ms);
            cm = sadd3(cm, sscale(ccX0(v, gi), ms));
            p = sadd3(p, sscale(ccVel(gi), ms));
        }
    }
    if (pinned)
    {
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            uint4 r0 = gStatic[v * 4u];
            for (uint i = 0u; i < r0.y; i++)
            {
                uint gi = r0.x + i;
                if (ccZoneOf(gi) != root) continue;
                gBeams[ccB(gi, 3u)].xyz = float3(0.0, 0.0, 0.0);
                gBeams[ccB(gi, 1u)].xyz = ccX0(v, gi);
            }
        }
        return true;
    }
    float3 c = float3(sdiv(cm.x, mass), sdiv(cm.y, mass), sdiv(cm.z, mass));
    float3 vc = float3(sdiv(p.x, mass), sdiv(p.y, mass), sdiv(p.z, mass));
    float3 l = float3(0.0, 0.0, 0.0);
    float3 I[3] = { float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0) };
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        for (uint i = 0u; i < r0.y; i++)
        {
            uint gi = r0.x + i;
            if (ccZoneOf(gi) != root) continue;
            float3 r = ssub3(ccX0(v, gi), c);
            float mm = gNodes[gi * 3u].w;
            l = sadd3(l, sscale(scross(r, ssub3(ccVel(gi), vc)), mm));
            float rr = sdot(r, r);
            for (uint a = 0u; a < 3u; a++)
            {
                float3 row;
                for (uint b = 0u; b < 3u; b++) row[b] = smul(mm, nf((a == b ? rr : 0.0) - smul(r[a], r[b])));
                I[a] = sadd3(I[a], row);
            }
        }
    }
    float3 n[3];
    float3 w = float3(0.0, 0.0, 0.0);
    if (ccInverse3(I, n))
        w = float3(nf(nf(smul(n[0].x, l.x) + smul(n[0].y, l.y)) + smul(n[0].z, l.z)),
                   nf(nf(smul(n[1].x, l.x) + smul(n[1].y, l.y)) + smul(n[1].z, l.z)),
                   nf(nf(smul(n[2].x, l.x) + smul(n[2].y, l.y)) + smul(n[2].z, l.z)));
    float wl = ccLen(w);
    float ang = smul(wl, STEP);
    if (ang > 1.0) return false;
    float2 sc = ccSinCos(ang);
    float3 axis = wl > 0.0 ? float3(sdiv(w.x, wl), sdiv(w.y, wl), sdiv(w.z, wl)) : float3(0.0, 0.0, 0.0);
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        for (uint i = 0u; i < r0.y; i++)
        {
            uint gi = r0.x + i;
            if (ccZoneOf(gi) != root) continue;
            float3 r = ssub3(ccX0(v, gi), c);
            float3 rot = sadd3(sadd3(sscale(r, sc.y), sscale(scross(axis, r), sc.x)), sscale(axis, smul(sdot(axis, r), nf(1.0 - sc.y))));
            gBeams[ccB(gi, 1u)].xyz = sadd3(sadd3(c, sscale(vc, STEP)), rot);
            gBeams[ccB(gi, 3u)].xyz = sadd3(vc, scross(w, r));
        }
    }
    return true;
}

// A pair whose four nodes are all in one zone moves rigidly: not tested again.
bool ccAcross(CcPair q)
{
    int l0 = ccLabel(q.k[0]);
    return !(l0 >= 0 && ccLabel(q.k[1]) == l0 && ccLabel(q.k[2]) == l0 && ccLabel(q.k[3]) == l0);
}

void ccRespond(uint off, uint k, float rq, uint lane)
{
    // The candidates: the speculative pass's (their gap past the thickness within the sides'
    // reach), with speeds.
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        for (uint p = lane, pStop = EB_END(900, p, ccPairs(v), LANES); p < pStop; p += LANES)
        {
            CcPair q = ccLoad(v, p);
            CcRec r = ccRecAt(v, p);
            if (nf(r.g - r.d) < nf(ccSide(q, true, true) + ccSide(q, false, true))) ccListAdd(v, 0u, p);
        }
    }
    sync();
    // The speculative inelastic contact (dat.rs speculative), CC_SPEC_ROUNDS rounds.
    for (uint round = 0u, rStop = EB_END_UNIFORM(901, round, CC_SPEC_ROUNDS, 1u, CC_SPEC_ROUNDS); round < rStop; round++)
    {
        if (lane == 0u) gAcc[ccC(CC_C_FLAG)].store(0);
        sync();
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(902, i, ccListN(v, 0u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 0u, i);
                CcPair q = ccLoad(v, p);
                CcRec r = ccRecAt(v, p);
                float rel = 0.0;
                for (uint j = 0u; j < 4u; j++) rel = nf(rel + smul(smul(ccSign(q.nn, j), r.w[j]), sdot(ccVel(q.k[j]), r.n)));
                float c = nf(rel + sdiv(max(nf(r.g - r.d), 0.0), STEP));
                if (c < 0.0)
                {
                    ccFlagSet(v, p, CC_F_ACT);
                    gAcc[ccC(CC_C_FLAG)].store(1);
                    for (uint j = 0u; j < 4u; j++)
                        if (r.w[j] > 0.0 && ccIm(q.k[j]) > 0.0) ccCountNode(q.k[j]);
                }
                else ccFlagClear(v, p, CC_F_ACT);
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccC(CC_C_FLAG)].load())) == 0u) break;
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(903, i, ccListN(v, 0u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 0u, i);
                if ((ccFlags(v, p) & CC_F_ACT) == 0u) continue;
                CcPair q = ccLoad(v, p);
                CcRec r = ccRecAt(v, p);
                float rel = 0.0;
                for (uint j = 0u; j < 4u; j++) rel = nf(rel + smul(smul(ccSign(q.nn, j), r.w[j]), sdot(ccVel(q.k[j]), r.n)));
                float c = nf(rel + sdiv(max(nf(r.g - r.d), 0.0), STEP));
                float wsum = 0.0;
                for (uint j = 0u; j < 4u; j++)
                    wsum = nf(wsum + smul(smul(smul(r.w[j], r.w[j]), ccIm(q.k[j])), float(gAcc[ccA(q.k[j], 6u)].load())));
                if (wsum <= 0.0) continue;
                float l = sdiv(-c, wsum);
                for (uint j = 0u; j < 4u; j++)
                {
                    float mi = ccIm(q.k[j]);
                    if (mi > 0.0 && r.w[j] > 0.0) ccAddV(q.k[j], sscale(r.n, smul(smul(smul(ccSign(q.nn, j), r.w[j]), mi), l)));
                }
            }
        }
        sync();
        ccApplyAll(off, k, false, lane);
    }
    // x1 += (v1 − v0) h, then the CCD's velocities: v0 + (x1 − x1 before) / h.
    for (uint m = 0u; m < k; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(904, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 v0 = gBeams[ccB(gi, 4u)].xyz;
            float3 x1 = sadd3(ccX1(gi), sscale(ssub3(ccVel(gi), v0), STEP));
            gBeams[ccB(gi, 1u)].xyz = x1;
            float3 d = ssub3(ssub3(x1, gBeams[ccB(gi, 2u)].xyz), float3(0.0, 0.0, 0.0));
            gBeams[ccB(gi, 3u)].xyz = sadd3(v0, float3(sdiv(d.x, STEP), sdiv(d.y, STEP), sdiv(d.z, STEP)));
        }
    }
    sync();
    // (The listing's bound: past half its own pairs' reach from its first node, an error counted.)
    for (uint m = 0u; m < k; m++)
    {
        uint vv = vvMember(off, m);
        uint4 r0 = gStatic[vv * 4u];
        for (uint i = lane, iStop = EB_END(917, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float own = min(rq, smul(4.0, nf(nf(ccReachV(vv) + ccReachV(vv)) + ccApartNear(off, k, vv, rq))));
            if (ccOnSurface(vv, gi) && ccLen(ssub3(ssub3(ccX1(gi), ccX0(vv, gi)), ccRefDx(vv))) > smul(0.5, own))
                gAcc[ccC(CC_C_REACH)].add(1);
        }
    }
    sync();
    // The CCD's pairs: their gap past the thickness within twice the sides' steps.
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        for (uint p = lane, pStop = EB_END(905, p, ccPairs(v), LANES); p < pStop; p += LANES)
        {
            CcPair q = ccLoad(v, p);
            CcRec r = ccRecAt(v, p);
            if (nf(r.g - r.d) < smul(2.0, nf(ccSide(q, true, false) + ccSide(q, false, false)))) ccListAdd(v, 1u, p);
        }
    }
    sync();
    // The impulse rounds (ccd.rs resolve).
    bool clear = false;
    for (uint round = 0u, rStop = EB_END_UNIFORM(906, round, CC_CCD_ROUNDS, 1u, CC_CCD_ROUNDS); round < rStop; round++)
    {
        if (lane == 0u) gAcc[ccC(CC_C_FLAG)].store(0);
        sync();
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(907, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                ccFlagClear(v, p, CC_F_ACT);
                CcPair q = ccLoad(v, p);
                float t = ccCollide(q, ccXi(ccRecAt(v, p)));
                gNext[ccPairAt(v, p) + 15u] = asuint(t);
                if (t < 0.0) continue;
                gAcc[ccC(CC_C_HITS)].add(1);
                gAcc[ccC(CC_C_FLAG)].store(1);
                float4 w;
                float3 dp[4];
                float3 n = contactNormOrZero(ccAt(q, t, w, dp));
                float vr = 0.0;
                for (uint j = 0u; j < 4u; j++) vr = nf(vr + smul(w[j], sdot(dp[j], n)));
                vr = sdiv(vr, STEP);
                if (all(n == float3(0.0, 0.0, 0.0)) || !(vr < 0.0)) continue;
                ccFlagSet(v, p, CC_F_ACT);
                for (uint j = 0u; j < 4u; j++)
                    if (w[j] != 0.0 && ccIm(q.k[j]) > 0.0) ccCountNode(q.k[j]);
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccC(CC_C_FLAG)].load())) == 0u) { clear = true; break; }
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(908, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                if ((ccFlags(v, p) & CC_F_ACT) == 0u) continue;
                CcPair q = ccLoad(v, p);
                float t = asfloat(gNext[ccPairAt(v, p) + 15u]);
                float4 w;
                float3 dp[4];
                float3 n = contactNormOrZero(ccAt(q, t, w, dp));
                float vr = 0.0;
                float vv = 0.0;
                for (uint j = 0u; j < 4u; j++)
                {
                    vr = nf(vr + smul(w[j], sdot(dp[j], n)));
                    vv = nf(vv + smul(w[j], sdot(ccVel(q.k[j]), n)));
                }
                vr = sdiv(vr, STEP);
                float wsum = 0.0;
                for (uint j = 0u; j < 4u; j++)
                    wsum = nf(wsum + smul(smul(smul(w[j], w[j]), ccIm(q.k[j])), float(gAcc[ccA(q.k[j], 6u)].load())));
                if (wsum <= 0.0) continue;
                float l = sdiv(-vr, wsum);
                float lv = min(max(sdiv(-vv, wsum), 0.0), l);
                for (uint j = 0u; j < 4u; j++)
                {
                    float mi = ccIm(q.k[j]);
                    if (mi > 0.0 && w[j] != 0.0)
                    {
                        ccAddX(q.k[j], sscale(n, smul(smul(smul(w[j], mi), l), STEP)));
                        ccAddV(q.k[j], sscale(n, smul(smul(w[j], mi), lv)));
                    }
                }
            }
        }
        sync();
        ccApplyAll(off, k, true, lane);
    }
    // Impact zones, while pairs still collide (ccd.rs resolve's zone rounds).
    for (uint round = 0u, rStop = EB_END_UNIFORM(909, round, clear ? 0u : CC_ZONE_ROUNDS, 1u, CC_ZONE_ROUNDS); round < rStop; round++)
    {
        if (lane == 0u) gAcc[ccC(CC_C_FLAG)].store(0);
        sync();
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(910, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                CcPair q = ccLoad(v, p);
                if (!ccAcross(q) || ccCollide(q, ccXi(ccRecAt(v, p))) < 0.0) continue;
                ccFlagSet(v, p, CC_F_ZONED);
                gAcc[ccC(CC_C_FLAG)].store(1);
                for (uint j = 0u; j < 4u; j++)
                    if (ccLabel(q.k[j]) < 0) gAcc[ccA(q.k[j], 7u)].store(int(q.k[j]));
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccC(CC_C_FLAG)].load())) == 0u) { clear = true; break; }
        // Label propagation over every zoned pair: each node the smallest label of its zone.
        for (uint it = 0u, itStop = EB_END_UNIFORM(911, it, 4096u, 1u, 4096u); it < itStop; it++)
        {
            if (lane == 0u) gAcc[ccC(CC_C_FLAG)].store(0);
            sync();
            for (uint m = 0u; m < k; m++)
            {
                uint v = vvMember(off, m);
                for (uint i = lane, iStop = EB_END(912, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
                {
                    uint p = ccListP(v, 1u, i);
                    if ((ccFlags(v, p) & CC_F_ZONED) == 0u) continue;
                    CcPair q = ccLoad(v, p);
                    int lo = min(min(ccLabel(q.k[0]), ccLabel(q.k[1])), min(ccLabel(q.k[2]), ccLabel(q.k[3])));
                    for (uint j = 0u; j < 4u; j++)
                        if (gAcc[ccA(q.k[j], 7u)].min(lo) > lo) gAcc[ccC(CC_C_FLAG)].store(1);
                }
            }
            sync();
            if (splitUniform(lane, uint(gAcc[ccC(CC_C_FLAG)].load())) == 0u) break;
        }
        for (uint m = 0u; m < k; m++)
        {
            uint4 r0 = gStatic[vvMember(off, m) * 4u];
            for (uint i = lane, iStop = EB_END(918, i, r0.y, LANES); i < iStop; i += LANES)
                gBeams[ccB(r0.x + i, 2u)].w = asfloat(ccLabel(r0.x + i));
        }
        sync();
        // Each zone (rooted at its smallest node) moved rigidly, a lane a zone.
        for (uint m = 0u; m < k; m++)
        {
            uint4 r0 = gStatic[vvMember(off, m) * 4u];
            for (uint i = lane, iStop = EB_END(913, i, r0.y, LANES); i < iStop; i += LANES)
            {
                uint gi = r0.x + i;
                if (ccZoneOf(gi) != int(gi)) continue;
                gAcc[ccC(CC_C_ZONES)].add(1);
                if (!ccRigidZone(off, k, int(gi))) gAcc[ccC(CC_C_SPIN)].add(1);
            }
        }
        sync();
    }
    if (!clear)
    {
        for (uint m = 0u; m < k; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(914, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                CcPair q = ccLoad(v, p);
                if (ccAcross(q) && ccCollide(q, ccXi(ccRecAt(v, p))) >= 0.0) gAcc[ccC(CC_C_UNRESOLVED)].add(1);
            }
        }
        sync();
    }
    // The nodes' new positions and velocities (relative to the origin), where they changed.
    for (uint m = 0u; m < k; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        float3 o = ccOrigin(v);
        for (uint i = lane, iStop = EB_END(915, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 x1 = ccX1(gi);
            float3 vs = ccVel(gi);
            if (any(x1 != gBeams[ccB(gi, 2u)].xyz) || any(vs != gBeams[ccB(gi, 4u)].xyz))
            {
                gNodes[gi * 3u + 1u].xyz = vs;
                gNodes[gi * 3u].xyz = ssub3(x1, o);
            }
        }
    }
    sync();
}

// The pass's listing (dispatch B): workgroup g lists vehicle g / CC_SLICES's points and edges
// (its slice g % CC_SLICES of them) against every member of its contact group. No barriers:
// the grids and measures are the prep's, the pairs appended atomically.
static const uint CC_SLICES = 32u;
void ccListWide(uint g, uint lane)
{
    uint a = g / CC_SLICES;
    uint slice = g % CC_SLICES;
    if (a >= gP.vehicles) return;
    uint4 fl = gMisc[vvRec(a, VV_FLAGS)];
    if (fl.y == 0u) return;
    uint4 G = gMisc[gP.vvGroups + fl.z];
    uint k = splitAwake(G);
    if (k == 0u) return;
    uint off = G.x;
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
