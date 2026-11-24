// Edge vehicle contact (ExplicitBeams::edge_contact, a Rubble setting, off by default):
// the CPU reference is engine-softbody's explicit_contact/edge.rs, and the design and
// its measurements docs/contact-design.md. With either vehicle of a pair carrying it, in
// the owner's turn and its staging:
// * edge against edge: the two skins' live edges (each pair once, in the lower vehicle's
//   turn), closest points strictly inside both, normal e × f out of the owner's skin;
// * node against edge: the partner's contact points against the owner's outer convex
//   edges, each in its edge's wedge;
// * the residual pass's accumulated clamp for every hit of such a pair (a hit's normal
//   force moves toward the stop at the forces now and never below zero).
// Every edge-contact pair is two-body and dissipative (the host refuses another upload),
// so an edge hit reads only the turn's snapshot and its split counts, never another
// hit's force, and adds in fixed point: it applies anywhere in its owner's level before
// the residual phases, in any order, bit for bit as the CPU's list.
//
// Layout (explicit_plan.rs W_EDGE): gStatic at gP.edgeBase, per vehicle two records
// (first edge record, edges, first scratch float4, first hit word) and (EDGE_ACC gAcc
// words: hits, listed edges, listed points, the edges' box as six keys; hit room; first
// triangle; listed edges in gNext, then listed points); per edge (global i, global j,
// first triangle entry in gLists, triangles); per entry the vehicle's own triangle index,
// bit 31 on the outer skin. A hit is two words: the owner's edge index with bit 31 for
// node against edge, then the partner's (vehicle << 20 | edge index) or its global node.

static const uint FLAG_EDGE = 2097152u; // vehicle flags bit 21 (explicit_plan.rs)
static const float EDGE_END = 1e-3;
static const float EDGE_DEPTH_M = 0.1; // CLIP_DEPTH (explicit_contact/edge.rs EDGE_DEPTH)
static const int FL_ERR_EDGES = 5;
static const uint EDGE_SCR = 5u;
static const uint EDGE_BUCKETS = 2048u; // explicit_plan.rs EDGE_BUCKETS
static const uint EDGE_BLOCK = 32u; // buckets a block of the starts' scan
static const uint EDGE_CAND = 32u; // candidate pairs a listed item (explicit_plan.rs EDGE_CAND)
static const float EDGE_SKIN = 0.05; // the candidate list's margin (m): rebuilt past it
// The candidate list's mode (a thread's): edgeTry keeps the pairs within reach + 2 skins
// instead of testing them.
static uint gEdgeCollect = 0u;
static float gEdgeCollectReach = 0.0;

bool edgeOn() { return gP.edgeBase != 0u; }
// Measurement only (`RUBBLE_EDGE_ABLATE`, time; results meaningless): 1 the pair tests, 2 the
// grids, 4 the edges' step data, 8 the hits (found, not kept) skipped.
uint edgeAblate() { return gStatic[gP.edgeBase - 1u].x; }
uint4 edgeV0(uint v) { return gStatic[gP.edgeBase + 2u * v]; }
uint4 edgeV1(uint v) { return gStatic[gP.edgeBase + 2u * v + 1u]; }
bool edgeVeh(uint v) { return (gStatic[v * 4u + 3u].x & FLAG_EDGE) != 0u; }
bool edgePairOn(uint a, uint b) { return edgeOn() && (edgeVeh(a) || edgeVeh(b)); }
float edgeRange(uint a, uint b) { return max(asfloat(gStatic[a * 4u + 3u].z), asfloat(gStatic[b * 4u + 3u].z)); }
// A node's world position as the CPU reference takes it (origin + position).
float3 edgeWorld(uint v, uint gi) { return nf3(asfloat(gMisc[v].xyz) + gNodes[gi * 3u].xyz); }

// glam's normalize_or_zero: v · (1 / |v|) when that reciprocal is finite and positive.
float3 edgeNormOrZero(float3 x)
{
    float rcp = sdiv(1.0, ssqrt(sdot(x, x)));
    if (!(rcp > 0.0 && rcp <= 3.4028235e38)) return float3(0.0, 0.0, 0.0);
    return sscale(x, rcp);
}

float edgeClamp01(float x) { return min(max(x, 0.0), 1.0); }
// engine_schema::feature_query::closest_seg_seg (feature_query.slang fqSegSeg).
float2 edgeSegSeg(float3 p1, float3 q1, float3 p2, float3 q2)
{
    float3 d1 = ssub3(q1, p1);
    float3 d2 = ssub3(q2, p2);
    float3 r = ssub3(p1, p2);
    float a = sdot(d1, d1);
    float e = sdot(d2, d2);
    float f = sdot(d2, r);
    if (a <= 1e-12 && e <= 1e-12) return float2(0.0, 0.0);
    if (a <= 1e-12) return float2(0.0, edgeClamp01(sdiv(f, e)));
    float c = sdot(d1, r);
    if (e <= 1e-12) return float2(edgeClamp01(sdiv(-c, a)), 0.0);
    float b = sdot(d1, d2);
    float denom = nf(smul(a, e) - smul(b, b));
    float s = denom > 1e-12 ? edgeClamp01(sdiv(nf(smul(b, f) - smul(c, e)), denom)) : 0.0;
    float t = sdiv(nf(smul(b, s) + f), e);
    if (t < 0.0)
    {
        t = 0.0;
        s = edgeClamp01(sdiv(-c, a));
    }
    else if (t > 1.0)
    {
        t = 1.0;
        s = edgeClamp01(sdiv(nf(b - c), a));
    }
    return float2(s, t);
}

// A skin edge this step (edge.rs SkinEdge): ends, outward normal, its faces' opposite
// corners, flags (1 live, 2 outer, 4 convex, 8 / 16 first / second corner).
struct SkinE
{
    uint i;
    uint j;
    float3 p;
    float3 q;
    float3 o;
    float3 c0;
    float3 c1;
    uint flags;
};
SkinE edgeLoad(uint v, uint k)
{
    uint at = edgeV0(v).z + EDGE_SCR * k;
    uint4 rec = gStatic[edgeV0(v).x + k];
    SkinE e;
    e.i = rec.x;
    e.j = rec.y;
    float4 s0 = gBeams[at];
    e.p = s0.xyz;
    e.flags = asuint(s0.w);
    e.q = gBeams[at + 1u].xyz;
    e.o = gBeams[at + 2u].xyz;
    e.c0 = gBeams[at + 3u].xyz;
    e.c1 = gBeams[at + 4u].xyz;
    return e;
}

// Vehicle v's edge k this step (edge.rs skin_edges): the live triangles' unit normals
// summed in the triangles' order, the outer skin's when the edge has any; written to
// its scratch, its ends into the vehicle's box when live.
void edgePrepare(uint v, uint k)
{
    if ((edgeAblate() & 4u) != 0u) return;
    uint4 v0 = edgeV0(v);
    uint4 v1 = edgeV1(v);
    uint4 rec = gStatic[v0.x + k];
    float3 wi = edgeWorld(v, rec.x);
    float3 wj = edgeWorld(v, rec.y);
    float3 sum = float3(0.0, 0.0, 0.0);
    bool outer = false;
    uint faces = 0u;
    float3 c0 = float3(0.0, 0.0, 0.0);
    float3 c1 = float3(0.0, 0.0, 0.0);
    float3 n0 = float3(0.0, 0.0, 0.0);
    float3 n1 = float3(0.0, 0.0, 0.0);
    for (uint q = 0u, qStop = EB_END(800, q, rec.w, 1u); q < qStop; q++)
    {
        uint word = gLists[rec.z + q];
        bool to = (word >> 31u) != 0u;
        uint gt = v1.z + (word & 0x7FFFFFFFu);
        if ((gMisc[gP.vehicles + gt].z & TRI_BREAKABLE) != 0u && triangleDead(gt)) continue;
        uint4 tn = gStatic[gP.triBase + gt];
        if (gP.retireBase != 0u && (nodeRetired(tn.x) || nodeRetired(tn.y) || nodeRetired(tn.z))) continue;
        if (to && !outer)
        {
            sum = float3(0.0, 0.0, 0.0);
            faces = 0u;
            outer = true;
        }
        if (to != outer) continue;
        float3 a = edgeWorld(v, tn.x);
        float3 b = edgeWorld(v, tn.y);
        float3 c = edgeWorld(v, tn.z);
        float3 nn = edgeNormOrZero(scross(ssub3(b, a), ssub3(c, a)));
        sum = sadd3(sum, nn);
        // The corner opposite the edge.
        float3 opp = (tn.x != rec.x && tn.x != rec.y) ? a : ((tn.y != rec.x && tn.y != rec.y) ? b : c);
        if (faces == 0u)
        {
            c0 = opp;
            n0 = nn;
        }
        else if (faces == 1u)
        {
            c1 = opp;
            n1 = nn;
        }
        faces++;
    }
    float3 o = edgeNormOrZero(sum);
    bool live = any(o != float3(0.0, 0.0, 0.0));
    bool convex = faces != 2u || (sdot(n0, ssub3(c1, wi)) <= 0.0 && sdot(n1, ssub3(c0, wi)) <= 0.0);
    uint flags = (live ? 1u : 0u) | (outer ? 2u : 0u) | (convex ? 4u : 0u) | (faces >= 1u ? 8u : 0u) | (faces >= 2u ? 16u : 0u);
    uint at = v0.z + EDGE_SCR * k;
    gBeams[at] = float4(wi, asfloat(flags));
    gBeams[at + 1u] = float4(wj, 0.0);
    gBeams[at + 2u] = float4(o, 0.0);
    gBeams[at + 3u] = float4(c0, 0.0);
    gBeams[at + 4u] = float4(c1, 0.0);
    if (live)
    {
        float3 lo = min(wi, wj);
        float3 hi = max(wi, wj);
        uint bx = v1.x + 3u;
        gAcc[bx].min(vvKey(lo.x));
        gAcc[bx + 1u].min(vvKey(lo.y));
        gAcc[bx + 2u].min(vvKey(lo.z));
        gAcc[bx + 3u].max(vvKey(hi.x));
        gAcc[bx + 4u].max(vvKey(hi.y));
        gAcc[bx + 5u].max(vvKey(hi.z));
    }
}

void edgeBox(uint v, out float3 lo, out float3 hi)
{
    uint bx = edgeV1(v).x + 3u;
    lo = float3(vvUnkey(gAcc[bx].load()), vvUnkey(gAcc[bx + 1u].load()), vvUnkey(gAcc[bx + 2u].load()));
    hi = float3(vvUnkey(gAcc[bx + 3u].load()), vvUnkey(gAcc[bx + 4u].load()), vvUnkey(gAcc[bx + 5u].load()));
}

// --- A hit's geometry (edge.rs edge_hits, vertex_edge_hits), found and re-derived ---

struct EdgeHitG
{
    uint va;
    uint vb;
    uint a0; // the owner edge's i (weight 1 − s), its j a1 (weight s): tri [i, j, j]
    uint a1;
    uint h; // the contacter: the partner edge's i (1 − t) and j (t), or one node (t 0)
    uint h2;
    float s;
    float t;
    float3 n;
    float pen;
};

// Owner va's edge e against vb's edge f (va < vb): false when no contact.
bool edgeEdgeTest(uint va, uint vb, SkinE e, SkinE f, out EdgeHitG H)
{
    H.va = va;
    H.vb = vb;
    H.a0 = e.i;
    H.a1 = e.j;
    H.h = f.i;
    H.h2 = f.j;
    H.s = 0.0;
    H.t = 0.0;
    H.n = float3(0.0, 0.0, 0.0);
    H.pen = 0.0;
    if ((e.flags & 1u) == 0u || (f.flags & 1u) == 0u) return false;
    float r = edgeRange(va, vb);
    float2 st = edgeSegSeg(e.p, e.q, f.p, f.q);
    float s = st.x;
    float t = st.y;
    if (s <= EDGE_END || s >= nf(1.0 - EDGE_END) || t <= EDGE_END || t >= nf(1.0 - EDGE_END)) return false;
    float3 n = scross(ssub3(e.q, e.p), ssub3(f.q, f.p));
    float len = ssqrt(sdot(n, n));
    if (len < 1e-8) return false;
    n = sdiv3(n, len);
    bool eo = (e.flags & 2u) != 0u;
    bool fo = (f.flags & 2u) != 0u;
    if (!eo && !fo) return false;
    if (!eo)
    {
        if (sdot(n, f.o) > 0.0) n = -n;
    }
    else if (sdot(n, e.o) < 0.0) n = -n;
    if (eo && fo && sdot(n, f.o) >= 0.0) return false;
    float3 ca = sadd3(e.p, sscale(ssub3(e.q, e.p), s));
    float3 cb = sadd3(f.p, sscale(ssub3(f.q, f.p), t));
    float g = sdot(n, ssub3(cb, ca));
    if (g > r) return false;
    if (g < -EDGE_DEPTH_M) return false;
    H.s = s;
    H.t = t;
    H.n = n;
    H.pen = min(max(nf(r - g), 0.0), r);
    return true;
}

// Vehicle vb's contact point h (world p) against owner va's edge e: false when none.
bool edgeVertexTest(uint va, uint vb, SkinE e, uint h, float3 p, out EdgeHitG H)
{
    H.va = va;
    H.vb = vb;
    H.a0 = e.i;
    H.a1 = e.j;
    H.h = h;
    H.h2 = h;
    H.s = 0.0;
    H.t = 0.0;
    H.n = float3(0.0, 0.0, 0.0);
    H.pen = 0.0;
    if ((e.flags & 7u) != 7u) return false;
    if (gP.retireBase != 0u && nodeRetired(h)) return false;
    float r = edgeRange(va, vb);
    float3 d = ssub3(e.q, e.p);
    float ll = sdot(d, d);
    if (ll < 1e-12) return false;
    float s = sdiv(sdot(ssub3(p, e.p), d), ll);
    if (s <= EDGE_END || s >= nf(1.0 - EDGE_END)) return false;
    float3 c = sadd3(e.p, sscale(d, s));
    float3 pc = ssub3(p, c);
    float g = ssqrt(sdot(pc, pc));
    if (g < 1e-6 || g > r) return false;
    float3 n = sdiv3(pc, g);
    if (sdot(n, e.o) <= 0.0) return false;
    for (uint k = 0u; k < 2u; k++)
    {
        if ((e.flags & (k == 0u ? 8u : 16u)) == 0u) continue;
        float3 oc = k == 0u ? e.c0 : e.c1;
        float3 fo = sadd3(e.p, sscale(d, sdiv(sdot(ssub3(oc, e.p), d), ll)));
        if (!(sdot(ssub3(p, fo), ssub3(fo, oc)) >= 0.0)) return false;
    }
    H.s = s;
    H.n = n;
    H.pen = min(max(nf(r - g), 0.0), r);
    return true;
}

// Owner va's listed hit (two words at `at`) again.
bool edgeHitAt(uint va, uint at, out EdgeHitG H)
{
    uint w0 = gNext[at];
    uint w1 = gNext[at + 1u];
    SkinE e = edgeLoad(va, w0 & 0x7FFFFFFFu);
    if ((w0 >> 31u) == 0u)
    {
        uint vb = w1 >> 20u;
        return edgeEdgeTest(va, vb, e, edgeLoad(vb, w1 & 0xFFFFFu), H);
    }
    uint vb = gStatic[gP.nodeInfoBase + w1].w;
    return edgeVertexTest(va, vb, e, w1, edgeWorld(vb, w1), H);
}

// A found hit into owner va's list with its split counts (vvSplitCount's: the owner
// edge's two nodes, the contacter's one or two, in the owner's slot).
void edgeHitAdd(uint va, uint slot, uint w0, uint w1, EdgeHitG H)
{
    if ((edgeAblate() & 8u) != 0u) return;
    uint4 v1 = edgeV1(va);
    uint n = uint(gAcc[v1.x].add(1));
    if (n >= v1.y)
    {
        gAcc[vvCtl(VV_OVERFLOW)].max(FL_ERR_EDGES);
        return;
    }
    uint at = edgeV0(va).w + 2u * n;
    gNext[at] = w0;
    gNext[at + 1u] = w1;
    gAcc[vvSplitTri(H.a1)].add(1);
    gAcc[vvSplitTri(H.a0)].add(1);
    gAcc[vvSplitPoint(slot, H.h)].add(1);
    if (H.t > 0.0) gAcc[vvSplitPoint(slot, H.h2)].add(1);
}

uint edgeHits(uint va) { return min(uint(gAcc[edgeV1(va).x].load()), edgeV1(va).y); }

// --- The hit's first pass (explicit_contact.rs's two-body, dissipative hit) ---

struct EdgeFirst
{
    float3 f;
    float3 v;
    float imh;
    float imh2;
    float im1;
    float im0;
    float m;
    float3 fh;
    float budget;
    float pen;
    uint dissipative;
};

EdgeFirst edgeFirst(EdgeHitG H, uint slot)
{
    EdgeFirst F;
    float alpha = H.s;
    float gamma = nf(1.0 - H.s);
    float u = nf(1.0 - H.t);
    float4 Ph = gNodes[H.h * 3u];
    float4 Vh = gNodes[H.h * 3u + 1u];
    float4 Ph2 = gNodes[H.h2 * 3u];
    float4 Vh2 = gNodes[H.h2 * 3u + 1u];
    float4 P1 = gNodes[H.a1 * 3u];
    float4 P0 = gNodes[H.a0 * 3u];
    float3 cvel = H.t > 0.0 ? sadd3(sscale(Vh.xyz, u), sscale(Vh2.xyz, H.t)) : Vh.xyz;
    float3 vt = sadd3(sadd3(sscale(gNodes[H.a1 * 3u + 1u].xyz, alpha), sscale(gNodes[H.a1 * 3u + 1u].xyz, 0.0)),
                      sscale(gNodes[H.a0 * 3u + 1u].xyz, gamma));
    F.v = ssub3(cvel, vt);
    F.imh = vvImmovable(H.h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, H.h)), Ph.w);
    F.imh2 = vvImmovable(H.h2) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, H.h2)), Ph2.w);
    F.im1 = vvImmovable(H.a1) ? 0.0 : sdiv(vvSplitK(vvSplitTri(H.a1)), P1.w);
    F.im0 = vvImmovable(H.a0) ? 0.0 : sdiv(vvSplitK(vvSplitTri(H.a0)), P0.w);
    float i1 = alpha > 0.0 ? smul(smul(alpha, alpha), F.im1) : 0.0;
    float i0 = gamma > 0.0 ? smul(smul(gamma, gamma), F.im0) : 0.0;
    float imt = nf(nf(i1 + 0.0) + i0);
    float inv = H.t > 0.0 ? nf(nf(smul(smul(u, u), F.imh) + smul(smul(H.t, H.t), F.imh2)) + imt) : nf(F.imh + imt);
    F.m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
    float3 at = sadd3(sadd3(vvAccel(gBeams[gP.vvSnap + H.a1].xyz, H.a1, P1.w, alpha),
                            vvAccel(gBeams[gP.vvSnap + H.a1].xyz, H.a1, P1.w, 0.0)),
                      vvAccel(gBeams[gP.vvSnap + H.a0].xyz, H.a0, P0.w, gamma));
    float3 ac = H.t > 0.0 ? sadd3(vvAccel(gBeams[gP.vvSnap + H.h].xyz, H.h, Ph.w, u),
                                  vvAccel(gBeams[gP.vvSnap + H.h2].xyz, H.h2, Ph2.w, H.t))
                          : vvAccel(gBeams[gP.vvSnap + H.h].xyz, H.h, Ph.w, 1.0);
    F.fh = sscale(ssub3(ac, at), F.m);
    Surface surf = contactSurfaceV(H.vb, H.va);
    F.dissipative = surf.dissipative;
    float friction = H.t > 0.0 ? min(Vh.w, Vh2.w) : Vh.w;
    F.pen = H.pen;
    F.f = contactResponse(F.fh, F.v, F.m, H.n, surf, H.pen, friction, F.budget);
    return F;
}

// Node x's share of a hit's force f by role (1 h, 2 the owner's j as tri[1], 3 its j as
// tri[2] with weight 0, 4 its i, 5 h2), as the CPU reference adds it.
float3 edgeShare(float3 f, uint role, EdgeHitG H)
{
    if (role == 1u) return H.t > 0.0 ? sscale(f, nf(1.0 - H.t)) : f;
    if (role == 5u) return sscale(f, H.t);
    if (role == 2u) return -sscale(f, H.s);
    if (role == 3u) return -sscale(f, 0.0);
    return -sscale(f, nf(1.0 - H.s));
}
uint edgeNode(uint role, EdgeHitG H) { return role == 1u ? H.h : (role == 5u ? H.h2 : (role == 4u ? H.a0 : H.a1)); }
bool edgeRoleOn(uint role, EdgeHitG H) { return role != 5u || H.t > 0.0; }

// The first pass's adds (vvTry's): the position step, then the force into A.
void edgeApplyOne(EdgeHitG H, uint slot)
{
    EdgeFirst F = edgeFirst(H, slot);
    if (F.dissipative != 0u && any(F.f != float3(0.0, 0.0, 0.0)))
    {
        float delta = shiftDepth(F.pen, F.budget, sdot(F.fh, H.n));
        if (delta > 0.0)
        {
            float q = smul(delta, F.m);
            if (H.t > 0.0)
            {
                shiftAdd(H.h, sscale(H.n, smul(nf(1.0 - H.t), smul(q, F.imh))));
                shiftAdd(H.h2, sscale(H.n, smul(H.t, smul(q, F.imh2))));
            }
            else shiftAdd(H.h, sscale(H.n, smul(q, F.imh)));
            if (H.s > 0.0) shiftAdd(H.a1, sscale(H.n, -smul(H.s, smul(q, F.im1))));
            float gamma = nf(1.0 - H.s);
            if (gamma > 0.0) shiftAdd(H.a0, sscale(H.n, -smul(gamma, smul(q, F.im0))));
        }
    }
    for (uint role = 1u; role <= 5u; role++)
    {
        if (edgeRoleOn(role, H)) accAdd(edgeNode(role, H), edgeShare(F.f, role, H));
    }
}

// --- The residual pass and work guard (explicit_contact.rs residual_vehicle_pass with
// the accumulated clamp, guard_vehicle_pass), staged as vehicle contact's are ---

float3 edgeAccNow(uint x, uint v, uint parity, bool withB)
{
    return vvResAcc(x, vvImmovable(x) ? float3(0.0, 0.0, 0.0) : vvBeamSumOf(x, v, parity), withB);
}

// The hit's residual change (0: none): toward the stop at the forces now (A), never
// below zero in total.
float edgeResOf(EdgeHitG H, EdgeFirst F, uint slot, uint parity)
{
    float alpha = H.s;
    float gamma = nf(1.0 - H.s);
    float u = nf(1.0 - H.t);
    float3 at = sadd3(sadd3(sscale(edgeAccNow(H.a1, H.va, parity, false), alpha),
                            sscale(edgeAccNow(H.a1, H.va, parity, false), 0.0)),
                      sscale(edgeAccNow(H.a0, H.va, parity, false), gamma));
    float3 cacc = H.t > 0.0 ? sadd3(sscale(edgeAccNow(H.h, H.vb, parity, false), u),
                                    sscale(edgeAccNow(H.h2, H.vb, parity, false), H.t))
                            : edgeAccNow(H.h, H.vb, parity, false);
    float an = sdot(ssub3(cacc, at), H.n);
    float vn = sdot(F.v, H.n);
    float lam0 = max(sdot(F.f, H.n), 0.0);
    float imc = H.t > 0.0 ? nf(smul(smul(u, u), F.imh) + smul(smul(H.t, H.t), F.imh2)) : F.imh;
    float i1 = alpha > 0.0 ? smul(smul(alpha, alpha), F.im1) : 0.0;
    float i0 = gamma > 0.0 ? smul(smul(gamma, gamma), F.im0) : 0.0;
    float inv = nf(imc + nf(nf(i1 + 0.0) + i0));
    if (!(inv > 0.0)) return 0.0;
    float m = sdiv(1.0, inv);
    float r = nf(-smul(an, m) - sdiv(smul(vn, m), STEP));
    return nf(max(nf(lam0 + r), 0.0) - lam0);
}

// The work guard of the hit's force t at A + B (guard_vehicle_pass).
bool edgeGuardOf(EdgeHitG H, EdgeFirst F, float3 t, uint parity, out float3 g)
{
    g = float3(0.0, 0.0, 0.0);
    if (all(t == float3(0.0, 0.0, 0.0))) return false;
    float alpha = H.s;
    float gamma = nf(1.0 - H.s);
    float u = nf(1.0 - H.t);
    float3 ca = H.t > 0.0 ? sadd3(sscale(edgeAccNow(H.h, H.vb, parity, true), u),
                                  sscale(edgeAccNow(H.h2, H.vb, parity, true), H.t))
                          : edgeAccNow(H.h, H.vb, parity, true);
    float ci = H.t > 0.0 ? nf(smul(smul(u, u), ownInvMass(H.h)) + smul(smul(H.t, H.t), ownInvMass(H.h2))) : ownInvMass(H.h);
    float3 a = ssub3(ca, sadd3(sadd3(sscale(edgeAccNow(H.a1, H.va, parity, true), alpha),
                                     sscale(edgeAccNow(H.a1, H.va, parity, true), 0.0)),
                               sscale(edgeAccNow(H.a0, H.va, parity, true), gamma)));
    float inv = nf(ci + nf(nf(selfTerm(alpha, ownInvMass(H.a1)) + selfTerm(0.0, ownInvMass(H.a1))) + selfTerm(gamma, ownInvMass(H.a0))));
    float3 rest = ssub3(a, sscale(t, inv));
    float aa = sdot(t, sadd3(sadd3(F.v, F.v), sscale(rest, STEP)));
    float bb = smul(smul(STEP, sdot(t, t)), inv);
    if (!(nf(aa + bb) > 0.0 && bb > 0.0)) return false;
    float k = aa < 0.0 ? min(sdiv(-aa, bb), 1.0) : 0.0;
    g = sscale(t, -nf(1.0 - k));
    return true;
}

// One phase over owner va's edge hits (lane of lanes; slot its two-body slot): 0 the
// residual into B, 1 the guard and the totals' deltas into D, 2 the merge.
void edgeResidualPhase(uint va, uint slot, uint parity, uint lane, uint lanes, uint phase)
{
    if (gP.selfSplit == 0u || gP.vvSplit == 0u) return;
    uint n = edgeHits(va);
    uint base = edgeV0(va).w;
    for (uint i = lane, iStop = EB_END(801, i, n, lanes); i < iStop; i += lanes)
    {
        EdgeHitG H;
        if (!edgeHitAt(va, base + 2u * i, H)) continue;
        if (phase == 2u)
        {
            for (uint role = 1u; role <= 5u; role++)
            {
                if (edgeRoleOn(role, H)) resMergeNode(edgeNode(role, H));
            }
            continue;
        }
        EdgeFirst F = edgeFirst(H, slot);
        float rr = edgeResOf(H, F, slot, parity);
        if (phase == 0u)
        {
            if (rr != 0.0)
            {
                float3 f = sscale(H.n, rr);
                for (uint role = 1u; role <= 5u; role++)
                {
                    if (!edgeRoleOn(role, H)) continue;
                    float3 s = edgeShare(f, role, H);
                    uint at = resAt(edgeNode(role, H));
                    gAcc[at + 1u].add(toFixed(s.x));
                    gAcc[at + 2u].add(toFixed(s.y));
                    gAcc[at + 3u].add(toFixed(s.z));
                }
            }
            continue;
        }
        float3 total = F.f;
        if (rr != 0.0) total = sadd3(total, sscale(H.n, rr));
        float3 g;
        if (edgeGuardOf(H, F, total, parity, g)) total = sadd3(total, g);
        for (uint role = 1u; role <= 5u; role++)
        {
            if (!edgeRoleOn(role, H)) continue;
            uint x = edgeNode(role, H);
            float3 st = edgeShare(total, role, H);
            int3 d = toFixed3(st) - toFixed3(edgeShare(F.f, role, H));
            uint at = resAt(x);
            if (d.x != 0) gAcc[at + 4u].add(d.x);
            if (d.y != 0) gAcc[at + 5u].add(d.y);
            if (d.z != 0) gAcc[at + 6u].add(d.z);
            cfAdd(x, st);
        }
    }
}

// The node residual's accumulated clamp for an edge-contact pair (vvResOf's caller):
// r toward the stop, never below zero in total, the hit's first-pass normal force λ0.
float edgeClampRes(float lam0, float r) { return nf(max(nf(lam0 + r), 0.0) - lam0); }

// --- The contact groups' passes (members off.., one workgroup) ---

// Every member's counters cleared and, with `prepare`, its edges this step (their scratch
// and box). Two barriers.
void edgeResetPrepare(uint k, uint off, uint lane, bool prepare)
{
    for (uint a = lane, aStop = EB_END(814, a, k, LANES); a < aStop; a += LANES)
    {
        uint v1x = edgeV1(vvMember(off, a)).x;
        gAcc[v1x].store(0);
        gAcc[v1x + 1u].store(0);
        gAcc[v1x + 2u].store(0);
        for (uint c = 0u; c < 6u; c++) gAcc[v1x + 3u + c].store(c < 3u ? VV_INT_MAX : VV_INT_MIN);
        gAcc[v1x + 9u].store(0);
    }
    sync();
    uint total = 0u;
    for (uint a = 0u, aStop = EB_END(815, a, prepare ? k : 0u, 1u); a < aStop; a++) total += edgeV0(vvMember(off, a)).y;
    for (uint p = lane, pStop = EB_END(802, p, total, LANES); p < pStop; p += LANES)
    {
        uint q = p;
        uint a = 0u;
        for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
        {
            EB_PASS(803, cGuard);
            uint n = edgeV0(vvMember(off, a)).y;
            if (q < n) break;
            q -= n;
            a++;
        }
        edgePrepare(vvMember(off, a), q);
    }
    sync();
}

// Vehicle v's live edges in [lo, hi] into its edge list, its contact points (`points`) too.
void edgeList(uint v, float3 lo, float3 hi, bool points, uint lane)
{
    uint4 v0 = edgeV0(v);
    uint4 v1 = edgeV1(v);
    for (uint e = lane, eStop = EB_END(825, e, v0.y, LANES); e < eStop; e += LANES)
    {
        float4 s0 = gBeams[v0.z + EDGE_SCR * e];
        if ((asuint(s0.w) & 1u) == 0u) continue;
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
float edgeLmax(uint v) { return asfloat(uint(gAcc[edgeV1(v).x + 9u].load())); }
uint edgeGridAt(uint v, uint kind) { return edgeV1(v).x + 12u + kind * 2u * EDGE_BUCKETS; }
uint edgeBlocksAt(uint v, uint kind) { return edgeV1(v).x + 12u + 4u * EDGE_BUCKETS + kind * (EDGE_BUCKETS / EDGE_BLOCK); }
uint edgeListsLen(uint v) { return edgeV0(v).y + vvCpCount(v); }
uint edgeEntriesAt(uint v, uint kind) { return edgeV1(v).w + edgeListsLen(v) + (kind == 0u ? 0u : EDGE_CELLS_MAX * edgeV0(v).y); }
uint edgeLongAt(uint v) { return edgeEntriesAt(v, 1u) + 8u * vvCpCount(v); }
// The owner side's work: per listed edge its first (edge, cell) item, then the scan's block sums.
uint edgeWorkAt(uint v) { return edgeLongAt(v) + edgeV0(v).y; }
uint edgeWorkBlocksAt(uint v) { return edgeWorkAt(v) + edgeV0(v).y; }
// The frame-coherent candidates: gAcc per vehicle after the grid (count, its signature when
// built, overflow), gNext after the work list (two words a pair, as a hit), gBeams after its
// edges' scratch its nodes at the last build.
uint edgeCandAcc(uint v) { return edgeV1(v).x + 12u + 4u * EDGE_BUCKETS + 2u * (EDGE_BUCKETS / EDGE_BLOCK); }
uint edgeCandAt(uint v) { return edgeWorkBlocksAt(v) + edgeV0(v).y / 32u + 1u; }
uint edgeCandRoom(uint v) { return EDGE_CAND * edgeListsLen(v); }
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
    if ((edgeAblate() & 2u) != 0u) return;
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
            uint n = gNext[work + i];
            gNext[work + i] = run;
            run += n;
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

// --- The small groups' pass (vvPassSmall: k ≤ VV_SIDE; gsVvPair the overlapping ordered
// pairs) ---

// Each member's edges and contact points within `reach` of a partner's edges' box (the
// lists' counts cleared first). Two barriers.
void edgeListsSmall(uint k, uint off, float reach, bool anyEdge, uint lane)
{
    for (uint a = lane, aStop = EB_END(866, a, k, LANES); a < aStop; a += LANES)
    {
        uint v1x = edgeV1(vvMember(off, a)).x;
        gAcc[v1x + 1u].store(0);
        gAcc[v1x + 2u].store(0);
    }
    sync();
    for (uint a = 0u, aStop = EB_END(817, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        float3 ulo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 uhi = -ulo;
        for (uint m = 0u, mStop = EB_END(818, m, k, 1u); m < mStop; m++)
        {
            if (m == a || (gsVvPair[a * VV_PART + m] & 1u) == 0u || !edgePairOn(va, vvMember(off, m))) continue;
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

// Every owner against each partner (the grid built). One barrier.
void edgePairsSmall(uint k, uint off, float cell, float reach, bool anyEdge, uint lane)
{
    for (uint a = 0u, aStop = EB_END(819, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        for (uint m = 0u, mStop = EB_END(820, m, k, 1u); m < mStop; m++)
        {
            uint vb = vvMember(off, m);
            if (m == a || (gsVvPair[a * VV_PART + m] & 1u) == 0u || !edgePairOn(va, vb)) continue;
            edgePairHits(va, vb, a, cell, reach, lane);
        }
    }
    sync();
}

// Before the tests' end: every member's edges this step, their boxes, the lists of each
// member's edges and contact points near its partners, then each owner's hits.
void edgeTestsSmall(uint k, uint off, uint parity, uint lane)
{
    if (!edgeOn()) return;
    // (No early return on the pairs: they are workgroup memory, and a barrier after a
    // branch on it is not uniform control flow to a browser; no pair, no work.)
    bool anyEdge = false;
    for (uint a = 0u, aStop = EB_END(812, a, k, 1u); a < aStop; a++)
        for (uint m = 0u, mStop = EB_END(813, m, k, 1u); m < mStop; m++)
            if (a != m && (gsVvPair[a * VV_PART + m] & 1u) != 0u && edgePairOn(vvMember(off, a), vvMember(off, m))) anyEdge = true;
    edgeResetPrepare(k, off, lane, anyEdge);
    float rmax = 0.0;
    for (uint a = 0u, aStop = EB_END(816, a, k, 1u); a < aStop; a++) rmax = max(rmax, asfloat(gStatic[vvMember(off, a) * 4u + 3u].z));
    float pad = nf(rmax + EDGE_DEPTH_M);
    // The candidates of the last build stand while no node of the group has moved a skin
    // since and no owner's partners changed (any pair in contact now was then within reach +
    // 2 skins); else they are found again. (Lossless: the same hits as a search every step.)
    for (uint a = 0u, aStop = EB_END(854, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
    {
        uint v = vvMember(off, a);
        if (gAcc[edgeCandAcc(v) + 1u].load() == 0) continue;
        uint4 r0 = gStatic[v * 4u];
        uint built = edgeBuiltAt(v);
        for (uint i = lane, iStop = EB_END(855, i, r0.y, LANES); i < iStop; i += LANES)
        {
            float3 d = ssub3(edgeWorld(v, r0.x + i), gBeams[built + i].xyz);
            gAcc[edgeV1(v).x + 9u].max(asint(sdot(d, d)));
        }
    }
    sync();
    if (lane == 0u)
    {
        uint rebuild = 0u;
        for (uint a = 0u, aStop = EB_END(856, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
        {
            uint va = vvMember(off, a);
            uint sig = 0x9E3779B9u;
            for (uint m = 0u, mStop = EB_END(857, m, k, 1u); m < mStop; m++)
            {
                uint vb = vvMember(off, m);
                if (m != a && (gsVvPair[a * VV_PART + m] & 1u) != 0u && edgePairOn(va, vb)) sig ^= (vb + 1u) * 2654435761u;
            }
            sig |= 1u;
            uint ca = edgeCandAcc(va);
            if (uint(gAcc[ca + 1u].load()) != sig || gAcc[ca + 2u].load() != 0
                || asfloat(gAcc[edgeV1(va).x + 9u].load()) > EDGE_SKIN * EDGE_SKIN)
                rebuild = 1u;
        }
        gsUniformLanes = rebuild;
    }
    sync();
    uint rebuild = gsUniformLanes;
    if (rebuild != 0u)
    {
        float reach = nf(pad + 2.0 * EDGE_SKIN);
        for (uint a = lane, aStop = EB_END(858, a, k, LANES); a < aStop; a += LANES)
        {
            uint ca = edgeCandAcc(vvMember(off, a));
            gAcc[ca].store(0);
            gAcc[ca + 2u].store(0);
        }
        edgeListsSmall(k, off, reach, anyEdge, lane);
        for (uint stage = 0u; stage < 5u; stage++)
        {
            for (uint a = 0u, aStop = EB_END(842, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
            {
                edgeGridStage(vvMember(off, a), 2.0 * reach, reach, stage, lane);
                edgeWorkStage(vvMember(off, a), 2.0 * reach, stage, lane);
            }
            sync();
        }
        gEdgeCollect = 1u;
        gEdgeCollectReach = reach;
        edgePairsSmall(k, off, 2.0 * reach, reach, anyEdge, lane);
        gEdgeCollect = 0u;
        // The build's positions and signatures (an overflowed list builds again next step).
        for (uint a = 0u, aStop = EB_END(859, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
        {
            uint v = vvMember(off, a);
            uint4 r0 = gStatic[v * 4u];
            uint built = edgeBuiltAt(v);
            for (uint i = lane, iStop = EB_END(860, i, r0.y, LANES); i < iStop; i += LANES) gBeams[built + i] = float4(edgeWorld(v, r0.x + i), 0.0);
            if (lane == 0u)
            {
                uint sig = 0x9E3779B9u;
                for (uint m = 0u, mStop = EB_END(861, m, k, 1u); m < mStop; m++)
                {
                    uint vb = vvMember(off, m);
                    if (m != a && (gsVvPair[a * VV_PART + m] & 1u) != 0u && edgePairOn(v, vb)) sig ^= (vb + 1u) * 2654435761u;
                }
                gAcc[edgeCandAcc(v) + 1u].store(gAcc[edgeCandAcc(v) + 2u].load() != 0 ? 0 : int(sig | 1u));
            }
        }
        sync();
    }
    // An overflowed build: this step's pairs by the search itself.
    if (lane == 0u)
    {
        uint over = 0u;
        for (uint a = 0u, aStop = EB_END(862, a, anyEdge ? k : 0u, 1u); a < aStop; a++) over |= uint(gAcc[edgeCandAcc(vvMember(off, a)) + 2u].load());
        gsUniformLanes = over;
    }
    sync();
    uint over = gsUniformLanes;
    if (over != 0u)
    {
        edgeListsSmall(k, off, pad, anyEdge, lane);
        for (uint stage = 0u; stage < 5u; stage++)
        {
            for (uint a = 0u, aStop = EB_END(863, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
            {
                edgeGridStage(vvMember(off, a), 2.0 * pad, pad, stage, lane);
                edgeWorkStage(vvMember(off, a), 2.0 * pad, stage, lane);
            }
            sync();
        }
        edgePairsSmall(k, off, 2.0 * pad, pad, anyEdge, lane);
    }
    else
    {
        // The candidates tested, each owner's on all lanes.
        for (uint a = 0u, aStop = EB_END(864, a, anyEdge ? k : 0u, 1u); a < aStop; a++)
        {
            uint va = vvMember(off, a);
            uint n = min(uint(gAcc[edgeCandAcc(va)].load()), edgeCandRoom(va));
            uint base = edgeCandAt(va);
            for (uint i = lane, iStop = EB_END(865, i, n, LANES); i < iStop; i += LANES)
            {
                EdgeHitG H;
                if (edgeHitAt(va, base + 2u * i, H)) edgeHitAdd(va, a, gNext[base + 2u * i], gNext[base + 2u * i + 1u], H);
            }
        }
    }
    sync();
    // (The levels give an owner with edge hits alone its level: vvLevelsSet.)
    for (uint a = lane, aStop = EB_END(824, a, k, LANES); a < aStop; a += LANES) gsEdgeHas[a] = edgeHits(vvMember(off, a)) != 0u ? 1u : 0u;
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
