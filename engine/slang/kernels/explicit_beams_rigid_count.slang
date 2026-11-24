// Every lane: its nodes against the proxy shapes, each inside counted on its body.
void cplCount(uint v, uint lane)
{
    cplCountBy(v, lane, LANES);
}
// The same over `lanes` lanes.
void cplCountBy(uint v, uint lane, uint lanes)
{
    uint4 hdr = cplVeh(v);
    if (hdr.y == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    float3 origin = asfloat(gMisc[v].xyz);
    uint acc = cplAcc(v);
    bool cull = cplCullOn();
    uint2 gr = cplGsRange(v);
    uint2 gn = cplGsNodes(v);
    for (uint i = lane, iStop = EB_END(458, i, r0.y, lanes); i < iStop; i += lanes)
    {
        uint gi = r0.x + i;
        uint range = CPL_NONE;
        if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) == 0u)
        {
            // Vehicle-local, as the proxy records (relative to the vehicle's origin).
            float3 x = gNodes[gi * 3u].xyz;
            range = cplCountShapes(v, x, gi, 0u, hdr.y, acc, gr, 0u, cull, 0.0, 0.0, 1u);
        }
        if (i < gn.y) gsCpl[CPL_GS_NODE + gn.x + i] = range;
    }
}

// --- The candidate list (lossless; engine_softbody::explicit_rigid::candidate_list, the
// same rule and arithmetic; ExplicitBeamsGpu::set_coupling_candidate_list) ---
// Every node tested every shape every step (cplCount, then cplContacts without node
// ranges): O(nodes x shapes) a step. With the list each node scans every shape only on a
// frame's first step (or when its check fails: it rebuilds its own list, no barrier) and
// lists the shapes within its skin m of their cull spheres (sqrt(r2) + m); later steps
// test only the listed shapes, in shape order, while |x - x0| + E < m - E(build), E the
// frame's running maximum of any shape centre's distance from its centre at the frame's
// first step. An unlisted shape then stays outside its true bounding radius (the cull
// sphere is grown past it by at least 1 cm: cplSpheresBy), so cplFace rejects it: the
// same counts, contacts and sums, bit for bit. Off with the coupling header's bit 128
// (and whenever the cull is off, bit 2); bits 256 and 512 are tests' skins.
//
// gAcc per vehicle 2 words right after the coupling's sums (CouplingLayout::list_acc):
// E and the shapes' fastest point speed, as float bits (atomic max; cleared at the
// frame's start, cplListBegin). gNext right after the spheres (CouplingLayout::list_base):
// per vehicle and shape its centre at the frame's first step (4 words), then per node
// (global index) 12 words: x0, the limit m - E(build) (-1: no list), then 16 shape
// indices, 16 bits each, in order, ended by 0xFFFF (0xFFFE first: more than 16 in reach,
// every shape tested).
static const uint CPL_LIST_CAP = 16u;
static const uint CPL_LIST_WORDS = 12u;
static const uint CPL_LIST_ALL = 0xFFFEu;
static const float CPL_LIST_MIN = 0.02;
static const float CPL_LIST_MAX = 4.0;

bool cplListOn() { return (gsCpl[0] & 130u) == 0u; }
uint cplListAcc(uint v) { return cplHdr(3u) + gP.vehicles * (cplB() + 6u * cplS()) + 2u * v; }
uint cplListShape(uint v, uint k) { return cplHdr(4u) + gP.vehicles * cplS() * CPL_SPHERE_WORDS + (v * cplS() + k) * 4u; }
uint cplListNode(uint gi) { return cplHdr(4u) + gP.vehicles * cplS() * (CPL_SPHERE_WORDS + 4u) + gi * CPL_LIST_WORDS; }
// A dispatch's first step: every node builds its list there (a later segment of a split
// frame starts its own: the shapes' centres taken again, every list rebuilt).
bool cplListFirst(uint s)
{
#ifdef EB_FRAME_SEG
    return s == 0u || s == gP.pad2;
#else
    return s == 0u;
#endif
}

// The frame's start (lane 0, before the barrier the spheres' pass follows): vehicle v's
// maxima cleared.
void cplListBegin(uint v)
{
    if (gsCpl[0] == 0u || !cplListOn()) return;
    uint la = cplListAcc(v);
    gAcc[la].store(0);
    gAcc[la + 1u].store(0);
}

// After cplSpheresBy, the same lanes and shapes (each lane reads the spheres it wrote):
// each shape's distance from its frame-start centre and its fastest point's speed into
// the vehicle's maxima (non-negative floats order as their bits; not finite: 3e38).
void cplListShapes(uint v, uint lane, uint lanes, uint s)
{
    if (!cplListOn()) return;
    uint shapes = cplVeh(v).y;
    uint la = cplListAcc(v);
    bool first = cplListFirst(s);
    for (uint k = lane, kStop = EB_END(460, k, shapes, lanes); k < kStop; k += lanes)
    {
        uint sp = cplSphere(v, k);
        float3 p = asfloat(uint3(gNext[sp], gNext[sp + 1u], gNext[sp + 2u]));
        uint r = cplListShape(v, k);
        float3 p0 = p;
        if (first)
        {
            gNext[r] = asuint(p.x);
            gNext[r + 1u] = asuint(p.y);
            gNext[r + 2u] = asuint(p.z);
        }
        else
            p0 = asfloat(uint3(gNext[r], gNext[r + 1u], gNext[r + 2u]));
        float e = length(p - p0);
        uint b = cplBody(v, gMisc[cplShape(v, k)].y);
        float3 c = asfloat(gMisc[b].xyz);
        float3 bv = asfloat(gMisc[b + 2u].xyz);
        float3 bw = asfloat(gMisc[b + 3u].xyz);
        float u = length(bv) + length(bw) * length(p - c);
        int ei = asint(e < 3.0e38 ? e : 3.0e38);
        int ui = asint(u < 3.0e38 ? u : 3.0e38);
        // A racy read first: most shapes raise neither maximum (no atomic).
        if (ei > gAcc[la].load()) gAcc[la].max(ei);
        if (ui > gAcc[la + 1u].load()) gAcc[la + 1u].max(ui);
    }
}

// A node's skin (candidate_list::margin): the closing distance over the frame's rest,
// a quarter more, within [CPL_LIST_MIN, CPL_LIST_MAX].
float cplListMargin(float speed, float vs, uint left)
{
    if ((gsCpl[0] & 512u) != 0u) return CPL_LIST_MAX;
    if ((gsCpl[0] & 256u) != 0u) vs = 0.0;
    float m = 1.25 * STEP * float(left) * (speed + vs) + CPL_LIST_MIN;
    return m < CPL_LIST_MAX ? max(m, CPL_LIST_MIN) : CPL_LIST_MAX;
}

// Shape k's sphere words (cplMay's).
uint4 cplSphereWords(uint v, uint k, uint2 gr)
{
    if (k < gr.y)
    {
        uint g = CPL_GS_SPH + 4u * (gr.x + k);
        return uint4(gsCpl[g], gsCpl[g + 1u], gsCpl[g + 2u], gsCpl[g + 3u]);
    }
    uint s = cplSphere(v, k);
    return uint4(gNext[s], gNext[s + 1u], gNext[s + 2u], gNext[s + 3u]);
}

// cplCountBy's test of one shape: counted on its body and in the range when inside.
uint cplCountOne(uint v, uint k, float3 x, uint acc, uint range)
{
    if (cplFace(v, k, x).w == 0.0) return range;
    gAcc[acc + gMisc[cplShape(v, k)].y].add(1);
    return range == CPL_NONE ? (k | (k << 16)) : ((range & 0xFFFFu) | (k << 16));
}

// Node x's (node gi's) shapes counted, the range of those it is inside: mode 0 every
// shape (cplCountBy; the sphere first when culled), 1 the node's list holds (its listed
// shapes, in order, the sphere first), 2 the node (re)building its list (every shape, the
// ones within its skin listed). One function, so a count step that takes one of the
// three per node has one copy of cplCountOne (cplFace) for them (the compilers inline
// every call): the same tests and writes in the same order as the three had.
uint cplCountShapes(uint v, float3 x, uint gi, uint nl, uint shapes, uint acc, uint2 gr, uint mode, bool cull, float E,
                    float vs, uint left)
{
    float m = mode == 2u ? cplListMargin(length(gNodes[gi * 3u + 1u].xyz), vs, left) : 0.0;
    uint range = CPL_NONE;
    uint n = 0u;
    uint word = 0xFFFFFFFFu;
    for (uint e = 0u, eStop = EB_END(461, e, mode == 1u ? CPL_LIST_CAP : shapes, 1u); e < eStop; e++)
    {
        uint k = e;
        if (mode == 1u)
        {
            k = (gNext[nl + 4u + (e >> 1)] >> (16u * (e & 1u))) & 0xFFFFu;
            if (k >= shapes) break;
        }
        // (cplMay's test: the sphere's words, inside when dd < its radius squared.)
        uint4 w = cplSphereWords(v, k, gr);
        float3 d = x - asfloat(w.xyz);
        float dd = dot(d, d);
        if ((mode == 0u && !cull) || dd < asfloat(w.w)) range = cplCountOne(v, k, x, acc, range);
        if (mode != 2u) continue;
        float l = sqrt(asfloat(w.w)) + m;
        // NaN lists the shape.
        if (!(dd >= l * l))
        {
            if (n < CPL_LIST_CAP)
            {
                word = (n & 1u) == 0u ? (0xFFFF0000u | k) : ((word & 0xFFFFu) | (k << 16));
                gNext[nl + 4u + (n >> 1)] = word;
            }
            n++;
        }
    }
    if (mode == 2u)
    {
        if (n > CPL_LIST_CAP)
            gNext[nl + 4u] = 0xFFFF0000u | CPL_LIST_ALL;
        else if ((n & 1u) == 0u && n < CPL_LIST_CAP)
            gNext[nl + 4u + (n >> 1)] = 0xFFFFFFFFu;
        gNext[nl] = asuint(x.x);
        gNext[nl + 1u] = asuint(x.y);
        gNext[nl + 2u] = asuint(x.z);
        gNext[nl + 3u] = asuint(n > CPL_LIST_CAP ? -1.0 : m - E);
    }
    return range;
}
// The node's mode for the listed count (cplCountShapes): 1 its list holds, 2 it builds.
uint cplListMode(uint gi, float3 x, float E, bool first)
{
    uint nl = cplListNode(gi);
    float4 h = asfloat(uint4(gNext[nl], gNext[nl + 1u], gNext[nl + 2u], gNext[nl + 3u]));
    return !first && length(x - h.xyz) + E < h.w ? 1u : 2u;
}

// cplCountBy with the list (step s; the cull on).
void cplCountListedGr(uint v, uint lane, uint lanes, uint s, uint2 gr);
void cplCountListed(uint v, uint lane, uint lanes, uint s)
{
    cplCountListedGr(v, lane, lanes, s, cplGsRange(v));
}
// The same with the spheres read from workgroup memory for the shapes below gr.y
// (cplGsRange), from gNext for the rest (gr (0, 0): every one from gNext).
void cplCountListedGr(uint v, uint lane, uint lanes, uint s, uint2 gr)
{
    uint4 hdr = cplVeh(v);
    if (hdr.y == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint acc = cplAcc(v);
    uint2 gn = cplGsNodes(v);
    uint la = cplListAcc(v);
    float E = asfloat(gAcc[la].load());
    float vs = asfloat(gAcc[la + 1u].load());
    bool first = cplListFirst(s);
    uint left = gP.steps > s ? gP.steps - s : 1u;
    for (uint i = lane, iStop = EB_END(463, i, r0.y, lanes); i < iStop; i += lanes)
    {
        uint gi = r0.x + i;
        uint range = CPL_NONE;
        if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) == 0u)
        {
            float3 x = gNodes[gi * 3u].xyz;
            range = cplCountShapes(v, x, gi, cplListNode(gi), hdr.y, acc, gr, cplListMode(gi, x, E, first), true, E, vs,
                                   left);
        }
        if (i < gn.y) gsCpl[CPL_GS_NODE + gn.x + i] = range;
    }
}

// The count pass (step s): with the list, or every shape.
void cplCountStep(uint v, uint lane, uint lanes, uint s)
{
    // (With the list or every shape, one node loop of cplCountShapes: cplCountListed's and
    // cplCountBy's in one, the mode chosen per node.)
    bool listOn = cplListOn();
    uint4 hdr = cplVeh(v);
    if (hdr.y == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint acc = cplAcc(v);
    uint2 gr = cplGsRange(v);
    uint2 gn = cplGsNodes(v);
    bool cull = cplCullOn();
    float E = 0.0;
    float vs = 0.0;
    if (listOn)
    {
        uint la = cplListAcc(v);
        E = asfloat(gAcc[la].load());
        vs = asfloat(gAcc[la + 1u].load());
    }
    bool first = cplListFirst(s);
    uint left = gP.steps > s ? gP.steps - s : 1u;
    for (uint i = lane, iStop = EB_END(650, i, r0.y, lanes); i < iStop; i += lanes)
    {
        uint gi = r0.x + i;
        uint range = CPL_NONE;
        if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) == 0u)
        {
            float3 x = gNodes[gi * 3u].xyz;
            range = cplCountShapes(v, x, gi, cplListNode(gi), hdr.y, acc, gr, listOn ? cplListMode(gi, x, E, first) : 0u,
                                   listOn || cull, E, vs, left);
        }
        if (i < gn.y) gsCpl[CPL_GS_NODE + gn.x + i] = range;
    }
}

// --- The folded step (lossless; 2 barriers a step, not 4) ---
// The coupling's step was four phases (cplReactBy | cplIntegrate | cplSpheresBy |
// cplCountBy), a barrier after each. Folded, it is two:
//  B  each shape's lane computes its body's integration itself (the sums of the body's
//     shapes read, not cleared, in shape order, the arithmetic of cplReactBy and
//     cplIntegrateBodySplit: the same bits) and from it the shape's sphere and step
//     frame (cplSpheresBy's arithmetic) into gNext, and the list's maxima; each body's
//     lane computes the same integration into registers and clears the body's count;
//  C  each body's lane writes its record; each shape's lane does cplReactBy's writes
//     (the window impulse and moment, its sums cleared) and copies its sphere into
//     workgroup memory; every node is counted (cplCountListedGr, the spheres from
//     gNext: their workgroup copies are being written in this phase).
// Nothing in B writes what B reads (the body records, the sums, the spheres' workgroup
// words); nothing in C reads what C writes. The node phase after C's barrier reads the
// same records, sums, counts and spheres as after the four phases. Only for the solo
// frame, with the list and the workgroup copies on, every sphere in workgroup memory
// (cplSplit) and no more bodies than lanes (one a lane: its registers across the
// barrier); otherwise the four phases. Off with the coupling header's bit 1024.
// The frame's fold decision for the barriers' control flow: lane 0 at the frame's start
// (cplFoldMark, after cplGsAdd) sets this bit of gsUniformCouple, which every lane reads
// with workgroupUniformLoad (a decision read from storage is not uniform for Chrome's
// WGSL: its barriers would be rejected).
static const uint CPL_FOLD_FRAME = 0x100000u;
bool cplFoldOn(uint v, uint lanes);
void cplFoldMark(uint v)
{
    // Written, never read here: every read of gsUniformCouple becomes a
    // workgroupUniformLoad (a barrier), which lane 0 alone must not execute. gsCpl[0] is
    // the same word (coupleBegin).
    if (gsCpl[0] != 0u && cplFoldOn(v, LANES)) gsUniformCouple = gsCpl[0] | CPL_FOLD_FRAME;
}
bool cplFoldOn(uint v, uint lanes)
{
    if ((gsCpl[0] & 1024u) != 0u || !cplListOn()) return false;
    uint4 hdr = cplVeh(v);
    return cplSplit(v, hdr.y) && hdr.x <= lanes;
}

// Body j's integration over this step's sums (cplReactBy then cplIntegrateBodySplit, the
// same arithmetic and order; the sums read, not cleared): its new centre (.w 1/M), rotation,
// velocity (.w its share) and angular velocity.
void cplBodyStepped(uint v, uint j, uint shapes, out float4 c0, out float4 q, out float4 c2, out float3 w)
{
    uint b = cplBody(v, j);
    uint acc = cplAcc(v);
    uint nb = cplB();
    c0 = asfloat(gMisc[b]);
    q = asfloat(gMisc[b + 1u]);
    c2 = asfloat(gMisc[b + 2u]);
    float4 c3 = asfloat(gMisc[b + 3u]);
    float4 c4 = asfloat(gMisc[b + 4u]);
    float3 force = float3(0.0, 0.0, 0.0);
    float3 torque = float3(0.0, 0.0, 0.0);
    // A body of infinite mass integrates no force: its sums are not read.
    if (c0.w != 0.0)
    {
        uint range = asuint(c4.w);
        uint k0 = range == 0u ? 0u : (range & 0xFFFFu) - 1u;
        uint k1 = range == 0u ? shapes : min(range >> 16, shapes);
        uint s0 = cplShape(v, 0u);
        for (uint k = k0, kStop = EB_END(465, k, k1, 1u); k < kStop; k++)
        {
            uint at = s0 + k * CPL_SHAPE_RECS;
            if (gMisc[at].y != j) continue;
            uint s = acc + nb + 6u * k;
            float3 F = float3(float(gAcc[s].load()), float(gAcc[s + 1u].load()), float(gAcc[s + 2u].load())) * (1.0 / FIXED);
            float3 M = float3(float(gAcc[s + 3u].load()), float(gAcc[s + 4u].load()), float(gAcc[s + 5u].load())) * (1.0 / FIXED);
            float3 ref = asfloat(gMisc[at + 6u].xyz);
            float3 t = cplTorqueTerm(ref, c0.xyz, F, M);
            force += F;
            torque += t;
        }
    }
    w = c3.xyz;
    cplIntegrateStep(c0, q, c2, w, c4, force, torque);
}

// Phase B for shape k: its sphere and step frame (cplSpheresBy's arithmetic) from its
// body's pose this step (stepped: cplBodyStepped; else the record's), into gNext only.
void cplShapeFolded(uint v, uint k, uint shapes, bool stepped)
{
    uint at = cplShape(v, k);
    uint4 h = gMisc[at];
    float4 c0;
    float4 q;
    if (stepped)
    {
        float4 c2;
        float3 w;
        cplBodyStepped(v, h.y, shapes, c0, q, c2, w);
    }
    else
    {
        uint b = cplBody(v, h.y);
        c0 = asfloat(gMisc[b]);
        q = asfloat(gMisc[b + 1u]);
    }
    float3 c = c0.xyz;
    float4 lp = asfloat(gMisc[at + 1u]);
    float4 pr = asfloat(gMisc[at + 3u]);
    float3 p = c + cplRot(q, lp.xyz);
    float big = max(max(abs(p.x), abs(p.y)), max(abs(p.z), abs(pr.z)));
    float r = pr.z * 1.001 + 0.01 + 1.0e-4 * big;
    bool finite = all(abs(p) < 1.0e30) && abs(r) < 1.0e18;
    uint s = cplSphere(v, k);
    gNext[s] = asuint(p.x);
    gNext[s + 1u] = asuint(p.y);
    gNext[s + 2u] = asuint(p.z);
    gNext[s + 3u] = asuint(finite ? r * r : 3.0e38);
    cplFrameStore(s, h, lp, pr, cplShapeRot(q, asfloat(gMisc[at + 2u])));
}

// The folded step of vehicle v (every lane, uniform: cplFoldOn): phases B and C, the
// barrier between and after.
// cplStepFolded's two phases as calls (the caller's barrier between and after): B, whose
// body lane keeps its integrated record in registers for C.
void cplFoldB(uint v, uint lane, uint lanes, uint s, out bool body, out float4 c0, out float4 q, out float4 c2,
              out float3 w)
{
    uint4 hdr = cplVeh(v);
    bool stepped = s > 0u;
    for (uint k = lane, kStop = EB_END(466, k, hdr.y, lanes); k < kStop; k += lanes) cplShapeFolded(v, k, hdr.y, stepped);
    cplListShapes(v, lane, lanes, s);
    body = stepped && lane < hdr.x;
    c0 = float4(0.0, 0.0, 0.0, 0.0);
    q = float4(0.0, 0.0, 0.0, 1.0);
    c2 = float4(0.0, 0.0, 0.0, 0.0);
    w = float3(0.0, 0.0, 0.0);
    if (body)
    {
        cplBodyStepped(v, lane, hdr.y, c0, q, c2, w);
        gAcc[cplAcc(v) + lane].store(0);
    }
}
void cplFoldC(uint v, uint lane, uint lanes, uint s, bool body, float4 c0, float4 q, float4 c2, float3 w)
{
    uint4 hdr = cplVeh(v);
    bool stepped = s > 0u;
    if (body)
    {
        uint b = cplBody(v, lane);
        gMisc[b] = asuint(c0);
        gMisc[b + 1u] = asuint(q);
        gMisc[b + 2u] = asuint(c2);
        gMisc[b + 3u] = uint4(asuint(w), gMisc[b + 3u].w);
    }
    uint2 gr = cplGsRange(v);
    uint acc = cplAcc(v);
    uint nb = cplB();
    for (uint k = lane, kStop = EB_END(467, k, hdr.y, lanes); k < kStop; k += lanes)
    {
        if (stepped)
        {
            // cplReactBy's writes (every shape's body is below the lanes here).
            uint at = cplShape(v, k);
            uint sm = acc + nb + 6u * k;
            float3 F = float3(float(gAcc[sm].load()), float(gAcc[sm + 1u].load()), float(gAcc[sm + 2u].load())) * (1.0 / FIXED);
            float3 M = float3(float(gAcc[sm + 3u].load()), float(gAcc[sm + 4u].load()), float(gAcc[sm + 5u].load())) * (1.0 / FIXED);
            for (uint u = 0u; u < 6u; u++) gAcc[sm + u].store(0);
            cplWindowAdd(at, F, M);
        }
        if (k < gr.y)
        {
            uint sp = cplSphere(v, k);
            uint g = CPL_GS_SPH + 4u * (gr.x + k);
            gsCpl[g] = gNext[sp];
            gsCpl[g + 1u] = gNext[sp + 1u];
            gsCpl[g + 2u] = gNext[sp + 2u];
            gsCpl[g + 3u] = gNext[sp + 3u];
        }
    }
    cplCountListedGr(v, lane, lanes, s, uint2(0u, 0u));
}
void cplStepFolded(uint v, uint lane, uint lanes, uint s)
{
    bool body;
    float4 c0;
    float4 q;
    float4 c2;
    float3 w;
    cplFoldB(v, lane, lanes, s, body, c0, q, c2, w);
    sync();
    cplFoldC(v, lane, lanes, s, body, c0, q, c2, w);
    sync();
}

// The coupling's part of a step of vehicle v, at the step's start (uniform `on`).
void coupleStep(uint v, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    // Folded (2 barriers; cplStepFolded): the solo frame's usual case (uniform: on).
    if ((on & CPL_FOLD_FRAME) != 0u)
    {
#ifdef EB_ABLATE
        if ((gP.ablate & 57344u) == 0u)
#endif
        {
            cplStepFolded(v, lane, LANES, s);
            return;
        }
    }
#ifdef EB_ABLATE
    // Measurement only: 8192 the bodies' integration, 16384 the spheres, 32768 the count.
    if (s > 0u && (gP.ablate & 8192u) == 0u)
    {
        cplReactBy(v, lane, LANES);
        sync();
        cplIntegrate(v, lane);
    }
    sync();
    if ((gP.ablate & 16384u) == 0u)
    {
        cplSpheres(v, lane);
        cplListShapes(v, lane, LANES, s);
    }
    sync();
    if ((gP.ablate & 32768u) == 0u) cplCountStep(v, lane, LANES, s);
    sync();
    return;
#endif
    if (s > 0u)
    {
        cplReactBy(v, lane, LANES);
        sync();
        cplIntegrate(v, lane);
    }
    sync();
    cplSpheres(v, lane);
    cplListShapes(v, lane, LANES, s);
    sync();
    cplCountStep(v, lane, LANES, s);
    sync();
}

// After a frame's last step (its closing barrier passed): that step's integration.
void coupleEnd(uint v, uint lane, uint on)
{
    if (on == 0u) return;
    if (gP.steps > 0u)
    {
        cplReactBy(v, lane, LANES);
        sync();
        cplIntegrate(v, lane);
    }
}

// coupleStep and coupleEnd for a contact group's k members.
void coupleStepMembers(uint off, uint k, uint lane, uint s, uint on)
{
    if (on == 0u) return;
#ifdef EB_ABLATE
    if (s > 0u && (gP.ablate & 8192u) == 0u)
    {
        for (uint m = 0u, mStop = EB_END(468, m, k, 1u); m < mStop; m++) cplReactBy(vvMember(off, m), lane, LANES);
        sync();
        for (uint m = 0u, mStop = EB_END(469, m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
    }
    sync();
    if ((gP.ablate & 16384u) == 0u)
        for (uint m = 0u, mStop = EB_END(470, m, k, 1u); m < mStop; m++)
        {
            cplSpheres(vvMember(off, m), lane);
            cplListShapes(vvMember(off, m), lane, LANES, s);
        }
    sync();
    if ((gP.ablate & 32768u) == 0u)
        for (uint m = 0u, mStop = EB_END(471, m, k, 1u); m < mStop; m++) cplCountStep(vvMember(off, m), lane, LANES, s);
    sync();
    return;
#endif
    if (k <= CPL_GS_VEHICLES && (on & 4u) == 0u)
    {
        // A small group: its members side by side, vvPartLanes(k) lanes each (the
        // members' proxies are their own: the same work per member as one after the
        // other, the counts integer sums).
        uint P = vvPartLanes(k);
        uint m = lane / P;
        uint pl = lane % P;
        bool on = m < k;
        uint v = vvMember(off, on ? m : 0u);
        if (s > 0u)
        {
            if (on) cplReactBy(v, pl, P);
            sync();
            if (on) cplIntegrateBy(v, pl, P);
        }
        sync();
        if (on)
        {
            cplSpheresBy(v, pl, P);
            cplListShapes(v, pl, P, s);
        }
        sync();
        if (on) cplCountStep(v, pl, P, s);
        sync();
        return;
    }
    if (s > 0u)
    {
        for (uint m = 0u, mStop = EB_END(472, m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
    }
    sync();
    for (uint m = 0u, mStop = EB_END(473, m, k, 1u); m < mStop; m++)
    {
        cplSpheres(vvMember(off, m), lane);
        cplListShapes(vvMember(off, m), lane, LANES, s);
    }
    sync();
    for (uint m = 0u, mStop = EB_END(474, m, k, 1u); m < mStop; m++) cplCountStep(vvMember(off, m), lane, LANES, s);
    sync();
}

void coupleEndMembers(uint off, uint k, uint lane, uint on)
{
    if (on == 0u) return;
    sync();
    if (gP.steps > 0u && k <= CPL_GS_VEHICLES && (on & 4u) == 0u)
    {
        uint P = vvPartLanes(k);
        uint m = lane / P;
        if (m < k) cplReactBy(vvMember(off, m), lane % P, P);
        sync();
        if (m < k) cplIntegrateBy(vvMember(off, m), lane % P, P);
    }
    else if (gP.steps > 0u)
    {
        for (uint m = 0u, mStop = EB_END(475, m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
    }
    sync();
}

// nodeUpdateShapes with the proxy contact after the static shapes (a vehicle with
// proxies this window; the static path's code stays as it was).
uint nodeUpdateCoupled(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                       float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                       bool solo)
{
    float4 P = gNodes[gi * 3u];
#ifdef EB_VV_SOLO
    if (solo && any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
#endif
    float4 V = gNodes[gi * 3u + 1u];
    cfRoll(gi, groupOff, flagsV);
    clipRoll(gi, flagsV);
    if (gP.selfSplit != 0u && (flagsV & FLAG_TWO_BODY) != 0u) gAcc[gP.selfSplit + gi].store(0);
    float3 shift = shiftTake(gi, flagsV); // §37's contact position step
    float lift = 0.0; // §37's ground position step
    float selfGroup = gNodes[gi * 3u + 2u].w;
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
#ifdef EB_AERO
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
#endif
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        if ((flags & NODE_TREAD) != 0u)
        {
            float3 c = treadGround(gi, f, vel, mass, V.w, ground - x.y, ground > x.y);
            if (ground > x.y && any(c != float3(0.0, 0.0, 0.0)))
            {
                cfAdd(gi, c);
                f += c;
            }
        }
        else if (ground > x.y)
        {
            float3 c = groundContact(f, vel, mass, vehicle, ground - x.y, V.w, lift);
            cfAdd(gi, c);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(476, b, boxCount, 1u); b < bStop; b++)
        {
            uint4 lo4 = gStatic[gP.boxBase + (boxOff + b) * 2u];
            uint4 hi4 = gStatic[gP.boxBase + (boxOff + b) * 2u + 1u];
            float3 lo = asfloat(lo4.xyz);
            float3 hi = asfloat(hi4.xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float4 n = float4(0.0, 0.0, -1.0, 1.0);
                if (lo4.w == 0u)
                {
                    float d = x.z - lo.z;
                    if (hi.z - x.z < d) { d = hi.z - x.z; n.xyz = float3(0.0, 0.0, 1.0); }
                    if (x.x - lo.x < d) { d = x.x - lo.x; n.xyz = float3(-1.0, 0.0, 0.0); }
                    if (hi.x - x.x < d) { d = hi.x - x.x; n.xyz = float3(1.0, 0.0, 0.0); }
                    if (x.y - lo.y < d) { d = x.y - lo.y; n.xyz = float3(0.0, -1.0, 0.0); }
                    if (hi.y - x.y < d) { d = hi.y - x.y; n.xyz = float3(0.0, 1.0, 0.0); }
                }
                else
                    n = shapeNormal(lo4.w, gP.boxBase + hi4.w, x);
                if (n.w != 0.0)
                {
                    float3 c = contactForce(f, vel, mass, n.xyz, surfaceV(boxSurface(), vehicle), 0.0, V.w);
                    cfAdd(gi, c);
                    f += c;
                }
            }
        }
#ifdef EB_ABLATE
        // Measurement only: 65536 the node's proxy contact.
        float3 fr = (gP.ablate & 65536u) != 0u ? f : cplContacts(vehicle, local, pos, f, vel, mass, V.w);
#else
        float3 fr = cplContacts(vehicle, local, pos, f, vel, mass, V.w);
#endif
        cfAdd(gi, fr - f);
        f = fr;
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
        pos = takeSteps(pos, shift, lift);
    }
    clipSpeed(vehicle, step, flagsV, vel);
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
        gNodes[gi * 3u] = float4(pos, mass);
        gNodes[gi * 3u + 1u] = float4(vel, V.w);
        gNodes[gi * 3u + 2u] = float4(f, selfGroup);
        return 0u;
    }
#endif
    f = float3(0.0, mass * GRAVITY, 0.0);
    if ((flagsV & FLAG_DRAG) != 0u)
    {
        float s = approxSqrt(dot(vel, vel));
        f += -DRAG * s * vel;
        if ((flagsV & FLAG_TURBULENCE) != 0u)
        {
            float t = DRAG * s * s * TURBULENCE;
            f += t * float3(noise(vehicle, step, local, 0u), noise(vehicle, step, local, 1u),
                            noise(vehicle, step, local, 2u));
        }
    }
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
    return 0u;
}

// Whether vehicle v has proxy shapes this window (the coupling on, read at the
// frame's start, and its region non-empty).
// The wide passes (no coupleBegin, gsCpl unset) read the header itself: the coupling
// is off whenever they run, so the record arithmetic never sees an unset copy.
bool cplHas(uint v)
{
#ifdef EB_CPL_DIRECT
    // (gsCpl[0] is coupleBegin's copy of the header's .x: the one condition.)
    return gMisc[coupleRec()].x != 0u && gMisc[cplBase(v)].y != 0u;
#else
    return gsCpl[0] != 0u && gMisc[coupleRec()].x != 0u && gMisc[cplBase(v)].y != 0u;
#endif
}

// The frame's start (lane 0, before the barrier the uniform read follows).
void coupleBegin()
{
    uint4 h = gMisc[coupleRec()];
    uint on = h.x;
    gsUniformCouple = on;
    gsCpl[0] = on;
    uint4 h1 = gMisc[coupleRec() + 1u];
    gsCpl[CPL_GS_HDR] = h.y;
    gsCpl[CPL_GS_HDR + 1u] = h.z;
    gsCpl[CPL_GS_HDR + 2u] = h.w;
    gsCpl[CPL_GS_HDR + 3u] = h1.x;
    gsCpl[CPL_GS_HDR + 4u] = h1.y;
}
