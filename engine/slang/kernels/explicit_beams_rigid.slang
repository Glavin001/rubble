// Explicit vehicles against rigid bodies (engine/docs/explicit-vehicles-in-world.md;
// CPU reference engine_softbody::explicit_rigid). Included by explicit_beams.slang.
//
// Off (the coupling header's .x is 0, the default) nothing here runs: the frame reads
// one record at its start and every vehicle steps as before.
//
// gMisc at coupleRec() (after the grab's records): (on, region base, B, S | planes << 16)
// with B proxy bodies, S proxy shapes and `planes` hull planes per vehicle. Per vehicle
// v its region at base + v * cplRecs():
//   header      (bodies, shapes, planes used, 0)
//   B bodies    5 records: (centre of mass, 1/M), rotation, (velocity, share V),
//               (angular velocity, engine body), (principal 1/I, 0)
//   S shapes    7 records: (kind, body slot, collider, plane offset | count << 16),
//               (local position, p0), local rotation, (p1, p2, cull radius, 0),
//               (window impulse, peak force), (window moment about the reference, 0),
//               (reference point, 0)
//   planes      (normal, offset) in the shape frame
// kinds: 0 box (half p0 p1 p2), 1 sphere (radius p0), 2 capsule along local y (radius
// p0, half length p1), 3 hull.
// gAcc at gMisc[coupleRec() + 1].x, per vehicle B + 6 S words: per body this step's contact count, per shape this step's reaction force and
// its moment about the reference, in 1/32 N (N m).
//
// Per step: the bodies integrate the last step's sums (lanes < B), a barrier, every
// node is counted against the proxy shapes (mass splitting), a barrier; the node phase
// (nodeUpdateCoupled) adds the two-body contact and sums the reactions. Every barrier
// is under gsUniformCouple (workgroupUniformLoad, uniform).

// The coupling header's .x at the frame's start: read once uniformly
// (gsUniformCouple, workgroupUniformLoad: the barriers below depend on it) and by the
// node phase (gsCpl[0]).
groupshared uint gsUniformCouple;
// gsCpl[0] is word 0 of gsCpl (below): Metal binds each workgroup variable as an
// argument, at most 31 per kernel.
uint vvMember(uint off, uint m);
uint vvPartLanes(uint k);

static const uint CPL_BOX = 0u;
static const uint CPL_SPHERE = 1u;
static const uint CPL_CAPSULE = 2u;
static const uint CPL_HULL = 3u;
static const uint CPL_BODY_RECS = 5u;
static const uint CPL_SHAPE_RECS = 7u;

static const uint CPL_GS = 256u;
static const uint CPL_GS_VEHICLES = 4u;
// Per node (its vehicle's first nodes from its node base, while they fit in
// CPL_GS_NODES): the first and last shape the count pass found it inside
// (first | last << 16), CPL_NONE when none. The node phase that step tests the same
// node position against the same shapes (nothing moves the nodes or the bodies
// between), so it starts at the first and stops after the last: the shapes outside
// that range were rejected (no force, no sum), the rest are tested as before, in order.
static const uint CPL_GS_NODES = 1024u;
static const uint CPL_NONE = 0xFFFFFFFFu;
// One workgroup array (Metal binds each workgroup variable as an argument, at most 31
// a kernel): word 0 the coupling header's .x, then per table entry (vehicle, base,
// count, node base, node count), then the coupling header's words (CPL_GS_HDR: region
// base, B, S | planes << 16, the accumulators' base, the spheres' base; constant
// through a frame, copied by coupleBegin so the record arithmetic reads no buffer),
// then the spheres (4 words each), then the nodes' ranges.
static const uint CPL_GS_TAB = 1u;
static const uint CPL_GS_HDR = 21u;
static const uint CPL_GS_SPH = 32u;
static const uint CPL_GS_NODE = CPL_GS_SPH + 4u * CPL_GS;
groupshared uint gsCpl[CPL_GS_NODE + CPL_GS_NODES];


uint coupleRec() { return grabRec() + 2u; }
// Header word i (CPL_GS_HDR order); from the buffer when the workgroup copies are
// off (the header's .x bit 2, tests).
uint cplHdr(uint i)
{
    if ((gsCpl[0] & 4u) == 0u) return gsCpl[CPL_GS_HDR + i];
    uint4 h = gMisc[coupleRec()];
    uint4 h1 = gMisc[coupleRec() + 1u];
    return i == 0u ? h.y : (i == 1u ? h.z : (i == 2u ? h.w : (i == 3u ? h1.x : h1.y)));
}
uint cplB() { return cplHdr(1u); }
uint cplS() { return cplHdr(2u) & 0xFFFFu; }
uint cplPlanes() { return cplHdr(2u) >> 16; }
uint cplRecs() { return 1u + cplB() * CPL_BODY_RECS + cplS() * CPL_SHAPE_RECS + cplPlanes(); }
uint cplBase(uint v) { return cplHdr(0u) + v * cplRecs(); }
uint cplBody(uint v, uint j) { return cplBase(v) + 1u + j * CPL_BODY_RECS; }
uint cplShape(uint v, uint k) { return cplBase(v) + 1u + cplB() * CPL_BODY_RECS + k * CPL_SHAPE_RECS; }
uint cplPlane(uint v, uint i) { return cplBase(v) + 1u + cplB() * CPL_BODY_RECS + cplS() * CPL_SHAPE_RECS + i; }
uint cplAcc(uint v) { return cplHdr(3u) + v * (cplB() + 6u * cplS()); }

float3 cplRot(float4 q, float3 v)
{
    float3 u = q.xyz;
    float3 t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}

float4 cplQmul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// The outward normal (.w 1) of the proxy shape's surface nearest x, for x strictly
// inside the shape, else .w 0 (the box rule; CPU `ProxySet::face`).
float4 cplFace(uint v, uint k, float3 x)
{
    uint at = cplShape(v, k);
    uint4 h = gMisc[at];
    uint b = cplBody(v, h.y);
    float3 c = asfloat(gMisc[b].xyz);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 lp = asfloat(gMisc[at + 1u]);
    float4 lq = asfloat(gMisc[at + 2u]);
    float4 pr = asfloat(gMisc[at + 3u]);
    float3 p = c + cplRot(q, lp.xyz);
    float3 d = x - p;
    // Cull: the shape's bounding sphere.
    if (dot(d, d) >= pr.z * pr.z) return float4(0.0, 0.0, 0.0, 0.0);
    float4 r = normalize(cplQmul(q, lq));
    float3 l = cplRot(float4(-r.xyz, r.w), d);
    if (h.x == CPL_BOX)
    {
        float3 hh = float3(lp.w, pr.x, pr.y);
        if (!(l.x > -hh.x && l.x < hh.x && l.y > -hh.y && l.y < hh.y && l.z > -hh.z && l.z < hh.z))
            return float4(0.0, 0.0, 0.0, 0.0);
        // Faces in box_face's order (-z, +z, -x, +x, -y, +y), the first nearest.
        float3 n = float3(0.0, 0.0, -1.0);
        float dist = l.z + hh.z;
        float t = hh.z - l.z;
        if (t < dist) { dist = t; n = float3(0.0, 0.0, 1.0); }
        t = l.x + hh.x;
        if (t < dist) { dist = t; n = float3(-1.0, 0.0, 0.0); }
        t = hh.x - l.x;
        if (t < dist) { dist = t; n = float3(1.0, 0.0, 0.0); }
        t = l.y + hh.y;
        if (t < dist) { dist = t; n = float3(0.0, -1.0, 0.0); }
        t = hh.y - l.y;
        if (t < dist) { dist = t; n = float3(0.0, 1.0, 0.0); }
        return float4(cplRot(r, n), 1.0);
    }
    if (h.x == CPL_SPHERE)
    {
        float len = length(d);
        if (!(len < lp.w)) return float4(0.0, 0.0, 0.0, 0.0);
        return len > 0.0 ? float4(d / len, 1.0) : float4(0.0, 1.0, 0.0, 1.0);
    }
    if (h.x == CPL_CAPSULE)
    {
        float y = clamp(l.y, -pr.x, pr.x);
        float3 e = l - float3(0.0, y, 0.0);
        float len = length(e);
        if (!(len < lp.w)) return float4(0.0, 0.0, 0.0, 0.0);
        return len > 0.0 ? float4(cplRot(r, e / len), 1.0) : float4(0.0, 1.0, 0.0, 1.0);
    }
    // Hull: inside every plane; the nearest plane (the first of equals).
    uint off = h.w & 0xFFFFu;
    uint count = h.w >> 16;
    float best = -3.0e38;
    float3 n = float3(0.0, 1.0, 0.0);
    for (uint i = 0u, iStop = EB_END(i, count, 1u); i < iStop; i++)
    {
        float4 pl = asfloat(gMisc[cplPlane(v, off + i)]);
        float s = dot(pl.xyz, l) - pl.w;
        if (!(s < 0.0)) return float4(0.0, 0.0, 0.0, 0.0);
        if (s > best) { best = s; n = pl.xyz; }
    }
    return float4(cplRot(r, n), 1.0);
}

// --- The per-step shape cull (lossless; engine/docs/explicit-vehicles-in-world.md §6) ---
// Every node tests every proxy shape of its vehicle (cplCount, cplContacts), and a
// shape whose bounding sphere does not hold the node returns nothing at once
// (cplFace) after reading its shape and body records and turning its position. Each
// step, once the bodies integrate, every shape's sphere is written once to the
// vehicle's row in gNext: centre and a radius grown well past the rounding of
// cplFace's own test. A node skips a shape outside that grown sphere, one cplFace
// would have rejected; the shapes it keeps are tested as before, in shape order, so
// forces, sums and their order are unchanged, bit for bit (engine-world
// tests/explicit_vehicles.rs checks it against the cull off). The coupling header's
// .x bit 1 turns the cull off.
//
// gNext at gMisc[coupleRec() + 1].y, per vehicle S x 4 words: per shape (centre,
// grown radius squared) (engine_exec::explicit_plan coupling layout).
uint cplSphere(uint v, uint k) { return cplHdr(4u) + (v * cplS() + k) * 4u; }
bool cplCullOn() { return (gsCpl[0] & 2u) == 0u; }

// The spheres in workgroup memory as well (the same words): the frame's vehicle, or a
// small contact group's members, each its first shapes from its base while they fit
// in CPL_GS (the rest read from gNext). A table of up to CPL_GS_VEHICLES vehicles and
// their bases, written by lane 0 at the frame's start (cplGsClear, cplGsAdd, before the
// barrier the reads follow); an empty table reads every sphere from gNext.

// Lane 0: an empty table.
void cplGsClear()
{
    for (uint e = 0u; e < CPL_GS_VEHICLES; e++)
    {
        gsCpl[CPL_GS_TAB + 5u * e + 0u] = 0xFFFFFFFFu;
        gsCpl[CPL_GS_TAB + 5u * e + 2u] = 0u;
        gsCpl[CPL_GS_TAB + 5u * e + 4u] = 0u;
    }
}
// Vehicle v as entry e: its spheres from `base`, its nodes' ranges from `nodeBase`.
void cplGsAdd(uint e, uint v, uint base, uint nodeBase)
{
    // The header's .x bit 2 (tests): no workgroup copies, every read from the buffers.
    if ((gsCpl[0] & 4u) != 0u) return;
    uint shapes = gsCpl[0] != 0u ? gMisc[cplBase(v)].y : 0u;
    uint nodes = gStatic[v * 4u].y;
    gsCpl[CPL_GS_TAB + 5u * e + 0u] = v;
    gsCpl[CPL_GS_TAB + 5u * e + 1u] = base;
    gsCpl[CPL_GS_TAB + 5u * e + 2u] = base < CPL_GS ? min(shapes, CPL_GS - base) : 0u;
    gsCpl[CPL_GS_TAB + 5u * e + 3u] = nodeBase;
    gsCpl[CPL_GS_TAB + 5u * e + 4u] = shapes > 0u && nodeBase < CPL_GS_NODES ? min(nodes, CPL_GS_NODES - nodeBase) : 0u;
}

// Vehicle v's node ranges in workgroup memory: (base, count), (0, 0) when none.
uint2 cplGsNodes(uint v)
{
    for (uint e = 0u; e < CPL_GS_VEHICLES; e++)
    {
        if (gsCpl[CPL_GS_TAB + 5u * e + 0u] == v) return uint2(gsCpl[CPL_GS_TAB + 5u * e + 3u], gsCpl[CPL_GS_TAB + 5u * e + 4u]);
    }
    return uint2(0u, 0u);
}

// Vehicle v's spheres in workgroup memory: (base, count), (0, 0) when none.
uint2 cplGsRange(uint v)
{
    for (uint e = 0u; e < CPL_GS_VEHICLES; e++)
    {
        if (gsCpl[CPL_GS_TAB + 5u * e + 0u] == v) return uint2(gsCpl[CPL_GS_TAB + 5u * e + 1u], gsCpl[CPL_GS_TAB + 5u * e + 2u]);
    }
    return uint2(0u, 0u);
}

// Every lane, after the bodies' integration (its barrier passed): each shape's sphere.
void cplSpheres(uint v, uint lane)
{
    cplSpheresBy(v, lane, LANES);
}
void cplSpheresBy(uint v, uint lane, uint lanes)
{
    uint shapes = cplCullOn() ? gMisc[cplBase(v)].y : 0u;
    uint2 gr = cplGsRange(v);
    for (uint k = lane, kStop = EB_END(k, shapes, lanes); k < kStop; k += lanes)
    {
        // The sphere as cplFace computes it.
        uint at = cplShape(v, k);
        uint4 h = gMisc[at];
        uint b = cplBody(v, h.y);
        float3 c = asfloat(gMisc[b].xyz);
        float4 q = asfloat(gMisc[b + 1u]);
        float4 lp = asfloat(gMisc[at + 1u]);
        float4 pr = asfloat(gMisc[at + 3u]);
        float3 p = c + cplRot(q, lp.xyz);
        float big = max(max(abs(p.x), abs(p.y)), max(abs(p.z), abs(pr.z)));
        float r = pr.z * 1.001 + 0.01 + 1.0e-4 * big;
        // Not finite (or NaN): every node tests it.
        bool finite = all(abs(p) < 1.0e30) && abs(r) < 1.0e18;
        uint s = cplSphere(v, k);
        uint4 w = uint4(asuint(p.x), asuint(p.y), asuint(p.z), asuint(finite ? r * r : 3.0e38));
        gNext[s] = w.x;
        gNext[s + 1u] = w.y;
        gNext[s + 2u] = w.z;
        gNext[s + 3u] = w.w;
        if (k < gr.y)
        {
            uint g = CPL_GS_SPH + 4u * (gr.x + k);
            gsCpl[g] = w.x;
            gsCpl[g + 1u] = w.y;
            gsCpl[g + 2u] = w.z;
            gsCpl[g + 3u] = w.w;
        }
    }
}

// Whether a node at world x may be inside shape k (false: cplFace rejects it).
// (gr: cplGsRange(v).)
bool cplMay(uint v, uint k, float3 x, bool cull, uint2 gr)
{
    if (!cull) return true;
    uint4 w;
    if (k < gr.y)
    {
        uint g = CPL_GS_SPH + 4u * (gr.x + k);
        w = uint4(gsCpl[g], gsCpl[g + 1u], gsCpl[g + 2u], gsCpl[g + 3u]);
    }
    else
    {
        uint s = cplSphere(v, k);
        w = uint4(gNext[s], gNext[s + 1u], gNext[s + 2u], gNext[s + 3u]);
    }
    float3 d = x - asfloat(w.xyz);
    return dot(d, d) < asfloat(w.w);
}

// A fast proxy's push direction for a node inside it (its own function: the shipped
// path keeps its code).
float4 cplFastSide(uint v, uint k, float3 vel, float4 n4)
{
    float3 vb;
    if (!surfProxyFast(v, k, vb)) return n4;
    float3 d = vb - vel;
    // Only while the body overtakes the node (CPU node_contact_with).
    if (!(dot(d, d) > 0.0) || !(dot(d, vb) > 0.0)) return n4;
    return float4(normalize(d), 1.0);
}

// The two-body contact on a node of vehicle v at world x (force f so far, velocity vel,
// mass, friction) from every proxy shape it is inside, in shape order; the reactions are
// summed into the shapes' accumulators. Returns the node's force with them added.
// `local`: the node's index in its vehicle (its count pass range, cplGsNodes).
bool surfProxyFast(uint v, uint k, out float3 vb);

float3 cplContacts(uint v, uint local, float3 x, float3 f, float3 vel, float mass, float friction)
{
    // §38.4 (explicit_beams_surface.slang FLAG_SURFACE): a fast proxy pushes the nodes
    // inside it out along its motion relative to them (CPU ProxySet::node_contact_with).
    bool fastSide = (gStatic[v * 4u + 3u].x & 131072u) != 0u && gP.surfSplit != 0u;
    uint4 hdr = gMisc[cplBase(v)];
    uint acc = cplAcc(v);
    uint nb = cplB();
    bool cull = cplCullOn();
    uint2 gr = cplGsRange(v);
    uint k0 = 0u;
    uint k1 = hdr.y;
    uint2 gn = cplGsNodes(v);
    if (local < gn.y)
    {
        uint r = gsCpl[CPL_GS_NODE + gn.x + local];
        if (r == CPL_NONE) return f;
        k0 = r & 0xFFFFu;
        k1 = min((r >> 16) + 1u, hdr.y);
    }
    for (uint k = k0, kStop = EB_END(k, k1, 1u); k < kStop; k++)
    {
        if (!cplMay(v, k, x, cull, gr)) continue;
        float4 n4 = cplFace(v, k, x);
        if (n4.w == 0.0) continue;
        if (fastSide) n4 = cplFastSide(v, k, vel, n4);
        uint at = cplShape(v, k);
        uint j = gMisc[at].y;
        uint b = cplBody(v, j);
        float4 c0 = asfloat(gMisc[b]);
        float4 q = asfloat(gMisc[b + 1u]);
        float4 c2 = asfloat(gMisc[b + 2u]);
        float4 c3 = asfloat(gMisc[b + 3u]);
        float3 r = x - c0.xyz;
        float3 vb = c2.xyz + cross(c3.xyz, r);
        float3 fc;
        if (c0.w == 0.0)
        {
            // Infinite mass: the static shape rule (relative to a kinematic body's motion).
            float3 u = all(vb == float3(0.0, 0.0, 0.0)) ? vel : vel - vb;
            fc = contactForce(f, u, mass, n4.xyz, boxSurface(), 0.0, friction);
        }
        else
        {
            float3 ii = asfloat(gMisc[b + 4u].xyz);
            float3 rn = cross(r, n4.xyz);
            float3 lr = cplRot(float4(-q.xyz, q.w), rn);
            float3 irn = cplRot(q, ii * lr);
            float split = float(max(uint(gAcc[acc + j].load()), 1u)) * max(c2.w, 1.0);
            float wb = split * (c0.w + dot(rn, irn));
            float wn = 1.0 / mass;
            float mred = 1.0 / (wn + wb);
            float share = wn / (wn + wb);
            fc = contactForce(f * share, vel - vb, mred, n4.xyz, boxSurface(), 0.0, friction);
        }
        f += fc;
        float3 react = -fc;
        float3 m = cross(x - asfloat(gMisc[at + 6u].xyz), react);
        uint s = acc + nb + 6u * k;
        gAcc[s].add(toFixed(react.x));
        gAcc[s + 1u].add(toFixed(react.y));
        gAcc[s + 2u].add(toFixed(react.z));
        gAcc[s + 3u].add(toFixed(m.x));
        gAcc[s + 4u].add(toFixed(m.y));
        gAcc[s + 5u].add(toFixed(m.z));
    }
    return f;
}

// Every body j < B (lane j, then j + LANES, ...: a vehicle may hold up to 1024 proxy
// bodies, more than its lanes) sums its shapes' step reactions (into the shapes' window
// totals, the sums cleared), then integrates (symplectic Euler, as the nodes); its count
// cleared.
void cplIntegrate(uint v, uint lane)
{
    uint4 hdr = gMisc[cplBase(v)];
    for (uint j = lane, jStop = EB_END(j, hdr.x, LANES); j < jStop; j += LANES) cplIntegrateBody(v, j);
}
// The same bodies (every j < B), body j by lane j of `lanes`, then j + lanes, ...
void cplIntegrateBy(uint v, uint lane, uint lanes)
{
    uint4 hdr = gMisc[cplBase(v)];
    for (uint j = lane, jStop = EB_END(j, hdr.x, lanes); j < jStop; j += lanes) cplIntegrateBody(v, j);
}
// Whether vehicle v's step sums go through workgroup memory (cplReactBy): its every
// sphere is there (so its shapes fit, 6 words each, in the spheres' and the nodes'
// ranges' words, free between the node phase and the spheres' pass).
bool cplSplit(uint v, uint shapes)
{
    uint2 gr = cplGsRange(v);
    return shapes > 0u && gr.y == shapes;
}

// Before the bodies' integration (a barrier between): each shape's step reaction, a
// lane a shape (the work cplIntegrateBody did per shape, one shape after another, for
// the same shapes: their body j < B and j < LANES), its force and its torque term
// about the body's centre left in workgroup memory for the body's sum in shape order.
void cplReactBy(uint v, uint lane, uint lanes)
{
    uint4 hdr = gMisc[cplBase(v)];
    if (!cplSplit(v, hdr.y)) return;
    uint2 gr = cplGsRange(v);
    uint bodies = min(hdr.x, LANES);
    uint acc = cplAcc(v);
    uint nb = cplB();
    uint s0 = cplShape(v, 0u);
    for (uint k = lane, kStop = EB_END(k, hdr.y, lanes); k < kStop; k += lanes)
    {
        uint at = s0 + k * CPL_SHAPE_RECS;
        uint j = gMisc[at].y;
        if (j >= bodies) continue;
        float3 c = asfloat(gMisc[cplBody(v, j)].xyz);
        uint s = acc + nb + 6u * k;
        float3 F = float3(float(gAcc[s].load()), float(gAcc[s + 1u].load()), float(gAcc[s + 2u].load())) * (1.0 / FIXED);
        float3 M = float3(float(gAcc[s + 3u].load()), float(gAcc[s + 4u].load()), float(gAcc[s + 5u].load())) * (1.0 / FIXED);
        for (uint w = 0u; w < 6u; w++) gAcc[s + w].store(0);
        float4 imp = asfloat(gMisc[at + 4u]);
        float4 mom = asfloat(gMisc[at + 5u]);
        float3 ref = asfloat(gMisc[at + 6u].xyz);
        gMisc[at + 4u] = asuint(float4(imp.xyz + F * STEP, max(imp.w, length(F))));
        gMisc[at + 5u] = asuint(float4(mom.xyz + M * STEP, 0.0));
        float3 t = M + cross(ref - c, F);
        uint g = CPL_GS_SPH + 6u * (gr.x + k);
        gsCpl[g] = asuint(F.x);
        gsCpl[g + 1u] = asuint(F.y);
        gsCpl[g + 2u] = asuint(F.z);
        gsCpl[g + 3u] = asuint(t.x);
        gsCpl[g + 4u] = asuint(t.y);
        gsCpl[g + 5u] = asuint(t.z);
    }
}

void cplIntegrateBody(uint v, uint j)
{
    // The split path its own function behind an early return (Metal's codegen of
    // the path below stays as it was).
    if (cplSplit(v, gMisc[cplBase(v)].y))
    {
        cplIntegrateBodySplit(v, j);
        return;
    }
    uint4 hdr = gMisc[cplBase(v)];
    uint b = cplBody(v, j);
    uint acc = cplAcc(v);
    uint nb = cplB();
    float4 c0 = asfloat(gMisc[b]);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 c2 = asfloat(gMisc[b + 2u]);
    float4 c3 = asfloat(gMisc[b + 3u]);
    float4 c4 = asfloat(gMisc[b + 4u]);
    float3 force = float3(0.0, 0.0, 0.0);
    float3 torque = float3(0.0, 0.0, 0.0);
    // Its shapes' range from the plan (c4.w, vehicle_coupling.slang cpl_plan; 0 every
    // shape): the shapes outside it are another body's, skipped either way. The shape
    // records from one base (the same addresses as cplShape).
    uint range = (gsCpl[0] & 4u) != 0u ? 0u : asuint(c4.w);
    uint k0 = range == 0u ? 0u : (range & 0xFFFFu) - 1u;
    uint k1 = range == 0u ? hdr.y : min(range >> 16, hdr.y);
    uint s0 = cplShape(v, 0u);
    for (uint k = k0, kStop = EB_END(k, k1, 1u); k < kStop; k++)
    {
        uint at = s0 + k * CPL_SHAPE_RECS;
        if (gMisc[at].y != j) continue;
        uint s = acc + nb + 6u * k;
        float3 F = float3(float(gAcc[s].load()), float(gAcc[s + 1u].load()), float(gAcc[s + 2u].load())) * (1.0 / FIXED);
        float3 M = float3(float(gAcc[s + 3u].load()), float(gAcc[s + 4u].load()), float(gAcc[s + 5u].load())) * (1.0 / FIXED);
        for (uint w = 0u; w < 6u; w++) gAcc[s + w].store(0);
        float4 imp = asfloat(gMisc[at + 4u]);
        float4 mom = asfloat(gMisc[at + 5u]);
        float3 ref = asfloat(gMisc[at + 6u].xyz);
        gMisc[at + 4u] = asuint(float4(imp.xyz + F * STEP, max(imp.w, length(F))));
        gMisc[at + 5u] = asuint(float4(mom.xyz + M * STEP, 0.0));
        force += F;
        torque += M + cross(ref - c0.xyz, F);
    }
    float3 vel = c2.xyz;
    float3 w = c3.xyz;
    if (c0.w != 0.0)
    {
        vel += force * (c0.w * STEP);
        float3 lt = cplRot(float4(-q.xyz, q.w), torque);
        w += cplRot(q, c4.xyz * lt) * STEP;
    }
    float3 pos = c0.xyz + vel * STEP;
    if (any(w != float3(0.0, 0.0, 0.0)))
    {
        float4 dq = cplQmul(float4(w, 0.0), q);
        q = normalize(q + 0.5 * STEP * dq);
    }
    gMisc[b] = asuint(float4(pos, c0.w));
    gMisc[b + 1u] = asuint(q);
    gMisc[b + 2u] = asuint(float4(vel, c2.w));
    gMisc[b + 3u] = uint4(asuint(w), gMisc[b + 3u].w);
    gAcc[acc + j].store(0);
}

void cplIntegrateBodySplit(uint v, uint j)
{
    uint4 hdr = gMisc[cplBase(v)];
    uint b = cplBody(v, j);
    uint acc = cplAcc(v);
    uint nb = cplB();
    float4 c0 = asfloat(gMisc[b]);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 c2 = asfloat(gMisc[b + 2u]);
    float4 c3 = asfloat(gMisc[b + 3u]);
    float4 c4 = asfloat(gMisc[b + 4u]);
    float3 force = float3(0.0, 0.0, 0.0);
    float3 torque = float3(0.0, 0.0, 0.0);
    // Its shapes' range from the plan (c4.w, vehicle_coupling.slang cpl_plan; 0 every
    // shape): the shapes outside it are another body's, skipped either way. The shape
    // records from one base (the same addresses as cplShape).
    uint range = (gsCpl[0] & 4u) != 0u ? 0u : asuint(c4.w);
    uint k0 = range == 0u ? 0u : (range & 0xFFFFu) - 1u;
    uint k1 = range == 0u ? hdr.y : min(range >> 16, hdr.y);
    uint s0 = cplShape(v, 0u);
    // The shapes' terms from cplReactBy, summed in shape order.
    uint gx = cplGsRange(v).x;
    for (uint k = k0, kStop = EB_END(k, k1, 1u); k < kStop; k++)
    {
        if (gMisc[s0 + k * CPL_SHAPE_RECS].y != j) continue;
        uint g = CPL_GS_SPH + 6u * (gx + k);
        force += asfloat(uint3(gsCpl[g], gsCpl[g + 1u], gsCpl[g + 2u]));
        torque += asfloat(uint3(gsCpl[g + 3u], gsCpl[g + 4u], gsCpl[g + 5u]));
    }
    float3 vel = c2.xyz;
    float3 w = c3.xyz;
    if (c0.w != 0.0)
    {
        vel += force * (c0.w * STEP);
        float3 lt = cplRot(float4(-q.xyz, q.w), torque);
        w += cplRot(q, c4.xyz * lt) * STEP;
    }
    float3 pos = c0.xyz + vel * STEP;
    if (any(w != float3(0.0, 0.0, 0.0)))
    {
        float4 dq = cplQmul(float4(w, 0.0), q);
        q = normalize(q + 0.5 * STEP * dq);
    }
    gMisc[b] = asuint(float4(pos, c0.w));
    gMisc[b + 1u] = asuint(q);
    gMisc[b + 2u] = asuint(float4(vel, c2.w));
    gMisc[b + 3u] = uint4(asuint(w), gMisc[b + 3u].w);
    gAcc[acc + j].store(0);
}

// Every lane: its nodes against the proxy shapes, each inside counted on its body.
void cplCount(uint v, uint lane)
{
    cplCountBy(v, lane, LANES);
}
// The same over `lanes` lanes.
void cplCountBy(uint v, uint lane, uint lanes)
{
    uint4 hdr = gMisc[cplBase(v)];
    if (hdr.y == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    float3 origin = asfloat(gMisc[v].xyz);
    uint acc = cplAcc(v);
    bool cull = cplCullOn();
    uint2 gr = cplGsRange(v);
    uint2 gn = cplGsNodes(v);
    for (uint i = lane, iStop = EB_END(i, r0.y, lanes); i < iStop; i += lanes)
    {
        uint gi = r0.x + i;
        uint range = CPL_NONE;
        if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) == 0u)
        {
            float3 x = origin + gNodes[gi * 3u].xyz;
            for (uint k = 0u, kStop = EB_END(k, hdr.y, 1u); k < kStop; k++)
            {
                if (cplMay(v, k, x, cull, gr) && cplFace(v, k, x).w != 0.0)
                {
                    gAcc[acc + gMisc[cplShape(v, k)].y].add(1);
                    range = range == CPL_NONE ? (k | (k << 16)) : ((range & 0xFFFFu) | (k << 16));
                }
            }
        }
        if (i < gn.y) gsCpl[CPL_GS_NODE + gn.x + i] = range;
    }
}

// The coupling's part of a step of vehicle v, at the step's start (uniform `on`).
void coupleStep(uint v, uint lane, uint s, uint on)
{
    if (on == 0u) return;
#ifdef EB_ABLATE
    // Measurement only: 8192 the bodies' integration, 16384 the spheres, 32768 the count.
    if (s > 0u && (gP.ablate & 8192u) == 0u)
    {
        cplReactBy(v, lane, LANES);
        sync();
        cplIntegrate(v, lane);
    }
    sync();
    if ((gP.ablate & 16384u) == 0u) cplSpheres(v, lane);
    sync();
    if ((gP.ablate & 32768u) == 0u) cplCount(v, lane);
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
    sync();
    cplCount(v, lane);
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
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplReactBy(vvMember(off, m), lane, LANES);
        sync();
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
    }
    sync();
    if ((gP.ablate & 16384u) == 0u)
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplSpheres(vvMember(off, m), lane);
    sync();
    if ((gP.ablate & 32768u) == 0u)
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplCount(vvMember(off, m), lane);
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
        if (on) cplSpheresBy(v, pl, P);
        sync();
        if (on) cplCountBy(v, pl, P);
        sync();
        return;
    }
    if (s > 0u)
    {
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
    }
    sync();
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplSpheres(vvMember(off, m), lane);
    sync();
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplCount(vvMember(off, m), lane);
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
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
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
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), surfaceV(groundSurface(), vehicle), ground - x.y, V.w);
            cfAdd(gi, c);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
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
        float3 fr = (gP.ablate & 65536u) != 0u ? f : cplContacts(vehicle, local, x, f, vel, mass, V.w);
#else
        float3 fr = cplContacts(vehicle, local, x, f, vel, mass, V.w);
#endif
        cfAdd(gi, fr - f);
        f = fr;
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
        pos = sadd3(pos, shift);
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
    return gsCpl[0] != 0u && gMisc[coupleRec()].x != 0u && gMisc[cplBase(v)].y != 0u;
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
