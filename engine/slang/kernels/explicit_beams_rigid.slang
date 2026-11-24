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
// Header word i (CPL_GS_HDR order): always the workgroup copy coupleBegin made from
// the buffer at the dispatch's start (the header is constant through a frame, so the
// copy is the buffer's value). The header's .x bit 2 (the workgroup copies off, tests)
// no longer reads the buffer here: a branch at each of the ~600 inlined uses made the
// drivers' compile ~15% longer (engine/docs/measurements/kernel-compile-time.md).
// (EB_CPL_DIRECT, a kernel a thread an element with no barrier, the surface pass's
// spread pipelines: the words from the buffer itself; the same values.)
uint cplHdr(uint i)
{
#ifdef EB_CPL_DIRECT
    uint4 h = gMisc[coupleRec()];
    uint4 h1 = gMisc[coupleRec() + 1u];
    return i == 0u ? h.y : (i == 1u ? h.z : (i == 2u ? h.w : (i == 3u ? h1.x : h1.y)));
#else
    return gsCpl[CPL_GS_HDR + i];
#endif
}
// The host allows at most CPL_MAX_SHAPES shapes and no more bodies per vehicle
// (explicit_plan CouplingLayout::new), so these bounds never change a valid header;
// they keep a bad one (an unset copy, a corrupt record) from driving a loop over
// millions of records: the loops fail loudly (the loop-cap flag) and stay short.
static const uint CPL_MAX_SHAPES = 1024u;
uint cplB() { return min(cplHdr(1u), CPL_MAX_SHAPES); }
uint cplS() { return min(cplHdr(2u) & 0xFFFFu, CPL_MAX_SHAPES); }
uint cplPlanes() { return cplHdr(2u) >> 16; }
uint cplRecs() { return 1u + cplB() * CPL_BODY_RECS + cplS() * CPL_SHAPE_RECS + cplPlanes(); }
uint cplBase(uint v) { return cplHdr(0u) + v * cplRecs(); }
uint cplBody(uint v, uint j) { return cplBase(v) + 1u + j * CPL_BODY_RECS; }
uint cplShape(uint v, uint k) { return cplBase(v) + 1u + cplB() * CPL_BODY_RECS + k * CPL_SHAPE_RECS; }
uint cplPlane(uint v, uint i) { return cplBase(v) + 1u + cplB() * CPL_BODY_RECS + cplS() * CPL_SHAPE_RECS + i; }
uint cplAcc(uint v) { return cplHdr(3u) + v * (cplB() + 6u * cplS()); }
// Vehicle v's record header (bodies, shapes, ...) with its counts held to the
// region's B and S: a count past them is a bad record (raised as a loop cap, the
// frame an error), never a loop past the vehicle's records.
uint4 cplVeh(uint v)
{
    uint4 h = gMisc[cplBase(v)];
    uint nb = cplB();
    uint ns = cplS();
    if (h.x > nb || h.y > ns)
    {
        ebLoopCapHit(447);
        h.x = min(h.x, nb);
        h.y = min(h.y, ns);
    }
    return h;
}

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

// --- One code shape per arithmetic (bit-identical paths on every device) ---
// The unfolded step (cplReactBy, cplIntegrateBody, cplIntegrateBodySplit), the folded
// step (cplBodyStepped, cplStepFolded's phase C) and a contact group's members compute
// the same window sums, torque terms and integration. Written out once per path, a
// device's compiler may contract each copy differently (a fused multiply-add in one, a
// rounded product in the other: NVIDIA's Vulkan compiler, not Metal's), and paths the
// tests compare bit for bit then differ in the last bit. Every path calls these.

// Shape record `at`'s window sums (its impulse, largest force and moment) with the
// step's reaction F, M added.
void cplWindowAdd(uint at, float3 F, float3 M)
{
    float4 imp = asfloat(gMisc[at + 4u]);
    float4 mom = asfloat(gMisc[at + 5u]);
    gMisc[at + 4u] = asuint(float4(imp.xyz + F * STEP, max(imp.w, length(F))));
    gMisc[at + 5u] = asuint(float4(mom.xyz + M * STEP, 0.0));
}

// A shape's torque term about its body's centre c (reference point ref).
float3 cplTorqueTerm(float3 ref, float3 c, float3 F, float3 M)
{
    return M + cross(ref - c, F);
}

// A body's symplectic Euler step over its summed force and torque: c0 its centre (.w
// 1/M), q its rotation, c2 its velocity, w (in: c3's angular velocity) its angular
// velocity, c4 its inverse inertia.
void cplIntegrateStep(inout float4 c0, inout float4 q, inout float4 c2, inout float3 w, float4 c4, float3 force, float3 torque)
{
    float3 vel = c2.xyz;
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
    c0 = float4(pos, c0.w);
    c2 = float4(vel, c2.w);
}

// A proxy shape's rotation in the vehicle frame: its body's, then its own.
float4 cplShapeRot(float4 q, float4 lq)
{
    return normalize(cplQmul(q, lq));
}

// --- The shapes' step frames (lossless; with the shape cull) ---
// cplFace read a shape's record, then its body's, and turned its position and rotation
// for every node it tested, the same each time within a step (the bodies move only in
// the step's integration). The spheres' pass (cplSpheresBy, a lane a shape, after the
// integration) computes them once with the same expressions and writes them beside
// the shape's sphere (CPL_SPHERE_WORDS words of gNext a shape): centre (the sphere's
// words 0..2), rotation (4..7), the shape's p1, p2, cull radius (8..11, its record 3),
// p0 (12), kind (13), planes (14). With the cull on, cplFace reads them (cplFaceFrame).
static const uint CPL_SPHERE_WORDS = 16u;
uint cplSphere(uint v, uint k);
float4 cplFaceFrame(uint v, uint k, float3 x);
bool cplCullOn();
void cplFrameStore(uint s, uint4 h, float4 lp, float4 pr, float4 r)
{
    gNext[s + 4u] = asuint(r.x);
    gNext[s + 5u] = asuint(r.y);
    gNext[s + 6u] = asuint(r.z);
    gNext[s + 7u] = asuint(r.w);
    gNext[s + 8u] = asuint(pr.x);
    gNext[s + 9u] = asuint(pr.y);
    gNext[s + 10u] = asuint(pr.z);
    gNext[s + 11u] = asuint(pr.w);
    gNext[s + 12u] = asuint(lp.w);
    gNext[s + 13u] = h.x;
    gNext[s + 14u] = h.w;
}

// The outward normal (.w 1) of the proxy shape's surface nearest x, for x strictly
// inside the shape, else .w 0 (the box rule; CPU `ProxySet::face`).
// cplFace past the cull: x's offset d from the shape's centre, its rotation r, kind
// and planes h (.x, .w), p0 (lp.w) and p1, p2 (pr.x, pr.y).
float4 cplFaceLocal(uint v, uint4 h, float4 lp, float4 pr, float4 r, float3 d)
{
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
    for (uint i = 0u, iStop = EB_END(448, i, count, 1u); i < iStop; i++)
    {
        float4 pl = asfloat(gMisc[cplPlane(v, off + i)]);
        float s = dot(pl.xyz, l) - pl.w;
        if (!(s < 0.0)) return float4(0.0, 0.0, 0.0, 0.0);
        if (s > best) { best = s; n = pl.xyz; }
    }
    return float4(cplRot(r, n), 1.0);
}

float4 cplFace(uint v, uint k, float3 x)
{
    if (cplCullOn()) return cplFaceFrame(v, k, x);
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
    float4 r = cplShapeRot(q, lq);
    return cplFaceLocal(v, h, lp, pr, r, d);
}

// cplFace from the step frames (cplFrameStore): the same values, read, not recomputed.
float4 cplFaceFrame(uint v, uint k, float3 x)
{
    uint s = cplSphere(v, k);
    float3 p = asfloat(uint3(gNext[s], gNext[s + 1u], gNext[s + 2u]));
    float4 pr = asfloat(uint4(gNext[s + 8u], gNext[s + 9u], gNext[s + 10u], gNext[s + 11u]));
    float3 d = x - p;
    // Cull: the shape's bounding sphere.
    if (dot(d, d) >= pr.z * pr.z) return float4(0.0, 0.0, 0.0, 0.0);
    float4 r = asfloat(uint4(gNext[s + 4u], gNext[s + 5u], gNext[s + 6u], gNext[s + 7u]));
    uint4 h = uint4(gNext[s + 13u], 0u, 0u, gNext[s + 14u]);
    float4 lp = float4(0.0, 0.0, 0.0, asfloat(gNext[s + 12u]));
    return cplFaceLocal(v, h, lp, pr, r, d);
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
uint cplSphere(uint v, uint k) { return cplHdr(4u) + (v * cplS() + k) * CPL_SPHERE_WORDS; }
#ifdef EB_CPL_DIRECT
bool cplCullOn() { return (gMisc[coupleRec()].x & 2u) == 0u; }
#else
bool cplCullOn() { return (gsCpl[0] & 2u) == 0u; }
#endif

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
    uint shapes = gsCpl[0] != 0u ? cplVeh(v).y : 0u;
    uint nodes = gStatic[v * 4u].y;
    gsCpl[CPL_GS_TAB + 5u * e + 0u] = v;
    gsCpl[CPL_GS_TAB + 5u * e + 1u] = base;
    gsCpl[CPL_GS_TAB + 5u * e + 2u] = base < CPL_GS ? min(shapes, CPL_GS - base) : 0u;
    gsCpl[CPL_GS_TAB + 5u * e + 3u] = nodeBase;
    // The header's .x bit 64 (a host setting, browsers' WebGPU): no node ranges, every
    // node's contacts scan its vehicle's shapes from the first (the same shapes pass the
    // cull and the face test, in the same order: the same sums). The range lookup in
    // cplContacts runs hundreds of times slower as Chrome compiles it for Metal (Tint's
    // bounds-checked form).
    bool ranges = (gsCpl[0] & 64u) == 0u;
    gsCpl[CPL_GS_TAB + 5u * e + 4u] = ranges && shapes > 0u && nodeBase < CPL_GS_NODES ? min(nodes, CPL_GS_NODES - nodeBase) : 0u;
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
    uint shapes = cplCullOn() ? cplVeh(v).y : 0u;
    uint2 gr = cplGsRange(v);
    for (uint k = lane, kStop = EB_END(449, k, shapes, lanes); k < kStop; k += lanes)
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
        cplFrameStore(s, h, lp, pr, cplShapeRot(q, asfloat(gMisc[at + 2u])));
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

// The two-body contact on a node of vehicle v at x (vehicle-local: relative to its
// origin, as every proxy record is) (force f so far, velocity vel,
// mass, friction) from every proxy shape it is inside, in shape order; the reactions are
// summed into the shapes' accumulators. Returns the node's force with them added.
// `local`: the node's index in its vehicle (its count pass range, cplGsNodes).
bool surfProxyFast(uint v, uint k, out float3 vb);

bool cplListOn();

// Vehicle v's static shape rule against an infinite-mass proxy takes its own surface
// (jbeam-semantics-spec.md §37.1: on the BeamNG profile the node pass's ground, boxes and
// shapes are dissipative and their friction law exact, and a World's static body is the
// same shape): true for a vehicle with either flag. The RoR profile keeps the plain
// boxSurface() path, bit for bit (its own function: Metal's codegen of it stays).
bool cplStaticOwnSurface(uint v)
{
    return (gStatic[v * 4u + 3u].x & (FLAG_DISSIPATIVE | FLAG_EXACT_FRICTION)) != 0u;
}
float3 cplStaticContactV(float3 f, float3 u, float mass, float3 n, uint v, float friction)
{
    return contactForce(f, u, mass, n, surfaceV(boxSurface(), v), 0.0, friction);
}
float3 cplContacts(uint v, uint local, float3 x, float3 f, float3 vel, float mass, float friction)
{
    // With the candidate list (cplList*, lossless) the node's listed shapes (the count pass
    // this step left it valid; every shape when it keeps none), else every shape of its
    // range (culled by the spheres when on): one loop, one copy of the contact for both
    // (the compilers inline every call).
    bool listed = cplListOn();
    // §38.4 (explicit_beams_surface.slang FLAG_SURFACE): a fast proxy pushes the nodes
    // inside it out along its motion relative to them (CPU ProxySet::node_contact_with).
#ifdef EB_NO_SURFACE
    bool fastSide = false; // gP.surfSplit is 0 in this module's uploads
#else
    bool fastSide = (gStatic[v * 4u + 3u].x & 131072u) != 0u && gP.surfSplit != 0u;
#endif
    uint4 hdr = cplVeh(v);
    uint acc = cplAcc(v);
    uint nb = cplB();
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
    bool cull = listed || cplCullOn();
    uint nl = cplListNode(gStatic[v * 4u].x + local);
    bool every = !listed || (gNext[nl + 4u] & 0xFFFFu) == CPL_LIST_ALL;
    uint count = every ? k1 : CPL_LIST_CAP;
    for (uint e = every ? k0 : 0u, eStop = EB_END(464, e, count, 1u); e < eStop; e++)
    {
        uint k = every ? e : ((gNext[nl + 4u + (e >> 1)] >> (16u * (e & 1u))) & 0xFFFFu);
        if (k >= k1) break;
        if (k < k0) continue;
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
            if (cplStaticOwnSurface(v)) fc = cplStaticContactV(f, u, mass, n4.xyz, v, friction);
            else fc = contactForce(f, u, mass, n4.xyz, boxSurface(), 0.0, friction);
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
    uint4 hdr = cplVeh(v);
    for (uint j = lane, jStop = EB_END(451, j, hdr.x, LANES); j < jStop; j += LANES) cplIntegrateBody(v, j);
}
// The same bodies (every j < B), body j by lane j of `lanes`, then j + lanes, ...
void cplIntegrateBy(uint v, uint lane, uint lanes)
{
    uint4 hdr = cplVeh(v);
    for (uint j = lane, jStop = EB_END(452, j, hdr.x, lanes); j < jStop; j += lanes) cplIntegrateBody(v, j);
}
// Whether vehicle v's step sums go through workgroup memory (cplReactBy): its every
// sphere is there (so its shapes fit, CPL_REACT_WORDS words each, in the spheres' and
// the nodes' ranges' words, free between the node phase and the spheres' pass).
// Per shape: its force, its torque term, its body slot.
static const uint CPL_REACT_WORDS = 7u;
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
    uint4 hdr = cplVeh(v);
    if (!cplSplit(v, hdr.y)) return;
    uint2 gr = cplGsRange(v);
    uint bodies = min(hdr.x, LANES);
    uint acc = cplAcc(v);
    uint nb = cplB();
    uint s0 = cplShape(v, 0u);
    for (uint k = lane, kStop = EB_END(453, k, hdr.y, lanes); k < kStop; k += lanes)
    {
        uint at = s0 + k * CPL_SHAPE_RECS;
        uint j = gMisc[at].y;
        // The shape's body slot beside its terms, for every shape: the bodies' sums
        // (cplIntegrateBodySplit) pick their shapes from workgroup memory.
        gsCpl[CPL_GS_SPH + CPL_REACT_WORDS * (gr.x + k) + 6u] = j;
        if (j >= bodies) continue;
        float3 c = asfloat(gMisc[cplBody(v, j)].xyz);
        uint s = acc + nb + 6u * k;
        float3 F = float3(float(gAcc[s].load()), float(gAcc[s + 1u].load()), float(gAcc[s + 2u].load())) * (1.0 / FIXED);
        float3 M = float3(float(gAcc[s + 3u].load()), float(gAcc[s + 4u].load()), float(gAcc[s + 5u].load())) * (1.0 / FIXED);
        for (uint w = 0u; w < 6u; w++) gAcc[s + w].store(0);
        float3 ref = asfloat(gMisc[at + 6u].xyz);
        cplWindowAdd(at, F, M);
        float3 t = cplTorqueTerm(ref, c, F, M);
        uint g = CPL_GS_SPH + CPL_REACT_WORDS * (gr.x + k);
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
    if (cplSplit(v, cplVeh(v).y))
    {
        cplIntegrateBodySplit(v, j);
        return;
    }
    uint4 hdr = cplVeh(v);
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
    for (uint k = k0, kStop = EB_END(454, k, k1, 1u); k < kStop; k++)
    {
        uint at = s0 + k * CPL_SHAPE_RECS;
        if (gMisc[at].y != j) continue;
        uint s = acc + nb + 6u * k;
        float3 F = float3(float(gAcc[s].load()), float(gAcc[s + 1u].load()), float(gAcc[s + 2u].load())) * (1.0 / FIXED);
        float3 M = float3(float(gAcc[s + 3u].load()), float(gAcc[s + 4u].load()), float(gAcc[s + 5u].load())) * (1.0 / FIXED);
        for (uint w = 0u; w < 6u; w++) gAcc[s + w].store(0);
        float3 ref = asfloat(gMisc[at + 6u].xyz);
        cplWindowAdd(at, F, M);
        force += F;
        torque += cplTorqueTerm(ref, c0.xyz, F, M);
    }
    float3 w = c3.xyz;
    cplIntegrateStep(c0, q, c2, w, c4, force, torque);
    gMisc[b] = asuint(c0);
    gMisc[b + 1u] = asuint(q);
    gMisc[b + 2u] = asuint(c2);
    gMisc[b + 3u] = uint4(asuint(w), gMisc[b + 3u].w);
    gAcc[acc + j].store(0);
}

void cplIntegrateBodySplit(uint v, uint j)
{
    uint4 hdr = cplVeh(v);
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
    // The shapes' terms from cplReactBy, summed in shape order; each shape's body slot
    // from workgroup memory (cplReactBy's copy of its record's .y).
    uint gx = cplGsRange(v).x;
    for (uint k = k0, kStop = EB_END(455, k, k1, 1u); k < kStop; k++)
    {
        uint g = CPL_GS_SPH + CPL_REACT_WORDS * (gx + k);
        if (gsCpl[g + 6u] != j) continue;
        force += asfloat(uint3(gsCpl[g], gsCpl[g + 1u], gsCpl[g + 2u]));
        torque += asfloat(uint3(gsCpl[g + 3u], gsCpl[g + 4u], gsCpl[g + 5u]));
    }
    float3 w = c3.xyz;
    cplIntegrateStep(c0, q, c2, w, c4, force, torque);
    gMisc[b] = asuint(c0);
    gMisc[b + 1u] = asuint(q);
    gMisc[b + 2u] = asuint(c2);
    gMisc[b + 3u] = uint4(asuint(w), gMisc[b + 3u].w);
    gAcc[acc + j].store(0);
}

// The origin rebase moved vehicle v's origin by `first`: its proxy records (bodies'
// centres, shapes' reference points: relative to the origin) move back by it. Every
// lane, inside the rebase's branch (no barrier; the rebase's own follows).
void cplRebase(uint v, uint lane, float3 first)
{
    // The header from the buffer: the rebase runs before coupleBegin copies it to
    // workgroup memory (cplHdr's copies are not yet written here).
    uint4 h = gMisc[coupleRec()];
    if (h.x == 0u) return;
    uint nb = h.z;
    uint ns = h.w & 0xFFFFu;
    uint recs = 1u + nb * CPL_BODY_RECS + ns * CPL_SHAPE_RECS + (h.w >> 16);
    uint base = h.y + v * recs;
    uint4 hdr = gMisc[base];
    for (uint j = lane, jStop = EB_END(456, j, min(hdr.x, nb), LANES); j < jStop; j += LANES)
    {
        uint b = base + 1u + j * CPL_BODY_RECS;
        float4 c = asfloat(gMisc[b]);
        gMisc[b] = asuint(float4(c.xyz - first, c.w));
    }
    for (uint k = lane, kStop = EB_END(457, k, min(hdr.y, ns), LANES); k < kStop; k += LANES)
    {
        uint at = base + 1u + nb * CPL_BODY_RECS + k * CPL_SHAPE_RECS + 6u;
        float4 r = asfloat(gMisc[at]);
        gMisc[at] = asuint(float4(r.xyz - first, r.w));
    }
}

#include "explicit_beams_rigid_count.slang"
