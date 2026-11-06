// Surface contact (jbeam-semantics-spec.md §38; CPU reference
// engine_softbody::explicit_surface): a vehicle's collision triangles, beams and nodes
// against its static box slots (axis-aligned boxes, oriented boxes, cylinders,
// capsules, spheres) and, with the World's coupling, its proxy shapes (boxes, spheres,
// capsules, hulls of static and dynamic bodies) by their surface, beside node contact.
// Included by explicit_beams.slang. The geometry is the shared strict closest-feature
// module (slang/physics/feature_query.slang, its Rust twin engine_schema::feature_query);
// the response (the §37 law, mass splitting, the fixed-point sums) is here.
//
// Off (no vehicle of the upload has FLAG_SURFACE: gP.surfSplit 0, a uniform) nothing
// here runs. On, per step after self-collision and before the slide nodes, three
// phases with a barrier before each (every lane reaches them; a vehicle without the
// flag does no work between them). No workgroup memory (Metal's threadgroup slots
// are spent by the contact modules):
//   snap   per node its force at the pass's start (beam sum + accumulator) into gBeams
//          at surfSnap, its hit count at surfSplit cleared; per proxy body its hit
//          count cleared
//   find   per collision triangle, segment and node the narrow phase against every
//          slot, then every proxy shape, near it; each hit counts its nodes (weight
//          > 0) at surfSplit and its proxy body
//   apply  the same hits recomputed (positions do not change), the two-body
//          dissipative response from the snapshot and the counts, into the
//          accumulators (fixed point, order free), the position steps and a proxy
//          body's step sums (integrated with the node pass's at the next step's start)
// Slots are numbered boxes and shapes first, then the proxy shapes (the CPU's
// surface_shapes order). Every operation strict, in the CPU reference's order.

#include "../physics/feature_query.slang"

static const uint FLAG_SURFACE = 131072u;

// The shared queries' hull records here: a proxy hull's (explicit_beams_rigid.slang
// coupling region) planes, then its vertices, then its edges four to a record
// (vehicle_coupling.slang cpl_plan, engine_exec explicit_plan::coupling). s.hull is the
// gMisc record of its first plane.
float4 fqHullPlane(FqShape s, uint f) { return asfloat(gMisc[s.hull + f]); }
float3 fqHullVertex(FqShape s, uint i) { return asfloat(gMisc[s.hull + s.nf + i].xyz); }
uint2 fqHullEdge(FqShape s, uint e)
{
    uint4 r = gMisc[s.hull + s.nf + s.nv + e / 4u];
    uint m = e % 4u;
    uint x = m == 0u ? r.x : (m == 1u ? r.y : (m == 2u ? r.z : r.w));
    return uint2(x & 255u, (x >> 8) & 255u);
}

// Box slot b of a vehicle (boxBase records: (lo, kind), (hi, at)) as a shape, and its
// cull box.
FqShape surfSlot(uint boxOff, uint b, out float3 lo, out float3 hi)
{
    uint4 L = gStatic[gP.boxBase + (boxOff + b) * 2u];
    uint4 H = gStatic[gP.boxBase + (boxOff + b) * 2u + 1u];
    lo = asfloat(L.xyz);
    hi = asfloat(H.xyz);
    FqShape s;
    s.kind = FQ_BOX;
    s.c = sscale(sadd3(lo, hi), 0.5);
    s.b = float3(0.0, 0.0, 0.0);
    s.ax0 = float3(1.0, 0.0, 0.0);
    s.ax1 = float3(0.0, 1.0, 0.0);
    s.ax2 = float3(0.0, 0.0, 1.0);
    s.half = sscale(ssub3(hi, lo), 0.5);
    s.hull = 0u;
    s.nv = 0u;
    s.nf = 0u;
    s.ne = 0u;
    s.moved = 0u;
    s.off = float3(0.0, 0.0, 0.0);
    if (L.w == 0u) return s;
    uint at = gP.boxBase + H.w;
    float4 p0 = asfloat(gStatic[at]);
    float4 p1 = asfloat(gStatic[at + 1u]);
    if (L.w == SHAPE_OBOX)
    {
        float4 p2 = asfloat(gStatic[at + 2u]);
        float4 p3 = asfloat(gStatic[at + 3u]);
        s.c = p0.xyz;
        s.ax0 = p1.xyz;
        s.ax1 = p2.xyz;
        s.ax2 = p3.xyz;
        s.half = float3(p0.w, p1.w, p2.w);
    }
    else if (L.w == SHAPE_CYLINDER)
    {
        s.kind = FQ_CYL;
        s.c = p0.xyz;
        s.half = float3(p0.w, p1.x, 0.0);
    }
    else if (L.w == SHAPE_CAPSULE)
    {
        s.kind = FQ_CAP;
        s.c = p0.xyz;
        s.b = p1.xyz;
        s.half = float3(p0.w, 0.0, 0.0);
    }
    else
    {
        s.kind = FQ_SPH;
        s.c = p0.xyz;
        s.half = float3(p0.w, 0.0, 0.0);
    }
    return s;
}

// --- Proxy shapes (the World's coupling; CPU explicit_rigid::surface_shape) ---

// gAcc of vehicle v's per-proxy-body surface hit counts (0: no surface proxies).
uint surfBodyBase() { return gMisc[coupleRec() + 1u].z; }

// Vehicle v's proxy shapes the surface pass meets this window (0 without the coupling's
// surface room, the coupling off or no proxies).
uint surfProxies(uint v)
{
    if (surfBodyBase() == 0u || !cplHas(v)) return 0u;
    return gMisc[cplBase(v)].y;
}

// Proxy shape k of vehicle v at its body's pose now, its body slot j: the strict pose
// (fqPose), a box's axes, a capsule's ends along its local y, a hull's records.
FqShape surfProxyShape(uint v, uint k, out uint j)
{
    uint at = cplShape(v, k);
    uint4 h = gMisc[at];
    j = h.y;
    uint b = cplBody(v, j);
    float3 c = asfloat(gMisc[b].xyz);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 lp = asfloat(gMisc[at + 1u]);
    float4 lq = asfloat(gMisc[at + 2u]);
    uint4 pw = gMisc[at + 3u];
    float4 pr = asfloat(pw);
    float4 r;
    float3 p = fqPose(c, q, lp.xyz, lq, r);
    FqShape s;
    s.kind = FQ_BOX;
    s.c = p;
    s.b = float3(0.0, 0.0, 0.0);
    s.ax0 = fqRot(r, float3(1.0, 0.0, 0.0));
    s.ax1 = fqRot(r, float3(0.0, 1.0, 0.0));
    s.ax2 = fqRot(r, float3(0.0, 0.0, 1.0));
    s.half = float3(lp.w, pr.x, pr.y);
    s.hull = 0u;
    s.nv = 0u;
    s.nf = 0u;
    s.ne = 0u;
    s.moved = 0u;
    s.off = float3(0.0, 0.0, 0.0);
    if (h.x == CPL_SPHERE)
    {
        s.kind = FQ_SPH;
        s.half = float3(lp.w, 0.0, 0.0);
    }
    else if (h.x == CPL_CAPSULE)
    {
        s.kind = FQ_CAP;
        float3 e = sscale(s.ax1, pr.x);
        s.c = ssub3(p, e);
        s.b = sadd3(p, e);
        s.half = float3(lp.w, 0.0, 0.0);
    }
    else if (h.x == CPL_HULL)
    {
        s.kind = FQ_HULL;
        s.hull = cplPlane(v, h.w & 0xFFFFu);
        s.nf = h.w >> 16;
        s.nv = pw.w & 255u;
        s.ne = (pw.w >> 8) & 255u;
    }
    return s;
}

// Fast proxy bodies (§38.4; CPU ProxySet::cast_velocities): whether shape k's body moves
// more than half the shape's least extent in a step, (|v| + |ω| R) h > e / 2 (R its
// bound radius, e a box's least half extent, a sphere's or capsule's radius, a hull's
// least plane offset, at least 1 cm); its velocity then in vb.
bool surfProxyFast(uint v, uint k, out float3 vb)
{
    vb = float3(0.0, 0.0, 0.0);
    // The coupling header's bit 8: casts off (measurement only).
    if ((gMisc[coupleRec()].x & 8u) != 0u) return false;
    uint at = cplShape(v, k);
    uint4 h = gMisc[at];
    uint b = cplBody(v, h.y);
    float4 lp = asfloat(gMisc[at + 1u]);
    float4 pr = asfloat(gMisc[at + 3u]);
    vb = asfloat(gMisc[b + 2u].xyz);
    float3 w = asfloat(gMisc[b + 3u].xyz);
    float e = lp.w;
    if (h.x == CPL_BOX) e = min(min(lp.w, pr.x), pr.y);
    else if (h.x == CPL_HULL)
    {
        uint base = cplPlane(v, h.w & 0xFFFFu);
        uint count = h.w >> 16;
        float m = 3.0e38;
        for (uint f = 0u; f < FQ_HULL_FACES; f++)
        {
            if (f >= count) break;
            m = min(m, asfloat(gMisc[base + f].w));
        }
        e = max(m, 0.01);
    }
    float travel = nf(smul(fqLen(vb), STEP) + smul(smul(fqLen(w), pr.z), STEP));
    return travel > smul(0.5, e);
}

// A fast shape's cast (CPU explicit_surface::cast_toi) against element x0.. (kind 0 a
// triangle, 1 a segment, 2 a point) over the relative motion d: in the swept cull box,
// the earliest fraction t > 0 at which it comes within the band; −1 when not met (or
// met at the start: the discrete contact's).
float surfCastToi(FqShape s, float3 x0, float3 x1, float3 x2, uint kind, float3 d, float r)
{
    float3 lo;
    float3 hi;
    fqBounds(s, lo, hi);
    float3 slo = sadd3(lo, min(d, float3(0.0, 0.0, 0.0)));
    float3 shi = sadd3(hi, max(d, float3(0.0, 0.0, 0.0)));
    float3 elo = kind == 0u ? min(min(x0, x1), x2) : (kind == 1u ? min(x0, x1) : x0);
    float3 ehi = kind == 0u ? max(max(x0, x1), x2) : (kind == 1u ? max(x0, x1) : x0);
    elo = ssub3(elo, float3(r, r, r));
    ehi = sadd3(ehi, float3(r, r, r));
    if (!(all(elo <= shi) && all(ehi >= slo))) return -1.0;
    float t = fqToi(s, x0, x1, x2, kind, d, r);
    return t > 0.0 ? t : -1.0;
}

// A fast body's earliest time of impact this step (CPU Casts::tmin), by step parity: gAcc
// after the hit counts, 0x7FFFFFFF − t's bits (atomic max: the earliest; 0 none). This
// step's is gathered in the snapshot phase (surfGather*), the other cleared there for the
// next step.
uint surfTminAt(uint v, uint j, uint parity)
{
    return surfBodyBase() + cplB() * (gP.vehicles * (1u + (parity & 1u)) + v) + j;
}

void surfGatherAt(uint v, uint j, uint parity, float t)
{
    if (t > 0.0) gAcc[surfTminAt(v, j, parity)].max(int(0x7FFFFFFFu - asuint(t)));
}

// The cast's contact (CPU Casts::hit, its second pass): only an element met at the
// body's earliest time of impact (within FQ_TIE along its motion), at its own, the shape
// moved there in sm.
bool surfCastUse(uint v, uint j, uint parity, FqShape s, float3 x0, float3 x1, float3 x2, uint kind, float3 d, float r,
                 out FqShape sm, out float3 shift)
{
    sm = s;
    shift = float3(0.0, 0.0, 0.0);
    float t = surfCastToi(s, x0, x1, x2, kind, d, r);
    if (!(t > 0.0)) return false;
    uint e = uint(gAcc[surfTminAt(v, j, parity)].load());
    float tmin = asfloat(0x7FFFFFFFu - e);
    if (!(t <= nf(tmin + sdiv(FQ_TIE, fqLen(d))))) return false;
    shift = sscale(d, t);
    sm = fqTranslate(s, shift);
    return true;
}

// Whether element box (tlo, thi) may meet proxy shape k: its grown sphere (cplSpheres,
// written each step from the same pose) holds the shape, so outside it the shape is
// farther than the band and its narrow phase would find nothing (a cull, lossless).
bool surfProxyMay(uint v, uint k, float3 tlo, float3 thi)
{
    if (!cplCullOn()) return true;
    uint sp = cplSphere(v, k);
    float3 c = asfloat(uint3(gNext[sp], gNext[sp + 1u], gNext[sp + 2u]));
    float3 d = c - clamp(c, tlo, thi);
    return dot(d, d) < asfloat(gNext[sp + 3u]);
}

float surfInvMass(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return 0.0;
    return sdiv(float(uint(gAcc[gP.surfSplit + x].load())), gNodes[x * 3u].w);
}

float3 surfAcc(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return float3(0.0, 0.0, 0.0);
    return sdiv3(gBeams[gP.surfSnap + x].xyz, gNodes[x * 3u].w);
}

float surfTerm(float w, float im) { return w > 0.0 ? smul(smul(w, w), im) : 0.0; }

// apply_surface_hits for one hit on element tn (global nodes) of vehicle v, against a
// static slot (k NONE) or proxy shape k of body slot j (q: the contact point).
void surfApplyTo(uint v, uint4 tn, float3 w, float3 nn, float gap, float r, uint k, uint j, float3 q)
{
    float im0 = surfInvMass(tn.x);
    float im1 = surfInvMass(tn.y);
    float im2 = surfInvMass(tn.z);
    float imt = nf(nf(surfTerm(w.x, im0) + surfTerm(w.y, im1)) + surfTerm(w.z, im2));
    float3 vt = sadd3(sadd3(sscale(gNodes[tn.x * 3u + 1u].xyz, w.x), sscale(gNodes[tn.y * 3u + 1u].xyz, w.y)),
                      sscale(gNodes[tn.z * 3u + 1u].xyz, w.z));
    float3 at = sadd3(sadd3(sscale(surfAcc(tn.x), w.x), sscale(surfAcc(tn.y), w.y)), sscale(surfAcc(tn.z), w.z));
    // The shape: static (infinite mass, at rest), or a proxy body (its velocity at the
    // point; finite mass split over its surface hits this step and its vehicles).
    float ims = 0.0;
    float3 vs = float3(0.0, 0.0, 0.0);
    if (k != 0xFFFFFFFFu)
    {
        uint b = cplBody(v, j);
        float4 c0 = asfloat(gMisc[b]);
        float4 bq = asfloat(gMisc[b + 1u]);
        float4 c2 = asfloat(gMisc[b + 2u]);
        float4 c3 = asfloat(gMisc[b + 3u]);
        float3 arm = ssub3(q, c0.xyz);
        vs = sadd3(c2.xyz, scross(c3.xyz, arm));
        if (c0.w != 0.0)
        {
            float3 ii = asfloat(gMisc[b + 4u].xyz);
            float3 rn = scross(arm, nn);
            float3 l = fqRot(float4(-bq.xyz, bq.w), rn);
            float3 irn = fqRot(bq, float3(smul(ii.x, l.x), smul(ii.y, l.y), smul(ii.z, l.z)));
            float hits = float(max(uint(gAcc[surfBodyBase() + v * cplB() + j].load()), 1u));
            float split = smul(hits, max(c2.w, 1.0));
            ims = smul(split, nf(c0.w + sdot(rn, irn)));
        }
    }
    float inv = nf(ims + imt);
    float m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
    float3 fh = sscale(ssub3(float3(0.0, 0.0, 0.0), at), m);
    float fr = nf(nf(smul(gNodes[tn.x * 3u + 1u].w, w.x) + smul(gNodes[tn.y * 3u + 1u].w, w.y))
                  + smul(gNodes[tn.z * 3u + 1u].w, w.z));
    float p = min(nf(r - gap), r);
    Surface surf = surfaceV(boxSurface(), v);
    surf.dissipative = 1u;
    float budget;
    float3 fc = contactResponse(fh, ssub3(vs, vt), m, nn, surf, p, fr, budget);
    if (all(fc == float3(0.0, 0.0, 0.0))) return;
    float delta = shiftDepth(p, budget, sdot(fh, nn));
    if (delta > 0.0)
    {
        float qd = smul(delta, m);
        if (w.x > 0.0) shiftAdd(tn.x, sscale(nn, -smul(w.x, smul(qd, im0))));
        if (w.y > 0.0) shiftAdd(tn.y, sscale(nn, -smul(w.y, smul(qd, im1))));
        if (w.z > 0.0) shiftAdd(tn.z, sscale(nn, -smul(w.z, smul(qd, im2))));
    }
    accAdd(tn.x, -sscale(fc, w.x));
    accAdd(tn.y, -sscale(fc, w.y));
    accAdd(tn.z, -sscale(fc, w.z));
    cfAdd(tn.x, sscale(fc, w.x));
    cfAdd(tn.y, sscale(fc, w.y));
    cfAdd(tn.z, sscale(fc, w.z));
    if (k != 0xFFFFFFFFu)
    {
        // The shape's reaction +fc at q, into this step's sums (cplIntegrate's).
        float3 mo = scross(ssub3(q, asfloat(gMisc[cplShape(v, k) + 6u].xyz)), fc);
        uint s = cplAcc(v) + cplB() + 6u * k;
        gAcc[s].add(toFixed(fc.x));
        gAcc[s + 1u].add(toFixed(fc.y));
        gAcc[s + 2u].add(toFixed(fc.z));
        gAcc[s + 3u].add(toFixed(mo.x));
        gAcc[s + 4u].add(toFixed(mo.y));
        gAcc[s + 5u].add(toFixed(mo.z));
    }
}

void surfApply(uint v, uint4 tn, float3 w, float3 nn, float gap, float r)
{
    surfApplyTo(v, tn, w, nn, gap, r, 0xFFFFFFFFu, 0u, float3(0.0, 0.0, 0.0));
}

// The find's slot masks (lossless: the apply recomputes only the slots its element
// hit, the same hits in the same order): bit b for slot b < 31, bit 31 any slot
// from 31 on.
uint surfMarkBit(uint b) { return b < 31u ? (1u << b) : 0x80000000u; }

// An element's mask at gAcc index mk: the find stores it (0 when it skips the
// element), the apply reads it.
uint surfMaskIn(uint mk, bool apply) { return apply ? uint(gAcc[mk].load()) : 0u; }
void surfMaskOut(uint mk, bool apply, uint mask)
{
    if (!apply) gAcc[mk].store(int(mask));
}

// A find's hit on proxy body slot j: its count (mass splitting).
void surfBodyHit(uint v, uint j) { gAcc[surfBodyBase() + v * cplB() + j].add(1); }

// Triangle gt (of vehicle v) against every box slot (the static path, as before the
// proxies), then, when the vehicle has them, every proxy shape (surfTriangleProxies):
// apply 0 counts its hits, 1 applies them.
uint surfTriangleProxies(uint v, uint4 tn, float3 x0, float3 x1, float3 x2, float r, uint boxCount, uint proxies,
                         uint parity, bool apply, uint mask, uint found)
{
    float3 tlo = ssub3(min(min(x0, x1), x2), float3(r, r, r));
    float3 thi = sadd3(max(max(x0, x1), x2), float3(r, r, r));
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        uint b = boxCount + k;
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 vb;
        bool fast = surfProxyFast(v, k, vb);
        if (!fast && !surfProxyMay(v, k, tlo, thi)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 w;
        float3 nn;
        float g;
        float3 q;
        bool hit = fqNear(x0, x1, x2, r, lo, hi) && fqContact(s, x0, x1, x2, r, w, nn, g, q);
        float3 shift = float3(0.0, 0.0, 0.0);
        if (!hit && fast)
        {
            float3 ve = sdiv3(sadd3(sadd3(gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz), gNodes[tn.z * 3u + 1u].xyz), 3.0);
            FqShape sm;
            hit = surfCastUse(v, j, parity, s, x0, x1, x2, 0u, sscale(ssub3(vb, ve), STEP), r, sm, shift)
                  && fqContact(sm, x0, x1, x2, r, w, nn, g, q);
            if (!hit) shift = float3(0.0, 0.0, 0.0);
        }
        if (!hit) continue;
        if (!apply)
        {
            found |= surfMarkBit(b);
            if (w.x > 0.0) gAcc[gP.surfSplit + tn.x].add(1);
            if (w.y > 0.0) gAcc[gP.surfSplit + tn.y].add(1);
            if (w.z > 0.0) gAcc[gP.surfSplit + tn.z].add(1);
            surfBodyHit(v, j);
        }
        else
            surfApplyTo(v, tn, w, nn, g, r, k, j, ssub3(q, shift));
    }
    return found;
}

void surfTriangle(uint v, uint gt, float3 origin, float r, uint boxOff, uint boxCount, uint proxies, uint parity,
                  bool apply)
{
    uint mk = gP.surfMark + gt;
    uint mask = surfMaskIn(mk, apply);
    if ((gP.triBreak != 0u && triangleDead(gt)) || (gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u
        || (apply && mask == 0u))
    {
        surfMaskOut(mk, apply, 0u);
        return;
    }
    uint4 tn = gStatic[gP.triBase + gt];
    float3 x0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 x1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 x2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    uint found = 0u;
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
    {
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 lo;
        float3 hi;
        FqShape s = surfSlot(boxOff, b, lo, hi);
        if (!fqNear(x0, x1, x2, r, lo, hi)) continue;
        float3 w;
        float3 nn;
        float g;
        float3 q;
        if (!fqContact(s, x0, x1, x2, r, w, nn, g, q)) continue;
        if (!apply)
        {
            found |= surfMarkBit(b);
            if (w.x > 0.0) gAcc[gP.surfSplit + tn.x].add(1);
            if (w.y > 0.0) gAcc[gP.surfSplit + tn.y].add(1);
            if (w.z > 0.0) gAcc[gP.surfSplit + tn.z].add(1);
        }
        else
            surfApply(v, tn, w, nn, g, r);
    }
    if (proxies != 0u) found = surfTriangleProxies(v, tn, x0, x1, x2, r, boxCount, proxies, parity, apply, mask, found);
    surfMaskOut(mk, apply, found);
}

// Beam j is broken as the CPU sees it after this step's beam pass: its flag, or a
// detacher-group break this step (the next node phase marks those). (The slide
// nodes' brokenNow, which only the EB_AERO modules have.)
bool surfBroken(uint j, uint groupOff, uint parity)
{
    if (gBeams[j * 2u + 1u].z != 0.0) return true;
    int g = asint(gStatic[gP.beamBase + j * 3u].w);
    if (g == 0) return false;
    return gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load() != NO_GROUP;
}

// Beam bn (nodes a, b) of vehicle v against every proxy shape (surfSegment's).
uint surfSegmentProxies(uint v, uint4 bn, float3 x0, float3 x1, float r, uint boxCount, uint proxies, uint parity,
                        bool apply, uint mask, uint found)
{
    float3 tlo = ssub3(min(x0, x1), float3(r, r, r));
    float3 thi = sadd3(max(x0, x1), float3(r, r, r));
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        uint b = boxCount + k;
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 vb;
        bool fast = surfProxyFast(v, k, vb);
        if (!fast && !surfProxyMay(v, k, tlo, thi)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 w;
        float3 nn;
        float g;
        float3 q;
        bool two;
        float3 w2;
        float3 q2;
        bool hit = fqSegNear(x0, x1, r, lo, hi) && fqSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2);
        float3 shift = float3(0.0, 0.0, 0.0);
        if (!hit && fast)
        {
            float3 ve = sscale(sadd3(gNodes[bn.x * 3u + 1u].xyz, gNodes[bn.y * 3u + 1u].xyz), 0.5);
            FqShape sm;
            hit = surfCastUse(v, j, parity, s, x0, x1, x1, 1u, sscale(ssub3(vb, ve), STEP), r, sm, shift)
                  && fqSegContact(sm, x0, x1, r, w, nn, g, q, two, w2, q2);
            if (!hit) shift = float3(0.0, 0.0, 0.0);
        }
        if (!hit) continue;
        uint hits = two ? 2u : 1u;
        if (!apply)
        {
            found |= surfMarkBit(b);
            gAcc[gP.surfSplit + bn.x].add(int(hits));
            gAcc[gP.surfSplit + bn.y].add(int(hits));
            surfBodyHit(v, j);
            if (two) surfBodyHit(v, j);
        }
        else
        {
            surfApplyTo(v, uint4(bn.x, bn.y, bn.x, 0u), w, nn, g, r, k, j, ssub3(q, shift));
            if (two) surfApplyTo(v, uint4(bn.x, bn.y, bn.x, 0u), w2, nn, g, r, k, j, ssub3(q2, shift));
        }
    }
    return found;
}

// Beam gj of vehicle v (a segment: beam-edge contact) against every box slot, then
// every proxy shape: apply 0 counts its hits, 1 applies them (its nodes (a, b, a), w.z 0).
void surfSegment(uint v, uint gj, uint mk, float3 origin, float r, uint boxOff, uint boxCount, uint proxies,
                 uint groupOff, uint parity, bool apply)
{
    uint mask = surfMaskIn(mk, apply);
    if ((apply && mask == 0u) || surfBroken(gj, groupOff, parity))
    {
        surfMaskOut(mk, apply, 0u);
        return;
    }
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    float3 x0 = nf3(origin + gNodes[bn.x * 3u].xyz);
    float3 x1 = nf3(origin + gNodes[bn.y * 3u].xyz);
    uint found = 0u;
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
    {
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 lo;
        float3 hi;
        FqShape s = surfSlot(boxOff, b, lo, hi);
        if (!fqSegNear(x0, x1, r, lo, hi)) continue;
        float3 w;
        float3 nn;
        float g;
        float3 q;
        bool two;
        float3 w2;
        float3 q2;
        if (!fqSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2)) continue;
        uint hits = two ? 2u : 1u;
        if (!apply)
        {
            found |= surfMarkBit(b);
            gAcc[gP.surfSplit + bn.x].add(int(hits));
            gAcc[gP.surfSplit + bn.y].add(int(hits));
        }
        else
        {
            surfApply(v, uint4(bn.x, bn.y, bn.x, 0u), w, nn, g, r);
            if (two) surfApply(v, uint4(bn.x, bn.y, bn.x, 0u), w2, nn, g, r);
        }
    }
    if (proxies != 0u) found = surfSegmentProxies(v, bn, x0, x1, r, boxCount, proxies, parity, apply, mask, found);
    surfMaskOut(mk, apply, found);
}

// Node gi of vehicle v at x against every proxy shape (surfPoint's).
uint surfPointProxies(uint v, uint gi, float3 x, float r, uint boxCount, uint proxies, uint parity, bool apply,
                      uint mask, uint found)
{
    float3 tlo = ssub3(x, float3(r, r, r));
    float3 thi = sadd3(x, float3(r, r, r));
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        uint b = boxCount + k;
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 vb;
        bool fast = surfProxyFast(v, k, vb);
        if (!fast && !surfProxyMay(v, k, tlo, thi)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 nn;
        float g;
        bool hit = fqSegNear(x, x, r, lo, hi) && fqPointContact(s, x, r, nn, g);
        float3 shift = float3(0.0, 0.0, 0.0);
        if (!hit && fast)
        {
            FqShape sm;
            hit = surfCastUse(v, j, parity, s, x, x, x, 2u, sscale(ssub3(vb, gNodes[gi * 3u + 1u].xyz), STEP), r, sm, shift)
                  && fqPointContact(sm, x, r, nn, g);
            if (!hit) shift = float3(0.0, 0.0, 0.0);
        }
        if (!hit) continue;
        if (!apply)
        {
            found |= surfMarkBit(b);
            gAcc[gP.surfSplit + gi].add(1);
            surfBodyHit(v, j);
        }
        else
            surfApplyTo(v, uint4(gi, gi, gi, 0u), float3(1.0, 0.0, 0.0), nn, g, r, k, j, ssub3(x, shift));
    }
    return found;
}

// Node gi of vehicle v (point contact in the band) against every box slot, then every
// proxy shape: apply 0 counts its hits, 1 applies them.
void surfPoint(uint v, uint gi, uint mk, float3 origin, float r, uint boxOff, uint boxCount, uint proxies, uint parity,
               bool apply)
{
    uint mask = surfMaskIn(mk, apply);
    if ((apply && mask == 0u) || (gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u)
    {
        surfMaskOut(mk, apply, 0u);
        return;
    }
    float3 x = nf3(origin + gNodes[gi * 3u].xyz);
    uint found = 0u;
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
    {
        if (apply && (mask & surfMarkBit(b)) == 0u) continue;
        float3 lo;
        float3 hi;
        FqShape s = surfSlot(boxOff, b, lo, hi);
        if (!fqSegNear(x, x, r, lo, hi)) continue;
        float3 nn;
        float g;
        if (!fqPointContact(s, x, r, nn, g)) continue;
        if (!apply)
        {
            found |= surfMarkBit(b);
            gAcc[gP.surfSplit + gi].add(1);
        }
        else
            surfApply(v, uint4(gi, gi, gi, 0u), float3(1.0, 0.0, 0.0), nn, g, r);
    }
    if (proxies != 0u) found = surfPointProxies(v, gi, x, r, boxCount, proxies, parity, apply, mask, found);
    surfMaskOut(mk, apply, found);
}

// The casts' gathering (CPU Casts, its first pass), in the snapshot phase: every element
// against every fast proxy it has no discrete contact with, its time of impact into its
// body's earliest. Triangle gt, segment beam gj, node gi.
void surfGatherTriangle(uint v, uint gt, float3 origin, float r, uint proxies, uint parity)
{
    if ((gP.triBreak != 0u && triangleDead(gt)) || (gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 x0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 x1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 x2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (!surfProxyFast(v, k, vb)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 w;
        float3 nn;
        float g;
        float3 q;
        if (fqNear(x0, x1, x2, r, lo, hi) && fqContact(s, x0, x1, x2, r, w, nn, g, q)) continue;
        float3 ve = sdiv3(sadd3(sadd3(gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz), gNodes[tn.z * 3u + 1u].xyz), 3.0);
        surfGatherAt(v, j, parity, surfCastToi(s, x0, x1, x2, 0u, sscale(ssub3(vb, ve), STEP), r));
    }
}

void surfGatherSegment(uint v, uint gj, float3 origin, float r, uint proxies, uint groupOff, uint parity)
{
    if (surfBroken(gj, groupOff, parity)) return;
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    float3 x0 = nf3(origin + gNodes[bn.x * 3u].xyz);
    float3 x1 = nf3(origin + gNodes[bn.y * 3u].xyz);
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (!surfProxyFast(v, k, vb)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 w;
        float3 nn;
        float g;
        float3 q;
        bool two;
        float3 w2;
        float3 q2;
        if (fqSegNear(x0, x1, r, lo, hi) && fqSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2)) continue;
        float3 ve = sscale(sadd3(gNodes[bn.x * 3u + 1u].xyz, gNodes[bn.y * 3u + 1u].xyz), 0.5);
        surfGatherAt(v, j, parity, surfCastToi(s, x0, x1, x1, 1u, sscale(ssub3(vb, ve), STEP), r));
    }
}

void surfGatherPoint(uint v, uint gi, float3 origin, float r, uint proxies, uint parity)
{
    if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u) return;
    float3 x = nf3(origin + gNodes[gi * 3u].xyz);
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (!surfProxyFast(v, k, vb)) continue;
        uint j;
        FqShape s = surfProxyShape(v, k, j);
        float3 lo;
        float3 hi;
        fqBounds(s, lo, hi);
        float3 nn;
        float g;
        if (fqSegNear(x, x, r, lo, hi) && fqPointContact(s, x, r, nn, g)) continue;
        surfGatherAt(v, j, parity, surfCastToi(s, x, x, x, 2u, sscale(ssub3(vb, gNodes[gi * 3u + 1u].xyz), STEP), r));
    }
}

// Whether vehicle v has a fast proxy this step (every lane reads its proxies).
bool surfAnyFast(uint v, uint proxies)
{
    for (uint k = 0u, kStop = EB_END(k, proxies, 1u); k < kStop; k++)
    {
        float3 vb;
        if (surfProxyFast(v, k, vb)) return true;
    }
    return false;
}

// Whether any vehicle of the upload has surface contact (uniform: a parameter).
uint surfaceOn() { return gP.surfSplit != 0u ? 1u : 0u; }

// Step's surface pass of vehicle v (uniform `on`), after its self-collision and
// before its slide nodes (parity: the step's): the barrier, the snapshot, the find,
// the apply. A vehicle without the flag reaches the barriers only. The caller's
// barrier follows.
void surfaceStep(uint v, uint lane, uint on, uint parity)
{
    if (on == 0u) return;
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
    sync();
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
    if (proxies != 0u && surfAnyFast(v, proxies))
    {
        for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
            surfGatherTriangle(v, r1.x + t, origin, r, proxies, parity);
        for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
            surfGatherSegment(v, gLists[sr.x + j], origin, r, proxies, r2.z, parity);
        for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
            surfGatherPoint(v, r0.x + i, origin, r, proxies, parity);
    }
    sync();
    for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
        surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, proxies, parity, false);
    for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
        surfSegment(v, gLists[sr.x + j], sr.z + j, origin, r, r1.z, r1.w, proxies, r2.z, parity, false);
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
        surfPoint(v, r0.x + i, sr.w + i, origin, r, r1.z, r1.w, proxies, parity, false);
    sync();
    for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
        surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, proxies, parity, true);
    for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
        surfSegment(v, gLists[sr.x + j], sr.z + j, origin, r, r1.z, r1.w, proxies, r2.z, parity, true);
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
        surfPoint(v, r0.x + i, sr.w + i, origin, r, r1.z, r1.w, proxies, parity, true);
    if (lane == 0u && mine) gsHit = 1u;
}
