// Per particle, after soft_collide_tris: its triangle-side contacts this step,
// (triangle << 5 | corner << 3 | kind), in the order softContactsFor's walk of
// its triangles visits them. The sweep then reads only real contacts instead of
// every slot of every triangle around it.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_contact_lists(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softListCount()) return;
    uint4 ent = SU(softListBase() + i);
    uint body = ent.x;
    uint ord = particleOrdinal(ent.w & 0xFFFFu, body);
    uint base = softContactListBase() + CONTACT_LIST_STRIDE * ord;
    uint n = 0u;
    uint4 slot = uint4(0, 0, 0, 0);
    if (softContactsOn() && SD(softFlagBase() + ord).x != 0.0)
    {
        uint adj = softAdjBase();
        uint triStart = ent.y + 2u * (ent.z & 0xFFFFu), triCount = ent.z >> 16;
        for (uint k = 0u; k < triCount && n <= CONTACT_LIST_MAX; k++)
        {
            uint wi = triStart + k;
            uint tt = SU(adj + (wi >> 2))[wi & 3u];
            uint4 tri = SU(softTriBase() + tt);
            uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
            uint j = tri.x == body ? 0u : (tri.y == body ? 1u : 2u);
            for (uint kind = 0u; kind <= TRI_NODE_CONTACTS && n <= CONTACT_LIST_MAX; kind++)
            {
                uint w = kind == 0u ? asuint(SD(tc).w) : asuint(SD(tc + 3u * kind).w);
                if (w == SOFT_NONE) continue;
                n++;
                if (n > CONTACT_LIST_MAX) break;
                slot[n & 3u] = (tt << 5) | (j << 3) | kind;
                if ((n & 3u) == 3u) { gSoftData[base + (n >> 2)] = asfloat(slot); slot = uint4(0, 0, 0, 0); }
            }
        }
    }
    if ((n & 3u) != 3u && n <= CONTACT_LIST_MAX) gSoftData[base + (n >> 2)] = asfloat(slot);
    uint4 head = SU(base);
    head.x = n > CONTACT_LIST_MAX ? CONTACT_LIST_ALL : n;
    gSoftData[base] = asfloat(head);
}

void softContactsFor(uint body, uint ord, float3 x, float3 x0, uint triStart, uint triCount, uint lane, uint lanes,
    inout float3 g, inout float3x3 H)
{
    if (!softContactsOn()) return;
    // Detection flags every particle in a soft contact this step (node or
    // triangle vertex); the rest have nothing to read.
    if (SD(softFlagBase() + ord).x == 0.0) return;
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * ord;
    for (uint j = lane; j < NODE_CONTACTS; j += lanes)
    {
        float4 s0 = SD(ns + 3u * j);
        uint t = asuint(s0.w);
        if (t == SOFT_NONE) continue;
        float4 s1 = SD(ns + 3u * j + 1u);
        uint4 tri = SU(softTriBase() + t);
        float C = dot(s0.xyz, x - triPoint(tri, s1.xyz)) - s1.w;
        softContactTerm(g, H, 1.0, C, SD(ns + 3u * j + 2u).x, s0.xyz, (x - x0) - triTravel(tri, s1.xyz), s1.w);
    }
    uint cl = softContactListBase() + CONTACT_LIST_STRIDE * ord;
    uint n = SU(cl).x;
    if (n != CONTACT_LIST_ALL)
    {
        for (uint k = lane; k < n; k += lanes)
        {
            uint w = k + 1u;
            uint code = SU(cl + (w >> 2))[w & 3u];
            uint tt = code >> 5, kind = code & 7u;
            uint4 tri = SU(softTriBase() + tt);
            uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
            if (kind == 0u) softTriObstacleTerm(tri, tc, (code >> 3) & 3u, g, H);
            else softTriNodeTerm(tri, tc, (code >> 3) & 3u, kind - 1u, g, H);
        }
        return;
    }
    // Too many for the list: walk every triangle around the particle.
    uint adj = softAdjBase();
    for (uint k = lane; k < triCount; k += lanes)
    {
        uint wi = triStart + k;
        uint tt = SU(adj + (wi >> 2))[wi & 3u];
        uint4 tri = SU(softTriBase() + tt);
        uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
        // This particle's weight in the triangle.
        uint j = tri.x == body ? 0u : (tri.y == body ? 1u : 2u);
        if (asuint(SD(tc).w) != SOFT_NONE) softTriObstacleTerm(tri, tc, j, g, H);
        for (uint m = 0u; m < TRI_NODE_CONTACTS; m++)
            if (asuint(SD(tc + 3u + 3u * m).w) != SOFT_NONE) softTriNodeTerm(tri, tc, j, m, g, H);
    }
}

static const uint SOFT_SKIP = 0u;
static const uint SOFT_FOLLOW = 1u;
static const uint SOFT_SOLVE = 2u;

// ---------------------------------------------------------------------------
// Elements (engine-softbody elements.rs): LBEAMs (the angle at the middle of
// three nodes), torsion bars (the dihedral of four about the 2–3 axis) and
// slidenodes on rails. Records at elementBase + 3·k: [nodes, or the slidenode,
// its rail list start and length, flags], [kind | flags | soft body << 16,
// spring, damping, rest angle], [deform, strength, tolerance, cap strength].
// A particle lists the elements it takes part in after its triangles.
// ---------------------------------------------------------------------------
static const uint ELEMENT_STRIDE = 3u;
static const uint ELEMENT_LBEAM = 0u;
static const uint ELEMENT_TORSION = 1u;
static const uint ELEMENT_SLIDE = 2u;
static const uint ELEMENT_BROKEN = 256u;
static const uint ELEMENT_YIELDED = 512u;
static const uint SLIDE_LOOPED = 1u;
static const uint SLIDE_CAPPED = 2u;
static const uint SLIDE_ATTACHED = 4u;
static const uint ENTRY_ELEMENT_SHIFT = 18u;
static const float SOFT_PI = 3.14159265358979;

uint railNode(uint i) { return SU(softRailNodeBase() + (i >> 2))[i & 3u]; }
float wrapAngle(float a)
{
    float t = 2.0 * SOFT_PI;
    a = a - t * floor((a + SOFT_PI) / t);
    return a;
}
float3 elemPos(uint ob, uint body) { return softLocal(ob + body); }
float3 elemPos0(uint ob, uint body) { return softLocal0(ob + body); }

// ∂θ/∂(a, m, b) of the angle at m, scaled by s/|u×v| where s is the generalised
// force; `straight` (θ₀ = π with the arms collinear) takes the limit −1/(|u||v|).
void lbeamDirs(float3 u, float3 v, out float3 da, out float3 db, out float cross_, out float theta)
{
    float lu2 = max(dot(u, u), 1e-12), lv2 = max(dot(v, v), 1e-12);
    cross_ = length(cross(u, v));
    theta = atan2(cross_, dot(u, v));
    da = -(v - u * (dot(u, v) / lu2));
    db = -(u - v * (dot(u, v) / lv2));
}

// Dihedral angle of 1–2–3–4 about 2→3 and its Jacobian (Blondel & Karplus);
// zero Jacobian where a plane is degenerate.
float dihedralOf(float3 x1, float3 x2, float3 x3, float3 x4, out float3 j1, out float3 j2, out float3 j3, out float3 j4)
{
    float3 b1 = x2 - x1, b2 = x3 - x2, b3 = x4 - x3;
    float3 n1 = cross(b1, b2), n2 = cross(b2, b3);
    float l1 = dot(n1, n1), l2 = dot(n2, n2), lb = length(b2);
    float phi = atan2(lb * dot(b1, n2), dot(n1, n2));
    j1 = float3(0, 0, 0); j2 = j1; j3 = j1; j4 = j1;
    if (l1 < 1e-18 || l2 < 1e-18 || lb < 1e-9) return phi;
    j1 = -n1 * (lb / l1);
    j4 = n2 * (lb / l2);
    float p = -dot(b1, b2) / (lb * lb), q = -dot(b3, b2) / (lb * lb);
    j2 = j1 * (p - 1.0) - j4 * q;
    j3 = j4 * (q - 1.0) - j1 * p;
    return phi;
}

// Closest point of p on rail (start, count) of local positions: the point, the
// unit tangent, the segment's end bodies and the parameter u on it.
void railClosest(uint ob, uint start, uint count, bool looped, float3 p,
    out float3 q, out float3 t, out uint ba, out uint bb, out float u, out bool offEnd)
{
    uint segs = looped ? count : count - 1u;
    float best = 1e30;
    q = p; t = float3(1, 0, 0); ba = railNode(start); bb = ba; u = 0.0;
    uint bestSeg = 0u;
    for (uint k = 0u; k < segs; k++)
    {
        uint na = railNode(start + k), nb = railNode(start + (k + 1u) % count);
        float3 a = elemPos(ob, na), b = elemPos(ob, nb);
        float3 ab = b - a;
        float l2 = max(dot(ab, ab), 1e-18);
        float s = clamp(dot(p - a, ab) / l2, 0.0, 1.0);
        float3 c = a + ab * s;
        float d = dot(p - c, p - c);
        if (d < best) { best = d; q = c; t = ab * rsqrt(l2); ba = na; bb = nb; u = s; bestSeg = k; }
    }
    offEnd = !looped && ((bestSeg == 0u && u <= 0.0 && dot(p - q, t) < -1e-9)
        || (bestSeg == segs - 1u && u >= 1.0 && dot(p - q, t) > 1e-9));
}

// This particle's share (gradient, block) of element k.
void softElementTerm(uint k, uint body, uint ob, float h, inout float3 g, inout float3x3 H)
{
    uint q = softElementBase() + ELEMENT_STRIDE * k;
    uint4 nodes = SU(q);
    uint4 r1 = SU(q + 1u);
    uint kind = r1.x & 255u;
    if ((r1.x & ELEMENT_BROKEN) != 0u) return;
    float kk = asfloat(r1.y), c = asfloat(r1.z) * softDampScale(), rest = asfloat(r1.w);
    float keff = kk + c / h;
    float3x3 I3 = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (kind == ELEMENT_LBEAM)
    {
        float3 xa = elemPos(ob, nodes.x), xm = elemPos(ob, nodes.y), xb = elemPos(ob, nodes.z);
        float3 u = xa - xm, v = xb - xm;
        float3 da, db; float cr, theta;
        lbeamDirs(u, v, da, db, cr, theta);
        float3 u0 = elemPos0(ob, nodes.x) - elemPos0(ob, nodes.y), v0 = elemPos0(ob, nodes.z) - elemPos0(ob, nodes.y);
        float theta0 = atan2(length(cross(u0, v0)), dot(u0, v0));
        float sGen = kk * (theta - rest) + c / h * (theta - theta0);
        float lu = length(u), lv = length(v);
        float ratio;
        if (cr > 1e-6 * lu * lv) ratio = sGen / cr;
        // Straight arms with a straight rest: (θ − π)/|u×v| → −1/(|u||v|).
        else if (abs(rest - SOFT_PI) < 1e-4) ratio = -kk / max(lu * lv, 1e-12);
        else ratio = 0.0;
        float3 ga = da * ratio, gb = db * ratio;
        float3 un = u / max(lu, 1e-9), vn = v / max(lv, 1e-9);
        float3x3 ha = (I3 - float3x3(un * un.x, un * un.y, un * un.z)) * (keff / max(lu * lu, 1e-12));
        float3x3 hb = (I3 - float3x3(vn * vn.x, vn * vn.y, vn * vn.z)) * (keff / max(lv * lv, 1e-12));
        if (body == nodes.x) { g += ga; H += ha; }
        else if (body == nodes.z) { g += gb; H += hb; }
        else { g -= ga + gb; H += (ha + hb) * 2.0; }
        return;
    }
    if (kind == ELEMENT_TORSION)
    {
        float3 j1, j2, j3, j4, i1, i2, i3, i4;
        float phi = dihedralOf(elemPos(ob, nodes.x), elemPos(ob, nodes.y), elemPos(ob, nodes.z), elemPos(ob, nodes.w), j1, j2, j3, j4);
        float phi0 = dihedralOf(elemPos0(ob, nodes.x), elemPos0(ob, nodes.y), elemPos0(ob, nodes.z), elemPos0(ob, nodes.w), i1, i2, i3, i4);
        float sGen = kk * wrapAngle(phi - rest) + c / h * wrapAngle(phi - phi0);
        float3 jm = body == nodes.x ? j1 : (body == nodes.y ? j2 : (body == nodes.z ? j3 : j4));
        g += jm * sGen;
        H += float3x3(jm * jm.x, jm * jm.y, jm * jm.z) * keff;
        return;
    }
    // Slidenode.
    if ((nodes.w & SLIDE_ATTACHED) == 0u) return;
    uint4 r2 = SU(q + 2u);
    float tol = asfloat(r2.z);
    float3 p = elemPos(ob, nodes.x);
    float3 cq, t; uint ba, bb; float uu; bool offEnd;
    railClosest(ob, nodes.y, nodes.z, (nodes.w & SLIDE_LOOPED) != 0u, p, cq, t, ba, bb, uu, offEnd);
    float3 d = p - cq;
    float3 perp = offEnd ? d : d - t * dot(d, t);
    float len = length(perp);
    if (len <= tol || len < 1e-9) return;
    float3 dir = perp / len;
    // Damping on the change of the across-rail offset this step.
    float3 p0 = elemPos0(ob, nodes.x);
    float3 q0 = elemPos0(ob, ba) * (1.0 - uu) + elemPos0(ob, bb) * uu;
    float rate = (len - dot(p0 - q0, dir)) / h;
    float f = kk * (len - tol) + c * rate;
    float3 gn = dir * f;
    float3x3 hn = (offEnd ? I3 : (I3 - float3x3(t * t.x, t * t.y, t * t.z))) * keff;
    if (body == nodes.x) { g += gn; H += hn; }
    else if (body == ba) { g -= gn * (1.0 - uu); H += hn * ((1.0 - uu) * (1.0 - uu)); }
    else if (body == bb) { g -= gn * uu; H += hn * (uu * uu); }
}

// Post-step: rest-angle plasticity and breaking (LBEAM, torsion bar), and a
// slidenode coming off its rail (past its strength, or off an uncapped end).
void softElementPlastic(uint k)
{
    uint q = softElementBase() + ELEMENT_STRIDE * k;
    uint4 nodes = SU(q);
    uint4 r1 = SU(q + 1u);
    if ((r1.x & ELEMENT_BROKEN) != 0u) return;
    uint s = r1.x >> 16;
    if (softIsRigid(s)) return;
    uint ob = asuint(softBodyRec1(s).w) - softBodyRec(s).x;
    uint kind = r1.x & 255u;
    float4 r2 = SD(q + 2u);
    float kk = asfloat(r1.y), rest = asfloat(r1.w);
    if (kind == ELEMENT_SLIDE)
    {
        if ((nodes.w & SLIDE_ATTACHED) == 0u) return;
        float3 p = elemPos(ob, nodes.x);
        float3 cq, t; uint ba, bb; float uu; bool offEnd;
        railClosest(ob, nodes.y, nodes.z, (nodes.w & SLIDE_LOOPED) != 0u, p, cq, t, ba, bb, uu, offEnd);
        float3 d = p - cq;
        float len = length(offEnd ? d : d - t * dot(d, t));
        float f = kk * max(len - r2.z, 0.0);
        bool capped = (nodes.w & SLIDE_CAPPED) != 0u;
        bool off = (offEnd && !capped) || f > (offEnd ? r2.y : r2.x) * softThresholdScale();
        if (off && (softFlags() & SOFT_BREAKING) != 0u)
            gSoftData[q] = asfloat(uint4(nodes.xyz, nodes.w & ~SLIDE_ATTACHED));
        return;
    }
    float ang, err;
    if (kind == ELEMENT_LBEAM)
    {
        float3 u = elemPos(ob, nodes.x) - elemPos(ob, nodes.y), v = elemPos(ob, nodes.z) - elemPos(ob, nodes.y);
        ang = atan2(length(cross(u, v)), dot(u, v));
        err = ang - rest;
    }
    else
    {
        float3 j1, j2, j3, j4;
        ang = dihedralOf(elemPos(ob, nodes.x), elemPos(ob, nodes.y), elemPos(ob, nodes.z), elemPos(ob, nodes.w), j1, j2, j3, j4);
        err = wrapAngle(ang - rest);
    }
    float moment = kk * err;
    float deform = r2.x * softThresholdScale();
    float carried = abs(moment);
    uint flags = r1.x;
    if ((softFlags() & SOFT_PLASTICITY) != 0u && abs(moment) > deform)
    {
        rest += sign(err) * (abs(moment) - deform) / kk;
        carried = deform;
        flags |= ELEMENT_YIELDED;
    }
    if ((softFlags() & SOFT_BREAKING) != 0u && carried > r2.y * softThresholdScale()) flags |= ELEMENT_BROKEN;
    gSoftData[q + 1u] = asfloat(uint4(flags, r1.y, r1.z, asuint(rest)));
}

// VBD block of one particle (colour-list entry `ent`), split over `lanes` lanes:
// this lane's share (lane 0: inertia, rigid edges and its node contact; beams and
// contact triangles strided over the lanes). The block is the sum of the shares.
// Returns SOFT_SKIP (asleep, static), SOFT_FOLLOW (rigid mode) or SOFT_SOLVE.
uint softAccumulate(uint4 ent, uint lane, uint lanes, out float3x3 H, out float3 g)
{
    H = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    g = float3(0.0, 0.0, 0.0);
    uint body = ent.x;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b)) return SOFT_SKIP;
    if (softIsRigid(ent.w & 0xFFFFu)) return SOFT_FOLLOW;
    AvbdParams p = PARAMS;
    // A crash island's substep (the whole step otherwise).
    float h = softH(ent.w & 0xFFFFu);

    // Inertia, contacts and joints: the rigid accumulation, rotation frozen.
    bool edges = SOLVE_INDEX(body) != SOLVE_INDEX(body + 1u) || (b.flags & BODY_VEHICLE) != 0u;
    if (!edges && lane != 0u) {}
    else if (!edges)
    {
        // No contact or joint edge: primalAccumulate's linear block is the inertia
        // term alone (same arithmetic), without its 6×6 accumulation.
        float k = (b.invMass > 1e-12 ? (1.0 / b.invMass) : 0.0) / (h * h);
        uint o = particleOrdinal(ent.w & 0xFFFFu, body);
        if (softBdf2For(o)) k *= 2.25;
        float3 dp = softLocal(o) - softTargetLocal(o);
        H = float3x3(k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, k);
        g = float3(k * dp.x, k * dp.y, k * dp.z);
    }
    else
    {
        // Its contact and joint edges strided over the particle's lanes (lane 0
        // also adds the inertia), as the rigid colour solve splits a body.
        Mat6 A;
        Vec6 rhs;
        primalAccumulate(body, lane, lanes, A, rhs);
        H = float3x3(
            A.m[0], A.m[6], A.m[12],
            A.m[6], A.m[7], A.m[13],
            A.m[12], A.m[13], A.m[14]);
        g = float3(rhs.v[0], rhs.v[1], rhs.v[2]);
        uint o = particleOrdinal(ent.w & 0xFFFFu, body);
        if (lane == 0u)
        {
            // primalAccumulate's inertia gradient came from the world copies:
            // swap it for the local one (the same term, without their rounding).
            // A substep weighs the inertia m/hₛ² (primalAccumulate added m/h²).
            float m = b.invMass > 1e-12 ? (1.0 / b.invMass) : 0.0;
            float kw = m / (p.dt * p.dt), k = m / (h * h);
            float3 dl = softLocal(o) - softTargetLocal(o);
            g += k * dl - kw * (b.position - gAabbs[body].mn);
            H += float3x3(k - kw, 0.0, 0.0, 0.0, k - kw, 0.0, 0.0, 0.0, k - kw);
            if (softBdf2For(o))
            {
                // BDF2 inertia weight 9/4: add the other 5/4 of the term.
                H += float3x3(1.25 * k, 0.0, 0.0, 0.0, 1.25 * k, 0.0, 0.0, 0.0, 1.25 * k);
                g += 1.25 * k * dl;
            }
        }
    }

    uint sbe = ent.w & 0xFFFFu;
    uint ordBase = asuint(softBodyRec1(sbe).w) - softBodyRec(sbe).x;
    float3 xa = softLocal(ordBase + body);
    float3 xa0 = softLocal0(ordBase + body);
    uint adj = softAdjBase();
    uint nBeams = ent.z & 0xFFFFu;
    bool plastic = (softFlags() & SOFT_PLASTICITY) != 0u;
    float ts = softThresholdScale();
    for (uint k = lane; k < nBeams; k += lanes)
    {
        // (beam, other end) pairs: the neighbour's position loads beside the beam.
        uint w = ent.y + 2u * k, w1 = w + 1u;
        uint e = SU(adj + (w >> 2))[w & 3u];
        uint other = SU(adj + (w1 >> 2))[w1 & 3u];
        // Fields, not the 112-byte record: WGSL copies a loaded struct whole.
        float3 oPos = softLocal(ordBase + other);
        float3 oPos0 = softLocal0(ordBase + other);
        SoftBeam sb = loadBeam(e);
        if (beamBroken(sb)) continue;
        float3 d = xa - oPos;
        float len = length(d);
        if (len < 1e-6) continue;
        float3 n = d / len;
        float2 el = beamElastic(sb, e, len, sb.penalty);
        float c = beamDampCoef(sb, e, len);
        float len0 = length(xa0 - oPos0);
        float d1 = el.x + c / h * (len - len0);
        float d2 = el.y + c / h;
        float tension = el.x;
        if (plastic && beamPlastic(beamKind(sb)) && sb.spring > 0.0 && !beamHard(sb))
        {
            // In-solve plasticity (engine_softbody::model::yield_cap): past its
            // threshold the beam flows at it. The curvature stays elastic, an upper
            // bound that keeps the undamped block step from overshooting.
            d1 = clamp(d1, -sb.deformC * ts, sb.deformT * ts);
            tension = min(tension, max(d1, 0.0));
        }
        float lat = max(tension / len, 0.0);
        g += n * d1;
        float3x3 nn = float3x3(n * n.x, n * n.y, n * n.z);
        H += nn * (d2 - lat);
        H[0][0] += lat; H[1][1] += lat; H[2][2] += lat;
    }
    // Contacts between bodies: in the world (both sides' world copies).
    softContactsFor(body, ordBase + body, b.position, b.initialPos, ent.y + 2u * nBeams, ent.z >> 16, lane, lanes, g, H);
    // Elements, after the triangles in the adjacency.
    uint nElem = ent.w >> ENTRY_ELEMENT_SHIFT;
    uint elemStart = ent.y + 2u * nBeams + (ent.z >> 16);
    for (uint k = lane; k < nElem; k += lanes)
    {
        uint wi = elemStart + k;
        softElementTerm(SU(adj + (wi >> 2))[wi & 3u], body, ordBase, h, g, H);
    }
    return SOFT_SOLVE;
}

// Displacement bounds (BeamNG-parity plan, phase 2; after Offset Geometric
// Contact, Chen et al. 2025): a particle in a soft contact may move toward each
// face it is in front of by at most γ times its distance from it at detection,
// measured from where it was detected (the initial guess); a triangle vertex
// may move toward the nodes on either side of its face by at most γ times the
// nearest one's distance. With γ < 1/2 neither side can close the gap alone, so no
// node crosses a face within the step, however hard it is pushed. Motion along
// the faces is free.
float3 softBounded(uint4 ent, uint ord, float3 xn)
{
    float gamma = softBoundGamma();
    if (gamma <= 0.0 || !softContactsOn() || SD(softFlagBase() + ord).x == 0.0) return xn;
    float3 xd = softDetectLocal(ord);
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * ord;
    for (uint j = 0u; j < NODE_CONTACTS; j++)
    {
        float4 s0 = SD(ns + 3u * j);
        if (asuint(s0.w) == SOFT_NONE) break;
        float sd = SD(ns + 3u * j + 2u).y;
        if (sd <= 0.0) continue;
        float c = dot(s0.xyz, xn - xd);
        float lim = -gamma * sd;
        if (c < lim) xn += s0.xyz * (lim - c);
    }
    uint adj = softAdjBase();
    uint triStart = ent.y + 2u * (ent.z & 0xFFFFu), triCount = ent.z >> 16;
    for (uint k = 0u; k < triCount; k++)
    {
        uint wi = triStart + k;
        uint tt = SU(adj + (wi >> 2))[wi & 3u];
        float2 near = SD(softTriContactBase() + TRI_CONTACT_STRIDE * tt + TRI_BOUND_SLOT).xy;
        if (near.x >= 1e29 && near.y >= 1e29) continue;
        // Toward the nodes in front of it (front face: +n), each side on its own.
        float3 n = SD(softTriPlaneBase() + 2u * tt).xyz;
        float c = dot(n, xn - xd);
        xn += n * (clamp(c, -gamma * near.y, gamma * near.x) - c);
    }
    return xn;
}

// Apply a particle's summed block: follow the core (rigid mode) or take the
// block step.
void softFinish(uint4 ent, uint mode, float3x3 H, float3 g)
{
    uint body = ent.x;
    if (mode == SOFT_SKIP) return;
    uint sb0 = ent.w & 0xFFFFu;
    if (mode == SOFT_FOLLOW)
    {
        // Rigid mode: follow the core's current iterate, no particle solve.
        BodyState c = gBodies[asuint(softBodyRec1(sb0).z)];
        uint of = particleOrdinal(sb0, body);
        float3 xw = c.position + quatRotate(c.orientation, SD(softRigidBase() + 2u * of).xyz);
        softSetLocal(body, of, sb0, xw - softOrigin(sb0));
        return;
    }
    AvbdParams p = PARAMS;
    uint ord = particleOrdinal(sb0, body);
    float3 xa = softLocal(ord);
    // Closed-form 3×3 solve (adjugate); skip near-singular blocks (VBD).
    float3 c0 = float3(H[0][0], H[1][0], H[2][0]);
    float3 c1 = float3(H[0][1], H[1][1], H[2][1]);
    float3 c2 = float3(H[0][2], H[1][2], H[2][2]);
    float3 r0 = cross(c1, c2);
    float det = dot(c0, r0);
    float scale = max(max(abs(H[0][0]), abs(H[1][1])), abs(H[2][2]));
    if (!(abs(det) > 1e-12 * scale * scale * scale)) return;
    float3 r1 = cross(c2, c0);
    float3 r2 = cross(c0, c1);
    float3 dx = -(r0 * g.x + r1 * g.y + r2 * g.z) / det;
    float maxLin = p.trustLin * 0.5;
    float l = length(dx);
    if (l > maxLin && l > 1e-12) dx *= maxLin / l;
    if (dx.x != dx.x || dx.y != dx.y || dx.z != dx.z) return;
    softSetLocal(body, ord, sb0, softBounded(ent, ord, xa + dx));
}

// VBD block solve of one particle on one lane.
void softSolveEntry(uint4 ent)
{
    float3x3 H;
    float3 g;
    uint mode = softAccumulate(ent, 0u, 1u, H, g);
    softFinish(ent, mode, H, g);
}

// One particle colour (PASS_U): VBD block solve of each particle.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_primal(uint3 tid : SV_DispatchThreadID)
{
    uint color = PASS_U & 255u;
    gStabilize = false;
    uint t = dispatchIndex(tid);
    uint4 cr = SU(softColorBase() + color);
    if (color >= softColorCount() || t >= cr.y) return;
    softSolveEntry(SU(softListBase() + cr.x + t));
}

// Chebyshev semi-iterative acceleration of the particle sweep (VBD §3.6),
// PASS_U = iteration index. Particles with any contact or joint edge this step
// are left alone.
// Chebyshev step of colour-list entry `i` after iteration `it`.
void softChebyshevEntry(uint i, uint it)
{
    float rho = softChebyshevRho();
    uint body = SU(softListBase() + i).x;
    uint q = softChebBase() + 3u * i;
    float3 prev2 = SD(q).xyz;
    float3 prev1 = SD(q + 1u).xyz;
    BodyState b = gBodies[body];
    uint4 ent = SU(softListBase() + i);
    uint ord = particleOrdinal(ent.w & 0xFFFFu, body);
    float3 x = softLocal(ord);
    bool soft = softContactsOn() && SD(softFlagBase() + particleOrdinal(ent.w & 0xFFFFu, body)).x != 0.0;
    if (softIsRigid(ent.w & 0xFFFFu)) soft = true; // rigid mode: no acceleration
    if (it > 0u && bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b) && !soft
        && SOLVE_INDEX(body + 1u) == SOLVE_INDEX(body))
    {
        float rho2 = rho * rho;
        float omega = 2.0 / (2.0 - rho2);
        for (uint k = 2u; k <= it; k++) omega = 4.0 / (4.0 - rho2 * omega);
        x = prev2 + (x - prev2) * omega;
        softSetLocal(body, ord, ent.w & 0xFFFFu, x);
    }
    gSoftData[q] = float4(prev1, 0.0);
    gSoftData[q + 1u] = float4(x, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_chebyshev(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softListCount() || softChebyshevRho() <= 0.0) return;
    softChebyshevEntry(i, PASS_U);
}

// One whole particle iteration in a single 256-lane workgroup: every colour in
// order with a device barrier between colours (the same Gauss–Seidel order as
// one soft_primal dispatch per colour), then the Chebyshev step. Hosts use it
// while every colour fits in one workgroup pass, where dispatch latency, not
// arithmetic, dominates a small scene. PASS_U = iteration index.
[shader("compute")]
[numthreads(256, 1, 1)]
void soft_primal_fused(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    gStabilize = false;
    if (local == 0u) gsUniformLanes = softColorCount();
    GROUP_SYNC();
    uint colors = gsUniformLanes;
    for (uint c = 0u; c < colors; c++)
    {
        uint4 cr = SU(softColorBase() + c);
        for (uint t = local; t < cr.y; t += 256u) softSolveEntry(SU(softListBase() + cr.x + t));
        DeviceMemoryBarrierWithGroupSync();
    }
    if (softChebyshevRho() > 0.0)
    {
        uint n = softListCount();
        for (uint i = local; i < n; i += 256u) softChebyshevEntry(i, PASS_U);
    }
}

// One whole particle iteration per soft body, one 256-lane workgroup each: the
// body's share of every colour in order with a barrier between colours (the same
// Gauss–Seidel order within the body as one soft_primal dispatch per colour),
// then its Chebyshev step. Bodies couple only through contacts frozen at the step
// start, read Jacobi-style as within a colour, so bodies need not wait for each
// other: one dispatch per iteration at any vehicle count. A colour holds a few
// dozen of a car's particles, so each particle gets up to 16 lanes: its beams and
// contact triangles strided over them, the 3×3 blocks summed in workgroup memory
// (a particle's solve is a chain of dependent reads; the lanes overlap them).
// PASS_U = iteration index.
static const uint SOFT_LANES_MAX = 16u;
groupshared float4 gsSoftRedA[256]; // H00 H01 H02 H11
groupshared float4 gsSoftRedB[256]; // H12 H22 g.x g.y
groupshared float gsSoftRedC[256];  // g.z
groupshared uint gsSoftAwake;
static const uint SOFT_ROLE_FOLLOW = 0xFFFFFFFEu;
static const uint SOFT_ROLE_VEHICLE = 0xFFFFFFFFu;

groupshared float gsVehL[256];  // per vehicle lane: its wheel's suspension length
groupshared float gsVehF[256];  // its wheel's normal load (after the anti-roll bars)
groupshared uint gsVehCt[256];  // its wheel touches the ground

// Lane-summed vehicle block: every lane adds its share of H's lower triangle and g
// into lane 0's, 9 values a round through gsSoftRed (soft workgroups' buffers).
static const uint VEH_LOWER[21] = { 0, 6, 7, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35 };
float vehBlockValue(Mat6 H, Vec6 g, uint k) { return k < 21u ? H.m[VEH_LOWER[k]] : g.v[k - 21u]; }

// vehicle_iterate for 64 vehicles, four lanes each (one wheel or spin group per
// lane): suspension per wheel, anti-roll bars and chassis rows on lane 0, spin
// groups per lane, then the lanes' blocks summed. The same rows as
// assembleVehicle in its order per lane; vehicles over four wheels take one lane.
void softVehicleLanes(uint group, uint local)
{
    uint slot = local / 4u, lane = local % 4u, first = slot * 4u;
    uint v = group * 64u + slot;
    bool act = v < vehicleCount();
    uint vr = act ? vehRec(v) : 0u;
    act = act && (VU(vr + V_D3).y & VS_ACTIVE) != 0u;
    uint4 cfg = act ? VU(vr + V_CFG) : uint4(0, 0, 0, 0);
    BodyState c = gBodies[cfg.x];
    act = act && bodyIsDynamic(c) && !bodyIsSleeping(c);
    bool split = act && cfg.z <= 4u;
    if (act && !split && lane == 0u) vehicleIterateOne(v);
    uint nW = split ? cfg.z : 0u;
    float3 dl = float3(0, 0, 0), da = float3(0, 0, 0);
    if (split) chassisDisplacement(c, dl, da);
    Mat6 H = mat6Zero();
    Vec6 g = vec6Zero();
    gsVehL[local] = 0.0;
    gsVehF[local] = 0.0;
    gsVehCt[local] = 0u;
    if (lane < nW)
    {
        if (wheelGround(wheelRec(cfg.y + lane)) != NO_INDEX) groundClear(cfg.y + lane);
        float L, Fn;
        bool ct;
        vehSuspension(lane, vr, cfg, c, dl, da, H, g, L, Fn, ct);
        gsVehL[local] = L;
        gsVehF[local] = Fn;
        gsVehCt[local] = ct ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    float Lw[4];
    float FnW[4];
    for (uint k = 0u; k < 4u; k++) { Lw[k] = gsVehL[first + k]; FnW[k] = gsVehF[first + k]; }
    if (split && lane == 0u)
    {
        vehAntiRoll<4>(vr, cfg, c, nW, Lw, FnW, H, g);
        for (uint k = 0u; k < 4u; k++) gsVehF[first + k] = FnW[k];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0u; k < 4u; k++) FnW[k] = gsVehF[first + k];
    if (lane < nW) vehSpinGroup<4>(lane, vr, cfg, c, dl, da, true, nW, FnW, H, g);
    if (split && lane == 0u)
    {
        uint contacts = gsVehCt[first] + gsVehCt[first + 1u] + gsVehCt[first + 2u] + gsVehCt[first + 3u];
        vehChassisRows<4>(vr, cfg, c, dl, da, true, nW, FnW, contacts, H, g);
    }
    // 27 values (21 lower-triangle entries of H, 6 of g) in three rounds of 9.
    for (uint round = 0u; round < 3u; round++)
    {
        uint k0 = round * 9u;
        gsSoftRedA[local] = float4(vehBlockValue(H, g, k0), vehBlockValue(H, g, k0 + 1u), vehBlockValue(H, g, k0 + 2u), vehBlockValue(H, g, k0 + 3u));
        gsSoftRedB[local] = float4(vehBlockValue(H, g, k0 + 4u), vehBlockValue(H, g, k0 + 5u), vehBlockValue(H, g, k0 + 6u), vehBlockValue(H, g, k0 + 7u));
        gsSoftRedC[local] = vehBlockValue(H, g, k0 + 8u);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            float vals[9];
            for (uint q = 0u; q < 9u; q++) vals[q] = 0.0;
            for (uint o = 0u; o < 4u; o++)
            {
                float4 a = gsSoftRedA[first + o], b = gsSoftRedB[first + o];
                vals[0] += a.x; vals[1] += a.y; vals[2] += a.z; vals[3] += a.w;
                vals[4] += b.x; vals[5] += b.y; vals[6] += b.z; vals[7] += b.w;
                vals[8] += gsSoftRedC[first + o];
            }
            for (uint q = 0u; q < 9u; q++)
            {
                uint k = k0 + q;
                if (k < 21u) H.m[VEH_LOWER[k]] = vals[q]; else g.v[k - 21u] = vals[q];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (split && lane == 0u)
    {
        writeVehicleBlock(vehicleAabbBlock(v), H, g, true);
        VSet(vr + V_REFP, float4(c.position, VD(vr + V_REFP).w));
        VSet(vr + V_REFQ, c.orientation);
    }
}

uint softLanesFor(uint n)
{
    uint q = 256u / max(n, 1u);
    return q >= 16u ? 16u : (q >= 8u ? 8u : (q >= 4u ? 4u : (q >= 2u ? 2u : 1u)));
}

