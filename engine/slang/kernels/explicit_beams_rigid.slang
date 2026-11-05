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
// node phase (gsCoupleOn).
groupshared uint gsUniformCouple;
groupshared uint gsCoupleOn;
uint vvMember(uint off, uint m);

static const uint CPL_BOX = 0u;
static const uint CPL_SPHERE = 1u;
static const uint CPL_CAPSULE = 2u;
static const uint CPL_HULL = 3u;
static const uint CPL_BODY_RECS = 5u;
static const uint CPL_SHAPE_RECS = 7u;

uint coupleRec() { return grabRec() + 2u; }
uint cplB() { return gMisc[coupleRec()].z; }
uint cplS() { return gMisc[coupleRec()].w & 0xFFFFu; }
uint cplPlanes() { return gMisc[coupleRec()].w >> 16; }
uint cplRecs() { return 1u + cplB() * CPL_BODY_RECS + cplS() * CPL_SHAPE_RECS + cplPlanes(); }
uint cplBase(uint v) { return gMisc[coupleRec()].y + v * cplRecs(); }
uint cplBody(uint v, uint j) { return cplBase(v) + 1u + j * CPL_BODY_RECS; }
uint cplShape(uint v, uint k) { return cplBase(v) + 1u + cplB() * CPL_BODY_RECS + k * CPL_SHAPE_RECS; }
uint cplPlane(uint v, uint i) { return cplBase(v) + 1u + cplB() * CPL_BODY_RECS + cplS() * CPL_SHAPE_RECS + i; }
uint cplAcc(uint v) { return gMisc[coupleRec() + 1u].x + v * (cplB() + 6u * cplS()); }

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

// The two-body contact on a node of vehicle v at world x (force f so far, velocity vel,
// mass, friction) from every proxy shape it is inside, in shape order; the reactions are
// summed into the shapes' accumulators. Returns the node's force with them added.
float3 cplContacts(uint v, float3 x, float3 f, float3 vel, float mass, float friction)
{
    uint4 hdr = gMisc[cplBase(v)];
    uint acc = cplAcc(v);
    uint nb = cplB();
    for (uint k = 0u, kStop = EB_END(k, hdr.y, 1u); k < kStop; k++)
    {
        float4 n4 = cplFace(v, k, x);
        if (n4.w == 0.0) continue;
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

// Lanes < B: each body sums its shapes' step reactions (into the shapes' window totals,
// the sums cleared), then integrates (symplectic Euler, as the nodes); its count cleared.
void cplIntegrate(uint v, uint lane)
{
    uint4 hdr = gMisc[cplBase(v)];
    if (lane >= hdr.x) return;
    uint j = lane;
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
    for (uint k = 0u, kStop = EB_END(k, hdr.y, 1u); k < kStop; k++)
    {
        uint at = cplShape(v, k);
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

// Every lane: its nodes against the proxy shapes, each inside counted on its body.
void cplCount(uint v, uint lane)
{
    uint4 hdr = gMisc[cplBase(v)];
    if (hdr.y == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    float3 origin = asfloat(gMisc[v].xyz);
    uint acc = cplAcc(v);
    for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u) continue;
        float3 x = origin + gNodes[gi * 3u].xyz;
        for (uint k = 0u, kStop = EB_END(k, hdr.y, 1u); k < kStop; k++)
        {
            if (cplFace(v, k, x).w != 0.0) gAcc[acc + gMisc[cplShape(v, k)].y].add(1);
        }
    }
}

// The coupling's part of a step of vehicle v, at the step's start (uniform `on`).
void coupleStep(uint v, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    if (s > 0u) cplIntegrate(v, lane);
    sync();
    cplCount(v, lane);
    sync();
}

// After a frame's last step (its closing barrier passed): that step's integration.
void coupleEnd(uint v, uint lane, uint on)
{
    if (on == 0u) return;
    if (gP.steps > 0u) cplIntegrate(v, lane);
}

// coupleStep and coupleEnd for a contact group's k members.
void coupleStepMembers(uint off, uint k, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    if (s > 0u)
    {
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplIntegrate(vvMember(off, m), lane);
    }
    sync();
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) cplCount(vvMember(off, m), lane);
    sync();
}

void coupleEndMembers(uint off, uint k, uint lane, uint on)
{
    if (on == 0u) return;
    sync();
    if (gP.steps > 0u)
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
    if (gP.selfSplit != 0u && (flagsV & FLAG_TWO_BODY) != 0u) gAcc[gP.selfSplit + gi].store(0);
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
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
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
                    float3 c = contactForce(f, vel, mass, n.xyz, boxSurface(), 0.0, V.w);
                    cfAdd(gi, c);
                    f += c;
                }
            }
        }
        float3 fr = cplContacts(vehicle, x, f, vel, mass, V.w);
        cfAdd(gi, fr - f);
        f = fr;
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
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
bool cplHas(uint v)
{
    return gsCoupleOn != 0u && gMisc[cplBase(v)].y != 0u;
}

// The frame's start (lane 0, before the barrier the uniform read follows).
void coupleBegin()
{
    uint on = gMisc[coupleRec()].x;
    gsUniformCouple = on;
    gsCoupleOn = on;
}
