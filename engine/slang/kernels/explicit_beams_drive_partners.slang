// --- Wheels on other vehicles (a contact group's members; explicit_contact.rs
// ExplicitBeamsWorld::wheels_begin, explicit_drive.rs ray_triangle) ---
// The drive header's .w: the partner records' base (0: none), DRV_PARTNER a wheel at
// base + (v · wheels room + k) · DRV_PARTNER (explicit_plan::drive::PARTNER_RECS):
//   0  the hit (normal, t), or during the rays the ray's start (World) and its reach
//   1  the triangle's weights (w0, w1, w2) and its global index
//   2  (the step it is for + 1, the partner vehicle, 0, 0)
//   3  the wheel's staged reaction on the triangle (force, 1: staged), applied before
//      the next step (the vehicle-contact pass's timing)
//   4  the triangle point's velocity and two-body inverse mass (unsplit), or during the
//      rays the ray's direction
// A group's members' rays run at each step's start where every member's state is final
// (wheelsPartners: contactGroup's step start, the split path's own dispatch), so a
// member's step reads only its own records. The scratch record's .z is DRV_PARTNER_HIT
// for a wheel on another vehicle.
static const uint DRV_PARTNER = 5u;
static const uint DRV_PARTNER_HIT = 0xFFFFFFFFu;
uint drvPartnerBase() { return gMisc[driveRec()].w; }
uint drvPartnerAt(uint v, uint k)
{
    uint nwMax = (gMisc[driveRec()].z - DRV_VEH) / (DRV_WHEEL + DRV_PRE);
    return drvPartnerBase() + (v * nwMax + k) * DRV_PARTNER;
}

// explicit_drive::ray_triangle: (normal facing the ray's start, t), .w < 0 none; the
// weights in `w`.
float4 drvTriangle(float3 p, float3 d, float3 x0, float3 x1, float3 x2, out float3 w)
{
    w = float3(0.0, 0.0, 0.0);
    float3 e1 = ssub3(x1, x0);
    float3 e2 = ssub3(x2, x0);
    float3 pv = scross(d, e2);
    float det = sdot(e1, pv);
    if (abs(det) < DRV_PARALLEL) return drvNone();
    float3 tv = ssub3(p, x0);
    float u = sdiv(sdot(tv, pv), det);
    if (!(u >= 0.0 && u <= 1.0)) return drvNone();
    float3 qv = scross(tv, e1);
    float v = sdiv(sdot(d, qv), det);
    if (!(v >= 0.0 && nf(u + v) <= 1.0)) return drvNone();
    float t = sdiv(sdot(e2, qv), det);
    if (!(t >= 0.0)) return drvNone();
    float3 n = drvNrm(scross(e1, e2));
    if (sdot(n, d) > 0.0) n = -n;
    w = float3(nf(nf(1.0 - u) - v), u, v);
    return float4(n, t);
}

// The ray p + t·d (vehicle frame) against convex planes from outside (explicit_drive::ray_planes).
float4 drvHullRay(uint v, uint off, uint count, float4 r, float3 c, float3 p, float3 d)
{
    float tIn = -DRV_FLT_MAX;
    float tOut = DRV_FLT_MAX;
    float3 n = float3(0.0, 0.0, 0.0);
    for (uint i = 0u, iStop = EB_END(292, i, count, 1u); i < iStop; i++)
    {
        float4 pl = asfloat(gMisc[cplPlane(v, off + i)]);
        float3 wn = cplRot(r, pl.xyz);
        float o = nf(pl.w + sdot(wn, c));
        float dist = nf(sdot(wn, p) - o);
        float den = sdot(wn, d);
        if (abs(den) < DRV_PARALLEL)
        {
            if (dist > 0.0) return drvNone();
            continue;
        }
        float t = sdiv(-dist, den);
        if (den < 0.0)
        {
            if (t > tIn) { tIn = t; n = wn; }
        }
        else if (t < tOut) tOut = t;
    }
    if (tIn >= 0.0 && tIn <= tOut) return float4(n, tIn);
    return drvNone();
}

// The ray against vehicle v's proxy shape k (explicit_drive::ray_proxy).
float4 drvProxy(uint v, uint k, float3 p, float3 d)
{
    uint at = cplShape(v, k);
    uint4 h = gMisc[at];
    uint b = cplBody(v, h.y);
    float4 c0 = asfloat(gMisc[b]);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 lp = asfloat(gMisc[at + 1u]);
    float4 lq = asfloat(gMisc[at + 2u]);
    float4 pr = asfloat(gMisc[at + 3u]);
    float3 c = sadd3(c0.xyz, cplRot(q, lp.xyz));
    float4 r = cplQmul(q, lq);
    if (h.x == CPL_BOX)
    {
        float3 a0 = cplRot(r, float3(1.0, 0.0, 0.0));
        float3 a1 = cplRot(r, float3(0.0, 1.0, 0.0));
        float3 a2 = cplRot(r, float3(0.0, 0.0, 1.0));
        float3 o = ssub3(p, c);
        float3 hh = float3(lp.w, pr.x, pr.y);
        return drvSlab(float3(sdot(a0, o), sdot(a1, o), sdot(a2, o)), float3(sdot(a0, d), sdot(a1, d), sdot(a2, d)), -hh, hh, a0, a1, a2);
    }
    if (h.x == CPL_SPHERE) return drvSphere(p, d, c, lp.w);
    if (h.x == CPL_CAPSULE)
    {
        float3 ax = cplRot(r, float3(0.0, 1.0, 0.0));
        float3 a = ssub3(c, sscale(ax, pr.x));
        float3 e = sadd3(c, sscale(ax, pr.x));
        float4 best = drvSphere(p, d, a, lp.w);
        best = drvCloser(best, drvSphere(p, d, e, lp.w));
        float l = smul(2.0, pr.x);
        if (l > 0.0)
        {
            float t, y;
            float3 n;
            if (drvTube(p, d, a, ax, lp.w, t, y, n) && y >= 0.0 && y <= l) best = drvCloser(best, float4(n, t));
        }
        return best;
    }
    return drvHullRay(v, h.w & 0xFFFFu, h.w >> 16, r, c, p, d);
}

// The velocity of shape k's body at x (vehicle frame).
float3 drvBodyVel(uint v, uint k, float3 x)
{
    uint b = cplBody(v, gMisc[cplShape(v, k)].y);
    float3 c = asfloat(gMisc[b].xyz);
    return sadd3(asfloat(gMisc[b + 2u].xyz), scross(asfloat(gMisc[b + 3u].xyz), ssub3(x, c)));
}

// drvRays with the proxies: the nearest of the static surfaces' hit and every shape's.
void drvRaysBodies(DrvCtx c, uint k)
{
    uint wr = drvWheel(c, k);
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 hp = sadd3(xh, sscale(c.up, w2.z));
    float tMax = nf(w2.y + w1.x);
    float3 d = -c.up;
    float4 ht = drvRay(hp, c.ow, d, tMax, c.ground, c.boxOff, c.boxCount);
    uint shape = 0u;
    uint4 hdr = gsCpl[0] != 0u ? cplVeh(c.v) : uint4(0u, 0u, 0u, 0u);
    bool cull = cplCullOn();
    for (uint s = 0u, sStop = EB_END(293, s, hdr.y, 1u); s < sStop; s++)
    {
        // The shape's grown bounding sphere this step (cplSpheresBy): a segment that
        // misses it misses the shape (lossless).
        if (cull)
        {
            uint sw = cplSphere(c.v, s);
            float3 sc = asfloat(uint3(gNext[sw], gNext[sw + 1u], gNext[sw + 2u]));
            float tc = clamp(dot(sc - hp, d), 0.0, tMax);
            float3 off = sc - (hp + d * tc);
            if (dot(off, off) >= asfloat(gNext[sw + 3u])) continue;
        }
        float4 hs = drvProxy(c.v, s, hp, d);
        if (hs.w >= 0.0 && (ht.w < 0.0 || hs.w < ht.w))
        {
            ht = hs;
            shape = s + 1u;
        }
    }
    // Another vehicle's triangle (its record for this step), after every other surface.
    uint pa = 0u;
    if (drvPartnerBase() != 0u)
    {
        pa = drvPartnerAt(c.v, k);
        float4 hq = DF(pa);
        if (gMisc[pa + 2u].x == c.step + 1u && hq.w >= 0.0 && (ht.w < 0.0 || hq.w < ht.w))
        {
            ht = hq;
            shape = DRV_PARTNER_HIT;
        }
    }
    if (ht.w > tMax)
    {
        ht = drvNone();
        shape = 0u;
    }
    float len = w2.y;
    float load = 0.0;
    float4 hit = drvNone();
    uint on = 0u;
    if (ht.w >= 0.0)
    {
        float cosT = sdot(ht.xyz, c.up);
        if (cosT >= c.c5.z)
        {
            float l = nf(ht.w - w1.x);
            float3 vr = vh;
            if (shape == DRV_PARTNER_HIT) vr = ssub3(vh, DF(pa + 4u).xyz);
            else if (shape != 0u) vr = ssub3(vh, drvBodyVel(c.v, shape - 1u, ssub3(hp, sscale(c.up, ht.w))));
            float rate = sdiv(sdot(vr, ht.xyz), cosT);
            float fs = nf(smul(w2.w, nf(l - w2.y)) + smul(w3.x, rate));
            float f = fs < 0.0 ? -fs : 0.0;
            if (l < w2.x && w3.y > 0.0)
            {
                float fb = nf(smul(w3.y, nf(l - w2.x)) + smul(w3.z, rate));
                if (fb < 0.0) f = nf(f - fb);
            }
            len = l;
            load = f;
            hit = ht;
            on = shape;
        }
    }
    gMisc[wr + DRV_SCRATCH] = uint4(asuint(len), asuint(load), on, 0u);
    DSet(wr + DRV_SCRATCH + 1u, hit);
}

// Wheel k's surface on another vehicle's triangle (explicit_drive::partner_surface):
// (effective mass, share), the point's velocity in vb.
float2 drvPartnerSurface(DrvCtx c, uint k, float m, out float3 vb)
{
    uint pa = drvPartnerAt(c.v, k);
    float4 p4 = DF(pa + 4u);
    vb = p4.xyz;
    uint pv = gMisc[pa + 2u].y;
    uint same = 0u;
    for (uint j = 0u, jStop = EB_END(294, j, c.nw, 1u); j < jStop; j++)
    {
        if (gMisc[drvWheel(c, j) + DRV_SCRATCH].z == DRV_PARTNER_HIT && gMisc[drvPartnerAt(c.v, j) + 2u].y == pv) same++;
    }
    float wb = smul(float(same), p4.w);
    float wn = sdiv(1.0, m);
    return float2(sdiv(1.0, nf(wn + wb)), sdiv(wn, nf(wn + wb)));
}

// Wheel k's surface on its body (explicit_drive::surface): (effective mass, share) and
// the body's velocity at the contact point `at`.
float2 drvSurface(DrvCtx c, uint k, uint shape, float3 n, float3 at, float m, out float3 vb)
{
    uint body = gMisc[cplShape(c.v, shape)].y;
    uint b = cplBody(c.v, body);
    float4 c0 = asfloat(gMisc[b]);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 c2 = asfloat(gMisc[b + 2u]);
    float3 rr = ssub3(at, c0.xyz);
    vb = sadd3(c2.xyz, scross(asfloat(gMisc[b + 3u].xyz), rr));
    if (c0.w == 0.0) return float2(m, 1.0);
    uint same = 0u;
    for (uint j = 0u, jStop = EB_END(295, j, c.nw, 1u); j < jStop; j++)
    {
        uint sj = gMisc[drvWheel(c, j) + DRV_SCRATCH].z;
        if (sj != 0u && gMisc[cplShape(c.v, sj - 1u)].y == body) same++;
    }
    float3 rn = scross(rr, n);
    float3 ii = asfloat(gMisc[b + 4u].xyz);
    float3 irn = cplRot(q, ii * cplRot(float4(-q.xyz, q.w), rn));
    float split = smul(float(max(same, 1u)), max(c2.w, 1.0));
    float wb = smul(split, nf(c0.w + sdot(rn, irn)));
    float wn = sdiv(1.0, m);
    return float2(sdiv(1.0, nf(wn + wb)), sdiv(wn, nf(wn + wb)));
}

// Wheel k's axle (section 2).
float3 drvAxle(DrvCtx c, uint k)
{
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    bool hubAxles = (c.c5y & 4u) != 0u;
    float3 a = c.left;
    if (w0.x != w0.y && hubAxles)
        a = drvNrm(ssub3(sadd3(c.o, gNodes[(c.nodeOff + w0.y) * 3u].xyz), sadd3(c.o, gNodes[(c.nodeOff + w0.x) * 3u].xyz)));
    if (sdot(a, c.left) < 0.0) a = -a;
    // Less the spawn's toe and camber (in the vehicle frame).
    float3 ar = hubAxles ? DF(wr + 20u).xyz : float3(0.0, 0.0, 1.0);
    return drvNrm(sadd3(sadd3(sscale(c.fw, nf(sdot(a, c.fw) - ar.x)), sscale(c.up, nf(sdot(a, c.up) - ar.y))),
                        sscale(c.left, nf(nf(sdot(a, c.left) - ar.z) + 1.0))));
}

// Section 2 for wheel k: its ray and suspension into its scratch records.
void drvRaysBodies(DrvCtx c, uint k);
bool drvSeeBodies(DrvCtx c);
void drvRays(DrvCtx c, uint k)
{
    // Wheels on rigid bodies: their own function (Metal's codegen of the path below
    // stays as it was).
    if (drvSeeBodies(c))
    {
        drvRaysBodies(c, k);
        return;
    }
    uint wr = drvWheel(c, k);
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 hp = sadd3(xh, sscale(c.up, w2.z));
    float4 ht = drvRay(hp, c.ow, -c.up, nf(w2.y + w1.x), c.ground, c.boxOff, c.boxCount);
    float len = w2.y;
    float load = 0.0;
    float4 hit = drvNone();
    if (ht.w >= 0.0)
    {
        float cosT = sdot(ht.xyz, c.up);
        if (cosT >= c.c5.z)
        {
            float l = nf(ht.w - w1.x);
            float rate = sdiv(sdot(vh, ht.xyz), cosT);
            float fs = nf(smul(w2.w, nf(l - w2.y)) + smul(w3.x, rate));
            float f = fs < 0.0 ? -fs : 0.0;
            if (l < w2.x && w3.y > 0.0)
            {
                float fb = nf(smul(w3.y, nf(l - w2.x)) + smul(w3.z, rate));
                if (fb < 0.0) f = nf(f - fb);
            }
            len = l;
            load = f;
            hit = ht;
        }
    }
    DSet(wr + DRV_SCRATCH, float4(len, load, 0.0, 0.0));
    DSet(wr + DRV_SCRATCH + 1u, hit);
}

// The anti-roll bars, in order, on the wheels' scratch loads.
void drvBars(DrvCtx c)
{
    for (uint i = 0u, iStop = EB_END(296, i, c.arCount, 1u); i < iStop; i++)
    {
        uint4 ar = gMisc[c.vr + 12u + i];
        uint l = ar.x, r = ar.y;
        uint sl = drvWheel(c, l) + DRV_SCRATCH;
        uint sr = drvWheel(c, r) + DRV_SCRATCH;
        if (DF(sl + 1u).w >= 0.0 && DF(sr + 1u).w >= 0.0)
        {
            float4 a = DF(sl);
            float4 b = DF(sr);
            float kar = asfloat(ar.z);
            float diff = nf(a.x - b.x);
            DSet(sl, float4(a.x, max(nf(a.y - smul(kar, diff)), 0.0), a.zw));
            b = DF(sr);
            DSet(sr, float4(b.x, max(nf(b.y + smul(kar, diff)), 0.0), b.zw));
        }
    }
}

// Sections 3 and 4 for wheel k: tyre, spin, state records, hub forces.
void drvTyreBody(DrvCtx c, uint k, uint step);
void drvTyre(DrvCtx c, uint k, uint step)
{
    // On a rigid body (the ray's proxy shape in the scratch record's .z): its own function.
    if (gMisc[drvWheel(c, k) + DRV_SCRATCH].z != 0u)
    {
        drvTyreBody(c, k, step);
        return;
    }
    float h = STEP;
    float wb = c.c4.x;
    float track = c.c4.y;
    float ack = c.c3.w;
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float4 w4 = DF(wr + 4u);
    float4 w5 = DF(wr + 5u);
    float4 w6 = DF(wr + 6u);
    float4 st = DF(wr + 15u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 a = drvAxle(c, k);
    float4 sc0 = DF(wr + DRV_SCRATCH);
    float4 hitK = DF(wr + DRV_SCRATCH + 1u);
    float lenK = sc0.x;
    float loadK = sc0.y;
    float angle = 0.0;
    if (w3.w > 0.0)
    {
        angle = smul(-c.steer, w3.w);
        if (ack > 0.0 && w6.y != 0.0 && abs(angle) > 1e-4 && wb > 0.0)
        {
            float2 sc = drvSinCos(abs(angle));
            float turnR = sdiv(wb, sdiv(sc.x, sc.y));
            bool inner = (angle > 0.0) == (w6.y > 0.0);
            float off = inner ? -0.5 : 0.5;
            float ak = sAtan2(wb, max(nf(turnR + smul(off, track)), 1e-3));
            angle = smul(drvSgn(angle), drvLerp(abs(angle), ak, ack));
        }
    }
    float3 uw = drvNrm(ssub3(c.up, sscale(a, sdot(a, c.up))));
    float3 f0 = scross(a, uw);
    float2 sca = drvSinCos(angle);
    float3 fwd = sadd3(sscale(f0, sca.y), sscale(scross(uw, f0), sca.x));
    float j = w1.z;
    float r = w1.x;
    float tau = 0.0;
    if ((w0.z & 1u) != 0u) tau = smul(smul(smul(smul(c.forward, c.c3.x), c.scale), w4.x), nf(1.0 - c.cut));
    if (c.top > 0.0 && smul(smul(drvSgn(c.forward), st.x), r) >= c.top) tau = 0.0;
    float wd = nf(smul(st.x, max(nf(1.0 - smul(w1.w, h)), 0.0)) + sdiv(smul(h, tau), j));
    float b = smul(c.brake, w4.y);
    if ((w0.z & 2u) != 0u) b = nf(b + smul(c.hand, w4.z));
    float wNew;
    float kappa = 0.0;
    float alpha = 0.0;
    float fnOut = 0.0;
    bool contact = hitK.w >= 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    if (contact)
    {
        float3 n = hitK.xyz;
        float fn_ = loadK;
        b = nf(b + smul(smul(w4.w, fn_), r));
        float3 t = drvNrm(ssub3(fwd, sscale(n, sdot(n, fwd))));
        float3 sd = scross(n, t);
        uint g1 = c.nodeOff + w0.x;
        float3 fpre = drvForceNow(g1, c.beamOff, c.groupOff, c.groupCount, step);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceNow(c.nodeOff + w0.y, c.beamOff, c.groupOff, c.groupCount, step));
        float vt = sdot(vh, t);
        float vs = sdot(vh, sd);
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
        kappa = sdiv(abs(nf(smul(st.x, r) - vt)), max(abs(vt), 1e-3));
        float vlen = ssqrt(nf(smul(vt, vt) + smul(vs, vs)));
        alpha = vlen < 1e-3 ? 0.0 : smul(sAtan2(abs(vs), abs(vt)), asfloat(0x42652ee1u));
        uint4 counts = gMisc[wr];
        uint nl = counts.w & 0xFFFFu;
        uint nt = counts.w >> 16;
        float muL = max(smul(drvCurve(wr + 7u, nl, max(kappa, w5.z)), w5.x), 0.0);
        muL = ssqrt(smul(muL, c.c5.w));
        float muT = max(smul(drvCurve(wr + 11u, nt, max(alpha, w5.w)), w5.y), 0.0);
        muT = ssqrt(smul(muT, c.c5.w));
        float al = smul(muL, fn_);
        float at = smul(muT, fn_);
        float ftRaw = -nf(fs + sdiv(smul(m, vs), h));
        float flRaw = drvSpin(m, vt, ft, j, r, wd, b, DRV_FLT_MAX).y;
        float alEff = al;
        float fT = ftRaw;
        if (at <= 1e-9)
        {
            fT = 0.0;
        }
        else
        {
            float e = 2.0;
            if (al > 1e-9)
            {
                float ql = sdiv(flRaw, al);
                float qt = sdiv(ftRaw, at);
                e = nf(smul(ql, ql) + smul(qt, qt));
            }
            if (e > 1.0)
            {
                float sc = sdiv(1.0, ssqrt(e));
                float tProp = smul(abs(ftRaw), sc);
                float lProp = smul(abs(flRaw), sc);
                float nx = sdiv(kappa, max(w5.z, 0.03));
                float tanA = asfloat(0x4103cf3au);
                if (vlen < 1e-3) tanA = 0.0;
                else if (abs(vt) > 0.0) tanA = min(sdiv(abs(vs), abs(vt)), asfloat(0x4103cf3au));
                float ny = sdiv(tanA, w6.x);
                float nn = ssqrt(nf(smul(nx, nx) + smul(ny, ny)));
                float ws = nn > 1e-6 ? drvClamp(nf(max(nx, ny) - 1.0), 0.0, 1.0) : 0.0;
                float tCap = drvLerp(tProp, nn > 1e-6 ? sdiv(smul(at, ny), nn) : tProp, ws);
                alEff = drvLerp(lProp, nn > 1e-6 ? sdiv(smul(al, nx), nn) : lProp, ws);
                fT = drvClamp(ftRaw, -tCap, tCap);
            }
            else
            {
                float qt = sdiv(ftRaw, at);
                alEff = smul(al, ssqrt(max(nf(1.0 - smul(qt, qt)), 0.0)));
            }
        }
        float2 sol = drvSpin(m, vt, ft, j, r, wd, b, alEff);
        wNew = sol.x;
        force = sadd3(sadd3(sscale(t, sol.y), sscale(sd, fT)), sscale(n, fn_));
        fnOut = fn_;
    }
    else
    {
        wNew = drvSpin(m, 0.0, 0.0, j, r, wd, b, 0.0).x;
    }
    float spin = nf(st.y + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, angle, lenK));
    DSet(wr + 16u, float4(fnOut, kappa, alpha, contact ? 1.0 : 0.0));
    DSet(wr + 17u, float4(sadd3(c.ow, sadd3(xh, sscale(c.up, nf(w2.z - lenK)))), 0.0));
    DSet(wr + 18u, float4(scross(uw, fwd), 0.0));
    DSet(wr + 19u, float4(uw, 0.0));
    // 4. Onto the hub nodes, by mass (fixed point, read by the node phase).
    if (contact)
    {
        uint g1 = c.nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = c.nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(gNodes[g1 * 3u].w, m)));
            accAdd(g2, sscale(force, sdiv(gNodes[g2 * 3u].w, m)));
        }
    }
}

void drvTyreBody(DrvCtx c, uint k, uint step)
{
    float h = STEP;
    float wb = c.c4.x;
    float track = c.c4.y;
    float ack = c.c3.w;
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float4 w4 = DF(wr + 4u);
    float4 w5 = DF(wr + 5u);
    float4 w6 = DF(wr + 6u);
    float4 st = DF(wr + 15u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 a = drvAxle(c, k);
    float4 sc0 = DF(wr + DRV_SCRATCH);
    float4 hitK = DF(wr + DRV_SCRATCH + 1u);
    uint shape = gMisc[wr + DRV_SCRATCH].z - 1u;
    float lenK = sc0.x;
    float loadK = sc0.y;
    float angle = 0.0;
    if (w3.w > 0.0)
    {
        angle = smul(-c.steer, w3.w);
        if (ack > 0.0 && w6.y != 0.0 && abs(angle) > 1e-4 && wb > 0.0)
        {
            float2 sc = drvSinCos(abs(angle));
            float turnR = sdiv(wb, sdiv(sc.x, sc.y));
            bool inner = (angle > 0.0) == (w6.y > 0.0);
            float off = inner ? -0.5 : 0.5;
            float ak = sAtan2(wb, max(nf(turnR + smul(off, track)), 1e-3));
            angle = smul(drvSgn(angle), drvLerp(abs(angle), ak, ack));
        }
    }
    float3 uw = drvNrm(ssub3(c.up, sscale(a, sdot(a, c.up))));
    float3 f0 = scross(a, uw);
    float2 sca = drvSinCos(angle);
    float3 fwd = sadd3(sscale(f0, sca.y), sscale(scross(uw, f0), sca.x));
    float j = w1.z;
    float r = w1.x;
    float tau = 0.0;
    if ((w0.z & 1u) != 0u) tau = smul(smul(smul(smul(c.forward, c.c3.x), c.scale), w4.x), nf(1.0 - c.cut));
    if (c.top > 0.0 && smul(smul(drvSgn(c.forward), st.x), r) >= c.top) tau = 0.0;
    float wd = nf(smul(st.x, max(nf(1.0 - smul(w1.w, h)), 0.0)) + sdiv(smul(h, tau), j));
    float b = smul(c.brake, w4.y);
    if ((w0.z & 2u) != 0u) b = nf(b + smul(c.hand, w4.z));
    float wNew;
    float kappa = 0.0;
    float alpha = 0.0;
    float fnOut = 0.0;
    bool contact = hitK.w >= 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    if (contact)
    {
        float3 n = hitK.xyz;
        // The two-body form against the body's surface at the contact point.
        float3 cp = ssub3(sadd3(xh, sscale(c.up, w2.z)), sscale(c.up, hitK.w));
        float3 vb;
        float2 ms;
        bool onPartner = gMisc[wr + DRV_SCRATCH].z == DRV_PARTNER_HIT;
        if (onPartner) ms = drvPartnerSurface(c, k, m, vb);
        else ms = drvSurface(c, k, shape, n, cp, m, vb);
        float mHub = m;
        m = ms.x;
        vh = ssub3(vh, vb);
        float fn_ = loadK;
        b = nf(b + smul(smul(w4.w, fn_), r));
        float3 t = drvNrm(ssub3(fwd, sscale(n, sdot(n, fwd))));
        float3 sd = scross(n, t);
        uint g1 = c.nodeOff + w0.x;
        float3 fpre = drvForceNow(g1, c.beamOff, c.groupOff, c.groupCount, step);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceNow(c.nodeOff + w0.y, c.beamOff, c.groupOff, c.groupCount, step));
        fpre = sscale(fpre, ms.y);
        float vt = sdot(vh, t);
        float vs = sdot(vh, sd);
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
        kappa = sdiv(abs(nf(smul(st.x, r) - vt)), max(abs(vt), 1e-3));
        float vlen = ssqrt(nf(smul(vt, vt) + smul(vs, vs)));
        alpha = vlen < 1e-3 ? 0.0 : smul(sAtan2(abs(vs), abs(vt)), asfloat(0x42652ee1u));
        uint4 counts = gMisc[wr];
        uint nl = counts.w & 0xFFFFu;
        uint nt = counts.w >> 16;
        float muL = max(smul(drvCurve(wr + 7u, nl, max(kappa, w5.z)), w5.x), 0.0);
        muL = ssqrt(smul(muL, c.c5.w));
        float muT = max(smul(drvCurve(wr + 11u, nt, max(alpha, w5.w)), w5.y), 0.0);
        muT = ssqrt(smul(muT, c.c5.w));
        float al = smul(muL, fn_);
        float at = smul(muT, fn_);
        float ftRaw = -nf(fs + sdiv(smul(m, vs), h));
        float flRaw = drvSpin(m, vt, ft, j, r, wd, b, DRV_FLT_MAX).y;
        float alEff = al;
        float fT = ftRaw;
        if (at <= 1e-9)
        {
            fT = 0.0;
        }
        else
        {
            float e = 2.0;
            if (al > 1e-9)
            {
                float ql = sdiv(flRaw, al);
                float qt = sdiv(ftRaw, at);
                e = nf(smul(ql, ql) + smul(qt, qt));
            }
            if (e > 1.0)
            {
                float sc = sdiv(1.0, ssqrt(e));
                float tProp = smul(abs(ftRaw), sc);
                float lProp = smul(abs(flRaw), sc);
                float nx = sdiv(kappa, max(w5.z, 0.03));
                float tanA = asfloat(0x4103cf3au);
                if (vlen < 1e-3) tanA = 0.0;
                else if (abs(vt) > 0.0) tanA = min(sdiv(abs(vs), abs(vt)), asfloat(0x4103cf3au));
                float ny = sdiv(tanA, w6.x);
                float nn = ssqrt(nf(smul(nx, nx) + smul(ny, ny)));
                float ws = nn > 1e-6 ? drvClamp(nf(max(nx, ny) - 1.0), 0.0, 1.0) : 0.0;
                float tCap = drvLerp(tProp, nn > 1e-6 ? sdiv(smul(at, ny), nn) : tProp, ws);
                alEff = drvLerp(lProp, nn > 1e-6 ? sdiv(smul(al, nx), nn) : lProp, ws);
                fT = drvClamp(ftRaw, -tCap, tCap);
            }
            else
            {
                float qt = sdiv(ftRaw, at);
                alEff = smul(al, ssqrt(max(nf(1.0 - smul(qt, qt)), 0.0)));
            }
        }
        float2 sol = drvSpin(m, vt, ft, j, r, wd, b, alEff);
        wNew = sol.x;
        force = sadd3(sadd3(sscale(t, sol.y), sscale(sd, fT)), sscale(n, fn_));
        fnOut = fn_;
        m = mHub;
        // Equal and opposite on the body, at the contact point (its step sums); on
        // another vehicle's triangle staged for its nodes before the next step.
        float3 react = -force;
        if (onPartner)
        {
            DSet(drvPartnerAt(c.v, k) + 3u, float4(react, asfloat(1u)));
        }
        else
        {
        uint sat = cplShape(c.v, shape);
        float3 mo = scross(ssub3(cp, asfloat(gMisc[sat + 6u].xyz)), react);
        uint sacc = cplAcc(c.v) + cplB() + 6u * shape;
        gAcc[sacc].add(toFixed(react.x));
        gAcc[sacc + 1u].add(toFixed(react.y));
        gAcc[sacc + 2u].add(toFixed(react.z));
        gAcc[sacc + 3u].add(toFixed(mo.x));
        gAcc[sacc + 4u].add(toFixed(mo.y));
        gAcc[sacc + 5u].add(toFixed(mo.z));
        }
    }
    else
    {
        wNew = drvSpin(m, 0.0, 0.0, j, r, wd, b, 0.0).x;
    }
    float spin = nf(st.y + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, angle, lenK));
    DSet(wr + 16u, float4(fnOut, kappa, alpha, contact ? 1.0 : 0.0));
    DSet(wr + 17u, float4(sadd3(c.ow, sadd3(xh, sscale(c.up, nf(w2.z - lenK)))), 0.0));
    DSet(wr + 18u, float4(scross(uw, fwd), 0.0));
    DSet(wr + 19u, float4(uw, 0.0));
    // 4. Onto the hub nodes, by mass (fixed point, read by the node phase).
    if (contact)
    {
        uint g1 = c.nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = c.nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(gNodes[g1 * 3u].w, m)));
            accAdd(g2, sscale(force, sdiv(gNodes[g2 * 3u].w, m)));
        }
    }
}

// The vehicle's input and cut state (after its wheels).
void drvEnd(DrvCtx c)
{
    DSet(c.vr + 1u, float4(c.forward, c.brake, c.hand, c.steer));
    DSet(c.vr + 6u, float4(c.c6.x, c.cut, c.speed, asfloat(c.arCount)));
}

// Whether no two wheels share a hub node (then a wheel per lane gives the same bits).
bool drvWheelsApart(DrvCtx c)
{
    bool apart = true;
    for (uint a = 0u, aStop = EB_END(297, a, c.nw, 1u); a < aStop; a++)
    {
        uint4 wa = gMisc[drvWheel(c, a)];
        for (uint b = a + 1u, bStop = EB_END(298, b, c.nw, 1u); b < bStop; b++)
        {
            uint4 wb = gMisc[drvWheel(c, b)];
            if (wa.x == wb.x || wa.x == wb.y || wa.y == wb.x || wa.y == wb.y) apart = false;
        }
    }
    return apart;
}

// The frame's start (lane 0, before the barrier the uniform read follows).
void driveBegin()
{
    gsUniformDrive = gMisc[driveRec()].x;
}

// Wheel `w` of vehicle v (live: the lane has one) at step `step`, the three parts of
// a step, each followed by a barrier (every lane reaches them). Wheels apart: wheel w
// on lane w; else lane 0 runs every wheel, in explicit_drive::drive_pass's order (the
// rays, the bars, the tyres, the end: the barriers between change nothing on one
// lane). One call site per part (the compilers inline every call).
void driveWheelsSplit(uint v, uint w, bool live, uint step)
{
    DrvCtx c;
    bool has = live && w < DRV_MAX_WHEELS && drvCtx(v, c);
    c.step = step;
    bool apart = has && drvWheelsApart(c);
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    for (uint k = k0, kStop = EB_END(299, k, k1, 1u); k < kStop; k++) drvRays(c, k);
    sync();
    if (has && w == 0u) drvBars(c);
    sync();
    for (uint k = k0, kStop = EB_END(300, k, k1, 1u); k < kStop; k++) drvTyre(c, k, step);
    if (has && w == 0u) drvEnd(c);
}

