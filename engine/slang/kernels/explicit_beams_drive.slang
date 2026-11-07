// Raycast wheels for explicit vehicles (CPU reference engine_softbody::explicit_drive,
// records engine_exec::explicit_plan::drive). Included by explicit_beams.slang.
//
// Off (the drive header's .x is 0: no vehicle has raycast wheels) nothing here runs.
// On, at the start of every step, before the node phase: lane 0 runs each vehicle's
// wheels (rays, suspension, tyres, spin) in the CPU's order and adds the wheel forces
// to the hub nodes' contact accumulators (fixed point, as self-collision's), sets
// gsHit so the node phase reads them, and a barrier follows. The barrier is under
// gsUniformDrive (workgroupUniformLoad, uniform). Strict arithmetic in the CPU's order.

groupshared uint gsUniformDrive;

static const uint DRV_VEH = 20u;
static const uint DRV_WHEEL = 23u;
static const uint DRV_MAX_WHEELS = 16u;
static const float DRV_PARALLEL = 1e-12;
static const float DRV_FLT_MAX = 3.40282347e38;

uint driveRec() { return coupleRec() + 2u; }
uint driveBase(uint v) { return gMisc[driveRec()].y + v * gMisc[driveRec()].z; }
float4 DF(uint r) { return asfloat(gMisc[r]); }
void DSet(uint r, float4 x) { gMisc[r] = asuint(x); }

float drvSgn(float x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
float drvClamp(float x, float lo, float hi) { return min(max(x, lo), hi); }
float drvLerp(float a, float b, float t) { return nf(a + smul(nf(b - a), t)); }
float3 drvNrm(float3 v)
{
    float l = ssqrt(sdot(v, v));
    if (l > 0.0) return sdiv3(v, l);
    return v;
}
float drvMoveToward(float cur, float target, float maxDelta)
{
    if (maxDelta <= 0.0) return target;
    float d = nf(target - cur);
    if (abs(d) <= maxDelta) return target;
    return nf(cur + smul(drvSgn(d), maxDelta));
}

// Curve of `count` points from record `at` (two per record): explicit_drive::curve_eval.
float2 drvPoint(uint at, uint k)
{
    float4 r = DF(at + k / 2u);
    return (k % 2u) == 0u ? r.xy : r.zw;
}
float drvCurve(uint at, uint count, float x)
{
    if (count == 0u) return 1.0;
    float2 p0 = drvPoint(at, 0u);
    if (x <= p0.x) return p0.y;
    for (uint i = 1u, iStop = EB_END(i, count, 1u); i < iStop; i++)
    {
        float2 a = drvPoint(at, i - 1u);
        float2 b = drvPoint(at, i);
        if (x <= b.x)
        {
            float t = sdiv(nf(x - a.x), max(nf(b.x - a.x), 1e-9));
            return nf(a.y + smul(nf(b.y - a.y), t));
        }
    }
    return drvPoint(at, count - 1u).y;
}

// explicit_drive::sin_cos.
float2 drvSinCos(float x)
{
    float x2 = smul(x, x);
    float s = asfloat(0xb2d7322bu);
    s = nf(smul(s, x2) + asfloat(0x3638ef1du));
    s = nf(smul(s, x2) + asfloat(0xb9500d01u));
    s = nf(smul(s, x2) + asfloat(0x3c088889u));
    s = nf(smul(s, x2) + asfloat(0xbe2aaaabu));
    s = nf(smul(s, x2) + 1.0);
    float c = asfloat(0x310f76c7u);
    c = nf(smul(c, x2) + asfloat(0xb493f27eu));
    c = nf(smul(c, x2) + asfloat(0x37d00d01u));
    c = nf(smul(c, x2) + asfloat(0xbab60b61u));
    c = nf(smul(c, x2) + asfloat(0x3d2aaaabu));
    c = nf(smul(c, x2) + asfloat(0xbf000000u));
    c = nf(smul(c, x2) + 1.0);
    return float2(smul(x, s), c);
}

// --- rays (t, normal); .w < 0: no hit ---
float4 drvNone() { return float4(0.0, 0.0, 0.0, -1.0); }

// explicit_drive::slab: the ray (pl, dl) in a frame of axes a0..a2 against lo..hi.
float4 drvSlab(float3 pl, float3 dl, float3 lo, float3 hi, float3 a0, float3 a1, float3 a2)
{
    float tIn = -DRV_FLT_MAX;
    float tOut = DRV_FLT_MAX;
    float3 n = float3(0.0, 0.0, 0.0);
    for (uint a = 0u; a < 3u; a++)
    {
        float p = pl[a];
        float d = dl[a];
        float l = lo[a];
        float h = hi[a];
        if (abs(d) < DRV_PARALLEL)
        {
            if (p < l || p > h) return drvNone();
            continue;
        }
        float t1 = sdiv(nf(l - p), d);
        float t2 = sdiv(nf(h - p), d);
        float nearT = t1 < t2 ? t1 : t2;
        float farT = t1 < t2 ? t2 : t1;
        float sg = t1 < t2 ? -1.0 : 1.0;
        if (nearT > tIn)
        {
            tIn = nearT;
            float3 ax = a == 0u ? a0 : (a == 1u ? a1 : a2);
            n = sscale(ax, sg);
        }
        if (farT < tOut) tOut = farT;
    }
    if (tIn >= 0.0 && tIn <= tOut) return float4(n, tIn);
    return drvNone();
}

float4 drvSphere(float3 p, float3 d, float3 c, float r)
{
    float3 oc = ssub3(p, c);
    float b = sdot(oc, d);
    float cc = nf(sdot(oc, oc) - smul(r, r));
    if (cc <= 0.0) return drvNone();
    float disc = nf(smul(b, b) - cc);
    if (disc < 0.0) return drvNone();
    float t = nf(-b - ssqrt(disc));
    if (t < 0.0) return drvNone();
    return float4(drvNrm(ssub3(sadd3(p, sscale(d, t)), c)), t);
}

// explicit_drive::ray_tube: (t, axial coordinate) in .xy, normal in .yzw of a pair.
bool drvTube(float3 p, float3 d, float3 c, float3 a, float r, out float t, out float y, out float3 n)
{
    t = 0.0; y = 0.0; n = float3(0.0, 0.0, 0.0);
    float3 oc = ssub3(p, c);
    float da = sdot(d, a);
    float oa = sdot(oc, a);
    float3 dp = ssub3(d, sscale(a, da));
    float3 op = ssub3(oc, sscale(a, oa));
    float qa = sdot(dp, dp);
    if (qa < DRV_PARALLEL) return false;
    float qb = sdot(op, dp);
    float qc = nf(sdot(op, op) - smul(r, r));
    if (qc <= 0.0) return false;
    float disc = nf(smul(qb, qb) - smul(qa, qc));
    if (disc < 0.0) return false;
    t = sdiv(nf(-qb - ssqrt(disc)), qa);
    if (t < 0.0) return false;
    y = nf(oa + smul(da, t));
    n = drvNrm(sadd3(op, sscale(dp, t)));
    return true;
}

float4 drvCloser(float4 best, float4 h)
{
    if (h.w >= 0.0 && (best.w < 0.0 || h.w < best.w)) return h;
    return best;
}

// One box slot (the frame's slots: boxes, then shapes; explicit_drive::ray_cast).
float4 drvSlot(uint slot, float3 p, float3 d)
{
    uint4 lo4 = gStatic[gP.boxBase + slot * 2u];
    uint4 hi4 = gStatic[gP.boxBase + slot * 2u + 1u];
    float3 X = float3(1.0, 0.0, 0.0), Y = float3(0.0, 1.0, 0.0), Z = float3(0.0, 0.0, 1.0);
    if (lo4.w == 0u) return drvSlab(p, d, asfloat(lo4.xyz), asfloat(hi4.xyz), X, Y, Z);
    uint at = gP.boxBase + hi4.w;
    float4 p0 = asfloat(gStatic[at]);
    float4 p1 = asfloat(gStatic[at + 1u]);
    if (lo4.w == SHAPE_OBOX)
    {
        float4 p2 = asfloat(gStatic[at + 2u]);
        float4 p3 = asfloat(gStatic[at + 3u]);
        float3 h = float3(p0.w, p1.w, p2.w);
        float3 o = ssub3(p, p0.xyz);
        float3 pl = float3(sdot(p1.xyz, o), sdot(p2.xyz, o), sdot(p3.xyz, o));
        float3 dl = float3(sdot(p1.xyz, d), sdot(p2.xyz, d), sdot(p3.xyz, d));
        return drvSlab(pl, dl, -h, h, p1.xyz, p2.xyz, p3.xyz);
    }
    if (lo4.w == SHAPE_CYLINDER)
    {
        float3 c = p0.xyz;
        float radius = p0.w;
        float hh = p1.x;
        float4 best = drvNone();
        if (abs(d.y) >= DRV_PARALLEL)
        {
            float y = d.y < 0.0 ? nf(c.y + hh) : nf(c.y - hh);
            float ny = d.y < 0.0 ? 1.0 : -1.0;
            float t = sdiv(nf(y - p.y), d.y);
            float3 q = ssub3(sadd3(p, sscale(d, t)), c);
            if (t >= 0.0 && nf(smul(q.x, q.x) + smul(q.z, q.z)) <= smul(radius, radius)) best = float4(0.0, ny, 0.0, t);
        }
        float t, y;
        float3 n;
        if (drvTube(p, d, c, Y, radius, t, y, n) && y >= -hh && y <= hh) best = drvCloser(best, float4(n, t));
        return best;
    }
    if (lo4.w == SHAPE_CAPSULE)
    {
        float3 a = p0.xyz;
        float3 b = p1.xyz;
        float radius = p0.w;
        float4 best = drvSphere(p, d, a, radius);
        best = drvCloser(best, drvSphere(p, d, b, radius));
        float3 ab = ssub3(b, a);
        float l = ssqrt(sdot(ab, ab));
        if (l > 0.0)
        {
            float t, y;
            float3 n;
            if (drvTube(p, d, a, sdiv3(ab, l), radius, t, y, n) && y >= 0.0 && y <= l) best = drvCloser(best, float4(n, t));
        }
        return best;
    }
    return drvSphere(p, d, p0.xyz, p0.w);
}

float4 drvRay(float3 p, float3 d, float tMax, float ground, uint boxOff, uint boxCount)
{
    float4 best = drvNone();
    if (d.y < -DRV_PARALLEL && p.y >= ground) best = float4(0.0, 1.0, 0.0, sdiv(nf(ground - p.y), d.y));
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++) best = drvCloser(best, drvSlot(boxOff + b, p, d));
    if (best.w > tMax) return drvNone();
    return best;
}

// explicit_drive::spin_solve: (new spin, tyre force).
float2 drvSpin(float m, float u, float ft, float j, float r, float wd, float b, float a)
{
    float h = STEP;
    float f0 = drvClamp(nf(sdiv(smul(m, nf(smul(0.0, r) - u)), h) - ft), -a, a);
    float g0 = nf(sdiv(smul(j, nf(0.0 - wd)), h) + smul(r, f0));
    if (nf(g0 - b) <= 0.0 && nf(g0 + b) >= 0.0) return float2(0.0, f0);
    float s = nf(g0 + b) < 0.0 ? 1.0 : -1.0;
    float num = nf(nf(nf(smul(j, wd) + smul(smul(r, m), u)) + smul(smul(h, r), ft)) - smul(smul(h, s), b));
    float den = nf(j + smul(smul(m, r), r));
    float wu = sdiv(num, den);
    float fu = nf(sdiv(smul(m, nf(smul(wu, r) - u)), h) - ft);
    float w = wu;
    float f = fu;
    if (!(abs(fu) <= a))
    {
        f = smul(drvSgn(fu), a);
        w = nf(wd - sdiv(smul(h, nf(smul(r, f) + smul(s, b))), j));
    }
    if (smul(s, w) < 0.0) w = 0.0;
    return float2(w, f);
}

// The node phase's force on node gi before the wheels (beamSum + contact sums).
float3 drvForceNow(uint gi, uint beamOff, uint groupOff, uint groupCount, uint step)
{
    uint4 info = gStatic[gP.nodeInfoBase + gi];
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, false);
    return f + accForce(gi);
}

// One vehicle's wheels (explicit_drive::drive_pass) in four parts, each in the CPU's
// order: drvCtx (frame, speed, inputs: section 1), drvRays (one wheel's ray and
// suspension: section 2), drvBars (the anti-roll bars), drvTyre (one wheel's tyre,
// spin and hub forces: sections 3 and 4) and drvEnd. driveWheelsSplit runs them on
// one lane, or a wheel per lane when no two wheels share a hub node (a
// wheel's tyre reads its hub nodes' accumulators, which only its own forces then
// touch, and its own records), else on one lane: the same arithmetic, so the same
// bits. Between the parts each wheel's (length, load) and hit are kept in its scratch
// records (DRV_SCRATCH).
static const uint DRV_SCRATCH = 21u;

struct DrvCtx
{
    uint vr;
    uint nw;
    uint nodeOff;
    uint beamOff;
    uint boxOff;
    uint boxCount;
    uint groupOff;
    uint groupCount;
    uint c5y;
    uint arCount;
    float ground;
    float3 o;
    float3 fw;
    float3 up;
    float3 left;
    float speed;
    float forward;
    float brake;
    float hand;
    float steer;
    float scale;
    float top;
    float cut;
    float4 c3;
    float4 c4;
    float4 c5;
    float4 c6;
};

uint drvWheel(DrvCtx c, uint k) { return c.vr + DRV_VEH + k * DRV_WHEEL; }

// Wheel k's hub: (mass, position, velocity) (section 1's per-wheel terms).
void drvHub(DrvCtx c, uint k, out float m, out float3 x, out float3 vel)
{
    uint4 w0 = gMisc[drvWheel(c, k)];
    uint g1 = c.nodeOff + w0.x;
    uint g2 = c.nodeOff + w0.y;
    float4 P1 = gNodes[g1 * 3u];
    float3 V1 = gNodes[g1 * 3u + 1u].xyz;
    if (w0.x == w0.y)
    {
        m = P1.w;
        x = sadd3(c.o, P1.xyz);
        vel = V1;
    }
    else
    {
        float4 P2 = gNodes[g2 * 3u];
        float3 V2 = gNodes[g2 * 3u + 1u].xyz;
        m = nf(P1.w + P2.w);
        x = sdiv3(sadd3(sscale(sadd3(c.o, P1.xyz), P1.w), sscale(sadd3(c.o, P2.xyz), P2.w)), m);
        vel = sdiv3(sadd3(sscale(V1, P1.w), sscale(V2, P2.w)), m);
    }
}

// Section 1: the frame, speed and inputs (false: no wheels).
bool drvCtx(uint v, out DrvCtx c)
{
    c = (DrvCtx)0;
    uint vr = driveBase(v);
    c.vr = vr;
    uint4 c2 = gMisc[vr + 2u];
    uint nw = min(c2.x, DRV_MAX_WHEELS);
    c.nw = nw;
    if (nw == 0u) return false;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    c.nodeOff = r0.x;
    c.beamOff = r0.z;
    c.boxOff = r1.z;
    c.boxCount = r1.w;
    c.groupOff = r2.z;
    c.groupCount = r2.w;
    c.ground = asfloat(r3.y);
    c.o = asfloat(gMisc[v].xyz);
    float h = STEP;
    float3 xc = sadd3(c.o, gNodes[(c.nodeOff + c2.y) * 3u].xyz);
    float3 xb = sadd3(c.o, gNodes[(c.nodeOff + c2.z) * 3u].xyz);
    float3 xl = sadd3(c.o, gNodes[(c.nodeOff + c2.w) * 3u].xyz);
    c.fw = drvNrm(ssub3(xc, xb));
    c.up = drvNrm(scross(c.fw, ssub3(xl, xc)));
    c.left = scross(c.up, c.fw);
    float msum = 0.0;
    float3 psum = float3(0.0, 0.0, 0.0);
    for (uint k = 0u, kStop = EB_END(k, nw, 1u); k < kStop; k++)
    {
        float m;
        float3 x, vel;
        drvHub(c, k, m, x, vel);
        msum = nf(msum + m);
        psum = sadd3(psum, sscale(vel, m));
    }
    float3 vm = sdiv3(psum, msum);
    float speed = sdot(vm, c.fw);
    c.speed = speed;
    float4 inp = DF(vr);
    float4 sm = DF(vr + 1u);
    c.c3 = DF(vr + 3u);
    c.c4 = DF(vr + 4u);
    uint4 c5u = gMisc[vr + 5u];
    c.c5 = asfloat(c5u);
    c.c5y = c5u.y;
    c.c6 = DF(vr + 6u);
    c.arCount = min(gMisc[vr + 6u].w, 8u);
    uint speedPts = gMisc[vr + 7u].x;
    float fin = drvClamp(inp.x, -1.0, 1.0);
    float sin_ = drvClamp(inp.y, -1.0, 1.0);
    float bin = drvClamp(inp.z, 0.0, 1.0);
    float hin = drvClamp(inp.w, 0.0, 1.0);
    if ((c5u.y & 1u) != 0u && smul(fin, speed) < 0.0 && abs(speed) > 1.0)
    {
        bin = max(bin, abs(fin));
        fin = 0.0;
    }
    c.forward = drvMoveToward(sm.x, fin, smul(c.c4.z, h));
    c.brake = drvMoveToward(sm.y, bin, smul(c.c4.w, h));
    c.hand = drvMoveToward(sm.z, hin, smul(c.c5.x, h));
    float target = smul(sin_, drvCurve(vr + 8u, speedPts, abs(speed)));
    c.steer = drvMoveToward(sm.w, target, smul(c.c3.z, h));
    float sp = abs(speed);
    c.scale = 1.0;
    c.top = c.c3.y;
    if (c.top > 0.0 && smul(c.forward, speed) > 0.0)
        c.scale = sp < sdiv(c.top, 3.0) ? 1.0 : drvClamp(sdiv(nf(c.top - sp), sdiv(smul(c.top, 2.0), 3.0)), 0.0, 1.0);
    c.cut = c.c6.y;
    if ((c5u.y & 2u) != 0u)
    {
        float vReg = sdiv(sp, max(sp, 3.0));
        float kMax = 0.0;
        for (uint k = 0u, kStop = EB_END(k, nw, 1u); k < kStop; k++)
        {
            uint wr = drvWheel(c, k);
            if ((gMisc[wr].z & 1u) != 0u && DF(wr + 16u).w != 0.0)
                kMax = max(kMax, sdiv(smul(DF(wr + 16u).y, vReg), max(DF(wr + 5u).z, 0.03)));
        }
        float cutTarget = nf(1.0 - drvClamp(nf(1.0 - sdiv(nf(kMax - 0.8), 1.0)), 0.2, 1.0));
        c.cut = drvLerp(c.cut, cutTarget, c.c6.x);
    }
    return true;
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
void drvRays(DrvCtx c, uint k)
{
    uint wr = drvWheel(c, k);
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 hp = sadd3(xh, sscale(c.up, w2.z));
    float4 ht = drvRay(hp, -c.up, nf(w2.y + w1.x), c.ground, c.boxOff, c.boxCount);
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
    for (uint i = 0u, iStop = EB_END(i, c.arCount, 1u); i < iStop; i++)
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
void drvTyre(DrvCtx c, uint k, uint step)
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
    DSet(wr + 17u, float4(sadd3(xh, sscale(c.up, nf(w2.z - lenK))), 0.0));
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
    for (uint a = 0u, aStop = EB_END(a, c.nw, 1u); a < aStop; a++)
    {
        uint4 wa = gMisc[drvWheel(c, a)];
        for (uint b = a + 1u, bStop = EB_END(b, c.nw, 1u); b < bStop; b++)
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
    bool apart = has && drvWheelsApart(c);
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    for (uint k = k0, kStop = EB_END(k, k1, 1u); k < kStop; k++) drvRays(c, k);
    sync();
    if (has && w == 0u) drvBars(c);
    sync();
    for (uint k = k0, kStop = EB_END(k, k1, 1u); k < kStop; k++) drvTyre(c, k, step);
    if (has && w == 0u) drvEnd(c);
}

// The drive's part of step s of vehicle v (uniform `on`): its wheels, then a barrier.
void driveStep(uint v, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    driveWheelsSplit(v, lane, true, gMisc[v].w + s);
    if (lane == 0u) gsHit = 1u;
    sync();
}

// driveStep for a contact group's k members (their node phases read the
// accumulators every step), DRV_MAX_WHEELS lanes a member.
void driveStepMembers(uint off, uint k, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    uint per = LANES / DRV_MAX_WHEELS;
    for (uint p = 0u, pStop = EB_END(p, k, per); p < pStop; p += per)
    {
        uint m = p + lane / DRV_MAX_WHEELS;
        bool live = m < k;
        uint v = live ? vvMember(off, m) : 0u;
        driveWheelsSplit(v, lane % DRV_MAX_WHEELS, live, live ? gMisc[v].w + s : 0u);
    }
    sync();
}
