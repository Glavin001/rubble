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
    for (uint i = 1u, iStop = EB_END(288, i, count, 1u); i < iStop; i++)
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

// `p` in the vehicle's frame (origin `ow`): the ground's height in that frame; the box
// slots in the World's (the ray's start moved to the World).
float4 drvRay(float3 p, float3 ow, float3 d, float tMax, float worldGround, uint boxOff, uint boxCount)
{
    float4 best = drvNone();
    float ground = nf(worldGround - ow.y);
    if (d.y < -DRV_PARALLEL && p.y >= ground) best = float4(0.0, 1.0, 0.0, sdiv(nf(ground - p.y), d.y));
    float3 pw = sadd3(ow, p);
    for (uint b = 0u, bStop = EB_END(289, b, boxCount, 1u); b < bStop; b++) best = drvCloser(best, drvSlot(boxOff + b, pw, d));
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
    uint v;
    uint step; // the step (driveWheelsSplit's; 0 on the ahead schedules)
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
    // The wheels' frame: the vehicle's (o is 0: positions relative to its origin); ow
    // its origin, added at the World's boundary (ground, box slots, reported centres).
    float3 o;
    float3 ow;
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
    c.v = v;
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
    c.ow = asfloat(gMisc[v].xyz);
    c.o = float3(0.0, 0.0, 0.0);
    float h = STEP;
    float3 xc = sadd3(c.o, gNodes[(c.nodeOff + c2.y) * 3u].xyz);
    float3 xb = sadd3(c.o, gNodes[(c.nodeOff + c2.z) * 3u].xyz);
    float3 xl = sadd3(c.o, gNodes[(c.nodeOff + c2.w) * 3u].xyz);
    c.fw = drvNrm(ssub3(xc, xb));
    c.up = drvNrm(scross(c.fw, ssub3(xl, xc)));
    c.left = scross(c.up, c.fw);
    float msum = 0.0;
    float3 psum = float3(0.0, 0.0, 0.0);
    for (uint k = 0u, kStop = EB_END(290, k, nw, 1u); k < kStop; k++)
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
        for (uint k = 0u, kStop = EB_END(291, k, nw, 1u); k < kStop; k++)
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

// --- Wheels on rigid bodies (engine_softbody::explicit_drive DriveConfig::see_bodies;
// the drive flags' bit 8) ---
// With the vehicle's coupling on, the rays also meet its proxy shapes at their bodies'
// poses now (after the step's integration: coupleStep runs before driveStep), the
// nearest of every surface; a wheel on a body takes the two-body form (the hub's
// velocity relative to the body's surface at the contact point, the reduced mass and
// the share of the hub's force, the body's inverse mass at the point split among this
// vehicle's wheels on it and the vehicles holding it), and the wheel force's reaction
// goes onto the shape's step sums at the contact point (fixed point, as node contact's).
// Positions are in the vehicle frame, as the coupling's records. The scratch record's
// .z holds the shape + 1 (0: a static surface). Its own functions behind early returns.
bool drvBodiesFlag(DrvCtx c) { return (c.c5y & 8u) != 0u && (gMisc[coupleRec()].x != 0u || gMisc[driveRec()].w != 0u); }
uint drvPartnerBase();
// (The coupling on, or other vehicles' records: wheelsPartners.)
bool drvSeeBodies(DrvCtx c) { return (c.c5y & 8u) != 0u && (gsCpl[0] != 0u || drvPartnerBase() != 0u); }

#include "explicit_beams_drive_partners.slang"
#include "explicit_beams_drive_ahead.slang"
