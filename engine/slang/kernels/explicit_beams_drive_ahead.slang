// The wheels ahead (perfFlags bit 2, the solo frame): one part of driveWheelsSplit for
// wheel lane `w` (0: the lane of the bars and the end), no barrier here. The rays and the
// bars read the nodes' positions and velocities (final from a step's node phase to the
// next), static shapes and the drive's records, so step s + 1's run beside step s's beam
// phase and after it; the tyres (which read their hubs' forces: the last beam phase's
// sums and the accumulators after selfApply) stay at the step's start, beside the
// coupling's first phase, which reads and writes none of it. The same calls on the same
// values as driveWheelsSplit.
// (The context once for a step's three parts where they run in one invocation: drvCtx
// reads the records drvEnd wrote and the nodes, neither of which changes between a
// step's rays and its tyres.)
void driveAheadCtx(uint v, uint w, out DrvCtx c, out bool has, out bool apart)
{
    has = w < DRV_MAX_WHEELS && drvCtx(v, c);
    apart = has && drvWheelsApart(c);
}
void driveAheadPart(DrvCtx c, bool has, bool apart, uint w, uint part, uint step)
{
    // The ahead schedules read the proxies before the step's integration: a host turns
    // them off while wheels see bodies (ExplicitBeamsGpu); a frame that still takes them
    // fails loudly, never with stale bodies.
    if (has && w == 0u && part == 0u && drvBodiesFlag(c)) ebLoopCapHit(301);
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    if (part == 1u)
    {
        if (has && w == 0u) drvBars(c);
        return;
    }
    // (One call site for the rays and one for the tyres, as driveWheelsSplit.)
    for (uint k = k0, kStop = EB_END(302, k, k1, 1u); k < kStop; k++)
    {
        if (part == 0u) drvRays(c, k);
        else drvTyre(c, k, step);
    }
    if (part == 2u && has && w == 0u) drvEnd(c);
}

// --- The tyres split at their hubs' forces (perfFlags 262144, PERF_TYRE_AHEAD, with
// PERF_AHEAD_APART and PERF_WIDE_FUSED; explicit_plan::PERF_TYRE_AHEAD) ---
// drvTyre reads its hubs' forces (drvForceNow: the beam sums and the contact
// accumulators after the step's hits) and nothing else that the step's walk, pairs and
// hits change. drvTyrePre runs everything before that read and every value it does not
// feed (the frame, the hub, the axle, the steer angle, the tyre's frame, spin drive and
// brake, the slip ratio and angle, the friction curves, the friction ellipse's shape;
// the outputs 17..19; the end), on the coupling's and wheels' own workgroup after the
// bars (wideAheadApart); drvTyrePost, on the hits kernel's lanes after selfApply, reads
// the hubs' forces and runs the rest: the same functions on the same values in the same
// order (the values cross in float records, exactly), so the same bits as drvTyre. The
// records: DRV_PRE a wheel after the vehicle's wheels (DriveLayout::stride_for), cleared
// by drvTyrePost (a frame leaves them zero, as without the split).
static const uint DRV_PRE = 8u;
uint drvPreAt(uint vr, uint k)
{
    uint nwMax = (gMisc[driveRec()].z - DRV_VEH) / (DRV_WHEEL + DRV_PRE);
    return vr + DRV_VEH + DRV_WHEEL * nwMax + DRV_PRE * k;
}
void drvTyrePre(DrvCtx c, uint k, bool apart)
{
    if (drvBodiesFlag(c)) ebLoopCapHit(303);
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
    bool contact = hitK.w >= 0.0;
    uint p = drvPreAt(c.vr, k);
    if (contact)
    {
        float3 n = hitK.xyz;
        float fn_ = loadK;
        b = nf(b + smul(smul(w4.w, fn_), r));
        float3 t = drvNrm(ssub3(fwd, sscale(n, sdot(n, fwd))));
        float3 sd = scross(n, t);
        float vt = sdot(vh, t);
        float vs = sdot(vh, sd);
        float kappa = sdiv(abs(nf(smul(st.x, r) - vt)), max(abs(vt), 1e-3));
        float vlen = ssqrt(nf(smul(vt, vt) + smul(vs, vs)));
        float alpha = vlen < 1e-3 ? 0.0 : smul(sAtan2(abs(vs), abs(vt)), asfloat(0x42652ee1u));
        uint4 counts = gMisc[wr];
        uint nl = counts.w & 0xFFFFu;
        uint nt = counts.w >> 16;
        float muL = max(smul(drvCurve(wr + 7u, nl, max(kappa, w5.z)), w5.x), 0.0);
        muL = ssqrt(smul(muL, c.c5.w));
        float muT = max(smul(drvCurve(wr + 11u, nt, max(alpha, w5.w)), w5.y), 0.0);
        muT = ssqrt(smul(muT, c.c5.w));
        float al = smul(muL, fn_);
        float at = smul(muT, fn_);
        // (drvTyre's friction ellipse's shape: there only when the ellipse binds, here
        // always; nothing else reads it.)
        float nx = sdiv(kappa, max(w5.z, 0.03));
        float tanA = asfloat(0x4103cf3au);
        if (vlen < 1e-3) tanA = 0.0;
        else if (abs(vt) > 0.0) tanA = min(sdiv(abs(vs), abs(vt)), asfloat(0x4103cf3au));
        float ny = sdiv(tanA, w6.x);
        float nn = ssqrt(nf(smul(nx, nx) + smul(ny, ny)));
        float ws = nn > 1e-6 ? drvClamp(nf(max(nx, ny) - 1.0), 0.0, 1.0) : 0.0;
        DSet(p, float4(t, vt));
        DSet(p + 1u, float4(sd, vs));
        DSet(p + 2u, float4(n, fn_));
        DSet(p + 3u, float4(m, j, r, wd));
        DSet(p + 4u, float4(b, al, at, nx));
        DSet(p + 5u, float4(ny, nn, ws, 0.0));
        DSet(p + 6u, float4(st.y, angle, lenK, kappa));
        DSet(p + 7u, float4(alpha, 1.0, 0.0, apart ? 1.0 : 0.0));
    }
    else
    {
        float wNew = drvSpin(m, 0.0, 0.0, j, r, wd, b, 0.0).x;
        DSet(p + 6u, float4(st.y, angle, lenK, 0.0));
        DSet(p + 7u, float4(0.0, 0.0, wNew, apart ? 1.0 : 0.0));
    }
    DSet(wr + 17u, float4(sadd3(c.ow, sadd3(xh, sscale(c.up, nf(w2.z - lenK)))), 0.0));
    DSet(wr + 18u, float4(scross(uw, fwd), 0.0));
    DSet(wr + 19u, float4(uw, 0.0));
}
void drvTyrePost(uint v, uint vr, uint k, uint step)
{
    float h = STEP;
    uint wr = vr + DRV_VEH + k * DRV_WHEEL;
    uint p = drvPreAt(vr, k);
    float4 p6 = DF(p + 6u);
    float4 p7 = DF(p + 7u);
    bool contact = p7.y != 0.0;
    float wNew;
    float fnOut = 0.0;
    float m = 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    uint4 w0 = gMisc[wr];
    uint nodeOff = gStatic[v * 4u].x;
    if (contact)
    {
        float4 p0 = DF(p);
        float4 p1 = DF(p + 1u);
        float4 p2 = DF(p + 2u);
        float4 p3 = DF(p + 3u);
        float4 p4 = DF(p + 4u);
        float4 p5 = DF(p + 5u);
        float3 t = p0.xyz;
        float vt = p0.w;
        float3 sd = p1.xyz;
        float vs = p1.w;
        float3 n = p2.xyz;
        float fn_ = p2.w;
        m = p3.x;
        float j = p3.y;
        float r = p3.z;
        float wd = p3.w;
        float b = p4.x;
        float al = p4.y;
        float at = p4.z;
        float nx = p4.w;
        float ny = p5.x;
        float nn = p5.y;
        float ws = p5.z;
        uint beamOff = gStatic[v * 4u].z;
        uint4 r2 = gStatic[v * 4u + 2u];
        uint g1 = nodeOff + w0.x;
        float3 fpre = drvForceNow(g1, beamOff, r2.z, r2.w, step);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceNow(nodeOff + w0.y, beamOff, r2.z, r2.w, step));
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
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
        wNew = p7.z;
    }
    float spin = nf(p6.x + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, p6.y, p6.z));
    DSet(wr + 16u, float4(fnOut, p6.w, p7.x, contact ? 1.0 : 0.0));
    for (uint i = 0u; i < DRV_PRE; i++) gMisc[p + i] = uint4(0u, 0u, 0u, 0u);
    if (contact)
    {
        uint g1 = nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(gNodes[g1 * 3u].w, m)));
            accAdd(g2, sscale(force, sdiv(gNodes[g2 * 3u].w, m)));
        }
    }
}
// The tyres' first part (wheel lane w; drvTyrePre), then the end (lane 0), as
// driveAheadPart's part 2 takes the wheels.
void driveTyresPre(DrvCtx c, bool has, bool apart, uint w)
{
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    for (uint k = k0, kStop = EB_END(304, k, k1, 1u); k < kStop; k++) drvTyrePre(c, k, apart);
    if (has && w == 0u) drvEnd(c);
}
// The tyres' second part (wheel lane w; drvTyrePost) for vehicle v's wheels as
// driveTyresPre took them: the wheels-apart decision from wheel w's record (wheel w is
// this lane's when apart; otherwise lane 0 takes every wheel, and any other lane reads
// 0, whether or not lane 0 has cleared the record).
void driveTyresPost(uint v, uint w, uint step)
{
    if (w >= DRV_MAX_WHEELS) return;
    uint vr = driveBase(v);
    uint nw = min(gMisc[vr + 2u].x, DRV_MAX_WHEELS);
    if (w >= nw) return;
    bool apart = DF(drvPreAt(vr, w) + 7u).w != 0.0;
    uint k0 = apart ? w : 0u;
    uint k1 = apart ? w + 1u : (w == 0u ? nw : 0u);
    for (uint k = k0, kStop = EB_END(305, k, k1, 1u); k < kStop; k++) drvTyrePost(v, vr, k, step);
}

// --- perfFlags bit 2097152 (explicit_plan::PERF_TYRE_EARLY, with PERF_TYRE_AHEAD): a
// wheel's second part loaded before the step's hits ---
// drvTyrePost reads, besides its hubs' accumulators (accForce), only what nothing in the
// hits kernel after its first barrier writes: its first part's records (the list
// dispatch's), the wheel's record, the hub nodes' masses and beam sums (the beam phase's
// incidence entries and group state; resolvePending ran before that barrier, and selfApply
// writes neither). With the wheels apart (wheel w on lane 255 - w, its own records), the
// lane loads all of that while the other lanes apply the hits (drvTyreLoad), and after
// them reads the accumulators and runs the rest (drvTyreFinish): drvTyrePost's arithmetic
// on the same bits in the same order (drvForceNow's `beam sum + accForce`, as written
// there). Not apart: drvTyrePost as before.
struct DrvPostIn
{
    bool ok;
    uint p;
    uint wr;
    uint4 w0;
    uint nodeOff;
    float4 p0;
    float4 p1;
    float4 p2;
    float4 p3;
    float4 p4;
    float4 p5;
    float4 p6;
    float4 p7;
    float3 b1;
    float3 b2;
    float m1;
    float m2;
};
DrvPostIn drvTyreLoad(uint v, uint w, uint step)
{
    DrvPostIn d = (DrvPostIn)0;
    d.ok = false;
    if (w >= DRV_MAX_WHEELS) return d;
    uint vr = driveBase(v);
    uint nw = min(gMisc[vr + 2u].x, DRV_MAX_WHEELS);
    if (w >= nw) return d;
    d.p = drvPreAt(vr, w);
    d.p7 = DF(d.p + 7u);
    if (d.p7.w == 0.0) return d;
    d.ok = true;
    d.wr = vr + DRV_VEH + w * DRV_WHEEL;
    d.p6 = DF(d.p + 6u);
    d.w0 = gMisc[d.wr];
    d.nodeOff = gStatic[v * 4u].x;
    if (d.p7.y == 0.0) return d;
    d.p0 = DF(d.p);
    d.p1 = DF(d.p + 1u);
    d.p2 = DF(d.p + 2u);
    d.p3 = DF(d.p + 3u);
    d.p4 = DF(d.p + 4u);
    d.p5 = DF(d.p + 5u);
    uint beamOff = gStatic[v * 4u].z;
    uint4 r2 = gStatic[v * 4u + 2u];
    uint g1 = d.nodeOff + d.w0.x;
    d.b1 = beamSum(gNodes[g1 * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + g1], beamOff, r2.z, r2.w, (step + 1u) & 1u, false);
    d.m1 = gNodes[g1 * 3u].w;
    if (d.w0.x != d.w0.y)
    {
        uint g2 = d.nodeOff + d.w0.y;
        d.b2 = beamSum(gNodes[g2 * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + g2], beamOff, r2.z, r2.w, (step + 1u) & 1u,
                       false);
        d.m2 = gNodes[g2 * 3u].w;
    }
    return d;
}
float3 drvForceFrom(float3 f, uint gi)
{
    return f + accForce(gi);
}
void drvTyreFinish(DrvPostIn d)
{
    float h = STEP;
    uint p = d.p;
    uint wr = d.wr;
    float4 p6 = d.p6;
    float4 p7 = d.p7;
    bool contact = p7.y != 0.0;
    float wNew;
    float fnOut = 0.0;
    float m = 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    uint4 w0 = d.w0;
    uint nodeOff = d.nodeOff;
    if (contact)
    {
        float3 t = d.p0.xyz;
        float vt = d.p0.w;
        float3 sd = d.p1.xyz;
        float vs = d.p1.w;
        float3 n = d.p2.xyz;
        float fn_ = d.p2.w;
        m = d.p3.x;
        float j = d.p3.y;
        float r = d.p3.z;
        float wd = d.p3.w;
        float b = d.p4.x;
        float al = d.p4.y;
        float at = d.p4.z;
        float nx = d.p4.w;
        float ny = d.p5.x;
        float nn = d.p5.y;
        float ws = d.p5.z;
        uint g1 = nodeOff + w0.x;
        float3 fpre = drvForceFrom(d.b1, g1);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceFrom(d.b2, nodeOff + w0.y));
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
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
        wNew = p7.z;
    }
    float spin = nf(p6.x + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, p6.y, p6.z));
    DSet(wr + 16u, float4(fnOut, p6.w, p7.x, contact ? 1.0 : 0.0));
    for (uint i = 0u; i < DRV_PRE; i++) gMisc[p + i] = uint4(0u, 0u, 0u, 0u);
    if (contact)
    {
        uint g1 = nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(d.m1, m)));
            accAdd(g2, sscale(force, sdiv(d.m2, m)));
        }
    }
}

// A contact group's wheels on each other (members `off`, k of them in vehicle order) at
// substep s's start, where every member's state is final (explicit_contact.rs
// ExplicitBeamsWorld::wheels_begin and wheels_end): the last step's staged reactions onto
// the partners' nodes' accumulators (read by the coming node phases), then (`rays`) every
// member's wheels' rays against the other members' live collision triangles, the nearest
// hit per wheel kept in its partner records for step gMisc[v].w + s. Every lane; barriers
// in uniform control flow (k uniform). `rays` false: the reactions only (after a frame's
// last step).
// The rays in rounds of WP_ROUND members (their wheels the round's slots): a slot's ray
// against the members whose allowed box (the frame's bound on its nodes, vvRec
// VV_ALLOWED_*: a node outside it is reported, vvReportOutside) its segment's box meets;
// the rest cannot hold a hit (lossless). Those (slot, member) pairs are listed with the
// prefix of their triangles, then every (pair, triangle) runs across the lanes: the least
// t a slot, then the first triangle at it (its global index: the CPU's order). Workgroup
// words in gsCount's first ones (free between a step's phases): the minima, the pairs'
// count and list.
static const uint WP_ROUND = 2u;
static const uint WP_SLOTS = WP_ROUND * DRV_MAX_WHEELS;
static const uint WP_COUNT = 2u * WP_SLOTS;
static const uint WP_LIST = WP_COUNT + 1u;
// Pairs a round may list (gsCount's words past the minima and the count); more fails
// the frame loudly (never a pair dropped). A round lists at most 32 slots × the members
// whose box a ray's meets.
static const uint WP_PAIRS = GRID_BUCKETS + REUSE_WORDS - WP_LIST - 1u;
// A listed pair: its triangles' prefix << 13 | slot << 7 | member.

// The ray of round slot q (member r0 + q / 16's wheel q % 16): its partner records `pa`,
// start and reach, direction. False: no such wheel, or it does not see bodies.
bool wpRay(uint off, uint k, uint r0, uint q, out uint pa, out float4 ray, out float3 d)
{
    pa = 0u;
    ray = float4(0.0, 0.0, 0.0, -1.0);
    d = float3(0.0, 0.0, 0.0);
    uint m = r0 + q / DRV_MAX_WHEELS;
    uint w = q % DRV_MAX_WHEELS;
    if (m >= k) return false;
    uint va = vvMember(off, m);
    if ((gMisc[driveBase(va) + 5u].y & 8u) == 0u || w >= min(gMisc[driveBase(va) + 2u].x, DRV_MAX_WHEELS)) return false;
    pa = drvPartnerAt(va, w);
    ray = DF(pa);
    d = DF(pa + 4u).xyz;
    return true;
}

void wheelsPartners(uint off, uint k, uint lane, uint s, bool rays)
{
    for (uint i = lane, iStop = EB_END(306, i, k * DRV_MAX_WHEELS, LANES); i < iStop; i += LANES)
    {
        uint v = vvMember(off, i / DRV_MAX_WHEELS);
        uint w = i % DRV_MAX_WHEELS;
        if (w >= min(gMisc[driveBase(v) + 2u].x, DRV_MAX_WHEELS)) continue;
        uint pa = drvPartnerAt(v, w);
        uint4 r = gMisc[pa + 3u];
        if (r.w == 0u) continue;
        float3 f = asfloat(r.xyz);
        float3 wt = DF(pa + 1u).xyz;
        uint4 tn = gStatic[gP.triBase + gMisc[pa + 1u].w];
        if (wt.x > 0.0) accAdd(tn.x, sscale(f, wt.x));
        if (wt.y > 0.0) accAdd(tn.y, sscale(f, wt.y));
        if (wt.z > 0.0) accAdd(tn.z, sscale(f, wt.z));
        gMisc[pa + 3u] = uint4(0u, 0u, 0u, 0u);
    }
    if (!rays)
    {
        sync();
        return;
    }
    // Every member's rays (World): start and reach in record 0, direction in record 4.
    for (uint i = lane, iStop = EB_END(307, i, k * DRV_MAX_WHEELS, LANES); i < iStop; i += LANES)
    {
        uint va = vvMember(off, i / DRV_MAX_WHEELS);
        uint w = i % DRV_MAX_WHEELS;
        if ((gMisc[driveBase(va) + 5u].y & 8u) == 0u) continue;
        DrvCtx c;
        if (!drvCtx(va, c) || w >= c.nw) continue;
        uint wr = drvWheel(c, w);
        float4 w1 = DF(wr + 1u);
        float4 w2 = DF(wr + 2u);
        float mh;
        float3 xh, vh;
        drvHub(c, w, mh, xh, vh);
        float3 hp = sadd3(xh, sscale(c.up, w2.z));
        uint pa = drvPartnerAt(va, w);
        DSet(pa, float4(sadd3(c.ow, hp), nf(w2.y + w1.x)));
        DSet(pa + 4u, float4(-c.up, 0.0));
    }
    sync();
    for (uint r0 = 0u, r0Stop = EB_END_UNIFORM(308, r0, k, WP_ROUND, EB_MAX_MEMBERS); r0 < r0Stop; r0 += WP_ROUND)
    {
        if (lane < WP_COUNT) gsCount[lane].store(0xFFFFFFFFu);
        if (lane == 0u) gsCount[WP_COUNT].store(0u);
        sync();
        // The pairs: (slot, member) whose boxes meet.
        for (uint i = lane, iStop = EB_END(309, i, WP_SLOTS * k, LANES); i < iStop; i += LANES)
        {
            uint q = i / k;
            uint j = i % k;
            uint pa;
            float4 ray;
            float3 d;
            if (!wpRay(off, k, r0, q, pa, ray, d) || r0 + q / DRV_MAX_WHEELS == j) continue;
            uint vb = vvMember(off, j);
            float3 e = sadd3(ray.xyz, sscale(d, ray.w));
            float3 lo = min(ray.xyz, e);
            float3 hi = max(ray.xyz, e);
            if (any(hi < asfloat(gMisc[vvRec(vb, VV_ALLOWED_LO)].xyz)) || any(lo > asfloat(gMisc[vvRec(vb, VV_ALLOWED_HI)].xyz))) continue;
            uint n = gsCount[WP_COUNT].add(1u);
            if (n < WP_PAIRS) gsCount[WP_LIST + n].store(q << 7 | j);
            else ebLoopCapHit(310);
        }
        sync();
        // Their triangles' prefix (lane 0, in list order: the sums matter, not the order).
        uint pairs = min(gsCount[WP_COUNT].load(), WP_PAIRS);
        if (lane == 0u)
        {
            uint run = 0u;
            for (uint p = 0u, pStop = EB_END(311, p, pairs, 1u); p < pStop; p++)
            {
                uint e = gsCount[WP_LIST + p].load();
                gsCount[WP_LIST + p].store(run << 13 | (e & 0x1FFFu));
                run += gStatic[vvMember(off, e & 0x7Fu) * 4u + 1u].y;
            }
            gsCount[WP_LIST + pairs].store(run << 13);
        }
        sync();
        uint total = gsCount[WP_LIST + pairs].load() >> 13;
        // The least t of each slot, then the first triangle at it.
        for (uint pass = 0u; pass < 2u; pass++)
        {
            for (uint i = lane, iStop = EB_END(312, i, total, LANES); i < iStop; i += LANES)
            {
                // The pair holding item i: the last whose prefix is at most i.
                uint lo = 0u;
                uint hi = pairs;
                for (uint g = 0u; g < 16u; g++)
                {
                    if (hi - lo <= 1u) break;
                    uint mid = (lo + hi) / 2u;
                    if ((gsCount[WP_LIST + mid].load() >> 13) <= i) lo = mid;
                    else hi = mid;
                }
                uint e = gsCount[WP_LIST + lo].load();
                uint q = (e >> 7) & 0x3Fu;
                uint vb = vvMember(off, e & 0x7Fu);
                uint pa;
                float4 ray;
                float3 d;
                wpRay(off, k, r0, q, pa, ray, d);
                uint gt = gStatic[vb * 4u + 1u].x + (i - (e >> 13));
                uint z = gMisc[gP.vehicles + gt].z;
                if ((z & TRI_ANTI_ONLY) != 0u || ((z & TRI_BREAKABLE) != 0u && triangleDead(gt))) continue;
                float3 ob = asfloat(gMisc[vb].xyz);
                uint4 tn = gStatic[gP.triBase + gt];
                float3 wt;
                float4 h = drvTriangle(ray.xyz, d, sadd3(ob, gNodes[tn.x * 3u].xyz), sadd3(ob, gNodes[tn.y * 3u].xyz),
                                       sadd3(ob, gNodes[tn.z * 3u].xyz), wt);
                if (h.w < 0.0 || h.w > ray.w) continue;
                uint key = asuint(h.w == 0.0 ? 0.0 : h.w);
                if (pass == 0u) gsCount[q].min(key);
                else if (key == gsCount[q].load()) gsCount[WP_SLOTS + q].min(gt);
            }
            sync();
        }
        // The winners' records.
        if (lane < WP_SLOTS)
        {
            uint pa;
            float4 ray;
            float3 d;
            if (wpRay(off, k, r0, lane, pa, ray, d))
            {
                uint va = vvMember(off, r0 + lane / DRV_MAX_WHEELS);
                uint step1 = gMisc[va].w + s + 1u;
                uint gt = gsCount[WP_SLOTS + lane].load();
                if (gt == 0xFFFFFFFFu)
                {
                    DSet(pa, drvNone());
                    gMisc[pa + 2u] = uint4(step1, 0xFFFFFFFFu, 0u, 0u);
                }
                else
                {
                    uint4 tn = gStatic[gP.triBase + gt];
                    float3 ob = asfloat(gMisc[tn.w].xyz);
                    float3 wt;
                    float4 h = drvTriangle(ray.xyz, d, sadd3(ob, gNodes[tn.x * 3u].xyz), sadd3(ob, gNodes[tn.y * 3u].xyz),
                                           sadd3(ob, gNodes[tn.z * 3u].xyz), wt);
                    // The point's velocity and inverse mass (explicit_contact::two_body_inverse_mass).
                    float3 v0 = gNodes[tn.x * 3u + 1u].xyz;
                    float3 v1 = gNodes[tn.y * 3u + 1u].xyz;
                    float3 v2 = gNodes[tn.z * 3u + 1u].xyz;
                    float3 vel = sadd3(sadd3(sscale(v0, wt.x), sscale(v1, wt.y)), sscale(v2, wt.z));
                    float3 im;
                    im.x = (gStatic[gP.nodeInfoBase + tn.x].x & NODE_IMMOVABLE) != 0u ? 0.0 : sdiv(1.0, gNodes[tn.x * 3u].w);
                    im.y = (gStatic[gP.nodeInfoBase + tn.y].x & NODE_IMMOVABLE) != 0u ? 0.0 : sdiv(1.0, gNodes[tn.y * 3u].w);
                    im.z = (gStatic[gP.nodeInfoBase + tn.z].x & NODE_IMMOVABLE) != 0u ? 0.0 : sdiv(1.0, gNodes[tn.z * 3u].w);
                    float t0 = wt.x > 0.0 ? smul(smul(wt.x, wt.x), im.x) : 0.0;
                    float t1 = wt.y > 0.0 ? smul(smul(wt.y, wt.y), im.y) : 0.0;
                    float t2 = wt.z > 0.0 ? smul(smul(wt.z, wt.z), im.z) : 0.0;
                    DSet(pa, h);
                    gMisc[pa + 1u] = uint4(asuint(wt), gt);
                    gMisc[pa + 2u] = uint4(step1, tn.w, 0u, 0u);
                    DSet(pa + 4u, float4(vel, nf(nf(t0 + t1) + t2)));
                }
            }
        }
        sync();
    }
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
    for (uint p = 0u, pStop = EB_END_UNIFORM(313, p, k, per, EB_MAX_MEMBERS); p < pStop; p += per)
    {
        uint m = p + lane / DRV_MAX_WHEELS;
        bool live = m < k;
        uint v = live ? vvMember(off, m) : 0u;
        driveWheelsSplit(v, lane % DRV_MAX_WHEELS, live, live ? gMisc[v].w + s : 0u);
    }
    sync();
}
