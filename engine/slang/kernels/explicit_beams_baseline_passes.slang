// --- passes ---

#ifdef EB_VV_SOLO
// Vehicle contact (the solo frame): this lane's nodes left the frame bounds
// (relative, in workgroup memory) at some step.
// A node of vehicle v outside its frame bounds (world positions): reported at
// once (rare), so nothing stays live across the step loop.
groupshared float3 gsVvBounds[2];
#endif

#ifdef EB_AERO
void nodeUpdateFuse(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
static const uint FLAG_FUSEDRAG = 8u;
// Vehicle flags: it has triggers (key-driven commands), slide nodes.
static const uint FLAG_KEYS = 16u;
static const uint FLAG_SLIDES = 32u;
// Node info flags above this shift: the node's slide-force slot + 1 (0: none).
static const uint NODE_SLOT_SHIFT = 8u;
float3 slideTake(float3 f, uint slot1);
#endif

void nodeUpdate(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        nodeUpdateFuse(gi, info, vehicle, origin, ground, boxOff, boxCount, beamOff, groupOff, groupCount,
                       (step + 1u) & 1u, hits);
        return;
    }
#endif
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
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
        if (ground > x.y)
            f += contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
        for (uint b = 0u; b < boxCount; b++)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u + 1u].xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float3 n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
                f += contactForce(f, vel, mass, n, boxSurface(), 0.0, V.w);
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
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
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
}
#ifdef EB_VV_SOLO
#ifdef EB_AERO
uint nodeUpdateFuseSolo(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
#endif
// nodeUpdate for the solo frame: also whether the node left the frame bounds.
uint nodeUpdateSolo(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        return nodeUpdateFuseSolo(gi, info, vehicle, origin, ground, boxOff, boxCount, beamOff, groupOff, groupCount,
                       (step + 1u) & 1u, hits);
    }
#endif
    float4 P = gNodes[gi * 3u];
    // The position the last step left (each step's checked once: the frame's
    // last at its end).
    if (any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
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
        if (ground > x.y)
            f += contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
        for (uint b = 0u; b < boxCount; b++)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u + 1u].xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float3 n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
                f += contactForce(f, vel, mass, n, boxSurface(), 0.0, V.w);
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
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
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
    return 0u;
}
#endif

// Live plain beams at a node, from the broken flags as they are now.
uint livePlain(uint node)
{
    uint4 info = gStatic[gP.nodeInfoBase + node];
    uint live = 0u;
    for (uint k = 0u; k < info.z; k++)
    {
        uint j = gLists[info.y + k] >> 1;
        uint kind = gStatic[gP.beamBase + j * 3u].z & 0xFFu;
        if (kind == KIND_PLAIN && gBeams[j * 2u + 1u].z == 0.0) live++;
    }
    return live;
}

#ifdef EB_AERO
void shock2Update(uint gj, uint local, bool fast, uint groupOff, uint parity, uint4 bn, float4 s1);
void triggerUpdate(uint gj, bool fast, uint parity, uint4 bn);
#endif

void beamUpdate(uint gj, uint local, bool fast, uint groupOff, uint parity)
{
    float4 s1 = gBeams[gj * 2u + 1u];
    if (s1.z != 0.0)
    {
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
#ifdef EB_AERO
    if ((bn.z & 0xFFu) == KIND_SHOCK2)
    {
        shock2Update(gj, local, fast, groupOff, parity, bn, s1);
        return;
    }
    if ((bn.z & 0xFFu) == KIND_TRIGGER)
    {
        triggerUpdate(gj, fast, parity, bn);
        return;
    }
#endif
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    float4 c2 = asfloat(gStatic[gP.beamBase + gj * 3u + 2u]);
    float4 s0 = gBeams[gj * 2u];
    uint kind = bn.z & 0xFFu;
    uint type = (bn.z >> 8) & 0xFFu;
    int group = asint(bn.w);
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    float rest = s0.x;
    float tc = s0.y;
    float tt = s0.z;
    float gate = s0.w;
    float strength = s1.x;
    // Strict, in the CPU reference's order (yield and break decide on these).
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float inv = fast ? sFastInvSqrt(q) : sdiv(1.0, ssqrt(q));
    float len = smul(q, inv);
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    float k = c1.x;
    float c = c1.y;
    bool broken = false;
    bool supportBroke = false;
    if (kind == KIND_LIMITED)
    {
        float lb = smul(c2.y, rest);
        float sb = smul(c2.x, rest);
        float o = 0.0;
        if (e > lb) o = nf(e - lb);
        else if (e < -sb) o = nf(-e - sb);
        if (o != 0.0)
        {
            k = nf(k + smul(nf(c1.z - k), o));
            c = nf(c + smul(nf(c1.w - c), o));
        }
    }
    else if (kind == KIND_SUPPORT)
    {
        if (e > 0.0)
        {
            k = 0.0;
            c = smul(c, 0.1);
            float factor = c2.y > 0.0 ? c2.y : SUPPORT_BREAK_FACTOR;
            if (e > smul(rest, factor))
            {
                broken = true;
                supportBroke = true;
            }
        }
    }
    else if (kind == KIND_ROPE)
    {
        if (e < 0.0)
        {
            k = 0.0;
            c = smul(c, 0.1);
        }
    }
    float f = nf(smul(-k, e) - smul(c, u));
    float last = f;
    float tau = abs(f);
    if (!supportBroke && tau > gate)
    {
        if (type == TYPE_DEFORMABLE && kind != KIND_LIMITED && k != 0.0)
        {
            if (f > tc && e < 0.0)
            {
                float delta = nf(e + smul(sdiv(tc, k), nf(1.0 - c2.z)));
                float old = rest;
                rest = max(nf(rest + delta), MIN_REST);
                f = nf(f - sdiv(nf(f - tc), 2.0));
                tau = f;
                if (rest < old)
                {
                    tc = sdiv(smul(tc, old), rest);
                    gate = min(min(tc, tt), strength);
                }
            }
            else if (f < -tt && e > 0.0)
            {
                float delta = nf(e - smul(sdiv(tt, k), nf(1.0 - c2.z)));
                float old = rest;
                rest = nf(rest + delta);
                f = nf(f - sdiv(nf(f + tt), 2.0));
                tau = -f;
                if (rest > old)
                {
                    tt = sdiv(smul(tt, rest), old);
                    gate = min(min(tc, tt), strength);
                }
                strength = nf(strength - smul(delta, k));
            }
        }
        if (tau > strength)
        {
            bool saved = false;
#ifdef EB_RACE_PROBE
            if ((gStatic[gP.nodeInfoBase + bn.x].x & NODE_ON_TRIANGLE) != 0u) probeLive(bn.x, 1u);
            if ((gStatic[gP.nodeInfoBase + bn.y].x & NODE_ON_TRIANGLE) != 0u) probeLive(bn.y, 1u);
#endif
            if ((gStatic[gP.nodeInfoBase + bn.x].x & NODE_ON_TRIANGLE) != 0u && livePlain(bn.x) < 3u) saved = true;
            if ((gStatic[gP.nodeInfoBase + bn.y].x & NODE_ON_TRIANGLE) != 0u && livePlain(bn.y) < 3u) saved = true;
            if (saved)
            {
                strength = 2.0 * gate;
            }
            else
            {
                broken = true;
                f = 0.0;
                if (group > 0)
                {
                    gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
#ifdef EB_RACE_PROBE
                    probeGroup(bn.z >> 16, uint(group));
#endif
                    gGroups[groupSlot(parity, groupOff, 0u)].min(local);
                }
            }
        }
    }
    // State records only when they change (yield, break): most steps change
    // neither. The last force (.y) is kept for a read-back only, so it is stored
    // with the record, not every step.
#ifdef EB_RACE_PROBE
    if (broken)
    {
        probeLive(bn.x, 2u);
        probeLive(bn.y, 2u);
    }
#endif
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken) gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// A triangle's throttle after a step: a waiting triangle counts down (and its idle
// steps up); one that was tested restarts its wait from its hit (gMisc .w, set by
// any of its lanes) or counts it up. A triangle that comes off idle waits at most
// 12 steps and is tested that same step (triangleTest reads it as not waiting).
int triangleFinish(uint gt)
{
    uint mi = gP.vehicles + gt;
    uint4 th = gMisc[mi];
    int wait = asint(th.x);
    int idle = asint(th.y);
    if (wait > 0)
    {
        wait = wait - 1;
        idle = idle + 1;
    }
    else
    {
        if (idle > 0)
        {
            wait = min(idle, 12);
            idle = 0;
        }
        wait = th.w != 0u ? -20000 : wait + 1;
    }
    gMisc[mi] = uint4(asuint(wait), asuint(idle), 0u, 0u);
    return wait;
}

// A triangle as its contact tests see it.
struct TriGeo
{
    uint mi;
    uint4 tn;
    float3 p0;
    float3 n;
    float3 ra;
    float3 rb;
    float3 rn;
    bool solvable;
    float3 lo;
    float3 hi;
};

// Contacter node h against triangle t: the contact force into the accumulators
// (h, and the triangle's corners by their weights) when h is within r of the face.
void tryContact(TriGeo t, uint h, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount,
                uint parity)
{
    uint4 tn = t.tn;
    if (h == tn.x || h == tn.y || h == tn.z) return;
    float4 P = gNodes[h * 3u];
    float3 p = nf3(origin + P.xyz);
    if (any(p < t.lo) || any(p > t.hi)) return;
    float3 q = nf3(p - t.p0);
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return;
    gMisc[t.mi].w = 1u;
    gsHit = 1u;
#ifdef EB_DET_HITS
    {
        uint base = gStatic[gP.nodeInfoBase + h].w * (1u + 2u * DET_CAP);
        uint i = gDet[base].add(1u);
        if (i < DET_CAP)
        {
            gDet[base + 1u + 2u * i].store(t.mi - gP.vehicles);
            gDet[base + 2u + 2u * i].store(h);
        }
        return;
    }
#endif
    float3 nn = t.n;
    if (delta < 0.0)
    {
        nn = -t.n;
        delta = -delta;
    }
    float4 V = gNodes[h * 3u + 1u];
    float4 P0 = gNodes[tn.x * 3u];
    float4 P1 = gNodes[tn.y * 3u];
    float4 P2 = gNodes[tn.z * 3u];
    float3 v = V.xyz - (alpha * gNodes[tn.y * 3u + 1u].xyz + beta * gNodes[tn.z * 3u + 1u].xyz
                        + gamma * gNodes[tn.x * 3u + 1u].xyz);
    float mt = alpha * P1.w + beta * P2.w + gamma * P0.w;
    float m = P.w * mt / (P.w + mt);
    float3 fh = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity, false)
        + accForce(h);
    float3 f = contactForce(fh, v, m, nn, boxSurface(), r - delta, V.w);
#ifdef EB_RACE_PROBE
    probeTouch(h, true);
    probeTouch(tn.x, false);
    probeTouch(tn.y, false);
    probeTouch(tn.z, false);
#endif
    accAdd(h, f);
    accAdd(tn.y, -alpha * f);
    accAdd(tn.z, -beta * f);
    accAdd(tn.x, -gamma * f);
}

// Triangle gt's geometry; mi < vehicles (none) when it waits this step.
TriGeo triangleGeo(uint gt, float3 origin, float r)
{
    TriGeo t;
    t.mi = gP.vehicles + gt;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    float3 u = nf3(p1 - t.p0);
    float3 w = nf3(p2 - t.p0);
    float3 cr = scross(u, w);
    float cl = ssqrt(sdot(cr, cr));
    t.n = cl > 0.0 ? float3(sdiv(cr.x, cl), sdiv(cr.y, cl), sdiv(cr.z, cl)) : cr;
    float det = sdot(u, scross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? sdiv(1.0, det) : 0.0;
    t.ra = sscale(scross(w, t.n), rd);
    t.rb = sscale(scross(t.n, u), rd);
    t.rn = sscale(scross(u, w), rd);
    t.lo = min(min(t.p0, p1), p2) - float3(r, r, r);
    t.hi = max(max(t.p0, p1), p2) + float3(r, r, r);
    return t;
}

// Lane `sub` of the `lanes` lanes of triangle gt tests candidates sub, sub + lanes, …
// Forces go to fixed-point atomics, so the lanes' order does not change the sums.
void triangleTest(uint gt, uint sub, uint lanes, float3 origin, float r, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r);
    for (uint k = sub; k < candCount; k += lanes)
        tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
}

// The grid's bucket of a cell.
uint gridBucket(int3 c)
{
    return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u) & (GRID_BUCKETS - 1u);
}

// Triangle gt against the contacters in the grid cells its box (padded by r)
// covers: each covered bucket's run once (two cells can share a bucket). A box
// over more than GRID_MAX_CELLS cells tests every candidate instead. The runs
// hold every contacter, so the contacts found are the all-pairs test's. GRID_SUBS
// lanes share a triangle: lane `sub` takes the cells sub, sub + GRID_SUBS, … (and
// the same share of the candidates when it falls back), so a triangle's walk is
// spread rather than serial on one lane.
static const uint GRID_SUBS = 8u;

// The triangles tested this step (wait <= 0 after the throttle update), listed by
// the node phase for a vehicle of up to TRI_LIST triangles; the triangle phase then
// walks only them. Counters by step parity: the beam phase clears the next step's.
static const uint TRI_LIST = 2048u;
groupshared uint gsTested[TRI_LIST];
groupshared Atomic<uint> gsTestedCount[2];

void triangleGrid(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub; i < cells; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        // Skip a bucket an earlier cell of this triangle already covers.
        bool dup = false;
        for (uint j = 0u; j < i; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        for (uint e = gsOff[b]; e < gsOff[b + 1u]; e++)
        {
            float4 E = gBeams[gP.gridBase + candOff + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
}

#ifdef EB_DET_HITS
// Tests only (EB_DET_HITS): the step's self-collision hits (found by the usual
// lanes) applied on one lane, sorted by (triangle, node), so each hit sees every
// earlier one in a fixed order: tryContact's force part.
void detApply(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint base = v * (1u + 2u * DET_CAP);
    uint n = gDet[base].load();
    if (n > DET_CAP)
    {
        gDet[gP.vehicles * (1u + 2u * DET_CAP)].store(1u);
        n = DET_CAP;
    }
    for (uint i = 1u; i < n; i++)
    {
        uint gt = gDet[base + 1u + 2u * i].load();
        uint h = gDet[base + 2u + 2u * i].load();
        uint j = i;
        while (j > 0u)
        {
            uint pg = gDet[base + 1u + 2u * (j - 1u)].load();
            uint ph = gDet[base + 2u + 2u * (j - 1u)].load();
            if (pg < gt || (pg == gt && ph <= h)) break;
            gDet[base + 1u + 2u * j].store(pg);
            gDet[base + 2u + 2u * j].store(ph);
            j--;
        }
        gDet[base + 1u + 2u * j].store(gt);
        gDet[base + 2u + 2u * j].store(h);
    }
    for (uint i = 0u; i < n; i++)
    {
        uint gt = gDet[base + 1u + 2u * i].load();
        uint h = gDet[base + 2u + 2u * i].load();
        TriGeo t = triangleGeo(gt, origin, r);
        uint4 tn = t.tn;
        float4 P = gNodes[h * 3u];
        float3 p = nf3(origin + P.xyz);
        float3 q = nf3(p - t.p0);
        float alpha = 0.0;
        float beta = 0.0;
        float delta = 0.0;
        if (t.solvable)
        {
            alpha = sdot(t.ra, q);
            beta = sdot(t.rb, q);
            delta = sdot(t.rn, q);
        }
        float gamma = nf(nf(1.0 - alpha) - beta);
        float3 nn = t.n;
        if (delta < 0.0)
        {
            nn = -t.n;
            delta = -delta;
        }
        float4 V = gNodes[h * 3u + 1u];
        float4 P0 = gNodes[tn.x * 3u];
        float4 P1 = gNodes[tn.y * 3u];
        float4 P2 = gNodes[tn.z * 3u];
        float3 vv = V.xyz - (alpha * gNodes[tn.y * 3u + 1u].xyz + beta * gNodes[tn.z * 3u + 1u].xyz
                             + gamma * gNodes[tn.x * 3u + 1u].xyz);
        float mt = alpha * P1.w + beta * P2.w + gamma * P0.w;
        float m = P.w * mt / (P.w + mt);
        float3 fh = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity, false)
            + accForce(h);
        float3 f = contactForce(fh, vv, m, nn, boxSurface(), r - delta, V.w);
        accAdd(h, f);
        accAdd(tn.y, -alpha * f);
        accAdd(tn.z, -beta * f);
        accAdd(tn.x, -gamma * f);
    }
    gDet[base].store(0u);
}
#endif

