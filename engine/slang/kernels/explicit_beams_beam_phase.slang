// --- The beam phase on its records sorted by kind (gP.beamOrder, explicit_plan
// W_BEAM_ORDER). The same updates as beamUpdate in file order; only the lane that runs
// a beam, and when, change: a beam writes only its own records and incidence entries,
// and group breaks are atomic minima. Per vehicle a header (entries, hot entries, its
// first incidence position), then its entries: the hot JBeam beams (jbeamHot), then
// every other beam by kind (beamUpdate), a two-record element as one entry whose
// second record is updated first, on the same lane, as the two lanes in step did.
static const uint BEAM_ORDER_SINGLE = 0u;
static const uint BEAM_ORDER_PAIR = 1u;
static const uint BEAM_ORDER_HOT = 2u;
static const uint BEAM_ORDER_ATRI = 3u;

// A hot beam's state and its ends (nothing in the beam phase writes them but the beam
// itself, after it has read them): loaded ahead of the beam's arithmetic.
struct HotIn
{
    float4 s0;
    float4 s1;
    float4 pa;
    float4 pb;
    float3 va;
    float3 vb;
};
HotIn hotLoad(uint gj, uint4 a, uint nodeOff)
{
    HotIn h;
    uint na = nodeOff + (a.z & 0xFFFFu);
    uint nb = nodeOff + (a.z >> 16);
    h.s1 = gBeams[gj * 2u + 1u];
    h.s0 = gBeams[gj * 2u];
    h.pa = gNodes[na * 3u];
    h.pb = gNodes[nb * 3u];
    h.va = gNodes[na * 3u + 1u].xyz;
    h.vb = gNodes[nb * 3u + 1u].xyz;
    return h;
}
// beamUpdate and jbeamUpdate for a JBeam plain, support, bounded or rope beam with no
// extension record, its static fields from its sorted entry (a, b): every load it needs
// is issued at once (the entry, then its state and its ends: hotLoad), no index waits
// on another.
// The rest of its records (limit k and c, short bound, plastic) only where the rule
// reads them. The arithmetic is jbeamUpdate's, operation for operation.
void jbeamHot(uint gj, uint4 a, uint4 b, HotIn h, uint incStart, uint groupOff, uint parity, bool centred, bool keep)
{
    uint local = a.x;
    cfFire(gj, groupOff, parity);
    uint slotA = gP.incBase + incStart + (a.w & 0xFFFFu);
    uint slotB = gP.incBase + incStart + (a.w >> 16);
    float4 s1 = h.s1;
    float4 s0 = h.s0;
    float4 pa = h.pa;
    float4 pb = h.pb;
    float3 va = h.va;
    float3 vb = h.vb;
    if (s1.z != 0.0)
    {
        // beamUpdate's broken beam (no extension record: extPending does nothing).
        if (s1.w != 0.0) gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
        gBeams[slotA] = float4(0.0, 0.0, 0.0, 0.0);
        gBeams[slotB] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    uint kind = (a.y >> 2) & 31u;
    uint type = (a.y >> 8) & 3u;
    int group = int(a.y) >> 16;
    bool broken = false;
    float strength = s1.x;
    float rest = s0.x;
    float tc = s0.y;
    float tt = s0.z;
    float gate = s0.w;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float2 li = sExactLength(q);
    float inv = li.y;
    float len = li.x;
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    float gain = 1.0;
    if (b.z != 0u)
    {
        uint fi = b.z - 1u;
        float4 flt = gBeams[fi];
        float y = nf(smul(flt.z, u) - smul(flt.w, flt.y));
        gBeams[fi] = float4(u, y, flt.z, flt.w);
        u = y;
        gain = flt.z;
    }
    float k = asfloat(b.x);
    float c = asfloat(b.y);
    float damp = smul(c, u);
    float slope = c;
    float last = 0.0;
    float f = 0.0;
    bool lengthBreak = false;
    if (kind == KIND_JBOUNDED)
    {
        float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
        float4 c2 = asfloat(gStatic[gP.beamBase + gj * 3u + 2u]);
        float au = abs(u);
        if (au > tc)
        {
            float m = nf(smul(c, tc) + smul(tt, nf(au - tc)));
            damp = u < 0.0 ? -m : m;
            slope = tt;
        }
        float lb = smul(c2.y, rest);
        float sb = smul(c2.x, rest);
        float o = 0.0;
        if (e > lb) o = nf(e - lb);
        else if (e < -sb) o = nf(-e - sb);
        if (o > 0.0)
        {
            float sz = c2.z > 0.0 ? min(sdiv(o, c2.z), 1.0) : 1.0;
            k = nf(k + smul(nf(c1.z - k), sz));
            damp = nf(damp + smul(nf(smul(c1.w, u) - damp), sz));
            slope = nf(slope + smul(nf(c1.w - slope), sz));
        }
    }
    else if (kind == KIND_JSUPPORT && e > 0.0)
    {
        if (len > asfloat(b.w)) lengthBreak = true;
        k = 0.0;
        damp = 0.0;
        slope = 0.0;
    }
    else if (kind == KIND_JROPE && e < 0.0)
    {
        k = 0.0;
        damp = smul(smul(c, 0.1), u);
        slope = smul(c, 0.1);
    }
    bool tear = false;
    if (lengthBreak)
    {
        broken = true;
        if (group > 0)
        {
            gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
            gGroups[groupSlot(parity, groupOff, 0u)].min(local);
        }
    }
    else
    {
        f = nf(smul(-k, e) - damp);
        last = f;
        float tau = abs(f);
        if (tau > gate && type == TYPE_DEFORMABLE && k != 0.0)
        {
            float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
            float4 c2 = asfloat(gStatic[gP.beamBase + gj * 3u + 2u]);
            if (f > tc && e < 0.0 && nf(e + smul(sdiv(tc, k), nf(1.0 - c2.z))) < 0.0)
            {
                float delta = nf(e + smul(sdiv(tc, k), nf(1.0 - c2.z)));
                float old = rest;
                float floorLen = max(c2.x, MIN_REST);
                float free = nf(rest + delta);
                if (free < floorLen)
                {
                    rest = min(floorLen, old);
                    f = nf(smul(-k, nf(len - rest)) - damp);
                }
                else
                {
                    rest = free;
                    f = nf(f - sdiv(nf(f - tc), 2.0));
                }
                tau = abs(f);
                if (rest < old)
                {
                    tc = min(sdiv(smul(tc, old), rest), max(c1.z, tc));
                    gate = min(min(tc, tt), strength);
                }
            }
            else if (f < -tt && e > 0.0 && nf(e - smul(sdiv(tt, k), nf(1.0 - c2.z))) > 0.0)
            {
                float delta = nf(e - smul(sdiv(tt, k), nf(1.0 - c2.z)));
                float old = rest;
                float free = nf(rest + delta);
                float ceilLen = abs(c2.y);
                if (free > ceilLen)
                {
                    rest = max(ceilLen, old);
                    f = nf(smul(-k, nf(len - rest)) - damp);
                    tear = c2.y < 0.0;
                }
                else
                {
                    rest = free;
                    f = nf(f - sdiv(nf(f + tt), 2.0));
                }
                tau = abs(f);
                if (rest > old)
                {
                    tt = min(sdiv(smul(tt, rest), old), max(c1.z, tt));
                    gate = min(min(tc, tt), strength);
                    if (!keep) strength = nf(strength - smul(nf(rest - old), k));
                }
            }
        }
        if (tau > strength || tear)
        {
            broken = true;
            f = 0.0;
            if (group > 0)
            {
                gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
                gGroups[groupSlot(parity, groupOff, 0u)].min(local);
            }
        }
    }
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[slotA] = float4(push, !broken && centred ? smul(slope, gain) : 0.0);
    gBeams[slotB] = float4(-push, 0.0);
}

// jbeamExtUpdate's aero triangle (KIND_JATRI) for an unbroken pair, its third node,
// factors and both records' slots from its sorted entry (a, b): the same arithmetic.
void atriHot(uint4 a, uint4 b, uint nodeOff, uint incStart)
{
    uint n0 = nodeOff + (a.z & 0xFFFFu);
    uint n1 = nodeOff + (a.z >> 16);
    uint n2 = nodeOff + (a.y >> 16);
    float3 x0 = gNodes[n0 * 3u].xyz;
    float3 x1 = gNodes[n1 * 3u].xyz;
    float3 x2 = gNodes[n2 * 3u].xyz;
    float3 v0 = gNodes[n0 * 3u + 1u].xyz;
    float3 v1 = gNodes[n1 * 3u + 1u].xyz;
    float3 v2 = gNodes[n2 * 3u + 1u].xyz;
    float4 e1 = float4(0.0, asfloat(b.x), asfloat(b.y), asfloat(b.z));
    float3 vm = sscale(sadd3(sadd3(v0, v1), v2), 1.0 / 3.0);
    float s2 = sdot(vm, vm);
    float3 cr = scross(ssub3(x1, x0), ssub3(x2, x0));
    float cl = ssqrt(sdot(cr, cr));
    float3 f = float3(0.0, 0.0, 0.0);
    if (s2 > 1e-12 && cl > 0.0)
    {
        float sp = ssqrt(s2);
        float3 u = sscale(vm, sdiv(-1.0, sp));
        float3 n = sscale(cr, sdiv(1.0, cl));
        float c = sdot(n, u);
        float sa = abs(c);
        float ca = ssqrt(max(nf(1.0 - smul(c, c)), 0.0));
        float q = smul(smul(0.5, cl), s2);
        f = sscale(u, smul(smul(e1.y, q), sa));
        float3 l = ssub3(n, sscale(u, c));
        float ll = ssqrt(sdot(l, l));
        if (ll > 0.0)
        {
            float cla = sa <= e1.w ? smul(sa, ca) : smul(e1.w, ca);
            float m = smul(smul(e1.z, q), cla);
            f = sadd3(f, sscale(l, sdiv(c < 0.0 ? -m : m, ll)));
        }
        f = sscale(f, 1.0 / 3.0);
    }
    uint base = gP.incBase + incStart;
    gBeams[base + (a.w & 0xFFFFu)] = float4(f, 0.0);
    gBeams[base + (a.w >> 16)] = float4(f, 0.0);
    gBeams[base + (b.w & 0xFFFFu)] = float4(f, 0.0);
    gBeams[base + (b.w >> 16)] = float4(0.0, 0.0, 0.0, 0.0);
}

// Vehicle v's beam phase on its sorted entries (`head`: its gStatic header at
// gP.beamOrder + v): lanes stride over the entries. Without the sorted records
// (gP.beamOrder 0), its `count` beams in file order, a beam a lane, as single entries:
// one call site of beamUpdate for both (kernel-compile-time.md).
void beamPhaseSorted(uint v, uint lane, uint lanes, uint4 head, bool on, uint nodeOff, uint beamOff, bool fast,
                     uint groupOff, uint parity, uint flagsV, uint count)
{
    bool sorted = gP.beamOrder != 0u;
    if (!sorted) head = uint4(count, 0u, 0u, 0u);
    uint entries = on ? head.x : 0u;
    bool centred = gP.centredDv != 0u && (flagsV & FLAG_CENTRED) != 0u;
    bool keep = (flagsV & FLAG_KEEP_STRENGTH) != 0u;
    uint at = gP.beamOrder + gP.vehicles + 2u * beamOff;
    uint hot = min(head.y, entries);
    uint k = lane;
    for (uint kStop = EB_END(10, k, hot, lanes); k < kStop; k += lanes)
    {
        uint4 a = gStatic[at + 2u * k];
        jbeamHot(beamOff + a.x, a, gStatic[at + 2u * k + 1u], hotLoad(beamOff + a.x, a, nodeOff), head.z, groupOff,
                 parity, centred, keep);
    }
    for (uint kStop = EB_END(11, k, entries, lanes); k < kStop; k += lanes)
    {
        uint4 a = sorted ? gStatic[at + 2u * k] : uint4(k, BEAM_ORDER_SINGLE, 0u, 0u);
        uint cls = a.y & 3u;
        uint gj = beamOff + a.x;
        if (cls == BEAM_ORDER_ATRI)
        {
            // Both records unbroken: their updates are the first's aero force alone
            // (the second's writes nothing), after their contact-force groups.
            if (gBeams[gj * 2u + 1u].z == 0.0 && gBeams[(gj + 1u) * 2u + 1u].z == 0.0)
            {
                cfFire(gj + 1u, groupOff, parity);
                cfFire(gj, groupOff, parity);
                atriHot(a, gStatic[at + 2u * k + 1u], nodeOff, head.z);
                continue;
            }
        }
        // A pair: the second record first (the first record writes its incidence
        // entries and, a torsion bar, its state), then the first. An unbroken second
        // record's update writes nothing (it returns on its kind) but its contact-force
        // group. One call site of beamUpdate (the compilers inline it).
        for (uint p = cls == BEAM_ORDER_PAIR || cls == BEAM_ORDER_ATRI ? 0u : 1u; p < 2u; p++)
        {
            uint up = 1u - p;
            if (p == 0u && gBeams[(gj + 1u) * 2u + 1u].z == 0.0)
            {
                cfFire(gj + 1u, groupOff, parity);
                continue;
            }
            beamUpdate(gj + up, a.x + up, fast, groupOff, parity);
        }
    }
}

// A triangle's throttle after a step: a waiting triangle counts down (and its idle
// steps up); one that was tested restarts its wait from its hit (gMisc .w, set by
// any of its lanes) or counts it up. A triangle that comes off idle waits at most
// 12 steps and is tested that same step (triangleTest reads it as not waiting).
// The largest wait triangle gt's throttle may hold: 0 when its vehicle runs without
// the throttle (FLAG_NO_THROTTLE), else unbounded.
int throttleCap(uint gt)
{
    uint v = gStatic[gP.triBase + gt].w;
    return (gStatic[v * 4u + 3u].x & FLAG_NO_THROTTLE) != 0u ? 0 : 0x7FFFFFFF;
}

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
        wait = th.w != 0u ? -20000 : min(wait + 1, throttleCap(gt));
    }
    gMisc[mi] = uint4(asuint(wait), asuint(idle), th.z, 0u);
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
    // Anti-clip (§36): 1 the vehicle has it, 2 the triangle is anti-clip only.
    uint clip;
    // Anti-clip's frame of the triangle a step back (selfClip's crossing test), when
    // qs is nonzero (triGeoLoad: computed once a step by triGeoFill); 0: selfClip
    // computes it.
    uint qs;
    float3 q0;
    float3 qra;
    float3 qrb;
    float3 qrn;
    bool qsolv;
};

// The hit test of contacter node h against triangle t: (alpha, beta, delta) when
// it hits, else none.
bool triangleHit(TriGeo t, float3 p, float r, out float alpha, out float beta, out float gamma, out float delta)
{
    float3 q = nf3(p - t.p0);
    alpha = 0.0;
    beta = 0.0;
    delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    gamma = nf(nf(1.0 - alpha) - beta);
    return alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r;
}

void hitForce(uint v, uint at, uint key, int3 sum, uint round, float3 origin, float r, uint beamOff, uint groupOff,
              uint groupCount, uint parity);

// Self-collision's two-body rule (jbeam-semantics-spec.md §36; the CPU's
// apply_two_body_self_hits): vehicle v has it (FLAG_TWO_BODY) and its counts room.
bool selfTwoBody(uint v)
{
    return gP.selfSplit != 0u && (gStatic[v * 4u + 3u].x & FLAG_TWO_BODY) != 0u;
}

// Node x as k = its hits this step sub-bodies: inverse mass k/m (0 immovable).
float selfInvMass(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return 0.0;
    return sdiv(float(uint(gAcc[gP.selfSplit + x].load())), gNodes[x * 3u].w);
}

// Node x's own inverse mass (0 immovable): §37's position step shares.
float ownInvMass(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return 0.0;
    return sdiv(1.0, gNodes[x * 3u].w);
}

// Node x's acceleration from its force at the pass's start (0 immovable).
// Node x's beam force for a hit's pass: its record (fb, hitFbFill's: the same sum's
// bits) or, fb 0, the sum from its incidence entries.
float3 hitBeamSum(uint x, uint fb, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (fb != 0u) return asfloat(uint3(gNext[fb], gNext[fb + 1u], gNext[fb + 2u]));
    return beamSum(gNodes[x * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + x], beamOff, groupOff, groupCount, parity,
                   false);
}

float3 selfAcc(uint x, uint fb, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return float3(0.0, 0.0, 0.0);
    float3 f = hitBeamSum(x, fb, beamOff, groupOff, groupCount, parity);
    return sdiv3(f, gNodes[x * 3u].w);
}

float selfTerm(float w, float im) { return w > 0.0 ? smul(smul(w, w), im) : 0.0; }

// The two-body force of the hit at record `at` (contacter h): reduced mass from the
// counts, the cancelled force (F_h/m_h − Σ w·F_k/m_k)·m, then the contact rule with
// the geometry the finding lane stored. Strict, in the CPU's order.
float3 selfTwoBodyCore(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 nn,
                       float pen, float3 vv, float friction, uint beamOff, uint groupOff, uint groupCount,
                       uint parity, bool shifts, uint fb)
{
    float imt = nf(nf(selfTerm(alpha, selfInvMass(t1)) + selfTerm(beta, selfInvMass(t2)))
                   + selfTerm(gamma, selfInvMass(t0)));
    float inv = nf(selfInvMass(h) + imt);
    float m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
    // a3 = α·a(t1) + β·a(t2) + γ·a(t0), then a(h): the four accelerations from one call
    // site (opaqueCount), the same strict sums in the same order.
    float3 a3 = float3(0.0, 0.0, 0.0);
    float3 ah = float3(0.0, 0.0, 0.0);
    for (uint k = 0u, kStop = EB_END(12, k, opaqueCount(4u), 1u); k < kStop; k++)
    {
        float3 acc = selfAcc(k == 0u ? t1 : (k == 1u ? t2 : (k == 2u ? t0 : h)),
                             fb == 0u ? 0u : fb + (k == 0u ? 6u : (k == 1u ? 9u : (k == 2u ? 3u : 0u))), beamOff, groupOff,
                             groupCount, parity);
        if (k == 0u) a3 = sscale(acc, alpha);
        else if (k == 1u) a3 = sadd3(a3, sscale(acc, beta));
        else if (k == 2u) a3 = sadd3(a3, sscale(acc, gamma));
        else ah = acc;
    }
    float3 fh = sscale(ssub3(ah, a3), m);
    uint hv = gStatic[gP.nodeInfoBase + h].w;
    Surface surf = contactSurfaceV(hv, hv);
    float budget;
    float3 f = contactResponse(fh, vv, m, nn, surf, pen, friction, budget);
    if (shifts && surf.dissipative != 0u && any(f != float3(0.0, 0.0, 0.0)))
    {
        // §37: the position step, a position impulse δ·m shared like the hit's
        // force, each node by its own inverse mass (explicit_beams.rs
        // apply_two_body_self_hits). Called once per hit (selfRound marks it
        // computed; selfSerial applies in order).
        float delta = shiftDepth(pen, budget, sdot(fh, nn));
        if (delta > 0.0)
        {
            float q = smul(delta, m);
            shiftAdd(h, sscale(nn, smul(q, ownInvMass(h))));
            if (alpha > 0.0) shiftAdd(t1, sscale(nn, -smul(alpha, smul(q, ownInvMass(t1)))));
            if (beta > 0.0) shiftAdd(t2, sscale(nn, -smul(beta, smul(q, ownInvMass(t2)))));
            if (gamma > 0.0) shiftAdd(t0, sscale(nn, -smul(gamma, smul(q, ownInvMass(t0)))));
        }
    }
    return f;
}

float3 selfTwoBodyForce(uint at, uint h, uint beamOff, uint groupOff, uint groupCount, uint parity, uint fb)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    return selfTwoBodyCore(h, gNext[at + 8u], gNext[at + 9u], gNext[at + 10u], alpha, beta,
                           nf(nf(1.0 - alpha) - beta), asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u])),
                           asfloat(gNext[at + 15u]), asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u])),
                           asfloat(gNext[at + 20u]), beamOff, groupOff, groupCount, parity, true, fb);
}

// Self-collision's anti-clip test (§36; explicit_beams.rs clip_hit, the order-free
// rule of explicit_contact.rs clip_test): contacter h at p against triangle t. The
// side h is kept on: held from last step, or the one it came from when it crossed
// the plane over the step inside the triangle (both a step back), or, new and in
// the band over the triangle, the one it is on. 0: no contact; 1: a hit (weights,
// depth into the contact, normal to h's side); 2: a contact without force (an
// anti-clip-only triangle, h in front of its plane). `word` the side to record.
TriGeo trianglePrev(TriGeo t, float3 origin);
uint selfClip(TriGeo t, uint h, float3 p, float3 origin, float r, out float alpha, out float beta,
              out float gamma, out float pen, out float3 nn, out uint word)
{
    float delta;
    triangleHit(t, p, r, alpha, beta, gamma, delta);
    pen = 0.0;
    nn = t.n;
    word = 0u;
    uint tl = t.mi - gP.vehicles - gStatic[t.tn.w * 4u + 1u].x;
    float side = clipHeld(h, CLIP_SELF, tl);
    bool over = alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0;
    bool have = over;
    if (side == 0.0)
    {
        bool crossed = false;
#ifdef EB_ABLATE
        // Measurement only: 33554432 no crossing test (the previous step's frame).
        if ((gP.ablate & 33554432u) != 0u)
        {
            if (!(over && abs(delta) <= r)) return 0u;
            side = delta < 0.0 ? -1.0 : 1.0;
        }
        else {
#endif
        TriGeo pf;
        if (t.qs != 0u)
        {
            pf.p0 = t.q0;
            pf.ra = t.qra;
            pf.rb = t.qrb;
            pf.rn = t.qrn;
            pf.solvable = t.qsolv;
        }
        else
            pf = trianglePrev(t, origin);
        float3 pp = ssub3(p, sscale(gNodes[h * 3u + 1u].xyz, STEP));
        float a0;
        float bb0;
        float g0;
        float d0;
        triangleHit(pf, pp, r, a0, bb0, g0, d0);
        if ((d0 > 0.0 && delta < 0.0) || (d0 < 0.0 && delta > 0.0))
        {
            float sc = sdiv(d0, nf(d0 - delta));
            float a = nf(a0 + smul(nf(alpha - a0), sc));
            float b = nf(bb0 + smul(nf(beta - bb0), sc));
            float g = nf(nf(1.0 - a) - b);
            if (a >= 0.0 && b >= 0.0 && g >= 0.0)
            {
                crossed = true;
                side = d0 > 0.0 ? 1.0 : -1.0;
                alpha = a;
                beta = b;
                gamma = g;
                have = true;
            }
        }
        if (!crossed)
        {
            if (!(over && abs(delta) <= r)) return 0u;
            side = delta < 0.0 ? -1.0 : 1.0;
        }
#ifdef EB_ABLATE
        }
#endif
    }
    float d = side * delta;
    if (!have || !(d <= r && d >= -CLIP_DEPTH)) return 0u;
    word = clipWord(CLIP_SELF, tl, side);
    nn = side > 0.0 ? t.n : -t.n;
    bool only = (t.clip & 2u) != 0u;
    if (only && d >= 0.0) return 2u;
    // At most r (explicit_contact.rs clip_depth): held behind, pushed as at the plane.
    pen = min(only ? -d : nf(r - d), r);
    return 1u;
}

// Anti-clip with FLAG_OWN_TRAVEL (explicit_beams.rs own_travel_reach): contacter h at p
// outside triangle t's box padded by r + (the triangle's fastest node's speed + h's own
// speed, at most the vehicle's fastest) · h + CLIP_DEPTH, the CPU's arithmetic (inside
// the fastest node's box, which the caller tested). False without the setting.
bool ownTravelOut(TriGeo t, uint h, float3 p, float3 origin, float r, uint parity)
{
    if ((t.clip & 1u) == 0u) return false;
    uint4 tn = t.tn;
    if ((gStatic[tn.w * 4u + 3u].x & FLAG_OWN_TRAVEL) == 0u) return false;
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    float tv = clipTriSpeed(tn);
    float vm = clipVmax(tn.w, parity);
    float3 V = gNodes[h * 3u + 1u].xyz;
    float vh = ssqrt(sdot(V, V));
    float sp = vh < vm ? vh : vm;
    float reach = nf(nf(r + smul(nf(tv + sp), STEP)) + CLIP_DEPTH);
    float3 lo = min(min(p0, p1), p2) - float3(reach, reach, reach);
    float3 hi = max(max(p0, p1), p2) + float3(reach, reach, reach);
    return any(p < lo) || any(p > hi);
}

// Contacter node h against triangle t: when h is within r of the face, the hit is
// recorded (the throttle's hit flag, and the vehicle's hit list for selfApply).
void tryContact(TriGeo t, uint h, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount,
                uint parity)
{
#ifdef EB_ABLATE
    // Measurement only (time): 2^27 nothing past the caller's geometry; 2^30 nothing
    // past the exclusions.
    if ((gP.ablate & 134217728u) != 0u) return;
#endif
    if (gP.retireBase != 0u && contactRetired(t.tn, h)) return;
    uint4 tn = t.tn;
    if (h == tn.x || h == tn.y || h == tn.z) return;
    // The tests' loads first, together (none waits on another; the tests have no side
    // effects, so their order is free).
    uint tg = gMisc[t.mi].z & ~(TRI_BREAKABLE | TRI_ANTI_ONLY);
    float hg = gNodes[h * 3u + 2u].w;
    float4 P = gNodes[h * 3u];
    uint4 ex = gP.triExcl != 0u ? gStatic[gP.triExcl + (t.mi - gP.vehicles)] : uint4(0u, 0u, 0u, 0u);
    // JBeam mode: a node skips the triangles of its own self-collision group.
    if (tg != 0u && uint(hg) == tg) return;
    // Spawn exclusions (spec §10.1): the nodes that started inside its range.
    for (uint k = 0u; k < TRI_EXCL_CAP; k++)
    {
        if (k >= ex.y) break;
        if (gLists[ex.x + k] == h) return;
    }
#ifdef EB_ABLATE
    if ((gP.ablate & 1073741824u) != 0u) return;
#endif
    float3 p = nf3(origin + P.xyz);
    if (any(p < t.lo) || any(p > t.hi)) return;
    if (ownTravelOut(t, h, p, origin, r, parity)) return;
#ifdef EB_ABLATE
    // Measurement only: 16777216 no hit test past the box.
    if ((gP.ablate & 16777216u) != 0u) return;
#endif
    float alpha;
    float beta;
    float gamma;
    float delta;
    float3 nn;
    float pen;
    if ((t.clip & 1u) != 0u)
    {
        // Anti-clip (§36): its contact recorded whether or not it takes a force.
        uint word;
        uint c = selfClip(t, h, p, origin, r, alpha, beta, gamma, pen, nn, word);
        if (c == 0u) return;
#if defined(EB_ABLATE) && defined(EB_SELF_PAIRS_ABLATE)
        // Measurement only (the self split's pairs and walk ablation builds alone, where
        // no coupling code runs): 65536 the contact's record (clipRecord's atomics).
        if ((gP.ablate & 65536u) == 0u)
#endif
        clipRecord(h, word);
        if (c == 2u) return;
    }
    else
    {
        if (!triangleHit(t, p, r, alpha, beta, gamma, delta)) return;
        nn = t.n;
        if (delta < 0.0)
        {
            nn = -t.n;
            delta = -delta;
        }
        pen = nf(r - delta);
    }
    gMisc[t.mi].w = 1u;
    gsHit = 1u;
    gsUniformHits = 1u;
    uint v = tn.w;
    // The two-body rule (spec §36): every hit's nodes count it (all of them, even
    // past the hit list's room, for selfSerial).
    bool twoBody = selfTwoBody(v);
    if (twoBody)
    {
        gAcc[gP.selfSplit + h].add(1);
        gAcc[gP.selfSplit + tn.x].add(1);
        gAcc[gP.selfSplit + tn.y].add(1);
        gAcc[gP.selfSplit + tn.z].add(1);
    }
    uint i = uint(gAcc[hitAcc(v)].add(1));
    if (i < HIT_CAP)
    {
        uint at = hitList(v) + i * HIT_WORDS;
        gNext[at] = ((t.mi - gP.vehicles - gStatic[v * 4u + 1u].x) << 16) | (h - gStatic[v * 4u].x);
        gNext[at + 6u] = 0u;
        gNext[at + 8u] = tn.x;
        gNext[at + 9u] = tn.y;
        gNext[at + 10u] = tn.z;
        // The contact's geometry (positions and velocities do not change in this
        // phase): selfApply then needs only the node's force.
        float4 V = gNodes[h * 3u + 1u];
        float4 P0 = gNodes[tn.x * 3u];
        float4 P1 = gNodes[tn.y * 3u];
        float4 P2 = gNodes[tn.z * 3u];
#ifdef EB_DET_HITS
        // The frozen baseline's expressions, as its one-lane oracle writes them.
        float3 vv = V.xyz - (alpha * gNodes[tn.y * 3u + 1u].xyz + beta * gNodes[tn.z * 3u + 1u].xyz
                             + gamma * gNodes[tn.x * 3u + 1u].xyz);
        float mt = alpha * P1.w + beta * P2.w + gamma * P0.w;
        float m = P.w * mt / (P.w + mt);
#else
        // Strict, in the CPU reference's order.
        float3 vv = ssub3(V.xyz, sadd3(sadd3(sscale(gNodes[tn.y * 3u + 1u].xyz, alpha),
                                             sscale(gNodes[tn.z * 3u + 1u].xyz, beta)),
                                       sscale(gNodes[tn.x * 3u + 1u].xyz, gamma)));
        float mt = nf(nf(smul(alpha, P1.w) + smul(beta, P2.w)) + smul(gamma, P0.w));
        float m = sdiv(smul(P.w, mt), nf(P.w + mt));
#endif
        gNext[at + 1u] = asuint(alpha);
        gNext[at + 2u] = asuint(beta);
        gNext[at + 12u] = asuint(nn.x);
        gNext[at + 13u] = asuint(nn.y);
        gNext[at + 14u] = asuint(nn.z);
        gNext[at + 15u] = asuint(pen);
        gNext[at + 16u] = asuint(vv.x);
        gNext[at + 17u] = asuint(vv.y);
        gNext[at + 18u] = asuint(vv.z);
        gNext[at + 19u] = asuint(m);
        gNext[at + 20u] = asuint(V.w);
        // The force as it is when no earlier hit touches h (the usual case), here
        // alongside the other lanes' tests; selfRound keeps it for a hit with no
        // earlier hit on its node. Not while beams of this step are pending (their
        // incidence forces are being rewritten).
        uint prov = 0u;
        if (!twoBody && gAcc[hitAcc(v) + 1u].load() == 0)
        {
            hitForce(v, at, gNext[at], int3(0, 0, 0), 0u, origin, r, beamOff, groupOff, groupCount, parity);
            prov = 1u;
        }
        gNext[at + 7u] = prov;
    }
}

