// --- Aerodynamics and commands with no input (EB_AERO modules) ---
// engine/docs/explicit-beams-aero-commands-spec.md; CPU reference
// engine_softbody::explicit_aero. The specification's double-precision
// sub-expressions run in strict single precision here (WGSL has no f64), so the GPU
// may differ from the CPU by an ulp in them; the angle of attack's acos is the
// device's.
//
//   gStatic at aeroBase per vehicle: (wingOff, wingCount, cmdOff, cmdCount) (wing and
//            command indices global), (front node global, width bits, node count,
//            0), (first slide node global, slide node count, 0, 0); at wingBase per wing: (FLD, FRD, FLU, FRU), (BLD, BRD, BLU, BRU)
//            (global nodes), (efficacy, break reference, table record, induced side
//            0 none 1 left 2 right), (chain span, chain area, 0, 0) as bits; at
//            cmdBase per command: (beam global, contract first, centre bits, 0 for a
//            self-centering command with no input, else contract key | extend key
//            << 8 | self-centering << 16 | 1 << 17 for a vehicle with triggers),
//            (shorten rate, lengthen rate, max contraction, max extension) bits;
//            a table record holds 3601 (cl, cd, cm, 0) as bits.
//   gBeams   at aeroState per vehicle (D, 0); then per wing (broken, 0, 0, 0),
//            induced drag, front force, back force; then per command (mode, 0, 0, 0).
#ifdef EB_AERO

uint wingState(uint gw) { return gP.aeroState + gP.vehicles + gw * 4u; }
uint cmdState(uint gc) { return gP.aeroState + gP.vehicles + gP.wingTotal * 4u + gc; }

float slen(float3 v) { return ssqrt(sdot(v, v)); }
float3 snormalise(float3 v)
{
    float l = slen(v);
    if (l > 0.0) return sscale(v, sdiv(1.0, l));
    return v;
}

// pow≈-based air density at height y; `t0` 288.15 (wings) or 288.1 (fuselage).
float airDensity(float y, float t0)
{
    float t = nf(1.0 - sdiv(smul(0.0065, y), t0));
    float p = smul(101325.0, approxPow(t, 5.24947));
    return smul(p, 0.0000120896);
}

// The signed angle of attack (degrees), as the CPU reference evaluates it.
float angleOfAttack(float3 c, float3 n, float3 b, float3 w)
{
    float3 q = snormalise(scross(n, c));
    float3 g = -w;
    float3 p;
    for (uint i = 0u; i < 3u; i++)
    {
        float3 m = float3(-smul(q[i], q.x), -smul(q[i], q.y), -smul(q[i], q.z));
        m[i] = nf(1.0 - smul(q[i], q[i]));
        p[i] = nf(nf(smul(m.x, g.x) + smul(m.y, g.y)) + smul(m.z, g.z));
    }
    float3 u = c;
    float3 t = -p;
    float a = ssqrt(smul(sdot(u, u), sdot(t, t)));
    float beta = nf(a + sdot(t, u));
    if (abs(nf(beta - smul(2.0, a))) <= 1.1920929e-7 || a == 0.0) return 0.0;
    float qw = beta;
    float3 ax = scross(u, t);
    if (beta < smul(1e-6, a))
    {
        qw = 0.0;
        ax = abs(u.x) > abs(u.z) ? float3(-u.y, u.x, 0.0) : float3(0.0, -u.z, u.y);
    }
    float nq = ssqrt(nf(nf(nf(smul(qw, qw) + smul(ax.x, ax.x)) + smul(ax.y, ax.y)) + smul(ax.z, ax.z)));
    float inv = sdiv(1.0, nq);
    qw = smul(qw, inv);
    ax = sscale(ax, inv);
    float s2 = sdot(ax, ax);
    float theta = 0.0;
    float3 axis = float3(1.0, 0.0, 0.0);
    if (s2 > 0.0)
    {
        float ac = qw >= 1.0 ? 0.0 : (qw <= -1.0 ? 3.14159274 : acos(qw));
        theta = smul(2.0, ac);
        axis = sscale(ax, sdiv(1.0, ssqrt(s2)));
    }
    float alpha = smul(theta, sdiv(180.0, 3.14159274));
    return sdot(axis, b) > 0.0 ? -alpha : alpha;
}

// (cl, cd, cm) of table record `table` at angle a (degrees), no deflection.
float3 airfoilLookup(uint table, float a)
{
    float k = trunc(sdiv(a, 360.0));
    float w = nf(a - smul(k, 360.0));
    if (w > 180.0) w = nf(w - 360.0);
    if (w < -180.0) w = nf(w + 360.0);
    float fi = trunc(smul(nf(w + 180.0), 10.0));
    uint i = uint(clamp(fi, 0.0, 3600.0));
    return asfloat(gStatic[table + i].xyz);
}

// Wing gw's forces this step into its state records (zero when broken).
void wingForces(uint gw, float3 origin)
{
    uint ws = wingState(gw);
    if (gBeams[ws].x != 0.0) return;
    uint4 A = gStatic[gP.wingBase + gw * 4u];
    uint4 B = gStatic[gP.wingBase + gw * 4u + 1u];
    uint4 C = gStatic[gP.wingBase + gw * 4u + 2u];
    float4 D = asfloat(gStatic[gP.wingBase + gw * 4u + 3u]);
    float3 xfld = gNodes[A.x * 3u].xyz;
    float3 xfrd = gNodes[A.y * 3u].xyz;
    float3 xbld = gNodes[B.x * 3u].xyz;
    float3 xbrd = gNodes[B.y * 3u].xyz;
    float3 w = -sscale(sadd3(gNodes[A.x * 3u + 1u].xyz, gNodes[A.y * 3u + 1u].xyz), 0.5);
    float speed = slen(w);
    float3 c = sscale(sadd3(ssub3(xbld, xfld), ssub3(xbrd, xfrd)), 0.5);
    float chord = slen(c);
    float3 b = sscale(sadd3(ssub3(xfrd, xfld), ssub3(xbrd, xbld)), 0.5);
    float span = slen(b);
    float3 l = scross(b, -w);
    float s = smul(span, chord);
    float3 n = snormalise(scross(c, b));
    float3 co = airfoilLookup(C.z, angleOfAttack(c, n, b, w));
    float rho = airDensity(nf(origin.y + xfld.y), 288.15);
    float kd = smul(smul(smul(smul(co.y, 0.5), rho), speed), s);
    float3 wf = sscale(w, kd);
    float3 ind = float3(0.0, 0.0, 0.0);
    if (C.w != 0u)
    {
        float num = smul(smul(smul(smul(smul(smul(co.y, co.y), 0.25), rho), speed), D.y), D.y);
        float den = smul(smul(3.14159, D.x), D.x);
        ind = sscale(w, sdiv(num, den));
    }
    float kl = smul(smul(smul(smul(co.x, 0.5), rho), speed), chord);
    wf = sadd3(wf, sscale(l, kl));
    float mu = smul(smul(smul(smul(smul(-co.z, 0.5), rho), speed), speed), s);
    float e = asfloat(C.x);
    float em = smul(e, mu);
    gBeams[ws + 1u] = float4(ind, 0.0);
    gBeams[ws + 2u] = float4(sadd3(sscale(wf, smul(e, 0.1875)), sscale(n, em)), 0.0);
    gBeams[ws + 3u] = float4(ssub3(sscale(wf, smul(e, 0.0625)), sscale(n, sdiv(em, 3.0))), 0.0);
}

void addForce(uint node, float3 f)
{
    float4 F = gNodes[node * 3u + 2u];
    gNodes[node * 3u + 2u] = float4(sadd3(F.xyz, f), F.w);
}

// One lane adds a vehicle's wing forces in file order, slot by slot, as the CPU.
void wingApply(uint wingOff, uint wingCount)
{
    for (uint k = 0u; k < wingCount; k++)
    {
        uint gw = wingOff + k;
        uint ws = wingState(gw);
        if (gBeams[ws].x != 0.0) continue;
        uint4 A = gStatic[gP.wingBase + gw * 4u];
        uint4 B = gStatic[gP.wingBase + gw * 4u + 1u];
        uint side = gStatic[gP.wingBase + gw * 4u + 2u].w;
        float3 ind = gBeams[ws + 1u].xyz;
        float3 f1 = gBeams[ws + 2u].xyz;
        float3 f2 = gBeams[ws + 3u].xyz;
        if (side == 1u) { addForce(B.z, ind); addForce(B.x, ind); }
        if (side == 2u) { addForce(B.w, ind); addForce(B.y, ind); }
        addForce(A.x, f1); addForce(A.z, f1); addForce(A.y, f1); addForce(A.w, f1);
        addForce(B.x, f2); addForce(B.z, f2); addForce(B.y, f2); addForce(B.w, f2);
    }
}

// The frame adds the wing forces per wing node instead of wingApply's one lane: a
// lane per node adds its entries (wing << 2 | 0 induced drag, 1 front, 2 back) in
// the order wingApply reaches that node (wings in file order, then its slots), so
// each node's force is the same sum, bit for bit.
void wingGather(uint wn)
{
    uint4 rec = gStatic[wingNodeBase() + wn];
    float4 F = gNodes[rec.x * 3u + 2u];
    float3 f = F.xyz;
    for (uint k = 0u; k < rec.z; k++)
    {
        uint e = gLists[rec.y + k];
        uint ws = wingState(e >> 2);
        if (gBeams[ws].x != 0.0) continue;
        f = sadd3(f, gBeams[ws + 1u + (e & 3u)].xyz);
    }
    gNodes[rec.x * 3u + 2u] = float4(f, F.w);
}

// The per-frame break check on world positions (before the origin rebase).
void wingCheck(uint gw, float3 origin)
{
    uint ws = wingState(gw);
    if (gBeams[ws].x != 0.0) return;
    uint4 A = gStatic[gP.wingBase + gw * 4u];
    uint4 B = gStatic[gP.wingBase + gw * 4u + 1u];
    float sref = asfloat(gStatic[gP.wingBase + gw * 4u + 2u].y);
    float3 fld = nf3(origin + gNodes[A.x * 3u].xyz);
    float3 frd = nf3(origin + gNodes[A.y * 3u].xyz);
    float3 bld = nf3(origin + gNodes[B.x * 3u].xyz);
    float3 brd = nf3(origin + gNodes[B.y * 3u].xyz);
    float3 x = ssub3(frd, fld);
    float3 cl = scross(x, ssub3(bld, fld));
    float3 cr = scross(x, ssub3(brd, frd));
    if (sdot(cl, cl) > sref || sdot(cr, cr) > sref) gBeams[ws] = float4(1.0, 0.0, 0.0, 0.0);
}

// The fuselage drag D from the front node's velocity and height, for the next step.
void fuseDrag(uint4 rec, float3 origin, uint vehicle)
{
    float width = asfloat(rec.y);
    if (!(width > 0.0)) return;
    float3 a = -gNodes[rec.x * 3u + 1u].xyz;
    float speed = slen(a);
    float rho = airDensity(nf(origin.y + gNodes[rec.x * 3u].y), 288.1);
    float k = sdiv(smul(smul(smul(smul(smul(width, width), 0.5), 0.5), rho), speed), float(rec.z));
    gBeams[gP.aeroState + vehicle] = float4(sscale(a, k), 0.0);
}

// One self-centering command's step: both entries in key order.
void commandUpdate(uint gc)
{
    uint4 R = gStatic[gP.cmdBase + gc * 2u];
    float4 rates = asfloat(gStatic[gP.cmdBase + gc * 2u + 1u]);
    uint j = R.x;
    float centre = asfloat(R.z);
    float4 s0 = gBeams[j * 2u];
    float rest = s0.x;
    float rest0 = asfloat(gStatic[gP.beamBase + j * 3u + 2u].w);
    float mode = gBeams[cmdState(gc)].x;
    for (uint e = 0u; e < 2u; e++)
    {
        float sigma = (e == 0u) == (R.y != 0u) ? -1.0 : 1.0;
        if (rest0 == 0.0 || rest == 0.0) continue;
        float r = sdiv(rest, rest0);
        if (abs(nf(r - centre)) < 1e-4) mode = 0.0;
        else
        {
            float old = mode;
            mode = r > centre ? -1.0 : 1.0;
            if (old != 0.0 && old != mode)
            {
                rest = smul(centre, rest0);
                mode = 0.0;
            }
        }
        r = sdiv(rest, rest0);
        bool may = (sigma > 0.0 && r < rates.w) || (sigma < 0.0 && r > rates.z);
        if (may)
        {
            float v = sigma * mode > 0.0 ? 1.0 : 0.0;
            float rate = sigma > 0.0 ? rates.y : rates.x;
            float x = sdiv(smul(smul(smul(rate, v), 1.0), STEP), rest);
            rest = nf(fma(sigma * x, rest, rest));
        }
    }
    if (rest != s0.x) gBeams[j * 2u] = float4(rest, s0.yzw);
    gBeams[cmdState(gc)] = float4(mode, 0.0, 0.0, 0.0);
}

// --- shocks2 (engine/docs/explicit-beams-wheels2-shocks2-spec.md §6; CPU
// reference engine_softbody::explicit_beams::Shock2) ---
//   gStatic  the beam's records hold (k in, c in, bump k, bump c) and (short bound,
//            long bound, shock index, L0); at shock2Base per shock (k out, c out,
//            progression k in, progression c in), (progression k out, progression
//            c out, soft bump, 0) as bits.
// Every float expression strict, in the specification's order.

// min(x, 1): 1 if 1 < x, else x (a NaN passes through).
float min1(float x) { return (1.0 < x) ? 1.0 : x; }
// max(a, b): b if a < b, else a.
float maxAB(float a, float b) { return (a < b) ? b : a; }

// k0 + (pk·k0)·λ and c0 + (pc·c0)·λ, λ = 1 for a zero bound, else min((e / (bound·L))², 1).
float2 shock2Progressive(float k0, float c0, float pk, float pc, float e, float bound, float rest)
{
    float lambda = 1.0;
    if (bound != 0.0)
    {
        float r = sdiv(e, smul(bound, rest));
        lambda = min1(smul(r, r));
    }
    return float2(nf(k0 + smul(smul(pk, k0), lambda)), nf(c0 + smul(smul(pc, c0), lambda)));
}

// The soft bump on (k, c): k + ((k + 100)·pk)·μ, μ = 1 for a zero bound, else
// min((((e − e_b)·5) / (bound·L))², 1).
float2 shock2Bump(float2 kc, float pk, float pc, float excess, float bound, float rest)
{
    float mu = 1.0;
    if (bound != 0.0)
    {
        float q = sdiv(smul(excess, 5.0), smul(bound, rest));
        mu = min1(smul(q, q));
    }
    return float2(nf(kc.x + smul(smul(nf(kc.x + 100.0), pk), mu)), nf(kc.y + smul(smul(nf(kc.y + 100.0), pc), mu)));
}

void shock2Update(uint gj, uint local, bool fast, uint groupOff, uint parity, uint4 bn, float4 s1)
{
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    // The host writes shock indices below shock2Count; clamp in any case.
    uint si = min(r2.z, max(gP.shock2Count, 1u) - 1u);
    float4 t0 = asfloat(gStatic[gP.shock2Base + si * 2u]);
    float4 t1 = asfloat(gStatic[gP.shock2Base + si * 2u + 1u]);
    float kIn = c1.x;
    float cIn = c1.y;
    float kBump = c1.z;
    float cBump = c1.w;
    float sb = asfloat(r2.x);
    float lb = asfloat(r2.y);
    float kOut = t0.x;
    float cOut = t0.y;
    float pkIn = t0.z;
    float pcIn = t0.w;
    float pkOut = t1.x;
    float pcOut = t1.y;
    bool soft = t1.z != 0.0;
    int group = asint(bn.w);
    float4 s0 = gBeams[gj * 2u];
    float rest = s0.x;
    float gate = s0.w;
    float strength = s1.x;
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float inv = fast ? sFastInvSqrt(q) : sdiv(1.0, ssqrt(q));
    float len = smul(q, inv);
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    // §6.1: the velocity picks the side.
    float2 kc = (u > 0.0) ? shock2Progressive(kOut, cOut, pkOut, pcOut, e, lb, rest)
                          : shock2Progressive(kIn, cIn, pkIn, pcIn, e, sb, rest);
    bool past = e > smul(lb, rest) || e < smul(-sb, rest);
    if (!soft)
    {
        // §6.2: the bump rates replace.
        if (past) kc = float2(kBump, cBump);
    }
    else
    {
        // §6.3.
        float softRest = smul(rest, 0.8);
        float el = smul(lb, softRest);
        float es = smul(-sb, softRest);
        if (e > el)
        {
            kc = shock2Bump(shock2Progressive(kOut, cOut, pkOut, pcOut, e, lb, rest), pkOut, pcOut, nf(e - el), lb,
                            rest);
            if (u < 0.0) kc = float2(kIn, cIn);
        }
        else if (e < es)
        {
            // The out factors here too (quirk).
            kc = shock2Bump(shock2Progressive(kIn, cIn, pkIn, pcIn, e, sb, rest), pkOut, pcOut, nf(e - es), sb,
                            rest);
            if (u > 0.0) kc = float2(kOut, cOut);
        }
        if (past) kc = float2(maxAB(kc.x, kBump), maxAB(kc.y, cBump));
    }
    float f = nf(smul(-kc.x, e) - smul(kc.y, u));
    float last = f;
    float tau = abs(f);
    bool broken = false;
    // No yield; the break test when |F| exceeds the gate and the strength.
    if (tau > gate && tau > strength)
    {
        uint ex = breakException(gj, bn, local, groupOff, parity);
        if (ex == 2u) return;
        bool saved = ex == 1u;
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
                gGroups[groupSlot(parity, groupOff, 0u)].min(local);
            }
        }
    }
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// --- Triggers, key-driven commands and slide nodes
// (engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md §2–3, §6; CPU
// reference engine_softbody::explicit_triggers) ---
//   gStatic  a trigger beam's third record holds (short bound, long bound, short key
//            | long key << 8 | enabled << 16, L0); at slideBase per slide node
//            (node global, first segment global or NONE, 0, 0), (spring, break
//            force, tolerance, 0) bits; at segBase per segment (beam global,
//            previous segment global or NONE, next or NONE, 0).
//   gNext    at keyBase per vehicle and step parity KEYS trigger inputs (1: held):
//            the beam phase of step s writes parity s & 1, the command phase of
//            step s + 1 reads it, the beam phase of step s + 1 clears it.
//   gBeams   at slideState per slide node (current segment or -1, broken away, 0,
//            0); at slideAcc per slide target node (a slide node or a rail end:
//            node info flags >> NODE_SLOT_SHIFT, 1-based) the slide forces of the
//            last step, added by its node phase and cleared.
// One lane per vehicle runs its slide nodes in spawn order during the
// self-collision phase (positions are final for the step there, and the beam
// phase's breaks are visible), writing only the slots, which no other phase reads
// before the next node phase. Every loop is bounded by a count from the static
// records, every index clamped.
static const uint KEYS = 85u;
static const uint NONE = 0xFFFFFFFFu;

uint keySlot(uint vehicle, uint parity, uint key)
{
    return gP.keyBase + (vehicle * 2u + parity) * KEYS + min(key, KEYS - 1u);
}

// A trigger beam: zero force (k = c = 0), then the bounds test for the next step's
// command pass (spec §2.5).
void triggerUpdate(uint gj, bool fast, uint parity, uint4 bn)
{
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    float4 s0 = gBeams[gj * 2u];
    float rest = s0.x;
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float inv = fast ? sFastInvSqrt(q) : sdiv(1.0, ssqrt(q));
    float len = smul(q, inv);
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    float f = nf(smul(-c1.x, e) - smul(c1.y, u));
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
    uint keys = r2.z;
    if (((keys >> 16) & 1u) == 0u) return;
    float sb = asfloat(r2.x);
    float lb = asfloat(r2.y);
    uint vehicle = bn.z >> 16;
    if (e > smul(lb, rest)) gNext[keySlot(vehicle, parity, (keys >> 8) & 0xFFu)] = 1u;
    else if (e < smul(-sb, rest)) gNext[keySlot(vehicle, parity, keys & 0xFFu)] = 1u;
}

// One command line's step with key input (spec §2.6): both entries in key order,
// the keys' values from the triggers of the last step (parity `prev`).
void commandUpdateKeys(uint gc, uint vehicle, uint prev)
{
    uint4 R = gStatic[gP.cmdBase + gc * 2u];
    float4 rates = asfloat(gStatic[gP.cmdBase + gc * 2u + 1u]);
    uint j = R.x;
    float centre = asfloat(R.z);
    uint ck = R.w & 0xFFu;
    uint ek = (R.w >> 8) & 0xFFu;
    bool centering = ((R.w >> 16) & 1u) != 0u;
    float vc = ck == 0u ? 0.0 : float(gNext[keySlot(vehicle, prev, ck)]);
    float ve = ek == 0u ? 0.0 : float(gNext[keySlot(vehicle, prev, ek)]);
    bool locked = centering && (vc >= 0.5 || ve >= 0.5);
    float4 s0 = gBeams[j * 2u];
    float rest = s0.x;
    float rest0 = asfloat(gStatic[gP.beamBase + j * 3u + 2u].w);
    float mode = gBeams[cmdState(gc)].x;
    if (locked) mode = 0.0;
    for (uint e = 0u; e < 2u; e++)
    {
        float sigma = (e == 0u) == (R.y != 0u) ? -1.0 : 1.0;
        if (rest0 == 0.0 || rest == 0.0) continue;
        if (centering && !locked)
        {
            float r = sdiv(rest, rest0);
            if (abs(nf(r - centre)) < 1e-4) mode = 0.0;
            else
            {
                float old = mode;
                mode = r > centre ? -1.0 : 1.0;
                if (old != 0.0 && old != mode)
                {
                    rest = smul(centre, rest0);
                    mode = 0.0;
                }
            }
        }
        float r = sdiv(rest, rest0);
        bool may = (sigma > 0.0 && r < rates.w) || (sigma < 0.0 && r > rates.z);
        if (may)
        {
            float v = sigma > 0.0 ? ve : vc;
            if (centering && !locked && sigma * mode > 0.0) v = 1.0;
            float rate = sigma > 0.0 ? rates.y : rates.x;
            float x = sdiv(smul(smul(smul(rate, v), 1.0), STEP), rest);
            rest = nf(fma(sigma * x, rest, rest));
        }
    }
    if (rest != s0.x) gBeams[j * 2u] = float4(rest, s0.yzw);
    gBeams[cmdState(gc)] = float4(mode, 0.0, 0.0, 0.0);
}

// Slot `slot`'s value `s` plus the contributions slideNode registered this step,
// added in spawn order (slide node, then own node, rail end a, rail end b: the
// order the serial pass adds them in); clears the registration.
float3 slidePending(float3 s, uint slot, uint n)
{
    uint4 rec = gStatic[slotRecBase() + slot];
    uint m = min(n, rec.y);
    for (uint i = 1u; i < m; i++)
    {
        uint key = gNext[gP.slotListBase + rec.x + i];
        uint j = i;
        while (j > 0u && gNext[gP.slotListBase + rec.x + j - 1u] > key)
        {
            gNext[gP.slotListBase + rec.x + j] = gNext[gP.slotListBase + rec.x + j - 1u];
            j--;
        }
        gNext[gP.slotListBase + rec.x + j] = key;
    }
    for (uint i = 0u; i < m; i++) s = sadd3(s, gBeams[slideContribBase() + gNext[gP.slotListBase + rec.x + i]].xyz);
    gAcc[gP.slotCntBase + slot].store(0);
    return s;
}

// The node phase: add and clear a slide target's forces (slot + 1 = slot1).
float3 slideTake(float3 f, uint slot1)
{
    uint slot = min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u);
    uint at = gP.slideAcc + slot;
    float3 s = gBeams[at].xyz;
    uint n = uint(gAcc[gP.slotCntBase + slot].load());
    if (n != 0u) s = slidePending(s, slot, n);
    gBeams[at] = float4(0.0, 0.0, 0.0, 0.0);
    return sadd3(f, s);
}

void slideAdd(uint node, float3 f)
{
    uint slot1 = gStatic[gP.nodeInfoBase + node].x >> NODE_SLOT_SHIFT;
    if (slot1 == 0u) return;
    uint at = gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u);
    gBeams[at] = float4(sadd3(gBeams[at].xyz, f), 0.0);
}

// Beam j is broken as the CPU sees it after this step's beam pass: its flag, or a
// detacher-group break this step (the next node phase marks those).
bool brokenNow(uint j, uint groupOff, uint parity)
{
    if (gBeams[j * 2u + 1u].z != 0.0) return true;
    int g = asint(gStatic[gP.beamBase + j * 3u].w);
    if (g == 0) return false;
    return gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load() != NO_GROUP;
}

// The nearest point of segment gs (world positions) to x, its distance and ratio.
struct SegPoint
{
    float3 q;
    float dist;
    float ratio;
};

SegPoint segmentPoint(uint gs, float3 origin, float3 x)
{
    uint j = gStatic[gP.segBase + gs].x;
    uint4 bn = gStatic[gP.beamBase + j * 3u];
    float3 p1 = nf3(origin + gNodes[bn.x * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[bn.y * 3u].xyz);
    float3 d = ssub3(p2, p1);
    float beta = slen(d);
    if (beta > 0.0) d = sscale(d, sdiv(1.0, beta));
    float proj = sdot(ssub3(x, p1), d);
    float m = (beta < proj) ? beta : proj;
    float tau = (0.0 < m) ? m : 0.0;
    SegPoint r;
    r.q = sadd3(p1, sscale(d, tau));
    r.dist = slen(ssub3(r.q, x));
    r.ratio = beta > 0.0 ? sdiv(tau, beta) : 0.0;
    return r;
}

uint clampSeg(uint gs) { return min(gs, max(gP.segTotal, 1u) - 1u); }

// A vehicle's slide nodes in spawn order (spec §3.4), one lane.
void slidePass(uint slideOff, uint slideCount, float3 origin, uint groupOff, uint parity)
{
    uint count = min(slideCount, gP.slideTotal - min(slideOff, gP.slideTotal));
    for (uint k = 0u; k < count; k++)
    {
        uint gs = slideOff + k;
        float4 st = gBeams[gP.slideState + gs];
        if (st.x < 0.0) continue;
        uint cur = clampSeg(uint(st.x));
        uint4 seg = gStatic[gP.segBase + cur];
        if (brokenNow(seg.x, groupOff, parity)) continue;
        uint4 S0 = gStatic[gP.slideBase + gs * 2u];
        float4 S1 = asfloat(gStatic[gP.slideBase + gs * 2u + 1u]);
        float3 x = nf3(origin + gNodes[S0.x * 3u].xyz);
        // At most one segment's hop.
        uint best = cur;
        SegPoint bp = segmentPoint(cur, origin, x);
        if (seg.y != NONE)
        {
            SegPoint p = segmentPoint(clampSeg(seg.y), origin, x);
            if (p.dist < bp.dist) { best = clampSeg(seg.y); bp = p; }
        }
        if (seg.z != NONE)
        {
            SegPoint p = segmentPoint(clampSeg(seg.z), origin, x);
            if (p.dist < bp.dist) { best = clampSeg(seg.z); bp = p; }
        }
        bool away = st.y != 0.0;
        if (!away)
        {
            float3 w = ssub3(bp.q, x);
            float delta = slen(w);
            w = snormalise(w);
            float stretch = nf(delta - S1.z);
            float fLen = smul(-S1.x, (0.0 < stretch) ? stretch : 0.0);
            float3 P = sscale(w, fLen);
            if (slen(P) > S1.y) away = true;
            uint4 bn = gStatic[gP.beamBase + gStatic[gP.segBase + best].x * 3u];
            slideAdd(S0.x, -P);
            slideAdd(bn.x, sscale(P, nf(1.0 - bp.ratio)));
            slideAdd(bn.y, sscale(P, bp.ratio));
        }
        gBeams[gP.slideState + gs] = float4(float(best), away ? 1.0 : 0.0, 0.0, 0.0);
    }
}

// The frame runs the slide nodes in parallel, one lane each: slideNode is one
// iteration of slidePass, but instead of adding into the slots in spawn order it
// stores its contributions (at slideContrib, 3 per slide node) and registers each
// with its slot (a count and a list, capacity: every contribution the slot can
// get); the next node phase (slideTake) or the frame's end (slideSettle) adds them
// in spawn order. The same sums, bit for bit; slide nodes do not read the slots.
void slideEmit(uint node, float3 f, uint gs, uint which)
{
    uint slot1 = gStatic[gP.nodeInfoBase + node].x >> NODE_SLOT_SHIFT;
    if (slot1 == 0u) return;
    uint slot = min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u);
    uint key = gs * 3u + which;
    gBeams[slideContribBase() + key] = float4(f, 0.0);
    uint4 rec = gStatic[slotRecBase() + slot];
    uint i = uint(gAcc[gP.slotCntBase + slot].add(1));
    if (i < rec.y) gNext[gP.slotListBase + rec.x + i] = key;
}

void slideNode(uint gs, float3 origin, uint groupOff, uint parity)
{
    float4 st = gBeams[gP.slideState + gs];
    if (st.x < 0.0) return;
    uint cur = clampSeg(uint(st.x));
    uint4 seg = gStatic[gP.segBase + cur];
    if (brokenNow(seg.x, groupOff, parity)) return;
    uint4 S0 = gStatic[gP.slideBase + gs * 2u];
    float4 S1 = asfloat(gStatic[gP.slideBase + gs * 2u + 1u]);
    float3 x = nf3(origin + gNodes[S0.x * 3u].xyz);
    uint best = cur;
    SegPoint bp = segmentPoint(cur, origin, x);
    if (seg.y != NONE)
    {
        SegPoint p = segmentPoint(clampSeg(seg.y), origin, x);
        if (p.dist < bp.dist) { best = clampSeg(seg.y); bp = p; }
    }
    if (seg.z != NONE)
    {
        SegPoint p = segmentPoint(clampSeg(seg.z), origin, x);
        if (p.dist < bp.dist) { best = clampSeg(seg.z); bp = p; }
    }
    bool away = st.y != 0.0;
    if (!away)
    {
        float3 w = ssub3(bp.q, x);
        float delta = slen(w);
        w = snormalise(w);
        float stretch = nf(delta - S1.z);
        float fLen = smul(-S1.x, (0.0 < stretch) ? stretch : 0.0);
        float3 P = sscale(w, fLen);
        if (slen(P) > S1.y) away = true;
        uint4 bn = gStatic[gP.beamBase + gStatic[gP.segBase + best].x * 3u];
        slideEmit(S0.x, -P, gs, 0u);
        slideEmit(bn.x, sscale(P, nf(1.0 - bp.ratio)), gs, 1u);
        slideEmit(bn.y, sscale(P, bp.ratio), gs, 2u);
    }
    gBeams[gP.slideState + gs] = float4(float(best), away ? 1.0 : 0.0, 0.0, 0.0);
}

// The frame's end: add a slot's registered contributions into it (as the serial
// pass would have left it).
void slideSettle(uint slot)
{
    uint n = uint(gAcc[gP.slotCntBase + slot].load());
    if (n == 0u) return;
    uint at = gP.slideAcc + slot;
    gBeams[at] = float4(slidePending(gBeams[at].xyz, slot, n), 0.0);
}

// nodeUpdate for a vehicle with a `fusedrag` line: the restart adds the stored
// fuselage drag D instead of the air drag and turbulence.
void nodeUpdateFuse(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits)
{
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, parity, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
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
    f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
}
#ifdef EB_VV_SOLO
uint nodeUpdateFuseSolo(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits)
{
    float4 P = gNodes[gi * 3u];
    // The position the last step left (each step's checked once: the frame's
    // last at its end).
    if (any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
    float4 V = gNodes[gi * 3u + 1u];
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, parity, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
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
    f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
    return 0u;
}
#endif
#endif

// --- Vehicle–vehicle contact: records the frame entry uses (the rest of the
// section is at the end of the file) ---

// Per vehicle VV_RECS records in gMisc at vvVeh: its padded frame box (lo, hi),
// the bounds its nodes must stay in this frame (lo, hi), flags (x: too big for
// the vehicle grid, y: grouped this frame, z: its group, w: its self-collision
// throttles reset at the next throttle update), member (x: its index in its
// group's sorted members).
static const uint VV_RECS = 6u;
static const uint VV_P_LO = 0u;
static const uint VV_P_HI = 1u;
static const uint VV_ALLOWED_LO = 2u;
static const uint VV_ALLOWED_HI = 3u;
static const uint VV_FLAGS = 4u;
static const uint VV_MEMBER = 5u;
uint vvRec(uint v, uint r) { return gP.vvVeh + v * VV_RECS + r; }

// Counters in gAcc after the box keys: overlapping pairs, big vehicles, solo
// vehicles, groups, a vehicle outside its bounds (index + 1), pair overflow, no
// convergence of the grouping.
static const uint VV_PAIRS = 0u;
static const uint VV_BIGS = 1u;
static const uint VV_SOLOS = 2u;
static const uint VV_GROUPS = 3u;
static const uint VV_OUTSIDE = 4u;
static const uint VV_OVERFLOW = 5u;
static const uint VV_UNCONVERGED = 6u;
static const uint VV_TOO_BIG = 7u;
static const uint VV_CTL_WORDS = 8u;
uint vvCtl(uint i) { return gP.vvAcc + gP.vehicles * 12u + i; }

// A value one lane decided, read by every lane as uniform (the WGSL
// post-process turns its reads into workgroupUniformLoad).
groupshared uint gsUniformLanes;

