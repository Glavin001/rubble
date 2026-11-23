// A coupler (the CPU reference's coupler_force; spec §7): a zero-length
// spring-damper (c1 = k, c) that breaks past its strength. `setOff`: a broken active
// coupler sets off its group now; a coupler in several break groups (an extension
// record, spec §7.1) leaves it to extPending (s1.w = 1), as other EXT_GROUPS beams.
void jcouplerUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1, float4 c1,
                    float3 pa, float3 pb, float3 va, float3 vb, bool setOff)
{
    int group = asint(bn.w);
    float strength = s1.x;
    bool broken = false;
    float3 fc = ssub3(sscale(ssub3(pa, pb), -c1.x), sscale(ssub3(va, vb), c1.y));
    float m = ssqrt(sdot(fc, fc));
    if (m > strength)
    {
        broken = true;
        fc = float3(0.0, 0.0, 0.0);
        if (setOff && group > 0)
        {
            gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
            gGroups[groupSlot(parity, groupOff, 0u)].min(local);
        }
    }
    if (broken) gBeams[gj * 2u + 1u] = float4(strength, m, 1.0, 1.0);
    else if (s1.w != 0.0) gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(fc, !broken && centredOn(bn) ? c1.y : 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-fc, 0.0);
}

// A JBeam-mode beam (kinds KIND_JPLAIN..KIND_JCOUPLER; the CPU reference's
// jbeam_force_filtered, coupler_force and beam_pass; engine/docs/jbeam-semantics-spec.md).
// Exact lengths (JBeam mode spawns without the fast inverse square root); no break
// exception. Records: (k, c, limit k or the deformLimitStress cap, limit c),
// (short, long, bound zone, damping filter record + 1 or 0); plain and support: the
// rest-length limits and the support's break length in metres; bounded: the bounds as
// fractions of the rest length, the velocity split and fast damping in Tc and Tt. A
// filter record in gBeams is (last rate, last output, a, b).
void jbeamUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1)
{
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    float4 c2 = asfloat(r2);
    float4 s0 = gBeams[gj * 2u];
    uint kind = bn.z & 0xFFu;
    uint type = (bn.z >> 8) & 0xFFu;
    int group = asint(bn.w);
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    bool broken = false;
    float strength = s1.x;
    if (kind == KIND_JTORSION2) return; // its element's first record writes its forces
    if (kind == KIND_JTORSION)
    {
        jtorsionUpdate(gj, local, groupOff, parity, bn, s1, c1, r2, pa.xyz, pb.xyz, va, vb);
        return;
    }
    if (kind == KIND_JCOUPLER)
    {
        jcouplerUpdate(gj, local, groupOff, parity, bn, s1, c1, pa.xyz, pb.xyz, va, vb, true);
        return;
    }
    float rest = s0.x;
    float tc = s0.y;
    float tt = s0.z;
    float gate = s0.w;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    // The exact length and inverse length (jbeam-semantics-spec.md §35; the CPU's
    // exact_length), bit for bit.
    float2 li = sExactLength(q);
    float inv = li.y;
    float len = li.x;
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    // The damping's slope in u and the low-pass's gain (centred damping, §39).
    float gain = 1.0;
    if (r2.w != 0u)
    {
        // dampCutoffHz: the RC low-pass of the rate (spec §6).
        uint fi = r2.w - 1u;
        float4 flt = gBeams[fi];
        float y = nf(smul(flt.z, u) - smul(flt.w, flt.y));
        gBeams[fi] = float4(u, y, flt.z, flt.w);
        u = y;
        gain = flt.z;
    }
    float k = c1.x;
    float c = c1.y;
    float damp = smul(c, u);
    float slope = c;
    float last = 0.0;
    float f = 0.0;
    bool lengthBreak = false;
    if (kind == KIND_JBOUNDED)
    {
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
        if (len > c2.y) lengthBreak = true;
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
                // A negative bound tears there (breakAtExpansion, §36).
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
                    if (!keepStrength(bn)) strength = nf(strength - smul(nf(rest - old), k));
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
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, !broken && centredOn(bn) ? smul(slope, gain) : 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// Set off group g (> 0) as master `local` in this beam phase.
void extSetOff(int g, uint local, uint groupOff, uint parity)
{
    if (g <= 0) return;
    gGroups[groupSlot(parity, groupOff, uint(g))].min(local);
    gGroups[groupSlot(parity, groupOff, 0u)].min(local);
}

// A beam with several break groups that broke on its own in the last beam phase
// (spec §17): sets off all its groups now, unless one of them had broken it first
// (a master before it in the last step's slots).
void extPending(uint gj, uint local, uint groupOff, uint parity)
{
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    if ((bn.z & (TYPE_EXT << 8)) == 0u) return;
    uint x = gLists[gP.extIndex + gj] - 1u;
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    if ((e0.x & EXT_GROUPS) == 0u) return;
    uint4 e3 = gStatic[gP.extBase + x * 4u + 3u];
    int g0 = asint(bn.w);
    int g1 = asint(e3.z);
    int g2 = asint(e3.w);
    uint last = parity ^ 1u;
    uint m = NO_GROUP;
    if (g0 != 0) m = min(m, gGroups[groupSlot(last, groupOff, uint(abs(g0)))].load());
    if (g1 != 0) m = min(m, gGroups[groupSlot(last, groupOff, uint(abs(g1)))].load());
    if (g2 != 0) m = min(m, gGroups[groupSlot(last, groupOff, uint(abs(g2)))].load());
    if (m != NO_GROUP && m < local) return;
    extSetOff(g0, local, groupOff, parity);
    extSetOff(g1, local, groupOff, parity);
    extSetOff(g2, local, groupOff, parity);
}

// A JBeam beam with an extension record (the CPU reference's ext_beam,
// jbeam_ext_force, lbeam_force and the pressure group; spec §16-§26).
void jbeamExtUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1)
{
    uint kind = bn.z & 0xFFu;
    uint x = gLists[gP.extIndex + gj] - 1u;
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    float4 e1 = asfloat(gStatic[gP.extBase + x * 4u + 1u]);
    float4 e2 = asfloat(gStatic[gP.extBase + x * 4u + 2u]);
    uint4 e3 = gStatic[gP.extBase + x * 4u + 3u];
    if (kind == KIND_JPTRI2 || kind == KIND_JATRI2 || kind == KIND_JLBEAM2) return; // the first record writes the forces
    if (kind == KIND_JCOUPLER)
    {
        float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
        jcouplerUpdate(gj, local, groupOff, parity, bn, s1, c1, gNodes[bn.x * 3u].xyz, gNodes[bn.y * 3u].xyz,
                       gNodes[bn.x * 3u + 1u].xyz, gNodes[bn.y * 3u + 1u].xyz, false);
        return;
    }
    if (kind == KIND_JATRI)
    {
        // A triangle's drag and lift in still air (explicit_jbeam_ext.rs
        // aero_tri_force): E1 = (x2, drag factor, lift factor, sin stall).
        uint n2 = asuint(e1.x);
        float3 x0 = gNodes[bn.x * 3u].xyz;
        float3 x1 = gNodes[bn.y * 3u].xyz;
        float3 x2 = gNodes[n2 * 3u].xyz;
        float3 v0 = gNodes[bn.x * 3u + 1u].xyz;
        float3 v1 = gNodes[bn.y * 3u + 1u].xyz;
        float3 v2 = gNodes[n2 * 3u + 1u].xyz;
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
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    if (kind == KIND_JPGROUP)
    {
        // The enclosed volume about node a, isothermal gauge pressure, capped.
        float3 c = gNodes[bn.x * 3u].xyz;
        uint first = asuint(e1.x);
        uint count = asuint(e1.y);
        float v = 0.0;
        for (uint t = 0u, tStop = EB_END(7, t, count, 1u); t < tStop; t++)
        {
            uint r = first + 2u * t;
            uint4 ra = gStatic[gP.beamBase + r * 3u];
            uint4 rb = gStatic[gP.beamBase + (r + 1u) * 3u];
            float3 x0 = ssub3(gNodes[ra.x * 3u].xyz, c);
            float3 x1 = ssub3(gNodes[ra.y * 3u].xyz, c);
            float3 x2 = ssub3(gNodes[rb.x * 3u].xyz, c);
            v = nf(v + sdot(x0, scross(x1, x2)));
        }
        v = max(sdiv(v, 6.0), smul(0.01, e1.z));
        float p = min(nf(smul(e1.w, sdiv(e1.z, v)) - ATMOSPHERE), e2.x);
        float4 st = gBeams[gP.extState + x];
        if (parity == 0u) st.y = p;
        else st.z = p;
        gBeams[gP.extState + x] = st;
        return;
    }
    if (kind == KIND_JPTRI)
    {
        uint g = asuint(e1.y);
        uint gx = gLists[gP.extIndex + g] - 1u;
        float4 gs = gBeams[gP.extState + gx];
        float p = parity == 0u ? gs.z : gs.y; // the last step's
        uint n2 = asuint(e1.x);
        float3 x0 = gNodes[bn.x * 3u].xyz;
        float3 x1 = gNodes[bn.y * 3u].xyz;
        float3 x2 = gNodes[n2 * 3u].xyz;
        float3 f = sscale(scross(ssub3(x1, x0), ssub3(x2, x0)), sdiv(p, 6.0));
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    float4 c2 = asfloat(r2);
    float4 s0 = gBeams[gj * 2u];
    uint type = (bn.z >> 8) & 3u;
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
    float4 st = gBeams[gP.extState + x];
    bool rampOn = st.x > 0.0;
    if (rampOn)
    {
        rest = nf(rest + asfloat(e3.x));
        st.x = st.x - 1.0;
        gBeams[gP.extState + x] = st;
    }
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float2 li = sExactLength(q); // spec §35, the CPU's exact_length
    float inv = li.y;
    float len = li.x;
    float u = smul(sdot(ssub3(va, vb), d), inv);
    // The damping's slope in u and the low-pass's gain (centred damping, §39).
    float gain = 1.0;
    if (r2.w != 0u)
    {
        uint fi = r2.w - 1u;
        float4 flt = gBeams[fi];
        float y = nf(smul(flt.z, u) - smul(flt.w, flt.y));
        gBeams[fi] = float4(u, y, flt.z, flt.w);
        u = y;
        gain = flt.z;
    }
    float k = c1.x;
    float c = c1.y;
    float slope = c;
    float e = 0.0;
    float f = 0.0;
    float last = 0.0;
    bool broken = false;
    bool tear = false;
    bool lengthBreak = false;
    bool done = false;
    float3 armA = float3(0.0, 0.0, 0.0);
    float3 armB = float3(0.0, 0.0, 0.0);
    if (kind == KIND_JLBEAM)
    {
        // The arms' law-of-cosines length at the rest angle (rest holds cos θ0).
        float3 xc = gNodes[asuint(e1.x) * 3u].xyz;
        float3 ra = ssub3(pa.xyz, xc);
        float3 rb = ssub3(pb.xyz, xc);
        float a2 = sdot(ra, ra);
        float b2 = sdot(rb, rb);
        float ab = ssqrt(smul(a2, b2));
        float lr = ssqrt(max(nf(nf(a2 + b2) - smul(smul(2.0, ab), rest)), 0.0));
        e = nf(len - lr);
        float spring = smul(k, e);
        f = nf(-spring - smul(c, u));
        if (abs(spring) > tc && ab > 1e-12)
        {
            float m = sdiv(tc, k);
            m = e < 0.0 ? -m : m;
            float l = nf(len - m);
            rest = clamp(sdiv(nf(nf(a2 + b2) - smul(l, l)), smul(2.0, ab)), -1.0, 1.0);
            f = nf((e < 0.0 ? tc : -tc) - smul(c, u));
            spring = e < 0.0 ? -tc : tc;
        }
        // The spring's arm terms (the gradient of ½k·e² on the arms; lbeam_force):
        // spring·∂L/∂a·â on id1, spring·∂L/∂b·b̂ on id2, their negative sum on id3.
        float la = ssqrt(a2);
        float lb = ssqrt(b2);
        float lnow = ssqrt(max(nf(nf(a2 + b2) - smul(smul(2.0, ab), rest)), 0.0));
        if (lnow > 1e-9 && la > 1e-12 && lb > 1e-12)
        {
            float g = sdiv(1.0, lnow);
            armA = sscale(sscale(ra, smul(g, nf(1.0 - smul(sdiv(lb, la), rest)))), spring);
            armB = sscale(sscale(rb, smul(g, nf(1.0 - smul(sdiv(la, lb), rest)))), spring);
        }
        last = f;
        if (abs(f) > strength) broken = true;
        done = true;
    }
    else
    {
        e = nf(len - rest);
    }
    float damp = smul(c, u);
    if (!done && kind == KIND_JPRESSURED)
    {
        float l0 = e1.w;
        float den = max(nf(l0 - smul(e1.z, nf(l0 - len))), smul(0.01, l0));
        float gauge = min(nf(smul(e1.y, sdiv(l0, den)) - ATMOSPHERE), e2.x);
        f = nf(nf(smul(gauge, e1.x) - smul(k, max(e, 0.0))) - damp);
        last = f;
        if (gauge > e2.y || abs(f) > strength) broken = true;
        done = true;
    }
    if (!done && kind == KIND_JBOUNDED)
    {
        float cs = c;
        float cf = tt;
        float split = tc;
        float lc = c1.w;
        if ((e0.x & EXT_REBOUND) != 0u && u > 0.0)
        {
            cs = e2.x;
            cf = e2.y;
            split = e2.z;
            lc = e2.w;
        }
        damp = smul(cs, u);
        slope = cs;
        float au = abs(u);
        if (au > split)
        {
            float m = nf(smul(cs, split) + smul(cf, nf(au - split)));
            damp = u < 0.0 ? -m : m;
            slope = cf;
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
            damp = nf(damp + smul(nf(smul(lc, u) - damp), sz));
            slope = nf(slope + smul(nf(lc - slope), sz));
        }
    }
    else if (!done && kind == KIND_JSUPPORT && e > 0.0)
    {
        if (len > c2.y) lengthBreak = true;
        k = 0.0;
        damp = 0.0;
        slope = 0.0;
    }
    else if (!done && kind == KIND_JROPE && e < 0.0)
    {
        k = 0.0;
        damp = smul(smul(c, 0.1), u);
        slope = smul(c, 0.1);
    }
    else if (!done && kind == KIND_JANISO)
    {
        if (len > e1.w) lengthBreak = true;
        // Past the spawned length (§20): rest + (E2 .y + ramp step · steps left).
        float sw = nf(rest + nf(e2.y + smul(asfloat(e3.x), st.x)));
        float past = nf(len - sw);
        if (past > 0.0)
        {
            float zone = smul(e1.z, sw);
            float sz = zone > 0.0 ? min(sdiv(past, zone), 1.0) : 1.0;
            k = nf(k + smul(nf(e1.x - k), sz));
            slope = nf(c + smul(nf(e1.y - c), sz));
            damp = smul(slope, u);
        }
    }
    if (lengthBreak)
    {
        broken = true;
        f = 0.0;
        last = 0.0;
    }
    else if (!done)
    {
        f = nf(smul(-k, e) - damp);
        last = f;
        float tau = abs(f);
        if (tau > gate && type == TYPE_DEFORMABLE && k != 0.0)
        {
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
                // A negative bound tears there (breakAtExpansion, §36).
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
                    if (!keepStrength(bn)) strength = nf(strength - smul(nf(rest - old), k));
                }
            }
        }
        if (tau > strength || tear) broken = true;
    }
    if (broken)
    {
        f = 0.0;
        if ((e0.x & EXT_GROUPS) == 0u) extSetOff(group, local, groupOff, parity);
    }
    if ((e0.x & EXT_DEFORM) != 0u)
    {
        bool yields = kind == KIND_JPLAIN || kind == KIND_JROPE || kind == KIND_JSUPPORT || kind == KIND_JANISO;
        if (broken || (yields && !(st.x > 0.0) && abs(nf(rest - e2.x)) > smul(asfloat(e0.w), e2.x))) extDeform(x, 0u);
    }
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    float slopeW = !broken && centredOn(bn) ? smul(slope, gain) : 0.0;
    if (kind == KIND_JLBEAM)
    {
        if (broken)
        {
            armA = float3(0.0, 0.0, 0.0);
            armB = float3(0.0, 0.0, 0.0);
        }
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(sadd3(push, armA), slopeW);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(sadd3(-push, armB), 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(-sadd3(armA, armB), 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, slopeW);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// --- Centred damping (jbeam-semantics-spec.md §39; the CPU reference's
// ExplicitBeams::centred_correction). The beam phase keeps each JBeam record's
// damping slope in its first incidence entry's .w (0: none). gBeams at
// gP.centredDv + node: the node phase's Δv (centredDelta, in the node loop: the
// velocity before it kept in a register). After the node phase each record writes its
// correction −½·slope·Δu into its incidence entries (centredBeam), and each node
// adds their sum as a velocity and position step (centredNode).
// Node gi's Δv after its node phase, from its velocity `v0` before it.
void centredDelta(uint gi, float3 v0)
{
    gBeams[gP.centredDv + gi] = float4(ssub3(gNodes[gi * 3u + 1u].xyz, v0), 0.0);
}

