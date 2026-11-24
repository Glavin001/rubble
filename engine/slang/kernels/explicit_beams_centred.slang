// --- The objective form (jbeam-semantics-spec.md §39.6, FLAG_CENTRED_OBJECTIVE; the
// CPU reference's centred_correction with x0). Its own functions, so the
// end-direction form's code is as it was.
// Node gi's position and velocity before its node phase (the CPU's x0, v0), after
// the Δv region.
void centredKeep(uint gi)
{
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    gBeams[gP.centredDv + gP.totalNodes + gi] = float4(P.xyz, 0.0);
    gBeams[gP.centredDv + 2u * gP.totalNodes + gi] = float4(V.xyz, 0.0);
}

// The dihedral's gradient at four positions (centredBeam's, explicit_beams.rs
// dihedral_jacobian).
void centredGrad(float3 x1, float3 x2, float3 x3, float3 x4, out float3 g1, out float3 g2, out float3 g3, out float3 g4)
{
    g1 = float3(0.0, 0.0, 0.0);
    g2 = g1;
    g3 = g1;
    g4 = g1;
    float3 b1 = ssub3(x2, x1);
    float3 b2 = ssub3(x3, x2);
    float3 b3 = ssub3(x4, x3);
    float3 n1 = scross(b1, b2);
    float3 n2 = scross(b2, b3);
    float l1 = sdot(n1, n1);
    float l2 = sdot(n2, n2);
    float ll = sdot(b2, b2);
    if (!(l1 < 1e-12 || l2 < 1e-12 || ll < 1e-12))
    {
        float lb = ssqrt(ll);
        g1 = sscale(n1, sdiv(-lb, l1));
        g4 = sscale(n2, sdiv(lb, l2));
        float p = sdiv(-sdot(b1, b2), ll);
        float q = sdiv(-sdot(b3, b2), ll);
        g2 = ssub3(sscale(g1, nf(p - 1.0)), sscale(g4, q));
        g3 = ssub3(sscale(g4, nf(q - 1.0)), sscale(g1, p));
    }
}

// centredBeam for a vehicle with the objective form: a plain beam's Δu is
// (v′_a − v′_b)·d̂′ − (v_a − v_b)·d̂, a torsion bar's Δφ̇ is g′·v′ − g·v with each
// gradient at its own positions; a coupler's correction is on the relative velocity
// itself, as before.
void centredBeamObjective(uint gj, uint4 bn, uint kind, uint sa, uint sb, float slope)
{
    uint xo = gP.centredDv + gP.totalNodes;
    uint vo = xo + gP.totalNodes;
    float3 fa = float3(0.0, 0.0, 0.0);
    float3 fb = fa;
    if (kind == KIND_JTORSION)
    {
        uint4 bn2 = gStatic[gP.beamBase + (gj + 1u) * 3u];
        float3 f3 = fa;
        float3 f4 = fa;
        if (slope != 0.0)
        {
            float3 g1, g2, g3, g4, h1, h2, h3, h4;
            centredGrad(gNodes[bn.x * 3u].xyz, gNodes[bn.y * 3u].xyz, gNodes[bn2.x * 3u].xyz, gNodes[bn2.y * 3u].xyz, g1, g2, g3, g4);
            centredGrad(gBeams[xo + bn.x].xyz, gBeams[xo + bn.y].xyz, gBeams[xo + bn2.x].xyz, gBeams[xo + bn2.y].xyz, h1, h2, h3, h4);
            float t1 = nf(sdot(g1, gNodes[bn.x * 3u + 1u].xyz) - sdot(h1, gBeams[vo + bn.x].xyz));
            float t2 = nf(sdot(g2, gNodes[bn.y * 3u + 1u].xyz) - sdot(h2, gBeams[vo + bn.y].xyz));
            float t3 = nf(sdot(g3, gNodes[bn2.x * 3u + 1u].xyz) - sdot(h3, gBeams[vo + bn2.x].xyz));
            float t4 = nf(sdot(g4, gNodes[bn2.y * 3u + 1u].xyz) - sdot(h4, gBeams[vo + bn2.y].xyz));
            float dr = nf(nf(nf(t1 + t2) + t3) + t4);
            float t = smul(smul(-0.5, slope), dr);
            fa = sscale(g1, t);
            fb = sscale(g2, t);
            f3 = sscale(g3, t);
            f4 = sscale(g4, t);
        }
        gBeams[sa] = float4(fa, 0.0);
        gBeams[sb] = float4(fb, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(f3, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(f4, 0.0);
        return;
    }
    if (slope != 0.0)
    {
        if (kind == KIND_JCOUPLER)
        {
            float3 dva = gBeams[gP.centredDv + bn.x].xyz;
            float3 dvb = gBeams[gP.centredDv + bn.y].xyz;
            fa = sscale(ssub3(dva, dvb), smul(-0.5, slope));
        }
        else
        {
            float3 d = ssub3(gNodes[bn.x * 3u].xyz, gNodes[bn.y * 3u].xyz);
            float inv = sExactInvSqrt(sdot(d, d));
            float3 d0 = ssub3(gBeams[xo + bn.x].xyz, gBeams[xo + bn.y].xyz);
            float inv0 = sExactInvSqrt(sdot(d0, d0));
            float u1 = smul(sdot(ssub3(gNodes[bn.x * 3u + 1u].xyz, gNodes[bn.y * 3u + 1u].xyz), d), inv);
            float u0 = smul(sdot(ssub3(gBeams[vo + bn.x].xyz, gBeams[vo + bn.y].xyz), d0), inv0);
            float du = nf(u1 - u0);
            fa = sscale(d, smul(smul(smul(-0.5, slope), du), inv));
        }
        fb = -fa;
    }
    gBeams[sa] = float4(fa, 0.0);
    gBeams[sb] = float4(fb, 0.0);
}

// Record gj's correction into its entries (a torsion bar's first record also the
// second's; the second writes nothing). No slope: zero. `objective`: its vehicle
// has FLAG_CENTRED_OBJECTIVE.
void centredBeam(uint gj, bool objective)
{
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    uint kind = bn.z & 0xFFu;
    if (kind == KIND_JTORSION2) return;
    uint sa = gP.incBase + gLists[gP.slotBase + gj * 2u];
    uint sb = gP.incBase + gLists[gP.slotBase + gj * 2u + 1u];
    float slope = gBeams[sa].w;
    if (objective)
    {
        centredBeamObjective(gj, bn, kind, sa, sb, slope);
        return;
    }
    float3 fa = float3(0.0, 0.0, 0.0);
    float3 fb = fa;
    if (kind == KIND_JTORSION)
    {
        uint4 bn2 = gStatic[gP.beamBase + (gj + 1u) * 3u];
        float3 f3 = fa;
        float3 f4 = fa;
        if (slope != 0.0)
        {
            float3 x1 = gNodes[bn.x * 3u].xyz;
            float3 x2 = gNodes[bn.y * 3u].xyz;
            float3 x3 = gNodes[bn2.x * 3u].xyz;
            float3 x4 = gNodes[bn2.y * 3u].xyz;
            float3 g1 = float3(0.0, 0.0, 0.0);
            float3 g2 = g1;
            float3 g3 = g1;
            float3 g4 = g1;
            float3 b1 = ssub3(x2, x1);
            float3 b2 = ssub3(x3, x2);
            float3 b3 = ssub3(x4, x3);
            float3 n1 = scross(b1, b2);
            float3 n2 = scross(b2, b3);
            float l1 = sdot(n1, n1);
            float l2 = sdot(n2, n2);
            float ll = sdot(b2, b2);
            if (!(l1 < 1e-12 || l2 < 1e-12 || ll < 1e-12))
            {
                float lb = ssqrt(ll);
                g1 = sscale(n1, sdiv(-lb, l1));
                g4 = sscale(n2, sdiv(lb, l2));
                float p = sdiv(-sdot(b1, b2), ll);
                float q = sdiv(-sdot(b3, b2), ll);
                g2 = ssub3(sscale(g1, nf(p - 1.0)), sscale(g4, q));
                g3 = ssub3(sscale(g4, nf(q - 1.0)), sscale(g1, p));
            }
            float3 d1 = gBeams[gP.centredDv + bn.x].xyz;
            float3 d2 = gBeams[gP.centredDv + bn.y].xyz;
            float3 d3 = gBeams[gP.centredDv + bn2.x].xyz;
            float3 d4 = gBeams[gP.centredDv + bn2.y].xyz;
            float dr = nf(nf(nf(sdot(g1, d1) + sdot(g2, d2)) + sdot(g3, d3)) + sdot(g4, d4));
            float t = smul(smul(-0.5, slope), dr);
            fa = sscale(g1, t);
            fb = sscale(g2, t);
            f3 = sscale(g3, t);
            f4 = sscale(g4, t);
        }
        gBeams[sa] = float4(fa, 0.0);
        gBeams[sb] = float4(fb, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(f3, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(f4, 0.0);
        return;
    }
    if (slope != 0.0)
    {
        float3 dva = gBeams[gP.centredDv + bn.x].xyz;
        float3 dvb = gBeams[gP.centredDv + bn.y].xyz;
        if (kind == KIND_JCOUPLER)
        {
            fa = sscale(ssub3(dva, dvb), smul(-0.5, slope));
        }
        else
        {
            float3 d = ssub3(gNodes[bn.x * 3u].xyz, gNodes[bn.y * 3u].xyz);
            float inv = sExactLength(sdot(d, d)).y;
            float du = smul(sdot(ssub3(dva, dvb), d), inv);
            fa = sscale(d, smul(smul(smul(-0.5, slope), du), inv));
        }
        fb = -fa;
    }
    gBeams[sa] = float4(fa, 0.0);
    gBeams[sb] = float4(fb, 0.0);
}

// Node gi's correction: its entries' sum (the beam phase's group drops, as the node
// phase's), v += h·c/m, x += h·(h·c/m).
void centredNode(uint gi, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    // Retired debris: left at rest (nodeRetire).
    if (gP.retireBase != 0u && nodeRetired(gi)) return;
    uint4 info = gStatic[gP.nodeInfoBase + gi];
    float3 c = beamSum(float3(0.0, 0.0, 0.0), info, beamOff, groupOff, groupCount, parity, false);
    if ((info.x & NODE_IMMOVABLE) != 0u) return;
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    float3 dv = sscale(sdiv3(c, P.w), STEP);
    // Both new records before either store (Slang's CUDA output may defer a load
    // to its use).
    float4 nV = float4(sadd3(V.xyz, dv), V.w);
    float4 nP = float4(sadd3(P.xyz, sscale(dv, STEP)), P.w);
    gNodes[gi * 3u + 1u] = nV;
    gNodes[gi * 3u] = nP;
}

// The correction after a vehicle's node phase (gP.centredDv nonzero, so the barriers
// are uniform): `lanes` lanes from `lane` over its nodes and records; a vehicle
// without FLAG_CENTRED only meets the barriers.
void centredPhase(uint flagsV, uint nodeOff, uint nodeCount, uint beamOff, uint beamCount, uint groupOff,
                  uint groupCount, uint step, uint lane, uint lanes)
{
    bool on = (flagsV & FLAG_CENTRED) != 0u;
    sync();
    if (on)
    {
        bool objective = (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u;
        for (uint j = lane, jStop = EB_END(8, j, beamCount, lanes); j < jStop; j += lanes) centredBeam(beamOff + j, objective);
    }
    sync();
    if (on)
    {
        uint parity = (step + 1u) & 1u;
        for (uint i = lane, iStop = EB_END(9, i, nodeCount, lanes); i < iStop; i += lanes)
            centredNode(nodeOff + i, beamOff, groupOff, groupCount, parity);
    }
    // No barrier here: what follows reads a node only after its own barrier, or
    // (memberBox) its lane's own nodes.
}

// A damper-only beam with a tear ratio (rubble_damper_tear: k = 0, its long bound
// the ratio) stretched past it: torn this step, with no force (beam_force's Plain arm). True: the beam's records are written. Its own function
// behind beamUpdate's early return, so every other beam's path is as it was.
bool damperTorn(uint gj, bool fast, uint4 bn, float4 s1)
{
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    float4 c2 = asfloat(gStatic[gP.beamBase + gj * 3u + 2u]);
    if (!(c1.x == 0.0 && c2.y > 0.0)) return false;
    float rest = gBeams[gj * 2u].x;
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float2 li = fast ? float2(0.0, sFastInvSqrt(q)) : sExactLength(q);
    if (fast) li.x = smul(q, li.y);
    float e = nf(li.x - rest);
    if (!(e > smul(rest, c2.y))) return false;
    // Torn: no force this step (nothing joins its ends any more).
    gBeams[gj * 2u + 1u] = float4(s1.x, 0.0, 1.0, 1.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
    return true;
}

void beamUpdate(uint gj, uint local, bool fast, uint groupOff, uint parity)
{
    cfFire(gj, groupOff, parity);
    float4 s1 = gBeams[gj * 2u + 1u];
    if (s1.z != 0.0)
    {
        // Broken in the last beam phase: no longer this step's break.
        if (s1.w != 0.0)
        {
            gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
            if (s1.w == 1.0) extPending(gj, local, groupOff, parity);
        }
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        // A broken LBEAM's vertex record (it has no update of its own).
        if ((gStatic[gP.beamBase + gj * 3u].z & 0xFFu) == KIND_JLBEAM)
            gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    if ((bn.z & 0xFFu) >= KIND_JPLAIN)
    {
        if ((bn.z & (TYPE_EXT << 8)) != 0u)
        {
            jbeamExtUpdate(gj, local, groupOff, parity, bn, s1);
            return;
        }
        jbeamUpdate(gj, local, groupOff, parity, bn, s1);
        return;
    }
    if ((bn.z & 0xFFu) == KIND_PLAIN && damperTorn(gj, fast, bn, s1)) return;
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
    float2 li = fast ? float2(0.0, sFastInvSqrt(q)) : sExactLength(q);
    if (fast) li.x = smul(q, li.y);
    float inv = li.y;
    float len = li.x;
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
    }
    // State records only when they change (yield, break): most steps change
    // neither. The last force (.y) is kept for a read-back only, so it is stored
    // with the record, not every step.
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

