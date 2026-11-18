// --- A group's levels with each hit worked out once (PERF_LEVELS_ONCE); included at the
// end of explicit_beams_fleet.slang, after explicit_beams_fleet_hitlat.slang;
// engine/docs/measurements/contact-groups-design.md §8.11 ---
// A level is a chain: the hits' application, then three residual phases, each as long as
// its slowest hit, one lane's dependent loads (vehicle-perf-first-principles.md §8.32,
// §8.37). fleetLevelsLat already keeps a hit's residual staging from phase 0 to phase 2;
// each level still found its hit again in every phase (the owner, the hit list, the
// triangle and the node: four loads in a row), worked out the application's triangle,
// clip test or vote, reduced mass and force, and the residual phases' staging, each after
// the phase before. Nothing of that changes within the levels: positions, velocities, the
// snapshot, the two-body counts and the held clip sides are written before the snapshot
// and read only here; the levels write the accumulators, their residual stages, the
// position steps and the contact-force log, integer sums all.
// fleetLevelsOnce is fleetLevelsLat with the same rule, rounds, order and leaf
// functions, and these changes of when:
//   - a lane's hit (the group's flattened hit `lane`; a list past the lanes: fleetLevelsLat)
//     found once, after the snapshot, and kept in the lane's registers: its owner, list
//     entry, triangle, node and level;
//   - a two-body hit's application worked out there too, in the owner's coordinates as
//     vvTry works it out (lvTryStage: vvTry's expressions without its adds; a two-body hit
//     reads the snapshot, not the accumulators, so it is the same value whenever it is
//     taken), and only its adds made in its level (lvTryAdds). A two-body hit's force
//     reads nothing another hit writes, so each lane adds its own hit, in place of the
//     contact point's chain on one lane (integer adds: any order, the same sums). A
//     pair that is not two-body keeps vvApplyOrdered in its level, as before;
//   - the residual phases' staged hit (vvStagedHit, in world coordinates) taken there too;
//   - phase 1's residual r kept from phase 0 (both read the accumulators A, which phase 0
//     does not write: it adds to B);
//   - when every hit of the group is two-body, a level's merge (phase 2) beside the next
//     level's application: the merge reads and clears B and D and adds D into A, the
//     application of a two-body hit reads none of them and adds into A (integer adds),
//     and the next level's phase 0 is after both. The last level's merge after the loop.
// The same results bit for bit (fast-check.sh perf=4, fleet_contact_gpu).
static const uint PERF_LEVELS_ONCE = 4u;
bool lvOnceOn() { return (gP.perfFlags & PERF_LEVELS_ONCE) != 0u; }

// A two-body hit's application as vvTry makes it, worked out ahead; its adds later.
struct LvApp
{
    bool hit;   // vvTry's hit test passed: the adds
    bool shift; // the position step's adds
    bool log;   // the contact-force log's adds (not dissipative)
    float alpha;
    float beta;
    float gamma;
    float3 f;
    float3 sh;
    float3 s1;
    float3 s2;
    float3 s0;
};

LvApp lvAppNone()
{
    LvApp s;
    s.hit = false;
    s.shift = false;
    s.log = false;
    s.alpha = 0.0;
    s.beta = 0.0;
    s.gamma = 0.0;
    s.f = float3(0.0, 0.0, 0.0);
    s.sh = float3(0.0, 0.0, 0.0);
    s.s1 = float3(0.0, 0.0, 0.0);
    s.s2 = float3(0.0, 0.0, 0.0);
    s.s0 = float3(0.0, 0.0, 0.0);
    return s;
}

// vvTry's values for a two-body pair (the caller has checked vvTwoBody on the triangle's
// first node's vehicle and vh, as vvTry does), its expressions in its order, without its
// adds. (vvTry's contact point force before the two-body branch is not used there.)
LvApp lvTryStage(VvTri t, uint gt, uint h, float3 p, uint vh, float r, uint parity, uint slot)
{
    LvApp s = lvAppNone();
    float alpha;
    float beta;
    float gamma;
    float pen;
    float3 nn;
    if (vvAnti(t, vh))
    {
        uint word;
        if (vvClip(t, gt, h, p, vh, r, alpha, beta, gamma, pen, nn, word) != 1u) return s;
    }
    else
    {
        float3 q = ssub3(p, t.p0);
        alpha = 0.0;
        beta = 0.0;
        float delta = 0.0;
        if (t.solvable)
        {
            alpha = sdot(t.ra, q);
            beta = sdot(t.rb, q);
            delta = sdot(t.rn, q);
        }
        gamma = nf(nf(1.0 - alpha) - beta);
        if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return s;
        nn = t.n;
        if (vvVote(t, h, vh, delta))
        {
            nn = -t.n;
            delta = -delta;
        }
        pen = nf(r - delta);
    }
    float4 P = gNodes[h * 3u];
    float4 V = gNodes[h * 3u + 1u];
    float4 P0 = gNodes[t.tn.x * 3u];
    float4 P1 = gNodes[t.tn.y * 3u];
    float4 P2 = gNodes[t.tn.z * 3u];
    float3 vt = sadd3(sadd3(sscale(gNodes[t.tn.y * 3u + 1u].xyz, alpha), sscale(gNodes[t.tn.z * 3u + 1u].xyz, beta)),
                      sscale(gNodes[t.tn.x * 3u + 1u].xyz, gamma));
    float3 v = ssub3(V.xyz, vt);
    float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), P.w);
    float i1 = (alpha > 0.0 && !vvImmovable(t.tn.y)) ? smul(smul(alpha, alpha), sdiv(vvSplitK(vvSplitTri(t.tn.y)), P1.w)) : 0.0;
    float i2 = (beta > 0.0 && !vvImmovable(t.tn.z)) ? smul(smul(beta, beta), sdiv(vvSplitK(vvSplitTri(t.tn.z)), P2.w)) : 0.0;
    float i0 = (gamma > 0.0 && !vvImmovable(t.tn.x)) ? smul(smul(gamma, gamma), sdiv(vvSplitK(vvSplitTri(t.tn.x)), P0.w)) : 0.0;
    float inv = nf(imh + nf(nf(i1 + i2) + i0));
    float m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
    float3 at = sadd3(sadd3(vvAccel(gBeams[gP.vvSnap + t.tn.y].xyz, t.tn.y, P1.w, alpha),
                            vvAccel(gBeams[gP.vvSnap + t.tn.z].xyz, t.tn.z, P2.w, beta)),
                      vvAccel(gBeams[gP.vvSnap + t.tn.x].xyz, t.tn.x, P0.w, gamma));
    float3 fh = sscale(ssub3(vvAccel(gBeams[gP.vvSnap + h].xyz, h, P.w, 1.0), at), m);
    Surface surf = contactSurfaceV(vh, gStatic[gP.nodeInfoBase + t.tn.x].w);
    float budget;
    float3 f = contactResponse(fh, v, m, nn, surf, pen, V.w, budget);
    s.hit = true;
    s.alpha = alpha;
    s.beta = beta;
    s.gamma = gamma;
    s.f = f;
    s.log = surf.dissipative == 0u;
    if (surf.dissipative != 0u && any(f != float3(0.0, 0.0, 0.0)))
    {
        float delta = shiftDepth(pen, budget, sdot(fh, nn));
        if (delta > 0.0)
        {
            float q = smul(delta, m);
            float im1 = vvImmovable(t.tn.y) ? 0.0 : sdiv(vvSplitK(vvSplitTri(t.tn.y)), P1.w);
            float im2 = vvImmovable(t.tn.z) ? 0.0 : sdiv(vvSplitK(vvSplitTri(t.tn.z)), P2.w);
            float im0 = vvImmovable(t.tn.x) ? 0.0 : sdiv(vvSplitK(vvSplitTri(t.tn.x)), P0.w);
            s.shift = true;
            s.sh = sscale(nn, smul(q, imh));
            s.s1 = sscale(nn, -smul(alpha, smul(q, im1)));
            s.s2 = sscale(nn, -smul(beta, smul(q, im2)));
            s.s0 = sscale(nn, -smul(gamma, smul(q, im0)));
        }
    }
    return s;
}

// vvTry's adds of a hit worked out by lvTryStage.
void lvTryAdds(LvApp s, uint h, uint4 tn)
{
    if (!s.hit) return;
    if (s.shift)
    {
        shiftAdd(h, s.sh);
        if (s.alpha > 0.0) shiftAdd(tn.y, s.s1);
        if (s.beta > 0.0) shiftAdd(tn.z, s.s2);
        if (s.gamma > 0.0) shiftAdd(tn.x, s.s0);
    }
    accAdd(h, s.f);
    accAdd(tn.y, -sscale(s.f, s.alpha));
    accAdd(tn.z, -sscale(s.f, s.beta));
    accAdd(tn.x, -sscale(s.f, s.gamma));
    if (!s.log) return;
    cfAdd(h, s.f);
    cfAdd(tn.y, sscale(s.f, s.alpha));
    cfAdd(tn.z, sscale(s.f, s.beta));
    cfAdd(tn.x, sscale(s.f, s.gamma));
}

// A hit's node as the residual phases read it (vvResAcc): what does not change within
// the levels (immovable, its beams' sum, its mass, its slide slot, the force log's flag),
// taken once; its accumulators read at each use.
struct LvNode
{
    bool still;
    float3 bs;
    float w;
    uint slot1;
    bool cf;
};

LvNode lvNode(uint x, uint vx, uint parity)
{
    LvNode n;
    uint4 info = gStatic[gP.nodeInfoBase + x];
    n.still = !vvImmovable(x);
    n.bs = vvBeamSumOf(x, vx, parity);
    n.w = gNodes[x * 3u].w;
#ifdef EB_AERO
    n.slot1 = info.x >> NODE_SLOT_SHIFT;
#else
    n.slot1 = 0u;
#endif
    n.cf = gP.cfAcc != 0u && (gStatic[info.w * 4u + 3u].x & FLAG_CFORCE) != 0u;
    return n;
}

// vvResAcc of a node taken by lvNode.
float3 lvAcc(LvNode n, uint x, bool withB)
{
    if (!n.still) return float3(0.0, 0.0, 0.0);
    int3 a = int3(gAcc[x * 3u].load(), gAcc[x * 3u + 1u].load(), gAcc[x * 3u + 2u].load());
    if (withB)
    {
        uint at = resAt(x);
        a += int3(gAcc[at + 1u].load(), gAcc[at + 2u].load(), gAcc[at + 3u].load());
    }
    float3 c = sadd3(n.bs, float3(float(a.x), float(a.y), float(a.z)) * (1.0 / FIXED));
#ifdef EB_AERO
    if (n.slot1 != 0u) c = sadd3(c, gBeams[gP.slideAcc + min(n.slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    return sdiv3(c, n.w);
}

// A staged hit's residual phases' values that do not change within the levels: its four
// nodes (h, then the triangle's n1, n2, n0), vvResOf's inverse mass and vvGuardOf's.
struct LvRes
{
    LvNode nh;
    LvNode n1;
    LvNode n2;
    LvNode n0;
    float inv;
    float guardInv;
};

LvRes lvResStage(uint va, uint vh, uint h, uint4 tn, float alpha, float beta, float gamma, uint parity, uint slot)
{
    LvRes s;
    s.nh = lvNode(h, vh, parity);
    s.n1 = lvNode(tn.y, va, parity);
    s.n2 = lvNode(tn.z, va, parity);
    s.n0 = lvNode(tn.x, va, parity);
    // (vvResOf's.)
    float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), gNodes[h * 3u].w);
    float i1 = (alpha > 0.0 && !vvImmovable(tn.y)) ? smul(smul(alpha, alpha), sdiv(vvSplitK(vvSplitTri(tn.y)), gNodes[tn.y * 3u].w)) : 0.0;
    float i2 = (beta > 0.0 && !vvImmovable(tn.z)) ? smul(smul(beta, beta), sdiv(vvSplitK(vvSplitTri(tn.z)), gNodes[tn.z * 3u].w)) : 0.0;
    float i0 = (gamma > 0.0 && !vvImmovable(tn.x)) ? smul(smul(gamma, gamma), sdiv(vvSplitK(vvSplitTri(tn.x)), gNodes[tn.x * 3u].w)) : 0.0;
    s.inv = nf(imh + nf(nf(i1 + i2) + i0));
    // (vvGuardOf's.)
    s.guardInv = nf(ownInvMass(h)
                    + nf(nf(selfTerm(alpha, ownInvMass(tn.y)) + selfTerm(beta, ownInvMass(tn.z))) + selfTerm(gamma, ownInvMass(tn.x))));
    return s;
}

// vvResOf with the staged values.
float lvResOf(LvRes s, uint h, uint4 tn, float alpha, float beta, float gamma, float3 nn, float3 v)
{
    float3 at = sadd3(sadd3(sscale(lvAcc(s.n1, tn.y, false), alpha), sscale(lvAcc(s.n2, tn.z, false), beta)),
                      sscale(lvAcc(s.n0, tn.x, false), gamma));
    float an = sdot(ssub3(lvAcc(s.nh, h, false), at), nn);
    float vn = sdot(v, nn);
    if (!(vn < 0.0 && nf(smul(an, STEP) + vn) < 0.0)) return 0.0;
    if (!(s.inv > 0.0)) return 0.0;
    float m = sdiv(1.0, s.inv);
    float r = nf(-smul(an, m) - sdiv(smul(vn, m), STEP));
    return r > 0.0 ? r : 0.0;
}

// vvGuardOf with the staged values.
bool lvGuardOf(LvRes s, uint h, uint4 tn, float alpha, float beta, float gamma, float3 v, float3 t, out float3 g)
{
    g = float3(0.0, 0.0, 0.0);
    if (all(t == float3(0.0, 0.0, 0.0))) return false;
    float3 a = ssub3(lvAcc(s.nh, h, true),
                     sadd3(sadd3(sscale(lvAcc(s.n1, tn.y, true), alpha), sscale(lvAcc(s.n2, tn.z, true), beta)),
                           sscale(lvAcc(s.n0, tn.x, true), gamma)));
    float inv = s.guardInv;
    float3 rest = ssub3(a, sscale(t, inv));
    float aa = sdot(t, sadd3(sadd3(v, v), sscale(rest, STEP)));
    float bb = smul(smul(STEP, sdot(t, t)), inv);
    if (!(nf(aa + bb) > 0.0 && bb > 0.0)) return false;
    float k = aa < 0.0 ? min(sdiv(-aa, bb), 1.0) : 0.0;
    g = sscale(t, -nf(1.0 - k));
    return true;
}

// resFinal with the force log's flags staged.
void lvResFinal(LvRes s, uint h, uint4 tn, float alpha, float beta, float gamma, float3 f1, float3 total)
{
    uint4 n4 = uint4(h, tn.y, tn.z, tn.x);
    for (uint role = 1u; role <= 4u; role++)
    {
        uint x = role == 1u ? n4.x : (role == 2u ? n4.y : (role == 3u ? n4.z : n4.w));
        bool cf = role == 1u ? s.nh.cf : (role == 2u ? s.n1.cf : (role == 3u ? s.n2.cf : s.n0.cf));
        float3 st = hitShare(total, role, alpha, beta, gamma);
        int3 d = toFixed3(st) - toFixed3(hitShare(f1, role, alpha, beta, gamma));
        uint at = resAt(x);
        if (d.x != 0) gAcc[at + 4u].add(d.x);
        if (d.y != 0) gAcc[at + 5u].add(d.y);
        if (d.z != 0) gAcc[at + 6u].add(d.z);
        if (cf) gAcc[gP.cfAcc + 3u * x].add(toFixed(ssqrt(sdot(st, st))));
    }
}

// fleetLevelsLat with each hit worked out once (see above). Every lane of the group's
// workgroup; a hit list past its room: fleetLevelsAny; more hits than lanes:
// fleetLevelsLat.
void fleetLevelsOnce(uint k, uint off, uint parity, uint lane, uint tag)
{
    uint open = 0u;
    if (lane == 0u)
    {
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            uint own = flOwn(vvMember(off, m));
            open += uint(gAcc[own + FL_OFFERED].load()) + uint(gAcc[own + FL_RESET].load()) + uint(gAcc[own + FL_FIX].load());
        }
    }
    open = splitUniform(lane, open);
    if (open == 0u)
    {
        fleetLevelsAny(k, off, parity, lane, tag);
        return;
    }
    if (lane == 0u) gsCount[0].store(0u);
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
    {
        if (flHits(vvMember(off, m)) != 0u) gsOff[gsCount[0].add(1u)] = m;
    }
    sync();
    uint over = 0u;
    if (lane == 0u)
    {
        uint H = gsCount[0].load();
        uint run = 0u;
        for (uint j = 0u, jStop = EB_END(j, H, 1u); j < jStop; j++)
        {
            uint v = vvMember(off, gsOff[j]);
            uint n = flHits(v);
            if (n > 2u * uint(gAcc[flOwn(v) + FL_OFFERED].load())) over = 1u;
            gsOff[LAT_PRE + j] = run;
            run += n;
        }
        gsOff[LAT_PRE + H] = run;
    }
    over = splitUniform(lane, over);
    if (over != 0u || (gP.ablate & (65536u | 32768u | 16384u | 262144u)) != 0u)
    {
        fleetLevelsAny(k, off, parity, lane, tag);
        return;
    }
    uint H = splitUniform(lane, lane == 0u ? gsCount[0].load() : 0u);
    uint T = splitUniform(lane, gsOff[LAT_PRE + H]);
    if (T > LANES)
    {
        // (A lane with a second hit: fleetLevelsLat, which takes it in each phase.)
        fleetLevelsLat(k, off, parity, lane, tag);
        return;
    }
    gFl = 2u;
    gFlOff = off;
    gVvTag = 0u;
    // The snapshot: a lane a node of a listed hit (fleetLevelsLat's).
    if (gP.vvSplit != 0u)
    {
        for (uint x = lane, xStop = EB_END(x, 4u * T, LANES); x < xStop; x += LANES)
        {
            uint g = x >> 2u;
            uint j = latOwner(H, g);
            uint base = vvHitSmall(0u, 0u, gsOff[j], 0u);
            uint i = g - gsOff[LAT_PRE + j];
            uint4 tn = gStatic[gP.triBase + gNext[base + 2u * i]];
            uint h = gNext[base + 2u * i + 1u];
            uint which = x & 3u;
            if (which == 3u)
                vvSnapNode(h, gStatic[gP.nodeInfoBase + h].w, parity, tag);
            else
                vvSnapNode(which == 0u ? tn.x : (which == 1u ? tn.y : tn.z), tn.w, parity, tag);
        }
        sync();
        gVvTag = tag;
    }
    bool res = gP.vvSplit != 0u && gP.selfSplit != 0u;
    // The lane's hit, found once and worked out once.
    bool mine = lane < T;
    uint a = 0u;
    uint va = 0u;
    uint base = 0u;
    uint i = 0u;
    uint n = 0u;
    uint gt = 0u;
    uint h = 0u;
    uint vh = 0u;
    uint4 tn = uint4(0u, 0u, 0u, 0u);
    float r = 0.0;
    bool two = false;
    LvApp app = lvAppNone();
    LatKept kept;
    LvRes rs;
    kept.staged = false;
    kept.alpha = 0.0;
    kept.beta = 0.0;
    kept.gamma = 0.0;
    kept.nn = float3(0.0, 0.0, 0.0);
    kept.v = float3(0.0, 0.0, 0.0);
    kept.f1 = float3(0.0, 0.0, 0.0);
    if (mine)
    {
        uint j = latOwner(H, lane);
        a = gsOff[j];
        va = vvMember(off, a);
        base = vvHitSmall(0u, 0u, a, 0u);
        i = lane - gsOff[LAT_PRE + j];
        n = flHits(va);
        r = asfloat(gStatic[va * 4u + 3u].z);
        gt = gNext[base + 2u * i];
        h = gNext[base + 2u * i + 1u];
        vh = gStatic[gP.nodeInfoBase + h].w;
        tn = gStatic[gP.triBase + gt];
        two = vvTwoBody(gStatic[gP.nodeInfoBase + tn.x].w, vh);
        // (vvApplyOrdered's coordinates: the owner's.)
        if (two) app = lvTryStage(vvTriangle(gt, vvRel(va, va), r), gt, h, nf3(vvRel(vh, va) + gNodes[h * 3u].xyz), vh, r, parity, a);
        // (The residual phases' coordinates: the World's, as latResidual takes them.)
        if (res)
        {
            VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
            float alpha;
            float beta;
            float gamma;
            float3 nn;
            float3 v;
            float3 f1;
            bool staged = vvStagedHit(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity, a, alpha,
                                      beta, gamma, nn, v, f1);
            kept.staged = staged;
            kept.alpha = alpha;
            kept.beta = beta;
            kept.gamma = gamma;
            kept.nn = nn;
            kept.v = v;
            kept.f1 = f1;
            if (staged) rs = lvResStage(va, vh, h, tn, alpha, beta, gamma, parity, a);
        }
    }
    // Every hit two-body: a level's merge beside the next level's application.
    if (lane == 0u) gsUniformLanes = 0u;
    sync();
    if (mine && !two) gsUniformLanes = 1u;
    sync();
    uint notTwo = gsUniformLanes;
    sync();
    bool fuse = res && notTwo == 0u;
    // The owners' levels: fleetLevelsAny's rule, partners and rounds (fleetLevelsLat's).
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES)
    {
        uint v = vvMember(off, m);
        gNext[flLevelAt(v)] = flHits(v) != 0u ? 1u : 0u;
    }
    for (uint j = lane, jStop = EB_END(j, H, LANES); j < jStop; j += LANES)
    {
        uint m = gsOff[j];
        uint v = vvMember(off, m);
        uint2 p = uint2(0u, 0u);
        uint e1 = gNext[flAdjFirst(v + 1u)];
        for (uint e = gNext[flAdjFirst(v)], eStop = EB_END(e, e1, 1u); e < eStop; e++)
        {
            uint d = gNext[flAdjAt(e)];
            if ((gNext[flBitsAt(d)] & 5u) != 1u) continue;
            uint rk = gMisc[vvRec(flPartner(d), VV_MEMBER)].x;
            if (rk < 32u) p.x |= 1u << rk;
            else if (rk < 64u) p.y |= 1u << (rk - 32u);
        }
        gsPart[4u * j] = p.x;
        gsPart[4u * j + 1u] = p.y;
        gsPart[4u * j + 2u] = p.x | (m < 32u ? 1u << m : 0u);
        gsPart[4u * j + 3u] = p.y | (m >= 32u ? 1u << (m - 32u) : 0u);
    }
    sync();
    for (uint round = 0u; round < EB_MAX_MEMBERS + 2u; round++)
    {
        if (H == 0u) break;
        if (lane == 0u) gsUniformLanes = 0u;
        sync();
        bool changed = false;
        for (uint j = lane, jStop = EB_END(j, H, LANES); j < jStop; j += LANES)
        {
            uint m = gsOff[j];
            uint v = vvMember(off, m);
            uint l = 1u;
            for (uint c = 0u, cStop = EB_END(c, H, 1u); c < cStop; c++)
            {
                uint mc = gsOff[c];
                if (mc >= m) continue;
                if ((gsPart[4u * j] & gsPart[4u * c + 2u]) == 0u && (gsPart[4u * j + 1u] & gsPart[4u * c + 3u]) == 0u) continue;
                l = max(l, gNext[flLevelAt(vvMember(off, mc))] + 1u);
            }
            if (l != gNext[flLevelAt(v)])
            {
                gNext[flLevelAt(v)] = l;
                changed = true;
            }
        }
        if (changed) gsUniformLanes = 1u;
        sync();
        uint go = gsUniformLanes;
        sync();
        if (go == 0u) break;
    }
    uint top = 1u;
    if (lane == 0u)
    {
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) top = max(top, gNext[flLevelAt(vvMember(off, m))]);
    }
    top = splitUniform(lane, top);
    uint lv = mine ? gNext[flLevelAt(va)] : 0u;
    // The phases' values of the lane's hit at its level.
    float rr = 0.0;
    for (uint l = 1u, lGuard = 0u; l <= top; l++)
    {
        EB_PASS_UNIFORM(lGuard, EB_MAX_MEMBERS + 1u);
        if (l == 1u && !flPostOn())
        {
            for (uint ta = 0u, taStop = EB_END(ta, k, 1u); ta < taStop; ta++)
            {
                uint vt = vvMember(off, ta);
                uint own = flOwn(vt);
                uint triOff = gStatic[vt * 4u + 1u].x;
                uint listed = uint(gAcc[own + FL_LISTED].load());
                bool fix = gAcc[own + FL_FIX].load() != 0;
                for (uint li = lane, liStop = EB_END(li, listed, LANES); li < liStop; li += LANES)
                {
                    uint lt = triOff + gNext[vvTestedAt(triOff + li)];
                    vvThrottleEnd(lt);
                    if (fix) gMisc[gP.vvTri + lt].x = 0u;
                }
            }
        }
        // The level before's merge (every hit two-body), beside this level's application.
        if (fuse && mine && kept.staged && lv + 1u == l) resMerge(h, tn.x, tn.y, tn.z);
        if (mine && lv == l)
        {
            if (two) lvTryAdds(app, h, tn);
            else vvApplyOrdered(va, base, n, r, parity, i, n, a);
        }
        sync();
        if (res)
        {
            if (mine && lv == l && kept.staged)
            {
                rr = lvResOf(rs, h, tn, kept.alpha, kept.beta, kept.gamma, kept.nn, kept.v);
                if (rr > 0.0) resStage(h, tn.x, tn.y, tn.z, kept.alpha, kept.beta, kept.gamma, sscale(kept.nn, rr));
            }
            sync();
            if (mine && lv == l && kept.staged)
            {
                float3 total = kept.f1;
                if (rr > 0.0) total = sadd3(total, sscale(kept.nn, rr));
                float3 g;
                if (lvGuardOf(rs, h, tn, kept.alpha, kept.beta, kept.gamma, kept.v, total, g)) total = sadd3(total, g);
                lvResFinal(rs, h, tn, kept.alpha, kept.beta, kept.gamma, kept.f1, total);
            }
            sync();
            if (!fuse)
            {
                if (mine && lv == l && kept.staged) resMerge(h, tn.x, tn.y, tn.z);
                sync();
            }
        }
    }
    // The last level's merge.
    if (fuse)
    {
        if (mine && kept.staged && lv == top) resMerge(h, tn.x, tn.y, tn.z);
        sync();
    }
    // The two-body counts the hits wrote, once all levels are applied.
    if (gP.vvSplit != 0u && mine)
    {
        gAcc[vvSplitTri(tn.x)].store(0);
        gAcc[vvSplitTri(tn.y)].store(0);
        gAcc[vvSplitTri(tn.z)].store(0);
        gAcc[vvSplitPoint(a, h)].store(0);
    }
    gVvTag = 0u;
    gFl = 0u;
    sync();
    for (uint m = lane, mStop = EB_END(m, k, LANES); m < mStop; m += LANES) flOwnerEnd(vvMember(off, m));
    flArgsZero(lane);
}
