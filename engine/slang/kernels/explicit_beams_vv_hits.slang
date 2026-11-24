// --- A hit's nodes' beam sums kept for the pass (PERF_VV_BEAM_SUM, lossless) ---
// A node's force is its beams' sum plus its accumulators. The beams' sum does not change
// within a contact pass (no beam and no position does), yet the pass took it again for
// each of a hit's four nodes at every use: the snapshot, the hit, and several times in
// each residual phase (a node's whole beam list each time: 13 ms a frame for 17 hits
// at eight cars' first collisions). With the flag the levels keep it beside the
// snapshot when they take it (gBeams at vvSnap + totalNodes + node; the snapshot's .w
// the pass's tag, so a node the pass did not take is summed as before), and every later
// use loads it: the same value.
static const uint PERF_VV_BEAM_SUM = 512u;
static uint gVvTag = 0u; // this pass's tag (0: no kept sums)
float3 vvBeamSumOf(uint h, uint vh, uint parity)
{
    if (gVvTag != 0u && asuint(gBeams[gP.vvSnap + h].w) == gVvTag) return gBeams[gP.vvSnap + gP.totalNodes + h].xyz;
    uint4 rb0 = gStatic[vh * 4u];
    uint4 rb2 = gStatic[vh * 4u + 2u];
    return beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], rb0.z, rb2.z, rb2.w, parity, false);
}
// The contact point's force now (beams, accumulator, slide).
float3 vvPointForce(uint h, uint vh, uint parity)
{
    uint4 info = gStatic[gP.nodeInfoBase + h];
    float3 fh = sadd3(vvBeamSumOf(h, vh, parity), accForce(h));
#ifdef EB_AERO
    uint slot1 = info.x >> NODE_SLOT_SHIFT;
    if (slot1 != 0u) fh = sadd3(fh, gBeams[gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    return fh;
}

// The pair has anti-clip (§36): the owner or the partner vehicle vh.
bool vvAnti(VvTri t, uint vh) { return (t.clip & 1u) != 0u || (gStatic[vh * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u; }

// The candidate box test (explicit_contact.rs triangle_loop): x in the triangle's
// box grown by the range; with anti-clip also by the triangle's and the point's
// travel over the step and CLIP_DEPTH.
bool vvInBox(VvTri t, float3 x, uint h, bool anti)
{
    if (!anti) return !(any(x < t.lo) || any(x > t.hi));
    float3 v = gNodes[h * 3u + 1u].xyz;
    float pad = nf(nf(t.travel + smul(ssqrt(sdot(v, v)), STEP)) + CLIP_DEPTH);
    float3 p3 = float3(pad, pad, pad);
    return all(x >= ssub3(t.lo, p3)) && all(x <= sadd3(t.hi, p3));
}

// The side vote (spec §5.3) of h (of vehicle vh) at distance delta: true behind.
bool vvVote(VvTri t, uint h, uint vh, float delta)
{
    // A vote of h and its spawn-time beam neighbours (its incidence entries' far
    // ends, in beam order).
    uint4 info = gStatic[gP.nodeInfoBase + h];
    int score = delta >= 0.0 ? 3 : -3;
    if (info.z > 3u)
    {
        float3 ob = vvRel(vh, gStatic[gP.nodeInfoBase + t.tn.x].w);
        uint base = gP.vvNbr + info.y;
        // Four independent loads at a time (the far ends are a contiguous list).
        uint k = 0u;
        for (uint kGuard = 0u; k + 4u <= info.z; k += 4u)
        {
            EB_PASS(187, kGuard);
            float3 y0 = gNodes[gLists[base + k] * 3u].xyz;
            float3 y1 = gNodes[gLists[base + k + 1u] * 3u].xyz;
            float3 y2 = gNodes[gLists[base + k + 2u] * 3u].xyz;
            float3 y3 = gNodes[gLists[base + k + 3u] * 3u].xyz;
            score += sdot(t.n, ssub3(nf3(ob + y0), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y1), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y2), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y3), t.p0)) >= 0.0 ? 1 : -1;
        }
        uint kStop = EB_END(188, k, info.z, 1u);
        for (; k < kStop; k++)
            score += sdot(t.n, ssub3(nf3(ob + gNodes[gLists[base + k] * 3u].xyz), t.p0)) >= 0.0 ? 1 : -1;
    }
    return score < 0;
}

// Vehicle contact's anti-clip test (§36; explicit_contact.rs triangle_loop with
// clip_test): partner point h (vehicle vh) at p against owner triangle gt. As
// selfClip, the band's side by the vote. 0: no contact; 1: a hit; 2: a contact
// without force (anti-clip only, in front). `word` the side to record.
uint vvClip(VvTri t, uint gt, uint h, float3 p, uint vh, float r, out float alpha, out float beta, out float gamma,
            out float pen, out float3 nn, out uint word)
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
    pen = 0.0;
    nn = t.n;
    word = 0u;
    uint va = t.tn.w;
    uint tl = gt - gStatic[va * 4u + 1u].x;
    float side = clipHeld(h, va, tl);
    bool over = alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0;
    bool have = over;
    if (side == 0.0)
    {
        bool crossed = false;
        float3 oa = vvRel(va, va);
        VvTri pf;
        pf.p0 = ssub3(t.p0, sscale(gNodes[t.tn.x * 3u + 1u].xyz, STEP));
        float3 b1 = ssub3(nf3(oa + gNodes[t.tn.y * 3u].xyz), sscale(gNodes[t.tn.y * 3u + 1u].xyz, STEP));
        float3 b2 = ssub3(nf3(oa + gNodes[t.tn.z * 3u].xyz), sscale(gNodes[t.tn.z * 3u + 1u].xyz, STEP));
        vvFrame(pf, b1, b2);
        float3 q0 = ssub3(ssub3(p, sscale(gNodes[h * 3u + 1u].xyz, STEP)), pf.p0);
        float a0 = 0.0;
        float b0 = 0.0;
        float d0 = 0.0;
        if (pf.solvable)
        {
            a0 = sdot(pf.ra, q0);
            b0 = sdot(pf.rb, q0);
            d0 = sdot(pf.rn, q0);
        }
        if ((d0 > 0.0 && delta < 0.0) || (d0 < 0.0 && delta > 0.0))
        {
            float sc = sdiv(d0, nf(d0 - delta));
            float a = nf(a0 + smul(nf(alpha - a0), sc));
            float b = nf(b0 + smul(nf(beta - b0), sc));
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
            side = vvVote(t, h, vh, delta) ? -1.0 : 1.0;
        }
    }
    float d = side * delta;
    if (!have || !(d <= r && d >= -CLIP_DEPTH)) return 0u;
    word = clipWord(va, tl, side);
    nn = side > 0.0 ? t.n : -t.n;
    bool only = (t.clip & 2u) != 0u;
    if (only && d >= 0.0) return 2u;
    // At most r (explicit_contact.rs clip_depth): held behind, pushed as at the plane.
    pen = min(only ? -d : nf(r - d), r);
    return 1u;
}

// The find phase's hit test (once per tested pair): with anti-clip the contact's
// side is recorded, hit or not.
bool vvFind(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti);
// The hit test alone (a re-walk).
bool vvHitTest(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti);

bool vvTry(VvTri t, uint gt, uint h, float3 p, uint vh, float r, uint parity, uint slot)
{
    float alpha;
    float beta;
    float gamma;
    float pen;
    float3 nn;
    if (vvAnti(t, vh))
    {
        uint word;
        if (vvClip(t, gt, h, p, vh, r, alpha, beta, gamma, pen, nn, word) != 1u) return false;
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
        if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return false;
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
    float mt = nf(nf(smul(P1.w, alpha) + smul(P2.w, beta)) + smul(P0.w, gamma));
    float m = sdiv(smul(P.w, mt), nf(P.w + mt));
    float3 fh = vvPointForce(h, vh, parity);
    if (vvTwoBody(gStatic[gP.nodeInfoBase + t.tn.x].w, vh))
    {
        // Each node one of its k sub-bodies of this turn (mass m/k, force F/k); the
        // triangle point's mass 1/Σ w²/m; only the share m_t/(m_h + m_t) of the
        // contacter's force (at its first hit of the turn) is cancelled.
        // Inverse masses of the split sub-bodies (immovable: 0), the triangle
        // point's Σ w²·k/m, and the pair's relative acceleration from the forces at
        // the pass's start (explicit_contact.rs).
        float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), P.w);
        float i1 = (alpha > 0.0 && !vvImmovable(t.tn.y)) ? smul(smul(alpha, alpha), sdiv(vvSplitK(vvSplitTri(t.tn.y)), P1.w)) : 0.0;
        float i2 = (beta > 0.0 && !vvImmovable(t.tn.z)) ? smul(smul(beta, beta), sdiv(vvSplitK(vvSplitTri(t.tn.z)), P2.w)) : 0.0;
        float i0 = (gamma > 0.0 && !vvImmovable(t.tn.x)) ? smul(smul(gamma, gamma), sdiv(vvSplitK(vvSplitTri(t.tn.x)), P0.w)) : 0.0;
        float inv = nf(imh + nf(nf(i1 + i2) + i0));
        m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
        float3 at = sadd3(sadd3(vvAccel(gBeams[gP.vvSnap + t.tn.y].xyz, t.tn.y, P1.w, alpha),
                                vvAccel(gBeams[gP.vvSnap + t.tn.z].xyz, t.tn.z, P2.w, beta)),
                          vvAccel(gBeams[gP.vvSnap + t.tn.x].xyz, t.tn.x, P0.w, gamma));
        fh = sscale(ssub3(vvAccel(gBeams[gP.vvSnap + h].xyz, h, P.w, 1.0), at), m);
    }
    Surface surf = contactSurfaceV(vh, gStatic[gP.nodeInfoBase + t.tn.x].w);
    float budget;
    float3 f = contactResponse(fh, v, m, nn, surf, pen, V.w, budget);
    if (surf.dissipative != 0u && vvTwoBody(gStatic[gP.nodeInfoBase + t.tn.x].w, vh)
        && any(f != float3(0.0, 0.0, 0.0)))
    {
        // §37: the position step, a position impulse δ·m shared by the sub-bodies'
        // inverse masses (explicit_contact.rs; self contact shares by own masses).
        float delta = shiftDepth(pen, budget, sdot(fh, nn));
        if (delta > 0.0)
        {
            float q = smul(delta, m);
            float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), P.w);
            float im1 = vvImmovable(t.tn.y) ? 0.0 : sdiv(vvSplitK(vvSplitTri(t.tn.y)), P1.w);
            float im2 = vvImmovable(t.tn.z) ? 0.0 : sdiv(vvSplitK(vvSplitTri(t.tn.z)), P2.w);
            float im0 = vvImmovable(t.tn.x) ? 0.0 : sdiv(vvSplitK(vvSplitTri(t.tn.x)), P0.w);
            shiftAdd(h, sscale(nn, smul(q, imh)));
            if (alpha > 0.0) shiftAdd(t.tn.y, sscale(nn, -smul(alpha, smul(q, im1))));
            if (beta > 0.0) shiftAdd(t.tn.z, sscale(nn, -smul(beta, smul(q, im2))));
            if (gamma > 0.0) shiftAdd(t.tn.x, sscale(nn, -smul(gamma, smul(q, im0))));
        }
    }
    accAdd(h, f);
    accAdd(t.tn.y, -sscale(f, alpha));
    accAdd(t.tn.z, -sscale(f, beta));
    accAdd(t.tn.x, -sscale(f, gamma));
    // (§37.1: a staged hit, two-body and dissipative, logs its total at the end.)
    if (surf.dissipative != 0u && vvTwoBody(gStatic[gP.nodeInfoBase + t.tn.x].w, vh)) return true;
    cfAdd(h, f);
    cfAdd(t.tn.y, sscale(f, alpha));
    cfAdd(t.tn.z, sscale(f, beta));
    cfAdd(t.tn.x, sscale(f, gamma));
    return true;
}

// --- §37.1's residual pass and work guard for vehicle contact (explicit_contact.rs
// residual_vehicle_pass, guard_vehicle_pass), over an owner's listed hits after
// they are applied: the same staging as self contact's (A the accumulators, B the
// residual, D the totals' deltas at resAt). A turn whose list overflowed applied
// its hits at once (out of the specification's order, as before) and stages none.

#include "explicit_beams_vv_sums.slang"

// Node x's acceleration from its force now, its beam sum f given (vvSumsOf): vvPointForce's
// beams, A, with B when `withB`; 0 immovable.
float3 vvResAcc(uint x, float3 f, bool withB)
{
    if (vvImmovable(x)) return float3(0.0, 0.0, 0.0);
    uint4 info = gStatic[gP.nodeInfoBase + x];
    int3 a = int3(gAcc[x * 3u].load(), gAcc[x * 3u + 1u].load(), gAcc[x * 3u + 2u].load());
    if (withB)
    {
        uint at = resAt(x);
        a += int3(gAcc[at + 1u].load(), gAcc[at + 2u].load(), gAcc[at + 3u].load());
    }
    float3 c = sadd3(f, float3(float(a.x), float(a.y), float(a.z)) * (1.0 / FIXED));
#ifdef EB_AERO
    uint slot1 = info.x >> NODE_SLOT_SHIFT;
    if (slot1 != 0u) c = sadd3(c, gBeams[gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    return sdiv3(c, gNodes[x * 3u].w);
}

// A listed hit's geometry and first-pass force again (vvTry without its adds and
// position step); false: not a staged hit (or no longer a hit).
bool vvStagedHit(VvTri t, uint gt, uint h, float3 p, uint vh, float r, uint parity, uint slot, out float alpha,
                 out float beta, out float gamma, out float3 nn, out float3 v, out float3 f)
{
    alpha = 0.0;
    beta = 0.0;
    gamma = 0.0;
    nn = float3(0.0, 0.0, 0.0);
    v = float3(0.0, 0.0, 0.0);
    f = float3(0.0, 0.0, 0.0);
    uint va = gStatic[gP.nodeInfoBase + t.tn.x].w;
    if (!vvTwoBody(va, vh)) return false;
    Surface surf = contactSurfaceV(vh, va);
    if (surf.dissipative == 0u) return false;
    float pen;
    if (vvAnti(t, vh))
    {
        uint word;
        if (vvClip(t, gt, h, p, vh, r, alpha, beta, gamma, pen, nn, word) != 1u) return false;
    }
    else
    {
        float3 q = ssub3(p, t.p0);
        float delta = 0.0;
        if (t.solvable)
        {
            alpha = sdot(t.ra, q);
            beta = sdot(t.rb, q);
            delta = sdot(t.rn, q);
        }
        gamma = nf(nf(1.0 - alpha) - beta);
        if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return false;
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
    v = ssub3(V.xyz, vt);
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
    float budget;
    f = contactResponse(fh, v, m, nn, surf, pen, V.w, budget);
    return true;
}

// The staged hit's residual r (0: none): approaching and still approaching at A,
// with the first pass's counts.
float vvResOf(uint h, uint4 tn, VvSums S, float alpha, float beta, float gamma, float3 nn, float3 v, uint slot)
{
    float3 at = sadd3(sadd3(sscale(vvResAcc(tn.y, S.y, false), alpha),
                            sscale(vvResAcc(tn.z, S.z, false), beta)),
                      sscale(vvResAcc(tn.x, S.x, false), gamma));
    float an = sdot(ssub3(vvResAcc(h, S.h, false), at), nn);
    float vn = sdot(v, nn);
    if (!(vn < 0.0 && nf(smul(an, STEP) + vn) < 0.0)) return 0.0;
    float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), gNodes[h * 3u].w);
    float i1 = (alpha > 0.0 && !vvImmovable(tn.y)) ? smul(smul(alpha, alpha), sdiv(vvSplitK(vvSplitTri(tn.y)), gNodes[tn.y * 3u].w)) : 0.0;
    float i2 = (beta > 0.0 && !vvImmovable(tn.z)) ? smul(smul(beta, beta), sdiv(vvSplitK(vvSplitTri(tn.z)), gNodes[tn.z * 3u].w)) : 0.0;
    float i0 = (gamma > 0.0 && !vvImmovable(tn.x)) ? smul(smul(gamma, gamma), sdiv(vvSplitK(vvSplitTri(tn.x)), gNodes[tn.x * 3u].w)) : 0.0;
    float inv = nf(imh + nf(nf(i1 + i2) + i0));
    if (!(inv > 0.0)) return 0.0;
    float m = sdiv(1.0, inv);
    float r = nf(-smul(an, m) - sdiv(smul(vn, m), STEP));
    return r > 0.0 ? r : 0.0;
}

// vvResOf for an edge-contact pair (explicit_beams_vv_edges.slang): the accumulated
// clamp, the change toward the stop at A, never below zero with the first pass's normal
// force f1 · nn (explicit_contact.rs residual_vehicle_pass, `clamp`).
bool edgePairOn(uint a, uint b);
float edgeClampRes(float lam0, float r);
float vvResOfClamp(uint h, uint4 tn, VvSums S, float alpha, float beta, float gamma, float3 nn, float3 v, uint slot,
                   float3 f1)
{
    float3 at = sadd3(sadd3(sscale(vvResAcc(tn.y, S.y, false), alpha),
                            sscale(vvResAcc(tn.z, S.z, false), beta)),
                      sscale(vvResAcc(tn.x, S.x, false), gamma));
    float an = sdot(ssub3(vvResAcc(h, S.h, false), at), nn);
    float vn = sdot(v, nn);
    float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), gNodes[h * 3u].w);
    float i1 = (alpha > 0.0 && !vvImmovable(tn.y)) ? smul(smul(alpha, alpha), sdiv(vvSplitK(vvSplitTri(tn.y)), gNodes[tn.y * 3u].w)) : 0.0;
    float i2 = (beta > 0.0 && !vvImmovable(tn.z)) ? smul(smul(beta, beta), sdiv(vvSplitK(vvSplitTri(tn.z)), gNodes[tn.z * 3u].w)) : 0.0;
    float i0 = (gamma > 0.0 && !vvImmovable(tn.x)) ? smul(smul(gamma, gamma), sdiv(vvSplitK(vvSplitTri(tn.x)), gNodes[tn.x * 3u].w)) : 0.0;
    float inv = nf(imh + nf(nf(i1 + i2) + i0));
    if (!(inv > 0.0)) return 0.0;
    float m = sdiv(1.0, inv);
    float r = nf(-smul(an, m) - sdiv(smul(vn, m), STEP));
    return edgeClampRes(max(sdot(f1, nn), 0.0), r);
}

// The guard of the staged hit's force t at A + B (explicit_contact.rs
// guard_vehicle_pass).
bool vvGuardOf(uint h, uint4 tn, VvSums S, float alpha, float beta, float gamma, float3 v, float3 t, out float3 g)
{
    g = float3(0.0, 0.0, 0.0);
    if (all(t == float3(0.0, 0.0, 0.0))) return false;
    float3 a = ssub3(vvResAcc(h, S.h, true),
                     sadd3(sadd3(sscale(vvResAcc(tn.y, S.y, true), alpha),
                                 sscale(vvResAcc(tn.z, S.z, true), beta)),
                           sscale(vvResAcc(tn.x, S.x, true), gamma)));
    float inv = nf(ownInvMass(h)
                   + nf(nf(selfTerm(alpha, ownInvMass(tn.y)) + selfTerm(beta, ownInvMass(tn.z))) + selfTerm(gamma, ownInvMass(tn.x))));
    float3 rest = ssub3(a, sscale(t, inv));
    float aa = sdot(t, sadd3(sadd3(v, v), sscale(rest, STEP)));
    float bb = smul(smul(STEP, sdot(t, t)), inv);
    if (!(nf(aa + bb) > 0.0 && bb > 0.0)) return false;
    float k = aa < 0.0 ? min(sdiv(-aa, bb), 1.0) : 0.0;
    g = sscale(t, -nf(1.0 - k));
    return true;
}

// One phase of the pass over owner va's n listed hits (two words each from `base`;
// lane `lane` of `lanes`): 0 the residuals into B; 1 the guards and totals into D;
// 2 the merge.
void vvResidualPhase(uint va, uint base, uint n, float r, uint parity, uint lane, uint lanes, uint slot, uint phase)
{
    if (gP.selfSplit == 0u || gP.vvSplit == 0u) return;
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = lane, iStop = EB_END(189, i, n, lanes); i < iStop; i += lanes)
    {
        uint gt = gNext[base + 2u * i];
        uint h = gNext[base + 2u * i + 1u];
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        VvTri t = vvTriangle(gt, oa, r);
        float alpha;
        float beta;
        float gamma;
        float3 nn;
        float3 v;
        float3 f1;
        if (!vvStagedHit(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity, slot, alpha, beta,
                         gamma, nn, v, f1))
            continue;
        if (phase == 2u)
        {
            resMerge(h, t.tn.x, t.tn.y, t.tn.z);
            continue;
        }
        VvSums S = vvSumsOf(va, vh, h, t.tn, parity);
        float rr = edgePairOn(va, vh) ? vvResOfClamp(h, t.tn, S, alpha, beta, gamma, nn, v, slot, f1)
                                      : vvResOf(h, t.tn, S, alpha, beta, gamma, nn, v, slot);
        if (phase == 0u)
        {
            if (rr != 0.0) resStage(h, t.tn.x, t.tn.y, t.tn.z, alpha, beta, gamma, sscale(nn, rr));
            continue;
        }
        float3 total = f1;
        if (rr != 0.0) total = sadd3(total, sscale(nn, rr));
        float3 g;
        if (vvGuardOf(h, t.tn, S, alpha, beta, gamma, v, total, g)) total = sadd3(total, g);
        resFinal(h, t.tn.x, t.tn.y, t.tn.z, alpha, beta, gamma, f1, total);
    }
}

// The owner's tested triangle t (list position p of GRID_SUBS lanes, lane `sub`)
// against the owner's candidates (partner contact points inside the owner's box
// grown by its range, compacted at the run offset): the box test, then the hit
// test, vote and force; marks the triangle's record (z: had candidates, w: hit).
// The hit test of vvTry alone.
bool vvInBand(VvTri t, float3 p, float r)
{
    float3 q = ssub3(p, t.p0);
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
    return alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r;
}

bool vvFind(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti)
{
    if (!anti) return vvInBand(t, p, r);
    float a;
    float b;
    float g;
    float pen;
    float3 nn;
    uint word;
    uint c = vvClip(t, gt, h, p, vh, r, a, b, g, pen, nn, word);
    if (c != 0u) clipRecord(h, word);
    return c == 1u;
}

bool vvHitTest(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti)
{
    if (!anti) return vvInBand(t, p, r);
    float a;
    float b;
    float g;
    float pen;
    float3 nn;
    uint word;
    return vvClip(t, gt, h, p, vh, r, a, b, g, pen, nn, word) == 1u;
}

// A tested triangle's marks (had candidates, hit; spec §8's throttle inputs): two
// plain words per triangle in gNext after the tested list, set by the triangle's
// sub-lanes and read and cleared by vvThrottleEnd. One word per mark, so lanes
// only ever store the same value to a word: a store to one component of a uint4
// record (the marks were .z and .w of the triangle's throttle record) may compile
// to a read-modify-write of the whole record (Metal), and a lane marking only a
// candidate then could undo another lane's hit mark (a rare nondeterministic
// throttle).
uint vvMarkAt(uint gt) { return gP.vvMembers + 5u * gP.vehicles + gP.totalTris + 2u * gt; }

void vvMark(uint gt, bool cand, bool hit)
{
    if (cand) gNext[vvMarkAt(gt)] = 1u;
    if (hit) gNext[vvMarkAt(gt) + 1u] = 1u;
}

// Hit list entry `i` of the group's list of this owner (q its counter parity).
uint vvHitAt(uint runOff, uint i) { return gP.vvHits + 2u * (2u * runOff + i); }

// vvInBox with the candidate's travel over the step given (PERF_VV_STREAM: the value
// vvInBox computes from its velocity, kept where the candidate was gathered).
void vvBigGridBuild(uint va, uint runOff, uint cands, uint lane);
void vvTriangleCandsBigGrid(uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r,
                            uint parity, uint q);
bool vvInBoxTravel(VvTri t, float3 x, float travelH, bool anti)
{
    if (!anti) return !(any(x < t.lo) || any(x > t.hi));
    float pad = nf(nf(t.travel + travelH) + CLIP_DEPTH);
    float3 p3 = float3(pad, pad, pad);
    return all(x >= ssub3(t.lo, p3)) && all(x <= sadd3(t.hi, p3));
}
uint vvSortAt(uint runOff);

void vvTriangleCands(uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r, uint parity,
                     uint q)
{
    VvTri t = vvTriangle(gt, vvRel(va, va), r);
    bool cand = false;
    bool hit = false;
    // (With the grid the sort records hold the sorted candidates: no stream.)
    bool stream = (gP.perfFlags & (PERF_VV_STREAM | PERF_VV_BIG_GRID)) == PERF_VV_STREAM;
    uint info = vvSortAt(runOff);
    for (uint e = sub, eStop = EB_END(190, e, cands, subs); e < eStop; e += subs)
    {
        float4 E = gBeams[gP.vvRun + runOff + e];
        uint h = asuint(E.w);
        uint vh;
        bool anti;
        if (stream)
        {
            // (travel, the vehicle's anti-clip, the vehicle: the gather's.)
            float4 S = gBeams[info + e];
            vh = asuint(S.z);
            anti = (t.clip & 1u) != 0u || S.y != 0.0;
            if (!vvInBoxTravel(t, E.xyz, S.x, anti)) continue;
        }
        else
        {
            vh = gStatic[gP.nodeInfoBase + h].w;
            anti = vvAnti(t, vh);
            if (!vvInBox(t, E.xyz, h, anti)) continue;
        }
        cand = true;
        if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) continue;
        hit = true;
        vvSplitCount(t, h, 0u);
        // The force later, one lane per hit (the vote and the force read many
        // nodes); a full list applies it here.
        uint i = gsVvHits[q].add(1u);
        if (i < 2u * points)
        {
            gNext[vvHitAt(runOff, i)] = gt;
            gNext[vvHitAt(runOff, i) + 1u] = h;
        }
        else
        {
            vvTry(t, gt, h, E.xyz, vh, r, parity, 0u);
        }
    }
    vvMark(gt, cand, hit);
}

// Owner va's n listed hits (triangle, node: two words each from `base`) in the
// specification's order (triangles in list order, then contact points in node
// order). A hit reads its contact point's force and adds to it and to the owner's
// triangle nodes; the owner's nodes are never contact points of its own hits, so
// only the hits on one contact point depend on each other: one lane per contact
// point runs its hits in triangle order (each reads the ones before), the points
// side by side.
// `bk` (the small groups' pass with its hit buckets, vvHitBuckets): each hit's earlier
// and later hits of its point searched among its bucket's (sorted at vvOrderAt + 2 VV_PART
// points, ranges in gsOff) instead of all n; the same answers (any earlier, the smallest
// later), so the same tries in the same order. One function for both, so one copy of
// vvTry where a pass takes either (the compilers inline every call).
void vvApplyOrderedIn(uint va, uint base, uint n, float r, uint parity, uint lane, uint lanes, uint slot, bool bk,
                      uint runOff, uint points)
{
    float3 oa = vvRel(va, va);
    uint sorted = vvOrderAt(runOff, 0u, 0u, 0u) + 2u * VV_PART * points;
    for (uint i = lane, iStop = EB_END(191, i, n, lanes); i < iStop; i += lanes)
    {
        uint gt = gNext[base + 2u * i];
        uint h = gNext[base + 2u * i + 1u];
        bool first = true;
#ifdef EB_ABLATE
        // Measurement only: 1048576 * 2 each hit applied alone, unordered (no scans).
        if (!bk && (gP.ablate & 2097152u) != 0u)
        {
            uint vh2 = gStatic[gP.nodeInfoBase + h].w;
            vvTry(vvTriangle(gt, oa, r), gt, h, nf3(vvRel(vh2, va) + gNodes[h * 3u].xyz), vh2, r, parity, slot);
            continue;
        }
#endif
        uint e0 = 0u;
        uint e1 = n;
        if (bk)
        {
            uint b = vvHitBucket(slot, h);
            e0 = gsOff[b];
            e1 = gsOff[b + 1u];
        }
        for (uint e = e0, eStop = EB_END(192, e, e1, 1u); e < eStop; e++)
        {
            uint j = bk ? gNext[sorted + e] : e;
            if (gNext[base + 2u * j + 1u] == h && gNext[base + 2u * j] < gt)
            {
                first = false;
                break;
            }
        }
        if (!first) continue;
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        for (uint it = 0u, itStop = EB_END(193, it, n, 1u); it < itStop; it++)
        {
            VvTri t = vvTriangle(gt, oa, r);
            vvTry(t, gt, h, nf3(vvRel(vh, va) + gNodes[h * 3u].xyz), vh, r, parity, slot);
            // The point's next hit: the smallest later triangle.
            uint next = 0xFFFFFFFFu;
            for (uint e = e0, eStop = EB_END(194, e, e1, 1u); e < eStop; e++)
            {
                uint j = bk ? gNext[sorted + e] : e;
                uint gj = gNext[base + 2u * j];
                if (gNext[base + 2u * j + 1u] == h && gj > gt && gj < next) next = gj;
            }
            if (next == 0xFFFFFFFFu) break;
            gt = next;
        }
    }
}
void vvApplyOrdered(uint va, uint base, uint n, float r, uint parity, uint lane, uint lanes, uint slot)
{
    vvApplyOrderedIn(va, base, n, r, parity, lane, lanes, slot, false, 0u, 0u);
}

// The listed hits of owner va, in order (vvApplyOrdered).
void vvApplyHits(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    vvApplyOrdered(va, vvHitAt(runOff, 0u), min(hits, 2u * points), r, parity, lane, LANES, 0u);
}


// A tested triangle's throttle after its tests (a hit → 0, no candidate → + 1).
void vvThrottleEnd(uint gt)
{
    uint4 c = gMisc[gP.vvTri + gt];
    uint mark = vvMarkAt(gt);
    bool had = gNext[mark] != 0u;
    bool hit = gNext[mark + 1u] != 0u;
    gNext[mark] = 0u;
    gNext[mark + 1u] = 0u;
    int cwait = asint(c.x);
    if (hit) cwait = 0;
    else if (!had) cwait = min(cwait + 1, throttleCap(gt));
    gMisc[gP.vvTri + gt] = uint4(asuint(cwait), c.y, 0u, 0u);
}

// Triangle gt of owner va: its box grown by the range overlaps the box `lo..hi`
// (the partners' boxes' union), so it may have candidates.
// `extra`: anti-clip's reach past the range (a bound: CLIP_DEPTH, the owner's and
// the partners' fastest travel, and a millimetre for rounding).
bool vvNearBox(uint gt, uint va, float r, float3 blo, float3 bhi, float extra)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 o = asfloat(gMisc[va].xyz);
    float3 p0 = nf3(o + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(o + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(o + gNodes[tn.z * 3u].xyz);
    float3 r3 = float3(r, r, r) + float3(extra, extra, extra);
    return vvOverlap(ssub3(min(min(p0, p1), p2), r3), sadd3(max(max(p0, p1), p2), r3), blo, bhi);
}

// Anti-clip's prefilter reach past the range (vvNearBox, the candidate runs): a
// bound on CLIP_DEPTH plus the travel of the triangle (the owner's fastest node)
// and of the point (`vmaxB` the partners' fastest), plus a millimetre.
float vvClipExtra(uint va, float vmaxB, uint parity)
{
    if (gP.clipMem == 0u) return 0.0;
    return CLIP_DEPTH + (clipVmax(va, parity) + vmaxB) * STEP + 0.001;
}

void vvClearCounters()
{
    for (uint q = 0u; q < 2u; q++)
    {
        gsVvOffered[q].store(0u);
        gsVvListed[q].store(0u);
        gsVvCands[q].store(0u);
        gsVvHits[q].store(0u);
        gsVvReset[q] = 0u;
    }
}

