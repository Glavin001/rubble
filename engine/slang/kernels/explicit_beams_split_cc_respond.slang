// Continuous contact's response (explicit_beams_split_cc.slang's pass, dispatch C): the speculative
// inelastic contact, the CCD impulse rounds and the impact zones (dat.rs, ccd.rs).

// --- The response (dat.rs dat_truncate after the listing; ccd.rs resolve) ---

static const uint CC_F_NEAR = 0x200u;
static const uint CC_F_CCD = 0x400u;
static const uint CC_F_ACT = 0x800u;
static const uint CC_F_ZONED = 0x1000u;
uint ccFlags(uint v, uint p) { return gNext[ccPairAt(v, p) + 4u]; }
void ccFlagSet(uint v, uint p, uint f) { gNext[ccPairAt(v, p) + 4u] = ccFlags(v, p) | f; }
void ccFlagClear(uint v, uint p, uint f) { gNext[ccPairAt(v, p) + 4u] = ccFlags(v, p) & ~f; }
uint ccPairs(uint v) { return min(asuint(gAcc[ccCount(v)].load()), ccVeh(v, 0u).w); }
// The compact lists of the speculative pass's pairs (list 0) and the CCD's (list 1): pair
// indices after the vehicle's pairs, their counts in its words 9 and 10.
uint ccListAt(uint v, uint which) { return ccVeh(v, 0u).z + CC_PAIR_WORDS * ccVeh(v, 0u).w + which * ccVeh(v, 0u).w; }
uint ccListN(uint v, uint which) { return min(asuint(gAcc[ccVW(v, 9u + which)].load()), ccVeh(v, 0u).w); }
uint ccListP(uint v, uint which, uint i) { return gNext[ccListAt(v, which) + i]; }
void ccListAdd(uint v, uint which, uint p) { gNext[ccListAt(v, which) + asuint(gAcc[ccVW(v, 9u + which)].add(1))] = p; }

// How far node gi can close this step (its step, or its speed times h: reach).
float ccReach(uint gi, uint v, bool speed)
{
    float d = ccLen(ssub3(ccX1(gi), ccX0(v, gi)));
    return speed ? max(d, smul(ccLen(ccVel(gi)), STEP)) : max(d, 0.0);
}
// A side's largest reach (side: the first nn nodes when first).
float ccSide(CcPair q, bool first, bool speed)
{
    float m = 0.0;
    for (uint j = 0u; j < 4u; j++)
        if ((j < q.nn) == first) m = max(m, ccReach(q.k[j], j < q.nn ? q.vb : q.va, speed));
    return m;
}

// The pair's closest points at path time t (ccd.rs closest): signed weights and gap vector.
float3 ccAt(CcPair q, float t, out float4 w, out float3 dp[4])
{
    float3 p[4];
    for (uint j = 0u; j < 4u; j++)
    {
        float3 a = ccX0(j < q.nn ? q.vb : q.va, q.k[j]);
        dp[j] = ssub3(ccX1(q.k[j]), a);
        p[j] = sadd3(a, sscale(dp[j], t));
    }
    return ccClosest(q.nn, p, w);
}
// collide: the time of impact of pair q over its paths (−1: none, or not closing there).
#ifdef EB_CC_VEL
float ccCollideAt(CcPair q, float xi);
float ccCollide(CcPair q, float xi)
{
    // Velocity contact: a pair that starts inside its gap and parts there can still turn and
    // pass through later in the step; an impact when it comes within half its start distance
    // (a second try, one call site).
    float3 a[4];
    for (uint j = 0u; j < 4u; j++) a[j] = ccX0(j < q.nn ? q.vb : q.va, q.k[j]);
    float4 w;
    float d0 = ccLen(ccClosest(q.nn, a, w));
    float t = -1.0;
    for (uint pass = 0u; pass < 2u; pass++)
    {
        t = ccCollideAt(q, pass == 0u ? xi : smul(0.5, d0));
        if (t >= 0.0 || !(d0 <= xi && d0 > 0.0)) break;
    }
    return t;
}
float ccCollideAt(CcPair q, float xi)
#else
float ccCollide(CcPair q, float xi)
#endif
{
    float3 a[4];
    float3 dp[4];
    for (uint j = 0u; j < 4u; j++)
    {
        a[j] = ccX0(j < q.nn ? q.vb : q.va, q.k[j]);
        dp[j] = ssub3(ccX1(q.k[j]), a[j]);
    }
    float t = ccAccd(q.nn, a, dp, xi);
    if (t < 0.0) return -1.0;
    float4 w;
    float3 d2[4];
    float3 n = contactNormOrZero(ccAt(q, t, w, d2));
    float c = 0.0;
    for (uint j = 0u; j < 4u; j++) c = nf(c + smul(w[j], sdot(dp[j], n)));
    return c < 0.0 ? t : -1.0;
}
float ccXi(CcRec r) { return max(smul(0.5, r.d), CC_MIN_GAP); }
// Velocity contact's gap: the whole thickness (the CPU's ccd_pairs share 1).
float ccXiV(CcRec r) { return max(r.d, CC_MIN_GAP); }
float ccXiOf(uint v, CcRec r) { return ccVelocity(v) ? ccXiV(r) : ccXi(r); }

// Applies every group node's sums (velocity; position too with `pos`) and clears them.
// The nodes a round's pairs touched (their count went from 0 to 1), a list a vehicle after
// its two pair lists (its word 11 the count): only those have sums to apply.
uint ccTouchedAt(uint v) { return ccListAt(v, 2u); }
void ccCountNode(uint gi)
{
    if (gAcc[ccA(gi, 6u)].add(1) == 0)
    {
        uint v = asuint(gAcc[ccA(gi, 8u)].load()) >> 1;
        gNext[ccTouchedAt(v) + asuint(gAcc[ccVW(v, 11u)].add(1))] = gi;
    }
}
void ccApplyAll(uint off, uint k, bool pos, float stamp, uint lane)
{
    for (uint m = 0u, mStop = EB_END(979, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        uint n = asuint(gAcc[ccVW(v, 11u)].load());
        for (uint i = lane, iStop = EB_END(899, i, n, LANES); i < iStop; i += LANES) ccApply(gNext[ccTouchedAt(v) + i], pos, stamp);
    }
    sync();
    if (lane < k) gAcc[ccVW(vvMember(off, lane), 11u)].store(0);
    sync();
}

// The group's round flag (its first member's word 12; a counter would be shared by every
// group's workgroup).
uint ccG(uint off) { return ccVW(vvMember(off, 0u), 12u); }

// Positions only change in the CCD's and the zones' rounds; a node's scratch slot 3 .w is the
// round (counted on through the zones' rounds) whose pass moved it last, 0 none this step. A
// pair none of whose nodes the last round moved has the same time of impact as then: its
// cached time (its word 15) reused, the same result without the ACCD.
bool ccMoved(CcPair q, float round)
{
    bool m = false;
    for (uint j = 0u; j < 4u; j++) m = m || gBeams[ccB(q.k[j], 3u)].w == round;
    return m;
}
float ccCollideCached(uint v, uint p, CcPair q, float round)
{
    if (!ccMoved(q, round)) return asfloat(gNext[ccPairAt(v, p) + 15u]);
    float t = ccCollide(q, ccXiOf(v, ccRecAt(v, p)));
    gNext[ccPairAt(v, p) + 15u] = asuint(t);
    return t;
}

// The zone label's root: the smallest node of the component (label propagation converged).
int ccLabel(uint gi) { return gAcc[ccA(gi, 7u)].load(); }
// (The converged labels copied into the scratch's slot 2 .w for the zones' sums.)
int ccZoneOf(uint gi) { return asint(gBeams[ccB(gi, 2u)].w); }

// One impact zone, rooted at node `root`, moved rigidly (ccd.rs rigid_zone): its nodes in
// ascending order over the group's; false when it would turn more than 1 rad.
// The zoned nodes of vehicle v, ascending (the touched list's second half; word 11 the count).
uint ccZonedN(uint v) { return asuint(gAcc[ccVW(v, 11u)].load()); }
uint ccZoned(uint v, uint j) { return gNext[ccTouchedAt(v) + gStatic[v * 4u].y + j]; }
bool ccRigidZone(uint off, uint k, int root, float stamp)
{
    bool pinned = false;
    float mass = 0.0;
    float3 cm = float3(0.0, 0.0, 0.0);
    float3 p = float3(0.0, 0.0, 0.0);
    for (uint m = 0u, mStop = EB_END(980, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        for (uint j = 0u, jStop = EB_END(981, j, ccZonedN(v), 1u); j < jStop; j++)
        {
            uint gi = ccZoned(v, j);
            if (ccZoneOf(gi) != root) continue;
            if (ccIm(gi) == 0.0) pinned = true;
            float ms = gNodes[gi * 3u].w;
            mass = nf(mass + ms);
            cm = sadd3(cm, sscale(ccX0(v, gi), ms));
            p = sadd3(p, sscale(ccVel(gi), ms));
        }
    }
    if (pinned)
    {
        for (uint m = 0u, mStop = EB_END(982, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            uint4 r0 = gStatic[v * 4u];
            for (uint j = 0u, jStop = EB_END(983, j, ccZonedN(v), 1u); j < jStop; j++)
            {
                uint gi = ccZoned(v, j);
                if (ccZoneOf(gi) != root) continue;
                gBeams[ccB(gi, 3u)] = float4(0.0, 0.0, 0.0, stamp);
                gBeams[ccB(gi, 1u)].xyz = ccX0(v, gi);
            }
        }
        return true;
    }
    float3 c = float3(sdiv(cm.x, mass), sdiv(cm.y, mass), sdiv(cm.z, mass));
    float3 vc = float3(sdiv(p.x, mass), sdiv(p.y, mass), sdiv(p.z, mass));
    float3 l = float3(0.0, 0.0, 0.0);
    float3 I[3] = { float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0) };
    for (uint m = 0u, mStop = EB_END(984, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        for (uint j = 0u, jStop = EB_END(985, j, ccZonedN(v), 1u); j < jStop; j++)
        {
            uint gi = ccZoned(v, j);
            if (ccZoneOf(gi) != root) continue;
            float3 r = ssub3(ccX0(v, gi), c);
            float mm = gNodes[gi * 3u].w;
            l = sadd3(l, sscale(scross(r, ssub3(ccVel(gi), vc)), mm));
            float rr = sdot(r, r);
            for (uint a = 0u; a < 3u; a++)
            {
                float3 row;
                for (uint b = 0u; b < 3u; b++) row[b] = smul(mm, nf((a == b ? rr : 0.0) - smul(r[a], r[b])));
                I[a] = sadd3(I[a], row);
            }
        }
    }
    float3 n[3];
    float3 w = float3(0.0, 0.0, 0.0);
    if (ccInverse3(I, n))
        w = float3(nf(nf(smul(n[0].x, l.x) + smul(n[0].y, l.y)) + smul(n[0].z, l.z)),
                   nf(nf(smul(n[1].x, l.x) + smul(n[1].y, l.y)) + smul(n[1].z, l.z)),
                   nf(nf(smul(n[2].x, l.x) + smul(n[2].y, l.y)) + smul(n[2].z, l.z)));
    float wl = ccLen(w);
    float ang = smul(wl, STEP);
    if (ang > 1.0) return false;
    float2 sc = ccSinCos(ang);
    float3 axis = wl > 0.0 ? float3(sdiv(w.x, wl), sdiv(w.y, wl), sdiv(w.z, wl)) : float3(0.0, 0.0, 0.0);
    for (uint m = 0u, mStop = EB_END(986, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        for (uint j = 0u, jStop = EB_END(987, j, ccZonedN(v), 1u); j < jStop; j++)
        {
            uint gi = ccZoned(v, j);
            if (ccZoneOf(gi) != root) continue;
            float3 r = ssub3(ccX0(v, gi), c);
            float3 rot = sadd3(sadd3(sscale(r, sc.y), sscale(scross(axis, r), sc.x)), sscale(axis, smul(sdot(axis, r), nf(1.0 - sc.y))));
            gBeams[ccB(gi, 1u)].xyz = sadd3(sadd3(c, sscale(vc, STEP)), rot);
            gBeams[ccB(gi, 3u)] = float4(sadd3(vc, scross(w, r)), stamp);
        }
    }
    return true;
}

// A pair whose four nodes are all in one zone moves rigidly: not tested again.
bool ccAcross(CcPair q)
{
    int l0 = ccLabel(q.k[0]);
    return !(l0 >= 0 && ccLabel(q.k[1]) == l0 && ccLabel(q.k[2]) == l0 && ccLabel(q.k[3]) == l0);
}

// Moreau's impact law over the response (dat.rs restitute, AGENTS.md "Contact never stands in
// for metal"): the response's velocity change d = v - v_before given back theta times,
// theta = e unless that would raise the group's kinetic energy above its value before the pass
// (a quadratic). The guard's sums in CC_STRIPES fixed stripes over the members' nodes in order,
// then the stripes in order: the CPU's order whatever the workgroup's width.
static const uint CC_STRIPES = 64u;
// The most members a group's guard labels (past it the frame fails: the overflow counter).
static const uint CC_GUARD_MEMBERS = 64u;
groupshared float4 gsCcE[CC_STRIPES];
groupshared uint gsCcLab[CC_GUARD_MEMBERS];
// Members a and b near by the listing's own test (ccListWide, a != b).
bool ccNear(uint a, uint b, float rq)
{
    float rr = min(rq, smul(4.0, nf(nf(ccReachV(a) + ccReachV(b)) + ccApart(a, b))));
    float r = nf(nf(rr + ccMaxOffset(a)) + ccMaxOffset(b));
    float3 alo;
    float3 ahi;
    float3 blo;
    float3 bhi;
    ccBox(a, alo, ahi);
    ccBox(b, blo, bhi);
    float3 r3 = float3(r, r, r);
    return !(any(ssub3(alo, r3) > bhi) || any(ssub3(blo, r3) > ahi));
}
// The guard's groups: members joined by ccNear, each labelled by its first member (dat.rs's
// labels: the smallest), then per group the stripes' sums, theta, and the velocities.
void ccRestitute(uint off, uint k, float rq, uint lane)
{
    float e = asfloat(ccVeh(vvMember(off, 0u), 2u).y);
    if (splitUniform(lane, e > 0.0 ? 1u : 0u) == 0u) return;
    if (splitUniform(lane, k > CC_GUARD_MEMBERS ? 1u : 0u) != 0u)
    {
        if (lane == 0u) gAcc[ccC(CC_C_OVERFLOW)].add(1);
        return;
    }
    if (lane == 0u)
    {
        for (uint m = 0u, mStop = EB_END(988, m, k, 1u); m < mStop; m++) gsCcLab[m] = m;
        bool changed = true;
        for (uint it = 0u, itStop = EB_END(1029, it, k, 1u); it < itStop && changed; it++)
        {
            changed = false;
            for (uint a = 0u, aStop = EB_END(989, a, k, 1u); a < aStop; a++)
            {
                for (uint b = a + 1u, bStop = EB_END(990, b, k, 1u); b < bStop; b++)
                {
                    if (!ccNear(vvMember(off, a), vvMember(off, b), rq)) continue;
                    uint lo = min(gsCcLab[a], gsCcLab[b]);
                    if (gsCcLab[a] != lo || gsCcLab[b] != lo)
                    {
                        gsCcLab[a] = lo;
                        gsCcLab[b] = lo;
                        changed = true;
                    }
                }
            }
        }
    }
    sync();
    for (uint c = 0u, cStop = EB_END_UNIFORM(991, c, k, 1u, EB_MAX_MEMBERS); c < cStop; c++)
    {
        if (splitUniform(lane, gsCcLab[c] == c ? 1u : 0u) == 0u) continue;
        if (lane < CC_STRIPES)
        {
            float4 s = float4(0.0, 0.0, 0.0, 0.0);
            for (uint m = 0u, mStop = EB_END(992, m, k, 1u); m < mStop; m++)
            {
                if (gsCcLab[m] != c) continue;
                uint4 r0 = gStatic[vvMember(off, m) * 4u];
                for (uint i = lane, iStop = EB_END(961, i, r0.y, CC_STRIPES); i < iStop; i += CC_STRIPES)
                {
                    uint gi = r0.x + i;
                    float ms = gNodes[gi * 3u].w;
                    float3 vm = gBeams[ccB(gi, 4u)].xyz;
                    float3 p = ccVel(gi);
                    float3 d = ssub3(p, vm);
                    s.x = nf(s.x + smul(ms, sdot(p, d)));
                    s.y = nf(s.y + smul(ms, sdot(d, d)));
                    s.z = nf(s.z + smul(ms, sdot(vm, vm)));
                    s.w = nf(s.w + smul(ms, sdot(p, p)));
                }
            }
            gsCcE[lane] = s;
        }
        sync();
        if (lane == 0u)
        {
            float4 t = float4(0.0, 0.0, 0.0, 0.0);
            for (uint l = 0u; l < CC_STRIPES; l++)
            {
                float4 g = gsCcE[l];
                t = float4(nf(t.x + g.x), nf(t.y + g.y), nf(t.z + g.z), nf(t.w + g.w));
            }
            float theta = 0.0;
            if (t.y > 0.0)
            {
                float disc = max(nf(smul(t.x, t.x) - smul(t.y, nf(t.w - t.z))), 0.0);
                float most = sdiv(nf(-t.x + ssqrtCr(disc)), t.y);
                theta = min(e, max(most, 0.0));
            }
            gsCcE[0] = float4(theta, 0.0, 0.0, 0.0);
        }
        sync();
        float th = gsCcE[0].x;
        for (uint m = 0u, mStop = EB_END(993, m, k, 1u); m < mStop; m++)
        {
            if (gsCcLab[m] != c) continue;
            uint4 r0 = gStatic[vvMember(off, m) * 4u];
            for (uint i = lane, iStop = EB_END(962, i, r0.y, LANES); i < iStop; i += LANES)
            {
                uint gi = r0.x + i;
                float3 p = ccVel(gi);
                gBeams[ccB(gi, 3u)].xyz = sadd3(p, sscale(ssub3(p, gBeams[ccB(gi, 4u)].xyz), th));
            }
        }
        sync();
    }
}

#ifdef EB_CC_VEL
#include "explicit_beams_split_cc_vel.slang"
#endif

void ccRespond(uint off, uint k, float rq, uint lane)
{
    bool vel = splitUniform(lane, ccVelocity(vvMember(off, 0u)) ? 1u : 0u) != 0u;
    // The candidates: the speculative pass's (their gap past the thickness within the sides'
    // reach), with speeds.
    for (uint m = 0u, mStop = EB_END(994, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        for (uint p = lane, pStop = EB_END(900, p, ccPairs(v), LANES); p < pStop; p += LANES)
        {
            CcPair q = ccLoad(v, p);
            CcRec r = ccRecAt(v, p);
            if (nf(r.g - r.d) < nf(ccSide(q, true, true) + ccSide(q, false, true))) ccListAdd(v, 0u, p);
        }
    }
    sync();
#ifndef EB_CC_VEL
    // The speculative inelastic contact (dat.rs speculative), CC_SPEC_ROUNDS rounds.
    uint specRounds = vel ? 0u : CC_SPEC_ROUNDS;
    for (uint round = 0u, roundStop = EB_END_UNIFORM(901, round, specRounds, 1u, CC_SPEC_ROUNDS); round < roundStop; round++)
    {
        if (lane == 0u) gAcc[ccG(off)].store(0);
        sync();
        for (uint m = 0u, mStop = EB_END(995, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(902, i, ccListN(v, 0u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 0u, i);
                CcPair q = ccLoad(v, p);
                CcRec r = ccRecAt(v, p);
                float rel = 0.0;
                for (uint j = 0u; j < 4u; j++) rel = nf(rel + smul(smul(ccSign(q.nn, j), r.w[j]), sdot(ccVel(q.k[j]), r.n)));
                float c = nf(rel + sdiv(max(nf(r.g - r.d), 0.0), STEP));
                if (c < 0.0)
                {
                    ccFlagSet(v, p, CC_F_ACT);
                    gAcc[ccG(off)].store(1);
                    for (uint j = 0u; j < 4u; j++)
                        if (r.w[j] > 0.0 && ccIm(q.k[j]) > 0.0) ccCountNode(q.k[j]);
                }
                else ccFlagClear(v, p, CC_F_ACT);
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccG(off)].load())) == 0u) break;
        for (uint m = 0u, mStop = EB_END(996, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(903, i, ccListN(v, 0u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 0u, i);
                if ((ccFlags(v, p) & CC_F_ACT) == 0u) continue;
                CcPair q = ccLoad(v, p);
                CcRec r = ccRecAt(v, p);
                float rel = 0.0;
                for (uint j = 0u; j < 4u; j++) rel = nf(rel + smul(smul(ccSign(q.nn, j), r.w[j]), sdot(ccVel(q.k[j]), r.n)));
                float c = nf(rel + sdiv(max(nf(r.g - r.d), 0.0), STEP));
                float wsum = 0.0;
                for (uint j = 0u; j < 4u; j++)
                    wsum = nf(wsum + smul(smul(smul(r.w[j], r.w[j]), ccIm(q.k[j])), float(gAcc[ccA(q.k[j], 6u)].load())));
                if (wsum <= 0.0) continue;
                float l = sdiv(-c, wsum);
                for (uint j = 0u; j < 4u; j++)
                {
                    float mi = ccIm(q.k[j]);
                    if (mi > 0.0 && r.w[j] > 0.0) ccAddV(q.k[j], sscale(r.n, smul(smul(smul(ccSign(q.nn, j), r.w[j]), mi), l)));
                }
            }
        }
        sync();
        ccApplyAll(off, k, false, 0.0, lane);
    }
#endif
    // x1 += (v1 − v0) h, then the CCD's velocities: v0 + (x1 − x1 before) / h.
    for (uint m = 0u, mStop = EB_END(997, m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(904, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 v0 = gBeams[ccB(gi, 4u)].xyz;
            float3 x1 = sadd3(ccX1(gi), sscale(ssub3(ccVel(gi), v0), STEP));
            gBeams[ccB(gi, 1u)].xyz = x1;
            float3 d = ssub3(ssub3(x1, gBeams[ccB(gi, 2u)].xyz), float3(0.0, 0.0, 0.0));
            gBeams[ccB(gi, 3u)].xyz = sadd3(v0, float3(sdiv(d.x, STEP), sdiv(d.y, STEP), sdiv(d.z, STEP)));
        }
    }
    sync();
    // (The listing's bound: past half its own pairs' reach from its first node, an error counted.)
    for (uint m = 0u, mStop = EB_END(998, m, k, 1u); m < mStop; m++)
    {
        uint vv = vvMember(off, m);
        uint4 r0 = gStatic[vv * 4u];
        float own = asfloat(gAcc[ccVW(vv, 15u)].load());
        for (uint i = lane, iStop = EB_END(917, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            if (ccOnSurface(vv, gi) && ccLen(ssub3(ssub3(ccX1(gi), ccX0(vv, gi)), ccRefDx(vv))) > smul(0.5, own))
                gAcc[ccC(CC_C_REACH)].add(1);
        }
    }
    sync();
    // The CCD's pairs: their gap past the thickness within twice the sides' steps.
    for (uint m = 0u, mStop = EB_END(999, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        for (uint p = lane, pStop = EB_END(905, p, ccPairs(v), LANES); p < pStop; p += LANES)
        {
            CcPair q = ccLoad(v, p);
            CcRec r = ccRecAt(v, p);
            if (nf(r.g - r.d) < smul(2.0, nf(ccSide(q, true, false) + ccSide(q, false, false)))) ccListAdd(v, 1u, p);
        }
    }
    sync();
    // The impulse rounds (ccd.rs resolve), or velocity contact's (ccd_vel.rs).
#ifdef EB_CC_VEL
    bool clear = vel && ccRespondVel(off, k, lane);
#else
    // (Velocity contact runs in its own module, ccvel: here an error, never a silent pass.)
    if (vel && lane == 0u) gAcc[ccC(CC_C_OVERFLOW)].add(1);
    bool clear = false;
#endif
#ifndef EB_CC_VEL
    uint ccdRounds = vel ? 0u : CC_CCD_ROUNDS;
    for (uint round = 0u, roundStop = EB_END_UNIFORM(906, round, ccdRounds, 1u, CC_CCD_ROUNDS); round < roundStop; round++)
    {
        if (lane == 0u) gAcc[ccG(off)].store(0);
        sync();
        for (uint m = 0u, mStop = EB_END(1000, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(907, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                CcPair q = ccLoad(v, p);
                if (round > 0u && !ccMoved(q, float(round)))
                {
                    // (Unchanged since the last round: its time, and whether it closes there.)
                    float tc = asfloat(gNext[ccPairAt(v, p) + 15u]);
                    if (tc < 0.0) continue;
                    gAcc[ccC(CC_C_HITS)].add(1);
                    gAcc[ccG(off)].store(1);
                    if ((ccFlags(v, p) & CC_F_ACT) == 0u) continue;
                    float4 wc;
                    float3 dc[4];
                    ccAt(q, tc, wc, dc);
                    for (uint j = 0u; j < 4u; j++)
                        if (wc[j] != 0.0 && ccIm(q.k[j]) > 0.0) ccCountNode(q.k[j]);
                    continue;
                }
                ccFlagClear(v, p, CC_F_ACT);
                float t = ccCollide(q, ccXi(ccRecAt(v, p)));
                gNext[ccPairAt(v, p) + 15u] = asuint(t);
                if (t < 0.0) continue;
                gAcc[ccC(CC_C_HITS)].add(1);
                gAcc[ccG(off)].store(1);
                float4 w;
                float3 dp[4];
                float3 n = contactNormOrZero(ccAt(q, t, w, dp));
                float vr = 0.0;
                for (uint j = 0u; j < 4u; j++) vr = nf(vr + smul(w[j], sdot(dp[j], n)));
                vr = sdiv(vr, STEP);
                if (all(n == float3(0.0, 0.0, 0.0)) || !(vr < 0.0)) continue;
                ccFlagSet(v, p, CC_F_ACT);
                for (uint j = 0u; j < 4u; j++)
                    if (w[j] != 0.0 && ccIm(q.k[j]) > 0.0) ccCountNode(q.k[j]);
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccG(off)].load())) == 0u) { clear = true; break; }
        for (uint m = 0u, mStop = EB_END(1001, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(908, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                if ((ccFlags(v, p) & CC_F_ACT) == 0u) continue;
                CcPair q = ccLoad(v, p);
                float t = asfloat(gNext[ccPairAt(v, p) + 15u]);
                float4 w;
                float3 dp[4];
                float3 n = contactNormOrZero(ccAt(q, t, w, dp));
                float vr = 0.0;
                float vv = 0.0;
                for (uint j = 0u; j < 4u; j++)
                {
                    vr = nf(vr + smul(w[j], sdot(dp[j], n)));
                    vv = nf(vv + smul(w[j], sdot(ccVel(q.k[j]), n)));
                }
                vr = sdiv(vr, STEP);
                float wsum = 0.0;
                for (uint j = 0u; j < 4u; j++)
                    wsum = nf(wsum + smul(smul(smul(w[j], w[j]), ccIm(q.k[j])), float(gAcc[ccA(q.k[j], 6u)].load())));
                if (wsum <= 0.0) continue;
                float l = sdiv(-vr, wsum);
                float lv = min(max(sdiv(-vv, wsum), 0.0), l);
                for (uint j = 0u; j < 4u; j++)
                {
                    float mi = ccIm(q.k[j]);
                    if (mi > 0.0 && w[j] != 0.0)
                    {
                        ccAddX(q.k[j], sscale(n, smul(smul(smul(w[j], mi), l), STEP)));
                        ccAddV(q.k[j], sscale(n, smul(smul(w[j], mi), lv)));
                    }
                }
            }
        }
        sync();
        ccApplyAll(off, k, true, float(round + 1u), lane);
    }
#endif
    // Impact zones, while pairs still collide (ccd.rs resolve's zone rounds).
    uint zoneRounds = clear ? 0u : CC_ZONE_ROUNDS;
    for (uint round = 0u, roundStop = EB_END_UNIFORM(909, round, zoneRounds, 1u, CC_ZONE_ROUNDS); round < roundStop; round++)
    {
        if (lane == 0u) gAcc[ccG(off)].store(0);
        sync();
        for (uint m = 0u, mStop = EB_END(1002, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(910, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                CcPair q = ccLoad(v, p);
                if (ccCollideCached(v, p, q, float(CC_CCD_ROUNDS + round)) < 0.0 || !ccAcross(q)) continue;
                ccFlagSet(v, p, CC_F_ZONED);
                gAcc[ccG(off)].store(1);
                for (uint j = 0u; j < 4u; j++)
                    if (ccLabel(q.k[j]) < 0) gAcc[ccA(q.k[j], 7u)].store(int(q.k[j]));
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccG(off)].load())) == 0u) { clear = true; break; }
        // Label propagation over every zoned pair: each node the smallest label of its zone.
        for (uint it = 0u, itStop = EB_END_UNIFORM(911, it, 4096u, 1u, 4096u); it < itStop; it++)
        {
            if (lane == 0u) gAcc[ccG(off)].store(0);
            sync();
            for (uint m = 0u, mStop = EB_END(1003, m, k, 1u); m < mStop; m++)
            {
                uint v = vvMember(off, m);
                for (uint i = lane, iStop = EB_END(912, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
                {
                    uint p = ccListP(v, 1u, i);
                    if ((ccFlags(v, p) & CC_F_ZONED) == 0u) continue;
                    CcPair q = ccLoad(v, p);
                    int lo = min(min(ccLabel(q.k[0]), ccLabel(q.k[1])), min(ccLabel(q.k[2]), ccLabel(q.k[3])));
                    for (uint j = 0u; j < 4u; j++)
                        if (gAcc[ccA(q.k[j], 7u)].min(lo) > lo) gAcc[ccG(off)].store(1);
                }
            }
            sync();
            if (splitUniform(lane, uint(gAcc[ccG(off)].load())) == 0u) break;
        }
        for (uint m = 0u, mStop = EB_END(1004, m, k, 1u); m < mStop; m++)
        {
            uint4 r0 = gStatic[vvMember(off, m) * 4u];
            for (uint i = lane, iStop = EB_END(918, i, r0.y, LANES); i < iStop; i += LANES)
                gBeams[ccB(r0.x + i, 2u)].w = asfloat(ccLabel(r0.x + i));
        }
        sync();
        // Each vehicle's zoned nodes listed, then ranked into ascending order (a node's rank:
        // the zoned nodes below it), so the zones' sums run in the CPU's order.
        for (uint m = 0u, mStop = EB_END_UNIFORM(1005, m, k, 1u, EB_MAX_MEMBERS); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            uint4 r0 = gStatic[v * 4u];
            if (lane == 0u) gAcc[ccVW(v, 11u)].store(0);
            sync();
            for (uint i = lane, iStop = EB_END(919, i, r0.y, LANES); i < iStop; i += LANES)
                if (ccLabel(r0.x + i) >= 0) gNext[ccTouchedAt(v) + asuint(gAcc[ccVW(v, 11u)].add(1))] = r0.x + i;
            sync();
            uint zn = ccZonedN(v);
            for (uint i = lane, iStop = EB_END(920, i, zn, LANES); i < iStop; i += LANES)
            {
                uint gi = gNext[ccTouchedAt(v) + i];
                uint rank = 0u;
                for (uint j = 0u, jStop = EB_END(921, j, zn, 1u); j < jStop; j++) rank += gNext[ccTouchedAt(v) + j] < gi ? 1u : 0u;
                gNext[ccTouchedAt(v) + r0.y + rank] = gi;
            }
            sync();
        }
        // Each zone (rooted at its smallest node) moved rigidly, a lane a zone.
        for (uint m = 0u, mStop = EB_END(1006, m, k, 1u); m < mStop; m++)
        {
            uint4 r0 = gStatic[vvMember(off, m) * 4u];
            for (uint i = lane, iStop = EB_END(913, i, r0.y, LANES); i < iStop; i += LANES)
            {
                uint gi = r0.x + i;
                if (ccZoneOf(gi) != int(gi)) continue;
                gAcc[ccC(CC_C_ZONES)].add(1);
                if (!ccRigidZone(off, k, int(gi), float(CC_CCD_ROUNDS + round + 1u))) gAcc[ccC(CC_C_SPIN)].add(1);
            }
        }
        sync();
    }
    if (!clear)
    {
        for (uint m = 0u, mStop = EB_END(1007, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(914, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                CcPair q = ccLoad(v, p);
                if (ccCollideCached(v, p, q, float(CC_CCD_ROUNDS + CC_ZONE_ROUNDS)) >= 0.0 && ccAcross(q)) gAcc[ccC(CC_C_UNRESOLVED)].add(1);
            }
        }
        sync();
    }
    ccRestitute(off, k, rq, lane);
    // The nodes' new positions and velocities (relative to the origin), where they changed.
    for (uint m = 0u, mStop = EB_END(1008, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        uint4 r0 = gStatic[v * 4u];
        float3 o = ccOrigin(v);
        for (uint i = lane, iStop = EB_END(915, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 x1 = ccX1(gi);
            float3 vs = ccVel(gi);
            if (any(x1 != gBeams[ccB(gi, 2u)].xyz) || any(vs != gBeams[ccB(gi, 4u)].xyz))
            {
                gNodes[gi * 3u + 1u].xyz = vs;
                gNodes[gi * 3u].xyz = ssub3(x1, o);
            }
        }
    }
    sync();
}

