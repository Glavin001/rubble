// --- PERF_HIT_ROLES: the first pass's adds and §37.1's pass a (hit, role) a lane ---
// Vehicle v's hits take it: the switch (with the beam forces' records) and the residual
// pass (two-body, dissipative).
bool hitRolesOn(uint v)
{
    return (gP.perfFlags & (PERF_HIT_ROLES | PERF_HIT_FB)) == (PERF_HIT_ROLES | PERF_HIT_FB) && selfResidual(v);
}

// A node phase of the pass over vehicle v's listed hits, item q = 4 · hit + role (role 0
// the contacter, 1..3 the triangle's n0, n1, n2: hitFbFill's) on lane `sub` of `lanes`:
// 0  the first pass's share into the accumulator A (selfAddOne's add for this node);
// 1  the node's acceleration from A and the inverse mass of its hit count (resAcc,
//    selfInvMass) into the hit's role record;
// 3  the residual's share into B (resStage's add for this node; r from phase 2);
// 4  the node's acceleration from A + B and its own inverse mass (resAccNow, ownInvMass)
//    into the role record;
// 6  the total's share less the first pass's into A and B cleared, the total's
//    magnitude into the contact-force log (the total from phase 5). resFinal adds that
//    difference to D and resMergeNode then moves the node's D into A and clears B and
//    D: nothing reads A or B after phase 4, so the difference goes into A at once (the
//    same integers added, in any order) and D stays zero.
// Each add is the integer selfAddOne, resStage or resFinal adds for that node and the
// sums are order-free; each acceleration is read after every lane's adds of the phase
// before (the caller's barriers), as the hit's lane read it.
void hitRolePhase(uint v, uint sub, uint lanes, uint phase, uint beamOff, uint groupOff, uint groupCount,
                  uint parity)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP || !hitRolesOn(v)) return;
    uint base = hitList(v);
    uint nodeOff = gStatic[v * 4u].x;
    for (uint q = sub, qStop = EB_END(51, q, count * 4u, lanes); q < qStop; q += lanes)
    {
        uint i = q >> 2;
        uint role = q & 3u;
        uint at = base + i * HIT_WORDS;
        uint x = role == 0u ? nodeOff + (gNext[at] & 0xFFFFu) : gNext[at + 7u + role];
        uint rec = hitRoleAt(v, i) + role * 4u;
        if (phase == 1u || phase == 4u)
        {
            float3 acc = resAccNow(x, hitFbAt(v, i) + role * 3u, beamOff, groupOff, groupCount, parity, phase == 4u);
            float im = phase == 1u ? selfInvMass(x) : ownInvMass(x);
            gNext[rec] = asuint(acc.x);
            gNext[rec + 1u] = asuint(acc.y);
            gNext[rec + 2u] = asuint(acc.z);
            gNext[rec + 3u] = asuint(im);
            continue;
        }
        // hitShare's role of this node: the contacter 1, n1 2, n2 3, n0 4.
        uint share = role == 0u ? 1u : (role == 1u ? 4u : role);
        float alpha = asfloat(gNext[at + 1u]);
        float beta = asfloat(gNext[at + 2u]);
        float gamma = nf(nf(1.0 - alpha) - beta);
        float3 f1 = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
        if (phase == 0u)
        {
            accAdd(x, hitShare(f1, share, alpha, beta, gamma));
            continue;
        }
        if (phase == 3u)
        {
            float r = asfloat(gNext[at + 21u]);
            if (r > 0.0)
            {
                float3 nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
                float3 s = hitShare(sscale(nn, r), share, alpha, beta, gamma);
                uint b = resAt(x);
                gAcc[b + 1u].add(toFixed(s.x));
                gAcc[b + 2u].add(toFixed(s.y));
                gAcc[b + 3u].add(toFixed(s.z));
            }
            continue;
        }
        uint tot = hitRoleAt(v, i) + 16u;
        float3 total = asfloat(uint3(gNext[tot], gNext[tot + 1u], gNext[tot + 2u]));
        float3 st = hitShare(total, share, alpha, beta, gamma);
        int3 d = toFixed3(st) - toFixed3(hitShare(f1, share, alpha, beta, gamma));
        uint b = resAt(x);
        gAcc[b + 1u].store(0);
        gAcc[b + 2u].store(0);
        gAcc[b + 3u].store(0);
        if (d.x != 0) gAcc[x * 3u].add(d.x);
        if (d.y != 0) gAcc[x * 3u + 1u].add(d.y);
        if (d.z != 0) gAcc[x * 3u + 2u].add(d.z);
        cfAdd(x, st);
    }
}

// A hit phase of the same pass, hit i on lane `sub` of `lanes`, from its four role
// records (the accelerations and inverse masses the phase before left):
// 2  the residual r into record word 21 (resOf: resDrive's and resForceR's arithmetic);
// 5  the total, first pass with residual and guard, into the hit's role record
//    (resTotal: guardOf's arithmetic).
void hitCombinePhase(uint v, uint sub, uint lanes, uint phase)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP || !hitRolesOn(v)) return;
    uint base = hitList(v);
    for (uint i = sub, iStop = EB_END(52, i, count, lanes); i < iStop; i += lanes)
    {
        uint at = base + i * HIT_WORDS;
        uint rec = hitRoleAt(v, i);
        float alpha = asfloat(gNext[at + 1u]);
        float beta = asfloat(gNext[at + 2u]);
        float gamma = nf(nf(1.0 - alpha) - beta);
        float3 nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
        float3 vv = asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u]));
        // The roles' records: the contacter, n0, n1, n2.
        float3 ah = asfloat(uint3(gNext[rec], gNext[rec + 1u], gNext[rec + 2u]));
        float imh = asfloat(gNext[rec + 3u]);
        float3 a0 = asfloat(uint3(gNext[rec + 4u], gNext[rec + 5u], gNext[rec + 6u]));
        float im0 = asfloat(gNext[rec + 7u]);
        float3 a1 = asfloat(uint3(gNext[rec + 8u], gNext[rec + 9u], gNext[rec + 10u]));
        float im1 = asfloat(gNext[rec + 11u]);
        float3 a2 = asfloat(uint3(gNext[rec + 12u], gNext[rec + 13u], gNext[rec + 14u]));
        float im2 = asfloat(gNext[rec + 15u]);
        // α·a(n1) + β·a(n2) + γ·a(n0), in that order (resDrive, guardOf).
        float3 aw = sscale(a1, alpha);
        aw = sadd3(aw, sscale(a2, beta));
        aw = sadd3(aw, sscale(a0, gamma));
        float imt = nf(nf(selfTerm(alpha, im1) + selfTerm(beta, im2)) + selfTerm(gamma, im0));
        if (phase == 2u)
        {
            float a = sdot(ssub3(ah, aw), nn);
            float vn = sdot(vv, nn);
            float x = nf(vn + max(vn, 0.0));
            float r = 0.0;
            if (vn < 0.0 && nf(smul(a, STEP) + x) < 0.0)
            {
                float inv = nf(imh + imt);
                if (inv > 0.0)
                {
                    float m = sdiv(1.0, inv);
                    float rr = nf(-smul(a, m) - sdiv(smul(x, m), STEP));
                    r = rr > 0.0 ? rr : 0.0;
                }
            }
            gNext[at + 21u] = asuint(r);
            continue;
        }
        float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
        float r = asfloat(gNext[at + 21u]);
        if (r > 0.0) f = sadd3(f, sscale(nn, r));
        if (!all(f == float3(0.0, 0.0, 0.0)))
        {
            float3 a = ssub3(ah, aw);
            float inv = nf(imh + imt);
            float3 rest = ssub3(a, sscale(f, inv));
            float aa = sdot(f, sadd3(sadd3(vv, vv), sscale(rest, STEP)));
            float bb = smul(smul(STEP, sdot(f, f)), inv);
            if (nf(aa + bb) > 0.0 && bb > 0.0)
            {
                float k = aa < 0.0 ? min(sdiv(-aa, bb), 1.0) : 0.0;
                f = sadd3(f, sscale(f, -nf(1.0 - k)));
            }
        }
        gNext[rec + 16u] = asuint(f.x);
        gNext[rec + 17u] = asuint(f.y);
        gNext[rec + 18u] = asuint(f.z);
    }
}

// selfApply's end with PERF_HIT_ROLES, after its rounds' last barrier (every listed hit
// of a vehicle that takes it has its first-pass force and none of its adds; any other
// vehicle's hits are complete and nothing here touches them): the phases above in order,
// a barrier after each. One call site each (the compilers inline every call).
void selfResidualRoles(uint v, bool on, uint sub, uint lanes, uint beamOff, uint groupOff, uint groupCount,
                       uint parity)
{
    for (uint phase = 0u; phase < 7u; phase++)
    {
        if (on)
        {
            if (phase == 2u || phase == 5u) hitCombinePhase(v, sub, lanes, phase);
            else hitRolePhase(v, sub, lanes, phase, beamOff, groupOff, groupCount, parity);
        }
        sync();
    }
}

// selfApply's node forces (perfFlags bit 1), before its rounds: each listed hit's four
// nodes' beam forces (beamSum: the incidence entries and group state of this step,
// which nothing in selfApply writes), one (hit, role) a lane, for the two-body rule's
// round and the residual pass's two phases to read instead of summing them again.
void hitFbFill(uint v, uint sub, uint lanes, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP || !selfTwoBody(v)) return;
    uint base = hitList(v);
    uint nodeOff = gStatic[v * 4u].x;
    for (uint q = sub, qStop = EB_END(53, q, count * 4u, lanes); q < qStop; q += lanes)
    {
        uint at = base + (q >> 2) * HIT_WORDS;
        uint role = q & 3u;
        uint x = role == 0u ? nodeOff + (gNext[at] & 0xFFFFu) : gNext[at + 7u + role];
        float3 f = beamSum(gNodes[x * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + x], beamOff, groupOff, groupCount, parity,
                           false);
        uint fb = hitFbAt(v, q >> 2) + role * 3u;
        gNext[fb] = asuint(f.x);
        gNext[fb + 1u] = asuint(f.y);
        gNext[fb + 2u] = asuint(f.z);
    }
}

// Vehicle v's self-collision hits of this step, after the lanes that found them
// (and resolvePending) passed a barrier: rounds of selfRound until no hit of any
// member is open (every lane of the workgroup runs them: barriers), then each hit's
// shares into the accumulators. `on`: this lane's partition has a vehicle. More
// hits than HIT_CAP: one lane runs the whole step's self-collision in order
// (selfSerial) instead.
void selfSerial(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity);

void selfApply(uint v, bool on, uint sub, uint lanes, uint lane, float3 origin, float r, uint beamOff, uint groupOff,
               uint groupCount, uint parity)
{
    // gsUniformHits: set by any lane that found a hit (tryContact), read after the
    // caller's barrier; none: nothing to apply, no further barrier.
    uint any = gsUniformHits;
    if (any == 0u) return;
    if ((gP.perfFlags & PERF_HIT_FB) != 0u)
    {
        if (on) hitFbFill(v, sub, lanes, beamOff, groupOff, groupCount, parity);
        sync();
    }
    // A lane with hits still open after round k sets gsUniformOpen to k: the round
    // numbers only grow within a step, so reading "at least k" after the barrier
    // is the same for every lane even if a fast lane has moved on to round k + 1.
    // Iterations 1..SELF_ROUNDS are the rounds (every lane, a barrier each); past them
    // (a dependency chain longer than the rounds) one lane per vehicle finishes the
    // open hits in key order, round 0 until none is open (at most HIT_CAP + 1 calls).
    // One call site of selfRound for both (the compilers inline every call): the same
    // calls in the same order as a loop each.
    uint go = 0u;
    bool done = false;
    for (uint it = 1u; it <= SELF_ROUNDS + HIT_CAP + 1u; it++)
    {
        bool last = it > SELF_ROUNDS;
        bool act = last ? (on && sub == 0u && !done) : on;
        uint open = act ? selfRound(v, last ? 0u : sub, last ? 1u : lanes, last ? 0u : it, origin, r, beamOff,
                                    groupOff, groupCount, parity)
                        : 0u;
        if (last)
        {
            if (act && open == 0u) done = true;
            continue;
        }
        if (open != 0u) gsUniformOpen = it;
        sync();
        uint seen = gsUniformOpen;
        go = seen >= it ? 1u : 0u;
        if (go == 0u) break;
    }
    if (go != 0u) sync();
    if (on && sub == 0u && uint(gAcc[hitAcc(v)].load()) > HIT_CAP)
        selfSerial(v, origin, r, beamOff, groupOff, groupCount, parity);
    // §37's residual pass (a listed vehicle's; selfSerial ran its own): the forces,
    // then the merge, each after every lane's previous phase.
    if (gP.selfSplit != 0u)
    {
        // (PERF_HIT_ROLES: a (hit, role) a lane, with the first pass's adds.)
        if ((gP.perfFlags & (PERF_HIT_ROLES | PERF_HIT_FB)) == (PERF_HIT_ROLES | PERF_HIT_FB))
        {
            selfResidualRoles(v, on, sub, lanes, beamOff, groupOff, groupCount, parity);
            return;
        }
        for (uint phase = 0u; phase < 3u; phase++)
        {
            sync();
            if (on) selfResidualPhase(v, sub, lanes, phase, beamOff, groupOff, groupCount, parity);
        }
        sync();
    }
}



// More hits in a step than the hit list holds: one lane runs vehicle v's
// self-collision of this step exactly as the specification orders it (tested
// triangles in file order, every candidate in node order, each hit added at once).
// Slow, and only for a step with more than HIT_CAP hits.
void selfSerialResidual(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity,
                        uint mode);
void selfSerialMerge(uint v);

void selfSerial(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    // The hits (mode 0), then (§37.1) the residual's modes 1 and 2, from one call site:
    // the compilers inline every call, so each site is a copy of the whole pass
    // (opaqueCount: the loop stays one; kernel-compile-time.md).
    for (uint mode = 0u, modeStop = EB_END(54, mode, opaqueCount(3u), 1u); mode < modeStop; mode++)
    {
        if (mode == 1u && (gP.selfSplit == 0u || !selfResidual(v))) return;
        selfSerialResidual(v, origin, r, beamOff, groupOff, groupCount, parity, mode);
    }
    selfSerialMerge(v);
}

// selfSerial's pass over the vehicle's tested triangles in file order, every candidate
// in node order. Mode 0: each hit applied as found (its position step too). §37.1's
// residual pass and guard, its hits found again (the tests have no side effects here):
// mode 1, the residual forces into B; mode 2, each hit's first-pass force again (without
// its position step), its residual (from A) and guard (A + B), the total into D.
void selfSerialResidual(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity,
                        uint mode)
{
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    for (uint t = 0u, tStop = EB_END(55, t, r1.y, 1u); t < tStop; t++)
    {
        uint gt = r1.x + t;
        if (asint(gMisc[gP.vehicles + gt].x) > 0) continue;
        TriGeo T = triangleGeo(gt, origin, r, parity);
        uint4 tn = T.tn;
        for (uint k = 0u, kStop = EB_END(56, k, r2.y, 1u); k < kStop; k++)
        {
            uint h = gLists[r2.x + k];
            if (h == tn.x || h == tn.y || h == tn.z) continue;
            float4 P = gNodes[h * 3u];
            float3 p = nf3(origin + P.xyz);
            if (any(p < T.lo) || any(p > T.hi)) continue;
            if (ownTravelOut(T, h, p, origin, r, parity)) continue;
            float alpha;
            float beta;
            float gamma;
            float delta;
            float3 nn;
            float pen;
            if ((T.clip & 1u) != 0u)
            {
                // Anti-clip: the test alone (tryContact recorded the sides).
                uint word;
                if (selfClip(T, h, p, origin, r, alpha, beta, gamma, pen, nn, word) != 1u) continue;
            }
            else
            {
                if (!triangleHit(T, p, r, alpha, beta, gamma, delta)) continue;
                nn = T.n;
                if (delta < 0.0)
                {
                    nn = -T.n;
                    delta = -delta;
                }
                pen = nf(r - delta);
            }
            if (mode == 0u) gMisc[T.mi].w = 1u;
            float4 V = gNodes[h * 3u + 1u];
            float4 P0 = gNodes[tn.x * 3u];
            float4 P1 = gNodes[tn.y * 3u];
            float4 P2 = gNodes[tn.z * 3u];
            bool two = mode != 0u || selfTwoBody(v);
            float3 rv = float3(0.0, 0.0, 0.0);
            if (two)
                rv = ssub3(V.xyz, sadd3(sadd3(sscale(gNodes[tn.y * 3u + 1u].xyz, alpha),
                                              sscale(gNodes[tn.z * 3u + 1u].xyz, beta)),
                                        sscale(gNodes[tn.x * 3u + 1u].xyz, gamma)));
            float rr = 0.0;
            if (mode != 0u)
            {
                rr = resOf(h, tn.x, tn.y, tn.z, alpha, beta, gamma, nn, rv, beamOff, groupOff, groupCount, parity, 0u);
                if (mode == 1u)
                {
                    if (rr > 0.0) resStage(h, tn.x, tn.y, tn.z, alpha, beta, gamma, sscale(nn, rr));
                    continue;
                }
            }
            float3 f;
            // (Mode 0 with its position step, mode 2 without: one call site.)
            if (two)
                f = selfTwoBodyCore(h, tn.x, tn.y, tn.z, alpha, beta, gamma, nn, pen, rv, V.w, beamOff, groupOff,
                                    groupCount, parity, mode == 0u, 0u);
            else
            {
                float3 fb = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff,
                                    groupCount, parity, false);
                f = hitContact(V, P, P0, P1, P2, gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz,
                               gNodes[tn.z * 3u + 1u].xyz, alpha, beta, gamma, pen, nn, fb, accForce(h), v);
            }
            if (mode == 2u)
            {
                float3 total = resTotal(h, tn.x, tn.y, tn.z, alpha, beta, gamma, nn, rv, f, rr, beamOff, groupOff,
                                        groupCount, parity, 0u);
                resFinal(h, tn.x, tn.y, tn.z, alpha, beta, gamma, f, total);
                continue;
            }
            accAdd(h, f);
            accAdd(tn.y, hitShare(f, 2u, alpha, beta, gamma));
            accAdd(tn.z, hitShare(f, 3u, alpha, beta, gamma));
            accAdd(tn.x, hitShare(f, 4u, alpha, beta, gamma));
            if (selfResidual(v)) continue; // (§37.1: logged as its total)
            cfAdd(h, f);
            cfAdd(tn.y, hitShare(f, 2u, alpha, beta, gamma));
            cfAdd(tn.z, hitShare(f, 3u, alpha, beta, gamma));
            cfAdd(tn.x, hitShare(f, 4u, alpha, beta, gamma));
        }
    }
}

// Vehicle v's nodes' D into their accumulators, B and D cleared (one lane).
void selfSerialMerge(uint v)
{
    uint4 r0 = gStatic[v * 4u];
    for (uint i = 0u, iStop = EB_END(57, i, r0.y, 1u); i < iStop; i++) resMergeNode(r0.x + i);
}

// Vehicle v's pending beams of this step's beam phase (breakException), on one lane
// after a barrier: each beam's update again, in spawn order, with the exception's
// live-beam counts as the sequential pass sees them.
void resolvePending(uint v, uint parity)
{
    uint n = uint(gAcc[hitAcc(v) + 1u].load());
    if (n == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint4 r2 = gStatic[v * 4u + 2u];
    bool fast = (gStatic[v * 4u + 3u].x & FLAG_FAST_INV_SQRT) != 0u;
    gResolving = 1u;
    // The listed beams in spawn order, or (more than the list holds) every pending beam
    // in index order: one call site of beamUpdate (the compilers inline every call).
    bool listed = n <= PEND_CAP;
    uint base = pendList(v);
    if (listed)
    {
        for (uint i = 1u, iStop = EB_END(58, i, n, 1u); i < iStop; i++)
        {
            uint x = gNext[base + i];
            uint j = i;
            uint jGuard = 0u;
            while (j > 0u && gNext[base + j - 1u] > x)
            {
                EB_PASS(59, jGuard);
                gNext[base + j] = gNext[base + j - 1u];
                j--;
            }
            gNext[base + j] = x;
        }
    }
    uint count = listed ? n : r0.w;
    for (uint i = 0u, iStop = EB_END(60, i, count, 1u); i < iStop; i++)
    {
        uint j = listed ? gNext[base + i] : i;
        if (!listed && gBeams[(r0.z + j) * 2u + 1u].w != 2.0) continue;
        beamUpdate(r0.z + j, j, fast, r2.z, parity);
    }
    gResolving = 0u;
}



void vvReportOutside(uint v)
{
    gAcc[vvCtl(VV_OUTSIDE)].max(int(v + 1u));
}

#ifdef EB_VV_GROUPS
void contactGroup(uint g, uint lane);
#endif

// Raycast wheels (engine_softbody::explicit_drive).
#include "explicit_beams_drive.slang"
#ifdef EB_SOLO_STEP
// The solo frame a step a dispatch (explicitBeamsFrameStep): what it keeps in workgroup
// memory from one step to the next, in the vehicle's records between dispatches: the hit
// flag (its segment record, free while contact runs: no wide passes) and the pair
// list's state (gsCount past the buckets, REUSE_WORDS words, at the vehicle's wide-pass
// grid counts in gAcc: free, a solo vehicle's grid is in workgroup memory).
uint soloStepAt(uint v) { return gP.gridCountBase + v * GRID_BUCKETS; }
void soloStepSave(uint v, uint lane)
{
    if (lane == 0u) gMisc[gP.wideMisc + 1u + v].x = gsHit;
    for (uint k = lane; k < REUSE_WORDS; k += LANES) gAcc[soloStepAt(v) + k].store(int(gsCount[GRID_BUCKETS + k].load()));
}
void soloStepLoad(uint v, uint lane)
{
    for (uint k = lane; k < REUSE_WORDS; k += LANES) gsCount[GRID_BUCKETS + k].store(uint(gAcc[soloStepAt(v) + k].load()));
}
#endif
#ifdef EB_SELF_SPLIT
// Self-collision's listed pairs across workgroups (selfcol-design.md).
#include "explicit_beams_selfsplit.slang"
#endif
// Surface contact (jbeam-semantics-spec.md §38).
#include "explicit_beams_surface.slang"
// Vehicle sleep (engine_softbody::explicit_sleep).
#include "explicit_beams_sleep.slang"
// The surface pass as its own pipelines (EB_SURF_PIPE modules only: the include under
// the define, so the other modules do not depend on the file).
#ifdef EB_SURF_PIPE
#include "explicit_beams_surface_pipes.slang"
#endif

