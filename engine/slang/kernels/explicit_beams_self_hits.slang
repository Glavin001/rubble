// --- Vehicle contact's kept candidates (PERF_FLEET_KEEP): addresses, after the fleet
// state's own (explicit_plan::fleet_acc_words, fleet_next_words; the passes are
// explicit_beams_fleet.slang's fk*) ---
// gAcc: per vehicle FK_ACC words (for its calm and its wild tracked nodes, the farthest
// one has moved since the build relative to the vehicle's reference motion and the
// fastest one's travel over a step, float bits, zero between passes; its kept pairs; the
// box of its wild candidates at the build, vvKey's keys; its tracked nodes' motions
// summed, fixed point, zero between passes), then FK_ACC_CTL words.
static const uint FK_ACC = 14u;
static const uint FK_SUM = 11u;
static const uint FK_DEV = 0u; // calm; + 2: wild
static const uint FK_TRAV = 1u; // calm; + 2: wild
static const uint FK_PAIRS = 4u;
static const uint FK_WILD_BOX = 5u;
static const uint FK_ACC_CTL = 4u;
uint fkAcc(uint v, uint w) { return flCtl(FL_CTL) + v * FK_ACC + w; }
uint fkAccCtl(uint i) { return fkAcc(gP.vehicles, 0u) + i; }
// gNext: per triangle whether its throttle step listed it this pass; per vehicle its
// partners the build covered (a count, then FK_COVER vehicles, the top bit: one of that
// partner's points was near one of this vehicle's triangles at the build); the prefixes over the
// vehicles of the build's candidate runs and of the kept pairs (one more each for the
// total); per vehicle its origin at the build (float bits); FK_CTL words.
static const uint FK_COVER = 8u;
static const uint FK_CTL = 16u;
uint fkListed(uint gt) { return flCtlNext(FL_CTL) + gt; }
uint fkCover(uint v, uint i) { return fkListed(gP.totalTris) + v * (FK_COVER + 1u) + i; }
uint fkRun(uint v) { return fkCover(gP.vehicles, 0u) + v; }
uint fkPrefix(uint v) { return fkRun(gP.vehicles + 1u) + v; }
// (The kept pairs' places: per owner the prefix of its share of the fleet's room at the
// build, fkAlloc; one more for the total; then per owner its share at the last build.)
uint fkAlloc(uint v) { return fkPrefix(gP.vehicles + 1u) + v; }
uint fkShareAt(uint v) { return fkAlloc(gP.vehicles + 1u) + v; }
uint fkOrigin(uint v) { return fkShareAt(gP.vehicles + 1u) + 3u * v; }
// (Per vehicle six words, float bits: its reference motion of this pass, 0..2, and its
// tracked nodes' mean motion of the pass before, 3..5.)
uint fkMean(uint v, uint i) { return fkOrigin(gP.vehicles) + 6u * v + i; }
uint fkCtl(uint i) { return fkMean(gP.vehicles, 0u) + i; }
// gBeams after the owners' boxes: per node its position (its vehicle's frame) at the build
// and its class there (.w: 0 calm, 1 wild).
uint fkRef(uint n) { return gP.vvFleetBox + 4u * gP.vehicles + n; }

// A value one lane decided, read by every lane as uniform (the WGSL
// post-process turns its reads into workgroupUniformLoad).
groupshared uint gsUniformLanes;

// --- Self-collision hits in the specification's order (see tryContact) ---

// Rounds of selfRound before one lane finishes what is left in key order.
static const uint SELF_ROUNDS = 32u;

// Node `h`'s roles in the hit recorded at `at` (key `key`): bit 0 the contacter,
// bits 1..3 the triangle's n1, n2, n0 (a triangle may name a node twice); 0 none.
uint hitRoles(uint at, uint key, uint h, uint nodeOff)
{
    if (nodeOff + (key & 0xFFFFu) == h) return 1u;
    return (gNext[at + 9u] == h ? 2u : 0u) | (gNext[at + 10u] == h ? 4u : 0u) | (gNext[at + 8u] == h ? 8u : 0u);
}

// The accumulator shares (fixed point) node roles `roles` get from the computed hit
// at record `at`, as accAdd adds them.
int3 hitShareFixed(uint at, uint roles)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    float gamma = nf(nf(1.0 - alpha) - beta);
    float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
    int3 sum = int3(0, 0, 0);
    for (uint role = 1u; role <= 4u; role++)
    {
        if ((roles & (1u << (role - 1u))) != 0u) sum += toFixed3(hitShare(f, role, alpha, beta, gamma));
    }
    return sum;
}

// A hit's contact force: contacter (velocity V, position and mass P) against the
// triangle (masses P0..P2, velocities v0..v2) at weights (alpha, beta, gamma),
// distance delta along normal nn; its force is fb (beams) plus a3 (accumulator).
// Strict, in the CPU reference's order (the test-only EB_DET_HITS build of hitForce
// keeps the frozen baseline's expressions instead).
float3 hitContact(float4 V, float4 P, float4 P0, float4 P1, float4 P2, float3 v0, float3 v1, float3 v2, float alpha,
                  float beta, float gamma, float pen, float3 nn, float3 fb, float3 a3, uint v)
{
    float3 vv = ssub3(V.xyz, sadd3(sadd3(sscale(v1, alpha), sscale(v2, beta)), sscale(v0, gamma)));
    float mt = nf(nf(smul(alpha, P1.w) + smul(beta, P2.w)) + smul(gamma, P0.w));
    float m = sdiv(smul(P.w, mt), nf(P.w + mt));
    return contactForce(sadd3(fb, a3), vv, m, nn, contactSurfaceV(v, v), pen, V.w);
}

// The force of the hit at record `at` (key `key`) of vehicle v: its node's force,
// with `sum` the fixed-point shares of every earlier hit on that node (as the
// accumulator would hold them had the hits been added in order), then the contact
// force from the geometry the finding lane stored; stored with `round`.
void hitForce(uint v, uint at, uint key, int3 sum, uint round, float3 origin, float r, uint beamOff, uint groupOff,
              uint groupCount, uint parity)
{
    uint h = gStatic[v * 4u].x + (key & 0xFFFFu);
    float3 nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
    float pen = asfloat(gNext[at + 15u]);
    float3 vv = asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u]));
    float m = asfloat(gNext[at + 19u]);
    float friction = asfloat(gNext[at + 20u]);
    // The accumulator as it would read with the earlier hits added (accForce). It
    // holds nothing else in this phase: the node phase took and cleared it, and
    // vehicle contact adds after the vehicles' steps.
    int3 acc = sum;
    float3 fb = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity,
                        false);
    float3 a3 = float3(float(acc.x), float(acc.y), float(acc.z)) * (1.0 / FIXED);
#ifdef EB_DET_HITS
    float3 f = contactForce(fb + a3, vv, m, nn, contactSurfaceV(v, v), pen, friction);
#else
    float3 f = contactForce(sadd3(fb, a3), vv, m, nn, contactSurfaceV(v, v), pen, friction);
#endif
    gNext[at + 3u] = asuint(f.x);
    gNext[at + 4u] = asuint(f.y);
    gNext[at + 5u] = asuint(f.z);
    gNext[at + 6u] = round;
}

bool selfResidual(uint v);
bool hitRolesOn(uint v);

// The computed hit at record `at` adds its shares to the accumulators.
void selfAddOne(uint at, uint key, uint nodeOff)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    float gamma = nf(nf(1.0 - alpha) - beta);
    float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
    accAdd(nodeOff + (key & 0xFFFFu), f);
    accAdd(gNext[at + 9u], hitShare(f, 2u, alpha, beta, gamma));
    accAdd(gNext[at + 10u], hitShare(f, 3u, alpha, beta, gamma));
    accAdd(gNext[at + 8u], hitShare(f, 4u, alpha, beta, gamma));
    // (§37.1: a staged hit logs its total at the end.)
    if (selfResidual(gStatic[gP.nodeInfoBase + nodeOff].w)) return;
    cfAdd(nodeOff + (key & 0xFFFFu), f);
    cfAdd(gNext[at + 9u], hitShare(f, 2u, alpha, beta, gamma));
    cfAdd(gNext[at + 10u], hitShare(f, 3u, alpha, beta, gamma));
    cfAdd(gNext[at + 8u], hitShare(f, 4u, alpha, beta, gamma));
}

// One round over vehicle v's hits (lane `sub` of `lanes`): every open hit whose
// earlier hits (smaller keys) on its node were computed in an earlier round is
// computed now. `round` 0 takes any hit whose earlier hits on its node are computed
// (the one-lane pass, in key order). Returns the hits left open.
uint selfRound(uint v, uint sub, uint lanes, uint round, float3 origin, float r, uint beamOff, uint groupOff,
               uint groupCount, uint parity)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP) return 0u;
    uint base = hitList(v);
    uint nodeOff = gStatic[v * 4u].x;
    uint open = 0u;
    for (uint i = sub, iStop = EB_END(46, i, count, lanes); i < iStop; i += lanes)
    {
        uint at = base + i * HIT_WORDS;
        if (gNext[at + 6u] != 0u) continue;
        uint key = gNext[at];
        uint h = nodeOff + (key & 0xFFFFu);
        if (selfTwoBody(v))
        {
            // Two-body (spec §36): no hit reads another's force.
            float3 f2 = selfTwoBodyForce(at, h, beamOff, groupOff, groupCount, parity,
                                         (gP.perfFlags & PERF_HIT_FB) != 0u ? hitFbAt(v, i) : 0u);
            gNext[at + 3u] = asuint(f2.x);
            gNext[at + 4u] = asuint(f2.y);
            gNext[at + 5u] = asuint(f2.z);
            gNext[at + 6u] = round != 0u ? round : SELF_ROUNDS + 1u;
            // (PERF_HIT_ROLES: its shares a role a lane, after this round's barrier.)
            if (!hitRolesOn(v)) selfAddOne(at, key, nodeOff);
            continue;
        }
        int3 sum = int3(0, 0, 0);
        bool ready = true;
        for (uint j = 0u, jStop = EB_END(47, j, count, 1u); j < jStop; j++)
        {
            uint bj = base + j * HIT_WORDS;
            uint kj = gNext[bj];
            if (kj >= key) continue;
            uint role = hitRoles(bj, kj, h, nodeOff);
            if (role == 0u) continue;
            uint rj = gNext[bj + 6u];
            if (rj == 0u || (round != 0u && rj == round))
            {
                ready = false;
                break;
            }
            sum += hitShareFixed(bj, role);
        }
        if (!ready)
        {
            open++;
            continue;
        }
        uint done = round != 0u ? round : SELF_ROUNDS + 1u;
        if (gNext[at + 7u] != 0u && all(sum == int3(0, 0, 0)))
            gNext[at + 6u] = done;
        else
            hitForce(v, at, key, sum, done, origin, r, beamOff, groupOff, groupCount, parity);
        // Its shares now: no later hit reads the accumulators (they sum the shares
        // from the records), so the adds need no order.
        selfAddOne(at, key, nodeOff);
    }
    return open;
}

// --- §37.1's residual pass and work guard (explicit_beams.rs residual_self_pass,
// friction_guard_pass) ---
// The first pass's sums are a staging (in the accumulators, A). The residual pass
// reads A and stages its forces in B; the guard reads A + B. Then each hit's total
// (first pass + residual + guard) replaces its first-pass adds: D gathers, per
// node, fix(total share) − fix(first-pass share) and is merged into A, so the
// accumulators end as they were plus one add per share of each total (the CPU's
// restore and re-add, integer for integer). Per node 8 words of gAcc at resAt:
// unused, B xyz, D xyz, unused (zero between steps).

uint resAt(uint x) { return gP.selfSplit + 4u * gP.totalNodes + 8u * x; }

// Vehicle v takes the residual pass: the two-body rule, dissipative.
bool selfResidual(uint v)
{
    return selfTwoBody(v) && (gStatic[v * 4u + 3u].x & FLAG_DISSIPATIVE) != 0u;
}

// Node x's acceleration from its force now (0 immovable): beams and A, with B when
// `withB` (the integer sum A + B, then to newtons, as the CPU's one sum).
float3 resAccNow(uint x, uint fb, uint beamOff, uint groupOff, uint groupCount, uint parity, bool withB)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return float3(0.0, 0.0, 0.0);
    float3 f = hitBeamSum(x, fb, beamOff, groupOff, groupCount, parity);
    int3 a = int3(gAcc[x * 3u].load(), gAcc[x * 3u + 1u].load(), gAcc[x * 3u + 2u].load());
    if (withB)
    {
        uint at = resAt(x);
        a += int3(gAcc[at + 1u].load(), gAcc[at + 2u].load(), gAcc[at + 3u].load());
    }
    float3 c = float3(float(a.x), float(a.y), float(a.z)) * (1.0 / FIXED);
    return sdiv3(sadd3(f, c), gNodes[x * 3u].w);
}
float3 resAcc(uint x, uint fb, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    return resAccNow(x, fb, beamOff, groupOff, groupCount, parity, false);
}

// The hit's pair (contacter h, triangle t0..t2 at weights, normal nn, relative
// velocity vv): its relative normal acceleration `a` now, `x` = vn + max(vn, 0),
// and whether it approaches (vn < 0) and still does over the step.
bool resDrive(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 nn, float3 vv,
              uint beamOff, uint groupOff, uint groupCount, uint parity, uint fb, out float a, out float x)
{
    // at = α·a(t1) + β·a(t2) + γ·a(t0), then a(h): from one call site (opaqueCount),
    // the same strict sums in the same order.
    float3 at = float3(0.0, 0.0, 0.0);
    float3 ah = float3(0.0, 0.0, 0.0);
    for (uint k = 0u, kStop = EB_END(48, k, opaqueCount(4u), 1u); k < kStop; k++)
    {
        float3 acc = resAcc(k == 0u ? t1 : (k == 1u ? t2 : (k == 2u ? t0 : h)),
                            fb == 0u ? 0u : fb + (k == 0u ? 6u : (k == 1u ? 9u : (k == 2u ? 3u : 0u))), beamOff, groupOff,
                            groupCount, parity);
        if (k == 0u) at = sscale(acc, alpha);
        else if (k == 1u) at = sadd3(at, sscale(acc, beta));
        else if (k == 2u) at = sadd3(at, sscale(acc, gamma));
        else ah = acc;
    }
    a = sdot(ssub3(ah, at), nn);
    float vn = sdot(vv, nn);
    x = nf(vn + max(vn, 0.0));
    return vn < 0.0 && nf(smul(a, STEP) + x) < 0.0;
}

// An active hit's residual normal force r (0: none), the first pass's counts.
float resForceR(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float a, float x)
{
    float imt = nf(nf(selfTerm(alpha, selfInvMass(t1)) + selfTerm(beta, selfInvMass(t2))) + selfTerm(gamma, selfInvMass(t0)));
    float inv = nf(selfInvMass(h) + imt);
    if (!(inv > 0.0)) return 0.0;
    float m = sdiv(1.0, inv);
    float r = nf(-smul(a, m) - sdiv(smul(x, m), STEP));
    return r > 0.0 ? r : 0.0;
}

// The hit's residual r (0: none) from A.
float resOf(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 nn, float3 vv,
            uint beamOff, uint groupOff, uint groupCount, uint parity, uint fb)
{
    float a;
    float x;
    if (!resDrive(h, t0, t1, t2, alpha, beta, gamma, nn, vv, beamOff, groupOff, groupCount, parity, fb, a, x)) return 0.0;
    return resForceR(h, t0, t1, t2, alpha, beta, gamma, a, x);
}

// The residual force's shares into B.
void resStage(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 f)
{
    uint4 n4 = uint4(h, t1, t2, t0);
    for (uint role = 1u; role <= 4u; role++)
    {
        uint x = role == 1u ? n4.x : (role == 2u ? n4.y : (role == 3u ? n4.z : n4.w));
        float3 s = hitShare(f, role, alpha, beta, gamma);
        uint at = resAt(x);
        gAcc[at + 1u].add(toFixed(s.x));
        gAcc[at + 2u].add(toFixed(s.y));
        gAcc[at + 3u].add(toFixed(s.z));
    }
}

// The work guard of a hit's force t (first pass with residual): −(1 − s)·t where
// its work at the pair's realised mean velocity (A + B) would be positive; 0 else.
bool guardOf(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 vv, float3 t,
             uint beamOff, uint groupOff, uint groupCount, uint parity, uint fb, out float3 g)
{
    g = float3(0.0, 0.0, 0.0);
    if (all(t == float3(0.0, 0.0, 0.0))) return false;
    // a = a(h) − (α·a(t1) + β·a(t2) + γ·a(t0)): the four from one call site
    // (opaqueCount), h first as written, the same strict sums in the same order.
    float3 ah = float3(0.0, 0.0, 0.0);
    float3 aw = float3(0.0, 0.0, 0.0);
    for (uint k = 0u, kStop = EB_END(49, k, opaqueCount(4u), 1u); k < kStop; k++)
    {
        float3 acc = resAccNow(k == 0u ? h : (k == 1u ? t1 : (k == 2u ? t2 : t0)),
                               fb == 0u ? 0u : fb + (k == 0u ? 0u : (k == 1u ? 6u : (k == 2u ? 9u : 3u))), beamOff, groupOff,
                               groupCount, parity, true);
        if (k == 0u) ah = acc;
        else if (k == 1u) aw = sscale(acc, alpha);
        else if (k == 2u) aw = sadd3(aw, sscale(acc, beta));
        else aw = sadd3(aw, sscale(acc, gamma));
    }
    float3 a = ssub3(ah, aw);
    float inv = nf(ownInvMass(h)
                   + nf(nf(selfTerm(alpha, ownInvMass(t1)) + selfTerm(beta, ownInvMass(t2))) + selfTerm(gamma, ownInvMass(t0))));
    float3 rest = ssub3(a, sscale(t, inv));
    float aa = sdot(t, sadd3(sadd3(vv, vv), sscale(rest, STEP)));
    float bb = smul(smul(STEP, sdot(t, t)), inv);
    if (!(nf(aa + bb) > 0.0 && bb > 0.0)) return false;
    float k = aa < 0.0 ? min(sdiv(-aa, bb), 1.0) : 0.0;
    g = sscale(t, -nf(1.0 - k));
    return true;
}

// A hit's total replaces its first-pass adds: per node, D += fix(total share) −
// fix(first share); the total's magnitudes into the contact-force log.
void resFinal(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 f1, float3 total)
{
    uint4 n4 = uint4(h, t1, t2, t0);
    for (uint role = 1u; role <= 4u; role++)
    {
        uint x = role == 1u ? n4.x : (role == 2u ? n4.y : (role == 3u ? n4.z : n4.w));
        float3 st = hitShare(total, role, alpha, beta, gamma);
        int3 d = toFixed3(st) - toFixed3(hitShare(f1, role, alpha, beta, gamma));
        uint at = resAt(x);
        if (d.x != 0) gAcc[at + 4u].add(d.x);
        if (d.y != 0) gAcc[at + 5u].add(d.y);
        if (d.z != 0) gAcc[at + 6u].add(d.z);
        cfAdd(x, st);
    }
}

// The total of a hit (first pass f1, residual r along nn) with its guard.
float3 resTotal(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 nn, float3 vv,
                float3 f1, float r, uint beamOff, uint groupOff, uint groupCount, uint parity, uint fb)
{
    float3 f = f1;
    if (r > 0.0) f = sadd3(f, sscale(nn, r));
    float3 g;
    if (guardOf(h, t0, t1, t2, alpha, beta, gamma, vv, f, beamOff, groupOff, groupCount, parity, fb, g)) f = sadd3(f, g);
    return f;
}

// Node x's D into its accumulator, its B and D cleared (the first to get here).
void resMergeNode(uint x)
{
    uint at = resAt(x);
    gAcc[at + 1u].store(0);
    gAcc[at + 2u].store(0);
    gAcc[at + 3u].store(0);
    int dx = gAcc[at + 4u].exchange(0);
    int dy = gAcc[at + 5u].exchange(0);
    int dz = gAcc[at + 6u].exchange(0);
    if (dx != 0) gAcc[x * 3u].add(dx);
    if (dy != 0) gAcc[x * 3u + 1u].add(dy);
    if (dz != 0) gAcc[x * 3u + 2u].add(dz);
}

void resMerge(uint h, uint t0, uint t1, uint t2)
{
    resMergeNode(h);
    resMergeNode(t0);
    resMergeNode(t1);
    resMergeNode(t2);
}

// One phase of the pass over vehicle v's listed hits (lane `sub` of `lanes`):
// 0 the residual (r in record word 21) into B; 1 the guard and the totals into D;
// 2 the merge.
void selfResidualPhase(uint v, uint sub, uint lanes, uint phase, uint beamOff, uint groupOff, uint groupCount,
                       uint parity)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP || !selfResidual(v)) return;
    uint base = hitList(v);
    uint nodeOff = gStatic[v * 4u].x;
    for (uint i = sub, iStop = EB_END(50, i, count, lanes); i < iStop; i += lanes)
    {
        uint at = base + i * HIT_WORDS;
        uint h = nodeOff + (gNext[at] & 0xFFFFu);
        uint t0 = gNext[at + 8u];
        uint t1 = gNext[at + 9u];
        uint t2 = gNext[at + 10u];
        if (phase == 2u)
        {
            resMerge(h, t0, t1, t2);
            continue;
        }
        uint fb = (gP.perfFlags & PERF_HIT_FB) != 0u ? hitFbAt(v, i) : 0u;
        float alpha = asfloat(gNext[at + 1u]);
        float beta = asfloat(gNext[at + 2u]);
        float gamma = nf(nf(1.0 - alpha) - beta);
        float3 nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
        float3 vv = asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u]));
        if (phase == 0u)
        {
            float r = resOf(h, t0, t1, t2, alpha, beta, gamma, nn, vv, beamOff, groupOff, groupCount, parity, fb);
            gNext[at + 21u] = asuint(r);
            if (r > 0.0) resStage(h, t0, t1, t2, alpha, beta, gamma, sscale(nn, r));
            continue;
        }
        float3 f1 = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
        float3 total = resTotal(h, t0, t1, t2, alpha, beta, gamma, nn, vv, f1, asfloat(gNext[at + 21u]), beamOff,
                                groupOff, groupCount, parity, fb);
        resFinal(h, t0, t1, t2, alpha, beta, gamma, f1, total);
    }
}

