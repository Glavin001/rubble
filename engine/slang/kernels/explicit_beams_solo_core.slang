// The solo car's core frame (perf doc §8.43, the floor kernel's layout: PR #137
// floor2.metal): one explicit vehicle touching nothing, all of a frame's steps in one
// dispatch of one 1,024-lane workgroup. Positions and velocities live in workgroup memory
// for the frame; each hot beam (the beam order's hot block: JBeam plain, support,
// bounded and rope beams without an extension record) hands both its ends one scalar
// (its force over its length) through a time-shared pool, and each node gathers its
// entries in the reference's order as (x_self − x_other)·g, which is the frame kernel's
// push vector at both ends bit for bit (IEEE subtraction and multiplication are sign
// symmetric). The other records (aero triangles, torsion bars, couplers, extension
// records) take the frame kernel's own functions on device memory, so the node pass
// also keeps gNodes current every step.
//
// The same steps as explicit_beams_frame for such a vehicle (no self-collision, no
// coupling, no drive, no wings, slides, commands or brakes, no sleep, no
// retired nodes, none of the contact-only node paths): the host's fit check
// (engine_exec::explicit_plan::solo_core_table) refuses any other, which stays on
// today's path. Bit for bit with that kernel and the CPU reference in every state word.
//
// The pool, one tenant at a time: the beam pass's scalars (read by the next node pass's
// gather), then the centred correction's scalars (read by the centred node pass). Its
// size and the nodes' arrays are the host's per car: the WGSL's sentinel sizes below are
// replaced at pipeline creation (explicit_beams_gpu.rs solo_core_wgsl), so Chrome
// counts the car's workgroup memory, not the most any car could need.
#include "explicit_beams.slang"

// Table (engine_exec::explicit_plan::solo_core_table): header SOLO_HDR words, then per
// node entry k (its incidence list's k-th) of local node i, at SOLO_HDR + k × nodes + i
// (adjacent lanes, adjacent words): 0 (a cold record's entry: its push is in gBeams) or
// SOLO_HOT | pool index | other node (local) << 12.
[[vk::binding(0, 1)]] StructuredBuffer<uint> gSolo;
static const uint SOLO_HDR = 8u;
static const uint SOLO_HOT = 0x80000000u;
// A pool value that hands no force (+0 at both ends: the frame kernel's broken beam and
// a beam with no centred slope), not a value the rules compute.
static const uint SOLO_ZERO = 0x7FC0DEADu;
// (Through a uniform: a NaN literal is not WGSL.)
float soloZero() { return asfloat(SOLO_ZERO | noFuseBits()); }
// Sentinel sizes (replaced by the host: 3 × nodes, the pool's words).
static const uint SOLO_XYZ = 3067u;
static const uint SOLO_POOL = 4091u;
#ifdef SOLO_V_NOCARRY
#define SOLO_CARRY(x) 0.5
#else
#define SOLO_CARRY(x) (x)
#endif
#ifdef SOLO_V_DEVGATHER
#define SOLO_DEV(x) true
#else
#define SOLO_DEV(x) (x)
#endif
#ifdef SOLO_V_LOOP
#define SOLO_UNROLL [loop]
#else
#define SOLO_UNROLL [ForceUnroll]
#endif
#ifdef SOLO_V_TGBAR
#define sync GroupMemoryBarrierWithGroupSync
#endif
static const uint SOLO_SLOTS = 4u; // hot records a lane at most (the host checks H ≤ 4 × 1,024)

groupshared float gsX[SOLO_XYZ];
groupshared float gsU[SOLO_XYZ];
groupshared float gsT[SOLO_POOL];
// Whether a group master broke in the last beam pass (the slot the node passes read):
// lane 0's one load, published by the barrier after it.
groupshared uint gsGrp;

float3 soloX(uint n) { return float3(gsX[n * 3u], gsX[n * 3u + 1u], gsX[n * 3u + 2u]); }
float3 soloV(uint n) { return float3(gsU[n * 3u], gsU[n * 3u + 1u], gsU[n * 3u + 2u]); }
void soloSetX(uint n, float3 p)
{
    gsX[n * 3u] = p.x;
    gsX[n * 3u + 1u] = p.y;
    gsX[n * 3u + 2u] = p.z;
}
void soloSetV(uint n, float3 p)
{
    gsU[n * 3u] = p.x;
    gsU[n * 3u + 1u] = p.y;
    gsU[n * 3u + 2u] = p.z;
}

// A node's entries summed onto f in the reference's order (beamSum's): a hot entry from
// the pool (`dev`: from gBeams, the step after a frame boundary), a cold one from
// gBeams; with a group's master broken (`grp`) each entry's group drop as beamSum's.
float3 soloSum(float3 f, uint4 info, float3 xi, uint local, uint nodes, bool dev, bool grp, uint beamOff,
               uint groupOff, uint parity, bool mark)
{
    uint base = gP.incBase + info.y;
    for (uint k = 0u, kStop = EB_END(k, info.z, 1u); k < kStop; k++)
    {
#ifdef SOLO_V_NOTABLE
        uint e = 0x80000000u | ((local * 7u + k) % 3000u) | (((local + k + 1u) % nodes) << 12);
#else
        uint e = gSolo[SOLO_HDR + k * nodes + local];
#endif
        float3 push;
        if (dev || e == 0u)
            push = gBeams[base + k].xyz;
        else
        {
            float g = gsT[e & 4095u];
            push = asuint(g) == SOLO_ZERO ? float3(0.0, 0.0, 0.0) : sscale(ssub3(xi, soloX((e >> 12) & 1023u)), g);
        }
        if (!grp || !groupDrops(gLists[info.y + k] >> 1, beamOff, groupOff, parity, mark)) f = sadd3(f, push);
    }
    return f;
}

// jbeamHot (explicit_beams.slang), operation for operation, with the ends from
// workgroup memory and the force handed on as one scalar into the pool at `p`. Returns
// the raw extension rate (the centred objective form's u0 for the next step) and, in
// `slopeW`, the slope the frame kernel keeps in the entry's .w. `last`: the frame's last
// step, whose pushes also go to gBeams as the frame kernel leaves them.
float soloHot(uint p, uint4 a, uint4 b, uint beamOff, uint incStart, uint groupOff, uint parity, bool centred,
              bool keep, bool last, out float slopeW)
{
    uint gj = beamOff + a.x;
    uint local = a.x;
    slopeW = 0.0;
    cfFire(gj, groupOff, parity);
    uint slotA = gP.incBase + incStart + (a.w & 0xFFFFu);
    uint slotB = gP.incBase + incStart + (a.w >> 16);
#ifdef SOLO_V_NOSTATE
    float4 s1 = float4(1e9, 0.0, 0.0, 0.0);
    float4 s0 = float4(asfloat(b.w) * 0.5 + float(gj) * 1e-9, 1e9, 1e9, 1e9);
#else
    float4 s1 = gBeams[gj * 2u + 1u];
    float4 s0 = gBeams[gj * 2u];
#endif
    uint na = a.z & 0xFFFFu;
    uint nb = a.z >> 16;
    float3 pa = soloX(na);
    float3 pb = soloX(nb);
    float3 va = soloV(na);
    float3 vb = soloV(nb);
    if (s1.z != 0.0)
    {
        if (s1.w != 0.0) gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
        gsT[p] = soloZero();
        if (last)
        {
            gBeams[slotA] = float4(0.0, 0.0, 0.0, 0.0);
            gBeams[slotB] = float4(0.0, 0.0, 0.0, 0.0);
        }
        return 0.0;
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
    float3 d = ssub3(pa, pb);
    float q = sdot(d, d);
    float2 li = sExactLength(q);
    float inv = li.y;
    float len = li.x;
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    float uRaw = u;
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
    float lastF = 0.0;
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
        lastF = f;
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
        gBeams[gj * 2u + 1u] = float4(strength, lastF, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float g = smul(f, inv);
    gsT[p] = g;
    slopeW = !broken && centred ? smul(slope, gain) : 0.0;
    if (last)
    {
        float3 push = sscale(d, g);
        gBeams[slotA] = float4(push, slopeW);
        gBeams[slotB] = float4(-push, 0.0);
    }
    return uRaw;
}

// The raw extension rate jbeamHot computes for a hot record from the current state (0
// for a broken one, which has no slope): the frame's first step's u0.
float soloRate(uint4 a, uint beamOff)
{
    uint gj = beamOff + a.x;
    if (gBeams[gj * 2u + 1u].z != 0.0) return 0.0;
    uint na = a.z & 0xFFFFu;
    uint nb = a.z >> 16;
    float3 d = ssub3(soloX(na), soloX(nb));
    float inv = sExactLength(sdot(d, d)).y;
    return smul(sdot(ssub3(soloV(na), soloV(nb)), d), inv);
}

// centredBeamObjective for a hot record: its correction scalar along d (its ends now and
// u0, the raw rate at the step's start); no slope: no force.
float soloCentredHot(uint4 a, float slope, float u0)
{
    if (slope == 0.0) return soloZero();
    uint na = a.z & 0xFFFFu;
    uint nb = a.z >> 16;
    float3 d = ssub3(soloX(na), soloX(nb));
    float inv = sExactInvSqrt(sdot(d, d));
    float u1 = smul(sdot(ssub3(soloV(na), soloV(nb)), d), inv);
    float du = nf(u1 - u0);
    return smul(smul(smul(-0.5, slope), du), inv);
}

// One cold entry k of the beam order (beamPhaseSorted's loop past the hot block).
void soloColdBeam(uint k, uint at, uint nodeOff, uint beamOff, bool fast, uint groupOff, uint parity, uint incStart)
{
    uint4 a = gStatic[at + 2u * k];
    uint cls = a.y & 3u;
    uint gj = beamOff + a.x;
    if (cls == BEAM_ORDER_ATRI)
    {
        if (gBeams[gj * 2u + 1u].z == 0.0 && gBeams[(gj + 1u) * 2u + 1u].z == 0.0)
        {
            cfFire(gj + 1u, groupOff, parity);
            cfFire(gj, groupOff, parity);
            atriHot(a, gStatic[at + 2u * k + 1u], nodeOff, incStart);
            return;
        }
    }
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

[shader("compute")]
[numthreads(1024, 1, 1)]
void explicit_beams_solo_core(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = 0u;
    uint lane = lid.x;
    uint4 r0 = gStatic[0u];
    uint4 r1 = gStatic[1u];
    uint4 r2 = gStatic[2u];
    uint4 r3 = gStatic[3u];
    uint nodeOff = r0.x;
    uint nodeCount = r0.y;
    uint beamOff = r0.z;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
    uint incStart = gSolo[3u];
    // Measurement only (time, not results): phases left out (the host's RUBBLE_SOLO_SKIP).
    uint skip = gSolo[4u];
    uint4 head = gStatic[gP.beamOrder + v];
    uint at = gP.beamOrder + gP.vehicles + 2u * beamOff;
    uint hot = head.y;
    uint entries = head.x;
    bool centred = gP.centredDv != 0u && (flagsV & FLAG_CENTRED) != 0u;
    bool keep = (flagsV & FLAG_KEEP_STRENGTH) != 0u;

    // The origin rebase (explicit_beams_frame's).
    float3 first = gNodes[nodeOff * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane, iStop = EB_END(i, nodeCount, 1024u); i < iStop; i += 1024u)
        {
            float4 P = gNodes[(nodeOff + i) * 3u];
            gNodes[(nodeOff + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
    }
    sync();
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint stepBase = misc.w;

    // This lane's node in registers (and in workgroup memory for the others).
    bool mine = lane < nodeCount;
    uint gi = nodeOff + min(lane, max(nodeCount, 1u) - 1u);
    uint4 info = gStatic[gP.nodeInfoBase + gi];
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    float4 F = gNodes[gi * 3u + 2u];
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    float3 restart = F.xyz;
    // The node grab (grabApply, lane 0 after the node pass): the grabbed node's restart
    // force goes through gNodes each step.
    uint4 grabW = gMisc[grabRec()];
    bool grabbed = mine && grabW.x != 0u && grabW.y == gi;
    if (mine)
    {
        soloSetX(lane, pos);
        soloSetV(lane, vel);
    }
    if (lane == 0u)
        gsGrp = groupCount != 0u && gGroups[groupSlot((stepBase + 1u) & 1u, groupOff, 0u)].load() != NO_GROUP ? 1u : 0u;
    sync();
    // The hot records' carried words: the slope the last beam pass kept in the entry's .w
    // and the raw rate at this step's start.
    float slopeK[SOLO_SLOTS];
    float u0K[SOLO_SLOTS];
    SOLO_UNROLL
    for (uint q = 0u; q < SOLO_SLOTS; q++)
    {
        slopeK[q] = 0.0;
        u0K[q] = 0.0;
        uint k = lane + q * 1024u;
        if (k < hot)
        {
            uint4 a = gStatic[at + 2u * k];
            slopeK[q] = gBeams[gP.incBase + incStart + (a.w & 0xFFFFu)].w;
            u0K[q] = soloRate(a, beamOff);
        }
    }

    bool objective = (flagsV & FLAG_CENTRED_OBJECTIVE) != 0u;
    for (uint s = 0u, sStop = EB_END_UNIFORM(s, gP.steps, 1u, EB_MAX_STEPS); s < sStop; s++)
    {
        uint step = stepBase + s;
        uint parity = step & 1u;
#ifdef SOLO_V_DEVGATHER
        bool last = true;
#else
        bool last = s + 1u == gP.steps;
#endif
        // ---- the node pass: last step's beams gathered, integrated, restarted ----
        float3 fG = float3(0.0, 0.0, 0.0);
        if (mine)
        {
            uint pr = (step + 1u) & 1u;
            bool grp = gsGrp != 0u;
            if (grabbed) restart = gNodes[gi * 3u + 2u].xyz;
            if (centred && objective)
            {
                gBeams[gP.centredDv + gP.totalNodes + gi] = float4(pos, 0.0);
                gBeams[gP.centredDv + 2u * gP.totalNodes + gi] = float4(vel, 0.0);
            }
            if ((skip & 16u) == 0u) fG = soloSum(restart, info, pos, lane, nodeCount, SOLO_DEV(s == 0u), grp, beamOff, groupOff, pr, true);
        }
        // (Every lane's gather reads its neighbours' positions before any lane moves its
        // node.)
        GroupMemoryBarrierWithGroupSync();
        if (mine)
        {
            float3 f = fG;
            float3 v0 = vel;
            if (s == 0u)
            {
                // (The frame's first step reads the contact sums: none here, but the
                // frame kernel's addition of them is kept, sign of zero and all.)
                f = f + accForce(gi);
                gAcc[gi * 3u].store(0);
                gAcc[gi * 3u + 1u].store(0);
                gAcc[gi * 3u + 2u].store(0);
            }
            uint flags = info.x;
            if ((flags & NODE_NO_GROUND) == 0u)
            {
                float3 x = origin + pos;
                f = groundBoxContacts(f, vel, mass, x, info, v, ground, boxOff, boxCount, V.w);
            }
            if ((flags & NODE_IMMOVABLE) == 0u)
            {
                vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
                pos = sadd3(pos, sscale(vel, STEP));
            }
            f = float3(0.0, mass * GRAVITY, 0.0);
            if ((flagsV & FLAG_DRAG) != 0u)
            {
                float sp = approxSqrt(dot(vel, vel));
                f += -DRAG * sp * vel;
                if ((flagsV & FLAG_TURBULENCE) != 0u)
                {
                    float t = DRAG * sp * sp * TURBULENCE;
                    f += t * float3(noise(v, step, lane, 0u), noise(v, step, lane, 1u), noise(v, step, lane, 2u));
                }
            }
            restart = f;
            if (grabbed) gNodes[gi * 3u + 2u] = float4(restart, F.w);
            soloSetX(lane, pos);
            soloSetV(lane, vel);
            if ((skip & 4u) == 0u)
            {
                gNodes[gi * 3u] = float4(pos, mass);
                gNodes[gi * 3u + 1u] = float4(vel, V.w);
            }
            if (centred && (skip & 4u) == 0u) gBeams[gP.centredDv + gi] = float4(ssub3(vel, v0), 0.0);
        }
        // ---- centred damping (centredPhase): the records' corrections, then each node's ----
        if (gP.centredDv != 0u)
        {
            sync();
            if (centred)
            {
                SOLO_UNROLL
                for (uint q = 0u; q < SOLO_SLOTS; q++)
                {
                    uint k = lane + q * 1024u;
                    if (k < hot && (skip & 32u) == 0u) gsT[k] = soloCentredHot(gStatic[at + 2u * k], SOLO_CARRY(slopeK[q]), SOLO_CARRY(u0K[q]));
                }
                for (uint k = hot + lane, kStop = EB_END(k, (skip & 2u) != 0u ? 0u : entries, 1024u); k < kStop; k += 1024u)
                {
                    uint4 a = gStatic[at + 2u * k];
                    uint cls = a.y & 3u;
#ifndef SOLO_NO_COLD
                    centredBeam(beamOff + a.x, objective);
                    if (cls == BEAM_ORDER_PAIR || cls == BEAM_ORDER_ATRI) centredBeam(beamOff + a.x + 1u, objective);
#endif
                }
            }
            sync();
            if (centred && mine)
            {
                uint pr = (step + 1u) & 1u;
                bool grp = gsGrp != 0u;
                if ((skip & 64u) == 0u) fG = soloSum(float3(0.0, 0.0, 0.0), info, pos, lane, nodeCount, false, grp, beamOff, groupOff, pr, false);
            }
            GroupMemoryBarrierWithGroupSync();
            if (centred && mine)
            {
                float3 c = fG;
                if ((info.x & NODE_IMMOVABLE) == 0u)
                {
                    float3 dv = sscale(sdiv3(c, mass), STEP);
                    vel = sadd3(vel, dv);
                    pos = sadd3(pos, sscale(dv, STEP));
                    soloSetX(lane, pos);
                    soloSetV(lane, vel);
                    gNodes[gi * 3u + 1u] = float4(vel, V.w);
                    gNodes[gi * 3u] = float4(pos, mass);
                }
            }
        }
        for (uint g = lane, gStop = EB_END(g, groupCount, 1024u); g < gStop; g += 1024u)
            gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
        if (lane == 0u)
        {
            gAcc[hitAcc(v)].store(0);
            gAcc[hitAcc(v) + 1u].store(0);
            clipVmaxClear(v, parity);
        }
        sync();
        if (lane == 0u && grabW.x != 0u) grabApply(nodeOff, nodeCount, origin);
        // ---- the beam pass ----
        SOLO_UNROLL
        for (uint q = 0u; q < SOLO_SLOTS; q++)
        {
            uint k = lane + q * 1024u;
            if (k < hot && (skip & 8u) == 0u)
            {
                float sw;
                u0K[q] = soloHot(k, gStatic[at + 2u * k], gStatic[at + 2u * k + 1u], beamOff, incStart, groupOff, parity,
                                 centred, keep, last, sw);
                slopeK[q] = sw;
            }
        }
        for (uint k = hot + lane, kStop = EB_END(k, (skip & 1u) != 0u ? 0u : entries, 1024u); k < kStop; k += 1024u)
        {
#ifndef SOLO_NO_COLD
            soloColdBeam(k, at, nodeOff, beamOff, fast, groupOff, parity, incStart);
#endif
        }
        sync();
        if (lane == 0u)
        {
            resolvePending(v, parity);
            gsGrp = groupCount != 0u && gGroups[groupSlot(parity, groupOff, 0u)].load() != NO_GROUP ? 1u : 0u;
        }
        sync();
    }
    // ---- the frame's end (explicit_beams_frame's for such a vehicle) ----
    if (mine && !grabbed) gNodes[gi * 3u + 2u] = float4(restart, F.w);
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint lastP = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane, jStop = EB_END(j, r0.w, 1024u); j < jStop; j += 1024u) groupDrops(beamOff + j, beamOff, groupOff, lastP, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
    ebLoopCapFlush();
}
