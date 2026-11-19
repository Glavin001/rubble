// The floor kernel, second layout (the host puts floor.metal's element functions before
// this file). Built from the measured cost model (microbench): a phase costs its warps'
// instructions (0.35 ns a warp-instruction), a barrier 0.07 µs, a threadgroup-memory access
// about one instruction, a device access 4 to 29 ns a warp, and more than about 48 live
// registers a lane spill. So:
//  - a beam's hot words (ends, k, c, rest length, last extension rate) stay in registers:
//    five a slot, four slots a lane; everything else of a beam is read from device memory
//    only on the step it yields, breaks or is one of the car's few special elements;
//  - a beam hands its ends one scalar (force over length) through threadgroup memory; each
//    node gathers its beams in the reference's order from a streamed list, (x_self −
//    x_other)·g, which is the reference's push bit for bit at both ends;
//  - self-collision is a lane a triangle over its own row of listed nodes, each pair
//    filtered by two plane distances (now, and a step ago) before the full rule;
//  - contact sums, side memory, break groups, triangle breaks: device memory, touched only
//    on steps flagged as having them.
// Host defines: NN NR NT NG ND LANES STEPS NC NCW NGS (scalar slots) NV (vector slots)
// AERO_LO T_LANE T_ENT T_VIDX MAXENT PROW REFNODE NCPL HIT_CAP HITW CR LIST_REACH LIST_EPS NBOX.

struct Params
{
    uint cars;
    uint frames;
    float ground;
    uint skip;
    uint rtLanes, rtNN, rtNT, rtNC, rtSteps, rtNG, rtRS, pad2;
};
#define SKIP(k) ((P.skip & (1u << (k))) != 0u)
#if RUNTIME_SIZES
#define Z_STEPS P.rtSteps
#else
#define Z_STEPS STEPS
#endif
#define RS 4
// REGMODE: which of a slot's words stay in registers between phases (the rest come from
// the slot's hot record, one streamed device load a slot a phase). 0: w0, k, c, rest, up;
// 1: w0, c, up; 2: w0, up.
#ifndef REGMODE
#define REGMODE 1
#endif
#define HOTS(s) hot[lid * 6 + (s)]
#define BAR threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define TGP(i) v3(tgPos[(i) * 3], tgPos[(i) * 3 + 1], tgPos[(i) * 3 + 2])
#define TGV(i) v3(tgVel[(i) * 3], tgVel[(i) * 3 + 1], tgVel[(i) * 3 + 2])
#define PUTV(vi, val) { V3 _v = (val); tgVec[(vi) * 3] = _v.x; tgVec[(vi) * 3 + 1] = _v.y; tgVec[(vi) * 3 + 2] = _v.z; }
// Slot word of a two-node beam: 3·a (11 bits), 3·b (11), alive (bit 22), slope on (23),
// deform asked (24), class (30..31). Of an aero triangle: its three nodes, 10 bits each.
#define W_ALIVE (1u << 22)
#define W_SLOPE (1u << 23)
#define W_DEFORM (1u << 24)
#define TG3P(o) v3(tgPos[o], tgPos[(o) + 1u], tgPos[(o) + 2u])
#define TG3V(o) v3(tgVel[o], tgVel[(o) + 1u], tgVel[(o) + 2u])
#define ENT_ADD(acc, e) { uint o3_ = (e) & 2047u; acc = acc + (Ps - TG3P(o3_)) * tgG[(e) >> 11]; }
#define C_PLAIN 0u
#define C_SUPPORT 1u
#define C_AERO 2u
#define C_GENERIC 3u
// Flags (threadgroup): see the kernel.
#define F_SHIFT 0
#define F_FIRED 1
#define F_REBUILD 2
#define F_HITS 3
#define F_BUILDS 4
#define F_MOST 5
#define F_STATUS 6
#define F_REC 7
#define F_HELD 8
#define F_GDIRTY 9
#define F_BROKE 10
#define TG_FLAGS 12

struct Slow { float g; float u; float rest; uint bits; };
// One two-node beam's whole rule from its device records (the step it may yield or break,
// and the special kinds every step): bits 1 alive, 2 slope on, 4 deform asked, 8 its group
// set off, 16 broke now.
inline Slow slowScalar(uint r, device const uint4* recA, device const float4* recK, device const float4* recB,
                       device const float4* recE, device float4* recState, device atomic_uint* masters,
                       V3 pa, V3 pb, V3 va, V3 vb)
{
    uint4 a = recA[r];
    uint kind = a.z & 0xFFu, ty = (a.z >> 8) & 0xFFu;
    bool ext = ((a.z >> 16) & 1u) != 0u;
    int group = as_type<int>(a.w);
    float4 kc = recK[r], bb = recB[r];
    float4 s0 = recState[r * 3], s1 = recState[r * 3 + 1];
    float4 e1 = float4(0), e2 = float4(0);
    if (ext) { e1 = recE[r * 2]; e2 = recE[r * 2 + 1]; }
    BeamOut o;
    if (kind == K_PLAIN) o = plainForce(kc, bb, s0, s1, pa, pb, va, vb);
    else o = jbeamForce(kind, ty, kc, bb, e1, e2, ext, s0, s1, pa, pb, va, vb);
    Slow out;
    out.u = dot3(va - vb, o.d) * o.inv;
    float f = o.f;
    bool brokeNow = o.broke;
    if (o.check) { brokeNow = true; f = 0.0f; }
    if (brokeNow) s1.w = 1.0f;
    s1.z = (brokeNow || kind == K_PLAIN) ? 0.0f : o.slope;
    uint bits = brokeNow ? 16u : 1u;
    if (s1.z != 0.0f) bits |= 2u;
    if (ext)
    {
        uint mask = as_type<uint>(bb.w);
        bool yields = kind == K_JPLAIN || kind == K_JSUPPORT || kind == K_JANISO;
        if ((mask & EXT_DEFORM) != 0u && (brokeNow || (yields && fabs(s0.x - e2.x) > e2.w * e2.x))) bits |= 4u;
    }
    if (brokeNow && group > 0)
    {
        atomic_fetch_min_explicit(&masters[group], r, memory_order_relaxed);
        bits |= 8u;
    }
    recState[r * 3] = s0;
    recState[r * 3 + 1] = s1;
    recState[r * 3 + 2] = float4(out.u, 0.0f, 0.0f, 0.0f);
    out.g = f * o.inv;
    out.rest = s0.x;
    out.bits = bits;
    return out;
}

kernel void floorFrame(
    constant Params& P [[buffer(0)]],
    device uint* big [[buffer(1)]],                  // every per-car word (CAR_WORDS a car), one base
    device const float4* nodeStatic [[buffer(2)]],
    device const uint4* recA [[buffer(4)]],
    device const float4* recK [[buffer(5)]],
    device const float4* recB [[buffer(6)]],
    device const float4* recE [[buffer(7)]],
    device const uint* tab [[buffer(12)]],           // lane table, node entry lists, vector indices
    device const uint* perm [[buffer(14)]],          // RS*LANES: the record of slot s of lane l
    device const uint4* triBreaks [[buffer(17)]],
    device const uint* cands [[buffer(22)]],
    device const uint4* tris [[buffer(23)]],
    device const uint* allowed [[buffer(24)]],
    uint lid [[thread_position_in_threadgroup]],
    uint car [[threadgroup_position_in_grid]])
{
    threadgroup float tgPos[NN * 3];
    threadgroup float tgVel[NN * 3];
    threadgroup float tgG[NGS];
    threadgroup float tgVec[NV * 3];
    threadgroup uchar tgHeld[NN];
    threadgroup atomic_uint tgCount[4]; // [0] the fastest node (bits), [2] hits
    threadgroup uint tgFlags[TG_FLAGS];
    threadgroup float tgPB[4];

    // One derived pointer a lane (a pointer a buffer was twenty live registers, past the
    // spill cliff: every operation then cost three times its price).
    device uint* const cb = big + car * CAR_WORDS;
    #define nodeState ((device float4*)(cb + O_NODE))
    #define recState ((device float4*)(cb + O_REC))
    #define groupsLast (cb + O_GROUPS)
    #define deform (cb + O_DEFORM)
    #define triDead (cb + O_TRIDEAD)
    #define acc ((device atomic_int*)(cb + O_ACC))
    #define shift ((device int*)(cb + O_SHIFT))
    #define misc (cb + O_MISC)
    #define pairs (cb + O_PAIRS)
    #define pBuild ((device float4*)(cb + O_PBUILD))
    #define clip ((device atomic_uint*)(cb + O_CLIP))
    #define hitBuf ((device float4*)(cb + O_HITS))
    #define stage ((device atomic_int*)(cb + O_SCRATCH))
    #define fPub ((device float*)(cb + O_SCRATCH + NN * 6))
    #define split ((device atomic_uint*)(cb + O_SCRATCH + NN * 9))
    #define masters ((device atomic_uint*)(cb + O_SCRATCH + NN * 10))
    #define cplV ((device float*)(cb + O_SCRATCH + NN * 11))
    #define hot ((device float4*)(cb + O_HOT))
    #define ent4 ((device const uint4*)(tab + T_ENT))

    // This lane's roles.
    const uint nd = tab[T_LANE + lid * 4];
    const uint entStart = tab[T_LANE + lid * 4 + 1];
    const uint entCounts = tab[T_LANE + lid * 4 + 2]; // leading scalar entries | all entries << 8
    const uint entScalar = entCounts & 255u, entCount = entCounts >> 8;
    const uint tri = tab[T_LANE + lid * 4 + 3];
    const bool isNode = nd != NONE;
    const bool isTri = tri != NONE;
    const uint ni = isNode ? nd : 0u;
    V3 F0 = v3(nodeState[ni * 3 + 2]);
    // (The node's mass in a register: three device loads a step otherwise. Friction and the
    // flags are read where a contact or an immovable node needs them.)
    const float massN = nodeStatic[ni].x;
    if (isNode)
    {
        float4 p = nodeState[ni * 3], v = nodeState[ni * 3 + 1];
        tgPos[ni * 3] = p.x; tgPos[ni * 3 + 1] = p.y; tgPos[ni * 3 + 2] = p.z;
        tgVel[ni * 3] = v.x; tgVel[ni * 3 + 1] = v.y; tgVel[ni * 3 + 2] = v.z;
        tgHeld[ni] = (atomic_load_explicit(&clip[ni * 4], memory_order_relaxed) != CLIP_EMPTY) ? 1 : 0;
    }
    // The slots' hot words: from the image the last dispatch left (six streamed loads), or
    // built from the records (the first dispatch after an upload).
    uint w0[RS];
    float up[RS];
#if REGMODE < 2
    float cc[RS];
#endif
#if REGMODE < 1
    float kk[RS], rest[RS];
#endif
    float G = 3.0e38f, L = 3.0e38f;
    const bool hotOk = misc[11] == 1u;
    if (hotOk)
    {
        uint4 hw = as_type<uint4>(hot[lid * 6 + 4]);
        float4 hg = hot[lid * 6 + 5];
        w0[0] = hw.x; w0[1] = hw.y; w0[2] = hw.z; w0[3] = hw.w;
        G = hg.x; L = hg.y;
        for (uint s = 0; s < RS; s++)
        {
            float4 hs = HOTS(s);
            up[s] = hs.w;
#if REGMODE < 2
            cc[s] = hs.y;
#endif
#if REGMODE < 1
            kk[s] = hs.x; rest[s] = hs.z;
#endif
        }
    }
    for (uint s = 0; s < RS && !hotOk; s++)
    {
        uint r = perm[s * LANES + lid];
        w0[s] = NONE;
        float4 hs = float4(0);
        up[s] = 0.0f;
#if REGMODE < 2
        cc[s] = 0.0f;
#endif
#if REGMODE < 1
        kk[s] = 0.0f; rest[s] = 0.0f;
#endif
        HOTS(s) = hs;
        if (r == NONE) continue;
        uint4 a = recA[r];
        uint kind = a.z & 0xFFu;
        float4 st0 = recState[r * 3], st1 = recState[r * 3 + 1], ax = recState[r * 3 + 2];
        if (kind == K_JPLAIN || kind == K_JSUPPORT)
        {
            w0[s] = a.x * 3u | (a.y * 3u) << 11 | (st1.w == 0.0f ? W_ALIVE : 0u) | (st1.z != 0.0f ? W_SLOPE : 0u) | (kind == K_JSUPPORT ? C_SUPPORT : C_PLAIN) << 30;
            float4 kc = recK[r];
            hs = float4(kc.x, kc.y, st0.x, ax.x);
            if (st1.w == 0.0f)
            {
                G = fmin(G, fmin(st0.w, st1.x));
                if (kind == K_JSUPPORT) L = fmin(L, recB[r].y);
            }
        }
        else if (kind == K_JATRI)
        {
            float4 e1 = recE[r * 2];
            w0[s] = a.x | a.y << 10 | as_type<uint>(e1.x) << 20 | C_AERO << 30;
            hs = float4(e1.y, e1.z, e1.w, 0.0f);
        }
        else
        {
            w0[s] = r | C_GENERIC << 30;
            if (kind == K_JCOUPLER)
            {
                // The velocities the last beam pass read (the centred correction's v0).
                uint ci = tab[T_VIDX + r] - CPL_BASE;
                float4 va = nodeState[a.x * 3 + 1], vb = nodeState[a.y * 3 + 1];
                cplV[ci * 6] = va.x; cplV[ci * 6 + 1] = va.y; cplV[ci * 6 + 2] = va.z;
                cplV[ci * 6 + 3] = vb.x; cplV[ci * 6 + 4] = vb.y; cplV[ci * 6 + 5] = vb.z;
            }
        }
        HOTS(s) = hs;
        up[s] = hs.w;
#if REGMODE < 2
        cc[s] = hs.y;
#endif
#if REGMODE < 1
        kk[s] = hs.x; rest[s] = hs.z;
#endif
    }
    if (lid < NG) atomic_store_explicit(&masters[lid], NONE, memory_order_relaxed);
    if (lid < 4) atomic_store_explicit(&tgCount[lid], 0u, memory_order_relaxed);
    if (lid == 0)
    {
        tgFlags[F_SHIFT] = misc[0]; tgFlags[F_STATUS] = misc[1]; tgFlags[F_REBUILD] = misc[3];
        tgFlags[F_HITS] = misc[4]; tgFlags[F_BUILDS] = misc[5]; tgFlags[F_MOST] = misc[6];
        tgFlags[F_REC] = misc[7]; tgFlags[F_HELD] = misc[8]; tgFlags[F_GDIRTY] = misc[9];
        tgFlags[F_FIRED] = 0u; tgFlags[F_BROKE] = misc[10];
        float4 pb = pBuild[REFNODE];
        tgPB[0] = pb.x; tgPB[1] = pb.y; tgPB[2] = pb.z;
    }
    BAR;

    for (uint frame = 0; frame < P.frames; frame++)
    for (uint step = 0; step < Z_STEPS; step++)
    {
        // ---- S0: the node pass ----
        const bool fShift = tgFlags[F_SHIFT] != 0u, fHits = tgFlags[F_HITS] != 0u;
        const bool fRoll = tgFlags[F_REC] != 0u || tgFlags[F_HELD] != 0u;
        // (F_GDIRTY: 1 the groups' words hold an older step's set, to clear; 2 they hold the
        // last step's already, written as the last dispatch ended.)
        const bool fFired = tgFlags[F_FIRED] != 0u, fDirty = tgFlags[F_GDIRTY] == 1u, fBroke = tgFlags[F_BROKE] != 0u;
        if (fFired)
        {
            // The groups the last beam pass set off (the torsion bars' watches read them).
            if (lid < NG)
            {
                groupsLast[lid] = atomic_load_explicit(&masters[lid], memory_order_relaxed) != NONE ? 1u : 0u;
                atomic_store_explicit(&masters[lid], NONE, memory_order_relaxed);
            }
        }
        else if (fDirty && lid < NG) groupsLast[lid] = 0u;
        if (fBroke && isTri && triDead[tri] == 0u)
        {
            // Triangle breaks (spec §18), on the steps after a beam broke.
            uint4 tb = triBreaks[tri];
            bool dead = false;
            for (uint q = 0; q < 4; q++)
            {
                uint j = tb[q];
                if (j != NONE && recState[j * 3 + 1].w != 0.0f) dead = true;
            }
            if (dead) triDead[tri] = 1u;
        }
        if (isNode && !SKIP(0))
        {
            const float mass = massN;
#if HAS_IMMOVABLE
            const uint nflags = as_type<uint>(nodeStatic[ni].z);
#else
            const uint nflags = 0u;
#endif
            V3 P0 = TGP(ni), V0 = TGV(ni);
            if (fRoll)
            {
                uint m2 = atomic_load_explicit(&clip[ni * 4 + 2], memory_order_relaxed);
                uint m3 = atomic_load_explicit(&clip[ni * 4 + 3], memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4], m2, memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4 + 1], m3, memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4 + 2], CLIP_EMPTY, memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4 + 3], CLIP_EMPTY, memory_order_relaxed);
                tgHeld[ni] = m2 != CLIP_EMPTY ? 1 : 0;
            }
            if (fHits)
            {
                for (uint q = 0; q < 6; q++) atomic_store_explicit(&stage[ni * 6 + q], 0, memory_order_relaxed);
                atomic_store_explicit(&split[ni], 0u, memory_order_relaxed);
                int ax_ = atomic_exchange_explicit(&acc[ni * 3], 0, memory_order_relaxed);
                int ay_ = atomic_exchange_explicit(&acc[ni * 3 + 1], 0, memory_order_relaxed);
                int az_ = atomic_exchange_explicit(&acc[ni * 3 + 2], 0, memory_order_relaxed);
                F0 = F0 + v3(float(ax_), float(ay_), float(az_)) * (1.0f / 32.0f);
            }
            else F0 = F0 + v3(0, 0, 0);
            V3 dshift = v3(0, 0, 0);
            if (fShift)
            {
                dshift = v3(float(shift[ni * 3]), float(shift[ni * 3 + 1]), float(shift[ni * 3 + 2])) * (1.0f / 16777216.0f);
                shift[ni * 3] = 0; shift[ni * 3 + 1] = 0; shift[ni * 3 + 2] = 0;
            }
            float lift = 0.0f;
            V3 x = v3(0, 0, 0) + P0;
            if (P.ground > x.y)
            {
                float budget;
                float depth = P.ground - x.y;
                V3 c = contactResponse(F0, V0, mass, v3(0, 1, 0), GROUND, depth, nodeStatic[ni].y, budget);
                if (!isZero3(c)) lift = shiftDepth(depth, budget, F0.y);
                F0 = F0 + c;
            }
#if NBOX > 0
            for (uint b = 0; b < NBOX; b++)
            {
                float3 lo = BOX_LO[b], hi = BOX_HI[b];
                if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
                {
                    V3 normal = v3(0.0f, 0.0f, -1.0f);
                    float dist = x.z - lo.z;
                    float dd[5] = {hi.z - x.z, x.x - lo.x, hi.x - x.x, x.y - lo.y, hi.y - x.y};
                    V3 nn[5] = {v3(0.0f, 0.0f, 1.0f), v3(-1.0f, 0.0f, 0.0f), v3(1.0f, 0.0f, 0.0f), v3(0.0f, -1.0f, 0.0f), v3(0.0f, 1.0f, 0.0f)};
                    for (uint q = 0; q < 5; q++)
                        if (dd[q] < dist) { dist = dd[q]; normal = nn[q]; }
                    float budget;
                    V3 c = contactResponse(F0, V0, mass, normal, BOXS, 0.0f, nodeStatic[ni].y, budget);
                    F0 = F0 + c;
                }
            }
#endif
            if ((nflags & NODE_IMMOVABLE) == 0u)
            {
                V0 = V0 + (F0 / mass) * H;
                P0 = P0 + V0 * H;
                if (fShift) P0 = P0 + dshift;
                if (lift != 0.0f) P0.y = P0.y + lift;
            }
            tgPos[ni * 3] = P0.x; tgPos[ni * 3 + 1] = P0.y; tgPos[ni * 3 + 2] = P0.z;
            tgVel[ni * 3] = V0.x; tgVel[ni * 3 + 1] = V0.y; tgVel[ni * 3 + 2] = V0.z;
        }
        if (lid == 0) atomic_store_explicit(&tgCount[0], 0u, memory_order_relaxed);
        if (lid == 2) atomic_store_explicit(&tgCount[2], 0u, memory_order_relaxed);
        BAR;
        if (lid == 0)
        {
            tgFlags[F_HELD] = tgFlags[F_REC]; tgFlags[F_REC] = 0u; tgFlags[F_HITS] = 0u; tgFlags[F_SHIFT] = 0u;
            tgFlags[F_GDIRTY] = (fFired || tgFlags[F_GDIRTY] == 2u) ? 1u : 0u; tgFlags[F_FIRED] = 0u; tgFlags[F_BROKE] = 0u;
        }

        // ---- S1: the centred damping's scalars (last beam pass's slopes) ----
        for (uint s = 0; s < RS && !SKIP(1); s++)
        {
            uint w = w0[s];
            // (The leading slots hold plain beams on every lane: no class to test there.)
            const bool sp = s < FULL_PLAIN_SLOTS;
            if (!sp && w == NONE) continue;
            const uint cls = sp ? C_PLAIN : (w >> 30);
            if (cls == C_AERO) continue;
            if (cls != C_GENERIC)
            {
                float gc = 0.0f;
                if ((w & W_SLOPE) != 0u)
                {
                    uint a = w & 2047u, b = (w >> 11) & 2047u;
                    V3 d = TG3P(a) - TG3P(b);
                    float inv = exactLength(dot3(d, d)).y;
                    float du = dot3(TG3V(a) - TG3V(b), d) * inv - up[s];
#if REGMODE < 2
                    gc = ((-0.5f * cc[s]) * du) * inv;
#else
                    gc = ((-0.5f * HOTS(s).y) * du) * inv;
#endif
                }
                tgG[s * LANES + lid] = gc;
                continue;
            }
            uint r = w & 0xFFFFFFu;
            uint4 a = recA[r];
            uint kind = a.z & 0xFFu;
            float slope = recState[r * 3 + 1].z;
            if (kind == K_JTORSION)
            {
                uint4 a2 = recA[r + 1];
                uint vi = tab[T_VIDX + r];
                uint ids[4] = {a.x, a.y, a2.x, a2.y};
                V3 o[4] = {v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0)};
                if (slope != 0.0f)
                {
                    float4 ax = recState[r * 3 + 2];
                    Jac J = dihedralJacobian(TGP(ids[0]), TGP(ids[1]), TGP(ids[2]), TGP(ids[3]));
                    float dr = (((dot3(J.g[0], TGV(ids[0])) - ax.x) + (dot3(J.g[1], TGV(ids[1])) - ax.y)) +
                                (dot3(J.g[2], TGV(ids[2])) - ax.z)) + (dot3(J.g[3], TGV(ids[3])) - ax.w);
                    float t = (-0.5f * slope) * dr;
                    for (uint q = 0; q < 4; q++) o[q] = J.g[q] * t;
                }
                for (uint q = 0; q < 4; q++) PUTV(vi + q, o[q]);
            }
            else if (kind == K_JCOUPLER)
            {
                uint vi = tab[T_VIDX + r];
                uint ci = vi - CPL_BASE;
                V3 push = v3(0, 0, 0);
                if (slope != 0.0f)
                {
                    V3 da = TGV(a.x) - v3(cplV[ci * 6], cplV[ci * 6 + 1], cplV[ci * 6 + 2]);
                    V3 db = TGV(a.y) - v3(cplV[ci * 6 + 3], cplV[ci * 6 + 4], cplV[ci * 6 + 5]);
                    push = (da - db) * (-0.5f * slope);
                }
                PUTV(vi, push);
            }
            else
            {
                float gc = 0.0f;
                if (slope != 0.0f)
                {
                    V3 d = TGP(a.x) - TGP(a.y);
                    float inv = exactLength(dot3(d, d)).y;
                    float du = dot3(TGV(a.x) - TGV(a.y), d) * inv - recState[r * 3 + 2].x;
                    gc = ((-0.5f * slope) * du) * inv;
                }
                tgG[s * LANES + lid] = gc;
            }
        }
        BAR;

        // ---- C1: the centred correction gathered per node (kept in registers) ----
        V3 Pn = v3(0, 0, 0), Vn = v3(0, 0, 0);
        if (isNode && !SKIP(2))
        {
            V3 Ps = TGP(ni);
            V3 c = v3(0, 0, 0);
            for (uint k = 0; k < MAXENT; k += 4u)
            {
                // The leading scalar entries, four a load: (x_self − x_other)·g.
                if (k >= entScalar) break;
                uint4 cur = ent4[(entStart >> 2) + (k >> 2)];
                ENT_ADD(c, cur.x);
                if (k + 1u < entScalar) ENT_ADD(c, cur.y);
                if (k + 2u < entScalar) ENT_ADD(c, cur.z);
                if (k + 3u < entScalar) ENT_ADD(c, cur.w);
            }
            for (uint k = entScalar; k < MAXENT; k++)
            {
                if (k >= entCount) break;
                uint e = ent4[(entStart >> 2) + (k >> 2)][k & 3u];
                if ((e >> 31) == 0u)
                {
                    uint o3 = e & 2047u;
                    c = c + (Ps - v3(tgPos[o3], tgPos[o3 + 1u], tgPos[o3 + 2u])) * tgG[e >> 11];
                }
                else if ((e & (1u << 29)) == 0u)
                {
                    uint vi = e & 0xFFFu;
                    V3 vv = v3(tgVec[vi * 3], tgVec[vi * 3 + 1], tgVec[vi * 3 + 2]);
                    c = c + ((e & (1u << 30)) != 0u ? -vv : vv);
                }
            }
            Pn = Ps; Vn = TGV(ni);
#if HAS_IMMOVABLE
            if ((as_type<uint>(nodeStatic[ni].z) & NODE_IMMOVABLE) == 0u)
#endif
            {
                V3 dv = (c / massN) * H;
                Vn = Vn + dv;
                Pn = Pn + dv * H;
            }
            atomic_fetch_max_explicit(&tgCount[0], as_type<uint>(len3(Vn)), memory_order_relaxed);
        }
        BAR;
        // ---- C2: published ----
        if (isNode && !SKIP(2))
        {
            tgPos[ni * 3] = Pn.x; tgPos[ni * 3 + 1] = Pn.y; tgPos[ni * 3 + 2] = Pn.z;
            tgVel[ni * 3] = Vn.x; tgVel[ni * 3 + 1] = Vn.y; tgVel[ni * 3 + 2] = Vn.z;
        }
        BAR;

        // ---- S3: the beam pass ----
        const float vmax = as_type<float>(atomic_load_explicit(&tgCount[0], memory_order_relaxed));
        for (uint s = 0; s < RS && !SKIP(3); s++)
        {
            uint w = w0[s];
            // (The leading slots hold plain beams on every lane: no class to test there.)
            const bool sp = s < FULL_PLAIN_SLOTS;
            if (!sp && w == NONE) continue;
            const uint cls = sp ? C_PLAIN : (w >> 30);
            if (cls == C_AERO)
            {
                uint a = w & 1023u, b = (w >> 10) & 1023u, c = (w >> 20) & 1023u;
                float4 ha = HOTS(s);
                V3 f = aeroTri(TGP(a), TGP(b), TGP(c), TGV(a), TGV(b), TGV(c), ha.x, ha.y, ha.z);
                PUTV(lid - AERO_LO, f);
                continue;
            }
            if (cls != C_GENERIC)
            {
                if ((w & W_ALIVE) == 0u)
                {
                    tgG[s * LANES + lid] = 0.0f;
                    w0[s] = w & ~W_SLOPE;
                    continue;
                }
                uint a = w & 2047u, b = (w >> 11) & 2047u;
                V3 pa = TG3P(a), pb = TG3P(b), va = TG3V(a), vb = TG3V(b);
                V3 d = pa - pb;
                float2 li = exactLength(dot3(d, d));
#if REGMODE >= 1
                float4 hs = HOTS(s);
                const float kS = hs.x, rS = hs.z;
#else
                const float kS = kk[s], rS = rest[s];
#endif
#if REGMODE >= 2
                const float cS = hs.y;
#else
                const float cS = cc[s];
#endif
                float e = li.x - rS;
                float u = dot3(va - vb, d) * li.y;
                float k = kS;
                float damp = cS * u;
                bool slow = false, slopeOn = true;
                if (cls == C_SUPPORT && e > 0.0f)
                {
                    slow = li.x > L;
                    k = 0.0f; damp = 0.0f; slopeOn = false;
                }
                float f = -k * e - damp;
                if (fabs(f) > G) slow = true;
                if (slow)
                {
                    // The step it may yield or break: the whole rule from its records.
                    Slow o = slowScalar(perm[s * LANES + lid], recA, recK, recB, recE, recState, masters, pa, pb, va, vb);
#if REGMODE < 1
                    rest[s] = o.rest;
#else
                    hs.z = o.rest;
                    HOTS(s) = hs;
#endif
                    up[s] = o.u;
                    w0[s] = (w & ~(W_ALIVE | W_SLOPE | W_DEFORM)) | ((o.bits & 1u) != 0u ? W_ALIVE : 0u) | ((o.bits & 2u) != 0u ? W_SLOPE : 0u) | ((o.bits & 4u) != 0u ? W_DEFORM : 0u);
                    if ((o.bits & 8u) != 0u) tgFlags[F_FIRED] = 1u;
                    if ((o.bits & 16u) != 0u) tgFlags[F_BROKE] = 1u;
                    tgG[s * LANES + lid] = o.g;
                    continue;
                }
                up[s] = u;
                w0[s] = slopeOn ? (w | W_SLOPE) : (w & ~W_SLOPE);
                tgG[s * LANES + lid] = f * li.y;
                continue;
            }
            // The car's few special elements, from their records every step.
            uint r = w & 0xFFFFFFu;
            uint4 a = recA[r];
            uint kind = a.z & 0xFFu;
            int group = as_type<int>(a.w);
            V3 pa = TGP(a.x), pb = TGP(a.y), va = TGV(a.x), vb = TGV(a.y);
            if (kind == K_JTORSION)
            {
                uint vi = tab[T_VIDX + r];
                float4 s0 = recState[r * 3], s1 = recState[r * 3 + 1];
                if (s1.w != 0.0f)
                {
                    for (uint q = 0; q < 4; q++) PUTV(vi + q, v3(0, 0, 0));
                    if (s1.z != 0.0f) { s1.z = 0.0f; recState[r * 3 + 1] = s1; }
                    continue;
                }
                float4 kc = recK[r];
                uint4 a2 = recA[r + 1];
                V3 x[4] = {pa, pb, TGP(a2.x), TGP(a2.y)};
                V3 v[4] = {va, vb, TGV(a2.x), TGV(a2.y)};
                float4 we = recE[r * 2];
                uint3 wt = uint3(as_type<uint>(we.x), as_type<uint>(we.y), as_type<uint>(we.z));
                bool watch = (wt.x != 0u && groupsLast[wt.x] != 0u) || (wt.y != 0u && groupsLast[wt.y] != 0u) ||
                             (wt.z != 0u && groupsLast[wt.z] != 0u);
                V3 o[4] = {v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0)};
                bool broke = false;
                Jac J = dihedralJacobian(x[0], x[1], x[2], x[3]);
                float gv[4];
                for (uint q = 0; q < 4; q++) gv[q] = dot3(J.g[q], v[q]);
                if (watch) { broke = true; s1.y = 0.0f; }
                else
                {
                    float phi = dihedral(x[0], x[1], x[2], x[3]);
                    float rate = ((gv[0] + gv[1]) + gv[2]) + gv[3];
                    float dd = wrapAngle(phi - s0.x);
                    if (kc.x * fabs(dd) > s0.y)
                    {
                        dd = copysign(s0.y / kc.x, dd);
                        s0.x = wrapAngle(phi - dd);
                    }
                    float t = -kc.x * dd - kc.y * rate;
                    s1.y = t;
                    if (fabs(t) > s1.x) broke = true;
                    else for (uint q = 0; q < 4; q++) o[q] = J.g[q] * t;
                }
                if (broke) { s1.w = 1.0f; recState[(r + 1) * 3 + 1].w = 1.0f; tgFlags[F_BROKE] = 1u; }
                s1.z = broke ? 0.0f : kc.y;
                recState[r * 3] = s0; recState[r * 3 + 1] = s1;
                recState[r * 3 + 2] = float4(gv[0], gv[1], gv[2], gv[3]);
                for (uint q = 0; q < 4; q++) PUTV(vi + q, o[q]);
                if (broke && group > 0) { atomic_fetch_min_explicit(&masters[group], r, memory_order_relaxed); tgFlags[F_FIRED] = 1u; }
            }
            else if (kind == K_JCOUPLER)
            {
                uint vi = tab[T_VIDX + r];
                uint ci = vi - CPL_BASE;
                float4 s1 = recState[r * 3 + 1];
                cplV[ci * 6] = va.x; cplV[ci * 6 + 1] = va.y; cplV[ci * 6 + 2] = va.z;
                cplV[ci * 6 + 3] = vb.x; cplV[ci * 6 + 4] = vb.y; cplV[ci * 6 + 5] = vb.z;
                if (s1.w != 0.0f)
                {
                    PUTV(vi, v3(0, 0, 0));
                    if (s1.z != 0.0f) { s1.z = 0.0f; recState[r * 3 + 1] = s1; }
                    continue;
                }
                float4 kc = recK[r];
                V3 f = (-kc.x) * (pa - pb) - kc.y * (va - vb);
                float m = len3(f);
                s1.y = m;
                bool broke = m > s1.x;
                if (broke) { s1.w = 1.0f; tgFlags[F_BROKE] = 1u; }
                s1.z = broke ? 0.0f : kc.y;
                recState[r * 3 + 1] = s1;
                PUTV(vi, broke ? v3(0, 0, 0) : f);
                if (broke && group > 0) { atomic_fetch_min_explicit(&masters[group], r, memory_order_relaxed); tgFlags[F_FIRED] = 1u; }
            }
            else
            {
                float4 s1 = recState[r * 3 + 1];
                if (s1.w != 0.0f)
                {
                    tgG[s * LANES + lid] = 0.0f;
                    if (s1.z != 0.0f) { s1.z = 0.0f; recState[r * 3 + 1] = s1; }
                    continue;
                }
                Slow o = slowScalar(r, recA, recK, recB, recE, recState, masters, pa, pb, va, vb);
                if ((o.bits & 4u) != 0u) w0[s] = w | W_DEFORM;
                if ((o.bits & 8u) != 0u) tgFlags[F_FIRED] = 1u;
                if ((o.bits & 16u) != 0u) tgFlags[F_BROKE] = 1u;
                tgG[s * LANES + lid] = o.g;
            }
        }
        // Whether the pair list still holds every pair the box test can pass: each node's
        // move since the build, less the reference node's, at most D an axis; the list's
        // box covers the reach now and 2·D.
        if (isNode && !SKIP(6))
        {
            float4 pb = pBuild[ni];
            V3 rel = (TGP(ni) - v3(pb)) - (TGP(REFNODE) - v3(tgPB[0], tgPB[1], tgPB[2]));
            float D = fmax(fmax(fabs(rel.x), fabs(rel.y)), fabs(rel.z));
            float need = ((CR + 2.0f * vmax * H) + CLIP_DEPTH) + 2.0f * D + LIST_EPS;
            if (!(need <= LIST_REACH)) tgFlags[F_REBUILD] = 1u;
        }
        BAR;

        // ---- break groups set off in this pass (rare) ----
        if (tgFlags[F_FIRED] != 0u)
        {
            for (uint s = 0; s < RS; s++)
            {
                uint w = w0[s];
                if (w == NONE || (w >> 30) == C_AERO) continue;
                uint cls = w >> 30;
                uint r = cls == C_GENERIC ? (w & 0xFFFFFFu) : perm[s * LANES + lid];
                uint4 a = recA[r];
                int group = as_type<int>(a.w);
                if (group == 0) continue;
                uint m = atomic_load_explicit(&masters[abs(group)], memory_order_relaxed);
                if (m == NONE) continue;
                // cascade(): every member broken; one after the first to break in this pass
                // was skipped (no force, no slope, no deform of its own).
                uint kind = a.z & 0xFFu;
                bool ext = ((a.z >> 16) & 1u) != 0u;
                float4 s1 = recState[r * 3 + 1];
                if (r > m)
                {
                    s1.z = 0.0f;
                    w = w & ~(W_SLOPE | W_DEFORM);
                    if (kind == K_JTORSION) { uint vi = tab[T_VIDX + r]; for (uint q = 0; q < 4; q++) PUTV(vi + q, v3(0, 0, 0)); }
                    else if (kind == K_JCOUPLER) { PUTV(tab[T_VIDX + r], v3(0, 0, 0)); }
                    else tgG[s * LANES + lid] = 0.0f;
                }
                s1.w = 1.0f;
                recState[r * 3 + 1] = s1;
                if (kind == K_JTORSION) recState[(r + 1) * 3 + 1].w = 1.0f;
                w = w & ~W_ALIVE;
                // cascade_ext(): a member with an extension record triggers its deform groups.
                if (ext && (as_type<uint>(recB[r].w) & EXT_DEFORM) != 0u) w |= W_DEFORM;
                w0[s] = w;
            }
            if (lid == 0) tgFlags[F_BROKE] = 1u;
            BAR;
        }
        // ---- the pair list built (rare) ----
        if (tgFlags[F_REBUILD] != 0u)
        {
            if (isTri)
            {
                uint4 tn = tris[tri];
                V3 p0 = v3(0, 0, 0) + TGP(tn.x), p1 = v3(0, 0, 0) + TGP(tn.y), p2 = v3(0, 0, 0) + TGP(tn.z);
                V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - LIST_REACH, rmin(rmin(p0.y, p1.y), p2.y) - LIST_REACH, rmin(rmin(p0.z, p1.z), p2.z) - LIST_REACH);
                V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + LIST_REACH, rmax(rmax(p0.y, p1.y), p2.y) + LIST_REACH, rmax(rmax(p0.z, p1.z), p2.z) + LIST_REACH);
                uint n = 0u;
                if (triDead[tri] == 0u)
                {
                    uint bitsw = 0u;
                    for (uint c = 0; c < NC; c++)
                    {
                        if ((c & 31u) == 0u) bitsw = allowed[tri * NCW + (c >> 5)];
                        if ((bitsw & (1u << (c & 31u))) == 0u) continue;
                        uint h = cands[c];
                        V3 p = v3(0, 0, 0) + TGP(h);
                        if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) continue;
                        if (n < PROW - 2u) pairs[tri * PROW + 2u + n] = h;
                        n++;
                    }
                }
                if (n > PROW - 2u) { tgFlags[F_STATUS] = tgFlags[F_STATUS] | 1u; n = PROW - 2u; }
                pairs[tri * PROW] = n;
                pairs[tri * PROW + 1u] = tn.x | tn.y << 10 | tn.z << 20;
            }
            if (isNode) pBuild[ni] = f4(TGP(ni), 0);
            if (lid == 0) { V3 pr = TGP(REFNODE); tgPB[0] = pr.x; tgPB[1] = pr.y; tgPB[2] = pr.z; }
            BAR;
            if (lid == 0) { tgFlags[F_REBUILD] = 0u; tgFlags[F_BUILDS] += 1u; }
        }

        // ---- S4: the beam pass gathered per node; the triangles' listed pairs ----
        if (isNode && !SKIP(4))
        {
            V3 Ps = TGP(ni);
            V3 f = v3(0.0f, massN * GRAVITY, 0.0f);
            // (A broken beam's scalar is zero: adding ±0 changes no bit of a sum that began at
            // +0 or at the weight, so it is the reference's skipped add.)
            for (uint k = 0; k < MAXENT; k += 4u)
            {
                if (k >= entScalar) break;
                uint4 cur = ent4[(entStart >> 2) + (k >> 2)];
                ENT_ADD(f, cur.x);
                if (k + 1u < entScalar) ENT_ADD(f, cur.y);
                if (k + 2u < entScalar) ENT_ADD(f, cur.z);
                if (k + 3u < entScalar) ENT_ADD(f, cur.w);
            }
            for (uint k = entScalar; k < MAXENT; k++)
            {
                if (k >= entCount) break;
                uint e = ent4[(entStart >> 2) + (k >> 2)][k & 3u];
                if ((e >> 31) == 0u)
                {
                    uint o3 = e & 2047u;
                    f = f + (Ps - v3(tgPos[o3], tgPos[o3 + 1u], tgPos[o3 + 2u])) * tgG[e >> 11];
                }
                else
                {
                    uint vi = e & 0xFFFu;
                    V3 vv = v3(tgVec[vi * 3], tgVec[vi * 3 + 1], tgVec[vi * 3 + 2]);
                    f = f + ((e & (1u << 30)) != 0u ? -vv : vv);
                }
            }
            F0 = f;
        }
        for (uint s = 0; s < RS; s++)
        {
            // Deform groups asked for in this pass (spec §19).
            uint w = w0[s];
            if (w != NONE && (w >> 30) != C_AERO && (w & W_DEFORM) != 0u)
            {
                uint r = (w >> 30) == C_GENERIC ? (w & 0xFFFFFFu) : perm[s * LANES + lid];
                uint emask = as_type<uint>(recB[r].w);
                uint d0 = (emask >> 8) & 0xFFu, d1 = (emask >> 16) & 0xFFu;
                if (d0 != 0u) deform[d0 - 1u] = 1u;
                if (d1 != 0u) deform[d1 - 1u] = 1u;
                w0[s] = w & ~W_DEFORM;
            }
        }
#if COLLIDE
        if (isTri && !SKIP(5))
        {
            // (The row's first load: its count, its triangle's nodes, its first two pairs.)
            device const uint4* row4 = (device const uint4*)(pairs + tri * PROW);
            uint4 cur = row4[0];
            const uint n = min(cur.x, (uint)(PROW - 2u));
            if (n != 0u)
            {
                const uint t0 = cur.y & 1023u, t1 = (cur.y >> 10) & 1023u, t2 = (cur.y >> 20) & 1023u;
                // (Only what the filter reads stays live over the row: two points and two
                // plane normals; a pair that passes rebuilds the rest, the same values. No
                // arrays: an array indexed in a loop is stack memory.)
                V3 f0, frd, b0, brd;
                bool fok, bok;
                {
                    V3 q0 = v3(0, 0, 0) + TGP(t0), q1 = v3(0, 0, 0) + TGP(t1), q2 = v3(0, 0, 0) + TGP(t2);
                    Frame F = triFrame(q0, q1, q2);
                    Frame B = triFrame(q0 - TGV(t0) * H, q1 - TGV(t1) * H, q2 - TGV(t2) * H);
                    f0 = F.p0; frd = F.rd; fok = F.ok; b0 = B.p0; brd = B.rd; bok = B.ok;
                }
                for (uint k = 0; k < PROW - 2u; k++)
                {
                    if (k >= n) break;
                    uint kk1 = k + 2u;
                    if ((kk1 & 3u) == 0u) cur = row4[kk1 >> 2];
                    uint h = cur[kk1 & 3u];
                    V3 ph = v3(0, 0, 0) + TGP(h);
                    V3 vh = TGV(h);
                    float d1 = fok ? dot3(frd, ph - f0) : 0.0f;
                    float d0 = bok ? dot3(brd, (ph - vh * H) - b0) : 0.0f;
                    bool cand = tgHeld[h] != 0 || (d0 > 0.0f && d1 < 0.0f) || (d0 < 0.0f && d1 > 0.0f) || fabs(d1) <= CR;
                    if (!cand) continue;
                    // The full rule (explicit_beams.rs collision_pass, clip_hit).
                    const uint4 tn = uint4(t0, t1, t2, 0u);
                    const V3 pA = v3(0, 0, 0) + TGP(t0), pB = v3(0, 0, 0) + TGP(t1), pC = v3(0, 0, 0) + TGP(t2);
                    const V3 vA = TGV(t0), vB = TGV(t1), vC = TGV(t2);
                    Frame F = triFrame(pA, pB, pC);
                    Frame B = triFrame(pA - vA * H, pB - vB * H, pC - vC * H);
                    float tv = rmax(rmax(rmax(0.0f, len3(vA)), len3(vB)), len3(vC));
                    float reach = (CR + (tv + vmax) * H) + CLIP_DEPTH;
                    V3 lo = v3(rmin(rmin(pA.x, pB.x), pC.x) - reach, rmin(rmin(pA.y, pB.y), pC.y) - reach, rmin(rmin(pA.z, pB.z), pC.z) - reach);
                    V3 hi = v3(rmax(rmax(pA.x, pB.x), pC.x) + reach, rmax(rmax(pA.y, pB.y), pC.y) + reach, rmax(rmax(pA.z, pB.z), pC.z) + reach);
                    if (ph.x < lo.x || ph.y < lo.y || ph.z < lo.z || ph.x > hi.x || ph.y > hi.y || ph.z > hi.z) continue;
                    float4 loc = frameLocal(F.p0, F.ra, F.rb, F.rd, F.ok, ph);
                    uint key = CLIP_SELF << 20 | tri;
                    float side = 0.0f;
                    uint c0 = atomic_load_explicit(&clip[h * 4], memory_order_relaxed);
                    uint c1 = atomic_load_explicit(&clip[h * 4 + 1], memory_order_relaxed);
                    if (c0 != CLIP_EMPTY && (c0 >> 1) == key) side = (c0 & 1u) != 0u ? 1.0f : -1.0f;
                    else if (c1 != CLIP_EMPTY && (c1 >> 1) == key) side = (c1 & 1u) != 0u ? 1.0f : -1.0f;
                    bool over = loc.x >= 0.0f && loc.y >= 0.0f && loc.z >= 0.0f;
                    bool haveW = false;
                    float3 wgt = float3(0);
                    if (side != 0.0f)
                    {
                        haveW = over;
                        wgt = float3(loc.x, loc.y, loc.z);
                    }
                    else
                    {
                        float4 l0 = frameLocal(B.p0, B.ra, B.rb, B.rd, B.ok, ph - vh * H);
                        float e0 = l0.w, e1 = loc.w;
                        bool crossed = false;
                        if ((e0 > 0.0f && e1 < 0.0f) || (e0 < 0.0f && e1 > 0.0f))
                        {
                            float sx = e0 / (e0 - e1);
                            float aa = l0.x + (loc.x - l0.x) * sx;
                            float bb2 = l0.y + (loc.y - l0.y) * sx;
                            float3 cw = float3(aa, bb2, (1.0f - aa) - bb2);
                            if (cw.x >= 0.0f && cw.y >= 0.0f && cw.z >= 0.0f)
                            {
                                crossed = true;
                                side = e0 > 0.0f ? 1.0f : -1.0f;
                                haveW = true;
                                wgt = cw;
                            }
                        }
                        if (!crossed)
                        {
                            if (over && fabs(loc.w) <= CR)
                            {
                                side = loc.w < 0.0f ? -1.0f : 1.0f;
                                haveW = true;
                                wgt = float3(loc.x, loc.y, loc.z);
                            }
                            else continue;
                        }
                    }
                    float dd = side * loc.w;
                    if (!(haveW && dd <= CR && dd >= -CLIP_DEPTH)) continue;
                    uint word = CLIP_SELF << 21 | tri << 1 | (side > 0.0f ? 1u : 0u);
                    uint old = atomic_fetch_min_explicit(&clip[h * 4 + 2], word, memory_order_relaxed);
                    atomic_fetch_min_explicit(&clip[h * 4 + 3], max(old, word), memory_order_relaxed);
                    tgFlags[F_REC] = 1u;
                    float pen = rmin(CR - dd, CR);
                    V3 nn = F.n * side;
                    uint i = atomic_fetch_add_explicit(&tgCount[2], 1u, memory_order_relaxed);
                    if (i < HIT_CAP)
                    {
                        hitBuf[i * HITW] = as_type<float4>(uint4(h, tn.x, tn.y, tn.z));
                        hitBuf[i * HITW + 1] = float4(wgt.x, wgt.y, wgt.z, pen);
                        hitBuf[i * HITW + 2] = f4(nn, 0);
                    }
                    atomic_fetch_add_explicit(&split[h], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&split[tn.x], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&split[tn.y], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&split[tn.z], 1u, memory_order_relaxed);
                }
            }
        }
#endif
        BAR;
#if COLLIDE
        const uint nhits = atomic_load_explicit(&tgCount[2], memory_order_relaxed);
        if (nhits != 0u && !SKIP(8))
        {
            // ---- the step's hits: the forces published, then the two-body rule's passes ----
            if (isNode) { fPub[ni * 3] = F0.x; fPub[ni * 3 + 1] = F0.y; fPub[ni * 3 + 2] = F0.z; }
            if (lid == 0)
            {
                tgFlags[F_HITS] = 1u;
                tgFlags[F_MOST] = max(tgFlags[F_MOST], nhits);
                if (nhits > HIT_CAP) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 2u;
            }
            BAR;
            const uint nh = min(nhits, (uint)HIT_CAP);
            #define NP(i) TGP(i)
            #define NV(i) TGV(i)
            #define NF(i) (v3(fPub[(i) * 3], fPub[(i) * 3 + 1], fPub[(i) * 3 + 2]) + v3(0, 0, 0))
            #define NM(i) nodeStatic[i].x
            #define NIMM(i) ((as_type<uint>(nodeStatic[i].z) & NODE_IMMOVABLE) != 0u)
            #define ACC(f, i) (NIMM(i) ? v3(0, 0, 0) : (f) / NM(i))
            #define IM(i, kk_) (NIMM(i) ? 0.0f : float(kk_) / NM(i))
            #define SPL(i) atomic_load_explicit(&split[i], memory_order_relaxed)
            for (uint k = lid; k < nh; k += LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z, depth = wp.w;
                V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                float imh = IM(hn, SPL(hn));
                float imt = tbInvMass(IM(i1, SPL(i1)), IM(i2, SPL(i2)), IM(i0, SPL(i0)), alpha, beta, gamma);
                float inv = imh + imt;
                float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                V3 at = (ACC(NF(i1), i1) * alpha + ACC(NF(i2), i2) * beta) + ACC(NF(i0), i0) * gamma;
                V3 fh = (ACC(NF(hn), hn) - at) * m;
                float budget;
                V3 f = contactResponse(fh, v, m, n, BOXS, depth, nodeStatic[hn].y, budget);
                if (!isZero3(f))
                {
                    float delta = shiftDepth(depth, budget, dot3(fh, n));
                    if (delta > 0.0f)
                    {
                        float q = delta * m;
                        V3 dh = n * (q * IM(hn, 1u));
                        for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[hn * 3 + a], toFixedShift(a == 0 ? dh.x : (a == 1 ? dh.y : dh.z)), memory_order_relaxed);
                        uint ti[3] = {i1, i2, i0};
                        float tw[3] = {alpha, beta, gamma};
                        for (uint e = 0; e < 3; e++)
                        {
                            if (tw[e] > 0.0f)
                            {
                                V3 dt = n * -(tw[e] * (q * IM(ti[e], 1u)));
                                for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[ti[e] * 3 + a], toFixedShift(a == 0 ? dt.x : (a == 1 ? dt.y : dt.z)), memory_order_relaxed);
                            }
                        }
                        tgFlags[F_SHIFT] = 1u;
                    }
                }
                hitBuf[k * HITW + 3] = f4(f, 0);
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&stage[ns[e] * 6], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&stage[ns[e] * 6 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&stage[ns[e] * 6 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            BAR;
            #define NOW1(i) (v3(fPub[(i) * 3], fPub[(i) * 3 + 1], fPub[(i) * 3 + 2]) + v3(float(atomic_load_explicit(&stage[(i) * 6], memory_order_relaxed)), float(atomic_load_explicit(&stage[(i) * 6 + 1], memory_order_relaxed)), float(atomic_load_explicit(&stage[(i) * 6 + 2], memory_order_relaxed))) * (1.0f / 32.0f))
            for (uint k = lid; k < nh; k += LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z;
                V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                V3 at = (ACC(NOW1(i1), i1) * alpha + ACC(NOW1(i2), i2) * beta) + ACC(NOW1(i0), i0) * gamma;
                float a = dot3(ACC(NOW1(hn), hn) - at, n);
                float vn = dot3(v, n);
                float x = vn + rmax(vn, 0.0f);
                float r = 0.0f;
                if (vn < 0.0f && a * H + x < 0.0f)
                {
                    float inv = IM(hn, SPL(hn)) + tbInvMass(IM(i1, SPL(i1)), IM(i2, SPL(i2)), IM(i0, SPL(i0)), alpha, beta, gamma);
                    if (inv > 0.0f)
                    {
                        float m = 1.0f / inv;
                        float rr = -(a * m) - (x * m) / H;
                        if (rr > 0.0f) r = rr;
                    }
                }
                float4 h3 = hitBuf[k * HITW + 3];
                hitBuf[k * HITW + 3] = float4(h3.x, h3.y, h3.z, r);
                if (r > 0.0f)
                {
                    V3 f = n * r;
                    V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                    uint ns[4] = {hn, i1, i2, i0};
                    for (uint e = 0; e < 4; e++)
                    {
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 3], toFixed(fs[e].x), memory_order_relaxed);
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 4], toFixed(fs[e].y), memory_order_relaxed);
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 5], toFixed(fs[e].z), memory_order_relaxed);
                    }
                }
            }
            BAR;
            #define SUM2(i, a) (atomic_load_explicit(&stage[(i) * 6 + (a)], memory_order_relaxed) + atomic_load_explicit(&stage[(i) * 6 + 3 + (a)], memory_order_relaxed))
            #define NOW2(i) (v3(fPub[(i) * 3], fPub[(i) * 3 + 1], fPub[(i) * 3 + 2]) + v3(float(SUM2(i, 0)), float(SUM2(i, 1)), float(SUM2(i, 2))) * (1.0f / 32.0f))
            for (uint k = lid; k < nh; k += LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                float4 h3 = hitBuf[k * HITW + 3];
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z;
                V3 t = v3(h3);
                if (h3.w > 0.0f) t = t + n * h3.w;
                V3 f = t;
                if (!isZero3(t))
                {
                    V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                    V3 a = ACC(NOW2(hn), hn) - ((ACC(NOW2(i1), i1) * alpha + ACC(NOW2(i2), i2) * beta) + ACC(NOW2(i0), i0) * gamma);
                    float inv = IM(hn, 1u) + tbInvMass(IM(i1, 1u), IM(i2, 1u), IM(i0, 1u), alpha, beta, gamma);
                    V3 aRest = a - t * inv;
                    float aa = dot3(t, (v + v) + aRest * H);
                    float bb = (H * dot3(t, t)) * inv;
                    if (aa + bb > 0.0f && bb > 0.0f)
                    {
                        float sc = aa < 0.0f ? rmin(-aa / bb, 1.0f) : 0.0f;
                        f = t + t * -(1.0f - sc);
                    }
                }
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&acc[ns[e] * 3], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            BAR;
        }
#endif
    }

    // ---- write back ----
    if (isNode)
    {
        nodeState[ni * 3] = f4(TGP(ni), 0);
        nodeState[ni * 3 + 1] = f4(TGV(ni), 0);
        nodeState[ni * 3 + 2] = f4(F0, 0);
    }
    hot[lid * 6 + 4] = as_type<float4>(uint4(w0[0], w0[1], w0[2], w0[3]));
    hot[lid * 6 + 5] = float4(G, L, 0.0f, 0.0f);
    for (uint s = 0; s < RS; s++)
    {
        float4 hs = HOTS(s);
        hs.w = up[s];
#if REGMODE < 1
        hs.x = kk[s]; hs.z = rest[s];
#endif
        HOTS(s) = hs;
        // The records' copies of the two words the hot path keeps (the last extension rate
        // and the damping slope), when the host asks to read the beams (P.pad2 bit 0).
        uint w = w0[s];
        if ((P.pad2 & 1u) == 0u || w == NONE || (w >> 30) >= C_AERO) continue;
        uint r = perm[s * LANES + lid];
        recState[r * 3 + 2] = float4(up[s], 0.0f, 0.0f, 0.0f);
        float4 s1 = recState[r * 3 + 1];
        s1.z = (w & W_SLOPE) != 0u ? hs.y : 0.0f;
        recState[r * 3 + 1] = s1;
    }
    if (lid < NG)
    {
        // (As the next dispatch's first node pass would find them.)
        if (tgFlags[F_FIRED] != 0u) groupsLast[lid] = atomic_load_explicit(&masters[lid], memory_order_relaxed) != NONE ? 1u : 0u;
        else if (tgFlags[F_GDIRTY] == 1u) groupsLast[lid] = 0u;
    }
    if (lid == 0)
    {
        misc[0] = tgFlags[F_SHIFT]; misc[1] = tgFlags[F_STATUS]; misc[3] = tgFlags[F_REBUILD];
        misc[4] = tgFlags[F_HITS]; misc[5] = tgFlags[F_BUILDS]; misc[6] = tgFlags[F_MOST];
        misc[7] = tgFlags[F_REC]; misc[8] = tgFlags[F_HELD]; misc[9] = tgFlags[F_FIRED] != 0u ? 2u : (tgFlags[F_GDIRTY] == 2u ? 2u : 0u); misc[10] = tgFlags[F_BROKE];
        misc[15] = 2u;
        misc[11] = 1u;
        uint np = 0u;
        misc[2] = np;
    }
}
