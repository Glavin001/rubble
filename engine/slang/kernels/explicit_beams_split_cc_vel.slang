// Velocity contact (the CPU's crates/engine-softbody/src/explicit_contact/ccd_vel.rs, Dat::velocity;
// docs/continuous-contact-design.md, "Velocity contact on the GPU"): the CCD's pairs that collide
// over the step's paths become rows on the velocities, Σ wⱼ n·vⱼ ≥ −b, solved by APGD with the
// Gershgorin preconditioner; positions follow the velocity change. Operation for operation in
// strict float as the CPU; the impulses summed per node in 64-bit fixed point as two 32-bit words
// (the low word's atomic add returns the old value and its carry goes into the high word: order
// free and exact). Its first port: no warm starts and no relisting (the CPU's Dat::warm_starts and
// Dat::relist off), no rigid spheres as bodies.

static const uint CC_F_ROW = 0x2000u;
// ccd_rows.rs ROW_GAP_SHARE and ROW_ITERS; ccd_vel.rs PUSH_TOL, push_share, ROUND_ULPS,
// APGD_RESTART, GERSH_FIXED, relin_share (1 − 2·ROW_GAP_SHARE, its f32 bits).
static const float CCV_GAP_SHARE = 0.01;
static const uint CCV_ITERS = 32u;
static const float CCV_PUSH_TOL = 1.0e-5;
static const float CCV_PUSH_SHARE = 1.0e-5;
static const float CCV_ROUND_ULPS = 4.0;
static const uint CCV_RESTART = 16u;
static const float CCV_GERSH_FIXED = 65536.0;
static const uint CCV_RELIN_BITS = 0x3f7ae148u;
static const uint CCV_BETA[15] = { 0x00000000u, 0x3e9041ffu, 0x3ede3adcu, 0x3f07f3ccu, 0x3f19498eu, 0x3f261fd6u,
    0x3f30098fu, 0x3f37ef9cu, 0x3f3e6289u, 0x3f43c112u, 0x3f484bf8u, 0x3f4c311du, 0x3f4f91e5u, 0x3f528746u, 0x3f552441u };
// The per-node words in velocity contact: 0..5 the λ sums (x, y, z as low, high words), 9..14 the
// iterate y's, 15 the node's rows, 16 its rows' |w| in CCV_GERSH_FIXED units.
static const uint CCV_SUM_L = 0u;
static const uint CCV_SUM_Y = 9u;
static const uint CCV_ROWS = 15u;
static const uint CCV_GERSH = 16u;
// The group's flags (its first member's words): 12 the round's hits, 13 a fresh row, 14 a row
// unsettled.
uint ccvG(uint off, uint k) { return ccVW(vvMember(off, 0u), k); }

// A row (the pair record's words 16..27): weights, normal, bound, λ, start separation, the
// iterate y, the preconditioner's diagonal.
uint ccvAt(uint v, uint p) { return ccPairAt(v, p) + 16u; }
float4 ccvW(uint at) { return float4(asfloat(gNext[at]), asfloat(gNext[at + 1u]), asfloat(gNext[at + 2u]), asfloat(gNext[at + 3u])); }
float3 ccvN(uint at) { return float3(asfloat(gNext[at + 4u]), asfloat(gNext[at + 5u]), asfloat(gNext[at + 6u])); }
float ccvF(uint at, uint k) { return asfloat(gNext[at + k]); }
void ccvSet(uint at, uint k, float x) { gNext[at + k] = asuint(x); }
static const uint CCV_B = 7u;
static const uint CCV_L = 8u;
static const uint CCV_S0 = 9u;
static const uint CCV_Y = 10u;
static const uint CCV_D = 11u;

// x·2²² rounded to nearest, ties to even, as a 64-bit integer (low, high), the CPU's
// (x as f64 · VEL_FIXED).round_ties_even() as i64; past ±4.6e18 counted and clamped.
uint2 ccvFix(float x)
{
    float y = nf(x * CC_VEL_FIXED);
    if (!(abs(y) < asfloat(0x5e7f59efu)))
    {
        gAcc[ccC(CC_C_OVERFLOW)].add(1);
        if (!(abs(y) >= 0.0)) return uint2(0u, 0u);
        return y > 0.0 ? uint2(0xcecc0000u, 0x3fd67ba0u) : uint2(0x31340000u, 0xc029845fu);
    }
    float a = abs(y);
    float f = floor(a);
    float d = nf(a - f);
    if (d > 0.5 || (d == 0.5 && floor(f * 0.5) * 2.0 != f)) f = f + 1.0;
    uint lo;
    uint hi;
    if (f < 2147483648.0)
    {
        lo = uint(f);
        hi = 0u;
    }
    else
    {
        uint b = asuint(f);
        uint m = (b & 0x7FFFFFu) | 0x800000u;
        uint s = ((b >> 23) & 0xFFu) - 150u;
        if (s < 32u)
        {
            lo = m << s;
            hi = m >> (32u - s);
        }
        else
        {
            lo = 0u;
            hi = m << (s - 32u);
        }
    }
    if (y < 0.0)
    {
        lo = ~lo + 1u;
        hi = ~hi + (lo == 0u ? 1u : 0u);
    }
    return uint2(lo, hi);
}
void ccvAdd64(uint at, uint2 x)
{
    uint old = asuint(gAcc[at].add(asint(x.x)));
    uint carry = old + x.x < old ? 1u : 0u;
    if (x.y + carry != 0u) gAcc[at + 1u].add(asint(x.y + carry));
}
// The 64-bit sum (low, high) over 2²², rounded once to f32 (the CPU's (a as f64 / VEL_FIXED) as f32).
float ccvUnfix(uint lo, uint hi)
{
    bool neg = (hi & 0x80000000u) != 0u;
    if (neg)
    {
        lo = ~lo + 1u;
        hi = ~hi + (lo == 0u ? 1u : 0u);
    }
    float f;
    if (hi == 0u && lo < 16777216u) f = float(lo);
    else
    {
        uint p = hi != 0u ? 32u + firstbithigh(hi) : firstbithigh(lo);
        uint s = p - 23u;
        uint m = s >= 32u ? hi >> (s - 32u) : ((lo >> s) | (hi << (32u - s)));
        uint r = s - 1u;
        bool half = ((r >= 32u ? (hi >> (r - 32u)) : (lo >> r)) & 1u) != 0u;
        bool sticky = r >= 32u ? (lo != 0u || (hi & ((1u << (r - 32u)) - 1u)) != 0u) : (r > 0u && (lo & ((1u << r) - 1u)) != 0u);
        if (half && (sticky || (m & 1u) != 0u)) m = m + 1u;
        if (m == 16777216u)
        {
            m = 8388608u;
            p = p + 1u;
        }
        f = asfloat(((p + 127u) << 23) | (m & 0x7FFFFFu));
    }
    f = smul(f, 2.384185791015625e-07);
    return neg ? -f : f;
}
// Node gi's velocity from its base and the sums at word `w` (velocities: base + M⁻¹Jᵀλ).
float3 ccvVel(uint gi, uint w)
{
    float3 base = gBeams[ccB(gi, 4u)].xyz;
    uint s[6];
    bool any0 = false;
    for (uint c = 0u; c < 6u; c++)
    {
        s[c] = asuint(gAcc[ccA(gi, w + c)].load());
        any0 = any0 || s[c] != 0u;
    }
    if (!any0) return base;
    float3 d;
    for (uint c = 0u; c < 3u; c++) d[c] = ccvUnfix(s[2u * c], s[2u * c + 1u]);
    return sadd3(base, d);
}
// Row at `at` of pair q adds its impulse λ into the sums at word `w`.
void ccvImpulse(CcPair q, uint at, float l, uint w)
{
    if (l == 0.0) return;
    float4 wr = ccvW(at);
    float3 n = ccvN(at);
    for (uint j = 0u; j < 4u; j++)
    {
        float m = ccIm(q.k[j]);
        if (m > 0.0 && wr[j] != 0.0)
        {
            float s = smul(smul(smul(wr[j], m), l), 1.0);
            float3 d = sscale(n, s);
            for (uint c = 0u; c < 3u; c++) ccvAdd64(ccA(q.k[j], w + 2u * c), ccvFix(d[c]));
        }
    }
}
// A row's closing speed J v (separating positive), v from the slot-3 velocities or the y sums.
float ccvNormal(CcPair q, uint at, bool fromY)
{
    float4 wr = ccvW(at);
    float3 n = ccvN(at);
    float c = 0.0;
    for (uint j = 0u; j < 4u; j++) c = nf(c + smul(wr[j], sdot(n, fromY ? ccvVel(q.k[j], CCV_SUM_Y) : ccVel(q.k[j]))));
    return c;
}
// Node gi's velocity as far as its storage and sums can tell it (Ctx::precision).
float ccvPrec(uint gi)
{
    float3 v = abs(ccVel(gi));
    float x = max(max(max(v.x, v.y), v.z), 1.17549435e-38);
    float ulp = nf(asfloat(asuint(x) + 1u) - x);
    return smul(CCV_ROUND_ULPS, nf(ulp + sdiv(float(gAcc[ccA(gi, CCV_ROWS)].load()), CC_VEL_FIXED)));
}
// Ctx::row_settled.
bool ccvSettled(CcPair q, uint at, float xi)
{
    float g = nf(ccvNormal(q, at, false) + ccvF(at, CCV_B));
    float4 wr = ccvW(at);
    float rounding = 0.0;
    for (uint j = 0u; j < 4u; j++) rounding = nf(rounding + smul(abs(wr[j]), ccvPrec(q.k[j])));
    float l = ccvF(at, CCV_L);
    float b = ccvF(at, CCV_B);
    return smul(-g, STEP) <= smul(CCV_GAP_SHARE, min(xi, max(ccvF(at, CCV_S0), 0.0)))
        && (smul(l, g) <= CCV_PUSH_TOL || g <= smul(CCV_PUSH_SHARE, abs(b)) || g <= rounding);
}
// The pair's closest gap vector at the end positions (x1).
float3 ccvEndGap(CcPair q)
{
    float3 p[4];
    for (uint j = 0u; j < 4u; j++) p[j] = ccX1(q.k[j]);
    float4 w;
    return ccClosest(q.nn, p, w);
}


// The rows' settle test into the group's flag 14 (set: some row unsettled); uniform result.
bool ccvAllSettled(uint off, uint k, uint lane)
{
    if (lane == 0u) gAcc[ccvG(off, 14u)].store(0);
    sync();
    for (uint m = 0u, mStop = EB_END(1009, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        for (uint i = lane, iStop = EB_END(936, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
        {
            uint p = ccListP(v, 1u, i);
            if ((ccFlags(v, p) & CC_F_ROW) == 0u) continue;
            CcPair q = ccLoad(v, p);
            uint at = ccvAt(v, p);
            if (!ccvSettled(q, at, ccXiV(ccRecAt(v, p)))) gAcc[ccvG(off, 14u)].store(1);
        }
    }
    sync();
    return splitUniform(lane, uint(gAcc[ccvG(off, 14u)].load())) == 0u;
}
// vel = base + the λ sums (slot 3), every group node; both sums cleared.
void ccvRebuild(uint off, uint k, uint lane)
{
    for (uint m = 0u, mStop = EB_END(1010, m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(937, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            float3 v = ccvVel(gi, CCV_SUM_L);
            gBeams[ccB(gi, 3u)].xyz = v;
            for (uint c = 0u; c < 6u; c++)
            {
                gAcc[ccA(gi, CCV_SUM_L + c)].store(0);
                gAcc[ccA(gi, CCV_SUM_Y + c)].store(0);
            }
        }
    }
    sync();
}

// One pass over the rows (a phase of Ctx::solve; one call site of the impulse, a heavy helper):
// mode 0 the preconditioner's diagonal, y = λ and λ's sums; 1 y's sums; 2 APGD's step to the
// next λ (from the velocities at y) and its sums.
void ccvRowPhase(uint off, uint k, uint lane, uint mode, uint used)
{
    for (uint m = 0u, mStop = EB_END(1013, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        for (uint i = lane, iStop = EB_END(940, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
        {
            uint p = ccListP(v, 1u, i);
            if ((ccFlags(v, p) & CC_F_ROW) == 0u) continue;
            CcPair q = ccLoad(v, p);
            uint at = ccvAt(v, p);
            float lam = ccvF(at, mode == 1u ? CCV_Y : CCV_L);
            if (mode == 0u)
            {
                float4 wr = ccvW(at);
                float d = 0.0;
                for (uint j = 0u; j < 4u; j++)
                    d = nf(d + smul(smul(abs(wr[j]), ccIm(q.k[j])), sdiv(float(gAcc[ccA(q.k[j], CCV_GERSH)].load()), CCV_GERSH_FIXED)));
                ccvSet(at, CCV_D, d);
                ccvSet(at, CCV_Y, lam);
            }
            else if (mode == 2u)
            {
                float d = ccvF(at, CCV_D);
                float y = ccvF(at, CCV_Y);
                float nx = 0.0;
                if (d > 0.0)
                {
                    float t = nf(y - sdiv(nf(ccvNormal(q, at, true) + ccvF(at, CCV_B)), d));
                    nx = t > 0.0 ? t : 0.0;
                }
                if (used % CCV_RESTART == 0u) y = nx;
                else y = nf(nx + smul(asfloat(CCV_BETA[used % CCV_RESTART - 1u]), nf(nx - lam)));
                ccvSet(at, CCV_Y, y);
                ccvSet(at, CCV_L, nx);
                lam = nx;
            }
            ccvImpulse(q, at, lam, mode == 1u ? CCV_SUM_Y : CCV_SUM_L);
        }
    }
    sync();
}

// Ctx::solve: APGD on the rows from their λ. Leaves slot 3 at the final λ's velocities; true
// when settled.
bool ccvSolve(uint off, uint k, uint lane)
{
    // The preconditioner (Gershgorin): each node's rows and their |w| in fixed point.
    for (uint m = 0u, mStop = EB_END(1011, m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(938, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            gAcc[ccA(gi, CCV_ROWS)].store(0);
            gAcc[ccA(gi, CCV_GERSH)].store(0);
        }
    }
    sync();
    for (uint m = 0u, mStop = EB_END(1012, m, k, 1u); m < mStop; m++)
    {
        uint v = vvMember(off, m);
        for (uint i = lane, iStop = EB_END(939, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
        {
            uint p = ccListP(v, 1u, i);
            if ((ccFlags(v, p) & CC_F_ROW) == 0u) continue;
            CcPair q = ccLoad(v, p);
            uint at = ccvAt(v, p);
            float4 wr = ccvW(at);
            for (uint j = 0u; j < 4u; j++)
                if (ccIm(q.k[j]) > 0.0)
                {
                    float f = nf(abs(wr[j]) * CCV_GERSH_FIXED);
                    float r = floor(f);
                    float d = nf(f - r);
                    if (d > 0.5 || (d == 0.5 && floor(r * 0.5) * 2.0 != r)) r = r + 1.0;
                    gAcc[ccA(q.k[j], CCV_GERSH)].add(int(r));
                    if (wr[j] != 0.0) gAcc[ccA(q.k[j], CCV_ROWS)].add(1);
                }
        }
    }
    sync();
    // Each row's diagonal; y = λ; the λ sums.
    ccvRowPhase(off, k, lane, 0u, 0u);
    ccvRebuild(off, k, lane);
    bool ok = ccvAllSettled(off, k, lane);
    uint iters = ok ? 1u : CCV_ITERS + 1u;
    for (uint used = 1u, usedStop = EB_END_UNIFORM(941, used, iters, 1u, CCV_ITERS + 1u); used < usedStop; used++)
    {
        if (lane == 0u) gAcc[ccC(CC_C_VITERS)].add(1);
        ccvRowPhase(off, k, lane, 1u, used);
        ccvRowPhase(off, k, lane, 2u, used);
        ccvRebuild(off, k, lane);
        ok = ccvAllSettled(off, k, lane);
        if (ok) break;
    }
    return ok;
}

// Ctx::rounds_inner (e = 0: no restitution pass): rows from the pairs colliding over the paths,
// solved, positions following, until nothing new collides. True when resolved.
bool ccRespondVel(uint off, uint k, uint lane)
{
    // The rounds' base: the velocities (slot 4) and end positions (slot 2) before them.
    for (uint m = 0u, mStop = EB_END(1016, m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(930, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            gBeams[ccB(gi, 4u)].xyz = ccVel(gi);
            gBeams[ccB(gi, 2u)].xyz = ccX1(gi);
            for (uint c = 0u; c < 6u; c++)
            {
                gAcc[ccA(gi, CCV_SUM_L + c)].store(0);
                gAcc[ccA(gi, CCV_SUM_Y + c)].store(0);
            }
        }
    }
    sync();
    bool settled = false;
    bool done = false;
    for (uint round = 0u, roundStop = EB_END_UNIFORM(931, round, CC_CCD_ROUNDS, 1u, CC_CCD_ROUNDS); round < roundStop; round++)
    {
        if (lane == 0u)
        {
            gAcc[ccvG(off, 12u)].store(0);
            gAcc[ccvG(off, 13u)].store(0);
            gAcc[ccC(CC_C_VROUNDS)].add(1);
        }
        sync();
        for (uint m = 0u, mStop = EB_END(1017, m, k, 1u); m < mStop; m++)
        {
            uint v = vvMember(off, m);
            for (uint i = lane, iStop = EB_END(932, i, ccListN(v, 1u), LANES); i < iStop; i += LANES)
            {
                uint p = ccListP(v, 1u, i);
                CcPair q = ccLoad(v, p);
                CcRec rec = ccRecAt(v, p);
                float xi = ccXiV(rec);
                float t = ccCollide(q, xi);
                if (t < 0.0) continue;
                gAcc[ccC(CC_C_HITS)].add(1);
                gAcc[ccvG(off, 12u)].store(1);
                uint at = ccvAt(v, p);
                bool old = (ccFlags(v, p) & CC_F_ROW) != 0u;
                float3 endGap = ccvEndGap(q);
                if (old)
                {
                    float4 wr = ccvW(at);
                    float3 n = ccvN(at);
                    float sEnd = 0.0;
                    for (uint j = 0u; j < 4u; j++) sEnd = nf(sEnd + smul(wr[j], sdot(n, ccX1(q.k[j]))));
                    float sTrue = sdot(endGap, n);
                    float s = min(sEnd, sTrue);
                    if (s > 0.0 && s >= smul(asfloat(CCV_RELIN_BITS), min(ccvF(at, CCV_S0), xi))) continue;
                    if (sdot(endGap, n) > 0.0) t = 1.0;
                }
                float4 w;
                float3 dp[4];
                float3 n = contactNormOrZero(ccAt(q, t, w, dp));
                if (all(n == float3(0.0, 0.0, 0.0))) continue;
                gAcc[ccvG(off, 13u)].store(1);
                float s0 = 0.0;
                for (uint j = 0u; j < 4u; j++) s0 = nf(s0 + smul(w[j], sdot(n, ccX0(j < q.nn ? q.vb : q.va, q.k[j]))));
                for (uint j = 0u; j < 4u; j++) ccvSet(at, j, w[j]);
                ccvSet(at, 4u, n.x);
                ccvSet(at, 5u, n.y);
                ccvSet(at, 6u, n.z);
                ccvSet(at, CCV_B, sdiv(max(nf(s0 - xi), 0.0), STEP));
                if (!old) ccvSet(at, CCV_L, 0.0);
                ccvSet(at, CCV_S0, s0);
                ccFlagSet(v, p, CC_F_ROW);
            }
        }
        sync();
        if (splitUniform(lane, uint(gAcc[ccvG(off, 12u)].load())) == 0u) { done = true; break; }
        if (settled && splitUniform(lane, uint(gAcc[ccvG(off, 13u)].load())) == 0u) { done = true; break; }
        settled = ccvSolve(off, k, lane);
        for (uint m = 0u, mStop = EB_END(1018, m, k, 1u); m < mStop; m++)
        {
            uint4 r0 = gStatic[vvMember(off, m) * 4u];
            for (uint i = lane, iStop = EB_END(933, i, r0.y, LANES); i < iStop; i += LANES)
            {
                uint gi = r0.x + i;
                gBeams[ccB(gi, 1u)].xyz = sadd3(gBeams[ccB(gi, 2u)].xyz, sscale(ssub3(ccVel(gi), gBeams[ccB(gi, 4u)].xyz), STEP));
            }
        }
        sync();
    }
    // (Every node moved for the zones' cached times: CC_CCD_ROUNDS, their first round's test.)
    for (uint m = 0u, mStop = EB_END(1019, m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(934, i, r0.y, LANES); i < iStop; i += LANES)
        {
            uint gi = r0.x + i;
            gBeams[ccB(gi, 3u)].w = float(CC_CCD_ROUNDS);
        }
    }
    sync();
    return done;
}
