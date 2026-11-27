// --- numerical primitives (compatible profile) ---

// Strict float arithmetic (slang/physics/strict.slang); its zero comes from a uniform.
uint noFuseBits() { return gP.vehicles >> 31; }
#include "../physics/strict.slang"
// n, from a uniform the compilers cannot see through (noFuseBits is 0 at run time): a
// loop `for (k < EB_END(k, opaqueCount(n), 1u))` over a call's n uses (the same calls in
// the same order) stays a loop, its body one inlined copy, where the n calls written
// out are n copies after the drivers inline everything
// (engine/docs/measurements/kernel-compile-time.md).
uint opaqueCount(uint n) { return n + noFuseBits(); }

float sFastInvSqrt(float x)
{
    int i = 0x5f3759df - (asint(x) >> 1);
    float y = asfloat(i);
    return smul(y, nf(1.5 - smul(smul(smul(0.5, x), y), y)));
}

// The exact path's inverse square root (rubble_exact_lengths; explicit_beams.rs
// exact_inv_sqrt): the fast estimate and three more Newton steps, products and
// differences in the CPU's order, so the CPU and GPU agree bit for bit.
float sExactInvSqrt(float x)
{
    int i = 0x5f3759df - (asint(x) >> 1);
    float y = asfloat(i);
    float hx = smul(0.5, x);
    y = smul(y, nf(1.5 - smul(smul(hx, y), y)));
    y = smul(y, nf(1.5 - smul(smul(hx, y), y)));
    y = smul(y, nf(1.5 - smul(smul(hx, y), y)));
    return smul(y, nf(1.5 - smul(smul(hx, y), y)));
}

// rubble_exact_lengths' (length, inverse length) of squared length q
// (explicit_beams.rs exact_length): the length q·y with one Newton correction
// through its exact residual.
float2 sExactLength(float q)
{
    float y = sExactInvSqrt(q);
    float l = smul(q, y);
    return float2(nf(fma(nf(fma(-l, l, q)), smul(0.5, y), l)), y);
}

float approxSqrt(float x)
{
    int i = ((asint(x) - 0x3F800000) >> 1) + 0x3F800000;
    return asfloat(i);
}

float approxExp(float x)
{
    if (x < -15.0) return 0.0;
    if (x > 88.0) return 1e38;
    return asfloat(int(nf(smul(12102203.0, x) + 1064652319.0)));
}

float approxPow(float x, float y)
{
    float e = float(asint(x) - 1065353216);
    return asfloat(int(nf(smul(y, e) + 1065353216.0)));
}

// explicit_beams.rs exact_exp / exact_ln / exact_pow: one specified sequence, strict,
// in the CPU's order (jbeam-semantics-spec.md §36).
static const float LN2_HI = 0.693359375;
static const float LN2_LO = -2.12194440e-4;
float sExactExp(float x)
{
    if (x < -87.0) return 0.0;
    if (x > 88.0) return 3.4e38;
    float k = floor(nf(smul(x, 1.44269502) + 0.5));
    float r = nf(nf(x - smul(k, LN2_HI)) - smul(k, LN2_LO));
    float p = 0.000198412701;
    p = nf(smul(p, r) + 0.00138888892);
    p = nf(smul(p, r) + 0.00833333377);
    p = nf(smul(p, r) + 0.0416666679);
    p = nf(smul(p, r) + 0.166666672);
    p = nf(smul(p, r) + 0.5);
    p = nf(smul(p, r) + 1.0);
    p = nf(smul(p, r) + 1.0);
    return smul(p, asfloat(uint(int(k) + 127) << 23));
}
float sExactLn(float x)
{
    if (!(x >= 1.17549435e-38)) return -1000.0;
    uint b = asuint(x);
    float e = float(int(b >> 23) - 127);
    float m = asfloat((b & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421354)
    {
        m = smul(m, 0.5);
        e = nf(e + 1.0);
    }
    float sv = sdiv(nf(m - 1.0), nf(m + 1.0));
    float s2 = smul(sv, sv);
    float q = sdiv(1.0, 11.0);
    q = nf(smul(q, s2) + sdiv(1.0, 9.0));
    q = nf(smul(q, s2) + sdiv(1.0, 7.0));
    q = nf(smul(q, s2) + sdiv(1.0, 5.0));
    q = nf(smul(q, s2) + sdiv(1.0, 3.0));
    q = nf(smul(q, s2) + 1.0);
    float lnm = smul(nf(sv + sv), q);
    return nf(smul(e, LN2_HI) + nf(lnm + smul(e, LN2_LO)));
}
float sExactPow(float x, float y)
{
    if (y == 0.0) return 1.0;
    if (x == 0.0 && y > 0.0) return 0.0;
    return sExactExp(smul(y, sExactLn(x)));
}
float frictionExp(Surface s, float x) { return s.exact != 0u ? sExactExp(x) : approxExp(x); }
float frictionPow(Surface s, float x, float y) { return s.exact != 0u ? sExactPow(x, y) : approxPow(x, y); }

// Turbulence: a well-mixed hash per vehicle, step, node and axis, as a value in
// [−1, 1) built like the CPU sequence's (a float in [2, 4) minus 3).
float noise(uint vehicle, uint step, uint node, uint axis)
{
    uint h = vehicle * 0x9E3779B9u ^ (step * 0x85EBCA6Bu) ^ (node * 3u + axis) * 0xC2B2AE35u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return asfloat(0x40000000u | (h & 0x7FFFFFu)) - 3.0;
}

// §37 (explicit_beams.rs no_positive_work): contact force c (normal part r·n) on a
// pair of relative velocity v, reduced mass m, under force f, its tangential part
// t = c − r·n scaled by the largest s in [0, 1] keeping the force's work over the
// step at the pair's mean velocity ≤ 0 (the positive root of a0 + a1·s + b·s²).
// Strict, in the CPU's order.
float3 noPositiveWork(float3 c, float r, float3 n, float3 f, float3 v, float m)
{
    if (m <= 0.0) return c;
    float hm = sdiv(STEP, m);
    float3 rn = sscale(n, r);
    float3 t = ssub3(c, rn);
    float3 u = sadd3(sadd3(v, v), sscale(f, hm));
    float a0 = nf(sdot(rn, u) + smul(sdot(rn, rn), hm));
    float a1 = nf(sdot(t, u) + smul(smul(2.0, hm), sdot(rn, t)));
    float b = smul(sdot(t, t), hm);
    float k = 1.0;
    bool whole = true;
    if (nf(nf(a0 + a1) + b) > 0.0 && b > 0.0)
    {
        whole = false;
        float disc = nf(smul(a1, a1) - smul(smul(4.0, b), a0));
        k = 0.0;
        if (!(disc < 0.0)) k = clamp(sdiv(nf(ssqrt(disc) - a1), smul(2.0, b)), 0.0, 1.0);
    }
    // Nor does friction reverse the slip (explicit_beams.rs no_positive_work).
    float3 vt = ssub3(v, sscale(n, sdot(v, n)));
    float g = nf(sdot(vt, vt) + smul(hm, sdot(f, vt)));
    float d = smul(hm, sdot(t, vt));
    if (d < 0.0 && nf(g + smul(d, k)) < 0.0)
    {
        k = clamp(sdiv(g, -d), 0.0, k);
        whole = false;
    }
    if (whole) return c;
    return sadd3(rn, sscale(t, k));
}

// §37 (explicit_beams.rs shift_depth): the penetration's position step δ (m) from
// the depth p, the normal force's budget (J) and the pair's normal force fn.
float shiftDepth(float p, float budget, float fn)
{
    float full = smul(0.2, p);
    if (fn >= 0.0 || smul(-fn, full) <= budget) return full;
    return sdiv(budget, -fn);
}

// contactForce, and on a dissipative surface (§37) the energy its normal force
// takes out of the pair this step (`budget`, J; 0 otherwise): explicit_beams.rs
// contact_response.
float3 contactResponse(float3 f, float3 v, float m, float3 n, Surface s, float p, float friction, out float budget)
{
    budget = 0.0;
    if (p < 0.0) return float3(0.0, 0.0, 0.0);
    // Strict, in the CPU reference's order (its branches decide on these values).
    float vn = sdot(v, n);
    float fn = sdot(f, n);
    float r = -fn;
    float bud = 0.0;
    if (s.dissipative != 0u)
    {
        // §37: the approach stopped, a separation never made:
        // r = −fn − (vn + max(vn, 0))·m/h.
        r = nf(-fn - sdiv(smul(nf(vn + max(vn, 0.0)), m), STEP));
        if (vn < 0.0) bud = max(nf(smul(smul(0.5, m), smul(vn, vn)) + smul(smul(0.5, STEP), smul(fn, vn))), 0.0);
    }
    else if (vn < 0.0)
        r = nf(r - sdiv(smul(nf(smul(0.8, vn) - sdiv(smul(0.2, p), STEP)), m), STEP));
    if (r <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 sf = ssub3(f, sscale(n, fn));
    float3 sl = ssub3(v, sscale(n, vn));
    float sigma = ssqrt(sdot(sl, sl));
    if (sigma > 0.0) sl = sdiv3(sl, sigma);
    float gr = smul(smul(r, s.strength), friction);
    float ms = smul(s.ms, gr);
    float3 c;
    if (sigma < s.va && gr > 0.0 && sdot(sf, sf) <= smul(ms, ms))
        c = ssub3(ssub3(sscale(n, r), sscale(sl, smul(ms, nf(1.0 - frictionExp(s, -sdiv(sigma, s.va)))))), sf);
    else
    {
        float g = nf(s.mc + smul(nf(s.ms - s.mc), frictionExp(s, -frictionPow(s, sdiv(sigma, s.vs), s.alpha))));
        c = ssub3(sscale(n, r), sscale(sl, smul(nf(g + min(smul(s.t2, sigma), 5.0)), gr)));
    }
    if (s.dissipative == 0u) return c;
    budget = bud;
    return noPositiveWork(c, r, n, f, v, m);
}

float3 contactForce(float3 f, float3 v, float m, float3 n, Surface s, float p, float friction)
{
    float budget;
    return contactResponse(f, v, m, n, s, p, friction, budget);
}

// The node pass's ground contact (depth > 0) of vehicle v's node: its force, and on
// the dissipative surface (§37) its position step up (`lift`, m; explicit_beams.rs
// node_pass).
float3 groundContact(float3 f, float3 vel, float mass, uint v, float depth, float friction, inout float lift)
{
    Surface s = surfaceV(groundSurface(), v);
    float budget;
    float3 c = contactResponse(f, vel, mass, float3(0.0, 1.0, 0.0), s, depth, friction, budget);
    if (s.dissipative != 0u && any(c != float3(0.0, 0.0, 0.0))) lift = shiftDepth(depth, budget, f.y);
    return c;
}

// §37 after the node's x + h·v: its contacts' position step d, then the ground's.
float3 takeSteps(float3 pos, float3 d, float lift)
{
    pos = sadd3(pos, d);
    if (lift != 0.0) pos.y = nf(pos.y + lift);
    return pos;
}

// A pressure-wheel node's ground contact (NODE_FRICTION_LIMIT, spec §28): the
// node rises at most half its depth p in one step, and the tangential part at
// most stops the slip in one step
// (explicit_jbeam_ext.rs friction_limit).
float3 frictionLimit(float3 c, float3 f, float3 v, float m, float3 n, float p)
{
    float cn = sdot(c, n);
    float out = nf(sdot(v, n) + smul(nf(sdot(f, n) + cn), sdiv(STEP, m)));
    float most = sdiv(smul(0.5, p), STEP);
    if (cn > 0.0 && out > most)
    {
        float capped = max(nf(smul(nf(most - sdot(v, n)), sdiv(m, STEP)) - sdot(f, n)), 0.0);
        c = sadd3(c, sscale(n, nf(capped - cn)));
    }
    float3 total = sadd3(f, c);
    float3 t = ssub3(total, sscale(n, sdot(total, n)));
    float3 vt = ssub3(v, sscale(n, sdot(v, n)));
    float3 dv = sscale(t, sdiv(STEP, m));
    if (sdot(dv, vt) < 0.0 && sdot(dv, dv) > sdot(vt, vt)) return sadd3(c, ssub3(sscale(-vt, sdiv(m, STEP)), t));
    return c;
}

// The node pass's ground and box contacts of a node at x (world), in order: the ground
// (depth ground − x.y > 0; with the friction limit when the node has it), then each box
// it is inside, f with each one's force added. One call site of contactForce for all of
// them (a loop over the ground, then the boxes: the same forces in the same order), where
// the drivers would inline a copy per call (engine/docs/measurements/kernel-compile-time.md).
float3 groundBoxContacts(float3 f, float3 vel, float mass, float3 x, uint4 info, uint vehicle, float ground,
                         uint boxOff, uint boxCount, float friction)
{
    for (uint b = 0u, bStop = EB_END(1, b, boxCount + 1u, 1u); b < bStop; b++)
    {
        float3 n = float3(0.0, 1.0, 0.0);
        float depth = ground - x.y;
        bool on = ground > x.y;
        if (b > 0u)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b - 1u) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b - 1u) * 2u + 1u].xyz);
            on = x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z;
            if (on)
            {
                n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
            }
            depth = 0.0;
        }
        if (on)
        {
            float3 c = contactForce(f, vel, mass, n, surfaceV(b == 0u ? groundSurface() : boxSurface(), vehicle), depth, friction);
            if (b == 0u && (info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            f += c;
        }
    }
    return f;
}

// The tread contact (NODE_TREAD, jbeam-semantics-spec.md §34; CPU
// explicit_beams::tread_hysteresis and the node pass's tread branch): a compliant,
// rate-independent hysteretic normal force k·(p + z) at depth p, z the node's
// hysteretic deflection (slope μ while |z| grows, λ after a reversal until it passes
// zero), the solver's friction law on that normal force, and the slip limit. State
// in gBeams at treadState per node: (z, the depth the step before), both 0 off the
// ground. Strict arithmetic in the CPU's order. Off the ground (`on` false) it only
// clears the state and returns 0.
static const uint TREAD_GROW_BITS = 0x3e3c7bbfu; // TREAD_GROW, μ
static const uint TREAD_REVERSE_BITS = 0x3f2c73d7u; // TREAD_REVERSE, λ
static const uint TREAD_INV_REVERSE_BITS = 0x3fbe0301u; // 1/λ
static const uint TREAD_K_BITS = 0x4a11dfb2u; // TREAD_K, 1/(h²(1 + λ))

// contactForce's friction law on a given normal force r > 0 (CPU contact_friction).
float3 contactFriction(float r, float3 f, float3 v, float3 n, Surface s, float friction)
{
    float vn = sdot(v, n);
    float fn = sdot(f, n);
    float3 sf = ssub3(f, sscale(n, fn));
    float3 sl = ssub3(v, sscale(n, vn));
    float sigma = ssqrt(sdot(sl, sl));
    if (sigma > 0.0) sl = sdiv3(sl, sigma);
    float gr = smul(smul(r, s.strength), friction);
    float ms = smul(s.ms, gr);
    if (sigma < s.va && gr > 0.0 && sdot(sf, sf) <= smul(ms, ms))
        return ssub3(ssub3(sscale(n, r), sscale(sl, smul(ms, nf(1.0 - frictionExp(s, -sdiv(sigma, s.va)))))), sf);
    float g = nf(s.mc + smul(nf(s.ms - s.mc), frictionExp(s, -frictionPow(s, sdiv(sigma, s.vs), s.alpha))));
    return ssub3(sscale(n, r), sscale(sl, smul(nf(g + min(smul(s.t2, sigma), 5.0)), gr)));
}

// frictionLimit's tangential part alone (CPU explicit_jbeam_ext::slip_limit).
float3 slipLimit(float3 c, float3 f, float3 v, float m, float3 n)
{
    float3 total = sadd3(f, c);
    float3 t = ssub3(total, sscale(n, sdot(total, n)));
    float3 vt = ssub3(v, sscale(n, sdot(v, n)));
    float3 dv = sscale(t, sdiv(STEP, m));
    if (sdot(dv, vt) < 0.0 && sdot(dv, dv) > sdot(vt, vt)) return sadd3(c, ssub3(sscale(-vt, sdiv(m, STEP)), t));
    return c;
}

float3 treadGround(uint gi, float3 f, float3 v, float m, float friction, float p, bool on)
{
    uint at = gP.treadState + gi;
    float4 st = gBeams[at];
    if (!on)
    {
        if (st.x != 0.0 || st.y != 0.0) gBeams[at] = float4(0.0, 0.0, 0.0, 0.0);
        return float3(0.0, 0.0, 0.0);
    }
    float z = st.x;
    float dp = nf(p - st.y);
    float z1;
    if (smul(z, dp) >= 0.0)
        z1 = nf(z + smul(asfloat(TREAD_GROW_BITS), dp));
    else
    {
        float t = nf(z + smul(asfloat(TREAD_REVERSE_BITS), dp));
        if ((t > 0.0) != (z > 0.0))
            z1 = smul(asfloat(TREAD_GROW_BITS), nf(dp + smul(z, asfloat(TREAD_INV_REVERSE_BITS))));
        else
            z1 = t;
    }
    if (z1 < -p) z1 = -p;
    gBeams[at] = float4(z1, p, 0.0, 0.0);
    float r = smul(smul(m, asfloat(TREAD_K_BITS)), nf(p + z1));
    if (!(r > 0.0)) return float3(0.0, 0.0, 0.0);
    float3 n = float3(0.0, 1.0, 0.0);
    return slipLimit(contactFriction(r, f, v, n, surfaceV(groundSurface(), gStatic[gP.nodeInfoBase + gi].w), friction), f, v, m, n);
}

float3 accForce(uint node)
{
    return float3(float(gAcc[node * 3u].load()), float(gAcc[node * 3u + 1u].load()),
                  float(gAcc[node * 3u + 2u].load())) * (1.0 / FIXED);
}

// x in 1/FIXED N units, clamped to ±2·10⁹, rounded to nearest with ties to even
// (explicit_beams.rs to_fixed): spelt out, as the targets' round differ at ties
// (WGSL's ties to even, Metal's away from zero). Every step is exact: y − floor(y)
// for |y| < 2²⁴ (above, y is whole), f·0.5, f + 1.
int toFixed(float x)
{
    float y = clamp(nf(x * FIXED), -2.0e9, 2.0e9);
    float f = floor(y);
    float d = nf(y - f);
    if (d > 0.5 || (d == 0.5 && floor(f * 0.5) * 2.0 != f)) f = f + 1.0;
    return int(f);
}

// §37's position steps: node x's 3 words in gAcc (after the two-body counts).
uint shiftAt(uint x) { return gP.selfSplit + gP.totalNodes + x * 3u; }

// x in 1/SHIFT_FIXED m units, as toFixed (explicit_beams.rs to_fixed_shift).
int toFixedShift(float x)
{
    float y = clamp(nf(x * SHIFT_FIXED), -2.0e9, 2.0e9);
    float f = floor(y);
    float d = nf(y - f);
    if (d > 0.5 || (d == 0.5 && floor(f * 0.5) * 2.0 != f)) f = f + 1.0;
    return int(f);
}

// Adds position step d (m) to node x's (ExplicitBeams::add_shift).
void shiftAdd(uint x, float3 d)
{
    uint at = shiftAt(x);
    gAcc[at].add(toFixedShift(d.x));
    gAcc[at + 1u].add(toFixedShift(d.y));
    gAcc[at + 2u].add(toFixedShift(d.z));
}

// The node phase: node gi's position step from the last contact pass (m), cleared
// (zero without FLAG_SHIFT_MEM).
float3 shiftTake(uint gi, uint flagsV)
{
    if (gP.selfSplit == 0u || (flagsV & FLAG_SHIFT_MEM) == 0u) return float3(0.0, 0.0, 0.0);
    uint at = shiftAt(gi);
    float3 d = float3(float(gAcc[at].load()), float(gAcc[at + 1u].load()), float(gAcc[at + 2u].load()))
               * (1.0 / SHIFT_FIXED);
    gAcc[at].store(0);
    gAcc[at + 1u].store(0);
    gAcc[at + 2u].store(0);
    return d;
}

void accAdd(uint node, float3 f)
{
    gAcc[node * 3u].add(toFixed(f.x));
    gAcc[node * 3u + 1u].add(toFixed(f.y));
    gAcc[node * 3u + 2u].add(toFixed(f.z));
}

uint groupSlot(uint parity, uint groupOff, uint g)
{
    return parity * gP.groupTotal + groupOff + g;
}

// --- Anti-clip (jbeam-semantics-spec.md §36; explicit_contact.rs clip_*) ---
// Per node: the sides held this step (last step's contacts) and those recorded for
// the next, each a word owner << 21 | triangle << 1 | (side > 0), CLIP_EMPTY none.
// A contact records its word into the two smallest (order free: atomic min, the
// displaced word carried to the second slot); the node phase rolls them.
static const float CLIP_DEPTH = 0.1;
static const uint CLIP_EMPTY = 0x7FFFFFFFu;
static const uint CLIP_SELF = 1023u;
uint clipAt(uint n) { return gP.clipMem + n * 4u; }
// The held side of node n for triangle t of `owner`: ±1, or 0 none.
float clipHeld(uint n, uint owner, uint t)
{
    uint key = (owner << 20) | t;
    for (uint k = 0u; k < 2u; k++)
    {
        uint w = uint(gAcc[clipAt(n) + k].load());
        if (w != CLIP_EMPTY && (w >> 1) == key) return (w & 1u) != 0u ? 1.0 : -1.0;
    }
    return 0.0;
}
uint clipWord(uint owner, uint t, float side) { return (owner << 21) | (t << 1) | (side > 0.0 ? 1u : 0u); }
void clipRecord(uint n, uint word)
{
    int x = int(word);
    int prev = gAcc[clipAt(n) + 2u].min(x);
    gAcc[clipAt(n) + 3u].min(max(prev, x));
}
// The node phase: last step's recorded words become the held ones.
void clipRoll(uint gi, uint flagsV)
{
    if (gP.clipMem == 0u || (flagsV & FLAG_CLIP_MEM) == 0u) return;
    uint at = clipAt(gi);
    gAcc[at].store(gAcc[at + 2u].load());
    gAcc[at + 1u].store(gAcc[at + 3u].load());
    gAcc[at + 2u].store(int(CLIP_EMPTY));
    gAcc[at + 3u].store(int(CLIP_EMPTY));
}
uint clipVmaxAt(uint v, uint parity) { return gP.clipVmax + v * 2u + parity; }
// The node phase: the vehicle's fastest node this step (|v| as the CPU's
// Vec3::length; NaN and 0 leave it).
void clipSpeed(uint vehicle, uint step, uint flagsV, float3 vel)
{
    if (gP.clipMem == 0u || (flagsV & FLAG_CLIP_MEM) == 0u) return;
    float sp = ssqrt(sdot(vel, vel));
    if (sp > 0.0) gAcc[clipVmaxAt(vehicle, step & 1u)].max(asint(sp));
}
// Clears the next step's slot (after this step's node phase, before the next's).
void clipVmaxClear(uint vehicle, uint parity)
{
    if (gP.clipMem != 0u) gAcc[clipVmaxAt(vehicle, parity ^ 1u)].store(0);
}
float clipVmax(uint vehicle, uint parity) { return asfloat(gAcc[clipVmaxAt(vehicle, parity)].load()); }
// The largest of a triangle's nodes' speeds, as the CPU folds it from 0.
float clipTriSpeed(uint4 tn)
{
    float a = ssqrt(sdot(gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.x * 3u + 1u].xyz));
    float b = ssqrt(sdot(gNodes[tn.y * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz));
    float c = ssqrt(sdot(gNodes[tn.z * 3u + 1u].xyz, gNodes[tn.z * 3u + 1u].xyz));
    float m = 0.0;
    if (a > m) m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}
// Velocity contact (the continuous tables' mode word, engine_exec continuous::cc_mode_words) runs
// no self-collision position steps (the CPU's ExplicitBeams::no_position_steps).
bool noPositionSteps(uint v) { return gP.cc != 0u && gStatic[gP.cc + 4u + 3u * v].z != 0u; }
// Self-collision's box reach (explicit_beams.rs collision_pass): r, or with
// anti-clip r + (triangle speed + vehicle's fastest) · h + CLIP_DEPTH.
float triReach(uint4 tn, float r, uint parity)
{
    if ((gStatic[tn.w * 4u + 3u].x & FLAG_ANTI_CLIP) == 0u) return r;
    float tv = clipTriSpeed(tn);
    return nf(nf(r + smul(nf(tv + clipVmax(tn.w, parity)), STEP)) + CLIP_DEPTH);
}

// --- The contact-force log (jbeam-semantics-spec.md §32; explicit_beams.rs's
// ContactForceLog): per node the sum of its contacts' force magnitudes in a step,
// fixed point; the node phase moves a complete step's sum to `last` (and the
// peak) and tests the node's contact break, which the next beam phase sets off
// once as if by its first beam. Separate functions, each behind an early return.
void cfAddMagnitude(uint gi, float m)
{
    if (gP.cfAcc == 0u) return;
    uint v = gStatic[gP.nodeInfoBase + gi].w;
    if ((gStatic[v * 4u + 3u].x & FLAG_CFORCE) == 0u) return;
    gAcc[gP.cfAcc + 3u * gi].add(toFixed(m));
}
void cfAdd(uint gi, float3 f)
{
    if (gP.cfAcc == 0u) return;
    cfAddMagnitude(gi, ssqrt(sdot(f, f)));
}
void cfRoll(uint gi, uint groupOff, uint flagsV)
{
    if (gP.cfAcc == 0u || (flagsV & FLAG_CFORCE) == 0u) return;
    uint at = gP.cfAcc + 3u * gi;
    int s = gAcc[at].load();
    gAcc[at + 1u].store(s);
    if (s > gAcc[at + 2u].load()) gAcc[at + 2u].store(s);
    gAcc[at].store(0);
    uint g = gLists[gP.cfNode + 2u * gi];
    if (g != 0u && float(s) * (1.0 / FIXED) > asfloat(gLists[gP.cfNode + 2u * gi + 1u]))
        gAcc[gP.cfPend + groupOff + g].max(1);
}
// Beam gj's group (its own) set off by the last step's contact force: the first
// lane to take the state from 1 to 2 sets the group off as master 0.
void cfFire(uint gj, uint groupOff, uint parity)
{
    if (gP.cfPend == 0u) return;
    int g = abs(asint(gStatic[gP.beamBase + gj * 3u].w));
    if (g == 0) return;
    uint at = gP.cfPend + groupOff + uint(g);
    if (gAcc[at].load() != 1) return;
    if (gAcc[at].exchange(2) != 1) return;
    gGroups[groupSlot(parity, groupOff, uint(g))].min(0u);
    gGroups[groupSlot(parity, groupOff, 0u)].min(0u);
}

// Beam j's group break as a node phase sees it: true for a beam after the first
// master that broke its group (step parity `parity`), which then drops the force it
// computed; when `mark`, any beam of such a group is marked broken.
bool groupDropsExt(uint j, uint beamOff, uint groupOff, uint parity, bool mark, uint master);

bool groupDrops(uint j, uint beamOff, uint groupOff, uint parity, bool mark)
{
    uint4 b0 = gStatic[gP.beamBase + j * 3u];
    int g = asint(b0.w);
    if (g == 0) return false;
    uint master = gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load();
    if ((b0.z & (TYPE_EXT << 8)) != 0u) return groupDropsExt(j, beamOff, groupOff, parity, mark, master);
    if (master == NO_GROUP) return false;
    if (mark)
    {
        float4 s1 = gBeams[j * 2u + 1u];
        if (s1.z == 0.0)
        {
            s1.z = 1.0;
            gBeams[j * 2u + 1u] = s1;
        }
    }
    return j - beamOff > master;
}

// The deform groups of extension x triggered (spec §19).
void extDeform(uint x, uint vehicle)
{
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    if (e0.y != 0u) gGroups[gP.deformBase + e0.y - 1u].max(1u);
    if (e0.z != 0u) gGroups[gP.deformBase + e0.z - 1u].max(1u);
}

// groupDrops for a beam with an extension record: its extra break groups too
// (spec §17), and its deform groups when a group breaks it.
bool groupDropsExt(uint j, uint beamOff, uint groupOff, uint parity, bool mark, uint master)
{
    uint x = gLists[gP.extIndex + j] - 1u;
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    uint4 e3 = gStatic[gP.extBase + x * 4u + 3u];
    if ((e0.x & EXT_GROUPS) != 0u)
    {
        if (asint(e3.z) != 0) master = min(master, gGroups[groupSlot(parity, groupOff, uint(abs(asint(e3.z))))].load());
        if (asint(e3.w) != 0) master = min(master, gGroups[groupSlot(parity, groupOff, uint(abs(asint(e3.w))))].load());
    }
    if (master == NO_GROUP) return false;
    if (mark)
    {
        float4 s1 = gBeams[j * 2u + 1u];
        if (s1.z == 0.0)
        {
            s1.z = 1.0;
            gBeams[j * 2u + 1u] = s1;
        }
        if ((e0.x & EXT_DEFORM) != 0u) extDeform(x, 0u);
    }
    return j - beamOff > master;
}

// A frame end's marks of the last step's group breaks, for the read-back (groupDrops' mark):
// a beam up to the group's master in beam order is pending (s1.w = 1), as a beam broken under
// its own load, torn for contact at its next update, as within a frame (the CPU's
// explicit_beams_torn.rs: a group's beams after its master torn in the pass, the others at the
// next), so how the steps are cut into frames changes nothing. Once a frame end, one call site
// each.
void groupDropsAtFrameEnd(uint j, uint beamOff, uint groupOff, uint parity)
{
    bool fresh = gBeams[j * 2u + 1u].z == 0.0;
    bool after = groupDrops(j, beamOff, groupOff, parity, true);
    if (fresh && !after && gBeams[j * 2u + 1u].z != 0.0) gBeams[j * 2u + 1u].w = 1.0;
}

// A node's force slot `f` plus its beam forces, in beam order as the CPU adds them.
// The beam phase writes each beam's force on each end into that end's incidence
// entry, so the entries of one node are contiguous and independent loads (no load
// waits on another's index).
float3 beamSum(float3 f, uint4 info, uint beamOff, uint groupOff, uint groupCount, uint parity, bool mark)
{
    uint base = gP.incBase + info.y;
    // Slot 0 of the vehicle's groups (no group has index 0) records that some
    // master broke in that step: without one no beam drops its force.
    if (groupCount == 0u || gGroups[groupSlot(parity, groupOff, 0u)].load() == NO_GROUP)
    {
        uint k = 0u;
        for (uint kGuard = 0u; k + 4u <= info.z; k += 4u)
        {
            EB_PASS(2, kGuard);
            float3 a = gBeams[base + k].xyz;
            float3 b = gBeams[base + k + 1u].xyz;
            float3 c = gBeams[base + k + 2u].xyz;
            float3 d = gBeams[base + k + 3u].xyz;
            f = sadd3(f, a);
            f = sadd3(f, b);
            f = sadd3(f, c);
            f = sadd3(f, d);
        }
        uint kStop = EB_END(3, k, info.z, 1u);
        for (; k < kStop; k++) f = sadd3(f, gBeams[base + k].xyz);
        return f;
    }
    for (uint k = 0u, kStop = EB_END(4, k, info.z, 1u); k < kStop; k++)
    {
        float3 push = gBeams[base + k].xyz;
        if (!groupDrops(gLists[info.y + k] >> 1, beamOff, groupOff, parity, mark)) f = sadd3(f, push);
    }
    return f;
}

// --- passes ---

#ifdef EB_VV_SOLO
// Vehicle contact (the solo frame): this lane's nodes left the frame bounds
// (relative, in workgroup memory) at some step.
// A node of vehicle v outside its frame bounds (world positions): reported at
// once (rare), so nothing stays live across the step loop.
groupshared float3 gsVvBounds[2];
#endif

#ifdef EB_AERO
void nodeUpdateFuse(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
static const uint FLAG_FUSEDRAG = 8u;
// Vehicle flags: it has triggers (key-driven commands), slide nodes.
static const uint FLAG_KEYS = 16u;
static const uint FLAG_SLIDES = 32u;
// Node info flags above this shift: the node's slide-force slot + 1 (0: none).
static const uint NODE_SLOT_SHIFT = 8u;
float3 slideTake(float3 f, uint slot1);
#endif

