// A free island's rigid modes, built as PhysX's native stage builds them on Apple GPUs
// (stressgpu/detail/StressMotionPairForest.cuh, StressMotionPairModes.cuh, StressMotionPair.cuh):
// exact float expansions for the positions and every decision, float pairs for the frame,
// stored in solver precision for the projection (blast_native.slang, blast_native_wide.slang).
// Its reference, with the construction in its docs, is stress-ref blast_bond/motion.rs.
//
// One group per island, after blast_topology.slang rebuilt the islands (its sort's dispatch:
// nothing runs on a frame without a break):
//   anchored        a live bond to a support: dimension 0, nothing else;
//   tree            levels from the root (the island's lowest node); a node's tree bond is its
//                   lowest-index live bond to the level before, its position the parent's plus
//                   that bond's o0 - o1 (exact). The target takes its tree from a union-find
//                   race; the exact sums make any tree's positions the same but for the
//                   rounding residue of the offsets;
//   closures        each live bond inside the island: p0 + (o0 - o1) - p1; one over 2^-17 of
//                   its coordinate scale is a moment; the lowest such bond is the closure;
//   rotations       3, or 1 about the closure's axis, or 0 when another significant closure
//                   is not collinear with it (exact products);
//   frame           the centre sum p / d_l^2 / sum 1 / d_l^2 and the Gram matrix of the basis
//                   (axis_k / d_a, (q x axis_k) / d_l), (0, e_j / d_l) in float pairs, scaled by
//                   its diagonal and factored (Cholesky); each node's q replaces its position.
//
// Float-pair and expansion arithmetic need round-to-nearest operations that are not
// reassociated, and a fused fma. The Metal library is compiled with precise math; WGSL and Metal
// source are compiled by the driver, on Apple GPUs with fast math, which folds TwoSum's error
// term to 0. So each error-free transformation passes its intermediates through `opaque`: their
// bits XORed with a 0 the compiler cannot see (params.opaque_zero). The value is unchanged, and
// float algebra cannot reach through an integer operation (a multiply by an unseen 1.0 does not
// do: the compiler factors it out of a sum and reassociates the rest). stress-gpu
// tests/blast_motion.rs checks every path.

import common;
import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> islands;
[[vk::binding(2, 0)]] RWStructuredBuffer<BlastNode> nodes;
[[vk::binding(3, 0)]] StructuredBuffer<uint> refs;
[[vk::binding(4, 0)]] StructuredBuffer<BlastBond> bonds;
[[vk::binding(5, 0)]] StructuredBuffer<uint> lists;
// Per node MOTION_WORDS float4: its exact position (x, y, z: 8 terms each), its level.
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> motion;

static const uint MOTION_WORDS = 7;
static const uint TERMS = 8;
static const uint UNSET = 0xFFFFFFFFu;
static const float LARGEST = 1.152921504606846976e18;    // 2^60
static const float SMALLEST = 7.888609052210118e-31;     // 2^-100

// ------------------------------------------------------------------ float pairs
// Joldes, Muller and Popescu (2017): AccurateDWPlusDW, DWTimesDW3, DWTimesFP3, DWDivDW2.

float opaque(float x)
{
    return asfloat(asuint(x) ^ params.opaque_zero);
}
void two_sum(float a, float b, out float s, out float e)
{
    s = opaque(a + b);
    float v = opaque(s - a);
    e = opaque(a - opaque(s - v)) + opaque(b - v);
}
void fast_two_sum(float a, float b, out float s, out float e)
{
    s = opaque(a + b);
    e = b - opaque(s - a);
}
void two_prod(float a, float b, out float p, out float e)
{
    p = opaque(a * b);
    e = fma(a, b, -p);
}
float2 p_add(float2 a, float2 b)
{
    float sh, sl, th, tl, vh, vl, zh, zl;
    two_sum(a.x, b.x, sh, sl);
    two_sum(a.y, b.y, th, tl);
    fast_two_sum(sh, sl + th, vh, vl);
    fast_two_sum(vh, tl + vl, zh, zl);
    return float2(zh, zl);
}
float2 p_neg(float2 a) { return -a; }
float2 p_sub(float2 a, float2 b) { return p_add(a, -b); }
float2 p_mul(float2 a, float2 b)
{
    float ch, cl, zh, zl;
    two_prod(a.x, b.x, ch, cl);
    float t = fma(a.y, b.x, fma(a.x, b.y, a.y * b.y));
    fast_two_sum(ch, cl + t, zh, zl);
    return float2(zh, zl);
}
float2 p_div(float2 a, float2 b)
{
    float th = a.x / b.x;
    float2 r = p_mul(b, float2(th, 0.0));
    float tl = ((a.x - r.x) + (a.y - r.y)) / b.x;
    float zh, zl;
    fast_two_sum(th, tl, zh, zl);
    return float2(zh, zl);
}
float2 p_sqrt(float2 a)
{
    if (!(a.x > 0.0))
        return float2(sqrt(a.x), 0.0);
    float s = sqrt(a.x);
    float p, e, zh, zl;
    two_prod(s, s, p, e);
    fast_two_sum(s, (((a.x - p) - e) + a.y) / (2.0 * s), zh, zl);
    return float2(zh, zl);
}
float2 p_abs(float2 a) { return a.x < 0.0 ? -a : a; }
bool p_finite(float2 a) { return !isinf(a.x) && !isnan(a.x) && !isinf(a.y) && !isnan(a.y); }
float2 p_inverse_square(float x)
{
    float p, e;
    two_prod(x, x, p, e);
    return p_div(float2(1.0, 0.0), float2(p, e));
}

struct P3
{
    float2 x;
    float2 y;
    float2 z;
};
P3 p3_sub(P3 a, P3 b) { P3 r; r.x = p_sub(a.x, b.x); r.y = p_sub(a.y, b.y); r.z = p_sub(a.z, b.z); return r; }
P3 p3_scale(P3 a, float2 k) { P3 r; r.x = p_mul(a.x, k); r.y = p_mul(a.y, k); r.z = p_mul(a.z, k); return r; }
P3 p3_cross(P3 a, P3 b)
{
    P3 r;
    r.x = p_sub(p_mul(a.y, b.z), p_mul(a.z, b.y));
    r.y = p_sub(p_mul(a.z, b.x), p_mul(a.x, b.z));
    r.z = p_sub(p_mul(a.x, b.y), p_mul(a.y, b.x));
    return r;
}
float2 p3_dot(P3 a, P3 b) { return p_add(p_add(p_mul(a.x, b.x), p_mul(a.y, b.y)), p_mul(a.z, b.z)); }
float2 p3_at(P3 a, uint i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }

// ------------------------------------------------------------------ exact expansions
// TERMS non-overlapping floats, the leading term first; each term normal and far from overflow
// (else the sum is reported inexact, as the target does).

struct Ex
{
    float t[TERMS];
};

Ex ex_zero()
{
    Ex r;
    for (uint i = 0; i < TERMS; i++)
        r.t[i] = 0.0;
    return r;
}
bool ex_representable(Ex a)
{
    for (uint i = 0; i < TERMS; i++)
        if (a.t[i] != 0.0)
        {
            float m = abs(a.t[i]);
            if (!(m <= LARGEST) || m < SMALLEST)
                return false;
        }
    return true;
}
// Canonical error-free distillation of v[0..count): VecSum passes move the exact total into the
// leading slots; terms past TERMS must vanish.
bool ex_distill(inout float v[2 * TERMS], uint count, out Ex out_ex)
{
    out_ex = ex_zero();
    bool exact = true;
    for (uint term = 0; term < count; term++)
    {
        for (uint pass = term; pass + 1 < count; pass++)
            for (uint i = count - 1; i > term; i--)
            {
                float s, e;
                two_sum(v[i - 1], v[i], s, e);
                v[i - 1] = s;
                v[i] = e;
            }
        if (term < TERMS)
            out_ex.t[term] = v[term];
        else if (v[term] != 0.0)
            exact = false;
    }
    for (uint pass = 0; pass + 1 < TERMS; pass++)
        for (uint i = 0; i + 1 < TERMS; i++)
            if (out_ex.t[i] == 0.0)
            {
                out_ex.t[i] = out_ex.t[i + 1];
                out_ex.t[i + 1] = 0.0;
            }
    return exact && ex_representable(out_ex);
}
Ex ex_add(Ex a, Ex b, inout bool exact)
{
    float v[2 * TERMS];
    for (uint i = 0; i < 2 * TERMS; i++)
        v[i] = 0.0;
    bool wide = false;
    for (uint i = 3; i < TERMS; i++)
        if (a.t[i] != 0.0 || b.t[i] != 0.0)
            wide = true;
    uint count;
    if (wide)
    {
        for (uint i = 0; i < TERMS; i++)
        {
            v[2 * i] = a.t[i];
            v[2 * i + 1] = b.t[i];
        }
        count = 2 * TERMS;
    }
    else if (a.t[2] != 0.0 || b.t[2] != 0.0)
    {
        v[0] = a.t[0]; v[1] = b.t[0]; v[2] = a.t[1]; v[3] = b.t[1]; v[4] = a.t[2]; v[5] = b.t[2];
        count = 6;
    }
    else
    {
        v[0] = a.t[0]; v[1] = b.t[0]; v[2] = a.t[1]; v[3] = b.t[1];
        count = 4;
    }
    Ex r;
    exact = ex_distill(v, count, r) && exact;
    return r;
}
Ex ex_neg(Ex a)
{
    for (uint i = 0; i < TERMS; i++)
        a.t[i] = -a.t[i];
    return a;
}
Ex ex_of(float f)
{
    Ex r = ex_zero();
    r.t[0] = f;
    return r;
}
// a - b of two floats, exactly.
Ex ex_difference(float a, float b, inout bool exact)
{
    if (isinf(a) || isnan(a) || isinf(b) || isnan(b))
    {
        exact = false;
        return ex_zero();
    }
    Ex r = ex_zero();
    two_sum(a, -b, r.t[0], r.t[1]);
    exact = ex_representable(r) && exact;
    return r;
}
bool ex_nonzero(Ex a) { return a.t[0] != 0.0; }
Ex ex_magnitude(Ex a) { return a.t[0] < 0.0 ? ex_neg(a) : a; }
Ex ex_scaled(Ex a, float power)
{
    for (uint i = 0; i < TERMS; i++)
        a.t[i] *= power;
    return a;
}
// |a| > |b| exactly (both given as magnitudes).
bool ex_greater(Ex a, Ex b)
{
    float v[2 * TERMS];
    for (uint i = 0; i < TERMS; i++)
    {
        v[2 * i] = a.t[i];
        v[2 * i + 1] = -b.t[i];
    }
    for (uint pass = 0; pass + 1 < 2 * TERMS; pass++)
        for (uint i = 2 * TERMS - 1; i > 0; i--)
        {
            float s, e;
            two_sum(v[i - 1], v[i], s, e);
            v[i - 1] = s;
            v[i] = e;
        }
    for (uint i = 0; i < 2 * TERMS; i++)
        if (v[i] != 0.0)
            return v[i] > 0.0;
    return false;
}
Ex ex_max(Ex a, Ex b) { return ex_greater(b, a) ? b : a; }
float2 ex_pair(Ex a)
{
    float tail = 0.0;
    for (uint i = TERMS - 1; i > 0; i--)
        tail += a.t[i];
    float s, e;
    two_sum(a.t[0], tail, s, e);
    return float2(s, e);
}

// Product expansion capacity of the collinearity test (the target's holds binary64 products;
// here each product is a float two-product, grown with zero elimination).
static const uint PRODUCT_TERMS = 64;
// a*b == c*d, exactly (Shewchuk grow-expansion); `exact` false when the capacity runs out.
bool ex_product_equal(Ex a, Ex b, Ex c, Ex d, inout bool exact)
{
    float e[PRODUCT_TERMS];
    for (uint z = 0; z < PRODUCT_TERMS; z++)
        e[z] = 0.0;
    uint count = 0;
    for (uint side = 0; side < 2; side++)
        for (uint i = 0; i < TERMS; i++)
            for (uint j = 0; j < TERMS; j++)
            {
                float x = side == 0 ? a.t[i] : c.t[i];
                float y = side == 0 ? b.t[j] : -d.t[j];
                if (x == 0.0 || y == 0.0)
                    continue;
                float p, pe;
                two_prod(x, y, p, pe);
                for (uint half = 0; half < 2; half++)
                {
                    float carry = half == 0 ? pe : p;
                    if (carry == 0.0)
                        continue;
                    uint next = 0;
                    for (uint k = 0; k < count; k++)
                    {
                        float s, h;
                        two_sum(carry, e[k], s, h);
                        if (h != 0.0)
                            e[next++] = h;
                        carry = s;
                    }
                    if (carry != 0.0)
                    {
                        if (next >= PRODUCT_TERMS)
                        {
                            exact = false;
                            return false;
                        }
                        e[next++] = carry;
                    }
                    count = next;
                }
            }
    return count == 0;
}

struct Ex3
{
    Ex x;
    Ex y;
    Ex z;
};
Ex3 ex3_add(Ex3 a, Ex3 b, inout bool exact)
{
    Ex3 r;
    r.x = ex_add(a.x, b.x, exact);
    r.y = ex_add(a.y, b.y, exact);
    r.z = ex_add(a.z, b.z, exact);
    return r;
}
Ex3 ex3_neg(Ex3 a)
{
    Ex3 r;
    r.x = ex_neg(a.x);
    r.y = ex_neg(a.y);
    r.z = ex_neg(a.z);
    return r;
}
bool ex3_nonzero(Ex3 a) { return ex_nonzero(a.x) || ex_nonzero(a.y) || ex_nonzero(a.z); }
bool ex3_collinear(Ex3 a, Ex3 b, inout bool exact)
{
    return ex_product_equal(a.y, b.z, a.z, b.y, exact) && ex_product_equal(a.z, b.x, a.x, b.z, exact)
        && ex_product_equal(a.x, b.y, a.y, b.x, exact);
}
P3 ex3_pair(Ex3 a)
{
    P3 r;
    r.x = ex_pair(a.x);
    r.y = ex_pair(a.y);
    r.z = ex_pair(a.z);
    return r;
}

// ------------------------------------------------------------------ the nodes' records

Ex load_ex(uint base)
{
    float4 a = motion[base];
    float4 b = motion[base + 1];
    Ex r;
    r.t[0] = a.x; r.t[1] = a.y; r.t[2] = a.z; r.t[3] = a.w;
    r.t[4] = b.x; r.t[5] = b.y; r.t[6] = b.z; r.t[7] = b.w;
    return r;
}
void store_ex(uint base, Ex e)
{
    motion[base] = float4(e.t[0], e.t[1], e.t[2], e.t[3]);
    motion[base + 1] = float4(e.t[4], e.t[5], e.t[6], e.t[7]);
}
Ex3 position(uint n)
{
    Ex3 r;
    r.x = load_ex(n * MOTION_WORDS);
    r.y = load_ex(n * MOTION_WORDS + 2);
    r.z = load_ex(n * MOTION_WORDS + 4);
    return r;
}
void set_position(uint n, Ex3 p)
{
    store_ex(n * MOTION_WORDS, p.x);
    store_ex(n * MOTION_WORDS + 2, p.y);
    store_ex(n * MOTION_WORDS + 4, p.z);
}
uint level(uint n) { return asuint(motion[n * MOTION_WORDS + 6].x); }
void set_level(uint n, uint l) { motion[n * MOTION_WORDS + 6] = float4(asfloat(l), 0.0, 0.0, 0.0); }

// motionOffset: a bond's o0 - o1, exactly.
Ex3 offset(BlastBond b, inout bool exact)
{
    Ex3 r;
    r.x = ex_difference(b.o0.x, b.o1.x, exact);
    r.y = ex_difference(b.o0.y, b.o1.y, exact);
    r.z = ex_difference(b.o0.z, b.o1.z, exact);
    return r;
}
Ex3 closure(BlastBond b, inout bool exact)
{
    return ex3_add(ex3_add(position(b.n0), offset(b, exact), exact), ex3_neg(position(b.n1)), exact);
}
// Whether a closure is a real moment: over 2^-17 of the cycle's coordinate scale (leading terms
// decide outside a 2^-20 band; inside it the exact values).
bool significant(BlastBond b, Ex3 c)
{
    Ex3 p = position(b.n0);
    Ex3 q = position(b.n1);
    float offsets = max(max(max(abs(b.o0.x), abs(b.o0.y)), max(abs(b.o0.z), abs(b.o1.x))), max(abs(b.o1.y), abs(b.o1.z)));
    float lead_scale = max(offsets, max(max(max(abs(p.x.t[0]), abs(p.y.t[0])), max(abs(p.z.t[0]), abs(q.x.t[0]))), max(abs(q.y.t[0]), abs(q.z.t[0]))));
    float lead_magnitude = max(max(abs(c.x.t[0]), abs(c.y.t[0])), abs(c.z.t[0]));
    float threshold = lead_scale * 7.62939453125e-6;    // 2^-17
    if (lead_magnitude > threshold * (1.0 + 9.5367431640625e-7))
        return true;
    if (lead_magnitude < threshold * (1.0 - 9.5367431640625e-7))
        return false;
    Ex scale = ex_max(ex_max(ex_magnitude(p.x), ex_magnitude(p.y)), ex_magnitude(p.z));
    scale = ex_max(scale, ex_max(ex_max(ex_magnitude(q.x), ex_magnitude(q.y)), ex_magnitude(q.z)));
    scale = ex_max(scale, ex_of(offsets));
    Ex magnitude = ex_max(ex_max(ex_magnitude(c.x), ex_magnitude(c.y)), ex_magnitude(c.z));
    return ex_greater(magnitude, ex_scaled(scale, 7.62939453125e-6));
}

// ------------------------------------------------------------------ group reductions

groupshared float2 g_pair[BLAST_THREADS];
groupshared uint g_word[BLAST_THREADS];
groupshared float2 g_frame[24];
groupshared uint g_rotations;

// Sum of float pairs over the group in a fixed order; on thread 0.
float2 pair_sum(uint tid, float2 v)
{
    g_pair[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = BLAST_THREADS / 2; k > 0; k >>= 1)
    {
        if (tid < k)
            g_pair[tid] = p_add(g_pair[tid], g_pair[tid + k]);
        GroupMemoryBarrierWithGroupSync();
    }
    float2 s = g_pair[0];
    GroupMemoryBarrierWithGroupSync();
    return s;
}

// Minimum of a word over the group, the same on every thread.
uint word_min(uint tid, uint v)
{
    g_word[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = BLAST_THREADS / 2; k > 0; k >>= 1)
    {
        if (tid < k)
            g_word[tid] = min(g_word[tid], g_word[tid + k]);
        GroupMemoryBarrierWithGroupSync();
    }
    return group_uniform(g_word[0], tid);
}

// ------------------------------------------------------------------ the frame record

void set_frame(uint k, uint w, uint value)
{
    uint at = blast_frame(params.node_count, k) + w / 4;
    uint4 v = islands[at];
    v[w % 4] = value;
    islands[at] = v;
}

[shader("compute")]
[numthreads(BLAST_THREADS, 1, 1)]
void blast_motion(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint k = group.x;
    uint4 range = group_uniform4(islands[k], tid);
    uint n0 = range.x;
    uint n1 = range.y;
    uint b0 = range.z;
    uint b1 = range.w;

    // Anchored: dimension 0.
    float anchor = 0.0;
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        BlastBond b = bonds[lists[q]];
        if (b.health > 0.0 && (!(nodes[b.n0].w_lin > 0.0) || !(nodes[b.n1].w_lin > 0.0)))
            anchor = 1.0;
    }
    if (tid == 0)
        for (uint w = 0; w < FRAME_WORDS; w++)
            set_frame(k, w, 0u);
    device_group_barrier();
    if (blast_group_sum(tid, float2(anchor, 0.0)).x > 0.0)
    {
        if (tid == 0)
            set_frame(k, F_ANCHORED, 1u);
        return;
    }

    // The tree, level by level from the root.
    bool exact = true;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        Ex3 zero;
        zero.x = ex_zero();
        zero.y = ex_zero();
        zero.z = ex_zero();
        set_position(i, zero);
        set_level(i, p == n0 ? 0u : UNSET);
    }
    device_group_barrier();
    for (uint l = 0;; l++)
    {
        float changed = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            if (level(i) != UNSET)
                continue;
            BlastNode n = nodes[i];
            uint best = UNSET;
            bool second = false;
            for (uint r = n.ref_begin; r < n.ref_end; r++)
            {
                uint ref = refs[r];
                uint j = ref & ~REF_SECOND;
                BlastBond b = bonds[j];
                uint other = (ref & REF_SECOND) != 0 ? b.n0 : b.n1;
                if (b.health > 0.0 && nodes[other].w_lin > 0.0 && level(other) == l && j < best)
                {
                    best = j;
                    second = (ref & REF_SECOND) != 0;
                }
            }
            if (best != UNSET)
            {
                BlastBond b = bonds[best];
                Ex3 d = offset(b, exact);
                // p1 = p0 + (o0 - o1).
                Ex3 at = second ? ex3_add(position(b.n0), d, exact) : ex3_add(position(b.n1), ex3_neg(d), exact);
                set_position(i, at);
                set_level(i, l + 1);
                changed = 1.0;
            }
        }
        device_group_barrier();
        if (blast_group_sum(tid, float2(changed, 0.0)).x == 0.0)
            break;
    }
    float unreached = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        if (level(lists[p]) == UNSET)
            unreached = 1.0;

    // Closures: the lowest live bond whose cycle closes on a real moment.
    uint seed = UNSET;
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        uint j = lists[q];
        BlastBond b = bonds[j];
        if (!(b.health > 0.0))
            continue;
        Ex3 c = closure(b, exact);
        if (ex3_nonzero(c) && significant(b, c))
            seed = min(seed, j);
    }
    seed = word_min(tid, seed);
    if (tid == 0)
    {
        g_rotations = seed == UNSET ? 3u : 1u;
        if (seed != UNSET)
        {
            bool e2 = true;
            P3 axis = ex3_pair(closure(bonds[seed], e2));
            float2 maximum = p_abs(axis.x);
            if (abs(axis.y.x) > maximum.x)
                maximum = p_abs(axis.y);
            if (abs(axis.z.x) > maximum.x)
                maximum = p_abs(axis.z);
            if (!(maximum.x > 0.0) || !p_finite(maximum))
                set_frame(k, F_ERROR, FRAME_ERROR_FRAME);
            axis = p3_scale(axis, p_div(float2(1.0, 0.0), maximum));
            float2 norm = p_sqrt(p3_dot(axis, axis));
            axis = p3_scale(axis, p_div(float2(1.0, 0.0), norm));
            g_frame[21] = axis.x;
            g_frame[22] = axis.y;
            g_frame[23] = axis.z;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // A second significant closure off the first's line leaves no rotation.
    float locked = 0.0;
    if (seed != UNSET)
    {
        bool e2 = true;
        Ex3 s = closure(bonds[seed], e2);
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            BlastBond b = bonds[lists[q]];
            if (!(b.health > 0.0))
                continue;
            Ex3 c = closure(b, exact);
            if (!ex3_nonzero(c) || !significant(b, c))
                continue;
            if (!ex3_collinear(s, c, exact))
                locked = 1.0;
        }
    }
    float2 flags = blast_group_sum(tid, float2(locked, exact ? 0.0 : 1.0));
    bool unconnected = blast_group_sum(tid, float2(unreached, 0.0)).x > 0.0;
    uint rotations = group_uniform(flags.x > 0.0 ? 0u : g_rotations, tid);
    P3 axis;
    axis.x = g_frame[21];
    axis.y = g_frame[22];
    axis.z = g_frame[23];

    // The frame: the centre, then the Gram matrix about it.
    float2 sums[4];
    for (uint z = 0; z < 4; z++)
        sums[z] = float2(0.0);
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        float2 w = p_inverse_square(nodes[i].w_lin);
        P3 at = ex3_pair(position(i));
        sums[0] = p_add(sums[0], w);
        sums[1] = p_add(sums[1], p_mul(at.x, w));
        sums[2] = p_add(sums[2], p_mul(at.y, w));
        sums[3] = p_add(sums[3], p_mul(at.z, w));
    }
    for (uint z = 0; z < 4; z++)
    {
        float2 s = pair_sum(tid, sums[z]);
        if (tid == 0)
            g_frame[z] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    P3 centre;
    centre.x = p_div(g_frame[1], g_frame[0]);
    centre.y = p_div(g_frame[2], g_frame[0]);
    centre.z = p_div(g_frame[3], g_frame[0]);
    bool centre_ok = g_frame[0].x > 0.0 && p_finite(g_frame[0]);
    GroupMemoryBarrierWithGroupSync();
    uint dimension = rotations + 3;
    float2 axial = rotations == 1 ? p3_dot(axis, axis) : float2(1.0, 0.0);
    float2 gram[21];
    for (uint z = 0; z < 21; z++)
        gram[z] = float2(0.0);
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        P3 q = p3_sub(ex3_pair(position(i)), centre);
        float2 w = p_inverse_square(n.w_lin);
        float2 angular = p_inverse_square(n.w_ang);
        n.position = float4(q.x.x, q.y.x, q.z.x, 0.0);
        nodes[i] = n;
        P3 moment[3];
        for (uint r = 0; r < 3; r++)
        {
            if (rotations == 1)
                moment[r] = p3_cross(q, axis);
            else
            {
                P3 e;
                e.x = float2(r == 0 ? 1.0 : 0.0, 0.0);
                e.y = float2(r == 1 ? 1.0 : 0.0, 0.0);
                e.z = float2(r == 2 ? 1.0 : 0.0, 0.0);
                moment[r] = p3_cross(q, e);
            }
        }
        for (uint row = 0; row < dimension; row++)
            for (uint col = 0; col <= row; col++)
            {
                float2 value = float2(0.0);
                if (row < rotations)
                    value = p_add(row == col ? p_mul(angular, axial) : float2(0.0), p_mul(w, p3_dot(moment[row], moment[col])));
                else if (col < rotations)
                    value = p_mul(w, p3_at(moment[col], row - rotations));
                else if (row == col)
                    value = w;
                uint t = row * (row + 1) / 2 + col;
                gram[t] = p_add(gram[t], value);
            }
    }
    for (uint z = 0; z < 21; z++)
    {
        float2 s = pair_sum(tid, gram[z]);
        if (tid == 0)
            gram[z] = s;
    }
    if (tid != 0)
        return;
    uint error = (unconnected ? FRAME_ERROR_TOPOLOGY : 0u) | (flags.y > 0.0 ? FRAME_ERROR_INEXACT : 0u);
    if (!centre_ok)
        error |= FRAME_ERROR_FRAME;
    float2 scale[6];
    float2 factor[21];
    for (uint z = 0; z < 6; z++)
        scale[z] = float2(0.0);
    for (uint z = 0; z < 21; z++)
        factor[z] = float2(0.0);
    for (uint i = 0; i < dimension && error == 0; i++)
    {
        float2 diagonal = gram[i * (i + 1) / 2 + i];
        if (!(diagonal.x > 0.0) || !p_finite(diagonal))
            error |= FRAME_ERROR_FRAME;
        else
            scale[i] = p_div(float2(1.0, 0.0), p_sqrt(diagonal));
    }
    for (uint row = 0; row < dimension && error == 0; row++)
        for (uint col = 0; col <= row && error == 0; col++)
        {
            float2 value = p_mul(p_mul(gram[row * (row + 1) / 2 + col], scale[row]), scale[col]);
            for (uint m = 0; m < col; m++)
                value = p_sub(value, p_mul(factor[row * (row + 1) / 2 + m], factor[col * (col + 1) / 2 + m]));
            if (row == col)
            {
                if (!(value.x > 0.0) || !p_finite(value))
                    error |= FRAME_ERROR_FRAME;
                value = p_sqrt(value);
            }
            else
                value = p_div(value, factor[col * (col + 1) / 2 + col]);
            factor[row * (row + 1) / 2 + col] = value;
        }
    set_frame(k, F_ROTATIONS, rotations);
    set_frame(k, F_ERROR, error | (islands[blast_frame(params.node_count, k) + F_ERROR / 4][F_ERROR % 4]));
    set_frame(k, F_AXIS, asuint(axis.x.x));
    set_frame(k, F_AXIS + 1, asuint(axis.y.x));
    set_frame(k, F_AXIS + 2, asuint(axis.z.x));
    for (uint i = 0; i < 6; i++)
        set_frame(k, F_SCALE + i, asuint(i < dimension && error == 0 ? scale[i].x : 0.0));
    for (uint t = 0; t < 21; t++)
        set_frame(k, F_FACTOR + t, asuint(t < dimension * (dimension + 1) / 2 && error == 0 ? factor[t].x : 0.0));
}

// ------------------------------------------------------------------ arithmetic self-test

// Exercises the arithmetic this file relies on (tests/blast_motion.rs): each row of `motion`
// holds inputs (x) and gets results.
[shader("compute")]
[numthreads(64, 1, 1)]
void blast_motion_selftest(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.node_count)
        return;
    float4 in0 = motion[2 * i];
    float s, e, p, pe;
    two_sum(in0.x, in0.y, s, e);
    two_prod(in0.z, in0.w, p, pe);
    motion[2 * i + 1] = float4(s, e, p, pe);
}
