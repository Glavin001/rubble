// Edge sample k's give (mpm_bake_shift): its column's mean outward
// displacement across its side, from the sorted edge lists (mpm_bake_edge_*).
float mpmEdgeGive(uint k, uint2 ij, uint axis, MpmParams p, Frame f)
{
    uint ni = p.window.z, nj = p.window.w;
    uint e = mpmEdgeList(int(ij.x), int(ij.y), p);
    uint first = ALOAD(gMpmCells[3u * e + 1u]), n = ALOAD(gMpmCells[3u * e]);
    float sum = 0.0, quanta = 0.0;
    for (uint r = first; r < first + n; r++)
    {
        MpmParticle q = gMpmParticles[gMpmSorted[r]];
        if (mpmParticleDead(q) || (asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) continue;
        float3 xl = frameToLocal(f, q.xm.xyz);
        float w = float(asuint(q.f1.w));
        sum += w * (axis == 0u ? xl.x - q.c1.w : xl.z - q.c2.w);
        quanta += w;
    }
    // Only outward: material moving out past the side's plane has moved the
    // ground with it (it cannot enter the ground); moving in, it left the
    // ground where it was.
    float mean = quanta > 0.0 ? sum / quanta : 0.0;
    bool low = k < 2u * nj ? k < nj : (k - 2u * nj) < ni;
    return low ? min(mean, 0.0) : max(mean, 0.0);
}

// Edge sample k of a window: its (sample, axis).
uint2 mpmEdgeSample(uint k, MpmParams p, out uint axis)
{
    uint ni = p.window.z, nj = p.window.w;
    if (k < 2u * nj)
    {
        axis = 0u;
        return uint2(k < nj ? 0u : ni - 1u, k % nj);
    }
    uint m = k - 2u * nj;
    axis = 2u;
    return uint2(m % ni, m < ni ? 0u : nj - 1u);
}

// ---- growth: the ground beyond the old walls (TerrainPlan::open) ----------
//
// An open area's absorbing walls give: the material against a side moves out
// past its plane by the ground's displacement there, over the samples beyond
// the window (its bake measures positions in the unactivated ground's frame,
// mpmGroundShift). When the area grows over those samples, the ground beyond
// becomes material, and it had moved too: it is activated with that
// displacement continuing outward, the old wall's measured give at the old
// face falling linearly to zero at the grown area's far face on that side
// (the uniform strain the bake already takes between opposite sides). The old
// material stays where the walls let it go, the new columns stand clear of
// it (nothing is counted twice), and both bake as material moved by the same
// field. Sides that stay walls keep their give and face displacements.
// Volume is untouched; the positions of the new particles carry the give, F
// does not (the ground's elastic strain under that give, about give/D, is not
// added: an unstressed start, as every activation's lateral state).

// gMpmParams[0].area.w bit (MPM_DOMAIN_KEEP + side, sides low x, high x, low
// z, high z): the selected old area's side stays a wall of the grown area.
static const uint MPM_DOMAIN_KEEP = 4u;

// Per edge sample of the selected old area (before the grown area
// activates): its give (0 on a side that stays a wall), into that sample's
// column record of the old area, word 1 across an x side, word 2 across a z
// side (the activation's ranges there are spent).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_grow_shift(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    uint ni = p.window.z, nj = p.window.w;
    if (k >= 2u * (ni + nj)) return;
    uint side = k < 2u * nj ? (k < nj ? 0u : 1u) : ((k - 2u * nj) < ni ? 2u : 3u);
    bool kept = ((gMpmParams[0].area.w >> (MPM_DOMAIN_KEEP + side)) & 1u) != 0u;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    uint axis;
    uint2 ij = mpmEdgeSample(k, p, axis);
    float give = kept ? 0.0 : mpmEdgeGive(k, ij, axis, p, f);
    ASTORE(gMpmColumns[mpmCol(p, ij.y * ni + ij.x) + (axis == 0u ? 1u : 2u)], asuint(give));
}

// A grown area's new particle at local xl (on sample ij of window p): the
// ground's give there, from each old area's sides it grew past. Zero inside
// the old windows and beyond the new far faces.
float3 mpmGrowGive(float3 xl, uint2 ij, MpmParams p, HeightFieldRef hf)
{
    float3 u = float3(0, 0, 0);
    uint mask = gMpmParams[0].area.z;
    uint4 nw = p.window;
    float nlx = hf.x0 + (float(nw.x) - 0.5) * hf.dx, nhx = hf.x0 + (float(nw.x + nw.z) - 0.5) * hf.dx;
    float nlz = hf.z0 + (float(nw.y) - 0.5) * hf.dz, nhz = hf.z0 + (float(nw.y + nw.w) - 0.5) * hf.dz;
    for (uint o = 0u; o < MPM_MAX_AREAS; o++)
    {
        if ((mask & (1u << o)) == 0u) continue;
        MpmParams q = mpmArea(o);
        if (q.boundary.z == 0u) continue;
        uint4 w = q.window;
        float lx = hf.x0 + (float(w.x) - 0.5) * hf.dx, hx = hf.x0 + (float(w.x + w.z) - 0.5) * hf.dx;
        float lz = hf.z0 + (float(w.y) - 0.5) * hf.dz, hz = hf.z0 + (float(w.y + w.w) - 0.5) * hf.dz;
        uint tj = uint(clamp(int(ij.y) - int(w.y), 0, int(w.w) - 1));
        uint ti = uint(clamp(int(ij.x) - int(w.x), 0, int(w.z) - 1));
        // Only across a side's own span: beyond its ends (the corners past
        // two sides) no wall of the old area stood, and the ground there
        // did not give.
        bool alongX = xl.z >= lz && xl.z <= hz, alongZ = xl.x >= lx && xl.x <= hx;
        // Past an x side it grew over: that row's give, fading to the far face.
        if (alongX && nw.x + nw.z > w.x + w.z && xl.x > hx)
        {
            float g = asfloat(ALOAD(gMpmColumns[mpmCol(q, tj * w.z + w.z - 1u) + 1u]));
            u.x += g * max(0.0, 1.0 - (xl.x - hx) / (nhx - hx));
        }
        if (alongX && nw.x < w.x && xl.x < lx)
        {
            float g = asfloat(ALOAD(gMpmColumns[mpmCol(q, tj * w.z) + 1u]));
            u.x += g * max(0.0, 1.0 - (lx - xl.x) / (lx - nlx));
        }
        if (alongZ && nw.y + nw.w > w.y + w.w && xl.z > hz)
        {
            float g = asfloat(ALOAD(gMpmColumns[mpmCol(q, (w.w - 1u) * w.z + ti) + 2u]));
            u.z += g * max(0.0, 1.0 - (xl.z - hz) / (nhz - hz));
        }
        if (alongZ && nw.y < w.y && xl.z < lz)
        {
            float g = asfloat(ALOAD(gMpmColumns[mpmCol(q, ti) + 2u]));
            u.z += g * max(0.0, 1.0 - (lz - xl.z) / (lz - nlz));
        }
    }
    return u;
}

// The walls' displacement at local point xl (the bed heightfield's frame), in
// x and z: each side's edge columns' (mpm_bake_shift), interpolated linearly
// between opposite sides along the window's row or column through xl.
float2 mpmGroundShift(float3 xl, HeightFieldRef hf, MpmParams p)
{
    int i0 = int(p.window.x), j0 = int(p.window.y);
    int ni = int(p.window.z), nj = int(p.window.w);
    float2 u = (xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
    int ti = clamp(int(round(u.x)) - i0, 0, ni - 1), tj = clamp(int(round(u.y)) - j0, 0, nj - 1);
    uint base = p.boundary.y;
    float sx = ni > 1 ? clamp((u.x - float(i0)) / float(ni - 1), 0.0, 1.0) : 0.5;
    float sz = nj > 1 ? clamp((u.y - float(j0)) / float(nj - 1), 0.0, 1.0) : 0.5;
    float lx = gMpmCoupled[base + uint(tj)].x, hx = gMpmCoupled[base + uint(nj + tj)].x;
    float lz = gMpmCoupled[base + uint(2 * nj + ti)].x, hz = gMpmCoupled[base + uint(2 * nj + ni + ti)].x;
    return float2(lerp(lx, hx, sx), lerp(lz, hz, sz));
}

// The share of a particle that is still the heightfield it came from (w
// above), and where its tent goes: its home sample plus its displacement, in
// sample coordinates. 0 for a particle not made by activation.
float mpmHomeShare(MpmParticle q, float3 xl, HeightFieldRef hf, MpmParams p, out float2 uHome)
{
    float2 d = float2(hf.dx, hf.dz), o = float2(hf.x0, hf.z0);
    uHome = (xl.xz - o) / d;
    if ((asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return 0.0;
    float2 x0 = float2(q.c1.w, q.c2.w);
    uHome = floor((x0 - o) / d + 0.5) + (xl.xz - x0) / d;
    return max(0.0, 1.0 - length(xl.xz - x0) / (p.limits.y * p.column.z));
}

// PCG integer hash (Jarzynski & Olano 2020).
uint mpmPcgHash(uint v)
{
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// A hash as a uniform offset in [0, 1) (24 bits, exact in f32).
float mpmDither(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }

// The tent split of `quanta` at sample coordinates u: the sample at or below u
// (s0) and its parts for (i, j), (i+1, j), (i, j+1), (i+1, j+1). Split along x
// first, then each half along z: whole quanta, never negative, summing exactly
// to `quanta`, each within one quantum of its exact share. A zero weight gets
// exactly nothing, so an unmoved particle stays whole.
// Each split rounds with a dither from `seed` (the particle's or parcel's
// slot): floor(share + r), r uniform in [0, 1), so its expected value is the
// exact share. Rounding half up instead errs the same way for every particle
// moved alike, and a deep column of N particles that moved together would be
// off by up to N/2 quanta, a height step between neighbours (at the browser
// demo's scale, about 240 particles a column, up to about 2 m). Deterministic: the seed is
// the slot, not the thread order.
void mpmTentSplit(float2 u, uint quanta, uint seed, out int2 s0, out uint4 parts)
{
    float2 b = floor(u);
    float2 t = u - b;
    s0 = int2(b);
    uint hx = mpmPcgHash(seed);
    float rx = mpmDither(hx), rz = mpmDither(mpmPcgHash(hx));
    uint q1 = min(uint(floor(float(quanta) * t.x + rx)), quanta);
    uint q0 = quanta - q1;
    uint q01 = min(uint(floor(float(q0) * t.y + rz)), q0);
    uint q11 = min(uint(floor(float(q1) * t.y + rz)), q1);
    parts = uint4(q0 - q01, q1 - q11, q01, q11);
}

int2 mpmTentSample(int2 s0, uint c) { return s0 + int2(int(c & 1u), int(c >> 1u)); }

// Quadratic B-spline weights at sample coordinate u (P2G's kernel, in sample
// spacings) over samples base, base + 1, base + 2.
float3 mpmBsplineWeights(float u, out int base)
{
    float b = floor(u - 0.5);
    float f = u - b;
    base = int(b);
    return float3(0.5 * (1.5 - f) * (1.5 - f), 0.75 - (f - 1.0) * (f - 1.0), 0.5 * (f - 0.5) * (f - 0.5));
}

// Whole-quanta split of q in the proportions w (summing to 1): the third part
// first, then the first two, each rounded with a dither from the chain h.
// Never negative, summing exactly to q; a zero weight gets nothing.
uint3 mpmSplit3(uint q, float3 w, inout uint h)
{
    h = mpmPcgHash(h);
    uint q2 = min(uint(floor(float(q) * w.z + mpmDither(h))), q);
    uint rest = q - q2;
    float ab = w.x + w.y;
    h = mpmPcgHash(h);
    uint q0 = ab > 0.0 ? min(uint(floor(float(rest) * (w.x / ab) + mpmDither(h))), rest) : 0u;
    return uint3(q0, rest - q0, q2);
}

// The B-spline split of `quanta` at sample coordinates u: parts[3k + m] for
// sample s0 + (m, k), split along x, then each third along z (dithered from
// `seed`, as mpmTentSplit).
void mpmBsplineSplit(float2 u, uint quanta, uint seed, out int2 s0, out uint parts[9])
{
    int bx, bz;
    float3 wx = mpmBsplineWeights(u.x, bx), wz = mpmBsplineWeights(u.y, bz);
    s0 = int2(bx, bz);
    uint h = mpmPcgHash(seed ^ 0xB5297A4Du);
    uint3 qx = mpmSplit3(quanta, wx, h);
    for (uint m = 0u; m < 3u; m++)
    {
        uint3 qz = mpmSplit3(qx[m], wz, h);
        parts[m] = qz.x;
        parts[3u + m] = qz.y;
        parts[6u + m] = qz.z;
    }
}

// `quanta` baked onto sample s: a window sample's column, or, outside the
// window (material that flowed past it inside the region's box), the field's
// deposit accumulator, as a landed parcel. Off the field's footprint, escaped.
void mpmBakeOnto(int2 s, uint quanta, float yl, MpmParams p, HeightFieldRef hf)
{
    if (quanta == 0u) return;
    if (s.x < 0 || s.y < 0 || s.x >= int(hf.nx) || s.y >= int(hf.nz))
    {
        AADD2(gMpmCounters[MPM_C_ESCAPED], quanta);
        return;
    }
    // Material below the unactivated ground beside the window (at local height
    // yl) is the region's own column pressed into that ground, which yields as
    // the absorbing walls do but stays put as a heightfield: it bakes into the
    // nearest window sample (the kernel reflects at the wall), never onto the
    // ground's top. Only what lies above that ground spills onto it.
    int2 wlo = int2(p.window.xy), whi = int2(p.window.xy + p.window.zw) - 1;
    if (any(s < wlo) || any(s > whi))
    {
        if (yl < hfHeight(hf, uint(s.x), uint(s.y))) s = clamp(s, wlo, whi);
    }
    int ti = s.x - int(p.window.x), tj = s.y - int(p.window.y);
    if (ti >= 0 && tj >= 0 && ti < int(p.window.z) && tj < int(p.window.w))
    {
        AADD2(gMpmColumns[mpmCol(p, uint(tj) * p.window.z + uint(ti)) + 3u], quanta);
        return;
    }
    uint at = uint(s.y) * hf.nx + uint(s.x);
    if (at < asuint(p.couple.w))
    {
        AADD2(gMpmDeposit[at], quanta);
        AADD2(gMpmCounters[MPM_C_DEPOSITED], quanta);
    }
    else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_FIELD);
}

// Where a particle's volume bakes: where it stands, unless it covers a dynamic
// body coupled to the region (it lies above part of the body). A heightfield
// cannot hold a buried body, so that sand is pushed aside: to just outside the
// body's footprint, straight away from its centre, far enough that the splat's
// samples (within one `spacing`, the field's sample spacing) lie beyond it.
// The body is left in a pit as deep as it went, and the volume is kept.
// Returns whether it moved x.
bool mpmBakeColumn(inout float3 x, MpmParams p, float spacing)
{
    uint slots = min(ALOAD(gMpmCouple[0]), asuint(p.couple.y));
    for (uint k = 0u; k < slots; k++)
    {
        Collider col = gColliders[asuint(gMpmCoupled[4u * k].w)];
        BodyState b = gBodies[col.body];
        if (!bodyIsDynamic(b)) continue;
        float3 lo, hi;
        mpmColliderBounds(col, b, p, lo, hi);
        if (any(x.xz < lo.xz) || any(x.xz > hi.xz) || x.y < lo.y) continue;
        // Does the vertical line below x meet the body? Sampled from the
        // body's bottom up to x (or its top).
        float top = min(x.y, hi.y);
        bool covers = false;
        for (uint j = 0u; j <= 8u && !covers; j++)
        {
            float3 at = float3(x.x, lerp(lo.y, top, float(j) / 8.0), x.z);
            float3 n;
            covers = mpmColliderSdf(col, b, at, n) < 0.0;
        }
        if (!covers) continue;
        float3 c = colliderFrame(col, b).p;
        float2 d = x.xz - c.xz;
        float len = length(d);
        float2 dir = len > 1e-6 ? d / len : float2(1.0, 0.0);
        float reach = 0.5 * length(hi.xz - lo.xz) + spacing;
        x = float3(c.x + dir.x * reach, x.y, c.z + dir.y * reach);
        return true;
    }
    return false;
}

// `quanta` of material the MPM moved, at sample coordinates u: the B-spline
// split onto its nine samples (mpmBakeOnto).
void mpmBakeMoved(float2 u, uint quanta, uint seed, float yl, MpmParams p, HeightFieldRef hf)
{
    int2 s0;
    uint parts[9];
    mpmBsplineSplit(u, quanta, seed, s0, parts);
    for (uint k = 0u; k < 9u; k++) mpmBakeOnto(s0 + int2(int(k % 3u), int(k / 3u)), parts[k], yl, p, hf);
}

// Per particle: its quanta split between the heightfield it came from and the
// material the MPM moved (see above).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_bin(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmSelArea();
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    // The selected area's particles only; with other areas still open they
    // leave the pool as baked (their slots are reused once it empties).
    if (!mpmBakeOwns(q.xm.xyz)) return;
    if ((gMpmParams[0].area.w & MPM_DOMAIN_CLOSE) == 0u)
    {
        gMpmParticles[i].f2.w = asfloat(asuint(q.f2.w) | MPM_PARTICLE_DEAD);
        AADD2(gMpmCounters[MPM_C_LIVE], 0xFFFFFFFFu);
    }
    float3 x = q.xm.xyz;
    bool pushed = mpmBakeColumn(x, p, max(hf.dx, hf.dz));
    float3 xl = frameToLocal(f, x);
    // On the absorbing ground, in the unactivated ground's frame.
    if ((p.boundary.x & MPM_BOUNDARY_ABSORBING) != 0u) xl.xz -= mpmGroundShift(xl, hf, p);
    uint quanta = asuint(q.f1.w);
    // The dithers' seed: the slot from the area's first.
    uint seed = i - ALOAD(gMpmCounters[MPM_C_AREA_FIRST + gMpmParams[0].area.y]);
    int2 s0;
    uint4 parts;
    if (pushed)
    {
        mpmTentSplit((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz), quanta, seed, s0, parts);
        for (uint c = 0u; c < 4u; c++) mpmBakeOnto(mpmTentSample(s0, c), parts[c], xl.y, p, hf);
        return;
    }
    float2 uHome;
    float w = mpmHomeShare(q, xl, hf, p, uHome);
    // The heightfield's share (all of it at rest: floor(q + r) = q).
    uint home = min(uint(floor(float(quanta) * w + mpmDither(mpmPcgHash(~seed)))), quanta);
    mpmTentSplit(uHome, home, seed, s0, parts);
    for (uint c = 0u; c < 4u; c++) mpmBakeOnto(mpmTentSample(s0, c), parts[c], xl.y, p, hf);
    if (home == quanta) return;
    mpmBakeMoved((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz), quanta - home, seed, xl.y, p, hf);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_raise(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmSelArea();
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint r0, r1;
    mpmWindowRecords(hf, p, r0, r1);
    uint r = r0 + mpmIndex(tid);
    if (r <= r1) mpmAdjustRecord(r, true, p, hf);
}

// After mpm_bake_raise: the field's max height covers the baked samples, and
// the region holds no particles.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_bake_header(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmSelArea();
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint b = ALOAD(gMpmCounters[MPM_C_BAKE_MAX]);
    if (b != 0u)
    {
        float top = asfloat((b & 0x80000000u) != 0u ? (b & 0x7FFFFFFFu) : ~b);
        uint4 dims = gHulls[hf.base + 1u];
        dims.w = asuint(max(asfloat(dims.w), top));
        gHulls[hf.base + 1u] = dims;
    }
    if ((gMpmParams[0].area.w & MPM_DOMAIN_CLOSE) != 0u)
    {
        ASTORE(gMpmCounters[MPM_C_PARTICLES], 0u);
        ASTORE(gMpmCounters[MPM_C_LIVE], 0u);
    }
    ASTORE(gMpmCounters[MPM_C_BAKE_MAX], 0u);
}

// ---- rigid coupling -----------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_couple_clear(uint3 tid : SV_DispatchThreadID)
{
    uint c = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (c == 0u)
    {
        ASTORE(gMpmCouple[0], 0u);
        // Once per frame: the frame's top speeds, over all its substeps.
        ASTORE(gMpmCounters[MPM_C_MAX_SPEED], 0u);
        ASTORE(gMpmCounters[MPM_C_BODY_SPEED], 0u);
        for (uint a = 0u; a < MPM_MAX_AREAS; a++)
        {
            ASTORE(gMpmCounters[MPM_C_AREA_SPEED + a], 0u);
            ASTORE(gMpmCounters[MPM_C_AREA_BODY + a], 0u);
            ASTORE(gMpmCounters[MPM_C_AREA_TOUCH + a], 0u);
        }
    }
    // The open areas' block cells (the plan dispatches as many as they use).
    if (c < p.counts.z) ASTORE(gMpmCouple[4u + c * (1u + MPM_CELL_COLLIDERS)], 0u);
}

// Per rigid collider: a coupling slot and a place in every block cell of the
// region its padded bounds reach (planes and heightfields are the bed's).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_couple_bin(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    MpmParams p = mpmParams();
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE || col.shape == SHAPE_HEIGHTFIELD) return;
    BodyState b = gBodies[col.body];
    float3 lo, hi;
    mpmColliderBounds(col, b, p, lo, hi);
    float block = p.origin.w * float(MPM_BLOCK);
    uint slot = MPM_NO_SLOT;
    for (uint a = 0u; a < p.area.x; a++)
    {
        MpmParams pa = mpmArea(a);
        if (pa.dims.x == 0u) continue;
        int3 k0 = int3(floor((lo - pa.origin.xyz) / block));
        int3 k1 = int3(floor((hi - pa.origin.xyz) / block));
        int3 blocks = int3(pa.dims.xyz / MPM_BLOCK);
        k0 = max(k0, int3(0, 0, 0));
        k1 = min(k1, blocks - 1);
        if (any(k1 < k0)) continue;
        // An area cannot settle while a dynamic body in it still moves: its
        // fastest surface point (linear plus spin over its bounds' half diagonal).
        if (b.invMass > 0.0)
        {
            float reach = 0.5 * length(hi - lo);
            uint speed = asuint(length(b.linearVel) + length(b.angularVel) * reach);
            AMAX2(gMpmCounters[MPM_C_BODY_SPEED], speed);
            if (p.area.x > 1u) AMAX2(gMpmCounters[MPM_C_AREA_BODY + a], speed);
        }
        if (slot == MPM_NO_SLOT)
        {
            AADD3(gMpmCouple[0], 1u, slot);
            if (slot >= asuint(p.couple.y))
            {
                AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_COUPLED);
                return;
            }
            gMpmCoupled[4u * slot] = float4(0, 0, 0, asfloat(i));
            gMpmCoupled[4u * slot + 1u] = float4(0, 0, 0, 0);
            gMpmCoupled[4u * slot + 2u] = float4(b.linearVel, 0);
            gMpmCoupled[4u * slot + 3u] = float4(b.angularVel, 0);
        }
        for (int z = k0.z; z <= k1.z; z++)
        for (int y = k0.y; y <= k1.y; y++)
        for (int x = k0.x; x <= k1.x; x++)
        {
            uint cell = mpmLatticeCell(int3(x, y, z), pa);
            if (cell == MPM_NO_BLOCK) continue;
            uint base = 4u + cell * (1u + MPM_CELL_COLLIDERS);
            uint at;
            AADD3(gMpmCouple[base], 1u, at);
            if (at < MPM_CELL_COLLIDERS) ASTORE(gMpmCouple[base + 1u + at], slot);
            else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_CELL);
        }
    }
}

groupshared float3 gsMpmJ[64];
groupshared float3 gsMpmL[64];
groupshared float gsMpmM[64];
groupshared uint gsMpmTouched[64];

// One workgroup per coupling slot (dynamic bodies): the substep's reaction on
// the body, minus the momentum its contacts gave the grid nodes, in a fixed order
// (blocks in lattice order, a lane per node, then a tree sum).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_couple_gather(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint slot = gid.x;
    uint l = lid.x;
    MpmParams p = mpmParams();
    bool live = slot < min(ALOAD(gMpmCouple[0]), asuint(p.couple.y));
    float3 J = float3(0, 0, 0), L = float3(0, 0, 0);
    float M = 0.0;
    // The areas whose material the body's contacts hold (bit per slot).
    uint touched = 0u;
    if (live)
    {
        uint ci = asuint(gMpmCoupled[4u * slot].w);
        Collider col = gColliders[ci];
        BodyState b = gBodies[col.body];
        if (bodyIsDynamic(b))
        {
            float3 lo, hi;
            mpmColliderBounds(col, b, p, lo, hi);
            float block = p.origin.w * float(MPM_BLOCK);
            // Each open area its bounds reach, in slot order, its blocks in
            // lattice order (in the area's frame).
            for (uint a = 0u; a < p.area.x; a++)
            {
                MpmParams pa = mpmArea(a);
                if (pa.dims.x == 0u) continue;
                int3 kb = mpmAreaOffset(pa) >> 2;
                int3 blocks = int3(pa.dims.xyz / MPM_BLOCK);
                int3 k0 = max(int3(floor((lo - pa.origin.xyz) / block)), int3(0, 0, 0));
                int3 k1 = min(int3(floor((hi - pa.origin.xyz) / block)), blocks - 1);
                for (int z = k0.z; z <= k1.z; z++)
                for (int y = k0.y; y <= k1.y; y++)
                for (int x = k0.x; x <= k1.x; x++)
                {
                    uint bk = mpmFindBlock(mpmBlockKey(int3(x, y, z) + kb), p);
                    if (bk == MPM_NO_BLOCK) continue;
                    float4 rec = gMpmGrid[2u * (bk * MPM_BLOCK_NODES + l) + 1u];
                    if (rec.w != float(slot)) continue;
                    int3 g = int3(x, y, z) * 4 + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
                    float3 xw = pa.origin.xyz + float3(g) * pa.origin.w;
                    J -= rec.xyz;
                    L -= cross(xw - b.position, rec.xyz);
                    float m = gMpmGrid[2u * (bk * MPM_BLOCK_NODES + l)].w;
                    M += m;
                    if (m > 0.0) touched |= 1u << a;
                }
            }
        }
    }
    gsMpmJ[l] = J;
    gsMpmL[l] = L;
    gsMpmM[l] = M;
    gsMpmTouched[l] = touched;
    GROUP_SYNC();
    for (uint d = 32u; d > 0u; d >>= 1u)
    {
        if (l < d)
        {
            gsMpmJ[l] += gsMpmJ[l + d];
            gsMpmL[l] += gsMpmL[l + d];
            gsMpmM[l] += gsMpmM[l + d];
            gsMpmTouched[l] |= gsMpmTouched[l + d];
        }
        GROUP_SYNC();
    }
    if (live && l == 0u)
    {
        float4 a = gMpmCoupled[4u * slot], c = gMpmCoupled[4u * slot + 1u];
        Collider col = gColliders[asuint(a.w)];
        BodyState b = gBodies[col.body];
        // The exchange is perfectly inelastic: the contact nodes and the body
        // take the share a = Me/(Me + m) of the impulse that would hold the nodes
        // to an immovable body (m: the contact nodes' mass). For a translating
        // body (Me = M) that is exactly their common velocity, stable at any mass
        // ratio. Me is the body's effective mass at the farthest reach of its
        // bounds (a lower bound over its contacts), so spin cannot overshoot.
        float share = 1.0;
        if (bodyIsDynamic(b))
        {
            float3 lo, hi;
            mpmColliderBounds(col, b, p, lo, hi);
            float reach = 0.5 * length(hi - lo);
            float invI = max(b.invInertia.x, max(b.invInertia.y, b.invInertia.z));
            float invMe = b.invMass + reach * reach * invI;
            share = 1.0 / (1.0 + gsMpmM[0] * invMe);
            // The local settle rule: an area stays open while a body that
            // touches its material moves (one above it, not touching, does not).
            uint speed = asuint(length(b.linearVel) + length(b.angularVel) * reach);
            for (uint t = gsMpmTouched[0]; t != 0u; t &= t - 1u)
                AMAX2(gMpmCounters[MPM_C_AREA_TOUCH + firstbitlow(t)], speed);
        }
        float3 Jt = share * gsMpmJ[0], Lt = share * gsMpmL[0];
        gMpmCoupled[4u * slot] = float4(a.xyz + Jt, a.w);
        gMpmCoupled[4u * slot + 1u] = float4(c.xyz + Lt, share);
        if (bodyIsDynamic(b))
        {
            float3 local = quatRotate(quatConjugate(b.orientation), Lt);
            float3 dw = quatRotate(b.orientation, local * b.invInertia);
            float3 dv = Jt * b.invMass + float3(0.0, p.step.x * p.step.y, 0.0);
            gMpmCoupled[4u * slot + 2u] += float4(dv, 0.0);
            gMpmCoupled[4u * slot + 3u] += float4(dw, 0.0);
        }
    }
}

// Per grid node, after mpm_couple_gather: a contact node takes its body's share
// of the holding impulse (the body took the reaction in the gather).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_couple_apply(uint3 tid : SV_DispatchThreadID)
{
    uint n = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (n >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z) * MPM_BLOCK_NODES) return;
    float4 rec = gMpmGrid[2u * n + 1u];
    if (rec.w < 0.0) return;
    float4 node = gMpmGrid[2u * n];
    float share = gMpmCoupled[4u * uint(rec.w) + 1u].w;
    gMpmGrid[2u * n] = float4(node.xyz + rec.xyz * (share / node.w), node.w);
}

// Per coupling slot of a dynamic body: its frame's reaction as one impulse record
// (linear through the centre, angular about it) for the rigid step, keyed after
// the host's records (by collider) so a body's records sum in a fixed order.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_couple_emit(uint3 tid : SV_DispatchThreadID)
{
    uint slot = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (slot >= min(ALOAD(gMpmCouple[0]), asuint(p.couple.y))) return;
    float4 a = gMpmCoupled[4u * slot], c = gMpmCoupled[4u * slot + 1u];
    uint ci = asuint(a.w);
    uint body = gColliders[ci].body;
    if (!bodyIsDynamic(gBodies[body]) || (all(a.xyz == float3(0, 0, 0)) && all(c.xyz == float3(0, 0, 0)))) return;
    uint r;
    AADD3(gScratchAtomics[IMPULSE_COUNT], 1u, r);
    if (r >= IMPULSE_CAP) { AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_IMPULSES); return; }
    uint w = IMPULSE_BASE + IMPULSE_WORDS * r;
    ASTORE(gScratchAtomics[w], body);
    ASTORE(gScratchAtomics[w + 1u], ((0x300000u + ci) << 8) | MPM_IMPULSE_LINEAR_ANGULAR);
    ASTORE(gScratchAtomics[w + 2u], asuint(a.x));
    ASTORE(gScratchAtomics[w + 3u], asuint(a.y));
    ASTORE(gScratchAtomics[w + 4u], asuint(a.z));
    ASTORE(gScratchAtomics[w + 5u], asuint(c.x));
    ASTORE(gScratchAtomics[w + 6u], asuint(c.y));
    ASTORE(gScratchAtomics[w + 7u], asuint(c.z));
}

// After mpm_couple_emit: the step's post-projectile pass applies only records past
// these (IMPULSE_FIRST), so the reactions are applied once, at the step's start.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_couple_finish(uint3 tid : SV_DispatchThreadID)
{
    ASTORE(gScratchAtomics[IMPULSE_FIRST], min(ALOAD(gScratchAtomics[IMPULSE_COUNT]), IMPULSE_CAP));
}

