// Render-only: the live ground surface over an open terrain region
// (engine/docs/deformable-terrain.md, "Rendering and showcase").
//
// While a region is open its heightfield samples are lowered to the bed and
// the material is particles. A renderer that draws those particles shows a
// cloud; a game shows ground that deforms, with only the material that left it
// flying as debris. These kernels rebuild, every frame, one height per window
// sample from the particles, with the bake's own splat, so the surface a host
// draws over the window is the heightfield the region would bake to now:
//
//   terrain_surface_clear    empty the occupancy (window samples × layers)
//   terrain_surface_shift    on the absorbing ground: the walls' give (a copy of
//                            mpm_bake_shift into this buffer)
//   terrain_surface_bin      per particle not in flight (below): its quanta split between samples as
//                            mpm_bake_bin splits them (the tent at home for the
//                            share at rest, P2G's B-spline for the share the MPM
//                            moved, in the unactivated ground's frame), and
//                            between the two layers around its height (a tent
//                            in y), in fixed point
//   terrain_surface_resolve  per sample: the material attached to the bed (the
//                            layers from the bed up to the first two in a row
//                            holding less than TS_FILLED of a full cell, those
//                            included), as a height above the sample's own
//                            (the lowered bed plus any parcels deposited on it)
//   terrain_surface_debris   per particle: listed when in flight or when it
//                            stands more than a layer above that surface, and
//                            counted into the debris draw's indirect arguments
//   terrain_surface_publish  (with a patch) the surface into a static
//                            heightfield patch over the window, so wheel rays
//                            and rigid contacts meet the live sand (below)
//
// In flight: faster than the header's ejecta speed (word 5, m/s; 0: never).
// The plan sets it to what a fall of one layer gives, √(2 g dy): sand moving
// faster has been thrown, whether or not a gap has opened under it yet (the
// excavation flow and the curtain at the crater's scale), and is debris, not
// ground. A settled region has nothing that fast, so the bake still sees the
// same quanta the surface holds.
//
// Material above the first gap of two layers (the ejecta curtain, a lifted
// flap, sand in flight inside the box) is not part of the surface; a renderer
// draws the debris list, and the parcels, as debris. Once nothing is in flight and the region bakes, the
// surface and the baked heightfield are the same splat of the same quanta, so
// the switch at the bake does not show (to the fixed point's rounding, and the
// material the bake pushes off a buried body, which this pass does not move).
//
// Nothing here is read by the simulation, and nothing the simulation reads is
// written: binding 53 is the only output. engine_terrain::surface is the plan.
#include "terrain_mpm.slang"

// u32 words (engine_terrain::surface). Host-written by the plan: [0] the shift
// section's first word, [1] the occupancy section's, [2] layers per column,
// [3] flags (bit 0: the absorbing ground's edge shift is used), [4] the debris
// list's first word, [5] the ejecta speed (f32 bits, m/s), [6] the surface
// patch's collider (0xFFFFFFFF: none), [7] the area slot it shows (several
// areas may be open; the surface is one window's). GPU-written by terrain_surface_resolve: [8..12) window
// i0, j0, ni, nj, [12] the bed's local y (f32 bits), [13] the layer height
// (f32), [14] frames resolved, [15] valid (1 while a region is open; the host
// clears it at the bake). [16..20) the debris draw's indirect arguments: the
// vertex count (a renderer writes it), the instance count (the debris listed
// this frame), 0, 0. Heights (f32 bits, the heightfield's local y) from word
// 24, one per window sample.
BIND(53) RWStructuredBuffer<AUint> gTerrainSurface;

static const uint TS_SHIFT_BASE = 0u;
static const uint TS_OCC_BASE = 1u;
static const uint TS_LAYERS = 2u;
static const uint TS_FLAGS = 3u;
static const uint TS_DEBRIS_BASE = 4u;
static const uint TS_EJECTA_SPEED = 5u;
static const uint TS_PATCH = 6u;
static const uint TS_AREA = 7u;
static const uint TS_NONE = 0xFFFFFFFFu;
static const uint TS_WINDOW = 8u;
static const uint TS_BED = 12u;
static const uint TS_DY = 13u;
static const uint TS_FRAMES = 14u;
static const uint TS_VALID = 15u;
static const uint TS_DEBRIS_COUNT = 17u;
static const uint TS_HEIGHTS = 24u;
static const uint TS_FLAG_SHIFT = 1u;
// Occupancy units per volume quantum (atomic adds are integers).
static const float TS_FIXED = 16.0;
// Two layers in a row holding less than this share of a full cell end the
// attached column.
static const float TS_FILLED = 0.3;

uint tsWord(uint k) { return ALOAD(gTerrainSurface[k]); }

// The area the surface shows (its record composed onto the domain's).
MpmParams tsArea() { return mpmArea(tsWord(TS_AREA)); }

// A particle of the area the surface shows.
bool tsMine(MpmParticle q) { return mpmAreaOf(q.xm.xyz, gMpmParams[0]) == tsWord(TS_AREA); }

bool tsInFlight(MpmParticle q)
{
    float v = asfloat(tsWord(TS_EJECTA_SPEED));
    return v > 0.0 && dot(q.vj.xyz, q.vj.xyz) > v * v;
}

// Layers are the region's cells in y from the bed up.
float tsLayerHeight(MpmParams p) { return p.origin.w; }

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_clear(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = tsArea();
    if (k < p.window.z * p.window.w * tsWord(TS_LAYERS)) ASTORE(gTerrainSurface[tsWord(TS_OCC_BASE) + k], 0u);
    if (k == 0u) ASTORE(gTerrainSurface[TS_DEBRIS_COUNT], 0u);
}

// mpm_bake_shift's measurement (per window edge sample: its column's mean
// outward displacement across that side), into this buffer's shift section:
// the face records it would overwrite are in use while the region is open.
[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_shift(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = tsArea();
    uint ni = p.window.z, nj = p.window.w;
    if (k >= 2u * (ni + nj)) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    uint2 ij;
    uint axis;
    if (k < 2u * nj)
    {
        ij = uint2(k < nj ? 0u : ni - 1u, k % nj);
        axis = 0u;
    }
    else
    {
        uint m = k - 2u * nj;
        ij = uint2(m % ni, m < ni ? 0u : nj - 1u);
        axis = 2u;
    }
    uint t = ij.y * ni + ij.x;
    // The column's last activation range (a grown area's older samples gave
    // none: their give is taken as zero here, render-only).
    uint first = ALOAD(gMpmColumns[mpmCol(p, t) + 2u]), n = ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
    float sum = 0.0, quanta = 0.0;
    for (uint s = first; s < first + n; s++)
    {
        MpmParticle q = gMpmParticles[s];
        if (mpmParticleDead(q) || (asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) continue;
        float3 xl = frameToLocal(f, q.xm.xyz);
        float w = float(asuint(q.f1.w));
        sum += w * (axis == 0u ? xl.x - q.c1.w : xl.z - q.c2.w);
        quanta += w;
    }
    float mean = quanta > 0.0 ? sum / quanta : 0.0;
    bool low = k < 2u * nj ? k < nj : (k - 2u * nj) < ni;
    ASTORE(gTerrainSurface[tsWord(TS_SHIFT_BASE) + k], asuint(low ? min(mean, 0.0) : max(mean, 0.0)));
}

// mpmGroundShift over this buffer's shift section.
float2 tsGroundShift(float3 xl, HeightFieldRef hf, MpmParams p)
{
    int i0 = int(p.window.x), j0 = int(p.window.y);
    int ni = int(p.window.z), nj = int(p.window.w);
    float2 u = (xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
    int ti = clamp(int(round(u.x)) - i0, 0, ni - 1), tj = clamp(int(round(u.y)) - j0, 0, nj - 1);
    uint base = tsWord(TS_SHIFT_BASE);
    float sx = ni > 1 ? clamp((u.x - float(i0)) / float(ni - 1), 0.0, 1.0) : 0.5;
    float sz = nj > 1 ? clamp((u.y - float(j0)) / float(nj - 1), 0.0, 1.0) : 0.5;
    float lx = asfloat(tsWord(base + uint(tj))), hx = asfloat(tsWord(base + uint(nj + tj)));
    float lz = asfloat(tsWord(base + uint(2 * nj + ti))), hz = asfloat(tsWord(base + uint(2 * nj + ni + ti)));
    return float2(lerp(lx, hx, sx), lerp(lz, hz, sz));
}

// `units` of a particle onto sample s at local height yl: a window sample's
// two layers around yl. Outside the window, material below the unactivated
// ground there folds into the nearest window sample (as mpmBakeOnto bakes it);
// material above it would bake onto that ground as a deposit and is not part
// of the window's surface.
void tsAdd(int2 s, float units, float yl, MpmParams p, HeightFieldRef hf)
{
    if (units <= 0.0) return;
    if (s.x < 0 || s.y < 0 || s.x >= int(hf.nx) || s.y >= int(hf.nz)) return;
    int2 wlo = int2(p.window.xy), whi = int2(p.window.xy + p.window.zw) - 1;
    if (any(s < wlo) || any(s > whi))
    {
        if (yl >= hfHeight(hf, uint(s.x), uint(s.y))) return;
        s = clamp(s, wlo, whi);
    }
    uint t = uint(s.y - wlo.y) * p.window.z + uint(s.x - wlo.x);
    uint layers = tsWord(TS_LAYERS);
    float fy = (yl - p.column.x) / tsLayerHeight(p) - 0.5;
    float b = floor(fy);
    float f = fy - b;
    int k0 = clamp(int(b), 0, int(layers) - 1), k1 = clamp(int(b) + 1, 0, int(layers) - 1);
    uint at = tsWord(TS_OCC_BASE) + t * layers;
    uint lo = uint(units * (1.0 - f) + 0.5), hi = uint(units * f + 0.5);
    if (lo > 0u) AADD2(gTerrainSurface[at + uint(k0)], lo);
    if (hi > 0u) AADD2(gTerrainSurface[at + uint(k1)], hi);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_bin(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = tsArea();
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || tsInFlight(q) || !tsMine(q)) return;
    float3 xl = frameToLocal(f, q.xm.xyz);
    if ((tsWord(TS_FLAGS) & TS_FLAG_SHIFT) != 0u) xl.xz -= tsGroundShift(xl, hf, p);
    uint quanta = asuint(q.f1.w);
    // A host-loaded particle carries no quanta: its rest volume in quanta.
    float units = (quanta > 0u ? float(quanta) : q.f0.w / p.column.y) * TS_FIXED;
    float2 uHome;
    float w = mpmHomeShare(q, xl, hf, p, uHome);
    if (w > 0.0)
    {
        // The share at rest: the tent at its home sample plus its displacement.
        float2 b = floor(uHome);
        float2 t = uHome - b;
        int2 s0 = int2(b);
        float h = units * w;
        tsAdd(s0, h * (1.0 - t.x) * (1.0 - t.y), xl.y, p, hf);
        tsAdd(s0 + int2(1, 0), h * t.x * (1.0 - t.y), xl.y, p, hf);
        tsAdd(s0 + int2(0, 1), h * (1.0 - t.x) * t.y, xl.y, p, hf);
        tsAdd(s0 + int2(1, 1), h * t.x * t.y, xl.y, p, hf);
    }
    if (w < 1.0)
    {
        // The share the MPM moved: P2G's B-spline at its own position.
        float2 u = (xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
        int bx, bz;
        float3 wx = mpmBsplineWeights(u.x, bx), wz = mpmBsplineWeights(u.y, bz);
        float m = units * (1.0 - w);
        for (uint k = 0u; k < 9u; k++)
        {
            uint a = k % 3u, c = k / 3u;
            tsAdd(int2(bx + int(a), bz + int(c)), m * mpmWeightAt(wx, a) * mpmWeightAt(wz, c), xl.y, p, hf);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_resolve(uint3 tid : SV_DispatchThreadID)
{
    uint t = mpmIndex(tid);
    MpmParams p = tsArea();
    uint n = p.window.z * p.window.w;
    if (t == 0u)
    {
        for (uint k = 0u; k < 4u; k++) ASTORE(gTerrainSurface[TS_WINDOW + k], p.window[k]);
        ASTORE(gTerrainSurface[TS_BED], asuint(p.column.x));
        ASTORE(gTerrainSurface[TS_DY], asuint(tsLayerHeight(p)));
        AADD2(gTerrainSurface[TS_FRAMES], 1u);
        ASTORE(gTerrainSurface[TS_VALID], 1u);
    }
    if (t >= n) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    float area = mpmSampleArea(hf);
    float full = area * tsLayerHeight(p) / p.column.y * TS_FIXED;
    uint layers = tsWord(TS_LAYERS);
    uint at = tsWord(TS_OCC_BASE) + t * layers;
    // Up to two sparse layers in a row (both counted): a gap of at least two
    // layers separates detached material. One sparse layer alone is the
    // loose top of the column (a resting top row split between two layers).
    float units = 0.0;
    uint sparse = 0u;
    for (uint k = 0u; k < layers && sparse < 2u; k++)
    {
        float o = float(tsWord(at + k));
        units += o;
        sparse = o < TS_FILLED * full ? sparse + 1u : 0u;
    }
    // On the sample's own height, as the bake raises it: the bed (to within
    // the activation's rounding) plus what parcels deposited there meanwhile.
    uint2 ij = mpmWindowSample(t, p);
    float base = hfHeight(hf, ij.x, ij.y);
    ASTORE(gTerrainSurface[TS_HEIGHTS + t], asuint(base + units / TS_FIXED * p.column.y / area));
}

// The live surface at local (x, z): bilinear over the samples, the window's
// rebuilt heights inside it and the heightfield outside.
float tsSurfaceSample(int i, int j, HeightFieldRef hf, MpmParams p)
{
    i = clamp(i, 0, int(hf.nx) - 1);
    j = clamp(j, 0, int(hf.nz) - 1);
    int ti = i - int(p.window.x), tj = j - int(p.window.y);
    if (ti >= 0 && tj >= 0 && ti < int(p.window.z) && tj < int(p.window.w))
        return asfloat(tsWord(TS_HEIGHTS + uint(tj) * p.window.z + uint(ti)));
    return hfHeight(hf, uint(i), uint(j));
}

float tsSurfaceAt(float2 xz, HeightFieldRef hf, MpmParams p)
{
    float2 u = (xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
    float2 b = floor(u);
    float2 f = u - b;
    int i = int(b.x), j = int(b.y);
    return lerp(lerp(tsSurfaceSample(i, j, hf, p), tsSurfaceSample(i + 1, j, hf, p), f.x),
        lerp(tsSurfaceSample(i, j + 1, hf, p), tsSurfaceSample(i + 1, j + 1, hf, p), f.x), f.y);
}

// After terrain_surface_resolve: every particle in flight or more than a layer
// above the surface (in the ground's frame the surface was built in), listed
// for the debris draw.
[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_debris(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = tsArea();
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || !tsMine(q)) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    float3 xl = frameToLocal(f, q.xm.xyz);
    if ((tsWord(TS_FLAGS) & TS_FLAG_SHIFT) != 0u) xl.xz -= tsGroundShift(xl, hf, p);
    if (!tsInFlight(q) && xl.y <= tsSurfaceAt(xl.xz, hf, p) + tsLayerHeight(p)) return;
    uint k;
    AADD3(gTerrainSurface[TS_DEBRIS_COUNT], 1u, k);
    ASTORE(gTerrainSurface[tsWord(TS_DEBRIS_BASE) + k], i);
}

// The surface patch: a static heightfield collider of its own (a body at rest,
// no rotation) that the host made at least two samples larger than the
// window. While a region is open the window's heightfield samples sit at the
// bed, under the sand, and a wheel's ray would find that bed; the patch is
// put over the window with a ring of the field's own samples, and given the
// surface's heights, so rays and contacts meet the live sand where it is.
// Without a region (valid word 0) it is sunk out of reach. One thread per
// patch record (four samples); thread 0 also places the patch and its height
// range. Physics reads this patch: it is the only output here that does.
[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_publish(uint3 tid : SV_DispatchThreadID)
{
    uint r = mpmIndex(tid);
    uint pc = tsWord(TS_PATCH);
    if (pc == TS_NONE) return;
    Collider lc = gColliders[pc];
    HeightFieldRef lh = heightFieldOf(lc);
    uint records = (lh.nx * lh.nz + 3u) / 4u;
    if (r >= records) return;
    MpmParams p = tsArea();
    bool on = tsWord(TS_VALID) == 1u && p.dims.w != MPM_NO_BLOCK;
    if (!on)
    {
        float far = -1.0e4;
        gHulls[lh.base + 2u + r] = uint4(asuint(far), asuint(far), asuint(far), asuint(far));
        if (r == 0u)
        {
            uint4 dims = gHulls[lh.base + 1u];
            dims.z = asuint(far);
            dims.w = asuint(far);
            gHulls[lh.base + 1u] = dims;
        }
        return;
    }
    Collider bc = gColliders[p.dims.w];
    HeightFieldRef bh = heightFieldOf(bc);
    Frame bf = colliderFrame(bc, gBodies[bc.body]);
    // Patch sample (a, b) is the field's sample (i0 - 1 + a, j0 - 1 + b).
    int i0 = int(p.window.x) - 1, j0 = int(p.window.y) - 1;
    if (r == 0u)
    {
        float3 at = bf.p + float3(bh.x0 + float(i0) * bh.dx - lh.x0, 0.0, bh.z0 + float(j0) * bh.dz - lh.z0);
        gBodies[lc.body].position = at - lc.localPos;
        uint4 dims = gHulls[lh.base + 1u];
        dims.z = asuint(p.column.x - tsLayerHeight(p));
        dims.w = asuint(p.origin.y + float(p.dims.y) * p.origin.w - bf.p.y);
        gHulls[lh.base + 1u] = dims;
    }
    uint4 rec;
    for (uint c = 0u; c < 4u; c++)
    {
        uint k = 4u * r + c;
        uint a = k % lh.nx, b = k / lh.nx;
        float h = b < lh.nz ? tsSurfaceSample(i0 + int(a), j0 + int(b), bh, p) : -1.0e4;
        rec[c] = asuint(h);
    }
    gHulls[lh.base + 2u + r] = rec;
}
