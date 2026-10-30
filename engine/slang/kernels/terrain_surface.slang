// Render-only: the live ground surface over every open terrain area
// (engine/docs/deformable-terrain.md, "Rendering and showcase").
//
// While an area is open its heightfield samples are lowered to the bed and
// the material is particles. A renderer that draws those particles shows a
// cloud; a game shows ground that deforms, with only the material that left it
// flying as debris. These kernels rebuild, every frame, one height per window
// sample from the particles, with the bake's own splat, so the surface a host
// draws over each window is the heightfield that area would bake to now.
//
// Each open area is a view (its slot's record in the view table below): its
// window's heights, the walls' give and its occupancy sit in sections of this
// buffer, and every kernel covers every view in one dispatch. Per-sample
// kernels find their view from the thread's range in the table (the plan lays
// the views back to back); per-particle kernels from the area the particle is
// in (mpmAreaOf), so each particle splats into its own area's view only.
//
//   terrain_surface_clear    empty the occupancy (window samples × layers)
//   terrain_surface_shift    on the absorbing ground: the walls' give, as
//   terrain_surface_shift_mean  mpm_bake_shift measures it, into this buffer
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
//   terrain_surface_publish  (with patches) each open area's surface into its
//                            slot's static heightfield patch over its window,
//                            so wheel rays and rigid contacts meet the live
//                            sand (below); a slot not open sinks its patch
//
// In flight: faster than the header's ejecta speed (word 5, m/s; 0: never).
// The plan sets it to what a fall of one layer gives, √(2 g dy): sand moving
// faster has been thrown, whether or not a gap has opened under it yet (the
// excavation flow and the curtain at the crater's scale), and is debris, not
// ground. A settled area has nothing that fast, so the bake still sees the
// same quanta the surface holds.
//
// Material above the first gap of two layers (the ejecta curtain, a lifted
// flap, sand in flight inside the box) is not part of the surface; a renderer
// draws the debris list, and the parcels, as debris. Once nothing is in flight and an area bakes, the
// surface and the baked heightfield are the same splat of the same quanta, so
// the switch at the bake does not show (to the fixed point's rounding, and the
// material the bake pushes off a buried body, which this pass does not move).
//
// Nothing here is read by the simulation, and nothing the simulation reads is
// written but the patches: binding 53 is the only other output.
// engine_terrain::surface is the plan.
#include "terrain_mpm.slang"

// u32 words (engine_terrain::surface). Host-written by the plan: [0] views to
// scan (one past the highest open slot), [4] the debris list's first word,
// [5] the ejecta speed (f32 bits, m/s). GPU-written: [14] frames resolved,
// [15] valid (1 while an area is open; the host clears it when the last one
// bakes), [16..20) the debris draw's indirect arguments: the vertex count (a
// renderer writes it), the instance count (the debris listed this frame), 0,
// 0. From word 24 the view table, one host-written record of 16 words per
// area slot: [0] flags (TS_OPEN, TS_ABSORBING), [1] layers, [2] its heights'
// first word (f32 bits, the heightfield's local y, one per window sample),
// [3] its shift section's, [4] its occupancy's, [5] its patch collider
// (TS_NONE: none), [6..10) the window i0, j0, ni, nj, [10] the bed's local y
// (f32), [11] the layer height (f32), [12..16) its first thread in the
// clear, shift-mean, resolve and publish dispatches.
BIND(53) RWStructuredBuffer<AUint> gTerrainSurface;

static const uint TS_VIEWS = 0u;
static const uint TS_DEBRIS_BASE = 4u;
static const uint TS_EJECTA_SPEED = 5u;
static const uint TS_NONE = 0xFFFFFFFFu;
static const uint TS_FRAMES = 14u;
static const uint TS_VALID = 15u;
static const uint TS_DEBRIS_COUNT = 17u;
static const uint TS_VIEW = 24u;
static const uint TS_VIEW_WORDS = 16u;
// View record words.
static const uint TS_V_FLAGS = 0u;
static const uint TS_V_LAYERS = 1u;
static const uint TS_V_HEIGHTS = 2u;
static const uint TS_V_SHIFT = 3u;
static const uint TS_V_OCC = 4u;
static const uint TS_V_PATCH = 5u;
static const uint TS_V_WINDOW = 6u;
static const uint TS_V_CLEAR = 12u;
static const uint TS_V_MEAN = 13u;
static const uint TS_V_RESOLVE = 14u;
static const uint TS_V_PUBLISH = 15u;
static const uint TS_OPEN = 1u;
static const uint TS_ABSORBING = 2u;
// Occupancy units per volume quantum (atomic adds are integers).
static const float TS_FIXED = 16.0;
// Two layers in a row holding less than this share of a full cell end the
// attached column.
static const float TS_FILLED = 0.3;

uint tsWord(uint k) { return ALOAD(gTerrainSurface[k]); }

// Word f of slot s's view record.
uint tsView(uint s, uint f) { return tsWord(TS_VIEW + s * TS_VIEW_WORDS + f); }

bool tsOpen(uint s) { return (tsView(s, TS_V_FLAGS) & TS_OPEN) != 0u; }

uint4 tsWindow(uint s)
{
    return uint4(tsView(s, TS_V_WINDOW), tsView(s, TS_V_WINDOW + 1u), tsView(s, TS_V_WINDOW + 2u), tsView(s, TS_V_WINDOW + 3u));
}

// The view a particle splats into: the open area it is in, or TS_NONE.
uint tsViewOf(MpmParticle q)
{
    uint s = mpmAreaOf(q.xm.xyz, gMpmParams[0]);
    return s != MPM_NO_AREA && s < tsWord(TS_VIEWS) && tsOpen(s) ? s : TS_NONE;
}

bool tsInFlight(MpmParticle q)
{
    float v = asfloat(tsWord(TS_EJECTA_SPEED));
    return v > 0.0 && dot(q.vj.xyz, q.vj.xyz) > v * v;
}

// Layers are the area's cells in y from the bed up.
float tsLayerHeight(MpmParams p) { return p.origin.w; }

// Threads view s takes in the dispatch whose first threads are word `range`.
uint tsThreads(uint s, uint range)
{
    uint4 w = tsWindow(s);
    uint samples = w.z * w.w, edge = 2u * (w.z + w.w);
    if (range == TS_V_CLEAR) return max(samples * tsView(s, TS_V_LAYERS), 2u * edge);
    if (range == TS_V_MEAN) return edge;
    return samples;
}

// The open view whose threads in the dispatch of `range` hold thread k (k
// becomes its index in that view), or TS_NONE past the last.
uint tsViewAt(uint range, inout uint k)
{
    uint n = tsWord(TS_VIEWS);
    for (uint s = 0u; s < n; s++)
    {
        if (!tsOpen(s)) continue;
        uint first = tsView(s, range);
        if (k >= first && k - first < tsThreads(s, range))
        {
            k -= first;
            return s;
        }
    }
    return TS_NONE;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_clear(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    if (k == 0u) ASTORE(gTerrainSurface[TS_DEBRIS_COUNT], 0u);
    uint s = tsViewAt(TS_V_CLEAR, k);
    if (s == TS_NONE) return;
    uint4 w = tsWindow(s);
    if (k < w.z * w.w * tsView(s, TS_V_LAYERS)) ASTORE(gTerrainSurface[tsView(s, TS_V_OCC) + k], 0u);
    // The walls' give accumulators (terrain_surface_shift).
    if (k < 4u * (w.z + w.w)) ASTORE(gTerrainSurface[tsShiftAcc(0u, s) + k], 0u);
}

// The walls' give, as mpm_bake_shift measures it (per window edge sample:
// its column's mean outward displacement across that side), into this
// buffer's shift section; the face records it would overwrite are in use
// while the area is open. Two kernels, from the particles themselves (an
// area that grew has no activation range for its older samples, and the
// bake's edge lists are built at the bake): terrain_surface_shift adds each
// particle made at an edge column (by its home) into that edge's
// accumulators, in fixed point (mm × quanta, signed, two's complement adds);
// terrain_surface_shift_mean divides. Edge k: the low-x side's rows, the
// high-x side's, then the low-z side's columns and the high-z side's.
// Accumulators: two words per edge after the view's 2(ni + nj) shift words.
static const float TS_SHIFT_FIXED = 1000.0;
uint tsShiftAcc(uint k, uint s)
{
    uint4 w = tsWindow(s);
    return tsView(s, TS_V_SHIFT) + 2u * (w.z + w.w) + 2u * k;
}

void tsShiftAdd(uint k, float disp, uint w, uint s)
{
    uint at = tsShiftAcc(k, s);
    AADD2(gTerrainSurface[at], uint(int(round(disp * TS_SHIFT_FIXED)) * int(w)));
    AADD2(gTerrainSurface[at + 1u], w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_shift(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || (asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return;
    uint s = tsViewOf(q);
    if (s == TS_NONE || (tsView(s, TS_V_FLAGS) & TS_ABSORBING) == 0u) return;
    MpmParams p = mpmArea(s);
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    int ni = int(p.window.z), nj = int(p.window.w);
    int hi = int(floor((q.c1.w - hf.x0) / hf.dx + 0.5)) - int(p.window.x);
    int hj = int(floor((q.c2.w - hf.z0) / hf.dz + 0.5)) - int(p.window.y);
    if (hi < 0 || hj < 0 || hi >= ni || hj >= nj) return;
    bool ex = hi == 0 || hi == ni - 1, ez = hj == 0 || hj == nj - 1;
    if (!ex && !ez) return;
    float3 xl = frameToLocal(f, q.xm.xyz);
    uint w = asuint(q.f1.w);
    if (hi == 0) tsShiftAdd(uint(hj), xl.x - q.c1.w, w, s);
    if (hi == ni - 1) tsShiftAdd(uint(nj + hj), xl.x - q.c1.w, w, s);
    if (hj == 0) tsShiftAdd(uint(2 * nj + hi), xl.z - q.c2.w, w, s);
    if (hj == nj - 1) tsShiftAdd(uint(2 * nj + ni + hi), xl.z - q.c2.w, w, s);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_shift_mean(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    uint s = tsViewAt(TS_V_MEAN, k);
    if (s == TS_NONE) return;
    uint4 win = tsWindow(s);
    uint ni = win.z, nj = win.w;
    uint at = tsShiftAcc(k, s);
    float quanta = float(tsWord(at + 1u));
    float mean = quanta > 0.0 ? float(int(tsWord(at))) / TS_SHIFT_FIXED / quanta : 0.0;
    // Only outward: material moving out past the side's plane has moved the
    // ground with it (it cannot enter the ground); moving in, it left the
    // ground where it was.
    bool low = k < 2u * nj ? k < nj : (k - 2u * nj) < ni;
    ASTORE(gTerrainSurface[tsView(s, TS_V_SHIFT) + k], asuint(low ? min(mean, 0.0) : max(mean, 0.0)));
}

// mpmGroundShift over view s's shift section.
float2 tsGroundShift(float3 xl, HeightFieldRef hf, MpmParams p, uint s)
{
    int i0 = int(p.window.x), j0 = int(p.window.y);
    int ni = int(p.window.z), nj = int(p.window.w);
    float2 u = (xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
    int ti = clamp(int(round(u.x)) - i0, 0, ni - 1), tj = clamp(int(round(u.y)) - j0, 0, nj - 1);
    uint base = tsView(s, TS_V_SHIFT);
    float sx = ni > 1 ? clamp((u.x - float(i0)) / float(ni - 1), 0.0, 1.0) : 0.5;
    float sz = nj > 1 ? clamp((u.y - float(j0)) / float(nj - 1), 0.0, 1.0) : 0.5;
    float lx = asfloat(tsWord(base + uint(tj))), hx = asfloat(tsWord(base + uint(nj + tj)));
    float lz = asfloat(tsWord(base + uint(2 * nj + ti))), hz = asfloat(tsWord(base + uint(2 * nj + ni + ti)));
    return float2(lerp(lx, hx, sx), lerp(lz, hz, sz));
}

// The particle's local position in the unactivated ground's frame (the
// walls' give taken out, on the absorbing ground).
float3 tsGroundLocal(MpmParticle q, Frame f, HeightFieldRef hf, MpmParams p, uint s)
{
    float3 xl = frameToLocal(f, q.xm.xyz);
    if ((tsView(s, TS_V_FLAGS) & TS_ABSORBING) != 0u) xl.xz -= tsGroundShift(xl, hf, p, s);
    return xl;
}

// `units` of a particle onto sample s of view v at local height yl: a window
// sample's two layers around yl. Outside the window, material below the
// unactivated ground there folds into the nearest window sample (as
// mpmBakeOnto bakes it); material above it would bake onto that ground as a
// deposit and is not part of the window's surface.
void tsAdd(int2 s, float units, float yl, MpmParams p, HeightFieldRef hf, uint v)
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
    uint layers = tsView(v, TS_V_LAYERS);
    float fy = (yl - p.column.x) / tsLayerHeight(p) - 0.5;
    float b = floor(fy);
    float f = fy - b;
    int k0 = clamp(int(b), 0, int(layers) - 1), k1 = clamp(int(b) + 1, 0, int(layers) - 1);
    uint at = tsView(v, TS_V_OCC) + t * layers;
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
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || tsInFlight(q)) return;
    uint v = tsViewOf(q);
    if (v == TS_NONE) return;
    MpmParams p = mpmArea(v);
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    float3 xl = tsGroundLocal(q, f, hf, p, v);
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
        tsAdd(s0, h * (1.0 - t.x) * (1.0 - t.y), xl.y, p, hf, v);
        tsAdd(s0 + int2(1, 0), h * t.x * (1.0 - t.y), xl.y, p, hf, v);
        tsAdd(s0 + int2(0, 1), h * (1.0 - t.x) * t.y, xl.y, p, hf, v);
        tsAdd(s0 + int2(1, 1), h * t.x * t.y, xl.y, p, hf, v);
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
            tsAdd(int2(bx + int(a), bz + int(c)), m * mpmWeightAt(wx, a) * mpmWeightAt(wz, c), xl.y, p, hf, v);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_resolve(uint3 tid : SV_DispatchThreadID)
{
    uint t = mpmIndex(tid);
    if (t == 0u)
    {
        AADD2(gTerrainSurface[TS_FRAMES], 1u);
        ASTORE(gTerrainSurface[TS_VALID], 1u);
    }
    uint v = tsViewAt(TS_V_RESOLVE, t);
    if (v == TS_NONE) return;
    MpmParams p = mpmArea(v);
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    float area = mpmSampleArea(hf);
    float full = area * tsLayerHeight(p) / p.column.y * TS_FIXED;
    uint layers = tsView(v, TS_V_LAYERS);
    uint at = tsView(v, TS_V_OCC) + t * layers;
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
    ASTORE(gTerrainSurface[tsView(v, TS_V_HEIGHTS) + t], asuint(base + units / TS_FIXED * p.column.y / area));
}

// The live surface at sample (i, j): an open window's rebuilt height inside
// it, the heightfield outside every one.
float tsSurfaceSample(int i, int j, HeightFieldRef hf)
{
    i = clamp(i, 0, int(hf.nx) - 1);
    j = clamp(j, 0, int(hf.nz) - 1);
    uint n = tsWord(TS_VIEWS);
    for (uint s = 0u; s < n; s++)
    {
        if (!tsOpen(s)) continue;
        uint4 w = tsWindow(s);
        int ti = i - int(w.x), tj = j - int(w.y);
        if (ti >= 0 && tj >= 0 && ti < int(w.z) && tj < int(w.w))
            return asfloat(tsWord(tsView(s, TS_V_HEIGHTS) + uint(tj) * w.z + uint(ti)));
    }
    return hfHeight(hf, uint(i), uint(j));
}

float tsSurfaceAt(float2 xz, HeightFieldRef hf)
{
    float2 u = (xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
    float2 b = floor(u);
    float2 f = u - b;
    int i = int(b.x), j = int(b.y);
    return lerp(lerp(tsSurfaceSample(i, j, hf), tsSurfaceSample(i + 1, j, hf), f.x),
        lerp(tsSurfaceSample(i, j + 1, hf), tsSurfaceSample(i + 1, j + 1, hf), f.x), f.y);
}

// After terrain_surface_resolve: every particle in flight or more than a layer
// above the surface (in the ground's frame its view was built in), listed
// for the debris draw.
[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_debris(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    uint v = tsViewOf(q);
    if (v == TS_NONE) return;
    MpmParams p = mpmArea(v);
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    float3 xl = tsGroundLocal(q, f, hf, p, v);
    if (!tsInFlight(q) && xl.y <= tsSurfaceAt(xl.xz, hf) + tsLayerHeight(p)) return;
    uint k;
    AADD3(gTerrainSurface[TS_DEBRIS_COUNT], 1u, k);
    ASTORE(gTerrainSurface[tsWord(TS_DEBRIS_BASE) + k], i);
}

// Records (four samples each) of patch collider pc.
uint tsPatchRecords(uint pc)
{
    HeightFieldRef lh = heightFieldOf(gColliders[pc]);
    return (lh.nx * lh.nz + 3u) / 4u;
}

// The surface patches: one static heightfield collider of its own per area
// slot (a body at rest, no rotation) that the host made at least two samples
// larger than the window each way. While an area is open its window's
// heightfield samples sit at the bed, under the sand, and a wheel's ray would
// find that bed; its slot's patch is put over the window with a ring of the
// field's own samples (and the live heights of any other open window it
// reaches), so rays and contacts meet the live sand where it is. A slot not
// open sinks its patch out of reach. One thread per patch record, the patches
// back to back in slot order; a patch's thread 0 also places it and its
// height range. Physics reads the patches: they are the only output here
// that it does.
[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_surface_publish(uint3 tid : SV_DispatchThreadID)
{
    uint r = mpmIndex(tid);
    uint s = TS_NONE, pc = TS_NONE;
    for (uint k = 0u; k < MPM_MAX_AREAS; k++)
    {
        uint c = tsView(k, TS_V_PATCH);
        if (c == TS_NONE) continue;
        uint first = tsView(k, TS_V_PUBLISH);
        if (r >= first && r - first < tsPatchRecords(c))
        {
            s = k;
            pc = c;
            r -= first;
            break;
        }
    }
    if (s == TS_NONE) return;
    Collider lc = gColliders[pc];
    HeightFieldRef lh = heightFieldOf(lc);
    MpmParams p = mpmArea(s);
    bool on = tsWord(TS_VALID) == 1u && tsOpen(s) && p.dims.w != MPM_NO_BLOCK;
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
        float h = b < lh.nz ? tsSurfaceSample(i0 + int(a), j0 + int(b), bh) : -1.0e4;
        rec[c] = asuint(h);
    }
    gHulls[lh.base + 2u + r] = rec;
}
