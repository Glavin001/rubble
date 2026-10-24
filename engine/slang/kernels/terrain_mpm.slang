// Deformable terrain MPM (T2; engine/docs/deformable-terrain.md, "T2 kernel design").
//
// Explicit MLS-MPM on a sparse grid of 4³-cell blocks inside one active region.
// Per substep:
//   mpm_clear       empty the block hash and the counters
//   mpm_touch       insert every block a particle's 3³ stencil reaches
//   mpm_count       count particles per cell (the cell of their stencil's base node)
//   mpm_scan_cells  per block: cell starts and the block's total
//   mpm_scan_blocks block starts (one workgroup)
//   mpm_scatter     particle ids into their cell's range
//   mpm_cell_sort   each cell's range by particle id (the sums' fixed order)
//   mpm_payload     per particle: mass, momentum and the affine term, in sorted order
//   mpm_p2g         per grid node: gather from the particles whose stencil covers it
//   mpm_grid        velocity, gravity, the heightfield bed
//   mpm_g2p         per particle: velocity, C, position and F from the grid
// P2G is a gather (no float atomics, which WGSL lacks): every node sums its
// particles in a fixed order, so runs and backends agree bit for bit.
//
// Region-local node g sits at origin + g·dx. A particle's base node is
// floor((x - origin)/dx - 0.5); its stencil covers base .. base + 2. Particles
// must stay one cell inside the region's box (MPM_OUTSIDE otherwise: the host
// fails the step; ejection to parcels is T4).
#include "world_step.slang"

static const uint MPM_GROUP = 64u;
static const uint MPM_BLOCK = 4u;
static const uint MPM_BLOCK_NODES = 64u;
static const uint MPM_NO_BLOCK = 0xFFFFFFFFu;

// gMpmCounters words.
static const uint MPM_C_BLOCKS = 0u;      // blocks inserted this substep
static const uint MPM_C_FLAGS = 1u;       // MPM_OVERFLOW_* | MPM_OUTSIDE
static const uint MPM_C_MAX_SPEED = 2u;   // max |v| after G2P (f32 bits; monotone for v ≥ 0)
static const uint MPM_OVERFLOW_BLOCKS = 1u;
static const uint MPM_OVERFLOW_HASH = 2u;
static const uint MPM_OUTSIDE = 4u;

// Material kinds (params u0.w).
static const uint MPM_MATERIAL_NONE = 0u; // no stress: transfers only (tests)

struct MpmParams
{
    float4 origin;  // xyz region origin, w = dx
    float4 step;    // x = dt, y = gravity (along y), z = node mass threshold, w = bed friction μ
    uint4 counts;   // x = particles, y = hash slots (power of two), z = max blocks, w = material
    uint4 dims;     // xyz = region size in cells, w = bed heightfield collider (MPM_NO_BLOCK: none)
};

// Per particle (128 B).
struct MpmParticle
{
    float4 xm;      // position, mass
    float4 vj;      // velocity, log Jp (plastic volume)
    float4 c0;      // C rows (APIC affine velocity)
    float4 c1;
    float4 c2;
    float4 f0;      // F rows; f0.w = rest volume
    float4 f1;
    float4 f2;
};

// Per particle in sorted order, what P2G reads (80 B).
struct MpmPayload
{
    float4 xm;      // position, mass
    float4 mv;      // momentum m·v
    float4 a0;      // affine term rows: m·C − dt·V·(4/dx²)·τ
    float4 a1;
    float4 a2;
};

BIND(37) RWStructuredBuffer<MpmParams> gMpmParams;
BIND(38) RWStructuredBuffer<AUint> gMpmCounters;
BIND(39) RWStructuredBuffer<MpmParticle> gMpmParticles;
BIND(40) RWStructuredBuffer<MpmPayload> gMpmPayload;
// Open-addressing block hash: [2s] = key + 1 (0 empty), [2s + 1] = block index.
BIND(41) RWStructuredBuffer<AUint> gMpmHash;
// Per block: [4b] = key, [4b + 1] = particles, [4b + 2] = first sorted slot.
BIND(42) RWStructuredBuffer<AUint> gMpmBlocks;
// Per cell (block · 64 + local): [4c] = count, [4c + 1] = start in its block, [4c + 2] = cursor.
BIND(43) RWStructuredBuffer<AUint> gMpmCells;
BIND(44) RWStructuredBuffer<uint> gMpmSorted;
// Per node (block · 64 + local): after P2G (m·v, m), after mpm_grid (v, m).
BIND(45) RWStructuredBuffer<float4> gMpmGrid;

groupshared uint gsMpmScan[256];

uint mpmIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }

MpmParams mpmParams() { return gMpmParams[0]; }

int3 mpmBaseNode(float3 x, MpmParams p)
{
    return int3(floor((x - p.origin.xyz) / p.origin.w - 0.5));
}

uint mpmBlockKey(int3 block) { return uint(block.x) | (uint(block.y) << 10) | (uint(block.z) << 20); }

uint mpmHashSlot(uint key, uint slots) { return (key * 0x9E3779B1u) & (slots - 1u); }

// Inserts `key`, giving a new block the next index (overflow flags past either limit).
void mpmInsertBlock(uint key, MpmParams p)
{
    uint slots = p.counts.y;
    uint s = mpmHashSlot(key, slots);
    for (uint probe = 0u; probe < slots; probe++)
    {
        uint old;
        ACAS4(gMpmHash[2u * s], 0u, key + 1u, old);
        if (old == 0u)
        {
            uint b;
            AADD3(gMpmCounters[MPM_C_BLOCKS], 1u, b);
            if (b >= p.counts.z)
            {
                AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_BLOCKS);
                b = MPM_NO_BLOCK;
            }
            else ASTORE(gMpmBlocks[4u * b], key);
            ASTORE(gMpmHash[2u * s + 1u], b);
            return;
        }
        if (old == key + 1u) return;
        s = (s + 1u) & (slots - 1u);
    }
    AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_HASH);
}

// Block index of `key`, or MPM_NO_BLOCK (read in dispatches after the inserts).
uint mpmFindBlock(uint key, MpmParams p)
{
    uint slots = p.counts.y;
    uint s = mpmHashSlot(key, slots);
    for (uint probe = 0u; probe < slots; probe++)
    {
        uint k = ALOAD(gMpmHash[2u * s]);
        if (k == key + 1u) return ALOAD(gMpmHash[2u * s + 1u]);
        if (k == 0u) return MPM_NO_BLOCK;
        s = (s + 1u) & (slots - 1u);
    }
    return MPM_NO_BLOCK;
}

// Cell index (block · 64 + local) of node g, or MPM_NO_BLOCK.
uint mpmCellOf(int3 g, MpmParams p)
{
    if (any(g < int3(0, 0, 0))) return MPM_NO_BLOCK;
    uint b = mpmFindBlock(mpmBlockKey(g >> 2), p);
    if (b == MPM_NO_BLOCK) return MPM_NO_BLOCK;
    uint3 l = uint3(g & 3);
    return b * MPM_BLOCK_NODES + l.x + 4u * l.y + 16u * l.z;
}

// A particle one cell inside the region box (its stencil stays on the grid).
bool mpmInside(float3 x, MpmParams p)
{
    float3 r = (x - p.origin.xyz) / p.origin.w;
    return all(r >= float3(1, 1, 1)) && all(r < float3(p.dims.xyz) - 1.0);
}

// Quadratic B-spline weights of the three nodes base + 0..2 along one axis,
// fx = x/dx − base in [0.5, 1.5).
float3 mpmWeights(float fx)
{
    return float3(0.5 * (1.5 - fx) * (1.5 - fx), 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5));
}

float mpmWeightAt(float3 wx, uint o) { return o == 0u ? wx.x : o == 1u ? wx.y : wx.z; }

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i < 2u * p.counts.y) ASTORE(gMpmHash[i], 0u);
    if (i < 4u * p.counts.z * MPM_BLOCK_NODES) ASTORE(gMpmCells[i], 0u);
    if (i == 0u)
    {
        ASTORE(gMpmCounters[MPM_C_BLOCKS], 0u);
        ASTORE(gMpmCounters[MPM_C_MAX_SPEED], 0u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_touch(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= p.counts.x) return;
    float3 x = gMpmParticles[i].xm.xyz;
    if (!mpmInside(x, p)) { AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OUTSIDE); return; }
    int3 b = mpmBaseNode(x, p);
    int3 lo = b >> 2, hi = (b + 2) >> 2;
    for (int bz = lo.z; bz <= hi.z; bz++)
    for (int by = lo.y; by <= hi.y; by++)
    for (int bx = lo.x; bx <= hi.x; bx++)
        mpmInsertBlock(mpmBlockKey(int3(bx, by, bz)), p);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= p.counts.x) return;
    float3 x = gMpmParticles[i].xm.xyz;
    if (!mpmInside(x, p)) return;
    uint c = mpmCellOf(mpmBaseNode(x, p), p);
    if (c != MPM_NO_BLOCK) AADD2(gMpmCells[4u * c], 1u);
}

// One workgroup per block: exclusive scan of its 64 cell counts.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scan_cells(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint b = gid.x + gid.y * 32768u;
    uint l = lid.x;
    MpmParams p = mpmParams();
    bool live = b < min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    uint c = b * MPM_BLOCK_NODES + l;
    uint n = live ? ALOAD(gMpmCells[4u * c]) : 0u;
    gsMpmScan[l] = n;
    GROUP_SYNC();
    // Hillis–Steele inclusive scan over 64 lanes.
    for (uint d = 1u; d < MPM_BLOCK_NODES; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    if (!live) return;
    ASTORE(gMpmCells[4u * c + 1u], gsMpmScan[l] - n);
    if (l == MPM_BLOCK_NODES - 1u) ASTORE(gMpmBlocks[4u * b + 1u], gsMpmScan[l]);
}

// One workgroup of 256: exclusive scan of the block totals (each lane a chunk).
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_scan_blocks(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmParams();
    uint blocks = min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    uint chunk = (blocks + 255u) / 256u;
    uint first = l * chunk, last = min(first + chunk, blocks);
    uint sum = 0u;
    for (uint b = first; b < last; b++) sum += ALOAD(gMpmBlocks[4u * b + 1u]);
    gsMpmScan[l] = sum;
    GROUP_SYNC();
    for (uint d = 1u; d < 256u; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    uint start = gsMpmScan[l] - sum;
    for (uint b = first; b < last; b++)
    {
        ASTORE(gMpmBlocks[4u * b + 2u], start);
        start += ALOAD(gMpmBlocks[4u * b + 1u]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= p.counts.x) return;
    float3 x = gMpmParticles[i].xm.xyz;
    if (!mpmInside(x, p)) return;
    uint c = mpmCellOf(mpmBaseNode(x, p), p);
    if (c == MPM_NO_BLOCK) return;
    uint k;
    AADD3(gMpmCells[4u * c + 2u], 1u, k);
    uint b = c / MPM_BLOCK_NODES;
    gMpmSorted[ALOAD(gMpmBlocks[4u * b + 2u]) + ALOAD(gMpmCells[4u * c + 1u]) + k] = i;
}

// Range [first, first + n) of cell c in gMpmSorted.
void mpmCellRange(uint c, out uint first, out uint n)
{
    uint b = c / MPM_BLOCK_NODES;
    first = ALOAD(gMpmBlocks[4u * b + 2u]) + ALOAD(gMpmCells[4u * c + 1u]);
    n = ALOAD(gMpmCells[4u * c]);
}

// Insertion sort of each cell's (short) range by particle id.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_cell_sort(uint3 tid : SV_DispatchThreadID)
{
    uint c = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (c >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z) * MPM_BLOCK_NODES) return;
    uint first, n;
    mpmCellRange(c, first, n);
    for (uint a = 1u; a < n; a++)
    {
        uint v = gMpmSorted[first + a];
        uint k = a;
        while (k > 0u && gMpmSorted[first + k - 1u] > v)
        {
            gMpmSorted[first + k] = gMpmSorted[first + k - 1u];
            k--;
        }
        gMpmSorted[first + k] = v;
    }
}

// Kirchhoff stress τ of a particle (zero for MPM_MATERIAL_NONE).
float3x3 mpmStress(MpmParticle q, MpmParams p)
{
    return float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_payload(uint3 tid : SV_DispatchThreadID)
{
    uint s = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint blocks = min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    if (blocks == 0u) return;
    uint total = ALOAD(gMpmBlocks[4u * (blocks - 1u) + 2u]) + ALOAD(gMpmBlocks[4u * (blocks - 1u) + 1u]);
    if (s >= total) return;
    MpmParticle q = gMpmParticles[gMpmSorted[s]];
    float m = q.xm.w;
    float dx = p.origin.w;
    float3x3 C = float3x3(q.c0.xyz, q.c1.xyz, q.c2.xyz);
    float3x3 A = C * m - mpmStress(q, p) * (p.step.x * q.f0.w * 4.0 / (dx * dx));
    MpmPayload o;
    o.xm = q.xm;
    o.mv = float4(q.vj.xyz * m, 0.0);
    o.a0 = float4(A[0], 0.0);
    o.a1 = float4(A[1], 0.0);
    o.a2 = float4(A[2], 0.0);
    gMpmPayload[s] = o;
}

// One thread per node: the 27 cells whose particles' stencils cover it, in a
// fixed order, each cell's particles by id.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_p2g(uint3 tid : SV_DispatchThreadID)
{
    uint n = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint b = n / MPM_BLOCK_NODES;
    if (b >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z)) return;
    uint key = ALOAD(gMpmBlocks[4u * b]);
    uint l = n % MPM_BLOCK_NODES;
    int3 g = int3(int(key & 1023u), int((key >> 10) & 1023u), int(key >> 20)) * 4
        + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
    float dx = p.origin.w;
    float3 mv = float3(0, 0, 0);
    float m = 0.0;
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        int3 base = g - int3(off);
        uint c = mpmCellOf(base, p);
        if (c == MPM_NO_BLOCK) continue;
        uint first, count;
        mpmCellRange(c, first, count);
        for (uint k = 0u; k < count; k++)
        {
            MpmPayload q = gMpmPayload[first + k];
            float3 fx = (q.xm.xyz - p.origin.xyz) / dx - float3(base);
            float w = mpmWeightAt(mpmWeights(fx.x), off.x) * mpmWeightAt(mpmWeights(fx.y), off.y)
                * mpmWeightAt(mpmWeights(fx.z), off.z);
            float3 dpos = (float3(off) - fx) * dx;
            float3x3 A = float3x3(q.a0.xyz, q.a1.xyz, q.a2.xyz);
            m += w * q.xm.w;
            mv += w * (q.mv.xyz + mul(A, dpos));
        }
    }
    gMpmGrid[n] = float4(mv, m);
}

// Grid velocity: momentum over mass, gravity, then the bed (the heightfield
// under the region) as a separating Coulomb contact.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_grid(uint3 tid : SV_DispatchThreadID)
{
    uint n = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint b = n / MPM_BLOCK_NODES;
    if (b >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z)) return;
    float4 node = gMpmGrid[n];
    if (node.w <= p.step.z) { gMpmGrid[n] = float4(0, 0, 0, node.w); return; }
    float3 v = node.xyz / node.w;
    v.y += p.step.x * p.step.y;
    if (p.dims.w != MPM_NO_BLOCK)
    {
        uint key = ALOAD(gMpmBlocks[4u * b]);
        uint l = n % MPM_BLOCK_NODES;
        int3 g = int3(int(key & 1023u), int((key >> 10) & 1023u), int(key >> 20)) * 4
            + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
        float3 xw = p.origin.xyz + float3(g) * p.origin.w;
        Collider col = gColliders[p.dims.w];
        Frame f = colliderFrame(col, gBodies[col.body]);
        HeightFieldRef hf = heightFieldOf(col);
        float3 xl = frameToLocal(f, xw);
        float h; float3 nl;
        if (hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0)
        {
            float3 nw = quatRotate(f.q, nl);
            float vn = dot(v, nw);
            if (vn < 0.0)
            {
                float3 vt = v - nw * vn;
                float st = length(vt);
                float mu = p.step.w;
                v = st <= -mu * vn ? float3(0, 0, 0) : vt * (1.0 + mu * vn / st);
            }
        }
    }
    gMpmGrid[n] = float4(v, node.w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_g2p(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= p.counts.x) return;
    MpmParticle q = gMpmParticles[i];
    if (!mpmInside(q.xm.xyz, p)) return;
    float dx = p.origin.w;
    float dt = p.step.x;
    int3 base = mpmBaseNode(q.xm.xyz, p);
    float3 fx = (q.xm.xyz - p.origin.xyz) / dx - float3(base);
    float3 wx = mpmWeights(fx.x), wy = mpmWeights(fx.y), wz = mpmWeights(fx.z);
    float3 v = float3(0, 0, 0);
    float3x3 B = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        uint c = mpmCellOf(base + int3(off), p);
        if (c == MPM_NO_BLOCK) continue;
        float w = mpmWeightAt(wx, off.x) * mpmWeightAt(wy, off.y) * mpmWeightAt(wz, off.z);
        float3 vi = gMpmGrid[c].xyz;
        float3 dpos = (float3(off) - fx) * dx;
        v += w * vi;
        B += float3x3(vi * dpos.x, vi * dpos.y, vi * dpos.z) * w;
    }
    // C = (4/dx²)·Σ w vᵢ dposᵀ: B above holds its columns (vᵢ·dpos_k) as rows.
    float3x3 C = transpose(B) * (4.0 / (dx * dx));
    q.xm.xyz += dt * v;
    q.vj.xyz = v;
    q.c0 = float4(C[0], 0.0);
    q.c1 = float4(C[1], 0.0);
    q.c2 = float4(C[2], 0.0);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    float3x3 I = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    F = mul(I + C * dt, F);
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    gMpmParticles[i] = q;
    AMAX2(gMpmCounters[MPM_C_MAX_SPEED], asuint(length(v)));
}
