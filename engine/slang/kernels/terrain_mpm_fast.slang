// Native-backend fast paths for terrain_mpm.slang (engine/docs/deformable-terrain.md,
// "Native backends"). Each replaces one or more of the plan's kernels when its
// capability flag is on (engine-metal `TerrainCaps`); the plan's portable kernels
// are unchanged and remain the default. Only native hosts compile this file: it
// uses bindings past the browser's.
//
// A fast path that duplicates a plan kernel's logic must follow it: engine-metal's
// terrain_parity checks every fast path against the portable kernels.
//
// p2g_cells: P2G in two passes, each particle read once.
//   mpm_p2g_cells  one SIMD group per cell, lane o < 27 one of the 27 nodes the
//                  cell's particles reach: it sums the cell's particles in their
//                  sorted order (the loads are uniform across the group) into a
//                  partial per (cell, node).
//   mpm_p2g_nodes  one thread per node: the 27 partials of the cells whose
//                  stencils cover it, in the portable kernel's cell order.
//   The portable mpm_p2g reads every particle 27 times (once per node it
//   reaches). The sums hold the same terms grouped by cell, so the result differs
//   from it only in rounding; it is deterministic (fixed orders, no atomics).
//
// sort_keys: the binning's particle reads, once.
//   mpm_touch_keys    mpm_touch, also writing each particle's base node (8 bytes)
//   mpm_count_keys    mpm_count from that key, replacing it with the cell index
//   mpm_scatter_keys  mpm_scatter from the cell index
//   The portable count and scatter each read the whole 128-byte particle (and
//   look its cell up in the block hash) for its position. Same cells, same
//   counts, same sorted order: bit for bit the portable binning.
#include "terrain_mpm.slang"

// Per region cell c (block · 64 + local) and stencil offset o (x + 3y + 9z):
// [27c + o] = (Σ w·(m v + A·dpos), Σ w·m) over the cell's particles.
BIND(53) RWStructuredBuffer<float4> gMpmPartials;

// Per particle: after mpm_touch_keys its base node (x | y << 16, z), after
// mpm_count_keys its cell (cell, 0); MPM_NO_BLOCK in both words when it is not
// binned (dead, or outside the region).
BIND(54) RWStructuredBuffer<uint2> gMpmKeys;

[shader("compute")]
[numthreads(1024, 1, 1)]
void mpm_p2g_cells(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint b = gid.x + gid.y * 32768u;
    MpmParams p = mpmParams();
    if (b >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z)) return;
    uint o = lid.x & 31u;
    if (o >= 27u) return;
    uint key = ALOAD(gMpmBlocks[4u * b]);
    int3 corner = int3(int(key & 1023u), int((key >> 10) & 1023u), int(key >> 20)) * 4;
    uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
    float dx = p.origin.w;
    for (uint l = lid.x >> 5; l < MPM_BLOCK_NODES; l += 32u)
    {
        uint c = b * MPM_BLOCK_NODES + l;
        uint first, count;
        mpmCellRange(c, first, count);
        if (count == 0u) continue;
        int3 base = corner + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
        float3 mv = float3(0, 0, 0);
        float m = 0.0;
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
        gMpmPartials[27u * c + o] = float4(mv, m);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_p2g_nodes(uint3 tid : SV_DispatchThreadID)
{
    uint n = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint b = n / MPM_BLOCK_NODES;
    if (b >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z)) return;
    uint key = ALOAD(gMpmBlocks[4u * b]);
    uint l = n % MPM_BLOCK_NODES;
    int3 g = int3(int(key & 1023u), int((key >> 10) & 1023u), int(key >> 20)) * 4
        + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
    float3 mv = float3(0, 0, 0);
    float m = 0.0;
    int3 k0 = (g - 2) >> 2;
    uint blk[8];
    mpmBlockCube(k0, p, blk);
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        uint c = mpmCellInCube(g - int3(off), k0, blk);
        if (c == MPM_NO_BLOCK || ALOAD(gMpmCells[4u * c]) == 0u) continue;
        float4 s = gMpmPartials[27u * c + o];
        mv += s.xyz;
        m += s.w;
    }
    gMpmGrid[2u * n] = float4(mv, m);
    gMpmGrid[2u * n + 1u] = float4(0, 0, 0, -1.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_touch_keys(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    uint2 none = uint2(MPM_NO_BLOCK, MPM_NO_BLOCK);
    if (mpmParticleDead(q)) { gMpmKeys[i] = none; return; }
    float3 x = q.xm.xyz;
    if (!mpmInside(x, p))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OUTSIDE);
        gMpmKeys[i] = none;
        return;
    }
    int3 b = mpmBaseNode(x, p);
    // Inside the box, the base node is in [0, dims): 16 bits a coordinate.
    gMpmKeys[i] = uint2(uint(b.x) | (uint(b.y) << 16), uint(b.z));
    int3 lo = b >> 2, hi = (b + 2) >> 2;
    for (int bz = lo.z; bz <= hi.z; bz++)
    for (int by = lo.y; by <= hi.y; by++)
    for (int bx = lo.x; bx <= hi.x; bx++)
        mpmInsertBlock(mpmBlockKey(int3(bx, by, bz)), p);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_count_keys(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    uint2 k = gMpmKeys[i];
    if (k.y == MPM_NO_BLOCK) return;
    int3 b = int3(int(k.x & 0xFFFFu), int(k.x >> 16), int(k.y));
    uint c = mpmCellOf(b, p);
    gMpmKeys[i] = uint2(c, 0u);
    if (c != MPM_NO_BLOCK) AADD2(gMpmCells[4u * c], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scatter_keys(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    uint c = gMpmKeys[i].x;
    if (c == MPM_NO_BLOCK) return;
    uint k;
    AADD3(gMpmCells[4u * c + 2u], 1u, k);
    uint b = c / MPM_BLOCK_NODES;
    gMpmSorted[ALOAD(gMpmBlocks[4u * b + 2u]) + ALOAD(gMpmCells[4u * c + 1u]) + k] = i;
}
