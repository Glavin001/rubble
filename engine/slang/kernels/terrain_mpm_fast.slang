// Native-backend fast paths for terrain_mpm.slang (engine/docs/deformable-terrain.md,
// "Native backends"). Each replaces one or more of the plan's kernels when its
// capability flag is on (engine-metal `TerrainCaps`); the plan's portable kernels
// are unchanged and remain the default. Only native hosts compile this file: it
// uses SIMD-group (wave) intrinsics and bindings past the browser's.
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
// g2p_simd: the plan's mpm_g2p compiled from this file (as mpm_g2p_simd), its
//   top-speed record reduced over the SIMD group before the one atomic max per
//   group (the portable kernel takes one per particle, all on one word). Max is
//   exact and order-free: bit for bit the portable kernel.
#define MPM_RECORD_SPEED(s)                                                   \
    {                                                                         \
        float _mpmTop = WaveActiveMax(s);                                     \
        if (WaveIsFirstLane()) AMAX2(gMpmCounters[MPM_C_MAX_SPEED], asuint(_mpmTop)); \
    }
#include "terrain_mpm.slang"

// Per region cell c (block · 64 + local) and stencil offset o (x + 3y + 9z):
// [27c + o] = (Σ w·(m v + A·dpos), Σ w·m) over the cell's particles.
BIND(53) RWStructuredBuffer<float4> gMpmPartials;

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
