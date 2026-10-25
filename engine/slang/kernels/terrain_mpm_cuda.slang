// CUDA entries for the terrain plan's indirect dispatches (engine/docs/
// deformable-terrain.md, "Native backends": the CUDA plan). CUDA has no
// indirect dispatch; rather than read binding 46's workgroup counts back to
// the host before each one, the host launches these with a fixed grid of
// MPM_GS_GROUPS workgroups, and each walks the plan's indirect range (the
// counts the GPU itself wrote: blocks, particles, parcels) in a grid-stride
// loop, calling the plan's kernel unchanged for each index. A frame then has
// no host-dependent launch size, so it can be captured as a CUDA graph.
//
// Per-index kernels see the same index they would under the indirect launch.
// Per-workgroup kernels (one workgroup per block, with group barriers) loop at
// workgroup granularity: every thread of a group runs the same trip count, so
// the barriers stay uniform. Bit for bit the indirect dispatch: each index or
// block runs the same code once.
#include "terrain_mpm_fast.slang"

// Workgroups of every grid-stride launch (the host's grid; must agree).
static const uint MPM_GS_GROUPS = 2048u;
static const uint MPM_GS_THREADS = MPM_GS_GROUPS * MPM_GROUP;

// The thread id the indirect launch would have given index i (rows of
// 32768 workgroups, as mpmIndex reads it).
uint3 mpmGsTid(uint i) { return uint3(i % (32768u * 64u), i / (32768u * 64u), 0u); }

uint3 mpmGsGroup(uint b) { return uint3(b % 32768u, b / 32768u, 0u); }

// Indirect slot words: [0] blocks, [3] particle groups, [6] parcel groups.
uint mpmGsBlocks() { return gMpmArgs[0]; }
uint mpmGsParticleThreads() { return gMpmArgs[3] * MPM_GROUP; }
uint mpmGsParcelThreads() { return gMpmArgs[6] * MPM_GROUP; }

#define MPM_GS_PER_INDEX(NAME, KERNEL, COUNT)                                 \
    [shader("compute")]                                                       \
    [numthreads(64, 1, 1)]                                                    \
    void NAME(uint3 tid : SV_DispatchThreadID)                                \
    {                                                                         \
        uint n = COUNT;                                                       \
        for (uint i = tid.x; i < n; i += MPM_GS_THREADS) KERNEL(mpmGsTid(i)); \
    }

// Per-block kernels indexed by thread (block · 64 + node).
#define MPM_GS_PER_NODE(NAME, KERNEL)                                         \
    [shader("compute")]                                                       \
    [numthreads(64, 1, 1)]                                                    \
    void NAME(uint3 tid : SV_DispatchThreadID)                                \
    {                                                                         \
        uint n = mpmGsBlocks() * MPM_GROUP;                                   \
        for (uint i = tid.x; i < n; i += MPM_GS_THREADS) KERNEL(mpmGsTid(i)); \
    }

MPM_GS_PER_INDEX(mpm_touch_gs, mpm_touch, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_count_gs, mpm_count, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_scatter_gs, mpm_scatter, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_payload_gs, mpm_payload, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_g2p_gs, mpm_g2p, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_geostatic_gs, mpm_geostatic, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_source_gs, mpm_source, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_bake_bin_gs, mpm_bake_bin, mpmGsParticleThreads())
MPM_GS_PER_INDEX(mpm_parcels_gs, mpm_parcels, mpmGsParcelThreads())
MPM_GS_PER_NODE(mpm_cell_sort_gs, mpm_cell_sort)
MPM_GS_PER_NODE(mpm_p2g_gs, mpm_p2g)
MPM_GS_PER_NODE(mpm_grid_gs, mpm_grid)
MPM_GS_PER_NODE(mpm_couple_apply_gs, mpm_couple_apply)
MPM_GS_PER_NODE(mpm_p2g_nodes_gs, mpm_p2g_nodes)

// mpm_p2g_cells: one 1024-thread workgroup per block, no barriers.
[shader("compute")]
[numthreads(1024, 1, 1)]
void mpm_p2g_cells_gs(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint n = mpmGsBlocks();
    for (uint b = gid.x; b < n; b += MPM_GS_GROUPS) mpm_p2g_cells(mpmGsGroup(b), lid);
}

// One workgroup per block, with barriers: the loop runs per workgroup.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scan_cells_gs(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint n = mpmGsBlocks();
    for (uint b = gid.x; b < n; b += MPM_GS_GROUPS) mpm_scan_cells(mpmGsGroup(b), lid);
}
