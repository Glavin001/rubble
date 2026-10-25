// Deformable terrain MPM (T2; engine/docs/deformable-terrain.md, "T2 kernel design").
//
// Explicit MLS-MPM on a sparse grid of 4³-cell blocks inside one active region.
// Per substep:
//   mpm_clear       empty the block hash and the counters
//   mpm_touch       insert every block a particle's 3³ stencil reaches (a new
//                   block clears its cells)
//   mpm_write_args  workgroup counts of the per-block kernels (one per block)
//   mpm_count       count particles per cell (the cell of their stencil's base node)
//   mpm_scan_cells  per block: cell starts, and the block's range claimed
//   mpm_scatter     particle ids into their cell's range
//   mpm_cell_sort   each cell's range by particle id (the sums' fixed order)
//   mpm_payload     per particle: mass, momentum and the affine term, in sorted order
//   mpm_p2g         per grid node: gather from the particles whose stencil covers it
//   mpm_grid        velocity, gravity, the heightfield bed (rigid, or the
//                   absorbing ground: mpmAbsorbNode)
//   mpm_g2p         per particle: velocity, C, position and F from the grid
// Rigid coupling (T3, explicit and staggered): mpm_couple_clear and mpm_couple_bin
// list, once per frame, the rigid colliders overlapping the region in its block
// cells; mpm_grid then holds every grid node inside one to the body's surface
// velocity (a separating Coulomb contact) and records the node's momentum change;
// mpm_couple_gather sums each body's reaction (minus those changes) per substep in
// a fixed order and moves the body's proxy velocity by it (and gravity), so the
// material meets a body that slows as it pushes back (a body held at its
// frame-start velocity drove the sand for the whole frame and was thrown up);
// mpm_couple_emit appends one LINEAR_ANGULAR impulse record per dynamic body for
// the rigid step that follows, which applies it at its start and ends the frame
// at the proxy's velocity.
// Ejecta (T4): a particle G2P would carry out of the region's box becomes a
// ballistic parcel holding its quanta (mpm_g2p; the particle's slot is left
// dead); mpm_parcels flies parcels under gravity: one landing over the region
// comes back as a particle, one landing elsewhere on the heightfield adds its
// quanta to the nine samples around it, split by P2G's B-spline in whole quanta
// (mpm_deposit_raise, once per frame), one leaving
// the field's footprint is counted as escaped. Activated quanta always equal
// particles + parcels + deposited + escaped.
// Activation (heightfield → particles) and bake (particles → heightfield) work on
// the window of heightfield samples under the region: mpm_activate_count,
// mpm_activate_scan, mpm_activate_emit, mpm_activate_lower; mpm_bake_bin,
// mpm_bake_raise, mpm_bake_header. Volume moves as whole quanta (params
// column.y m³): a sample lowered by n quanta becomes particles holding exactly n,
// and baking splits each particle's quanta back in whole quanta, the share
// still at rest with the tent from its home sample, the share the MPM moved
// with P2G's B-spline (mpmHomeShare: unmoved material bakes back exactly where
// it came from).
// P2G is a gather (no float atomics, which WGSL lacks): every node sums its
// particles in a fixed order, so runs and backends agree bit for bit.
//
// Region-local node g sits at origin + g·dx. A particle's base node is
// floor((x - origin)/dx - 0.5); its stencil covers base .. base + 2. Particles
// must stay one cell inside the region's box (MPM_OUTSIDE otherwise: the host
// fails the step; ejection to parcels is T4).
#include "world_step.slang"
#include "../physics/mpm_material.slang"

static const uint MPM_GROUP = 64u;
static const uint MPM_BLOCK = 4u;
static const uint MPM_BLOCK_NODES = 64u;
static const uint MPM_NO_BLOCK = 0xFFFFFFFFu;

// gMpmCounters words.
static const uint MPM_C_BLOCKS = 0u;      // blocks inserted this substep
static const uint MPM_C_FLAGS = 1u;       // MPM_OVERFLOW_* | MPM_OUTSIDE
static const uint MPM_C_MAX_SPEED = 2u;   // max |v| after G2P over the frame's substeps (f32 bits; monotone for v ≥ 0)
static const uint MPM_C_PARTICLES = 3u;   // particles in the region (host load, activation)
static const uint MPM_C_BAKE_MAX = 4u;    // largest baked or deposited height (order-preserving bits)
static const uint MPM_C_PARCELS = 5u;     // parcels ever ejected this region (slots, live or not)
static const uint MPM_C_DEPOSITED = 6u;   // quanta parcels have laid on the heightfield
static const uint MPM_C_ESCAPED = 7u;     // quanta that left the heightfield's footprint
static const uint MPM_C_SORTED = 8u;      // sorted slots claimed by blocks this substep
static const uint MPM_C_AIRBORNE = 9u;    // parcels still in flight after this substep
static const uint MPM_C_BODY_SPEED = 10u; // fastest dynamic body coupled this frame (f32 bits)
static const uint MPM_OVERFLOW_BLOCKS = 1u;
static const uint MPM_OVERFLOW_HASH = 2u;
static const uint MPM_OUTSIDE = 4u;
static const uint MPM_CFL = 8u;           // a particle outran the substep's speed bound
static const uint MPM_INVALID = 16u;      // a particle's F or plastic state is not finite
static const uint MPM_OVERFLOW_PARTICLES = 32u; // activation would pass max particles (nothing changed)
static const uint MPM_OVERFLOW_COUPLED = 64u;   // more rigid colliders in the region than coupling slots
static const uint MPM_OVERFLOW_CELL = 128u;     // more colliders in one block cell than it lists
static const uint MPM_OVERFLOW_IMPULSES = 256u; // the rigid step's impulse records were full
static const uint MPM_OVERFLOW_PARCELS = 512u;  // more ejecta than parcel slots
static const uint MPM_OVERFLOW_FIELD = 1024u;   // the bed heightfield has more samples than deposit slots
// MpmParams.boundary.x bits.
static const uint MPM_BOUNDARY_ABSORBING = 1u; // bed and walls absorb (mpmAbsorbNode)
static const uint MPM_BOUNDARY_GEOSTATIC = 2u; // the first step's mpm_geostatic sets F
// MpmParticle.f2.w bits.
static const uint MPM_PARTICLE_DEAD = 1u;       // left as a parcel: no longer simulated
static const uint MPM_PARTICLE_HOME = 2u;       // activated: c1.w, c2.w hold where (the bake's reference)
// Colliders one block cell of the region lists.
static const uint MPM_CELL_COLLIDERS = 8u;
static const uint MPM_NO_SLOT = 0xFFFFFFFFu;
// apply_impulses' mode for a linear impulse through the centre and an angular
// impulse (ccd_step.slang IMPULSE_LINEAR_ANGULAR).
static const uint MPM_IMPULSE_LINEAR_ANGULAR = 2u;

// Material kinds (params u0.w).
static const uint MPM_MATERIAL_NONE = 0u; // no stress: transfers only (tests)
static const uint MPM_MATERIAL_SAND = 1u; // Drucker–Prager sand (mpm_material.slang)
static const uint MPM_MATERIAL_SAND_MC = 2u; // Mohr–Coulomb sand
static const uint MPM_MATERIAL_SOIL_MC = 3u; // cohesive Mohr–Coulomb soil (T5)
static const uint MPM_MATERIAL_ROCK = 4u; // damage-softening rock (T5); damage in MpmParticle.c0.w

struct MpmParams
{
    float4 origin;  // xyz region origin, w = dx
    float4 step;    // x = dt, y = gravity (along y), z = node mass threshold, w = bed friction μ
    uint4 counts;   // x = max particles, y = hash slots (power of two), z = max blocks, w = material
    uint4 dims;     // xyz = region size in cells, w = bed heightfield collider (MPM_NO_BLOCK: none)
    float4 limits;  // x = largest particle speed the substep's dt allows (CFL),
                    // y = the bake's home fade (lattice spacings; TerrainConfig::bake_home_fade)
    float4 sand;    // λ, μ, then sand/soil: friction (DP α, or sin φ for Mohr–Coulomb), cohesion
                    // (DP: apex stretch; dry MC: 0; soil: c·cos φ in Pa); rock: f_t, E·G_f/f_t
    uint4 window;   // heightfield samples under the region: i0, j0, ni, nj
    float4 column;  // bed height (heightfield-local y), volume quantum (m³), lattice spacing, density
    float4 couple;  // x = frame duration (s, bounds pad by motion), y = max coupled colliders (uint bits),
                    // z = max parcels (uint bits; 0: material leaving the region is an error),
                    // w = heightfield samples parcels may deposit on (uint bits)
    float4 source0; // Z-model source: surface point (world), Z
    float4 source1; // flow-centre depth, core radius, strength α, unused
    uint4 boundary; // x = MPM_BOUNDARY_* bits, y = first absorbing-node record in
                    // gMpmCoupled, z = records the region uses, w unused
};

// Per particle (128 B).
struct MpmParticle
{
    // f1.w holds the particle's volume quanta (uint bits).
    float4 xm;      // position, mass
    float4 vj;      // velocity, log Jp (plastic volume)
    float4 c0;      // C rows (APIC affine velocity); c0.w = rock damage
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
// Per node n (block · 64 + local): [2n] after P2G (m·v, m), after mpm_grid (v, m);
// [2n + 1] the rigid contact mpm_grid applied: (momentum change, coupling slot bits).
BIND(45) RWStructuredBuffer<float4> gMpmGrid;
// Indirect dispatch: [0..3) per-block kernels (blocks, 1, 1), [3..6) per-particle
// kernels (⌈particles/64⌉, 1, 1).
BIND(46) RWStructuredBuffer<uint> gMpmArgs;
// Per window sample: [4t] = quanta, [4t + 1] = particles, [4t + 2] = first particle,
// [4t + 3] = quanta baked onto it.
BIND(47) RWStructuredBuffer<AUint> gMpmColumns;
// [0] coupled colliders; per region block cell c (x + X·(y + Y·z)):
// [4 + c·(1 + MPM_CELL_COLLIDERS)] count, then coupling slots.
BIND(48) RWStructuredBuffer<AUint> gMpmCouple;
// Ejecta parcels: [2k] = (position, quanta bits), [2k + 1] = (velocity, mass; 0 once landed).
BIND(50) RWStructuredBuffer<float4> gMpmParcels;
// Per heightfield sample (j·nx + i): quanta deposited this frame.
BIND(51) RWStructuredBuffer<AUint> gMpmDeposit;
// Per coupling slot s: [4s] = (impulse on the body, collider bits), [4s + 1] =
// (angular impulse about its centre, 0), summed over the frame's substeps;
// [4s + 2], [4s + 3] = the body's proxy linear and angular velocity. From
// params.boundary.y on, per absorbing-boundary node record (mpmAbsorbSlot): the
// node's displacement since activation, and the surface height of the column
// above it at activation (the bed heightfield's local y).
BIND(49) RWStructuredBuffer<float4> gMpmCoupled;

groupshared uint gsMpmScan[256];

uint mpmIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }

MpmParams mpmParams() { return gMpmParams[0]; }

uint mpmParticleCount() { return ALOAD(gMpmCounters[MPM_C_PARTICLES]); }

bool mpmParticleDead(MpmParticle q) { return (asuint(q.f2.w) & MPM_PARTICLE_DEAD) != 0u; }

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
        // Most particles find their block already there: read before the CAS
        // (thousands of particles inserting one key serialised on it).
        uint seen = ALOAD(gMpmHash[2u * s]);
        if (seen == key + 1u) return;
        if (seen != 0u) { s = (s + 1u) & (slots - 1u); continue; }
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
            else
            {
                ASTORE(gMpmBlocks[4u * b], key);
                // Its cells start empty (read from mpm_count on).
                for (uint w = 0u; w < 4u * MPM_BLOCK_NODES; w++)
                    ASTORE(gMpmCells[4u * b * MPM_BLOCK_NODES + w], 0u);
            }
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

// The 2×2×2 blocks from block coordinates k0 (each MPM_NO_BLOCK if absent), so a
// 3³ stencil's 27 nodes cost 8 hash lookups instead of 27.
void mpmBlockCube(int3 k0, MpmParams p, out uint blk[8])
{
    for (uint i = 0u; i < 8u; i++)
    {
        int3 k = k0 + int3(int(i & 1u), int((i >> 1) & 1u), int(i >> 2));
        blk[i] = any(k < int3(0, 0, 0)) ? MPM_NO_BLOCK : mpmFindBlock(mpmBlockKey(k), p);
    }
}

// Cell (node) index of g from the cube of blocks at k0 (g's block within it).
uint mpmCellInCube(int3 g, int3 k0, uint blk[8])
{
    int3 d = (g >> 2) - k0;
    if (any(g < int3(0, 0, 0)) || any(d < int3(0, 0, 0)) || any(d > int3(1, 1, 1))) return MPM_NO_BLOCK;
    uint b = blk[uint(d.x) + 2u * uint(d.y) + 4u * uint(d.z)];
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
    if (i == 0u)
    {
        ASTORE(gMpmCounters[MPM_C_BLOCKS], 0u);
        ASTORE(gMpmCounters[MPM_C_SORTED], 0u);
        ASTORE(gMpmCounters[MPM_C_AIRBORNE], 0u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_touch(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float3 x = q.xm.xyz;
    if (!mpmInside(x, p)) { AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OUTSIDE); return; }
    int3 b = mpmBaseNode(x, p);
    int3 lo = b >> 2, hi = (b + 2) >> 2;
    for (int bz = lo.z; bz <= hi.z; bz++)
    for (int by = lo.y; by <= hi.y; by++)
    for (int bx = lo.x; bx <= hi.x; bx++)
        mpmInsertBlock(mpmBlockKey(int3(bx, by, bz)), p);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_write_args(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmParams();
    gMpmArgs[0] = min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    gMpmArgs[1] = 1u;
    gMpmArgs[2] = 1u;
    gMpmArgs[3] = (mpmParticleCount() + MPM_GROUP - 1u) / MPM_GROUP;
    gMpmArgs[4] = 1u;
    gMpmArgs[5] = 1u;
    uint parcels = min(ALOAD(gMpmCounters[MPM_C_PARCELS]), asuint(p.couple.z));
    gMpmArgs[6] = (parcels + MPM_GROUP - 1u) / MPM_GROUP;
    gMpmArgs[7] = 1u;
    gMpmArgs[8] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    float3 x = q.xm.xyz;
    if (mpmParticleDead(q) || !mpmInside(x, p)) return;
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
    if (l == MPM_BLOCK_NODES - 1u)
    {
        // The block's range in gMpmSorted. Where it lands does not change any
        // sum (P2G adds by cell and particle id), so an atomic claim does.
        uint total = gsMpmScan[l];
        uint start;
        AADD3(gMpmCounters[MPM_C_SORTED], total, start);
        ASTORE(gMpmBlocks[4u * b + 1u], total);
        ASTORE(gMpmBlocks[4u * b + 2u], start);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    float3 x = q.xm.xyz;
    if (mpmParticleDead(q) || !mpmInside(x, p)) return;
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

// Cohesive soil (T5): F returned to the stress-space Mohr–Coulomb surface and
// its apex, updating q's F and log Jp, then its Hencky stress.
float3x3 mpmSoilStress(inout MpmParticle q, MpmParams p)
{
    float3x3 zero = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    if (!mpmSvdInputValid(F) || !mpmSandInputValid(float3(1, 1, 1), q.vj.w))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_INVALID);
        return zero;
    }
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 s;
    float logJp;
    mpmSoilProjectMohrCoulomb(sigma, q.vj.w, p.sand.z, p.sand.w, p.sand.x, p.sand.y, s, logJp);
    F = mul(U, mul(float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z), transpose(V)));
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    q.vj.w = logJp;
    return mpmHenckyKirchhoff(U, s, p.sand.x, p.sand.y);
}

// Rock (T5): F is elastic; the damage in q.c0.w grows with the largest
// principal stress, and the stress is Hencky's with its tensile part scaled by
// (1 − d). The crack band is the cell size.
float3x3 mpmRockStress(inout MpmParticle q, MpmParams p)
{
    float3x3 zero = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    if (!mpmSvdInputValid(F) || !(q.c0.w >= 0.0 && q.c0.w <= 1.0))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_INVALID);
        return zero;
    }
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 t;
    float d;
    mpmRockDamage(sigma, q.c0.w, p.sand.x, p.sand.y, p.sand.z, p.sand.w, p.origin.w, t, d);
    q.c0.w = d;
    return mpmPrincipalKirchhoff(U, t);
}

// Kirchhoff stress τ of particle q (zero for MPM_MATERIAL_NONE). Sand returns F
// to its yield surface first (Mohr–Coulomb, or Klár et al.'s Drucker–Prager
// cone), updating q's F and log Jp. Cohesive soil and rock have their own
// functions, ahead of the dry-sand path.
float3x3 mpmStress(inout MpmParticle q, MpmParams p)
{
    if (p.counts.w == MPM_MATERIAL_SOIL_MC) return mpmSoilStress(q, p);
    if (p.counts.w == MPM_MATERIAL_ROCK) return mpmRockStress(q, p);
    float3x3 zero = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (p.counts.w != MPM_MATERIAL_SAND && p.counts.w != MPM_MATERIAL_SAND_MC) return zero;
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    if (!mpmSvdInputValid(F) || !mpmSandInputValid(float3(1, 1, 1), q.vj.w))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_INVALID);
        return zero;
    }
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 s;
    float logJp;
    if (p.counts.w == MPM_MATERIAL_SAND_MC)
        mpmSandProjectMohrCoulomb(sigma, q.vj.w, p.sand.z, p.sand.w, p.sand.x, p.sand.y, s, logJp);
    else mpmSandProject(sigma, q.vj.w, p.sand.z, p.sand.w, p.sand.x, p.sand.y, s, logJp);
    F = mul(U, mul(float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z), transpose(V)));
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    q.vj.w = logJp;
    return mpmHenckyKirchhoff(U, s, p.sand.x, p.sand.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_payload(uint3 tid : SV_DispatchThreadID)
{
    uint s = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (s >= ALOAD(gMpmCounters[MPM_C_SORTED])) return;
    uint id = gMpmSorted[s];
    MpmParticle q = gMpmParticles[id];
    float m = q.xm.w;
    float dx = p.origin.w;
    float3x3 C = float3x3(q.c0.xyz, q.c1.xyz, q.c2.xyz);
    float3x3 tau = mpmStress(q, p);
    if (p.counts.w != MPM_MATERIAL_NONE) gMpmParticles[id] = q;
    float3x3 A = C * m - tau * (p.step.x * q.f0.w * 4.0 / (dx * dx));
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
    // Bases g − 2 .. g lie in the blocks from (g − 2) >> 2.
    int3 k0 = (g - 2) >> 2;
    uint blk[8];
    mpmBlockCube(k0, p, blk);
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        int3 base = g - int3(off);
        uint c = mpmCellInCube(base, k0, blk);
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
    gMpmGrid[2u * n] = float4(mv, m);
    // The node's coupling record: impulse, coupled slot (a float, exact below
    // 2²⁴; −1 for none: an all-ones bit pattern is a NaN WGSL cannot spell).
    gMpmGrid[2u * n + 1u] = float4(0, 0, 0, -1.0);
}

// Signed distance from world point x to collider c (negative inside) and the
// outward normal there. Exact inside, which is all the grid contact uses; boxes and
// hulls by their face planes.
float mpmColliderSdf(Collider c, BodyState b, float3 x, out float3 n)
{
    Frame f = colliderFrame(c, b);
    float3 xl = frameToLocal(f, x);
    float d;
    float3 nl;
    if (c.shape == SHAPE_SPHERE || c.shape == SHAPE_CAPSULE)
    {
        float half = c.shape == SHAPE_CAPSULE ? c.params.y : 0.0;
        float3 core = float3(0.0, clamp(xl.y, -half, half), 0.0);
        float3 r = xl - core;
        float len = length(r);
        nl = len > 1e-12 ? r / len : float3(0, 1, 0);
        d = len - c.params.x;
    }
    else
    {
        HullRef h = hullOf(c);
        d = -1e30;
        nl = float3(0, 1, 0);
        for (uint k = 0u; k < h.nf; k++)
        {
            float4 pl = hullPlane(h, k);
            float s = dot(pl.xyz, xl) - pl.w;
            if (s > d) { d = s; nl = pl.xyz; }
        }
    }
    n = quatRotate(f.q, nl);
    return d;
}

// Bounds of collider c on body b padded by its motion over the frame and a cell.
void mpmColliderBounds(Collider c, BodyState b, MpmParams p, out float3 lo, out float3 hi)
{
    float3 x = transformPoint(b.position, b.orientation, c.localPos);
    float r = c.params.w;
    float pad = (length(b.linearVel) + length(b.angularVel) * r) * p.couple.x + p.origin.w;
    lo = x - (r + pad);
    hi = x + (r + pad);
}

// Region block cell of block coordinates k (region-local), or MPM_NO_BLOCK.
uint mpmLatticeCell(int3 k, MpmParams p)
{
    int3 blocks = int3(p.dims.xyz / MPM_BLOCK);
    if (any(k < int3(0, 0, 0)) || any(k >= blocks)) return MPM_NO_BLOCK;
    return uint(k.x + blocks.x * (k.y + blocks.y * k.z));
}

// Node n (world position from its block) inside a coupled collider: hold it to
// the body's surface velocity (separating, Coulomb friction p.step.w) and record
// the momentum change for that body's reaction.
void mpmRigidContact(uint n, uint b, float m, MpmParams p, inout float3 v)
{
    uint key = ALOAD(gMpmBlocks[4u * b]);
    int3 kb = int3(int(key & 1023u), int((key >> 10) & 1023u), int(key >> 20));
    uint cell = mpmLatticeCell(kb, p);
    if (cell == MPM_NO_BLOCK) return;
    uint base = 4u + cell * (1u + MPM_CELL_COLLIDERS);
    uint count = min(ALOAD(gMpmCouple[base]), MPM_CELL_COLLIDERS);
    if (count == 0u) return;
    uint l = n % MPM_BLOCK_NODES;
    int3 g = kb * 4 + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
    float3 xw = p.origin.xyz + float3(g) * p.origin.w;
    float best = 0.0;
    uint bestSlot = MPM_NO_SLOT, bestCollider = 0xFFFFFFFFu;
    float3 bestN = float3(0, 1, 0);
    for (uint k = 0u; k < count; k++)
    {
        uint slot = ALOAD(gMpmCouple[base + 1u + k]);
        uint ci = asuint(gMpmCoupled[4u * slot].w);
        Collider col = gColliders[ci];
        float3 nn;
        float d = mpmColliderSdf(col, gBodies[col.body], xw, nn);
        // Deepest wins; ties go to the lower collider (order-independent).
        if (d < best || (d == best && d < 0.0 && ci < bestCollider))
        {
            best = d;
            bestSlot = slot;
            bestCollider = ci;
            bestN = nn;
        }
    }
    if (bestSlot == MPM_NO_SLOT) return;
    BodyState body = gBodies[gColliders[bestCollider].body];
    float3 vb = gMpmCoupled[4u * bestSlot + 2u].xyz
        + cross(gMpmCoupled[4u * bestSlot + 3u].xyz, xw - body.position);
    float3 rel = v - vb;
    float vn = dot(rel, bestN);
    if (vn >= 0.0) return;
    float3 vt = rel - bestN * vn;
    float st = length(vt);
    float mu = p.step.w;
    float3 relNew = st <= -mu * vn ? float3(0, 0, 0) : vt * (1.0 + mu * vn / st);
    float3 vNew = vb + relNew;
    // The impulse that would hold the node to the body as if the body were
    // immovable; mpm_couple_apply applies the share the body's mass allows.
    gMpmGrid[2u * n + 1u] = float4(m * (vNew - v), float(bestSlot));
}

// Grid velocity: momentum over mass, gravity, then the bed (the heightfield
// under the region) as a separating Coulomb contact.
// Whether a grid node at xl (the bed's local frame) lies in the rigid bed, and
// the bed's outward normal there. With an activation window open, the rigid
// bed is the window's lowered floor and, around it, the ground left as it was:
// a node under that ground faces the pit through the window's side (a wall),
// never up through the ground's own surface, which would let the region's
// material squeeze out sideways beneath it. Without a window: the field's
// surface.
bool mpmBedNormal(HeightFieldRef hf, float3 xl, MpmParams p, out float3 nl)
{
    nl = float3(0, 1, 0);
    if (p.window.z == 0u || p.window.w == 0u)
    {
        float h;
        return hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0;
    }
    int si = clamp(int(round((xl.x - hf.x0) / hf.dx)), 0, int(hf.nx) - 1);
    int sj = clamp(int(round((xl.z - hf.z0) / hf.dz)), 0, int(hf.nz) - 1);
    int2 lo = int2(p.window.xy), hi = int2(p.window.xy + p.window.zw) - 1;
    float wx = si < lo.x ? 1.0 : (si > hi.x ? -1.0 : 0.0);
    float wz = sj < lo.y ? 1.0 : (sj > hi.y ? -1.0 : 0.0);
    if (wx == 0.0 && wz == 0.0) return xl.y <= hfHeight(hf, uint(si), uint(sj));
    float h;
    if (xl.y < hfHeight(hf, uint(si), uint(sj)))
    {
        nl = normalize(float3(wx, 0.0, wz));
        return true;
    }
    return hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0;
}

// ---- absorbing boundary (RegionBoundary::Absorbing) ----------------------------
//
// The ground around an activated region is an elastic half-space of the
// region's own material, not a rigid wall. Its faces are the floor (the bed
// plane the window was lowered to) and the window's four sides up to the
// unactivated ground's surface. On each face the ground pushes on the material
// with the traction
//   t = p0·n − (k_N·u_n + ρ·c_p·v_n)·n − (k_T·u_t + ρ·c_s·v_t),
// n into the material, u and v the material's displacement since activation
// and its velocity there:
// - the dashpots are Lysmer & Kuhlemeyer's (1969), exact for plane waves at
//   normal incidence;
// - the springs are the viscous-spring boundary's (Liu et al. 2006),
//   k_N = (4/3)·μ/R and k_T = (2/3)·μ/R with R the distance to the region's
//   centre: the static stiffness of the ground beyond, so a change of load
//   gives a bounded displacement instead of creep;
// - p0 is the geostatic traction at activation (ρ·g·d on the floor, K0·ρ·g·d
//   on the sides, K0 = λ/(λ + 2μ)), the stress the material starts with
//   (InitialStress::Geostatic), so material at rest stays at rest.
// The ground has the strength of granular ground: a face carries no tension
// (one whose normal traction would pull lets the node go: sand separates from
// the ground), and no more shear than the bed friction μ allows (one whose
// tangential traction would pass μ·t_n slides at μ·t_n). The dashpots model
// small-strain waves, not plastic flow along the face, which would otherwise
// meet a viscous drag ρ·c_s·v_t far above any frictional strength.
// It is discretised like any traction on the grid: node i takes ∫ w_i·t dS over
// the faces (w_i its B-spline), lumped, so its share of a face is ∫ w_i dS and
// its dashpot and spring act on its own velocity and displacement. That is the
// force the particles' geostatic stress transmits through the face, so the
// activated state is in equilibrium (a floor half a cell above a node layer
// splits between two layers). Each node is solved implicitly (stable for any
// dt) and keeps its displacement in a record of its own (mpmAbsorbSlot).

// The region's absorbing faces in grid units (cells from the region's origin):
// the window's low and high corners (lo.y is the floor), and the first node of
// the three node layers each face reaches (a B-spline reaches 1.5 cells).
struct MpmAbsorbGeom
{
    float3 lo;
    float3 hi;
    int3 band0; // floor (y), low-x and low-z sides
    int3 band1; // high-x and high-z sides (y unused)
};

int mpmBandStart(float plane) { return int(floor(plane - 1.5)) + 1; }

MpmAbsorbGeom mpmAbsorbGeometry(HeightFieldRef hf, Frame f, MpmParams p)
{
    float dx = p.origin.w;
    float3 lo = float3(hf.x0 + (float(p.window.x) - 0.5) * hf.dx, p.column.x,
        hf.z0 + (float(p.window.y) - 0.5) * hf.dz);
    float3 hi = float3(hf.x0 + (float(p.window.x + p.window.z) - 0.5) * hf.dx, p.column.x,
        hf.z0 + (float(p.window.y + p.window.w) - 0.5) * hf.dz);
    MpmAbsorbGeom a;
    a.lo = (frameToWorld(f, lo) - p.origin.xyz) / dx;
    a.hi = (frameToWorld(f, hi) - p.origin.xyz) / dx;
    a.band0 = int3(mpmBandStart(a.lo.x), mpmBandStart(a.lo.y), mpmBandStart(a.lo.z));
    a.band1 = int3(mpmBandStart(a.hi.x), 0, mpmBandStart(a.hi.z));
    return a;
}

// Records: the floor's three node layers over the region's nodes, then each x
// side's three layers, then each z side's. A node in several takes the first.
// (TerrainPlan::boundary_nodes counts them.)
uint mpmAbsorbSlot(int3 g, MpmAbsorbGeom a, MpmParams p)
{
    int3 n = int3(p.dims.xyz) + 1;
    if (any(g < int3(0, 0, 0)) || any(g >= n)) return MPM_NO_SLOT;
    int l = g.y - a.band0.y;
    if (l >= 0 && l < 3) return uint((l * n.z + g.z) * n.x + g.x);
    uint floorRecords = 3u * uint(n.x * n.z);
    for (int side = 0; side < 2; side++)
    {
        l = g.x - (side == 0 ? a.band0.x : a.band1.x);
        if (l >= 0 && l < 3) return floorRecords + uint(((side * 3 + l) * n.y + g.y) * n.z + g.z);
    }
    uint xRecords = 6u * uint(n.y * n.z);
    for (int side = 0; side < 2; side++)
    {
        l = g.z - (side == 0 ? a.band0.z : a.band1.z);
        if (l >= 0 && l < 3) return floorRecords + xRecords + uint(((side * 3 + l) * n.y + g.y) * n.x + g.x);
    }
    return MPM_NO_SLOT;
}

// The node record `slot` stands for (mpmAbsorbSlot's inverse).
int3 mpmAbsorbNodeOf(uint slot, MpmAbsorbGeom a, MpmParams p)
{
    int3 n = int3(p.dims.xyz) + 1;
    int s = int(slot);
    int floorRecords = 3 * n.x * n.z;
    if (s < floorRecords) return int3(s % n.x, a.band0.y + s / (n.x * n.z), (s / n.x) % n.z);
    s -= floorRecords;
    int xRecords = 6 * n.y * n.z;
    if (s < xRecords)
    {
        int k = s / (n.y * n.z);
        return int3((k < 3 ? a.band0.x : a.band1.x) + k % 3, (s / n.z) % n.y, s % n.z);
    }
    s -= xRecords;
    int k = s / (n.y * n.x);
    return int3(s % n.x, (s / n.x) % n.y, (k < 3 ? a.band0.z : a.band1.z) + k % 3);
}

// The quadratic B-spline (in cells) and its integral from −∞ to u.
float mpmSpline(float u)
{
    float a = abs(u);
    return a < 0.5 ? 0.75 - a * a : (a < 1.5 ? 0.5 * (1.5 - a) * (1.5 - a) : 0.0);
}

float mpmSplineCdf(float u)
{
    if (u <= -1.5) return 0.0;
    if (u >= 1.5) return 1.0;
    if (u < -0.5)
    {
        float a = u + 1.5;
        return a * a * a / 6.0;
    }
    if (u < 0.5) return 0.5 + u * (0.75 - u * u / 3.0);
    float b = 1.5 - u;
    return 1.0 - b * b * b / 6.0;
}

// ∫ w over [lo, hi] (cells) for a node at x: its share of that span.
float mpmSplineSpan(float lo, float hi, float x)
{
    return hi > lo ? mpmSplineCdf(hi - x) - mpmSplineCdf(lo - x) : 0.0;
}

// The unactivated ground's surface at sample (i, j) beside the window, in grid units.
float mpmGroundTop(HeightFieldRef hf, Frame f, MpmParams p, int i, int j)
{
    float3 xl = float3(0.0, hfHeight(hf, uint(i), uint(j)), 0.0);
    return (frameToWorld(f, xl).y - p.origin.y) / p.origin.w;
}

// Grid velocity v (after gravity) of node g, of mass m, on the absorbing faces.
// False when no face reaches the node: it keeps the rigid bed.
bool mpmAbsorbNode(int3 g, float m, HeightFieldRef hf, Frame f, MpmParams p, inout float3 v)
{
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    uint slot = mpmAbsorbSlot(g, a, p);
    if (slot == MPM_NO_SLOT || slot >= p.boundary.z) return false;
    float3 r = float3(g);
    // The window's sample row and column nearest the node, and the ground
    // beside each side there (the sides' tops).
    int i0 = int(p.window.x), j0 = int(p.window.y);
    int i1 = i0 + int(p.window.z) - 1, j1 = j0 + int(p.window.w) - 1;
    float3 xl = frameToLocal(f, p.origin.xyz + r * p.origin.w);
    int si = clamp(int(round((xl.x - hf.x0) / hf.dx)), i0, i1);
    int sj = clamp(int(round((xl.z - hf.z0) / hf.dz)), j0, j1);
    float ys = a.lo.y;
    float ix = mpmSplineSpan(a.lo.x, a.hi.x, r.x);
    float iz = mpmSplineSpan(a.lo.z, a.hi.z, r.z);
    // The node's shares of the five faces (cells²): floor, low x, high x, low z, high z.
    float share[5];
    share[0] = mpmSpline(ys - r.y) * ix * iz;
    share[1] = mpmSpline(a.lo.x - r.x) * iz * mpmSplineSpan(ys, mpmGroundTop(hf, f, p, i0 - 1, sj), r.y);
    share[2] = mpmSpline(a.hi.x - r.x) * iz * mpmSplineSpan(ys, mpmGroundTop(hf, f, p, i1 + 1, sj), r.y);
    share[3] = mpmSpline(a.lo.z - r.z) * ix * mpmSplineSpan(ys, mpmGroundTop(hf, f, p, si, j0 - 1), r.y);
    share[4] = mpmSpline(a.hi.z - r.z) * ix * mpmSplineSpan(ys, mpmGroundTop(hf, f, p, si, j1 + 1), r.y);
    if (share[0] + share[1] + share[2] + share[3] + share[4] <= 0.0) return false;
    uint at = p.boundary.y + slot;
    float4 rec = gMpmCoupled[at];
    float3 u = rec.xyz;
    // The material's density, wave speeds and K0; the springs' R.
    float dx = p.origin.w, dt = p.step.x;
    float rho = p.column.w;
    float lam = p.sand.x, mu = p.sand.y;
    float cp = sqrt((lam + 2.0 * mu) / rho), cs = sqrt(mu / rho);
    float k0 = lam / (lam + 2.0 * mu);
    float R = max(length(r - 0.5 * float3(p.dims.xyz)), 1.0) * dx;
    float kn = (4.0 / 3.0) * mu / R, kt = (2.0 / 3.0) * mu / R;
    // Geostatic tractions from the surface of the column over the node at activation.
    float top = (frameToWorld(f, float3(0.0, rec.w, 0.0)).y - p.origin.y) / dx;
    float weight = rho * max(-p.step.y, 0.0) * dx;
    float p0[5];
    p0[0] = weight * max(top - ys, 0.0);
    float lateral = k0 * weight * max(top - max(r.y, ys), 0.0);
    for (uint k = 1u; k < 5u; k++) p0[k] = lateral;
    uint axis[5] = { 1u, 0u, 0u, 2u, 2u };
    float sgn[5] = { 1.0, 1.0, -1.0, 1.0, -1.0 };
    float area = dx * dx;
    float mu_g = p.step.w;
    float3 vNew = v;
    float3 held = float3(0, 0, 0);
    // Solve with every face sticking; then again with any face that would
    // pull let go, and any whose shear would pass the ground's friction
    // sliding (its tangential traction μ·t_n, along the sticking one).
    uint live = 31u, slide = 0u;
    float3 slip[5];
    for (uint k = 0u; k < 5u; k++) slip[k] = float3(0, 0, 0);
    for (uint pass = 0u; pass < 3u; pass++)
    {
        float3 damp = float3(0, 0, 0), stiff = float3(0, 0, 0), push = float3(0, 0, 0);
        held = float3(0, 0, 0);
        for (uint k = 0u; k < 5u; k++)
        {
            if ((live & (1u << k)) == 0u || !(share[k] > 0.0)) continue;
            float s = share[k] * area;
            float3 e = float3(axis[k] == 0u ? 1.0 : 0.0, axis[k] == 1u ? 1.0 : 0.0, axis[k] == 2u ? 1.0 : 0.0);
            damp += s * rho * cp * e;
            stiff += s * kn * e;
            push += s * sgn[k] * p0[k] * e;
            held = max(held, e);
            if ((slide & (1u << k)) != 0u) push += s * slip[k];
            else
            {
                damp += s * rho * cs * (1.0 - e);
                stiff += s * kt * (1.0 - e);
                held = float3(1, 1, 1);
            }
        }
        vNew = (m * v + dt * (push - stiff * u)) / (m + dt * damp + dt * dt * stiff);
        if (pass == 2u) break;
        float3 uNew = u + dt * vNew;
        uint keep = live, slip0 = slide;
        for (uint k = 0u; k < 5u; k++)
        {
            if ((live & (1u << k)) == 0u || !(share[k] > 0.0)) continue;
            float un = sgn[k] * uNew[axis[k]], vn = sgn[k] * vNew[axis[k]];
            float tn = p0[k] - kn * un - rho * cp * vn;
            if (tn < 0.0)
            {
                keep &= ~(1u << k);
                continue;
            }
            if ((slide & (1u << k)) != 0u) continue;
            float3 e = float3(axis[k] == 0u ? 1.0 : 0.0, axis[k] == 1u ? 1.0 : 0.0, axis[k] == 2u ? 1.0 : 0.0);
            float3 tt = -(kt * uNew + rho * cs * vNew) * (1.0 - e);
            // Its two components, without squaring tiny ones.
            float t = axis[k] == 1u ? mpmHypot(tt.x, tt.z) : (axis[k] == 0u ? mpmHypot(tt.y, tt.z) : mpmHypot(tt.x, tt.y));
            if (t > mu_g * tn)
            {
                slide |= 1u << k;
                slip[k] = tt * (mu_g * tn / t);
            }
        }
        if (keep == live && slide == slip0) break;
        live = keep;
    }
    v = vNew;
    // The ground follows the material along every axis a face still holds;
    // along the others (a face let go, or sliding) it stays where it was.
    float3 uNext = u + dt * vNew * held;
    if (any(held > float3(0, 0, 0))) gMpmCoupled[at] = float4(uNext, rec.w);
    return true;
}

// Once at activation, per absorbing-node record: no displacement yet, and the
// surface height of the column over the node (the window sample nearest it).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_absorb_init(uint3 tid : SV_DispatchThreadID)
{
    uint slot = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (slot >= p.boundary.z || mpmActivationRefused()) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    int3 g = mpmAbsorbNodeOf(slot, a, p);
    float3 xl = frameToLocal(f, p.origin.xyz + float3(g) * p.origin.w);
    int si = clamp(int(round((xl.x - hf.x0) / hf.dx)), int(p.window.x), int(p.window.x + p.window.z) - 1);
    int sj = clamp(int(round((xl.z - hf.z0) / hf.dz)), int(p.window.y), int(p.window.y + p.window.w) - 1);
    uint t = uint(sj - int(p.window.y)) * p.window.z + uint(si - int(p.window.x));
    float dh = float(ALOAD(gMpmColumns[4u * t])) * p.column.y / mpmSampleArea(hf);
    gMpmCoupled[p.boundary.y + slot] = float4(0, 0, 0, p.column.x + dh);
}

// Once, in the first step after a geostatic activation (it needs gravity; the
// activation made every particle of the region, and none has moved yet): each
// particle's F from its depth d below the surface of the column it came from
// (its home sample, MpmParticle c1.w and c2.w), the elastic state of a layer
// under its own weight with no lateral strain. Hencky's τ_yy = (λ + 2μ)·ln F_yy
// = −ρ·g·d and τ_xx = τ_zz = λ·ln F_yy = K0·τ_yy.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_geostatic(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || (asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return;
    float modulus = p.sand.x + 2.0 * p.sand.y;
    if (!(modulus > 0.0) || !(p.step.y < 0.0)) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    int si = clamp(int(round((q.c1.w - hf.x0) / hf.dx)), int(p.window.x), int(p.window.x + p.window.z) - 1);
    int sj = clamp(int(round((q.c2.w - hf.z0) / hf.dz)), int(p.window.y), int(p.window.y + p.window.w) - 1);
    uint t = uint(sj - int(p.window.y)) * p.window.z + uint(si - int(p.window.x));
    float top = p.column.x + float(ALOAD(gMpmColumns[4u * t])) * p.column.y / mpmSampleArea(hf);
    float depth = max(top - frameToLocal(f, q.xm.xyz).y, 0.0);
    q.f1.y = exp(p.step.y * p.column.w * depth / modulus);
    gMpmParticles[i] = q;
}

// A node over the unactivated ground beside the window, at or below that
// ground's surface: a separating Coulomb contact with the surface (as the rigid
// bed's hfSurface case). Nodes inside the window, or under the ground's top
// (the walls), are the absorbing faces' and are left alone.
void mpmGroundSurfaceContact(HeightFieldRef hf, Frame f, float3 xl, MpmParams p, inout float3 v)
{
    int si = clamp(int(round((xl.x - hf.x0) / hf.dx)), 0, int(hf.nx) - 1);
    int sj = clamp(int(round((xl.z - hf.z0) / hf.dz)), 0, int(hf.nz) - 1);
    int2 lo = int2(p.window.xy), hi = int2(p.window.xy + p.window.zw) - 1;
    bool outside = si < lo.x || si > hi.x || sj < lo.y || sj > hi.y;
    if (!outside || xl.y < hfHeight(hf, uint(si), uint(sj))) return;
    float h;
    float3 nl;
    if (!(hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0)) return;
    float3 nw = quatRotate(f.q, nl);
    float vn = dot(v, nw);
    if (vn >= 0.0) return;
    float3 vt = v - nw * vn;
    float st = length(vt);
    float mu = p.step.w;
    v = st <= -mu * vn ? float3(0, 0, 0) : vt * (1.0 + mu * vn / st);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_grid(uint3 tid : SV_DispatchThreadID)
{
    uint n = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint b = n / MPM_BLOCK_NODES;
    if (b >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z)) return;
    float4 node = gMpmGrid[2u * n];
    if (node.w <= p.step.z) { gMpmGrid[2u * n] = float4(0, 0, 0, node.w); return; }
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
        // A node the absorbing faces reach (its own function, ahead of the rigid bed).
        if ((p.boundary.x & MPM_BOUNDARY_ABSORBING) != 0u && mpmAbsorbNode(g, node.w, hf, f, p, v))
        {
            // The absorbing faces are the floor and the walls below the
            // ground's top; the ground's own surface beside the window (where
            // the rim spills over) stays rigid ground.
            mpmGroundSurfaceContact(hf, f, frameToLocal(f, xw), p, v);
            mpmRigidContact(n, b, node.w, p, v);
            gMpmGrid[2u * n] = float4(v, node.w);
            return;
        }
        float3 xl = frameToLocal(f, xw);
        float3 nl;
        if (mpmBedNormal(hf, xl, p, nl))
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
    mpmRigidContact(n, b, node.w, p, v);
    gMpmGrid[2u * n] = float4(v, node.w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_g2p(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || !mpmInside(q.xm.xyz, p)) return;
    float dx = p.origin.w;
    float dt = p.step.x;
    int3 base = mpmBaseNode(q.xm.xyz, p);
    float3 fx = (q.xm.xyz - p.origin.xyz) / dx - float3(base);
    float3 wx = mpmWeights(fx.x), wy = mpmWeights(fx.y), wz = mpmWeights(fx.z);
    float3 v = float3(0, 0, 0);
    float3x3 B = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    // Nodes base .. base + 2 lie in the blocks from base >> 2.
    int3 k0 = base >> 2;
    uint blk[8];
    mpmBlockCube(k0, p, blk);
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        uint c = mpmCellInCube(base + int3(off), k0, blk);
        if (c == MPM_NO_BLOCK) continue;
        float w = mpmWeightAt(wx, off.x) * mpmWeightAt(wy, off.y) * mpmWeightAt(wz, off.z);
        float3 vi = gMpmGrid[2u * c].xyz;
        float3 dpos = (float3(off) - fx) * dx;
        v += w * vi;
        B += float3x3(vi * dpos.x, vi * dpos.y, vi * dpos.z) * w;
    }
    // C = (4/dx²)·Σ w vᵢ dposᵀ: B above holds its columns (vᵢ·dpos_k) as rows.
    float3x3 C = transpose(B) * (4.0 / (dx * dx));
    q.xm.xyz += dt * v;
    q.vj.xyz = v;
    q.c0 = float4(C[0], q.c0.w); // c0.w: rock damage, carried
    q.c1 = float4(C[1], q.c1.w); // c1.w, c2.w: where it was activated, carried
    q.c2 = float4(C[2], q.c2.w);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    float3x3 I = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    F = mul(I + C * dt, F);
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    float speed = length(v);
    AMAX2(gMpmCounters[MPM_C_MAX_SPEED], asuint(speed));
    if (!(speed <= p.limits.x)) AOR2(gMpmCounters[MPM_C_FLAGS], MPM_CFL);
    // Leaving the box: a parcel carries the material on (with ejecta enabled).
    uint maxParcels = asuint(p.couple.z);
    if (!mpmInside(q.xm.xyz, p) && maxParcels > 0u)
    {
        uint k;
        AADD3(gMpmCounters[MPM_C_PARCELS], 1u, k);
        if (k < maxParcels)
        {
            gMpmParcels[2u * k] = float4(q.xm.xyz, q.f1.w);
            gMpmParcels[2u * k + 1u] = float4(v, q.xm.w);
            q.f2.w = asfloat(asuint(q.f2.w) | MPM_PARTICLE_DEAD);
        }
        else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARCELS);
    }
    gMpmParticles[i] = q;
}

// ---- activation and bake -----------------------------------------------------

// Area each heightfield sample stands for (its cell around it) and the window
// sample t's grid indices.
float mpmSampleArea(HeightFieldRef hf) { return hf.dx * hf.dz; }

uint2 mpmWindowSample(uint t, MpmParams p)
{
    return uint2(p.window.x + t % p.window.z, p.window.y + t / p.window.z);
}

// Particles a sample's column becomes: a lattice of spacing ≈ column.z over its
// cell and its height above the bed, never more particles than quanta.
uint3 mpmColumnLattice(HeightFieldRef hf, float dh, uint quanta, MpmParams p)
{
    if (quanta == 0u) return uint3(0, 0, 0);
    float s = p.column.z;
    uint3 n = uint3(max(1.0, round(hf.dx / s)), max(1.0, round(dh / s)), max(1.0, round(hf.dz / s)));
    while (n.x * n.y * n.z > quanta)
    {
        if (n.y > 1u) n.y--;
        else if (n.x >= n.z && n.x > 1u) n.x--;
        else n.z--;
    }
    return n;
}

// Per window sample: the quanta above the bed (rounded to whole quanta) and its
// particle count.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_count(uint3 tid : SV_DispatchThreadID)
{
    uint t = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (t >= p.window.z * p.window.w) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint2 ij = mpmWindowSample(t, p);
    float dh = hfHeight(hf, ij.x, ij.y) - p.column.x;
    float area = mpmSampleArea(hf);
    uint quanta = dh > 0.0 ? uint(floor(dh * area / p.column.y + 0.5)) : 0u;
    uint3 n = mpmColumnLattice(hf, float(quanta) * p.column.y / area, quanta, p);
    ASTORE(gMpmColumns[4u * t], quanta);
    ASTORE(gMpmColumns[4u * t + 1u], n.x * n.y * n.z);
    ASTORE(gMpmColumns[4u * t + 3u], 0u);
}

// One workgroup: each sample's first particle (after the particles already in
// the region), in window order; all or nothing against max particles.
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_activate_scan(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmParams();
    uint samples = p.window.z * p.window.w;
    uint chunk = (samples + 255u) / 256u;
    uint first = l * chunk, last = min(first + chunk, samples);
    uint sum = 0u;
    for (uint t = first; t < last; t++) sum += ALOAD(gMpmColumns[4u * t + 1u]);
    gsMpmScan[l] = sum;
    GROUP_SYNC();
    for (uint d = 1u; d < 256u; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    uint have = mpmParticleCount();
    uint start = have + gsMpmScan[l] - sum;
    for (uint t = first; t < last; t++)
    {
        ASTORE(gMpmColumns[4u * t + 2u], start);
        start += ALOAD(gMpmColumns[4u * t + 1u]);
    }
    GROUP_SYNC();
    if (l == 255u)
    {
        uint total = gsMpmScan[255];
        if (have + total > p.counts.x)
            AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARTICLES);
        else ASTORE(gMpmCounters[MPM_C_PARTICLES], have + total);
    }
}

bool mpmActivationRefused() { return (ALOAD(gMpmCounters[MPM_C_FLAGS]) & MPM_OVERFLOW_PARTICLES) != 0u; }

// Per window sample: its particles, sharing its quanta (the first `quanta mod n`
// take one more).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_emit(uint3 tid : SV_DispatchThreadID)
{
    uint t = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (t >= p.window.z * p.window.w || mpmActivationRefused()) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    uint2 ij = mpmWindowSample(t, p);
    uint quanta = ALOAD(gMpmColumns[4u * t]);
    uint total = ALOAD(gMpmColumns[4u * t + 1u]);
    if (total == 0u) return;
    uint first = ALOAD(gMpmColumns[4u * t + 2u]);
    float area = mpmSampleArea(hf);
    float dh = float(quanta) * p.column.y / area;
    uint3 n = mpmColumnLattice(hf, dh, quanta, p);
    float3 lo = float3(hf.x0 + (float(ij.x) - 0.5) * hf.dx, p.column.x, hf.z0 + (float(ij.y) - 0.5) * hf.dz);
    float3 cell = float3(hf.dx / float(n.x), dh / float(n.y), hf.dz / float(n.z));
    uint each = quanta / total, extra = quanta % total;
    for (uint k = 0u; k < total; k++)
    {
        uint3 c = uint3(k % n.x, (k / n.x) % n.y, k / (n.x * n.y));
        float3 xl = lo + (float3(c) + 0.5) * cell;
        uint mine = each + (k < extra ? 1u : 0u);
        float vol = float(mine) * p.column.y;
        float3 xw = frameToWorld(f, xl);
        // Where the bake will see it if it never moves (the same transform
        // mpm_bake_bin applies), so its displacement is then exactly zero.
        float3 home = frameToLocal(f, xw);
        MpmParticle q;
        q.xm = float4(xw, vol * p.column.w);
        q.vj = float4(0, 0, 0, 0);
        q.c0 = float4(0, 0, 0, 0);
        q.c1 = float4(0, 0, 0, home.x);
        q.c2 = float4(0, 0, 0, home.z);
        q.f0 = float4(1, 0, 0, vol);
        q.f1 = float4(0, 1, 0, asfloat(mine));
        q.f2 = float4(0, 0, 1, asfloat(MPM_PARTICLE_HOME));
        gMpmParticles[first + k] = q;
    }
}

// Heightfield records (four samples each) touching the window, as a range.
void mpmWindowRecords(HeightFieldRef hf, MpmParams p, out uint r0, out uint r1)
{
    r0 = (p.window.y * hf.nx + p.window.x) >> 2;
    r1 = ((p.window.y + p.window.w - 1u) * hf.nx + p.window.x + p.window.z - 1u) >> 2;
}

// Per heightfield record (its own four samples, so no two threads share one):
// window samples lowered by the quanta they gave (activate) or raised by the
// quanta baked onto them (bake).
void mpmAdjustRecord(uint r, bool raise, MpmParams p, HeightFieldRef hf)
{
    uint4 rec = gHulls[hf.base + 2u + r];
    float area = mpmSampleArea(hf);
    for (uint c = 0u; c < 4u; c++)
    {
        uint k = 4u * r + c;
        uint i = k % hf.nx, j = k / hf.nx;
        if (j >= hf.nz || i < p.window.x || i >= p.window.x + p.window.z || j < p.window.y || j >= p.window.y + p.window.w)
            continue;
        uint t = (j - p.window.y) * p.window.z + (i - p.window.x);
        float h = asfloat(rec[c]);
        if (raise)
        {
            h += float(ALOAD(gMpmColumns[4u * t + 3u])) * p.column.y / area;
            // Order-preserving bits of h (a float's sign flips the order).
            uint b = asuint(h);
            AMAX2(gMpmCounters[MPM_C_BAKE_MAX], (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u));
        }
        else h -= float(ALOAD(gMpmColumns[4u * t])) * p.column.y / area;
        rec[c] = asuint(h);
    }
    gHulls[hf.base + 2u + r] = rec;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_lower(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmParams();
    if (mpmActivationRefused()) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint r0, r1;
    mpmWindowRecords(hf, p, r0, r1);
    uint r = r0 + mpmIndex(tid);
    if (r <= r1) mpmAdjustRecord(r, false, p, hf);
    if (mpmIndex(tid) == 0u)
    {
        // Lowered samples sit within half a quantum's height of the bed.
        uint4 dims = gHulls[hf.base + 1u];
        float lowest = p.column.x - p.column.y / mpmSampleArea(hf);
        dims.z = asuint(min(asfloat(dims.z), lowest));
        gHulls[hf.base + 1u] = dims;
    }
}

// The bake (mpm_bake_bin) and the parcels' deposits (mpm_parcels) split each
// particle's or parcel's quanta between heightfield samples, in whole quanta
// that sum exactly to the particle's (mpmTentSplit, mpmBsplineSplit).
//
// Two kernels, blended by how far a particle made by activation has moved,
// w = max(0, 1 - |x - x₀|/(fade·s)) (x₀ where it was emitted, in MpmParticle
// c1.w, c2.w; s the activation lattice's spacing, params column.z; fade =
// params limits.y, TerrainConfig::bake_home_fade, by default half a lattice
// spacing: the particle has left its own lattice cell):
// - The share w is the heightfield the particle came from: the tent (bilinear,
//   four samples) at its home sample plus its displacement. Activation lays a
//   sample's particles on a lattice across the sample's cell; at zero
//   displacement this puts all of it back on the home sample, so
//   activate-then-bake is the identity exactly.
// - The share 1 - w is material the MPM moved: the quadratic B-spline (nine
//   samples) at its own position, the kernel P2G gives the grid its mass
//   with. The simulation resolves no finer than that: particles may part
//   within a cell (a sub-cell crack, as along a crater's mirror planes through
//   the impact point, where nothing mixes across) without any force on the
//   grid to close it. A narrower kernel bakes such a gap as a trench a sample
//   wide; the grid's own kernel bakes the mass the simulation computed with.
// Particles not made by activation (loaded by the host, or landed parcels)
// have w = 0. Sand pushed off a buried body (mpmBakeColumn) is placed by that
// rule, not by the flow: it splats with the tent at its new place.

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
        AADD2(gMpmColumns[4u * (uint(tj) * p.window.z + uint(ti)) + 3u], quanta);
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
    MpmParams p = mpmParams();
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float3 x = q.xm.xyz;
    bool pushed = mpmBakeColumn(x, p, max(hf.dx, hf.dz));
    float3 xl = frameToLocal(f, x);
    uint quanta = asuint(q.f1.w);
    int2 s0;
    uint4 parts;
    if (pushed)
    {
        mpmTentSplit((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz), quanta, i, s0, parts);
        for (uint c = 0u; c < 4u; c++) mpmBakeOnto(mpmTentSample(s0, c), parts[c], xl.y, p, hf);
        return;
    }
    float2 uHome;
    float w = mpmHomeShare(q, xl, hf, p, uHome);
    // The heightfield's share (all of it at rest: floor(q + r) = q).
    uint home = min(uint(floor(float(quanta) * w + mpmDither(mpmPcgHash(~i)))), quanta);
    mpmTentSplit(uHome, home, i, s0, parts);
    for (uint c = 0u; c < 4u; c++) mpmBakeOnto(mpmTentSample(s0, c), parts[c], xl.y, p, hf);
    if (home == quanta) return;
    mpmBakeMoved((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz), quanta - home, i, xl.y, p, hf);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_raise(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmParams();
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
    MpmParams p = mpmParams();
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
    ASTORE(gMpmCounters[MPM_C_PARTICLES], 0u);
    ASTORE(gMpmCounters[MPM_C_BAKE_MAX], 0u);
}

// ---- rigid coupling -----------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_couple_clear(uint3 tid : SV_DispatchThreadID)
{
    uint c = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint3 blocks = p.dims.xyz / MPM_BLOCK;
    if (c == 0u)
    {
        ASTORE(gMpmCouple[0], 0u);
        // Once per frame: the frame's top speed, over all its substeps.
        ASTORE(gMpmCounters[MPM_C_MAX_SPEED], 0u);
        ASTORE(gMpmCounters[MPM_C_BODY_SPEED], 0u);
    }
    if (c < blocks.x * blocks.y * blocks.z) ASTORE(gMpmCouple[4u + c * (1u + MPM_CELL_COLLIDERS)], 0u);
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
    int3 k0 = int3(floor((lo - p.origin.xyz) / block));
    int3 k1 = int3(floor((hi - p.origin.xyz) / block));
    int3 blocks = int3(p.dims.xyz / MPM_BLOCK);
    k0 = max(k0, int3(0, 0, 0));
    k1 = min(k1, blocks - 1);
    if (any(k1 < k0)) return;
    uint slot;
    // The region cannot settle while a dynamic body in it still moves: its
    // fastest surface point (linear plus spin over its bounds' half diagonal).
    if (b.invMass > 0.0)
    {
        float reach = 0.5 * length(hi - lo);
        AMAX2(gMpmCounters[MPM_C_BODY_SPEED], asuint(length(b.linearVel) + length(b.angularVel) * reach));
    }
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
    for (int z = k0.z; z <= k1.z; z++)
    for (int y = k0.y; y <= k1.y; y++)
    for (int x = k0.x; x <= k1.x; x++)
    {
        uint base = 4u + mpmLatticeCell(int3(x, y, z), p) * (1u + MPM_CELL_COLLIDERS);
        uint at;
        AADD3(gMpmCouple[base], 1u, at);
        if (at < MPM_CELL_COLLIDERS) ASTORE(gMpmCouple[base + 1u + at], slot);
        else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_CELL);
    }
}

groupshared float3 gsMpmJ[64];
groupshared float3 gsMpmL[64];
groupshared float gsMpmM[64];

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
            int3 blocks = int3(p.dims.xyz / MPM_BLOCK);
            int3 k0 = max(int3(floor((lo - p.origin.xyz) / block)), int3(0, 0, 0));
            int3 k1 = min(int3(floor((hi - p.origin.xyz) / block)), blocks - 1);
            for (int z = k0.z; z <= k1.z; z++)
            for (int y = k0.y; y <= k1.y; y++)
            for (int x = k0.x; x <= k1.x; x++)
            {
                uint bk = mpmFindBlock(mpmBlockKey(int3(x, y, z)), p);
                if (bk == MPM_NO_BLOCK) continue;
                float4 rec = gMpmGrid[2u * (bk * MPM_BLOCK_NODES + l) + 1u];
                if (rec.w != float(slot)) continue;
                int3 g = int3(x, y, z) * 4 + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4));
                float3 xw = p.origin.xyz + float3(g) * p.origin.w;
                J -= rec.xyz;
                L -= cross(xw - b.position, rec.xyz);
                M += gMpmGrid[2u * (bk * MPM_BLOCK_NODES + l)].w;
            }
        }
    }
    gsMpmJ[l] = J;
    gsMpmL[l] = L;
    gsMpmM[l] = M;
    GROUP_SYNC();
    for (uint d = 32u; d > 0u; d >>= 1u)
    {
        if (l < d)
        {
            gsMpmJ[l] += gsMpmJ[l + d];
            gsMpmL[l] += gsMpmL[l + d];
            gsMpmM[l] += gsMpmM[l + d];
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

// ---- ejecta parcels -------------------------------------------------------------

uint mpmOrderedBits(float h)
{
    uint b = asuint(h);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

// A parcel landed at `xl` (heightfield-local, over the field): its quanta
// split onto the nine samples around it with the B-spline (mpmBsplineSplit,
// seeded by `seed`), as the bake splits moved material, all or none. A share
// on a sample past the field's edge escapes. Returns false, raising
// MPM_OVERFLOW_FIELD and depositing nothing, when a share's sample has no
// deposit slot.
bool mpmParcelDeposit(float3 xl, uint quanta, uint seed, MpmParams p, HeightFieldRef hf)
{
    int2 s0;
    uint parts[9];
    mpmBsplineSplit((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz), quanta, seed, s0, parts);
    for (uint k = 0u; k < 9u; k++)
    {
        int2 s = s0 + int2(int(k % 3u), int(k / 3u));
        bool on = s.x >= 0 && s.y >= 0 && s.x < int(hf.nx) && s.y < int(hf.nz);
        if (parts[k] != 0u && on && uint(s.y) * hf.nx + uint(s.x) >= asuint(p.couple.w))
        {
            AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_FIELD);
            return false;
        }
    }
    for (uint k = 0u; k < 9u; k++)
    {
        if (parts[k] == 0u) continue;
        int2 s = s0 + int2(int(k % 3u), int(k / 3u));
        if (s.x < 0 || s.y < 0 || s.x >= int(hf.nx) || s.y >= int(hf.nz))
        {
            AADD2(gMpmCounters[MPM_C_ESCAPED], parts[k]);
            continue;
        }
        AADD2(gMpmDeposit[uint(s.y) * hf.nx + uint(s.x)], parts[k]);
        AADD2(gMpmCounters[MPM_C_DEPOSITED], parts[k]);
    }
    return true;
}

// Per parcel, per substep: ballistic flight (symplectic Euler, as particles), then
// landing on the heightfield (bed collider): back into the region as a particle
// when over its box, else deposited on the four samples around it; off the
// field's footprint, escaped.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_parcels(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (k >= min(ALOAD(gMpmCounters[MPM_C_PARCELS]), asuint(p.couple.z))) return;
    float4 xq = gMpmParcels[2u * k], vm = gMpmParcels[2u * k + 1u];
    if (vm.w <= 0.0) return;
    float dt = p.step.x;
    vm.y += dt * p.step.y;
    xq.xyz += dt * vm.xyz;
    uint quanta = asuint(xq.w);
    bool landed = false;
    if (p.dims.w != MPM_NO_BLOCK)
    {
        Collider col = gColliders[p.dims.w];
        Frame f = colliderFrame(col, gBodies[col.body]);
        HeightFieldRef hf = heightFieldOf(col);
        float3 xl = frameToLocal(f, xq.xyz);
        float h; float3 n;
        if (!hfSurface(hf, xl.x, xl.z, h, n))
        {
            AADD2(gMpmCounters[MPM_C_ESCAPED], quanta);
            landed = true;
        }
        else if (xl.y <= h)
        {
            landed = true;
            if (mpmInside(xq.xyz, p))
            {
                uint slot;
                AADD3(gMpmCounters[MPM_C_PARTICLES], 1u, slot);
                if (slot < p.counts.x)
                {
                    MpmParticle q;
                    float vol = float(quanta) * p.column.y;
                    q.xm = float4(xq.xyz, vm.w);
                    q.vj = float4(vm.xyz, 0.0);
                    // Rock comes back as rubble: thrown fragments are broken.
                    q.c0 = float4(0, 0, 0, p.counts.w == MPM_MATERIAL_ROCK ? 1.0 : 0.0);
                    q.c1 = float4(0, 0, 0, 0);
                    q.c2 = float4(0, 0, 0, 0);
                    q.f0 = float4(1, 0, 0, vol > 0.0 ? vol : vm.w / max(p.column.w, 1e-30));
                    q.f1 = float4(0, 1, 0, asfloat(quanta));
                    q.f2 = float4(0, 0, 1, 0);
                    gMpmParticles[slot] = q;
                }
                else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARTICLES);
            }
            else landed = mpmParcelDeposit(xl, quanta, ~k, p, hf);
        }
    }
    if (!landed) AADD2(gMpmCounters[MPM_C_AIRBORNE], 1u);
    gMpmParcels[2u * k] = xq;
    gMpmParcels[2u * k + 1u] = float4(vm.xyz, landed ? 0.0 : vm.w);
}

// Once per frame, per heightfield record: the quanta deposited on its samples,
// as height, and the field's max height raised over them.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_deposit_raise(uint3 tid : SV_DispatchThreadID)
{
    uint r = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (p.dims.w == MPM_NO_BLOCK) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    if (r >= (min(hf.nx * hf.nz, asuint(p.couple.w)) + 3u) / 4u) return;
    uint4 rec = gHulls[hf.base + 2u + r];
    float area = mpmSampleArea(hf);
    bool changed = false;
    for (uint c = 0u; c < 4u; c++)
    {
        uint k = 4u * r + c;
        if (k >= min(hf.nx * hf.nz, asuint(p.couple.w))) continue;
        uint n;
        AXCHG3(gMpmDeposit[k], 0u, n);
        if (n == 0u) continue;
        float h = asfloat(rec[c]) + float(n) * p.column.y / area;
        rec[c] = asuint(h);
        AMAX2(gMpmCounters[MPM_C_BAKE_MAX], mpmOrderedBits(h));
        changed = true;
    }
    if (changed) gHulls[hf.base + 2u + r] = rec;
}

// After mpm_deposit_raise: the field's max height covers the deposits.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_deposit_header(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmParams();
    if (p.dims.w == MPM_NO_BLOCK) return;
    HeightFieldRef hf = heightFieldOf(gColliders[p.dims.w]);
    uint b;
    AXCHG3(gMpmCounters[MPM_C_BAKE_MAX], 0u, b);
    if (b == 0u) return;
    float top = asfloat((b & 0x80000000u) != 0u ? (b & 0x7FFFFFFFu) : ~b);
    uint4 dims = gHulls[hf.base + 1u];
    dims.w = asuint(max(asfloat(dims.w), top));
    gHulls[hf.base + 1u] = dims;
}

// ---- km/s sources (T4) -------------------------------------------------------------

// Maxwell's Z-model excavation velocity at offset d from the source's surface
// point (+y up, the target below): radial from the flow centre a depth below, with
// the tangential part that makes the flow divergence-free, speed α·r^−Z; inside
// the core a uniform radial expansion. Mirrors engine-terrain ZModel::velocity.
float3 mpmZModelVelocity(float3 d0, float z, float depth, float core, float alpha)
{
    float3 d = d0 + float3(0.0, depth, 0.0);
    float r = length(d);
    if (r < core) return d * (alpha * pow(core, -z) / core);
    float c = -d.y / r;
    float f = (z - 2.0) / (1.0 + c);
    float h = d.x * d.x + d.z * d.z;
    float3 tangential = float3(c * d.x / r, h / (r * r), c * d.z / r) * f;
    return (d / r + tangential) * (alpha * pow(r, -z));
}

// Per live particle: the source's excavation velocity added (the shock and
// contact stage replaced by its outcome; the MPM computes the rest).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_source(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmParams();
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float3 d = q.xm.xyz - p.source0.xyz;
    if (d.y > 0.0) return; // the field is the target's (y ≤ 0)
    if (length(d) > p.source1.w) return; // beyond the transient crater: at rest
    q.vj.xyz += mpmZModelVelocity(d, p.source0.w, p.source1.x, p.source1.y, p.source1.z);
    gMpmParticles[i] = q;
}
