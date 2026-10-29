// Deformable terrain MPM (T2; engine/docs/deformable-terrain.md, "T2 kernel design").
//
// Explicit MLS-MPM on a sparse grid of 4³-cell blocks over the open areas.
//
// Areas (engine/docs/deformable-terrain.md, "Multiple impacts"): one sparse
// domain holds every open area. They share its cell size and block lattice:
// block keys are integer block coordinates from the domain's origin (the first
// area's corner), so material anywhere in any area is one hash. Each area is a
// box of whole blocks with its own window of heightfield samples, columns,
// absorbing faces and bake; its record is gMpmParams[1 + slot] (mpmArea), and
// its box sits at an integer node offset from the domain's origin. A particle
// belongs to the area whose box holds it (boxes keep two blocks apart), and
// is computed in that area's own frame: base node and weights from the area's
// origin, plus its offset (mpmBaseNodeIn), so an area steps exactly as it
// would alone. Kernels that work on one area (activation, a source, a bake)
// take it from gMpmParams[0].area.y (mpmSelArea).
//
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
static const uint MPM_C_FLYING = 12u;     // parcels thrown before this substep's G2P (mpm_parcels flies them)
static const uint MPM_C_GEOSTATIC = 11u;  // nonzero: this substep's mpm_payload sets the geostatic F first
static const uint MPM_C_SURFACE = 13u;    // the material's top under a source point (ordered bits)
static const uint MPM_C_AREA_SPEED = 16u; // + slot: an area's fastest particle this frame (f32 bits)
static const uint MPM_C_AREA_BODY = 32u;  // + slot: an area's fastest coupled body this frame
static const uint MPM_C_AREA_FIRST = 48u; // + slot: the pool's particle count when the area opened
static const uint MPM_MAX_AREAS = 16u;
static const uint MPM_NO_AREA = 0xFFFFFFFFu;
// gMpmParams[0].area.w bits.
static const uint MPM_DOMAIN_CLOSE = 1u;   // this bake closes the last area: the pool empties
static const uint MPM_DOMAIN_SURFACE = 2u; // the source sits on the material's top (mpm_source_surface)
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
static const uint MPM_PARTICLE_FRESH = 4u;      // activated since the last step (mpm_geostatic's)
// Colliders one block cell of the region lists.
static const uint MPM_CELL_COLLIDERS = 256u;
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
                    // gMpmCoupled, z = records the region uses, w = first record in gMpmColumns
    uint4 area;     // domain record: area slots in use, the selected area, the areas a growing
                    // one takes over (bits), MPM_DOMAIN_* flags; an area record: its origin's
                    // node offset from the domain's (int bits), its first block cell in gMpmCouple
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

// Per particle in sorted order, what P2G reads (64 B): position and mass,
// momentum m·v and the affine term A = m·C − dt·V·(4/dx²)·τ, row-major,
// packed: p0 = (m·v, A00), p1 = (A01, A02, A10, A11), p2 = (A12, A20, A21, A22).
struct MpmPayload
{
    float4 xm;
    float4 p0;
    float4 p1;
    float4 p2;
};

BIND(37) RWStructuredBuffer<MpmParams> gMpmParams;
BIND(38) RWStructuredBuffer<AUint> gMpmCounters;
BIND(39) RWStructuredBuffer<MpmParticle> gMpmParticles;
BIND(40) RWStructuredBuffer<MpmPayload> gMpmPayload;
// Open-addressing block hash: [2s] = key + 1 (0 empty), [2s + 1] = block index.
BIND(41) RWStructuredBuffer<AUint> gMpmHash;
// Per block: [4b] = key, [4b + 1] = particles, [4b + 2] = first sorted slot.
BIND(42) RWStructuredBuffer<AUint> gMpmBlocks;
// Per cell (block · 64 + local): [4c] = count, [4c + 1] = start in its block.
BIND(43) RWStructuredBuffer<AUint> gMpmCells;
// [0, N) particle ids by cell (N = max particles); per particle i, [N + i] and
// [2N + i]: its base cell's block key and cell in the block (mpm_touch), then
// its cell index and rank in the cell (mpm_count), so mpm_count and
// mpm_scatter read 4-8 B per particle instead of the particle.
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
// Ejecta parcels: [2k] = (position, quanta bits), [2k + 1] = (velocity, mass; 0 once landed);
// from [2·max parcels] on, per parcel k, x = the particle it left (uint bits).
// Slots k are handed out by an atomic, so their order is the GPU's: nothing
// that changes results depends on k (the deposit's rounding is seeded by the
// particle, and a parcel landing back in the region takes that particle's
// own slot again), so runs agree bit for bit.
BIND(50) RWStructuredBuffer<float4> gMpmParcels;
// Per heightfield sample (j·nx + i): quanta deposited this frame.
BIND(51) RWStructuredBuffer<AUint> gMpmDeposit;
// Per coupling slot s: [4s] = (impulse on the body, collider bits), [4s + 1] =
// (angular impulse about its centre, 0), summed over the frame's substeps;
// [4s + 2], [4s + 3] = the body's proxy linear and angular velocity. From
// params.boundary.y on, per absorbing face point (mpmFaceRecord): the face's
// displacement since activation there, and the surface height of the column
// over it at activation (the bed heightfield's local y).
BIND(49) RWStructuredBuffer<float4> gMpmCoupled;

groupshared uint gsMpmScan[256];

uint mpmIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }

MpmParams mpmParams() { return gMpmParams[0]; }

// Area slot k's record composed onto the domain's: its box (origin, dims, bed
// collider), window, columns, faces and offset; the rest is the domain's.
MpmParams mpmArea(uint k)
{
    MpmParams p = gMpmParams[0];
    MpmParams a = gMpmParams[1u + k];
    p.origin = a.origin;
    p.dims = a.dims;
    p.window = a.window;
    p.column = a.column;
    p.boundary = a.boundary;
    p.area = a.area;
    return p;
}

// The area an activation, a source or a bake works on.
MpmParams mpmSelArea() { return mpmArea(gMpmParams[0].area.y); }

int3 mpmAreaOffset(MpmParams a) { return int3(asint(a.area.x), asint(a.area.y), asint(a.area.z)); }

int3 mpmSlotOffset(uint k)
{
    uint4 a = gMpmParams[1u + k].area;
    return int3(asint(a.x), asint(a.y), asint(a.z));
}

// Record t of an area's window in gMpmColumns (its first word).
uint mpmCol(MpmParams p, uint t) { return 4u * (p.boundary.w + t); }

// x one cell inside area k's box (its stencil stays on the area's grid).
bool mpmInsideArea(float3 x, uint k)
{
    float4 o = gMpmParams[1u + k].origin;
    uint3 d = gMpmParams[1u + k].dims.xyz;
    float3 r = (x - o.xyz) / o.w;
    return all(r >= float3(1, 1, 1)) && all(r < float3(d) - 1.0);
}

// The open area whose box holds x one cell in, or MPM_NO_AREA (boxes keep
// apart, so at most one does).
uint mpmAreaOf(float3 x, MpmParams p)
{
    for (uint k = 0u; k < p.area.x; k++)
    {
        if (gMpmParams[1u + k].dims.x == 0u) continue;
        if (mpmInsideArea(x, k)) return k;
    }
    return MPM_NO_AREA;
}

// Whether the selected area's bake takes material at x: every live particle
// when it is the last area, else those within a block of its box (boxes keep
// two blocks apart, so no other area's are).
bool mpmBakeOwns(float3 x)
{
    MpmParams d = gMpmParams[0];
    if ((d.area.w & MPM_DOMAIN_CLOSE) != 0u) return true;
    float4 o = gMpmParams[1u + d.area.y].origin;
    float3 r = (x - o.xyz) / o.w;
    float3 n = float3(gMpmParams[1u + d.area.y].dims.xyz);
    float pad = float(MPM_BLOCK);
    return all(r >= -pad) && all(r <= n + pad);
}

// The open area whose box (nodes 0 ..= dims) holds domain node g, or MPM_NO_AREA.
uint mpmNodeArea(int3 g, MpmParams p)
{
    for (uint k = 0u; k < p.area.x; k++)
    {
        uint3 d = gMpmParams[1u + k].dims.xyz;
        if (d.x == 0u) continue;
        int3 l = g - mpmSlotOffset(k);
        if (all(l >= int3(0, 0, 0)) && all(l <= int3(d))) return k;
    }
    return MPM_NO_AREA;
}

uint mpmParticleCount() { return ALOAD(gMpmCounters[MPM_C_PARTICLES]); }

bool mpmParticleDead(MpmParticle q) { return (asuint(q.f2.w) & MPM_PARTICLE_DEAD) != 0u; }

// A particle's base node in area k, on the domain's lattice: from the area's
// own origin (as the area alone would compute it), plus its offset.
int3 mpmBaseNodeIn(float3 x, uint k)
{
    float4 o = gMpmParams[1u + k].origin;
    return int3(floor((x - o.xyz) / o.w - 0.5)) + mpmSlotOffset(k);
}

// Block keys: domain block coordinates biased into 11, 10 and 11 bits (the
// top x and z values are left out, so key + 1 never wraps to the hash's empty 0).
static const int3 MPM_KEY_BIAS = int3(1024, 512, 1024);

bool mpmKeyOk(int3 block)
{
    int3 u = block + MPM_KEY_BIAS;
    return all(u >= int3(0, 0, 0)) && u.x < 2047 && u.y < 1024 && u.z < 2047;
}

uint mpmBlockKey(int3 block)
{
    uint3 u = uint3(block + MPM_KEY_BIAS);
    return u.x | (u.y << 11) | (u.z << 21);
}

int3 mpmKeyBlock(uint key) { return int3(int(key & 2047u), int((key >> 11) & 1023u), int(key >> 21)) - MPM_KEY_BIAS; }

// Domain node of local node l (0..63) in the block of `key`.
int3 mpmKeyNode(uint key, uint l) { return mpmKeyBlock(key) * 4 + int3(int(l & 3u), int((l >> 2) & 3u), int(l >> 4)); }

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
    if (!mpmKeyOk(g >> 2)) return MPM_NO_BLOCK;
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
        blk[i] = !mpmKeyOk(k) ? MPM_NO_BLOCK : mpmFindBlock(mpmBlockKey(k), p);
    }
}

// Cell (node) index of g from the cube of blocks at k0 (g's block within it).
uint mpmCellInCube(int3 g, int3 k0, uint blk[8])
{
    int3 d = (g >> 2) - k0;
    if (any(d < int3(0, 0, 0)) || any(d > int3(1, 1, 1))) return MPM_NO_BLOCK;
    uint b = blk[uint(d.x) + 2u * uint(d.y) + 4u * uint(d.z)];
    if (b == MPM_NO_BLOCK) return MPM_NO_BLOCK;
    uint3 l = uint3(g & 3);
    return b * MPM_BLOCK_NODES + l.x + 4u * l.y + 16u * l.z;
}


// Quadratic B-spline weights of the three nodes base + 0..2 along one axis,
// fx = x/dx − base in [0.5, 1.5).
float3 mpmWeights(float fx)
{
    return float3(0.5 * (1.5 - fx) * (1.5 - fx), 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5));
}

float mpmWeightAt(float3 wx, uint o) { return o == 0u ? wx.x : o == 1u ? wx.y : wx.z; }

// The indirect workgroup counts from the counters: blocks, particles, parcels.
void mpmWriteArgs(MpmParams p)
{
    gMpmArgs[0] = min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    gMpmArgs[1] = 1u;
    gMpmArgs[2] = 1u;
    gMpmArgs[3] = (mpmParticleCount() + MPM_GROUP - 1u) / MPM_GROUP;
    gMpmArgs[4] = 1u;
    gMpmArgs[5] = 1u;
    uint parcels = min(ALOAD(gMpmCounters[MPM_C_PARCELS]), asuint(p.couple.z));
    gMpmArgs[6] = (parcels + MPM_GROUP - 1u) / MPM_GROUP;
    // Exactly these fly in mpm_parcels: G2P adds parcels after this, in the
    // GPU's slot order, and its last workgroup would fly some of them.
    ASTORE(gMpmCounters[MPM_C_FLYING], parcels);
    gMpmArgs[7] = 1u;
    gMpmArgs[8] = 1u;
}

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
        // mpm_touch's workgroups (mpm_write_args, fused: one dispatch less
        // per substep).
        mpmWriteArgs(p);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_touch(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    // Only its position and flags (not the whole particle).
    float4 xm = gMpmParticles[i].xm;
    uint flags = asuint(gMpmParticles[i].f2.w);
    uint n = p.counts.x;
    gMpmSorted[n + i] = MPM_NO_BLOCK;
    if ((flags & MPM_PARTICLE_DEAD) != 0u) return;
    float3 x = xm.xyz;
    uint area = mpmAreaOf(x, p);
    if (area == MPM_NO_AREA) { AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OUTSIDE); return; }
    int3 b = mpmBaseNodeIn(x, area);
    int3 lo = b >> 2, hi = (b + 2) >> 2;
    if (!mpmKeyOk(lo) || !mpmKeyOk(hi)) { AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OUTSIDE); return; }
    // Its base cell for mpm_count: block key and the cell in the block.
    uint3 l = uint3(b & 3);
    gMpmSorted[n + i] = mpmBlockKey(b >> 2);
    gMpmSorted[2u * n + i] = l.x + 4u * l.y + 16u * l.z;
    for (int bz = lo.z; bz <= hi.z; bz++)
    for (int by = lo.y; by <= hi.y; by++)
    for (int bx = lo.x; bx <= hi.x; bx++)
        mpmInsertBlock(mpmBlockKey(int3(bx, by, bz)), p);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_write_args(uint3 tid : SV_DispatchThreadID)
{
    mpmWriteArgs(mpmParams());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    uint n = p.counts.x;
    uint key = gMpmSorted[n + i];
    uint c = MPM_NO_BLOCK;
    if (key != MPM_NO_BLOCK)
    {
        uint b = mpmFindBlock(key, p);
        if (b != MPM_NO_BLOCK) c = b * MPM_BLOCK_NODES + gMpmSorted[2u * n + i];
    }
    // Its cell and its rank among the cell's particles, for mpm_scatter (the
    // rank's order is the atomics'; mpm_cell_sort fixes the order).
    gMpmSorted[n + i] = c;
    if (c == MPM_NO_BLOCK) return;
    uint k;
    AADD3(gMpmCells[4u * c], 1u, k);
    gMpmSorted[2u * n + i] = k;
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
    uint n = p.counts.x;
    uint c = gMpmSorted[n + i];
    if (c == MPM_NO_BLOCK) return;
    uint k = gMpmSorted[2u * n + i];
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

// The payload of slot s, the first substep after a geostatic activation:
// particle q's geostatic F first (mpmGeostaticStart), then as mpm_payload.
// A function of its own behind mpm_payload's early return, so the usual
// path's code (and its fast-math results) is unchanged.
void mpmPayloadGeostatic(uint s, MpmParams p)
{
    uint id = gMpmSorted[s];
    MpmParticle q = gMpmParticles[id];
    mpmGeostaticFresh(q, p);
    float m = q.xm.w;
    float dx = p.origin.w;
    float3x3 C = float3x3(q.c0.xyz, q.c1.xyz, q.c2.xyz);
    float3x3 tau = mpmStress(q, p);
    gMpmParticles[id] = q;
    float3x3 A = C * m - tau * (p.step.x * q.f0.w * 4.0 / (dx * dx));
    MpmPayload o;
    o.xm = q.xm;
    o.p0 = float4(q.vj.xyz * m, A[0].x);
    o.p1 = float4(A[0].y, A[0].z, A[1].x, A[1].y);
    o.p2 = float4(A[1].z, A[2].x, A[2].y, A[2].z);
    gMpmPayload[s] = o;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_payload(uint3 tid : SV_DispatchThreadID)
{
    uint s = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (s >= ALOAD(gMpmCounters[MPM_C_SORTED])) return;
    // The first substep after a geostatic activation: its F first.
    if (ALOAD(gMpmCounters[MPM_C_GEOSTATIC]) != 0u)
    {
        mpmPayloadGeostatic(s, p);
        return;
    }
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
    o.p0 = float4(q.vj.xyz * m, A[0].x);
    o.p1 = float4(A[0].y, A[0].z, A[1].x, A[1].y);
    o.p2 = float4(A[1].z, A[2].x, A[2].y, A[2].z);
    gMpmPayload[s] = o;
}

// P2G's lanes per node: each sums the cells o = lane, lane + 4, ... of
// the 27, and the lanes' sums add in lane order: a fixed order, so runs and
// backends still agree bit for bit. Each node's gather is a chain of loads;
// more lanes per node run more of them at once.
static const uint MPM_P2G_LANES = 4u;
groupshared float4 gsMpmP2g[64u * MPM_P2G_LANES];

// One workgroup per block, MPM_P2G_LANES threads per node: the 27 cells whose
// particles' stencils cover the node, each cell's particles by id.
[shader("compute")]
[numthreads(64 * 4, 1, 1)]
void mpm_p2g(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint b = gid.x + gid.y * 32768u;
    uint l = lid.x % MPM_BLOCK_NODES;
    uint lane = lid.x / MPM_BLOCK_NODES;
    MpmParams p = mpmParams();
    bool live = b < min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    float3 mv = float3(0, 0, 0);
    float m = 0.0;
    if (live)
    {
        uint key = ALOAD(gMpmBlocks[4u * b]);
        int3 g = mpmKeyNode(key, l);
        float dx = p.origin.w;
        // The node's area: every particle whose stencil reaches it is that
        // area's, in that area's frame (none: the node has no mass).
        uint area = mpmNodeArea(g, p);
        float3 ao = area == MPM_NO_AREA ? float3(0, 0, 0) : gMpmParams[1u + area].origin.xyz;
        int3 aoff = area == MPM_NO_AREA ? int3(0, 0, 0) : mpmSlotOffset(area);
        // Bases g - 2 .. g lie in the blocks from (g - 2) >> 2.
        int3 k0 = (g - 2) >> 2;
        uint blk[8];
        mpmBlockCube(k0, p, blk);
        for (uint o = lane; o < 27u && area != MPM_NO_AREA; o += MPM_P2G_LANES)
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
                float3 fx = (q.xm.xyz - ao) / dx - float3(base - aoff);
                float w = mpmWeightAt(mpmWeights(fx.x), off.x) * mpmWeightAt(mpmWeights(fx.y), off.y)
                    * mpmWeightAt(mpmWeights(fx.z), off.z);
                float3 dpos = (float3(off) - fx) * dx;
                float3x3 A = float3x3(float3(q.p0.w, q.p1.x, q.p1.y), float3(q.p1.z, q.p1.w, q.p2.x),
                    float3(q.p2.y, q.p2.z, q.p2.w));
                m += w * q.xm.w;
                mv += w * (q.p0.xyz + mul(A, dpos));
            }
        }
    }
    gsMpmP2g[lid.x] = float4(mv, m);
    GROUP_SYNC();
    if (!live || lane != 0u) return;
    float4 sum = gsMpmP2g[l];
    for (uint j = 1u; j < MPM_P2G_LANES; j++) sum += gsMpmP2g[l + j * MPM_BLOCK_NODES];
    uint n = b * MPM_BLOCK_NODES + l;
    gMpmGrid[2u * n] = sum;
    // The node's coupling record: impulse, coupled slot (a float, exact below
    // 2^24; -1 for none: an all-ones bit pattern is a NaN WGSL cannot spell).
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

// The block cell of an area's block k (area-local block coordinates) in
// gMpmCouple's cells (the area's run from its record's area.w), or MPM_NO_BLOCK.
uint mpmLatticeCell(int3 k, MpmParams p)
{
    int3 blocks = int3(p.dims.xyz / MPM_BLOCK);
    if (any(k < int3(0, 0, 0)) || any(k >= blocks)) return MPM_NO_BLOCK;
    uint c = p.area.w + uint(k.x + blocks.x * (k.y + blocks.y * k.z));
    return c < p.counts.z ? c : MPM_NO_BLOCK;
}

// Node n (area-local node g of area p) inside a coupled collider: hold it to
// the body's surface velocity (separating, Coulomb friction p.step.w) and record
// the momentum change for that body's reaction.
void mpmRigidContact(uint n, int3 g, float m, MpmParams p, inout float3 v)
{
    uint cell = mpmLatticeCell(g >> 2, p);
    if (cell == MPM_NO_BLOCK) return;
    uint base = 4u + cell * (1u + MPM_CELL_COLLIDERS);
    uint count = min(ALOAD(gMpmCouple[base]), MPM_CELL_COLLIDERS);
    if (count == 0u) return;
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
// n into the material, v the material's velocity at the face and u the face's
// displacement since activation:
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
// (one whose normal traction would pull lets the material go: sand separates
// from the ground), and no more shear than the bed friction μ allows (one
// whose tangential traction would pass μ·t_n slides at μ·t_n). The dashpots
// model small-strain waves, not plastic flow along the face, which would
// otherwise meet a viscous drag ρ·c_s·v_t far above any frictional strength.
// It is discretised like any traction on the grid: node i takes ∫ w_i·t dS over
// the faces (w_i its B-spline), lumped, so its share of a face is ∫ w_i dS and
// the dashpot acts on its own velocity. That is the force the particles'
// geostatic stress transmits through the face, so the activated state is in
// equilibrium (a floor half a cell above a node layer splits between two
// layers). Each node is solved implicitly (stable for any dt).
// The displacement u belongs to the face, not to the nodes: one record per
// face point (the node lattice's columns through the face, mpmFaceRecord),
// moved once per substep, after G2P, by the material's velocity there (the
// mass-weighted B-spline mean of the node layers across the face,
// mpm_absorb_faces). Every node the face reaches springs against that one
// displacement. A record per node instead integrated each node's own Eulerian
// velocity: under the strike's flow the node layers across a face drifted
// metres apart, and their springs then pushed opposite ways across it.

// The region's absorbing faces in grid units (cells from the region's origin):
// the window's low and high corners (lo.y is the floor).
struct MpmAbsorbGeom
{
    float3 lo;
    float3 hi;
};

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
    return a;
}

// The five faces: floor, low x, high x, low z, high z. Each one's normal axis,
// the sign of its normal into the material, and its plane (grid units).
uint mpmFaceAxis(uint k) { return k == 0u ? 1u : (k < 3u ? 0u : 2u); }

float mpmFaceSign(uint k) { return (k == 2u || k == 4u) ? -1.0 : 1.0; }

float3 mpmAxisVector(uint axis)
{
    return float3(axis == 0u ? 1.0 : 0.0, axis == 1u ? 1.0 : 0.0, axis == 2u ? 1.0 : 0.0);
}

float mpmFacePlane(uint k, MpmAbsorbGeom a)
{
    return k == 0u ? a.lo.y : (k == 1u ? a.lo.x : (k == 2u ? a.hi.x : (k == 3u ? a.lo.z : a.hi.z)));
}

// Face records: the floor's points over the node lattice's (x, z) columns,
// then each x side's over (y, z), then each z side's over (y, x)
// (TerrainPlan::boundary_nodes counts them). The record of face k at node g's
// column.
uint mpmFaceRecord(uint k, int3 g, MpmParams p)
{
    int3 n = int3(p.dims.xyz) + 1;
    if (any(g < int3(0, 0, 0)) || any(g >= n)) return MPM_NO_SLOT;
    if (k == 0u) return uint(g.z * n.x + g.x);
    uint base = uint(n.x * n.z);
    if (k < 3u) return base + (k - 1u) * uint(n.y * n.z) + uint(g.y * n.z + g.z);
    base += 2u * uint(n.y * n.z);
    return base + (k - 3u) * uint(n.y * n.x) + uint(g.y * n.x + g.x);
}

// Face records in all (TerrainPlan::boundary_nodes).
uint mpmFaceCount(MpmParams p)
{
    uint3 n = p.dims.xyz + 1u;
    return n.x * n.z + 2u * n.y * (n.x + n.z);
}

// The face and point (grid units, on its plane) of record `slot`.
float3 mpmFacePoint(uint slot, MpmAbsorbGeom a, MpmParams p, out uint k)
{
    int3 n = int3(p.dims.xyz) + 1;
    int s = int(slot);
    k = 0u;
    if (s < n.x * n.z) return float3(float(s % n.x), a.lo.y, float(s / n.x));
    s -= n.x * n.z;
    if (s < 2 * n.y * n.z)
    {
        k = s < n.y * n.z ? 1u : 2u;
        s %= n.y * n.z;
        return float3(mpmFacePlane(k, a), float(s / n.z), float(s % n.z));
    }
    s -= 2 * n.y * n.z;
    k = s < n.y * n.x ? 3u : 4u;
    s %= n.y * n.x;
    return float3(float(s % n.x), float(s / n.x), mpmFacePlane(k, a));
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

// The window's samples nearest grid point r (clamped into the window).
int2 mpmWindowNearest(float3 r, HeightFieldRef hf, Frame f, MpmParams p)
{
    float3 xl = frameToLocal(f, p.origin.xyz + r * p.origin.w);
    int2 lo = int2(p.window.xy), hi = lo + int2(p.window.zw) - 1;
    return clamp(int2(round((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz))), lo, hi);
}

// The top of side k (1..4) at the window sample s beside it: the unactivated
// ground's surface there.
float mpmSideTop(uint k, int2 s, HeightFieldRef hf, Frame f, MpmParams p)
{
    int2 lo = int2(p.window.xy), hi = lo + int2(p.window.zw) - 1;
    int2 o = k == 1u ? int2(lo.x - 1, s.y) : (k == 2u ? int2(hi.x + 1, s.y) : (k == 3u ? int2(s.x, lo.y - 1) : int2(s.x, hi.y + 1)));
    return mpmGroundTop(hf, f, p, o.x, o.y);
}

// The material's elastic constants for the faces.
struct MpmGround
{
    float rho, cp, cs, k0, weight, mu;
};

MpmGround mpmGroundOf(MpmParams p)
{
    MpmGround m;
    m.rho = p.column.w;
    float lam = p.sand.x;
    m.mu = p.sand.y;
    m.cp = sqrt((lam + 2.0 * m.mu) / m.rho);
    m.cs = sqrt(m.mu / m.rho);
    m.k0 = lam / (lam + 2.0 * m.mu);
    m.weight = m.rho * max(-p.step.y, 0.0) * p.origin.w;
    return m;
}

// The springs per unit area at grid point r: k_N, k_T.
float2 mpmFaceSprings(float3 r, MpmGround m, MpmParams p)
{
    float R = max(length(r - 0.5 * float3(p.dims.xyz)), 1.0) * p.origin.w;
    return float2(4.0 / 3.0, 2.0 / 3.0) * (m.mu / R);
}

// Face k's geostatic traction at height y (grid units) under a column whose
// surface (at activation) is at `top`, floor at ys.
float mpmFaceP0(uint k, float top, float y, float ys, MpmGround m)
{
    return k == 0u ? m.weight * max(top - ys, 0.0) : m.k0 * m.weight * max(top - max(y, ys), 0.0);
}

// Grid velocity v (after gravity) of node g, of mass m, on the absorbing faces.
// False when no face reaches the node: it keeps the rigid bed.
bool mpmAbsorbNode(int3 g, float m, HeightFieldRef hf, Frame f, MpmParams p, inout float3 v)
{
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    float3 r = float3(g);
    int2 s = mpmWindowNearest(r, hf, f, p);
    float ys = a.lo.y;
    float ix = mpmSplineSpan(a.lo.x, a.hi.x, r.x);
    float iz = mpmSplineSpan(a.lo.z, a.hi.z, r.z);
    // The node's shares of the five faces (cells²).
    float share[5];
    share[0] = mpmSpline(ys - r.y) * ix * iz;
    share[1] = mpmSpline(a.lo.x - r.x) * iz * mpmSplineSpan(ys, mpmSideTop(1u, s, hf, f, p), r.y);
    share[2] = mpmSpline(a.hi.x - r.x) * iz * mpmSplineSpan(ys, mpmSideTop(2u, s, hf, f, p), r.y);
    share[3] = mpmSpline(a.lo.z - r.z) * ix * mpmSplineSpan(ys, mpmSideTop(3u, s, hf, f, p), r.y);
    share[4] = mpmSpline(a.hi.z - r.z) * ix * mpmSplineSpan(ys, mpmSideTop(4u, s, hf, f, p), r.y);
    if (share[0] + share[1] + share[2] + share[3] + share[4] <= 0.0) return false;
    MpmGround gr = mpmGroundOf(p);
    float2 kk = mpmFaceSprings(r, gr, p);
    float kn = kk.x, kt = kk.y;
    float dx = p.origin.w, dt = p.step.x;
    float area = dx * dx;
    float mu_g = p.step.w;
    // Each face's displacement at the node's column, and its geostatic traction.
    float3 u[5];
    float p0[5];
    for (uint k = 0u; k < 5u; k++)
    {
        u[k] = float3(0, 0, 0);
        p0[k] = 0.0;
        if (!(share[k] > 0.0)) continue;
        uint rec = mpmFaceRecord(k, g, p);
        if (rec == MPM_NO_SLOT || rec >= p.boundary.z)
        {
            share[k] = 0.0;
            continue;
        }
        float4 d = gMpmCoupled[p.boundary.y + rec];
        u[k] = d.xyz;
        float top = (frameToWorld(f, float3(0.0, d.w, 0.0)).y - p.origin.y) / dx;
        p0[k] = mpmFaceP0(k, top, r.y, ys, gr);
    }
    float3 vNew = v;
    // Solve with every face sticking; then again with any face that would
    // pull let go, and any whose shear would pass the ground's friction
    // sliding (its tangential traction μ·t_n, along the sticking one).
    uint live = 31u, slide = 0u;
    float3 slip[5];
    for (uint k = 0u; k < 5u; k++) slip[k] = float3(0, 0, 0);
    for (uint pass = 0u; pass < 3u; pass++)
    {
        float3 damp = float3(0, 0, 0), stiff = float3(0, 0, 0), push = float3(0, 0, 0);
        for (uint k = 0u; k < 5u; k++)
        {
            if ((live & (1u << k)) == 0u || !(share[k] > 0.0)) continue;
            float sk = share[k] * area;
            float3 e = mpmAxisVector(mpmFaceAxis(k));
            float3 st = kn * e;
            damp += sk * gr.rho * gr.cp * e;
            push += sk * mpmFaceSign(k) * p0[k] * e;
            if ((slide & (1u << k)) != 0u) push += sk * slip[k];
            else
            {
                damp += sk * gr.rho * gr.cs * (1.0 - e);
                st += kt * (1.0 - e);
            }
            // The spring on the face's displacement, implicit in the node's
            // own velocity (the face moves with the material there).
            stiff += sk * st;
            push -= sk * st * u[k];
        }
        vNew = (m * v + dt * push) / (m + dt * damp + dt * dt * stiff);
        if (pass == 2u) break;
        uint keep = live, slip0 = slide;
        for (uint k = 0u; k < 5u; k++)
        {
            if ((live & (1u << k)) == 0u || !(share[k] > 0.0)) continue;
            uint ax = mpmFaceAxis(k);
            float3 uNew = u[k] + dt * vNew;
            float sg = mpmFaceSign(k);
            float tn = p0[k] - kn * sg * uNew[ax] - gr.rho * gr.cp * sg * vNew[ax];
            if (tn < 0.0)
            {
                keep &= ~(1u << k);
                continue;
            }
            if ((slide & (1u << k)) != 0u) continue;
            float3 tt = -(kt * uNew + gr.rho * gr.cs * vNew) * (1.0 - mpmAxisVector(ax));
            // Its two components, without squaring tiny ones.
            float t = ax == 1u ? mpmHypot(tt.x, tt.z) : (ax == 0u ? mpmHypot(tt.y, tt.z) : mpmHypot(tt.x, tt.y));
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
    return true;
}

// Once at activation, per face record: no displacement yet, and the surface
// height of the column over the face point (the window sample nearest it).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_absorb_init(uint3 tid : SV_DispatchThreadID)
{
    uint slot = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (slot >= min(p.boundary.z, mpmFaceCount(p)) || mpmActivationRefused()) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    uint k;
    int2 s = mpmWindowNearest(mpmFacePoint(slot, a, p, k), hf, f, p);
    uint t = uint(s.y - int(p.window.y)) * p.window.z + uint(s.x - int(p.window.x));
    float dh = float(ALOAD(gMpmColumns[mpmCol(p, t)])) * p.column.y / mpmSampleArea(hf);
    float3 u = float3(0, 0, 0);
    if (gMpmParams[0].area.z != 0u) u = mpmMergedFaceDisplacement(k, mpmFacePoint(slot, a, p, k), a, p, hf, f);
    gMpmCoupled[p.boundary.y + slot] = float4(u, p.column.x + dh);
}

// A grown area's face point that one of the areas it took over had too (the
// same face, on the same plane: a side that did not move, or the floor at the
// same bed) keeps that face's displacement; elsewhere the ground beyond starts
// at rest.
float3 mpmMergedFaceDisplacement(uint k, float3 r, MpmAbsorbGeom a, MpmParams p, HeightFieldRef hf, Frame f)
{
    uint mask = gMpmParams[0].area.z;
    uint ax = mpmFaceAxis(k);
    int3 g = int3(round(r)) + mpmAreaOffset(p);
    float plane = mpmFacePlane(k, a) + float(mpmAreaOffset(p)[ax]);
    for (uint o = 0u; o < MPM_MAX_AREAS; o++)
    {
        if ((mask & (1u << o)) == 0u) continue;
        MpmParams q = mpmArea(o);
        if (q.boundary.z == 0u) continue;
        MpmAbsorbGeom b = mpmAbsorbGeometry(hf, f, q);
        int3 qo = mpmAreaOffset(q);
        if (!(abs(mpmFacePlane(k, b) + float(qo[ax]) - plane) < 1e-3)) continue;
        uint rec = mpmFaceRecord(k, g - qo, q);
        if (rec == MPM_NO_SLOT || rec >= q.boundary.z) continue;
        return gMpmCoupled[q.boundary.y + rec].xyz;
    }
    return float3(0, 0, 0);
}

// Per face record, each substep after G2P: the face point moves with the
// material's velocity there, the mass-weighted B-spline mean of the node
// layers across the face, while the material presses on it. It stays where it
// was while the face lets the material go (its normal traction would pull, as
// mpmAbsorbNode decides).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_absorb_faces(uint3 tid : SV_DispatchThreadID)
{
    // Face records of every open area: thread s is record s of the face pool.
    uint s = mpmIndex(tid);
    MpmParams pd = mpmParams();
    uint at = 4u * asuint(pd.couple.y) + s;
    uint area = MPM_NO_AREA;
    for (uint a = 0u; a < pd.area.x; a++)
    {
        uint4 bd = gMpmParams[1u + a].boundary;
        if (gMpmParams[1u + a].dims.x != 0u && at >= bd.y && at < bd.y + bd.z) area = a;
    }
    if (area == MPM_NO_AREA) return;
    MpmParams p = mpmArea(area);
    uint slot = at - p.boundary.y;
    int3 off = mpmAreaOffset(p);
    if (slot >= min(p.boundary.z, mpmFaceCount(p))) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    uint k;
    float3 r = mpmFacePoint(slot, a, p, k);
    uint ax = mpmFaceAxis(k);
    // Only where a node reaches the face: within 1.5 cells of the window's
    // span, and of a side's floor and top (the ground beside it).
    if (any(r.xz < a.lo.xz - 1.5) || any(r.xz > a.hi.xz + 1.5)) return;
    if (k != 0u && (r.y < a.lo.y - 1.5 || r.y > mpmSideTop(k, mpmWindowNearest(r, hf, f, p), hf, f, p) + 1.5))
        return;
    float plane = r[ax];
    int first = int(floor(plane - 1.5)) + 1;
    float3 mv = float3(0, 0, 0);
    float mw = 0.0;
    for (int l = 0; l < 3; l++)
    {
        int3 g = int3(round(r));
        g[ax] = first + l;
        float w = mpmSpline(plane - float(g[ax]));
        uint c = mpmCellOf(g + off, p);
        if (c == MPM_NO_BLOCK || !(w > 0.0)) continue;
        float4 node = gMpmGrid[2u * c];
        if (!(node.w > p.step.z)) continue;
        mv += w * node.w * node.xyz;
        mw += w * node.w;
    }
    if (!(mw > 0.0)) return;
    float3 v = mv / mw;
    float4 d = gMpmCoupled[p.boundary.y + slot];
    MpmGround gr = mpmGroundOf(p);
    float2 kk = mpmFaceSprings(r, gr, p);
    float dx = p.origin.w, dt = p.step.x;
    float top = (frameToWorld(f, float3(0.0, d.w, 0.0)).y - p.origin.y) / dx;
    float p0 = mpmFaceP0(k, top, r.y, a.lo.y, gr);
    float sg = mpmFaceSign(k);
    float3 uNew = d.xyz + dt * v;
    float tn = p0 - kk.x * sg * uNew[ax] - gr.rho * gr.cp * sg * v[ax];
    if (tn < 0.0) return;
    gMpmCoupled[p.boundary.y + slot] = float4(uNew, d.w);
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
    if (mpmParticleDead(q)) return;
    if (mpmGeostaticFresh(q, p)) gMpmParticles[i] = q;
}

// The geostatic start of particle q when it was activated since the last
// step (MPM_PARTICLE_FRESH, which it clears): its area's (the others' material
// is moving already). True when q changed.
bool mpmGeostaticFresh(inout MpmParticle q, MpmParams p)
{
    if ((asuint(q.f2.w) & MPM_PARTICLE_FRESH) == 0u) return false;
    q.f2.w = asfloat(asuint(q.f2.w) & ~MPM_PARTICLE_FRESH);
    uint area = mpmAreaOf(q.xm.xyz, p);
    if (area != MPM_NO_AREA) mpmGeostaticStart(q, mpmArea(area));
    return true;
}

// The geostatic F of activated particle q (see mpm_geostatic); false when it
// takes none. The plan applies it in the first substep's mpm_payload, just
// before the stress (MPM_C_GEOSTATIC): the same F, one pass over the
// particles less on the strike frame.
bool mpmGeostaticStart(inout MpmParticle q, MpmParams p)
{
    if ((asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return false;
    float modulus = p.sand.x + 2.0 * p.sand.y;
    if (!(modulus > 0.0) || !(p.step.y < 0.0)) return false;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    int si = clamp(int(round((q.c1.w - hf.x0) / hf.dx)), int(p.window.x), int(p.window.x + p.window.z) - 1);
    int sj = clamp(int(round((q.c2.w - hf.z0) / hf.dz)), int(p.window.y), int(p.window.y + p.window.w) - 1);
    uint t = uint(sj - int(p.window.y)) * p.window.z + uint(si - int(p.window.x));
    float top = p.column.x + float(ALOAD(gMpmColumns[mpmCol(p, t)])) * p.column.y / mpmSampleArea(hf);
    float depth = max(top - frameToLocal(f, q.xm.xyz).y, 0.0);
    q.f1.y = exp(p.step.y * p.column.w * depth / modulus);
    return true;
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
    // mpm_payload has set the geostatic F: later substeps do not.
    if (n == 0u) ASTORE(gMpmCounters[MPM_C_GEOSTATIC], 0u);
    float4 node = gMpmGrid[2u * n];
    if (node.w <= p.step.z) { gMpmGrid[2u * n] = float4(0, 0, 0, node.w); return; }
    float3 v = node.xyz / node.w;
    v.y += p.step.x * p.step.y;
    // The node's area, in its own frame (node g from the area's origin).
    uint area = mpmNodeArea(mpmKeyNode(ALOAD(gMpmBlocks[4u * b]), n % MPM_BLOCK_NODES), p);
    if (area == MPM_NO_AREA)
    {
        gMpmGrid[2u * n] = float4(v, node.w);
        return;
    }
    int3 g = mpmKeyNode(ALOAD(gMpmBlocks[4u * b]), n % MPM_BLOCK_NODES) - mpmSlotOffset(area);
    p = mpmArea(area);
    if (p.dims.w != MPM_NO_BLOCK)
    {
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
            mpmRigidContact(n, g, node.w, p, v);
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
    mpmRigidContact(n, g, node.w, p, v);
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
    if (mpmParticleDead(q)) return;
    uint area = mpmAreaOf(q.xm.xyz, p);
    if (area == MPM_NO_AREA) return;
    float dx = p.origin.w;
    float dt = p.step.x;
    // In the area's frame (exactly as the area alone), on the domain's lattice.
    float3 ao = gMpmParams[1u + area].origin.xyz;
    int3 aoff = mpmSlotOffset(area);
    int3 local = int3(floor((q.xm.xyz - ao) / dx - 0.5));
    float3 fx = (q.xm.xyz - ao) / dx - float3(local);
    int3 base = local + aoff;
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
    // Per area only with more than slot 0 in use (one area's is the frame's:
    // TerrainPlan::check), sparing every particle a second contended atomic.
    if (p.area.x > 1u) AMAX2(gMpmCounters[MPM_C_AREA_SPEED + area], asuint(speed));
    if (!(speed <= p.limits.x)) AOR2(gMpmCounters[MPM_C_FLAGS], MPM_CFL);
    // Leaving its area's box: a parcel carries the material on (with ejecta enabled).
    uint maxParcels = asuint(p.couple.z);
    if (!mpmInsideArea(q.xm.xyz, area) && maxParcels > 0u)
    {
        uint k;
        AADD3(gMpmCounters[MPM_C_PARCELS], 1u, k);
        if (k < maxParcels)
        {
            gMpmParcels[2u * k] = float4(q.xm.xyz, q.f1.w);
            gMpmParcels[2u * k + 1u] = float4(v, q.xm.w);
            gMpmParcels[2u * maxParcels + k] = float4(asfloat(i), 0, 0, 0);
            // In flight from now (mpm_parcels flies it from the next substep).
            AADD2(gMpmCounters[MPM_C_AIRBORNE], 1u);
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

// The area being taken over by a growing one (gMpmParams[0].area.z) whose
// window holds sample ij, or MPM_NO_AREA.
uint mpmMergedAt(uint2 ij)
{
    uint mask = gMpmParams[0].area.z;
    for (uint k = 0u; k < MPM_MAX_AREAS; k++)
    {
        if ((mask & (1u << k)) == 0u) continue;
        uint4 w = gMpmParams[1u + k].window;
        if (ij.x >= w.x && ij.x < w.x + w.z && ij.y >= w.y && ij.y < w.y + w.w) return k;
    }
    return MPM_NO_AREA;
}

// Per window sample: the quanta above the bed (rounded to whole quanta), to
// emit now ([3]), all it has given ([0]), and its particle count. A sample a
// growing area takes over is active already: it keeps the quanta it gave, and
// gives only the slab between its old bed and a deeper new one.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_count(uint3 tid : SV_DispatchThreadID)
{
    uint t = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (t >= p.window.z * p.window.w) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint2 ij = mpmWindowSample(t, p);
    float dh = hfHeight(hf, ij.x, ij.y) - p.column.x;
    float area = mpmSampleArea(hf);
    uint quanta = dh > 0.0 ? uint(floor(dh * area / p.column.y + 0.5)) : 0u;
    uint total = quanta;
    uint old = mpmMergedAt(ij);
    if (old != MPM_NO_AREA)
    {
        MpmParams o = mpmArea(old);
        uint to = (ij.y - o.window.y) * o.window.z + (ij.x - o.window.x);
        if (!(p.column.x < o.column.x)) quanta = 0u;
        total = ALOAD(gMpmColumns[mpmCol(o, to)]) + quanta;
    }
    uint3 n = mpmColumnLattice(hf, float(quanta) * p.column.y / area, quanta, p);
    ASTORE(gMpmColumns[mpmCol(p, t)], total);
    ASTORE(gMpmColumns[mpmCol(p, t) + 1u], n.x * n.y * n.z);
    ASTORE(gMpmColumns[mpmCol(p, t) + 3u], quanta);
}

// One workgroup: each sample's first particle (after the particles already in
// the region), in window order; all or nothing against max particles.
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_activate_scan(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmSelArea();
    uint samples = p.window.z * p.window.w;
    uint chunk = (samples + 255u) / 256u;
    uint first = l * chunk, last = min(first + chunk, samples);
    uint sum = 0u;
    for (uint t = first; t < last; t++) sum += ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
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
        ASTORE(gMpmColumns[mpmCol(p, t) + 2u], start);
        start += ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
    }
    GROUP_SYNC();
    if (l == 255u)
    {
        uint total = gsMpmScan[255];
        if (have + total > p.counts.x)
            AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARTICLES);
        else
        {
            ASTORE(gMpmCounters[MPM_C_PARTICLES], have + total);
            // The area's first slot (a grown one keeps its oldest area's): the
            // bake's dithers count slots from it, so an area bakes as it
            // would alone however many particles other areas hold.
            uint first = have;
            uint mask = gMpmParams[0].area.z;
            for (uint k = 0u; k < MPM_MAX_AREAS; k++)
                if ((mask & (1u << k)) != 0u) first = min(first, ALOAD(gMpmCounters[MPM_C_AREA_FIRST + k]));
            ASTORE(gMpmCounters[MPM_C_AREA_FIRST + gMpmParams[0].area.y], first);
        }
    }
}

bool mpmActivationRefused() { return (ALOAD(gMpmCounters[MPM_C_FLAGS]) & MPM_OVERFLOW_PARTICLES) != 0u; }

// Per particle of the window (one thread each: a column's particles were a
// serial loop of one thread per sample before, the strike frame's longest
// activation kernel): the sample whose range holds it, by binary search over
// the samples' first particles, and its place in that sample's lattice. The
// first `quanta mod n` of a sample take one more quantum. Writes exactly what
// the per-sample loop wrote.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_emit(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    uint samples = p.window.z * p.window.w;
    if (samples == 0u || mpmActivationRefused() || i >= mpmParticleCount()) return;
    // Particles already in the pool before this activation are not its.
    if (i < ALOAD(gMpmColumns[mpmCol(p, 0u) + 2u])) return;
    // The last sample whose first particle is at or before i (samples with
    // no particles share the next one's first, so this one holds i).
    uint lo = 0u, hi = samples - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1u) / 2u;
        if (ALOAD(gMpmColumns[mpmCol(p, mid) + 2u]) <= i) lo = mid;
        else hi = mid - 1u;
    }
    uint t = lo;
    uint total = ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
    uint first = ALOAD(gMpmColumns[mpmCol(p, t) + 2u]);
    if (i >= first + total) return;
    uint k = i - first;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    uint2 ij = mpmWindowSample(t, p);
    // The quanta it gives now (a grown area's old sample: only a deeper slab).
    uint quanta = ALOAD(gMpmColumns[mpmCol(p, t) + 3u]);
    float area = mpmSampleArea(hf);
    float dh = float(quanta) * p.column.y / area;
    uint3 n = mpmColumnLattice(hf, dh, quanta, p);
    float3 lo3 = float3(hf.x0 + (float(ij.x) - 0.5) * hf.dx, p.column.x, hf.z0 + (float(ij.y) - 0.5) * hf.dz);
    float3 cell = float3(hf.dx / float(n.x), dh / float(n.y), hf.dz / float(n.z));
    uint each = quanta / total, extra = quanta % total;
    uint3 c = uint3(k % n.x, (k / n.x) % n.y, k / (n.x * n.y));
    float3 xl = lo3 + (float3(c) + 0.5) * cell;
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
    q.f2 = float4(0, 0, 1, asfloat(MPM_PARTICLE_HOME | MPM_PARTICLE_FRESH));
    gMpmParticles[i] = q;
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
            h += float(ALOAD(gMpmColumns[mpmCol(p, t) + 3u])) * p.column.y / area;
            // Order-preserving bits of h (a float's sign flips the order).
            uint b = asuint(h);
            AMAX2(gMpmCounters[MPM_C_BAKE_MAX], (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u));
        }
        else
        {
            // The quanta it gave now; [3] then starts the bake's tally at 0.
            h -= float(ALOAD(gMpmColumns[mpmCol(p, t) + 3u])) * p.column.y / area;
            ASTORE(gMpmColumns[mpmCol(p, t) + 3u], 0u);
        }
        rec[c] = asuint(h);
    }
    gHulls[hf.base + 2u + r] = rec;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_lower(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmSelArea();
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
//
// On the absorbing ground the bake measures positions in the unactivated
// ground's frame (mpmGroundShift). That ground is a heightfield and cannot
// move, but the absorbing walls give, and the material against them moves
// out with them: the window's edge columns end displaced outward by the
// walls' give (metres at the browser demo's scale, where the ground is
// 50 MPa sand 350 m deep). Splatted in the region's frame, that shift crosses
// the window's edge and folds back onto its last samples (mpmBakeOnto): a
// ridge a sample wide, tens of metres high, over a deficit spread inside. In
// the ground's frame the walls are where they were: each particle's position
// has the walls' displacement taken out, interpolated linearly between
// opposite walls (the uniform strain an elastic layer between two displaced
// walls takes). The walls' displacement is their edge columns' mean outward
// displacement (mpm_bake_shift): material can only move out past a side's
// plane by moving the ground with it. (The faces' own records hold the grid's
// velocity at the plane, integrated, which the dashpot-held node layers there
// pull to a fraction of the material's a cell in.) Volume is kept exactly;
// the region's lateral give comes back as a uniform rise of its columns.

// Once before the bake on the absorbing ground, per sample along the window's
// edge (low-x then high-x side's rows, then low-z then high-z side's
// columns): its column's mean displacement across that side (quanta-weighted,
// the particles activation made there, in slot order), into the face records'
// space (the faces are done with).
// The list (in gMpmCells, three words each: count, start, cursor) of the
// window's edge sample t (window-local ti, tj), or MPM_NO_SLOT for an inner
// sample: low-z row, high-z row, then the low-x and high-x columns between.
uint mpmEdgeList(int ti, int tj, MpmParams p)
{
    int ni = int(p.window.z), nj = int(p.window.w);
    if (tj == 0) return uint(ti);
    if (tj == nj - 1) return uint(ni + ti);
    if (ti == 0) return uint(2 * ni + tj - 1);
    if (ti == ni - 1) return uint(2 * ni + nj - 2 + tj - 1);
    return MPM_NO_SLOT;
}

// The edge list of particle q's home sample in the selected area p (live,
// made by activation, inside p's box), or MPM_NO_SLOT.
uint mpmParticleEdgeList(MpmParticle q, MpmParams p, HeightFieldRef hf)
{
    if (mpmParticleDead(q) || (asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return MPM_NO_SLOT;
    if (!mpmBakeOwns(q.xm.xyz)) return MPM_NO_SLOT;
    int si = clamp(int(round((q.c1.w - hf.x0) / hf.dx)), int(p.window.x), int(p.window.x + p.window.z) - 1);
    int sj = clamp(int(round((q.c2.w - hf.z0) / hf.dz)), int(p.window.y), int(p.window.y + p.window.w) - 1);
    return mpmEdgeList(si - int(p.window.x), sj - int(p.window.y), p);
}

// The bake's edge lists: the particles activation made on each sample along
// the selected area's window edge, by slot (a counting sort, so the walls'
// give sums them in a fixed order however the pool's slots lie).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_clear(uint3 tid : SV_DispatchThreadID)
{
    uint e = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (e >= 2u * (p.window.z + p.window.w)) return;
    for (uint w = 0u; w < 3u; w++) ASTORE(gMpmCells[3u * e + w], 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmSelArea();
    HeightFieldRef hf = heightFieldOf(gColliders[p.dims.w]);
    uint e = mpmParticleEdgeList(gMpmParticles[i], p, hf);
    if (e != MPM_NO_SLOT) AADD2(gMpmCells[3u * e], 1u);
}

// One workgroup: each list's start (exclusive prefix over the lists' counts).
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_bake_edge_scan(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmSelArea();
    uint lists = 2u * (p.window.z + p.window.w);
    uint chunk = (lists + 255u) / 256u;
    uint first = l * chunk, last = min(first + chunk, lists);
    uint sum = 0u;
    for (uint e = first; e < last; e++) sum += ALOAD(gMpmCells[3u * e]);
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
    for (uint e = first; e < last; e++)
    {
        ASTORE(gMpmCells[3u * e + 1u], start);
        start += ALOAD(gMpmCells[3u * e]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmSelArea();
    HeightFieldRef hf = heightFieldOf(gColliders[p.dims.w]);
    uint e = mpmParticleEdgeList(gMpmParticles[i], p, hf);
    if (e == MPM_NO_SLOT) return;
    uint k;
    AADD3(gMpmCells[3u * e + 2u], 1u, k);
    gMpmSorted[ALOAD(gMpmCells[3u * e + 1u]) + k] = i;
}

// Per list: its range by slot (insertion sort; the order the sums take).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_sort(uint3 tid : SV_DispatchThreadID)
{
    uint e = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (e >= 2u * (p.window.z + p.window.w)) return;
    uint first = ALOAD(gMpmCells[3u * e + 1u]), n = ALOAD(gMpmCells[3u * e]);
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

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_shift(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmSelArea();
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
    gMpmCoupled[p.boundary.y + k] = float4(mpmEdgeGive(k, ij, axis, p, f), 0, 0, 0);
}

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

// ---- growth: the old walls' give taken out (TerrainPlan::open) -------------
//
// An open area's absorbing walls give: the material against a side moves
// out past its plane by the ground's displacement there, over the samples
// beyond the window. The bake measures positions in the unactivated
// ground's frame, so that give comes back into the window (mpmGroundShift).
// When the area grows, those samples beyond are activated in full, and the
// material that pushed into them would count twice: once in their new
// columns, once over them in the old area's particles (a ridge a sample wide
// along the old window's ring). So before the grown area activates, each old
// area's sides that stop being walls have their give taken out of the old
// area's particles, exactly as its bake would take it out: the old ground
// beyond is now material starting at rest where it was. Sides that stay walls
// keep their give (and their face displacements). Volume is untouched; only
// where the old material stands changes, by the walls' elastic give.

// gMpmParams[0].area.w bit (MPM_DOMAIN_KEEP + side, sides low x, high x, low
// z, high z): the selected old area's side stays a wall of the grown area.
static const uint MPM_DOMAIN_KEEP = 4u;

// Per edge sample of the selected old area: its give (0 on a side that
// stays a wall), into the cells after its edge lists.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_grow_shift(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    uint ni = p.window.z, nj = p.window.w;
    uint edges = 2u * (ni + nj);
    if (k >= edges) return;
    uint side = k < 2u * nj ? (k < nj ? 0u : 1u) : ((k - 2u * nj) < ni ? 2u : 3u);
    bool kept = ((gMpmParams[0].area.w >> (MPM_DOMAIN_KEEP + side)) & 1u) != 0u;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    uint axis;
    uint2 ij = mpmEdgeSample(k, p, axis);
    float give = kept ? 0.0 : mpmEdgeGive(k, ij, axis, p, f);
    ASTORE(gMpmCells[3u * edges + k], asuint(give));
}

// mpmGroundShift over mpm_grow_shift's gives.
float2 mpmGrowShift(float3 xl, HeightFieldRef hf, MpmParams p)
{
    int i0 = int(p.window.x), j0 = int(p.window.y);
    int ni = int(p.window.z), nj = int(p.window.w);
    float2 u = (xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz);
    int ti = clamp(int(round(u.x)) - i0, 0, ni - 1), tj = clamp(int(round(u.y)) - j0, 0, nj - 1);
    uint base = 3u * 2u * uint(ni + nj);
    float sx = ni > 1 ? clamp((u.x - float(i0)) / float(ni - 1), 0.0, 1.0) : 0.5;
    float sz = nj > 1 ? clamp((u.y - float(j0)) / float(nj - 1), 0.0, 1.0) : 0.5;
    float lx = asfloat(ALOAD(gMpmCells[base + uint(tj)])), hx = asfloat(ALOAD(gMpmCells[base + uint(nj + tj)]));
    float lz = asfloat(ALOAD(gMpmCells[base + uint(2 * nj + ti)]));
    float hz = asfloat(ALOAD(gMpmCells[base + uint(2 * nj + ni + ti)]));
    return float2(lerp(lx, hx, sx), lerp(lz, hz, sz));
}

// Per particle of the selected old area: its position less the give of the
// old walls that stop being walls.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_grow_apply(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmSelArea();
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q) || !mpmBakeOwns(q.xm.xyz)) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    float3 xl = frameToLocal(f, q.xm.xyz);
    xl.xz -= mpmGrowShift(xl, hf, p);
    gMpmParticles[i].xm = float4(frameToWorld(f, xl), q.xm.w);
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
        gMpmParticles[i].f2.w = asfloat(asuint(q.f2.w) | MPM_PARTICLE_DEAD);
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
    if ((gMpmParams[0].area.w & MPM_DOMAIN_CLOSE) != 0u) ASTORE(gMpmCounters[MPM_C_PARTICLES], 0u);
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
                    M += gMpmGrid[2u * (bk * MPM_BLOCK_NODES + l)].w;
                }
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
    if (k >= ALOAD(gMpmCounters[MPM_C_FLYING])) return;
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
            // The particle it left (its slot is dead while it flies).
            uint source = asuint(gMpmParcels[2u * asuint(p.couple.z) + k].x);
            if (mpmAreaOf(xq.xyz, p) != MPM_NO_AREA)
            {
                // Back as a particle, in the slot of the one it left: no
                // atomic, so particle ids (P2G's order) do not depend on the
                // parcels' slots.
                uint slot = source;
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
            }
            else landed = mpmParcelDeposit(xl, quanta, ~source, p, hf);
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

// Before a source on material that has moved: the top of the material under
// the source point, the highest of the selected area's particles within a
// lattice spacing of its vertical line (plus half a spacing, the particle's
// own extent), or the heightfield's surface there if higher (bare bed), as
// order-preserving bits in MPM_C_SURFACE (which the host zeroes first).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_source_surface(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i == 0u && p.dims.w != MPM_NO_BLOCK)
    {
        Collider col = gColliders[p.dims.w];
        Frame f = colliderFrame(col, gBodies[col.body]);
        HeightFieldRef hf = heightFieldOf(col);
        float3 xl = frameToLocal(f, p.source0.xyz);
        float h;
        float3 n;
        if (hfSurface(hf, xl.x, xl.z, h, n))
            AMAX2(gMpmCounters[MPM_C_SURFACE], mpmOrderedBits(frameToWorld(f, float3(xl.x, h, xl.z)).y));
    }
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float s = p.column.z;
    float2 d = q.xm.xz - p.source0.xz;
    if (abs(d.x) > s || abs(d.y) > s) return;
    if (mpmAreaOf(q.xm.xyz, p) != p.area.y) return;
    AMAX2(gMpmCounters[MPM_C_SURFACE], mpmOrderedBits(q.xm.y + 0.5 * s));
}

float mpmFromOrderedBits(uint b) { return asfloat((b & 0x80000000u) != 0u ? (b & 0x7FFFFFFFu) : ~b); }

// Per live particle of the selected area: the source's excavation velocity
// added to its own (the shock and contact stage replaced by its outcome; the
// MPM computes the rest). Several sources superpose: each adds its field to
// the material's velocity where it stands.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_source(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmParams();
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float3 s0 = p.source0.xyz;
    if ((p.area.w & MPM_DOMAIN_SURFACE) != 0u)
    {
        uint top = ALOAD(gMpmCounters[MPM_C_SURFACE]);
        if (top != 0u) s0.y = mpmFromOrderedBits(top);
    }
    float3 d = q.xm.xyz - s0;
    if (d.y > 0.0) return; // the field is the target's (y ≤ 0)
    if (length(d) > p.source1.w) return; // beyond the transient crater: at rest
    if (mpmAreaOf(q.xm.xyz, p) != p.area.y) return;
    q.vj.xyz += mpmZModelVelocity(d, p.source0.w, p.source1.x, p.source1.y, p.source1.z);
    gMpmParticles[i] = q;
}
