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
// Wheel loads (two-way wheel coupling): mpm_wheel_loads lists, once per
// coupled frame and in wheel order, each vehicle wheel the rigid step before
// stood on an open area's live surface (a static heightfield other than the
// bed: the area's surface patch) with the force it pressed there, the
// reaction of the ground's force on the chassis; mpm_wheel_mass sums, each
// substep after P2G, the material's mass M at the contact (B-spline weighted,
// over the nodes that carry velocity); mpm_grid then moves each node of the
// contact's stencil by F·dt·w/M (mpmWheelLoads), so the material takes F·dt
// per substep, exactly, and the frame's substeps the impulse the chassis took
// over the step. The sand ruts, berms and sinks under the load from its own
// strength; the wheel follows the surface down.
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
#include "../physics/vehicle_layout.slang"

// The vehicle pool (vehicle_step.slang): the wheels' contacts and forces from
// the rigid step before (mpm_wheel_loads reads it; nothing here writes it).
BIND(31) RWStructuredBuffer<float4> gVehicleData;

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
static const uint MPM_C_WHEELS = 14u;     // wheel loads mpm_wheel_loads listed this frame
static const uint MPM_C_AREA_SPEED = 16u; // + slot: an area's fastest particle this frame (f32 bits)
static const uint MPM_C_AREA_BODY = 32u;  // + slot: an area's fastest coupled body this frame
static const uint MPM_C_AREA_FIRST = 48u; // + slot: the pool's particle count when the area opened
static const uint MPM_C_SKIP = 64u;       // area clocks: bit k, area slot k sits out this pass
static const uint MPM_C_PASS = 65u;       // area clocks: the frame's next pass (mpm_pass)
static const uint MPM_C_SLEEP = 66u;      // block sleep: the rest speed (f32 bits; 0: off)
static const uint MPM_C_SLEEP_AFTER = 67u; // block sleep: substeps at rest before a particle sleeps
static const uint MPM_C_SLEEP_RING0 = 68u; // block sleep: blocks below this are the awake blocks
static const uint MPM_C_SLEEP_RING1 = 69u; // block sleep: below this, their neighbours (ring 1)
static const uint MPM_C_AREA_AIR = 70u;    // + slot: parcels in flight whose path can still reach the area
static const uint MPM_C_AREA_TOUCH = 86u;  // + slot: an area's fastest body touching its material this frame
static const uint MPM_C_POOL_TOP = 102u;   // pool reuse: one past the highest slot still held
static const uint MPM_C_HOLES = 103u;      // pool reuse: free slots found (then: the ones the activation takes)
static const uint MPM_C_REUSE = 104u;      // pool reuse: nonzero while a pass reuses slots (an activation's particles)
static const uint MPM_C_LIVE = 105u;       // live particles (the pool's held slots: these plus the parcels in flight)
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
static const uint MPM_OVERFLOW_WHEELS = 2048u;  // more wheels on open sand than max wheel loads
// Wheel-load records in gMpmCoupled from params.limits.z (uint bits) on, at
// most params.limits.w: (the contact's offset in its base cell, area slot
// bits), (its base node on the domain lattice, int bits), (force on the
// material N, its mass at the contact this substep).
static const uint MPM_WHEEL_RECORDS = 3u;
// MpmParams.boundary.x bits.
static const uint MPM_BOUNDARY_ABSORBING = 1u; // bed and walls absorb (mpmAbsorbNode)
static const uint MPM_BOUNDARY_GEOSTATIC = 2u; // the first step's mpm_geostatic sets F
// MpmParticle.f2.w bits.
static const uint MPM_PARTICLE_DEAD = 1u;       // left as a parcel: no longer simulated
static const uint MPM_PARTICLE_HOME = 2u;       // activated: c1.w, c2.w hold where (the bake's reference)
static const uint MPM_PARTICLE_FRESH = 4u;      // activated since the last step (mpm_geostatic's)
static const uint MPM_PARTICLE_ASLEEP = 8u;     // block sleep: at rest, frozen (TerrainConfig::sleep_speed)
static const uint MPM_PARTICLE_RESERVED = 16u;  // pool reuse: a dead slot a parcel in flight comes back to
static const uint MPM_PARTICLE_QUIET = 0xFF00u; // block sleep: substeps at rest in a row (bits 8-15)
static const uint MPM_PARTICLE_QUIET_ONE = 0x100u;
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
                    // y = the bake's home fade (lattice spacings; TerrainConfig::bake_home_fade),
                    // z = first wheel-load record in gMpmCoupled, w = max wheel loads (uint bits)
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

// A parcel in flight at x with velocity v: every open area its ballistic path
// can still reach (its box padded by a block, before the parcel falls below
// the box's floor) counts it (MPM_C_AREA_AIR, the local settle rule). It may
// land on the heightfield sooner: counting it then only keeps the area open.
void mpmParcelReach(float3 x, float3 v, MpmParams p)
{
    float g = p.step.y;
    for (uint a = 0u; a < p.area.x; a++)
    {
        MpmParams pa = gMpmParams[1u + a];
        if (pa.dims.x == 0u) continue;
        float pad = pa.origin.w * float(MPM_BLOCK);
        float3 lo = pa.origin.xyz - pad;
        float3 hi = pa.origin.xyz + float3(pa.dims.xyz) * pa.origin.w + pad;
        // When it falls below the floor: y + v.y·t + g·t²/2 = lo.y, the later root.
        float t = 0.0;
        if (x.y >= lo.y)
        {
            if (g < 0.0)
            {
                float c = x.y - lo.y;
                t = (v.y + sqrt(v.y * v.y - 2.0 * g * c)) / -g;
            }
            else t = 3.0e38;
        }
        // Its horizontal path over [0, t] against the box's footprint (slabs).
        float t0 = 0.0, t1 = t;
        bool miss = false;
        for (uint ax = 0u; ax < 2u; ax++)
        {
            float xa = ax == 0u ? x.x : x.z, va = ax == 0u ? v.x : v.z;
            float la = ax == 0u ? lo.x : lo.z, ha = ax == 0u ? hi.x : hi.z;
            if (abs(va) < 1e-20)
            {
                if (xa < la || xa > ha) miss = true;
                continue;
            }
            float ta = (la - xa) / va, tb = (ha - xa) / va;
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
        if (!miss && t0 <= t1) AADD2(gMpmCounters[MPM_C_AREA_AIR + a], 1u);
    }
}

// Area clocks (TerrainConfig::area_clocks): area slot k sits out this substep
// pass (mpm_pass sets which; none without area clocks).
bool mpmAreaSkips(uint k) { return ((ALOAD(gMpmCounters[MPM_C_SKIP]) >> k) & 1u) != 0u; }

// Block sleep (TerrainConfig::sleep_speed): the rest speed, 0 with sleep off.
float mpmSleepSpeed() { return asfloat(ALOAD(gMpmCounters[MPM_C_SLEEP])); }

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
        for (uint a = 0u; a < MPM_MAX_AREAS; a++) ASTORE(gMpmCounters[MPM_C_AREA_AIR + a], 0u);
        // Block sleep's rings: unset until mpm_sleep_ring.
        ASTORE(gMpmCounters[MPM_C_SLEEP_RING0], MPM_NO_BLOCK);
        ASTORE(gMpmCounters[MPM_C_SLEEP_RING1], MPM_NO_BLOCK);
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
    // Area clocks: an area sitting out this pass takes no part in it.
    if (mpmAreaSkips(area)) return;
    // Block sleep: an asleep particle joins after the awake blocks are known
    // (mpm_sleep_touch), unless a wake source reaches it now.
    if ((flags & MPM_PARTICLE_ASLEEP) != 0u && mpmTouchStaysAsleep(i, x, area, flags)) return;
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

#include "terrain_mpm_sleep.slang"
#include "terrain_mpm_boundary.slang"
#include "terrain_mpm_g2p.slang"
#include "terrain_mpm_growth.slang"
#include "terrain_mpm_ejecta.slang"
