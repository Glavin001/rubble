// FROZEN REFERENCE (test oracle only). A copy of explicit_beams.slang at
// feat/softbody-beamng-parity dcaa231b (ordered self-collision, before the
// feat/explicit-latency work). tests/explicit_beams_bitwise.rs runs it and the
// current kernel from the same GPU state each frame and requires the same state bit
// for bit. Never edit its physics; never ship it.
//
// Explicit beams: Rubble's fixed-step node–beam solver on the GPU
// (engine/docs/explicit-beams.md; CPU reference engine_softbody::explicit_beams,
// host engine_webgpu::ExplicitBeamsGpu). Standalone WGSL module.
//
// One 256-lane workgroup per vehicle; one dispatch per frame runs the origin rebase
// and all the frame's steps. Lanes stride over nodes, beams and triangles. Each step
// is three phases with a barrier after each:
//
//   nodes      the force consumed is the node's force slot (last step's restart)
//              plus its incidence entries' beam forces (contiguous), plus its
//              self-collision accumulator; then ground and box contact, integrate,
//              and restart the force slot (gravity, drag, turbulence)
//   beams      force, kind rules, yield, break test; the force on each end goes to
//              that end's incidence entry; a breaking group master
//              records its index in its group slot (atomic minimum)
//   collide    contacters against the vehicle's collision triangles (all pairs, or
//              with a grid cell size the contacters in the cells each triangle's
//              box covers: the same contacts, fewer tests), forces into
//              per-node integer atomics in fixed point (1/32 N); a node's current
//              force is its force slot plus its incidence entries plus its accumulator
//
// Detacher groups: the group slots are double-buffered by step parity. A beam of a
// group whose slot holds a master index counts as broken from the next node phase
// on (the node phase marks it), and a beam after the master drops the force it
// computed in the step its master broke; earlier ones keep theirs. The node phase
// also clears the slots the next beam phase writes. After the last step the
// frame marks the last step's group breaks, so a read-back sees them.
//
// Every barrier is at the top level of the step loop, whose bound comes from the
// uniform buffer: no barrier sits under a data-dependent branch (Chrome's WebGPU
// rejects that). No float atomics.
//
// Records (host: explicit_beams_gpu.rs):
//   gStatic  (read-only uint4) per vehicle 4 records: (nodeOff, nodeCount, beamOff,
//            beamCount), (triOff, triCount, boxOff, boxCount), (candOff, candCount,
//            groupOff, groupCount), (flags, ground bits, range bits, grid cell bits); at
//            nodeInfoBase per node (flags, incidence start, incidence count, 0); at
//            beamBase per beam 3 records: (a, b, kind | type << 8 | vehicle << 16, group as int),
//            (k, c, limit k, limit c), (short bound, long bound, plastic, L0) as
//            float bits; at triBase per triangle (n0, n1, n2, 0); at boxBase per box
//            (lo bits, 0), (hi bits, 0). Node and beam indices are global.
//   gLists   (read-only uint) incidence entries (global beam << 1 | is end b) and
//            self-collision candidates (global node indices); at slotBase per beam
//            the gLists positions of its end a's and end b's incidence entries.
//   gNodes   per node 3 × float4: (relative position, mass), (velocity, friction),
//            (force, 0).
//   gBeams   per beam 2 × float4: (L, Tc, Tt, G), (S, last force when S or broken
//            last changed, broken, 0);
//            at incBase per gLists position, the force of an incidence entry's beam
//            on that node (unused at candidate positions).
//   gAcc     per node 3 × int: self-collision force in 1/32 N.
//   gGroups  two parities × per vehicle group slot: index (local) of the first
//            breaking master, 0xFFFFFFFF when none.
//   gNext    per candidate (gLists position) its grid bucket | rank in it << 10.
//   Node info .w and triangle records .w hold the vehicle index (wide passes).
//   gMisc    per vehicle (origin bits xyz, steps run); then per triangle
//            (wait, idle) as int bits, hit this step (0 between steps).

static const float STEP = 0.0005;
static const float GRAVITY = -9.807;
static const float DRAG = 0.05;
static const float MIN_REST = 0.1;
static const float SUPPORT_BREAK_FACTOR = 4.0;
static const float REBASE_RADIUS = 100.0;
static const float TURBULENCE = 0.005;
static const float FIXED = 32.0;

static const uint NODE_NO_GROUND = 1u;
static const uint NODE_IMMOVABLE = 2u;
static const uint NODE_ON_TRIANGLE = 4u;

static const uint KIND_PLAIN = 0u;
static const uint KIND_LIMITED = 1u;
static const uint KIND_SUPPORT = 2u;
static const uint KIND_ROPE = 3u;
static const uint KIND_SHOCK2 = 4u; // EB_AERO modules only
static const uint KIND_TRIGGER = 5u; // EB_AERO modules only
static const uint TYPE_DEFORMABLE = 0u;

static const uint FLAG_FAST_INV_SQRT = 1u;
static const uint FLAG_DRAG = 2u;
static const uint FLAG_TURBULENCE = 4u;

static const uint NO_GROUP = 0xFFFFFFFFu;

struct Params
{
    uint steps;
    uint vehicles;
    uint nodeInfoBase;
    uint beamBase;
    uint triBase;
    uint boxBase;
    uint groupTotal; // group slots per parity (all vehicles)
    uint incBase; // gBeams record of incidence entry 0's force
    uint slotBase; // gLists entry of beam 0's two incidence positions
    uint gridBase; // gBeams record of the self-collision grid's sorted runs
    uint grid; // nonzero when any vehicle uses the grid (the scan phase runs)
    uint pad2; // the wide passes: the step within the frame (per-step parameter copies)
    // The wide passes (explicit_beams_wide_*): element totals and gAcc / gMisc /
    // gLists regions.
    uint totalNodes;
    uint totalBeams;
    uint totalTris;
    uint totalCands;
    uint wideMisc; // gMisc: (unused) then per vehicle its rebase shift
    uint gridCountBase; // gAcc: per vehicle GRID_BUCKETS counts
    uint gridOffBase; // gNext: per vehicle GRID_BUCKETS + 1 offsets
    uint testedBase; // gAcc: per vehicle 2 tested counts (by step parity)
    uint testedListBase; // gNext: per triangle (vehicle's list at its triOff)
    uint candTable; // gLists: per candidate its gLists position
    uint slotBase16; // gBeams: per vehicle and bucket WIDE_SLOTS (position, node)
    uint overflowBase; // gAcc: per vehicle the overflow count; gNext: overflow nodes at candOff
    // Aerodynamics and commands (the EB_AERO modules only; see the aero section).
    uint aeroBase; // gStatic: per vehicle 3 records
    uint wingBase; // gStatic: per wing 4 records
    uint cmdBase; // gStatic: per self-centering command 2 records
    uint aeroState; // gBeams: per vehicle D, then per wing 4 records, then per command 1
    uint wingTotal; // wings over every vehicle
    uint shock2Base; // gStatic: per shocks2 shock 2 records (EB_AERO modules)
    uint shock2Count; // shocks2 shocks over every vehicle
    // Triggers, key-driven commands and slide nodes (EB_AERO modules; see that
    // section).
    uint keyBase; // gNext: per vehicle 2 parities × KEYS trigger inputs
    uint slideBase; // gStatic: per slide node 2 records
    uint segBase; // gStatic: per rail segment 1 record
    uint slideState; // gBeams: per slide node (segment, broken away, 0, 0)
    uint slideAcc; // gBeams: per slide target node its slide force
    uint slideTotal; // slide nodes over every vehicle
    uint segTotal; // rail segments over every vehicle
    uint slotTotal; // slide target nodes over every vehicle
    // Vehicle–vehicle contact (see that section). vvOn nonzero: the frame entry
    // runs the vehicles of the solo list and checks their nodes stay in their
    // frame bounds.
    uint vvOn;
    uint vehExtBase; // gStatic: per vehicle its slide slots (then the slot records)
    uint vvStatic; // gStatic: per vehicle (contact point offset in gLists, count, 0, 0)
    uint vvTri; // gMisc: per triangle (cwait, cidle, had candidates, hit)
    uint vvVeh; // gMisc: per vehicle VV_RECS records
    uint vvAcc; // gAcc: per vehicle 2 parities × 6 box keys; then VV_CTL counters
    uint vvLabel; // gAcc: per vehicle its group label, then 3 per vehicle group sums
    uint vvGroups; // gMisc: per group (member offset, members, run offset, points)
    uint vvMembers; // gNext: members (vehicles words), sorted members, member point prefix
    uint vvPairs; // gNext: overlapping pairs (2 words each)
    uint vvPairCap;
    uint vvGrid; // gNext: vehicle grid bucket slots (VV_SLOTS per bucket)
    uint vvGridCount; // gAcc: per bucket its count
    uint vvGridMask; // buckets − 1
    uint vvCell; // float bits: the vehicle grid's cell size
    uint vvRun; // gBeams: per group its contact points sorted by cell
    uint slotCntBase; // gAcc: per slide slot its contribution count
    uint slotListBase; // gNext: the slide slots' contribution lists
    uint vvMargin; // float bits: frame bounds margin past the travel at the fastest node's speed
    uint vvBig; // gNext: vehicles too big for the vehicle grid
    uint vvFusedPairs; // nonzero: explicit_beams_vv_group finds the pairs itself (small fleets)
    uint vvPlanSteps; // steps the plan's frame bounds cover (one frame or more)
    uint vvNbr; // gLists: per incidence entry (same position + vvNbr) its far end
    uint vvHits; // gNext: per group 2 × its points hits (triangle, node) of one owner
    uint ablate; // measurement builds only (EB_ABLATE): phases to skip
};

// Slide nodes in parallel (EB_AERO frame, see slideNode): gStatic at vehExtBase per
// vehicle (first slide slot, slide slots, 0, 0), then per slide slot (list offset,
// capacity, 0, 0); gAcc at slotCntBase per slot this step's contributions; gNext at
// slotListBase the slots' contribution lists; gBeams after the slide-force slots
// per slide node 3 contributions (own node, rail end a, rail end b).
uint slotRecBase() { return gP.vehExtBase + gP.vehicles; }
// Then per wing node (EB_AERO frame, see wingGather) (node global, first entry in
// gLists, entries, 0); a vehicle's wing nodes are its vehExt record's .z, .w.
uint wingNodeBase() { return slotRecBase() + gP.slotTotal; }
uint slideContribBase() { return gP.slideAcc + gP.slotTotal; }

struct Surface
{
    float va;
    float ms;
    float mc;
    float t2;
    float vs;
    float alpha;
    float strength;
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> gP;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> gStatic;
[[vk::binding(2, 0)]] StructuredBuffer<uint> gLists;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gNodes;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> gBeams;
[[vk::binding(5, 0)]] RWStructuredBuffer<Atomic<int>> gAcc;
[[vk::binding(6, 0)]] RWStructuredBuffer<Atomic<uint>> gGroups;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> gMisc;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> gNext;
#ifdef EB_DET_HITS
// Tests only: the self-collision hit forces with the frozen baseline's arithmetic
// (hitForce), so this kernel's ordered application compares bit for bit with the
// baseline's one-lane sorted one (EB_DET_HITS there).
#endif

// Nonzero when a self-collision hit may have left forces in gAcc since the node
// phase last read and cleared it (set at frame start, since the last frame's
// last step may have hit): without one, every accumulator is zero.
groupshared uint gsHit;
// Workgroup-uniform flags of the self-collision apply (selfApply; the WGSL
// post-process turns their reads into workgroupUniformLoad): a hit this step, and
// the last round with open hits. Both cleared by lane 0 in the node phase.
groupshared uint gsUniformHits;
groupshared uint gsUniformOpen;

// Lanes per vehicle (one workgroup): 256, or 512 for large vehicles (a second
// module, ENGINE: EXPLICIT_BEAMS_WIDE_WGSL).
#ifndef EB_LANES
#define EB_LANES 256
#endif
static const uint LANES = EB_LANES;

// Self-collision grid (vehicles whose cell size, gStatic record 3 .w, is nonzero),
// a counting sort each step: the node phase clears the bucket counts; the beam
// phase (positions final for the step) counts each contacter into its bucket,
// keeping its rank (gNext: bucket | rank << 10); a scan phase turns the counts into
// offsets (gsOff, one past the end at GRID_BUCKETS); a scatter writes each
// contacter's position and node index to its bucket's run at gridBase. A
// triangle then reads its buckets' runs, contiguous independent loads.
static const uint GRID_BUCKETS = 1024u;
static const uint GRID_MAX_CELLS = 64u;
static const uint GRID_BLOCKS = 16u;
groupshared Atomic<uint> gsCount[GRID_BUCKETS];
groupshared uint gsOff[GRID_BUCKETS + 1u];
groupshared uint gsPart[LANES];
groupshared uint gsBlk[GRID_BLOCKS];

// Storage writes visible to the whole workgroup.
void sync()
{
#ifdef EB_ABLATE
    // Measurement only: 64 one barrier instead of two, 128 none.
    if ((gP.ablate & 128u) != 0u) return;
    if ((gP.ablate & 64u) != 0u)
    {
        GroupMemoryBarrierWithGroupSync();
        return;
    }
#endif
    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
}

Surface groundSurface()
{
    Surface s;
    s.va = 3.0; s.ms = 0.85; s.mc = 0.6; s.t2 = 0.006; s.vs = 3.0; s.alpha = 2.0; s.strength = 1.0;
    return s;
}

Surface boxSurface()
{
    Surface s;
    s.va = 3.0; s.ms = 1.2; s.mc = 0.75; s.t2 = 0.01; s.vs = 6.0; s.alpha = 2.0; s.strength = 1.0;
    return s;
}

// --- numerical primitives (compatible profile) ---

// Strict float arithmetic where a decision or a bit trick depends on rounding.
// Metal compiles with fast math (fused multiply-add, reassociation, approximate
// division and square root). Then a contacter lying in a triangle's plane can see
// the other sign of δ than the CPU reference and be pushed out the other side, and
// the bit-trick exponential (a·x + b to integer bits) can land on other bits. These
// helpers give IEEE single precision in the CPU's order: every intermediate passes
// through an XOR with a uniform the compiler cannot know is zero (no fusing or
// reassociation across it); division and square root take one exact fma
// correction.
uint noFuseBits() { return gP.vehicles >> 31; }
float nf(float x) { return asfloat(asuint(x) ^ noFuseBits()); }
float3 nf3(float3 v) { return float3(nf(v.x), nf(v.y), nf(v.z)); }
float smul(float a, float b) { return nf(a * b); }
float sdot(float3 a, float3 b) { return nf(nf(smul(a.x, b.x) + smul(a.y, b.y)) + smul(a.z, b.z)); }
float3 scross(float3 a, float3 b)
{
    return float3(nf(smul(a.y, b.z) - smul(b.y, a.z)), nf(smul(a.z, b.x) - smul(b.z, a.x)),
                  nf(smul(a.x, b.y) - smul(b.x, a.y)));
}
float3 sscale(float3 v, float k) { return float3(smul(v.x, k), smul(v.y, k), smul(v.z, k)); }
float sdiv(float a, float b)
{
    float r = 1.0 / b;
    float q = nf(a * r);
    return nf(fma(fma(-q, b, a), r, q));
}
float ssqrt(float x)
{
    if (x <= 0.0) return sqrt(x);
    float s = nf(sqrt(x));
    return nf(fma(fma(-s, s, x), 0.5 / s, s));
}

float3 sadd3(float3 a, float3 b) { return nf3(a + b); }
float3 ssub3(float3 a, float3 b) { return nf3(a - b); }
float3 sdiv3(float3 v, float k) { return float3(sdiv(v.x, k), sdiv(v.y, k), sdiv(v.z, k)); }

float sFastInvSqrt(float x)
{
    int i = 0x5f3759df - (asint(x) >> 1);
    float y = asfloat(i);
    return smul(y, nf(1.5 - smul(smul(smul(0.5, x), y), y)));
}

float approxSqrt(float x)
{
    int i = ((asint(x) - 0x3F800000) >> 1) + 0x3F800000;
    return asfloat(i);
}

float approxExp(float x)
{
    if (x < -15.0) return 0.0;
    if (x > 88.0) return 1e38;
    return asfloat(int(nf(smul(12102203.0, x) + 1064652319.0)));
}

float approxPow(float x, float y)
{
    float e = float(asint(x) - 1065353216);
    return asfloat(int(nf(smul(y, e) + 1065353216.0)));
}

// Turbulence: a well-mixed hash per vehicle, step, node and axis, as a value in
// [−1, 1) built like the CPU sequence's (a float in [2, 4) minus 3).
float noise(uint vehicle, uint step, uint node, uint axis)
{
    uint h = vehicle * 0x9E3779B9u ^ (step * 0x85EBCA6Bu) ^ (node * 3u + axis) * 0xC2B2AE35u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return asfloat(0x40000000u | (h & 0x7FFFFFu)) - 3.0;
}

float3 contactForce(float3 f, float3 v, float m, float3 n, Surface s, float p, float friction)
{
    if (p < 0.0) return float3(0.0, 0.0, 0.0);
    // Strict, in the CPU reference's order (its branches decide on these values).
    float vn = sdot(v, n);
    float fn = sdot(f, n);
    float r = -fn;
    if (vn < 0.0)
        r = nf(r - sdiv(smul(nf(smul(0.8, vn) - sdiv(smul(0.2, p), STEP)), m), STEP));
    if (r <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 sf = ssub3(f, sscale(n, fn));
    float3 sl = ssub3(v, sscale(n, vn));
    float sigma = ssqrt(sdot(sl, sl));
    if (sigma > 0.0) sl = sdiv3(sl, sigma);
    float gr = smul(smul(r, s.strength), friction);
    float ms = smul(s.ms, gr);
    if (sigma < s.va && gr > 0.0 && sdot(sf, sf) <= smul(ms, ms))
        return ssub3(ssub3(sscale(n, r), sscale(sl, smul(ms, nf(1.0 - approxExp(-sdiv(sigma, s.va)))))), sf);
    float g = nf(s.mc + smul(nf(s.ms - s.mc), approxExp(-approxPow(sdiv(sigma, s.vs), s.alpha))));
    return ssub3(sscale(n, r), sscale(sl, smul(nf(g + min(smul(s.t2, sigma), 5.0)), gr)));
}

float3 accForce(uint node)
{
    return float3(float(gAcc[node * 3u].load()), float(gAcc[node * 3u + 1u].load()),
                  float(gAcc[node * 3u + 2u].load())) * (1.0 / FIXED);
}

int toFixed(float x)
{
    return int(round(clamp(x * FIXED, -2.0e9, 2.0e9)));
}

void accAdd(uint node, float3 f)
{
    gAcc[node * 3u].add(toFixed(f.x));
    gAcc[node * 3u + 1u].add(toFixed(f.y));
    gAcc[node * 3u + 2u].add(toFixed(f.z));
}

uint groupSlot(uint parity, uint groupOff, uint g)
{
    return parity * gP.groupTotal + groupOff + g;
}

// Beam j's group break as a node phase sees it: true for a beam after the first
// master that broke its group (step parity `parity`), which then drops the force it
// computed; when `mark`, any beam of such a group is marked broken.
bool groupDrops(uint j, uint beamOff, uint groupOff, uint parity, bool mark)
{
    int g = asint(gStatic[gP.beamBase + j * 3u].w);
    if (g == 0) return false;
    uint master = gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load();
    if (master == NO_GROUP) return false;
    if (mark)
    {
        float4 s1 = gBeams[j * 2u + 1u];
        if (s1.z == 0.0)
        {
            s1.z = 1.0;
            gBeams[j * 2u + 1u] = s1;
        }
    }
    return j - beamOff > master;
}

// A node's force slot `f` plus its beam forces, in beam order as the CPU adds them.
// The beam phase writes each beam's force on each end into that end's incidence
// entry, so the entries of one node are contiguous and independent loads (no load
// waits on another's index).
float3 beamSum(float3 f, uint4 info, uint beamOff, uint groupOff, uint groupCount, uint parity, bool mark)
{
    uint base = gP.incBase + info.y;
    // Slot 0 of the vehicle's groups (no group has index 0) records that some
    // master broke in that step: without one no beam drops its force.
    if (groupCount == 0u || gGroups[groupSlot(parity, groupOff, 0u)].load() == NO_GROUP)
    {
        uint k = 0u;
        for (; k + 4u <= info.z; k += 4u)
        {
            float3 a = gBeams[base + k].xyz;
            float3 b = gBeams[base + k + 1u].xyz;
            float3 c = gBeams[base + k + 2u].xyz;
            float3 d = gBeams[base + k + 3u].xyz;
            f = sadd3(f, a);
            f = sadd3(f, b);
            f = sadd3(f, c);
            f = sadd3(f, d);
        }
        for (; k < info.z; k++) f = sadd3(f, gBeams[base + k].xyz);
        return f;
    }
    for (uint k = 0u; k < info.z; k++)
    {
        float3 push = gBeams[base + k].xyz;
        if (!groupDrops(gLists[info.y + k] >> 1, beamOff, groupOff, parity, mark)) f = sadd3(f, push);
    }
    return f;
}

#include "explicit_beams_ref_passes.slang"
#include "explicit_beams_ref_aero.slang"
#include "explicit_beams_ref_hits.slang"
#include "explicit_beams_ref_wide.slang"
#include "explicit_beams_ref_groups.slang"
#include "explicit_beams_ref_apply.slang"
