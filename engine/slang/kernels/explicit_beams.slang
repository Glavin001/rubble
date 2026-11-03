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
// Tests only: per vehicle a hit count and up to DET_CAP (triangle, node) hits of the
// step's self-collision; after every vehicle's region, an overflow flag.
[[vk::binding(12, 0)]] RWStructuredBuffer<Atomic<uint>> gDet;
static const uint DET_CAP = 8192u;
#endif

// Nonzero when a self-collision hit may have left forces in gAcc since the node
// phase last read and cleared it (set at frame start, since the last frame's
// last step may have hit): without one, every accumulator is zero.
groupshared uint gsHit;

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

// --- passes ---

#ifdef EB_VV_SOLO
// Vehicle contact (the solo frame): this lane's nodes left the frame bounds
// (relative, in workgroup memory) at some step.
// A node of vehicle v outside its frame bounds (world positions): reported at
// once (rare), so nothing stays live across the step loop.
groupshared float3 gsVvBounds[2];
#endif

#ifdef EB_AERO
void nodeUpdateFuse(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
static const uint FLAG_FUSEDRAG = 8u;
// Vehicle flags: it has triggers (key-driven commands), slide nodes.
static const uint FLAG_KEYS = 16u;
static const uint FLAG_SLIDES = 32u;
// Node info flags above this shift: the node's slide-force slot + 1 (0: none).
static const uint NODE_SLOT_SHIFT = 8u;
float3 slideTake(float3 f, uint slot1);
#endif

void nodeUpdate(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        nodeUpdateFuse(gi, info, vehicle, origin, ground, boxOff, boxCount, beamOff, groupOff, groupCount,
                       (step + 1u) & 1u, hits);
        return;
    }
#endif
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
#ifdef EB_AERO
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
#endif
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        if (ground > x.y)
            f += contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
        for (uint b = 0u; b < boxCount; b++)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u + 1u].xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float3 n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
                f += contactForce(f, vel, mass, n, boxSurface(), 0.0, V.w);
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    f = float3(0.0, mass * GRAVITY, 0.0);
    if ((flagsV & FLAG_DRAG) != 0u)
    {
        float s = approxSqrt(dot(vel, vel));
        f += -DRAG * s * vel;
        if ((flagsV & FLAG_TURBULENCE) != 0u)
        {
            float t = DRAG * s * s * TURBULENCE;
            f += t * float3(noise(vehicle, step, local, 0u), noise(vehicle, step, local, 1u),
                            noise(vehicle, step, local, 2u));
        }
    }
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
}
#ifdef EB_VV_SOLO
#ifdef EB_AERO
uint nodeUpdateFuseSolo(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                        uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
#endif
// nodeUpdate for the solo frame: also whether the node left the frame bounds.
uint nodeUpdateSolo(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        return nodeUpdateFuseSolo(gi, info, vehicle, origin, ground, boxOff, boxCount, beamOff, groupOff, groupCount,
                       (step + 1u) & 1u, hits);
    }
#endif
    float4 P = gNodes[gi * 3u];
    // The position the last step left (each step's checked once: the frame's
    // last at its end).
    if (any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
#ifdef EB_AERO
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
#endif
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        if (ground > x.y)
            f += contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
        for (uint b = 0u; b < boxCount; b++)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u + 1u].xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float3 n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
                f += contactForce(f, vel, mass, n, boxSurface(), 0.0, V.w);
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    f = float3(0.0, mass * GRAVITY, 0.0);
    if ((flagsV & FLAG_DRAG) != 0u)
    {
        float s = approxSqrt(dot(vel, vel));
        f += -DRAG * s * vel;
        if ((flagsV & FLAG_TURBULENCE) != 0u)
        {
            float t = DRAG * s * s * TURBULENCE;
            f += t * float3(noise(vehicle, step, local, 0u), noise(vehicle, step, local, 1u),
                            noise(vehicle, step, local, 2u));
        }
    }
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
    return 0u;
}
#endif

// Live plain beams at a node, from the broken flags as they are now.
uint livePlain(uint node)
{
    uint4 info = gStatic[gP.nodeInfoBase + node];
    uint live = 0u;
    for (uint k = 0u; k < info.z; k++)
    {
        uint j = gLists[info.y + k] >> 1;
        uint kind = gStatic[gP.beamBase + j * 3u].z & 0xFFu;
        if (kind == KIND_PLAIN && gBeams[j * 2u + 1u].z == 0.0) live++;
    }
    return live;
}

#ifdef EB_AERO
void shock2Update(uint gj, uint local, bool fast, uint groupOff, uint parity, uint4 bn, float4 s1);
void triggerUpdate(uint gj, bool fast, uint parity, uint4 bn);
#endif

void beamUpdate(uint gj, uint local, bool fast, uint groupOff, uint parity)
{
    float4 s1 = gBeams[gj * 2u + 1u];
    if (s1.z != 0.0)
    {
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
#ifdef EB_AERO
    if ((bn.z & 0xFFu) == KIND_SHOCK2)
    {
        shock2Update(gj, local, fast, groupOff, parity, bn, s1);
        return;
    }
    if ((bn.z & 0xFFu) == KIND_TRIGGER)
    {
        triggerUpdate(gj, fast, parity, bn);
        return;
    }
#endif
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    float4 c2 = asfloat(gStatic[gP.beamBase + gj * 3u + 2u]);
    float4 s0 = gBeams[gj * 2u];
    uint kind = bn.z & 0xFFu;
    uint type = (bn.z >> 8) & 0xFFu;
    int group = asint(bn.w);
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    float rest = s0.x;
    float tc = s0.y;
    float tt = s0.z;
    float gate = s0.w;
    float strength = s1.x;
    // Strict, in the CPU reference's order (yield and break decide on these).
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float inv = fast ? sFastInvSqrt(q) : sdiv(1.0, ssqrt(q));
    float len = smul(q, inv);
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    float k = c1.x;
    float c = c1.y;
    bool broken = false;
    bool supportBroke = false;
    if (kind == KIND_LIMITED)
    {
        float lb = smul(c2.y, rest);
        float sb = smul(c2.x, rest);
        float o = 0.0;
        if (e > lb) o = nf(e - lb);
        else if (e < -sb) o = nf(-e - sb);
        if (o != 0.0)
        {
            k = nf(k + smul(nf(c1.z - k), o));
            c = nf(c + smul(nf(c1.w - c), o));
        }
    }
    else if (kind == KIND_SUPPORT)
    {
        if (e > 0.0)
        {
            k = 0.0;
            c = smul(c, 0.1);
            float factor = c2.y > 0.0 ? c2.y : SUPPORT_BREAK_FACTOR;
            if (e > smul(rest, factor))
            {
                broken = true;
                supportBroke = true;
            }
        }
    }
    else if (kind == KIND_ROPE)
    {
        if (e < 0.0)
        {
            k = 0.0;
            c = smul(c, 0.1);
        }
    }
    float f = nf(smul(-k, e) - smul(c, u));
    float last = f;
    float tau = abs(f);
    if (!supportBroke && tau > gate)
    {
        if (type == TYPE_DEFORMABLE && kind != KIND_LIMITED && k != 0.0)
        {
            if (f > tc && e < 0.0)
            {
                float delta = nf(e + smul(sdiv(tc, k), nf(1.0 - c2.z)));
                float old = rest;
                rest = max(nf(rest + delta), MIN_REST);
                f = nf(f - sdiv(nf(f - tc), 2.0));
                tau = f;
                if (rest < old)
                {
                    tc = sdiv(smul(tc, old), rest);
                    gate = min(min(tc, tt), strength);
                }
            }
            else if (f < -tt && e > 0.0)
            {
                float delta = nf(e - smul(sdiv(tt, k), nf(1.0 - c2.z)));
                float old = rest;
                rest = nf(rest + delta);
                f = nf(f - sdiv(nf(f + tt), 2.0));
                tau = -f;
                if (rest > old)
                {
                    tt = sdiv(smul(tt, rest), old);
                    gate = min(min(tc, tt), strength);
                }
                strength = nf(strength - smul(delta, k));
            }
        }
        if (tau > strength)
        {
            bool saved = false;
            if ((gStatic[gP.nodeInfoBase + bn.x].x & NODE_ON_TRIANGLE) != 0u && livePlain(bn.x) < 3u) saved = true;
            if ((gStatic[gP.nodeInfoBase + bn.y].x & NODE_ON_TRIANGLE) != 0u && livePlain(bn.y) < 3u) saved = true;
            if (saved)
            {
                strength = 2.0 * gate;
            }
            else
            {
                broken = true;
                f = 0.0;
                if (group > 0)
                {
                    gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
                    gGroups[groupSlot(parity, groupOff, 0u)].min(local);
                }
            }
        }
    }
    // State records only when they change (yield, break): most steps change
    // neither. The last force (.y) is kept for a read-back only, so it is stored
    // with the record, not every step.
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken) gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// A triangle's throttle after a step: a waiting triangle counts down (and its idle
// steps up); one that was tested restarts its wait from its hit (gMisc .w, set by
// any of its lanes) or counts it up. A triangle that comes off idle waits at most
// 12 steps and is tested that same step (triangleTest reads it as not waiting).
int triangleFinish(uint gt)
{
    uint mi = gP.vehicles + gt;
    uint4 th = gMisc[mi];
    int wait = asint(th.x);
    int idle = asint(th.y);
    if (wait > 0)
    {
        wait = wait - 1;
        idle = idle + 1;
    }
    else
    {
        if (idle > 0)
        {
            wait = min(idle, 12);
            idle = 0;
        }
        wait = th.w != 0u ? -20000 : wait + 1;
    }
    gMisc[mi] = uint4(asuint(wait), asuint(idle), 0u, 0u);
    return wait;
}

// A triangle as its contact tests see it.
struct TriGeo
{
    uint mi;
    uint4 tn;
    float3 p0;
    float3 n;
    float3 ra;
    float3 rb;
    float3 rn;
    bool solvable;
    float3 lo;
    float3 hi;
};

// Contacter node h against triangle t: the contact force into the accumulators
// (h, and the triangle's corners by their weights) when h is within r of the face.
void tryContact(TriGeo t, uint h, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount,
                uint parity)
{
    uint4 tn = t.tn;
    if (h == tn.x || h == tn.y || h == tn.z) return;
    float4 P = gNodes[h * 3u];
    float3 p = nf3(origin + P.xyz);
    if (any(p < t.lo) || any(p > t.hi)) return;
    float3 q = nf3(p - t.p0);
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return;
    gMisc[t.mi].w = 1u;
    gsHit = 1u;
#ifdef EB_DET_HITS
    {
        uint base = gStatic[gP.nodeInfoBase + h].w * (1u + 2u * DET_CAP);
        uint i = gDet[base].add(1u);
        if (i < DET_CAP)
        {
            gDet[base + 1u + 2u * i].store(t.mi - gP.vehicles);
            gDet[base + 2u + 2u * i].store(h);
        }
        return;
    }
#endif
    float3 nn = t.n;
    if (delta < 0.0)
    {
        nn = -t.n;
        delta = -delta;
    }
    float4 V = gNodes[h * 3u + 1u];
    float4 P0 = gNodes[tn.x * 3u];
    float4 P1 = gNodes[tn.y * 3u];
    float4 P2 = gNodes[tn.z * 3u];
    float3 v = V.xyz - (alpha * gNodes[tn.y * 3u + 1u].xyz + beta * gNodes[tn.z * 3u + 1u].xyz
                        + gamma * gNodes[tn.x * 3u + 1u].xyz);
    float mt = alpha * P1.w + beta * P2.w + gamma * P0.w;
    float m = P.w * mt / (P.w + mt);
    float3 fh = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity, false)
        + accForce(h);
    float3 f = contactForce(fh, v, m, nn, boxSurface(), r - delta, V.w);
    accAdd(h, f);
    accAdd(tn.y, -alpha * f);
    accAdd(tn.z, -beta * f);
    accAdd(tn.x, -gamma * f);
}

// Triangle gt's geometry; mi < vehicles (none) when it waits this step.
TriGeo triangleGeo(uint gt, float3 origin, float r)
{
    TriGeo t;
    t.mi = gP.vehicles + gt;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    float3 u = nf3(p1 - t.p0);
    float3 w = nf3(p2 - t.p0);
    float3 cr = scross(u, w);
    float cl = ssqrt(sdot(cr, cr));
    t.n = cl > 0.0 ? float3(sdiv(cr.x, cl), sdiv(cr.y, cl), sdiv(cr.z, cl)) : cr;
    float det = sdot(u, scross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? sdiv(1.0, det) : 0.0;
    t.ra = sscale(scross(w, t.n), rd);
    t.rb = sscale(scross(t.n, u), rd);
    t.rn = sscale(scross(u, w), rd);
    t.lo = min(min(t.p0, p1), p2) - float3(r, r, r);
    t.hi = max(max(t.p0, p1), p2) + float3(r, r, r);
    return t;
}

// Lane `sub` of the `lanes` lanes of triangle gt tests candidates sub, sub + lanes, …
// Forces go to fixed-point atomics, so the lanes' order does not change the sums.
void triangleTest(uint gt, uint sub, uint lanes, float3 origin, float r, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r);
    for (uint k = sub; k < candCount; k += lanes)
        tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
}

// The grid's bucket of a cell.
uint gridBucket(int3 c)
{
    return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u) & (GRID_BUCKETS - 1u);
}

// Triangle gt against the contacters in the grid cells its box (padded by r)
// covers: each covered bucket's run once (two cells can share a bucket). A box
// over more than GRID_MAX_CELLS cells tests every candidate instead. The runs
// hold every contacter, so the contacts found are the all-pairs test's. GRID_SUBS
// lanes share a triangle: lane `sub` takes the cells sub, sub + GRID_SUBS, … (and
// the same share of the candidates when it falls back), so a triangle's walk is
// spread rather than serial on one lane.
static const uint GRID_SUBS = 8u;

// The triangles tested this step (wait <= 0 after the throttle update), listed by
// the node phase for a vehicle of up to TRI_LIST triangles; the triangle phase then
// walks only them. Counters by step parity: the beam phase clears the next step's.
static const uint TRI_LIST = 2048u;
groupshared uint gsTested[TRI_LIST];
groupshared Atomic<uint> gsTestedCount[2];

void triangleGrid(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub; i < cells; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        // Skip a bucket an earlier cell of this triangle already covers.
        bool dup = false;
        for (uint j = 0u; j < i; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        for (uint e = gsOff[b]; e < gsOff[b + 1u]; e++)
        {
            float4 E = gBeams[gP.gridBase + candOff + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
}

// A step with few (triangle, contacter) pairs skips the grid's counting sort (its
// scan costs five barriers) and walks every candidate instead, keeping exactly the
// ones the grid walk would test: inside the triangle's box and in a bucket its
// cells cover (or every candidate when the box covers too many cells), with the
// grid's own expressions. So the hits are the grid's, bit for bit.
static const uint BRUTE_PAIRS_PER_LANE = 16u;
void triangleGridBrute(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                       uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r);
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub; k < candCount; k += subs)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint k = sub; k < candCount; k += subs)
    {
        uint h = gLists[candOff + k];
        float3 e = gNodes[h * 3u].xyz;
        float3 p = origin + e;
        if (any(p < t.lo) || any(p > t.hi)) continue;
        int3 c = int3(floor(e / cell));
        if (!(all(c >= c0) && all(c < c0 + span)))
        {
            // Outside the covered cells: tested only when its bucket is a covered
            // cell's (a hash collision the grid walk would visit).
            uint b = gridBucket(c);
            bool covered = false;
            for (uint i = 0u; i < cells; i++)
                covered = covered
                    || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
            if (!covered) continue;
        }
        tryContact(t, h, origin, r, beamOff, groupOff, groupCount, parity);
    }
}

// A step without the grid finds its self-collision candidates early: the beam
// phase (positions are final after the node phase) lists the (triangle, candidate)
// pairs that pass the tests before tryContact's hit test (the box, and for a grid
// vehicle the grid's covered buckets, with the same expressions), on the lanes the
// beams leave idle; the triangle phase then runs tryContact on the listed pairs
// only (usually none). A fuller list than PAIR_LIST runs the step's usual walk.
static const uint PAIR_LIST = 1024u;
groupshared uint gsPairs[PAIR_LIST];
groupshared Atomic<uint> gsPairCount;

void trianglePairs(uint t, uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                   uint candCount)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    float3 lo = min(min(p0, p1), p2) - float3(r, r, r);
    float3 hi = max(max(p0, p1), p2) + float3(r, r, r);
    bool every = true;
    int3 c0 = int3(0, 0, 0);
    int3 span = int3(0, 0, 0);
    if (cell > 0.0)
    {
        float3 f0 = floor((lo - origin) / cell);
        float3 fs = floor((hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
        every = !(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS));
        if (!every)
        {
            c0 = int3(f0);
            span = int3(fs);
        }
    }
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint k = sub; k < candCount; k += subs)
    {
        uint h = gLists[candOff + k];
        if (h == tn.x || h == tn.y || h == tn.z) continue;
        float3 e = gNodes[h * 3u].xyz;
        if (!every)
        {
            // The grid walk's tests (triangleGridBrute).
            float3 p = origin + e;
            if (any(p < lo) || any(p > hi)) continue;
            int3 c = int3(floor(e / cell));
            if (!(all(c >= c0) && all(c < c0 + span)))
            {
                uint b = gridBucket(c);
                bool covered = false;
                for (uint i = 0u; i < cells; i++)
                    covered = covered
                        || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
                if (!covered) continue;
            }
        }
        // tryContact's box test.
        float3 x = nf3(origin + e);
        if (any(x < lo) || any(x > hi)) continue;
        uint i = gsPairCount.add(1u);
        if (i < PAIR_LIST) gsPairs[i] = (t << 16) | k;
    }
}

#ifdef EB_DET_HITS
// Tests only (EB_DET_HITS): the step's self-collision hits (found by the usual
// lanes) applied on one lane, sorted by (triangle, node), so each hit sees every
// earlier one in a fixed order: tryContact's force part.
void detApply(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint base = v * (1u + 2u * DET_CAP);
    uint n = gDet[base].load();
    if (n > DET_CAP)
    {
        gDet[gP.vehicles * (1u + 2u * DET_CAP)].store(1u);
        n = DET_CAP;
    }
    for (uint i = 1u; i < n; i++)
    {
        uint gt = gDet[base + 1u + 2u * i].load();
        uint h = gDet[base + 2u + 2u * i].load();
        uint j = i;
        while (j > 0u)
        {
            uint pg = gDet[base + 1u + 2u * (j - 1u)].load();
            uint ph = gDet[base + 2u + 2u * (j - 1u)].load();
            if (pg < gt || (pg == gt && ph <= h)) break;
            gDet[base + 1u + 2u * j].store(pg);
            gDet[base + 2u + 2u * j].store(ph);
            j--;
        }
        gDet[base + 1u + 2u * j].store(gt);
        gDet[base + 2u + 2u * j].store(h);
    }
    for (uint i = 0u; i < n; i++)
    {
        uint gt = gDet[base + 1u + 2u * i].load();
        uint h = gDet[base + 2u + 2u * i].load();
        TriGeo t = triangleGeo(gt, origin, r);
        uint4 tn = t.tn;
        float4 P = gNodes[h * 3u];
        float3 p = nf3(origin + P.xyz);
        float3 q = nf3(p - t.p0);
        float alpha = 0.0;
        float beta = 0.0;
        float delta = 0.0;
        if (t.solvable)
        {
            alpha = sdot(t.ra, q);
            beta = sdot(t.rb, q);
            delta = sdot(t.rn, q);
        }
        float gamma = nf(nf(1.0 - alpha) - beta);
        float3 nn = t.n;
        if (delta < 0.0)
        {
            nn = -t.n;
            delta = -delta;
        }
        float4 V = gNodes[h * 3u + 1u];
        float4 P0 = gNodes[tn.x * 3u];
        float4 P1 = gNodes[tn.y * 3u];
        float4 P2 = gNodes[tn.z * 3u];
        float3 vv = V.xyz - (alpha * gNodes[tn.y * 3u + 1u].xyz + beta * gNodes[tn.z * 3u + 1u].xyz
                             + gamma * gNodes[tn.x * 3u + 1u].xyz);
        float mt = alpha * P1.w + beta * P2.w + gamma * P0.w;
        float m = P.w * mt / (P.w + mt);
        float3 fh = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity, false)
            + accForce(h);
        float3 f = contactForce(fh, vv, m, nn, boxSurface(), r - delta, V.w);
        accAdd(h, f);
        accAdd(tn.y, -alpha * f);
        accAdd(tn.z, -beta * f);
        accAdd(tn.x, -gamma * f);
    }
    gDet[base].store(0u);
}
#endif

// --- Aerodynamics and commands with no input (EB_AERO modules) ---
// engine/docs/explicit-beams-aero-commands-spec.md; CPU reference
// engine_softbody::explicit_aero. The specification's double-precision
// sub-expressions run in strict single precision here (WGSL has no f64), so the GPU
// may differ from the CPU by an ulp in them; the angle of attack's acos is the
// device's.
//
//   gStatic at aeroBase per vehicle: (wingOff, wingCount, cmdOff, cmdCount) (wing and
//            command indices global), (front node global, width bits, node count,
//            0), (first slide node global, slide node count, 0, 0); at wingBase per wing: (FLD, FRD, FLU, FRU), (BLD, BRD, BLU, BRU)
//            (global nodes), (efficacy, break reference, table record, induced side
//            0 none 1 left 2 right), (chain span, chain area, 0, 0) as bits; at
//            cmdBase per command: (beam global, contract first, centre bits, 0 for a
//            self-centering command with no input, else contract key | extend key
//            << 8 | self-centering << 16 | 1 << 17 for a vehicle with triggers),
//            (shorten rate, lengthen rate, max contraction, max extension) bits;
//            a table record holds 3601 (cl, cd, cm, 0) as bits.
//   gBeams   at aeroState per vehicle (D, 0); then per wing (broken, 0, 0, 0),
//            induced drag, front force, back force; then per command (mode, 0, 0, 0).
#ifdef EB_AERO

uint wingState(uint gw) { return gP.aeroState + gP.vehicles + gw * 4u; }
uint cmdState(uint gc) { return gP.aeroState + gP.vehicles + gP.wingTotal * 4u + gc; }

float slen(float3 v) { return ssqrt(sdot(v, v)); }
float3 snormalise(float3 v)
{
    float l = slen(v);
    if (l > 0.0) return sscale(v, sdiv(1.0, l));
    return v;
}

// pow≈-based air density at height y; `t0` 288.15 (wings) or 288.1 (fuselage).
float airDensity(float y, float t0)
{
    float t = nf(1.0 - sdiv(smul(0.0065, y), t0));
    float p = smul(101325.0, approxPow(t, 5.24947));
    return smul(p, 0.0000120896);
}

// The signed angle of attack (degrees), as the CPU reference evaluates it.
float angleOfAttack(float3 c, float3 n, float3 b, float3 w)
{
    float3 q = snormalise(scross(n, c));
    float3 g = -w;
    float3 p;
    for (uint i = 0u; i < 3u; i++)
    {
        float3 m = float3(-smul(q[i], q.x), -smul(q[i], q.y), -smul(q[i], q.z));
        m[i] = nf(1.0 - smul(q[i], q[i]));
        p[i] = nf(nf(smul(m.x, g.x) + smul(m.y, g.y)) + smul(m.z, g.z));
    }
    float3 u = c;
    float3 t = -p;
    float a = ssqrt(smul(sdot(u, u), sdot(t, t)));
    float beta = nf(a + sdot(t, u));
    if (abs(nf(beta - smul(2.0, a))) <= 1.1920929e-7 || a == 0.0) return 0.0;
    float qw = beta;
    float3 ax = scross(u, t);
    if (beta < smul(1e-6, a))
    {
        qw = 0.0;
        ax = abs(u.x) > abs(u.z) ? float3(-u.y, u.x, 0.0) : float3(0.0, -u.z, u.y);
    }
    float nq = ssqrt(nf(nf(nf(smul(qw, qw) + smul(ax.x, ax.x)) + smul(ax.y, ax.y)) + smul(ax.z, ax.z)));
    float inv = sdiv(1.0, nq);
    qw = smul(qw, inv);
    ax = sscale(ax, inv);
    float s2 = sdot(ax, ax);
    float theta = 0.0;
    float3 axis = float3(1.0, 0.0, 0.0);
    if (s2 > 0.0)
    {
        float ac = qw >= 1.0 ? 0.0 : (qw <= -1.0 ? 3.14159274 : acos(qw));
        theta = smul(2.0, ac);
        axis = sscale(ax, sdiv(1.0, ssqrt(s2)));
    }
    float alpha = smul(theta, sdiv(180.0, 3.14159274));
    return sdot(axis, b) > 0.0 ? -alpha : alpha;
}

// (cl, cd, cm) of table record `table` at angle a (degrees), no deflection.
float3 airfoilLookup(uint table, float a)
{
    float k = trunc(sdiv(a, 360.0));
    float w = nf(a - smul(k, 360.0));
    if (w > 180.0) w = nf(w - 360.0);
    if (w < -180.0) w = nf(w + 360.0);
    float fi = trunc(smul(nf(w + 180.0), 10.0));
    uint i = uint(clamp(fi, 0.0, 3600.0));
    return asfloat(gStatic[table + i].xyz);
}

// Wing gw's forces this step into its state records (zero when broken).
void wingForces(uint gw, float3 origin)
{
    uint ws = wingState(gw);
    if (gBeams[ws].x != 0.0) return;
    uint4 A = gStatic[gP.wingBase + gw * 4u];
    uint4 B = gStatic[gP.wingBase + gw * 4u + 1u];
    uint4 C = gStatic[gP.wingBase + gw * 4u + 2u];
    float4 D = asfloat(gStatic[gP.wingBase + gw * 4u + 3u]);
    float3 xfld = gNodes[A.x * 3u].xyz;
    float3 xfrd = gNodes[A.y * 3u].xyz;
    float3 xbld = gNodes[B.x * 3u].xyz;
    float3 xbrd = gNodes[B.y * 3u].xyz;
    float3 w = -sscale(sadd3(gNodes[A.x * 3u + 1u].xyz, gNodes[A.y * 3u + 1u].xyz), 0.5);
    float speed = slen(w);
    float3 c = sscale(sadd3(ssub3(xbld, xfld), ssub3(xbrd, xfrd)), 0.5);
    float chord = slen(c);
    float3 b = sscale(sadd3(ssub3(xfrd, xfld), ssub3(xbrd, xbld)), 0.5);
    float span = slen(b);
    float3 l = scross(b, -w);
    float s = smul(span, chord);
    float3 n = snormalise(scross(c, b));
    float3 co = airfoilLookup(C.z, angleOfAttack(c, n, b, w));
    float rho = airDensity(nf(origin.y + xfld.y), 288.15);
    float kd = smul(smul(smul(smul(co.y, 0.5), rho), speed), s);
    float3 wf = sscale(w, kd);
    float3 ind = float3(0.0, 0.0, 0.0);
    if (C.w != 0u)
    {
        float num = smul(smul(smul(smul(smul(smul(co.y, co.y), 0.25), rho), speed), D.y), D.y);
        float den = smul(smul(3.14159, D.x), D.x);
        ind = sscale(w, sdiv(num, den));
    }
    float kl = smul(smul(smul(smul(co.x, 0.5), rho), speed), chord);
    wf = sadd3(wf, sscale(l, kl));
    float mu = smul(smul(smul(smul(smul(-co.z, 0.5), rho), speed), speed), s);
    float e = asfloat(C.x);
    float em = smul(e, mu);
    gBeams[ws + 1u] = float4(ind, 0.0);
    gBeams[ws + 2u] = float4(sadd3(sscale(wf, smul(e, 0.1875)), sscale(n, em)), 0.0);
    gBeams[ws + 3u] = float4(ssub3(sscale(wf, smul(e, 0.0625)), sscale(n, sdiv(em, 3.0))), 0.0);
}

void addForce(uint node, float3 f)
{
    float4 F = gNodes[node * 3u + 2u];
    gNodes[node * 3u + 2u] = float4(sadd3(F.xyz, f), F.w);
}

// One lane adds a vehicle's wing forces in file order, slot by slot, as the CPU.
void wingApply(uint wingOff, uint wingCount)
{
    for (uint k = 0u; k < wingCount; k++)
    {
        uint gw = wingOff + k;
        uint ws = wingState(gw);
        if (gBeams[ws].x != 0.0) continue;
        uint4 A = gStatic[gP.wingBase + gw * 4u];
        uint4 B = gStatic[gP.wingBase + gw * 4u + 1u];
        uint side = gStatic[gP.wingBase + gw * 4u + 2u].w;
        float3 ind = gBeams[ws + 1u].xyz;
        float3 f1 = gBeams[ws + 2u].xyz;
        float3 f2 = gBeams[ws + 3u].xyz;
        if (side == 1u) { addForce(B.z, ind); addForce(B.x, ind); }
        if (side == 2u) { addForce(B.w, ind); addForce(B.y, ind); }
        addForce(A.x, f1); addForce(A.z, f1); addForce(A.y, f1); addForce(A.w, f1);
        addForce(B.x, f2); addForce(B.z, f2); addForce(B.y, f2); addForce(B.w, f2);
    }
}

// The frame adds the wing forces per wing node instead of wingApply's one lane: a
// lane per node adds its entries (wing << 2 | 0 induced drag, 1 front, 2 back) in
// the order wingApply reaches that node (wings in file order, then its slots), so
// each node's force is the same sum, bit for bit.
void wingGather(uint wn)
{
    uint4 rec = gStatic[wingNodeBase() + wn];
    float4 F = gNodes[rec.x * 3u + 2u];
    float3 f = F.xyz;
    for (uint k = 0u; k < rec.z; k++)
    {
        uint e = gLists[rec.y + k];
        uint ws = wingState(e >> 2);
        if (gBeams[ws].x != 0.0) continue;
        f = sadd3(f, gBeams[ws + 1u + (e & 3u)].xyz);
    }
    gNodes[rec.x * 3u + 2u] = float4(f, F.w);
}

// The per-frame break check on world positions (before the origin rebase).
void wingCheck(uint gw, float3 origin)
{
    uint ws = wingState(gw);
    if (gBeams[ws].x != 0.0) return;
    uint4 A = gStatic[gP.wingBase + gw * 4u];
    uint4 B = gStatic[gP.wingBase + gw * 4u + 1u];
    float sref = asfloat(gStatic[gP.wingBase + gw * 4u + 2u].y);
    float3 fld = nf3(origin + gNodes[A.x * 3u].xyz);
    float3 frd = nf3(origin + gNodes[A.y * 3u].xyz);
    float3 bld = nf3(origin + gNodes[B.x * 3u].xyz);
    float3 brd = nf3(origin + gNodes[B.y * 3u].xyz);
    float3 x = ssub3(frd, fld);
    float3 cl = scross(x, ssub3(bld, fld));
    float3 cr = scross(x, ssub3(brd, frd));
    if (sdot(cl, cl) > sref || sdot(cr, cr) > sref) gBeams[ws] = float4(1.0, 0.0, 0.0, 0.0);
}

// The fuselage drag D from the front node's velocity and height, for the next step.
void fuseDrag(uint4 rec, float3 origin, uint vehicle)
{
    float width = asfloat(rec.y);
    if (!(width > 0.0)) return;
    float3 a = -gNodes[rec.x * 3u + 1u].xyz;
    float speed = slen(a);
    float rho = airDensity(nf(origin.y + gNodes[rec.x * 3u].y), 288.1);
    float k = sdiv(smul(smul(smul(smul(smul(width, width), 0.5), 0.5), rho), speed), float(rec.z));
    gBeams[gP.aeroState + vehicle] = float4(sscale(a, k), 0.0);
}

// One self-centering command's step: both entries in key order.
void commandUpdate(uint gc)
{
    uint4 R = gStatic[gP.cmdBase + gc * 2u];
    float4 rates = asfloat(gStatic[gP.cmdBase + gc * 2u + 1u]);
    uint j = R.x;
    float centre = asfloat(R.z);
    float4 s0 = gBeams[j * 2u];
    float rest = s0.x;
    float rest0 = asfloat(gStatic[gP.beamBase + j * 3u + 2u].w);
    float mode = gBeams[cmdState(gc)].x;
    for (uint e = 0u; e < 2u; e++)
    {
        float sigma = (e == 0u) == (R.y != 0u) ? -1.0 : 1.0;
        if (rest0 == 0.0 || rest == 0.0) continue;
        float r = sdiv(rest, rest0);
        if (abs(nf(r - centre)) < 1e-4) mode = 0.0;
        else
        {
            float old = mode;
            mode = r > centre ? -1.0 : 1.0;
            if (old != 0.0 && old != mode)
            {
                rest = smul(centre, rest0);
                mode = 0.0;
            }
        }
        r = sdiv(rest, rest0);
        bool may = (sigma > 0.0 && r < rates.w) || (sigma < 0.0 && r > rates.z);
        if (may)
        {
            float v = sigma * mode > 0.0 ? 1.0 : 0.0;
            float rate = sigma > 0.0 ? rates.y : rates.x;
            float x = sdiv(smul(smul(smul(rate, v), 1.0), STEP), rest);
            rest = nf(fma(sigma * x, rest, rest));
        }
    }
    if (rest != s0.x) gBeams[j * 2u] = float4(rest, s0.yzw);
    gBeams[cmdState(gc)] = float4(mode, 0.0, 0.0, 0.0);
}

// --- shocks2 (engine/docs/explicit-beams-wheels2-shocks2-spec.md §6; CPU
// reference engine_softbody::explicit_beams::Shock2) ---
//   gStatic  the beam's records hold (k in, c in, bump k, bump c) and (short bound,
//            long bound, shock index, L0); at shock2Base per shock (k out, c out,
//            progression k in, progression c in), (progression k out, progression
//            c out, soft bump, 0) as bits.
// Every float expression strict, in the specification's order.

// min(x, 1): 1 if 1 < x, else x (a NaN passes through).
float min1(float x) { return (1.0 < x) ? 1.0 : x; }
// max(a, b): b if a < b, else a.
float maxAB(float a, float b) { return (a < b) ? b : a; }

// k0 + (pk·k0)·λ and c0 + (pc·c0)·λ, λ = 1 for a zero bound, else min((e / (bound·L))², 1).
float2 shock2Progressive(float k0, float c0, float pk, float pc, float e, float bound, float rest)
{
    float lambda = 1.0;
    if (bound != 0.0)
    {
        float r = sdiv(e, smul(bound, rest));
        lambda = min1(smul(r, r));
    }
    return float2(nf(k0 + smul(smul(pk, k0), lambda)), nf(c0 + smul(smul(pc, c0), lambda)));
}

// The soft bump on (k, c): k + ((k + 100)·pk)·μ, μ = 1 for a zero bound, else
// min((((e − e_b)·5) / (bound·L))², 1).
float2 shock2Bump(float2 kc, float pk, float pc, float excess, float bound, float rest)
{
    float mu = 1.0;
    if (bound != 0.0)
    {
        float q = sdiv(smul(excess, 5.0), smul(bound, rest));
        mu = min1(smul(q, q));
    }
    return float2(nf(kc.x + smul(smul(nf(kc.x + 100.0), pk), mu)), nf(kc.y + smul(smul(nf(kc.y + 100.0), pc), mu)));
}

void shock2Update(uint gj, uint local, bool fast, uint groupOff, uint parity, uint4 bn, float4 s1)
{
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    // The host writes shock indices below shock2Count; clamp in any case.
    uint si = min(r2.z, max(gP.shock2Count, 1u) - 1u);
    float4 t0 = asfloat(gStatic[gP.shock2Base + si * 2u]);
    float4 t1 = asfloat(gStatic[gP.shock2Base + si * 2u + 1u]);
    float kIn = c1.x;
    float cIn = c1.y;
    float kBump = c1.z;
    float cBump = c1.w;
    float sb = asfloat(r2.x);
    float lb = asfloat(r2.y);
    float kOut = t0.x;
    float cOut = t0.y;
    float pkIn = t0.z;
    float pcIn = t0.w;
    float pkOut = t1.x;
    float pcOut = t1.y;
    bool soft = t1.z != 0.0;
    int group = asint(bn.w);
    float4 s0 = gBeams[gj * 2u];
    float rest = s0.x;
    float gate = s0.w;
    float strength = s1.x;
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float inv = fast ? sFastInvSqrt(q) : sdiv(1.0, ssqrt(q));
    float len = smul(q, inv);
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    // §6.1: the velocity picks the side.
    float2 kc = (u > 0.0) ? shock2Progressive(kOut, cOut, pkOut, pcOut, e, lb, rest)
                          : shock2Progressive(kIn, cIn, pkIn, pcIn, e, sb, rest);
    bool past = e > smul(lb, rest) || e < smul(-sb, rest);
    if (!soft)
    {
        // §6.2: the bump rates replace.
        if (past) kc = float2(kBump, cBump);
    }
    else
    {
        // §6.3.
        float softRest = smul(rest, 0.8);
        float el = smul(lb, softRest);
        float es = smul(-sb, softRest);
        if (e > el)
        {
            kc = shock2Bump(shock2Progressive(kOut, cOut, pkOut, pcOut, e, lb, rest), pkOut, pcOut, nf(e - el), lb,
                            rest);
            if (u < 0.0) kc = float2(kIn, cIn);
        }
        else if (e < es)
        {
            // The out factors here too (quirk).
            kc = shock2Bump(shock2Progressive(kIn, cIn, pkIn, pcIn, e, sb, rest), pkOut, pcOut, nf(e - es), sb,
                            rest);
            if (u > 0.0) kc = float2(kOut, cOut);
        }
        if (past) kc = float2(maxAB(kc.x, kBump), maxAB(kc.y, cBump));
    }
    float f = nf(smul(-kc.x, e) - smul(kc.y, u));
    float last = f;
    float tau = abs(f);
    bool broken = false;
    // No yield; the break test when |F| exceeds the gate and the strength.
    if (tau > gate && tau > strength)
    {
        bool saved = false;
        if ((gStatic[gP.nodeInfoBase + bn.x].x & NODE_ON_TRIANGLE) != 0u && livePlain(bn.x) < 3u) saved = true;
        if ((gStatic[gP.nodeInfoBase + bn.y].x & NODE_ON_TRIANGLE) != 0u && livePlain(bn.y) < 3u) saved = true;
        if (saved)
        {
            strength = 2.0 * gate;
        }
        else
        {
            broken = true;
            f = 0.0;
            if (group > 0)
            {
                gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
                gGroups[groupSlot(parity, groupOff, 0u)].min(local);
            }
        }
    }
    if (strength != s1.x || broken) gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// --- Triggers, key-driven commands and slide nodes
// (engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md §2–3, §6; CPU
// reference engine_softbody::explicit_triggers) ---
//   gStatic  a trigger beam's third record holds (short bound, long bound, short key
//            | long key << 8 | enabled << 16, L0); at slideBase per slide node
//            (node global, first segment global or NONE, 0, 0), (spring, break
//            force, tolerance, 0) bits; at segBase per segment (beam global,
//            previous segment global or NONE, next or NONE, 0).
//   gNext    at keyBase per vehicle and step parity KEYS trigger inputs (1: held):
//            the beam phase of step s writes parity s & 1, the command phase of
//            step s + 1 reads it, the beam phase of step s + 1 clears it.
//   gBeams   at slideState per slide node (current segment or -1, broken away, 0,
//            0); at slideAcc per slide target node (a slide node or a rail end:
//            node info flags >> NODE_SLOT_SHIFT, 1-based) the slide forces of the
//            last step, added by its node phase and cleared.
// One lane per vehicle runs its slide nodes in spawn order during the
// self-collision phase (positions are final for the step there, and the beam
// phase's breaks are visible), writing only the slots, which no other phase reads
// before the next node phase. Every loop is bounded by a count from the static
// records, every index clamped.
static const uint KEYS = 85u;
static const uint NONE = 0xFFFFFFFFu;

uint keySlot(uint vehicle, uint parity, uint key)
{
    return gP.keyBase + (vehicle * 2u + parity) * KEYS + min(key, KEYS - 1u);
}

// A trigger beam: zero force (k = c = 0), then the bounds test for the next step's
// command pass (spec §2.5).
void triggerUpdate(uint gj, bool fast, uint parity, uint4 bn)
{
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    float4 s0 = gBeams[gj * 2u];
    float rest = s0.x;
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float inv = fast ? sFastInvSqrt(q) : sdiv(1.0, ssqrt(q));
    float len = smul(q, inv);
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    float f = nf(smul(-c1.x, e) - smul(c1.y, u));
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
    uint keys = r2.z;
    if (((keys >> 16) & 1u) == 0u) return;
    float sb = asfloat(r2.x);
    float lb = asfloat(r2.y);
    uint vehicle = bn.z >> 16;
    if (e > smul(lb, rest)) gNext[keySlot(vehicle, parity, (keys >> 8) & 0xFFu)] = 1u;
    else if (e < smul(-sb, rest)) gNext[keySlot(vehicle, parity, keys & 0xFFu)] = 1u;
}

// One command line's step with key input (spec §2.6): both entries in key order,
// the keys' values from the triggers of the last step (parity `prev`).
void commandUpdateKeys(uint gc, uint vehicle, uint prev)
{
    uint4 R = gStatic[gP.cmdBase + gc * 2u];
    float4 rates = asfloat(gStatic[gP.cmdBase + gc * 2u + 1u]);
    uint j = R.x;
    float centre = asfloat(R.z);
    uint ck = R.w & 0xFFu;
    uint ek = (R.w >> 8) & 0xFFu;
    bool centering = ((R.w >> 16) & 1u) != 0u;
    float vc = ck == 0u ? 0.0 : float(gNext[keySlot(vehicle, prev, ck)]);
    float ve = ek == 0u ? 0.0 : float(gNext[keySlot(vehicle, prev, ek)]);
    bool locked = centering && (vc >= 0.5 || ve >= 0.5);
    float4 s0 = gBeams[j * 2u];
    float rest = s0.x;
    float rest0 = asfloat(gStatic[gP.beamBase + j * 3u + 2u].w);
    float mode = gBeams[cmdState(gc)].x;
    if (locked) mode = 0.0;
    for (uint e = 0u; e < 2u; e++)
    {
        float sigma = (e == 0u) == (R.y != 0u) ? -1.0 : 1.0;
        if (rest0 == 0.0 || rest == 0.0) continue;
        if (centering && !locked)
        {
            float r = sdiv(rest, rest0);
            if (abs(nf(r - centre)) < 1e-4) mode = 0.0;
            else
            {
                float old = mode;
                mode = r > centre ? -1.0 : 1.0;
                if (old != 0.0 && old != mode)
                {
                    rest = smul(centre, rest0);
                    mode = 0.0;
                }
            }
        }
        float r = sdiv(rest, rest0);
        bool may = (sigma > 0.0 && r < rates.w) || (sigma < 0.0 && r > rates.z);
        if (may)
        {
            float v = sigma > 0.0 ? ve : vc;
            if (centering && !locked && sigma * mode > 0.0) v = 1.0;
            float rate = sigma > 0.0 ? rates.y : rates.x;
            float x = sdiv(smul(smul(smul(rate, v), 1.0), STEP), rest);
            rest = nf(fma(sigma * x, rest, rest));
        }
    }
    if (rest != s0.x) gBeams[j * 2u] = float4(rest, s0.yzw);
    gBeams[cmdState(gc)] = float4(mode, 0.0, 0.0, 0.0);
}

// Slot `slot`'s value `s` plus the contributions slideNode registered this step,
// added in spawn order (slide node, then own node, rail end a, rail end b: the
// order the serial pass adds them in); clears the registration.
float3 slidePending(float3 s, uint slot, uint n)
{
    uint4 rec = gStatic[slotRecBase() + slot];
    uint m = min(n, rec.y);
    for (uint i = 1u; i < m; i++)
    {
        uint key = gNext[gP.slotListBase + rec.x + i];
        uint j = i;
        while (j > 0u && gNext[gP.slotListBase + rec.x + j - 1u] > key)
        {
            gNext[gP.slotListBase + rec.x + j] = gNext[gP.slotListBase + rec.x + j - 1u];
            j--;
        }
        gNext[gP.slotListBase + rec.x + j] = key;
    }
    for (uint i = 0u; i < m; i++) s = sadd3(s, gBeams[slideContribBase() + gNext[gP.slotListBase + rec.x + i]].xyz);
    gAcc[gP.slotCntBase + slot].store(0);
    return s;
}

// The node phase: add and clear a slide target's forces (slot + 1 = slot1).
float3 slideTake(float3 f, uint slot1)
{
    uint slot = min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u);
    uint at = gP.slideAcc + slot;
    float3 s = gBeams[at].xyz;
    uint n = uint(gAcc[gP.slotCntBase + slot].load());
    if (n != 0u) s = slidePending(s, slot, n);
    gBeams[at] = float4(0.0, 0.0, 0.0, 0.0);
    return sadd3(f, s);
}

void slideAdd(uint node, float3 f)
{
    uint slot1 = gStatic[gP.nodeInfoBase + node].x >> NODE_SLOT_SHIFT;
    if (slot1 == 0u) return;
    uint at = gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u);
    gBeams[at] = float4(sadd3(gBeams[at].xyz, f), 0.0);
}

// Beam j is broken as the CPU sees it after this step's beam pass: its flag, or a
// detacher-group break this step (the next node phase marks those).
bool brokenNow(uint j, uint groupOff, uint parity)
{
    if (gBeams[j * 2u + 1u].z != 0.0) return true;
    int g = asint(gStatic[gP.beamBase + j * 3u].w);
    if (g == 0) return false;
    return gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load() != NO_GROUP;
}

// The nearest point of segment gs (world positions) to x, its distance and ratio.
struct SegPoint
{
    float3 q;
    float dist;
    float ratio;
};

SegPoint segmentPoint(uint gs, float3 origin, float3 x)
{
    uint j = gStatic[gP.segBase + gs].x;
    uint4 bn = gStatic[gP.beamBase + j * 3u];
    float3 p1 = nf3(origin + gNodes[bn.x * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[bn.y * 3u].xyz);
    float3 d = ssub3(p2, p1);
    float beta = slen(d);
    if (beta > 0.0) d = sscale(d, sdiv(1.0, beta));
    float proj = sdot(ssub3(x, p1), d);
    float m = (beta < proj) ? beta : proj;
    float tau = (0.0 < m) ? m : 0.0;
    SegPoint r;
    r.q = sadd3(p1, sscale(d, tau));
    r.dist = slen(ssub3(r.q, x));
    r.ratio = beta > 0.0 ? sdiv(tau, beta) : 0.0;
    return r;
}

uint clampSeg(uint gs) { return min(gs, max(gP.segTotal, 1u) - 1u); }

// A vehicle's slide nodes in spawn order (spec §3.4), one lane.
void slidePass(uint slideOff, uint slideCount, float3 origin, uint groupOff, uint parity)
{
    uint count = min(slideCount, gP.slideTotal - min(slideOff, gP.slideTotal));
    for (uint k = 0u; k < count; k++)
    {
        uint gs = slideOff + k;
        float4 st = gBeams[gP.slideState + gs];
        if (st.x < 0.0) continue;
        uint cur = clampSeg(uint(st.x));
        uint4 seg = gStatic[gP.segBase + cur];
        if (brokenNow(seg.x, groupOff, parity)) continue;
        uint4 S0 = gStatic[gP.slideBase + gs * 2u];
        float4 S1 = asfloat(gStatic[gP.slideBase + gs * 2u + 1u]);
        float3 x = nf3(origin + gNodes[S0.x * 3u].xyz);
        // At most one segment's hop.
        uint best = cur;
        SegPoint bp = segmentPoint(cur, origin, x);
        if (seg.y != NONE)
        {
            SegPoint p = segmentPoint(clampSeg(seg.y), origin, x);
            if (p.dist < bp.dist) { best = clampSeg(seg.y); bp = p; }
        }
        if (seg.z != NONE)
        {
            SegPoint p = segmentPoint(clampSeg(seg.z), origin, x);
            if (p.dist < bp.dist) { best = clampSeg(seg.z); bp = p; }
        }
        bool away = st.y != 0.0;
        if (!away)
        {
            float3 w = ssub3(bp.q, x);
            float delta = slen(w);
            w = snormalise(w);
            float stretch = nf(delta - S1.z);
            float fLen = smul(-S1.x, (0.0 < stretch) ? stretch : 0.0);
            float3 P = sscale(w, fLen);
            if (slen(P) > S1.y) away = true;
            uint4 bn = gStatic[gP.beamBase + gStatic[gP.segBase + best].x * 3u];
            slideAdd(S0.x, -P);
            slideAdd(bn.x, sscale(P, nf(1.0 - bp.ratio)));
            slideAdd(bn.y, sscale(P, bp.ratio));
        }
        gBeams[gP.slideState + gs] = float4(float(best), away ? 1.0 : 0.0, 0.0, 0.0);
    }
}

// The frame runs the slide nodes in parallel, one lane each: slideNode is one
// iteration of slidePass, but instead of adding into the slots in spawn order it
// stores its contributions (at slideContrib, 3 per slide node) and registers each
// with its slot (a count and a list, capacity: every contribution the slot can
// get); the next node phase (slideTake) or the frame's end (slideSettle) adds them
// in spawn order. The same sums, bit for bit; slide nodes do not read the slots.
void slideEmit(uint node, float3 f, uint gs, uint which)
{
    uint slot1 = gStatic[gP.nodeInfoBase + node].x >> NODE_SLOT_SHIFT;
    if (slot1 == 0u) return;
    uint slot = min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u);
    uint key = gs * 3u + which;
    gBeams[slideContribBase() + key] = float4(f, 0.0);
    uint4 rec = gStatic[slotRecBase() + slot];
    uint i = uint(gAcc[gP.slotCntBase + slot].add(1));
    if (i < rec.y) gNext[gP.slotListBase + rec.x + i] = key;
}

void slideNode(uint gs, float3 origin, uint groupOff, uint parity)
{
    float4 st = gBeams[gP.slideState + gs];
    if (st.x < 0.0) return;
    uint cur = clampSeg(uint(st.x));
    uint4 seg = gStatic[gP.segBase + cur];
    if (brokenNow(seg.x, groupOff, parity)) return;
    uint4 S0 = gStatic[gP.slideBase + gs * 2u];
    float4 S1 = asfloat(gStatic[gP.slideBase + gs * 2u + 1u]);
    float3 x = nf3(origin + gNodes[S0.x * 3u].xyz);
    uint best = cur;
    SegPoint bp = segmentPoint(cur, origin, x);
    if (seg.y != NONE)
    {
        SegPoint p = segmentPoint(clampSeg(seg.y), origin, x);
        if (p.dist < bp.dist) { best = clampSeg(seg.y); bp = p; }
    }
    if (seg.z != NONE)
    {
        SegPoint p = segmentPoint(clampSeg(seg.z), origin, x);
        if (p.dist < bp.dist) { best = clampSeg(seg.z); bp = p; }
    }
    bool away = st.y != 0.0;
    if (!away)
    {
        float3 w = ssub3(bp.q, x);
        float delta = slen(w);
        w = snormalise(w);
        float stretch = nf(delta - S1.z);
        float fLen = smul(-S1.x, (0.0 < stretch) ? stretch : 0.0);
        float3 P = sscale(w, fLen);
        if (slen(P) > S1.y) away = true;
        uint4 bn = gStatic[gP.beamBase + gStatic[gP.segBase + best].x * 3u];
        slideEmit(S0.x, -P, gs, 0u);
        slideEmit(bn.x, sscale(P, nf(1.0 - bp.ratio)), gs, 1u);
        slideEmit(bn.y, sscale(P, bp.ratio), gs, 2u);
    }
    gBeams[gP.slideState + gs] = float4(float(best), away ? 1.0 : 0.0, 0.0, 0.0);
}

// The frame's end: add a slot's registered contributions into it (as the serial
// pass would have left it).
void slideSettle(uint slot)
{
    uint n = uint(gAcc[gP.slotCntBase + slot].load());
    if (n == 0u) return;
    uint at = gP.slideAcc + slot;
    gBeams[at] = float4(slidePending(gBeams[at].xyz, slot, n), 0.0);
}

// nodeUpdate for a vehicle with a `fusedrag` line: the restart adds the stored
// fuselage drag D instead of the air drag and turbulence.
void nodeUpdateFuse(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits)
{
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, parity, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        if (ground > x.y)
            f += contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
        for (uint b = 0u; b < boxCount; b++)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u + 1u].xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float3 n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
                f += contactForce(f, vel, mass, n, boxSurface(), 0.0, V.w);
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
}
#ifdef EB_VV_SOLO
uint nodeUpdateFuseSolo(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits)
{
    float4 P = gNodes[gi * 3u];
    // The position the last step left (each step's checked once: the frame's
    // last at its end).
    if (any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
    float4 V = gNodes[gi * 3u + 1u];
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, parity, true);
    if (hits)
    {
        f = f + accForce(gi);
        gAcc[gi * 3u].store(0);
        gAcc[gi * 3u + 1u].store(0);
        gAcc[gi * 3u + 2u].store(0);
    }
    if ((info.x >> NODE_SLOT_SHIFT) != 0u) f = slideTake(f, info.x >> NODE_SLOT_SHIFT);
    uint flags = info.x;
    float3 pos = P.xyz;
    float3 vel = V.xyz;
    float mass = P.w;
    if ((flags & NODE_NO_GROUND) == 0u)
    {
        float3 x = origin + pos;
        if (ground > x.y)
            f += contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), groundSurface(), ground - x.y, V.w);
        for (uint b = 0u; b < boxCount; b++)
        {
            float3 lo = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u].xyz);
            float3 hi = asfloat(gStatic[gP.boxBase + (boxOff + b) * 2u + 1u].xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float3 n = float3(0.0, 0.0, -1.0);
                float d = x.z - lo.z;
                if (hi.z - x.z < d) { d = hi.z - x.z; n = float3(0.0, 0.0, 1.0); }
                if (x.x - lo.x < d) { d = x.x - lo.x; n = float3(-1.0, 0.0, 0.0); }
                if (hi.x - x.x < d) { d = hi.x - x.x; n = float3(1.0, 0.0, 0.0); }
                if (x.y - lo.y < d) { d = x.y - lo.y; n = float3(0.0, -1.0, 0.0); }
                if (hi.y - x.y < d) { d = hi.y - x.y; n = float3(0.0, 1.0, 0.0); }
                f += contactForce(f, vel, mass, n, boxSurface(), 0.0, V.w);
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
    gNodes[gi * 3u] = float4(pos, mass);
    gNodes[gi * 3u + 1u] = float4(vel, V.w);
    gNodes[gi * 3u + 2u] = float4(f, 0.0);
    return 0u;
}
#endif
#endif

// --- Vehicle–vehicle contact: records the frame entry uses (the rest of the
// section is at the end of the file) ---

// Per vehicle VV_RECS records in gMisc at vvVeh: its padded frame box (lo, hi),
// the bounds its nodes must stay in this frame (lo, hi), flags (x: too big for
// the vehicle grid, y: grouped this frame, z: its group, w: its self-collision
// throttles reset at the next throttle update), member (x: its index in its
// group's sorted members).
static const uint VV_RECS = 6u;
static const uint VV_P_LO = 0u;
static const uint VV_P_HI = 1u;
static const uint VV_ALLOWED_LO = 2u;
static const uint VV_ALLOWED_HI = 3u;
static const uint VV_FLAGS = 4u;
static const uint VV_MEMBER = 5u;
uint vvRec(uint v, uint r) { return gP.vvVeh + v * VV_RECS + r; }

// Counters in gAcc after the box keys: overlapping pairs, big vehicles, solo
// vehicles, groups, a vehicle outside its bounds (index + 1), pair overflow, no
// convergence of the grouping.
static const uint VV_PAIRS = 0u;
static const uint VV_BIGS = 1u;
static const uint VV_SOLOS = 2u;
static const uint VV_GROUPS = 3u;
static const uint VV_OUTSIDE = 4u;
static const uint VV_OVERFLOW = 5u;
static const uint VV_UNCONVERGED = 6u;
static const uint VV_TOO_BIG = 7u;
static const uint VV_CTL_WORDS = 8u;
uint vvCtl(uint i) { return gP.vvAcc + gP.vehicles * 12u + i; }

// A value one lane decided, read by every lane as uniform (the WGSL
// post-process turns its reads into workgroupUniformLoad).
groupshared uint gsUniformLanes;



void vvReportOutside(uint v)
{
    gAcc[vvCtl(VV_OUTSIDE)].max(int(v + 1u));
}

#ifdef EB_VV_GROUPS
void contactGroup(uint g, uint lane);
#endif

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_frame(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
#ifdef EB_VV_SOLO
#ifdef EB_VV_GROUPS
    // Workgroups past the vehicles step the contact groups (one dispatch: groups
    // and solo vehicles run side by side).
    if (gid.x >= gP.vehicles)
    {
        contactGroup(gid.x - gP.vehicles, lane);
        return;
    }
#endif
    // A vehicle of a contact group this frame is stepped by its group's workgroup.
    if (lane == 0u) gsUniformLanes = gMisc[vvRec(v, VV_FLAGS)].y;
    sync();
    if (gsUniformLanes != 0u) return;
#endif
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = r0.y;
    uint beamOff = r0.z;
    uint beamCount = r0.w;
    uint triOff = r1.x;
    uint triCount = r1.y;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = r2.y;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
    // The wing break check, once per frame on the positions before the rebase.
    {
        float3 o = asfloat(gMisc[v].xyz);
        for (uint w = lane; w < aero0.y; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif

    // Origin rebase: every lane reads node 0 before any lane moves it.
    float3 first = gNodes[nodeOff * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane; i < nodeCount; i += LANES)
        {
            float4 P = gNodes[(nodeOff + i) * 3u];
            gNodes[(nodeOff + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
    }
    sync();
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint stepBase = misc.w;
    if (lane == 0u)
    {
        gsHit = 1u;
        gsTestedCount[0].store(0u);
        gsTestedCount[1].store(0u);
    }
    // Static records the lane reads every step, kept in registers: its first node's.
    uint4 myInfo = gStatic[gP.nodeInfoBase + nodeOff + min(lane, max(nodeCount, 1u) - 1u)];
#ifdef EB_VV_SOLO
    if (lane == 0u)
    {
        // Centre and half extents (relative).
        float3 lo = asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz) - origin;
        float3 hi = asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz) - origin;
        gsVvBounds[0] = 0.5 * (lo + hi);
        gsVvBounds[1] = 0.5 * (hi - lo);
    }
#endif

    sync();

    for (uint s = 0u; s < gP.steps; s++)
    {
        uint step = stepBase + s;
        uint parity = step & 1u;
        bool hits = gsHit != 0u;
#ifdef EB_VV_SOLO
        for (uint i = lane; i < nodeCount; i += LANES)
            nodeUpdateSolo(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
#else
#ifdef EB_ABLATE
        uint nodeEnd = (gP.ablate & 2u) != 0u ? 0u : ((gP.ablate & 4u) != 0u ? min(nodeCount, LANES) : nodeCount);
#else
        uint nodeEnd = nodeCount;
#endif
        for (uint i = lane; i < nodeEnd; i += LANES)
            nodeUpdate(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
#endif

        for (uint g = lane; g < groupCount; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
        if (candCount > 0u)
        {
            for (uint t = lane; t < triCount; t += LANES)
            {
                int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
                if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
            }
        }
        for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
        if (lane == 0u) gsPairCount.store(0u);
        sync();
        if (lane == 0u)
        {
            gsHit = 0u;
            gsTestedCount[parity ^ 1u].store(0u);
            // This step's self-collision: the grid, or (few pairs) every candidate.
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            gsUniformLanes = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u;
        }
        if (cell > 0.0)
        {
            for (uint c = lane; c < candCount; c += LANES)
            {
                float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
                uint b = gridBucket(int3(floor(p / cell)));
                gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
            }
        }
#ifdef EB_AERO
#ifdef EB_ABLATE
        if ((gP.ablate & 16u) == 0u) {
#endif
        // Wings (forces from this step's integrated state), the fuselage drag for
        // the next step, and the commands' rest lengths; then one lane adds the
        // wings' forces in file order while the others start the beams.
        for (uint w = lane; w < aero0.y; w += LANES) wingForces(aero0.x + w, origin);
        if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
        if ((flagsV & FLAG_KEYS) != 0u)
        {
            for (uint c = lane; c < aero0.w; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
        }
        else
        {
            for (uint c = lane; c < aero0.w; c += LANES) commandUpdate(aero0.z + c);
        }
        sync();
        {
            uint4 ext = gStatic[gP.vehExtBase + v];
#ifdef EB_ABLATE
            if ((gP.ablate & 8u) != 0u) ext.w = 0u;
#endif
            for (uint k = lane; k < ext.w; k += LANES) wingGather(ext.z + k);
        }
        // The last step's trigger inputs, read by this step's commands: cleared for
        // the next step's beam phase.
        if ((flagsV & FLAG_KEYS) != 0u)
        {
            for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
        }
#ifdef EB_ABLATE
        }
#endif
#endif
#ifdef EB_ABLATE
        uint beamEnd = (gP.ablate & 32u) != 0u ? 0u : beamCount;
#else
        uint beamEnd = beamCount;
#endif
        for (uint j = lane; j < beamEnd; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
        // Without the grid this step: list the candidate pairs now (lanes in
        // reverse, so those with one beam fewer start first).
        uint listedB = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        bool early = candCount > 0u && !(cell > 0.0 && listedB * candCount > BRUTE_PAIRS_PER_LANE * LANES);
        if (early)
        {
            uint lanes = clamp(LANES / max(listedB, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            uint rl = LANES - 1u - lane;
            for (uint p = rl; p < listedB * lanes && rl < stride; p += stride)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
                trianglePairs(t, triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount);
            }
        }
        sync();
        uint gridOn = gsUniformLanes;
#ifdef EB_ABLATE
        if (gridOn != 0u && (gP.ablate & 1u) == 0u)
#else
        if (gridOn != 0u)
#endif
        {
            // Offsets: per lane its buckets' sum, a scan over blocks of lanes, then
            // the blocks' totals, then each lane's buckets.
            uint per = GRID_BUCKETS / LANES;
            uint block = LANES / GRID_BLOCKS;
            uint sum = 0u;
            for (uint k = 0u; k < per; k++) sum += gsCount[lane * per + k].load();
            gsPart[lane] = sum;
            sync();
            if (lane < GRID_BLOCKS)
            {
                uint run = 0u;
                for (uint k = 0u; k < block; k++)
                {
                    uint x = gsPart[lane * block + k];
                    gsPart[lane * block + k] = run;
                    run += x;
                }
                gsBlk[lane] = run;
            }
            sync();
            if (lane == 0u)
            {
                uint run = 0u;
                for (uint k = 0u; k < GRID_BLOCKS; k++)
                {
                    uint x = gsBlk[k];
                    gsBlk[k] = run;
                    run += x;
                }
                gsOff[GRID_BUCKETS] = run;
            }
            sync();
            uint run = gsPart[lane] + gsBlk[lane / block];
            for (uint k = 0u; k < per; k++)
            {
                gsOff[lane * per + k] = run;
                run += gsCount[lane * per + k].load();
            }
            sync();
            if (cell > 0.0)
            {
                for (uint c = lane; c < candCount; c += LANES)
                {
                    uint br = gNext[candOff + c];
                    uint h = gLists[candOff + c];
                    gBeams[gP.gridBase + candOff + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] =
                        float4(gNodes[h * 3u].xyz, asfloat(h));
                }
            }
            sync();
        }
        uint pairs = gsPairCount.load();
#ifdef EB_ABLATE
        if ((gP.ablate & 256u) != 0u) {} else
#endif
        if (early && pairs <= PAIR_LIST)
        {
            for (uint i = lane; i < pairs; i += LANES)
            {
                uint e = gsPairs[i];
                tryContact(triangleGeo(triOff + (e >> 16), origin, range), gLists[candOff + (e & 0xFFFFu)], origin,
                           range, beamOff, groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u && cell > 0.0 && gridOn != 0u)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            for (uint p = lane; p < listed * GRID_SUBS; p += LANES)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
                triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff,
                             groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u && cell > 0.0)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            uint lanes = clamp(LANES / max(listed, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            for (uint p = lane; p < listed * lanes && lane < stride; p += stride)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
                triangleGridBrute(triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, beamOff,
                                  groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u)
        {
            // Up to LANES lanes per step: `lanes` per triangle, one per candidate when
            // they fit.
            uint lanes = clamp(LANES / max(triCount, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            for (uint p = lane; p < triCount * lanes && lane < stride; p += stride)
                triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff,
                             groupOff, groupCount, parity);
        }
#ifdef EB_DET_HITS
        sync();
        if (lane == 0u) detApply(v, origin, range, beamOff, groupOff, groupCount, parity);
#endif
#ifdef EB_AERO
        // Slide nodes, after this step's beams (alongside the contacts), a lane each.
        if ((flagsV & FLAG_SLIDES) != 0u)
        {
            uint count = min(aero2.y, gP.slideTotal - min(aero2.x, gP.slideTotal));
            for (uint k = lane; k < count; k += LANES) slideNode(aero2.x + k, origin, groupOff, parity);
        }
#endif
        sync();
    }
    if (gP.steps > 0u && candCount > 0u)
    {
        for (uint t = lane; t < triCount; t += LANES) triangleFinish(triOff + t);
    }
#ifdef EB_AERO
    if ((flagsV & FLAG_SLIDES) != 0u)
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        for (uint k = lane; k < ext.y; k += LANES) slideSettle(ext.x + k);
    }
#endif
    // The last step's group breaks, for a read-back (the next node phase applies
    // them to the forces).
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint last = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane; j < beamCount; j += LANES) groupDrops(beamOff + j, beamOff, groupOff, last, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
#ifdef EB_VV_SOLO
    // The last step's positions (each earlier step's were checked at the start of
    // the next step's node phase).
    for (uint i = lane; i < nodeCount; i += LANES)
    {
        float3 p = gNodes[(nodeOff + i) * 3u].xyz;
        if (any(abs(p - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(v);
    }
#endif
}

// Bucket slots of the wide passes' grid (a fuller bucket overflows to a list).
static const uint WIDE_SLOTS = 16u;

// --- Wide passes: one thread per element across every vehicle, a dispatch per
// pass, for large vehicles (the whole GPU on one vehicle). Per frame:
// wide_begin, wide_rebase, then per step wide_nodes, wide_beams, wide_tris
// (each step binding its own parameter copy, whose pad2 is the step), then
// wide_end, wide_close. Same per-element functions as
// explicit_beams_frame, in the same order within a step; the grid offsets and
// tested lists live in buffers instead of workgroup memory, and the grid is
// fixed bucket slots (no scan). Every vehicle in
// a wide dispatch has run the same number of steps (the host checks).

// The wide passes' vehicle table, in constant memory: per vehicle its 4 static
// records, then per vehicle (origin bits xyz, steps run) copied from gMisc after
// the frame's rebase.
static const uint WIDE_MAX_VEHICLES = 256u;
struct VehicleTable
{
    uint4 rec[WIDE_MAX_VEHICLES * 5u];
};
[[vk::binding(9, 0)]] ConstantBuffer<VehicleTable> gVt;

struct Vehicle
{
    uint nodeOff, nodeCount, beamOff, beamCount, triOff, triCount, boxOff, boxCount;
    uint candOff, candCount, groupOff, groupCount, flags;
    float ground, range, cell;
};

Vehicle vehicleAt(uint v)
{
    uint4 r0 = gVt.rec[v * 4u];
    uint4 r1 = gVt.rec[v * 4u + 1u];
    uint4 r2 = gVt.rec[v * 4u + 2u];
    uint4 r3 = gVt.rec[v * 4u + 3u];
    Vehicle x;
    x.nodeOff = r0.x; x.nodeCount = r0.y; x.beamOff = r0.z; x.beamCount = r0.w;
    x.triOff = r1.x; x.triCount = r1.y; x.boxOff = r1.z; x.boxCount = r1.w;
    x.candOff = r2.x; x.candCount = r2.y; x.groupOff = r2.z; x.groupCount = r2.w;
    x.flags = r3.x; x.ground = asfloat(r3.y); x.range = asfloat(r3.z); x.cell = asfloat(r3.w);
    return x;
}

// The step within the frame (each step's dispatches bind their own parameter
// copy) and a vehicle's step.
uint wideStep() { return gP.pad2; }
uint vehicleStep(uint v) { return gVt.rec[WIDE_MAX_VEHICLES * 4u + v].w + wideStep(); }
float3 vehicleOrigin(uint v) { return asfloat(gVt.rec[WIDE_MAX_VEHICLES * 4u + v].xyz); }

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_begin(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= gP.vehicles) return;
    Vehicle x = vehicleAt(v);
    float3 first = gNodes[x.nodeOff * 3u].xyz;
    float3 shift = length(first) > REBASE_RADIUS ? first : float3(0.0, 0.0, 0.0);
    gMisc[gP.wideMisc + 1u + v] = uint4(asuint(shift), 0u);
    gAcc[gP.testedBase + v * 2u].store(0);
    gAcc[gP.testedBase + v * 2u + 1u].store(0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_rebase(uint3 id : SV_DispatchThreadID)
{
    uint gi = id.x;
    if (gi >= gP.totalNodes) return;
    uint v = gStatic[gP.nodeInfoBase + gi].w;
    float3 shift = asfloat(gMisc[gP.wideMisc + 1u + v].xyz);
    if (all(shift == float3(0.0, 0.0, 0.0))) return;
    float4 P = gNodes[gi * 3u];
    gNodes[gi * 3u] = float4(P.xyz - shift, P.w);
    if (gi == gStatic[v * 4u].x)
    {
        uint4 m = gMisc[v];
        gMisc[v] = uint4(asuint(asfloat(m.xyz) + shift), m.w);
    }
}

// Nodes, then group slots, grid counts (and overflow and this step's tested counts).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_nodes(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < gP.totalNodes)
    {
        uint4 info = gStatic[gP.nodeInfoBase + i];
        uint v = info.w;
        Vehicle x = vehicleAt(v);
        float3 origin = vehicleOrigin(v);
        nodeUpdate(i, info, v, i - x.nodeOff, vehicleStep(v), origin, x.flags, x.ground, x.boxOff, x.boxCount,
                   x.beamOff, x.groupOff, x.groupCount, true);
        return;
    }
    i -= gP.totalNodes;
    if (i < gP.groupTotal)
    {
        // Every vehicle has the same step parity.
        gGroups[groupSlot(vehicleStep(0u) & 1u, 0u, i)].store(NO_GROUP);
        return;
    }
    i -= gP.groupTotal;
    if (i < gP.vehicles * GRID_BUCKETS)
    {
        gAcc[gP.gridCountBase + i].store(0);
        if (i < gP.vehicles)
        {
            gAcc[gP.overflowBase + i].store(0);
            gAcc[gP.testedBase + i * 2u + (vehicleStep(i) & 1u)].store(0);
        }
    }
}

// Beams, then grid filling (candidates), then triangles' throttle and tested list.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_beams(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        Vehicle x = vehicleAt(v);
        beamUpdate(i, i - x.beamOff, (x.flags & FLAG_FAST_INV_SQRT) != 0u, x.groupOff, vehicleStep(v) & 1u);
        return;
    }
    i -= gP.totalBeams;
    if (i < gP.totalCands)
    {
        uint pos = gLists[gP.candTable + i];
        uint h = gLists[pos];
        uint v = gStatic[gP.nodeInfoBase + h].w;
        float cell = asfloat(gVt.rec[v * 4u + 3u].w);
        if (cell > 0.0)
        {
            float3 p = gNodes[h * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            uint rank = uint(gAcc[gP.gridCountBase + v * GRID_BUCKETS + b].add(1));
            if (rank < WIDE_SLOTS)
                gBeams[gP.slotBase16 + (v * GRID_BUCKETS + b) * WIDE_SLOTS + rank] = float4(p, asfloat(h));
            else
            {
                uint o = uint(gAcc[gP.overflowBase + v].add(1));
                gNext[gVt.rec[v * 4u + 2u].x + o] = h;
            }
        }
        return;
    }
    i -= gP.totalCands;
    if (i < gP.totalTris)
    {
        // The last step's throttle, and this step's tested list (independent of the
        // beams).
        uint v = gStatic[gP.triBase + i].w;
        Vehicle x = vehicleAt(v);
        if (x.candCount == 0u) return;
        int w = wideStep() > 0u ? triangleFinish(i) : asint(gMisc[gP.vehicles + i].x);
        if (w <= 0)
        {
            uint k = uint(gAcc[gP.testedBase + v * 2u + (vehicleStep(v) & 1u)].add(1));
            gNext[gP.testedListBase + x.triOff + k] = i - x.triOff;
        }
    }
}

// triangleGrid with the offsets in gAcc.
void triangleGridWide(uint gt, uint sub, uint v, float3 origin, float r, float cell, uint candOff, uint candCount,
                      uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    TriGeo t = triangleGeo(gt, origin, r);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub; i < cells; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        for (uint j = 0u; j < i; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        uint n = min(uint(gAcc[gP.gridCountBase + v * GRID_BUCKETS + b].load()), WIDE_SLOTS);
        uint base = gP.slotBase16 + (v * GRID_BUCKETS + b) * WIDE_SLOTS;
        for (uint e = 0u; e < n; e++)
        {
            float4 E = gBeams[base + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
    // Candidates past a full bucket: every triangle tests them (its share of lanes).
    uint over = uint(gAcc[gP.overflowBase + v].load());
    for (uint k = sub; k < over; k += GRID_SUBS)
        tryContact(t, gNext[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
}

// GRID_SUBS lanes per listed triangle (a vehicle's k-th listed triangle on the
// lanes of its k-th triangle record).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_tris(uint3 id : SV_DispatchThreadID)
{
    uint gt = id.x / GRID_SUBS;
    uint sub = id.x % GRID_SUBS;
    if (gt >= gP.totalTris) return;
    uint v = gStatic[gP.triBase + gt].w;
    Vehicle x = vehicleAt(v);
    if (x.candCount == 0u) return;
    uint step = vehicleStep(v);
    uint k = gt - x.triOff;
    if (k >= uint(gAcc[gP.testedBase + v * 2u + (step & 1u)].load())) return;
    uint t = x.triOff + gNext[gP.testedListBase + x.triOff + k];
    float3 origin = vehicleOrigin(v);
    if (x.cell > 0.0)
        triangleGridWide(t, sub, v, origin, x.range, x.cell, x.candOff, x.candCount, x.beamOff, x.groupOff,
                         x.groupCount, step & 1u);
    else
        triangleTest(t, sub, GRID_SUBS, origin, x.range, x.candOff, x.candCount, x.beamOff, x.groupOff,
                     x.groupCount, step & 1u);
}

// After the last step: its triangles' throttle and its group breaks.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_end(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (gP.steps == 0u) return;
    if (i < gP.totalTris)
    {
        uint v = gStatic[gP.triBase + i].w;
        if (gStatic[v * 4u + 2u].y > 0u) triangleFinish(i);
        return;
    }
    i -= gP.totalTris;
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        Vehicle x = vehicleAt(v);
        if (x.groupCount > 0u) groupDrops(i, x.beamOff, x.groupOff, (vehicleStep(v) - 1u) & 1u, true);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_close(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= gP.vehicles) return;
    uint4 m = gMisc[v];
    gMisc[v] = uint4(m.xyz, m.w + wideStep());
}

// --- Vehicle–vehicle contact (engine/docs/explicit-beams-vehicle-contact-spec.md,
// CPU reference engine_softbody::explicit_contact) ---
//
// A frame with vehicle contact on (more than one vehicle):
//   explicit_beams_vv_bounds  one workgroup per vehicle: its box over the node
//                             world positions; its frame bounds (the box grown by
//                             the travel of its fastest node over the steps the plan
//                             covers, twice, plus a margin) and the padded frame box
//                             (+ 0.05 m);
//                             the box into a hashed vehicle grid (or the big list)
//   explicit_beams_vv_pairs   one thread per vehicle: pairs of overlapping padded
//                             frame boxes
//   explicit_beams_vv_group   one workgroup: connected components of the pairs
//                             (label propagation), the solo list, the groups'
//                             member lists, the indirect dispatch arguments
//   (the plan passes run when the last plan's steps are used up: it covers at
//   least VV_PLAN_STEPS on the host, so at 120 fps every other frame)
//   explicit_beams_frame      (indirect) the solo vehicles, a whole frame each; a
//                             node leaving its frame bounds is reported (loud)
//   explicit_beams_contact    (indirect) one workgroup per group: every step, each
//                             member's step in vehicle order, then the contact pass
//                             (owners in vehicle order)
// Two vehicles whose padded frame boxes do not overlap, with every node inside
// its frame bounds, cannot be partners in any step of the frame: the solo
// vehicles need no contact pass. A node outside its bounds is reported to the
// host, which fails (the frame may have missed a contact).

static const int VV_INT_MAX = 2147483647;
static const int VV_INT_MIN = -2147483647 - 1;
static const float VV_BOX_MARGIN = 0.05;
static const uint VV_SLOTS = 8u;
static const uint VV_MAX_SPAN = 3u;
static const uint VV_MAX_CELLS = 27u;
static const float VV_APPROACH_SQ = 16.0;
// Members a contact group may have (one workgroup steps them one after another);
// a larger group's vehicles run alone that frame and the host fails.
static const uint VV_MAX_GROUP = 64u;
static const int VV_MAX_WAIT = 12;

[[vk::binding(10, 0)]] RWStructuredBuffer<uint> gArgs;

groupshared Atomic<int> gsVvBox[12];
groupshared uint gsVvA[256];
groupshared uint gsVvB[256];
groupshared Atomic<uint> gsVvOffered[2];
groupshared Atomic<uint> gsVvListed[2];
groupshared Atomic<uint> gsVvCands[2];
groupshared Atomic<uint> gsVvHits[2];
groupshared uint gsVvReset[2];

// Box coordinates as integers ordered like the floats (atomic minimum, maximum).
int vvKey(float f)
{
    int i = asint(f);
    return i >= 0 ? i : (i ^ 0x7FFFFFFF);
}
float vvUnkey(int k) { return asfloat(k >= 0 ? k : (k ^ 0x7FFFFFFF)); }

uint vvBoxKey(uint v, uint parity) { return gP.vvAcc + (v * 2u + parity) * 6u; }
uint vvCpOff(uint v) { return gStatic[gP.vvStatic + v].x; }
uint vvCpCount(uint v) { return gStatic[gP.vvStatic + v].y; }

bool vvOverlap(float3 alo, float3 ahi, float3 blo, float3 bhi)
{
    return !(any(ahi < blo) || any(bhi < alo));
}

uint vvHash(int3 c)
{
    return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u) & gP.vvGridMask;
}

// The vehicle grid cells of a padded frame box: false when it spans too many (or
// is not finite).
bool vvCells(float3 lo, float3 hi, out int3 c0, out int3 span)
{
    float cell = asfloat(gP.vvCell);
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    c0 = int3(0, 0, 0);
    span = int3(0, 0, 0);
    if (!(all(fs <= float(VV_MAX_SPAN)) && all(fs >= 1.0) && fs.x * fs.y * fs.z <= float(VV_MAX_CELLS)))
        return false;
    c0 = int3(f0);
    span = int3(fs);
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_bounds(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    uint4 r0 = gStatic[v * 4u];
    float3 origin = asfloat(gMisc[v].xyz);
    // Keys: position lo xyz, hi xyz; velocity lo xyz, hi xyz.
    if (lane < 12u) gsVvBox[lane].store((lane % 6u) < 3u ? VV_INT_MAX : VV_INT_MIN);
    sync();
    float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 hi = -lo;
    float3 vlo = float3(0.0, 0.0, 0.0);
    float3 vhi = float3(0.0, 0.0, 0.0);
    bool has = false;
    for (uint i = lane; i < r0.y; i += 256u)
    {
        uint gi = r0.x + i;
        float3 x = nf3(origin + gNodes[gi * 3u].xyz);
        float3 vel = gNodes[gi * 3u + 1u].xyz;
        lo = min(lo, x);
        hi = max(hi, x);
        vlo = min(vlo, vel);
        vhi = max(vhi, vel);
        has = true;
    }
    if (has)
    {
        gsVvBox[0].min(vvKey(lo.x));
        gsVvBox[1].min(vvKey(lo.y));
        gsVvBox[2].min(vvKey(lo.z));
        gsVvBox[3].max(vvKey(hi.x));
        gsVvBox[4].max(vvKey(hi.y));
        gsVvBox[5].max(vvKey(hi.z));
        gsVvBox[6].min(vvKey(vlo.x));
        gsVvBox[7].min(vvKey(vlo.y));
        gsVvBox[8].min(vvKey(vlo.z));
        gsVvBox[9].max(vvKey(vhi.x));
        gsVvBox[10].max(vvKey(vhi.y));
        gsVvBox[11].max(vvKey(vhi.z));
    }
    sync();
    if (lane == 0u)
    {
        float3 blo = float3(vvUnkey(gsVvBox[0].load()), vvUnkey(gsVvBox[1].load()), vvUnkey(gsVvBox[2].load()));
        float3 bhi = float3(vvUnkey(gsVvBox[3].load()), vvUnkey(gsVvBox[4].load()), vvUnkey(gsVvBox[5].load()));
        // Each face moves out by twice the farthest a node moving that way at its
        // velocity covers over the plan, plus the margin (min/max start at 0).
        float3 vl = float3(vvUnkey(gsVvBox[6].load()), vvUnkey(gsVvBox[7].load()), vvUnkey(gsVvBox[8].load()));
        float3 vh = float3(vvUnkey(gsVvBox[9].load()), vvUnkey(gsVvBox[10].load()), vvUnkey(gsVvBox[11].load()));
        float time = 2.0 * float(gP.vvPlanSteps) * STEP;
        float margin = asfloat(gP.vvMargin);
        float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
        // The positions the last frame left, against the bounds that held for it
        // (a solo frame checks each step's positions at the next step's start).
        if ((gP.vvOn & 4u) == 0u
            && (any(blo < asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz)) || any(bhi > asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz))))
            vvReportOutside(v);
        float3 alo = blo - (-vl * time + float3(margin, margin, margin));
        float3 ahi = bhi + (vh * time + float3(margin, margin, margin));
        float3 plo = alo - m3;
        float3 phi = ahi + m3;
        gMisc[vvRec(v, VV_P_LO)] = uint4(asuint(plo), 0u);
        gMisc[vvRec(v, VV_P_HI)] = uint4(asuint(phi), 0u);
        gMisc[vvRec(v, VV_ALLOWED_LO)] = uint4(asuint(alo), 0u);
        gMisc[vvRec(v, VV_ALLOWED_HI)] = uint4(asuint(ahi), 0u);
        int3 c0;
        int3 span;
        // A box that is not finite (a node at infinity or NaN) partners no one
        // here (a GPU deviation: the specification's rule would partner it with
        // every vehicle and group the whole fleet).
        bool finite = all(abs(plo) < 3.0e38) && all(abs(phi) < 3.0e38);
        bool big = finite && !vvCells(plo, phi, c0, span);
        if (finite && !big)
        {
            for (int x = 0; x < span.x; x++)
                for (int y = 0; y < span.y; y++)
                    for (int z = 0; z < span.z; z++)
                    {
                        uint b = vvHash(c0 + int3(x, y, z));
                        uint slot = uint(gAcc[gP.vvGridCount + b].add(1));
                        if (slot < VV_SLOTS) gNext[gP.vvGrid + b * VV_SLOTS + slot] = v;
                        else big = true;
                    }
        }
        if (big)
        {
            uint at = uint(gAcc[vvCtl(VV_BIGS)].add(1));
            gNext[gP.vvBig + at] = v;
        }
        uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
        gMisc[vvRec(v, VV_FLAGS)] = uint4(finite ? (big ? 1u : 0u) : 2u, 0u, 0u, fl.w);
    }
}

// The frame's dispatch arguments: with no group the solo frame over the vehicles
// (words 0..2), else the frame with the groups over vehicles + groups (words 4..6).
void vvArgs(uint nv, uint groups)
{
    gArgs[0] = groups == 0u ? nv : 0u;
    gArgs[1] = 1u;
    gArgs[2] = 1u;
    gArgs[4] = groups == 0u ? 0u : nv + groups;
    gArgs[5] = 1u;
    gArgs[6] = 1u;
}

void vvEmitPair(uint a, uint b)
{
    uint at = uint(gAcc[vvCtl(VV_PAIRS)].add(1));
    if (at < gP.vvPairCap)
    {
        gNext[gP.vvPairs + 2u * at] = a;
        gNext[gP.vvPairs + 2u * at + 1u] = b;
    }
    else
    {
        gAcc[vvCtl(VV_OVERFLOW)].store(1);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_pairs(uint3 id : SV_DispatchThreadID)
{
    if (id.x < gP.vehicles) vvFindPairs(id.x);
}

// The overlapping pairs of vehicle v's padded frame box (each pair once).
void vvFindPairs(uint v)
{
    float3 plo = asfloat(gMisc[vvRec(v, VV_P_LO)].xyz);
    float3 phi = asfloat(gMisc[vvRec(v, VV_P_HI)].xyz);
    uint kind = gMisc[vvRec(v, VV_FLAGS)].x;
    if (kind == 2u) return;
    if (kind == 1u)
    {
        // Against every vehicle (two big ones once).
        for (uint w = 0u; w < gP.vehicles; w++)
        {
            if (w == v) continue;
            uint kw = gMisc[vvRec(w, VV_FLAGS)].x;
            if (kw == 2u || (kw == 1u && w < v)) continue;
            if (vvOverlap(plo, phi, asfloat(gMisc[vvRec(w, VV_P_LO)].xyz), asfloat(gMisc[vvRec(w, VV_P_HI)].xyz)))
                vvEmitPair(v, w);
        }
        return;
    }
    int3 c0;
    int3 span;
    vvCells(plo, phi, c0, span);
    for (int x = 0; x < span.x; x++)
        for (int y = 0; y < span.y; y++)
            for (int z = 0; z < span.z; z++)
            {
                int3 c = c0 + int3(x, y, z);
                uint b = vvHash(c);
                uint n = min(uint(gAcc[gP.vvGridCount + b].load()), VV_SLOTS);
                for (uint e = 0u; e < n; e++)
                {
                    uint w = gNext[gP.vvGrid + b * VV_SLOTS + e];
                    // Each pair once: the higher index in the lower's thread, in
                    // the first cell both cover; big vehicles find their own.
                    if (w <= v || gMisc[vvRec(w, VV_FLAGS)].x != 0u) continue;
                    float3 wlo = asfloat(gMisc[vvRec(w, VV_P_LO)].xyz);
                    float3 whi = asfloat(gMisc[vvRec(w, VV_P_HI)].xyz);
                    if (!vvOverlap(plo, phi, wlo, whi)) continue;
                    int3 cw;
                    int3 sw;
                    vvCells(wlo, whi, cw, sw);
                    if (all(c == max(c0, cw))) vvEmitPair(v, w);
                }
            }
}

uint vvLabel(uint v) { return uint(gAcc[gP.vvLabel + v].load()); }
uint vvSum(uint g, uint k) { return gP.vvLabel + gP.vehicles + g * 3u + k; }

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_group(uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint nv = gP.vehicles;
    if (gP.vvFusedPairs != 0u)
    {
        for (uint v = lane; v < nv; v += 256u) vvFindPairs(v);
        sync();
    }
    uint pairs = min(uint(gAcc[vvCtl(VV_PAIRS)].load()), gP.vvPairCap);
    // No overlapping frame boxes (the common case): every vehicle is solo.
    if (lane == 0u) gsUniformLanes = pairs;
    sync();
    uint anyPairs = gsUniformLanes;
    sync();
    if (anyPairs == 0u)
    {
        for (uint b = lane; b <= gP.vvGridMask; b += 256u) gAcc[gP.vvGridCount + b].store(0);
        if (lane == 0u)
        {
            vvArgs(nv, 0u);
            gAcc[vvCtl(VV_BIGS)].store(0);
        }
        return;
    }
    for (uint v = lane; v < nv; v += 256u)
    {
        gAcc[gP.vvLabel + v].store(int(v));
        gMisc[vvRec(v, VV_FLAGS)].y = 0u;
    }
    sync();
    for (uint p = lane; p < pairs; p += 256u)
    {
        gMisc[vvRec(gNext[gP.vvPairs + 2u * p], VV_FLAGS)].y = 1u;
        gMisc[vvRec(gNext[gP.vvPairs + 2u * p + 1u], VV_FLAGS)].y = 1u;
    }
    // Connected components: every vehicle of a group ends with its smallest index.
    uint go = 1u;
    for (uint it = 0u; it < 64u; it++)
    {
        if (lane == 0u) gsUniformLanes = 0u;
        sync();
        bool changed = false;
        for (uint p = lane; p < pairs; p += 256u)
        {
            uint a = gNext[gP.vvPairs + 2u * p];
            uint b = gNext[gP.vvPairs + 2u * p + 1u];
            uint la = vvLabel(a);
            uint lb = vvLabel(b);
            if (la != lb)
            {
                int m = int(min(la, lb));
                gAcc[gP.vvLabel + a].min(m);
                gAcc[gP.vvLabel + b].min(m);
                gAcc[gP.vvLabel + la].min(m);
                gAcc[gP.vvLabel + lb].min(m);
                changed = true;
            }
        }
        for (uint v = lane; v < nv; v += 256u)
        {
            uint l = vvLabel(v);
            uint ll = vvLabel(l);
            if (ll < l) gAcc[gP.vvLabel + v].min(int(ll));
        }
        if (changed) gsUniformLanes = 1u;
        sync();
        go = gsUniformLanes;
        sync();
        if (go == 0u) break;
    }
    if (lane == 0u && go != 0u) gAcc[vvCtl(VV_UNCONVERGED)].store(1);
    // Groups (numbered by their smallest vehicle, in no particular order) and the
    // solo list.
    for (uint v = lane; v < nv; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u && vvLabel(v) == v)
            gMisc[vvRec(v, VV_FLAGS)].z = uint(gAcc[vvCtl(VV_GROUPS)].add(1));
    }
    sync();
    for (uint v = lane; v < nv; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u)
        {
            uint g = gMisc[vvRec(vvLabel(v), VV_FLAGS)].z;
            gMisc[vvRec(v, VV_FLAGS)].z = g;
            gAcc[vvSum(g, 0u)].add(1);
            gAcc[vvSum(g, 2u)].add(int(vvCpCount(v)));
        }

    }
    sync();
    // Offsets of the groups' members and contact point runs.
    uint groups = uint(gAcc[vvCtl(VV_GROUPS)].load());
    uint chunk = (groups + 255u) / 256u;
    uint g0 = min(lane * chunk, groups);
    uint g1 = min(g0 + chunk, groups);
    uint sm = 0u;
    uint sp = 0u;
    for (uint g = g0; g < g1; g++)
    {
        sm += uint(gAcc[vvSum(g, 0u)].load());
        sp += uint(gAcc[vvSum(g, 2u)].load());
    }
    gsVvA[lane] = sm;
    gsVvB[lane] = sp;
    sync();
    if (lane == 0u)
    {
        uint ra = 0u;
        uint rb = 0u;
        for (uint k = 0u; k < 256u; k++)
        {
            uint a = gsVvA[k];
            uint b = gsVvB[k];
            gsVvA[k] = ra;
            gsVvB[k] = rb;
            ra += a;
            rb += b;
        }
    }
    sync();
    uint ra = gsVvA[lane];
    uint rb = gsVvB[lane];
    for (uint g = g0; g < g1; g++)
    {
        uint m = uint(gAcc[vvSum(g, 0u)].load());
        uint pts = uint(gAcc[vvSum(g, 2u)].load());
        // Too many members: none stepped as a group (they run alone; reported).
        gMisc[gP.vvGroups + g] = m <= VV_MAX_GROUP ? uint4(ra, m, rb, pts) : uint4(ra, 0u, rb, 0u);
        ra += m;
        rb += pts;
    }
    sync();
    for (uint v = lane; v < nv; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u)
        {
            uint g = gMisc[vvRec(v, VV_FLAGS)].z;
            if (gMisc[gP.vvGroups + g].y == 0u)
            {
                // Stepped alone (by the solo frame) this frame.
                gMisc[vvRec(v, VV_FLAGS)].y = 0u;
                gAcc[vvCtl(VV_TOO_BIG)].store(1);
                continue;
            }
            uint slot = uint(gAcc[vvSum(g, 1u)].add(1));
            gNext[gP.vvMembers + gMisc[gP.vvGroups + g].x + slot] = v;
        }
    }
    sync();
    if (lane == 0u)
    {
        vvArgs(nv, groups);
    }
    // Clear for the next frame.
    for (uint b = lane; b <= gP.vvGridMask; b += 256u) gAcc[gP.vvGridCount + b].store(0);
    for (uint g = lane; g < groups; g += 256u)
    {
        gAcc[vvSum(g, 0u)].store(0);
        gAcc[vvSum(g, 1u)].store(0);
        gAcc[vvSum(g, 2u)].store(0);
    }
    sync();
    if (lane == 0u)
    {
        gAcc[vvCtl(VV_PAIRS)].store(0);
        gAcc[vvCtl(VV_BIGS)].store(0);
        gAcc[vvCtl(VV_SOLOS)].store(0);
        gAcc[vvCtl(VV_GROUPS)].store(0);
    }
}

// --- The contact groups: explicit_beams_contact ---

// Group member m (sorted by vehicle index) and its partner / reset flags, its
// contact points' first run index; per group at its member offset in regions of
// one word per vehicle.
uint vvMember(uint off, uint m) { return gNext[gP.vvMembers + gP.vehicles + off + m]; }
uint vvPrefixAt(uint off, uint m) { return gP.vvMembers + 2u * gP.vehicles + off + m; }
uint vvPartnerAt(uint off, uint m) { return gP.vvMembers + 3u * gP.vehicles + off + m; }
uint vvResetAt(uint off, uint m) { return gP.vvMembers + 4u * gP.vehicles + off + m; }
uint vvTestedAt(uint i) { return gP.vvMembers + 5u * gP.vehicles + i; }

// A member's step: explicit_beams_frame's step for vehicle v (every barrier at the
// top level), with the self-collision accumulators always read (the contact pass
// adds into them), the tested-triangle list restarted per member, and a pending
// reset of the self-collision throttles (a fast approach in the last contact pass)
// applied after their update.
void contactMemberStep(uint v, uint lane, uint s, bool first)
{
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = r0.y;
    uint beamOff = r0.z;
    uint beamCount = r0.w;
    uint triOff = r1.x;
    uint triCount = r1.y;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = r2.y;
    uint groupOff = r2.z;
    uint groupCount = r2.w;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
#endif
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint step = misc.w + s;
    uint parity = step & 1u;
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    if (lane == 0u)
    {
        gsTestedCount[parity].store(0u);
        if (first) vvClearCounters();
    }
    sync();
    {
        // The contact pass's box of this step (world positions after integration)
        // into this parity's keys (the other parity's cleared for the next step),
        // and the frame bounds check.
        float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 hi = -lo;
        bool has = false;
        for (uint i = lane; i < nodeCount; i += LANES)
        {
            nodeUpdate(nodeOff + i, gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground,
                       boxOff, boxCount, beamOff, groupOff, groupCount, true);
            float3 x = nf3(origin + gNodes[(nodeOff + i) * 3u].xyz);
            lo = min(lo, x);
            hi = max(hi, x);
            has = true;
        }
        if (lane < 6u) gAcc[vvBoxKey(v, parity ^ 1u) + lane].store(lane < 3u ? VV_INT_MAX : VV_INT_MIN);
        if (has)
        {
            uint key = vvBoxKey(v, parity);
            gAcc[key].min(vvKey(lo.x));
            gAcc[key + 1u].min(vvKey(lo.y));
            gAcc[key + 2u].min(vvKey(lo.z));
            gAcc[key + 3u].max(vvKey(hi.x));
            gAcc[key + 4u].max(vvKey(hi.y));
            gAcc[key + 5u].max(vvKey(hi.z));
            if (any(lo < asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz)) || any(hi > asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz)))
                vvReportOutside(v);
        }
    }
    for (uint g = lane; g < groupCount; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = lane; t < triCount; t += LANES)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (s > 0u && pending)
            {
                w = 0;
                gMisc[gP.vehicles + triOff + t].x = 0u;
            }
            if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
        }
    }
    for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
    sync();
    if (lane == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    if (lane == 0u)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        gsUniformLanes = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u;
    }
    if (cell > 0.0)
    {
        for (uint c = lane; c < candCount; c += LANES)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = lane; w < aero0.y; w += LANES) wingForces(aero0.x + w, origin);
    if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = lane; c < aero0.w; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = lane; c < aero0.w; c += LANES) commandUpdate(aero0.z + c);
    }
    sync();
    if (lane == 0u) wingApply(aero0.x, aero0.y);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
    }
#endif
    for (uint j = lane; j < beamCount; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    sync();
    uint gridOn = gsUniformLanes;
    if (gridOn != 0u)
    {
        uint per = GRID_BUCKETS / LANES;
        uint block = LANES / GRID_BLOCKS;
        uint sum = 0u;
        for (uint k = 0u; k < per; k++) sum += gsCount[lane * per + k].load();
        gsPart[lane] = sum;
        sync();
        if (lane < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint k = 0u; k < block; k++)
            {
                uint x = gsPart[lane * block + k];
                gsPart[lane * block + k] = run;
                run += x;
            }
            gsBlk[lane] = run;
        }
        sync();
        if (lane == 0u)
        {
            uint run = 0u;
            for (uint k = 0u; k < GRID_BLOCKS; k++)
            {
                uint x = gsBlk[k];
                gsBlk[k] = run;
                run += x;
            }
            gsOff[GRID_BUCKETS] = run;
        }
        sync();
        uint run = gsPart[lane] + gsBlk[lane / block];
        for (uint k = 0u; k < per; k++)
        {
            gsOff[lane * per + k] = run;
            run += gsCount[lane * per + k].load();
        }
        sync();
        if (cell > 0.0)
        {
            for (uint c = lane; c < candCount; c += LANES)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gsOff[br & (GRID_BUCKETS - 1u)] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
    if (candCount > 0u && cell > 0.0 && gridOn != 0u)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        for (uint p = lane; p < listed * GRID_SUBS; p += LANES)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
            triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
        }
    }
    else if (candCount > 0u && cell > 0.0)
    {
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        uint lanes = clamp(LANES / max(listed, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        for (uint p = lane; p < listed * lanes && lane < stride; p += stride)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
            triangleGridBrute(triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, beamOff,
                              groupOff, groupCount, parity);
        }
    }
    else if (candCount > 0u)
    {
        uint lanes = clamp(LANES / max(triCount, 1u), 1u, candCount);
        uint stride = (LANES / lanes) * lanes;
        for (uint p = lane; p < triCount * lanes && lane < stride; p += stride)
            triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
    }
#ifdef EB_DET_HITS
    sync();
    if (lane == 0u) detApply(v, origin, range, beamOff, groupOff, groupCount, parity);
#endif
#ifdef EB_AERO
    if (lane == LANES - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, origin, groupOff, parity);
#endif
    sync();
}

// A member's frame start: the wing check and the origin rebase.
void contactMemberBegin(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    {
        float3 o = asfloat(gMisc[v].xyz);
        for (uint w = lane; w < aero0.y; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif
    float3 first = gNodes[r0.x * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane; i < r0.y; i += LANES)
        {
            float4 P = gNodes[(r0.x + i) * 3u];
            gNodes[(r0.x + i) * 3u] = float4(P.xyz - first, P.w);
        }
        if (lane == 0u)
        {
            uint4 m = gMisc[v];
            gMisc[v] = uint4(asuint(asfloat(m.xyz) + first), m.w);
        }
    }
    sync();
}

// A member's frame end: the last step's throttle update (and a pending reset), the
// last step's group breaks, the steps run; its box keys cleared for a later frame.
void contactMemberEnd(uint v, uint lane)
{
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 misc = gMisc[v];
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    if (gP.steps > 0u && r2.y > 0u)
    {
        for (uint t = lane; t < r1.y; t += LANES)
        {
            triangleFinish(r1.x + t);
            if (pending) gMisc[gP.vehicles + r1.x + t].x = 0u;
        }
    }
    if (gP.steps > 0u && r2.w > 0u)
    {
        uint last = (misc.w + gP.steps - 1u) & 1u;
        for (uint j = lane; j < r0.w; j += LANES) groupDrops(r0.z + j, r0.z, r2.z, last, true);
    }
    if (lane < 12u) gAcc[gP.vvAcc + v * 12u + lane].store((lane % 6u) < 3u ? VV_INT_MAX : VV_INT_MIN);
    sync();
    if (lane == 0u)
    {
        gMisc[v] = uint4(misc.xyz, misc.w + gP.steps);
        gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    }
}

// A collision triangle as the contact pass sees it (spec §5).
struct VvTri
{
    uint4 tn;
    float3 p0;
    float3 n;
    float3 ra;
    float3 rb;
    float3 rn;
    bool solvable;
    float3 lo;
    float3 hi;
};

VvTri vvTriangle(uint gt, float3 origin, float r)
{
    VvTri t;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    float3 u = ssub3(p1, t.p0);
    float3 w = ssub3(p2, t.p0);
    float3 cr = scross(u, w);
    float cl = ssqrt(sdot(cr, cr));
    t.n = cl > 1e-8 ? sscale(cr, sdiv(1.0, cl)) : cr;
    float det = sdot(u, scross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? sdiv(1.0, det) : 0.0;
    t.ra = sscale(scross(w, t.n), rd);
    t.rb = sscale(scross(t.n, u), rd);
    t.rn = sscale(scross(u, w), rd);
    float3 r3 = float3(r, r, r);
    t.lo = ssub3(min(min(t.p0, p1), p2), r3);
    t.hi = sadd3(max(max(t.p0, p1), p2), r3);
    return t;
}

// Candidate h (global node, world position p, of vehicle vh) against triangle gt of
// the owner: the hit test, the side vote, the contact force into the accumulators
// (h, and the triangle's nodes by their weights). Returns whether it hit.
bool vvTry(VvTri t, uint gt, uint h, float3 p, uint vh, float r, uint parity)
{
    float3 q = ssub3(p, t.p0);
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return false;
    // The side: a vote of h and its spawn-time beam neighbours (its incidence
    // entries' far ends, in beam order).
    uint4 info = gStatic[gP.nodeInfoBase + h];
    int score = delta >= 0.0 ? 3 : -3;
    if (info.z > 3u)
    {
        float3 ob = asfloat(gMisc[vh].xyz);
        uint base = gP.vvNbr + info.y;
        // Four independent loads at a time (the far ends are a contiguous list).
        uint k = 0u;
        for (; k + 4u <= info.z; k += 4u)
        {
            float3 y0 = gNodes[gLists[base + k] * 3u].xyz;
            float3 y1 = gNodes[gLists[base + k + 1u] * 3u].xyz;
            float3 y2 = gNodes[gLists[base + k + 2u] * 3u].xyz;
            float3 y3 = gNodes[gLists[base + k + 3u] * 3u].xyz;
            score += sdot(t.n, ssub3(nf3(ob + y0), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y1), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y2), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y3), t.p0)) >= 0.0 ? 1 : -1;
        }
        for (; k < info.z; k++)
            score += sdot(t.n, ssub3(nf3(ob + gNodes[gLists[base + k] * 3u].xyz), t.p0)) >= 0.0 ? 1 : -1;
    }
    float3 nn = t.n;
    if (score < 0)
    {
        nn = -t.n;
        delta = -delta;
    }
    float pen = nf(r - delta);
    float4 P = gNodes[h * 3u];
    float4 V = gNodes[h * 3u + 1u];
    float4 P0 = gNodes[t.tn.x * 3u];
    float4 P1 = gNodes[t.tn.y * 3u];
    float4 P2 = gNodes[t.tn.z * 3u];
    float3 vt = sadd3(sadd3(sscale(gNodes[t.tn.y * 3u + 1u].xyz, alpha), sscale(gNodes[t.tn.z * 3u + 1u].xyz, beta)),
                      sscale(gNodes[t.tn.x * 3u + 1u].xyz, gamma));
    float3 v = ssub3(V.xyz, vt);
    float mt = nf(nf(smul(P1.w, alpha) + smul(P2.w, beta)) + smul(P0.w, gamma));
    float m = sdiv(smul(P.w, mt), nf(P.w + mt));
    uint4 rb0 = gStatic[vh * 4u];
    uint4 rb2 = gStatic[vh * 4u + 2u];
    float3 fh = sadd3(beamSum(gNodes[h * 3u + 2u].xyz, info, rb0.z, rb2.z, rb2.w, parity, false), accForce(h));
#ifdef EB_AERO
    uint slot1 = info.x >> NODE_SLOT_SHIFT;
    if (slot1 != 0u) fh = sadd3(fh, gBeams[gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    float3 f = contactForce(fh, v, m, nn, boxSurface(), pen, V.w);
    accAdd(h, f);
    accAdd(t.tn.y, -sscale(f, alpha));
    accAdd(t.tn.z, -sscale(f, beta));
    accAdd(t.tn.x, -sscale(f, gamma));
    return true;
}

// The owner's tested triangle t (list position p of GRID_SUBS lanes, lane `sub`)
// against the owner's candidates (partner contact points inside the owner's box
// grown by its range, compacted at the run offset): the box test, then the hit
// test, vote and force; marks the triangle's record (z: had candidates, w: hit).
// The hit test of vvTry alone.
bool vvInBand(VvTri t, float3 p, float r)
{
    float3 q = ssub3(p, t.p0);
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    float gamma = nf(nf(1.0 - alpha) - beta);
    return alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r;
}

// Hit list entry `i` of the group's list of this owner (q its counter parity).
uint vvHitAt(uint runOff, uint i) { return gP.vvHits + 2u * (2u * runOff + i); }

void vvTriangleCands(uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r, uint parity,
                     uint q)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    bool cand = false;
    bool hit = false;
    for (uint e = sub; e < cands; e += subs)
    {
        float4 E = gBeams[gP.vvRun + runOff + e];
        if (any(E.xyz < t.lo) || any(E.xyz > t.hi)) continue;
        uint h = asuint(E.w);
        cand = true;
        if (!vvInBand(t, E.xyz, r)) continue;
        hit = true;
        // The force later, one lane per hit (the vote and the force read many
        // nodes); a full list applies it here.
        uint i = gsVvHits[q].add(1u);
        if (i < 2u * points)
        {
            gNext[vvHitAt(runOff, i)] = gt;
            gNext[vvHitAt(runOff, i) + 1u] = h;
        }
        else
        {
            vvTry(t, gt, h, E.xyz, gStatic[gP.nodeInfoBase + h].w, r, parity);
        }
    }
    if (cand) gMisc[gP.vvTri + gt].z = 1u;
    if (hit) gMisc[gP.vvTri + gt].w = 1u;
}

// The listed hits of owner va: one lane each, the vote and the force.
void vvApplyHits(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = lane; i < min(hits, 2u * points); i += LANES)
    {
        uint gt = gNext[vvHitAt(runOff, i)];
        uint h = gNext[vvHitAt(runOff, i) + 1u];
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        VvTri t = vvTriangle(gt, oa, r);
        vvTry(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity);
    }
}

#ifdef EB_DET_HITS
// Tests only: vvApplyHits on one lane, the hits sorted by (triangle, node), so the
// order is fixed and each hit sees every earlier one.
void vvApplyHitsDet(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    if (lane != 0u) return;
    uint n = min(hits, 2u * points);
    for (uint i = 1u; i < n; i++)
    {
        uint gt = gNext[vvHitAt(runOff, i)];
        uint h = gNext[vvHitAt(runOff, i) + 1u];
        uint j = i;
        while (j > 0u)
        {
            uint pg = gNext[vvHitAt(runOff, j - 1u)];
            uint ph = gNext[vvHitAt(runOff, j - 1u) + 1u];
            if (pg < gt || (pg == gt && ph <= h)) break;
            gNext[vvHitAt(runOff, j)] = pg;
            gNext[vvHitAt(runOff, j) + 1u] = ph;
            j--;
        }
        gNext[vvHitAt(runOff, j)] = gt;
        gNext[vvHitAt(runOff, j) + 1u] = h;
    }
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = 0u; i < n; i++)
    {
        uint gt = gNext[vvHitAt(runOff, i)];
        uint h = gNext[vvHitAt(runOff, i) + 1u];
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        VvTri t = vvTriangle(gt, oa, r);
        vvTry(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity);
    }
}
#endif

// A tested triangle's throttle after its tests (a hit → 0, no candidate → + 1).
void vvThrottleEnd(uint gt)
{
    uint4 c = gMisc[gP.vvTri + gt];
    int cwait = asint(c.x);
    if (c.w != 0u) cwait = 0;
    else if (c.z == 0u) cwait = cwait + 1;
    gMisc[gP.vvTri + gt] = uint4(asuint(cwait), c.y, 0u, 0u);
}

// Triangle gt of owner va: its box grown by the range overlaps the box `lo..hi`
// (the partners' boxes' union), so it may have candidates.
bool vvNearBox(uint gt, uint va, float r, float3 blo, float3 bhi)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 o = asfloat(gMisc[va].xyz);
    float3 p0 = nf3(o + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(o + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(o + gNodes[tn.z * 3u].xyz);
    float3 r3 = float3(r, r, r);
    return vvOverlap(ssub3(min(min(p0, p1), p2), r3), sadd3(max(max(p0, p1), p2), r3), blo, bhi);
}

void vvClearCounters()
{
    for (uint q = 0u; q < 2u; q++)
    {
        gsVvOffered[q].store(0u);
        gsVvListed[q].store(0u);
        gsVvCands[q].store(0u);
        gsVvHits[q].store(0u);
        gsVvReset[q] = 0u;
    }
}

// The contact pass after every member's step (their boxes already in the step's
// keys): each owner in vehicle order, three barriers each. Phase S: the partner
// scan over the other members (and the previous owner's throttle updates); phase
// P: contact throttle resets, the owner's throttle step listing its tested
// triangles, and its candidates compacted; phase T: tested triangles against the
// candidates.
void vvPass(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint prevTri = 0u;
    uint prevListed = 0u;
    uint prevVa = 0u;
    float prevRange = 0.0;
    for (uint a = 0u; a < k; a++)
    {
        uint q = a & 1u;
        uint va = vvMember(off, a);
        uint key = vvBoxKey(va, parity);
        float3 blo = float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load()));
        float3 bhi = float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load()));
        float3 alo = ssub3(blo, m3);
        float3 ahi = sadd3(bhi, m3);
        float3 va0 = gNodes[gStatic[va * 4u].x * 3u + 1u].xyz;
        // Phase S (with the previous owner's hits).
        for (uint i = lane; i < prevListed; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_DET_HITS
        if (a > 0u) vvApplyHitsDet(prevVa, runOff, points, gsVvHits[q ^ 1u].load(), prevRange, parity, lane);
#else
        if (a > 0u) vvApplyHits(prevVa, runOff, points, gsVvHits[q ^ 1u].load(), prevRange, parity, lane);
#endif
        for (uint b = lane; b < k; b += LANES)
        {
            uint vb = vvMember(off, b);
            uint kb = vvBoxKey(vb, parity);
            float3 olo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
            float3 ohi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
            bool partner = b != a && vvOverlap(alo, ahi, olo, ohi);
            bool reset = false;
            if (partner)
            {
                gsVvOffered[q].add(vvCpCount(vb));
                float3 d = ssub3(va0, gNodes[gStatic[vb * 4u].x * 3u + 1u].xyz);
                if (sdot(d, d) > VV_APPROACH_SQ)
                {
                    reset = true;
                    gsVvReset[q] = 1u;
                    gMisc[vvRec(vb, VV_FLAGS)].w = 1u;
                    gMisc[vvRec(va, VV_FLAGS)].w = 1u;
                }
            }
            gNext[vvPartnerAt(off, b)] = partner ? 1u : 0u;
            gNext[vvResetAt(off, b)] = reset ? 1u : 0u;
        }
        sync();
        // Phase P.
        uint offered = gsVvOffered[q].load();
        bool ownerReset = gsVvReset[q] != 0u;
        if (lane == 0u)
        {
            gsVvOffered[q ^ 1u].store(0u);
            gsVvListed[q ^ 1u].store(0u);
            gsVvCands[q ^ 1u].store(0u);
            gsVvHits[q ^ 1u].store(0u);
            gsVvReset[q ^ 1u] = 0u;
        }
        for (uint m = 0u; m < k; m++)
        {
            if (m == a || gNext[vvResetAt(off, m)] == 0u) continue;
            uint4 r1 = gStatic[vvMember(off, m) * 4u + 1u];
            for (uint t = lane; t < r1.y; t += LANES) gMisc[gP.vvTri + r1.x + t].x = 0u;
        }
        uint4 ra1 = gStatic[va * 4u + 1u];
        float range = asfloat(gStatic[va * 4u + 3u].z);
        // The partners' boxes' union (their contact points all lie in it).
        float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 phi = -plo;
        for (uint m = 0u; m < k; m++)
        {
            if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
            uint kb = vvBoxKey(vvMember(off, m), parity);
            plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
            phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
        }
        for (uint t = lane; t < ra1.y; t += LANES)
        {
            uint4 c = gMisc[gP.vvTri + ra1.x + t];
            int cwait = ownerReset ? 0 : asint(c.x);
            int cidle = asint(c.y);
            if (offered == 0u)
            {
                if (ownerReset) gMisc[gP.vvTri + ra1.x + t].x = 0u;
                continue;
            }
            if (cwait > 0)
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait - 1), asuint(cidle + 1), 0u, 0u);
                continue;
            }
            cwait = VV_MAX_WAIT < cidle ? VV_MAX_WAIT : cidle;
            // A triangle whose box (grown by the range) meets no partner's box has
            // no candidate: its throttle update at once, no tests.
            if (!vvNearBox(ra1.x + t, va, range, plo, phi))
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait + 1), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvListed[q].add(1u))] = t;
        }
        if (offered > 0u)
        {
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(blo, r3);
            float3 chi = sadd3(bhi, r3);
            for (uint m = 0u; m < k; m++)
            {
                if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                for (uint i = lane; i < cp.y; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    if (any(x < clo) || any(x > chi)) continue;
                    gBeams[gP.vvRun + runOff + gsVvCands[q].add(1u)] = float4(x, asfloat(h));
                }
            }
        }
        sync();
        // Phase T.
        uint listed = gsVvListed[q].load();
        uint cands = gsVvCands[q].load();
        // Lanes per triangle: all the lanes over the listed triangles.
        uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
        for (uint p = lane; p < listed * subs; p += LANES)
        {
            uint t = gNext[vvTestedAt(ra1.x + p / subs)];
            vvTriangleCands(va, ra1.x + t, p % subs, subs, runOff, points, cands, range, parity, q);
        }
        sync();
        prevTri = ra1.x;
        prevListed = listed;
        prevVa = va;
        prevRange = range;
    }
    for (uint i = lane; i < prevListed; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_DET_HITS
    if (k > 0u) vvApplyHitsDet(prevVa, runOff, points, gsVvHits[(k - 1u) & 1u].load(), prevRange, parity, lane);
#else
    if (k > 0u) vvApplyHits(prevVa, runOff, points, gsVvHits[(k - 1u) & 1u].load(), prevRange, parity, lane);
#endif
    // (The counters are cleared by the next step's first member step, before its
    // first barrier.)
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_contact(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    contactGroup(gid.x, lid.x);
}

// Contact group g: its members stepped together with the contact pass.
void contactGroup(uint g, uint lane)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    if (lane == 0u) gsUniformLanes = G.y;
    sync();
    uint k = gsUniformLanes;
    sync();
    // Members in vehicle order (a rank sort), their run prefixes and member
    // indices.
    for (uint i = lane; i < k; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u; j < k; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u; m < k; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
        vvClearCounters();
    }
    sync();
    for (uint m = 0u; m < k; m++) contactMemberBegin(vvMember(off, m), lane);
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    for (uint s = 0u; s < gP.steps; s++)
    {
        for (uint m = 0u; m < k; m++) contactMemberStep(vvMember(off, m), lane, s, m == 0u);
        vvPass(k, off, runOff, points, stepBase + s, lane);
    }
    for (uint m = 0u; m < k; m++) contactMemberEnd(vvMember(off, m), lane);
}
