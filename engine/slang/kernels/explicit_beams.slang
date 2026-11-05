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
// JBeam mode (engine/docs/jbeam-semantics-spec.md; jbeamUpdate).
static const uint KIND_JPLAIN = 6u;
static const uint KIND_JSUPPORT = 7u;
static const uint KIND_JBOUNDED = 8u;
static const uint KIND_JCOUPLER = 9u;
static const uint KIND_JTORSION = 10u; // nodes 1-2; the next record carries nodes 3-4
static const uint KIND_JTORSION2 = 11u;
// JBeam Part II (engine/docs/jbeam-semantics-spec.md §16-§28): kinds and the type
// byte's extension flag; such beams take jbeamExtUpdate (explicit_jbeam_ext.rs).
static const uint KIND_JANISO = 12u;
static const uint KIND_JPRESSURED = 13u;
static const uint KIND_JLBEAM = 14u;
static const uint KIND_JPGROUP = 15u;
static const uint KIND_JPTRI = 16u;
static const uint KIND_JPTRI2 = 17u;
static const uint TRI_EXCL_CAP = 16u; // explicit_jbeam_ext.rs TRI_EXCL_CAP
static const uint KIND_JATRI = 18u; // a triangle's aerodynamics (spec §31), nodes x0-x1
static const uint KIND_JATRI2 = 19u; // its x2-x2 record
static const uint KIND_JLBEAM2 = 20u; // an LBEAM's x3-x3 record (the vertex's force)
static const uint TYPE_EXT = 4u;
static const uint EXT_GROUPS = 1u;
static const uint EXT_DEFORM = 2u;
static const uint EXT_REBOUND = 4u;
static const uint EXT_RAMP = 8u;
static const uint NODE_FRICTION_LIMIT = 32u;
static const uint NODE_TREAD = 64u; // a pressure wheel's tread node (spec §34, treadGround)
static const uint TRI_BREAKABLE = 0x80000000u;
static const float ATMOSPHERE = 101325.0;
static const uint TYPE_DEFORMABLE = 0u;

static const uint FLAG_FAST_INV_SQRT = 1u;
static const uint FLAG_DRAG = 2u;
static const uint FLAG_TURBULENCE = 4u;
// The vehicle has static shapes other than axis-aligned boxes (nodeUpdateShapes).
static const uint FLAG_SHAPES = 64u;
// Vehicle contact's two-body rule (jbeam-semantics-spec.md §10.3): a hit where this
// vehicle is the owner or the partner splits masses and cancels a share.
static const uint FLAG_TWO_BODY = 128u;
static const uint FLAG_CFORCE = 256u; // the contact-force log (jbeam-semantics-spec.md §32)
// The vehicle has tread nodes (NODE_TREAD): its nodes take the shapes path, whose
// ground contact runs treadGround for them (jbeam-semantics-spec.md §34).
static const uint FLAG_TREAD = 512u;
// The BeamNG profile without Rigs of Rods' idle-triangle throttle
// (jbeam-semantics-spec.md §36): the vehicle's triangles never wait.
static const uint FLAG_NO_THROTTLE = 1024u;
// Anti-clip (jbeam-semantics-spec.md §36): the vehicle's self-collision and its
// vehicle contact (as owner or partner) keep each node on the side it came from.
static const uint FLAG_ANTI_CLIP = 2048u;
// Some vehicle of the upload has anti-clip: every vehicle's node phase rolls its
// nodes' side memory and takes its fastest node's speed (the shapes path).
static const uint FLAG_CLIP_MEM = 4096u;
// A collision triangle that is anti-clip only (BeamNG NONCOLLIDABLE; gMisc .z).
static const uint TRI_ANTI_ONLY = 0x40000000u;
// The BeamNG profile's plasticity (jbeam-semantics-spec.md §36): a beam's strength
// does not fall with its plastic stretch (RoR's rule; `plastic_strength_drop` off).
static const uint FLAG_KEEP_STRENGTH = 8192u;
// The BeamNG profile's friction law with exact exponentials (§36, exact_exp/pow).
static const uint FLAG_EXACT_FRICTION = 16384u;

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
    uint vvTri; // gMisc: per triangle (cwait, cidle, 0, 0); its marks in gNext (vvMarkAt)
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
    // JBeam Part II (jbeamExtUpdate): gLists per beam its extension + 1 (0 none);
    // gStatic per extension 4 records; gBeams per extension 1 state record;
    // gStatic per triangle its 4 break beams; gGroups the deform group words.
    uint extIndex;
    uint extBase;
    uint extState;
    uint triBreak;
    uint deformBase;
    uint triExcl; // gStatic: per triangle (start in gLists, count) of self-collision nodes it skips (0: none)
    uint loopCapWord; // gAcc: the loop-cap flag (see EB_LOOP_CAP)
    uint vvSplit; // gAcc: two-body mass splitting counts, per node: its triangle hits, then VV_PART slots of its point hits (0: no two-body vehicle)
    uint vvSnap; // gBeams: per node its force at the contact pass's start (two-body contact)
    uint cfAcc; // gAcc: the contact-force log, per node (this step's sum, last step's, peak), fixed point (0: off)
    uint cfNode; // gLists: per node its contact break (break group, 0 none; force N's bits)
    uint cfPend; // gAcc: per group slot its contact-break state (0 idle, 1 set off, 2 done)
    uint treadState; // gBeams: per node its tread contact's (z, depth the step before, 0, 0) (0: no tread nodes)
    uint selfSplit; // gAcc: per node its self-collision two-body hit count this step (0: no two-body vehicle)
    uint clipMem; // gAcc: anti-clip's side memory, per node 4 words (held m0 m1, recorded p0 p1) (0: no anti-clip vehicle)
    uint clipVmax; // gAcc: per vehicle 2 parities its fastest node's speed (float bits) this step
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
    uint exact; // the friction law's exponentials exact (sExactExp, sExactPow; §36)
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

// Loop caps (every target: WGSL, MSL, CUDA): see explicit_loop_cap.slang. Every
// loop of this file and its includes whose bound is not a constant is capped.
#define EB_LOOP_CAP_WORD gP.loopCapWord
#include "explicit_loop_cap.slang"
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
    s.exact = 0u;
    return s;
}

Surface boxSurface()
{
    Surface s;
    s.va = 3.0; s.ms = 1.2; s.mc = 0.75; s.t2 = 0.01; s.vs = 6.0; s.alpha = 2.0; s.strength = 1.0;
    s.exact = 0u;
    return s;
}

// Surface s as vehicle v's friction law takes it: exact with FLAG_EXACT_FRICTION.
Surface surfaceV(Surface s, uint v)
{
    s.exact = (gStatic[v * 4u + 3u].x & FLAG_EXACT_FRICTION) != 0u ? 1u : 0u;
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

// The exact path's inverse square root (rubble_exact_lengths; explicit_beams.rs
// exact_inv_sqrt): the fast estimate and three more Newton steps, products and
// differences in the CPU's order, so the CPU and GPU agree bit for bit.
float sExactInvSqrt(float x)
{
    int i = 0x5f3759df - (asint(x) >> 1);
    float y = asfloat(i);
    float hx = smul(0.5, x);
    y = smul(y, nf(1.5 - smul(smul(hx, y), y)));
    y = smul(y, nf(1.5 - smul(smul(hx, y), y)));
    y = smul(y, nf(1.5 - smul(smul(hx, y), y)));
    return smul(y, nf(1.5 - smul(smul(hx, y), y)));
}

// rubble_exact_lengths' (length, inverse length) of squared length q
// (explicit_beams.rs exact_length): the length q·y with one Newton correction
// through its exact residual.
float2 sExactLength(float q)
{
    float y = sExactInvSqrt(q);
    float l = smul(q, y);
    return float2(nf(fma(nf(fma(-l, l, q)), smul(0.5, y), l)), y);
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

// explicit_beams.rs exact_exp / exact_ln / exact_pow: one specified sequence, strict,
// in the CPU's order (jbeam-semantics-spec.md §36).
static const float LN2_HI = 0.693359375;
static const float LN2_LO = -2.12194440e-4;
float sExactExp(float x)
{
    if (x < -87.0) return 0.0;
    if (x > 88.0) return 3.4e38;
    float k = floor(nf(smul(x, 1.44269502) + 0.5));
    float r = nf(nf(x - smul(k, LN2_HI)) - smul(k, LN2_LO));
    float p = 0.000198412701;
    p = nf(smul(p, r) + 0.00138888892);
    p = nf(smul(p, r) + 0.00833333377);
    p = nf(smul(p, r) + 0.0416666679);
    p = nf(smul(p, r) + 0.166666672);
    p = nf(smul(p, r) + 0.5);
    p = nf(smul(p, r) + 1.0);
    p = nf(smul(p, r) + 1.0);
    return smul(p, asfloat(uint(int(k) + 127) << 23));
}
float sExactLn(float x)
{
    if (!(x >= 1.17549435e-38)) return -1000.0;
    uint b = asuint(x);
    float e = float(int(b >> 23) - 127);
    float m = asfloat((b & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421354)
    {
        m = smul(m, 0.5);
        e = nf(e + 1.0);
    }
    float sv = sdiv(nf(m - 1.0), nf(m + 1.0));
    float s2 = smul(sv, sv);
    float q = sdiv(1.0, 11.0);
    q = nf(smul(q, s2) + sdiv(1.0, 9.0));
    q = nf(smul(q, s2) + sdiv(1.0, 7.0));
    q = nf(smul(q, s2) + sdiv(1.0, 5.0));
    q = nf(smul(q, s2) + sdiv(1.0, 3.0));
    q = nf(smul(q, s2) + 1.0);
    float lnm = smul(nf(sv + sv), q);
    return nf(smul(e, LN2_HI) + nf(lnm + smul(e, LN2_LO)));
}
float sExactPow(float x, float y)
{
    if (y == 0.0) return 1.0;
    if (x == 0.0 && y > 0.0) return 0.0;
    return sExactExp(smul(y, sExactLn(x)));
}
float frictionExp(Surface s, float x) { return s.exact != 0u ? sExactExp(x) : approxExp(x); }
float frictionPow(Surface s, float x, float y) { return s.exact != 0u ? sExactPow(x, y) : approxPow(x, y); }

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
        return ssub3(ssub3(sscale(n, r), sscale(sl, smul(ms, nf(1.0 - frictionExp(s, -sdiv(sigma, s.va)))))), sf);
    float g = nf(s.mc + smul(nf(s.ms - s.mc), frictionExp(s, -frictionPow(s, sdiv(sigma, s.vs), s.alpha))));
    return ssub3(sscale(n, r), sscale(sl, smul(nf(g + min(smul(s.t2, sigma), 5.0)), gr)));
}

// A pressure-wheel node's ground contact (NODE_FRICTION_LIMIT, spec §28): the
// node rises at most half its depth p in one step, and the tangential part at
// most stops the slip in one step
// (explicit_jbeam_ext.rs friction_limit).
float3 frictionLimit(float3 c, float3 f, float3 v, float m, float3 n, float p)
{
    float cn = sdot(c, n);
    float out = nf(sdot(v, n) + smul(nf(sdot(f, n) + cn), sdiv(STEP, m)));
    float most = sdiv(smul(0.5, p), STEP);
    if (cn > 0.0 && out > most)
    {
        float capped = max(nf(smul(nf(most - sdot(v, n)), sdiv(m, STEP)) - sdot(f, n)), 0.0);
        c = sadd3(c, sscale(n, nf(capped - cn)));
    }
    float3 total = sadd3(f, c);
    float3 t = ssub3(total, sscale(n, sdot(total, n)));
    float3 vt = ssub3(v, sscale(n, sdot(v, n)));
    float3 dv = sscale(t, sdiv(STEP, m));
    if (sdot(dv, vt) < 0.0 && sdot(dv, dv) > sdot(vt, vt)) return sadd3(c, ssub3(sscale(-vt, sdiv(m, STEP)), t));
    return c;
}

// The tread contact (NODE_TREAD, jbeam-semantics-spec.md §34; CPU
// explicit_beams::tread_hysteresis and the node pass's tread branch): a compliant,
// rate-independent hysteretic normal force k·(p + z) at depth p, z the node's
// hysteretic deflection (slope μ while |z| grows, λ after a reversal until it passes
// zero), the solver's friction law on that normal force, and the slip limit. State
// in gBeams at treadState per node: (z, the depth the step before), both 0 off the
// ground. Strict arithmetic in the CPU's order. Off the ground (`on` false) it only
// clears the state and returns 0.
static const uint TREAD_GROW_BITS = 0x3e3c7bbfu; // TREAD_GROW, μ
static const uint TREAD_REVERSE_BITS = 0x3f2c73d7u; // TREAD_REVERSE, λ
static const uint TREAD_INV_REVERSE_BITS = 0x3fbe0301u; // 1/λ
static const uint TREAD_K_BITS = 0x4a11dfb2u; // TREAD_K, 1/(h²(1 + λ))

// contactForce's friction law on a given normal force r > 0 (CPU contact_friction).
float3 contactFriction(float r, float3 f, float3 v, float3 n, Surface s, float friction)
{
    float vn = sdot(v, n);
    float fn = sdot(f, n);
    float3 sf = ssub3(f, sscale(n, fn));
    float3 sl = ssub3(v, sscale(n, vn));
    float sigma = ssqrt(sdot(sl, sl));
    if (sigma > 0.0) sl = sdiv3(sl, sigma);
    float gr = smul(smul(r, s.strength), friction);
    float ms = smul(s.ms, gr);
    if (sigma < s.va && gr > 0.0 && sdot(sf, sf) <= smul(ms, ms))
        return ssub3(ssub3(sscale(n, r), sscale(sl, smul(ms, nf(1.0 - frictionExp(s, -sdiv(sigma, s.va)))))), sf);
    float g = nf(s.mc + smul(nf(s.ms - s.mc), frictionExp(s, -frictionPow(s, sdiv(sigma, s.vs), s.alpha))));
    return ssub3(sscale(n, r), sscale(sl, smul(nf(g + min(smul(s.t2, sigma), 5.0)), gr)));
}

// frictionLimit's tangential part alone (CPU explicit_jbeam_ext::slip_limit).
float3 slipLimit(float3 c, float3 f, float3 v, float m, float3 n)
{
    float3 total = sadd3(f, c);
    float3 t = ssub3(total, sscale(n, sdot(total, n)));
    float3 vt = ssub3(v, sscale(n, sdot(v, n)));
    float3 dv = sscale(t, sdiv(STEP, m));
    if (sdot(dv, vt) < 0.0 && sdot(dv, dv) > sdot(vt, vt)) return sadd3(c, ssub3(sscale(-vt, sdiv(m, STEP)), t));
    return c;
}

float3 treadGround(uint gi, float3 f, float3 v, float m, float friction, float p, bool on)
{
    uint at = gP.treadState + gi;
    float4 st = gBeams[at];
    if (!on)
    {
        if (st.x != 0.0 || st.y != 0.0) gBeams[at] = float4(0.0, 0.0, 0.0, 0.0);
        return float3(0.0, 0.0, 0.0);
    }
    float z = st.x;
    float dp = nf(p - st.y);
    float z1;
    if (smul(z, dp) >= 0.0)
        z1 = nf(z + smul(asfloat(TREAD_GROW_BITS), dp));
    else
    {
        float t = nf(z + smul(asfloat(TREAD_REVERSE_BITS), dp));
        if ((t > 0.0) != (z > 0.0))
            z1 = smul(asfloat(TREAD_GROW_BITS), nf(dp + smul(z, asfloat(TREAD_INV_REVERSE_BITS))));
        else
            z1 = t;
    }
    if (z1 < -p) z1 = -p;
    gBeams[at] = float4(z1, p, 0.0, 0.0);
    float r = smul(smul(m, asfloat(TREAD_K_BITS)), nf(p + z1));
    if (!(r > 0.0)) return float3(0.0, 0.0, 0.0);
    float3 n = float3(0.0, 1.0, 0.0);
    return slipLimit(contactFriction(r, f, v, n, surfaceV(groundSurface(), gStatic[gP.nodeInfoBase + gi].w), friction), f, v, m, n);
}

float3 accForce(uint node)
{
    return float3(float(gAcc[node * 3u].load()), float(gAcc[node * 3u + 1u].load()),
                  float(gAcc[node * 3u + 2u].load())) * (1.0 / FIXED);
}

// x in 1/FIXED N units, clamped to ±2·10⁹, rounded to nearest with ties to even
// (explicit_beams.rs to_fixed): spelt out, as the targets' round differ at ties
// (WGSL's ties to even, Metal's away from zero). Every step is exact: y − floor(y)
// for |y| < 2²⁴ (above, y is whole), f·0.5, f + 1.
int toFixed(float x)
{
    float y = clamp(nf(x * FIXED), -2.0e9, 2.0e9);
    float f = floor(y);
    float d = nf(y - f);
    if (d > 0.5 || (d == 0.5 && floor(f * 0.5) * 2.0 != f)) f = f + 1.0;
    return int(f);
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

// --- Anti-clip (jbeam-semantics-spec.md §36; explicit_contact.rs clip_*) ---
// Per node: the sides held this step (last step's contacts) and those recorded for
// the next, each a word owner << 21 | triangle << 1 | (side > 0), CLIP_EMPTY none.
// A contact records its word into the two smallest (order free: atomic min, the
// displaced word carried to the second slot); the node phase rolls them.
static const float CLIP_DEPTH = 0.1;
static const uint CLIP_EMPTY = 0x7FFFFFFFu;
static const uint CLIP_SELF = 1023u;
uint clipAt(uint n) { return gP.clipMem + n * 4u; }
// The held side of node n for triangle t of `owner`: ±1, or 0 none.
float clipHeld(uint n, uint owner, uint t)
{
    uint key = (owner << 20) | t;
    for (uint k = 0u; k < 2u; k++)
    {
        uint w = uint(gAcc[clipAt(n) + k].load());
        if (w != CLIP_EMPTY && (w >> 1) == key) return (w & 1u) != 0u ? 1.0 : -1.0;
    }
    return 0.0;
}
uint clipWord(uint owner, uint t, float side) { return (owner << 21) | (t << 1) | (side > 0.0 ? 1u : 0u); }
void clipRecord(uint n, uint word)
{
    int x = int(word);
    int prev = gAcc[clipAt(n) + 2u].min(x);
    gAcc[clipAt(n) + 3u].min(max(prev, x));
}
// The node phase: last step's recorded words become the held ones.
void clipRoll(uint gi, uint flagsV)
{
    if (gP.clipMem == 0u || (flagsV & FLAG_CLIP_MEM) == 0u) return;
    uint at = clipAt(gi);
    gAcc[at].store(gAcc[at + 2u].load());
    gAcc[at + 1u].store(gAcc[at + 3u].load());
    gAcc[at + 2u].store(int(CLIP_EMPTY));
    gAcc[at + 3u].store(int(CLIP_EMPTY));
}
uint clipVmaxAt(uint v, uint parity) { return gP.clipVmax + v * 2u + parity; }
// The node phase: the vehicle's fastest node this step (|v| as the CPU's
// Vec3::length; NaN and 0 leave it).
void clipSpeed(uint vehicle, uint step, uint flagsV, float3 vel)
{
    if (gP.clipMem == 0u || (flagsV & FLAG_CLIP_MEM) == 0u) return;
    float sp = ssqrt(sdot(vel, vel));
    if (sp > 0.0) gAcc[clipVmaxAt(vehicle, step & 1u)].max(asint(sp));
}
// Clears the next step's slot (after this step's node phase, before the next's).
void clipVmaxClear(uint vehicle, uint parity)
{
    if (gP.clipMem != 0u) gAcc[clipVmaxAt(vehicle, parity ^ 1u)].store(0);
}
float clipVmax(uint vehicle, uint parity) { return asfloat(gAcc[clipVmaxAt(vehicle, parity)].load()); }
// The largest of a triangle's nodes' speeds, as the CPU folds it from 0.
float clipTriSpeed(uint4 tn)
{
    float a = ssqrt(sdot(gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.x * 3u + 1u].xyz));
    float b = ssqrt(sdot(gNodes[tn.y * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz));
    float c = ssqrt(sdot(gNodes[tn.z * 3u + 1u].xyz, gNodes[tn.z * 3u + 1u].xyz));
    float m = 0.0;
    if (a > m) m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    return m;
}
// Self-collision's box reach (explicit_beams.rs collision_pass): r, or with
// anti-clip r + (triangle speed + vehicle's fastest) · h + CLIP_DEPTH.
float triReach(uint4 tn, float r, uint parity)
{
    if ((gStatic[tn.w * 4u + 3u].x & FLAG_ANTI_CLIP) == 0u) return r;
    float tv = clipTriSpeed(tn);
    return nf(nf(r + smul(nf(tv + clipVmax(tn.w, parity)), STEP)) + CLIP_DEPTH);
}

// --- The contact-force log (jbeam-semantics-spec.md §32; explicit_beams.rs's
// ContactForceLog): per node the sum of its contacts' force magnitudes in a step,
// fixed point; the node phase moves a complete step's sum to `last` (and the
// peak) and tests the node's contact break, which the next beam phase sets off
// once as if by its first beam. Separate functions, each behind an early return.
void cfAddMagnitude(uint gi, float m)
{
    if (gP.cfAcc == 0u) return;
    uint v = gStatic[gP.nodeInfoBase + gi].w;
    if ((gStatic[v * 4u + 3u].x & FLAG_CFORCE) == 0u) return;
    gAcc[gP.cfAcc + 3u * gi].add(toFixed(m));
}
void cfAdd(uint gi, float3 f)
{
    if (gP.cfAcc == 0u) return;
    cfAddMagnitude(gi, ssqrt(sdot(f, f)));
}
void cfRoll(uint gi, uint groupOff, uint flagsV)
{
    if (gP.cfAcc == 0u || (flagsV & FLAG_CFORCE) == 0u) return;
    uint at = gP.cfAcc + 3u * gi;
    int s = gAcc[at].load();
    gAcc[at + 1u].store(s);
    if (s > gAcc[at + 2u].load()) gAcc[at + 2u].store(s);
    gAcc[at].store(0);
    uint g = gLists[gP.cfNode + 2u * gi];
    if (g != 0u && float(s) * (1.0 / FIXED) > asfloat(gLists[gP.cfNode + 2u * gi + 1u]))
        gAcc[gP.cfPend + groupOff + g].max(1);
}
// Beam gj's group (its own) set off by the last step's contact force: the first
// lane to take the state from 1 to 2 sets the group off as master 0.
void cfFire(uint gj, uint groupOff, uint parity)
{
    if (gP.cfPend == 0u) return;
    int g = abs(asint(gStatic[gP.beamBase + gj * 3u].w));
    if (g == 0) return;
    uint at = gP.cfPend + groupOff + uint(g);
    if (gAcc[at].load() != 1) return;
    if (gAcc[at].exchange(2) != 1) return;
    gGroups[groupSlot(parity, groupOff, uint(g))].min(0u);
    gGroups[groupSlot(parity, groupOff, 0u)].min(0u);
}

// Beam j's group break as a node phase sees it: true for a beam after the first
// master that broke its group (step parity `parity`), which then drops the force it
// computed; when `mark`, any beam of such a group is marked broken.
bool groupDropsExt(uint j, uint beamOff, uint groupOff, uint parity, bool mark, uint master);

bool groupDrops(uint j, uint beamOff, uint groupOff, uint parity, bool mark)
{
    uint4 b0 = gStatic[gP.beamBase + j * 3u];
    int g = asint(b0.w);
    if (g == 0) return false;
    uint master = gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load();
    if ((b0.z & (TYPE_EXT << 8)) != 0u) return groupDropsExt(j, beamOff, groupOff, parity, mark, master);
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

// The deform groups of extension x triggered (spec §19).
void extDeform(uint x, uint vehicle)
{
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    if (e0.y != 0u) gGroups[gP.deformBase + e0.y - 1u].max(1u);
    if (e0.z != 0u) gGroups[gP.deformBase + e0.z - 1u].max(1u);
}

// groupDrops for a beam with an extension record: its extra break groups too
// (spec §17), and its deform groups when a group breaks it.
bool groupDropsExt(uint j, uint beamOff, uint groupOff, uint parity, bool mark, uint master)
{
    uint x = gLists[gP.extIndex + j] - 1u;
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    uint4 e3 = gStatic[gP.extBase + x * 4u + 3u];
    if ((e0.x & EXT_GROUPS) != 0u)
    {
        if (asint(e3.z) != 0) master = min(master, gGroups[groupSlot(parity, groupOff, uint(abs(asint(e3.z))))].load());
        if (asint(e3.w) != 0) master = min(master, gGroups[groupSlot(parity, groupOff, uint(abs(asint(e3.w))))].load());
    }
    if (master == NO_GROUP) return false;
    if (mark)
    {
        float4 s1 = gBeams[j * 2u + 1u];
        if (s1.z == 0.0)
        {
            s1.z = 1.0;
            gBeams[j * 2u + 1u] = s1;
        }
        if ((e0.x & EXT_DEFORM) != 0u) extDeform(x, 0u);
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
        for (uint kGuard = 0u; k + 4u <= info.z; k += 4u)
        {
            EB_PASS(kGuard);
            float3 a = gBeams[base + k].xyz;
            float3 b = gBeams[base + k + 1u].xyz;
            float3 c = gBeams[base + k + 2u].xyz;
            float3 d = gBeams[base + k + 3u].xyz;
            f = sadd3(f, a);
            f = sadd3(f, b);
            f = sadd3(f, c);
            f = sadd3(f, d);
        }
        uint kStop = EB_END(k, info.z, 1u);
        for (; k < kStop; k++) f = sadd3(f, gBeams[base + k].xyz);
        return f;
    }
    for (uint k = 0u, kStop = EB_END(k, info.z, 1u); k < kStop; k++)
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

uint nodeUpdateShapes(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                      float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                      bool solo);
// Explicit vehicles against rigid bodies (explicit_beams_rigid.slang).
bool cplHas(uint v);
uint nodeUpdateCoupled(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                       float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                       bool solo);

void nodeUpdate(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
    if (cplHas(vehicle))
    {
        nodeUpdateCoupled(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                          groupCount, hits, false);
        return;
    }
    if ((flagsV & (FLAG_SHAPES | FLAG_CFORCE | FLAG_TREAD | FLAG_TWO_BODY | FLAG_CLIP_MEM)) != 0u)
    {
        nodeUpdateShapes(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                         groupCount, hits, false);
        return;
    }
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
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
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
        {
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), surfaceV(groundSurface(), vehicle), ground - x.y, V.w);
            if ((info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
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
                f += contactForce(f, vel, mass, n, surfaceV(boxSurface(), vehicle), 0.0, V.w);
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
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
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
    if (cplHas(vehicle))
    {
        return nodeUpdateCoupled(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff,
                                 groupOff, groupCount, hits, true);
    }
    if ((flagsV & (FLAG_SHAPES | FLAG_CFORCE | FLAG_TREAD | FLAG_TWO_BODY | FLAG_CLIP_MEM)) != 0u)
    {
        return nodeUpdateShapes(gi, info, vehicle, local, step, origin, flagsV, ground, boxOff, boxCount, beamOff,
                                groupOff, groupCount, hits, true);
    }
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
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
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
        {
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), surfaceV(groundSurface(), vehicle), ground - x.y, V.w);
            if ((info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
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
                f += contactForce(f, vel, mass, n, surfaceV(boxSurface(), vehicle), 0.0, V.w);
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
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
    return 0u;
}
#endif

// --- Node grab (Rubble only; CPU reference ExplicitBeams::grab_pass): an external
// spring–damper on one node, set by the host per frame (ExplicitBeamsGpu::set_grab).
// gMisc at grabRec(): (on, global node, k bits, c bits), then (target world xyz
// bits, 0). After a step's node phase (its barrier), one lane of the grabbed
// node's vehicle adds k·(target − x) − c·v to the node's force slot (the restart),
// before anything else adds to it or reads it; strict arithmetic in the CPU's
// order. Off (on = 0) it returns at once and nothing changes.
uint grabRec() { return gP.wideMisc + 1u + gP.vehicles; }
void grabApply(uint nodeOff, uint nodeCount, float3 origin)
{
    uint4 g = gMisc[grabRec()];
    if (g.x == 0u) return;
    uint gi = g.y;
    if (gi < nodeOff || gi - nodeOff >= nodeCount) return;
    float3 target = asfloat(gMisc[grabRec() + 1u].xyz);
    float k = asfloat(g.z);
    float c = asfloat(g.w);
    float3 x = sadd3(origin, gNodes[gi * 3u].xyz);
    float3 vel = gNodes[gi * 3u + 1u].xyz;
    float4 F = gNodes[gi * 3u + 2u];
    float3 pull = ssub3(sscale(ssub3(target, x), k), sscale(vel, c));
    gNodes[gi * 3u + 2u] = float4(sadd3(F.xyz, pull), F.w);
}

// --- static shapes (Rubble only): oriented boxes, vertical cylinders, capsules,
// spheres. A vehicle with any (FLAG_SHAPES) steps its nodes here instead of
// nodeUpdate (whose code stays as it was for every other vehicle). Its box slots
// are its axis-aligned boxes then its shapes: per slot (lo, kind) and (hi, at);
// kind 0 an axis-aligned box (lo, hi), else lo, hi the shape's culling box and
// `at` its parameter records (from gP.boxBase): an oriented box (centre, half.x),
// (axis x, half.y), (axis y, half.z), (axis z, 0); a cylinder (centre, radius),
// (half height, 0, 0, 0); a capsule (a, radius), (b, 0); a sphere (centre, radius).
// Strict arithmetic in the CPU reference's order (`shape_face`).

static const uint SHAPE_OBOX = 1u;
static const uint SHAPE_CYLINDER = 2u;
static const uint SHAPE_CAPSULE = 3u;

// Pushed out along d (+y at the centre) when strictly inside the radius.
float4 roundNormal(float3 d, float radius)
{
    float l = ssqrt(sdot(d, d));
    if (!(l < radius)) return float4(0.0, 0.0, 0.0, 0.0);
    if (l > 0.0) return float4(sdiv3(d, l), 1.0);
    return float4(0.0, 1.0, 0.0, 1.0);
}

// The outward normal of the nearest surface of the shape at records `at` (kind
// `kind`) for x strictly inside it (.w 1), else .w 0.
float4 shapeNormal(uint kind, uint at, float3 x)
{
    float4 p0 = asfloat(gStatic[at]);
    float4 p1 = asfloat(gStatic[at + 1u]);
    if (kind == SHAPE_OBOX)
    {
        float4 p2 = asfloat(gStatic[at + 2u]);
        float4 p3 = asfloat(gStatic[at + 3u]);
        float3 h = float3(p0.w, p1.w, p2.w);
        float3 d = ssub3(x, p0.xyz);
        float3 l = float3(sdot(p1.xyz, d), sdot(p2.xyz, d), sdot(p3.xyz, d));
        if (!(l.x > -h.x && l.x < h.x && l.y > -h.y && l.y < h.y && l.z > -h.z && l.z < h.z))
            return float4(0.0, 0.0, 0.0, 0.0);
        float3 n = -p3.xyz;
        float dist = nf(l.z + h.z);
        float t = nf(h.z - l.z);
        if (t < dist) { dist = t; n = p3.xyz; }
        t = nf(l.x + h.x);
        if (t < dist) { dist = t; n = -p1.xyz; }
        t = nf(h.x - l.x);
        if (t < dist) { dist = t; n = p1.xyz; }
        t = nf(l.y + h.y);
        if (t < dist) { dist = t; n = -p2.xyz; }
        t = nf(h.y - l.y);
        if (t < dist) { dist = t; n = p2.xyz; }
        return float4(n, 1.0);
    }
    if (kind == SHAPE_CYLINDER)
    {
        float radius = p0.w;
        float hh = p1.x;
        float3 d = ssub3(x, p0.xyz);
        float r = ssqrt(nf(smul(d.x, d.x) + smul(d.z, d.z)));
        if (!(r < radius && d.y > -hh && d.y < hh)) return float4(0.0, 0.0, 0.0, 0.0);
        float3 n = float3(1.0, 0.0, 0.0);
        if (r > 0.0) n = float3(sdiv(d.x, r), 0.0, sdiv(d.z, r));
        float dist = nf(radius - r);
        float t = nf(hh - d.y);
        if (t < dist) { dist = t; n = float3(0.0, 1.0, 0.0); }
        t = nf(d.y + hh);
        if (t < dist) n = float3(0.0, -1.0, 0.0);
        return float4(n, 1.0);
    }
    if (kind == SHAPE_CAPSULE)
    {
        float3 ab = ssub3(p1.xyz, p0.xyz);
        float l2 = sdot(ab, ab);
        float t = 0.0;
        if (l2 > 0.0) t = min(max(sdiv(sdot(ssub3(x, p0.xyz), ab), l2), 0.0), 1.0);
        return roundNormal(ssub3(x, sadd3(p0.xyz, sscale(ab, t))), p0.w);
    }
    return roundNormal(ssub3(x, p0.xyz), p0.w);
}

// nodeUpdate (and, `solo`, nodeUpdateSolo; with a fuselage drag, nodeUpdateFuse)
// for a vehicle with static shapes: its box slots tested in order as above.
uint nodeUpdateShapes(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV,
                      float ground, uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits,
                      bool solo)
{
    float4 P = gNodes[gi * 3u];
#ifdef EB_VV_SOLO
    if (solo && any(abs(P.xyz - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(vehicle);
#endif
    float4 V = gNodes[gi * 3u + 1u];
    cfRoll(gi, groupOff, flagsV);
    clipRoll(gi, flagsV);
    if (gP.selfSplit != 0u && (flagsV & FLAG_TWO_BODY) != 0u) gAcc[gP.selfSplit + gi].store(0);
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
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
        if ((flags & NODE_TREAD) != 0u)
        {
            float3 c = treadGround(gi, f, vel, mass, V.w, ground - x.y, ground > x.y);
            if (ground > x.y && any(c != float3(0.0, 0.0, 0.0)))
            {
                cfAdd(gi, c);
                f += c;
            }
        }
        else if (ground > x.y)
        {
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), surfaceV(groundSurface(), vehicle), ground - x.y, V.w);
            if ((info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            cfAdd(gi, c);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
        {
            uint4 lo4 = gStatic[gP.boxBase + (boxOff + b) * 2u];
            uint4 hi4 = gStatic[gP.boxBase + (boxOff + b) * 2u + 1u];
            float3 lo = asfloat(lo4.xyz);
            float3 hi = asfloat(hi4.xyz);
            if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
            {
                float4 n = float4(0.0, 0.0, -1.0, 1.0);
                if (lo4.w == 0u)
                {
                    float d = x.z - lo.z;
                    if (hi.z - x.z < d) { d = hi.z - x.z; n.xyz = float3(0.0, 0.0, 1.0); }
                    if (x.x - lo.x < d) { d = x.x - lo.x; n.xyz = float3(-1.0, 0.0, 0.0); }
                    if (hi.x - x.x < d) { d = hi.x - x.x; n.xyz = float3(1.0, 0.0, 0.0); }
                    if (x.y - lo.y < d) { d = x.y - lo.y; n.xyz = float3(0.0, -1.0, 0.0); }
                    if (hi.y - x.y < d) { d = hi.y - x.y; n.xyz = float3(0.0, 1.0, 0.0); }
                }
                else
                    n = shapeNormal(lo4.w, gP.boxBase + hi4.w, x);
                if (n.w != 0.0)
                {
                    float3 c = contactForce(f, vel, mass, n.xyz, surfaceV(boxSurface(), vehicle), 0.0, V.w);
                    cfAdd(gi, c);
                    f += c;
                }
            }
        }
    }
    if ((flags & NODE_IMMOVABLE) == 0u)
    {
        vel = sadd3(vel, sscale(sdiv3(f, mass), STEP));
        pos = sadd3(pos, sscale(vel, STEP));
    }
    clipSpeed(vehicle, step, flagsV, vel);
#ifdef EB_AERO
    if ((flagsV & FLAG_FUSEDRAG) != 0u)
    {
        f = sadd3(float3(0.0, smul(mass, GRAVITY), 0.0), gBeams[gP.aeroState + vehicle].xyz);
        gNodes[gi * 3u] = float4(pos, mass);
        gNodes[gi * 3u + 1u] = float4(vel, V.w);
        gNodes[gi * 3u + 2u] = float4(f, selfGroup);
        return 0u;
    }
#endif
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
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
    return 0u;
}

#include "explicit_beams_rigid.slang"

// Self-collision hits are found in parallel and applied in the specification's
// order (triangles in file order, then candidates in node order): a hit's contact
// force reads its node's force with every earlier hit's share added, as the CPU
// reference does, whatever order the lanes find them in. The finding lanes record
// each hit's key (triangle << 16 | node, both vehicle-local) in the vehicle's hit
// list; selfApply then computes the forces in rounds (a hit waits until every
// earlier hit touching its node has its force) and adds them to the fixed-point
// accumulators. Per vehicle (gStatic at vvStatic, .z and .w): HIT_CAP records of
// HIT_WORDS words in gNext (key, alpha, beta, force xyz, round, 0, the triangle's
// n0, n1, n2, 0, then from the finding lane the contact's normal xyz, penetration,
// relative velocity xyz, reduced mass, friction) then PEND_CAP
// pending beams; two gAcc counters (hits, pending beams), cleared by the node
// phase.
static const uint HIT_CAP = 256u;
static const uint HIT_WORDS = 24u;
static const uint PEND_CAP = 64u;
uint hitList(uint v) { return gStatic[gP.vvStatic + v].z; }
uint hitAcc(uint v) { return gStatic[gP.vvStatic + v].w; }
uint pendList(uint v) { return hitList(v) + HIT_CAP * HIT_WORDS; }

// The break exception (spec §2, step 4: an end on a collision triangle with fewer
// than 3 live plain beams saves the beam) reads the beams' state in spawn order: a
// beam sees the breaks of earlier beams in its step and none of later ones. The
// beam phase runs in parallel, so a beam whose test reaches the exception is
// deferred (marked pending: its record's .w = 2, nothing else written) and one lane
// then resolves the vehicle's pending beams in spawn order (resolvePending), each
// counting live beams as the sequential pass would see them. A beam that breaks in a
// beam phase carries .w = 1 until its next beam phase clears it.
static uint gResolving = 0u;

// Live plain beams at `node` as beam `j` (vehicle-local) of this step's beam phase
// sees them: broken before this step, by an earlier beam of this step, or by an
// earlier master's group break of this step (group slot of this parity) is dead.
uint liveAt(uint node, uint j, uint beamOff, uint groupOff, uint parity)
{
    uint4 info = gStatic[gP.nodeInfoBase + node];
    uint live = 0u;
    for (uint k = 0u, kStop = EB_END(k, info.z, 1u); k < kStop; k++)
    {
        uint b = gLists[info.y + k] >> 1;
        uint4 bn = gStatic[gP.beamBase + b * 3u];
        if ((bn.z & 0xFFu) != KIND_PLAIN) continue;
        float4 s1 = gBeams[b * 2u + 1u];
        bool dead = s1.z != 0.0 && !(s1.w == 1.0 && b - beamOff > j);
        int g = asint(bn.w);
        if (!dead && g != 0)
        {
            uint master = gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load();
            dead = master != NO_GROUP && master < j;
        }
        if (!dead) live++;
    }
    return live;
}

// The break exception of beam gj (vehicle-local index `local`) whose force passed its
// strength: 0 it breaks, 1 an end saves it, 2 deferred (beam phase).
uint breakException(uint gj, uint4 bn, uint local, uint groupOff, uint parity)
{
    bool ta = (gStatic[gP.nodeInfoBase + bn.x].x & NODE_ON_TRIANGLE) != 0u;
    bool tb = (gStatic[gP.nodeInfoBase + bn.y].x & NODE_ON_TRIANGLE) != 0u;
    if (!ta && !tb) return 0u;
    uint v = bn.z >> 16;
    if (gResolving == 0u)
    {
        uint i = uint(gAcc[hitAcc(v) + 1u].add(1));
        if (i < PEND_CAP) gNext[pendList(v) + i] = local;
        float4 s1 = gBeams[gj * 2u + 1u];
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 2.0);
        return 2u;
    }
    uint beamOff = gStatic[v * 4u].z;
    if (ta && liveAt(bn.x, local, beamOff, groupOff, parity) < 3u) return 1u;
    if (tb && liveAt(bn.y, local, beamOff, groupOff, parity) < 3u) return 1u;
    return 0u;
}

#ifdef EB_AERO
void shock2Update(uint gj, uint local, bool fast, uint groupOff, uint parity, uint4 bn, float4 s1);
void triggerUpdate(uint gj, bool fast, uint parity, uint4 bn);
#endif

// A JBeam torsion element (KIND_JTORSION, the CPU reference's torsion_force): the
// torque -k wrap(phi - phi0) - c dphi/dt on the dihedral angle phi of nodes 1-2-3-4
// about the 2->3 axis, as forces torque * dphi/dx_i on the four nodes (this record's
// incidence entries for nodes 1 and 2, the next record's for 3 and 4). State: phi0 in
// the rest slot, the deform torque in Tc. Static: (k, c, -, -), the watched break
// groups (c2.xyz): it breaks the step after one of them breaks (the last step's
// group slots, parity ^ 1).
// atan2 by the CPU's explicit sequence (explicit_beams.rs exact_atan2,
// jbeam-semantics-spec.md §35): no hardware atan2, bit for bit.
float sAtan2(float y, float x)
{
    float ax = abs(x);
    float ay = abs(y);
    bool swap = ay > ax;
    float lo = swap ? ax : ay;
    float hi = swap ? ay : ax;
    if (hi == 0.0) return 0.0;
    float a = sdiv(lo, hi);
    float t = a;
    float base = 0.0;
    if (a > asfloat(0x3ed413cdu))
    {
        t = sdiv(nf(a - 1.0), nf(a + 1.0));
        base = asfloat(0x3f490fdbu);
    }
    float t2 = smul(t, t);
    // The series coefficients (-1)^k / (2k + 1), k = 11 down to 0 (the CPU's ATAN_C).
    float p = asfloat(0xbd321643u);
    p = nf(smul(p, t2) + asfloat(0x3d430c31u));
    p = nf(smul(p, t2) + asfloat(0xbd579436u));
    p = nf(smul(p, t2) + asfloat(0x3d70f0f1u));
    p = nf(smul(p, t2) + asfloat(0xbd888889u));
    p = nf(smul(p, t2) + asfloat(0x3d9d89d9u));
    p = nf(smul(p, t2) + asfloat(0xbdba2e8cu));
    p = nf(smul(p, t2) + asfloat(0x3de38e39u));
    p = nf(smul(p, t2) + asfloat(0xbe124925u));
    p = nf(smul(p, t2) + asfloat(0x3e4ccccdu));
    p = nf(smul(p, t2) + asfloat(0xbeaaaaabu));
    p = nf(smul(p, t2) + asfloat(0x3f800000u));
    float r = nf(base + smul(t, p));
    if (swap) r = nf(asfloat(0x3fc90fdbu) - r);
    if (x < 0.0) r = nf(asfloat(0x40490fdbu) - r);
    if (y < 0.0) r = -r;
    return r;
}

float wrapAngle(float a)
{
    if (a > 3.14159274) return nf(a - 6.28318548);
    if (a <= -3.14159274) return nf(a + 6.28318548);
    return a;
}

void jtorsionUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1, float4 c1, uint4 r2,
                    float3 x1, float3 x2, float3 v1, float3 v2)
{
    uint4 bn2 = gStatic[gP.beamBase + (gj + 1u) * 3u];
    float4 s0 = gBeams[gj * 2u];
    int group = asint(bn.w);
    float3 x3 = gNodes[bn2.x * 3u].xyz;
    float3 x4 = gNodes[bn2.y * 3u].xyz;
    float3 v3 = gNodes[bn2.x * 3u + 1u].xyz;
    float3 v4 = gNodes[bn2.y * 3u + 1u].xyz;
    bool broken = false;
    if (r2.x != 0u && gGroups[groupSlot(parity ^ 1u, groupOff, r2.x)].load() != NO_GROUP) broken = true;
    if (r2.y != 0u && gGroups[groupSlot(parity ^ 1u, groupOff, r2.y)].load() != NO_GROUP) broken = true;
    if (r2.z != 0u && gGroups[groupSlot(parity ^ 1u, groupOff, r2.z)].load() != NO_GROUP) broken = true;
    float rest = s0.x;
    float t = 0.0;
    float3 g1 = float3(0.0, 0.0, 0.0);
    float3 g2 = g1;
    float3 g3 = g1;
    float3 g4 = g1;
    if (!broken)
    {
        float3 b1 = ssub3(x2, x1);
        float3 b2 = ssub3(x3, x2);
        float3 b3 = ssub3(x4, x3);
        float3 n1 = scross(b1, b2);
        float3 n2 = scross(b2, b3);
        float l1 = sdot(n1, n1);
        float l2 = sdot(n2, n2);
        float ll = sdot(b2, b2);
        float lb = ssqrt(ll);
        float phi = sAtan2(smul(lb, sdot(b1, n2)), sdot(n1, n2));
        if (!(l1 < 1e-12 || l2 < 1e-12 || ll < 1e-12))
        {
            g1 = sscale(n1, sdiv(-lb, l1));
            g4 = sscale(n2, sdiv(lb, l2));
            float p = sdiv(-sdot(b1, b2), ll);
            float q = sdiv(-sdot(b3, b2), ll);
            g2 = ssub3(sscale(g1, nf(p - 1.0)), sscale(g4, q));
            g3 = ssub3(sscale(g4, nf(q - 1.0)), sscale(g1, p));
        }
        float rate = nf(nf(nf(sdot(g1, v1) + sdot(g2, v2)) + sdot(g3, v3)) + sdot(g4, v4));
        float d = wrapAngle(nf(phi - rest));
        if (smul(c1.x, abs(d)) > s0.y)
        {
            float m = sdiv(s0.y, c1.x);
            d = d < 0.0 ? -m : m;
            rest = wrapAngle(nf(phi - d));
        }
        t = nf(smul(-c1.x, d) - smul(c1.y, rate));
        if (abs(t) > s1.x) broken = true;
    }
    if (broken)
    {
        t = 0.0;
        if (group > 0)
        {
            gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
            gGroups[groupSlot(parity, groupOff, 0u)].min(local);
        }
        gBeams[gj * 2u + 1u] = float4(s1.x, 0.0, 1.0, 1.0);
        float4 p1 = gBeams[(gj + 1u) * 2u + 1u];
        gBeams[(gj + 1u) * 2u + 1u] = float4(p1.x, 0.0, 1.0, 1.0);
    }
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    if (rest != s0.x) gBeams[gj * 2u] = float4(rest, s0.yzw);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(sscale(g1, t), 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(sscale(g2, t), 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(sscale(g3, t), 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(sscale(g4, t), 0.0);
}

// Beam bn's vehicle keeps strength under plastic stretch (FLAG_KEEP_STRENGTH).
bool keepStrength(uint4 bn) { return (gStatic[(bn.z >> 16) * 4u + 3u].x & FLAG_KEEP_STRENGTH) != 0u; }

// A JBeam-mode beam (kinds KIND_JPLAIN..KIND_JCOUPLER; the CPU reference's
// jbeam_force_filtered, coupler_force and beam_pass; engine/docs/jbeam-semantics-spec.md).
// Exact lengths (JBeam mode spawns without the fast inverse square root); no break
// exception. Records: (k, c, limit k or the deformLimitStress cap, limit c),
// (short, long, bound zone, damping filter record + 1 or 0); plain and support: the
// rest-length limits and the support's break length in metres; bounded: the bounds as
// fractions of the rest length, the velocity split and fast damping in Tc and Tt. A
// filter record in gBeams is (last rate, last output, a, b).
void jbeamUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1)
{
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    float4 c2 = asfloat(r2);
    float4 s0 = gBeams[gj * 2u];
    uint kind = bn.z & 0xFFu;
    uint type = (bn.z >> 8) & 0xFFu;
    int group = asint(bn.w);
    float4 pa = gNodes[bn.x * 3u];
    float4 pb = gNodes[bn.y * 3u];
    float3 va = gNodes[bn.x * 3u + 1u].xyz;
    float3 vb = gNodes[bn.y * 3u + 1u].xyz;
    bool broken = false;
    float strength = s1.x;
    if (kind == KIND_JTORSION2) return; // its element's first record writes its forces
    if (kind == KIND_JTORSION)
    {
        jtorsionUpdate(gj, local, groupOff, parity, bn, s1, c1, r2, pa.xyz, pb.xyz, va, vb);
        return;
    }
    if (kind == KIND_JCOUPLER)
    {
        float3 fc = ssub3(sscale(ssub3(pa.xyz, pb.xyz), -c1.x), sscale(ssub3(va, vb), c1.y));
        float m = ssqrt(sdot(fc, fc));
        if (m > strength)
        {
            broken = true;
            fc = float3(0.0, 0.0, 0.0);
            if (group > 0)
            {
                gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
                gGroups[groupSlot(parity, groupOff, 0u)].min(local);
            }
        }
        if (broken) gBeams[gj * 2u + 1u] = float4(strength, m, 1.0, 1.0);
        else if (s1.w != 0.0) gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(fc, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-fc, 0.0);
        return;
    }
    float rest = s0.x;
    float tc = s0.y;
    float tt = s0.z;
    float gate = s0.w;
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    // The exact length and inverse length (jbeam-semantics-spec.md §35; the CPU's
    // exact_length), bit for bit.
    float2 li = sExactLength(q);
    float inv = li.y;
    float len = li.x;
    float e = nf(len - rest);
    float u = smul(sdot(ssub3(va, vb), d), inv);
    if (r2.w != 0u)
    {
        // dampCutoffHz: the RC low-pass of the rate (spec §6).
        uint fi = r2.w - 1u;
        float4 flt = gBeams[fi];
        float y = nf(smul(flt.z, u) - smul(flt.w, flt.y));
        gBeams[fi] = float4(u, y, flt.z, flt.w);
        u = y;
    }
    float k = c1.x;
    float c = c1.y;
    float damp = smul(c, u);
    float last = 0.0;
    float f = 0.0;
    bool lengthBreak = false;
    if (kind == KIND_JBOUNDED)
    {
        float au = abs(u);
        if (au > tc)
        {
            float m = nf(smul(c, tc) + smul(tt, nf(au - tc)));
            damp = u < 0.0 ? -m : m;
        }
        float lb = smul(c2.y, rest);
        float sb = smul(c2.x, rest);
        float o = 0.0;
        if (e > lb) o = nf(e - lb);
        else if (e < -sb) o = nf(-e - sb);
        if (o > 0.0)
        {
            float sz = c2.z > 0.0 ? min(sdiv(o, c2.z), 1.0) : 1.0;
            k = nf(k + smul(nf(c1.z - k), sz));
            damp = nf(damp + smul(nf(smul(c1.w, u) - damp), sz));
        }
    }
    else if (kind == KIND_JSUPPORT && e > 0.0)
    {
        if (len > c2.y) lengthBreak = true;
        k = 0.0;
        damp = 0.0;
    }
    if (lengthBreak)
    {
        broken = true;
        if (group > 0)
        {
            gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
            gGroups[groupSlot(parity, groupOff, 0u)].min(local);
        }
    }
    else
    {
        f = nf(smul(-k, e) - damp);
        last = f;
        float tau = abs(f);
        if (tau > gate && type == TYPE_DEFORMABLE && k != 0.0)
        {
            if (f > tc && e < 0.0)
            {
                float delta = nf(e + smul(sdiv(tc, k), nf(1.0 - c2.z)));
                float old = rest;
                float floorLen = max(c2.x, MIN_REST);
                float free = nf(rest + delta);
                if (free < floorLen)
                {
                    rest = min(floorLen, old);
                    f = nf(smul(-k, nf(len - rest)) - damp);
                }
                else
                {
                    rest = free;
                    f = nf(f - sdiv(nf(f - tc), 2.0));
                }
                tau = abs(f);
                if (rest < old)
                {
                    tc = min(sdiv(smul(tc, old), rest), max(c1.z, tc));
                    gate = min(min(tc, tt), strength);
                }
            }
            else if (f < -tt && e > 0.0)
            {
                float delta = nf(e - smul(sdiv(tt, k), nf(1.0 - c2.z)));
                float old = rest;
                float free = nf(rest + delta);
                if (free > c2.y)
                {
                    rest = max(c2.y, old);
                    f = nf(smul(-k, nf(len - rest)) - damp);
                }
                else
                {
                    rest = free;
                    f = nf(f - sdiv(nf(f + tt), 2.0));
                }
                tau = abs(f);
                if (rest > old)
                {
                    tt = min(sdiv(smul(tt, rest), old), max(c1.z, tt));
                    gate = min(min(tc, tt), strength);
                    if (!keepStrength(bn)) strength = nf(strength - smul(nf(rest - old), k));
                }
            }
        }
        if (tau > strength)
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
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// Set off group g (> 0) as master `local` in this beam phase.
void extSetOff(int g, uint local, uint groupOff, uint parity)
{
    if (g <= 0) return;
    gGroups[groupSlot(parity, groupOff, uint(g))].min(local);
    gGroups[groupSlot(parity, groupOff, 0u)].min(local);
}

// A beam with several break groups that broke on its own in the last beam phase
// (spec §17): sets off all its groups now, unless one of them had broken it first
// (a master before it in the last step's slots).
void extPending(uint gj, uint local, uint groupOff, uint parity)
{
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    if ((bn.z & (TYPE_EXT << 8)) == 0u) return;
    uint x = gLists[gP.extIndex + gj] - 1u;
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    if ((e0.x & EXT_GROUPS) == 0u) return;
    uint4 e3 = gStatic[gP.extBase + x * 4u + 3u];
    int g0 = asint(bn.w);
    int g1 = asint(e3.z);
    int g2 = asint(e3.w);
    uint last = parity ^ 1u;
    uint m = NO_GROUP;
    if (g0 != 0) m = min(m, gGroups[groupSlot(last, groupOff, uint(abs(g0)))].load());
    if (g1 != 0) m = min(m, gGroups[groupSlot(last, groupOff, uint(abs(g1)))].load());
    if (g2 != 0) m = min(m, gGroups[groupSlot(last, groupOff, uint(abs(g2)))].load());
    if (m != NO_GROUP && m < local) return;
    extSetOff(g0, local, groupOff, parity);
    extSetOff(g1, local, groupOff, parity);
    extSetOff(g2, local, groupOff, parity);
}

// A JBeam beam with an extension record (the CPU reference's ext_beam,
// jbeam_ext_force, lbeam_force and the pressure group; spec §16-§26).
void jbeamExtUpdate(uint gj, uint local, uint groupOff, uint parity, uint4 bn, float4 s1)
{
    uint kind = bn.z & 0xFFu;
    uint x = gLists[gP.extIndex + gj] - 1u;
    uint4 e0 = gStatic[gP.extBase + x * 4u];
    float4 e1 = asfloat(gStatic[gP.extBase + x * 4u + 1u]);
    float4 e2 = asfloat(gStatic[gP.extBase + x * 4u + 2u]);
    uint4 e3 = gStatic[gP.extBase + x * 4u + 3u];
    if (kind == KIND_JPTRI2 || kind == KIND_JATRI2 || kind == KIND_JLBEAM2) return; // the first record writes the forces
    if (kind == KIND_JATRI)
    {
        // A triangle's drag and lift in still air (explicit_jbeam_ext.rs
        // aero_tri_force): E1 = (x2, drag factor, lift factor, sin stall).
        uint n2 = asuint(e1.x);
        float3 x0 = gNodes[bn.x * 3u].xyz;
        float3 x1 = gNodes[bn.y * 3u].xyz;
        float3 x2 = gNodes[n2 * 3u].xyz;
        float3 v0 = gNodes[bn.x * 3u + 1u].xyz;
        float3 v1 = gNodes[bn.y * 3u + 1u].xyz;
        float3 v2 = gNodes[n2 * 3u + 1u].xyz;
        float3 vm = sscale(sadd3(sadd3(v0, v1), v2), 1.0 / 3.0);
        float s2 = sdot(vm, vm);
        float3 cr = scross(ssub3(x1, x0), ssub3(x2, x0));
        float cl = ssqrt(sdot(cr, cr));
        float3 f = float3(0.0, 0.0, 0.0);
        if (s2 > 1e-12 && cl > 0.0)
        {
            float sp = ssqrt(s2);
            float3 u = sscale(vm, sdiv(-1.0, sp));
            float3 n = sscale(cr, sdiv(1.0, cl));
            float c = sdot(n, u);
            float sa = abs(c);
            float ca = ssqrt(max(nf(1.0 - smul(c, c)), 0.0));
            float q = smul(smul(0.5, cl), s2);
            f = sscale(u, smul(smul(e1.y, q), sa));
            float3 l = ssub3(n, sscale(u, c));
            float ll = ssqrt(sdot(l, l));
            if (ll > 0.0)
            {
                float cla = sa <= e1.w ? smul(sa, ca) : smul(e1.w, ca);
                float m = smul(smul(e1.z, q), cla);
                f = sadd3(f, sscale(l, sdiv(c < 0.0 ? -m : m, ll)));
            }
            f = sscale(f, 1.0 / 3.0);
        }
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    if (kind == KIND_JPGROUP)
    {
        // The enclosed volume about node a, isothermal gauge pressure, capped.
        float3 c = gNodes[bn.x * 3u].xyz;
        uint first = asuint(e1.x);
        uint count = asuint(e1.y);
        float v = 0.0;
        for (uint t = 0u, tStop = EB_END(t, count, 1u); t < tStop; t++)
        {
            uint r = first + 2u * t;
            uint4 ra = gStatic[gP.beamBase + r * 3u];
            uint4 rb = gStatic[gP.beamBase + (r + 1u) * 3u];
            float3 x0 = ssub3(gNodes[ra.x * 3u].xyz, c);
            float3 x1 = ssub3(gNodes[ra.y * 3u].xyz, c);
            float3 x2 = ssub3(gNodes[rb.x * 3u].xyz, c);
            v = nf(v + sdot(x0, scross(x1, x2)));
        }
        v = max(sdiv(v, 6.0), smul(0.01, e1.z));
        float p = min(nf(smul(e1.w, sdiv(e1.z, v)) - ATMOSPHERE), e2.x);
        float4 st = gBeams[gP.extState + x];
        if (parity == 0u) st.y = p;
        else st.z = p;
        gBeams[gP.extState + x] = st;
        return;
    }
    if (kind == KIND_JPTRI)
    {
        uint g = asuint(e1.y);
        uint gx = gLists[gP.extIndex + g] - 1u;
        float4 gs = gBeams[gP.extState + gx];
        float p = parity == 0u ? gs.z : gs.y; // the last step's
        uint n2 = asuint(e1.x);
        float3 x0 = gNodes[bn.x * 3u].xyz;
        float3 x1 = gNodes[bn.y * 3u].xyz;
        float3 x2 = gNodes[n2 * 3u].xyz;
        float3 f = sscale(scross(ssub3(x1, x0), ssub3(x2, x0)), sdiv(p, 6.0));
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(f, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    uint4 r2 = gStatic[gP.beamBase + gj * 3u + 2u];
    float4 c2 = asfloat(r2);
    float4 s0 = gBeams[gj * 2u];
    uint type = (bn.z >> 8) & 3u;
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
    float4 st = gBeams[gP.extState + x];
    bool rampOn = st.x > 0.0;
    if (rampOn)
    {
        rest = nf(rest + asfloat(e3.x));
        st.x = st.x - 1.0;
        gBeams[gP.extState + x] = st;
    }
    float3 d = ssub3(pa.xyz, pb.xyz);
    float q = sdot(d, d);
    float2 li = sExactLength(q); // spec §35, the CPU's exact_length
    float inv = li.y;
    float len = li.x;
    float u = smul(sdot(ssub3(va, vb), d), inv);
    if (r2.w != 0u)
    {
        uint fi = r2.w - 1u;
        float4 flt = gBeams[fi];
        float y = nf(smul(flt.z, u) - smul(flt.w, flt.y));
        gBeams[fi] = float4(u, y, flt.z, flt.w);
        u = y;
    }
    float k = c1.x;
    float c = c1.y;
    float e = 0.0;
    float f = 0.0;
    float last = 0.0;
    bool broken = false;
    bool lengthBreak = false;
    bool done = false;
    float3 armA = float3(0.0, 0.0, 0.0);
    float3 armB = float3(0.0, 0.0, 0.0);
    if (kind == KIND_JLBEAM)
    {
        // The arms' law-of-cosines length at the rest angle (rest holds cos θ0).
        float3 xc = gNodes[asuint(e1.x) * 3u].xyz;
        float3 ra = ssub3(pa.xyz, xc);
        float3 rb = ssub3(pb.xyz, xc);
        float a2 = sdot(ra, ra);
        float b2 = sdot(rb, rb);
        float ab = ssqrt(smul(a2, b2));
        float lr = ssqrt(max(nf(nf(a2 + b2) - smul(smul(2.0, ab), rest)), 0.0));
        e = nf(len - lr);
        float spring = smul(k, e);
        f = nf(-spring - smul(c, u));
        if (abs(spring) > tc && ab > 1e-12)
        {
            float m = sdiv(tc, k);
            m = e < 0.0 ? -m : m;
            float l = nf(len - m);
            rest = clamp(sdiv(nf(nf(a2 + b2) - smul(l, l)), smul(2.0, ab)), -1.0, 1.0);
            f = nf((e < 0.0 ? tc : -tc) - smul(c, u));
            spring = e < 0.0 ? -tc : tc;
        }
        // The spring's arm terms (the gradient of ½k·e² on the arms; lbeam_force):
        // spring·∂L/∂a·â on id1, spring·∂L/∂b·b̂ on id2, their negative sum on id3.
        float la = ssqrt(a2);
        float lb = ssqrt(b2);
        float lnow = ssqrt(max(nf(nf(a2 + b2) - smul(smul(2.0, ab), rest)), 0.0));
        if (lnow > 1e-9 && la > 1e-12 && lb > 1e-12)
        {
            float g = sdiv(1.0, lnow);
            armA = sscale(sscale(ra, smul(g, nf(1.0 - smul(sdiv(lb, la), rest)))), spring);
            armB = sscale(sscale(rb, smul(g, nf(1.0 - smul(sdiv(la, lb), rest)))), spring);
        }
        last = f;
        if (abs(f) > strength) broken = true;
        done = true;
    }
    else
    {
        e = nf(len - rest);
    }
    float damp = smul(c, u);
    if (!done && kind == KIND_JPRESSURED)
    {
        float l0 = e1.w;
        float den = max(nf(l0 - smul(e1.z, nf(l0 - len))), smul(0.01, l0));
        float gauge = min(nf(smul(e1.y, sdiv(l0, den)) - ATMOSPHERE), e2.x);
        f = nf(nf(smul(gauge, e1.x) - smul(k, max(e, 0.0))) - damp);
        last = f;
        if (gauge > e2.y || abs(f) > strength) broken = true;
        done = true;
    }
    if (!done && kind == KIND_JBOUNDED)
    {
        float cs = c;
        float cf = tt;
        float split = tc;
        float lc = c1.w;
        if ((e0.x & EXT_REBOUND) != 0u && u > 0.0)
        {
            cs = e2.x;
            cf = e2.y;
            split = e2.z;
            lc = e2.w;
        }
        damp = smul(cs, u);
        float au = abs(u);
        if (au > split)
        {
            float m = nf(smul(cs, split) + smul(cf, nf(au - split)));
            damp = u < 0.0 ? -m : m;
        }
        float lb = smul(c2.y, rest);
        float sb = smul(c2.x, rest);
        float o = 0.0;
        if (e > lb) o = nf(e - lb);
        else if (e < -sb) o = nf(-e - sb);
        if (o > 0.0)
        {
            float sz = c2.z > 0.0 ? min(sdiv(o, c2.z), 1.0) : 1.0;
            k = nf(k + smul(nf(c1.z - k), sz));
            damp = nf(damp + smul(nf(smul(lc, u) - damp), sz));
        }
    }
    else if (!done && kind == KIND_JSUPPORT && e > 0.0)
    {
        if (len > c2.y) lengthBreak = true;
        k = 0.0;
        damp = 0.0;
    }
    else if (!done && kind == KIND_JANISO)
    {
        if (len > e1.w) lengthBreak = true;
        // Past the spawned length (§20): rest + (E2 .y + ramp step · steps left).
        float sw = nf(rest + nf(e2.y + smul(asfloat(e3.x), st.x)));
        float past = nf(len - sw);
        if (past > 0.0)
        {
            float zone = smul(e1.z, sw);
            float sz = zone > 0.0 ? min(sdiv(past, zone), 1.0) : 1.0;
            k = nf(k + smul(nf(e1.x - k), sz));
            damp = smul(nf(c + smul(nf(e1.y - c), sz)), u);
        }
    }
    if (lengthBreak)
    {
        broken = true;
        f = 0.0;
        last = 0.0;
    }
    else if (!done)
    {
        f = nf(smul(-k, e) - damp);
        last = f;
        float tau = abs(f);
        if (tau > gate && type == TYPE_DEFORMABLE && k != 0.0)
        {
            if (f > tc && e < 0.0)
            {
                float delta = nf(e + smul(sdiv(tc, k), nf(1.0 - c2.z)));
                float old = rest;
                float floorLen = max(c2.x, MIN_REST);
                float free = nf(rest + delta);
                if (free < floorLen)
                {
                    rest = min(floorLen, old);
                    f = nf(smul(-k, nf(len - rest)) - damp);
                }
                else
                {
                    rest = free;
                    f = nf(f - sdiv(nf(f - tc), 2.0));
                }
                tau = abs(f);
                if (rest < old)
                {
                    tc = min(sdiv(smul(tc, old), rest), max(c1.z, tc));
                    gate = min(min(tc, tt), strength);
                }
            }
            else if (f < -tt && e > 0.0)
            {
                float delta = nf(e - smul(sdiv(tt, k), nf(1.0 - c2.z)));
                float old = rest;
                float free = nf(rest + delta);
                if (free > c2.y)
                {
                    rest = max(c2.y, old);
                    f = nf(smul(-k, nf(len - rest)) - damp);
                }
                else
                {
                    rest = free;
                    f = nf(f - sdiv(nf(f + tt), 2.0));
                }
                tau = abs(f);
                if (rest > old)
                {
                    tt = min(sdiv(smul(tt, rest), old), max(c1.z, tt));
                    gate = min(min(tc, tt), strength);
                    if (!keepStrength(bn)) strength = nf(strength - smul(nf(rest - old), k));
                }
            }
        }
        if (tau > strength) broken = true;
    }
    if (broken)
    {
        f = 0.0;
        if ((e0.x & EXT_GROUPS) == 0u) extSetOff(group, local, groupOff, parity);
    }
    if ((e0.x & EXT_DEFORM) != 0u)
    {
        bool yields = kind == KIND_JPLAIN || kind == KIND_JSUPPORT || kind == KIND_JANISO;
        if (broken || (yields && !(st.x > 0.0) && abs(nf(rest - e2.x)) > smul(asfloat(e0.w), e2.x))) extDeform(x, 0u);
    }
    float4 n0 = float4(rest, tc, tt, gate);
    if (any(n0 != s0)) gBeams[gj * 2u] = n0;
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    if (kind == KIND_JLBEAM)
    {
        if (broken)
        {
            armA = float3(0.0, 0.0, 0.0);
            armB = float3(0.0, 0.0, 0.0);
        }
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(sadd3(push, armA), 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(sadd3(-push, armB), 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(-sadd3(armA, armB), 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

void beamUpdate(uint gj, uint local, bool fast, uint groupOff, uint parity)
{
    cfFire(gj, groupOff, parity);
    float4 s1 = gBeams[gj * 2u + 1u];
    if (s1.z != 0.0)
    {
        // Broken in the last beam phase: no longer this step's break.
        if (s1.w != 0.0)
        {
            gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
            if (s1.w == 1.0) extPending(gj, local, groupOff, parity);
        }
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        // A broken LBEAM's vertex record (it has no update of its own).
        if ((gStatic[gP.beamBase + gj * 3u].z & 0xFFu) == KIND_JLBEAM)
            gBeams[gP.incBase + gLists[gP.slotBase + (gj + 1u) * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    if ((bn.z & 0xFFu) >= KIND_JPLAIN)
    {
        if ((bn.z & (TYPE_EXT << 8)) != 0u)
        {
            jbeamExtUpdate(gj, local, groupOff, parity, bn, s1);
            return;
        }
        jbeamUpdate(gj, local, groupOff, parity, bn, s1);
        return;
    }
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
    float2 li = fast ? float2(0.0, sFastInvSqrt(q)) : sExactLength(q);
    if (fast) li.x = smul(q, li.y);
    float inv = li.y;
    float len = li.x;
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
            uint ex = breakException(gj, bn, local, groupOff, parity);
            if (ex == 2u) return;
            bool saved = ex == 1u;
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
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
    float3 push = sscale(d, smul(f, inv));
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// A triangle's throttle after a step: a waiting triangle counts down (and its idle
// steps up); one that was tested restarts its wait from its hit (gMisc .w, set by
// any of its lanes) or counts it up. A triangle that comes off idle waits at most
// 12 steps and is tested that same step (triangleTest reads it as not waiting).
// The largest wait triangle gt's throttle may hold: 0 when its vehicle runs without
// the throttle (FLAG_NO_THROTTLE), else unbounded.
int throttleCap(uint gt)
{
    uint v = gStatic[gP.triBase + gt].w;
    return (gStatic[v * 4u + 3u].x & FLAG_NO_THROTTLE) != 0u ? 0 : 0x7FFFFFFF;
}

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
        wait = th.w != 0u ? -20000 : min(wait + 1, throttleCap(gt));
    }
    gMisc[mi] = uint4(asuint(wait), asuint(idle), th.z, 0u);
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
    // Anti-clip (§36): 1 the vehicle has it, 2 the triangle is anti-clip only.
    uint clip;
};

// The hit test of contacter node h against triangle t: (alpha, beta, delta) when
// it hits, else none.
bool triangleHit(TriGeo t, float3 p, float r, out float alpha, out float beta, out float gamma, out float delta)
{
    float3 q = nf3(p - t.p0);
    alpha = 0.0;
    beta = 0.0;
    delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    gamma = nf(nf(1.0 - alpha) - beta);
    return alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r;
}

void hitForce(uint v, uint at, uint key, int3 sum, uint round, float3 origin, float r, uint beamOff, uint groupOff,
              uint groupCount, uint parity);

// Self-collision's two-body rule (jbeam-semantics-spec.md §36; the CPU's
// apply_two_body_self_hits): vehicle v has it (FLAG_TWO_BODY) and its counts room.
bool selfTwoBody(uint v)
{
    return gP.selfSplit != 0u && (gStatic[v * 4u + 3u].x & FLAG_TWO_BODY) != 0u;
}

// Node x as k = its hits this step sub-bodies: inverse mass k/m (0 immovable).
float selfInvMass(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return 0.0;
    return sdiv(float(uint(gAcc[gP.selfSplit + x].load())), gNodes[x * 3u].w);
}

// Node x's acceleration from its force at the pass's start (0 immovable).
float3 selfAcc(uint x, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return float3(0.0, 0.0, 0.0);
    float3 f = beamSum(gNodes[x * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + x], beamOff, groupOff, groupCount, parity,
                       false);
    return sdiv3(f, gNodes[x * 3u].w);
}

float selfTerm(float w, float im) { return w > 0.0 ? smul(smul(w, w), im) : 0.0; }

// The two-body force of the hit at record `at` (contacter h): reduced mass from the
// counts, the cancelled force (F_h/m_h − Σ w·F_k/m_k)·m, then the contact rule with
// the geometry the finding lane stored. Strict, in the CPU's order.
float3 selfTwoBodyCore(uint h, uint t0, uint t1, uint t2, float alpha, float beta, float gamma, float3 nn,
                       float pen, float3 vv, float friction, uint beamOff, uint groupOff, uint groupCount,
                       uint parity)
{
    float imt = nf(nf(selfTerm(alpha, selfInvMass(t1)) + selfTerm(beta, selfInvMass(t2)))
                   + selfTerm(gamma, selfInvMass(t0)));
    float inv = nf(selfInvMass(h) + imt);
    float m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
    float3 a3 = sadd3(sadd3(sscale(selfAcc(t1, beamOff, groupOff, groupCount, parity), alpha),
                            sscale(selfAcc(t2, beamOff, groupOff, groupCount, parity), beta)),
                      sscale(selfAcc(t0, beamOff, groupOff, groupCount, parity), gamma));
    float3 fh = sscale(ssub3(selfAcc(h, beamOff, groupOff, groupCount, parity), a3), m);
    return contactForce(fh, vv, m, nn, surfaceV(boxSurface(), gStatic[gP.nodeInfoBase + h].w), pen, friction);
}

float3 selfTwoBodyForce(uint at, uint h, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    return selfTwoBodyCore(h, gNext[at + 8u], gNext[at + 9u], gNext[at + 10u], alpha, beta,
                           nf(nf(1.0 - alpha) - beta), asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u])),
                           asfloat(gNext[at + 15u]), asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u])),
                           asfloat(gNext[at + 20u]), beamOff, groupOff, groupCount, parity);
}

// Self-collision's anti-clip test (§36; explicit_beams.rs clip_hit, the order-free
// rule of explicit_contact.rs clip_test): contacter h at p against triangle t. The
// side h is kept on: held from last step, or the one it came from when it crossed
// the plane over the step inside the triangle (both a step back), or, new and in
// the band over the triangle, the one it is on. 0: no contact; 1: a hit (weights,
// depth into the contact, normal to h's side); 2: a contact without force (an
// anti-clip-only triangle, h in front of its plane). `word` the side to record.
uint selfClip(TriGeo t, uint h, float3 p, float3 origin, float r, out float alpha, out float beta,
              out float gamma, out float pen, out float3 nn, out uint word)
{
    float delta;
    triangleHit(t, p, r, alpha, beta, gamma, delta);
    pen = 0.0;
    nn = t.n;
    word = 0u;
    uint tl = t.mi - gP.vehicles - gStatic[t.tn.w * 4u + 1u].x;
    float side = clipHeld(h, CLIP_SELF, tl);
    bool over = alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0;
    bool have = over;
    if (side == 0.0)
    {
        bool crossed = false;
        TriGeo pf;
        pf.p0 = ssub3(t.p0, sscale(gNodes[t.tn.x * 3u + 1u].xyz, STEP));
        float3 b1 = ssub3(nf3(origin + gNodes[t.tn.y * 3u].xyz), sscale(gNodes[t.tn.y * 3u + 1u].xyz, STEP));
        float3 b2 = ssub3(nf3(origin + gNodes[t.tn.z * 3u].xyz), sscale(gNodes[t.tn.z * 3u + 1u].xyz, STEP));
        triangleFrame(pf, b1, b2);
        float3 pp = ssub3(p, sscale(gNodes[h * 3u + 1u].xyz, STEP));
        float a0;
        float bb0;
        float g0;
        float d0;
        triangleHit(pf, pp, r, a0, bb0, g0, d0);
        if ((d0 > 0.0 && delta < 0.0) || (d0 < 0.0 && delta > 0.0))
        {
            float sc = sdiv(d0, nf(d0 - delta));
            float a = nf(a0 + smul(nf(alpha - a0), sc));
            float b = nf(bb0 + smul(nf(beta - bb0), sc));
            float g = nf(nf(1.0 - a) - b);
            if (a >= 0.0 && b >= 0.0 && g >= 0.0)
            {
                crossed = true;
                side = d0 > 0.0 ? 1.0 : -1.0;
                alpha = a;
                beta = b;
                gamma = g;
                have = true;
            }
        }
        if (!crossed)
        {
            if (!(over && abs(delta) <= r)) return 0u;
            side = delta < 0.0 ? -1.0 : 1.0;
        }
    }
    float d = side * delta;
    if (!have || !(d <= r && d >= -CLIP_DEPTH)) return 0u;
    word = clipWord(CLIP_SELF, tl, side);
    nn = side > 0.0 ? t.n : -t.n;
    bool only = (t.clip & 2u) != 0u;
    if (only && d >= 0.0) return 2u;
    // At most r (explicit_contact.rs clip_depth): held behind, pushed as at the plane.
    pen = min(only ? -d : nf(r - d), r);
    return 1u;
}

// Contacter node h against triangle t: when h is within r of the face, the hit is
// recorded (the throttle's hit flag, and the vehicle's hit list for selfApply).
void tryContact(TriGeo t, uint h, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount,
                uint parity)
{
    uint4 tn = t.tn;
    if (h == tn.x || h == tn.y || h == tn.z) return;
    // JBeam mode: a node skips the triangles of its own self-collision group.
    uint tg = gMisc[t.mi].z & ~(TRI_BREAKABLE | TRI_ANTI_ONLY);
    if (tg != 0u && uint(gNodes[h * 3u + 2u].w) == tg) return;
    // Spawn exclusions (spec §10.1): the nodes that started inside its range.
    if (gP.triExcl != 0u)
    {
        uint4 ex = gStatic[gP.triExcl + (t.mi - gP.vehicles)];
        for (uint k = 0u; k < TRI_EXCL_CAP; k++)
        {
            if (k >= ex.y) break;
            if (gLists[ex.x + k] == h) return;
        }
    }
    float4 P = gNodes[h * 3u];
    float3 p = nf3(origin + P.xyz);
    if (any(p < t.lo) || any(p > t.hi)) return;
    float alpha;
    float beta;
    float gamma;
    float delta;
    float3 nn;
    float pen;
    if ((t.clip & 1u) != 0u)
    {
        // Anti-clip (§36): its contact recorded whether or not it takes a force.
        uint word;
        uint c = selfClip(t, h, p, origin, r, alpha, beta, gamma, pen, nn, word);
        if (c == 0u) return;
        clipRecord(h, word);
        if (c == 2u) return;
    }
    else
    {
        if (!triangleHit(t, p, r, alpha, beta, gamma, delta)) return;
        nn = t.n;
        if (delta < 0.0)
        {
            nn = -t.n;
            delta = -delta;
        }
        pen = nf(r - delta);
    }
    gMisc[t.mi].w = 1u;
    gsHit = 1u;
    gsUniformHits = 1u;
    uint v = tn.w;
    // The two-body rule (spec §36): every hit's nodes count it (all of them, even
    // past the hit list's room, for selfSerial).
    bool twoBody = selfTwoBody(v);
    if (twoBody)
    {
        gAcc[gP.selfSplit + h].add(1);
        gAcc[gP.selfSplit + tn.x].add(1);
        gAcc[gP.selfSplit + tn.y].add(1);
        gAcc[gP.selfSplit + tn.z].add(1);
    }
    uint i = uint(gAcc[hitAcc(v)].add(1));
    if (i < HIT_CAP)
    {
        uint at = hitList(v) + i * HIT_WORDS;
        gNext[at] = ((t.mi - gP.vehicles - gStatic[v * 4u + 1u].x) << 16) | (h - gStatic[v * 4u].x);
        gNext[at + 6u] = 0u;
        gNext[at + 8u] = tn.x;
        gNext[at + 9u] = tn.y;
        gNext[at + 10u] = tn.z;
        // The contact's geometry (positions and velocities do not change in this
        // phase): selfApply then needs only the node's force.
        float4 V = gNodes[h * 3u + 1u];
        float4 P0 = gNodes[tn.x * 3u];
        float4 P1 = gNodes[tn.y * 3u];
        float4 P2 = gNodes[tn.z * 3u];
#ifdef EB_DET_HITS
        // The frozen baseline's expressions, as its one-lane oracle writes them.
        float3 vv = V.xyz - (alpha * gNodes[tn.y * 3u + 1u].xyz + beta * gNodes[tn.z * 3u + 1u].xyz
                             + gamma * gNodes[tn.x * 3u + 1u].xyz);
        float mt = alpha * P1.w + beta * P2.w + gamma * P0.w;
        float m = P.w * mt / (P.w + mt);
#else
        // Strict, in the CPU reference's order.
        float3 vv = ssub3(V.xyz, sadd3(sadd3(sscale(gNodes[tn.y * 3u + 1u].xyz, alpha),
                                             sscale(gNodes[tn.z * 3u + 1u].xyz, beta)),
                                       sscale(gNodes[tn.x * 3u + 1u].xyz, gamma)));
        float mt = nf(nf(smul(alpha, P1.w) + smul(beta, P2.w)) + smul(gamma, P0.w));
        float m = sdiv(smul(P.w, mt), nf(P.w + mt));
#endif
        gNext[at + 1u] = asuint(alpha);
        gNext[at + 2u] = asuint(beta);
        gNext[at + 12u] = asuint(nn.x);
        gNext[at + 13u] = asuint(nn.y);
        gNext[at + 14u] = asuint(nn.z);
        gNext[at + 15u] = asuint(pen);
        gNext[at + 16u] = asuint(vv.x);
        gNext[at + 17u] = asuint(vv.y);
        gNext[at + 18u] = asuint(vv.z);
        gNext[at + 19u] = asuint(m);
        gNext[at + 20u] = asuint(V.w);
        // The force as it is when no earlier hit touches h (the usual case), here
        // alongside the other lanes' tests; selfRound keeps it for a hit with no
        // earlier hit on its node. Not while beams of this step are pending (their
        // incidence forces are being rewritten).
        uint prov = 0u;
        if (!twoBody && gAcc[hitAcc(v) + 1u].load() == 0)
        {
            hitForce(v, at, gNext[at], int3(0, 0, 0), 0u, origin, r, beamOff, groupOff, groupCount, parity);
            prov = 1u;
        }
        gNext[at + 7u] = prov;
    }
}

// The share of a hit's force f that node role r of the hit gets (1 the contacter,
// 2..4 the triangle's n1, n2, n0), as the accumulators add it.
float3 hitShare(float3 f, uint role, float alpha, float beta, float gamma)
{
    if (role == 1u) return f;
    if (role == 2u) return -alpha * f;
    if (role == 3u) return -beta * f;
    return -gamma * f;
}

int3 toFixed3(float3 f) { return int3(toFixed(f.x), toFixed(f.y), toFixed(f.z)); }

// Triangle gt's geometry; mi < vehicles (none) when it waits this step.
// A breakable triangle (spec §18) is dead once any of its four listed beams was
// broken before this beam phase (.z set, .w not this phase's own break).
bool triangleDead(uint gt)
{
    uint4 l = gStatic[gP.triBreak + gt];
    for (uint k = 0u; k < 4u; k++)
    {
        uint j = l[k];
        if (j == 0xFFFFFFFFu) continue;
        float4 s1 = gBeams[j * 2u + 1u];
        if (s1.z != 0.0 && s1.w != 1.0) return true;
    }
    return false;
}

// A triangle's frame from its three world positions (TriFrame::new).
void triangleFrame(inout TriGeo t, float3 p1, float3 p2)
{
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
}

TriGeo triangleGeo(uint gt, float3 origin, float r, uint parity)
{
    TriGeo t;
    t.mi = gP.vehicles + gt;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    triangleFrame(t, p1, p2);
    uint z = gMisc[t.mi].z;
    t.clip = ((gStatic[t.tn.w * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u ? 1u : 0u) | ((z & TRI_ANTI_ONLY) != 0u ? 2u : 0u);
    float reach = (t.clip & 1u) != 0u ? triReach(t.tn, r, parity) : r;
    t.lo = min(min(t.p0, p1), p2) - float3(reach, reach, reach);
    t.hi = max(max(t.p0, p1), p2) + float3(reach, reach, reach);
    // A broken triangle (spec §18), or an anti-clip-only one without anti-clip: none.
    if (((z & TRI_BREAKABLE) != 0u && triangleDead(gt)) || t.clip == 2u)
    {
        t.lo = float3(3.0e38, 3.0e38, 3.0e38);
        t.hi = -t.lo;
    }
    return t;
}

// Lane `sub` of the `lanes` lanes of triangle gt tests candidates sub, sub + lanes, …
// Forces go to fixed-point atomics, so the lanes' order does not change the sums.
void triangleTest(uint gt, uint sub, uint lanes, float3 origin, float r, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r, parity);
    for (uint k = sub, kStop = EB_END(k, candCount, lanes); k < kStop; k += lanes)
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
    TriGeo t = triangleGeo(gt, origin, r, parity);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub, kStop = EB_END(k, candCount, GRID_SUBS); k < kStop; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub, iStop = EB_END(i, cells, GRID_SUBS); i < iStop; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        // Skip a bucket an earlier cell of this triangle already covers.
        bool dup = false;
        for (uint j = 0u, jStop = EB_END(j, i, 1u); j < jStop; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        for (uint e = gsOff[b], eGuard = 0u; e < gsOff[b + 1u]; e++)
        {
            EB_PASS(eGuard);
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
// 0: a vehicle with a grid always uses it. The skip (16 pairs a lane) gave the same
// hits but resolves the documented same-contacter race differently (the lanes of a
// wave read before any adds), which moved the E36 drive ensemble against RoR from
// 115-150 mm to 310-520 mm (fail; 3 runs each): reverted for accuracy.
static const uint BRUTE_PAIRS_PER_LANE = 16u;
void triangleGridBrute(uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                       uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r, parity);
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub, kStop = EB_END(k, candCount, subs); k < kStop; k += subs)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint k = sub, kStop = EB_END(k, candCount, subs); k < kStop; k += subs)
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
            for (uint i = 0u, iStop = EB_END(i, cells, 1u); i < iStop; i++)
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
                   uint candCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    // The box tryContact tests (triangleGeo's: with anti-clip, its reach).
    float reach = triReach(tn, r, parity);
    float3 lo = min(min(p0, p1), p2) - float3(reach, reach, reach);
    float3 hi = max(max(p0, p1), p2) + float3(reach, reach, reach);
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
    for (uint k = sub, kStop = EB_END(k, candCount, subs); k < kStop; k += subs)
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
                for (uint i = 0u, iStop = EB_END(i, cells, 1u); i < iStop; i++)
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
    for (uint k = 0u, kStop = EB_END(k, wingCount, 1u); k < kStop; k++)
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
    for (uint k = 0u, kStop = EB_END(k, rec.z, 1u); k < kStop; k++)
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
    float2 li = fast ? float2(0.0, sFastInvSqrt(q)) : sExactLength(q);
    if (fast) li.x = smul(q, li.y);
    float inv = li.y;
    float len = li.x;
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
        uint ex = breakException(gj, bn, local, groupOff, parity);
        if (ex == 2u) return;
        bool saved = ex == 1u;
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
    if (strength != s1.x || broken)
        gBeams[gj * 2u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, broken ? 1.0 : 0.0);
    else if (s1.w != 0.0)
        gBeams[gj * 2u + 1u] = float4(s1.xyz, 0.0);
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
    float2 li = fast ? float2(0.0, sFastInvSqrt(q)) : sExactLength(q);
    if (fast) li.x = smul(q, li.y);
    float inv = li.y;
    float len = li.x;
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
    for (uint i = 1u, iStop = EB_END(i, m, 1u); i < iStop; i++)
    {
        uint key = gNext[gP.slotListBase + rec.x + i];
        uint j = i;
        uint jGuard = 0u;
        while (j > 0u && gNext[gP.slotListBase + rec.x + j - 1u] > key)
        {
            EB_PASS(jGuard);
            gNext[gP.slotListBase + rec.x + j] = gNext[gP.slotListBase + rec.x + j - 1u];
            j--;
        }
        gNext[gP.slotListBase + rec.x + j] = key;
    }
    for (uint i = 0u, iStop = EB_END(i, m, 1u); i < iStop; i++) s = sadd3(s, gBeams[slideContribBase() + gNext[gP.slotListBase + rec.x + i]].xyz);
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
    for (uint k = 0u, kStop = EB_END(k, count, 1u); k < kStop; k++)
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
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
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
        {
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), surfaceV(groundSurface(), vehicle), ground - x.y, V.w);
            if ((info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
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
                f += contactForce(f, vel, mass, n, surfaceV(boxSurface(), vehicle), 0.0, V.w);
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
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
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
    float selfGroup = gNodes[gi * 3u + 2u].w; // JBeam self-collision group (kept)
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
        {
            float3 c = contactForce(f, vel, mass, float3(0.0, 1.0, 0.0), surfaceV(groundSurface(), vehicle), ground - x.y, V.w);
            if ((info.x & NODE_FRICTION_LIMIT) != 0u) c = frictionLimit(c, f, vel, mass, float3(0.0, 1.0, 0.0), ground - x.y);
            f += c;
        }
        for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
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
                f += contactForce(f, vel, mass, n, surfaceV(boxSurface(), vehicle), 0.0, V.w);
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
    gNodes[gi * 3u + 2u] = float4(f, selfGroup);
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

// --- Self-collision hits in the specification's order (see tryContact) ---

// Rounds of selfRound before one lane finishes what is left in key order.
static const uint SELF_ROUNDS = 32u;

// Node `h`'s roles in the hit recorded at `at` (key `key`): bit 0 the contacter,
// bits 1..3 the triangle's n1, n2, n0 (a triangle may name a node twice); 0 none.
uint hitRoles(uint at, uint key, uint h, uint nodeOff)
{
    if (nodeOff + (key & 0xFFFFu) == h) return 1u;
    return (gNext[at + 9u] == h ? 2u : 0u) | (gNext[at + 10u] == h ? 4u : 0u) | (gNext[at + 8u] == h ? 8u : 0u);
}

// The accumulator shares (fixed point) node roles `roles` get from the computed hit
// at record `at`, as accAdd adds them.
int3 hitShareFixed(uint at, uint roles)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    float gamma = nf(nf(1.0 - alpha) - beta);
    float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
    int3 sum = int3(0, 0, 0);
    for (uint role = 1u; role <= 4u; role++)
    {
        if ((roles & (1u << (role - 1u))) != 0u) sum += toFixed3(hitShare(f, role, alpha, beta, gamma));
    }
    return sum;
}

// A hit's contact force: contacter (velocity V, position and mass P) against the
// triangle (masses P0..P2, velocities v0..v2) at weights (alpha, beta, gamma),
// distance delta along normal nn; its force is fb (beams) plus a3 (accumulator).
// Strict, in the CPU reference's order (the test-only EB_DET_HITS build of hitForce
// keeps the frozen baseline's expressions instead).
float3 hitContact(float4 V, float4 P, float4 P0, float4 P1, float4 P2, float3 v0, float3 v1, float3 v2, float alpha,
                  float beta, float gamma, float pen, float3 nn, float3 fb, float3 a3, uint v)
{
    float3 vv = ssub3(V.xyz, sadd3(sadd3(sscale(v1, alpha), sscale(v2, beta)), sscale(v0, gamma)));
    float mt = nf(nf(smul(alpha, P1.w) + smul(beta, P2.w)) + smul(gamma, P0.w));
    float m = sdiv(smul(P.w, mt), nf(P.w + mt));
    return contactForce(sadd3(fb, a3), vv, m, nn, surfaceV(boxSurface(), v), pen, V.w);
}

// The force of the hit at record `at` (key `key`) of vehicle v: its node's force,
// with `sum` the fixed-point shares of every earlier hit on that node (as the
// accumulator would hold them had the hits been added in order), then the contact
// force from the geometry the finding lane stored; stored with `round`.
void hitForce(uint v, uint at, uint key, int3 sum, uint round, float3 origin, float r, uint beamOff, uint groupOff,
              uint groupCount, uint parity)
{
    uint h = gStatic[v * 4u].x + (key & 0xFFFFu);
    float3 nn = asfloat(uint3(gNext[at + 12u], gNext[at + 13u], gNext[at + 14u]));
    float pen = asfloat(gNext[at + 15u]);
    float3 vv = asfloat(uint3(gNext[at + 16u], gNext[at + 17u], gNext[at + 18u]));
    float m = asfloat(gNext[at + 19u]);
    float friction = asfloat(gNext[at + 20u]);
    // The accumulator as it would read with the earlier hits added (accForce). It
    // holds nothing else in this phase: the node phase took and cleared it, and
    // vehicle contact adds after the vehicles' steps.
    int3 acc = sum;
    float3 fb = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity,
                        false);
    float3 a3 = float3(float(acc.x), float(acc.y), float(acc.z)) * (1.0 / FIXED);
#ifdef EB_DET_HITS
    float3 f = contactForce(fb + a3, vv, m, nn, surfaceV(boxSurface(), v), pen, friction);
#else
    float3 f = contactForce(sadd3(fb, a3), vv, m, nn, surfaceV(boxSurface(), v), pen, friction);
#endif
    gNext[at + 3u] = asuint(f.x);
    gNext[at + 4u] = asuint(f.y);
    gNext[at + 5u] = asuint(f.z);
    gNext[at + 6u] = round;
}

// The computed hit at record `at` adds its shares to the accumulators.
void selfAddOne(uint at, uint key, uint nodeOff)
{
    float alpha = asfloat(gNext[at + 1u]);
    float beta = asfloat(gNext[at + 2u]);
    float gamma = nf(nf(1.0 - alpha) - beta);
    float3 f = asfloat(uint3(gNext[at + 3u], gNext[at + 4u], gNext[at + 5u]));
    accAdd(nodeOff + (key & 0xFFFFu), f);
    accAdd(gNext[at + 9u], hitShare(f, 2u, alpha, beta, gamma));
    accAdd(gNext[at + 10u], hitShare(f, 3u, alpha, beta, gamma));
    accAdd(gNext[at + 8u], hitShare(f, 4u, alpha, beta, gamma));
    cfAdd(nodeOff + (key & 0xFFFFu), f);
    cfAdd(gNext[at + 9u], hitShare(f, 2u, alpha, beta, gamma));
    cfAdd(gNext[at + 10u], hitShare(f, 3u, alpha, beta, gamma));
    cfAdd(gNext[at + 8u], hitShare(f, 4u, alpha, beta, gamma));
}

// One round over vehicle v's hits (lane `sub` of `lanes`): every open hit whose
// earlier hits (smaller keys) on its node were computed in an earlier round is
// computed now. `round` 0 takes any hit whose earlier hits on its node are computed
// (the one-lane pass, in key order). Returns the hits left open.
uint selfRound(uint v, uint sub, uint lanes, uint round, float3 origin, float r, uint beamOff, uint groupOff,
               uint groupCount, uint parity)
{
    uint count = uint(gAcc[hitAcc(v)].load());
    if (count > HIT_CAP) return 0u;
    uint base = hitList(v);
    uint nodeOff = gStatic[v * 4u].x;
    uint open = 0u;
    for (uint i = sub, iStop = EB_END(i, count, lanes); i < iStop; i += lanes)
    {
        uint at = base + i * HIT_WORDS;
        if (gNext[at + 6u] != 0u) continue;
        uint key = gNext[at];
        uint h = nodeOff + (key & 0xFFFFu);
        if (selfTwoBody(v))
        {
            // Two-body (spec §36): no hit reads another's force.
            float3 f2 = selfTwoBodyForce(at, h, beamOff, groupOff, groupCount, parity);
            gNext[at + 3u] = asuint(f2.x);
            gNext[at + 4u] = asuint(f2.y);
            gNext[at + 5u] = asuint(f2.z);
            gNext[at + 6u] = round != 0u ? round : SELF_ROUNDS + 1u;
            selfAddOne(at, key, nodeOff);
            continue;
        }
        int3 sum = int3(0, 0, 0);
        bool ready = true;
        for (uint j = 0u, jStop = EB_END(j, count, 1u); j < jStop; j++)
        {
            uint bj = base + j * HIT_WORDS;
            uint kj = gNext[bj];
            if (kj >= key) continue;
            uint role = hitRoles(bj, kj, h, nodeOff);
            if (role == 0u) continue;
            uint rj = gNext[bj + 6u];
            if (rj == 0u || (round != 0u && rj == round))
            {
                ready = false;
                break;
            }
            sum += hitShareFixed(bj, role);
        }
        if (!ready)
        {
            open++;
            continue;
        }
        uint done = round != 0u ? round : SELF_ROUNDS + 1u;
        if (gNext[at + 7u] != 0u && all(sum == int3(0, 0, 0)))
            gNext[at + 6u] = done;
        else
            hitForce(v, at, key, sum, done, origin, r, beamOff, groupOff, groupCount, parity);
        // Its shares now: no later hit reads the accumulators (they sum the shares
        // from the records), so the adds need no order.
        selfAddOne(at, key, nodeOff);
    }
    return open;
}

// Vehicle v's self-collision hits of this step, after the lanes that found them
// (and resolvePending) passed a barrier: rounds of selfRound until no hit of any
// member is open (every lane of the workgroup runs them: barriers), then each hit's
// shares into the accumulators. `on`: this lane's partition has a vehicle. More
// hits than HIT_CAP: one lane runs the whole step's self-collision in order
// (selfSerial) instead.
void selfSerial(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity);

void selfApply(uint v, bool on, uint sub, uint lanes, uint lane, float3 origin, float r, uint beamOff, uint groupOff,
               uint groupCount, uint parity)
{
    // gsUniformHits: set by any lane that found a hit (tryContact), read after the
    // caller's barrier; none: nothing to apply, no further barrier.
    uint any = gsUniformHits;
    if (any == 0u) return;
    // A lane with hits still open after round k sets gsUniformOpen to k: the round
    // numbers only grow within a step, so reading "at least k" after the barrier
    // is the same for every lane even if a fast lane has moved on to round k + 1.
    uint go = 0u;
    for (uint round = 1u; round <= SELF_ROUNDS; round++)
    {
        uint open = on ? selfRound(v, sub, lanes, round, origin, r, beamOff, groupOff, groupCount, parity) : 0u;
        if (open != 0u) gsUniformOpen = round;
        sync();
        uint last = gsUniformOpen;
        go = last >= round ? 1u : 0u;
        if (go == 0u) break;
    }
    if (go != 0u)
    {
        // A dependency chain longer than the rounds: one lane per vehicle finishes
        // the open hits in key order.
        if (on && sub == 0u)
        {
            for (uint it = 0u; it <= HIT_CAP; it++)
            {
                if (selfRound(v, 0u, 1u, 0u, origin, r, beamOff, groupOff, groupCount, parity) == 0u) break;
            }
        }
        sync();
    }
    if (on && sub == 0u && uint(gAcc[hitAcc(v)].load()) > HIT_CAP)
        selfSerial(v, origin, r, beamOff, groupOff, groupCount, parity);
}



// More hits in a step than the hit list holds: one lane runs vehicle v's
// self-collision of this step exactly as the specification orders it (tested
// triangles in file order, every candidate in node order, each hit added at once).
// Slow, and only for a step with more than HIT_CAP hits.
void selfSerial(uint v, float3 origin, float r, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    for (uint t = 0u, tStop = EB_END(t, r1.y, 1u); t < tStop; t++)
    {
        uint gt = r1.x + t;
        if (asint(gMisc[gP.vehicles + gt].x) > 0) continue;
        TriGeo T = triangleGeo(gt, origin, r, parity);
        uint4 tn = T.tn;
        for (uint k = 0u, kStop = EB_END(k, r2.y, 1u); k < kStop; k++)
        {
            uint h = gLists[r2.x + k];
            if (h == tn.x || h == tn.y || h == tn.z) continue;
            float4 P = gNodes[h * 3u];
            float3 p = nf3(origin + P.xyz);
            if (any(p < T.lo) || any(p > T.hi)) continue;
            float alpha;
            float beta;
            float gamma;
            float delta;
            float3 nn;
            float pen;
            if ((T.clip & 1u) != 0u)
            {
                // Anti-clip: the test alone (tryContact recorded the sides).
                uint word;
                if (selfClip(T, h, p, origin, r, alpha, beta, gamma, pen, nn, word) != 1u) continue;
            }
            else
            {
                if (!triangleHit(T, p, r, alpha, beta, gamma, delta)) continue;
                nn = T.n;
                if (delta < 0.0)
                {
                    nn = -T.n;
                    delta = -delta;
                }
                pen = nf(r - delta);
            }
            gMisc[T.mi].w = 1u;
            float4 V = gNodes[h * 3u + 1u];
            float4 P0 = gNodes[tn.x * 3u];
            float4 P1 = gNodes[tn.y * 3u];
            float4 P2 = gNodes[tn.z * 3u];
            float3 f;
            if (selfTwoBody(v))
            {
                float3 vv = ssub3(V.xyz, sadd3(sadd3(sscale(gNodes[tn.y * 3u + 1u].xyz, alpha),
                                                     sscale(gNodes[tn.z * 3u + 1u].xyz, beta)),
                                               sscale(gNodes[tn.x * 3u + 1u].xyz, gamma)));
                f = selfTwoBodyCore(h, tn.x, tn.y, tn.z, alpha, beta, gamma, nn, pen, vv, V.w, beamOff,
                                    groupOff, groupCount, parity);
            }
            else
            {
                float3 fb = beamSum(gNodes[h * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + h], beamOff, groupOff,
                                    groupCount, parity, false);
                f = hitContact(V, P, P0, P1, P2, gNodes[tn.x * 3u + 1u].xyz, gNodes[tn.y * 3u + 1u].xyz,
                               gNodes[tn.z * 3u + 1u].xyz, alpha, beta, gamma, pen, nn, fb, accForce(h), v);
            }
            accAdd(h, f);
            accAdd(tn.y, hitShare(f, 2u, alpha, beta, gamma));
            accAdd(tn.z, hitShare(f, 3u, alpha, beta, gamma));
            accAdd(tn.x, hitShare(f, 4u, alpha, beta, gamma));
            cfAdd(h, f);
            cfAdd(tn.y, hitShare(f, 2u, alpha, beta, gamma));
            cfAdd(tn.z, hitShare(f, 3u, alpha, beta, gamma));
            cfAdd(tn.x, hitShare(f, 4u, alpha, beta, gamma));
        }
    }
}

// Vehicle v's pending beams of this step's beam phase (breakException), on one lane
// after a barrier: each beam's update again, in spawn order, with the exception's
// live-beam counts as the sequential pass sees them.
void resolvePending(uint v, uint parity)
{
    uint n = uint(gAcc[hitAcc(v) + 1u].load());
    if (n == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint4 r2 = gStatic[v * 4u + 2u];
    bool fast = (gStatic[v * 4u + 3u].x & FLAG_FAST_INV_SQRT) != 0u;
    gResolving = 1u;
    if (n <= PEND_CAP)
    {
        uint base = pendList(v);
        for (uint i = 1u, iStop = EB_END(i, n, 1u); i < iStop; i++)
        {
            uint x = gNext[base + i];
            uint j = i;
            uint jGuard = 0u;
            while (j > 0u && gNext[base + j - 1u] > x)
            {
                EB_PASS(jGuard);
                gNext[base + j] = gNext[base + j - 1u];
                j--;
            }
            gNext[base + j] = x;
        }
        for (uint i = 0u, iStop = EB_END(i, n, 1u); i < iStop; i++)
        {
            uint j = gNext[base + i];
            beamUpdate(r0.z + j, j, fast, r2.z, parity);
        }
    }
    else
    {
        for (uint j = 0u, jStop = EB_END(j, r0.w, 1u); j < jStop; j++)
        {
            if (gBeams[(r0.z + j) * 2u + 1u].w == 2.0) beamUpdate(r0.z + j, j, fast, r2.z, parity);
        }
    }
    gResolving = 0u;
}



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
        return ebLoopCapFlush();
    }
#endif
    // A vehicle of a contact group this frame is stepped by its group's workgroup.
    if (lane == 0u) gsUniformLanes = gMisc[vvRec(v, VV_FLAGS)].y;
    sync();
    if (gsUniformLanes != 0u) return ebLoopCapFlush();
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
        for (uint w = lane, wStop = EB_END(w, aero0.y, LANES); w < wStop; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif

    // Origin rebase: every lane reads node 0 before any lane moves it.
    float3 first = gNodes[nodeOff * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
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
        // Self-collision hits or pending beams possible (selfApply's barriers).
        gsUniformLanes = triCount > 0u ? 1u : 0u;
        coupleBegin();
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
    uint selfOn = gsUniformLanes;
    uint coupleOn = gsUniformCouple;

    for (uint s = 0u, sStop = EB_END(s, gP.steps, 1u); s < sStop; s++)
    {
        coupleStep(v, lane, s, coupleOn);
        uint step = stepBase + s;
        uint parity = step & 1u;
        bool hits = gsHit != 0u;
#ifdef EB_VV_SOLO
        for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
            nodeUpdateSolo(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
#else
#ifdef EB_ABLATE
        uint nodeEnd = (gP.ablate & 2u) != 0u ? 0u : ((gP.ablate & 4u) != 0u ? min(nodeCount, LANES) : nodeCount);
#else
        uint nodeEnd = nodeCount;
#endif
        for (uint i = lane, iStop = EB_END(i, nodeEnd, LANES); i < iStop; i += LANES)
            nodeUpdate(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
#endif

        for (uint g = lane, gStop = EB_END(g, groupCount, LANES); g < gStop; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
        if (candCount > 0u)
        {
            for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
            {
                int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
                if (triCount <= TRI_LIST && w <= 0) gsTested[gsTestedCount[parity].add(1u)] = t;
            }
        }
        for (uint b = lane; b < GRID_BUCKETS; b += LANES) gsCount[b].store(0u);
        if (lane == 0u)
        {
            gsPairCount.store(0u);
            gAcc[hitAcc(v)].store(0);
            gAcc[hitAcc(v) + 1u].store(0);
            clipVmaxClear(v, parity);
            gsUniformHits = 0u;
            gsUniformOpen = 0u;
        }
        sync();
        if (lane == 0u) grabApply(nodeOff, nodeCount, origin);
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
            for (uint c = lane, cStop = EB_END(c, candCount, LANES); c < cStop; c += LANES)
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
        for (uint w = lane, wStop = EB_END(w, aero0.y, LANES); w < wStop; w += LANES) wingForces(aero0.x + w, origin);
        if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
        if ((flagsV & FLAG_KEYS) != 0u)
        {
            for (uint c = lane, cStop = EB_END(c, aero0.w, LANES); c < cStop; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
        }
        else
        {
            for (uint c = lane, cStop = EB_END(c, aero0.w, LANES); c < cStop; c += LANES) commandUpdate(aero0.z + c);
        }
        sync();
        {
            uint4 ext = gStatic[gP.vehExtBase + v];
#ifdef EB_ABLATE
            if ((gP.ablate & 8u) != 0u) ext.w = 0u;
#endif
            for (uint k = lane, kStop = EB_END(k, ext.w, LANES); k < kStop; k += LANES) wingGather(ext.z + k);
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
        for (uint j = lane, jStop = EB_END(j, beamEnd, LANES); j < jStop; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
        // Without the grid this step: list the candidate pairs now (lanes in
        // reverse, so those with one beam fewer start first).
        uint listedB = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        bool early = candCount > 0u && !(cell > 0.0 && listedB * candCount > BRUTE_PAIRS_PER_LANE * LANES);
        if (early)
        {
            uint lanes = clamp(LANES / max(listedB, 1u), 1u, candCount);
            uint stride = (LANES / lanes) * lanes;
            uint rl = LANES - 1u - lane;
            for (uint p = rl, pStop = EB_END(p, listedB * lanes, stride); p < pStop && rl < stride; p += stride)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / lanes] : p / lanes;
                trianglePairs(t, triOff + t, p % lanes, lanes, origin, range, cell, candOff, candCount, parity);
            }
        }
        sync();
        uint gridOn = gsUniformLanes;
        // This step's deferred breaks, in spawn order (before selfApply's barrier).
        if (lane == 0u) resolvePending(v, parity);
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
            for (uint k = 0u, kStop = EB_END(k, per, 1u); k < kStop; k++) sum += gsCount[lane * per + k].load();
            gsPart[lane] = sum;
            sync();
            if (lane < GRID_BLOCKS)
            {
                uint run = 0u;
                for (uint k = 0u, kStop = EB_END(k, block, 1u); k < kStop; k++)
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
            for (uint k = 0u, kStop = EB_END(k, per, 1u); k < kStop; k++)
            {
                gsOff[lane * per + k] = run;
                run += gsCount[lane * per + k].load();
            }
            sync();
            if (cell > 0.0)
            {
                for (uint c = lane, cStop = EB_END(c, candCount, LANES); c < cStop; c += LANES)
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
            for (uint i = lane, iStop = EB_END(i, pairs, LANES); i < iStop; i += LANES)
            {
                uint e = gsPairs[i];
                tryContact(triangleGeo(triOff + (e >> 16), origin, range, parity), gLists[candOff + (e & 0xFFFFu)], origin,
                           range, beamOff, groupOff, groupCount, parity);
            }
        }
        else if (candCount > 0u && cell > 0.0 && gridOn != 0u)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            for (uint p = lane, pStop = EB_END(p, listed * GRID_SUBS, LANES); p < pStop; p += LANES)
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
            for (uint p = lane, pStop = EB_END(p, listed * lanes, stride); p < pStop && lane < stride; p += stride)
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
            for (uint p = lane, pStop = EB_END(p, triCount * lanes, stride); p < pStop && lane < stride; p += stride)
                triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff,
                             groupOff, groupCount, parity);
        }
        if (selfOn != 0u)
        {
            sync();
            selfApply(v, true, lane, LANES, lane, origin, range, beamOff, groupOff, groupCount, parity);
        }
#ifdef EB_AERO
        // Slide nodes, after this step's beams and self-collision, a lane each.
        if ((flagsV & FLAG_SLIDES) != 0u)
        {
            uint count = min(aero2.y, gP.slideTotal - min(aero2.x, gP.slideTotal));
            for (uint k = lane, kStop = EB_END(k, count, LANES); k < kStop; k += LANES) slideNode(aero2.x + k, origin, groupOff, parity);
        }
#endif
        sync();
    }
    coupleEnd(v, lane, coupleOn);
    if (gP.steps > 0u && candCount > 0u)
    {
        for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES) triangleFinish(triOff + t);
    }
#ifdef EB_AERO
    if ((flagsV & FLAG_SLIDES) != 0u)
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        for (uint k = lane, kStop = EB_END(k, ext.y, LANES); k < kStop; k += LANES) slideSettle(ext.x + k);
    }
#endif
    // The last step's group breaks, for a read-back (the next node phase applies
    // them to the forces).
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint last = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane, jStop = EB_END(j, beamCount, LANES); j < jStop; j += LANES) groupDrops(beamOff + j, beamOff, groupOff, last, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
#ifdef EB_VV_SOLO
    // The last step's positions (each earlier step's were checked at the start of
    // the next step's node phase).
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
    {
        float3 p = gNodes[(nodeOff + i) * 3u].xyz;
        if (any(abs(p - gsVvBounds[0]) > gsVvBounds[1])) vvReportOutside(v);
    }
#endif
    ebLoopCapFlush();
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
        return ebLoopCapFlush();
    }
    i -= gP.totalNodes;
    if (i < gP.groupTotal)
    {
        // Every vehicle has the same step parity.
        gGroups[groupSlot(vehicleStep(0u) & 1u, 0u, i)].store(NO_GROUP);
        return ebLoopCapFlush();
    }
    i -= gP.groupTotal;
    if (i < gP.vehicles * GRID_BUCKETS)
    {
        gAcc[gP.gridCountBase + i].store(0);
        if (i < gP.vehicles)
        {
            gAcc[gP.overflowBase + i].store(0);
            gAcc[gP.testedBase + i * 2u + (vehicleStep(i) & 1u)].store(0);
            gAcc[hitAcc(i)].store(0);
            gAcc[hitAcc(i) + 1u].store(0);
            clipVmaxClear(i, vehicleStep(i) & 1u);
        }
    }
    ebLoopCapFlush();
}

// Beams, then grid filling (candidates), then triangles' throttle and tested list.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_beams(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    // The node grab, after this step's wide_nodes (one thread).
    if (i == 0u) grabApplyWide();
    if (i < gP.totalBeams)
    {
        uint v = gStatic[gP.beamBase + i * 3u].z >> 16;
        Vehicle x = vehicleAt(v);
        beamUpdate(i, i - x.beamOff, (x.flags & FLAG_FAST_INV_SQRT) != 0u, x.groupOff, vehicleStep(v) & 1u);
        return ebLoopCapFlush();
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
        return ebLoopCapFlush();
    }
    i -= gP.totalCands;
    if (i < gP.totalTris)
    {
        // The last step's throttle, and this step's tested list (independent of the
        // beams).
        uint v = gStatic[gP.triBase + i].w;
        Vehicle x = vehicleAt(v);
        if (x.candCount == 0u) return ebLoopCapFlush();
        int w = wideStep() > 0u ? triangleFinish(i) : asint(gMisc[gP.vehicles + i].x);
        if (w <= 0)
        {
            uint k = uint(gAcc[gP.testedBase + v * 2u + (vehicleStep(v) & 1u)].add(1));
            gNext[gP.testedListBase + x.triOff + k] = i - x.triOff;
        }
    }
    ebLoopCapFlush();
}

// The node grab (grabApply) in the wide passes: the grabbed node's vehicle from its
// node record.
void grabApplyWide()
{
    uint4 g = gMisc[grabRec()];
    if (g.x == 0u || g.y >= gP.totalNodes) return;
    uint v = gStatic[gP.nodeInfoBase + g.y].w;
    Vehicle x = vehicleAt(v);
    grabApply(x.nodeOff, x.nodeCount, vehicleOrigin(v));
}

// triangleGrid with the offsets in gAcc.
void triangleGridWide(uint gt, uint sub, uint v, float3 origin, float r, float cell, uint candOff, uint candCount,
                      uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    TriGeo t = triangleGeo(gt, origin, r, parity);
    // The box's cells, counted in float: a stretched or non-finite triangle (a
    // flung node) must take the every-candidate path, not overflow an integer
    // product into an unbounded cell loop (a GPU hang).
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub, kStop = EB_END(k, candCount, GRID_SUBS); k < kStop; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub, iStop = EB_END(i, cells, GRID_SUBS); i < iStop; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        for (uint j = 0u, jStop = EB_END(j, i, 1u); j < jStop; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        uint n = min(uint(gAcc[gP.gridCountBase + v * GRID_BUCKETS + b].load()), WIDE_SLOTS);
        uint base = gP.slotBase16 + (v * GRID_BUCKETS + b) * WIDE_SLOTS;
        for (uint e = 0u, eStop = EB_END(e, n, 1u); e < eStop; e++)
        {
            float4 E = gBeams[base + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
    // Candidates past a full bucket: every triangle tests them (its share of lanes).
    uint over = uint(gAcc[gP.overflowBase + v].load());
    for (uint k = sub, kStop = EB_END(k, over, GRID_SUBS); k < kStop; k += GRID_SUBS)
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
    if (gt >= gP.totalTris) return ebLoopCapFlush();
    uint v = gStatic[gP.triBase + gt].w;
    Vehicle x = vehicleAt(v);
    if (x.candCount == 0u) return ebLoopCapFlush();
    uint step = vehicleStep(v);
    uint k = gt - x.triOff;
    if (k >= uint(gAcc[gP.testedBase + v * 2u + (step & 1u)].load())) return ebLoopCapFlush();
    uint t = x.triOff + gNext[gP.testedListBase + x.triOff + k];
    float3 origin = vehicleOrigin(v);
    if (x.cell > 0.0)
        triangleGridWide(t, sub, v, origin, x.range, x.cell, x.candOff, x.candCount, x.beamOff, x.groupOff,
                         x.groupCount, step & 1u);
    else
        triangleTest(t, sub, GRID_SUBS, origin, x.range, x.candOff, x.candCount, x.beamOff, x.groupOff,
                     x.groupCount, step & 1u);
    ebLoopCapFlush();
}

// After the triangles: a workgroup per vehicle resolves its deferred breaks and
// applies its self-collision hits in order (selfApply).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_hits(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    Vehicle x = vehicleAt(v);
    uint parity = vehicleStep(v) & 1u;
    if (lane == 0u)
    {
        resolvePending(v, parity);
        gsUniformHits = uint(gAcc[hitAcc(v)].load()) > 0u ? 1u : 0u;
        gsUniformOpen = 0u;
    }
    sync();
    uint on = gsUniformHits;
    if (on != 0u)
        selfApply(v, true, lane, 256u, lane, vehicleOrigin(v), x.range, x.beamOff, x.groupOff, x.groupCount, parity);
    ebLoopCapFlush();
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
    for (uint i = lane, iStop = EB_END(i, r0.y, 256u); i < iStop; i += 256u)
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
            for (int x = 0; x < min(span.x, int(VV_MAX_SPAN)); x++)
                for (int y = 0; y < min(span.y, int(VV_MAX_SPAN)); y++)
                    for (int z = 0; z < min(span.z, int(VV_MAX_SPAN)); z++)
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
    ebLoopCapFlush();
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
    ebLoopCapFlush();
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
        for (uint w = 0u, wStop = EB_END(w, gP.vehicles, 1u); w < wStop; w++)
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
    for (int x = 0; x < min(span.x, int(VV_MAX_SPAN)); x++)
        for (int y = 0; y < min(span.y, int(VV_MAX_SPAN)); y++)
            for (int z = 0; z < min(span.z, int(VV_MAX_SPAN)); z++)
            {
                int3 c = c0 + int3(x, y, z);
                uint b = vvHash(c);
                uint n = min(uint(gAcc[gP.vvGridCount + b].load()), VV_SLOTS);
                for (uint e = 0u, eStop = EB_END(e, n, 1u); e < eStop; e++)
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
        for (uint v = lane, vStop = EB_END(v, nv, 256u); v < vStop; v += 256u) vvFindPairs(v);
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
        for (uint b = lane, bGuard = 0u; b <= gP.vvGridMask; b += 256u)
        {
            EB_PASS(bGuard);
            gAcc[gP.vvGridCount + b].store(0);
        }
        if (lane == 0u)
        {
            vvArgs(nv, 0u);
            gAcc[vvCtl(VV_BIGS)].store(0);
        }
        return ebLoopCapFlush();
    }
    for (uint v = lane, vStop = EB_END(v, nv, 256u); v < vStop; v += 256u)
    {
        gAcc[gP.vvLabel + v].store(int(v));
        gMisc[vvRec(v, VV_FLAGS)].y = 0u;
    }
    sync();
    for (uint p = lane, pStop = EB_END(p, pairs, 256u); p < pStop; p += 256u)
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
        for (uint p = lane, pStop = EB_END(p, pairs, 256u); p < pStop; p += 256u)
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
        for (uint v = lane, vStop = EB_END(v, nv, 256u); v < vStop; v += 256u)
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
    for (uint v = lane, vStop = EB_END(v, nv, 256u); v < vStop; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u && vvLabel(v) == v)
            gMisc[vvRec(v, VV_FLAGS)].z = uint(gAcc[vvCtl(VV_GROUPS)].add(1));
    }
    sync();
    for (uint v = lane, vStop = EB_END(v, nv, 256u); v < vStop; v += 256u)
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
    for (uint g = g0, gStop = EB_END(g, g1, 1u); g < gStop; g++)
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
    for (uint g = g0, gStop = EB_END(g, g1, 1u); g < gStop; g++)
    {
        uint m = uint(gAcc[vvSum(g, 0u)].load());
        uint pts = uint(gAcc[vvSum(g, 2u)].load());
        // Too many members: none stepped as a group (they run alone; reported).
        gMisc[gP.vvGroups + g] = m <= VV_MAX_GROUP ? uint4(ra, m, rb, pts) : uint4(ra, 0u, rb, 0u);
        ra += m;
        rb += pts;
    }
    sync();
    for (uint v = lane, vStop = EB_END(v, nv, 256u); v < vStop; v += 256u)
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
    for (uint b = lane, bGuard = 0u; b <= gP.vvGridMask; b += 256u)
    {
        EB_PASS(bGuard);
        gAcc[gP.vvGridCount + b].store(0);
    }
    for (uint g = lane, gStop = EB_END(g, groups, 256u); g < gStop; g += 256u)
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
    ebLoopCapFlush();
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
        for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
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
    for (uint g = lane, gStop = EB_END(g, groupCount, LANES); g < gStop; g += LANES) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
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
    if (lane == 0u)
    {
        gAcc[hitAcc(v)].store(0);
        gAcc[hitAcc(v) + 1u].store(0);
        clipVmaxClear(v, parity);
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    sync();
    if (lane == 0u) grabApply(nodeOff, nodeCount, origin);
    if (lane == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    if (lane == 0u)
    {
        // Bit 0: the grid this step; bit 1: self-collision hits or pending beams
        // possible (selfApply's barriers).
        uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        gsUniformLanes = (cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * LANES ? 1u : 0u)
            | (triCount > 0u ? 2u : 0u);
    }
    if (cell > 0.0)
    {
        for (uint c = lane, cStop = EB_END(c, candCount, LANES); c < cStop; c += LANES)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (gsCount[b].add(1u) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = lane, wStop = EB_END(w, aero0.y, LANES); w < wStop; w += LANES) wingForces(aero0.x + w, origin);
    if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = lane, cStop = EB_END(c, aero0.w, LANES); c < cStop; c += LANES) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = lane, cStop = EB_END(c, aero0.w, LANES); c < cStop; c += LANES) commandUpdate(aero0.z + c);
    }
    sync();
    if (lane == 0u) wingApply(aero0.x, aero0.y);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint k = lane; k < KEYS; k += LANES) gNext[keySlot(v, parity ^ 1u, k)] = 0u;
    }
#endif
    for (uint j = lane, jStop = EB_END(j, beamCount, LANES); j < jStop; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    sync();
    uint flagsU = gsUniformLanes;
    uint gridOn = flagsU & 1u;
    if (lane == 0u) resolvePending(v, parity);
    if (gridOn != 0u)
    {
        uint per = GRID_BUCKETS / LANES;
        uint block = LANES / GRID_BLOCKS;
        uint sum = 0u;
        for (uint k = 0u, kStop = EB_END(k, per, 1u); k < kStop; k++) sum += gsCount[lane * per + k].load();
        gsPart[lane] = sum;
        sync();
        if (lane < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint k = 0u, kStop = EB_END(k, block, 1u); k < kStop; k++)
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
        for (uint k = 0u, kStop = EB_END(k, per, 1u); k < kStop; k++)
        {
            gsOff[lane * per + k] = run;
            run += gsCount[lane * per + k].load();
        }
        sync();
        if (cell > 0.0)
        {
            for (uint c = lane, cStop = EB_END(c, candCount, LANES); c < cStop; c += LANES)
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
        for (uint p = lane, pStop = EB_END(p, listed * GRID_SUBS, LANES); p < pStop; p += LANES)
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
        for (uint p = lane, pStop = EB_END(p, listed * lanes, stride); p < pStop && lane < stride; p += stride)
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
        for (uint p = lane, pStop = EB_END(p, triCount * lanes, stride); p < pStop && lane < stride; p += stride)
            triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
    }
    if ((flagsU & 2u) != 0u)
    {
        sync();
        selfApply(v, true, lane, LANES, lane, origin, range, beamOff, groupOff, groupCount, parity);
    }
#ifdef EB_AERO
    if (lane == LANES - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, origin, groupOff, parity);
#endif
    sync();
}

// --- A small group's members stepped side by side (k <= VV_PART members) ---
// The workgroup's lanes split into k partitions of P lanes (a power of two dividing
// GRID_BUCKETS); partition m steps member m exactly as contactMemberStep would,
// with its own shared counters and its self-collision tested list, grid counts and
// offsets in the per-vehicle global regions the wide passes use. Members do not
// read each other's state within their steps (contact reaches them only through
// the accumulators of the contact pass), so every member's step is the same, bit
// for bit, while the barriers are shared: k members cost about one member's step.
static const uint VV_PART = 4u;
groupshared Atomic<uint> gsTestedM[2u * VV_PART];
groupshared Atomic<uint> gsPairCountM[VV_PART];
groupshared uint gsBlkM[GRID_BLOCKS * VV_PART];
static const uint PAIR_PART = PAIR_LIST / VV_PART;

// Lanes per member for k members: LANES >> ceil(log2 k).
uint vvPartLanes(uint k)
{
    uint p = LANES;
    uint pGuard = 0u;
    while (p * k > LANES && p > 1u)
    {
        EB_PASS(pGuard);
        p = p >> 1u;
    }
    return p;
}

// triangleGrid with the offsets in gNext at offBase (a member's grid).
void triangleGridAt(uint gt, uint sub, float3 origin, float r, float cell, uint candOff, uint candCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, uint offBase)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    TriGeo t = triangleGeo(gt, origin, r, parity);
    float3 f0 = floor((t.lo - origin) / cell);
    float3 fs = floor((t.hi - origin) / cell) - f0 + float3(1.0, 1.0, 1.0);
    if (!(all(fs <= float(GRID_MAX_CELLS)) && fs.x * fs.y * fs.z <= float(GRID_MAX_CELLS)))
    {
        for (uint k = sub, kStop = EB_END(k, candCount, GRID_SUBS); k < kStop; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
    int3 c0 = int3(f0);
    int3 span = int3(fs);
    uint yz = uint(span.y * span.z);
    uint cells = uint(span.x) * yz;
    for (uint i = sub, iStop = EB_END(i, cells, GRID_SUBS); i < iStop; i += GRID_SUBS)
    {
        uint b = gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z))));
        bool dup = false;
        for (uint j = 0u, jStop = EB_END(j, i, 1u); j < jStop; j++)
            dup = dup || gridBucket(c0 + int3(int(j / yz), int((j / uint(span.z)) % uint(span.y)), int(j % uint(span.z)))) == b;
        if (dup) continue;
        for (uint e = gNext[offBase + b], eGuard = 0u; e < gNext[offBase + b + 1u]; e++)
        {
            EB_PASS(eGuard);
            float4 E = gBeams[gP.gridBase + candOff + e];
            float3 p = origin + E.xyz;
            if (any(p < t.lo) || any(p > t.hi)) continue;
            tryContact(t, asuint(E.w), origin, r, beamOff, groupOff, groupCount, parity);
        }
    }
}

// trianglePairs into member m's slice of the pair list.
void trianglePairsM(uint m, uint t, uint gt, uint sub, uint subs, float3 origin, float r, float cell, uint candOff,
                    uint candCount, uint parity)
{
    if (asint(gMisc[gP.vehicles + gt].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    // The box tryContact tests (triangleGeo's: with anti-clip, its reach).
    float reach = triReach(tn, r, parity);
    float3 lo = min(min(p0, p1), p2) - float3(reach, reach, reach);
    float3 hi = max(max(p0, p1), p2) + float3(reach, reach, reach);
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
    for (uint k = sub, kStop = EB_END(k, candCount, subs); k < kStop; k += subs)
    {
        uint h = gLists[candOff + k];
        if (h == tn.x || h == tn.y || h == tn.z) continue;
        float3 e = gNodes[h * 3u].xyz;
        if (!every)
        {
            float3 p = origin + e;
            if (any(p < lo) || any(p > hi)) continue;
            int3 c = int3(floor(e / cell));
            if (!(all(c >= c0) && all(c < c0 + span)))
            {
                uint b = gridBucket(c);
                bool covered = false;
                for (uint i = 0u, iStop = EB_END(i, cells, 1u); i < iStop; i++)
                    covered = covered
                        || gridBucket(c0 + int3(int(i / yz), int((i / uint(span.z)) % uint(span.y)), int(i % uint(span.z)))) == b;
                if (!covered) continue;
            }
        }
        float3 x = nf3(origin + e);
        if (any(x < lo) || any(x > hi)) continue;
        uint i = gsPairCountM[m].add(1u);
        if (i < PAIR_PART) gsPairs[m * PAIR_PART + i] = (t << 16) | k;
    }
}

void contactMembersStep(uint off, uint k, uint lane, uint s)
{
    uint P = vvPartLanes(k);
    uint m = lane / P;
    uint pl = lane % P;
    bool on = m < k;
    uint v = vvMember(off, on ? m : 0u);
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    uint nodeOff = r0.x;
    uint nodeCount = on ? r0.y : 0u;
    uint beamOff = r0.z;
    uint beamCount = on ? r0.w : 0u;
    uint triOff = r1.x;
    uint triCount = on ? r1.y : 0u;
    uint boxOff = r1.z;
    uint boxCount = r1.w;
    uint candOff = r2.x;
    uint candCount = on ? r2.y : 0u;
    uint groupOff = r2.z;
    uint groupCount = on ? r2.w : 0u;
    uint flagsV = r3.x;
    float ground = asfloat(r3.y);
    float range = asfloat(r3.z);
    float cell = asfloat(r3.w);
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;
#ifdef EB_AERO
    uint4 aero0 = gStatic[gP.aeroBase + v * 3u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 3u + 1u];
    uint4 aero2 = gStatic[gP.aeroBase + v * 3u + 2u];
    if (!on)
    {
        aero0 = uint4(0u, 0u, 0u, 0u);
        flagsV = flagsV & ~(FLAG_FUSEDRAG | FLAG_SLIDES | FLAG_KEYS);
    }
#endif
    uint4 misc = gMisc[v];
    float3 origin = asfloat(misc.xyz);
    uint step = misc.w + s;
    uint parity = step & 1u;
    bool pending = gMisc[vvRec(v, VV_FLAGS)].w != 0u;
    // The member's grid: counts (gAcc), offsets (gNext), as the wide passes keep them.
    uint countBase = gP.gridCountBase + v * GRID_BUCKETS;
    uint offBase = gP.gridOffBase + v * (GRID_BUCKETS + 1u);
    uint listBase = gP.testedListBase + triOff;
    if (lane == 0u)
    {
        vvClearCounters();
        vvClearSmall();
        gsUniformHits = 0u;
        gsUniformOpen = 0u;
    }
    if (on && pl == 0u)
    {
        gsTestedM[2u * m + parity].store(0u);
        gsPairCountM[m].store(0u);
        gAcc[hitAcc(v)].store(0);
        gAcc[hitAcc(v) + 1u].store(0);
        clipVmaxClear(v, parity);
    }
    sync();
    {
        float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 hi = -lo;
        bool has = false;
        for (uint i = pl, iStop = EB_END(i, nodeCount, P); i < iStop; i += P)
        {
            nodeUpdate(nodeOff + i, gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground,
                       boxOff, boxCount, beamOff, groupOff, groupCount, true);
            float3 x = nf3(origin + gNodes[(nodeOff + i) * 3u].xyz);
            lo = min(lo, x);
            hi = max(hi, x);
            has = true;
        }
        if (on && pl < 6u) gAcc[vvBoxKey(v, parity ^ 1u) + pl].store(pl < 3u ? VV_INT_MAX : VV_INT_MIN);
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
    for (uint g = pl, gStop = EB_END(g, groupCount, P); g < gStop; g += P) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
    if (candCount > 0u)
    {
        for (uint t = pl, tStop = EB_END(t, triCount, P); t < tStop; t += P)
        {
            int w = s > 0u ? triangleFinish(triOff + t) : asint(gMisc[gP.vehicles + triOff + t].x);
            if (s > 0u && pending)
            {
                w = 0;
                gMisc[gP.vehicles + triOff + t].x = 0u;
            }
            if (w <= 0) gNext[listBase + gsTestedM[2u * m + parity].add(1u)] = t;
        }
    }
    if (on && cell > 0.0)
    {
        for (uint b = pl; b < GRID_BUCKETS; b += P) gAcc[countBase + b].store(0);
    }
    sync();
    if (on && pl == 0u) grabApply(nodeOff, nodeCount, origin);
    if (on && pl == 0u && s > 0u) gMisc[vvRec(v, VV_FLAGS)].w = 0u;
    // This step's self-collision per member: the grid or (few pairs) the early list.
    uint listed = on ? gsTestedM[2u * m + parity].load() : 0u;
    bool gridM = cell > 0.0 && candCount > 0u && listed * candCount > BRUTE_PAIRS_PER_LANE * P;
    if (lane == 0u)
    {
        uint any = 0u;
        for (uint j = 0u, jStop = EB_END(j, k, 1u); j < jStop; j++)
        {
            uint vj = vvMember(off, j);
            uint4 q1 = gStatic[vj * 4u + 1u];
            uint4 q2 = gStatic[vj * 4u + 2u];
            float cj = asfloat(gStatic[vj * 4u + 3u].w);
            uint lj = gsTestedM[2u * j + (((gMisc[vj].w + s) & 1u))].load();
            if (cj > 0.0 && q2.y > 0u && lj * q2.y > BRUTE_PAIRS_PER_LANE * P) any = any | 1u;
            // Self-collision hits or pending beams possible (selfApply's barriers).
            if (q1.y > 0u) any = any | 2u;
        }
        gsUniformLanes = any;
    }
    if (gridM)
    {
        for (uint c = pl, cStop = EB_END(c, candCount, P); c < cStop; c += P)
        {
            float3 p = gNodes[gLists[candOff + c] * 3u].xyz;
            uint b = gridBucket(int3(floor(p / cell)));
            gNext[candOff + c] = b | (uint(gAcc[countBase + b].add(1)) << 10);
        }
    }
#ifdef EB_AERO
    for (uint w = pl, wStop = EB_END(w, aero0.y, P); w < wStop; w += P) wingForces(aero0.x + w, origin);
    if (on && pl == P - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint c = pl, cStop = EB_END(c, aero0.w, P); c < cStop; c += P) commandUpdateKeys(aero0.z + c, v, parity ^ 1u);
    }
    else
    {
        for (uint c = pl, cStop = EB_END(c, aero0.w, P); c < cStop; c += P) commandUpdate(aero0.z + c);
    }
    sync();
    {
        uint4 ext = gStatic[gP.vehExtBase + v];
        if (!on) ext.w = 0u;
        for (uint q = pl, qStop = EB_END(q, ext.w, P); q < qStop; q += P) wingGather(ext.z + q);
    }
    if ((flagsV & FLAG_KEYS) != 0u)
    {
        for (uint q = pl; q < KEYS; q += P) gNext[keySlot(v, parity ^ 1u, q)] = 0u;
    }
#endif
    for (uint j = pl, jStop = EB_END(j, beamCount, P); j < jStop; j += P) beamUpdate(beamOff + j, j, fast, groupOff, parity);
    bool early = candCount > 0u && !gridM;
    if (early)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        uint rl = P - 1u - pl;
        for (uint q = rl, qStop = EB_END(q, listed * lanes, stride); q < qStop && rl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            trianglePairsM(m, t, triOff + t, q % lanes, lanes, origin, range, cell, candOff, candCount, parity);
        }
    }
    sync();
    uint flagsU = gsUniformLanes;
    uint gridAny = flagsU & 1u;
    if (on && pl == 0u) resolvePending(v, parity);
    if (gridAny != 0u)
    {
        uint per = GRID_BUCKETS / P;
        uint block = P / GRID_BLOCKS;
        uint sum = 0u;
        if (gridM)
        {
            for (uint q = 0u, qStop = EB_END(q, per, 1u); q < qStop; q++) sum += uint(gAcc[countBase + pl * per + q].load());
        }
        gsPart[lane] = sum;
        sync();
        if (gridM && pl < GRID_BLOCKS)
        {
            uint run = 0u;
            for (uint q = 0u, qStop = EB_END(q, block, 1u); q < qStop; q++)
            {
                uint x = gsPart[m * P + pl * block + q];
                gsPart[m * P + pl * block + q] = run;
                run += x;
            }
            gsBlkM[m * GRID_BLOCKS + pl] = run;
        }
        sync();
        if (gridM && pl == 0u)
        {
            uint run = 0u;
            for (uint q = 0u; q < GRID_BLOCKS; q++)
            {
                uint x = gsBlkM[m * GRID_BLOCKS + q];
                gsBlkM[m * GRID_BLOCKS + q] = run;
                run += x;
            }
            gNext[offBase + GRID_BUCKETS] = run;
        }
        sync();
        if (gridM)
        {
            uint run = gsPart[lane] + gsBlkM[m * GRID_BLOCKS + pl / block];
            for (uint q = 0u, qStop = EB_END(q, per, 1u); q < qStop; q++)
            {
                gNext[offBase + pl * per + q] = run;
                run += uint(gAcc[countBase + pl * per + q].load());
            }
        }
        sync();
        if (gridM)
        {
            for (uint c = pl, cStop = EB_END(c, candCount, P); c < cStop; c += P)
            {
                uint br = gNext[candOff + c];
                uint h = gLists[candOff + c];
                gBeams[gP.gridBase + candOff + gNext[offBase + (br & (GRID_BUCKETS - 1u))] + (br >> 10)] =
                    float4(gNodes[h * 3u].xyz, asfloat(h));
            }
        }
        sync();
    }
    uint pairs = on ? gsPairCountM[m].load() : 0u;
    if (early && pairs <= PAIR_PART)
    {
        for (uint i = pl, iStop = EB_END(i, pairs, P); i < iStop; i += P)
        {
            uint e = gsPairs[m * PAIR_PART + i];
            tryContact(triangleGeo(triOff + (e >> 16), origin, range, parity), gLists[candOff + (e & 0xFFFFu)], origin, range,
                       beamOff, groupOff, groupCount, parity);
        }
    }
    else if (candCount > 0u && cell > 0.0 && gridM)
    {
        for (uint q = pl, qStop = EB_END(q, listed * GRID_SUBS, P); q < qStop; q += P)
        {
            uint t = gNext[listBase + q / GRID_SUBS];
            triangleGridAt(triOff + t, q % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff, groupOff,
                           groupCount, parity, offBase);
        }
    }
    else if (candCount > 0u && cell > 0.0)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        for (uint q = pl, qStop = EB_END(q, listed * lanes, stride); q < qStop && pl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            triangleGridBrute(triOff + t, q % lanes, lanes, origin, range, cell, candOff, candCount, beamOff,
                              groupOff, groupCount, parity);
        }
    }
    else if (candCount > 0u)
    {
        uint lanes = clamp(P / max(listed, 1u), 1u, candCount);
        uint stride = (P / lanes) * lanes;
        for (uint q = pl, qStop = EB_END(q, listed * lanes, stride); q < qStop && pl < stride; q += stride)
        {
            uint t = gNext[listBase + q / lanes];
            triangleTest(triOff + t, q % lanes, lanes, origin, range, candOff, candCount, beamOff, groupOff,
                         groupCount, parity);
        }
    }
    if ((flagsU & 2u) != 0u)
    {
        sync();
        selfApply(v, on, pl, P, lane, origin, range, beamOff, groupOff, groupCount, parity);
    }
#ifdef EB_AERO
    if (on && pl == P - 1u && (flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, origin, groupOff, parity);
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
        for (uint w = lane, wStop = EB_END(w, aero0.y, LANES); w < wStop; w += LANES) wingCheck(aero0.x + w, o);
    }
#endif
    float3 first = gNodes[r0.x * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop; i += LANES)
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
        for (uint t = lane, tStop = EB_END(t, r1.y, LANES); t < tStop; t += LANES)
        {
            triangleFinish(r1.x + t);
            if (pending) gMisc[gP.vehicles + r1.x + t].x = 0u;
        }
    }
    if (gP.steps > 0u && r2.w > 0u)
    {
        uint last = (misc.w + gP.steps - 1u) & 1u;
        for (uint j = lane, jStop = EB_END(j, r0.w, LANES); j < jStop; j += LANES) groupDrops(r0.z + j, r0.z, r2.z, last, true);
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
    // Anti-clip (§36): bit 0 the owner has it, bit 1 the triangle is anti-clip
    // only; `travel` its fastest node's travel over the step (with any anti-clip).
    uint clip;
    float travel;
};

// A contact triangle's frame from its world positions (ContactFrame::new).
void vvFrame(inout VvTri t, float3 p1, float3 p2)
{
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
}

VvTri vvTriangle(uint gt, float3 origin, float r)
{
    VvTri t;
    t.tn = gStatic[gP.triBase + gt];
    t.p0 = nf3(origin + gNodes[t.tn.x * 3u].xyz);
    float3 p1 = nf3(origin + gNodes[t.tn.y * 3u].xyz);
    float3 p2 = nf3(origin + gNodes[t.tn.z * 3u].xyz);
    vvFrame(t, p1, p2);
    float3 r3 = float3(r, r, r);
    t.lo = ssub3(min(min(t.p0, p1), p2), r3);
    t.hi = sadd3(max(max(t.p0, p1), p2), r3);
    uint z = gMisc[gP.vehicles + gt].z;
    t.clip = ((gStatic[t.tn.w * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u ? 1u : 0u) | ((z & TRI_ANTI_ONLY) != 0u ? 2u : 0u);
    t.travel = gP.clipMem != 0u ? smul(clipTriSpeed(t.tn), STEP) : 0.0;
    if (((z & TRI_BREAKABLE) != 0u && triangleDead(gt)) || t.clip == 2u)
    {
        t.lo = float3(3.0e38, 3.0e38, 3.0e38);
        t.hi = -t.lo;
    }
    return t;
}

// Candidate h (global node, world position p, of vehicle vh) against triangle gt of
// the owner: the hit test, the side vote, the contact force into the accumulators
// (h, and the triangle's nodes by their weights). Returns whether it hit.
// Two-body contact (spec §10.3): the pair (owner va, partner vh) opts in.
bool vvTwoBody(uint va, uint vh)
{
    return gP.vvSplit != 0u && ((gStatic[va * 4u + 3u].x | gStatic[vh * 4u + 3u].x) & FLAG_TWO_BODY) != 0u;
}
uint vvSplitTri(uint n) { return gP.vvSplit + n; }
uint vvSplitPoint(uint slot, uint n) { return gP.vvSplit + (1u + slot) * gP.totalNodes + n; }
// A two-body hit found by a walk: its nodes' hit counts of this owner's turn.
void vvSplitCount(VvTri t, uint h, uint slot)
{
    uint va = gStatic[gP.nodeInfoBase + t.tn.x].w;
    uint vh = gStatic[gP.nodeInfoBase + h].w;
    if (!vvTwoBody(va, vh)) return;
    gAcc[vvSplitTri(t.tn.x)].add(1);
    gAcc[vvSplitTri(t.tn.y)].add(1);
    gAcc[vvSplitTri(t.tn.z)].add(1);
    gAcc[vvSplitPoint(slot, h)].add(1);
}
float vvSplitK(uint at) { return float(max(gAcc[at].load(), 1)); }
bool vvImmovable(uint n) { return (gStatic[gP.nodeInfoBase + n].x & NODE_IMMOVABLE) != 0u; }

// Node n's acceleration (F/m)·w, component by component (explicit_contact.rs's
// order); 0 for an immovable node.
float3 vvAccel(float3 f, uint n, float m, float w)
{
    if (vvImmovable(n)) return float3(0.0, 0.0, 0.0);
    return sscale(float3(sdiv(f.x, m), sdiv(f.y, m), sdiv(f.z, m)), w);
}

// Every member's node force at the contact pass's start (two-body contact).
void vvSnapshot(uint off, uint k, uint parity, uint lane)
{
    for (uint mi = 0u, miStop = EB_END(mi, k, 1u); mi < miStop; mi++)
    {
        uint v = vvMember(off, mi);
        uint4 r0 = gStatic[v * 4u];
        for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop; i += LANES)
            gBeams[gP.vvSnap + r0.x + i] = float4(vvPointForce(r0.x + i, v, parity), 0.0);
    }
}

// Zeroes the two-body counts of the group's members' nodes (triangle counts and
// point slots 0..slots − 1), after a turn's hits are applied.
void vvSplitClear(uint off, uint k, uint slots, uint lane)
{
    if (gP.vvSplit == 0u) return;
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
    {
        uint4 r0 = gStatic[vvMember(off, m) * 4u];
        for (uint i = lane, iStop = EB_END(i, r0.y, LANES); i < iStop; i += LANES)
        {
            gAcc[vvSplitTri(r0.x + i)].store(0);
            for (uint sl = 0u; sl < VV_PART; sl++)
            {
                if (sl < slots) gAcc[vvSplitPoint(sl, r0.x + i)].store(0);
            }
        }
    }
}
// The contact point's force now (beams, accumulator, slide).
float3 vvPointForce(uint h, uint vh, uint parity)
{
    uint4 info = gStatic[gP.nodeInfoBase + h];
    uint4 rb0 = gStatic[vh * 4u];
    uint4 rb2 = gStatic[vh * 4u + 2u];
    float3 fh = sadd3(beamSum(gNodes[h * 3u + 2u].xyz, info, rb0.z, rb2.z, rb2.w, parity, false), accForce(h));
#ifdef EB_AERO
    uint slot1 = info.x >> NODE_SLOT_SHIFT;
    if (slot1 != 0u) fh = sadd3(fh, gBeams[gP.slideAcc + min(slot1 - 1u, max(gP.slotTotal, 1u) - 1u)].xyz);
#endif
    return fh;
}

// The pair has anti-clip (§36): the owner or the partner vehicle vh.
bool vvAnti(VvTri t, uint vh) { return (t.clip & 1u) != 0u || (gStatic[vh * 4u + 3u].x & FLAG_ANTI_CLIP) != 0u; }

// The candidate box test (explicit_contact.rs triangle_loop): x in the triangle's
// box grown by the range; with anti-clip also by the triangle's and the point's
// travel over the step and CLIP_DEPTH.
bool vvInBox(VvTri t, float3 x, uint h, bool anti)
{
    if (!anti) return !(any(x < t.lo) || any(x > t.hi));
    float3 v = gNodes[h * 3u + 1u].xyz;
    float pad = nf(nf(t.travel + smul(ssqrt(sdot(v, v)), STEP)) + CLIP_DEPTH);
    float3 p3 = float3(pad, pad, pad);
    return all(x >= ssub3(t.lo, p3)) && all(x <= sadd3(t.hi, p3));
}

// The side vote (spec §5.3) of h (of vehicle vh) at distance delta: true behind.
bool vvVote(VvTri t, uint h, uint vh, float delta)
{
    // A vote of h and its spawn-time beam neighbours (its incidence entries' far
    // ends, in beam order).
    uint4 info = gStatic[gP.nodeInfoBase + h];
    int score = delta >= 0.0 ? 3 : -3;
    if (info.z > 3u)
    {
        float3 ob = asfloat(gMisc[vh].xyz);
        uint base = gP.vvNbr + info.y;
        // Four independent loads at a time (the far ends are a contiguous list).
        uint k = 0u;
        for (uint kGuard = 0u; k + 4u <= info.z; k += 4u)
        {
            EB_PASS(kGuard);
            float3 y0 = gNodes[gLists[base + k] * 3u].xyz;
            float3 y1 = gNodes[gLists[base + k + 1u] * 3u].xyz;
            float3 y2 = gNodes[gLists[base + k + 2u] * 3u].xyz;
            float3 y3 = gNodes[gLists[base + k + 3u] * 3u].xyz;
            score += sdot(t.n, ssub3(nf3(ob + y0), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y1), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y2), t.p0)) >= 0.0 ? 1 : -1;
            score += sdot(t.n, ssub3(nf3(ob + y3), t.p0)) >= 0.0 ? 1 : -1;
        }
        uint kStop = EB_END(k, info.z, 1u);
        for (; k < kStop; k++)
            score += sdot(t.n, ssub3(nf3(ob + gNodes[gLists[base + k] * 3u].xyz), t.p0)) >= 0.0 ? 1 : -1;
    }
    return score < 0;
}

// Vehicle contact's anti-clip test (§36; explicit_contact.rs triangle_loop with
// clip_test): partner point h (vehicle vh) at p against owner triangle gt. As
// selfClip, the band's side by the vote. 0: no contact; 1: a hit; 2: a contact
// without force (anti-clip only, in front). `word` the side to record.
uint vvClip(VvTri t, uint gt, uint h, float3 p, uint vh, float r, out float alpha, out float beta, out float gamma,
            out float pen, out float3 nn, out uint word)
{
    float3 q = ssub3(p, t.p0);
    alpha = 0.0;
    beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = sdot(t.ra, q);
        beta = sdot(t.rb, q);
        delta = sdot(t.rn, q);
    }
    gamma = nf(nf(1.0 - alpha) - beta);
    pen = 0.0;
    nn = t.n;
    word = 0u;
    uint va = t.tn.w;
    uint tl = gt - gStatic[va * 4u + 1u].x;
    float side = clipHeld(h, va, tl);
    bool over = alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0;
    bool have = over;
    if (side == 0.0)
    {
        bool crossed = false;
        float3 oa = asfloat(gMisc[va].xyz);
        VvTri pf;
        pf.p0 = ssub3(t.p0, sscale(gNodes[t.tn.x * 3u + 1u].xyz, STEP));
        float3 b1 = ssub3(nf3(oa + gNodes[t.tn.y * 3u].xyz), sscale(gNodes[t.tn.y * 3u + 1u].xyz, STEP));
        float3 b2 = ssub3(nf3(oa + gNodes[t.tn.z * 3u].xyz), sscale(gNodes[t.tn.z * 3u + 1u].xyz, STEP));
        vvFrame(pf, b1, b2);
        float3 q0 = ssub3(ssub3(p, sscale(gNodes[h * 3u + 1u].xyz, STEP)), pf.p0);
        float a0 = 0.0;
        float b0 = 0.0;
        float d0 = 0.0;
        if (pf.solvable)
        {
            a0 = sdot(pf.ra, q0);
            b0 = sdot(pf.rb, q0);
            d0 = sdot(pf.rn, q0);
        }
        if ((d0 > 0.0 && delta < 0.0) || (d0 < 0.0 && delta > 0.0))
        {
            float sc = sdiv(d0, nf(d0 - delta));
            float a = nf(a0 + smul(nf(alpha - a0), sc));
            float b = nf(b0 + smul(nf(beta - b0), sc));
            float g = nf(nf(1.0 - a) - b);
            if (a >= 0.0 && b >= 0.0 && g >= 0.0)
            {
                crossed = true;
                side = d0 > 0.0 ? 1.0 : -1.0;
                alpha = a;
                beta = b;
                gamma = g;
                have = true;
            }
        }
        if (!crossed)
        {
            if (!(over && abs(delta) <= r)) return 0u;
            side = vvVote(t, h, vh, delta) ? -1.0 : 1.0;
        }
    }
    float d = side * delta;
    if (!have || !(d <= r && d >= -CLIP_DEPTH)) return 0u;
    word = clipWord(va, tl, side);
    nn = side > 0.0 ? t.n : -t.n;
    bool only = (t.clip & 2u) != 0u;
    if (only && d >= 0.0) return 2u;
    // At most r (explicit_contact.rs clip_depth): held behind, pushed as at the plane.
    pen = min(only ? -d : nf(r - d), r);
    return 1u;
}

// The find phase's hit test (once per tested pair): with anti-clip the contact's
// side is recorded, hit or not.
bool vvFind(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti);
// The hit test alone (a re-walk).
bool vvHitTest(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti);

bool vvTry(VvTri t, uint gt, uint h, float3 p, uint vh, float r, uint parity, uint slot)
{
    float alpha;
    float beta;
    float gamma;
    float pen;
    float3 nn;
    if (vvAnti(t, vh))
    {
        uint word;
        if (vvClip(t, gt, h, p, vh, r, alpha, beta, gamma, pen, nn, word) != 1u) return false;
    }
    else
    {
        float3 q = ssub3(p, t.p0);
        alpha = 0.0;
        beta = 0.0;
        float delta = 0.0;
        if (t.solvable)
        {
            alpha = sdot(t.ra, q);
            beta = sdot(t.rb, q);
            delta = sdot(t.rn, q);
        }
        gamma = nf(nf(1.0 - alpha) - beta);
        if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) return false;
        nn = t.n;
        if (vvVote(t, h, vh, delta))
        {
            nn = -t.n;
            delta = -delta;
        }
        pen = nf(r - delta);
    }
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
    float3 fh = vvPointForce(h, vh, parity);
    if (vvTwoBody(gStatic[gP.nodeInfoBase + t.tn.x].w, vh))
    {
        // Each node one of its k sub-bodies of this turn (mass m/k, force F/k); the
        // triangle point's mass 1/Σ w²/m; only the share m_t/(m_h + m_t) of the
        // contacter's force (at its first hit of the turn) is cancelled.
        // Inverse masses of the split sub-bodies (immovable: 0), the triangle
        // point's Σ w²·k/m, and the pair's relative acceleration from the forces at
        // the pass's start (explicit_contact.rs).
        float imh = vvImmovable(h) ? 0.0 : sdiv(vvSplitK(vvSplitPoint(slot, h)), P.w);
        float i1 = (alpha > 0.0 && !vvImmovable(t.tn.y)) ? smul(smul(alpha, alpha), sdiv(vvSplitK(vvSplitTri(t.tn.y)), P1.w)) : 0.0;
        float i2 = (beta > 0.0 && !vvImmovable(t.tn.z)) ? smul(smul(beta, beta), sdiv(vvSplitK(vvSplitTri(t.tn.z)), P2.w)) : 0.0;
        float i0 = (gamma > 0.0 && !vvImmovable(t.tn.x)) ? smul(smul(gamma, gamma), sdiv(vvSplitK(vvSplitTri(t.tn.x)), P0.w)) : 0.0;
        float inv = nf(imh + nf(nf(i1 + i2) + i0));
        m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
        float3 at = sadd3(sadd3(vvAccel(gBeams[gP.vvSnap + t.tn.y].xyz, t.tn.y, P1.w, alpha),
                                vvAccel(gBeams[gP.vvSnap + t.tn.z].xyz, t.tn.z, P2.w, beta)),
                          vvAccel(gBeams[gP.vvSnap + t.tn.x].xyz, t.tn.x, P0.w, gamma));
        fh = sscale(ssub3(vvAccel(gBeams[gP.vvSnap + h].xyz, h, P.w, 1.0), at), m);
    }
    float3 f = contactForce(fh, v, m, nn, surfaceV(boxSurface(), vh), pen, V.w);
    accAdd(h, f);
    accAdd(t.tn.y, -sscale(f, alpha));
    accAdd(t.tn.z, -sscale(f, beta));
    accAdd(t.tn.x, -sscale(f, gamma));
    cfAdd(h, f);
    cfAdd(t.tn.y, sscale(f, alpha));
    cfAdd(t.tn.z, sscale(f, beta));
    cfAdd(t.tn.x, sscale(f, gamma));
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

bool vvFind(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti)
{
    if (!anti) return vvInBand(t, p, r);
    float a;
    float b;
    float g;
    float pen;
    float3 nn;
    uint word;
    uint c = vvClip(t, gt, h, p, vh, r, a, b, g, pen, nn, word);
    if (c != 0u) clipRecord(h, word);
    return c == 1u;
}

bool vvHitTest(VvTri t, uint gt, uint h, float3 p, uint vh, float r, bool anti)
{
    if (!anti) return vvInBand(t, p, r);
    float a;
    float b;
    float g;
    float pen;
    float3 nn;
    uint word;
    return vvClip(t, gt, h, p, vh, r, a, b, g, pen, nn, word) == 1u;
}

// A tested triangle's marks (had candidates, hit; spec §8's throttle inputs): two
// plain words per triangle in gNext after the tested list, set by the triangle's
// sub-lanes and read and cleared by vvThrottleEnd. One word per mark, so lanes
// only ever store the same value to a word: a store to one component of a uint4
// record (the marks were .z and .w of the triangle's throttle record) may compile
// to a read-modify-write of the whole record (Metal), and a lane marking only a
// candidate then could undo another lane's hit mark (a rare nondeterministic
// throttle).
uint vvMarkAt(uint gt) { return gP.vvMembers + 5u * gP.vehicles + gP.totalTris + 2u * gt; }

void vvMark(uint gt, bool cand, bool hit)
{
    if (cand) gNext[vvMarkAt(gt)] = 1u;
    if (hit) gNext[vvMarkAt(gt) + 1u] = 1u;
}

// Hit list entry `i` of the group's list of this owner (q its counter parity).
uint vvHitAt(uint runOff, uint i) { return gP.vvHits + 2u * (2u * runOff + i); }

void vvTriangleCands(uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands, float r, uint parity,
                     uint q)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    bool cand = false;
    bool hit = false;
    for (uint e = sub, eStop = EB_END(e, cands, subs); e < eStop; e += subs)
    {
        float4 E = gBeams[gP.vvRun + runOff + e];
        uint h = asuint(E.w);
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        bool anti = vvAnti(t, vh);
        if (!vvInBox(t, E.xyz, h, anti)) continue;
        cand = true;
        if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) continue;
        hit = true;
        vvSplitCount(t, h, 0u);
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
            vvTry(t, gt, h, E.xyz, vh, r, parity, 0u);
        }
    }
    vvMark(gt, cand, hit);
}

// Owner va's n listed hits (triangle, node: two words each from `base`) in the
// specification's order (triangles in list order, then contact points in node
// order). A hit reads its contact point's force and adds to it and to the owner's
// triangle nodes; the owner's nodes are never contact points of its own hits, so
// only the hits on one contact point depend on each other: one lane per contact
// point runs its hits in triangle order (each reads the ones before), the points
// side by side.
void vvApplyOrdered(uint va, uint base, uint n, float r, uint parity, uint lane, uint lanes, uint slot)
{
    float3 oa = asfloat(gMisc[va].xyz);
    for (uint i = lane, iStop = EB_END(i, n, lanes); i < iStop; i += lanes)
    {
        uint gt = gNext[base + 2u * i];
        uint h = gNext[base + 2u * i + 1u];
        bool first = true;
        for (uint j = 0u, jStop = EB_END(j, n, 1u); j < jStop; j++)
        {
            if (gNext[base + 2u * j + 1u] == h && gNext[base + 2u * j] < gt)
            {
                first = false;
                break;
            }
        }
        if (!first) continue;
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        for (uint it = 0u, itStop = EB_END(it, n, 1u); it < itStop; it++)
        {
            VvTri t = vvTriangle(gt, oa, r);
            vvTry(t, gt, h, nf3(asfloat(gMisc[vh].xyz) + gNodes[h * 3u].xyz), vh, r, parity, slot);
            // The point's next hit: the smallest later triangle.
            uint next = 0xFFFFFFFFu;
            for (uint j = 0u, jStop = EB_END(j, n, 1u); j < jStop; j++)
            {
                uint gj = gNext[base + 2u * j];
                if (gNext[base + 2u * j + 1u] == h && gj > gt && gj < next) next = gj;
            }
            if (next == 0xFFFFFFFFu) break;
            gt = next;
        }
    }
}

// The listed hits of owner va, in order (vvApplyOrdered).
void vvApplyHits(uint va, uint runOff, uint points, uint hits, float r, uint parity, uint lane)
{
    vvApplyOrdered(va, vvHitAt(runOff, 0u), min(hits, 2u * points), r, parity, lane, LANES, 0u);
}


// A tested triangle's throttle after its tests (a hit → 0, no candidate → + 1).
void vvThrottleEnd(uint gt)
{
    uint4 c = gMisc[gP.vvTri + gt];
    uint mark = vvMarkAt(gt);
    bool had = gNext[mark] != 0u;
    bool hit = gNext[mark + 1u] != 0u;
    gNext[mark] = 0u;
    gNext[mark + 1u] = 0u;
    int cwait = asint(c.x);
    if (hit) cwait = 0;
    else if (!had) cwait = min(cwait + 1, throttleCap(gt));
    gMisc[gP.vvTri + gt] = uint4(asuint(cwait), c.y, 0u, 0u);
}

// Triangle gt of owner va: its box grown by the range overlaps the box `lo..hi`
// (the partners' boxes' union), so it may have candidates.
// `extra`: anti-clip's reach past the range (a bound: CLIP_DEPTH, the owner's and
// the partners' fastest travel, and a millimetre for rounding).
bool vvNearBox(uint gt, uint va, float r, float3 blo, float3 bhi, float extra)
{
    uint4 tn = gStatic[gP.triBase + gt];
    float3 o = asfloat(gMisc[va].xyz);
    float3 p0 = nf3(o + gNodes[tn.x * 3u].xyz);
    float3 p1 = nf3(o + gNodes[tn.y * 3u].xyz);
    float3 p2 = nf3(o + gNodes[tn.z * 3u].xyz);
    float3 r3 = float3(r, r, r) + float3(extra, extra, extra);
    return vvOverlap(ssub3(min(min(p0, p1), p2), r3), sadd3(max(max(p0, p1), p2), r3), blo, bhi);
}

// Anti-clip's prefilter reach past the range (vvNearBox, the candidate runs): a
// bound on CLIP_DEPTH plus the travel of the triangle (the owner's fastest node)
// and of the point (`vmaxB` the partners' fastest), plus a millimetre.
float vvClipExtra(uint va, float vmaxB, uint parity)
{
    if (gP.clipMem == 0u) return 0.0;
    return CLIP_DEPTH + (clipVmax(va, parity) + vmaxB) * STEP + 0.001;
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
    if (gP.vvSplit != 0u)
    {
        vvSnapshot(off, k, parity, lane);
        sync();
    }
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint prevTri = 0u;
    uint prevListed = 0u;
    uint prevVa = 0u;
    float prevRange = 0.0;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
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
        for (uint i = lane, iStop = EB_END(i, prevListed, LANES); i < iStop; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
#ifdef EB_ABLATE
        if ((gP.ablate & 1024u) == 0u)
#endif
        uint prevHits = a > 0u ? gsVvHits[q ^ 1u].load() : 0u;
        if (a > 0u) vvApplyHits(prevVa, runOff, points, prevHits, prevRange, parity, lane);
        for (uint b = lane, bStop = EB_END(b, k, LANES); b < bStop; b += LANES)
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
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            if (m == a || gNext[vvResetAt(off, m)] == 0u) continue;
            uint4 r1 = gStatic[vvMember(off, m) * 4u + 1u];
            for (uint t = lane, tStop = EB_END(t, r1.y, LANES); t < tStop; t += LANES) gMisc[gP.vvTri + r1.x + t].x = 0u;
        }
        // The previous owner's two-body counts (its hits were applied in phase S).
        if (prevHits > 0u) vvSplitClear(off, k, 1u, lane);
        uint4 ra1 = gStatic[va * 4u + 1u];
        float range = asfloat(gStatic[va * 4u + 3u].z);
        // The partners' boxes' union (their contact points all lie in it).
        float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
        float3 phi = -plo;
        float vmaxB = 0.0;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
            uint kb = vvBoxKey(vvMember(off, m), parity);
            plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
            phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
            if (gP.clipMem != 0u) vmaxB = max(vmaxB, clipVmax(vvMember(off, m), parity));
        }
        float extra = vvClipExtra(va, vmaxB, parity);
        for (uint t = lane, tStop = EB_END(t, ra1.y, LANES); t < tStop; t += LANES)
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
            if (!vvNearBox(ra1.x + t, va, range, plo, phi, extra))
            {
                gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(min(cwait + 1, throttleCap(ra1.x + t))), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + ra1.x + t] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvListed[q].add(1u))] = t;
        }
#ifdef EB_ABLATE
        if (offered > 0u && (gP.ablate & 4096u) == 0u)
#else
        if (offered > 0u)
#endif
        {
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(blo, r3);
            float3 chi = sadd3(bhi, r3);
            // Anti-clip: the point's own travel (per point) on top of the bound.
            float ownerReach = gP.clipMem != 0u ? CLIP_DEPTH + clipVmax(va, parity) * STEP + 0.001 : 0.0;
            for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
            {
                if (m == a || gNext[vvPartnerAt(off, m)] == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                for (uint i = lane, iStop = EB_END(i, cp.y, LANES); i < iStop; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    float ex = 0.0;
                    if (gP.clipMem != 0u)
                    {
                        float3 vh3 = gNodes[h * 3u + 1u].xyz;
                        ex = ownerReach + sqrt(dot(vh3, vh3)) * STEP;
                    }
                    float3 e3 = float3(ex, ex, ex);
                    if (any(x < clo - e3) || any(x > chi + e3)) continue;
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
#ifdef EB_ABLATE
        if ((gP.ablate & 2048u) != 0u) listed = 0u;
#endif
        for (uint p = lane, pStop = EB_END(p, listed * subs, LANES); p < pStop; p += LANES)
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
    for (uint i = lane, iStop = EB_END(i, prevListed, LANES); i < iStop; i += LANES) vvThrottleEnd(prevTri + gNext[vvTestedAt(prevTri + i)]);
    uint lastHits = k > 0u ? gsVvHits[(k - 1u) & 1u].load() : 0u;
    if (k > 0u) vvApplyHits(prevVa, runOff, points, lastHits, prevRange, parity, lane);
    if (gP.vvSplit != 0u)
    {
        sync();
        if (lastHits > 0u) vvSplitClear(off, k, 1u, lane);
    }
    // (The counters are cleared by the next step's first member step, before its
    // first barrier.)
}

// --- The contact pass of a small group (k <= VV_PART owners), owners side by side ---
// The same results as vvPass's owner-by-owner pass. What depends on the owner order
// is only the forces: a hit reads its candidate's current force. Everything else
// reads positions, which the contact pass does not move, or commutes: (S) every
// owner's partner scan (symmetric: a fast approach marks both), (P) every owner's
// throttle step and candidates, (T) every owner's triangle tests into its own hit
// list, each with one barrier for all owners. The throttle resets need no order
// either: an owner whose partner approaches fast resets its own triangles (as its
// own scan finds the same partner), and a later owner's reset of an earlier owner's
// triangles, which in vvPass comes after that owner's throttle update, is applied
// after it here (gsVvFix). Then the hits are applied in levels: an owner joins the
// level after every earlier owner with hits whose vehicles it may touch (itself and
// its partners: a hit reads and adds to a partner's contact point and adds to the
// owner's triangle nodes), so each owner's hits see exactly the earlier owners'
// forces, as in vvPass. Forces are fixed-point sums, so the order inside a level
// does not matter. An owner whose hits overflow its list re-walks its triangles at
// its level, applying directly (vvPass's overflow path).
groupshared uint gsVvPair[VV_PART * VV_PART];
groupshared Atomic<uint> gsVvOff4[VV_PART];
groupshared Atomic<uint> gsVvList4[VV_PART];
groupshared Atomic<uint> gsVvCand4[VV_PART];
groupshared Atomic<uint> gsVvHit4[VV_PART];
groupshared uint gsVvReset4[VV_PART];
groupshared uint gsVvFix[VV_PART];

void vvClearSmall()
{
    for (uint a = 0u; a < VV_PART; a++)
    {
        gsVvOff4[a].store(0u);
        gsVvList4[a].store(0u);
        gsVvCand4[a].store(0u);
        gsVvHit4[a].store(0u);
        gsVvReset4[a] = 0u;
        gsVvFix[a] = 0u;
        for (uint b = 0u; b < VV_PART; b++) gsVvPair[a * VV_PART + b] = 0u;
    }
}

// Owner a's candidate run and hit list (each owner its own, within the group's
// regions, which hold four times the group's points).
uint vvRunSmall(uint runOff, uint points, uint a) { return gP.vvRun + 4u * runOff + a * points; }
uint vvHitSmall(uint runOff, uint points, uint a, uint i) { return gP.vvHits + 2u * (8u * runOff + 2u * a * points + i); }

// vvTriangleCands into owner a's own list (capacity: twice the group's points, as
// vvPass's); on overflow the list keeps its first entries and the owner re-walks at
// its level.
void vvTriangleCandsSmall(uint a, uint va, uint gt, uint sub, uint subs, uint runOff, uint points, uint cands,
                          float r)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    bool cand = false;
    bool hit = false;
    uint run = vvRunSmall(runOff, points, a);
    for (uint e = sub, eStop = EB_END(e, cands, subs); e < eStop; e += subs)
    {
        float4 E = gBeams[run + e];
        uint h = asuint(E.w);
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        bool anti = vvAnti(t, vh);
        if (!vvInBox(t, E.xyz, h, anti)) continue;
        cand = true;
        if (!vvFind(t, gt, h, E.xyz, vh, r, anti)) continue;
        hit = true;
        vvSplitCount(t, h, a);
        uint i = gsVvHit4[a].add(1u);
        if (i < 2u * points)
        {
            gNext[vvHitSmall(runOff, points, a, i)] = gt;
            gNext[vvHitSmall(runOff, points, a, i) + 1u] = h;
        }
    }
    vvMark(gt, cand, hit);
}

// The overflow path: owner a's triangle walk again, applying each hit at once.
void vvTriangleCandsApply(uint va, uint gt, uint sub, uint subs, uint run, uint cands, float r, uint parity, uint slot)
{
    VvTri t = vvTriangle(gt, asfloat(gMisc[va].xyz), r);
    for (uint e = sub, eStop = EB_END(e, cands, subs); e < eStop; e += subs)
    {
        float4 E = gBeams[run + e];
        uint h = asuint(E.w);
        uint vh = gStatic[gP.nodeInfoBase + h].w;
        bool anti = vvAnti(t, vh);
        if (!vvInBox(t, E.xyz, h, anti)) continue;
        if (!vvHitTest(t, gt, h, E.xyz, vh, r, anti)) continue;
        vvTry(t, gt, h, E.xyz, vh, r, parity, slot);
    }
}


// Owner a's level: 1 + the level of every earlier owner with hits whose vehicles
// ({itself} and its partners) meet a's (0: a has no hits).
uint vvLevel(uint a, uint k, uint points)
{
    uint lv[VV_PART];
    uint mine = 0u;
    for (uint b = 0u, bStop = EB_END(b, k, 1u); b < bStop; b++)
    {
        lv[b] = 0u;
        if (gsVvHit4[b].load() == 0u) continue;
        uint l = 1u;
        for (uint c = 0u, cStop = EB_END(c, b, 1u); c < cStop; c++)
        {
            if (lv[c] == 0u) continue;
            // Touched vehicle sets of b and c meet: b == c's partner, or a common
            // partner (bit 1 of gsVvPair: partners).
            bool meet = (gsVvPair[b * VV_PART + c] & 1u) != 0u;
            for (uint d = 0u, dStop = EB_END(d, k, 1u); d < dStop; d++)
                meet = meet || ((gsVvPair[b * VV_PART + d] & 1u) != 0u && (gsVvPair[c * VV_PART + d] & 1u) != 0u);
            if (meet) l = max(l, lv[c] + 1u);
        }
        lv[b] = l;
        if (b == a) mine = l;
    }
    return mine;
}

void vvPassSmall(uint k, uint off, uint runOff, uint points, uint step, uint lane)
{
    uint parity = step & 1u;
    if (gP.vvSplit != 0u)
    {
        vvSnapshot(off, k, parity, lane);
        sync();
    }
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    // (S) the partner scans: lane per ordered pair (a, b).
    for (uint p = lane, pStop = EB_END(p, k * k, LANES); p < pStop; p += LANES)
    {
        uint a = p / k;
        uint b = p % k;
        if (a == b) continue;
        uint va = vvMember(off, a);
        uint vb = vvMember(off, b);
        uint ka = vvBoxKey(va, parity);
        uint kb = vvBoxKey(vb, parity);
        float3 alo = ssub3(float3(vvUnkey(gAcc[ka].load()), vvUnkey(gAcc[ka + 1u].load()), vvUnkey(gAcc[ka + 2u].load())), m3);
        float3 ahi = sadd3(float3(vvUnkey(gAcc[ka + 3u].load()), vvUnkey(gAcc[ka + 4u].load()), vvUnkey(gAcc[ka + 5u].load())), m3);
        float3 olo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
        float3 ohi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
        if (!vvOverlap(alo, ahi, olo, ohi)) continue;
        uint bits = 1u;
        gsVvOff4[a].add(vvCpCount(vb));
        float3 d = ssub3(gNodes[gStatic[va * 4u].x * 3u + 1u].xyz, gNodes[gStatic[vb * 4u].x * 3u + 1u].xyz);
        if (sdot(d, d) > VV_APPROACH_SQ)
        {
            bits = 3u;
            gsVvReset4[a] = 1u;
            if (b > a) gsVvFix[a] = 1u;
            gMisc[vvRec(vb, VV_FLAGS)].w = 1u;
            gMisc[vvRec(va, VV_FLAGS)].w = 1u;
        }
        gsVvPair[a * VV_PART + b] = bits;
    }
    sync();
    // (P) every owner's throttle step and candidates.
    {
        uint triTotal = 0u;
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) triTotal += gStatic[vvMember(off, a) * 4u + 1u].y;
        for (uint p = lane, pStop = EB_END(p, triTotal, LANES); p < pStop; p += LANES)
        {
            uint a = 0u;
            uint t = p;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(cGuard);
                uint n = gStatic[vvMember(off, a) * 4u + 1u].y;
                if (t < n) break;
                t -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint gt = ra1.x + t;
            uint offered = gsVvOff4[a].load();
            bool ownerReset = gsVvReset4[a] != 0u;
            bool fix = gsVvFix[a] != 0u;
            uint4 c = gMisc[gP.vvTri + gt];
            int cwait = ownerReset ? 0 : asint(c.x);
            int cidle = asint(c.y);
            if (offered == 0u)
            {
                if (ownerReset || fix) gMisc[gP.vvTri + gt].x = 0u;
                continue;
            }
            if (cwait > 0)
            {
                gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(cwait - 1), asuint(cidle + 1), 0u, 0u);
                continue;
            }
            cwait = VV_MAX_WAIT < cidle ? VV_MAX_WAIT : cidle;
            float range = asfloat(gStatic[va * 4u + 3u].z);
            float3 plo = float3(3.0e38, 3.0e38, 3.0e38);
            float3 phi = -plo;
            float vmaxB = 0.0;
            for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
            {
                if ((gsVvPair[a * VV_PART + m] & 1u) == 0u) continue;
                uint kb = vvBoxKey(vvMember(off, m), parity);
                plo = min(plo, float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())));
                phi = max(phi, float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())));
                if (gP.clipMem != 0u) vmaxB = max(vmaxB, clipVmax(vvMember(off, m), parity));
            }
            if (!vvNearBox(gt, va, range, plo, phi, vvClipExtra(va, vmaxB, parity)))
            {
                gMisc[gP.vvTri + gt] = uint4(fix ? 0u : asuint(min(cwait + 1, throttleCap(gt))), 0u, 0u, 0u);
                continue;
            }
            gMisc[gP.vvTri + gt] = uint4(asuint(cwait), 0u, 0u, 0u);
            gNext[vvTestedAt(ra1.x + gsVvList4[a].add(1u))] = t;
        }
        // Candidates: each owner's partners' contact points in its box grown by
        // its range.
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
        {
            if (gsVvOff4[a].load() == 0u) continue;
            uint va = vvMember(off, a);
            uint key = vvBoxKey(va, parity);
            float range = asfloat(gStatic[va * 4u + 3u].z);
            float3 r3 = float3(range, range, range);
            float3 clo = ssub3(float3(vvUnkey(gAcc[key].load()), vvUnkey(gAcc[key + 1u].load()), vvUnkey(gAcc[key + 2u].load())), r3);
            float3 chi = sadd3(float3(vvUnkey(gAcc[key + 3u].load()), vvUnkey(gAcc[key + 4u].load()), vvUnkey(gAcc[key + 5u].load())), r3);
            float ownerReach = gP.clipMem != 0u ? CLIP_DEPTH + clipVmax(va, parity) * STEP + 0.001 : 0.0;
            uint run = vvRunSmall(runOff, points, a);
            for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
            {
                if ((gsVvPair[a * VV_PART + m] & 1u) == 0u) continue;
                uint v = vvMember(off, m);
                uint4 cp = gStatic[gP.vvStatic + v];
                float3 o = asfloat(gMisc[v].xyz);
                for (uint i = lane, iStop = EB_END(i, cp.y, LANES); i < iStop; i += LANES)
                {
                    uint h = gLists[cp.x + i];
                    float3 x = nf3(o + gNodes[h * 3u].xyz);
                    float ex = 0.0;
                    if (gP.clipMem != 0u)
                    {
                        float3 vh3 = gNodes[h * 3u + 1u].xyz;
                        ex = ownerReach + sqrt(dot(vh3, vh3)) * STEP;
                    }
                    float3 e3 = float3(ex, ex, ex);
                    if (any(x < clo - e3) || any(x > chi + e3)) continue;
                    gBeams[run + gsVvCand4[a].add(1u)] = float4(x, asfloat(h));
                }
            }
        }
    }
    sync();
    // (T) every owner's listed triangles against its candidates.
    {
        uint total = 0u;
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) total += gsVvList4[a].load();
        uint subs = clamp(LANES / max(total, 1u), 1u, 64u);
        for (uint p = lane, pStop = EB_END(p, total * subs, LANES); p < pStop; p += LANES)
        {
            uint q = p / subs;
            uint a = 0u;
            for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
            {
                EB_PASS(cGuard);
                uint n = gsVvList4[a].load();
                if (q < n) break;
                q -= n;
                a++;
            }
            uint va = vvMember(off, a);
            uint4 ra1 = gStatic[va * 4u + 1u];
            uint t = gNext[vvTestedAt(ra1.x + q)];
            vvTriangleCandsSmall(a, va, ra1.x + t, p % subs, subs, runOff, points, gsVvCand4[a].load(),
                                 asfloat(gStatic[va * 4u + 3u].z));
        }
    }
    sync();
    // The hits, level by level (at most k levels); with the first, every listed
    // triangle's throttle update (its marks are final) and a later owner's reset.
    for (uint l = 1u, lGuard = 0u; l <= k; l++)
    {
        EB_PASS(lGuard);
        if (l == 1u)
        {
            uint total = 0u;
            for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) total += gsVvList4[a].load();
            for (uint p = lane, pStop = EB_END(p, total, LANES); p < pStop; p += LANES)
            {
                uint q = p;
                uint a = 0u;
                for (uint c = 0u, cGuard = 0u; c + 1u < k; c++)
                {
                    EB_PASS(cGuard);
                    uint n = gsVvList4[a].load();
                    if (q < n) break;
                    q -= n;
                    a++;
                }
                uint triOff = gStatic[vvMember(off, a) * 4u + 1u].x;
                uint gt = triOff + gNext[vvTestedAt(triOff + q)];
                vvThrottleEnd(gt);
                if (gsVvFix[a] != 0u) gMisc[gP.vvTri + gt].x = 0u;
            }
        }
        for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++)
        {
            if (vvLevel(a, k, points) != l) continue;
            uint va = vvMember(off, a);
            float range = asfloat(gStatic[va * 4u + 3u].z);
            uint hits = gsVvHit4[a].load();
            if (hits <= 2u * points)
                vvApplyOrdered(va, vvHitSmall(runOff, points, a, 0u), hits, range, parity, lane, LANES, a);
            else
            {
                uint4 ra1 = gStatic[va * 4u + 1u];
                uint listed = gsVvList4[a].load();
                uint subs = clamp(LANES / max(listed, 1u), 1u, 64u);
                for (uint p = lane, pStop = EB_END(p, listed * subs, LANES); p < pStop; p += LANES)
                {
                    uint t = gNext[vvTestedAt(ra1.x + p / subs)];
                    vvTriangleCandsApply(va, ra1.x + t, p % subs, subs, vvRunSmall(runOff, points, a),
                                         gsVvCand4[a].load(), range, parity, a);
                }
            }
        }
        sync();
    }
    // The two-body counts of every owner's turn, once all levels are applied.
    uint anyHits = 0u;
    for (uint a = 0u, aStop = EB_END(a, k, 1u); a < aStop; a++) anyHits += gsVvHit4[a].load();
    if (anyHits > 0u) vvSplitClear(off, k, k, lane);
}

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_contact(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    contactGroup(gid.x, lid.x);
    ebLoopCapFlush();
}

// Contact group g: its members stepped together with the contact pass.
void contactGroup(uint g, uint lane)
{
    uint4 G = gMisc[gP.vvGroups + g];
    uint off = G.x;
    uint runOff = G.z;
    uint points = G.w;
    if (lane == 0u)
    {
        gsUniformLanes = G.y;
        coupleBegin();
    }
    sync();
    uint k = gsUniformLanes;
    uint coupleOn = gsUniformCouple;
    sync();
    // Members in vehicle order (a rank sort), their run prefixes and member
    // indices.
    for (uint i = lane, iStop = EB_END(i, k, LANES); i < iStop; i += LANES)
    {
        uint v = gNext[gP.vvMembers + off + i];
        uint rank = 0u;
        for (uint j = 0u, jStop = EB_END(j, k, 1u); j < jStop; j++) rank += gNext[gP.vvMembers + off + j] < v ? 1u : 0u;
        gNext[gP.vvMembers + gP.vehicles + off + rank] = v;
        gMisc[vvRec(v, VV_MEMBER)].x = rank;
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++)
        {
            gNext[vvPrefixAt(off, m)] = run;
            run += vvCpCount(vvMember(off, m));
        }
        vvClearCounters();
    }
    sync();
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) contactMemberBegin(vvMember(off, m), lane);
    uint stepBase = gMisc[vvMember(off, 0u)].w;
    for (uint s = 0u, sStop = EB_END(s, gP.steps, 1u); s < sStop; s++)
    {
        coupleStepMembers(off, k, lane, s, coupleOn);
        if (k <= VV_PART)
        {
            contactMembersStep(off, k, lane, s);
        }
        else
        {
            for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) contactMemberStep(vvMember(off, m), lane, s, m == 0u);
        }
#ifdef EB_ABLATE
        if ((gP.ablate & 512u) != 0u) {} else
#endif
        if (k <= VV_PART)
        {
            vvPassSmall(k, off, runOff, points, stepBase + s, lane);
        }
        else
        {
            vvPass(k, off, runOff, points, stepBase + s, lane);
        }
    }
    coupleEndMembers(off, k, lane, coupleOn);
    for (uint m = 0u, mStop = EB_END(m, k, 1u); m < mStop; m++) contactMemberEnd(vvMember(off, m), lane);
}
