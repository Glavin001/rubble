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
// A rope by JBeam's NORMAL rule (the BeamNG profile's .truck ropes, §36): in
// compression no spring and a tenth of its damping, in tension a KIND_JPLAIN beam.
static const uint KIND_JROPE = 21u;
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
// The BeamNG profile's self- and vehicle contact do no positive work
// (jbeam-semantics-spec.md §37, `dissipative_contact`; vehicle contact: the owner's
// or the partner's flag).
static const uint FLAG_DISSIPATIVE = 32768u;
// Some vehicle of the upload has FLAG_DISSIPATIVE: every vehicle's node phase takes
// its nodes' contact position steps (gAcc at shiftAt, §37; the shapes path).
static const uint FLAG_SHIFT_MEM = 65536u;
// Centred damping (jbeam-semantics-spec.md §39, ExplicitBeamsSettings::centred_damping):
// the beam phase keeps each JBeam record's damping slope in its first incidence
// entry's .w, and after the node phase each record corrects its damping by
// −½·slope·Δu (centredBeam, centredNode). gP.centredDv nonzero: some vehicle has it.
static const uint FLAG_CENTRED = 262144u;
// Centred damping's objective form (jbeam-semantics-spec.md §39.6,
// ExplicitBeamsSettings::centred_objective; with FLAG_CENTRED): a record's Δu is the
// change of its extension rate itself, each rate along the beam's direction at its
// own time, so a beam that only turns reads zero (centredKeep, centredBeamObjective).
static const uint FLAG_CENTRED_OBJECTIVE = 524288u;
// Anti-clip's box padding by each contacter's own travel (ExplicitBeams::own_travel_reach;
// with FLAG_ANTI_CLIP): ownTravelOut.
static const uint FLAG_OWN_TRAVEL = 1048576u;
// The contact position steps' fixed point, units per metre (explicit_beams.rs
// SHIFT_FIXED, 2^24).
static const float SHIFT_FIXED = 16777216.0;

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
    // Surface contact (§38, explicit_beams_surface.slang; 0: no vehicle has it).
    uint surfSnap; // gBeams: per node its force at the surface pass's start
    uint surfSplit; // gAcc: per node its surface hit count this step
    uint surfSeg; // gStatic: per vehicle (segment start in gLists, segments); gLists the global beam indices
    uint surfMark; // gAcc: the find's slot masks: per triangle (global), then segments, nodes (their records' .z, .w)
    uint centredDv; // gBeams: per node its node phase's Δv (centred damping, §39; 0: no vehicle); with the objective form (§39.6) then per node its position, then its velocity, before the node phase
    // Vehicle sleep (explicit_beams_sleep.slang): gMisc record of its header, 0 off; the
    // awake-list dispatch flag.
    uint sleepRec;
    uint sleepPad0;
    uint brakeBase; // gStatic: per parking brake 2 records (EB_AERO modules; see brakeApply)
    uint loopBudget; // passes one invocation may take in a frame (explicit_loop_cap.slang; 0 none)
    uint triGeo; // gBeams: per triangle TRI_GEO_RECS records of its step geometry (triGeoFill; 0 none)
    uint triPairs; // gNext: per vehicle 2 · TRI_PAIR_CAP words, the grid walk's pairs (0: the walk tests them)
    // The pair list reused over steps (pairReuse*): gBeams per node its build record, then
    // per vehicle its statistics (0: none); the margin M (float bits; 0: off, the walk).
    uint pairBuild;
    uint pairReuse;
    uint pairStreak; // a list reused on fewer steps than this: the walk for REUSE_BACKOFF grid steps
    // The pair list's pieces (pieces*: gNext per node its piece label, then per vehicle
    // its active word, per node its separations, per triangle the volatile list; 0 none).
    uint pieceBase;
    // Debris retired by the host (DebrisTtl, engine_softbody::explicit_debris): gMisc per
    // node a bit (128 nodes a record; 0: none retired, nothing below runs).
    uint retireBase;
    // The beam phase's records sorted by kind in gStatic (beamPhaseSorted; 0: the beams
    // in file order, beamUpdate for each). explicit_plan::W_BEAM_ORDER.
    uint beamOrder;
    uint beamOrderPad0;
    // Lossless schedule switches (explicit_plan::W_PERF_FLAGS; the same results bit for
    // bit either way): 1 selfApply's node forces once a hit and role (hitFbFill).
    uint perfFlags;
    // Vehicle contact's own scratch on the split group path (explicit_plan's
    // W_VV_SCRATCH, W_VV_SCRATCH_OFF, W_VV_TRI_PAIRS; fleets of two vehicles or more, else
    // 0): per vehicle GRID_BUCKETS words of gAcc and GRID_BUCKETS + 1 of gNext (a group's
    // counters and grid offsets between its pass's dispatches, at its first member's:
    // splitVvAt, splitVvOffAt), and per vehicle 2 · TRI_PAIR_CAP words of gNext, the
    // pass's listed (triangle, candidate) pairs of that owner (vvTriPairsAt; 0: no list).
    // Apart from the members' own self-collision regions (triPairsAt, gridCountBase,
    // gridOffBase), which a member's step may keep from one substep to the next.
    uint vvScratch;
    uint vvScratchOff;
    uint vvTriPairs;
    // Vehicle contact's state by owner vehicle (explicit_plan's W_VV_FLEET,
    // W_VV_FLEET_NEXT, W_VV_FLEET_BOX; the fl* address functions; 0: one vehicle).
    uint vvFleet;
    uint vvFleetNext;
    uint vvFleetBox;
    // The partners' room (explicit_plan's W_VV_ROOM): the candidate run, the hit list and
    // their ranks hold this many times the fleet's contact points, and a node's two-body
    // point counts this many owners (0: VV_PART, a layout before the word).
    uint vvRoom;
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
    uint dissipative; // the normal force does no positive work (§37, FLAG_DISSIPATIVE)
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
#define EB_LOOP_BUDGET gP.loopBudget
#include "explicit_loop_cap.slang"
// Barrier loops' real maxima (EB_END_UNIFORM, EB_PASS_UNIFORM): steps a frame (the
// host's PARAM_COPIES - 1), members a contact group (VV_MAX_GROUP), the awake list's
// chunks of 256 vehicles.
static const uint EB_MAX_STEPS = 128u;
static const uint EB_MAX_MEMBERS = 64u;
static const uint EB_MAX_CHUNKS = 4096u;
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
// Past the buckets, the reused pair list's words (gsReuse; never cleared by the steps'
// bucket clears; no workgroup variable of its own: Metal's threadgroup slots are full).
static const uint REUSE_WORDS = 160u;
groupshared Atomic<uint> gsCount[GRID_BUCKETS + REUSE_WORDS];
groupshared uint gsOff[GRID_BUCKETS + 1u];
groupshared uint gsPart[LANES];
groupshared uint gsBlk[GRID_BLOCKS];

#ifdef EB_ABLATE
// Measurement only (time): a second mask, in the free parameter word after beamOrder
// (ExplicitBeamsGpu::set_ablation2), of the solo frame's smaller phases: 1 the raycast
// wheels' step, 2 the triangles' finish and tested list, 4 the grid's bucket clears and
// counts, 8 the pair list's tracking, 32 the triangles' step geometry (triGeoFill), 64
// selfApply, 256 the group slots' clears, 512 the coupling's step, 1024 the whole step
// (the frame's start, end and dispatches alone), 2048 lane 0's grab, brake and pending
// breaks.
#define EB_A2(b) ((gP.beamOrderPad0 & (b)) != 0u)
#else
#define EB_A2(b) false
#endif
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
    s.dissipative = 0u;
    return s;
}

Surface boxSurface()
{
    Surface s;
    s.va = 3.0; s.ms = 1.2; s.mc = 0.75; s.t2 = 0.01; s.vs = 6.0; s.alpha = 2.0; s.strength = 1.0;
    s.exact = 0u;
    s.dissipative = 0u;
    return s;
}

// Surface s as vehicle v's friction law takes it: exact with FLAG_EXACT_FRICTION;
// dissipative (§37: the node pass's ground, box and shape contacts too) with
// FLAG_DISSIPATIVE.
Surface surfaceV(Surface s, uint v)
{
    s.exact = (gStatic[v * 4u + 3u].x & FLAG_EXACT_FRICTION) != 0u ? 1u : 0u;
    s.dissipative = (gStatic[v * 4u + 3u].x & FLAG_DISSIPATIVE) != 0u ? 1u : 0u;
    return s;
}

// Self- and vehicle contact's surface: surfaceV(boxSurface(), v), dissipative when
// vehicle v or w has FLAG_DISSIPATIVE (self-collision: w = v).
Surface contactSurfaceV(uint v, uint w)
{
    Surface s = surfaceV(boxSurface(), v);
    s.dissipative = ((gStatic[v * 4u + 3u].x | gStatic[w * 4u + 3u].x) & FLAG_DISSIPATIVE) != 0u ? 1u : 0u;
    return s;
}

#include "explicit_beams_numerics.slang"
#include "explicit_beams_nodes.slang"
#include "explicit_beams_jbeams.slang"
#include "explicit_beams_centred.slang"
#include "explicit_beams_beam_phase.slang"
#include "explicit_beams_triangles.slang"
#include "explicit_beams_pieces.slang"
#include "explicit_beams_self_hits.slang"
#include "explicit_beams_hit_roles.slang"
#include "explicit_beams_frame.slang"
#include "explicit_beams_wide.slang"
#include "explicit_beams_hits_local.slang"
#include "explicit_beams_vv.slang"
#include "explicit_beams_groups.slang"
#include "explicit_beams_members.slang"
#include "explicit_beams_vv_hits.slang"
#include "explicit_beams_vv_small.slang"
#include "explicit_beams_vv_big.slang"
#include "explicit_beams_vv_levels.slang"
