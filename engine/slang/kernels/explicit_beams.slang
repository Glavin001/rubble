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
    uint aeroBase; // gStatic: per vehicle 2 records
    uint wingBase; // gStatic: per wing 4 records
    uint cmdBase; // gStatic: per self-centering command 2 records
    uint aeroState; // gBeams: per vehicle D, then per wing 4 records, then per command 1
    uint wingTotal; // wings over every vehicle
    uint shock2Base; // gStatic: per shocks2 shock 2 records (EB_AERO modules)
    uint shock2Count; // shocks2 shocks over every vehicle
};

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

#ifdef EB_AERO
void nodeUpdateFuse(uint gi, uint4 info, uint vehicle, float3 origin, float ground, uint boxOff, uint boxCount,
                    uint beamOff, uint groupOff, uint groupCount, uint parity, bool hits);
static const uint FLAG_FUSEDRAG = 8u;
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


// --- Aerodynamics and commands with no input (EB_AERO modules) ---
// engine/docs/explicit-beams-aero-commands-spec.md; CPU reference
// engine_softbody::explicit_aero. The specification's double-precision
// sub-expressions run in strict single precision here (WGSL has no f64), so the GPU
// may differ from the CPU by an ulp in them; the angle of attack's acos is the
// device's.
//
//   gStatic at aeroBase per vehicle: (wingOff, wingCount, cmdOff, cmdCount) (wing and
//            command indices global), (front node global, width bits, node count,
//            0); at wingBase per wing: (FLD, FRD, FLU, FRU), (BLD, BRD, BLU, BRU)
//            (global nodes), (efficacy, break reference, table record, induced side
//            0 none 1 left 2 right), (chain span, chain area, 0, 0) as bits; at
//            cmdBase per command: (beam global, contract first, centre bits, 0),
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
#endif

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_frame(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
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
    uint4 aero0 = gStatic[gP.aeroBase + v * 2u];
    uint4 aero1 = gStatic[gP.aeroBase + v * 2u + 1u];
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
    sync();

    for (uint s = 0u; s < gP.steps; s++)
    {
        uint step = stepBase + s;
        uint parity = step & 1u;
        bool hits = gsHit != 0u;
        for (uint i = lane; i < nodeCount; i += LANES)
            nodeUpdate(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
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
        sync();
        if (lane == 0u)
        {
            gsHit = 0u;
            gsTestedCount[parity ^ 1u].store(0u);
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
        // Wings (forces from this step's integrated state), the fuselage drag for
        // the next step, and the commands' rest lengths; then one lane adds the
        // wings' forces in file order while the others start the beams.
        for (uint w = lane; w < aero0.y; w += LANES) wingForces(aero0.x + w, origin);
        if (lane == LANES - 1u && (flagsV & FLAG_FUSEDRAG) != 0u) fuseDrag(aero1, origin, v);
        for (uint c = lane; c < aero0.w; c += LANES) commandUpdate(aero0.z + c);
        sync();
        if (lane == 0u) wingApply(aero0.x, aero0.y);
#endif
        for (uint j = lane; j < beamCount; j += LANES) beamUpdate(beamOff + j, j, fast, groupOff, parity);
        sync();
        if (gP.grid != 0u)
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
        if (candCount > 0u && cell > 0.0)
        {
            uint listed = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
            for (uint p = lane; p < listed * GRID_SUBS; p += LANES)
            {
                uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
                triangleGrid(triOff + t, p % GRID_SUBS, origin, range, cell, candOff, candCount, beamOff,
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
        sync();
    }
    if (gP.steps > 0u && candCount > 0u)
    {
        for (uint t = lane; t < triCount; t += LANES) triangleFinish(triOff + t);
    }
    // The last step's group breaks, for a read-back (the next node phase applies
    // them to the forces).
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint last = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane; j < beamCount; j += LANES) groupDrops(beamOff + j, beamOff, groupOff, last, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
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
