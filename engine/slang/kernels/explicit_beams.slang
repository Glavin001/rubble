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
//            beamBase per beam 3 records: (a, b, kind | type << 8, group as int),
//            (k, c, limit k, limit c), (short bound, long bound, plastic, L0) as
//            float bits; at triBase per triangle (n0, n1, n2, 0); at boxBase per box
//            (lo bits, 0), (hi bits, 0). Node and beam indices are global.
//   gLists   (read-only uint) incidence entries (global beam << 1 | is end b) and
//            self-collision candidates (global node indices); at slotBase per beam
//            the gLists positions of its end a's and end b's incidence entries.
//   gNodes   per node 3 × float4: (relative position, mass), (velocity, friction),
//            (force, 0).
//   gBeams   per beam 3 × float4: (L, Tc, Tt, G), (S, last force when S or broken
//            last changed, broken, 0), (0);
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

float fastInvSqrt(float x)
{
    int i = 0x5f3759df - (asint(x) >> 1);
    float y = asfloat(i);
    return y * (1.5 - 0.5 * x * y * y);
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
    return asfloat(int(12102203.0 * x + 1064652319.0));
}

float approxPow(float x, float y)
{
    float e = float(asint(x) - 1065353216);
    return asfloat(int(y * e + 1065353216.0));
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
    float vn = dot(v, n);
    float fn = dot(f, n);
    float r = -fn;
    if (vn < 0.0) r -= (0.8 * vn - 0.2 * p / STEP) * m / STEP;
    if (r <= 0.0) return float3(0.0, 0.0, 0.0);
    float3 sf = f - fn * n;
    float3 sl = v - vn * n;
    float sigma = length(sl);
    if (sigma > 0.0) sl = sl / sigma;
    float gr = r * s.strength * friction;
    float ms = s.ms * gr;
    if (sigma < s.va && gr > 0.0 && dot(sf, sf) <= ms * ms)
        return r * n - ms * (1.0 - approxExp(-sigma / s.va)) * sl - sf;
    float g = s.mc + (s.ms - s.mc) * approxExp(-approxPow(sigma / s.vs, s.alpha));
    return r * n - (g + min(s.t2 * sigma, 5.0)) * gr * sl;
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
        float4 s1 = gBeams[j * 3u + 1u];
        if (s1.z == 0.0)
        {
            s1.z = 1.0;
            gBeams[j * 3u + 1u] = s1;
        }
    }
    return j - beamOff > master;
}

// The sum of a node's beam forces. The beam phase writes each beam's force on each
// end into that end's incidence entry, so the entries of one node are contiguous
// and independent loads (no load waits on another's index).
float3 beamSum(uint4 info, uint beamOff, uint groupOff, uint groupCount, uint parity, bool mark)
{
    uint base = gP.incBase + info.y;
    float3 f = float3(0.0, 0.0, 0.0);
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
            f = f + a;
            f = f + b;
            f = f + c;
            f = f + d;
        }
        for (; k < info.z; k++) f = f + gBeams[base + k].xyz;
        return f;
    }
    for (uint k = 0u; k < info.z; k++)
    {
        float3 push = gBeams[base + k].xyz;
        if (!groupDrops(gLists[info.y + k] >> 1, beamOff, groupOff, parity, mark)) f = f + push;
    }
    return f;
}

// --- passes ---

void nodeUpdate(uint gi, uint4 info, uint vehicle, uint local, uint step, float3 origin, uint flagsV, float ground,
                uint boxOff, uint boxCount, uint beamOff, uint groupOff, uint groupCount, bool hits)
{
    float4 P = gNodes[gi * 3u];
    float4 V = gNodes[gi * 3u + 1u];
    // Last step's beams (with its group breaks) and self-collision.
    float3 f = gNodes[gi * 3u + 2u].xyz + beamSum(info, beamOff, groupOff, groupCount, (step + 1u) & 1u, true);
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
        vel = vel + f / mass * STEP;
        pos = pos + vel * STEP;
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
        if (kind == KIND_PLAIN && gBeams[j * 3u + 1u].z == 0.0) live++;
    }
    return live;
}

void beamUpdate(uint gj, uint local, bool fast, uint groupOff, uint parity)
{
    float4 s1 = gBeams[gj * 3u + 1u];
    if (s1.z != 0.0)
    {
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(0.0, 0.0, 0.0, 0.0);
        gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    float4 c1 = asfloat(gStatic[gP.beamBase + gj * 3u + 1u]);
    float4 c2 = asfloat(gStatic[gP.beamBase + gj * 3u + 2u]);
    float4 s0 = gBeams[gj * 3u];
    uint kind = bn.z & 0xFFu;
    uint type = bn.z >> 8;
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
    float3 d = pa.xyz - pb.xyz;
    float q = dot(d, d);
    float inv = fast ? fastInvSqrt(q) : 1.0 / sqrt(q);
    float len = q * inv;
    float e = len - rest;
    float u = dot(va - vb, d) * inv;
    float k = c1.x;
    float c = c1.y;
    bool broken = false;
    bool supportBroke = false;
    if (kind == KIND_LIMITED)
    {
        float lb = c2.y * rest;
        float sb = c2.x * rest;
        float o = 0.0;
        if (e > lb) o = e - lb;
        else if (e < -sb) o = -e - sb;
        if (o != 0.0)
        {
            k = k + (c1.z - k) * o;
            c = c + (c1.w - c) * o;
        }
    }
    else if (kind == KIND_SUPPORT)
    {
        if (e > 0.0)
        {
            k = 0.0;
            c = c * 0.1;
            float factor = c2.y > 0.0 ? c2.y : SUPPORT_BREAK_FACTOR;
            if (e > rest * factor)
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
            c = c * 0.1;
        }
    }
    float f = -k * e - c * u;
    float last = f;
    float tau = abs(f);
    if (!supportBroke && tau > gate)
    {
        if (type == TYPE_DEFORMABLE && kind != KIND_LIMITED && k != 0.0)
        {
            if (f > tc && e < 0.0)
            {
                float delta = e + (tc / k) * (1.0 - c2.z);
                float old = rest;
                rest = max(rest + delta, MIN_REST);
                f -= (f - tc) / 2.0;
                tau = f;
                if (rest < old)
                {
                    tc = tc * old / rest;
                    gate = min(min(tc, tt), strength);
                }
            }
            else if (f < -tt && e > 0.0)
            {
                float delta = e - (tt / k) * (1.0 - c2.z);
                float old = rest;
                rest += delta;
                f -= (f + tt) / 2.0;
                tau = -f;
                if (rest > old)
                {
                    tt = tt * rest / old;
                    gate = min(min(tc, tt), strength);
                }
                strength -= delta * k;
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
    if (any(n0 != s0)) gBeams[gj * 3u] = n0;
    if (strength != s1.x || broken) gBeams[gj * 3u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, 0.0);
    float3 push = d * (f * inv);
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
    float3 p = origin + P.xyz;
    if (any(p < t.lo) || any(p > t.hi)) return;
    float3 q = p - t.p0;
    float alpha = 0.0;
    float beta = 0.0;
    float delta = 0.0;
    if (t.solvable)
    {
        alpha = dot(t.ra, q);
        beta = dot(t.rb, q);
        delta = dot(t.rn, q);
    }
    float gamma = 1.0 - alpha - beta;
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
    float3 fh = gNodes[h * 3u + 2u].xyz + beamSum(gStatic[gP.nodeInfoBase + h], beamOff, groupOff, groupCount, parity, false)
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
    t.p0 = origin + gNodes[t.tn.x * 3u].xyz;
    float3 p1 = origin + gNodes[t.tn.y * 3u].xyz;
    float3 p2 = origin + gNodes[t.tn.z * 3u].xyz;
    float3 u = p1 - t.p0;
    float3 w = p2 - t.p0;
    float3 cr = cross(u, w);
    float cl = length(cr);
    t.n = cl > 0.0 ? cr / cl : cr;
    float det = dot(u, cross(w, t.n));
    t.solvable = abs(det) > 1e-6;
    float rd = t.solvable ? 1.0 / det : 0.0;
    t.ra = cross(w, t.n) * rd;
    t.rb = cross(t.n, u) * rd;
    t.rn = cross(u, w) * rd;
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
    int3 c0 = int3(floor((t.lo - origin) / cell));
    int3 c1 = int3(floor((t.hi - origin) / cell));
    int3 span = c1 - c0 + int3(1, 1, 1);
    if (span.x * span.y * span.z > int(GRID_MAX_CELLS))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
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
        uint v = gStatic[gP.nodeInfoBase + gStatic[gP.beamBase + i * 3u].x].w;
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
    int3 c0 = int3(floor((t.lo - origin) / cell));
    int3 c1 = int3(floor((t.hi - origin) / cell));
    int3 span = c1 - c0 + int3(1, 1, 1);
    if (span.x * span.y * span.z > int(GRID_MAX_CELLS))
    {
        for (uint k = sub; k < candCount; k += GRID_SUBS)
            tryContact(t, gLists[candOff + k], origin, r, beamOff, groupOff, groupCount, parity);
        return;
    }
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
        uint v = gStatic[gP.nodeInfoBase + gStatic[gP.beamBase + i * 3u].x].w;
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
