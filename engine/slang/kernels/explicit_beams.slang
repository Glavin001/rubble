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
//   collide    contacters against the vehicle's collision triangles, forces into
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
//            groupOff, groupCount), (flags, ground bits, range bits, 0); at
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
//   gBeams   per beam 3 × float4: (L, Tc, Tt, G), (S, last force, broken, 0), (0);
//            at incBase per gLists position, the force of an incidence entry's beam
//            on that node (unused at candidate positions).
//   gAcc     per node 3 × int: self-collision force in 1/32 N.
//   gGroups  two parities × per vehicle group slot: index (local) of the first
//            breaking master, 0xFFFFFFFF when none.
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
    uint pad0;
    uint pad1;
    uint pad2;
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

// Nonzero when a self-collision hit may have left forces in gAcc since the node
// phase last read and cleared it (set at frame start, since the last frame's
// last step may have hit): without one, every accumulator is zero.
groupshared uint gsHit;

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
    if (groupCount == 0u)
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
                if (group > 0) gGroups[groupSlot(parity, groupOff, uint(group))].min(local);
            }
        }
    }
    gBeams[gj * 3u] = float4(rest, tc, tt, gate);
    gBeams[gj * 3u + 1u] = float4(strength, last, broken ? 1.0 : 0.0, 0.0);
    float3 push = d * (f * inv);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u]] = float4(push, 0.0);
    gBeams[gP.incBase + gLists[gP.slotBase + gj * 2u + 1u]] = float4(-push, 0.0);
}

// A triangle's throttle after a step: a waiting triangle counts down (and its idle
// steps up); one that was tested restarts its wait from its hit (gMisc .w, set by
// any of its lanes) or counts it up. A triangle that comes off idle waits at most
// 12 steps and is tested that same step (triangleTest reads it as not waiting).
void triangleFinish(uint gt)
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
}

// Lane `sub` of the `lanes` lanes of triangle gt tests candidates sub, sub + lanes, …
// Forces go to fixed-point atomics, so the lanes' order does not change the sums.
void triangleTest(uint gt, uint sub, uint lanes, float3 origin, float r, uint candOff, uint candCount,
                  uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    uint mi = gP.vehicles + gt;
    if (asint(gMisc[mi].x) > 0) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 p0 = origin + gNodes[tn.x * 3u].xyz;
    float3 p1 = origin + gNodes[tn.y * 3u].xyz;
    float3 p2 = origin + gNodes[tn.z * 3u].xyz;
    float3 u = p1 - p0;
    float3 w = p2 - p0;
    float3 cr = cross(u, w);
    float cl = length(cr);
    float3 n = cl > 0.0 ? cr / cl : cr;
    float det = dot(u, cross(w, n));
    bool solvable = abs(det) > 1e-6;
    float rd = solvable ? 1.0 / det : 0.0;
    float3 ra = cross(w, n) * rd;
    float3 rb = cross(n, u) * rd;
    float3 rn = cross(u, w) * rd;
    float3 lo = min(min(p0, p1), p2) - float3(r, r, r);
    float3 hi = max(max(p0, p1), p2) + float3(r, r, r);
    for (uint k = sub; k < candCount; k += lanes)
    {
        uint h = gLists[candOff + k];
        if (h == tn.x || h == tn.y || h == tn.z) continue;
        float4 P = gNodes[h * 3u];
        float3 p = origin + P.xyz;
        if (any(p < lo) || any(p > hi)) continue;
        float3 q = p - p0;
        float alpha = 0.0;
        float beta = 0.0;
        float delta = 0.0;
        if (solvable)
        {
            alpha = dot(ra, q);
            beta = dot(rb, q);
            delta = dot(rn, q);
        }
        float gamma = 1.0 - alpha - beta;
        if (!(alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && abs(delta) <= r)) continue;
        gMisc[mi].w = 1u;
        gsHit = 1u;
        float3 nn = n;
        if (delta < 0.0)
        {
            nn = -n;
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
}

[shader("compute")]
[numthreads(256, 1, 1)]
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
    bool fast = (flagsV & FLAG_FAST_INV_SQRT) != 0u;

    // Origin rebase: every lane reads node 0 before any lane moves it.
    float3 first = gNodes[nodeOff * 3u].xyz;
    sync();
    if (length(first) > REBASE_RADIUS)
    {
        for (uint i = lane; i < nodeCount; i += 256u)
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
    if (lane == 0u) gsHit = 1u;
    // Static records the lane reads every step, kept in registers: its first node's.
    uint4 myInfo = gStatic[gP.nodeInfoBase + nodeOff + min(lane, max(nodeCount, 1u) - 1u)];
    sync();

    for (uint s = 0u; s < gP.steps; s++)
    {
        uint step = stepBase + s;
        uint parity = step & 1u;
        bool hits = gsHit != 0u;
        for (uint i = lane; i < nodeCount; i += 256u)
            nodeUpdate(nodeOff + i, i == lane ? myInfo : gStatic[gP.nodeInfoBase + nodeOff + i], v, i, step, origin, flagsV, ground, boxOff, boxCount, beamOff, groupOff,
                       groupCount, hits);
        for (uint g = lane; g < groupCount; g += 256u) gGroups[groupSlot(parity, groupOff, g)].store(NO_GROUP);
        if (s > 0u && candCount > 0u)
        {
            for (uint t = lane; t < triCount; t += 256u) triangleFinish(triOff + t);
        }
        sync();
        if (lane == 0u) gsHit = 0u;
        for (uint j = lane; j < beamCount; j += 256u) beamUpdate(beamOff + j, j, fast, groupOff, parity);
        sync();
        if (candCount > 0u)
        {
            // Up to 256 lanes per step: `lanes` per triangle, one per candidate when
            // they fit.
            uint lanes = clamp(256u / max(triCount, 1u), 1u, candCount);
            uint stride = (256u / lanes) * lanes;
            for (uint p = lane; p < triCount * lanes && lane < stride; p += stride)
                triangleTest(triOff + p / lanes, p % lanes, lanes, origin, range, candOff, candCount, beamOff,
                             groupOff, groupCount, parity);
        }
        sync();
    }
    if (gP.steps > 0u && candCount > 0u)
    {
        for (uint t = lane; t < triCount; t += 256u) triangleFinish(triOff + t);
    }
    // The last step's group breaks, for a read-back (the next node phase applies
    // them to the forces).
    if (gP.steps > 0u && groupCount > 0u)
    {
        uint last = (stepBase + gP.steps - 1u) & 1u;
        for (uint j = lane; j < beamCount; j += 256u) groupDrops(beamOff + j, beamOff, groupOff, last, true);
    }
    if (lane == 0u) gMisc[v] = uint4(misc.xyz, stepBase + gP.steps);
}
