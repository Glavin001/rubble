// Directed debris (opt-in): free-flying guided bodies are steered toward
// authored zones by a small velocity kick at launch plus a constant extra
// acceleration held for the rest of the flight. Gravity plus a constant
// acceleration is still an exact parabola, so the arc never bends or homes in.
//
// The passes run after the constraint graph and before predict_bodies:
//   guide_scan   (bodies)      steer, retire on a push, flag candidates, rank per block
//   guide_select (1 group)     rotated prefix over block counts, at most planBudget
//   guide_rank   (bodies)      write each selected candidate into its plan slot
//   guide_plan   (1 group per slot)  lanes split the zones; one weighted pick
//   guide_commit (1 group)     apply the plans in rank order (deterministic quotas)
// A world without guidance never encodes them, so its steps are unchanged.
//
// Adding a·dt to the velocity before prediction moves the inertial target by
// a·dt², which is exactly an external acceleration while no constraint acts on
// the body: steered bodies have no joints, and the first step a contact pushes
// one ends its guidance.
#include "../physics/common.slang"
#include "../physics/atomics.slang"

struct GuideParams
{
    float time;
    float dt;
    float gravityY;
    uint zoneCount;
    uint bodyCount;
    uint planBudget;
    uint settleFrames;
    uint replanFrames;
    float kickFrac;    // launch kick budget, a fraction of the launch speed
    float kickWindow;  // s after becoming free in which the kick is allowed
    float accelMax;    // m/s²
    float maxJerk;     // m/s³, re-plans toward moving anchors
    float tMin;        // shortest flight worth steering (s)
    float tMax;        // longest flight considered (s)
    float yGround;
    float participation;
    uint kickFrames;
    uint seed;
    float minSpeed;
    uint blockCount;
    uint maxZones;
    uint jointCount;   // CSR ids below this are joints
    float impactDv;    // m/s off the ballistic update that counts as a contact
    uint _pad2;
};

static const uint ZONE_POINT = 0u;
static const uint ZONE_SPHERE = 1u;
static const uint ZONE_BOX = 2u;
static const uint ZONE_RING = 3u;
static const uint ZONE_ARC = 4u;
static const uint ZONE_FRAME_ANCHOR = 1u;
static const uint NO_ANCHOR = 0xFFFFFFFFu;
static const uint NO_ZONE = 0xFFFFFFFFu;

// Matches engine-webgpu `GuideZoneGpu` (80 B).
struct GuideZone
{
    float3 center;  // world position (no anchor) or ignored (anchored)
    uint shape;
    float3 extent;  // sphere (r); box half extents; ring/arc (r_in, r_out, height)
    uint anchor;    // body index or NO_ANCHOR
    float3 offset;  // in the zone frame
    float weight;
    float tStart;
    float tEnd;
    uint quota;     // 0 = unlimited
    uint tags;
    float yaw;
    uint frame;     // 0 world, 1 anchor-relative (anchor yaw)
    float angleFrom;
    float angleTo;
};

static const uint GUIDE_OFF = 0u;
static const uint GUIDE_ARMED = 1u;
static const uint GUIDE_STEER = 2u;
static const uint GUIDE_DONE = 3u;

static const uint GFLAG_WAS_FREE = 1u;
static const uint GFLAG_CANDIDATE = 2u;

// Matches engine-webgpu `GuideBodyGpu` (64 B).
struct GuideBody
{
    uint state;
    uint zone;
    uint generation; // the host handle's, for seeds (the host clears a freed slot)
    uint freeFrames;
    float3 accel;
    float hitTime;
    float3 kick;
    uint kickLeft;
    uint rank;      // rank among its block's candidates (guide_scan)
    uint sincePlan;
    uint tagMask;
    uint flags;
};

// One plan per selected candidate, in rank order (64 B).
struct GuideSlot
{
    uint body;
    uint zone;      // NO_ZONE: nothing reachable
    uint replan;
    uint _pad;
    float3 accel;
    float hitTime;
    float3 kick;
    float _pad1;
    float4 _pad2;
};

BIND(0) RWStructuredBuffer<BodyState> gBodies;
#if defined(ENGINE_WGSL)
// As world_step.slang: prev bodies at gBodies[HOST_S(1)..], the CSR index at
// gConstraintIds[HOST_S(24)..], the adjacency ids at gConstraintIds[0..].
struct HostUniforms { AvbdParams params; uint4 s[8]; };
BIND(6) ConstantBuffer<HostUniforms> gHost;
BIND(14) RWStructuredBuffer<uint> gConstraintIds;
#define CINDEX(i) gConstraintIds[gHost.s[6].x + (i)]
#define CSR_ID(e) gConstraintIds[e]
#define PREV_VEL(i) gBodies[gHost.s[0].y + (i)].linearVel
#else
RWStructuredBuffer<BodyState> gBodiesPrev;
RWStructuredBuffer<uint> gConstraintIndex;
RWStructuredBuffer<uint> gConstraintIds;
#define CINDEX(i) gConstraintIndex[i]
#define CSR_ID(e) gConstraintIds[e]
#define PREV_VEL(i) gBodiesPrev[i].linearVel
#endif
BIND(36) RWStructuredBuffer<GuideParams> gGuideParams;
BIND(37) RWStructuredBuffer<GuideZone> gZones;
BIND(38) RWStructuredBuffer<GuideBody> gGuide;
// [0] cursor block, [1] selected this tick; claims from 16; then per block
// (candidate count, first rank).
BIND(39) RWStructuredBuffer<uint> gGuideScratch;
BIND(40) RWStructuredBuffer<GuideSlot> gGuideSlots;

#define GP (gGuideParams[0])
// Bodies per scan block (and lanes of the single-group passes); plan slots
// per step are capped at this (GuidanceCapacity::plan_budget).
#define GUIDE_BLOCK 256u
#define PLAN_LANES 64u
static const uint SCRATCH_CLAIMS = 16u;
uint blockCountWord(uint b) { return SCRATCH_CLAIMS + GP.maxZones + 2u * b; }
uint blockStartWord(uint b) { return blockCountWord(b) + 1u; }

uint dispatchIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }
// Workgroup index of a dispatch whose rows hold 32768 groups (any group size).
uint groupIndex(uint3 gid) { return gid.x + gid.y * 32768u; }

uint pcgHash(uint v)
{
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float unitFloat(uint h) { return float(h >> 8) * (1.0 / 16777216.0); }

// Time until y reaches the ground plane under gravity gy (0 when below it).
float timeToGround(float y, float vy, float gy, float yGround)
{
    float h = y - yGround;
    if (h <= 0.0) return 0.0;
    if (gy >= -1e-6) return 1e6;
    float disc = vy * vy - 2.0 * gy * h;
    return (-vy - sqrt(max(disc, 0.0))) / gy;
}

float3 rotateYaw(float c, float s, float3 v)
{
    return float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// A point of the zone's shape in its frame, seeded by h.
float3 sampleShape(GuideZone z, uint h)
{
    uint h1 = pcgHash(h);
    uint h2 = pcgHash(h1);
    uint h3 = pcgHash(h2);
    float u1 = unitFloat(h1);
    float u2 = unitFloat(h2);
    float u3 = unitFloat(h3);
    if (z.shape == ZONE_SPHERE)
    {
        float cz = 2.0 * u2 - 1.0;
        float sz = sqrt(max(1.0 - cz * cz, 0.0));
        float ph = 6.2831853 * u3;
        return z.extent.x * pow(u1, 1.0 / 3.0) * float3(sz * cos(ph), cz, sz * sin(ph));
    }
    if (z.shape == ZONE_BOX)
    {
        return (2.0 * float3(u1, u2, u3) - 1.0) * z.extent;
    }
    if (z.shape == ZONE_RING || z.shape == ZONE_ARC)
    {
        float ri = z.extent.x;
        float ro = max(z.extent.y, ri);
        float r = sqrt(lerp(ri * ri, ro * ro, u1));
        float ang = z.shape == ZONE_RING ? 6.2831853 * u2 : lerp(z.angleFrom, z.angleTo, u2);
        // Angle 0 is the frame's +Z (an anchor's forward), increasing toward +X.
        return float3(sin(ang) * r, u3 * z.extent.z, cos(ang) * r);
    }
    return float3(0.0, 0.0, 0.0);
}

// A body's aim point moves in a straight line with its anchor: P(t) = P0 + V·t.
struct AimLine
{
    float3 p0;
    float3 v;
};

AimLine aimLine(GuideZone z, float3 local)
{
    AimLine a;
    float3 origin = z.center;
    float3 vel = float3(0.0, 0.0, 0.0);
    float yaw = z.yaw;
    if (z.anchor != NO_ANCHOR)
    {
        BodyState b = gBodies[z.anchor];
        origin = b.position;
        vel = b.linearVel;
        if (z.frame == ZONE_FRAME_ANCHOR)
        {
            float3 fwd = quatRotate(b.orientation, float3(0.0, 0.0, 1.0));
            yaw += atan2(fwd.x, fwd.z);
        }
    }
    a.p0 = origin + rotateYaw(cos(yaw), sin(yaw), z.offset + local);
    a.v = vel;
    return a;
}

uint aimSeed(uint body, uint generation, uint zone)
{
    return pcgHash(body * 0x9E3779B9u ^ pcgHash(zone ^ (generation * 0x85EBCA77u) ^ GP.seed));
}

bool zoneOpen(GuideZone z, uint zi, uint tagMask)
{
    if (z.weight <= 0.0 || GP.time < z.tStart || GP.time > z.tEnd) return false;
    if ((z.tags & tagMask) == 0u) return false;
    return z.quota == 0u || gGuideScratch[SCRATCH_CLAIMS + zi] < z.quota;
}

float3 arcAt(float3 p0, float3 v0, float3 acc, float t)
{
    return p0 + v0 * t + 0.5 * acc * (t * t);
}

// The miss vector at time t: aim minus arc = d0 + dv·t - ½·acc·t².
float3 missAt(float3 d0, float3 dv, float3 acc, float t)
{
    return d0 + dv * t - 0.5 * acc * (t * t);
}

// Closest approach of the arc to the moving aim point over [lo, hi].
float closestTime(float3 d0, float3 dv, float3 acc, float lo, float hi)
{
    float best = lo;
    float bestD = 3.4e38;
    float step = (hi - lo) / 15.0;
    for (uint j = 0u; j < 16u; j++)
    {
        float t = lo + step * float(j);
        float3 d = missAt(d0, dv, acc, t);
        float dd = dot(d, d);
        if (dd < bestD)
        {
            bestD = dd;
            best = t;
        }
    }
    float a = max(lo, best - step);
    float b = min(hi, best + step);
    for (uint k = 0u; k < 8u; k++)
    {
        float m1 = a + (b - a) * 0.381966;
        float m2 = a + (b - a) * 0.618034;
        float3 d1 = missAt(d0, dv, acc, m1);
        float3 d2 = missAt(d0, dv, acc, m2);
        if (dot(d1, d1) < dot(d2, d2)) b = m2; else a = m1;
    }
    return 0.5 * (a + b);
}

// ---------------------------------------------------------------------------

groupshared uint guideShared[GUIDE_BLOCK];

// Exclusive prefix over the group's GUIDE_BLOCK lanes; every lane must call it.
uint groupPrefix(uint local, uint value)
{
    guideShared[local] = value;
    GROUP_SYNC();
    for (uint offset = 1u; offset < GUIDE_BLOCK; offset <<= 1u)
    {
        uint addend = local >= offset ? guideShared[local - offset] : 0u;
        GROUP_SYNC();
        guideShared[local] += addend;
        GROUP_SYNC();
    }
    uint inclusive = guideShared[local];
    GROUP_SYNC();
    return inclusive - value;
}

// A joint (a bond still intact, a hinge) keeps the body out of free flight.
bool jointed(uint i)
{
    uint lo = CINDEX(i);
    uint hi = CINDEX(i + 1u);
    if (hi - lo > 64u) return true;
    for (uint e = lo; e < hi; e++)
    {
        if (CSR_ID(e) < GP.jointCount) return true;
    }
    return false;
}

// Contacts are not enough to tell: look-ahead contacts with a neighbour flying
// alongside never push. A push shows in the velocity: in free flight the last
// step changed it by exactly g·dt from what prediction carried in (that
// already holds the guidance's own a·dt). A hit, a scrape or a support does not.
bool impacted(uint i, float3 vel)
{
    float3 expected = PREV_VEL(i) + float3(0.0, GP.gravityY * GP.dt, 0.0);
    return length(vel - expected) > GP.impactDv;
}

// Steers, retires and flags one body; returns 1 when it wants a plan this
// tick. The scan touches every body each guided step, so it loads and stores
// only the fields it needs: an unguided body costs one word.
uint scanBody(uint i)
{
    uint state = gGuide[i].state;
    if (state == GUIDE_OFF || state == GUIDE_DONE) return 0u;
    uint bflags = gBodies[i].flags;
    float3 vel = gBodies[i].linearVel;
    uint skip = BODY_KINEMATIC | BODY_SLEEPING | BODY_PARTICLE | BODY_FOLLOW | BODY_VEHICLE;
    bool movable = (bflags & BODY_DYNAMIC) != 0u && gBodies[i].invMass > 0.0 && (bflags & skip) == 0u;
    bool free = movable && !jointed(i) && !impacted(i, vel);
    uint gflags = gGuide[i].flags & ~GFLAG_CANDIDATE;
    uint want = 0u;
    if (state == GUIDE_STEER)
    {
        // One arc: the first push, or the planned hit time, ends guidance.
        if (!free || GP.time >= gGuide[i].hitTime)
        {
            gGuide[i].state = GUIDE_DONE;
            gGuide[i].flags = gflags;
            return 0u;
        }
        float3 dv = gGuide[i].accel * GP.dt;
        uint kickLeft = gGuide[i].kickLeft;
        if (kickLeft > 0u)
        {
            dv += gGuide[i].kick;
            kickLeft -= 1u;
            gGuide[i].kickLeft = kickLeft;
        }
        gBodies[i].linearVel = vel + dv;
        uint since = gGuide[i].sincePlan + 1u;
        gGuide[i].sincePlan = since;
        if (since >= GP.replanFrames && kickLeft == 0u)
        {
            uint zone = gGuide[i].zone;
            if (zone < GP.zoneCount && gZones[zone].anchor != NO_ANCHOR) want = 1u;
        }
    }
    else if (!free)
    {
        // Pushed by something after its first free flight: it keeps its own arc.
        if ((gflags & GFLAG_WAS_FREE) != 0u) gGuide[i].state = GUIDE_DONE;
        gGuide[i].freeFrames = 0u;
    }
    else
    {
        uint frames = gGuide[i].freeFrames + 1u;
        gGuide[i].freeFrames = frames;
        if (frames >= GP.settleFrames)
        {
            gflags |= GFLAG_WAS_FREE;
            float y = gBodies[i].position.y;
            float t = min(timeToGround(y, vel.y, GP.gravityY, GP.yGround), GP.tMax);
            uint h = pcgHash(i * 0x27D4EB2Fu ^ pcgHash(gGuide[i].generation ^ GP.seed ^ 0xB5297A4Du));
            bool joins = unitFloat(h) < GP.participation;
            if (t < GP.tMin || !joins || length(vel) < GP.minSpeed)
            {
                gGuide[i].state = GUIDE_DONE;
            }
            else
            {
                want = 1u;
            }
        }
    }
    if (want != 0u) gflags |= GFLAG_CANDIDATE;
    gGuide[i].flags = gflags;
    return want;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void guide_scan(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint i = groupIndex(gid) * GUIDE_BLOCK + lid.x;
    uint want = i < GP.bodyCount ? scanBody(i) : 0u;
    uint rank = groupPrefix(lid.x, want);
    if (want != 0u) gGuide[i].rank = rank;
    uint block = groupIndex(gid);
    if (lid.x == GUIDE_BLOCK - 1u && block < GP.blockCount)
    {
        gGuideScratch[blockCountWord(block)] = rank + want;
    }
}

// Ranks candidates across blocks, starting at the cursor block so a backlog
// drains oldest-block-first, and keeps the first planBudget of them.
[shader("compute")]
[numthreads(256, 1, 1)]
void guide_select(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    uint nb = max(GP.blockCount, 1u);
    uint cursor = gGuideScratch[0] % nb;
    uint running = 0u;
    for (uint base = 0u; base < nb; base += GUIDE_BLOCK)
    {
        uint k = base + local;
        uint b = (cursor + k) % nb;
        uint count = k < nb ? gGuideScratch[blockCountWord(b)] : 0u;
        uint start = running + groupPrefix(local, count);
        if (k < nb) gGuideScratch[blockStartWord(b)] = start;
        // Every lane reads the group's total from the last lane.
        guideShared[local] = start + count;
        GROUP_SYNC();
        running = guideShared[GUIDE_BLOCK - 1u];
        GROUP_SYNC();
    }
    uint selected = min(running, GP.planBudget);
    // The block where this tick's selection ends is where the next one starts.
    for (uint base = 0u; base < nb; base += GUIDE_BLOCK)
    {
        uint k = base + local;
        if (k < nb)
        {
            uint b = (cursor + k) % nb;
            uint count = gGuideScratch[blockCountWord(b)];
            uint start = gGuideScratch[blockStartWord(b)];
            if (count > 0u && start < selected && start + count >= selected)
            {
                gGuideScratch[0] = start + count > selected ? b : (b + 1u) % nb;
            }
        }
    }
    if (local == 0u) gGuideScratch[1] = selected;
}

// Each selected candidate claims its plan slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void guide_rank(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= GP.bodyCount) return;
    GuideBody g = gGuide[i];
    if ((g.flags & GFLAG_CANDIDATE) == 0u) return;
    uint rank = gGuideScratch[blockStartWord(i / GUIDE_BLOCK)] + g.rank;
    if (rank >= gGuideScratch[1]) return;
    gGuideSlots[rank].body = i;
}

// One lane's pick over its share of the zones (a weighted reservoir), then
// one pick among the lanes by their totals: exactly weight-proportional over
// every zone, and deterministic.
groupshared float planTotal[PLAN_LANES];
groupshared uint planZone[PLAN_LANES];
groupshared float4 planKick[PLAN_LANES];   // xyz kick, w time to the aim point
groupshared float4 planAccel[PLAN_LANES];

void replanSlot(uint rank, uint i, BodyState b, GuideBody g)
{
    // Same zone and aim point; only the held acceleration moves, within the
    // jerk limit, toward where a moving anchor will be at the hit time.
    GuideSlot slot = gGuideSlots[rank];
    slot.zone = g.zone;
    slot.replan = 1u;
    slot.hitTime = g.hitTime;
    slot.accel = g.accel;
    slot.kick = float3(0.0, 0.0, 0.0);
    float dt = GP.dt;
    float t = g.hitTime - GP.time - dt;
    if (t > dt && g.zone < GP.zoneCount)
    {
        GuideZone z = gZones[g.zone];
        AimLine aim = aimLine(z, sampleShape(z, aimSeed(i, g.generation, g.zone)));
        float3 grav = float3(0.0, GP.gravityY, 0.0);
        float3 miss = aim.p0 + aim.v * t - arcAt(b.position, b.linearVel, grav + g.accel, t);
        float3 want = g.accel + 2.0 * miss / (t * t);
        float lw = length(want);
        if (lw > GP.accelMax) want *= GP.accelMax / lw;
        float3 delta = want - g.accel;
        float maxDelta = GP.maxJerk * float(GP.replanFrames) * dt;
        float ld = length(delta);
        if (ld > maxDelta) delta *= maxDelta / ld;
        slot.accel = g.accel + delta;
    }
    gGuideSlots[rank] = slot;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void guide_plan(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint rank = groupIndex(gid);
    uint lane = lid.x;
    // The whole group shares one slot, so these exits are uniform.
    if (rank >= gGuideScratch[1]) return;
    uint i = gGuideSlots[rank].body;
    GuideBody g = gGuide[i];
    BodyState b = gBodies[i];
    if (g.state == GUIDE_STEER)
    {
        if (lane == 0u) replanSlot(rank, i, b, g);
        return;
    }

    float3 grav = float3(0.0, GP.gravityY, 0.0);
    float3 p0 = b.position;
    float3 v0 = b.linearVel;
    float dt = GP.dt;
    float k = float(g.freeFrames) * dt <= GP.kickWindow ? length(v0) * GP.kickFrac : 0.0;
    float amax = GP.accelMax;
    float tg = min(timeToGround(p0.y, v0.y, GP.gravityY, GP.yGround), GP.tMax);
    // The kick lands over the next kickFrames steps and the acceleration from
    // the next step: their lever arms are that much shorter than t.
    float kickLag = 0.5 * float(GP.kickFrames + 1u) * dt;

    float total = 0.0;
    uint pick = NO_ZONE;
    float3 pickDir = float3(0.0, 0.0, 0.0);
    float pickS = 0.0;
    float pickT = 0.0;
    uint n = min(GP.zoneCount, GP.maxZones);
    for (uint zi = lane; zi < n; zi += PLAN_LANES)
    {
        GuideZone z = gZones[zi];
        if (!zoneOpen(z, zi, g.tagMask)) continue;
        uint seed = aimSeed(i, g.generation, zi);
        AimLine aim = aimLine(z, sampleShape(z, seed));
        float3 d0 = aim.p0 - p0;
        float3 dv = aim.v - v0;
        float t = closestTime(d0, dv, grav, min(GP.tMin, tg), tg);
        if (t < GP.tMin) continue;
        float3 miss = missAt(d0, dv, grav, t);
        float tk = max(t - kickLag, 0.0);
        float ta = max(t - dt, 0.0);
        float reach = k * tk + 0.5 * amax * ta * ta;
        float dist = length(miss);
        if (reach <= 1e-6 || dist > reach) continue;
        // Each budget is used in the same fraction, so the kick stays inside
        // its cone and the acceleration inside its limit.
        float s = dist / reach;
        float score = z.weight * (1.0 - s) * (1.0 - s) + 1e-12;
        total += score;
        if (unitFloat(pcgHash(seed ^ 0x68E31DA4u)) * total < score)
        {
            pick = zi;
            pickDir = dist > 1e-9 ? miss / dist : float3(0.0, 0.0, 0.0);
            pickS = s;
            pickT = t;
        }
    }
    planTotal[lane] = total;
    planZone[lane] = pick;
    planKick[lane] = float4(pickDir * (pickS * k), pickT);
    planAccel[lane] = float4(pickDir * (pickS * amax), 0.0);
    GROUP_SYNC();
    if (lane != 0u) return;
    float sum = 0.0;
    for (uint l = 0u; l < PLAN_LANES; l++) sum += planTotal[l];
    GuideSlot slot = gGuideSlots[rank];
    slot.zone = NO_ZONE;
    slot.replan = 0u;
    slot.accel = float3(0.0, 0.0, 0.0);
    slot.kick = float3(0.0, 0.0, 0.0);
    slot.hitTime = 0.0;
    if (sum > 0.0)
    {
        float u = unitFloat(pcgHash(i * 0x632BE5ABu ^ pcgHash(g.generation ^ GP.seed ^ 0x1B873593u))) * sum;
        uint chosen = NO_ZONE;
        float acc = 0.0;
        for (uint l = 0u; l < PLAN_LANES; l++)
        {
            acc += planTotal[l];
            if (chosen == NO_ZONE && planZone[l] != NO_ZONE && (u < acc || l == PLAN_LANES - 1u))
            {
                chosen = l;
            }
        }
        if (chosen == NO_ZONE)
        {
            for (uint l = 0u; l < PLAN_LANES; l++)
            {
                if (chosen == NO_ZONE && planZone[l] != NO_ZONE) chosen = l;
            }
        }
        slot.zone = planZone[chosen];
        slot.kick = planKick[chosen].xyz / float(max(GP.kickFrames, 1u));
        slot.accel = planAccel[chosen].xyz;
        slot.hitTime = GP.time + planKick[chosen].w;
    }
    gGuideSlots[rank] = slot;
}

// Plans are applied in rank order: a zone's requests this step are numbered
// by rank, and the first ones up to its remaining quota are accepted, so
// quotas resolve the same way every run. A candidate turned away stays armed
// and plans again next step without that (now full) zone.
groupshared uint commitZone[GUIDE_BLOCK];

[shader("compute")]
[numthreads(256, 1, 1)]
void guide_commit(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint r = lid.x;
    uint selected = gGuideScratch[1];
    bool valid = r < selected;
    GuideSlot s = gGuideSlots[min(r, GP.planBudget - 1u)];
    bool request = valid && s.replan == 0u && s.zone != NO_ZONE;
    commitZone[r] = request ? s.zone : NO_ZONE;
    uint claimed = request ? gGuideScratch[SCRATCH_CLAIMS + s.zone] : 0u;
    GROUP_SYNC();
    uint before = 0u;
    bool last = true;
    if (request)
    {
        for (uint q = 0u; q < selected; q++)
        {
            if (commitZone[q] == s.zone)
            {
                if (q < r) before += 1u;
                if (q > r) last = false;
            }
        }
    }
    if (valid)
    {
        GuideBody g = gGuide[s.body];
        g.flags &= ~GFLAG_CANDIDATE;
        if (s.replan != 0u)
        {
            g.accel = s.accel;
            g.sincePlan = 0u;
        }
        else if (s.zone == NO_ZONE)
        {
            g.state = GUIDE_DONE;
        }
        else
        {
            uint quota = gZones[s.zone].quota;
            if (quota == 0u || claimed + before < quota)
            {
                g.state = GUIDE_STEER;
                g.zone = s.zone;
                g.accel = s.accel;
                g.kick = s.kick;
                g.kickLeft = GP.kickFrames;
                g.hitTime = s.hitTime;
                g.sincePlan = 0u;
            }
            if (last)
            {
                uint asked = before + 1u;
                uint room = quota == 0u ? asked : (quota > claimed ? quota - claimed : 0u);
                gGuideScratch[SCRATCH_CLAIMS + s.zone] = claimed + min(asked, room);
            }
        }
        gGuide[s.body] = g;
    }
}
