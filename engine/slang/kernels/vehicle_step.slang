// Vehicle controller kernels (raycast wheels solved inside AVBD).
//
// Attached wheels are not bodies. Each wheel contributes rows that act only on
// its chassis: a soft one-sided suspension (implicit spring-damper), a compliant
// bump stop, an anti-roll pair row, and hard AVBD longitudinal/lateral friction
// rows bounded by a friction ellipse μ(slip)·Fn. Wheel spin is an extra 1-DoF
// variable y (rotation this step) coupled through the longitudinal row; it is
// condensed into the chassis 6×6 block with an exact Schur complement and
// back-substituted after every primal pass. The block is added to the chassis
// primal by primalAccumulate (BODY_VEHICLE), so vehicles add no graph edges.
//
// Step placement (host):
//   apply_commands → vehicle_input → build_aabbs → wheel_query → … → predict_bodies
//   → vehicle_prepare → per iteration { primal colors → solve_dual → vehicle_iterate }
//   → extract_velocity_sleep → vehicle_post → sleep gate/commit
//
// The ground is treated as moving-but-given during the step (its displacement is
// explicit); vehicle_post hands the reaction back to dynamic grounds as a velocity
// change for the next step. Brakes, handbrake and rolling resistance share one
// bounded servo row on y, so locked wheels and parked cars hold positionally.

#include "world_step.slang"
#include "../physics/vehicle_layout.slang"
#include "../physics/soil.slang"

BIND(31) RWStructuredBuffer<float4> gVehicleData; // 29, 30: convex_step

// ---------------------------------------------------------------------------
// Record access
// ---------------------------------------------------------------------------
float4 VD(uint q) { return gVehicleData[q]; }
uint4 VU(uint q) { return asuint(gVehicleData[q]); }
void VSet(uint q, float4 v) { gVehicleData[q] = v; }
void VSetU(uint q, uint4 v) { gVehicleData[q] = asfloat(v); }

uint vehicleCount() { return VU(VH_COUNTS).z; }
uint wheelCountTotal() { return VU(VH_COUNTS).w; }
uint vehRec(uint v) { return VU(VH_BASES).x + v * VEH_STRIDE; }
uint wheelRec(uint w) { return VU(VH_BASES).y + w * WHEEL_STRIDE; }
uint vehicleAabbBlock(uint v) { return VU(VH_BASES).w + 4u * v; }
uint partRec(uint p) { return VU(VH_POOLS).z + p * PART_STRIDE; }

static const float TWO_PI = 6.28318530718;
static const float RAD_PER_S_TO_RPM = 9.54929658551; // 60 / 2π

float4 quatAxisAngle(float3 axis, float angle)
{
    float h = 0.5 * angle;
    return float4(axis * sin(h), cos(h));
}

float moveToward(float cur, float target, float maxDelta)
{
    if (maxDelta <= 0.0) return target;
    float d = target - cur;
    if (abs(d) <= maxDelta) return target;
    return cur + sign(d) * maxDelta;
}

// World inverse inertia applied to a vector (diagonal body-space inverse inertia).
float3 applyInvInertiaWorld(BodyState b, float3 v)
{
    float3 local = quatRotate(quatConjugate(b.orientation), v);
    return quatRotate(b.orientation, local * b.invInertia);
}

// ---------------------------------------------------------------------------
// Piecewise-linear curves: handle = startPoint | count << 16 (NO_INDEX = 1.0)
// ---------------------------------------------------------------------------
float2 curvePoint(uint i)
{
    float4 q = VD(VU(VH_BASES).z + (i >> 1));
    return (i & 1u) == 0u ? q.xy : q.zw;
}

float curveEval(uint h, float x)
{
    if (h == NO_INDEX) return 1.0;
    uint start = h & 0xFFFFu;
    uint count = h >> 16;
    if (count == 0u) return 1.0;
    float2 prev = curvePoint(start);
    if (x <= prev.x) return prev.y;
    for (uint i = 1u; i < count; i++)
    {
        float2 p = curvePoint(start + i);
        if (x <= p.x)
        {
            float t = (x - prev.x) / max(p.x - prev.x, 1e-9);
            return lerp(prev.y, p.y, t);
        }
        prev = p;
    }
    return prev.y;
}

// x of the curve's maximum (the slip where grip peaks).
float curvePeakX(uint h)
{
    if (h == NO_INDEX) return 0.0;
    uint start = h & 0xFFFFu;
    uint count = h >> 16;
    float m = -1e30, x = 0.0;
    for (uint i = 0u; i < count; i++)
    {
        float2 p = curvePoint(start + i);
        if (p.y > m) { m = p.y; x = p.x; }
    }
    return x;
}

float curvePeak(uint h)
{
    if (h == NO_INDEX) return 1.0;
    uint start = h & 0xFFFFu;
    uint count = h >> 16;
    float m = 0.0;
    for (uint i = 0u; i < count; i++) m = max(m, curvePoint(start + i).y);
    return count == 0u ? 1.0 : m;
}

#include "scene_query.slang"

// ---------------------------------------------------------------------------
// Wheel frame helpers
// ---------------------------------------------------------------------------
float3 vehicleUpLocal(uint vr) { return VD(vr + V_UP).xyz; }
float3 vehicleFwdLocal(uint vr) { return VD(vr + V_FWD).xyz; }
float3 axleLocal(uint vr) { return normalize(cross(vehicleUpLocal(vr), vehicleFwdLocal(vr))); }

float4 wheelSteerRot(uint wr, float steer)
{
    return quatAxisAngle(normalize(VD(wr + W_STEERAX).xyz), steer);
}

// Mass properties removed from the chassis when a wheel or part leaves it.
void chassisRemoveMass(uint chassis, uint vr, float m, float3 iLocal, float3 com)
{
    float4 mass = VD(vr + V_MASS);
    float3 d2 = com * com;
    float3 parallel = m * float3(d2.y + d2.z, d2.x + d2.z, d2.x + d2.y);
    float newM = max(mass.x - m, 0.05 * mass.x);
    float3 newI = max(mass.yzw - iLocal - parallel, 0.05 * mass.yzw);
    VSet(vr + V_MASS, float4(newM, newI));
    gBodies[chassis].invMass = 1.0 / newM;
    gBodies[chassis].invInertia = 1.0 / newI;
}

void chassisAddMass(uint chassis, uint vr, float m, float3 iLocal, float3 com)
{
    float4 mass = VD(vr + V_MASS);
    float3 d2 = com * com;
    float3 parallel = m * float3(d2.y + d2.z, d2.x + d2.z, d2.x + d2.y);
    float newM = mass.x + m;
    float3 newI = mass.yzw + iLocal + parallel;
    VSet(vr + V_MASS, float4(newM, newI));
    gBodies[chassis].invMass = 1.0 / newM;
    gBodies[chassis].invInertia = 1.0 / newI;
}

float3 wheelInertiaLocal(uint vr, uint wr)
{
    float wheelMass = VD(wr + W_MISC).x;
    float radius = VD(wr + W_HP).w;
    float width = VD(wr + W_DIR).w;
    float im = wheelMass * (3.0 * radius * radius + width * width) / 12.0;
    float ia = 0.5 * wheelMass * radius * radius;
    float3 axl = axleLocal(vr);
    return float3(im, im, im) + (ia - im) * axl * axl;
}

// Severed pieces ignore their former chassis until they have separated.
static const float SEVER_GRACE = 0.5;

void setColliderFilter(uint start, uint count, uint value)
{
    for (uint k = 0u; k < count; k++) gColliders[start + k].generation = value;
}

// aabbOverlap: world_step.slang.

// True while any collider in [start, start+count) overlaps an attached chassis part.
bool overlapsChassis(uint vr, uint start, uint count)
{
    uint base = HOST_S(1);
    uint4 pool2 = VU(vr + V_POOL2);
    for (uint k = 0u; k < count; k++)
    {
        Aabb a = gAabbs[base + start + k];
        for (uint pi = 0u; pi < pool2.y; pi++)
        {
            uint pr = partRec(pool2.x + pi);
            if (asuint(VD(pr + P_COM).w) != PS_ATTACHED) continue;
            uint4 pc = VU(pr + P_CFG);
            for (uint c = 0u; c < pc.y; c++)
                if (aabbOverlap(a, gAabbs[base + pc.x + c])) return true;
        }
    }
    return false;
}

void updateSeverGrace(uint vr, uint4 cfg, float dt)
{
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        float4 gv = VD(wr + W_GV);
        if ((VU(wr + W_C0).z & WD_DETACHED) == 0u || gv.w <= 0.0) continue;
        float grace = gv.w - dt;
        uint4 park = VU(wr + W_PARK);
        if (grace <= 0.0)
        {
            if (overlapsChassis(vr, park.y, park.z)) grace = dt;
            else setColliderFilter(park.y, park.z, 0u);
        }
        VSet(wr + W_GV, float4(gv.xyz, max(grace, 0.0)));
    }
    uint4 pool2 = VU(vr + V_POOL2);
    for (uint pi = 0u; pi < pool2.y; pi++)
    {
        uint pr = partRec(pool2.x + pi);
        float4 park = VD(pr + P_PARK);
        if (park.z <= 0.0 || asuint(VD(pr + P_COM).w) != PS_DETACHED) continue;
        float grace = park.z - dt;
        uint4 pc = VU(pr + P_CFG);
        if (grace <= 0.0)
        {
            if (overlapsChassis(vr, pc.x, pc.y)) grace = dt;
            else setColliderFilter(pc.x, pc.y, 0u);
        }
        VSet(pr + P_PARK, float4(park.xy, max(grace, 0.0), park.w));
    }
}

// Activate a parked body as a free rigid body with the given pose and velocity.
void activateParkedBody(uint pb, float3 pos, float4 rot, float3 v, float3 w)
{
    BodyState b = gBodies[pb];
    b.position = pos;
    b.orientation = rot;
    b.initialPos = pos;
    b.initialRot = rot;
    b.linearVel = v;
    b.angularVel = w;
    b.flags = BODY_DYNAMIC | BODY_AWAKE;
    b.sleepTimer = 0.0;
    b.generation = HOST_S(17);
    gBodies[pb] = b;
    // Adaptive warm start compares against the previous step's velocity.
    BODY_PREV(pb).linearVel = v;
}

void wakeBody(uint i)
{
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b)) return;
    if (bodyIsSleeping(b) || b.sleepTimer > 0.0)
    {
        gBodies[i].flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[i].sleepTimer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// vehicle_input: controls, functional state, severing, steering, wake
// ---------------------------------------------------------------------------
void detachWheel(uint v, uint vr, uint chassis, uint wi, bool spawnBody)
{
    uint w = VU(vr + V_CFG).y + wi;
    uint wr = wheelRec(w);
    uint4 dyn = VU(wr + W_C0);
    if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) return;
    BodyState c = gBodies[chassis];
    float wheelMass = VD(wr + W_MISC).x;
    float3 hp = VD(wr + W_HP).xyz;
    float3 axl = axleLocal(vr);
    float3 iLocal = wheelInertiaLocal(vr, wr);
    chassisRemoveMass(chassis, vr, wheelMass, iLocal, hp + VD(wr + W_DIR).xyz * VD(wr + W_SUSP).y);
    if (spawnBody)
    {
        uint pb = VU(wr + W_PARK).x;
        float3 center = VD(wr + W_O_POS).xyz;
        float4 rot = VD(wr + W_O_ROT);
        if (dot(rot, rot) < 0.5)
        {
            center = transformPoint(c.position, c.orientation, hp + VD(wr + W_DIR).xyz * VD(wr + W_SUSP).y);
            rot = c.orientation;
        }
        float omega = VD(wr + W_W).x;
        float3 axleWorld = quatRotate(rot, axl);
        float3 vel = c.linearVel + cross(c.angularVel, center - c.position);
        float3 ang = c.angularVel + axleWorld * omega;
        activateParkedBody(pb, center, rot, vel, ang);
        setColliderFilter(VU(wr + W_PARK).y, VU(wr + W_PARK).z, chassis + 1u);
        float4 gv = VD(wr + W_GV);
        VSet(wr + W_GV, float4(gv.xyz, SEVER_GRACE));
    }
    dyn.z = (dyn.z & ~(WD_CONTACT | WD_STICK_L | WD_STICK_T | WD_SUSP | WD_BUMP | WD_SLIDE))
        | (spawnBody ? WD_DETACHED : WD_DISABLED);
    VSetU(wr + W_C0, dyn);
    gBodies[chassis].generation = HOST_S(17);
}

void detachPart(uint vr, uint chassis, uint p, bool spawnBody)
{
    uint pr = partRec(p);
    float4 com = VD(pr + P_COM);
    if (asuint(com.w) != PS_ATTACHED) return;
    uint4 cfg = VU(pr + P_CFG);
    float4 mass = VD(pr + P_MASS);
    BodyState c = gBodies[chassis];
    chassisRemoveMass(chassis, vr, mass.x, mass.yzw, com.xyz);
    uint target = VU(vr + V_POOL2).z;
    float3 offset = float3(0, 0, 0);
    if (spawnBody)
    {
        target = VU(pr + P_PARK).x;
        float3 pos = transformPoint(c.position, c.orientation, com.xyz);
        float3 vel = c.linearVel + cross(c.angularVel, pos - c.position);
        activateParkedBody(target, pos, c.orientation, vel, c.angularVel);
        offset = com.xyz;
    }
    for (uint k = 0u; k < cfg.y; k++)
    {
        uint ci = cfg.x + k;
        gColliders[ci].body = target;
        gColliders[ci].localPos = gColliders[ci].localPos - offset;
        gColliders[ci].generation = spawnBody ? chassis + 1u : 0u;
    }
    if (spawnBody)
    {
        float4 park = VD(pr + P_PARK);
        VSet(pr + P_PARK, float4(park.xy, SEVER_GRACE, park.w));
    }
    VSet(pr + P_COM, float4(com.xyz, asfloat(spawnBody ? PS_DETACHED : PS_DISABLED)));
    gBodies[chassis].generation = HOST_S(17);

    // Parts bolted to this one (P_TREE parent, transitively) leave with it as one
    // rigid piece: their colliders join its body (or go with it), their mass
    // leaves the chassis and joins that body. Children have higher indices.
    uint4 pool2 = VU(vr + V_POOL2);
    uint local = p - pool2.x;
    uint moving = 1u << local;
    float mTotal = mass.x;
    float3 mCom = mass.x * com.xyz;
    for (uint ci = local + 1u; ci < pool2.y && ci < 32u; ci++)
    {
        uint cr = partRec(pool2.x + ci);
        float4 ccom = VD(cr + P_COM);
        uint parent = VU(cr + P_TREE).x;
        if (asuint(ccom.w) != PS_ATTACHED || parent >= 32u || (moving & (1u << parent)) == 0u) continue;
        moving |= 1u << ci;
        float4 cm = VD(cr + P_MASS);
        chassisRemoveMass(chassis, vr, cm.x, cm.yzw, ccom.xyz);
        uint4 cc = VU(cr + P_CFG);
        for (uint k = 0u; k < cc.y; k++)
        {
            uint col = cc.x + k;
            gColliders[col].body = target;
            gColliders[col].localPos = gColliders[col].localPos - offset;
            gColliders[col].generation = spawnBody ? chassis + 1u : 0u;
        }
        if (spawnBody)
        {
            float4 park = VD(cr + P_PARK);
            VSet(cr + P_PARK, float4(park.xy, SEVER_GRACE, park.w));
        }
        mTotal += cm.x;
        mCom += cm.x * ccom.xyz;
        VSetU(cr + P_TREE, uint4(parent, local, 0u, 0u));
        VSet(cr + P_COM, float4(ccom.xyz, asfloat(spawnBody ? PS_DETACHED : PS_DISABLED)));
    }
    if (!spawnBody || moving == (1u << local)) return;
    // One rigid piece: its origin moves to the combined centre of mass (chassis
    // axes, the body's at detach), with that point's velocity, and inertia
    // about it; colliders keep their places. The parent's P_COM records the
    // piece's origin (renderers re-base the piece's parts on it).
    float3 centre = mCom / mTotal;
    float3 shift = centre - com.xyz;
    float3 inertia = float3(0, 0, 0);
    for (uint ci = local; ci < pool2.y && ci < 32u; ci++)
    {
        if ((moving & (1u << ci)) == 0u) continue;
        uint cr = partRec(pool2.x + ci);
        float4 cm = VD(cr + P_MASS);
        float3 d = VD(cr + P_COM).xyz - centre;
        float3 d2 = d * d;
        inertia += cm.yzw + cm.x * float3(d2.y + d2.z, d2.x + d2.z, d2.x + d2.y);
        uint4 cc = VU(cr + P_CFG);
        for (uint k = 0u; k < cc.y; k++) gColliders[cc.x + k].localPos = gColliders[cc.x + k].localPos - shift;
    }
    BodyState b = gBodies[target];
    b.position = transformPoint(c.position, c.orientation, centre);
    b.initialPos = b.position;
    b.linearVel = c.linearVel + cross(c.angularVel, b.position - c.position);
    b.invMass = 1.0 / mTotal;
    b.invInertia = 1.0 / inertia;
    gBodies[target] = b;
    BODY_PREV(target).linearVel = b.linearVel;
    VSet(pr + P_COM, float4(centre, asfloat(PS_DETACHED)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_input(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    if ((cfg.w & VF_ENABLED) == 0u) return;
    uint chassis = cfg.x;
    AvbdParams p = PARAMS;
    float dt = p.dt;

    // --- severing requests (applied once; masks are latched in V_D4) ---
    // Host requests, plus mounts vehicle_loads found overloaded last step.
    uint4 applied = VU(vr + V_D4);
    uint4 in1 = VU(vr + V_IN1);
    uint4 in3 = VU(vr + V_IN3);
    uint4 brk = VU(vr + V_BRK);
    bool changed = false;
    uint newDetach = (in1.z | brk.y) & ~applied.x;
    uint newDisable = in1.w & ~applied.y & ~(in1.z | brk.y);
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint bit = 1u << wi;
        if ((newDetach & bit) != 0u) { detachWheel(v, vr, chassis, wi, true); changed = true; }
        else if ((newDisable & bit) != 0u) { detachWheel(v, vr, chassis, wi, false); changed = true; }
    }
    applied.x |= newDetach;
    applied.y |= newDisable;
    uint4 pool2 = VU(vr + V_POOL2);
    uint newPartDetach = (in3.x | brk.x) & ~applied.z;
    uint newPartDisable = in3.y & ~applied.w & ~(in3.x | brk.x);
    for (uint pi = 0u; pi < pool2.y && pi < 32u; pi++)
    {
        uint bit = 1u << pi;
        if ((newPartDetach & bit) != 0u) { detachPart(vr, chassis, pool2.x + pi, true); changed = true; }
        else if ((newPartDisable & bit) != 0u) { detachPart(vr, chassis, pool2.x + pi, false); changed = true; }
    }
    applied.z |= newPartDetach;
    applied.w |= newPartDisable;
    VSetU(vr + V_D4, applied);
    if ((brk.x | brk.y) != 0u)
    {
        VSetU(vr + V_BRK, uint4(0u, 0u, brk.z | brk.x, brk.w | brk.y));
        changed = true;
    }

    // --- wheels attached at runtime by the host (spare slots) ---
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & WD_PENDING_ADD) == 0u) continue;
        float3 hp = VD(wr + W_HP).xyz + VD(wr + W_DIR).xyz * VD(wr + W_SUSP).y;
        chassisAddMass(chassis, vr, VD(wr + W_MISC).x, wheelInertiaLocal(vr, wr), hp);
        dyn.z &= ~(WD_PENDING_ADD | WD_DISABLED | WD_DETACHED);
        VSetU(wr + W_C0, dyn);
        changed = true;
    }

    // --- functional state: derived from part ownership, then host overrides ---
    uint wheelMask = 0u, driveMask = 0u, steerMask = 0u, brakeMask = 0u;
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint wf = VU(wr + W_CFG).y;
        uint wd = VU(wr + W_C0).z;
        if ((wf & WF_ENABLED) == 0u || (wd & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
        uint bit = 1u << wi;
        wheelMask |= bit;
        brakeMask |= bit;
        if ((wf & WF_DRIVEN) != 0u) driveMask |= bit;
        if ((wf & WF_STEER) != 0u) steerMask |= bit;
    }
    bool engineConnected = (in3.z & 1u) != 0u;
    uint partsLost = 0u;
    for (uint pi = 0u; pi < pool2.y && pi < 32u; pi++)
    {
        uint pr = partRec(pool2.x + pi);
        uint4 pc = VU(pr + P_CFG);
        float4 com = VD(pr + P_COM);
        uint state = asuint(com.w);
        if (state == PS_ATTACHED && pc.y > 0u && gColliders[pc.x].body != chassis)
        {
            // Someone else (fracture, gameplay) moved the part off the chassis.
            float4 m = VD(pr + P_MASS);
            chassisRemoveMass(chassis, vr, m.x, m.yzw, com.xyz);
            state = PS_LOST;
            VSet(pr + P_COM, float4(com.xyz, asfloat(state)));
            changed = true;
        }
        if (state == PS_ATTACHED) continue;
        partsLost |= 1u << pi;
        if (pc.z == PR_ENGINE) engineConnected = false;
        else if (pc.z == PR_DRIVESHAFT) driveMask &= ~pc.w;
        else if (pc.z == PR_STEERING) steerMask &= ~pc.w;
        else if (pc.z == PR_BRAKE) brakeMask &= ~pc.w;
    }
    uint4 over = VU(vr + V_IN2);
    wheelMask &= over.x;
    driveMask &= over.y & wheelMask;
    steerMask &= over.z & wheelMask;
    brakeMask &= over.w & wheelMask;

    uint4 prevMasks = VU(vr + V_O2);
    if (prevMasks.x != wheelMask || prevMasks.y != driveMask || prevMasks.z != steerMask) changed = true;
    uint4 o3 = VU(vr + V_O3);
    if ((o3.x != 0u) != engineConnected) changed = true;
    VSetU(vr + V_O2, uint4(wheelMask, driveMask, steerMask, prevMasks.w + (changed ? 1u : 0u)));
    VSetU(vr + V_O3, uint4(engineConnected ? 1u : 0u, partsLost, o3.z, o3.w));

    // --- input smoothing ---
    float4 in0 = VD(vr + V_IN0);
    float4 d1 = VD(vr + V_D1);
    float4 d2 = VD(vr + V_D2);
    float4 resp = VD(vr + V_RESP);
    BodyState c = gBodies[chassis];
    float3 fwdW = quatRotate(c.orientation, vehicleFwdLocal(vr));
    float speed = dot(c.linearVel, fwdW);
    float forwardIn = clamp(in0.x, -1.0, 1.0);
    float brakeIn = clamp(in0.z, 0.0, 1.0);
    // Throttle against the motion brakes first. Not for a steering tank: a pivot
    // swings the hull's forward speed negative and would brake the turn away.
    bool steeringTank = (cfg.w & VF_TRACKED) != 0u && abs(in0.y) > 0.1;
    if ((cfg.w & VF_REVERSE_BRAKE) != 0u && !steeringTank && forwardIn * speed < 0.0 && abs(speed) > 1.0)
    {
        brakeIn = max(brakeIn, abs(forwardIn));
        forwardIn = 0.0;
    }
    float steerTarget = clamp(in0.y, -1.0, 1.0);
    if ((cfg.w & VF_TRACKED) != 0u)
    {
        // Arcade skid steer: each track gets throttle ± steer (left = f + s, right =
        // f − s), so steer always yaws the hull the same way: a pivot from rest, a
        // track-ratio turn while driving, the same rotation sense in reverse. The
        // faster track sets the throttle and drive direction; the ratios are each
        // track's share of it (negative = counter-rotating), feeding Jolt's
        // tracked drivetrain unchanged.
        float4 tr = VD(vr + V_TRACK);
        float turnAW = VD(vr + V_TRACK_D).w;
        float s = steerTarget * tr.z;
        float trackL = forwardIn + s, trackR = forwardIn - s;
        float mag = max(abs(trackL), abs(trackR));
        float ratioL = 1.0, ratioR = 1.0;
        if (mag > 0.02)
        {
            float dir = trackL + trackR < -1e-4 ? -1.0 : 1.0;
            ratioL = trackL / (dir * mag);
            ratioR = trackR / (dir * mag);
            forwardIn = dir * min(mag, 1.0);
            // Both tracks the same way at speed: cap the turn's lateral acceleration
            // (yaw ≈ (1 − ratio)·v / W, lateral accel ≈ (1 − ratio)·v² / W).
            if (ratioL * ratioR >= 0.0 && turnAW > 0.0 && speed * speed > 1e-3)
            {
                float floorRatio = 1.0 - turnAW / (speed * speed);
                ratioL = max(ratioL, floorRatio);
                ratioR = max(ratioR, floorRatio);
            }
        }
        VSet(vr + V_TRACK_D, float4(ratioL, ratioR, forwardIn, turnAW));
        steerTarget = 0.0;
    }
    float forward = moveToward(d1.z, forwardIn, resp.x * dt);
    float brake = moveToward(d1.w, brakeIn, resp.y * dt);
    float handbrake = moveToward(d2.x, clamp(in0.w, 0.0, 1.0), resp.z * dt);
    float steerRate = VD(vr + V_STEER).w;
    uint steerCurve = asuint(VD(vr + V_AERO0).w);
    steerTarget *= curveEval(steerCurve, abs(speed));
    float steer = moveToward(d2.y, steerTarget, steerRate * dt);
    VSet(vr + V_D1, float4(d1.x, d1.y, forward, brake));
    VSet(vr + V_D2, float4(handbrake, steer, asfloat(wheelMask), asfloat(driveMask)));
    uint4 d3 = VU(vr + V_D3);
    d3.x = steerMask;
    d3.z = brakeMask;

    // --- wake on intent (inputs never wake a parked car by themselves; intent does) ---
    bool intent = abs(forwardIn) > 1e-4 || abs(steerTarget - steer) > 1e-4 || abs(steer - d2.y) > 1e-5
        || changed || in3.w != d3.w || ((cfg.w & VF_TRACKED) != 0u && abs(in0.y) > 0.02);
    d3.w = in3.w;
    VSetU(vr + V_D3, d3);
    if (intent) wakeBody(chassis);

    // --- per-wheel steer angle (Ackermann; motorcycles cap it by lean) ---
    float4 st = VD(vr + V_STEER);
    float4 bike0 = VD(vr + V_BIKE0);
    bool leanLimit = bike0.x > 0.0 && VD(vr + V_BIKE1).z > 0.5;
    float gScale = VD(vr + V_AERO0).z;
    float g = abs(PARAMS.gravityY) * (gScale > 0.0 ? gScale : 1.0);
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        float maxSteer = VD(wr + W_STEERAX).w;
        float side = VD(wr + W_FWD).w;
        float angle = 0.0;
        if ((steerMask & (1u << wi)) != 0u)
        {
            angle = -steer * maxSteer; // + = left (Jolt convention); + input = right
            float v2 = speed * speed;
            float cosCaster = dot(normalize(VD(wr + W_STEERAX).xyz), vehicleUpLocal(vr));
            if (leanLimit && v2 > 1e-6 && cosCaster > 1e-6 && st.y > 0.0)
            {
                // SteerAngle = asin(wheelbase·tan(maxLean)·g / (v²·cos caster)) (Jolt).
                float lim = asin(clamp(st.y * tan(bike0.x) * g / (v2 * cosCaster), 0.0, 1.0));
                angle = clamp(angle, -lim, lim);
            }
            if (st.x > 0.0 && side != 0.0 && abs(angle) > 1e-4 && st.y > 0.0)
            {
                float turnR = st.y / tan(abs(angle));
                bool inner = (angle > 0.0) == (side > 0.0);
                float ack = atan(st.y / max(turnR + (inner ? -0.5 : 0.5) * st.z, 1e-3));
                angle = sign(angle) * lerp(abs(angle), ack, st.x);
            }
        }
        float4 c5 = VD(wr + W_C5);
        VSet(wr + W_C5, float4(c5.xyz, angle));
    }
}

// ---------------------------------------------------------------------------
// wheel_query: one workgroup per wheel casts along the suspension at x⁻
#include "vehicle_step_wheels.slang"
#include "vehicle_step_ground.slang"
#include "vehicle_step_prepare.slang"
#include "vehicle_step_soil.slang"
