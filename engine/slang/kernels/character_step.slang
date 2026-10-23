// Character controllers, in-step scene queries and projectile segment casts
// (engine/docs/characters.md, engine/docs/ccd.md row 5).
//
// A character is a kinematic mover (Jolt's CharacterVirtual, on the GPU): one
// workgroup per character, inside the step's x⁻ window (after the broadphase,
// whose grid it walks). It gathers the colliders around its capsule, turns
// them into contact planes that move with their bodies, solves its
// displacement against them (slide, crease, stop), verifies the move with a
// capsule cast, steps up stairs and sticks to the floor, probes the ground,
// and drives an inner kinematic capsule body to its new pose: the world's
// dynamic bodies collide with that capsule in the same solve. char_post reads
// the ground's motion for the next step and puts the character's weight on a
// dynamic support.
//
// Step placement (engine_exec::rigid_plan::encode_step):
//   … broadphase_query → broadphase_fast → wheel_query → vehicle_link
//   → projectile_step → apply_impulses (late) → query_batch → char_move
//   → … narrowphase … → extract_velocity_sleep → vehicle_post → char_post

#include "world_step.slang"
#include "scene_query.slang"
#include "../physics/character_layout.slang"

BIND(32) RWStructuredBuffer<float4> gCharData;

float4 CD(uint q) { return gCharData[q]; }
uint4 CU(uint q) { return asuint(gCharData[q]); }
void CSet(uint q, float4 v) { gCharData[q] = v; }
void CSetU(uint q, uint4 v) { gCharData[q] = asfloat(v); }

uint charCount() { return CU(CH_COUNTS).y; }
uint charRec(uint c) { return CU(CH_BASES).x + c * CHAR_STRIDE; }
uint queryCount() { return CU(CH_COUNTS).w; }
uint queryRec(uint q) { return CU(CH_BASES).y + q * QUERY_STRIDE; }
uint projCount() { return CU(CH_MISC).x; }
uint projRec(uint p) { return CU(CH_BASES).z + p * PROJ_STRIDE; }
bool charFullScan() { return CU(CH_MISC).y == 1u; }

static const float3 CHAR_UP = float3(0, 1, 0);
static const uint OVERFLOW_CHAR_CANDIDATES = 16384u;

// Upright capsule collider record for casts (segment along +Y).
Collider capsuleShape(float r, float hh)
{
    Collider c;
    c.body = NO_INDEX;
    c.shape = SHAPE_CAPSULE;
    c.material = 0u;
    c.generation = 0u;
    c.localPos = float3(0, 0, 0);
    c.density = 0.0;
    c.localRot = float4(0, 0, 0, 1);
    c.params = float4(r, hh, 0.0, r + hh);
    return c;
}

// ---------------------------------------------------------------------------
// Workgroup state of one character
// ---------------------------------------------------------------------------
groupshared AUint gsCandCount;
groupshared uint gsCand[CHAR_MAX_CANDIDATES];
groupshared float4 gsPlane[CHAR_MAX_CANDIDATES];  // normal toward the character, gap (m)
groupshared float4 gsMove[CHAR_MAX_CANDIDATES];   // obstacle displacement at the contact this step, asfloat(info)
groupshared float4 gsSurf[CHAR_MAX_CANDIDATES];   // surface normal under the contact (faces, not edges)
groupshared float4 gsShare[4];

static const uint PI_VALID = 1u;
static const uint PI_BLOCKS = 2u;   // static, kinematic or heavy dynamic: the character cannot push it
static const uint PI_DYNAMIC = 4u;

struct CollectVisitor : IColliderVisitor
{
    float3 mn; float3 mx;
    QueryFilter filter;
    [mutating] void visit(uint ci)
    {
        Collider col = gColliders[ci];
        if (!queryAccepts(col, filter)) return;
        if (col.shape != SHAPE_PLANE && !segmentHitsAabb(mn, mx, gAabbs[HOST_S(1) + ci])) return;
        uint slot;
        AADD3(gsCandCount, 1u, slot);
        if (slot < CHAR_MAX_CANDIDATES) gsCand[slot] = ci;
    }
};

uint candidateCount() { return min(ALOAD(gsCandCount), CHAR_MAX_CANDIDATES); }

// Deepest contact of the character's capsule (centre P, upright) with collider
// ci within `spec` of touching: plane normal toward the character, gap (< 0:
// overlap) and the obstacle's contact point. False when farther than spec.
bool capsuleContact(uint ci, float3 P, float r, float hh, float spec, out float3 n, out float gap, out float3 point)
{
    n = CHAR_UP;
    gap = 1e30;
    point = P;
    Collider col = gColliders[ci];
    BodyState b = gBodies[col.body];
    float4 qCap = float4(0, 0, 0, 1);
    float3 pc; float4 qc; colliderPose(col, b, pc, qc);
    BoxManifold m;
    m.count = 0;
    m.normal = CHAR_UP;
    bool charIsA = true;
    if (col.shape == SHAPE_SPHERE)
    {
        float3 a0 = P - CHAR_UP * hh;
        float t = clamp(dot(pc - a0, CHAR_UP) / max(2.0 * hh, 1e-9), 0.0, 1.0);
        float3 q = a0 + CHAR_UP * (2.0 * hh * t);
        float3 d = q - pc;
        float dist = length(d);
        float R = col.params.x;
        if (dist - r - R > spec) return false;
        n = dist > 1e-9 ? d / dist : CHAR_UP;
        gap = dist - r - R;
        point = pc + n * R;
        return true;
    }
    if (col.shape == SHAPE_CAPSULE)
        m = collideCapsules(P, qCap, r, hh, pc, qc, col.params.x, col.params.y, spec);
    else if (col.shape == SHAPE_BOX)
    {
        m = collideBoxCapsule(pc, qc, col.params.xyz, P, qCap, r, hh, spec);
        charIsA = false;
    }
    else if (col.shape == SHAPE_CONVEX)
    {
        Frame fh; fh.p = pc; fh.q = qc;
        Frame fc; fc.p = P; fc.q = qCap;
        m = convexCapsuleManifold(hullOf(col), fh, fc, r, hh, true, spec);
        charIsA = false;
    }
    else
        m = collideCapsulePlane(P, qCap, r, hh, col.params.xyz, col.params.w, spec);
    if (m.count <= 0) return false;
    // Normal from the obstacle toward the character.
    n = charIsA ? -m.normal : m.normal;
    for (int k = 0; k < m.count; k++)
    {
        float3 pChar = charIsA ? m.pts[k].pA : m.pts[k].pB;
        float3 pObs = charIsA ? m.pts[k].pB : m.pts[k].pA;
        float g = dot(n, pChar - pObs);
        if (g < gap) { gap = g; point = pObs; }
    }
    return gap <= spec;
}

// Displacement of a body's point p over the step (0 for statics and sleepers).
float3 bodyPointMotion(BodyState b, float3 p)
{
    if (!bodyMovingOrKinematic(b)) return float3(0, 0, 0);
    return (b.linearVel + cross(b.angularVel, p - b.position)) * PARAMS.dt;
}

// Capsule cast of the character from S along disp against the gathered
// candidates, each in its own frame: MOTION_NONE leaves obstacles at x⁻,
// MOTION_DURING moves them by their step motion during the cast (relative
// displacement), MOTION_AFTER places them where the step leaves them (the
// cast starts at S − U). Lanes write their earliest fraction of disp to
// gsRayT / gsRayC; then rayReduce.
static const uint MOTION_NONE = 0u;
static const uint MOTION_DURING = 1u;
static const uint MOTION_AFTER = 2u;

void castCandidates(uint lane, bool live, float3 S, float3 disp, float r, float hh, float skin, uint motion, bool blockingOnly,
    bool upFacingOnly)
{
    // The ground probe (which moves nothing) counts only surfaces that face
    // up: a riser beside the capsule is not ground and must not hide the floor
    // below. Casts that move the character stop at whatever they hit first.
    float bestF = 1e30;
    uint bestC = NO_INDEX;
    uint n = live ? candidateCount() : 0u;
    Collider cap = capsuleShape(r, hh);
    for (uint k = lane; k < n; k += 64u)
    {
        uint info = asuint(gsMove[k].w);
        if ((info & PI_VALID) == 0u) continue;
        if (blockingOnly && (info & PI_BLOCKS) == 0u) continue;
        float3 rel = disp - (motion == MOTION_DURING ? gsMove[k].xyz : float3(0, 0, 0));
        float3 start = S - (motion == MOTION_AFTER ? gsMove[k].xyz : float3(0, 0, 0));
        float L = length(rel);
        if (L < 1e-6) continue;
        float t;
        float3 nrm;
        if (!shapeCastCollider(cap, start, float4(0, 0, 0, 1), rel / L, L + skin, gsCand[k], t, nrm)) continue;
        if (t < 0.0) continue; // overlapping at the start: the planes own it
        if (upFacingOnly && castSurfaceNormal(gsCand[k], r, hh, start, rel / L, t, nrm).y < 0.1) continue;
        // Stop `skin` short of the hit along the cast (the plane solve keeps
        // the perpendicular skin; this cast only catches what it missed).
        float back = skin;
        float f = clamp((t - back) / L, 0.0, 1.0);
        if (f < bestF) { bestF = f; bestC = k; }
    }
    gsRayT[lane] = bestF;
    gsRayC[lane] = bestC;
}

// Surface normal under a capsule cast hit (start S along unit dir, distance t,
// contact normal n): the touching point is the capsule's support point toward
// the obstacle at the hit pose.
float3 castSurfaceNormal(uint ci, float r, float hh, float3 S, float3 dir, float t, float3 n)
{
    // The cast's normal is approximate at edges: take the exact closest
    // features at the hit pose (the narrowphase capsule tests) instead.
    float3 hit = S + dir * t;
    float3 cn; float gap; float3 point;
    if (capsuleContact(ci, hit, r, hh, 0.02, cn, gap, point))
        return surfaceNormal(ci, point, cn, CHAR_UP, 0.005);
    float3 p = shapeSupport(capsuleShape(r, hh), hit, float4(0, 0, 0, 1), -n);
    return surfaceNormal(ci, p, n, CHAR_UP, 0.01);
}

// ---------------------------------------------------------------------------
// char_move: one workgroup per character
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void char_move(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint c = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    bool live = c < charCount();
    uint cr = live ? charRec(c) : 0u;
    uint4 cfg = live ? CU(cr + C_CFG) : uint4(0, 0, 0, 0);
    live = live && (cfg.y & CF_ENABLED) != 0u;
    float4 shape = live ? CD(cr + C_SHAPE) : float4(0.3, 0.6, 0.02, 0.7);
    float4 mv = live ? CD(cr + C_MOVE) : float4(0, 0, 0, 0);
    float4 mass = live ? CD(cr + C_MASS) : float4(1, 1, 0.1, 1);
    float4 inp = live ? CD(cr + C_IN) : float4(0, 0, 0, 0);
    float4 pos = live ? CD(cr + C_POS) : float4(0, 0, 0, 0);
    float4 vel = live ? CD(cr + C_VEL) : float4(0, 0, 0, asfloat(GS_IN_AIR));
    float4 gvel = live ? CD(cr + C_GVEL) : float4(0, 0, 0, 0);
    float4 gnd = live ? CD(cr + C_GND) : float4(0, 1, 0, asfloat(NO_INDEX));
    float r = shape.x, hh = shape.y, skin = shape.z, cosSlope = shape.w;
    float dt = PARAMS.dt;
    float3 g = float3(0, PARAMS.gravityY * mass.w, 0);
    float3 P = pos.xyz;
    uint state = asuint(vel.w);
    bool wasGround = state == GS_ON_GROUND;
    // 1. Velocity (every lane computes the same values).
    uint jumpSeq = asuint(inp.w);
    bool jump = wasGround && jumpSeq != asuint(pos.w);
    float3 V;
    if (wasGround && !jump)
    {
        // Walk along the ground plane, carried by the ground.
        float3 n = gnd.xyz;
        float3 h = float3(inp.x, 0.0, inp.z);
        float vy = abs(n.y) > 1e-3 ? -(n.x * h.x + n.z * h.z) / n.y : 0.0;
        V = float3(h.x, vy, h.z) + gvel.xyz;
    }
    else
    {
        V = vel.xyz;
        if (jump) V = float3(inp.x + gvel.x, mv.z + max(gvel.y, 0.0), inp.z + gvel.z);
        float k = min(mv.w * dt, 1.0);
        V.x += (inp.x - V.x) * k;
        V.z += (inp.z - V.z) * k;
        V += g * dt;
    }
    float3 D = V * dt;
    // 2. Gather: everything near the capsule swept along D, plus room for the
    // step-up and stick-to-floor casts.
    float reach = r + hh + skin + max(mv.x, mv.y) + 0.05;
    if (lane == 0u) ASTORE(gsCandCount, 0u);
    GROUP_SYNC();
    {
        CollectVisitor v;
        v.mn = min(P, P + D) - reach;
        v.mx = max(P, P + D) + reach;
        v.filter.excludeBody = cfg.x;
        v.filter.mask = cfg.z;
        sceneWalk(lane, live, P, P + D, reach, charFullScan(), v);
    }
    GROUP_SYNC();
    uint nCand = live ? candidateCount() : 0u;
    if (live && lane == 0u && ALOAD(gsCandCount) > CHAR_MAX_CANDIDATES)
        AOR2(gCounters[0].overflowFlags, OVERFLOW_CHAR_CANDIDATES);
    // 3. Contact planes at P, each with its body's motion this step.
    float spec = length(D) + skin + 0.05;
    for (uint k = lane; k < nCand; k += 64u)
    {
        uint ci = gsCand[k];
        float3 n; float gap; float3 point;
        uint info = 0u;
        float3 U = float3(0, 0, 0);
        float3 sn = CHAR_UP;
        if (capsuleContact(ci, P, r, hh, spec, n, gap, point))
        {
            sn = surfaceNormal(ci, point, n, CHAR_UP, 0.01);
            BodyState ob = gBodies[gColliders[ci].body];
            info = PI_VALID;
            bool dyn = bodyIsDynamic(ob);
            if (dyn) info |= PI_DYNAMIC;
            float m = ob.invMass > 0.0 ? 1.0 / ob.invMass : 1e30;
            if (!dyn || m >= mass.x * mass.y) info |= PI_BLOCKS;
            U = bodyPointMotion(ob, point);
        }
        gsPlane[k] = float4(n, gap);
        gsMove[k] = float4(U, asfloat(info));
        gsSurf[k] = float4(sn, 0.0);
    }
    GROUP_SYNC();
    // 4. Plane solve (lane 0): the displacement closest to D that keeps every
    // blocking plane `skin` away, each plane moving with its body. Cyclic
    // projection: one plane slides, two meet in a crease, three stop.
    if (lane == 0u)
    {
        float3 X = D;
        uint blocking = 0u;
        for (uint it = 0u; it < 16u; it++)
        {
            bool moved = false;
            for (uint k = 0u; k < nCand; k++)
            {
                uint info = asuint(gsMove[k].w);
                if ((info & (PI_VALID | PI_BLOCKS)) != (PI_VALID | PI_BLOCKS)) continue;
                float3 n = gsPlane[k].xyz;
                float need = min(skin - gsPlane[k].w, mass.z) + dot(n, gsMove[k].xyz);
                float cv = dot(n, X) - need;
                if (cv < -1e-6)
                {
                    X -= n * cv;
                    moved = true;
                }
            }
            if (it == 0u)
                for (uint k = 0u; k < nCand; k++)
                    if ((asuint(gsMove[k].w) & (PI_VALID | PI_BLOCKS)) == (PI_VALID | PI_BLOCKS)) blocking++;
            if (!moved) break;
        }
        gsShare[0] = float4(X, asfloat(blocking));
    }
    GROUP_SYNC();
    float3 X = gsShare[0].xyz;
    uint blocking = asuint(gsShare[0].w);
    uint outFlags = jump ? CO_JUMPED : 0u;
    // 5. Verify along X: geometry the planes missed (edges, thin pieces).
    castCandidates(lane, live, P, X, r, hh, skin, MOTION_DURING, true, false);
    rayReduce(lane);
    float castF = gsRayT[0];
    GROUP_SYNC(); // every lane has read the result before the next cast writes
    if (castF < 1.0)
    {
        X *= castF;
        outFlags |= CO_CAST_CUT;
    }
    // 6. Step up: on the ground, blocked horizontally by something steep.
    float3 Dh = float3(D.x, 0.0, D.z);
    float3 Xh = float3(X.x, 0.0, X.z);
    bool tryStep = live && wasGround && !jump && mv.x > 0.0 && length(Dh) > 1e-4 &&
        length(Xh) < 0.9 * length(Dh);
    float stepped = 0.0;
    if (tryStep) // workgroup-uniform: every lane reads the same record
    {
        // Up (ceiling), across, then down onto the step.
        castCandidates(lane, live, P, CHAR_UP * mv.x, r, hh, skin, MOTION_NONE, true, false);
        rayReduce(lane);
        float up = mv.x * min(gsRayT[0], 1.0);
        GROUP_SYNC();
        float3 P1 = P + CHAR_UP * up;
        castCandidates(lane, live, P1, Dh, r, hh, skin, MOTION_NONE, true, false);
        rayReduce(lane);
        float across = min(gsRayT[0], 1.0);
        GROUP_SYNC();
        float3 P2 = P1 + Dh * across;
        castCandidates(lane, live, P2, -CHAR_UP * (up + skin), r, hh, skin, MOTION_NONE, true, false);
        rayReduce(lane);
        float downF = gsRayT[0];
        uint downK = gsRayC[0];
        float3 P3 = P2 - CHAR_UP * ((up + skin) * min(downF, 1.0));
        bool landed = downF < 1.0;
        bool walkable = false;
        if (landed && downK != NO_INDEX)
        {
            float t; float3 nrm;
            Collider cap = capsuleShape(r, hh);
            if (shapeCastCollider(cap, P2, float4(0, 0, 0, 1), -CHAR_UP, up + 2.0 * skin, gsCand[downK], t, nrm))
                walkable = castSurfaceNormal(gsCand[downK], r, hh, P2, -CHAR_UP, t, nrm).y >= cosSlope;
        }
        float3 Xs = P3 - P;
        if (landed && walkable && length(float3(Xs.x, 0.0, Xs.z)) > length(Xh) + 1e-4 && Xs.y > 1e-4)
        {
            X = Xs;
            stepped = Xs.y;
            outFlags |= CO_STEPPED_UP;
        }
        GROUP_SYNC();
    }
    // 7. Stick to the floor: walking off a step or down a slope keeps contact.
    bool tryStick = live && wasGround && !jump && mv.y > 0.0 && X.y <= 1e-4;
    if (tryStick)
    {
        float3 S = P + X;
        castCandidates(lane, live, S, -CHAR_UP * (mv.y + skin), r, hh, skin, MOTION_AFTER, true, false);
        rayReduce(lane);
        float f = gsRayT[0];
        uint kk = gsRayC[0];
        if (f < 1.0 && kk != NO_INDEX && lane == 0u)
        {
            float t; float3 nrm;
            Collider cap = capsuleShape(r, hh);
            if (shapeCastCollider(cap, S - gsMove[kk].xyz, float4(0, 0, 0, 1), -CHAR_UP, mv.y + 2.0 * skin, gsCand[kk], t, nrm) &&
                castSurfaceNormal(gsCand[kk], r, hh, S - gsMove[kk].xyz, -CHAR_UP, t, nrm).y >= cosSlope)
                gsShare[1] = float4(-CHAR_UP * ((mv.y + skin) * f), 1.0);
            else
                gsShare[1] = float4(0, 0, 0, 0);
        }
        else if (lane == 0u)
            gsShare[1] = float4(0, 0, 0, 0);
        GROUP_SYNC();
        if (gsShare[1].w > 0.5)
        {
            X += gsShare[1].xyz;
            outFlags |= CO_STUCK_TO_FLOOR;
        }
    }
    // 8. Ground probe at the new pose (obstacles where the step leaves them).
    float3 Pn = P + X;
    float probe = skin + 0.03;
    castCandidates(lane, live, Pn, -CHAR_UP * probe, r, hh, 0.0, MOTION_AFTER, false, true);
    rayReduce(lane);
    float gF = gsRayT[0];
    uint gK = gsRayC[0];
    if (!live || lane != 0u) return;

    uint newState = GS_IN_AIR;
    float3 gN = CHAR_UP;
    uint gBody = NO_INDEX;
    uint gCol = NO_INDEX;
    float3 gPoint = Pn - CHAR_UP * (hh + r);
    if (gF <= 1.0 && gK != NO_INDEX)
    {
        float t; float3 nrm;
        Collider cap = capsuleShape(r, hh);
        if (shapeCastCollider(cap, Pn - gsMove[gK].xyz, float4(0, 0, 0, 1), -CHAR_UP, probe + skin, gsCand[gK], t, nrm))
        {
            // Walkability from the face under the contact, not an edge's
            // tilted contact normal (stairs: no steep-ground flicker).
            float3 contactN = nrm;
            nrm = castSurfaceNormal(gsCand[gK], r, hh, Pn - gsMove[gK].xyz, -CHAR_UP, t, nrm);
            gN = nrm;
            gCol = gsCand[gK];
            gBody = gColliders[gCol].body;
            newState = nrm.y >= cosSlope ? GS_ON_GROUND : GS_ON_STEEP;
            gPoint = Pn - CHAR_UP * (hh + r + max(t, 0.0));
        }
    }
    // Overlapping planes from the ground's side count as ground too (a body
    // resting exactly at the skin, or pressed into it).
    if (newState == GS_IN_AIR && !jump)
    {
        for (uint k = 0u; k < nCand; k++)
        {
            uint info = asuint(gsMove[k].w);
            float4 pl = gsPlane[k];
            float gapNow = pl.w + dot(pl.xyz, X - gsMove[k].xyz);
            float3 sn = gsSurf[k].xyz;
            // Only surfaces that face up support the character.
            if ((info & PI_VALID) == 0u || gapNow > probe || pl.y <= 0.1 || sn.y <= 0.1) continue;
            if (sn.y > gN.y || newState == GS_IN_AIR)
            {
                gN = sn;
                gCol = gsCand[k];
                gBody = gColliders[gCol].body;
                newState = sn.y >= cosSlope ? GS_ON_GROUND : GS_ON_STEEP;
            }
        }
    }
    // Crushed: a dynamic body presses down into a grounded character.
    for (uint k = 0u; k < nCand; k++)
    {
        uint info = asuint(gsMove[k].w);
        if ((info & (PI_VALID | PI_DYNAMIC)) == (PI_VALID | PI_DYNAMIC) && gsPlane[k].y < -0.5 &&
            gsPlane[k].w < -0.25 * r && newState == GS_ON_GROUND)
            outFlags |= CO_CRUSHED;
    }
    // Velocity for the next step: the move made (walls and ceilings stop it).
    float3 Vn = X / max(dt, 1e-8);
    if (newState == GS_ON_GROUND) Vn.y = min(Vn.y, max(gvel.y, 0.0) + max(V.y, 0.0));
    uint4 out = CU(cr + C_OUT);
    if (wasGround && newState == GS_IN_AIR && !jump)
    {
        emitEvent(EVENT_CHAR_SUPPORT_LOST, c, asuint(gnd.w), 0u);
        out.x++; out.y = EVENT_CHAR_SUPPORT_LOST;
    }
    if (!wasGround && newState == GS_ON_GROUND)
    {
        emitEvent(EVENT_CHAR_LANDED, c, gBody, asuint(length(vel.xyz)));
        out.x++; out.y = EVENT_CHAR_LANDED;
    }
    if ((outFlags & CO_CRUSHED) != 0u)
    {
        emitEvent(EVENT_CHAR_CRUSHED, c, gBody, 0u);
        out.x++; out.y = EVENT_CHAR_CRUSHED;
    }
    out.z = nCand;
    out.w = outFlags;
    CSetU(cr + C_OUT, out);
    CSet(cr + C_POS, float4(Pn, asfloat(jump ? jumpSeq : asuint(pos.w))));
    CSet(cr + C_VEL, float4(Vn, asfloat(newState)));
    CSet(cr + C_GND, float4(gN, asfloat(gBody)));
    CSet(cr + C_GPT, float4(gPoint, asfloat(gCol)));
    CSet(cr + C_DBG, float4(castF, stepped, float(blocking), 0.0));
    // The inner capsule follows: a kinematic body moved by v·h at predict.
    uint inner = cfg.x;
    if (inner < gCounters[0].bodyCount)
    {
        BodyState ib = gBodies[inner];
        gBodies[inner].linearVel = (Pn - ib.position) / max(dt, 1e-8);
        gBodies[inner].angularVel = float3(0, 0, 0);
    }
}

// ---------------------------------------------------------------------------
// char_post: one workgroup; lane 0 walks the characters in order, so weights
// on a shared support add up in the same order every run.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void char_post(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint n = charCount();
    float dt = PARAMS.dt;
    for (uint c = 0u; c < n; c++)
    {
        uint cr = charRec(c);
        uint4 cfg = CU(cr + C_CFG);
        if ((cfg.y & CF_ENABLED) == 0u) continue;
        float4 vel = CD(cr + C_VEL);
        float4 gnd = CD(cr + C_GND);
        float4 gpt = CD(cr + C_GPT);
        float4 gv = CD(cr + C_GVEL);
        uint state = asuint(vel.w);
        uint gb = asuint(gnd.w);
        float3 groundVel = float3(0, 0, 0);
        float3 push = float3(0, 0, 0);
        if (state != GS_IN_AIR && gb < gCounters[0].bodyCount)
        {
            BodyState b = gBodies[gb];
            if (bodyMovingOrKinematic(b)) groundVel = b.linearVel + cross(b.angularVel, gpt.xyz - b.position);
            // The character's weight on a dynamic support, as the vehicles hand
            // their reaction back (vehicle_post): a velocity change for the next
            // step, at the contact point.
            if ((cfg.y & CF_PUSH_WEIGHT) != 0u && bodyIsDynamic(b) && state == GS_ON_GROUND)
            {
                float4 mass = CD(cr + C_MASS);
                float3 F = float3(0, PARAMS.gravityY * mass.w * mass.x, 0);
                float3 J = F * dt;
                float3 torque = cross(gpt.xyz - b.position, J);
                float3 local = quatRotate(quatConjugate(b.orientation), torque);
                gBodies[gb].linearVel = b.linearVel + J * b.invMass;
                gBodies[gb].angularVel = b.angularVel + quatRotate(b.orientation, local * b.invInertia);
                if (bodyIsSleeping(b))
                {
                    gBodies[gb].flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
                    gBodies[gb].sleepTimer = 0.0;
                }
                push = F;
            }
        }
        float air = state == GS_IN_AIR ? gv.w + dt : 0.0;
        CSet(cr + C_GVEL, float4(groundVel, air));
        CSet(cr + C_PUSH, float4(push, CD(cr + C_PUSH).w));
    }
}

// ---------------------------------------------------------------------------
// query_batch: in-step scene queries, one workgroup each (results are read
// back asynchronously by the host after the step).
// ---------------------------------------------------------------------------
struct CapsuleVisitor : IColliderVisitor
{
    float3 o; float3 d; float maxT; float r; float hh;
    float3 mn; float3 mx;
    QueryFilter filter;
    float bestT; uint bestC;
    [mutating] void visit(uint ci)
    {
        Collider col = gColliders[ci];
        if (!queryAccepts(col, filter)) return;
        if (col.shape != SHAPE_PLANE && !segmentHitsAabb(mn, mx, gAabbs[HOST_S(1) + ci])) return;
        float t;
        float3 n;
        if (shapeCastCollider(capsuleShape(r, hh), o, float4(0, 0, 0, 1), d, maxT, ci, t, n) && t <= maxT && t < bestT)
        {
            bestT = t;
            bestC = ci;
        }
    }
};

[shader("compute")]
[numthreads(64, 1, 1)]
void query_batch(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint q = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    bool live = q < queryCount();
    uint qr = live ? queryRec(q) : 0u;
    float4 a = live ? CD(qr + Q_IN0) : float4(0, 0, 0, 0);
    float4 b = live ? CD(qr + Q_IN1) : float4(0, -1, 0, 0);
    uint4 k = live ? CU(qr + Q_IN2) : uint4(NO_INDEX, 0, 0, 0);
    float3 dir = length(b.xyz) > 1e-12 ? normalize(b.xyz) : float3(0, -1, 0);
    QueryFilter f;
    f.excludeBody = k.x;
    f.mask = k.y;
    float hh = asfloat(k.w);
    // Kind: bit 0 capsule cast; bit 8 walks every collider (the oracle the
    // grid walk is checked against).
    bool capsule = (k.z & 1u) != 0u;
    bool full = charFullScan() || (k.z & 256u) != 0u;
    if (capsule)
    {
        CapsuleVisitor v;
        v.o = a.xyz; v.d = dir; v.maxT = a.w; v.r = b.w; v.hh = hh;
        float3 e = a.xyz + dir * a.w;
        v.mn = min(a.xyz, e) - (b.w + hh);
        v.mx = max(a.xyz, e) + (b.w + hh);
        v.filter = f;
        v.bestT = 1e30;
        v.bestC = NO_INDEX;
        sceneWalk(lane, live, a.xyz, e, b.w + hh, full, v);
        gsRayT[lane] = v.bestT;
        gsRayC[lane] = v.bestC;
    }
    else
        rayLaneWalk(lane, live, a.xyz, dir, a.w, b.w, -2.0, 0.0, f, full);
    rayReduce(lane);
    if (!live || lane != 0u) return;
    uint best = gsRayC[0];
    if (best == NO_INDEX)
    {
        CSet(qr + Q_OUT0, float4(0, 0, 0, -1.0));
        CSet(qr + Q_OUT1, float4(0, 0, 0, asfloat(NO_INDEX)));
        return;
    }
    float t;
    float3 n;
    if (capsule) shapeCastCollider(capsuleShape(b.w, hh), a.xyz, float4(0, 0, 0, 1), dir, a.w, best, t, n);
    else rayCollider(best, a.xyz, dir, b.w, t, n);
    float tHit = max(gsRayT[0], 0.0);
    if (gsRayT[0] < 0.0) n = -dir;
    if (dot(n, dir) > 0.0) n = -n;
    // Ray: the surface point. Casts: the shape centre at the hit, less radius along n.
    float3 p = a.xyz + dir * tHit - n * b.w;
    CSet(qr + Q_OUT0, float4(p, tHit));
    CSet(qr + Q_OUT1, float4(n, asfloat(best)));
}

// ---------------------------------------------------------------------------
// projectile_step: one workgroup per projectile (CCD audit row 5). A ballistic
// arc in PROJ_SEGMENTS segments per step; each segment casts (ray, or sphere
// of the projectile's radius) through the grid walk, earliest hit first. A hit
// costs energy per metre of collider crossed (its thickness along the path);
// with energy left the cast continues from the exit point, otherwise the
// projectile stops there. Hits push dynamic bodies (impulse records, applied
// by apply_impulses right after) and are reported on the event ring.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void projectile_step(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint p = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    bool live = p < projCount();
    uint pr = live ? projRec(p) : 0u;
    uint4 cfg = live ? CU(pr + PJ_CFG) : uint4(0, 0, 0, 0);
    live = live && (cfg.x & PF_ALIVE) != 0u;
    float4 ps = live ? CD(pr + PJ_POS) : float4(0, 0, 0, 0);
    float4 pv = live ? CD(pr + PJ_VEL) : float4(0, 0, 0, 1);
    float4 pen = live ? CD(pr + PJ_PEN) : float4(0, 0, 0, 0);
    float rad = ps.w;
    float m = max(pv.w, 1e-6);
    uint maxHits = pen.w > 0.0 ? min(uint(pen.w), PROJ_MAX_HITS) : PROJ_MAX_HITS;
    float3 x = ps.xyz;
    float3 v = pv.xyz;
    float3 g = float3(0, PARAMS.gravityY, 0);
    float h = PARAMS.dt / float(PROJ_SEGMENTS);
    uint hits = 0u;
    uint lastHit = NO_INDEX;
    float3 lastN = float3(0, 0, 0);
    float3 lastP = x;
    bool alive = live;
    QueryFilter f;
    f.excludeBody = cfg.z;
    f.mask = cfg.y;
    for (uint s = 0u; s < PROJ_SEGMENTS; s++)
    {
        float3 x1 = x + v * h + g * (0.5 * h * h);
        float3 v1 = (v + g * h) * exp(-pen.z * h);
        float3 seg = x1 - x;
        float len = length(seg);
        bool segLive = alive && len > 1e-9;
        float3 dir = segLive ? seg / len : float3(0, -1, 0);
        float3 from = x;
        float travelled = 0.0;
        // Hits along this segment, nearest first.
        for (uint it = 0u; it < PROJ_MAX_HITS; it++)
        {
            bool castLive = segLive && hits < maxHits && travelled < len;
            QueryFilter fk = f;
            rayLaneWalk(lane, castLive, from, dir, len - travelled, rad, -2.0, 0.0, fk, charFullScan(), 0.0);
            rayReduce(lane);
            float t = gsRayT[0];
            uint ci = gsRayC[0];
            bool hit = castLive && ci != NO_INDEX && t <= len - travelled;
            float thick = 0.0;
            float3 n = -dir;
            float3 hp = from;
            if (hit)
            {
                t = max(t, 0.0);
                rayCollider(ci, from, dir, rad, t, n);
                t = max(t, 0.0);
                if (dot(n, dir) > 0.0) n = -n;
                hp = from + dir * t;
                // Thickness along the path: cast back from beyond the collider.
                Collider col = gColliders[ci];
                float bound = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
                float span = 2.0 * bound + 2.0 * rad + 1e-3;
                float tb;
                float3 nb;
                if (col.shape != SHAPE_PLANE && rayCollider(ci, hp + dir * span, -dir, rad, tb, nb))
                    thick = max(span - tb, 0.0);
                else
                    thick = 1e30;
            }
            GROUP_SYNC();
            if (!hit) break;
            // Energy to cross the collider (its thickness along the path); the
            // rest carries on beyond it.
            float cost = pen.y * thick;
            Collider col = gColliders[ci];
            BodyState hb = gBodies[col.body];
            float speed = length(v1);
            float energy = 0.5 * m * speed * speed;
            float3 vBefore = dir * speed;
            bool through = energy > cost && thick < 1e29;
            float3 vAfter = through ? dir * sqrt(max(2.0 * (energy - cost) / m, 0.0)) : float3(0, 0, 0);
            if (lane == 0u)
            {
                emitEvent(EVENT_PROJECTILE_HIT, p, ci, asuint(through ? thick : -1.0));
                if ((cfg.x & PF_IMPULSES) != 0u && bodyIsDynamic(hb))
                {
                    uint slot;
                    AADD3(gScratchAtomics[IMPULSE_COUNT], 1u, slot);
                    if (slot < IMPULSE_CAP)
                    {
                        uint rec = IMPULSE_BASE + IMPULSE_WORDS * slot;
                        float3 J = (vBefore - vAfter) * m;
                        // Key: after every host record, by projectile then hit.
                        uint key = 0x100000u + p * 16u + hits;
                        ASTORE(gScratchAtomics[rec], col.body);
                        ASTORE(gScratchAtomics[rec + 1u], (key << 8) | IMPULSE_AT_POINT);
                        ASTORE(gScratchAtomics[rec + 2u], asuint(J.x));
                        ASTORE(gScratchAtomics[rec + 3u], asuint(J.y));
                        ASTORE(gScratchAtomics[rec + 4u], asuint(J.z));
                        ASTORE(gScratchAtomics[rec + 5u], asuint(hp.x));
                        ASTORE(gScratchAtomics[rec + 6u], asuint(hp.y));
                        ASTORE(gScratchAtomics[rec + 7u], asuint(hp.z));
                    }
                }
            }
            hits++;
            lastHit = ci;
            lastN = n;
            lastP = hp;
            if (!through)
            {
                alive = false;
                x1 = hp;
                v1 = float3(0, 0, 0);
                if (lane == 0u) emitEvent(EVENT_PROJECTILE_STOPPED, p, ci, 0u);
                break;
            }
            v1 = normalize(v1) * length(vAfter);
            float adv = t + thick + 1e-4;
            travelled += adv;
            from += dir * adv;
        }
        // A hit's exit can lie past the segment's end: carry on from there.
        x = travelled > len ? from : x1;
        v = v1;
        if (!alive) break;
    }
    if (!live || lane != 0u) return;
    float energy = alive ? 0.5 * m * dot(v, v) : 0.0;
    CSet(pr + PJ_POS, float4(x, rad));
    CSet(pr + PJ_VEL, float4(v, pv.w));
    CSetU(pr + PJ_CFG, uint4(alive ? cfg.x : (cfg.x & ~PF_ALIVE), cfg.y, cfg.z, cfg.w + hits));
    CSet(pr + PJ_PEN, float4(energy, pen.yzw));
    if (hits > 0u)
    {
        CSet(pr + PJ_OUT, float4(lastP, asfloat(lastHit)));
        CSet(pr + PJ_OUT2, float4(lastN, asfloat(hits)));
    }
    else
        CSet(pr + PJ_OUT2, float4(CD(pr + PJ_OUT2).xyz, asfloat(0u)));
}
