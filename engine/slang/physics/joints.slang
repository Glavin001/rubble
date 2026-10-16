#ifndef ENGINE_PHYSICS_JOINTS_SLANG
#define ENGINE_PHYSICS_JOINTS_SLANG

// D6 joint rows for AVBD (three-avbd joints + PhysX D6 motions, limits, drives).
//
// A joint has seven scalar row slots: 0..2 linear (frame-A axes, world axes when
// all locked, or one radial row), 3 twist, 4 swing1 / cone, 5 swing2, 6 drive.
// Each row is re-evaluated at the current pose every primal / dual iteration
// (AVBD keeps C(x) nonlinear for joints). Row coordinates are raw (m or rad);
// hard angular rows are scaled by the torque arm so their penalties share the
// linear range (three-avbd), and soft rows use the user's physical stiffness.
//
// Sign convention: the coordinate's gradient w.r.t. body A is (lin, angA) and
// w.r.t. body B is (-lin, angB). Forces f = dE/dC; rhs += J·f, A += h·JJᵀ.

static const uint JROW_NONE = 0u;
static const uint JROW_EQ = 1u;     // v == lo
static const uint JROW_LIMIT = 2u;  // lo <= v <= hi, one-sided on the nearer bound
static const uint JROW_MOTOR = 3u;  // hard: dv/dt == vel, |f| <= fmax (three-avbd motor)
static const uint JROW_SERVO = 4u;  // soft: k (v - target) + d (dv/dt - vel), |f| <= fmax

static const float JOINT_TWO_PI = 6.28318530718;

struct JointCtx
{
    float3 pA, pB; // world anchors
    float3 rA, rB; // lever arms from each body's center
    float3 xA;     // body A center
    float3 aX, aY, aZ, bX;
    float4 rel;    // conj(cA)·cB, w >= 0 (neighbourhood operator)
    float3 eLock;  // 2·vec(cA·conj(cB)), w >= 0: small-angle world error (three-avbd)
};

struct JointRow
{
    uint mode;
    float v;
    float lo;
    float hi;
    float vel;
    float scale;   // torque arm for hard angular rows, 1 otherwise
    float kSoft;   // 0 = hard (λ + ramped penalty); > 0 = spring stiffness
    float damp;
    float fmax;
    bool angular;
    bool wrap;
    float3 lin;
    float3 angA;
    float3 angB;
};

uint jointMotion(Joint j, uint dof) { return (j.motion >> (2u * dof)) & 3u; }
bool jointLinAllLocked(Joint j) { return (j.motion & 0x3Fu) == 0u && (j.flags & JOINT_FLAG_RADIAL) == 0u; }
bool jointAngAllLocked(Joint j) { return ((j.motion >> 6u) & 0x3Fu) == 0u; }
bool jointBroken(Joint j) { return (j.flags & JOINT_FLAG_BROKEN) != 0u; }

float jointWrap(float a) { return a - JOINT_TWO_PI * round(a / JOINT_TWO_PI); }

JointCtx jointContext(Joint j, BodyState a, BodyState b)
{
    JointCtx c;
    float4 cA = quatMul(a.orientation, j.frameA);
    float4 cB = quatMul(b.orientation, j.frameB);
    c.rA = quatRotate(a.orientation, j.anchorA);
    c.rB = quatRotate(b.orientation, j.anchorB);
    c.xA = a.position;
    c.pA = a.position + c.rA;
    c.pB = b.position + c.rB;
    c.aX = quatRotate(cA, float3(1.0, 0.0, 0.0));
    c.aY = quatRotate(cA, float3(0.0, 1.0, 0.0));
    c.aZ = quatRotate(cA, float3(0.0, 0.0, 1.0));
    c.bX = quatRotate(cB, float3(1.0, 0.0, 0.0));
    float4 rel = quatMul(quatConjugate(cA), cB);
    c.rel = rel.w < 0.0 ? -rel : rel;
    float4 d = quatMul(cA, quatConjugate(cB));
    c.eLock = 2.0 * (d.w < 0.0 ? -d.xyz : d.xyz);
    return c;
}

JointRow jointRowNone()
{
    JointRow r;
    r.mode = JROW_NONE;
    r.v = 0.0; r.lo = 0.0; r.hi = 0.0; r.vel = 0.0;
    r.scale = 1.0; r.kSoft = 0.0; r.damp = 0.0; r.fmax = JOINT_BIG;
    r.angular = false; r.wrap = false;
    r.lin = float3(0.0); r.angA = float3(0.0); r.angB = float3(0.0);
    return r;
}

float3 jointUnit(uint i) { return float3(i == 0u ? 1.0 : 0.0, i == 1u ? 1.0 : 0.0, i == 2u ? 1.0 : 0.0); }

// Translation of anchor B along frame-A axis `ax`. The exact gradient puts the
// lever arm for body A at anchor B (PhysX ConstraintHelper): rotating A also
// swings the axis, and (rA + pB - pA) = pB - xA.
void jointLinearAxis(JointCtx c, float3 ax, inout JointRow r)
{
    r.v = dot(c.pB - c.pA, ax);
    r.lin = -ax;
    r.angA = -cross(c.pB - c.xA, ax);
    r.angB = cross(c.rB, ax);
}

// Twist of frame B about frame-A X (PhysX computePhi on the swing-twist split).
// Exact gradient: with q = conj(cA)·cB, dq = ½(u,0)⊗q for u = R_Aᵀ(δθB − δθA),
// so dφ = g·u with g = (qw²+qx², qw·qz+qx·qy, qx·qz−qw·qy)/(qw²+qx²). Using bX
// instead (exact only at zero swing) pushes the wrong way under large swing.
// qw² + qx² = cos²(swing / 2): twist is undefined as the swing nears π, so
// the row drops out past ~150° of swing (|g| stays below ~4).
static const float JOINT_TWIST_MIN_DEN = 0.067;

bool jointTwistDefined(JointCtx c) { return c.rel.w * c.rel.w + c.rel.x * c.rel.x > JOINT_TWIST_MIN_DEN; }

void jointTwist(JointCtx c, inout JointRow r)
{
    float4 q = c.rel;
    float den = max(q.w * q.w + q.x * q.x, JOINT_TWIST_MIN_DEN);
    float3 g = float3(q.w * q.w + q.x * q.x, q.w * q.z + q.x * q.y, q.x * q.z - q.w * q.y) / den;
    float3 t = g.x * c.aX + g.y * c.aY + g.z * c.aZ;
    r.v = 2.0 * atan2(q.x, q.w);
    r.angA = -t;
    r.angB = t;
    r.angular = true;
    r.wrap = true;
}

// Single-axis swing angles of frame-B X, measured in frame A with the exact
// atan2 gradient. Twist leaves bX unchanged, so these never fight the twist row.
void jointSwing(JointCtx c, uint which, inout JointRow r)
{
    float w = dot(c.bX, c.aX);
    float u = which == 1u ? -dot(c.bX, c.aZ) : dot(c.bX, c.aY);
    float3 perp = which == 1u ? (-w * c.aZ - u * c.aX) : (w * c.aY - u * c.aX);
    float den = max(u * u + w * w, 1e-12);
    float3 g = cross(c.bX, perp) / den;
    r.v = atan2(u, w);
    r.angA = -g;
    r.angB = g;
    r.angular = true;
    r.wrap = true;
}

JointRow jointRow(Joint j, JointCtx c, uint slot)
{
    JointRow r = jointRowNone();
    float hardLin = j.linStiffness >= JOINT_HARD ? 0.0 : j.linStiffness;
    float hardAng = j.angStiffness >= JOINT_HARD ? 0.0 : j.angStiffness;
    if (slot < 3u)
    {
        if ((j.flags & JOINT_FLAG_RADIAL) != 0u)
        {
            if (slot != 0u) return r;
            float3 d = c.pA - c.pB;
            float len = length(d);
            float3 n = len > 1e-6 ? d / len : c.aX;
            r.v = len;
            r.lin = n;
            r.angA = cross(c.rA, n);
            r.angB = -cross(c.rB, n);
            r.lo = j.radialMin > 0.0 ? j.radialMin : -JOINT_BIG;
            r.hi = j.radialMax;
            r.mode = j.radialMin == j.radialMax ? JROW_EQ : JROW_LIMIT;
            r.kSoft = hardLin;
            return r;
        }
        if (jointLinAllLocked(j))
        {
            float3 n = jointUnit(slot);
            r.v = (c.pA - c.pB)[slot];
            r.lin = n;
            r.angA = cross(c.rA, n);
            r.angB = -cross(c.rB, n);
            r.mode = JROW_EQ;
            r.kSoft = hardLin;
            return r;
        }
        uint m = jointMotion(j, slot);
        if (m == MOTION_FREE) return r;
        jointLinearAxis(c, slot == 0u ? c.aX : slot == 1u ? c.aY : c.aZ, r);
        if (m == MOTION_LOCKED)
        {
            r.mode = JROW_EQ;
            r.kSoft = hardLin;
        }
        else
        {
            r.mode = JROW_LIMIT;
            r.lo = j.linLo[slot];
            r.hi = j.linHi[slot];
            r.kSoft = j.limitStiffness;
            r.damp = j.limitDamping;
        }
        return r;
    }
    if (slot < 6u)
    {
        if (jointAngAllLocked(j))
        {
            float3 n = jointUnit(slot - 3u);
            r.v = c.eLock[slot - 3u];
            r.angA = n;
            r.angB = -n;
            r.mode = JROW_EQ;
            r.angular = true;
            r.kSoft = hardAng;
            r.scale = hardAng > 0.0 ? 1.0 : j.torqueArm;
            return r;
        }
        uint m;
        if (slot == 3u)
        {
            m = jointMotion(j, 3u);
            if (m == MOTION_FREE || !jointTwistDefined(c)) return r;
            jointTwist(c, r);
            r.lo = m == MOTION_LOCKED ? 0.0 : j.twistLo;
            r.hi = m == MOTION_LOCKED ? 0.0 : j.twistHi;
        }
        else
        {
            uint s1 = jointMotion(j, 4u);
            uint s2 = jointMotion(j, 5u);
            bool cone = (j.flags & JOINT_FLAG_CONE) != 0u && s1 != MOTION_LOCKED && s2 != MOTION_LOCKED
                && (s1 == MOTION_LIMITED || s2 == MOTION_LIMITED);
            if (cone)
            {
                if (slot != 4u) return r;
                // Elliptical cone on the angle between the X axes. The local
                // semi-axis interpolates the Y / Z limits by swing direction.
                float3 nx = cross(c.aX, c.bX);
                float s = length(nx);
                if (s < 1e-6) return r; // aligned (inside) or antipodal (no direction)
                float u = dot(c.bX, c.aY); // toward Y: swing about Z
                float w = dot(c.bX, c.aZ); // toward Z: swing about Y
                float ulen = max(sqrt(u * u + w * w), 1e-9);
                float zLim = s2 == MOTION_LIMITED ? max(j.swingZ, 1e-4) : 3.14159265;
                float yLim = s1 == MOTION_LIMITED ? max(j.swingY, 1e-4) : 3.14159265;
                float cu = (u / ulen) / zLim;
                float cv = (w / ulen) / yLim;
                float limit = 1.0 / sqrt(cu * cu + cv * cv);
                float3 n = nx / s;
                r.v = acos(clamp(dot(c.aX, c.bX), -1.0, 1.0));
                r.angA = -n;
                r.angB = n;
                r.angular = true;
                r.lo = -JOINT_BIG;
                r.hi = limit;
                m = MOTION_LIMITED;
            }
            else
            {
                m = slot == 4u ? s1 : s2;
                if (m == MOTION_FREE) return r;
                jointSwing(c, slot == 4u ? 1u : 2u, r);
                float lim = slot == 4u ? j.swingY : j.swingZ;
                r.lo = m == MOTION_LOCKED ? 0.0 : -lim;
                r.hi = m == MOTION_LOCKED ? 0.0 : lim;
            }
        }
        if (m == MOTION_LOCKED)
        {
            r.mode = JROW_EQ;
            r.kSoft = hardAng;
        }
        else
        {
            r.mode = JROW_LIMIT;
            r.kSoft = j.limitStiffness;
            r.damp = j.limitDamping;
        }
        r.scale = r.kSoft > 0.0 ? 1.0 : j.torqueArm;
        return r;
    }
    // Slot 6: drive on one linear axis (frame A) or the twist.
    if ((j.flags & JOINT_FLAG_DRIVE) == 0u) return r;
    uint dof = (j.flags >> JOINT_DRIVE_DOF_SHIFT) & 7u;
    if (dof < 3u)
        jointLinearAxis(c, dof == 0u ? c.aX : dof == 1u ? c.aY : c.aZ, r);
    else if (jointTwistDefined(c))
        jointTwist(c, r);
    else
        return r;
    r.vel = j.driveVelocity;
    r.fmax = j.driveForceLimit;
    if (j.driveStiffness > 0.0)
    {
        r.mode = JROW_SERVO;
        r.lo = j.driveTarget;
        r.kSoft = j.driveStiffness;
        r.damp = j.driveDamping;
    }
    else
    {
        r.mode = JROW_MOTOR;
        r.scale = r.angular ? j.torqueArm : 1.0;
    }
    return r;
}

// Row force f (already signed for the row's Jacobian) and Hessian weight h.
// `c0` is the row coordinate at step start; `creg` returns the stabilised
// residual the dual ramps on.
void jointRowForce(JointRow r, float c0, float lam, float pen, float alpha, float dt,
                   out float f, out float h, out float creg)
{
    f = 0.0;
    h = 0.0;
    creg = 0.0;
    float s = r.scale;
    if (r.mode == JROW_EQ)
    {
        float e = r.v - r.lo;
        float e0 = c0 - r.lo;
        if (r.wrap) { e = jointWrap(e); e0 = jointWrap(e0); }
        if (r.kSoft > 0.0)
        {
            f = r.kSoft * e * s;
            h = r.kSoft;
            creg = e * s;
        }
        else
        {
            // three-avbd Eq. 18: hard rows solve C - α·C0.
            creg = (e - alpha * e0) * s;
            f = pen * creg + lam;
            h = pen;
        }
    }
    else if (r.mode == JROW_LIMIT)
    {
        bool lower;
        float C, C0;
        if (r.wrap)
        {
            // Angles: measure from the range midpoint on the circle, so a
            // joint forced past ±π still pushes back toward the near bound.
            float mid = 0.5 * (r.lo + r.hi);
            float half = 0.5 * (r.hi - r.lo);
            float d = jointWrap(r.v - mid);
            float d0 = jointWrap(c0 - mid);
            lower = d < 0.0;
            C = (half - abs(d)) * s;
            C0 = (half - (lower ? -d0 : d0)) * s;
        }
        else
        {
            lower = r.v < 0.5 * (r.lo + r.hi);
            C = (lower ? r.v - r.lo : r.hi - r.v) * s;
            C0 = (lower ? c0 - r.lo : r.hi - c0) * s;
        }
        float fr;
        if (r.kSoft > 0.0)
        {
            fr = r.kSoft * C;
            if (fr < 0.0) fr = min(fr + r.damp * (C - C0) / dt, 0.0);
            h = r.kSoft + r.damp / dt;
            creg = C;
        }
        else
        {
            // Only stabilise an existing violation; a positive C0 is slack.
            creg = C - alpha * min(C0, 0.0);
            fr = min(pen * creg + lam, 0.0);
            h = pen;
        }
        if (fr >= 0.0)
        {
            h = 0.0;
            fr = 0.0;
        }
        f = lower ? fr : -fr;
    }
    else if (r.mode == JROW_MOTOR)
    {
        float dv = r.v - c0;
        if (r.wrap) dv = jointWrap(dv);
        creg = (dv - r.vel * dt) * s;
        float F = r.fmax / s;
        float raw = pen * creg + lam;
        f = clamp(raw, -F, F);
        // A saturated row is a constant force: dropping its stiffness lets the
        // body accelerate at F / I instead of being pinned by the penalty
        // (three-avbd 2D Eq. 14 rescales clamped rows the same way).
        h = abs(raw) > F ? 0.0 : pen;
    }
    else if (r.mode == JROW_SERVO)
    {
        float e = r.v - r.lo;
        float dv = r.v - c0;
        if (r.wrap) { e = jointWrap(e); dv = jointWrap(dv); }
        float raw = r.kSoft * e + r.damp * (dv / dt - r.vel);
        f = clamp(raw, -r.fmax, r.fmax);
        h = abs(raw) > r.fmax ? 0.0 : r.kSoft + r.damp / dt;
        creg = e;
    }
}

// Ball fast path (ropes, chain mail): the fields three world-axis rows need,
// loaded field by field so the solver never pulls the whole 288 B record.
struct JointBall
{
    uint bodyA;
    uint bodyB;
    float3 anchorA;
    float3 anchorB;
    float linStiffness;
    float breakForce;
    float3 c0;
    float3 lambda;
    float3 penalty;
};

void jointContributeBall(inout Mat6 A, inout Vec6 rhs, JointBall j, BodyState a, BodyState b,
                         bool isA, float alpha, float kStart)
{
    float3 rA = quatRotate(a.orientation, j.anchorA);
    float3 rB = quatRotate(b.orientation, j.anchorB);
    float3 C = (a.position + rA) - (b.position + rB);
    float3 r = isA ? rA : -rB;
    float s = isA ? 1.0 : -1.0;
    bool hard = j.linStiffness >= JOINT_HARD;
    float3 F;
    [unroll]
    for (uint axis = 0u; axis < 3u; axis++)
    {
        float3 n = jointUnit(axis);
        float pen = max(j.penalty[axis], kStart);
        float f = hard ? pen * (C[axis] - alpha * j.c0[axis]) + j.lambda[axis] : j.linStiffness * C[axis];
        float h = hard ? pen : j.linStiffness;
        F[axis] = f;
        float3 t = cross(r, n);
        Vec6 jj;
        jj.v[0] = s * n.x; jj.v[1] = s * n.y; jj.v[2] = s * n.z;
        jj.v[3] = t.x; jj.v[4] = t.y; jj.v[5] = t.z;
        mat6AddOuter(A, jj, h);
        vec6AddScaled(rhs, jj, f);
    }
    // three-avbd Sec. 3.5 geometric stiffness, diagonally lumped.
    float rf = dot(r, F);
    float3 gs = float3(length(r * F.x - float3(rf, 0.0, 0.0)),
                       length(r * F.y - float3(0.0, rf, 0.0)),
                       length(r * F.z - float3(0.0, 0.0, rf)));
    mat6AddDiag(A, float3(0.0), gs);
}

float3 jointBallC(JointBall j, BodyState a, BodyState b)
{
    return transformPoint(a.position, a.orientation, j.anchorA) - transformPoint(b.position, b.orientation, j.anchorB);
}

// Ball dual (three-avbd dualJoint, linear rows only). Returns true if it broke.
bool jointDualBall(inout JointBall j, BodyState a, BodyState b, float alpha, float kStart, float kMax, float betaLin)
{
    float3 C = jointBallC(j, a, b);
    if (j.linStiffness < JOINT_HARD)
    {
        j.lambda = j.linStiffness * C;
    }
    else
    {
        float3 creg = C - alpha * j.c0;
        float3 pen = max(j.penalty, float3(kStart));
        j.lambda = pen * creg + j.lambda;
        j.penalty = min(pen + betaLin * abs(creg), float3(kMax));
    }
    return length(j.lambda) > j.breakForce;
}

// Accumulate this joint's rows into body `isA ? A : B`'s 6x6 Newton system.
void jointContributePrimal(inout Mat6 A, inout Vec6 rhs, Joint j, BodyState a, BodyState b,
                           bool isA, float alpha, float dt, float kStart)
{
    JointCtx c = jointContext(j, a, b);
    float3 F = float3(0.0);
    [unroll]
    for (uint slot = 0u; slot < 7u; slot++)
    {
        JointRow r = jointRow(j, c, slot);
        if (r.mode == JROW_NONE) continue;
        float pen = max(j.penalty[slot], kStart);
        float f, h, creg;
        jointRowForce(r, j.c0[slot], j.lambda[slot], pen, alpha, dt, f, h, creg);
        if (slot < 3u) F[slot] = f;
        if (h == 0.0 && f == 0.0) continue;
        Vec6 jj;
        float3 l = (isA ? r.lin : -r.lin) * r.scale;
        float3 g = (isA ? r.angA : r.angB) * r.scale;
        jj.v[0] = l.x; jj.v[1] = l.y; jj.v[2] = l.z;
        jj.v[3] = g.x; jj.v[4] = g.y; jj.v[5] = g.z;
        mat6AddOuter(A, jj, h);
        vec6AddScaled(rhs, jj, f);
    }
    if (jointLinAllLocked(j))
    {
        // three-avbd Sec. 3.5 geometric stiffness, diagonally lumped: column
        // norms of -(r·F)I + rFᵀ keep heavy chains from going soft.
        float3 r = isA ? c.rA : -c.rB;
        float rf = dot(r, F);
        float3 gs = float3(length(r * F.x - float3(rf, 0.0, 0.0)),
                           length(r * F.y - float3(0.0, rf, 0.0)),
                           length(r * F.z - float3(0.0, 0.0, rf)));
        mat6AddDiag(A, float3(0.0), gs);
    }
}

// Step-start bookkeeping (three-avbd Joint::initialize / Eq. 19).
void jointWarmstart(inout Joint j, BodyState a, BodyState b, float alpha, float gamma,
                    float kStart, float kMax)
{
    JointCtx c = jointContext(j, a, b);
    [unroll]
    for (uint slot = 0u; slot < 7u; slot++)
    {
        JointRow r = jointRow(j, c, slot);
        j.c0[slot] = r.v;
        if (r.mode == JROW_NONE)
        {
            j.lambda[slot] = 0.0;
            j.penalty[slot] = kStart;
            continue;
        }
        bool hard = r.kSoft == 0.0;
        j.lambda[slot] = hard ? j.lambda[slot] * alpha * gamma : 0.0;
        j.penalty[slot] = clamp(j.penalty[slot] * gamma, kStart, kMax);
    }
}

// Dual update, then breaking and plastic yield. Returns true if the joint broke.
bool jointDual(inout Joint j, BodyState a, BodyState b, float alpha, float dt,
               float kStart, float kMax, float betaLin, float betaAng)
{
    JointCtx c = jointContext(j, a, b);
    [unroll]
    for (uint slot = 0u; slot < 7u; slot++)
    {
        JointRow r = jointRow(j, c, slot);
        if (r.mode == JROW_NONE) continue;
        float pen = max(j.penalty[slot], kStart);
        float f, h, creg;
        jointRowForce(r, j.c0[slot], j.lambda[slot], pen, alpha, dt, f, h, creg);
        if (r.kSoft > 0.0 || r.mode == JROW_SERVO)
        {
            // Soft rows carry no multiplier; keep the force for readout / breaking.
            j.lambda[slot] = f;
            continue;
        }
        float beta = r.angular ? betaAng : betaLin;
        if (r.mode == JROW_LIMIT)
        {
            float lam = min(pen * creg + j.lambda[slot], 0.0);
            j.lambda[slot] = lam;
            // A slack limit restarts soft, like a new contact: keeping a ramped
            // penalty makes the next touch snap back at full stiffness (pops).
            j.penalty[slot] = lam < 0.0 ? min(pen + beta * abs(creg), kMax) : kStart;
        }
        else if (r.mode == JROW_MOTOR)
        {
            j.lambda[slot] = f;
            if (abs(f) < r.fmax / r.scale) j.penalty[slot] = min(pen + beta * abs(creg), kMax);
        }
        else
        {
            j.lambda[slot] = f;
            j.penalty[slot] = min(pen + beta * abs(creg), kMax);
        }
    }

    float3 lamLin = float3(j.lambda[0], j.lambda[1], j.lambda[2]);
    float3 lamAng = float3(j.lambda[3], j.lambda[4], j.lambda[5]);
    float torque = length(lamAng) * j.torqueArm;
    bool broken = length(lamLin) > j.breakForce || torque > j.breakTorque;
    if (!broken && jointAngAllLocked(j) && j.yieldTorque < JOINT_BIG && torque > j.yieldTorque)
    {
        // three-avbd plastic yield: clamp the angular force to the yield value
        // and let the rest pose follow the bodies; break once bent too far.
        float sc = j.yieldTorque / torque;
        j.lambda[3] *= sc; j.lambda[4] *= sc; j.lambda[5] *= sc;
        j.frameB = quatNormalize(quatMul(quatConjugate(b.orientation), quatMul(a.orientation, j.frameA)));
        j.c0[3] = 0.0; j.c0[4] = 0.0; j.c0[5] = 0.0;
        float4 d = quatMul(quatConjugate(j.restStart), j.frameB);
        float bend = 2.0 * atan2(length(d.xyz), abs(d.w));
        broken = bend > j.breakBend;
    }
    if (broken)
    {
        j.flags |= JOINT_FLAG_BROKEN;
        [unroll]
        for (uint k = 0u; k < 8u; k++) { j.lambda[k] = 0.0; j.penalty[k] = 0.0; }
    }
    return broken;
}

#endif
