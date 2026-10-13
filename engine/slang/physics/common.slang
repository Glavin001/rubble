#ifndef ENGINE_PHYSICS_COMMON_SLANG
#define ENGINE_PHYSICS_COMMON_SLANG

// ==== types ====
struct BodyState
{
    float3 position;
    float invMass;
    float4 orientation;
    float3 invInertia;
    uint flags;
    float3 linearVel;
    float sleepTimer;
    float3 angularVel;
    uint generation;
};

static const uint BODY_DYNAMIC = 1u;
static const uint BODY_KINEMATIC = 2u;
static const uint BODY_SLEEPING = 4u;
static const uint BODY_AWAKE = 8u;

struct Collider
{
    uint body;
    uint shape;
    uint material;
    uint generation;
    float3 localPos;
    float density;
    float4 localRot;
    float4 params;
};

static const uint SHAPE_SPHERE = 0u;
static const uint SHAPE_CAPSULE = 1u;
static const uint SHAPE_BOX = 2u;
static const uint SHAPE_CONVEX = 3u;
static const uint SHAPE_PLANE = 4u;

struct Material
{
    float friction;
    float restitution;
    float contactOffset;
    float _pad;
};

struct Contact
{
    uint colliderA;
    uint colliderB;
    uint featureId;
    uint alive;
    float3 normal;
    float penetration;
    float3 anchorA;
    float c0N;
    float3 anchorB;
    float friction;
    float3 tangent1;
    float lambdaN;
    float3 tangent2;
    float lambdaT1;
    float kN;
    float kT;
    float lambdaT2;
    float c0T1;
    float c0T2;
    float _pad0; // stick flag (1 = static friction held)
    float _pad1;
    float _pad2; // ABI: keep Contact = 128 B (matches engine-schema)
};

// three-avbd Stage 8b: one CSR/dual edge per pair; contacts are a contiguous range.
struct Manifold
{
    uint colliderA;
    uint colliderB;
    uint contactStart;
    uint contactCount;
    float3 normal;
    float friction;
};

struct Joint
{
    uint bodyA;
    uint bodyB;
    uint jointType;
    uint flags;
    float3 localAnchorA;
    float stiffness;
    float3 localAnchorB;
    float damping;
    float3 axisA;
    float limitMin;
    float3 axisB;
    float limitMax;
    float3 c0;
    float _padC0;
    float lambda0;
    float lambda1;
    float lambda2;
    float lambda3;
    float lambda4;
    float lambda5;
    float k0;
    float k1;
    float k2;
    float k3;
    float k4;
    float k5;
};

static const uint JOINT_BALL = 0u;
static const uint JOINT_FIXED = 1u;
static const uint JOINT_HINGE = 2u;
static const uint JOINT_SLIDER = 3u;
static const uint JOINT_D6 = 4u;
static const uint JOINT_DISTANCE = 5u;
static const uint JOINT_MAX_DISTANCE = 6u;

struct AvbdParams
{
    float dt;
    float gravityY;
    uint iterations;
    uint flags; // bit0 = reuseContacts
    float alpha;
    float gamma;
    float betaLin;
    float betaAng;
    float kStart;
    float kMaxN;
    float kMaxT;
    float contactOffset;
    float sleepLin;
    float sleepAng;
    float sleepTime;
    float residualExit;
    float trustLin;
    float trustAng;
    float pivotFloor;
    float gridCellSize; // three-avbd ≈ 2·maxSmallRadius; host sets each step
};

struct StepCounters
{
    uint bodyCount;
    uint colliderCount;
    uint contactCount;
    uint jointCount;
    uint candidateCount;
    uint awakeCount;
    uint colorCount;
    uint overflowFlags;
    uint pivotFloorHits;
    uint sleepEvents;
    uint wakeEvents;
    uint iteration;
    uint manifoldCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

static const uint OVERFLOW_CONTACTS = 1u;
static const uint OVERFLOW_CANDIDATES = 2u;
static const uint OVERFLOW_JOINTS = 4u;
static const uint OVERFLOW_COLORS = 8u;
static const uint CCD_BUDGET_EXHAUSTED = 16u;
static const uint CCD_CANDIDATE_MISS = 32u;
static const uint OVERFLOW_MANIFOLDS = 64u;

struct GpuCommand
{
    uint op;
    uint a;
    uint b;
    uint c;
    float4 f0;
    float4 f1;
    float4 f2;
};

static const uint CMD_NOP = 0u;
static const uint CMD_WAKE_BODY = 3u;
static const uint CMD_TELEPORT = 4u;
static const uint CMD_SET_VELOCITY = 5u;
static const uint CMD_REPARENT = 8u;

struct Aabb
{
    float3 mn;
    float _pad0;
    float3 mx;
    float _pad1;
};

struct CandidatePair
{
    uint colliderA;
    uint colliderB;
    uint key;
    uint _pad;
};

// ==== math ====
float4 quatIdentity() { return float4(0.0, 0.0, 0.0, 1.0); }

float4 quatNormalize(float4 q)
{
    float n = length(q);
    return n > 1e-12 ? q / n : quatIdentity();
}

float4 quatMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float4 quatConjugate(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }

float3 quatRotate(float4 q, float3 v)
{
    float3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

float4 quatApplyDelta(float4 q, float3 dTheta)
{
    float angle = length(dTheta);
    float4 dq;
    if (angle < 1e-8)
        dq = float4(0.5 * dTheta, 1.0);
    else
    {
        float half = 0.5 * angle;
        float s = sin(half) / angle;
        dq = float4(dTheta * s, cos(half));
    }
    return quatNormalize(quatMul(dq, q));
}

float3 transformPoint(float3 pos, float4 rot, float3 local)
{
    return pos + quatRotate(rot, local);
}

void orthonormalBasis(float3 n, out float3 t1, out float3 t2)
{
    float3 a = abs(n.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
    t1 = normalize(cross(n, a));
    t2 = cross(n, t1);
}

// ==== 6x6 LDLT ====
struct Mat6 { float m[36]; };
struct Vec6 { float v[6]; };

Mat6 mat6Zero()
{
    Mat6 a;
    for (int i = 0; i < 36; i++) a.m[i] = 0.0;
    return a;
}

Vec6 vec6Zero()
{
    Vec6 a;
    for (int i = 0; i < 6; i++) a.v[i] = 0.0;
    return a;
}

void mat6AddOuter(inout Mat6 a, Vec6 j, float k)
{
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            a.m[r * 6 + c] += k * j.v[r] * j.v[c];
}

void mat6AddDiag(inout Mat6 a, float3 lin, float3 ang)
{
    a.m[0] += lin.x; a.m[7] += lin.y; a.m[14] += lin.z;
    a.m[21] += ang.x; a.m[28] += ang.y; a.m[35] += ang.z;
}

void vec6AddScaled(inout Vec6 a, Vec6 j, float s)
{
    for (int i = 0; i < 6; i++) a.v[i] += j.v[i] * s;
}

Vec6 solve6x6(Mat6 A, Vec6 b, float pivotFloor, inout uint floorHits)
{
    float a[36];
    for (int i = 0; i < 36; i++) a[i] = A.m[i];
    float scale[6];
    for (int i = 0; i < 6; i++)
    {
        float d = a[i * 6 + i];
        scale[i] = d > 1e-12 ? rsqrt(d) : 1.0;
    }
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            a[r * 6 + c] *= scale[r] * scale[c];

    float L[36];
    float D[6];
    for (int i = 0; i < 36; i++) L[i] = 0.0;
    for (int i = 0; i < 6; i++)
    {
        L[i * 6 + i] = 1.0;
        float diag = a[i * 6 + i];
        for (int k = 0; k < i; k++)
        {
            float lik = L[i * 6 + k];
            diag -= lik * lik * D[k];
        }
        if (diag < pivotFloor)
        {
            diag = pivotFloor;
            floorHits += 1u;
        }
        D[i] = diag;
        for (int j = i + 1; j < 6; j++)
        {
            float v = a[j * 6 + i];
            for (int k = 0; k < i; k++)
                v -= L[j * 6 + k] * L[i * 6 + k] * D[k];
            L[j * 6 + i] = v / diag;
        }
    }
    float y[6];
    for (int i = 0; i < 6; i++)
    {
        float s = b.v[i] * scale[i];
        for (int k = 0; k < i; k++) s -= L[i * 6 + k] * y[k];
        y[i] = s;
    }
    float z[6];
    for (int i = 0; i < 6; i++) z[i] = y[i] / D[i];
    float x[6];
    for (int i = 5; i >= 0; i--)
    {
        float s = z[i];
        for (int k = i + 1; k < 6; k++) s -= L[k * 6 + i] * x[k];
        x[i] = s;
    }
    Vec6 outv = vec6Zero();
    for (int i = 0; i < 6; i++) outv.v[i] = x[i] * scale[i];
    return outv;
}

// ==== shapes ====
void worldAabbSphere(float3 c, float r, out float3 mn, out float3 mx)
{
    float3 e = float3(r, r, r);
    mn = c - e; mx = c + e;
}

void worldAabbBox(float3 c, float4 q, float3 half, out float3 mn, out float3 mx)
{
    float3 ex = abs(quatRotate(q, float3(half.x, 0, 0)))
              + abs(quatRotate(q, float3(0, half.y, 0)))
              + abs(quatRotate(q, float3(0, 0, half.z)));
    mn = c - ex; mx = c + ex;
}

void worldAabbCapsule(float3 c, float4 q, float radius, float halfLen, out float3 mn, out float3 mx)
{
    float3 axis = quatRotate(q, float3(0, 1, 0)) * halfLen;
    float3 e = abs(axis) + float3(radius, radius, radius);
    mn = c - e; mx = c + e;
}

bool sphereSphere(
    float3 cA, float rA, float3 cB, float rB, float margin,
    out float3 normal, out float penetration, out float3 pA, out float3 pB)
{
    float3 d = cB - cA;
    float dist = length(d);
    float r = rA + rB + margin;
    if (dist > r && dist > 1e-8) return false;
    normal = dist > 1e-8 ? d / dist : float3(0, 1, 0);
    penetration = r - dist;
    pA = cA + normal * rA;
    pB = cB - normal * rB;
    return true;
}

float3 projectContactForce(float3 f, float mu)
{
    f.x = min(f.x, 0.0);
    float ft = length(f.yz);
    float maxT = mu * (-f.x);
    if (ft > maxT && ft > 1e-12) f.yz *= maxT / ft;
    return f;
}

Vec6 jacobianLin(float3 n, float3 r, bool isBodyA)
{
    Vec6 j = vec6Zero();
    float s = isBodyA ? -1.0 : 1.0;
    j.v[0] = s * n.x; j.v[1] = s * n.y; j.v[2] = s * n.z;
    float3 t = s * cross(r, n);
    j.v[3] = t.x; j.v[4] = t.y; j.v[5] = t.z;
    return j;
}

void contributeContact(
    inout Mat6 A, inout Vec6 rhs,
    Contact c, bool isA, float alpha, float3 bodyPos, float4 bodyRot,
    float3 C /* n,t1,t2 geometric residuals at current pose */,
    float contactOffset)
{
    float3 r = isA ? quatRotate(bodyRot, c.anchorA) : quatRotate(bodyRot, c.anchorB);
    float3 C0 = float3(c.c0N, c.c0T1, c.c0T2);
    // three-avbd: C0 includes margin on the normal; J·dq tracks geometric Δ only, so
    // Ĉ = (1-α)·C0 + (C_geom - C_geom0) = C - α·C0 + (margin, 0, 0).
    // Using C - α·C0 alone drives compression at exact touch and explodes stacks.
    float3 cReg = C - alpha * C0 + float3(contactOffset, 0.0, 0.0);
    float3 f = float3(c.kN, c.kT, c.kT) * cReg + float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
    f = projectContactForce(f, c.friction);
    Vec6 jN = jacobianLin(c.normal, r, isA);
    Vec6 jT1 = jacobianLin(c.tangent1, r, isA);
    Vec6 jT2 = jacobianLin(c.tangent2, r, isA);
    mat6AddOuter(A, jN, c.kN);
    mat6AddOuter(A, jT1, c.kT);
    mat6AddOuter(A, jT2, c.kT);
    vec6AddScaled(rhs, jN, f.x);
    vec6AddScaled(rhs, jT1, f.y);
    vec6AddScaled(rhs, jT2, f.z);
}

// Ball joint: shared C = anchorA - anchorB. Body A uses J=+I, body B uses J=-I
// (jacobianLin's isBodyA flag is inverted relative to contact convention).
void contributeBallJoint(
    inout Mat6 A, inout Vec6 rhs,
    float3 bodyPos, float4 bodyRot, float3 localAnchor,
    float3 C, float3 C0, float k, float3 lambda, bool isBodyA, float alpha)
{
    float3 r = quatRotate(bodyRot, localAnchor);
    // demo3d: C -= C0 * alpha  (infinite-stiffness stabilization)
    float3 Creg = C - alpha * C0;
    for (int axis = 0; axis < 3; axis++)
    {
        float3 n = float3(axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0);
        // Contact jacobianLin(isA=true) → -n; joints want +n on body A.
        Vec6 j = jacobianLin(n, r, !isBodyA);
        float f = k * Creg[axis] + lambda[axis];
        mat6AddOuter(A, j, k);
        vec6AddScaled(rhs, j, f);
    }
}

#endif
