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
    // x⁻ pose, written by predict. Kept on this record so the primal does not
    // chase gBodies[maxBodies + i] (a separate cache line per contact).
    float3 initialPos;
    float initialPad;
    float4 initialRot;
};

static const uint BODY_DYNAMIC = 1u;
static const uint BODY_KINEMATIC = 2u;
static const uint BODY_SLEEPING = 4u;
static const uint BODY_AWAKE = 8u;
// Vehicle chassis: initialPad holds asfloat(gAabbs index) of its 4-entry wheel block.
static const uint BODY_VEHICLE = 16u;

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
    uint alive; // 0 dead, 1 live; during NP only, >=2 encodes previous contact index + 2
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
    float _pad2; // second tangent penalty kT2; occupies the old padding word (128 B ABI)
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
    float breakForce;  // N; |λ_lin| above this breaks the joint (0 = unbreakable)
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
    float3 c0Ang;      // angular rows' C = axisA×axisB at step start (warmstart_joints)
    float breakTorque; // N·m; |λ_ang| above this breaks it (0 = unbreakable). Joint = 160 B
};

static const uint JOINT_BALL = 0u;
static const uint JOINT_FIXED = 1u;
static const uint JOINT_HINGE = 2u;
static const uint JOINT_SLIDER = 3u;
static const uint JOINT_D6 = 4u;
static const uint JOINT_DISTANCE = 5u;
static const uint JOINT_MAX_DISTANCE = 6u;
// Vehicle wheel on a dynamic body: links chassis and ground in the constraint graph;
// the ground body's primal adds the wheel's ground block (flags = block index).
static const uint JOINT_TYRE = 7u;
// Joint.flags bit for user joints (tyre joints use flags as their ground block):
// set by solve_dual when the joint breaks; every joint kernel then skips it.
static const uint JOINT_FLAG_BROKEN = 0x80000000u;
// Set by solve_dual while the multipliers are still moving: the joint's bodies may
// not fall asleep with unconverged forces (sleep_island_gate_joints).
static const uint JOINT_FLAG_UNSETTLED = 0x40000000u;

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
static const uint OVERFLOW_BP_CHAIN = 128u;
static const uint OVERFLOW_WARM_HASH = 256u; // record warm hash probe limit exhausted
static const uint OVERFLOW_LARGE_STATICS = 512u; // more oversized statics than LARGE_STATIC_CAP

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
    // Unit-quaternion form also leaves the rotation axis exact in f32. The
    // expanded q*v*conj(q) scales it by dot(q,q), losing touching ring contacts.
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
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
    // Lower triangle only. solve6x6 equilibrates every slot, then factors a6, a12, a13,
    // a18..a20, a24..a27, a30..a34 and the diagonal. The upper triangle is never read.
    float j0 = j.v[0], j1 = j.v[1], j2 = j.v[2], j3 = j.v[3], j4 = j.v[4], j5 = j.v[5];
    a.m[0] += k * j0 * j0;
    a.m[6] += k * j1 * j0; a.m[7] += k * j1 * j1;
    a.m[12] += k * j2 * j0; a.m[13] += k * j2 * j1; a.m[14] += k * j2 * j2;
    a.m[18] += k * j3 * j0; a.m[19] += k * j3 * j1; a.m[20] += k * j3 * j2; a.m[21] += k * j3 * j3;
    a.m[24] += k * j4 * j0; a.m[25] += k * j4 * j1; a.m[26] += k * j4 * j2; a.m[27] += k * j4 * j3; a.m[28] += k * j4 * j4;
    a.m[30] += k * j5 * j0; a.m[31] += k * j5 * j1; a.m[32] += k * j5 * j2; a.m[33] += k * j5 * j3; a.m[34] += k * j5 * j4; a.m[35] += k * j5 * j5;
}

void mat6AddDiag(inout Mat6 a, float3 lin, float3 ang)
{
    a.m[0] += lin.x; a.m[7] += lin.y; a.m[14] += lin.z;
    a.m[21] += ang.x; a.m[28] += ang.y; a.m[35] += ang.z;
}

void vec6AddScaled(inout Vec6 a, Vec6 j, float s)
{
    a.v[0] += j.v[0] * s; a.v[1] += j.v[1] * s; a.v[2] += j.v[2] * s;
    a.v[3] += j.v[3] * s; a.v[4] += j.v[4] * s; a.v[5] += j.v[5] * s;
}

Vec6 solve6x6(Mat6 A, Vec6 b, float pivotFloor, inout uint floorHits)
{
    float a0 = A.m[0];
    float a1 = A.m[1];
    float a2 = A.m[2];
    float a3 = A.m[3];
    float a4 = A.m[4];
    float a5 = A.m[5];
    float a6 = A.m[6];
    float a7 = A.m[7];
    float a8 = A.m[8];
    float a9 = A.m[9];
    float a10 = A.m[10];
    float a11 = A.m[11];
    float a12 = A.m[12];
    float a13 = A.m[13];
    float a14 = A.m[14];
    float a15 = A.m[15];
    float a16 = A.m[16];
    float a17 = A.m[17];
    float a18 = A.m[18];
    float a19 = A.m[19];
    float a20 = A.m[20];
    float a21 = A.m[21];
    float a22 = A.m[22];
    float a23 = A.m[23];
    float a24 = A.m[24];
    float a25 = A.m[25];
    float a26 = A.m[26];
    float a27 = A.m[27];
    float a28 = A.m[28];
    float a29 = A.m[29];
    float a30 = A.m[30];
    float a31 = A.m[31];
    float a32 = A.m[32];
    float a33 = A.m[33];
    float a34 = A.m[34];
    float a35 = A.m[35];
    float s0 = a0 > 1e-12 ? rsqrt(a0) : 1.0;
    float s1 = a7 > 1e-12 ? rsqrt(a7) : 1.0;
    float s2 = a14 > 1e-12 ? rsqrt(a14) : 1.0;
    float s3 = a21 > 1e-12 ? rsqrt(a21) : 1.0;
    float s4 = a28 > 1e-12 ? rsqrt(a28) : 1.0;
    float s5 = a35 > 1e-12 ? rsqrt(a35) : 1.0;
    a0 = a0 * s0 * s0;
    a1 = a1 * s0 * s1;
    a2 = a2 * s0 * s2;
    a3 = a3 * s0 * s3;
    a4 = a4 * s0 * s4;
    a5 = a5 * s0 * s5;
    a6 = a6 * s1 * s0;
    a7 = a7 * s1 * s1;
    a8 = a8 * s1 * s2;
    a9 = a9 * s1 * s3;
    a10 = a10 * s1 * s4;
    a11 = a11 * s1 * s5;
    a12 = a12 * s2 * s0;
    a13 = a13 * s2 * s1;
    a14 = a14 * s2 * s2;
    a15 = a15 * s2 * s3;
    a16 = a16 * s2 * s4;
    a17 = a17 * s2 * s5;
    a18 = a18 * s3 * s0;
    a19 = a19 * s3 * s1;
    a20 = a20 * s3 * s2;
    a21 = a21 * s3 * s3;
    a22 = a22 * s3 * s4;
    a23 = a23 * s3 * s5;
    a24 = a24 * s4 * s0;
    a25 = a25 * s4 * s1;
    a26 = a26 * s4 * s2;
    a27 = a27 * s4 * s3;
    a28 = a28 * s4 * s4;
    a29 = a29 * s4 * s5;
    a30 = a30 * s5 * s0;
    a31 = a31 * s5 * s1;
    a32 = a32 * s5 * s2;
    a33 = a33 * s5 * s3;
    a34 = a34 * s5 * s4;
    a35 = a35 * s5 * s5;
    float L0 = 0.0;
    float L1 = 0.0;
    float L2 = 0.0;
    float L3 = 0.0;
    float L4 = 0.0;
    float L5 = 0.0;
    float L6 = 0.0;
    float L7 = 0.0;
    float L8 = 0.0;
    float L9 = 0.0;
    float L10 = 0.0;
    float L11 = 0.0;
    float L12 = 0.0;
    float L13 = 0.0;
    float L14 = 0.0;
    float L15 = 0.0;
    float L16 = 0.0;
    float L17 = 0.0;
    float L18 = 0.0;
    float L19 = 0.0;
    float L20 = 0.0;
    float L21 = 0.0;
    float L22 = 0.0;
    float L23 = 0.0;
    float L24 = 0.0;
    float L25 = 0.0;
    float L26 = 0.0;
    float L27 = 0.0;
    float L28 = 0.0;
    float L29 = 0.0;
    float L30 = 0.0;
    float L31 = 0.0;
    float L32 = 0.0;
    float L33 = 0.0;
    float L34 = 0.0;
    float L35 = 0.0;
    float D0;
    L0 = 1.0;
    float diag0 = a0;
    if (diag0 < pivotFloor) { diag0 = pivotFloor; floorHits = floorHits + 1u; }
    D0 = diag0;
    float v1_0 = a6;
    L6 = v1_0 / diag0;
    float v2_0 = a12;
    L12 = v2_0 / diag0;
    float v3_0 = a18;
    L18 = v3_0 / diag0;
    float v4_0 = a24;
    L24 = v4_0 / diag0;
    float v5_0 = a30;
    L30 = v5_0 / diag0;
    float D1;
    L7 = 1.0;
    float diag1 = a7;
    diag1 = diag1 - L6 * L6 * D0;
    if (diag1 < pivotFloor) { diag1 = pivotFloor; floorHits = floorHits + 1u; }
    D1 = diag1;
    float v2_1 = a13;
    v2_1 = v2_1 - L12 * L6 * D0;
    L13 = v2_1 / diag1;
    float v3_1 = a19;
    v3_1 = v3_1 - L18 * L6 * D0;
    L19 = v3_1 / diag1;
    float v4_1 = a25;
    v4_1 = v4_1 - L24 * L6 * D0;
    L25 = v4_1 / diag1;
    float v5_1 = a31;
    v5_1 = v5_1 - L30 * L6 * D0;
    L31 = v5_1 / diag1;
    float D2;
    L14 = 1.0;
    float diag2 = a14;
    diag2 = diag2 - L12 * L12 * D0;
    diag2 = diag2 - L13 * L13 * D1;
    if (diag2 < pivotFloor) { diag2 = pivotFloor; floorHits = floorHits + 1u; }
    D2 = diag2;
    float v3_2 = a20;
    v3_2 = v3_2 - L18 * L12 * D0;
    v3_2 = v3_2 - L19 * L13 * D1;
    L20 = v3_2 / diag2;
    float v4_2 = a26;
    v4_2 = v4_2 - L24 * L12 * D0;
    v4_2 = v4_2 - L25 * L13 * D1;
    L26 = v4_2 / diag2;
    float v5_2 = a32;
    v5_2 = v5_2 - L30 * L12 * D0;
    v5_2 = v5_2 - L31 * L13 * D1;
    L32 = v5_2 / diag2;
    float D3;
    L21 = 1.0;
    float diag3 = a21;
    diag3 = diag3 - L18 * L18 * D0;
    diag3 = diag3 - L19 * L19 * D1;
    diag3 = diag3 - L20 * L20 * D2;
    if (diag3 < pivotFloor) { diag3 = pivotFloor; floorHits = floorHits + 1u; }
    D3 = diag3;
    float v4_3 = a27;
    v4_3 = v4_3 - L24 * L18 * D0;
    v4_3 = v4_3 - L25 * L19 * D1;
    v4_3 = v4_3 - L26 * L20 * D2;
    L27 = v4_3 / diag3;
    float v5_3 = a33;
    v5_3 = v5_3 - L30 * L18 * D0;
    v5_3 = v5_3 - L31 * L19 * D1;
    v5_3 = v5_3 - L32 * L20 * D2;
    L33 = v5_3 / diag3;
    float D4;
    L28 = 1.0;
    float diag4 = a28;
    diag4 = diag4 - L24 * L24 * D0;
    diag4 = diag4 - L25 * L25 * D1;
    diag4 = diag4 - L26 * L26 * D2;
    diag4 = diag4 - L27 * L27 * D3;
    if (diag4 < pivotFloor) { diag4 = pivotFloor; floorHits = floorHits + 1u; }
    D4 = diag4;
    float v5_4 = a34;
    v5_4 = v5_4 - L30 * L24 * D0;
    v5_4 = v5_4 - L31 * L25 * D1;
    v5_4 = v5_4 - L32 * L26 * D2;
    v5_4 = v5_4 - L33 * L27 * D3;
    L34 = v5_4 / diag4;
    float D5;
    L35 = 1.0;
    float diag5 = a35;
    diag5 = diag5 - L30 * L30 * D0;
    diag5 = diag5 - L31 * L31 * D1;
    diag5 = diag5 - L32 * L32 * D2;
    diag5 = diag5 - L33 * L33 * D3;
    diag5 = diag5 - L34 * L34 * D4;
    if (diag5 < pivotFloor) { diag5 = pivotFloor; floorHits = floorHits + 1u; }
    D5 = diag5;
    float y0 = b.v[0] * s0;
    float y1 = b.v[1] * s1;
    y1 = y1 - L6 * y0;
    float y2 = b.v[2] * s2;
    y2 = y2 - L12 * y0;
    y2 = y2 - L13 * y1;
    float y3 = b.v[3] * s3;
    y3 = y3 - L18 * y0;
    y3 = y3 - L19 * y1;
    y3 = y3 - L20 * y2;
    float y4 = b.v[4] * s4;
    y4 = y4 - L24 * y0;
    y4 = y4 - L25 * y1;
    y4 = y4 - L26 * y2;
    y4 = y4 - L27 * y3;
    float y5 = b.v[5] * s5;
    y5 = y5 - L30 * y0;
    y5 = y5 - L31 * y1;
    y5 = y5 - L32 * y2;
    y5 = y5 - L33 * y3;
    y5 = y5 - L34 * y4;
    float z0 = y0 / D0;
    float z1 = y1 / D1;
    float z2 = y2 / D2;
    float z3 = y3 / D3;
    float z4 = y4 / D4;
    float z5 = y5 / D5;
    float x5 = z5;
    float x4 = z4;
    x4 = x4 - L34 * x5;
    float x3 = z3;
    x3 = x3 - L27 * x4;
    x3 = x3 - L33 * x5;
    float x2 = z2;
    x2 = x2 - L20 * x3;
    x2 = x2 - L26 * x4;
    x2 = x2 - L32 * x5;
    float x1 = z1;
    x1 = x1 - L13 * x2;
    x1 = x1 - L19 * x3;
    x1 = x1 - L25 * x4;
    x1 = x1 - L31 * x5;
    float x0 = z0;
    x0 = x0 - L6 * x1;
    x0 = x0 - L12 * x2;
    x0 = x0 - L18 * x3;
    x0 = x0 - L24 * x4;
    x0 = x0 - L30 * x5;
    Vec6 outv;
    outv.v[0] = x0 * s0;
    outv.v[1] = x1 * s1;
    outv.v[2] = x2 * s2;
    outv.v[3] = x3 * s3;
    outv.v[4] = x4 * s4;
    outv.v[5] = x5 * s5;
    return outv;
}

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
    float contactOffset, float3 rWorld)
{
    float3 r = rWorld;
    float3 C0 = float3(c.c0N, c.c0T1, c.c0T2);
    // three-avbd: C0 includes margin on the normal; J·dq tracks geometric Δ only, so
    // Ĉ = (1-α)·C0 + (C_geom - C_geom0) = C - α·C0 + (margin, 0, 0).
    // Using C - α·C0 alone drives compression at exact touch and explodes stacks.
    float3 cReg = C - alpha * C0 + float3(contactOffset, 0.0, 0.0);
    float3 fRaw = float3(c.kN, c.kT, c._pad2) * cReg + float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
    float3 f = projectContactForce(fRaw, c.friction);
    // Keep the Hessian penalties for both tangent rows, as in three-avbd.
    // Dropping a saturated row's stiffness makes the Newton step overshoot.
    Vec6 jN = jacobianLin(c.normal, r, isA);
    Vec6 jT1 = jacobianLin(c.tangent1, r, isA);
    Vec6 jT2 = jacobianLin(c.tangent2, r, isA);
    mat6AddOuter(A, jN, c.kN);
    mat6AddOuter(A, jT1, c.kT);
    mat6AddOuter(A, jT2, c._pad2);
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
