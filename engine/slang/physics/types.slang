// Shared GPU ABI — must match engine-schema layouts.

public struct BodyState
{
    float3 position;
    float invMass;
    float4 orientation; // xyzw
    float3 invInertia;
    uint flags;
    float3 linearVel;
    float sleepTimer;
    float3 angularVel;
    uint generation;
};

public static const uint BODY_DYNAMIC = 1u;
public static const uint BODY_KINEMATIC = 2u;
public static const uint BODY_SLEEPING = 4u;
public static const uint BODY_AWAKE = 8u;

public struct Collider
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

public static const uint SHAPE_SPHERE = 0u;
public static const uint SHAPE_CAPSULE = 1u;
public static const uint SHAPE_BOX = 2u;
public static const uint SHAPE_CONVEX = 3u;
public static const uint SHAPE_PLANE = 4u;

public struct Material
{
    float friction;
    float restitution;
    float contactOffset;
    float _pad;
};

public struct Contact
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

public struct Manifold
{
    uint colliderA;
    uint colliderB;
    uint contactStart;
    uint contactCount;
    float3 normal;
    float friction;
};

public struct Joint
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

public static const uint JOINT_BALL = 0u;
public static const uint JOINT_FIXED = 1u;
public static const uint JOINT_HINGE = 2u;
public static const uint JOINT_SLIDER = 3u;
public static const uint JOINT_D6 = 4u;
public static const uint JOINT_DISTANCE = 5u;
public static const uint JOINT_MAX_DISTANCE = 6u;

public struct AvbdParams
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
    float _pad1;
};

public struct StepCounters
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

public static const uint OVERFLOW_CONTACTS = 1u;
public static const uint OVERFLOW_CANDIDATES = 2u;
public static const uint OVERFLOW_JOINTS = 4u;
public static const uint OVERFLOW_COLORS = 8u;
public static const uint OVERFLOW_MANIFOLDS = 64u;
public static const uint CCD_BUDGET_EXHAUSTED = 16u;
public static const uint CCD_CANDIDATE_MISS = 32u;
public static const uint OVERFLOW_BP_CHAIN = 128u; // linked-list BP hit per-slot walk cap

public struct GpuCommand
{
    uint op;
    uint a;
    uint b;
    uint c;
    float4 f0;
    float4 f1;
    float4 f2;
};

public static const uint CMD_NOP = 0u;
public static const uint CMD_SPAWN_BODY = 1u;
public static const uint CMD_DESTROY_BODY = 2u;
public static const uint CMD_WAKE_BODY = 3u;
public static const uint CMD_TELEPORT = 4u;
public static const uint CMD_SET_VELOCITY = 5u;
public static const uint CMD_SPAWN_COLLIDER = 6u;
public static const uint CMD_DESTROY_COLLIDER = 7u;
public static const uint CMD_REPARENT = 8u;
public static const uint CMD_SPAWN_JOINT = 9u;
public static const uint CMD_DESTROY_JOINT = 10u;

public struct Aabb
{
    float3 mn;
    float _pad0;
    float3 mx;
    float _pad1;
};

public struct CandidatePair
{
    uint colliderA;
    uint colliderB;
    uint key;
    uint _pad;
};
