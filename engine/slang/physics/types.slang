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
    float3 initialPos;
    float initialPad;
    float4 initialRot;
};

public static const uint BODY_DYNAMIC = 1u;
public static const uint BODY_KINEMATIC = 2u;
public static const uint BODY_SLEEPING = 4u;
public static const uint BODY_AWAKE = 8u;
public static const uint BODY_VEHICLE = 32u; // 16 is BODY_WAKE_PENDING
public static const uint BODY_PARTICLE = 64u;
public static const uint BODY_FOLLOW = 128u;

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
    // Layout mirrors engine-schema `Joint` (288 B). Row math: joints.slang.
    uint bodyA;
    uint bodyB;
    uint flags;
    uint motion; // 2 bits per DOF (X, Y, Z, twist, swing1, swing2)
    float3 anchorA;
    float torqueArm;
    float4 frameA;
    float3 anchorB;
    float linStiffness;
    float4 frameB;
    float4 restStart;
    float3 linLo;
    float twistLo;
    float3 linHi;
    float twistHi;
    float swingY;
    float swingZ;
    float radialMin;
    float radialMax;
    float angStiffness;
    float limitStiffness;
    float limitDamping;
    float driveForceLimit;
    float breakForce;
    float breakTorque;
    float yieldTorque;
    float breakBend;
    float driveTarget;
    float driveVelocity;
    float driveStiffness;
    float driveDamping;
    // λ, penalty and C0 per row slot (slots 0..3 in *Lo, 4..7 in *Hi). float4
    // pairs rather than float[8]: struct copies then need no SPIR-V
    // OpCopyLogical, which naga's SPIR-V front end rejects.
    float4 lambdaLo;
    float4 lambdaHi;
    float4 penaltyLo;
    float4 penaltyHi;
    float4 c0Lo;
    float4 c0Hi;
};

public static const uint JOINT_FLAG_BROKEN = 1u;
public static const uint JOINT_FLAG_IDLE = 65536u;
public static const uint JOINT_FLAG_COLLIDE = 2u;
public static const uint JOINT_FLAG_RADIAL = 4u;
public static const uint JOINT_FLAG_CONE = 8u;
public static const uint JOINT_FLAG_DRIVE = 16u;
public static const uint JOINT_DRIVE_DOF_SHIFT = 8u;
public static const uint MOTION_LOCKED = 0u;
public static const uint MOTION_LIMITED = 1u;
public static const uint MOTION_FREE = 2u;
public static const float JOINT_HARD = 1.0e30;
public static const float JOINT_BIG = 3.0e38;
// Fracture bond (JOINT_FLAG_BOND): a weld whose six rows run in the bond frame
// (frame A: X = normal A->B, Y = u, Z = v) with fixed per-row penalties set by
// the host, breaking by Shardwright's utilization against the capacities
// stored in linLo (tension, compression, shear) and twistLo (bending).
public static const uint JOINT_FLAG_BOND = 32u;
// Vehicle wheel standing on a dynamic body (vehicle_link): links chassis and
// ground in the graph; the ground body's primal adds the wheel's ground block,
// whose gAabbs index is in Joint.motion. Pair hash, warm start and dual skip it.
public static const uint JOINT_FLAG_TYRE = 64u;
// Set by solve_dual on the final iteration while the multipliers still move:
// the joint's bodies may not fall asleep with unconverged forces.
public static const uint JOINT_FLAG_UNSETTLED = 128u;

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
public static const uint OVERFLOW_WARM_HASH = 256u; // record warm hash probe limit exhausted
public static const uint OVERFLOW_LARGE_STATICS = 512u; // more oversized statics than LARGE_STATIC_CAP

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
public static const uint CMD_ADD_FORCE = 11u;
public static const uint CMD_APPLY_IMPULSE = 12u;
public static const uint CMD_RUN_MEMBER = 0xFFFFFFFFu;
public static const uint CMD_ADD_FORCE_AT = 13u;
public static const uint CMD_ANGULAR_IMPULSE = 14u;
public static const uint CMD_SET_MASS_FRAME = 15u;
public static const uint CMD_SET_KINEMATIC_TARGET = 16u;
public static const uint CMD_NO_WAKE = 1u;
public static const uint CMD_AT_CENTER = 2u;
public static const uint CMD_LINEAR_ONLY = 4u;
public static const uint CMD_ANGULAR_ONLY = 8u;

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
