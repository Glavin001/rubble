// Contact storage is private to the WGSL backend. Canonical kernels keep their
// existing ABI; persistent kernels retain the hot 64 bytes across frame swaps.
#if defined(ENGINE_COMPACT_SOLVE)
struct SolverPoint
{
    float3 anchorA; float c0N;
    float3 anchorB; float c0T1;
    float3 lambda; float c0T2;
    float3 penalty; uint pairAndStick;
};
BIND(27) RWStructuredBuffer<SolverPoint> gSolverPoints;
static const uint SOLVER_DEAD = 0x7FFFFFFFu;
static const uint SOLVER_STICK = 0x80000000u;
#endif

#if defined(ENGINE_PERSISTENT_CONTACTS)
// Keep every feature/generation bit. In particular sphere fallback features may
// use bit 31 and must not be combined with the sticking flag.
struct ContactMeta
{
    uint colliderA; uint colliderB; uint featureId; uint generation;
    float3 normal; float friction;
};
BIND(4) RWStructuredBuffer<ContactMeta> gContactMeta;
static const uint PERSIST_OWNER_MASK = 0x3FFFFFFFu;
static const uint PERSIST_REUSED = 0x40000000u;

uint persistentContactState(uint absolute)
{
    uint tag = gSolverPoints[absolute].pairAndStick;
    if ((tag & PERSIST_OWNER_MASK) == 0u) return 0u;
    return (tag & PERSIST_REUSED) != 0u ? gContactMeta[absolute].featureId : 1u;
}

Contact loadPersistentContact(uint absolute)
{
    SolverPoint pt = gSolverPoints[absolute];
    ContactMeta meta = gContactMeta[absolute];
    Contact c = (Contact)0;
    c.colliderA = meta.colliderA; c.colliderB = meta.colliderB;
    c.featureId = meta.featureId;
    c.alive = persistentContactState(absolute);
    c.normal = meta.normal; c.friction = meta.friction;
    orthonormalBasis(c.normal, c.tangent1, c.tangent2);
    c.anchorA = pt.anchorA; c.anchorB = pt.anchorB;
    c.c0N = pt.c0N; c.c0T1 = pt.c0T1; c.c0T2 = pt.c0T2;
    c.penetration = pt.c0N; // Solver residual is evaluated from poses, never persisted.
    c.lambdaN = pt.lambda.x; c.lambdaT1 = pt.lambda.y; c.lambdaT2 = pt.lambda.z;
    c.kN = pt.penalty.x; c.kT = pt.penalty.y; c._pad2 = pt.penalty.z;
    c._pad0 = (pt.pairAndStick & SOLVER_STICK) != 0u ? 1.0 : 0.0;
    c._pad1 = asfloat(meta.generation);
    return c;
}

void storePersistentContact(uint absolute, uint owner, Contact c)
{
    ContactMeta meta;
    meta.colliderA = c.colliderA; meta.colliderB = c.colliderB;
    meta.featureId = c.featureId; meta.generation = asuint(c._pad1);
    meta.normal = c.normal; meta.friction = c.friction;
    gContactMeta[absolute] = meta;
    SolverPoint pt;
    pt.anchorA = c.anchorA; pt.anchorB = c.anchorB;
    pt.c0N = c.c0N; pt.c0T1 = c.c0T1; pt.c0T2 = c.c0T2;
    pt.lambda = float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
    pt.penalty = float3(c.kN, c.kT, c._pad2);
    pt.pairAndStick = c.alive == 0u ? 0u : (owner + 1u) |
        (c._pad0 != 0.0 ? SOLVER_STICK : 0u);
    gSolverPoints[absolute] = pt;
}

void updatePersistentContact(uint absolute, Contact c)
{
    uint ownerTag = gSolverPoints[absolute].pairAndStick & PERSIST_OWNER_MASK;
    storePersistentContact(absolute, max(ownerTag, 1u) - 1u, c);
}

void markPersistentReuse(uint absolute, uint owner, uint sourceTag)
{
    // featureId temporarily holds the canonical reuse source tag (previous + 2).
    // Persistence consumes it before installing the complete previous record.
    gContactMeta[absolute].featureId = sourceTag;
    gSolverPoints[absolute].pairAndStick = (owner + 1u) | PERSIST_REUSED;
}

#define READ_CONTACT(i) loadPersistentContact(i)
#define CONTACT_STATE(i) persistentContactState(i)
#define STORE_CONTACT(i, c) updatePersistentContact(i, c)
#define STORE_NEW_CONTACT(i, owner, c) storePersistentContact(i, owner, c)
#define MARK_REUSED_CONTACT(i, owner, tag) markPersistentReuse(i, owner, tag)
#else
#define READ_CONTACT(i) gContacts[i]
#define CONTACT_STATE(i) gContacts[i].alive
#define STORE_CONTACT(i, c) gContacts[i] = c
#define STORE_NEW_CONTACT(i, owner, c) gContacts[i] = c
#define MARK_REUSED_CONTACT(i, owner, tag) gContacts[i].alive = tag
#endif

#if defined(ENGINE_COMPACT_SOLVE)
uint solverPointIndex(uint local)
{
#if defined(ENGINE_PERSISTENT_CONTACTS)
    return HOST_S(11) + local;
#else
    return local;
#endif
}

SolverPoint loadSolverPoint(uint local)
{
    SolverPoint pt = gSolverPoints[solverPointIndex(local)];
#if defined(ENGINE_PERSISTENT_CONTACTS)
    uint ownerTag = pt.pairAndStick & PERSIST_OWNER_MASK;
    pt.pairAndStick = (ownerTag == 0u ? SOLVER_DEAD : ownerTag - 1u) |
        (pt.pairAndStick & SOLVER_STICK);
#endif
    return pt;
}

uint solverPointTag(uint pairIndex, bool stick)
{
#if defined(ENGINE_PERSISTENT_CONTACTS)
    return (pairIndex + 1u) | (stick ? SOLVER_STICK : 0u);
#else
    return pairIndex | (stick ? SOLVER_STICK : 0u);
#endif
}
#endif
