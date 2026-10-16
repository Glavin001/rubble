// Thin C API over Jolt Physics 5.6 for the benchmark adapter (src/main.rs drives it over FFI).
// Compiled by build.rs with the exact JPH_* defines the static library was built with; the
// create call also checks VerifyJoltVersionID() at runtime.
//
// The struct layouts below must match the #[repr(C)] mirrors in src/main.rs.

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/GroupFilter.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace JPH;

// ---- read-only access to two private counters (contact manifolds / body pairs of the last step).
// Explicit template instantiation may name private members; nothing is modified.
template <typename Tag, typename Tag::type M> struct BenchRob { friend typename Tag::type bench_get(Tag) { return M; } };
template <typename Tag, auto M> struct BenchRobAuto { friend auto bench_get(Tag) { return M; } };
struct BenchCMTag { using type = ContactConstraintManager PhysicsSystem::*; friend type bench_get(BenchCMTag); };
template struct BenchRob<BenchCMTag, &PhysicsSystem::mContactManager>;
struct BenchIdxTag { using type = int ContactConstraintManager::*; friend type bench_get(BenchIdxTag); };
template struct BenchRob<BenchIdxTag, &ContactConstraintManager::mCacheWriteIdx>;
struct BenchCacheTag { friend auto bench_get(BenchCacheTag); };
template struct BenchRobAuto<BenchCacheTag, &ContactConstraintManager::mCache>;

extern "C" {

struct BenchJBody {
    uint32_t shape;      // 0 box, 1 sphere, 2 convex hulls, 3 plane (static half-space)
    uint32_t is_static;
    float half[3];       // box half extents, or [radius, 0, 0]
    uint32_t hull_first;
    uint32_t hull_count;
    float mass;
    float inertia[3];
    float friction;
    float restitution;
    float position[3];
    float rotation[4];   // x y z w
    float linvel[3];
    float angvel[3];
};

struct BenchJJoint {
    uint32_t body_a;
    uint32_t body_b;
    uint32_t collide;
    float anchor_a[3];
    float anchor_b[3];
    uint32_t kind;             // 0 ball, 1 D6
    float frame_a[4];          // x y z w
    float frame_b[4];
    uint32_t motion;           // 2 bits per DOF X Y Z twist swing1 swing2: 0 locked 1 limited 2 free
    float twist[2];
    float swing[2];            // half-angles about frame Y, Z
    uint32_t cone;
    float linear_x[2];
    uint32_t drive_dof;        // 0..2 linear, 3 twist, ~0 none
    float drive_velocity;
    float drive_force_limit;
    float drive_stiffness;     // 0: velocity motor
    float drive_damping;
    float drive_target;
};

struct BenchJHulls {
    uint32_t n_hulls;
    const uint32_t* first;
    const uint32_t* count;
    uint32_t n_vertices;
    const float* vertices;
};

struct BenchJSettings {
    float gravity[3];
    float dt;
    uint32_t threads;          // total threads stepping: caller + (threads - 1) pool workers
    uint32_t vel_steps;
    uint32_t pos_steps;
    uint32_t sleep;
    uint32_t max_bodies;
    uint32_t max_pairs;
    uint32_t max_contacts;
    uint32_t joint_mode;       // 0 specialised constraints (point/hinge/swing-twist/6DOF), 1 all D6 as 6DOF
    uint64_t temp_bytes;
    float max_linear_velocity;
    float max_angular_velocity;
    float box_radius_frac;     // box/hull convex radius = min(max_convex_radius, frac * min half extent)
    float max_convex_radius;
    float plane_half_extent;
    uint32_t max_jobs;
    uint32_t max_barriers;
};

struct BenchJInfo {
    uint32_t version_major, version_minor, version_patch;
    uint32_t version_id_ok;
    float baumgarte, penetration_slop, speculative_distance, max_penetration_distance;
    float linear_cast_threshold, manifold_tolerance, min_velocity_for_restitution;
    float time_before_sleep, point_velocity_sleep_threshold;
    uint32_t deterministic, warm_start, body_pair_cache, manifold_reduction, large_island_splitter, check_active_edges;
    uint32_t vel_steps, pos_steps;
    uint32_t workers;
    uint32_t max_constraint_size;
    // shapes
    uint32_t boxes, spheres, planes, box_shapes_unique;
    float box_radius_min, box_radius_max;
    uint32_t hulls_built, hull_vertices_in, hull_vertices_out, hulls_changed, compounds;
    float hull_radius_min, hull_radius_max, hull_max_outside, max_com_shift;
    float hull_tolerance, hull_max_error_convex_radius;
    // joints
    uint32_t n_point, n_hinge, n_swing_twist, n_six_dof, n_motors, filtered_bodies, filtered_pairs;
    float default_convex_radius;
};

struct BenchJCounters {
    uint32_t bodies;
    uint32_t active_bodies;
    uint32_t body_pairs;
    uint32_t manifolds;
    uint32_t constraints;
    uint32_t errors_last;
    uint32_t errors_sticky;
    uint32_t steps;
};

}  // extern "C"

namespace {

namespace Layers {
static constexpr ObjectLayer NON_MOVING = 0;
static constexpr ObjectLayer MOVING = 1;
static constexpr ObjectLayer NUM = 2;
}  // namespace Layers
namespace BPLayers {
static constexpr BroadPhaseLayer NON_MOVING(0);
static constexpr BroadPhaseLayer MOVING(1);
static constexpr uint NUM = 2;
}  // namespace BPLayers

class ObjectLayerPairFilterImpl final : public ObjectLayerPairFilter {
public:
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override {
        if (a == Layers::NON_MOVING) return b == Layers::MOVING;
        return true;
    }
};

class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
public:
    uint GetNumBroadPhaseLayers() const override { return BPLayers::NUM; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer l) const override { return l == Layers::NON_MOVING ? BPLayers::NON_MOVING : BPLayers::MOVING; }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(BroadPhaseLayer l) const override { return l == BPLayers::NON_MOVING ? "NON_MOVING" : "MOVING"; }
#endif
};

class ObjectVsBPLayerFilterImpl final : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer a, BroadPhaseLayer b) const override {
        if (a == Layers::NON_MOVING) return b == BPLayers::MOVING;
        return true;
    }
};

// Joined bodies do not collide: every body in a non-colliding joint gets this filter with
// GroupID = scene index; CanCollide looks the pair up in a CSR adjacency list (a few entries
// per body). Bodies without joints have no filter and never reach this code.
class JointPairFilter final : public GroupFilter {
public:
    std::vector<uint32_t> offsets;  // n + 1
    std::vector<uint32_t> partners;
    bool CanCollide(const CollisionGroup& g1, const CollisionGroup& g2) const override {
        CollisionGroup::GroupID a = g1.GetGroupID(), b = g2.GetGroupID();
        if (a == CollisionGroup::cInvalidGroup || b == CollisionGroup::cInvalidGroup) return true;
        for (uint32_t k = offsets[a]; k < offsets[a + 1]; k++)
            if (partners[k] == b) return false;
        return true;
    }
};

void trace_impl(const char* fmt, ...) {
    va_list list;
    va_start(list, fmt);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, list);
    va_end(list);
    fprintf(stderr, "[jolt] %s\n", buf);
}

struct World {
    BPLayerInterfaceImpl bp_layers;
    ObjectVsBPLayerFilterImpl obj_vs_bp;
    ObjectLayerPairFilterImpl obj_pair;
    TempAllocatorImpl* temp = nullptr;
    JobSystemThreadPool* jobs = nullptr;
    PhysicsSystem* system = nullptr;
    std::vector<Body*> bodies;
    std::vector<uint8_t> is_dynamic;
    Ref<JointPairFilter> filter;
    uint32_t n = 0;
    uint32_t n_constraints = 0;
    uint32_t errors_last = 0, errors_sticky = 0, steps = 0;
    float dt = 0;

    ~World() {
        delete system;  // releases constraints and bodies
        delete jobs;
        delete temp;
    }
};

bool g_registered = false;

inline Vec3 v3(const float* f) { return Vec3(f[0], f[1], f[2]); }
inline Quat q4(const float* f) { return Quat(f[0], f[1], f[2], f[3]); }

}  // namespace

extern "C" {

uint32_t bench_jolt_max_constraint_size() { return ContactConstraintManager::cMaxConstraintSize; }

void* bench_jolt_create(const BenchJBody* in_bodies, uint32_t nb, const BenchJJoint* joints, uint32_t nj, const BenchJHulls* hulls,
                        const BenchJSettings* s, BenchJInfo* info, char* err, uint32_t errlen) {
#define BENCH_FAIL(...)                              \
    do {                                             \
        snprintf(err, errlen, __VA_ARGS__);          \
        delete w;                                    \
        return nullptr;                              \
    } while (0)
    World* w = nullptr;
    std::memset(info, 0, sizeof(*info));
    info->version_major = JPH_VERSION_MAJOR;
    info->version_minor = JPH_VERSION_MINOR;
    info->version_patch = JPH_VERSION_PATCH;
    info->version_id_ok = VerifyJoltVersionID() ? 1 : 0;
    if (!info->version_id_ok) BENCH_FAIL("VerifyJoltVersionID() failed: the shim's JPH_* defines differ from the library's");
    if (!g_registered) {
        RegisterDefaultAllocator();
        Trace = trace_impl;
        Factory::sInstance = new Factory();
        RegisterTypes();
        g_registered = true;
    }
    w = new World();
    w->n = nb;
    w->dt = s->dt;
    w->temp = new TempAllocatorImpl(size_t(s->temp_bytes));
    uint32_t workers = s->threads > 0 ? s->threads - 1 : 0;
    w->jobs = new JobSystemThreadPool(s->max_jobs, s->max_barriers, int(workers));
    info->workers = workers;
    info->max_constraint_size = ContactConstraintManager::cMaxConstraintSize;

    w->system = new PhysicsSystem();
    w->system->Init(s->max_bodies, 0, s->max_pairs, s->max_contacts, w->bp_layers, w->obj_vs_bp, w->obj_pair);
    PhysicsSettings ps = w->system->GetPhysicsSettings();
    ps.mNumVelocitySteps = s->vel_steps;
    ps.mNumPositionSteps = s->pos_steps;
    ps.mAllowSleeping = s->sleep != 0;
    w->system->SetPhysicsSettings(ps);
    w->system->SetGravity(v3(s->gravity));
    info->baumgarte = ps.mBaumgarte;
    info->penetration_slop = ps.mPenetrationSlop;
    info->speculative_distance = ps.mSpeculativeContactDistance;
    info->max_penetration_distance = ps.mMaxPenetrationDistance;
    info->linear_cast_threshold = ps.mLinearCastThreshold;
    info->manifold_tolerance = ps.mManifoldTolerance;
    info->min_velocity_for_restitution = ps.mMinVelocityForRestitution;
    info->time_before_sleep = ps.mTimeBeforeSleep;
    info->point_velocity_sleep_threshold = ps.mPointVelocitySleepThreshold;
    info->deterministic = ps.mDeterministicSimulation;
    info->warm_start = ps.mConstraintWarmStart;
    info->body_pair_cache = ps.mUseBodyPairContactCache;
    info->manifold_reduction = ps.mUseManifoldReduction;
    info->large_island_splitter = ps.mUseLargeIslandSplitter;
    info->check_active_edges = ps.mCheckActiveEdges;
    info->vel_steps = ps.mNumVelocitySteps;
    info->pos_steps = ps.mNumPositionSteps;
    info->default_convex_radius = cDefaultConvexRadius;
    info->box_radius_min = FLT_MAX;
    info->hull_radius_min = FLT_MAX;
    {
        ConvexHullShapeSettings def;
        info->hull_tolerance = def.mHullTolerance;
        info->hull_max_error_convex_radius = def.mMaxErrorConvexRadius;
    }

    // Collision filter for joined bodies
    w->filter = new JointPairFilter();
    {
        std::vector<std::vector<uint32_t>> adj(nb);
        for (uint32_t j = 0; j < nj; j++) {
            const BenchJJoint& jt = joints[j];
            if (jt.body_a >= nb || jt.body_b >= nb) BENCH_FAIL("joint %u: body index out of range", j);
            if (jt.collide) continue;
            adj[jt.body_a].push_back(jt.body_b);
            adj[jt.body_b].push_back(jt.body_a);
        }
        w->filter->offsets.assign(nb + 1, 0);
        for (uint32_t i = 0; i < nb; i++) {
            std::sort(adj[i].begin(), adj[i].end());
            adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
            w->filter->offsets[i + 1] = w->filter->offsets[i] + uint32_t(adj[i].size());
            if (!adj[i].empty()) info->filtered_bodies++;
        }
        w->filter->partners.reserve(w->filter->offsets[nb]);
        for (uint32_t i = 0; i < nb; i++) w->filter->partners.insert(w->filter->partners.end(), adj[i].begin(), adj[i].end());
        info->filtered_pairs = w->filter->offsets[nb] / 2;
    }

    auto radius_for = [&](float min_half) { return std::max(0.0f, std::min(s->max_convex_radius, s->box_radius_frac * min_half)); };

    std::map<std::tuple<float, float, float>, RefConst<Shape>> box_cache;
    std::map<float, RefConst<Shape>> sphere_cache;
    BodyInterface& bi = w->system->GetBodyInterfaceNoLock();
    w->bodies.resize(nb);
    w->is_dynamic.resize(nb);
    std::vector<BodyID> ids(nb);
    for (uint32_t i = 0; i < nb; i++) {
        const BenchJBody& b = in_bodies[i];
        RefConst<Shape> shape;
        switch (b.shape) {
            case 0: {
                auto key = std::make_tuple(b.half[0], b.half[1], b.half[2]);
                auto it = box_cache.find(key);
                if (it != box_cache.end()) {
                    shape = it->second;
                } else {
                    float r = radius_for(std::min(b.half[0], std::min(b.half[1], b.half[2])));
                    BoxShapeSettings bs(v3(b.half), r);
                    ShapeSettings::ShapeResult res = bs.Create();
                    if (res.HasError()) BENCH_FAIL("body %u: BoxShape: %s", i, res.GetError().c_str());
                    shape = res.Get();
                    box_cache.emplace(key, shape);
                    float used = static_cast<const BoxShape*>(shape.GetPtr())->GetConvexRadius();
                    info->box_radius_min = std::min(info->box_radius_min, used);
                    info->box_radius_max = std::max(info->box_radius_max, used);
                }
                info->boxes++;
                break;
            }
            case 1: {
                auto it = sphere_cache.find(b.half[0]);
                if (it != sphere_cache.end()) {
                    shape = it->second;
                } else {
                    shape = new SphereShape(b.half[0]);
                    sphere_cache.emplace(b.half[0], shape);
                }
                info->spheres++;
                break;
            }
            case 2: {
                if (b.hull_first + b.hull_count > hulls->n_hulls || b.hull_count == 0) BENCH_FAIL("body %u: hull range out of bounds", i);
                std::vector<RefConst<Shape>> parts;
                for (uint32_t h = b.hull_first; h < b.hull_first + b.hull_count; h++) {
                    uint32_t first = hulls->first[h], count = hulls->count[h];
                    if (first + count > hulls->n_vertices) BENCH_FAIL("hull %u: vertex range out of bounds", h);
                    Array<Vec3> pts;
                    pts.reserve(count);
                    Vec3 lo = Vec3::sReplicate(FLT_MAX), hi = Vec3::sReplicate(-FLT_MAX);
                    for (uint32_t v = 0; v < count; v++) {
                        Vec3 p = v3(hulls->vertices + 3 * size_t(first + v));
                        pts.push_back(p);
                        lo = Vec3::sMin(lo, p);
                        hi = Vec3::sMax(hi, p);
                    }
                    float r = radius_for(0.5f * (hi - lo).ReduceMin());
                    ConvexHullShapeSettings hs(pts, r);
                    ShapeSettings::ShapeResult res = hs.Create();
                    if (res.HasError()) BENCH_FAIL("hull %u (body %u): ConvexHullShape: %s", h, i, res.GetError().c_str());
                    const ConvexHullShape* ch = static_cast<const ConvexHullShape*>(res.Get().GetPtr());
                    info->hulls_built++;
                    info->hull_vertices_in += count;
                    info->hull_vertices_out += ch->GetNumPoints();
                    if (ch->GetNumPoints() != count) info->hulls_changed++;
                    info->hull_radius_min = std::min(info->hull_radius_min, ch->GetConvexRadius());
                    info->hull_radius_max = std::max(info->hull_radius_max, ch->GetConvexRadius());
                    // Planes are in the hull's own COM-relative space
                    Vec3 com = ch->GetCenterOfMass();
                    for (const Vec3& p : pts) {
                        float worst = -FLT_MAX;
                        for (const Plane& pl : ch->GetPlanes()) worst = std::max(worst, pl.SignedDistance(p - com));
                        info->hull_max_outside = std::max(info->hull_max_outside, worst);
                    }
                    parts.push_back(res.Get());
                }
                RefConst<Shape> inner;
                if (parts.size() == 1) {
                    inner = parts[0];
                } else {
                    StaticCompoundShapeSettings cs;
                    for (auto& p : parts) cs.AddShape(Vec3::sZero(), Quat::sIdentity(), p);
                    ShapeSettings::ShapeResult res = cs.Create();
                    if (res.HasError()) BENCH_FAIL("body %u: StaticCompoundShape: %s", i, res.GetError().c_str());
                    inner = res.Get();
                    info->compounds++;
                }
                // Jolt places the body at its shape's centre of mass (computed from the hull
                // volumes). The scene's body frame IS the centre of mass, so shift it back to 0.
                Vec3 com = inner->GetCenterOfMass();
                info->max_com_shift = std::max(info->max_com_shift, com.Length());
                OffsetCenterOfMassShapeSettings os(-com, inner);
                ShapeSettings::ShapeResult res = os.Create();
                if (res.HasError()) BENCH_FAIL("body %u: OffsetCenterOfMassShape: %s", i, res.GetError().c_str());
                shape = res.Get();
                break;
            }
            case 3: {
                if (!b.is_static) BENCH_FAIL("body %u: a plane must be static", i);
                shape = new PlaneShape(Plane(Vec3::sAxisY(), 0.0f), nullptr, s->plane_half_extent);
                info->planes++;
                break;
            }
            default:
                BENCH_FAIL("body %u: unknown shape %u", i, b.shape);
        }
        bool dyn = !b.is_static;
        BodyCreationSettings bcs(shape, RVec3(v3(b.position)), q4(b.rotation), dyn ? EMotionType::Dynamic : EMotionType::Static,
                                 dyn ? Layers::MOVING : Layers::NON_MOVING);
        bcs.mFriction = b.friction;
        bcs.mRestitution = b.restitution;
        if (dyn) {
            bcs.mMotionQuality = EMotionQuality::Discrete;
            bcs.mAllowSleeping = s->sleep != 0;
            bcs.mLinearDamping = 0.0f;
            bcs.mAngularDamping = 0.0f;
            bcs.mMaxLinearVelocity = s->max_linear_velocity;
            bcs.mMaxAngularVelocity = s->max_angular_velocity;
            bcs.mGravityFactor = 1.0f;
            bcs.mLinearVelocity = v3(b.linvel);
            bcs.mAngularVelocity = v3(b.angvel);
            bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
            bcs.mMassPropertiesOverride.mMass = b.mass;
            bcs.mMassPropertiesOverride.mInertia = Mat44::sScale(v3(b.inertia));
        }
        if (w->filter->offsets[i + 1] > w->filter->offsets[i])
            bcs.mCollisionGroup = CollisionGroup(w->filter, CollisionGroup::GroupID(i), 0);
        Body* body = bi.CreateBody(bcs);
        if (!body) BENCH_FAIL("CreateBody failed at body %u (max bodies %u)", i, s->max_bodies);
        w->bodies[i] = body;
        w->is_dynamic[i] = dyn;
        ids[i] = body->GetID();
    }
    info->box_shapes_unique = uint32_t(box_cache.size());
    if (info->box_radius_min == FLT_MAX) info->box_radius_min = 0;
    if (info->hull_radius_min == FLT_MAX) info->hull_radius_min = 0;
    {
        // Add in batches; dynamic bodies start active
        const uint32_t batch = 1 << 16;
        for (uint32_t i = 0; i < nb; i += batch) {
            int k = int(std::min(batch, nb - i));
            BodyInterface::AddState st = bi.AddBodiesPrepare(ids.data() + i, k);
            bi.AddBodiesFinalize(ids.data() + i, k, st, EActivation::Activate);
        }
    }

    // Joints
    std::vector<Constraint*> cons;
    cons.reserve(nj);
    for (uint32_t j = 0; j < nj; j++) {
        const BenchJJoint& jt = joints[j];
        Body& A = *w->bodies[jt.body_a];
        Body& B = *w->bodies[jt.body_b];
        if (jt.kind == 0) {
            PointConstraintSettings pc;
            pc.mSpace = EConstraintSpace::LocalToBodyCOM;
            pc.mPoint1 = RVec3(v3(jt.anchor_a));
            pc.mPoint2 = RVec3(v3(jt.anchor_b));
            cons.push_back(pc.Create(A, B));
            info->n_point++;
            continue;
        }
        if (jt.kind != 1) BENCH_FAIL("joint %u: unknown kind %u", j, jt.kind);
        uint32_t m[6];
        for (int d = 0; d < 6; d++) m[d] = (jt.motion >> (2 * d)) & 3;
        Quat qa = q4(jt.frame_a), qb = q4(jt.frame_b);
        Vec3 xa = qa * Vec3::sAxisX(), ya = qa * Vec3::sAxisY(), xb = qb * Vec3::sAxisX(), yb = qb * Vec3::sAxisY();
        bool lin_locked = m[0] == 0 && m[1] == 0 && m[2] == 0;
        bool has_drive = jt.drive_dof <= 3;
        const float PI = JPH_PI;
        if (s->joint_mode == 0 && lin_locked && m[4] == 0 && m[5] == 0 && m[3] != 0 && (!has_drive || jt.drive_dof == 3) &&
            (m[3] == 2 || (jt.twist[0] <= 0.0f && jt.twist[1] >= 0.0f && jt.twist[0] >= -PI && jt.twist[1] <= PI))) {
            // Hinge: twist (frame X) is the only rotational DOF
            HingeConstraintSettings hs;
            hs.mSpace = EConstraintSpace::LocalToBodyCOM;
            hs.mPoint1 = RVec3(v3(jt.anchor_a));
            hs.mPoint2 = RVec3(v3(jt.anchor_b));
            hs.mHingeAxis1 = xa;
            hs.mNormalAxis1 = ya;
            hs.mHingeAxis2 = xb;
            hs.mNormalAxis2 = yb;
            if (m[3] == 1) {
                hs.mLimitsMin = jt.twist[0];
                hs.mLimitsMax = jt.twist[1];
            }
            if (has_drive) {
                hs.mMotorSettings.mMinTorqueLimit = -jt.drive_force_limit;
                hs.mMotorSettings.mMaxTorqueLimit = jt.drive_force_limit;
                if (jt.drive_stiffness > 0.0f)
                    hs.mMotorSettings.mSpringSettings = SpringSettings(ESpringMode::StiffnessAndDamping, jt.drive_stiffness, jt.drive_damping);
            }
            HingeConstraint* h = static_cast<HingeConstraint*>(hs.Create(A, B));
            if (has_drive) {
                if (jt.drive_stiffness > 0.0f) {
                    h->SetMotorState(EMotorState::Position);
                    h->SetTargetAngle(jt.drive_target);
                } else {
                    h->SetMotorState(EMotorState::Velocity);
                    h->SetTargetAngularVelocity(jt.drive_velocity);
                }
                info->n_motors++;
            }
            cons.push_back(h);
            info->n_hinge++;
            continue;
        }
        if (s->joint_mode == 0 && lin_locked && !has_drive && (jt.cone || (m[4] != 1 && m[5] != 1)) && (m[4] != 0 || m[5] != 0) &&
            (m[3] != 1 || (jt.twist[0] >= -PI && jt.twist[1] <= PI))) {
            // Swing-twist (Jolt's ragdoll joint): elliptical swing cone + twist range
            SwingTwistConstraintSettings st;
            st.mSpace = EConstraintSpace::LocalToBodyCOM;
            st.mPosition1 = RVec3(v3(jt.anchor_a));
            st.mPosition2 = RVec3(v3(jt.anchor_b));
            st.mTwistAxis1 = xa;
            st.mPlaneAxis1 = ya;
            st.mTwistAxis2 = xb;
            st.mPlaneAxis2 = yb;
            st.mSwingType = ESwingType::Cone;
            auto half = [&](uint32_t mm, float lim) { return mm == 2 ? PI : (mm == 1 ? lim : 0.0f); };
            st.mPlaneHalfConeAngle = half(m[4], jt.swing[0]);   // swing about frame Y
            st.mNormalHalfConeAngle = half(m[5], jt.swing[1]);  // swing about frame Z
            st.mTwistMinAngle = m[3] == 2 ? -PI : (m[3] == 1 ? jt.twist[0] : 0.0f);
            st.mTwistMaxAngle = m[3] == 2 ? PI : (m[3] == 1 ? jt.twist[1] : 0.0f);
            cons.push_back(st.Create(A, B));
            info->n_swing_twist++;
            continue;
        }
        // General case: 6DOF
        SixDOFConstraintSettings sd;
        using EA = SixDOFConstraintSettings::EAxis;
        sd.mSpace = EConstraintSpace::LocalToBodyCOM;
        sd.mPosition1 = RVec3(v3(jt.anchor_a));
        sd.mPosition2 = RVec3(v3(jt.anchor_b));
        sd.mAxisX1 = xa;
        sd.mAxisY1 = ya;
        sd.mAxisX2 = xb;
        sd.mAxisY2 = yb;
        sd.mSwingType = jt.cone ? ESwingType::Cone : ESwingType::Pyramid;
        for (int d = 0; d < 6; d++) {
            EA ax = EA(d);
            if (m[d] == 0) {
                sd.MakeFixedAxis(ax);
            } else if (m[d] == 2) {
                sd.MakeFreeAxis(ax);
            } else if (m[d] == 1) {
                if (d == 0) sd.SetLimitedAxis(ax, jt.linear_x[0], jt.linear_x[1]);
                else if (d == 3) sd.SetLimitedAxis(ax, jt.twist[0], jt.twist[1]);
                else if (d == 4) sd.SetLimitedAxis(ax, -jt.swing[0], jt.swing[0]);
                else if (d == 5) sd.SetLimitedAxis(ax, -jt.swing[1], jt.swing[1]);
                else BENCH_FAIL("joint %u: limited linear Y/Z motion is not in the scene format", j);
            } else {
                BENCH_FAIL("joint %u: bad motion", j);
            }
        }
        EA drive_axis = EA::Num;
        if (has_drive) {
            drive_axis = jt.drive_dof < 3 ? EA(EA::TranslationX + jt.drive_dof) : EA::RotationX;
            MotorSettings& ms = sd.mMotorSettings[drive_axis];
            if (drive_axis < EA::NumTranslation) {
                ms.mMinForceLimit = -jt.drive_force_limit;
                ms.mMaxForceLimit = jt.drive_force_limit;
            } else {
                ms.mMinTorqueLimit = -jt.drive_force_limit;
                ms.mMaxTorqueLimit = jt.drive_force_limit;
            }
            if (jt.drive_stiffness > 0.0f) ms.mSpringSettings = SpringSettings(ESpringMode::StiffnessAndDamping, jt.drive_stiffness, jt.drive_damping);
        }
        SixDOFConstraint* c = static_cast<SixDOFConstraint*>(sd.Create(A, B));
        if (has_drive) {
            bool servo = jt.drive_stiffness > 0.0f;
            c->SetMotorState(drive_axis, servo ? EMotorState::Position : EMotorState::Velocity);
            if (drive_axis < EA::NumTranslation) {
                Vec3 v = Vec3::sZero();
                v.SetComponent(int(drive_axis), servo ? jt.drive_target : jt.drive_velocity);
                if (servo) c->SetTargetPositionCS(v);
                else c->SetTargetVelocityCS(v);
            } else if (servo) {
                c->SetTargetOrientationCS(Quat::sRotation(Vec3::sAxisX(), jt.drive_target));
            } else {
                c->SetTargetAngularVelocityCS(Vec3(jt.drive_velocity, 0, 0));
            }
            info->n_motors++;
        }
        cons.push_back(c);
        info->n_six_dof++;
    }
    if (!cons.empty()) w->system->AddConstraints(cons.data(), int(cons.size()));
    w->n_constraints = uint32_t(cons.size());

    w->system->OptimizeBroadPhase();
    return w;
#undef BENCH_FAIL
}

// Returns the EPhysicsUpdateError bits of this step (0 = fine).
uint32_t bench_jolt_step(void* handle) {
    World* w = static_cast<World*>(handle);
    EPhysicsUpdateError e = w->system->Update(w->dt, 1, w->temp, w->jobs);
    uint32_t bits = uint32_t(e);
    w->errors_last = bits;
    w->errors_sticky |= bits;
    w->steps++;
    return bits;
}

// out: n * 13 floats (pos3, quat4 xyzw, linvel3, angvel3), scene order. Position = centre of mass.
void bench_jolt_read_state(void* handle, float* out) {
    World* w = static_cast<World*>(handle);
    for (uint32_t i = 0; i < w->n; i++) {
        float* o = out + 13 * size_t(i);
        const Body* b = w->bodies[i];
        RVec3 p = b->GetCenterOfMassPosition();
        Quat q = b->GetRotation();
        o[0] = float(p.GetX()); o[1] = float(p.GetY()); o[2] = float(p.GetZ());
        o[3] = q.GetX(); o[4] = q.GetY(); o[5] = q.GetZ(); o[6] = q.GetW();
        if (w->is_dynamic[i]) {
            Vec3 v = b->GetLinearVelocity(), a = b->GetAngularVelocity();
            o[7] = v.GetX(); o[8] = v.GetY(); o[9] = v.GetZ();
            o[10] = a.GetX(); o[11] = a.GetY(); o[12] = a.GetZ();
        } else {
            for (int k = 7; k < 13; k++) o[k] = 0.0f;
        }
    }
}

// Element types of the enumerators are private typedefs; deduce them instead of naming them.
extern "C++" {
template <class C, class BP, class CBP, class MK>
void count_cache(const C& cache, void (C::*all_pairs)(Array<const BP*>&) const, void (C::*all_manifolds)(const CBP&, Array<const MK*>&) const,
                 uint32_t& n_pairs, uint32_t& n_manifolds) {
    Array<const BP*> pairs;
    (cache.*all_pairs)(pairs);
    n_pairs = uint32_t(pairs.size());
    Array<const MK*> m;
    for (const BP* bp : pairs) {
        m.clear();
        (cache.*all_manifolds)(bp->GetValue(), m);
        n_manifolds += uint32_t(m.size());
    }
}
}  // extern "C++"

void bench_jolt_counters(void* handle, BenchJCounters* c) {
    World* w = static_cast<World*>(handle);
    std::memset(c, 0, sizeof(*c));
    c->bodies = w->system->GetNumBodies();
    c->active_bodies = w->system->GetNumActiveBodies(EBodyType::RigidBody);
    ContactConstraintManager& cm = (*w->system).*bench_get(BenchCMTag{});
    int write_idx = cm.*bench_get(BenchIdxTag{});
    auto& caches = cm.*bench_get(BenchCacheTag{});
    // After Update() the write index has flipped: the last step's cache is the other one
    auto& last = caches[write_idx ^ 1];
    // Public (untimed, sorting) enumerators: body pairs with a cached manifold, and their manifolds
    count_cache(last, &std::remove_reference_t<decltype(last)>::GetAllBodyPairsSorted,
                &std::remove_reference_t<decltype(last)>::GetAllManifoldsSorted, c->body_pairs, c->manifolds);
    c->constraints = w->n_constraints;
    c->errors_last = w->errors_last;
    c->errors_sticky = w->errors_sticky;
    c->steps = w->steps;
}

void bench_jolt_destroy(void* handle) { delete static_cast<World*>(handle); }

}  // extern "C"
