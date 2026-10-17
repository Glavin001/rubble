// Thin C API over the public PhysX 5 SDK for the benchmark adapter (src/main.rs drives it
// over FFI). Written against the public headers only, so the same file builds against
//   * PhysX 5.1.3 as vendored by the physx-sys crate (CPU only, macOS/Linux/Windows), and
//   * the official NVIDIA-Omniverse/PhysX SDK 5.x (CPU, or CUDA with -DBENCH_PHYSX_GPU).
//
// One world per process (PhysX allows a single PxFoundation).
//
// The struct layouts below must match the #[repr(C)] mirrors in src/main.rs.

#ifdef BENCH_PHYSX_FORCE_GPU_DECLS
// Compile-check aid only: expose the CUDA declarations on platforms where PhysX hides them
// (macOS), so `-DBENCH_PHYSX_GPU -fsyntax-only` can type-check the GPU path anywhere.
#include "foundation/PxPreprocessor.h"
#undef PX_SUPPORT_GPU_PHYSX
#define PX_SUPPORT_GPU_PHYSX 1
#endif

#include "PxPhysicsAPI.h"
#ifdef BENCH_PHYSX_GPU
#include "cudamanager/PxCudaContextManager.h"
#include "gpu/PxGpu.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

using namespace physx;

extern "C" {

struct BenchPxBody {
    uint32_t shape;      // 0 box, 1 sphere, 2 convex hulls, 3 plane (static half-space)
    uint32_t is_static;  // 1 = PxRigidStatic
    float half[3];       // box half extents, or [radius, 0, 0] (sphere; hulls: bounding radius)
    uint32_t hull_first; // hulls: BenchPxHulls entries hull_first .. hull_first + hull_count
    uint32_t hull_count;
    float mass;
    float inertia[3];    // principal moments, body frame
    float friction;
    float restitution;
    float position[3];
    float rotation[4];   // x y z w
    float linvel[3];
    float angvel[3];
};

struct BenchPxJoint {
    uint32_t body_a;
    uint32_t body_b;
    uint32_t collide;
    float anchor_a[3];
    float anchor_b[3];
    // Scene format v3 (kind 1 = D6; kind 0 keeps PxSphericalJoint)
    uint32_t kind;
    float frame_a[4];          // x y z w
    float frame_b[4];
    uint32_t motion;           // 2 bits per DOF X Y Z twist swing1 swing2: 0 locked 1 limited 2 free
    float twist[2];
    float swing[2];            // half-angles about frame Y, Z
    uint32_t cone;             // limited swings: 1 elliptical cone, 0 pyramid
    float linear_x[2];
    uint32_t drive_dof;        // 0..2 linear, 3 twist, ~0 none
    float drive_velocity;
    float drive_force_limit;
    float drive_stiffness;     // 0: velocity motor
    float drive_damping;
    float drive_target;
};

// Convex hulls (scene format v2): hull h is vertices first[h] .. first[h] + count[h] (xyz,
// in the owning body's frame).
struct BenchPxHulls {
    uint32_t n_hulls;
    const uint32_t* first;
    const uint32_t* count;
    uint32_t n_vertices;
    const float* vertices;
};

struct BenchPxSettings {
    float gravity[3];
    float dt;
    uint32_t threads;          // worker threads of PxDefaultCpuDispatcher
    uint32_t solver;           // 0 TGS, 1 PGS
    uint32_t pos_iters;
    uint32_t vel_iters;
    uint32_t broadphase;       // 0 ABP, 1 SAP, 2 MBP, 3 PABP, 4 GPU
    uint32_t sleep;            // 0 = never sleep, 1 = PhysX defaults
    uint32_t dispatch_mode;    // 0 wait-for-work, 1 yield-thread, 2 yield-processor
    float max_angular_velocity;
    float max_depenetration_velocity;  // <= 0: PhysX default
    uint32_t gpu;              // 1 = GPU dynamics (needs BENCH_PHYSX_GPU)
    int32_t gpu_device;        // CUDA ordinal, -1 = PhysX's choice
    uint32_t gpu_max_partitions;
    uint32_t gpu_max_rigid_contacts;
    uint32_t gpu_max_rigid_patches;
    uint32_t gpu_found_lost_pairs;
    uint32_t gpu_found_lost_aggregate_pairs;
    uint32_t gpu_total_aggregate_pairs;
    uint32_t gpu_heap_capacity;
    uint32_t gpu_collision_stack;
    uint64_t gpu_temp_buffer;
};

struct BenchPxInfo {
    uint32_t version_major;
    uint32_t version_minor;
    uint32_t version_bugfix;
    uint32_t gpu_active;       // scene really runs GPU dynamics
    uint32_t broadphase;       // as reported by the scene, same coding as settings
    uint32_t solver;           // as reported by the scene
    float contact_offset;
    float rest_offset;
    float max_depenetration_velocity;
    float max_angular_velocity;
    float max_linear_velocity;
    float sleep_threshold;
    float wake_counter_reset;
    float bounce_threshold;
    float friction_offset_threshold;
    float friction_correlation_distance;
    float tolerance_length;
    float tolerance_speed;
    char device[128];
    uint32_t hulls_cooked;
    uint32_t hull_vertices_in;
    uint32_t hull_vertices_out;
    uint32_t hulls_vertex_count_changed;  // cooked vertex count != input vertex count
    uint32_t hull_polygons;
    uint32_t hull_max_polygons;
    uint32_t hulls_gpu_incompatible;      // PxConvexMesh::isGpuCompatible() false
    uint32_t hulls_new_vertices;          // hulls with a cooked vertex that is no input point
    float hull_max_outside;               // farthest an input point lies outside its cooked hull
    uint32_t cook_gpu_flag;               // PxConvexFlag::eGPU_COMPATIBLE set
    uint32_t cook_build_gpu_data;         // PxCookingParams::buildGPUData
    uint32_t cook_vertex_limit;
    uint32_t cook_polygon_limit;
    uint32_t planes;
};

struct BenchPxCounters {
    uint32_t awake;                // dynamic actors with isSleeping() == false
    uint32_t active_dynamic;       // PxSimulationStatistics::nbActiveDynamicBodies
    uint32_t active_constraints;
    uint32_t axis_constraints;
    uint32_t contact_pairs;        // nbDiscreteContactPairsTotal
    uint32_t contact_pairs_touching;
    uint32_t contact_pairs_cache_hits;
    uint32_t new_pairs;
    uint32_t lost_pairs;
    uint32_t new_touches;
    uint32_t lost_touches;
    uint32_t partitions;
    uint32_t dynamic_bodies;
    uint32_t static_bodies;
    uint32_t errors;               // PhysX errors/warnings reported so far
};

}  // extern "C"

namespace {

class BenchErrorCallback : public PxErrorCallback {
public:
    uint32_t count = 0;
    char last[512] = {0};
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override {
        if (code == PxErrorCode::eDEBUG_INFO) return;
        count++;
        std::snprintf(last, sizeof(last), "PhysX error %d: %s (%s:%d)", int(code), message ? message : "", file ? file : "?", line);
        std::fprintf(stderr, "%s\n", last);
    }
};

struct World {
    PxDefaultAllocator allocator;
    BenchErrorCallback errors;
    PxFoundation* foundation = nullptr;
    PxPhysics* physics = nullptr;
    bool extensions = false;
    PxDefaultCpuDispatcher* dispatcher = nullptr;
    PxScene* scene = nullptr;
#ifdef BENCH_PHYSX_GPU
    PxCudaContextManager* cuda = nullptr;
#endif
    PxRigidActor** actors = nullptr;   // scene order
    uint8_t* is_dynamic = nullptr;
    uint32_t n = 0;
    uint32_t n_dynamic = 0;
    uint32_t n_static = 0;

    void release() {
        if (scene) scene->release();
        scene = nullptr;
        if (dispatcher) dispatcher->release();
        dispatcher = nullptr;
#ifdef BENCH_PHYSX_GPU
        if (cuda) cuda->release();
        cuda = nullptr;
#endif
        if (extensions) PxCloseExtensions();
        extensions = false;
        if (physics) physics->release();
        physics = nullptr;
        if (foundation) foundation->release();
        foundation = nullptr;
        delete[] actors;
        actors = nullptr;
        delete[] is_dynamic;
        is_dynamic = nullptr;
    }
};

uint32_t encode_bp(PxBroadPhaseType::Enum t) {
    switch (t) {
        case PxBroadPhaseType::eABP: return 0;
        case PxBroadPhaseType::eSAP: return 1;
        case PxBroadPhaseType::eMBP: return 2;
        case PxBroadPhaseType::ePABP: return 3;
        case PxBroadPhaseType::eGPU: return 4;
        default: return 99;
    }
}

#ifdef BENCH_PHYSX_GPU
// PxCudaContextManagerDesc::deviceOrdinal exists in newer SDKs only (not in 5.1).
template <typename D>
auto set_device_ordinal(D& d, int ordinal, int) -> decltype(d.deviceOrdinal = ordinal, true) {
    d.deviceOrdinal = ordinal;
    return true;
}
template <typename D>
bool set_device_ordinal(D&, int, long) {
    return false;
}
#endif

// PxConvexFlag::eGPU_COMPATIBLE exists up to PhysX 5.3 (5.1 vendored); later SDKs dropped it
// (buildGPUData + the 64-vertex/64-polygon limits are what make a hull GPU compatible there).
template <typename F>
auto gpu_compatible_flag(int) -> decltype(void(F::eGPU_COMPATIBLE), PxConvexFlags()) {
    return PxConvexFlags(F::eGPU_COMPATIBLE);
}
template <typename F>
PxConvexFlags gpu_compatible_flag(long) {
    return PxConvexFlags();
}

void fail(char* err, uint32_t errlen, const char* msg) {
    if (err && errlen) {
        std::snprintf(err, errlen, "%s", msg);
    }
}

}  // namespace

extern "C" {

void* bench_px_create(const BenchPxBody* bodies, uint32_t nb, const BenchPxJoint* joints, uint32_t nj,
                      const BenchPxHulls* hulls, const BenchPxSettings* s, BenchPxInfo* info, char* err, uint32_t errlen) {
    World* w = new (std::nothrow) World();
    if (!w) {
        fail(err, errlen, "out of memory");
        return nullptr;
    }
    std::memset(info, 0, sizeof(*info));
    info->version_major = PX_PHYSICS_VERSION_MAJOR;
    info->version_minor = PX_PHYSICS_VERSION_MINOR;
    info->version_bugfix = PX_PHYSICS_VERSION_BUGFIX;

#define BENCH_FAIL(msg)                                                                     \
    do {                                                                                    \
        char buf_[768];                                                                     \
        std::snprintf(buf_, sizeof(buf_), "%s%s%s", msg, w->errors.count ? "; last: " : "", \
                      w->errors.count ? w->errors.last : "");                               \
        fail(err, errlen, buf_);                                                            \
        w->release();                                                                       \
        delete w;                                                                           \
        return nullptr;                                                                     \
    } while (0)

    w->foundation = PxCreateFoundation(PX_PHYSICS_VERSION, w->allocator, w->errors);
    if (!w->foundation) BENCH_FAIL("PxCreateFoundation failed");
    PxTolerancesScale scale;  // length 1, speed 10 (PhysX defaults)
    w->physics = PxCreatePhysics(PX_PHYSICS_VERSION, *w->foundation, scale, false, nullptr);
    if (!w->physics) BENCH_FAIL("PxCreatePhysics failed");
    if (!PxInitExtensions(*w->physics, nullptr)) BENCH_FAIL("PxInitExtensions failed");
    w->extensions = true;

    PxDefaultCpuDispatcherWaitForWorkMode::Enum mode = PxDefaultCpuDispatcherWaitForWorkMode::eWAIT_FOR_WORK;
    PxU32 yield_count = 0;
    if (s->dispatch_mode == 1) mode = PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_THREAD;
    if (s->dispatch_mode == 2) {
        mode = PxDefaultCpuDispatcherWaitForWorkMode::eYIELD_PROCESSOR;
        yield_count = 1;
    }
    w->dispatcher = PxDefaultCpuDispatcherCreate(s->threads, nullptr, mode, yield_count);
    if (!w->dispatcher) BENCH_FAIL("PxDefaultCpuDispatcherCreate failed");

    PxSceneDesc desc(w->physics->getTolerancesScale());
    desc.gravity = PxVec3(s->gravity[0], s->gravity[1], s->gravity[2]);
    desc.cpuDispatcher = w->dispatcher;
    desc.filterShader = PxDefaultSimulationFilterShader;
    desc.solverType = s->solver == 1 ? PxSolverType::ePGS : PxSolverType::eTGS;
    switch (s->broadphase) {
        case 1: desc.broadPhaseType = PxBroadPhaseType::eSAP; break;
        case 2: desc.broadPhaseType = PxBroadPhaseType::eMBP; break;
        case 3: desc.broadPhaseType = PxBroadPhaseType::ePABP; break;
        case 4: desc.broadPhaseType = PxBroadPhaseType::eGPU; break;
        default: desc.broadPhaseType = PxBroadPhaseType::eABP; break;
    }
    // PCM on (PhysX 5 default); CCD, stabilization, enhanced determinism off.
    desc.flags |= PxSceneFlag::eENABLE_PCM;
    desc.flags.clear(PxSceneFlag::eENABLE_CCD);
    desc.flags.clear(PxSceneFlag::eENABLE_STABILIZATION);
    desc.flags.clear(PxSceneFlag::eENABLE_ENHANCED_DETERMINISM);

    if (s->gpu) {
#ifdef BENCH_PHYSX_GPU
        PxCudaContextManagerDesc cdesc;
        if (s->gpu_device >= 0 && !set_device_ordinal(cdesc, s->gpu_device, 0))
            BENCH_FAIL("gpu_device: this PhysX version's PxCudaContextManagerDesc has no deviceOrdinal (use CUDA_VISIBLE_DEVICES)");
        w->cuda = PxCreateCudaContextManager(*w->foundation, cdesc, PxGetProfilerCallback());
        if (!w->cuda) BENCH_FAIL("PxCreateCudaContextManager failed (is libPhysXGpu_64.so / PhysXGpu_64.dll next to the binary or on the library path, and a CUDA driver installed?)");
        if (!w->cuda->contextIsValid()) BENCH_FAIL("CUDA context is not valid (no CUDA device / driver too old?)");
        const char* dn = w->cuda->getDeviceName();
        std::snprintf(info->device, sizeof(info->device), "%s", dn ? dn : "CUDA device");
        desc.cudaContextManager = w->cuda;
        desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        desc.broadPhaseType = PxBroadPhaseType::eGPU;
        desc.gpuMaxNumPartitions = s->gpu_max_partitions;
        desc.gpuDynamicsConfig.maxRigidContactCount = s->gpu_max_rigid_contacts;
        desc.gpuDynamicsConfig.maxRigidPatchCount = s->gpu_max_rigid_patches;
        desc.gpuDynamicsConfig.foundLostPairsCapacity = s->gpu_found_lost_pairs;
        desc.gpuDynamicsConfig.foundLostAggregatePairsCapacity = s->gpu_found_lost_aggregate_pairs;
        desc.gpuDynamicsConfig.totalAggregatePairsCapacity = s->gpu_total_aggregate_pairs;
        desc.gpuDynamicsConfig.heapCapacity = s->gpu_heap_capacity;
        desc.gpuDynamicsConfig.collisionStackSize = s->gpu_collision_stack;
        desc.gpuDynamicsConfig.tempBufferCapacity =
            static_cast<decltype(desc.gpuDynamicsConfig.tempBufferCapacity)>(s->gpu_temp_buffer);  // PxU32 in 5.1, PxU64 later
#else
        BENCH_FAIL("GPU dynamics requested but the shim was built without BENCH_PHYSX_GPU (cargo feature `gpu`)");
#endif
    }
    if (!desc.isValid()) BENCH_FAIL("PxSceneDesc is not valid");
    w->scene = w->physics->createScene(desc);
    if (!w->scene) BENCH_FAIL("createScene failed");
#ifdef BENCH_PHYSX_GPU
    if (s->gpu && !(w->scene->getFlags() & PxSceneFlag::eENABLE_GPU_DYNAMICS))
        BENCH_FAIL("scene fell back to CPU dynamics (GPU dynamics unavailable)");
#endif

    // Materials, one per distinct (friction, restitution); combine mode average.
    const uint32_t max_mats = 64;
    PxMaterial* mats[max_mats];
    float mat_f[max_mats], mat_r[max_mats];
    uint32_t n_mats = 0;
    auto material = [&](float f, float r) -> PxMaterial* {
        for (uint32_t i = 0; i < n_mats; i++)
            if (mat_f[i] == f && mat_r[i] == r) return mats[i];
        if (n_mats == max_mats) return nullptr;
        PxMaterial* m = w->physics->createMaterial(f, f, r);
        if (!m) return nullptr;
        m->setFrictionCombineMode(PxCombineMode::eAVERAGE);
        m->setRestitutionCombineMode(PxCombineMode::eAVERAGE);
        mats[n_mats] = m;
        mat_f[n_mats] = f;
        mat_r[n_mats] = r;
        n_mats++;
        return m;
    };

    // Convex cooking: the scene's hull points (already convex, <= 64 vertices) re-hulled by
    // PhysX's own quickhull. Same limits on CPU and GPU so both cook the same shapes; GPU data
    // only for GPU dynamics.
    PxCookingParams cook(w->physics->getTolerancesScale());
    cook.buildGPUData = s->gpu != 0;
    PxConvexFlags cook_flags = PxConvexFlag::eCOMPUTE_CONVEX;
    const PxConvexFlags gpu_flag = s->gpu ? gpu_compatible_flag<PxConvexFlag>(0) : PxConvexFlags();
    cook_flags |= gpu_flag;
    const PxU16 cook_vertex_limit = 64, cook_polygon_limit = 64;
    info->cook_gpu_flag = bool(gpu_flag) ? 1 : 0;
    info->cook_build_gpu_data = cook.buildGPUData ? 1 : 0;
    info->cook_vertex_limit = cook_vertex_limit;
    info->cook_polygon_limit = cook_polygon_limit;
    PxConvexMesh** meshes = nullptr;
    if (hulls && hulls->n_hulls) {
        meshes = new (std::nothrow) PxConvexMesh*[hulls->n_hulls]();
        if (!meshes) BENCH_FAIL("out of memory");
    }
    // Cooked on first use (only hulls some body references); released after the actors exist
    // (shapes keep their own reference).
    auto release_meshes = [&]() {
        if (!meshes) return;
        for (uint32_t h = 0; h < hulls->n_hulls; h++)
            if (meshes[h]) meshes[h]->release();
        delete[] meshes;
        meshes = nullptr;
    };
    auto convex = [&](uint32_t h) -> PxConvexMesh* {
        if (meshes[h]) return meshes[h];
        if (hulls->first[h] + hulls->count[h] > hulls->n_vertices) return nullptr;
        PxConvexMeshDesc cd;
        cd.points.count = hulls->count[h];
        cd.points.stride = sizeof(float) * 3;
        cd.points.data = hulls->vertices + 3 * size_t(hulls->first[h]);
        cd.flags = cook_flags;
        cd.vertexLimit = cook_vertex_limit;
        cd.polygonLimit = cook_polygon_limit;
        PxConvexMeshCookingResult::Enum result = PxConvexMeshCookingResult::eSUCCESS;
        PxConvexMesh* m = PxCreateConvexMesh(cook, cd, w->physics->getPhysicsInsertionCallback(), &result);
        if (!m) return nullptr;
        meshes[h] = m;
        info->hulls_cooked++;
        info->hull_vertices_in += hulls->count[h];
        info->hull_vertices_out += m->getNbVertices();
        if (m->getNbVertices() != hulls->count[h]) info->hulls_vertex_count_changed++;
        info->hull_polygons += m->getNbPolygons();
        if (m->getNbPolygons() > info->hull_max_polygons) info->hull_max_polygons = m->getNbPolygons();
        if (s->gpu && !m->isGpuCompatible()) info->hulls_gpu_incompatible++;
        // Did cooking change the shape? Input points outside the cooked planes (it shrank), and
        // cooked vertices that are not input points (vertex-limit slicing moved corners).
        const float* in = hulls->vertices + 3 * size_t(hulls->first[h]);
        const PxU32 nin = hulls->count[h];
        for (PxU32 k = 0; k < m->getNbPolygons(); k++) {
            PxHullPolygon poly;
            if (!m->getPolygonData(k, poly)) continue;
            PxVec3 n(poly.mPlane[0], poly.mPlane[1], poly.mPlane[2]);
            for (PxU32 v = 0; v < nin; v++) {
                float d = n.dot(PxVec3(in[3 * v], in[3 * v + 1], in[3 * v + 2])) + poly.mPlane[3];
                if (d > info->hull_max_outside) info->hull_max_outside = d;
            }
        }
        const PxVec3* out = m->getVertices();
        bool moved = false;
        for (PxU32 k = 0; k < m->getNbVertices() && !moved; k++) {
            float best = PX_MAX_F32;
            for (PxU32 v = 0; v < nin; v++)
                best = PxMin(best, (out[k] - PxVec3(in[3 * v], in[3 * v + 1], in[3 * v + 2])).magnitudeSquared());
            if (best > 1e-12f) moved = true;
        }
        if (moved) info->hulls_new_vertices++;
        return m;
    };
#define BENCH_FAIL_MESHES(msg) \
    do {                       \
        release_meshes();      \
        BENCH_FAIL(msg);       \
    } while (0)

    w->n = nb;
    w->actors = new (std::nothrow) PxRigidActor*[nb];
    w->is_dynamic = new (std::nothrow) uint8_t[nb];
    if (!w->actors || !w->is_dynamic) BENCH_FAIL_MESHES("out of memory");
    PxBounds3 world_bounds = PxBounds3::empty();
    bool first_shape = true;
    bool first_dynamic = true;
    for (uint32_t i = 0; i < nb; i++) {
        const BenchPxBody& b = bodies[i];
        PxQuat q(b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]);
        // Keep the scene's quaternion bit-exact unless it is visibly off unit length.
        if (std::fabs(q.magnitudeSquared() - 1.0f) > 1e-4f) q.normalize();
        PxTransform pose(PxVec3(b.position[0], b.position[1], b.position[2]), q);
        PxMaterial* m = material(b.friction, b.restitution);
        if (!m) BENCH_FAIL_MESHES("too many distinct materials");
        PxRigidActor* actor;
        PxRigidDynamic* dyn = nullptr;
        if (b.is_static) {
            actor = w->physics->createRigidStatic(pose);
        } else {
            dyn = w->physics->createRigidDynamic(pose);
            actor = dyn;
        }
        if (!actor) BENCH_FAIL_MESHES("creating an actor failed");
        PxShape* shape = nullptr;
        float r = 0.0f;
        if (b.shape == 1) {
            shape = PxRigidActorExt::createExclusiveShape(*actor, PxSphereGeometry(b.half[0]), *m);
            r = b.half[0];
        } else if (b.shape == 2) {
            // One convex shape per hull; vertices are already in the body frame (centre of mass,
            // principal axes), so every shape has an identity local pose.
            if (!hulls || b.hull_count == 0 || b.hull_first + b.hull_count > hulls->n_hulls) BENCH_FAIL_MESHES("hull body references hulls out of range");
            for (uint32_t h = b.hull_first; h < b.hull_first + b.hull_count; h++) {
                PxConvexMesh* cm = convex(h);
                if (!cm) BENCH_FAIL_MESHES("PxCreateConvexMesh failed");
                shape = PxRigidActorExt::createExclusiveShape(*actor, PxConvexMeshGeometry(cm), *m);
                if (!shape) break;
            }
            r = b.half[0];
        } else if (b.shape == 3) {
            if (!b.is_static) BENCH_FAIL_MESHES("a plane must be static");
            // Half-space through the body position with normal = body +y, solid below. PhysX
            // planes have normal +x: the shape's local pose is the plane-equation transform
            // expressed in the actor frame (the actor keeps the scene pose, so readback matches).
            PxVec3 n = q.rotate(PxVec3(0.0f, 1.0f, 0.0f));
            PxTransform plane_pose = PxTransformFromPlaneEquation(PxPlane(n, -n.dot(pose.p)));
            shape = PxRigidActorExt::createExclusiveShape(*actor, PxPlaneGeometry(), *m);
            if (shape) shape->setLocalPose(pose.getInverse() * plane_pose);
            info->planes++;
        } else {
            shape = PxRigidActorExt::createExclusiveShape(*actor, PxBoxGeometry(b.half[0], b.half[1], b.half[2]), *m);
            r = std::sqrt(b.half[0] * b.half[0] + b.half[1] * b.half[1] + b.half[2] * b.half[2]);
        }
        if (!shape) BENCH_FAIL_MESHES("creating a shape failed");
        if (first_shape) {
            info->contact_offset = shape->getContactOffset();
            info->rest_offset = shape->getRestOffset();
            first_shape = false;
        }
        if (b.shape != 3) world_bounds.include(PxBounds3::centerExtents(pose.p, PxVec3(r)));
        if (dyn) {
            // Exact scene mass properties (never PxRigidBodyExt::updateMassAndInertia)
            dyn->setMass(b.mass);
            dyn->setMassSpaceInertiaTensor(PxVec3(b.inertia[0], b.inertia[1], b.inertia[2]));
            dyn->setCMassLocalPose(PxTransform(PxIdentity));
            dyn->setLinearDamping(0.0f);
            dyn->setAngularDamping(0.0f);
            dyn->setMaxAngularVelocity(s->max_angular_velocity);
            if (s->max_depenetration_velocity > 0.0f) dyn->setMaxDepenetrationVelocity(s->max_depenetration_velocity);
            dyn->setSolverIterationCounts(s->pos_iters, s->vel_iters);
            dyn->setLinearVelocity(PxVec3(b.linvel[0], b.linvel[1], b.linvel[2]), false);
            dyn->setAngularVelocity(PxVec3(b.angvel[0], b.angvel[1], b.angvel[2]), false);
            if (!s->sleep) dyn->setSleepThreshold(0.0f);
            if (first_dynamic) {
                info->max_depenetration_velocity = dyn->getMaxDepenetrationVelocity();
                info->max_angular_velocity = dyn->getMaxAngularVelocity();
                info->max_linear_velocity = dyn->getMaxLinearVelocity();
                info->sleep_threshold = dyn->getSleepThreshold();
                first_dynamic = false;
            }
            w->n_dynamic++;
        } else {
            w->n_static++;
        }
        w->actors[i] = actor;
        w->is_dynamic[i] = dyn ? 1 : 0;
    }
    release_meshes();
#undef BENCH_FAIL_MESHES

    if (desc.broadPhaseType == PxBroadPhaseType::eMBP) {
        // MBP needs explicit regions: a 4x4 grid over the (padded) scene bounds.
        PxVec3 c = world_bounds.getCenter(), e = world_bounds.getExtents();
        PxBounds3 padded = PxBounds3::centerExtents(c, e * 1.5f + PxVec3(10.0f));
        PxBounds3 regions[256];
        PxU32 nr = PxBroadPhaseExt::createRegionsFromWorldBounds(regions, padded, 4, 1);
        for (PxU32 i = 0; i < nr; i++) {
            PxBroadPhaseRegion region;
            region.mBounds = regions[i];
            region.mUserData = nullptr;
            w->scene->addBroadPhaseRegion(region);
        }
    }

    // Add actors in batches (scene order is kept in w->actors; PhysX's internal order is irrelevant).
    {
        const uint32_t batch = 4096;
        PxActor* buf[batch];
        uint32_t k = 0;
        for (uint32_t i = 0; i < nb; i++) {
            buf[k++] = w->actors[i];
            if (k == batch || i + 1 == nb) {
                if (!w->scene->addActors(buf, k)) BENCH_FAIL("addActors failed");
                k = 0;
            }
        }
    }
    if (!s->sleep) {
        // Belt and braces: with sleepThreshold 0 the wake counter is reset every step anyway.
        for (uint32_t i = 0; i < nb; i++)
            if (w->is_dynamic[i]) static_cast<PxRigidDynamic*>(w->actors[i])->setWakeCounter(1e10f);
    }

    for (uint32_t j = 0; j < nj; j++) {
        const BenchPxJoint& jt = joints[j];
        if (jt.body_a >= nb || jt.body_b >= nb) BENCH_FAIL("joint body index out of range");
        if (jt.kind == 1) {
            // PxD6Joint is PhysX's GPU-prepared joint (every other joint type is prepped on the CPU)
            PxTransform la(PxVec3(jt.anchor_a[0], jt.anchor_a[1], jt.anchor_a[2]), PxQuat(jt.frame_a[0], jt.frame_a[1], jt.frame_a[2], jt.frame_a[3]));
            PxTransform lb(PxVec3(jt.anchor_b[0], jt.anchor_b[1], jt.anchor_b[2]), PxQuat(jt.frame_b[0], jt.frame_b[1], jt.frame_b[2], jt.frame_b[3]));
            PxD6Joint* d6 = PxD6JointCreate(*w->physics, w->actors[jt.body_a], la, w->actors[jt.body_b], lb);
            if (!d6) BENCH_FAIL("PxD6JointCreate failed");
            static const PxD6Axis::Enum axes[6] = {PxD6Axis::eX, PxD6Axis::eY, PxD6Axis::eZ, PxD6Axis::eTWIST, PxD6Axis::eSWING1, PxD6Axis::eSWING2};
            static const PxD6Motion::Enum motions[3] = {PxD6Motion::eLOCKED, PxD6Motion::eLIMITED, PxD6Motion::eFREE};
            for (int d = 0; d < 6; d++) d6->setMotion(axes[d], motions[(jt.motion >> (2 * d)) & 3]);
            const PxTolerancesScale& tol = w->physics->getTolerancesScale();
            if (((jt.motion >> 6) & 3) == 1) d6->setTwistLimit(PxJointAngularLimitPair(jt.twist[0], jt.twist[1]));
            bool s1 = ((jt.motion >> 8) & 3) == 1, s2 = ((jt.motion >> 10) & 3) == 1;
            if (s1 || s2) {
                float y = s1 ? jt.swing[0] : PxPi * 0.999f, z = s2 ? jt.swing[1] : PxPi * 0.999f;
                if (jt.cone) d6->setSwingLimit(PxJointLimitCone(y, z));
                else d6->setPyramidSwingLimit(PxJointLimitPyramid(-y, y, -z, z));
            }
            if ((jt.motion & 3) == 1) d6->setLinearLimit(PxD6Axis::eX, PxJointLinearLimitPair(tol, jt.linear_x[0], jt.linear_x[1]));
            if (jt.drive_dof <= 3) {
                static const PxD6Drive::Enum drives[4] = {PxD6Drive::eX, PxD6Drive::eY, PxD6Drive::eZ, PxD6Drive::eTWIST};
                // A velocity motor is a stiff damper toward the target speed, capped at the force limit
                float damping = jt.drive_stiffness > 0.0f ? jt.drive_damping : 1.0e6f;
                d6->setDrive(drives[jt.drive_dof], PxD6JointDrive(jt.drive_stiffness, damping, jt.drive_force_limit, false));
                PxVec3 lin(0.0f), ang(0.0f);
                if (jt.drive_dof < 3) lin[jt.drive_dof] = jt.drive_velocity; else ang.x = jt.drive_velocity;
                d6->setDriveVelocity(lin, ang);
                if (jt.drive_stiffness > 0.0f) {
                    PxVec3 p(0.0f);
                    if (jt.drive_dof < 3) p[jt.drive_dof] = jt.drive_target;
                    PxQuat q = jt.drive_dof == 3 ? PxQuat(jt.drive_target, PxVec3(1, 0, 0)) : PxQuat(PxIdentity);
                    d6->setDrivePosition(PxTransform(p, q));
                }
            }
            d6->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, jt.collide != 0);
            continue;
        }
        PxSphericalJoint* sj = PxSphericalJointCreate(
            *w->physics, w->actors[jt.body_a], PxTransform(PxVec3(jt.anchor_a[0], jt.anchor_a[1], jt.anchor_a[2])),
            w->actors[jt.body_b], PxTransform(PxVec3(jt.anchor_b[0], jt.anchor_b[1], jt.anchor_b[2])));
        if (!sj) BENCH_FAIL("PxSphericalJointCreate failed");
        // Joined bodies do not collide unless asked (PhysX default: eCOLLISION_ENABLED off).
        sj->setConstraintFlag(PxConstraintFlag::eCOLLISION_ENABLED, jt.collide != 0);
    }

    info->gpu_active = (w->scene->getFlags() & PxSceneFlag::eENABLE_GPU_DYNAMICS) ? 1 : 0;
    info->broadphase = encode_bp(w->scene->getBroadPhaseType());
    info->solver = w->scene->getSolverType() == PxSolverType::ePGS ? 1 : 0;
    info->wake_counter_reset = w->scene->getWakeCounterResetValue();
    info->bounce_threshold = w->scene->getBounceThresholdVelocity();
    info->friction_offset_threshold = desc.frictionOffsetThreshold;
    info->friction_correlation_distance = desc.frictionCorrelationDistance;
    info->tolerance_length = w->physics->getTolerancesScale().length;
    info->tolerance_speed = w->physics->getTolerancesScale().speed;
    return w;
#undef BENCH_FAIL
}

int bench_px_step(void* handle, float dt) {
    World* w = static_cast<World*>(handle);
    if (!w->scene->simulate(dt)) return 1;
    if (!w->scene->fetchResults(true)) return 2;
    return 0;
}

// out: n * 13 floats (pos3, quat4 xyzw, linvel3, angvel3), scene order.
void bench_px_read_state(void* handle, float* out) {
    World* w = static_cast<World*>(handle);
    for (uint32_t i = 0; i < w->n; i++) {
        float* o = out + 13 * size_t(i);
        PxTransform t = w->actors[i]->getGlobalPose();
        o[0] = t.p.x; o[1] = t.p.y; o[2] = t.p.z;
        o[3] = t.q.x; o[4] = t.q.y; o[5] = t.q.z; o[6] = t.q.w;
        if (w->is_dynamic[i]) {
            PxRigidDynamic* d = static_cast<PxRigidDynamic*>(w->actors[i]);
            PxVec3 v = d->getLinearVelocity(), a = d->getAngularVelocity();
            o[7] = v.x; o[8] = v.y; o[9] = v.z;
            o[10] = a.x; o[11] = a.y; o[12] = a.z;
        } else {
            for (int k = 7; k < 13; k++) o[k] = 0.0f;
        }
    }
}

void bench_px_counters(void* handle, BenchPxCounters* c) {
    World* w = static_cast<World*>(handle);
    std::memset(c, 0, sizeof(*c));
    for (uint32_t i = 0; i < w->n; i++)
        if (w->is_dynamic[i] && !static_cast<PxRigidDynamic*>(w->actors[i])->isSleeping()) c->awake++;
    PxSimulationStatistics st;
    w->scene->getSimulationStatistics(st);
    c->active_dynamic = st.nbActiveDynamicBodies;
    c->active_constraints = st.nbActiveConstraints;
    c->axis_constraints = st.nbAxisSolverConstraints;
    c->contact_pairs = st.nbDiscreteContactPairsTotal;
    c->contact_pairs_touching = st.nbDiscreteContactPairsWithContacts;
    c->contact_pairs_cache_hits = st.nbDiscreteContactPairsWithCacheHits;
    c->new_pairs = st.nbNewPairs;
    c->lost_pairs = st.nbLostPairs;
    c->new_touches = st.nbNewTouches;
    c->lost_touches = st.nbLostTouches;
    c->partitions = st.nbPartitions;
    c->dynamic_bodies = w->n_dynamic;
    c->static_bodies = w->n_static;
    c->errors = w->errors.count;
}

void bench_px_destroy(void* handle) {
    World* w = static_cast<World*>(handle);
    if (!w) return;
    w->release();
    delete w;
}

}  // extern "C"
