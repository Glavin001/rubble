// Shared by the stress kernels: chunk and island data, rigid-body math in the
// reference's conventions (stress-ref math.rs), compensated arithmetic.

module common;

public static const uint SUPPORT_NONE = 0;
public static const uint SUPPORT_FIXED = 1;
public static const uint SUPPORT_PINNED = 2;

public static const uint ISLAND_ANCHORED = 1;
public static const uint ISLAND_DRIVEN = 2;
// Island status bits.
public static const uint ISLAND_HALTED = 1;
// Island flag: the island takes part in the per-substep contact pipeline.
public static const uint ISLAND_CONTACT = 4;
// Island flag: the island spreads over many threadgroups (wide kernels).
public static const uint ISLAND_WIDE = 8;
// Island flag: held at quasi-static equilibrium (adaptive and quasi-static modes): it
// moves rigidly only, its chunk state frozen.
public static const uint ISLAND_SETTLED = 16;
// Island status: a settled island woke (its external loads changed) during the batch.
public static const uint ISLAND_WOKE = 4;
// Island flag: settled, anchored, without load terms or contact candidates: nothing can
// wake or move it, so its substeps only keep count (and record its probes).
public static const uint ISLAND_QUIET = 32;
// Island status (settled_fatigue): a bond's damage grew; a bond disconnected.
public static const uint ISLAND_DAMAGED = 16;
public static const uint ISLAND_DISCONNECTED = 32;

// Path coverage (src/coverage.rs mirrors these): which code paths ran, OR-ed into
// `Island.coverage` (kernels without an island: the sentinel island's). For tests only:
// never read by the physics.
public static const uint COV_PAIR_CONTACT = 1u << 0;     // a chunk pair pushed (penalty force)
public static const uint COV_IMPACTOR_CONTACT = 1u << 1; // an impactor touched a chunk
public static const uint COV_GROUND_CONTACT = 1u << 2;   // a chunk touched the ground
public static const uint COV_IMPACTOR_GROUND = 1u << 3;  // an impactor touched the ground
public static const uint COV_CRUSH = 1u << 4;            // an impactor's crush cap acted
public static const uint COV_CRACKED = 1u << 5;          // a bond's first damage (crack event)
public static const uint COV_CREAKED = 1u << 6;          // static fatigue's creak event
public static const uint COV_DISCONNECTED = 1u << 7;     // a bond disconnected
public static const uint COV_WIDE = 1u << 8;             // a wide island ran a substep
public static const uint COV_SETTLED = 1u << 9;          // a settled narrow island moved rigidly
public static const uint COV_SETTLED_WIDE = 1u << 10;    // a settled wide island checked its loads
public static const uint COV_WOKE = 1u << 11;            // a settled island woke on the GPU
public static const uint COV_QUIET = 1u << 12;           // a quiet island kept count only
public static const uint COV_STATICS = 1u << 13;         // a static solve ran (host: from its result)
public static const uint COV_SETTLED_FATIGUE = 1u << 14; // settled bonds' static fatigue ran
public static const uint COV_LOAD_TERMS = 1u << 15;      // a scripted or engine load term was evaluated
public static const uint COV_BLAST = 1u << 16;           // a blast time function was evaluated
public static const uint COV_PULSE = 1u << 17;           // an engine impact pulse was evaluated
public static const uint COV_PROBES = 1u << 18;          // probe items were recorded
public static const uint COV_CONTACT_PIPELINE = 1u << 19; // an island ran in the per-substep pipeline
public static const uint COV_STOP_NOW = 1u << 20;        // the travel check stopped the batch
public static const uint COV_RESUMED_ON_DEVICE = 1u << 21; // a disconnection left its island whole (checked on the GPU)

// Static data of a chunk (body frame of its cluster), with its contact data.
public struct ChunkStatic
{
    public float4 center;     // w: mass
    public float4 inertia0;   // rows of the inertia
    public float4 inertia1;
    public float4 inertia2;
    public float4 inv0;       // rows of the inverse inertia
    public float4 inv1;
    public float4 inv2;
    public float4 scale;      // inertia scale mu, 1 / (m mu), 1 / mu, -
    public uint4 info;        // support, island, structure, chunk
    public uint4 load_range;  // load terms [x, y) in `loads`
    public float4 half;       // half extents; w: bounding radius
    public float4 crot0;      // chunk rotation rows (body frame of the structure)
    public float4 crot1;
    public float4 crot2;
    public float4 cmat;       // contact: E' (scaled), friction, -, -
    public uint4 cinfo;       // contact contributions [x, y) in `index`; z: 1 = in contact; w: 1 = turn checked, 2 = falls
};

// An island (cluster): static mass data and dynamic rigid state.
public struct Island
{
    public uint4 range;       // chunks [x, y), bonds [z, w)
    public uint4 info;        // flags (ISLAND_*), substeps to run, status (ISLAND_HALTED), absolute substep at start
    public float4 com;        // body frame centre of mass; w: mass
    public float4 inertia0;   // rows of the body inertia about com
    public float4 inertia1;
    public float4 inertia2;
    public float4 inv0;       // rows of its inverse
    public float4 inv1;
    public float4 inv2;
    public float4 wcom;       // drift-removal weights (deformation inertia): com, w mass
    public float4 winv0;      // rows of the inverse weighted inertia
    public float4 winv1;
    public float4 winv2;
    public float4 rotation;   // quaternion (x, y, z, w)
    public float4 position;   // world, compensated with position_err
    public float4 position_err;
    public float4 velocity;   // com velocity, compensated with velocity_err
    public float4 velocity_err;
    public float4 angular_velocity;
    // One uint4 on the host (`done`): scalars so that `coverage` has an address (Metal
    // atomics take none of a vector element).
    public uint done;         // substeps run (narrow: in the last dispatch)
    public uint coverage;     // path coverage (COV_*)
    public uint wide_groups;  // wide chunk groups
    public uint done_pad;
    public uint4 probes;      // probe items [x, y) in `loads`
    public float4 energy;     // external work (compensated: x + y); settled: z load norm when it settled, w the change that wakes it
    public float4 rotation_err; // rotation = (rotation + rotation_err): see `turn_left`
    public float4 momentum;     // world angular momentum about the com, compensated with
    public float4 momentum_err; // momentum_err; angular_velocity is derived from it
};

// ------------------------------------------------------------------ rigid math

public struct Quat
{
    public float w;
    public float x;
    public float y;
    public float z;
};

public Quat quat_of(float4 q)
{
    Quat r;
    r.x = q.x;
    r.y = q.y;
    r.z = q.z;
    r.w = q.w;
    return r;
}

public float4 quat_vec(Quat q)
{
    return float4(q.x, q.y, q.z, q.w);
}

public float3 rotate(Quat q, float3 v)
{
    float3 qv = float3(q.x, q.y, q.z);
    float3 t = cross(qv, v) * 2.0;
    return v + t * q.w + cross(qv, t);
}

public float3 inverse_rotate(Quat q, float3 v)
{
    Quat c;
    c.w = q.w;
    c.x = -q.x;
    c.y = -q.y;
    c.z = -q.z;
    return rotate(c, v);
}

public Quat quat_mul(Quat a, Quat o)
{
    Quat r;
    r.w = a.w * o.w - a.x * o.x - a.y * o.y - a.z * o.z;
    r.x = a.w * o.x + a.x * o.w + a.y * o.z - a.z * o.y;
    r.y = a.w * o.y - a.x * o.z + a.y * o.w + a.z * o.x;
    r.z = a.w * o.z + a.x * o.y - a.y * o.x + a.z * o.w;
    return r;
}

public Quat normalized(Quat q)
{
    float n = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    Quat r;
    r.w = q.w / n;
    r.x = q.x / n;
    r.y = q.y / n;
    r.z = q.z / n;
    return r;
}

// Vec3::normalized: zero for a (near-)zero input.
public float3 safe_normalize(float3 v)
{
    float n = length(v);
    return n > 1e-30 ? v / n : float3(0.0);
}

public Quat from_axis_angle(float3 axis, float angle)
{
    float3 a = safe_normalize(axis);
    float s = sin(0.5 * angle);
    float c = cos(0.5 * angle);
    Quat q;
    q.w = c;
    q.x = a.x * s;
    q.y = a.y * s;
    q.z = a.z * s;
    return q;
}

public Quat integrate_rotation(Quat q, float3 omega, float dt)
{
    float angle = length(omega) * dt;
    if (angle < 1e-30)
        return q;
    return normalized(quat_mul(from_axis_angle(omega, angle), q));
}

public float3 rows_mul(float4 r0, float4 r1, float4 r2, float3 v)
{
    return float3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

// (R M R^T) v for a body-frame matrix M given by rows.
public float3 world_mul(Quat q, float4 r0, float4 r1, float4 r2, float3 v)
{
    return rotate(q, rows_mul(r0, r1, r2, inverse_rotate(q, v)));
}

// Compensated quaternions (hi + lo). A substep turns a cluster by ~1e-6 rad: as an f32
// quaternion product and renormalisation that increment is rounded to a few percent,
// and the orientation of a spinning fragment drifts by ~1e-4 rad over 1e4 substeps
// (millimetres at its contacts, which then fire at other times than the reference's).
// Instead the increment (dq - 1) q is formed on its own and added to the pair, which
// keeps the rotation to f32 resolution of the increment; dq is unit, so the pair stays
// normalised to that resolution too.

// (hi, lo) += x as a double-float: the sum's rounding error joins `lo`, then `hi` takes
// what of `lo` it can hold, so `hi` alone stays the rotation to f32 rounding.
public void quat_accumulate(inout Quat hi, inout float4 lo, float4 x)
{
    float4 h = quat_vec(hi);
    float4 sum = h;
    comp_add4(sum, lo, x);
    float4 t = sum + lo;
    lo = lo - (t - sum);
    hi = quat_of(t);
}

// (dq - 1) for a turn by `angle` about the unit `axis`: (axis sin(a/2), cos(a/2) - 1).
public float4 turn_minus_one(float3 axis, float angle)
{
    float s = sin(0.25 * angle);
    return float4(axis * sin(0.5 * angle), -2.0 * s * s);
}

// (hi, lo) = dq (hi + lo), dq the turn by `omega dt` (world frame, `integrate_rotation`).
public void turn_left(inout Quat hi, inout float4 lo, float3 omega, float dt)
{
    float angle = length(omega) * dt;
    if (angle < 1e-30)
        return;
    float4 d = turn_minus_one(omega / length(omega), angle);
    Quat dq;
    dq.x = d.x;
    dq.y = d.y;
    dq.z = d.z;
    dq.w = d.w;
    quat_accumulate(hi, lo, quat_vec(quat_mul(dq, hi)));
}

// (hi, lo) = (hi + lo) dq, dq the turn by `phi` (body frame, as `q * from_axis_angle(phi)`).
public void turn_right(inout Quat hi, inout float4 lo, float3 phi)
{
    float angle = length(phi);
    if (angle < 1e-30)
        return;
    float4 d = turn_minus_one(phi / angle, angle);
    Quat dq;
    dq.x = d.x;
    dq.y = d.y;
    dq.z = d.z;
    dq.w = d.w;
    quat_accumulate(hi, lo, quat_vec(quat_mul(hi, dq)));
}

// R v - v for the turn by `omega dt` (world frame), without the cancellation of the
// difference of two rotated (possibly long) vectors.
public float3 turn_difference(float3 omega, float dt, float3 v)
{
    float angle = length(omega) * dt;
    if (angle < 1e-30)
        return float3(0.0);
    float3 a = omega / length(omega);
    float s = sin(0.5 * angle);
    float3 av = cross(a, v);
    return av * sin(angle) + cross(a, av) * (2.0 * s * s);
}

// Compensated accumulation: (sum, err) += x (Neumaier).
public void comp_add(inout float3 sum, inout float3 err, float3 x)
{
    float3 t = sum + x;
    float3 big = select(abs(sum) >= abs(x), sum, x);
    float3 small = select(abs(sum) >= abs(x), x, sum);
    err += (big - t) + small;
    sum = t;
}

public void comp_add4(inout float4 sum, inout float4 err, float4 x)
{
    float4 t = sum + x;
    float4 big = select(abs(sum) >= abs(x), sum, x);
    float4 small = select(abs(sum) >= abs(x), x, sum);
    err += (big - t) + small;
    sum = t;
}

public void comp_add1(inout float sum, inout float err, float x)
{
    float t = sum + x;
    if (abs(sum) >= abs(x))
        err += (sum - t) + x;
    else
        err += (x - t) + sum;
    sum = t;
}
