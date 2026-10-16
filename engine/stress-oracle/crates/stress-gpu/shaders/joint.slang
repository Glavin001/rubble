// The bond's constitutive law (stress-ref joint.rs `JointModel::evaluate` and
// `secant_factors`) in f32, operation for operation: failure surface (tension cutoff,
// Mohr-Coulomb shear, crushing, buckling), strength multipliers (Weibull, dynamic
// increase factor of the filtered strain rate, static fatigue), damage with
// fracture-energy softening (brittle, ductile), the cracked share's no-tension contact
// patch (6 x 6 springs or exact) with Coulomb sliding and torsional friction, and
// elastic-plastic rebar. Imported by the step kernels.

module joint;

public static const uint KIND_BRITTLE = 0;
public static const uint KIND_DUCTILE = 1;

public static const uint MODE_NONE = 0;
public static const uint MODE_TENSION = 1;
public static const uint MODE_SHEAR = 2;
public static const uint MODE_COMPRESSION = 3;
public static const uint MODE_BUCKLING = 4;

// Material flags.
public static const uint FLAG_SOFTENING = 1;
public static const uint FLAG_CRACK_CONTACT = 2;
public static const uint FLAG_EXACT_PATCH = 4;
public static const uint FLAG_PATCH_AFTER_DAMAGE = 8;
public static const uint FLAG_EXACT_RATE_FILTER = 16;
public static const uint FLAG_DIF = 32;
public static const uint FLAG_FATIGUE = 64;

// Strength of a material (joint.rs `JointStrength`, the per-material part).
public struct JointMaterial
{
    public float4 strength;  // tensile, compressive (inf without crushing), cohesion, friction
    public float4 energy;    // shear cap (inf if none), G_f tension, shear, compression
    public float4 dif;       // reference rate, exponent, transition rate, high exponent
    public float4 misc;      // DIF max, fatigue exponent, fatigue test time, Young's modulus (physical)
    public uint4 kind_flags; // x: KIND_*, y: FLAG_*
};

// Per-bond law data (geometry, stiffness, strength scatter, rebar).
public struct JointBond
{
    public float4 geom0;   // area, width0, width1, torsion modulus
    public float4 geom1;   // s_t1, s_t2, friction radius, Weibull factor
    public float4 stiff0;  // kn, ks, kb_t1, kb_t2
    public float4 stiff1;  // kt, buckling load (0: none), rate filter time, has rebar (0/1)
    public float4 rebar0;  // k_axial, k_dowel, yield force, dowel capacity
    public float4 rebar1;  // rupture work, -, -, -
    public uint4 ids;      // x: material
};

// History of one bond (joint.rs `JointState`).
public struct JointState
{
    public float damage;
    public float crush;
    public float kappa;
    public float kappa_c;
    public float ductility;
    public float ductility_c;
    public float life;          // static fatigue: remaining life 1 - omega (see fatigue_factor)
    public float plastic_x;     // contact sliding offsets and twist
    public float plastic_y;
    public float plastic_t;
    public float rebar_plastic;
    public float rebar_slip0;
    public float rebar_slip1;
    public float rebar_work;
    public float rebar_broken;  // 0 or 1
    public float strain_rate;
    public float governing_stress;
    public float dissipated;
    public float utilization;
    public uint mode;           // MODE_*
};

public struct Measures
{
    public float tension;
    public float shear;
    public float normal_compression;
    public float compression;
    public float compressive_force;
};

public struct JointResponse
{
    public float3 force_lin;
    public float3 force_ang;
    public JointState state;
    public float dissipated;
    public float overshoot;
    public float stored;
    public bool disconnected;
    public float fatigue_before; // fatigue_factor at the evaluation's starting life
    public Measures measures;
};

// Rust `f64::signum`: the sign bit decides (-0 gives -1), never 0.
float signum(float x)
{
    return (asuint(x) & 0x80000000u) != 0 ? -1.0 : 1.0;
}

float infinity()
{
    return asfloat(0x7f800000u);
}

float sq(float x)
{
    return x * x;
}

// Division, square root and power on Metal's fast paths (a few ulp, no IEEE rounding
// sequence): the joint law is a long dependent chain, and its latency sets the substep
// floor. Only here: the compensated position, time and contact-depth arithmetic
// elsewhere keeps IEEE operations. Other targets use the plain operations.
float fdiv(float a, float b)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "metal::fast::divide($0, $1)";
    default:
        return a / b;
    }
}

float fsqrt(float a)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "metal::fast::sqrt($0)";
    default:
        return sqrt(a);
    }
}

float fpow(float a, float b)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "metal::fast::pow($0, $1)";
    default:
        return pow(a, b);
    }
}

public bool connected(JointState st, bool has_rebar)
{
    return st.damage < 1.0 || (has_rebar && st.rebar_broken == 0.0);
}

public bool is_damaged(JointState st)
{
    return st.damage > 0.0 || st.crush > 0.0;
}

public Measures stress_measures(JointBond b, float3 q_lin, float3 q_ang)
{
    float area = b.geom0.x;
    float axial = fdiv(q_lin.z, area);
    float bending = fdiv(abs(q_ang.x), b.geom1.x) + fdiv(abs(q_ang.y), b.geom1.y);
    float shear = fdiv(fsqrt(q_lin.x * q_lin.x + q_lin.y * q_lin.y), area) + fdiv(abs(q_ang.z), b.geom0.w);
    Measures m;
    m.tension = axial + bending;
    m.shear = shear;
    m.normal_compression = max(-axial, 0.0);
    m.compression = -axial + bending;
    m.compressive_force = max(-q_lin.z, 0.0);
    return m;
}

// Failure indices: tension, shear, compression, buckling.
public float4 failure_indices(JointMaterial mat, JointBond b, Measures m, float multiplier)
{
    float ft = mat.strength.x * multiplier;
    float fc = mat.strength.y * multiplier;
    float tau_max = min(mat.strength.z * multiplier + mat.strength.w * m.normal_compression, mat.energy.x * multiplier);
    float4 idx;
    idx.x = max(fdiv(m.tension, ft), 0.0);
    idx.y = tau_max > 0.0 ? fdiv(m.shear, tau_max) : infinity();
    idx.z = max(fdiv(m.compression, fc), 0.0);
    idx.w = b.stiff1.y > 0.0 ? fdiv(m.compressive_force, b.stiff1.y) : 0.0;
    return idx;
}

// Residual strength factor at the remaining life `life` = 1 - omega. The GPU keeps the
// remaining life, not the consumed fraction: near failure omega sits within an ulp of 1
// in f32, where its increments round away and the life never runs out (the joint then
// softens gradually at a residual strength instead of failing as the reference's does).
public float fatigue_factor(JointMaterial mat, float life)
{
    if ((mat.kind_flags.y & FLAG_FATIGUE) == 0)
        return 1.0;
    return fpow(clamp(life, 0.0, 1.0), fdiv(1.0, mat.misc.y - 2.0));
}

float life_rate(JointMaterial mat, float s)
{
    if (s <= 0.0)
        return 0.0;
    return fdiv((mat.misc.y + 1.0) * fpow(s, mat.misc.y), mat.misc.z);
}

float dif_factor(JointMaterial mat, float strain_rate)
{
    float r = abs(strain_rate);
    float ref = mat.dif.x;
    if (r <= ref)
        return 1.0;
    float f;
    if (r <= mat.dif.z)
        f = fpow(fdiv(r, ref), mat.dif.y);
    else
        f = fpow(fdiv(mat.dif.z, ref), mat.dif.y) * fpow(fdiv(r, mat.dif.z), mat.dif.w);
    return clamp(f, 1.0, mat.misc.x);
}

public float damage_law(uint kind, float kappa, float r)
{
    if (kappa <= 1.0)
        return 0.0;
    if (kind == KIND_BRITTLE)
    {
        if (r <= 1.0)
            return 1.0;
        return min(fdiv(r * (kappa - 1.0), kappa * (r - 1.0)), 1.0);
    }
    float kappa_u = 0.5 * (r + 1.0);
    if (kappa >= kappa_u)
        return 1.0;
    return 1.0 - fdiv(1.0, kappa);
}

// (new damage, released energy)
[noinline]
float2 damage_increment(uint kind, float kappa_old, float lambda, float r, float d_old, float psi)
{
    float d_new = max(damage_law(kind, lambda, r), d_old);
    if (d_new <= d_old || d_old >= 1.0)
        return float2(d_old, 0.0);
    float u0 = fdiv(psi, lambda * lambda);
    float k1 = max(kappa_old, 1.0);
    if (kind == KIND_BRITTLE)
    {
        if (r > 1.0)
            return float2(d_new, fdiv(u0 * r, r - 1.0) * max(min(lambda, r) - min(k1, r), 0.0));
        return float2(d_new, (1.0 - d_old) * psi);
    }
    float ku = 0.5 * (r + 1.0);
    float plateau = u0 * max(min(lambda, ku) - min(k1, ku), 0.0);
    float snap = d_new >= 1.0 ? u0 * ku : 0.0;
    return float2(d_new, plateau + snap);
}

// Area, centroid and central second moments of the compressed part of the patch.
void compressed_region(float w0, float w1, float dz, float ax, float ay, out float region[6])
{
    float h0 = 0.5 * w0;
    float h1 = 0.5 * w1;
    float2 rect[4] = { float2(-h0, -h1), float2(h0, -h1), float2(h0, h1), float2(-h0, h1) };
    float2 poly[8];
    uint count = 0;
    for (uint i = 0; i < 4; i++)
    {
        float2 p = rect[i];
        float2 q = rect[(i + 1) % 4];
        float fp = dz + ax * p.y - ay * p.x;
        float fq = dz + ax * q.y - ay * q.x;
        if (fp < 0.0)
            poly[count++] = p;
        if ((fp < 0.0) != (fq < 0.0))
        {
            float t = fp / (fp - fq);
            poly[count++] = float2(p.x + t * (q.x - p.x), p.y + t * (q.y - p.y));
        }
    }
    for (uint k = 0; k < 6; k++)
        region[k] = 0.0;
    if (count < 3)
        return;
    float2 o = poly[0];
    float a = 0.0, sx = 0.0, sy = 0.0, ixx = 0.0, iyy = 0.0, ixy = 0.0;
    for (uint i = 0; i < count; i++)
    {
        float x0 = poly[i].x - o.x;
        float y0 = poly[i].y - o.y;
        float x1 = poly[(i + 1) % count].x - o.x;
        float y1 = poly[(i + 1) % count].y - o.y;
        float cr = x0 * y1 - x1 * y0;
        a += cr / 2.0;
        sx += (x0 + x1) * cr / 6.0;
        sy += (y0 + y1) * cr / 6.0;
        ixx += (x0 * x0 + x0 * x1 + x1 * x1) * cr / 12.0;
        iyy += (y0 * y0 + y0 * y1 + y1 * y1) * cr / 12.0;
        ixy += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cr / 24.0;
    }
    if (a <= 0.0)
        return;
    float cx = sx / a;
    float cy = sy / a;
    region[0] = a;
    region[1] = o.x + cx;
    region[2] = o.y + cy;
    region[3] = ixx - a * cx * cx;
    region[4] = iyy - a * cy * cy;
    region[5] = ixy - a * cx * cy;
}

// Exact no-tension layer: (normal force, moment 1, moment 2, energy).
[noinline]
float4 no_tension_patch(float kn, float w0, float w1, float dz, float ax, float ay)
{
    float r[6];
    compressed_region(w0, w1, dz, ax, ay, r);
    float a = r[0];
    if (a == 0.0)
        return float4(0.0);
    float c1 = r[1], c2 = r[2], i11 = r[3], i22 = r[4], i12 = r[5];
    float k = kn / (w0 * w1);
    float fc = dz + ax * c2 - ay * c1;
    float fs1 = a * fc * c1 + (-ay * i11 + ax * i12);
    float fs2 = a * fc * c2 + (-ay * i12 + ax * i22);
    float n = k * a * fc;
    float energy = 0.5 * k * (a * fc * fc + ay * ay * i11 + ax * ax * i22 - 2.0 * ax * ay * i12);
    return float4(n, k * fs2, -k * fs1, energy);
}

// 1-D return map of an elastic-perfectly-plastic spring: (force, plastic increment).
float2 return_map(float k, float total, float plastic, float cap)
{
    float trial = k * (total - plastic);
    if (abs(trial) <= cap)
        return float2(trial, 0.0);
    float f = cap * signum(trial);
    return float2(f, fdiv(trial - f, k));
}

static const uint CONTACT_SPRINGS = 6;
// Spring positions across the patch, (i + 0.5) / n - 0.5 for n = 6, as constants: the
// divisions in the loop were most of an uncracked bond's evaluation time on Apple GPUs.
static const float SPRING_AT[6] = { -5.0 / 12.0, -0.25, -1.0 / 12.0, 1.0 / 12.0, 0.25, 5.0 / 12.0 };

// The cracked share's contact: force (lin, ang), energy, dissipation; the friction
// offsets after the evaluation in `plastic` (committed by the caller when it should).
struct Contact
{
    float3 q_lin;
    float3 q_ang;
    float energy;
    float diss;
    float3 plastic; // x, y sliding; z twist
};

[noinline]
Contact contact_part(JointMaterial mat, JointBond b, float crush, float3 plastic, float3 d_lin, float3 d_ang)
{
    Contact c;
    c.q_lin = float3(0.0);
    c.q_ang = float3(0.0);
    c.energy = 0.0;
    c.diss = 0.0;
    c.plastic = plastic;
    if ((mat.kind_flags.y & FLAG_CRACK_CONTACT) == 0)
        return c;
    float kn = b.stiff0.x;
    float ks = b.stiff0.y;
    float kt = b.stiff1.x;
    float w0 = b.geom0.y;
    float w1 = b.geom0.z;
    float nc_sum = 0.0, m1 = 0.0, m2 = 0.0, energy = 0.0;
    if ((mat.kind_flags.y & FLAG_EXACT_PATCH) != 0)
    {
        float4 p = no_tension_patch(kn * (1.0 - crush), w0, w1, d_lin.z, d_ang.x, d_ang.y);
        nc_sum = p.x;
        m1 = p.y;
        m2 = p.z;
        energy = p.w;
    }
    else
    {
        float ki = kn * (1.0 - crush) / float(CONTACT_SPRINGS * CONTACT_SPRINGS);
        // The spring closures di = dz + ax s2 - ay s1 are linear over the grid: when the
        // corner springs are all open (or all closed) so is every spring, and the sums
        // have a closed form (sum s = 0, sum s^2 = w^2 (n^2 - 1) / (12 n)). The loop is
        // only for a partly closed patch.
        float n = float(CONTACT_SPRINGS);
        float edge = 0.5 - 0.5 / n;
        float spread = abs(d_ang.x) * edge * w1 + abs(d_ang.y) * edge * w0;
        float slack = 1e-6 * (abs(d_lin.z) + spread);
        if (d_lin.z - spread > slack)
        {
            // Every spring open: no contact force.
        }
        else if (d_lin.z + spread < -slack)
        {
            float q = (n * n - 1.0) / (12.0 * n);
            float i1 = n * q * w0 * w0; // sum over the grid of s1^2
            float i2 = n * q * w1 * w1;
            float area = n * n;
            nc_sum = ki * area * d_lin.z;
            m1 = ki * d_ang.x * i2;
            m2 = ki * d_ang.y * i1;
            energy = 0.5 * ki * (area * d_lin.z * d_lin.z + d_ang.x * d_ang.x * i2 + d_ang.y * d_ang.y * i1);
        }
        else
        for (uint i = 0; i < CONTACT_SPRINGS; i++)
        {
            float s1 = SPRING_AT[i] * w0;
            for (uint j = 0; j < CONTACT_SPRINGS; j++)
            {
                float s2 = SPRING_AT[j] * w1;
                float di = d_lin.z + d_ang.x * s2 - d_ang.y * s1;
                if (di < 0.0)
                {
                    float f = ki * di;
                    nc_sum += f;
                    m1 += f * s2;
                    m2 -= f * s1;
                    energy += 0.5 * ki * di * di;
                }
            }
        }
    }
    float nc = -nc_sum;
    float3 p = plastic;
    c.q_lin = float3(0.0, 0.0, nc_sum);
    c.q_ang = float3(m1, m2, 0.0);
    float diss = 0.0;
    float friction = mat.strength.w;
    // Coulomb sliding (isotropic in the patch plane).
    float slide_cap = friction * nc;
    float2 trial = float2(ks * (d_lin.x - p.x), ks * (d_lin.y - p.y));
    float tn = fsqrt(trial.x * trial.x + trial.y * trial.y);
    if (tn > slide_cap && tn > 0.0)
    {
        float2 dir = float2(fdiv(trial.x, tn), fdiv(trial.y, tn));
        float dslip = fdiv(tn - slide_cap, ks);
        p.x += dir.x * dslip;
        p.y += dir.y * dslip;
        c.q_lin.x = dir.x * slide_cap;
        c.q_lin.y = dir.y * slide_cap;
        diss += slide_cap * dslip;
    }
    else
    {
        c.q_lin.x = trial.x;
        c.q_lin.y = trial.y;
    }
    // Torsional friction.
    float2 tq = return_map(kt, d_ang.z, p.z, friction * nc * b.geom1.z);
    diss += abs(tq.x) * abs(tq.y);
    p.z += tq.y;
    c.q_ang.z = tq.x;
    energy += 0.5 * (fdiv(sq(c.q_lin.x), ks) + fdiv(sq(c.q_lin.y), ks) + fdiv(sq(tq.x), kt));
    c.energy = energy;
    c.diss = diss;
    c.plastic = p;
    return c;
}

// The friction offsets `contact_part` leaves after an evaluation, without the force,
// moments and energy: an uncracked joint (damage 0) weights those by zero, so only the
// offsets of its contact interface evolve. The same operations as `contact_part`, in the
// same order, for the normal force and the two return maps.
float3 contact_offsets(JointMaterial mat, JointBond b, float crush, float3 plastic, float3 d_lin, float3 d_ang)
{
    if ((mat.kind_flags.y & FLAG_CRACK_CONTACT) == 0)
        return plastic;
    float kn = b.stiff0.x;
    float ks = b.stiff0.y;
    float kt = b.stiff1.x;
    float w0 = b.geom0.y;
    float w1 = b.geom0.z;
    float nc_sum = 0.0;
    if ((mat.kind_flags.y & FLAG_EXACT_PATCH) != 0)
    {
        nc_sum = no_tension_patch(kn * (1.0 - crush), w0, w1, d_lin.z, d_ang.x, d_ang.y).x;
    }
    else
    {
        float ki = kn * (1.0 - crush) / float(CONTACT_SPRINGS * CONTACT_SPRINGS);
        float n = float(CONTACT_SPRINGS);
        float edge = 0.5 - 0.5 / n;
        float spread = abs(d_ang.x) * edge * w1 + abs(d_ang.y) * edge * w0;
        float slack = 1e-6 * (abs(d_lin.z) + spread);
        if (d_lin.z - spread > slack)
        {
        }
        else if (d_lin.z + spread < -slack)
        {
            float area = n * n;
            nc_sum = ki * area * d_lin.z;
        }
        else
        for (uint i = 0; i < CONTACT_SPRINGS; i++)
        {
            float s1 = SPRING_AT[i] * w0;
            for (uint j = 0; j < CONTACT_SPRINGS; j++)
            {
                float s2 = SPRING_AT[j] * w1;
                float di = d_lin.z + d_ang.x * s2 - d_ang.y * s1;
                if (di < 0.0)
                    nc_sum += ki * di;
            }
        }
    }
    float nc = -nc_sum;
    float3 p = plastic;
    float friction = mat.strength.w;
    float slide_cap = friction * nc;
    float2 trial = float2(ks * (d_lin.x - p.x), ks * (d_lin.y - p.y));
    float tn = fsqrt(trial.x * trial.x + trial.y * trial.y);
    if (tn > slide_cap && tn > 0.0)
    {
        float2 dir = float2(fdiv(trial.x, tn), fdiv(trial.y, tn));
        float dslip = fdiv(tn - slide_cap, ks);
        p.x += dir.x * dslip;
        p.y += dir.y * dslip;
    }
    float2 tq = return_map(kt, d_ang.z, p.z, friction * nc * b.geom1.z);
    p.z += tq.y;
    return p;
}

float expm1_accurate(float x)
{
    // exp(x) - 1 without cancellation for small x.
    if (abs(x) < 1e-3)
        return x * (1.0 + x * (0.5 + x * (1.0 / 6.0)));
    return exp(x) - 1.0;
}

// joint.rs `JointModel::evaluate`.
public JointResponse joint_evaluate(JointMaterial mat, JointBond b, JointState state, float3 d_lin, float3 d_ang, float dt, bool fracture)
{
    float kn = b.stiff0.x;
    float ks = b.stiff0.y;
    float kb1 = b.stiff0.z;
    float kb2 = b.stiff0.w;
    float kt = b.stiff1.x;
    bool has_rebar = b.stiff1.w != 0.0;
    uint kind = mat.kind_flags.x;
    uint flags = mat.kind_flags.y;
    bool softening = (flags & FLAG_SOFTENING) != 0;
    JointState st = state;
    bool was_connected = connected(st, has_rebar);

    // Effective (undamaged) generalised force and its stress measures.
    float3 qe_lin = d_lin * float3(ks, ks, kn);
    float3 qe_ang = d_ang * float3(kb1, kb2, kt);
    Measures measures = stress_measures(b, qe_lin, qe_ang);

    // Strain rate from the governing effective stress (loading only).
    float governing = max(max(measures.tension, measures.shear), measures.compression);
    if (dt > 0.0)
    {
        float raw = fdiv(max(fdiv(governing - st.governing_stress, dt), 0.0), mat.misc.w);
        float tau = b.stiff1.z;
        float a = (flags & FLAG_EXACT_RATE_FILTER) != 0 ? -expm1_accurate(-fdiv(dt, tau)) : min(fdiv(dt, tau), 1.0);
        st.strain_rate += (raw - st.strain_rate) * a;
        st.governing_stress = governing;
    }
    float dif = (flags & FLAG_DIF) != 0 ? dif_factor(mat, st.strain_rate) : 1.0;
    float weibull = b.geom1.w;
    float fatigue_before = fatigue_factor(mat, st.life);
    float multiplier = weibull * dif * fatigue_before;
    float4 idx = failure_indices(mat, b, measures, multiplier);
    st.utilization = max(max(idx.x, idx.y), max(idx.z, idx.w));

    float dissipated = 0.0;
    float overshoot = 0.0;
    float psi_ts = 0.5 * (ks * (sq(d_lin.x) + sq(d_lin.y)) + kb1 * sq(d_ang.x) + kb2 * sq(d_ang.y) + kt * sq(d_ang.z)
                          + (d_lin.z > 0.0 ? kn * sq(d_lin.z) : 0.0));
    float psi_c = d_lin.z < 0.0 ? 0.5 * kn * sq(d_lin.z) : 0.0;
    float3 plastic = float3(st.plastic_x, st.plastic_y, st.plastic_t);

    // Damage evolution.
    if (fracture)
    {
        float lambda_ts = idx.x >= idx.y ? idx.x : idx.y;
        uint mode_ts = idx.x >= idx.y ? MODE_TENSION : MODE_SHEAR;
        if (lambda_ts > st.kappa && lambda_ts > 1.0 && psi_ts > 0.0)
        {
            float g_f = mode_ts == MODE_TENSION ? mat.energy.y : mat.energy.z;
            float r = softening ? fdiv(g_f * b.geom0.x * lambda_ts * lambda_ts, psi_ts) : 0.0;
            st.ductility = r;
            uint law = softening ? kind : KIND_BRITTLE;
            float2 inc = damage_increment(law, st.kappa, lambda_ts, r, st.damage, psi_ts);
            if (inc.x > st.damage)
            {
                // The contact energy at the old damage (a probe: friction not committed).
                Contact probe = contact_part(mat, b, st.crush, plastic, d_lin, d_ang);
                float excess = max(probe.energy - (1.0 - st.crush) * psi_c, 0.0);
                float law_released = max(inc.y - excess * (inc.x - st.damage), 0.0);
                dissipated += law_released;
                overshoot += max((psi_ts - excess) * (inc.x - st.damage) - law_released, 0.0);
                st.damage = inc.x;
                st.mode = mode_ts;
            }
        }
        st.kappa = max(st.kappa, lambda_ts);

        // Crushing acts on the stress the joint actually transmits in compression.
        float3 ang = qe_ang;
        float d_old = state.damage;
        if (d_old > 0.0)
        {
            Contact probe = contact_part(mat, b, state.crush, float3(state.plastic_x, state.plastic_y, state.plastic_t), d_lin, d_ang);
            ang = qe_ang * (1.0 - d_old) + probe.q_ang * d_old;
        }
        Measures mc = stress_measures(b, float3(0.0, 0.0, min(qe_lin.z, 0.0)), ang);
        float4 comp_idx = failure_indices(mat, b, mc, multiplier);
        float lambda_c = comp_idx.z >= comp_idx.w ? comp_idx.z : comp_idx.w;
        uint mode_c = comp_idx.z >= comp_idx.w ? MODE_COMPRESSION : MODE_BUCKLING;
        if (lambda_c > st.kappa_c && lambda_c > 1.0 && psi_c > 0.0)
        {
            float r = softening ? fdiv(mat.energy.w * b.geom0.x * lambda_c * lambda_c, psi_c) : 0.0;
            st.ductility_c = r;
            uint law = !softening ? KIND_BRITTLE : (mode_c == MODE_BUCKLING ? KIND_DUCTILE : kind);
            float2 inc = damage_increment(law, st.kappa_c, lambda_c, r, st.crush, psi_c);
            if (inc.x > st.crush)
            {
                dissipated += inc.y;
                overshoot += max(psi_c * (inc.x - st.crush) - inc.y, 0.0);
                st.crush = inc.x;
                st.mode = mode_c;
                if (st.crush >= 1.0 && st.damage < 1.0)
                {
                    // A crushed joint is gone: release the rest of its energy too.
                    dissipated += psi_ts * (1.0 - st.damage);
                    st.damage = 1.0;
                }
            }
        }
        st.kappa_c = max(st.kappa_c, lambda_c);
    }

    // Actual force: intact share + contact share + rebar.
    float dmg = st.damage;
    float3 qc_lin = float3(0.0);
    float3 qc_ang = float3(0.0);
    float psi_contact = 0.0;
    float diss_contact = 0.0;
    if (dmg == 0.0)
    {
        // Uncracked: the contact share is weighted by zero; only its offsets move.
        if ((flags & FLAG_PATCH_AFTER_DAMAGE) == 0)
        {
            float3 p = contact_offsets(mat, b, st.crush, plastic, d_lin, d_ang);
            st.plastic_x = p.x;
            st.plastic_y = p.y;
            st.plastic_t = p.z;
        }
    }
    else
    {
        Contact c = contact_part(mat, b, st.crush, plastic, d_lin, d_ang);
        qc_lin = c.q_lin;
        qc_ang = c.q_ang;
        psi_contact = c.energy;
        diss_contact = c.diss;
        st.plastic_x = c.plastic.x;
        st.plastic_y = c.plastic.y;
        st.plastic_t = c.plastic.z;
    }
    dissipated += dmg * diss_contact;
    float intact_normal = d_lin.z > 0.0 ? kn * d_lin.z : (1.0 - st.crush) * kn * d_lin.z;
    float3 force_lin = float3((1.0 - dmg) * qe_lin.x + dmg * qc_lin.x, (1.0 - dmg) * qe_lin.y + dmg * qc_lin.y, (1.0 - dmg) * intact_normal + dmg * qc_lin.z);
    float3 force_ang = qe_ang * (1.0 - dmg) + qc_ang * dmg;
    float stored = (1.0 - dmg) * (psi_ts + (1.0 - st.crush) * psi_c) + dmg * psi_contact;

    if (has_rebar && st.rebar_broken == 0.0)
    {
        float k_axial = b.rebar0.x;
        float k_dowel = b.rebar0.y;
        float yield_force = b.rebar0.z;
        float dowel_capacity = b.rebar0.w;
        float2 nr = return_map(k_axial, d_lin.z, st.rebar_plastic, yield_force);
        float2 v1 = return_map(k_dowel, d_lin.x, st.rebar_slip0, dowel_capacity);
        float2 v2 = return_map(k_dowel, d_lin.y, st.rebar_slip1, dowel_capacity);
        float work = yield_force * abs(nr.y) + dowel_capacity * (abs(v1.y) + abs(v2.y));
        st.rebar_plastic += nr.y;
        st.rebar_slip0 += v1.y;
        st.rebar_slip1 += v2.y;
        st.rebar_work += work;
        dissipated += work;
        float elastic = 0.5 * (fdiv(sq(nr.x), k_axial) + fdiv(sq(v1.x) + sq(v2.x), k_dowel));
        if (fracture && st.rebar_work >= b.rebar1.x)
        {
            st.rebar_broken = 1.0;
            dissipated += elastic;
        }
        else
        {
            force_lin += float3(v1.x, v2.x, nr.x);
            stored += elastic;
        }
    }

    // Static fatigue consumes life at the actual stress ratio against the static
    // (rate-free, undamaged) strength.
    if (fracture && dt > 0.0 && (flags & FLAG_FATIGUE) != 0)
    {
        Measures actual = stress_measures(b, force_lin, force_ang);
        float4 static_idx = failure_indices(mat, b, actual, weibull);
        float ratio = max(max(static_idx.x, static_idx.y), static_idx.z);
        st.life = max(st.life - life_rate(mat, ratio) * dt, 0.0);
    }

    st.dissipated += dissipated;
    JointResponse resp;
    resp.force_lin = force_lin;
    resp.force_ang = force_ang;
    resp.state = st;
    resp.dissipated = dissipated;
    resp.overshoot = overshoot;
    resp.stored = stored;
    resp.disconnected = was_connected && !connected(st, has_rebar);
    resp.fatigue_before = fatigue_before;
    resp.measures = measures;
    return resp;
}

// joint.rs `secant_factors`: per-component factors on the dashpots.
public void secant_factors(JointBond b, JointState st, float3 d_lin, out float3 f_lin, out float3 f_ang)
{
    float dmg = st.damage;
    bool compressed = d_lin.z < 0.0;
    float contact = compressed ? dmg : 0.0;
    float ts = max(1.0 - dmg + contact, 1e-6);
    float normal = compressed ? max(1.0 - st.crush, 1e-6) : max(1.0 - dmg, 1e-6);
    f_lin = float3(ts, ts, normal);
    f_ang = float3(ts);
    if (b.stiff1.w != 0.0 && st.rebar_broken == 0.0)
    {
        f_lin.z += fdiv(b.rebar0.x, b.stiff0.x);
        f_lin.x += fdiv(b.rebar0.y, b.stiff0.y);
        f_lin.y += fdiv(b.rebar0.y, b.stiff0.y);
    }
}
