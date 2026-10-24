// MPM sand material (deformable terrain T2; engine/docs/deformable-terrain.md,
// "T2 kernel design"): the 3×3 SVD of a deformation gradient, the Drucker–Prager
// return map in Hencky strain and the Kirchhoff stress. Functions only; the MLS-MPM
// kernels include this file. engine-terrain `material.rs` is the CPU mirror, step
// for step, and engine-webgpu `tests/mpm_material.rs` checks the two agree.
//
// Matrices are handled as columns (float3) so the code reads the same as the
// mirror; float3x3 appears only at the interface, built row by row.
//
// Inputs are validated with ordered range checks (mpmSvdInputValid,
// mpmSandInputValid): a NaN fails them on every backend, whereas `x != x` is
// folded away under Metal fast-math.
#ifndef ENGINE_PHYSICS_MPM_MATERIAL_SLANG
#define ENGINE_PHYSICS_MPM_MATERIAL_SLANG

// Jacobi sweeps of mpmSvd3, each conjugating the pairs (0,1), (0,2), (1,2).
static const uint MPM_SVD_SWEEPS = 6u;
// (1 + √2)²: below it the approximate half angle is within π/8 (McAdams et al. 2011).
static const float MPM_SVD_GAMMA = 5.828427124746190;
static const float MPM_COS_PI_8 = 0.9238795325112867;
static const float MPM_SIN_PI_8 = 0.3826834323650898;
// Largest |entry| accepted: keeps FᵀF and the log strains finite in f32.
static const float MPM_INPUT_LIMIT = 1.0e18;
// Smallest |σ| the return map takes the log of (Klár et al. 2016).
static const float MPM_SIGMA_FLOOR = 1.0e-4;

float3x3 mpmFromCols(float3 a, float3 b, float3 c)
{
    return float3x3(a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z);
}

float3 mpmCol(float3x3 m, uint j)
{
    return float3(m[0][j], m[1][j], m[2][j]);
}

// Hamilton product, (x, y, z, w) layout.
float4 mpmQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

void mpmQuatToCols(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    c0 = float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    c1 = float3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    c2 = float3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}

// One Jacobi conjugation S ← GᵀSG of the symmetric S on the pair (p, q), with k the
// third axis; the entries are passed permuted to (p, q, k). G turns e_p towards e_q
// by θ. Its half angle (ch, sh) is McAdams' approximation tan(θ/2) ≈ s_pq / 2(s_pp −
// s_qq), or π/8 where that would exceed π/8. A pair already diagonal (s_pq = 0) is
// not turned, so a diagonal F (F = I at rest) stays exact. Returns the half angle.
void mpmJacobiPair(inout float spp, inout float sqq, inout float spq, inout float spk,
                   inout float sqk, out float ch, out float sh)
{
    ch = 2.0 * (spp - sqq);
    sh = spq;
    if (sh == 0.0)
    {
        ch = 1.0;
        return;
    }
    if (MPM_SVD_GAMMA * sh * sh < ch * ch)
    {
        float w = 1.0 / sqrt(ch * ch + sh * sh);
        ch *= w;
        sh *= w;
    }
    else
    {
        ch = MPM_COS_PI_8;
        sh = MPM_SIN_PI_8;
    }
    float c = ch * ch - sh * sh;
    float s = 2.0 * ch * sh;
    float app = spp, aqq = sqq, apq = spq, apk = spk, aqk = sqk;
    spp = c * c * app + 2.0 * c * s * apq + s * s * aqq;
    sqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;
    spq = (c * c - s * s) * apq - c * s * (app - aqq);
    spk = c * apk + s * aqk;
    sqk = c * aqk - s * apk;
}

// Swap columns a and b, negating one, so the matrix keeps its determinant.
void mpmSwapCols(inout float3 a, inout float3 b)
{
    float3 t = a;
    a = b;
    b = -t;
}

// (x, y) ← (c·x + s·y, c·y − s·x).
void mpmRotatePair(float c, float s, inout float x, inout float y)
{
    float t = x;
    x = c * t + s * y;
    y = c * y - s * t;
}

// (a, b) ← (c·a + s·b, c·b − s·a): columns a, b times the rotation G that turns
// e_a towards e_b.
void mpmRotateCols(float c, float s, inout float3 a, inout float3 b)
{
    float3 t = a;
    a = c * t + s * b;
    b = c * b - s * t;
}

// The rotation (c, s) with GᵀMG diagonal for the symmetric M = [[p, q], [q, r]]:
// t = tan θ is the root of q·t² + (p − r)·t − q = 0 with |t| ≤ 1.
void mpmSym2Jacobi(float p, float q, float r, out float c, out float s)
{
    if (q == 0.0)
    {
        c = 1.0;
        s = 0.0;
        return;
    }
    float h = p - r;
    float root = sqrt(h * h + 4.0 * q * q);
    float t = 2.0 * q / (h >= 0.0 ? h + root : h - root);
    c = 1.0 / sqrt(1.0 + t * t);
    s = t * c;
}

// The Givens rotation (c, s) that zeroes b against the pivot a, leaving the pivot
// √(a² + b²) ≥ 0. No rotation when both are zero.
void mpmGivens(float a, float b, out float c, out float s)
{
    float r = sqrt(a * a + b * b);
    if (r > 0.0)
    {
        c = a / r;
        s = b / r;
    }
    else
    {
        c = 1.0;
        s = 0.0;
    }
}

// F's SVD entries pass mpmSvd3 without overflow (false for NaN or ±inf).
bool mpmSvdInputValid(float3x3 F)
{
    float3 lim = float3(MPM_INPUT_LIMIT);
    return all(abs(F[0]) <= lim) && all(abs(F[1]) <= lim) && all(abs(F[2]) <= lim);
}

// F = U·diag(sigma)·Vᵀ by McAdams et al. (2011): Jacobi eigen-decomposition of FᵀF
// with V accumulated as a quaternion, columns of B = F·V sorted by decreasing norm,
// then QR of B by Givens rotations (U = G₁G₂G₃, R's diagonal is sigma), and a 2×2
// SVD of R's trailing block. U and V are rotations (det +1); |sigma| decreases, and
// only sigma.z is negative (det F < 0). ‖UΣVᵀ − F‖ ≈ 1e-6·‖F‖.
void mpmSvd3(float3x3 F, out float3x3 U, out float3 sigma, out float3x3 V)
{
    float3 f0 = mpmCol(F, 0u), f1 = mpmCol(F, 1u), f2 = mpmCol(F, 2u);
    float s00 = dot(f0, f0), s11 = dot(f1, f1), s22 = dot(f2, f2);
    float s01 = dot(f0, f1), s02 = dot(f0, f2), s12 = dot(f1, f2);
    float4 q = float4(0.0, 0.0, 0.0, 1.0);
    float ch, sh;
    for (uint sweep = 0u; sweep < MPM_SVD_SWEEPS; sweep++)
    {
        // (0,1) turns about +z, (0,2) about −y, (1,2) about +x.
        mpmJacobiPair(s00, s11, s01, s02, s12, ch, sh);
        q = mpmQuatMul(q, float4(0.0, 0.0, sh, ch));
        mpmJacobiPair(s00, s22, s02, s01, s12, ch, sh);
        q = mpmQuatMul(q, float4(0.0, -sh, 0.0, ch));
        mpmJacobiPair(s11, s22, s12, s01, s02, ch, sh);
        q = mpmQuatMul(q, float4(sh, 0.0, 0.0, ch));
    }
    float3 v0, v1, v2;
    mpmQuatToCols(normalize(q), v0, v1, v2);

    float3 b0 = f0 * v0.x + f1 * v0.y + f2 * v0.z;
    float3 b1 = f0 * v1.x + f1 * v1.y + f2 * v1.z;
    float3 b2 = f0 * v2.x + f1 * v2.y + f2 * v2.z;
    float r0 = dot(b0, b0), r1 = dot(b1, b1), r2 = dot(b2, b2);
    if (r0 < r1)
    {
        mpmSwapCols(b0, b1);
        mpmSwapCols(v0, v1);
        float t = r0;
        r0 = r1;
        r1 = t;
    }
    if (r0 < r2)
    {
        mpmSwapCols(b0, b2);
        mpmSwapCols(v0, v2);
        float t = r0;
        r0 = r2;
        r2 = t;
    }
    if (r1 < r2)
    {
        mpmSwapCols(b1, b2);
        mpmSwapCols(v1, v2);
    }

    float3 u0 = float3(1.0, 0.0, 0.0), u1 = float3(0.0, 1.0, 0.0), u2 = float3(0.0, 0.0, 1.0);
    float c, s;
    // Rows (0,1), then (0,2), zero column 0 below the diagonal; rows (1,2) zero b1.z.
    mpmGivens(b0.x, b0.y, c, s);
    mpmRotatePair(c, s, b0.x, b0.y);
    mpmRotatePair(c, s, b1.x, b1.y);
    mpmRotatePair(c, s, b2.x, b2.y);
    mpmRotateCols(c, s, u0, u1);
    mpmGivens(b0.x, b0.z, c, s);
    mpmRotatePair(c, s, b0.x, b0.z);
    mpmRotatePair(c, s, b1.x, b1.z);
    mpmRotatePair(c, s, b2.x, b2.z);
    mpmRotateCols(c, s, u0, u2);
    mpmGivens(b1.y, b1.z, c, s);
    mpmRotatePair(c, s, b1.y, b1.z);
    mpmRotatePair(c, s, b2.y, b2.z);
    mpmRotateCols(c, s, u1, u2);

    // 2×2 SVD of the trailing block T = [[b1.y, b2.y], [b1.z, b2.z]]. B's entries
    // are accurate to ε·‖F‖, so this resolves the plane of σ₂, σ₃ that FᵀF cannot
    // when F is nearly rank 1. A left rotation makes T symmetric, then a two-sided
    // Jacobi rotation diagonalises it.
    mpmGivens(b1.y + b2.z, b1.z - b2.y, c, s);
    mpmRotatePair(c, s, b1.y, b1.z);
    mpmRotatePair(c, s, b2.y, b2.z);
    mpmRotateCols(c, s, u1, u2);
    mpmSym2Jacobi(b1.y, 0.5 * (b2.y + b1.z), b2.z, c, s);
    mpmRotatePair(c, s, b1.y, b1.z);
    mpmRotatePair(c, s, b2.y, b2.z);
    mpmRotateCols(c, s, b1, b2);
    mpmRotateCols(c, s, u1, u2);
    mpmRotateCols(c, s, v1, v2);
    // Restore the sign rule: |σ₂| ≥ |σ₃| and σ₂ ≥ 0, so σ₃ carries det F's sign.
    if (abs(b1.y) < abs(b2.z))
    {
        mpmSwapCols(u1, u2);
        mpmSwapCols(v1, v2);
        float t = b1.y;
        b1.y = b2.z;
        b2.z = t;
    }
    if (b1.y < 0.0)
    {
        b1.y = -b1.y;
        b2.z = -b2.z;
        u1 = -u1;
        u2 = -u2;
    }

    U = mpmFromCols(u0, u1, u2);
    sigma = float3(b0.x, b1.y, b2.z);
    V = mpmFromCols(v0, v1, v2);
}

// The Drucker–Prager α for friction angle φ: √(2/3)·2 sin φ / (3 − sin φ).
float mpmFrictionAlpha(float phiRadians)
{
    float s = sin(phiRadians);
    return sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s);
}

// The return map's inputs are in range (false for NaN or ±inf).
bool mpmSandInputValid(float3 sigma, float logJp)
{
    return all(abs(sigma) <= float3(MPM_INPUT_LIMIT)) && abs(logJp) <= MPM_INPUT_LIMIT;
}

// Drucker–Prager return map in Hencky strain (Klár et al. 2016) with the log-volume
// correction (Tampubolon et al. 2017, β = 1): ε = ln max(|σ|, 1e-4) − cohesion +
// logJp/3, so compression first consumes the volume a particle gained when it
// separated, and tr = Σε. Expansion (tr ≥ 0) projects to the cone's tip and logJp
// keeps tr; otherwise logJp is cleared and ε moves onto the cone along its
// (traceless) deviator ε̂ when it lies outside. sigmaOut is positive.
void mpmSandProject(float3 sigma, float logJp, float alpha, float cohesion, float lambda,
                    float mu, out float3 sigmaOut, out float logJpOut)
{
    float3 eps = log(max(abs(sigma), float3(MPM_SIGMA_FLOOR))) - cohesion + logJp / 3.0;
    float tr = eps.x + eps.y + eps.z;
    if (tr >= 0.0)
    {
        sigmaOut = float3(exp(cohesion));
        logJpOut = tr;
        return;
    }
    logJpOut = 0.0;
    float3 epsHat = eps - tr / 3.0;
    float epsHatNorm = length(epsHat);
    float dGamma = epsHatNorm + (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * tr * alpha;
    // ‖ε̂‖ = 0 implies dGamma ≤ 0 when α ≥ 0; the norm test keeps the division safe
    // for any α.
    if (dGamma > 0.0 && epsHatNorm > 0.0)
    {
        eps -= (dGamma / epsHatNorm) * epsHat;
    }
    sigmaOut = exp(eps + cohesion);
}

// Mohr–Coulomb return map in Hencky strain, with the same volume correction,
// tip and cohesion shift as mpmSandProject, so φ is the friction angle in every
// stress state (a Drucker–Prager cone matches it only on one meridian). Flow is
// non-dilatant (ψ = 0): the plastic strain is deviatoric, as Klár's projection.
// With principal Kirchhoff stresses τᵢ = 2μεᵢ + λ tr ε (compression negative,
// ε₁ ≥ ε₂ ≥ ε₃, the SVD's order) the yield function is
//   f = (τ₁ − τ₃) + (τ₁ + τ₃)·sin φ ≤ 0.
// A flow direction N with N·1 = 0 changes τ by 2μN, so the return onto the main
// plane is closed form (Δγ = f/4μ along N = (1, 0, −1)); when it would reorder
// the strains it returns onto the edge (τ₁ = τ₂ or τ₂ = τ₃) by a 2×2 solve. Under
// compression an edge return always exists (both multipliers ≥ 0); they are
// clamped at 0 only against rounding, for a state already on the edge.
void mpmSandProjectMohrCoulomb(float3 sigma, float logJp, float sinPhi, float cohesion, float lambda,
                               float mu, out float3 sigmaOut, out float logJpOut)
{
    float3 eps = log(max(abs(sigma), float3(MPM_SIGMA_FLOOR))) - cohesion + logJp / 3.0;
    float tr = eps.x + eps.y + eps.z;
    if (tr >= 0.0)
    {
        sigmaOut = float3(exp(cohesion));
        logJpOut = tr;
        return;
    }
    logJpOut = 0.0;
    float3 tau = 2.0 * mu * eps + lambda * tr;
    float3 gA = float3(1.0 + sinPhi, 0.0, -(1.0 - sinPhi));
    float fA = dot(gA, tau);
    if (fA <= 0.0)
    {
        sigmaOut = exp(eps + cohesion);
        return;
    }
    float3 nA = float3(1.0, 0.0, -1.0);
    float3 e = eps - (fA / (4.0 * mu)) * nA;
    if (e.x < e.y || e.y < e.z)
    {
        // The edge the main-plane return crossed into: τ₁ = τ₂ (planes (1,3)
        // and (2,3)) or τ₂ = τ₃ (planes (1,3) and (1,2)).
        bool right = e.y > e.x;
        float3 nB = right ? float3(0.0, 1.0, -1.0) : float3(1.0, -1.0, 0.0);
        float3 gB = right ? float3(0.0, 1.0 + sinPhi, -(1.0 - sinPhi)) : float3(1.0 + sinPhi, -(1.0 - sinPhi), 0.0);
        float fB = dot(gB, tau);
        // [gA·nA gA·nB; gB·nA gB·nB]·2μ·(Δa, Δb) = (fA, fB).
        float a11 = 2.0, a12 = dot(gA, nB), a21 = dot(gB, nA), a22 = 2.0;
        float det = (a11 * a22 - a12 * a21) * 2.0 * mu;
        float da = max((a22 * fA - a12 * fB) / det, 0.0);
        float db = max((a11 * fB - a21 * fA) / det, 0.0);
        e = eps - da * nA - db * nB;
    }
    sigmaOut = exp(e + cohesion);
}

// Kirchhoff stress of the Hencky (St. Venant–Kirchhoff in log strain) model:
// τ = U·diag(2μ ln σ + λ Σ ln σ)·Uᵀ, for sigma > 0 (after mpmSandProject).
float3x3 mpmHenckyKirchhoff(float3x3 U, float3 sigma, float lambda, float mu)
{
    float3 e = log(sigma);
    float3 d = 2.0 * mu * e + lambda * (e.x + e.y + e.z);
    float3 u0 = mpmCol(U, 0u), u1 = mpmCol(U, 1u), u2 = mpmCol(U, 2u);
    float3 t0 = u0 * (d.x * u0.x) + u1 * (d.y * u1.x) + u2 * (d.z * u2.x);
    float3 t1 = u0 * (d.x * u0.y) + u1 * (d.y * u1.y) + u2 * (d.z * u2.y);
    float3 t2 = u0 * (d.x * u0.z) + u1 * (d.y * u1.z) + u2 * (d.z * u2.z);
    return mpmFromCols(t0, t1, t2);
}

#endif
