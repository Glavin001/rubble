// Test probe for physics/mpm_material.slang: one thread per case runs the SVD, the
// material update (a sand or soil return map, or rock damage) and the Kirchhoff
// stress, so engine-webgpu
// `tests/mpm_material.rs` can compare them with the engine-terrain CPU mirror.
// Not part of any step.
#include "../physics/mpm_material.slang"

// Matches `ProbeCase` in tests/mpm_material.rs (80 B).
struct MpmProbeCase
{
    float4 f0; // F's columns (xyz)
    float4 f1;
    float4 f2;
    float4 material; // sand/soil: friction (DP alpha, or sin φ), cohesion word; rock: f_t,
                     // E·G_f/f_t; then lambda, mu
    float4 state; // logJp, model (0 Drucker–Prager, 1 Mohr–Coulomb, 2 cohesive soil, 3 rock),
                  // damage, dx
};

// Matches `ProbeResult` (176 B).
struct MpmProbeResult
{
    float4 u0; // U's columns
    float4 u1;
    float4 u2;
    float4 v0; // V's columns
    float4 v1;
    float4 v2;
    float4 sigma; // xyz; w = 1 when both input checks pass
    float4 projected; // sigmaOut (xyz), logJpOut (w); rock: sigma, damage out
    float4 tau0; // τ's columns
    float4 tau1;
    float4 tau2;
};

[[vk::binding(0, 0)]] StructuredBuffer<MpmProbeCase> gProbeCases;
[[vk::binding(1, 0)]] RWStructuredBuffer<MpmProbeResult> gProbeResults;
// [0]: cases to run.
[[vk::binding(2, 0)]] StructuredBuffer<uint> gProbeCount;

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_material_probe(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gProbeCount[0]) return;
    MpmProbeCase k = gProbeCases[i];
    float3x3 F = mpmFromCols(k.f0.xyz, k.f1.xyz, k.f2.xyz);
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 sigmaOut;
    float logJpOut;
    float3x3 tau;
    if (k.state.y > 2.5)
    {
        float3 t;
        mpmRockDamage(sigma, k.state.z, k.material.z, k.material.w, k.material.x, k.material.y,
                      k.state.w, t, logJpOut);
        sigmaOut = sigma;
        tau = mpmPrincipalKirchhoff(U, t);
    }
    else
    {
        if (k.state.y > 1.5)
            mpmSoilProjectMohrCoulomb(sigma, k.state.x, k.material.x, k.material.y, k.material.z,
                                      k.material.w, sigmaOut, logJpOut);
        else if (k.state.y > 0.5)
            mpmSandProjectMohrCoulomb(sigma, k.state.x, k.material.x, k.material.y, k.material.z,
                                      k.material.w, sigmaOut, logJpOut);
        else
            mpmSandProject(sigma, k.state.x, k.material.x, k.material.y, k.material.z, k.material.w,
                           sigmaOut, logJpOut);
        tau = mpmHenckyKirchhoff(U, sigmaOut, k.material.z, k.material.w);
    }
    bool valid = mpmSvdInputValid(F) && mpmSandInputValid(sigma, k.state.x);

    MpmProbeResult r;
    r.u0 = float4(mpmCol(U, 0u), 0.0);
    r.u1 = float4(mpmCol(U, 1u), 0.0);
    r.u2 = float4(mpmCol(U, 2u), 0.0);
    r.v0 = float4(mpmCol(V, 0u), 0.0);
    r.v1 = float4(mpmCol(V, 1u), 0.0);
    r.v2 = float4(mpmCol(V, 2u), 0.0);
    r.sigma = float4(sigma, valid ? 1.0 : 0.0);
    r.projected = float4(sigmaOut, logJpOut);
    r.tau0 = float4(mpmCol(tau, 0u), 0.0);
    r.tau1 = float4(mpmCol(tau, 1u), 0.0);
    r.tau2 = float4(mpmCol(tau, 2u), 0.0);
    gProbeResults[i] = r;
}
