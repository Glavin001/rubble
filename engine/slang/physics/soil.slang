// Bekker–Wong soil for vehicle ruts (deformable terrain T5;
// engine/docs/deformable-terrain.md, "T5 ruts"). Pure relations; the kernels
// are wheel_soil and wheel_soil_press in kernels/vehicle_step.slang. Mirrors
// engine-schema/src/soil.rs (its docs carry the derivations).
//
// A soil heightfield (header flags HF_SOIL) has, after its H = ⌈nx·nz/4⌉ height
// records, two parameter records and a plane of uncompacted heights:
//   2 + H:  (k_c, k_phi, n, c_n) as f32 bits
//   3 + H:  (max_sinkage, 0, 0, 0)
//   4 + H.. uncompacted heights, packed as the heights are
#ifndef ENGINE_PHYSICS_SOIL_SLANG
#define ENGINE_PHYSICS_SOIL_SLANG

static const uint HF_SOIL = 1u;

// k = k_c/b + k_φ.
float soilModulus(float kc, float kphi, float b) { return kc / b + kphi; }

// Rigid-wheel sinkage under load W: W = b·k·√D·c·z^(n+½).
float soilWheelSinkage(float W, float b, float D, float k, float n, float c)
{
    float base = max(W, 0.0) / (b * k * sqrt(D) * c);
    if (base <= 0.0) return 0.0;
    return pow(base, 2.0 / (2.0 * n + 1.0));
}

// Load the soil carries under a rigid wheel sunk z: b·k·√D·c·z^(n+½).
float soilWheelLoad(float z, float b, float D, float k, float n, float c)
{
    if (z <= 0.0) return 0.0;
    return b * k * sqrt(D) * c * pow(z, n + 0.5);
}

// The sinkage a step reaches with the soil in series with the suspension (the
// wheel's bottom a massless node between them): the soil yields until it
// carries what the suspension pushes, and the suspension's push falls by
// S = K + D/dt per metre the ground gives way under it this step (spring and
// damper, implicit). From depth zNow under load W: the root of
// soil(z) = W − S·(z − zNow), between the static sinkage of W and zNow
// (bisection; 32 halvings reach f32 resolution). S = 0 gives the static
// sinkage of W.
float soilSeriesSinkage(float W, float zNow, float S, float b, float D, float k, float n, float c)
{
    float zs = soilWheelSinkage(W, b, D, k, n, c);
    float lo = min(zs, zNow);
    float hi = max(zs, zNow);
    for (uint it = 0u; it < 32u; it++)
    {
        float mid = 0.5 * (lo + hi);
        float g = soilWheelLoad(mid, b, D, k, n, c) - W + S * (mid - zNow);
        if (g > 0.0) hi = mid;
        else lo = mid;
    }
    return 0.5 * (lo + hi);
}

// Compaction resistance b·k·(z^(n+1) − z₀^(n+1))/(n+1) for soil already
// compacted to z₀ (never negative).
float soilCompaction(float b, float k, float n, float z, float z0)
{
    float zz = max(z, 0.0);
    if (zz <= 0.0) return 0.0;
    float z1 = clamp(z0, 0.0, zz);
    float before = z1 > 0.0 ? pow(z1, n + 1.0) : 0.0;
    return b * k * (pow(zz, n + 1.0) - before) / (n + 1.0);
}

// Depth of the imprint a wheel's bottom leaves at a point as it sweeps a
// segment: lateral / longitudinal offsets from the segment's start, the
// segment in the same axes, the imprint's half width, sinkage z, radius R. The
// imprint is the wheel's lower circle (depth z − (R − √(R² − u²)) at u along
// the track) across its width; the swept point nearest along the track is the
// deepest.
float soilImprintDepth(float lat, float lon, float dLat, float dLon, float hw, float z, float R)
{
    float s0 = 0.0, s1 = 1.0;
    if (abs(dLat) > 1e-9)
    {
        float a = (lat - hw) / dLat;
        float b = (lat + hw) / dLat;
        s0 = max(s0, min(a, b));
        s1 = min(s1, max(a, b));
    }
    else if (abs(lat) > hw) return 0.0;
    if (s0 > s1) return 0.0;
    float s = abs(dLon) > 1e-9 ? clamp(lon / dLon, s0, s1) : s0;
    float u = abs(lon - s * dLon);
    if (u >= R) return 0.0;
    float sag = R - sqrt(R * R - u * u);
    return max(z - sag, 0.0);
}

#endif
