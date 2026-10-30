// Terrain look-ahead: after the rigid step, every awake dynamic collider's
// ballistic path over the next frames is marched against the watched
// heightfields, and the strike with the most kinetic energy is reported, so the
// host can open a terrain region *before* the body reaches the ground (forward
// only: the region exists when the contact happens, nothing is re-stepped). A
// hit event arrives frames after the rigid step has already stopped the body on
// rigid ground, too late to deform it.
//
// Three passes over the colliders pick one strike deterministically: the most
// energetic (energy bits, positive floats order as uints), then the lowest
// collider index among equals, whose record is then written. The pass only
// reads physics state.
#include "world_step.slang"

// Binding 52 (u32 words): host parameters, then the GPU's pick.
//   0: watched heightfield colliders (at most 4), 1..4: their indices,
//   5: horizon (s, f32), 6: samples, 7: gravity y (f32), 8: min impact speed
//   (f32), 9: min impact kinetic energy (J, f32), 10: earliest strike (s, f32).
//   16: best energy bits, 17: best collider + 1 (min), 18..29: the record.
BIND(52) RWStructuredBuffer<AUint> gTerrainWatch;

static const uint TW_OUT = 16u;
static const uint TW_MAX_FIELDS = 4u;
static const uint TW_BISECT = 12u;

uint twIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }
float twParam(uint k) { return asfloat(ALOAD(gTerrainWatch[k])); }

float twBoundingRadius(Collider c)
{
    if (c.shape == SHAPE_SPHERE) return c.params.x;
    if (c.shape == SHAPE_CAPSULE) return c.params.x + c.params.y;
    if (c.shape == SHAPE_BOX) return length(c.params.xyz);
    return c.params.w;
}

struct TwStrike
{
    float energy;  // kinetic energy at the strike (J); 0: none
    float time;    // from now (s)
    float3 point;  // collider centre at the strike (world)
    float3 vel;    // velocity at the strike
    float radius;
    float mass;
    uint field;    // the heightfield collider struck
};

// How far the collider's lowest point is above field `hc` at time t (negative:
// below), and whether (x, z) is over the field at all.
bool twClearance(Collider hc, float3 c0, float3 v, float3 g, float r, float t, out float gap)
{
    Frame f = colliderFrame(hc, gBodies[hc.body]);
    HeightFieldRef hf = heightFieldOf(hc);
    float3 x = c0 + v * t + 0.5 * g * t * t;
    float3 xl = frameToLocal(f, x);
    float h;
    float3 n;
    gap = 0.0;
    if (!hfSurface(hf, xl.x, xl.z, h, n)) return false;
    gap = xl.y - r - h;
    return true;
}

TwStrike twStrike(uint i)
{
    TwStrike s;
    s.energy = 0.0;
    s.time = 0.0;
    s.point = float3(0, 0, 0);
    s.vel = float3(0, 0, 0);
    s.radius = 0.0;
    s.mass = 0.0;
    s.field = 0u;
    if (i >= gCounters[0].colliderCount) return s;
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE || col.shape == SHAPE_HEIGHTFIELD) return s;
    BodyState b = gBodies[col.body];
    if (b.invMass <= 0.0 || bodyIsSleeping(b)) return s;
    float r = twBoundingRadius(col);
    float3 c0 = colliderFrame(col, b).p;
    float3 v = b.linearVel;
    float3 g = float3(0.0, twParam(7), 0.0);
    float horizon = twParam(5);
    uint samples = max(ALOAD(gTerrainWatch[6]), 1u);
    uint fields = min(ALOAD(gTerrainWatch[0]), TW_MAX_FIELDS);
    float best = horizon + 1.0;
    for (uint k = 0u; k < fields; k++)
    {
        uint fi = ALOAD(gTerrainWatch[1u + k]);
        Collider hc = gColliders[fi];
        float gap;
        // Already touching or beside the field now: not a strike to predict.
        if (!twClearance(hc, c0, v, g, r, 0.0, gap) || gap <= 0.0) continue;
        float t0 = 0.0;
        for (uint j = 1u; j <= samples; j++)
        {
            float t1 = horizon * float(j) / float(samples);
            if (t1 >= best) break;
            bool over = twClearance(hc, c0, v, g, r, t1, gap);
            if (over && gap <= 0.0)
            {
                // Bisect [t0, t1] for the first touch (to horizon/samples/2^12).
                for (uint it = 0u; it < TW_BISECT; it++)
                {
                    float tm = 0.5 * (t0 + t1);
                    float gm;
                    bool om = twClearance(hc, c0, v, g, r, tm, gm);
                    if (om && gm <= 0.0) t1 = tm;
                    else t0 = tm;
                }
                best = t1;
                s.field = fi;
                break;
            }
            t0 = t1;
        }
    }
    if (best > horizon || best < twParam(10)) return s;
    float3 vi = v + g * best;
    float m = 1.0 / b.invMass;
    float e = 0.5 * m * dot(vi, vi);
    if (length(vi) < twParam(8) || e < twParam(9)) return s;
    s.energy = e;
    s.time = best;
    s.point = c0 + v * best + 0.5 * g * best * best;
    s.vel = vi;
    s.radius = r;
    s.mass = m;
    return s;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_watch_energy(uint3 tid : SV_DispatchThreadID)
{
    TwStrike s = twStrike(twIndex(tid));
    if (s.energy > 0.0) AMAX2(gTerrainWatch[TW_OUT], asuint(s.energy));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_watch_pick(uint3 tid : SV_DispatchThreadID)
{
    uint i = twIndex(tid);
    TwStrike s = twStrike(i);
    if (s.energy > 0.0 && asuint(s.energy) == ALOAD(gTerrainWatch[TW_OUT]))
        AMIN2(gTerrainWatch[TW_OUT + 1u], i + 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_watch_write(uint3 tid : SV_DispatchThreadID)
{
    uint i = twIndex(tid);
    if (ALOAD(gTerrainWatch[TW_OUT + 1u]) != i + 1u) return;
    TwStrike s = twStrike(i);
    uint o = TW_OUT + 2u;
    ASTORE(gTerrainWatch[o + 0u], gColliders[i].body);
    ASTORE(gTerrainWatch[o + 1u], s.field);
    ASTORE(gTerrainWatch[o + 2u], asuint(s.time));
    ASTORE(gTerrainWatch[o + 3u], asuint(s.point.x));
    ASTORE(gTerrainWatch[o + 4u], asuint(s.point.y));
    ASTORE(gTerrainWatch[o + 5u], asuint(s.point.z));
    ASTORE(gTerrainWatch[o + 6u], asuint(s.vel.x));
    ASTORE(gTerrainWatch[o + 7u], asuint(s.vel.y));
    ASTORE(gTerrainWatch[o + 8u], asuint(s.vel.z));
    ASTORE(gTerrainWatch[o + 9u], asuint(s.radius));
    ASTORE(gTerrainWatch[o + 10u], asuint(s.mass));
    ASTORE(gTerrainWatch[o + 11u], asuint(s.energy));
}
