// What every render layer shares: the engine's body and collider records, the frame (view,
// eye, light, the preset's look), and the two lighting models the pages use.

struct Body
{
    float3 position; float invMass;
    float4 orientation;
    float3 invInertia; uint flags;
    float3 linearVel; float sleepTimer;
    float3 angularVel; uint generation;
    float3 initialPos; float initialPad;
    float4 initialRot;
};

struct Collider
{
    uint body; uint shape; uint material; uint generation;
    float3 localPos; float density;
    float4 localRot;
    float4 params;
};

// Colours (style.x): the demo's position tint, sleep state, speed; the terrain page's shapes.
static const uint COLOR_POSITION = 0u;
static const uint COLOR_SLEEP = 1u;
static const uint COLOR_SPEED = 2u;
static const uint COLOR_SHAPES = 3u;
// The motion page's: characters, kinematic platforms, CCD bodies, parked and control balls.
static const uint COLOR_MOTION = 4u;
// Lighting (style.y): the demo's (Lambert, fixed light) or sky, specular and fog.
static const uint LIGHT_DEMO = 0u;
static const uint LIGHT_SKY = 1u;

// The view-projection by columns; the eye; the light direction; style: colours, lighting,
// the projectile's body (0xffffffff: every sphere), the heightfield's hull arena base;
// params: the ground square's half side and height, the bounds scale (corner × half × scale:
// 2 exact, 2.02 padded), the fog distance; fog: its colour, and the specular strength (w);
// hide: bodies below this height are not drawn (light.w, 0: none).
struct Frame
{
    float4 c0; float4 c1; float4 c2; float4 c3;
    float4 eye;
    float4 light;
    uint4 style;
    float4 params;
    float4 fog;
    // The projectile pool's records: base, stride (vec4s).
    uint4 pool;
};

float3 quatRotate(float4 q, float3 v)
{
    float3 u = q.xyz;
    return u * (2.0 * dot(u, v)) + v * (q.w * q.w - dot(u, u)) + cross(u, v) * (2.0 * q.w);
}

float4 clip(Frame f, float3 w) { return f.c0 * w.x + f.c1 * w.y + f.c2 * w.z + f.c3; }

float hash(uint n)
{
    uint x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return float((x >> 22u) ^ x) / 4294967295.0;
}

float3 shade(Frame f, float3 c, float3 n, float3 w)
{
    if (f.style.y == LIGHT_SKY)
    {
        float3 l = normalize(f.light.xyz);
        float diff = max(dot(n, l), 0.0);
        float3 v = normalize(f.eye.xyz - w);
        float spec = pow(max(dot(reflect(-l, n), v), 0.0), 32.0) * f.fog.w;
        float sky = 0.5 + 0.5 * n.y;
        float fog = clamp(length(f.eye.xyz - w) / f.params.w, 0.0, 1.0);
        return lerp(c * (0.22 + 0.18 * sky + 0.75 * diff) + spec, f.fog.xyz, fog * fog);
    }
    float3 light = normalize(float3(0.35, 0.85, 0.4));
    float ndl = max(dot(n, light), 0.0);
    return c * (0.28 + 0.72 * ndl);
}
