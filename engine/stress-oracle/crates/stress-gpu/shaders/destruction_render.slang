// Draws the engine-coupled world (`GpuWorld`) straight from the stress solver's GPU
// state: every chunk its convex hull (`chunk_hull_vs`, triangles in the chunk's own frame
// from the host) or its oriented box (`chunk_vs`) at its island's pose (world.slang `chunk_world`,
// `chunk_box`: island rotation x hidden rotation x chunk rotation, centre at rest centre
// + u), coloured by the largest utilization (failure index) of its unbroken bonds, read
// from the bonds' joint state through the chunk's CSR incidence. Chunks without an
// unbroken bond (loose pieces) are drawn grey. Chunks the solver keeps out of its GPU
// tables (quiet clusters) come from the host's mirror (`host_chunks`). Impactors are
// drawn from the impactor records (spheres and boxes), and the ground as a checkered
// plane.
// Vertices come from the vertex index and storage buffers: no vertex buffers.

import joint;
import common;

struct Camera
{
    float4 view_proj[4]; // column-major view-projection
    float4 light;        // xyz: direction towards the light
    float4 ground;       // x: height, y: half extent, zw: centre (x, y)
};

// world.slang `BondDyn` (same members, same layout).
struct BondDyn
{
    JointState js;
    float4 force_lin;
    float4 force_ang;
    float4 sums;
    float4 comps;
    uint4 events; // absolute substep of: crack, creak, break (0: none)
};

// world.slang `Impactor` (same members, same layout).
struct Impactor
{
    float4 position;
    float4 position_err;
    float4 velocity;
    float4 velocity_err;
    float4 angular_velocity;
    float4 rotation;
    float4 inertia0;
    float4 inertia1;
    float4 inertia2;
    float4 inv0;
    float4 inv1;
    float4 inv2;
    float4 shape; // x: 0 sphere, 1 box; y: radius
    float4 half;  // box half extents (sphere: radius)
    float4 mat;
    float4 crush;
    float4 geom;
    float4 rotation_err;
    float4 ledger;
    uint4 cand;
    float4 momentum;
    float4 momentum_err;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(1, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(2, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(3, 0)]] StructuredBuffer<float4> state; // 4 per chunk: u, theta, v, w
[[vk::binding(4, 0)]] StructuredBuffer<uint> index;   // CSR: [chunk_count + 1] offsets, then 2 * bond + side
[[vk::binding(5, 0)]] StructuredBuffer<BondDyn> bond_dyn;
[[vk::binding(6, 0)]] StructuredBuffer<Impactor> impactors;
// Chunks the solver leaves out of its GPU tables (quiet clusters: they do not move while
// out), as the host's mirror has them: world boxes, coloured the same way.
struct HostChunk
{
    float4 center; // world; w: utilization (< 0: no unbroken bond)
    float4 axis0;  // world box axes; w: half extent along it
    float4 axis1;
    float4 axis2;
};
[[vk::binding(7, 0)]] StructuredBuffer<HostChunk> host_chunks;
// Hull triangles in a chunk's own frame (centred, chunk axes: the box's frame), three
// vertices each; pos.w holds the chunk's index (GPU table index for `hull_verts`, host
// chunk index for `host_hull_verts`).
struct HullVertex
{
    float4 pos;
    float4 normal;
};
[[vk::binding(8, 0)]] StructuredBuffer<HullVertex> hull_verts;
[[vk::binding(9, 0)]] StructuredBuffer<HullVertex> host_hull_verts;

struct VertexOut
{
    float4 position : SV_Position;
    float3 normal;
    float3 color;
    float3 world;
};

static const uint SPHERE_LON = 16;
static const uint SPHERE_LAT = 10;
static const float PI = 3.14159265;

float4 clip_of(float3 world)
{
    return camera.view_proj[0] * world.x + camera.view_proj[1] * world.y + camera.view_proj[2] * world.z + camera.view_proj[3];
}

float3 ramp(float t)
{
    t = saturate(t);
    float3 c0 = float3(0.15, 0.25, 0.85);
    float3 c1 = float3(0.10, 0.75, 0.85);
    float3 c2 = float3(0.25, 0.80, 0.30);
    float3 c3 = float3(0.95, 0.85, 0.20);
    float3 c4 = float3(0.90, 0.20, 0.15);
    float s = t * 4.0;
    if (s < 1.0) return lerp(c0, c1, s);
    if (s < 2.0) return lerp(c1, c2, s - 1.0);
    if (s < 3.0) return lerp(c2, c3, s - 2.0);
    return lerp(c3, c4, s - 3.0);
}

// Corner `vertex` (of 36) of the unit box [-1, 1]^3 and its face normal.
void box_corner(uint vertex, out float3 p, out float3 n)
{
    uint face = vertex / 6;
    uint corner = vertex % 6;
    uint axis = face / 2;
    float sign = (face % 2 == 0) ? 1.0 : -1.0;
    float2 quad[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float2 st = quad[corner];
    if (sign < 0.0)
        st.x = -st.x;
    if (axis == 0) { n = float3(sign, 0, 0); p = float3(sign, st.x, st.y); }
    else if (axis == 1) { n = float3(0, sign, 0); p = float3(st.y, sign, st.x); }
    else { n = float3(0, 0, sign); p = float3(st.x, st.y, sign); }
}

Quat rotation_of(float4 hi, float4 lo)
{
    return normalized(quat_of(hi + lo));
}

// A point and normal in GPU chunk `c`'s own frame, at the chunk's pose and colour.
VertexOut chunk_vertex(uint c, float3 p, float3 n)
{
    ChunkStatic ch = chunks[c];
    Island isl = islands[ch.info.y];
    Quat q = rotation_of(isl.rotation, isl.rotation_err);
    float3 th = state[4 * c + 1].xyz;
    Quat hidden = from_axis_angle(th, length(th));
    float3 local = float3(dot(ch.crot0.xyz, p), dot(ch.crot1.xyz, p), dot(ch.crot2.xyz, p));
    float3 local_n = float3(dot(ch.crot0.xyz, n), dot(ch.crot1.xyz, n), dot(ch.crot2.xyz, n));
    float3 body = ch.center.xyz + state[4 * c].xyz + rotate(hidden, local);
    float3 world = (isl.position.xyz + isl.position_err.xyz) + rotate(q, body);

    // The largest utilization of the chunk's unbroken bonds.
    float util = 0.0;
    uint intact = 0;
    for (uint e = index[c]; e < index[c + 1]; e++)
    {
        BondDyn bd = bond_dyn[index[e] >> 1];
        if (bd.events.z != 0)
            continue;
        intact++;
        util = max(util, bd.js.utilization);
    }

    VertexOut o;
    o.position = clip_of(world);
    o.normal = rotate(q, rotate(hidden, local_n));
    o.color = intact > 0 ? ramp(util) : float3(0.55, 0.52, 0.48);
    o.world = world;
    return o;
}

[shader("vertex")]
VertexOut chunk_vs(uint vertex: SV_VertexID, uint instance: SV_InstanceID)
{
    float3 p;
    float3 n;
    box_corner(vertex, p, n);
    return chunk_vertex(instance, p * chunks[instance].half.xyz, n);
}

[shader("vertex")]
VertexOut chunk_hull_vs(uint vertex: SV_VertexID)
{
    HullVertex v = hull_verts[vertex];
    return chunk_vertex(uint(v.pos.w), v.pos.xyz, v.normal.xyz);
}

// A point and normal in host chunk `hc`'s own frame, at its (world) pose and colour.
VertexOut host_vertex(HostChunk hc, float3 p, float3 n)
{
    float3 world = hc.center.xyz + hc.axis0.xyz * p.x + hc.axis1.xyz * p.y + hc.axis2.xyz * p.z;
    VertexOut o;
    o.position = clip_of(world);
    o.normal = hc.axis0.xyz * n.x + hc.axis1.xyz * n.y + hc.axis2.xyz * n.z;
    o.color = hc.center.w >= 0.0 ? ramp(hc.center.w) : float3(0.55, 0.52, 0.48);
    o.world = world;
    return o;
}

[shader("vertex")]
VertexOut host_chunk_vs(uint vertex: SV_VertexID, uint instance: SV_InstanceID)
{
    float3 p;
    float3 n;
    box_corner(vertex, p, n);
    HostChunk hc = host_chunks[instance];
    return host_vertex(hc, p * float3(hc.axis0.w, hc.axis1.w, hc.axis2.w), n);
}

[shader("vertex")]
VertexOut host_hull_vs(uint vertex: SV_VertexID)
{
    HullVertex v = host_hull_verts[vertex];
    return host_vertex(host_chunks[uint(v.pos.w)], v.pos.xyz, v.normal.xyz);
}

[shader("vertex")]
VertexOut impactor_vs(uint vertex: SV_VertexID, uint instance: SV_InstanceID)
{
    Impactor imp = impactors[instance];
    Quat q = rotation_of(imp.rotation, imp.rotation_err);
    float3 center = imp.position.xyz + imp.position_err.xyz;
    float3 p;
    float3 n;
    VertexOut o;
    if (imp.shape.x > 0.5)
    {
        if (vertex >= 36)
        {
            o.position = float4(0.0, 0.0, 2.0, 1.0); // beyond the far plane: clipped
            o.normal = float3(0.0, 0.0, 1.0);
            o.color = float3(0.0);
            o.world = float3(0.0);
            return o;
        }
        box_corner(vertex, p, n);
        p *= imp.half.xyz;
    }
    else
    {
        uint quad = vertex / 6;
        uint corner = vertex % 6;
        uint2 offs[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 0), uint2(1, 1), uint2(0, 1) };
        uint i = quad % SPHERE_LON + offs[corner].x;
        uint j = quad / SPHERE_LON + offs[corner].y;
        float theta = PI * float(j) / float(SPHERE_LAT);
        float phi = 2.0 * PI * float(i) / float(SPHERE_LON);
        n = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        p = n * imp.shape.y;
    }
    float3 world = center + rotate(q, p);
    o.position = clip_of(world);
    o.normal = rotate(q, n);
    o.color = float3(0.78, 0.80, 0.84);
    o.world = world;
    return o;
}

[shader("vertex")]
VertexOut ground_vs(uint vertex: SV_VertexID)
{
    float2 quad[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float3 world = float3(camera.ground.zw + quad[vertex] * camera.ground.y, camera.ground.x);
    VertexOut o;
    o.position = clip_of(world);
    o.normal = float3(0.0, 0.0, 1.0);
    o.color = float3(0.0);
    o.world = world;
    return o;
}

[shader("fragment")]
float4 lit_fs(VertexOut input) : SV_Target
{
    float diffuse = max(dot(normalize(input.normal), normalize(camera.light.xyz)), 0.0);
    return float4(input.color * (0.35 + 0.65 * diffuse), 1.0);
}

[shader("fragment")]
float4 ground_fs(VertexOut input) : SV_Target
{
    int2 cell = int2(floor(input.world.xy));
    float checker = ((cell.x + cell.y) & 1) == 0 ? 1.0 : 0.0;
    float3 color = lerp(float3(0.16, 0.17, 0.19), float3(0.20, 0.21, 0.23), checker);
    // Fade to the background towards the edge of the plane.
    float edge = saturate(length(input.world.xy - camera.ground.zw) / camera.ground.y);
    return float4(lerp(color, float3(0.05, 0.06, 0.08), edge * edge), 1.0);
}
