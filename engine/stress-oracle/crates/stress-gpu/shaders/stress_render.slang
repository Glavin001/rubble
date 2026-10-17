// Draws the stress solver's chunks straight from its GPU state: every chunk an
// oriented box at its cluster pose times (rest centre + u), turned by the hidden
// rotation theta (as stress-ref's `chunk_box`), coloured by its largest bond stress.
// Vertices come from the vertex index (36 per box): no vertex buffers.

struct Camera
{
    float4 view_proj[4]; // column-major view-projection
    float4 light;        // xyz: direction towards the light
    float stress_max;    // stress at full colour (Pa)
    float exaggerate;    // hidden displacement drawn at this multiple
    float pad0;
    float pad1;
};

struct RenderChunk
{
    float4 center;    // rest centre (body frame)
    float4 half;      // half extents
    float4 rot0;      // chunk rotation rows (body frame)
    float4 rot1;
    float4 rot2;
    float4 pose_pos;  // cluster pose: position
    float4 pose_rot0; // cluster pose: rotation rows
    float4 pose_rot1;
    float4 pose_rot2;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(1, 0)]] StructuredBuffer<RenderChunk> render_chunks;
[[vk::binding(2, 0)]] StructuredBuffer<float4> state; // 4 per chunk: u (w: stress), theta, v, w

struct VertexOut
{
    float4 position : SV_Position;
    float3 normal;
    float3 color;
};

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

float3 apply_rows(float4 r0, float4 r1, float4 r2, float3 x)
{
    return float3(dot(r0.xyz, x), dot(r1.xyz, x), dot(r2.xyz, x));
}

// Rotation by the rotation vector theta (Rodrigues).
float3 rotate_by(float3 theta, float3 x)
{
    float angle = length(theta);
    if (angle < 1e-12)
        return x + cross(theta, x);
    float3 k = theta / angle;
    return x * cos(angle) + cross(k, x) * sin(angle) + k * dot(k, x) * (1.0 - cos(angle));
}

[shader("vertex")]
VertexOut stress_vs(uint vertex: SV_VertexID, uint instance: SV_InstanceID)
{
    uint face = vertex / 6;
    uint corner = vertex % 6;
    uint axis = face / 2;
    float sign = (face % 2 == 0) ? 1.0 : -1.0;
    float2 quad[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float2 st = quad[corner];
    if (sign < 0.0)
        st.x = -st.x;
    float3 n = float3(0.0);
    float3 p = float3(0.0);
    if (axis == 0) { n = float3(sign, 0, 0); p = float3(sign, st.x, st.y); }
    else if (axis == 1) { n = float3(0, sign, 0); p = float3(st.y, sign, st.x); }
    else { n = float3(0, 0, sign); p = float3(st.x, st.y, sign); }

    RenderChunk rc = render_chunks[instance];
    float4 u = state[4 * instance + 0];
    float3 theta = state[4 * instance + 1].xyz * camera.exaggerate;
    // Body frame: rest centre plus hidden displacement, box turned by chunk then hidden rotation.
    float3 local = apply_rows(rc.rot0, rc.rot1, rc.rot2, p * rc.half.xyz);
    float3 body = rc.center.xyz + u.xyz * camera.exaggerate + rotate_by(theta, local);
    float3 normal_body = rotate_by(theta, apply_rows(rc.rot0, rc.rot1, rc.rot2, n));
    float3 world = rc.pose_pos.xyz + apply_rows(rc.pose_rot0, rc.pose_rot1, rc.pose_rot2, body);

    VertexOut o;
    o.position = camera.view_proj[0] * world.x + camera.view_proj[1] * world.y + camera.view_proj[2] * world.z + camera.view_proj[3];
    o.normal = apply_rows(rc.pose_rot0, rc.pose_rot1, rc.pose_rot2, normal_body);
    o.color = ramp(u.w / camera.stress_max);
    return o;
}

[shader("fragment")]
float4 stress_fs(VertexOut input) : SV_Target
{
    float diffuse = max(dot(normalize(input.normal), normalize(camera.light.xyz)), 0.0);
    return float4(input.color * (0.35 + 0.65 * diffuse), 1.0);
}
