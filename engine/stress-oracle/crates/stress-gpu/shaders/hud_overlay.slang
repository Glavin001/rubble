// A text overlay: a rectangle of the window shows the host's RGBA8 pixels (one packed
// uint per pixel, straight alpha), blended over the frame.

struct Hud
{
    float4 rect;   // x, y (top-left, pixels), width, height
    float4 target; // window width, height
};

[[vk::binding(0, 0)]] ConstantBuffer<Hud> hud;
[[vk::binding(1, 0)]] StructuredBuffer<uint> pixels;

struct VertexOut
{
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOut hud_vs(uint vertex: SV_VertexID)
{
    float2 quad[6] = { float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1) };
    float2 p = hud.rect.xy + quad[vertex] * hud.rect.zw;
    VertexOut o;
    o.position = float4(2.0 * p.x / hud.target.x - 1.0, 1.0 - 2.0 * p.y / hud.target.y, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 hud_fs(VertexOut input) : SV_Target
{
    uint2 q = uint2(input.position.xy - hud.rect.xy);
    uint w = uint(hud.rect.z);
    uint c = pixels[min(q.y, uint(hud.rect.w) - 1) * w + min(q.x, w - 1)];
    return float4(float(c & 0xff), float((c >> 8) & 0xff), float((c >> 16) & 0xff), float(c >> 24)) / 255.0;
}
