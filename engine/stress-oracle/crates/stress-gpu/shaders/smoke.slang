// Toolchain smoke test: a 3-vector kernel (cross product, fused multiply-add,
// sqrt) compiled from Slang to WGSL and run through wgpu, checked against the CPU.

struct Params
{
    uint count;
    float scale;
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] StructuredBuffer<float4> a;
[[vk::binding(2, 0)]] StructuredBuffer<float4> b;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> result;

[shader("compute")]
[numthreads(64, 1, 1)]
void smoke(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
        return;
    float3 c = cross(a[i].xyz, b[i].xyz) * params.scale;
    result[i] = float4(c, sqrt(dot(c, c)));
}
