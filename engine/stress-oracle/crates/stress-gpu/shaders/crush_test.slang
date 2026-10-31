// Parity kernel: one crush-law step per thread (crush.slang `crush_step`), for comparison
// with stress-ref crush.rs `crush_step_f32` on the CPU.

import crush;

struct TestParams
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<TestParams> params;
// 4 per case: (virial diagonal, volume), (virial off-diagonal, dt), (cap, cohesion, slope,
// energy), (viscosity, damage, -, -).
[[vk::binding(1, 0)]] StructuredBuffer<float4> inputs;
// Per case: (damage, p, q, crushed).
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> outputs;

[shader("compute")]
[numthreads(64, 1, 1)]
void crush_step_test(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
        return;
    float4 a = inputs[4 * i];
    float4 b = inputs[4 * i + 1];
    float4 m0 = inputs[4 * i + 2];
    float4 c = inputs[4 * i + 3];
    Sym v;
    v.d = a.xyz;
    v.o = b.xyz;
    outputs[i] = crush_step(v, a.w, b.w, m0, c.x, c.y);
}
