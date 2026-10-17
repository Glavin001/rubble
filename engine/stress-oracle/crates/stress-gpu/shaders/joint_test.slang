// Parity kernel: one joint-law evaluation per thread, from given states and
// deformations, for comparison with stress-ref `JointModel::evaluate` on the CPU.

import joint;

struct TestParams
{
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<TestParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<JointMaterial> materials;
[[vk::binding(2, 0)]] StructuredBuffer<JointBond> bonds;
[[vk::binding(3, 0)]] StructuredBuffer<JointState> states_in;
// 2 per case: (d_lin, dt), (d_ang, fracture 0/1).
[[vk::binding(4, 0)]] StructuredBuffer<float4> inputs;
[[vk::binding(5, 0)]] RWStructuredBuffer<JointState> states_out;
// 5 per case: (force_lin, dissipated), (force_ang, overshoot), (secant lin, stored),
// (secant ang, disconnected), (tension, shear, compression, compressive force).
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> outputs;

[shader("compute")]
[numthreads(64, 1, 1)]
void joint_eval_test(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
        return;
    JointBond b = bonds[i];
    JointMaterial m = materials[b.ids.x];
    float4 a = inputs[2 * i];
    float4 c = inputs[2 * i + 1];
    JointResponse r = joint_evaluate(m, b, states_in[i], a.xyz, c.xyz, a.w, c.w != 0.0);
    float3 f_lin;
    float3 f_ang;
    secant_factors(b, r.state, a.xyz, f_lin, f_ang);
    states_out[i] = r.state;
    outputs[5 * i + 0] = float4(r.force_lin, r.dissipated);
    outputs[5 * i + 1] = float4(r.force_ang, r.overshoot);
    outputs[5 * i + 2] = float4(f_lin, r.stored);
    outputs[5 * i + 3] = float4(f_ang, r.disconnected ? 1.0 : 0.0);
    outputs[5 * i + 4] = float4(r.measures.tension, r.measures.shear, r.measures.compression, r.measures.compressive_force);
}
