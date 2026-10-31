#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Sym_0
{
    float3 d_0;
    float3 o_0;
};

float2 crush_invariants_0(const Sym_0 thread* v_0, float volume_0)
{
    float inv_0 = 1.0f / volume_0;
    float xx_0 = v_0->d_0.x * inv_0;
    float yy_0 = v_0->d_0.y * inv_0;
    float zz_0 = v_0->d_0.z * inv_0;
    float xy_0 = v_0->o_0.x * inv_0;
    float xz_0 = v_0->o_0.y * inv_0;
    float yz_0 = v_0->o_0.z * inv_0;
    float p_0 = - (xx_0 + yy_0 + zz_0) / 3.0f;
    float dx_0 = xx_0 + p_0;
    float dy_0 = yy_0 + p_0;
    float dz_0 = zz_0 + p_0;
    return float2(p_0, sqrt(1.5f * max(dx_0 * dx_0 + dy_0 * dy_0 + dz_0 * dz_0 + 2.0f * (xy_0 * xy_0 + xz_0 * xz_0 + yz_0 * yz_0), 0.0f)));
}

float crush_excess_0(float2 pq_0, float4 m0_0)
{
    float _S1 = pq_0.x;
    if(!(_S1 > 0.0f))
    {
        return 0.0f;
    }
    float _S2 = max(pq_0.y - (m0_0.y + m0_0.z * _S1), _S1 - m0_0.x);
    float _S3;
    if(_S2 > 0.0f)
    {
        _S3 = _S2;
    }
    else
    {
        _S3 = 0.0f;
    }
    return _S3;
}

float4 crush_step_0(const Sym_0 thread* v_1, float volume_1, float dt_0, float4 m0_1, float viscosity_0, float damage_0)
{
    float2 _S4 = crush_invariants_0(v_1, volume_1);
    float excess_0 = crush_excess_0(_S4, m0_1);
    if(!(excess_0 > 0.0f))
    {
        return float4(damage_0, _S4, 0.0f);
    }
    float d_1 = damage_0 + excess_0 * excess_0 * dt_0 / (viscosity_0 * m0_1.w);
    if(d_1 >= 1.0f)
    {
        return float4(1.0f, _S4, 1.0f);
    }
    return float4(d_1, _S4, 0.0f);
}

struct TestParams_0
{
    uint count_0;
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    TestParams_0 constant* params_0;
    packed_float4 device* inputs_0;
    packed_float4 device* outputs_0;
};

[[kernel]] void crush_step_test(uint3 id_0 [[thread_position_in_grid]], TestParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* inputs_1 [[buffer(1)]], packed_float4 device* outputs_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->inputs_0 = inputs_1;
    (&kernelContext_0)->outputs_0 = outputs_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_1->count_0))
    {
        return;
    }
    uint _S5 = 4U * i_0;
    float4 _S6 = float4(*((&kernelContext_0)->inputs_0+_S5)) ;
    float4 _S7 = float4(*((&kernelContext_0)->inputs_0+(_S5 + 1U))) ;
    float4 _S8 = float4(*((&kernelContext_0)->inputs_0+(_S5 + 2U))) ;
    float4 _S9 = float4(*((&kernelContext_0)->inputs_0+(_S5 + 3U))) ;
    thread Sym_0 v_2;
    (&v_2)->d_0 = _S6.xyz;
    (&v_2)->o_0 = _S7.xyz;
    packed_float4 device* _S10 = (&kernelContext_0)->outputs_0+i_0;
    float _S11 = _S6.w;
    float _S12 = _S7.w;
    float _S13 = _S9.x;
    float _S14 = _S9.y;
    thread Sym_0 _S15 = v_2;
    float4 _S16 = crush_step_0(&_S15, _S11, _S12, _S8, _S13, _S14);
    *_S10 = packed_float4(_S16) ;
    return;
}

