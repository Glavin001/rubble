#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct pixelOutput_0
{
    float4 output_0 [[color(0)]];
};

struct Hud_0
{
    float4 rect_0;
    float4 target_0;
};

struct KernelContext_0
{
    Hud_0 constant* hud_0;
    uint device* pixels_0;
};

[[fragment]] pixelOutput_0 hud_fs(float4 position_0 [[position]], Hud_0 constant* hud_1 [[buffer(0)]], uint device* pixels_1 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->hud_0 = hud_1;
    (&kernelContext_0)->pixels_0 = pixels_1;
    uint2 _S1 = uint2(position_0.xy - hud_1->rect_0.xy);
    uint w_0 = uint(hud_1->rect_0.z);
    uint c_0 = pixels_1[min(_S1.y, uint(hud_1->rect_0.w) - 1U) * w_0 + min(_S1.x, w_0 - 1U)];
    pixelOutput_0 _S2 = { float4(float(c_0 & 255U), float((c_0 >> 8U) & 255U), float((c_0 >> 16U) & 255U), float(c_0 >> 24U)) / float4(255.0f)  };
    return _S2;
}

struct hud_vs_Result_0
{
    float4 position_1 [[position]];
};

struct VertexOut_0
{
    float4 position_2;
};

[[vertex]] hud_vs_Result_0 hud_vs(uint vertex_0 [[vertex_id]], Hud_0 constant* hud_2 [[buffer(0)]], uint device* pixels_2 [[buffer(1)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->hud_0 = hud_2;
    (&kernelContext_1)->pixels_0 = pixels_2;
    float2 _S3 = float2(0.0f, 0.0f);
    float2 _S4 = float2(1.0f, 1.0f);
    array<float2, int(6)> quad_0 = { { _S3, float2(1.0f, 0.0f), _S4, _S3, _S4, float2(0.0f, 1.0f) } };
    float2 p_0 = hud_2->rect_0.xy + quad_0[vertex_0] * hud_2->rect_0.zw;
    thread VertexOut_0 o_0;
    (&o_0)->position_2 = float4(2.0f * p_0.x / hud_2->target_0.x - 1.0f, 1.0f - 2.0f * p_0.y / hud_2->target_0.y, 0.0f, 1.0f);
    thread hud_vs_Result_0 _S5;
    (&_S5)->position_1 = o_0.position_2;
    return _S5;
}

