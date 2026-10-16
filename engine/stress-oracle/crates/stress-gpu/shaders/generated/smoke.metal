#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Params_0
{
    uint count_0;
    float scale_0;
};

struct KernelContext_0
{
    Params_0 constant* params_0;
    packed_float4 device* a_0;
    packed_float4 device* b_0;
    packed_float4 device* result_0;
};

[[kernel]] void smoke(uint3 id_0 [[thread_position_in_grid]], Params_0 constant* params_1 [[buffer(0)]], packed_float4 device* a_1 [[buffer(1)]], packed_float4 device* b_1 [[buffer(2)]], packed_float4 device* result_1 [[buffer(3)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->a_0 = a_1;
    (&kernelContext_0)->b_0 = b_1;
    (&kernelContext_0)->result_0 = result_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_1->count_0))
    {
        return;
    }
    float3 c_0 = cross((float4(*((&kernelContext_0)->a_0+i_0)) ).xyz, (float4(*((&kernelContext_0)->b_0+i_0)) ).xyz) * float3((&kernelContext_0)->params_0->scale_0) ;
    *((&kernelContext_0)->result_0+i_0) = packed_float4(float4(c_0, sqrt(dot(c_0, c_0)))) ;
    return;
}

