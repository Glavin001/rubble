#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct Quat_0
{
    float w_0;
    float x_0;
    float y_0;
    float z_0;
};

Quat_0 quat_of_0(float4 q_0)
{
    thread Quat_0 r_0;
    (&r_0)->x_0 = q_0.x;
    (&r_0)->y_0 = q_0.y;
    (&r_0)->z_0 = q_0.z;
    (&r_0)->w_0 = q_0.w;
    return r_0;
}

float3 safe_normalize_0(float3 v_0)
{
    float n_0 = length(v_0);
    float3 _S1;
    if(n_0 > 1.00000000317107685e-30f)
    {
        _S1 = v_0 / float3(n_0) ;
    }
    else
    {
        _S1 = float3(0.0f) ;
    }
    return _S1;
}

Quat_0 from_axis_angle_0(float3 axis_0, float angle_0)
{
    float3 a_0 = safe_normalize_0(axis_0);
    float _S2 = 0.5f * angle_0;
    float s_0 = sin(_S2);
    thread Quat_0 q_1;
    (&q_1)->w_0 = cos(_S2);
    (&q_1)->x_0 = a_0.x * s_0;
    (&q_1)->y_0 = a_0.y * s_0;
    (&q_1)->z_0 = a_0.z * s_0;
    return q_1;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_1, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_1->w_0 * o_0->w_0 - a_1->x_0 * o_0->x_0 - a_1->y_0 * o_0->y_0 - a_1->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_1->w_0 * o_0->x_0 + a_1->x_0 * o_0->w_0 + a_1->y_0 * o_0->z_0 - a_1->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_1->w_0 * o_0->y_0 - a_1->x_0 * o_0->z_0 + a_1->y_0 * o_0->w_0 + a_1->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_1->w_0 * o_0->z_0 + a_1->x_0 * o_0->y_0 - a_1->y_0 * o_0->x_0 + a_1->z_0 * o_0->w_0;
    return r_1;
}

Quat_0 conj_of_0(const Quat_0 thread* q_2)
{
    thread Quat_0 c_0;
    (&c_0)->w_0 = q_2->w_0;
    (&c_0)->x_0 = - q_2->x_0;
    (&c_0)->y_0 = - q_2->y_0;
    (&c_0)->z_0 = - q_2->z_0;
    return c_0;
}

float3 rotate_0(const Quat_0 thread* q_3, float3 v_1)
{
    float3 qv_0 = float3(q_3->x_0, q_3->y_0, q_3->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_3->w_0)  + cross(qv_0, t_0);
}

Quat_0 normalized_0(const Quat_0 thread* q_4)
{
    float n_1 = sqrt(q_4->w_0 * q_4->w_0 + q_4->x_0 * q_4->x_0 + q_4->y_0 * q_4->y_0 + q_4->z_0 * q_4->z_0);
    thread Quat_0 r_2;
    (&r_2)->w_0 = q_4->w_0 / n_1;
    (&r_2)->x_0 = q_4->x_0 / n_1;
    (&r_2)->y_0 = q_4->y_0 / n_1;
    (&r_2)->z_0 = q_4->z_0 / n_1;
    return r_2;
}

float4 quat_vec_0(const Quat_0 thread* q_5)
{
    return float4(q_5->x_0, q_5->y_0, q_5->z_0, q_5->w_0);
}

struct ShapeParams_0
{
    uint collider_count_0;
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    ShapeParams_0 constant* params_0;
    packed_float4 device* rest_0;
    packed_uint4 device* collider_info_0;
    packed_float4 device* state_0;
    packed_float4 device* colliders_0;
};

[[kernel]] void shapes_deform(uint3 id_0 [[thread_position_in_grid]], ShapeParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* rest_1 [[buffer(1)]], packed_uint4 device* collider_info_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], packed_float4 device* colliders_1 [[buffer(4)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->rest_0 = rest_1;
    (&kernelContext_0)->collider_info_0 = collider_info_1;
    (&kernelContext_0)->state_0 = state_1;
    (&kernelContext_0)->colliders_0 = colliders_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_1->collider_count_0))
    {
        return;
    }
    uint _S3 = 4U * i_0;
    float4 _S4 = float4(*((&kernelContext_0)->rest_0+_S3)) ;
    uint4 _S5 = uint4(*((&kernelContext_0)->collider_info_0+2U * i_0)) ;
    bool _S6;
    if((_S4.w) == 0.0f)
    {
        _S6 = true;
    }
    else
    {
        _S6 = (_S5.z) != 2U;
    }
    if(_S6)
    {
        _S6 = true;
    }
    else
    {
        _S6 = (_S5.x) == 4294967295U;
    }
    if(_S6)
    {
        return;
    }
    uint _S7 = 4U * _S5.x;
    float3 u_0 = (float4(*((&kernelContext_0)->state_0+_S7)) ).xyz;
    float3 th_0 = (float4(*((&kernelContext_0)->state_0+(_S7 + 1U))) ).xyz;
    uint _S8 = _S3 + 1U;
    Quat_0 to_body_0 = quat_of_0(float4(*((&kernelContext_0)->rest_0+_S8)) );
    Quat_0 _S9 = from_axis_angle_0(th_0, length(th_0));
    thread Quat_0 _S10 = to_body_0;
    thread Quat_0 _S11 = _S9;
    Quat_0 _S12 = quat_mul_0(&_S10, &_S11);
    thread Quat_0 _S13 = to_body_0;
    Quat_0 _S14 = conj_of_0(&_S13);
    thread Quat_0 _S15 = _S12;
    thread Quat_0 _S16 = _S14;
    Quat_0 _S17 = quat_mul_0(&_S15, &_S16);
    uint _S18 = _S3 + 2U;
    float3 p0_0 = (float4(*((&kernelContext_0)->rest_0+_S18)) ).xyz;
    Quat_0 r0_0 = quat_of_0(float4(*((&kernelContext_0)->rest_0+(_S3 + 3U))) );
    float3 _S19 = _S4.xyz;
    thread Quat_0 _S20 = to_body_0;
    float3 _S21 = rotate_0(&_S20, u_0);
    float3 _S22 = _S19 + _S21;
    float3 _S23 = p0_0 - _S19;
    thread Quat_0 _S24 = _S17;
    float3 _S25 = rotate_0(&_S24, _S23);
    *((&kernelContext_0)->colliders_0+_S8) = packed_float4(float4(_S22 + _S25, (float4(*((&kernelContext_0)->colliders_0+_S8)) ).w)) ;
    packed_float4 device* _S26 = (&kernelContext_0)->colliders_0+_S18;
    thread Quat_0 _S27 = _S17;
    thread Quat_0 _S28 = r0_0;
    Quat_0 _S29 = quat_mul_0(&_S27, &_S28);
    thread Quat_0 _S30 = _S29;
    Quat_0 _S31 = normalized_0(&_S30);
    thread Quat_0 _S32 = _S31;
    float4 _S33 = quat_vec_0(&_S32);
    *_S26 = packed_float4(_S33) ;
    return;
}

