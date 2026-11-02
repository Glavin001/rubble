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

Quat_0 conj_0(const Quat_0 thread* q_1)
{
    thread Quat_0 c_0;
    (&c_0)->w_0 = q_1->w_0;
    (&c_0)->x_0 = - q_1->x_0;
    (&c_0)->y_0 = - q_1->y_0;
    (&c_0)->z_0 = - q_1->z_0;
    return c_0;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_0, const Quat_0 thread* o_0)
{
    thread Quat_0 r_1;
    (&r_1)->w_0 = a_0->w_0 * o_0->w_0 - a_0->x_0 * o_0->x_0 - a_0->y_0 * o_0->y_0 - a_0->z_0 * o_0->z_0;
    (&r_1)->x_0 = a_0->w_0 * o_0->x_0 + a_0->x_0 * o_0->w_0 + a_0->y_0 * o_0->z_0 - a_0->z_0 * o_0->y_0;
    (&r_1)->y_0 = a_0->w_0 * o_0->y_0 - a_0->x_0 * o_0->z_0 + a_0->y_0 * o_0->w_0 + a_0->z_0 * o_0->x_0;
    (&r_1)->z_0 = a_0->w_0 * o_0->z_0 + a_0->x_0 * o_0->y_0 - a_0->y_0 * o_0->x_0 + a_0->z_0 * o_0->w_0;
    return r_1;
}

Quat_0 normalized_0(const Quat_0 thread* q_2)
{
    float n_0 = sqrt(q_2->w_0 * q_2->w_0 + q_2->x_0 * q_2->x_0 + q_2->y_0 * q_2->y_0 + q_2->z_0 * q_2->z_0);
    thread Quat_0 r_2;
    (&r_2)->w_0 = q_2->w_0 / n_0;
    (&r_2)->x_0 = q_2->x_0 / n_0;
    (&r_2)->y_0 = q_2->y_0 / n_0;
    (&r_2)->z_0 = q_2->z_0 / n_0;
    return r_2;
}

Quat_0 rot_from_rubble_0(const Quat_0 thread* q_3)
{
    thread Quat_0 axes_0;
    (&axes_0)->w_0 = 0.70710676908493042f;
    (&axes_0)->x_0 = 0.70710676908493042f;
    (&axes_0)->y_0 = 0.0f;
    (&axes_0)->z_0 = 0.0f;
    thread Quat_0 _S1 = axes_0;
    Quat_0 _S2 = quat_mul_0(&_S1, q_3);
    thread Quat_0 _S3 = _S2;
    Quat_0 _S4 = normalized_0(&_S3);
    return _S4;
}

float3 rotate_0(const Quat_0 thread* q_4, float3 v_0)
{
    float3 qv_0 = float3(q_4->x_0, q_4->y_0, q_4->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_4->w_0)  + cross(qv_0, t_0);
}

float3 inverse_rotate_0(const Quat_0 thread* q_5, float3 v_1)
{
    thread Quat_0 c_1;
    (&c_1)->w_0 = q_5->w_0;
    (&c_1)->x_0 = - q_5->x_0;
    (&c_1)->y_0 = - q_5->y_0;
    (&c_1)->z_0 = - q_5->z_0;
    thread Quat_0 _S5 = c_1;
    float3 _S6 = rotate_0(&_S5, v_1);
    return _S6;
}

float3 from_rubble_0(float3 v_2)
{
    return float3(v_2.x, - v_2.z, v_2.y);
}

struct BlastParams_0
{
    uint island_count_0;
    uint max_iterations_0;
    uint warm_0;
    uint skip_settled_0;
    float tolerance_0;
    float recip_length_0;
    uint node_count_0;
    uint bond_count_0;
    float linear_scale_0;
    float angular_scale_0;
    float bend_gain_max_0;
    uint fibres_0;
    float dt_0;
    float damage_rate_0;
    uint apply_damage_0;
    float force_tolerance_0;
    float4 gravity_0;
    uint w_alive_0;
    uint a_dirty_0;
    uint a_counters_0;
    uint wide_min_0;
    uint wide_rows_0;
    uint wide_blocks_0;
    uint opaque_zero_0;
    uint first_polynomial_0;
    uint rotation_0;
    uint node_vectors_0;
    uint carry_0;
    uint section_bending_0;
    uint chunk_loads_0;
};

struct BlastBond_natural_0
{
    uint n0_0;
    uint n1_0;
    float s_0;
    float health_0;
    packed_float4 o0_0;
    packed_float4 o1_0;
};

struct KernelContext_0
{
    BlastParams_0 constant* params_0;
    uint device* work_0;
    packed_float4 device* frames_0;
    packed_float4 device* bodies_0;
    packed_float4 device* node_frame_0;
    packed_float4 device* loads_0;
    packed_float4 device* accel_0;
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* x_1;
    atomic<uint> device* topo_0;
    atomic<uint> device* atom_0;
};

[[kernel]] void blast_loads(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], uint device* work_1 [[buffer(1)]], packed_float4 device* frames_1 [[buffer(3)]], packed_float4 device* bodies_1 [[buffer(4)]], packed_float4 device* node_frame_1 [[buffer(6)]], packed_float4 device* loads_1 [[buffer(5)]], packed_float4 device* accel_1 [[buffer(7)]], BlastBond_natural_0 device* bonds_1 [[buffer(8)]], packed_float4 device* x_2 [[buffer(10)]], atomic<uint> device* topo_1 [[buffer(9)]], atomic<uint> device* atom_1 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->work_0 = work_1;
    (&kernelContext_0)->frames_0 = frames_1;
    (&kernelContext_0)->bodies_0 = bodies_1;
    (&kernelContext_0)->node_frame_0 = node_frame_1;
    (&kernelContext_0)->loads_0 = loads_1;
    (&kernelContext_0)->accel_0 = accel_1;
    (&kernelContext_0)->bonds_0 = bonds_1;
    (&kernelContext_0)->x_1 = x_2;
    (&kernelContext_0)->topo_0 = topo_1;
    (&kernelContext_0)->atom_0 = atom_1;
    uint k_0 = id_0.x;
    if(k_0 >= (params_1->node_count_0))
    {
        return;
    }
    uint device* _S7 = (&kernelContext_0)->work_0+k_0;
    uint b_0 = 7U * *_S7;
    uint _S8 = 2U * *_S7;
    thread Quat_0 _S9 = quat_of_0(float4(*((&kernelContext_0)->frames_0+(_S8 + 1U))) );
    Quat_0 _S10 = conj_0(&_S9);
    thread Quat_0 _S11 = quat_of_0(float4(*((&kernelContext_0)->bodies_0+(b_0 + 1U))) );
    Quat_0 _S12 = rot_from_rubble_0(&_S11);
    thread Quat_0 _S13 = _S12;
    thread Quat_0 _S14 = _S10;
    Quat_0 _S15 = quat_mul_0(&_S13, &_S14);
    thread Quat_0 _S16 = _S15;
    Quat_0 _S17 = normalized_0(&_S16);
    uint _S18 = 6U * k_0;
    float4 _S19 = float4(*((&kernelContext_0)->node_frame_0+_S18)) ;
    float3 f_0 = (float4(*((&kernelContext_0)->loads_0+k_0)) ).yzw;
    float3 _S20 = (&kernelContext_0)->params_0->gravity_0.xyz;
    float _S21 = _S19.w;
    bool _S22 = _S21 > 0.0f;
    float3 alpha_0;
    if(_S22)
    {
        alpha_0 = f_0 / float3(_S21) ;
    }
    else
    {
        alpha_0 = float3(0.0f) ;
    }
    float3 a_1 = _S20 + alpha_0;
    float3 _S23 = float3(0.0f) ;
    bool _S24;
    if(((&kernelContext_0)->params_0->chunk_loads_0) != 0U)
    {
        _S24 = _S22;
    }
    else
    {
        _S24 = false;
    }
    float3 a_2;
    if(_S24)
    {
        float3 torque_0 = (float4(*((&kernelContext_0)->node_frame_0+(_S18 + 4U))) ).xyz + (float4(*((&kernelContext_0)->node_frame_0+(_S18 + 5U))) ).xyz / float3((&kernelContext_0)->params_0->dt_0) ;
        float3 a_3 = a_1 + ((float4(*((&kernelContext_0)->node_frame_0+(_S18 + 2U))) ).xyz + (float4(*((&kernelContext_0)->node_frame_0+(_S18 + 3U))) ).xyz / float3((&kernelContext_0)->params_0->dt_0) ) / float3(_S21) ;
        float inertia_0 = (float4(*((&kernelContext_0)->node_frame_0+(_S18 + 1U))) ).x;
        if(inertia_0 > 0.0f)
        {
            alpha_0 = - torque_0 / float3(inertia_0) ;
        }
        else
        {
            alpha_0 = _S23;
        }
        a_2 = a_3;
    }
    else
    {
        alpha_0 = _S23;
        a_2 = a_1;
    }
    thread Quat_0 _S25 = _S17;
    float3 _S26 = inverse_rotate_0(&_S25, a_2);
    float3 _S27 = from_rubble_0((float4(*((&kernelContext_0)->bodies_0+(b_0 + 4U))) ).xyz);
    thread Quat_0 _S28 = _S17;
    float3 _S29 = inverse_rotate_0(&_S28, _S27);
    float3 a_local_0 = _S26 - cross(_S29, cross(_S29, _S19.xyz - (float4(*((&kernelContext_0)->frames_0+_S8)) ).xyz));
    uint _S30 = 2U * k_0;
    packed_float4 device* _S31 = (&kernelContext_0)->accel_0+_S30;
    thread Quat_0 _S32 = _S17;
    float3 _S33 = inverse_rotate_0(&_S32, alpha_0);
    *_S31 = packed_float4(float4(_S33, 0.0f)) ;
    *((&kernelContext_0)->accel_0+(_S30 + 1U)) = packed_float4(float4(a_local_0, 0.0f)) ;
    return;
}

uint topo_total_0(uint node_count_1)
{
    return 10U * node_count_1 + 3U * ((node_count_1 + 1023U) / 1024U);
}

uint topo_dirty_0(uint node_count_2)
{
    return topo_total_0(node_count_2) + 4U;
}

[[kernel]] void blast_sync(uint3 id_1 [[thread_position_in_grid]], BlastParams_0 constant* params_2 [[buffer(0)]], uint device* work_2 [[buffer(1)]], packed_float4 device* frames_2 [[buffer(3)]], packed_float4 device* bodies_2 [[buffer(4)]], packed_float4 device* node_frame_2 [[buffer(6)]], packed_float4 device* loads_2 [[buffer(5)]], packed_float4 device* accel_2 [[buffer(7)]], BlastBond_natural_0 device* bonds_2 [[buffer(8)]], packed_float4 device* x_3 [[buffer(10)]], atomic<uint> device* topo_2 [[buffer(9)]], atomic<uint> device* atom_2 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_2;
    (&kernelContext_1)->work_0 = work_2;
    (&kernelContext_1)->frames_0 = frames_2;
    (&kernelContext_1)->bodies_0 = bodies_2;
    (&kernelContext_1)->node_frame_0 = node_frame_2;
    (&kernelContext_1)->loads_0 = loads_2;
    (&kernelContext_1)->accel_0 = accel_2;
    (&kernelContext_1)->bonds_0 = bonds_2;
    (&kernelContext_1)->x_1 = x_3;
    (&kernelContext_1)->topo_0 = topo_2;
    (&kernelContext_1)->atom_0 = atom_2;
    uint j_0 = id_1.x;
    bool _S34;
    if(j_0 >= (params_2->bond_count_0))
    {
        _S34 = true;
    }
    else
    {
        _S34 = (*((&kernelContext_1)->work_0+((&kernelContext_1)->params_0->w_alive_0 + j_0))) != 0U;
    }
    if(_S34)
    {
        return;
    }
    if(!(((*((&kernelContext_1)->bonds_0+j_0)).health_0) > 0.0f))
    {
        return;
    }
    ((&kernelContext_1)->bonds_0+j_0)->health_0 = 0.0f;
    uint _S35 = j_0 * 2U;
    packed_float4 _S36 = packed_float4(float4(0.0f) ) ;
    *((&kernelContext_1)->x_1+_S35) = _S36;
    *((&kernelContext_1)->x_1+(_S35 + 1U)) = _S36;
    atomic_store_explicit((&kernelContext_1)->topo_0+topo_dirty_0((&kernelContext_1)->params_0->node_count_0), 1U, memory_order_relaxed);
    return;
}

[[kernel]] void blast_breaks(uint3 id_2 [[thread_position_in_grid]], BlastParams_0 constant* params_3 [[buffer(0)]], uint device* work_3 [[buffer(1)]], packed_float4 device* frames_3 [[buffer(3)]], packed_float4 device* bodies_3 [[buffer(4)]], packed_float4 device* node_frame_3 [[buffer(6)]], packed_float4 device* loads_3 [[buffer(5)]], packed_float4 device* accel_3 [[buffer(7)]], BlastBond_natural_0 device* bonds_3 [[buffer(8)]], packed_float4 device* x_4 [[buffer(10)]], atomic<uint> device* topo_3 [[buffer(9)]], atomic<uint> device* atom_3 [[buffer(2)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_3;
    (&kernelContext_2)->work_0 = work_3;
    (&kernelContext_2)->frames_0 = frames_3;
    (&kernelContext_2)->bodies_0 = bodies_3;
    (&kernelContext_2)->node_frame_0 = node_frame_3;
    (&kernelContext_2)->loads_0 = loads_3;
    (&kernelContext_2)->accel_0 = accel_3;
    (&kernelContext_2)->bonds_0 = bonds_3;
    (&kernelContext_2)->x_1 = x_4;
    (&kernelContext_2)->topo_0 = topo_3;
    (&kernelContext_2)->atom_0 = atom_3;
    uint j_1 = id_2.x;
    bool _S37;
    if(j_1 >= (params_3->bond_count_0))
    {
        _S37 = true;
    }
    else
    {
        _S37 = (*((&kernelContext_2)->work_0+((&kernelContext_2)->params_0->w_alive_0 + j_1))) == 0U;
    }
    if(_S37)
    {
        return;
    }
    BlastBond_natural_0 device* _S38 = (&kernelContext_2)->bonds_0+j_1;
    BlastBond_natural_0 bond_0 = *_S38;
    if(((*_S38).health_0) > 0.0f)
    {
        return;
    }
    *((&kernelContext_2)->work_0+((&kernelContext_2)->params_0->w_alive_0 + j_1)) = 0U;
    uint _S39 = atomic_fetch_or_explicit((&kernelContext_2)->atom_0+((&kernelContext_2)->params_0->a_dirty_0 + *((&kernelContext_2)->work_0+bond_0.n0_0)), 1U, memory_order_relaxed);
    uint _S40 = atomic_fetch_add_explicit((&kernelContext_2)->atom_0+(&kernelContext_2)->params_0->a_counters_0, 1U, memory_order_relaxed);
    uint _S41 = atomic_fetch_add_explicit((&kernelContext_2)->atom_0+((&kernelContext_2)->params_0->a_counters_0 + 1U), 1U, memory_order_relaxed);
    return;
}

