#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
void box_corner_0(uint vertex_0, float3 thread* p_0, float3 thread* n_0)
{
    uint face_0 = vertex_0 / 6U;
    uint corner_0 = vertex_0 % 6U;
    uint axis_0 = face_0 / 2U;
    float sign_0;
    if((face_0 % 2U) == 0U)
    {
        sign_0 = 1.0f;
    }
    else
    {
        sign_0 = -1.0f;
    }
    float2 _S1 = float2(-1.0f, -1.0f);
    float2 _S2 = float2(1.0f, 1.0f);
    array<float2, int(6)> quad_0 = { { _S1, float2(1.0f, -1.0f), _S2, _S1, _S2, float2(-1.0f, 1.0f) } };
    thread float2 st_0 = quad_0[corner_0];
    if(sign_0 < 0.0f)
    {
        st_0.x = - st_0.x;
    }
    if(axis_0 == 0U)
    {
        *n_0 = float3(sign_0, 0.0f, 0.0f);
        *p_0 = float3(sign_0, st_0.x, st_0.y);
    }
    else
    {
        if(axis_0 == 1U)
        {
            *n_0 = float3(0.0f, sign_0, 0.0f);
            *p_0 = float3(st_0.y, sign_0, st_0.x);
        }
        else
        {
            *n_0 = float3(0.0f, 0.0f, sign_0);
            *p_0 = float3(st_0.x, st_0.y, sign_0);
        }
    }
    return;
}

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

Quat_0 normalized_0(const Quat_0 thread* q_1)
{
    float n_1 = sqrt(q_1->w_0 * q_1->w_0 + q_1->x_0 * q_1->x_0 + q_1->y_0 * q_1->y_0 + q_1->z_0 * q_1->z_0);
    thread Quat_0 r_1;
    (&r_1)->w_0 = q_1->w_0 / n_1;
    (&r_1)->x_0 = q_1->x_0 / n_1;
    (&r_1)->y_0 = q_1->y_0 / n_1;
    (&r_1)->z_0 = q_1->z_0 / n_1;
    return r_1;
}

Quat_0 rotation_of_0(float4 hi_0, float4 lo_0)
{
    thread Quat_0 _S3 = quat_of_0(hi_0 + lo_0);
    Quat_0 _S4 = normalized_0(&_S3);
    return _S4;
}

float3 safe_normalize_0(float3 v_0)
{
    float n_2 = length(v_0);
    float3 _S5;
    if(n_2 > 1.00000000317107685e-30f)
    {
        _S5 = v_0 / float3(n_2) ;
    }
    else
    {
        _S5 = float3(0.0f) ;
    }
    return _S5;
}

Quat_0 from_axis_angle_0(float3 axis_1, float angle_0)
{
    float3 a_0 = safe_normalize_0(axis_1);
    float _S6 = 0.5f * angle_0;
    float s_0 = sin(_S6);
    thread Quat_0 q_2;
    (&q_2)->w_0 = cos(_S6);
    (&q_2)->x_0 = a_0.x * s_0;
    (&q_2)->y_0 = a_0.y * s_0;
    (&q_2)->z_0 = a_0.z * s_0;
    return q_2;
}

float3 rotate_0(const Quat_0 thread* q_3, float3 v_1)
{
    float3 qv_0 = float3(q_3->x_0, q_3->y_0, q_3->z_0);
    float3 t_0 = cross(qv_0, v_1) * float3(2.0f) ;
    return v_1 + t_0 * float3(q_3->w_0)  + cross(qv_0, t_0);
}

struct ChunkStatic_natural_0
{
    packed_float4 center_0;
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
    packed_float4 scale_0;
    packed_uint4 info_0;
    packed_uint4 load_range_0;
    packed_float4 half_0;
    packed_float4 crot0_0;
    packed_float4 crot1_0;
    packed_float4 crot2_0;
    packed_float4 cmat_0;
    packed_uint4 cinfo_0;
};

struct Island_natural_0
{
    packed_uint4 range_0;
    packed_uint4 info_1;
    packed_float4 com_0;
    packed_float4 inertia0_1;
    packed_float4 inertia1_1;
    packed_float4 inertia2_1;
    packed_float4 inv0_1;
    packed_float4 inv1_1;
    packed_float4 inv2_1;
    packed_float4 wcom_0;
    packed_float4 winv0_0;
    packed_float4 winv1_0;
    packed_float4 winv2_0;
    packed_float4 rotation_0;
    packed_float4 position_0;
    packed_float4 position_err_0;
    packed_float4 velocity_0;
    packed_float4 velocity_err_0;
    packed_float4 angular_velocity_0;
    uint done_0;
    uint coverage_0;
    uint wide_groups_0;
    uint done_pad_0;
    packed_uint4 probes_0;
    packed_float4 energy_0;
    packed_float4 rotation_err_0;
    packed_float4 momentum_0;
    packed_float4 momentum_err_0;
};

struct JointState_0
{
    float damage_0;
    float crush_0;
    float kappa_0;
    float kappa_c_0;
    float ductility_0;
    float ductility_c_0;
    float life_0;
    float plastic_x_0;
    float plastic_y_0;
    float plastic_t_0;
    float rebar_plastic_0;
    float rebar_slip0_0;
    float rebar_slip1_0;
    float rebar_work_0;
    float rebar_broken_0;
    float strain_rate_0;
    float governing_stress_0;
    float dissipated_0;
    float utilization_0;
    uint mode_0;
};

struct BondDyn_natural_0
{
    JointState_0 js_0;
    packed_float4 force_lin_0;
    packed_float4 force_ang_0;
    packed_float4 sums_0;
    packed_float4 comps_0;
    packed_uint4 events_0;
};

struct Camera_0
{
    array<float4, int(4)> view_proj_0;
    float4 light_0;
    float4 ground_0;
};

struct HostChunk_natural_0
{
    packed_float4 center_1;
    packed_float4 axis0_0;
    packed_float4 axis1_0;
    packed_float4 axis2_0;
};

struct Impactor_natural_0
{
    packed_float4 position_1;
    packed_float4 position_err_1;
    packed_float4 velocity_1;
    packed_float4 velocity_err_1;
    packed_float4 angular_velocity_1;
    packed_float4 rotation_1;
    packed_float4 inertia0_2;
    packed_float4 inertia1_2;
    packed_float4 inertia2_2;
    packed_float4 inv0_2;
    packed_float4 inv1_2;
    packed_float4 inv2_2;
    packed_float4 shape_0;
    packed_float4 half_1;
    packed_float4 mat_0;
    packed_float4 crush_1;
    packed_float4 geom_0;
    packed_float4 rotation_err_1;
    packed_float4 ledger_0;
    packed_uint4 cand_0;
    packed_float4 momentum_1;
    packed_float4 momentum_err_1;
};

struct KernelContext_0
{
    ChunkStatic_natural_0 device* chunks_0;
    Island_natural_0 device* islands_0;
    packed_float4 device* state_0;
    uint device* index_0;
    BondDyn_natural_0 device* bond_dyn_0;
    Camera_0 constant* camera_0;
    HostChunk_natural_0 device* host_chunks_0;
    Impactor_natural_0 device* impactors_0;
};

float4 clip_of_0(float3 world_0, KernelContext_0 thread* kernelContext_0)
{
    return kernelContext_0->camera_0->view_proj_0[int(0)] * float4(world_0.x)  + kernelContext_0->camera_0->view_proj_0[int(1)] * float4(world_0.y)  + kernelContext_0->camera_0->view_proj_0[int(2)] * float4(world_0.z)  + kernelContext_0->camera_0->view_proj_0[int(3)];
}

float3 ramp_0(float t_1)
{
    float3 c0_0 = float3(0.15000000596046448f, 0.25f, 0.85000002384185791f);
    float3 c1_0 = float3(0.10000000149011612f, 0.75f, 0.85000002384185791f);
    float3 c2_0 = float3(0.25f, 0.80000001192092896f, 0.30000001192092896f);
    float3 c3_0 = float3(0.94999998807907104f, 0.85000002384185791f, 0.20000000298023224f);
    float3 c4_0 = float3(0.89999997615814209f, 0.20000000298023224f, 0.15000000596046448f);
    float s_1 = saturate(t_1) * 4.0f;
    if(s_1 < 1.0f)
    {
        return mix(c0_0, c1_0, float3(s_1) );
    }
    if(s_1 < 2.0f)
    {
        return mix(c1_0, c2_0, float3((s_1 - 1.0f)) );
    }
    if(s_1 < 3.0f)
    {
        return mix(c2_0, c3_0, float3((s_1 - 2.0f)) );
    }
    return mix(c3_0, c4_0, float3((s_1 - 3.0f)) );
}

struct pixelOutput_0
{
    float4 output_0 [[color(0)]];
};

struct pixelInput_0
{
    float3 normal_0 [[user(_SLANG_ATTR)]];
    float3 color_0 [[user(_SLANG_ATTR_1)]];
    float3 world_1 [[user(_SLANG_ATTR_2)]];
};

[[fragment]] pixelOutput_0 lit_fs(pixelInput_0 _S7 [[stage_in]], float4 position_2 [[position]], ChunkStatic_natural_0 device* chunks_1 [[buffer(1)]], Island_natural_0 device* islands_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], uint device* index_1 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(5)]], Camera_0 constant* camera_1 [[buffer(0)]], HostChunk_natural_0 device* host_chunks_1 [[buffer(7)]], Impactor_natural_0 device* impactors_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->chunks_0 = chunks_1;
    (&kernelContext_1)->islands_0 = islands_1;
    (&kernelContext_1)->state_0 = state_1;
    (&kernelContext_1)->index_0 = index_1;
    (&kernelContext_1)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_1)->camera_0 = camera_1;
    (&kernelContext_1)->host_chunks_0 = host_chunks_1;
    (&kernelContext_1)->impactors_0 = impactors_1;
    pixelOutput_0 _S8 = { float4(_S7.color_0 * float3((0.34999999403953552f + 0.64999997615814209f * max(dot(normalize(_S7.normal_0), normalize(camera_1->light_0.xyz)), 0.0f))) , 1.0f) };
    return _S8;
}

struct pixelOutput_1
{
    float4 output_1 [[color(0)]];
};

struct pixelInput_1
{
    float3 normal_1 [[user(_SLANG_ATTR)]];
    float3 color_1 [[user(_SLANG_ATTR_1)]];
    float3 world_2 [[user(_SLANG_ATTR_2)]];
};

[[fragment]] pixelOutput_1 ground_fs(pixelInput_1 _S9 [[stage_in]], float4 position_3 [[position]], ChunkStatic_natural_0 device* chunks_2 [[buffer(1)]], Island_natural_0 device* islands_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], uint device* index_2 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(5)]], Camera_0 constant* camera_2 [[buffer(0)]], HostChunk_natural_0 device* host_chunks_2 [[buffer(7)]], Impactor_natural_0 device* impactors_2 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->chunks_0 = chunks_2;
    (&kernelContext_2)->islands_0 = islands_2;
    (&kernelContext_2)->state_0 = state_2;
    (&kernelContext_2)->index_0 = index_2;
    (&kernelContext_2)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_2)->camera_0 = camera_2;
    (&kernelContext_2)->host_chunks_0 = host_chunks_2;
    (&kernelContext_2)->impactors_0 = impactors_2;
    float2 _S10 = _S9.world_2.xy;
    int2 _S11 = int2(floor(_S10));
    float checker_0;
    if(((_S11.x + _S11.y) & int(1)) == int(0))
    {
        checker_0 = 1.0f;
    }
    else
    {
        checker_0 = 0.0f;
    }
    float edge_0 = saturate(length(_S10 - (&kernelContext_2)->camera_0->ground_0.zw) / (&kernelContext_2)->camera_0->ground_0.y);
    pixelOutput_1 _S12 = { float4(mix(mix(float3(0.15999999642372131f, 0.17000000178813934f, 0.18999999761581421f), float3(0.20000000298023224f, 0.20999999344348907f, 0.23000000417232513f), float3(checker_0) ), float3(0.05000000074505806f, 0.05999999865889549f, 0.07999999821186066f), float3((edge_0 * edge_0)) ), 1.0f) };
    return _S12;
}

struct chunk_vs_Result_0
{
    float4 position_4 [[position]];
    float3 normal_2 [[user(_SLANG_ATTR)]];
    float3 color_2 [[user(_SLANG_ATTR_1)]];
    float3 world_3 [[user(_SLANG_ATTR_2)]];
};

struct VertexOut_0
{
    float4 position_5;
    float3 normal_3;
    float3 color_3;
    float3 world_4;
};

[[vertex]] chunk_vs_Result_0 chunk_vs(uint vertex_1 [[vertex_id]], uint instance_0 [[instance_id]], uint base_instance_0 [[base_instance]], ChunkStatic_natural_0 device* chunks_3 [[buffer(1)]], Island_natural_0 device* islands_3 [[buffer(2)]], packed_float4 device* state_3 [[buffer(3)]], uint device* index_3 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(5)]], Camera_0 constant* camera_3 [[buffer(0)]], HostChunk_natural_0 device* host_chunks_3 [[buffer(7)]], Impactor_natural_0 device* impactors_3 [[buffer(6)]])
{
    VertexOut_0 _S13;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->chunks_0 = chunks_3;
    (&kernelContext_3)->islands_0 = islands_3;
    (&kernelContext_3)->state_0 = state_3;
    (&kernelContext_3)->index_0 = index_3;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_3)->camera_0 = camera_3;
    (&kernelContext_3)->host_chunks_0 = host_chunks_3;
    (&kernelContext_3)->impactors_0 = impactors_3;
    uint _S14 = instance_0 - base_instance_0;
    for(;;)
    {
        thread float3 p_1;
        thread float3 n_3;
        box_corner_0(vertex_1, &p_1, &n_3);
        ChunkStatic_natural_0 device* _S15 = (&kernelContext_3)->chunks_0+_S14;
        Island_natural_0 device* _S16 = (&kernelContext_3)->islands_0+(uint4(_S15->info_0) ).y;
        Quat_0 q_4 = rotation_of_0(float4(_S16->rotation_0) , float4(_S16->rotation_err_0) );
        uint _S17 = 4U * _S14;
        float3 th_0 = (float4(*((&kernelContext_3)->state_0+(_S17 + 1U))) ).xyz;
        Quat_0 hidden_0 = from_axis_angle_0(th_0, length(th_0));
        float3 _S18 = (float4(_S15->crot0_0) ).xyz;
        float3 _S19 = (float4(_S15->half_0) ).xyz;
        float3 _S20 = (float4(_S15->crot1_0) ).xyz;
        float3 _S21 = (float4(_S15->crot2_0) ).xyz;
        float3 local_0 = float3(dot(_S18, p_1 * _S19), dot(_S20, p_1 * _S19), dot(_S21, p_1 * _S19));
        float3 local_n_0 = float3(dot(_S18, n_3), dot(_S20, n_3), dot(_S21, n_3));
        float3 _S22 = (float4(_S15->center_0) ).xyz + (float4(*((&kernelContext_3)->state_0+_S17)) ).xyz;
        thread Quat_0 _S23 = hidden_0;
        float3 _S24 = rotate_0(&_S23, local_0);
        float3 body_0 = _S22 + _S24;
        float3 _S25 = (float4(_S16->position_0) ).xyz + (float4(_S16->position_err_0) ).xyz;
        thread Quat_0 _S26 = q_4;
        float3 _S27 = rotate_0(&_S26, body_0);
        float3 world_5 = _S25 + _S27;
        uint _S28 = (&kernelContext_3)->index_0[_S14];
        float util_0 = 0.0f;
        uint e_0 = _S28;
        uint intact_0 = 0U;
        for(;;)
        {
            if(e_0 < ((&kernelContext_3)->index_0)[_S14 + 1U])
            {
            }
            else
            {
                break;
            }
            BondDyn_natural_0 device* _S29 = (&kernelContext_3)->bond_dyn_0+((&kernelContext_3)->index_0[e_0] >> 1U);
            if(((uint4(_S29->events_0) ).z) != 0U)
            {
                e_0 = e_0 + 1U;
                continue;
            }
            uint _S30 = intact_0 + 1U;
            util_0 = max(util_0, _S29->js_0.utilization_0);
            intact_0 = _S30;
            e_0 = e_0 + 1U;
        }
        thread VertexOut_0 o_0;
        float4 _S31 = clip_of_0(world_5, &kernelContext_3);
        (&o_0)->position_5 = _S31;
        thread Quat_0 _S32 = hidden_0;
        float3 _S33 = rotate_0(&_S32, local_n_0);
        thread Quat_0 _S34 = q_4;
        float3 _S35 = rotate_0(&_S34, _S33);
        (&o_0)->normal_3 = _S35;
        float3 _S36;
        if(intact_0 > 0U)
        {
            _S36 = ramp_0(util_0);
        }
        else
        {
            _S36 = float3(0.55000001192092896f, 0.51999998092651367f, 0.47999998927116394f);
        }
        (&o_0)->color_3 = _S36;
        (&o_0)->world_4 = world_5;
        _S13 = o_0;
        break;
    }
    thread chunk_vs_Result_0 _S37;
    (&_S37)->position_4 = _S13.position_5;
    (&_S37)->normal_2 = _S13.normal_3;
    (&_S37)->color_2 = _S13.color_3;
    (&_S37)->world_3 = _S13.world_4;
    return _S37;
}

struct host_chunk_vs_Result_0
{
    float4 position_6 [[position]];
    float3 normal_4 [[user(_SLANG_ATTR)]];
    float3 color_4 [[user(_SLANG_ATTR_1)]];
    float3 world_6 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] host_chunk_vs_Result_0 host_chunk_vs(uint vertex_2 [[vertex_id]], uint instance_1 [[instance_id]], uint base_instance_1 [[base_instance]], ChunkStatic_natural_0 device* chunks_4 [[buffer(1)]], Island_natural_0 device* islands_4 [[buffer(2)]], packed_float4 device* state_4 [[buffer(3)]], uint device* index_4 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(5)]], Camera_0 constant* camera_4 [[buffer(0)]], HostChunk_natural_0 device* host_chunks_4 [[buffer(7)]], Impactor_natural_0 device* impactors_4 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->chunks_0 = chunks_4;
    (&kernelContext_4)->islands_0 = islands_4;
    (&kernelContext_4)->state_0 = state_4;
    (&kernelContext_4)->index_0 = index_4;
    (&kernelContext_4)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_4)->camera_0 = camera_4;
    (&kernelContext_4)->host_chunks_0 = host_chunks_4;
    (&kernelContext_4)->impactors_0 = impactors_4;
    uint _S38 = instance_1 - base_instance_1;
    thread float3 p_2;
    thread float3 n_4;
    box_corner_0(vertex_2, &p_2, &n_4);
    HostChunk_natural_0 hc_0 = host_chunks_4[_S38];
    float4 _S39 = float4(hc_0.center_1) ;
    float4 _S40 = float4(hc_0.axis0_0) ;
    float3 _S41 = _S40.xyz;
    float4 _S42 = float4(hc_0.axis1_0) ;
    float3 _S43 = _S42.xyz;
    float4 _S44 = float4(hc_0.axis2_0) ;
    float3 _S45 = _S44.xyz;
    float3 world_7 = _S39.xyz + _S41 * float3((p_2.x * _S40.w))  + _S43 * float3((p_2.y * _S42.w))  + _S45 * float3((p_2.z * _S44.w)) ;
    thread VertexOut_0 o_1;
    float4 _S46 = clip_of_0(world_7, &kernelContext_4);
    (&o_1)->position_5 = _S46;
    (&o_1)->normal_3 = _S41 * float3(n_4.x)  + _S43 * float3(n_4.y)  + _S45 * float3(n_4.z) ;
    float _S47 = _S39.w;
    float3 _S48;
    if(_S47 >= 0.0f)
    {
        _S48 = ramp_0(_S47);
    }
    else
    {
        _S48 = float3(0.55000001192092896f, 0.51999998092651367f, 0.47999998927116394f);
    }
    (&o_1)->color_3 = _S48;
    (&o_1)->world_4 = world_7;
    VertexOut_0 _S49 = o_1;
    thread host_chunk_vs_Result_0 _S50;
    (&_S50)->position_6 = _S49.position_5;
    (&_S50)->normal_4 = _S49.normal_3;
    (&_S50)->color_4 = _S49.color_3;
    (&_S50)->world_6 = _S49.world_4;
    return _S50;
}

struct impactor_vs_Result_0
{
    float4 position_7 [[position]];
    float3 normal_5 [[user(_SLANG_ATTR)]];
    float3 color_5 [[user(_SLANG_ATTR_1)]];
    float3 world_8 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] impactor_vs_Result_0 impactor_vs(uint vertex_3 [[vertex_id]], uint instance_2 [[instance_id]], uint base_instance_2 [[base_instance]], ChunkStatic_natural_0 device* chunks_5 [[buffer(1)]], Island_natural_0 device* islands_5 [[buffer(2)]], packed_float4 device* state_5 [[buffer(3)]], uint device* index_5 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(5)]], Camera_0 constant* camera_5 [[buffer(0)]], HostChunk_natural_0 device* host_chunks_5 [[buffer(7)]], Impactor_natural_0 device* impactors_5 [[buffer(6)]])
{
    VertexOut_0 _S51;
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->chunks_0 = chunks_5;
    (&kernelContext_5)->islands_0 = islands_5;
    (&kernelContext_5)->state_0 = state_5;
    (&kernelContext_5)->index_0 = index_5;
    (&kernelContext_5)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_5)->camera_0 = camera_5;
    (&kernelContext_5)->host_chunks_0 = host_chunks_5;
    (&kernelContext_5)->impactors_0 = impactors_5;
    uint _S52 = instance_2 - base_instance_2;
    for(;;)
    {
        Impactor_natural_0 device* _S53 = (&kernelContext_5)->impactors_0+_S52;
        Quat_0 q_5 = rotation_of_0(float4(_S53->rotation_1) , float4(_S53->rotation_err_1) );
        float3 center_2 = (float4(_S53->position_1) ).xyz + (float4(_S53->position_err_1) ).xyz;
        thread float3 p_3;
        thread float3 n_5;
        thread VertexOut_0 o_2;
        float4 _S54 = float4(_S53->shape_0) ;
        if((_S54.x) > 0.5f)
        {
            if(vertex_3 >= 36U)
            {
                (&o_2)->position_5 = float4(0.0f, 0.0f, 2.0f, 1.0f);
                (&o_2)->normal_3 = float3(0.0f, 0.0f, 1.0f);
                float3 _S55 = float3(0.0f) ;
                (&o_2)->color_3 = _S55;
                (&o_2)->world_4 = _S55;
                _S51 = o_2;
                break;
            }
            box_corner_0(vertex_3, &p_3, &n_5);
            p_3 = p_3 * (float4(_S53->half_1) ).xyz;
        }
        else
        {
            uint quad_1 = vertex_3 / 6U;
            uint corner_1 = vertex_3 % 6U;
            uint2 _S56 = uint2(0U, 0U);
            uint2 _S57 = uint2(1U, 1U);
            array<uint2, int(6)> offs_0 = { { _S56, uint2(1U, 0U), _S57, _S56, _S57, uint2(0U, 1U) } };
            float theta_0 = 3.14159274101257324f * float(quad_1 / 16U + offs_0[corner_1].y) / 10.0f;
            float phi_0 = 6.28318548202514648f * float(quad_1 % 16U + offs_0[corner_1].x) / 16.0f;
            float _S58 = sin(theta_0);
            float3 _S59 = float3(_S58 * cos(phi_0), _S58 * sin(phi_0), cos(theta_0));
            n_5 = _S59;
            p_3 = _S59 * float3(_S54.y) ;
        }
        thread Quat_0 _S60 = q_5;
        float3 _S61 = rotate_0(&_S60, p_3);
        float3 world_9 = center_2 + _S61;
        float4 _S62 = clip_of_0(world_9, &kernelContext_5);
        (&o_2)->position_5 = _S62;
        thread Quat_0 _S63 = q_5;
        float3 _S64 = rotate_0(&_S63, n_5);
        (&o_2)->normal_3 = _S64;
        (&o_2)->color_3 = float3(0.77999997138977051f, 0.80000001192092896f, 0.8399999737739563f);
        (&o_2)->world_4 = world_9;
        _S51 = o_2;
        break;
    }
    thread impactor_vs_Result_0 _S65;
    (&_S65)->position_7 = _S51.position_5;
    (&_S65)->normal_5 = _S51.normal_3;
    (&_S65)->color_5 = _S51.color_3;
    (&_S65)->world_8 = _S51.world_4;
    return _S65;
}

struct ground_vs_Result_0
{
    float4 position_8 [[position]];
    float3 normal_6 [[user(_SLANG_ATTR)]];
    float3 color_6 [[user(_SLANG_ATTR_1)]];
    float3 world_10 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] ground_vs_Result_0 ground_vs(uint vertex_4 [[vertex_id]], ChunkStatic_natural_0 device* chunks_6 [[buffer(1)]], Island_natural_0 device* islands_6 [[buffer(2)]], packed_float4 device* state_6 [[buffer(3)]], uint device* index_6 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(5)]], Camera_0 constant* camera_6 [[buffer(0)]], HostChunk_natural_0 device* host_chunks_6 [[buffer(7)]], Impactor_natural_0 device* impactors_6 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->chunks_0 = chunks_6;
    (&kernelContext_6)->islands_0 = islands_6;
    (&kernelContext_6)->state_0 = state_6;
    (&kernelContext_6)->index_0 = index_6;
    (&kernelContext_6)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_6)->camera_0 = camera_6;
    (&kernelContext_6)->host_chunks_0 = host_chunks_6;
    (&kernelContext_6)->impactors_0 = impactors_6;
    float2 _S66 = float2(-1.0f, -1.0f);
    float2 _S67 = float2(1.0f, 1.0f);
    array<float2, int(6)> quad_2 = { { _S66, float2(1.0f, -1.0f), _S67, _S66, _S67, float2(-1.0f, 1.0f) } };
    float3 world_11 = float3(camera_6->ground_0.zw + quad_2[vertex_4] * float2(camera_6->ground_0.y) , camera_6->ground_0.x);
    thread VertexOut_0 o_3;
    float4 _S68 = clip_of_0(world_11, &kernelContext_6);
    (&o_3)->position_5 = _S68;
    (&o_3)->normal_3 = float3(0.0f, 0.0f, 1.0f);
    (&o_3)->color_3 = float3(0.0f) ;
    (&o_3)->world_4 = world_11;
    thread ground_vs_Result_0 _S69;
    (&_S69)->position_8 = o_3.position_5;
    (&_S69)->normal_6 = o_3.normal_3;
    (&_S69)->color_6 = o_3.color_3;
    (&_S69)->world_10 = o_3.world_4;
    return _S69;
}

