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

struct HullVertex_natural_0
{
    packed_float4 pos_0;
    packed_float4 normal_0;
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
    HullVertex_natural_0 device* hull_verts_0;
    HostChunk_natural_0 device* host_chunks_0;
    HullVertex_natural_0 device* host_hull_verts_0;
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

struct VertexOut_0
{
    float4 position_2;
    float3 normal_1;
    float3 color_0;
    float3 world_1;
};

VertexOut_0 chunk_vertex_0(uint c_0, float3 p_1, float3 n_3, KernelContext_0 thread* kernelContext_1)
{
    ChunkStatic_natural_0 device* _S7 = kernelContext_1->chunks_0+c_0;
    Island_natural_0 device* _S8 = kernelContext_1->islands_0+(uint4(_S7->info_0) ).y;
    Quat_0 q_4 = rotation_of_0(float4(_S8->rotation_0) , float4(_S8->rotation_err_0) );
    uint _S9 = 4U * c_0;
    float3 th_0 = (float4(*(kernelContext_1->state_0+(_S9 + 1U))) ).xyz;
    Quat_0 hidden_0 = from_axis_angle_0(th_0, length(th_0));
    float3 _S10 = (float4(_S7->crot0_0) ).xyz;
    float3 _S11 = (float4(_S7->crot1_0) ).xyz;
    float3 _S12 = (float4(_S7->crot2_0) ).xyz;
    float3 local_0 = float3(dot(_S10, p_1), dot(_S11, p_1), dot(_S12, p_1));
    float3 local_n_0 = float3(dot(_S10, n_3), dot(_S11, n_3), dot(_S12, n_3));
    float3 _S13 = (float4(_S7->center_0) ).xyz + (float4(*(kernelContext_1->state_0+_S9)) ).xyz;
    thread Quat_0 _S14 = hidden_0;
    float3 _S15 = rotate_0(&_S14, local_0);
    float3 body_0 = _S13 + _S15;
    float3 _S16 = (float4(_S8->position_0) ).xyz + (float4(_S8->position_err_0) ).xyz;
    thread Quat_0 _S17 = q_4;
    float3 _S18 = rotate_0(&_S17, body_0);
    float3 world_2 = _S16 + _S18;
    uint _S19 = kernelContext_1->index_0[c_0];
    float util_0 = 0.0f;
    uint e_0 = _S19;
    uint intact_0 = 0U;
    for(;;)
    {
        if(e_0 < (kernelContext_1->index_0)[c_0 + 1U])
        {
        }
        else
        {
            break;
        }
        BondDyn_natural_0 device* _S20 = kernelContext_1->bond_dyn_0+(kernelContext_1->index_0[e_0] >> 1U);
        if(((uint4(_S20->events_0) ).z) != 0U)
        {
            e_0 = e_0 + 1U;
            continue;
        }
        uint _S21 = intact_0 + 1U;
        util_0 = max(util_0, _S20->js_0.utilization_0);
        intact_0 = _S21;
        e_0 = e_0 + 1U;
    }
    thread VertexOut_0 o_0;
    float4 _S22 = clip_of_0(world_2, kernelContext_1);
    (&o_0)->position_2 = _S22;
    thread Quat_0 _S23 = hidden_0;
    float3 _S24 = rotate_0(&_S23, local_n_0);
    thread Quat_0 _S25 = q_4;
    float3 _S26 = rotate_0(&_S25, _S24);
    (&o_0)->normal_1 = _S26;
    float3 _S27;
    if(intact_0 > 0U)
    {
        _S27 = ramp_0(util_0);
    }
    else
    {
        _S27 = float3(0.55000001192092896f, 0.51999998092651367f, 0.47999998927116394f);
    }
    (&o_0)->color_0 = _S27;
    (&o_0)->world_1 = world_2;
    return o_0;
}

VertexOut_0 host_vertex_0(const HostChunk_natural_0 thread* hc_0, float3 p_2, float3 n_4, KernelContext_0 thread* kernelContext_2)
{
    float4 _S28 = float4(hc_0->center_1) ;
    float3 _S29 = (float4(hc_0->axis0_0) ).xyz;
    float3 _S30 = (float4(hc_0->axis1_0) ).xyz;
    float3 _S31 = (float4(hc_0->axis2_0) ).xyz;
    float3 world_3 = _S28.xyz + _S29 * float3(p_2.x)  + _S30 * float3(p_2.y)  + _S31 * float3(p_2.z) ;
    thread VertexOut_0 o_1;
    float4 _S32 = clip_of_0(world_3, kernelContext_2);
    (&o_1)->position_2 = _S32;
    (&o_1)->normal_1 = _S29 * float3(n_4.x)  + _S30 * float3(n_4.y)  + _S31 * float3(n_4.z) ;
    float _S33 = _S28.w;
    float3 _S34;
    if(_S33 >= 0.0f)
    {
        _S34 = ramp_0(_S33);
    }
    else
    {
        _S34 = float3(0.55000001192092896f, 0.51999998092651367f, 0.47999998927116394f);
    }
    (&o_1)->color_0 = _S34;
    (&o_1)->world_1 = world_3;
    return o_1;
}

struct pixelOutput_0
{
    float4 output_0 [[color(0)]];
};

struct pixelInput_0
{
    float3 normal_2 [[user(_SLANG_ATTR)]];
    float3 color_1 [[user(_SLANG_ATTR_1)]];
    float3 world_4 [[user(_SLANG_ATTR_2)]];
};

[[fragment]] pixelOutput_0 lit_fs(pixelInput_0 _S35 [[stage_in]], float4 position_3 [[position]], ChunkStatic_natural_0 device* chunks_1 [[buffer(1)]], Island_natural_0 device* islands_1 [[buffer(2)]], packed_float4 device* state_1 [[buffer(3)]], uint device* index_1 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(5)]], Camera_0 constant* camera_1 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_1 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_1 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_1 [[buffer(9)]], Impactor_natural_0 device* impactors_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->chunks_0 = chunks_1;
    (&kernelContext_3)->islands_0 = islands_1;
    (&kernelContext_3)->state_0 = state_1;
    (&kernelContext_3)->index_0 = index_1;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_3)->camera_0 = camera_1;
    (&kernelContext_3)->hull_verts_0 = hull_verts_1;
    (&kernelContext_3)->host_chunks_0 = host_chunks_1;
    (&kernelContext_3)->host_hull_verts_0 = host_hull_verts_1;
    (&kernelContext_3)->impactors_0 = impactors_1;
    pixelOutput_0 _S36 = { float4(_S35.color_1 * float3((0.34999999403953552f + 0.64999997615814209f * max(dot(normalize(_S35.normal_2), normalize(camera_1->light_0.xyz)), 0.0f))) , 1.0f) };
    return _S36;
}

struct pixelOutput_1
{
    float4 output_1 [[color(0)]];
};

struct pixelInput_1
{
    float3 normal_3 [[user(_SLANG_ATTR)]];
    float3 color_2 [[user(_SLANG_ATTR_1)]];
    float3 world_5 [[user(_SLANG_ATTR_2)]];
};

[[fragment]] pixelOutput_1 ground_fs(pixelInput_1 _S37 [[stage_in]], float4 position_4 [[position]], ChunkStatic_natural_0 device* chunks_2 [[buffer(1)]], Island_natural_0 device* islands_2 [[buffer(2)]], packed_float4 device* state_2 [[buffer(3)]], uint device* index_2 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(5)]], Camera_0 constant* camera_2 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_2 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_2 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_2 [[buffer(9)]], Impactor_natural_0 device* impactors_2 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->chunks_0 = chunks_2;
    (&kernelContext_4)->islands_0 = islands_2;
    (&kernelContext_4)->state_0 = state_2;
    (&kernelContext_4)->index_0 = index_2;
    (&kernelContext_4)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_4)->camera_0 = camera_2;
    (&kernelContext_4)->hull_verts_0 = hull_verts_2;
    (&kernelContext_4)->host_chunks_0 = host_chunks_2;
    (&kernelContext_4)->host_hull_verts_0 = host_hull_verts_2;
    (&kernelContext_4)->impactors_0 = impactors_2;
    float2 _S38 = _S37.world_5.xy;
    int2 _S39 = int2(floor(_S38));
    float checker_0;
    if(((_S39.x + _S39.y) & int(1)) == int(0))
    {
        checker_0 = 1.0f;
    }
    else
    {
        checker_0 = 0.0f;
    }
    float edge_0 = saturate(length(_S38 - (&kernelContext_4)->camera_0->ground_0.zw) / (&kernelContext_4)->camera_0->ground_0.y);
    pixelOutput_1 _S40 = { float4(mix(mix(float3(0.15999999642372131f, 0.17000000178813934f, 0.18999999761581421f), float3(0.20000000298023224f, 0.20999999344348907f, 0.23000000417232513f), float3(checker_0) ), float3(0.05000000074505806f, 0.05999999865889549f, 0.07999999821186066f), float3((edge_0 * edge_0)) ), 1.0f) };
    return _S40;
}

struct chunk_vs_Result_0
{
    float4 position_5 [[position]];
    float3 normal_4 [[user(_SLANG_ATTR)]];
    float3 color_3 [[user(_SLANG_ATTR_1)]];
    float3 world_6 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] chunk_vs_Result_0 chunk_vs(uint vertex_1 [[vertex_id]], uint instance_0 [[instance_id]], uint base_instance_0 [[base_instance]], ChunkStatic_natural_0 device* chunks_3 [[buffer(1)]], Island_natural_0 device* islands_3 [[buffer(2)]], packed_float4 device* state_3 [[buffer(3)]], uint device* index_3 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(5)]], Camera_0 constant* camera_3 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_3 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_3 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_3 [[buffer(9)]], Impactor_natural_0 device* impactors_3 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->chunks_0 = chunks_3;
    (&kernelContext_5)->islands_0 = islands_3;
    (&kernelContext_5)->state_0 = state_3;
    (&kernelContext_5)->index_0 = index_3;
    (&kernelContext_5)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_5)->camera_0 = camera_3;
    (&kernelContext_5)->hull_verts_0 = hull_verts_3;
    (&kernelContext_5)->host_chunks_0 = host_chunks_3;
    (&kernelContext_5)->host_hull_verts_0 = host_hull_verts_3;
    (&kernelContext_5)->impactors_0 = impactors_3;
    uint _S41 = instance_0 - base_instance_0;
    thread float3 p_3;
    thread float3 n_5;
    box_corner_0(vertex_1, &p_3, &n_5);
    VertexOut_0 _S42 = chunk_vertex_0(_S41, p_3 * (float4((chunks_3+_S41)->half_0) ).xyz, n_5, &kernelContext_5);
    thread chunk_vs_Result_0 _S43;
    (&_S43)->position_5 = _S42.position_2;
    (&_S43)->normal_4 = _S42.normal_1;
    (&_S43)->color_3 = _S42.color_0;
    (&_S43)->world_6 = _S42.world_1;
    return _S43;
}

struct chunk_hull_vs_Result_0
{
    float4 position_6 [[position]];
    float3 normal_5 [[user(_SLANG_ATTR)]];
    float3 color_4 [[user(_SLANG_ATTR_1)]];
    float3 world_7 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] chunk_hull_vs_Result_0 chunk_hull_vs(uint vertex_2 [[vertex_id]], ChunkStatic_natural_0 device* chunks_4 [[buffer(1)]], Island_natural_0 device* islands_4 [[buffer(2)]], packed_float4 device* state_4 [[buffer(3)]], uint device* index_4 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(5)]], Camera_0 constant* camera_4 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_4 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_4 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_4 [[buffer(9)]], Impactor_natural_0 device* impactors_4 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->chunks_0 = chunks_4;
    (&kernelContext_6)->islands_0 = islands_4;
    (&kernelContext_6)->state_0 = state_4;
    (&kernelContext_6)->index_0 = index_4;
    (&kernelContext_6)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_6)->camera_0 = camera_4;
    (&kernelContext_6)->hull_verts_0 = hull_verts_4;
    (&kernelContext_6)->host_chunks_0 = host_chunks_4;
    (&kernelContext_6)->host_hull_verts_0 = host_hull_verts_4;
    (&kernelContext_6)->impactors_0 = impactors_4;
    HullVertex_natural_0 v_2 = hull_verts_4[vertex_2];
    float4 _S44 = float4(v_2.pos_0) ;
    VertexOut_0 _S45 = chunk_vertex_0(uint(_S44.w), _S44.xyz, (float4(v_2.normal_0) ).xyz, &kernelContext_6);
    thread chunk_hull_vs_Result_0 _S46;
    (&_S46)->position_6 = _S45.position_2;
    (&_S46)->normal_5 = _S45.normal_1;
    (&_S46)->color_4 = _S45.color_0;
    (&_S46)->world_7 = _S45.world_1;
    return _S46;
}

struct host_chunk_vs_Result_0
{
    float4 position_7 [[position]];
    float3 normal_6 [[user(_SLANG_ATTR)]];
    float3 color_5 [[user(_SLANG_ATTR_1)]];
    float3 world_8 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] host_chunk_vs_Result_0 host_chunk_vs(uint vertex_3 [[vertex_id]], uint instance_1 [[instance_id]], uint base_instance_1 [[base_instance]], ChunkStatic_natural_0 device* chunks_5 [[buffer(1)]], Island_natural_0 device* islands_5 [[buffer(2)]], packed_float4 device* state_5 [[buffer(3)]], uint device* index_5 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(5)]], Camera_0 constant* camera_5 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_5 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_5 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_5 [[buffer(9)]], Impactor_natural_0 device* impactors_5 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->chunks_0 = chunks_5;
    (&kernelContext_7)->islands_0 = islands_5;
    (&kernelContext_7)->state_0 = state_5;
    (&kernelContext_7)->index_0 = index_5;
    (&kernelContext_7)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_7)->camera_0 = camera_5;
    (&kernelContext_7)->hull_verts_0 = hull_verts_5;
    (&kernelContext_7)->host_chunks_0 = host_chunks_5;
    (&kernelContext_7)->host_hull_verts_0 = host_hull_verts_5;
    (&kernelContext_7)->impactors_0 = impactors_5;
    uint _S47 = instance_1 - base_instance_1;
    thread float3 p_4;
    thread float3 n_6;
    box_corner_0(vertex_3, &p_4, &n_6);
    HostChunk_natural_0 hc_1 = host_chunks_5[_S47];
    thread HostChunk_natural_0 _S48 = hc_1;
    float3 _S49 = p_4 * float3((float4((&_S48)->axis0_0) ).w, (float4((&_S48)->axis1_0) ).w, (float4((&_S48)->axis2_0) ).w);
    _S48 = hc_1;
    VertexOut_0 _S50 = host_vertex_0(&_S48, _S49, n_6, &kernelContext_7);
    thread host_chunk_vs_Result_0 _S51;
    (&_S51)->position_7 = _S50.position_2;
    (&_S51)->normal_6 = _S50.normal_1;
    (&_S51)->color_5 = _S50.color_0;
    (&_S51)->world_8 = _S50.world_1;
    return _S51;
}

struct host_hull_vs_Result_0
{
    float4 position_8 [[position]];
    float3 normal_7 [[user(_SLANG_ATTR)]];
    float3 color_6 [[user(_SLANG_ATTR_1)]];
    float3 world_9 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] host_hull_vs_Result_0 host_hull_vs(uint vertex_4 [[vertex_id]], ChunkStatic_natural_0 device* chunks_6 [[buffer(1)]], Island_natural_0 device* islands_6 [[buffer(2)]], packed_float4 device* state_6 [[buffer(3)]], uint device* index_6 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(5)]], Camera_0 constant* camera_6 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_6 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_6 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_6 [[buffer(9)]], Impactor_natural_0 device* impactors_6 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->chunks_0 = chunks_6;
    (&kernelContext_8)->islands_0 = islands_6;
    (&kernelContext_8)->state_0 = state_6;
    (&kernelContext_8)->index_0 = index_6;
    (&kernelContext_8)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_8)->camera_0 = camera_6;
    (&kernelContext_8)->hull_verts_0 = hull_verts_6;
    (&kernelContext_8)->host_chunks_0 = host_chunks_6;
    (&kernelContext_8)->host_hull_verts_0 = host_hull_verts_6;
    (&kernelContext_8)->impactors_0 = impactors_6;
    HullVertex_natural_0 v_3 = host_hull_verts_6[vertex_4];
    float4 _S52 = float4(v_3.pos_0) ;
    float3 _S53 = _S52.xyz;
    float3 _S54 = (float4(v_3.normal_0) ).xyz;
    thread HostChunk_natural_0 _S55 = host_chunks_6[uint(_S52.w)];
    VertexOut_0 _S56 = host_vertex_0(&_S55, _S53, _S54, &kernelContext_8);
    thread host_hull_vs_Result_0 _S57;
    (&_S57)->position_8 = _S56.position_2;
    (&_S57)->normal_7 = _S56.normal_1;
    (&_S57)->color_6 = _S56.color_0;
    (&_S57)->world_9 = _S56.world_1;
    return _S57;
}

struct impactor_vs_Result_0
{
    float4 position_9 [[position]];
    float3 normal_8 [[user(_SLANG_ATTR)]];
    float3 color_7 [[user(_SLANG_ATTR_1)]];
    float3 world_10 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] impactor_vs_Result_0 impactor_vs(uint vertex_5 [[vertex_id]], uint instance_2 [[instance_id]], uint base_instance_2 [[base_instance]], ChunkStatic_natural_0 device* chunks_7 [[buffer(1)]], Island_natural_0 device* islands_7 [[buffer(2)]], packed_float4 device* state_7 [[buffer(3)]], uint device* index_7 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(5)]], Camera_0 constant* camera_7 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_7 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_7 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_7 [[buffer(9)]], Impactor_natural_0 device* impactors_7 [[buffer(6)]])
{
    VertexOut_0 _S58;
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->chunks_0 = chunks_7;
    (&kernelContext_9)->islands_0 = islands_7;
    (&kernelContext_9)->state_0 = state_7;
    (&kernelContext_9)->index_0 = index_7;
    (&kernelContext_9)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_9)->camera_0 = camera_7;
    (&kernelContext_9)->hull_verts_0 = hull_verts_7;
    (&kernelContext_9)->host_chunks_0 = host_chunks_7;
    (&kernelContext_9)->host_hull_verts_0 = host_hull_verts_7;
    (&kernelContext_9)->impactors_0 = impactors_7;
    uint _S59 = instance_2 - base_instance_2;
    for(;;)
    {
        Impactor_natural_0 device* _S60 = (&kernelContext_9)->impactors_0+_S59;
        Quat_0 q_5 = rotation_of_0(float4(_S60->rotation_1) , float4(_S60->rotation_err_1) );
        float3 center_2 = (float4(_S60->position_1) ).xyz + (float4(_S60->position_err_1) ).xyz;
        thread float3 p_5;
        thread float3 n_7;
        thread VertexOut_0 o_2;
        float4 _S61 = float4(_S60->shape_0) ;
        if((_S61.x) > 0.5f)
        {
            if(vertex_5 >= 36U)
            {
                (&o_2)->position_2 = float4(0.0f, 0.0f, 2.0f, 1.0f);
                (&o_2)->normal_1 = float3(0.0f, 0.0f, 1.0f);
                float3 _S62 = float3(0.0f) ;
                (&o_2)->color_0 = _S62;
                (&o_2)->world_1 = _S62;
                _S58 = o_2;
                break;
            }
            box_corner_0(vertex_5, &p_5, &n_7);
            p_5 = p_5 * (float4(_S60->half_1) ).xyz;
        }
        else
        {
            uint quad_1 = vertex_5 / 6U;
            uint corner_1 = vertex_5 % 6U;
            uint2 _S63 = uint2(0U, 0U);
            uint2 _S64 = uint2(1U, 1U);
            array<uint2, int(6)> offs_0 = { { _S63, uint2(1U, 0U), _S64, _S63, _S64, uint2(0U, 1U) } };
            float theta_0 = 3.14159274101257324f * float(quad_1 / 16U + offs_0[corner_1].y) / 10.0f;
            float phi_0 = 6.28318548202514648f * float(quad_1 % 16U + offs_0[corner_1].x) / 16.0f;
            float _S65 = sin(theta_0);
            float3 _S66 = float3(_S65 * cos(phi_0), _S65 * sin(phi_0), cos(theta_0));
            n_7 = _S66;
            p_5 = _S66 * float3(_S61.y) ;
        }
        thread Quat_0 _S67 = q_5;
        float3 _S68 = rotate_0(&_S67, p_5);
        float3 world_11 = center_2 + _S68;
        float4 _S69 = clip_of_0(world_11, &kernelContext_9);
        (&o_2)->position_2 = _S69;
        thread Quat_0 _S70 = q_5;
        float3 _S71 = rotate_0(&_S70, n_7);
        (&o_2)->normal_1 = _S71;
        (&o_2)->color_0 = float3(0.77999997138977051f, 0.80000001192092896f, 0.8399999737739563f);
        (&o_2)->world_1 = world_11;
        _S58 = o_2;
        break;
    }
    thread impactor_vs_Result_0 _S72;
    (&_S72)->position_9 = _S58.position_2;
    (&_S72)->normal_8 = _S58.normal_1;
    (&_S72)->color_7 = _S58.color_0;
    (&_S72)->world_10 = _S58.world_1;
    return _S72;
}

struct ground_vs_Result_0
{
    float4 position_10 [[position]];
    float3 normal_9 [[user(_SLANG_ATTR)]];
    float3 color_8 [[user(_SLANG_ATTR_1)]];
    float3 world_12 [[user(_SLANG_ATTR_2)]];
};

[[vertex]] ground_vs_Result_0 ground_vs(uint vertex_6 [[vertex_id]], ChunkStatic_natural_0 device* chunks_8 [[buffer(1)]], Island_natural_0 device* islands_8 [[buffer(2)]], packed_float4 device* state_8 [[buffer(3)]], uint device* index_8 [[buffer(4)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(5)]], Camera_0 constant* camera_8 [[buffer(0)]], HullVertex_natural_0 device* hull_verts_8 [[buffer(8)]], HostChunk_natural_0 device* host_chunks_8 [[buffer(7)]], HullVertex_natural_0 device* host_hull_verts_8 [[buffer(9)]], Impactor_natural_0 device* impactors_8 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->chunks_0 = chunks_8;
    (&kernelContext_10)->islands_0 = islands_8;
    (&kernelContext_10)->state_0 = state_8;
    (&kernelContext_10)->index_0 = index_8;
    (&kernelContext_10)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_10)->camera_0 = camera_8;
    (&kernelContext_10)->hull_verts_0 = hull_verts_8;
    (&kernelContext_10)->host_chunks_0 = host_chunks_8;
    (&kernelContext_10)->host_hull_verts_0 = host_hull_verts_8;
    (&kernelContext_10)->impactors_0 = impactors_8;
    float2 _S73 = float2(-1.0f, -1.0f);
    float2 _S74 = float2(1.0f, 1.0f);
    array<float2, int(6)> quad_2 = { { _S73, float2(1.0f, -1.0f), _S74, _S73, _S74, float2(-1.0f, 1.0f) } };
    float3 world_13 = float3(camera_8->ground_0.zw + quad_2[vertex_6] * float2(camera_8->ground_0.y) , camera_8->ground_0.x);
    thread VertexOut_0 o_3;
    float4 _S75 = clip_of_0(world_13, &kernelContext_10);
    (&o_3)->position_2 = _S75;
    (&o_3)->normal_1 = float3(0.0f, 0.0f, 1.0f);
    (&o_3)->color_0 = float3(0.0f) ;
    (&o_3)->world_1 = world_13;
    thread ground_vs_Result_0 _S76;
    (&_S76)->position_10 = o_3.position_2;
    (&_S76)->normal_9 = o_3.normal_1;
    (&_S76)->color_8 = o_3.color_0;
    (&_S76)->world_12 = o_3.world_1;
    return _S76;
}

