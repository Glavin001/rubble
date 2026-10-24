#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float3 rotate_q_0(float4 q_0, float3 v_0)
{
    float3 u_0 = q_0.xyz;
    float3 t_0 = float3(2.0f)  * cross(u_0, v_0);
    return v_0 + float3(q_0.w)  * t_0 + cross(u_0, t_0);
}

struct ColliderInfo_0
{
    uint4 a_0;
    float4 b_0;
};

struct IngestParams_0
{
    uint row_offset_0;
    uint max_rows_0;
    uint substep_0;
    uint chunk_count_0;
    uint record_count_0;
    uint max_bodies_0;
    float h_0;
    uint wheels_0;
    uint wheel_base_0;
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct Record_natural_0
{
    packed_uint4 ids_0;
    packed_float4 point_0;
    packed_float4 impulse_0;
    packed_float4 other_0;
    packed_uint4 key_0;
};

struct EngineContactRow_natural_0
{
    packed_uint4 ids_1;
    packed_float4 normal_pen_0;
    packed_float4 anchor_a_0;
    packed_float4 anchor_b_0;
    packed_float4 t1_lambda_n_0;
    packed_float4 t2_lambda_t1_0;
    packed_float4 kn_kt_lt2_c0t1_0;
    packed_float4 tail_0;
};

struct EngineCollider_natural_0
{
    packed_uint4 ids_2;
    packed_float4 local_pos_0;
    packed_float4 local_rot_0;
    packed_float4 params_0;
};

struct EngineBody_natural_0
{
    packed_float4 pos_inv_mass_0;
    packed_float4 rot_0;
    packed_float4 inv_inertia_flags_0;
    packed_float4 vel_sleep_0;
    packed_float4 ang_vel_gen_0;
    packed_float4 initial_pos_0;
    packed_float4 initial_rot_0;
};

struct ColliderInfo_natural_0
{
    packed_uint4 a_0;
    packed_float4 b_0;
};

struct KernelContext_0
{
    IngestParams_0 constant* params_1;
    uint device* row_counts_0;
    uint device* counters_0;
    Record_natural_0 device* records_0;
    EngineContactRow_natural_0 device* contacts_0;
    EngineCollider_natural_0 device* colliders_0;
    EngineBody_natural_0 device* bodies_0;
    ColliderInfo_natural_0 device* collider_info_0;
    EngineBody_natural_0 device* start_bodies_0;
    packed_uint4 device* vehicle_pool_0;
    atomic<uint> device* counts_0;
    uint device* sort_0;
    Record_natural_0 device* manifolds_0;
    array<uint, int(1024)> threadgroup* g_scan_0;
};

float3 point_velocity_0(const ColliderInfo_0 thread* info_0, uint body_0, float3 p_0, KernelContext_0 thread* kernelContext_0)
{
    if((info_0->a_0.z) != 2U)
    {
        return float3(0.0f) ;
    }
    EngineBody_natural_0 s_0 = kernelContext_0->start_bodies_0[body_0];
    return (float4(s_0.vel_sleep_0) ).xyz + cross((float4(s_0.ang_vel_gen_0) ).xyz, p_0 - (float4(s_0.pos_inv_mass_0) ).xyz);
}

float3 to_zup_0(float3 v_1)
{
    return float3(v_1.x, - v_1.z, v_1.y);
}

struct Record_0
{
    uint4 ids_0;
    float4 point_0;
    float4 impulse_0;
    float4 other_0;
    uint4 key_0;
};

[[kernel]] void ingest_rows(uint3 id_0 [[thread_position_in_grid]], IngestParams_0 constant* params_2 [[buffer(0)]], uint device* row_counts_1 [[buffer(11)]], uint device* counters_1 [[buffer(5)]], Record_natural_0 device* records_1 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_1 [[buffer(1)]], EngineCollider_natural_0 device* colliders_1 [[buffer(2)]], EngineBody_natural_0 device* bodies_1 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_1 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_1 [[buffer(4)]], packed_uint4 device* vehicle_pool_1 [[buffer(12)]], atomic<uint> device* counts_1 [[buffer(10)]], uint device* sort_1 [[buffer(8)]], Record_natural_0 device* manifolds_1 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_1 = params_2;
    (&kernelContext_1)->row_counts_0 = row_counts_1;
    (&kernelContext_1)->counters_0 = counters_1;
    (&kernelContext_1)->records_0 = records_1;
    (&kernelContext_1)->contacts_0 = contacts_1;
    (&kernelContext_1)->colliders_0 = colliders_1;
    (&kernelContext_1)->bodies_0 = bodies_1;
    (&kernelContext_1)->collider_info_0 = collider_info_1;
    (&kernelContext_1)->start_bodies_0 = start_bodies_1;
    (&kernelContext_1)->vehicle_pool_0 = vehicle_pool_1;
    (&kernelContext_1)->counts_0 = counts_1;
    (&kernelContext_1)->sort_0 = sort_1;
    (&kernelContext_1)->manifolds_0 = manifolds_1;
    threadgroup array<uint, int(1024)> g_scan_1;
    (&kernelContext_1)->g_scan_0 = &g_scan_1;
    uint i_0 = id_0.x;
    if(i_0 >= (params_2->max_rows_0))
    {
        return;
    }
    uint base_0 = 2U * ((&kernelContext_1)->params_1->substep_0 * params_2->max_rows_0 + i_0);
    if(i_0 == 0U)
    {
        *((&kernelContext_1)->row_counts_0+(&kernelContext_1)->params_1->substep_0) = (&kernelContext_1)->counters_0[int(2)];
    }
    if(i_0 >= ((&kernelContext_1)->counters_0)[int(2)])
    {
        return;
    }
    thread Record_0 none_0;
    (&none_0)->ids_0 = uint4(4294967295U, 4294967295U, base_0, 0U);
    float4 _S1 = float4(0.0f) ;
    (&none_0)->point_0 = _S1;
    (&none_0)->impulse_0 = _S1;
    (&none_0)->other_0 = _S1;
    (&none_0)->key_0 = uint4(int4(int(0)) );
    Record_natural_0 device* _S2 = (&kernelContext_1)->records_0+base_0;
    _S2->ids_0 = packed_uint4(none_0.ids_0) ;
    _S2->point_0 = packed_float4(none_0.point_0) ;
    _S2->impulse_0 = packed_float4(none_0.impulse_0) ;
    _S2->other_0 = packed_float4(none_0.other_0) ;
    _S2->key_0 = packed_uint4(none_0.key_0) ;
    uint _S3 = base_0 + 1U;
    (&none_0)->ids_0.z = _S3;
    Record_natural_0 device* _S4 = (&kernelContext_1)->records_0+_S3;
    _S4->ids_0 = packed_uint4(none_0.ids_0) ;
    _S4->point_0 = packed_float4(none_0.point_0) ;
    _S4->impulse_0 = packed_float4(none_0.impulse_0) ;
    _S4->other_0 = packed_float4(none_0.other_0) ;
    _S4->key_0 = packed_uint4(none_0.key_0) ;
    EngineContactRow_natural_0 c_0 = (&kernelContext_1)->contacts_0[(&kernelContext_1)->params_1->row_offset_0 + i_0];
    uint4 _S5 = uint4(c_0.ids_1) ;
    if((_S5.w) == 0U)
    {
        return;
    }
    float3 n_0 = (float4(c_0.normal_pen_0) ).xyz;
    float4 _S6 = float4(c_0.t1_lambda_n_0) ;
    float4 _S7 = float4(c_0.t2_lambda_t1_0) ;
    float3 jb_0 = - (n_0 * float3(_S6.w)  + _S6.xyz * float3(_S7.w)  + _S7.xyz * float3((float4(c_0.kn_kt_lt2_c0t1_0) ).z) ) * float3((&kernelContext_1)->params_1->h_0) ;
    bool _S8;
    if(all(jb_0 == (float3(0.0f) )))
    {
        _S8 = true;
    }
    else
    {
        _S8 = any(isnan(jb_0));
    }
    if(_S8)
    {
        _S8 = true;
    }
    else
    {
        _S8 = any(isinf(jb_0));
    }
    if(_S8)
    {
        return;
    }
    uint _S9 = _S5.x;
    uint _S10 = _S5.y;
    uint _S11 = (uint4((&kernelContext_1)->colliders_0[_S9].ids_2) ).x;
    EngineBody_natural_0 ba_0 = (&kernelContext_1)->bodies_0[_S11];
    uint _S12 = (uint4((&kernelContext_1)->colliders_0[_S10].ids_2) ).x;
    EngineBody_natural_0 bb_0 = (&kernelContext_1)->bodies_0[_S12];
    float3 _S13 = ((float4(ba_0.pos_inv_mass_0) ).xyz + rotate_q_0(float4(ba_0.rot_0) , (float4(c_0.anchor_a_0) ).xyz) + ((float4(bb_0.pos_inv_mass_0) ).xyz + rotate_q_0(float4(bb_0.rot_0) , (float4(c_0.anchor_b_0) ).xyz))) * float3(0.5f) ;
    ColliderInfo_natural_0 _S14 = (&kernelContext_1)->collider_info_0[_S9];
    uint4 _S15 = uint4(_S14.a_0) ;
    float4 _S16 = float4(_S14.b_0) ;
    ColliderInfo_natural_0 _S17 = (&kernelContext_1)->collider_info_0[_S10];
    uint4 _S18 = uint4(_S17.a_0) ;
    float4 _S19 = float4(_S17.b_0) ;
    uint side_0 = 0U;
    for(;;)
    {
        if(side_0 < 2U)
        {
        }
        else
        {
            break;
        }
        bool _S20 = side_0 == 0U;
        ColliderInfo_0 mine_0;
        if(_S20)
        {
            (&mine_0)->a_0 = _S15;
            (&mine_0)->b_0 = _S16;
        }
        else
        {
            (&mine_0)->a_0 = _S18;
            (&mine_0)->b_0 = _S19;
        }
        ColliderInfo_0 other_1;
        if(_S20)
        {
            (&other_1)->a_0 = _S18;
            (&other_1)->b_0 = _S19;
        }
        else
        {
            (&other_1)->a_0 = _S15;
            (&other_1)->b_0 = _S16;
        }
        uint mine_body_0;
        if(_S20)
        {
            mine_body_0 = _S11;
        }
        else
        {
            mine_body_0 = _S12;
        }
        uint other_body_0;
        if(_S20)
        {
            other_body_0 = _S12;
        }
        else
        {
            other_body_0 = _S11;
        }
        ColliderInfo_0 _S21 = mine_0;
        uint _S22 = mine_0.a_0.x;
        if(_S22 == 4294967295U)
        {
            _S8 = true;
        }
        else
        {
            _S8 = (_S21.a_0.w) == 0U;
        }
        bool _S23;
        if(_S8)
        {
            _S23 = true;
        }
        else
        {
            _S23 = (other_1.a_0.z) == 0U;
        }
        if(_S23)
        {
            side_0 = side_0 + 1U;
            continue;
        }
        float3 j_0;
        if(_S20)
        {
            j_0 = - jb_0;
        }
        else
        {
            j_0 = jb_0;
        }
        float3 push_0;
        if(_S20)
        {
            push_0 = - n_0;
        }
        else
        {
            push_0 = n_0;
        }
        thread ColliderInfo_0 _S24 = other_1;
        float3 _S25 = point_velocity_0(&_S24, other_body_0, _S13, &kernelContext_1);
        thread ColliderInfo_0 _S26 = mine_0;
        float3 _S27 = point_velocity_0(&_S26, mine_body_0, _S13, &kernelContext_1);
        float _S28 = max(dot(_S25 - _S27, push_0), 0.0f);
        thread Record_0 r_0;
        ColliderInfo_0 _S29 = other_1;
        uint _S30;
        if((other_1.a_0.z) == 2U)
        {
            _S30 = _S29.a_0.y;
        }
        else
        {
            _S30 = 4294967295U;
        }
        uint _S31 = base_0 + side_0;
        (&r_0)->ids_0 = uint4(_S22, _S30, _S31, 1U);
        float3 jz_0 = to_zup_0(j_0);
        (&r_0)->point_0 = float4(to_zup_0(_S13), length(jz_0));
        (&r_0)->impulse_0 = float4(jz_0, _S28);
        (&r_0)->other_0 = other_1.b_0;
        uint _S32 = (&kernelContext_1)->params_1->substep_0;
        uint _S33;
        if(_S20)
        {
            _S33 = _S9;
        }
        else
        {
            _S33 = _S10;
        }
        uint _S34;
        if(_S20)
        {
            _S34 = _S10;
        }
        else
        {
            _S34 = _S9;
        }
        (&r_0)->key_0 = uint4(_S32, _S33, _S34, 2U * _S5.z + side_0);
        Record_natural_0 device* _S35 = (&kernelContext_1)->records_0+_S31;
        _S35->ids_0 = packed_uint4(r_0.ids_0) ;
        _S35->point_0 = packed_float4(r_0.point_0) ;
        _S35->impulse_0 = packed_float4(r_0.impulse_0) ;
        _S35->other_0 = packed_float4(r_0.other_0) ;
        _S35->key_0 = packed_uint4(r_0.key_0) ;
        side_0 = side_0 + 1U;
    }
    return;
}

[[kernel]] void ingest_wheels(uint3 id_1 [[thread_position_in_grid]], IngestParams_0 constant* params_3 [[buffer(0)]], uint device* row_counts_2 [[buffer(11)]], uint device* counters_2 [[buffer(5)]], Record_natural_0 device* records_2 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_2 [[buffer(1)]], EngineCollider_natural_0 device* colliders_2 [[buffer(2)]], EngineBody_natural_0 device* bodies_2 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_2 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_2 [[buffer(4)]], packed_uint4 device* vehicle_pool_2 [[buffer(12)]], atomic<uint> device* counts_2 [[buffer(10)]], uint device* sort_2 [[buffer(8)]], Record_natural_0 device* manifolds_2 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_1 = params_3;
    (&kernelContext_2)->row_counts_0 = row_counts_2;
    (&kernelContext_2)->counters_0 = counters_2;
    (&kernelContext_2)->records_0 = records_2;
    (&kernelContext_2)->contacts_0 = contacts_2;
    (&kernelContext_2)->colliders_0 = colliders_2;
    (&kernelContext_2)->bodies_0 = bodies_2;
    (&kernelContext_2)->collider_info_0 = collider_info_2;
    (&kernelContext_2)->start_bodies_0 = start_bodies_2;
    (&kernelContext_2)->vehicle_pool_0 = vehicle_pool_2;
    (&kernelContext_2)->counts_0 = counts_2;
    (&kernelContext_2)->sort_0 = sort_2;
    (&kernelContext_2)->manifolds_0 = manifolds_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_2)->g_scan_0 = &g_scan_2;
    uint i_1 = id_1.x;
    if(i_1 >= (params_3->wheels_0))
    {
        return;
    }
    uint slot_0 = (&kernelContext_2)->params_1->wheel_base_0 + i_1;
    thread Record_0 none_1;
    (&none_1)->ids_0 = uint4(4294967295U, 4294967295U, slot_0, 0U);
    float4 _S36 = float4(0.0f) ;
    (&none_1)->point_0 = _S36;
    (&none_1)->impulse_0 = _S36;
    (&none_1)->other_0 = _S36;
    (&none_1)->key_0 = uint4(int4(int(0)) );
    Record_natural_0 device* _S37 = (&kernelContext_2)->records_0+slot_0;
    _S37->ids_0 = packed_uint4(none_1.ids_0) ;
    _S37->point_0 = packed_float4(none_1.point_0) ;
    _S37->impulse_0 = packed_float4(none_1.impulse_0) ;
    _S37->other_0 = packed_float4(none_1.other_0) ;
    _S37->key_0 = packed_uint4(none_1.key_0) ;
    if(i_1 >= ((uint4(*((&kernelContext_2)->vehicle_pool_0+0U)) ).w))
    {
        return;
    }
    uint wr_0 = (uint4(*((&kernelContext_2)->vehicle_pool_0+1U)) ).y + i_1 * 48U;
    uint4 _S38 = uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 12U))) ;
    uint _S39 = _S38.z;
    bool _S40;
    if((_S39 & 1U) == 0U)
    {
        _S40 = true;
    }
    else
    {
        _S40 = (_S39 & 24U) != 0U;
    }
    if(_S40)
    {
        return;
    }
    uint _S41 = _S38.y;
    ColliderInfo_natural_0 mine_1 = (&kernelContext_2)->collider_info_0[_S41];
    uint4 _S42 = uint4(mine_1.a_0) ;
    uint _S43 = _S42.x;
    if(_S43 == 4294967295U)
    {
        _S40 = true;
    }
    else
    {
        _S40 = (_S42.w) == 0U;
    }
    if(_S40)
    {
        return;
    }
    float4 f_0 = (as_type<float4>((uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 23U))) )));
    float3 j_1 = - ((as_type<float3>(((uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 14U))) ).xyz))) * float3(f_0.x)  - (as_type<float3>(((uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 15U))) ).xyz))) * float3(f_0.y)  - (as_type<float3>(((uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 16U))) ).xyz))) * float3(f_0.z) ) * float3((&kernelContext_2)->params_1->h_0) ;
    if(all(j_1 == (float3(0.0f) )))
    {
        _S40 = true;
    }
    else
    {
        _S40 = any(isnan(j_1));
    }
    if(_S40)
    {
        _S40 = true;
    }
    else
    {
        _S40 = any(isinf(j_1));
    }
    if(_S40)
    {
        return;
    }
    uint vr_0 = (uint4(*((&kernelContext_2)->vehicle_pool_0+1U)) ).x + (uint4(*((&kernelContext_2)->vehicle_pool_0+wr_0)) ).x * 44U;
    uint chassis_0 = (uint4(*((&kernelContext_2)->vehicle_pool_0+vr_0)) ).x;
    thread Record_0 r_1;
    (&r_1)->ids_0 = uint4(_S43, chassis_0, slot_0, 1U);
    float3 jz_1 = to_zup_0(j_1);
    (&r_1)->point_0 = float4(to_zup_0((as_type<float3>(((uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 25U))) ).xyz)))), length(jz_1));
    (&r_1)->impulse_0 = float4(jz_1, 0.0f);
    (&r_1)->other_0 = float4((as_type<float>(((uint4(*((&kernelContext_2)->vehicle_pool_0+(vr_0 + 15U))) ).x))), (float4(mine_1.b_0) ).y, (as_type<float>(((uint4(*((&kernelContext_2)->vehicle_pool_0+(wr_0 + 1U))) ).w))), 0.0f);
    (&r_1)->key_0 = uint4(4294967295U, _S41, chassis_0, i_1);
    Record_natural_0 device* _S44 = (&kernelContext_2)->records_0+slot_0;
    _S44->ids_0 = packed_uint4(r_1.ids_0) ;
    _S44->point_0 = packed_float4(r_1.point_0) ;
    _S44->impulse_0 = packed_float4(r_1.impulse_0) ;
    _S44->other_0 = packed_float4(r_1.other_0) ;
    _S44->key_0 = packed_uint4(r_1.key_0) ;
    return;
}

[[kernel]] void ingest_clear(uint3 id_2 [[thread_position_in_grid]], IngestParams_0 constant* params_4 [[buffer(0)]], uint device* row_counts_3 [[buffer(11)]], uint device* counters_3 [[buffer(5)]], Record_natural_0 device* records_3 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_3 [[buffer(1)]], EngineCollider_natural_0 device* colliders_3 [[buffer(2)]], EngineBody_natural_0 device* bodies_3 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_3 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_3 [[buffer(4)]], packed_uint4 device* vehicle_pool_3 [[buffer(12)]], atomic<uint> device* counts_3 [[buffer(10)]], uint device* sort_3 [[buffer(8)]], Record_natural_0 device* manifolds_3 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_1 = params_4;
    (&kernelContext_3)->row_counts_0 = row_counts_3;
    (&kernelContext_3)->counters_0 = counters_3;
    (&kernelContext_3)->records_0 = records_3;
    (&kernelContext_3)->contacts_0 = contacts_3;
    (&kernelContext_3)->colliders_0 = colliders_3;
    (&kernelContext_3)->bodies_0 = bodies_3;
    (&kernelContext_3)->collider_info_0 = collider_info_3;
    (&kernelContext_3)->start_bodies_0 = start_bodies_3;
    (&kernelContext_3)->vehicle_pool_0 = vehicle_pool_3;
    (&kernelContext_3)->counts_0 = counts_3;
    (&kernelContext_3)->sort_0 = sort_3;
    (&kernelContext_3)->manifolds_0 = manifolds_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_3)->g_scan_0 = &g_scan_3;
    uint _S45 = id_2.x;
    if(_S45 < (2U * params_4->chunk_count_0))
    {
        atomic_store_explicit((&kernelContext_3)->counts_0+_S45, 0U, memory_order_relaxed);
    }
    return;
}

uint record_slot_0(uint i_2, KernelContext_0 thread* kernelContext_4)
{
    uint _S46;
    if(i_2 < (kernelContext_4->params_1->record_count_0))
    {
        _S46 = i_2;
    }
    else
    {
        _S46 = kernelContext_4->params_1->wheel_base_0 + (i_2 - kernelContext_4->params_1->record_count_0);
    }
    return _S46;
}

bool record_live_0(uint slot_1, KernelContext_0 thread* kernelContext_5)
{
    if(slot_1 >= (kernelContext_5->params_1->wheel_base_0))
    {
        return true;
    }
    uint per_0 = 2U * kernelContext_5->params_1->max_rows_0;
    uint _S47 = slot_1 % per_0;
    uint _S48 = _S47 / 2U;
    uint _S49 = slot_1 / per_0;
    return _S48 < (*(kernelContext_5->row_counts_0+_S49));
}

[[kernel]] void ingest_count(uint3 id_3 [[thread_position_in_grid]], IngestParams_0 constant* params_5 [[buffer(0)]], uint device* row_counts_4 [[buffer(11)]], uint device* counters_4 [[buffer(5)]], Record_natural_0 device* records_4 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_4 [[buffer(1)]], EngineCollider_natural_0 device* colliders_4 [[buffer(2)]], EngineBody_natural_0 device* bodies_4 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_4 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_4 [[buffer(4)]], packed_uint4 device* vehicle_pool_4 [[buffer(12)]], atomic<uint> device* counts_4 [[buffer(10)]], uint device* sort_4 [[buffer(8)]], Record_natural_0 device* manifolds_4 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->params_1 = params_5;
    (&kernelContext_6)->row_counts_0 = row_counts_4;
    (&kernelContext_6)->counters_0 = counters_4;
    (&kernelContext_6)->records_0 = records_4;
    (&kernelContext_6)->contacts_0 = contacts_4;
    (&kernelContext_6)->colliders_0 = colliders_4;
    (&kernelContext_6)->bodies_0 = bodies_4;
    (&kernelContext_6)->collider_info_0 = collider_info_4;
    (&kernelContext_6)->start_bodies_0 = start_bodies_4;
    (&kernelContext_6)->vehicle_pool_0 = vehicle_pool_4;
    (&kernelContext_6)->counts_0 = counts_4;
    (&kernelContext_6)->sort_0 = sort_4;
    (&kernelContext_6)->manifolds_0 = manifolds_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_6)->g_scan_0 = &g_scan_4;
    uint _S50 = id_3.x;
    if(_S50 >= (params_5->record_count_0 + params_5->wheels_0))
    {
        return;
    }
    uint _S51 = record_slot_0(_S50, &kernelContext_6);
    bool _S52 = record_live_0(_S51, &kernelContext_6);
    if(!_S52)
    {
        return;
    }
    uint4 _S53 = uint4((*((&kernelContext_6)->records_0+_S51)).ids_0) ;
    if((_S53.w) != 0U)
    {
        uint _S54 = atomic_fetch_add_explicit((&kernelContext_6)->counts_0+_S53.x, 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void ingest_scan(uint3 thread_0 [[thread_position_in_threadgroup]], IngestParams_0 constant* params_6 [[buffer(0)]], uint device* row_counts_5 [[buffer(11)]], uint device* counters_5 [[buffer(5)]], Record_natural_0 device* records_5 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_5 [[buffer(1)]], EngineCollider_natural_0 device* colliders_5 [[buffer(2)]], EngineBody_natural_0 device* bodies_5 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_5 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_5 [[buffer(4)]], packed_uint4 device* vehicle_pool_5 [[buffer(12)]], atomic<uint> device* counts_5 [[buffer(10)]], uint device* sort_5 [[buffer(8)]], Record_natural_0 device* manifolds_5 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->params_1 = params_6;
    (&kernelContext_7)->row_counts_0 = row_counts_5;
    (&kernelContext_7)->counters_0 = counters_5;
    (&kernelContext_7)->records_0 = records_5;
    (&kernelContext_7)->contacts_0 = contacts_5;
    (&kernelContext_7)->colliders_0 = colliders_5;
    (&kernelContext_7)->bodies_0 = bodies_5;
    (&kernelContext_7)->collider_info_0 = collider_info_5;
    (&kernelContext_7)->start_bodies_0 = start_bodies_5;
    (&kernelContext_7)->vehicle_pool_0 = vehicle_pool_5;
    (&kernelContext_7)->counts_0 = counts_5;
    (&kernelContext_7)->sort_0 = sort_5;
    (&kernelContext_7)->manifolds_0 = manifolds_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_7)->g_scan_0 = &g_scan_5;
    uint tid_0 = thread_0.x;
    uint n_1 = params_6->chunk_count_0;
    uint base_1 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(base_1 < n_1)
        {
        }
        else
        {
            break;
        }
        uint k_0 = base_1 + tid_0;
        bool _S55 = k_0 < n_1;
        uint v_2;
        if(_S55)
        {
            uint _S56 = atomic_load_explicit((&kernelContext_7)->counts_0+k_0, memory_order_relaxed);
            v_2 = _S56;
        }
        else
        {
            v_2 = 0U;
        }
        (*(&kernelContext_7)->g_scan_0)[tid_0] = v_2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint step_0 = 1U;
        for(;;)
        {
            if(step_0 < 1024U)
            {
            }
            else
            {
                break;
            }
            uint add_0;
            if(tid_0 >= step_0)
            {
                add_0 = (*(&kernelContext_7)->g_scan_0)[tid_0 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_7)->g_scan_0)[tid_0] = (*(&kernelContext_7)->g_scan_0)[tid_0] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(_S55)
        {
            *((&kernelContext_7)->sort_0+k_0) = carry_0 + (*(&kernelContext_7)->g_scan_0)[tid_0] - v_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_7)->g_scan_0)[int(1023)];
        base_1 = base_1 + 1024U;
        carry_0 = carry_1;
    }
    if(tid_0 == 0U)
    {
        *((&kernelContext_7)->sort_0+n_1) = carry_0;
    }
    return;
}

[[kernel]] void ingest_scatter(uint3 id_4 [[thread_position_in_grid]], IngestParams_0 constant* params_7 [[buffer(0)]], uint device* row_counts_6 [[buffer(11)]], uint device* counters_6 [[buffer(5)]], Record_natural_0 device* records_6 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_6 [[buffer(1)]], EngineCollider_natural_0 device* colliders_6 [[buffer(2)]], EngineBody_natural_0 device* bodies_6 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_6 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_6 [[buffer(4)]], packed_uint4 device* vehicle_pool_6 [[buffer(12)]], atomic<uint> device* counts_6 [[buffer(10)]], uint device* sort_6 [[buffer(8)]], Record_natural_0 device* manifolds_6 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_8;
    (&kernelContext_8)->params_1 = params_7;
    (&kernelContext_8)->row_counts_0 = row_counts_6;
    (&kernelContext_8)->counters_0 = counters_6;
    (&kernelContext_8)->records_0 = records_6;
    (&kernelContext_8)->contacts_0 = contacts_6;
    (&kernelContext_8)->colliders_0 = colliders_6;
    (&kernelContext_8)->bodies_0 = bodies_6;
    (&kernelContext_8)->collider_info_0 = collider_info_6;
    (&kernelContext_8)->start_bodies_0 = start_bodies_6;
    (&kernelContext_8)->vehicle_pool_0 = vehicle_pool_6;
    (&kernelContext_8)->counts_0 = counts_6;
    (&kernelContext_8)->sort_0 = sort_6;
    (&kernelContext_8)->manifolds_0 = manifolds_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_8)->g_scan_0 = &g_scan_6;
    uint _S57 = id_4.x;
    if(_S57 >= (params_7->record_count_0 + params_7->wheels_0))
    {
        return;
    }
    uint _S58 = record_slot_0(_S57, &kernelContext_8);
    bool _S59 = record_live_0(_S58, &kernelContext_8);
    if(!_S59)
    {
        return;
    }
    uint4 _S60 = uint4((*((&kernelContext_8)->records_0+_S58)).ids_0) ;
    if((_S60.w) == 0U)
    {
        return;
    }
    uint n_2 = (&kernelContext_8)->params_1->chunk_count_0;
    uint _S61 = _S60.x;
    uint at_0 = atomic_fetch_add_explicit((&kernelContext_8)->counts_0+((&kernelContext_8)->params_1->chunk_count_0 + _S61), 1U, memory_order_relaxed);
    *((&kernelContext_8)->sort_0+(2U * n_2 + 1U + *((&kernelContext_8)->sort_0+_S61) + at_0)) = _S58;
    return;
}

bool record_before_0(uint a_1, uint b_1, KernelContext_0 thread* kernelContext_9)
{
    uint4 _S62 = uint4((kernelContext_9->records_0+a_1)->key_0) ;
    uint4 _S63 = uint4((kernelContext_9->records_0+b_1)->key_0) ;
    uint q_1 = 0U;
    for(;;)
    {
        if(q_1 < 4U)
        {
        }
        else
        {
            break;
        }
        uint _S64 = q_1;
        uint _S65 = q_1;
        if((_S62[q_1]) != (_S63[q_1]))
        {
            return (_S62[_S64]) < (_S63[_S65]);
        }
        q_1 = q_1 + 1U;
    }
    return a_1 < b_1;
}

[[kernel]] void ingest_merge(uint3 id_5 [[thread_position_in_grid]], IngestParams_0 constant* params_8 [[buffer(0)]], uint device* row_counts_7 [[buffer(11)]], uint device* counters_7 [[buffer(5)]], Record_natural_0 device* records_7 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_7 [[buffer(1)]], EngineCollider_natural_0 device* colliders_7 [[buffer(2)]], EngineBody_natural_0 device* bodies_7 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_7 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_7 [[buffer(4)]], packed_uint4 device* vehicle_pool_7 [[buffer(12)]], atomic<uint> device* counts_7 [[buffer(10)]], uint device* sort_7 [[buffer(8)]], Record_natural_0 device* manifolds_7 [[buffer(9)]])
{
    uint b_2;
    uint m_0;
    thread KernelContext_0 kernelContext_10;
    (&kernelContext_10)->params_1 = params_8;
    (&kernelContext_10)->row_counts_0 = row_counts_7;
    (&kernelContext_10)->counters_0 = counters_7;
    (&kernelContext_10)->records_0 = records_7;
    (&kernelContext_10)->contacts_0 = contacts_7;
    (&kernelContext_10)->colliders_0 = colliders_7;
    (&kernelContext_10)->bodies_0 = bodies_7;
    (&kernelContext_10)->collider_info_0 = collider_info_7;
    (&kernelContext_10)->start_bodies_0 = start_bodies_7;
    (&kernelContext_10)->vehicle_pool_0 = vehicle_pool_7;
    (&kernelContext_10)->counts_0 = counts_7;
    (&kernelContext_10)->sort_0 = sort_7;
    (&kernelContext_10)->manifolds_0 = manifolds_7;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_10)->g_scan_0 = &g_scan_7;
    uint k_1 = id_5.x;
    uint n_3 = params_8->chunk_count_0;
    if(k_1 >= (params_8->chunk_count_0))
    {
        return;
    }
    uint device* _S66 = (&kernelContext_10)->sort_0+k_1;
    uint begin_0 = *_S66;
    uint _S67 = *((&kernelContext_10)->sort_0+(k_1 + 1U));
    uint _S68 = 2U * n_3 + 1U;
    uint a_2 = *_S66 + 1U;
    for(;;)
    {
        if(a_2 < _S67)
        {
        }
        else
        {
            break;
        }
        uint v_3 = *((&kernelContext_10)->sort_0+(_S68 + a_2));
        b_2 = a_2;
        for(;;)
        {
            bool _S69;
            if(b_2 > begin_0)
            {
                bool _S70 = record_before_0(v_3, *((&kernelContext_10)->sort_0+(_S68 + b_2 - 1U)), &kernelContext_10);
                _S69 = _S70;
            }
            else
            {
                _S69 = false;
            }
            if(_S69)
            {
            }
            else
            {
                break;
            }
            uint _S71 = _S68 + b_2;
            *((&kernelContext_10)->sort_0+_S71) = *((&kernelContext_10)->sort_0+(_S71 - 1U));
            b_2 = b_2 - 1U;
        }
        *((&kernelContext_10)->sort_0+(_S68 + b_2)) = v_3;
        a_2 = a_2 + 1U;
    }
    a_2 = begin_0;
    b_2 = 0U;
    for(;;)
    {
        if(a_2 < _S67)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_2 = *((&kernelContext_10)->records_0+*((&kernelContext_10)->sort_0+(_S68 + a_2)));
        uint q_2 = 0U;
        for(;;)
        {
            if(q_2 < b_2)
            {
            }
            else
            {
                m_0 = 4294967295U;
                break;
            }
            if(((uint4(((&kernelContext_10)->manifolds_0+(begin_0 + q_2))->ids_0) ).y) == ((uint4(r_2.ids_0) ).y))
            {
                m_0 = q_2;
                break;
            }
            q_2 = q_2 + 1U;
        }
        if(m_0 == 4294967295U)
        {
            *((&kernelContext_10)->manifolds_0+(begin_0 + b_2)) = r_2;
            b_2 = b_2 + 1U;
            a_2 = a_2 + 1U;
            continue;
        }
        uint _S72 = begin_0 + m_0;
        Record_natural_0 device* _S73 = (&kernelContext_10)->manifolds_0+_S72;
        float4 _S74 = float4((*_S73).point_0) ;
        float4 _S75 = float4((*_S73).impulse_0) ;
        float4 _S76 = float4((*_S73).other_0) ;
        uint4 _S77 = uint4((*_S73).key_0) ;
        thread Record_0 acc_0;
        (&acc_0)->ids_0 = uint4((*_S73).ids_0) ;
        (&acc_0)->point_0 = _S74;
        (&acc_0)->impulse_0 = _S75;
        (&acc_0)->other_0 = _S76;
        (&acc_0)->key_0 = _S77;
        float w0_0 = (&acc_0)->point_0.w;
        float4 _S78 = float4(r_2.point_0) ;
        float w_0 = _S78.w;
        float total_0 = w0_0 + w_0;
        if(total_0 > 0.0f)
        {
            (&acc_0)->point_0.xyz = ((&acc_0)->point_0.xyz * float3(w0_0)  + _S78.xyz * float3(w_0) ) / float3(total_0) ;
            (&acc_0)->impulse_0.w = ((&acc_0)->impulse_0.w * w0_0 + (float4(r_2.impulse_0) ).w * w_0) / total_0;
        }
        (&acc_0)->impulse_0.xyz = (&acc_0)->impulse_0.xyz + (float4(r_2.impulse_0) ).xyz;
        (&acc_0)->point_0.w = total_0;
        Record_natural_0 device* _S79 = (&kernelContext_10)->manifolds_0+_S72;
        _S79->ids_0 = packed_uint4(acc_0.ids_0) ;
        _S79->point_0 = packed_float4(acc_0.point_0) ;
        _S79->impulse_0 = packed_float4(acc_0.impulse_0) ;
        _S79->other_0 = packed_float4(acc_0.other_0) ;
        _S79->key_0 = packed_uint4(acc_0.key_0) ;
        a_2 = a_2 + 1U;
    }
    *((&kernelContext_10)->sort_0+(n_3 + 1U + k_1)) = b_2;
    return;
}

