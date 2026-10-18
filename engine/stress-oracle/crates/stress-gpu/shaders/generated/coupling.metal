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
    uint pad_0;
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

[[kernel]] void ingest_rows(uint3 id_0 [[thread_position_in_grid]], IngestParams_0 constant* params_2 [[buffer(0)]], uint device* row_counts_1 [[buffer(11)]], uint device* counters_1 [[buffer(5)]], Record_natural_0 device* records_1 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_1 [[buffer(1)]], EngineCollider_natural_0 device* colliders_1 [[buffer(2)]], EngineBody_natural_0 device* bodies_1 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_1 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_1 [[buffer(4)]], atomic<uint> device* counts_1 [[buffer(10)]], uint device* sort_1 [[buffer(8)]], Record_natural_0 device* manifolds_1 [[buffer(9)]])
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

[[kernel]] void ingest_clear(uint3 id_1 [[thread_position_in_grid]], IngestParams_0 constant* params_3 [[buffer(0)]], uint device* row_counts_2 [[buffer(11)]], uint device* counters_2 [[buffer(5)]], Record_natural_0 device* records_2 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_2 [[buffer(1)]], EngineCollider_natural_0 device* colliders_2 [[buffer(2)]], EngineBody_natural_0 device* bodies_2 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_2 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_2 [[buffer(4)]], atomic<uint> device* counts_2 [[buffer(10)]], uint device* sort_2 [[buffer(8)]], Record_natural_0 device* manifolds_2 [[buffer(9)]])
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
    (&kernelContext_2)->counts_0 = counts_2;
    (&kernelContext_2)->sort_0 = sort_2;
    (&kernelContext_2)->manifolds_0 = manifolds_2;
    threadgroup array<uint, int(1024)> g_scan_2;
    (&kernelContext_2)->g_scan_0 = &g_scan_2;
    uint _S36 = id_1.x;
    if(_S36 < (2U * params_3->chunk_count_0))
    {
        atomic_store_explicit((&kernelContext_2)->counts_0+_S36, 0U, memory_order_relaxed);
    }
    return;
}

bool record_live_0(uint slot_0, KernelContext_0 thread* kernelContext_3)
{
    uint per_0 = 2U * kernelContext_3->params_1->max_rows_0;
    uint _S37 = slot_0 % per_0;
    uint _S38 = _S37 / 2U;
    uint _S39 = slot_0 / per_0;
    return _S38 < (*(kernelContext_3->row_counts_0+_S39));
}

[[kernel]] void ingest_count(uint3 id_2 [[thread_position_in_grid]], IngestParams_0 constant* params_4 [[buffer(0)]], uint device* row_counts_3 [[buffer(11)]], uint device* counters_3 [[buffer(5)]], Record_natural_0 device* records_3 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_3 [[buffer(1)]], EngineCollider_natural_0 device* colliders_3 [[buffer(2)]], EngineBody_natural_0 device* bodies_3 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_3 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_3 [[buffer(4)]], atomic<uint> device* counts_3 [[buffer(10)]], uint device* sort_3 [[buffer(8)]], Record_natural_0 device* manifolds_3 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_1 = params_4;
    (&kernelContext_4)->row_counts_0 = row_counts_3;
    (&kernelContext_4)->counters_0 = counters_3;
    (&kernelContext_4)->records_0 = records_3;
    (&kernelContext_4)->contacts_0 = contacts_3;
    (&kernelContext_4)->colliders_0 = colliders_3;
    (&kernelContext_4)->bodies_0 = bodies_3;
    (&kernelContext_4)->collider_info_0 = collider_info_3;
    (&kernelContext_4)->start_bodies_0 = start_bodies_3;
    (&kernelContext_4)->counts_0 = counts_3;
    (&kernelContext_4)->sort_0 = sort_3;
    (&kernelContext_4)->manifolds_0 = manifolds_3;
    threadgroup array<uint, int(1024)> g_scan_3;
    (&kernelContext_4)->g_scan_0 = &g_scan_3;
    uint _S40 = id_2.x;
    bool _S41;
    if(_S40 >= (params_4->record_count_0))
    {
        _S41 = true;
    }
    else
    {
        bool _S42 = record_live_0(_S40, &kernelContext_4);
        _S41 = !_S42;
    }
    if(_S41)
    {
        return;
    }
    uint4 _S43 = uint4((*((&kernelContext_4)->records_0+_S40)).ids_0) ;
    if((_S43.w) != 0U)
    {
        uint _S44 = atomic_fetch_add_explicit((&kernelContext_4)->counts_0+_S43.x, 1U, memory_order_relaxed);
    }
    return;
}

[[kernel]] void ingest_scan(uint3 thread_0 [[thread_position_in_threadgroup]], IngestParams_0 constant* params_5 [[buffer(0)]], uint device* row_counts_4 [[buffer(11)]], uint device* counters_4 [[buffer(5)]], Record_natural_0 device* records_4 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_4 [[buffer(1)]], EngineCollider_natural_0 device* colliders_4 [[buffer(2)]], EngineBody_natural_0 device* bodies_4 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_4 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_4 [[buffer(4)]], atomic<uint> device* counts_4 [[buffer(10)]], uint device* sort_4 [[buffer(8)]], Record_natural_0 device* manifolds_4 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_1 = params_5;
    (&kernelContext_5)->row_counts_0 = row_counts_4;
    (&kernelContext_5)->counters_0 = counters_4;
    (&kernelContext_5)->records_0 = records_4;
    (&kernelContext_5)->contacts_0 = contacts_4;
    (&kernelContext_5)->colliders_0 = colliders_4;
    (&kernelContext_5)->bodies_0 = bodies_4;
    (&kernelContext_5)->collider_info_0 = collider_info_4;
    (&kernelContext_5)->start_bodies_0 = start_bodies_4;
    (&kernelContext_5)->counts_0 = counts_4;
    (&kernelContext_5)->sort_0 = sort_4;
    (&kernelContext_5)->manifolds_0 = manifolds_4;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_5)->g_scan_0 = &g_scan_4;
    uint tid_0 = thread_0.x;
    uint n_1 = params_5->chunk_count_0;
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
        bool _S45 = k_0 < n_1;
        uint v_2;
        if(_S45)
        {
            uint _S46 = atomic_load_explicit((&kernelContext_5)->counts_0+k_0, memory_order_relaxed);
            v_2 = _S46;
        }
        else
        {
            v_2 = 0U;
        }
        (*(&kernelContext_5)->g_scan_0)[tid_0] = v_2;
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
                add_0 = (*(&kernelContext_5)->g_scan_0)[tid_0 - step_0];
            }
            else
            {
                add_0 = 0U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            (*(&kernelContext_5)->g_scan_0)[tid_0] = (*(&kernelContext_5)->g_scan_0)[tid_0] + add_0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            step_0 = step_0 << 1U;
        }
        if(_S45)
        {
            *((&kernelContext_5)->sort_0+k_0) = carry_0 + (*(&kernelContext_5)->g_scan_0)[tid_0] - v_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_5)->g_scan_0)[int(1023)];
        base_1 = base_1 + 1024U;
        carry_0 = carry_1;
    }
    if(tid_0 == 0U)
    {
        *((&kernelContext_5)->sort_0+n_1) = carry_0;
    }
    return;
}

[[kernel]] void ingest_scatter(uint3 id_3 [[thread_position_in_grid]], IngestParams_0 constant* params_6 [[buffer(0)]], uint device* row_counts_5 [[buffer(11)]], uint device* counters_5 [[buffer(5)]], Record_natural_0 device* records_5 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_5 [[buffer(1)]], EngineCollider_natural_0 device* colliders_5 [[buffer(2)]], EngineBody_natural_0 device* bodies_5 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_5 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_5 [[buffer(4)]], atomic<uint> device* counts_5 [[buffer(10)]], uint device* sort_5 [[buffer(8)]], Record_natural_0 device* manifolds_5 [[buffer(9)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->params_1 = params_6;
    (&kernelContext_6)->row_counts_0 = row_counts_5;
    (&kernelContext_6)->counters_0 = counters_5;
    (&kernelContext_6)->records_0 = records_5;
    (&kernelContext_6)->contacts_0 = contacts_5;
    (&kernelContext_6)->colliders_0 = colliders_5;
    (&kernelContext_6)->bodies_0 = bodies_5;
    (&kernelContext_6)->collider_info_0 = collider_info_5;
    (&kernelContext_6)->start_bodies_0 = start_bodies_5;
    (&kernelContext_6)->counts_0 = counts_5;
    (&kernelContext_6)->sort_0 = sort_5;
    (&kernelContext_6)->manifolds_0 = manifolds_5;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_6)->g_scan_0 = &g_scan_5;
    uint _S47 = id_3.x;
    bool _S48;
    if(_S47 >= (params_6->record_count_0))
    {
        _S48 = true;
    }
    else
    {
        bool _S49 = record_live_0(_S47, &kernelContext_6);
        _S48 = !_S49;
    }
    if(_S48)
    {
        return;
    }
    uint4 _S50 = uint4((*((&kernelContext_6)->records_0+_S47)).ids_0) ;
    if((_S50.w) == 0U)
    {
        return;
    }
    uint n_2 = (&kernelContext_6)->params_1->chunk_count_0;
    uint _S51 = _S50.x;
    uint at_0 = atomic_fetch_add_explicit((&kernelContext_6)->counts_0+((&kernelContext_6)->params_1->chunk_count_0 + _S51), 1U, memory_order_relaxed);
    *((&kernelContext_6)->sort_0+(2U * n_2 + 1U + *((&kernelContext_6)->sort_0+_S51) + at_0)) = _S47;
    return;
}

bool record_before_0(uint a_1, uint b_1, KernelContext_0 thread* kernelContext_7)
{
    uint4 _S52 = uint4((kernelContext_7->records_0+a_1)->key_0) ;
    uint4 _S53 = uint4((kernelContext_7->records_0+b_1)->key_0) ;
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
        uint _S54 = q_1;
        uint _S55 = q_1;
        if((_S52[q_1]) != (_S53[q_1]))
        {
            return (_S52[_S54]) < (_S53[_S55]);
        }
        q_1 = q_1 + 1U;
    }
    return a_1 < b_1;
}

[[kernel]] void ingest_merge(uint3 id_4 [[thread_position_in_grid]], IngestParams_0 constant* params_7 [[buffer(0)]], uint device* row_counts_6 [[buffer(11)]], uint device* counters_6 [[buffer(5)]], Record_natural_0 device* records_6 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_6 [[buffer(1)]], EngineCollider_natural_0 device* colliders_6 [[buffer(2)]], EngineBody_natural_0 device* bodies_6 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_6 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_6 [[buffer(4)]], atomic<uint> device* counts_6 [[buffer(10)]], uint device* sort_6 [[buffer(8)]], Record_natural_0 device* manifolds_6 [[buffer(9)]])
{
    uint b_2;
    uint m_0;
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
    (&kernelContext_8)->counts_0 = counts_6;
    (&kernelContext_8)->sort_0 = sort_6;
    (&kernelContext_8)->manifolds_0 = manifolds_6;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_8)->g_scan_0 = &g_scan_6;
    uint k_1 = id_4.x;
    uint n_3 = params_7->chunk_count_0;
    if(k_1 >= (params_7->chunk_count_0))
    {
        return;
    }
    uint device* _S56 = (&kernelContext_8)->sort_0+k_1;
    uint begin_0 = *_S56;
    uint _S57 = *((&kernelContext_8)->sort_0+(k_1 + 1U));
    uint _S58 = 2U * n_3 + 1U;
    uint a_2 = *_S56 + 1U;
    for(;;)
    {
        if(a_2 < _S57)
        {
        }
        else
        {
            break;
        }
        uint v_3 = *((&kernelContext_8)->sort_0+(_S58 + a_2));
        b_2 = a_2;
        for(;;)
        {
            bool _S59;
            if(b_2 > begin_0)
            {
                bool _S60 = record_before_0(v_3, *((&kernelContext_8)->sort_0+(_S58 + b_2 - 1U)), &kernelContext_8);
                _S59 = _S60;
            }
            else
            {
                _S59 = false;
            }
            if(_S59)
            {
            }
            else
            {
                break;
            }
            uint _S61 = _S58 + b_2;
            *((&kernelContext_8)->sort_0+_S61) = *((&kernelContext_8)->sort_0+(_S61 - 1U));
            b_2 = b_2 - 1U;
        }
        *((&kernelContext_8)->sort_0+(_S58 + b_2)) = v_3;
        a_2 = a_2 + 1U;
    }
    a_2 = begin_0;
    b_2 = 0U;
    for(;;)
    {
        if(a_2 < _S57)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_1 = *((&kernelContext_8)->records_0+*((&kernelContext_8)->sort_0+(_S58 + a_2)));
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
            if(((uint4(((&kernelContext_8)->manifolds_0+(begin_0 + q_2))->ids_0) ).y) == ((uint4(r_1.ids_0) ).y))
            {
                m_0 = q_2;
                break;
            }
            q_2 = q_2 + 1U;
        }
        if(m_0 == 4294967295U)
        {
            *((&kernelContext_8)->manifolds_0+(begin_0 + b_2)) = r_1;
            b_2 = b_2 + 1U;
            a_2 = a_2 + 1U;
            continue;
        }
        uint _S62 = begin_0 + m_0;
        Record_natural_0 device* _S63 = (&kernelContext_8)->manifolds_0+_S62;
        float4 _S64 = float4((*_S63).point_0) ;
        float4 _S65 = float4((*_S63).impulse_0) ;
        float4 _S66 = float4((*_S63).other_0) ;
        uint4 _S67 = uint4((*_S63).key_0) ;
        thread Record_0 acc_0;
        (&acc_0)->ids_0 = uint4((*_S63).ids_0) ;
        (&acc_0)->point_0 = _S64;
        (&acc_0)->impulse_0 = _S65;
        (&acc_0)->other_0 = _S66;
        (&acc_0)->key_0 = _S67;
        float w0_0 = (&acc_0)->point_0.w;
        float4 _S68 = float4(r_1.point_0) ;
        float w_0 = _S68.w;
        float total_0 = w0_0 + w_0;
        if(total_0 > 0.0f)
        {
            (&acc_0)->point_0.xyz = ((&acc_0)->point_0.xyz * float3(w0_0)  + _S68.xyz * float3(w_0) ) / float3(total_0) ;
            (&acc_0)->impulse_0.w = ((&acc_0)->impulse_0.w * w0_0 + (float4(r_1.impulse_0) ).w * w_0) / total_0;
        }
        (&acc_0)->impulse_0.xyz = (&acc_0)->impulse_0.xyz + (float4(r_1.impulse_0) ).xyz;
        (&acc_0)->point_0.w = total_0;
        Record_natural_0 device* _S69 = (&kernelContext_8)->manifolds_0+_S62;
        _S69->ids_0 = packed_uint4(acc_0.ids_0) ;
        _S69->point_0 = packed_float4(acc_0.point_0) ;
        _S69->impulse_0 = packed_float4(acc_0.impulse_0) ;
        _S69->other_0 = packed_float4(acc_0.other_0) ;
        _S69->key_0 = packed_uint4(acc_0.key_0) ;
        a_2 = a_2 + 1U;
    }
    *((&kernelContext_8)->sort_0+(n_3 + 1U + k_1)) = b_2;
    return;
}

