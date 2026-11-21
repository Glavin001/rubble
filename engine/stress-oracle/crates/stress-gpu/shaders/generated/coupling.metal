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
    uint gate_0;
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
    uint device* scan_blocks_0;
    Record_natural_0 device* manifolds_0;
    uint device* gate_1;
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

[[kernel]] void ingest_rows(uint3 id_0 [[thread_position_in_grid]], IngestParams_0 constant* params_2 [[buffer(0)]], uint device* row_counts_1 [[buffer(11)]], uint device* counters_1 [[buffer(5)]], Record_natural_0 device* records_1 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_1 [[buffer(1)]], EngineCollider_natural_0 device* colliders_1 [[buffer(2)]], EngineBody_natural_0 device* bodies_1 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_1 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_1 [[buffer(4)]], atomic<uint> device* counts_1 [[buffer(10)]], uint device* sort_1 [[buffer(8)]], uint device* scan_blocks_1 [[buffer(12)]], Record_natural_0 device* manifolds_1 [[buffer(9)]], uint device* gate_2 [[buffer(13)]])
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
    (&kernelContext_1)->scan_blocks_0 = scan_blocks_1;
    (&kernelContext_1)->manifolds_0 = manifolds_1;
    (&kernelContext_1)->gate_1 = gate_2;
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

[[kernel]] void ingest_clear(uint3 id_1 [[thread_position_in_grid]], IngestParams_0 constant* params_3 [[buffer(0)]], uint device* row_counts_2 [[buffer(11)]], uint device* counters_2 [[buffer(5)]], Record_natural_0 device* records_2 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_2 [[buffer(1)]], EngineCollider_natural_0 device* colliders_2 [[buffer(2)]], EngineBody_natural_0 device* bodies_2 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_2 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_2 [[buffer(4)]], atomic<uint> device* counts_2 [[buffer(10)]], uint device* sort_2 [[buffer(8)]], uint device* scan_blocks_2 [[buffer(12)]], Record_natural_0 device* manifolds_2 [[buffer(9)]], uint device* gate_3 [[buffer(13)]])
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
    (&kernelContext_2)->scan_blocks_0 = scan_blocks_2;
    (&kernelContext_2)->manifolds_0 = manifolds_2;
    (&kernelContext_2)->gate_1 = gate_3;
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

[[kernel]] void ingest_count(uint3 id_2 [[thread_position_in_grid]], IngestParams_0 constant* params_4 [[buffer(0)]], uint device* row_counts_3 [[buffer(11)]], uint device* counters_3 [[buffer(5)]], Record_natural_0 device* records_3 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_3 [[buffer(1)]], EngineCollider_natural_0 device* colliders_3 [[buffer(2)]], EngineBody_natural_0 device* bodies_3 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_3 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_3 [[buffer(4)]], atomic<uint> device* counts_3 [[buffer(10)]], uint device* sort_3 [[buffer(8)]], uint device* scan_blocks_3 [[buffer(12)]], Record_natural_0 device* manifolds_3 [[buffer(9)]], uint device* gate_4 [[buffer(13)]])
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
    (&kernelContext_4)->scan_blocks_0 = scan_blocks_3;
    (&kernelContext_4)->manifolds_0 = manifolds_3;
    (&kernelContext_4)->gate_1 = gate_4;
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

void group_scan_0(uint tid_0, KernelContext_0 thread* kernelContext_5)
{
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
            add_0 = (*kernelContext_5->g_scan_0)[tid_0 - step_0];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*kernelContext_5->g_scan_0)[tid_0] = (*kernelContext_5->g_scan_0)[tid_0] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_0 = step_0 << 1U;
    }
    return;
}

[[kernel]] void ingest_scan(uint3 thread_0 [[thread_position_in_threadgroup]], uint3 group_0 [[threadgroup_position_in_grid]], IngestParams_0 constant* params_5 [[buffer(0)]], uint device* row_counts_4 [[buffer(11)]], uint device* counters_4 [[buffer(5)]], Record_natural_0 device* records_4 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_4 [[buffer(1)]], EngineCollider_natural_0 device* colliders_4 [[buffer(2)]], EngineBody_natural_0 device* bodies_4 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_4 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_4 [[buffer(4)]], atomic<uint> device* counts_4 [[buffer(10)]], uint device* sort_4 [[buffer(8)]], uint device* scan_blocks_4 [[buffer(12)]], Record_natural_0 device* manifolds_4 [[buffer(9)]], uint device* gate_5 [[buffer(13)]])
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
    (&kernelContext_6)->counts_0 = counts_4;
    (&kernelContext_6)->sort_0 = sort_4;
    (&kernelContext_6)->scan_blocks_0 = scan_blocks_4;
    (&kernelContext_6)->manifolds_0 = manifolds_4;
    (&kernelContext_6)->gate_1 = gate_5;
    threadgroup array<uint, int(1024)> g_scan_4;
    (&kernelContext_6)->g_scan_0 = &g_scan_4;
    uint tid_1 = thread_0.x;
    uint _S45 = group_0.x;
    uint k_0 = _S45 * 1024U + tid_1;
    bool _S46 = k_0 < (params_5->chunk_count_0);
    uint v_2;
    if(_S46)
    {
        uint _S47 = atomic_load_explicit((&kernelContext_6)->counts_0+k_0, memory_order_relaxed);
        v_2 = _S47;
    }
    else
    {
        v_2 = 0U;
    }
    (*(&kernelContext_6)->g_scan_0)[tid_1] = v_2;
    group_scan_0(tid_1, &kernelContext_6);
    if(_S46)
    {
        *((&kernelContext_6)->sort_0+k_0) = (*(&kernelContext_6)->g_scan_0)[tid_1] - v_2;
    }
    if(tid_1 == 1023U)
    {
        *((&kernelContext_6)->scan_blocks_0+_S45) = (*(&kernelContext_6)->g_scan_0)[int(1023)];
    }
    return;
}

[[kernel]] void ingest_scan_blocks(uint3 thread_1 [[thread_position_in_threadgroup]], IngestParams_0 constant* params_6 [[buffer(0)]], uint device* row_counts_5 [[buffer(11)]], uint device* counters_5 [[buffer(5)]], Record_natural_0 device* records_5 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_5 [[buffer(1)]], EngineCollider_natural_0 device* colliders_5 [[buffer(2)]], EngineBody_natural_0 device* bodies_5 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_5 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_5 [[buffer(4)]], atomic<uint> device* counts_5 [[buffer(10)]], uint device* sort_5 [[buffer(8)]], uint device* scan_blocks_5 [[buffer(12)]], Record_natural_0 device* manifolds_5 [[buffer(9)]], uint device* gate_6 [[buffer(13)]])
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
    (&kernelContext_7)->counts_0 = counts_5;
    (&kernelContext_7)->sort_0 = sort_5;
    (&kernelContext_7)->scan_blocks_0 = scan_blocks_5;
    (&kernelContext_7)->manifolds_0 = manifolds_5;
    (&kernelContext_7)->gate_1 = gate_6;
    threadgroup array<uint, int(1024)> g_scan_5;
    (&kernelContext_7)->g_scan_0 = &g_scan_5;
    uint tid_2 = thread_1.x;
    uint _S48 = (params_6->chunk_count_0 + 1023U) / 1024U;
    uint base_1 = 0U;
    uint carry_0 = 0U;
    for(;;)
    {
        if(base_1 < _S48)
        {
        }
        else
        {
            break;
        }
        uint b_1 = base_1 + tid_2;
        bool _S49 = b_1 < _S48;
        uint v_3;
        if(_S49)
        {
            v_3 = *((&kernelContext_7)->scan_blocks_0+b_1);
        }
        else
        {
            v_3 = 0U;
        }
        (*(&kernelContext_7)->g_scan_0)[tid_2] = v_3;
        group_scan_0(tid_2, &kernelContext_7);
        if(_S49)
        {
            *((&kernelContext_7)->scan_blocks_0+b_1) = carry_0 + (*(&kernelContext_7)->g_scan_0)[tid_2] - v_3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint carry_1 = carry_0 + (*(&kernelContext_7)->g_scan_0)[int(1023)];
        base_1 = base_1 + 1024U;
        carry_0 = carry_1;
    }
    if(tid_2 == 0U)
    {
        *((&kernelContext_7)->sort_0+params_6->chunk_count_0) = carry_0;
    }
    return;
}

[[kernel]] void ingest_scan_add(uint3 id_3 [[thread_position_in_grid]], IngestParams_0 constant* params_7 [[buffer(0)]], uint device* row_counts_6 [[buffer(11)]], uint device* counters_6 [[buffer(5)]], Record_natural_0 device* records_6 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_6 [[buffer(1)]], EngineCollider_natural_0 device* colliders_6 [[buffer(2)]], EngineBody_natural_0 device* bodies_6 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_6 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_6 [[buffer(4)]], atomic<uint> device* counts_6 [[buffer(10)]], uint device* sort_6 [[buffer(8)]], uint device* scan_blocks_6 [[buffer(12)]], Record_natural_0 device* manifolds_6 [[buffer(9)]], uint device* gate_7 [[buffer(13)]])
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
    (&kernelContext_8)->counts_0 = counts_6;
    (&kernelContext_8)->sort_0 = sort_6;
    (&kernelContext_8)->scan_blocks_0 = scan_blocks_6;
    (&kernelContext_8)->manifolds_0 = manifolds_6;
    (&kernelContext_8)->gate_1 = gate_7;
    threadgroup array<uint, int(1024)> g_scan_6;
    (&kernelContext_8)->g_scan_0 = &g_scan_6;
    uint k_1 = id_3.x;
    if(k_1 < (params_7->chunk_count_0))
    {
        uint device* _S50 = (&kernelContext_8)->sort_0+k_1;
        *_S50 = *_S50 + *((&kernelContext_8)->scan_blocks_0+k_1 / 1024U);
    }
    return;
}

[[kernel]] void ingest_scatter(uint3 id_4 [[thread_position_in_grid]], IngestParams_0 constant* params_8 [[buffer(0)]], uint device* row_counts_7 [[buffer(11)]], uint device* counters_7 [[buffer(5)]], Record_natural_0 device* records_7 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_7 [[buffer(1)]], EngineCollider_natural_0 device* colliders_7 [[buffer(2)]], EngineBody_natural_0 device* bodies_7 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_7 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_7 [[buffer(4)]], atomic<uint> device* counts_7 [[buffer(10)]], uint device* sort_7 [[buffer(8)]], uint device* scan_blocks_7 [[buffer(12)]], Record_natural_0 device* manifolds_7 [[buffer(9)]], uint device* gate_8 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_9;
    (&kernelContext_9)->params_1 = params_8;
    (&kernelContext_9)->row_counts_0 = row_counts_7;
    (&kernelContext_9)->counters_0 = counters_7;
    (&kernelContext_9)->records_0 = records_7;
    (&kernelContext_9)->contacts_0 = contacts_7;
    (&kernelContext_9)->colliders_0 = colliders_7;
    (&kernelContext_9)->bodies_0 = bodies_7;
    (&kernelContext_9)->collider_info_0 = collider_info_7;
    (&kernelContext_9)->start_bodies_0 = start_bodies_7;
    (&kernelContext_9)->counts_0 = counts_7;
    (&kernelContext_9)->sort_0 = sort_7;
    (&kernelContext_9)->scan_blocks_0 = scan_blocks_7;
    (&kernelContext_9)->manifolds_0 = manifolds_7;
    (&kernelContext_9)->gate_1 = gate_8;
    threadgroup array<uint, int(1024)> g_scan_7;
    (&kernelContext_9)->g_scan_0 = &g_scan_7;
    uint _S51 = id_4.x;
    bool _S52;
    if(_S51 >= (params_8->record_count_0))
    {
        _S52 = true;
    }
    else
    {
        bool _S53 = record_live_0(_S51, &kernelContext_9);
        _S52 = !_S53;
    }
    if(_S52)
    {
        return;
    }
    uint4 _S54 = uint4((*((&kernelContext_9)->records_0+_S51)).ids_0) ;
    if((_S54.w) == 0U)
    {
        return;
    }
    uint n_1 = (&kernelContext_9)->params_1->chunk_count_0;
    uint _S55 = _S54.x;
    uint at_0 = atomic_fetch_add_explicit((&kernelContext_9)->counts_0+((&kernelContext_9)->params_1->chunk_count_0 + _S55), 1U, memory_order_relaxed);
    *((&kernelContext_9)->sort_0+(2U * n_1 + 1U + *((&kernelContext_9)->sort_0+_S55) + at_0)) = _S51;
    return;
}

bool record_before_0(uint a_1, uint b_2, KernelContext_0 thread* kernelContext_10)
{
    uint4 _S56 = uint4((kernelContext_10->records_0+a_1)->key_0) ;
    uint4 _S57 = uint4((kernelContext_10->records_0+b_2)->key_0) ;
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
        uint _S58 = q_1;
        uint _S59 = q_1;
        if((_S56[q_1]) != (_S57[q_1]))
        {
            return (_S56[_S58]) < (_S57[_S59]);
        }
        q_1 = q_1 + 1U;
    }
    return a_1 < b_2;
}

[[kernel]] void ingest_merge(uint3 id_5 [[thread_position_in_grid]], IngestParams_0 constant* params_9 [[buffer(0)]], uint device* row_counts_8 [[buffer(11)]], uint device* counters_8 [[buffer(5)]], Record_natural_0 device* records_8 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_8 [[buffer(1)]], EngineCollider_natural_0 device* colliders_8 [[buffer(2)]], EngineBody_natural_0 device* bodies_8 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_8 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_8 [[buffer(4)]], atomic<uint> device* counts_8 [[buffer(10)]], uint device* sort_8 [[buffer(8)]], uint device* scan_blocks_8 [[buffer(12)]], Record_natural_0 device* manifolds_8 [[buffer(9)]], uint device* gate_9 [[buffer(13)]])
{
    uint b_3;
    uint m_0;
    thread KernelContext_0 kernelContext_11;
    (&kernelContext_11)->params_1 = params_9;
    (&kernelContext_11)->row_counts_0 = row_counts_8;
    (&kernelContext_11)->counters_0 = counters_8;
    (&kernelContext_11)->records_0 = records_8;
    (&kernelContext_11)->contacts_0 = contacts_8;
    (&kernelContext_11)->colliders_0 = colliders_8;
    (&kernelContext_11)->bodies_0 = bodies_8;
    (&kernelContext_11)->collider_info_0 = collider_info_8;
    (&kernelContext_11)->start_bodies_0 = start_bodies_8;
    (&kernelContext_11)->counts_0 = counts_8;
    (&kernelContext_11)->sort_0 = sort_8;
    (&kernelContext_11)->scan_blocks_0 = scan_blocks_8;
    (&kernelContext_11)->manifolds_0 = manifolds_8;
    (&kernelContext_11)->gate_1 = gate_9;
    threadgroup array<uint, int(1024)> g_scan_8;
    (&kernelContext_11)->g_scan_0 = &g_scan_8;
    uint k_2 = id_5.x;
    uint n_2 = params_9->chunk_count_0;
    if(k_2 >= (params_9->chunk_count_0))
    {
        return;
    }
    uint device* _S60 = (&kernelContext_11)->sort_0+k_2;
    uint begin_0 = *_S60;
    uint _S61 = *((&kernelContext_11)->sort_0+(k_2 + 1U));
    uint _S62 = 2U * n_2 + 1U;
    uint a_2 = *_S60 + 1U;
    for(;;)
    {
        if(a_2 < _S61)
        {
        }
        else
        {
            break;
        }
        uint v_4 = *((&kernelContext_11)->sort_0+(_S62 + a_2));
        b_3 = a_2;
        for(;;)
        {
            bool _S63;
            if(b_3 > begin_0)
            {
                bool _S64 = record_before_0(v_4, *((&kernelContext_11)->sort_0+(_S62 + b_3 - 1U)), &kernelContext_11);
                _S63 = _S64;
            }
            else
            {
                _S63 = false;
            }
            if(_S63)
            {
            }
            else
            {
                break;
            }
            uint _S65 = _S62 + b_3;
            *((&kernelContext_11)->sort_0+_S65) = *((&kernelContext_11)->sort_0+(_S65 - 1U));
            b_3 = b_3 - 1U;
        }
        *((&kernelContext_11)->sort_0+(_S62 + b_3)) = v_4;
        a_2 = a_2 + 1U;
    }
    a_2 = begin_0;
    b_3 = 0U;
    for(;;)
    {
        if(a_2 < _S61)
        {
        }
        else
        {
            break;
        }
        Record_natural_0 r_1 = *((&kernelContext_11)->records_0+*((&kernelContext_11)->sort_0+(_S62 + a_2)));
        uint q_2 = 0U;
        for(;;)
        {
            if(q_2 < b_3)
            {
            }
            else
            {
                m_0 = 4294967295U;
                break;
            }
            if(((uint4(((&kernelContext_11)->manifolds_0+(begin_0 + q_2))->ids_0) ).y) == ((uint4(r_1.ids_0) ).y))
            {
                m_0 = q_2;
                break;
            }
            q_2 = q_2 + 1U;
        }
        if(m_0 == 4294967295U)
        {
            *((&kernelContext_11)->manifolds_0+(begin_0 + b_3)) = r_1;
            b_3 = b_3 + 1U;
            a_2 = a_2 + 1U;
            continue;
        }
        uint _S66 = begin_0 + m_0;
        Record_natural_0 device* _S67 = (&kernelContext_11)->manifolds_0+_S66;
        float4 _S68 = float4((*_S67).point_0) ;
        float4 _S69 = float4((*_S67).impulse_0) ;
        float4 _S70 = float4((*_S67).other_0) ;
        uint4 _S71 = uint4((*_S67).key_0) ;
        thread Record_0 acc_0;
        (&acc_0)->ids_0 = uint4((*_S67).ids_0) ;
        (&acc_0)->point_0 = _S68;
        (&acc_0)->impulse_0 = _S69;
        (&acc_0)->other_0 = _S70;
        (&acc_0)->key_0 = _S71;
        float w0_0 = (&acc_0)->point_0.w;
        float4 _S72 = float4(r_1.point_0) ;
        float w_0 = _S72.w;
        float total_0 = w0_0 + w_0;
        if(total_0 > 0.0f)
        {
            (&acc_0)->point_0.xyz = ((&acc_0)->point_0.xyz * float3(w0_0)  + _S72.xyz * float3(w_0) ) / float3(total_0) ;
            (&acc_0)->impulse_0.w = ((&acc_0)->impulse_0.w * w0_0 + (float4(r_1.impulse_0) ).w * w_0) / total_0;
        }
        (&acc_0)->impulse_0.xyz = (&acc_0)->impulse_0.xyz + (float4(r_1.impulse_0) ).xyz;
        (&acc_0)->point_0.w = total_0;
        Record_natural_0 device* _S73 = (&kernelContext_11)->manifolds_0+_S66;
        _S73->ids_0 = packed_uint4(acc_0.ids_0) ;
        _S73->point_0 = packed_float4(acc_0.point_0) ;
        _S73->impulse_0 = packed_float4(acc_0.impulse_0) ;
        _S73->other_0 = packed_float4(acc_0.other_0) ;
        _S73->key_0 = packed_uint4(acc_0.key_0) ;
        a_2 = a_2 + 1U;
    }
    *((&kernelContext_11)->sort_0+(n_2 + 1U + k_2)) = b_3;
    return;
}

[[kernel]] void ingest_any(uint3 id_6 [[thread_position_in_grid]], IngestParams_0 constant* params_10 [[buffer(0)]], uint device* row_counts_9 [[buffer(11)]], uint device* counters_9 [[buffer(5)]], Record_natural_0 device* records_9 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_9 [[buffer(1)]], EngineCollider_natural_0 device* colliders_9 [[buffer(2)]], EngineBody_natural_0 device* bodies_9 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_9 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_9 [[buffer(4)]], atomic<uint> device* counts_9 [[buffer(10)]], uint device* sort_9 [[buffer(8)]], uint device* scan_blocks_9 [[buffer(12)]], Record_natural_0 device* manifolds_9 [[buffer(9)]], uint device* gate_10 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_12;
    (&kernelContext_12)->params_1 = params_10;
    (&kernelContext_12)->row_counts_0 = row_counts_9;
    (&kernelContext_12)->counters_0 = counters_9;
    (&kernelContext_12)->records_0 = records_9;
    (&kernelContext_12)->contacts_0 = contacts_9;
    (&kernelContext_12)->colliders_0 = colliders_9;
    (&kernelContext_12)->bodies_0 = bodies_9;
    (&kernelContext_12)->collider_info_0 = collider_info_9;
    (&kernelContext_12)->start_bodies_0 = start_bodies_9;
    (&kernelContext_12)->counts_0 = counts_9;
    (&kernelContext_12)->sort_0 = sort_9;
    (&kernelContext_12)->scan_blocks_0 = scan_blocks_9;
    (&kernelContext_12)->manifolds_0 = manifolds_9;
    (&kernelContext_12)->gate_1 = gate_10;
    threadgroup array<uint, int(1024)> g_scan_9;
    (&kernelContext_12)->g_scan_0 = &g_scan_9;
    uint _S74 = id_6.x;
    bool _S75;
    if(_S74 >= (params_10->record_count_0))
    {
        _S75 = true;
    }
    else
    {
        bool _S76 = record_live_0(_S74, &kernelContext_12);
        _S75 = !_S76;
    }
    if(_S75)
    {
        return;
    }
    if(((uint4(((&kernelContext_12)->records_0+_S74)->ids_0) ).w) != 0U)
    {
        atomic_store_explicit((&kernelContext_12)->counts_0+2U * (&kernelContext_12)->params_1->chunk_count_0, 1U, memory_order_relaxed);
    }
    return;
}

void gate_size_0(uint slot_1, uint groups_0, KernelContext_0 thread* kernelContext_13)
{
    uint at_1 = 3U * (kernelContext_13->params_1->gate_0 + slot_1);
    *(kernelContext_13->gate_1+at_1) = groups_0;
    *(kernelContext_13->gate_1+(at_1 + 1U)) = 1U;
    *(kernelContext_13->gate_1+(at_1 + 2U)) = 1U;
    return;
}

[[kernel]] void ingest_gate(uint3 id_7 [[thread_position_in_grid]], IngestParams_0 constant* params_11 [[buffer(0)]], uint device* row_counts_10 [[buffer(11)]], uint device* counters_10 [[buffer(5)]], Record_natural_0 device* records_10 [[buffer(7)]], EngineContactRow_natural_0 device* contacts_10 [[buffer(1)]], EngineCollider_natural_0 device* colliders_10 [[buffer(2)]], EngineBody_natural_0 device* bodies_10 [[buffer(3)]], ColliderInfo_natural_0 device* collider_info_10 [[buffer(6)]], EngineBody_natural_0 device* start_bodies_10 [[buffer(4)]], atomic<uint> device* counts_10 [[buffer(10)]], uint device* sort_10 [[buffer(8)]], uint device* scan_blocks_10 [[buffer(12)]], Record_natural_0 device* manifolds_10 [[buffer(9)]], uint device* gate_11 [[buffer(13)]])
{
    thread KernelContext_0 kernelContext_14;
    (&kernelContext_14)->params_1 = params_11;
    (&kernelContext_14)->row_counts_0 = row_counts_10;
    (&kernelContext_14)->counters_0 = counters_10;
    (&kernelContext_14)->records_0 = records_10;
    (&kernelContext_14)->contacts_0 = contacts_10;
    (&kernelContext_14)->colliders_0 = colliders_10;
    (&kernelContext_14)->bodies_0 = bodies_10;
    (&kernelContext_14)->collider_info_0 = collider_info_10;
    (&kernelContext_14)->start_bodies_0 = start_bodies_10;
    (&kernelContext_14)->counts_0 = counts_10;
    (&kernelContext_14)->sort_0 = sort_10;
    (&kernelContext_14)->scan_blocks_0 = scan_blocks_10;
    (&kernelContext_14)->manifolds_0 = manifolds_10;
    (&kernelContext_14)->gate_1 = gate_11;
    threadgroup array<uint, int(1024)> g_scan_10;
    (&kernelContext_14)->g_scan_0 = &g_scan_10;
    uint n_3 = params_11->chunk_count_0;
    uint _S77 = 2U * params_11->chunk_count_0;
    uint _S78 = atomic_load_explicit(counts_10+_S77, memory_order_relaxed);
    bool any_0 = _S78 != 0U;
    atomic_store_explicit((&kernelContext_14)->counts_0+_S77, 0U, memory_order_relaxed);
    uint state_0 = 3U * ((&kernelContext_14)->params_1->gate_0 + 7U);
    uint _S79 = state_0 + 1U;
    uint streak_0 = *((&kernelContext_14)->gate_1+_S79);
    bool zero_0;
    if((*((&kernelContext_14)->gate_1+state_0)) != 0U)
    {
        zero_0 = !any_0;
    }
    else
    {
        zero_0 = false;
    }
    bool skip_0;
    if(zero_0)
    {
        skip_0 = streak_0 >= 2U;
    }
    else
    {
        skip_0 = false;
    }
    uint device* _S80 = (&kernelContext_14)->gate_1+_S79;
    uint _S81;
    if(zero_0)
    {
        _S81 = min(streak_0 + 1U, 2U);
    }
    else
    {
        _S81 = 0U;
    }
    *_S80 = _S81;
    uint device* _S82 = (&kernelContext_14)->gate_1+(state_0 + 2U);
    int _S83;
    if(skip_0)
    {
        _S83 = int(1);
    }
    else
    {
        _S83 = int(0);
    }
    *_S82 = uint(_S83);
    if(skip_0)
    {
        *((&kernelContext_14)->gate_1+3U * (&kernelContext_14)->params_1->gate_0) = 0U;
        *((&kernelContext_14)->gate_1+3U * ((&kernelContext_14)->params_1->gate_0 + 1U)) = 0U;
    }
    if(skip_0)
    {
        _S81 = 0U;
    }
    else
    {
        _S81 = (_S77 + 63U) / 64U;
    }
    gate_size_0(2U, _S81, &kernelContext_14);
    if(skip_0)
    {
        _S81 = 0U;
    }
    else
    {
        _S81 = ((&kernelContext_14)->params_1->record_count_0 + 63U) / 64U;
    }
    gate_size_0(3U, _S81, &kernelContext_14);
    if(skip_0)
    {
        _S81 = 0U;
    }
    else
    {
        _S81 = (n_3 + 1023U) / 1024U;
    }
    gate_size_0(4U, _S81, &kernelContext_14);
    if(skip_0)
    {
        _S83 = int(0);
    }
    else
    {
        _S83 = int(1);
    }
    gate_size_0(5U, uint(_S83), &kernelContext_14);
    if(skip_0)
    {
        _S81 = 0U;
    }
    else
    {
        _S81 = (n_3 + 63U) / 64U;
    }
    gate_size_0(6U, _S81, &kernelContext_14);
    return;
}

