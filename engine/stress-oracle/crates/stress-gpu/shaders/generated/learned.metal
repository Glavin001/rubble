#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

template<typename T, int A, int B>
matrix<T,A,B> _slang_matrixCompMult(matrix<T,A,B> m1, matrix<T,A,B> m2)
{
    matrix<T,A,B> result;
    for (int i = 0; i < A; i++)
        result[i] = m1[i] * m2[i];
    return result;
}

uint4 group_uniform4_0(uint4 value_0, uint thread_0)
{
    return value_0;
}

struct DecodeParams_0
{
    uint call_count_0;
    uint pad0_0;
    float log_tau_0;
    float tolerance_0;
};

struct KernelContext_0
{
    DecodeParams_0 constant* params_0;
    packed_float4 device* calls_0;
    packed_float4 device* nodes_0;
    packed_float4 device* bonds_0;
    array<array<float4, int(256)>, int(4)> threadgroup* g_red_0;
    array<float, int(6)> threadgroup* g_lambda_0;
};

float4 node_0(uint n_0, uint k_0, KernelContext_0 thread* kernelContext_0)
{
    return float4(*(kernelContext_0->nodes_0+(12U * n_0 + k_0))) ;
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

float3 rotate_0(const Quat_0 thread* q_1, float3 v_0)
{
    float3 qv_0 = float3(q_1->x_0, q_1->y_0, q_1->z_0);
    float3 t_0 = cross(qv_0, v_0) * float3(2.0f) ;
    return v_0 + t_0 * float3(q_1->w_0)  + cross(qv_0, t_0);
}

void reduce_store_0(uint t_1, float4 a_0, float4 b_0, float4 c_0, float4 d_0, KernelContext_0 thread* kernelContext_1)
{
    (*kernelContext_1->g_red_0)[int(0)][t_1] = a_0;
    (*kernelContext_1->g_red_0)[int(1)][t_1] = b_0;
    (*kernelContext_1->g_red_0)[int(2)][t_1] = c_0;
    (*kernelContext_1->g_red_0)[int(3)][t_1] = d_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = 128U;
    for(;;)
    {
        if(s_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_0)
        {
            uint _S1 = t_1 + s_0;
            (*kernelContext_1->g_red_0)[int(0)][t_1] = (*kernelContext_1->g_red_0)[int(0)][t_1] + (*kernelContext_1->g_red_0)[int(0)][_S1];
            (*kernelContext_1->g_red_0)[int(1)][t_1] = (*kernelContext_1->g_red_0)[int(1)][t_1] + (*kernelContext_1->g_red_0)[int(1)][_S1];
            (*kernelContext_1->g_red_0)[int(2)][t_1] = (*kernelContext_1->g_red_0)[int(2)][t_1] + (*kernelContext_1->g_red_0)[int(2)][_S1];
            (*kernelContext_1->g_red_0)[int(3)][t_1] = (*kernelContext_1->g_red_0)[int(3)][t_1] + (*kernelContext_1->g_red_0)[int(3)][_S1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    return;
}

void reduce_read_0(float4 thread* a_1, float4 thread* b_1, float4 thread* c_1, float4 thread* d_1, KernelContext_0 thread* kernelContext_2)
{
    *a_1 = (*kernelContext_2->g_red_0)[int(0)][int(0)];
    *b_1 = (*kernelContext_2->g_red_0)[int(1)][int(0)];
    *c_1 = (*kernelContext_2->g_red_0)[int(2)][int(0)];
    *d_1 = (*kernelContext_2->g_red_0)[int(3)][int(0)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return;
}

matrix<float,int(3),int(3)>  quat_matrix_0(float4 q_2)
{
    float w_1 = q_2.w;
    float x_1 = q_2.x;
    float y_1 = q_2.y;
    float z_1 = q_2.z;
    float _S2 = y_1 * y_1;
    float _S3 = z_1 * z_1;
    float _S4 = x_1 * y_1;
    float _S5 = w_1 * z_1;
    float _S6 = x_1 * z_1;
    float _S7 = w_1 * y_1;
    float _S8 = x_1 * x_1;
    float _S9 = y_1 * z_1;
    float _S10 = w_1 * x_1;
    return matrix<float,int(3),int(3)> (float3(1.0f - 2.0f * (_S2 + _S3), 2.0f * (_S4 - _S5), 2.0f * (_S6 + _S7)), float3(2.0f * (_S4 + _S5), 1.0f - 2.0f * (_S8 + _S3), 2.0f * (_S9 - _S10)), float3(2.0f * (_S6 - _S7), 2.0f * (_S9 + _S10), 1.0f - 2.0f * (_S8 + _S2)));
}

matrix<float,int(3),int(3)>  diag3_0(float3 d_2)
{
    return matrix<float,int(3),int(3)> (float3(d_2.x, 0.0f, 0.0f), float3(0.0f, d_2.y, 0.0f), float3(0.0f, 0.0f, d_2.z));
}

matrix<float,int(3),int(3)>  world_inertia_0(float4 q_3, float3 principal_0)
{
    matrix<float,int(3),int(3)>  r_1 = quat_matrix_0(q_3);
    return (((transpose(r_1)) * ((((diag3_0(principal_0)) * (r_1))))));
}

matrix<float,int(3),int(3)>  skew_0(float3 r_2)
{
    float _S11 = r_2.z;
    float _S12 = r_2.y;
    float _S13 = r_2.x;
    return matrix<float,int(3),int(3)> (float3(0.0f, - _S11, _S12), float3(_S11, 0.0f, - _S13), float3(- _S12, _S13, 0.0f));
}

matrix<float,int(3),int(3)>  rotvec_matrix_0(float3 r_3)
{
    float th_0 = length(r_3);
    matrix<float,int(3),int(3)>  eye_0 = matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
    if(th_0 < 9.99999997475242708e-07f)
    {
        return eye_0 + skew_0(r_3);
    }
    matrix<float,int(3),int(3)>  k_1 = skew_0(r_3 / float3(th_0) );
    return eye_0 + _slang_matrixCompMult(matrix<float,int(3),int(3)> (sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0), sin(th_0)), k_1) + _slang_matrixCompMult(matrix<float,int(3),int(3)> (1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0), 1.0f - cos(th_0)), (((k_1) * (k_1))));
}

matrix<float,int(3),int(3)>  outer3_0(float3 a_2, float3 b_2)
{
    return matrix<float,int(3),int(3)> (a_2 * float3(b_2.x) , a_2 * float3(b_2.y) , a_2 * float3(b_2.z) );
}

uint group_uniform_0(uint value_1, uint thread_1)
{
    return value_1;
}

float3 solve3_0(matrix<float,int(3),int(3)>  a_3, float3 b_3)
{
    float3 c0_0 = float3(a_3[int(0)][int(0)], a_3[int(1)][int(0)], a_3[int(2)][int(0)]);
    float3 c1_0 = float3(a_3[int(0)][int(1)], a_3[int(1)][int(1)], a_3[int(2)][int(1)]);
    float3 c2_0 = float3(a_3[int(0)][int(2)], a_3[int(1)][int(2)], a_3[int(2)][int(2)]);
    float3 _S14 = cross(c1_0, c2_0);
    return float3(dot(b_3, _S14), dot(c0_0, cross(b_3, c2_0)), dot(c0_0, cross(c1_0, b_3))) / float3(dot(c0_0, _S14)) ;
}

[[kernel]] void learned_decode(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], DecodeParams_0 constant* params_1 [[buffer(0)]], packed_float4 device* calls_1 [[buffer(1)]], packed_float4 device* nodes_1 [[buffer(2)]], packed_float4 device* bonds_1 [[buffer(3)]])
{
    float3 v_end_0;
    float3 w_end_0;
    uint q_4;
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_1;
    (&kernelContext_3)->calls_0 = calls_1;
    (&kernelContext_3)->nodes_0 = nodes_1;
    (&kernelContext_3)->bonds_0 = bonds_1;
    threadgroup array<array<float4, int(256)>, int(4)> g_red_1;
    (&kernelContext_3)->g_red_0 = &g_red_1;
    threadgroup array<float, int(6)> g_lambda_1;
    (&kernelContext_3)->g_lambda_0 = &g_lambda_1;
    uint g_0 = group_0.x;
    uint t_2 = thread_2.x;
    if(g_0 >= (params_1->call_count_0))
    {
        return;
    }
    uint _S15 = 8U * g_0;
    float4 _S16 = float4(*((&kernelContext_3)->calls_0+_S15)) ;
    float4 _S17 = float4(*((&kernelContext_3)->calls_0+(_S15 + 2U))) ;
    uint _S18 = _S15 + 4U;
    float4 _S19 = float4(*((&kernelContext_3)->calls_0+_S18)) ;
    float _S20 = _S16.x;
    float _S21 = _S16.z;
    float dt_0 = (float4(*((&kernelContext_3)->calls_0+(_S15 + 1U))) ).w;
    float e_in_0 = _S17.x;
    float _S22 = _S19.y;
    uint4 u_0 = group_uniform4_0(uint4((as_type<uint>((_S17.y))), (as_type<uint>((_S17.z))), (as_type<uint>((_S17.w))), (as_type<uint>((_S22)))), t_2);
    uint flags_0 = u_0.x;
    uint _S23 = u_0.y;
    uint _S24 = u_0.z;
    uint _S25 = u_0.w;
    float _S26 = _S19.x;
    uint _S27 = (as_type<uint>((_S26)));
    float3 grav_0 = (float4(*((&kernelContext_3)->calls_0+(_S15 + 3U))) ).xyz;
    bool free_0 = (flags_0 & 1U) != 0U;
    uint j_0 = t_2;
    for(;;)
    {
        if(j_0 < _S24)
        {
        }
        else
        {
            break;
        }
        uint k_2 = _S23 + j_0;
        float4 _S28 = node_0(k_2, 1U, &kernelContext_3);
        float4 _S29 = node_0(k_2, 3U, &kernelContext_3);
        float3 v_1 = _S29.xyz;
        float4 _S30 = node_0(k_2, 4U, &kernelContext_3);
        float3 w_2 = _S30.xyz;
        float4 _S31 = node_0(k_2, 5U, &kernelContext_3);
        float4 _S32 = node_0(k_2, 6U, &kernelContext_3);
        float4 _S33 = node_0(k_2, 7U, &kernelContext_3);
        Quat_0 qq_0 = quat_of_0(_S28);
        float3 _S34 = _S31.xyz;
        thread Quat_0 _S35 = qq_0;
        float3 _S36 = rotate_0(&_S35, _S34);
        float3 _S37 = float3(_S21) ;
        float3 _S38 = v_1 + _S36 * _S37;
        float3 _S39 = float3(_S31.w, _S32.x, _S32.y);
        thread Quat_0 _S40 = qq_0;
        float3 _S41 = rotate_0(&_S40, _S39);
        float3 _S42 = float3(_S20) ;
        float3 _S43 = w_2 + _S41 * _S37 / _S42;
        float3 _S44 = float3(dt_0) ;
        float3 _S45 = v_1 * _S44 + float3(0.5f)  * grav_0 * _S44 * _S44;
        float3 _S46 = float3(_S32.z, _S32.w, _S33.x);
        thread Quat_0 _S47 = qq_0;
        float3 _S48 = rotate_0(&_S47, _S46);
        float3 _S49 = _S45 + _S48 * _S37 * _S44;
        float3 _S50 = w_2 * _S44;
        float3 _S51 = _S33.yzw;
        thread Quat_0 _S52 = qq_0;
        float3 _S53 = rotate_0(&_S52, _S51);
        float3 _S54 = _S50 + _S53 * _S37 * _S44 / _S42;
        float4 _S55 = node_0(k_2, 2U, &kernelContext_3);
        float3 dx_0;
        float3 dr_0;
        if((_S55.w) != 0.0f)
        {
            float3 _S56 = float3(0.0f) ;
            v_end_0 = _S56;
            w_end_0 = _S56;
            dx_0 = _S56;
            dr_0 = _S56;
        }
        else
        {
            v_end_0 = _S38;
            w_end_0 = _S43;
            dx_0 = _S49;
            dr_0 = _S54;
        }
        uint _S57 = 12U * k_2;
        *((&kernelContext_3)->nodes_0+(_S57 + 8U)) = packed_float4(float4(v_end_0, 0.0f)) ;
        *((&kernelContext_3)->nodes_0+(_S57 + 9U)) = packed_float4(float4(w_end_0, 0.0f)) ;
        *((&kernelContext_3)->nodes_0+(_S57 + 10U)) = packed_float4(float4(dx_0, 0.0f)) ;
        *((&kernelContext_3)->nodes_0+(_S57 + 11U)) = packed_float4(float4(dr_0, 0.0f)) ;
        j_0 = j_0 + 256U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    thread float4 a_4;
    thread float4 b_4;
    thread float4 c_2;
    thread float4 d_3;
    float4 _S58 = float4(0.0f) ;
    j_0 = t_2;
    float4 sa_0 = _S58;
    float4 sb_0 = _S58;
    float4 sc_0 = _S58;
    for(;;)
    {
        if(j_0 < _S24)
        {
        }
        else
        {
            break;
        }
        uint k_3 = _S23 + j_0;
        float4 _S59 = node_0(k_3, 0U, &kernelContext_3);
        float m_0 = _S59.w;
        float3 _S60 = float3(m_0) ;
        float4 sa_1 = sa_0 + float4(_S59.xyz * _S60, m_0);
        float4 _S61 = node_0(k_3, 3U, &kernelContext_3);
        float4 sb_1 = sb_0 + float4(_S61.xyz * _S60, 0.0f);
        float4 _S62 = node_0(k_3, 10U, &kernelContext_3);
        float4 sc_1 = sc_0 + float4(_S62.xyz * _S60, 0.0f);
        j_0 = j_0 + 256U;
        sa_0 = sa_1;
        sb_0 = sb_1;
        sc_0 = sc_1;
    }
    reduce_store_0(t_2, sa_0, sb_0, sc_0, _S58, &kernelContext_3);
    reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
    float mt_0 = a_4.w;
    float3 _S63 = float3(mt_0) ;
    float3 _S64 = a_4.xyz / _S63;
    float3 p0_0 = b_4.xyz;
    float3 have_0 = c_2.xyz;
    float3 _S65 = float3(0.0f) ;
    if(free_0)
    {
        float3 _S66 = float3(dt_0) ;
        v_end_0 = (p0_0 * _S66 + float3((0.5f * mt_0))  * grav_0 * _S66 * _S66 - have_0) / _S63;
    }
    else
    {
        v_end_0 = _S65;
    }
    j_0 = t_2;
    sa_0 = _S58;
    sb_0 = _S58;
    for(;;)
    {
        if(j_0 < _S24)
        {
        }
        else
        {
            break;
        }
        uint k_4 = _S23 + j_0;
        float4 _S67 = node_0(k_4, 0U, &kernelContext_3);
        float m_1 = _S67.w;
        float4 _S68 = node_0(k_4, 10U, &kernelContext_3);
        float3 dx_1 = _S68.xyz;
        if(free_0)
        {
            w_end_0 = dx_1 + v_end_0;
        }
        else
        {
            w_end_0 = dx_1;
        }
        float3 _S69 = _S67.xyz;
        float3 _S70 = float3(m_1) ;
        float4 sa_2 = sa_0 + float4((_S69 + w_end_0) * _S70, 0.0f);
        float3 r0_0 = _S69 - _S64;
        float4 _S71 = node_0(k_4, 1U, &kernelContext_3);
        float4 _S72 = node_0(k_4, 2U, &kernelContext_3);
        matrix<float,int(3),int(3)>  i0_0 = world_inertia_0(_S71, _S72.xyz);
        float4 _S73 = node_0(k_4, 3U, &kernelContext_3);
        float3 _S74 = cross(r0_0, _S70 * _S73.xyz);
        float4 _S75 = node_0(k_4, 4U, &kernelContext_3);
        float4 sb_2 = sb_0 + float4(_S74 + (((_S75.xyz) * (i0_0))), 0.0f);
        j_0 = j_0 + 256U;
        sa_0 = sa_2;
        sb_0 = sb_2;
    }
    reduce_store_0(t_2, sa_0, sb_0, _S58, _S58, &kernelContext_3);
    reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
    float3 _S76 = a_4.xyz / _S63;
    float3 l0_0 = b_4.xyz;
    uint dec_0;
    uint bad_0;
    float4 ta_0;
    float4 tb_0;
    float4 tc_0;
    if(free_0)
    {
        j_0 = t_2;
        for(;;)
        {
            if(j_0 < _S24)
            {
            }
            else
            {
                break;
            }
            uint k_5 = _S23 + j_0;
            packed_float4 device* _S77 = (&kernelContext_3)->nodes_0+(12U * k_5 + 10U);
            float4 _S78 = node_0(k_5, 10U, &kernelContext_3);
            *_S77 = packed_float4(float4(_S78.xyz + v_end_0, 0.0f)) ;
            j_0 = j_0 + 256U;
        }
        j_0 = t_2;
        sa_0 = _S58;
        sb_0 = _S58;
        sc_0 = _S58;
        ta_0 = _S58;
        tb_0 = _S58;
        tc_0 = _S58;
        for(;;)
        {
            if(j_0 < _S24)
            {
            }
            else
            {
                break;
            }
            uint k_6 = _S23 + j_0;
            float4 _S79 = node_0(k_6, 0U, &kernelContext_3);
            float m_2 = _S79.w;
            float3 _S80 = _S79.xyz;
            float4 _S81 = node_0(k_6, 10U, &kernelContext_3);
            float3 r_4 = _S80 + _S81.xyz - _S76;
            float4 _S82 = node_0(k_6, 11U, &kernelContext_3);
            matrix<float,int(3),int(3)>  rd_0 = rotvec_matrix_0(_S82.xyz);
            float4 _S83 = node_0(k_6, 1U, &kernelContext_3);
            float4 _S84 = node_0(k_6, 2U, &kernelContext_3);
            matrix<float,int(3),int(3)>  i1_0 = (((transpose(rd_0)) * ((((world_inertia_0(_S83, _S84.xyz)) * (rd_0))))));
            float4 _S85 = node_0(k_6, 8U, &kernelContext_3);
            float3 v_2 = _S85.xyz;
            float4 _S86 = node_0(k_6, 9U, &kernelContext_3);
            float3 _S87 = float3(m_2) ;
            float3 _S88 = _S87 * v_2;
            float4 sa_3 = sa_0 + float4(_S88, 0.0f);
            float4 sb_3 = sb_0 + float4(cross(r_4, _S88) + (((_S86.xyz) * (i1_0))), 0.0f);
            float4 sc_2 = sc_0 + float4(_S87 * r_4, 0.0f);
            matrix<float,int(3),int(3)>  jj_0 = _slang_matrixCompMult(matrix<float,int(3),int(3)> (m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2, m_2), _slang_matrixCompMult(matrix<float,int(3),int(3)> (dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4), dot(r_4, r_4)), matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f))) - outer3_0(r_4, r_4)) + i1_0;
            float4 sd_0 = ta_0 + float4(jj_0[int(0)], 0.0f);
            float4 se_0 = tb_0 + float4(jj_0[int(1)], 0.0f);
            float4 sf_0 = tc_0 + float4(jj_0[int(2)], 0.0f);
            j_0 = j_0 + 256U;
            sa_0 = sa_3;
            sb_0 = sb_3;
            sc_0 = sc_2;
            ta_0 = sd_0;
            tb_0 = se_0;
            tc_0 = sf_0;
        }
        reduce_store_0(t_2, sa_0, sb_0, sc_0, ta_0, &kernelContext_3);
        thread float4 ax_p_0;
        thread float4 ax_l_0;
        thread float4 smr_0;
        thread float4 j0_0;
        reduce_read_0(&ax_p_0, &ax_l_0, &smr_0, &j0_0, &kernelContext_3);
        reduce_store_0(t_2, tb_0, tc_0, _S58, _S58, &kernelContext_3);
        thread float4 j1_0;
        thread float4 j2_0;
        thread float4 unused0_0;
        thread float4 unused1_0;
        reduce_read_0(&j1_0, &j2_0, &unused0_0, &unused1_0, &kernelContext_3);
        if(t_2 == 0U)
        {
            float3 _S89 = p0_0 + _S63 * grav_0 * float3(dt_0)  - ax_p_0.xyz;
            float3 _S90 = l0_0 - ax_l_0.xyz;
            matrix<float,int(3),int(3)>  _S91 = skew_0(smr_0.xyz);
            thread array<array<float, int(7)>, int(6)> m6_0;
            dec_0 = 0U;
            for(;;)
            {
                if(dec_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                q_4 = 0U;
                for(;;)
                {
                    if(q_4 < 7U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    m6_0[dec_0][q_4] = 0.0f;
                    q_4 = q_4 + 1U;
                }
                dec_0 = dec_0 + 1U;
            }
            array<float3, int(3)> _S92 = { { j0_0.xyz, j1_0.xyz, j2_0.xyz } };
            dec_0 = 0U;
            for(;;)
            {
                if(dec_0 < 3U)
                {
                }
                else
                {
                    break;
                }
                m6_0[dec_0][dec_0] = mt_0;
                q_4 = 0U;
                for(;;)
                {
                    if(q_4 < 3U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint _S93 = 3U + q_4;
                    m6_0[dec_0][_S93] = - _S91[dec_0][q_4];
                    m6_0[3U + dec_0][q_4] = _S91[dec_0][q_4];
                    m6_0[3U + dec_0][_S93] = _S92[dec_0][q_4];
                    q_4 = q_4 + 1U;
                }
                m6_0[dec_0][int(6)] = _S89[dec_0];
                m6_0[3U + dec_0][int(6)] = _S90[dec_0];
                dec_0 = dec_0 + 1U;
            }
            bad_0 = 0U;
            for(;;)
            {
                if(bad_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                uint _S94 = bad_0 + 1U;
                q_4 = bad_0;
                dec_0 = _S94;
                for(;;)
                {
                    if(dec_0 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    if((abs(m6_0[dec_0][bad_0])) > (abs(m6_0[q_4][bad_0])))
                    {
                        q_4 = dec_0;
                    }
                    dec_0 = dec_0 + 1U;
                }
                uint q_5 = 0U;
                for(;;)
                {
                    if(q_5 < 7U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float tmp_0 = m6_0[bad_0][q_5];
                    m6_0[bad_0][q_5] = m6_0[q_4][q_5];
                    m6_0[q_4][q_5] = tmp_0;
                    q_5 = q_5 + 1U;
                }
                uint r_5 = _S94;
                for(;;)
                {
                    if(r_5 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S95 = m6_0[r_5][bad_0] / m6_0[bad_0][bad_0];
                    uint q_6 = bad_0;
                    for(;;)
                    {
                        if(q_6 < 7U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        m6_0[r_5][q_6] = m6_0[r_5][q_6] - _S95 * m6_0[bad_0][q_6];
                        q_6 = q_6 + 1U;
                    }
                    r_5 = r_5 + 1U;
                }
                bad_0 = _S94;
            }
            thread array<float, int(6)> x6_0;
            x6_0[int(0)] = 0.0f;
            x6_0[int(1)] = 0.0f;
            x6_0[int(2)] = 0.0f;
            x6_0[int(3)] = 0.0f;
            x6_0[int(4)] = 0.0f;
            x6_0[int(5)] = 0.0f;
            int r_6 = int(5);
            for(;;)
            {
                if(r_6 >= int(0))
                {
                }
                else
                {
                    break;
                }
                q_4 = uint(r_6) + 1U;
                float acc_0 = m6_0[r_6][int(6)];
                for(;;)
                {
                    if(q_4 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float acc_1 = acc_0 - m6_0[r_6][q_4] * x6_0[q_4];
                    q_4 = q_4 + 1U;
                    acc_0 = acc_1;
                }
                x6_0[r_6] = acc_0 / m6_0[r_6][r_6];
                r_6 = r_6 - int(1);
            }
            dec_0 = 0U;
            for(;;)
            {
                if(dec_0 < 6U)
                {
                }
                else
                {
                    break;
                }
                (*(&kernelContext_3)->g_lambda_0)[dec_0] = x6_0[dec_0];
                dec_0 = dec_0 + 1U;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float3 _S96 = float3((*(&kernelContext_3)->g_lambda_0)[int(0)], (*(&kernelContext_3)->g_lambda_0)[int(1)], (*(&kernelContext_3)->g_lambda_0)[int(2)]);
        float3 _S97 = float3((*(&kernelContext_3)->g_lambda_0)[int(3)], (*(&kernelContext_3)->g_lambda_0)[int(4)], (*(&kernelContext_3)->g_lambda_0)[int(5)]);
        j_0 = t_2;
        for(;;)
        {
            if(j_0 < _S24)
            {
            }
            else
            {
                break;
            }
            uint k_7 = _S23 + j_0;
            float4 _S98 = node_0(k_7, 0U, &kernelContext_3);
            float3 _S99 = _S98.xyz;
            float4 _S100 = node_0(k_7, 10U, &kernelContext_3);
            float3 r_7 = _S99 + _S100.xyz - _S76;
            uint _S101 = 12U * k_7;
            packed_float4 device* _S102 = (&kernelContext_3)->nodes_0+(_S101 + 8U);
            float4 _S103 = node_0(k_7, 8U, &kernelContext_3);
            *_S102 = packed_float4(float4(_S103.xyz + _S96 + cross(_S97, r_7), 0.0f)) ;
            packed_float4 device* _S104 = (&kernelContext_3)->nodes_0+(_S101 + 9U);
            float4 _S105 = node_0(k_7, 9U, &kernelContext_3);
            *_S104 = packed_float4(float4(_S105.xyz + _S97, 0.0f)) ;
            j_0 = j_0 + 256U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    }
    if((flags_0 & 2U) == 0U)
    {
        j_0 = t_2;
        sa_0 = _S58;
        sb_0 = _S58;
        for(;;)
        {
            if(j_0 < _S24)
            {
            }
            else
            {
                break;
            }
            uint k_8 = _S23 + j_0;
            float4 _S106 = node_0(k_8, 0U, &kernelContext_3);
            float m_3 = _S106.w;
            float4 _S107 = node_0(k_8, 1U, &kernelContext_3);
            float4 _S108 = node_0(k_8, 2U, &kernelContext_3);
            matrix<float,int(3),int(3)>  i0_1 = world_inertia_0(_S107, _S108.xyz);
            float4 _S109 = node_0(k_8, 11U, &kernelContext_3);
            matrix<float,int(3),int(3)>  rd_1 = rotvec_matrix_0(_S109.xyz);
            matrix<float,int(3),int(3)>  i1_1 = (((transpose(rd_1)) * ((((i0_1) * (rd_1))))));
            float4 _S110 = node_0(k_8, 3U, &kernelContext_3);
            float3 v0_0 = _S110.xyz;
            float4 _S111 = node_0(k_8, 4U, &kernelContext_3);
            float3 w0_0 = _S111.xyz;
            float4 _S112 = node_0(k_8, 8U, &kernelContext_3);
            float3 v_3 = _S112.xyz;
            float4 _S113 = node_0(k_8, 9U, &kernelContext_3);
            float3 w_3 = _S113.xyz;
            float ke_in_0 = 0.5f * (m_3 * dot(v0_0, v0_0) + dot(w0_0, (((w0_0) * (i0_1)))));
            float4 _S114 = node_0(k_8, 10U, &kernelContext_3);
            float4 sa_4 = sa_0 + float4(ke_in_0, m_3 * dot(grav_0, _S114.xyz), 0.5f * (m_3 * dot(v_3, v_3) + dot(w_3, (((w_3) * (i1_1))))), 0.0f);
            float4 sb_4 = sb_0 + float4(float3(m_3)  * v_3, 0.0f);
            j_0 = j_0 + 256U;
            sa_0 = sa_4;
            sb_0 = sb_4;
        }
        reduce_store_0(t_2, sa_0, sb_0, _S58, _S58, &kernelContext_3);
        reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
        float bound_0 = a_4.x + a_4.y + (&kernelContext_3)->params_0->tolerance_0 * e_in_0;
        float ke_0 = a_4.z;
        if(ke_0 > bound_0)
        {
            j_0 = 1U;
        }
        else
        {
            j_0 = 0U;
        }
        if((group_uniform_0(j_0, t_2)) != 0U)
        {
            if(free_0)
            {
                float3 vc_0 = b_4.xyz / _S63;
                j_0 = t_2;
                ta_0 = _S58;
                tb_0 = _S58;
                tc_0 = _S58;
                float4 td_0 = _S58;
                for(;;)
                {
                    if(j_0 < _S24)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_9 = _S23 + j_0;
                    float4 _S115 = node_0(k_9, 0U, &kernelContext_3);
                    float m_4 = _S115.w;
                    float4 _S116 = node_0(k_9, 0U, &kernelContext_3);
                    float3 _S117 = _S116.xyz;
                    float4 _S118 = node_0(k_9, 10U, &kernelContext_3);
                    float3 r_8 = _S117 + _S118.xyz - _S76;
                    float4 _S119 = node_0(k_9, 11U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  rd_2 = rotvec_matrix_0(_S119.xyz);
                    float4 _S120 = node_0(k_9, 1U, &kernelContext_3);
                    float4 _S121 = node_0(k_9, 2U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  i1_2 = (((transpose(rd_2)) * ((((world_inertia_0(_S120, _S121.xyz)) * (rd_2))))));
                    matrix<float,int(3),int(3)>  jj_1 = _slang_matrixCompMult(matrix<float,int(3),int(3)> (m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4, m_4), _slang_matrixCompMult(matrix<float,int(3),int(3)> (dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8), dot(r_8, r_8)), matrix<float,int(3),int(3)> (float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f))) - outer3_0(r_8, r_8)) + i1_2;
                    float4 _S122 = node_0(k_9, 8U, &kernelContext_3);
                    float4 ta_1 = ta_0 + float4(jj_1[int(0)], 0.0f);
                    float4 tb_1 = tb_0 + float4(jj_1[int(1)], 0.0f);
                    float4 tc_1 = tc_0 + float4(jj_1[int(2)], 0.0f);
                    float3 _S123 = cross(r_8, float3(m_4)  * (_S122.xyz - vc_0));
                    float4 _S124 = node_0(k_9, 9U, &kernelContext_3);
                    float4 td_1 = td_0 + float4(_S123 + (((_S124.xyz) * (i1_2))), 0.0f);
                    j_0 = j_0 + 256U;
                    ta_0 = ta_1;
                    tb_0 = tb_1;
                    tc_0 = tc_1;
                    td_0 = td_1;
                }
                reduce_store_0(t_2, ta_0, tb_0, tc_0, td_0, &kernelContext_3);
                reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
                float3 lrel_0 = d_3.xyz;
                float3 omega_0 = solve3_0(matrix<float,int(3),int(3)> (a_4.xyz, b_4.xyz, c_2.xyz), lrel_0);
                float ke_base_0 = 0.5f * (mt_0 * dot(vc_0, vc_0) + dot(omega_0, lrel_0));
                thread float4 ra_0 = _S58;
                j_0 = t_2;
                for(;;)
                {
                    if(j_0 < _S24)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_10 = _S23 + j_0;
                    float4 _S125 = node_0(k_10, 0U, &kernelContext_3);
                    float m_5 = _S125.w;
                    float4 _S126 = node_0(k_10, 0U, &kernelContext_3);
                    float3 _S127 = _S126.xyz;
                    float4 _S128 = node_0(k_10, 10U, &kernelContext_3);
                    float3 r_9 = _S127 + _S128.xyz - _S76;
                    float4 _S129 = node_0(k_10, 11U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  rd_3 = rotvec_matrix_0(_S129.xyz);
                    float4 _S130 = node_0(k_10, 1U, &kernelContext_3);
                    float4 _S131 = node_0(k_10, 2U, &kernelContext_3);
                    matrix<float,int(3),int(3)>  i1_3 = (((transpose(rd_3)) * ((((world_inertia_0(_S130, _S131.xyz)) * (rd_3))))));
                    float4 _S132 = node_0(k_10, 8U, &kernelContext_3);
                    float3 uu_0 = _S132.xyz - vc_0 - cross(omega_0, r_9);
                    float4 _S133 = node_0(k_10, 9U, &kernelContext_3);
                    float3 wr_0 = _S133.xyz - omega_0;
                    ra_0.x = ra_0.x + 0.5f * (m_5 * dot(uu_0, uu_0) + dot(wr_0, (((wr_0) * (i1_3)))));
                    j_0 = j_0 + 256U;
                }
                reduce_store_0(t_2, ra_0, _S58, _S58, _S58, &kernelContext_3);
                reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
                float _S134 = sqrt(clamp((bound_0 - ke_base_0) / max(a_4.x, 1.00000000317107685e-30f), 0.0f, 1.0f));
                if(ke_base_0 > bound_0)
                {
                    dec_0 = 3U;
                }
                else
                {
                    dec_0 = 1U;
                }
                j_0 = t_2;
                for(;;)
                {
                    if(j_0 < _S24)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_11 = _S23 + j_0;
                    float4 _S135 = node_0(k_11, 0U, &kernelContext_3);
                    float3 _S136 = _S135.xyz;
                    float4 _S137 = node_0(k_11, 10U, &kernelContext_3);
                    float3 r_10 = _S136 + _S137.xyz - _S76;
                    float4 _S138 = node_0(k_11, 8U, &kernelContext_3);
                    float3 _S139 = cross(omega_0, r_10);
                    float3 uu_1 = _S138.xyz - vc_0 - _S139;
                    float4 _S140 = node_0(k_11, 9U, &kernelContext_3);
                    float3 wr_1 = _S140.xyz - omega_0;
                    uint _S141 = 12U * k_11;
                    float3 _S142 = float3(_S134) ;
                    *((&kernelContext_3)->nodes_0+(_S141 + 8U)) = packed_float4(float4(vc_0 + _S139 + _S142 * uu_1, 0.0f)) ;
                    *((&kernelContext_3)->nodes_0+(_S141 + 9U)) = packed_float4(float4(omega_0 + _S142 * wr_1, 0.0f)) ;
                    j_0 = j_0 + 256U;
                }
            }
            else
            {
                float _S143 = sqrt(min(max(bound_0, 0.0f) / max(ke_0, 1.00000000317107685e-30f), 1.0f));
                j_0 = t_2;
                for(;;)
                {
                    if(j_0 < _S24)
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint k_12 = _S23 + j_0;
                    uint _S144 = 12U * k_12;
                    packed_float4 device* _S145 = (&kernelContext_3)->nodes_0+(_S144 + 8U);
                    float4 _S146 = node_0(k_12, 8U, &kernelContext_3);
                    float3 _S147 = float3(_S143) ;
                    *_S145 = packed_float4(float4(_S146.xyz * _S147, 0.0f)) ;
                    packed_float4 device* _S148 = (&kernelContext_3)->nodes_0+(_S144 + 9U);
                    float4 _S149 = node_0(k_12, 9U, &kernelContext_3);
                    *_S148 = packed_float4(float4(_S149.xyz * _S147, 0.0f)) ;
                    j_0 = j_0 + 256U;
                }
                dec_0 = 1U;
            }
        }
        else
        {
            dec_0 = 0U;
        }
    }
    else
    {
        dec_0 = 0U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    j_0 = t_2;
    for(;;)
    {
        if(j_0 < _S25)
        {
        }
        else
        {
            break;
        }
        uint _S150 = 2U * (_S27 + j_0);
        float4 _S151 = float4(*((&kernelContext_3)->bonds_0+_S150)) ;
        uint _S152 = _S150 + 1U;
        float4 _S153 = float4(*((&kernelContext_3)->bonds_0+_S152)) ;
        float _S154 = min(_S151.x + _S151.z, 0.99999898672103882f);
        packed_float4 device* _S155 = (&kernelContext_3)->bonds_0+_S152;
        if((_S151.y) >= ((&kernelContext_3)->params_0->log_tau_0))
        {
            bad_0 = 1U;
        }
        else
        {
            bad_0 = 0U;
        }
        *_S155 = packed_float4(float4(_S154, (as_type<float>((bad_0))), _S153.z, _S153.w)) ;
        j_0 = j_0 + 256U;
    }
    bad_0 = 0U;
    j_0 = t_2;
    for(;;)
    {
        if(j_0 < _S24)
        {
        }
        else
        {
            break;
        }
        uint k_13 = _S23 + j_0;
        float4 _S156 = node_0(k_13, 8U, &kernelContext_3);
        float4 _S157 = float4(1.0f) ;
        float _S158 = dot(_S156, _S157);
        float4 _S159 = node_0(k_13, 9U, &kernelContext_3);
        float _S160 = _S158 + dot(_S159, _S157);
        float4 _S161 = node_0(k_13, 10U, &kernelContext_3);
        float _S162 = _S160 + dot(_S161, _S157);
        float4 _S163 = node_0(k_13, 11U, &kernelContext_3);
        if(!((abs(_S162 + dot(_S163, _S157))) < 3.00000000549775576e+38f))
        {
            bad_0 = 1U;
        }
        j_0 = j_0 + 256U;
    }
    reduce_store_0(t_2, float4(float(bad_0), 0.0f, 0.0f, 0.0f), _S58, _S58, _S58, &kernelContext_3);
    reduce_read_0(&a_4, &b_4, &c_2, &d_3, &kernelContext_3);
    if((a_4.x) > 0.0f)
    {
        dec_0 = dec_0 | 4U;
    }
    if(t_2 == 0U)
    {
        *((&kernelContext_3)->calls_0+_S18) = packed_float4(float4(_S26, _S22, (as_type<float>((dec_0))), _S19.w)) ;
    }
    return;
}

