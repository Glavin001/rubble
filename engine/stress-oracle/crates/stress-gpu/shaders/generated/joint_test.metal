#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
constant array<float, int(6)> SPRING_AT_0 = { { -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f } };
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

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S1;
    if((st_0->damage_0) < 1.0f)
    {
        _S1 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S1 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S1 = false;
        }
    }
    return _S1;
}

struct Measures_0
{
    float tension_0;
    float shear_0;
    float normal_compression_0;
    float compression_0;
    float compressive_force_0;
};

struct JointBond_natural_0
{
    packed_float4 geom0_0;
    packed_float4 geom1_0;
    packed_float4 stiff0_0;
    packed_float4 stiff1_0;
    packed_float4 rebar0_0;
    packed_float4 rebar1_0;
    packed_uint4 ids_0;
};

Measures_0 stress_measures_0(const JointBond_natural_0 thread* b_0, float3 q_lin_0, float3 q_ang_0)
{
    float4 _S2 = float4(b_0->geom0_0) ;
    float area_0 = _S2.x;
    float _S3 = q_lin_0.z;
    float axial_0 = (metal::fast::divide((_S3), (area_0)));
    float4 _S4 = float4(b_0->geom1_0) ;
    float _S5 = (metal::fast::divide((abs(q_ang_0.x)), (_S4.x)));
    float _S6 = (metal::fast::divide((abs(q_ang_0.y)), (_S4.y)));
    float bending_0 = _S5 + _S6;
    float _S7 = q_lin_0.x;
    float _S8 = q_lin_0.y;
    float _S9 = (metal::fast::sqrt((_S7 * _S7 + _S8 * _S8)));
    float _S10 = (metal::fast::divide((_S9), (area_0)));
    float _S11 = (metal::fast::divide((abs(q_ang_0.z)), (_S2.w)));
    float shear_1 = _S10 + _S11;
    thread Measures_0 m_0;
    (&m_0)->tension_0 = axial_0 + bending_0;
    (&m_0)->shear_0 = shear_1;
    float _S12 = - axial_0;
    (&m_0)->normal_compression_0 = max(_S12, 0.0f);
    (&m_0)->compression_0 = _S12 + bending_0;
    (&m_0)->compressive_force_0 = max(- _S3, 0.0f);
    return m_0;
}

float expm1_accurate_0(float x_0)
{
    if((abs(x_0)) < 0.00100000004749745f)
    {
        return x_0 * (1.0f + x_0 * (0.5f + x_0 * 0.1666666716337204f));
    }
    return exp(x_0) - 1.0f;
}

struct JointMaterial_natural_0
{
    packed_float4 strength_0;
    packed_float4 energy_0;
    packed_float4 dif_0;
    packed_float4 misc_0;
    packed_uint4 kind_flags_0;
};

float dif_factor_0(const JointMaterial_natural_0 thread* mat_0, float strain_rate_1)
{
    float r_0 = abs(strain_rate_1);
    float4 _S13 = float4(mat_0->dif_0) ;
    float ref_0 = _S13.x;
    if(r_0 <= ref_0)
    {
        return 1.0f;
    }
    float _S14 = _S13.z;
    float f_0;
    if(r_0 <= _S14)
    {
        float _S15 = (metal::fast::divide((r_0), (ref_0)));
        float _S16 = (metal::fast::pow((_S15), (_S13.y)));
        f_0 = _S16;
    }
    else
    {
        float _S17 = (metal::fast::divide((_S14), (ref_0)));
        float _S18 = (metal::fast::pow((_S17), (_S13.y)));
        float _S19 = (metal::fast::divide((r_0), (_S14)));
        float _S20 = (metal::fast::pow((_S19), (_S13.w)));
        f_0 = _S18 * _S20;
    }
    return clamp(f_0, 1.0f, (float4(mat_0->misc_0) ).x);
}

float fatigue_factor_0(const JointMaterial_natural_0 thread* mat_1, float life_1)
{
    if((((uint4(mat_1->kind_flags_0) ).y) & 64U) == 0U)
    {
        return 1.0f;
    }
    float _S21 = clamp(life_1, 0.0f, 1.0f);
    float _S22 = (metal::fast::divide((1.0f), ((float4(mat_1->misc_0) ).y - 2.0f)));
    float _S23 = (metal::fast::pow((_S21), (_S22)));
    return _S23;
}

float infinity_0()
{
    return (as_type<float>((2139095040U)));
}

float4 failure_indices_0(const JointMaterial_natural_0 thread* mat_2, const JointBond_natural_0 thread* b_1, const Measures_0 thread* m_1, float multiplier_0)
{
    float4 _S24 = float4(mat_2->strength_0) ;
    float fc_0 = _S24.y * multiplier_0;
    float _S25 = min(_S24.z * multiplier_0 + _S24.w * m_1->normal_compression_0, (float4(mat_2->energy_0) ).x * multiplier_0);
    thread float4 idx_0;
    float _S26 = (metal::fast::divide((m_1->tension_0), (_S24.x * multiplier_0)));
    idx_0.x = max(_S26, 0.0f);
    float _S27;
    if(_S25 > 0.0f)
    {
        float _S28 = (metal::fast::divide((m_1->shear_0), (_S25)));
        _S27 = _S28;
    }
    else
    {
        _S27 = infinity_0();
    }
    idx_0.y = _S27;
    float _S29 = (metal::fast::divide((m_1->compression_0), (fc_0)));
    idx_0.z = max(_S29, 0.0f);
    float _S30 = (float4(b_1->stiff1_0) ).y;
    if(_S30 > 0.0f)
    {
        float _S31 = (metal::fast::divide((m_1->compressive_force_0), (_S30)));
        _S27 = _S31;
    }
    else
    {
        _S27 = 0.0f;
    }
    idx_0.w = _S27;
    return idx_0;
}

float sq_0(float x_1)
{
    return x_1 * x_1;
}

float damage_law_0(uint kind_0, float kappa_1, float r_1)
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_0 == 0U)
    {
        if(r_1 <= 1.0f)
        {
            return 1.0f;
        }
        float _S32 = (metal::fast::divide((r_1 * (kappa_1 - 1.0f)), (kappa_1 * (r_1 - 1.0f))));
        return min(_S32, 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_1 + 1.0f)))
    {
        return 1.0f;
    }
    float _S33 = (metal::fast::divide((1.0f), (kappa_1)));
    return 1.0f - _S33;
}

__attribute__((noinline))
float2 damage_increment_0(uint kind_1, float kappa_old_0, float lambda_0, float r_2, float d_old_0, float psi_0)
{
    float _S34 = damage_law_0(kind_1, lambda_0, r_2);
    float _S35 = max(_S34, d_old_0);
    bool _S36;
    if(_S35 <= d_old_0)
    {
        _S36 = true;
    }
    else
    {
        _S36 = d_old_0 >= 1.0f;
    }
    if(_S36)
    {
        return float2(d_old_0, 0.0f);
    }
    float u0_0 = (metal::fast::divide((psi_0), (lambda_0 * lambda_0)));
    float _S37 = max(kappa_old_0, 1.0f);
    if(kind_1 == 0U)
    {
        if(r_2 > 1.0f)
        {
            float _S38 = (metal::fast::divide((u0_0 * r_2), (r_2 - 1.0f)));
            return float2(_S35, _S38 * max(min(lambda_0, r_2) - min(_S37, r_2), 0.0f));
        }
        return float2(_S35, (1.0f - d_old_0) * psi_0);
    }
    float ku_0 = 0.5f * (r_2 + 1.0f);
    float plateau_0 = u0_0 * max(min(lambda_0, ku_0) - min(_S37, ku_0), 0.0f);
    float snap_0;
    if(_S35 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return float2(_S35, plateau_0 + snap_0);
}

void compressed_region_0(float w0_0, float w1_0, float dz_0, float ax_0, float ay_0, array<float, int(6)> thread* region_0)
{
    uint count_0;
    float h0_0 = 0.5f * w0_0;
    float h1_0 = 0.5f * w1_0;
    float _S39 = - h0_0;
    float _S40 = - h1_0;
    array<float2, int(4)> _S41 = { { float2(_S39, _S40), float2(h0_0, _S40), float2(h0_0, h1_0), float2(_S39, h1_0) } };
    thread array<float2, int(8)> poly_0;
    uint i_0 = 0U;
    uint count_1 = 0U;
    for(;;)
    {
        if(i_0 < 4U)
        {
        }
        else
        {
            break;
        }
        uint _S42 = i_0;
        uint _S43 = i_0 + 1U;
        uint _S44 = _S43 % 4U;
        float _S45 = _S41[i_0].y;
        float _S46 = _S41[i_0].x;
        float fp_0 = dz_0 + ax_0 * _S45 - ay_0 * _S46;
        float _S47 = _S41[_S44].y;
        float _S48 = _S41[_S44].x;
        float fq_0 = dz_0 + ax_0 * _S47 - ay_0 * _S48;
        bool _S49 = fp_0 < 0.0f;
        if(_S49)
        {
            uint _S50 = count_1 + 1U;
            poly_0[count_1] = _S41[_S42];
            count_0 = _S50;
        }
        else
        {
            count_0 = count_1;
        }
        if(_S49 != (fq_0 < 0.0f))
        {
            float t_0 = fp_0 / (fp_0 - fq_0);
            uint _S51 = count_0 + 1U;
            poly_0[count_0] = float2(_S46 + t_0 * (_S48 - _S46), _S45 + t_0 * (_S47 - _S45));
            count_1 = _S51;
        }
        else
        {
            count_1 = count_0;
        }
        i_0 = _S43;
    }
    count_0 = 0U;
    for(;;)
    {
        if(count_0 < 6U)
        {
        }
        else
        {
            break;
        }
        (*region_0)[count_0] = 0.0f;
        count_0 = count_0 + 1U;
    }
    if(count_1 < 3U)
    {
        return;
    }
    float2 o_0 = poly_0[int(0)];
    i_0 = 0U;
    float a_0 = 0.0f;
    float sx_0 = 0.0f;
    float sy_0 = 0.0f;
    float ixx_0 = 0.0f;
    float iyy_0 = 0.0f;
    float ixy_0 = 0.0f;
    for(;;)
    {
        if(i_0 < count_1)
        {
        }
        else
        {
            break;
        }
        float _S52 = o_0.x;
        float x0_0 = poly_0[i_0].x - _S52;
        float _S53 = o_0.y;
        float y0_0 = poly_0[i_0].y - _S53;
        uint _S54 = i_0 + 1U;
        uint _S55 = _S54 % count_1;
        float x1_0 = poly_0[_S55].x - _S52;
        float y1_0 = poly_0[_S55].y - _S53;
        float _S56 = x0_0 * y1_0;
        float _S57 = x1_0 * y0_0;
        float cr_0 = _S56 - _S57;
        float a_1 = a_0 + cr_0 / 2.0f;
        float sx_1 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        float sy_1 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        float ixx_1 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        float iyy_1 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        float ixy_1 = ixy_0 + (_S56 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S57) * cr_0 / 24.0f;
        i_0 = _S54;
        a_0 = a_1;
        sx_0 = sx_1;
        sy_0 = sy_1;
        ixx_0 = ixx_1;
        iyy_0 = iyy_1;
        ixy_0 = ixy_1;
    }
    if(a_0 <= 0.0f)
    {
        return;
    }
    float cx_0 = sx_0 / a_0;
    float cy_0 = sy_0 / a_0;
    (*region_0)[int(0)] = a_0;
    (*region_0)[int(1)] = o_0.x + cx_0;
    (*region_0)[int(2)] = o_0.y + cy_0;
    float _S58 = a_0 * cx_0;
    (*region_0)[int(3)] = ixx_0 - _S58 * cx_0;
    (*region_0)[int(4)] = iyy_0 - a_0 * cy_0 * cy_0;
    (*region_0)[int(5)] = ixy_0 - _S58 * cy_0;
    return;
}

__attribute__((noinline))
float4 no_tension_patch_0(float kn_0, float w0_1, float w1_1, float dz_1, float ax_1, float ay_1)
{
    thread array<float, int(6)> r_3;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &r_3);
    float a_2 = r_3[int(0)];
    if((r_3[int(0)]) == 0.0f)
    {
        return float4(0.0f) ;
    }
    float k_0 = kn_0 / (w0_1 * w1_1);
    float fc_1 = dz_1 + ax_1 * r_3[int(2)] - ay_1 * r_3[int(1)];
    float _S59 = a_2 * fc_1;
    float _S60 = - ay_1;
    return float4(k_0 * a_2 * fc_1, k_0 * (_S59 * r_3[int(2)] + (_S60 * r_3[int(5)] + ax_1 * r_3[int(4)])), - k_0 * (_S59 * r_3[int(1)] + (_S60 * r_3[int(3)] + ax_1 * r_3[int(5)])), 0.5f * k_0 * (_S59 * fc_1 + ay_1 * ay_1 * r_3[int(3)] + ax_1 * ax_1 * r_3[int(4)] - 2.0f * ax_1 * ay_1 * r_3[int(5)]));
}

float signum_0(float x_2)
{
    float _S61;
    if(((as_type<uint>((x_2))) & 2147483648U) != 0U)
    {
        _S61 = -1.0f;
    }
    else
    {
        _S61 = 1.0f;
    }
    return _S61;
}

float2 return_map_0(float k_1, float total_0, float plastic_0, float cap_0)
{
    float trial_0 = k_1 * (total_0 - plastic_0);
    if((abs(trial_0)) <= cap_0)
    {
        return float2(trial_0, 0.0f);
    }
    float f_1 = cap_0 * signum_0(trial_0);
    float _S62 = (metal::fast::divide((trial_0 - f_1), (k_1)));
    return float2(f_1, _S62);
}

struct Contact_0
{
    float3 q_lin_1;
    float3 q_ang_1;
    float energy_1;
    float diss_0;
    float3 plastic_1;
};

__attribute__((noinline))
Contact_0 contact_part_0(const JointMaterial_natural_0 thread* mat_3, const JointBond_natural_0 thread* b_2, float crush_1, float3 plastic_2, float3 d_lin_0, float3 d_ang_0)
{
    thread Contact_0 c_0;
    float3 _S63 = float3(0.0f) ;
    (&c_0)->q_lin_1 = _S63;
    (&c_0)->q_ang_1 = _S63;
    (&c_0)->energy_1 = 0.0f;
    (&c_0)->diss_0 = 0.0f;
    (&c_0)->plastic_1 = plastic_2;
    uint _S64 = (uint4(mat_3->kind_flags_0) ).y;
    if((_S64 & 2U) == 0U)
    {
        return c_0;
    }
    float4 _S65 = float4(b_2->stiff0_0) ;
    float kn_1 = _S65.x;
    float ks_0 = _S65.y;
    float kt_0 = (float4(b_2->stiff1_0) ).x;
    float4 _S66 = float4(b_2->geom0_0) ;
    float w0_2 = _S66.y;
    float w1_2 = _S66.z;
    float diss_1;
    float nc_sum_0;
    float m1_0;
    float m2_0;
    float energy_2;
    if((_S64 & 4U) != 0U)
    {
        float4 p_0 = no_tension_patch_0(kn_1 * (1.0f - crush_1), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        float _S67 = p_0.y;
        float _S68 = p_0.z;
        float _S69 = p_0.w;
        nc_sum_0 = p_0.x;
        m1_0 = _S67;
        m2_0 = _S68;
        energy_2 = _S69;
    }
    else
    {
        float ki_0 = kn_1 * (1.0f - crush_1) / 36.0f;
        float _S70 = d_ang_0.x;
        float _S71 = d_ang_0.y;
        float spread_0 = abs(_S70) * 0.4166666567325592f * w1_2 + abs(_S71) * 0.4166666567325592f * w0_2;
        float _S72 = d_lin_0.z;
        float slack_0 = 9.99999997475242708e-07f * (abs(_S72) + spread_0);
        if((_S72 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_2 = 0.0f;
        }
        else
        {
            if((_S72 + spread_0) < (- slack_0))
            {
                float i1_0 = 2.91666650772094727f * w0_2 * w0_2;
                float i2_0 = 2.91666650772094727f * w1_2 * w1_2;
                float _S73 = ki_0 * _S70 * i2_0;
                float _S74 = ki_0 * _S71 * i1_0;
                float _S75 = 0.5f * ki_0 * (36.0f * _S72 * _S72 + _S70 * _S70 * i2_0 + _S71 * _S71 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S72;
                m1_0 = _S73;
                m2_0 = _S74;
                energy_2 = _S75;
            }
            else
            {
                uint i_1 = 0U;
                diss_1 = 0.0f;
                float m1_1 = 0.0f;
                float m2_1 = 0.0f;
                float energy_3 = 0.0f;
                for(;;)
                {
                    if(i_1 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S76 = SPRING_AT_0[i_1] * w0_2;
                    uint j_0 = 0U;
                    nc_sum_0 = diss_1;
                    m1_0 = m1_1;
                    m2_0 = m2_1;
                    energy_2 = energy_3;
                    for(;;)
                    {
                        if(j_0 < 6U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float s2_0 = SPRING_AT_0[j_0] * w1_2;
                        float di_0 = _S72 + _S70 * s2_0 - _S71 * _S76;
                        if(di_0 < 0.0f)
                        {
                            float f_2 = ki_0 * di_0;
                            float m1_2 = m1_0 + f_2 * s2_0;
                            float m2_2 = m2_0 - f_2 * _S76;
                            float energy_4 = energy_2 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_2;
                            m1_0 = m1_2;
                            m2_0 = m2_2;
                            energy_2 = energy_4;
                        }
                        j_0 = j_0 + 1U;
                    }
                    i_1 = i_1 + 1U;
                    diss_1 = nc_sum_0;
                    m1_1 = m1_0;
                    m2_1 = m2_0;
                    energy_3 = energy_2;
                }
                nc_sum_0 = diss_1;
                m1_0 = m1_1;
                m2_0 = m2_1;
                energy_2 = energy_3;
            }
        }
    }
    float nc_0 = - nc_sum_0;
    thread float3 p_1 = plastic_2;
    (&c_0)->q_lin_1 = float3(0.0f, 0.0f, nc_sum_0);
    (&c_0)->q_ang_1 = float3(m1_0, m2_0, 0.0f);
    float slide_cap_0 = (float4(mat_3->strength_0) ).w * nc_0;
    float _S77 = ks_0 * (d_lin_0.x - plastic_2.x);
    float _S78 = ks_0 * (d_lin_0.y - plastic_2.y);
    float tn_0 = (metal::fast::sqrt((_S77 * _S77 + _S78 * _S78)));
    bool _S79;
    if(tn_0 > slide_cap_0)
    {
        _S79 = tn_0 > 0.0f;
    }
    else
    {
        _S79 = false;
    }
    if(_S79)
    {
        float _S80 = (metal::fast::divide((_S77), (tn_0)));
        float _S81 = (metal::fast::divide((_S78), (tn_0)));
        float dslip_0 = (metal::fast::divide((tn_0 - slide_cap_0), (ks_0)));
        p_1.x = p_1.x + _S80 * dslip_0;
        p_1.y = p_1.y + _S81 * dslip_0;
        (&c_0)->q_lin_1.x = _S80 * slide_cap_0;
        (&c_0)->q_lin_1.y = _S81 * slide_cap_0;
        diss_1 = slide_cap_0 * dslip_0;
    }
    else
    {
        (&c_0)->q_lin_1.x = _S77;
        (&c_0)->q_lin_1.y = _S78;
        diss_1 = 0.0f;
    }
    float2 tq_0 = return_map_0(kt_0, d_ang_0.z, p_1.z, slide_cap_0 * (float4(b_2->geom1_0) ).z);
    float _S82 = tq_0.x;
    float _S83 = tq_0.y;
    float diss_2 = diss_1 + abs(_S82) * abs(_S83);
    p_1.z = p_1.z + _S83;
    (&c_0)->q_ang_1.z = _S82;
    float _S84 = (metal::fast::divide((sq_0((&c_0)->q_lin_1.x)), (ks_0)));
    float _S85 = (metal::fast::divide((sq_0((&c_0)->q_lin_1.y)), (ks_0)));
    float _S86 = _S84 + _S85;
    float _S87 = (metal::fast::divide((sq_0(_S82)), (kt_0)));
    (&c_0)->energy_1 = energy_2 + 0.5f * (_S86 + _S87);
    (&c_0)->diss_0 = diss_2;
    (&c_0)->plastic_1 = p_1;
    return c_0;
}

float3 contact_offsets_0(const JointMaterial_natural_0 thread* mat_4, const JointBond_natural_0 thread* b_3, float crush_2, float3 plastic_3, float3 d_lin_1, float3 d_ang_1)
{
    uint _S88 = (uint4(mat_4->kind_flags_0) ).y;
    if((_S88 & 2U) == 0U)
    {
        return plastic_3;
    }
    float4 _S89 = float4(b_3->stiff0_0) ;
    float kn_2 = _S89.x;
    float ks_1 = _S89.y;
    float kt_1 = (float4(b_3->stiff1_0) ).x;
    float4 _S90 = float4(b_3->geom0_0) ;
    float w0_3 = _S90.y;
    float w1_3 = _S90.z;
    float nc_sum_1;
    if((_S88 & 4U) != 0U)
    {
        float4 _S91 = no_tension_patch_0(kn_2 * (1.0f - crush_2), w0_3, w1_3, d_lin_1.z, d_ang_1.x, d_ang_1.y);
        nc_sum_1 = _S91.x;
    }
    else
    {
        float ki_1 = kn_2 * (1.0f - crush_2) / 36.0f;
        float _S92 = d_ang_1.x;
        float _S93 = d_ang_1.y;
        float spread_1 = abs(_S92) * 0.4166666567325592f * w1_3 + abs(_S93) * 0.4166666567325592f * w0_3;
        float _S94 = d_lin_1.z;
        float slack_1 = 9.99999997475242708e-07f * (abs(_S94) + spread_1);
        if((_S94 - spread_1) > slack_1)
        {
            nc_sum_1 = 0.0f;
        }
        else
        {
            if((_S94 + spread_1) < (- slack_1))
            {
                nc_sum_1 = ki_1 * 36.0f * _S94;
            }
            else
            {
                uint i_2 = 0U;
                float nc_sum_2 = 0.0f;
                for(;;)
                {
                    if(i_2 < 6U)
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S95 = SPRING_AT_0[i_2] * w0_3;
                    uint j_1 = 0U;
                    nc_sum_1 = nc_sum_2;
                    for(;;)
                    {
                        if(j_1 < 6U)
                        {
                        }
                        else
                        {
                            break;
                        }
                        float di_1 = _S94 + _S92 * (SPRING_AT_0[j_1] * w1_3) - _S93 * _S95;
                        if(di_1 < 0.0f)
                        {
                            nc_sum_1 = nc_sum_1 + ki_1 * di_1;
                        }
                        j_1 = j_1 + 1U;
                    }
                    i_2 = i_2 + 1U;
                    nc_sum_2 = nc_sum_1;
                }
                nc_sum_1 = nc_sum_2;
            }
        }
    }
    float nc_1 = - nc_sum_1;
    thread float3 p_2 = plastic_3;
    float slide_cap_1 = (float4(mat_4->strength_0) ).w * nc_1;
    float _S96 = ks_1 * (d_lin_1.x - plastic_3.x);
    float _S97 = ks_1 * (d_lin_1.y - plastic_3.y);
    float tn_1 = (metal::fast::sqrt((_S96 * _S96 + _S97 * _S97)));
    bool _S98;
    if(tn_1 > slide_cap_1)
    {
        _S98 = tn_1 > 0.0f;
    }
    else
    {
        _S98 = false;
    }
    if(_S98)
    {
        float _S99 = (metal::fast::divide((_S96), (tn_1)));
        float _S100 = (metal::fast::divide((_S97), (tn_1)));
        float dslip_1 = (metal::fast::divide((tn_1 - slide_cap_1), (ks_1)));
        p_2.x = p_2.x + _S99 * dslip_1;
        p_2.y = p_2.y + _S100 * dslip_1;
    }
    float2 tq_1 = return_map_0(kt_1, d_ang_1.z, p_2.z, slide_cap_1 * (float4(b_3->geom1_0) ).z);
    p_2.z = p_2.z + tq_1.y;
    return p_2;
}

float life_rate_0(const JointMaterial_natural_0 thread* mat_5, float s_0)
{
    if(s_0 <= 0.0f)
    {
        return 0.0f;
    }
    float4 _S101 = float4(mat_5->misc_0) ;
    float _S102 = _S101.y;
    float _S103 = _S102 + 1.0f;
    float _S104 = (metal::fast::pow((s_0), (_S102)));
    float _S105 = (metal::fast::divide((_S103 * _S104), (_S101.z)));
    return _S105;
}

struct JointResponse_0
{
    float3 force_lin_0;
    float3 force_ang_0;
    JointState_0 state_0;
    float dissipated_1;
    float overshoot_0;
    float stored_0;
    bool disconnected_0;
    float fatigue_before_0;
    Measures_0 measures_0;
};

JointResponse_0 joint_evaluate_0(const JointMaterial_natural_0 thread* mat_6, const JointBond_natural_0 thread* b_4, const JointState_0 thread* state_1, float3 d_lin_2, float3 d_ang_2, float dt_0, bool fracture_0)
{
    float4 _S106 = float4(b_4->stiff0_0) ;
    float kn_3 = _S106.x;
    float ks_2 = _S106.y;
    float kb1_0 = _S106.z;
    float kb2_0 = _S106.w;
    float4 _S107 = float4(b_4->stiff1_0) ;
    float kt_2 = _S107.x;
    bool has_rebar_1 = (_S107.w) != 0.0f;
    uint4 _S108 = uint4(mat_6->kind_flags_0) ;
    uint kind_2 = _S108.x;
    uint flags_0 = _S108.y;
    bool softening_0 = (flags_0 & 1U) != 0U;
    thread JointState_0 st_1 = *state_1;
    bool _S109 = connected_0(state_1, has_rebar_1);
    float3 qe_lin_0 = d_lin_2 * float3(ks_2, ks_2, kn_3);
    float3 qe_ang_0 = d_ang_2 * float3(kb1_0, kb2_0, kt_2);
    Measures_0 _S110 = stress_measures_0(b_4, qe_lin_0, qe_ang_0);
    float _S111 = max(max(_S110.tension_0, _S110.shear_0), _S110.compression_0);
    bool _S112 = dt_0 > 0.0f;
    float dif_1;
    if(_S112)
    {
        float _S113 = (metal::fast::divide((_S111 - (&st_1)->governing_stress_0), (dt_0)));
        float raw_0 = (metal::fast::divide((max(_S113, 0.0f)), ((float4(mat_6->misc_0) ).w)));
        float tau_0 = _S107.z;
        if((flags_0 & 16U) != 0U)
        {
            float _S114 = (metal::fast::divide((dt_0), (tau_0)));
            dif_1 = - expm1_accurate_0(- _S114);
        }
        else
        {
            float _S115 = (metal::fast::divide((dt_0), (tau_0)));
            dif_1 = min(_S115, 1.0f);
        }
        (&st_1)->strain_rate_0 = (&st_1)->strain_rate_0 + (raw_0 - (&st_1)->strain_rate_0) * dif_1;
        (&st_1)->governing_stress_0 = _S111;
    }
    if((flags_0 & 32U) != 0U)
    {
        float _S116 = dif_factor_0(mat_6, (&st_1)->strain_rate_0);
        dif_1 = _S116;
    }
    else
    {
        dif_1 = 1.0f;
    }
    float weibull_0 = (float4(b_4->geom1_0) ).w;
    float _S117 = fatigue_factor_0(mat_6, (&st_1)->life_0);
    float multiplier_1 = weibull_0 * dif_1 * _S117;
    thread Measures_0 _S118 = _S110;
    float4 _S119 = failure_indices_0(mat_6, b_4, &_S118, multiplier_1);
    float _S120 = _S119.x;
    float _S121 = _S119.y;
    (&st_1)->utilization_0 = max(max(_S120, _S121), max(_S119.z, _S119.w));
    float _S122 = d_lin_2.x;
    float _S123 = d_lin_2.y;
    float _S124 = ks_2 * (sq_0(_S122) + sq_0(_S123)) + kb1_0 * sq_0(d_ang_2.x) + kb2_0 * sq_0(d_ang_2.y) + kt_2 * sq_0(d_ang_2.z);
    float _S125 = d_lin_2.z;
    bool _S126 = _S125 > 0.0f;
    if(_S126)
    {
        dif_1 = kn_3 * sq_0(_S125);
    }
    else
    {
        dif_1 = 0.0f;
    }
    float psi_ts_0 = 0.5f * (_S124 + dif_1);
    float psi_c_0;
    if(_S125 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S125);
    }
    else
    {
        psi_c_0 = 0.0f;
    }
    float3 plastic_4 = float3((&st_1)->plastic_x_0, (&st_1)->plastic_y_0, (&st_1)->plastic_t_0);
    float diss_contact_0;
    float psi_contact_0;
    float intact_normal_0;
    float dissipated_2;
    float overshoot_1;
    bool _S127;
    float3 qc_lin_0;
    if(fracture_0)
    {
        bool _S128 = _S120 >= _S121;
        if(_S128)
        {
            diss_contact_0 = _S120;
        }
        else
        {
            diss_contact_0 = _S121;
        }
        uint mode_ts_0;
        if(_S128)
        {
            mode_ts_0 = 1U;
        }
        else
        {
            mode_ts_0 = 2U;
        }
        if(diss_contact_0 > ((&st_1)->kappa_0))
        {
            _S127 = diss_contact_0 > 1.0f;
        }
        else
        {
            _S127 = false;
        }
        if(_S127)
        {
            _S127 = psi_ts_0 > 0.0f;
        }
        else
        {
            _S127 = false;
        }
        uint mode_c_0;
        if(_S127)
        {
            if(mode_ts_0 == 1U)
            {
                psi_contact_0 = (float4(mat_6->energy_0) ).y;
            }
            else
            {
                psi_contact_0 = (float4(mat_6->energy_0) ).z;
            }
            if(softening_0)
            {
                float _S129 = (metal::fast::divide((psi_contact_0 * (float4(b_4->geom0_0) ).x * diss_contact_0 * diss_contact_0), (psi_ts_0)));
                intact_normal_0 = _S129;
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            (&st_1)->ductility_0 = intact_normal_0;
            if(softening_0)
            {
                mode_c_0 = kind_2;
            }
            else
            {
                mode_c_0 = 0U;
            }
            float2 inc_0 = damage_increment_0(mode_c_0, (&st_1)->kappa_0, diss_contact_0, intact_normal_0, (&st_1)->damage_0, psi_ts_0);
            float _S130 = inc_0.x;
            if(_S130 > ((&st_1)->damage_0))
            {
                Contact_0 _S131 = contact_part_0(mat_6, b_4, (&st_1)->crush_0, plastic_4, d_lin_2, d_ang_2);
                float _S132 = max(_S131.energy_1 - (1.0f - (&st_1)->crush_0) * psi_c_0, 0.0f);
                float _S133 = max(inc_0.y - _S132 * (_S130 - (&st_1)->damage_0), 0.0f);
                float _S134 = max((psi_ts_0 - _S132) * (_S130 - (&st_1)->damage_0) - _S133, 0.0f);
                (&st_1)->damage_0 = _S130;
                (&st_1)->mode_0 = mode_ts_0;
                dissipated_2 = _S133;
                overshoot_1 = _S134;
            }
            else
            {
                dissipated_2 = 0.0f;
                overshoot_1 = 0.0f;
            }
        }
        else
        {
            dissipated_2 = 0.0f;
            overshoot_1 = 0.0f;
        }
        (&st_1)->kappa_0 = max((&st_1)->kappa_0, diss_contact_0);
        float _S135 = state_1->damage_0;
        if((state_1->damage_0) > 0.0f)
        {
            Contact_0 _S136 = contact_part_0(mat_6, b_4, state_1->crush_0, float3(state_1->plastic_x_0, state_1->plastic_y_0, state_1->plastic_t_0), d_lin_2, d_ang_2);
            qc_lin_0 = qe_ang_0 * float3((1.0f - _S135))  + _S136.q_ang_1 * float3(_S135) ;
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        Measures_0 _S137 = stress_measures_0(b_4, float3(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        thread Measures_0 _S138 = _S137;
        float4 _S139 = failure_indices_0(mat_6, b_4, &_S138, multiplier_1);
        float _S140 = _S139.z;
        float _S141 = _S139.w;
        bool _S142 = _S140 >= _S141;
        if(_S142)
        {
            psi_contact_0 = _S140;
        }
        else
        {
            psi_contact_0 = _S141;
        }
        if(_S142)
        {
            mode_c_0 = 3U;
        }
        else
        {
            mode_c_0 = 4U;
        }
        if(psi_contact_0 > ((&st_1)->kappa_c_0))
        {
            _S127 = psi_contact_0 > 1.0f;
        }
        else
        {
            _S127 = false;
        }
        if(_S127)
        {
            _S127 = psi_c_0 > 0.0f;
        }
        else
        {
            _S127 = false;
        }
        if(_S127)
        {
            if(softening_0)
            {
                float _S143 = (metal::fast::divide(((float4(mat_6->energy_0) ).w * (float4(b_4->geom0_0) ).x * psi_contact_0 * psi_contact_0), (psi_c_0)));
                intact_normal_0 = _S143;
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            (&st_1)->ductility_c_0 = intact_normal_0;
            uint law_0;
            if(!softening_0)
            {
                law_0 = 0U;
            }
            else
            {
                if(mode_c_0 == 4U)
                {
                    mode_ts_0 = 1U;
                }
                else
                {
                    mode_ts_0 = kind_2;
                }
                law_0 = mode_ts_0;
            }
            float2 inc_1 = damage_increment_0(law_0, (&st_1)->kappa_c_0, psi_contact_0, intact_normal_0, (&st_1)->crush_0, psi_c_0);
            float _S144 = inc_1.x;
            if(_S144 > ((&st_1)->crush_0))
            {
                float _S145 = inc_1.y;
                float dissipated_3 = dissipated_2 + _S145;
                float overshoot_2 = overshoot_1 + max(psi_c_0 * (_S144 - (&st_1)->crush_0) - _S145, 0.0f);
                (&st_1)->crush_0 = _S144;
                (&st_1)->mode_0 = mode_c_0;
                if(_S144 >= 1.0f)
                {
                    _S127 = ((&st_1)->damage_0) < 1.0f;
                }
                else
                {
                    _S127 = false;
                }
                if(_S127)
                {
                    float dissipated_4 = dissipated_3 + psi_ts_0 * (1.0f - (&st_1)->damage_0);
                    (&st_1)->damage_0 = 1.0f;
                    dissipated_2 = dissipated_4;
                }
                else
                {
                    dissipated_2 = dissipated_3;
                }
                overshoot_1 = overshoot_2;
            }
        }
        (&st_1)->kappa_c_0 = max((&st_1)->kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_2 = 0.0f;
        overshoot_1 = 0.0f;
    }
    float dmg_0 = (&st_1)->damage_0;
    float3 _S146 = float3(0.0f) ;
    float3 qc_ang_0;
    if(((&st_1)->damage_0) == 0.0f)
    {
        if((flags_0 & 8U) == 0U)
        {
            float3 _S147 = contact_offsets_0(mat_6, b_4, (&st_1)->crush_0, plastic_4, d_lin_2, d_ang_2);
            (&st_1)->plastic_x_0 = _S147.x;
            (&st_1)->plastic_y_0 = _S147.y;
            (&st_1)->plastic_t_0 = _S147.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S146;
        qc_ang_0 = _S146;
        psi_contact_0 = 0.0f;
    }
    else
    {
        Contact_0 _S148 = contact_part_0(mat_6, b_4, (&st_1)->crush_0, plastic_4, d_lin_2, d_ang_2);
        (&st_1)->plastic_x_0 = _S148.plastic_1.x;
        (&st_1)->plastic_y_0 = _S148.plastic_1.y;
        (&st_1)->plastic_t_0 = _S148.plastic_1.z;
        diss_contact_0 = _S148.diss_0;
        qc_lin_0 = _S148.q_lin_1;
        qc_ang_0 = _S148.q_ang_1;
        psi_contact_0 = _S148.energy_1;
    }
    float dissipated_5 = dissipated_2 + dmg_0 * diss_contact_0;
    if(_S126)
    {
        intact_normal_0 = kn_3 * _S125;
    }
    else
    {
        intact_normal_0 = (1.0f - (&st_1)->crush_0) * kn_3 * _S125;
    }
    float _S149 = 1.0f - dmg_0;
    float3 force_lin_1 = float3(_S149 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S149 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S149 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    float3 force_ang_1 = qe_ang_0 * float3(_S149)  + qc_ang_0 * float3(dmg_0) ;
    float stored_1 = _S149 * (psi_ts_0 + (1.0f - (&st_1)->crush_0) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_1)
    {
        _S127 = ((&st_1)->rebar_broken_0) == 0.0f;
    }
    else
    {
        _S127 = false;
    }
    float stored_2;
    float3 force_lin_2;
    if(_S127)
    {
        float4 _S150 = float4(b_4->rebar0_0) ;
        float k_axial_0 = _S150.x;
        float k_dowel_0 = _S150.y;
        float yield_force_0 = _S150.z;
        float dowel_capacity_0 = _S150.w;
        float2 nr_0 = return_map_0(k_axial_0, _S125, (&st_1)->rebar_plastic_0, yield_force_0);
        float2 v1_0 = return_map_0(k_dowel_0, _S122, (&st_1)->rebar_slip0_0, dowel_capacity_0);
        float2 v2_0 = return_map_0(k_dowel_0, _S123, (&st_1)->rebar_slip1_0, dowel_capacity_0);
        float _S151 = nr_0.y;
        float _S152 = v1_0.y;
        float _S153 = v2_0.y;
        float work_0 = yield_force_0 * abs(_S151) + dowel_capacity_0 * (abs(_S152) + abs(_S153));
        (&st_1)->rebar_plastic_0 = (&st_1)->rebar_plastic_0 + _S151;
        (&st_1)->rebar_slip0_0 = (&st_1)->rebar_slip0_0 + _S152;
        (&st_1)->rebar_slip1_0 = (&st_1)->rebar_slip1_0 + _S153;
        (&st_1)->rebar_work_0 = (&st_1)->rebar_work_0 + work_0;
        float dissipated_6 = dissipated_5 + work_0;
        float _S154 = nr_0.x;
        float _S155 = (metal::fast::divide((sq_0(_S154)), (k_axial_0)));
        float _S156 = v1_0.x;
        float _S157 = v2_0.x;
        float _S158 = (metal::fast::divide((sq_0(_S156) + sq_0(_S157)), (k_dowel_0)));
        float elastic_0 = 0.5f * (_S155 + _S158);
        if(fracture_0)
        {
            _S127 = ((&st_1)->rebar_work_0) >= ((float4(b_4->rebar1_0) ).x);
        }
        else
        {
            _S127 = false;
        }
        if(_S127)
        {
            (&st_1)->rebar_broken_0 = 1.0f;
            float dissipated_7 = dissipated_6 + elastic_0;
            force_lin_2 = force_lin_1;
            dissipated_2 = dissipated_7;
            stored_2 = stored_1;
        }
        else
        {
            float stored_3 = stored_1 + elastic_0;
            force_lin_2 = force_lin_1 + float3(_S156, _S157, _S154);
            dissipated_2 = dissipated_6;
            stored_2 = stored_3;
        }
    }
    else
    {
        force_lin_2 = force_lin_1;
        dissipated_2 = dissipated_5;
        stored_2 = stored_1;
    }
    if(fracture_0)
    {
        _S127 = _S112;
    }
    else
    {
        _S127 = false;
    }
    if(_S127)
    {
        _S127 = (flags_0 & 64U) != 0U;
    }
    else
    {
        _S127 = false;
    }
    if(_S127)
    {
        Measures_0 _S159 = stress_measures_0(b_4, force_lin_2, force_ang_1);
        thread Measures_0 _S160 = _S159;
        float4 _S161 = failure_indices_0(mat_6, b_4, &_S160, weibull_0);
        float _S162 = life_rate_0(mat_6, max(max(_S161.x, _S161.y), _S161.z));
        (&st_1)->life_0 = max((&st_1)->life_0 - _S162 * dt_0, 0.0f);
    }
    (&st_1)->dissipated_0 = (&st_1)->dissipated_0 + dissipated_2;
    thread JointResponse_0 resp_0;
    (&resp_0)->force_lin_0 = force_lin_2;
    (&resp_0)->force_ang_0 = force_ang_1;
    (&resp_0)->state_0 = st_1;
    (&resp_0)->dissipated_1 = dissipated_2;
    (&resp_0)->overshoot_0 = overshoot_1;
    (&resp_0)->stored_0 = stored_2;
    if(_S109)
    {
        thread JointState_0 _S163 = st_1;
        bool _S164 = connected_0(&_S163, has_rebar_1);
        _S127 = !_S164;
    }
    else
    {
        _S127 = false;
    }
    (&resp_0)->disconnected_0 = _S127;
    (&resp_0)->fatigue_before_0 = _S117;
    (&resp_0)->measures_0 = _S110;
    return resp_0;
}

void secant_factors_0(const JointBond_natural_0 thread* b_5, const JointState_0 thread* st_2, float3 d_lin_3, float3 thread* f_lin_0, float3 thread* f_ang_0)
{
    float _S165 = st_2->damage_0;
    bool compressed_0 = (d_lin_3.z) < 0.0f;
    float contact_0;
    if(compressed_0)
    {
        contact_0 = _S165;
    }
    else
    {
        contact_0 = 0.0f;
    }
    float _S166 = 1.0f - _S165;
    float _S167 = max(_S166 + contact_0, 9.99999997475242708e-07f);
    float normal_0;
    if(compressed_0)
    {
        normal_0 = max(1.0f - st_2->crush_0, 9.99999997475242708e-07f);
    }
    else
    {
        normal_0 = max(_S166, 9.99999997475242708e-07f);
    }
    *f_lin_0 = float3(_S167, _S167, normal_0);
    *f_ang_0 = float3(_S167) ;
    bool _S168;
    if(((float4(b_5->stiff1_0) ).w) != 0.0f)
    {
        _S168 = (st_2->rebar_broken_0) == 0.0f;
    }
    else
    {
        _S168 = false;
    }
    if(_S168)
    {
        float4 _S169 = float4(b_5->rebar0_0) ;
        float4 _S170 = float4(b_5->stiff0_0) ;
        float _S171 = (metal::fast::divide((_S169.x), (_S170.x)));
        (*f_lin_0).z = (*f_lin_0).z + _S171;
        float _S172 = _S169.y;
        float _S173 = _S170.y;
        float _S174 = (metal::fast::divide((_S172), (_S173)));
        (*f_lin_0).x = (*f_lin_0).x + _S174;
        float _S175 = (metal::fast::divide((_S172), (_S173)));
        (*f_lin_0).y = (*f_lin_0).y + _S175;
    }
    return;
}

struct TestParams_0
{
    uint count_2;
    uint pad0_0;
    uint pad1_0;
    uint pad2_0;
};

struct KernelContext_0
{
    TestParams_0 constant* params_0;
    JointBond_natural_0 device* bonds_0;
    JointMaterial_natural_0 device* materials_0;
    packed_float4 device* inputs_0;
    JointState_0 device* states_in_0;
    JointState_0 device* states_out_0;
    packed_float4 device* outputs_0;
};

[[kernel]] void joint_eval_test(uint3 id_0 [[thread_position_in_grid]], TestParams_0 constant* params_1 [[buffer(0)]], JointBond_natural_0 device* bonds_1 [[buffer(2)]], JointMaterial_natural_0 device* materials_1 [[buffer(1)]], packed_float4 device* inputs_1 [[buffer(4)]], JointState_0 device* states_in_1 [[buffer(3)]], JointState_0 device* states_out_1 [[buffer(5)]], packed_float4 device* outputs_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->bonds_0 = bonds_1;
    (&kernelContext_0)->materials_0 = materials_1;
    (&kernelContext_0)->inputs_0 = inputs_1;
    (&kernelContext_0)->states_in_0 = states_in_1;
    (&kernelContext_0)->states_out_0 = states_out_1;
    (&kernelContext_0)->outputs_0 = outputs_1;
    uint i_3 = id_0.x;
    if(i_3 >= (params_1->count_2))
    {
        return;
    }
    JointBond_natural_0 b_6 = (&kernelContext_0)->bonds_0[i_3];
    thread JointBond_natural_0 _S176 = b_6;
    uint _S177 = 2U * i_3;
    float4 _S178 = float4(*((&kernelContext_0)->inputs_0+_S177)) ;
    float4 _S179 = float4(*((&kernelContext_0)->inputs_0+(_S177 + 1U))) ;
    JointState_0 _S180 = (&kernelContext_0)->states_in_0[i_3];
    float3 _S181 = _S178.xyz;
    float3 _S182 = _S179.xyz;
    float _S183 = _S178.w;
    bool _S184 = (_S179.w) != 0.0f;
    thread JointMaterial_natural_0 _S185 = (&kernelContext_0)->materials_0[(uint4((&_S176)->ids_0) ).x];
    _S176 = b_6;
    thread JointState_0 _S186 = _S180;
    JointResponse_0 _S187 = joint_evaluate_0(&_S185, &_S176, &_S186, _S181, _S182, _S183, _S184);
    thread JointState_0 _S188 = _S187.state_0;
    thread float3 f_lin_1;
    thread float3 f_ang_1;
    secant_factors_0(&_S176, &_S188, _S181, &f_lin_1, &f_ang_1);
    *((&kernelContext_0)->states_out_0+i_3) = _S187.state_0;
    uint _S189 = 5U * i_3;
    *((&kernelContext_0)->outputs_0+_S189) = packed_float4(float4(_S187.force_lin_0, _S187.dissipated_1)) ;
    *((&kernelContext_0)->outputs_0+(_S189 + 1U)) = packed_float4(float4(_S187.force_ang_0, _S187.overshoot_0)) ;
    *((&kernelContext_0)->outputs_0+(_S189 + 2U)) = packed_float4(float4(f_lin_1, _S187.stored_0)) ;
    packed_float4 device* _S190 = (&kernelContext_0)->outputs_0+(_S189 + 3U);
    float3 _S191 = f_ang_1;
    float _S192;
    if(_S187.disconnected_0)
    {
        _S192 = 1.0f;
    }
    else
    {
        _S192 = 0.0f;
    }
    *_S190 = packed_float4(float4(_S191, _S192)) ;
    *((&kernelContext_0)->outputs_0+(_S189 + 4U)) = packed_float4(float4(_S187.measures_0.tension_0, _S187.measures_0.shear_0, _S187.measures_0.compression_0, _S187.measures_0.compressive_force_0)) ;
    return;
}

