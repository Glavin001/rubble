#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
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
    uint r_virial_0;
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
    BlastBond_natural_0 device* bonds_0;
    packed_float4 device* x_0;
    packed_float4 device* stiffness_0;
    packed_float4 device* out_0;
    packed_float4 device* geometry_0;
    packed_float4 device* sections_0;
    packed_float4 device* materials_0;
    atomic<uint> device* topo_0;
};

float3 stiff_0(uint j_0, uint which_0, float3 v_0, KernelContext_0 thread* kernelContext_0)
{
    uint _S1 = 6U * which_0;
    thread array<float, int(6)> m_0;
    uint k_0 = 0U;
    for(;;)
    {
        if(k_0 < 6U)
        {
        }
        else
        {
            break;
        }
        uint _S2 = _S1 + k_0;
        m_0[k_0] = (float4(*(kernelContext_0->stiffness_0+(j_0 * 6U + _S2 / 4U))) )[_S2 % 4U];
        k_0 = k_0 + 1U;
    }
    float _S3 = v_0.x;
    float _S4 = v_0.y;
    float _S5 = v_0.z;
    return float3(m_0[int(0)] * _S3 + m_0[int(3)] * _S4 + m_0[int(4)] * _S5, m_0[int(3)] * _S3 + m_0[int(1)] * _S4 + m_0[int(5)] * _S5, m_0[int(4)] * _S3 + m_0[int(5)] * _S4 + m_0[int(2)] * _S5);
}

float fdot_0(float3 a_0, float3 b_0)
{
    return fma(a_0.z, b_0.z, fma(a_0.y, b_0.y, a_0.x * b_0.x));
}

float perpendicular_0(float3 v_1, float3 n_0, float along_0)
{
    float _S6 = - along_0;
    float px_0 = fma(_S6, n_0.x, v_1.x);
    float py_0 = fma(_S6, n_0.y, v_1.y);
    float pz_0 = fma(_S6, n_0.z, v_1.z);
    return sqrt(fma(pz_0, pz_0, fma(py_0, py_0, px_0 * px_0)));
}

float excess_0(float value_0, float elastic_0, float fatal_0)
{
    float d_0 = fatal_0 - elastic_0;
    float _S7 = value_0 - elastic_0;
    float _S8;
    if(d_0 > 0.0f)
    {
        _S8 = d_0;
    }
    else
    {
        _S8 = 1.0f;
    }
    return _S7 / _S8;
}

uint topo_total_0(uint node_count_1)
{
    return 10U * node_count_1 + 3U * ((node_count_1 + 1023U) / 1024U);
}

uint topo_dirty_0(uint node_count_2)
{
    return topo_total_0(node_count_2) + 4U;
}

[[kernel]] void blast_bond_stress(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], BlastBond_natural_0 device* bonds_1 [[buffer(1)]], packed_float4 device* x_1 [[buffer(2)]], packed_float4 device* stiffness_1 [[buffer(7)]], packed_float4 device* out_1 [[buffer(5)]], packed_float4 device* geometry_1 [[buffer(3)]], packed_float4 device* sections_1 [[buffer(8)]], packed_float4 device* materials_1 [[buffer(4)]], atomic<uint> device* topo_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_1;
    (&kernelContext_1)->params_0 = params_1;
    (&kernelContext_1)->bonds_0 = bonds_1;
    (&kernelContext_1)->x_0 = x_1;
    (&kernelContext_1)->stiffness_0 = stiffness_1;
    (&kernelContext_1)->out_0 = out_1;
    (&kernelContext_1)->geometry_0 = geometry_1;
    (&kernelContext_1)->sections_0 = sections_1;
    (&kernelContext_1)->materials_0 = materials_1;
    (&kernelContext_1)->topo_0 = topo_1;
    uint j_1 = id_0.x;
    if(j_1 >= (params_1->bond_count_0))
    {
        return;
    }
    BlastBond_natural_0 b_1 = *((&kernelContext_1)->bonds_0+j_1);
    uint _S9 = j_1 * 2U;
    float3 xa_0 = (float4(*((&kernelContext_1)->x_0+_S9)) ).xyz;
    uint _S10 = _S9 + 1U;
    float3 xl_0 = (float4(*((&kernelContext_1)->x_0+_S10)) ).xyz;
    float3 xa_1;
    if(((&kernelContext_1)->params_0->rotation_0) != 0U)
    {
        float3 _S11 = stiff_0(j_1, 0U, xa_0, &kernelContext_1);
        xa_1 = _S11;
    }
    else
    {
        xa_1 = xa_0;
    }
    float3 xl_1;
    if(((&kernelContext_1)->params_0->rotation_0) == 2U)
    {
        float3 _S12 = stiff_0(j_1, 2U, xl_0, &kernelContext_1);
        xl_1 = _S12;
    }
    else
    {
        xl_1 = xl_0;
    }
    float3 ia_0 = xa_1 * float3((b_1.s_0 * (&kernelContext_1)->params_0->angular_scale_0)) ;
    float3 il_0 = xl_1 * float3((b_1.s_0 * (&kernelContext_1)->params_0->linear_scale_0)) ;
    uint _S13 = j_1 * 4U;
    *((&kernelContext_1)->out_0+_S13) = packed_float4(float4(ia_0, 0.0f)) ;
    *((&kernelContext_1)->out_0+(_S13 + 1U)) = packed_float4(float4(il_0, 0.0f)) ;
    if(!((b_1.health_0) > 0.0f))
    {
        packed_float4 _S14 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_1)->out_0+(_S13 + 2U)) = _S14;
        *((&kernelContext_1)->out_0+(_S13 + 3U)) = _S14;
        return;
    }
    float4 _S15 = float4(*((&kernelContext_1)->geometry_0+_S9)) ;
    float4 _S16 = float4(*((&kernelContext_1)->geometry_0+_S10)) ;
    float3 n_1 = _S15.xyz;
    float node_distance_0 = _S15.w;
    float linear_normal_0 = fdot_0(il_0, n_1);
    float normal_0 = linear_normal_0 / b_1.health_0;
    float shear_0 = perpendicular_0(il_0, n_1, linear_normal_0) / b_1.health_0;
    float angular_normal_0 = fdot_0(ia_0, n_1);
    float twist_0 = abs(angular_normal_0) / b_1.health_0;
    float bend_0 = perpendicular_0(ia_0, n_1, angular_normal_0) / b_1.health_0;
    float gain_max_0 = (&kernelContext_1)->params_0->bend_gain_max_0;
    float bend_stress_0;
    float bearing_tension_0;
    float normal_1;
    float shear_1;
    bool _S17;
    if(((&kernelContext_1)->params_0->section_bending_0) != 0U)
    {
        uint _S18 = j_1 * 3U;
        float4 _S19 = float4(*((&kernelContext_1)->sections_0+_S18)) ;
        float4 _S20 = float4(*((&kernelContext_1)->sections_0+(_S18 + 1U))) ;
        float3 m_1 = ia_0 + cross((float4(*((&kernelContext_1)->sections_0+(_S18 + 2U))) ).xyz, il_0);
        float m_normal_0 = fdot_0(m_1, n_1);
        float tw_0 = abs(m_normal_0);
        float3 axis_0 = _S19.xyz;
        float _S21 = n_1.y;
        float _S22 = axis_0.z;
        float _S23 = n_1.z;
        float _S24 = axis_0.y;
        float _S25 = axis_0.x;
        float _S26 = n_1.x;
        float3 axis1_0 = float3(fma(_S21, _S22, - _S23 * _S24), fma(_S23, _S25, - _S26 * _S22), fma(_S26, _S24, - _S21 * _S25));
        float original_0 = _S16.x;
        float _S27 = _S19.w;
        if(_S27 > 0.0f)
        {
            _S17 = (_S20.x) > 0.0f;
        }
        else
        {
            _S17 = false;
        }
        if(_S17)
        {
            _S17 = (_S20.y) > 0.0f;
        }
        else
        {
            _S17 = false;
        }
        if(_S17)
        {
            _S17 = original_0 > 0.0f;
        }
        else
        {
            _S17 = false;
        }
        if(_S17)
        {
            float live_0 = b_1.health_0 / original_0;
            float _S28 = fma(tw_0, 1.0f / (_S20.y * live_0), shear_0);
            bend_stress_0 = (abs(fdot_0(m_1, axis_0)) / _S27 + abs(fdot_0(m_1, axis1_0)) / _S20.x) / live_0;
            normal_1 = _S28;
        }
        else
        {
            float cube_0 = b_1.health_0 * sqrt(b_1.health_0);
            float _S29 = fma(tw_0, 4.24264049530029297f / cube_0, shear_0);
            bend_stress_0 = 6.0f * perpendicular_0(m_1, n_1, m_normal_0) / cube_0;
            normal_1 = _S29;
        }
        float _S30 = _S20.z;
        if(_S30 > 0.0f)
        {
            _S17 = (_S20.w) > 0.0f;
        }
        else
        {
            _S17 = false;
        }
        if(_S17)
        {
            bearing_tension_0 = max(0.0f, abs(dot(m_1, axis_0)) / _S30 + abs(dot(m_1, axis1_0)) / _S20.w + normal_0 * b_1.health_0) / b_1.health_0;
        }
        else
        {
            bearing_tension_0 = -1.0f;
        }
        float _S31 = normal_1;
        normal_1 = normal_0;
        shear_1 = _S31;
    }
    else
    {
        if(gain_max_0 > 0.0f)
        {
            bool _S32 = (b_1.health_0) > 9.99999997475242708e-07f;
            if(_S32)
            {
                bend_stress_0 = b_1.health_0;
            }
            else
            {
                bend_stress_0 = 9.99999997475242708e-07f;
            }
            float torsion_0 = 4.80999994277954102f / sqrt(bend_stress_0);
            if(torsion_0 < gain_max_0)
            {
                bend_stress_0 = torsion_0;
            }
            else
            {
                bend_stress_0 = gain_max_0;
            }
            float _S33 = fma(twist_0, bend_stress_0, shear_0);
            if(_S32)
            {
                bend_stress_0 = b_1.health_0;
            }
            else
            {
                bend_stress_0 = 9.99999997475242708e-07f;
            }
            float section_0 = 6.0f / sqrt(bend_stress_0);
            if(section_0 < gain_max_0)
            {
                bend_stress_0 = section_0;
            }
            else
            {
                bend_stress_0 = gain_max_0;
            }
            bend_stress_0 = bend_0 * bend_stress_0;
            bearing_tension_0 = normal_0;
            normal_1 = _S33;
        }
        else
        {
            float shear_2 = shear_0 + twist_0 * 2.0f / node_distance_0;
            float folded_0 = bend_0 * 2.0f / node_distance_0;
            if(((as_type<uint>((normal_0))) & 2147483648U) != 0U)
            {
                bend_stress_0 = - folded_0;
            }
            else
            {
                bend_stress_0 = folded_0;
            }
            float normal_2 = normal_0 + bend_stress_0;
            bend_stress_0 = 0.0f;
            bearing_tension_0 = normal_2;
            normal_1 = shear_2;
        }
        float _S34 = bearing_tension_0;
        float _S35 = normal_1;
        bearing_tension_0 = -1.0f;
        normal_1 = _S34;
        shear_1 = _S35;
    }
    if(((&kernelContext_1)->params_0->fibres_0) == 0U)
    {
        _S17 = true;
    }
    else
    {
        if(gain_max_0 <= 0.0f)
        {
            _S17 = ((&kernelContext_1)->params_0->section_bending_0) == 0U;
        }
        else
        {
            _S17 = false;
        }
    }
    float compression_0;
    float tension_0;
    if(_S17)
    {
        if(((as_type<uint>((normal_1))) & 2147483648U) != 0U)
        {
            compression_0 = - bend_stress_0;
        }
        else
        {
            compression_0 = bend_stress_0;
        }
        float combined_0 = normal_1 + compression_0;
        if(combined_0 <= 0.0f)
        {
            compression_0 = - combined_0;
        }
        else
        {
            compression_0 = 0.0f;
        }
        if(combined_0 > 0.0f)
        {
            tension_0 = combined_0;
        }
        else
        {
            tension_0 = 0.0f;
        }
    }
    else
    {
        float t_0 = normal_1 + bend_stress_0;
        float c_0 = bend_stress_0 - normal_1;
        if(t_0 > 0.0f)
        {
            compression_0 = t_0;
        }
        else
        {
            compression_0 = 0.0f;
        }
        if(c_0 > 0.0f)
        {
            tension_0 = c_0;
        }
        else
        {
            tension_0 = 0.0f;
        }
        float _S36 = compression_0;
        compression_0 = tension_0;
        tension_0 = _S36;
    }
    if(bearing_tension_0 >= 0.0f)
    {
        tension_0 = bearing_tension_0;
    }
    *((&kernelContext_1)->out_0+(_S13 + 2U)) = packed_float4(float4(normal_1, shear_1, bend_stress_0, compression_0)) ;
    uint _S37 = (as_type<uint>((_S16.y))) * 3U;
    float4 _S38 = float4(*((&kernelContext_1)->materials_0+_S37)) ;
    float4 _S39 = float4(*((&kernelContext_1)->materials_0+(_S37 + 1U))) ;
    float4 _S40 = float4(*((&kernelContext_1)->materials_0+(_S37 + 2U))) ;
    float _S41 = _S40.x;
    if(_S41 > 0.0f)
    {
        _S17 = normal_1 < 0.0f;
    }
    else
    {
        _S17 = false;
    }
    float graded_shear_0;
    if(_S17)
    {
        float term_0 = - _S41 * normal_1;
        float _S42 = _S40.y;
        if(_S42 > 0.0f)
        {
            float room_0 = _S42 - _S39.y;
            if(room_0 > 0.0f)
            {
                if(term_0 < room_0)
                {
                    bend_stress_0 = term_0;
                }
                else
                {
                    bend_stress_0 = room_0;
                }
            }
            else
            {
                bend_stress_0 = 0.0f;
            }
            graded_shear_0 = bend_stress_0;
        }
        else
        {
            graded_shear_0 = term_0;
        }
        graded_shear_0 = max(0.0f, shear_1 - graded_shear_0);
    }
    else
    {
        graded_shear_0 = shear_1;
    }
    float _S43 = _S38.x;
    float axial_0;
    if(compression_0 > _S43)
    {
        axial_0 = excess_0(compression_0, _S43, _S38.y);
    }
    else
    {
        axial_0 = 0.0f;
    }
    float _S44 = _S38.z;
    if(tension_0 > _S44)
    {
        float t_1 = excess_0(tension_0, _S44, _S38.w);
        if(axial_0 < t_1)
        {
            bend_stress_0 = t_1;
        }
        else
        {
            bend_stress_0 = axial_0;
        }
        axial_0 = bend_stress_0;
    }
    float _S45 = _S39.x;
    float multiplier_0;
    if(graded_shear_0 > _S45)
    {
        multiplier_0 = axial_0 + excess_0(graded_shear_0, _S45, _S39.y);
    }
    else
    {
        multiplier_0 = axial_0;
    }
    bool command_0 = multiplier_0 > 0.0f;
    float damage_0;
    if(command_0)
    {
        if(multiplier_0 >= 1.0f)
        {
            damage_0 = b_1.health_0;
        }
        else
        {
            if(((&kernelContext_1)->params_0->dt_0) > 0.0f)
            {
                float fraction_0 = multiplier_0 * (&kernelContext_1)->params_0->dt_0 * (&kernelContext_1)->params_0->damage_rate_0;
                if(fraction_0 < 1.0f)
                {
                    bend_stress_0 = fraction_0;
                }
                else
                {
                    bend_stress_0 = 1.0f;
                }
                damage_0 = b_1.health_0 * bend_stress_0;
            }
            else
            {
                damage_0 = b_1.health_0 * multiplier_0;
            }
        }
        if(multiplier_0 < 1.0f)
        {
            _S17 = (_S39.z) > 0.0f;
        }
        else
        {
            _S17 = false;
        }
        if(_S17)
        {
            float floor_0 = _S16.x * _S39.z;
            if((b_1.health_0 - damage_0) < floor_0)
            {
                if((b_1.health_0) > floor_0)
                {
                    bend_stress_0 = b_1.health_0 - floor_0;
                }
                else
                {
                    bend_stress_0 = 0.0f;
                }
                damage_0 = bend_stress_0;
            }
        }
    }
    else
    {
        damage_0 = 0.0f;
    }
    packed_float4 device* _S46 = (&kernelContext_1)->out_0+(_S13 + 3U);
    if(command_0)
    {
        bend_stress_0 = 1.0f;
    }
    else
    {
        bend_stress_0 = 0.0f;
    }
    *_S46 = packed_float4(float4(tension_0, damage_0, multiplier_0, bend_stress_0)) ;
    if(((&kernelContext_1)->params_0->apply_damage_0) != 0U)
    {
        _S17 = command_0;
    }
    else
    {
        _S17 = false;
    }
    if(_S17)
    {
        float health_1 = b_1.health_0 - damage_0;
        bool _S47 = health_1 > 0.0f;
        if(_S47)
        {
            bend_stress_0 = health_1;
        }
        else
        {
            bend_stress_0 = 0.0f;
        }
        ((&kernelContext_1)->bonds_0+j_1)->health_0 = bend_stress_0;
        if(!_S47)
        {
            packed_float4 _S48 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_1)->x_0+_S9) = _S48;
            *((&kernelContext_1)->x_0+_S10) = _S48;
            atomic_store_explicit((&kernelContext_1)->topo_0+topo_dirty_0((&kernelContext_1)->params_0->node_count_0), 1U, memory_order_relaxed);
        }
    }
    return;
}

