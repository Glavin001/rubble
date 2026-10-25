#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float fdot_0(float3 a_0, float3 b_0)
{
    return fma(a_0.z, b_0.z, fma(a_0.y, b_0.y, a_0.x * b_0.x));
}

float perpendicular_0(float3 v_0, float3 n_0, float along_0)
{
    float _S1 = - along_0;
    float px_0 = fma(_S1, n_0.x, v_0.x);
    float py_0 = fma(_S1, n_0.y, v_0.y);
    float pz_0 = fma(_S1, n_0.z, v_0.z);
    return sqrt(fma(pz_0, pz_0, fma(py_0, py_0, px_0 * px_0)));
}

float excess_0(float value_0, float elastic_0, float fatal_0)
{
    float d_0 = fatal_0 - elastic_0;
    float _S2 = value_0 - elastic_0;
    float _S3;
    if(d_0 > 0.0f)
    {
        _S3 = d_0;
    }
    else
    {
        _S3 = 1.0f;
    }
    return _S2 / _S3;
}

uint topo_total_0(uint node_count_0)
{
    return 10U * node_count_0 + 3U * ((node_count_0 + 1023U) / 1024U);
}

uint topo_dirty_0(uint node_count_1)
{
    return topo_total_0(node_count_1) + 4U;
}

struct BlastParams_0
{
    uint island_count_0;
    uint max_iterations_0;
    uint warm_0;
    uint skip_settled_0;
    float tolerance_0;
    float recip_length_0;
    uint node_count_2;
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
    uint pad1_0;
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
    packed_float4 device* out_0;
    packed_float4 device* geometry_0;
    packed_float4 device* materials_0;
    atomic<uint> device* topo_0;
};

[[kernel]] void blast_bond_stress(uint3 id_0 [[thread_position_in_grid]], BlastParams_0 constant* params_1 [[buffer(0)]], BlastBond_natural_0 device* bonds_1 [[buffer(1)]], packed_float4 device* x_1 [[buffer(2)]], packed_float4 device* out_1 [[buffer(5)]], packed_float4 device* geometry_1 [[buffer(3)]], packed_float4 device* materials_1 [[buffer(4)]], atomic<uint> device* topo_1 [[buffer(6)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->bonds_0 = bonds_1;
    (&kernelContext_0)->x_0 = x_1;
    (&kernelContext_0)->out_0 = out_1;
    (&kernelContext_0)->geometry_0 = geometry_1;
    (&kernelContext_0)->materials_0 = materials_1;
    (&kernelContext_0)->topo_0 = topo_1;
    uint j_0 = id_0.x;
    if(j_0 >= (params_1->bond_count_0))
    {
        return;
    }
    BlastBond_natural_0 device* _S4 = (&kernelContext_0)->bonds_0+j_0;
    BlastBond_natural_0 b_1 = *_S4;
    uint _S5 = j_0 * 2U;
    float3 ia_0 = (float4(*((&kernelContext_0)->x_0+_S5)) ).xyz * float3(((*_S4).s_0 * (&kernelContext_0)->params_0->angular_scale_0)) ;
    uint _S6 = _S5 + 1U;
    float3 il_0 = (float4(*((&kernelContext_0)->x_0+_S6)) ).xyz * float3(((*_S4).s_0 * (&kernelContext_0)->params_0->linear_scale_0)) ;
    uint _S7 = j_0 * 4U;
    *((&kernelContext_0)->out_0+_S7) = packed_float4(float4(ia_0, 0.0f)) ;
    *((&kernelContext_0)->out_0+(_S7 + 1U)) = packed_float4(float4(il_0, 0.0f)) ;
    if(!((b_1.health_0) > 0.0f))
    {
        packed_float4 _S8 = packed_float4(float4(0.0f) ) ;
        *((&kernelContext_0)->out_0+(_S7 + 2U)) = _S8;
        *((&kernelContext_0)->out_0+(_S7 + 3U)) = _S8;
        return;
    }
    float4 _S9 = float4(*((&kernelContext_0)->geometry_0+_S5)) ;
    float4 _S10 = float4(*((&kernelContext_0)->geometry_0+_S6)) ;
    float3 n_1 = _S9.xyz;
    float node_distance_0 = _S9.w;
    float linear_normal_0 = fdot_0(il_0, n_1);
    float normal_0 = linear_normal_0 / b_1.health_0;
    float shear_0 = perpendicular_0(il_0, n_1, linear_normal_0) / b_1.health_0;
    float angular_normal_0 = fdot_0(ia_0, n_1);
    float twist_0 = abs(angular_normal_0) / b_1.health_0;
    float bend_0 = perpendicular_0(ia_0, n_1, angular_normal_0) / b_1.health_0;
    float gain_max_0 = (&kernelContext_0)->params_0->bend_gain_max_0;
    float bend_stress_0;
    float normal_1;
    float shear_1;
    if(((&kernelContext_0)->params_0->bend_gain_max_0) > 0.0f)
    {
        bool _S11 = (b_1.health_0) > 9.99999997475242708e-07f;
        if(_S11)
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
        float _S12 = fma(twist_0, bend_stress_0, shear_0);
        if(_S11)
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
        normal_1 = normal_0;
        shear_1 = _S12;
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
        normal_1 = normal_2;
        shear_1 = shear_2;
    }
    bool _S13;
    if(((&kernelContext_0)->params_0->fibres_0) == 0U)
    {
        _S13 = true;
    }
    else
    {
        _S13 = gain_max_0 <= 0.0f;
    }
    float compression_0;
    float tension_0;
    if(_S13)
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
        float _S14 = compression_0;
        compression_0 = tension_0;
        tension_0 = _S14;
    }
    *((&kernelContext_0)->out_0+(_S7 + 2U)) = packed_float4(float4(normal_1, shear_1, bend_stress_0, compression_0)) ;
    uint _S15 = (as_type<uint>((_S10.y))) * 2U;
    float4 _S16 = float4(*((&kernelContext_0)->materials_0+_S15)) ;
    float4 _S17 = float4(*((&kernelContext_0)->materials_0+(_S15 + 1U))) ;
    float _S18 = _S16.x;
    float axial_0;
    if(compression_0 > _S18)
    {
        axial_0 = excess_0(compression_0, _S18, _S16.y);
    }
    else
    {
        axial_0 = 0.0f;
    }
    float _S19 = _S16.z;
    if(tension_0 > _S19)
    {
        float t_1 = excess_0(tension_0, _S19, _S16.w);
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
    float _S20 = _S17.x;
    float multiplier_0;
    if(shear_1 > _S20)
    {
        multiplier_0 = axial_0 + excess_0(shear_1, _S20, _S17.y);
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
            if(((&kernelContext_0)->params_0->dt_0) > 0.0f)
            {
                float fraction_0 = multiplier_0 * (&kernelContext_0)->params_0->dt_0 * (&kernelContext_0)->params_0->damage_rate_0;
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
            _S13 = (_S17.z) > 0.0f;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
            float floor_0 = _S10.x * _S17.z;
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
    packed_float4 device* _S21 = (&kernelContext_0)->out_0+(_S7 + 3U);
    if(command_0)
    {
        bend_stress_0 = 1.0f;
    }
    else
    {
        bend_stress_0 = 0.0f;
    }
    *_S21 = packed_float4(float4(tension_0, damage_0, multiplier_0, bend_stress_0)) ;
    if(((&kernelContext_0)->params_0->apply_damage_0) != 0U)
    {
        _S13 = command_0;
    }
    else
    {
        _S13 = false;
    }
    if(_S13)
    {
        float health_1 = b_1.health_0 - damage_0;
        bool _S22 = health_1 > 0.0f;
        if(_S22)
        {
            bend_stress_0 = health_1;
        }
        else
        {
            bend_stress_0 = 0.0f;
        }
        ((&kernelContext_0)->bonds_0+j_0)->health_0 = bend_stress_0;
        if(!_S22)
        {
            packed_float4 _S23 = packed_float4(float4(0.0f) ) ;
            *((&kernelContext_0)->x_0+_S5) = _S23;
            *((&kernelContext_0)->x_0+_S6) = _S23;
            atomic_store_explicit((&kernelContext_0)->topo_0+topo_dirty_0((&kernelContext_0)->params_0->node_count_2), 1U, memory_order_relaxed);
        }
    }
    return;
}

