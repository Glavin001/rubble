struct BlastParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) max_iterations_0 : u32,
    @align(8) warm_0 : u32,
    @align(4) skip_settled_0 : u32,
    @align(16) tolerance_0 : f32,
    @align(4) recip_length_0 : f32,
    @align(8) node_count_0 : u32,
    @align(4) bond_count_0 : u32,
    @align(16) linear_scale_0 : f32,
    @align(4) angular_scale_0 : f32,
    @align(8) bend_gain_max_0 : f32,
    @align(4) fibres_0 : u32,
    @align(16) dt_0 : f32,
    @align(4) damage_rate_0 : f32,
    @align(8) apply_damage_0 : u32,
    @align(4) force_tolerance_0 : f32,
    @align(16) gravity_0 : vec4<f32>,
    @align(16) w_alive_0 : u32,
    @align(4) a_dirty_0 : u32,
    @align(8) a_counters_0 : u32,
    @align(4) wide_min_0 : u32,
    @align(16) wide_rows_0 : u32,
    @align(4) wide_blocks_0 : u32,
    @align(8) opaque_zero_0 : u32,
    @align(4) first_polynomial_0 : u32,
    @align(16) rotation_0 : u32,
    @align(4) node_vectors_0 : u32,
    @align(8) carry_0 : u32,
    @align(4) section_bending_0 : u32,
    @align(16) chunk_loads_0 : u32,
    @align(4) r_virial_0 : u32,
    @align(8) a_events_0 : u32,
    @align(4) ev_broken_cap_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : BlastParams_std140_0;
struct BlastBond_std430_0
{
    @align(16) n0_0 : u32,
    @align(4) n1_0 : u32,
    @align(8) s_0 : f32,
    @align(4) health_0 : f32,
    @align(16) o0_0 : vec4<f32>,
    @align(16) o1_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read_write> bonds_0 : array<BlastBond_std430_0>;

@binding(2) @group(0) var<storage, read_write> x_0 : array<vec4<f32>>;

@binding(7) @group(0) var<storage, read> stiffness_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read_write> out_0 : array<vec4<f32>>;

@binding(3) @group(0) var<storage, read> geometry_0 : array<vec4<f32>>;

@binding(8) @group(0) var<storage, read> sections_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> materials_0 : array<vec4<f32>>;

@binding(6) @group(0) var<storage, read_write> topo_0 : array<atomic<u32>>;

fn stiff_0( j_0 : u32,  which_0 : u32,  v_0 : vec3<f32>) -> vec3<f32>
{
    var _S1 : u32 = u32(6) * which_0;
    var m_0 : array<f32, i32(6)>;
    var k_0 : u32 = u32(0);
    loop
    {
        if(k_0 < u32(6))
        {
        }
        else
        {
            break;
        }
        var _S2 : u32 = _S1 + k_0;
        m_0[k_0] = stiffness_0[j_0 * u32(6) + _S2 / u32(4)][_S2 % u32(4)];
        k_0 = k_0 + u32(1);
    }
    var _S3 : f32 = v_0.x;
    var _S4 : f32 = v_0.y;
    var _S5 : f32 = v_0.z;
    return vec3<f32>(m_0[i32(0)] * _S3 + m_0[i32(3)] * _S4 + m_0[i32(4)] * _S5, m_0[i32(3)] * _S3 + m_0[i32(1)] * _S4 + m_0[i32(5)] * _S5, m_0[i32(4)] * _S3 + m_0[i32(5)] * _S4 + m_0[i32(2)] * _S5);
}

fn fdot_0( a_0 : vec3<f32>,  b_0 : vec3<f32>) -> f32
{
    return fma(a_0.z, b_0.z, fma(a_0.y, b_0.y, a_0.x * b_0.x));
}

fn perpendicular_0( v_1 : vec3<f32>,  n_0 : vec3<f32>,  along_0 : f32) -> f32
{
    var _S6 : f32 = - along_0;
    var px_0 : f32 = fma(_S6, n_0.x, v_1.x);
    var py_0 : f32 = fma(_S6, n_0.y, v_1.y);
    var pz_0 : f32 = fma(_S6, n_0.z, v_1.z);
    return sqrt(fma(pz_0, pz_0, fma(py_0, py_0, px_0 * px_0)));
}

fn excess_0( value_0 : f32,  elastic_0 : f32,  fatal_0 : f32) -> f32
{
    var d_0 : f32 = fatal_0 - elastic_0;
    var _S7 : f32 = value_0 - elastic_0;
    var _S8 : f32;
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

fn topo_total_0( node_count_1 : u32) -> u32
{
    return u32(10) * node_count_1 + u32(3) * ((node_count_1 + u32(1023)) / u32(1024));
}

fn topo_dirty_0( node_count_2 : u32) -> u32
{
    return topo_total_0(node_count_2) + u32(4);
}

@compute
@workgroup_size(64, 1, 1)
fn blast_bond_stress(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var j_1 : u32 = id_0.x;
    if(j_1 >= (params_0.bond_count_0))
    {
        return;
    }
    var b_1 : BlastBond_std430_0 = bonds_0[j_1];
    var _S9 : u32 = j_1 * u32(2);
    var xa_0 : vec3<f32> = x_0[_S9].xyz;
    var _S10 : u32 = _S9 + u32(1);
    var xl_0 : vec3<f32> = x_0[_S10].xyz;
    var xa_1 : vec3<f32>;
    if((params_0.rotation_0) != u32(0))
    {
        xa_1 = stiff_0(j_1, u32(0), xa_0);
    }
    else
    {
        xa_1 = xa_0;
    }
    var xl_1 : vec3<f32>;
    if((params_0.rotation_0) == u32(2))
    {
        xl_1 = stiff_0(j_1, u32(2), xl_0);
    }
    else
    {
        xl_1 = xl_0;
    }
    var ia_0 : vec3<f32> = xa_1 * vec3<f32>((b_1.s_0 * params_0.angular_scale_0));
    var il_0 : vec3<f32> = xl_1 * vec3<f32>((b_1.s_0 * params_0.linear_scale_0));
    var _S11 : u32 = j_1 * u32(4);
    out_0[_S11] = vec4<f32>(ia_0, 0.0f);
    out_0[_S11 + u32(1)] = vec4<f32>(il_0, 0.0f);
    if(!((b_1.health_0) > 0.0f))
    {
        var _S12 : vec4<f32> = vec4<f32>(0.0f);
        out_0[_S11 + u32(2)] = _S12;
        out_0[_S11 + u32(3)] = _S12;
        return;
    }
    var g0_0 : vec4<f32> = geometry_0[_S9];
    var g1_0 : vec4<f32> = geometry_0[_S10];
    var n_1 : vec3<f32> = g0_0.xyz;
    var node_distance_0 : f32 = g0_0.w;
    var linear_normal_0 : f32 = fdot_0(il_0, n_1);
    var normal_0 : f32 = linear_normal_0 / b_1.health_0;
    var shear_0 : f32 = perpendicular_0(il_0, n_1, linear_normal_0) / b_1.health_0;
    var angular_normal_0 : f32 = fdot_0(ia_0, n_1);
    var twist_0 : f32 = abs(angular_normal_0) / b_1.health_0;
    var bend_0 : f32 = perpendicular_0(ia_0, n_1, angular_normal_0) / b_1.health_0;
    var gain_max_0 : f32 = params_0.bend_gain_max_0;
    var bend_stress_0 : f32;
    var bearing_tension_0 : f32;
    var normal_1 : f32;
    var shear_1 : f32;
    var _S13 : bool;
    if((params_0.section_bending_0) != u32(0))
    {
        var _S14 : u32 = j_1 * u32(3);
        var s0_0 : vec4<f32> = sections_0[_S14];
        var s1_0 : vec4<f32> = sections_0[_S14 + u32(1)];
        var m_1 : vec3<f32> = ia_0 + cross(sections_0[_S14 + u32(2)].xyz, il_0);
        var m_normal_0 : f32 = fdot_0(m_1, n_1);
        var tw_0 : f32 = abs(m_normal_0);
        var axis_0 : vec3<f32> = s0_0.xyz;
        var _S15 : f32 = n_1.y;
        var _S16 : f32 = axis_0.z;
        var _S17 : f32 = n_1.z;
        var _S18 : f32 = axis_0.y;
        var _S19 : f32 = axis_0.x;
        var _S20 : f32 = n_1.x;
        var axis1_0 : vec3<f32> = vec3<f32>(fma(_S15, _S16, - _S17 * _S18), fma(_S17, _S19, - _S20 * _S16), fma(_S20, _S18, - _S15 * _S19));
        var original_0 : f32 = g1_0.x;
        var _S21 : f32 = s0_0.w;
        if(_S21 > 0.0f)
        {
            _S13 = (s1_0.x) > 0.0f;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
            _S13 = (s1_0.y) > 0.0f;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
            _S13 = original_0 > 0.0f;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
            var live_0 : f32 = b_1.health_0 / original_0;
            var _S22 : f32 = fma(tw_0, 1.0f / (s1_0.y * live_0), shear_0);
            bend_stress_0 = (abs(fdot_0(m_1, axis_0)) / _S21 + abs(fdot_0(m_1, axis1_0)) / s1_0.x) / live_0;
            normal_1 = _S22;
        }
        else
        {
            var cube_0 : f32 = b_1.health_0 * sqrt(b_1.health_0);
            var _S23 : f32 = fma(tw_0, 4.24264049530029297f / cube_0, shear_0);
            bend_stress_0 = 6.0f * perpendicular_0(m_1, n_1, m_normal_0) / cube_0;
            normal_1 = _S23;
        }
        var _S24 : f32 = s1_0.z;
        if(_S24 > 0.0f)
        {
            _S13 = (s1_0.w) > 0.0f;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
            bearing_tension_0 = max(0.0f, abs(dot(m_1, axis_0)) / _S24 + abs(dot(m_1, axis1_0)) / s1_0.w + normal_0 * b_1.health_0) / b_1.health_0;
        }
        else
        {
            bearing_tension_0 = -1.0f;
        }
        var _S25 : f32 = normal_1;
        normal_1 = normal_0;
        shear_1 = _S25;
    }
    else
    {
        if(gain_max_0 > 0.0f)
        {
            var _S26 : bool = (b_1.health_0) > 9.99999997475242708e-07f;
            if(_S26)
            {
                bend_stress_0 = b_1.health_0;
            }
            else
            {
                bend_stress_0 = 9.99999997475242708e-07f;
            }
            var torsion_0 : f32 = 4.80999994277954102f / sqrt(bend_stress_0);
            if(torsion_0 < gain_max_0)
            {
                bend_stress_0 = torsion_0;
            }
            else
            {
                bend_stress_0 = gain_max_0;
            }
            var _S27 : f32 = fma(twist_0, bend_stress_0, shear_0);
            if(_S26)
            {
                bend_stress_0 = b_1.health_0;
            }
            else
            {
                bend_stress_0 = 9.99999997475242708e-07f;
            }
            var section_0 : f32 = 6.0f / sqrt(bend_stress_0);
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
            normal_1 = _S27;
        }
        else
        {
            var shear_2 : f32 = shear_0 + twist_0 * 2.0f / node_distance_0;
            var folded_0 : f32 = bend_0 * 2.0f / node_distance_0;
            if((((bitcast<u32>((normal_0))) & (u32(2147483648)))) != u32(0))
            {
                bend_stress_0 = - folded_0;
            }
            else
            {
                bend_stress_0 = folded_0;
            }
            var normal_2 : f32 = normal_0 + bend_stress_0;
            bend_stress_0 = 0.0f;
            bearing_tension_0 = normal_2;
            normal_1 = shear_2;
        }
        var _S28 : f32 = bearing_tension_0;
        var _S29 : f32 = normal_1;
        bearing_tension_0 = -1.0f;
        normal_1 = _S28;
        shear_1 = _S29;
    }
    if((params_0.fibres_0) == u32(0))
    {
        _S13 = true;
    }
    else
    {
        if(gain_max_0 <= 0.0f)
        {
            _S13 = (params_0.section_bending_0) == u32(0);
        }
        else
        {
            _S13 = false;
        }
    }
    var compression_0 : f32;
    var tension_0 : f32;
    if(_S13)
    {
        if((((bitcast<u32>((normal_1))) & (u32(2147483648)))) != u32(0))
        {
            compression_0 = - bend_stress_0;
        }
        else
        {
            compression_0 = bend_stress_0;
        }
        var combined_0 : f32 = normal_1 + compression_0;
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
        var t_0 : f32 = normal_1 + bend_stress_0;
        var c_0 : f32 = bend_stress_0 - normal_1;
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
        var _S30 : f32 = compression_0;
        compression_0 = tension_0;
        tension_0 = _S30;
    }
    if(bearing_tension_0 >= 0.0f)
    {
        tension_0 = bearing_tension_0;
    }
    else
    {
    }
    out_0[_S11 + u32(2)] = vec4<f32>(normal_1, shear_1, bend_stress_0, compression_0);
    var _S31 : u32 = (bitcast<u32>((g1_0.y))) * u32(3);
    var m0_0 : vec4<f32> = materials_0[_S31];
    var m1_0 : vec4<f32> = materials_0[_S31 + u32(1)];
    var m2_0 : vec4<f32> = materials_0[_S31 + u32(2)];
    var _S32 : f32 = m2_0.x;
    if(_S32 > 0.0f)
    {
        _S13 = normal_1 < 0.0f;
    }
    else
    {
        _S13 = false;
    }
    var graded_shear_0 : f32;
    if(_S13)
    {
        var term_0 : f32 = - _S32 * normal_1;
        var _S33 : f32 = m2_0.y;
        if(_S33 > 0.0f)
        {
            var room_0 : f32 = _S33 - m1_0.y;
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
    var _S34 : f32 = m0_0.x;
    var axial_0 : f32;
    if(compression_0 > _S34)
    {
        axial_0 = excess_0(compression_0, _S34, m0_0.y);
    }
    else
    {
        axial_0 = 0.0f;
    }
    var _S35 : f32 = m0_0.z;
    if(tension_0 > _S35)
    {
        var t_1 : f32 = excess_0(tension_0, _S35, m0_0.w);
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
    else
    {
    }
    var _S36 : f32 = m1_0.x;
    var multiplier_0 : f32;
    if(graded_shear_0 > _S36)
    {
        multiplier_0 = axial_0 + excess_0(graded_shear_0, _S36, m1_0.y);
    }
    else
    {
        multiplier_0 = axial_0;
    }
    var command_0 : bool = multiplier_0 > 0.0f;
    var damage_0 : f32;
    if(command_0)
    {
        if(multiplier_0 >= 1.0f)
        {
            damage_0 = b_1.health_0;
        }
        else
        {
            if((params_0.dt_0) > 0.0f)
            {
                var fraction_0 : f32 = multiplier_0 * params_0.dt_0 * params_0.damage_rate_0;
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
            _S13 = (m1_0.z) > 0.0f;
        }
        else
        {
            _S13 = false;
        }
        if(_S13)
        {
            var floor_0 : f32 = g1_0.x * m1_0.z;
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
            else
            {
            }
        }
        else
        {
        }
    }
    else
    {
        damage_0 = 0.0f;
    }
    if(command_0)
    {
        bend_stress_0 = 1.0f;
    }
    else
    {
        bend_stress_0 = 0.0f;
    }
    out_0[_S11 + u32(3)] = vec4<f32>(tension_0, damage_0, multiplier_0, bend_stress_0);
    if((params_0.apply_damage_0) != u32(0))
    {
        _S13 = command_0;
    }
    else
    {
        _S13 = false;
    }
    if(_S13)
    {
        var health_1 : f32 = b_1.health_0 - damage_0;
        var _S37 : bool = health_1 > 0.0f;
        if(_S37)
        {
            bend_stress_0 = health_1;
        }
        else
        {
            bend_stress_0 = 0.0f;
        }
        bonds_0[j_1].health_0 = bend_stress_0;
        if(!_S37)
        {
            var _S38 : vec4<f32> = vec4<f32>(0.0f);
            x_0[_S9] = _S38;
            x_0[_S10] = _S38;
            atomicStore(&(topo_0[topo_dirty_0(params_0.node_count_0)]), u32(1));
        }
    }
    return;
}

