struct TestParams_std140_0
{
    @align(16) count_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) pad1_0 : u32,
    @align(4) pad2_0 : u32,
};

@binding(0) @group(0) var<uniform> params_0 : TestParams_std140_0;
struct JointBond_std430_0
{
    @align(16) geom0_0 : vec4<f32>,
    @align(16) geom1_0 : vec4<f32>,
    @align(16) stiff0_0 : vec4<f32>,
    @align(16) stiff1_0 : vec4<f32>,
    @align(16) rebar0_0 : vec4<f32>,
    @align(16) rebar1_0 : vec4<f32>,
    @align(16) ids_0 : vec4<u32>,
};

@binding(2) @group(0) var<storage, read> bonds_0 : array<JointBond_std430_0>;

struct JointMaterial_std430_0
{
    @align(16) strength_0 : vec4<f32>,
    @align(16) energy_0 : vec4<f32>,
    @align(16) dif_0 : vec4<f32>,
    @align(16) misc_0 : vec4<f32>,
    @align(16) kind_flags_0 : vec4<u32>,
};

@binding(1) @group(0) var<storage, read> materials_0 : array<JointMaterial_std430_0>;

@binding(4) @group(0) var<storage, read> inputs_0 : array<vec4<f32>>;

struct JointState_std430_0
{
    @align(4) damage_0 : f32,
    @align(4) crush_0 : f32,
    @align(4) kappa_0 : f32,
    @align(4) kappa_c_0 : f32,
    @align(4) ductility_0 : f32,
    @align(4) ductility_c_0 : f32,
    @align(4) life_0 : f32,
    @align(4) plastic_x_0 : f32,
    @align(4) plastic_y_0 : f32,
    @align(4) plastic_t_0 : f32,
    @align(4) rebar_plastic_0 : f32,
    @align(4) rebar_slip0_0 : f32,
    @align(4) rebar_slip1_0 : f32,
    @align(4) rebar_work_0 : f32,
    @align(4) rebar_broken_0 : f32,
    @align(4) strain_rate_0 : f32,
    @align(4) governing_stress_0 : f32,
    @align(4) dissipated_0 : f32,
    @align(4) utilization_0 : f32,
    @align(4) mode_0 : u32,
};

@binding(3) @group(0) var<storage, read> states_in_0 : array<JointState_std430_0>;

@binding(5) @group(0) var<storage, read_write> states_out_0 : array<JointState_std430_0>;

@binding(6) @group(0) var<storage, read_write> outputs_0 : array<vec4<f32>>;

var<private> SPRING_AT_0 : array<f32, i32(6)> = array<f32, i32(6)>( -0.4166666567325592f, -0.25f, -0.0833333358168602f, 0.0833333358168602f, 0.25f, 0.4166666567325592f );
fn connected_0( st_0 : ptr<function, JointState_std430_0>,  has_rebar_0 : bool) -> bool
{
    var _S1 : bool;
    if(((*st_0).damage_0) < 1.0f)
    {
        _S1 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S1 = ((*st_0).rebar_broken_0) == 0.0f;
        }
        else
        {
            _S1 = false;
        }
    }
    return _S1;
}

struct JointState_0
{
     damage_0 : f32,
     crush_0 : f32,
     kappa_0 : f32,
     kappa_c_0 : f32,
     ductility_0 : f32,
     ductility_c_0 : f32,
     life_0 : f32,
     plastic_x_0 : f32,
     plastic_y_0 : f32,
     plastic_t_0 : f32,
     rebar_plastic_0 : f32,
     rebar_slip0_0 : f32,
     rebar_slip1_0 : f32,
     rebar_work_0 : f32,
     rebar_broken_0 : f32,
     strain_rate_0 : f32,
     governing_stress_0 : f32,
     dissipated_0 : f32,
     utilization_0 : f32,
     mode_0 : u32,
};

fn connected_1( st_1 : JointState_0,  has_rebar_1 : bool) -> bool
{
    var _S2 : bool;
    if((st_1.damage_0) < 1.0f)
    {
        _S2 = true;
    }
    else
    {
        if(has_rebar_1)
        {
            _S2 = (st_1.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S2 = false;
        }
    }
    return _S2;
}

fn fdiv_0( a_0 : f32,  b_0 : f32) -> f32
{
    return a_0 / b_0;
}

fn fsqrt_0( a_1 : f32) -> f32
{
    return sqrt(a_1);
}

struct Measures_0
{
     tension_0 : f32,
     shear_0 : f32,
     normal_compression_0 : f32,
     compression_0 : f32,
     compressive_force_0 : f32,
};

fn stress_measures_0( b_1 : ptr<function, JointBond_std430_0>,  q_lin_0 : vec3<f32>,  q_ang_0 : vec3<f32>) -> Measures_0
{
    var area_0 : f32 = (*b_1).geom0_0.x;
    var _S3 : f32 = q_lin_0.z;
    var axial_0 : f32 = fdiv_0(_S3, area_0);
    var bending_0 : f32 = fdiv_0(abs(q_ang_0.x), (*b_1).geom1_0.x) + fdiv_0(abs(q_ang_0.y), (*b_1).geom1_0.y);
    var _S4 : f32 = q_lin_0.x;
    var _S5 : f32 = q_lin_0.y;
    var shear_1 : f32 = fdiv_0(fsqrt_0(_S4 * _S4 + _S5 * _S5), area_0) + fdiv_0(abs(q_ang_0.z), (*b_1).geom0_0.w);
    var m_0 : Measures_0;
    m_0.tension_0 = axial_0 + bending_0;
    m_0.shear_0 = shear_1;
    var _S6 : f32 = - axial_0;
    m_0.normal_compression_0 = max(_S6, 0.0f);
    m_0.compression_0 = _S6 + bending_0;
    m_0.compressive_force_0 = max(- _S3, 0.0f);
    return m_0;
}

fn expm1_accurate_0( x_0 : f32) -> f32
{
    if((abs(x_0)) < 0.00100000004749745f)
    {
        return x_0 * (1.0f + x_0 * (0.5f + x_0 * 0.1666666716337204f));
    }
    return exp(x_0) - 1.0f;
}

fn fpow_0( a_2 : f32,  b_2 : f32) -> f32
{
    return pow(a_2, b_2);
}

fn dif_factor_0( mat_0 : ptr<function, JointMaterial_std430_0>,  strain_rate_1 : f32) -> f32
{
    var r_0 : f32 = abs(strain_rate_1);
    var _S7 : vec4<f32> = (*mat_0).dif_0;
    var ref_0 : f32 = (*mat_0).dif_0.x;
    if(r_0 <= ref_0)
    {
        return 1.0f;
    }
    var _S8 : f32 = _S7.z;
    var f_0 : f32;
    if(r_0 <= _S8)
    {
        f_0 = fpow_0(fdiv_0(r_0, ref_0), _S7.y);
    }
    else
    {
        f_0 = fpow_0(fdiv_0(_S8, ref_0), _S7.y) * fpow_0(fdiv_0(r_0, _S8), _S7.w);
    }
    return clamp(f_0, 1.0f, (*mat_0).misc_0.x);
}

fn fatigue_factor_0( mat_1 : ptr<function, JointMaterial_std430_0>,  life_1 : f32) -> f32
{
    if(((((*mat_1).kind_flags_0.y) & (u32(64)))) == u32(0))
    {
        return 1.0f;
    }
    return fpow_0(clamp(life_1, 0.0f, 1.0f), fdiv_0(1.0f, (*mat_1).misc_0.y - 2.0f));
}

fn infinity_0() -> f32
{
    return (bitcast<f32>((u32(2139095040))));
}

fn failure_indices_0( mat_2 : ptr<function, JointMaterial_std430_0>,  b_3 : ptr<function, JointBond_std430_0>,  m_1 : Measures_0,  multiplier_0 : f32) -> vec4<f32>
{
    var fc_0 : f32 = (*mat_2).strength_0.y * multiplier_0;
    var _S9 : f32 = min((*mat_2).strength_0.z * multiplier_0 + (*mat_2).strength_0.w * m_1.normal_compression_0, (*mat_2).energy_0.x * multiplier_0);
    var idx_0 : vec4<f32>;
    idx_0[i32(0)] = max(fdiv_0(m_1.tension_0, (*mat_2).strength_0.x * multiplier_0), 0.0f);
    var _S10 : f32;
    if(_S9 > 0.0f)
    {
        _S10 = fdiv_0(m_1.shear_0, _S9);
    }
    else
    {
        _S10 = infinity_0();
    }
    idx_0[i32(1)] = _S10;
    idx_0[i32(2)] = max(fdiv_0(m_1.compression_0, fc_0), 0.0f);
    var _S11 : f32 = (*b_3).stiff1_0.y;
    if(_S11 > 0.0f)
    {
        _S10 = fdiv_0(m_1.compressive_force_0, _S11);
    }
    else
    {
        _S10 = 0.0f;
    }
    idx_0[i32(3)] = _S10;
    return idx_0;
}

fn sq_0( x_1 : f32) -> f32
{
    return x_1 * x_1;
}

fn compressed_region_0( w0_0 : f32,  w1_0 : f32,  dz_0 : f32,  ax_0 : f32,  ay_0 : f32,  region_0 : ptr<function, array<f32, i32(6)>>)
{
    var count_1 : u32;
    var h0_0 : f32 = 0.5f * w0_0;
    var h1_0 : f32 = 0.5f * w1_0;
    var _S12 : f32 = - h0_0;
    var _S13 : f32 = - h1_0;
    var _S14 : array<vec2<f32>, i32(4)> = array<vec2<f32>, i32(4)>( vec2<f32>(_S12, _S13), vec2<f32>(h0_0, _S13), vec2<f32>(h0_0, h1_0), vec2<f32>(_S12, h1_0) );
    var poly_0 : array<vec2<f32>, i32(8)>;
    var i_0 : u32 = u32(0);
    var count_2 : u32 = u32(0);
    loop
    {
        if(i_0 < u32(4))
        {
        }
        else
        {
            break;
        }
        var _S15 : u32 = i_0;
        var _S16 : u32 = i_0 + u32(1);
        var _S17 : u32 = _S16 % u32(4);
        var _S18 : f32 = _S14[i_0].y;
        var _S19 : f32 = _S14[i_0].x;
        var fp_0 : f32 = dz_0 + ax_0 * _S18 - ay_0 * _S19;
        var _S20 : f32 = _S14[_S17].y;
        var _S21 : f32 = _S14[_S17].x;
        var fq_0 : f32 = dz_0 + ax_0 * _S20 - ay_0 * _S21;
        var _S22 : bool = fp_0 < 0.0f;
        if(_S22)
        {
            var _S23 : u32 = count_2 + u32(1);
            poly_0[count_2] = _S14[_S15];
            count_1 = _S23;
        }
        else
        {
            count_1 = count_2;
        }
        if(_S22 != (fq_0 < 0.0f))
        {
            var t_0 : f32 = fp_0 / (fp_0 - fq_0);
            var _S24 : u32 = count_1 + u32(1);
            poly_0[count_1] = vec2<f32>(_S19 + t_0 * (_S21 - _S19), _S18 + t_0 * (_S20 - _S18));
            count_2 = _S24;
        }
        else
        {
            count_2 = count_1;
        }
        i_0 = _S16;
    }
    count_1 = u32(0);
    loop
    {
        if(count_1 < u32(6))
        {
        }
        else
        {
            break;
        }
        (*region_0)[count_1] = 0.0f;
        count_1 = count_1 + u32(1);
    }
    if(count_2 < u32(3))
    {
        return;
    }
    var o_0 : vec2<f32> = poly_0[i32(0)];
    i_0 = u32(0);
    var a_3 : f32 = 0.0f;
    var sx_0 : f32 = 0.0f;
    var sy_0 : f32 = 0.0f;
    var ixx_0 : f32 = 0.0f;
    var iyy_0 : f32 = 0.0f;
    var ixy_0 : f32 = 0.0f;
    loop
    {
        if(i_0 < count_2)
        {
        }
        else
        {
            break;
        }
        var _S25 : f32 = o_0.x;
        var x0_0 : f32 = poly_0[i_0].x - _S25;
        var _S26 : f32 = o_0.y;
        var y0_0 : f32 = poly_0[i_0].y - _S26;
        var _S27 : u32 = i_0 + u32(1);
        var _S28 : u32 = _S27 % count_2;
        var x1_0 : f32 = poly_0[_S28].x - _S25;
        var y1_0 : f32 = poly_0[_S28].y - _S26;
        var _S29 : f32 = x0_0 * y1_0;
        var _S30 : f32 = x1_0 * y0_0;
        var cr_0 : f32 = _S29 - _S30;
        var a_4 : f32 = a_3 + cr_0 / 2.0f;
        var sx_1 : f32 = sx_0 + (x0_0 + x1_0) * cr_0 / 6.0f;
        var sy_1 : f32 = sy_0 + (y0_0 + y1_0) * cr_0 / 6.0f;
        var ixx_1 : f32 = ixx_0 + (x0_0 * x0_0 + x0_0 * x1_0 + x1_0 * x1_0) * cr_0 / 12.0f;
        var iyy_1 : f32 = iyy_0 + (y0_0 * y0_0 + y0_0 * y1_0 + y1_0 * y1_0) * cr_0 / 12.0f;
        var ixy_1 : f32 = ixy_0 + (_S29 + 2.0f * x0_0 * y0_0 + 2.0f * x1_0 * y1_0 + _S30) * cr_0 / 24.0f;
        i_0 = _S27;
        a_3 = a_4;
        sx_0 = sx_1;
        sy_0 = sy_1;
        ixx_0 = ixx_1;
        iyy_0 = iyy_1;
        ixy_0 = ixy_1;
    }
    if(a_3 <= 0.0f)
    {
        return;
    }
    var cx_0 : f32 = sx_0 / a_3;
    var cy_0 : f32 = sy_0 / a_3;
    (*region_0)[i32(0)] = a_3;
    (*region_0)[i32(1)] = o_0.x + cx_0;
    (*region_0)[i32(2)] = o_0.y + cy_0;
    var _S31 : f32 = a_3 * cx_0;
    (*region_0)[i32(3)] = ixx_0 - _S31 * cx_0;
    (*region_0)[i32(4)] = iyy_0 - a_3 * cy_0 * cy_0;
    (*region_0)[i32(5)] = ixy_0 - _S31 * cy_0;
    return;
}

fn no_tension_patch_0( kn_0 : f32,  w0_1 : f32,  w1_1 : f32,  dz_1 : f32,  ax_1 : f32,  ay_1 : f32) -> vec4<f32>
{
    var r_1 : array<f32, i32(6)>;
    compressed_region_0(w0_1, w1_1, dz_1, ax_1, ay_1, &(r_1));
    var a_5 : f32 = r_1[i32(0)];
    if((r_1[i32(0)]) == 0.0f)
    {
        return vec4<f32>(0.0f);
    }
    var k_0 : f32 = kn_0 / (w0_1 * w1_1);
    var fc_1 : f32 = dz_1 + ax_1 * r_1[i32(2)] - ay_1 * r_1[i32(1)];
    var _S32 : f32 = a_5 * fc_1;
    var _S33 : f32 = - ay_1;
    return vec4<f32>(k_0 * a_5 * fc_1, k_0 * (_S32 * r_1[i32(2)] + (_S33 * r_1[i32(5)] + ax_1 * r_1[i32(4)])), - k_0 * (_S32 * r_1[i32(1)] + (_S33 * r_1[i32(3)] + ax_1 * r_1[i32(5)])), 0.5f * k_0 * (_S32 * fc_1 + ay_1 * ay_1 * r_1[i32(3)] + ax_1 * ax_1 * r_1[i32(4)] - 2.0f * ax_1 * ay_1 * r_1[i32(5)]));
}

fn contact_normal_0( mat_3 : ptr<function, JointMaterial_std430_0>,  b_4 : ptr<function, JointBond_std430_0>,  crush_1 : f32,  d_lin_0 : vec3<f32>,  d_ang_0 : vec3<f32>) -> vec4<f32>
{
    var kn_1 : f32 = (*b_4).stiff0_0.x;
    var w0_2 : f32 = (*b_4).geom0_0.y;
    var w1_2 : f32 = (*b_4).geom0_0.z;
    var nc_sum_0 : f32;
    var m1_0 : f32;
    var m2_0 : f32;
    var energy_1 : f32;
    if(((((*mat_3).kind_flags_0.y) & (u32(4)))) != u32(0))
    {
        var p_0 : vec4<f32> = no_tension_patch_0(kn_1 * (1.0f - crush_1), w0_2, w1_2, d_lin_0.z, d_ang_0.x, d_ang_0.y);
        var _S34 : f32 = p_0.y;
        var _S35 : f32 = p_0.z;
        var _S36 : f32 = p_0.w;
        nc_sum_0 = p_0.x;
        m1_0 = _S34;
        m2_0 = _S35;
        energy_1 = _S36;
    }
    else
    {
        var ki_0 : f32 = kn_1 * (1.0f - crush_1) / 36.0f;
        var _S37 : f32 = d_ang_0.x;
        var _S38 : f32 = d_ang_0.y;
        var spread_0 : f32 = abs(_S37) * 0.4166666567325592f * w1_2 + abs(_S38) * 0.4166666567325592f * w0_2;
        var _S39 : f32 = d_lin_0.z;
        var slack_0 : f32 = 9.99999997475242708e-07f * (abs(_S39) + spread_0);
        if((_S39 - spread_0) > slack_0)
        {
            nc_sum_0 = 0.0f;
            m1_0 = 0.0f;
            m2_0 = 0.0f;
            energy_1 = 0.0f;
        }
        else
        {
            if((_S39 + spread_0) < (- slack_0))
            {
                var i1_0 : f32 = 2.91666650772094727f * w0_2 * w0_2;
                var i2_0 : f32 = 2.91666650772094727f * w1_2 * w1_2;
                var _S40 : f32 = ki_0 * _S37 * i2_0;
                var _S41 : f32 = ki_0 * _S38 * i1_0;
                var _S42 : f32 = 0.5f * ki_0 * (36.0f * _S39 * _S39 + _S37 * _S37 * i2_0 + _S38 * _S38 * i1_0);
                nc_sum_0 = ki_0 * 36.0f * _S39;
                m1_0 = _S40;
                m2_0 = _S41;
                energy_1 = _S42;
            }
            else
            {
                var i_1 : u32 = u32(0);
                var nc_sum_1 : f32 = 0.0f;
                var m1_1 : f32 = 0.0f;
                var m2_1 : f32 = 0.0f;
                var energy_2 : f32 = 0.0f;
                loop
                {
                    if(i_1 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S43 : f32 = SPRING_AT_0[i_1] * w0_2;
                    var j_0 : u32 = u32(0);
                    nc_sum_0 = nc_sum_1;
                    m1_0 = m1_1;
                    m2_0 = m2_1;
                    energy_1 = energy_2;
                    loop
                    {
                        if(j_0 < u32(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        var s2_0 : f32 = SPRING_AT_0[j_0] * w1_2;
                        var di_0 : f32 = _S39 + _S37 * s2_0 - _S38 * _S43;
                        if(di_0 < 0.0f)
                        {
                            var f_1 : f32 = ki_0 * di_0;
                            var m1_2 : f32 = m1_0 + f_1 * s2_0;
                            var m2_2 : f32 = m2_0 - f_1 * _S43;
                            var energy_3 : f32 = energy_1 + 0.5f * ki_0 * di_0 * di_0;
                            nc_sum_0 = nc_sum_0 + f_1;
                            m1_0 = m1_2;
                            m2_0 = m2_2;
                            energy_1 = energy_3;
                        }
                        j_0 = j_0 + u32(1);
                    }
                    i_1 = i_1 + u32(1);
                    nc_sum_1 = nc_sum_0;
                    m1_1 = m1_0;
                    m2_1 = m2_0;
                    energy_2 = energy_1;
                }
                nc_sum_0 = nc_sum_1;
                m1_0 = m1_1;
                m2_0 = m2_1;
                energy_1 = energy_2;
            }
        }
    }
    return vec4<f32>(nc_sum_0, m1_0, m2_0, energy_1);
}

fn damage_law_0( kind_0 : u32,  kappa_1 : f32,  r_2 : f32) -> f32
{
    if(kappa_1 <= 1.0f)
    {
        return 0.0f;
    }
    if(kind_0 == u32(0))
    {
        if(r_2 <= 1.0f)
        {
            return 1.0f;
        }
        return min(fdiv_0(r_2 * (kappa_1 - 1.0f), kappa_1 * (r_2 - 1.0f)), 1.0f);
    }
    if(kappa_1 >= (0.5f * (r_2 + 1.0f)))
    {
        return 1.0f;
    }
    return 1.0f - fdiv_0(1.0f, kappa_1);
}

fn damage_increment_0( kind_1 : u32,  kappa_old_0 : f32,  lambda_0 : f32,  r_3 : f32,  d_old_0 : f32,  psi_0 : f32) -> vec2<f32>
{
    var _S44 : f32 = max(damage_law_0(kind_1, lambda_0, r_3), d_old_0);
    var _S45 : bool;
    if(_S44 <= d_old_0)
    {
        _S45 = true;
    }
    else
    {
        _S45 = d_old_0 >= 1.0f;
    }
    if(_S45)
    {
        return vec2<f32>(d_old_0, 0.0f);
    }
    var u0_0 : f32 = fdiv_0(psi_0, lambda_0 * lambda_0);
    var _S46 : f32 = max(kappa_old_0, 1.0f);
    if(kind_1 == u32(0))
    {
        if(r_3 > 1.0f)
        {
            return vec2<f32>(_S44, fdiv_0(u0_0 * r_3, r_3 - 1.0f) * max(min(lambda_0, r_3) - min(_S46, r_3), 0.0f));
        }
        return vec2<f32>(_S44, (1.0f - d_old_0) * psi_0);
    }
    var ku_0 : f32 = 0.5f * (r_3 + 1.0f);
    var plateau_0 : f32 = u0_0 * max(min(lambda_0, ku_0) - min(_S46, ku_0), 0.0f);
    var snap_0 : f32;
    if(_S44 >= 1.0f)
    {
        snap_0 = u0_0 * ku_0;
    }
    else
    {
        snap_0 = 0.0f;
    }
    return vec2<f32>(_S44, plateau_0 + snap_0);
}

fn signum_0( x_2 : f32) -> f32
{
    var _S47 : f32;
    if((((bitcast<u32>((x_2))) & (u32(2147483648)))) != u32(0))
    {
        _S47 = -1.0f;
    }
    else
    {
        _S47 = 1.0f;
    }
    return _S47;
}

fn return_map_0( k_1 : f32,  total_0 : f32,  plastic_0 : f32,  cap_0 : f32) -> vec2<f32>
{
    var trial_0 : f32 = k_1 * (total_0 - plastic_0);
    if((abs(trial_0)) <= cap_0)
    {
        return vec2<f32>(trial_0, 0.0f);
    }
    var f_2 : f32 = cap_0 * signum_0(trial_0);
    return vec2<f32>(f_2, fdiv_0(trial_0 - f_2, k_1));
}

struct Contact_0
{
     q_lin_1 : vec3<f32>,
     q_ang_1 : vec3<f32>,
     energy_4 : f32,
     diss_0 : f32,
     plastic_1 : vec3<f32>,
};

fn contact_part_0( mat_4 : ptr<function, JointMaterial_std430_0>,  b_5 : ptr<function, JointBond_std430_0>,  crush_2 : f32,  plastic_2 : vec3<f32>,  d_lin_1 : vec3<f32>,  d_ang_1 : vec3<f32>) -> Contact_0
{
    var c_0 : Contact_0;
    var _S48 : vec3<f32> = vec3<f32>(0.0f);
    c_0.q_lin_1 = _S48;
    c_0.q_ang_1 = _S48;
    c_0.energy_4 = 0.0f;
    c_0.diss_0 = 0.0f;
    c_0.plastic_1 = plastic_2;
    if(((((*mat_4).kind_flags_0.y) & (u32(2)))) == u32(0))
    {
        return c_0;
    }
    var ks_0 : f32 = (*b_5).stiff0_0.y;
    var kt_0 : f32 = (*b_5).stiff1_0.x;
    var _S49 : vec4<f32> = contact_normal_0(&((*mat_4)), &((*b_5)), crush_2, d_lin_1, d_ang_1);
    var nc_sum_2 : f32 = _S49.x;
    var m1_3 : f32 = _S49.y;
    var m2_3 : f32 = _S49.z;
    var energy_5 : f32 = _S49.w;
    var nc_0 : f32 = - nc_sum_2;
    var p_1 : vec3<f32> = plastic_2;
    c_0.q_lin_1 = vec3<f32>(0.0f, 0.0f, nc_sum_2);
    c_0.q_ang_1 = vec3<f32>(m1_3, m2_3, 0.0f);
    var slide_cap_0 : f32 = (*mat_4).strength_0.w * nc_0;
    var _S50 : f32 = ks_0 * (d_lin_1.x - plastic_2.x);
    var _S51 : f32 = ks_0 * (d_lin_1.y - plastic_2.y);
    var tn_0 : f32 = fsqrt_0(_S50 * _S50 + _S51 * _S51);
    var _S52 : bool;
    if(tn_0 > slide_cap_0)
    {
        _S52 = tn_0 > 0.0f;
    }
    else
    {
        _S52 = false;
    }
    var diss_1 : f32;
    if(_S52)
    {
        var _S53 : f32 = fdiv_0(_S50, tn_0);
        var _S54 : f32 = fdiv_0(_S51, tn_0);
        var dslip_0 : f32 = fdiv_0(tn_0 - slide_cap_0, ks_0);
        p_1[i32(0)] = p_1[i32(0)] + _S53 * dslip_0;
        p_1[i32(1)] = p_1[i32(1)] + _S54 * dslip_0;
        c_0.q_lin_1[i32(0)] = _S53 * slide_cap_0;
        c_0.q_lin_1[i32(1)] = _S54 * slide_cap_0;
        diss_1 = slide_cap_0 * dslip_0;
    }
    else
    {
        c_0.q_lin_1[i32(0)] = _S50;
        c_0.q_lin_1[i32(1)] = _S51;
        diss_1 = 0.0f;
    }
    var tq_0 : vec2<f32> = return_map_0(kt_0, d_ang_1.z, p_1.z, slide_cap_0 * (*b_5).geom1_0.z);
    var _S55 : f32 = tq_0.x;
    var _S56 : f32 = tq_0.y;
    var diss_2 : f32 = diss_1 + abs(_S55) * abs(_S56);
    p_1[i32(2)] = p_1[i32(2)] + _S56;
    c_0.q_ang_1[i32(2)] = _S55;
    c_0.energy_4 = energy_5 + 0.5f * (fdiv_0(sq_0(c_0.q_lin_1.x), ks_0) + fdiv_0(sq_0(c_0.q_lin_1.y), ks_0) + fdiv_0(sq_0(_S55), kt_0));
    c_0.diss_0 = diss_2;
    c_0.plastic_1 = p_1;
    return c_0;
}

fn contact_offsets_0( mat_5 : ptr<function, JointMaterial_std430_0>,  b_6 : ptr<function, JointBond_std430_0>,  crush_3 : f32,  plastic_3 : vec3<f32>,  d_lin_2 : vec3<f32>,  d_ang_2 : vec3<f32>) -> vec3<f32>
{
    var _S57 : u32 = (*mat_5).kind_flags_0.y;
    if(((_S57 & (u32(2)))) == u32(0))
    {
        return plastic_3;
    }
    var kn_2 : f32 = (*b_6).stiff0_0.x;
    var ks_1 : f32 = (*b_6).stiff0_0.y;
    var kt_1 : f32 = (*b_6).stiff1_0.x;
    var w0_3 : f32 = (*b_6).geom0_0.y;
    var w1_3 : f32 = (*b_6).geom0_0.z;
    var nc_sum_3 : f32;
    if(((_S57 & (u32(4)))) != u32(0))
    {
        var _S58 : vec4<f32> = no_tension_patch_0(kn_2 * (1.0f - crush_3), w0_3, w1_3, d_lin_2.z, d_ang_2.x, d_ang_2.y);
        nc_sum_3 = _S58.x;
    }
    else
    {
        var ki_1 : f32 = kn_2 * (1.0f - crush_3) / 36.0f;
        var _S59 : f32 = d_ang_2.x;
        var _S60 : f32 = d_ang_2.y;
        var spread_1 : f32 = abs(_S59) * 0.4166666567325592f * w1_3 + abs(_S60) * 0.4166666567325592f * w0_3;
        var _S61 : f32 = d_lin_2.z;
        var slack_1 : f32 = 9.99999997475242708e-07f * (abs(_S61) + spread_1);
        if((_S61 - spread_1) > slack_1)
        {
            nc_sum_3 = 0.0f;
        }
        else
        {
            if((_S61 + spread_1) < (- slack_1))
            {
                nc_sum_3 = ki_1 * 36.0f * _S61;
            }
            else
            {
                var i_2 : u32 = u32(0);
                var nc_sum_4 : f32 = 0.0f;
                loop
                {
                    if(i_2 < u32(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    var _S62 : f32 = SPRING_AT_0[i_2] * w0_3;
                    var j_1 : u32 = u32(0);
                    nc_sum_3 = nc_sum_4;
                    loop
                    {
                        if(j_1 < u32(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        var di_1 : f32 = _S61 + _S59 * (SPRING_AT_0[j_1] * w1_3) - _S60 * _S62;
                        if(di_1 < 0.0f)
                        {
                            nc_sum_3 = nc_sum_3 + ki_1 * di_1;
                        }
                        j_1 = j_1 + u32(1);
                    }
                    i_2 = i_2 + u32(1);
                    nc_sum_4 = nc_sum_3;
                }
                nc_sum_3 = nc_sum_4;
            }
        }
    }
    var nc_1 : f32 = - nc_sum_3;
    var p_2 : vec3<f32> = plastic_3;
    var slide_cap_1 : f32 = (*mat_5).strength_0.w * nc_1;
    var _S63 : f32 = ks_1 * (d_lin_2.x - plastic_3.x);
    var _S64 : f32 = ks_1 * (d_lin_2.y - plastic_3.y);
    var tn_1 : f32 = fsqrt_0(_S63 * _S63 + _S64 * _S64);
    var _S65 : bool;
    if(tn_1 > slide_cap_1)
    {
        _S65 = tn_1 > 0.0f;
    }
    else
    {
        _S65 = false;
    }
    if(_S65)
    {
        var _S66 : f32 = fdiv_0(_S64, tn_1);
        var dslip_1 : f32 = fdiv_0(tn_1 - slide_cap_1, ks_1);
        p_2[i32(0)] = p_2[i32(0)] + fdiv_0(_S63, tn_1) * dslip_1;
        p_2[i32(1)] = p_2[i32(1)] + _S66 * dslip_1;
    }
    p_2[i32(2)] = p_2[i32(2)] + return_map_0(kt_1, d_ang_2.z, p_2.z, slide_cap_1 * (*b_6).geom1_0.z).y;
    return p_2;
}

fn life_rate_0( mat_6 : ptr<function, JointMaterial_std430_0>,  s_0 : f32) -> f32
{
    if(s_0 <= 0.0f)
    {
        return 0.0f;
    }
    var _S67 : f32 = (*mat_6).misc_0.y;
    return fdiv_0((_S67 + 1.0f) * fpow_0(s_0, _S67), (*mat_6).misc_0.z);
}

struct JointResponse_0
{
     force_lin_0 : vec3<f32>,
     force_ang_0 : vec3<f32>,
     state_0 : JointState_0,
     dissipated_1 : f32,
     overshoot_0 : f32,
     stored_0 : f32,
     disconnected_0 : bool,
     fatigue_before_0 : f32,
     measures_0 : Measures_0,
};

fn joint_evaluate_0( mat_7 : ptr<function, JointMaterial_std430_0>,  b_7 : ptr<function, JointBond_std430_0>,  state_1 : ptr<function, JointState_std430_0>,  d_lin_3 : vec3<f32>,  d_ang_3 : vec3<f32>,  dt_0 : f32,  fracture_0 : bool) -> JointResponse_0
{
    var kn_3 : f32 = (*b_7).stiff0_0.x;
    var ks_2 : f32 = (*b_7).stiff0_0.y;
    var kb1_0 : f32 = (*b_7).stiff0_0.z;
    var kb2_0 : f32 = (*b_7).stiff0_0.w;
    var _S68 : vec4<f32> = (*b_7).stiff1_0;
    var kt_2 : f32 = (*b_7).stiff1_0.x;
    var has_rebar_2 : bool = ((*b_7).stiff1_0.w) != 0.0f;
    var kind_2 : u32 = (*mat_7).kind_flags_0.x;
    var flags_0 : u32 = (*mat_7).kind_flags_0.y;
    var softening_0 : bool = ((flags_0 & (u32(1)))) != u32(0);
    var st_2 : JointState_0;
    st_2.damage_0 = (*state_1).damage_0;
    st_2.crush_0 = (*state_1).crush_0;
    st_2.kappa_0 = (*state_1).kappa_0;
    st_2.kappa_c_0 = (*state_1).kappa_c_0;
    st_2.ductility_0 = (*state_1).ductility_0;
    st_2.ductility_c_0 = (*state_1).ductility_c_0;
    st_2.life_0 = (*state_1).life_0;
    st_2.plastic_x_0 = (*state_1).plastic_x_0;
    st_2.plastic_y_0 = (*state_1).plastic_y_0;
    st_2.plastic_t_0 = (*state_1).plastic_t_0;
    st_2.rebar_plastic_0 = (*state_1).rebar_plastic_0;
    st_2.rebar_slip0_0 = (*state_1).rebar_slip0_0;
    st_2.rebar_slip1_0 = (*state_1).rebar_slip1_0;
    st_2.rebar_work_0 = (*state_1).rebar_work_0;
    st_2.rebar_broken_0 = (*state_1).rebar_broken_0;
    st_2.strain_rate_0 = (*state_1).strain_rate_0;
    st_2.governing_stress_0 = (*state_1).governing_stress_0;
    st_2.dissipated_0 = (*state_1).dissipated_0;
    st_2.utilization_0 = (*state_1).utilization_0;
    st_2.mode_0 = (*state_1).mode_0;
    var _S69 : bool = connected_0(&((*state_1)), has_rebar_2);
    var qe_lin_0 : vec3<f32> = d_lin_3 * vec3<f32>(ks_2, ks_2, kn_3);
    var qe_ang_0 : vec3<f32> = d_ang_3 * vec3<f32>(kb1_0, kb2_0, kt_2);
    var _S70 : Measures_0 = stress_measures_0(&((*b_7)), qe_lin_0, qe_ang_0);
    var _S71 : f32 = max(max(_S70.tension_0, _S70.shear_0), _S70.compression_0);
    var _S72 : bool = dt_0 > 0.0f;
    var dif_1 : f32;
    if(_S72)
    {
        var raw_0 : f32 = fdiv_0(max(fdiv_0(_S71 - st_2.governing_stress_0, dt_0), 0.0f), (*mat_7).misc_0.w);
        var tau_0 : f32 = _S68.z;
        if(((flags_0 & (u32(16)))) != u32(0))
        {
            dif_1 = - expm1_accurate_0(- fdiv_0(dt_0, tau_0));
        }
        else
        {
            dif_1 = min(fdiv_0(dt_0, tau_0), 1.0f);
        }
        st_2.strain_rate_0 = st_2.strain_rate_0 + (raw_0 - st_2.strain_rate_0) * dif_1;
        st_2.governing_stress_0 = _S71;
    }
    if(((flags_0 & (u32(32)))) != u32(0))
    {
        var _S73 : f32 = dif_factor_0(&((*mat_7)), st_2.strain_rate_0);
        dif_1 = _S73;
    }
    else
    {
        dif_1 = 1.0f;
    }
    var weibull_0 : f32 = (*b_7).geom1_0.w;
    var _S74 : f32 = fatigue_factor_0(&((*mat_7)), st_2.life_0);
    var multiplier_1 : f32 = weibull_0 * dif_1 * _S74;
    var _S75 : vec4<f32> = failure_indices_0(&((*mat_7)), &((*b_7)), _S70, multiplier_1);
    var _S76 : f32 = _S75.x;
    var _S77 : f32 = _S75.y;
    st_2.utilization_0 = max(max(_S76, _S77), max(_S75.z, _S75.w));
    var _S78 : f32 = d_lin_3.x;
    var _S79 : f32 = d_lin_3.y;
    var _S80 : f32 = d_ang_3.z;
    var _S81 : f32 = ks_2 * (sq_0(_S78) + sq_0(_S79)) + kb1_0 * sq_0(d_ang_3.x) + kb2_0 * sq_0(d_ang_3.y) + kt_2 * sq_0(_S80);
    var _S82 : f32 = d_lin_3.z;
    var _S83 : bool = _S82 > 0.0f;
    if(_S83)
    {
        dif_1 = kn_3 * sq_0(_S82);
    }
    else
    {
        dif_1 = 0.0f;
    }
    var psi_ts_0 : f32 = 0.5f * (_S81 + dif_1);
    var psi_c_0 : f32;
    if(_S82 < 0.0f)
    {
        psi_c_0 = 0.5f * kn_3 * sq_0(_S82);
    }
    else
    {
        psi_c_0 = 0.0f;
    }
    var plastic_4 : vec3<f32> = vec3<f32>(st_2.plastic_x_0, st_2.plastic_y_0, st_2.plastic_t_0);
    var changed_0 : bool;
    if(((flags_0 & (u32(2)))) != u32(0))
    {
        changed_0 = ((*state_1).damage_0) != 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var patch_0 : f32;
    if(changed_0)
    {
        var _S84 : vec4<f32> = contact_normal_0(&((*mat_7)), &((*b_7)), (*state_1).crush_0, d_lin_3, d_ang_3);
        patch_0 = _S84.w + 0.5f * (ks_2 * (sq_0(_S78 - (*state_1).plastic_x_0) + sq_0(_S79 - (*state_1).plastic_y_0)) + kt_2 * sq_0(_S80 - (*state_1).plastic_t_0));
    }
    else
    {
        patch_0 = 0.0f;
    }
    if(has_rebar_2)
    {
        changed_0 = ((*state_1).rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var rebar_0 : f32;
    if(changed_0)
    {
        rebar_0 = 0.5f * ((*b_7).rebar0_0.x * sq_0(_S82 - (*state_1).rebar_plastic_0) + (*b_7).rebar0_0.y * (sq_0(_S78 - (*state_1).rebar_slip0_0) + sq_0(_S79 - (*state_1).rebar_slip1_0)));
    }
    else
    {
        rebar_0 = 0.0f;
    }
    var _S85 : f32 = (*state_1).damage_0;
    var _S86 : f32 = 1.0f - (*state_1).damage_0;
    var _S87 : f32 = (*state_1).crush_0;
    var frozen_0 : f32 = _S86 * (psi_ts_0 + (1.0f - (*state_1).crush_0) * psi_c_0) + (*state_1).damage_0 * patch_0 + rebar_0;
    var diss_contact_0 : f32;
    var psi_contact_0 : f32;
    var intact_normal_0 : f32;
    var dissipated_2 : f32;
    var qc_lin_0 : vec3<f32>;
    if(fracture_0)
    {
        var _S88 : bool = _S76 >= _S77;
        if(_S88)
        {
            diss_contact_0 = _S76;
        }
        else
        {
            diss_contact_0 = _S77;
        }
        var mode_ts_0 : u32;
        if(_S88)
        {
            mode_ts_0 = u32(1);
        }
        else
        {
            mode_ts_0 = u32(2);
        }
        if(diss_contact_0 > (st_2.kappa_0))
        {
            changed_0 = diss_contact_0 > 1.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            changed_0 = psi_ts_0 > 0.0f;
        }
        else
        {
            changed_0 = false;
        }
        var mode_c_0 : u32;
        if(changed_0)
        {
            if(mode_ts_0 == u32(1))
            {
                psi_contact_0 = (*mat_7).energy_0.y;
            }
            else
            {
                psi_contact_0 = (*mat_7).energy_0.z;
            }
            if(softening_0)
            {
                intact_normal_0 = fdiv_0(psi_contact_0 * (*b_7).geom0_0.x * diss_contact_0 * diss_contact_0, psi_ts_0);
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            st_2.ductility_0 = intact_normal_0;
            if(softening_0)
            {
                mode_c_0 = kind_2;
            }
            else
            {
                mode_c_0 = u32(0);
            }
            var inc_0 : vec2<f32> = damage_increment_0(mode_c_0, st_2.kappa_0, diss_contact_0, intact_normal_0, st_2.damage_0, psi_ts_0);
            var _S89 : f32 = inc_0.x;
            if(_S89 > (st_2.damage_0))
            {
                var _S90 : Contact_0 = contact_part_0(&((*mat_7)), &((*b_7)), st_2.crush_0, plastic_4, d_lin_3, d_ang_3);
                var _S91 : f32 = max(inc_0.y - max(_S90.energy_4 - (1.0f - st_2.crush_0) * psi_c_0, 0.0f) * (_S89 - st_2.damage_0), 0.0f);
                st_2.damage_0 = _S89;
                st_2.mode_0 = mode_ts_0;
                dissipated_2 = _S91;
            }
            else
            {
                dissipated_2 = 0.0f;
            }
        }
        else
        {
            dissipated_2 = 0.0f;
        }
        st_2.kappa_0 = max(st_2.kappa_0, diss_contact_0);
        if(_S85 > 0.0f)
        {
            var _S92 : Contact_0 = contact_part_0(&((*mat_7)), &((*b_7)), _S87, vec3<f32>((*state_1).plastic_x_0, (*state_1).plastic_y_0, (*state_1).plastic_t_0), d_lin_3, d_ang_3);
            qc_lin_0 = qe_ang_0 * vec3<f32>(_S86) + _S92.q_ang_1 * vec3<f32>(_S85);
        }
        else
        {
            qc_lin_0 = qe_ang_0;
        }
        var _S93 : Measures_0 = stress_measures_0(&((*b_7)), vec3<f32>(0.0f, 0.0f, min(qe_lin_0.z, 0.0f)), qc_lin_0);
        var _S94 : vec4<f32> = failure_indices_0(&((*mat_7)), &((*b_7)), _S93, multiplier_1);
        var _S95 : f32 = _S94.z;
        var _S96 : f32 = _S94.w;
        var _S97 : bool = _S95 >= _S96;
        if(_S97)
        {
            psi_contact_0 = _S95;
        }
        else
        {
            psi_contact_0 = _S96;
        }
        if(_S97)
        {
            mode_c_0 = u32(3);
        }
        else
        {
            mode_c_0 = u32(4);
        }
        if(psi_contact_0 > (st_2.kappa_c_0))
        {
            changed_0 = psi_contact_0 > 1.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            changed_0 = psi_c_0 > 0.0f;
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            if(softening_0)
            {
                intact_normal_0 = fdiv_0((*mat_7).energy_0.w * (*b_7).geom0_0.x * psi_contact_0 * psi_contact_0, psi_c_0);
            }
            else
            {
                intact_normal_0 = 0.0f;
            }
            st_2.ductility_c_0 = intact_normal_0;
            var law_0 : u32;
            if(!softening_0)
            {
                law_0 = u32(0);
            }
            else
            {
                if(mode_c_0 == u32(4))
                {
                    mode_ts_0 = u32(1);
                }
                else
                {
                    mode_ts_0 = kind_2;
                }
                law_0 = mode_ts_0;
            }
            var inc_1 : vec2<f32> = damage_increment_0(law_0, st_2.kappa_c_0, psi_contact_0, intact_normal_0, st_2.crush_0, psi_c_0);
            var _S98 : f32 = inc_1.x;
            if(_S98 > (st_2.crush_0))
            {
                var dissipated_3 : f32 = dissipated_2 + inc_1.y;
                st_2.crush_0 = _S98;
                st_2.mode_0 = mode_c_0;
                if(_S98 >= 1.0f)
                {
                    changed_0 = (st_2.damage_0) < 1.0f;
                }
                else
                {
                    changed_0 = false;
                }
                if(changed_0)
                {
                    var dissipated_4 : f32 = dissipated_3 + psi_ts_0 * (1.0f - st_2.damage_0);
                    st_2.damage_0 = 1.0f;
                    dissipated_2 = dissipated_4;
                }
                else
                {
                    dissipated_2 = dissipated_3;
                }
            }
        }
        st_2.kappa_c_0 = max(st_2.kappa_c_0, psi_contact_0);
    }
    else
    {
        dissipated_2 = 0.0f;
    }
    var dmg_0 : f32 = st_2.damage_0;
    var _S99 : vec3<f32> = vec3<f32>(0.0f);
    var qc_ang_0 : vec3<f32>;
    if((st_2.damage_0) == 0.0f)
    {
        if(((flags_0 & (u32(8)))) == u32(0))
        {
            var _S100 : vec3<f32> = contact_offsets_0(&((*mat_7)), &((*b_7)), st_2.crush_0, plastic_4, d_lin_3, d_ang_3);
            st_2.plastic_x_0 = _S100.x;
            st_2.plastic_y_0 = _S100.y;
            st_2.plastic_t_0 = _S100.z;
        }
        diss_contact_0 = 0.0f;
        qc_lin_0 = _S99;
        qc_ang_0 = _S99;
        psi_contact_0 = 0.0f;
    }
    else
    {
        var _S101 : Contact_0 = contact_part_0(&((*mat_7)), &((*b_7)), st_2.crush_0, plastic_4, d_lin_3, d_ang_3);
        st_2.plastic_x_0 = _S101.plastic_1.x;
        st_2.plastic_y_0 = _S101.plastic_1.y;
        st_2.plastic_t_0 = _S101.plastic_1.z;
        diss_contact_0 = _S101.diss_0;
        qc_lin_0 = _S101.q_lin_1;
        qc_ang_0 = _S101.q_ang_1;
        psi_contact_0 = _S101.energy_4;
    }
    var dissipated_5 : f32 = dissipated_2 + dmg_0 * diss_contact_0;
    if(_S83)
    {
        intact_normal_0 = kn_3 * _S82;
    }
    else
    {
        intact_normal_0 = (1.0f - st_2.crush_0) * kn_3 * _S82;
    }
    var _S102 : f32 = 1.0f - dmg_0;
    var force_lin_1 : vec3<f32> = vec3<f32>(_S102 * qe_lin_0.x + dmg_0 * qc_lin_0.x, _S102 * qe_lin_0.y + dmg_0 * qc_lin_0.y, _S102 * intact_normal_0 + dmg_0 * qc_lin_0.z);
    var force_ang_1 : vec3<f32> = qe_ang_0 * vec3<f32>(_S102) + qc_ang_0 * vec3<f32>(dmg_0);
    var stored_1 : f32 = _S102 * (psi_ts_0 + (1.0f - st_2.crush_0) * psi_c_0) + dmg_0 * psi_contact_0;
    if(has_rebar_2)
    {
        changed_0 = (st_2.rebar_broken_0) == 0.0f;
    }
    else
    {
        changed_0 = false;
    }
    var stored_2 : f32;
    var force_lin_2 : vec3<f32>;
    if(changed_0)
    {
        var k_axial_0 : f32 = (*b_7).rebar0_0.x;
        var k_dowel_0 : f32 = (*b_7).rebar0_0.y;
        var yield_force_0 : f32 = (*b_7).rebar0_0.z;
        var dowel_capacity_0 : f32 = (*b_7).rebar0_0.w;
        var nr_0 : vec2<f32> = return_map_0(k_axial_0, _S82, st_2.rebar_plastic_0, yield_force_0);
        var v1_0 : vec2<f32> = return_map_0(k_dowel_0, _S78, st_2.rebar_slip0_0, dowel_capacity_0);
        var v2_0 : vec2<f32> = return_map_0(k_dowel_0, _S79, st_2.rebar_slip1_0, dowel_capacity_0);
        var _S103 : f32 = nr_0.y;
        var _S104 : f32 = v1_0.y;
        var _S105 : f32 = v2_0.y;
        var work_0 : f32 = yield_force_0 * abs(_S103) + dowel_capacity_0 * (abs(_S104) + abs(_S105));
        st_2.rebar_plastic_0 = st_2.rebar_plastic_0 + _S103;
        st_2.rebar_slip0_0 = st_2.rebar_slip0_0 + _S104;
        st_2.rebar_slip1_0 = st_2.rebar_slip1_0 + _S105;
        st_2.rebar_work_0 = st_2.rebar_work_0 + work_0;
        var dissipated_6 : f32 = dissipated_5 + work_0;
        var _S106 : f32 = nr_0.x;
        var _S107 : f32 = v1_0.x;
        var _S108 : f32 = v2_0.x;
        var elastic_0 : f32 = 0.5f * (fdiv_0(sq_0(_S106), k_axial_0) + fdiv_0(sq_0(_S107) + sq_0(_S108), k_dowel_0));
        if(fracture_0)
        {
            changed_0 = (st_2.rebar_work_0) >= ((*b_7).rebar1_0.x);
        }
        else
        {
            changed_0 = false;
        }
        if(changed_0)
        {
            st_2.rebar_broken_0 = 1.0f;
            var dissipated_7 : f32 = dissipated_6 + elastic_0;
            force_lin_2 = force_lin_1;
            stored_2 = stored_1;
            dissipated_2 = dissipated_7;
        }
        else
        {
            var stored_3 : f32 = stored_1 + elastic_0;
            force_lin_2 = force_lin_1 + vec3<f32>(_S107, _S108, _S106);
            stored_2 = stored_3;
            dissipated_2 = dissipated_6;
        }
    }
    else
    {
        force_lin_2 = force_lin_1;
        stored_2 = stored_1;
        dissipated_2 = dissipated_5;
    }
    if(fracture_0)
    {
        changed_0 = _S72;
    }
    else
    {
        changed_0 = false;
    }
    if(changed_0)
    {
        changed_0 = ((flags_0 & (u32(64)))) != u32(0);
    }
    else
    {
        changed_0 = false;
    }
    if(changed_0)
    {
        var _S109 : Measures_0 = stress_measures_0(&((*b_7)), force_lin_2, force_ang_1);
        var _S110 : vec4<f32> = failure_indices_0(&((*mat_7)), &((*b_7)), _S109, weibull_0);
        var _S111 : f32 = life_rate_0(&((*mat_7)), max(max(_S110.x, _S110.y), _S110.z));
        st_2.life_0 = max(st_2.life_0 - _S111 * dt_0, 0.0f);
    }
    if((st_2.damage_0) != _S85)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.crush_0) != _S87;
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.plastic_x_0) != ((*state_1).plastic_x_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.plastic_y_0) != ((*state_1).plastic_y_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.plastic_t_0) != ((*state_1).plastic_t_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.rebar_plastic_0) != ((*state_1).rebar_plastic_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.rebar_slip0_0) != ((*state_1).rebar_slip0_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.rebar_slip1_0) != ((*state_1).rebar_slip1_0);
    }
    if(changed_0)
    {
        changed_0 = true;
    }
    else
    {
        changed_0 = (st_2.rebar_broken_0) != ((*state_1).rebar_broken_0);
    }
    var released_0 : f32;
    if(changed_0)
    {
        released_0 = frozen_0 - stored_2;
    }
    else
    {
        released_0 = 0.0f;
    }
    var _S112 : f32 = max(min(dissipated_2, released_0), 0.0f);
    var overshoot_1 : f32 = released_0 - _S112;
    st_2.dissipated_0 = st_2.dissipated_0 + _S112;
    var resp_0 : JointResponse_0;
    resp_0.force_lin_0 = force_lin_2;
    resp_0.force_ang_0 = force_ang_1;
    resp_0.state_0 = st_2;
    resp_0.dissipated_1 = _S112;
    resp_0.overshoot_0 = overshoot_1;
    resp_0.stored_0 = stored_2;
    if(_S69)
    {
        changed_0 = !connected_1(st_2, has_rebar_2);
    }
    else
    {
        changed_0 = false;
    }
    resp_0.disconnected_0 = changed_0;
    resp_0.fatigue_before_0 = _S74;
    resp_0.measures_0 = _S70;
    return resp_0;
}

fn secant_factors_0( b_8 : ptr<function, JointBond_std430_0>,  st_3 : JointState_0,  d_lin_4 : vec3<f32>,  f_lin_0 : ptr<function, vec3<f32>>,  f_ang_0 : ptr<function, vec3<f32>>)
{
    var compressed_0 : bool = (d_lin_4.z) < 0.0f;
    var contact_0 : f32;
    if(compressed_0)
    {
        contact_0 = st_3.damage_0;
    }
    else
    {
        contact_0 = 0.0f;
    }
    var _S113 : f32 = 1.0f - st_3.damage_0;
    var _S114 : f32 = max(_S113 + contact_0, 9.99999997475242708e-07f);
    var normal_0 : f32;
    if(compressed_0)
    {
        normal_0 = max(1.0f - st_3.crush_0, 9.99999997475242708e-07f);
    }
    else
    {
        normal_0 = max(_S113, 9.99999997475242708e-07f);
    }
    (*f_lin_0) = vec3<f32>(_S114, _S114, normal_0);
    (*f_ang_0) = vec3<f32>(_S114);
    var _S115 : bool;
    if(((*b_8).stiff1_0.w) != 0.0f)
    {
        _S115 = (st_3.rebar_broken_0) == 0.0f;
    }
    else
    {
        _S115 = false;
    }
    if(_S115)
    {
        var _S116 : vec4<f32> = (*b_8).rebar0_0;
        var _S117 : vec4<f32> = (*b_8).stiff0_0;
        (*f_lin_0)[i32(2)] = (*f_lin_0)[i32(2)] + fdiv_0((*b_8).rebar0_0.x, (*b_8).stiff0_0.x);
        var _S118 : f32 = fdiv_0(_S116.y, _S117.y);
        (*f_lin_0)[i32(0)] = (*f_lin_0)[i32(0)] + _S118;
        (*f_lin_0)[i32(1)] = (*f_lin_0)[i32(1)] + _S118;
    }
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn joint_eval_test(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var i_3 : u32 = id_0.x;
    if(i_3 >= (params_0.count_0))
    {
        return;
    }
    var _S119 : JointBond_std430_0 = bonds_0[i_3];
    var _S120 : JointMaterial_std430_0 = materials_0[_S119.ids_0.x];
    var _S121 : u32 = u32(2) * i_3;
    var a_6 : vec4<f32> = inputs_0[_S121];
    var c_1 : vec4<f32> = inputs_0[_S121 + u32(1)];
    var _S122 : JointState_std430_0 = states_in_0[i_3];
    var _S123 : vec3<f32> = a_6.xyz;
    var _S124 : JointResponse_0 = joint_evaluate_0(&(_S120), &(_S119), &(_S122), _S123, c_1.xyz, a_6.w, (c_1.w) != 0.0f);
    var f_lin_1 : vec3<f32>;
    var f_ang_1 : vec3<f32>;
    secant_factors_0(&(_S119), _S124.state_0, _S123, &(f_lin_1), &(f_ang_1));
    states_out_0[i_3].damage_0 = _S124.state_0.damage_0;
    states_out_0[i_3].crush_0 = _S124.state_0.crush_0;
    states_out_0[i_3].kappa_0 = _S124.state_0.kappa_0;
    states_out_0[i_3].kappa_c_0 = _S124.state_0.kappa_c_0;
    states_out_0[i_3].ductility_0 = _S124.state_0.ductility_0;
    states_out_0[i_3].ductility_c_0 = _S124.state_0.ductility_c_0;
    states_out_0[i_3].life_0 = _S124.state_0.life_0;
    states_out_0[i_3].plastic_x_0 = _S124.state_0.plastic_x_0;
    states_out_0[i_3].plastic_y_0 = _S124.state_0.plastic_y_0;
    states_out_0[i_3].plastic_t_0 = _S124.state_0.plastic_t_0;
    states_out_0[i_3].rebar_plastic_0 = _S124.state_0.rebar_plastic_0;
    states_out_0[i_3].rebar_slip0_0 = _S124.state_0.rebar_slip0_0;
    states_out_0[i_3].rebar_slip1_0 = _S124.state_0.rebar_slip1_0;
    states_out_0[i_3].rebar_work_0 = _S124.state_0.rebar_work_0;
    states_out_0[i_3].rebar_broken_0 = _S124.state_0.rebar_broken_0;
    states_out_0[i_3].strain_rate_0 = _S124.state_0.strain_rate_0;
    states_out_0[i_3].governing_stress_0 = _S124.state_0.governing_stress_0;
    states_out_0[i_3].dissipated_0 = _S124.state_0.dissipated_0;
    states_out_0[i_3].utilization_0 = _S124.state_0.utilization_0;
    states_out_0[i_3].mode_0 = _S124.state_0.mode_0;
    var _S125 : u32 = u32(5) * i_3;
    outputs_0[_S125] = vec4<f32>(_S124.force_lin_0, _S124.dissipated_1);
    outputs_0[_S125 + u32(1)] = vec4<f32>(_S124.force_ang_0, _S124.overshoot_0);
    outputs_0[_S125 + u32(2)] = vec4<f32>(f_lin_1, _S124.stored_0);
    var _S126 : vec3<f32> = f_ang_1;
    var _S127 : f32;
    if(_S124.disconnected_0)
    {
        _S127 = 1.0f;
    }
    else
    {
        _S127 = 0.0f;
    }
    outputs_0[_S125 + u32(3)] = vec4<f32>(_S126, _S127);
    outputs_0[_S125 + u32(4)] = vec4<f32>(_S124.measures_0.tension_0, _S124.measures_0.shear_0, _S124.measures_0.compression_0, _S124.measures_0.compressive_force_0);
    return;
}

