struct ThresholdParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) eload_base_0 : u32,
    @align(8) pending_0 : u32,
    @align(4) impactor_count_0 : u32,
    @align(16) engine_base_0 : u32,
    @align(4) pad0_0 : u32,
    @align(8) dt_0 : f32,
    @align(4) speed_0 : f32,
};

@binding(0) @group(0) var<uniform> params_0 : ThresholdParams_std140_0;
@binding(6) @group(0) var<storage, read_write> status_0 : array<u32>;

struct Island_std430_0
{
    @align(16) range_0 : vec4<u32>,
    @align(16) info_0 : vec4<u32>,
    @align(16) com_0 : vec4<f32>,
    @align(16) inertia0_0 : vec4<f32>,
    @align(16) inertia1_0 : vec4<f32>,
    @align(16) inertia2_0 : vec4<f32>,
    @align(16) inv0_0 : vec4<f32>,
    @align(16) inv1_0 : vec4<f32>,
    @align(16) inv2_0 : vec4<f32>,
    @align(16) wcom_0 : vec4<f32>,
    @align(16) winv0_0 : vec4<f32>,
    @align(16) winv1_0 : vec4<f32>,
    @align(16) winv2_0 : vec4<f32>,
    @align(16) rotation_0 : vec4<f32>,
    @align(16) position_0 : vec4<f32>,
    @align(16) position_err_0 : vec4<f32>,
    @align(16) velocity_0 : vec4<f32>,
    @align(16) velocity_err_0 : vec4<f32>,
    @align(16) angular_velocity_0 : vec4<f32>,
    @align(16) done_0 : u32,
    @align(4) coverage_0 : u32,
    @align(8) wide_groups_0 : u32,
    @align(4) done_pad_0 : u32,
    @align(16) probes_0 : vec4<u32>,
    @align(16) energy_0 : vec4<f32>,
    @align(16) rotation_err_0 : vec4<f32>,
    @align(16) momentum_0 : vec4<f32>,
    @align(16) momentum_err_0 : vec4<f32>,
};

@binding(1) @group(0) var<storage, read_write> islands_0 : array<Island_std430_0>;

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

struct BondStatic_std430_0
{
    @align(16) t1_0 : vec4<f32>,
    @align(16) t2_0 : vec4<f32>,
    @align(16) normal_0 : vec4<f32>,
    @align(16) ra_0 : vec4<f32>,
    @align(16) rb_0 : vec4<f32>,
    @align(16) centroid_0 : vec4<f32>,
    @align(16) c_lin_0 : vec4<f32>,
    @align(16) c_ang_0 : vec4<f32>,
    @align(16) law_0 : JointBond_std430_0,
};

@binding(2) @group(0) var<storage, read> bond_static_0 : array<BondStatic_std430_0>;

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

struct BondDyn_std430_0
{
    @align(16) js_0 : JointState_std430_0,
    @align(16) force_lin_0 : vec4<f32>,
    @align(16) force_ang_0 : vec4<f32>,
    @align(16) sums_0 : vec4<f32>,
    @align(16) comps_0 : vec4<f32>,
    @align(16) events_0 : vec4<u32>,
};

@binding(3) @group(0) var<storage, read_write> bond_dyn_0 : array<BondDyn_std430_0>;

struct JointMaterial_std140_0
{
    @align(16) strength_0 : vec4<f32>,
    @align(16) energy_1 : vec4<f32>,
    @align(16) dif_0 : vec4<f32>,
    @align(16) misc_0 : vec4<f32>,
    @align(16) kind_flags_0 : vec4<u32>,
};

struct _Array_std140_JointMaterial64_0
{
    @align(16) data_0 : array<JointMaterial_std140_0, i32(64)>,
};

struct MaterialTable_std140_0
{
    @align(16) m_0 : _Array_std140_JointMaterial64_0,
};

@binding(5) @group(0) var<uniform> materials_0 : MaterialTable_std140_0;
@binding(4) @group(0) var<storage, read> ctab_0 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read> fast_0 : array<vec4<f32>>;

struct EngineBody_std430_0
{
    @align(16) pos_inv_mass_0 : vec4<f32>,
    @align(16) rot_0 : vec4<f32>,
    @align(16) inv_inertia_flags_0 : vec4<f32>,
    @align(16) vel_sleep_0 : vec4<f32>,
    @align(16) ang_vel_gen_0 : vec4<f32>,
    @align(16) initial_pos_0 : vec4<f32>,
    @align(16) initial_rot_0 : vec4<f32>,
};

@binding(9) @group(0) var<storage, read> bodies_0 : array<EngineBody_std430_0>;

struct ChunkStatic_std430_0
{
    @align(16) center_0 : vec4<f32>,
    @align(16) inertia0_1 : vec4<f32>,
    @align(16) inertia1_1 : vec4<f32>,
    @align(16) inertia2_1 : vec4<f32>,
    @align(16) inv0_1 : vec4<f32>,
    @align(16) inv1_1 : vec4<f32>,
    @align(16) inv2_1 : vec4<f32>,
    @align(16) scale_0 : vec4<f32>,
    @align(16) info_1 : vec4<u32>,
    @align(16) load_range_0 : vec4<u32>,
    @align(16) half_0 : vec4<f32>,
    @align(16) crot0_0 : vec4<f32>,
    @align(16) crot1_0 : vec4<f32>,
    @align(16) crot2_0 : vec4<f32>,
    @align(16) cmat_0 : vec4<f32>,
    @align(16) cinfo_0 : vec4<u32>,
};

@binding(7) @group(0) var<storage, read> chunks_0 : array<ChunkStatic_std430_0>;

@binding(8) @group(0) var<storage, read> state_0 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> impactors_0 : array<vec4<f32>>;

var<workgroup> g_group_uniform_0 : u32;

fn group_uniform_0( value_0 : u32,  thread_0 : u32) -> u32
{
    if(thread_0 == u32(0))
    {
        g_group_uniform_0 = value_0;
    }
    var _S1 : u32 = (workgroupUniformLoad(&((g_group_uniform_0))));
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

fn connected_0( st_0 : JointState_0,  has_rebar_0 : bool) -> bool
{
    var _S2 : bool;
    if((st_0.damage_0) < 1.0f)
    {
        _S2 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S2 = (st_0.rebar_broken_0) == 0.0f;
        }
        else
        {
            _S2 = false;
        }
    }
    return _S2;
}

fn contact_force_0( chunk_0 : u32) -> f32
{
    return length(ctab_0[params_0.eload_base_0 + u32(2) * chunk_0].xyz);
}

var<workgroup> g_broke_0 : array<u32, i32(256)>;

struct Island_0
{
     range_0 : vec4<u32>,
     info_0 : vec4<u32>,
     com_0 : vec4<f32>,
     inertia0_0 : vec4<f32>,
     inertia1_0 : vec4<f32>,
     inertia2_0 : vec4<f32>,
     inv0_0 : vec4<f32>,
     inv1_0 : vec4<f32>,
     inv2_0 : vec4<f32>,
     wcom_0 : vec4<f32>,
     winv0_0 : vec4<f32>,
     winv1_0 : vec4<f32>,
     winv2_0 : vec4<f32>,
     rotation_0 : vec4<f32>,
     position_0 : vec4<f32>,
     position_err_0 : vec4<f32>,
     velocity_0 : vec4<f32>,
     velocity_err_0 : vec4<f32>,
     angular_velocity_0 : vec4<f32>,
     done_0 : u32,
     coverage_0 : u32,
     wide_groups_0 : u32,
     done_pad_0 : u32,
     probes_0 : vec4<u32>,
     energy_0 : vec4<f32>,
     rotation_err_0 : vec4<f32>,
     momentum_0 : vec4<f32>,
     momentum_err_0 : vec4<f32>,
};

struct BondDyn_0
{
     js_0 : JointState_0,
     force_lin_0 : vec4<f32>,
     force_ang_0 : vec4<f32>,
     sums_0 : vec4<f32>,
     comps_0 : vec4<f32>,
     events_0 : vec4<u32>,
};

@compute
@workgroup_size(256, 1, 1)
fn threshold_breaks(@builtin(workgroup_id) group_0 : vec3<u32>, @builtin(local_invocation_id) thread_1 : vec3<u32>)
{
    var broke_0 : u32;
    var i_0 : u32 = group_0.x;
    var t_0 : u32 = thread_1.x;
    var _S3 : bool;
    if(i_0 >= (params_0.island_count_0))
    {
        _S3 = true;
    }
    else
    {
        _S3 = status_0[u32(0)] == u32(0);
    }
    if(_S3)
    {
        _S3 = true;
    }
    else
    {
        _S3 = ((status_0[u32(16) + i_0] & (u32(1)))) == u32(0);
    }
    var broke_1 : u32;
    if(_S3)
    {
        broke_1 = u32(1);
    }
    else
    {
        broke_1 = u32(0);
    }
    var _S4 : u32 = group_uniform_0(broke_1, t_0);
    if(_S4 != u32(0))
    {
        return;
    }
    var isl_0 : Island_0;
    isl_0.range_0 = islands_0[i_0].range_0;
    isl_0.info_0 = islands_0[i_0].info_0;
    isl_0.com_0 = islands_0[i_0].com_0;
    isl_0.inertia0_0 = islands_0[i_0].inertia0_0;
    isl_0.inertia1_0 = islands_0[i_0].inertia1_0;
    isl_0.inertia2_0 = islands_0[i_0].inertia2_0;
    isl_0.inv0_0 = islands_0[i_0].inv0_0;
    isl_0.inv1_0 = islands_0[i_0].inv1_0;
    isl_0.inv2_0 = islands_0[i_0].inv2_0;
    isl_0.wcom_0 = islands_0[i_0].wcom_0;
    isl_0.winv0_0 = islands_0[i_0].winv0_0;
    isl_0.winv1_0 = islands_0[i_0].winv1_0;
    isl_0.winv2_0 = islands_0[i_0].winv2_0;
    isl_0.rotation_0 = islands_0[i_0].rotation_0;
    isl_0.position_0 = islands_0[i_0].position_0;
    isl_0.position_err_0 = islands_0[i_0].position_err_0;
    isl_0.velocity_0 = islands_0[i_0].velocity_0;
    isl_0.velocity_err_0 = islands_0[i_0].velocity_err_0;
    isl_0.angular_velocity_0 = islands_0[i_0].angular_velocity_0;
    isl_0.done_0 = islands_0[i_0].done_0;
    isl_0.coverage_0 = islands_0[i_0].coverage_0;
    isl_0.wide_groups_0 = islands_0[i_0].wide_groups_0;
    isl_0.done_pad_0 = islands_0[i_0].done_pad_0;
    isl_0.probes_0 = islands_0[i_0].probes_0;
    isl_0.energy_0 = islands_0[i_0].energy_0;
    isl_0.rotation_err_0 = islands_0[i_0].rotation_err_0;
    isl_0.momentum_0 = islands_0[i_0].momentum_0;
    isl_0.momentum_err_0 = islands_0[i_0].momentum_err_0;
    var _S5 : u32 = isl_0.range_0.z + t_0;
    broke_1 = u32(0);
    var bi_0 : u32 = _S5;
    loop
    {
        if(bi_0 < (isl_0.range_0.w))
        {
        }
        else
        {
            break;
        }
        var law_1 : JointBond_std430_0 = bond_static_0[bi_0].law_0;
        var _S6 : JointState_0 = JointState_0( bond_dyn_0[bi_0].js_0.damage_0, bond_dyn_0[bi_0].js_0.crush_0, bond_dyn_0[bi_0].js_0.kappa_0, bond_dyn_0[bi_0].js_0.kappa_c_0, bond_dyn_0[bi_0].js_0.ductility_0, bond_dyn_0[bi_0].js_0.ductility_c_0, bond_dyn_0[bi_0].js_0.life_0, bond_dyn_0[bi_0].js_0.plastic_x_0, bond_dyn_0[bi_0].js_0.plastic_y_0, bond_dyn_0[bi_0].js_0.plastic_t_0, bond_dyn_0[bi_0].js_0.rebar_plastic_0, bond_dyn_0[bi_0].js_0.rebar_slip0_0, bond_dyn_0[bi_0].js_0.rebar_slip1_0, bond_dyn_0[bi_0].js_0.rebar_work_0, bond_dyn_0[bi_0].js_0.rebar_broken_0, bond_dyn_0[bi_0].js_0.strain_rate_0, bond_dyn_0[bi_0].js_0.governing_stress_0, bond_dyn_0[bi_0].js_0.dissipated_0, bond_dyn_0[bi_0].js_0.utilization_0, bond_dyn_0[bi_0].js_0.mode_0 );
        var dyn_0 : BondDyn_0;
        dyn_0.js_0 = _S6;
        dyn_0.force_lin_0 = bond_dyn_0[bi_0].force_lin_0;
        dyn_0.force_ang_0 = bond_dyn_0[bi_0].force_ang_0;
        dyn_0.sums_0 = bond_dyn_0[bi_0].sums_0;
        dyn_0.comps_0 = bond_dyn_0[bi_0].comps_0;
        dyn_0.events_0 = bond_dyn_0[bi_0].events_0;
        if(!connected_0(dyn_0.js_0, (bond_static_0[bi_0].law_0.stiff1_0.w) != 0.0f))
        {
            if((params_0.pending_0) != u32(0))
            {
                broke_0 = u32(1);
            }
            else
            {
                broke_0 = broke_1;
            }
            var bi_1 : u32 = bi_0 + u32(256);
            broke_1 = broke_0;
            bi_0 = bi_1;
            continue;
        }
        if((params_0.pending_0) != u32(0))
        {
            broke_0 = broke_1;
            var bi_1 : u32 = bi_0 + u32(256);
            broke_1 = broke_0;
            bi_0 = bi_1;
            continue;
        }
        if((max(contact_force_0(law_1.ids_0.y), contact_force_0(law_1.ids_0.z))) > (materials_0.m_0.data_0[law_1.ids_0.x].strength_0.x * law_1.geom0_0.x * law_1.geom1_0.w))
        {
            dyn_0.js_0.damage_0 = 1.0f;
            dyn_0.js_0.rebar_broken_0 = 1.0f;
            dyn_0.events_0[i32(2)] = max(dyn_0.events_0.z, u32(1));
            bond_dyn_0[bi_0].js_0.damage_0 = dyn_0.js_0.damage_0;
            bond_dyn_0[bi_0].js_0.crush_0 = dyn_0.js_0.crush_0;
            bond_dyn_0[bi_0].js_0.kappa_0 = dyn_0.js_0.kappa_0;
            bond_dyn_0[bi_0].js_0.kappa_c_0 = dyn_0.js_0.kappa_c_0;
            bond_dyn_0[bi_0].js_0.ductility_0 = dyn_0.js_0.ductility_0;
            bond_dyn_0[bi_0].js_0.ductility_c_0 = dyn_0.js_0.ductility_c_0;
            bond_dyn_0[bi_0].js_0.life_0 = dyn_0.js_0.life_0;
            bond_dyn_0[bi_0].js_0.plastic_x_0 = dyn_0.js_0.plastic_x_0;
            bond_dyn_0[bi_0].js_0.plastic_y_0 = dyn_0.js_0.plastic_y_0;
            bond_dyn_0[bi_0].js_0.plastic_t_0 = dyn_0.js_0.plastic_t_0;
            bond_dyn_0[bi_0].js_0.rebar_plastic_0 = dyn_0.js_0.rebar_plastic_0;
            bond_dyn_0[bi_0].js_0.rebar_slip0_0 = dyn_0.js_0.rebar_slip0_0;
            bond_dyn_0[bi_0].js_0.rebar_slip1_0 = dyn_0.js_0.rebar_slip1_0;
            bond_dyn_0[bi_0].js_0.rebar_work_0 = dyn_0.js_0.rebar_work_0;
            bond_dyn_0[bi_0].js_0.rebar_broken_0 = dyn_0.js_0.rebar_broken_0;
            bond_dyn_0[bi_0].js_0.strain_rate_0 = dyn_0.js_0.strain_rate_0;
            bond_dyn_0[bi_0].js_0.governing_stress_0 = dyn_0.js_0.governing_stress_0;
            bond_dyn_0[bi_0].js_0.dissipated_0 = dyn_0.js_0.dissipated_0;
            bond_dyn_0[bi_0].js_0.utilization_0 = dyn_0.js_0.utilization_0;
            bond_dyn_0[bi_0].js_0.mode_0 = dyn_0.js_0.mode_0;
            bond_dyn_0[bi_0].force_lin_0 = dyn_0.force_lin_0;
            bond_dyn_0[bi_0].force_ang_0 = dyn_0.force_ang_0;
            bond_dyn_0[bi_0].sums_0 = dyn_0.sums_0;
            bond_dyn_0[bi_0].comps_0 = dyn_0.comps_0;
            bond_dyn_0[bi_0].events_0 = dyn_0.events_0;
            broke_0 = u32(1);
        }
        else
        {
            broke_0 = broke_1;
        }
        var bi_1 : u32 = bi_0 + u32(256);
        broke_1 = broke_0;
        bi_0 = bi_1;
    }
    g_broke_0[t_0] = broke_1;
    workgroupBarrier();
    broke_0 = u32(128);
    loop
    {
        if(broke_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_0 < broke_0)
        {
            g_broke_0[t_0] = ((g_broke_0[t_0]) | ((g_broke_0[t_0 + broke_0])));
        }
        workgroupBarrier();
        broke_0 = (broke_0 >> (u32(1)));
    }
    if(t_0 != u32(0))
    {
        return;
    }
    status_0[u32(16) + i_0] = (status_0[u32(16) + i_0] | (u32(2)));
    if((g_broke_0[i32(0)]) != u32(0))
    {
        isl_0.info_0[i32(2)] = ((isl_0.info_0[i32(2)]) | (u32(48)));
        islands_0[i_0].range_0 = isl_0.range_0;
        islands_0[i_0].info_0 = isl_0.info_0;
        islands_0[i_0].com_0 = isl_0.com_0;
        islands_0[i_0].inertia0_0 = isl_0.inertia0_0;
        islands_0[i_0].inertia1_0 = isl_0.inertia1_0;
        islands_0[i_0].inertia2_0 = isl_0.inertia2_0;
        islands_0[i_0].inv0_0 = isl_0.inv0_0;
        islands_0[i_0].inv1_0 = isl_0.inv1_0;
        islands_0[i_0].inv2_0 = isl_0.inv2_0;
        islands_0[i_0].wcom_0 = isl_0.wcom_0;
        islands_0[i_0].winv0_0 = isl_0.winv0_0;
        islands_0[i_0].winv1_0 = isl_0.winv1_0;
        islands_0[i_0].winv2_0 = isl_0.winv2_0;
        islands_0[i_0].rotation_0 = isl_0.rotation_0;
        islands_0[i_0].position_0 = isl_0.position_0;
        islands_0[i_0].position_err_0 = isl_0.position_err_0;
        islands_0[i_0].velocity_0 = isl_0.velocity_0;
        islands_0[i_0].velocity_err_0 = isl_0.velocity_err_0;
        islands_0[i_0].angular_velocity_0 = isl_0.angular_velocity_0;
        islands_0[i_0].done_0 = isl_0.done_0;
        islands_0[i_0].coverage_0 = isl_0.coverage_0;
        islands_0[i_0].wide_groups_0 = isl_0.wide_groups_0;
        islands_0[i_0].done_pad_0 = isl_0.done_pad_0;
        islands_0[i_0].probes_0 = isl_0.probes_0;
        islands_0[i_0].energy_0 = isl_0.energy_0;
        islands_0[i_0].rotation_err_0 = isl_0.rotation_err_0;
        islands_0[i_0].momentum_0 = isl_0.momentum_0;
        islands_0[i_0].momentum_err_0 = isl_0.momentum_err_0;
    }
    return;
}

struct Quat_0
{
     w_0 : f32,
     x_0 : f32,
     y_0 : f32,
     z_0 : f32,
};

fn quat_of_0( q_0 : vec4<f32>) -> Quat_0
{
    var r_0 : Quat_0;
    r_0.x_0 = q_0.x;
    r_0.y_0 = q_0.y;
    r_0.z_0 = q_0.z;
    r_0.w_0 = q_0.w;
    return r_0;
}

fn quat_mul_0( a_0 : Quat_0,  o_0 : Quat_0) -> Quat_0
{
    var r_1 : Quat_0;
    r_1.w_0 = a_0.w_0 * o_0.w_0 - a_0.x_0 * o_0.x_0 - a_0.y_0 * o_0.y_0 - a_0.z_0 * o_0.z_0;
    r_1.x_0 = a_0.w_0 * o_0.x_0 + a_0.x_0 * o_0.w_0 + a_0.y_0 * o_0.z_0 - a_0.z_0 * o_0.y_0;
    r_1.y_0 = a_0.w_0 * o_0.y_0 - a_0.x_0 * o_0.z_0 + a_0.y_0 * o_0.w_0 + a_0.z_0 * o_0.x_0;
    r_1.z_0 = a_0.w_0 * o_0.z_0 + a_0.x_0 * o_0.y_0 - a_0.y_0 * o_0.x_0 + a_0.z_0 * o_0.w_0;
    return r_1;
}

fn normalized_0( q_1 : Quat_0) -> Quat_0
{
    var _S7 : f32 = q_1.w_0;
    var _S8 : f32 = q_1.x_0;
    var _S9 : f32 = q_1.y_0;
    var _S10 : f32 = q_1.z_0;
    var n_0 : f32 = sqrt(_S7 * _S7 + _S8 * _S8 + _S9 * _S9 + _S10 * _S10);
    var r_2 : Quat_0;
    r_2.w_0 = q_1.w_0 / n_0;
    r_2.x_0 = q_1.x_0 / n_0;
    r_2.y_0 = q_1.y_0 / n_0;
    r_2.z_0 = q_1.z_0 / n_0;
    return r_2;
}

fn to_zup_0( v_0 : vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(v_0.x, - v_0.z, v_0.y);
}

fn rotate_0( q_2 : Quat_0,  v_1 : vec3<f32>) -> vec3<f32>
{
    var qv_0 : vec3<f32> = vec3<f32>(q_2.x_0, q_2.y_0, q_2.z_0);
    var t_1 : vec3<f32> = cross(qv_0, v_1) * vec3<f32>(2.0f);
    return v_1 + t_1 * vec3<f32>(q_2.w_0) + cross(qv_0, t_1);
}

fn float_bits_0( bits_0 : u32) -> f32
{
    return (bitcast<f32>((bits_0)));
}

var<workgroup> g_lo_0 : array<vec3<f32>, i32(256)>;

var<workgroup> g_hi_0 : array<vec3<f32>, i32(256)>;

var<workgroup> g_a_0 : array<f32, i32(256)>;

var<workgroup> g_idx_0 : array<u32, i32(256)>;

struct Strike_0
{
     isl_1 : Island_0,
     n_1 : u32,
     origin_0 : vec3<f32>,
     q_3 : Quat_0,
     hit_0 : vec3<f32>,
     d_0 : vec3<f32>,
     speed_1 : f32,
     face_0 : f32,
     m_imp_0 : f32,
     e_imp_0 : f32,
     entry_0 : u32,
     entry_k_0 : u32,
     entry_depth_0 : f32,
     from_0 : vec3<f32>,
     rel_0 : vec3<f32>,
     radius_0 : f32,
};

fn find_strike_0( i_1 : u32,  t_2 : u32,  max_chunks_0 : u32,  st_1 : ptr<function, Strike_0>) -> bool
{
    var ax_0 : u32;
    var r_3 : f32;
    var t0_0 : f32;
    var _S11 : Island_std430_0 = islands_0[i_1];
    (*st_1).isl_1.range_0 = islands_0[i_1].range_0;
    (*st_1).isl_1.info_0 = _S11.info_0;
    (*st_1).isl_1.com_0 = _S11.com_0;
    (*st_1).isl_1.inertia0_0 = _S11.inertia0_0;
    (*st_1).isl_1.inertia1_0 = _S11.inertia1_0;
    (*st_1).isl_1.inertia2_0 = _S11.inertia2_0;
    (*st_1).isl_1.inv0_0 = _S11.inv0_0;
    (*st_1).isl_1.inv1_0 = _S11.inv1_0;
    (*st_1).isl_1.inv2_0 = _S11.inv2_0;
    (*st_1).isl_1.wcom_0 = _S11.wcom_0;
    (*st_1).isl_1.winv0_0 = _S11.winv0_0;
    (*st_1).isl_1.winv1_0 = _S11.winv1_0;
    (*st_1).isl_1.winv2_0 = _S11.winv2_0;
    (*st_1).isl_1.rotation_0 = _S11.rotation_0;
    (*st_1).isl_1.position_0 = _S11.position_0;
    (*st_1).isl_1.position_err_0 = _S11.position_err_0;
    (*st_1).isl_1.velocity_0 = _S11.velocity_0;
    (*st_1).isl_1.velocity_err_0 = _S11.velocity_err_0;
    (*st_1).isl_1.angular_velocity_0 = _S11.angular_velocity_0;
    (*st_1).isl_1.done_0 = _S11.done_0;
    (*st_1).isl_1.coverage_0 = _S11.coverage_0;
    (*st_1).isl_1.wide_groups_0 = _S11.wide_groups_0;
    (*st_1).isl_1.done_pad_0 = _S11.done_pad_0;
    (*st_1).isl_1.probes_0 = _S11.probes_0;
    (*st_1).isl_1.energy_0 = _S11.energy_0;
    (*st_1).isl_1.rotation_err_0 = _S11.rotation_err_0;
    (*st_1).isl_1.momentum_0 = _S11.momentum_0;
    (*st_1).isl_1.momentum_err_0 = _S11.momentum_err_0;
    (*st_1).n_1 = u32(0);
    var _S12 : vec3<f32> = vec3<f32>(0.0f);
    (*st_1).origin_0 = _S12;
    (*st_1).q_3 = quat_of_0(vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f));
    (*st_1).hit_0 = _S12;
    (*st_1).d_0 = _S12;
    (*st_1).speed_1 = 0.0f;
    (*st_1).face_0 = 0.0f;
    (*st_1).m_imp_0 = 0.0f;
    (*st_1).e_imp_0 = 0.0f;
    (*st_1).entry_0 = u32(4294967295);
    (*st_1).entry_k_0 = u32(4294967295);
    (*st_1).entry_depth_0 = 0.0f;
    (*st_1).from_0 = _S12;
    (*st_1).rel_0 = _S12;
    (*st_1).radius_0 = 0.0f;
    var isl_2 : Island_std430_0 = islands_0[i_1];
    var _S13 : u32 = islands_0[i_1].range_0.x;
    var n_2 : u32 = group_uniform_0(islands_0[i_1].range_0.y - _S13, t_2);
    var _S14 : bool;
    if((islands_0[i_1].range_0.z) >= (islands_0[i_1].range_0.w))
    {
        _S14 = true;
    }
    else
    {
        _S14 = n_2 > max_chunks_0;
    }
    if(_S14)
    {
        _S14 = true;
    }
    else
    {
        _S14 = (params_0.impactor_count_0) == u32(0);
    }
    var c_0 : u32;
    if(_S14)
    {
        c_0 = u32(1);
    }
    else
    {
        c_0 = u32(0);
    }
    var _S15 : u32 = group_uniform_0(c_0, t_2);
    if(_S15 != u32(0))
    {
        return false;
    }
    var _S16 : u32 = u32(3) * i_1;
    var body_0 : u32 = (bitcast<u32>((ctab_0[params_0.engine_base_0 + _S16].x)));
    var _S17 : Quat_0 = quat_of_0(isl_2.rotation_0);
    var _S18 : vec3<f32> = isl_2.position_0.xyz;
    var _S19 : vec3<f32> = isl_2.velocity_0.xyz;
    var origin_1 : vec3<f32>;
    var v_2 : vec3<f32>;
    var q_4 : Quat_0;
    if(body_0 != u32(4294967295))
    {
        var p_inv_0 : Quat_0 = quat_of_0(ctab_0[params_0.engine_base_0 + _S16 + u32(1)]);
        var com_1 : vec3<f32> = ctab_0[params_0.engine_base_0 + _S16 + u32(2)].xyz;
        var b_0 : EngineBody_std430_0 = bodies_0[body_0];
        var a_inv_0 : Quat_0;
        a_inv_0.w_0 = 0.70710676908493042f;
        a_inv_0.x_0 = 0.70710676908493042f;
        a_inv_0.y_0 = 0.0f;
        a_inv_0.z_0 = 0.0f;
        var q_5 : Quat_0 = normalized_0(quat_mul_0(quat_mul_0(a_inv_0, quat_of_0(b_0.rot_0)), p_inv_0));
        var _S20 : vec3<f32> = to_zup_0(b_0.vel_sleep_0.xyz);
        origin_1 = to_zup_0(b_0.pos_inv_mass_0.xyz) - rotate_0(q_5, com_1);
        q_4 = q_5;
        v_2 = _S20;
    }
    else
    {
        origin_1 = _S18;
        q_4 = _S17;
        v_2 = _S19;
    }
    var inf_0 : f32 = float_bits_0(u32(2139095040));
    var _S21 : vec3<f32> = vec3<f32>(- inf_0);
    var lo_0 : vec3<f32> = vec3<f32>(inf_0);
    var hi_0 : vec3<f32> = _S21;
    c_0 = t_2;
    loop
    {
        if(c_0 < n_2)
        {
        }
        else
        {
            break;
        }
        var k_0 : u32 = _S13 + c_0;
        var p_0 : vec3<f32> = origin_1 + rotate_0(q_4, chunks_0[k_0].center_0.xyz + state_0[u32(4) * k_0].xyz);
        var _S22 : vec3<f32> = min(lo_0, p_0 - vec3<f32>(chunks_0[k_0].half_0.w));
        var _S23 : vec3<f32> = max(hi_0, p_0 + vec3<f32>(chunks_0[k_0].half_0.w));
        var c_1 : u32 = c_0 + u32(256);
        lo_0 = _S22;
        hi_0 = _S23;
        c_0 = c_1;
    }
    g_lo_0[t_2] = lo_0;
    g_hi_0[t_2] = hi_0;
    workgroupBarrier();
    var s_0 : u32 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_2 < s_0)
        {
            var _S24 : u32 = t_2 + s_0;
            g_lo_0[t_2] = min(g_lo_0[t_2], g_lo_0[_S24]);
            g_hi_0[t_2] = max(g_hi_0[t_2], g_hi_0[_S24]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var _S25 : vec3<f32> = g_lo_0[i32(0)];
    var _S26 : vec3<f32> = g_hi_0[i32(0)];
    var _S27 : u32 = (bitcast<u32>((fast_0[i32(0)].x)));
    var hit_k_0 : u32 = u32(4294967295);
    var hit_1 : vec3<f32> = _S12;
    var d_1 : vec3<f32> = _S12;
    var speed_2 : f32 = 0.0f;
    var face_1 : f32 = 0.0f;
    var m_imp_1 : f32 = 0.0f;
    var e_imp_1 : f32 = 0.0f;
    var hit_c_0 : vec3<f32> = _S12;
    var hit_rel_0 : vec3<f32> = _S12;
    var p_radius_0 : f32 = 0.0f;
    var k_1 : u32 = u32(0);
    loop
    {
        if(k_1 < (params_0.impactor_count_0 + _S27))
        {
            _S14 = hit_k_0 == u32(4294967295);
        }
        else
        {
            _S14 = false;
        }
        if(_S14)
        {
        }
        else
        {
            break;
        }
        var rel_1 : vec3<f32>;
        var c_2 : vec3<f32>;
        var info_2 : vec4<f32>;
        if(k_1 < (params_0.impactor_count_0))
        {
            var _S28 : u32 = u32(2) * k_1;
            var a_1 : vec4<f32> = impactors_0[_S28];
            var b_1 : EngineBody_std430_0 = bodies_0[(bitcast<u32>((a_1.x)))];
            var _S29 : vec3<f32> = to_zup_0(b_1.pos_inv_mass_0.xyz);
            var _S30 : f32 = a_1.y;
            var _S31 : vec4<f32> = vec4<f32>(a_1.z, a_1.w, impactors_0[_S28 + u32(1)].x, 0.0f);
            rel_1 = to_zup_0(b_1.vel_sleep_0.xyz) - v_2;
            r_3 = _S30;
            c_2 = _S29;
            info_2 = _S31;
        }
        else
        {
            var f_0 : u32 = u32(1) + u32(3) * (k_1 - params_0.impactor_count_0);
            var _S32 : u32 = f_0 + u32(1);
            if((bitcast<u32>((fast_0[_S32].w))) == i_1)
            {
                k_1 = k_1 + u32(1);
                continue;
            }
            var _S33 : vec3<f32> = fast_0[f_0].xyz;
            var _S34 : f32 = fast_0[f_0].w;
            var _S35 : vec4<f32> = fast_0[f_0 + u32(2)];
            rel_1 = fast_0[_S32].xyz - v_2;
            r_3 = _S34;
            c_2 = _S33;
            info_2 = _S35;
        }
        var sp_0 : f32 = length(rel_1);
        if(!(sp_0 > (params_0.speed_0)))
        {
            k_1 = k_1 + u32(1);
            continue;
        }
        var _S36 : vec3<f32> = vec3<f32>(r_3);
        var _S37 : vec3<f32> = _S25 - _S36;
        var _S38 : vec3<f32> = _S26 + _S36;
        var t0_1 : f32 = 0.0f;
        var t1_1 : f32 = params_0.dt_0;
        ax_0 = u32(0);
        loop
        {
            if(ax_0 < u32(3))
            {
            }
            else
            {
                break;
            }
            var _S39 : vec3<f32> = rel_1;
            var _S40 : u32 = ax_0;
            if((abs(rel_1[ax_0])) < 9.999999960041972e-13f)
            {
                var _S41 : vec3<f32> = c_2;
                var _S42 : u32 = ax_0;
                var _S43 : bool;
                if((c_2[ax_0]) < (_S37[ax_0]))
                {
                    _S43 = true;
                }
                else
                {
                    _S43 = (_S41[_S42]) > (_S38[ax_0]);
                }
                if(_S43)
                {
                    t0_0 = inf_0;
                }
                else
                {
                    t0_0 = t0_1;
                }
                t0_1 = t0_0;
                ax_0 = ax_0 + u32(1);
                continue;
            }
            var ta_0 : f32 = (_S37[ax_0] - c_2[ax_0]) / _S39[_S40];
            var tb_0 : f32 = (_S38[ax_0] - c_2[ax_0]) / _S39[_S40];
            var _S44 : f32 = min(t1_1, max(ta_0, tb_0));
            t0_1 = max(t0_1, min(ta_0, tb_0));
            t1_1 = _S44;
            ax_0 = ax_0 + u32(1);
        }
        var hit_k_1 : u32;
        var hit_2 : vec3<f32>;
        var d_2 : vec3<f32>;
        var hit_c_1 : vec3<f32>;
        var hit_rel_1 : vec3<f32>;
        var face_2 : f32;
        var m_imp_2 : f32;
        var e_imp_2 : f32;
        var p_radius_1 : f32;
        if(t0_1 <= t1_1)
        {
            var _S45 : vec3<f32> = c_2 + rel_1 * vec3<f32>(t0_1);
            var _S46 : vec3<f32> = rel_1 / vec3<f32>(sp_0);
            var _S47 : f32 = info_2.x;
            var _S48 : f32 = info_2.y;
            var _S49 : f32 = info_2.z;
            hit_k_1 = k_1;
            hit_2 = _S45;
            d_2 = _S46;
            t0_0 = sp_0;
            face_2 = _S47;
            m_imp_2 = _S48;
            e_imp_2 = _S49;
            hit_c_1 = c_2;
            hit_rel_1 = rel_1;
            p_radius_1 = r_3;
        }
        else
        {
            hit_k_1 = hit_k_0;
            hit_2 = hit_1;
            d_2 = d_1;
            t0_0 = speed_2;
            face_2 = face_1;
            m_imp_2 = m_imp_1;
            e_imp_2 = e_imp_1;
            hit_c_1 = hit_c_0;
            hit_rel_1 = hit_rel_0;
            p_radius_1 = p_radius_0;
        }
        hit_k_0 = hit_k_1;
        hit_1 = hit_2;
        d_1 = d_2;
        speed_2 = t0_0;
        face_1 = face_2;
        m_imp_1 = m_imp_2;
        e_imp_1 = e_imp_2;
        hit_c_0 = hit_c_1;
        hit_rel_0 = hit_rel_1;
        p_radius_0 = p_radius_1;
        k_1 = k_1 + u32(1);
    }
    if(hit_k_0 == u32(4294967295))
    {
        c_0 = u32(1);
    }
    else
    {
        c_0 = u32(0);
    }
    var _S50 : u32 = group_uniform_0(c_0, t_2);
    if(_S50 != u32(0))
    {
        return false;
    }
    r_3 = inf_0;
    ax_0 = u32(4294967295);
    c_0 = t_2;
    loop
    {
        if(c_0 < n_2)
        {
        }
        else
        {
            break;
        }
        var k_2 : u32 = _S13 + c_0;
        var r_4 : vec3<f32> = origin_1 + rotate_0(q_4, chunks_0[k_2].center_0.xyz + state_0[u32(4) * k_2].xyz) - hit_1;
        var along_0 : f32 = dot(r_4, d_1);
        var dist_0 : f32 = length(r_4 - d_1 * vec3<f32>(along_0)) + abs(along_0) * 0.00100000004749745f;
        if(dist_0 < r_3)
        {
            r_3 = dist_0;
            ax_0 = c_0;
        }
        c_0 = c_0 + u32(256);
    }
    g_a_0[t_2] = r_3;
    g_idx_0[t_2] = ax_0;
    workgroupBarrier();
    s_0 = u32(128);
    loop
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_2 < s_0)
        {
            var _S51 : u32 = t_2 + s_0;
            if((g_a_0[_S51]) < (g_a_0[t_2]))
            {
                _S14 = true;
            }
            else
            {
                if((g_a_0[_S51]) == (g_a_0[t_2]))
                {
                    _S14 = (g_idx_0[_S51]) < (g_idx_0[t_2]);
                }
                else
                {
                    _S14 = false;
                }
            }
        }
        else
        {
            _S14 = false;
        }
        if(_S14)
        {
            var _S52 : u32 = t_2 + s_0;
            g_a_0[t_2] = g_a_0[_S52];
            g_idx_0[t_2] = g_idx_0[_S52];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var entry_1 : u32 = g_idx_0[i32(0)];
    var entry_k_1 : u32 = _S13 + g_idx_0[i32(0)];
    var entry_depth_1 : f32 = dot(origin_1 + rotate_0(q_4, chunks_0[entry_k_1].center_0.xyz + state_0[u32(4) * entry_k_1].xyz) - hit_1, d_1);
    (*st_1).isl_1.range_0 = isl_2.range_0;
    (*st_1).isl_1.info_0 = isl_2.info_0;
    (*st_1).isl_1.com_0 = isl_2.com_0;
    (*st_1).isl_1.inertia0_0 = isl_2.inertia0_0;
    (*st_1).isl_1.inertia1_0 = isl_2.inertia1_0;
    (*st_1).isl_1.inertia2_0 = isl_2.inertia2_0;
    (*st_1).isl_1.inv0_0 = isl_2.inv0_0;
    (*st_1).isl_1.inv1_0 = isl_2.inv1_0;
    (*st_1).isl_1.inv2_0 = isl_2.inv2_0;
    (*st_1).isl_1.wcom_0 = isl_2.wcom_0;
    (*st_1).isl_1.winv0_0 = isl_2.winv0_0;
    (*st_1).isl_1.winv1_0 = isl_2.winv1_0;
    (*st_1).isl_1.winv2_0 = isl_2.winv2_0;
    (*st_1).isl_1.rotation_0 = isl_2.rotation_0;
    (*st_1).isl_1.position_0 = isl_2.position_0;
    (*st_1).isl_1.position_err_0 = isl_2.position_err_0;
    (*st_1).isl_1.velocity_0 = isl_2.velocity_0;
    (*st_1).isl_1.velocity_err_0 = isl_2.velocity_err_0;
    (*st_1).isl_1.angular_velocity_0 = isl_2.angular_velocity_0;
    (*st_1).isl_1.done_0 = isl_2.done_0;
    (*st_1).isl_1.coverage_0 = isl_2.coverage_0;
    (*st_1).isl_1.wide_groups_0 = isl_2.wide_groups_0;
    (*st_1).isl_1.done_pad_0 = isl_2.done_pad_0;
    (*st_1).isl_1.probes_0 = isl_2.probes_0;
    (*st_1).isl_1.energy_0 = isl_2.energy_0;
    (*st_1).isl_1.rotation_err_0 = isl_2.rotation_err_0;
    (*st_1).isl_1.momentum_0 = isl_2.momentum_0;
    (*st_1).isl_1.momentum_err_0 = isl_2.momentum_err_0;
    (*st_1).n_1 = n_2;
    (*st_1).origin_0 = origin_1;
    (*st_1).q_3 = q_4;
    (*st_1).hit_0 = hit_1;
    (*st_1).d_0 = d_1;
    (*st_1).speed_1 = speed_2;
    (*st_1).face_0 = face_1;
    (*st_1).m_imp_0 = m_imp_1;
    (*st_1).e_imp_0 = e_imp_1;
    (*st_1).entry_0 = entry_1;
    (*st_1).entry_k_0 = entry_k_1;
    (*st_1).entry_depth_0 = entry_depth_1;
    (*st_1).from_0 = hit_c_0;
    (*st_1).rel_0 = hit_rel_0;
    (*st_1).radius_0 = p_radius_0;
    return true;
}

@compute
@workgroup_size(256, 1, 1)
fn threshold_predict(@builtin(workgroup_id) group_1 : vec3<u32>, @builtin(local_invocation_id) thread_2 : vec3<u32>)
{
    var ax_1 : u32;
    var t_in_0 : f32;
    var i_2 : u32 = group_1.x;
    var t_3 : u32 = thread_2.x;
    if(i_2 >= (params_0.island_count_0))
    {
        return;
    }
    var _S53 : u32 = group_uniform_0((bitcast<u32>((fast_0[i32(0)].y))), t_3);
    if(_S53 == u32(0))
    {
        if(t_3 == u32(0))
        {
            status_0[u32(16) + i_2] = (status_0[u32(16) + i_2] & (u32(4294967287)));
        }
        return;
    }
    var st_2 : Strike_0;
    var found_0 : bool = find_strike_0(i_2, t_3, u32(4294967295), &(st_2));
    var _S54 : u32 = group_uniform_0(st_2.n_1, t_3);
    var c_3 : u32;
    if(found_0)
    {
        c_3 = u32(1);
    }
    else
    {
        c_3 = u32(0);
    }
    var _S55 : u32 = group_uniform_0(c_3, t_3);
    if(_S55 == u32(0))
    {
        if(t_3 == u32(0))
        {
            status_0[u32(16) + i_2] = (status_0[u32(16) + i_2] & (u32(4294967287)));
        }
        return;
    }
    var inf_1 : f32 = float_bits_0(u32(2139095040));
    var first_0 : f32 = inf_1;
    c_3 = t_3;
    loop
    {
        if(c_3 < _S54)
        {
        }
        else
        {
            break;
        }
        var k_3 : u32 = st_2.isl_1.range_0.x + c_3;
        var _S56 : vec3<f32> = st_2.origin_0 + rotate_0(st_2.q_3, chunks_0[k_3].center_0.xyz + state_0[u32(4) * k_3].xyz);
        t_in_0 = 0.0f;
        var t_out_0 : f32 = params_0.dt_0;
        ax_1 = u32(0);
        loop
        {
            if(ax_1 < u32(3))
            {
            }
            else
            {
                break;
            }
            var axis_0 : vec3<f32> = rotate_0(st_2.q_3, vec3<f32>(chunks_0[k_3].crot0_0[ax_1], chunks_0[k_3].crot1_0[ax_1], chunks_0[k_3].crot2_0[ax_1]));
            var o_1 : f32 = dot(st_2.from_0 - _S56, axis_0);
            var dv_0 : f32 = dot(st_2.rel_0, axis_0);
            var ext_0 : f32 = chunks_0[k_3].half_0[ax_1] + st_2.radius_0;
            if((abs(dv_0)) < 9.999999960041972e-13f)
            {
                var t_in_1 : f32;
                if((abs(o_1)) > ext_0)
                {
                    t_in_1 = inf_1;
                }
                else
                {
                    t_in_1 = t_in_0;
                }
                t_in_0 = t_in_1;
                ax_1 = ax_1 + u32(1);
                continue;
            }
            var ta_1 : f32 = (- ext_0 - o_1) / dv_0;
            var tb_1 : f32 = (ext_0 - o_1) / dv_0;
            var _S57 : f32 = min(t_out_0, max(ta_1, tb_1));
            t_in_0 = max(t_in_0, min(ta_1, tb_1));
            t_out_0 = _S57;
            ax_1 = ax_1 + u32(1);
        }
        if(t_in_0 <= t_out_0)
        {
            first_0 = min(first_0, t_in_0);
        }
        c_3 = c_3 + u32(256);
    }
    g_a_0[t_3] = first_0;
    workgroupBarrier();
    ax_1 = u32(128);
    loop
    {
        if(ax_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_3 < ax_1)
        {
            g_a_0[t_3] = min(g_a_0[t_3], g_a_0[t_3 + ax_1]);
        }
        workgroupBarrier();
        ax_1 = (ax_1 >> (u32(1)));
    }
    var t_in_2 : f32 = g_a_0[i32(0)];
    if((g_a_0[i32(0)]) < inf_1)
    {
        c_3 = u32(0);
    }
    else
    {
        c_3 = u32(1);
    }
    var _S58 : u32 = group_uniform_0(c_3, t_3);
    if(_S58 != u32(0))
    {
        if(t_3 == u32(0))
        {
            status_0[u32(16) + i_2] = (status_0[u32(16) + i_2] & (u32(4294967287)));
        }
        return;
    }
    var _S59 : vec3<f32> = st_2.from_0 + st_2.rel_0 * vec3<f32>(t_in_2) + st_2.d_0 * vec3<f32>(st_2.radius_0);
    var m_isl_0 : f32 = st_2.isl_1.com_0.w;
    if((((st_2.isl_1.info_0.x) & (u32(1)))) != u32(0))
    {
        t_in_0 = st_2.m_imp_0;
    }
    else
    {
        t_in_0 = st_2.m_imp_0 * m_isl_0 / (st_2.m_imp_0 + m_isl_0);
    }
    var modulus_0 : f32 = 1.0f / (1.0f / st_2.e_imp_0 + 1.0f / chunks_0[st_2.entry_k_0].cmat_0.x);
    var _S60 : f32 = t_in_0 * st_2.speed_1 / (2.86999988555908203f * pow(t_in_0 * t_in_0 / (max(st_2.face_0, 0.00100000004749745f) * modulus_0 * modulus_0 * max(st_2.speed_1, 0.00100000004749745f)), 0.20000000298023224f));
    var _S61 : u32 = st_2.isl_1.range_0.z + t_3;
    var broke_2 : u32 = u32(0);
    var bi_2 : u32 = _S61;
    loop
    {
        if(bi_2 < (st_2.isl_1.range_0.w))
        {
        }
        else
        {
            break;
        }
        var law_2 : JointBond_std430_0 = bond_static_0[bi_2].law_0;
        var _S62 : JointState_0 = JointState_0( bond_dyn_0[bi_2].js_0.damage_0, bond_dyn_0[bi_2].js_0.crush_0, bond_dyn_0[bi_2].js_0.kappa_0, bond_dyn_0[bi_2].js_0.kappa_c_0, bond_dyn_0[bi_2].js_0.ductility_0, bond_dyn_0[bi_2].js_0.ductility_c_0, bond_dyn_0[bi_2].js_0.life_0, bond_dyn_0[bi_2].js_0.plastic_x_0, bond_dyn_0[bi_2].js_0.plastic_y_0, bond_dyn_0[bi_2].js_0.plastic_t_0, bond_dyn_0[bi_2].js_0.rebar_plastic_0, bond_dyn_0[bi_2].js_0.rebar_slip0_0, bond_dyn_0[bi_2].js_0.rebar_slip1_0, bond_dyn_0[bi_2].js_0.rebar_work_0, bond_dyn_0[bi_2].js_0.rebar_broken_0, bond_dyn_0[bi_2].js_0.strain_rate_0, bond_dyn_0[bi_2].js_0.governing_stress_0, bond_dyn_0[bi_2].js_0.dissipated_0, bond_dyn_0[bi_2].js_0.utilization_0, bond_dyn_0[bi_2].js_0.mode_0 );
        var dyn_1 : BondDyn_0;
        dyn_1.js_0 = _S62;
        dyn_1.force_lin_0 = bond_dyn_0[bi_2].force_lin_0;
        dyn_1.force_ang_0 = bond_dyn_0[bi_2].force_ang_0;
        dyn_1.sums_0 = bond_dyn_0[bi_2].sums_0;
        dyn_1.comps_0 = bond_dyn_0[bi_2].comps_0;
        dyn_1.events_0 = bond_dyn_0[bi_2].events_0;
        if(!connected_0(dyn_1.js_0, (bond_static_0[bi_2].law_0.stiff1_0.w) != 0.0f))
        {
            bi_2 = bi_2 + u32(256);
            continue;
        }
        var _S63 : u32 = law_2.ids_0.y;
        var _S64 : u32 = law_2.ids_0.z;
        var dist_1 : f32 = length(vec3<f32>(0.5f) * (st_2.origin_0 + rotate_0(st_2.q_3, chunks_0[_S63].center_0.xyz + state_0[u32(4) * _S63].xyz) + (st_2.origin_0 + rotate_0(st_2.q_3, chunks_0[_S64].center_0.xyz + state_0[u32(4) * _S64].xyz))) - _S59);
        var _S65 : f32 = law_2.geom0_0.x;
        var broke_3 : u32;
        if((_S60 * min(1.0f, _S65 / (6.28318548202514648f * max(dist_1 * dist_1, 9.999999960041972e-13f)))) > (materials_0.m_0.data_0[law_2.ids_0.x].strength_0.x * _S65 * law_2.geom1_0.w))
        {
            dyn_1.js_0.damage_0 = 1.0f;
            dyn_1.js_0.rebar_broken_0 = 1.0f;
            dyn_1.events_0[i32(2)] = max(dyn_1.events_0.z, u32(1));
            bond_dyn_0[bi_2].js_0.damage_0 = dyn_1.js_0.damage_0;
            bond_dyn_0[bi_2].js_0.crush_0 = dyn_1.js_0.crush_0;
            bond_dyn_0[bi_2].js_0.kappa_0 = dyn_1.js_0.kappa_0;
            bond_dyn_0[bi_2].js_0.kappa_c_0 = dyn_1.js_0.kappa_c_0;
            bond_dyn_0[bi_2].js_0.ductility_0 = dyn_1.js_0.ductility_0;
            bond_dyn_0[bi_2].js_0.ductility_c_0 = dyn_1.js_0.ductility_c_0;
            bond_dyn_0[bi_2].js_0.life_0 = dyn_1.js_0.life_0;
            bond_dyn_0[bi_2].js_0.plastic_x_0 = dyn_1.js_0.plastic_x_0;
            bond_dyn_0[bi_2].js_0.plastic_y_0 = dyn_1.js_0.plastic_y_0;
            bond_dyn_0[bi_2].js_0.plastic_t_0 = dyn_1.js_0.plastic_t_0;
            bond_dyn_0[bi_2].js_0.rebar_plastic_0 = dyn_1.js_0.rebar_plastic_0;
            bond_dyn_0[bi_2].js_0.rebar_slip0_0 = dyn_1.js_0.rebar_slip0_0;
            bond_dyn_0[bi_2].js_0.rebar_slip1_0 = dyn_1.js_0.rebar_slip1_0;
            bond_dyn_0[bi_2].js_0.rebar_work_0 = dyn_1.js_0.rebar_work_0;
            bond_dyn_0[bi_2].js_0.rebar_broken_0 = dyn_1.js_0.rebar_broken_0;
            bond_dyn_0[bi_2].js_0.strain_rate_0 = dyn_1.js_0.strain_rate_0;
            bond_dyn_0[bi_2].js_0.governing_stress_0 = dyn_1.js_0.governing_stress_0;
            bond_dyn_0[bi_2].js_0.dissipated_0 = dyn_1.js_0.dissipated_0;
            bond_dyn_0[bi_2].js_0.utilization_0 = dyn_1.js_0.utilization_0;
            bond_dyn_0[bi_2].js_0.mode_0 = dyn_1.js_0.mode_0;
            bond_dyn_0[bi_2].force_lin_0 = dyn_1.force_lin_0;
            bond_dyn_0[bi_2].force_ang_0 = dyn_1.force_ang_0;
            bond_dyn_0[bi_2].sums_0 = dyn_1.sums_0;
            bond_dyn_0[bi_2].comps_0 = dyn_1.comps_0;
            bond_dyn_0[bi_2].events_0 = dyn_1.events_0;
            broke_3 = u32(1);
        }
        else
        {
            broke_3 = broke_2;
        }
        broke_2 = broke_3;
        bi_2 = bi_2 + u32(256);
    }
    g_idx_0[t_3] = broke_2;
    workgroupBarrier();
    ax_1 = u32(128);
    loop
    {
        if(ax_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(t_3 < ax_1)
        {
            g_idx_0[t_3] = ((g_idx_0[t_3]) | ((g_idx_0[t_3 + ax_1])));
        }
        workgroupBarrier();
        ax_1 = (ax_1 >> (u32(1)));
    }
    if(t_3 != u32(0))
    {
        return;
    }
    if((g_idx_0[i32(0)]) != u32(0))
    {
        status_0[u32(16) + i_2] = (status_0[u32(16) + i_2] | (u32(8)));
        status_0[u32(1)] = u32(1);
    }
    else
    {
        status_0[u32(16) + i_2] = (status_0[u32(16) + i_2] & (u32(4294967287)));
    }
    return;
}

