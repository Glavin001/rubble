struct ThresholdParams_std140_0
{
    @align(16) island_count_0 : u32,
    @align(4) eload_base_0 : u32,
    @align(8) pad0_0 : u32,
    @align(4) pad1_0 : u32,
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
            bi_0 = bi_0 + u32(256);
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
        broke_1 = broke_0;
        bi_0 = bi_0 + u32(256);
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

