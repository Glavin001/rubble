#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct FeatParams_0
{
    uint impactor_count_0;
    uint frame_0;
    uint layout_bonds_0;
    uint rows_0;
    float dt_0;
    float speed_0;
    float restitution_0;
    float contact_friction_0;
    float4 gravity_0;
    uint4 bases_0;
    uint4 bases2_0;
    uint4 caps_0;
    uint count_only_0;
    uint max_nodes_0;
    uint layout_cap_0;
    uint max_bonds_cap_0;
    uint prev_frame_0;
    uint engine_base_0;
    uint chunk_count_0;
    uint pad1_0;
};

struct Island_natural_0
{
    packed_uint4 range_0;
    packed_uint4 info_0;
    packed_float4 com_0;
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
    packed_float4 wcom_0;
    packed_float4 winv0_0;
    packed_float4 winv1_0;
    packed_float4 winv2_0;
    packed_float4 rotation_0;
    packed_float4 position_0;
    packed_float4 position_err_0;
    packed_float4 velocity_0;
    packed_float4 velocity_err_0;
    packed_float4 angular_velocity_0;
    uint done_0;
    uint coverage_0;
    uint wide_groups_0;
    uint done_pad_0;
    packed_uint4 probes_0;
    packed_float4 energy_0;
    packed_float4 rotation_err_0;
    packed_float4 momentum_0;
    packed_float4 momentum_err_0;
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

struct BondStatic_natural_0
{
    packed_float4 t1_0;
    packed_float4 t2_0;
    packed_float4 normal_0;
    packed_float4 ra_0;
    packed_float4 rb_0;
    packed_float4 centroid_0;
    packed_float4 c_lin_0;
    packed_float4 c_ang_0;
    JointBond_natural_0 law_0;
};

struct ChunkStatic_natural_0
{
    packed_float4 center_0;
    packed_float4 inertia0_1;
    packed_float4 inertia1_1;
    packed_float4 inertia2_1;
    packed_float4 inv0_1;
    packed_float4 inv1_1;
    packed_float4 inv2_1;
    packed_float4 scale_0;
    packed_uint4 info_1;
    packed_uint4 load_range_0;
    packed_float4 half_0;
    packed_float4 crot0_0;
    packed_float4 crot1_0;
    packed_float4 crot2_0;
    packed_float4 cmat_0;
    packed_uint4 cinfo_0;
};

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

struct BondDyn_natural_0
{
    JointState_0 js_0;
    packed_float4 force_lin_0;
    packed_float4 force_ang_0;
    packed_float4 sums_0;
    packed_float4 comps_0;
    packed_uint4 events_0;
};

struct KernelContext_0
{
    FeatParams_0 constant* params_0;
    uint device* bmap_0;
    uint device* counters_0;
    Island_natural_0 device* islands_0;
    BondStatic_natural_0 device* bond_static_0;
    ChunkStatic_natural_0 device* chunks_0;
    uint device* sinc_0;
    packed_uint4 device* slook_0;
    BondDyn_natural_0 device* bond_dyn_0;
    uint device* ctl_0;
    float device* work_0;
    packed_float4 device* nodes_0;
    packed_uint2 device* key_node_0;
    packed_float4 device* sc_0;
    packed_float4 device* bonds_0;
    packed_float4 device* sb_0;
    packed_uint4 device* bedge_0;
    packed_float4 device* calls_0;
    packed_float4 device* imps_0;
    packed_uint4 device* cisl_0;
    packed_float4 device* cislf_0;
    packed_float4 device* contacts_0;
    uint device* fctl_0;
    uint device* edges_0;
    uint device* cols_0;
    float device* feat_0;
    packed_float4 device* ctab_0;
    packed_float4 device* bodies_0;
    packed_float4 device* state_0;
    array<array<float4, int(64)>, int(2)> threadgroup* g_red_0;
    array<float, int(64)> threadgroup* g_mv_0;
    array<uint, int(64)> threadgroup* g_mi_0;
    array<uint, int(4096)> threadgroup* g_plug_0;
    uint threadgroup* g_flag_0;
    array<uint, int(64)> threadgroup* g_scan_0;
    float4 threadgroup* g_scalars_0;
};

[[kernel]] void feat_unmap(uint3 id_0 [[thread_position_in_grid]], FeatParams_0 constant* params_1 [[buffer(0)]], uint device* bmap_1 [[buffer(11)]], uint device* counters_1 [[buffer(28)]], Island_natural_0 device* islands_1 [[buffer(24)]], BondStatic_natural_0 device* bond_static_1 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(13)]], uint device* sinc_1 [[buffer(9)]], packed_uint4 device* slook_1 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(15)]], uint device* ctl_1 [[buffer(1)]], float device* work_1 [[buffer(18)]], packed_float4 device* nodes_1 [[buffer(3)]], packed_uint2 device* key_node_1 [[buffer(12)]], packed_float4 device* sc_1 [[buffer(7)]], packed_float4 device* bonds_1 [[buffer(4)]], packed_float4 device* sb_1 [[buffer(8)]], packed_uint4 device* bedge_1 [[buffer(22)]], packed_float4 device* calls_1 [[buffer(2)]], packed_float4 device* imps_1 [[buffer(16)]], packed_uint4 device* cisl_1 [[buffer(5)]], packed_float4 device* cislf_1 [[buffer(6)]], packed_float4 device* contacts_1 [[buffer(17)]], uint device* fctl_1 [[buffer(23)]], uint device* edges_1 [[buffer(20)]], uint device* cols_1 [[buffer(21)]], float device* feat_1 [[buffer(19)]], packed_float4 device* ctab_1 [[buffer(25)]], packed_float4 device* bodies_1 [[buffer(26)]], packed_float4 device* state_1 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->params_0 = params_1;
    (&kernelContext_0)->bmap_0 = bmap_1;
    (&kernelContext_0)->counters_0 = counters_1;
    (&kernelContext_0)->islands_0 = islands_1;
    (&kernelContext_0)->bond_static_0 = bond_static_1;
    (&kernelContext_0)->chunks_0 = chunks_1;
    (&kernelContext_0)->sinc_0 = sinc_1;
    (&kernelContext_0)->slook_0 = slook_1;
    (&kernelContext_0)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_0)->ctl_0 = ctl_1;
    (&kernelContext_0)->work_0 = work_1;
    (&kernelContext_0)->nodes_0 = nodes_1;
    (&kernelContext_0)->key_node_0 = key_node_1;
    (&kernelContext_0)->sc_0 = sc_1;
    (&kernelContext_0)->bonds_0 = bonds_1;
    (&kernelContext_0)->sb_0 = sb_1;
    (&kernelContext_0)->bedge_0 = bedge_1;
    (&kernelContext_0)->calls_0 = calls_1;
    (&kernelContext_0)->imps_0 = imps_1;
    (&kernelContext_0)->cisl_0 = cisl_1;
    (&kernelContext_0)->cislf_0 = cislf_1;
    (&kernelContext_0)->contacts_0 = contacts_1;
    (&kernelContext_0)->fctl_0 = fctl_1;
    (&kernelContext_0)->edges_0 = edges_1;
    (&kernelContext_0)->cols_0 = cols_1;
    (&kernelContext_0)->feat_0 = feat_1;
    (&kernelContext_0)->ctab_0 = ctab_1;
    (&kernelContext_0)->bodies_0 = bodies_1;
    (&kernelContext_0)->state_0 = state_1;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_1;
    (&kernelContext_0)->g_red_0 = &g_red_1;
    threadgroup array<float, int(64)> g_mv_1;
    (&kernelContext_0)->g_mv_0 = &g_mv_1;
    threadgroup array<uint, int(64)> g_mi_1;
    (&kernelContext_0)->g_mi_0 = &g_mi_1;
    threadgroup array<uint, int(4096)> g_plug_1;
    (&kernelContext_0)->g_plug_0 = &g_plug_1;
    threadgroup uint g_flag_1;
    (&kernelContext_0)->g_flag_0 = &g_flag_1;
    threadgroup array<uint, int(64)> g_scan_1;
    (&kernelContext_0)->g_scan_0 = &g_scan_1;
    threadgroup float4 g_scalars_1;
    (&kernelContext_0)->g_scalars_0 = &g_scalars_1;
    uint _S1 = id_0.x;
    if(_S1 < (params_1->rows_0))
    {
        *((&kernelContext_0)->bmap_0+((&kernelContext_0)->params_0->layout_cap_0 + _S1)) = 4294967295U;
    }
    return;
}

uint chunk_row_of_0(uint layout_chunk_0, KernelContext_0 thread* kernelContext_1)
{
    uint4 _S2 = uint4((kernelContext_1->chunks_0+layout_chunk_0)->info_1) ;
    return kernelContext_1->sinc_0[2U + _S2.z] + _S2.w;
}

uint bond_row_lookup_0(uint ga_0, uint gb_0, KernelContext_0 thread* kernelContext_2)
{
    bool _S3;
    uint _S4 = min(ga_0, gb_0);
    uint _S5 = max(ga_0, gb_0);
    uint _S6 = kernelContext_2->params_0->rows_0;
    uint lo_0 = 0U;
    uint hi_0 = _S6;
    for(;;)
    {
        if(lo_0 < hi_0)
        {
        }
        else
        {
            break;
        }
        uint mid_0 = (lo_0 + hi_0) / 2U;
        uint4 _S7 = uint4(*(kernelContext_2->slook_0+mid_0)) ;
        uint _S8 = _S7.x;
        if(_S8 < _S4)
        {
            _S3 = true;
        }
        else
        {
            if(_S8 == _S4)
            {
                _S3 = (_S7.y) < _S5;
            }
            else
            {
                _S3 = false;
            }
        }
        if(_S3)
        {
            lo_0 = mid_0 + 1U;
        }
        else
        {
            hi_0 = mid_0;
        }
    }
    if(lo_0 < (kernelContext_2->params_0->rows_0))
    {
        _S3 = ((uint4(*(kernelContext_2->slook_0+lo_0)) ).x) == _S4;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        _S3 = ((uint4(*(kernelContext_2->slook_0+lo_0)) ).y) == _S5;
    }
    else
    {
        _S3 = false;
    }
    if(_S3)
    {
        return (uint4(*(kernelContext_2->slook_0+lo_0)) ).z;
    }
    return 4294967295U;
}

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S9;
    if((st_0->damage_0) < 1.0f)
    {
        _S9 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S9 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S9 = false;
        }
    }
    return _S9;
}

[[kernel]] void feat_map(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_0 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_2 [[buffer(0)]], uint device* bmap_2 [[buffer(11)]], uint device* counters_2 [[buffer(28)]], Island_natural_0 device* islands_2 [[buffer(24)]], BondStatic_natural_0 device* bond_static_2 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(13)]], uint device* sinc_2 [[buffer(9)]], packed_uint4 device* slook_2 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(15)]], uint device* ctl_2 [[buffer(1)]], float device* work_2 [[buffer(18)]], packed_float4 device* nodes_2 [[buffer(3)]], packed_uint2 device* key_node_2 [[buffer(12)]], packed_float4 device* sc_2 [[buffer(7)]], packed_float4 device* bonds_2 [[buffer(4)]], packed_float4 device* sb_2 [[buffer(8)]], packed_uint4 device* bedge_2 [[buffer(22)]], packed_float4 device* calls_2 [[buffer(2)]], packed_float4 device* imps_2 [[buffer(16)]], packed_uint4 device* cisl_2 [[buffer(5)]], packed_float4 device* cislf_2 [[buffer(6)]], packed_float4 device* contacts_2 [[buffer(17)]], uint device* fctl_2 [[buffer(23)]], uint device* edges_2 [[buffer(20)]], uint device* cols_2 [[buffer(21)]], float device* feat_2 [[buffer(19)]], packed_float4 device* ctab_2 [[buffer(25)]], packed_float4 device* bodies_2 [[buffer(26)]], packed_float4 device* state_2 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_3;
    (&kernelContext_3)->params_0 = params_2;
    (&kernelContext_3)->bmap_0 = bmap_2;
    (&kernelContext_3)->counters_0 = counters_2;
    (&kernelContext_3)->islands_0 = islands_2;
    (&kernelContext_3)->bond_static_0 = bond_static_2;
    (&kernelContext_3)->chunks_0 = chunks_2;
    (&kernelContext_3)->sinc_0 = sinc_2;
    (&kernelContext_3)->slook_0 = slook_2;
    (&kernelContext_3)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_3)->ctl_0 = ctl_2;
    (&kernelContext_3)->work_0 = work_2;
    (&kernelContext_3)->nodes_0 = nodes_2;
    (&kernelContext_3)->key_node_0 = key_node_2;
    (&kernelContext_3)->sc_0 = sc_2;
    (&kernelContext_3)->bonds_0 = bonds_2;
    (&kernelContext_3)->sb_0 = sb_2;
    (&kernelContext_3)->bedge_0 = bedge_2;
    (&kernelContext_3)->calls_0 = calls_2;
    (&kernelContext_3)->imps_0 = imps_2;
    (&kernelContext_3)->cisl_0 = cisl_2;
    (&kernelContext_3)->cislf_0 = cislf_2;
    (&kernelContext_3)->contacts_0 = contacts_2;
    (&kernelContext_3)->fctl_0 = fctl_2;
    (&kernelContext_3)->edges_0 = edges_2;
    (&kernelContext_3)->cols_0 = cols_2;
    (&kernelContext_3)->feat_0 = feat_2;
    (&kernelContext_3)->ctab_0 = ctab_2;
    (&kernelContext_3)->bodies_0 = bodies_2;
    (&kernelContext_3)->state_0 = state_2;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_2;
    (&kernelContext_3)->g_red_0 = &g_red_2;
    threadgroup array<float, int(64)> g_mv_2;
    (&kernelContext_3)->g_mv_0 = &g_mv_2;
    threadgroup array<uint, int(64)> g_mi_2;
    (&kernelContext_3)->g_mi_0 = &g_mi_2;
    threadgroup array<uint, int(4096)> g_plug_2;
    (&kernelContext_3)->g_plug_0 = &g_plug_2;
    threadgroup uint g_flag_2;
    (&kernelContext_3)->g_flag_0 = &g_flag_2;
    threadgroup array<uint, int(64)> g_scan_2;
    (&kernelContext_3)->g_scan_0 = &g_scan_2;
    threadgroup float4 g_scalars_2;
    (&kernelContext_3)->g_scalars_0 = &g_scalars_2;
    uint i_0 = group_0.x;
    if(i_0 >= counters_2[int(0)])
    {
        return;
    }
    uint4 _S10 = uint4(((&kernelContext_3)->islands_0+i_0)->range_0) ;
    uint l_0 = _S10.z + thread_0.x;
    for(;;)
    {
        if(l_0 < (_S10.w))
        {
        }
        else
        {
            break;
        }
        JointBond_natural_0 law_1 = ((&kernelContext_3)->bond_static_0+l_0)->law_0;
        uint4 _S11 = uint4(((&kernelContext_3)->bond_static_0+l_0)->law_0.ids_0) ;
        uint _S12 = chunk_row_of_0(_S11.y, &kernelContext_3);
        uint _S13 = chunk_row_of_0(_S11.z, &kernelContext_3);
        uint _S14 = bond_row_lookup_0(_S12, _S13, &kernelContext_3);
        *((&kernelContext_3)->bmap_0+l_0) = _S14;
        if(_S14 == 4294967295U)
        {
            l_0 = l_0 + 64U;
            continue;
        }
        *((&kernelContext_3)->bmap_0+((&kernelContext_3)->params_0->layout_cap_0 + _S14)) = l_0;
        bool _S15 = ((float4(law_1.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S16 = ((&kernelContext_3)->bond_dyn_0+l_0)->js_0;
        bool _S17 = connected_0(&_S16, _S15);
        uint at_0 = (&kernelContext_3)->params_0->layout_cap_0 + (&kernelContext_3)->params_0->rows_0 + _S14;
        bool _S18;
        if(!_S17)
        {
            _S18 = (*((&kernelContext_3)->bmap_0+at_0)) == 4294967295U;
        }
        else
        {
            _S18 = false;
        }
        if(_S18)
        {
            *((&kernelContext_3)->bmap_0+at_0) = (&kernelContext_3)->params_0->frame_0;
        }
        l_0 = l_0 + 64U;
    }
    return;
}

float4 node_0(uint n_0, uint k_0, KernelContext_0 thread* kernelContext_4)
{
    return float4(*(kernelContext_4->nodes_0+(18U * n_0 + k_0))) ;
}

[[kernel]] void feat_keys(uint3 id_1 [[thread_position_in_grid]], FeatParams_0 constant* params_3 [[buffer(0)]], uint device* bmap_3 [[buffer(11)]], uint device* counters_3 [[buffer(28)]], Island_natural_0 device* islands_3 [[buffer(24)]], BondStatic_natural_0 device* bond_static_3 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(13)]], uint device* sinc_3 [[buffer(9)]], packed_uint4 device* slook_3 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(15)]], uint device* ctl_3 [[buffer(1)]], float device* work_3 [[buffer(18)]], packed_float4 device* nodes_3 [[buffer(3)]], packed_uint2 device* key_node_3 [[buffer(12)]], packed_float4 device* sc_3 [[buffer(7)]], packed_float4 device* bonds_3 [[buffer(4)]], packed_float4 device* sb_3 [[buffer(8)]], packed_uint4 device* bedge_3 [[buffer(22)]], packed_float4 device* calls_3 [[buffer(2)]], packed_float4 device* imps_3 [[buffer(16)]], packed_uint4 device* cisl_3 [[buffer(5)]], packed_float4 device* cislf_3 [[buffer(6)]], packed_float4 device* contacts_3 [[buffer(17)]], uint device* fctl_3 [[buffer(23)]], uint device* edges_3 [[buffer(20)]], uint device* cols_3 [[buffer(21)]], float device* feat_3 [[buffer(19)]], packed_float4 device* ctab_3 [[buffer(25)]], packed_float4 device* bodies_3 [[buffer(26)]], packed_float4 device* state_3 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_5;
    (&kernelContext_5)->params_0 = params_3;
    (&kernelContext_5)->bmap_0 = bmap_3;
    (&kernelContext_5)->counters_0 = counters_3;
    (&kernelContext_5)->islands_0 = islands_3;
    (&kernelContext_5)->bond_static_0 = bond_static_3;
    (&kernelContext_5)->chunks_0 = chunks_3;
    (&kernelContext_5)->sinc_0 = sinc_3;
    (&kernelContext_5)->slook_0 = slook_3;
    (&kernelContext_5)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_5)->ctl_0 = ctl_3;
    (&kernelContext_5)->work_0 = work_3;
    (&kernelContext_5)->nodes_0 = nodes_3;
    (&kernelContext_5)->key_node_0 = key_node_3;
    (&kernelContext_5)->sc_0 = sc_3;
    (&kernelContext_5)->bonds_0 = bonds_3;
    (&kernelContext_5)->sb_0 = sb_3;
    (&kernelContext_5)->bedge_0 = bedge_3;
    (&kernelContext_5)->calls_0 = calls_3;
    (&kernelContext_5)->imps_0 = imps_3;
    (&kernelContext_5)->cisl_0 = cisl_3;
    (&kernelContext_5)->cislf_0 = cislf_3;
    (&kernelContext_5)->contacts_0 = contacts_3;
    (&kernelContext_5)->fctl_0 = fctl_3;
    (&kernelContext_5)->edges_0 = edges_3;
    (&kernelContext_5)->cols_0 = cols_3;
    (&kernelContext_5)->feat_0 = feat_3;
    (&kernelContext_5)->ctab_0 = ctab_3;
    (&kernelContext_5)->bodies_0 = bodies_3;
    (&kernelContext_5)->state_0 = state_3;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_3;
    (&kernelContext_5)->g_red_0 = &g_red_3;
    threadgroup array<float, int(64)> g_mv_3;
    (&kernelContext_5)->g_mv_0 = &g_mv_3;
    threadgroup array<uint, int(64)> g_mi_3;
    (&kernelContext_5)->g_mi_0 = &g_mi_3;
    threadgroup array<uint, int(4096)> g_plug_3;
    (&kernelContext_5)->g_plug_0 = &g_plug_3;
    threadgroup uint g_flag_3;
    (&kernelContext_5)->g_flag_0 = &g_flag_3;
    threadgroup array<uint, int(64)> g_scan_3;
    (&kernelContext_5)->g_scan_0 = &g_scan_3;
    threadgroup float4 g_scalars_3;
    (&kernelContext_5)->g_scalars_0 = &g_scalars_3;
    uint n_1 = id_1.x;
    if(n_1 >= ctl_3[int(2)])
    {
        return;
    }
    uint k_1 = 0U;
    for(;;)
    {
        if(k_1 < 24U)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_5)->work_0+(24U * n_1 + k_1)) = 0.0f;
        k_1 = k_1 + 1U;
    }
    float4 _S19 = node_0(n_1, 4U, &kernelContext_5);
    if((as_type<uint>((_S19.w))) != 0U)
    {
        return;
    }
    float4 _S20 = node_0(n_1, 12U, &kernelContext_5);
    uint row_0 = (&kernelContext_5)->sinc_0[2U + (as_type<uint>((_S20.z)))] + (as_type<uint>((_S20.w)));
    *((&kernelContext_5)->key_node_0+row_0) = packed_uint2(uint2(n_1 + 1U, (&kernelContext_5)->params_0->frame_0)) ;
    uint _S21 = 6U * row_0;
    float4 _S22 = float4(*((&kernelContext_5)->sc_0+(_S21 + 1U))) ;
    float4 _S23 = float4(*((&kernelContext_5)->sc_0+(_S21 + 2U))) ;
    float4 _S24 = float4(*((&kernelContext_5)->sc_0+(_S21 + 3U))) ;
    uint _S25 = 24U * n_1;
    uint _S26 = _S25 + 9U;
    *((&kernelContext_5)->work_0+_S26) = _S23.x;
    *((&kernelContext_5)->work_0+(_S26 + 1U)) = _S23.y;
    *((&kernelContext_5)->work_0+(_S26 + 2U)) = _S23.z;
    *((&kernelContext_5)->work_0+(_S25 + 12U)) = _S23.w;
    *((&kernelContext_5)->work_0+(_S25 + 13U)) = _S22.y;
    float device* _S27 = (&kernelContext_5)->work_0+(_S25 + 14U);
    float _S28;
    if((_S22.z) != 0.0f)
    {
        _S28 = 1.0f;
    }
    else
    {
        if((_S22.w) != 0.0f)
        {
            _S28 = 2.0f;
        }
        else
        {
            _S28 = 0.0f;
        }
    }
    *_S27 = _S28;
    float4 _S29 = float4(*((&kernelContext_5)->sc_0+(_S21 + 4U))) ;
    float4 _S30 = float4(*((&kernelContext_5)->sc_0+(_S21 + 5U))) ;
    *((&kernelContext_5)->work_0+(_S25 + 15U)) = (float4(*((&kernelContext_5)->sc_0+_S21)) ).y;
    *((&kernelContext_5)->work_0+(_S25 + 16U)) = (as_type<float>((row_0)));
    k_1 = 0U;
    for(;;)
    {
        if(k_1 < 4U)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_5)->work_0+(_S25 + 17U + k_1)) = _S30[k_1];
        k_1 = k_1 + 1U;
    }
    k_1 = 0U;
    for(;;)
    {
        if(k_1 < 3U)
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_5)->work_0+(_S25 + 21U + k_1)) = _S29[k_1];
        k_1 = k_1 + 1U;
    }
    float4 _S31 = node_0(n_1, 13U, &kernelContext_5);
    uint _S32 = 18U * n_1;
    *((&kernelContext_5)->nodes_0+(_S32 + 13U)) = packed_float4(float4(_S31.xyz, _S24.w)) ;
    packed_float4 device* _S33 = (&kernelContext_5)->nodes_0+(_S32 + 2U);
    float3 _S34 = _S24.xyz;
    float4 _S35 = node_0(n_1, 2U, &kernelContext_5);
    *_S33 = packed_float4(float4(_S34, _S35.w)) ;
    return;
}

uint node_of_key_0(uint row_1, KernelContext_0 thread* kernelContext_6)
{
    uint2 _S36 = uint2(*(kernelContext_6->key_node_0+row_1)) ;
    uint _S37 = _S36.x;
    bool _S38;
    if(_S37 != 0U)
    {
        _S38 = (_S36.y) == (kernelContext_6->params_0->frame_0);
    }
    else
    {
        _S38 = false;
    }
    uint _S39;
    if(_S38)
    {
        _S39 = _S37 - 1U;
    }
    else
    {
        _S39 = 4294967295U;
    }
    return _S39;
}

[[kernel]] void feat_brow(uint3 id_2 [[thread_position_in_grid]], FeatParams_0 constant* params_4 [[buffer(0)]], uint device* bmap_4 [[buffer(11)]], uint device* counters_4 [[buffer(28)]], Island_natural_0 device* islands_4 [[buffer(24)]], BondStatic_natural_0 device* bond_static_4 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(13)]], uint device* sinc_4 [[buffer(9)]], packed_uint4 device* slook_4 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(15)]], uint device* ctl_4 [[buffer(1)]], float device* work_4 [[buffer(18)]], packed_float4 device* nodes_4 [[buffer(3)]], packed_uint2 device* key_node_4 [[buffer(12)]], packed_float4 device* sc_4 [[buffer(7)]], packed_float4 device* bonds_4 [[buffer(4)]], packed_float4 device* sb_4 [[buffer(8)]], packed_uint4 device* bedge_4 [[buffer(22)]], packed_float4 device* calls_4 [[buffer(2)]], packed_float4 device* imps_4 [[buffer(16)]], packed_uint4 device* cisl_4 [[buffer(5)]], packed_float4 device* cislf_4 [[buffer(6)]], packed_float4 device* contacts_4 [[buffer(17)]], uint device* fctl_4 [[buffer(23)]], uint device* edges_4 [[buffer(20)]], uint device* cols_4 [[buffer(21)]], float device* feat_4 [[buffer(19)]], packed_float4 device* ctab_4 [[buffer(25)]], packed_float4 device* bodies_4 [[buffer(26)]], packed_float4 device* state_4 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_7;
    (&kernelContext_7)->params_0 = params_4;
    (&kernelContext_7)->bmap_0 = bmap_4;
    (&kernelContext_7)->counters_0 = counters_4;
    (&kernelContext_7)->islands_0 = islands_4;
    (&kernelContext_7)->bond_static_0 = bond_static_4;
    (&kernelContext_7)->chunks_0 = chunks_4;
    (&kernelContext_7)->sinc_0 = sinc_4;
    (&kernelContext_7)->slook_0 = slook_4;
    (&kernelContext_7)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_7)->ctl_0 = ctl_4;
    (&kernelContext_7)->work_0 = work_4;
    (&kernelContext_7)->nodes_0 = nodes_4;
    (&kernelContext_7)->key_node_0 = key_node_4;
    (&kernelContext_7)->sc_0 = sc_4;
    (&kernelContext_7)->bonds_0 = bonds_4;
    (&kernelContext_7)->sb_0 = sb_4;
    (&kernelContext_7)->bedge_0 = bedge_4;
    (&kernelContext_7)->calls_0 = calls_4;
    (&kernelContext_7)->imps_0 = imps_4;
    (&kernelContext_7)->cisl_0 = cisl_4;
    (&kernelContext_7)->cislf_0 = cislf_4;
    (&kernelContext_7)->contacts_0 = contacts_4;
    (&kernelContext_7)->fctl_0 = fctl_4;
    (&kernelContext_7)->edges_0 = edges_4;
    (&kernelContext_7)->cols_0 = cols_4;
    (&kernelContext_7)->feat_0 = feat_4;
    (&kernelContext_7)->ctab_0 = ctab_4;
    (&kernelContext_7)->bodies_0 = bodies_4;
    (&kernelContext_7)->state_0 = state_4;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_4;
    (&kernelContext_7)->g_red_0 = &g_red_4;
    threadgroup array<float, int(64)> g_mv_4;
    (&kernelContext_7)->g_mv_0 = &g_mv_4;
    threadgroup array<uint, int(64)> g_mi_4;
    (&kernelContext_7)->g_mi_0 = &g_mi_4;
    threadgroup array<uint, int(4096)> g_plug_4;
    (&kernelContext_7)->g_plug_0 = &g_plug_4;
    threadgroup uint g_flag_4;
    (&kernelContext_7)->g_flag_0 = &g_flag_4;
    threadgroup array<uint, int(64)> g_scan_4;
    (&kernelContext_7)->g_scan_0 = &g_scan_4;
    threadgroup float4 g_scalars_4;
    (&kernelContext_7)->g_scalars_0 = &g_scalars_4;
    uint b_0 = id_2.x;
    if(b_0 >= ctl_4[int(3)])
    {
        return;
    }
    uint l_1 = (as_type<uint>(((float4(*((&kernelContext_7)->bonds_0+(2U * b_0 + 1U))) ).z)));
    uint device* _S40 = (&kernelContext_7)->bmap_0+l_1;
    uint4 ends_0 = (as_type<uint4>((float4(*((&kernelContext_7)->sb_0+(9U * *_S40 + 7U))) )));
    packed_uint4 device* _S41 = (&kernelContext_7)->bedge_0+b_0;
    uint _S42 = node_of_key_0(ends_0.x, &kernelContext_7);
    uint _S43 = node_of_key_0(ends_0.y, &kernelContext_7);
    *_S41 = packed_uint4(uint4(_S42, _S43, *_S40, l_1)) ;
    return;
}

uint group_uniform_0(uint value_0, uint thread_1)
{
    return value_0;
}

uint call_mask_0(uint g_0, KernelContext_0 thread* kernelContext_8)
{
    return (as_type<uint>(((float4(*(kernelContext_8->calls_0+(10U * g_0 + 5U))) ).w)));
}

uint call_e0_0(uint g_1, KernelContext_0 thread* kernelContext_9)
{
    return (as_type<uint>(((float4(*(kernelContext_9->calls_0+(10U * g_1 + 6U))) ).x)));
}

uint call_ne_0(uint g_2, KernelContext_0 thread* kernelContext_10)
{
    return (as_type<uint>(((float4(*(kernelContext_10->calls_0+(10U * g_2 + 6U))) ).y)));
}

uint call_first_0(uint g_3, KernelContext_0 thread* kernelContext_11)
{
    return (as_type<uint>(((float4(*(kernelContext_11->calls_0+(10U * g_3 + 2U))) ).z)));
}

uint call_chunks_0(uint g_4, KernelContext_0 thread* kernelContext_12)
{
    return (as_type<uint>(((float4(*(kernelContext_12->calls_0+(10U * g_4 + 3U))) ).w)));
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float4 group_min_0(uint t_0, float4 a_0, KernelContext_0 thread* kernelContext_13)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_13->g_red_0)[int(0)][t_0] = a_0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_0 = 32U;
    for(;;)
    {
        if(s_0 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_0 < s_0)
        {
            (*kernelContext_13->g_red_0)[int(0)][t_0] = min((*kernelContext_13->g_red_0)[int(0)][t_0], (*kernelContext_13->g_red_0)[int(0)][t_0 + s_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_13->g_red_0)[int(0)][int(0)];
}

float4 group_sum_0(uint t_1, float4 a_1, KernelContext_0 thread* kernelContext_14)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_14->g_red_0)[int(0)][t_1] = a_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_1 = 32U;
    for(;;)
    {
        if(s_1 > 0U)
        {
        }
        else
        {
            break;
        }
        if(t_1 < s_1)
        {
            (*kernelContext_14->g_red_0)[int(0)][t_1] = (*kernelContext_14->g_red_0)[int(0)][t_1] + (*kernelContext_14->g_red_0)[int(0)][t_1 + s_1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_1 = s_1 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_14->g_red_0)[int(0)][int(0)];
}

uint group_argmin_0(uint t_2, float v_0, uint i_1, KernelContext_0 thread* kernelContext_15)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_15->g_mv_0)[t_2] = v_0;
    (*kernelContext_15->g_mi_0)[t_2] = i_1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s_2 = 32U;
    for(;;)
    {
        if(s_2 > 0U)
        {
        }
        else
        {
            break;
        }
        bool _S44;
        if(t_2 < s_2)
        {
            uint _S45 = t_2 + s_2;
            if(((*kernelContext_15->g_mv_0)[_S45]) < (*kernelContext_15->g_mv_0)[t_2])
            {
                _S44 = true;
            }
            else
            {
                if(((*kernelContext_15->g_mv_0)[_S45]) == (*kernelContext_15->g_mv_0)[t_2])
                {
                    _S44 = ((*kernelContext_15->g_mi_0)[_S45]) < (*kernelContext_15->g_mi_0)[t_2];
                }
                else
                {
                    _S44 = false;
                }
            }
        }
        else
        {
            _S44 = false;
        }
        if(_S44)
        {
            uint _S46 = t_2 + s_2;
            (*kernelContext_15->g_mv_0)[t_2] = (*kernelContext_15->g_mv_0)[_S46];
            (*kernelContext_15->g_mi_0)[t_2] = (*kernelContext_15->g_mi_0)[_S46];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_2 = s_2 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_15->g_mi_0)[int(0)];
}

float hertz_duration_0(float m_0, float r_0, float e_0, float v_1)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_0 * e_0 * e_0 * max(abs(v_1), 0.00100000004749745f)), 0.20000000298023224f);
}

[[kernel]] void feat_contacts(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_2 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_5 [[buffer(0)]], uint device* bmap_5 [[buffer(11)]], uint device* counters_5 [[buffer(28)]], Island_natural_0 device* islands_5 [[buffer(24)]], BondStatic_natural_0 device* bond_static_5 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(13)]], uint device* sinc_5 [[buffer(9)]], packed_uint4 device* slook_5 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(15)]], uint device* ctl_5 [[buffer(1)]], float device* work_5 [[buffer(18)]], packed_float4 device* nodes_5 [[buffer(3)]], packed_uint2 device* key_node_5 [[buffer(12)]], packed_float4 device* sc_5 [[buffer(7)]], packed_float4 device* bonds_5 [[buffer(4)]], packed_float4 device* sb_5 [[buffer(8)]], packed_uint4 device* bedge_5 [[buffer(22)]], packed_float4 device* calls_5 [[buffer(2)]], packed_float4 device* imps_5 [[buffer(16)]], packed_uint4 device* cisl_5 [[buffer(5)]], packed_float4 device* cislf_5 [[buffer(6)]], packed_float4 device* contacts_5 [[buffer(17)]], uint device* fctl_5 [[buffer(23)]], uint device* edges_5 [[buffer(20)]], uint device* cols_5 [[buffer(21)]], float device* feat_5 [[buffer(19)]], packed_float4 device* ctab_5 [[buffer(25)]], packed_float4 device* bodies_5 [[buffer(26)]], packed_float4 device* state_5 [[buffer(27)]])
{
    float t0_0;
    bool _S47;
    uint iter_0;
    uint j_0;
    uint bj_0;
    bool marked_0;
    bool _S48;
    bool _S49;
    thread KernelContext_0 kernelContext_16;
    (&kernelContext_16)->params_0 = params_5;
    (&kernelContext_16)->bmap_0 = bmap_5;
    (&kernelContext_16)->counters_0 = counters_5;
    (&kernelContext_16)->islands_0 = islands_5;
    (&kernelContext_16)->bond_static_0 = bond_static_5;
    (&kernelContext_16)->chunks_0 = chunks_5;
    (&kernelContext_16)->sinc_0 = sinc_5;
    (&kernelContext_16)->slook_0 = slook_5;
    (&kernelContext_16)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_16)->ctl_0 = ctl_5;
    (&kernelContext_16)->work_0 = work_5;
    (&kernelContext_16)->nodes_0 = nodes_5;
    (&kernelContext_16)->key_node_0 = key_node_5;
    (&kernelContext_16)->sc_0 = sc_5;
    (&kernelContext_16)->bonds_0 = bonds_5;
    (&kernelContext_16)->sb_0 = sb_5;
    (&kernelContext_16)->bedge_0 = bedge_5;
    (&kernelContext_16)->calls_0 = calls_5;
    (&kernelContext_16)->imps_0 = imps_5;
    (&kernelContext_16)->cisl_0 = cisl_5;
    (&kernelContext_16)->cislf_0 = cislf_5;
    (&kernelContext_16)->contacts_0 = contacts_5;
    (&kernelContext_16)->fctl_0 = fctl_5;
    (&kernelContext_16)->edges_0 = edges_5;
    (&kernelContext_16)->cols_0 = cols_5;
    (&kernelContext_16)->feat_0 = feat_5;
    (&kernelContext_16)->ctab_0 = ctab_5;
    (&kernelContext_16)->bodies_0 = bodies_5;
    (&kernelContext_16)->state_0 = state_5;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_5;
    (&kernelContext_16)->g_red_0 = &g_red_5;
    threadgroup array<float, int(64)> g_mv_5;
    (&kernelContext_16)->g_mv_0 = &g_mv_5;
    threadgroup array<uint, int(64)> g_mi_5;
    (&kernelContext_16)->g_mi_0 = &g_mi_5;
    threadgroup array<uint, int(4096)> g_plug_5;
    (&kernelContext_16)->g_plug_0 = &g_plug_5;
    threadgroup uint g_flag_5;
    (&kernelContext_16)->g_flag_0 = &g_flag_5;
    threadgroup array<uint, int(64)> g_scan_5;
    (&kernelContext_16)->g_scan_0 = &g_scan_5;
    threadgroup float4 g_scalars_5;
    (&kernelContext_16)->g_scalars_0 = &g_scalars_5;
    uint g_5 = group_1.x;
    uint t_3 = thread_2.x;
    uint k_2;
    if(g_5 >= ctl_5[int(1)])
    {
        k_2 = 1U;
    }
    else
    {
        k_2 = 0U;
    }
    if((group_uniform_0(k_2, t_3)) != 0U)
    {
        return;
    }
    uint _S50 = call_mask_0(g_5, &kernelContext_16);
    uint _S51 = group_uniform_0(_S50, t_3);
    uint _S52 = call_e0_0(g_5, &kernelContext_16);
    uint _S53 = group_uniform_0(_S52, t_3);
    uint _S54 = call_ne_0(g_5, &kernelContext_16);
    uint _S55 = group_uniform_0(_S54, t_3);
    uint _S56 = call_first_0(g_5, &kernelContext_16);
    uint _S57 = call_chunks_0(g_5, &kernelContext_16);
    uint _S58 = 10U * g_5 + 8U;
    uint _S59 = (as_type<uint>(((float4(*((&kernelContext_16)->calls_0+_S58)) ).x)));
    float vmax_0 = 0.0f;
    k_2 = 0U;
    uint rows_1 = 0U;
    for(;;)
    {
        if(k_2 < 32U)
        {
        }
        else
        {
            break;
        }
        uint _S60 = 1U << k_2;
        float vmax_1;
        if((_S51 & _S60) == 0U)
        {
            vmax_1 = vmax_0;
            uint _S61 = k_2 + 1U;
            vmax_0 = vmax_1;
            k_2 = _S61;
            continue;
        }
        uint inode_0 = _S56 + _S57 + ((uint)popcount((_S51 & (_S60 - 1U))));
        float4 _S62 = node_0(inode_0, 0U, &kernelContext_16);
        float3 _S63 = _S62.xyz;
        float4 _S64 = node_0(inode_0, 3U, &kernelContext_16);
        float3 _S65 = _S64.xyz;
        uint _S66 = 4U * k_2;
        float _S67 = (float4(*((&kernelContext_16)->imps_0+(_S66 + 2U))) ).x;
        float _S68 = (float4(*((&kernelContext_16)->imps_0+(_S66 + 3U))) ).z;
        float _S69 = (float4(*((&kernelContext_16)->imps_0+_S66)) ).w;
        float _S70 = (float4(*((&kernelContext_16)->imps_0+(_S66 + 1U))) ).w;
        vmax_1 = vmax_0;
        uint e_1 = _S53;
        uint rows_2 = rows_1;
        for(;;)
        {
            if(e_1 < (_S53 + _S55))
            {
            }
            else
            {
                break;
            }
            uint _S71 = 2U * e_1;
            uint4 _S72 = uint4(*((&kernelContext_16)->cisl_0+_S71)) ;
            uint _S73 = _S71 + 1U;
            uint4 _S74 = uint4(*((&kernelContext_16)->cisl_0+_S73)) ;
            float4 _S75 = float4(*((&kernelContext_16)->cislf_0+_S73)) ;
            uint at_1 = _S59 + rows_2;
            uint _S76 = rows_2 + 1U;
            bool _S77 = t_3 == 0U;
            if(_S77)
            {
                *((&kernelContext_16)->contacts_0+(6U * at_1 + 5U)) = packed_float4(float4(0.0f) ) ;
            }
            uint n_isl_0 = group_uniform_0(_S72.z, t_3);
            bool _S78;
            if(n_isl_0 == 0U)
            {
                _S78 = true;
            }
            else
            {
                _S78 = (_S74.z) == 0U;
            }
            if(_S78)
            {
                e_1 = e_1 + 1U;
                rows_2 = _S76;
                continue;
            }
            float3 v_2 = _S65 - _S75.xyz;
            float speed_1 = length(v_2);
            uint _S79;
            if(speed_1 > 0.0f)
            {
                _S79 = 1U;
            }
            else
            {
                _S79 = 0U;
            }
            if((group_uniform_0(_S79, t_3)) == 0U)
            {
                e_1 = e_1 + 1U;
                rows_2 = _S76;
                continue;
            }
            float inf_0 = float_bits_0(2139095040U);
            float4 _S80 = float4(- inf_0) ;
            float4 lo_1 = float4(inf_0) ;
            float4 hi_1 = _S80;
            uint j_1 = t_3;
            float msum_0 = 0.0f;
            for(;;)
            {
                if(j_1 < n_isl_0)
                {
                }
                else
                {
                    break;
                }
                uint n_2 = _S72.y + j_1;
                float4 _S81 = node_0(n_2, 0U, &kernelContext_16);
                float3 p_0 = _S81.xyz;
                float3 _S82 = float3((*((&kernelContext_16)->work_0+(24U * n_2 + 12U)) + _S67)) ;
                float4 _S83 = min(lo_1, float4(p_0 - _S82, 0.0f));
                float4 _S84 = max(hi_1, float4(p_0 + _S82, 0.0f));
                float4 _S85 = node_0(n_2, 0U, &kernelContext_16);
                float msum_1 = msum_0 + _S85.w;
                uint j_2 = j_1 + 64U;
                lo_1 = _S83;
                hi_1 = _S84;
                j_1 = j_2;
                msum_0 = msum_1;
            }
            float4 _S86 = group_min_0(t_3, lo_1, &kernelContext_16);
            float4 _S87 = group_min_0(t_3, - hi_1, &kernelContext_16);
            float4 _S88 = - _S87;
            float4 _S89 = group_sum_0(t_3, float4(msum_0, 0.0f, 0.0f, 0.0f), &kernelContext_16);
            float cl_mass_0 = _S89.x;
            float _S90 = (&kernelContext_16)->params_0->dt_0;
            float t0_1 = 0.0f;
            float t1_1 = _S90;
            uint a_2 = 0U;
            for(;;)
            {
                if(a_2 < 3U)
                {
                }
                else
                {
                    break;
                }
                uint _S91 = a_2;
                if((abs(v_2[a_2])) < 9.999999960041972e-13f)
                {
                    uint _S92 = a_2;
                    if((_S63[a_2]) < (_S86[a_2]))
                    {
                        _S47 = true;
                    }
                    else
                    {
                        _S47 = (_S63[_S92]) > (_S88[a_2]);
                    }
                    if(_S47)
                    {
                        t0_0 = inf_0;
                    }
                    else
                    {
                        t0_0 = t0_1;
                    }
                    t0_1 = t0_0;
                    a_2 = a_2 + 1U;
                    continue;
                }
                float ta_0 = (_S86[a_2] - _S63[a_2]) / v_2[_S91];
                float tb_0 = (_S88[a_2] - _S63[a_2]) / v_2[_S91];
                float _S93 = min(t1_1, max(ta_0, tb_0));
                t0_1 = max(t0_1, min(ta_0, tb_0));
                t1_1 = _S93;
                a_2 = a_2 + 1U;
            }
            uint _S94;
            if(t0_1 > t1_1)
            {
                _S94 = 1U;
            }
            else
            {
                _S94 = 0U;
            }
            if((group_uniform_0(_S94, t_3)) != 0U)
            {
                e_1 = e_1 + 1U;
                rows_2 = _S76;
                continue;
            }
            float3 d_0 = v_2 / float3(speed_1) ;
            float3 hit_0 = _S63 + v_2 * float3(t0_1) ;
            t0_0 = inf_0;
            uint best_j_0 = 4294967295U;
            uint j_3 = t_3;
            for(;;)
            {
                if(j_3 < n_isl_0)
                {
                }
                else
                {
                    break;
                }
                float4 _S95 = node_0(_S72.y + j_3, 0U, &kernelContext_16);
                float3 q_0 = _S95.xyz - hit_0;
                float along_0 = dot(q_0, d_0);
                float dist_0 = length(q_0 - d_0 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
                if(dist_0 < t0_0)
                {
                    t0_0 = dist_0;
                    best_j_0 = j_3;
                }
                j_3 = j_3 + 64U;
            }
            uint _S96 = group_argmin_0(t_3, t0_0, best_j_0, &kernelContext_16);
            uint _S97 = _S72.y;
            uint entry_n_0 = _S97 + _S96;
            float4 _S98 = node_0(entry_n_0, 0U, &kernelContext_16);
            float _S99 = dot(_S98.xyz - hit_0, d_0);
            bool _S100 = n_isl_0 <= 4096U;
            uint j_4 = t_3;
            for(;;)
            {
                if(j_4 < (min(n_isl_0, 4096U)))
                {
                }
                else
                {
                    break;
                }
                if(j_4 == _S96)
                {
                    iter_0 = 1U;
                }
                else
                {
                    iter_0 = 0U;
                }
                (*(&kernelContext_16)->g_plug_0)[j_4] = iter_0;
                j_4 = j_4 + 64U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            iter_0 = 0U;
            for(;;)
            {
                if(iter_0 < n_isl_0)
                {
                    _S47 = _S100;
                }
                else
                {
                    _S47 = false;
                }
                if(_S47)
                {
                }
                else
                {
                    break;
                }
                if(_S77)
                {
                    *(&kernelContext_16)->g_flag_0 = 0U;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                j_0 = t_3;
                for(;;)
                {
                    if(j_0 < (_S74.z))
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint4 _S101 = uint4(*((&kernelContext_16)->bedge_0+(_S74.y + j_0))) ;
                    uint _S102 = _S101.x;
                    if(_S102 == 4294967295U)
                    {
                        marked_0 = true;
                    }
                    else
                    {
                        marked_0 = (_S101.y) == 4294967295U;
                    }
                    if(marked_0)
                    {
                        j_0 = j_0 + 64U;
                        continue;
                    }
                    uint la_0 = _S102 - _S97;
                    uint lb_0 = _S101.y - _S97;
                    if(la_0 >= n_isl_0)
                    {
                        _S48 = true;
                    }
                    else
                    {
                        _S48 = lb_0 >= n_isl_0;
                    }
                    if(_S48)
                    {
                        _S49 = true;
                    }
                    else
                    {
                        _S49 = ((*(&kernelContext_16)->g_plug_0)[la_0]) == (*(&kernelContext_16)->g_plug_0)[lb_0];
                    }
                    if(_S49)
                    {
                        j_0 = j_0 + 64U;
                        continue;
                    }
                    if(((*(&kernelContext_16)->g_plug_0)[la_0]) != 0U)
                    {
                        bj_0 = lb_0;
                    }
                    else
                    {
                        bj_0 = la_0;
                    }
                    uint _S103 = _S97 + bj_0;
                    float4 _S104 = node_0(_S103, 0U, &kernelContext_16);
                    float3 q_1 = _S104.xyz - hit_0;
                    float along_1 = dot(q_1, d_0);
                    float lateral_0 = length(q_1 - d_0 * float3(along_1) );
                    bool _S105;
                    if(along_1 >= (_S99 - *((&kernelContext_16)->work_0+(24U * _S103 + 12U))))
                    {
                        _S105 = lateral_0 <= (_S68 + max(along_1 - _S99, 0.0f));
                    }
                    else
                    {
                        _S105 = false;
                    }
                    if(_S105)
                    {
                        (*(&kernelContext_16)->g_plug_0)[bj_0] = 1U;
                        *(&kernelContext_16)->g_flag_0 = 1U;
                    }
                    j_0 = j_0 + 64U;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if((group_uniform_0(*(&kernelContext_16)->g_flag_0, t_3)) == 0U)
                {
                    break;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                iter_0 = iter_0 + 1U;
            }
            thread float4 acc_0 = float4(0.0f) ;
            j_0 = t_3;
            for(;;)
            {
                if(j_0 < n_isl_0)
                {
                }
                else
                {
                    break;
                }
                if(((*(&kernelContext_16)->g_plug_0)[j_0]) != 0U)
                {
                    float4 _S106 = node_0(_S97 + j_0, 0U, &kernelContext_16);
                    acc_0 = acc_0 + float4(_S106.w, 1.0f, 0.0f, 0.0f);
                }
                j_0 = j_0 + 64U;
            }
            bj_0 = t_3;
            for(;;)
            {
                if(bj_0 < (_S74.z))
                {
                }
                else
                {
                    break;
                }
                uint4 _S107 = uint4(*((&kernelContext_16)->bedge_0+(_S74.y + bj_0))) ;
                uint _S108 = _S107.x;
                if(_S108 == 4294967295U)
                {
                    marked_0 = true;
                }
                else
                {
                    marked_0 = (_S107.y) == 4294967295U;
                }
                if(marked_0)
                {
                    bj_0 = bj_0 + 64U;
                    continue;
                }
                uint la_1 = _S108 - _S97;
                uint lb_1 = _S107.y - _S97;
                if(la_1 < n_isl_0)
                {
                    _S48 = lb_1 < n_isl_0;
                }
                else
                {
                    _S48 = false;
                }
                if(_S48)
                {
                    _S49 = (((*(&kernelContext_16)->g_plug_0)[la_1]) != 0U) != (((*(&kernelContext_16)->g_plug_0)[lb_1]) != 0U);
                }
                else
                {
                    _S49 = false;
                }
                if(_S49)
                {
                    uint _S109 = 9U * _S107.z;
                    acc_0.z = acc_0.z + (float4(*((&kernelContext_16)->sb_0+(_S109 + 5U))) ).x * (float4(*((&kernelContext_16)->sb_0+_S109)) ).w * (float4(*((&kernelContext_16)->sb_0+(_S109 + 4U))) ).z;
                }
                bj_0 = bj_0 + 64U;
            }
            float4 _S110 = group_sum_0(t_3, acc_0, &kernelContext_16);
            acc_0 = _S110;
            float m_plug_0 = _S110.x;
            float _S111 = float(n_isl_0);
            bool has_plug_0 = (_S110.y) != _S111;
            float capacity_0 = _S110.z;
            float _S112 = 1.0f / _S70;
            float4 _S113 = node_0(entry_n_0, 13U, &kernelContext_16);
            float modulus_0 = 1.0f / (_S112 + 1.0f / _S113.w);
            float m_red_0;
            if((_S75.w) != 0.0f)
            {
                m_red_0 = _S69;
            }
            else
            {
                m_red_0 = cl_mass_0 * _S69 / (cl_mass_0 + _S69);
            }
            float hertz_0 = hertz_duration_0(m_red_0, _S68, modulus_0, speed_1);
            float peak_0 = 3.14159274101257324f * (2.0f * m_red_0 * speed_1) / (2.0f * hertz_0);
            float ratio_0;
            if(has_plug_0)
            {
                float m_eff_0 = m_plug_0 * _S69 / (m_plug_0 + _S69);
                ratio_0 = m_eff_0 * speed_1 / hertz_duration_0(m_eff_0, _S68, modulus_0, speed_1) / max(capacity_0, 1.00000000317107685e-30f);
            }
            else
            {
                ratio_0 = 0.0f;
            }
            if(has_plug_0)
            {
                marked_0 = speed_1 > ((&kernelContext_16)->params_0->speed_0);
            }
            else
            {
                marked_0 = false;
            }
            if(marked_0)
            {
                uint j_5 = t_3;
                for(;;)
                {
                    if(j_5 < n_isl_0)
                    {
                    }
                    else
                    {
                        break;
                    }
                    if(((*(&kernelContext_16)->g_plug_0)[j_5]) != 0U)
                    {
                        *((&kernelContext_16)->work_0+(24U * (_S97 + j_5) + 3U)) = float(at_1 - _S59 + 1U);
                    }
                    j_5 = j_5 + 64U;
                }
            }
            if(_S77)
            {
                uint _S114 = 6U * at_1;
                *((&kernelContext_16)->contacts_0+_S114) = packed_float4(float4((as_type<float>((k_2))), (as_type<float>((entry_n_0))), (as_type<float>((e_1))), t0_1)) ;
                *((&kernelContext_16)->contacts_0+(_S114 + 1U)) = packed_float4(float4(hit_0, speed_1)) ;
                *((&kernelContext_16)->contacts_0+(_S114 + 2U)) = packed_float4(float4(d_0, m_red_0)) ;
                *((&kernelContext_16)->contacts_0+(_S114 + 3U)) = packed_float4(float4(modulus_0, _S68, hertz_0, peak_0)) ;
                packed_float4 device* _S115 = (&kernelContext_16)->contacts_0+(_S114 + 4U);
                float _S116;
                if(has_plug_0)
                {
                    _S116 = m_plug_0;
                }
                else
                {
                    _S116 = 0.0f;
                }
                float _S117;
                if(marked_0)
                {
                    _S117 = 1.0f;
                }
                else
                {
                    _S117 = 0.0f;
                }
                *_S115 = packed_float4(float4(ratio_0, _S116, _S117, _S111)) ;
                *((&kernelContext_16)->contacts_0+(_S114 + 5U)) = packed_float4(float4(1.0f, 0.0f, 0.0f, 0.0f)) ;
            }
            float _S118 = max(vmax_1, speed_1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            vmax_1 = _S118;
            e_1 = e_1 + 1U;
            rows_2 = _S76;
        }
        rows_1 = rows_2;
        uint _S61 = k_2 + 1U;
        vmax_0 = vmax_1;
        k_2 = _S61;
    }
    if(t_3 == 0U)
    {
        *((&kernelContext_16)->calls_0+_S58) = packed_float4(float4((float4(*((&kernelContext_16)->calls_0+_S58)) ).x, (as_type<float>((rows_1))), 0.0f, vmax_0)) ;
    }
    return;
}

uint call_bfirst_0(uint g_6, KernelContext_0 thread* kernelContext_17)
{
    return (as_type<uint>(((float4(*(kernelContext_17->calls_0+(10U * g_6 + 4U))) ).x)));
}

uint call_nbonds_0(uint g_7, KernelContext_0 thread* kernelContext_18)
{
    return (as_type<uint>(((float4(*(kernelContext_18->calls_0+(10U * g_7 + 4U))) ).y)));
}

uint group_scan_0(uint t_4, uint v_3, KernelContext_0 thread* kernelContext_19)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_19->g_scan_0)[t_4] = v_3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint step_0 = 1U;
    for(;;)
    {
        if(step_0 < 64U)
        {
        }
        else
        {
            break;
        }
        uint add_0;
        if(t_4 >= step_0)
        {
            add_0 = (*kernelContext_19->g_scan_0)[t_4 - step_0];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*kernelContext_19->g_scan_0)[t_4] = (*kernelContext_19->g_scan_0)[t_4] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_0 = step_0 << 1U;
    }
    return (*kernelContext_19->g_scan_0)[t_4];
}

uint group_total_0(KernelContext_0 thread* kernelContext_20)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_20->g_scan_0)[63U];
}

void sort_work_0(uint t_5, uint base_0, uint n_3, KernelContext_0 thread* kernelContext_21)
{
    uint m_1 = 1U;
    for(;;)
    {
        if(m_1 < n_3)
        {
        }
        else
        {
            break;
        }
        m_1 = m_1 << 1U;
    }
    float _S119 = float_bits_0(2139095040U);
    uint j_6 = n_3 + t_5;
    for(;;)
    {
        if(j_6 < m_1)
        {
        }
        else
        {
            break;
        }
        *(kernelContext_21->work_0+(base_0 + j_6)) = _S119;
        j_6 = j_6 + 64U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    uint size_0 = 2U;
    for(;;)
    {
        if(size_0 <= m_1)
        {
        }
        else
        {
            break;
        }
        uint stride_0 = size_0 >> 1U;
        for(;;)
        {
            if(stride_0 > 0U)
            {
            }
            else
            {
                break;
            }
            j_6 = t_5;
            for(;;)
            {
                if(j_6 < m_1)
                {
                }
                else
                {
                    break;
                }
                uint p_1 = j_6 ^ stride_0;
                if(p_1 > j_6)
                {
                    uint _S120 = base_0 + j_6;
                    float device* _S121 = kernelContext_21->work_0+_S120;
                    float a_3 = *_S121;
                    uint _S122 = base_0 + p_1;
                    float device* _S123 = kernelContext_21->work_0+_S122;
                    float b_1 = *_S123;
                    if(((*_S121) > (*_S123)) == ((j_6 & size_0) == 0U))
                    {
                        *(kernelContext_21->work_0+_S120) = b_1;
                        *(kernelContext_21->work_0+_S122) = a_3;
                    }
                }
                j_6 = j_6 + 64U;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            stride_0 = stride_0 >> 1U;
        }
        size_0 = size_0 << 1U;
    }
    return;
}

float median_work_0(uint base_1, uint n_4, KernelContext_0 thread* kernelContext_22)
{
    uint k_3 = n_4 / 2U;
    float _S124;
    if((n_4 % 2U) == 1U)
    {
        _S124 = *(kernelContext_22->work_0+(base_1 + k_3));
    }
    else
    {
        uint _S125 = base_1 + k_3;
        _S124 = 0.5f * (*(kernelContext_22->work_0+(_S125 - 1U)) + *(kernelContext_22->work_0+_S125));
    }
    return _S124;
}

[[kernel]] void feat_scales(uint3 thread_3 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_6 [[buffer(0)]], uint device* bmap_6 [[buffer(11)]], uint device* counters_6 [[buffer(28)]], Island_natural_0 device* islands_6 [[buffer(24)]], BondStatic_natural_0 device* bond_static_6 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(13)]], uint device* sinc_6 [[buffer(9)]], packed_uint4 device* slook_6 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(15)]], uint device* ctl_6 [[buffer(1)]], float device* work_6 [[buffer(18)]], packed_float4 device* nodes_6 [[buffer(3)]], packed_uint2 device* key_node_6 [[buffer(12)]], packed_float4 device* sc_6 [[buffer(7)]], packed_float4 device* bonds_6 [[buffer(4)]], packed_float4 device* sb_6 [[buffer(8)]], packed_uint4 device* bedge_6 [[buffer(22)]], packed_float4 device* calls_6 [[buffer(2)]], packed_float4 device* imps_6 [[buffer(16)]], packed_uint4 device* cisl_6 [[buffer(5)]], packed_float4 device* cislf_6 [[buffer(6)]], packed_float4 device* contacts_6 [[buffer(17)]], uint device* fctl_6 [[buffer(23)]], uint device* edges_6 [[buffer(20)]], uint device* cols_6 [[buffer(21)]], float device* feat_6 [[buffer(19)]], packed_float4 device* ctab_6 [[buffer(25)]], packed_float4 device* bodies_6 [[buffer(26)]], packed_float4 device* state_6 [[buffer(27)]])
{
    bool _S126;
    uint ok_0;
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->params_0 = params_6;
    (&kernelContext_23)->bmap_0 = bmap_6;
    (&kernelContext_23)->counters_0 = counters_6;
    (&kernelContext_23)->islands_0 = islands_6;
    (&kernelContext_23)->bond_static_0 = bond_static_6;
    (&kernelContext_23)->chunks_0 = chunks_6;
    (&kernelContext_23)->sinc_0 = sinc_6;
    (&kernelContext_23)->slook_0 = slook_6;
    (&kernelContext_23)->bond_dyn_0 = bond_dyn_6;
    (&kernelContext_23)->ctl_0 = ctl_6;
    (&kernelContext_23)->work_0 = work_6;
    (&kernelContext_23)->nodes_0 = nodes_6;
    (&kernelContext_23)->key_node_0 = key_node_6;
    (&kernelContext_23)->sc_0 = sc_6;
    (&kernelContext_23)->bonds_0 = bonds_6;
    (&kernelContext_23)->sb_0 = sb_6;
    (&kernelContext_23)->bedge_0 = bedge_6;
    (&kernelContext_23)->calls_0 = calls_6;
    (&kernelContext_23)->imps_0 = imps_6;
    (&kernelContext_23)->cisl_0 = cisl_6;
    (&kernelContext_23)->cislf_0 = cislf_6;
    (&kernelContext_23)->contacts_0 = contacts_6;
    (&kernelContext_23)->fctl_0 = fctl_6;
    (&kernelContext_23)->edges_0 = edges_6;
    (&kernelContext_23)->cols_0 = cols_6;
    (&kernelContext_23)->feat_0 = feat_6;
    (&kernelContext_23)->ctab_0 = ctab_6;
    (&kernelContext_23)->bodies_0 = bodies_6;
    (&kernelContext_23)->state_0 = state_6;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_6;
    (&kernelContext_23)->g_red_0 = &g_red_6;
    threadgroup array<float, int(64)> g_mv_6;
    (&kernelContext_23)->g_mv_0 = &g_mv_6;
    threadgroup array<uint, int(64)> g_mi_6;
    (&kernelContext_23)->g_mi_0 = &g_mi_6;
    threadgroup array<uint, int(4096)> g_plug_6;
    (&kernelContext_23)->g_plug_0 = &g_plug_6;
    threadgroup uint g_flag_6;
    (&kernelContext_23)->g_flag_0 = &g_flag_6;
    threadgroup array<uint, int(64)> g_scan_6;
    (&kernelContext_23)->g_scan_0 = &g_scan_6;
    threadgroup float4 g_scalars_6;
    (&kernelContext_23)->g_scalars_0 = &g_scalars_6;
    uint t_6 = thread_3.x;
    uint _S127 = group_uniform_0(ctl_6[int(1)], t_6);
    uint _S128 = 24U * params_6->max_nodes_0;
    uint g_8 = 0U;
    for(;;)
    {
        if(g_8 < _S127)
        {
        }
        else
        {
            break;
        }
        uint _S129 = call_first_0(g_8, &kernelContext_23);
        uint _S130 = call_chunks_0(g_8, &kernelContext_23);
        uint nc_0 = group_uniform_0(_S130, t_6);
        uint _S131 = call_bfirst_0(g_8, &kernelContext_23);
        uint _S132 = call_nbonds_0(g_8, &kernelContext_23);
        uint _S133 = group_uniform_0(_S132, t_6);
        uint _S134 = 10U * g_8;
        float L_0 = (float4(*((&kernelContext_23)->calls_0+_S134)) ).x;
        float best_0 = -1.0f;
        uint best_j_1 = 0U;
        uint j_7 = t_6;
        for(;;)
        {
            if(j_7 < nc_0)
            {
            }
            else
            {
                break;
            }
            float4 _S135 = node_0(_S129 + j_7, 0U, &kernelContext_23);
            float m_2 = _S135.w;
            if(m_2 >= best_0)
            {
                best_0 = m_2;
                best_j_1 = j_7;
            }
            j_7 = j_7 + 64U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_23)->g_mv_0)[t_6] = best_0;
        (*(&kernelContext_23)->g_mi_0)[t_6] = best_j_1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint s_3 = 32U;
        for(;;)
        {
            if(s_3 > 0U)
            {
            }
            else
            {
                break;
            }
            if(t_6 < s_3)
            {
                uint _S136 = t_6 + s_3;
                if(((*(&kernelContext_23)->g_mv_0)[_S136]) > (*(&kernelContext_23)->g_mv_0)[t_6])
                {
                    _S126 = true;
                }
                else
                {
                    if(((*(&kernelContext_23)->g_mv_0)[_S136]) == (*(&kernelContext_23)->g_mv_0)[t_6])
                    {
                        _S126 = ((*(&kernelContext_23)->g_mi_0)[_S136]) > (*(&kernelContext_23)->g_mi_0)[t_6];
                    }
                    else
                    {
                        _S126 = false;
                    }
                }
            }
            else
            {
                _S126 = false;
            }
            if(_S126)
            {
                uint _S137 = t_6 + s_3;
                (*(&kernelContext_23)->g_mv_0)[t_6] = (*(&kernelContext_23)->g_mv_0)[_S137];
                (*(&kernelContext_23)->g_mi_0)[t_6] = (*(&kernelContext_23)->g_mi_0)[_S137];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            s_3 = s_3 >> 1U;
        }
        uint dom_0 = _S129 + (*(&kernelContext_23)->g_mi_0)[int(0)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 _S138 = node_0(dom_0, 12U, &kernelContext_23);
        uint _S139 = 6U * ((&kernelContext_23)->sinc_0[2U + (as_type<uint>((_S138.z)))] + (as_type<uint>((_S138.w))));
        float4 _S140 = float4(*((&kernelContext_23)->sc_0+_S139)) ;
        float4 _S141 = float4(*((&kernelContext_23)->sc_0+(_S139 + 1U))) ;
        float c_wave_0 = sqrt(_S140.y / _S140.z);
        thread float4 ga_1 = float4(0.0f) ;
        uint b0_0 = 0U;
        uint carry_0 = 0U;
        for(;;)
        {
            if(b0_0 < _S133)
            {
            }
            else
            {
                break;
            }
            uint j_8 = b0_0 + t_6;
            bool _S142 = j_8 < _S133;
            uint4 be_0;
            if(_S142)
            {
                be_0 = uint4(*((&kernelContext_23)->bedge_0+(_S131 + j_8))) ;
            }
            else
            {
                be_0 = uint4(4294967295U) ;
            }
            if(_S142)
            {
                _S126 = (be_0.x) != 4294967295U;
            }
            else
            {
                _S126 = false;
            }
            bool _S143;
            if(_S126)
            {
                _S143 = (be_0.y) != 4294967295U;
            }
            else
            {
                _S143 = false;
            }
            if(_S143)
            {
                ok_0 = 1U;
            }
            else
            {
                ok_0 = 0U;
            }
            uint _S144 = group_scan_0(t_6, ok_0, &kernelContext_23);
            uint at_2 = carry_0 + _S144 - ok_0;
            if(ok_0 != 0U)
            {
                uint _S145 = 9U * be_0.z;
                float area_0 = (float4(*((&kernelContext_23)->sb_0+_S145)) ).w;
                float gab_0 = (float4(*((&kernelContext_23)->sb_0+(_S145 + 6U))) ).y * area_0;
                *((&kernelContext_23)->work_0+(_S128 + at_2)) = (float4(*((&kernelContext_23)->sb_0+(_S145 + 5U))) ).x;
                *((&kernelContext_23)->work_0+(_S128 + 131072U + at_2)) = gab_0;
                *((&kernelContext_23)->work_0+(_S128 + 262144U + at_2)) = area_0;
                ga_1.x = ga_1.x + gab_0;
                if(((*((&kernelContext_23)->work_0+(24U * be_0.x + 3U))) != 0.0f) != ((*((&kernelContext_23)->work_0+(24U * be_0.y + 3U))) != 0.0f))
                {
                    ga_1.y = ga_1.y + gab_0;
                }
            }
            uint _S146 = group_total_0(&kernelContext_23);
            uint carry_1 = carry_0 + _S146;
            b0_0 = b0_0 + 64U;
            carry_0 = carry_1;
        }
        uint nlive_0 = group_uniform_0(carry_0, t_6);
        float4 _S147 = group_sum_0(t_6, ga_1, &kernelContext_23);
        ga_1 = _S147;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float _S148 = _S140.w;
        float _S149 = _S141.x * L_0 * L_0;
        float _S150 = L_0 * L_0;
        float f_t_0;
        float GA_0;
        float med_area_0;
        if(nlive_0 > 0U)
        {
            sort_work_0(t_6, _S128, nlive_0, &kernelContext_23);
            uint _S151 = _S128 + 131072U;
            sort_work_0(t_6, _S151, nlive_0, &kernelContext_23);
            uint _S152 = _S128 + 262144U;
            sort_work_0(t_6, _S152, nlive_0, &kernelContext_23);
            if(t_6 == 0U)
            {
                float _S153 = median_work_0(_S128, nlive_0, &kernelContext_23);
                float _S154 = median_work_0(_S151, nlive_0, &kernelContext_23);
                float _S155 = median_work_0(_S152, nlive_0, &kernelContext_23);
                *(&kernelContext_23)->g_scalars_0 = float4(_S153, _S154, _S155, 0.0f);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            float _S156 = (*(&kernelContext_23)->g_scalars_0).x;
            float _S157 = (*(&kernelContext_23)->g_scalars_0).y;
            float _S158 = (*(&kernelContext_23)->g_scalars_0).z;
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            f_t_0 = _S156;
            GA_0 = _S157;
            med_area_0 = _S158;
        }
        else
        {
            f_t_0 = _S148;
            GA_0 = _S149;
            med_area_0 = _S150;
        }
        ok_0 = t_6;
        for(;;)
        {
            if(ok_0 < nc_0)
            {
            }
            else
            {
                break;
            }
            float4 _S159 = node_0(_S129 + ok_0, 12U, &kernelContext_23);
            *((&kernelContext_23)->work_0+(_S128 + ok_0)) = (float4(*((&kernelContext_23)->sc_0+6U * ((&kernelContext_23)->sinc_0[2U + (as_type<uint>((_S159.z)))] + (as_type<uint>((_S159.w)))))) ).y;
            ok_0 = ok_0 + 64U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        sort_work_0(t_6, _S128, nc_0, &kernelContext_23);
        bool _S160 = t_6 == 0U;
        if(_S160)
        {
            float _S161 = median_work_0(_S128, nc_0, &kernelContext_23);
            (*(&kernelContext_23)->g_scalars_0).w = _S161;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float e_med_0 = (*(&kernelContext_23)->g_scalars_0).w;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if(_S160)
        {
            float4 _S162 = float4(*((&kernelContext_23)->calls_0+_S134)) ;
            uint _S163 = _S134 + 1U;
            float4 _S164 = float4(*((&kernelContext_23)->calls_0+_S163)) ;
            *((&kernelContext_23)->calls_0+_S134) = packed_float4(float4(_S162.x, _S162.y, _S162.z, c_wave_0)) ;
            *((&kernelContext_23)->calls_0+_S163) = packed_float4(float4(f_t_0, e_med_0, GA_0, _S164.w)) ;
            *((&kernelContext_23)->calls_0+(_S134 + 9U)) = packed_float4(float4(med_area_0, ga_1.x, ga_1.y, 0.0f)) ;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        g_8 = g_8 + 1U;
    }
    return;
}

[[kernel]] void feat_summ(uint3 id_3 [[thread_position_in_grid]], FeatParams_0 constant* params_7 [[buffer(0)]], uint device* bmap_7 [[buffer(11)]], uint device* counters_7 [[buffer(28)]], Island_natural_0 device* islands_7 [[buffer(24)]], BondStatic_natural_0 device* bond_static_7 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(13)]], uint device* sinc_7 [[buffer(9)]], packed_uint4 device* slook_7 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(15)]], uint device* ctl_7 [[buffer(1)]], float device* work_7 [[buffer(18)]], packed_float4 device* nodes_7 [[buffer(3)]], packed_uint2 device* key_node_7 [[buffer(12)]], packed_float4 device* sc_7 [[buffer(7)]], packed_float4 device* bonds_7 [[buffer(4)]], packed_float4 device* sb_7 [[buffer(8)]], packed_uint4 device* bedge_7 [[buffer(22)]], packed_float4 device* calls_7 [[buffer(2)]], packed_float4 device* imps_7 [[buffer(16)]], packed_uint4 device* cisl_7 [[buffer(5)]], packed_float4 device* cislf_7 [[buffer(6)]], packed_float4 device* contacts_7 [[buffer(17)]], uint device* fctl_7 [[buffer(23)]], uint device* edges_7 [[buffer(20)]], uint device* cols_7 [[buffer(21)]], float device* feat_7 [[buffer(19)]], packed_float4 device* ctab_7 [[buffer(25)]], packed_float4 device* bodies_7 [[buffer(26)]], packed_float4 device* state_7 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_24;
    (&kernelContext_24)->params_0 = params_7;
    (&kernelContext_24)->bmap_0 = bmap_7;
    (&kernelContext_24)->counters_0 = counters_7;
    (&kernelContext_24)->islands_0 = islands_7;
    (&kernelContext_24)->bond_static_0 = bond_static_7;
    (&kernelContext_24)->chunks_0 = chunks_7;
    (&kernelContext_24)->sinc_0 = sinc_7;
    (&kernelContext_24)->slook_0 = slook_7;
    (&kernelContext_24)->bond_dyn_0 = bond_dyn_7;
    (&kernelContext_24)->ctl_0 = ctl_7;
    (&kernelContext_24)->work_0 = work_7;
    (&kernelContext_24)->nodes_0 = nodes_7;
    (&kernelContext_24)->key_node_0 = key_node_7;
    (&kernelContext_24)->sc_0 = sc_7;
    (&kernelContext_24)->bonds_0 = bonds_7;
    (&kernelContext_24)->sb_0 = sb_7;
    (&kernelContext_24)->bedge_0 = bedge_7;
    (&kernelContext_24)->calls_0 = calls_7;
    (&kernelContext_24)->imps_0 = imps_7;
    (&kernelContext_24)->cisl_0 = cisl_7;
    (&kernelContext_24)->cislf_0 = cislf_7;
    (&kernelContext_24)->contacts_0 = contacts_7;
    (&kernelContext_24)->fctl_0 = fctl_7;
    (&kernelContext_24)->edges_0 = edges_7;
    (&kernelContext_24)->cols_0 = cols_7;
    (&kernelContext_24)->feat_0 = feat_7;
    (&kernelContext_24)->ctab_0 = ctab_7;
    (&kernelContext_24)->bodies_0 = bodies_7;
    (&kernelContext_24)->state_0 = state_7;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_7;
    (&kernelContext_24)->g_red_0 = &g_red_7;
    threadgroup array<float, int(64)> g_mv_7;
    (&kernelContext_24)->g_mv_0 = &g_mv_7;
    threadgroup array<uint, int(64)> g_mi_7;
    (&kernelContext_24)->g_mi_0 = &g_mi_7;
    threadgroup array<uint, int(4096)> g_plug_7;
    (&kernelContext_24)->g_plug_0 = &g_plug_7;
    threadgroup uint g_flag_7;
    (&kernelContext_24)->g_flag_0 = &g_flag_7;
    threadgroup array<uint, int(64)> g_scan_7;
    (&kernelContext_24)->g_scan_0 = &g_scan_7;
    threadgroup float4 g_scalars_7;
    (&kernelContext_24)->g_scalars_0 = &g_scalars_7;
    uint n_5 = id_3.x;
    bool _S165;
    if(n_5 >= ctl_7[int(2)])
    {
        _S165 = true;
    }
    else
    {
        float4 _S166 = node_0(n_5, 4U, &kernelContext_24);
        _S165 = (as_type<uint>((_S166.w))) != 0U;
    }
    if(_S165)
    {
        return;
    }
    float4 _S167 = node_0(n_5, 12U, &kernelContext_24);
    uint _S168 = (&kernelContext_24)->sinc_0[int(1)];
    uint off_0 = 2U + (&kernelContext_24)->sinc_0[int(0)];
    uint _S169 = off_0 + ((&kernelContext_24)->sinc_0[2U + (as_type<uint>((_S167.z)))] + (as_type<uint>((_S167.w))));
    uint a_4 = (&kernelContext_24)->sinc_0[_S169];
    uint _S170 = (&kernelContext_24)->sinc_0[_S169 + 1U];
    float dmax_0 = 0.0f;
    uint i_2 = a_4;
    float nlive_1 = 0.0f;
    float dsum_0 = 0.0f;
    float nbroken_0 = 0.0f;
    for(;;)
    {
        if(i_2 < _S170)
        {
        }
        else
        {
            break;
        }
        uint device* _S171 = (&kernelContext_24)->bmap_0+((&kernelContext_24)->params_0->layout_cap_0 + (&kernelContext_24)->sinc_0[off_0 + _S168 + 1U + i_2]);
        uint l_2 = *_S171;
        if((*_S171) != 4294967295U)
        {
            bool _S172 = ((float4(((&kernelContext_24)->bond_static_0+l_2)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S173 = ((&kernelContext_24)->bond_dyn_0+l_2)->js_0;
            bool _S174 = connected_0(&_S173, _S172);
            _S165 = _S174;
        }
        else
        {
            _S165 = false;
        }
        if(_S165)
        {
            float nlive_2 = nlive_1 + 1.0f;
            float dsum_1 = dsum_0 + ((&kernelContext_24)->bond_dyn_0+l_2)->js_0.damage_0;
            dmax_0 = max(dmax_0, ((&kernelContext_24)->bond_dyn_0+l_2)->js_0.damage_0);
            nlive_1 = nlive_2;
            dsum_0 = dsum_1;
        }
        else
        {
            nbroken_0 = nbroken_0 + 1.0f;
        }
        i_2 = i_2 + 1U;
    }
    uint _S175 = 24U * n_5;
    *((&kernelContext_24)->work_0+(_S175 + 5U)) = dsum_0;
    *((&kernelContext_24)->work_0+(_S175 + 6U)) = dmax_0;
    *((&kernelContext_24)->work_0+(_S175 + 7U)) = nlive_1;
    *((&kernelContext_24)->work_0+(_S175 + 8U)) = nbroken_0;
    return;
}

float sphere_contact_time_0(float3 dx_0, float3 dv_0, float r_1, float dt_1)
{
    float c_0 = dot(dx_0, dx_0) - r_1 * r_1;
    if(c_0 <= 0.0f)
    {
        return 0.0f;
    }
    float a_5 = dot(dv_0, dv_0);
    float b_2 = dot(dx_0, dv_0);
    bool _S176;
    if(a_5 <= 0.0f)
    {
        _S176 = true;
    }
    else
    {
        _S176 = b_2 >= 0.0f;
    }
    if(_S176)
    {
        return -1.0f;
    }
    float disc_0 = b_2 * b_2 - a_5 * c_0;
    if(disc_0 < 0.0f)
    {
        return -1.0f;
    }
    float t_7 = (- b_2 - sqrt(disc_0)) / a_5;
    float _S177;
    if(t_7 <= dt_1)
    {
        _S177 = max(t_7, 0.0f);
    }
    else
    {
        _S177 = -1.0f;
    }
    return _S177;
}

[[kernel]] void feat_graph(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_8 [[buffer(0)]], uint device* bmap_8 [[buffer(11)]], uint device* counters_8 [[buffer(28)]], Island_natural_0 device* islands_8 [[buffer(24)]], BondStatic_natural_0 device* bond_static_8 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(13)]], uint device* sinc_8 [[buffer(9)]], packed_uint4 device* slook_8 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(15)]], uint device* ctl_8 [[buffer(1)]], float device* work_8 [[buffer(18)]], packed_float4 device* nodes_8 [[buffer(3)]], packed_uint2 device* key_node_8 [[buffer(12)]], packed_float4 device* sc_8 [[buffer(7)]], packed_float4 device* bonds_8 [[buffer(4)]], packed_float4 device* sb_8 [[buffer(8)]], packed_uint4 device* bedge_8 [[buffer(22)]], packed_float4 device* calls_8 [[buffer(2)]], packed_float4 device* imps_8 [[buffer(16)]], packed_uint4 device* cisl_8 [[buffer(5)]], packed_float4 device* cislf_8 [[buffer(6)]], packed_float4 device* contacts_8 [[buffer(17)]], uint device* fctl_8 [[buffer(23)]], uint device* edges_8 [[buffer(20)]], uint device* cols_8 [[buffer(21)]], float device* feat_8 [[buffer(19)]], packed_float4 device* ctab_8 [[buffer(25)]], packed_float4 device* bodies_8 [[buffer(26)]], packed_float4 device* state_8 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_25;
    (&kernelContext_25)->params_0 = params_8;
    (&kernelContext_25)->bmap_0 = bmap_8;
    (&kernelContext_25)->counters_0 = counters_8;
    (&kernelContext_25)->islands_0 = islands_8;
    (&kernelContext_25)->bond_static_0 = bond_static_8;
    (&kernelContext_25)->chunks_0 = chunks_8;
    (&kernelContext_25)->sinc_0 = sinc_8;
    (&kernelContext_25)->slook_0 = slook_8;
    (&kernelContext_25)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_25)->ctl_0 = ctl_8;
    (&kernelContext_25)->work_0 = work_8;
    (&kernelContext_25)->nodes_0 = nodes_8;
    (&kernelContext_25)->key_node_0 = key_node_8;
    (&kernelContext_25)->sc_0 = sc_8;
    (&kernelContext_25)->bonds_0 = bonds_8;
    (&kernelContext_25)->sb_0 = sb_8;
    (&kernelContext_25)->bedge_0 = bedge_8;
    (&kernelContext_25)->calls_0 = calls_8;
    (&kernelContext_25)->imps_0 = imps_8;
    (&kernelContext_25)->cisl_0 = cisl_8;
    (&kernelContext_25)->cislf_0 = cislf_8;
    (&kernelContext_25)->contacts_0 = contacts_8;
    (&kernelContext_25)->fctl_0 = fctl_8;
    (&kernelContext_25)->edges_0 = edges_8;
    (&kernelContext_25)->cols_0 = cols_8;
    (&kernelContext_25)->feat_0 = feat_8;
    (&kernelContext_25)->ctab_0 = ctab_8;
    (&kernelContext_25)->bodies_0 = bodies_8;
    (&kernelContext_25)->state_0 = state_8;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_8;
    (&kernelContext_25)->g_red_0 = &g_red_8;
    threadgroup array<float, int(64)> g_mv_8;
    (&kernelContext_25)->g_mv_0 = &g_mv_8;
    threadgroup array<uint, int(64)> g_mi_8;
    (&kernelContext_25)->g_mi_0 = &g_mi_8;
    threadgroup array<uint, int(4096)> g_plug_8;
    (&kernelContext_25)->g_plug_0 = &g_plug_8;
    threadgroup uint g_flag_8;
    (&kernelContext_25)->g_flag_0 = &g_flag_8;
    threadgroup array<uint, int(64)> g_scan_8;
    (&kernelContext_25)->g_scan_0 = &g_scan_8;
    threadgroup float4 g_scalars_8;
    (&kernelContext_25)->g_scalars_0 = &g_scalars_8;
    uint g_9 = group_2.x;
    uint t_8 = thread_4.x;
    uint j_9;
    if(g_9 >= ctl_8[int(1)])
    {
        j_9 = 1U;
    }
    else
    {
        j_9 = 0U;
    }
    if((group_uniform_0(j_9, t_8)) != 0U)
    {
        return;
    }
    uint _S178 = call_first_0(g_9, &kernelContext_25);
    uint _S179 = call_chunks_0(g_9, &kernelContext_25);
    uint _S180 = group_uniform_0(_S179, t_8);
    uint _S181 = call_bfirst_0(g_9, &kernelContext_25);
    uint _S182 = call_nbonds_0(g_9, &kernelContext_25);
    uint _S183 = group_uniform_0(_S182, t_8);
    uint _S184 = 10U * g_9;
    uint _S185 = _S184 + 8U;
    uint _S186 = (as_type<uint>(((float4(*((&kernelContext_25)->calls_0+_S185)) ).x)));
    uint _S187 = group_uniform_0((as_type<uint>(((float4(*((&kernelContext_25)->calls_0+_S185)) ).y))), t_8);
    float _S188 = (float4(*((&kernelContext_25)->calls_0+_S184)) ).w;
    float _S189 = float_bits_0(2139095040U);
    float _S190 = (&kernelContext_25)->params_0->dt_0;
    j_9 = t_8;
    for(;;)
    {
        if(j_9 < _S180)
        {
        }
        else
        {
            break;
        }
        uint _S191 = 24U * (_S178 + j_9);
        *((&kernelContext_25)->work_0+_S191) = _S189;
        *((&kernelContext_25)->work_0+(_S191 + 1U)) = _S189;
        float device* _S192 = (&kernelContext_25)->work_0+(_S191 + 2U);
        float _S193;
        if((*((&kernelContext_25)->work_0+(_S191 + 14U))) != 0.0f)
        {
            _S193 = 0.0f;
        }
        else
        {
            _S193 = _S189;
        }
        *_S192 = _S193;
        j_9 = j_9 + 64U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    bool _S194 = t_8 == 0U;
    uint iter_1;
    if(_S194)
    {
        iter_1 = 0U;
        for(;;)
        {
            if(iter_1 < _S187)
            {
            }
            else
            {
                break;
            }
            uint _S195 = 6U * (_S186 + iter_1);
            float4 _S196 = float4(*((&kernelContext_25)->contacts_0+_S195)) ;
            if(((float4(*((&kernelContext_25)->contacts_0+(_S195 + 5U))) ).x) == 0.0f)
            {
                iter_1 = iter_1 + 1U;
                continue;
            }
            uint en_0 = (as_type<uint>((_S196.y)));
            uint k_4 = (as_type<uint>((_S196.x)));
            uint _S197 = 24U * en_0;
            *((&kernelContext_25)->work_0+_S197) = 0.0f;
            uint _S198 = _S178 + _S180;
            uint _S199 = call_mask_0(g_9, &kernelContext_25);
            uint inode_1 = _S198 + ((uint)popcount((_S199 & ((1U << k_4) - 1U))));
            float r_imp_0 = (float4(*((&kernelContext_25)->imps_0+(4U * k_4 + 2U))) ).x;
            float4 _S200 = node_0(en_0, 13U, &kernelContext_25);
            float3 _S201 = _S200.xyz;
            float4 _S202 = node_0(inode_1, 13U, &kernelContext_25);
            float3 _S203 = _S201 - _S202.xyz;
            float4 _S204 = node_0(en_0, 3U, &kernelContext_25);
            float3 _S205 = _S204.xyz;
            float4 _S206 = node_0(inode_1, 3U, &kernelContext_25);
            float tc_0 = sphere_contact_time_0(_S203, _S205 - _S206.xyz, r_imp_0 + *((&kernelContext_25)->work_0+(_S197 + 12U)), _S190);
            if(tc_0 >= 0.0f)
            {
                uint _S207 = _S197 + 1U;
                *((&kernelContext_25)->work_0+_S207) = min(*((&kernelContext_25)->work_0+_S207), tc_0);
            }
            iter_1 = iter_1 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    iter_1 = 0U;
    for(;;)
    {
        if(iter_1 < (_S180 + 1U))
        {
        }
        else
        {
            break;
        }
        if(_S194)
        {
            *(&kernelContext_25)->g_flag_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        j_9 = t_8;
        for(;;)
        {
            if(j_9 < _S183)
            {
            }
            else
            {
                break;
            }
            uint4 _S208 = uint4(*((&kernelContext_25)->bedge_0+(_S181 + j_9))) ;
            uint _S209 = _S208.x;
            bool _S210;
            if(_S209 == 4294967295U)
            {
                _S210 = true;
            }
            else
            {
                _S210 = (_S208.y) == 4294967295U;
            }
            if(_S210)
            {
                j_9 = j_9 + 64U;
                continue;
            }
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
                bool _S211 = side_0 == 0U;
                uint u_0;
                if(_S211)
                {
                    u_0 = _S209;
                }
                else
                {
                    u_0 = _S208.y;
                }
                uint w_0;
                if(_S211)
                {
                    w_0 = _S208.y;
                }
                else
                {
                    w_0 = _S209;
                }
                uint _S212 = 24U * u_0;
                float hu_0 = *((&kernelContext_25)->work_0+_S212) + 1.0f;
                uint _S213 = 24U * w_0;
                if(hu_0 < (*((&kernelContext_25)->work_0+_S213)))
                {
                    *((&kernelContext_25)->work_0+_S213) = hu_0;
                    *(&kernelContext_25)->g_flag_0 = 1U;
                }
                float au_0 = *((&kernelContext_25)->work_0+(_S212 + 2U)) + 1.0f;
                uint _S214 = _S213 + 2U;
                if(au_0 < (*((&kernelContext_25)->work_0+_S214)))
                {
                    *((&kernelContext_25)->work_0+_S214) = au_0;
                    *(&kernelContext_25)->g_flag_0 = 1U;
                }
                float4 _S215 = node_0(u_0, 13U, &kernelContext_25);
                float3 _S216 = _S215.xyz;
                float4 _S217 = node_0(w_0, 13U, &kernelContext_25);
                float tu_0 = *((&kernelContext_25)->work_0+(_S212 + 1U)) + length(_S216 - _S217.xyz) / _S188;
                uint _S218 = _S213 + 1U;
                if(tu_0 < (*((&kernelContext_25)->work_0+_S218)))
                {
                    *((&kernelContext_25)->work_0+_S218) = tu_0;
                    *(&kernelContext_25)->g_flag_0 = 1U;
                }
                side_0 = side_0 + 1U;
            }
            j_9 = j_9 + 64U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if((group_uniform_0(*(&kernelContext_25)->g_flag_0, t_8)) == 0U)
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        iter_1 = iter_1 + 1U;
    }
    return;
}

uint edge_base_0(uint g_10, uint s_4, KernelContext_0 thread* kernelContext_26)
{
    return *(kernelContext_26->fctl_0+(8U + 6U * g_10 + 3U + s_4));
}

matrix<float,int(3),int(3)>  quat_matrix_0(float4 q_2)
{
    float w_1 = q_2.w;
    float x_0 = q_2.x;
    float y_0 = q_2.y;
    float z_0 = q_2.z;
    float _S219 = y_0 * y_0;
    float _S220 = z_0 * z_0;
    float _S221 = x_0 * y_0;
    float _S222 = w_1 * z_0;
    float _S223 = x_0 * z_0;
    float _S224 = w_1 * y_0;
    float _S225 = x_0 * x_0;
    float _S226 = y_0 * z_0;
    float _S227 = w_1 * x_0;
    return matrix<float,int(3),int(3)> (float3(1.0f - 2.0f * (_S219 + _S220), 2.0f * (_S221 - _S222), 2.0f * (_S223 + _S224)), float3(2.0f * (_S221 + _S222), 1.0f - 2.0f * (_S225 + _S220), 2.0f * (_S226 - _S227)), float3(2.0f * (_S223 - _S224), 2.0f * (_S226 + _S227), 1.0f - 2.0f * (_S225 + _S219)));
}

float3 col3_0(matrix<float,int(3),int(3)>  m_3, uint c_1)
{
    return float3(m_3[int(0)][c_1], m_3[int(1)][c_1], m_3[int(2)][c_1]);
}

float3 tangent_of_0(float3 n_6, matrix<float,int(3),int(3)>  body_0)
{
    uint a_6 = 0U;
    for(;;)
    {
        if(a_6 < 2U)
        {
        }
        else
        {
            break;
        }
        float3 axis_0 = col3_0(body_0, a_6);
        float3 t_9 = axis_0 - n_6 * float3(dot(axis_0, n_6)) ;
        if((length(t_9)) > 9.99999997475242708e-07f)
        {
            return normalize(t_9);
        }
        a_6 = a_6 + 1U;
    }
    float3 axis_1 = col3_0(body_0, 2U);
    return normalize(axis_1 - n_6 * float3(dot(axis_1, n_6)) );
}

float support_of_0(matrix<float,int(3),int(3)>  r_2, float3 half_1, float3 n_7)
{
    float3 l_3 = (((n_7) * (transpose(r_2))));
    return abs(l_3.x) * half_1.x + abs(l_3.y) * half_1.y + abs(l_3.z) * half_1.z;
}

float ln_ratio_0(float a_7, float b_3)
{
    return log(max(a_7, 9.99999997475242708e-07f) / max(b_3, 9.99999997475242708e-07f));
}

uint call_cbase_0(uint g_11, KernelContext_0 thread* kernelContext_27)
{
    return (as_type<uint>(((float4(*(kernelContext_27->calls_0+(10U * g_11 + 6U))) ).z)));
}

uint call_ibase_0(uint g_12, KernelContext_0 thread* kernelContext_28)
{
    return (as_type<uint>(((float4(*(kernelContext_28->calls_0+(10U * g_12 + 6U))) ).w)));
}

uint global_node_0(uint g_13, uint j_10, KernelContext_0 thread* kernelContext_29)
{
    uint _S228 = call_chunks_0(g_13, kernelContext_29);
    if(j_10 < _S228)
    {
        uint _S229 = call_cbase_0(g_13, kernelContext_29);
        return _S229 + j_10;
    }
    uint _S230 = *(kernelContext_29->fctl_0+int(4));
    uint _S231 = call_ibase_0(g_13, kernelContext_29);
    return _S230 + _S231 + (j_10 - _S228);
}

void put_edge_0(uint section_0, uint row_2, uint src_0, uint dst_0, KernelContext_0 thread* kernelContext_30)
{
    uint base_2;
    if(section_0 == 4U)
    {
        base_2 = 0U;
    }
    else
    {
        if(section_0 == 5U)
        {
            base_2 = 2U * kernelContext_30->params_0->caps_0.x;
        }
        else
        {
            base_2 = 2U * (kernelContext_30->params_0->caps_0.x + kernelContext_30->params_0->caps_0.y);
        }
    }
    uint _S232 = base_2 + 2U * row_2;
    *(kernelContext_30->edges_0+_S232) = src_0;
    *(kernelContext_30->edges_0+(_S232 + 1U)) = dst_0;
    return;
}

uint col_of_0(uint section_1, uint k_5, KernelContext_0 thread* kernelContext_31)
{
    return kernelContext_31->cols_0[section_1 * 65U + 1U + k_5];
}

uint stride_of_0(uint section_2, KernelContext_0 thread* kernelContext_32)
{
    return kernelContext_32->cols_0[section_2 * 65U];
}

void put_0(uint base_3, uint section_3, uint row_3, uint k_6, float v_4, KernelContext_0 thread* kernelContext_33)
{
    uint _S233 = col_of_0(section_3, k_6, kernelContext_33);
    if(_S233 != 4294967295U)
    {
        uint _S234 = stride_of_0(section_3, kernelContext_33);
        *(kernelContext_33->feat_0+(base_3 + row_3 * _S234 + _S233)) = v_4;
    }
    return;
}

void put_1(uint base_4, uint section_4, uint row_4, uint k_7, float v_5, KernelContext_0 thread* kernelContext_34)
{
    uint _S235 = col_of_0(section_4, k_7, kernelContext_34);
    if(_S235 != 4294967295U)
    {
        uint _S236 = stride_of_0(section_4, kernelContext_34);
        *(kernelContext_34->feat_0+(base_4 + row_4 * _S236 + _S235)) = v_5;
    }
    return;
}

void put_rot6_0(uint base_5, uint section_5, uint row_5, uint k0_0, float3 t1_2, float3 t2_1, matrix<float,int(3),int(3)>  body_1, KernelContext_0 thread* kernelContext_35)
{
    matrix<float,int(3),int(3)>  _S237 = transpose(body_1);
    float3 a_8 = (((t1_2) * (_S237)));
    float3 b_4 = (((t2_1) * (_S237)));
    put_1(base_5, section_5, row_5, k0_0, a_8.x, kernelContext_35);
    put_1(base_5, section_5, row_5, k0_0 + 1U, a_8.y, kernelContext_35);
    put_1(base_5, section_5, row_5, k0_0 + 2U, a_8.z, kernelContext_35);
    put_1(base_5, section_5, row_5, k0_0 + 3U, b_4.x, kernelContext_35);
    put_1(base_5, section_5, row_5, k0_0 + 4U, b_4.y, kernelContext_35);
    put_1(base_5, section_5, row_5, k0_0 + 5U, b_4.z, kernelContext_35);
    return;
}

void put_key_0(uint section_6, uint k_8, uint a_9, uint b_5, KernelContext_0 thread* kernelContext_36)
{
    uint _S238 = 4U * (kernelContext_36->params_0->caps_0.x + kernelContext_36->params_0->caps_0.y + kernelContext_36->params_0->caps_0.z);
    uint _S239;
    if(section_6 == 4U)
    {
        _S239 = 0U;
    }
    else
    {
        if(section_6 == 5U)
        {
            _S239 = kernelContext_36->params_0->caps_0.x;
        }
        else
        {
            _S239 = kernelContext_36->params_0->caps_0.x + kernelContext_36->params_0->caps_0.y;
        }
    }
    uint _S240 = _S238 + _S239 + 2U * k_8;
    *(kernelContext_36->edges_0+_S240) = a_9;
    *(kernelContext_36->edges_0+(_S240 + 1U)) = b_5;
    return;
}

void edge_count_0(uint g_14, uint s_5, uint n_8, KernelContext_0 thread* kernelContext_37)
{
    *(kernelContext_37->fctl_0+(8U + 6U * g_14 + s_5)) = n_8;
    return;
}

[[kernel]] void feat_ic(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_9 [[buffer(0)]], uint device* bmap_9 [[buffer(11)]], uint device* counters_9 [[buffer(28)]], Island_natural_0 device* islands_9 [[buffer(24)]], BondStatic_natural_0 device* bond_static_9 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(13)]], uint device* sinc_9 [[buffer(9)]], packed_uint4 device* slook_9 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_9 [[buffer(15)]], uint device* ctl_9 [[buffer(1)]], float device* work_9 [[buffer(18)]], packed_float4 device* nodes_9 [[buffer(3)]], packed_uint2 device* key_node_9 [[buffer(12)]], packed_float4 device* sc_9 [[buffer(7)]], packed_float4 device* bonds_9 [[buffer(4)]], packed_float4 device* sb_9 [[buffer(8)]], packed_uint4 device* bedge_9 [[buffer(22)]], packed_float4 device* calls_9 [[buffer(2)]], packed_float4 device* imps_9 [[buffer(16)]], packed_uint4 device* cisl_9 [[buffer(5)]], packed_float4 device* cislf_9 [[buffer(6)]], packed_float4 device* contacts_9 [[buffer(17)]], uint device* fctl_9 [[buffer(23)]], uint device* edges_9 [[buffer(20)]], uint device* cols_9 [[buffer(21)]], float device* feat_9 [[buffer(19)]], packed_float4 device* ctab_9 [[buffer(25)]], packed_float4 device* bodies_9 [[buffer(26)]], packed_float4 device* state_9 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_38;
    (&kernelContext_38)->params_0 = params_9;
    (&kernelContext_38)->bmap_0 = bmap_9;
    (&kernelContext_38)->counters_0 = counters_9;
    (&kernelContext_38)->islands_0 = islands_9;
    (&kernelContext_38)->bond_static_0 = bond_static_9;
    (&kernelContext_38)->chunks_0 = chunks_9;
    (&kernelContext_38)->sinc_0 = sinc_9;
    (&kernelContext_38)->slook_0 = slook_9;
    (&kernelContext_38)->bond_dyn_0 = bond_dyn_9;
    (&kernelContext_38)->ctl_0 = ctl_9;
    (&kernelContext_38)->work_0 = work_9;
    (&kernelContext_38)->nodes_0 = nodes_9;
    (&kernelContext_38)->key_node_0 = key_node_9;
    (&kernelContext_38)->sc_0 = sc_9;
    (&kernelContext_38)->bonds_0 = bonds_9;
    (&kernelContext_38)->sb_0 = sb_9;
    (&kernelContext_38)->bedge_0 = bedge_9;
    (&kernelContext_38)->calls_0 = calls_9;
    (&kernelContext_38)->imps_0 = imps_9;
    (&kernelContext_38)->cisl_0 = cisl_9;
    (&kernelContext_38)->cislf_0 = cislf_9;
    (&kernelContext_38)->contacts_0 = contacts_9;
    (&kernelContext_38)->fctl_0 = fctl_9;
    (&kernelContext_38)->edges_0 = edges_9;
    (&kernelContext_38)->cols_0 = cols_9;
    (&kernelContext_38)->feat_0 = feat_9;
    (&kernelContext_38)->ctab_0 = ctab_9;
    (&kernelContext_38)->bodies_0 = bodies_9;
    (&kernelContext_38)->state_0 = state_9;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_9;
    (&kernelContext_38)->g_red_0 = &g_red_9;
    threadgroup array<float, int(64)> g_mv_9;
    (&kernelContext_38)->g_mv_0 = &g_mv_9;
    threadgroup array<uint, int(64)> g_mi_9;
    (&kernelContext_38)->g_mi_0 = &g_mi_9;
    threadgroup array<uint, int(4096)> g_plug_9;
    (&kernelContext_38)->g_plug_0 = &g_plug_9;
    threadgroup uint g_flag_9;
    (&kernelContext_38)->g_flag_0 = &g_flag_9;
    threadgroup array<uint, int(64)> g_scan_9;
    (&kernelContext_38)->g_scan_0 = &g_scan_9;
    threadgroup float4 g_scalars_9;
    (&kernelContext_38)->g_scalars_0 = &g_scalars_9;
    uint g_15 = group_3.x;
    uint t_10 = thread_5.x;
    bool _S241;
    if(g_15 >= ctl_9[int(1)])
    {
        _S241 = true;
    }
    else
    {
        _S241 = t_10 != 0U;
    }
    if(_S241)
    {
        return;
    }
    uint _S242 = call_first_0(g_15, &kernelContext_38);
    uint _S243 = call_chunks_0(g_15, &kernelContext_38);
    uint _S244 = 10U * g_15;
    uint _S245 = _S244 + 8U;
    uint _S246 = (as_type<uint>(((float4(*((&kernelContext_38)->calls_0+_S245)) ).x)));
    uint _S247 = (as_type<uint>(((float4(*((&kernelContext_38)->calls_0+_S245)) ).y)));
    float4 _S248 = float4(*((&kernelContext_38)->calls_0+_S244)) ;
    float _S249 = _S248.x;
    float _S250 = _S248.z;
    float _S251 = _S248.w;
    float _S252 = (float4(*((&kernelContext_38)->calls_0+(_S244 + 1U))) ).x;
    uint _S253;
    if(((&kernelContext_38)->params_0->count_only_0) != 0U)
    {
        _S253 = 0U;
    }
    else
    {
        uint _S254 = edge_base_0(g_15, 1U, &kernelContext_38);
        _S253 = _S254;
    }
    uint _S255 = (&kernelContext_38)->params_0->bases2_0.y;
    uint _S256 = _S244 + 9U;
    float _S257 = (float4(*((&kernelContext_38)->calls_0+_S256)) ).x;
    uint r_3 = 0U;
    uint count_0 = 0U;
    for(;;)
    {
        if(r_3 < _S247)
        {
        }
        else
        {
            break;
        }
        uint _S258 = 6U * (_S246 + r_3);
        if(((float4(*((&kernelContext_38)->contacts_0+(_S258 + 5U))) ).x) == 0.0f)
        {
            r_3 = r_3 + 1U;
            continue;
        }
        if(((&kernelContext_38)->params_0->count_only_0) == 0U)
        {
            float4 _S259 = float4(*((&kernelContext_38)->contacts_0+_S258)) ;
            float4 _S260 = float4(*((&kernelContext_38)->contacts_0+(_S258 + 1U))) ;
            float4 _S261 = float4(*((&kernelContext_38)->contacts_0+(_S258 + 2U))) ;
            float4 _S262 = float4(*((&kernelContext_38)->contacts_0+(_S258 + 3U))) ;
            uint k_9 = (as_type<uint>((_S259.x)));
            uint en_1 = (as_type<uint>((_S259.y)));
            uint _S263 = _S242 + _S243;
            uint _S264 = call_mask_0(g_15, &kernelContext_38);
            uint inode_2 = _S263 + ((uint)popcount((_S264 & ((1U << k_9) - 1U))));
            float4 _S265 = node_0(inode_2, 1U, &kernelContext_38);
            matrix<float,int(3),int(3)>  ri_0 = quat_matrix_0(_S265);
            float4 _S266 = node_0(en_1, 1U, &kernelContext_38);
            matrix<float,int(3),int(3)>  rb_1 = quat_matrix_0(_S266);
            float3 nrm_0 = normalize(_S261.xyz);
            float3 t1_3 = tangent_of_0(nrm_0, ri_0);
            float3 _S267 = cross(nrm_0, t1_3);
            uint _S268 = 24U * en_1;
            uint _S269 = _S268 + 9U;
            float3 half_c_0 = float3(*((&kernelContext_38)->work_0+_S269), *((&kernelContext_38)->work_0+(_S269 + 1U)), *((&kernelContext_38)->work_0+(_S269 + 2U)));
            uint _S270 = 4U * k_9;
            uint _S271 = _S270 + 2U;
            float hn_0 = (float4(*((&kernelContext_38)->imps_0+_S271)) ).x;
            float3 half_i_0 = float3((float4(*((&kernelContext_38)->imps_0+_S271)) ).y, (float4(*((&kernelContext_38)->imps_0+_S271)) ).z, (float4(*((&kernelContext_38)->imps_0+_S271)) ).w);
            float4 _S272 = node_0(en_1, 13U, &kernelContext_38);
            float3 _S273 = _S272.xyz;
            float4 _S274 = node_0(inode_2, 13U, &kernelContext_38);
            float _S275 = dot(_S273 - _S274.xyz, nrm_0) - support_of_0(ri_0, half_i_0, nrm_0) - support_of_0(rb_1, half_c_0, nrm_0);
            float _S276 = _S260.w;
            float ek_0 = 0.5f * _S261.w * _S276 * _S276;
            uint plug_row_0 = uint(*((&kernelContext_38)->work_0+(_S268 + 3U)));
            float ga_plug_0 = (float4(*((&kernelContext_38)->calls_0+_S256)) ).z;
            if(ga_plug_0 > 0.0f)
            {
                _S241 = plug_row_0 != 0U;
            }
            else
            {
                _S241 = false;
            }
            float _S277;
            if(_S241)
            {
                _S277 = ln_ratio_0(ek_0, ga_plug_0);
            }
            else
            {
                _S277 = 0.0f;
            }
            uint _S278 = _S270 + 3U;
            float fr_imp_0 = (float4(*((&kernelContext_38)->imps_0+_S278)) ).y;
            float fr_ch_0 = *((&kernelContext_38)->work_0+(_S268 + 13U));
            float _S279;
            if(((&kernelContext_38)->params_0->contact_friction_0) >= 0.0f)
            {
                _S279 = (&kernelContext_38)->params_0->contact_friction_0;
            }
            else
            {
                _S279 = min(fr_imp_0, fr_ch_0);
            }
            float _S280 = (float4(*((&kernelContext_38)->imps_0+_S278)) ).z;
            float4 _S281 = node_0(en_1, 13U, &kernelContext_38);
            float3 _S282 = _S281.xyz;
            float4 _S283 = node_0(inode_2, 13U, &kernelContext_38);
            float3 _S284 = _S282 - _S283.xyz;
            float4 _S285 = node_0(en_1, 3U, &kernelContext_38);
            float3 _S286 = _S285.xyz;
            float4 _S287 = node_0(inode_2, 3U, &kernelContext_38);
            float tc_1 = sphere_contact_time_0(_S284, _S286 - _S287.xyz, hn_0 + *((&kernelContext_38)->work_0+(_S268 + 12U)), (&kernelContext_38)->params_0->dt_0);
            float _S288;
            if(tc_1 < 0.0f)
            {
                _S288 = 1.0f;
            }
            else
            {
                if(tc_1 <= 0.0f)
                {
                    _S288 = -1.0f;
                }
                else
                {
                    _S288 = tc_1 / (&kernelContext_38)->params_0->dt_0;
                }
            }
            float _S289 = (float4(*((&kernelContext_38)->imps_0+_S278)) ).x;
            float _S290 = *((&kernelContext_38)->work_0+(_S268 + 15U));
            uint side_1 = 0U;
            for(;;)
            {
                if(side_1 < 2U)
                {
                }
                else
                {
                    break;
                }
                uint row_6 = _S253 + count_0 + side_1;
                bool _S291 = side_1 == 0U;
                uint sn_0;
                if(_S291)
                {
                    sn_0 = inode_2;
                }
                else
                {
                    sn_0 = en_1;
                }
                uint dn_0;
                if(_S291)
                {
                    dn_0 = en_1;
                }
                else
                {
                    dn_0 = inode_2;
                }
                float3 ft1_0;
                if(_S291)
                {
                    ft1_0 = t1_3;
                }
                else
                {
                    ft1_0 = - t1_3;
                }
                float3 fn_0;
                if(_S291)
                {
                    fn_0 = nrm_0;
                }
                else
                {
                    fn_0 = - nrm_0;
                }
                float4 _S292 = node_0(sn_0, 3U, &kernelContext_38);
                float3 _S293 = _S292.xyz;
                float4 _S294 = node_0(dn_0, 3U, &kernelContext_38);
                float3 rel_0 = _S293 - _S294.xyz;
                float _S295 = dot(ft1_0, rel_0);
                float _S296 = dot(_S267, rel_0);
                float _S297 = dot(fn_0, rel_0);
                float4 _S298 = node_0(inode_2, 4U, &kernelContext_38);
                float3 wi_0 = _S298.xyz * float3((_S280 / _S250)) ;
                float _S299 = dot(ft1_0, wi_0);
                float _S300 = dot(_S267, wi_0);
                float _S301 = dot(fn_0, wi_0);
                uint _S302 = global_node_0(g_15, sn_0 - _S242, &kernelContext_38);
                uint _S303 = global_node_0(g_15, dn_0 - _S242, &kernelContext_38);
                put_edge_0(5U, row_6, _S302, _S303, &kernelContext_38);
                put_0(_S255, 5U, row_6, 0U, _S297 / _S250, &kernelContext_38);
                put_0(_S255, 5U, row_6, 1U, _S295 / _S250, &kernelContext_38);
                put_0(_S255, 5U, row_6, 2U, _S296 / _S250, &kernelContext_38);
                put_0(_S255, 5U, row_6, 3U, _S297 / _S251, &kernelContext_38);
                float4 _S304 = node_0(dn_0, 0U, &kernelContext_38);
                float _S305 = _S304.w;
                float4 _S306 = node_0(sn_0, 0U, &kernelContext_38);
                put_0(_S255, 5U, row_6, 4U, ln_ratio_0(_S305, _S306.w), &kernelContext_38);
                put_0(_S255, 5U, row_6, 5U, ln_ratio_0(_S280, _S249), &kernelContext_38);
                put_0(_S255, 5U, row_6, 6U, ln_ratio_0(_S289, _S290), &kernelContext_38);
                put_0(_S255, 5U, row_6, 7U, _S299, &kernelContext_38);
                put_0(_S255, 5U, row_6, 8U, _S300, &kernelContext_38);
                put_0(_S255, 5U, row_6, 9U, _S301, &kernelContext_38);
                put_0(_S255, 5U, row_6, 10U, _S288, &kernelContext_38);
                put_0(_S255, 5U, row_6, 11U, _S277, &kernelContext_38);
                put_0(_S255, 5U, row_6, 12U, ln_ratio_0(_S262.w, _S252 * _S257), &kernelContext_38);
                put_0(_S255, 5U, row_6, 13U, _S262.z / (&kernelContext_38)->params_0->dt_0, &kernelContext_38);
                float _S307;
                if(_S291)
                {
                    _S307 = 1.0f;
                }
                else
                {
                    _S307 = 0.0f;
                }
                put_0(_S255, 5U, row_6, 14U, _S307, &kernelContext_38);
                put_0(_S255, 5U, row_6, 15U, max(- _S275, 0.0f) / _S249, &kernelContext_38);
                put_0(_S255, 5U, row_6, 16U, _S279, &kernelContext_38);
                put_0(_S255, 5U, row_6, 17U, (&kernelContext_38)->params_0->restitution_0, &kernelContext_38);
                float4 _S308 = node_0(sn_0, 1U, &kernelContext_38);
                put_rot6_0(_S255, 5U, row_6, 18U, ft1_0, _S267, quat_matrix_0(_S308), &kernelContext_38);
                float4 _S309 = node_0(dn_0, 1U, &kernelContext_38);
                put_rot6_0(_S255, 5U, row_6, 24U, ft1_0, _S267, quat_matrix_0(_S309), &kernelContext_38);
                *((&kernelContext_38)->work_0+(24U * sn_0 + 4U)) = 1.0f;
                side_1 = side_1 + 1U;
            }
            uint _S310 = (_S253 + count_0) / 2U;
            float4 _S311 = node_0(en_1, 12U, &kernelContext_38);
            uint _S312 = (as_type<uint>((_S311.z)));
            float4 _S313 = node_0(en_1, 12U, &kernelContext_38);
            put_key_0(5U, _S310, _S312, (as_type<uint>((_S313.w))) | (k_9 << 24U), &kernelContext_38);
        }
        count_0 = count_0 + 2U;
        r_3 = r_3 + 1U;
    }
    if(((&kernelContext_38)->params_0->count_only_0) != 0U)
    {
        edge_count_0(g_15, 1U, count_0, &kernelContext_38);
    }
    return;
}

[[kernel]] void feat_cracked_find(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_10 [[buffer(0)]], uint device* bmap_10 [[buffer(11)]], uint device* counters_10 [[buffer(28)]], Island_natural_0 device* islands_10 [[buffer(24)]], BondStatic_natural_0 device* bond_static_10 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(13)]], uint device* sinc_10 [[buffer(9)]], packed_uint4 device* slook_10 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_10 [[buffer(15)]], uint device* ctl_10 [[buffer(1)]], float device* work_10 [[buffer(18)]], packed_float4 device* nodes_10 [[buffer(3)]], packed_uint2 device* key_node_10 [[buffer(12)]], packed_float4 device* sc_10 [[buffer(7)]], packed_float4 device* bonds_10 [[buffer(4)]], packed_float4 device* sb_10 [[buffer(8)]], packed_uint4 device* bedge_10 [[buffer(22)]], packed_float4 device* calls_10 [[buffer(2)]], packed_float4 device* imps_10 [[buffer(16)]], packed_uint4 device* cisl_10 [[buffer(5)]], packed_float4 device* cislf_10 [[buffer(6)]], packed_float4 device* contacts_10 [[buffer(17)]], uint device* fctl_10 [[buffer(23)]], uint device* edges_10 [[buffer(20)]], uint device* cols_10 [[buffer(21)]], float device* feat_10 [[buffer(19)]], packed_float4 device* ctab_10 [[buffer(25)]], packed_float4 device* bodies_10 [[buffer(26)]], packed_float4 device* state_10 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_39;
    (&kernelContext_39)->params_0 = params_10;
    (&kernelContext_39)->bmap_0 = bmap_10;
    (&kernelContext_39)->counters_0 = counters_10;
    (&kernelContext_39)->islands_0 = islands_10;
    (&kernelContext_39)->bond_static_0 = bond_static_10;
    (&kernelContext_39)->chunks_0 = chunks_10;
    (&kernelContext_39)->sinc_0 = sinc_10;
    (&kernelContext_39)->slook_0 = slook_10;
    (&kernelContext_39)->bond_dyn_0 = bond_dyn_10;
    (&kernelContext_39)->ctl_0 = ctl_10;
    (&kernelContext_39)->work_0 = work_10;
    (&kernelContext_39)->nodes_0 = nodes_10;
    (&kernelContext_39)->key_node_0 = key_node_10;
    (&kernelContext_39)->sc_0 = sc_10;
    (&kernelContext_39)->bonds_0 = bonds_10;
    (&kernelContext_39)->sb_0 = sb_10;
    (&kernelContext_39)->bedge_0 = bedge_10;
    (&kernelContext_39)->calls_0 = calls_10;
    (&kernelContext_39)->imps_0 = imps_10;
    (&kernelContext_39)->cisl_0 = cisl_10;
    (&kernelContext_39)->cislf_0 = cislf_10;
    (&kernelContext_39)->contacts_0 = contacts_10;
    (&kernelContext_39)->fctl_0 = fctl_10;
    (&kernelContext_39)->edges_0 = edges_10;
    (&kernelContext_39)->cols_0 = cols_10;
    (&kernelContext_39)->feat_0 = feat_10;
    (&kernelContext_39)->ctab_0 = ctab_10;
    (&kernelContext_39)->bodies_0 = bodies_10;
    (&kernelContext_39)->state_0 = state_10;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_10;
    (&kernelContext_39)->g_red_0 = &g_red_10;
    threadgroup array<float, int(64)> g_mv_10;
    (&kernelContext_39)->g_mv_0 = &g_mv_10;
    threadgroup array<uint, int(64)> g_mi_10;
    (&kernelContext_39)->g_mi_0 = &g_mi_10;
    threadgroup array<uint, int(4096)> g_plug_10;
    (&kernelContext_39)->g_plug_0 = &g_plug_10;
    threadgroup uint g_flag_10;
    (&kernelContext_39)->g_flag_0 = &g_flag_10;
    threadgroup array<uint, int(64)> g_scan_10;
    (&kernelContext_39)->g_scan_0 = &g_scan_10;
    threadgroup float4 g_scalars_10;
    (&kernelContext_39)->g_scalars_0 = &g_scalars_10;
    uint g_16 = group_4.x;
    uint t_11 = thread_6.x;
    bool _S314;
    if(g_16 >= ctl_10[int(1)])
    {
        _S314 = true;
    }
    else
    {
        _S314 = t_11 != 0U;
    }
    if(_S314)
    {
        return;
    }
    uint _S315 = call_first_0(g_16, &kernelContext_39);
    uint _S316 = call_chunks_0(g_16, &kernelContext_39);
    uint _S317 = (&kernelContext_39)->sinc_0[int(1)];
    uint _S318 = 2U + (&kernelContext_39)->sinc_0[int(0)];
    uint slots_0 = (&kernelContext_39)->params_0->max_bonds_cap_0 + 8U * _S315;
    uint cap_0 = 8U * _S316 - 1U;
    uint j_11 = 0U;
    uint count_1 = 0U;
    for(;;)
    {
        if(j_11 < _S316)
        {
        }
        else
        {
            break;
        }
        uint na_0 = _S315 + j_11;
        float4 _S319 = node_0(na_0, 12U, &kernelContext_39);
        uint _S320 = (&kernelContext_39)->sinc_0[2U + (as_type<uint>((_S319.z)))];
        float4 _S321 = node_0(na_0, 12U, &kernelContext_39);
        uint rowa_0 = _S320 + (as_type<uint>((_S321.w)));
        uint _S322 = _S318 + rowa_0;
        uint i_3 = (&kernelContext_39)->sinc_0[_S322];
        uint count_2 = count_1;
        for(;;)
        {
            if(i_3 < ((&kernelContext_39)->sinc_0)[_S322 + 1U])
            {
            }
            else
            {
                break;
            }
            uint br_0 = (&kernelContext_39)->sinc_0[_S318 + _S317 + 1U + i_3];
            uint4 ends_1 = (as_type<uint4>((float4(*((&kernelContext_39)->sb_0+(9U * br_0 + 7U))) )));
            if((ends_1.x) != rowa_0)
            {
                i_3 = i_3 + 1U;
                continue;
            }
            uint _S323 = node_of_key_0(ends_1.y, &kernelContext_39);
            if(_S323 == 4294967295U)
            {
                _S314 = true;
            }
            else
            {
                _S314 = _S323 < _S315;
            }
            bool _S324;
            if(_S314)
            {
                _S324 = true;
            }
            else
            {
                _S324 = _S323 >= (_S315 + _S316);
            }
            if(_S324)
            {
                i_3 = i_3 + 1U;
                continue;
            }
            uint device* _S325 = (&kernelContext_39)->bmap_0+((&kernelContext_39)->params_0->layout_cap_0 + br_0);
            uint l_4 = *_S325;
            bool _S326;
            if((*_S325) != 4294967295U)
            {
                bool _S327 = ((float4(((&kernelContext_39)->bond_static_0+l_4)->law_0.stiff1_0) ).w) != 0.0f;
                thread JointState_0 _S328 = ((&kernelContext_39)->bond_dyn_0+l_4)->js_0;
                bool _S329 = connected_0(&_S328, _S327);
                _S326 = _S329;
            }
            else
            {
                _S326 = false;
            }
            if(_S326)
            {
                i_3 = i_3 + 1U;
                continue;
            }
            if(count_2 < cap_0)
            {
                *((&kernelContext_39)->bedge_0+(slots_0 + 1U + count_2)) = packed_uint4(uint4(br_0, na_0, _S323, *((&kernelContext_39)->bmap_0+((&kernelContext_39)->params_0->layout_cap_0 + (&kernelContext_39)->params_0->rows_0 + br_0)))) ;
            }
            count_2 = count_2 + 1U;
            i_3 = i_3 + 1U;
        }
        j_11 = j_11 + 1U;
        count_1 = count_2;
    }
    packed_uint4 device* _S330 = (&kernelContext_39)->bedge_0+slots_0;
    uint _S331 = min(count_1, cap_0);
    if(count_1 > cap_0)
    {
        j_11 = 1U;
    }
    else
    {
        j_11 = 0U;
    }
    *_S330 = packed_uint4(uint4(_S331, j_11, 0U, 0U)) ;
    return;
}

[[kernel]] void feat_cracked(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_11 [[buffer(0)]], uint device* bmap_11 [[buffer(11)]], uint device* counters_11 [[buffer(28)]], Island_natural_0 device* islands_11 [[buffer(24)]], BondStatic_natural_0 device* bond_static_11 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(13)]], uint device* sinc_11 [[buffer(9)]], packed_uint4 device* slook_11 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_11 [[buffer(15)]], uint device* ctl_11 [[buffer(1)]], float device* work_11 [[buffer(18)]], packed_float4 device* nodes_11 [[buffer(3)]], packed_uint2 device* key_node_11 [[buffer(12)]], packed_float4 device* sc_11 [[buffer(7)]], packed_float4 device* bonds_11 [[buffer(4)]], packed_float4 device* sb_11 [[buffer(8)]], packed_uint4 device* bedge_11 [[buffer(22)]], packed_float4 device* calls_11 [[buffer(2)]], packed_float4 device* imps_11 [[buffer(16)]], packed_uint4 device* cisl_11 [[buffer(5)]], packed_float4 device* cislf_11 [[buffer(6)]], packed_float4 device* contacts_11 [[buffer(17)]], uint device* fctl_11 [[buffer(23)]], uint device* edges_11 [[buffer(20)]], uint device* cols_11 [[buffer(21)]], float device* feat_11 [[buffer(19)]], packed_float4 device* ctab_11 [[buffer(25)]], packed_float4 device* bodies_11 [[buffer(26)]], packed_float4 device* state_11 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_40;
    (&kernelContext_40)->params_0 = params_11;
    (&kernelContext_40)->bmap_0 = bmap_11;
    (&kernelContext_40)->counters_0 = counters_11;
    (&kernelContext_40)->islands_0 = islands_11;
    (&kernelContext_40)->bond_static_0 = bond_static_11;
    (&kernelContext_40)->chunks_0 = chunks_11;
    (&kernelContext_40)->sinc_0 = sinc_11;
    (&kernelContext_40)->slook_0 = slook_11;
    (&kernelContext_40)->bond_dyn_0 = bond_dyn_11;
    (&kernelContext_40)->ctl_0 = ctl_11;
    (&kernelContext_40)->work_0 = work_11;
    (&kernelContext_40)->nodes_0 = nodes_11;
    (&kernelContext_40)->key_node_0 = key_node_11;
    (&kernelContext_40)->sc_0 = sc_11;
    (&kernelContext_40)->bonds_0 = bonds_11;
    (&kernelContext_40)->sb_0 = sb_11;
    (&kernelContext_40)->bedge_0 = bedge_11;
    (&kernelContext_40)->calls_0 = calls_11;
    (&kernelContext_40)->imps_0 = imps_11;
    (&kernelContext_40)->cisl_0 = cisl_11;
    (&kernelContext_40)->cislf_0 = cislf_11;
    (&kernelContext_40)->contacts_0 = contacts_11;
    (&kernelContext_40)->fctl_0 = fctl_11;
    (&kernelContext_40)->edges_0 = edges_11;
    (&kernelContext_40)->cols_0 = cols_11;
    (&kernelContext_40)->feat_0 = feat_11;
    (&kernelContext_40)->ctab_0 = ctab_11;
    (&kernelContext_40)->bodies_0 = bodies_11;
    (&kernelContext_40)->state_0 = state_11;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_11;
    (&kernelContext_40)->g_red_0 = &g_red_11;
    threadgroup array<float, int(64)> g_mv_11;
    (&kernelContext_40)->g_mv_0 = &g_mv_11;
    threadgroup array<uint, int(64)> g_mi_11;
    (&kernelContext_40)->g_mi_0 = &g_mi_11;
    threadgroup array<uint, int(4096)> g_plug_11;
    (&kernelContext_40)->g_plug_0 = &g_plug_11;
    threadgroup uint g_flag_11;
    (&kernelContext_40)->g_flag_0 = &g_flag_11;
    threadgroup array<uint, int(64)> g_scan_11;
    (&kernelContext_40)->g_scan_0 = &g_scan_11;
    threadgroup float4 g_scalars_11;
    (&kernelContext_40)->g_scalars_0 = &g_scalars_11;
    uint g_17 = group_5.x;
    uint t_12 = thread_7.x;
    bool _S332;
    if(g_17 >= ctl_11[int(1)])
    {
        _S332 = true;
    }
    else
    {
        _S332 = t_12 != 0U;
    }
    if(_S332)
    {
        return;
    }
    uint _S333 = call_first_0(g_17, &kernelContext_40);
    float4 _S334 = float4(*((&kernelContext_40)->calls_0+10U * g_17)) ;
    float _S335 = _S334.x;
    float _S336 = _S334.z;
    uint _S337;
    if(((&kernelContext_40)->params_0->count_only_0) != 0U)
    {
        _S337 = 0U;
    }
    else
    {
        uint _S338 = edge_base_0(g_17, 0U, &kernelContext_40);
        _S337 = _S338;
    }
    uint _S339 = (&kernelContext_40)->params_0->bases2_0.x;
    uint slots_1 = (&kernelContext_40)->params_0->max_bonds_cap_0 + 8U * _S333;
    uint _S340 = (uint4(*((&kernelContext_40)->bedge_0+slots_1)) ).x;
    uint ci_0 = 0U;
    uint count_3 = 0U;
    for(;;)
    {
        if(ci_0 < _S340)
        {
        }
        else
        {
            break;
        }
        uint4 _S341 = uint4(*((&kernelContext_40)->bedge_0+(slots_1 + 1U + ci_0))) ;
        uint br_1 = _S341.x;
        uint na_1 = _S341.y;
        uint nb_0 = _S341.z;
        uint stamp_0 = _S341.w;
        float4 _S342 = node_0(na_1, 1U, &kernelContext_40);
        matrix<float,int(3),int(3)>  ra_1 = quat_matrix_0(_S342);
        float4 _S343 = node_0(nb_0, 1U, &kernelContext_40);
        matrix<float,int(3),int(3)>  rbm_0 = quat_matrix_0(_S343);
        uint _S344 = 24U * na_1;
        uint _S345 = _S344 + 17U;
        matrix<float,int(3),int(3)>  rs_0 = (((transpose(quat_matrix_0(float4(*((&kernelContext_40)->work_0+_S345), *((&kernelContext_40)->work_0+(_S345 + 1U)), *((&kernelContext_40)->work_0+(_S345 + 2U)), *((&kernelContext_40)->work_0+(_S345 + 3U)))))) * (ra_1)));
        uint _S346 = 9U * br_1;
        float4 _S347 = float4(*((&kernelContext_40)->sb_0+_S346)) ;
        uint4 ends_2 = (as_type<uint4>((float4(*((&kernelContext_40)->sb_0+(_S346 + 7U))) )));
        float3 nrm_1 = ((((float4(*((&kernelContext_40)->sb_0+(_S346 + 1U))) ).xyz) * (rs_0)));
        float3 t1_4 = ((((float4(*((&kernelContext_40)->sb_0+(_S346 + 2U))) ).xyz) * (rs_0)));
        float4 _S348 = node_0(na_1, 13U, &kernelContext_40);
        float3 xa_0 = _S348.xyz;
        float4 _S349 = node_0(nb_0, 13U, &kernelContext_40);
        float3 _S350 = _S349.xyz - xa_0;
        float3 nrm_2;
        float3 t1_5;
        if((dot(nrm_1, _S350)) < 0.0f)
        {
            float3 _S351 = - t1_4;
            nrm_2 = - nrm_1;
            t1_5 = _S351;
        }
        else
        {
            nrm_2 = nrm_1;
            t1_5 = t1_4;
        }
        uint _S352 = _S344 + 9U;
        float3 ha_0 = float3(*((&kernelContext_40)->work_0+_S352), *((&kernelContext_40)->work_0+(_S352 + 1U)), *((&kernelContext_40)->work_0+(_S352 + 2U)));
        uint _S353 = 24U * nb_0;
        uint _S354 = _S353 + 9U;
        float3 hb_0 = float3(*((&kernelContext_40)->work_0+_S354), *((&kernelContext_40)->work_0+(_S354 + 1U)), *((&kernelContext_40)->work_0+(_S354 + 2U)));
        float gap_0 = dot(_S350, nrm_2) - support_of_0(ra_1, ha_0, nrm_2) - support_of_0(rbm_0, hb_0, nrm_2);
        float3 t2_2 = cross(nrm_2, t1_5);
        bool across1_0 = (abs(dot(_S350, t1_5))) <= (support_of_0(ra_1, ha_0, t1_5) + support_of_0(rbm_0, hb_0, t1_5));
        bool across2_0 = (abs(dot(_S350, t2_2))) <= (support_of_0(ra_1, ha_0, t2_2) + support_of_0(rbm_0, hb_0, t2_2));
        if(gap_0 > (0.25f * _S335))
        {
            _S332 = true;
        }
        else
        {
            _S332 = !across1_0;
        }
        bool _S355;
        if(_S332)
        {
            _S355 = true;
        }
        else
        {
            _S355 = !across2_0;
        }
        if(_S355)
        {
            ci_0 = ci_0 + 1U;
            continue;
        }
        if(((&kernelContext_40)->params_0->count_only_0) == 0U)
        {
            uint _S356 = _S344 + 21U;
            float3 _S357 = xa_0 + (((_S347.xyz - float3(*((&kernelContext_40)->work_0+_S356), *((&kernelContext_40)->work_0+(_S356 + 1U)), *((&kernelContext_40)->work_0+(_S356 + 2U)))) * (rs_0)));
            float _S358;
            if(stamp_0 == 4294967295U)
            {
                _S358 = 1.0f;
            }
            else
            {
                _S358 = min(float((&kernelContext_40)->params_0->frame_0 - stamp_0) / 8.0f, 1.0f);
            }
            float fa_0 = *((&kernelContext_40)->work_0+(_S344 + 13U));
            float fbr_0 = *((&kernelContext_40)->work_0+(_S353 + 13U));
            float _S359;
            if(((&kernelContext_40)->params_0->contact_friction_0) >= 0.0f)
            {
                _S359 = (&kernelContext_40)->params_0->contact_friction_0;
            }
            else
            {
                _S359 = min(fa_0, fbr_0);
            }
            uint side_2 = 0U;
            for(;;)
            {
                if(side_2 < 2U)
                {
                }
                else
                {
                    break;
                }
                uint row_7 = _S337 + count_3 + side_2;
                bool _S360 = side_2 == 0U;
                uint sn_1;
                if(_S360)
                {
                    sn_1 = na_1;
                }
                else
                {
                    sn_1 = nb_0;
                }
                uint dn_1;
                if(_S360)
                {
                    dn_1 = nb_0;
                }
                else
                {
                    dn_1 = na_1;
                }
                float3 ft1_1;
                if(_S360)
                {
                    ft1_1 = t1_5;
                }
                else
                {
                    ft1_1 = - t1_5;
                }
                float3 fn_1;
                if(_S360)
                {
                    fn_1 = nrm_2;
                }
                else
                {
                    fn_1 = - nrm_2;
                }
                float4 _S361 = node_0(sn_1, 3U, &kernelContext_40);
                float3 _S362 = _S361.xyz;
                float4 _S363 = node_0(sn_1, 4U, &kernelContext_40);
                float3 _S364 = _S363.xyz;
                float4 _S365 = node_0(sn_1, 13U, &kernelContext_40);
                float3 vs_0 = _S362 + cross(_S364, _S357 - _S365.xyz);
                float4 _S366 = node_0(dn_1, 3U, &kernelContext_40);
                float3 _S367 = _S366.xyz;
                float4 _S368 = node_0(dn_1, 4U, &kernelContext_40);
                float3 _S369 = _S368.xyz;
                float4 _S370 = node_0(dn_1, 13U, &kernelContext_40);
                float3 vd_0 = _S367 + cross(_S369, _S357 - _S370.xyz);
                float4 _S371 = node_0(sn_1, 13U, &kernelContext_40);
                float3 q_3 = _S357 - _S371.xyz;
                float3 dv_1 = vd_0 - vs_0;
                uint _S372 = global_node_0(g_17, sn_1 - _S333, &kernelContext_40);
                uint _S373 = global_node_0(g_17, dn_1 - _S333, &kernelContext_40);
                put_edge_0(4U, row_7, _S372, _S373, &kernelContext_40);
                put_0(_S339, 4U, row_7, 0U, ln_ratio_0(_S347.w, _S335 * _S335), &kernelContext_40);
                put_0(_S339, 4U, row_7, 1U, dot(ft1_1, q_3) / _S335, &kernelContext_40);
                put_0(_S339, 4U, row_7, 2U, dot(t2_2, q_3) / _S335, &kernelContext_40);
                put_0(_S339, 4U, row_7, 3U, dot(fn_1, q_3) / _S335, &kernelContext_40);
                put_0(_S339, 4U, row_7, 4U, gap_0 / _S335, &kernelContext_40);
                put_0(_S339, 4U, row_7, 5U, _S359, &kernelContext_40);
                put_0(_S339, 4U, row_7, 6U, dot(ft1_1, dv_1) / _S336, &kernelContext_40);
                put_0(_S339, 4U, row_7, 7U, dot(t2_2, dv_1) / _S336, &kernelContext_40);
                put_0(_S339, 4U, row_7, 8U, dot(fn_1, dv_1) / _S336, &kernelContext_40);
                put_0(_S339, 4U, row_7, 9U, _S358, &kernelContext_40);
                float4 _S374 = node_0(sn_1, 1U, &kernelContext_40);
                put_rot6_0(_S339, 4U, row_7, 10U, ft1_1, t2_2, quat_matrix_0(_S374), &kernelContext_40);
                float4 _S375 = node_0(dn_1, 1U, &kernelContext_40);
                put_rot6_0(_S339, 4U, row_7, 16U, ft1_1, t2_2, quat_matrix_0(_S375), &kernelContext_40);
                *((&kernelContext_40)->work_0+(24U * sn_1 + 4U)) = 1.0f;
                side_2 = side_2 + 1U;
            }
            put_key_0(4U, (_S337 + count_3) / 2U, ends_2.z, ends_2.w, &kernelContext_40);
        }
        count_3 = count_3 + 2U;
        ci_0 = ci_0 + 1U;
    }
    if(((&kernelContext_40)->params_0->count_only_0) != 0U)
    {
        edge_count_0(g_17, 0U, count_3, &kernelContext_40);
    }
    return;
}

[[kernel]] void feat_sc(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_12 [[buffer(0)]], uint device* bmap_12 [[buffer(11)]], uint device* counters_12 [[buffer(28)]], Island_natural_0 device* islands_12 [[buffer(24)]], BondStatic_natural_0 device* bond_static_12 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(13)]], uint device* sinc_12 [[buffer(9)]], packed_uint4 device* slook_12 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_12 [[buffer(15)]], uint device* ctl_12 [[buffer(1)]], float device* work_12 [[buffer(18)]], packed_float4 device* nodes_12 [[buffer(3)]], packed_uint2 device* key_node_12 [[buffer(12)]], packed_float4 device* sc_12 [[buffer(7)]], packed_float4 device* bonds_12 [[buffer(4)]], packed_float4 device* sb_12 [[buffer(8)]], packed_uint4 device* bedge_12 [[buffer(22)]], packed_float4 device* calls_12 [[buffer(2)]], packed_float4 device* imps_12 [[buffer(16)]], packed_uint4 device* cisl_12 [[buffer(5)]], packed_float4 device* cislf_12 [[buffer(6)]], packed_float4 device* contacts_12 [[buffer(17)]], uint device* fctl_12 [[buffer(23)]], uint device* edges_12 [[buffer(20)]], uint device* cols_12 [[buffer(21)]], float device* feat_12 [[buffer(19)]], packed_float4 device* ctab_12 [[buffer(25)]], packed_float4 device* bodies_12 [[buffer(26)]], packed_float4 device* state_12 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->params_0 = params_12;
    (&kernelContext_41)->bmap_0 = bmap_12;
    (&kernelContext_41)->counters_0 = counters_12;
    (&kernelContext_41)->islands_0 = islands_12;
    (&kernelContext_41)->bond_static_0 = bond_static_12;
    (&kernelContext_41)->chunks_0 = chunks_12;
    (&kernelContext_41)->sinc_0 = sinc_12;
    (&kernelContext_41)->slook_0 = slook_12;
    (&kernelContext_41)->bond_dyn_0 = bond_dyn_12;
    (&kernelContext_41)->ctl_0 = ctl_12;
    (&kernelContext_41)->work_0 = work_12;
    (&kernelContext_41)->nodes_0 = nodes_12;
    (&kernelContext_41)->key_node_0 = key_node_12;
    (&kernelContext_41)->sc_0 = sc_12;
    (&kernelContext_41)->bonds_0 = bonds_12;
    (&kernelContext_41)->sb_0 = sb_12;
    (&kernelContext_41)->bedge_0 = bedge_12;
    (&kernelContext_41)->calls_0 = calls_12;
    (&kernelContext_41)->imps_0 = imps_12;
    (&kernelContext_41)->cisl_0 = cisl_12;
    (&kernelContext_41)->cislf_0 = cislf_12;
    (&kernelContext_41)->contacts_0 = contacts_12;
    (&kernelContext_41)->fctl_0 = fctl_12;
    (&kernelContext_41)->edges_0 = edges_12;
    (&kernelContext_41)->cols_0 = cols_12;
    (&kernelContext_41)->feat_0 = feat_12;
    (&kernelContext_41)->ctab_0 = ctab_12;
    (&kernelContext_41)->bodies_0 = bodies_12;
    (&kernelContext_41)->state_0 = state_12;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_12;
    (&kernelContext_41)->g_red_0 = &g_red_12;
    threadgroup array<float, int(64)> g_mv_12;
    (&kernelContext_41)->g_mv_0 = &g_mv_12;
    threadgroup array<uint, int(64)> g_mi_12;
    (&kernelContext_41)->g_mi_0 = &g_mi_12;
    threadgroup array<uint, int(4096)> g_plug_12;
    (&kernelContext_41)->g_plug_0 = &g_plug_12;
    threadgroup uint g_flag_12;
    (&kernelContext_41)->g_flag_0 = &g_flag_12;
    threadgroup array<uint, int(64)> g_scan_12;
    (&kernelContext_41)->g_scan_0 = &g_scan_12;
    threadgroup float4 g_scalars_12;
    (&kernelContext_41)->g_scalars_0 = &g_scalars_12;
    uint g_18 = group_6.x;
    uint t_13 = thread_8.x;
    bool _S376;
    if(g_18 >= ctl_12[int(1)])
    {
        _S376 = true;
    }
    else
    {
        _S376 = t_13 != 0U;
    }
    if(_S376)
    {
        return;
    }
    uint _S377 = call_first_0(g_18, &kernelContext_41);
    uint _S378 = call_e0_0(g_18, &kernelContext_41);
    uint _S379 = call_ne_0(g_18, &kernelContext_41);
    float4 _S380 = float4(*((&kernelContext_41)->calls_0+10U * g_18)) ;
    float _S381 = _S380.x;
    float _S382 = _S380.z;
    float _S383 = _S380.w;
    uint _S384;
    if(((&kernelContext_41)->params_0->count_only_0) != 0U)
    {
        _S384 = 0U;
    }
    else
    {
        uint _S385 = edge_base_0(g_18, 2U, &kernelContext_41);
        _S384 = _S385;
    }
    uint _S386 = (&kernelContext_41)->params_0->bases2_0.z;
    uint ea_0 = _S378;
    uint count_4 = 0U;
    for(;;)
    {
        uint _S387 = _S378 + _S379;
        if(ea_0 < _S387)
        {
        }
        else
        {
            break;
        }
        uint4 _S388 = uint4(*((&kernelContext_41)->cisl_0+2U * ea_0)) ;
        uint ja_0 = 0U;
        for(;;)
        {
            if(ja_0 < (_S388.z))
            {
            }
            else
            {
                break;
            }
            uint na_2 = _S388.y + ja_0;
            uint _S389 = 24U * na_2;
            uint _S390 = (as_type<uint>((*((&kernelContext_41)->work_0+(_S389 + 16U)))));
            float4 _S391 = node_0(na_2, 13U, &kernelContext_41);
            float3 _S392 = _S391.xyz;
            float _S393 = *((&kernelContext_41)->work_0+(_S389 + 12U));
            uint e_2 = _S378;
            uint count_5 = count_4;
            for(;;)
            {
                if(e_2 < _S387)
                {
                }
                else
                {
                    break;
                }
                uint count_6;
                if(e_2 == ea_0)
                {
                    count_6 = count_5;
                    e_2 = e_2 + 1U;
                    count_5 = count_6;
                    continue;
                }
                uint4 _S394 = uint4(*((&kernelContext_41)->cisl_0+2U * e_2)) ;
                uint j_12 = 0U;
                count_6 = count_5;
                for(;;)
                {
                    if(j_12 < (_S394.z))
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint nb_1 = _S394.y + j_12;
                    uint _S395 = 24U * nb_1;
                    uint rb_2 = (as_type<uint>((*((&kernelContext_41)->work_0+(_S395 + 16U)))));
                    if(rb_2 <= _S390)
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    float4 _S396 = node_0(nb_1, 13U, &kernelContext_41);
                    float3 d_1 = _S396.xyz - _S392;
                    float _S397 = length(d_1);
                    if(!(_S397 < (_S393 + *((&kernelContext_41)->work_0+(_S395 + 12U)))))
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    uint _S398 = bond_row_lookup_0(_S390, rb_2, &kernelContext_41);
                    if(_S398 != 4294967295U)
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    if(_S397 < 9.999999960041972e-13f)
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    float3 nrm_3 = normalize(d_1);
                    float4 _S399 = node_0(na_2, 1U, &kernelContext_41);
                    matrix<float,int(3),int(3)>  rma_0 = quat_matrix_0(_S399);
                    float4 _S400 = node_0(nb_1, 1U, &kernelContext_41);
                    uint _S401 = _S389 + 9U;
                    uint _S402 = _S395 + 9U;
                    float gap_1 = _S397 - support_of_0(rma_0, float3(*((&kernelContext_41)->work_0+_S401), *((&kernelContext_41)->work_0+(_S401 + 1U)), *((&kernelContext_41)->work_0+(_S401 + 2U))), nrm_3) - support_of_0(quat_matrix_0(_S400), float3(*((&kernelContext_41)->work_0+_S402), *((&kernelContext_41)->work_0+(_S402 + 1U)), *((&kernelContext_41)->work_0+(_S402 + 2U))), nrm_3);
                    if(gap_1 > (0.25f * _S381))
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    if(((&kernelContext_41)->params_0->count_only_0) == 0U)
                    {
                        float3 t1_6 = tangent_of_0(nrm_3, rma_0);
                        float3 _S403 = cross(nrm_3, t1_6);
                        float fa_1 = *((&kernelContext_41)->work_0+(_S389 + 13U));
                        float fbr_1 = *((&kernelContext_41)->work_0+(_S395 + 13U));
                        float _S404;
                        if(((&kernelContext_41)->params_0->contact_friction_0) >= 0.0f)
                        {
                            _S404 = (&kernelContext_41)->params_0->contact_friction_0;
                        }
                        else
                        {
                            _S404 = min(fa_1, fbr_1);
                        }
                        float4 _S405 = node_0(na_2, 12U, &kernelContext_41);
                        uint _S406 = (as_type<uint>((_S405.z)));
                        float4 _S407 = node_0(nb_1, 12U, &kernelContext_41);
                        uint _S408 = (as_type<uint>((_S407.z)));
                        uint side_3 = 0U;
                        for(;;)
                        {
                            if(side_3 < 2U)
                            {
                            }
                            else
                            {
                                break;
                            }
                            uint row_8 = _S384 + count_6 + side_3;
                            bool _S409 = side_3 == 0U;
                            uint sn_2;
                            if(_S409)
                            {
                                sn_2 = na_2;
                            }
                            else
                            {
                                sn_2 = nb_1;
                            }
                            uint dn_2;
                            if(_S409)
                            {
                                dn_2 = nb_1;
                            }
                            else
                            {
                                dn_2 = na_2;
                            }
                            float3 ft1_2;
                            if(_S409)
                            {
                                ft1_2 = t1_6;
                            }
                            else
                            {
                                ft1_2 = - t1_6;
                            }
                            float3 fn_2;
                            if(_S409)
                            {
                                fn_2 = nrm_3;
                            }
                            else
                            {
                                fn_2 = - nrm_3;
                            }
                            float4 _S410 = node_0(sn_2, 3U, &kernelContext_41);
                            float3 _S411 = _S410.xyz;
                            float4 _S412 = node_0(dn_2, 3U, &kernelContext_41);
                            float3 rel_1 = _S411 - _S412.xyz;
                            float _S413 = dot(ft1_2, rel_1);
                            float _S414 = dot(_S403, rel_1);
                            float _S415 = dot(fn_2, rel_1);
                            float t_con_0;
                            if(gap_1 <= 0.0f)
                            {
                                t_con_0 = -1.0f;
                            }
                            else
                            {
                                if(_S415 > 0.0f)
                                {
                                    _S376 = (gap_1 / _S415) < ((&kernelContext_41)->params_0->dt_0);
                                }
                                else
                                {
                                    _S376 = false;
                                }
                                if(_S376)
                                {
                                    t_con_0 = gap_1 / _S415 / (&kernelContext_41)->params_0->dt_0;
                                }
                                else
                                {
                                    t_con_0 = 1.0f;
                                }
                            }
                            uint _S416 = global_node_0(g_18, sn_2 - _S377, &kernelContext_41);
                            uint _S417 = global_node_0(g_18, dn_2 - _S377, &kernelContext_41);
                            put_edge_0(6U, row_8, _S416, _S417, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 0U, _S415 / _S382, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 1U, _S413 / _S382, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 2U, _S414 / _S382, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 3U, _S415 / _S383, &kernelContext_41);
                            float4 _S418 = node_0(dn_2, 0U, &kernelContext_41);
                            float _S419 = _S418.w;
                            float4 _S420 = node_0(sn_2, 0U, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 4U, ln_ratio_0(_S419, _S420.w), &kernelContext_41);
                            put_0(_S386, 6U, row_8, 5U, t_con_0, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 6U, max(- gap_1, 0.0f) / _S381, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 7U, _S404, &kernelContext_41);
                            put_0(_S386, 6U, row_8, 8U, 0.0f, &kernelContext_41);
                            float _S421;
                            if(_S406 == _S408)
                            {
                                _S421 = 1.0f;
                            }
                            else
                            {
                                _S421 = 0.0f;
                            }
                            put_0(_S386, 6U, row_8, 9U, _S421, &kernelContext_41);
                            float4 _S422 = node_0(sn_2, 1U, &kernelContext_41);
                            put_rot6_0(_S386, 6U, row_8, 10U, ft1_2, _S403, quat_matrix_0(_S422), &kernelContext_41);
                            float4 _S423 = node_0(dn_2, 1U, &kernelContext_41);
                            put_rot6_0(_S386, 6U, row_8, 16U, ft1_2, _S403, quat_matrix_0(_S423), &kernelContext_41);
                            *((&kernelContext_41)->work_0+(24U * sn_2 + 4U)) = 1.0f;
                            side_3 = side_3 + 1U;
                        }
                        put_key_0(6U, (_S384 + count_6) / 2U, _S390, rb_2, &kernelContext_41);
                    }
                    count_6 = count_6 + 2U;
                    j_12 = j_12 + 1U;
                }
                e_2 = e_2 + 1U;
                count_5 = count_6;
            }
            ja_0 = ja_0 + 1U;
            count_4 = count_5;
        }
        ea_0 = ea_0 + 1U;
    }
    if(((&kernelContext_41)->params_0->count_only_0) != 0U)
    {
        edge_count_0(g_18, 2U, count_4, &kernelContext_41);
    }
    return;
}

[[kernel]] void feat_prefix(FeatParams_0 constant* params_13 [[buffer(0)]], uint device* bmap_13 [[buffer(11)]], uint device* counters_13 [[buffer(28)]], Island_natural_0 device* islands_13 [[buffer(24)]], BondStatic_natural_0 device* bond_static_13 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(13)]], uint device* sinc_13 [[buffer(9)]], packed_uint4 device* slook_13 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_13 [[buffer(15)]], uint device* ctl_13 [[buffer(1)]], float device* work_13 [[buffer(18)]], packed_float4 device* nodes_13 [[buffer(3)]], packed_uint2 device* key_node_13 [[buffer(12)]], packed_float4 device* sc_13 [[buffer(7)]], packed_float4 device* bonds_13 [[buffer(4)]], packed_float4 device* sb_13 [[buffer(8)]], packed_uint4 device* bedge_13 [[buffer(22)]], packed_float4 device* calls_13 [[buffer(2)]], packed_float4 device* imps_13 [[buffer(16)]], packed_uint4 device* cisl_13 [[buffer(5)]], packed_float4 device* cislf_13 [[buffer(6)]], packed_float4 device* contacts_13 [[buffer(17)]], uint device* fctl_13 [[buffer(23)]], uint device* edges_13 [[buffer(20)]], uint device* cols_13 [[buffer(21)]], float device* feat_13 [[buffer(19)]], packed_float4 device* ctab_13 [[buffer(25)]], packed_float4 device* bodies_13 [[buffer(26)]], packed_float4 device* state_13 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_42;
    (&kernelContext_42)->params_0 = params_13;
    (&kernelContext_42)->bmap_0 = bmap_13;
    (&kernelContext_42)->counters_0 = counters_13;
    (&kernelContext_42)->islands_0 = islands_13;
    (&kernelContext_42)->bond_static_0 = bond_static_13;
    (&kernelContext_42)->chunks_0 = chunks_13;
    (&kernelContext_42)->sinc_0 = sinc_13;
    (&kernelContext_42)->slook_0 = slook_13;
    (&kernelContext_42)->bond_dyn_0 = bond_dyn_13;
    (&kernelContext_42)->ctl_0 = ctl_13;
    (&kernelContext_42)->work_0 = work_13;
    (&kernelContext_42)->nodes_0 = nodes_13;
    (&kernelContext_42)->key_node_0 = key_node_13;
    (&kernelContext_42)->sc_0 = sc_13;
    (&kernelContext_42)->bonds_0 = bonds_13;
    (&kernelContext_42)->sb_0 = sb_13;
    (&kernelContext_42)->bedge_0 = bedge_13;
    (&kernelContext_42)->calls_0 = calls_13;
    (&kernelContext_42)->imps_0 = imps_13;
    (&kernelContext_42)->cisl_0 = cisl_13;
    (&kernelContext_42)->cislf_0 = cislf_13;
    (&kernelContext_42)->contacts_0 = contacts_13;
    (&kernelContext_42)->fctl_0 = fctl_13;
    (&kernelContext_42)->edges_0 = edges_13;
    (&kernelContext_42)->cols_0 = cols_13;
    (&kernelContext_42)->feat_0 = feat_13;
    (&kernelContext_42)->ctab_0 = ctab_13;
    (&kernelContext_42)->bodies_0 = bodies_13;
    (&kernelContext_42)->state_0 = state_13;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_13;
    (&kernelContext_42)->g_red_0 = &g_red_13;
    threadgroup array<float, int(64)> g_mv_13;
    (&kernelContext_42)->g_mv_0 = &g_mv_13;
    threadgroup array<uint, int(64)> g_mi_13;
    (&kernelContext_42)->g_mi_0 = &g_mi_13;
    threadgroup array<uint, int(4096)> g_plug_13;
    (&kernelContext_42)->g_plug_0 = &g_plug_13;
    threadgroup uint g_flag_13;
    (&kernelContext_42)->g_flag_0 = &g_flag_13;
    threadgroup array<uint, int(64)> g_scan_13;
    (&kernelContext_42)->g_scan_0 = &g_scan_13;
    threadgroup float4 g_scalars_13;
    (&kernelContext_42)->g_scalars_0 = &g_scalars_13;
    uint _S424 = ctl_13[int(1)];
    uint g_19 = 0U;
    uint tot0_0 = 0U;
    uint tot1_0 = 0U;
    uint tot2_0 = 0U;
    for(;;)
    {
        if(g_19 < _S424)
        {
        }
        else
        {
            break;
        }
        uint _S425 = 8U + 6U * g_19;
        *((&kernelContext_42)->fctl_0+(_S425 + 3U)) = tot0_0;
        *((&kernelContext_42)->fctl_0+(_S425 + 4U)) = tot1_0;
        *((&kernelContext_42)->fctl_0+(_S425 + 5U)) = tot2_0;
        uint tot0_1 = tot0_0 + *((&kernelContext_42)->fctl_0+_S425);
        uint tot1_1 = tot1_0 + *((&kernelContext_42)->fctl_0+(_S425 + 1U));
        uint tot2_1 = tot2_0 + *((&kernelContext_42)->fctl_0+(_S425 + 2U));
        g_19 = g_19 + 1U;
        tot0_0 = tot0_1;
        tot1_0 = tot1_1;
        tot2_0 = tot2_1;
    }
    uint over_0;
    if(tot0_0 > (2U * (&kernelContext_42)->params_0->caps_0.x))
    {
        over_0 = 1U;
    }
    else
    {
        over_0 = 0U;
    }
    if(tot1_0 > (2U * (&kernelContext_42)->params_0->caps_0.y))
    {
        over_0 = over_0 | 2U;
    }
    if(tot2_0 > (2U * (&kernelContext_42)->params_0->caps_0.z))
    {
        over_0 = over_0 | 4U;
    }
    g_19 = 0U;
    uint cand_over_0 = 0U;
    for(;;)
    {
        if(g_19 < _S424)
        {
        }
        else
        {
            break;
        }
        uint _S426 = (&kernelContext_42)->params_0->max_bonds_cap_0;
        uint _S427 = call_first_0(g_19, &kernelContext_42);
        uint cand_over_1 = cand_over_0 | ((uint4(*((&kernelContext_42)->bedge_0+(_S426 + 8U * _S427))) ).y);
        g_19 = g_19 + 1U;
        cand_over_0 = cand_over_1;
    }
    *((&kernelContext_42)->fctl_0+int(5)) = cand_over_0;
    *((&kernelContext_42)->fctl_0+int(0)) = tot0_0;
    *((&kernelContext_42)->fctl_0+int(1)) = tot1_0;
    *((&kernelContext_42)->fctl_0+int(2)) = tot2_0;
    *((&kernelContext_42)->fctl_0+int(3)) = over_0;
    g_19 = 0U;
    uint chunks_n_0 = 0U;
    for(;;)
    {
        if(g_19 < _S424)
        {
        }
        else
        {
            break;
        }
        uint _S428 = call_chunks_0(g_19, &kernelContext_42);
        uint chunks_n_1 = chunks_n_0 + _S428;
        g_19 = g_19 + 1U;
        chunks_n_0 = chunks_n_1;
    }
    *((&kernelContext_42)->fctl_0+int(4)) = chunks_n_0;
    return;
}

uint call_nodes_0(uint g_20, KernelContext_0 thread* kernelContext_43)
{
    return (as_type<uint>(((float4(*(kernelContext_43->calls_0+(10U * g_20 + 2U))) ).w)));
}

float cap4_0(float x_1)
{
    float _S429;
    if(x_1 < 3.00000000549775576e+38f)
    {
        _S429 = min(x_1, 4.0f);
    }
    else
    {
        _S429 = 4.0f;
    }
    return _S429;
}

[[kernel]] void feat_nodes(uint3 group_7 [[threadgroup_position_in_grid]], uint3 thread_9 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_14 [[buffer(0)]], uint device* bmap_14 [[buffer(11)]], uint device* counters_14 [[buffer(28)]], Island_natural_0 device* islands_14 [[buffer(24)]], BondStatic_natural_0 device* bond_static_14 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_14 [[buffer(13)]], uint device* sinc_14 [[buffer(9)]], packed_uint4 device* slook_14 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_14 [[buffer(15)]], uint device* ctl_14 [[buffer(1)]], float device* work_14 [[buffer(18)]], packed_float4 device* nodes_14 [[buffer(3)]], packed_uint2 device* key_node_14 [[buffer(12)]], packed_float4 device* sc_14 [[buffer(7)]], packed_float4 device* bonds_14 [[buffer(4)]], packed_float4 device* sb_14 [[buffer(8)]], packed_uint4 device* bedge_14 [[buffer(22)]], packed_float4 device* calls_14 [[buffer(2)]], packed_float4 device* imps_14 [[buffer(16)]], packed_uint4 device* cisl_14 [[buffer(5)]], packed_float4 device* cislf_14 [[buffer(6)]], packed_float4 device* contacts_14 [[buffer(17)]], uint device* fctl_14 [[buffer(23)]], uint device* edges_14 [[buffer(20)]], uint device* cols_14 [[buffer(21)]], float device* feat_14 [[buffer(19)]], packed_float4 device* ctab_14 [[buffer(25)]], packed_float4 device* bodies_14 [[buffer(26)]], packed_float4 device* state_14 [[buffer(27)]])
{
    float ke_imp_0;
    float _S430;
    bool _S431;
    thread KernelContext_0 kernelContext_44;
    (&kernelContext_44)->params_0 = params_14;
    (&kernelContext_44)->bmap_0 = bmap_14;
    (&kernelContext_44)->counters_0 = counters_14;
    (&kernelContext_44)->islands_0 = islands_14;
    (&kernelContext_44)->bond_static_0 = bond_static_14;
    (&kernelContext_44)->chunks_0 = chunks_14;
    (&kernelContext_44)->sinc_0 = sinc_14;
    (&kernelContext_44)->slook_0 = slook_14;
    (&kernelContext_44)->bond_dyn_0 = bond_dyn_14;
    (&kernelContext_44)->ctl_0 = ctl_14;
    (&kernelContext_44)->work_0 = work_14;
    (&kernelContext_44)->nodes_0 = nodes_14;
    (&kernelContext_44)->key_node_0 = key_node_14;
    (&kernelContext_44)->sc_0 = sc_14;
    (&kernelContext_44)->bonds_0 = bonds_14;
    (&kernelContext_44)->sb_0 = sb_14;
    (&kernelContext_44)->bedge_0 = bedge_14;
    (&kernelContext_44)->calls_0 = calls_14;
    (&kernelContext_44)->imps_0 = imps_14;
    (&kernelContext_44)->cisl_0 = cisl_14;
    (&kernelContext_44)->cislf_0 = cislf_14;
    (&kernelContext_44)->contacts_0 = contacts_14;
    (&kernelContext_44)->fctl_0 = fctl_14;
    (&kernelContext_44)->edges_0 = edges_14;
    (&kernelContext_44)->cols_0 = cols_14;
    (&kernelContext_44)->feat_0 = feat_14;
    (&kernelContext_44)->ctab_0 = ctab_14;
    (&kernelContext_44)->bodies_0 = bodies_14;
    (&kernelContext_44)->state_0 = state_14;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_14;
    (&kernelContext_44)->g_red_0 = &g_red_14;
    threadgroup array<float, int(64)> g_mv_14;
    (&kernelContext_44)->g_mv_0 = &g_mv_14;
    threadgroup array<uint, int(64)> g_mi_14;
    (&kernelContext_44)->g_mi_0 = &g_mi_14;
    threadgroup array<uint, int(4096)> g_plug_14;
    (&kernelContext_44)->g_plug_0 = &g_plug_14;
    threadgroup uint g_flag_14;
    (&kernelContext_44)->g_flag_0 = &g_flag_14;
    threadgroup array<uint, int(64)> g_scan_14;
    (&kernelContext_44)->g_scan_0 = &g_scan_14;
    threadgroup float4 g_scalars_14;
    (&kernelContext_44)->g_scalars_0 = &g_scalars_14;
    uint g_21 = group_7.x;
    uint t_14 = thread_9.x;
    uint j_13;
    if(g_21 >= ctl_14[int(1)])
    {
        j_13 = 1U;
    }
    else
    {
        j_13 = 0U;
    }
    if((group_uniform_0(j_13, t_14)) != 0U)
    {
        return;
    }
    uint _S432 = call_first_0(g_21, &kernelContext_44);
    uint _S433 = call_nodes_0(g_21, &kernelContext_44);
    uint n_9 = group_uniform_0(_S433, t_14);
    uint _S434 = call_chunks_0(g_21, &kernelContext_44);
    uint nc_1 = group_uniform_0(_S434, t_14);
    uint _S435 = 10U * g_21;
    float4 _S436 = float4(*((&kernelContext_44)->calls_0+_S435)) ;
    float L_1 = _S436.x;
    float _S437 = _S436.y;
    float vref_0 = _S436.z;
    float c_wave_1 = _S436.w;
    float _S438 = (float4(*((&kernelContext_44)->calls_0+(_S435 + 1U))) ).y;
    float dt_2 = (&kernelContext_44)->params_0->dt_0;
    float3 grav_0 = (&kernelContext_44)->params_0->gravity_0.xyz;
    float4 _S439 = float4(0.0f) ;
    j_13 = t_14;
    float4 a_10 = _S439;
    float4 b_6 = _S439;
    for(;;)
    {
        if(j_13 < n_9)
        {
        }
        else
        {
            break;
        }
        uint _S440 = _S432 + j_13;
        float4 _S441 = node_0(_S440, 0U, &kernelContext_44);
        float m_4 = _S441.w;
        float4 _S442 = node_0(_S440, 3U, &kernelContext_44);
        float4 a_11 = a_10 + float4(_S442.xyz * float3(m_4) , m_4);
        bool _S443 = j_13 < nc_1;
        if(_S443)
        {
            ke_imp_0 = m_4;
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        if(_S443)
        {
            float4 _S444 = node_0(_S440, 2U, &kernelContext_44);
            _S431 = (_S444.w) != 0.0f;
        }
        else
        {
            _S431 = false;
        }
        if(_S431)
        {
            _S430 = 1.0f;
        }
        else
        {
            _S430 = 0.0f;
        }
        float4 b_7 = b_6 + float4(ke_imp_0, _S430, 0.0f, 0.0f);
        j_13 = j_13 + 64U;
        a_10 = a_11;
        b_6 = b_7;
    }
    float4 _S445 = group_sum_0(t_14, a_10, &kernelContext_44);
    float4 _S446 = group_sum_0(t_14, b_6, &kernelContext_44);
    float3 _S447 = _S445.xyz / float3(_S445.w) ;
    float _S448 = _S446.x;
    bool anchored_0 = (_S446.y) > 0.0f;
    uint _S449 = _S435 + 9U;
    float _S450 = (float4(*((&kernelContext_44)->calls_0+_S449)) ).y;
    float _S451 = (float4(*((&kernelContext_44)->calls_0+_S449)) ).z;
    uint _S452 = (&kernelContext_44)->params_0->bases_0.x;
    uint _S453 = (&kernelContext_44)->params_0->bases_0.y;
    uint fbg_0 = (&kernelContext_44)->params_0->bases_0.z;
    uint _S454 = (&kernelContext_44)->params_0->bases2_0.w;
    j_13 = t_14;
    ke_imp_0 = 0.0f;
    for(;;)
    {
        if(j_13 < n_9)
        {
        }
        else
        {
            break;
        }
        uint k_10 = _S432 + j_13;
        float4 _S455 = node_0(k_10, 1U, &kernelContext_44);
        matrix<float,int(3),int(3)>  rb_3 = quat_matrix_0(_S455);
        matrix<float,int(3),int(3)>  bt_0 = transpose(rb_3);
        float4 _S456 = node_0(k_10, 0U, &kernelContext_44);
        float m_5 = _S456.w;
        float3 gb_1 = (((grav_0) * (bt_0))) / float3(9.81000041961669922f) ;
        float4 _S457 = node_0(k_10, 1U, &kernelContext_44);
        uint _S458 = global_node_0(g_21, j_13, &kernelContext_44);
        uint _S459 = _S454 + 7U * _S458;
        float device* _S460 = (&kernelContext_44)->feat_0+_S459;
        float4 _S461 = node_0(k_10, 0U, &kernelContext_44);
        *_S460 = _S461.x;
        float device* _S462 = (&kernelContext_44)->feat_0+(_S459 + 1U);
        float4 _S463 = node_0(k_10, 0U, &kernelContext_44);
        *_S462 = _S463.y;
        float device* _S464 = (&kernelContext_44)->feat_0+(_S459 + 2U);
        float4 _S465 = node_0(k_10, 0U, &kernelContext_44);
        *_S464 = _S465.z;
        *((&kernelContext_44)->feat_0+(_S459 + 3U)) = _S457.w;
        *((&kernelContext_44)->feat_0+(_S459 + 4U)) = _S457.x;
        *((&kernelContext_44)->feat_0+(_S459 + 5U)) = _S457.y;
        *((&kernelContext_44)->feat_0+(_S459 + 6U)) = _S457.z;
        uint c_2;
        if(j_13 < nc_1)
        {
            uint _S466 = call_cbase_0(g_21, &kernelContext_44);
            uint row_9 = _S466 + j_13;
            float4 _S467 = node_0(k_10, 2U, &kernelContext_44);
            float3 inr_0 = _S467.xyz;
            uint _S468 = 24U * k_10;
            uint _S469 = _S468 + 9U;
            float device* _S470 = (&kernelContext_44)->work_0+_S469;
            float device* _S471 = (&kernelContext_44)->work_0+(_S469 + 1U);
            float device* _S472 = (&kernelContext_44)->work_0+(_S469 + 2U);
            float device* _S473 = (&kernelContext_44)->work_0+(_S468 + 14U);
            float fixed_0 = *_S473;
            float nlive_3 = *((&kernelContext_44)->work_0+(_S468 + 7U));
            float4 _S474 = node_0(k_10, 3U, &kernelContext_44);
            float3 vr_0 = (((_S474.xyz - _S447) * (bt_0))) / float3(vref_0) ;
            float4 _S475 = node_0(k_10, 4U, &kernelContext_44);
            float3 wb_0 = (((_S475.xyz) * (bt_0))) * float3((L_1 / vref_0)) ;
            put_0(_S452, 0U, row_9, 0U, ln_ratio_0(m_5, _S437), &kernelContext_44);
            float _S476 = m_5 * L_1 * L_1;
            put_0(_S452, 0U, row_9, 1U, inr_0.x / _S476, &kernelContext_44);
            put_0(_S452, 0U, row_9, 2U, inr_0.y / _S476, &kernelContext_44);
            put_0(_S452, 0U, row_9, 3U, inr_0.z / _S476, &kernelContext_44);
            put_0(_S452, 0U, row_9, 4U, *_S470 / L_1, &kernelContext_44);
            put_0(_S452, 0U, row_9, 5U, *_S471 / L_1, &kernelContext_44);
            put_0(_S452, 0U, row_9, 6U, *_S472 / L_1, &kernelContext_44);
            if((*_S473) == 1.0f)
            {
                _S430 = 1.0f;
            }
            else
            {
                _S430 = 0.0f;
            }
            put_0(_S452, 0U, row_9, 7U, _S430, &kernelContext_44);
            float _S477;
            if(fixed_0 == 2.0f)
            {
                _S477 = 1.0f;
            }
            else
            {
                _S477 = 0.0f;
            }
            put_0(_S452, 0U, row_9, 8U, _S477, &kernelContext_44);
            float _S478;
            if(nlive_3 > 0.0f)
            {
                _S478 = *((&kernelContext_44)->work_0+(_S468 + 5U)) / nlive_3;
            }
            else
            {
                _S478 = 0.0f;
            }
            put_0(_S452, 0U, row_9, 9U, _S478, &kernelContext_44);
            put_0(_S452, 0U, row_9, 10U, *((&kernelContext_44)->work_0+(_S468 + 6U)), &kernelContext_44);
            put_0(_S452, 0U, row_9, 11U, nlive_3 / 6.0f, &kernelContext_44);
            put_0(_S452, 0U, row_9, 12U, *((&kernelContext_44)->work_0+(_S468 + 8U)) / 6.0f, &kernelContext_44);
            put_0(_S452, 0U, row_9, 13U, cap4_0(*((&kernelContext_44)->work_0+_S468) / 16.0f), &kernelContext_44);
            put_0(_S452, 0U, row_9, 14U, cap4_0(*((&kernelContext_44)->work_0+(_S468 + 1U)) / dt_2), &kernelContext_44);
            put_0(_S452, 0U, row_9, 15U, cap4_0(*((&kernelContext_44)->work_0+(_S468 + 2U)) / 16.0f), &kernelContext_44);
            put_0(_S452, 0U, row_9, 16U, gb_1.x, &kernelContext_44);
            put_0(_S452, 0U, row_9, 17U, gb_1.y, &kernelContext_44);
            put_0(_S452, 0U, row_9, 18U, gb_1.z, &kernelContext_44);
            put_0(_S452, 0U, row_9, 19U, vr_0.x, &kernelContext_44);
            put_0(_S452, 0U, row_9, 20U, vr_0.y, &kernelContext_44);
            put_0(_S452, 0U, row_9, 21U, vr_0.z, &kernelContext_44);
            put_0(_S452, 0U, row_9, 22U, wb_0.x, &kernelContext_44);
            put_0(_S452, 0U, row_9, 23U, wb_0.y, &kernelContext_44);
            put_0(_S452, 0U, row_9, 24U, wb_0.z, &kernelContext_44);
            float _S479;
            if((*((&kernelContext_44)->work_0+(_S468 + 3U))) != 0.0f)
            {
                _S479 = 1.0f;
            }
            else
            {
                _S479 = 0.0f;
            }
            put_0(_S452, 0U, row_9, 25U, _S479, &kernelContext_44);
            c_2 = 26U;
            for(;;)
            {
                if(c_2 < 32U)
                {
                }
                else
                {
                    break;
                }
                put_0(_S452, 0U, row_9, c_2, 0.0f, &kernelContext_44);
                c_2 = c_2 + 1U;
            }
            put_0(_S452, 0U, row_9, 32U, *((&kernelContext_44)->work_0+(_S468 + 4U)), &kernelContext_44);
            uint c_3 = 33U;
            for(;;)
            {
                if(c_3 < 39U)
                {
                }
                else
                {
                    break;
                }
                put_0(_S452, 0U, row_9, c_3, 0.0f, &kernelContext_44);
                c_3 = c_3 + 1U;
            }
        }
        else
        {
            float4 _S480 = node_0(k_10, 12U, &kernelContext_44);
            uint ki_0 = (as_type<uint>((_S480.x)));
            uint _S481 = call_ibase_0(g_21, &kernelContext_44);
            uint row_10 = _S481 + (j_13 - nc_1);
            uint _S482 = 4U * ki_0 + 3U;
            float r_face_0 = (float4(*((&kernelContext_44)->imps_0+_S482)) ).z;
            float4 _S483 = node_0(k_10, 2U, &kernelContext_44);
            float3 inr_1 = _S483.xyz;
            float r2_0 = r_face_0 * r_face_0;
            float4 _S484 = node_0(k_10, 3U, &kernelContext_44);
            float3 vr_1 = (((_S484.xyz - _S447) * (bt_0)));
            float4 _S485 = node_0(k_10, 3U, &kernelContext_44);
            float3 v_6 = _S485.xyz;
            float4 _S486 = node_0(k_10, 4U, &kernelContext_44);
            float3 w_2 = _S486.xyz;
            float _S487 = inr_1.x;
            float _S488 = inr_1.y;
            float _S489 = inr_1.z;
            float ke_imp_1 = ke_imp_0 + (0.5f * m_5 * dot(v_6, v_6) + 0.5f * dot(w_2, (((w_2) * ((((bt_0) * ((((matrix<float,int(3),int(3)> (float3(_S487, 0.0f, 0.0f), float3(0.0f, _S488, 0.0f), float3(0.0f, 0.0f, _S489))) * (rb_3)))))))))));
            put_0(_S453, 1U, row_10, 0U, ln_ratio_0(m_5, _S448), &kernelContext_44);
            put_0(_S453, 1U, row_10, 1U, ln_ratio_0(m_5, _S437), &kernelContext_44);
            float _S490 = m_5 * r2_0;
            put_0(_S453, 1U, row_10, 2U, _S487 / _S490, &kernelContext_44);
            put_0(_S453, 1U, row_10, 3U, _S488 / _S490, &kernelContext_44);
            put_0(_S453, 1U, row_10, 4U, _S489 / _S490, &kernelContext_44);
            put_0(_S453, 1U, row_10, 5U, ln_ratio_0(r_face_0, L_1), &kernelContext_44);
            put_0(_S453, 1U, row_10, 6U, ln_ratio_0((float4(*((&kernelContext_44)->imps_0+_S482)) ).x, _S438), &kernelContext_44);
            put_0(_S453, 1U, row_10, 7U, (float4(*((&kernelContext_44)->imps_0+_S482)) ).w, &kernelContext_44);
            put_0(_S453, 1U, row_10, 8U, gb_1.x, &kernelContext_44);
            put_0(_S453, 1U, row_10, 9U, gb_1.y, &kernelContext_44);
            put_0(_S453, 1U, row_10, 10U, gb_1.z, &kernelContext_44);
            put_0(_S453, 1U, row_10, 11U, vr_1.x / vref_0, &kernelContext_44);
            put_0(_S453, 1U, row_10, 12U, vr_1.y / vref_0, &kernelContext_44);
            put_0(_S453, 1U, row_10, 13U, vr_1.z / vref_0, &kernelContext_44);
            float4 _S491 = node_0(k_10, 3U, &kernelContext_44);
            put_0(_S453, 1U, row_10, 14U, length(_S491.xyz - _S447) / c_wave_1, &kernelContext_44);
            put_0(_S453, 1U, row_10, 15U, *((&kernelContext_44)->work_0+(24U * k_10 + 4U)), &kernelContext_44);
            c_2 = 16U;
            for(;;)
            {
                if(c_2 < 22U)
                {
                }
                else
                {
                    break;
                }
                put_0(_S453, 1U, row_10, c_2, 0.0f, &kernelContext_44);
                c_2 = c_2 + 1U;
            }
            ke_imp_0 = ke_imp_1;
        }
        j_13 = j_13 + 64U;
    }
    float4 _S492 = group_sum_0(t_14, float4(ke_imp_0, 0.0f, 0.0f, 0.0f), &kernelContext_44);
    float ke_imp_2 = _S492.x;
    if(t_14 == 0U)
    {
        uint ni_0 = n_9 - nc_1;
        bool _S493 = ke_imp_2 > 0.0f;
        if(_S493)
        {
            _S431 = _S450 > 0.0f;
        }
        else
        {
            _S431 = false;
        }
        if(_S431)
        {
            ke_imp_0 = ln_ratio_0(ke_imp_2, _S450);
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, 2U, g_21, 0U, ke_imp_0, &kernelContext_44);
        if(_S493)
        {
            _S431 = _S451 > 0.0f;
        }
        else
        {
            _S431 = false;
        }
        if(_S431)
        {
            ke_imp_0 = ln_ratio_0(ke_imp_2, _S451);
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, 2U, g_21, 1U, ke_imp_0, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 2U, log(dt_2 * c_wave_1 / L_1), &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 3U, dt_2 * vref_0 / L_1, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 4U, float(ni_0) / 4.0f, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 5U, log(float(nc_1)) / 10.0f, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 6U, length(grav_0) / 9.81000041961669922f, &kernelContext_44);
        if(anchored_0)
        {
            ke_imp_0 = 1.0f;
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, 2U, g_21, 7U, ke_imp_0, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 8U, 0.0f, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 9U, 0.0f, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 10U, 0.0f, &kernelContext_44);
        put_0(fbg_0, 2U, g_21, 11U, vref_0 / c_wave_1, &kernelContext_44);
    }
    return;
}

float fail_margin_of_0(const JointState_0 thread* js_1, float4 s8_0, bool rebar_0, bool live_0)
{
    bool ductile_0 = (s8_0.x) != 0.0f;
    bool _S494 = !((s8_0.y) != 0.0f);
    float kf_ts_0;
    if(_S494)
    {
        kf_ts_0 = 1.0f;
    }
    else
    {
        if(ductile_0)
        {
            kf_ts_0 = 0.5f * (js_1->ductility_0 + 1.0f);
        }
        else
        {
            kf_ts_0 = max(js_1->ductility_0, 1.0f);
        }
    }
    float kf_c_0;
    if(_S494)
    {
        kf_c_0 = 1.0f;
    }
    else
    {
        if(ductile_0)
        {
            kf_c_0 = 0.5f * (js_1->ductility_c_0 + 1.0f);
        }
        else
        {
            kf_c_0 = max(js_1->ductility_c_0, 1.0f);
        }
    }
    float _S495 = max(js_1->kappa_0 / kf_ts_0, js_1->kappa_c_0 / kf_c_0);
    bool _S496;
    if(rebar_0)
    {
        _S496 = (s8_0.z) > 0.0f;
    }
    else
    {
        _S496 = false;
    }
    float q_4;
    if(_S496)
    {
        q_4 = min(_S495, js_1->rebar_work_0 / s8_0.z);
    }
    else
    {
        q_4 = _S495;
    }
    if(live_0)
    {
        _S496 = q_4 >= 1.0f;
    }
    else
    {
        _S496 = false;
    }
    if(_S496)
    {
        q_4 = 0.99999994039535522f;
    }
    return q_4;
}

[[kernel]] void feat_bonds(uint3 id_4 [[thread_position_in_grid]], FeatParams_0 constant* params_15 [[buffer(0)]], uint device* bmap_15 [[buffer(11)]], uint device* counters_15 [[buffer(28)]], Island_natural_0 device* islands_15 [[buffer(24)]], BondStatic_natural_0 device* bond_static_15 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_15 [[buffer(13)]], uint device* sinc_15 [[buffer(9)]], packed_uint4 device* slook_15 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_15 [[buffer(15)]], uint device* ctl_15 [[buffer(1)]], float device* work_15 [[buffer(18)]], packed_float4 device* nodes_15 [[buffer(3)]], packed_uint2 device* key_node_15 [[buffer(12)]], packed_float4 device* sc_15 [[buffer(7)]], packed_float4 device* bonds_15 [[buffer(4)]], packed_float4 device* sb_15 [[buffer(8)]], packed_uint4 device* bedge_15 [[buffer(22)]], packed_float4 device* calls_15 [[buffer(2)]], packed_float4 device* imps_15 [[buffer(16)]], packed_uint4 device* cisl_15 [[buffer(5)]], packed_float4 device* cislf_15 [[buffer(6)]], packed_float4 device* contacts_15 [[buffer(17)]], uint device* fctl_15 [[buffer(23)]], uint device* edges_15 [[buffer(20)]], uint device* cols_15 [[buffer(21)]], float device* feat_15 [[buffer(19)]], packed_float4 device* ctab_15 [[buffer(25)]], packed_float4 device* bodies_15 [[buffer(26)]], packed_float4 device* state_15 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_45;
    (&kernelContext_45)->params_0 = params_15;
    (&kernelContext_45)->bmap_0 = bmap_15;
    (&kernelContext_45)->counters_0 = counters_15;
    (&kernelContext_45)->islands_0 = islands_15;
    (&kernelContext_45)->bond_static_0 = bond_static_15;
    (&kernelContext_45)->chunks_0 = chunks_15;
    (&kernelContext_45)->sinc_0 = sinc_15;
    (&kernelContext_45)->slook_0 = slook_15;
    (&kernelContext_45)->bond_dyn_0 = bond_dyn_15;
    (&kernelContext_45)->ctl_0 = ctl_15;
    (&kernelContext_45)->work_0 = work_15;
    (&kernelContext_45)->nodes_0 = nodes_15;
    (&kernelContext_45)->key_node_0 = key_node_15;
    (&kernelContext_45)->sc_0 = sc_15;
    (&kernelContext_45)->bonds_0 = bonds_15;
    (&kernelContext_45)->sb_0 = sb_15;
    (&kernelContext_45)->bedge_0 = bedge_15;
    (&kernelContext_45)->calls_0 = calls_15;
    (&kernelContext_45)->imps_0 = imps_15;
    (&kernelContext_45)->cisl_0 = cisl_15;
    (&kernelContext_45)->cislf_0 = cislf_15;
    (&kernelContext_45)->contacts_0 = contacts_15;
    (&kernelContext_45)->fctl_0 = fctl_15;
    (&kernelContext_45)->edges_0 = edges_15;
    (&kernelContext_45)->cols_0 = cols_15;
    (&kernelContext_45)->feat_0 = feat_15;
    (&kernelContext_45)->ctab_0 = ctab_15;
    (&kernelContext_45)->bodies_0 = bodies_15;
    (&kernelContext_45)->state_0 = state_15;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_15;
    (&kernelContext_45)->g_red_0 = &g_red_15;
    threadgroup array<float, int(64)> g_mv_15;
    (&kernelContext_45)->g_mv_0 = &g_mv_15;
    threadgroup array<uint, int(64)> g_mi_15;
    (&kernelContext_45)->g_mi_0 = &g_mi_15;
    threadgroup array<uint, int(4096)> g_plug_15;
    (&kernelContext_45)->g_plug_0 = &g_plug_15;
    threadgroup uint g_flag_15;
    (&kernelContext_45)->g_flag_0 = &g_flag_15;
    threadgroup array<uint, int(64)> g_scan_15;
    (&kernelContext_45)->g_scan_0 = &g_scan_15;
    threadgroup float4 g_scalars_15;
    (&kernelContext_45)->g_scalars_0 = &g_scalars_15;
    uint b_8 = id_4.x;
    if(b_8 >= ctl_15[int(3)])
    {
        return;
    }
    uint4 _S497 = uint4(*((&kernelContext_45)->bedge_0+b_8)) ;
    uint _S498 = _S497.x;
    bool _S499;
    if(_S498 == 4294967295U)
    {
        _S499 = true;
    }
    else
    {
        _S499 = (_S497.y) == 4294967295U;
    }
    if(_S499)
    {
        return;
    }
    float4 _S500 = node_0(_S498, 3U, &kernelContext_45);
    uint _S501 = 10U * (as_type<uint>((_S500.w)));
    float4 _S502 = float4(*((&kernelContext_45)->calls_0+_S501)) ;
    float4 _S503 = float4(*((&kernelContext_45)->calls_0+(_S501 + 1U))) ;
    float L_2 = _S502.x;
    float _S504 = _S502.z;
    float _S505 = _S503.x;
    float e_med_1 = _S503.y;
    float _S506 = _S503.z;
    uint _S507 = 9U * _S497.z;
    float4 _S508 = float4(*((&kernelContext_45)->sb_0+_S507)) ;
    float4 _S509 = float4(*((&kernelContext_45)->sb_0+(_S507 + 1U))) ;
    float4 _S510 = float4(*((&kernelContext_45)->sb_0+(_S507 + 2U))) ;
    float4 _S511 = float4(*((&kernelContext_45)->sb_0+(_S507 + 3U))) ;
    float4 _S512 = float4(*((&kernelContext_45)->sb_0+(_S507 + 4U))) ;
    float4 _S513 = float4(*((&kernelContext_45)->sb_0+(_S507 + 5U))) ;
    float4 _S514 = float4(*((&kernelContext_45)->sb_0+(_S507 + 6U))) ;
    float4 _S515 = float4(*((&kernelContext_45)->sb_0+(_S507 + 8U))) ;
    uint nb_2 = _S497.y;
    float4 _S516 = node_0(_S498, 1U, &kernelContext_45);
    uint _S517 = 24U * _S498;
    uint _S518 = _S517 + 17U;
    matrix<float,int(3),int(3)>  rs_1 = (((transpose(quat_matrix_0(float4(*((&kernelContext_45)->work_0+_S518), *((&kernelContext_45)->work_0+(_S518 + 1U)), *((&kernelContext_45)->work_0+(_S518 + 2U)), *((&kernelContext_45)->work_0+(_S518 + 3U)))))) * (quat_matrix_0(_S516))));
    float4 _S519 = node_0(_S498, 13U, &kernelContext_45);
    float3 xa_1 = _S519.xyz;
    float4 _S520 = node_0(nb_2, 13U, &kernelContext_45);
    float3 nrm_4 = (((_S509.xyz) * (rs_1)));
    float3 t1_7 = (((_S510.xyz) * (rs_1)));
    float3 nrm_5;
    float3 t1_8;
    if((dot(nrm_4, _S520.xyz - xa_1)) < 0.0f)
    {
        float3 _S521 = - t1_7;
        nrm_5 = - nrm_4;
        t1_8 = _S521;
    }
    else
    {
        nrm_5 = nrm_4;
        t1_8 = t1_7;
    }
    float3 _S522 = cross(nrm_5, t1_8);
    uint _S523 = _S517 + 21U;
    float3 _S524 = xa_1 + (((_S508.xyz - float3(*((&kernelContext_45)->work_0+_S523), *((&kernelContext_45)->work_0+(_S523 + 1U)), *((&kernelContext_45)->work_0+(_S523 + 2U)))) * (rs_1)));
    float area_1 = _S508.w;
    float _S525 = e_med_1 * area_1;
    float _S526 = _S525 / L_2;
    float _S527 = _S525 * L_2;
    bool rebar_1 = (_S512.w) != 0.0f;
    JointState_0 _S528 = ((&kernelContext_45)->bond_dyn_0+_S497.w)->js_0;
    thread JointState_0 _S529 = ((&kernelContext_45)->bond_dyn_0+_S497.w)->js_0;
    float _S530 = fail_margin_of_0(&_S529, _S515, rebar_1, true);
    uint pa_0 = uint(*((&kernelContext_45)->work_0+(_S517 + 3U)));
    uint pb_0 = uint(*((&kernelContext_45)->work_0+(24U * nb_2 + 3U)));
    bool in_plug_0 = (pa_0 != 0U) != (pb_0 != 0U);
    float ratio_1;
    if(in_plug_0)
    {
        ratio_1 = ln_ratio_0((float4(*((&kernelContext_45)->contacts_0+(6U * ((as_type<uint>(((float4(*((&kernelContext_45)->calls_0+(_S501 + 8U))) ).x))) + max(pa_0, pb_0) - 1U) + 4U))) ).x, 1.0f);
    }
    else
    {
        ratio_1 = 0.0f;
    }
    float _S531 = _S513.x;
    float _S532 = _S513.y;
    float _S533 = _S513.z;
    float _S534 = _S514.y;
    float _S535 = _S514.z;
    float _S536 = _S514.w;
    uint _S537 = (&kernelContext_45)->params_0->bases_0.w;
    uint side_4 = 0U;
    for(;;)
    {
        if(side_4 < 2U)
        {
        }
        else
        {
            break;
        }
        uint r_4 = 2U * b_8 + side_4;
        bool _S538 = side_4 == 0U;
        uint sn_3;
        if(_S538)
        {
            sn_3 = _S498;
        }
        else
        {
            sn_3 = nb_2;
        }
        uint dn_3;
        if(_S538)
        {
            dn_3 = nb_2;
        }
        else
        {
            dn_3 = _S498;
        }
        float3 ft1_3;
        if(_S538)
        {
            ft1_3 = t1_8;
        }
        else
        {
            ft1_3 = - t1_8;
        }
        float3 fn_3;
        if(_S538)
        {
            fn_3 = nrm_5;
        }
        else
        {
            fn_3 = - nrm_5;
        }
        float4 _S539 = node_0(sn_3, 3U, &kernelContext_45);
        float3 _S540 = _S539.xyz;
        float4 _S541 = node_0(sn_3, 4U, &kernelContext_45);
        float3 _S542 = _S541.xyz;
        float4 _S543 = node_0(sn_3, 13U, &kernelContext_45);
        float3 vs_1 = _S540 + cross(_S542, _S524 - _S543.xyz);
        float4 _S544 = node_0(dn_3, 3U, &kernelContext_45);
        float3 _S545 = _S544.xyz;
        float4 _S546 = node_0(dn_3, 4U, &kernelContext_45);
        float3 _S547 = _S546.xyz;
        float4 _S548 = node_0(dn_3, 13U, &kernelContext_45);
        float3 vd_1 = _S545 + cross(_S547, _S524 - _S548.xyz);
        float4 _S549 = node_0(sn_3, 13U, &kernelContext_45);
        float3 q_5 = _S524 - _S549.xyz;
        float3 dv_2 = vd_1 - vs_1;
        put_0(_S537, 3U, r_4, 0U, ln_ratio_0(area_1, L_2 * L_2), &kernelContext_45);
        put_0(_S537, 3U, r_4, 1U, dot(ft1_3, q_5) / L_2, &kernelContext_45);
        put_0(_S537, 3U, r_4, 2U, dot(_S522, q_5) / L_2, &kernelContext_45);
        put_0(_S537, 3U, r_4, 3U, dot(fn_3, q_5) / L_2, &kernelContext_45);
        put_0(_S537, 3U, r_4, 4U, ln_ratio_0(_S511.y, _S526), &kernelContext_45);
        float _S550 = ln_ratio_0(_S511.z, _S526);
        put_0(_S537, 3U, r_4, 5U, _S550, &kernelContext_45);
        put_0(_S537, 3U, r_4, 6U, _S550, &kernelContext_45);
        put_0(_S537, 3U, r_4, 7U, ln_ratio_0(_S511.w, _S527), &kernelContext_45);
        put_0(_S537, 3U, r_4, 8U, ln_ratio_0(_S512.x, _S527), &kernelContext_45);
        put_0(_S537, 3U, r_4, 9U, ln_ratio_0(_S512.y, _S527), &kernelContext_45);
        put_0(_S537, 3U, r_4, 10U, ln_ratio_0(_S531, _S505), &kernelContext_45);
        put_0(_S537, 3U, r_4, 11U, ln_ratio_0(_S533, _S505), &kernelContext_45);
        put_0(_S537, 3U, r_4, 12U, ln_ratio_0(_S532, _S505), &kernelContext_45);
        put_0(_S537, 3U, r_4, 13U, ln_ratio_0(_S534 * area_1, _S506), &kernelContext_45);
        put_0(_S537, 3U, r_4, 14U, ln_ratio_0(_S535 * area_1, _S506), &kernelContext_45);
        put_0(_S537, 3U, r_4, 15U, ln_ratio_0(_S536 * area_1, _S506), &kernelContext_45);
        put_0(_S537, 3U, r_4, 16U, _S512.z, &kernelContext_45);
        put_0(_S537, 3U, r_4, 17U, _S528.damage_0, &kernelContext_45);
        put_0(_S537, 3U, r_4, 18U, _S528.crush_0, &kernelContext_45);
        put_0(_S537, 3U, r_4, 19U, _S530, &kernelContext_45);
        float _S551 = log(1.0f + max(_S528.utilization_0, 0.0f));
        put_0(_S537, 3U, r_4, 20U, _S551, &kernelContext_45);
        put_0(_S537, 3U, r_4, 21U, _S551, &kernelContext_45);
        float _S552;
        if(rebar_1)
        {
            _S552 = 1.0f;
        }
        else
        {
            _S552 = 0.0f;
        }
        put_0(_S537, 3U, r_4, 22U, _S552, &kernelContext_45);
        put_0(_S537, 3U, r_4, 23U, 0.0f, &kernelContext_45);
        float _S553;
        if(in_plug_0)
        {
            _S553 = 1.0f;
        }
        else
        {
            _S553 = 0.0f;
        }
        put_0(_S537, 3U, r_4, 24U, _S553, &kernelContext_45);
        put_0(_S537, 3U, r_4, 25U, ratio_1, &kernelContext_45);
        put_0(_S537, 3U, r_4, 26U, dot(ft1_3, dv_2) / _S504, &kernelContext_45);
        put_0(_S537, 3U, r_4, 27U, dot(_S522, dv_2) / _S504, &kernelContext_45);
        put_0(_S537, 3U, r_4, 28U, dot(fn_3, dv_2) / _S504, &kernelContext_45);
        float4 _S554 = node_0(sn_3, 1U, &kernelContext_45);
        put_rot6_0(_S537, 3U, r_4, 29U, ft1_3, _S522, quat_matrix_0(_S554), &kernelContext_45);
        float4 _S555 = node_0(dn_3, 1U, &kernelContext_45);
        put_rot6_0(_S537, 3U, r_4, 35U, ft1_3, _S522, quat_matrix_0(_S555), &kernelContext_45);
        put_0(_S537, 3U, r_4, 41U, 0.0f, &kernelContext_45);
        put_0(_S537, 3U, r_4, 42U, 0.0f, &kernelContext_45);
        put_0(_S537, 3U, r_4, 43U, 0.0f, &kernelContext_45);
        side_4 = side_4 + 1U;
    }
    return;
}

struct Quat_0
{
    float w_3;
    float x_2;
    float y_1;
    float z_1;
};

Quat_0 quat_of_0(float4 q_6)
{
    thread Quat_0 r_5;
    (&r_5)->x_2 = q_6.x;
    (&r_5)->y_1 = q_6.y;
    (&r_5)->z_1 = q_6.z;
    (&r_5)->w_3 = q_6.w;
    return r_5;
}

Quat_0 quat_mul_0(const Quat_0 thread* a_12, const Quat_0 thread* o_0)
{
    thread Quat_0 r_6;
    (&r_6)->w_3 = a_12->w_3 * o_0->w_3 - a_12->x_2 * o_0->x_2 - a_12->y_1 * o_0->y_1 - a_12->z_1 * o_0->z_1;
    (&r_6)->x_2 = a_12->w_3 * o_0->x_2 + a_12->x_2 * o_0->w_3 + a_12->y_1 * o_0->z_1 - a_12->z_1 * o_0->y_1;
    (&r_6)->y_1 = a_12->w_3 * o_0->y_1 - a_12->x_2 * o_0->z_1 + a_12->y_1 * o_0->w_3 + a_12->z_1 * o_0->x_2;
    (&r_6)->z_1 = a_12->w_3 * o_0->z_1 + a_12->x_2 * o_0->y_1 - a_12->y_1 * o_0->x_2 + a_12->z_1 * o_0->w_3;
    return r_6;
}

float4 quat_vec_0(const Quat_0 thread* q_7)
{
    return float4(q_7->x_2, q_7->y_1, q_7->z_1, q_7->w_3);
}

float4 qmulf_0(float4 a_13, float4 b_9)
{
    Quat_0 _S556 = quat_of_0(b_9);
    thread Quat_0 _S557 = quat_of_0(a_13);
    thread Quat_0 _S558 = _S556;
    Quat_0 _S559 = quat_mul_0(&_S557, &_S558);
    thread Quat_0 _S560 = _S559;
    float4 _S561 = quat_vec_0(&_S560);
    return _S561;
}

float3 to_zup_f_0(float3 v_7)
{
    return float3(v_7.x, - v_7.z, v_7.y);
}

float3 rotate_0(const Quat_0 thread* q_8, float3 v_8)
{
    float3 qv_0 = float3(q_8->x_2, q_8->y_1, q_8->z_1);
    float3 t_15 = cross(qv_0, v_8) * float3(2.0f) ;
    return v_8 + t_15 * float3(q_8->w_3)  + cross(qv_0, t_15);
}

float3 inverse_rotate_0(const Quat_0 thread* q_9, float3 v_9)
{
    thread Quat_0 c_4;
    (&c_4)->w_3 = q_9->w_3;
    (&c_4)->x_2 = - q_9->x_2;
    (&c_4)->y_1 = - q_9->y_1;
    (&c_4)->z_1 = - q_9->z_1;
    thread Quat_0 _S562 = c_4;
    float3 _S563 = rotate_0(&_S562, v_9);
    return _S563;
}

float4 rotvec_quat_f_0(float3 r_7)
{
    float th_0 = length(r_7);
    if(th_0 < 9.999999960041972e-13f)
    {
        return normalize(float4(float3(0.5f)  * r_7, 1.0f));
    }
    float _S564 = 0.5f * th_0;
    return float4(r_7 / float3(th_0)  * float3(sin(_S564)) , cos(_S564));
}

float4 qconj_0(float4 q_10)
{
    return float4(- q_10.xyz, q_10.w);
}

float3 quat_rotvec_0(float4 q_11)
{
    float4 p_2;
    if((q_11.w) < 0.0f)
    {
        p_2 = - q_11;
    }
    else
    {
        p_2 = q_11;
    }
    float3 _S565 = p_2.xyz;
    float s_6 = length(_S565);
    if(s_6 < 9.999999960041972e-13f)
    {
        return float3(2.0f)  * _S565;
    }
    return _S565 / float3(s_6)  * float3((2.0f * atan2(s_6, p_2.w))) ;
}

[[kernel]] void feat_deform(uint3 id_5 [[thread_position_in_grid]], FeatParams_0 constant* params_16 [[buffer(0)]], uint device* bmap_16 [[buffer(11)]], uint device* counters_16 [[buffer(28)]], Island_natural_0 device* islands_16 [[buffer(24)]], BondStatic_natural_0 device* bond_static_16 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_16 [[buffer(13)]], uint device* sinc_16 [[buffer(9)]], packed_uint4 device* slook_16 [[buffer(10)]], BondDyn_natural_0 device* bond_dyn_16 [[buffer(15)]], uint device* ctl_16 [[buffer(1)]], float device* work_16 [[buffer(18)]], packed_float4 device* nodes_16 [[buffer(3)]], packed_uint2 device* key_node_16 [[buffer(12)]], packed_float4 device* sc_16 [[buffer(7)]], packed_float4 device* bonds_16 [[buffer(4)]], packed_float4 device* sb_16 [[buffer(8)]], packed_uint4 device* bedge_16 [[buffer(22)]], packed_float4 device* calls_16 [[buffer(2)]], packed_float4 device* imps_16 [[buffer(16)]], packed_uint4 device* cisl_16 [[buffer(5)]], packed_float4 device* cislf_16 [[buffer(6)]], packed_float4 device* contacts_16 [[buffer(17)]], uint device* fctl_16 [[buffer(23)]], uint device* edges_16 [[buffer(20)]], uint device* cols_16 [[buffer(21)]], float device* feat_16 [[buffer(19)]], packed_float4 device* ctab_16 [[buffer(25)]], packed_float4 device* bodies_16 [[buffer(26)]], packed_float4 device* state_16 [[buffer(27)]])
{
    thread KernelContext_0 kernelContext_46;
    (&kernelContext_46)->params_0 = params_16;
    (&kernelContext_46)->bmap_0 = bmap_16;
    (&kernelContext_46)->counters_0 = counters_16;
    (&kernelContext_46)->islands_0 = islands_16;
    (&kernelContext_46)->bond_static_0 = bond_static_16;
    (&kernelContext_46)->chunks_0 = chunks_16;
    (&kernelContext_46)->sinc_0 = sinc_16;
    (&kernelContext_46)->slook_0 = slook_16;
    (&kernelContext_46)->bond_dyn_0 = bond_dyn_16;
    (&kernelContext_46)->ctl_0 = ctl_16;
    (&kernelContext_46)->work_0 = work_16;
    (&kernelContext_46)->nodes_0 = nodes_16;
    (&kernelContext_46)->key_node_0 = key_node_16;
    (&kernelContext_46)->sc_0 = sc_16;
    (&kernelContext_46)->bonds_0 = bonds_16;
    (&kernelContext_46)->sb_0 = sb_16;
    (&kernelContext_46)->bedge_0 = bedge_16;
    (&kernelContext_46)->calls_0 = calls_16;
    (&kernelContext_46)->imps_0 = imps_16;
    (&kernelContext_46)->cisl_0 = cisl_16;
    (&kernelContext_46)->cislf_0 = cislf_16;
    (&kernelContext_46)->contacts_0 = contacts_16;
    (&kernelContext_46)->fctl_0 = fctl_16;
    (&kernelContext_46)->edges_0 = edges_16;
    (&kernelContext_46)->cols_0 = cols_16;
    (&kernelContext_46)->feat_0 = feat_16;
    (&kernelContext_46)->ctab_0 = ctab_16;
    (&kernelContext_46)->bodies_0 = bodies_16;
    (&kernelContext_46)->state_0 = state_16;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_16;
    (&kernelContext_46)->g_red_0 = &g_red_16;
    threadgroup array<float, int(64)> g_mv_16;
    (&kernelContext_46)->g_mv_0 = &g_mv_16;
    threadgroup array<uint, int(64)> g_mi_16;
    (&kernelContext_46)->g_mi_0 = &g_mi_16;
    threadgroup array<uint, int(4096)> g_plug_16;
    (&kernelContext_46)->g_plug_0 = &g_plug_16;
    threadgroup uint g_flag_16;
    (&kernelContext_46)->g_flag_0 = &g_flag_16;
    threadgroup array<uint, int(64)> g_scan_16;
    (&kernelContext_46)->g_scan_0 = &g_scan_16;
    threadgroup float4 g_scalars_16;
    (&kernelContext_46)->g_scalars_0 = &g_scalars_16;
    uint k_11 = id_5.x;
    if(k_11 >= (params_16->chunk_count_0))
    {
        return;
    }
    uint4 _S566 = uint4(((&kernelContext_46)->chunks_0+k_11)->info_1) ;
    uint2 _S567 = uint2(*((&kernelContext_46)->key_node_0+((&kernelContext_46)->sinc_0[2U + _S566.z] + _S566.w))) ;
    uint _S568 = _S567.x;
    bool _S569;
    if(_S568 == 0U)
    {
        _S569 = true;
    }
    else
    {
        _S569 = (_S567.y) != ((&kernelContext_46)->params_0->prev_frame_0);
    }
    if(_S569)
    {
        return;
    }
    uint n_10 = _S568 - 1U;
    float4 _S570 = node_0(n_10, 3U, &kernelContext_46);
    if((as_type<uint>(((float4(*((&kernelContext_46)->calls_0+(10U * (as_type<uint>((_S570.w))) + 4U))) ).w))) != 0U)
    {
        return;
    }
    uint i_4 = _S566.y;
    Island_natural_0 device* _S571 = (&kernelContext_46)->islands_0+i_4;
    bool anchored_1 = (((uint4(_S571->info_0) ).x) & 1U) != 0U;
    float4 _S572 = float4(_S571->rotation_0) ;
    float3 com_b_0 = (float4(_S571->com_0) ).xyz;
    float3 _S573 = float3(0.0f) ;
    uint _S574 = 3U * i_4;
    uint body_2 = (as_type<uint>(((float4(*((&kernelContext_46)->ctab_0+((&kernelContext_46)->params_0->engine_base_0 + _S574))) ).x)));
    float3 com_w_0;
    float3 com_b_1;
    float3 wb_1;
    float3 vb_0;
    float4 q_12;
    if(body_2 != 4294967295U)
    {
        uint _S575 = 7U * body_2;
        float4 _S576 = normalize(qmulf_0(qmulf_0(float4(0.70710676908493042f, 0.0f, 0.0f, 0.70710676908493042f), float4(*((&kernelContext_46)->bodies_0+(_S575 + 1U))) ), float4(*((&kernelContext_46)->ctab_0+((&kernelContext_46)->params_0->engine_base_0 + _S574 + 1U))) ));
        float3 _S577 = (float4(*((&kernelContext_46)->ctab_0+((&kernelContext_46)->params_0->engine_base_0 + _S574 + 2U))) ).xyz;
        float3 _S578 = to_zup_f_0((float4(*((&kernelContext_46)->bodies_0+(_S575 + 3U))) ).xyz);
        float3 _S579 = to_zup_f_0((float4(*((&kernelContext_46)->bodies_0+(_S575 + 4U))) ).xyz);
        com_w_0 = to_zup_f_0((float4(*((&kernelContext_46)->bodies_0+_S575)) ).xyz);
        q_12 = _S576;
        com_b_1 = _S577;
        wb_1 = _S579;
        vb_0 = _S578;
    }
    else
    {
        float3 _S580 = (float4(_S571->position_0) ).xyz;
        thread Quat_0 _S581 = quat_of_0(_S572);
        float3 _S582 = rotate_0(&_S581, com_b_0);
        com_w_0 = _S580 + _S582;
        q_12 = _S572;
        com_b_1 = com_b_0;
        wb_1 = _S573;
        vb_0 = _S573;
    }
    Quat_0 _S583 = quat_of_0(q_12);
    thread Quat_0 _S584 = _S583;
    float3 _S585 = rotate_0(&_S584, com_b_1);
    float3 origin_0 = com_w_0 - _S585;
    float3 center_1 = (float4(((&kernelContext_46)->chunks_0+k_11)->center_0) ).xyz;
    float4 _S586 = node_0(n_10, 13U, &kernelContext_46);
    float3 _S587 = _S586.xyz;
    float4 _S588 = node_0(n_10, 10U, &kernelContext_46);
    float3 _S589 = _S587 + _S588.xyz - origin_0;
    thread Quat_0 _S590 = _S583;
    float3 _S591 = inverse_rotate_0(&_S590, _S589);
    float3 u_1 = _S591 - center_1;
    uint _S592 = 24U * n_10 + 17U;
    float4 rest_0 = float4(*((&kernelContext_46)->work_0+_S592), *((&kernelContext_46)->work_0+(_S592 + 1U)), *((&kernelContext_46)->work_0+(_S592 + 2U)), *((&kernelContext_46)->work_0+(_S592 + 3U)));
    float4 _S593 = node_0(n_10, 11U, &kernelContext_46);
    float4 _S594 = rotvec_quat_f_0(_S593.xyz);
    float4 _S595 = node_0(n_10, 1U, &kernelContext_46);
    float3 th_1 = quat_rotvec_0(normalize(qmulf_0(qconj_0(q_12), normalize(qmulf_0(_S594, qmulf_0(_S595, qconj_0(rest_0)))))));
    float3 _S596 = center_1 + u_1 - com_b_1;
    thread Quat_0 _S597 = _S583;
    float3 _S598 = rotate_0(&_S597, _S596);
    float4 _S599 = node_0(n_10, 8U, &kernelContext_46);
    float3 v1_0 = _S599.xyz;
    float4 _S600 = node_0(n_10, 9U, &kernelContext_46);
    float3 w1_0 = _S600.xyz;
    float3 vh_0;
    if(anchored_1)
    {
        float3 _S601 = v1_0 - cross(wb_1, _S598);
        thread Quat_0 _S602 = _S583;
        float3 _S603 = inverse_rotate_0(&_S602, _S601);
        vh_0 = _S603;
    }
    else
    {
        float3 _S604 = v1_0 - vb_0 - cross(wb_1, _S598);
        thread Quat_0 _S605 = _S583;
        float3 _S606 = inverse_rotate_0(&_S605, _S604);
        vh_0 = _S606;
    }
    float3 wh_0;
    if(anchored_1)
    {
        thread Quat_0 _S607 = _S583;
        float3 _S608 = inverse_rotate_0(&_S607, w1_0);
        wh_0 = _S608;
    }
    else
    {
        float3 _S609 = w1_0 - wb_1;
        thread Quat_0 _S610 = _S583;
        float3 _S611 = inverse_rotate_0(&_S610, _S609);
        wh_0 = _S611;
    }
    uint _S612 = 4U * k_11;
    *((&kernelContext_46)->state_0+_S612) = packed_float4(float4(u_1, (float4(*((&kernelContext_46)->state_0+_S612)) ).w)) ;
    uint _S613 = _S612 + 1U;
    *((&kernelContext_46)->state_0+_S613) = packed_float4(float4(th_1, (float4(*((&kernelContext_46)->state_0+_S613)) ).w)) ;
    uint _S614 = _S612 + 2U;
    *((&kernelContext_46)->state_0+_S614) = packed_float4(float4(vh_0, (float4(*((&kernelContext_46)->state_0+_S614)) ).w)) ;
    uint _S615 = _S612 + 3U;
    *((&kernelContext_46)->state_0+_S615) = packed_float4(float4(wh_0, (float4(*((&kernelContext_46)->state_0+_S615)) ).w)) ;
    return;
}

