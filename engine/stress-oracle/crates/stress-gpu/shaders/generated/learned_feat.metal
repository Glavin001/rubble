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
    packed_float4 inertia0_0;
    packed_float4 inertia1_0;
    packed_float4 inertia2_0;
    packed_float4 inv0_0;
    packed_float4 inv1_0;
    packed_float4 inv2_0;
    packed_float4 scale_0;
    packed_uint4 info_0;
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
    BondStatic_natural_0 device* bond_static_0;
    ChunkStatic_natural_0 device* chunks_0;
    uint device* sinc_0;
    packed_uint4 device* slook_0;
    uint device* bmap_0;
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
    array<array<float4, int(64)>, int(2)> threadgroup* g_red_0;
    array<float, int(64)> threadgroup* g_mv_0;
    array<uint, int(64)> threadgroup* g_mi_0;
    array<uint, int(4096)> threadgroup* g_plug_0;
    uint threadgroup* g_flag_0;
    array<uint, int(64)> threadgroup* g_scan_0;
    float4 threadgroup* g_scalars_0;
};

uint chunk_row_of_0(uint layout_chunk_0, KernelContext_0 thread* kernelContext_0)
{
    uint4 _S1 = uint4((kernelContext_0->chunks_0+layout_chunk_0)->info_0) ;
    return kernelContext_0->sinc_0[2U + _S1.z] + _S1.w;
}

uint bond_row_lookup_0(uint ga_0, uint gb_0, KernelContext_0 thread* kernelContext_1)
{
    bool _S2;
    uint _S3 = min(ga_0, gb_0);
    uint _S4 = max(ga_0, gb_0);
    uint _S5 = kernelContext_1->params_0->rows_0;
    uint lo_0 = 0U;
    uint hi_0 = _S5;
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
        uint4 _S6 = uint4(*(kernelContext_1->slook_0+mid_0)) ;
        uint _S7 = _S6.x;
        if(_S7 < _S3)
        {
            _S2 = true;
        }
        else
        {
            if(_S7 == _S3)
            {
                _S2 = (_S6.y) < _S4;
            }
            else
            {
                _S2 = false;
            }
        }
        if(_S2)
        {
            lo_0 = mid_0 + 1U;
        }
        else
        {
            hi_0 = mid_0;
        }
    }
    if(lo_0 < (kernelContext_1->params_0->rows_0))
    {
        _S2 = ((uint4(*(kernelContext_1->slook_0+lo_0)) ).x) == _S3;
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        _S2 = ((uint4(*(kernelContext_1->slook_0+lo_0)) ).y) == _S4;
    }
    else
    {
        _S2 = false;
    }
    if(_S2)
    {
        return (uint4(*(kernelContext_1->slook_0+lo_0)) ).z;
    }
    return 4294967295U;
}

bool connected_0(const JointState_0 thread* st_0, bool has_rebar_0)
{
    bool _S8;
    if((st_0->damage_0) < 1.0f)
    {
        _S8 = true;
    }
    else
    {
        if(has_rebar_0)
        {
            _S8 = (st_0->rebar_broken_0) == 0.0f;
        }
        else
        {
            _S8 = false;
        }
    }
    return _S8;
}

[[kernel]] void feat_map(uint3 id_0 [[thread_position_in_grid]], FeatParams_0 constant* params_1 [[buffer(0)]], BondStatic_natural_0 device* bond_static_1 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_1 [[buffer(13)]], uint device* sinc_1 [[buffer(9)]], packed_uint4 device* slook_1 [[buffer(10)]], uint device* bmap_1 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_1 [[buffer(15)]], uint device* ctl_1 [[buffer(1)]], float device* work_1 [[buffer(18)]], packed_float4 device* nodes_1 [[buffer(3)]], packed_uint2 device* key_node_1 [[buffer(12)]], packed_float4 device* sc_1 [[buffer(7)]], packed_float4 device* bonds_1 [[buffer(4)]], packed_float4 device* sb_1 [[buffer(8)]], packed_uint4 device* bedge_1 [[buffer(22)]], packed_float4 device* calls_1 [[buffer(2)]], packed_float4 device* imps_1 [[buffer(16)]], packed_uint4 device* cisl_1 [[buffer(5)]], packed_float4 device* cislf_1 [[buffer(6)]], packed_float4 device* contacts_1 [[buffer(17)]], uint device* fctl_1 [[buffer(23)]], uint device* edges_1 [[buffer(20)]], uint device* cols_1 [[buffer(21)]], float device* feat_1 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_2;
    (&kernelContext_2)->params_0 = params_1;
    (&kernelContext_2)->bond_static_0 = bond_static_1;
    (&kernelContext_2)->chunks_0 = chunks_1;
    (&kernelContext_2)->sinc_0 = sinc_1;
    (&kernelContext_2)->slook_0 = slook_1;
    (&kernelContext_2)->bmap_0 = bmap_1;
    (&kernelContext_2)->bond_dyn_0 = bond_dyn_1;
    (&kernelContext_2)->ctl_0 = ctl_1;
    (&kernelContext_2)->work_0 = work_1;
    (&kernelContext_2)->nodes_0 = nodes_1;
    (&kernelContext_2)->key_node_0 = key_node_1;
    (&kernelContext_2)->sc_0 = sc_1;
    (&kernelContext_2)->bonds_0 = bonds_1;
    (&kernelContext_2)->sb_0 = sb_1;
    (&kernelContext_2)->bedge_0 = bedge_1;
    (&kernelContext_2)->calls_0 = calls_1;
    (&kernelContext_2)->imps_0 = imps_1;
    (&kernelContext_2)->cisl_0 = cisl_1;
    (&kernelContext_2)->cislf_0 = cislf_1;
    (&kernelContext_2)->contacts_0 = contacts_1;
    (&kernelContext_2)->fctl_0 = fctl_1;
    (&kernelContext_2)->edges_0 = edges_1;
    (&kernelContext_2)->cols_0 = cols_1;
    (&kernelContext_2)->feat_0 = feat_1;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_1;
    (&kernelContext_2)->g_red_0 = &g_red_1;
    threadgroup array<float, int(64)> g_mv_1;
    (&kernelContext_2)->g_mv_0 = &g_mv_1;
    threadgroup array<uint, int(64)> g_mi_1;
    (&kernelContext_2)->g_mi_0 = &g_mi_1;
    threadgroup array<uint, int(4096)> g_plug_1;
    (&kernelContext_2)->g_plug_0 = &g_plug_1;
    threadgroup uint g_flag_1;
    (&kernelContext_2)->g_flag_0 = &g_flag_1;
    threadgroup array<uint, int(64)> g_scan_1;
    (&kernelContext_2)->g_scan_0 = &g_scan_1;
    threadgroup float4 g_scalars_1;
    (&kernelContext_2)->g_scalars_0 = &g_scalars_1;
    uint l_0 = id_0.x;
    if(l_0 >= (params_1->layout_bonds_0))
    {
        return;
    }
    JointBond_natural_0 law_1 = ((&kernelContext_2)->bond_static_0+l_0)->law_0;
    uint4 _S9 = uint4(((&kernelContext_2)->bond_static_0+l_0)->law_0.ids_0) ;
    uint _S10 = chunk_row_of_0(_S9.y, &kernelContext_2);
    uint _S11 = chunk_row_of_0(_S9.z, &kernelContext_2);
    uint _S12 = bond_row_lookup_0(_S10, _S11, &kernelContext_2);
    *((&kernelContext_2)->bmap_0+l_0) = _S12;
    if(_S12 == 4294967295U)
    {
        return;
    }
    *((&kernelContext_2)->bmap_0+((&kernelContext_2)->params_0->layout_cap_0 + _S12)) = l_0;
    bool _S13 = ((float4(law_1.stiff1_0) ).w) != 0.0f;
    thread JointState_0 _S14 = ((&kernelContext_2)->bond_dyn_0+l_0)->js_0;
    bool _S15 = connected_0(&_S14, _S13);
    uint at_0 = (&kernelContext_2)->params_0->layout_cap_0 + (&kernelContext_2)->params_0->rows_0 + _S12;
    bool _S16;
    if(!_S15)
    {
        _S16 = (*((&kernelContext_2)->bmap_0+at_0)) == 4294967295U;
    }
    else
    {
        _S16 = false;
    }
    if(_S16)
    {
        *((&kernelContext_2)->bmap_0+at_0) = (&kernelContext_2)->params_0->frame_0;
    }
    return;
}

float4 node_0(uint n_0, uint k_0, KernelContext_0 thread* kernelContext_3)
{
    return float4(*(kernelContext_3->nodes_0+(14U * n_0 + k_0))) ;
}

[[kernel]] void feat_keys(uint3 id_1 [[thread_position_in_grid]], FeatParams_0 constant* params_2 [[buffer(0)]], BondStatic_natural_0 device* bond_static_2 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_2 [[buffer(13)]], uint device* sinc_2 [[buffer(9)]], packed_uint4 device* slook_2 [[buffer(10)]], uint device* bmap_2 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_2 [[buffer(15)]], uint device* ctl_2 [[buffer(1)]], float device* work_2 [[buffer(18)]], packed_float4 device* nodes_2 [[buffer(3)]], packed_uint2 device* key_node_2 [[buffer(12)]], packed_float4 device* sc_2 [[buffer(7)]], packed_float4 device* bonds_2 [[buffer(4)]], packed_float4 device* sb_2 [[buffer(8)]], packed_uint4 device* bedge_2 [[buffer(22)]], packed_float4 device* calls_2 [[buffer(2)]], packed_float4 device* imps_2 [[buffer(16)]], packed_uint4 device* cisl_2 [[buffer(5)]], packed_float4 device* cislf_2 [[buffer(6)]], packed_float4 device* contacts_2 [[buffer(17)]], uint device* fctl_2 [[buffer(23)]], uint device* edges_2 [[buffer(20)]], uint device* cols_2 [[buffer(21)]], float device* feat_2 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_4;
    (&kernelContext_4)->params_0 = params_2;
    (&kernelContext_4)->bond_static_0 = bond_static_2;
    (&kernelContext_4)->chunks_0 = chunks_2;
    (&kernelContext_4)->sinc_0 = sinc_2;
    (&kernelContext_4)->slook_0 = slook_2;
    (&kernelContext_4)->bmap_0 = bmap_2;
    (&kernelContext_4)->bond_dyn_0 = bond_dyn_2;
    (&kernelContext_4)->ctl_0 = ctl_2;
    (&kernelContext_4)->work_0 = work_2;
    (&kernelContext_4)->nodes_0 = nodes_2;
    (&kernelContext_4)->key_node_0 = key_node_2;
    (&kernelContext_4)->sc_0 = sc_2;
    (&kernelContext_4)->bonds_0 = bonds_2;
    (&kernelContext_4)->sb_0 = sb_2;
    (&kernelContext_4)->bedge_0 = bedge_2;
    (&kernelContext_4)->calls_0 = calls_2;
    (&kernelContext_4)->imps_0 = imps_2;
    (&kernelContext_4)->cisl_0 = cisl_2;
    (&kernelContext_4)->cislf_0 = cislf_2;
    (&kernelContext_4)->contacts_0 = contacts_2;
    (&kernelContext_4)->fctl_0 = fctl_2;
    (&kernelContext_4)->edges_0 = edges_2;
    (&kernelContext_4)->cols_0 = cols_2;
    (&kernelContext_4)->feat_0 = feat_2;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_2;
    (&kernelContext_4)->g_red_0 = &g_red_2;
    threadgroup array<float, int(64)> g_mv_2;
    (&kernelContext_4)->g_mv_0 = &g_mv_2;
    threadgroup array<uint, int(64)> g_mi_2;
    (&kernelContext_4)->g_mi_0 = &g_mi_2;
    threadgroup array<uint, int(4096)> g_plug_2;
    (&kernelContext_4)->g_plug_0 = &g_plug_2;
    threadgroup uint g_flag_2;
    (&kernelContext_4)->g_flag_0 = &g_flag_2;
    threadgroup array<uint, int(64)> g_scan_2;
    (&kernelContext_4)->g_scan_0 = &g_scan_2;
    threadgroup float4 g_scalars_2;
    (&kernelContext_4)->g_scalars_0 = &g_scalars_2;
    uint n_1 = id_1.x;
    if(n_1 >= ctl_2[int(2)])
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
        *((&kernelContext_4)->work_0+(24U * n_1 + k_1)) = 0.0f;
        k_1 = k_1 + 1U;
    }
    float4 _S17 = node_0(n_1, 4U, &kernelContext_4);
    if((as_type<uint>((_S17.w))) != 0U)
    {
        return;
    }
    float4 _S18 = node_0(n_1, 12U, &kernelContext_4);
    uint row_0 = (&kernelContext_4)->sinc_0[2U + (as_type<uint>((_S18.z)))] + (as_type<uint>((_S18.w)));
    *((&kernelContext_4)->key_node_0+row_0) = packed_uint2(uint2(n_1 + 1U, (&kernelContext_4)->params_0->frame_0)) ;
    uint _S19 = 6U * row_0;
    float4 _S20 = float4(*((&kernelContext_4)->sc_0+(_S19 + 1U))) ;
    float4 _S21 = float4(*((&kernelContext_4)->sc_0+(_S19 + 2U))) ;
    float4 _S22 = float4(*((&kernelContext_4)->sc_0+(_S19 + 3U))) ;
    uint _S23 = 24U * n_1;
    uint _S24 = _S23 + 9U;
    *((&kernelContext_4)->work_0+_S24) = _S21.x;
    *((&kernelContext_4)->work_0+(_S24 + 1U)) = _S21.y;
    *((&kernelContext_4)->work_0+(_S24 + 2U)) = _S21.z;
    *((&kernelContext_4)->work_0+(_S23 + 12U)) = _S21.w;
    *((&kernelContext_4)->work_0+(_S23 + 13U)) = _S20.y;
    float device* _S25 = (&kernelContext_4)->work_0+(_S23 + 14U);
    float _S26;
    if((_S20.z) != 0.0f)
    {
        _S26 = 1.0f;
    }
    else
    {
        if((_S20.w) != 0.0f)
        {
            _S26 = 2.0f;
        }
        else
        {
            _S26 = 0.0f;
        }
    }
    *_S25 = _S26;
    float4 _S27 = float4(*((&kernelContext_4)->sc_0+(_S19 + 4U))) ;
    float4 _S28 = float4(*((&kernelContext_4)->sc_0+(_S19 + 5U))) ;
    *((&kernelContext_4)->work_0+(_S23 + 15U)) = (float4(*((&kernelContext_4)->sc_0+_S19)) ).y;
    *((&kernelContext_4)->work_0+(_S23 + 16U)) = (as_type<float>((row_0)));
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
        *((&kernelContext_4)->work_0+(_S23 + 17U + k_1)) = _S28[k_1];
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
        *((&kernelContext_4)->work_0+(_S23 + 21U + k_1)) = _S27[k_1];
        k_1 = k_1 + 1U;
    }
    float4 _S29 = node_0(n_1, 13U, &kernelContext_4);
    uint _S30 = 14U * n_1;
    *((&kernelContext_4)->nodes_0+(_S30 + 13U)) = packed_float4(float4(_S29.xyz, _S22.w)) ;
    packed_float4 device* _S31 = (&kernelContext_4)->nodes_0+(_S30 + 2U);
    float3 _S32 = _S22.xyz;
    float4 _S33 = node_0(n_1, 2U, &kernelContext_4);
    *_S31 = packed_float4(float4(_S32, _S33.w)) ;
    return;
}

uint node_of_key_0(uint row_1, KernelContext_0 thread* kernelContext_5)
{
    uint2 _S34 = uint2(*(kernelContext_5->key_node_0+row_1)) ;
    uint _S35 = _S34.x;
    bool _S36;
    if(_S35 != 0U)
    {
        _S36 = (_S34.y) == (kernelContext_5->params_0->frame_0);
    }
    else
    {
        _S36 = false;
    }
    uint _S37;
    if(_S36)
    {
        _S37 = _S35 - 1U;
    }
    else
    {
        _S37 = 4294967295U;
    }
    return _S37;
}

[[kernel]] void feat_brow(uint3 id_2 [[thread_position_in_grid]], FeatParams_0 constant* params_3 [[buffer(0)]], BondStatic_natural_0 device* bond_static_3 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_3 [[buffer(13)]], uint device* sinc_3 [[buffer(9)]], packed_uint4 device* slook_3 [[buffer(10)]], uint device* bmap_3 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_3 [[buffer(15)]], uint device* ctl_3 [[buffer(1)]], float device* work_3 [[buffer(18)]], packed_float4 device* nodes_3 [[buffer(3)]], packed_uint2 device* key_node_3 [[buffer(12)]], packed_float4 device* sc_3 [[buffer(7)]], packed_float4 device* bonds_3 [[buffer(4)]], packed_float4 device* sb_3 [[buffer(8)]], packed_uint4 device* bedge_3 [[buffer(22)]], packed_float4 device* calls_3 [[buffer(2)]], packed_float4 device* imps_3 [[buffer(16)]], packed_uint4 device* cisl_3 [[buffer(5)]], packed_float4 device* cislf_3 [[buffer(6)]], packed_float4 device* contacts_3 [[buffer(17)]], uint device* fctl_3 [[buffer(23)]], uint device* edges_3 [[buffer(20)]], uint device* cols_3 [[buffer(21)]], float device* feat_3 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_6;
    (&kernelContext_6)->params_0 = params_3;
    (&kernelContext_6)->bond_static_0 = bond_static_3;
    (&kernelContext_6)->chunks_0 = chunks_3;
    (&kernelContext_6)->sinc_0 = sinc_3;
    (&kernelContext_6)->slook_0 = slook_3;
    (&kernelContext_6)->bmap_0 = bmap_3;
    (&kernelContext_6)->bond_dyn_0 = bond_dyn_3;
    (&kernelContext_6)->ctl_0 = ctl_3;
    (&kernelContext_6)->work_0 = work_3;
    (&kernelContext_6)->nodes_0 = nodes_3;
    (&kernelContext_6)->key_node_0 = key_node_3;
    (&kernelContext_6)->sc_0 = sc_3;
    (&kernelContext_6)->bonds_0 = bonds_3;
    (&kernelContext_6)->sb_0 = sb_3;
    (&kernelContext_6)->bedge_0 = bedge_3;
    (&kernelContext_6)->calls_0 = calls_3;
    (&kernelContext_6)->imps_0 = imps_3;
    (&kernelContext_6)->cisl_0 = cisl_3;
    (&kernelContext_6)->cislf_0 = cislf_3;
    (&kernelContext_6)->contacts_0 = contacts_3;
    (&kernelContext_6)->fctl_0 = fctl_3;
    (&kernelContext_6)->edges_0 = edges_3;
    (&kernelContext_6)->cols_0 = cols_3;
    (&kernelContext_6)->feat_0 = feat_3;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_3;
    (&kernelContext_6)->g_red_0 = &g_red_3;
    threadgroup array<float, int(64)> g_mv_3;
    (&kernelContext_6)->g_mv_0 = &g_mv_3;
    threadgroup array<uint, int(64)> g_mi_3;
    (&kernelContext_6)->g_mi_0 = &g_mi_3;
    threadgroup array<uint, int(4096)> g_plug_3;
    (&kernelContext_6)->g_plug_0 = &g_plug_3;
    threadgroup uint g_flag_3;
    (&kernelContext_6)->g_flag_0 = &g_flag_3;
    threadgroup array<uint, int(64)> g_scan_3;
    (&kernelContext_6)->g_scan_0 = &g_scan_3;
    threadgroup float4 g_scalars_3;
    (&kernelContext_6)->g_scalars_0 = &g_scalars_3;
    uint b_0 = id_2.x;
    if(b_0 >= ctl_3[int(3)])
    {
        return;
    }
    uint l_1 = (as_type<uint>(((float4(*((&kernelContext_6)->bonds_0+(2U * b_0 + 1U))) ).z)));
    uint device* _S38 = (&kernelContext_6)->bmap_0+l_1;
    uint4 ends_0 = (as_type<uint4>((float4(*((&kernelContext_6)->sb_0+(9U * *_S38 + 7U))) )));
    packed_uint4 device* _S39 = (&kernelContext_6)->bedge_0+b_0;
    uint _S40 = node_of_key_0(ends_0.x, &kernelContext_6);
    uint _S41 = node_of_key_0(ends_0.y, &kernelContext_6);
    *_S39 = packed_uint4(uint4(_S40, _S41, *_S38, l_1)) ;
    return;
}

uint group_uniform_0(uint value_0, uint thread_0)
{
    return value_0;
}

uint call_mask_0(uint g_0, KernelContext_0 thread* kernelContext_7)
{
    return (as_type<uint>(((float4(*(kernelContext_7->calls_0+(10U * g_0 + 5U))) ).w)));
}

uint call_e0_0(uint g_1, KernelContext_0 thread* kernelContext_8)
{
    return (as_type<uint>(((float4(*(kernelContext_8->calls_0+(10U * g_1 + 6U))) ).x)));
}

uint call_ne_0(uint g_2, KernelContext_0 thread* kernelContext_9)
{
    return (as_type<uint>(((float4(*(kernelContext_9->calls_0+(10U * g_2 + 6U))) ).y)));
}

uint call_first_0(uint g_3, KernelContext_0 thread* kernelContext_10)
{
    return (as_type<uint>(((float4(*(kernelContext_10->calls_0+(10U * g_3 + 2U))) ).z)));
}

uint call_chunks_0(uint g_4, KernelContext_0 thread* kernelContext_11)
{
    return (as_type<uint>(((float4(*(kernelContext_11->calls_0+(10U * g_4 + 3U))) ).w)));
}

float float_bits_0(uint bits_0)
{
    return (as_type<float>((bits_0)));
}

float4 group_min_0(uint t_0, float4 a_0, KernelContext_0 thread* kernelContext_12)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_12->g_red_0)[int(0)][t_0] = a_0;
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
            (*kernelContext_12->g_red_0)[int(0)][t_0] = min((*kernelContext_12->g_red_0)[int(0)][t_0], (*kernelContext_12->g_red_0)[int(0)][t_0 + s_0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_0 = s_0 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_12->g_red_0)[int(0)][int(0)];
}

float4 group_sum_0(uint t_1, float4 a_1, KernelContext_0 thread* kernelContext_13)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_13->g_red_0)[int(0)][t_1] = a_1;
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
            (*kernelContext_13->g_red_0)[int(0)][t_1] = (*kernelContext_13->g_red_0)[int(0)][t_1] + (*kernelContext_13->g_red_0)[int(0)][t_1 + s_1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_1 = s_1 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_13->g_red_0)[int(0)][int(0)];
}

uint group_argmin_0(uint t_2, float v_0, uint i_0, KernelContext_0 thread* kernelContext_14)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_14->g_mv_0)[t_2] = v_0;
    (*kernelContext_14->g_mi_0)[t_2] = i_0;
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
        bool _S42;
        if(t_2 < s_2)
        {
            uint _S43 = t_2 + s_2;
            if(((*kernelContext_14->g_mv_0)[_S43]) < (*kernelContext_14->g_mv_0)[t_2])
            {
                _S42 = true;
            }
            else
            {
                if(((*kernelContext_14->g_mv_0)[_S43]) == (*kernelContext_14->g_mv_0)[t_2])
                {
                    _S42 = ((*kernelContext_14->g_mi_0)[_S43]) < (*kernelContext_14->g_mi_0)[t_2];
                }
                else
                {
                    _S42 = false;
                }
            }
        }
        else
        {
            _S42 = false;
        }
        if(_S42)
        {
            uint _S44 = t_2 + s_2;
            (*kernelContext_14->g_mv_0)[t_2] = (*kernelContext_14->g_mv_0)[_S44];
            (*kernelContext_14->g_mi_0)[t_2] = (*kernelContext_14->g_mi_0)[_S44];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_2 = s_2 >> 1U;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_14->g_mi_0)[int(0)];
}

float hertz_duration_0(float m_0, float r_0, float e_0, float v_1)
{
    return 2.86999988555908203f * pow(m_0 * m_0 / (r_0 * e_0 * e_0 * max(abs(v_1), 0.00100000004749745f)), 0.20000000298023224f);
}

[[kernel]] void feat_contacts(uint3 group_0 [[threadgroup_position_in_grid]], uint3 thread_1 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_4 [[buffer(0)]], BondStatic_natural_0 device* bond_static_4 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_4 [[buffer(13)]], uint device* sinc_4 [[buffer(9)]], packed_uint4 device* slook_4 [[buffer(10)]], uint device* bmap_4 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_4 [[buffer(15)]], uint device* ctl_4 [[buffer(1)]], float device* work_4 [[buffer(18)]], packed_float4 device* nodes_4 [[buffer(3)]], packed_uint2 device* key_node_4 [[buffer(12)]], packed_float4 device* sc_4 [[buffer(7)]], packed_float4 device* bonds_4 [[buffer(4)]], packed_float4 device* sb_4 [[buffer(8)]], packed_uint4 device* bedge_4 [[buffer(22)]], packed_float4 device* calls_4 [[buffer(2)]], packed_float4 device* imps_4 [[buffer(16)]], packed_uint4 device* cisl_4 [[buffer(5)]], packed_float4 device* cislf_4 [[buffer(6)]], packed_float4 device* contacts_4 [[buffer(17)]], uint device* fctl_4 [[buffer(23)]], uint device* edges_4 [[buffer(20)]], uint device* cols_4 [[buffer(21)]], float device* feat_4 [[buffer(19)]])
{
    float t0_0;
    bool _S45;
    uint iter_0;
    uint j_0;
    uint bj_0;
    bool marked_0;
    bool _S46;
    bool _S47;
    thread KernelContext_0 kernelContext_15;
    (&kernelContext_15)->params_0 = params_4;
    (&kernelContext_15)->bond_static_0 = bond_static_4;
    (&kernelContext_15)->chunks_0 = chunks_4;
    (&kernelContext_15)->sinc_0 = sinc_4;
    (&kernelContext_15)->slook_0 = slook_4;
    (&kernelContext_15)->bmap_0 = bmap_4;
    (&kernelContext_15)->bond_dyn_0 = bond_dyn_4;
    (&kernelContext_15)->ctl_0 = ctl_4;
    (&kernelContext_15)->work_0 = work_4;
    (&kernelContext_15)->nodes_0 = nodes_4;
    (&kernelContext_15)->key_node_0 = key_node_4;
    (&kernelContext_15)->sc_0 = sc_4;
    (&kernelContext_15)->bonds_0 = bonds_4;
    (&kernelContext_15)->sb_0 = sb_4;
    (&kernelContext_15)->bedge_0 = bedge_4;
    (&kernelContext_15)->calls_0 = calls_4;
    (&kernelContext_15)->imps_0 = imps_4;
    (&kernelContext_15)->cisl_0 = cisl_4;
    (&kernelContext_15)->cislf_0 = cislf_4;
    (&kernelContext_15)->contacts_0 = contacts_4;
    (&kernelContext_15)->fctl_0 = fctl_4;
    (&kernelContext_15)->edges_0 = edges_4;
    (&kernelContext_15)->cols_0 = cols_4;
    (&kernelContext_15)->feat_0 = feat_4;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_4;
    (&kernelContext_15)->g_red_0 = &g_red_4;
    threadgroup array<float, int(64)> g_mv_4;
    (&kernelContext_15)->g_mv_0 = &g_mv_4;
    threadgroup array<uint, int(64)> g_mi_4;
    (&kernelContext_15)->g_mi_0 = &g_mi_4;
    threadgroup array<uint, int(4096)> g_plug_4;
    (&kernelContext_15)->g_plug_0 = &g_plug_4;
    threadgroup uint g_flag_4;
    (&kernelContext_15)->g_flag_0 = &g_flag_4;
    threadgroup array<uint, int(64)> g_scan_4;
    (&kernelContext_15)->g_scan_0 = &g_scan_4;
    threadgroup float4 g_scalars_4;
    (&kernelContext_15)->g_scalars_0 = &g_scalars_4;
    uint g_5 = group_0.x;
    uint t_3 = thread_1.x;
    uint k_2;
    if(g_5 >= ctl_4[int(1)])
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
    uint _S48 = call_mask_0(g_5, &kernelContext_15);
    uint _S49 = group_uniform_0(_S48, t_3);
    uint _S50 = call_e0_0(g_5, &kernelContext_15);
    uint _S51 = group_uniform_0(_S50, t_3);
    uint _S52 = call_ne_0(g_5, &kernelContext_15);
    uint _S53 = group_uniform_0(_S52, t_3);
    uint _S54 = call_first_0(g_5, &kernelContext_15);
    uint _S55 = call_chunks_0(g_5, &kernelContext_15);
    uint _S56 = 10U * g_5 + 8U;
    uint _S57 = (as_type<uint>(((float4(*((&kernelContext_15)->calls_0+_S56)) ).x)));
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
        uint _S58 = 1U << k_2;
        float vmax_1;
        if((_S49 & _S58) == 0U)
        {
            vmax_1 = vmax_0;
            uint _S59 = k_2 + 1U;
            vmax_0 = vmax_1;
            k_2 = _S59;
            continue;
        }
        uint inode_0 = _S54 + _S55 + ((uint)popcount((_S49 & (_S58 - 1U))));
        float4 _S60 = node_0(inode_0, 0U, &kernelContext_15);
        float3 _S61 = _S60.xyz;
        float4 _S62 = node_0(inode_0, 3U, &kernelContext_15);
        float3 _S63 = _S62.xyz;
        uint _S64 = 4U * k_2;
        float _S65 = (float4(*((&kernelContext_15)->imps_0+(_S64 + 2U))) ).x;
        float _S66 = (float4(*((&kernelContext_15)->imps_0+(_S64 + 3U))) ).z;
        float _S67 = (float4(*((&kernelContext_15)->imps_0+_S64)) ).w;
        float _S68 = (float4(*((&kernelContext_15)->imps_0+(_S64 + 1U))) ).w;
        vmax_1 = vmax_0;
        uint e_1 = _S51;
        uint rows_2 = rows_1;
        for(;;)
        {
            if(e_1 < (_S51 + _S53))
            {
            }
            else
            {
                break;
            }
            uint _S69 = 2U * e_1;
            uint4 _S70 = uint4(*((&kernelContext_15)->cisl_0+_S69)) ;
            uint _S71 = _S69 + 1U;
            uint4 _S72 = uint4(*((&kernelContext_15)->cisl_0+_S71)) ;
            float4 _S73 = float4(*((&kernelContext_15)->cislf_0+_S71)) ;
            uint at_1 = _S57 + rows_2;
            uint _S74 = rows_2 + 1U;
            bool _S75 = t_3 == 0U;
            if(_S75)
            {
                *((&kernelContext_15)->contacts_0+(6U * at_1 + 5U)) = packed_float4(float4(0.0f) ) ;
            }
            uint n_isl_0 = group_uniform_0(_S70.z, t_3);
            bool _S76;
            if(n_isl_0 == 0U)
            {
                _S76 = true;
            }
            else
            {
                _S76 = (_S72.z) == 0U;
            }
            if(_S76)
            {
                e_1 = e_1 + 1U;
                rows_2 = _S74;
                continue;
            }
            float3 v_2 = _S63 - _S73.xyz;
            float speed_1 = length(v_2);
            uint _S77;
            if(speed_1 > 0.0f)
            {
                _S77 = 1U;
            }
            else
            {
                _S77 = 0U;
            }
            if((group_uniform_0(_S77, t_3)) == 0U)
            {
                e_1 = e_1 + 1U;
                rows_2 = _S74;
                continue;
            }
            float inf_0 = float_bits_0(2139095040U);
            float4 _S78 = float4(- inf_0) ;
            float4 lo_1 = float4(inf_0) ;
            float4 hi_1 = _S78;
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
                uint n_2 = _S70.y + j_1;
                float4 _S79 = node_0(n_2, 0U, &kernelContext_15);
                float3 p_0 = _S79.xyz;
                float3 _S80 = float3((*((&kernelContext_15)->work_0+(24U * n_2 + 12U)) + _S65)) ;
                float4 _S81 = min(lo_1, float4(p_0 - _S80, 0.0f));
                float4 _S82 = max(hi_1, float4(p_0 + _S80, 0.0f));
                float4 _S83 = node_0(n_2, 0U, &kernelContext_15);
                float msum_1 = msum_0 + _S83.w;
                uint j_2 = j_1 + 64U;
                lo_1 = _S81;
                hi_1 = _S82;
                j_1 = j_2;
                msum_0 = msum_1;
            }
            float4 _S84 = group_min_0(t_3, lo_1, &kernelContext_15);
            float4 _S85 = group_min_0(t_3, - hi_1, &kernelContext_15);
            float4 _S86 = - _S85;
            float4 _S87 = group_sum_0(t_3, float4(msum_0, 0.0f, 0.0f, 0.0f), &kernelContext_15);
            float cl_mass_0 = _S87.x;
            float _S88 = (&kernelContext_15)->params_0->dt_0;
            float t0_1 = 0.0f;
            float t1_1 = _S88;
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
                uint _S89 = a_2;
                if((abs(v_2[a_2])) < 9.999999960041972e-13f)
                {
                    uint _S90 = a_2;
                    if((_S61[a_2]) < (_S84[a_2]))
                    {
                        _S45 = true;
                    }
                    else
                    {
                        _S45 = (_S61[_S90]) > (_S86[a_2]);
                    }
                    if(_S45)
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
                float ta_0 = (_S84[a_2] - _S61[a_2]) / v_2[_S89];
                float tb_0 = (_S86[a_2] - _S61[a_2]) / v_2[_S89];
                float _S91 = min(t1_1, max(ta_0, tb_0));
                t0_1 = max(t0_1, min(ta_0, tb_0));
                t1_1 = _S91;
                a_2 = a_2 + 1U;
            }
            uint _S92;
            if(t0_1 > t1_1)
            {
                _S92 = 1U;
            }
            else
            {
                _S92 = 0U;
            }
            if((group_uniform_0(_S92, t_3)) != 0U)
            {
                e_1 = e_1 + 1U;
                rows_2 = _S74;
                continue;
            }
            float3 d_0 = v_2 / float3(speed_1) ;
            float3 hit_0 = _S61 + v_2 * float3(t0_1) ;
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
                float4 _S93 = node_0(_S70.y + j_3, 0U, &kernelContext_15);
                float3 q_0 = _S93.xyz - hit_0;
                float along_0 = dot(q_0, d_0);
                float dist_0 = length(q_0 - d_0 * float3(along_0) ) + abs(along_0) * 0.00100000004749745f;
                if(dist_0 < t0_0)
                {
                    t0_0 = dist_0;
                    best_j_0 = j_3;
                }
                j_3 = j_3 + 64U;
            }
            uint _S94 = group_argmin_0(t_3, t0_0, best_j_0, &kernelContext_15);
            uint _S95 = _S70.y;
            uint entry_n_0 = _S95 + _S94;
            float4 _S96 = node_0(entry_n_0, 0U, &kernelContext_15);
            float _S97 = dot(_S96.xyz - hit_0, d_0);
            bool _S98 = n_isl_0 <= 4096U;
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
                if(j_4 == _S94)
                {
                    iter_0 = 1U;
                }
                else
                {
                    iter_0 = 0U;
                }
                (*(&kernelContext_15)->g_plug_0)[j_4] = iter_0;
                j_4 = j_4 + 64U;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            iter_0 = 0U;
            for(;;)
            {
                if(iter_0 < n_isl_0)
                {
                    _S45 = _S98;
                }
                else
                {
                    _S45 = false;
                }
                if(_S45)
                {
                }
                else
                {
                    break;
                }
                if(_S75)
                {
                    *(&kernelContext_15)->g_flag_0 = 0U;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                j_0 = t_3;
                for(;;)
                {
                    if(j_0 < (_S72.z))
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint4 _S99 = uint4(*((&kernelContext_15)->bedge_0+(_S72.y + j_0))) ;
                    uint _S100 = _S99.x;
                    if(_S100 == 4294967295U)
                    {
                        marked_0 = true;
                    }
                    else
                    {
                        marked_0 = (_S99.y) == 4294967295U;
                    }
                    if(marked_0)
                    {
                        j_0 = j_0 + 64U;
                        continue;
                    }
                    uint la_0 = _S100 - _S95;
                    uint lb_0 = _S99.y - _S95;
                    if(la_0 >= n_isl_0)
                    {
                        _S46 = true;
                    }
                    else
                    {
                        _S46 = lb_0 >= n_isl_0;
                    }
                    if(_S46)
                    {
                        _S47 = true;
                    }
                    else
                    {
                        _S47 = ((*(&kernelContext_15)->g_plug_0)[la_0]) == (*(&kernelContext_15)->g_plug_0)[lb_0];
                    }
                    if(_S47)
                    {
                        j_0 = j_0 + 64U;
                        continue;
                    }
                    if(((*(&kernelContext_15)->g_plug_0)[la_0]) != 0U)
                    {
                        bj_0 = lb_0;
                    }
                    else
                    {
                        bj_0 = la_0;
                    }
                    uint _S101 = _S95 + bj_0;
                    float4 _S102 = node_0(_S101, 0U, &kernelContext_15);
                    float3 q_1 = _S102.xyz - hit_0;
                    float along_1 = dot(q_1, d_0);
                    float lateral_0 = length(q_1 - d_0 * float3(along_1) );
                    bool _S103;
                    if(along_1 >= (_S97 - *((&kernelContext_15)->work_0+(24U * _S101 + 12U))))
                    {
                        _S103 = lateral_0 <= (_S66 + max(along_1 - _S97, 0.0f));
                    }
                    else
                    {
                        _S103 = false;
                    }
                    if(_S103)
                    {
                        (*(&kernelContext_15)->g_plug_0)[bj_0] = 1U;
                        *(&kernelContext_15)->g_flag_0 = 1U;
                    }
                    j_0 = j_0 + 64U;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if((group_uniform_0(*(&kernelContext_15)->g_flag_0, t_3)) == 0U)
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
                if(((*(&kernelContext_15)->g_plug_0)[j_0]) != 0U)
                {
                    float4 _S104 = node_0(_S95 + j_0, 0U, &kernelContext_15);
                    acc_0 = acc_0 + float4(_S104.w, 1.0f, 0.0f, 0.0f);
                }
                j_0 = j_0 + 64U;
            }
            bj_0 = t_3;
            for(;;)
            {
                if(bj_0 < (_S72.z))
                {
                }
                else
                {
                    break;
                }
                uint4 _S105 = uint4(*((&kernelContext_15)->bedge_0+(_S72.y + bj_0))) ;
                uint _S106 = _S105.x;
                if(_S106 == 4294967295U)
                {
                    marked_0 = true;
                }
                else
                {
                    marked_0 = (_S105.y) == 4294967295U;
                }
                if(marked_0)
                {
                    bj_0 = bj_0 + 64U;
                    continue;
                }
                uint la_1 = _S106 - _S95;
                uint lb_1 = _S105.y - _S95;
                if(la_1 < n_isl_0)
                {
                    _S46 = lb_1 < n_isl_0;
                }
                else
                {
                    _S46 = false;
                }
                if(_S46)
                {
                    _S47 = (((*(&kernelContext_15)->g_plug_0)[la_1]) != 0U) != (((*(&kernelContext_15)->g_plug_0)[lb_1]) != 0U);
                }
                else
                {
                    _S47 = false;
                }
                if(_S47)
                {
                    uint _S107 = 9U * _S105.z;
                    acc_0.z = acc_0.z + (float4(*((&kernelContext_15)->sb_0+(_S107 + 5U))) ).x * (float4(*((&kernelContext_15)->sb_0+_S107)) ).w * (float4(*((&kernelContext_15)->sb_0+(_S107 + 4U))) ).z;
                }
                bj_0 = bj_0 + 64U;
            }
            float4 _S108 = group_sum_0(t_3, acc_0, &kernelContext_15);
            acc_0 = _S108;
            float m_plug_0 = _S108.x;
            float _S109 = float(n_isl_0);
            bool has_plug_0 = (_S108.y) != _S109;
            float capacity_0 = _S108.z;
            float _S110 = 1.0f / _S68;
            float4 _S111 = node_0(entry_n_0, 13U, &kernelContext_15);
            float modulus_0 = 1.0f / (_S110 + 1.0f / _S111.w);
            float m_red_0;
            if((_S73.w) != 0.0f)
            {
                m_red_0 = _S67;
            }
            else
            {
                m_red_0 = cl_mass_0 * _S67 / (cl_mass_0 + _S67);
            }
            float hertz_0 = hertz_duration_0(m_red_0, _S66, modulus_0, speed_1);
            float peak_0 = 3.14159274101257324f * (2.0f * m_red_0 * speed_1) / (2.0f * hertz_0);
            float ratio_0;
            if(has_plug_0)
            {
                float m_eff_0 = m_plug_0 * _S67 / (m_plug_0 + _S67);
                ratio_0 = m_eff_0 * speed_1 / hertz_duration_0(m_eff_0, _S66, modulus_0, speed_1) / max(capacity_0, 1.00000000317107685e-30f);
            }
            else
            {
                ratio_0 = 0.0f;
            }
            if(has_plug_0)
            {
                marked_0 = speed_1 > ((&kernelContext_15)->params_0->speed_0);
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
                    if(((*(&kernelContext_15)->g_plug_0)[j_5]) != 0U)
                    {
                        *((&kernelContext_15)->work_0+(24U * (_S95 + j_5) + 3U)) = float(at_1 - _S57 + 1U);
                    }
                    j_5 = j_5 + 64U;
                }
            }
            if(_S75)
            {
                uint _S112 = 6U * at_1;
                *((&kernelContext_15)->contacts_0+_S112) = packed_float4(float4((as_type<float>((k_2))), (as_type<float>((entry_n_0))), (as_type<float>((e_1))), t0_1)) ;
                *((&kernelContext_15)->contacts_0+(_S112 + 1U)) = packed_float4(float4(hit_0, speed_1)) ;
                *((&kernelContext_15)->contacts_0+(_S112 + 2U)) = packed_float4(float4(d_0, m_red_0)) ;
                *((&kernelContext_15)->contacts_0+(_S112 + 3U)) = packed_float4(float4(modulus_0, _S66, hertz_0, peak_0)) ;
                packed_float4 device* _S113 = (&kernelContext_15)->contacts_0+(_S112 + 4U);
                float _S114;
                if(has_plug_0)
                {
                    _S114 = m_plug_0;
                }
                else
                {
                    _S114 = 0.0f;
                }
                float _S115;
                if(marked_0)
                {
                    _S115 = 1.0f;
                }
                else
                {
                    _S115 = 0.0f;
                }
                *_S113 = packed_float4(float4(ratio_0, _S114, _S115, _S109)) ;
                *((&kernelContext_15)->contacts_0+(_S112 + 5U)) = packed_float4(float4(1.0f, 0.0f, 0.0f, 0.0f)) ;
            }
            float _S116 = max(vmax_1, speed_1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            vmax_1 = _S116;
            e_1 = e_1 + 1U;
            rows_2 = _S74;
        }
        rows_1 = rows_2;
        uint _S59 = k_2 + 1U;
        vmax_0 = vmax_1;
        k_2 = _S59;
    }
    if(t_3 == 0U)
    {
        *((&kernelContext_15)->calls_0+_S56) = packed_float4(float4((float4(*((&kernelContext_15)->calls_0+_S56)) ).x, (as_type<float>((rows_1))), 0.0f, vmax_0)) ;
    }
    return;
}

uint call_bfirst_0(uint g_6, KernelContext_0 thread* kernelContext_16)
{
    return (as_type<uint>(((float4(*(kernelContext_16->calls_0+(10U * g_6 + 4U))) ).x)));
}

uint call_nbonds_0(uint g_7, KernelContext_0 thread* kernelContext_17)
{
    return (as_type<uint>(((float4(*(kernelContext_17->calls_0+(10U * g_7 + 4U))) ).y)));
}

uint group_scan_0(uint t_4, uint v_3, KernelContext_0 thread* kernelContext_18)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    (*kernelContext_18->g_scan_0)[t_4] = v_3;
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
            add_0 = (*kernelContext_18->g_scan_0)[t_4 - step_0];
        }
        else
        {
            add_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*kernelContext_18->g_scan_0)[t_4] = (*kernelContext_18->g_scan_0)[t_4] + add_0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        step_0 = step_0 << 1U;
    }
    return (*kernelContext_18->g_scan_0)[t_4];
}

uint group_total_0(KernelContext_0 thread* kernelContext_19)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return (*kernelContext_19->g_scan_0)[63U];
}

void sort_work_0(uint t_5, uint base_0, uint n_3, KernelContext_0 thread* kernelContext_20)
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
    float _S117 = float_bits_0(2139095040U);
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
        *(kernelContext_20->work_0+(base_0 + j_6)) = _S117;
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
                    uint _S118 = base_0 + j_6;
                    float device* _S119 = kernelContext_20->work_0+_S118;
                    float a_3 = *_S119;
                    uint _S120 = base_0 + p_1;
                    float device* _S121 = kernelContext_20->work_0+_S120;
                    float b_1 = *_S121;
                    if(((*_S119) > (*_S121)) == ((j_6 & size_0) == 0U))
                    {
                        *(kernelContext_20->work_0+_S118) = b_1;
                        *(kernelContext_20->work_0+_S120) = a_3;
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

float median_work_0(uint base_1, uint n_4, KernelContext_0 thread* kernelContext_21)
{
    uint k_3 = n_4 / 2U;
    float _S122;
    if((n_4 % 2U) == 1U)
    {
        _S122 = *(kernelContext_21->work_0+(base_1 + k_3));
    }
    else
    {
        uint _S123 = base_1 + k_3;
        _S122 = 0.5f * (*(kernelContext_21->work_0+(_S123 - 1U)) + *(kernelContext_21->work_0+_S123));
    }
    return _S122;
}

[[kernel]] void feat_scales(uint3 thread_2 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_5 [[buffer(0)]], BondStatic_natural_0 device* bond_static_5 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_5 [[buffer(13)]], uint device* sinc_5 [[buffer(9)]], packed_uint4 device* slook_5 [[buffer(10)]], uint device* bmap_5 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_5 [[buffer(15)]], uint device* ctl_5 [[buffer(1)]], float device* work_5 [[buffer(18)]], packed_float4 device* nodes_5 [[buffer(3)]], packed_uint2 device* key_node_5 [[buffer(12)]], packed_float4 device* sc_5 [[buffer(7)]], packed_float4 device* bonds_5 [[buffer(4)]], packed_float4 device* sb_5 [[buffer(8)]], packed_uint4 device* bedge_5 [[buffer(22)]], packed_float4 device* calls_5 [[buffer(2)]], packed_float4 device* imps_5 [[buffer(16)]], packed_uint4 device* cisl_5 [[buffer(5)]], packed_float4 device* cislf_5 [[buffer(6)]], packed_float4 device* contacts_5 [[buffer(17)]], uint device* fctl_5 [[buffer(23)]], uint device* edges_5 [[buffer(20)]], uint device* cols_5 [[buffer(21)]], float device* feat_5 [[buffer(19)]])
{
    bool _S124;
    uint ok_0;
    thread KernelContext_0 kernelContext_22;
    (&kernelContext_22)->params_0 = params_5;
    (&kernelContext_22)->bond_static_0 = bond_static_5;
    (&kernelContext_22)->chunks_0 = chunks_5;
    (&kernelContext_22)->sinc_0 = sinc_5;
    (&kernelContext_22)->slook_0 = slook_5;
    (&kernelContext_22)->bmap_0 = bmap_5;
    (&kernelContext_22)->bond_dyn_0 = bond_dyn_5;
    (&kernelContext_22)->ctl_0 = ctl_5;
    (&kernelContext_22)->work_0 = work_5;
    (&kernelContext_22)->nodes_0 = nodes_5;
    (&kernelContext_22)->key_node_0 = key_node_5;
    (&kernelContext_22)->sc_0 = sc_5;
    (&kernelContext_22)->bonds_0 = bonds_5;
    (&kernelContext_22)->sb_0 = sb_5;
    (&kernelContext_22)->bedge_0 = bedge_5;
    (&kernelContext_22)->calls_0 = calls_5;
    (&kernelContext_22)->imps_0 = imps_5;
    (&kernelContext_22)->cisl_0 = cisl_5;
    (&kernelContext_22)->cislf_0 = cislf_5;
    (&kernelContext_22)->contacts_0 = contacts_5;
    (&kernelContext_22)->fctl_0 = fctl_5;
    (&kernelContext_22)->edges_0 = edges_5;
    (&kernelContext_22)->cols_0 = cols_5;
    (&kernelContext_22)->feat_0 = feat_5;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_5;
    (&kernelContext_22)->g_red_0 = &g_red_5;
    threadgroup array<float, int(64)> g_mv_5;
    (&kernelContext_22)->g_mv_0 = &g_mv_5;
    threadgroup array<uint, int(64)> g_mi_5;
    (&kernelContext_22)->g_mi_0 = &g_mi_5;
    threadgroup array<uint, int(4096)> g_plug_5;
    (&kernelContext_22)->g_plug_0 = &g_plug_5;
    threadgroup uint g_flag_5;
    (&kernelContext_22)->g_flag_0 = &g_flag_5;
    threadgroup array<uint, int(64)> g_scan_5;
    (&kernelContext_22)->g_scan_0 = &g_scan_5;
    threadgroup float4 g_scalars_5;
    (&kernelContext_22)->g_scalars_0 = &g_scalars_5;
    uint t_6 = thread_2.x;
    uint _S125 = group_uniform_0(ctl_5[int(1)], t_6);
    uint _S126 = 24U * params_5->max_nodes_0;
    uint g_8 = 0U;
    for(;;)
    {
        if(g_8 < _S125)
        {
        }
        else
        {
            break;
        }
        uint _S127 = call_first_0(g_8, &kernelContext_22);
        uint _S128 = call_chunks_0(g_8, &kernelContext_22);
        uint nc_0 = group_uniform_0(_S128, t_6);
        uint _S129 = call_bfirst_0(g_8, &kernelContext_22);
        uint _S130 = call_nbonds_0(g_8, &kernelContext_22);
        uint _S131 = group_uniform_0(_S130, t_6);
        uint _S132 = 10U * g_8;
        float L_0 = (float4(*((&kernelContext_22)->calls_0+_S132)) ).x;
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
            float4 _S133 = node_0(_S127 + j_7, 0U, &kernelContext_22);
            float m_2 = _S133.w;
            if(m_2 >= best_0)
            {
                best_0 = m_2;
                best_j_1 = j_7;
            }
            j_7 = j_7 + 64U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        (*(&kernelContext_22)->g_mv_0)[t_6] = best_0;
        (*(&kernelContext_22)->g_mi_0)[t_6] = best_j_1;
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
                uint _S134 = t_6 + s_3;
                if(((*(&kernelContext_22)->g_mv_0)[_S134]) > (*(&kernelContext_22)->g_mv_0)[t_6])
                {
                    _S124 = true;
                }
                else
                {
                    if(((*(&kernelContext_22)->g_mv_0)[_S134]) == (*(&kernelContext_22)->g_mv_0)[t_6])
                    {
                        _S124 = ((*(&kernelContext_22)->g_mi_0)[_S134]) > (*(&kernelContext_22)->g_mi_0)[t_6];
                    }
                    else
                    {
                        _S124 = false;
                    }
                }
            }
            else
            {
                _S124 = false;
            }
            if(_S124)
            {
                uint _S135 = t_6 + s_3;
                (*(&kernelContext_22)->g_mv_0)[t_6] = (*(&kernelContext_22)->g_mv_0)[_S135];
                (*(&kernelContext_22)->g_mi_0)[t_6] = (*(&kernelContext_22)->g_mi_0)[_S135];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            s_3 = s_3 >> 1U;
        }
        uint dom_0 = _S127 + (*(&kernelContext_22)->g_mi_0)[int(0)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 _S136 = node_0(dom_0, 12U, &kernelContext_22);
        uint _S137 = 6U * ((&kernelContext_22)->sinc_0[2U + (as_type<uint>((_S136.z)))] + (as_type<uint>((_S136.w))));
        float4 _S138 = float4(*((&kernelContext_22)->sc_0+_S137)) ;
        float4 _S139 = float4(*((&kernelContext_22)->sc_0+(_S137 + 1U))) ;
        float c_wave_0 = sqrt(_S138.y / _S138.z);
        thread float4 ga_1 = float4(0.0f) ;
        uint b0_0 = 0U;
        uint carry_0 = 0U;
        for(;;)
        {
            if(b0_0 < _S131)
            {
            }
            else
            {
                break;
            }
            uint j_8 = b0_0 + t_6;
            bool _S140 = j_8 < _S131;
            uint4 be_0;
            if(_S140)
            {
                be_0 = uint4(*((&kernelContext_22)->bedge_0+(_S129 + j_8))) ;
            }
            else
            {
                be_0 = uint4(4294967295U) ;
            }
            if(_S140)
            {
                _S124 = (be_0.x) != 4294967295U;
            }
            else
            {
                _S124 = false;
            }
            bool _S141;
            if(_S124)
            {
                _S141 = (be_0.y) != 4294967295U;
            }
            else
            {
                _S141 = false;
            }
            if(_S141)
            {
                ok_0 = 1U;
            }
            else
            {
                ok_0 = 0U;
            }
            uint _S142 = group_scan_0(t_6, ok_0, &kernelContext_22);
            uint at_2 = carry_0 + _S142 - ok_0;
            if(ok_0 != 0U)
            {
                uint _S143 = 9U * be_0.z;
                float area_0 = (float4(*((&kernelContext_22)->sb_0+_S143)) ).w;
                float gab_0 = (float4(*((&kernelContext_22)->sb_0+(_S143 + 6U))) ).y * area_0;
                *((&kernelContext_22)->work_0+(_S126 + at_2)) = (float4(*((&kernelContext_22)->sb_0+(_S143 + 5U))) ).x;
                *((&kernelContext_22)->work_0+(_S126 + 131072U + at_2)) = gab_0;
                *((&kernelContext_22)->work_0+(_S126 + 262144U + at_2)) = area_0;
                ga_1.x = ga_1.x + gab_0;
                if(((*((&kernelContext_22)->work_0+(24U * be_0.x + 3U))) != 0.0f) != ((*((&kernelContext_22)->work_0+(24U * be_0.y + 3U))) != 0.0f))
                {
                    ga_1.y = ga_1.y + gab_0;
                }
            }
            uint _S144 = group_total_0(&kernelContext_22);
            uint carry_1 = carry_0 + _S144;
            b0_0 = b0_0 + 64U;
            carry_0 = carry_1;
        }
        uint nlive_0 = group_uniform_0(carry_0, t_6);
        float4 _S145 = group_sum_0(t_6, ga_1, &kernelContext_22);
        ga_1 = _S145;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float _S146 = _S138.w;
        float _S147 = _S139.x * L_0 * L_0;
        float _S148 = L_0 * L_0;
        float f_t_0;
        float GA_0;
        float med_area_0;
        if(nlive_0 > 0U)
        {
            sort_work_0(t_6, _S126, nlive_0, &kernelContext_22);
            uint _S149 = _S126 + 131072U;
            sort_work_0(t_6, _S149, nlive_0, &kernelContext_22);
            uint _S150 = _S126 + 262144U;
            sort_work_0(t_6, _S150, nlive_0, &kernelContext_22);
            if(t_6 == 0U)
            {
                float _S151 = median_work_0(_S126, nlive_0, &kernelContext_22);
                float _S152 = median_work_0(_S149, nlive_0, &kernelContext_22);
                float _S153 = median_work_0(_S150, nlive_0, &kernelContext_22);
                *(&kernelContext_22)->g_scalars_0 = float4(_S151, _S152, _S153, 0.0f);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            float _S154 = (*(&kernelContext_22)->g_scalars_0).x;
            float _S155 = (*(&kernelContext_22)->g_scalars_0).y;
            float _S156 = (*(&kernelContext_22)->g_scalars_0).z;
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
            f_t_0 = _S154;
            GA_0 = _S155;
            med_area_0 = _S156;
        }
        else
        {
            f_t_0 = _S146;
            GA_0 = _S147;
            med_area_0 = _S148;
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
            float4 _S157 = node_0(_S127 + ok_0, 12U, &kernelContext_22);
            *((&kernelContext_22)->work_0+(_S126 + ok_0)) = (float4(*((&kernelContext_22)->sc_0+6U * ((&kernelContext_22)->sinc_0[2U + (as_type<uint>((_S157.z)))] + (as_type<uint>((_S157.w)))))) ).y;
            ok_0 = ok_0 + 64U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        sort_work_0(t_6, _S126, nc_0, &kernelContext_22);
        bool _S158 = t_6 == 0U;
        if(_S158)
        {
            float _S159 = median_work_0(_S126, nc_0, &kernelContext_22);
            (*(&kernelContext_22)->g_scalars_0).w = _S159;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        float e_med_0 = (*(&kernelContext_22)->g_scalars_0).w;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if(_S158)
        {
            float4 _S160 = float4(*((&kernelContext_22)->calls_0+_S132)) ;
            uint _S161 = _S132 + 1U;
            float4 _S162 = float4(*((&kernelContext_22)->calls_0+_S161)) ;
            *((&kernelContext_22)->calls_0+_S132) = packed_float4(float4(_S160.x, _S160.y, _S160.z, c_wave_0)) ;
            *((&kernelContext_22)->calls_0+_S161) = packed_float4(float4(f_t_0, e_med_0, GA_0, _S162.w)) ;
            *((&kernelContext_22)->calls_0+(_S132 + 9U)) = packed_float4(float4(med_area_0, ga_1.x, ga_1.y, 0.0f)) ;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        g_8 = g_8 + 1U;
    }
    return;
}

[[kernel]] void feat_summ(uint3 id_3 [[thread_position_in_grid]], FeatParams_0 constant* params_6 [[buffer(0)]], BondStatic_natural_0 device* bond_static_6 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_6 [[buffer(13)]], uint device* sinc_6 [[buffer(9)]], packed_uint4 device* slook_6 [[buffer(10)]], uint device* bmap_6 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_6 [[buffer(15)]], uint device* ctl_6 [[buffer(1)]], float device* work_6 [[buffer(18)]], packed_float4 device* nodes_6 [[buffer(3)]], packed_uint2 device* key_node_6 [[buffer(12)]], packed_float4 device* sc_6 [[buffer(7)]], packed_float4 device* bonds_6 [[buffer(4)]], packed_float4 device* sb_6 [[buffer(8)]], packed_uint4 device* bedge_6 [[buffer(22)]], packed_float4 device* calls_6 [[buffer(2)]], packed_float4 device* imps_6 [[buffer(16)]], packed_uint4 device* cisl_6 [[buffer(5)]], packed_float4 device* cislf_6 [[buffer(6)]], packed_float4 device* contacts_6 [[buffer(17)]], uint device* fctl_6 [[buffer(23)]], uint device* edges_6 [[buffer(20)]], uint device* cols_6 [[buffer(21)]], float device* feat_6 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_23;
    (&kernelContext_23)->params_0 = params_6;
    (&kernelContext_23)->bond_static_0 = bond_static_6;
    (&kernelContext_23)->chunks_0 = chunks_6;
    (&kernelContext_23)->sinc_0 = sinc_6;
    (&kernelContext_23)->slook_0 = slook_6;
    (&kernelContext_23)->bmap_0 = bmap_6;
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
    uint n_5 = id_3.x;
    bool _S163;
    if(n_5 >= ctl_6[int(2)])
    {
        _S163 = true;
    }
    else
    {
        float4 _S164 = node_0(n_5, 4U, &kernelContext_23);
        _S163 = (as_type<uint>((_S164.w))) != 0U;
    }
    if(_S163)
    {
        return;
    }
    float4 _S165 = node_0(n_5, 12U, &kernelContext_23);
    uint _S166 = (&kernelContext_23)->sinc_0[int(1)];
    uint off_0 = 2U + (&kernelContext_23)->sinc_0[int(0)];
    uint _S167 = off_0 + ((&kernelContext_23)->sinc_0[2U + (as_type<uint>((_S165.z)))] + (as_type<uint>((_S165.w))));
    uint a_4 = (&kernelContext_23)->sinc_0[_S167];
    uint _S168 = (&kernelContext_23)->sinc_0[_S167 + 1U];
    float dmax_0 = 0.0f;
    uint i_1 = a_4;
    float nlive_1 = 0.0f;
    float dsum_0 = 0.0f;
    float nbroken_0 = 0.0f;
    for(;;)
    {
        if(i_1 < _S168)
        {
        }
        else
        {
            break;
        }
        uint device* _S169 = (&kernelContext_23)->bmap_0+((&kernelContext_23)->params_0->layout_cap_0 + (&kernelContext_23)->sinc_0[off_0 + _S166 + 1U + i_1]);
        JointState_0 _S170 = ((&kernelContext_23)->bond_dyn_0+*_S169)->js_0;
        bool _S171 = ((float4(((&kernelContext_23)->bond_static_0+*_S169)->law_0.stiff1_0) ).w) != 0.0f;
        thread JointState_0 _S172 = ((&kernelContext_23)->bond_dyn_0+*_S169)->js_0;
        bool _S173 = connected_0(&_S172, _S171);
        if(_S173)
        {
            float nlive_2 = nlive_1 + 1.0f;
            float dsum_1 = dsum_0 + _S170.damage_0;
            dmax_0 = max(dmax_0, _S170.damage_0);
            nlive_1 = nlive_2;
            dsum_0 = dsum_1;
        }
        else
        {
            nbroken_0 = nbroken_0 + 1.0f;
        }
        i_1 = i_1 + 1U;
    }
    uint _S174 = 24U * n_5;
    *((&kernelContext_23)->work_0+(_S174 + 5U)) = dsum_0;
    *((&kernelContext_23)->work_0+(_S174 + 6U)) = dmax_0;
    *((&kernelContext_23)->work_0+(_S174 + 7U)) = nlive_1;
    *((&kernelContext_23)->work_0+(_S174 + 8U)) = nbroken_0;
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
    bool _S175;
    if(a_5 <= 0.0f)
    {
        _S175 = true;
    }
    else
    {
        _S175 = b_2 >= 0.0f;
    }
    if(_S175)
    {
        return -1.0f;
    }
    float disc_0 = b_2 * b_2 - a_5 * c_0;
    if(disc_0 < 0.0f)
    {
        return -1.0f;
    }
    float t_7 = (- b_2 - sqrt(disc_0)) / a_5;
    float _S176;
    if(t_7 <= dt_1)
    {
        _S176 = max(t_7, 0.0f);
    }
    else
    {
        _S176 = -1.0f;
    }
    return _S176;
}

[[kernel]] void feat_graph(uint3 group_1 [[threadgroup_position_in_grid]], uint3 thread_3 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_7 [[buffer(0)]], BondStatic_natural_0 device* bond_static_7 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_7 [[buffer(13)]], uint device* sinc_7 [[buffer(9)]], packed_uint4 device* slook_7 [[buffer(10)]], uint device* bmap_7 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_7 [[buffer(15)]], uint device* ctl_7 [[buffer(1)]], float device* work_7 [[buffer(18)]], packed_float4 device* nodes_7 [[buffer(3)]], packed_uint2 device* key_node_7 [[buffer(12)]], packed_float4 device* sc_7 [[buffer(7)]], packed_float4 device* bonds_7 [[buffer(4)]], packed_float4 device* sb_7 [[buffer(8)]], packed_uint4 device* bedge_7 [[buffer(22)]], packed_float4 device* calls_7 [[buffer(2)]], packed_float4 device* imps_7 [[buffer(16)]], packed_uint4 device* cisl_7 [[buffer(5)]], packed_float4 device* cislf_7 [[buffer(6)]], packed_float4 device* contacts_7 [[buffer(17)]], uint device* fctl_7 [[buffer(23)]], uint device* edges_7 [[buffer(20)]], uint device* cols_7 [[buffer(21)]], float device* feat_7 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_24;
    (&kernelContext_24)->params_0 = params_7;
    (&kernelContext_24)->bond_static_0 = bond_static_7;
    (&kernelContext_24)->chunks_0 = chunks_7;
    (&kernelContext_24)->sinc_0 = sinc_7;
    (&kernelContext_24)->slook_0 = slook_7;
    (&kernelContext_24)->bmap_0 = bmap_7;
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
    uint g_9 = group_1.x;
    uint t_8 = thread_3.x;
    uint j_9;
    if(g_9 >= ctl_7[int(1)])
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
    uint _S177 = call_first_0(g_9, &kernelContext_24);
    uint _S178 = call_chunks_0(g_9, &kernelContext_24);
    uint _S179 = group_uniform_0(_S178, t_8);
    uint _S180 = call_bfirst_0(g_9, &kernelContext_24);
    uint _S181 = call_nbonds_0(g_9, &kernelContext_24);
    uint _S182 = group_uniform_0(_S181, t_8);
    uint _S183 = 10U * g_9;
    uint _S184 = _S183 + 8U;
    uint _S185 = (as_type<uint>(((float4(*((&kernelContext_24)->calls_0+_S184)) ).x)));
    uint _S186 = group_uniform_0((as_type<uint>(((float4(*((&kernelContext_24)->calls_0+_S184)) ).y))), t_8);
    float _S187 = (float4(*((&kernelContext_24)->calls_0+_S183)) ).w;
    float _S188 = float_bits_0(2139095040U);
    float _S189 = (&kernelContext_24)->params_0->dt_0;
    j_9 = t_8;
    for(;;)
    {
        if(j_9 < _S179)
        {
        }
        else
        {
            break;
        }
        uint _S190 = 24U * (_S177 + j_9);
        *((&kernelContext_24)->work_0+_S190) = _S188;
        *((&kernelContext_24)->work_0+(_S190 + 1U)) = _S188;
        float device* _S191 = (&kernelContext_24)->work_0+(_S190 + 2U);
        float _S192;
        if((*((&kernelContext_24)->work_0+(_S190 + 14U))) != 0.0f)
        {
            _S192 = 0.0f;
        }
        else
        {
            _S192 = _S188;
        }
        *_S191 = _S192;
        j_9 = j_9 + 64U;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    bool _S193 = t_8 == 0U;
    uint iter_1;
    if(_S193)
    {
        iter_1 = 0U;
        for(;;)
        {
            if(iter_1 < _S186)
            {
            }
            else
            {
                break;
            }
            uint _S194 = 6U * (_S185 + iter_1);
            float4 _S195 = float4(*((&kernelContext_24)->contacts_0+_S194)) ;
            if(((float4(*((&kernelContext_24)->contacts_0+(_S194 + 5U))) ).x) == 0.0f)
            {
                iter_1 = iter_1 + 1U;
                continue;
            }
            uint en_0 = (as_type<uint>((_S195.y)));
            uint k_4 = (as_type<uint>((_S195.x)));
            uint _S196 = 24U * en_0;
            *((&kernelContext_24)->work_0+_S196) = 0.0f;
            uint _S197 = _S177 + _S179;
            uint _S198 = call_mask_0(g_9, &kernelContext_24);
            uint inode_1 = _S197 + ((uint)popcount((_S198 & ((1U << k_4) - 1U))));
            float r_imp_0 = (float4(*((&kernelContext_24)->imps_0+(4U * k_4 + 2U))) ).x;
            float4 _S199 = node_0(en_0, 13U, &kernelContext_24);
            float3 _S200 = _S199.xyz;
            float4 _S201 = node_0(inode_1, 13U, &kernelContext_24);
            float3 _S202 = _S200 - _S201.xyz;
            float4 _S203 = node_0(en_0, 3U, &kernelContext_24);
            float3 _S204 = _S203.xyz;
            float4 _S205 = node_0(inode_1, 3U, &kernelContext_24);
            float tc_0 = sphere_contact_time_0(_S202, _S204 - _S205.xyz, r_imp_0 + *((&kernelContext_24)->work_0+(_S196 + 12U)), _S189);
            if(tc_0 >= 0.0f)
            {
                uint _S206 = _S196 + 1U;
                *((&kernelContext_24)->work_0+_S206) = min(*((&kernelContext_24)->work_0+_S206), tc_0);
            }
            iter_1 = iter_1 + 1U;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
    iter_1 = 0U;
    for(;;)
    {
        if(iter_1 < (_S179 + 1U))
        {
        }
        else
        {
            break;
        }
        if(_S193)
        {
            *(&kernelContext_24)->g_flag_0 = 0U;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        j_9 = t_8;
        for(;;)
        {
            if(j_9 < _S182)
            {
            }
            else
            {
                break;
            }
            uint4 _S207 = uint4(*((&kernelContext_24)->bedge_0+(_S180 + j_9))) ;
            uint _S208 = _S207.x;
            bool _S209;
            if(_S208 == 4294967295U)
            {
                _S209 = true;
            }
            else
            {
                _S209 = (_S207.y) == 4294967295U;
            }
            if(_S209)
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
                bool _S210 = side_0 == 0U;
                uint u_0;
                if(_S210)
                {
                    u_0 = _S208;
                }
                else
                {
                    u_0 = _S207.y;
                }
                uint w_0;
                if(_S210)
                {
                    w_0 = _S207.y;
                }
                else
                {
                    w_0 = _S208;
                }
                uint _S211 = 24U * u_0;
                float hu_0 = *((&kernelContext_24)->work_0+_S211) + 1.0f;
                uint _S212 = 24U * w_0;
                if(hu_0 < (*((&kernelContext_24)->work_0+_S212)))
                {
                    *((&kernelContext_24)->work_0+_S212) = hu_0;
                    *(&kernelContext_24)->g_flag_0 = 1U;
                }
                float au_0 = *((&kernelContext_24)->work_0+(_S211 + 2U)) + 1.0f;
                uint _S213 = _S212 + 2U;
                if(au_0 < (*((&kernelContext_24)->work_0+_S213)))
                {
                    *((&kernelContext_24)->work_0+_S213) = au_0;
                    *(&kernelContext_24)->g_flag_0 = 1U;
                }
                float4 _S214 = node_0(u_0, 13U, &kernelContext_24);
                float3 _S215 = _S214.xyz;
                float4 _S216 = node_0(w_0, 13U, &kernelContext_24);
                float tu_0 = *((&kernelContext_24)->work_0+(_S211 + 1U)) + length(_S215 - _S216.xyz) / _S187;
                uint _S217 = _S212 + 1U;
                if(tu_0 < (*((&kernelContext_24)->work_0+_S217)))
                {
                    *((&kernelContext_24)->work_0+_S217) = tu_0;
                    *(&kernelContext_24)->g_flag_0 = 1U;
                }
                side_0 = side_0 + 1U;
            }
            j_9 = j_9 + 64U;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture | mem_flags::mem_threadgroup_imageblock);
        if((group_uniform_0(*(&kernelContext_24)->g_flag_0, t_8)) == 0U)
        {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        iter_1 = iter_1 + 1U;
    }
    return;
}

uint edge_base_0(uint g_10, uint s_4, KernelContext_0 thread* kernelContext_25)
{
    return *(kernelContext_25->fctl_0+(8U + 6U * g_10 + 3U + s_4));
}

matrix<float,int(3),int(3)>  quat_matrix_0(float4 q_2)
{
    float w_1 = q_2.w;
    float x_0 = q_2.x;
    float y_0 = q_2.y;
    float z_0 = q_2.z;
    float _S218 = y_0 * y_0;
    float _S219 = z_0 * z_0;
    float _S220 = x_0 * y_0;
    float _S221 = w_1 * z_0;
    float _S222 = x_0 * z_0;
    float _S223 = w_1 * y_0;
    float _S224 = x_0 * x_0;
    float _S225 = y_0 * z_0;
    float _S226 = w_1 * x_0;
    return matrix<float,int(3),int(3)> (float3(1.0f - 2.0f * (_S218 + _S219), 2.0f * (_S220 - _S221), 2.0f * (_S222 + _S223)), float3(2.0f * (_S220 + _S221), 1.0f - 2.0f * (_S224 + _S219), 2.0f * (_S225 - _S226)), float3(2.0f * (_S222 - _S223), 2.0f * (_S225 + _S226), 1.0f - 2.0f * (_S224 + _S218)));
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
    float3 l_2 = (((n_7) * (transpose(r_2))));
    return abs(l_2.x) * half_1.x + abs(l_2.y) * half_1.y + abs(l_2.z) * half_1.z;
}

float ln_ratio_0(float a_7, float b_3)
{
    return log(max(a_7, 9.99999997475242708e-07f) / max(b_3, 9.99999997475242708e-07f));
}

uint call_cbase_0(uint g_11, KernelContext_0 thread* kernelContext_26)
{
    return (as_type<uint>(((float4(*(kernelContext_26->calls_0+(10U * g_11 + 6U))) ).z)));
}

uint call_ibase_0(uint g_12, KernelContext_0 thread* kernelContext_27)
{
    return (as_type<uint>(((float4(*(kernelContext_27->calls_0+(10U * g_12 + 6U))) ).w)));
}

uint global_node_0(uint g_13, uint j_10, KernelContext_0 thread* kernelContext_28)
{
    uint _S227 = call_chunks_0(g_13, kernelContext_28);
    if(j_10 < _S227)
    {
        uint _S228 = call_cbase_0(g_13, kernelContext_28);
        return _S228 + j_10;
    }
    uint _S229 = *(kernelContext_28->fctl_0+int(4));
    uint _S230 = call_ibase_0(g_13, kernelContext_28);
    return _S229 + _S230 + (j_10 - _S227);
}

void put_edge_0(uint section_0, uint row_2, uint src_0, uint dst_0, KernelContext_0 thread* kernelContext_29)
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
            base_2 = 2U * kernelContext_29->params_0->caps_0.x;
        }
        else
        {
            base_2 = 2U * (kernelContext_29->params_0->caps_0.x + kernelContext_29->params_0->caps_0.y);
        }
    }
    uint _S231 = base_2 + 2U * row_2;
    *(kernelContext_29->edges_0+_S231) = src_0;
    *(kernelContext_29->edges_0+(_S231 + 1U)) = dst_0;
    return;
}

uint col_of_0(uint section_1, uint k_5, KernelContext_0 thread* kernelContext_30)
{
    return kernelContext_30->cols_0[section_1 * 65U + 1U + k_5];
}

uint stride_of_0(uint section_2, KernelContext_0 thread* kernelContext_31)
{
    return kernelContext_31->cols_0[section_2 * 65U];
}

void put_0(uint base_3, uint section_3, uint row_3, uint k_6, float v_4, KernelContext_0 thread* kernelContext_32)
{
    uint _S232 = col_of_0(section_3, k_6, kernelContext_32);
    if(_S232 != 4294967295U)
    {
        uint _S233 = stride_of_0(section_3, kernelContext_32);
        *(kernelContext_32->feat_0+(base_3 + row_3 * _S233 + _S232)) = v_4;
    }
    return;
}

void put_1(uint base_4, uint section_4, uint row_4, uint k_7, float v_5, KernelContext_0 thread* kernelContext_33)
{
    uint _S234 = col_of_0(section_4, k_7, kernelContext_33);
    if(_S234 != 4294967295U)
    {
        uint _S235 = stride_of_0(section_4, kernelContext_33);
        *(kernelContext_33->feat_0+(base_4 + row_4 * _S235 + _S234)) = v_5;
    }
    return;
}

void put_rot6_0(uint base_5, uint section_5, uint row_5, uint k0_0, float3 t1_2, float3 t2_1, matrix<float,int(3),int(3)>  body_1, KernelContext_0 thread* kernelContext_34)
{
    matrix<float,int(3),int(3)>  _S236 = transpose(body_1);
    float3 a_8 = (((t1_2) * (_S236)));
    float3 b_4 = (((t2_1) * (_S236)));
    put_1(base_5, section_5, row_5, k0_0, a_8.x, kernelContext_34);
    put_1(base_5, section_5, row_5, k0_0 + 1U, a_8.y, kernelContext_34);
    put_1(base_5, section_5, row_5, k0_0 + 2U, a_8.z, kernelContext_34);
    put_1(base_5, section_5, row_5, k0_0 + 3U, b_4.x, kernelContext_34);
    put_1(base_5, section_5, row_5, k0_0 + 4U, b_4.y, kernelContext_34);
    put_1(base_5, section_5, row_5, k0_0 + 5U, b_4.z, kernelContext_34);
    return;
}

void put_key_0(uint section_6, uint k_8, uint a_9, uint b_5, KernelContext_0 thread* kernelContext_35)
{
    uint _S237 = 4U * (kernelContext_35->params_0->caps_0.x + kernelContext_35->params_0->caps_0.y + kernelContext_35->params_0->caps_0.z);
    uint _S238;
    if(section_6 == 4U)
    {
        _S238 = 0U;
    }
    else
    {
        if(section_6 == 5U)
        {
            _S238 = kernelContext_35->params_0->caps_0.x;
        }
        else
        {
            _S238 = kernelContext_35->params_0->caps_0.x + kernelContext_35->params_0->caps_0.y;
        }
    }
    uint _S239 = _S237 + _S238 + 2U * k_8;
    *(kernelContext_35->edges_0+_S239) = a_9;
    *(kernelContext_35->edges_0+(_S239 + 1U)) = b_5;
    return;
}

void edge_count_0(uint g_14, uint s_5, uint n_8, KernelContext_0 thread* kernelContext_36)
{
    *(kernelContext_36->fctl_0+(8U + 6U * g_14 + s_5)) = n_8;
    return;
}

[[kernel]] void feat_ic(uint3 group_2 [[threadgroup_position_in_grid]], uint3 thread_4 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_8 [[buffer(0)]], BondStatic_natural_0 device* bond_static_8 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_8 [[buffer(13)]], uint device* sinc_8 [[buffer(9)]], packed_uint4 device* slook_8 [[buffer(10)]], uint device* bmap_8 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_8 [[buffer(15)]], uint device* ctl_8 [[buffer(1)]], float device* work_8 [[buffer(18)]], packed_float4 device* nodes_8 [[buffer(3)]], packed_uint2 device* key_node_8 [[buffer(12)]], packed_float4 device* sc_8 [[buffer(7)]], packed_float4 device* bonds_8 [[buffer(4)]], packed_float4 device* sb_8 [[buffer(8)]], packed_uint4 device* bedge_8 [[buffer(22)]], packed_float4 device* calls_8 [[buffer(2)]], packed_float4 device* imps_8 [[buffer(16)]], packed_uint4 device* cisl_8 [[buffer(5)]], packed_float4 device* cislf_8 [[buffer(6)]], packed_float4 device* contacts_8 [[buffer(17)]], uint device* fctl_8 [[buffer(23)]], uint device* edges_8 [[buffer(20)]], uint device* cols_8 [[buffer(21)]], float device* feat_8 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_37;
    (&kernelContext_37)->params_0 = params_8;
    (&kernelContext_37)->bond_static_0 = bond_static_8;
    (&kernelContext_37)->chunks_0 = chunks_8;
    (&kernelContext_37)->sinc_0 = sinc_8;
    (&kernelContext_37)->slook_0 = slook_8;
    (&kernelContext_37)->bmap_0 = bmap_8;
    (&kernelContext_37)->bond_dyn_0 = bond_dyn_8;
    (&kernelContext_37)->ctl_0 = ctl_8;
    (&kernelContext_37)->work_0 = work_8;
    (&kernelContext_37)->nodes_0 = nodes_8;
    (&kernelContext_37)->key_node_0 = key_node_8;
    (&kernelContext_37)->sc_0 = sc_8;
    (&kernelContext_37)->bonds_0 = bonds_8;
    (&kernelContext_37)->sb_0 = sb_8;
    (&kernelContext_37)->bedge_0 = bedge_8;
    (&kernelContext_37)->calls_0 = calls_8;
    (&kernelContext_37)->imps_0 = imps_8;
    (&kernelContext_37)->cisl_0 = cisl_8;
    (&kernelContext_37)->cislf_0 = cislf_8;
    (&kernelContext_37)->contacts_0 = contacts_8;
    (&kernelContext_37)->fctl_0 = fctl_8;
    (&kernelContext_37)->edges_0 = edges_8;
    (&kernelContext_37)->cols_0 = cols_8;
    (&kernelContext_37)->feat_0 = feat_8;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_8;
    (&kernelContext_37)->g_red_0 = &g_red_8;
    threadgroup array<float, int(64)> g_mv_8;
    (&kernelContext_37)->g_mv_0 = &g_mv_8;
    threadgroup array<uint, int(64)> g_mi_8;
    (&kernelContext_37)->g_mi_0 = &g_mi_8;
    threadgroup array<uint, int(4096)> g_plug_8;
    (&kernelContext_37)->g_plug_0 = &g_plug_8;
    threadgroup uint g_flag_8;
    (&kernelContext_37)->g_flag_0 = &g_flag_8;
    threadgroup array<uint, int(64)> g_scan_8;
    (&kernelContext_37)->g_scan_0 = &g_scan_8;
    threadgroup float4 g_scalars_8;
    (&kernelContext_37)->g_scalars_0 = &g_scalars_8;
    uint g_15 = group_2.x;
    uint t_10 = thread_4.x;
    bool _S240;
    if(g_15 >= ctl_8[int(1)])
    {
        _S240 = true;
    }
    else
    {
        _S240 = t_10 != 0U;
    }
    if(_S240)
    {
        return;
    }
    uint _S241 = call_first_0(g_15, &kernelContext_37);
    uint _S242 = call_chunks_0(g_15, &kernelContext_37);
    uint _S243 = 10U * g_15;
    uint _S244 = _S243 + 8U;
    uint _S245 = (as_type<uint>(((float4(*((&kernelContext_37)->calls_0+_S244)) ).x)));
    uint _S246 = (as_type<uint>(((float4(*((&kernelContext_37)->calls_0+_S244)) ).y)));
    float4 _S247 = float4(*((&kernelContext_37)->calls_0+_S243)) ;
    float _S248 = _S247.x;
    float _S249 = _S247.z;
    float _S250 = _S247.w;
    float _S251 = (float4(*((&kernelContext_37)->calls_0+(_S243 + 1U))) ).x;
    uint _S252;
    if(((&kernelContext_37)->params_0->count_only_0) != 0U)
    {
        _S252 = 0U;
    }
    else
    {
        uint _S253 = edge_base_0(g_15, 1U, &kernelContext_37);
        _S252 = _S253;
    }
    uint _S254 = (&kernelContext_37)->params_0->bases2_0.y;
    uint _S255 = _S243 + 9U;
    float _S256 = (float4(*((&kernelContext_37)->calls_0+_S255)) ).x;
    uint r_3 = 0U;
    uint count_0 = 0U;
    for(;;)
    {
        if(r_3 < _S246)
        {
        }
        else
        {
            break;
        }
        uint _S257 = 6U * (_S245 + r_3);
        if(((float4(*((&kernelContext_37)->contacts_0+(_S257 + 5U))) ).x) == 0.0f)
        {
            r_3 = r_3 + 1U;
            continue;
        }
        if(((&kernelContext_37)->params_0->count_only_0) == 0U)
        {
            float4 _S258 = float4(*((&kernelContext_37)->contacts_0+_S257)) ;
            float4 _S259 = float4(*((&kernelContext_37)->contacts_0+(_S257 + 1U))) ;
            float4 _S260 = float4(*((&kernelContext_37)->contacts_0+(_S257 + 2U))) ;
            float4 _S261 = float4(*((&kernelContext_37)->contacts_0+(_S257 + 3U))) ;
            uint k_9 = (as_type<uint>((_S258.x)));
            uint en_1 = (as_type<uint>((_S258.y)));
            uint _S262 = _S241 + _S242;
            uint _S263 = call_mask_0(g_15, &kernelContext_37);
            uint inode_2 = _S262 + ((uint)popcount((_S263 & ((1U << k_9) - 1U))));
            float4 _S264 = node_0(inode_2, 1U, &kernelContext_37);
            matrix<float,int(3),int(3)>  ri_0 = quat_matrix_0(_S264);
            float4 _S265 = node_0(en_1, 1U, &kernelContext_37);
            matrix<float,int(3),int(3)>  rb_1 = quat_matrix_0(_S265);
            float3 nrm_0 = normalize(_S260.xyz);
            float3 t1_3 = tangent_of_0(nrm_0, ri_0);
            float3 _S266 = cross(nrm_0, t1_3);
            uint _S267 = 24U * en_1;
            uint _S268 = _S267 + 9U;
            float3 half_c_0 = float3(*((&kernelContext_37)->work_0+_S268), *((&kernelContext_37)->work_0+(_S268 + 1U)), *((&kernelContext_37)->work_0+(_S268 + 2U)));
            uint _S269 = 4U * k_9;
            uint _S270 = _S269 + 2U;
            float hn_0 = (float4(*((&kernelContext_37)->imps_0+_S270)) ).x;
            float3 half_i_0 = float3((float4(*((&kernelContext_37)->imps_0+_S270)) ).y, (float4(*((&kernelContext_37)->imps_0+_S270)) ).z, (float4(*((&kernelContext_37)->imps_0+_S270)) ).w);
            float4 _S271 = node_0(en_1, 13U, &kernelContext_37);
            float3 _S272 = _S271.xyz;
            float4 _S273 = node_0(inode_2, 13U, &kernelContext_37);
            float _S274 = dot(_S272 - _S273.xyz, nrm_0) - support_of_0(ri_0, half_i_0, nrm_0) - support_of_0(rb_1, half_c_0, nrm_0);
            float _S275 = _S259.w;
            float ek_0 = 0.5f * _S260.w * _S275 * _S275;
            uint plug_row_0 = uint(*((&kernelContext_37)->work_0+(_S267 + 3U)));
            float ga_plug_0 = (float4(*((&kernelContext_37)->calls_0+_S255)) ).z;
            if(ga_plug_0 > 0.0f)
            {
                _S240 = plug_row_0 != 0U;
            }
            else
            {
                _S240 = false;
            }
            float _S276;
            if(_S240)
            {
                _S276 = ln_ratio_0(ek_0, ga_plug_0);
            }
            else
            {
                _S276 = 0.0f;
            }
            uint _S277 = _S269 + 3U;
            float fr_imp_0 = (float4(*((&kernelContext_37)->imps_0+_S277)) ).y;
            float fr_ch_0 = *((&kernelContext_37)->work_0+(_S267 + 13U));
            float _S278;
            if(((&kernelContext_37)->params_0->contact_friction_0) >= 0.0f)
            {
                _S278 = (&kernelContext_37)->params_0->contact_friction_0;
            }
            else
            {
                _S278 = min(fr_imp_0, fr_ch_0);
            }
            float _S279 = (float4(*((&kernelContext_37)->imps_0+_S277)) ).z;
            float4 _S280 = node_0(en_1, 13U, &kernelContext_37);
            float3 _S281 = _S280.xyz;
            float4 _S282 = node_0(inode_2, 13U, &kernelContext_37);
            float3 _S283 = _S281 - _S282.xyz;
            float4 _S284 = node_0(en_1, 3U, &kernelContext_37);
            float3 _S285 = _S284.xyz;
            float4 _S286 = node_0(inode_2, 3U, &kernelContext_37);
            float tc_1 = sphere_contact_time_0(_S283, _S285 - _S286.xyz, hn_0 + *((&kernelContext_37)->work_0+(_S267 + 12U)), (&kernelContext_37)->params_0->dt_0);
            float _S287;
            if(tc_1 < 0.0f)
            {
                _S287 = 1.0f;
            }
            else
            {
                if(tc_1 <= 0.0f)
                {
                    _S287 = -1.0f;
                }
                else
                {
                    _S287 = tc_1 / (&kernelContext_37)->params_0->dt_0;
                }
            }
            float _S288 = (float4(*((&kernelContext_37)->imps_0+_S277)) ).x;
            float _S289 = *((&kernelContext_37)->work_0+(_S267 + 15U));
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
                uint row_6 = _S252 + count_0 + side_1;
                bool _S290 = side_1 == 0U;
                uint sn_0;
                if(_S290)
                {
                    sn_0 = inode_2;
                }
                else
                {
                    sn_0 = en_1;
                }
                uint dn_0;
                if(_S290)
                {
                    dn_0 = en_1;
                }
                else
                {
                    dn_0 = inode_2;
                }
                float3 ft1_0;
                if(_S290)
                {
                    ft1_0 = t1_3;
                }
                else
                {
                    ft1_0 = - t1_3;
                }
                float3 fn_0;
                if(_S290)
                {
                    fn_0 = nrm_0;
                }
                else
                {
                    fn_0 = - nrm_0;
                }
                float4 _S291 = node_0(sn_0, 3U, &kernelContext_37);
                float3 _S292 = _S291.xyz;
                float4 _S293 = node_0(dn_0, 3U, &kernelContext_37);
                float3 rel_0 = _S292 - _S293.xyz;
                float _S294 = dot(ft1_0, rel_0);
                float _S295 = dot(_S266, rel_0);
                float _S296 = dot(fn_0, rel_0);
                float4 _S297 = node_0(inode_2, 4U, &kernelContext_37);
                float3 wi_0 = _S297.xyz * float3((_S279 / _S249)) ;
                float _S298 = dot(ft1_0, wi_0);
                float _S299 = dot(_S266, wi_0);
                float _S300 = dot(fn_0, wi_0);
                uint _S301 = global_node_0(g_15, sn_0 - _S241, &kernelContext_37);
                uint _S302 = global_node_0(g_15, dn_0 - _S241, &kernelContext_37);
                put_edge_0(5U, row_6, _S301, _S302, &kernelContext_37);
                put_0(_S254, 5U, row_6, 0U, _S296 / _S249, &kernelContext_37);
                put_0(_S254, 5U, row_6, 1U, _S294 / _S249, &kernelContext_37);
                put_0(_S254, 5U, row_6, 2U, _S295 / _S249, &kernelContext_37);
                put_0(_S254, 5U, row_6, 3U, _S296 / _S250, &kernelContext_37);
                float4 _S303 = node_0(dn_0, 0U, &kernelContext_37);
                float _S304 = _S303.w;
                float4 _S305 = node_0(sn_0, 0U, &kernelContext_37);
                put_0(_S254, 5U, row_6, 4U, ln_ratio_0(_S304, _S305.w), &kernelContext_37);
                put_0(_S254, 5U, row_6, 5U, ln_ratio_0(_S279, _S248), &kernelContext_37);
                put_0(_S254, 5U, row_6, 6U, ln_ratio_0(_S288, _S289), &kernelContext_37);
                put_0(_S254, 5U, row_6, 7U, _S298, &kernelContext_37);
                put_0(_S254, 5U, row_6, 8U, _S299, &kernelContext_37);
                put_0(_S254, 5U, row_6, 9U, _S300, &kernelContext_37);
                put_0(_S254, 5U, row_6, 10U, _S287, &kernelContext_37);
                put_0(_S254, 5U, row_6, 11U, _S276, &kernelContext_37);
                put_0(_S254, 5U, row_6, 12U, ln_ratio_0(_S261.w, _S251 * _S256), &kernelContext_37);
                put_0(_S254, 5U, row_6, 13U, _S261.z / (&kernelContext_37)->params_0->dt_0, &kernelContext_37);
                float _S306;
                if(_S290)
                {
                    _S306 = 1.0f;
                }
                else
                {
                    _S306 = 0.0f;
                }
                put_0(_S254, 5U, row_6, 14U, _S306, &kernelContext_37);
                put_0(_S254, 5U, row_6, 15U, max(- _S274, 0.0f) / _S248, &kernelContext_37);
                put_0(_S254, 5U, row_6, 16U, _S278, &kernelContext_37);
                put_0(_S254, 5U, row_6, 17U, (&kernelContext_37)->params_0->restitution_0, &kernelContext_37);
                float4 _S307 = node_0(sn_0, 1U, &kernelContext_37);
                put_rot6_0(_S254, 5U, row_6, 18U, ft1_0, _S266, quat_matrix_0(_S307), &kernelContext_37);
                float4 _S308 = node_0(dn_0, 1U, &kernelContext_37);
                put_rot6_0(_S254, 5U, row_6, 24U, ft1_0, _S266, quat_matrix_0(_S308), &kernelContext_37);
                *((&kernelContext_37)->work_0+(24U * sn_0 + 4U)) = 1.0f;
                side_1 = side_1 + 1U;
            }
            uint _S309 = (_S252 + count_0) / 2U;
            float4 _S310 = node_0(en_1, 12U, &kernelContext_37);
            uint _S311 = (as_type<uint>((_S310.z)));
            float4 _S312 = node_0(en_1, 12U, &kernelContext_37);
            put_key_0(5U, _S309, _S311, (as_type<uint>((_S312.w))) | (k_9 << 24U), &kernelContext_37);
        }
        count_0 = count_0 + 2U;
        r_3 = r_3 + 1U;
    }
    if(((&kernelContext_37)->params_0->count_only_0) != 0U)
    {
        edge_count_0(g_15, 1U, count_0, &kernelContext_37);
    }
    return;
}

[[kernel]] void feat_cracked_find(uint3 group_3 [[threadgroup_position_in_grid]], uint3 thread_5 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_9 [[buffer(0)]], BondStatic_natural_0 device* bond_static_9 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_9 [[buffer(13)]], uint device* sinc_9 [[buffer(9)]], packed_uint4 device* slook_9 [[buffer(10)]], uint device* bmap_9 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_9 [[buffer(15)]], uint device* ctl_9 [[buffer(1)]], float device* work_9 [[buffer(18)]], packed_float4 device* nodes_9 [[buffer(3)]], packed_uint2 device* key_node_9 [[buffer(12)]], packed_float4 device* sc_9 [[buffer(7)]], packed_float4 device* bonds_9 [[buffer(4)]], packed_float4 device* sb_9 [[buffer(8)]], packed_uint4 device* bedge_9 [[buffer(22)]], packed_float4 device* calls_9 [[buffer(2)]], packed_float4 device* imps_9 [[buffer(16)]], packed_uint4 device* cisl_9 [[buffer(5)]], packed_float4 device* cislf_9 [[buffer(6)]], packed_float4 device* contacts_9 [[buffer(17)]], uint device* fctl_9 [[buffer(23)]], uint device* edges_9 [[buffer(20)]], uint device* cols_9 [[buffer(21)]], float device* feat_9 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_38;
    (&kernelContext_38)->params_0 = params_9;
    (&kernelContext_38)->bond_static_0 = bond_static_9;
    (&kernelContext_38)->chunks_0 = chunks_9;
    (&kernelContext_38)->sinc_0 = sinc_9;
    (&kernelContext_38)->slook_0 = slook_9;
    (&kernelContext_38)->bmap_0 = bmap_9;
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
    uint g_16 = group_3.x;
    uint t_11 = thread_5.x;
    bool _S313;
    if(g_16 >= ctl_9[int(1)])
    {
        _S313 = true;
    }
    else
    {
        _S313 = t_11 != 0U;
    }
    if(_S313)
    {
        return;
    }
    uint _S314 = call_first_0(g_16, &kernelContext_38);
    uint _S315 = call_chunks_0(g_16, &kernelContext_38);
    uint _S316 = (&kernelContext_38)->sinc_0[int(1)];
    uint _S317 = 2U + (&kernelContext_38)->sinc_0[int(0)];
    uint slots_0 = (&kernelContext_38)->params_0->max_bonds_cap_0 + 8U * _S314;
    uint cap_0 = 8U * _S315 - 1U;
    uint j_11 = 0U;
    uint count_1 = 0U;
    for(;;)
    {
        if(j_11 < _S315)
        {
        }
        else
        {
            break;
        }
        uint na_0 = _S314 + j_11;
        float4 _S318 = node_0(na_0, 12U, &kernelContext_38);
        uint _S319 = (&kernelContext_38)->sinc_0[2U + (as_type<uint>((_S318.z)))];
        float4 _S320 = node_0(na_0, 12U, &kernelContext_38);
        uint rowa_0 = _S319 + (as_type<uint>((_S320.w)));
        uint _S321 = _S317 + rowa_0;
        uint i_2 = (&kernelContext_38)->sinc_0[_S321];
        uint count_2 = count_1;
        for(;;)
        {
            if(i_2 < ((&kernelContext_38)->sinc_0)[_S321 + 1U])
            {
            }
            else
            {
                break;
            }
            uint br_0 = (&kernelContext_38)->sinc_0[_S317 + _S316 + 1U + i_2];
            uint4 ends_1 = (as_type<uint4>((float4(*((&kernelContext_38)->sb_0+(9U * br_0 + 7U))) )));
            if((ends_1.x) != rowa_0)
            {
                i_2 = i_2 + 1U;
                continue;
            }
            uint _S322 = node_of_key_0(ends_1.y, &kernelContext_38);
            if(_S322 == 4294967295U)
            {
                _S313 = true;
            }
            else
            {
                _S313 = _S322 < _S314;
            }
            bool _S323;
            if(_S313)
            {
                _S323 = true;
            }
            else
            {
                _S323 = _S322 >= (_S314 + _S315);
            }
            if(_S323)
            {
                i_2 = i_2 + 1U;
                continue;
            }
            uint device* _S324 = (&kernelContext_38)->bmap_0+((&kernelContext_38)->params_0->layout_cap_0 + br_0);
            bool _S325 = ((float4(((&kernelContext_38)->bond_static_0+*_S324)->law_0.stiff1_0) ).w) != 0.0f;
            thread JointState_0 _S326 = ((&kernelContext_38)->bond_dyn_0+*_S324)->js_0;
            bool _S327 = connected_0(&_S326, _S325);
            if(_S327)
            {
                i_2 = i_2 + 1U;
                continue;
            }
            if(count_2 < cap_0)
            {
                *((&kernelContext_38)->bedge_0+(slots_0 + 1U + count_2)) = packed_uint4(uint4(br_0, na_0, _S322, *((&kernelContext_38)->bmap_0+((&kernelContext_38)->params_0->layout_cap_0 + (&kernelContext_38)->params_0->rows_0 + br_0)))) ;
            }
            count_2 = count_2 + 1U;
            i_2 = i_2 + 1U;
        }
        j_11 = j_11 + 1U;
        count_1 = count_2;
    }
    packed_uint4 device* _S328 = (&kernelContext_38)->bedge_0+slots_0;
    uint _S329 = min(count_1, cap_0);
    if(count_1 > cap_0)
    {
        j_11 = 1U;
    }
    else
    {
        j_11 = 0U;
    }
    *_S328 = packed_uint4(uint4(_S329, j_11, 0U, 0U)) ;
    return;
}

[[kernel]] void feat_cracked(uint3 group_4 [[threadgroup_position_in_grid]], uint3 thread_6 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_10 [[buffer(0)]], BondStatic_natural_0 device* bond_static_10 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_10 [[buffer(13)]], uint device* sinc_10 [[buffer(9)]], packed_uint4 device* slook_10 [[buffer(10)]], uint device* bmap_10 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_10 [[buffer(15)]], uint device* ctl_10 [[buffer(1)]], float device* work_10 [[buffer(18)]], packed_float4 device* nodes_10 [[buffer(3)]], packed_uint2 device* key_node_10 [[buffer(12)]], packed_float4 device* sc_10 [[buffer(7)]], packed_float4 device* bonds_10 [[buffer(4)]], packed_float4 device* sb_10 [[buffer(8)]], packed_uint4 device* bedge_10 [[buffer(22)]], packed_float4 device* calls_10 [[buffer(2)]], packed_float4 device* imps_10 [[buffer(16)]], packed_uint4 device* cisl_10 [[buffer(5)]], packed_float4 device* cislf_10 [[buffer(6)]], packed_float4 device* contacts_10 [[buffer(17)]], uint device* fctl_10 [[buffer(23)]], uint device* edges_10 [[buffer(20)]], uint device* cols_10 [[buffer(21)]], float device* feat_10 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_39;
    (&kernelContext_39)->params_0 = params_10;
    (&kernelContext_39)->bond_static_0 = bond_static_10;
    (&kernelContext_39)->chunks_0 = chunks_10;
    (&kernelContext_39)->sinc_0 = sinc_10;
    (&kernelContext_39)->slook_0 = slook_10;
    (&kernelContext_39)->bmap_0 = bmap_10;
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
    uint g_17 = group_4.x;
    uint t_12 = thread_6.x;
    bool _S330;
    if(g_17 >= ctl_10[int(1)])
    {
        _S330 = true;
    }
    else
    {
        _S330 = t_12 != 0U;
    }
    if(_S330)
    {
        return;
    }
    uint _S331 = call_first_0(g_17, &kernelContext_39);
    float4 _S332 = float4(*((&kernelContext_39)->calls_0+10U * g_17)) ;
    float _S333 = _S332.x;
    float _S334 = _S332.z;
    uint _S335;
    if(((&kernelContext_39)->params_0->count_only_0) != 0U)
    {
        _S335 = 0U;
    }
    else
    {
        uint _S336 = edge_base_0(g_17, 0U, &kernelContext_39);
        _S335 = _S336;
    }
    uint _S337 = (&kernelContext_39)->params_0->bases2_0.x;
    uint slots_1 = (&kernelContext_39)->params_0->max_bonds_cap_0 + 8U * _S331;
    uint _S338 = (uint4(*((&kernelContext_39)->bedge_0+slots_1)) ).x;
    uint ci_0 = 0U;
    uint count_3 = 0U;
    for(;;)
    {
        if(ci_0 < _S338)
        {
        }
        else
        {
            break;
        }
        uint4 _S339 = uint4(*((&kernelContext_39)->bedge_0+(slots_1 + 1U + ci_0))) ;
        uint br_1 = _S339.x;
        uint na_1 = _S339.y;
        uint nb_0 = _S339.z;
        uint stamp_0 = _S339.w;
        float4 _S340 = node_0(na_1, 1U, &kernelContext_39);
        matrix<float,int(3),int(3)>  ra_1 = quat_matrix_0(_S340);
        float4 _S341 = node_0(nb_0, 1U, &kernelContext_39);
        matrix<float,int(3),int(3)>  rbm_0 = quat_matrix_0(_S341);
        uint _S342 = 24U * na_1;
        uint _S343 = _S342 + 17U;
        matrix<float,int(3),int(3)>  rs_0 = (((transpose(quat_matrix_0(float4(*((&kernelContext_39)->work_0+_S343), *((&kernelContext_39)->work_0+(_S343 + 1U)), *((&kernelContext_39)->work_0+(_S343 + 2U)), *((&kernelContext_39)->work_0+(_S343 + 3U)))))) * (ra_1)));
        uint _S344 = 9U * br_1;
        float4 _S345 = float4(*((&kernelContext_39)->sb_0+_S344)) ;
        uint4 ends_2 = (as_type<uint4>((float4(*((&kernelContext_39)->sb_0+(_S344 + 7U))) )));
        float3 nrm_1 = ((((float4(*((&kernelContext_39)->sb_0+(_S344 + 1U))) ).xyz) * (rs_0)));
        float3 t1_4 = ((((float4(*((&kernelContext_39)->sb_0+(_S344 + 2U))) ).xyz) * (rs_0)));
        float4 _S346 = node_0(na_1, 13U, &kernelContext_39);
        float3 xa_0 = _S346.xyz;
        float4 _S347 = node_0(nb_0, 13U, &kernelContext_39);
        float3 _S348 = _S347.xyz - xa_0;
        float3 nrm_2;
        float3 t1_5;
        if((dot(nrm_1, _S348)) < 0.0f)
        {
            float3 _S349 = - t1_4;
            nrm_2 = - nrm_1;
            t1_5 = _S349;
        }
        else
        {
            nrm_2 = nrm_1;
            t1_5 = t1_4;
        }
        uint _S350 = _S342 + 9U;
        float3 ha_0 = float3(*((&kernelContext_39)->work_0+_S350), *((&kernelContext_39)->work_0+(_S350 + 1U)), *((&kernelContext_39)->work_0+(_S350 + 2U)));
        uint _S351 = 24U * nb_0;
        uint _S352 = _S351 + 9U;
        float3 hb_0 = float3(*((&kernelContext_39)->work_0+_S352), *((&kernelContext_39)->work_0+(_S352 + 1U)), *((&kernelContext_39)->work_0+(_S352 + 2U)));
        float gap_0 = dot(_S348, nrm_2) - support_of_0(ra_1, ha_0, nrm_2) - support_of_0(rbm_0, hb_0, nrm_2);
        float3 t2_2 = cross(nrm_2, t1_5);
        bool across1_0 = (abs(dot(_S348, t1_5))) <= (support_of_0(ra_1, ha_0, t1_5) + support_of_0(rbm_0, hb_0, t1_5));
        bool across2_0 = (abs(dot(_S348, t2_2))) <= (support_of_0(ra_1, ha_0, t2_2) + support_of_0(rbm_0, hb_0, t2_2));
        if(gap_0 > (0.25f * _S333))
        {
            _S330 = true;
        }
        else
        {
            _S330 = !across1_0;
        }
        bool _S353;
        if(_S330)
        {
            _S353 = true;
        }
        else
        {
            _S353 = !across2_0;
        }
        if(_S353)
        {
            ci_0 = ci_0 + 1U;
            continue;
        }
        if(((&kernelContext_39)->params_0->count_only_0) == 0U)
        {
            uint _S354 = _S342 + 21U;
            float3 _S355 = xa_0 + (((_S345.xyz - float3(*((&kernelContext_39)->work_0+_S354), *((&kernelContext_39)->work_0+(_S354 + 1U)), *((&kernelContext_39)->work_0+(_S354 + 2U)))) * (rs_0)));
            float _S356;
            if(stamp_0 == 4294967295U)
            {
                _S356 = 1.0f;
            }
            else
            {
                _S356 = min(float((&kernelContext_39)->params_0->frame_0 - stamp_0) / 8.0f, 1.0f);
            }
            float fa_0 = *((&kernelContext_39)->work_0+(_S342 + 13U));
            float fbr_0 = *((&kernelContext_39)->work_0+(_S351 + 13U));
            float _S357;
            if(((&kernelContext_39)->params_0->contact_friction_0) >= 0.0f)
            {
                _S357 = (&kernelContext_39)->params_0->contact_friction_0;
            }
            else
            {
                _S357 = min(fa_0, fbr_0);
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
                uint row_7 = _S335 + count_3 + side_2;
                bool _S358 = side_2 == 0U;
                uint sn_1;
                if(_S358)
                {
                    sn_1 = na_1;
                }
                else
                {
                    sn_1 = nb_0;
                }
                uint dn_1;
                if(_S358)
                {
                    dn_1 = nb_0;
                }
                else
                {
                    dn_1 = na_1;
                }
                float3 ft1_1;
                if(_S358)
                {
                    ft1_1 = t1_5;
                }
                else
                {
                    ft1_1 = - t1_5;
                }
                float3 fn_1;
                if(_S358)
                {
                    fn_1 = nrm_2;
                }
                else
                {
                    fn_1 = - nrm_2;
                }
                float4 _S359 = node_0(sn_1, 3U, &kernelContext_39);
                float3 _S360 = _S359.xyz;
                float4 _S361 = node_0(sn_1, 4U, &kernelContext_39);
                float3 _S362 = _S361.xyz;
                float4 _S363 = node_0(sn_1, 13U, &kernelContext_39);
                float3 vs_0 = _S360 + cross(_S362, _S355 - _S363.xyz);
                float4 _S364 = node_0(dn_1, 3U, &kernelContext_39);
                float3 _S365 = _S364.xyz;
                float4 _S366 = node_0(dn_1, 4U, &kernelContext_39);
                float3 _S367 = _S366.xyz;
                float4 _S368 = node_0(dn_1, 13U, &kernelContext_39);
                float3 vd_0 = _S365 + cross(_S367, _S355 - _S368.xyz);
                float4 _S369 = node_0(sn_1, 13U, &kernelContext_39);
                float3 q_3 = _S355 - _S369.xyz;
                float3 dv_1 = vd_0 - vs_0;
                uint _S370 = global_node_0(g_17, sn_1 - _S331, &kernelContext_39);
                uint _S371 = global_node_0(g_17, dn_1 - _S331, &kernelContext_39);
                put_edge_0(4U, row_7, _S370, _S371, &kernelContext_39);
                put_0(_S337, 4U, row_7, 0U, ln_ratio_0(_S345.w, _S333 * _S333), &kernelContext_39);
                put_0(_S337, 4U, row_7, 1U, dot(ft1_1, q_3) / _S333, &kernelContext_39);
                put_0(_S337, 4U, row_7, 2U, dot(t2_2, q_3) / _S333, &kernelContext_39);
                put_0(_S337, 4U, row_7, 3U, dot(fn_1, q_3) / _S333, &kernelContext_39);
                put_0(_S337, 4U, row_7, 4U, gap_0 / _S333, &kernelContext_39);
                put_0(_S337, 4U, row_7, 5U, _S357, &kernelContext_39);
                put_0(_S337, 4U, row_7, 6U, dot(ft1_1, dv_1) / _S334, &kernelContext_39);
                put_0(_S337, 4U, row_7, 7U, dot(t2_2, dv_1) / _S334, &kernelContext_39);
                put_0(_S337, 4U, row_7, 8U, dot(fn_1, dv_1) / _S334, &kernelContext_39);
                put_0(_S337, 4U, row_7, 9U, _S356, &kernelContext_39);
                float4 _S372 = node_0(sn_1, 1U, &kernelContext_39);
                put_rot6_0(_S337, 4U, row_7, 10U, ft1_1, t2_2, quat_matrix_0(_S372), &kernelContext_39);
                float4 _S373 = node_0(dn_1, 1U, &kernelContext_39);
                put_rot6_0(_S337, 4U, row_7, 16U, ft1_1, t2_2, quat_matrix_0(_S373), &kernelContext_39);
                *((&kernelContext_39)->work_0+(24U * sn_1 + 4U)) = 1.0f;
                side_2 = side_2 + 1U;
            }
            put_key_0(4U, (_S335 + count_3) / 2U, ends_2.z, ends_2.w, &kernelContext_39);
        }
        count_3 = count_3 + 2U;
        ci_0 = ci_0 + 1U;
    }
    if(((&kernelContext_39)->params_0->count_only_0) != 0U)
    {
        edge_count_0(g_17, 0U, count_3, &kernelContext_39);
    }
    return;
}

[[kernel]] void feat_sc(uint3 group_5 [[threadgroup_position_in_grid]], uint3 thread_7 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_11 [[buffer(0)]], BondStatic_natural_0 device* bond_static_11 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_11 [[buffer(13)]], uint device* sinc_11 [[buffer(9)]], packed_uint4 device* slook_11 [[buffer(10)]], uint device* bmap_11 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_11 [[buffer(15)]], uint device* ctl_11 [[buffer(1)]], float device* work_11 [[buffer(18)]], packed_float4 device* nodes_11 [[buffer(3)]], packed_uint2 device* key_node_11 [[buffer(12)]], packed_float4 device* sc_11 [[buffer(7)]], packed_float4 device* bonds_11 [[buffer(4)]], packed_float4 device* sb_11 [[buffer(8)]], packed_uint4 device* bedge_11 [[buffer(22)]], packed_float4 device* calls_11 [[buffer(2)]], packed_float4 device* imps_11 [[buffer(16)]], packed_uint4 device* cisl_11 [[buffer(5)]], packed_float4 device* cislf_11 [[buffer(6)]], packed_float4 device* contacts_11 [[buffer(17)]], uint device* fctl_11 [[buffer(23)]], uint device* edges_11 [[buffer(20)]], uint device* cols_11 [[buffer(21)]], float device* feat_11 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_40;
    (&kernelContext_40)->params_0 = params_11;
    (&kernelContext_40)->bond_static_0 = bond_static_11;
    (&kernelContext_40)->chunks_0 = chunks_11;
    (&kernelContext_40)->sinc_0 = sinc_11;
    (&kernelContext_40)->slook_0 = slook_11;
    (&kernelContext_40)->bmap_0 = bmap_11;
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
    uint g_18 = group_5.x;
    uint t_13 = thread_7.x;
    bool _S374;
    if(g_18 >= ctl_11[int(1)])
    {
        _S374 = true;
    }
    else
    {
        _S374 = t_13 != 0U;
    }
    if(_S374)
    {
        return;
    }
    uint _S375 = call_first_0(g_18, &kernelContext_40);
    uint _S376 = call_e0_0(g_18, &kernelContext_40);
    uint _S377 = call_ne_0(g_18, &kernelContext_40);
    float4 _S378 = float4(*((&kernelContext_40)->calls_0+10U * g_18)) ;
    float _S379 = _S378.x;
    float _S380 = _S378.z;
    float _S381 = _S378.w;
    uint _S382;
    if(((&kernelContext_40)->params_0->count_only_0) != 0U)
    {
        _S382 = 0U;
    }
    else
    {
        uint _S383 = edge_base_0(g_18, 2U, &kernelContext_40);
        _S382 = _S383;
    }
    uint _S384 = (&kernelContext_40)->params_0->bases2_0.z;
    uint ea_0 = _S376;
    uint count_4 = 0U;
    for(;;)
    {
        uint _S385 = _S376 + _S377;
        if(ea_0 < _S385)
        {
        }
        else
        {
            break;
        }
        uint4 _S386 = uint4(*((&kernelContext_40)->cisl_0+2U * ea_0)) ;
        uint ja_0 = 0U;
        for(;;)
        {
            if(ja_0 < (_S386.z))
            {
            }
            else
            {
                break;
            }
            uint na_2 = _S386.y + ja_0;
            uint _S387 = 24U * na_2;
            uint _S388 = (as_type<uint>((*((&kernelContext_40)->work_0+(_S387 + 16U)))));
            float4 _S389 = node_0(na_2, 13U, &kernelContext_40);
            float3 _S390 = _S389.xyz;
            float _S391 = *((&kernelContext_40)->work_0+(_S387 + 12U));
            uint e_2 = _S376;
            uint count_5 = count_4;
            for(;;)
            {
                if(e_2 < _S385)
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
                uint4 _S392 = uint4(*((&kernelContext_40)->cisl_0+2U * e_2)) ;
                uint j_12 = 0U;
                count_6 = count_5;
                for(;;)
                {
                    if(j_12 < (_S392.z))
                    {
                    }
                    else
                    {
                        break;
                    }
                    uint nb_1 = _S392.y + j_12;
                    uint _S393 = 24U * nb_1;
                    uint rb_2 = (as_type<uint>((*((&kernelContext_40)->work_0+(_S393 + 16U)))));
                    if(rb_2 <= _S388)
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    float4 _S394 = node_0(nb_1, 13U, &kernelContext_40);
                    float3 d_1 = _S394.xyz - _S390;
                    float _S395 = length(d_1);
                    if(!(_S395 < (_S391 + *((&kernelContext_40)->work_0+(_S393 + 12U)))))
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    uint _S396 = bond_row_lookup_0(_S388, rb_2, &kernelContext_40);
                    if(_S396 != 4294967295U)
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    if(_S395 < 9.999999960041972e-13f)
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    float3 nrm_3 = normalize(d_1);
                    float4 _S397 = node_0(na_2, 1U, &kernelContext_40);
                    matrix<float,int(3),int(3)>  rma_0 = quat_matrix_0(_S397);
                    float4 _S398 = node_0(nb_1, 1U, &kernelContext_40);
                    uint _S399 = _S387 + 9U;
                    uint _S400 = _S393 + 9U;
                    float gap_1 = _S395 - support_of_0(rma_0, float3(*((&kernelContext_40)->work_0+_S399), *((&kernelContext_40)->work_0+(_S399 + 1U)), *((&kernelContext_40)->work_0+(_S399 + 2U))), nrm_3) - support_of_0(quat_matrix_0(_S398), float3(*((&kernelContext_40)->work_0+_S400), *((&kernelContext_40)->work_0+(_S400 + 1U)), *((&kernelContext_40)->work_0+(_S400 + 2U))), nrm_3);
                    if(gap_1 > (0.25f * _S379))
                    {
                        j_12 = j_12 + 1U;
                        continue;
                    }
                    if(((&kernelContext_40)->params_0->count_only_0) == 0U)
                    {
                        float3 t1_6 = tangent_of_0(nrm_3, rma_0);
                        float3 _S401 = cross(nrm_3, t1_6);
                        float fa_1 = *((&kernelContext_40)->work_0+(_S387 + 13U));
                        float fbr_1 = *((&kernelContext_40)->work_0+(_S393 + 13U));
                        float _S402;
                        if(((&kernelContext_40)->params_0->contact_friction_0) >= 0.0f)
                        {
                            _S402 = (&kernelContext_40)->params_0->contact_friction_0;
                        }
                        else
                        {
                            _S402 = min(fa_1, fbr_1);
                        }
                        float4 _S403 = node_0(na_2, 12U, &kernelContext_40);
                        uint _S404 = (as_type<uint>((_S403.z)));
                        float4 _S405 = node_0(nb_1, 12U, &kernelContext_40);
                        uint _S406 = (as_type<uint>((_S405.z)));
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
                            uint row_8 = _S382 + count_6 + side_3;
                            bool _S407 = side_3 == 0U;
                            uint sn_2;
                            if(_S407)
                            {
                                sn_2 = na_2;
                            }
                            else
                            {
                                sn_2 = nb_1;
                            }
                            uint dn_2;
                            if(_S407)
                            {
                                dn_2 = nb_1;
                            }
                            else
                            {
                                dn_2 = na_2;
                            }
                            float3 ft1_2;
                            if(_S407)
                            {
                                ft1_2 = t1_6;
                            }
                            else
                            {
                                ft1_2 = - t1_6;
                            }
                            float3 fn_2;
                            if(_S407)
                            {
                                fn_2 = nrm_3;
                            }
                            else
                            {
                                fn_2 = - nrm_3;
                            }
                            float4 _S408 = node_0(sn_2, 3U, &kernelContext_40);
                            float3 _S409 = _S408.xyz;
                            float4 _S410 = node_0(dn_2, 3U, &kernelContext_40);
                            float3 rel_1 = _S409 - _S410.xyz;
                            float _S411 = dot(ft1_2, rel_1);
                            float _S412 = dot(_S401, rel_1);
                            float _S413 = dot(fn_2, rel_1);
                            float t_con_0;
                            if(gap_1 <= 0.0f)
                            {
                                t_con_0 = -1.0f;
                            }
                            else
                            {
                                if(_S413 > 0.0f)
                                {
                                    _S374 = (gap_1 / _S413) < ((&kernelContext_40)->params_0->dt_0);
                                }
                                else
                                {
                                    _S374 = false;
                                }
                                if(_S374)
                                {
                                    t_con_0 = gap_1 / _S413 / (&kernelContext_40)->params_0->dt_0;
                                }
                                else
                                {
                                    t_con_0 = 1.0f;
                                }
                            }
                            uint _S414 = global_node_0(g_18, sn_2 - _S375, &kernelContext_40);
                            uint _S415 = global_node_0(g_18, dn_2 - _S375, &kernelContext_40);
                            put_edge_0(6U, row_8, _S414, _S415, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 0U, _S413 / _S380, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 1U, _S411 / _S380, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 2U, _S412 / _S380, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 3U, _S413 / _S381, &kernelContext_40);
                            float4 _S416 = node_0(dn_2, 0U, &kernelContext_40);
                            float _S417 = _S416.w;
                            float4 _S418 = node_0(sn_2, 0U, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 4U, ln_ratio_0(_S417, _S418.w), &kernelContext_40);
                            put_0(_S384, 6U, row_8, 5U, t_con_0, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 6U, max(- gap_1, 0.0f) / _S379, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 7U, _S402, &kernelContext_40);
                            put_0(_S384, 6U, row_8, 8U, 0.0f, &kernelContext_40);
                            float _S419;
                            if(_S404 == _S406)
                            {
                                _S419 = 1.0f;
                            }
                            else
                            {
                                _S419 = 0.0f;
                            }
                            put_0(_S384, 6U, row_8, 9U, _S419, &kernelContext_40);
                            float4 _S420 = node_0(sn_2, 1U, &kernelContext_40);
                            put_rot6_0(_S384, 6U, row_8, 10U, ft1_2, _S401, quat_matrix_0(_S420), &kernelContext_40);
                            float4 _S421 = node_0(dn_2, 1U, &kernelContext_40);
                            put_rot6_0(_S384, 6U, row_8, 16U, ft1_2, _S401, quat_matrix_0(_S421), &kernelContext_40);
                            *((&kernelContext_40)->work_0+(24U * sn_2 + 4U)) = 1.0f;
                            side_3 = side_3 + 1U;
                        }
                        put_key_0(6U, (_S382 + count_6) / 2U, _S388, rb_2, &kernelContext_40);
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
    if(((&kernelContext_40)->params_0->count_only_0) != 0U)
    {
        edge_count_0(g_18, 2U, count_4, &kernelContext_40);
    }
    return;
}

[[kernel]] void feat_prefix(FeatParams_0 constant* params_12 [[buffer(0)]], BondStatic_natural_0 device* bond_static_12 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_12 [[buffer(13)]], uint device* sinc_12 [[buffer(9)]], packed_uint4 device* slook_12 [[buffer(10)]], uint device* bmap_12 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_12 [[buffer(15)]], uint device* ctl_12 [[buffer(1)]], float device* work_12 [[buffer(18)]], packed_float4 device* nodes_12 [[buffer(3)]], packed_uint2 device* key_node_12 [[buffer(12)]], packed_float4 device* sc_12 [[buffer(7)]], packed_float4 device* bonds_12 [[buffer(4)]], packed_float4 device* sb_12 [[buffer(8)]], packed_uint4 device* bedge_12 [[buffer(22)]], packed_float4 device* calls_12 [[buffer(2)]], packed_float4 device* imps_12 [[buffer(16)]], packed_uint4 device* cisl_12 [[buffer(5)]], packed_float4 device* cislf_12 [[buffer(6)]], packed_float4 device* contacts_12 [[buffer(17)]], uint device* fctl_12 [[buffer(23)]], uint device* edges_12 [[buffer(20)]], uint device* cols_12 [[buffer(21)]], float device* feat_12 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_41;
    (&kernelContext_41)->params_0 = params_12;
    (&kernelContext_41)->bond_static_0 = bond_static_12;
    (&kernelContext_41)->chunks_0 = chunks_12;
    (&kernelContext_41)->sinc_0 = sinc_12;
    (&kernelContext_41)->slook_0 = slook_12;
    (&kernelContext_41)->bmap_0 = bmap_12;
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
    uint _S422 = ctl_12[int(1)];
    uint g_19 = 0U;
    uint tot0_0 = 0U;
    uint tot1_0 = 0U;
    uint tot2_0 = 0U;
    for(;;)
    {
        if(g_19 < _S422)
        {
        }
        else
        {
            break;
        }
        uint _S423 = 8U + 6U * g_19;
        *((&kernelContext_41)->fctl_0+(_S423 + 3U)) = tot0_0;
        *((&kernelContext_41)->fctl_0+(_S423 + 4U)) = tot1_0;
        *((&kernelContext_41)->fctl_0+(_S423 + 5U)) = tot2_0;
        uint tot0_1 = tot0_0 + *((&kernelContext_41)->fctl_0+_S423);
        uint tot1_1 = tot1_0 + *((&kernelContext_41)->fctl_0+(_S423 + 1U));
        uint tot2_1 = tot2_0 + *((&kernelContext_41)->fctl_0+(_S423 + 2U));
        g_19 = g_19 + 1U;
        tot0_0 = tot0_1;
        tot1_0 = tot1_1;
        tot2_0 = tot2_1;
    }
    uint over_0;
    if(tot0_0 > (2U * (&kernelContext_41)->params_0->caps_0.x))
    {
        over_0 = 1U;
    }
    else
    {
        over_0 = 0U;
    }
    if(tot1_0 > (2U * (&kernelContext_41)->params_0->caps_0.y))
    {
        over_0 = over_0 | 2U;
    }
    if(tot2_0 > (2U * (&kernelContext_41)->params_0->caps_0.z))
    {
        over_0 = over_0 | 4U;
    }
    g_19 = 0U;
    uint cand_over_0 = 0U;
    for(;;)
    {
        if(g_19 < _S422)
        {
        }
        else
        {
            break;
        }
        uint _S424 = (&kernelContext_41)->params_0->max_bonds_cap_0;
        uint _S425 = call_first_0(g_19, &kernelContext_41);
        uint cand_over_1 = cand_over_0 | ((uint4(*((&kernelContext_41)->bedge_0+(_S424 + 8U * _S425))) ).y);
        g_19 = g_19 + 1U;
        cand_over_0 = cand_over_1;
    }
    *((&kernelContext_41)->fctl_0+int(5)) = cand_over_0;
    *((&kernelContext_41)->fctl_0+int(0)) = tot0_0;
    *((&kernelContext_41)->fctl_0+int(1)) = tot1_0;
    *((&kernelContext_41)->fctl_0+int(2)) = tot2_0;
    *((&kernelContext_41)->fctl_0+int(3)) = over_0;
    g_19 = 0U;
    uint chunks_n_0 = 0U;
    for(;;)
    {
        if(g_19 < _S422)
        {
        }
        else
        {
            break;
        }
        uint _S426 = call_chunks_0(g_19, &kernelContext_41);
        uint chunks_n_1 = chunks_n_0 + _S426;
        g_19 = g_19 + 1U;
        chunks_n_0 = chunks_n_1;
    }
    *((&kernelContext_41)->fctl_0+int(4)) = chunks_n_0;
    return;
}

uint call_nodes_0(uint g_20, KernelContext_0 thread* kernelContext_42)
{
    return (as_type<uint>(((float4(*(kernelContext_42->calls_0+(10U * g_20 + 2U))) ).w)));
}

float cap4_0(float x_1)
{
    float _S427;
    if(x_1 < 3.00000000549775576e+38f)
    {
        _S427 = min(x_1, 4.0f);
    }
    else
    {
        _S427 = 4.0f;
    }
    return _S427;
}

[[kernel]] void feat_nodes(uint3 group_6 [[threadgroup_position_in_grid]], uint3 thread_8 [[thread_position_in_threadgroup]], FeatParams_0 constant* params_13 [[buffer(0)]], BondStatic_natural_0 device* bond_static_13 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_13 [[buffer(13)]], uint device* sinc_13 [[buffer(9)]], packed_uint4 device* slook_13 [[buffer(10)]], uint device* bmap_13 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_13 [[buffer(15)]], uint device* ctl_13 [[buffer(1)]], float device* work_13 [[buffer(18)]], packed_float4 device* nodes_13 [[buffer(3)]], packed_uint2 device* key_node_13 [[buffer(12)]], packed_float4 device* sc_13 [[buffer(7)]], packed_float4 device* bonds_13 [[buffer(4)]], packed_float4 device* sb_13 [[buffer(8)]], packed_uint4 device* bedge_13 [[buffer(22)]], packed_float4 device* calls_13 [[buffer(2)]], packed_float4 device* imps_13 [[buffer(16)]], packed_uint4 device* cisl_13 [[buffer(5)]], packed_float4 device* cislf_13 [[buffer(6)]], packed_float4 device* contacts_13 [[buffer(17)]], uint device* fctl_13 [[buffer(23)]], uint device* edges_13 [[buffer(20)]], uint device* cols_13 [[buffer(21)]], float device* feat_13 [[buffer(19)]])
{
    float ke_imp_0;
    float _S428;
    bool _S429;
    thread KernelContext_0 kernelContext_43;
    (&kernelContext_43)->params_0 = params_13;
    (&kernelContext_43)->bond_static_0 = bond_static_13;
    (&kernelContext_43)->chunks_0 = chunks_13;
    (&kernelContext_43)->sinc_0 = sinc_13;
    (&kernelContext_43)->slook_0 = slook_13;
    (&kernelContext_43)->bmap_0 = bmap_13;
    (&kernelContext_43)->bond_dyn_0 = bond_dyn_13;
    (&kernelContext_43)->ctl_0 = ctl_13;
    (&kernelContext_43)->work_0 = work_13;
    (&kernelContext_43)->nodes_0 = nodes_13;
    (&kernelContext_43)->key_node_0 = key_node_13;
    (&kernelContext_43)->sc_0 = sc_13;
    (&kernelContext_43)->bonds_0 = bonds_13;
    (&kernelContext_43)->sb_0 = sb_13;
    (&kernelContext_43)->bedge_0 = bedge_13;
    (&kernelContext_43)->calls_0 = calls_13;
    (&kernelContext_43)->imps_0 = imps_13;
    (&kernelContext_43)->cisl_0 = cisl_13;
    (&kernelContext_43)->cislf_0 = cislf_13;
    (&kernelContext_43)->contacts_0 = contacts_13;
    (&kernelContext_43)->fctl_0 = fctl_13;
    (&kernelContext_43)->edges_0 = edges_13;
    (&kernelContext_43)->cols_0 = cols_13;
    (&kernelContext_43)->feat_0 = feat_13;
    threadgroup array<array<float4, int(64)>, int(2)> g_red_13;
    (&kernelContext_43)->g_red_0 = &g_red_13;
    threadgroup array<float, int(64)> g_mv_13;
    (&kernelContext_43)->g_mv_0 = &g_mv_13;
    threadgroup array<uint, int(64)> g_mi_13;
    (&kernelContext_43)->g_mi_0 = &g_mi_13;
    threadgroup array<uint, int(4096)> g_plug_13;
    (&kernelContext_43)->g_plug_0 = &g_plug_13;
    threadgroup uint g_flag_13;
    (&kernelContext_43)->g_flag_0 = &g_flag_13;
    threadgroup array<uint, int(64)> g_scan_13;
    (&kernelContext_43)->g_scan_0 = &g_scan_13;
    threadgroup float4 g_scalars_13;
    (&kernelContext_43)->g_scalars_0 = &g_scalars_13;
    uint g_21 = group_6.x;
    uint t_14 = thread_8.x;
    uint j_13;
    if(g_21 >= ctl_13[int(1)])
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
    uint _S430 = call_first_0(g_21, &kernelContext_43);
    uint _S431 = call_nodes_0(g_21, &kernelContext_43);
    uint n_9 = group_uniform_0(_S431, t_14);
    uint _S432 = call_chunks_0(g_21, &kernelContext_43);
    uint nc_1 = group_uniform_0(_S432, t_14);
    uint _S433 = 10U * g_21;
    float4 _S434 = float4(*((&kernelContext_43)->calls_0+_S433)) ;
    float L_1 = _S434.x;
    float _S435 = _S434.y;
    float vref_0 = _S434.z;
    float c_wave_1 = _S434.w;
    float _S436 = (float4(*((&kernelContext_43)->calls_0+(_S433 + 1U))) ).y;
    float dt_2 = (&kernelContext_43)->params_0->dt_0;
    float3 grav_0 = (&kernelContext_43)->params_0->gravity_0.xyz;
    float4 _S437 = float4(0.0f) ;
    j_13 = t_14;
    float4 a_10 = _S437;
    float4 b_6 = _S437;
    for(;;)
    {
        if(j_13 < n_9)
        {
        }
        else
        {
            break;
        }
        uint _S438 = _S430 + j_13;
        float4 _S439 = node_0(_S438, 0U, &kernelContext_43);
        float m_4 = _S439.w;
        float4 _S440 = node_0(_S438, 3U, &kernelContext_43);
        float4 a_11 = a_10 + float4(_S440.xyz * float3(m_4) , m_4);
        bool _S441 = j_13 < nc_1;
        if(_S441)
        {
            ke_imp_0 = m_4;
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        if(_S441)
        {
            float4 _S442 = node_0(_S438, 2U, &kernelContext_43);
            _S429 = (_S442.w) != 0.0f;
        }
        else
        {
            _S429 = false;
        }
        if(_S429)
        {
            _S428 = 1.0f;
        }
        else
        {
            _S428 = 0.0f;
        }
        float4 b_7 = b_6 + float4(ke_imp_0, _S428, 0.0f, 0.0f);
        j_13 = j_13 + 64U;
        a_10 = a_11;
        b_6 = b_7;
    }
    float4 _S443 = group_sum_0(t_14, a_10, &kernelContext_43);
    float4 _S444 = group_sum_0(t_14, b_6, &kernelContext_43);
    float3 _S445 = _S443.xyz / float3(_S443.w) ;
    float _S446 = _S444.x;
    bool anchored_0 = (_S444.y) > 0.0f;
    uint _S447 = _S433 + 9U;
    float _S448 = (float4(*((&kernelContext_43)->calls_0+_S447)) ).y;
    float _S449 = (float4(*((&kernelContext_43)->calls_0+_S447)) ).z;
    uint _S450 = (&kernelContext_43)->params_0->bases_0.x;
    uint _S451 = (&kernelContext_43)->params_0->bases_0.y;
    uint fbg_0 = (&kernelContext_43)->params_0->bases_0.z;
    uint _S452 = (&kernelContext_43)->params_0->bases2_0.w;
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
        uint k_10 = _S430 + j_13;
        float4 _S453 = node_0(k_10, 1U, &kernelContext_43);
        matrix<float,int(3),int(3)>  rb_3 = quat_matrix_0(_S453);
        matrix<float,int(3),int(3)>  bt_0 = transpose(rb_3);
        float4 _S454 = node_0(k_10, 0U, &kernelContext_43);
        float m_5 = _S454.w;
        float3 gb_1 = (((grav_0) * (bt_0))) / float3(9.81000041961669922f) ;
        float4 _S455 = node_0(k_10, 1U, &kernelContext_43);
        uint _S456 = global_node_0(g_21, j_13, &kernelContext_43);
        uint _S457 = _S452 + 7U * _S456;
        float device* _S458 = (&kernelContext_43)->feat_0+_S457;
        float4 _S459 = node_0(k_10, 0U, &kernelContext_43);
        *_S458 = _S459.x;
        float device* _S460 = (&kernelContext_43)->feat_0+(_S457 + 1U);
        float4 _S461 = node_0(k_10, 0U, &kernelContext_43);
        *_S460 = _S461.y;
        float device* _S462 = (&kernelContext_43)->feat_0+(_S457 + 2U);
        float4 _S463 = node_0(k_10, 0U, &kernelContext_43);
        *_S462 = _S463.z;
        *((&kernelContext_43)->feat_0+(_S457 + 3U)) = _S455.w;
        *((&kernelContext_43)->feat_0+(_S457 + 4U)) = _S455.x;
        *((&kernelContext_43)->feat_0+(_S457 + 5U)) = _S455.y;
        *((&kernelContext_43)->feat_0+(_S457 + 6U)) = _S455.z;
        uint c_2;
        if(j_13 < nc_1)
        {
            uint _S464 = call_cbase_0(g_21, &kernelContext_43);
            uint row_9 = _S464 + j_13;
            float4 _S465 = node_0(k_10, 2U, &kernelContext_43);
            float3 inr_0 = _S465.xyz;
            uint _S466 = 24U * k_10;
            uint _S467 = _S466 + 9U;
            float device* _S468 = (&kernelContext_43)->work_0+_S467;
            float device* _S469 = (&kernelContext_43)->work_0+(_S467 + 1U);
            float device* _S470 = (&kernelContext_43)->work_0+(_S467 + 2U);
            float device* _S471 = (&kernelContext_43)->work_0+(_S466 + 14U);
            float fixed_0 = *_S471;
            float nlive_3 = *((&kernelContext_43)->work_0+(_S466 + 7U));
            float4 _S472 = node_0(k_10, 3U, &kernelContext_43);
            float3 vr_0 = (((_S472.xyz - _S445) * (bt_0))) / float3(vref_0) ;
            float4 _S473 = node_0(k_10, 4U, &kernelContext_43);
            float3 wb_0 = (((_S473.xyz) * (bt_0))) * float3((L_1 / vref_0)) ;
            put_0(_S450, 0U, row_9, 0U, ln_ratio_0(m_5, _S435), &kernelContext_43);
            float _S474 = m_5 * L_1 * L_1;
            put_0(_S450, 0U, row_9, 1U, inr_0.x / _S474, &kernelContext_43);
            put_0(_S450, 0U, row_9, 2U, inr_0.y / _S474, &kernelContext_43);
            put_0(_S450, 0U, row_9, 3U, inr_0.z / _S474, &kernelContext_43);
            put_0(_S450, 0U, row_9, 4U, *_S468 / L_1, &kernelContext_43);
            put_0(_S450, 0U, row_9, 5U, *_S469 / L_1, &kernelContext_43);
            put_0(_S450, 0U, row_9, 6U, *_S470 / L_1, &kernelContext_43);
            if((*_S471) == 1.0f)
            {
                _S428 = 1.0f;
            }
            else
            {
                _S428 = 0.0f;
            }
            put_0(_S450, 0U, row_9, 7U, _S428, &kernelContext_43);
            float _S475;
            if(fixed_0 == 2.0f)
            {
                _S475 = 1.0f;
            }
            else
            {
                _S475 = 0.0f;
            }
            put_0(_S450, 0U, row_9, 8U, _S475, &kernelContext_43);
            float _S476;
            if(nlive_3 > 0.0f)
            {
                _S476 = *((&kernelContext_43)->work_0+(_S466 + 5U)) / nlive_3;
            }
            else
            {
                _S476 = 0.0f;
            }
            put_0(_S450, 0U, row_9, 9U, _S476, &kernelContext_43);
            put_0(_S450, 0U, row_9, 10U, *((&kernelContext_43)->work_0+(_S466 + 6U)), &kernelContext_43);
            put_0(_S450, 0U, row_9, 11U, nlive_3 / 6.0f, &kernelContext_43);
            put_0(_S450, 0U, row_9, 12U, *((&kernelContext_43)->work_0+(_S466 + 8U)) / 6.0f, &kernelContext_43);
            put_0(_S450, 0U, row_9, 13U, cap4_0(*((&kernelContext_43)->work_0+_S466) / 16.0f), &kernelContext_43);
            put_0(_S450, 0U, row_9, 14U, cap4_0(*((&kernelContext_43)->work_0+(_S466 + 1U)) / dt_2), &kernelContext_43);
            put_0(_S450, 0U, row_9, 15U, cap4_0(*((&kernelContext_43)->work_0+(_S466 + 2U)) / 16.0f), &kernelContext_43);
            put_0(_S450, 0U, row_9, 16U, gb_1.x, &kernelContext_43);
            put_0(_S450, 0U, row_9, 17U, gb_1.y, &kernelContext_43);
            put_0(_S450, 0U, row_9, 18U, gb_1.z, &kernelContext_43);
            put_0(_S450, 0U, row_9, 19U, vr_0.x, &kernelContext_43);
            put_0(_S450, 0U, row_9, 20U, vr_0.y, &kernelContext_43);
            put_0(_S450, 0U, row_9, 21U, vr_0.z, &kernelContext_43);
            put_0(_S450, 0U, row_9, 22U, wb_0.x, &kernelContext_43);
            put_0(_S450, 0U, row_9, 23U, wb_0.y, &kernelContext_43);
            put_0(_S450, 0U, row_9, 24U, wb_0.z, &kernelContext_43);
            float _S477;
            if((*((&kernelContext_43)->work_0+(_S466 + 3U))) != 0.0f)
            {
                _S477 = 1.0f;
            }
            else
            {
                _S477 = 0.0f;
            }
            put_0(_S450, 0U, row_9, 25U, _S477, &kernelContext_43);
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
                put_0(_S450, 0U, row_9, c_2, 0.0f, &kernelContext_43);
                c_2 = c_2 + 1U;
            }
            put_0(_S450, 0U, row_9, 32U, *((&kernelContext_43)->work_0+(_S466 + 4U)), &kernelContext_43);
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
                put_0(_S450, 0U, row_9, c_3, 0.0f, &kernelContext_43);
                c_3 = c_3 + 1U;
            }
        }
        else
        {
            float4 _S478 = node_0(k_10, 12U, &kernelContext_43);
            uint ki_0 = (as_type<uint>((_S478.x)));
            uint _S479 = call_ibase_0(g_21, &kernelContext_43);
            uint row_10 = _S479 + (j_13 - nc_1);
            uint _S480 = 4U * ki_0 + 3U;
            float r_face_0 = (float4(*((&kernelContext_43)->imps_0+_S480)) ).z;
            float4 _S481 = node_0(k_10, 2U, &kernelContext_43);
            float3 inr_1 = _S481.xyz;
            float r2_0 = r_face_0 * r_face_0;
            float4 _S482 = node_0(k_10, 3U, &kernelContext_43);
            float3 vr_1 = (((_S482.xyz - _S445) * (bt_0)));
            float4 _S483 = node_0(k_10, 3U, &kernelContext_43);
            float3 v_6 = _S483.xyz;
            float4 _S484 = node_0(k_10, 4U, &kernelContext_43);
            float3 w_2 = _S484.xyz;
            float _S485 = inr_1.x;
            float _S486 = inr_1.y;
            float _S487 = inr_1.z;
            float ke_imp_1 = ke_imp_0 + (0.5f * m_5 * dot(v_6, v_6) + 0.5f * dot(w_2, (((w_2) * ((((bt_0) * ((((matrix<float,int(3),int(3)> (float3(_S485, 0.0f, 0.0f), float3(0.0f, _S486, 0.0f), float3(0.0f, 0.0f, _S487))) * (rb_3)))))))))));
            put_0(_S451, 1U, row_10, 0U, ln_ratio_0(m_5, _S446), &kernelContext_43);
            put_0(_S451, 1U, row_10, 1U, ln_ratio_0(m_5, _S435), &kernelContext_43);
            float _S488 = m_5 * r2_0;
            put_0(_S451, 1U, row_10, 2U, _S485 / _S488, &kernelContext_43);
            put_0(_S451, 1U, row_10, 3U, _S486 / _S488, &kernelContext_43);
            put_0(_S451, 1U, row_10, 4U, _S487 / _S488, &kernelContext_43);
            put_0(_S451, 1U, row_10, 5U, ln_ratio_0(r_face_0, L_1), &kernelContext_43);
            put_0(_S451, 1U, row_10, 6U, ln_ratio_0((float4(*((&kernelContext_43)->imps_0+_S480)) ).x, _S436), &kernelContext_43);
            put_0(_S451, 1U, row_10, 7U, (float4(*((&kernelContext_43)->imps_0+_S480)) ).w, &kernelContext_43);
            put_0(_S451, 1U, row_10, 8U, gb_1.x, &kernelContext_43);
            put_0(_S451, 1U, row_10, 9U, gb_1.y, &kernelContext_43);
            put_0(_S451, 1U, row_10, 10U, gb_1.z, &kernelContext_43);
            put_0(_S451, 1U, row_10, 11U, vr_1.x / vref_0, &kernelContext_43);
            put_0(_S451, 1U, row_10, 12U, vr_1.y / vref_0, &kernelContext_43);
            put_0(_S451, 1U, row_10, 13U, vr_1.z / vref_0, &kernelContext_43);
            float4 _S489 = node_0(k_10, 3U, &kernelContext_43);
            put_0(_S451, 1U, row_10, 14U, length(_S489.xyz - _S445) / c_wave_1, &kernelContext_43);
            put_0(_S451, 1U, row_10, 15U, *((&kernelContext_43)->work_0+(24U * k_10 + 4U)), &kernelContext_43);
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
                put_0(_S451, 1U, row_10, c_2, 0.0f, &kernelContext_43);
                c_2 = c_2 + 1U;
            }
            ke_imp_0 = ke_imp_1;
        }
        j_13 = j_13 + 64U;
    }
    float4 _S490 = group_sum_0(t_14, float4(ke_imp_0, 0.0f, 0.0f, 0.0f), &kernelContext_43);
    float ke_imp_2 = _S490.x;
    if(t_14 == 0U)
    {
        uint ni_0 = n_9 - nc_1;
        bool _S491 = ke_imp_2 > 0.0f;
        if(_S491)
        {
            _S429 = _S448 > 0.0f;
        }
        else
        {
            _S429 = false;
        }
        if(_S429)
        {
            ke_imp_0 = ln_ratio_0(ke_imp_2, _S448);
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, 2U, g_21, 0U, ke_imp_0, &kernelContext_43);
        if(_S491)
        {
            _S429 = _S449 > 0.0f;
        }
        else
        {
            _S429 = false;
        }
        if(_S429)
        {
            ke_imp_0 = ln_ratio_0(ke_imp_2, _S449);
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, 2U, g_21, 1U, ke_imp_0, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 2U, log(dt_2 * c_wave_1 / L_1), &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 3U, dt_2 * vref_0 / L_1, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 4U, float(ni_0) / 4.0f, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 5U, log(float(nc_1)) / 10.0f, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 6U, length(grav_0) / 9.81000041961669922f, &kernelContext_43);
        if(anchored_0)
        {
            ke_imp_0 = 1.0f;
        }
        else
        {
            ke_imp_0 = 0.0f;
        }
        put_0(fbg_0, 2U, g_21, 7U, ke_imp_0, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 8U, 0.0f, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 9U, 0.0f, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 10U, 0.0f, &kernelContext_43);
        put_0(fbg_0, 2U, g_21, 11U, vref_0 / c_wave_1, &kernelContext_43);
    }
    return;
}

float fail_margin_of_0(const JointState_0 thread* js_1, float4 s8_0, bool rebar_0, bool live_0)
{
    bool ductile_0 = (s8_0.x) != 0.0f;
    bool _S492 = !((s8_0.y) != 0.0f);
    float kf_ts_0;
    if(_S492)
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
    if(_S492)
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
    float _S493 = max(js_1->kappa_0 / kf_ts_0, js_1->kappa_c_0 / kf_c_0);
    bool _S494;
    if(rebar_0)
    {
        _S494 = (s8_0.z) > 0.0f;
    }
    else
    {
        _S494 = false;
    }
    float q_4;
    if(_S494)
    {
        q_4 = min(_S493, js_1->rebar_work_0 / s8_0.z);
    }
    else
    {
        q_4 = _S493;
    }
    if(live_0)
    {
        _S494 = q_4 >= 1.0f;
    }
    else
    {
        _S494 = false;
    }
    if(_S494)
    {
        q_4 = 0.99999994039535522f;
    }
    return q_4;
}

[[kernel]] void feat_bonds(uint3 id_4 [[thread_position_in_grid]], FeatParams_0 constant* params_14 [[buffer(0)]], BondStatic_natural_0 device* bond_static_14 [[buffer(14)]], ChunkStatic_natural_0 device* chunks_14 [[buffer(13)]], uint device* sinc_14 [[buffer(9)]], packed_uint4 device* slook_14 [[buffer(10)]], uint device* bmap_14 [[buffer(11)]], BondDyn_natural_0 device* bond_dyn_14 [[buffer(15)]], uint device* ctl_14 [[buffer(1)]], float device* work_14 [[buffer(18)]], packed_float4 device* nodes_14 [[buffer(3)]], packed_uint2 device* key_node_14 [[buffer(12)]], packed_float4 device* sc_14 [[buffer(7)]], packed_float4 device* bonds_14 [[buffer(4)]], packed_float4 device* sb_14 [[buffer(8)]], packed_uint4 device* bedge_14 [[buffer(22)]], packed_float4 device* calls_14 [[buffer(2)]], packed_float4 device* imps_14 [[buffer(16)]], packed_uint4 device* cisl_14 [[buffer(5)]], packed_float4 device* cislf_14 [[buffer(6)]], packed_float4 device* contacts_14 [[buffer(17)]], uint device* fctl_14 [[buffer(23)]], uint device* edges_14 [[buffer(20)]], uint device* cols_14 [[buffer(21)]], float device* feat_14 [[buffer(19)]])
{
    thread KernelContext_0 kernelContext_44;
    (&kernelContext_44)->params_0 = params_14;
    (&kernelContext_44)->bond_static_0 = bond_static_14;
    (&kernelContext_44)->chunks_0 = chunks_14;
    (&kernelContext_44)->sinc_0 = sinc_14;
    (&kernelContext_44)->slook_0 = slook_14;
    (&kernelContext_44)->bmap_0 = bmap_14;
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
    uint b_8 = id_4.x;
    if(b_8 >= ctl_14[int(3)])
    {
        return;
    }
    uint4 _S495 = uint4(*((&kernelContext_44)->bedge_0+b_8)) ;
    uint _S496 = _S495.x;
    bool _S497;
    if(_S496 == 4294967295U)
    {
        _S497 = true;
    }
    else
    {
        _S497 = (_S495.y) == 4294967295U;
    }
    if(_S497)
    {
        return;
    }
    float4 _S498 = node_0(_S496, 3U, &kernelContext_44);
    uint _S499 = 10U * (as_type<uint>((_S498.w)));
    float4 _S500 = float4(*((&kernelContext_44)->calls_0+_S499)) ;
    float4 _S501 = float4(*((&kernelContext_44)->calls_0+(_S499 + 1U))) ;
    float L_2 = _S500.x;
    float _S502 = _S500.z;
    float _S503 = _S501.x;
    float e_med_1 = _S501.y;
    float _S504 = _S501.z;
    uint _S505 = 9U * _S495.z;
    float4 _S506 = float4(*((&kernelContext_44)->sb_0+_S505)) ;
    float4 _S507 = float4(*((&kernelContext_44)->sb_0+(_S505 + 1U))) ;
    float4 _S508 = float4(*((&kernelContext_44)->sb_0+(_S505 + 2U))) ;
    float4 _S509 = float4(*((&kernelContext_44)->sb_0+(_S505 + 3U))) ;
    float4 _S510 = float4(*((&kernelContext_44)->sb_0+(_S505 + 4U))) ;
    float4 _S511 = float4(*((&kernelContext_44)->sb_0+(_S505 + 5U))) ;
    float4 _S512 = float4(*((&kernelContext_44)->sb_0+(_S505 + 6U))) ;
    float4 _S513 = float4(*((&kernelContext_44)->sb_0+(_S505 + 8U))) ;
    uint nb_2 = _S495.y;
    float4 _S514 = node_0(_S496, 1U, &kernelContext_44);
    uint _S515 = 24U * _S496;
    uint _S516 = _S515 + 17U;
    matrix<float,int(3),int(3)>  rs_1 = (((transpose(quat_matrix_0(float4(*((&kernelContext_44)->work_0+_S516), *((&kernelContext_44)->work_0+(_S516 + 1U)), *((&kernelContext_44)->work_0+(_S516 + 2U)), *((&kernelContext_44)->work_0+(_S516 + 3U)))))) * (quat_matrix_0(_S514))));
    float4 _S517 = node_0(_S496, 13U, &kernelContext_44);
    float3 xa_1 = _S517.xyz;
    float4 _S518 = node_0(nb_2, 13U, &kernelContext_44);
    float3 nrm_4 = (((_S507.xyz) * (rs_1)));
    float3 t1_7 = (((_S508.xyz) * (rs_1)));
    float3 nrm_5;
    float3 t1_8;
    if((dot(nrm_4, _S518.xyz - xa_1)) < 0.0f)
    {
        float3 _S519 = - t1_7;
        nrm_5 = - nrm_4;
        t1_8 = _S519;
    }
    else
    {
        nrm_5 = nrm_4;
        t1_8 = t1_7;
    }
    float3 _S520 = cross(nrm_5, t1_8);
    uint _S521 = _S515 + 21U;
    float3 _S522 = xa_1 + (((_S506.xyz - float3(*((&kernelContext_44)->work_0+_S521), *((&kernelContext_44)->work_0+(_S521 + 1U)), *((&kernelContext_44)->work_0+(_S521 + 2U)))) * (rs_1)));
    float area_1 = _S506.w;
    float _S523 = e_med_1 * area_1;
    float _S524 = _S523 / L_2;
    float _S525 = _S523 * L_2;
    bool rebar_1 = (_S510.w) != 0.0f;
    JointState_0 _S526 = ((&kernelContext_44)->bond_dyn_0+_S495.w)->js_0;
    thread JointState_0 _S527 = ((&kernelContext_44)->bond_dyn_0+_S495.w)->js_0;
    float _S528 = fail_margin_of_0(&_S527, _S513, rebar_1, true);
    uint pa_0 = uint(*((&kernelContext_44)->work_0+(_S515 + 3U)));
    uint pb_0 = uint(*((&kernelContext_44)->work_0+(24U * nb_2 + 3U)));
    bool in_plug_0 = (pa_0 != 0U) != (pb_0 != 0U);
    float ratio_1;
    if(in_plug_0)
    {
        ratio_1 = ln_ratio_0((float4(*((&kernelContext_44)->contacts_0+(6U * ((as_type<uint>(((float4(*((&kernelContext_44)->calls_0+(_S499 + 8U))) ).x))) + max(pa_0, pb_0) - 1U) + 4U))) ).x, 1.0f);
    }
    else
    {
        ratio_1 = 0.0f;
    }
    float _S529 = _S511.x;
    float _S530 = _S511.y;
    float _S531 = _S511.z;
    float _S532 = _S512.y;
    float _S533 = _S512.z;
    float _S534 = _S512.w;
    uint _S535 = (&kernelContext_44)->params_0->bases_0.w;
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
        bool _S536 = side_4 == 0U;
        uint sn_3;
        if(_S536)
        {
            sn_3 = _S496;
        }
        else
        {
            sn_3 = nb_2;
        }
        uint dn_3;
        if(_S536)
        {
            dn_3 = nb_2;
        }
        else
        {
            dn_3 = _S496;
        }
        float3 ft1_3;
        if(_S536)
        {
            ft1_3 = t1_8;
        }
        else
        {
            ft1_3 = - t1_8;
        }
        float3 fn_3;
        if(_S536)
        {
            fn_3 = nrm_5;
        }
        else
        {
            fn_3 = - nrm_5;
        }
        float4 _S537 = node_0(sn_3, 3U, &kernelContext_44);
        float3 _S538 = _S537.xyz;
        float4 _S539 = node_0(sn_3, 4U, &kernelContext_44);
        float3 _S540 = _S539.xyz;
        float4 _S541 = node_0(sn_3, 13U, &kernelContext_44);
        float3 vs_1 = _S538 + cross(_S540, _S522 - _S541.xyz);
        float4 _S542 = node_0(dn_3, 3U, &kernelContext_44);
        float3 _S543 = _S542.xyz;
        float4 _S544 = node_0(dn_3, 4U, &kernelContext_44);
        float3 _S545 = _S544.xyz;
        float4 _S546 = node_0(dn_3, 13U, &kernelContext_44);
        float3 vd_1 = _S543 + cross(_S545, _S522 - _S546.xyz);
        float4 _S547 = node_0(sn_3, 13U, &kernelContext_44);
        float3 q_5 = _S522 - _S547.xyz;
        float3 dv_2 = vd_1 - vs_1;
        put_0(_S535, 3U, r_4, 0U, ln_ratio_0(area_1, L_2 * L_2), &kernelContext_44);
        put_0(_S535, 3U, r_4, 1U, dot(ft1_3, q_5) / L_2, &kernelContext_44);
        put_0(_S535, 3U, r_4, 2U, dot(_S520, q_5) / L_2, &kernelContext_44);
        put_0(_S535, 3U, r_4, 3U, dot(fn_3, q_5) / L_2, &kernelContext_44);
        put_0(_S535, 3U, r_4, 4U, ln_ratio_0(_S509.y, _S524), &kernelContext_44);
        float _S548 = ln_ratio_0(_S509.z, _S524);
        put_0(_S535, 3U, r_4, 5U, _S548, &kernelContext_44);
        put_0(_S535, 3U, r_4, 6U, _S548, &kernelContext_44);
        put_0(_S535, 3U, r_4, 7U, ln_ratio_0(_S509.w, _S525), &kernelContext_44);
        put_0(_S535, 3U, r_4, 8U, ln_ratio_0(_S510.x, _S525), &kernelContext_44);
        put_0(_S535, 3U, r_4, 9U, ln_ratio_0(_S510.y, _S525), &kernelContext_44);
        put_0(_S535, 3U, r_4, 10U, ln_ratio_0(_S529, _S503), &kernelContext_44);
        put_0(_S535, 3U, r_4, 11U, ln_ratio_0(_S531, _S503), &kernelContext_44);
        put_0(_S535, 3U, r_4, 12U, ln_ratio_0(_S530, _S503), &kernelContext_44);
        put_0(_S535, 3U, r_4, 13U, ln_ratio_0(_S532 * area_1, _S504), &kernelContext_44);
        put_0(_S535, 3U, r_4, 14U, ln_ratio_0(_S533 * area_1, _S504), &kernelContext_44);
        put_0(_S535, 3U, r_4, 15U, ln_ratio_0(_S534 * area_1, _S504), &kernelContext_44);
        put_0(_S535, 3U, r_4, 16U, _S510.z, &kernelContext_44);
        put_0(_S535, 3U, r_4, 17U, _S526.damage_0, &kernelContext_44);
        put_0(_S535, 3U, r_4, 18U, _S526.crush_0, &kernelContext_44);
        put_0(_S535, 3U, r_4, 19U, _S528, &kernelContext_44);
        float _S549 = log(1.0f + max(_S526.utilization_0, 0.0f));
        put_0(_S535, 3U, r_4, 20U, _S549, &kernelContext_44);
        put_0(_S535, 3U, r_4, 21U, _S549, &kernelContext_44);
        float _S550;
        if(rebar_1)
        {
            _S550 = 1.0f;
        }
        else
        {
            _S550 = 0.0f;
        }
        put_0(_S535, 3U, r_4, 22U, _S550, &kernelContext_44);
        put_0(_S535, 3U, r_4, 23U, 0.0f, &kernelContext_44);
        float _S551;
        if(in_plug_0)
        {
            _S551 = 1.0f;
        }
        else
        {
            _S551 = 0.0f;
        }
        put_0(_S535, 3U, r_4, 24U, _S551, &kernelContext_44);
        put_0(_S535, 3U, r_4, 25U, ratio_1, &kernelContext_44);
        put_0(_S535, 3U, r_4, 26U, dot(ft1_3, dv_2) / _S502, &kernelContext_44);
        put_0(_S535, 3U, r_4, 27U, dot(_S520, dv_2) / _S502, &kernelContext_44);
        put_0(_S535, 3U, r_4, 28U, dot(fn_3, dv_2) / _S502, &kernelContext_44);
        float4 _S552 = node_0(sn_3, 1U, &kernelContext_44);
        put_rot6_0(_S535, 3U, r_4, 29U, ft1_3, _S520, quat_matrix_0(_S552), &kernelContext_44);
        float4 _S553 = node_0(dn_3, 1U, &kernelContext_44);
        put_rot6_0(_S535, 3U, r_4, 35U, ft1_3, _S520, quat_matrix_0(_S553), &kernelContext_44);
        put_0(_S535, 3U, r_4, 41U, 0.0f, &kernelContext_44);
        put_0(_S535, 3U, r_4, 42U, 0.0f, &kernelContext_44);
        put_0(_S535, 3U, r_4, 43U, 0.0f, &kernelContext_44);
        side_4 = side_4 + 1U;
    }
    return;
}

