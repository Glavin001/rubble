//! The joint law's GPU data (`shaders/joint.slang`): materials, per-bond law data and
//! history, converted from and to stress-ref's types.

use stress_ref::bond::{BondGeometry, BondStiffness};
use stress_ref::joint::{FailureMode, JointState, JointStrength, Local6Ser, RebarParams};
use stress_ref::material::BondKind;

pub const FLAG_SOFTENING: u32 = 1;
pub const FLAG_CRACK_CONTACT: u32 = 2;
pub const FLAG_EXACT_PATCH: u32 = 4;
pub const FLAG_PATCH_AFTER_DAMAGE: u32 = 8;
pub const FLAG_EXACT_RATE_FILTER: u32 = 16;
pub const FLAG_DIF: u32 = 32;
pub const FLAG_FATIGUE: u32 = 64;

/// `JointMaterial` of joint.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuJointMaterial {
    pub strength: [f32; 4],
    pub energy: [f32; 4],
    pub dif: [f32; 4],
    pub misc: [f32; 4],
    pub kind_flags: [u32; 4],
}

/// `JointBond` of joint.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuJointBond {
    pub geom0: [f32; 4],
    pub geom1: [f32; 4],
    pub stiff0: [f32; 4],
    pub stiff1: [f32; 4],
    pub rebar0: [f32; 4],
    pub rebar1: [f32; 4],
    pub ids: [u32; 4],
}

/// `JointState` of joint.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuJointState {
    pub damage: f32,
    pub crush: f32,
    pub kappa: f32,
    pub kappa_c: f32,
    pub ductility: f32,
    pub ductility_c: f32,
    /// Static fatigue as the remaining life `1 - omega` (f32 resolves it near failure,
    /// where `omega` itself would stall an ulp short of 1).
    pub life: f32,
    pub plastic_x: f32,
    pub plastic_y: f32,
    pub plastic_t: f32,
    pub rebar_plastic: f32,
    pub rebar_slip0: f32,
    pub rebar_slip1: f32,
    pub rebar_work: f32,
    pub rebar_broken: f32,
    pub strain_rate: f32,
    pub governing_stress: f32,
    pub dissipated: f32,
    pub utilization: f32,
    pub mode: u32,
}

/// The per-material part of a bond's strength.
pub fn material_of(s: &JointStrength) -> GpuJointMaterial {
    let mut flags = 0;
    for (on, flag) in [
        (s.softening, FLAG_SOFTENING),
        (s.crack_contact, FLAG_CRACK_CONTACT),
        (s.exact_patch, FLAG_EXACT_PATCH),
        (s.patch_after_damage, FLAG_PATCH_AFTER_DAMAGE),
        (s.exact_rate_filter, FLAG_EXACT_RATE_FILTER),
        (s.dif.is_some(), FLAG_DIF),
        (s.static_fatigue.is_some(), FLAG_FATIGUE),
    ] {
        if on {
            flags |= flag;
        }
    }
    let dif = s.dif.unwrap_or(stress_ref::material::DifParams { reference_rate: 1.0, exponent: 0.0, transition_rate: 1.0, exponent_high: 0.0, max: 1.0 });
    let (fat_n, fat_t) = s.static_fatigue.map(|f| (f.exponent, f.test_time)).unwrap_or((3.0, 1.0));
    GpuJointMaterial {
        strength: [s.tensile as f32, s.compressive as f32, s.cohesion as f32, s.friction as f32],
        energy: [s.shear_cap as f32, s.g_tension as f32, s.g_shear as f32, s.g_compression as f32],
        dif: [dif.reference_rate as f32, dif.exponent as f32, dif.transition_rate as f32, dif.exponent_high as f32],
        misc: [dif.max as f32, fat_n as f32, fat_t as f32, s.youngs_modulus as f32],
        kind_flags: [if s.kind == BondKind::Ductile { 1 } else { 0 }, flags, 0, 0],
    }
}

/// Deduplicated materials, in first-use order.
#[derive(Default)]
pub struct MaterialTable {
    pub materials: Vec<GpuJointMaterial>,
}

impl MaterialTable {
    pub fn index(&mut self, s: &JointStrength) -> u32 {
        let m = material_of(s);
        if let Some(i) = self.materials.iter().position(|x| *x == m) {
            return i as u32;
        }
        self.materials.push(m);
        (self.materials.len() - 1) as u32
    }
}

/// The per-bond law data.
pub fn bond_law(g: &BondGeometry, k: &BondStiffness, s: &JointStrength, rebar: Option<&RebarParams>, weibull: f64, material: u32) -> GpuJointBond {
    let r = rebar.copied().unwrap_or(RebarParams { k_axial: 0.0, k_dowel: 0.0, yield_force: 0.0, dowel_capacity: 0.0, rupture_work: 0.0 });
    GpuJointBond {
        geom0: [g.area as f32, g.width[0] as f32, g.width[1] as f32, g.torsion_modulus as f32],
        geom1: [g.s_t1 as f32, g.s_t2 as f32, g.friction_radius as f32, weibull as f32],
        stiff0: [k.kn as f32, k.ks as f32, k.kb_t1 as f32, k.kb_t2 as f32],
        stiff1: [k.kt as f32, s.buckling_load.unwrap_or(0.0) as f32, s.rate_filter_time as f32, if rebar.is_some() { 1.0 } else { 0.0 }],
        rebar0: [r.k_axial as f32, r.k_dowel as f32, r.yield_force as f32, r.dowel_capacity as f32],
        rebar1: [r.rupture_work as f32, 0.0, 0.0, 0.0],
        ids: [material, 0, 0, 0],
    }
}

fn mode_code(m: Option<FailureMode>) -> u32 {
    match m {
        None => 0,
        Some(FailureMode::Tension) => 1,
        Some(FailureMode::Shear) => 2,
        Some(FailureMode::Compression) => 3,
        Some(FailureMode::Buckling) => 4,
    }
}

pub fn mode_of(code: u32) -> Option<FailureMode> {
    match code {
        1 => Some(FailureMode::Tension),
        2 => Some(FailureMode::Shear),
        3 => Some(FailureMode::Compression),
        4 => Some(FailureMode::Buckling),
        _ => None,
    }
}

pub fn to_gpu_state(s: &JointState) -> GpuJointState {
    GpuJointState {
        damage: s.damage as f32,
        crush: s.crush as f32,
        kappa: s.kappa as f32,
        kappa_c: s.kappa_c as f32,
        ductility: s.ductility as f32,
        ductility_c: s.ductility_c as f32,
        life: (1.0 - s.fatigue) as f32,
        plastic_x: s.plastic.lin[0] as f32,
        plastic_y: s.plastic.lin[1] as f32,
        plastic_t: s.plastic.ang[2] as f32,
        rebar_plastic: s.rebar_plastic as f32,
        rebar_slip0: s.rebar_slip[0] as f32,
        rebar_slip1: s.rebar_slip[1] as f32,
        rebar_work: s.rebar_work as f32,
        rebar_broken: if s.rebar_broken { 1.0 } else { 0.0 },
        strain_rate: s.strain_rate as f32,
        governing_stress: s.governing_stress as f32,
        dissipated: s.dissipated as f32,
        utilization: s.utilization as f32,
        mode: mode_code(s.mode),
    }
}

pub fn from_gpu_state(g: &GpuJointState, previous: &JointState) -> JointState {
    JointState {
        damage: g.damage as f64,
        crush: g.crush as f64,
        kappa: g.kappa as f64,
        kappa_c: g.kappa_c as f64,
        ductility: g.ductility as f64,
        ductility_c: g.ductility_c as f64,
        fatigue: 1.0 - g.life as f64,
        plastic: Local6Ser { lin: [g.plastic_x as f64, g.plastic_y as f64, previous.plastic.lin[2]], ang: [previous.plastic.ang[0], previous.plastic.ang[1], g.plastic_t as f64] },
        rebar_plastic: g.rebar_plastic as f64,
        rebar_slip: [g.rebar_slip0 as f64, g.rebar_slip1 as f64],
        rebar_work: g.rebar_work as f64,
        rebar_broken: g.rebar_broken != 0.0,
        strain_rate: g.strain_rate as f64,
        governing_stress: g.governing_stress as f64,
        mode: mode_of(g.mode),
        dissipated: g.dissipated as f64,
        utilization: g.utilization as f64,
    }
}
