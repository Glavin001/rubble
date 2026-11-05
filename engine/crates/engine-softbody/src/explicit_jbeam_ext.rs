//! JBeam mode, Part II (`engine/docs/jbeam-semantics-spec.md` §16–§27): the
//! features of BeamNG's JBeam the explicit solver runs through extension records.
//! Every function here is reached only from a beam with an extension record
//! ([`ExplicitBeams::ext_of`]), a vehicle with breakable triangles
//! ([`ExplicitBeams::tri_breaks`]) or a spawn that asks for one of these features:
//! a vehicle that uses none of them runs exactly the code it ran before.
//!
//! The GPU runs the same rules (`slang/kernels/explicit_beams.slang`,
//! `jbeamExtUpdate` and friends, host `engine_webgpu::ExplicitBeamsGpu`) on the same
//! records: [`BeamExt`] is laid out as the kernel's four static records and one
//! state record.
use crate::desc::{BeamKind, JbeamBeam, JbeamPressureWheel, SoftVehicleDesc};
use crate::explicit_beams::{
    Beam, BeamOut, DampFilter, ExplicitBeams, ExplicitKind, ExplicitType, Node, Tyre, MIN_REST,
    NODE_CONTACTER, NODE_TYRE, STEP,
};
use glam::Vec3;
use std::collections::HashMap;

/// How near a collision triangle's plane a self-collision node lies "on" it at
/// spawn (spec §10.1): 1 mm, the precision JBeam positions are written to.
pub const SURFACE_TOLERANCE: f32 = 0.001;
/// Ambient pressure (Pa): PRESSURED beams and pressure groups take gauge pressures
/// over it (spec §21, §26).
pub const ATMOSPHERE: f32 = 101_325.0;
/// One pound per square inch in pascals.
pub const PSI: f32 = 6894.757;
/// Extension record feature bits (E0 .x).
pub const EXT_GROUPS: u32 = 1;
pub const EXT_DEFORM: u32 = 2;
pub const EXT_REBOUND: u32 = 4;
pub const EXT_RAMP: u32 = 8;
/// The GPU type byte's flag of a beam with an extension record.
pub const TYPE_EXT: u32 = 4;
/// No beam (triangle break lists).
pub const NO_BEAM: u32 = u32::MAX;

/// A beam's extension record (spec §16). E0 = (mask, deform groups, trigger ratio),
/// E1 and E2 the kind's parameters (index fields as `f32` bits), E3 = (precompression
/// ramp step, ramp steps, extra break groups); then the state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeamExt {
    /// The beam (vehicle-local index).
    pub beam: u32,
    pub mask: u32,
    /// Deform groups (state slot + 1, 0 none) and `deformationTriggerRatio`.
    pub deform: [u32; 2],
    pub trigger: f32,
    pub e1: [f32; 4],
    pub e2: [f32; 4],
    pub ramp_delta: f32,
    pub ramp_steps: u32,
    /// The second and third break groups, signed as the beam's own (0 none).
    pub extra_groups: [i32; 2],
    // State.
    pub ramp_left: u32,
    /// Broke on its own with several groups: sets them off in the next beam pass.
    pub pending: bool,
    /// Pressure groups: gauge pressure by step parity (Pa).
    pub pressure: [f32; 2],
}

impl BeamExt {
    /// The record's E1 .x as an index.
    pub fn index(&self, k: usize) -> u32 {
        self.e1[k].to_bits()
    }
}

fn bits(u: u32) -> f32 {
    f32::from_bits(u)
}

/// The JBeam yield rule (spec §3) shared with `jbeam_force_filtered`.
fn jbeam_yield(beam: &mut Beam, len: f32, e: f32, k: f32, damp: f32, mut f: f32) -> (f32, f32) {
    let mut tau = f.abs();
    if tau > beam.gate && beam.ty == ExplicitType::Deformable && k != 0.0 {
        if f > beam.tc && e < 0.0 {
            let delta = e + (beam.tc / k) * (1.0 - beam.plastic);
            let old = beam.rest;
            let floor = beam.short_bound.max(MIN_REST);
            let free = beam.rest + delta;
            if free < floor {
                beam.rest = floor.min(old);
                f = -k * (len - beam.rest) - damp;
            } else {
                beam.rest = free;
                f -= (f - beam.tc) / 2.0;
            }
            tau = f.abs();
            if beam.rest < old {
                beam.tc = (beam.tc * old / beam.rest).min(beam.limit_k.max(beam.tc));
                beam.gate = beam.tc.min(beam.tt).min(beam.strength);
            }
        } else if f < -beam.tt && e > 0.0 {
            let delta = e - (beam.tt / k) * (1.0 - beam.plastic);
            let old = beam.rest;
            let ceiling = beam.long_bound;
            let free = beam.rest + delta;
            if free > ceiling {
                beam.rest = ceiling.max(old);
                f = -k * (len - beam.rest) - damp;
            } else {
                beam.rest = free;
                f -= (f + beam.tt) / 2.0;
            }
            tau = f.abs();
            if beam.rest > old {
                beam.tt = (beam.tt * beam.rest / old).min(beam.limit_k.max(beam.tt));
                beam.gate = beam.tc.min(beam.tt).min(beam.strength);
                beam.strength -= (beam.rest - old) * k;
            }
        }
    }
    (f, tau)
}

/// The force of a JBeam beam with an extension record (kinds `JbeamPlain`,
/// `JbeamSupport`, `JbeamBounded`, `JbeamAniso`, `JbeamPressured`): §2–§5 with the
/// precompression ramp (§23), rebound damping (§22), the anisotropic rates (§20) and
/// the pressured cylinder (§21). Breaks past a length mark the beam broken with no
/// force (the caller sets off its groups); the strength test is `check_break`.
pub fn jbeam_ext_force(
    beam: &mut Beam,
    ext: &mut BeamExt,
    na: &Node,
    nb: &Node,
    filter: Option<&mut DampFilter>,
) -> BeamOut {
    if ext.ramp_left > 0 {
        beam.rest += ext.ramp_delta;
        ext.ramp_left -= 1;
    }
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let inv = 1.0 / q.sqrt();
    let len = q * inv;
    let e = len - beam.rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let u = match filter {
        Some(f) => f.step(u),
        None => u,
    };
    let mut k = beam.k;
    let c = beam.c;
    let mut damp = c * u;
    let broke = |beam: &mut Beam| {
        beam.broken = true;
        beam.force = 0.0;
        BeamOut {
            d,
            inv,
            e,
            force: 0.0,
            check_break: false,
        }
    };
    match beam.kind {
        ExplicitKind::JbeamBounded => {
            // Compression or expansion side (§22).
            let (cs, cf, split, lc) = if ext.mask & EXT_REBOUND != 0 && u > 0.0 {
                (ext.e2[0], ext.e2[1], ext.e2[2], ext.e2[3])
            } else {
                (c, beam.tt, beam.tc, beam.limit_c)
            };
            damp = cs * u;
            let au = u.abs();
            if au > split {
                damp = (cs * split + cf * (au - split)).copysign(u);
            }
            let long = beam.long_bound * beam.rest;
            let short = beam.short_bound * beam.rest;
            let o = if e > long {
                e - long
            } else if e < -short {
                -e - short
            } else {
                0.0
            };
            if o > 0.0 {
                let zone = beam.plastic;
                let s = if zone > 0.0 { (o / zone).min(1.0) } else { 1.0 };
                k += (beam.limit_k - k) * s;
                damp += (lc * u - damp) * s;
            }
        }
        ExplicitKind::JbeamSupport => {
            if e > 0.0 {
                if len > beam.long_bound {
                    return broke(beam);
                }
                k = 0.0;
                damp = 0.0;
            }
        }
        ExplicitKind::JbeamAniso => {
            if len > ext.e1[3] {
                return broke(beam);
            }
            if e > 0.0 {
                let zone = ext.e1[2] * beam.rest;
                let s = if zone > 0.0 { (e / zone).min(1.0) } else { 1.0 };
                k += (ext.e1[0] - k) * s;
                damp = (c + (ext.e1[1] - c) * s) * u;
            }
        }
        ExplicitKind::JbeamPressured => {
            let l0 = ext.e1[3];
            let den = (l0 - ext.e1[2] * (l0 - len)).max(0.01 * l0);
            let gauge =
                (ext.e1[1] * (l0 / den) - crate::explicit_jbeam_ext::ATMOSPHERE).min(ext.e2[0]);
            let f = gauge * ext.e1[0] - k * e.max(0.0) - damp;
            beam.force = f;
            return BeamOut {
                d,
                inv,
                e,
                force: f,
                check_break: gauge > ext.e2[1] || f.abs() > beam.strength,
            };
        }
        _ => {}
    }
    let f = -k * e - damp;
    beam.force = f;
    let (f, tau) = jbeam_yield(beam, len, e, k, damp, f);
    BeamOut {
        d,
        inv,
        e,
        force: f,
        check_break: tau > beam.strength,
    }
}

/// An LBEAM's force (spec §24) between `na` (id1) and `nb` (id2) with vertex `nc`:
/// a beam whose rest length is the law-of-cosines length of the current arms at the
/// rest angle (`rest` holds cos θ₀); past the deform force (`tc`) the rest angle
/// follows so the spring force stays at it.
pub fn lbeam_force(
    beam: &mut Beam,
    na: &Node,
    nb: &Node,
    nc: &Node,
    filter: Option<&mut DampFilter>,
) -> BeamOut {
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let inv = 1.0 / q.sqrt();
    let len = q * inv;
    let ra = na.pos - nc.pos;
    let rb = nb.pos - nc.pos;
    let a2 = ra.dot(ra);
    let b2 = rb.dot(rb);
    let ab = (a2 * b2).sqrt();
    let rest = (a2 + b2 - 2.0 * ab * beam.rest).max(0.0).sqrt();
    let e = len - rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let u = match filter {
        Some(f) => f.step(u),
        None => u,
    };
    let spring = beam.k * e;
    let mut f = -spring - beam.c * u;
    if spring.abs() > beam.tc && ab > 1e-12 {
        let m = (beam.tc / beam.k).copysign(e);
        let l = len - m;
        beam.rest = ((a2 + b2 - l * l) / (2.0 * ab)).clamp(-1.0, 1.0);
        f = -beam.tc.copysign(e) - beam.c * u;
    }
    beam.force = f;
    BeamOut {
        d,
        inv,
        e,
        force: f,
        check_break: f.abs() > beam.strength,
    }
}

/// A pressure group's gauge pressure from its triangle records (spec §26): the
/// enclosed volume about node `c`, isothermal from (V₀, absolute p₀), capped.
pub fn pressure_group_gauge(beams: &[Beam], nodes: &[Node], ext: &BeamExt, c: Vec3) -> f32 {
    let first = ext.index(0) as usize;
    let count = ext.index(1) as usize;
    let mut v = 0.0f32;
    for t in 0..count {
        let r = first + 2 * t;
        let x0 = nodes[beams[r].a as usize].pos - c;
        let x1 = nodes[beams[r].b as usize].pos - c;
        let x2 = nodes[beams[r + 1].a as usize].pos - c;
        v += x0.dot(x1.cross(x2));
    }
    let v = (v / 6.0).max(0.01 * ext.e1[2]);
    (ext.e1[3] * (ext.e1[2] / v) - ATMOSPHERE).min(ext.e2[0])
}

/// A pressure triangle's push on each of its three nodes at gauge pressure `p`.
pub fn pressure_tri_force(x0: Vec3, x1: Vec3, x2: Vec3, p: f32) -> Vec3 {
    (x1 - x0).cross(x2 - x0) * (p / 6.0)
}

/// A ground contact force `c` on a node (force `f` before it, velocity `v`, mass
/// `m`, normal `n`, depth `p`) with its normal part limited so the node rises at
/// most half its depth in one step `h`, and its tangential part so it at most
/// stops the node's slip in one step (spec §28): on a light, heavily loaded node
/// the explicit laws otherwise overshoot every step.
pub fn friction_limit(c: Vec3, f: Vec3, v: Vec3, m: f32, n: Vec3, h: f32, p: f32) -> Vec3 {
    // The node leaves at most half its depth `p` in one step: the solver's pop-out
    // (0.2 of the depth per step, as a velocity the node keeps) otherwise throws a
    // heavily loaded light node clear of the ground, and its load drives it back
    // in at metres per second the next step. Half keeps it in contact.
    let mut c = c;
    let cn = c.dot(n);
    let out = v.dot(n) + (f.dot(n) + cn) * (h / m);
    let most = 0.5 * p / h;
    if cn > 0.0 && out > most {
        let capped = ((most - v.dot(n)) * (m / h) - f.dot(n)).max(0.0);
        c += n * (capped - cn);
    }
    let total = f + c;
    let t = total - total.dot(n) * n;
    let vt = v - v.dot(n) * n;
    let dv = t * (h / m);
    // Decelerating past zero slip: the tangential force that stops it exactly.
    if dv.dot(vt) < 0.0 && dv.length_squared() > vt.length_squared() {
        c + (-vt * (m / h) - t)
    } else {
        c
    }
}

impl ExplicitBeams {
    fn ext_index(&self, j: usize) -> usize {
        self.ext_of[j] as usize - 1
    }

    /// Mark deform groups of extension record `x` triggered.
    fn trigger_deform(&mut self, x: usize) {
        for g in self.beam_ext[x].deform {
            if g != 0 {
                self.deform_state[g as usize - 1] = 1;
            }
        }
    }

    /// The triggered deform groups (spec §19): per group (names in
    /// [`Self::deform_names`]) 0 or 1.
    pub fn deform_triggered(&self) -> &[u32] {
        &self.deform_state
    }

    /// The members of group `g` by extension (their extra groups) are broken, and
    /// every broken member with a deform group triggers it.
    pub(crate) fn cascade_ext(&mut self, g: i32) {
        for x in 0..self.beam_ext.len() {
            let ext = self.beam_ext[x];
            let j = ext.beam as usize;
            let member =
                self.beams[j].group.abs() == g || ext.extra_groups.iter().any(|&e| e.abs() == g);
            if !member {
                continue;
            }
            self.beams[j].broken = true;
            if ext.mask & EXT_DEFORM != 0 {
                self.trigger_deform(x);
            }
        }
    }

    /// Triangles whose listed beams are broken die (spec §18); at the start of the
    /// beam pass.
    pub(crate) fn triangle_breaks(&mut self) {
        for t in 0..self.tri_breaks.len() {
            if self.tri_dead[t] {
                continue;
            }
            self.tri_dead[t] = self.tri_breaks[t]
                .iter()
                .any(|&j| j != NO_BEAM && self.beams[j as usize].broken);
        }
    }

    /// A broken beam with an extension record in the beam pass: a pending break of
    /// several groups sets them off now (spec §17).
    pub(crate) fn ext_broken(&mut self, j: usize) {
        let x = self.ext_index(j);
        if !self.beam_ext[x].pending {
            return;
        }
        self.beam_ext[x].pending = false;
        let own = self.beams[j].group;
        let extra = self.beam_ext[x].extra_groups;
        for g in [own, extra[0], extra[1]] {
            if g > 0 {
                self.cascade(g);
            }
        }
    }

    /// A live beam with an extension record in the beam pass.
    pub(crate) fn ext_beam(&mut self, j: usize) {
        let x = self.ext_index(j);
        let mut beam = self.beams[j];
        match beam.kind {
            ExplicitKind::JbeamPressureTri2 => return,
            ExplicitKind::JbeamPressureGroup => {
                let c = self.nodes[beam.a as usize].pos;
                let p = pressure_group_gauge(&self.beams, &self.nodes, &self.beam_ext[x], c);
                self.beam_ext[x].pressure[(self.steps & 1) as usize] = p;
                return;
            }
            ExplicitKind::JbeamPressureTri => {
                let ext = self.beam_ext[x];
                let group = ext.index(1) as usize;
                let p =
                    self.beam_ext[self.ext_index(group)].pressure[((self.steps & 1) ^ 1) as usize];
                let n2 = ext.index(0);
                let (x0, x1, x2) = (
                    self.nodes[beam.a as usize].pos,
                    self.nodes[beam.b as usize].pos,
                    self.nodes[n2 as usize].pos,
                );
                let f = pressure_tri_force(x0, x1, x2, p);
                self.nodes[beam.a as usize].force += f;
                self.nodes[beam.b as usize].force += f;
                self.nodes[n2 as usize].force += f;
                return;
            }
            _ => {}
        }
        let na = self.nodes[beam.a as usize];
        let nb = self.nodes[beam.b as usize];
        let filter = match beam.shock {
            0 => None,
            i => Some(&mut self.damp_filters[i as usize - 1]),
        };
        let out = if beam.kind == ExplicitKind::JbeamLbeam {
            let nc = self.nodes[self.beam_ext[x].index(0) as usize];
            lbeam_force(&mut beam, &na, &nb, &nc, filter)
        } else {
            jbeam_ext_force(&mut beam, &mut self.beam_ext[x], &na, &nb, filter)
        };
        let mut f = out.force;
        let mut broke_now = beam.broken;
        if out.check_break {
            beam.broken = true;
            f = 0.0;
            broke_now = true;
        }
        self.beams[j] = beam;
        let ext = self.beam_ext[x];
        // Yield moves only plain, support and anisotropic rest lengths (E2 .x holds
        // the spawn rest length of those); not while the precompression ramps in.
        let yields = matches!(
            beam.kind,
            ExplicitKind::JbeamPlain | ExplicitKind::JbeamSupport | ExplicitKind::JbeamAniso
        );
        if ext.mask & EXT_DEFORM != 0
            && (broke_now
                || (yields
                    && ext.ramp_left == 0
                    && (beam.rest - ext.e2[0]).abs() > ext.trigger * ext.e2[0]))
        {
            self.trigger_deform(x);
        }
        if broke_now && beam.group > 0 {
            if ext.mask & EXT_GROUPS != 0 {
                self.beam_ext[x].pending = true;
            } else {
                self.cascade(beam.group);
            }
        }
        let push = out.d * (f * out.inv);
        self.nodes[beam.a as usize].force += push;
        self.nodes[beam.b as usize].force -= push;
    }

    /// Give beam `j` (just pushed by `push_jbeam_beam`) its Part II features
    /// (spec §16–§23); `spawn` is its nodes' spawn distance.
    pub(crate) fn attach_ext(&mut self, j: usize, jb: &JbeamBeam, spawn: f32) {
        let beam = &mut self.beams[j];
        if beam.kind == ExplicitKind::JbeamSupport {
            if let Some(x) = jb.long_extent {
                beam.long_bound = x;
            }
        }
        if !jb.needs_ext() {
            return;
        }
        let mut ext = BeamExt {
            beam: j as u32,
            ..Default::default()
        };
        if jb.extra_break_groups != [0, 0] && beam.group != 0 {
            ext.mask |= EXT_GROUPS;
            let sign = beam.group.signum();
            ext.extra_groups = jb.extra_break_groups.map(|g| sign * g as i32);
        }
        if jb.deform_groups != [0, 0] {
            ext.mask |= EXT_DEFORM;
            ext.deform = jb.deform_groups;
            ext.trigger = jb.deformation_trigger_ratio;
        }
        if jb.anisotropic {
            beam.kind = ExplicitKind::JbeamAniso;
            let length = match jb.long_extent {
                Some(x) => x,
                None => (1.0 + jb.long_bound) * spawn,
            };
            ext.e1 = [
                jb.spring_expansion.unwrap_or(beam.k),
                jb.damp_expansion.unwrap_or(beam.c),
                jb.transition_zone,
                length,
            ];
        }
        if jb.pressured {
            beam.kind = ExplicitKind::JbeamPressured;
            beam.ty = ExplicitType::Actuator;
            ext.e1 = [jb.surface, ATMOSPHERE + jb.pressure, jb.volume_coef, spawn];
            ext.e2 = [jb.pressure_limit, jb.max_pressure, 0.0, 0.0];
        }
        if beam.kind == ExplicitKind::JbeamBounded
            && (jb.damp_rebound.is_some()
                || jb.damp_rebound_fast.is_some()
                || jb.damp_velocity_split_rebound.is_some()
                || jb.limit_damp_rebound.is_some())
        {
            ext.mask |= EXT_REBOUND;
            let slow = jb.damp_rebound.unwrap_or(beam.c);
            ext.e2 = [
                slow,
                jb.damp_rebound_fast.unwrap_or(slow),
                jb.damp_velocity_split_rebound.unwrap_or(beam.tc),
                jb.limit_damp_rebound.unwrap_or(beam.limit_c),
            ];
        }
        if jb.precompression_time > 0.0 {
            let n = (jb.precompression_time / STEP).round().max(1.0) as u32;
            ext.mask |= EXT_RAMP;
            ext.ramp_steps = n;
            ext.ramp_left = n;
            ext.ramp_delta = (beam.rest - spawn) / n as f32;
            beam.rest = spawn;
        }
        if matches!(
            beam.kind,
            ExplicitKind::JbeamPlain | ExplicitKind::JbeamSupport | ExplicitKind::JbeamAniso
        ) {
            ext.e2[0] = beam.rest0;
        }
        self.push_ext(j, ext);
    }

    fn push_ext(&mut self, j: usize, ext: BeamExt) {
        self.beam_ext.push(ext);
        if self.ext_of.len() <= j {
            self.ext_of.resize(j + 1, 0);
        }
        self.ext_of[j] = self.beam_ext.len() as u32;
    }

    /// The desc's LBEAMs (spec §24): id1, id2, vertex id3.
    pub(crate) fn add_lbeams(&mut self, desc: &SoftVehicleDesc) {
        for e in desc
            .elements
            .iter()
            .filter(|e| e.kind == crate::desc::ElementKind::Lbeam)
        {
            let [a, b, c, _] = e.nodes;
            self.push_lbeam(a, b, c, e.spring, e.damp, e.deform, e.strength, 0.0, 1.0);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_lbeam(
        &mut self,
        a: u32,
        b: u32,
        c: u32,
        k: f32,
        damp: f32,
        deform: f32,
        strength: f32,
        cutoff: f32,
        precompression: f32,
    ) {
        let (xa, xb, xc) = (
            self.nodes[a as usize].pos,
            self.nodes[b as usize].pos,
            self.nodes[c as usize].pos,
        );
        let ra = xa - xc;
        let rb = xb - xc;
        let (la, lb) = (ra.length(), rb.length());
        let l = xa.distance(xb) * precompression;
        let cos0 = if la * lb > 1e-9 {
            ((la * la + lb * lb - l * l) / (2.0 * la * lb)).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let mut beam = Beam::new(
            a,
            b,
            ExplicitKind::JbeamLbeam,
            ExplicitType::Actuator,
            k,
            damp,
            cos0,
            f32::INFINITY,
            strength,
            0.0,
        );
        beam.tc = deform;
        if cutoff > 0.0 {
            self.damp_filters.push(DampFilter::new(cutoff));
            beam.shock = self.damp_filters.len() as u32;
        }
        self.beams.push(beam);
        let j = self.beams.len() - 1;
        let ext = BeamExt {
            beam: j as u32,
            e1: [bits(c), 0.0, 0.0, 0.0],
            ..Default::default()
        };
        self.push_ext(j, ext);
    }

    /// A pressure group over `tris` (outward-wound node triples) at gauge `p0`
    /// (Pa), the gauge never above `ceiling` (spec §26).
    pub(crate) fn push_pressure_group(&mut self, tris: &[[u32; 3]], p0: f32, ceiling: f32) {
        let g = self.beams.len();
        let c = tris[0][0];
        let mut group = Beam::new(
            c,
            c,
            ExplicitKind::JbeamPressureGroup,
            ExplicitType::Actuator,
            0.0,
            0.0,
            0.0,
            f32::INFINITY,
            f32::INFINITY,
            0.0,
        );
        group.rest0 = 0.0;
        self.beams.push(group);
        let first = self.beams.len() as u32;
        let cpos = self.nodes[c as usize].pos;
        // Outward winding whatever order the surface came in (a wheel's follows its
        // axle nodes' order): a surface enclosing negative volume is turned over.
        let signed: f32 = tris
            .iter()
            .map(|t| {
                let x = t.map(|n| self.nodes[n as usize].pos - cpos);
                x[0].dot(x[1].cross(x[2]))
            })
            .sum();
        let mut v0 = 0.0f32;
        for t in tris {
            let t = if signed < 0.0 { &[t[0], t[2], t[1]] } else { t };
            let x = t.map(|n| self.nodes[n as usize].pos - cpos);
            v0 += x[0].dot(x[1].cross(x[2]));
            let mut r1 = Beam::new(
                t[0],
                t[1],
                ExplicitKind::JbeamPressureTri,
                ExplicitType::Actuator,
                0.0,
                0.0,
                0.0,
                f32::INFINITY,
                f32::INFINITY,
                0.0,
            );
            r1.rest0 = 0.0;
            let mut r2 = r1;
            r2.kind = ExplicitKind::JbeamPressureTri2;
            r2.a = t[2];
            r2.b = t[2];
            self.beams.push(r1);
            let j = self.beams.len() - 1;
            self.push_ext(
                j,
                BeamExt {
                    beam: j as u32,
                    e1: [bits(t[2]), bits(g as u32), 0.0, 0.0],
                    ..Default::default()
                },
            );
            self.beams.push(r2);
            let j = self.beams.len() - 1;
            self.push_ext(
                j,
                BeamExt {
                    beam: j as u32,
                    ..Default::default()
                },
            );
        }
        let v0 = (v0 / 6.0).max(1e-9);
        let ext = BeamExt {
            beam: g as u32,
            e1: [bits(first), bits(tris.len() as u32), v0, ATMOSPHERE + p0],
            e2: [ceiling, 0.0, 0.0, 0.0],
            pressure: [p0.min(ceiling); 2],
            ..Default::default()
        };
        self.push_ext(g, ext);
    }

    /// Break lists for the collision triangles (spec §18), in [`Self::triangles`]
    /// order; none when no triangle can break.
    pub(crate) fn add_triangle_breaks(
        &mut self,
        desc: &SoftVehicleDesc,
        warnings: &mut Vec<String>,
    ) {
        let mut side: HashMap<(u32, u32), u32> = HashMap::new();
        for (i, b) in desc.beams.iter().enumerate() {
            if b.jbeam.is_some_and(|j| j.disable_triangle_breaking) {
                continue;
            }
            side.entry((b.a.min(b.b), b.a.max(b.b))).or_insert(i as u32);
        }
        let mut rep: HashMap<u32, u32> = HashMap::new();
        for (i, b) in self.beams.iter().enumerate() {
            let single = self
                .ext_of
                .get(i)
                .is_none_or(|&x| x == 0 || self.beam_ext[x as usize - 1].mask & EXT_GROUPS == 0);
            if b.group > 0 && single {
                rep.entry(b.group as u32).or_insert(i as u32);
            }
        }
        let mut lists = Vec::new();
        let mut any = false;
        let mut no_rep = 0;
        for t in desc.triangles.iter().filter(|t| t.collision) {
            let [a, b, c] = t.nodes;
            let mut l = [NO_BEAM; 4];
            for (k, (p, q)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
                if let Some(&j) = side.get(&(p.min(q), p.max(q))) {
                    l[k] = j;
                }
            }
            if t.break_group != 0 {
                match rep.get(&t.break_group) {
                    Some(&j) => l[3] = j,
                    None => no_rep += 1,
                }
            }
            any |= l.iter().any(|&j| j != NO_BEAM);
            lists.push(l);
        }
        if no_rep > 0 {
            warnings.push(format!(
                "{no_rep} triangles' break groups have no member that sets them off: \
                 those triangles break by their sides only"
            ));
        }
        if any {
            self.tri_dead = vec![false; lists.len()];
            self.tri_breaks = lists;
        }
    }

    /// Self-collision nodes lying on a collision triangle at spawn (spec §10.1):
    /// inside it and within [`SURFACE_TOLERANCE`] of its plane. Such a node has
    /// no side to be pushed out to; the triangle takes the node's self-collision
    /// group (when the triangle has none), so the node skips it.
    pub(crate) fn surface_nodes(&mut self, warnings: &mut Vec<String>) {
        let mut groups = vec![0u32; self.triangles.len()];
        let (mut taken, mut left) = (0, 0);
        for t in 0..self.triangles.len() {
            if self.triangle_self_group(t) != 0 {
                continue;
            }
            let [i0, i1, i2] = self.triangles[t].nodes;
            let p0 = self.nodes[i0 as usize].pos;
            let (e1, e2) = (
                self.nodes[i1 as usize].pos - p0,
                self.nodes[i2 as usize].pos - p0,
            );
            let n = e1.cross(e2).normalize_or_zero();
            let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
            let det = d11 * d22 - d12 * d12;
            if n == Vec3::ZERO || det <= 0.0 {
                continue;
            }
            for &c in &self.candidates {
                if c == i0 || c == i1 || c == i2 {
                    continue;
                }
                let q = self.nodes[c as usize].pos - p0;
                let delta = n.dot(q);
                if delta.abs() > SURFACE_TOLERANCE {
                    continue;
                }
                let (q1, q2) = (q.dot(e1), q.dot(e2));
                let a = (d22 * q1 - d12 * q2) / det;
                let b = (d11 * q2 - d12 * q1) / det;
                if a < 0.0 || b < 0.0 || a + b > 1.0 {
                    continue;
                }
                let g = self.self_groups[c as usize];
                if g != 0 && (groups[t] == 0 || groups[t] == g) {
                    groups[t] = g;
                    taken += 1;
                } else {
                    left += 1;
                }
            }
        }
        if taken > 0 {
            self.tri_groups = groups;
            warnings.push(format!(
                "{taken} self-collision nodes lie on a collision triangle at spawn: \
                 those triangles take the node's group (spec §10.1)"
            ));
        }
        if left > 0 {
            warnings.push(format!(
                "{left} self-collision nodes lie on a collision triangle at spawn \
                 with no group to give it: they still collide with it"
            ));
        }
    }

    /// The desc's pressure wheels (spec §25), after its nodes and beams; returns the
    /// warnings for what is not simulated.
    pub(crate) fn add_pressure_wheels(
        &mut self,
        desc: &SoftVehicleDesc,
        velocity: Vec3,
    ) -> Vec<String> {
        let mut warnings = Vec::new();
        let mut unsimulated: std::collections::BTreeSet<&str> = Default::default();
        for pw in &desc.pressure_wheels {
            for key in [
                "conicityFactor",
                "hubWeightGainRatio",
                "brakeTorque",
                "noLoadCoef",
                "enableHubcaps",
            ] {
                if pw.values.contains_key(key) || pw.flags.get(key) == Some(&true) {
                    unsimulated.insert(key);
                }
            }
            self.add_pressure_wheel(pw, velocity);
        }
        if !desc.pressure_wheels.is_empty() {
            warnings.push(format!(
                "{} pressure wheels simulated as node rings with internal pressure",
                desc.pressure_wheels.len()
            ));
        }
        for k in unsimulated {
            warnings.push(format!("pressure wheels: `{k}` is not simulated"));
        }
        warnings
    }

    fn add_pressure_wheel(&mut self, pw: &JbeamPressureWheel, velocity: Vec3) {
        let v = |k: &str, d: f32| pw.values.get(k).copied().unwrap_or(d);
        let flag = |k: &str, d: bool| pw.flags.get(k).copied().unwrap_or(d);
        let (n1, n2) = (pw.node1, pw.node2);
        let p1 = self.nodes[n1 as usize].pos;
        let p2 = self.nodes[n2 as usize].pos;
        let span = p2 - p1;
        let axis = span.normalize_or(Vec3::X);
        let rays = ((v("numRays", 10.0).round() as u32).max(4) + 1) & !1;
        let radius = v("radius", 0.5);
        let hub_radius = v("hubRadius", 0.65 * radius);
        let hub_width = v("hubWidth", span.length());
        let tire_width = v("tireWidth", span.length());
        let centre = 0.5 * (p1 + p2) + v("wheelOffset", 0.0) * axis;
        let reference = pw
            .arm
            .map(|a| self.nodes[a as usize].pos - centre)
            .unwrap_or(Vec3::Y);
        let mut r0 = reference - reference.dot(axis) * axis;
        if r0.length_squared() < 1e-8 {
            r0 = axis.any_orthonormal_vector();
        }
        let r0 = r0.normalize();
        let r1 = axis.cross(r0);
        let has_tire = flag("hasTire", true);
        let n = rays as usize;
        let mut hub_mass = v("hubNodeWeight", 1.0);
        if let Some(w) = pw.values.get("hubWeight") {
            hub_mass = w / (2 * n) as f32;
        }
        let mut tire_mass = v("nodeWeight", 0.5);
        if let Some(w) = pw.values.get("tireWeight") {
            tire_mass = w / (2 * n) as f32;
        }
        // Tyre friction: BeamNG's load-sensitive model is not simulated; a tyre node
        // takes the model's no-load (peak) coefficient when the file gives one, else
        // its `frictionCoef` (spec §25: in BeamNG that one multiplies the ground
        // model's, which Rubble's contact model does not have).
        let friction = v("noLoadCoef", v("frictionCoef", 1.0));
        let first = self.nodes.len() as u32;
        let mut ring = |s: &mut Self, r: f32, half: f32, k: usize, mass: f32| -> Vec<u32> {
            (0..n)
                .map(|i| {
                    let a = std::f32::consts::TAU * i as f32 / n as f32;
                    let dir = r0 * a.cos() + r1 * a.sin();
                    let pos = centre + (k as f32 - 0.5) * 2.0 * half * axis + r * dir;
                    s.nodes.push(Node {
                        pos,
                        vel: velocity,
                        force: Vec3::ZERO,
                        mass,
                        friction,
                        flags: NODE_CONTACTER
                            | NODE_TYRE
                            | crate::explicit_beams::NODE_FRICTION_LIMIT,
                    });
                    s.nodes.len() as u32 - 1
                })
                .collect()
        };
        let hub = [
            ring(self, hub_radius, 0.5 * hub_width, 0, hub_mass),
            ring(self, hub_radius, 0.5 * hub_width, 1, hub_mass),
        ];
        let tire = if has_tire {
            Some([
                ring(self, radius, 0.5 * tire_width, 0, tire_mass),
                ring(self, radius, 0.5 * tire_width, 1, tire_mass),
            ])
        } else {
            None
        };
        self.tyres.push(Tyre {
            axle: [n1, n2],
            first,
            count: self.nodes.len() as u32 - first,
            radius,
        });
        let axle = [n1, n2];
        // A beam of set `name` (its Spring, Damp, Deform, Strength, DampCutoffHz,
        // Precompression; anisotropic sets their expansion values).
        // Hub sets the file leaves unset: stiff as the fixed step allows on the hub
        // nodes with margin (ω·h = 0.2 per beam, damping ratio 0.1; spec §25).
        let hub_k = 0.04 * hub_mass / (STEP * STEP);
        let hub_c = 0.2 * (hub_k * hub_mass).sqrt();
        let mut beam = |s: &mut Self, name: &str, a: u32, b: u32, aniso: bool| {
            let hub = name.starts_with("hub");
            let (dk, dc) = if hub {
                (hub_k, hub_c)
            } else {
                (4_300_000.0, 580.0)
            };
            // A reinforcement set the file gives no Spring or Damp takes its base
            // set's (spec §25), as the L-beams take the side beam's.
            let base = match name {
                "wheelTreadReinfBeam" => Some("wheelTreadBeam"),
                "wheelPeripheryReinfBeam" => Some("wheelPeripheryBeam"),
                "wheelSideReinfBeam" | "wheelReinfBeam" => Some("wheelSideBeam"),
                _ => None,
            };
            let g = |k: &str, d: f32| {
                let d = match k {
                    "Spring" => dk,
                    "Damp" => dc,
                    _ => d,
                };
                let d = match (base, k) {
                    (Some(b), "Spring" | "Damp") => v(&format!("{b}{k}"), d),
                    _ => d,
                };
                v(&format!("{name}{k}"), d)
            };
            let jb = JbeamBeam {
                deform: g("Deform", f32::INFINITY),
                precompression: g("Precompression", 1.0).max(0.01),
                damp_cutoff_hz: g("DampCutoffHz", 0.0).max(0.0),
                anisotropic: aniso,
                spring_expansion: aniso.then(|| g("SpringExpansion", g("Spring", 0.0))),
                damp_expansion: aniso.then(|| g("DampExpansion", g("Damp", 0.0))),
                transition_zone: if aniso {
                    v(
                        &format!("{}TransitionZone", name.trim_end_matches("Beam")),
                        0.0,
                    )
                } else {
                    0.0
                },
                long_bound: f32::INFINITY,
                ..Default::default()
            };
            let sb = crate::desc::SoftBeam {
                a,
                b,
                kind: if aniso {
                    BeamKind::Anisotropic
                } else {
                    BeamKind::Normal
                },
                spring: g("Spring", 0.0),
                damp: g("Damp", 0.0),
                deform: jb.deform,
                strength: g("Strength", f32::INFINITY),
                precompression: jb.precompression,
                jbeam: Some(jb),
                ..Default::default()
            };
            s.push_jbeam_beam(&sb);
        };
        for k in 0..2 {
            for i in 0..n {
                beam(self, "hubBeam", hub[k][i], axle[1 - k], false);
                beam(self, "hubSideBeam", hub[k][i], axle[k], false);
                beam(
                    self,
                    "hubPeripheryBeam",
                    hub[k][i],
                    hub[k][(i + 1) % n],
                    false,
                );
            }
        }
        for i in 0..n {
            beam(self, "hubTreadBeam", hub[0][i], hub[1][i], false);
            beam(self, "hubTreadBeam", hub[0][i], hub[1][(i + 1) % n], false);
        }
        let Some(tire) = tire else {
            return;
        };
        let lbeams = flag("enableTireLbeams", false);
        for k in 0..2 {
            for i in 0..n {
                let t = tire[k][i];
                beam(self, "wheelSideBeam", t, hub[k][i], true);
                if flag("enableTireSideReinfBeams", false) {
                    beam(self, "wheelSideReinfBeam", t, hub[k][(i + 1) % n], true);
                    beam(self, "wheelSideReinfBeam", t, hub[k][(i + n - 1) % n], true);
                }
                if flag("enableTireReinfBeams", false) && !lbeams {
                    beam(self, "wheelReinfBeam", t, hub[1 - k][i], false);
                }
                beam(self, "wheelPeripheryBeam", t, tire[k][(i + 1) % n], false);
                if flag("enableTirePeripheryReinfBeams", false) {
                    beam(
                        self,
                        "wheelPeripheryReinfBeam",
                        t,
                        tire[k][(i + 2) % n],
                        false,
                    );
                }
                if flag("enableTireSupportBeams", false) {
                    let sb = crate::desc::SoftBeam {
                        a: axle[k],
                        b: t,
                        kind: BeamKind::Support,
                        spring: v("tireSupportBeamSpring", 4_300_000.0),
                        damp: v("tireSupportBeamDamp", 580.0),
                        strength: v("tireSupportBeamStrength", f32::INFINITY),
                        jbeam: Some(JbeamBeam {
                            deform: v("tireSupportBeamDeform", 220_000.0),
                            long_bound: f32::INFINITY,
                            ..Default::default()
                        }),
                        ..Default::default()
                    };
                    self.push_jbeam_beam(&sb);
                }
            }
        }
        for i in 0..n {
            beam(self, "wheelTreadBeam", tire[0][i], tire[1][i], false);
            if flag("enableTreadReinfBeams", false) {
                beam(
                    self,
                    "wheelTreadReinfBeam",
                    tire[0][i],
                    tire[1][(i + 1) % n],
                    false,
                );
            }
        }
        if lbeams {
            // [WH]: "the wheelReinfBeam setting covers these diagonal beams or
            // L-beams"; a file without it falls back to its side beams' values.
            let g = |k: &str, d: f32| {
                v(
                    &format!("wheelLbeam{k}"),
                    v(
                        &format!("wheelReinfBeam{k}"),
                        v(&format!("wheelSideBeam{k}"), d),
                    ),
                )
            };
            let (k, c) = (g("Spring", 4_300_000.0), g("Damp", 580.0));
            let (deform, strength) = (g("Deform", f32::INFINITY), g("Strength", f32::INFINITY));
            let cutoff = g("DampCutoffHz", 0.0).max(0.0);
            for s in 0..2 {
                for i in 0..n {
                    let t = tire[s][i];
                    self.push_lbeam(
                        hub[s][i],
                        tire[1 - s][i],
                        t,
                        k,
                        c,
                        deform,
                        strength,
                        cutoff,
                        1.0,
                    );
                    self.push_lbeam(
                        hub[s][i],
                        tire[s][(i + 1) % n],
                        t,
                        k,
                        c,
                        deform,
                        strength,
                        cutoff,
                        1.0,
                    );
                    self.push_lbeam(
                        hub[s][i],
                        tire[s][(i + n - 1) % n],
                        t,
                        k,
                        c,
                        deform,
                        strength,
                        cutoff,
                        1.0,
                    );
                }
            }
        }
        // The closed surface, wound outward: side 0 faces −axis, side 1 +axis, the
        // tread outward, the hub band inward (toward the axle).
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(8 * n);
        let quad = |tris: &mut Vec<[u32; 3]>, a: u32, b: u32, c: u32, d: u32| {
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        };
        for i in 0..n {
            let j = (i + 1) % n;
            quad(&mut tris, hub[0][i], tire[0][i], tire[0][j], hub[0][j]);
            quad(&mut tris, hub[1][i], hub[1][j], tire[1][j], tire[1][i]);
            quad(&mut tris, tire[0][i], tire[1][i], tire[1][j], tire[0][j]);
            quad(&mut tris, hub[0][i], hub[0][j], hub[1][j], hub[1][i]);
        }
        // Fix the winding against the enclosed volume's sign.
        let c = self.nodes[tris[0][0] as usize].pos;
        let vol: f32 = tris
            .iter()
            .map(|t| {
                let x = t.map(|i| self.nodes[i as usize].pos - c);
                x[0].dot(x[1].cross(x[2]))
            })
            .sum();
        if vol < 0.0 {
            for t in &mut tris {
                t.swap(1, 2);
            }
        }
        let p0 = v("pressurePSI", 0.0) * PSI;
        let ceiling = p0 + v("maxPressurePSI", 10.0) * PSI;
        if p0 > 0.0 {
            self.push_pressure_group(&tris, p0, ceiling);
        }
    }

    /// The desc's rails and slide nodes on the solver's slide nodes (spec §27).
    pub(crate) fn add_jbeam_rails(&mut self, desc: &SoftVehicleDesc, warnings: &mut Vec<String>) {
        if desc.slidenodes.is_empty() {
            return;
        }
        let mut first: HashMap<(u32, u32), u32> = HashMap::new();
        for (j, b) in self.beams.iter().enumerate() {
            first
                .entry((b.a.min(b.b), b.a.max(b.b)))
                .or_insert(j as u32);
        }
        // Per rail its segments' first index (made once, shared by its nodes).
        let mut rail_segs: HashMap<u32, (u32, u32)> = HashMap::new();
        let (mut uncapped, mut no_fix) = (0usize, 0usize);
        for sn in &desc.slidenodes {
            let rail = &desc.rails[sn.rail as usize];
            if !rail.capped {
                uncapped += 1;
            }
            let _ = &mut no_fix;
            let (base, count) = *rail_segs.entry(sn.rail).or_insert_with(|| {
                let mut links = rail.nodes.clone();
                if rail.looped && links.first() != links.last() {
                    links.push(links[0]);
                }
                let base = self.segments.len() as u32;
                let n = links.len().saturating_sub(1) as u32;
                for (k, p) in links.windows(2).enumerate() {
                    let key = (p[0].min(p[1]), p[0].max(p[1]));
                    let beam = match first.get(&key) {
                        Some(&j) => j,
                        None => {
                            // A rail pair with no beam: an inert one to carry it.
                            let b = Beam::new(
                                p[0],
                                p[1],
                                ExplicitKind::JbeamPlain,
                                ExplicitType::Actuator,
                                0.0,
                                0.0,
                                self.nodes[p[0] as usize]
                                    .pos
                                    .distance(self.nodes[p[1] as usize].pos),
                                f32::INFINITY,
                                f32::INFINITY,
                                0.0,
                            );
                            self.beams.push(b);
                            let j = self.beams.len() as u32 - 1;
                            first.insert(key, j);
                            j
                        }
                    };
                    let k = k as u32;
                    let prev = if k > 0 {
                        Some(base + k - 1)
                    } else if rail.looped {
                        Some(base + n - 1)
                    } else {
                        None
                    };
                    let next = if k + 1 < n {
                        Some(base + k + 1)
                    } else if rail.looped {
                        Some(base)
                    } else {
                        None
                    };
                    self.segments
                        .push(crate::explicit_triggers::RailSegment { beam, prev, next });
                }
                (base, n)
            });
            let mut slide = crate::explicit_triggers::SlideNode {
                node: sn.node,
                spring: sn.spring,
                break_force: sn.strength,
                tolerance: sn.tolerance,
                segment: None,
                broken_away: !sn.attached,
            };
            if count > 0 {
                let x = self.origin + self.nodes[sn.node as usize].pos;
                let mut best = (base, f32::INFINITY);
                for seg in base..base + count {
                    let b = &self.beams[self.segments[seg as usize].beam as usize];
                    let p1 = self.origin + self.nodes[b.a as usize].pos;
                    let p2 = self.origin + self.nodes[b.b as usize].pos;
                    let d = crate::explicit_triggers::segment_point(p1, p2, x).1;
                    if d < best.1 {
                        best = (seg, d);
                    }
                }
                slide.segment = Some(best.0);
                if sn.fix_to_rail {
                    // `fixToRail` [R]: onto the rail at spawn (or to within its
                    // tolerance).
                    let b = &self.beams[self.segments[best.0 as usize].beam as usize];
                    let p1 = self.origin + self.nodes[b.a as usize].pos;
                    let p2 = self.origin + self.nodes[b.b as usize].pos;
                    let (q, d, _) = crate::explicit_triggers::segment_point(p1, p2, x);
                    if d > sn.tolerance && d > 0.0 {
                        let to = q + (x - q) * (sn.tolerance / d);
                        self.nodes[sn.node as usize].pos = to - self.origin;
                    }
                }
            }
            self.slides.push(slide);
        }
        if uncapped > 0 {
            warnings.push(format!(
                "{uncapped} slide nodes ride uncapped rails: their ends hold them (no run-off)"
            ));
        }
    }

    /// Dampers the fixed step cannot integrate (spec §28): a JBeam-mode beam whose
    /// damping c gives c·h/μ > 1 on its nodes' reduced mass μ takes the
    /// backward-Euler damping of that pair, c/(1 + c·h/μ) (its slow, fast, limit and
    /// rebound rates alike). Returns how many beams.
    pub(crate) fn stiff_dampers(&mut self) -> usize {
        let mut n = 0;
        for j in 0..self.beams.len() {
            let b = self.beams[j];
            let plain = matches!(
                b.kind,
                ExplicitKind::JbeamPlain
                    | ExplicitKind::JbeamSupport
                    | ExplicitKind::JbeamBounded
                    | ExplicitKind::JbeamAniso
                    | ExplicitKind::JbeamPressured
                    | ExplicitKind::JbeamLbeam
            );
            if !plain || b.a == b.b {
                continue;
            }
            let (ma, mb) = (self.nodes[b.a as usize].mass, self.nodes[b.b as usize].mass);
            if !(ma > 0.0 && mb > 0.0) {
                continue;
            }
            let mu = ma * mb / (ma + mb);
            let fix = |c: f32| {
                let x = c * STEP / mu;
                if x > 1.0 && c.is_finite() {
                    c / (1.0 + x)
                } else {
                    c
                }
            };
            let mut changed = false;
            let mut set = |c: &mut f32| {
                let f = fix(*c);
                if f != *c {
                    *c = f;
                    changed = true;
                }
            };
            let beam = &mut self.beams[j];
            set(&mut beam.c);
            if beam.kind == ExplicitKind::JbeamBounded {
                set(&mut beam.limit_c);
                set(&mut beam.tt);
            }
            if let Some(&x) = self.ext_of.get(j).filter(|&&x| x != 0) {
                let ext = &mut self.beam_ext[x as usize - 1];
                if ext.mask & EXT_REBOUND != 0 {
                    for k in [0, 1, 3] {
                        set(&mut ext.e2[k]);
                    }
                }
                if beam.kind == ExplicitKind::JbeamAniso {
                    set(&mut ext.e1[1]);
                }
            }
            if changed {
                n += 1;
            }
        }
        n
    }
}
