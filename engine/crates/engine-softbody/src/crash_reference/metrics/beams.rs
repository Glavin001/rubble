//! Per beam: what kind of record it is for measurement, the elastic energy it stores, and
//! the plastic work its yield takes, by the solver's own rules (`explicit_beams.rs`,
//! `beam_force` and `jbeam_force_filtered`).
//!
//! **Plastic flow, as the solver runs it.** A deformable beam whose force passes its
//! threshold moves its rest length so that its elastic part is exactly the threshold
//! (JBeam beams spawn with `plastic` = 0), and the threshold then hardens with the set:
//! compression `tc ← tc · old / new`, tension `tt ← tt · new / old` (JBeam kinds capped at
//! max(`deformLimitStress`, the threshold); a rest length at its `deformLimit` floor stays
//! there, elastic). So the true plastic work of a monotone set from L₀ to L is
//! ∫ τ(r) dr = tc₀ L₀ ln(L₀ / L) in compression (more than tc₀ |L − L₀|) and
//! tt₀ (L² − L₀²) / 2L₀ in tension. [`Plastic::advance`] integrates that between
//! samples, taking each sample-to-sample change of rest as monotone (a beam that both
//! shortens and lengthens by yield inside one 4 ms sample is integrated on its net change).

use crate::explicit_beams::{dihedral, wrap_angle, Beam, ExplicitBeams, ExplicitKind, ExplicitType};
use crate::explicit_crush::{Cell, Crush, Curve};
use glam::Vec3;

/// A beam that is a length between two nodes (counted in the bent and broken fractions).
pub fn is_length(k: ExplicitKind) -> bool {
    use ExplicitKind::*;
    matches!(k, Plain | LimitedSpring | Support | Rope | Shock2 | JbeamPlain | JbeamSupport | JbeamBounded | JbeamAniso | JbeamPressured | JbeamRope)
}

/// A beam that ties its two nodes together for the parts count (aerodynamic, pressure
/// and second records carry forces of a triangle or element, not a bond).
pub fn is_link(k: ExplicitKind) -> bool {
    use ExplicitKind::*;
    is_length(k) || matches!(k, JbeamCoupler | JbeamTorsion | JbeamLbeam)
}

/// A beam that can yield.
pub fn can_yield(b: &Beam) -> bool {
    b.ty == ExplicitType::Deformable && b.k != 0.0 && b.kind != ExplicitKind::LimitedSpring && is_length(b.kind)
}

/// ∫₀ᵉ k(x) x dx for a stiffness k(x) that is constant or linear in x between the cuts
/// (Simpson's rule on each piece is then exact: the integrand is at most cubic). e may be
/// negative (compression); the energy is positive either way.
pub(super) fn stored(e: f64, cuts: [f64; 2], stiff: impl Fn(f64) -> f64) -> f64 {
    let mut xs = vec![0.0, e];
    xs.extend(cuts.into_iter().filter(|c| c.abs() < e.abs() && c.signum() == e.signum() && *c != 0.0));
    xs.sort_by(f64::total_cmp);
    xs.windows(2)
        .map(|w| {
            let (x0, x1) = (w[0], w[1]);
            let xm = 0.5 * (x0 + x1);
            (x1 - x0) / 6.0 * (stiff(x0) * x0 + 4.0 * stiff(xm) * xm + stiff(x1) * x1)
        })
        .sum::<f64>()
        .abs()
}

/// A stiffness k moving to `k_lim` over `zone` past `bound` (k + (k_lim − k) · min(o /
/// zone, 1), o the extension past the bound; a zero zone steps).
pub(super) fn ramp(k: f64, k_lim: f64, bound: f64, zone: f64, o: f64) -> f64 {
    let o = o - bound;
    if o <= 0.0 {
        k
    } else {
        k + (k_lim - k) * if zone > 0.0 { (o / zone).min(1.0) } else { 1.0 }
    }
}

/// Elastic energy (J) stored in beam `j` of `beams` (spawn records, for kind and rates)
/// with rest length `rest` and nodes at `pos` (`ext`: the vehicle, for anisotropic beams'
/// extension records). Broken beams store none. Counted: plain,
/// support (compressed), rope (stretched), bounded (with its limit spring), anisotropic
/// (with its expansion rate), coupler,
/// torsion, LBEAM ([`lbeam_rest`]); others (pressure groups, aerodynamic records,
/// triggers) store none here (the crash scenes' designs have no pressure groups).
pub fn elastic(beams: &[Beam], ext: Option<&ExplicitBeams>, j: usize, rest: f32, broken: bool, pos: &[Vec3]) -> f64 {
    use ExplicitKind::*;
    let b = &beams[j];
    if broken {
        return 0.0;
    }
    let (Some(pa), Some(pb)) = (pos.get(b.a as usize), pos.get(b.b as usize)) else { return 0.0 };
    let len = f64::from(pa.distance(*pb));
    let e = len - f64::from(rest);
    let k = f64::from(b.k);
    match b.kind {
        Plain | Shock2 | LimitedSpring | JbeamPlain | JbeamPressured => 0.5 * k * e * e,
        Support | JbeamSupport if e < 0.0 => 0.5 * k * e * e,
        Rope | JbeamRope if e > 0.0 => 0.5 * k * e * e,
        JbeamBounded => {
            let r = f64::from(rest);
            let bound = if e > 0.0 { f64::from(b.long_bound) * r } else { f64::from(b.short_bound) * r };
            let (k_lim, zone) = (f64::from(b.limit_k), f64::from(b.plastic));
            stored(e, [e.signum() * bound, e.signum() * (bound + zone)], |x| ramp(k, k_lim, bound, zone, x.abs()))
        }
        JbeamAniso => {
            // Past the switch length (rest + E2 .y) the expansion rate E1 .x over a zone of
            // E1 .z × the switch length (spec §20).
            let Some(x) = ext.and_then(|v| v.ext_of.get(j).filter(|&&i| i > 0).and_then(|&i| v.beam_ext.get(i as usize - 1))) else { return 0.5 * k * e * e };
            let off = f64::from(x.e2[1]);
            let zone = f64::from(x.e1[2]) * (f64::from(rest) + off);
            let k1 = f64::from(x.e1[0]);
            stored(e, [off, off + zone], |y| if y > 0.0 { ramp(k, k1, off, zone, y) } else { k })
        }
        JbeamCoupler => 0.5 * k * len * len,
        JbeamLbeam => {
            // ½k (ℓ − L)², L the rest length id1–id2 the arms and the rest angle give
            // (`lbeam_force`; the solver's `elastic_energy` the same).
            let Some(l) = ext.and_then(|v| lbeam_rest(v, j, rest, pos)) else { return 0.0 };
            0.5 * k * (len - l).powi(2)
        }
        JbeamTorsion => {
            let Some(pair) = beams.get(j + 1) else { return 0.0 };
            let (Some(p3), Some(p4)) = (pos.get(pair.a as usize), pos.get(pair.b as usize)) else { return 0.0 };
            let phi = f64::from(wrap_angle(dihedral([*pa, *pb, *p3, *p4]) - rest));
            0.5 * k * phi * phi
        }
        _ => 0.0,
    }
}

/// An LBEAM's rest length id1–id2 (m) for rest cos θ₀ `cos` with its nodes at `pos`:
/// √(a² + b² − 2ab cos θ₀), a and b its arms to the vertex id3 (`lbeam_force`). None when
/// beam `j` is no LBEAM of `v` or a node is missing.
pub fn lbeam_rest(v: &ExplicitBeams, j: usize, cos: f32, pos: &[Vec3]) -> Option<f64> {
    let b = v.beams.get(j).filter(|b| b.kind == ExplicitKind::JbeamLbeam)?;
    let x = *v.ext_of.get(j)? as usize;
    let c = v.beam_ext.get(x.checked_sub(1)?)?.index(0) as usize;
    let (pa, pb, pc) = (pos.get(b.a as usize)?, pos.get(b.b as usize)?, pos.get(c)?);
    let (a, bb) = (f64::from(pa.distance(*pc)), f64::from(pb.distance(*pc)));
    Some((a * a + bb * bb - 2.0 * a * bb * f64::from(cos)).max(0.0).sqrt())
}

/// An LBEAM's yield, as the solver runs it (`lbeam_force`): past its deform force (`tc`)
/// its rest angle moves so that its spring stays at that force, perfectly plastic (no
/// hardening). The work of a move of the rest angle between samples is the deform force ×
/// the move of its rest length id1–id2 at the arms of the later sample (exact while the
/// arms do not change between them).
#[derive(Clone, Copy, Debug)]
pub struct LbeamPlastic {
    tc: f64,
    cos: f32,
    pub work: f64,
}

impl LbeamPlastic {
    /// None for a beam that is no LBEAM or cannot yield.
    pub fn new(b: &Beam) -> Option<Self> {
        (b.kind == ExplicitKind::JbeamLbeam && b.tc.is_finite() && b.k != 0.0).then(|| Self { tc: f64::from(b.tc), cos: b.rest, work: 0.0 })
    }

    pub fn advance(&mut self, v: &ExplicitBeams, j: usize, cos: f32, pos: &[Vec3]) {
        if cos != self.cos {
            if let (Some(a), Some(b)) = (lbeam_rest(v, j, self.cos, pos), lbeam_rest(v, j, cos, pos)) {
                self.work += self.tc * (b - a).abs();
            }
            self.cos = cos;
        }
    }
}

/// One yieldable beam's thresholds as they harden, and the plastic work so far.
#[derive(Clone, Copy, Debug)]
pub struct Plastic {
    tc: f64,
    tt: f64,
    /// JBeam kinds' hardening cap (`deformLimitStress`); infinite for the others.
    cap: f64,
    rest: f64,
    pub work: f64,
    /// The crush law's capacity curve and compressive set so far (`explicit_crush`): its
    /// compression work is ∫ C ds over the set. None: the JBeam rule.
    crush: Option<(Curve, f64)>,
    /// A crush cell's chord while the cell acts ([`CellPlastic`] counts its work then).
    pub dormant: bool,
}

impl Plastic {
    pub fn new(b: &Beam) -> Self {
        let cap = if b.kind.is_jbeam() { f64::from(b.limit_k) } else { f64::INFINITY };
        Self { tc: f64::from(b.tc), tt: f64::from(b.tt), cap, rest: f64::from(b.rest), work: 0.0, crush: None, dormant: false }
    }

    /// [`Self::new`] for beam `j` of `v`, with its crush law if it has one (dormant while in a crush
    /// cell).
    pub fn of(v: &ExplicitBeams, j: usize) -> Self {
        let mut p = Self::new(&v.beams[j]);
        if let Some(c) = crush_of(v, j) {
            p.crush = Some((c.curve, f64::from(c.set)));
            p.dormant = c.cell.is_some();
        }
        p
    }

    /// A crush cell's chord carries on alone (its cell stopped acting) at compressive set `set`.
    pub fn wake(&mut self, set: f64) {
        self.dormant = false;
        if let Some((_, s)) = self.crush.as_mut() {
            *s = set;
        }
    }

    /// The rest length is now `rest`: the work of the set since the last sample.
    pub fn advance(&mut self, rest: f32) {
        let (old, new) = (self.rest, f64::from(rest));
        self.rest = new;
        if !(new > 0.0 && old > 0.0) || new == old || self.dormant {
            return;
        }
        if let (true, Some((curve, set))) = (new < old, self.crush.as_mut()) {
            self.work += curve.work(*set, *set + (old - new));
            *set += old - new;
            return;
        }
        if new < old && self.tc.is_finite() {
            // τ(r) = tc·old/r up to the cap C (reached at r* = tc·old/C), C below it.
            let c = self.cap.max(self.tc);
            let r_star = self.tc * old / c;
            let lo = new.max(r_star);
            self.work += self.tc * old * (old / lo).ln() + if r_star > new { c * (r_star - new) } else { 0.0 };
            self.tc = (self.tc * old / new).min(c);
        } else if new > old && self.tt.is_finite() {
            // τ(r) = tt·r/old up to the cap C (reached at r* = C·old/tt), C past it.
            let c = self.cap.max(self.tt);
            let r_star = c * old / self.tt;
            let hi = new.min(r_star);
            self.work += self.tt * (hi * hi - old * old) / (2.0 * old) + if new > r_star { c * (new - r_star) } else { 0.0 };
            self.tt = (self.tt * new / old).min(c);
        }
    }
}

/// Beam `j`'s crush-law record, if it has one.
pub fn crush_of(v: &ExplicitBeams, j: usize) -> Option<Crush> {
    let x = *v.ext_of.get(j).filter(|&&x| x > 0)? as usize;
    v.beam_ext.get(x - 1)?.crush
}

/// A crush cell's plastic work (`explicit_crush`): each sample's change of the four chords' rest
/// lengths split into the cell's modes (Δπ = Φ·Δr, Φ orthonormal), each mode's flow times its
/// capacity: the axial mode ∫ C ds in compression and its tension capacity in tension, the others
/// their capacity × |Δπ|. Exact while each mode flows one way between samples. Counted while the
/// cell acts (until a chord breaks; then its chords count alone: [`Plastic::dormant`]).
#[derive(Clone, Copy, Debug)]
pub struct CellPlastic {
    pub cell: Cell,
    rest: [f64; 4],
    /// The axial mode's compressive set (a chord's is half).
    pub set: f64,
    pub alive: bool,
    pub work: f64,
}

impl CellPlastic {
    /// The cell whose master is beam `j` of `v` (None: no cell master).
    pub fn of(v: &ExplicitBeams, j: usize) -> Option<Self> {
        let c = crush_of(v, j)?.cell?;
        (c.beams[0] as usize == j && c.alive).then(|| Self { cell: c, rest: c.beams.map(|b| f64::from(v.beams[b as usize].rest)), set: f64::from(c.set), alive: true, work: 0.0 })
    }

    /// The chords' rests and broken flags now; returns true the sample the cell stops acting.
    pub fn advance(&mut self, rest: [f32; 4], broken: [bool; 4]) -> bool {
        if !self.alive {
            return false;
        }
        let r = rest.map(f64::from);
        for m in 0..4 {
            let dpi: f64 = (0..4).map(|i| f64::from(self.cell.phi[m][i]) * (r[i] - self.rest[i])).sum();
            if m == 0 && dpi < 0.0 {
                self.work += self.cell.axial.work(self.set, self.set - dpi);
                self.set -= dpi;
            } else {
                self.work += f64::from(self.cell.cap[m]) * dpi.abs();
            }
        }
        self.rest = r;
        if broken.iter().any(|b| *b) {
            self.alive = false;
            return true;
        }
        false
    }
}
