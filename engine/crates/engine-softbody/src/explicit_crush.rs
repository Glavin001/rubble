//! Rubble's crush law and crush cells (`engine/docs/beam-yield-law.md`): an opt-in yield law for
//! a JBeam NORMAL beam, and a four-chord cell whose chords yield together as one member. **CPU
//! reference only**: no kernel runs either, and [`assert_cpu_only`] refuses a GPU upload of a
//! vehicle that uses them. A beam without the keys below runs exactly the code it ran before (its
//! extension record, if any, prints and compares as before: [`BeamExt`]'s `Debug`).
//!
//! **The crush law** (keys `rubbleCrushPlateau`, `rubbleCrushDrop`, `rubbleMinRest`,
//! `rubbleTensionYield`): in compression the beam is elastic (spring k) up to its peak
//! P = `beamDeform`; then its rest length moves (return mapping, no lag) so that its elastic force
//! is the capacity C(s) of its compressive plastic set s: C falls linearly from P to the plateau Q
//! over the set `rubbleCrushDrop` d, then stays at Q; at the floor max((1 − `deformLimit`)·L₀,
//! `rubbleMinRest`) the rest length stops and the beam is a spring about it (densified). The set s
//! only grows (a fold does not unfold), so a reload continues on the plateau. Tension keeps the
//! JBeam rule (§3 of the semantics spec) with Tt = `rubbleTensionYield` (default `beamDeform`).
//! Plastic work ∫C ds ≥ 0; the drop from P to Q is a softening of slope H = (P − Q)/d, stable
//! (no snap-back, explicit-stable tangent) while H ≤ k/2, which the spawn requires.
//!
//! **A crush cell** (keys `rubbleCrushCell` naming four chords, `rubbleCellMoment` [strong, weak]
//! N·m): the four chords' rest-length changes are split into four orthonormal modes of the cell
//! (axial, bending about each principal axis, warping) by their spawn positions; each mode yields
//! on its own generalised force: the axial mode by the chords' crush law summed (peak, plateau,
//! drop, tension), the bending modes at the given collapse moments, the warping mode at the
//! smaller of the two. So the cell folds axially at its plateau and keeps its bending strength
//! while it does (a pin-jointed cell of independent chords cannot: beam-yield-law.md §3.5).
//! Forces stay along the chords (momentum), the stored energy is the chords' ½k(ℓ − r)², and each
//! mode's plastic work is its force times its flow, never negative.
use crate::desc::JbeamBeam;
use crate::explicit_beams::{exact_length, Beam, ExplicitBeams, ExplicitKind, ExplicitType, MIN_REST};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Extension record feature bits (E0 .x, after `explicit_jbeam_ext`'s 1, 2, 4, 8): the crush law,
/// a crush cell's chord.
pub const EXT_CRUSH: u32 = 16;
pub const EXT_CELL: u32 = 32;
/// The least `rubbleMinRest` a file may set (m): a chord locked at its least rest length must
/// stop its nodes' closing before they meet. At 200 km/h (55.6 m/s) one 0.5 ms step closes
/// 27.8 mm, and a 2 kg node pair on a 4 MN/m chord needs v·√(μ/k) ≈ 39 mm of spring to stop
/// (beam-yield-law.md §3.4); below this the spawn fails.
pub const MIN_REST_FLOOR: f32 = 0.04;

/// A beam's crush-law keys as the file gives them (`JbeamBeam::crush`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrushLaw {
    /// `rubbleCrushPlateau` (N): the force after the drop.
    pub plateau: f32,
    /// `rubbleCrushDrop` (m): the plastic set over which the force falls from the peak to the plateau.
    pub drop: f32,
    /// `rubbleMinRest` (m): the least rest length (default the solver's `MIN_REST`).
    pub min_rest: f32,
    /// `rubbleTensionYield` (N; 0: `beamDeform`).
    pub tension: f32,
    /// `rubbleCrushCell`: the cell's name hashed (0: no cell).
    pub cell: u32,
    /// `rubbleCellMoment` (N·m): collapse moments about the cell's strong and weak axes.
    pub moment: [f32; 2],
}

/// The capacity curve C(s) of a compressive set s: P falling linearly to Q over `drop`, then Q.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    pub peak: f32,
    pub plateau: f32,
    pub drop: f32,
}

impl Curve {
    pub fn at(&self, s: f32) -> f32 {
        if s >= self.drop {
            self.plateau
        } else {
            self.peak + (self.plateau - self.peak) * (s / self.drop)
        }
    }

    /// The softening slope H = (P − Q) / d on the drop (0 without one).
    pub fn slope(&self) -> f32 {
        if self.drop > 0.0 {
            (self.peak - self.plateau) / self.drop
        } else {
            0.0
        }
    }

    /// Return mapping: the set Δ ≥ 0 that brings an elastic force `g` (stiffness `k`) at set `s`
    /// down to the capacity at s + Δ (g − kΔ = C(s + Δ)); 0 when g is within it.
    pub fn flow(&self, g: f32, k: f32, s: f32) -> f32 {
        if g <= self.at(s) {
            return 0.0;
        }
        if s < self.drop {
            let d = (g - self.at(s)) / (k - self.slope());
            if s + d <= self.drop {
                return d;
            }
        }
        (g - self.plateau) / k
    }

    /// ∫ C ds from `s0` to `s1` (s1 ≥ s0), exact.
    pub fn work(&self, s0: f64, s1: f64) -> f64 {
        let (p, q, d) = (f64::from(self.peak), f64::from(self.plateau), f64::from(self.drop));
        let c = |s: f64| if s >= d { q } else { p + (q - p) * s / d };
        let mid = d.clamp(s0, s1);
        0.5 * (c(s0) + c(mid)) * (mid - s0) + q * (s1 - mid)
    }

    /// Whether the drop is stable at spring `k`: H ≤ k/2 (beam-yield-law.md §3.3).
    pub fn stable(&self, k: f32) -> bool {
        self.slope() <= 0.5 * k
    }
}

/// A beam's crush-law record (in its [`BeamExt`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crush {
    pub curve: Curve,
    /// The least rest length (m): max(`deformLimit`'s floor, `rubbleMinRest`).
    pub floor: f32,
    /// State: the compressive plastic set so far (m).
    pub set: f32,
    pub cell: Option<Cell>,
}

/// A crush cell. Every chord's record holds one; the master's (its first chord's) is the cell's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub id: u32,
    /// The master's extension record.
    pub master: u32,
    /// The chords (beam indices; the master's record only).
    pub beams: [u32; 4],
    /// Modes over the chords: axial, strong bending, weak bending, warping (orthonormal rows).
    pub phi: [[f32; 4]; 4],
    /// Generalised capacities: tension of the axial mode, then the three others (N).
    pub cap: [f32; 4],
    /// The axial mode's compressive curve in its generalised set s_G (chord set = s_G / 2).
    pub axial: Curve,
    pub k: f32,
    /// Lever of each bending mode (Σ φ·y, Σ φ·x: moment = generalised force × lever), for reports.
    pub lever: [f32; 2],
    /// State: the axial mode's compressive set s_G, whether the cell still acts, and the step it
    /// dissolved in after its master computed it (its other chords skip that step).
    pub set: f32,
    pub alive: bool,
    pub skip: u64,
}

/// A beam's extension record (spec §16). E0 = (mask, deform groups, trigger ratio), E1 and E2 the
/// kind's parameters (index fields as `f32` bits), E3 = (precompression ramp step, ramp steps,
/// extra break groups); then the state; `crush`: Rubble's crush law (CPU only).
#[derive(Clone, Copy, Default, PartialEq)]
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
    pub crush: Option<Crush>,
}

impl std::fmt::Debug for BeamExt {
    /// As `#[derive(Debug)]` printed it before `crush` (so a fleet without the crush law prints
    /// word for word as before: `crash_fleet_digests`), with `crush` when set.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("BeamExt");
        d.field("beam", &self.beam).field("mask", &self.mask).field("deform", &self.deform).field("trigger", &self.trigger);
        d.field("e1", &self.e1).field("e2", &self.e2).field("ramp_delta", &self.ramp_delta).field("ramp_steps", &self.ramp_steps);
        d.field("extra_groups", &self.extra_groups).field("ramp_left", &self.ramp_left).field("pending", &self.pending).field("pressure", &self.pressure);
        if let Some(c) = &self.crush {
            d.field("crush", c);
        }
        d.finish()
    }
}

impl BeamExt {
    /// The record's E1 .x as an index.
    pub fn index(&self, k: usize) -> u32 {
        self.e1[k].to_bits()
    }
}

/// FNV-1a of a cell's name (never 0).
pub fn cell_id(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h.max(1)
}

/// The JBeam beam options' keys this module reads (and `disableTriangleBreaking`, read here to keep
/// `jbeam.rs` within its size).
pub fn read_keys(jb: &mut JbeamBeam, f: &serde_json::Map<String, serde_json::Value>) {
    jb.disable_triangle_breaking = f.get("disableTriangleBreaking").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let num = |k: &str| f.get(k).and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).map(|x| x as f32);
    let Some(plateau) = num("rubbleCrushPlateau") else { return };
    let moment = match f.get("rubbleCellMoment").and_then(serde_json::Value::as_array) {
        Some(a) if a.len() == 2 => [0, 1].map(|i| a[i].as_f64().unwrap_or(f64::NAN) as f32),
        _ => [f32::NAN; 2],
    };
    jb.crush = Some(CrushLaw {
        plateau,
        drop: num("rubbleCrushDrop").unwrap_or(0.0),
        min_rest: num("rubbleMinRest").unwrap_or(MIN_REST),
        tension: num("rubbleTensionYield").unwrap_or(0.0),
        cell: f.get("rubbleCrushCell").and_then(serde_json::Value::as_str).map_or(0, cell_id),
        moment,
    });
}

/// At spawn ([`ExplicitBeams::attach_ext`], before the record is pushed): the crush law's record and
/// the beam's tension threshold. Panics on a law the solver cannot run as written (an explicit
/// configuration error, AGENTS.md): not a deformable NORMAL beam, a least rest under
/// [`MIN_REST_FLOOR`], a drop steeper than k/2, a negative or missing value.
pub fn attach(beam: &mut Beam, ext: &mut BeamExt, jb: &JbeamBeam) {
    let Some(law) = jb.crush else { return };
    let at = || format!("beam {}–{}", beam.a, beam.b);
    assert!(beam.kind == ExplicitKind::JbeamPlain && beam.ty == ExplicitType::Deformable, "{}: the crush law is for deformable NORMAL beams", at());
    assert!(law.min_rest >= MIN_REST_FLOOR, "{}: rubbleMinRest {} m is under the least the 0.5 ms step allows ({MIN_REST_FLOOR} m, beam-yield-law.md §3.4)", at(), law.min_rest);
    assert!(law.plateau > 0.0 && law.drop >= 0.0 && law.tension >= 0.0, "{}: crush law values must be positive", at());
    let curve = Curve { peak: beam.tc, plateau: law.plateau, drop: law.drop };
    assert!(curve.peak.is_finite() && (law.drop > 0.0 || curve.peak == curve.plateau), "{}: a peak above the plateau needs rubbleCrushDrop > 0", at());
    assert!(curve.stable(beam.k), "{}: the drop from {} to {} N over {} m is steeper than k/2 = {} N/m (needs rubbleCrushDrop ≥ {} m, beam-yield-law.md §3.3)", at(), curve.peak, curve.plateau, law.drop, 0.5 * beam.k, 2.0 * (curve.peak - curve.plateau) / beam.k);
    if law.tension > 0.0 {
        beam.tt = law.tension;
        beam.gate = beam.tc.min(beam.tt).min(beam.strength);
    }
    ext.mask |= EXT_CRUSH;
    ext.crush = Some(Crush { curve, floor: beam.short_bound.max(law.min_rest), set: 0.0, cell: None });
    if law.cell != 0 {
        assert!(law.moment.iter().all(|m| m.is_finite() && *m > 0.0), "{}: a crush cell's chord needs rubbleCellMoment [strong, weak] (N·m)", at());
        assert!(beam.shock == 0, "{}: a crush cell's chord takes no dampCutoffHz", at());
        ext.mask |= EXT_CELL;
        ext.crush.as_mut().unwrap().cell = Some(Cell { id: law.cell, master: 0, beams: [0; 4], phi: [[0.0; 4]; 4], cap: [0.0, law.moment[0], law.moment[1], 0.0], axial: curve, k: beam.k, lever: [0.0; 2], set: 0.0, alive: false, skip: u64::MAX });
    }
}

/// A NORMAL beam's yield under the crush law (`jbeam_ext_force_with`'s yield step): None for the
/// JBeam rule (no crush law, or tension: the beam's own Tt rule), else (force, |force| for the break
/// test). `f` is the trial force −k·e − damping; the law reads the elastic part only.
#[allow(clippy::too_many_arguments)]
pub fn crush_yield(beam: &mut Beam, ext: &mut BeamExt, len: f32, e: f32, k: f32, damp: f32, f: f32) -> Option<(f32, f32)> {
    let mut c = ext.crush?;
    if e >= 0.0 || k == 0.0 || beam.ty != ExplicitType::Deformable {
        return None;
    }
    let d = c.curve.flow(-k * e, k, c.set);
    if d <= 0.0 {
        return Some((f, f.abs()));
    }
    let old = beam.rest;
    let rest = (old - d).max(c.floor.min(old));
    beam.rest = rest;
    c.set += old - rest;
    ext.crush = Some(c);
    let f = -k * (len - rest) - damp;
    beam.force = f;
    Some((f, f.abs()))
}

/// Refuse a GPU upload of a fleet that uses the crush law or crush cells (no kernel implements
/// them; beam-yield-law.md §7). Called by `engine_exec::explicit_plan::ExplicitUpload::with_sleep`,
/// the one packer every GPU executor (WebGPU, Metal, CUDA) uploads through.
pub fn assert_cpu_only(vehicles: &[ExplicitBeams]) {
    for (i, v) in vehicles.iter().enumerate() {
        let n = v.beam_ext.iter().filter(|x| x.mask & (EXT_CRUSH | EXT_CELL) != 0).count();
        assert!(n == 0, "explicit vehicles: vehicle {i} has {n} beams with Rubble's crush law (rubbleCrushPlateau / rubbleCrushCell, engine/docs/beam-yield-law.md): it runs on the CPU reference only, no GPU kernel implements it");
    }
}

/// The four chords' spawn geometry into a cell's modes and capacities (`chords`: their beams' and
/// records' indices; `moment` [strong, weak] N·m). Panics on a cell that cannot be built.
fn build(v: &ExplicitBeams, chords: [(usize, usize); 4], moment: [f32; 2]) -> Cell {
    let b = chords.map(|(j, _)| v.beams[j]);
    let p = |n: u32| v.nodes[n as usize].pos;
    let k = b[0].k;
    let name = format!("crush cell of beams {:?}", chords.map(|c| c.0));
    assert!(b.iter().all(|x| (x.k - k).abs() <= 1e-6 * k), "{name}: its chords need one spring (k {:?})", b.map(|x| x.k));
    let crush = chords.map(|(_, x)| v.beam_ext[x].crush.unwrap());
    let drop = crush[0].curve.drop;
    assert!(crush.iter().all(|c| c.curve.drop == drop), "{name}: its chords need one rubbleCrushDrop");
    // the axis (chords aligned with the first) and each chord's midpoint about the centroid
    let first = p(b[0].a) - p(b[0].b);
    let dir = b.map(|x| {
        let d = p(x.a) - p(x.b);
        if d.dot(first) < 0.0 { -d } else { d }
    });
    let t = (dir[0] + dir[1] + dir[2] + dir[3]).normalize();
    let mid = b.map(|x| 0.5 * (p(x.a) + p(x.b)));
    let c = (mid[0] + mid[1] + mid[2] + mid[3]) * 0.25;
    let q = mid.map(|m| (m - c) - t * (m - c).dot(t));
    let u = q.iter().copied().max_by(|a, b| a.length().total_cmp(&b.length())).unwrap().normalize();
    let w = t.cross(u);
    let (suu, sww, suw) = q.iter().fold((0.0f32, 0.0f32, 0.0f32), |s, x| (s.0 + x.dot(u).powi(2), s.1 + x.dot(w).powi(2), s.2 + x.dot(u) * x.dot(w)));
    let th = 0.5 * (2.0 * suw).atan2(suu - sww);
    let (e1, e2) = (u * th.cos() + w * th.sin(), -u * th.sin() + w * th.cos());
    let y = q.map(|x| x.dot(e1));
    let x = q.map(|x| x.dot(e2));
    let dot = |a: [f32; 4], b: [f32; 4]| (0..4).map(|i| a[i] * b[i]).sum::<f32>();
    let unit = |a: [f32; 4]| {
        let n = dot(a, a).sqrt();
        a.map(|v| v / n)
    };
    let less = |a: [f32; 4], m: [f32; 4]| {
        let s = dot(a, m);
        [0, 1, 2, 3].map(|i| a[i] - s * m[i])
    };
    let p0 = [0.5f32; 4];
    let p1 = unit(y);
    let x2 = less(x, p1);
    assert!(dot(x2, x2).sqrt() > 0.05 * dot(y, y).sqrt(), "{name}: its chords lie in one plane");
    let p2 = unit(x2);
    let p3 = (0..4)
        .map(|i| {
            let mut e = [0.0f32; 4];
            e[i] = 1.0;
            less(less(less(e, p0), p1), p2)
        })
        .max_by(|a, b| dot(*a, *a).total_cmp(&dot(*b, *b)))
        .map(unit)
        .unwrap();
    let lever = [dot(p1, y), dot(p2, x)];
    let (m1, m2) = (moment[0] / lever[0], moment[1] / lever[1]);
    let sum = |f: &dyn Fn(&Crush) -> f32| crush.iter().map(f).sum::<f32>();
    let axial = Curve { peak: 0.5 * sum(&|c| c.curve.peak), plateau: 0.5 * sum(&|c| c.curve.plateau), drop: 2.0 * drop };
    assert!(axial.stable(k), "{name}: its drop is steeper than k/2");
    Cell {
        id: crush[0].cell.unwrap().id,
        master: chords[0].1 as u32,
        beams: chords.map(|(j, _)| j as u32),
        phi: [p0, p1, p2, p3],
        cap: [0.5 * b.iter().map(|x| x.tt).sum::<f32>(), m1, m2, m1.min(m2)],
        axial,
        k,
        lever,
        set: 0.0,
        alive: true,
        skip: u64::MAX,
    }
}

impl ExplicitBeams {
    /// After extension record `x` is pushed: when it completes a crush cell (its fourth
    /// chord), build the cell into its first chord's record. Panics on a fifth chord.
    pub(crate) fn crush_attached(&mut self, x: usize) {
        let Some(id) = self.beam_ext[x].crush.and_then(|c| c.cell).map(|c| c.id) else { return };
        let members: Vec<usize> = (0..self.beam_ext.len()).filter(|&i| self.beam_ext[i].crush.and_then(|c| c.cell).is_some_and(|c| c.id == id)).collect();
        assert!(members.len() <= 4, "crush cell {id:#x}: more than four chords");
        if members.len() < 4 {
            return;
        }
        let chords = [0, 1, 2, 3].map(|i| (self.beam_ext[members[i]].beam as usize, members[i]));
        let first = self.beam_ext[members[0]].crush.unwrap().cell.unwrap();
        let moment = [first.cap[1], first.cap[2]];
        let cell = build(self, chords, moment);
        for (n, &m) in members.iter().enumerate() {
            let c = self.beam_ext[m].crush.as_mut().unwrap();
            c.cell = Some(if n == 0 { cell } else { Cell { master: cell.master, ..c.cell.unwrap() } });
        }
    }

    /// A crush cell's chord `j` (record `x`) in the beam pass: true when the cell handled it (the
    /// master computes all four chords; the others skip), false when the chord runs on its own (the
    /// cell dissolved: a chord broke).
    pub(crate) fn crush_cell(&mut self, j: usize, x: usize) -> bool {
        let Some(m) = self.beam_ext[x].crush.and_then(|c| c.cell).map(|c| c.master as usize) else { return false };
        let mut cell = self.beam_ext[m].crush.unwrap().cell.unwrap();
        if !cell.alive {
            return cell.skip == self.steps;
        }
        let ch = cell.beams.map(|b| b as usize);
        if ch.iter().any(|&b| self.beams[b].broken) {
            self.dissolve(m, cell, u64::MAX);
            return false;
        }
        if x != m {
            return true;
        }
        let _ = j;
        self.cell_forces(m, &mut cell);
        true
    }

    /// The cell's chords' forces this step (its master's record `m`).
    fn cell_forces(&mut self, m: usize, cell: &mut Cell) {
        let ch = cell.beams.map(|b| b as usize);
        let k = cell.k;
        let mut d = [Vec3::ZERO; 4];
        let (mut len, mut inv, mut u, mut r, mut floor) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        for i in 0..4 {
            let b = &self.beams[ch[i]];
            let (na, nb) = (self.nodes[b.a as usize], self.nodes[b.b as usize]);
            d[i] = na.pos - nb.pos;
            (len[i], inv[i]) = exact_length(d[i].dot(d[i]));
            u[i] = (na.vel - nb.vel).dot(d[i]) * inv[i];
            r[i] = b.rest;
            floor[i] = self.beam_ext[self.ext_of[ch[i]] as usize - 1].crush.unwrap().floor;
        }
        for mode in 0..4 {
            let phi = cell.phi[mode];
            let g: f32 = (0..4).map(|i| phi[i] * (k * (r[i] - len[i]))).sum();
            let dpi = if mode == 0 {
                if g > 0.0 {
                    -cell.axial.flow(g, k, cell.set)
                } else if g < -cell.cap[0] {
                    (-g - cell.cap[0]) / k
                } else {
                    0.0
                }
            } else if g > cell.cap[mode] {
                -(g - cell.cap[mode]) / k
            } else if g < -cell.cap[mode] {
                (-g - cell.cap[mode]) / k
            } else {
                0.0
            };
            if dpi == 0.0 {
                continue;
            }
            // densified: no chord's rest below its floor (the mode's flow stops there)
            let mut a = 1.0f32;
            for i in 0..4 {
                let step = phi[i] * dpi;
                if step < 0.0 && r[i] + step < floor[i] {
                    a = a.min(((r[i] - floor[i]) / -step).max(0.0));
                }
            }
            for i in 0..4 {
                r[i] += a * phi[i] * dpi;
            }
            if mode == 0 && dpi < 0.0 {
                cell.set -= a * dpi;
            }
        }
        let mut broke = false;
        for i in 0..4 {
            let mut b = self.beams[ch[i]];
            b.rest = r[i];
            let mut f = k * (r[i] - len[i]) - b.c * u[i];
            b.force = f;
            if f.abs() > b.strength {
                b.broken = true;
                f = 0.0;
                broke = true;
            }
            self.beams[ch[i]] = b;
            if self.settings.centred_damping && !b.broken {
                self.damp_slopes[ch[i]] = b.c;
            }
            if b.broken && b.group > 0 {
                self.cascade(b.group);
            }
            let push = d[i] * (f * inv[i]);
            self.nodes[b.a as usize].force += push;
            self.nodes[b.b as usize].force -= push;
        }
        self.beam_ext[m].crush.as_mut().unwrap().cell = Some(*cell);
        if broke {
            let steps = self.steps;
            self.dissolve(m, *cell, steps);
        }
    }

    /// The cell stops acting: each chord carries on alone, its own crush law at the cell's chord set.
    fn dissolve(&mut self, m: usize, mut cell: Cell, skip: u64) {
        cell.alive = false;
        cell.skip = skip;
        for b in cell.beams {
            let x = self.ext_of[b as usize] as usize - 1;
            let c = self.beam_ext[x].crush.as_mut().unwrap();
            c.set = 0.5 * cell.set;
            if x == m {
                c.cell = Some(cell);
            }
        }
    }
}
