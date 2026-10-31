// The threshold destructibles' side of blast loading (`crate::blast`): each blast-loaded
// destructible's faces, exposure and shadowing from its chunks' poses on the GPU, the
// loads of a step's window, the breaks they cause (the threshold rule, decided on the
// host before the step and split on the GPU, `GpuFracture::break_bonds` and
// `encode_split`), and the pieces' impulses.
//
// A child of `threshold::imp`: it reads the threshold slots' state as they keep it.

use super::ThresholdSlot;
use crate::blast::{model, ActiveBlast, ChunkBlastLoad, LoadCause, MIN_SCALED_DISTANCE};
use crate::{DestructibleId, World};
use engine_schema::BodyState;
use glam::{DQuat, DVec3};
use std::collections::BTreeMap;
use stress_gpu::fracture::FrameRecord;
use stress_ref::blast::FaceBlast;
use stress_ref::contact::OBox;
use stress_ref::scene::Scene;
use stress_ref::solver::ReferenceSolver;

/// A face a blast loads: its centre and outward normal in its structure's frame (the
/// chunk's centre plus the face's offset), exposed area, and pulse.
#[derive(Clone, Copy, Debug)]
struct CachedFace {
    chunk: u32,
    point: DVec3,
    normal: DVec3,
    area: f64,
    blast: FaceBlast,
}

/// Floats per chunk in a slot's load buffer (4 float4: force and cause; moment and peak
/// pressure; `Σ sym(r ⊗ F)` xx yy zz xy; xz yz and 0 0).
const LOAD_FLOATS: usize = 16;

/// A blast-loaded threshold destructible's blast state.
pub(crate) struct BlastSlot {
    /// The oracle's solver for the destructible's scene, its bonds' states mirrored from
    /// the GPU's: its exposure and clearing distances are the oracle's.
    mirror: ReferenceSolver,
    /// Per GPU bond: its structure and bond in the mirror.
    bond_source: Vec<(usize, usize)>,
    /// Per chunk: its structure and chunk.
    chunk_of: Vec<(usize, usize)>,
    shadowing: bool,
    clearing: bool,
    /// Per active blast: the faces it loads, as last built.
    faces: Vec<(u64, Vec<CachedFace>)>,
    /// The loads of the last step taken.
    pub last: Vec<ChunkBlastLoad>,
    /// The same per chunk on the GPU (`LOAD_FLOATS` each, world frame, zero where unloaded),
    /// for kernels that compose with them (chunk crushing).
    pub loads: wgpu::Buffer,
    loads_written: bool,
}

impl BlastSlot {
    pub(crate) fn new(
        device: &wgpu::Device,
        mirror: ReferenceSolver,
        scene: &Scene,
        bond_source: Vec<(usize, usize)>,
        chunk_of: Vec<(usize, usize)>,
    ) -> Self {
        let size = (4 * LOAD_FLOATS * chunk_of.len().max(1)) as u64;
        let loads = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threshold blast loads"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            mirror,
            bond_source,
            chunk_of,
            shadowing: scene.sim.features.blast_shadowing,
            clearing: scene.sim.features.blast_clearing,
            faces: Vec::new(),
            last: Vec::new(),
            loads,
            loads_written: false,
        }
    }
}

fn sv(v: DVec3) -> stress_ref::math::Vec3 {
    stress_ref::math::Vec3::new(v.x, v.y, v.z)
}

fn gv(v: stress_ref::math::Vec3) -> DVec3 {
    DVec3::new(v.x, v.y, v.z)
}

fn mat(q: DQuat) -> stress_ref::math::Mat3 {
    let m = glam::DMat3::from_quat(q);
    stress_ref::math::Mat3::from_cols(sv(m.x_axis), sv(m.y_axis), sv(m.z_axis))
}

/// Per chunk of `t`: its structure frame's rotation and origin in the world, from its
/// body's record and the body's frame record (`fracture.slang` `structure_pose`, in
/// rubble's frame: the body's orientation less its principal axes, its centre of mass less
/// the turned centre).
fn chunk_frames(
    t: &ThresholdSlot,
    bodies: &[BodyState],
    frames: &[FrameRecord],
) -> Vec<(DQuat, DVec3)> {
    t.chunk_body
        .iter()
        .map(|&b| {
            let s = &bodies[b as usize];
            let o = s.orientation.map(f64::from);
            let q_body = DQuat::from_xyzw(o[0], o[1], o[2], o[3]).normalize();
            let f = frames[b as usize];
            let com = DVec3::new(f[0][0] as f64, f[0][1] as f64, f[0][2] as f64);
            let p = DQuat::from_xyzw(
                f[1][0] as f64,
                f[1][1] as f64,
                f[1][2] as f64,
                f[1][3] as f64,
            )
            .normalize();
            let q = (q_body * p.conjugate()).normalize();
            let origin = DVec3::from(s.position.map(f64::from)) - q * com;
            (q, origin)
        })
        .collect()
}

/// A blast window's crush damage on each crushable chunk of `t` its faces load
/// (`model::crush_damage`, `engine/docs/destruction-reactions.md` section 1, "Blast"),
/// with its reactions through its intact bonds (less `pending`, broken already) at their
/// area-weighted centroid, as `fracture.slang` `bond_centroid` takes them: per chunk
/// (damage, peak p, peak q, 0) for `fr_crush_blast`. `None`: no crushable chunk is
/// loaded (or the destructible does not crush).
#[allow(clippy::too_many_arguments)]
fn blast_crush(
    t: &ThresholdSlot,
    pending: &[u32],
    crushed: &[bool],
    faces: &[Vec<model::Face>],
    centres: &[DVec3],
    poses: &[(DQuat, DVec3)],
    dt: f64,
) -> Option<Vec<[f32; 4]>> {
    let ct = t.tables.crush.as_ref()?;
    let n = faces.len();
    let can =
        |c: usize| !faces[c].is_empty() && !crushed[c] && ct.chunks[c][1][2].to_bits() & 1 != 0;
    if !(0..n).any(can) {
        return None;
    }
    let mut gone = vec![false; t.intact.len()];
    for &b in pending {
        gone[b as usize] = true;
    }
    // Each loaded chunk's intact bonds: area-weighted centroid (relative to it) and area.
    let mut sums = vec![(DVec3::ZERO, 0.0f64); n];
    for (i, bond) in t.tables.bonds.iter().enumerate() {
        if !t.intact[i] || gone[i] {
            continue;
        }
        let bc = ct.bonds[i];
        let local = DVec3::new(bc[0] as f64, bc[1] as f64, bc[2] as f64);
        for end in [bond[0], bond[1]] {
            let c = end as usize;
            if !can(c) {
                continue;
            }
            let (q, o) = poses[c];
            sums[c].0 += bc[3] as f64 * (o + q * local - centres[c]);
            sums[c].1 += bc[3] as f64;
        }
    }
    let mut words = vec![[0.0f32; 4]; n];
    let mut any = false;
    for c in (0..n).filter(|&c| can(c)) {
        let [m0, m1] = ct.chunks[c];
        let params = stress_ref::crush::CrushParams {
            cap_pressure: m0[0] as f64,
            cohesion: m0[1] as f64,
            friction_slope: m0[2] as f64,
            crush_energy: m0[3] as f64,
            crush_viscosity: m1[0] as f64,
        };
        let bonds = (sums[c].1 > 0.0).then(|| sums[c].0 / sums[c].1);
        let (damage, p, q) = model::crush_damage(
            &faces[c],
            centres[c],
            bonds,
            m1[1] as f64,
            &params,
            0.0,
            dt,
            model::GRADE,
        );
        if damage > 0.0 || p > 0.0 || q > 0.0 {
            words[c] = [damage as f32, p as f32, q as f32, 0.0];
            any = true;
        }
    }
    any.then_some(words)
}

impl World {
    fn blast_slots(&self) -> Vec<usize> {
        self.fracture
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.as_ref().is_some_and(|t| t.blast.is_some()))
            .map(|(k, _)| k)
            .collect()
    }

    /// The chunks' frames of every blast-loaded slot (indexed by slot), read from the GPU.
    fn blast_poses(&self, slots: &[usize]) -> Vec<Vec<(DQuat, DVec3)>> {
        let gpu = self
            .fracture
            .gpu
            .as_ref()
            .expect("a threshold destructible's GPU");
        let bodies = self.sim.download_bodies();
        let mut out = vec![Vec::new(); self.fracture.slots.len()];
        for &k in slots {
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let frames = t.fracture.body_frames(gpu);
            out[k] = chunk_frames(t, &bodies, &frames);
        }
        out
    }

    /// Bonds broken across the blast-loaded destructibles (their topology).
    fn blast_version(&self, slots: &[usize]) -> u64 {
        slots
            .iter()
            .map(|&k| {
                let t = self.fracture.slots[k].as_ref().expect("live slot");
                t.intact.iter().filter(|b| !**b).count() as u64
            })
            .sum()
    }

    /// Build the face caches a blast lacks, or every blast's after the topology changed
    /// (stress-ref `World::rebuild_blast`'s rule), and each blast's end.
    fn blast_refresh(
        &mut self,
        slots: &[usize],
        poses: &[Vec<(DQuat, DVec3)>],
    ) -> anyhow::Result<()> {
        let version = self.blast_version(slots);
        let all = self.blasts.version != Some(version);
        self.blasts.version = Some(version);
        // The mirrors' bonds as the GPU holds them.
        for &k in slots {
            let t = self.fracture.slots[k].as_mut().expect("live slot");
            let b = t.blast.as_mut().expect("blast-loaded");
            for (i, &(s, bi)) in b.bond_source.iter().enumerate() {
                b.mirror.bonds[s][bi].alive = t.intact[i];
            }
        }
        // Every blast-loaded chunk's box in the world: the occluders.
        let mut boxes: Vec<(usize, u32, OBox)> = Vec::new();
        for &k in slots {
            let t = self.fracture.slots[k].as_ref().expect("live slot");
            let b = t.blast.as_ref().expect("blast-loaded");
            for (c, &(s, sc)) in b.chunk_of.iter().enumerate() {
                let ch = &b.mirror.structures[s].chunks[sc];
                let (q, o) = poses[k][c];
                boxes.push((
                    k,
                    c as u32,
                    OBox {
                        center: sv(o + q * gv(ch.center)),
                        rotation: mat(q) * ch.rotation,
                        half: ch.half_extents,
                        hull: ch.hull.clone(),
                        parts: ch.parts.clone(),
                    },
                ));
            }
        }
        let blasts: Vec<ActiveBlast> = self.blasts.active.clone();
        for blast in &blasts {
            let missing = slots.iter().any(|&k| {
                let b = self.fracture.slots[k]
                    .as_ref()
                    .unwrap()
                    .blast
                    .as_ref()
                    .unwrap();
                !b.faces.iter().any(|f| f.0 == blast.id)
            });
            if !(all || missing) {
                continue;
            }
            let position = DVec3::from(blast.position);
            let near = |b: &OBox| (gv(b.center) - position).length() - b.bounding_radius();
            let mut by_reach: Vec<(f64, usize)> = boxes
                .iter()
                .enumerate()
                .map(|(i, b)| (near(&b.2), i))
                .collect();
            let finite_reach = by_reach.iter().all(|r| r.0.is_finite());
            by_reach.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut end: f64 = 0.0;
            for &k in slots {
                let faces =
                    self.blast_faces(k, blast, &poses[k], &boxes, &by_reach, finite_reach)?;
                end = faces
                    .iter()
                    .map(|f| f.blast.arrival + f.blast.duration)
                    .fold(end, f64::max);
                let b = self.fracture.slots[k]
                    .as_mut()
                    .unwrap()
                    .blast
                    .as_mut()
                    .unwrap();
                b.faces.retain(|f| f.0 != blast.id);
                b.faces.push((blast.id, faces));
            }
            if let Some(a) = self.blasts.active.iter_mut().find(|a| a.id == blast.id) {
                a.end = end;
            }
        }
        Ok(())
    }

    /// The faces of slot `k` a blast loads (stress-gpu `blast_faces`, stress-ref
    /// `rebuild_blast`): every exposed face turned towards the charge (a face turned away
    /// carries no pressure), shadowed when its line of sight crosses another chunk.
    #[allow(clippy::too_many_arguments)]
    fn blast_faces(
        &self,
        k: usize,
        blast: &ActiveBlast,
        poses: &[(DQuat, DVec3)],
        boxes: &[(usize, u32, OBox)],
        by_reach: &[(f64, usize)],
        finite_reach: bool,
    ) -> anyhow::Result<Vec<CachedFace>> {
        let t = self.fracture.slots[k].as_ref().expect("live slot");
        let b = t.blast.as_ref().expect("blast-loaded");
        let m = &b.mirror;
        let exposure = stress_gpu::loads::exposure(m);
        let clearing: Vec<_> = (0..m.structures.len())
            .map(|s| {
                b.clearing
                    .then(|| stress_gpu::loads::clearing_distances(m, &exposure, s))
            })
            .collect();
        let position = DVec3::from(blast.position);
        let w_third = blast.tnt.cbrt();
        let mut out = Vec::new();
        for (c, &(s, sc)) in b.chunk_of.iter().enumerate() {
            let ch = &m.structures[s].chunks[sc];
            let (q, o) = poses[c];
            for (f, face) in ch.faces.iter().enumerate() {
                let area = exposure[s][sc][f];
                if area <= 0.0 {
                    continue;
                }
                let local = gv(ch.center + face.offset);
                let x = o + q * local;
                let n = q * gv(face.normal);
                let to_charge = position - x;
                let r = to_charge.length().max(1e-6);
                let z = r / w_third;
                anyhow::ensure!(
                    z >= MIN_SCALED_DISTANCE,
                    "blast: a face of chunk {c} is {r:.4} m from the charge, scaled distance {z:.4} m/kg^(1/3), nearer than the Kinney-Graham fits reach ({MIN_SCALED_DISTANCE})"
                );
                let cos_theta = n.dot(to_charge) / r;
                if cos_theta <= 0.0 {
                    continue;
                }
                let start = x + n * 1e-6;
                let blocks = |i: usize| {
                    let (bk, bc, ob) = &boxes[i];
                    (*bk, *bc) != (k, c as u32) && ob.segment_hit(sv(start), sv(position)).is_some()
                };
                let shadowed = b.shadowing && {
                    let reach = (start - position).length();
                    let scale = start
                        .to_array()
                        .iter()
                        .chain(position.to_array().iter())
                        .map(|x| x.abs())
                        .fold(reach, f64::max);
                    let limit = reach * (1.0 + 1e-9) + 1e-9 * scale;
                    if finite_reach && limit.is_finite() {
                        let end = by_reach.partition_point(|r| r.0 <= limit);
                        by_reach[..end].iter().any(|&(_, i)| blocks(i))
                    } else {
                        (0..boxes.len()).any(blocks)
                    }
                };
                let s_clear = match &clearing[s] {
                    Some(d) => d.get(&(sc, f)).copied().unwrap_or(0.0),
                    None => f64::INFINITY,
                };
                out.push(CachedFace {
                    chunk: c as u32,
                    point: local,
                    normal: gv(face.normal),
                    area,
                    blast: FaceBlast::new(blast.tnt, 0.0, r, cos_theta, shadowed, s_clear),
                });
            }
        }
        Ok(out)
    }

    /// At `detonate`: the new blast's faces, from the GPU's state now (the start of the
    /// next step), so a face out of the fits' range is the call's error.
    pub(crate) fn blast_prepare(&mut self) -> anyhow::Result<()> {
        let slots = self.blast_slots();
        if slots.is_empty() {
            return Ok(());
        }
        self.blast_settle()?;
        let poses = self.blast_poses(&slots);
        self.blast_refresh(&slots, &poses)
    }

    /// Drop a blast's face caches.
    pub(crate) fn blast_forget(&mut self, id: u64) {
        for t in self.fracture.slots.iter_mut().flatten() {
            if let Some(b) = t.blast.as_mut() {
                b.faces.retain(|f| f.0 != id);
            }
        }
    }

    /// Before a step's engine work: every active blast's loads over the step's window on
    /// every blast-loaded destructible, the bonds they break (split on the GPU now, so the
    /// step meets the pieces), and the pieces' impulses; blasts whose pulses have ended
    /// retire.
    pub(crate) fn blast_before_step(&mut self) -> anyhow::Result<()> {
        if self.blasts.active.is_empty() {
            self.blast_idle();
            return Ok(());
        }
        let slots = self.blast_slots();
        let step = self.step;
        let dt = f64::from(self.sim.params().dt);
        if !slots.is_empty() {
            self.blast_settle()?;
            let poses = self.blast_poses(&slots);
            self.blast_refresh(&slots, &poses)?;
            for &k in &slots {
                self.blast_slot_step(k, step, dt, &poses[k])?;
            }
        }
        let done: Vec<u64> = self
            .blasts
            .active
            .iter()
            .filter(|b| (step + 1 - b.start) as f64 * dt >= b.end)
            .map(|b| b.id)
            .collect();
        for id in done {
            self.blasts.active.retain(|b| b.id != id);
            self.blast_forget(id);
        }
        self.dirty = true;
        Ok(())
    }

    /// A step no blast reaches: no loads reported, the GPU's copy zeroed once.
    fn blast_idle(&mut self) {
        let Some(gpu) = self.fracture.gpu.clone() else {
            return;
        };
        for t in self.fracture.slots.iter_mut().flatten() {
            if let Some(b) = t.blast.as_mut() {
                b.last.clear();
                if b.loads_written {
                    let zeros = vec![0f32; LOAD_FLOATS * b.chunk_of.len()];
                    gpu.queue
                        .write_buffer(&b.loads, 0, bytemuck::cast_slice(&zeros));
                    b.loads_written = false;
                }
            }
        }
    }

    fn blast_slot_step(
        &mut self,
        k: usize,
        step: u64,
        dt: f64,
        poses: &[(DQuat, DVec3)],
    ) -> anyhow::Result<()> {
        let gpu = self
            .fracture
            .gpu
            .clone()
            .expect("a threshold destructible's GPU");
        let t = self.fracture.slots[k].as_ref().expect("live slot");
        let b = t.blast.as_ref().expect("blast-loaded");
        let n = b.chunk_of.len();
        // Every active blast's faces over this step's window, as pulses in step time.
        let mut faces: Vec<Vec<model::Face>> = vec![Vec::new(); n];
        for (id, list) in &b.faces {
            let Some(blast) = self.blasts.active.iter().find(|a| a.id == *id) else {
                continue;
            };
            let t0 = (step - blast.start) as f64 * dt;
            for f in list {
                let mut fb = f.blast;
                fb.arrival -= t0;
                if fb.arrival >= dt || fb.arrival + fb.duration <= 0.0 {
                    continue;
                }
                let (q, o) = poses[f.chunk as usize];
                faces[f.chunk as usize].push(model::Face {
                    point: o + q * f.point,
                    normal: q * f.normal,
                    area: f.area,
                    blast: fb,
                });
            }
        }
        let peak = t.tables.impact_peak;
        let mut windows = vec![model::Window::default(); n];
        let mut centres = vec![DVec3::ZERO; n];
        for c in 0..n {
            let (s, sc) = b.chunk_of[c];
            let (q, o) = poses[c];
            centres[c] = o + q * gv(b.mirror.structures[s].chunks[sc].center);
            windows[c] = model::window(&faces[c], centres[c], 0.0, dt, peak);
        }
        // Bonds support loss broke that no readback has shown yet: not the blast's, and they
        // break with its split if the GPU has not broken them yet.
        let pending: Vec<u32> = match &t.support {
            Some(sl) => (0..t.intact.len() as u32)
                .filter(|&i| sl.sent[i as usize] && t.intact[i as usize])
                .collect(),
            None => Vec::new(),
        };
        // Crushing: each crushable chunk's damage over the window, from its faces and its
        // bonds as the window's start meets them (before the breaks below).
        let crushed = &self.destructibles[t.destructible as usize].crushed;
        let crush_words = blast_crush(t, &pending, crushed, &faces, &centres, poses, dt);
        // The threshold rule (`fr_breaks`): a bond breaks when the load on either of its
        // chunks beats its tensile capacity.
        let demand: Vec<f32> = windows.iter().map(|w| w.force.length() as f32).collect();
        let broken: Vec<u32> = (0..t.intact.len() as u32)
            .filter(|&i| {
                let bond = t.tables.bonds[i as usize];
                t.intact[i as usize]
                    && !pending.contains(&i)
                    && demand[bond[0] as usize].max(demand[bond[1] as usize])
                        > f32::from_bits(bond[2])
            })
            .collect();
        let loads: Vec<ChunkBlastLoad> = windows
            .iter()
            .enumerate()
            .filter(|(_, w)| !w.is_zero())
            .map(|(c, w)| ChunkBlastLoad {
                chunk: c as u32,
                force: w.force.as_vec3().to_array(),
                moment: w.moment.as_vec3().to_array(),
                stress_volume: w.stress_volume.map(|x| x as f32),
                impulse: w.impulse.as_vec3().to_array(),
                angular_impulse: w.angular_impulse.as_vec3().to_array(),
                peak_pressure: w.peak_pressure as f32,
                cause: LoadCause::Blast,
            })
            .collect();
        let mut crushes: Vec<stress_gpu::fracture::CrushEvent> = Vec::new();
        let split = !broken.is_empty() || crush_words.is_some();
        if split {
            let t = self.fracture.slots[k].as_mut().expect("live slot");
            let all: Vec<u32> = broken.iter().chain(&pending).copied().collect();
            // Broken on the GPU with the split below, each counted once.
            t.fracture.break_bonds_sparse(&all);
            if let Some(words) = &crush_words {
                t.fracture
                    .set_crush_blast(&gpu, words)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("blast split"),
                });
            t.fracture.encode_split_crush(
                &gpu,
                &mut enc,
                self.sim.steps_done(),
                crush_words.is_some(),
            );
            gpu.queue.submit([enc.finish()]);
            let counters = t.fracture.counters(&gpu);
            anyhow::ensure!(
                counters.error & stress_gpu::fracture::CRUSH_ERROR_FRAGMENTS == 0,
                "step {step}: a blast's chunk crush found the fragment bodies spent: raise CrushMass::Conserve's budget"
            );
            anyhow::ensure!(
                counters.error & stress_gpu::fracture::CRUSH_ERROR_EVENTS == 0,
                "step {step}: a blast crushed more chunks in one step than Crush::max_per_step"
            );
            anyhow::ensure!(
                counters.error == 0,
                "step {step}: blast split error {:#x} (1: no free body slot, 2: mass sum out of range, 4: union did not settle)",
                counters.error
            );
            if crush_words.is_some() {
                crushes = t.fracture.crush_events(&gpu);
            }
            let chunks = t.fracture.chunk_bodies(&gpu);
            let intact = t.fracture.bonds_intact(&gpu);
            let frames: Vec<[f32; 4]> =
                t.fracture.body_frames(&gpu).into_iter().flatten().collect();
            let crushed_now: Vec<u32> = crushes.iter().map(|e| e.chunk).collect();
            // The pieces take their frames from this step on: the step simulates them.
            let mut events = self.apply_fracture_from(
                k,
                step,
                chunks,
                intact,
                &frames,
                &crushed_now,
                crate::BreakCause::Blast,
            )?;
            // The frames as of the split, for the change log's later steps.
            let t = self.fracture.slots[k].as_mut().expect("live slot");
            t.frames = frames.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            let all_broken: Vec<u32> = events
                .iter()
                .filter_map(|e| match e {
                    crate::DestructionEvent::BondsBroken { bonds, .. } => Some(bonds.clone()),
                    _ => None,
                })
                .flatten()
                .collect();
            events.extend(self.apply_crush(k, step, &crushes, &frames)?);
            // The weight the breaks left without a path: support loss, its breaks split
            // with this step's frame.
            if !all_broken.is_empty() {
                self.support_loss(k, &all_broken);
            }
            self.blasts
                .events
                .extend(events.into_iter().map(|e| (step, e)));
        }
        // Each body's share of its chunks' impulses, about its centre of mass as it is now.
        let bodies = if split {
            Some(self.sim.download_bodies())
        } else {
            None
        };
        let t = self.fracture.slots[k].as_ref().expect("live slot");
        let mut per_body: BTreeMap<u32, (DVec3, DVec3)> = BTreeMap::new();
        for l in &loads {
            let c = l.chunk as usize;
            let w = &windows[c];
            // The moment about the body's centre is added below, once it is known.
            let about_origin = w.angular_impulse + centres[c].cross(w.impulse);
            // A crushed chunk's share goes to its fragments, equally (they are centred on
            // its centre of mass: the total's moment is kept); removed, it leaves with its
            // mass.
            if let Some(e) = crushes.iter().find(|e| e.chunk == l.chunk) {
                let (Some(first), Some(cs)) = (e.first_fragment, t.crush.as_ref()) else {
                    continue;
                };
                let n = cs.fragments as f64;
                for i in 0..cs.fragments {
                    let body = cs.pool[(first + i) as usize].0;
                    let entry = per_body.entry(body).or_default();
                    entry.0 += w.impulse / n;
                    entry.1 += about_origin / n;
                }
                continue;
            }
            let entry = per_body.entry(t.chunk_body[c]).or_default();
            entry.0 += w.impulse;
            entry.1 += about_origin;
        }
        let fresh;
        let bodies: &[BodyState] = match &bodies {
            Some(v) => v,
            None => {
                fresh = self.sim.download_bodies();
                &fresh
            }
        };
        let mut records = Vec::new();
        for (body, (j, l)) in per_body {
            let s = bodies[body as usize];
            if s.inv_mass <= 0.0 || j == DVec3::ZERO && l == DVec3::ZERO {
                continue;
            }
            let com = DVec3::from(s.position.map(f64::from));
            // Σ (x_c - com) × J_c = Σ x_c × J_c - com × Σ J_c.
            let about = l - com.cross(j);
            records.push((body, j.as_vec3().to_array(), about.as_vec3().to_array()));
        }
        anyhow::ensure!(
            records.len() <= engine_schema::IMPULSE_CAP as usize,
            "step {step}: a blast pushes {} bodies; one step's impulse records hold {}",
            records.len(),
            engine_schema::IMPULSE_CAP
        );
        for (body, j, l) in records {
            self.sim.apply_linear_angular_impulse(body, j, l);
        }
        // The GPU's copy for kernels that compose with the loads.
        let t = self.fracture.slots[k].as_mut().expect("live slot");
        let b = t.blast.as_mut().expect("blast-loaded");
        if !loads.is_empty() || b.loads_written {
            let mut words = vec![0f32; LOAD_FLOATS * n];
            for l in &loads {
                let w = &mut words[LOAD_FLOATS * l.chunk as usize..][..LOAD_FLOATS];
                w[..3].copy_from_slice(&l.force);
                w[3] = f32::from_bits(LoadCause::Blast.code());
                w[4..7].copy_from_slice(&l.moment);
                w[7] = l.peak_pressure;
                w[8..12].copy_from_slice(&l.stress_volume[..4]);
                w[12..14].copy_from_slice(&l.stress_volume[4..]);
            }
            gpu.queue
                .write_buffer(&b.loads, 0, bytemuck::cast_slice(&words));
            b.loads_written = !loads.is_empty();
        }
        b.last = loads;
        Ok(())
    }

    /// The blast loads of a blast-loaded threshold destructible's chunks in the last step
    /// taken (only the loaded chunks; empty in a step no blast reached).
    pub fn destructible_blast_loads(
        &self,
        id: DestructibleId,
    ) -> anyhow::Result<Vec<ChunkBlastLoad>> {
        let d = &self.destructibles[self.live_destructible(id)?];
        let k = d
            .threshold
            .ok_or_else(|| anyhow::anyhow!("blast loads: not a threshold destructible"))?;
        let t = self.fracture.slots[k as usize].as_ref().expect("live slot");
        let b = t.blast.as_ref().ok_or_else(|| {
            anyhow::anyhow!("blast loads: the destructible was placed without blast_loading")
        })?;
        Ok(b.last.clone())
    }
}
