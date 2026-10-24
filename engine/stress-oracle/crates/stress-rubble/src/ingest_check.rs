//! The GPU contact ingest (stress-gpu `ingest`, docs/stress-fracture-gpu-resident.md
//! phase 2) checked against the host's: with a check attached, every engine step also
//! runs the ingest on rubble's own buffers, reads its manifolds back and compares them
//! with the manifolds the host builds from `contacts_of_step` (stress-ref `manifolds`'
//! rule). A validation harness, not the resident path: it reads back on purpose.

use std::collections::HashMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::ingest::{ColliderInfo, EngineBuffers, EngineIngest, KIND_BODY, KIND_STATIC, NONE};
use stress_ref::engine::{BodyKey, EngineContact};
use stress_ref::math::Vec3;

use crate::{Owner, RubbleEngine};

pub(crate) fn body_code(k: BodyKey) -> u32 {
    match k {
        BodyKey::Cluster(id) => (id as u32) << 1,
        BodyKey::Impactor(i) => ((i as u32) << 1) | 1,
    }
}

/// What a check has compared so far.
#[derive(Clone, Copy, Debug, Default)]
pub struct IngestStats {
    /// Engine steps checked.
    pub steps: u64,
    /// Most substeps in a checked step.
    pub substeps: u32,
    /// Manifolds compared.
    pub manifolds: u64,
    /// Steps whose manifold sets differ (chunk, count or other bodies).
    pub mismatched_steps: u64,
    /// Worst impulse error relative to the manifold's impulse (or 1 N s).
    pub impulse: f64,
    /// Worst point error (m).
    pub point: f64,
    /// Worst closing-speed error (m/s).
    pub closing: f64,
}

pub struct IngestCheck {
    gpu: Gpu,
    ingest: Option<(EngineIngest, u32)>,
    chunks: HashMap<(usize, usize), u32>,
    pub stats: std::rc::Rc<std::cell::Cell<IngestStats>>,
}

impl IngestCheck {
    pub fn new(gpu: Gpu) -> IngestCheck {
        IngestCheck { gpu, ingest: None, chunks: HashMap::new(), stats: Default::default() }
    }
}

impl RubbleEngine {
    /// Run and check the GPU contact ingest with every step (`gpu`: this engine's
    /// device); the returned cell follows what the check has compared.
    pub fn attach_ingest_check(&mut self, gpu: Gpu) -> std::rc::Rc<std::cell::Cell<IngestStats>> {
        let check = IngestCheck::new(gpu);
        let stats = check.stats.clone();
        self.ingest_check = Some(check);
        stats
    }

    fn engine_buffers(&self) -> EngineBuffers<'_> {
        EngineBuffers { bodies: self.sim.body_buffer(), colliders: self.sim.collider_buffer(), contacts: self.sim.contact_buffer(), counters: self.sim.counters_buffer() }
    }

    /// The collider table and chunk numbering of the current owners.
    fn ingest_table(&self) -> (Vec<ColliderInfo>, HashMap<(usize, usize), u32>) {
        let mut chunks = HashMap::new();
        let table = self
            .owners
            .iter()
            .map(|o| match *o {
                Owner::Body { key, chunk } => {
                    let index = chunk.map_or(NONE, |c| {
                        let next = chunks.len() as u32;
                        *chunks.entry(c).or_insert(next)
                    });
                    ColliderInfo { a: [index, body_code(key), KIND_BODY, chunk.is_some() as u32], b: [0.0; 4] }
                }
                Owner::Ground | Owner::Retired | Owner::Foreign => ColliderInfo { a: [NONE, NONE, KIND_STATIC, 0], b: [0.0; 4] },
            })
            .collect();
        (table, chunks)
    }

    /// Before the step's first substep (after its body writes).
    pub(crate) fn ingest_begin(&mut self, substeps: u32) {
        if self.ingest_check.is_none() {
            return;
        }
        let (table, chunks) = self.ingest_table();
        let capacity = self.sim.capacity();
        let check = self.ingest_check.as_mut().unwrap();
        let fits = check.ingest.as_ref().is_some_and(|(_, cap)| *cap >= substeps) && check.chunks.len() == chunks.len();
        if !fits {
            let cap = substeps.next_power_of_two();
            let ingest = EngineIngest::new(&check.gpu, capacity.max_contacts, cap, capacity.max_bodies, chunks.len().max(1) as u32, &table);
            check.ingest = Some((ingest, cap));
        } else {
            check.ingest.as_mut().unwrap().0.set_colliders(&check.gpu, &table);
        }
        check.chunks = chunks;
        let check = self.ingest_check.as_ref().unwrap();
        let mut encoder = check.gpu.device.create_command_encoder(&Default::default());
        check.ingest.as_ref().unwrap().0.begin_frame(&mut encoder, &self.engine_buffers());
        check.gpu.queue.submit([encoder.finish()]);
    }

    /// After substep `s` (of length `h`).
    pub(crate) fn ingest_substep(&self, s: u32, h: f64) {
        let Some(check) = self.ingest_check.as_ref() else { return };
        let mut encoder = check.gpu.device.create_command_encoder(&Default::default());
        check.ingest.as_ref().unwrap().0.substep(&check.gpu, &mut encoder, &self.engine_buffers(), self.sim.last_contact_offset(), s, h);
        check.gpu.queue.submit([encoder.finish()]);
    }

    /// After the step's `substeps`: merge on the GPU, read back, compare with `host`.
    pub(crate) fn ingest_end(&mut self, substeps: u32, host: &[EngineContact]) {
        let Some(check) = self.ingest_check.as_ref() else { return };
        let ingest = &check.ingest.as_ref().unwrap().0;
        let mut encoder = check.gpu.device.create_command_encoder(&Default::default());
        ingest.merge(&check.gpu, &mut encoder, &self.engine_buffers(), substeps);
        check.gpu.queue.submit([encoder.finish()]);
        let gpu_manifolds = ingest.read_manifolds(&check.gpu);
        // The host's manifolds, per chunk, by stress-ref `manifolds`' rule.
        struct M {
            other: u32,
            point: Vec3,
            impulse: Vec3,
            closing: f64,
            weight: f64,
        }
        let mut want: Vec<Vec<M>> = (0..gpu_manifolds.len()).map(|_| Vec::new()).collect();
        for c in host {
            let k = check.chunks[&(c.structure, c.chunk)] as usize;
            let other = c.other.map_or(NONE, body_code);
            let w = c.impulse.norm();
            match want[k].iter_mut().find(|m| m.other == other) {
                Some(m) => {
                    let total = m.weight + w;
                    if total > 0.0 {
                        m.point = (m.point * m.weight + c.point * w) * (1.0 / total);
                        m.closing = (m.closing * m.weight + c.approach_speed * w) / total;
                    }
                    m.impulse += c.impulse;
                    m.weight = total;
                }
                None => want[k].push(M { other, point: c.point, impulse: c.impulse, closing: c.approach_speed, weight: w }),
            }
        }
        let mut stats = check.stats.get();
        stats.steps += 1;
        stats.substeps = stats.substeps.max(substeps);
        let mut same = true;
        // Each chunk's manifolds by other body: their order is the contacts' (the GPU's
        // by contact identity, the host's by engine row, which the engine's atomic
        // appends order differently from run to run), not part of what is compared.
        let mut gpu_manifolds = gpu_manifolds;
        let mut want = want;
        for g in gpu_manifolds.iter_mut() {
            g.sort_by_key(|g| g.ids[1]);
        }
        for w in want.iter_mut() {
            w.sort_by_key(|w| w.other);
        }
        for (got, want) in gpu_manifolds.iter().zip(&want) {
            if got.len() != want.len() || got.iter().zip(want).any(|(g, w)| g.ids[1] != w.other) {
                same = false;
                continue;
            }
            for (g, w) in got.iter().zip(want) {
                let gi = Vec3::new(g.impulse[0] as f64, g.impulse[1] as f64, g.impulse[2] as f64);
                let gp = Vec3::new(g.point[0] as f64, g.point[1] as f64, g.point[2] as f64);
                stats.manifolds += 1;
                stats.impulse = stats.impulse.max((gi - w.impulse).norm() / w.impulse.norm().max(1.0));
                stats.point = stats.point.max((gp - w.point).norm());
                stats.closing = stats.closing.max((g.impulse[3] as f64 - w.closing).abs());
            }
        }
        if !same {
            stats.mismatched_steps += 1;
        }
        check.stats.set(stats);
    }
}
