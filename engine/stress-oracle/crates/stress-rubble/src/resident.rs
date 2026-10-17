//! rubble as the resident frame's engine (stress-gpu `resident::ResidentEngine`,
//! docs/stress-fracture-gpu-resident.md phase 3): each substep is submitted without
//! waiting (`WgpuSim::step_submit`, counts forwarded on the GPU) and its contact rows go
//! straight into the GPU ingest; nothing is read back.

use stress_gpu::gpu::Gpu;
use stress_gpu::ingest::{ColliderInfo, EngineBuffers, EngineIngest, KIND_BODY, KIND_STATIC, NONE};
use stress_gpu::resident::{EngineFrame, OtherProps, ResidentEngine};
use stress_ref::engine::BodyKey;

use crate::ingest_check::body_code;
use crate::{Owner, RubbleEngine};

impl ResidentEngine for RubbleEngine {
    fn engine_buffers(&self) -> EngineBuffers<'_> {
        EngineBuffers { bodies: self.sim.body_buffer(), colliders: self.sim.collider_buffer(), contacts: self.sim.contact_buffer(), counters: self.sim.counters_buffer() }
    }

    fn engine_capacity(&self) -> (u32, u32) {
        let c = self.sim.capacity();
        (c.max_contacts, c.max_bodies)
    }

    fn layout_version(&self) -> u64 {
        self.layout_version
    }

    fn collider_table(&self, chunk_of: &dyn Fn(usize, usize) -> Option<u32>, other: OtherProps<'_>) -> Vec<ColliderInfo> {
        self.owners
            .iter()
            .map(|o| match *o {
                Owner::Body { key, chunk } => {
                    let gpu_chunk = chunk.and_then(|(s, c)| chunk_of(s, c));
                    let p = other(Some(key));
                    ColliderInfo { a: [gpu_chunk.unwrap_or(NONE), body_code(key), KIND_BODY, gpu_chunk.is_some() as u32], b: [p[0], p[1], p[2], 0.0] }
                }
                Owner::Ground | Owner::Retired => {
                    let p = other(None);
                    ColliderInfo { a: [NONE, NONE, KIND_STATIC, 0], b: [p[0], p[1], p[2], 0.0] }
                }
            })
            .collect()
    }

    fn engine_frame(&self, key: BodyKey) -> Option<EngineFrame> {
        let e = self.bodies.get(&key)?;
        (!e.fixed).then_some(EngineFrame { body: e.id, principal: e.principal, com: e.com })
    }

    fn resident_substeps(&mut self, dt: f64) -> u32 {
        let mut n = self.substeps.max(1);
        let ccd: Vec<u32> = self.bodies.iter().filter(|(k, e)| e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k)).map(|(_, e)| e.id).collect();
        if !ccd.is_empty() {
            // The CCD bound needs the fast bodies' speeds: read back (counted).
            self.resident_reads += 1;
            let all = self.bodies_now();
            for (_, e) in self.bodies.iter().filter(|(k, e)| e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k)) {
                let s = &all[e.id as usize];
                let travel = glam::Vec3::from_array(s.linear_vel).length() as f64 * dt + glam::Vec3::from_array(s.angular_vel).length() as f64 * e.thin * dt;
                let need = (travel / (0.5 * e.thin)).ceil() as u32;
                n = n.max(need.min(self.max_substeps));
            }
        }
        if !self.bodies.iter().any(|(k, e)| e.inv_mass > 0.0 && !self.frozen.contains(k)) {
            return 0;
        }
        n
    }

    fn step_resident(&mut self, gpu: &Gpu, dt: f64, n: u32, ingest: &EngineIngest) -> Result<(), String> {
        if !self.frozen.is_empty() {
            return Err("resident frame: the engine holds frozen bodies".into());
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        ingest.begin_frame(&mut encoder, &self.engine_buffers());
        gpu.queue.submit([encoder.finish()]);
        if n > 0 {
            let h = dt / n as f64;
            let mut params = self.sim.params();
            params.dt = h as f32;
            params.contact_offset = 0.01;
            self.sim.set_params(params);
            for s in 0..n {
                self.sim.step_submit().map_err(|e| format!("rubble step failed: {e:?}"))?;
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                ingest.substep(gpu, &mut encoder, &self.engine_buffers(), self.sim.last_contact_offset(), s, h);
                gpu.queue.submit([encoder.finish()]);
            }
            self.bodies_changed();
        }
        self.last_substeps = n;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        ingest.merge(gpu, &mut encoder, &self.engine_buffers(), n);
        gpu.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn resident_finish(&mut self) -> Result<(), String> {
        self.sim.drain().map(|_| ()).map_err(|e| format!("rubble step failed: {e:?} (capacity overflow)"))
    }
}
