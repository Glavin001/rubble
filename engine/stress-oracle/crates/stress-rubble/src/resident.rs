//! rubble as the resident frame's engine (stress-gpu `resident::ResidentEngine`,
//! docs/stress-fracture-gpu-resident.md phase 3): each substep is submitted without
//! waiting (`WgpuSim::step_submit`, counts forwarded on the GPU) and its contact rows go
//! straight into the GPU ingest; nothing is read back.

use std::collections::HashMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::ingest::{ColliderInfo, EngineBuffers, EngineIngest, KIND_BODY, KIND_STATIC, NONE};
use stress_gpu::resident::{EngineFrame, OtherProps, ResidentEngine};
use stress_ref::engine::BodyKey;
use stress_ref::math::Vec3;

use crate::ingest_check::body_code;
use crate::{Owner, RubbleEngine};
use engine_schema::BodyState;

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
        self.collider_table_range(0..self.owners.len() as u32, chunk_of, other)
    }

    fn collider_table_range(&self, range: std::ops::Range<u32>, chunk_of: &dyn Fn(usize, usize) -> Option<u32>, other: OtherProps<'_>) -> Vec<ColliderInfo> {
        self.owners[range.start as usize..range.end as usize]
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

    fn collider_changes_since(&self, version: u64) -> Option<Vec<std::ops::Range<u32>>> {
        self.collider_changes(version)
    }

    fn last_row_offset(&self) -> u32 {
        self.sim.last_contact_offset()
    }

    fn engine_frame(&self, key: BodyKey) -> Option<EngineFrame> {
        let e = self.bodies.get(&key)?;
        (!e.fixed).then_some(EngineFrame { body: e.id, principal: e.principal, com: e.com })
    }

    fn resident_substeps(&mut self, dt: f64) -> u32 {
        let mut n = self.substeps.max(1);
        if self.needs_ccd_bodies() {
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
        // `STRESS_RUBBLE_MAX_SUBSTEPS=n` (a measurement of the CCD policy, not adopted):
        // at most n substeps a frame (1: no CCD, as vibe-land's PhysX runs).
        if let Some(m) = std::env::var("STRESS_RUBBLE_MAX_SUBSTEPS").ok().and_then(|v| v.parse::<u32>().ok()) {
            n = n.min(m.max(1));
        }
        // `STRESS_RUBBLE_SPECULATIVE=1` (as `RigidEngine::step`; measured, not adopted):
        // one step whose speculative contact range covers the farthest a CCD body travels.
        self.resident_offset = 0.01;
        if n > 1 && std::env::var("STRESS_RUBBLE_SPECULATIVE").is_ok_and(|v| v == "1") {
            let all = self.bodies_now();
            let mut travel = 0.0f64;
            for (_, e) in self.bodies.iter().filter(|(k, e)| e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k)) {
                let s = &all[e.id as usize];
                travel = travel.max(glam::Vec3::from_array(s.linear_vel).length() as f64 * dt + glam::Vec3::from_array(s.angular_vel).length() as f64 * e.thin * dt);
            }
            self.resident_offset = self.resident_offset.max(travel as f32);
            return 1;
        }
        n
    }

    fn step_resident(&mut self, gpu: &Gpu, dt: f64, n: u32, ingest: &EngineIngest, encoder: &mut wgpu::CommandEncoder) -> Result<(), String> {
        if !self.frozen.is_empty() {
            return Err("resident frame: the engine holds frozen bodies".into());
        }
        // Each part its own submission (the default); `STRESS_FRAME_SUBMITS=one` (plan 5d,
        // in progress): into the frame's encoder.
        if !stress_gpu::one_command_buffer() {
            return self.step_resident_submitting(gpu, dt, n, ingest);
        }
        ingest.begin_frame(encoder, &self.engine_buffers());
        if n > 0 {
            let h = dt / n as f64;
            let mut params = self.sim.params();
            params.dt = h as f32;
            params.contact_offset = self.resident_offset;
            self.sim.set_params(params);
            for s in 0..n {
                // rubble writes its host uniforms per step (they land at the submission):
                // a second substep's go after a submission of what came before.
                if s > 0 {
                    let done = std::mem::replace(encoder, gpu.device.create_command_encoder(&Default::default()));
                    engine_gpu_io::submit();
                    gpu.queue.submit([done.finish()]);
                }
                self.sim.step_encode(encoder).map_err(|e| format!("rubble step failed: {e:?}"))?;
                ingest.substep(gpu, encoder, &self.engine_buffers(), self.sim.last_contact_offset(), s, h);
            }
            self.bodies_changed();
        }
        self.last_substeps = n;
        ingest.merge(gpu, encoder, &self.engine_buffers(), n);
        Ok(())
    }

    fn resident_status(&self) -> Vec<(&wgpu::Buffer, u64, u64)> {
        let mut parts = vec![(self.sim.counters_buffer(), 0, std::mem::size_of::<engine_schema::StepCounters>() as u64)];
        // The next frame's CCD bound reads the fast bodies' speeds: the bodies as this
        // frame leaves them, in this read (`bodies_now` then has them).
        if self.needs_ccd_bodies() {
            parts.push((self.sim.body_buffer(), 0, self.sim.body_count() as u64 * std::mem::size_of::<BodyState>() as u64));
        }
        parts
    }

    fn part_records(&self, sizes: &[usize]) -> Option<Vec<Vec<Vec<[[f32; 4]; 8]>>>> {
        sizes
            .iter()
            .enumerate()
            .map(|(s, &n)| (0..n).map(|c| self.chunk_parts.get(&(s, c)).map(|list| list.iter().map(Self::part_words).collect())).collect())
            .collect()
    }

    fn body_words(&self, key: BodyKey) -> Option<([[f32; 4]; 7], Vec<[[f32; 4]; 4]>)> {
        let e = self.bodies.get(&key)?;
        let s = self.bodies_now()[e.id as usize];
        let body: [[f32; 4]; 7] = bytemuck::cast(s);
        let cols = (e.colliders.start..e.colliders.end).map(|i| bytemuck::cast::<_, [[f32; 4]; 4]>(self.cols[i as usize])).collect();
        Some((body, cols))
    }

    fn body_slot(&self, key: BodyKey) -> Option<u32> {
        self.bodies.get(&key).map(|e| e.id)
    }

    fn spawn_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        Some((self.sim.body_buffer(), self.sim.collider_buffer(), self.sim.capacity().max_bodies))
    }

    fn frame_record(&self, key: BodyKey) -> Option<(u32, [f32; 3], [f32; 4])> {
        self.bodies.get(&key).map(|e| (e.id, [e.com.x as f32, e.com.y as f32, e.com.z as f32], [e.principal.x as f32, e.principal.y as f32, e.principal.z as f32, e.principal.w as f32]))
    }

    fn retire_record(&self, key: BodyKey) -> Option<[u32; 3]> {
        self.bodies.get(&key).map(|e| [e.id, e.colliders.start, e.colliders.end])
    }

    fn adopt_spawned(&mut self, adopted: Vec<stress_gpu::resident::AdoptedBody>, retired: &[BodyKey]) -> Result<(), String> {
        use engine_schema::convex::CollisionFilter;
        // The parents, retired on the GPU (`remove_body`'s bookkeeping, no upload).
        let mut retired_ids: HashMap<BodyKey, u32> = HashMap::new();
        for key in retired {
            let Some(e) = self.bodies.remove(key) else { continue };
            retired_ids.insert(*key, e.id);
            self.sim.mirror_retire(e.id);
            let never = CollisionFilter { categories: 0x8000, mask: 0, group: 0 };
            for id in e.colliders.clone() {
                never.apply(&mut self.cols[id as usize]);
                self.owners[id as usize] = Owner::Retired;
            }
            self.sim.mirror_colliders(e.colliders.start, &self.cols[e.colliders.start as usize..e.colliders.end as usize]);
            self.colliders_changed(e.colliders.clone());
            self.frozen.retain(|k| k != key);
        }
        // The children, made at the next slots in order (`create_body`'s bookkeeping).
        let bodies: Vec<engine_schema::BodyState> = adopted.iter().map(|a| bytemuck::cast(a.body)).collect();
        let cols: Vec<engine_schema::Collider> = adopted.iter().flat_map(|a| a.colliders.iter().map(|c| bytemuck::cast::<_, engine_schema::Collider>(*c))).collect();
        if let Some(first) = adopted.first() {
            if first.slot != self.sim.body_count() || first.first_collider != self.sim.collider_count() {
                return Err(format!("GPU bodies at {}/{} but the engine's next slots are {}/{}", first.slot, first.first_collider, self.sim.body_count(), self.sim.collider_count()));
            }
        }
        let (body_range, col_range) = self.sim.adopt_appended(&bodies, &cols);
        let mut col = col_range.start;
        for (a, state) in adopted.iter().zip(&bodies) {
            let range = col..col + a.colliders.len() as u32;
            col = range.end;
            self.cols.extend(a.colliders.iter().map(|c| bytemuck::cast::<_, engine_schema::Collider>(*c)));
            self.owners.extend(a.chunks.iter().map(|&chunk| Owner::Body { key: a.key, chunk: Some(chunk) }));
            self.bodies.insert(
                a.key,
                crate::Entry {
                    id: a.slot,
                    signature: a.signature,
                    fixed: a.fixed,
                    colliders: range,
                    com: Vec3::new(a.com[0], a.com[1], a.com[2]),
                    principal: stress_ref::math::Quat { w: a.principal[3], x: a.principal[0], y: a.principal[1], z: a.principal[2] },
                    inv_mass: state.inv_mass,
                    inv_inertia: state.inv_inertia,
                    ccd: false,
                    // Only CCD bodies' substep bound reads it.
                    thin: 0.0,
                },
            );
        }
        debug_assert_eq!(body_range.end, self.sim.body_count());
        debug_assert_eq!(self.owners.len(), self.sim.collider_count() as usize);
        // The bodies' snapshot (if any) follows the GPU's writes: the retired parents
        // stilled, the children appended (no download for the next frame's CCD bound).
        {
            let mut read = self.bodies_read.borrow_mut();
            if let Some(all) = read.as_mut() {
                for key in retired {
                    if let Some(id) = retired_ids.get(key) {
                        let b = &mut all[*id as usize];
                        b.inv_mass = 0.0;
                        b.inv_inertia = [0.0; 3];
                        b.linear_vel = [0.0; 3];
                        b.angular_vel = [0.0; 3];
                        b.flags = 0;
                    }
                }
                if all.len() == body_range.start as usize {
                    all.extend_from_slice(&bodies);
                } else {
                    *read = None;
                }
            }
        }
        self.colliders_changed(col_range);
        Ok(())
    }

    fn collider_capacity(&self) -> u32 {
        self.sim.capacity().max_colliders
    }

    fn body_counts(&self) -> (u32, u32) {
        (self.sim.body_count(), self.sim.collider_count())
    }

    fn resident_finish(&mut self, status: &[Vec<u8>]) -> Result<(), String> {
        let counters: engine_schema::StepCounters = bytemuck::pod_read_unaligned(&status[0][..std::mem::size_of::<engine_schema::StepCounters>()]);
        if std::env::var("STRESS_RESIDENT_PHASES").is_ok() {
            eprintln!("resident phases: rubble counters: {} bodies, {} colliders, {} contacts, {} candidates, {} awake, {} colours", counters.body_count, counters.collider_count, counters.contact_count, counters.candidate_count, counters.awake_count, counters.color_count);
        }
        if std::env::var("STRESS_GPU_SPLIT_HASH").is_ok() {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytemuck::cast_slice::<_, u8>(&self.sim.download_bodies()).hash(&mut h);
            eprintln!("frame hash: rubble bodies {:x}", h.finish());
        }
        if let Some(bytes) = status.get(1) {
            let n = self.sim.body_count() as usize;
            let bodies: Vec<BodyState> = (0..n).map(|i| bytemuck::pod_read_unaligned(&bytes[i * std::mem::size_of::<BodyState>()..(i + 1) * std::mem::size_of::<BodyState>()])).collect();
            *self.bodies_read.borrow_mut() = Some(bodies);
        }
        self.sim.drained(counters).map(|_| ()).map_err(|e| format!("rubble step failed: {e:?} (capacity overflow)"))
    }
}

impl RubbleEngine {
    /// One part as `spawn.slang` reads it (8 float4).
    fn part_words(p: &crate::Part) -> [[f32; 4]; 8] {
        let f = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32];
        let row = |r: usize| [p.inertia.m[r][0] as f32, p.inertia.m[r][1] as f32, p.inertia.m[r][2] as f32, 0.0];
        let (c, o) = (f(p.center), f(p.origin));
        let q = p.rotation;
        let col = &p.collider;
        [
            [c[0], c[1], c[2], p.mass as f32],
            row(0),
            row(1),
            row(2),
            [o[0], o[1], o[2], col.density],
            [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
            [f32::from_bits(col.shape), f32::from_bits(col.material), f32::from_bits(col.generation), 0.0],
            col.params,
        ]
    }

    /// Whether the frame's substep count reads CCD bodies' speeds (`resident_substeps`).
    fn needs_ccd_bodies(&self) -> bool {
        !self.speculative_ccd && self.bodies.iter().any(|(k, e)| e.ccd && e.inv_mass > 0.0 && !self.frozen.contains(k))
    }
}

impl RubbleEngine {
    /// `step_resident` with each part its own submission (the default).
    fn step_resident_submitting(&mut self, gpu: &Gpu, dt: f64, n: u32, ingest: &EngineIngest) -> Result<(), String> {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        ingest.begin_frame(&mut encoder, &self.engine_buffers());
        engine_gpu_io::submit();
        gpu.queue.submit([encoder.finish()]);
        if n > 0 {
            let h = dt / n as f64;
            let mut params = self.sim.params();
            params.dt = h as f32;
            params.contact_offset = self.resident_offset;
            self.sim.set_params(params);
            let phases = std::env::var("STRESS_RESIDENT_PHASES").is_ok();
            for s in 0..n {
                let t = std::time::Instant::now();
                self.sim.step_submit().map_err(|e| format!("rubble step failed: {e:?}"))?;
                if phases {
                    engine_gpu_io::wait();
                    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                    eprintln!("resident phases: rubble substep {s}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
                }
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                ingest.substep(gpu, &mut encoder, &self.engine_buffers(), self.sim.last_contact_offset(), s, h);
                let t_ingest = std::time::Instant::now();
                engine_gpu_io::submit();
                gpu.queue.submit([encoder.finish()]);
                if std::env::var("RUBBLE_HOST_TIMING").is_ok() {
                    engine_gpu_io::wait();
                    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                    eprintln!("ingest substep {s}: {:.2} ms", t_ingest.elapsed().as_secs_f64() * 1e3);
                }
            }
            self.bodies_changed();
        }
        self.last_substeps = n;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        ingest.merge(gpu, &mut encoder, &self.engine_buffers(), n);
        engine_gpu_io::submit();
        gpu.queue.submit([encoder.finish()]);
        Ok(())
    }
}
