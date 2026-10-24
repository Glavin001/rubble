//! The GPU-resident coupling's contact ingest (`shaders/coupling.slang`,
//! docs/stress-fracture-gpu-resident.md phase 2): an engine's solved contact rows, read
//! on the GPU where the engine left them, as contact manifolds per (GPU chunk, other
//! body), without a host round trip. The engine's buffers are bound as they are (their
//! layouts are engine-schema's `BodyState`, `Collider`, `Contact`, `StepCounters`); the
//! coupling supplies the collider table (`ColliderInfo`).

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::shaders;

/// What the coupling knows of an engine collider (coupling.slang `ColliderInfo`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColliderInfo {
    /// (GPU chunk or `NONE`, body code, kind, chunk side taken (0/1)).
    pub a: [u32; 4],
    /// (mass (inf: anchored or static), Young's modulus, radius, -) as the other side of
    /// a contact sees this collider's body.
    pub b: [f32; 4],
}

pub const NONE: u32 = u32::MAX;
pub const KIND_SKIP: u32 = 0;
pub const KIND_STATIC: u32 = 1;
pub const KIND_BODY: u32 = 2;

/// A chunk-side record or a manifold (coupling.slang `Record`): `ids` = (GPU chunk,
/// other body code, slot, valid); `point` (w: |impulse| summed); `impulse` (w: closing
/// speed, impulse-weighted); `other` (mass, modulus, radius).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Record {
    pub ids: [u32; 4],
    pub point: [f32; 4],
    pub impulse: [f32; 4],
    pub other: [f32; 4],
    /// The contact's identity (coupling.slang `Record::key`): substep, this side's
    /// collider, the other's, feature id * 2 + side.
    pub key: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct IngestParams {
    row_offset: u32,
    max_rows: u32,
    substep: u32,
    chunk_count: u32,
    record_count: u32,
    max_bodies: u32,
    h: f32,
    /// Wheel slots after the contact records (`ingest_wheels`), 0: none; their first slot.
    wheels: u32,
    wheel_base: u32,
    pad: [u32; 3],
}

/// The engine's buffers the ingest reads.
pub struct EngineBuffers<'a> {
    pub bodies: &'a wgpu::Buffer,
    pub colliders: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub counters: &'a wgpu::Buffer,
}

const BINDINGS: [Binding; 13] = [
    Binding::Uniform,   // params
    Binding::Storage,   // contacts
    Binding::Storage,   // colliders
    Binding::Storage,   // bodies (substep end)
    Binding::Storage,   // start bodies
    Binding::Storage,   // counters
    Binding::Storage,   // collider info
    Binding::StorageRw, // records
    Binding::StorageRw, // sort
    Binding::StorageRw, // manifolds
    Binding::StorageRw, // counts
    Binding::StorageRw, // row counts (per substep)
    Binding::Storage,   // vehicle pool (ingest_wheels)
];

const BODY_BYTES: u64 = 112;
const EMPTY_TABLE: [ColliderInfo; 1] = [ColliderInfo { a: [NONE, NONE, KIND_SKIP, 0], b: [0.0; 4] }];

fn table(colliders: &[ColliderInfo]) -> &[ColliderInfo] {
    if colliders.is_empty() {
        &EMPTY_TABLE
    } else {
        colliders
    }
}

pub struct EngineIngest {
    rows: Kernel,
    clear: Kernel,
    count: Kernel,
    scan: Kernel,
    scatter: Kernel,
    merge: Kernel,
    wheel_kernel: Kernel,
    max_rows: u32,
    max_substeps: u32,
    max_bodies: u32,
    chunk_count: u32,
    pub start_bodies: wgpu::Buffer,
    pub collider_info: wgpu::Buffer,
    pub records: wgpu::Buffer,
    pub sort: wgpu::Buffer,
    pub manifolds: wgpu::Buffer,
    counts: wgpu::Buffer,
    /// Per substep, the engine's contact rows (`ingest_rows`; `record_live`).
    row_counts: wgpu::Buffer,
    /// Raycast wheel slots (`with_wheels`): their records follow the contacts'.
    max_wheels: u32,
    /// Bound as the vehicle pool by the kernels that do not read one.
    no_pool: wgpu::Buffer,
    /// Whether this frame's wheels were recorded (`wheels`), for its `merge`.
    wheels_recorded: std::cell::Cell<bool>,
}

impl EngineIngest {
    /// An ingest for up to `max_substeps` engine substeps per frame of up to `max_rows`
    /// contact rows each, into `chunk_count` GPU chunks, for an engine of `max_bodies`
    /// bodies; `colliders` is the collider table (indexed by engine collider id).
    pub fn new(gpu: &Gpu, max_rows: u32, max_substeps: u32, max_bodies: u32, chunk_count: u32, colliders: &[ColliderInfo]) -> EngineIngest {
        EngineIngest::with_wheels(gpu, max_rows, max_substeps, max_bodies, chunk_count, colliders, 0)
    }

    /// `new`, with room for the loads of `max_wheels` raycast wheel slots (the engine's
    /// vehicle capacity; [`EngineIngest::wheels`]).
    pub fn with_wheels(gpu: &Gpu, max_rows: u32, max_substeps: u32, max_bodies: u32, chunk_count: u32, colliders: &[ColliderInfo], max_wheels: u32) -> EngineIngest {
        let k = |e: &str| gpu.compute(&shaders::COUPLING, e, &BINDINGS);
        let records = (2 * max_rows as u64 * max_substeps as u64 + max_wheels as u64).max(1);
        let storage = |label: &str, bytes: u64| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let n = chunk_count as u64;
        EngineIngest {
            rows: k("ingest_rows"),
            clear: k("ingest_clear"),
            count: k("ingest_count"),
            scan: k("ingest_scan"),
            scatter: k("ingest_scatter"),
            merge: k("ingest_merge"),
            wheel_kernel: k("ingest_wheels"),
            max_rows,
            max_substeps,
            max_bodies,
            chunk_count,
            start_bodies: storage("ingest start bodies", BODY_BYTES * max_bodies as u64),
            collider_info: gpu.storage("ingest collider info", table(colliders)),
            records: storage("ingest records", std::mem::size_of::<Record>() as u64 * records),
            sort: storage("ingest sort", 4 * (2 * n + 1 + records)),
            manifolds: storage("ingest manifolds", std::mem::size_of::<Record>() as u64 * records),
            counts: storage("ingest counts", 4 * 2 * n),
            row_counts: storage("ingest row counts", 4 * 1024),
            max_wheels,
            no_pool: storage("ingest no vehicle pool", 16),
            wheels_recorded: std::cell::Cell::new(false),
        }
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunk_count
    }

    /// The rows and bodies it was made for (`new`).
    pub fn capacity(&self) -> (u32, u32) {
        (self.max_rows, self.max_bodies)
    }

    /// Room for `max_substeps` substeps per frame: only the per-substep buffers are made
    /// anew (records, their sort and manifolds); the collider table and the rest stay.
    pub fn grow_substeps(&mut self, gpu: &Gpu, max_substeps: u32) {
        if max_substeps <= self.max_substeps {
            return;
        }
        let records = (2 * self.max_rows as u64 * max_substeps as u64 + self.max_wheels as u64).max(1);
        let storage = |label: &str, bytes: u64| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let n = self.chunk_count as u64;
        self.records = storage("ingest records", std::mem::size_of::<Record>() as u64 * records);
        self.sort = storage("ingest sort", 4 * (2 * n + 1 + records));
        self.manifolds = storage("ingest manifolds", std::mem::size_of::<Record>() as u64 * records);
        self.max_substeps = max_substeps;
    }

    /// Replace the collider table (a layout change on either side).
    pub fn set_colliders(&mut self, gpu: &Gpu, colliders: &[ColliderInfo]) {
        self.collider_info = gpu.storage_into("ingest collider info", table(colliders), Some(&self.collider_info));
    }

    /// `set_colliders` into a table with room for `capacity` colliders (the GPU keeps the
    /// rest current: colliders.rs): the entries past the host's are left as they are.
    pub fn set_colliders_with_room(&mut self, gpu: &Gpu, colliders: &[ColliderInfo], capacity: u32) {
        let bytes = (std::mem::size_of::<ColliderInfo>() * colliders.len().max(capacity as usize).max(1)) as u64;
        if self.collider_info.size() < bytes {
            self.collider_info = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ingest collider info"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        gpu.queue.write_buffer(&self.collider_info, 0, bytemuck::cast_slice(table(colliders)));
    }

    fn bind(&self, gpu: &Gpu, kernel: &Kernel, params: &wgpu::Buffer, e: &EngineBuffers) -> wgpu::BindGroup {
        // Only `ingest_wheels` reads the vehicle pool (the layouts hold what each uses).
        self.bind_with(gpu, kernel, params, e, &self.no_pool)
    }

    fn bind_with(&self, gpu: &Gpu, kernel: &Kernel, params: &wgpu::Buffer, e: &EngineBuffers, vehicle_pool: &wgpu::Buffer) -> wgpu::BindGroup {
        gpu.bind(&kernel.group, &[params, e.contacts, e.colliders, e.bodies, &self.start_bodies, e.counters, &self.collider_info, &self.records, &self.sort, &self.manifolds, &self.counts, &self.row_counts, vehicle_pool])
    }

    /// The first wheel record slot: after every substep's contact records.
    fn wheel_base(&self) -> u32 {
        2 * self.max_rows * self.max_substeps
    }

    /// The frame's start: the engine's bodies as they are now (closing speeds use them).
    pub fn begin_frame(&self, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers) {
        encoder.copy_buffer_to_buffer(e.bodies, 0, &self.start_bodies, 0, BODY_BYTES * self.max_bodies as u64);
    }

    fn params(&self, gpu: &Gpu, row_offset: u32, substep: u32, h: f64) -> wgpu::Buffer {
        let p = IngestParams {
            row_offset,
            max_rows: self.max_rows,
            substep,
            chunk_count: self.chunk_count,
            record_count: 2 * self.max_rows * self.max_substeps,
            max_bodies: self.max_bodies,
            h: h as f32,
            wheels: 0,
            wheel_base: self.wheel_base(),
            pad: [0; 3],
        };
        gpu.uniform("ingest params", &p)
    }

    /// One engine substep's rows (the step just encoded or submitted before this):
    /// its records, at slot `substep`.
    pub fn substep(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, row_offset: u32, substep: u32, h: f64) {
        assert!(substep < self.max_substeps, "ingest: {substep} substeps, room for {}", self.max_substeps);
        let params = self.params(gpu, row_offset, substep, h);
        let bind = self.bind(gpu, &self.rows, &params, e);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.rows.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups(self.max_rows as usize, 64), 1, 1);
    }

    /// The loads of the engine's raycast wheels over its step of `h` s (W5,
    /// `ingest_wheels`): each wheel on a chunk, from `vehicle_pool` (the engine's vehicle
    /// pool, engine-schema `vehicle.rs`), for this frame's `merge`. Nothing without wheel
    /// slots (`with_wheels`).
    pub fn wheels(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, vehicle_pool: &wgpu::Buffer, h: f64) {
        if self.max_wheels == 0 {
            return;
        }
        let mut p = self.base_params();
        p.wheels = self.max_wheels;
        p.h = h as f32;
        let params = gpu.uniform("ingest params", &p);
        let bind = self.bind_with(gpu, &self.wheel_kernel, &params, e, vehicle_pool);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.wheel_kernel.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups(self.max_wheels as usize, 64), 1, 1);
        self.wheels_recorded.set(true);
    }

    fn base_params(&self) -> IngestParams {
        IngestParams {
            row_offset: 0,
            max_rows: self.max_rows,
            substep: 0,
            chunk_count: self.chunk_count,
            record_count: 0,
            max_bodies: self.max_bodies,
            h: 0.0,
            wheels: 0,
            wheel_base: self.wheel_base(),
            pad: [0; 3],
        }
    }

    /// The frame's manifolds, from the records of `substeps` substeps (the rest of the
    /// slots cleared).
    pub fn merge(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, substeps: u32) {
        // Only the substeps run are merged, and in each only its rows (`record_live`): the
        // rest of the records is never read, so nothing is cleared.
        let mut params = self.base_params();
        params.record_count = if substeps == 0 { 0 } else { 2 * self.max_rows * substeps };
        // This frame's wheel records, if `wheels` wrote them.
        if self.wheels_recorded.replace(false) {
            params.wheels = self.max_wheels;
        }
        let records = (params.record_count + params.wheels).max(2 * self.max_rows) as usize;
        let params = gpu.uniform("ingest params", &params);
        let n = self.chunk_count as usize;
        for (k, size) in [(&self.clear, groups(2 * n, 64)), (&self.count, groups(records, 64)), (&self.scan, 1), (&self.scatter, groups(records, 64)), (&self.merge, groups(n, 64))] {
            let bind = self.bind(gpu, k, &params, e);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size, 1, 1);
        }
    }

    /// Diagnostic read-back: each chunk's manifolds, in their order.
    pub fn read_manifolds(&self, gpu: &Gpu) -> Vec<Vec<Record>> {
        let n = self.chunk_count as usize;
        let sort: Vec<u32> = gpu.read_range(&self.sort, 0, 2 * n + 1);
        let man: Vec<Record> = gpu.read(&self.manifolds);
        (0..n).map(|k| man[sort[k] as usize..sort[k] as usize + sort[n + 1 + k] as usize].to_vec()).collect()
    }
}
