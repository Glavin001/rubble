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
    /// `sort` word of the touched chunks' list (coupling.slang `list_base`).
    list_base: u32,
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
    Binding::StorageRw, // scan blocks (per 1024 chunks)
];

const BODY_BYTES: u64 = 112;
const LISTED: [&str; 7] = ["ingest_clear_listed", "ingest_clear_manifold_counts", "ingest_list_reset", "ingest_count_listed", "ingest_list_args", "ingest_scan_listed", "ingest_merge_listed"];
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
    scan_top: Kernel,
    scan_add: Kernel,
    scatter: Kernel,
    merge: Kernel,
    /// The listed merge's (coupling.slang, "The listed merge").
    listed_kernels: [Kernel; 7],
    /// Whether the merge runs over the touched chunks only ([`EngineIngest::set_listed`]).
    listed: bool,
    /// Whether the listed state is current: every count off the list zero and the last
    /// frame's list in `sort` (false after a full merge or a new `sort`).
    listed_ready: std::cell::Cell<bool>,
    /// The touched list's indirect size (copied from `counts`; [`EngineIngest::list_args`]).
    list_args: wgpu::Buffer,
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
    /// Per block of 1024 chunks, the scan's block totals and starts (`ingest_scan`).
    scan_blocks: wgpu::Buffer,
}

impl EngineIngest {
    /// An ingest for up to `max_substeps` engine substeps per frame of up to `max_rows`
    /// contact rows each, into `chunk_count` GPU chunks, for an engine of `max_bodies`
    /// bodies; `colliders` is the collider table (indexed by engine collider id).
    pub fn new(gpu: &Gpu, max_rows: u32, max_substeps: u32, max_bodies: u32, chunk_count: u32, colliders: &[ColliderInfo]) -> EngineIngest {
        let k = |e: &str| gpu.compute(&shaders::COUPLING, e, &BINDINGS);
        let records = (2 * max_rows as u64 * max_substeps as u64).max(1);
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
            scan_top: k("ingest_scan_blocks"),
            scan_add: k("ingest_scan_add"),
            scatter: k("ingest_scatter"),
            merge: k("ingest_merge"),
            listed_kernels: LISTED.map(k),
            listed: false,
            listed_ready: std::cell::Cell::new(false),
            list_args: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ingest list args"),
                size: 16,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            max_rows,
            max_substeps,
            max_bodies,
            chunk_count,
            start_bodies: storage("ingest start bodies", BODY_BYTES * max_bodies as u64),
            collider_info: gpu.storage("ingest collider info", table(colliders)),
            records: storage("ingest records", std::mem::size_of::<Record>() as u64 * records),
            sort: storage("ingest sort", 4 * (2 * n + 1 + records + 1 + n)),
            manifolds: storage("ingest manifolds", std::mem::size_of::<Record>() as u64 * records),
            counts: storage("ingest counts", 4 * (2 * n + 5)),
            row_counts: storage("ingest row counts", 4 * 1024),
            scan_blocks: storage("ingest scan blocks", 4 * n.div_ceil(1024)),
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
        let records = (2 * self.max_rows as u64 * max_substeps as u64).max(1);
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
        self.sort = storage("ingest sort", 4 * (2 * n + 1 + records + 1 + n));
        self.listed_ready.set(false);
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
        gpu.bind(&kernel.group, &[params, e.contacts, e.colliders, e.bodies, &self.start_bodies, e.counters, &self.collider_info, &self.records, &self.sort, &self.manifolds, &self.counts, &self.row_counts, &self.scan_blocks])
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
            list_base: self.list_base(),
        };
        gpu.uniform("ingest params", &p)
    }

    /// One engine substep's rows (the step just encoded or submitted before this):
    /// its records, at slot `substep`.
    pub fn substep(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, row_offset: u32, substep: u32, h: f64) {
        assert!(substep < self.max_substeps, "ingest: {substep} substeps, room for {}", self.max_substeps);
        let params = self.params(gpu, row_offset, substep, h);
        let bind = self.bind(gpu, &self.rows, &params, e);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ingest rows"), timestamp_writes: None });
        pass.set_pipeline(&self.rows.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups(self.max_rows as usize, 64), 1, 1);
    }

    /// The frame's manifolds, from the records of `substeps` substeps (the rest of the
    /// slots cleared).
    pub fn merge(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, substeps: u32) {
        if self.listed {
            let (params, records) = self.merge_params(gpu, substeps);
            self.merge_listed(gpu, encoder, e, &params, records);
        } else {
            self.listed_ready.set(false);
            self.merge_full(gpu, encoder, e, substeps);
        }
    }

    fn merge_params(&self, gpu: &Gpu, substeps: u32) -> (wgpu::Buffer, usize) {
        // Only the substeps run are merged, and in each only its rows (`record_live`): the
        // rest of the records is never read, so nothing is cleared.
        let mut params = IngestParams {
            row_offset: 0,
            max_rows: self.max_rows,
            substep: 0,
            chunk_count: self.chunk_count,
            record_count: 2 * self.max_rows * substeps.max(1),
            max_bodies: self.max_bodies,
            h: 0.0,
            list_base: self.list_base(),
        };
        if substeps == 0 {
            params.record_count = 0;
        }
        (gpu.uniform("ingest params", &params), (2 * self.max_rows * substeps.max(1)) as usize)
    }

    /// The merge over every chunk, whatever [`EngineIngest::set_listed`] says (a listed
    /// ingest's next merge then starts over: [`ListedCheck`](crate::fracture::ListedCheck)
    /// restores its state after this).
    pub fn merge_full(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, substeps: u32) {
        let (params, records) = self.merge_params(gpu, substeps);
        let n = self.chunk_count as usize;
        for (k, size, label) in [(&self.clear, groups(2 * n, 64), "ingest clear"), (&self.count, groups(records, 64), "ingest count"), (&self.scan, groups(n, 1024), "ingest scan"), (&self.scan_top, 1, "ingest scan blocks"), (&self.scan_add, groups(n, 64), "ingest scan add"), (&self.scatter, groups(records, 64), "ingest scatter"), (&self.merge, groups(n, 64), "ingest merge")] {
            let bind = self.bind(gpu, k, &params, e);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size, 1, 1);
        }
    }

    /// The merge over the touched chunks only (coupling.slang, "The listed merge"): the
    /// same manifolds, counts and segments per chunk as the full merge.
    fn merge_listed(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, e: &EngineBuffers, params: &wgpu::Buffer, records: usize) {
        let n = self.chunk_count as usize;
        let run = |encoder: &mut wgpu::CommandEncoder, k: &Kernel, size: Option<u32>, label: &str| {
            let bind = self.bind(gpu, k, params, e);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: None });
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            match size {
                Some(g) => pass.dispatch_workgroups(g, 1, 1),
                None => pass.dispatch_workgroups_indirect(&self.list_args, 0),
            }
        };
        let [clear, clear_counts, reset, count, args, scan, merge] = &self.listed_kernels;
        if self.listed_ready.get() {
            // The last frame's touched chunks (its list and size are still in place).
            run(encoder, clear, None, "ingest clear listed");
        } else {
            run(encoder, &self.clear, Some(groups(2 * n, 64)), "ingest clear");
            run(encoder, clear_counts, Some(groups(n, 64)), "ingest clear manifold counts");
        }
        run(encoder, reset, Some(1), "ingest list reset");
        run(encoder, count, Some(groups(records, 64)), "ingest count listed");
        run(encoder, args, Some(1), "ingest list args");
        encoder.copy_buffer_to_buffer(&self.counts, 4 * (2 * n as u64 + 1), &self.list_args, 0, 12);
        run(encoder, scan, None, "ingest scan listed");
        run(encoder, &self.scatter, Some(groups(records, 64)), "ingest scatter");
        run(encoder, merge, None, "ingest merge listed");
        self.listed_ready.set(true);
    }

    /// The merge over only the chunks with records this frame (`listed`), or over every
    /// chunk (the default). Both give every chunk the same manifolds; the listed one also
    /// keeps the frame's touched chunks in `sort` ([`EngineIngest::list_base`]) for the
    /// fracture's listed passes.
    pub fn set_listed(&mut self, listed: bool) {
        if listed != self.listed {
            self.listed = listed;
            self.listed_ready.set(false);
        }
    }

    pub fn listed(&self) -> bool {
        self.listed
    }

    /// `sort` word of the touched chunks' list: its count, then the chunks.
    pub fn list_base(&self) -> u32 {
        2 * self.chunk_count + 1 + (2 * self.max_rows * self.max_substeps).max(1)
    }

    /// The touched list's indirect size (64 chunks a group), written by the last listed
    /// merge; with [`EngineIngest::list_base`].
    pub fn list_args(&self) -> &wgpu::Buffer {
        &self.list_args
    }

    /// The buffers a listed merge changes besides its outputs (the counts and the list):
    /// with `sort` and `manifolds`, a frame's ingest state (a diagnostic's backup).
    pub fn counts(&self) -> &wgpu::Buffer {
        &self.counts
    }

    /// Diagnostic read-back: each chunk's manifolds, in their order.
    pub fn read_manifolds(&self, gpu: &Gpu) -> Vec<Vec<Record>> {
        let n = self.chunk_count as usize;
        let sort: Vec<u32> = gpu.read_range(&self.sort, 0, 2 * n + 1);
        let man: Vec<Record> = gpu.read(&self.manifolds);
        (0..n).map(|k| man[sort[k] as usize..sort[k] as usize + sort[n + 1 + k] as usize].to_vec()).collect()
    }
}
