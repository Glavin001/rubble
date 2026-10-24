//! A ring of staging buffers: each step copies its counters, the event ring,
//! the moved-body list and any extra outputs the host asked for (vehicle
//! state, soft-body nodes, skinned meshes, a full pose resync) into one slot,
//! maps it asynchronously and is read when the map completes, so stepping
//! never waits on a readback.

use engine_schema::StepCounters;
use engine_webgpu::{PoseRecord, WgpuSim};
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// What `World::step` does when every readback slot is still in flight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backpressure {
    /// Wait for the oldest slot (the host fell more than `readback_frames`
    /// behind).
    #[default]
    Block,
    /// Skip this step's readback and mark its frame `dropped`. The next frame
    /// delivered carries every body that changed in between (no resync call
    /// needed).
    Drop,
}

const COUNTERS_OFFSET: u64 = 0;
const COUNTERS_BYTES: u64 = std::mem::size_of::<StepCounters>() as u64;
const RING_OFFSET: u64 = COUNTERS_BYTES.div_ceil(256) * 256;
const ALIGN: u64 = 256;

/// One extra copy into a slot, tagged so the frame decoder knows what it is.
pub(crate) struct Extra<'a, K> {
    pub kind: K,
    pub buffer: &'a wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}

struct Slot<K> {
    buffer: wgpu::Buffer,
    capacity: u64,
    /// Bytes mapped by the last `map`.
    mapped: u64,
    /// Where the moved records start and how many fit; extras that follow.
    records: u64,
    record_capacity: u32,
    extras: Vec<(K, u64, u64)>,
    map: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// One step's raw results.
pub(crate) struct Readback<K> {
    pub counters: StepCounters,
    pub ring: Vec<u32>,
    pub moved: Vec<PoseRecord>,
    pub extras: Vec<(K, Vec<u8>)>,
}

pub(crate) enum Pending {
    /// Copied into this slot; read when its map completes.
    Mapped { step: u64, slot: usize },
    /// The step was skipped (nothing awake) or its readback dropped.
    Empty { step: u64, dropped: bool },
}

pub(crate) struct ReadbackRing<K> {
    slots: Vec<Slot<K>>,
    ring_bytes: u64,
    header: u64,
    free: Vec<usize>,
    pub pending: std::collections::VecDeque<Pending>,
}

impl<K: Clone> ReadbackRing<K> {
    pub fn new(sim: &WgpuSim, frames: u32) -> Self {
        let (_, _, ring_bytes) = sim.event_ring_range();
        let header = (RING_OFFSET + ring_bytes).div_ceil(ALIGN) * ALIGN;
        let n = frames.max(1) as usize;
        let initial = header + 16;
        let slots = (0..n)
            .map(|i| Slot {
                buffer: staging(sim.device(), i, initial),
                capacity: initial,
                mapped: 0,
                records: header + 16,
                record_capacity: 0,
                extras: Vec::new(),
                map: None,
            })
            .collect();
        Self {
            slots,
            ring_bytes,
            header,
            free: (0..n).rev().collect(),
            pending: Default::default(),
        }
    }

    pub fn has_free(&self) -> bool {
        !self.free.is_empty()
    }

    /// Encode this step's copies into a free slot (the caller made sure one is
    /// free). Returns the slot.
    pub fn encode(
        &mut self,
        sim: &WgpuSim,
        enc: &mut wgpu::CommandEncoder,
        step: u64,
        extras: &[Extra<'_, K>],
    ) -> usize {
        let slot = self.free.pop().expect("a free readback slot");
        let bodies = sim.body_count();
        let records = self.header + 16;
        let mut end = records + u64::from(bodies) * engine_webgpu::POSE_RECORD_BYTES;
        let mut placed = Vec::with_capacity(extras.len());
        for e in extras {
            end = end.div_ceil(ALIGN) * ALIGN;
            placed.push((e.kind.clone(), end, e.size));
            end += e.size;
        }
        let s = &mut self.slots[slot];
        if s.capacity < end {
            // Grow geometrically so a growing scene reallocates rarely.
            s.capacity = end.max(s.capacity * 2);
            s.buffer = staging(sim.device(), slot, s.capacity);
        }
        s.records = records;
        s.record_capacity = bodies;
        s.mapped = end;
        enc.copy_buffer_to_buffer(
            sim.counters_buffer(),
            0,
            &s.buffer,
            COUNTERS_OFFSET,
            COUNTERS_BYTES,
        );
        let (ring, ring_offset, ring_bytes) = sim.event_ring_range();
        enc.copy_buffer_to_buffer(ring, ring_offset, &s.buffer, RING_OFFSET, ring_bytes);
        enc.copy_buffer_to_buffer(sim.moved_header_buffer(), 0, &s.buffer, self.header, 16);
        if bodies > 0 {
            enc.copy_buffer_to_buffer(
                sim.moved_buffer(),
                0,
                &s.buffer,
                records,
                u64::from(bodies) * engine_webgpu::POSE_RECORD_BYTES,
            );
        }
        for (e, (_, at, size)) in extras.iter().zip(&placed) {
            if *size > 0 {
                enc.copy_buffer_to_buffer(e.buffer, e.offset, &s.buffer, *at, *size);
            }
        }
        s.extras = placed;
        self.pending.push_back(Pending::Mapped { step, slot });
        slot
    }

    /// Start mapping `slot` (after its copies were submitted).
    pub fn map(&mut self, slot: usize) {
        let (tx, rx) = channel();
        let s = &mut self.slots[slot];
        s.buffer
            .slice(..s.mapped)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        s.map = Some(rx);
    }

    /// Whether `slot`'s map has completed (`wait`: block until it has).
    pub fn ready(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        wait: bool,
    ) -> anyhow::Result<bool> {
        let rx = self.slots[slot].map.as_ref().expect("slot in flight");
        if wait {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            engine_gpu_wait::recv(rx).map_err(|e| anyhow::anyhow!("readback map failed: {e:?}"))?;
            self.slots[slot].map = None;
            return Ok(true);
        }
        match rx.try_recv() {
            Ok(r) => {
                r.map_err(|e| anyhow::anyhow!("readback map failed: {e:?}"))?;
                self.slots[slot].map = None;
                Ok(true)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => anyhow::bail!("readback map callback dropped"),
        }
    }

    /// Read a mapped slot, unmap it and return it to the free list.
    pub fn take(&mut self, slot: usize) -> Readback<K> {
        let ring_bytes = self.ring_bytes;
        let header = self.header;
        let s = &mut self.slots[slot];
        let out = {
            let data = s
                .buffer
                .slice(..s.mapped)
                .get_mapped_range()
                .expect("mapped range");
            let words: &[u32] = bytemuck::cast_slice(&data[..header as usize + 16]);
            let w = |byte: u64| (byte / 4) as usize;
            let counters: StepCounters = bytemuck::pod_read_unaligned(
                &data[COUNTERS_OFFSET as usize..][..COUNTERS_BYTES as usize],
            );
            let ring = words[w(RING_OFFSET)..w(RING_OFFSET + ring_bytes)].to_vec();
            let count = (words[w(header)] as usize).min(s.record_capacity as usize);
            let records = &data[s.records as usize..];
            let moved = (0..count)
                .map(|k| bytemuck::pod_read_unaligned(&records[k * 32..k * 32 + 32]))
                .collect();
            let extras = s
                .extras
                .iter()
                .map(|(kind, at, size)| {
                    let bytes = data[*at as usize..(*at + *size) as usize].to_vec();
                    (kind.clone(), bytes)
                })
                .collect();
            Readback {
                counters,
                ring,
                moved,
                extras,
            }
        };
        s.buffer.unmap();
        s.extras.clear();
        self.free.push(slot);
        out
    }
}

fn staging(device: &wgpu::Device, i: usize, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("world readback {i}")),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
