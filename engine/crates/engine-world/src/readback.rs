//! A ring of staging buffers: each step copies its counters, the event ring and
//! the moved-body list into one slot, maps it asynchronously and is read when
//! the map completes, so stepping never waits on a readback.

use engine_schema::StepCounters;
use engine_webgpu::{PoseRecord, WgpuSim};
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// What `World::step` does when every readback slot is still in flight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backpressure {
    /// Wait for the oldest slot (the host fell more than `readback_frames` behind).
    #[default]
    Block,
    /// Skip this step's readback and mark its frame `dropped`.
    Drop,
}

const COUNTERS_OFFSET: u64 = 0;
const COUNTERS_BYTES: u64 = std::mem::size_of::<StepCounters>() as u64;
const RING_OFFSET: u64 = COUNTERS_BYTES.div_ceil(256) * 256;
const ALIGN: u64 = 256;

/// Byte layout of one slot.
#[derive(Clone, Copy, Debug)]
struct Layout {
    ring_bytes: u64,
    header: u64,
    records: u64,
}

impl Layout {
    fn new(ring_bytes: u64) -> Self {
        let header = (RING_OFFSET + ring_bytes).div_ceil(ALIGN) * ALIGN;
        Self {
            ring_bytes,
            header,
            records: header + 16,
        }
    }

    fn bytes(&self, bodies: u32) -> u64 {
        self.records + u64::from(bodies) * engine_webgpu::POSE_RECORD_BYTES
    }
}

struct Slot {
    buffer: wgpu::Buffer,
    capacity: u64,
    /// Bytes mapped by the last `map`.
    mapped: u64,
    map: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// One step's raw results.
pub(crate) struct Readback {
    pub counters: StepCounters,
    pub ring: Vec<u32>,
    pub moved: Vec<PoseRecord>,
}

pub(crate) enum Pending {
    /// Copied into this slot; read when its map completes.
    Mapped { step: u64, slot: usize },
    /// The step was skipped (nothing awake) or its readback dropped.
    Empty { step: u64, dropped: bool },
}

pub(crate) struct ReadbackRing {
    slots: Vec<Slot>,
    layout: Layout,
    free: Vec<usize>,
    pub pending: std::collections::VecDeque<Pending>,
}

impl ReadbackRing {
    pub fn new(sim: &WgpuSim, frames: u32) -> Self {
        let (_, _, ring_bytes) = sim.event_ring_range();
        let layout = Layout::new(ring_bytes);
        let n = frames.max(1) as usize;
        let slots = (0..n)
            .map(|i| Slot {
                buffer: staging(sim.device(), i, layout.bytes(0)),
                capacity: layout.bytes(0),
                mapped: 0,
                map: None,
            })
            .collect();
        Self {
            slots,
            layout,
            free: (0..n).rev().collect(),
            pending: Default::default(),
        }
    }

    pub fn has_free(&self) -> bool {
        !self.free.is_empty()
    }

    /// Encode this step's copies into a free slot (the caller made sure one is free).
    pub fn encode(&mut self, sim: &WgpuSim, enc: &mut wgpu::CommandEncoder, step: u64) -> usize {
        let slot = self.free.pop().expect("a free readback slot");
        let bodies = sim.body_count();
        let need = self.layout.bytes(bodies);
        let s = &mut self.slots[slot];
        if s.capacity < need {
            // Grow geometrically so a growing scene reallocates rarely.
            s.capacity = need.max(s.capacity * 2);
            s.buffer = staging(sim.device(), slot, s.capacity);
        }
        let l = self.layout;
        enc.copy_buffer_to_buffer(
            sim.counters_buffer(),
            0,
            &s.buffer,
            COUNTERS_OFFSET,
            COUNTERS_BYTES,
        );
        let (ring, ring_offset, ring_bytes) = sim.event_ring_range();
        enc.copy_buffer_to_buffer(ring, ring_offset, &s.buffer, RING_OFFSET, ring_bytes);
        enc.copy_buffer_to_buffer(sim.moved_header_buffer(), 0, &s.buffer, l.header, 16);
        if bodies > 0 {
            enc.copy_buffer_to_buffer(
                sim.moved_buffer(),
                0,
                &s.buffer,
                l.records,
                u64::from(bodies) * engine_webgpu::POSE_RECORD_BYTES,
            );
        }
        self.pending.push_back(Pending::Mapped { step, slot });
        slot
    }

    /// Start mapping `slot` (after the copies were submitted).
    pub fn map(&mut self, slot: usize, bodies: u32) {
        let (tx, rx) = channel();
        let bytes = self.layout.bytes(bodies);
        self.slots[slot]
            .buffer
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.slots[slot].map = Some(rx);
        self.slots[slot].mapped = bytes;
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
    pub fn take(&mut self, slot: usize) -> Readback {
        let l = self.layout;
        let s = &self.slots[slot];
        let out = {
            let data = s
                .buffer
                .slice(..s.mapped)
                .get_mapped_range()
                .expect("mapped range");
            let words: &[u32] = bytemuck::cast_slice(&data);
            let w = |byte: u64| (byte / 4) as usize;
            let counters: StepCounters = bytemuck::pod_read_unaligned(
                &data[COUNTERS_OFFSET as usize..][..COUNTERS_BYTES as usize],
            );
            let ring = words[w(RING_OFFSET)..w(RING_OFFSET + l.ring_bytes)].to_vec();
            let count = words[w(l.header)] as usize;
            let records = &data[l.records as usize..];
            let available = records.len() / std::mem::size_of::<PoseRecord>();
            let moved = (0..count.min(available))
                .map(|k| bytemuck::pod_read_unaligned(&records[k * 32..k * 32 + 32]))
                .collect();
            Readback {
                counters,
                ring,
                moved,
            }
        };
        s.buffer.unmap();
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
