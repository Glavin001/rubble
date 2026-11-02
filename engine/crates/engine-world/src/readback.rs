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
    /// The moved records past `record_capacity` (a GPU copy, not mapped): read back only
    /// when the step moved more than the slot copied (`spill_read`).
    spill: Option<wgpu::Buffer>,
    spill_read: Option<Spill>,
    /// `ReadbackRing::check`: every moved record (the full copy), to compare.
    check: Option<(wgpu::Buffer, Option<MapRx>)>,
}

type MapRx = Receiver<Result<(), wgpu::BufferAsyncError>>;

/// The rest of an overflowing step's moved records, being read back.
struct Spill {
    buffer: Option<wgpu::Buffer>,
    records: u32,
    map: Option<MapRx>,
}

/// Moved records a slot copies when the last step moved `last`: the last count plus
/// headroom (the rest goes to the slot's spill, read only when needed).
fn record_room(last: u32, bodies: u32) -> u32 {
    last.saturating_add(last / 2)
        .saturating_add(1024)
        .min(bodies)
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
    /// The moved count of the last step read: the next slots' copy size.
    last_moved: u32,
    /// `RUBBLE_READBACK_POSES=full`: every slot copies a record per body (A/B runs).
    full_poses: bool,
    /// Diagnostic ([`ReadbackRing::set_check`]): each slot also copies every record, and
    /// `take` compares the sized copy's records against it: (frames checked, differing).
    /// Every third slot then copies half the last count (the spill's read exercised).
    check: bool,
    pub checked: (u64, u64),
    /// Frames whose moved records overflowed their slot's copy (read from the spill).
    pub spills: u64,
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
                spill: None,
                spill_read: None,
                check: None,
            })
            .collect();
        Self {
            slots,
            ring_bytes,
            header,
            free: (0..n).rev().collect(),
            pending: Default::default(),
            last_moved: 0,
            full_poses: std::env::var("RUBBLE_READBACK_POSES").is_ok_and(|v| v == "full"),
            check: false,
            checked: (0, 0),
            spills: 0,
        }
    }

    /// Diagnostic: every slot from the next on also copies the full moved range, and the
    /// frame's records are compared against it ([`ReadbackRing::checked`]).
    pub fn set_check(&mut self, on: bool) {
        self.check = on;
    }

    /// Slots in the ring: how many steps' readbacks can be in flight.
    pub fn frames(&self) -> usize {
        self.slots.len()
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
        late: &[(K, u64)],
    ) -> usize {
        let slot = self.free.pop().expect("a free readback slot");
        let bodies = sim.body_count();
        let records = self.header + 16;
        // The moved records are compacted to the front (`pose_pack`), their count in the
        // header: the slot copies as many as the last step moved plus headroom, and the
        // rest to its spill, read back only if this step moved more (`ready`).
        let room = if self.full_poses {
            bodies
        } else if self.check && step % 3 == 0 {
            // Checking: every third slot copies half the last count, so the spill's read
            // runs whenever bodies move (and is compared against the full copy).
            (self.last_moved / 2).min(bodies)
        } else {
            record_room(self.last_moved, bodies)
        };
        let mut end = records + u64::from(room) * engine_webgpu::POSE_RECORD_BYTES;
        let mut placed = Vec::with_capacity(extras.len());
        for e in extras {
            end = end.div_ceil(ALIGN) * ALIGN;
            placed.push((e.kind.clone(), end, e.size));
            end += e.size;
        }
        // Copied by the host in a later submission (`late_target`).
        for (kind, size) in late {
            end = end.div_ceil(ALIGN) * ALIGN;
            placed.push((kind.clone(), end, *size));
            end += size;
        }
        let s = &mut self.slots[slot];
        if s.capacity < end {
            // Grow geometrically so a growing scene reallocates rarely.
            s.capacity = end.max(s.capacity * 2);
            s.buffer = staging(sim.device(), slot, s.capacity);
        }
        s.records = records;
        s.record_capacity = room;
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
        let record = engine_webgpu::POSE_RECORD_BYTES;
        if room > 0 {
            enc.copy_buffer_to_buffer(
                sim.moved_buffer(),
                0,
                &s.buffer,
                records,
                u64::from(room) * record,
            );
        }
        if room < bodies {
            let rest = u64::from(bodies - room) * record;
            if s.spill.as_ref().is_none_or(|b| b.size() < rest) {
                s.spill = Some(sim.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("world readback spill"),
                    size: rest.max(s.spill.as_ref().map_or(0, |b| 2 * b.size())),
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let spill = s.spill.as_ref().expect("spill");
            enc.copy_buffer_to_buffer(sim.moved_buffer(), u64::from(room) * record, spill, 0, rest);
        }
        s.check = None;
        if self.check && bodies > 0 {
            let full = staging(sim.device(), slot, u64::from(bodies) * record);
            enc.copy_buffer_to_buffer(sim.moved_buffer(), 0, &full, 0, full.size());
            s.check = Some((full, None));
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

    /// Where the `i`-th late extra of `slot`'s last `encode` goes: the
    /// slot's buffer and byte offset. The host copies it there in a
    /// submission after the step's own, before [`ReadbackRing::map`].
    pub fn late_target(&self, slot: usize, i: usize) -> (&wgpu::Buffer, u64) {
        let s = &self.slots[slot];
        let n = s.extras.len();
        let (_, at, _) = &s.extras[n - 1 - i];
        (&s.buffer, *at)
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
        if let Some((full, map)) = s.check.as_mut() {
            let (tx, rx) = channel();
            full.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            *map = Some(rx);
        }
    }

    /// Whether `slot`'s map has completed (`wait`: block until it has).
    pub fn ready(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        wait: bool,
    ) -> anyhow::Result<bool> {
        let header = self.header;
        let (s, spills) = (&mut self.slots[slot], &mut self.spills);
        if !wait_map(device, &mut s.map, wait)? {
            return Ok(false);
        }
        if s.spill_read.is_none() {
            // More moved than the slot copied: the rest from its spill, as a second read.
            let count = {
                let data = s
                    .buffer
                    .slice(header..header + 4)
                    .get_mapped_range()
                    .expect("mapped range");
                u32::from_le_bytes(data[..4].try_into().expect("count"))
            };
            let room = s.record_capacity;
            let records = count
                .saturating_sub(room)
                .min(s.spill.as_ref().map_or(0, |b| (b.size() / 32) as u32));
            let mut read = Spill {
                buffer: None,
                records,
                map: None,
            };
            if records > 0 {
                *spills += 1;
                let buffer = staging(device, slot, 32 * u64::from(records));
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("world readback spill"),
                });
                let spill = s.spill.as_ref().expect("spill");
                enc.copy_buffer_to_buffer(spill, 0, &buffer, 0, 32 * u64::from(records));
                queue.submit([enc.finish()]);
                let (tx, rx) = channel();
                buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                read.map = Some(rx);
                read.buffer = Some(buffer);
            }
            s.spill_read = Some(read);
        }
        if !wait_map(
            device,
            &mut s.spill_read.as_mut().expect("spill read").map,
            wait,
        )? {
            return Ok(false);
        }
        if let Some((_, map)) = s.check.as_mut() {
            if !wait_map(device, map, wait)? {
                return Ok(false);
            }
        }
        Ok(true)
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
            let spill = s.spill_read.take().filter(|r| r.records > 0);
            let rest = spill.as_ref().map_or(0, |r| r.records as usize);
            let count = (words[w(header)] as usize).min(s.record_capacity as usize + rest);
            self.last_moved = count as u32;
            let records = &data[s.records as usize..];
            let first = count.min(s.record_capacity as usize);
            let mut moved: Vec<PoseRecord> = (0..first)
                .map(|k| bytemuck::pod_read_unaligned(&records[k * 32..k * 32 + 32]))
                .collect();
            if let Some(b) = spill.as_ref().and_then(|r| r.buffer.as_ref()) {
                let more = b.slice(..).get_mapped_range().expect("spill mapped");
                moved.extend((0..count - first).map(|k| {
                    bytemuck::pod_read_unaligned::<PoseRecord>(&more[k * 32..k * 32 + 32])
                }));
            }
            if let Some((full, _)) = s.check.take() {
                let all = full.slice(..).get_mapped_range().expect("check mapped");
                let same = moved.len() == count
                    && (0..count)
                        .all(|k| bytemuck::bytes_of(&moved[k]) == &all[k * 32..k * 32 + 32]);
                self.checked.0 += 1;
                self.checked.1 += u64::from(!same);
            }
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

/// Whether `map` has completed (`wait`: block until it has); `None` is done.
fn wait_map(device: &wgpu::Device, map: &mut Option<MapRx>, wait: bool) -> anyhow::Result<bool> {
    let Some(rx) = map.as_ref() else {
        return Ok(true);
    };
    if wait {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        engine_gpu_wait::recv(rx).map_err(|e| anyhow::anyhow!("readback map failed: {e:?}"))?;
        *map = None;
        return Ok(true);
    }
    match rx.try_recv() {
        Ok(r) => {
            r.map_err(|e| anyhow::anyhow!("readback map failed: {e:?}"))?;
            *map = None;
            Ok(true)
        }
        Err(TryRecvError::Empty) => Ok(false),
        Err(TryRecvError::Disconnected) => anyhow::bail!("readback map callback dropped"),
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
