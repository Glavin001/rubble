//! Async snapshot ring for non-blocking GPU->CPU state readback (W4b).
//!
//! The physics step writes body state to a GPU buffer every frame. Rendering
//! needs that state on the CPU, but a blocking `download` stalls the frame on a
//! `mapAsync` round trip (measured ~4.8 ms in Chrome — pure scheduling latency,
//! not bandwidth). This ring removes the stall by returning the most recently
//! *completed* snapshot (typically one frame old, which is imperceptible at
//! 60 fps and standard practice for GPU-resident simulation) while the current
//! frame's copy maps in the background.
//!
//! WASM-only: native builds keep the blocking download (native has no event-loop
//! latency problem, and the oracle/invariant tests want exact synchronous state).

#![cfg(target_arch = "wasm32")]

use crate::GpuContext;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Number of staging buffers in the ring. Three is enough to keep a slot free
/// for the current frame while up to two prior copies are still mapping.
const RING_SLOTS: usize = 3;

struct Slot {
    staging: wgpu::Buffer,
    /// Byte capacity of `staging` (grown on demand).
    cap_bytes: u64,
    /// Set by the `map_async` callback when this slot's copy is readable.
    done: Arc<AtomicBool>,
    /// Elements in the in-flight copy (valid only while `inflight`).
    len: u32,
    inflight: bool,
}

/// A ring of mapped staging buffers producing lag-tolerant CPU snapshots of a
/// GPU buffer without ever blocking the hot path.
pub struct SnapshotRing<T: bytemuck::Pod + bytemuck::Zeroable> {
    slots: Vec<Slot>,
    /// In-flight slot indices in submission order (front = oldest).
    order: VecDeque<usize>,
    /// Newest completed snapshot (CPU-side, properly aligned for `T`).
    latest: Vec<T>,
    have_latest: bool,
    elem_size: u64,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> SnapshotRing<T> {
    /// Create a ring sized for `capacity_elems` elements per snapshot (grows on
    /// demand if a later frame needs more).
    pub fn new(ctx: &GpuContext, capacity_elems: usize) -> Self {
        let elem_size = std::mem::size_of::<T>() as u64;
        let cap_bytes = (capacity_elems as u64 * elem_size).max(4);
        let slots = (0..RING_SLOTS)
            .map(|i| Slot {
                staging: Self::make_buffer(ctx, cap_bytes, i),
                cap_bytes,
                done: Arc::new(AtomicBool::new(false)),
                len: 0,
                inflight: false,
            })
            .collect();
        Self {
            slots,
            order: VecDeque::with_capacity(RING_SLOTS),
            latest: Vec::new(),
            have_latest: false,
            elem_size,
            _marker: PhantomData,
        }
    }

    fn make_buffer(ctx: &GpuContext, cap_bytes: u64, i: usize) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(match i {
                0 => "SnapshotRing slot 0",
                1 => "SnapshotRing slot 1",
                _ => "SnapshotRing slot 2",
            }),
            size: cap_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Record a copy of `src[0..len]` into a free slot, submit it, and register a
    /// non-blocking map. Never blocks. If every slot is still in flight (GPU
    /// stalled), the copy is skipped this frame and retried next frame — the
    /// consumer keeps the last completed snapshot.
    pub fn enqueue(&mut self, ctx: &GpuContext, src: &wgpu::Buffer, len: u32) {
        // Reclaim completed slots so a free one is available.
        self.poll();
        if len == 0 {
            return;
        }
        let byte_len = len as u64 * self.elem_size;

        let Some(slot_idx) = self.slots.iter().position(|s| !s.inflight) else {
            // All slots mapping — drop this frame's snapshot (rare).
            return;
        };

        // Grow the slot if this frame needs more room (e.g. body count rose).
        if byte_len > self.slots[slot_idx].cap_bytes {
            let new_cap = byte_len.next_power_of_two();
            self.slots[slot_idx].staging = Self::make_buffer(ctx, new_cap, slot_idx);
            self.slots[slot_idx].cap_bytes = new_cap;
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("snapshot_ring_copy"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &self.slots[slot_idx].staging, 0, byte_len);
        ctx.queue.submit(Some(encoder.finish()));

        let done = Arc::new(AtomicBool::new(false));
        {
            let done_cb = done.clone();
            self.slots[slot_idx]
                .staging
                .slice(..byte_len)
                .map_async(wgpu::MapMode::Read, move |result| {
                    // A failed map leaves `done` false; the slot is reclaimed on
                    // a later poll only once it eventually resolves. Errors here
                    // are non-fatal (the consumer just keeps the older snapshot).
                    if result.is_ok() {
                        done_cb.store(true, Ordering::SeqCst);
                    }
                });
        }
        self.slots[slot_idx].done = done;
        self.slots[slot_idx].len = len;
        self.slots[slot_idx].inflight = true;
        self.order.push_back(slot_idx);
    }

    /// Consume any completed copies (in submission order) into `latest`, so
    /// `latest` ends holding the newest completed snapshot. Non-blocking.
    pub fn poll(&mut self) {
        while let Some(&front) = self.order.front() {
            if !self.slots[front].done.load(Ordering::SeqCst) {
                // GPU completes copies in submission order, so once the oldest
                // in-flight copy isn't done, none behind it are either.
                break;
            }
            self.consume(front);
            self.order.pop_front();
        }
    }

    /// Copy a completed slot's mapped range into `latest` and free the slot.
    fn consume(&mut self, slot_idx: usize) {
        let len = self.slots[slot_idx].len;
        let byte_len = len as u64 * self.elem_size;
        {
            let slice = self.slots[slot_idx].staging.slice(..byte_len);
            let mapped = slice.get_mapped_range();
            // WebGPU mapped ranges may not be aligned for `T`; copy into an
            // aligned Vec (same pattern as GpuBuffer::download_async).
            let mut out = vec![T::zeroed(); len as usize];
            let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
            dst.copy_from_slice(&mapped[..dst.len()]);
            drop(mapped);
            self.latest = out;
        }
        self.slots[slot_idx].staging.unmap();
        self.slots[slot_idx].inflight = false;
        self.slots[slot_idx].done = Arc::new(AtomicBool::new(false));
        self.have_latest = true;
    }

    /// The most recently completed snapshot, or `None` before any completes.
    pub fn latest(&mut self) -> Option<&[T]> {
        self.poll();
        if self.have_latest {
            Some(&self.latest)
        } else {
            None
        }
    }

    /// Block (by yielding to the event loop) until the most recently enqueued
    /// copy is readable, then return it — a zero-lag drain for tests/queries.
    pub async fn sync(&mut self) -> &[T] {
        if let Some(&newest) = self.order.back() {
            let done = self.slots[newest].done.clone();
            while !done.load(Ordering::SeqCst) {
                crate::yield_now().await;
            }
            self.poll();
        }
        &self.latest
    }
}
