//! Reads that resolve later (a GPU copy and `mapAsync`), never waiting in place.

use crate::WebWorld;
use wasm_bindgen::prelude::*;

/// Floats per body in [`WebWorld::read_bodies`].
pub const BODY_READ_STRIDE: usize = 14;

#[wasm_bindgen]
impl WebWorld {
    /// The GPU state of body slots `slots` as the queue leaves it, resolved once
    /// the copy is mapped: per slot [`BODY_READ_STRIDE`] floats, the solver-frame
    /// (centre of mass) position xyz, rotation xyzw, linear velocity xyz,
    /// angular velocity xyz and 1 if awake. For checks and occasional reads; the
    /// per-frame path is [`WebWorld::poll`]. The copy is submitted when the promise's
    /// work first runs ([`EXISTING_READS`]).
    #[wasm_bindgen(js_name = readBodies)]
    pub fn read_bodies(&self, slots: Vec<u32>) -> js_sys::Promise {
        let device = self.world.device().clone();
        let queue = self.world.queue().clone();
        let source = self.world.sim().body_buffer().clone();
        let record = std::mem::size_of::<engine_schema::BodyState>() as u64;
        let n = slots.len().max(1) as u64;
        let copies: Vec<u32> = slots.clone();
        let copy = copy_at(EXISTING_READS, move || {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readBodies"),
                size: n * record,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            for (k, &s) in copies.iter().enumerate() {
                enc.copy_buffer_to_buffer(&source, u64::from(s) * record, &staging, k as u64 * record, record);
            }
            queue.submit([enc.finish()]);
            staging
        });
        wasm_bindgen_futures::future_to_promise(async move {
            let staging = copy.await;
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx.await
                .map_err(|_| crate::err("readBodies: map dropped"))?
                .map_err(|e| crate::err(format!("readBodies: {e}")))?;
            let mut out = Vec::with_capacity(slots.len() * BODY_READ_STRIDE);
            {
                let data = staging.slice(..).get_mapped_range().expect("mapped");
                for k in 0..slots.len() {
                    let b: engine_schema::BodyState =
                        bytemuck::pod_read_unaligned(&data[k * record as usize..][..record as usize]);
                    out.extend_from_slice(&b.position);
                    out.extend_from_slice(&b.orientation);
                    out.extend_from_slice(&b.linear_vel);
                    out.extend_from_slice(&b.angular_vel);
                    let asleep = b.flags & engine_schema::BodyFlags::Sleeping as u32 != 0;
                    out.push(if asleep { 0.0 } else { 1.0 });
                }
            }
            staging.unmap();
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }
}

/// Copies of `(buffer, byte offset, byte size)` ranges, back to back, resolved
/// once mapped as one `Uint32Array` (sizes are multiples of 4).
pub(crate) fn read_ranges(world: &rubble::World, ranges: Vec<(wgpu::Buffer, u64, u64)>) -> js_sys::Promise {
    read_ranges_with(world, ranges, |words| Ok(js_sys::Uint32Array::from(words.as_slice()).into()))
}

/// [`read_ranges`], the words handed to `decode` once mapped; resolves to what
/// it returns.
pub(crate) fn read_ranges_with(
    world: &rubble::World,
    ranges: Vec<(wgpu::Buffer, u64, u64)>,
    decode: impl FnOnce(Vec<u32>) -> Result<JsValue, JsValue> + 'static,
) -> js_sys::Promise {
    read_ranges_on(world.device().clone(), world.queue().clone(), ranges, decode)
}

/// When a read's copy of the GPU state is encoded and submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopyAt {
    /// When the returned promise's work first runs: a microtask after the caller's
    /// task (`future_to_promise` spawns it), so after everything that task submits. A
    /// read issued before a step in the same task holds the state after that step.
    Polled,
    /// Before the read returns: the state after exactly the work submitted so far (a
    /// frame submitted next, in the same task, is not in it).
    Call,
}

/// The order of every read that pages already use (`readBodies` and the `read_ranges*`
/// family: poses, damage, terrain, guidance, mount loads, ...): as it has always been.
/// Pages issue these un-awaited inside a frame, before or after the step they submit
/// in the same task; changing the order moves what they see by a frame.
pub(crate) const EXISTING_READS: CopyAt = CopyAt::Polled;

/// `submit`'s result, with `submit` run now ([`CopyAt::Call`]) or when the returned
/// future is first polled ([`CopyAt::Polled`]).
pub(crate) fn copy_at<S: 'static>(
    at: CopyAt,
    submit: impl FnOnce() -> S + 'static,
) -> impl std::future::Future<Output = S> + 'static {
    let pending = match at {
        CopyAt::Call => Ok(submit()),
        CopyAt::Polled => Err(submit),
    };
    async move {
        match pending {
            Ok(done) => done,
            Err(submit) => submit(),
        }
    }
}

/// [`read_ranges_with`] on any device and queue (the explicit solver's buffers), the
/// copy submitted as every existing read's is ([`EXISTING_READS`]).
pub(crate) fn read_ranges_on(
    device: wgpu::Device,
    queue: wgpu::Queue,
    ranges: Vec<(wgpu::Buffer, u64, u64)>,
    decode: impl FnOnce(Vec<u32>) -> Result<JsValue, JsValue> + 'static,
) -> js_sys::Promise {
    read_ranges_at(EXISTING_READS, device, queue, ranges, decode)
}

/// [`read_ranges_on`] with its copy submitted before it returns: for reads issued
/// between frames submitted in one task, each stamped with its own step (the crash
/// lab's frame states, the drive state).
pub(crate) fn read_ranges_now_on(
    device: wgpu::Device,
    queue: wgpu::Queue,
    ranges: Vec<(wgpu::Buffer, u64, u64)>,
    decode: impl FnOnce(Vec<u32>) -> Result<JsValue, JsValue> + 'static,
) -> js_sys::Promise {
    read_ranges_at(CopyAt::Call, device, queue, ranges, decode)
}

fn read_ranges_at(
    at: CopyAt,
    device: wgpu::Device,
    queue: wgpu::Queue,
    ranges: Vec<(wgpu::Buffer, u64, u64)>,
    decode: impl FnOnce(Vec<u32>) -> Result<JsValue, JsValue> + 'static,
) -> js_sys::Promise {
    let total: u64 = ranges.iter().map(|r| r.2).sum();
    let copy = copy_at(at, move || {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readRanges"),
            size: total.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        let mut at = 0;
        for (b, offset, size) in &ranges {
            if *size > 0 {
                enc.copy_buffer_to_buffer(b, *offset, &staging, at, *size);
            }
            at += size;
        }
        queue.submit([enc.finish()]);
        staging
    });
    wasm_bindgen_futures::future_to_promise(async move {
        let staging = copy.await;
        let (tx, rx) = futures_channel::oneshot::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx.await
            .map_err(|_| crate::err("read: map dropped"))?
            .map_err(|e| crate::err(format!("read: {e}")))?;
        let words: Vec<u32> = {
            let data = staging.slice(..).get_mapped_range().expect("mapped");
            bytemuck::pod_collect_to_vec(&data[..total as usize])
        };
        staging.unmap();
        decode(words)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    fn poll<F: Future>(f: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        f.poll(&mut Context::from_waker(Waker::noop()))
    }

    /// Main's order, which every page that existed there depends on: a read's copy is
    /// submitted when its promise's work first runs, not when the read is called. A
    /// `readBodies()` followed by `step()` in one task holds the state after that step.
    #[test]
    fn existing_reads_submit_their_copy_when_polled_as_on_main() {
        assert_eq!(EXISTING_READS, CopyAt::Polled);
        let submits = Rc::new(Cell::new(0u32));
        let s = submits.clone();
        let read = copy_at(EXISTING_READS, move || {
            s.set(s.get() + 1);
            s.get()
        });
        // The caller's task goes on (a step is submitted here on a page).
        assert_eq!(submits.get(), 0, "nothing is submitted when the read is called");
        let mut read = std::pin::pin!(read);
        assert_eq!(poll(read.as_mut()), Poll::Ready(1));
        assert_eq!(submits.get(), 1);
    }

    /// The new reads (the crash lab's frame states, the drive state) submit before
    /// they return, once.
    #[test]
    fn a_read_now_submits_before_it_returns() {
        let submits = Rc::new(Cell::new(0u32));
        let s = submits.clone();
        let read = copy_at(CopyAt::Call, move || {
            s.set(s.get() + 1);
            s.get()
        });
        assert_eq!(submits.get(), 1);
        let mut read = std::pin::pin!(read);
        assert_eq!(poll(read.as_mut()), Poll::Ready(1));
        assert_eq!(submits.get(), 1, "polled, it does not submit again");
    }
}
