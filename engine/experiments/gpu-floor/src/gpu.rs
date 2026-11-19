//! A minimal Metal host: device, shared buffers, a compiled kernel, timed dispatches.
use metal::*;

pub struct Gpu {
    pub device: Device,
    pub queue: CommandQueue,
}

impl Gpu {
    pub fn new() -> anyhow::Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    /// `fast`: Metal's fast math (contraction, approximate division and square root).
    pub fn pipeline(&self, src: &str, entry: &str, fast: bool) -> anyhow::Result<ComputePipelineState> {
        let o = CompileOptions::new();
        o.set_fast_math_enabled(fast);
        o.set_language_version(MTLLanguageVersion::V3_0);
        let lib = self.device.new_library_with_source(src, &o).map_err(|e| anyhow::anyhow!("MSL: {e}"))?;
        let f = lib.get_function(entry, None).map_err(|e| anyhow::anyhow!("{e}"))?;
        self.device.new_compute_pipeline_state_with_function(&f).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let len = bytes.len().max(16) as u64;
        let b = self.device.new_buffer(len, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents() as *mut u8, bytes.len()) };
        b
    }

    pub fn zeros(&self, bytes: usize) -> Buffer {
        let b = self.device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, bytes.max(16)) };
        b
    }

    /// One dispatch of `groups` threadgroups of `threads`; returns the GPU ms.
    pub fn run(&self, pso: &ComputePipelineState, bufs: &[&Buffer], groups: u64, threads: u64) -> f64 {
        self.run_n(pso, bufs, groups, threads, 1)
    }

    /// `n` dependent dispatches in one serial encoder (in order; each sees the last one's writes).
    pub fn run_n(&self, pso: &ComputePipelineState, bufs: &[&Buffer], groups: u64, threads: u64, n: usize) -> f64 {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for _ in 0..n {
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        gpu_ms(cb)
    }
}

pub fn read<T: bytemuck::Pod + Default + Clone>(b: &Buffer, n: usize) -> Vec<T> {
    let mut out = vec![T::default(); n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut out);
    unsafe { std::ptr::copy_nonoverlapping(b.contents() as *const u8, bytes.as_mut_ptr(), bytes.len()) };
    out
}

pub fn write<T: bytemuck::Pod>(b: &Buffer, data: &[T]) {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents() as *mut u8, bytes.len()) };
}

/// GPU execution time of a completed command buffer.
pub fn gpu_ms(cb: &CommandBufferRef) -> f64 {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    let obj = cb.as_ptr() as *mut objc::runtime::Object;
    let (start, end): (f64, f64) = unsafe { (msg_send![obj, GPUStartTime], msg_send![obj, GPUEndTime]) };
    (end - start) * 1000.0
}
