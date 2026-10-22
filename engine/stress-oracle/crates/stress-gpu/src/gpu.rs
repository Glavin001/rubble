//! The wgpu device and the helpers the solver's kernels share: shader loading (the
//! Slang-generated Metal library natively, WGSL elsewhere), explicit bind group layouts,
//! storage buffers, dispatch and readback.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::ShaderSet;

/// Which generated form of the shaders the device runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderPath {
    /// Slang's Metal output, compiled to a `.metallib` by `build.rs` (Metal backend).
    MetalLib,
    /// Slang's Metal output, compiled by the driver at load (Metal backend, no Xcode at build).
    MetalSource,
    /// Slang's WGSL output, translated by wgpu (WebGPU, or any backend).
    Wgsl,
}

/// Time spent blocked on the device in reads (`read_ranges`, `read_range`), in ns: what a
/// frame waits on the GPU; the rest of its wall time is the host's.
pub static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn wait(device: &wgpu::Device) {
    let t = web_time::Instant::now();
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    WAIT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// A GPU device and its queue (cloning shares the device: wgpu handles).
#[derive(Clone)]
pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub shader_path: ShaderPath,
    /// The device accepts Metal libraries directly (Metal backend).
    pub passthrough: bool,
    /// Metal libraries built with fast math (`STRESS_GPU_MATH=fast`); the default is the
    /// parity build without it.
    pub fast_math: bool,
    /// Storage buffers a shader stage may bind: the device's limit (10 in Chrome, 31 on
    /// Metal), or a lower `STRESS_GPU_STORAGE_LIMIT` to run the browser's per-entry
    /// layouts natively.
    pub storage_limit: u32,
}

/// What a binding of a kernel's group 0 holds (bindings are numbered in order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    Uniform,
    Storage,
    StorageRw,
}

impl Gpu {
    /// The default high-performance adapter (Metal on macOS, WebGPU in a browser).
    /// `STRESS_GPU_SHADERS=metallib|msl|wgsl` picks the shader path (default: the Metal
    /// library on Metal, WGSL elsewhere).
    pub fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        pollster::block_on(Self::with_instance(&instance, None))
    }

    /// A device on `instance` able to present to `surface` (if given).
    pub async fn with_instance(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Gpu, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let adapter_info = adapter.get_info();
        let metal = adapter_info.backend == wgpu::Backend::Metal;
        let passthrough = metal && adapter.features().contains(wgpu::Features::PASSTHROUGH_SHADERS);
        let requested = std::env::var("STRESS_GPU_SHADERS").ok();
        let shader_path = match (requested.as_deref(), passthrough) {
            (Some("wgsl"), _) | (None, false) => ShaderPath::Wgsl,
            (Some("msl"), true) => ShaderPath::MetalSource,
            (Some("metallib") | None, true) => ShaderPath::MetalLib,
            (Some(other), _) => return Err(format!("STRESS_GPU_SHADERS={other}: expected wgsl, or msl/metallib on the Metal backend")),
        };
        let mut required_features = if passthrough { wgpu::Features::PASSTHROUGH_SHADERS } else { wgpu::Features::empty() };
        // GPU timestamps for profiling, where available.
        required_features |= adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("stress-gpu"),
                required_features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU device: {e}"))?;
        let fast_math = match std::env::var("STRESS_GPU_MATH").as_deref() {
            Ok("fast") => true,
            Ok("precise") | Err(_) => false,
            Ok(other) => return Err(format!("STRESS_GPU_MATH={other}: expected fast or precise")),
        };
        let device_limit = device.limits().max_storage_buffers_per_shader_stage;
        let storage_limit = match std::env::var("STRESS_GPU_STORAGE_LIMIT") {
            Err(_) => device_limit,
            Ok(v) => match v.parse::<u32>() {
                Ok(n) if n <= device_limit => n,
                _ => return Err(format!("STRESS_GPU_STORAGE_LIMIT={v}: expected a count up to the device's {device_limit}")),
            },
        };
        Ok(Gpu { adapter, device, queue, adapter_info, shader_path, passthrough, fast_math, storage_limit })
    }

    /// Run later-built pipelines from another shader form; false (unchanged) when the
    /// device cannot load it.
    pub fn set_shader_path(&mut self, path: ShaderPath) -> bool {
        if path != ShaderPath::Wgsl && !self.passthrough {
            return false;
        }
        self.shader_path = path;
        true
    }

    /// The shader module of a Slang file, in the form this device runs.
    pub fn module(&self, set: &ShaderSet) -> wgpu::ShaderModule {
        let path = match self.shader_path {
            ShaderPath::MetalLib if set.metallib.is_empty() => ShaderPath::MetalSource,
            p => p,
        };
        if path == ShaderPath::Wgsl {
            return self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(set.name), source: wgpu::ShaderSource::Wgsl(set.wgsl.into()) });
        }
        let entry_points: Vec<wgpu::PassthroughShaderEntryPoint> =
            set.entries.iter().map(|&(name, size)| wgpu::PassthroughShaderEntryPoint { name: Cow::Borrowed(name), workgroup_size: size }).collect();
        let mut desc = wgpu::ShaderModuleDescriptorPassthrough { label: Some(set.name), entry_points: Cow::Owned(entry_points), ..Default::default() };
        if path == ShaderPath::MetalLib {
            let fast = self.fast_math && !set.metallib_fast.is_empty();
            desc.metallib = Some(Cow::Borrowed(if fast { set.metallib_fast } else { set.metallib }));
        } else {
            desc.msl = Some(Cow::Borrowed(set.msl));
        }
        // SAFETY: the library is Slang's output for this set's entry points, and every
        // pipeline using it is built with `layout`, whose group 0 lists the shader's
        // bindings in declaration order: the Metal buffer slots Slang assigns.
        unsafe { self.device.create_shader_module_passthrough(desc) }
    }

    /// The pipeline layout of a shader whose group 0 holds `bindings` (in order), seen
    /// by `stages`. Explicit, because a passed-through Metal library has no reflection.
    pub fn layout(&self, label: &str, bindings: &[Binding], stages: wgpu::ShaderStages) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let numbered: Vec<(u32, Binding)> = bindings.iter().enumerate().map(|(i, &b)| (i as u32, b)).collect();
        self.layout_numbered(label, &numbered, stages)
    }

    /// `layout` with each binding's number given: a layout may skip bindings the
    /// pipeline's entry point does not use.
    fn layout_numbered(&self, label: &str, bindings: &[(u32, Binding)], stages: wgpu::ShaderStages) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .map(|&(i, b)| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: match b {
                        Binding::Uniform => wgpu::BufferBindingType::Uniform,
                        Binding::Storage => wgpu::BufferBindingType::Storage { read_only: true },
                        Binding::StorageRw => wgpu::BufferBindingType::Storage { read_only: false },
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let group = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries });
        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&group)], immediate_size: 0 });
        (group, layout)
    }

    /// A compute pipeline for `entry` of `set`, with group 0 holding `bindings`.
    pub fn compute(&self, set: &ShaderSet, entry: &str, bindings: &[Binding]) -> Kernel {
        let module = self.module(set);
        // WGSL checks each storage binding's access against the layout (Metal libraries do
        // not): there the generated WGSL's declarations, the shader's own, say read or
        // read-write.
        let wgsl_bindings;
        let bindings = if self.shader_path == ShaderPath::Wgsl {
            wgsl_bindings = wgsl_access(set.wgsl, bindings);
            &wgsl_bindings[..]
        } else {
            bindings
        };
        let used = self.entry_subset(set, entry);
        let (group, layout) = match used {
            Some(used) => {
                let numbered: Vec<(u32, Binding)> = used.iter().map(|&n| (n, bindings[n as usize])).collect();
                let storage = numbered.iter().filter(|b| b.1 != Binding::Uniform).count() as u32;
                assert!(
                    storage <= self.storage_limit,
                    "{}::{entry} uses {storage} storage buffers; this device allows {} per shader stage",
                    set.name,
                    self.storage_limit
                );
                self.layout_numbered(entry, &numbered, wgpu::ShaderStages::COMPUTE)
            }
            None => self.layout(entry, bindings, wgpu::ShaderStages::COMPUTE),
        };
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let size = set.entries.iter().find(|e| e.0 == entry).map(|e| e.1 .0).unwrap_or(64);
        Kernel { pipeline, group: GroupLayout { layout: group, used: used.map(|u| u.to_vec()) }, workgroup: size }
    }

    /// Whether `entry` of `set` binds no more storage buffers than this device allows per
    /// stage (on the Metal path a pipeline binds its module's whole group 0).
    pub fn entry_fits(&self, set: &ShaderSet, entry: &str) -> bool {
        let storage = if self.shader_path == ShaderPath::Wgsl { crate::reflection::entry_storage(set.name, entry) } else { crate::reflection::module_storage(set.name) };
        storage.unwrap_or(0) <= self.storage_limit
    }

    /// `fused`, or where it binds more storage buffers than this device allows (WebGPU's
    /// 10), its phases (`reflection::PHASED`): entry points that do the same work in order,
    /// each binding fewer (dispatched as `fused` would be, one after another).
    pub fn staged(&self, set: &ShaderSet, fused: &str, bindings: &[Binding]) -> Staged {
        let entries: Vec<&str> = if self.entry_fits(set, fused) {
            vec![fused]
        } else {
            crate::reflection::phases(set.name, fused).unwrap_or_else(|| panic!("{}::{fused} binds more storage buffers than this device allows, and has no phases", set.name)).to_vec()
        };
        Staged { kernels: entries.iter().map(|e| self.compute(set, e, bindings)).collect() }
    }

    /// The bindings `entry`'s own layout holds, when its module declares more storage
    /// buffers than this device allows per stage (`None`: the module's whole group 0, as
    /// always on the Metal path, whose buffer slots follow the layout's order).
    fn entry_subset(&self, set: &ShaderSet, entry: &str) -> Option<&'static [u32]> {
        if self.shader_path != ShaderPath::Wgsl || crate::reflection::module_storage(set.name).unwrap_or(0) <= self.storage_limit {
            return None;
        }
        Some(crate::reflection::entry_bindings(set.name, entry).unwrap_or_else(|| panic!("no binding reflection for {}::{entry}", set.name)))
    }

    /// A bind group for `layout`, binding `buffers[i]` at binding `i` (only the bindings
    /// the layout holds).
    pub fn bind<L: BindLayout + ?Sized>(&self, layout: &L, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let keep = |i: usize| layout.used().is_none_or(|u| u.contains(&(i as u32)));
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: layout.raw(), entries: &entries })
    }

    /// A storage buffer initialised with `data` (copyable both ways).
    /// `storage`, rewriting `old` in place when it has the same size (queue order puts the
    /// write before any later submission, so the GPU sees the same contents).
    pub fn storage_into<T: bytemuck::Pod>(&self, label: &str, data: &[T], old: Option<&wgpu::Buffer>) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        match old {
            Some(old) if bytes.len() >= 16 && old.size() == bytes.len() as u64 => {
                self.queue.write_buffer(old, 0, bytes);
                old.clone()
            }
            _ => self.storage(label, data),
        }
    }

    pub fn storage<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        // Zero-length bindings are invalid: keep at least 16 bytes.
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let padded;
        let contents = if bytes.len() < 16 {
            padded = [bytes, &[0u8; 16][bytes.len()..]].concat();
            &padded[..]
        } else {
            bytes
        };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// A uniform buffer holding `value`.
    pub fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        // Padded to 16 bytes: a shader's uniform struct is (Metal checks the binding's
        // length against it).
        let bytes = bytemuck::bytes_of(value);
        let padded = [bytes, &[0u8; 16][..bytes.len().next_multiple_of(16) - bytes.len()]].concat();
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &padded,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// A uniform buffer holding an array.
    pub fn uniform_slice<T: bytemuck::Pod>(&self, label: &str, values: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Copy a buffer back to the host (blocking).
    pub fn read<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        self.read_range(buffer, 0, buffer.size() as usize / std::mem::size_of::<T>())
    }

    /// Byte ranges `(buffer, offset, size)` read back with one submission and one wait
    /// (each `read_range` costs a round trip of its own).
    pub fn read_ranges(&self, parts: &[(&wgpu::Buffer, u64, u64)]) -> Vec<Vec<u8>> {
        READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sizes: Vec<u64> = parts.iter().map(|p| p.2.div_ceil(4) * 4).collect();
        let total: u64 = sizes.iter().sum();
        if total == 0 {
            return parts.iter().map(|_| Vec::new()).collect();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: total,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut at = 0;
        for (p, &size) in parts.iter().zip(&sizes) {
            if size > 0 {
                encoder.copy_buffer_to_buffer(p.0, p.1, &staging, at, size);
            }
            at += size;
        }
        self.queue.submit([encoder.finish()]);
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = mapped.clone();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            r.expect("map readback");
            flag.store(true, std::sync::atomic::Ordering::Release);
        });
        wait(&self.device);
        engine_gpu_wait::until(|| mapped.load(std::sync::atomic::Ordering::Acquire));
        let view = staging.get_mapped_range(..).expect("mapped readback");
        let mut out = Vec::with_capacity(parts.len());
        let mut at = 0usize;
        for (p, &size) in parts.iter().zip(&sizes) {
            out.push(view[at..at + p.2 as usize].to_vec());
            at += size as usize;
        }
        drop(view);
        staging.unmap();
        out
    }

    /// Elements `[first, first + count)` of a buffer of `T`.
    pub fn read_range<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, first: usize, count: usize) -> Vec<T> {
        READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let offset = (first * std::mem::size_of::<T>()) as u64;
        let size = ((count * std::mem::size_of::<T>()) as u64).div_ceil(4) * 4;
        if size == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = mapped.clone();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            r.expect("map readback");
            flag.store(true, std::sync::atomic::Ordering::Release);
        });
        wait(&self.device);
        engine_gpu_wait::until(|| mapped.load(std::sync::atomic::Ordering::Acquire));
        let view = staging.get_mapped_range(..).expect("mapped readback");
        // Copied out element by element: the mapping need not be aligned for `T`.
        let out = bytemuck::pod_collect_to_vec(&view);
        drop(view);
        staging.unmap();
        out
    }
}

/// A kernel's group-0 layout, and which bindings it holds when that is fewer than its
/// module declares (`Gpu::compute`).
pub struct GroupLayout {
    pub layout: wgpu::BindGroupLayout,
    pub used: Option<Vec<u32>>,
}

impl std::ops::Deref for GroupLayout {
    type Target = wgpu::BindGroupLayout;
    fn deref(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

/// A layout `Gpu::bind` can build a group for.
pub trait BindLayout {
    fn raw(&self) -> &wgpu::BindGroupLayout;
    /// The bindings it holds (`None`: every binding).
    fn used(&self) -> Option<&[u32]>;
}

impl BindLayout for wgpu::BindGroupLayout {
    fn raw(&self) -> &wgpu::BindGroupLayout {
        self
    }
    fn used(&self) -> Option<&[u32]> {
        None
    }
}

impl BindLayout for GroupLayout {
    fn raw(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
    fn used(&self) -> Option<&[u32]> {
        self.used.as_deref()
    }
}

/// A compute pipeline and its group-0 layout.
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub group: GroupLayout,
    /// Threads per workgroup (x).
    pub workgroup: u32,
}

impl Kernel {
    /// Record a dispatch covering `n` items.
    pub fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, bind: &wgpu::BindGroup, n: usize) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups(n, self.workgroup), 1, 1);
    }
}

/// A kernel, or its phases in order (`Gpu::staged`): every one dispatched the same way.
pub struct Staged {
    pub kernels: Vec<Kernel>,
}

impl Staged {
    pub fn iter(&self) -> std::slice::Iter<'_, Kernel> {
        self.kernels.iter()
    }
}

/// Workgroups covering `n` items at `size` per group.
/// Read-backs (`Gpu::read_ranges` / `read_range`, each one wait on the GPU) since the
/// process started, on every device: the frame's one-read rule is checked against it.
pub static READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `READS` now.
pub fn reads() -> u64 {
    READS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn groups(n: usize, size: u32) -> u32 {
    (n as u32).div_ceil(size).max(1)
}

/// `bindings` with each storage binding read-only or read-write as the WGSL module declares
/// it (`@binding(N) @group(0) var<storage, read_write>`); uniforms and bindings the module
/// does not declare keep the table's kind.
fn wgsl_access(wgsl: &str, bindings: &[Binding]) -> Vec<Binding> {
    let mut out = bindings.to_vec();
    for line in wgsl.lines() {
        let Some(rest) = line.trim_start().strip_prefix("@binding(") else { continue };
        let Some((n, decl)) = rest.split_once(')') else { continue };
        let (Ok(n), true) = (n.parse::<usize>(), decl.contains("@group(0)")) else { continue };
        if let Some(b) = out.get_mut(n).filter(|b| **b != Binding::Uniform) {
            if decl.contains("var<storage, read_write>") {
                *b = Binding::StorageRw;
            } else if decl.contains("var<storage, read>") || decl.contains("var<storage>") {
                *b = Binding::Storage;
            }
        }
    }
    out
}
