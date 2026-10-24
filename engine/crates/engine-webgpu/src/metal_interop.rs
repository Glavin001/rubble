//! Rubble on a host renderer's Metal device: a wgpu device and queue opened over the
//! host's `MTLDevice` and `MTLCommandQueue`, and the host's `MTLBuffer`s imported as wgpu
//! buffers. A renderer (Unreal's Metal RHI) then reads Rubble's outputs in place, and Rubble's
//! submissions go on the host's queue, so Metal's queue order orders them with its frames.
//!
//! The world itself is unchanged: `WgpuSim` and `rubble::World` take the returned device and
//! queue as any shared device (`GpuContext::Shared`).

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};
use wgpu::hal;

/// Open wgpu over the host's Metal device and command queue.
///
/// # Safety
/// `device` and `queue` are a live `MTLDevice` and an `MTLCommandQueue` of it; both are
/// retained here and outlive nothing the host frees.
pub unsafe fn shared_device(
    device: *mut c_void,
    queue: *mut c_void,
) -> anyhow::Result<(wgpu::AdapterInfo, wgpu::Device, wgpu::Queue)> {
    let raw_device = unsafe { Retained::retain(device.cast::<ProtocolObject<dyn MTLDevice>>()) }
        .ok_or_else(|| anyhow::anyhow!("null MTLDevice"))?;
    let raw_queue = unsafe { Retained::retain(queue.cast::<ProtocolObject<dyn MTLCommandQueue>>()) }
        .ok_or_else(|| anyhow::anyhow!("null MTLCommandQueue"))?;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::METAL;
    let instance = wgpu::Instance::new(desc);
    // The adapter for the same GPU (by Metal's registry id): its limits and features are the
    // device's.
    let wanted = raw_device.registryID();
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL))
        .into_iter()
        .find(|a| unsafe { a.as_hal::<hal::api::Metal>() }.is_some_and(|h| h.raw_device().registryID() == wanted))
        .ok_or_else(|| anyhow::anyhow!("no wgpu Metal adapter for the host's device"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let features = wgpu::Features::empty();
    let hal_device = unsafe { hal::metal::Device::device_from_raw(raw_device, features, &limits) };
    let hal_queue = unsafe { hal::metal::Queue::queue_from_raw(raw_queue, 1.0) };
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            hal::OpenDevice::<hal::api::Metal> {
                device: hal_device,
                queue: hal_queue,
            },
            &wgpu::DeviceDescriptor {
                label: Some("rubble on the host's Metal device"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            },
        )
    }?;
    Ok((info, device, queue))
}

/// The host's `MTLBuffer` as a wgpu buffer (copy destination and storage). Its whole length is
/// imported; hosts that sub-allocate pass the range they own as offsets to the copies.
///
/// # Safety
/// `buffer` is a live `MTLBuffer` of the device `device` was opened on.
pub unsafe fn import_buffer(device: &wgpu::Device, buffer: *mut c_void, label: &str) -> anyhow::Result<wgpu::Buffer> {
    let raw = unsafe { Retained::retain(buffer.cast::<ProtocolObject<dyn MTLBuffer>>()) }
        .ok_or_else(|| anyhow::anyhow!("null MTLBuffer"))?;
    let size = raw.length() as u64;
    let hal_buffer = unsafe { hal::metal::Device::buffer_from_raw(raw, size) };
    Ok(unsafe {
        device.create_buffer_from_hal::<hal::api::Metal>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )
    })
}
