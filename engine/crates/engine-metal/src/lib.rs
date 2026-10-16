//! Metal backend: Slang → MSL, GPU-resident AVBD world.

mod device;
mod pipelines;
mod sim;
mod world;

pub use device::MetalDevice;
pub use sim::MetalSim;
pub use world::{MetalWorld, SceneBuilder};

use engine_schema::AbiProbe;
use glam::{Mat3, Vec3};
use metal::*;
use std::mem;

/// Gate 0: layout + atomic round-trip on the M3 GPU.
pub fn run_abi_probe() -> anyhow::Result<AbiProbe> {
    let device = MetalDevice::new()?;
    let source = include_str!(concat!(env!("OUT_DIR"), "/msl/abi_roundtrip.metal"));
    let lib = device
        .device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("MSL compile: {e}"))?;
    let func = lib
        .get_function("abi_roundtrip", None)
        .map_err(|e| anyhow::anyhow!("entry: {e}"))?;
    let pipeline = device
        .device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("pipeline: {e}"))?;

    let probe = AbiProbe::new(Vec3::new(10.0, 20.0, 30.0), Mat3::IDENTITY, 7);
    let probe_buf = device.device.new_buffer_with_data(
        &probe as *const _ as *const _,
        mem::size_of::<AbiProbe>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut atomic = 0u32;
    let atomic_buf = device.device.new_buffer_with_data(
        &mut atomic as *mut _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let t0 = std::time::Instant::now();
    let cb = device.queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&probe_buf), 0);
    encoder.set_buffer(1, Some(&atomic_buf), 0);
    let grid = MTLSize::new(64, 1, 1);
    let tg = MTLSize::new(64, 1, 1);
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let out = unsafe { *(probe_buf.contents() as *const AbiProbe) };
    let atom = unsafe { *(atomic_buf.contents() as *const u32) };
    anyhow::ensure!(atom == 1, "atomic counter expected 1, got {atom}");
    anyhow::ensure!(
        (out.position() - Vec3::new(11.0, 22.0, 33.0)).length() < 1e-5,
        "position mismatch {:?}",
        out.position()
    );
    anyhow::ensure!(out.inertia_col0[0] == 2.0, "inertia col0.x");
    anyhow::ensure!(out.inertia_col1[1] == 2.0, "inertia col1.y");
    anyhow::ensure!(out.inertia_col2[2] == 2.0, "inertia col2.z");
    anyhow::ensure!(out.counter == 8, "counter");
    log::info!("abi_probe ok in {gpu_ms:.3} ms (includes wait)");
    let _ = gpu_ms;
    Ok(out)
}
