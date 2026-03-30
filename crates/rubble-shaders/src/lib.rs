#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::spirv;
use spirv_std::glam::UVec3;

/// Trivial test kernel: multiply each element by 2.
#[spirv(compute(threads(64)))]
pub fn multiply_by_two(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32],
) {
    let idx = id.x as usize;
    if idx < data.len() {
        data[idx] *= 2.0;
    }
}
