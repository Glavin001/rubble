//! Native Metal fracture viewer: Shardwright GLB → Slang AVBD (Slang → MSL) → Metal draw.
//! Native counterpart of `engine-webgpu/fracture/` (same kernels, dispatch order and ABI).

pub mod render;
pub mod sim;

pub mod kernels {
    pub struct Kernel {
        pub name: &'static str,
        pub entry: &'static str,
        pub msl: &'static str,
        pub threads: [u32; 3],
        /// (Metal `[[buffer(n)]]`, shared `BIND(slot)`)
        pub binds: &'static [(u32, u32)],
    }
    include!(concat!(env!("OUT_DIR"), "/fracture_kernels.rs"));
}

pub use render::{Orbit, Renderer};
pub use sim::FractureSim;
