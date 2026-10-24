//! Native Metal fracture lab: the browser lab's host plan, scenes and Slang
//! kernels (`engine-fracture`, Slang -> MSL) on Metal, drawn with the shared
//! Slang draw shader. Counterpart of `engine-webgpu/fracture/`.

pub mod render;
pub mod sim;
pub mod volley;

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
