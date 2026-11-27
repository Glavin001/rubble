//! The crash reference suite on CUDA: [`engine_softbody::crash_reference::Runner`] over
//! [`CudaExplicitBeams`]. The scenario, its fleet, the sampling and every metric are the
//! shared host code; this is the whole of what CUDA adds.

use crate::CudaExplicitBeams;
use engine_softbody::crash_reference::{BeamState, Built, Contact, Runner, Sample};

pub struct CudaRunner(pub CudaExplicitBeams);

impl CudaRunner {
    pub fn new(built: &Built) -> anyhow::Result<Self> {
        let mut g = CudaExplicitBeams::new(&built.fleet)?;
        g.set_vehicle_contact(true);
        g.set_group_split(true);
        g.set_continuous(built.contact == Contact::Continuous).map_err(anyhow::Error::msg)?;
        Ok(Self(g))
    }
}

impl Runner for CudaRunner {
    fn frame(&mut self, steps: u32) -> Result<(), String> {
        self.0.frame(steps).map_err(|e| e.to_string())
    }
    fn sample(&mut self) -> Result<Sample, String> {
        let e = |e: engine_exec::ExecError| e.to_string();
        Ok(Sample {
            t: 0.0,
            pos: self.0.positions().map_err(e)?,
            vel: self.0.velocities().map_err(e)?,
            beams: self.0.beams().map_err(e)?.into_iter().map(|v| v.into_iter().map(|b| BeamState { rest: b.rest, broken: b.broken }).collect()).collect(),
        })
    }
}
