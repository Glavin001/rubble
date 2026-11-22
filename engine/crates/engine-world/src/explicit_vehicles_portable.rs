//! The World with its explicit vehicles as a portable World
//! (`engine_exec::world_portable`): the scenario matrix's `--world` rows for a host
//! without this World (engine-cuda's `world_replay`). A child of `explicit_vehicles`
//! (it reads the coupling's own buffers and clock).

use super::ExplicitVehicles;
use crate::World;
use engine_exec::world_portable::{CouplingHeader, Header, PortableWorld, VERSION};

impl ExplicitVehicles {
    /// The coupling's carried state and its work buffer's bytes. Blocking read.
    fn portable_coupling(&self, sim: &engine_webgpu::WgpuSim) -> (CouplingHeader, Vec<u8>) {
        let l = self.work_layout;
        let work = engine_webgpu::explicit_coupling::read_words(sim.device(), sim.queue(), &self.work, 0, l.bytes());
        // The indirect arguments hold the proxied list's count as the last plan left it
        // (copied from work, where it stays until the next window's plan).
        let list_args = [work[l.list_word as usize], 1, 1, 0];
        let header = CouplingHeader {
            layout: self.layout,
            work_layout: l,
            categories: self.def.categories,
            mask: self.def.mask,
            margin: self.plan_margin(),
            hull_geom: self.sims.iter().any(|v| v.surface_contact),
            piece_records: self.gpu.piece_window_records(),
            clock: self.clock,
            sleep_mask: self.sleep_mask.clone(),
            any_awake: self.any_awake,
            list_args,
            planned_by_query: self.last_query,
        };
        (header, bytemuck::cast_slice(&work).to_vec())
    }
}

impl World {
    /// This World at this moment (after a step and its poll) as a portable World: the
    /// rigid scene (`WgpuSim::portable_rigid`: bodies, colliders, hull arena, joints,
    /// materials, parameters; not the solver's caches), the coupling's carried state
    /// and the explicit vehicles' portable state, with no inputs yet
    /// ([`Self::save_explicit_world`] records them). Blocking reads. Not for a World with
    /// a fracture, terrain, characters or rigid vehicles: a loading host has none.
    pub fn explicit_world_portable(&self, t: f64, note: &str) -> anyhow::Result<PortableWorld> {
        let x = self.explicit.as_ref().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        x.gpu.wait();
        let (rigid, mut blobs) = self.sim.portable_rigid();
        let (coupling, work) = x.portable_coupling(&self.sim);
        blobs.push(work);
        blobs.push(x.gpu.portable(t, note).to_bytes());
        let header = Header {
            version: VERSION,
            t,
            note: note.to_string(),
            dt: self.sim.params().dt,
            rigid,
            coupling,
            inputs: Vec::new(),
            blobs: Vec::new(),
        };
        Ok(PortableWorld::new(header, blobs))
    }

    /// [`Self::explicit_world_portable`] now, then `steps` World steps (`before` first,
    /// each: a lab's controls), each vehicle's drive input of every step recorded with
    /// the state (read from the vehicles' records after the step): the file's bytes. The
    /// World is `steps` steps on when this returns. What `before` does beyond drive
    /// inputs (piece labels, retired nodes, grabs) is not recorded.
    pub fn save_explicit_world(
        &mut self,
        t: f64,
        note: &str,
        steps: usize,
        before: &mut dyn FnMut(&mut World) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u8>> {
        let mut p = self.explicit_world_portable(t, note)?;
        let (drive, vehicles) = {
            let x = self.explicit.as_ref().expect("checked");
            (x.gpu.layout().drive, x.sims.len())
        };
        for _ in 0..steps {
            before(self)?;
            self.step()?;
            self.poll_blocking()?;
            let x = self.explicit.as_ref().expect("checked");
            let inputs = (0..vehicles)
                .map(|v| {
                    drive.map(|d| {
                        let w = engine_webgpu::explicit_coupling::read_words(self.sim.device(), self.sim.queue(), x.gpu.misc_buffer(), d.input_offset(v), 16);
                        [w[0], w[1], w[2], w[3]]
                    })
                })
                .collect();
            p.header.inputs.push(inputs);
        }
        Ok(p.to_bytes())
    }
}
