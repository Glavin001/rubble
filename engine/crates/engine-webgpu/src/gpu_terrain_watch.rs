//! Terrain look-ahead on the native host (`slang/kernels/terrain_watch.slang`,
//! binding 52): after the rigid step, the strike on a watched heightfield with
//! the most kinetic energy within a horizon, so a host can open a terrain region
//! before the body reaches the ground. Encode-only: the pick rides in the host's
//! readback ([`WgpuSim::terrain_watch_buffer`]), like the step's counters.

use super::WgpuSim;
use engine_terrain::layout::MPM_WATCH_BINDING;

/// Heightfields watched at once.
pub const TERRAIN_WATCH_FIELDS: usize = 4;
const OUT_WORD: u64 = 16;
const OUT_WORDS: u64 = 14;

/// What the look-ahead watches for.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainWatch {
    /// Heightfield colliders to predict strikes on (at most
    /// [`TERRAIN_WATCH_FIELDS`]).
    pub fields: Vec<u32>,
    /// How far ahead each frame looks (s).
    pub horizon: f32,
    /// Samples of each path over the horizon (each straddle is bisected).
    pub samples: u32,
    /// A strike counts at this impact speed (m/s) ...
    pub min_speed: f32,
    /// ... and this kinetic energy (J) or more ...
    pub min_energy: f32,
    /// ... no sooner than this (s): a strike the host cannot open a region
    /// for in time is not reported.
    pub min_time: f32,
}

/// One predicted strike (as of the step that predicted it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainStrike {
    pub body: u32,
    /// The heightfield collider struck.
    pub field: u32,
    /// From the end of the predicting step (s).
    pub time: f32,
    /// The striking collider's centre at contact, and its velocity.
    pub point: [f32; 3],
    pub velocity: [f32; 3],
    /// That collider's bounding radius and its body's mass.
    pub radius: f32,
    pub mass: f32,
    pub energy: f32,
}

impl WgpuSim {
    /// Set what [`Self::terrain_encode_watch`] looks for.
    pub fn terrain_set_watch(&mut self, watch: &TerrainWatch) -> anyhow::Result<()> {
        anyhow::ensure!(
            watch.fields.len() <= TERRAIN_WATCH_FIELDS,
            "the terrain look-ahead watches at most {TERRAIN_WATCH_FIELDS} heightfields ({} given)",
            watch.fields.len()
        );
        for &f in &watch.fields {
            anyhow::ensure!(f < self.collider_count, "collider {f} does not exist");
            anyhow::ensure!(
                self.collider(f).shape == engine_schema::ShapeType::HeightField as u32,
                "collider {f} is not a heightfield"
            );
        }
        for (name, v) in [
            ("horizon", watch.horizon),
            ("min_speed", watch.min_speed),
            ("min_energy", watch.min_energy),
            ("min_time", watch.min_time),
        ] {
            anyhow::ensure!(
                v.is_finite() && v >= 0.0,
                "{name} must be finite and at least 0"
            );
        }
        anyhow::ensure!(watch.samples > 0, "samples must be at least 1");
        let mut w = [0u32; OUT_WORD as usize];
        w[0] = watch.fields.len() as u32;
        w[1..1 + watch.fields.len()].copy_from_slice(&watch.fields);
        w[5] = watch.horizon.to_bits();
        w[6] = watch.samples;
        w[7] = self.params.gravity_y.to_bits();
        w[8] = watch.min_speed.to_bits();
        w[9] = watch.min_energy.to_bits();
        w[10] = watch.min_time.to_bits();
        self.queue
            .write_buffer(&self.bufs[MPM_WATCH_BINDING], 0, bytemuck::cast_slice(&w));
        self.terrain_watch_on = !watch.fields.is_empty();
        Ok(())
    }

    /// Encode the look-ahead after the rigid step in `enc` (it reads the
    /// step's end state). Nothing when no field is watched.
    pub fn terrain_encode_watch(&mut self, enc: &mut wgpu::CommandEncoder) {
        if !self.terrain_watch_on {
            return;
        }
        // No pick yet: energy 0, collider "none" for the min.
        let mut reset = [0u32; OUT_WORDS as usize];
        reset[1] = u32::MAX;
        // Gravity may have changed since the watch was set.
        self.queue.write_buffer(
            &self.bufs[MPM_WATCH_BINDING],
            7 * 4,
            bytemuck::bytes_of(&self.params.gravity_y.to_bits()),
        );
        self.queue.write_buffer(
            &self.bufs[MPM_WATCH_BINDING],
            OUT_WORD * 4,
            bytemuck::cast_slice(&reset),
        );
        let colliders = u64::from(self.collider_count);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain watch"),
            timestamp_writes: None,
        });
        for name in [
            "terrain_watch_energy",
            "terrain_watch_pick",
            "terrain_watch_write",
        ] {
            self.disp(&mut pass, name, colliders, None);
        }
    }

    /// The look-ahead's pick, for a host's readback: (buffer, offset, bytes).
    pub fn terrain_watch_buffer(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.bufs[MPM_WATCH_BINDING], OUT_WORD * 4, OUT_WORDS * 4)
    }

    /// Whether a field is watched (the step then carries a watch readback).
    pub fn terrain_watch_enabled(&self) -> bool {
        self.terrain_watch_on
    }

    /// The strike a copy of [`Self::terrain_watch_buffer`] holds, if any.
    pub fn decode_terrain_strike(words: &[u32]) -> Option<TerrainStrike> {
        if words.len() < OUT_WORDS as usize || words[1] == u32::MAX || words[0] == 0 {
            return None;
        }
        let f = |k: usize| f32::from_bits(words[2 + k]);
        Some(TerrainStrike {
            body: words[2],
            field: words[3],
            time: f(2),
            point: [f(3), f(4), f(5)],
            velocity: [f(6), f(7), f(8)],
            radius: f(9),
            mass: f(10),
            energy: f(11),
        })
    }
}
