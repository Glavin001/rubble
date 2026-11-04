//! The timber yard: Shardwright's timber-frame house and two plank walls
//! (`fracture/assets/timber-yard.glb`) as one `World` on a [`Range`], shot apart with
//! the same cannonballs as Broadside. Free play by default (the page aims and fires); the
//! raze take sweeps the yard with guns until almost nothing stands, then rakes what is left.
//!
//! Frames. The world is the yard's own Shardwright frame (metres, Y up): the house from
//! x 0 to 8 along its front (z 0) and back to z 6, on a 0.3 m plinth, its upper floor at
//! 3.0 and its ceiling at 5.4; in front of it at z −2.5, the plain wall from x −5 to −1 and
//! the windowed wall from x −10 to −6, their boards facing −z.

pub use crate::range::*;
use rubble::*;

/// The yard's levels: 0 its 1518 members whole, 1 each member in about 0.3 m pieces, 2
/// (the leaf) in grain-stretched cells.
pub const LEVEL: u8 = 2;

/// Yard dimensions (Shardwright's `TimberHouse::default` and `timber_yard`).
pub mod yard {
    /// The house's footprint and storey heights.
    pub const HOUSE_X: [f32; 2] = [0.0, 8.0];
    pub const HOUSE_Z: [f32; 2] = [0.0, 6.0];
    pub const PLINTH: f32 = 0.3;
    pub const UPPER_FLOOR: f32 = 3.0;
    pub const CEILING: f32 = 5.4;
    /// The plank walls: x spans, the clad face's z, sill top and wall top.
    pub const PLAIN_WALL_X: [f32; 2] = [-5.0, -1.0];
    pub const WINDOW_WALL_X: [f32; 2] = [-10.0, -6.0];
    pub const WALL_FACE_Z: f32 = -2.5;
    pub const WALL_DEPTH: f32 = 0.115;
    pub const WALL_Y: [f32; 2] = [0.15, 2.55];
    /// Where the sweep runs (x) and the yard's middle (the camera's).
    pub const SWEEP_X: [f32; 2] = [-10.5, 8.5];
    pub const CENTRE: [f32; 3] = [-1.0, 2.5, 2.0];
}

/// The take.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YardTake {
    /// Nothing fires but the page.
    #[default]
    Free,
    /// Guns sweep the yard from the window wall to the house's far end and back, each
    /// ball at a chunk still part of a large piece, until almost none is; then a closing
    /// volley from front and back.
    Raze,
}

/// The raze's guns fire every this many steps (Broadside's every step): a house's worth of
/// breaking is costlier per ball than the ship's, and the take should stay near real time.
pub const FIRE_EVERY: u32 = 3;

/// How far the guns stand off their targets (m).
pub const STAND_OFF: f32 = 20.0;

/// The yard on a [`Range`] (its `structure`; `Deref`s to it), with the take's script.
pub struct TimberYard {
    pub range: Range,
    pub take: YardTake,
    /// The take's script runs; off: free play only.
    pub scripted: bool,
    pub razing: Razing,
    /// The frame the closing volley fired.
    pub finale: Option<u32>,
}

impl std::ops::Deref for TimberYard {
    type Target = Range;
    fn deref(&self) -> &Range {
        &self.range
    }
}

impl std::ops::DerefMut for TimberYard {
    fn deref_mut(&mut self) -> &mut Range {
        &mut self.range
    }
}

impl TimberYard {
    /// The yard from its baked GLB, on `gpu`, at fracture `level`, running `take`.
    pub fn new(gpu: GpuContext, glb: &[u8], level: u8, take: YardTake) -> anyhow::Result<Self> {
        let c = yard::CENTRE;
        let range = Range::new(gpu, glb, level, [c[0], c[2]])?;
        let chunks = range.chunks as usize;
        Ok(Self {
            range,
            take,
            scripted: take == YardTake::Raze,
            razing: Razing::new(chunks),
            finale: None,
        })
    }

    /// The raze is over: its closing volley landed and the wreck has had time to fall.
    pub fn over(&self) -> bool {
        self.finale
            .is_some_and(|f| (self.frame - f) as f32 * self.dt > 3.5)
    }

    /// One step; its events come back on a later `poll`.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.range.begin_step()?;
        if self.scripted && self.take == YardTake::Raze {
            self.raze()?;
        }
        self.range.end_step()
    }

    /// Where a gun stands to hit `target`: off the nearest face of what it is part of (the
    /// walls from the front; the house from its nearest side, or `far` from the opposite
    /// one, raking through), `drop` above it; the roof from above, front or back.
    pub fn muzzle(target: [f32; 3], far: bool, drop: f32) -> [f32; 3] {
        use yard::*;
        let [x, y, z] = target;
        if z < HOUSE_Z[0] - 1.0 {
            return [x, y + drop, z - STAND_OFF];
        }
        let side = if z < 0.5 * (HOUSE_Z[0] + HOUSE_Z[1]) {
            -1.0
        } else {
            1.0
        };
        let side = if far { -side } else { side };
        if y > CEILING {
            return [x, y + 0.6 * STAND_OFF, z + side * 0.5 * STAND_OFF];
        }
        // Nearest face: front/back (z) or the gable ends (x).
        let dz = (z - HOUSE_Z[0]).min(HOUSE_Z[1] - z);
        let dx = (x - HOUSE_X[0]).min(HOUSE_X[1] - x);
        if dx < dz {
            let end = if x < 0.5 * (HOUSE_X[0] + HOUSE_X[1]) {
                -1.0
            } else {
                1.0
            };
            let end = if far { -end } else { end };
            [x + end * STAND_OFF, y + drop, z]
        } else {
            [x, y + drop, z + side * STAND_OFF]
        }
    }

    /// Fire at `target` from `from`, landing on a whole step.
    fn fire(&mut self, from: [f32; 3], target: [f32; 3], extra: f32) -> anyhow::Result<()> {
        let d = glam::Vec3::from(target) - glam::Vec3::from(from);
        let t = ((d.length() / BALL_SPEED) / self.dt).ceil().max(1.0) * self.dt + extra;
        let v = self.launch(from, target, t);
        self.spawn_ball(from, v)
    }

    /// The raze's guns for one step, each at an intact chunk near the sweep (a plane
    /// running x across the yard and back); once almost nothing is intact (or at
    /// [`RAZE_SECONDS`]) the closing volley.
    fn raze(&mut self) -> anyhow::Result<()> {
        if self.finale.is_some() {
            return Ok(());
        }
        if self.razing.finished(&self.range) {
            return self.raze_volley();
        }
        if self.frame % FIRE_EVERY != 0 {
            return Ok(());
        }
        let t = self.seconds();
        let sweep = Razing::sweep(t, yard::SWEEP_X[0], yard::SWEEP_X[1]);
        let poses = self.chunk_poses();
        let frame = self.frame;
        for gun in 0..Razing::guns(t) {
            let Some(c) = self
                .razing
                .pick(&self.range, &poses, gun, |p| (p[0] - sweep).abs())
            else {
                break;
            };
            let h = (c ^ frame.wrapping_mul(97) ^ gun.wrapping_mul(7919)).wrapping_mul(2654435761);
            let target = poses[c as usize].position;
            let from = Self::muzzle(target, h % 4 == 0, 0.3 + 1.0 * ((h >> 8) % 3) as f32);
            self.fire(from, target, 0.0)?;
        }
        Ok(())
    }

    /// The closing volley: front and back at once into what still stands (or along the
    /// house if nothing does), ripple-fired a frame apart in pairs.
    pub fn raze_volley(&mut self) -> anyhow::Result<()> {
        self.finale = Some(self.frame);
        self.razing.refresh(&self.range);
        let poses = self.chunk_poses();
        let intact = &self.razing.intact;
        let mut aims: Vec<[f32; 3]> = intact
            .iter()
            .step_by((intact.len() / RAZE_VOLLEY).max(1))
            .take(RAZE_VOLLEY)
            .map(|&c| poses[c as usize].position)
            .collect();
        let mut k = 0;
        while aims.len() < RAZE_VOLLEY {
            let x = yard::HOUSE_X[0] + 8.0 * (k as f32 + 0.5) / RAZE_VOLLEY as f32;
            aims.push([x, yard::PLINTH + 0.5 + 0.8 * (k % 3) as f32, 3.0]);
            k += 1;
        }
        for (i, target) in aims.into_iter().enumerate() {
            let from = Self::muzzle(target, i % 2 == 1, 0.3);
            self.fire(from, target, (i / 2) as f32 * self.dt)?;
        }
        Ok(())
    }

    /// The raze's camera (eye, target): a slow orbit about the yard from the front left,
    /// low then rising, pulling out wide for the closing volley. Free play: none (the
    /// page's camera).
    pub fn camera(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.take != YardTake::Raze {
            return None;
        }
        let t = self.seconds();
        let centre = glam::Vec3::from(yard::CENTRE);
        let (radius, height) = match self.finale {
            Some(_) => (26.0, 10.0),
            None => (18.0 + 0.1 * t, 4.0 + 0.12 * t),
        };
        let a = -2.2 + 0.07 * t;
        let eye = centre + glam::Vec3::new(radius * a.cos(), height, radius * a.sin());
        Some((eye.to_array(), centre.to_array()))
    }

    /// Simulated seconds per wall second: real time in free play; the raze a little slow,
    /// slower as the closing volley lands.
    pub fn time_scale(&self) -> f32 {
        if self.take != YardTake::Raze {
            return 1.0;
        }
        match self.finale {
            Some(f) if (self.frame - f) as f32 * self.dt < 2.0 => 0.25,
            Some(_) => 0.5,
            None => 0.5,
        }
    }
}
