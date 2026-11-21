//! The large-maps benchmark city (engine/docs/large-maps.md, "Gates and benchmark"): K
//! threshold cuboid houses on a grid over one square map, and the scenarios a step is
//! measured in. One builder for the native benchmark (`tests/city_scale.rs`) and the
//! browser's `city.html` (engine-world-wasm `city.rs`), so both build the same city
//! through [`World`] with the same parameters.
//!
//! Placement is incremental ([`City::place`]): a caller places as many houses as it likes
//! per call, so a page spreads a map's worth over frames and never blocks its thread.

use crate::*;

/// Steps every scenario settles the placed city for before it starts.
pub const CITY_SETTLE: u32 = 120;
/// Steps the benchmark measures after a scenario starts.
pub const CITY_MEASURE: u32 = 120;
/// The map's side (m): the benchmark's 2.2 km square.
pub const CITY_MAP: f32 = 2200.0;
/// Houses struck at once in [`CityScenario::War`].
pub const CITY_WAR_SHOTS: usize = 20;

/// What a city is: its house count, map, sleep policy and piece pool.
#[derive(Clone, Debug)]
pub struct CityDef {
    /// Houses (K): laid on a ceil(sqrt(K))² grid.
    pub houses: usize,
    /// The map's side (m), centred on the origin.
    pub map: f32,
    /// The large-map sleep policy ([`DestructibleAssetDesc::sleeping_solver_for`]):
    /// houses are placed awake, settle, and fall asleep with their resting contacts
    /// archived. `false`: the threshold solver's own suggestion (never asleep).
    pub sleep: bool,
    /// A world piece pool of this many slots ([`WorldDef::fracture_pieces`]); `None`:
    /// a body slot per chunk.
    pub pool: Option<u32>,
}

impl Default for CityDef {
    fn default() -> Self {
        Self {
            houses: 16,
            map: CITY_MAP,
            sleep: false,
            pool: None,
        }
    }
}

/// A scenario: what happens after the city settles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CityScenario {
    /// Nothing moves.
    Still,
    /// One kinematic car drives a street at 15 m/s, touching nothing.
    Drive,
    /// A static box far above the map, touching nothing.
    Spawn,
    /// One house (the middle lot) is struck.
    Demolish,
    /// [`CITY_WAR_SHOTS`] houses spread over the map are struck at once.
    War,
}

impl CityScenario {
    pub fn from_name(name: &str) -> anyhow::Result<Self> {
        Ok(match name {
            "still" => Self::Still,
            "drive" => Self::Drive,
            "spawn" => Self::Spawn,
            "demolish" => Self::Demolish,
            "war" => Self::War,
            _ => {
                anyhow::bail!("unknown city scenario {name:?} (still, drive, spawn, demolish, war)")
            }
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Still => "still",
            Self::Drive => "drive",
            Self::Spawn => "spawn",
            Self::Demolish => "demolish",
            Self::War => "war",
        }
    }
}

/// The city's house: the fracture lab's cuboid house (715 chunks, 1,549 bonds).
pub fn city_house() -> anyhow::Result<DestructibleAssetDesc> {
    DestructibleAssetDesc::cuboid_house(&Default::default())
}

impl CityDef {
    /// The solver: the threshold model's, with the large-map sleep policy if
    /// [`Self::sleep`].
    pub fn solver(&self, house: &DestructibleAssetDesc) -> anyhow::Result<AvbdParams> {
        if self.sleep {
            house.sleeping_solver_for(StrengthModel::Threshold)
        } else {
            house.solver_for(StrengthModel::Threshold)
        }
        .ok_or_else(|| anyhow::anyhow!("the city's house has no threshold solver"))
    }

    /// Room for K houses: clusters (a few dozen a house) and the pool, or a slot per
    /// chunk without one; the houses' own shapes; contacts for every chunk.
    pub fn capacity(&self, house: &DestructibleAssetDesc) -> anyhow::Result<WorldCapacity> {
        let k = self.houses;
        let stress = house
            .stress
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the city's house has no stress source"))?;
        let solver = house
            .solver_for(StrengthModel::Threshold)
            .ok_or_else(|| anyhow::anyhow!("the city's house has no threshold solver"))?;
        let scene = stress.threshold_scene(1.0, solver.dt)?;
        let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum::<usize>() * k;
        let one = stress_rubble::capacity_for(&scene);
        let mut c = one.clone();
        c.max_bodies = match self.pool {
            // Clusters (a few dozen a house) and the pool, not a slot per chunk.
            Some(p) => ((64 * k) as u32 + p + 64).next_power_of_two(),
            None => ((chunks + 64 + 4) as u32).next_power_of_two(),
        };
        // The houses' own shapes (a collider each), not k copies of one scene's headroom.
        let shapes: usize = house.chunks.iter().map(|c| c.shapes.len()).sum::<usize>() * k;
        c.max_colliders = ((shapes + 4096) as u32).next_power_of_two();
        c.max_hull_vertices = c
            .max_hull_vertices
            .max(((shapes * 4 + 4096) as u32).next_power_of_two());
        c.max_contacts = stress_rubble::threshold::contact_capacity(
            chunks,
            one.max_contacts * k as u32,
            1 << 30,
        );
        c.max_candidates = c.max_candidates.max(c.max_contacts);
        c.max_joints = c.max_joints.max(1 << 16);
        Ok(c)
    }

    /// The city's world: its capacity, the house's threshold solver and gravity,
    /// canonical contacts, sleep as [`Self::sleep`], the piece pool. Everything else
    /// is [`WorldDef::default`] (a page may set its timing, readback and kill bounds).
    pub fn world_def(&self, house: &DestructibleAssetDesc) -> anyhow::Result<WorldDef> {
        let solver = self.solver(house)?;
        Ok(WorldDef {
            capacity: self.capacity(house)?,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: self.sleep,
            contacts: ContactStorage::Canonical,
            fracture_pieces: self.pool,
            ..WorldDef::default()
        })
    }

    /// Lot centres (x, z): a ceil(sqrt(K))² grid over the map, centred on the origin;
    /// and the grid's pitch.
    pub fn lots(&self) -> (Vec<[f32; 2]>, f32) {
        let k = self.houses;
        let side = (k as f64).sqrt().ceil().max(1.0) as usize;
        let pitch = self.map / side as f32;
        let origin = -self.map / 2.0 + pitch / 2.0;
        let lots = (0..k)
            .map(|i| {
                [
                    origin + (i % side) as f32 * pitch,
                    origin + (i / side) as f32 * pitch,
                ]
            })
            .collect();
        (lots, pitch)
    }
}

/// A city being built in a world, then run: its ground, the house asset and the houses
/// placed so far.
#[derive(Clone, Debug)]
pub struct City {
    pub def: CityDef,
    pub lots: Vec<[f32; 2]>,
    pub pitch: f32,
    /// The ground plane's height.
    pub ground: f32,
    pub asset: DestructibleAssetId,
    /// The houses placed so far, in lot order.
    pub houses: Vec<DestructibleId>,
    /// The house asset's frame less its placement: where lot (0, 0)'s house goes.
    offset: [f32; 3],
}

impl City {
    /// Start a city in `world` (made from [`CityDef::world_def`]): the ground plane and
    /// the house asset; no houses yet ([`Self::place`]).
    pub fn begin(
        world: &mut World,
        house: &DestructibleAssetDesc,
        def: CityDef,
    ) -> anyhow::Result<Self> {
        let stress = house
            .stress
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the city's house has no stress source"))?;
        let placement = stress.placement();
        let ground = stress.ground_height().unwrap_or(0.0) - placement[1] as f32;
        let g = world.create_body(&BodyDef::default())?;
        world.create_shape(
            g,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: ground,
            },
        )?;
        let asset = world.create_destructible_asset(house)?;
        let (lots, pitch) = def.lots();
        Ok(Self {
            def,
            lots,
            pitch,
            ground,
            asset,
            houses: Vec::new(),
            offset: [
                -placement[0] as f32,
                -placement[1] as f32,
                -placement[2] as f32,
            ],
        })
    }

    /// Place up to `n` more houses (awake, at their lots, in lot order). Returns the
    /// houses still to place.
    pub fn place(&mut self, world: &mut World, n: usize) -> anyhow::Result<usize> {
        let end = (self.houses.len() + n).min(self.lots.len());
        for i in self.houses.len()..end {
            let l = self.lots[i];
            let p = [l[0] + self.offset[0], self.offset[1], l[1] + self.offset[2]];
            let d = world.create_destructible(&DestructibleDef {
                model: StrengthModel::Threshold,
                transform: Transform {
                    position: p,
                    ..Transform::default()
                },
                asleep: false,
                ..DestructibleDef::new(self.asset)
            })?;
            self.houses.push(d);
        }
        Ok(self.remaining())
    }

    /// Houses still to place.
    pub fn remaining(&self) -> usize {
        self.lots.len() - self.houses.len()
    }

    /// The street the drive scenario uses: halfway between the first two rows of lots
    /// (its z), and where the car starts (x, the map's west edge).
    pub fn street(&self) -> (f32, f32) {
        (-self.def.map / 2.0, self.lots[0][1] + self.pitch / 2.0)
    }

    /// The lots the shots of `scenario` strike (none for the others).
    pub fn targets(&self, scenario: CityScenario) -> Vec<[f32; 2]> {
        let k = self.lots.len();
        match scenario {
            CityScenario::Demolish if k > 0 => vec![self.lots[k / 2]],
            CityScenario::War => {
                let n = CITY_WAR_SHOTS.min(k);
                (0..n).map(|i| self.lots[i * k / n]).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Start `scenario` (the city placed and settled): the bodies it adds.
    pub fn start(&self, world: &mut World, scenario: CityScenario) -> anyhow::Result<Vec<BodyId>> {
        let mut added = Vec::new();
        match scenario {
            CityScenario::Still => {}
            CityScenario::Drive => {
                let (x, z) = self.street();
                let car = world.create_body(&BodyDef {
                    body_type: BodyType::Kinematic,
                    linear_velocity: [15.0, 0.0, 0.0],
                    ..BodyDef::dynamic([x, self.ground + 1.0, z])
                })?;
                world.create_box_shape(car, &ShapeDef::default(), [2.0, 0.75, 1.0])?;
                added.push(car);
            }
            CityScenario::Spawn => {
                let b = world.create_body(&BodyDef {
                    body_type: BodyType::Static,
                    ..BodyDef::dynamic([0.0, 500.0, 0.0])
                })?;
                world.create_box_shape(b, &ShapeDef::default(), [1.0, 1.0, 1.0])?;
                added.push(b);
            }
            CityScenario::Demolish | CityScenario::War => {
                for at in self.targets(scenario) {
                    added.push(city_shot(world, at, self.ground)?);
                }
            }
        }
        Ok(added)
    }
}

/// A shot at the house on lot `at`: a 0.6 m steel ball (a projectile) dropped from 30 m
/// above the ground at 40 m/s.
pub fn city_shot(world: &mut World, at: [f32; 2], ground: f32) -> anyhow::Result<BodyId> {
    city_shell(world, [at[0], ground + 30.0, at[1]], [0.0, -40.0, 0.0], 0.6)
}

/// A steel ball of `radius` m (a projectile, swept as a bullet) at `origin` moving at
/// `velocity` (m/s): a city shot, or one the page fires.
pub fn city_shell(
    world: &mut World,
    origin: [f32; 3],
    velocity: [f32; 3],
    radius: f32,
) -> anyhow::Result<BodyId> {
    let b = world.create_body(&BodyDef {
        linear_velocity: velocity,
        is_bullet: true,
        ..BodyDef::dynamic(origin)
    })?;
    world.create_sphere_shape(
        b,
        &ShapeDef {
            density: 7800.0,
            ..ShapeDef::default()
        },
        [0.0; 3],
        radius,
    )?;
    world.body_set_projectile(b, Some(Projectile::default()))?;
    Ok(b)
}

/// Bonds broken in a polled frame's destruction events.
pub fn bonds_broken<'a>(events: impl IntoIterator<Item = &'a DestructionEvent>) -> u32 {
    events
        .into_iter()
        .map(|e| match e {
            DestructionEvent::BondsBroken { bonds, .. } => bonds.len() as u32,
            _ => 0,
        })
        .sum()
}
