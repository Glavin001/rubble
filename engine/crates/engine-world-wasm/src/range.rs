//! The shooting range under Broadside (`broadside.rs`) and the timber yard
//! (`timber_yard.rs`): one baked Shardwright GLB as one `World`, a `Threshold` destructible
//! (support loss, softwood crushing to splinter particles) on a static ground slab, and the
//! cannonballs fired at it. Each ball is a threshold striker whose breaks are predicted at
//! the end of the frame before it lands (fracture is forward only, AGENTS.md). The scene's
//! script (if any) sits on top: it fires through [`Range::spawn_ball`] between
//! [`Range::begin_step`] and [`Range::end_step`].

use rubble::*;

/// A cannonball: 24 lb of iron (10.9 kg, 0.074 m radius).
pub const BALL_MASS: f32 = 10.9;
pub const BALL_RADIUS: f32 = 0.074;
/// Muzzle velocity: a 24-pounder's ~450 m/s, spent to this at close range.
pub const BALL_SPEED: f32 = 300.0;
/// Balls in flight or at rest the scene keeps (the oldest is removed past this).
pub const MAX_BALLS: usize = 256;
/// Balls in strike prediction's shot table at once (its limit is 64).
pub const MAX_STRIKING: usize = 64;
/// Crushed wood becomes particles (`CrushMass::Particles`): about one per `PARTICLE_SIZE`
/// cube of the chunk's volume, at most `PARTICLES_PER_CHUNK`, its mass conserved. Each
/// collides with the structure and the ground but not with another particle; the page draws
/// each as a few splinters.
pub const PARTICLE_SIZE: f32 = 0.08;
pub const PARTICLES_PER_CHUNK: u32 = 24;
/// Continuous collision for fast particles. Off: the takes have more fast particles at
/// once than the engine's CCD slots (`CCD_BODY_CAP`), so a fast particle may pass
/// through a thin plank.
pub const PARTICLE_CCD: bool = false;
/// Particle bodies reserved when the structure is placed: Broadside's raze leaves about
/// 23,000, so room for it and for free play after either take. A run that crushes more
/// fails with an error.
pub const CRUSH_BUDGET: u32 = 49152;
/// Chunks one step may crush: the most the engine takes (its event table is under 4096; a
/// step past it fails with an error). The default 256 is too few for a wrecked house: a
/// pile of wreck coming down on weakened members crushes a burst of chunks at once (a
/// player's sustained fire into the timber yard reached 183 in one step, and past 256).
/// Each step reads back the table (96 bytes a slot, ~390 KB).
pub const CRUSH_PER_STEP: u32 = 4095;

/// Crushing for the wood: the stress import's `timber` (structural softwood, the stress
/// oracle's preset). Iron and concrete are not crushed.
pub fn crush(budget: u32) -> Crush {
    let p = stress_ref::crush::presets::softwood();
    Crush {
        materials: vec![(
            "timber".into(),
            CrushMaterial {
                cap_pressure: p.cap_pressure,
                cohesion: p.cohesion,
                friction_slope: p.friction_slope,
                crush_energy: p.crush_energy,
                crush_viscosity: p.crush_viscosity,
            },
        )],
        mass: CrushMass::Particles {
            size: PARTICLE_SIZE,
            max_per_chunk: PARTICLES_PER_CHUNK,
            budget,
            continuous: PARTICLE_CCD,
        },
        max_per_step: CRUSH_PER_STEP,
    }
}

/// Crushed wood's debris: a crush particle (`half` its radius, thrice; drawn as a few
/// splinters) or a box fragment (`half` its extents); where it was made.
#[derive(Clone, Copy, Debug)]
pub struct Splinter {
    pub body: BodyId,
    pub half: [f32; 3],
    pub at: [f32; 3],
    pub particle: bool,
}

/// What a frame did (since the last `take_report`).
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub broken: usize,
    pub crushed: usize,
    pub splits: usize,
    pub shots: usize,
}

/// Razing: a chunk in a piece of fewer than this many chunks is destroyed (loose wreckage).
pub const RAZE_PIECE: usize = 8;
/// Guns firing each step, rising through the take.
pub const RAZE_GUNS: [(f32, u32); 3] = [(0.0, 1), (6.0, 2), (14.0, 3)];
/// Seconds one sweep of the structure's length takes.
pub const RAZE_SWEEP: f32 = 7.0;
/// A chunk aimed at is not aimed at again for this long (its ball breaks it first).
pub const RAZE_REAIM: f32 = 0.8;
/// The closing volley fires once the intact share is under this, once it has stopped
/// falling (by `RAZE_PROGRESS` in `RAZE_STALL` seconds: what is left is wreckage the balls
/// no longer break, or what is anchored), or at `RAZE_SECONDS`.
pub const RAZE_DONE: f32 = 0.03;
pub const RAZE_SECONDS: f32 = 40.0;
pub const RAZE_PROGRESS: f32 = 0.005;
pub const RAZE_STALL: f32 = 3.0;
/// Balls in the closing volley.
pub const RAZE_VOLLEY: usize = 40;

pub struct Range {
    pub world: World,
    /// The structure: the GLB's chunks as one threshold destructible.
    pub structure: DestructibleId,
    pub dt: f32,
    /// Steps taken.
    pub frame: u32,
    /// Each chunk's rest pose in the render scene (`engine_fracture::build_scene`), the
    /// frame its render vertices are in.
    pub rest: Vec<Transform>,
    /// The render scene's body count (the ground, the chunks, its parked projectiles).
    pub render_bodies: usize,
    pub chunks: u32,
    /// Each chunk's Shardwright fragment and its render colour (its mesh's mean).
    pub fragment: Vec<Option<u32>>,
    pub color: Vec<[f32; 3]>,
    /// Balls in the world, oldest first, with the frame each was fired.
    pub balls: Vec<(BodyId, u32)>,
    /// The balls still in strike prediction's shot table, oldest first.
    striking: Vec<(BodyId, u32)>,
    /// Every shot fired: (frame, from, velocity), for the camera and the report.
    pub log: Vec<(u32, [f32; 3], [f32; 3])>,
    pub total: Report,
    /// Free play: shots asked for by the page, fired before the next step.
    pending: Vec<([f32; 3], [f32; 3])>,
    /// Chunks crushed so far (hidden: their mass is in `splinters`).
    pub crushed: Vec<bool>,
    /// What crushed wood left (particles or box fragments); `splinters_version` counts
    /// changes.
    pub splinters: Vec<Splinter>,
    pub splinters_version: u32,
}

fn capacity(asset: &DestructibleAssetDesc) -> anyhow::Result<WorldCapacity> {
    let stress = asset
        .stress
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("asset has no stress import"))?;
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("asset has no threshold solver"))?;
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    // Every chunk may become a body, plus the crush splinters, the balls and a character.
    let bodies = chunks as u32 + CRUSH_BUDGET + MAX_BALLS as u32 + 64;
    c.max_bodies = c.max_bodies.max(bodies.next_power_of_two());
    // Colliders: every chunk's hulls (a chunk of a coarse level is a compound of several),
    // one for each splinter, ball and the rest.
    let hulls: usize = scene
        .bodies
        .iter()
        .flat_map(|b| &b.chunks)
        .map(|c| c.compound.as_ref().map_or(1, |p| p.len().max(1)))
        .sum();
    c.max_colliders = c
        .max_colliders
        .max(hulls as u32 + CRUSH_BUDGET + MAX_BALLS as u32 + 64);
    // Contacts for every body the take may have: the wreck piles up on the ground.
    c.max_contacts = stress_rubble::threshold::contact_capacity(
        chunks + CRUSH_BUDGET as usize + MAX_BALLS,
        c.max_contacts,
        1 << 30,
    );
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    // A deep pile of wreck needs more colours than standing structure; colours past the
    // planned ones run in the sparse tail, solved the same way.
    c.max_colors = c.max_colors.max(64);
    Ok(c)
}

impl Range {
    /// The structure from its baked GLB at fracture `level`, on `gpu`, standing on a static
    /// ground slab (its top at y 0) centred under `ground_at` (x, z).
    pub fn new(
        gpu: GpuContext,
        glb: &[u8],
        level: u8,
        ground_at: [f32; 2],
    ) -> anyhow::Result<Self> {
        let parsed = engine_fracture::parse_fracture_glb(glb)?;
        let scene = engine_fracture::build_scene(&parsed, level, None)?;
        let asset = DestructibleAssetDesc::from_parsed(&parsed, &scene, level)?;
        drop(parsed);
        let solver = asset
            .solver_for(StrengthModel::Threshold)
            .ok_or_else(|| anyhow::anyhow!("asset has no threshold solver"))?;
        let def = WorldDef {
            capacity: capacity(&asset)?,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            // One character; its candidate scratch is CHAR_MAX_CANDIDATES records.
            characters: CharacterCapacity {
                max_characters: 4,
                ..Default::default()
            },
            ..WorldDef::default()
        };
        let mut world = World::new(gpu, &def)?;
        let a = world.create_destructible_asset(&asset)?;
        let placement = asset
            .stress
            .as_ref()
            .map(|s| s.placement())
            .unwrap_or([0.0; 3]);
        let structure = world.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(placement.map(|x| -x as f32), [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            support_loss: true,
            crush: Some(crush(CRUSH_BUDGET)),
            ..DestructibleDef::new(a)
        })?;
        // The ground the wreck falls on: a static slab, its top at y 0 (the drawn ground;
        // the structure stands only by its anchored chunks).
        let ground = world.create_body(&BodyDef {
            position: [ground_at[0], -5.0, ground_at[1]],
            ..BodyDef::default()
        })?;
        world.create_box_shape(ground, &ShapeDef::default(), [200.0, 5.0, 200.0])?;
        let chunks = asset.chunks.len() as u32;
        // Render bodies: the ground (0), then one per chunk in order (from_glb's order).
        let f = |v: [f64; 3]| v.map(|x| x as f32);
        let rest: Vec<Transform> = scene
            .bodies
            .iter()
            .skip(1)
            .take(chunks as usize)
            .map(|b| Transform::new(f(b.p), b.q.map(|x| x as f32)))
            .collect();
        let fragment = asset.chunks.iter().map(|c| c.fragment).collect();
        let mut sums = vec![([0.0f32; 3], 0u32); chunks as usize];
        for v in scene.vertices.chunks_exact(10) {
            if let Some(k) = (v[9] as usize)
                .checked_sub(1)
                .filter(|&k| k < chunks as usize)
            {
                for i in 0..3 {
                    sums[k].0[i] += v[6 + i];
                }
                sums[k].1 += 1;
            }
        }
        let color = sums
            .iter()
            .map(|(c, n)| c.map(|x| x / (*n).max(1) as f32))
            .collect();
        Ok(Self {
            world,
            structure,
            dt: solver.dt,
            frame: 0,
            rest,
            render_bodies: scene.bodies.len(),
            chunks,
            fragment,
            color,
            balls: Vec::new(),
            striking: Vec::new(),
            log: Vec::new(),
            total: Report::default(),
            pending: Vec::new(),
            crushed: vec![false; chunks as usize],
            splinters: Vec::new(),
            splinters_version: 0,
        })
    }

    pub fn seconds(&self) -> f32 {
        self.frame as f32 * self.dt
    }

    /// Fire a ball from `from` at `velocity` before the next step.
    pub fn shoot(&mut self, from: [f32; 3], velocity: [f32; 3]) {
        self.pending.push((from, velocity));
    }

    /// The launch that reaches `target` from `from` in `t` seconds under gravity.
    pub fn launch(&self, from: [f32; 3], target: [f32; 3], t: f32) -> [f32; 3] {
        let g = -9.81f32;
        [
            (target[0] - from[0]) / t,
            (target[1] - from[1]) / t - 0.5 * g * t,
            (target[2] - from[2]) / t,
        ]
    }

    /// Balls half a second old are spent (through the structure or stopped in it): they leave
    /// strike prediction's shot table (64 at most) and roll on as plain rigid balls.
    pub fn retire_spent(&mut self) -> anyhow::Result<()> {
        let spent = (0.5 / self.dt).round() as u32;
        while self
            .striking
            .first()
            .is_some_and(|&(_, f)| self.frame >= f + spent)
        {
            let (ball, _) = self.striking.remove(0);
            if self.world.body_is_valid(ball) {
                self.world.body_set_projectile(ball, None)?;
            }
        }
        Ok(())
    }

    pub fn spawn_ball(&mut self, from: [f32; 3], v: [f32; 3]) -> anyhow::Result<()> {
        if self.balls.len() >= MAX_BALLS {
            let (old, _) = self.balls.remove(0);
            self.striking.retain(|(b, _)| *b != old);
            self.world.destroy_body(old)?;
        }
        // A full shot table (raze fires up to three a step): the oldest leaves it early,
        // long through the structure by now.
        if self.striking.len() >= MAX_STRIKING {
            let (old, _) = self.striking.remove(0);
            if self.world.body_is_valid(old) {
                self.world.body_set_projectile(old, None)?;
            }
        }
        let ball = self.world.create_body(&BodyDef {
            linear_velocity: v,
            is_bullet: true,
            ..BodyDef::dynamic(from)
        })?;
        self.world.create_sphere_shape(
            ball,
            &ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            BALL_RADIUS,
        )?;
        let i = 0.4 * BALL_MASS * BALL_RADIUS * BALL_RADIUS;
        self.world.body_set_mass_data(
            ball,
            MassData {
                mass: BALL_MASS,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )?;
        self.world
            .body_set_projectile(ball, Some(Projectile::default()))?;
        self.balls.push((ball, self.frame));
        self.striking.push((ball, self.frame));
        self.log.push((self.frame, from, v));
        self.total.shots += 1;
        Ok(())
    }

    /// The start of a step, before the script fires: balls half a second old leave strike
    /// prediction.
    pub fn begin_step(&mut self) -> anyhow::Result<()> {
        self.retire_spent()
    }

    /// The end of a step, after the script fired: the shots asked for, then the step; its
    /// events come back on a later `poll`.
    pub fn end_step(&mut self) -> anyhow::Result<()> {
        for (from, v) in std::mem::take(&mut self.pending) {
            self.spawn_ball(from, v)?;
        }
        self.world.step()?;
        self.frame += 1;
        Ok(())
    }

    /// One step with no script.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.begin_step()?;
        self.end_step()
    }

    /// Chunks still in large pieces: not crushed, and on a body with at least `piece` of
    /// the structure's chunks.
    pub fn intact(&self, piece: usize) -> Vec<u32> {
        let bodies: Vec<Option<BodyId>> = (0..self.chunks)
            .map(|c| {
                (!self.crushed[c as usize])
                    .then(|| self.world.destructible_chunk_body(self.structure, c).ok())
                    .flatten()
            })
            .collect();
        let mut count = std::collections::HashMap::<BodyId, usize>::new();
        for b in bodies.iter().flatten() {
            *count.entry(*b).or_default() += 1;
        }
        (0..self.chunks)
            .filter(|&c| bodies[c as usize].is_some_and(|b| count[&b] >= piece))
            .collect()
    }

    /// Collect the frames the GPU has finished: destruction counts and debris.
    pub fn poll(&mut self) -> anyhow::Result<Report> {
        let mut r = Report::default();
        for f in self.world.poll()? {
            self.absorb(&f, &mut r);
        }
        Ok(r)
    }

    /// Wait for every step in flight.
    pub fn poll_blocking(&mut self) -> anyhow::Result<Report> {
        let mut r = Report::default();
        for f in self.world.poll_blocking()? {
            self.absorb(&f, &mut r);
        }
        Ok(r)
    }

    fn absorb(&mut self, f: &Frame, r: &mut Report) {
        for e in &f.destruction_events {
            match e {
                DestructionEvent::BondsBroken { bonds, .. } => r.broken += bonds.len(),
                DestructionEvent::ChunkCrushed {
                    chunk, fragments, ..
                } => {
                    r.crushed += 1;
                    if let Some(c) = self.crushed.get_mut(*chunk as usize) {
                        *c = true;
                    }
                    for &f in fragments {
                        let half = self
                            .world
                            .body_shapes(f)
                            .ok()
                            .and_then(|s| s.first().copied())
                            .and_then(|s| match self.world.shape_geometry(s) {
                                Ok(Geometry::Box { half, .. }) => Some(*half),
                                Ok(Geometry::Sphere { radius: r, .. }) => Some([*r; 3]),
                                _ => None,
                            });
                        let particle = half.is_some_and(|h| h[0] == h[1] && h[1] == h[2]);
                        let at = self
                            .world
                            .body_transform(f)
                            .ok()
                            .flatten()
                            .map_or([0.0; 3], |t| t.position);
                        if let Some(half) = half {
                            self.splinters.push(Splinter {
                                body: f,
                                half,
                                at,
                                particle,
                            });
                        }
                    }
                    self.splinters_version += 1;
                }
                DestructionEvent::Split { .. } => r.splits += 1,
                _ => {}
            }
        }
        self.total.broken += r.broken;
        self.total.crushed += r.crushed;
        self.total.splits += r.splits;
        r.broken = 0;
        r.crushed = 0;
        r.splits = 0;
    }

    /// Each chunk's world pose: its body's transform (every body origin is the instance
    /// frame) times its render rest pose. A chunk whose body has no pose yet is at rest.
    pub fn chunk_poses(&self) -> Vec<Transform> {
        (0..self.chunks)
            .map(|c| {
                let body = self.world.destructible_chunk_body(self.structure, c).ok();
                let t = body
                    .and_then(|b| self.world.body_transform(b).ok().flatten())
                    .unwrap_or(Transform::IDENTITY);
                compose(&t, &self.rest[c as usize])
            })
            .collect()
    }

    /// Every ball's pose, oldest first.
    pub fn ball_poses(&self) -> Vec<Transform> {
        self.balls
            .iter()
            .filter_map(|(b, _)| self.world.body_transform(*b).ok().flatten())
            .collect()
    }
}

pub fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (
        glam::Quat::from_array(a.rotation),
        glam::Vec3::from(a.position),
    );
    let (qb, pb) = (
        glam::Quat::from_array(b.rotation),
        glam::Vec3::from(b.position),
    );
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}

/// A raze's aim and progress over a [`Range`]: the guns sweep a plane along the structure
/// and back, each ball at a chunk still part of a large piece ([`RAZE_PIECE`]), until
/// almost none is ([`Razing::finished`]).
#[derive(Clone, Debug)]
pub struct Razing {
    /// The frame each chunk was last aimed at.
    pub aimed: Vec<Option<u32>>,
    /// The chunks still in large pieces (refreshed every few steps) and their share.
    pub intact: Vec<u32>,
    pub intact_share: f32,
    /// The lowest intact share so far and the frame it last fell by `RAZE_PROGRESS`.
    progress: (f32, u32),
}

impl Razing {
    pub fn new(chunks: usize) -> Self {
        Self {
            aimed: vec![None; chunks],
            intact: Vec::new(),
            intact_share: 1.0,
            progress: (1.0, 0),
        }
    }

    pub fn refresh(&mut self, r: &Range) {
        self.intact = r.intact(RAZE_PIECE);
        self.intact_share = self.intact.len() as f32 / r.chunks.max(1) as f32;
    }

    /// Refresh (every few steps) and say whether the raze is done: the intact share under
    /// [`RAZE_DONE`], stalled for [`RAZE_STALL`], or [`RAZE_SECONDS`] up.
    pub fn finished(&mut self, r: &Range) -> bool {
        if r.frame % 6 == 0 || self.intact.is_empty() {
            self.refresh(r);
        }
        let t = r.seconds();
        if self.intact_share < self.progress.0 - RAZE_PROGRESS {
            self.progress = (self.intact_share, r.frame);
        }
        let stalled = r.frame >= self.progress.1 + (RAZE_STALL / r.dt).round() as u32;
        r.frame > 30 && (self.intact_share < RAZE_DONE || stalled || t >= RAZE_SECONDS)
    }

    /// Guns firing each step `t` seconds in ([`RAZE_GUNS`]).
    pub fn guns(t: f32) -> u32 {
        RAZE_GUNS
            .iter()
            .rev()
            .find(|(after, _)| t >= *after)
            .map_or(1, |g| g.1)
    }

    /// Where the sweep is `t` seconds in: from `b` to `a` and back every [`RAZE_SWEEP`].
    pub fn sweep(t: f32, a: f32, b: f32) -> f32 {
        let phase = (t / RAZE_SWEEP).fract();
        a + (b - a) * (2.0 * phase - 1.0).abs()
    }

    /// Gun `gun`'s chunk this step: an intact chunk not aimed at in the last
    /// [`RAZE_REAIM`], within 1.5 m of the sweep (`off` a pose's distance from it) in a
    /// scattered order, else the nearest to it. Marks it aimed at.
    pub fn pick(
        &mut self,
        r: &Range,
        poses: &[Transform],
        gun: u32,
        off: impl Fn([f32; 3]) -> f32,
    ) -> Option<u32> {
        let frame = r.frame;
        let reaim = (RAZE_REAIM / r.dt).round() as u32;
        let aimed = &self.aimed;
        let free = |c: &u32| aimed[*c as usize].is_none_or(|f| frame >= f + reaim);
        let h =
            |c: u32| (c ^ frame.wrapping_mul(97) ^ gun.wrapping_mul(7919)).wrapping_mul(2654435761);
        let near = self
            .intact
            .iter()
            .copied()
            .filter(free)
            .filter(|&c| off(poses[c as usize].position) < 1.5)
            .min_by_key(|&c| h(c));
        let pick = near.or_else(|| {
            self.intact.iter().copied().filter(free).min_by(|&a, &b| {
                off(poses[a as usize].position).total_cmp(&off(poses[b as usize].position))
            })
        });
        if let Some(c) = pick {
            self.aimed[c as usize] = Some(frame);
        }
        pick
    }
}
