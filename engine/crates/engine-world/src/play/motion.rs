//! The motion demo's scenes (`motion.html`) placed in a [`World`]: the tunnelling gallery
//! (CCD off and on, side by side), the character playground and bullets through walls. One
//! definition ([`MotionSceneDef`]) gives the world's definition ([`world_def`]) and the
//! bodies ([`place`]); the page keeps its input, camera, firing and checks over what
//! [`Placed`] returns.
//!
//! Every value is the page's: the same bodies in the same order, its math in f64 and cast
//! to f32, so a host placing a scene here builds the world the page built.

use crate::*;
use serde::{Deserialize, Serialize};

/// The scenes: name and title.
pub const SCENES: [(&str, &str); 3] = [
    ("ccd", "Tunnelling gallery: CCD off vs on"),
    ("character", "Character playground"),
    ("bullets", "Bullets through walls"),
];

/// A motion scene and the page's switches.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MotionSceneDef {
    pub scene: String,
    /// Speculative velocity (`?lookahead=`, on by default).
    pub look_ahead: bool,
}

impl Default for MotionSceneDef {
    fn default() -> Self {
        Self {
            scene: "ccd".into(),
            look_ahead: true,
        }
    }
}

impl MotionSceneDef {
    fn known(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            SCENES.iter().any(|(s, _)| *s == self.scene),
            "unknown motion scene {}",
            self.scene
        );
        Ok(())
    }
}

/// The world a motion scene needs.
pub fn world_def(def: &MotionSceneDef) -> anyhow::Result<WorldDef> {
    def.known()?;
    Ok(WorldDef {
        gravity: [0.0, -10.0, 0.0],
        contacts: ContactStorage::Canonical,
        capacity: WorldCapacity {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 65536,
            max_candidates: 131072,
            max_joints: 64,
            max_commands: 2048,
            ..WorldCapacity::default()
        },
        ..WorldDef::default()
    })
}

/// A body the page moves back on reset: its start pose and whether it starts asleep.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Reset {
    pub id: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub asleep: bool,
}

/// A lane of the tunnelling gallery: a wall, a control ball and a CCD ball.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Lane {
    pub name: String,
    pub x: f32,
    pub dynamic: bool,
    /// The wall's thickness and its near face (z).
    pub depth: f32,
    pub wall_z: f32,
    pub control: u64,
    pub ccd: u64,
    /// A dynamic wall's bricks in each ball's path.
    pub control_bricks: Vec<u64>,
    pub ccd_bricks: Vec<u64>,
}

/// A debris chunk of the shower and where it is launched from.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Chunk {
    pub body: u64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// The debris shower: 0.1 m chunks onto a slab, CCD off (control) and on.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Shower {
    pub control: Vec<Chunk>,
    pub ccd: Vec<Chunk>,
}

/// A character and its start.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacedCharacter {
    pub h: u64,
    pub start: [f32; 3],
}

/// A walking NPC: its character and its phase on the circle.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Npc {
    pub h: u64,
    pub phase: f32,
}

/// What the page needs of a placed scene.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Placed {
    /// The dynamic and kinematic bodies and their starts, for reset.
    pub resets: Vec<Reset>,
    pub characters: Vec<PlacedCharacter>,
    pub lanes: Vec<Lane>,
    pub ball_radius: f32,
    pub shower: Option<Shower>,
    pub player: Option<u64>,
    pub npcs: Vec<Npc>,
    pub platform: Option<u64>,
    pub lift: Option<u64>,
    pub rockets: Vec<u64>,
    pub walls: Vec<u64>,
    pub targets: Vec<u64>,
}

const IDENT: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
const GROUND: f32 = 0.9;
const PROP: f32 = 0.6;

fn f3(v: [f64; 3]) -> [f32; 3] {
    v.map(|x| x as f32)
}

fn f4(v: [f64; 4]) -> [f32; 4] {
    v.map(|x| x as f32)
}

fn q_axis(ax: [f64; 3], ang: f64) -> [f64; 4] {
    let s = (ang / 2.0).sin();
    [ax[0] * s, ax[1] * s, ax[2] * s, (ang / 2.0).cos()]
}

fn box_i(m: f64, h: [f64; 3]) -> [f64; 3] {
    [
        (m * (h[1] * h[1] + h[2] * h[2])) / 3.0,
        (m * (h[0] * h[0] + h[2] * h[2])) / 3.0,
        (m * (h[0] * h[0] + h[1] * h[1])) / 3.0,
    ]
}

/// The page's placement helpers over a world.
struct Builder<'w> {
    world: &'w mut World,
    out: Placed,
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn body(
        &mut self,
        body_type: BodyType,
        p: [f64; 3],
        q: [f64; 4],
        awake: bool,
        bullet: bool,
        geometry: Geometry,
        friction: f32,
        mass: Option<(f64, [f64; 3])>,
    ) -> anyhow::Result<u64> {
        let def = BodyDef {
            body_type,
            position: f3(p),
            rotation: f4(q),
            is_awake: awake,
            is_bullet: bullet,
            ..BodyDef::default()
        };
        let id = self.world.create_body(&def)?;
        let shape = ShapeDef {
            material: SurfaceMaterial {
                friction,
                ..SurfaceMaterial::default()
            },
            ..ShapeDef::default()
        };
        self.world.create_shape(id, &shape, geometry)?;
        if let Some((m, i)) = mass {
            self.world
                .body_set_mass_data(id, MassData::principal(m as f32, [0.0; 3], f3(i)))?;
        }
        if body_type != BodyType::Static {
            self.out.resets.push(Reset {
                id: id.to_bits(),
                position: def.position,
                rotation: def.rotation,
                asleep: !awake,
            });
        }
        Ok(id.to_bits())
    }

    fn cuboid(half: [f64; 3]) -> Geometry {
        Geometry::Box {
            half: f3(half),
            transform: Transform::IDENTITY,
        }
    }

    fn ball(r: f64) -> Geometry {
        Geometry::Sphere {
            center: [0.0; 3],
            radius: r as f32,
        }
    }

    fn static_box(
        &mut self,
        p: [f64; 3],
        half: [f64; 3],
        q: [f64; 4],
        friction: f32,
    ) -> anyhow::Result<u64> {
        self.body(
            BodyType::Static,
            p,
            q,
            true,
            false,
            Self::cuboid(half),
            friction,
            None,
        )
    }

    fn kinematic_box(&mut self, p: [f64; 3], half: [f64; 3]) -> anyhow::Result<u64> {
        self.body(
            BodyType::Kinematic,
            p,
            IDENT,
            true,
            false,
            Self::cuboid(half),
            GROUND,
            None,
        )
    }

    fn dyn_box(&mut self, p: [f64; 3], half: [f64; 3], mass: f64) -> anyhow::Result<u64> {
        self.body(
            BodyType::Dynamic,
            p,
            IDENT,
            true,
            false,
            Self::cuboid(half),
            PROP,
            Some((mass, box_i(mass, half))),
        )
    }

    /// Parked far below until launched: an asleep dynamic body (CCD when `ccd`).
    fn parked(
        &mut self,
        p: [f64; 3],
        geometry: Geometry,
        mass: f64,
        inertia: [f64; 3],
        ccd: bool,
    ) -> anyhow::Result<u64> {
        self.body(
            BodyType::Dynamic,
            p,
            IDENT,
            false,
            ccd,
            geometry,
            PROP,
            Some((mass, inertia)),
        )
    }

    fn character(&mut self, radius: f32, half_height: f32, p: [f64; 3]) -> anyhow::Result<u64> {
        let def = CharacterDef {
            position: f3(p),
            desc: CharacterDesc {
                radius,
                half_height,
                ..CharacterDesc::default()
            },
            user_data: 0,
        };
        let h = self.world.create_character(&def)?.to_bits();
        self.out.characters.push(PlacedCharacter {
            h,
            start: def.position,
        });
        Ok(h)
    }
}

/// Place scene `def` in `world` (made from [`world_def`], empty): its solver, the ground
/// slab (collider 0) and the scene's bodies and characters.
pub fn place(world: &mut World, def: &MotionSceneDef) -> anyhow::Result<Placed> {
    def.known()?;
    anyhow::ensure!(
        world.body_count() == 0,
        "a motion scene is placed in an empty world"
    );
    let mut solver = world.solver();
    solver.dt = (1.0f64 / 60.0) as f32;
    solver.iterations = 10;
    solver.flags = if def.look_ahead {
        engine_schema::FLAG_SPEC_VELOCITY
    } else {
        0
    };
    world.set_solver(solver);
    let mut b = Builder {
        world,
        out: Placed::default(),
    };
    b.static_box([0.0, -5.0, 0.0], [400.0, 5.0, 400.0], IDENT, GROUND)?;
    match def.scene.as_str() {
        "ccd" => {
            // Each lane: a control ball (no CCD) and a CCD ball, 2 m apart, fired along +z
            // at a wall whose near face is at z = 30.
            const WALL_Z: f64 = 30.0;
            const BALL_R: f64 = 0.15;
            b.out.ball_radius = BALL_R as f32;
            let lanes: [(&str, f64, f64, u8); 5] = [
                ("5 cm slab", -16.0, 0.05, 0),
                ("20 cm slab", -8.0, 0.2, 1),
                ("1 m wall", 0.0, 1.0, 2),
                ("brick wall (grid)", 8.0, 0.2, 3),
                ("brick wall (dynamic)", 16.0, 0.2, 4),
            ];
            for (name, x, depth, build) in lanes {
                let mut bricks: Vec<(u64, f64)> = Vec::new();
                match build {
                    0 => {
                        b.static_box([x, 1.5, WALL_Z + 0.025], [2.5, 1.5, 0.025], IDENT, PROP)?;
                    }
                    1 => {
                        b.static_box([x, 1.5, WALL_Z + 0.1], [2.5, 1.5, 0.1], IDENT, PROP)?;
                    }
                    2 => {
                        b.static_box([x, 1.5, WALL_Z + 0.5], [2.5, 1.5, 0.5], IDENT, PROP)?;
                    }
                    3 => {
                        for iy in 0..15 {
                            for ix in 0..25 {
                                let p = [
                                    x - 2.4 + ix as f64 * 0.2,
                                    0.1 + iy as f64 * 0.2,
                                    WALL_Z + 0.1,
                                ];
                                b.static_box(p, [0.1, 0.1, 0.1], IDENT, PROP)?;
                            }
                        }
                    }
                    _ => {
                        for iy in 0..12 {
                            for ix in 0..20 {
                                let p = [
                                    x - 1.9 + ix as f64 * 0.2,
                                    0.1 + iy as f64 * 0.2,
                                    WALL_Z + 0.1,
                                ];
                                bricks.push((b.dyn_box(p, [0.1, 0.1, 0.1], 4.0)?, p[0]));
                            }
                        }
                    }
                }
                // A dynamic wall: the bricks in each ball's path. A ball that passes the
                // wall without moving them tunnelled; one that knocked them out smashed
                // through.
                let in_path = |dx: f64| -> Vec<u64> {
                    bricks
                        .iter()
                        .filter(|(_, bx)| (bx - (x + dx)).abs() < 0.25)
                        .map(|(id, _)| *id)
                        .collect()
                };
                let (control_bricks, ccd_bricks) = (in_path(-1.0), in_path(1.0));
                let inertia = [1.0 / 250.0; 3];
                let control = b.parked(
                    [x - 1.0, -500.0, 0.0],
                    Builder::ball(BALL_R),
                    5.0,
                    inertia,
                    false,
                )?;
                let ccd = b.parked(
                    [x + 1.0, -500.0, 0.0],
                    Builder::ball(BALL_R),
                    5.0,
                    inertia,
                    true,
                )?;
                b.out.lanes.push(Lane {
                    name: name.into(),
                    x: x as f32,
                    dynamic: build == 4,
                    depth: depth as f32,
                    wall_z: WALL_Z as f32,
                    control,
                    ccd,
                    control_bricks,
                    ccd_bricks,
                });
            }
            // Debris shower: 0.1 m chunks at 40 m/s onto a 0.2 m slab, CCD off (left) and
            // on (right).
            b.static_box([-6.0, 0.1 + 6.0, -14.0], [2.5, 0.1, 2.5], IDENT, GROUND)?;
            b.static_box([6.0, 0.1 + 6.0, -14.0], [2.5, 0.1, 2.5], IDENT, GROUND)?;
            let mut shower = Shower::default();
            for (x0, ccd) in [(-6.0f64, false), (6.0, true)] {
                for i in 0..49u32 {
                    let x = x0 + (i % 7) as f64 * 0.6 - 1.8;
                    let z = -14.0 + (i / 7) as f64 * 0.6 - 1.8;
                    let half = [0.05; 3];
                    let body = b.parked(
                        [x, -600.0, z],
                        Builder::cuboid(half),
                        0.3,
                        box_i(0.3, half),
                        ccd,
                    )?;
                    let chunk = Chunk {
                        body,
                        x: x as f32,
                        y: (9.0 + 0.13 * (i % 5) as f64) as f32,
                        z: z as f32,
                    };
                    if ccd {
                        &mut shower.ccd
                    } else {
                        &mut shower.control
                    }
                    .push(chunk);
                }
            }
            b.out.shower = Some(shower);
        }
        "character" => {
            // Stairs: six 0.3 m steps (0.5 m run) from x = 4.
            for k in 0..6 {
                let top = 0.3 * (k + 1) as f64;
                b.static_box(
                    [4.0 + 0.5 * k as f64 + 3.0, top / 2.0, 6.0],
                    [3.0, top / 2.0, 2.0],
                    IDENT,
                    PROP,
                )?;
            }
            // A walkable 30° ramp and a 55° slide.
            let pi = std::f64::consts::PI;
            b.static_box(
                [-6.0, 1.0, 8.0],
                [3.0, 0.2, 2.0],
                q_axis([0.0, 0.0, 1.0], -pi / 6.0),
                PROP,
            )?;
            b.static_box(
                [-6.0, 2.2, 14.0],
                [2.4, 0.2, 2.0],
                q_axis([0.0, 0.0, 1.0], -55.0 * pi / 180.0),
                PROP,
            )?;
            // A moving platform (kinematic) and a lift.
            b.out.platform = Some(b.kinematic_box([10.0, 0.25, -6.0], [2.0, 0.25, 2.0])?);
            b.out.lift = Some(b.kinematic_box([-10.0, 0.25, -4.0], [1.6, 0.25, 1.6])?);
            b.static_box([-10.0, 2.5, -7.0], [2.0, 2.5, 1.0], IDENT, PROP)?;
            // Light crates (pushed through) and a heavy crate (blocks).
            for i in 0..5 {
                b.dyn_box([2.0 + i as f64 * 0.7, 0.25, -3.0], [0.25; 3], 8.0)?;
            }
            b.dyn_box([2.0, 0.6, -8.0], [0.6; 3], 2500.0)?;
            // A seesaw plank on a pivot.
            b.static_box([-2.0, 0.25, -14.0], [0.15, 0.25, 1.2], IDENT, PROP)?;
            b.dyn_box([-2.0, 0.6, -14.0], [2.5, 0.06, 0.8], 25.0)?;
            // A brick wall for CCD rockets and a debris pile.
            for iy in 0..10 {
                for ix in 0..16 {
                    b.dyn_box(
                        [14.0 + ix as f64 * 0.4 - 3.0, 0.1 + iy as f64 * 0.2, 14.0],
                        [0.2, 0.1, 0.1],
                        6.0,
                    )?;
                }
            }
            for i in 0..120u32 {
                let p = [
                    -12.0 + (i % 6) as f64 * 0.3,
                    0.15 + (i / 36) as f64 * 0.3,
                    4.0 + ((i / 6) % 6) as f64 * 0.3,
                ];
                b.dyn_box(p, [0.12; 3], 2.0)?;
            }
            // The player and three NPCs.
            b.out.player = Some(b.character(0.3, 0.6, [0.0, 1.0, 0.0])?);
            for k in 0..3 {
                let h = b.character(0.28, 0.55, [-4.0 + 2.0 * k as f64, 1.0, -2.0])?;
                b.out.npcs.push(Npc {
                    h,
                    phase: (k as f64 * 2.1) as f32,
                });
            }
            // Rockets: parked CCD balls.
            for i in 0..8 {
                let p = [-3000.0 - i as f64 * 5.0, -3000.0, -3000.0];
                let r = b.parked(p, Builder::ball(0.12), 4.0, [1.0 / 250.0; 3], true)?;
                b.out.rockets.push(r);
            }
        }
        _ => {
            // Five 5 cm walls, then three dynamic targets.
            for k in 0..5 {
                let w = b.static_box(
                    [0.0, 1.5, 10.0 + 2.0 * k as f64 + 0.025],
                    [2.0, 1.5, 0.025],
                    IDENT,
                    PROP,
                )?;
                b.out.walls.push(w);
            }
            for k in 0..3 {
                let t = b.dyn_box([(k as f64 - 1.0) * 1.2, 0.4, 24.0], [0.4; 3], 3.0)?;
                b.out.targets.push(t);
            }
        }
    }
    Ok(b.out)
}
