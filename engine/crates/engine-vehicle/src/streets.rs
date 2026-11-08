//! The vehicle demo's street and range layouts: the Gauntlet's street of masonry facades,
//! the meteor run's longer one and the meteor range's target pads. Pure data, shared by
//! the scene that places them (`rubble::play::vehicles`) and the page scripts that shell,
//! steer and judge them (`engine-vehicle-wasm`).

/// The Gauntlet's street (street frame: the road runs along +Z, centred on x = 0; facades
/// on both sides with static building cores behind them; the ground is y = 0).
pub mod gauntlet {
    use serde::Serialize;

    /// Half the street's width to the facades' street faces (m).
    pub const STREET_HALF: f32 = 8.0;
    /// Facade block half extents (x thickness, y height, z length) and mass (kg).
    pub const BLOCK_HALF: [f32; 3] = [0.25, 0.25, 0.4];
    pub const BLOCK_MASS: f32 = 150.0;
    /// Rows of blocks per facade (12 m).
    pub const ROWS: u32 = 24;
    /// Building spans along the street (z from, z to), the same on both sides.
    pub const SPANS: [(f32, f32); 4] = [(0.0, 30.0), (40.0, 70.0), (80.0, 110.0), (120.0, 150.0)];
    /// Where the player starts (and heads +Z).
    pub const SPAWN: [f32; 3] = [-2.5, 3.5, -25.0];

    #[derive(Clone, Copy, Debug, Serialize)]
    pub struct Block {
        pub p: [f32; 3],
        pub half: [f32; 3],
        pub mass: f32,
    }

    #[derive(Clone, Copy, Debug, Serialize)]
    pub struct Parked {
        pub p: [f32; 3],
        pub yaw: f32,
    }

    #[derive(Clone, Debug, Serialize)]
    pub struct Layout {
        pub spawn: [f32; 3],
        pub blocks: Vec<Block>,
        /// Static building cores behind the facades.
        pub cores: Vec<Block>,
        pub parked: Vec<Parked>,
    }

    /// Parked cars at the kerbs (x, z, yaw).
    pub const PARKED: [(f32, f32, f32); 6] = [
        (-5.6, 22.0, 0.0),
        (5.6, 48.0, std::f32::consts::PI),
        (-5.6, 75.0, 0.0),
        (5.6, 96.0, std::f32::consts::PI),
        (-5.6, 128.0, 0.0),
        (5.6, 140.0, std::f32::consts::PI),
    ];

    /// The street: facade blocks in running bond, cores and parked cars.
    pub fn layout() -> Layout {
        layout_of(&SPANS, &PARKED)
    }

    /// A street of buildings over `spans` with cars parked at `parked`.
    pub fn layout_of(spans: &[(f32, f32)], parked: &[(f32, f32, f32)]) -> Layout {
        let [hx, hy, hz] = BLOCK_HALF;
        let mut blocks = Vec::new();
        let mut cores = Vec::new();
        for side in [-1.0f32, 1.0] {
            let x = side * (STREET_HALF + hx);
            for &(z0, z1) in spans {
                for row in 0..ROWS {
                    // Odd rows shift half a block and hold one fewer.
                    let shift = if row % 2 == 1 { hz } else { 0.0 };
                    let n = ((z1 - z0) / (2.0 * hz)).floor() as u32 - (row % 2);
                    for k in 0..n {
                        let z = z0 + shift + hz + 2.0 * hz * k as f32;
                        // 1 mm gaps so the stack starts without overlap.
                        let y = hy + row as f32 * (2.0 * hy + 0.001);
                        blocks.push(Block {
                            p: [x, y, z],
                            half: BLOCK_HALF,
                            mass: BLOCK_MASS,
                        });
                    }
                }
                let height = ROWS as f32 * 2.0 * hy;
                let depth = 8.0;
                cores.push(Block {
                    p: [
                        side * (STREET_HALF + 2.0 * hx + 0.01 + depth),
                        height / 2.0,
                        (z0 + z1) / 2.0,
                    ],
                    half: [depth, height / 2.0, (z1 - z0) / 2.0],
                    mass: 0.0,
                });
            }
        }
        Layout {
            spawn: SPAWN,
            blocks,
            cores,
            parked: parked
                .iter()
                .map(|&(x, z, yaw)| Parked {
                    p: [x, 2.2, z],
                    yaw,
                })
                .collect(),
        }
    }
}

/// The meteor run: the Gauntlet's street, twice as long, with a finish line.
pub mod run {
    use super::gauntlet;

    /// Building spans along the street (z from, z to), both sides: 310 m.
    pub const SPANS: [(f32, f32); 8] = [
        (0.0, 30.0),
        (40.0, 70.0),
        (80.0, 110.0),
        (120.0, 150.0),
        (160.0, 190.0),
        (200.0, 230.0),
        (240.0, 270.0),
        (280.0, 310.0),
    ];
    /// Parked cars at the kerbs (x, z, yaw).
    pub const PARKED: [(f32, f32, f32); 10] = [
        (-5.6, 22.0, 0.0),
        (5.6, 48.0, std::f32::consts::PI),
        (-5.6, 75.0, 0.0),
        (5.6, 96.0, std::f32::consts::PI),
        (-5.6, 128.0, 0.0),
        (5.6, 165.0, std::f32::consts::PI),
        (-5.6, 205.0, 0.0),
        (5.6, 238.0, std::f32::consts::PI),
        (-5.6, 262.0, 0.0),
        (5.6, 296.0, std::f32::consts::PI),
    ];
    /// The finish line (z): the end of the street.
    pub const FINISH: f32 = 315.0;
    /// Meteor radius (m), mass (kg) and how many the page keeps parked for launch.
    pub const METEOR_RADIUS: f32 = 0.8;
    pub const METEOR_MASS: f32 = 2500.0;
    pub const METEORS: u32 = 12;

    /// The street.
    pub fn layout() -> gauntlet::Layout {
        gauntlet::layout_of(&SPANS, &PARKED)
    }
}

/// The meteor range: two lanes of target pads (steered, unsteered).
pub mod meteors {
    use serde::Serialize;

    /// Lane centres (x): steered, then unsteered (the steered lane is on the
    /// left of a camera looking down the range, +Z).
    pub const LANES: [f32; 2] = [15.0, -15.0];
    /// Pad centres within a lane (x offset, z) and their half size (m).
    pub const PADS: [[f32; 2]; 3] = [[-4.0, 10.0], [4.0, 18.0], [0.0, 28.0]];
    pub const PAD_HALF: f32 = 1.0;
    /// Meteor radius (m) and mass (kg).
    pub const RADIUS: f32 = 0.5;
    pub const MASS: f32 = 500.0;

    #[derive(Clone, Copy, Debug, Serialize)]
    pub struct Pad {
        pub p: [f32; 3],
        pub half: [f32; 3],
        pub lane: u32,
    }

    /// Pad centres on the ground, both lanes.
    pub fn pads() -> Vec<Pad> {
        LANES
            .iter()
            .enumerate()
            .flat_map(|(l, &x)| {
                PADS.iter().map(move |p| Pad {
                    p: [x + p[0], 0.0, p[1]],
                    half: [PAD_HALF, 0.02, PAD_HALF],
                    lane: l as u32,
                })
            })
            .collect()
    }
}
