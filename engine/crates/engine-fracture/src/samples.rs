//! The lab's scene catalogue (the Scene menu of both hosts) and how a sample
//! is built: one house (baked level or procedural), optionally bond-preloaded
//! by the host, then laid out as a row, grid or suburb.

use crate::asset::FractureAsset;
use crate::cuboid::{build_cuboid_house, cuboid_pieces, cuboid_stress_json, CuboidOptions};
use crate::scene::*;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// How copies of one house are placed.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Layout {
    #[serde(default)]
    pub suburb: bool,
    #[serde(default = "one")]
    pub columns: u32,
    #[serde(default = "one")]
    pub rows: u32,
}

fn one() -> u32 {
    1
}

impl Layout {
    pub const fn grid(columns: u32, rows: u32) -> Self {
        Self {
            suburb: false,
            columns,
            rows,
        }
    }
    pub const SUBURB: Self = Self {
        suburb: true,
        columns: 1,
        rows: 1,
    };
    /// A suburb of `blocks_x x blocks_z` blocks (four lots each); the default
    /// suburb (`SUBURB`) is 4 x 3.
    pub const fn suburb_blocks(blocks_x: u32, blocks_z: u32) -> Self {
        Self {
            suburb: true,
            columns: blocks_x,
            rows: blocks_z,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sample {
    pub key: &'static str,
    /// Scene menu group (`GROUPS`, in menu order).
    pub group: &'static str,
    /// Scene menu entry: what, then how many.
    pub menu: &'static str,
    /// Short name for status lines.
    pub label: &'static str,
    /// Bundled GLB (`fracture/assets/`), or None for the procedural cuboid house.
    pub file: Option<&'static str>,
    pub layout: Option<Layout>,
    /// Preferred level (falls back to the first level present).
    pub level: Option<u8>,
    /// Engine-wide hull vertex limit for this asset (a larger hull is a load error).
    pub max_hull_vertices: Option<usize>,
}

impl Sample {
    pub fn procedural(&self) -> bool {
        self.file.is_none()
    }
}

/// The Scene menu's groups, in order (the pack samples' last).
pub const GROUPS: &[&str] = &[
    "One structure",
    "Rows and grids",
    "Suburbs (lots and roads)",
    WRECKING_YARD,
    BRIDGE_OUT,
    TOPPLE,
    RAM_RAID,
    FLY_THROUGH,
];

/// The Scene menu group of the Wrecking Yard structures.
pub const WRECKING_YARD: &str = "Wrecking Yard";
/// The Scene menu group of the vehicle's target (`stress_ref::buildings`).
pub const RAM_RAID: &str = "Ram Raid";
/// The Scene menu group of the aircraft's targets (`stress_ref::buildings`).
pub const FLY_THROUGH: &str = "Fly-through";

/// The Scene menu group of the bridge-failure structures (`stress_ref::viaduct`).
pub const BRIDGE_OUT: &str = "Bridge Out";

/// The Scene menu groups of the pack samples, in menu order (the last of `GROUPS`).
pub const PACK_GROUPS: &[&str] = &[WRECKING_YARD, BRIDGE_OUT, TOPPLE, RAM_RAID, FLY_THROUGH];

/// The Scene menu group of the structures brought down by their demolition charges
/// (`stress_ref::topple`; the pack's `defaults.charges`).
pub const TOPPLE: &str = "Topple";

/// A Scene menu entry built in Rust as a scene pack (`stress_ref::wrecking`'s structures,
/// with `stress_ref::buildings`' for the Ram Raid and Fly-through groups): the hosts
/// load its JSON (`pack_sample_json`) as they load any pack (`pack::build_pack_scene`,
/// stress-web `stress_set_pack`).
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PackSample {
    /// Also the pack's key (`stress_ref::wrecking::STRUCTURES`).
    pub key: &'static str,
    /// Scene menu group (`GROUPS`).
    pub group: &'static str,
    pub menu: &'static str,
    pub label: &'static str,
}

/// The Scene menu's pack samples, in menu order (after `SAMPLES`).
pub const PACK_SAMPLES: &[PackSample] = &[
    PackSample {
        key: "wy_frame10",
        group: WRECKING_YARD,
        menu: "10-floor concrete frame · 1,264 chunks",
        label: "10-floor frame",
    },
    PackSample {
        key: "wy_round_tower",
        group: WRECKING_YARD,
        menu: "Round tower on columns · 337 chunks",
        label: "round tower",
    },
    PackSample {
        key: "wy_brick_tower",
        group: WRECKING_YARD,
        menu: "Brick tower · 1,444 chunks",
        label: "brick tower",
    },
    PackSample {
        key: "bo_viaduct",
        group: BRIDGE_OUT,
        menu: "Highway viaduct, four spans · 347 chunks",
        label: "viaduct",
    },
    PackSample {
        key: "topple_tower",
        group: TOPPLE,
        menu: "High-rise beside a street, with charges · 425 chunks",
        label: "topple tower",
    },
    PackSample {
        key: "rr_storefront_row",
        group: RAM_RAID,
        menu: "Storefront row (5 shops) · 1,203 chunks",
        label: "storefront row",
    },
    PackSample {
        key: "ft_apartment_block",
        group: FLY_THROUGH,
        menu: "7-storey apartment block · 1,355 chunks",
        label: "apartment block",
    },
    PackSample {
        key: "ft_glass_tower",
        group: FLY_THROUGH,
        menu: "11-storey glass tower · 1,332 chunks",
        label: "glass tower",
    },
];

/// A pack sample's scene pack JSON.
pub fn pack_sample_json(key: &str) -> Result<Vec<u8>> {
    if !PACK_SAMPLES.iter().any(|s| s.key == key) {
        bail!("Unknown bundled scene");
    }
    let pack =
        stress_ref::wrecking::pack(key).ok_or_else(|| anyhow::anyhow!("no structure {key}"))?;
    Ok(serde_json::to_vec(&pack)?)
}

/// The Scene menu, grouped (`GROUPS`) and in menu order.
pub const SAMPLES: &[Sample] = &[
    Sample {
        key: "house",
        group: "One structure",
        menu: "Veneer house · 131 parts",
        label: "house",
        file: Some("veneer-house.glb"),
        layout: None,
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboid",
        group: "One structure",
        menu: "Cuboid house · 715 boxes",
        label: "cuboid house",
        file: None,
        layout: None,
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "arch",
        group: "One structure",
        menu: "Masonry arch · 23 stones",
        label: "arch",
        file: Some("masonry-arch.glb"),
        layout: None,
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "row",
        group: "Rows and grids",
        menu: "Veneer houses · row of 20 (level 3)",
        label: "20 houses",
        file: Some("veneer-house-hull16.glb"),
        layout: Some(Layout::grid(1, 20)),
        level: Some(3),
        max_hull_vertices: Some(16),
    },
    Sample {
        key: "grid",
        group: "Rows and grids",
        menu: "Veneer houses · 7×7 grid of 49 (level 2)",
        label: "49 houses",
        file: Some("veneer-house.glb"),
        layout: Some(Layout::grid(7, 7)),
        level: Some(2),
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboidGrid",
        group: "Rows and grids",
        menu: "Cuboid houses · 7×7 grid of 49",
        label: "49 cuboid houses",
        file: None,
        layout: Some(Layout::grid(7, 7)),
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "suburb",
        group: "Suburbs (lots and roads)",
        menu: "Veneer houses · suburb of 48 (level 2)",
        label: "48 houses",
        file: Some("veneer-house.glb"),
        layout: Some(Layout::SUBURB),
        level: Some(2),
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboids",
        group: "Suburbs (lots and roads)",
        menu: "Cuboid houses · suburb of 48",
        label: "48 cuboid houses",
        file: None,
        layout: Some(Layout::SUBURB),
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboids2x",
        group: "Suburbs (lots and roads)",
        menu: "Cuboid houses · large suburb of 96",
        label: "96 cuboid houses",
        file: None,
        layout: Some(Layout::suburb_blocks(6, 4)),
        level: None,
        max_hull_vertices: None,
    },
];

pub fn sample(key: &str) -> Result<&'static Sample> {
    match SAMPLES.iter().find(|s| s.key == key) {
        Some(s) => Ok(s),
        None => bail!("Unknown bundled scene"),
    }
}

/// The level a sample opens at: its preferred level if present, else the first.
pub fn initial_level(asset: &FractureAsset, preferred: Option<u8>) -> Option<u8> {
    let levels = asset.levels();
    preferred
        .filter(|l| levels.contains(l))
        .or(levels.first().copied())
}

/// One house: a baked level, or the procedural cuboid house when `asset` is None.
pub fn base_scene(
    asset: Option<&FractureAsset>,
    level: u8,
    max_hull_vertices: Option<usize>,
) -> Result<Scene> {
    match asset {
        Some(asset) => build_scene(asset, level, max_hull_vertices),
        None => build_cuboid_house(&CuboidOptions::default()),
    }
}

/// Lay out copies of `one` (already bond-preloaded, or without joints).
pub fn layout_scene(one: Scene, layout: Option<Layout>, procedural: bool) -> Scene {
    let mut s = match layout {
        None => one,
        // A suburb's columns and rows are its blocks (1 x 1: the default 4 x 3).
        Some(l) if l.suburb && (l.columns, l.rows) != (1, 1) => build_suburb(
            &one,
            &SuburbOptions {
                blocks_x: l.columns,
                blocks_z: l.rows,
                ..SuburbOptions::default()
            },
        ),
        Some(l) if l.suburb => build_suburb(&one, &SuburbOptions::default()),
        Some(l) => build_grid(
            &one,
            &GridOptions {
                columns: l.columns,
                rows: l.rows,
                ..GridOptions::default()
            },
        ),
    };
    if procedural {
        // A brick in a wall touches up to six neighbours (four contacts each).
        s.capacity.candidates_per_collider = 12;
        s.capacity.contacts_per_collider = 24;
    }
    s
}

/// Cuboid houses laid out as `layout` (`layout_scene`): the lab's scene (its camera,
/// projectile and placement) and the same houses as the stress oracle's input
/// (`cuboid_stress_json`), one per copy at its laid-out offset.
pub fn cuboid_stress_scene(layout: Option<Layout>) -> Result<(Scene, serde_json::Value)> {
    let o = CuboidOptions::default();
    let one = build_cuboid_house(&o)?;
    let lab = layout_scene(one.clone(), layout, true);
    let stride = lab.instance_stride.max(1) as usize;
    let first = one.bodies[1].p;
    let offsets: Vec<[f64; 3]> = (0..lab.copies.max(1) as usize)
        .map(|k| {
            let p = lab.bodies[1 + k * stride].p;
            [p[0] - first[0], p[1] - first[1], p[2] - first[2]]
        })
        .collect();
    Ok((lab, cuboid_stress_json(&cuboid_pieces(&o), &offsets)))
}
