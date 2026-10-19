//! The lab's scene catalogue (the Scene menu of both hosts) and how a sample
//! is built: one house (baked level or procedural), optionally bond-preloaded
//! by the host, then laid out as a row, grid or suburb.

use crate::asset::FractureAsset;
use crate::cuboid::{build_cuboid_house, CuboidOptions};
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
    /// Scene menu entry.
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

pub const SAMPLES: &[Sample] = &[
    Sample {
        key: "house",
        menu: "House · 131 building parts",
        label: "house",
        file: Some("veneer-house.glb"),
        layout: None,
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "row",
        menu: "20 houses · Level 3 in a row",
        label: "20 houses",
        file: Some("veneer-house-hull16.glb"),
        layout: Some(Layout::grid(1, 20)),
        level: Some(3),
        max_hull_vertices: Some(16),
    },
    Sample {
        key: "grid",
        menu: "49 houses · Level 2 in a 7×7 grid",
        label: "49 houses",
        file: Some("veneer-house.glb"),
        layout: Some(Layout::grid(7, 7)),
        level: Some(2),
        max_hull_vertices: None,
    },
    Sample {
        key: "suburb",
        menu: "Suburb · 48 houses on lots with roads",
        label: "48 houses",
        file: Some("veneer-house.glb"),
        layout: Some(Layout::SUBURB),
        level: Some(2),
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboid",
        menu: "Cuboid house · 1 box-built house",
        label: "cuboid house",
        file: None,
        layout: None,
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboids",
        menu: "Cuboid suburb · 48 box-built houses",
        label: "48 cuboid houses",
        file: None,
        layout: Some(Layout::SUBURB),
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboids2x",
        menu: "Large cuboid suburb · 96 box-built houses",
        label: "96 cuboid houses",
        file: None,
        layout: Some(Layout::suburb_blocks(6, 4)),
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "cuboidGrid",
        menu: "49 box-built houses in a 7×7 grid",
        label: "49 cuboid houses",
        file: None,
        layout: Some(Layout::grid(7, 7)),
        level: None,
        max_hull_vertices: None,
    },
    Sample {
        key: "arch",
        menu: "Masonry arch · 23 pieces",
        label: "arch",
        file: Some("masonry-arch.glb"),
        layout: None,
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
