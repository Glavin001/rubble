//! Shared demo scene definitions for the Rubble physics engine.
//!
//! Both the native `rubble-viewer` examples and the `rubble-wasm` web bindings
//! consume scene builders from this crate so the two viewers stay in sync.

pub mod scenes_2d;
pub mod scenes_3d;

use rubble2d::RigidBodyDesc2D;
use rubble3d::RigidBodyDesc;

/// A named 3D demo scene.
pub struct Scene3D {
    pub name: &'static str,
    pub build: fn() -> Vec<RigidBodyDesc>,
}

/// A named 2D demo scene.
pub struct Scene2D {
    pub name: &'static str,
    pub build: fn() -> Vec<RigidBodyDesc2D>,
}

/// Default scene that viewers should select on startup.
pub const INITIAL_SCENE_3D: &str = "Pyramid";
/// Default scene that viewers should select on startup.
pub const INITIAL_SCENE_2D: &str = "Pyramid";

/// Registry of every 3D demo scene, in display order.
pub fn scenes_3d() -> &'static [Scene3D] {
    &SCENES_3D
}

/// Registry of every 2D demo scene, in display order.
pub fn scenes_2d() -> &'static [Scene2D] {
    &SCENES_2D
}

static SCENES_3D: [Scene3D; 14] = [
    Scene3D {
        name: "Empty",
        build: scenes_3d::scene_empty,
    },
    Scene3D {
        name: "Ground",
        build: scenes_3d::scene_ground,
    },
    Scene3D {
        name: "Dynamic Friction",
        build: scenes_3d::scene_dynamic_friction,
    },
    Scene3D {
        name: "Static Friction",
        build: scenes_3d::scene_static_friction,
    },
    Scene3D {
        name: "Pyramid",
        build: scenes_3d::scene_pyramid,
    },
    Scene3D {
        name: "Stack",
        build: scenes_3d::scene_stack,
    },
    Scene3D {
        name: "Stack Ratio",
        build: scenes_3d::scene_stack_ratio,
    },
    Scene3D {
        name: "Scatter",
        build: scenes_3d::scene_scatter,
    },
    Scene3D {
        name: "Scatter Boxes",
        build: scenes_3d::scene_scatter_boxes,
    },
    Scene3D {
        name: "Grid Boxes",
        build: scenes_3d::scene_grid_boxes,
    },
    Scene3D {
        name: "Slanted Grid",
        build: scenes_3d::scene_slanted_grid_boxes,
    },
    Scene3D {
        name: "10k Grid",
        build: scenes_3d::scene_grid_10k_boxes,
    },
    Scene3D {
        name: "Stress Mixed",
        build: scenes_3d::scene_stress_20k_mixed,
    },
    Scene3D {
        name: "100k Grid",
        build: scenes_3d::scene_grid_100k_boxes,
    },
];

static SCENES_2D: [Scene2D; 9] = [
    Scene2D {
        name: "Empty",
        build: scenes_2d::scene_empty,
    },
    Scene2D {
        name: "Ground",
        build: scenes_2d::scene_ground,
    },
    Scene2D {
        name: "Dynamic Friction",
        build: scenes_2d::scene_dynamic_friction,
    },
    Scene2D {
        name: "Static Friction",
        build: scenes_2d::scene_static_friction,
    },
    Scene2D {
        name: "Pyramid",
        build: scenes_2d::scene_pyramid,
    },
    Scene2D {
        name: "Cards",
        build: scenes_2d::scene_cards,
    },
    Scene2D {
        name: "Stack",
        build: scenes_2d::scene_stack,
    },
    Scene2D {
        name: "Stack Ratio",
        build: scenes_2d::scene_stack_ratio,
    },
    Scene2D {
        name: "Scatter",
        build: scenes_2d::scene_scatter,
    },
];
