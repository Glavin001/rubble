//! Deformable terrain through the C ABI: an impact digs a crater that bakes
//! into the heightfield the host reads back, and heights cross in the host's
//! order and units (Unreal: centimetres, A/B axes swapped).

use rubble_ffi::coords::RbCoordinates;
use rubble_ffi::*;
use std::ffi::CStr;
use std::ptr;

fn ok(r: RbResult) {
    if r != RbResult::RB_OK {
        let msg = unsafe { CStr::from_ptr(rbGetLastError()) };
        panic!("{r:?}: {}", msg.to_string_lossy());
    }
}

unsafe fn world(coordinates: RbCoordinates, gravity: RbVec3) -> *mut RbWorld {
    let mut def = rbDefaultWorldDef();
    def.coordinates = coordinates;
    def.gravity = gravity;
    let mut w = ptr::null_mut();
    ok(rbCreateWorld(&def, &mut w));
    w
}

unsafe fn field(w: *mut RbWorld, a: u32, b: u32, cell: f32, heights: &[f32]) -> RbShapeId {
    let mut g = 0;
    ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
    let mut s = 0;
    ok(rbCreateHeightFieldShape(
        w,
        g,
        &rbDefaultShapeDef(),
        a,
        b,
        cell,
        cell,
        heights.as_ptr(),
        RbTransform::default(),
        &mut s,
    ));
    s
}

#[test]
fn an_impact_bakes_a_crater_the_host_reads_back() {
    unsafe {
        let w = world(
            RbCoordinates::RB_COORDINATES_Y_UP_METRES,
            RbVec3 {
                x: 0.0,
                y: -9.81,
                z: 0.0,
            },
        );
        let mut def = rbDefaultTerrainDef();
        def.max_particles = 1 << 20;
        def.max_parcels = 1 << 18;
        def.max_substeps = 1 << 13;
        def.cells_per_radius = 8.0;
        def.min_cell_size = 0.002;
        def.region_margin = 0.6;
        def.settle_speed = 0.05;
        def.settle_frames = 3;
        ok(rbWorld_EnableTerrain(w, &def));
        let n = 97u32;
        let s = field(w, n, n, 0.147 / 8.0, &vec![0.0; (n * n) as usize]);
        let mut crater = RbCraterPrediction::default();
        let zero = RbVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        ok(rbTerrain_Impact(
            w,
            s,
            0.005,
            2700.0,
            RbVec3 {
                x: 0.0,
                y: -2000.0,
                z: 0.0,
            },
            zero,
            &mut crater,
        ));
        assert!(
            (crater.transient_radius - 0.147).abs() < 0.002,
            "{crater:?}"
        );
        let mut open = true;
        let mut frames = 0;
        while open {
            assert!(frames < 240, "the region never settled");
            ok(rbWorld_Step(w));
            let mut f = std::mem::zeroed();
            ok(rbWorld_Poll(w, false, false, &mut f));
            ok(rbTerrain_IsOpen(w, &mut open));
            frames += 1;
        }
        // The bake's heights reach the host's copy with a later frame.
        ok(rbWorld_Step(w));
        let mut f = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        let mut heights = vec![0.0f32; (n * n) as usize];
        ok(rbHeightField_GetHeights(w, s, n, n, heights.as_mut_ptr()));
        let centre = heights[(n / 2 * n + n / 2) as usize];
        assert!(
            centre < -0.25 * crater.transient_depth,
            "centre at {centre}; transient depth {}",
            crater.transient_depth
        );
        let mut missed = 1u64;
        ok(rbTerrain_MissedStrikes(w, &mut missed));
        assert_eq!(missed, 0);
        rbDestroyWorld(w);
    }
}

#[test]
fn two_impacts_far_apart_open_two_areas_that_bake_on_their_own() {
    unsafe {
        let w = world(
            RbCoordinates::RB_COORDINATES_Y_UP_METRES,
            RbVec3 {
                x: 0.0,
                y: -9.81,
                z: 0.0,
            },
        );
        let mut def = rbDefaultTerrainDef();
        def.max_particles = 1 << 20;
        def.max_parcels = 1 << 18;
        def.max_substeps = 1 << 13;
        def.cells_per_radius = 8.0;
        def.min_cell_size = 0.002;
        def.region_margin = 0.6;
        def.settle_speed = 0.05;
        def.settle_frames = 3;
        assert_eq!(def.max_regions, 4);
        ok(rbWorld_EnableTerrain(w, &def));
        let (nx, nz) = (129u32, 65u32);
        let s = field(w, nx, nz, 0.147 / 8.0, &vec![0.0; (nx * nz) as usize]);
        for x in [-0.6f32, 0.6] {
            ok(rbTerrain_Impact(
                w,
                s,
                0.005,
                2700.0,
                RbVec3 {
                    x: 0.0,
                    y: -2000.0,
                    z: 0.0,
                },
                RbVec3 { x, y: 0.0, z: 0.0 },
                std::ptr::null_mut(),
            ));
        }
        let mut areas = 0u32;
        ok(rbTerrain_AreaCount(w, &mut areas));
        assert_eq!(areas, 2);
        let mut frames = 0;
        while areas > 0 {
            assert!(frames < 300, "the areas never settled");
            ok(rbWorld_Step(w));
            let mut f = std::mem::zeroed();
            ok(rbWorld_Poll(w, false, false, &mut f));
            ok(rbTerrain_AreaCount(w, &mut areas));
            frames += 1;
        }
        ok(rbWorld_Step(w));
        let mut f = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        let mut heights = vec![0.0f32; (nx * nz) as usize];
        ok(rbHeightField_GetHeights(w, s, nx, nz, heights.as_mut_ptr()));
        // Two craters, alike.
        let at = |x: f32| {
            let i = ((x / (0.147 / 8.0)) + 0.5 * (nx - 1) as f32).round() as u32;
            heights[(nz / 2 * nx + i) as usize]
        };
        let (a, b) = (at(-0.6), at(0.6));
        assert!(a < -0.01 && b < -0.01, "{a} {b}");
        assert!((a - b).abs() < 0.1 * a.abs(), "{a} {b}");
        rbDestroyWorld(w);
    }
}

#[test]
fn heights_come_back_in_host_order_and_units() {
    unsafe {
        let w = world(
            RbCoordinates::RB_COORDINATES_UNREAL,
            RbVec3 {
                x: 0.0,
                y: 0.0,
                z: -981.0,
            },
        );
        ok(rbWorld_EnableTerrain(w, &rbDefaultTerrainDef()));
        let (a, b) = (5u32, 3u32);
        let host: Vec<f32> = (0..a * b).map(|k| 10.0 * k as f32).collect();
        let s = field(w, a, b, 100.0, &host);
        let mut back = vec![0.0f32; (a * b) as usize];
        ok(rbHeightField_GetHeights(w, s, a, b, back.as_mut_ptr()));
        for (got, want) in back.iter().zip(&host) {
            assert!((got - want).abs() < 1e-3, "{back:?} vs {host:?}");
        }
        assert_eq!(
            rbHeightField_GetHeights(w, s, b, a, back.as_mut_ptr()),
            RbResult::RB_INVALID_ARGUMENT
        );
        rbDestroyWorld(w);
    }
}
