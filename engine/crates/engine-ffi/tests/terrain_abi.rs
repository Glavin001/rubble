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
        let mut def = rbDefaultTerrainDef(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
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
        let mut def = rbDefaultTerrainDef(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
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
        ok(rbWorld_EnableTerrain(
            w,
            &rbDefaultTerrainDef(RbCoordinates::RB_COORDINATES_UNREAL),
        ));
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

/// Unreal (cm, Z up, A = +X, B = +Y): a strike off centre on a field longer
/// in A than in B. Frames carry the area's live ground as host samples and
/// heights in cm, and the debris in host axes; the bake comes as an event,
/// and the host's copy of the field then holds the last live ground.
#[test]
fn unreal_frames_carry_the_live_surface_debris_and_the_bake() {
    unsafe {
        let w = world(
            RbCoordinates::RB_COORDINATES_UNREAL,
            RbVec3 {
                x: 0.0,
                y: 0.0,
                z: -981.0,
            },
        );
        let mut def = rbDefaultTerrainDef(RbCoordinates::RB_COORDINATES_UNREAL);
        assert!(
            (def.settle_speed - 1.0).abs() < 1e-4,
            "1 cm/s: {}",
            def.settle_speed
        );
        assert!(
            (def.density - 1.6e-3).abs() < 1e-4,
            "kg/cm³: {}",
            def.density
        );
        def.max_particles = 1 << 20;
        def.max_parcels = 1 << 18;
        def.max_substeps = 1 << 13;
        def.cells_per_radius = 8.0;
        def.min_cell_size = 0.2;
        def.region_margin = 0.6;
        def.settle_speed = 5.0;
        def.settle_frames = 3;
        def.enable_surface = true;
        def.surface_patch_samples = 64;
        def.surface_max_debris = 1 << 16;
        ok(rbWorld_EnableTerrain(w, &def));
        let (na, nb) = (97u32, 65u32);
        let cell = 14.7 / 8.0;
        let s = field(w, na, nb, cell, &vec![0.0; (na * nb) as usize]);
        let at = RbVec3 {
            x: 12.0,
            y: -7.0,
            z: 0.0,
        };
        let mut predicted = RbCraterPrediction::default();
        let down = RbVec3 {
            x: 0.0,
            y: 0.0,
            z: -200000.0,
        };
        ok(rbTerrain_PredictImpact(
            w,
            0.5,
            2.7e-3,
            down,
            &mut predicted,
        ));
        let mut crater = RbCraterPrediction::default();
        ok(rbTerrain_Impact(w, s, 0.5, 2.7e-3, down, at, &mut crater));
        assert_eq!(predicted, crater, "the prediction is the strike's");
        assert!((crater.transient_radius - 14.7).abs() < 0.2, "{crater:?}");
        let mut area = RbTerrainArea::default();
        let mut count = 0;
        ok(rbTerrain_GetAreas(w, &mut area, 1, &mut count));
        assert_eq!(count, 1);
        assert!(
            area.lower.x < at.x
                && at.x < area.upper.x
                && area.lower.y < at.y
                && at.y < area.upper.y
        );
        let mut last: Option<(RbTerrainSurface, Vec<f32>)> = None;
        let mut most_debris = 0u32;
        let mut debris_near = true;
        let mut baked = None;
        for _ in 0..400 {
            ok(rbWorld_Step(w));
            let mut f: RbFrame = std::mem::zeroed();
            ok(rbWorld_Poll(w, false, false, &mut f));
            if f.terrain_surface_count > 0 {
                assert_eq!(f.terrain_surface_count, 1);
                assert!(f.has_terrain);
                let v = *f.terrain_surfaces;
                assert_eq!(v.shape, s);
                assert_eq!(
                    [v.a0, v.b0, v.count_a, v.count_b],
                    [area.a0, area.b0, area.count_a, area.count_b],
                    "the area's samples"
                );
                let h = std::slice::from_raw_parts(v.heights, (v.count_a * v.count_b) as usize);
                last = Some((v, h.to_vec()));
                most_debris = most_debris.max(f.terrain_debris_count);
                for k in 0..f.terrain_debris_count as usize {
                    let d = *f.terrain_debris.add(k);
                    let r = (d.position.x - at.x).hypot(d.position.y - at.y);
                    debris_near &= r < 3.0 * crater.final_rim_radius && d.volume > 0.0;
                }
            }
            if f.terrain_bake_count > 0 {
                baked = Some(*f.terrain_bakes);
                break;
            }
        }
        let b = baked.expect("the area baked");
        assert_eq!((b.shape, b.slot), (s, area.slot));
        assert!(most_debris > 0, "the excavation flow as debris");
        assert!(
            debris_near,
            "debris round the strike, in host axes and units"
        );
        let (v, live) = last.expect("frames carried the surface");
        // The host's copy of the field holds the last live ground there.
        let mut heights = vec![0.0f32; (na * nb) as usize];
        ok(rbHeightField_GetHeights(w, s, na, nb, heights.as_mut_ptr()));
        let mut worst = 0.0f32;
        let mut deepest = 0.0f32;
        for b in 0..v.count_b {
            for a in 0..v.count_a {
                let h = live[(b * v.count_a + a) as usize];
                let g = heights[((v.b0 + b) * na + v.a0 + a) as usize];
                worst = worst.max((h - g).abs());
                deepest = deepest.min(g);
            }
        }
        eprintln!(
            "window a {}+{} b {}+{}: last live ground vs the bake {worst:.3} cm; crater {deepest:.2} cm; \
             debris at most {most_debris}",
            v.a0, v.count_a, v.b0, v.count_b
        );
        assert!(deepest < -3.0, "a crater, in cm");
        // The deepest sample sits where the strike was: host axes.
        let mut k_min = 0;
        for k in 0..heights.len() {
            if heights[k] < heights[k_min] {
                k_min = k;
            }
        }
        let (a, b) = ((k_min as u32 % na) as f32, (k_min as u32 / na) as f32);
        let (x, y) = (
            (a - 0.5 * (na - 1) as f32) * cell,
            (b - 0.5 * (nb - 1) as f32) * cell,
        );
        assert!(
            (x - at.x).hypot(y - at.y) < 0.5 * crater.transient_radius,
            "deepest at ({x}, {y})"
        );
        assert!(worst < 0.1 * cell, "{worst} cm");
        rbDestroyWorld(w);
    }
}

#[test]
fn close_area_bakes_one_area_while_the_other_stays_open() {
    unsafe {
        let w = world(
            RbCoordinates::RB_COORDINATES_Y_UP_METRES,
            RbVec3 {
                x: 0.0,
                y: -9.81,
                z: 0.0,
            },
        );
        let mut def = rbDefaultTerrainDef(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        def.max_particles = 1 << 20;
        def.max_parcels = 1 << 18;
        def.max_substeps = 1 << 13;
        def.cells_per_radius = 8.0;
        def.min_cell_size = 0.002;
        def.region_margin = 0.6;
        def.settle_speed = 0.05;
        def.settle_frames = 3;
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
        let mut areas = [RbTerrainArea::default(); 4];
        let mut count = 0;
        ok(rbTerrain_GetAreas(w, areas.as_mut_ptr(), 4, &mut count));
        assert_eq!(count, 2);
        assert_ne!(areas[0].slot, areas[1].slot);
        let left = *areas[..2]
            .iter()
            .find(|a| a.upper.x < 0.0)
            .expect("one on the left");
        let right = *areas[..2]
            .iter()
            .find(|a| a.lower.x > 0.0)
            .expect("one on the right");
        assert!(left.count_a > 0 && left.count_b > 0 && left.a0 + left.count_a <= nx);
        ok(rbWorld_Step(w));
        ok(rbTerrain_CloseArea(w, left.slot));
        ok(rbTerrain_GetAreas(w, areas.as_mut_ptr(), 4, &mut count));
        assert_eq!(count, 1);
        assert_eq!(areas[0].slot, right.slot);
        assert!(areas[0].age > 0.0);
        assert_eq!(
            rbTerrain_CloseArea(w, left.slot),
            RbResult::RB_ERROR,
            "already closed"
        );
        ok(rbWorld_Step(w));
        let mut f: RbFrame = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        let bakes = std::slice::from_raw_parts(f.terrain_bakes, f.terrain_bake_count as usize);
        assert_eq!(bakes.len(), 1);
        assert_eq!((bakes[0].shape, bakes[0].slot), (s, left.slot));
        assert!(f.has_terrain, "the other area still steps");
        rbDestroyWorld(w);
    }
}
