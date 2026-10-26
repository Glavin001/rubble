//! Support loss in the world API (`DestructibleDef::support_loss`, threshold model): a slab on
//! a row of columns, and a block on three columns, built as scene packs with the bond
//! strengths and masses set exactly. One column is shot away (its bonds are weak in tension
//! and shear, so the strike breaks only them); what the rest does is support loss's call.
#![cfg(feature = "stress")]

use rubble::*;

#[path = "support/frames.rs"]
mod frames;
use frames::{frame, Frame, AREA, G};

/// The capacity the threshold tests give a scene (every split's piece a body).
fn capacity(asset: &DestructibleAssetDesc) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

struct Run {
    world: World,
    id: DestructibleId,
    asset: DestructibleAssetDesc,
    broken: usize,
    /// The ground plane's height (world frame).
    ground: f32,
    /// The top chunk's body transform after every frame.
    trace: Vec<(glam::Vec3, glam::Quat)>,
}

/// Place the frame (support loss on or off, `factor` its dynamic factor), shoot the columns
/// `shoot` (indices into the frame's columns) from the side, step `frames`.
fn run(f: &Frame, support_loss: bool, factor: f32, shoot: &[usize], frames: u32) -> Run {
    let asset = DestructibleAssetDesc::from_pack(&f.pack).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let ground = w.create_body(&BodyDef::default()).unwrap();
    let h = stress.ground_height().unwrap_or(0.0) + offset[1];
    w.create_shape(
        ground,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: h,
        },
    )
    .unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            support_loss,
            support_loss_dynamic_factor: factor,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    // Each shot from -z through the middle of its column, in the asset frame, from 2.5 m
    // away: its first contact is then three frames' travel off (40 m/s at 60 Hz), exactly
    // one frame's when the second prediction checks it. The strike is predicted the frame
    // before it, so the shot meets a freed column. (The strike prediction once picked its
    // struck chunk by a distance recomputed bit for bit: here no chunk matched, the strike
    // went unpredicted, and the shot met the column still welded and static and stopped
    // dead. The start was moved to 2.3 m around it.)
    for &column in shoot {
        let (r, m) = (0.3f32, 400.0f32);
        let p = asset.chunks[f.columns[column] as usize].pose.position;
        let at = [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2] - 2.5];
        let b = w
            .create_body(&BodyDef {
                linear_velocity: [0.0, 0.0, 40.0],
                is_bullet: true,
                ..BodyDef::dynamic(at)
            })
            .unwrap();
        w.create_sphere_shape(
            b,
            &ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            r,
        )
        .unwrap();
        let i = 0.4 * m * r * r;
        w.body_set_mass_data(
            b,
            MassData {
                mass: m,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )
        .unwrap();
        w.body_set_projectile(b, Some(Projectile::default()))
            .unwrap();
    }
    let mut broken = 0;
    let mut trace = Vec::new();
    for n in 0..frames {
        if std::env::var("SL_DEBUG").is_ok() && n % 10 == 0 {
            let body = w.destructible_chunk_body(id, f.top).unwrap();
            let st = w.read_body_state(body).unwrap();
            let pose = asset.chunks[f.top as usize].pose.position;
            let p = glam::Quat::from_array(st.transform.rotation) * glam::Vec3::from(pose)
                + glam::Vec3::from(st.transform.position);
            let ty = w.body_type(body).ok();
            eprintln!(
                "frame {n}: top {p} body {body:?} {ty:?} v {:?}",
                st.linear_velocity
            );
        }
        w.step().unwrap();
        for fr in w.poll_blocking().unwrap() {
            for e in &fr.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    broken += bonds.len();
                }
            }
        }
        let st = w
            .read_body_state(w.destructible_chunk_body(id, f.top).unwrap())
            .unwrap();
        trace.push((
            glam::Vec3::from(st.transform.position),
            glam::Quat::from_array(st.transform.rotation),
        ));
    }
    Run {
        world: w,
        id,
        asset,
        broken,
        ground: h,
        trace,
    }
}

impl Run {
    /// Whether `chunk` still hangs on any of `footings` (one actor).
    fn on_footings(&self, chunk: u32, footings: &[u32]) -> bool {
        let actor = self.world.destructible_actor_of(self.id, chunk).unwrap();
        footings.iter().any(|f| actor.chunks.contains(f))
    }

    /// `chunk`'s centre of mass now (world frame), and its body's rotation.
    fn centre(&self, chunk: u32) -> (glam::Vec3, glam::Quat) {
        let body = self.world.destructible_chunk_body(self.id, chunk).unwrap();
        let st = self.world.read_body_state(body).unwrap();
        // Chunk world transform = body transform * the chunk's pose in the asset frame.
        let pose = self.asset.chunks[chunk as usize].pose.position;
        let q = glam::Quat::from_array(st.transform.rotation);
        (
            q * glam::Vec3::from(pose) + glam::Vec3::from(st.transform.position),
            q,
        )
    }

    /// The world corners of the top (a box of `half` about its pose in the asset frame)
    /// with its body at `at` (an entry of `trace`).
    fn top_corners(&self, f: &Frame, at: (glam::Vec3, glam::Quat)) -> Vec<glam::Vec3> {
        let c = glam::Vec3::from(self.asset.chunks[f.top as usize].pose.position);
        let h = glam::DVec3::from(f.top_half).as_vec3();
        (0..8)
            .map(|i| {
                let s = glam::Vec3::new(
                    if i & 1 == 0 { -1.0 } else { 1.0 },
                    if i & 2 == 0 { -1.0 } else { 1.0 },
                    if i & 4 == 0 { -1.0 } else { 1.0 },
                );
                at.1 * (c + s * h) + at.0
            })
            .collect()
    }

    /// How deep the world point `p` is inside the top with its body at `at` (m; negative
    /// outside).
    fn depth_in_top(&self, f: &Frame, at: (glam::Vec3, glam::Quat), p: glam::Vec3) -> f32 {
        let c = glam::Vec3::from(self.asset.chunks[f.top as usize].pose.position);
        let h = glam::DVec3::from(f.top_half).as_vec3();
        let local = at.1.inverse() * (p - at.0) - c;
        (h - local.abs()).min_element()
    }
}

/// A 7.2 m slab of 100 t on four columns in a row (x = -3, -1, 1, 3); the one at x = -3 is
/// shot away. Intact, each column carries a quarter of the slab; after, a third.
fn slab_row(utilisation_after: f64) -> Frame {
    let weight = 1.0e5 * G;
    let compression = weight / 3.0 / utilisation_after / AREA;
    let columns = [[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
    frame(&columns, &[0], 1.0e5, [3.6, 0.2, 0.6], compression)
}

#[test]
fn a_slab_whose_column_is_shot_away_collapses_onto_the_rest() {
    // The remaining three columns' bonds at 120% of capacity once the fourth is gone (each
    // held 90% before: it stands until then).
    let f = slab_row(1.2);
    let start = run(&f, true, 1.0, &[], 1).centre(f.top).0;
    let off = run(&f, false, 1.0, &[0], 90);
    assert!(off.broken > 0, "the shot breaks the shot column's bonds");
    assert!(
        !off.on_footings(f.columns[0], &f.footings[..1]),
        "the shot column is off its footing"
    );
    assert!(
        off.on_footings(f.top, &f.footings[1..]),
        "without support loss the slab stands on three columns"
    );
    assert!((off.centre(f.top).0 - start).length() < 0.05);
    let on = run(&f, true, 1.0, &[0], 90);
    eprintln!(
        "support loss: {} bonds broken (without: {})",
        on.broken, off.broken
    );
    assert!(
        !on.on_footings(f.top, &f.footings),
        "the slab's weight breaks the columns left"
    );
    let drop = start.y - on.centre(f.top).0.y;
    assert!(drop > 0.3, "the slab comes down ({drop} m)");
    // The record of the weight's breaks: only with support loss, and among the breaks.
    let by_weight = on.world.destructible_support_loss_breaks(on.id).unwrap();
    assert!(!by_weight.is_empty() && by_weight.len() < on.broken);
    assert!(off
        .world
        .destructible_support_loss_breaks(off.id)
        .unwrap()
        .is_empty());
}

#[test]
fn a_structure_intact_under_its_own_weight_never_breaks() {
    // Every column at 75% under the slab, the dynamic factor at its maximum: nothing struck,
    // nothing breaks, nothing moves.
    let f = slab_row(1.0);
    let start = run(&f, true, 2.0, &[], 1).centre(f.top).0;
    let r = run(&f, true, 2.0, &[], 120);
    assert_eq!(r.broken, 0);
    assert!(r.on_footings(f.top, &f.footings));
    assert!((r.centre(f.top).0 - start).length() < 0.01);
    // Struck, with every column left at 75% (the shot column's own bonds aside): the pass runs
    // on the piece and breaks nothing more.
    let f = slab_row(0.75);
    let r = run(&f, true, 1.0, &[0], 90);
    assert!(
        r.broken > 0 && r.on_footings(f.top, &f.footings[1..]),
        "{} broken",
        r.broken
    );
    assert!((r.centre(f.top).0 - start).length() < 0.05);
}

#[test]
fn a_sudden_loss_with_the_dynamic_factor_fails_a_member_the_static_flow_holds() {
    // The demand after a sudden loss is before + k (after - before) (the step response,
    // `support_loss.rs`). Slab weight W = 1e5 kg * 9.81 = 981 kN on four equal columns:
    // before the loss each column's top bond carries W/4, after it W/3. With the static
    // utilisation after u = (W/3) / C:
    //   top bond:    (W/3 + (k - 1)(W/3 - W/4)) / C = u (1 + (k - 1)/4)
    //   bottom bond: the same plus the column's own 400 kg (3.9 kN), which does not
    //                change: + 3924 / C = + 0.012 u
    // u = 0.9, k = 1: 0.900 and 0.911, both hold; k = 2: 1.125 and 1.136, both break.
    let f = slab_row(0.9);
    let quasi = run(&f, true, 1.0, &[0], 90);
    assert!(
        quasi.on_footings(f.top, &f.footings[1..]),
        "factor 1: 90% holds"
    );
    let dynamic = run(&f, true, 2.0, &[0], 90);
    assert!(
        !dynamic.on_footings(f.top, &f.footings),
        "factor 2: 112% of capacity breaks the columns"
    );
    eprintln!(
        "factor 1: {} bonds broken; factor 2: {}",
        quasi.broken, dynamic.broken
    );
    assert!(dynamic.broken > quasi.broken);
    // u = 0.6, k = 2: 0.75 and 0.757, both hold. (Factor 2 on the whole post-loss demand,
    // the codes' linear static convention, would ask 1.2 of them and break them.)
    let f = slab_row(0.6);
    let r = run(&f, true, 2.0, &[0], 90);
    assert!(
        r.on_footings(f.top, &f.footings[1..]),
        "factor 2 at 60%: the increase alone is amplified, and it holds"
    );
    assert_eq!(r.broken, quasi.broken, "only the shot column's bonds break");
}

#[test]
fn a_block_whose_centre_leaves_its_footprint_tips() {
    // A 4 m block of 40 t on two columns at x = -1 (z = +-1) and one at x = 1.6, shot away:
    // its centre (x = 0) is then off the line the two make. The columns are strong: weight
    // alone breaks nothing, only the geometry does.
    let columns = [[-1.0, -1.0], [-1.0, 1.0], [1.6, 0.0]];
    let f = frame(&columns, &[2], 4.0e4, [2.0, 2.0, 1.4], 1.0e9);
    let (start, _) = run(&f, true, 1.0, &[], 1).centre(f.top);
    let off = run(&f, false, 1.0, &[2], 90);
    assert!(
        off.on_footings(f.top, &f.footings[..2]),
        "without support loss it hangs on two welds"
    );
    let on = run(&f, true, 1.0, &[2], 150);
    assert!(
        !on.on_footings(f.top, &f.footings),
        "it lets go of its last columns"
    );
    // The columns it let go of stay on their footings.
    for k in 0..2 {
        assert!(
            on.on_footings(f.columns[k], &f.footings[k..=k]),
            "column {k} stands"
        );
    }
    // Free of its welds it tips off over the tops of the columns it let go of (they are
    // static pieces): no frame has a column's top inside the block, and it never goes below
    // the ground.
    let (now, _) = on.centre(f.top);
    let offset = glam::DVec3::from(on.asset.stress.as_ref().unwrap().placement()).as_vec3();
    for k in 0..2 {
        let c = glam::Vec3::from(on.asset.chunks[f.columns[k] as usize].pose.position) - offset;
        // 0.1 m below the column's top (its half height is 1 m).
        let top = c + glam::Vec3::new(0.0, 0.9, 0.0);
        for (n, &at) in on.trace.iter().enumerate() {
            let d = on.depth_in_top(&f, at, top);
            assert!(
                d < 0.05,
                "frame {n}: column {k}'s top {d} m inside the block"
            );
        }
    }
    for (n, &at) in on.trace.iter().enumerate() {
        let low = on
            .top_corners(&f, at)
            .iter()
            .map(|c| c.y)
            .fold(f32::INFINITY, f32::min);
        assert!(
            low > on.ground - 0.05,
            "frame {n}: the block's lowest corner at {low}"
        );
    }
    let body = on.world.destructible_chunk_body(on.id, f.top).unwrap();
    assert_eq!(on.world.body_type(body).unwrap(), BodyType::Dynamic);
    eprintln!("block: centre {start} -> {now}");
    // Over to the side the shot column held (+x), off the deck and down onto the ground and
    // the footing there.
    assert!(
        now.y < start.y - 1.5 && now.x > 1.5,
        "it tips off its columns towards +x (centre {now})"
    );
}

#[test]
fn the_dynamic_factor_is_validated() {
    let f = slab_row(1.0);
    let asset = DestructibleAssetDesc::from_pack(&f.pack).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let place = |w: &mut World, support_loss: bool, k: f32| {
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            support_loss,
            support_loss_dynamic_factor: k,
            ..DestructibleDef::new(a)
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    for k in [0.5, 2.5, f32::NAN, f32::INFINITY] {
        let e = place(&mut w, true, k).unwrap_err();
        assert!(e.contains("outside 1 to 2"), "{k}: {e}");
    }
    let e = place(&mut w, false, 2.0).unwrap_err();
    assert!(e.contains("needs support_loss"), "{e}");
    assert_eq!(DestructibleDef::new(a).support_loss_dynamic_factor, 1.0);
    assert!(!DestructibleDef::new(a).support_loss);
    let e = w
        .create_destructible(&DestructibleDef {
            support_loss: true,
            ..DestructibleDef::new(a)
        })
        .unwrap_err()
        .to_string();
    assert!(e.contains("Threshold strength model"), "{e}");
}

/// A piece split off a pack-built destructible lands on the world's ground plane and on the
/// static pieces, with support loss off: all three of the block's columns shot away, the
/// free block falls onto the footings and the ground and comes to rest there. (It once
/// fell straight through: the GPU's split put the 4 m block on a dynamic body the host's
/// mirror still saw as static, so the broadphase grid had no level for a collider that
/// large moving, and it took the oversized-static list, which never queries the statics.)
/// `SL_DEBUG=1` prints the block's centre every 10 frames.
#[test]
fn a_dropped_block_lands_on_the_ground() {
    let columns = [[-1.0, -1.0], [-1.0, 1.0], [1.6, 0.0]];
    let f = frame(&columns, &[0, 1, 2], 4.0e4, [2.0, 2.0, 1.4], 1.0e9);
    let (start, _) = run(&f, false, 1.0, &[], 1).centre(f.top);
    let r = run(&f, false, 1.0, &[0, 1, 2], 400);
    let (now, _) = r.centre(f.top);
    eprintln!(
        "{} bonds broken; the block's centre {start} -> {now} (ground at y = {})",
        r.broken, r.ground
    );
    assert!(
        !r.on_footings(f.top, &f.footings),
        "the block is off its columns"
    );
    // It came down by more than a column's height (2 m) onto the footings (0.5 m) and
    // the ground: its centre is at most its half height (2 m) above the footings' tops.
    assert!(now.y < start.y - 1.5, "the block came down: {now}");
    let lowest = |at| {
        r.top_corners(&f, at)
            .iter()
            .map(|c| c.y)
            .fold(f32::INFINITY, f32::min)
    };
    // No frame has it below the ground (a corner by more than a contact's slop).
    for (n, &at) in r.trace.iter().enumerate() {
        let low = lowest(at);
        assert!(
            low > r.ground - 0.05,
            "frame {n}: the block's lowest corner at {low}, the ground at {}",
            r.ground
        );
    }
    // And at rest.
    let body = r.world.destructible_chunk_body(r.id, f.top).unwrap();
    let st = r.world.read_body_state(body).unwrap();
    let speed = glam::Vec3::from(st.linear_velocity).length();
    assert!(speed < 0.2, "the block rests ({speed} m/s)");
}

/// Place one column under a `top_mass` block with support loss and crushing on (`crush`:
/// the concrete's crush parameters); step `frames`. Returns the world, the destructible,
/// the frame and the crush events by frame.
fn crush_column(
    top_mass: f64,
    crush: CrushMaterial,
    frames: u32,
) -> (World, DestructibleId, Frame, Vec<(u32, DestructionEvent)>) {
    let f = frame(&[[0.0, 0.0]], &[], top_mass, [0.5, 0.25, 0.5], 1e12);
    let asset = DestructibleAssetDesc::from_pack(&f.pack).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let ground = w.create_body(&BodyDef::default()).unwrap();
    let h = stress.ground_height().unwrap_or(0.0) + offset[1];
    w.create_shape(
        ground,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: h,
        },
    )
    .unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            support_loss: true,
            crush: Some(Crush {
                materials: vec![("concrete".into(), crush)],
                ..Crush::default()
            }),
            ..DestructibleDef::new(a)
        })
        .unwrap();
    let mut events = Vec::new();
    for n in 0..frames {
        w.step().unwrap();
        for fr in w.poll_blocking().unwrap() {
            for e in &fr.destruction_events {
                events.push((n, e.clone()));
            }
        }
    }
    (w, id, f, events)
}

/// The column's flow stress (`engine/docs/destruction-reactions.md` section 2, "Loads
/// through bonds"): the weight above it reaches its top bond, that plus its own its bottom
/// bond; pushing into it at its ends (each its half height `h` from its centre) they give
/// `sigma V = -h (L_top + L_bot)`, the uniaxial `sigma = (L_top + L_bot) / 2A`: `q = sigma`,
/// `p = sigma / 3`.
#[test]
fn support_loss_flow_loads_a_column_with_the_weight_above_it() {
    let strong = CrushMaterial {
        cap_pressure: 1e12,
        cohesion: 1e12,
        friction_slope: 0.0,
        crush_energy: 1e9,
        crush_viscosity: 1e9,
    };
    let top = 20_000.0;
    let (w, id, f, events) = crush_column(top, strong, 5);
    assert!(events.is_empty(), "{events:?}");
    let state = w.destructible_crush_state(id).unwrap();
    let (damage, p, q, crushed) = state[f.columns[0] as usize];
    let sigma = ((top * G) + (top + 400.0) * G) / (2.0 * AREA);
    assert!(damage == 0.0 && !crushed);
    assert!(
        ((q as f64) - sigma).abs() < 1e-4 * sigma,
        "q {q} vs {sigma}"
    );
    assert!(
        ((p as f64) - sigma / 3.0).abs() < 1e-4 * sigma,
        "p {p} vs {}",
        sigma / 3.0
    );
    // The block on it: its one bond pushes it up at its bottom face.
    let (_, pt, qt, _) = state[f.top as usize];
    let st = top * G * f.top_half[1] / (8.0 * f.top_half[0] * f.top_half[1] * f.top_half[2]);
    assert!(((qt as f64) - st).abs() < 1e-4 * st, "top q {qt} vs {st}");
    assert!(pt > 0.0);
}

/// A column whose cohesion its load beats crushes with cause Load when the law's damage
/// reaches 1: `excess = q - cohesion` (no friction, no cap), `D = n dt excess^2 /
/// (crushViscosity crushEnergy)` after n frames. Its mass goes to its fragments; the block
/// falls onto them.
#[test]
fn a_column_crushes_under_the_weight_above_it_with_cause_load() {
    let top = 20_000.0;
    let sigma = ((top * G) + (top + 400.0) * G) / (2.0 * AREA);
    let weak = CrushMaterial {
        cap_pressure: 1e12,
        cohesion: 0.5e6,
        friction_slope: 0.0,
        crush_energy: 1e6,
        crush_viscosity: 1e5,
    };
    let excess = sigma - weak.cohesion;
    let dt = 1.0 / 60.0;
    let want = (weak.crush_viscosity * weak.crush_energy / (excess * excess * dt)).ceil() as u32;
    assert!(want > 3 && want < 40, "{want}");
    let (w, id, f, events) = crush_column(top, weak, want + 20);
    let crushes: Vec<&(u32, DestructionEvent)> = events
        .iter()
        .filter(|e| matches!(e.1, DestructionEvent::ChunkCrushed { .. }))
        .collect();
    let (frame, first) = crushes.first().expect("the column crushes");
    let DestructionEvent::ChunkCrushed {
        chunk,
        cause,
        mass,
        fragments,
        peak_deviator,
        ..
    } = first
    else {
        unreachable!()
    };
    assert_eq!(*chunk, f.columns[0]);
    assert_eq!(*cause, CrushCause::Load);
    assert!(
        (*frame as i64 - (want as i64 - 1)).abs() <= 1,
        "crushed in frame {frame}, the law says after {want} frames"
    );
    assert!(((*peak_deviator as f64) - sigma).abs() < 1e-4 * sigma);
    // Conserve: its fragments carry its mass.
    let carried: f32 = fragments
        .iter()
        .map(|b| w.body_mass_data(*b).unwrap().unwrap().mass)
        .sum();
    assert!((carried - mass).abs() < 1e-3 * mass, "{carried} vs {mass}");
    // Its bonds broke with cause Crush.
    assert!(events.iter().any(|e| matches!(
        &e.1,
        DestructionEvent::BondsBroken { cause: BreakCause::Crush, bonds, .. } if bonds.len() == 2
    )));
    // The block came down.
    let body = w.destructible_chunk_body(id, f.top).unwrap();
    let v = w.read_body_state(body).unwrap().linear_velocity;
    let y0 = DestructibleAssetDesc::from_pack(&f.pack).unwrap().chunks[f.top as usize]
        .pose
        .position[1];
    let y = w.read_body_state(body).unwrap().transform.position[1];
    eprintln!("top block: origin y {y} (chunk at {y0}), v {v:?}");
    assert!(
        y < -0.5 || v[1] < -1.0,
        "the block falls once its column is crushed"
    );
}
