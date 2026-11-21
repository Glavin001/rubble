//! The demos' scenes on a C world: the same operations, by the same names, as the web's
//! `WebWorld` (`rubble::play::api::HOST_API`), each one call into `rubble::play`.
//!
//! Structured data is JSON text with the types of `engine-webgpu/types/rubble-play.d.ts`
//! (generated from the Rust types): a scene (`SceneDef`), an action (`Action`), events,
//! presets, the render manifest. JSON a call returns stays valid until the next
//! JSON-returning call on the same world (the world's own) or thread (`rbPresetList`,
//! `rbPresetScene`, `rbSceneFromQuery`).
//!
//! Coordinates: an action's positions and directions are in the world's coordinates
//! (`RbCoordinates`) and converted here. A scene definition, its script's state and its
//! events are scene data in Rubble's frame (Y up, metres). The render manifest is in
//! Rubble's frame, as the poses `rbWorld_ExportPoses` writes, which it is read with.

use super::*;
use rubble::play::{api, Action, Aim, SceneDef};
use std::cell::RefCell;

thread_local! {
    static JSON: RefCell<CString> = RefCell::new(CString::default());
}

fn c_string(text: String) -> CString {
    CString::new(text.replace('\0', " ")).unwrap_or_default()
}

/// Hand `v` out as JSON from this thread's slot.
unsafe fn out_thread_json(
    v: &impl serde::Serialize,
    out: *mut *const c_char,
) -> anyhow::Result<()> {
    if out.is_null() {
        return Err(invalid("null output pointer"));
    }
    let c = c_string(serde_json::to_string(v)?);
    JSON.with(|j| {
        *j.borrow_mut() = c;
        out.write(j.borrow().as_ptr());
    });
    Ok(())
}

/// Hand `v` out as JSON from `w`'s slot.
unsafe fn out_world_json(
    w: &mut RbWorld,
    v: &impl serde::Serialize,
    out: *mut *const c_char,
) -> anyhow::Result<()> {
    if out.is_null() {
        return Err(invalid("null output pointer"));
    }
    w.json = c_string(serde_json::to_string(v)?);
    out.write(w.json.as_ptr());
    Ok(())
}

unsafe fn text<'a>(s: *const c_char, what: &'static str) -> anyhow::Result<&'a str> {
    if s.is_null() {
        return Err(invalid(what));
    }
    Ok(CStr::from_ptr(s).to_str()?)
}

/// `action` from the host's coordinates into Rubble's. A ground point (`at`) is the host's
/// two horizontal axes: x and z for Y-up hosts, x and y for Unreal.
fn action_in(c: &Coords, action: Action) -> Action {
    let f = |v: [f64; 3]| v.map(|x| x as f32);
    let d = |v: [f32; 3]| v.map(f64::from);
    let aim = |a: Aim| Aim {
        eye: d(c.position_in(f(a.eye))),
        dir: d(c.direction_in(f(a.dir))),
    };
    let ground = |at: [f64; 2]| {
        let p = if c.unreal() {
            c.position_in([at[0] as f32, at[1] as f32, 0.0])
        } else {
            [at[0] as f32, 0.0, at[1] as f32]
        };
        [f64::from(p[0]), f64::from(p[2])]
    };
    match action {
        Action::Fire { aim: a } => Action::Fire { aim: a.map(aim) },
        Action::Cannon { aim: a } => Action::Cannon { aim: aim(a) },
        Action::Crater { at } => Action::Crater { at: at.map(ground) },
        Action::Strike { at } => Action::Strike { at: at.map(ground) },
        Action::Shell {
            origin,
            velocity,
            radius,
        } => Action::Shell {
            origin: c.position_in(origin),
            velocity: c.linear_in(velocity),
            radius: c.length_in(radius),
        },
        a => a,
    }
}

/// Every preset (`PresetInfo[]` JSON): name, title, family, hub, page, query, and whether
/// its scene is placed in Rust (`placed`: `rbPresetScene` gives it).
#[no_mangle]
pub unsafe extern "C" fn rbPresetList(out_json: *mut *const c_char) -> RbResult {
    guard_args(|| out_thread_json(&api::preset_list(), out_json))
}

/// Preset `name`'s scene (`SceneDef` JSON), as its page builds it.
#[no_mangle]
pub unsafe extern "C" fn rbPresetScene(
    name: *const c_char,
    out_json: *mut *const c_char,
) -> RbResult {
    guard_args(|| {
        let scene = api::preset_scene(text(name, "null preset name")?)?;
        out_thread_json(&scene, out_json)
    })
}

/// The scene family `family`'s page builds from URL switches `query` (`scene=tbone&speed=80`;
/// null or empty: its defaults), as `SceneDef` JSON a host may edit before placing it.
#[no_mangle]
pub unsafe extern "C" fn rbSceneFromQuery(
    family: *const c_char,
    query: *const c_char,
    out_json: *mut *const c_char,
) -> RbResult {
    guard_args(|| {
        let family = text(family, "null family")?;
        let query = if query.is_null() {
            ""
        } else {
            text(query, "null query")?
        };
        out_thread_json(&api::scene_from_query(family, query)?, out_json)
    })
}

unsafe fn create_scene_world(
    scene_json: *const c_char,
    coordinates: RbCoordinates,
    gpu: GpuContext,
    out: *mut *mut RbWorld,
) -> anyhow::Result<()> {
    let scene: SceneDef = serde_json::from_str(text(scene_json, "null scene")?)?;
    let world = World::new(gpu, &scene.world_def()?)?;
    write(
        out,
        Box::into_raw(Box::new(RbWorld::of(world, Coords::new(coordinates)))),
    )
}

/// The world scene `scene_json` (`SceneDef`) needs, on the default GPU, nothing placed
/// yet (`rbWorld_PlaceScene`). `coordinates`: the host's, as `RbWorldDef::coordinates`.
#[no_mangle]
pub unsafe extern "C" fn rbCreateSceneWorld(
    scene_json: *const c_char,
    coordinates: RbCoordinates,
    out: *mut *mut RbWorld,
) -> RbResult {
    guard_args(|| create_scene_world(scene_json, coordinates, GpuContext::Default, out))
}

/// `rbCreateSceneWorld` on the host renderer's Metal device and queue (as
/// `rbCreateWorldOnMetal`). Apple platforms only (elsewhere `RB_UNSUPPORTED`).
#[no_mangle]
pub unsafe extern "C" fn rbCreateSceneWorldOnMetal(
    scene_json: *const c_char,
    coordinates: RbCoordinates,
    mtl_device: *mut c_void,
    mtl_command_queue: *mut c_void,
    out: *mut *mut RbWorld,
) -> RbResult {
    #[cfg(target_vendor = "apple")]
    {
        guard_args(|| {
            let (info, device, queue) =
                engine_webgpu::metal_interop::shared_device(mtl_device, mtl_command_queue)?;
            create_scene_world(
                scene_json,
                coordinates,
                GpuContext::Shared {
                    info,
                    device,
                    queue,
                },
                out,
            )
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (scene_json, coordinates, mtl_device, mtl_command_queue, out);
        guard_args(|| {
            Err(anyhow::Error::new(Unsupported(
                "rbCreateSceneWorldOnMetal needs an Apple platform",
            )))
        })
    }
}

/// Place scene `scene_json` (`SceneDef`) in this world (made by `rbCreateSceneWorld` from
/// the same scene) and start its script, which then runs before every `rbWorld_Step`.
/// `out_json` (optional): the family's placement (handles and layout) as JSON.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_PlaceScene(
    world: *mut RbWorld,
    scene_json: *const c_char,
    out_json: *mut *const c_char,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let scene: SceneDef = serde_json::from_str(text(scene_json, "null scene")?)?;
        let (placed, director) = api::place(&scene, &mut w.world)?;
        w.director = Some(director);
        if !out_json.is_null() {
            out_world_json(w, &placed.to_json()?, out_json)?;
        }
        Ok(())
    })
}

/// A player action on the placed scene before the next step (`Action` JSON:
/// `{"action":"fire","aim":{"eye":[..],"dir":[..]}}`, `{"action":"reset"}`, ...).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_Act(world: *mut RbWorld, action_json: *const c_char) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let action: Action = serde_json::from_str(text(action_json, "null action")?)?;
        let action = action_in(&w.coords, action);
        let RbWorld {
            world, director, ..
        } = w;
        director
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no scene placed (rbWorld_PlaceScene first)"))?
            .act(world, &action)
    })
}

/// The placed scene's frame scripts before the frame's steps (the crusher's winch, the
/// traffic drivers): `dt` the frame's time (s); `user_car` the soft body the player steers
/// now (−1: none; its driver leaves it alone). No scene placed: nothing.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_BeforeFrame(
    world: *mut RbWorld,
    dt: f64,
    user_car: i32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        match &mut w.director {
            Some(d) => d.before_frame(&mut w.world, dt, u32::try_from(user_car).ok()),
            None => Ok(()),
        }
    })
}

/// After the frame's steps: start the reads the scripts use next frame (never waits).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_AfterFrame(world: *mut RbWorld) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        match &mut w.director {
            Some(d) => d.after_frame(&w.world),
            None => Ok(()),
        }
    })
}

/// Frame-locked runs: start every read the next `rbWorld_BeforeFrame` needs; then poll
/// while `rbWorld_ScriptPending`.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_PrepareFrame(world: *mut RbWorld) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        match &mut w.director {
            Some(d) => d.prepare(&w.world),
            None => Ok(()),
        }
    })
}

/// Whether a script read is still in flight (`out_pending`).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_ScriptPending(
    world: *mut RbWorld,
    out_pending: *mut bool,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let pending = w.director.as_ref().is_some_and(|d| d.pending(&w.world));
        write(out_pending, pending)
    })
}

/// The soft body the player drives (the ram aims at it). No scene placed: nothing.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_SetPlayer(world: *mut RbWorld, soft: u32) -> RbResult {
    guard_args(|| {
        if let Some(d) = &mut world_mut(world)?.director {
            d.set_player(soft);
        }
        Ok(())
    })
}

/// Whether the placed scene's script fires on its own. No scene placed: nothing.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_SetAutofire(world: *mut RbWorld, on: bool) -> RbResult {
    guard_args(|| {
        if let Some(d) = &mut world_mut(world)?.director {
            d.set_autofire(on);
        }
        Ok(())
    })
}

/// The placed scene's script state (JSON; `null` with no scene placed).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_GetScriptState(
    world: *mut RbWorld,
    out_json: *mut *const c_char,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let state = w.director.as_ref().map(|d| d.state());
        out_world_json(w, &state, out_json)
    })
}

/// What the placed scene's director did since the last call (`Event[]` JSON, oldest first):
/// the script's own volleys as well as the player's actions.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_TakeEvents(
    world: *mut RbWorld,
    out_json: *mut *const c_char,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let events = w
            .director
            .as_mut()
            .map(|d| d.take_events())
            .unwrap_or_default();
        out_world_json(w, &events, out_json)
    })
}

/// Changes whenever a collider is added, removed or rewritten, or a heightfield is edited:
/// a renderer rebuilds its draw groups when it does.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_GetRenderVersion(
    world: *mut RbWorld,
    out_version: *mut u64,
) -> RbResult {
    guard_args(|| write(out_version, world_mut(world)?.world.render_version()))
}

/// What to draw (`RenderManifest` JSON): every drawable collider grouped by geometry
/// (instanced boxes, spheres and capsules by collider index; merged hull and heightfield
/// meshes), planes, and what was skipped and why. In Rubble's frame: a vertex reaches the
/// world through its collider's and body's poses (`engine-world/src/render_manifest.rs`).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_GetRenderManifest(
    world: *mut RbWorld,
    out_json: *mut *const c_char,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let m = w.world.render_manifest();
        out_world_json(w, &m, out_json)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 1e-5)
    }

    /// An Unreal action lands where the same action in Rubble's frame does.
    #[test]
    fn actions_convert_from_the_hosts_coordinates() {
        let ue = Coords::new(RbCoordinates::RB_COORDINATES_UNREAL);
        // Rubble (x, y, z) m is Unreal (-z, x, y) cm.
        let ue_point = |p: [f64; 3]| [-p[2] * 100.0, p[0] * 100.0, p[1] * 100.0];
        let ue_dir = |d: [f64; 3]| [-d[2], d[0], d[1]];
        let (eye, dir) = ([1.0, 2.0, 3.0], [0.0, 0.6, 0.8]);
        let a = action_in(
            &ue,
            Action::Cannon {
                aim: Aim {
                    eye: ue_point(eye),
                    dir: ue_dir(dir),
                },
            },
        );
        let Action::Cannon { aim } = a else { panic!() };
        assert!(close(aim.eye, eye) && close(aim.dir, dir), "{aim:?}");

        // A ground point: Unreal's (x, y) is Rubble's (x, z).
        let g = ue_point([4.0, 0.0, -5.0]);
        let Action::Crater { at: Some(at) } = action_in(
            &ue,
            Action::Crater {
                at: Some([g[0], g[1]]),
            },
        ) else {
            panic!()
        };
        assert!(
            (at[0] - 4.0).abs() < 1e-5 && (at[1] + 5.0).abs() < 1e-5,
            "{at:?}"
        );

        let Action::Shell {
            origin,
            velocity,
            radius,
        } = action_in(
            &ue,
            Action::Shell {
                origin: [0.0, 100.0, 200.0],
                velocity: [0.0, 0.0, -1000.0],
                radius: 50.0,
            },
        )
        else {
            panic!()
        };
        assert!(close(origin.map(f64::from), [1.0, 2.0, 0.0]), "{origin:?}");
        assert!(
            close(velocity.map(f64::from), [0.0, -10.0, 0.0]),
            "{velocity:?}"
        );
        assert!((radius - 0.5).abs() < 1e-6);

        // Rubble's own frame passes through.
        let y = Coords::new(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        assert_eq!(
            action_in(
                &y,
                Action::Crater {
                    at: Some([1.0, 2.0])
                }
            ),
            Action::Crater {
                at: Some([1.0, 2.0])
            }
        );
    }
}
