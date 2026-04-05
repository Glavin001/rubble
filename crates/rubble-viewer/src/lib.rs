pub mod camera;
pub mod mesh;
pub mod overlay;
pub mod renderer;

use camera::{Camera2D, OrbitCamera};
use glam::{Quat, Vec2, Vec3};
use renderer::{model_matrix, palette_color, static_color, DrawList, InstanceData, Renderer};
use rubble_math::BodyHandle;
use std::{sync::Arc, time::Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

const CONTROLS_3D: &[&str] = &[
    "Rotate camera: left drag",
    "Pan camera: right or middle drag",
    "Zoom camera: mouse wheel",
];

const CONTROLS_2D: &[&str] = &["Pan view: left drag", "Zoom view: mouse wheel"];

// ---------------------------------------------------------------------------
// Shape tracking (mirrors rubble-wasm bookkeeping)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum ShapeInfo3D {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { half_height: f32, radius: f32 },
    Plane,
}

#[derive(Clone)]
enum ShapeInfo2D {
    Circle { radius: f32 },
    Rect { half_extents: Vec2 },
    Capsule { half_height: f32, radius: f32 },
}

struct Scene3D {
    name: String,
    descs: Vec<(rubble3d::RigidBodyDesc, ShapeInfo3D)>,
}

impl Scene3D {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            descs: Vec::new(),
        }
    }
}

struct Scene2D {
    name: String,
    descs: Vec<(rubble2d::RigidBodyDesc2D, ShapeInfo2D)>,
}

impl Scene2D {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            descs: Vec::new(),
        }
    }
}

fn shape_info_3d(shape: &rubble3d::ShapeDesc) -> ShapeInfo3D {
    match shape {
        rubble3d::ShapeDesc::Sphere { radius } => ShapeInfo3D::Sphere { radius: *radius },
        rubble3d::ShapeDesc::Box { half_extents } => ShapeInfo3D::Box {
            half_extents: *half_extents,
        },
        rubble3d::ShapeDesc::Capsule {
            half_height,
            radius,
        } => ShapeInfo3D::Capsule {
            half_height: *half_height,
            radius: *radius,
        },
        rubble3d::ShapeDesc::Plane { .. } => ShapeInfo3D::Plane,
        _ => panic!("Viewer3D does not support rendering this shape descriptor"),
    }
}

fn shape_info_2d(shape: &rubble2d::ShapeDesc2D) -> ShapeInfo2D {
    match shape {
        rubble2d::ShapeDesc2D::Circle { radius } => ShapeInfo2D::Circle { radius: *radius },
        rubble2d::ShapeDesc2D::Rect { half_extents } => ShapeInfo2D::Rect {
            half_extents: *half_extents,
        },
        rubble2d::ShapeDesc2D::Capsule {
            half_height,
            radius,
        } => ShapeInfo2D::Capsule {
            half_height: *half_height,
            radius: *radius,
        },
        _ => panic!("Viewer2D does not support rendering this shape descriptor"),
    }
}

fn build_world_3d(
    gravity: Vec3,
    scene: &Scene3D,
) -> (rubble3d::World, Vec<BodyHandle>, Vec<ShapeInfo3D>) {
    let config = rubble3d::SimConfig {
        gravity,
        ..Default::default()
    };
    let mut world = rubble3d::World::new(config).expect("GPU physics init failed");
    let mut handles = Vec::with_capacity(scene.descs.len());
    let mut shapes = Vec::with_capacity(scene.descs.len());
    for (desc, shape) in &scene.descs {
        handles.push(world.add_body(desc));
        shapes.push(shape.clone());
    }
    (world, handles, shapes)
}

fn build_world_2d(
    gravity: Vec2,
    scene: &Scene2D,
) -> (rubble2d::World2D, Vec<BodyHandle>, Vec<ShapeInfo2D>) {
    let config = rubble2d::SimConfig2D {
        gravity,
        ..Default::default()
    };
    let mut world = rubble2d::World2D::new(config).expect("GPU physics init failed");
    let mut handles = Vec::with_capacity(scene.descs.len());
    let mut shapes = Vec::with_capacity(scene.descs.len());
    for (desc, shape) in &scene.descs {
        handles.push(world.add_body(desc));
        shapes.push(shape.clone());
    }
    (world, handles, shapes)
}

// ---------------------------------------------------------------------------
// Viewer3D
// ---------------------------------------------------------------------------

pub struct Viewer3D {
    gravity: Vec3,
    scenes: Vec<Scene3D>,
    initial_scene: usize,
}

impl Viewer3D {
    pub fn new(gx: f32, gy: f32, gz: f32) -> Self {
        Self {
            gravity: Vec3::new(gx, gy, gz),
            scenes: Vec::new(),
            initial_scene: 0,
        }
    }

    fn ensure_default_scene(&mut self) -> usize {
        if self.scenes.is_empty() {
            self.scenes.push(Scene3D::new("Scene"));
            self.initial_scene = 0;
        }
        0
    }

    pub fn add_scene(&mut self, name: impl Into<String>) -> usize {
        self.scenes.push(Scene3D::new(name));
        self.scenes.len() - 1
    }

    pub fn add_scene_descs<I>(&mut self, name: impl Into<String>, descs: I) -> usize
    where
        I: IntoIterator<Item = rubble3d::RigidBodyDesc>,
    {
        let scene_idx = self.add_scene(name);
        for desc in descs {
            self.add_body_desc_to_scene(scene_idx, desc);
        }
        scene_idx
    }

    pub fn set_initial_scene(&mut self, scene_idx: usize) {
        assert!(
            scene_idx < self.scenes.len(),
            "initial scene index out of bounds"
        );
        self.initial_scene = scene_idx;
    }

    pub fn add_body_desc(&mut self, desc: rubble3d::RigidBodyDesc) {
        let scene_idx = self.ensure_default_scene();
        self.add_body_desc_to_scene(scene_idx, desc);
    }

    pub fn add_body_desc_to_scene(&mut self, scene_idx: usize, desc: rubble3d::RigidBodyDesc) {
        let shape = shape_info_3d(&desc.shape);
        self.scenes[scene_idx].descs.push((desc, shape));
    }

    pub fn add_sphere(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        self.add_body_desc(rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass: 1.0,
            shape: rubble3d::ShapeDesc::Sphere { radius },
            ..Default::default()
        });
    }

    pub fn add_box(&mut self, x: f32, y: f32, z: f32, hw: f32, hh: f32, hd: f32) {
        self.add_body_desc(rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass: 1.0,
            shape: rubble3d::ShapeDesc::Box {
                half_extents: Vec3::new(hw, hh, hd),
            },
            ..Default::default()
        });
    }

    pub fn add_capsule(&mut self, x: f32, y: f32, z: f32, half_height: f32, radius: f32) {
        self.add_body_desc(rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass: 1.0,
            shape: rubble3d::ShapeDesc::Capsule {
                half_height,
                radius,
            },
            ..Default::default()
        });
    }

    pub fn add_ground_plane(&mut self, y: f32) {
        self.add_body_desc(rubble3d::RigidBodyDesc {
            position: Vec3::ZERO,
            mass: 0.0,
            shape: rubble3d::ShapeDesc::Plane {
                normal: Vec3::Y,
                distance: y,
            },
            ..Default::default()
        });
    }

    pub fn add_static_box(&mut self, x: f32, y: f32, z: f32, hw: f32, hh: f32, hd: f32) {
        self.add_body_desc(rubble3d::RigidBodyDesc {
            position: Vec3::new(x, y, z),
            mass: 0.0,
            shape: rubble3d::ShapeDesc::Box {
                half_extents: Vec3::new(hw, hh, hd),
            },
            ..Default::default()
        });
    }

    pub fn run(mut self) {
        self.ensure_default_scene();
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App3D {
            viewer: self,
            state: None,
        };
        event_loop.run_app(&mut app).expect("Event loop failed");
    }
}

struct State3D {
    renderer: Renderer,
    window: Arc<Window>,
    world: rubble3d::World,
    handles: Vec<BodyHandle>,
    shapes: Vec<ShapeInfo3D>,
    scene_names: Vec<String>,
    current_scene: usize,
    camera: OrbitCamera,
    draw_list: DrawList,
    mouse_pressed: bool,
    right_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    frame_count: u32,
    fps: f32,
    fps_instant: Instant,
    render_ms: f32,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
}

struct App3D {
    viewer: Viewer3D,
    state: Option<State3D>,
}

impl ApplicationHandler for App3D {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Rubble 3D Physics")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
                )
                .expect("Failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(window.clone()));

        let current_scene = self.viewer.initial_scene;
        let (world, handles, shapes) =
            build_world_3d(self.viewer.gravity, &self.viewer.scenes[current_scene]);
        let scene_names = self
            .viewer
            .scenes
            .iter()
            .map(|scene| scene.name.clone())
            .collect();

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &window,
            None,
            None,
            None,
        );

        self.state = Some(State3D {
            renderer,
            window,
            world,
            handles,
            shapes,
            scene_names,
            current_scene,
            camera: OrbitCamera::default(),
            draw_list: DrawList::default(),
            mouse_pressed: false,
            right_pressed: false,
            last_mouse: None,
            frame_count: 0,
            fps: 0.0,
            fps_instant: Instant::now(),
            render_ms: 0.0,
            egui_ctx,
            egui_state,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        stacker::maybe_grow(512 * 1024, 8 * 1024 * 1024, || {
            self.handle_event_3d(event_loop, event);
        });
    }
}

impl App3D {
    fn handle_event_3d(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        let _ = state.egui_state.on_window_event(&state.window, &event);
        let pointer_over_ui = state.egui_ctx.is_pointer_over_egui();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
            }
            WindowEvent::MouseInput {
                state: btn, button, ..
            } => {
                let pressed = btn == ElementState::Pressed;
                if !pressed || !pointer_over_ui {
                    match button {
                        MouseButton::Left => state.mouse_pressed = pressed,
                        MouseButton::Right | MouseButton::Middle => state.right_pressed = pressed,
                        _ => {}
                    }
                }
                if !pressed {
                    state.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = state.last_mouse {
                    let dx = position.x - lx;
                    let dy = position.y - ly;
                    if state.mouse_pressed {
                        state.camera.rotate(dx as f32, dy as f32);
                    }
                    if state.right_pressed {
                        state.camera.pan(dx as f32, dy as f32);
                    }
                }
                state.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } if !pointer_over_ui => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.3,
                };
                state.camera.zoom(scroll);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Character("r".into()) =>
            {
                let (world, handles, shapes) = build_world_3d(
                    self.viewer.gravity,
                    &self.viewer.scenes[state.current_scene],
                );
                state.world = world;
                state.handles = handles;
                state.shapes = shapes;
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.world.step();
                let timings = *state.world.last_step_timings();

                state.draw_list.clear();
                for (i, handle) in state.handles.iter().enumerate() {
                    let pos = state.world.get_position(*handle).unwrap_or(Vec3::ZERO);
                    let rot = state.world.get_rotation(*handle).unwrap_or(Quat::IDENTITY);
                    let is_static = matches!(state.shapes[i], ShapeInfo3D::Plane);
                    let color = if is_static {
                        static_color()
                    } else {
                        palette_color(i)
                    };

                    match &state.shapes[i] {
                        ShapeInfo3D::Sphere { radius } => {
                            let s = Vec3::splat(*radius);
                            state.draw_list.spheres.push(InstanceData {
                                model: model_matrix(pos, rot, s).to_cols_array_2d(),
                                color,
                            });
                        }
                        ShapeInfo3D::Box { half_extents } => {
                            state.draw_list.cubes.push(InstanceData {
                                model: model_matrix(pos, rot, *half_extents).to_cols_array_2d(),
                                color,
                            });
                        }
                        ShapeInfo3D::Capsule {
                            half_height,
                            radius,
                        } => {
                            let s = Vec3::new(*radius, *half_height, *radius);
                            state.draw_list.capsules.push(InstanceData {
                                model: model_matrix(pos, rot, s).to_cols_array_2d(),
                                color,
                            });
                        }
                        ShapeInfo3D::Plane => {}
                    }
                }

                state.frame_count += 1;
                let elapsed = state.fps_instant.elapsed();
                if elapsed.as_secs_f32() >= 0.5 {
                    state.fps = state.frame_count as f32 / elapsed.as_secs_f32();
                    state.frame_count = 0;
                    state.fps_instant = Instant::now();
                }

                let raw_input = state.egui_state.take_egui_input(&state.window);
                let mut selected_scene = state.current_scene;
                let mut reset_requested = false;
                let full_output = state.egui_ctx.run_ui(raw_input, |ctx| {
                    overlay::draw_panel(
                        ctx,
                        "Rubble 3D Viewer",
                        CONTROLS_3D,
                        &state.scene_names,
                        &mut selected_scene,
                        &mut reset_requested,
                        state.fps,
                        state.handles.len(),
                        &timings,
                        state.render_ms,
                    );
                });
                if selected_scene != state.current_scene || reset_requested {
                    let (world, handles, shapes) =
                        build_world_3d(self.viewer.gravity, &self.viewer.scenes[selected_scene]);
                    state.world = world;
                    state.handles = handles;
                    state.shapes = shapes;
                    state.current_scene = selected_scene;
                }
                state
                    .egui_state
                    .handle_platform_output(&state.window, full_output.platform_output);
                let primitives = state
                    .egui_ctx
                    .tessellate(full_output.shapes, full_output.pixels_per_point);

                let size = state.window.inner_size();
                let sd = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [size.width, size.height],
                    pixels_per_point: full_output.pixels_per_point,
                };

                let vp = state.camera.view_proj(state.renderer.aspect());
                let eye = state.camera.eye();

                let t_render = Instant::now();
                state.renderer.render_3d(
                    vp,
                    eye,
                    &state.draw_list,
                    true,
                    &primitives,
                    &full_output.textures_delta,
                    &sd,
                );
                state.render_ms = t_render.elapsed().as_secs_f32() * 1000.0;

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Viewer2D
// ---------------------------------------------------------------------------

pub struct Viewer2D {
    gravity: Vec2,
    scenes: Vec<Scene2D>,
    initial_scene: usize,
}

impl Viewer2D {
    pub fn new(gx: f32, gy: f32) -> Self {
        Self {
            gravity: Vec2::new(gx, gy),
            scenes: Vec::new(),
            initial_scene: 0,
        }
    }

    fn ensure_default_scene(&mut self) -> usize {
        if self.scenes.is_empty() {
            self.scenes.push(Scene2D::new("Scene"));
            self.initial_scene = 0;
        }
        0
    }

    pub fn add_scene(&mut self, name: impl Into<String>) -> usize {
        self.scenes.push(Scene2D::new(name));
        self.scenes.len() - 1
    }

    pub fn add_scene_descs<I>(&mut self, name: impl Into<String>, descs: I) -> usize
    where
        I: IntoIterator<Item = rubble2d::RigidBodyDesc2D>,
    {
        let scene_idx = self.add_scene(name);
        for desc in descs {
            self.add_body_desc_to_scene(scene_idx, desc);
        }
        scene_idx
    }

    pub fn set_initial_scene(&mut self, scene_idx: usize) {
        assert!(
            scene_idx < self.scenes.len(),
            "initial scene index out of bounds"
        );
        self.initial_scene = scene_idx;
    }

    pub fn add_body_desc(&mut self, desc: rubble2d::RigidBodyDesc2D) {
        let scene_idx = self.ensure_default_scene();
        self.add_body_desc_to_scene(scene_idx, desc);
    }

    pub fn add_body_desc_to_scene(&mut self, scene_idx: usize, desc: rubble2d::RigidBodyDesc2D) {
        let shape = shape_info_2d(&desc.shape);
        self.scenes[scene_idx].descs.push((desc, shape));
    }

    pub fn add_circle(&mut self, x: f32, y: f32, radius: f32) {
        self.add_body_desc(rubble2d::RigidBodyDesc2D {
            x,
            y,
            mass: 1.0,
            shape: rubble2d::ShapeDesc2D::Circle { radius },
            ..Default::default()
        });
    }

    pub fn add_rect(&mut self, x: f32, y: f32, hw: f32, hh: f32, angle: f32, mass: f32) {
        self.add_body_desc(rubble2d::RigidBodyDesc2D {
            x,
            y,
            angle,
            mass,
            shape: rubble2d::ShapeDesc2D::Rect {
                half_extents: Vec2::new(hw, hh),
            },
            ..Default::default()
        });
    }

    pub fn add_static_rect(&mut self, x: f32, y: f32, hw: f32, hh: f32, angle: f32) {
        self.add_rect(x, y, hw, hh, angle, 0.0);
    }

    pub fn add_capsule(&mut self, x: f32, y: f32, half_height: f32, radius: f32) {
        self.add_body_desc(rubble2d::RigidBodyDesc2D {
            x,
            y,
            mass: 1.0,
            shape: rubble2d::ShapeDesc2D::Capsule {
                half_height,
                radius,
            },
            ..Default::default()
        });
    }

    pub fn run(mut self) {
        self.ensure_default_scene();
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = App2D {
            viewer: self,
            state: None,
        };
        event_loop.run_app(&mut app).expect("Event loop failed");
    }
}

struct State2D {
    renderer: Renderer,
    window: Arc<Window>,
    world: rubble2d::World2D,
    handles: Vec<BodyHandle>,
    shapes: Vec<ShapeInfo2D>,
    scene_names: Vec<String>,
    current_scene: usize,
    camera: Camera2D,
    draw_list: DrawList,
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    frame_count: u32,
    fps: f32,
    fps_instant: Instant,
    render_ms: f32,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
}

struct App2D {
    viewer: Viewer2D,
    state: Option<State2D>,
}

impl ApplicationHandler for App2D {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Rubble 2D Physics")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
                )
                .expect("Failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(window.clone()));

        let current_scene = self.viewer.initial_scene;
        let (world, handles, shapes) =
            build_world_2d(self.viewer.gravity, &self.viewer.scenes[current_scene]);
        let scene_names = self
            .viewer
            .scenes
            .iter()
            .map(|scene| scene.name.clone())
            .collect();

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &window,
            None,
            None,
            None,
        );

        self.state = Some(State2D {
            renderer,
            window,
            world,
            handles,
            shapes,
            scene_names,
            current_scene,
            camera: Camera2D::default(),
            draw_list: DrawList::default(),
            mouse_pressed: false,
            last_mouse: None,
            frame_count: 0,
            fps: 0.0,
            fps_instant: Instant::now(),
            render_ms: 0.0,
            egui_ctx,
            egui_state,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        stacker::maybe_grow(512 * 1024, 8 * 1024 * 1024, || {
            self.handle_event_2d(event_loop, event);
        });
    }
}

impl App2D {
    fn handle_event_2d(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        let _ = state.egui_state.on_window_event(&state.window, &event);
        let pointer_over_ui = state.egui_ctx.is_pointer_over_egui();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
            }
            WindowEvent::MouseInput {
                state: btn, button, ..
            } => {
                let pressed = btn == ElementState::Pressed;
                if (!pressed || !pointer_over_ui) && button == MouseButton::Left {
                    state.mouse_pressed = pressed;
                }
                if !pressed {
                    state.last_mouse = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = state.last_mouse {
                    let dx = position.x - lx;
                    let dy = position.y - ly;
                    if state.mouse_pressed {
                        state.camera.pan(dx as f32, dy as f32);
                    }
                }
                state.last_mouse = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } if !pointer_over_ui => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.3,
                };
                state.camera.zoom_by(scroll);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Character("r".into()) =>
            {
                let (world, handles, shapes) = build_world_2d(
                    self.viewer.gravity,
                    &self.viewer.scenes[state.current_scene],
                );
                state.world = world;
                state.handles = handles;
                state.shapes = shapes;
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.world.step();
                let timings = *state.world.last_step_timings();

                state.draw_list.clear();
                for (i, handle) in state.handles.iter().enumerate() {
                    let pos2 = state.world.get_position(*handle).unwrap_or(Vec2::ZERO);
                    let angle = state.world.get_angle(*handle).unwrap_or(0.0);
                    let pos = Vec3::new(pos2.x, pos2.y, 0.0);
                    let rot = Quat::from_rotation_z(angle);
                    let color = palette_color(i);

                    match &state.shapes[i] {
                        ShapeInfo2D::Circle { radius } => {
                            let s = Vec3::new(*radius, *radius, 1.0);
                            state.draw_list.spheres.push(InstanceData {
                                model: model_matrix(pos, rot, s).to_cols_array_2d(),
                                color,
                            });
                        }
                        ShapeInfo2D::Rect { half_extents } => {
                            let s = Vec3::new(half_extents.x, half_extents.y, 1.0);
                            state.draw_list.cubes.push(InstanceData {
                                model: model_matrix(pos, rot, s).to_cols_array_2d(),
                                color,
                            });
                        }
                        ShapeInfo2D::Capsule {
                            half_height,
                            radius,
                        } => {
                            let s = Vec3::new(*radius, *half_height, 1.0);
                            state.draw_list.capsules.push(InstanceData {
                                model: model_matrix(pos, rot, s).to_cols_array_2d(),
                                color,
                            });
                        }
                    }
                }

                state.frame_count += 1;
                let elapsed = state.fps_instant.elapsed();
                if elapsed.as_secs_f32() >= 0.5 {
                    state.fps = state.frame_count as f32 / elapsed.as_secs_f32();
                    state.frame_count = 0;
                    state.fps_instant = Instant::now();
                }

                let raw_input = state.egui_state.take_egui_input(&state.window);
                let mut selected_scene = state.current_scene;
                let mut reset_requested = false;
                let full_output = state.egui_ctx.run_ui(raw_input, |ctx| {
                    overlay::draw_panel(
                        ctx,
                        "Rubble 2D Viewer",
                        CONTROLS_2D,
                        &state.scene_names,
                        &mut selected_scene,
                        &mut reset_requested,
                        state.fps,
                        state.handles.len(),
                        &timings,
                        state.render_ms,
                    );
                });
                if selected_scene != state.current_scene || reset_requested {
                    let (world, handles, shapes) =
                        build_world_2d(self.viewer.gravity, &self.viewer.scenes[selected_scene]);
                    state.world = world;
                    state.handles = handles;
                    state.shapes = shapes;
                    state.current_scene = selected_scene;
                }
                state
                    .egui_state
                    .handle_platform_output(&state.window, full_output.platform_output);
                let primitives = state
                    .egui_ctx
                    .tessellate(full_output.shapes, full_output.pixels_per_point);

                let size = state.window.inner_size();
                let sd = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [size.width, size.height],
                    pixels_per_point: full_output.pixels_per_point,
                };

                let vp = state.camera.view_proj(state.renderer.aspect());

                let t_render = Instant::now();
                state.renderer.render_2d(
                    vp,
                    &state.draw_list,
                    &primitives,
                    &full_output.textures_delta,
                    &sd,
                );
                state.render_ms = t_render.elapsed().as_secs_f32() * 1000.0;

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}
