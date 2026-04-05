use rubble_scenes::{scenes_3d, INITIAL_SCENE_3D};
use rubble_viewer::Viewer3D;

fn main() {
    let mut viewer = Viewer3D::new(0.0, -9.81, 0.0);

    let mut initial_scene_idx = 0;
    for scene in scenes_3d() {
        let descs = (scene.build)();
        let scene_idx = viewer.add_scene_descs(scene.name, descs);
        if scene.name == INITIAL_SCENE_3D {
            initial_scene_idx = scene_idx;
        }
    }
    viewer.set_initial_scene(initial_scene_idx);
    viewer.run();
}
