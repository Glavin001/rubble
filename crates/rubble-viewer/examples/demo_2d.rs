use rubble_scenes::{scenes_2d, INITIAL_SCENE_2D};
use rubble_viewer::Viewer2D;

fn main() {
    let mut viewer = Viewer2D::new(0.0, -9.81);

    let mut initial_scene_idx = 0;
    for scene in scenes_2d() {
        let descs = (scene.build)();
        let scene_idx = viewer.add_scene_descs(scene.name, descs);
        if scene.name == INITIAL_SCENE_2D {
            initial_scene_idx = scene_idx;
        }
    }
    viewer.set_initial_scene(initial_scene_idx);
    viewer.run();
}
