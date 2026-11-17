// The rigid family in the playground shell (play.html): stacks, rings, piles and joint
// rigs (rubble::play::rigid, as demo.html places them). Physics, scene and solver
// constants are Rust's (the scene's own WorldDef); this module adds the camera the scene
// table names and the scene menu.
import { demoScenes } from './demo-config.mjs';

const SCENES = ['wall', 'ring', 'gables', 'pyramid', 'pile', 'columns', 'ring110k', 'gables506k'];

/** WorldDef fields over the scene's own (as demo.html runs it): bodies thrown off the
 * ground below y = −50 are removed, so a settled scene can stop stepping. */
export const settings = () => ({ killBelow: -50 });

export async function setup({ sceneDef, lookFrom, setShadowRadius, ui }) {
  const id = new URLSearchParams(location.search).get('scene') ?? sceneDef.scene;
  const cfg = demoScenes(new URLSearchParams(location.search))[id];
  if (cfg?.eye && cfg?.look) {
    lookFrom(cfg.eye, cfg.look);
    setShadowRadius(Math.max(20, Math.hypot(...cfg.eye.map((v, i) => v - cfg.look[i])) * 0.8));
  }
  ui.select('Scene', 'scene', SCENES.map((s) => [s, demoScenes()[s]?.title ?? s]), id);
  ui.help('Drag to orbit · scroll to zoom · right-drag to pan.');
  return {};
}
