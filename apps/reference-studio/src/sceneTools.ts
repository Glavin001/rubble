// The Scene view's page tools (shared/tools.ts, side "page"): what window.studio and WebMCP
// call to set up, simulate and review a scene. They act on the scene controller (scene.ts);
// data edits go through the data tools like any other edit.

import { StudioError } from '../shared/store.ts';
import { currentStore, tool } from './api.ts';
import { scene, type View3 } from './scene.ts';
import type { PageTools } from './studio.ts';

const vec = (v: unknown, k: string): [number, number, number] => {
  if (!Array.isArray(v) || v.length !== 3 || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) throw new StudioError(`${k} must be [x, y, z]`);
  return v as [number, number, number];
};

function describe() {
  const st = scene.state;
  const info = st.info;
  return {
    scenario: st.scenario,
    editable: st.editable,
    mode: st.mode,
    busy: st.busy,
    error: st.error,
    first_contact_ms: info ? info.impact_s * 1000 : null,
    duration_s: info?.duration_s ?? null,
    bodies: info?.bodies.map((b) => ({ name: b.name, design: b.design, at: b.at, yaw_deg: b.yaw_deg, speed_kmh: b.speed_kmh, mass_kg: b.mass_kg, nodes: b.nodes, surface_triangles: b.triangles.length / 3 })) ?? [],
    boxes: info?.boxes ?? [],
    selected: st.selected,
    recording: st.status,
    time_ms: st.ms,
    playing: st.playing,
    speed: st.speed,
    camera: st.camera,
    image: st.image,
    shot: st.shot,
    place_ms: st.placeMs,
    edit_ms: st.editMs,
    webgpu: st.webgpu,
    cpu_fallback: st.cpuFallback,
    undo: st.undo,
    redo: st.redo,
  };
}

async function jump(frame: string) {
  const f = currentStore().frame(frame);
  scene.setImage(f.id);
  if (f.camera) scene.setCamera({ eye: f.camera.eye, target: f.camera.target, up: f.camera.up, fov_y_deg: f.camera.fov_y_deg });
  if (f.pin && scene.state.status?.present) {
    if (scene.state.mode !== 'review') scene.setMode('review');
    scene.seek(f.pin.at_ms);
  }
  return { frame: f.id, pinned: !!f.pin, at_ms: f.pin?.at_ms ?? null, camera: f.camera, check: !!f.pin && f.pin.recording !== scene.state.status?.hash };
}

export function sceneTools(openView: () => void): PageTools {
  return {
    open_scene: async (i) => {
      openView();
      const info = await scene.open(String(i.scenario), typeof i.shot === 'string' ? i.shot : null);
      return { ...describe(), title: info.title };
    },
    set_mode: (i) => {
      scene.setMode(i.mode === 'review' ? 'review' : 'setup');
      return { mode: scene.state.mode };
    },
    simulate: async (i) => {
      const t0 = performance.now();
      const s = await scene.simulate({ cpu: i.cpu === true ? true : undefined });
      return { ...s, wall_s: (performance.now() - t0) / 1000 };
    },
    get_recording_status: () => scene.state.status ?? scene.engine?.studio.recordingStatus() ?? null,
    play: () => {
      scene.play();
      return { playing: true, ms: scene.state.ms };
    },
    pause: () => {
      scene.pause();
      return { playing: false, ms: scene.state.ms };
    },
    reset: () => {
      scene.reset();
      return { ms: scene.state.ms };
    },
    seek: (i) => {
      if (scene.state.mode !== 'review' && scene.state.status?.present) scene.setMode('review');
      scene.seek(Number(i.ms));
      return { ms: scene.state.ms };
    },
    step: (i) => {
      scene.step(Math.round(Number(i.samples)));
      return { ms: scene.state.ms };
    },
    set_playback_speed: (i) => {
      scene.setSpeed(Number(i.speed));
      return { speed: scene.state.speed };
    },
    get_report: () => {
      const e = scene.engine;
      if (!e) throw new StudioError('no scene open');
      return { executor: scene.state.status?.executor ?? null, fresh: scene.state.status?.fresh ?? false, report: e.studio.report() };
    },
    set_scene_camera: (i) => {
      const c = scene.state.camera;
      const next: View3 = { eye: i.eye !== undefined ? vec(i.eye, 'eye') : c.eye, target: i.target !== undefined ? vec(i.target, 'target') : c.target, up: i.up !== undefined ? vec(i.up, 'up') : c.up, fov_y_deg: typeof i.fov_y_deg === 'number' ? i.fov_y_deg : c.fov_y_deg };
      scene.setCamera(next);
      return next;
    },
    select_image: (i) => jump(String(i.frame)),
    jump_to_pin: (i) => jump(String(i.frame)),
    pin_image: async (i) => {
      const st = scene.state;
      if (!st.status?.present) throw new StudioError('no recording: simulate first, then scrub to the image\'s moment');
      const f = currentStore().frame(String(i.frame));
      const region = f.camera?.region ?? f.regions.find((r) => r.tracing.instances.length)?.id ?? null;
      const cam = scene.frameCamera(region);
      await tool('set_frame_camera', { frame: f.id, eye: cam.eye, target: cam.target, up: cam.up, fov_y_deg: cam.fov_y_deg, region: region ?? '' });
      const near = st.info?.moments.reduce<{ name: string; d: number } | null>((b, m) => (Math.abs(m.after_impact_ms - st.ms) < (b?.d ?? Infinity) ? { name: m.name, d: Math.abs(m.after_impact_ms - st.ms) } : b), null);
      await tool('set_frame_pin', { frame: f.id, at_ms: st.ms, recording: st.status.hash, label: near?.name ?? '' });
      scene.setImage(f.id);
      return { frame: f.id, at_ms: Math.round(st.ms * 100) / 100, recording: st.status.hash, camera: cam };
    },
    select_object: (i) => {
      const kind = i.kind === 'box' ? 'box' : 'body';
      const info = scene.state.info;
      if (!info || !(kind === 'box' ? info.boxes : info.bodies).some((x) => x.name === i.name)) throw new StudioError(`no ${kind} ${String(i.name)} in the scene`);
      scene.select({ kind, name: String(i.name) });
      return scene.state.selected;
    },
    undo_scene: () => ({ scenario: scene.history('undo') }),
    redo_scene: () => ({ scenario: scene.history('redo') }),
    describe_scene: () => describe(),
  };
}
