// The crash lab's drawn car bodies (crash-lab.html): the textured body skinned to the
// solver's nodes (showcase-car-render.mjs), live and replayed alike from the lab renderer's
// positions. The body is the vehicle's: the authored showcase sedan's visual, an imported
// JBeam vehicle's flexbodies and props (?mod=, beamng-mod.mjs, with its textures), or in a
// reference crash each generated design's own mod folder's. Also the skin binding select
// (#bindSel) and whether the body draws its own wheels. Lab only (no body in the plain views).

import { modVisual, modTextures } from './beamng-mod.mjs';
import { createShowcaseCarRenderer } from './showcase-car-render.mjs';

/** page.draw.bodyR (or null), bodyOn(), syncBody(), ownWheels(), bodyFor. */
export function createBody(page, { sampleCount }) {
  const { device, format, qs, cfg, draw } = page;
  const veh = page.vehicle;
  const { importSrc, carKind } = veh;
  const raycast = cfg.raycast;
  let importVis = null;
  const bodyVisual = () => carKind === 'showcase' ? (() => { const b = veh.showcase(); b.visual.lod ??= [b.visualLod, b.visualFar]; return b.visual; })()
    : (importVis ??= (() => { const r = modVisual(importSrc, { native: cfg.jbeamNative, raycast }); console.log('flexbodies', r.report); window.__flexReport = r.report; loadBodyTextures(r.visual); return r.visual; })());
  // An imported body's colour and data maps (normal, roughness, AO; decoded in the wasm),
  // uploaded once they arrive; until then it shades untextured.
  const loadBodyTextures = (vis) => {
    if (!importSrc.base || qs.get('tex') === '0') return;
    if (vis.textures?.length) modTextures(importSrc, vis.textures).then((layers) => bodyR.setTextures(layers, 1024));
    if (vis.dataTextures?.length) modTextures(importSrc, vis.dataTextures).then((layers) => bodyR.setDataTextures(layers, 1024));
  };
  const bodyR = draw.bodyR = cfg.lab && (carKind === 'showcase' || importSrc?.meshes?.length || importSrc?.skin) ? createShowcaseCarRenderer(device, { format, depthFormat: 'depth24plus', sampleCount }) : null;
  // A reference run's other generated designs (S09's 4x4 beside the wagon): each vehicle's
  // design is Rust's (crashInfo().bodies, in vehicle order), its visual that design's mod
  // folder's, built once (the first design's is bodyVisual()).
  const designVis = new Map();
  const crashDesignVisual = (i) => {
    const d = page.sim.crashInfo()?.bodies?.[i]?.design;
    const m = page.crash.folders.mods.find((x) => x.design === d);
    if (!m) return null;
    if (m.mod === importSrc) return bodyVisual();
    if (!designVis.has(d)) {
      const r = modVisual(m.mod, { native: cfg.jbeamNative, raycast });
      // One texture array serves every car (setTextures): a second design's maps would
      // replace the first's, so it shades untextured (the generated designs have none).
      if (r.visual.textures?.length || r.visual.dataTextures?.length) console.warn(`crash reference: ${d}'s textures are not drawn (one texture set per page)`);
      designVis.set(d, r.visual);
    }
    return designVis.get(d);
  };
  const bodyFor = draw.bodyFor = { layout: null, pos: null };
  let bodyIds = [];
  // The bodies on the current layout's vehicles, and the lab renderer's solid and cable
  // ranges for the rest; false until the lab has positions.
  draw.syncBody = () => {
    const labR = draw.labR, layout = page.layout, sim = page.sim;
    const pos = labR.positionsBuffer();
    if (!pos) return false;
    if (bodyFor.layout !== layout) {
      for (const id of bodyIds) bodyR.removeCar(id);
      // The body goes on the vehicles it was built for (vehicle 0's kind; in a reference run
      // each generated design's own): a scene's other bodies (the FMVSS 214 barrier) are
      // drawn solid, and beams a rig added past the body's own (the door pull's cables) as cables.
      const visOf = layout.vehicles.map((v, i) => {
        const vis = page.crash ? crashDesignVisual(i) : v.name === layout.vehicles[0].name ? bodyVisual() : null;
        return vis && v.nodes >= vis.nodeCount ? vis : null;
      });
      const own = (v) => !!visOf[layout.vehicles.indexOf(v)];
      bodyIds = layout.vehicles.filter(own).map((v) => { const vis = visOf[layout.vehicles.indexOf(v)]; return bodyR.addCar(vis, { nodeOffset: v.nodeOffset, beamOffset: v.beamOffset, vehicle: layout.vehicles.indexOf(v), lod: vis.lod ?? null,
        // Drive mode: the wheels' pieces ride the raycast wheels' records.
        wheelRecords: raycast ? sim.wheelRecords(layout.vehicles.indexOf(v)) : null }); });
      let tri = 0;
      const solidTris = [];
      let cables = [0, 0];
      for (const [i, v] of layout.vehicles.entries()) {
        const vis = visOf[i];
        if (!vis) solidTris.push([tri, v.triangles]);
        else if (v.beams > vis.beamRest0.length && cables[1] === 0) cables = [v.beamOffset + vis.beamRest0.length, v.beamOffset + v.beams];
        tri += v.triangles;
      }
      // A reference crash's bodies without a skin of their own (the FMVSS 214 barrier, whose
      // collision triangles are its face alone): their beams drawn as cables too, the last
      // such bodies' (one beam range).
      if (page.crash && cables[1] === 0) {
        const bare = layout.vehicles.filter((v) => !own(v));
        const last = layout.vehicles.length - bare.length;
        if (bare.length && bare.every((v, i) => v === layout.vehicles[last + i])) cables = [bare[0].beamOffset, bare.at(-1).beamOffset + bare.at(-1).beams];
      }
      labR.set({ solidTris, cables });
      bodyFor.step = null;
      bodyFor.layout = layout;
    }
    if (bodyFor.pos !== pos || bodyFor.sim !== sim) { bodyR.setPositions(pos, labR.restBuffer(), raycast ? sim.miscBuffer() : null); bodyFor.pos = pos; bodyFor.sim = sim; }
    return true;
  };
  draw.bodyOn = () => bodyR && draw.labR.opts.skin === 'body';
  // Whether the body draws its own wheels (no stand-in boxes): the showcase body from the
  // drive state (skin.rs bind_raycast_wheels), an import's wheel pieces on the raycast wheels
  // (flexmesh), or wheels that are parts of its own, skinned to their axle nodes (the
  // generated cars: the report's wheelsSkinned).
  draw.ownWheels = () => !!bodyR && (carKind === 'showcase' || (!!importSrc && (bodyVisual().rigid?.some((r) => r >= 2) || window.__flexReport?.wheelsSkinned > 0)));
  // The skin binding control: rebuild only the showcase car's visual (the physics, its
  // truck, stays) and put it on the vehicles again next frame.
  const sel = page.el.bindSel;
  sel.value = veh.skinBind;
  sel.hidden = !(carKind === 'showcase' && bodyR);
  sel.onchange = () => {
    veh.skinBind = sel.value;
    const cache = veh.showcaseBuilt();
    if (!cache) return;
    const c = veh.showcaseOf(veh.skinBind);
    Object.assign(cache, { visual: c.visual, visualLod: c.visualLod, visualFar: c.visualFar });
    bodyFor.layout = null;
  };
}
