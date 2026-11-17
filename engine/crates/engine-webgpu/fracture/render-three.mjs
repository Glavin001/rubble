// three.js draw of the fracture lab (and the pages built on it: compare, Broadside, the
// timber yard), the same interface as render.mjs. The scene's render vertices come from
// Rust (engine-fracture: body-local position, normal, colour and body id, 10 floats each);
// each vertex is posed in TSL from the live body records (the World's body buffer, or the
// stress models' pose stream), read in place on the World's device. The camera is Rust's
// (engine-fracture view.rs `Orbit`: orbit, first-person walk and fly, shot rays); three.js
// draws from its eye with lights, shadows and PBR.
import * as THREE from 'three/webgpu';
import {
  storage, attribute, instanceIndex, vertexIndex, vec3, vec4, float, uint, varying, uniform, select, mix, normalize, cross,
  dot, length, abs, fract, sin, positionWorld, smoothstep, max, min, step, floatBitsToUint, cameraViewMatrix,
} from 'three/tsl';
import { Orbit, IndexedMesh, sceneHandle } from './wasm.mjs';
import { attachCameraInput } from './camera-input.mjs';
import { externalStorage, BODY_VEC4 } from '../rubble-three.mjs';

const JOINT_WORDS = 72, JOINT_FLAG_BROKEN = 1;
const qrot = (q, v) => v.add(cross(q.xyz, cross(q.xyz, v).add(v.mul(q.w))).mul(2));

// The bodies view's colour of `slot` (fracture_draw.slang `slotColor`): hues a golden-ratio
// step apart.
const slotColor = (slot) => {
  const h = fract(float(slot).mul(0.6180339887)).mul(6);
  const s = 0.62, v = 0.92, f = fract(h);
  const p = v * (1 - s), q = float(v).mul(float(1).sub(f.mul(s))), t = float(v).mul(float(1).sub(float(1).sub(f).mul(s)));
  const k = h.floor();
  return select(k.lessThan(1), vec3(v, t, p), select(k.lessThan(2), vec3(q, v, p), select(k.lessThan(3), vec3(p, v, t),
    select(k.lessThan(4), vec3(p, q, v), select(k.lessThan(5), vec3(t, p, v), vec3(v, p, q))))));
};
// Load over strength as a colour (fracture_draw.slang `stressColor`).
const stressColor = (u) => {
  const t = u.clamp(0, 1).mul(4);
  const c = [vec3(0.10, 0.32, 0.86), vec3(0.10, 0.74, 0.86), vec3(0.30, 0.82, 0.36), vec3(0.96, 0.84, 0.16), vec3(0.93, 0.36, 0.10)];
  return select(t.lessThan(1), mix(c[0], c[1], t), select(t.lessThan(2), mix(c[1], c[2], t.sub(1)), select(t.lessThan(3), mix(c[2], c[3], t.sub(2)), mix(c[3], c[4], t.sub(3)))));
};

export async function createRenderer(device, canvas, simulation, scene, onShoot = () => {}) {
  const renderer = new THREE.WebGPURenderer({ device, canvas, antialias: true });
  await renderer.init();
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const view = new THREE.Scene();
  const extent = Math.max(10, scene.extent ?? 50);
  view.backgroundNode = mix(vec3(0.72, 0.78, 0.84), vec3(0.36, 0.52, 0.72), smoothstep(0, 0.6, normalize(positionWorld).y));
  view.background = new THREE.Color(0xb7c6d6);
  view.fog = new THREE.Fog(0xb7c6d6, extent * 4, extent * 30);
  view.add(new THREE.HemisphereLight(0xe8f0ff, 0x4a4032, 1.3));
  const sun = new THREE.DirectionalLight(0xfff1dc, 2.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0005;
  sun.position.set(-0.5 * extent * 2, extent * 2, 0.7 * extent * 2);
  const sc = sun.shadow.camera;
  sc.left = sc.bottom = -extent * 1.2; sc.right = sc.top = extent * 1.2; sc.near = 0.1; sc.far = extent * 8;
  view.add(sun, sun.target);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 1000);

  // The body records the vertices are posed from; a buffer can be replaced as it grows.
  const bodyBufferOf = () => simulation.renderBodyBuffer ?? simulation.bodyBuffer;
  // A lab run's body records stay one buffer for the run (lab-sim.mjs).
  const bodies = externalStorage(renderer, bodyBufferOf(), 'vec4');
  const joints = simulation.jointBuffer ? externalStorage(renderer, simulation.jointBuffer, 'uint') : null;
  const jointsPerCopy = simulation.jointBuffer && scene.jointCount ? scene.jointCount / (scene.copies || 1) : 0;
  const stride = scene.instanceStride || 0;
  const colorMode = uniform(0, 'uint');
  const poseStream = simulation.poseDraw ? 1 : 0;

  // A mesh of render vertices (10 floats each). `copies`: house copies drawn instanced
  // (copy k on bodies id + k × instance stride; body 0, the ground, shared).
  const meshes = [];
  function bodyMesh(data, { indices = null, copies = 1, lines = false, lit = true } = {}) {
    const n = data.length / 10;
    const g = new THREE.InstancedBufferGeometry();
    const pos = new Float32Array(n * 3), nrm = new Float32Array(n * 3), col = new Float32Array(n * 3), body = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      pos.set(data.subarray(i * 10, i * 10 + 3), i * 3);
      nrm.set(data.subarray(i * 10 + 3, i * 10 + 6), i * 3);
      col.set(data.subarray(i * 10 + 6, i * 10 + 9), i * 3);
      body[i] = data[i * 10 + 9];
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('rubbleBody', new THREE.BufferAttribute(body, 1));
    if (indices) g.setIndex(new THREE.BufferAttribute(indices, 1));
    g.instanceCount = copies;
    const id = attribute('rubbleBody', 'uint');
    const index = select(id.equal(uint(0)), uint(0), id.add(instanceIndex.mul(uint(stride))));
    const p = bodies.element(index.mul(BODY_VEC4));
    const q = bodies.element(index.mul(BODY_VEC4).add(1));
    const invI = bodies.element(index.mul(BODY_VEC4).add(2));
    const v = bodies.element(index.mul(BODY_VEC4).add(3));
    const local = attribute('position', 'vec3'), nl = attribute('normal', 'vec3');
    let world = p.xyz.add(qrot(q, local));
    if (lines) {
      // A bond line: normal = (-(partner + 2), rest distance, rest height). Hidden once its
      // joint breaks, or (no joints) once its pieces part.
      let parted;
      if (jointsPerCopy > 0) {
        const j = vertexIndex.div(uint(2)).add(instanceIndex.mul(uint(jointsPerCopy)));
        parted = joints.element(j.mul(uint(JOINT_WORDS)).add(uint(2))).bitAnd(uint(JOINT_FLAG_BROKEN)).notEqual(uint(0));
      } else {
        const partner = uint(nl.x.negate().sub(2).add(0.5));
        const slack = nl.y.mul(0.02).add(0.02);
        const other = bodies.element(partner.add(instanceIndex.mul(uint(stride))).mul(BODY_VEC4)).xyz;
        parted = select(partner.equal(uint(0)), abs(p.y.sub(nl.z)).greaterThan(slack), abs(length(other.sub(p.xyz)).sub(nl.y)).greaterThan(slack));
      }
      world = select(parted, vec3(0, -1e6, 0), world);
    }
    const material = lit ? new THREE.MeshStandardNodeMaterial({ roughness: 0.8, metalness: 0.05 }) : new THREE.LineBasicNodeMaterial();
    material.positionNode = world;
    if (lit) material.normalNode = varying(normalize(cameraViewMatrix.mul(vec4(qrot(q, nl), 0)).xyz));
    const base = attribute('color', 'vec3');
    const flags = floatBitsToUint(invI.w);
    const asleep = flags.bitAnd(uint(4)).notEqual(uint(0));
    const sleepC = select(flags.bitAnd(uint(1)).notEqual(uint(0)).and(asleep.not()), vec3(1, 0.45, 0.12), select(asleep, vec3(0.25, 0.45, 0.85), vec3(0.45, 0.45, 0.48)));
    const slot = poseStream ? floatBitsToUint(p.w) : index.add(uint(1));
    const bodiesC = select(slot.notEqual(uint(0)), slotColor(slot.sub(uint(1))), base);
    const stressC = poseStream ? select(v.x.greaterThan(0), stressColor(v.x.sub(1)), vec3(0.42, 0.44, 0.47)) : base;
    const ground = index.equal(uint(0));
    material.colorNode = varying(select(ground.or(colorMode.equal(uint(0))), base,
      select(colorMode.equal(uint(1)), sleepC, select(colorMode.equal(uint(2)), bodiesC, stressC))));
    const mesh = lines ? new THREE.LineSegments(g, material) : new THREE.Mesh(g, material);
    mesh.frustumCulled = false;
    mesh.castShadow = !lines;
    mesh.receiveShadow = !lines;
    view.add(mesh);
    meshes.push(mesh);
    return mesh;
  }

  // Streams drawn once name their bodies by the simulation's slots (`remap`: the World's).
  const once = (data) => (simulation.remap ? simulation.remap(data) : data);
  const indexed = new IndexedMesh(scene.vertices);
  const main = bodyMesh(indexed.vertices(), { indices: indexed.index(), copies: scene.copies || 1 });
  indexed.free();
  const projectiles = bodyMesh(once(scene.projectileVertices));
  const perBall = scene.projectileCount ? scene.projectileVertices.length / 10 / scene.projectileCount : 0;
  if (scene.propVertices?.length) bodyMesh(once(scene.propVertices));
  if (scene.roadVertices?.length) {
    // Road decals lie millimetres above the ground: a depth offset keeps them in front.
    const m = bodyMesh(scene.roadVertices).material;
    Object.assign(m, { polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -16 });
  }
  const bonds = scene.lines?.length ? bodyMesh(scene.lines, { lines: true, lit: false, copies: scene.copies || 1 }) : null;
  if (bonds) bonds.visible = false;

  // The ground: a large plane with a world-space grid.
  {
    const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.95 });
    const uv = positionWorld.xz;
    const gr = abs(fract(uv.sub(0.5)).sub(0.5));
    const line = smoothstep(0.46, 0.5, max(gr.x, gr.y));
    material.colorNode = mix(vec3(0.33, 0.35, 0.37), vec3(0.45, 0.48, 0.5), line.mul(0.6));
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(extent * 60, extent * 60), material);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.002;
    plane.receiveShadow = true;
    view.add(plane);
  }

  // Trails (8 floats per vertex: xyz, pad, rgba) and zone outlines (render vertices).
  const trails = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
  trails.frustumCulled = false;
  view.add(trails);
  let zones = null, dyn = null, dynData = null;

  const orbit = new Orbit(sceneHandle(scene));
  const aspect = () => canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const shotAt = (u, v) => orbit.aim(u, v, aspect(), scene.projectileRadius);
  const input = attachCameraInput(canvas, orbit, (u, v) => onShoot(shotAt(u, v)));
  let width = 0, height = 0, gpuMs = null;

  return {
    aim(clientX, clientY) { const r = canvas.getBoundingClientRect(); return shotAt((clientX - r.left) / r.width, (clientY - r.top) / r.height); },
    setZones(vertices) { if (zones) { view.remove(zones); meshes.splice(meshes.indexOf(zones), 1); } zones = vertices.length ? bodyMesh(vertices, { lines: true, lit: false }) : null; },
    setTrails(data) {
      const n = data.length / 8, pos = new Float32Array(n * 3), col = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) { pos.set(data.subarray(i * 8, i * 8 + 3), i * 3); col.set(data.subarray(i * 8 + 4, i * 8 + 8), i * 4); }
      trails.geometry.dispose();
      trails.geometry = new THREE.BufferGeometry();
      trails.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      trails.geometry.setAttribute('color', new THREE.BufferAttribute(col, 4));
    },
    setDynamic(data) {
      if (dyn) { view.remove(dyn); meshes.splice(meshes.indexOf(dyn), 1); dyn.geometry.dispose(); }
      dynData = data.slice();
      dyn = data.length ? bodyMesh(dynData) : null;
    },
    appendDynamic(data, at) {
      // Rebuilt whole (cheap at debris sizes): the kept prefix plus the new vertices.
      const keep = dynData ? dynData.subarray(0, at * 10) : new Float32Array(0);
      const next = new Float32Array(keep.length + data.length);
      next.set(keep); next.set(data, keep.length);
      this.setDynamic(next);
      return true;
    },
    orbit(dx, dy) { orbit.drag(dx, dy); }, zoom(delta) { orbit.zoom(delta); },
    lookAt(eye, target) { orbit.lookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2]); input.modeChanged(); },
    orbitFrom(eye, target) { orbit.orbitFrom(eye[0], eye[1], eye[2], target[0], target[1], target[2]); input.modeChanged(); },
    get viewMode() { return orbit.mode; },
    set viewMode(mode) { orbit.setMode(mode | 0, aspect()); input.modeChanged(); },
    get gpuFrameMs() { return gpuMs; },
    set colorMode(mode) { colorMode.value = mode | 0; },
    async profileDraw(showBonds = false) {
      const t = performance.now();
      this.draw(showBonds);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - t;
    },
    draw(showBonds) {
      const dpr = Math.min(devicePixelRatio, 2), w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (w !== width || h !== height) { width = w; height = h; renderer.setPixelRatio(dpr); renderer.setSize(canvas.clientWidth, canvas.clientHeight, false); }
      input.update();
      const [ex, ey, ez, fx, fy, fz, fov, near, far] = orbit.camera(w / h);
      camera.fov = (fov * 180) / Math.PI; camera.near = near; camera.far = far; camera.aspect = w / h;
      camera.position.set(ex, ey, ez);
      camera.lookAt(ex + fx, ey + fy, ez + fz);
      camera.updateProjectionMatrix();
      projectiles.visible = !!simulation.shots;
      projectiles.geometry.setDrawRange(0, Math.min(simulation.shots, scene.projectileCount) * perBall);
      if (bonds) bonds.visible = !!showBonds;
      const t = performance.now();
      renderer.render(view, camera);
      gpuMs = performance.now() - t;
    },
    dispose() { input.dispose(); orbit.free(); for (const m of meshes) { m.geometry.dispose(); m.material.dispose(); } renderer.dispose(); },
  };
}
