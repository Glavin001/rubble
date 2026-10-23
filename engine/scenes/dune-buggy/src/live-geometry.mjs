import * as T from 'three';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { buildBuggy } from './buggy.mjs';

const Z = new T.Vector3(0, 0, 1);
const one = new T.Vector3(1, 1, 1);
const vec = a => new T.Vector3(...a);
const pose = (c, rot = [0, 0, 0], scale = [1, 1, 1]) => new T.Matrix4().compose(
  vec(c), new T.Quaternion().setFromEuler(new T.Euler(...rot.map(T.MathUtils.degToRad), 'ZYX')), vec(scale));

class Solid {
  constructor(geometry, matrix = new T.Matrix4()) { this.geometry = geometry; this.matrix = matrix; }
  transform(elements) { return new Solid(this.geometry, new T.Matrix4().fromArray(elements).multiply(this.matrix)); }
  rotate(rot) { return new Solid(this.geometry, pose([0, 0, 0], rot).multiply(this.matrix)); }
  translate(c) { return new Solid(this.geometry, new T.Matrix4().makeTranslation(...c).multiply(this.matrix)); }
  delete() {} // Geometry is owned by the persistent template cache.
}

/** The same part recipe, with reusable full-detail render primitives. No CSG,
 * worker transfer, mesh installation, debounce or LOD swap on dimension edits.
 * This visual representation is deliberately never used for game exports.
 */
export class LiveGeometry {
  constructor() { this.templates = new Map(); this.dynamic = new Map(); }
  begin() { this.parts = []; this.springIndex = 0; this.polyIndex = 0; }
  template(key, create) {
    if (!this.templates.has(key)) {
      const g = create();
      if (!g.attributes.uv) {
        const uv = new Float32Array(g.attributes.position.count * 2);
        for (let i = 0; i < g.attributes.position.count; i++) {
          uv[i * 2] = g.attributes.position.getX(i) + g.attributes.position.getY(i) * .71;
          uv[i * 2 + 1] = g.attributes.position.getZ(i) + g.attributes.position.getY(i) * .43;
        }
        g.setAttribute('uv', new T.BufferAttribute(uv, 2));
      }
      g.computeBoundingBox(); g.computeBoundingSphere(); this.templates.set(key, g);
    }
    return this.templates.get(key);
  }
  cube = (size, c, rot = [0, 0, 0], radius = 0) => {
    const key = 'box:' + [...size, radius].join(',');
    const geometry = this.template(key, () => {
      if (!radius) return new T.BoxGeometry(...size);
      const h = size.map(x => x / 2), r = Math.min(radius, ...h.map(x => x * .45)), points = [];
      for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) for (let k = 0; k < 3; k++)
        points.push(vec([x * (h[0] - (k === 0 ? 0 : r)), y * (h[1] - (k === 1 ? 0 : r)), z * (h[2] - (k === 2 ? 0 : r))]));
      return new ConvexGeometry(points);
    });
    return new Solid(geometry, pose(c, rot));
  };
  poly = points => {
    const key='panel:'+this.polyIndex++,signature=JSON.stringify(points);
    const geometry=this.template(key,()=>new ConvexGeometry(points.map(vec)));
    if(this.dynamic.get(key)!==signature){
      const next=new ConvexGeometry(points.map(vec));
      for(const name of ['position','normal']){
        if(geometry.attributes[name].array.length===next.attributes[name].array.length){geometry.attributes[name].copyArray(next.attributes[name].array);geometry.attributes[name].needsUpdate=true;}
        else geometry.setAttribute(name,next.attributes[name].clone());
      }
      if(geometry.attributes.uv.count!==geometry.attributes.position.count)geometry.setAttribute('uv',new T.BufferAttribute(new Float32Array(geometry.attributes.position.count*2),2));
      for(let i=0;i<geometry.attributes.position.count;i++)geometry.attributes.uv.setXY(i,geometry.attributes.position.getX(i)+geometry.attributes.position.getY(i)*.71,geometry.attributes.position.getZ(i)+geometry.attributes.position.getY(i)*.43);
      geometry.attributes.uv.needsUpdate=true;geometry.computeBoundingBox();geometry.computeBoundingSphere();next.dispose();this.dynamic.set(key,signature);
    }
    return new Solid(geometry);
  };
  cyl = (radius, length, c, rot = [90, 0, 0], n = 32) => {
    const geometry = this.template('cylinder:' + n, () => new T.CylinderGeometry(1, 1, 1, n).rotateX(Math.PI / 2).rotateZ(Math.PI / 2));
    return new Solid(geometry, pose(c, rot, [radius, radius, length]));
  };
  ring = (outer, inner, length, c, rot = [0, 90, 0], n = 64) => {
    const ratio = inner / outer;
    const geometry = this.template(`ring:${ratio}:${n}`, () => {
      const shape = new T.Shape(); shape.absarc(0, 0, 1, 0, Math.PI * 2, false);
      const hole = new T.Path(); hole.absarc(0, 0, ratio, 0, Math.PI * 2, true); shape.holes.push(hole);
      const raw = new T.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false, curveSegments: n / 2, steps: 1 }).translate(0, 0, -.5);
      const smooth = toCreasedNormals(raw, Math.PI / 3); raw.dispose(); return smooth;
    });
    return new Solid(geometry, pose(c, rot, [outer, outer, length]));
  };
  beam = (a, b, radius = .026, hollow = false) => {
    const A = vec(a), B = vec(b), d = B.clone().sub(A);
    const solid = hollow ? this.ring(radius, radius - .003, d.length(), [0, 0, 0], [0, 0, 0], 32)
      : this.cyl(radius, d.length(), [0, 0, 0], [0, 0, 0], 32);
    const matrix = new T.Matrix4().compose(A.add(B).multiplyScalar(.5), new T.Quaternion().setFromUnitVectors(Z, d.normalize()), one);
    solid.matrix.premultiply(matrix); return solid;
  };
  sphere(radius, n) {
    return new Solid(this.template('sphere:' + n, () => new T.SphereGeometry(1, n, n / 2)), pose([0, 0, 0], [0, 0, 0], [radius, radius, radius]));
  }
  revolve(profile, n) {
    const key = 'tire', signature = JSON.stringify(profile);
    const geometry = this.template(key, () => {
      const closed = [...profile, profile[0]].map(([r, z]) => new T.Vector2(r, z));
      return new T.LatheGeometry(closed, n).rotateX(Math.PI / 2);
    });
    if (this.dynamic.get(key) !== signature) {
      const next = new T.LatheGeometry([...profile, profile[0]].map(([r, z]) => new T.Vector2(r, z)), n).rotateX(Math.PI / 2);
      geometry.attributes.position.copyArray(next.attributes.position.array);
      geometry.attributes.normal.copyArray(next.attributes.normal.array);
      geometry.attributes.position.needsUpdate = geometry.attributes.normal.needsUpdate = true;
      geometry.computeBoundingBox(); geometry.computeBoundingSphere(); next.dispose(); this.dynamic.set(key, signature);
    }
    return new Solid(geometry);
  }
  spring(top, bottom) {
    const key = 'spring:' + this.springIndex++, d = vec(bottom).sub(vec(top)), length = d.length();
    // Keep all four geometries resident. Only length changes require a curve update.
    const create = () => {
      const points = [];
      for (let k = 0; k <= 240; k++) { const t = k / 240, a = t * Math.PI * 20; points.push(new T.Vector3(Math.cos(a) * .059, -Math.sin(a) * .059, t * length * .68)); }
      return new T.TubeGeometry(new T.CatmullRomCurve3(points), 320, .009, 8, false);
    };
    const existed = this.templates.has(key), geometry = this.template(key, create);
    if (existed && this.dynamic.get(key) !== length) {
      const next = create();
      geometry.attributes.position.copyArray(next.attributes.position.array); geometry.attributes.normal.copyArray(next.attributes.normal.array);
      geometry.attributes.position.needsUpdate = geometry.attributes.normal.needsUpdate = true;
      geometry.computeBoundingBox(); geometry.computeBoundingSphere(); next.dispose();
    }
    this.dynamic.set(key, length);
    // Match the recipe's helix orientation: local (cos, sin, axial) -> (cos, axial, sin).
    const orientation = new T.Quaternion().setFromUnitVectors(new T.Vector3(0, 1, 0), d.normalize());
    const matrix = new T.Matrix4().compose(vec(top), orientation, one).multiply(new T.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1));
    // The reflection reverses winding. Use an equivalent positive-determinant
    // frame, and reverse the helix's Y coordinate in its geometry instead.
    matrix.multiply(new T.Matrix4().makeScale(1, -1, 1));
    return new Solid(geometry, matrix);
  }
  add(name, system, material, solid, motion = null) {
    const id = `${system.toLowerCase()}-${String(this.parts.length).padStart(4, '0')}`;
    const bounds = solid.geometry.boundingBox.clone().applyMatrix4(solid.matrix), center = bounds.getCenter(new T.Vector3()).toArray();
    this.parts.push({ id, name, system, material, motion, geometry: solid.geometry, matrix: solid.matrix, center, bounds: { min: bounds.min.toArray(), max: bounds.max.toArray() } });
    return id;
  }
  finish(parameters) {
    return { parameters, parts: this.parts, joints: [], quality: 'live', detail: 'high', units: 'metres', axes: { up: '+Y', front: '-Z' },
      report: { parts: this.parts.length, triangles: this.parts.reduce((n, p) => n + (p.geometry.index?.count ?? p.geometry.attributes.position.count) / 3, 0), joints: null, components: null } };
  }
  build(parameters) { return buildBuggy(null, parameters, undefined, false, { live: this }); }
  dispose() { for (const g of this.templates.values()) g.dispose(); this.templates.clear(); }
}

/** How a batch of repeated parts is drawn: an InstancedMesh, unless the
 * renderer registers something else (the WebGPU path draws shader-shareable
 * batches; vehicles/partBatchNodes.ts). */
let createBatch = (geometry, material, count) => new T.InstancedMesh(geometry, material, count);
export function setPartBatchFactory(factory) { createBatch = factory; }
/** A batch of `count` instances of `geometry`, as the assembly draws its parts. */
export function createPartBatchMesh(geometry, material, count) { return createBatch(geometry, material, count); }

const HIDDEN = new T.Matrix4().makeScale(0, 0, 0);

/** Group repeated geometry into GPU instances. A resize changes matrices in the
 * existing buffers; each instance still maps to a stable, selectable part ID. */
export class LiveAssembly {
  constructor(group, materials) {
    this.group = group; this.materials = materials; this.batches = new Map(); this.meshes = [];
    /** Part id -> { mesh, index, part }: where each part is drawn. */
    this.slots = new Map();
    /** Parts not drawn on the car (broken off: drawn as loose bodies instead). */
    this.hidden = new Set();
  }
  update(model, explosion, hidden, explosionCenters) {
    const buckets = new Map();
    for (const part of model.parts) {
      const key = part.geometry.uuid + '|' + part.system + '|' + part.material;
      if (!buckets.has(key)) buckets.set(key, []); buckets.get(key).push(part);
    }
    for (const [key, batch] of this.batches) if (!buckets.has(key)) { this.group.remove(batch); batch.dispose(); this.batches.delete(key); }
    const matrix = new T.Matrix4(), direction = new T.Vector3();
    for (const [key, parts] of buckets) {
      let mesh = this.batches.get(key);
      if (!mesh || mesh.instanceMatrix.count < parts.length) {
        if(mesh){this.group.remove(mesh);mesh.dispose();}
        mesh = createBatch(parts[0].geometry, this.materials[parts[0].material], parts.length);
        mesh.instanceMatrix.setUsage(T.DynamicDrawUsage); mesh.castShadow = mesh.receiveShadow = true;
        // Bounds move continuously, so skip expensive per-frame aggregate bounds.
        mesh.frustumCulled = false; this.batches.set(key, mesh); this.group.add(mesh);
      }
      mesh.count=parts.length;
      mesh.userData = { system: parts[0].system, material: parts[0].material, parts }; mesh.visible = !hidden.has(parts[0].system);
      for (let i = 0; i < parts.length; i++) {
        matrix.copy(parts[i].matrix); const c = explosionCenters?.get(parts[i].id) ?? parts[i].center;
        direction.set(c[0], c[1] - .65, c[2]); if (direction.lengthSq() < .01) direction.set(0, 1, 0);
        direction.normalize().multiplyScalar(explosion * .9); matrix.elements[12] += direction.x; matrix.elements[13] += direction.y; matrix.elements[14] += direction.z;
        mesh.setMatrixAt(i, matrix);
      }
      mesh.instanceMatrix.needsUpdate = true; mesh.boundingSphere = null;
    }
    this.meshes = [...this.batches.values()];
    this.slots = new Map();
    for (const mesh of this.meshes) mesh.userData.parts.forEach((part, index) => this.slots.set(part.id, { mesh, index, part }));
    this.hidden.clear();
  }
  bindMotion() {
    this.motionSlots = [];
    for (const mesh of this.meshes) for (let index = 0; index < mesh.userData.parts.length; index++) {
      const part = mesh.userData.parts[index];
      if (part.motion) this.motionSlots.push({ mesh, index, part });
    }
  }
  /** Pose the moving parts (suspension, wheels, steering) on the car. A part
   * that broke off is a body of its own and stays hidden here. */
  applyMotion(rig) {
    const touched = new Set();
    for (const { mesh, index, part } of this.motionSlots) {
      if (this.hidden.has(part.id)) continue;
      mesh.setMatrixAt(index, rig.matrixFor(part));
      touched.add(mesh);
    }
    for (const mesh of touched) mesh.instanceMatrix.needsUpdate = true;
  }
  /** Stop drawing a part on the car (it broke off). */
  hidePart(id) {
    const slot = this.slots.get(id); if (!slot || this.hidden.has(id)) return;
    this.hidden.add(id);
    slot.mesh.setMatrixAt(slot.index, HIDDEN); slot.mesh.instanceMatrix.needsUpdate = true;
  }
  /** Draw a part on the car again, at `matrix` (its rest or motion pose). */
  showPart(id, matrix) {
    const slot = this.slots.get(id); if (!slot || !this.hidden.delete(id)) return;
    slot.mesh.setMatrixAt(slot.index, matrix); slot.mesh.instanceMatrix.needsUpdate = true;
  }
  dispose() { for (const mesh of this.meshes) { this.group.remove(mesh); mesh.dispose(); } this.batches.clear(); this.meshes = []; }
}
