// Rubble in three.js: draw a Rubble World with three's WebGPURenderer, reading the
// World's own GPU buffers in place (no readback, no copy). Physics and scene setup stay
// in Rust; this file only turns the World's render manifest (`world.renderManifest()`,
// engine-world `render_manifest.rs`) into three objects whose vertex stage reads each
// collider's body pose straight from the engine's body and collider buffers.
//
//   import * as THREE from 'three/webgpu';
//   const renderer = new THREE.WebGPURenderer({ device: world.device(), antialias: true });
//   await renderer.init();
//   const rubble = new RubbleThree(renderer, world);
//   scene.add(rubble.object);
//   // every frame, after stepping:
//   rubble.update();
//   renderer.render(scene, camera);
//
// The renderer must run on the World's device (`world.device()`): three binds the
// engine's buffers, which belong to that device.
import * as THREE from 'three/webgpu';
import {
  Fn, storage, instanceIndex, attribute, vec3, vec4, float, uint, varying, positionLocal, normalLocal,
  cameraViewMatrix, floatBitsToUint, uniform, mix, hash, select, normalize, cross, dot, length, clamp, fract,
} from 'three/tsl';

/** Engine record layouts in vec4 units (engine-schema `Body`, 112 bytes; `Collider`, 64). */
export const BODY_VEC4 = 7;
export const COLLIDER_VEC4 = 4;

/**
 * A three storage attribute over a GPUBuffer three did not create: three binds
 * `backend.get(attribute).buffer` when it is set, so the engine's buffer is used as is.
 * `type` is the WGSL element ('vec4' or 'uvec4').
 */
export function externalStorage(renderer, gpuBuffer, type = 'vec4') {
  const array = type === 'uvec4' ? new Uint32Array(4) : new Float32Array(4);
  const attr = new THREE.StorageBufferAttribute(array, 4);
  renderer.backend.get(attr).buffer = gpuBuffer;
  return storage(attr, type, 0).toReadOnly();
}

// q · v (unit quaternion xyzw).
const quatRotate = (q, v) => {
  const u = q.xyz;
  return u.mul(dot(u, v).mul(2)).add(v.mul(q.w.mul(q.w).sub(dot(u, u)))).add(cross(u, v).mul(q.w.mul(2)));
};

/** Default look per material tag (a tag the manifest names; unknown tags get `default`). */
export const DEFAULT_LOOKS = {
  default: { color: 0xb5653a, vary: 0.35, roughness: 0.8, metalness: 0 },
  brick: { color: 0xa4512f, vary: 0.35, roughness: 0.85, metalness: 0 },
  concrete: { color: 0x9a968e, vary: 0.15, roughness: 0.9, metalness: 0 },
  wood: { color: 0x8a5a2b, vary: 0.3, roughness: 0.75, metalness: 0 },
  glass: { color: 0x9fc6d6, vary: 0.05, roughness: 0.05, metalness: 0, opacity: 0.45 },
  steel: { color: 0xa8adb3, vary: 0.05, roughness: 0.35, metalness: 0.9 },
  ball: { color: 0xc0c6cf, vary: 0, roughness: 0.3, metalness: 0.8 },
  ground: { color: 0x3a3f46, vary: 0, roughness: 0.95, metalness: 0 },
};

/** Debug colourings shared by every group (`rubble.colorMode`). */
export const COLOR_MODES = { material: 0, sleep: 1, speed: 2 };

export class RubbleThree {
  /**
   * @param {THREE.WebGPURenderer} renderer on `world.device()`
   * @param {object} world a WebWorld (engine-world-wasm)
   * @param {{ looks?: object, castShadow?: boolean, receiveShadow?: boolean }} [opts]
   */
  constructor(renderer, world, opts = {}) {
    if (renderer.backend.device !== world.device()) {
      throw new Error('RubbleThree: the renderer must run on the World\'s device (new WebGPURenderer({ device: world.device() }))');
    }
    this.renderer = renderer;
    this.world = world;
    this.looks = { ...DEFAULT_LOOKS, ...(opts.looks ?? {}) };
    this.castShadow = opts.castShadow ?? true;
    this.receiveShadow = opts.receiveShadow ?? true;
    /** Everything Rubble draws; add it to a scene. */
    this.object = new THREE.Group();
    this.object.name = 'rubble';
    /** 0 material, 1 awake/asleep, 2 speed (COLOR_MODES). */
    this.colorMode = uniform(0, 'uint');
    this._version = -1;
    this._buffers = [];
    this._meshes = [];
    this.update();
  }

  /** Rebuild when the World's shapes or buffers changed; cheap otherwise. Call each frame. */
  update() {
    const buffers = [this.world.bodyBuffer(), this.world.colliderBuffer()];
    const version = this.world.renderVersion();
    if (version === this._version && buffers.every((b, i) => b === this._buffers[i])) return false;
    this._version = version;
    this._buffers = buffers;
    this._build(this.world.renderManifest(), buffers);
    return true;
  }

  /** Draw call and instance totals (for HUDs). */
  stats() {
    let instances = 0, vertices = 0;
    for (const m of this._meshes) {
      instances += m.geometry.instanceCount ?? 1;
      vertices += m.geometry.attributes.position.count;
    }
    return { draws: this._meshes.length, instances, vertices };
  }

  dispose() {
    for (const m of this._meshes) {
      this.object.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    }
    this._meshes = [];
  }

  _build(manifest, [bodyBuf, colliderBuf]) {
    this.dispose();
    const bodies = externalStorage(this.renderer, bodyBuf, 'vec4');
    const collidersF = externalStorage(this.renderer, colliderBuf, 'vec4');
    const collidersU = externalStorage(this.renderer, colliderBuf, 'uvec4');
    const ctx = { bodies, collidersF, collidersU };
    for (const g of manifest.groups) {
      const mesh = g.kind === 'mesh' ? this._meshGroup(g, ctx) : this._primitiveGroup(g, ctx);
      if (!mesh) continue;
      mesh.name = `rubble:${g.kind}:${g.tag}`;
      mesh.frustumCulled = false;
      mesh.castShadow = this.castShadow && g.tag !== 'ground';
      mesh.receiveShadow = this.receiveShadow;
      this.object.add(mesh);
      this._meshes.push(mesh);
    }
  }

  // Box, sphere and capsule groups: one unit primitive, instanced over the group's
  // colliders (`g.colliders`), scaled by each collider's params on the GPU.
  _primitiveGroup(g, ctx) {
    const n = g.colliders.length;
    if (n === 0) return null;
    let base;
    if (g.kind === 'box') base = new THREE.BoxGeometry(2, 2, 2);
    else if (g.kind === 'sphere') base = new THREE.SphereGeometry(1, 16, 12);
    else if (g.kind === 'capsule') base = new THREE.CapsuleGeometry(1, 2, 6, 12); // radius 1, segment 2 (half 1)
    else return null;
    const geometry = new THREE.InstancedBufferGeometry().copy(base);
    base.dispose();
    geometry.instanceCount = n;
    const ids = new THREE.StorageBufferAttribute(new Uint32Array(g.colliders), 1);
    const idNode = storage(ids, 'uint', n).toReadOnly();
    const collider = idNode.element(instanceIndex);
    const kind = g.kind;
    const local = Fn(([params]) => {
      const p = positionLocal;
      if (kind === 'box') return p.mul(params.xyz);
      if (kind === 'sphere') return p.mul(params.x);
      // Capsule: hemispheres moved out to ±half length, then scaled by the radius.
      const ySign = select(p.y.greaterThan(0), float(1), float(-1));
      return vec3(p.x.mul(params.x), p.y.sub(ySign).mul(params.x).add(ySign.mul(params.y)), p.z.mul(params.x));
    });
    const localNormal = (params) => (kind === 'box' ? normalLocal : normalLocal);
    return this._material(geometry, g, ctx, collider, local, localNormal);
  }

  // Hulls (and any other mesh the manifest gives): one merged geometry, each vertex
  // carrying its collider; vertices are in the collider's frame.
  _meshGroup(g, ctx) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(g.positions), 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(g.normals), 3));
    geometry.setAttribute('rubbleCollider', new THREE.BufferAttribute(new Uint32Array(g.vertexColliders), 1));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(g.indices), 1));
    const collider = attribute('rubbleCollider', 'uint');
    const local = Fn(() => positionLocal);
    return this._material(geometry, g, ctx, collider, local, () => normalLocal);
  }

  _material(geometry, g, ctx, collider, local, localNormal) {
    const look = this.looks[g.tag] ?? this.looks.default;
    const material = new THREE.MeshStandardNodeMaterial({
      roughness: look.roughness, metalness: look.metalness,
      transparent: look.opacity !== undefined, opacity: look.opacity ?? 1,
    });
    const { bodies, collidersF, collidersU } = ctx;
    const head = collidersU.element(collider.mul(COLLIDER_VEC4));
    const bodyIdx = head.x;
    const localPos = collidersF.element(collider.mul(COLLIDER_VEC4).add(1)).xyz;
    const localRot = collidersF.element(collider.mul(COLLIDER_VEC4).add(2));
    const params = collidersF.element(collider.mul(COLLIDER_VEC4).add(3));
    const b0 = bodies.element(bodyIdx.mul(BODY_VEC4));
    const bodyRot = bodies.element(bodyIdx.mul(BODY_VEC4).add(1));
    const b2 = bodies.element(bodyIdx.mul(BODY_VEC4).add(2));
    const b3 = bodies.element(bodyIdx.mul(BODY_VEC4).add(3));
    // World = body.pos + body.q · (collider.localPos + collider.localRot · v): the
    // collider record is already in the body's solver frame (engine-world `collider`).
    const worldPos = b0.xyz.add(quatRotate(bodyRot, localPos.add(quatRotate(localRot, local(params)))));
    const worldNormal = quatRotate(bodyRot, quatRotate(localRot, localNormal(params)));
    material.positionNode = worldPos;
    material.normalNode = varying(normalize(cameraViewMatrix.mul(vec4(worldNormal, 0)).xyz));
    const base = new THREE.Color(look.color);
    const tint = varying(hash(bodyIdx).mul(2).sub(1).mul(look.vary ?? 0));
    const lit = vec3(base.r, base.g, base.b).mul(tint.mul(0.5).add(1));
    const asleep = floatBitsToUint(b2.w).bitAnd(uint(4)).notEqual(uint(0));
    const sleepColor = varying(select(asleep, vec3(0.28, 0.48, 0.82), vec3(0.95, 0.58, 0.16)));
    const speed = varying(clamp(length(b3.xyz).div(3), 0, 1));
    const speedColor = mix(vec3(0.22, 0.32, 0.48), vec3(0.95, 0.22, 0.16), speed);
    material.colorNode = select(this.colorMode.equal(uint(1)), sleepColor,
      select(this.colorMode.equal(uint(2)), speedColor, lit));
    return new THREE.Mesh(geometry, material);
  }
}
