import * as T from 'three';
import { createRigDefinition, cornerIds, solveCorner, neutralPose } from './vehicle-rig.mjs';
const X = new T.Vector3(1, 0, 0), Y = new T.Vector3(0, 1, 0), Z = new T.Vector3(0, 0, 1), ONE = new T.Vector3(1, 1, 1);
const v = a => new T.Vector3(...a);
const pivot = (out, point, rotation) => out.makeTranslation(...point).multiply(new T.Matrix4().makeRotationFromQuaternion(rotation)).multiply(new T.Matrix4().makeTranslation(-point[0], -point[1], -point[2]));
/** GPU-independent pose solver + Three.js matrix adapter. No world step or filters. */
export class VisualRig {
  constructor(model) {
    this.definition = createRigDefinition(model.parameters, model.parts);
    this.states = {}; this.parts = model.parts; this.pose = neutralPose();
    this.result = new T.Matrix4(); this.a = new T.Vector3(); this.b = new T.Vector3(); this.d = new T.Vector3(); this.q = new T.Quaternion(); this.scale = new T.Vector3();
    this.steeringDelta = new T.Matrix4();
    this.springs = new Map(model.parts.filter(p => p.motion?.role === 'spring').map(p => [p.motion.corner, { geometry: p.geometry, lastLength: -1, originalPosition: p.geometry.attributes.position.array.slice(), originalNormal: p.geometry.attributes.normal.array.slice() }]));
    this.springFrames = Array.from({ length: 321 }, (_, i) => { const a = i / 320 * Math.PI * 20; return [Math.cos(a), Math.sin(a), i / 320]; });
    this.wireFrames = Array.from({ length: 9 }, (_, i) => [Math.cos(i / 8 * Math.PI * 2), Math.sin(i / 8 * Math.PI * 2)]);
    this.applyPose(this.pose);
  }
  applyPose(pose) {
    this.pose = pose;
    for (const id of cornerIds) {
      const h = this.definition.corners[id], input = pose.wheels[id], k = solveCorner(h, input.travelM, input.steeringRad);
      const lower = v(k.lower), upper = v(k.upper), neutralLower = v(h.lower);
      const camberQ = new T.Quaternion().setFromAxisAngle(Z, k.camber);
      const camber = new T.Matrix4().compose(lower, camberQ, ONE).multiply(new T.Matrix4().makeTranslation(...neutralLower.clone().negate().toArray()));
      const axis = upper.clone().sub(lower).normalize(), steerQ = new T.Quaternion().setFromAxisAngle(axis, h.front ? input.steeringRad : 0);
      const steer = pivot(new T.Matrix4(), k.lower, steerQ).multiply(camber);
      const hub = v(h.hub).applyMatrix4(steer), wheelAxis = X.clone().applyQuaternion(camberQ).applyQuaternion(steerQ);
      const spinQ = new T.Quaternion().setFromAxisAngle(wheelAxis, input.rotationRad);
      const wheel = pivot(new T.Matrix4(), hub.toArray(), spinQ).multiply(steer);
      const armAngle = (k.angle - h.neutralAngle) * h.side;
      const bottom = v(h.shockBottom).sub(v(h.lowerPivot)).applyAxisAngle(Z, armAngle).add(v(h.lowerPivot));
      const top = v(h.shockTop), shockDir = bottom.clone().sub(top).normalize(), length = top.distanceTo(bottom);
      const shockQ = new T.Quaternion().setFromUnitVectors(Y, shockDir);
      const springMatrix = new T.Matrix4().compose(top, shockQ, ONE).multiply(new T.Matrix4().set(1,0,0,0, 0,0,1,0, 0,-1,0,0, 0,0,0,1));
      const coilLength = length - .32 * h.shockLength;
      const tieOuter = v(h.tieOuter).applyMatrix4(steer), tieInner = v(h.tieInner);
      const rodLength = v(h.tieOuter).distanceTo(tieInner);
      // Each virtual rack end slides in X; the exposed tie rod retains its length.
      tieInner.x = tieOuter.x - h.side * Math.sqrt(Math.max(.0001, rodLength ** 2 - (tieOuter.y - tieInner.y) ** 2 - (tieOuter.z - tieInner.z) ** 2));
      const axleDir = hub.clone().sub(v(h.axleInner)).normalize();
      this.states[id] = { h, k, lower, upper, hub, steer, wheel, top, bottom, shockDir, length, coilLength, springMatrix, tieOuter, tieInner, axleDir, wheelAxis, rotation: input.rotationRad };
      this.updateSpring(id, coilLength);
    }
    const { columnStart, columnEnd } = this.definition.steering;
    const q = new T.Quaternion().setFromAxisAngle(v(columnEnd).sub(v(columnStart)).normalize(), pose.steeringWheelRad ?? ((pose.wheels.fl.steeringRad + pose.wheels.fr.steeringRad) * .5 * this.definition.steering.ratio));
    pivot(this.steeringDelta, columnEnd, q);
  }
  restore() {
    for (const s of this.springs.values()) {
      s.geometry.attributes.position.array.set(s.originalPosition); s.geometry.attributes.normal.array.set(s.originalNormal);
      s.geometry.attributes.position.needsUpdate = s.geometry.attributes.normal.needsUpdate = true;
    }
  }
  updateSpring(id, length) {
    const spring = this.springs.get(id); if (!spring || Math.abs(spring.lastLength - length) < 1e-7) return;
    const p = spring.geometry.attributes.position, n = spring.geometry.attributes.normal;
    const pitch = .059 * Math.PI * 20, inv = 1 / Math.hypot(length, pitch);
    let index = 0;
    for (const [ca, sa, t] of this.springFrames) for (const [cv, sv] of this.wireFrames) {
      const nx = ca * cv + length * sa * inv * sv, ny = -sa * cv + length * ca * inv * sv, nz = pitch * inv * sv;
      p.setXYZ(index, .059 * ca + .009 * nx, -.059 * sa + .009 * ny, t * length + .009 * nz);
      n.setXYZ(index++, nx, ny, nz);
    }
    p.needsUpdate = n.needsUpdate = true; spring.lastLength = length;
  }
  beam(a, b, radius) {
    this.a.copy(a); this.b.copy(b); this.d.subVectors(b, a);
    this.scale.set(radius, radius, this.d.length()); this.q.setFromUnitVectors(Z, this.d.normalize());
    return this.result.compose(this.a.add(this.b).multiplyScalar(.5), this.q, this.scale);
  }
  matrixFor(part) {
    const m = part.motion;
    if(m.role==='trailer'||m.role==='trailerWheel'){
      const h=this.definition.trailer.hitch,t=this.pose.trailer??{};
      const q=new T.Quaternion().setFromEuler(new T.Euler(t.pitchRad??0,t.yawRad??0,0,'YXZ'));
      pivot(this.result,h,q);
      if(m.role==='trailerWheel'){
        const rotation=new T.Matrix4();pivot(rotation,m.center,new T.Quaternion().setFromAxisAngle(X,t.rotationRad??0));this.result.multiply(rotation);
      }
      return this.result.multiply(part.matrix);
    }
    if (m.role === 'steering') return this.result.multiplyMatrices(this.steeringDelta, part.matrix);
    const s = this.states[m.corner], { h } = s;
    switch (m.role) {
      case 'hub': case 'wheel': return this.result.multiplyMatrices(s.wheel, part.matrix);
      case 'knuckle': return this.result.multiplyMatrices(s.steer, part.matrix);
      case 'lowerArm': return this.beam(this.a.fromArray(m.endpoints[0]), s.lower, m.radius);
      case 'upperArm': return this.beam(this.a.fromArray(m.endpoints[0]), s.upper, m.radius);
      case 'upright': return this.beam(s.lower, s.upper, m.radius);
      case 'damper': return this.beam(s.top, this.b.copy(s.top).addScaledVector(s.shockDir, h.shockLength * .67), m.radius);
      case 'piston': return this.beam(this.a.copy(s.top).addScaledVector(s.shockDir, h.shockLength * .6), s.bottom, m.radius);
      case 'spring': return s.springMatrix;
      case 'topSeat': case 'bottomSeat': {
        const t = m.role === 'topSeat' ? .02 : s.coilLength;
        return this.beam(this.a.copy(s.top).addScaledVector(s.shockDir, t - .012), this.b.copy(s.top).addScaledVector(s.shockDir, t + .012), m.radius);
      }
      case 'shockEye': return this.beam(this.a.copy(s.bottom).addScaledVector(s.shockDir, -.06), s.lower, m.radius);
      case 'tieRod': return this.beam(s.tieInner, s.tieOuter, m.radius);
      case 'axle': {
        this.beam(this.a.fromArray(h.axleInner), s.hub, m.radius);
        return this.result.multiply(new T.Matrix4().makeRotationZ(s.rotation * h.side));
      }
      case 'cvBoot': {
        // Rubber ribs retain their section and articulate along the plunging shaft.
        const offset = Math.abs(part.matrix.elements[12] - h.hub[0]);
        this.a.copy(s.hub).addScaledVector(s.axleDir, -offset);
        this.q.setFromUnitVectors(X, s.axleDir.clone().multiplyScalar(h.side));
        this.q.multiply(new T.Quaternion().setFromAxisAngle(X, s.rotation));
        return this.result.makeTranslation(...this.a.toArray()).multiply(new T.Matrix4().makeRotationFromQuaternion(this.q)).multiply(new T.Matrix4().makeTranslation(-part.matrix.elements[12], -part.matrix.elements[13], -part.matrix.elements[14])).multiply(part.matrix);
      }
      default: return part.matrix;
    }
  }
}
