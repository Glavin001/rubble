// The dune buggy's render assets (engine/scenes/dune-buggy, `bun run bake:mesh`,
// run by build-web.sh): buggy-rig.mjs must pose every moving part exactly as
// vibe-land's Three.js VisualRig did (the bake records its matrices at a dozen
// poses), and the mesh must agree with the preset the physics runs.
// Run `bash engine/scripts/build-web.sh` (or wgsl_export + bake:mesh) first.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BuggyRig, CORNER_IDS, rest4 } from '../buggy-rig.mjs';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const mesh = read('../generated/dune-buggy/buggy-mesh.json');
const fixtures = read('../generated/dune-buggy/rig-fixtures.json');

test('buggy-rig.mjs matches the original visual rig at every recorded pose', () => {
  const rig = new BuggyRig(mesh.rig);
  let checked = 0, worst = 0;
  for (const pose of fixtures.poses) {
    rig.applyPose(pose.wheels);
    for (const id of CORNER_IDS) {
      assert.ok(Math.abs(rig.states[id].coilLength - pose.coil[id]) < 1e-6, `${id} coil`);
    }
    for (const [i, expected] of pose.matrices) {
      const inst = mesh.instances[i];
      const got = rig.matrixFor(inst);
      for (let k = 0; k < 16; k++) {
        const err = Math.abs(got[k] - expected[k]);
        worst = Math.max(worst, err);
        assert.ok(err < 2e-5, `${inst.name} (${inst.role}) element ${k}: ${got[k]} vs ${expected[k]}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 1000, `checked ${checked} matrices (worst ${worst})`);
});

test('at neutral every moving part sits at its rest pose', () => {
  const rig = new BuggyRig(mesh.rig);
  for (const inst of mesh.instances) {
    if (!inst.role || inst.role === 'spring' || ['lowerArm', 'upperArm', 'upright', 'damper', 'piston', 'topSeat', 'bottomSeat', 'shockEye', 'tieRod', 'axle'].includes(inst.role)) continue;
    const got = rig.matrixFor(inst), rest = rest4(inst.matrix);
    for (let k = 0; k < 16; k++) assert.ok(Math.abs(got[k] - rest[k]) < 1e-5, `${inst.name}: ${k}`);
  }
});

test('the mesh and the dune_buggy preset agree on parts and wheels', () => {
  const presets = read('../generated/vehicle_presets.json');
  const buggy = presets.find((p) => p.name === 'dune_buggy');
  assert.ok(buggy, 'dune_buggy preset exported');
  assert.deepEqual(mesh.parts, buggy.parts.map((p) => p.name));
  assert.deepEqual(mesh.corners, ['fl', 'fr', 'rl', 'rr']);
  assert.equal(buggy.wheels.length, 4);
  for (const inst of mesh.instances) {
    assert.ok(inst.template >= 0 && inst.template < mesh.templates.length, inst.name);
    assert.ok(inst.part >= 0 && inst.part < mesh.parts.length, inst.name);
    assert.ok(inst.material >= 0 && inst.material < mesh.palette.length, inst.name);
    if (mesh.classes[inst.cls] === 'wheel' || inst.role) assert.ok(inst.corner >= 0 || inst.role === 'steering', `${inst.name} has a corner`);
  }
  // Rubble's wheel i is the recipe corner i after R (180° about Y): the hub's
  // design-space X is the negated recipe X (+X left in Rubble).
  CORNER_IDS.forEach((id, i) => {
    const hub = mesh.rig.corners[id].hub;
    assert.ok(Math.sign(buggy.wheels[i].position[0]) === -Math.sign(hub[0]), `${id} side`);
    assert.ok(Math.sign(buggy.wheels[i].position[2]) === -Math.sign(hub[2]), `${id} end`);
  });
});
