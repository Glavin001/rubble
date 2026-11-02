import test from 'node:test';
import assert from 'node:assert/strict';
import { parseDemoQuery, DROPPED_SWITCHES } from '../demo-config.mjs';

// demo.html runs on World: its step is engine-exec's rigid_plan, whose
// dependency order (persistence after narrowphase, packing before CSR and the
// solve, the persist list's slots, carried rest pairs, local repair, the sleep
// gate) is checked in Rust (rigid_plan::tests::every_mode_keeps_its_contact_dependencies).
// What stays in the page is how its URL maps onto World's ContactStorage and
// EngineFeatures (demo-config.mjs): the defaults the JS host ran, the explicit
// controls, and errors before any GPU work.
const parse = (q) => parseDemoQuery(q);

for (const [query, expected] of [
  ['?scene=ring110k', 10], ['?scene=columns', 32], ['?scene=gables506k', 32],
  ['?scene=ring110k&sparse=32', 32], ['?scene=ring110k&sparse=0', 0], ['?scene=ring110k&sparse=8', 8],
]) {
  test(`sparse scheduling keeps the scene defaults and explicit controls (${query})`, () => {
    assert.equal(parse(query).features.sparseColors, expected);
  });
}
for (const value of ['-1', '33', '2.5', 'invalid']) {
  test(`invalid sparse color ${value} fails before the world is made`, () => {
    assert.throws(() => parse(`?scene=ring110k&sparse=${value}`), /sparse must be an integer color index from 0 through 32/);
  });
}

for (const [query, mode] of [
  ['?scene=ring110k', 'persistent'], ['?scene=gables506k', 'persistent'], ['?scene=columns', 'persistent'],
  ['?scene=gables506k&contacts=canonical', 'canonical'], ['?scene=ring110k&contacts=compact', 'compact'],
]) {
  test(`contact storage is World's ContactStorage (${query})`, () => {
    assert.equal(parse(query).contacts, mode);
  });
}

// The page's default on every scene is the engine's shipped feature set (with
// the ring's sparse start): the before/after comparison runs the same paths.
test('defaults are the shipped EngineFeatures', () => {
  assert.deepEqual(parse('?scene=ring110k').features, {
    sparseColors: 10, manifoldPersistence: 'contacts', warmHash: 'record', reuseCopy: true,
    npSplit: true, persistList: true, restingContacts: 'archive', carryRest: true, localRepair: true, grid: 'split',
  });
  const wall = parse('?scene=wall');
  assert.equal(wall.features.sparseColors, 32);
  assert.equal(wall.ccd, true, 'smash balls carry CCD by default');
  assert.equal(parse('?scene=ring110k&check=ring').ccd, false, 'checks compare against native runs without CCD');
  assert.equal(parse('?scene=ring110k&check=ring&ccd=1').ccd, true);
});

for (const [query, key, value] of [
  ['?scene=ring110k&hashInsert=manifold', 'warmHash', 'manifold'], ['?scene=ring110k&hashInsert=contact', 'warmHash', 'contact'],
  ['?scene=ring110k&grid=scaled', 'grid', 'scaled'], ['?scene=ring110k&grid=fixed', 'grid', 'fixed'],
  ['?scene=ring110k&np=fused', 'npSplit', false], ['?scene=ring110k&reuse=pass', 'reuseCopy', false],
  ['?scene=ring110k&reuse=pass', 'npSplit', false], ['?scene=ring110k&reuse=pass', 'persistList', false],
  ['?scene=ring110k&persist=scan', 'persistList', false], ['?scene=ring110k&repair=rounds', 'localRepair', false],
  ['?scene=ring110k&rest=drop', 'carryRest', false], ['?scene=ring110k&persistence=manifold', 'manifoldPersistence', 'manifolds'],
  ['?scene=ring110k&persistence=manifold', 'reuseCopy', false], ['?scene=ring110k&persistence=fused', 'manifoldPersistence', 'fused'],
  ['?scene=columns&contacts=canonical', 'reuseCopy', false], ['?scene=ring110k&colors=full', 'colorHeadroom', null],
  ['?scene=ring110k&rounds=full', 'colorRoundHeadroom', null],
]) {
  test(`${query} sets features.${key}`, () => {
    assert.equal(parse(query).features[key], value);
  });
}

for (const [query, message] of [
  ['?scene=ring110k&hashInsert=pair', /hashInsert must be contact, manifold or record/],
  ['?scene=ring110k&grid=half', /grid must be fixed, scaled or split/],
  ['?scene=ring110k&np=pair', /np must be fused or split/],
  ['?scene=ring110k&reuse=pass&np=split', /np=split requires reuse=copy/],
  ['?scene=ring110k&reuse=pass&persist=list', /persist=list requires reuse=copy/],
  ['?scene=ring110k&contacts=canonical&reuse=copy', /reuse=copy requires persistent contacts/],
  ['?scene=ring110k&rest=keep', /rest must be archive, carry or drop/],
  ['?scene=ring110k&repair=x', /repair must be rounds or local/],
  ['?scene=ring110k&contacts=wide', /contacts must be persistent, compact or canonical/],
  ['?scene=ring110k&sleep=maybe', /sleep must be on or off/],
  ['?scene=ring110k&check=tower', /check must be ring or gables/],
  ['?scene=nowhere', /unknown scene nowhere/],
  ['?scene=ring110k&endpoints=manifold', /JS-host experiment/],
]) {
  test(`${query} fails before the world is made`, () => assert.throws(() => parse(query), message));
}

// The JS host's experiments with no path in Rust's plan: refused with the
// reason, while their values equal to the plan's own are accepted.
for (const [key, d] of Object.entries(DROPPED_SWITCHES)) {
  test(`${key}: the JS host's experiment is refused, the plan's own value accepted`, () => {
    const other = { coloring: 'rebuild', prefix: 'serial', packing: 'atomic', adjacency: 'append', profile: 'detail' }[key];
    assert.throws(() => parse(`?scene=ring110k&${key}=${other}`), /JS-host experiment/);
    if (d.keep) assert.doesNotThrow(() => parse(`?scene=ring110k&${key}=${d.keep}`));
  });
}

test('sleep follows the scene default unless the URL sets it', () => {
  assert.equal(parse('?scene=ring110k').sleepOn, false, 'the three-avbd comparison scenes run awake');
  assert.equal(parse('?scene=ring110k&sleep=on').sleepOn, true);
  assert.equal(parse('?scene=ring110k&sleep=on').sleepEnabledTime, 0.5);
  assert.equal(parse('?scene=pile').sleepEnabledTime, 0.25);
  assert.equal(parse('?scene=wall').sleepOn, true);
});
