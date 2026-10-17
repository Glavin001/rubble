// The cross-engine benchmark suite: which scenes, how long each runs, and the quality gates a
// run must pass before its timing counts. Scenes are built by the pinned three-avbd builders
// (the same functions its own bench3d suite calls), so the AVBD oracle and every other engine
// start from bit-identical bodies. See ../README.md for the protocol.

import { boxColumns, boxPile, brickGables, brickPyramid, brickRing, chainMail, jointedDrop, wallSmash } from '../../inspiration/pinned/three-avbd/src/avbd3d/bench-scenes.ts';
import type { Solver } from '../../inspiration/pinned/three-avbd/src/avbd3d/ref/solver.ts';

/** Same tiers as three-avbd's bench-cases.ts. */
export type Tier = 'small' | 'paper' | 'large' | 'steady';

/** Body index lists, computed from the built scene (canonical, Y-up). */
export interface GroupContext {
  /** Canonical Y-up position of each body. */
  pos: [number, number, number][];
  isSphere: boolean[];
  isStatic: boolean[];
  radius: number[];
}

/**
 * A gate: `metric` of `group` at checkpoint step `at` must be ≤ max and/or ≥ min. Metrics are
 * computed by the judge from each engine's pose dumps (crates/bench-judge), never by the engine.
 */
export interface Gate {
  /** Checkpoint: the number of completed steps (0 = the initial state, read back from the engine). */
  at: number;
  metric: string;
  group?: string;
  max?: number;
  min?: number;
  /** Why the bound is what it is (shown in the report when a gate fails). */
  why: string;
}

export interface SceneSpec {
  id: string;
  /** three-avbd's display name for the same case (bench-cases.ts), for cross-reference. */
  threeAvbdName: string;
  tier: Tier;
  /** The reference AVBD iteration count (three-avbd's BENCH_CASES). */
  iterations: number;
  /** Untimed steps before the timing window (three-avbd's `lead`). */
  lead: number;
  /** Total steps simulated (lead + timing window + whatever the last gate needs). */
  steps: number;
  build: (s: Solver) => void;
  groups?: (c: GroupContext) => Record<string, number[]>;
  gates: Gate[];
}

/** The timing window every adapter runs: 2 rounds × (3 untimed + 5 batches × 10 steps). */
export const PROTOCOL = { rounds: 2, warmSteps: 3, batches: 5, perBatch: 10 } as const;
export const TIMED_SPAN = PROTOCOL.rounds * (PROTOCOL.warmSteps + PROTOCOL.batches * PROTOCOL.perBatch);

const dynamic = (c: GroupContext) => c.pos.map((_, i) => i).filter((i) => !c.isStatic[i]);
const balls = (c: GroupContext) => dynamic(c).filter((i) => c.isSphere[i] && c.radius[i] >= 1);

/** Gates every scene shares: nothing blew up, nothing left the world. */
const sane = (at: number): Gate[] => [
  { at, metric: 'nonfinite', max: 0, why: 'NaN/Inf pose or velocity' },
  { at, metric: 'escaped', max: 0, why: 'body fell through the ground or left the scene bounds' },
];

export const SUITE: SceneSpec[] = [
  {
    id: 'wall_smash_2k',
    threeAvbdName: 'Wall smash 2k',
    tier: 'small',
    iterations: 10,
    lead: 30,
    steps: 240,
    build: (s) => wallSmash(s),
    groups: (c) => ({ ball: balls(c) }),
    gates: [
      ...sane(240),
      { at: 240, metric: 'pen_p99', max: 0.03, why: 'resting/impact contacts stay within 3 cm' },
      // Ball starts 25 m in front of the 0.5 m-thick wall moving through it at 30 m/s
      { at: 240, metric: 'pos_z', group: 'ball', max: -2, why: 'ball broke through the wall (canonical −z is three-avbd +y)' },
    ],
  },
  {
    id: 'chain_mail_1.6k',
    threeAvbdName: 'Chain mail 1.6k',
    tier: 'small',
    iterations: 10,
    lead: 60,
    steps: 300,
    build: (s) => chainMail(s),
    groups: (c) => ({ ball: balls(c) }),
    gates: [
      ...sane(300),
      { at: 300, metric: 'joint_gap_p99', max: 0.05, why: 'ball joints hold their anchors together' },
      { at: 300, metric: 'pos_y', group: 'ball', min: 3, why: 'the net catches the ball above the ground' },
      { at: 300, metric: 'pen_p99', max: 0.05, why: 'the ball rests on the 10 cm links, not inside them' },
    ],
  },
  {
    id: 'brick_ring_9k',
    threeAvbdName: 'Brick ring 9k',
    tier: 'small',
    iterations: 4,
    lead: 60,
    // The 4 m ball topples this whole 12 m ring in every engine, and leaves the 200 m ground
    // slab after ~6 s, so it is judged as a smash only, over 4 s
    steps: 240,
    build: (s) => brickRing(s, 12, 20, 2, 10),
    groups: (c) => ({ ball: balls(c) }),
    gates: smashGates(240, 12, 4),
  },
  {
    id: 'brick_gables_6.7k',
    threeAvbdName: 'Brick gables 6.7k',
    tier: 'small',
    iterations: 3,
    lead: 60,
    steps: 240,
    build: (s) => brickGables(s, 4, 8, 20),
    groups: (c) => gablesGroups(c, 20),
    gates: gablesGates(),
  },
  {
    id: 'jointed_drop_6k',
    threeAvbdName: 'Jointed drop 6k',
    tier: 'small',
    iterations: 10,
    lead: 60,
    steps: 300,
    build: (s) => jointedDrop(s, 5, 4, 32),
    gates: jointedDropGates(),
  },
  {
    id: 'brick_ring_110k',
    threeAvbdName: 'Brick ring 110k',
    tier: 'paper',
    iterations: 4,
    lead: 60,
    steps: 480,
    build: (s) => brickRing(s),
    groups: (c) => ({ ball: balls(c), flank: flank(c, 40) }),
    gates: ringGates(80, 8),
  },
  {
    id: 'jointed_drop_34k',
    threeAvbdName: 'Jointed drop 34k (71k joints)',
    tier: 'paper',
    iterations: 10,
    lead: 60,
    steps: 300,
    build: (s) => jointedDrop(s),
    gates: jointedDropGates(),
  },
  {
    id: 'brick_gables_506k',
    threeAvbdName: 'Brick gables 506k',
    tier: 'large',
    iterations: 3,
    lead: 60,
    steps: 240,
    build: (s) => brickGables(s),
    groups: (c) => gablesGroups(c, 30),
    gates: gablesGates(),
  },
  {
    id: 'settled_pile_32k',
    threeAvbdName: 'Settled pile 32k',
    tier: 'steady',
    iterations: 4,
    lead: 180,
    steps: 600,
    build: (s) => boxPile(s, 40, 20),
    gates: [
      ...sane(600),
      { at: 600, metric: 'pen_p99', max: 0.03, why: 'a settled pile keeps contacts within 3 cm' },
      // boxPile drops 20 layers of overlapping, turned boxes: bodies pushed through the 1 m ground slab escape
      { at: 600, metric: 'mean_speed', group: 'dynamic', max: 0.2, why: 'the pile comes to rest after 10 s' },
    ],
  },
  {
    id: 'box_columns_100k',
    threeAvbdName: 'Box columns 100k',
    tier: 'steady',
    iterations: 4,
    lead: 180,
    steps: 300,
    build: (s) => boxColumns(s, 100, 10),
    gates: [
      ...sane(300),
      // Ten stacked contacts settle by their collision margins; the columns must not lean or slide
      { at: 300, metric: 'max_horizontal_drift', group: 'dynamic', max: 0.1, why: 'no column of 10 resting boxes leans or slides' },
      { at: 300, metric: 'max_abs_dy', group: 'dynamic', max: 0.25, why: 'columns settle by at most a few margins' },
      { at: 300, metric: 'pen_p99', max: 0.02, why: 'resting stacks keep contacts within 2 cm' },
    ],
  },
  {
    id: 'brick_pyramid_12',
    threeAvbdName: 'Brick pyramid 12',
    tier: 'steady',
    iterations: 10,
    lead: 180,
    steps: 300,
    build: (s) => brickPyramid(s, 12),
    gates: pyramidGates(),
  },
  {
    id: 'brick_pyramid_30',
    threeAvbdName: 'Brick pyramid 30',
    tier: 'steady',
    iterations: 10,
    lead: 180,
    steps: 300,
    build: (s) => brickPyramid(s, 30),
    gates: pyramidGates(),
  },
];

/** Ring bricks well away from the ball's line (canonical x is three-avbd x). */
function flank(c: GroupContext, minAbsX: number): number[] {
  return dynamic(c).filter((i) => !c.isSphere[i] && Math.abs(c.pos[i][0]) > minAbsX);
}

/**
 * Brick ring: the ball rolls in along canonical −z, through the near wall, across the arena
 * and out through the far wall, while the flanks stand. Bounds follow rubble's ring_hold
 * regression (engine/crates/engine-webgpu/tests/ring_hold.rs), which recorded three-avbd's
 * flank drift: mean ≈0.23 m at 4 s and ≈1.0 m at 8 s.
 */
function ringGates(radius: number, ballRadius: number): Gate[] {
  return [
    ...smashGates(480, radius, ballRadius),
    // Checkpoints stay outside the timed window (steps lead..lead+106), so 3 s rather than 2 s
    { at: 180, metric: 'mean_abs_dy', group: 'flank', max: 0.3, why: 'flanks stand during the first 3 s (no spontaneous collapse)' },
    { at: 240, metric: 'mean_abs_dy', group: 'flank', max: 0.5, why: 'impact spreads, but flanks mostly stand at 4 s' },
    { at: 480, metric: 'mean_abs_dy', group: 'flank', max: 1.6, why: 'no widespread collapse at 8 s' },
  ];
}

/** A ball rolling in along canonical −z through a brick structure: it crosses, stays on the floor, nothing explodes. */
function smashGates(at: number, radius: number, ballRadius: number): Gate[] {
  return [
    ...sane(at),
    // The ground top is y = 0.5, so a resting ball's centre is at r + 0.5
    { at, metric: 'pos_y', group: 'ball', min: ballRadius, why: 'ball did not sink into the floor' },
    { at, metric: 'pos_y', group: 'ball', max: ballRadius + 2, why: 'ball did not launch off the floor' },
    { at, metric: 'pos_z', group: 'ball', max: -(radius + 2), why: 'ball crossed both walls of the ring' },
    { at, metric: 'max_speed', group: 'dynamic', max: 60, why: 'no body moves at twice the 30 m/s ball (energy injection)' },
  ];
}

/** Gables: walls in a column a ball rolls down vs. the rest, which must stand. */
function gablesGroups(c: GroupContext, base: number): Record<string, number[]> {
  const pitchX = base + 2;
  const ballIds = balls(c);
  const ballXs = ballIds.map((i) => c.pos[i][0]);
  const bricks = dynamic(c).filter((i) => !c.isSphere[i]);
  const unhit = bricks.filter((i) => ballXs.every((x) => Math.abs(c.pos[i][0] - x) > pitchX / 2 + 1));
  return { balls: ballIds, unhit };
}

function gablesGates(): Gate[] {
  return [
    ...sane(240),
    // rubble's gables_hold envelope: unhit mean drift < 0.2 m, no unhit brick > 0.6 m
    { at: 240, metric: 'max_abs_dy', group: 'unhit', max: 0.6, why: 'walls no ball reaches stay standing' },
    { at: 240, metric: 'mean_abs_dy', group: 'unhit', max: 0.2, why: 'walls no ball reaches do not sag' },
    { at: 240, metric: 'min_displacement', group: 'balls', min: 30, why: 'both balls kept rolling through the rubble (≥30 m in 4 s)' },
  ];
}

function pyramidGates(): Gate[] {
  return [
    ...sane(300),
    { at: 300, metric: 'max_horizontal_drift', group: 'dynamic', max: 0.25, why: 'the pyramid does not slide apart' },
    { at: 300, metric: 'mean_abs_dy', group: 'dynamic', max: 0.15, why: 'courses do not sink or compact' },
    { at: 300, metric: 'pen_p99', max: 0.02, why: 'resting contacts within 2 cm' },
  ];
}

/** Plates of jointed cubes fall ~5 m onto a pinned net: joints hold, nothing sinks, and they really fall. */
function jointedDropGates(): Gate[] {
  return [
    ...sane(300),
    { at: 300, metric: 'joint_gap_p99', max: 0.05, why: 'ball joints hold their anchors together' },
    { at: 300, metric: 'pen_p99', max: 0.05, why: 'plates rest on the net and each other within 5 cm' },
    // Without this a solver that never lets the jointed plates move would pass every other gate
    { at: 300, metric: 'mean_drift', group: 'dynamic', min: 2, why: 'the plates fell onto the net (three-avbd: ≈5.4 m mean drift)' },
  ];
}
