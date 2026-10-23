// How each recipe part of vibe-land's buggy (src/buggy.mjs) maps onto Rubble.
// Both bakes (mesh and physics) and both build modes (rigid, hybrid) read this
// one table, so a part never belongs to one group in the render and another in
// the physics.
//
// Frames: the recipe is vibe-land's (metres, +Y up, -Z front, left at x < 0).
// Rubble vehicles are +Z forward, +X left (engine-vehicle presets.rs), so every
// point goes through R(x, y, z) = (-x, y, -z): a 180° turn about Y, which maps
// the recipe corners fl/fr/rl/rr onto Rubble's wheel order FL, FR, RL, RR.

export const toRubble = ([x, y, z]) => [-x, y, -z];

/** Rubble vehicle parts, in VehicleDesc.parts order. Part 0 is the root: it
 * stays the car and its mount is never checked. `role` is serde PartRole. */
export const RUBBLE_PARTS = [
  { name: 'frame', role: { type: 'body' } },
  { name: 'front_bumper', role: { type: 'body' } },
  { name: 'nose', role: { type: 'body' } },
  { name: 'headlamp_l', role: { type: 'body' } },
  { name: 'headlamp_r', role: { type: 'body' } },
  { name: 'mirror_l', role: { type: 'body' } },
  { name: 'mirror_r', role: { type: 'body' } },
  { name: 'rear_bumper', role: { type: 'body' } },
  { name: 'rear_lamp_l', role: { type: 'body' } },
  { name: 'rear_lamp_r', role: { type: 'body' } },
  { name: 'engine', role: { type: 'engine' } },
  // All four wheels are driven (vibe-land's garage default drivetrain: 'awd').
  { name: 'transmission', role: { type: 'drive_shaft', wheels: 0b1111 } },
  { name: 'fuel_tank', role: { type: 'body' } },
  { name: 'seat_l', role: { type: 'body' } },
  { name: 'seat_r', role: { type: 'body' } },
  { name: 'extinguisher', role: { type: 'body' } },
  { name: 'steering_column', role: { type: 'steering', wheels: 0b0011 } },
];
export const PART_INDEX = Object.fromEntries(RUBBLE_PARTS.map((p, i) => [p.name, i]));

/** Recipe corner ids → Rubble wheel index (FL, FR, RL, RR after R). */
export const CORNERS = ['fl', 'fr', 'rl', 'rr'];

/** Hybrid-mode behaviour of a recipe part:
 * - cell: stiff node-beam safety cell (bends only in a severe crash)
 * - crumple: node-beam crumple zone (bends first; bumpers then tear off)
 * - panel: braced sheet that dents, then tears off its mounts
 * - rigid: rigid body on breakable joints (snaps off)
 * - wheel: travels with its raycast wheel (detaches with the wheel mount)
 * - rig: posed by the suspension rig, rigid for now
 */
export const CLASSES = ['cell', 'crumple', 'panel', 'rigid', 'wheel', 'rig'];

const side = (center) => (center[0] < 0 ? 'l' : 'r'); // recipe frame: left at x < 0

const cornerOf = (name) => {
  const m = /^(Front|Rear) (left|right) /.exec(name);
  return m ? (m[1] === 'Front' ? 'f' : 'r') + (m[2] === 'left' ? 'l' : 'r') : null;
};

// Cage members by index (buggy.mjs pairs): 1 A-B (front lower rail), 2 B-C,
// 3 C-D, 4 B-E, 5 E-F, 6 F-C, 7 E-G (A-pillar), 8 G-H (roof rail), 9 H-F
// (main hoop), 10 F-I, 11 I-D, 12 D-H, 13 B-F, 14 C-E. A is the nose, D the
// rear chassis corner, I the engine cage top.
const CRUMPLE_MEMBERS = new Set([1, 10, 11, 12]);

/**
 * Classify one recipe part.
 * @param {{name: string, system: string, motion: object|null, center: number[]}} p
 * @returns {{part: number, cls: string, corner: string|null, collider: boolean}}
 *   `part` is the Rubble part (wheel-class parts keep the frame's index; the
 *   wheel corner owns them), `collider` whether the physics bake gives it a
 *   collision shape.
 */
export function classify(p) {
  const { name, system, motion, center } = p;
  const corner = motion?.corner ?? cornerOf(name);
  const role = motion?.role ?? null;
  const at = (part, cls, collider = true) => ({ part: PART_INDEX[part], cls, corner, collider });

  // Moving corner parts: the raycast wheel and the rig own them.
  if (role === 'wheel' || role === 'hub') return at('frame', 'wheel', false);
  if (role) {
    if (role === 'steering') return at('steering_column', 'rigid', name === 'Steering column' || name === 'Steering center');
    return at('frame', 'rig', false);
  }

  if (system === 'Frame') {
    let m;
    if (/^(Left|Right) front bumper upright$/.test(name) || name === 'Front bumper crossmember') return at('front_bumper', 'crumple');
    if (name === 'Rear bumper' || name === 'Rear bumper stay') return at('rear_bumper', 'crumple');
    if (/hood rail$/.test(name) || name === 'Front chassis crossmember') return at('frame', 'crumple');
    if ((m = /^(?:Left|Right) cage member (\d+)$/.exec(name))) return at('frame', CRUMPLE_MEMBERS.has(+m[1]) ? 'crumple' : 'cell');
    return at('frame', 'cell');
  }
  if (system === 'Suspension') {
    // Static corner hardware welded to the frame (moving parts carry a motion role).
    return at('frame', 'cell', /shock tower|chassis mount/.test(name));
  }
  if (system === 'Body') {
    if (/^(Hood mounting standoff|Nose panel|Hood accent stripe|Hood fastener)$/.test(name))
      return at('nose', 'panel', name === 'Nose panel' || name === 'Hood mounting standoff');
    if (/^(Headlamp bracket|Headlight housing|Headlight lens)$/.test(name))
      return at(`headlamp_${side(center)}`, 'rigid', name !== 'Headlight lens');
    if (/^Mirror /.test(name)) return at(`mirror_${side(center)}`, 'rigid', name !== 'Mirror glass');
    if (/^Rear lamp /.test(name)) return at(`rear_lamp_${side(center)}`, 'rigid', name === 'Rear lamp mount');
    if (/^(Extinguisher|Fire extinguisher)/.test(name)) return at('extinguisher', 'rigid', name === 'Fire extinguisher' || name === 'Extinguisher bracket');
  }
  if (system === 'Drivetrain') {
    if (name === 'Transmission case') return at('transmission', 'rigid');
    if (/^(Fuel tank|Fuel tank saddle|Tank strap|Fuel cap)$/.test(name))
      return at('fuel_tank', 'rigid', name === 'Fuel tank' || name === 'Fuel tank saddle');
    // Everything else under Drivetrain without a motion role is the engine:
    // crankcase, barrels, fins, heads, exhaust, fan, intake, mounting struts.
    return at('engine', 'rigid', !/^(Cooling fin|Filter pleat|Fan pulley)/.test(name));
  }
  if (system === 'Cabin') {
    if (/^(Seat pan|Seat back|Headrest support|Head restraint|Seat base side bolster|Seat shoulder bolster|Shoulder harness|Lap harness|Harness buckle|Harness attachment)$/.test(name))
      return at(`seat_${side(center)}`, 'rigid', /^(Seat pan|Seat back|Head restraint)$/.test(name));
    if (/^Steering /.test(name)) return at('steering_column', 'rigid', name === 'Steering column bracket');
    // Floors, rails, dashboard, gauges, pedals, shifter: fixed to the frame.
    const solid = /^(Driver floor|Passenger floor|Seat mounting rail|Dashboard|Dashboard cross support)$/.test(name);
    return at('frame', /floor$/.test(name) ? 'panel' : 'cell', solid);
  }
  throw new Error(`unclassified recipe part: ${system} / ${name}`);
}
