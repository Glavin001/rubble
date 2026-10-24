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

/** Every Rubble part a vibe-land model can have, in VehicleDesc.parts order
 * (a model keeps the ones it has: `partsFor`). Part 0 is the root: it stays the
 * car and its mount is never checked. A part carried by another (a mirror on a
 * door, a tailgate on the bed) comes after it. `role` is serde PartRole. */
const body = { type: 'body' };
export const ALL_PARTS = [
  { name: 'frame', role: body },
  { name: 'front_bumper', role: body },
  { name: 'nose', role: body },
  { name: 'hood', role: body },
  { name: 'fascia', role: body },
  { name: 'front_wing', role: body },
  { name: 'headlamp_l', role: body },
  { name: 'headlamp_r', role: body },
  { name: 'door_l', role: body },
  { name: 'door_r', role: body },
  { name: 'mirror_l', role: body },
  { name: 'mirror_r', role: body },
  { name: 'windshield', role: body },
  { name: 'rear_window', role: body },
  { name: 'roof', role: body },
  { name: 'light_bar', role: body },
  { name: 'top_wing', role: body },
  { name: 'roof_scoop', role: body },
  { name: 'arch_fl', role: body },
  { name: 'arch_fr', role: body },
  { name: 'arch_rl', role: body },
  { name: 'arch_rr', role: body },
  { name: 'side_skin_l', role: body },
  { name: 'side_skin_r', role: body },
  { name: 'nerf_l', role: body },
  { name: 'nerf_r', role: body },
  { name: 'bed', role: body },
  { name: 'tailgate', role: body },
  { name: 'hatch', role: body },
  { name: 'trunk_lid', role: body },
  { name: 'rear_wing', role: body },
  { name: 'stack_l', role: body },
  { name: 'stack_r', role: body },
  { name: 'rear_bumper', role: body },
  { name: 'rear_lamp_l', role: body },
  { name: 'rear_lamp_r', role: body },
  { name: 'engine', role: { type: 'engine' } },
  // All four wheels are driven (vibe-land's garage default drivetrain: 'awd').
  { name: 'transmission', role: { type: 'drive_shaft', wheels: 0b1111 } },
  { name: 'fuel_tank', role: body },
  { name: 'seat_l', role: body },
  { name: 'seat_r', role: body },
  { name: 'extinguisher', role: body },
  { name: 'steering_column', role: { type: 'steering', wheels: 0b0011 } },
];

/** The Rubble parts of a model: those its classified recipe parts use, in
 * ALL_PARTS order. Throws past the engine's 32 parts a vehicle. */
export function partsFor(classes) {
  const used = new Set(classes.filter((c) => c.cls !== 'wheel' && c.cls !== 'trailer').map((c) => c.name));
  const list = ALL_PARTS.filter((p) => used.has(p.name));
  if (list.length > 32) throw new Error(`${list.length} parts: at most 32 a vehicle`);
  return list;
}

/** The buggy's parts (the dune_buggy preset and its hybrid build). */
export const RUBBLE_PARTS = ALL_PARTS.filter((p) => ['frame', 'front_bumper', 'nose', 'headlamp_l', 'headlamp_r', 'mirror_l', 'mirror_r', 'rear_bumper',
  'rear_lamp_l', 'rear_lamp_r', 'engine', 'transmission', 'fuel_tank', 'seat_l', 'seat_r', 'extinguisher', 'steering_column'].includes(p.name));
export const PART_INDEX = Object.fromEntries(RUBBLE_PARTS.map((p, i) => [p.name, i]));

/** vibe-land's catalog (src/vehicle-catalog.mjs) as Rubble presets. */
export const MODELS = ['buggy', 'trophy', 'rally', 'monster', 'derby', 'sprint', 'semi'];
export const presetName = (model) => (model === 'buggy' ? 'dune_buggy' : `vibe_${model}`);

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
export const CLASSES = ['cell', 'crumple', 'panel', 'rigid', 'wheel', 'rig', 'trailer'];

const side = (center) => (center[0] < 0 ? 'l' : 'r'); // recipe frame: left at x < 0

// Bodywork of the bodied models (vehicle-body.mjs, vehicle-specials.mjs):
// [pattern, part (or a function of the side), class, has a collider].
const STRUCTURE = 'cell';
const BODY_RULES = [
  // Structure welded to the cage: the frame.
  [/^(Body outrigger front|Body outrigger rear|Body sill front mount|Body sill rear mount|Body sill|A pillar|Roof side rail|B pillar|Windshield lower rail|Windshield upper rail|Windshield crash bar|Rear cab crossbar|Hood support|Fender bracket|Rear window pillar|Trunk side rail|Trunk mount|Trunk rear stay|Trunk cross rail|Hatch pillar|Tail support|Bed front post|Fifth wheel rail|Fifth wheel plate|Cab back|Cab step|Step mount|Stack mount)$/, 'frame', STRUCTURE, true],
  [/^(Door outer skin|Door recessed handle|Door belt rail|Door impact rail|Side window|Window guard|Window divider)$/, (s) => `door_${s}`, 'rigid', (n) => !/handle|divider/.test(n)],
  [/^(Door mirror arm|Door mirror)$/, (s) => `mirror_${s}`, 'rigid', true],
  [/^Windshield$/, 'windshield', 'rigid', true],
  [/^(Cab rear window|Hatch glass)$/, 'rear_window', 'rigid', true],
  [/^(Roof panel|Race number upright)$/, 'roof', 'rigid', true],
  [/^Light bar (foot|housing|cell)/, 'light_bar', 'rigid', (n) => !/cell/.test(n)],
  [/^(Hood panel|Hood edge|Hood race stripe)$/, 'hood', 'rigid', (n) => !/stripe/.test(n)],
  [/^(Front fascia|Recessed grille|Grille slat)$/, 'fascia', 'rigid', (n) => n === 'Front fascia'],
  [/^(Front impact bumper|Front bumper stay)$/, 'front_bumper', 'rigid', true],
  [/^Front lamp (housing|lens)$/, (s) => `headlamp_${s}`, 'rigid', (n) => /housing/.test(n)],
  [/^(Front|Rear) wheel arch \d+$/, null, 'rigid', true], // corner from the name and side
  [/^(Bed upper rail|Bed rear support|Bed side panel|Bed diagonal|Bed floor|Bed floor rib)$/, 'bed', 'rigid', (n) => !/rib/.test(n)],
  [/^(Tailgate|Tailgate handle)$/, 'tailgate', 'rigid', (n) => n === 'Tailgate'],
  [/^(Rear quarter upper|Tail panel|Rear quarter panel)$/, 'hatch', 'rigid', true],
  [/^(Trunk lid|Trunk retaining strap)$/, 'trunk_lid', 'rigid', (n) => n === 'Trunk lid'],
  [/^(Rear wing|Wing pedestal)$/, 'rear_wing', 'rigid', true],
  [/^Roof intake (side|cap)$/, 'roof_scoop', 'rigid', true],
  [/^(Rear impact bumper|Rear crash bumper|Rear bumper mount|Rear push bar|Rear push bar stay|Rally mud flap|Mud flap hanger)$/, 'rear_bumper', 'rigid', (n) => !/flap|hanger/.test(n)],
  [/^(Truck tail light|Tail light)$/, (s) => `rear_lamp_${s}`, 'rigid', true],
  [/^(Exhaust stack|Exhaust tip)$/, (s) => `stack_${s}`, 'rigid', (n) => n === 'Exhaust stack'],
  // Sprint car (vehicle-specials.mjs).
  [/^(Sprint nose|Sprint lamp support|Nose flange)$/, 'nose', 'rigid', (n) => n === 'Sprint nose'],
  [/^Cockpit side skin$/, (s) => `side_skin_${s}`, 'rigid', true],
  [/^(Side nerf bar|Nerf rear return)$/, (s) => `nerf_${s}`, 'rigid', true],
  [/^(Wing front support|Wing rear support|Top wing chord \d+|Wing end plate|Wing number stroke)$/, 'top_wing', 'rigid', (n) => !/stroke/.test(n)],
  [/^(Front wing support|Front wing end plate|Front wing)$/, 'front_wing', 'rigid', true],
];

function bodyPart(name, center) {
  for (const [re, part, cls, collider] of BODY_RULES) {
    const m = re.exec(name);
    if (!m) continue;
    const s = side(center);
    const p = part === null ? `arch_${m[1] === 'Front' ? 'f' : 'r'}${s}` : typeof part === 'function' ? part(s) : part;
    return [p, cls, typeof collider === 'function' ? collider(name) : collider];
  }
  return null;
}

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
 * Classify one recipe part of `model`.
 * @param {{name: string, system: string, motion: object|null, center: number[]}} p
 * @returns {{name: string, part: number, cls: string, corner: string|null, collider: boolean}}
 *   `name` is the Rubble part (wheel-class parts name the frame; the wheel
 *   corner owns them), `part` its index among the buggy's parts (buggy only),
 *   `collider` whether the physics bake gives it a collision shape. Semi
 *   trailer parts are class `trailer` (the trailer is a vehicle of its own).
 */
export function classify(p, model = 'buggy') {
  const { name, system, motion, center } = p;
  const corner = motion?.corner ?? cornerOf(name);
  const role = motion?.role ?? null;
  const at = (part, cls, collider = true) => ({ name: part, part: PART_INDEX[part], cls, corner, collider });
  const bodied = !['buggy', 'sprint'].includes(model);

  // The semi's trailer: its own vehicle (trailer wheels spin with its wheels).
  if (role === 'trailer' || role === 'trailerWheel') {
    return { name: 'trailer', part: undefined, cls: 'trailer', corner: null, collider: role === 'trailer' && !/stake|flap|hanger|plank|leaf spring|lamp/.test(name), trailerWheel: role === 'trailerWheel' };
  }

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
      // Under a bodied model's hood the buggy's nose panel is part of it.
      return at(bodied ? 'hood' : 'nose', 'panel', name === 'Nose panel' || name === 'Hood mounting standoff');
    const b = bodyPart(name, center);
    if (b) return at(b[0], b[1], b[2]);
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
