/** How much of a modelled solid is material.
 *
 * Many parts are modelled as the solid envelope of something hollow or thin: a
 * tyre is a rubber carcass around air, a crankcase a casting around the crank,
 * a hood a pressed sheet drawn 2-3 cm thick. Massed as solids at their
 * material's density they made the cars 2-5x their real weight (a buggy at
 * 2.5 t, a monster truck at 7.9 t with 859 kg front wheels), and every impact,
 * bump and landing then loaded the joints with that much more momentum: a
 * monster truck lost its wheels running into a wall at 4 m/s
 * (garage_car_survives_ordinary_driving, 2026-09-30).
 *
 * `shell`: a closed hollow body; material volume = surface area x wall.
 * `sheet`: a single skin modelled with thickness; material volume = half the
 *          surface area (one face) x sheet thickness.
 * A part's mass is density x min(solid volume, material volume); anything not
 * listed is solid. Names are the recipe's part names (side prefixes and
 * indices included), so every build shares the table.
 */
export const constructions = [
  // Tyre carcass and tread, ~2.5 cm of rubber around air.
  [/tire carcass$/, { shell: 0.025 }],
  // Castings around moving internals; walls plus webs.
  [/^Engine crankcase$/, { shell: 0.012 }],
  [/^Transmission case$/, { shell: 0.008 }],
  // Pressed and welded thin-wall bodies.
  [/^Fuel tank$/, { shell: 0.003 }],
  [/^Exhaust muffler$/, { shell: 0.0015 }],
  [/(^| )(lamp|Light bar) housing$/i, { shell: 0.002 }],
  [/impact bumper$|crash bumper$/, { shell: 0.004 }],
  // Body skins: 1-1.5 mm steel, floors and beds 2-3 mm.
  [/^(Hood panel|Trunk lid|Roof panel|Tailgate|Door outer skin|Front fascia|Nose panel|Sprint nose|Cockpit side skin|Bed side panel|Rear quarter panel|Rear quarter upper|Tail panel)( \d+)?$/, { sheet: 0.0015 }],
  [/^(Top wing chord|Wing end plate|Front wing end plate|Rear wing)( \d+)?$/, { sheet: 0.002 }],
  [/^(Bed floor|Driver floor|Passenger floor)$/, { sheet: 0.003 }],
  [/^(Fan shroud|Dashboard)$/, { sheet: 0.002 }],
];

/** Material volume of a part (m³) given its solid volume and surface area. */
export function materialVolume(name, volume, surfaceArea) {
  const found = constructions.find(([pattern]) => pattern.test(name));
  if (!found) return volume;
  const { shell, sheet } = found[1];
  const hollow = shell ? surfaceArea * shell : surfaceArea / 2 * sheet;
  return Math.min(volume, hollow);
}
