/**
 * Real joint capacities for the vehicles (VIBE_REAL_VEHICLE_JOINTS=1, opt-in;
 * docs/verification/SCENARIOS.md, "Recalibrations").
 *
 * The joint profile (strength-profile.mjs) gives each joint an effective stress
 * (parent metal x joint efficiency, reality.mjs jointReferences) and the stage
 * multiplies it by the bond's measured contact area. For two members that
 * merely touch over a face -- a roof panel lying on its cage, a fascia on its
 * grille -- that area is the face, up to 0.24 m2, and the joint reads as 70 MN:
 * 1,400 times the truck's weight and ~100-1000 times what the welds or bolts
 * that really hold it can carry (spot welds 4-10 kN each, AWS D8.9M / ISO 14273
 * tensile-shear; M10 8.8 bolts 46 kN, EN 1993-1-8 Table 3.4). The vehicle then
 * cannot lose a panel, while its wheels -- whose lug studs are modelled at
 * their real section (ISO 898-1) -- are the weakest joints it has.
 *
 * A joint can never carry more than the members it joins: a weld or a bolt
 * group fails no later than the parent metal beside it (EN 1993-1-8 4.5.2,
 * 4.7: a full-strength weld develops the connected part's resistance; 3.6.1:
 * net-section rupture of the connected part). So each joint's capacity is
 * bounded by the smaller of its two members' cross-sections at the joint's
 * stress: section = the member's material volume over its length (a tube's
 * annulus, a panel's thickness x width), its volume from its (budgeted) mass
 * and its material's density. Stiffness keeps the measured area (geometry).
 * A wheel's mount keeps its counted studs.
 */
import { jointMaterials } from './strength-profile.mjs';

// Opt-in on its own (VIBE_REAL_VEHICLE_JOINTS). Under the bound alone the
// monster truck broke its rear corners coasting (vehicle lab `coast`,
// 2026-10-08): its rear wheels mounting the paved lane's 25 mm lip at 23 m/s
// put 80 kN on each (Vehicle2's damper in one tick), and brittle steel joints at
// real capacity cascaded (251 bonds, 4 wheels). Metal joints are therefore
// ductile with the bound (applyDuctility, below): the same run breaks nothing.
export const realJointCapacitiesEnabled = () => (globalThis.process?.env?.VIBE_REAL_VEHICLE_JOINTS ?? '0') === '1';
/** VIBE_VEHICLE_JOINTS_BRITTLE=1 (A/B only): real capacities without applyDuctility, as before 2026-10-08. */
export const vehicleJointsBrittle = () => (globalThis.process?.env?.VIBE_VEHICLE_JOINTS_BRITTLE ?? '0') === '1';
export const REAL_JOINT_CAPACITY_VERSION = vehicleJointsBrittle() ? 'section-bound-1' : (globalThis.process?.env?.PX_DESTRUCTION_STATIC_DUCTILE ?? '0') === '1' ? 'section-bound-4-yield' : 'section-bound-3-ductile';

/** Density of each structural category's material (kg/m3). */
export const DENSITY = Object.freeze({
  frame: 7850, orange: 7850, steel: 7850, dark: 7850,         // steel (EN 10025: 7850)
  alloy: 2700,                                               // aluminium alloys (EN 573: 2650-2800)
  rubber: 1150,                                              // vulcanised rubber 1100-1200
  seat: 1800, red: 1800, race: 1800, housing: 1800,          // glass-fibre composite (FRP 1500-2000)
  belt: 1380,                                                // polyester webbing
  glass: 2500, lens: 1200,                                   // soda-lime glass; polycarbonate lens
});

/** Mean cross-section (m2) of a part: its material volume over its longest extent. */
export function memberSection(part) {
  const rho = DENSITY[part.material];
  if (!rho || !(part.mass > 0)) return Infinity;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const shape of part.shapes ?? []) for (const v of shape.vertices ?? []) for (let k = 0; k < 3; k += 1) {
    const x = (part.position?.[k] ?? 0) + (shape.position?.[k] ?? 0) + v[k];
    lo[k] = Math.min(lo[k], x); hi[k] = Math.max(hi[k], x);
  }
  const length = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return length > 0 && Number.isFinite(length) ? part.mass / rho / length : Infinity;
}

const LIMITS = ['compressionElastic', 'compressionFatal', 'tensionElastic', 'tensionFatal', 'shearElastic', 'shearFatal'];

/**
 * Bound each bond's capacity by its members' sections: where the smaller
 * section is under the bond's area, its stress limits scale by section / area
 * (so limit x area = joint stress x section). Returns the bonds changed.
 */
export function applySectionBound(parts, bonds) {
  const byId = new Map(parts.map((p) => [p.id, p]));
  const section = new Map(parts.map((p) => [p.id, memberSection(p)]));
  const changed = [];
  for (const bond of bonds) {
    if (bond.attachment === 'wheel-mount') continue;
    if (!byId.has(bond.a) || !byId.has(bond.b)) continue;
    const limit = Math.min(section.get(bond.a), section.get(bond.b));
    if (!(limit < bond.area)) continue;
    const scale = limit / bond.area;
    bond.strength = { ...bond.strength, ...Object.fromEntries(LIMITS.map((k) => [k, bond.strength[k] * scale])) };
    bond.realCapacity = { version: REAL_JOINT_CAPACITY_VERSION, sectionM2: limit, scale };
    changed.push(bond);
  }
  return changed;
}

/**
 * Ductile metal joints (docs/verification/SCENARIOS.md, engine item 5).
 *
 * A metal joint does not fracture at its capacity: it yields and keeps
 * carrying it while it deforms, and ruptures only once its deformation is
 * spent. The stage models that as a ductile material
 * (PxDestructionMaterial::ductileSlip, the impact solve): at capacity the
 * joint carries its capacity and breaks when its slip over a tick passes the
 * ultimate slip. It is the building joints' model (town-kit materials.mjs
 * ULTIMATE_SLIP; veneer-houses.mjs jointMaterial), and as there, under real
 * capacities a joint is elastic up to its capacity (elastic = fatal): the
 * stage's only path between yield and rupture is section loss at its damage
 * rate (FIDELITY_AUDIT C2, a MODEL), where steel between f_y and f_u strain
 * hardens and holds (EN 1993-1-1 3.2.2: f_u / f_y >= 1.10).
 *
 * Ultimate slip: a metal joint ruptures by necking of the metal that carries
 * it -- the parent member beside a full-strength weld (EN 1993-1-8 4.7), the
 * net section, the fastener. Necking elongation scales with the square root of
 * the section (Barba's law), which is why tensile elongation is specified on
 * the proportional gauge L0 = 5.65 sqrt(S0) (EN ISO 6892-1). So the slip at
 * rupture is A L0 = A 5.65 sqrt(S), with S the section that carries the joint
 * (the member's section where the section bound applies, else the joint's
 * area; one stud of a wheel's set) and A the metal's elongation after fracture:
 * - steel (structural members and their welds): 15% (EN 1993-1-1 3.2.2(1));
 * - wheel studs, property class 10.9: 9% (ISO 898-1 Table 3);
 * - aluminium alloy 6061-T6: 8% (EN 755-2, extrusions).
 * Composites, glass, rubber, webbing and upholstery joints stay brittle.
 */
export const ELONGATION = Object.freeze({ steel: 0.15, stud: 0.09, alloy: 0.08 });
/**
 * Yield over ultimate strength, f_y / f_u, of each joint metal: where its
 * joint starts to yield (the elastic limit) below the capacity it ruptures at
 * (the fatal limit). With the stage's static ductility
 * (PX_DESTRUCTION_STATIC_DUCTILE) a metal joint strain-hardens between the two
 * with no section loss and necks past fatal; without it the band is the
 * damage law's section loss, so it is authored only with the flag
 * (staticDuctility()).
 * - steel: S355, f_y 355 / f_u 490 MPa (EN 10025-2; EN 1993-1-1 Table 3.1);
 * - wheel studs, property class 10.9: R_p0.2 940 / R_m 1040 MPa (ISO 898-1 Table 3);
 * - aluminium 6061-T6 extrusion: R_p0.2 240 / R_m 260 MPa (EN 755-2).
 */
export const YIELD_RATIO = Object.freeze({ steel: 355 / 490, stud: 940 / 1040, alloy: 240 / 260 });
/** PX_DESTRUCTION_STATIC_DUCTILE=1: the stage yields and necks metal joints in its static verdict. */
export const staticDuctility = () => (globalThis.process?.env?.PX_DESTRUCTION_STATIC_DUCTILE ?? '0') === '1';
/** EN ISO 6892-1 proportional gauge: L0 = 5.65 sqrt(S0). */
export const PROPORTIONAL_GAUGE = 5.65;
/** Studs per wheel mount the measured interface stands for (strength-profile.mjs: ten M22 studs, 38 cm2). */
export const STUDS_PER_MOUNT = 10;

/**
 * Make each metal joint ductile: its ultimate slip from the section that
 * carries it, elastic up to its capacity. A joint's metal is the joint profile
 * it was given (strength-profile.mjs jointStrength: a dissimilar joint takes
 * its weaker constituent's, so steel-to-alloy is alloy and steel-to-glass is
 * glass), read from its modulus, not from its parts' labels (a merged chunk
 * keeps one label for several materials). Run after applySectionBound (it
 * reads the bound's section). Returns the bonds changed.
 */
export function applyDuctility(parts, bonds) {
  const changed = [];
  for (const bond of bonds) {
    const metal = Object.keys(ELONGATION).find((m) => bond.strength.elasticModulus === jointMaterials[m].elasticModulus);
    if (!metal) continue;
    const section = bond.attachment === 'wheel-mount' ? bond.area / STUDS_PER_MOUNT : (bond.realCapacity?.sectionM2 ?? bond.area);
    if (!(section > 0)) continue;
    const s = bond.strength;
    // Elastic to capacity, unless the stage yields metal joints statically:
    // then they yield at f_y / f_u of it (YIELD_RATIO).
    const y = staticDuctility() ? YIELD_RATIO[metal] : 1;
    bond.strength = { ...s, compressionElastic: s.compressionFatal * y, tensionElastic: s.tensionFatal * y, shearElastic: s.shearFatal * y,
      ductileSlip: ELONGATION[metal] * PROPORTIONAL_GAUGE * Math.sqrt(section) };
    changed.push(bond);
  }
  return changed;
}
