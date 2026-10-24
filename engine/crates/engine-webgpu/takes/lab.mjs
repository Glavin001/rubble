// Fracture lab takes (record-takes.mjs), run through fracture/director.mjs `__record`.
// Segment fields: n (video frames at 60 fps), slow (1/slow speed, lossless), caption,
// cam ({orbit: [yaw, pitch, dist], to?, spin?, target?} or {rel, relTo?, lift?, shift?}, and
// at / atTo: a target in build coordinates; fracture/director.mjs), act ('fire' | {shoot:
// [u, v]} | {ray: [origin, direction]} | {detonate: name} | {launch: name}: a pack's launched
// bonded bodies, '' for all), view ({bonds, color}: color 3, materials untinted, unless a
// segment says otherwise), mass / speed (the shot sliders, powers of two).

// The lab's page state for a take: its Scene menu key and the menus' values, then a load.
// `preset`: the stress modes' Ball menu (stress_ref::wrecking::BALLS: 0 the 10 t ball,
// 1 the 50 t ball, 2 the 15 m demolition sphere, 3 the 2 t car stand-in, 4 the 40 t jet
// stand-in), applied by reloading once it is listed.
const setup = ({ scene, bonds = 'none', hz = 0, radius = 0, strength = 0, level, preset }) => `
  const $ = (id) => document.getElementById(id);
  $('bond-mode').value = ${JSON.stringify(bonds)};
  $('physics-hz').value = ${JSON.stringify(String(hz))};
  $('ball-radius').value = ${JSON.stringify(String(radius))};
  $('bond-strength').value = ${JSON.stringify(String(strength))};
  await window.__FRACTURE_DEMO__.load(${JSON.stringify(scene)});
  ${level === undefined ? '' : `await window.__FRACTURE_DEMO__.selectLevel(${level});`}
  ${preset === undefined ? '' : `$('ball-preset').value = ${JSON.stringify(String(preset))}; await window.__FRACTURE_DEMO__.selectLevel(0);`}
  const err = document.getElementById('status');
  if (err.classList.contains('error')) throw new Error(err.textContent);`;

const take = (state, script) => ({ page: 'fracture/index.html', setup: setup(state), script });

// Bonded vehicles (stress_ref::vehicles), threshold profile: the launch released at the
// take's frame ({launch: ''}), the camera following it in (`at` / `atTo`: the orbit's target
// as built, Z-up).
const VEHICLE_TAKES = {
  // A crash test (V1 V2): the 1.4 t car slides 12 m on its locked wheels into a rigid
  // barrier at about 60 km/h; its front comes apart and the cabin holds.
  vh_car_barrier_crash: take({ scene: 'vh_car_barrier', bonds: 'threshold' }, [
    { n: 40, caption: 'A 1.4 t car of 50 bonded parts: chassis, engine bay, doors, wheels, glass', cam: { orbit: [-0.75, 0.2, 9], to: [-0.6, 0.17, 8.5], at: [-14.25, 0, 0.6] } },
    { n: 1, act: { launch: '' }, caption: 'Launched at 71 km/h: it slides 12 m on its locked wheels into a rigid barrier' },
    { n: 36, caption: 'Launched at 71 km/h: it slides 12 m on its locked wheels into a rigid barrier', cam: { orbit: [-0.6, 0.17, 8.5], to: [-0.2, 0.14, 7.5], at: [-14.25, 0, 0.6], atTo: [-2.6, 0, 0.6] } },
    { n: 200, slow: 4, caption: 'Quarter speed, 60 km/h into the barrier: the front comes apart, the cabin holds', cam: { orbit: [-0.2, 0.14, 7.5], to: [0.75, 0.3, 7], at: [-2.6, 0, 0.6] } },
    { n: 120, caption: 'Real time', cam: { orbit: [0.75, 0.3, 7], to: [1.05, 0.38, 9], at: [-2.6, 0, 0.6] } },
  ]),
  // The 10.4 t bus into the barrier (V1 V4).
  vh_bus_barrier_crash: take({ scene: 'vh_bus_barrier', bonds: 'threshold' }, [
    { n: 40, caption: 'A 10.4 t bus of 87 bonded parts: underframes, axles, panels, pillars, glass', cam: { orbit: [-0.8, 0.2, 22], to: [-0.65, 0.18, 21], at: [-18.2, 0, 1.6] } },
    { n: 1, act: { launch: '' }, caption: 'Launched at 71 km/h: it slides 12 m on its locked wheels into a rigid barrier' },
    { n: 36, caption: 'Launched at 71 km/h: it slides 12 m on its locked wheels into a rigid barrier', cam: { orbit: [-0.65, 0.18, 21], to: [-0.3, 0.15, 18], at: [-18.2, 0, 1.6], atTo: [-6.5, 0, 1.6] } },
    { n: 200, slow: 4, caption: 'Quarter speed: the wheels and the bumper tear off; the body holds', cam: { orbit: [-0.3, 0.15, 18], to: [0.7, 0.3, 17], at: [-6.5, 0, 1.6] } },
    { n: 120, caption: 'Real time', cam: { orbit: [0.7, 0.3, 17], to: [1.0, 0.38, 21], at: [-6.5, 0, 1.6] } },
  ]),
  // Ram Raid (V5): the car, then the bus, driven at the storefront row from a street level
  // with its footings; each slides 12 m and hits at about 54 km/h.
  rr_storefront_car_bus: take({ scene: 'rr_vehicles_storefront', bonds: 'threshold' }, [
    { n: 40, caption: 'Five brick shops on a street; a 1.4 t car lines up on the middle shop window', cam: { orbit: [-0.45, 0.16, 12], to: [-0.4, 0.14, 11], at: [0.55, -17.75, 1.2] } },
    { n: 1, act: { launch: 'car' }, caption: 'The car: launched at 69 km/h, 12 m out' },
    { n: 42, caption: 'The car: launched at 69 km/h, 12 m out', cam: { orbit: [-0.4, 0.14, 11], to: [-0.8, 0.12, 10], at: [0.55, -17.75, 1.2], atTo: [0.55, -4.5, 1.4] } },
    { n: 200, slow: 4, caption: 'Quarter speed: through the shop window into the shop; the car crumples', cam: { orbit: [-0.8, 0.12, 10], to: [-0.45, 0.2, 10], at: [0.55, -4.5, 1.4], atTo: [0.55, -1.5, 1.4] } },
    { n: 30, caption: 'Now the bus: 10.4 t, at the fourth shop', cam: { orbit: [-0.35, 0.16, 20], to: [-0.3, 0.15, 19], at: [11.5, -21.7, 1.8] } },
    { n: 1, act: { launch: 'bus' }, caption: 'The bus: launched at 69 km/h, 12 m out' },
    { n: 42, caption: 'The bus: launched at 69 km/h, 12 m out', cam: { orbit: [-0.3, 0.15, 19], to: [-0.75, 0.15, 17], at: [11.5, -21.7, 1.8], atTo: [11.5, -7.5, 2.2] } },
    { n: 200, slow: 4, caption: 'Quarter speed: the window, piers and fascia give; the bus drives into the shop', cam: { orbit: [-0.75, 0.15, 17], to: [-1.2, 0.22, 17], at: [11.5, -7.5, 2.2] } },
    { n: 120, caption: 'Real time', cam: { orbit: [-1.2, 0.22, 17], to: [-0.6, 0.3, 30], at: [11.5, -7.5, 2.2], atTo: [5, -3, 2.5] } },
  ]),
  // V5 driven (stress_ref::drive): rubble's raycast sedan and six-wheel truck, in the world
  // API with the threshold model, driven by their wheels from 30 m out at full throttle;
  // their chassis boxes strike the shop fronts (strike prediction), so they break in. The
  // truck's front wheels meet the stallriser (a raycast wheel cannot climb it): it slews in.
  rr_drive_storefront: take({ scene: 'rr_drive_storefront', bonds: 'threshold' }, [
    { n: 60, caption: 'A 1.3 t sedan, driven: engine, gearbox and wheels take it up the street at the middle shop', cam: { orbit: [-0.5, 0.18, 12], to: [-0.45, 0.16, 11], at: [0.55, -35.7, 1.2], atTo: [0.55, -33, 1.2] } },
    { n: 140, caption: 'Full throttle from a standstill: about 50 km/h at the shop front', cam: { orbit: [-0.45, 0.16, 11], to: [-0.6, 0.14, 10], at: [0.55, -33, 1.2], atTo: [0.55, -10, 1.2] } },
    { n: 240, slow: 4, caption: 'Quarter speed: its body strikes the window a step before it meets it, and it drives into the shop', cam: { orbit: [-0.6, 0.14, 10], to: [-1.0, 0.2, 10], at: [0.55, -10, 1.2], atTo: [0.55, -2, 1.3] } },
    { n: 60, caption: 'A 5.5 t six-wheel truck, a second behind, at the fourth shop', cam: { orbit: [-0.5, 0.18, 20], to: [-0.6, 0.16, 18], at: [11.5, -16, 1.8], atTo: [11.5, -8, 1.8] } },
    { n: 240, slow: 4, caption: 'Quarter speed: its front wheels meet the brick under the window; the truck breaks through and slews in', cam: { orbit: [-0.6, 0.16, 18], to: [-1.1, 0.22, 16], at: [11.5, -8, 1.8], atTo: [11.5, -3, 2.0] } },
    { n: 150, caption: 'Real time: both vehicles drove in on their own wheels', cam: { orbit: [-1.1, 0.22, 16], to: [-0.5, 0.3, 32], at: [11.5, -3, 2.0], atTo: [6, -3, 2.5] } },
  ]),
  // V5 into the apartment block's ground floor: the car at a brick-sill bay, then the bus.
  rr_apartment_car_bus: take({ scene: 'rr_vehicles_apartment', bonds: 'threshold' }, [
    { n: 40, caption: 'A 7-storey apartment block; a 1.4 t car lines up on a ground-floor bay', cam: { orbit: [-0.45, 0.16, 13], to: [-0.4, 0.14, 12], at: [-9, -20.4, 1.2] } },
    { n: 1, act: { launch: 'car' }, caption: 'The car: launched at 69 km/h, 12 m out' },
    { n: 42, caption: 'The car: launched at 69 km/h, 12 m out', cam: { orbit: [-0.4, 0.14, 12], to: [-0.85, 0.14, 11], at: [-9, -20.4, 1.2], atTo: [-9, -7.5, 1.4] } },
    { n: 160, slow: 4, caption: 'Quarter speed: the brick sill gives, but the car does not get through', cam: { orbit: [-0.85, 0.14, 11], to: [-1.25, 0.22, 12], at: [-9, -7.5, 1.4] } },
    { n: 30, caption: 'Now the bus: 10.4 t, at the next bay but one', cam: { orbit: [0.35, 0.16, 20], to: [0.3, 0.15, 19], at: [9, -24.3, 1.8] } },
    { n: 1, act: { launch: 'bus' }, caption: 'The bus: launched at 69 km/h, 12 m out' },
    { n: 42, caption: 'The bus: launched at 69 km/h, 12 m out', cam: { orbit: [0.3, 0.15, 19], to: [0.75, 0.15, 17], at: [9, -24.3, 1.8], atTo: [9, -10, 2.2] } },
    { n: 200, slow: 4, caption: 'Quarter speed: the infill breaks under the first-floor beam; the bus is thrown back', cam: { orbit: [0.75, 0.15, 17], to: [1.2, 0.22, 18], at: [9, -10, 2.2] } },
    { n: 120, caption: 'Real time', cam: { orbit: [1.2, 0.22, 18], to: [0.5, 0.3, 34], at: [9, -10, 2.2], atTo: [0, -6, 4] } },
  ]),
};

export const LAB_TAKES = {
  ...VEHICLE_TAKES,
  // Ram Raid target (stress_ref::buildings): the 2 t car stand-in into a shopfront.
  rr_storefront_car: take({ scene: 'rr_storefront_row', bonds: 'stress', preset: 3 }, [
    { n: 40, caption: 'Five shops in brick, glass and concrete: 1,203 bonded chunks', cam: { rel: [0.32, -0.05, 0.55], relTo: [0.22, -0.05, 0.5] } },
    { n: 1, act: 'fire', caption: 'A 2 t car stand-in at 15 m/s into a shop window' },
    { n: 30, caption: 'A 2 t car stand-in at 15 m/s into a shop window' },
    { n: 180, slow: 2, caption: 'Half speed: the window and the stallriser go; the shop stands', cam: { rel: [0.22, -0.05, 0.5], relTo: [-0.25, 0.0, 0.55] } },
  ]),
  // Fly-through (S3): a bonded 35 t regional jet (stress_ref::vehicles::airliner) thrown at
  // the building at 70 m/s along its line of attack, released 40 m out (threshold); the
  // camera follows it in. Replaces the 40 t ball stand-in.
  ft_glass_tower_airliner: take({ scene: 'ft_airliner_tower', bonds: 'threshold' }, [
    { n: 1, act: { launch: 'airliner' }, caption: 'A 35 t, 30 m regional jet of 36 bonded parts, thrown at 252 km/h', cam: { orbit: [-0.6, 0.1, 40], at: [0, -61.2, 20.3] } },
    { n: 18, caption: 'A 35 t, 30 m regional jet of 36 bonded parts, thrown at 252 km/h', cam: { orbit: [-0.6, 0.1, 40], to: [-0.7, 0.1, 40], at: [0, -61.2, 20.3], atTo: [0, -39.2, 20.3] } },
    { n: 100, slow: 4, caption: 'Quarter speed: it meets the facade at 70 m/s', cam: { orbit: [-0.7, 0.1, 40], to: [-1.15, 0.12, 46], at: [0, -39.2, 20.3], atTo: [0, -8, 20.3] } },
    { n: 200, slow: 2, caption: 'Half speed: the nose breaks up in the curtain wall; the jet pitches up and slides down the facade', cam: { orbit: [-1.15, 0.12, 46], to: [-0.75, 0.2, 85], at: [0, -8, 20.3], atTo: [0, 0, 20.3] } },
    { n: 300, caption: 'Real time: a slot of broken panes and floors; the tower stands (no stress solve in threshold)', cam: { orbit: [-0.75, 0.2, 85], to: [-0.35, 0.25, 95], at: [0, 0, 20.3], atTo: [0, 0, 16.2] } },
  ]),
  ft_apartment_airliner: take({ scene: 'ft_airliner_apartment', bonds: 'threshold' }, [
    { n: 1, act: { launch: 'airliner' }, caption: 'A 35 t, 30 m regional jet thrown at the apartment block at 252 km/h', cam: { orbit: [-0.6, 0.1, 40], at: [0, -61.2, 11.0] } },
    { n: 18, caption: 'A 35 t, 30 m regional jet thrown at the apartment block at 252 km/h', cam: { orbit: [-0.6, 0.1, 40], to: [-0.7, 0.1, 40], at: [0, -61.2, 11.0], atTo: [0, -39.2, 11.0] } },
    { n: 100, slow: 4, caption: 'Quarter speed: it meets the facade at 70 m/s', cam: { orbit: [-0.7, 0.1, 40], to: [-1.15, 0.12, 46], at: [0, -39.2, 11.0], atTo: [0, -8, 11.0] } },
    { n: 200, slow: 2, caption: 'Half speed: it punches through the middle bays, five storeys of them', cam: { orbit: [-1.15, 0.12, 46], to: [-0.75, 0.2, 70], at: [0, -8, 11.0], atTo: [0, 0, 11.0] } },
    { n: 300, caption: 'Real time: the bays above the hole hang on; the jet falls back out', cam: { orbit: [-0.75, 0.2, 70], to: [-0.35, 0.25, 80], at: [0, 0, 11.0], atTo: [0, 0, 8.8] } },
  ]),
  // Topple (stress_ref::topple), threshold profile: the charges cut the street-side
  // columns, then the back line. The tower leans out over the street as one mass, turning
  // on its hinge line (about 4 s to its front edge meeting the ground), and lands across
  // the street at about 6.3 s, in Chrome as natively (README, Topple).
  topple_detonate: take({ scene: 'topple_tower', bonds: 'threshold' }, [
    { n: 50, caption: 'A 10-storey tower beside a street: 425 bonded chunks, charges on the ground storey', cam: { rel: [0.9, 0.0, 1.15], relTo: [1.1, 0.0, 1.1], shift: [0, 0, -8] } },
    { n: 1, act: { detonate: '' }, caption: 'Charges: the street-side columns, then the back line 0.25 s later' },
    { n: 120, caption: 'Charges: the street-side columns, then the back line 0.25 s later', cam: { rel: [1.1, 0.0, 1.1], relTo: [1.3, 0.02, 1.15], shift: [0, 0, -8] } },
    { n: 150, caption: 'It leans out over the street as one mass, turning on its back line', cam: { rel: [1.3, 0.02, 1.15], relTo: [1.45, 0.04, 1.2], shift: [0, 0, -8] } },
    { n: 240, slow: 2, caption: 'Half speed: it lands across the street; its base and street face break up', cam: { rel: [1.45, 0.04, 1.2], relTo: [1.6, 0.05, 1.2], shift: [0, 0, -8] } },
    { n: 120, caption: 'Real time: the upper storeys lie across the street in one long piece', cam: { rel: [1.6, 0.05, 1.2], relTo: [1.85, 0.12, 1.1], shift: [0, 0, -8] } },
  ]),
  // Skybridge (stress_ref::skybridge) under blast_parity: one charge takes out the east
  // connection; the stress solve tears the west bearings, and the walkway hangs from the
  // west ledge on its keepers, east end down, until the ledge tears out of the facade; it
  // comes down on the plaza in its sections. The camera targets the walkway (lab frame:
  // Y-up, the footprint centred).
  sb_skybridge: take({ scene: 'sb_skybridge', bonds: 'stress' }, [
    { n: 50, caption: 'Skybridge between two towers: 992 bonded chunks', cam: { rel: [0.0, -0.1, 0.44], relTo: [0.08, -0.1, 0.41], target: [0, 10.5, -1.75] } },
    { n: 1, act: { detonate: 'east connection' }, caption: 'A charge takes out the east connection' },
    { n: 20, caption: 'A charge takes out the east connection', cam: { rel: [0.08, -0.1, 0.41], relTo: [0.1, -0.1, 0.4], target: [0, 10.5, -1.75] } },
    { n: 160, slow: 2, caption: 'Half speed: it hangs from the west end, then tears out', cam: { rel: [0.1, -0.1, 0.4], relTo: [0.25, -0.08, 0.4], target: [0, 10.5, -1.75] } },
    { n: 120, slow: 2, caption: 'Half speed: big sections come down on the plaza', cam: { rel: [0.25, -0.08, 0.4], relTo: [0.35, -0.03, 0.43], target: [0, 10.5, -1.75] } },
    { n: 120, caption: 'Real time: the towers stand', cam: { rel: [0.35, -0.03, 0.43], relTo: [0.45, 0.04, 0.52], target: [0, 10.5, -1.75] } },
  ]),
  // Bridge Out (stress_ref::viaduct): the 50 t ball into the hinge pier under
  // blast_parity; the two drop-in spans swing down whole and break on the road below.
  bo_viaduct_pier: take({ scene: 'bo_viaduct', bonds: 'stress', preset: 1 }, [
    { n: 50, caption: 'Four-span concrete viaduct, 10 m up: 347 bonded chunks under the GPU stress solver', cam: { rel: [0.4, -0.02, 0.6], relTo: [0.3, -0.04, 0.5] } },
    { n: 1, act: 'fire', caption: 'A 50 t wrecking ball into the middle pier' },
    { n: 25, caption: 'A 50 t wrecking ball into the middle pier' },
    { n: 220, slow: 2, caption: 'Half speed: the hinge lets go and both spans swing down whole', cam: { rel: [0.3, -0.04, 0.5], relTo: [0.5, 0.0, 0.5] } },
    { n: 240, caption: 'They break on the road below; the outer piers follow', cam: { rel: [0.5, 0.0, 0.5], relTo: [0.75, 0.06, 0.6] } },
  ]),
  // Bridge Out with traffic (S2 + V3 + V4, stress_ref::vehicles): 7 cars and 2 buses
  // stalled on the two drop-in spans and on the road under span 2; the same 50 t ball into
  // the hinge pier under blast_parity. The spans carry the traffic down onto the road.
  bo_viaduct_traffic: take({ scene: 'bo_viaduct_traffic', bonds: 'stress', preset: 1 }, [
    { n: 50, caption: 'Stalled traffic: 7 cars and 2 buses on the drop-in spans and the road below', cam: { rel: [0.45, 0.05, 0.5], relTo: [0.3, 0.03, 0.42] } },
    { n: 1, act: 'fire', caption: 'A 50 t wrecking ball into the middle pier' },
    { n: 25, caption: 'A 50 t wrecking ball into the middle pier' },
    { n: 240, slow: 2, caption: 'Half speed: the hinge lets go; the spans swing down with the traffic on them', cam: { rel: [0.3, 0.03, 0.42], relTo: [0.75, 0.1, 0.45] } },
    { n: 240, caption: 'They land on the road and its queue; the outer spans follow', cam: { rel: [0.75, 0.1, 0.45], relTo: [1.1, 0.16, 0.6] } },
  ]),
  // Bridge Out driven (stress_ref::vehicles `bo_drive_viaduct`): rubble's raycast sedans and
  // six-wheel trucks cruising both ways over the viaduct and along the road under it, in
  // the world API with the threshold model. The hinge pier's charge (cut between two steps)
  // frees the drop-in spans from their seats; they come down with their traffic. (Under
  // the threshold model the ball alone breaks only what it meets: bo_viaduct_traffic is the
  // ball's beat, under blast_parity.)
  bo_drive_viaduct: take({ scene: 'bo_drive_viaduct', bonds: 'threshold' }, [
    { n: 120, caption: 'Traffic on a four-span viaduct, 10 m up: six vehicles driving across it, two on the road below', cam: { rel: [0.45, 0.05, 0.5], relTo: [0.32, 0.03, 0.42] } },
    { n: 1, act: { detonate: 'hinge pier' }, caption: 'Charges blow the middle pier and cut the two drop-in spans loose' },
    { n: 30, caption: 'Charges blow the middle pier and cut the two drop-in spans loose' },
    { n: 240, slow: 2, caption: 'Half speed: the spans swing down about their seats with the traffic on them', cam: { rel: [0.32, 0.03, 0.42], relTo: [0.7, 0.1, 0.45] } },
    { n: 270, caption: 'Real time: the traffic behind drives on into the gap', cam: { rel: [0.7, 0.1, 0.45], relTo: [1.05, 0.16, 0.6] } },
  ]),
  // Wrecking Yard (stress_ref::wrecking): the 50 t ball into the 10-floor frame's corner
  // column line under blast_parity; the collapse climbs while the core stands.
  wy_frame10_ball: take({ scene: 'wy_frame10', bonds: 'stress', preset: 1 }, [
    { n: 50, caption: '10-floor concrete frame: 1,264 bonded chunks under the GPU stress solver', cam: { rel: [0.35, 0.02, 1.2], relTo: [0.15, 0.0, 1.12], lift: -1 } },
    { n: 1, act: 'fire', caption: 'A 50 t wrecking ball at 20 m/s into the corner columns' },
    { n: 30, caption: 'A 50 t wrecking ball at 20 m/s into the corner columns' },
    { n: 240, slow: 2, caption: 'Half speed: the ground storeys give way and the collapse climbs', cam: { rel: [0.15, 0.0, 1.12], relTo: [-0.35, 0.06, 1.15], lift: -1 } },
    { n: 300, caption: 'The upper storeys, the core and the far bays stay standing', cam: { rel: [-0.35, 0.06, 1.15], relTo: [-0.9, 0.04, 1.15], lift: -1 } },
  ]),
  wy_frame10_xray: take({ scene: 'wy_frame10', bonds: 'stress', preset: 1 }, [
    { n: 40, view: { bonds: true, color: 2 }, caption: 'X-ray: every bond of the frame, one colour per rigid body', cam: { rel: [0.35, 0.02, 1.2], lift: -1 } },
    { n: 1, act: 'fire', caption: 'The same 50 t ball' },
    { n: 330, slow: 2, caption: 'Half speed: failed bonds drop out; each piece that breaks away gets its own colour', cam: { rel: [0.35, 0.02, 1.2], relTo: [-0.3, 0.08, 1.18], lift: -1 } },
  ]),
  // The 15 m demolition sphere (277 t, rolling in at 12 m/s) through the round tower.
  wy_round_tower_sphere: take({ scene: 'wy_round_tower', bonds: 'stress', preset: 2 }, [
    { n: 40, caption: 'Round tower: eight floors on a core and a ring of columns', cam: { rel: [0.6, 0.0, 1.15], relTo: [0.5, 0.0, 1.1], lift: -2 } },
    { n: 1, act: 'fire', caption: 'A 15 m demolition sphere rolls in' },
    { n: 60, caption: 'A 15 m demolition sphere rolls in' },
    { n: 240, slow: 2, caption: 'Half speed: the slabs drop floor by floor; the core stays', cam: { rel: [0.5, 0.0, 1.1], relTo: [-0.2, 0.08, 1.1], lift: -2 } },
    { n: 180, caption: 'Real time', cam: { rel: [-0.2, 0.08, 1.1], relTo: [-0.5, 0.14, 1.15], lift: -2 } },
  ]),
  // Threshold fracture of the baked veneer house: one heavy ball, the impact again at
  // quarter speed, then the settle.
  lab_house_threshold: take({ scene: 'house', bonds: 'threshold', level: 1 }, [
    { n: 50, caption: 'Veneer house, Shardwright bake: bonds break when a contact force beats their strength', cam: { orbit: [0.3, 0.22, 13], to: [0.4, 0.2, 15] } },
    { n: 1, act: { shoot: [0.5, 0.6] }, mass: 5, speed: 2, caption: 'A 32x ball at 4x speed' },
    { n: 14, caption: 'Incoming', cam: { orbit: [0.4, 0.2, 15], to: [0.45, 0.2, 12] } },
    { n: 160, slow: 4, caption: 'Quarter speed: the wall lets go in large pieces first, then they break again', cam: { orbit: [0.45, 0.2, 12], to: [0.95, 0.26, 11] } },
    { n: 180, caption: 'Real time: the roof comes down on the debris', cam: { orbit: [0.95, 0.26, 11], to: [1.5, 0.32, 13] } },
  ]),
  // The quasi-static stress solver (blast_parity) with the bond graph drawn: the load
  // path before and after the shot.
  lab_house_stress_xray: take({ scene: 'house', bonds: 'stress' }, [
    { n: 60, view: { bonds: true, color: 2 }, caption: 'Stress solver (blast_parity): every bond, and one colour per rigid body', cam: { orbit: [0.4, 0.3, 16], to: [0.6, 0.3, 22] } },
    { n: 1, act: { shoot: [0.5, 0.6] }, mass: 5, caption: 'A 32x ball into the front wall' },
    { n: 160, slow: 2, caption: 'Half speed: bonds that fail drop out of the graph; each new body gets its own colour', cam: { orbit: [0.6, 0.3, 22], to: [1.1, 0.33, 14] } },
    { n: 120, view: { bonds: false, color: 3 }, caption: 'Materials view', cam: { orbit: [1.1, 0.33, 14], to: [1.5, 0.3, 15] } },
  ]),
  // The lab's wrecking ball (200 t at 40 m/s) through the 7 x 7 grid of cuboid houses.
  lab_cuboid_grid_ball: take({ scene: 'cuboidGrid' }, [
    { n: 60, caption: '49 cuboid houses, every box a rigid body', cam: { orbit: [0.35, 0.35, 140], to: [0.5, 0.3, 120] } },
    { n: 1, act: 'fire', caption: 'The wrecking ball: 200 t at 40 m/s' },
    { n: 300, caption: 'Through the grid', cam: { orbit: [0.5, 0.3, 120], to: [0.9, 0.25, 110] } },
    { n: 180, slow: 2, caption: 'Half speed', cam: { orbit: [0.9, 0.25, 110], to: [1.3, 0.4, 130] } },
  ]),
};
