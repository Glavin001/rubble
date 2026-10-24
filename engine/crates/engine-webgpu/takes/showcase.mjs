// Showcase takes (record-takes.mjs --page showcase): rubble's world API in the browser
// (showcase/index.html, engine-world-wasm), its scenarios drawn by the lab's renderer and
// run by the lab's director (fracture/director.mjs: the segment fields are the lab takes',
// takes/lab.mjs). The same scenes as the lab's world-API takes (rr_drive_storefront,
// bo_drive_viaduct), here in the product page instead of the stress lab.

// The page's state for a take: the scenario, loaded paused (the take starts it).
const take = (scenario, script) => ({ page: 'showcase/index.html', setup: `await window.__SHOWCASE__.open(${JSON.stringify(scenario)});`, script });

export const SHOWCASE_TAKES = {
  // V5 driven: the sedan and the six-wheel truck into the storefront row on their wheels.
  sc_ram_raid: take('rr_drive_storefront', [
    { n: 60, caption: 'Showcase · a 1.3 t sedan, driven by its engine and wheels up the street at the middle shop', cam: { orbit: [-0.5, 0.18, 12], to: [-0.45, 0.16, 11], at: [0.55, -35.7, 1.2], atTo: [0.55, -33, 1.2] } },
    { n: 140, caption: 'Full throttle from a standstill: about 50 km/h at the shop front', cam: { orbit: [-0.45, 0.16, 11], to: [-0.6, 0.14, 10], at: [0.55, -33, 1.2], atTo: [0.55, -10, 1.2] } },
    { n: 240, slow: 4, caption: 'Quarter speed: its body strikes the window a step before it meets it, and it drives into the shop', cam: { orbit: [-0.6, 0.14, 10], to: [-1.0, 0.2, 10], at: [0.55, -10, 1.2], atTo: [0.55, -2, 1.3] } },
    { n: 60, caption: 'A 5.5 t six-wheel truck, a second behind, at the fourth shop', cam: { orbit: [-0.5, 0.18, 20], to: [-0.6, 0.16, 18], at: [11.5, -16, 1.8], atTo: [11.5, -8, 1.8] } },
    { n: 240, slow: 4, caption: 'Quarter speed: its front wheels meet the brick under the window; the truck breaks through and slews in', cam: { orbit: [-0.6, 0.16, 18], to: [-1.1, 0.22, 16], at: [11.5, -8, 1.8], atTo: [11.5, -3, 2.0] } },
    { n: 150, caption: 'Real time: both vehicles drove in on their own wheels', cam: { orbit: [-1.1, 0.22, 16], to: [-0.5, 0.3, 32], at: [11.5, -3, 2.0], atTo: [6, -3, 2.5] } },
  ]),
  // Bridge Out driven: traffic over the viaduct and under it; the hinge pier's charge drops
  // the drop-in spans with their traffic.
  sc_bridge_out: take('bo_drive_viaduct', [
    { n: 120, caption: 'Showcase · traffic on a four-span viaduct, 10 m up: six vehicles driving across it, two on the road below', cam: { rel: [0.45, 0.05, 0.5], relTo: [0.32, 0.03, 0.42] } },
    { n: 1, act: { detonate: 'hinge pier' }, caption: 'Charges blow the middle pier and cut the two drop-in spans loose' },
    { n: 30, caption: 'Charges blow the middle pier and cut the two drop-in spans loose' },
    { n: 240, slow: 2, caption: 'Half speed: the spans swing down about their seats with the traffic on them', cam: { rel: [0.32, 0.03, 0.42], relTo: [0.7, 0.1, 0.45] } },
    { n: 270, caption: 'Real time: the traffic behind drives on into the gap', cam: { rel: [0.7, 0.1, 0.45], relTo: [1.05, 0.16, 0.6] } },
  ]),
};
