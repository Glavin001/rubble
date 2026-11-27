// Generated from engine-softbody's Rust types (crash_reference::types::typescript). Do not edit:
// `BLESS=1 cargo test -p engine-softbody --test studio_types` regenerates it.

export type JsonValue = number | string | boolean | Array<JsonValue> | { [key in string]?: JsonValue } | null;

export type ScenarioDesc = { id: string, title: string, notes: string, bodies: Array<BodyDesc>, boxes: Array<BoxDesc>, 
/**
 * Named cameras (world). A reference image's own camera lives on its frame.
 */
cameras: Array<Camera>, 
/**
 * Named instants (ms after the computed first contact).
 */
moments: Array<Moment>, 
/**
 * Simulated seconds after first contact.
 */
secs_after_impact: number, 
/**
 * The contact setting it is run in by default.
 */
contact: Contact, };

export type BodyDesc = { 
/**
 * The name metrics and reports use (`car`, `truck`...).
 */
name: string, 
/**
 * A generated design (`suv_4x4`, `sedan`, `wagon`, `truck_cab`, `box_trailer`, `mdb214`...).
 */
design: string, 
/**
 * Where its own origin stands (m, world; y is the ground's height, normally 0: it is
 * stood on its wheels from there).
 */
at: [number, number, number], 
/**
 * Its heading from +Z toward +X (degrees).
 */
yaw_deg: number, 
/**
 * Its speed along its heading (km/h).
 */
speed_kmh: number, 
/**
 * Its velocity when it does not move along its heading (m/s, world); overrides the
 * speed.
 */
velocity?: [number, number, number], 
/**
 * It rides a side-impact carrier (its tyres on low-friction pads).
 */
carrier: boolean, };

export type BoxDesc = { name: string, 
/**
 * Centre (m, world).
 */
centre: [number, number, number], 
/**
 * Half extents along its own axes (m).
 */
half: [number, number, number], 
/**
 * Its turn about +Y (degrees; 0: axis-aligned).
 */
yaw_deg: number, };

export type Camera = { name: string, eye: [number, number, number], target: [number, number, number], };

export type Moment = { name: string, after_impact_ms: number, };

export type Contact = "discrete" | "edge" | "continuous";

export type CrashReport = { id: string, contact: Contact, run: number, metrics: { [key in string]: number }, results: Array<CriterionResult>, 
/**
 * The run's time histories ([`metrics::series`]); None from an executor that keeps only
 * the metrics.
 */
series?: unknown, };

export type CriterionResult = { name: string, metric: string, value: number | null, lo: number | null, hi: number | null, tier: Tier, pass: boolean, 
/**
 * Failing it fails the run ([`Criterion::asserted`]).
 */
asserted: boolean, why: string, basis: string, };

export type Tier = "invariant" | "physics" | "shape";

export type Registry = { scenarios: Array<ScenarioInfo>, contacts: Array<Contact>, runs: number, sampleSteps: number, dt: number, };

export type ScenarioInfo = { id: string, title: string, reference: ReferenceInfo, designs: Array<string>, 
/**
 * The designs a page loads a mod folder for (the others are built from code).
 */
folderDesigns: Array<string>, secsAfterImpact: number, cameras: Array<Camera>, moments: Array<Moment>, criteria: Array<CriterionInfo>, 
/**
 * Recreated from data in the Reference Studio (law-based criteria only).
 */
recreated: boolean, 
/**
 * The contact setting it runs in by default.
 */
contact: Contact, 
/**
 * A recreated scenario's description.
 */
desc?: ScenarioDesc, };

export type ReferenceInfo = { video: string, atS: Array<number>, 
/**
 * The video at each of its times.
 */
links: Array<string>, realTest: string, };

export type CriterionInfo = { name: string, metric: string, lo: number | null, hi: number | null, tier: Tier, why: string, basis: BasisInfo, asserted: boolean, };

export type BasisInfo = { kind: string, source?: string, };

export type Header = { scenario: string, 
/**
 * What it was made from (`ScenarioDesc::digest`, or `builtin-<id>-<contact>`).
 */
hash: string, 
/**
 * What ran it: `webgpu` (the page's GPU solver, what production runs) or `cpu-fallback`
 * (the wasm CPU reference: the automated smoke test only).
 */
executor: string, contact: Contact, run: number, 
/**
 * First contact (s from the start).
 */
impact_s: number, 
/**
 * Between samples (s).
 */
sample_s: number, samples: number, bodies: Array<RecBody>, 
/**
 * The run's report (metrics and criteria), once complete.
 */
report?: CrashReport, };

export type RecBody = { name: string, design: string, nodes: number, beams: number, };

export type FreeCamera = { name: string, eye: [number, number, number], target: [number, number, number], up: [number, number, number], fov_y_deg: number, };

export type FrameCamera = { eye: [number, number, number], target: [number, number, number], up: [number, number, number], fov_y_deg: number, 
/**
 * The region it applies to (its crop's aspect), or null for the whole image.
 */
region: string | null, };

export type Vis = "clear" | "blurred" | "hidden";

export type ElementValue = { 
/**
 * A keypoint.
 */
p?: [number, number], 
/**
 * A polyline, in order.
 */
pts?: Array<[number, number]>, 
/**
 * A circle's centre and a point on its rim.
 */
c?: [number, number], rim?: [number, number], vis: Vis, };

export type Instance = { id: string, 
/**
 * A structure schema id (`structures/<id>.json`).
 */
schema: string, 
/**
 * "left" or "right": where the structure's front is in the image (default right).
 */
front: 'left' | 'right', elements: Record<string, ElementValue>, };

export type SceneSource = { desc?: ScenarioDesc, id?: string, 
/**
 * The contact setting (default: the description's, or edge).
 */
contact?: Contact, };

export type SceneBody = { name: string, design: string, 
/**
 * Its first node in the scene's position arrays (all bodies' nodes in order).
 */
node_offset: number, nodes: number, 
/**
 * Its collision surface: node indices (its own), three per triangle.
 */
triangles: Array<number>, mass_kg: number, at: [number, number, number], yaw_deg: number, speed_kmh: number, 
/**
 * Its box at spawn (world), as the first contact is computed from.
 */
box_lo: [number, number, number], box_hi: [number, number, number], };

export type SceneBox = { name: string, centre: [number, number, number], half: [number, number, number], yaw_deg: number, };

export type SceneInfo = { id: string, title: string, recreated: boolean, 
/**
 * What its recording is filed under (the description and contact setting).
 */
hash: string, contact: Contact, 
/**
 * First contact (s from the start): computed for a data scenario
 * ([`desc::first_contact`]), stated by a Rust one.
 */
impact_s: number, duration_s: number, samples: number, sample_s: number, bodies: Array<SceneBody>, boxes: Array<SceneBox>, cameras: Array<Camera>, moments: Array<Moment>, criteria: Array<CriterionInfo>, 
/**
 * Milliseconds the placement took (the live-edit cost).
 */
place_ms: number, };

export type RecordingStatus = { present: boolean, 
/**
 * Made from the scene as it is now (its hash is the scene's).
 */
fresh: boolean, executor: string, hash: string, samples: number, duration_s: number, impact_s: number, sample_s: number, 
/**
 * The first and last sample's time (ms after first contact).
 */
start_ms: number, end_ms: number, has_report: boolean, };

export type CpuProgress = { sample: number, samples: number, done: boolean, };

export type RubbleSpec = { 
/**
 * Ms after first contact (between samples: interpolated).
 */
at_ms: number, 
/**
 * The image's camera; absent: the car's own side frame (metres).
 */
camera?: FrameCamera, width: number, height: number, body: number, };

export type TracingSpec = { instance: Instance, width: number, height: number, };

export type ShapeSpec = { tracing?: TracingSpec, rubble?: RubbleSpec, };

export type CompareSource = { name: string, before: ShapeSpec, now: ShapeSpec, };
