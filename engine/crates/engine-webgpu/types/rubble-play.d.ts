// Generated from engine-world's Rust types (play::api::typescript). Do not edit:
// `cargo test -p engine-world --test api_types` regenerates it with BLESS=1.
// The web gets these as JS objects; C gets the same JSON (rubble.h, rbWorld_*).

export type JsonValue = number | string | boolean | Array<JsonValue> | { [key in string]?: JsonValue } | null;

export type SceneDef = { "family": "rigid" } & RigidSceneDef | { "family": "vehicle" } & VehicleSceneDef | { "family": "soft" } & SoftSceneDef | { "family": "motion" } & MotionSceneDef | { "family": "terrain" } & HillsDef | { "family": "city" } & CitySceneDef | { "family": "impact" } & ImpactSceneDef | { "family": "showcase" } & ShowcaseSceneDef | { "family": "range" } & RangeSceneDef;

export type RigidSceneDef = { scene: string, levels: number | null, ccd: boolean, spatial: boolean, };

export type VehicleSceneDef = { scene: string, 
/**
 * The player's preset (none: the scene's).
 */
preset: string | null, 
/**
 * Multiplies every mount's break limits (`?strength=`).
 */
strength: number, 
/**
 * The buggies' node–beam frames (`?build=hybrid`).
 */
hybrid: boolean, 
/**
 * Rigid mode for the hybrid buggies (`?rigid=0` turns it off).
 */
rigidMode: boolean, 
/**
 * The player's spawn (x, z, yaw), any part of it (`?spawn=`).
 */
spawn: Array<number>, 
/**
 * Traffic's AI vehicles (`?vehicles=`, 1..60).
 */
vehicles: number, 
/**
 * The bridge's heavy vehicle (`?heavy=`; "none": the player alone).
 */
heavy: string | null, 
/**
 * The tank as before track pods (`?tracks=shells`).
 */
tracksShells: boolean, 
/**
 * Sleep off (`?sleep=off`).
 */
sleepOff: boolean, 
/**
 * The resting-pair carry (`?rest=carry`).
 */
carryRest: boolean, 
/**
 * Continuous collision for the cannonballs (`?ccd=0` turns it off).
 */
cannonCcd: boolean, 
/**
 * The gauntlet's or the meteor run's street.
 */
street: StreetLayout | null, 
/**
 * The meteor range's pads and meteors.
 */
range: RangeLayout | null, 
/**
 * Directed debris on (`?guide=0`: off) and its strength (`?strength=` 0, 1, 2 as
 * the page's menu reads it; default 1).
 */
guide: boolean, guideStrength: number, 
/**
 * The range's carts move (`?targets=moving`); steering knows their routes
 * (`?routes=0`: not).
 */
targetsMoving: boolean, routes: boolean, };

export type StreetLayout = { spawn: [number, number, number], blocks: Array<LayoutBlock>, cores: Array<LayoutBlock>, parked: Array<LayoutParked>, finish: number | null, meteor: LayoutMeteor | null, };

export type LayoutBlock = { p: [number, number, number], half: [number, number, number], mass: number, };

export type LayoutParked = { p: [number, number, number], yaw: number, };

export type LayoutMeteor = { count: number, radius: number, mass: number, };

export type RangeLayout = { pads: Array<RangePad>, radius: number, mass: number, };

export type RangePad = { p: [number, number, number], half: [number, number, number], lane: number, };

export type SoftSceneDef = { scene: string, 
/**
 * `sedan`, `buggy`, `hires` or `import` (with `import`).
 */
car: string, import: ImportSource | null, 
/**
 * The vehicle runs on the explicit solver (the page's sim).
 */
explicit: boolean, 
/**
 * The explicit sim's static shapes: rows [centre xyz, half xyz, rotation xyzw, kind]
 * (missing rotation: identity; kind 0/1 box, 2 vertical cylinder, 3 capsule, 4 sphere).
 */
explicitBoxes: Array<Array<number>>, 
/**
 * The buggy's mount scale (`?strength=`; 0: 1).
 */
strength: number, 
/**
 * `?deform=`: the procedural car rebuilt from `.truck` text with this threshold.
 */
deform: number, 
/**
 * Step rate, iterations, thresholds × (`?threshold=`), particle sweeps, rigid mode.
 */
hz: number, iters: number, threshold: number, sweeps: number, rigid: boolean, 
/**
 * Crash parameters by name (`speed`, `overlap`, `angle`, `offset`, `at`, `mass`,
 * `carSpeed`, `height`, `pitch`, `bar`, `parked`, `liftAt`, `liftTo`); absent ones
 * take each scene's standard test.
 */
params: { [key in string]: number }, 
/**
 * Extra approach distance before a crash (m).
 */
runup: number, 
/**
 * Traffic: how many cars (`?cars=`, 1–128, default 20).
 */
cars: number, 
/**
 * The import scene's target: `wall` or `car`.
 */
target: string, 
/**
 * Traffic drivers (`?ai=`): empty, the derby (two in three wander, one chases);
 * `cruise`, everyone wanders and brakes for cars ahead; `0`, none.
 */
ai: string, };

export type ImportSource = { truck: string | null, jbeam: Array<[string, string]> | null, pc: string | null, };

export type MotionSceneDef = { scene: string, 
/**
 * Speculative velocity (`?lookahead=`, on by default).
 */
lookAhead: boolean, 
/**
 * The gallery's shot speed (m/s, `?speed=`).
 */
speed: number, 
/**
 * Volleys and bursts on their own cadence (`?autofire=0`: only on F).
 */
autofire: boolean, };

export type HillsDef = { n: number, };

export type CitySceneDef = { 
/**
 * Houses (`?k=`, default 1,024).
 */
houses: number, 
/**
 * Houses fall asleep once settled (`?sleep=0`: never).
 */
sleep: boolean, 
/**
 * Piece slots (`?pool=`; none: a slot per chunk). Default 131,072 on the dense town,
 * 32,768 on the benchmark's map.
 */
pool: number | null, 
/**
 * The lot pitch (m, `?spacing=`, default 16); none: the benchmark's 2.2 km map.
 */
pitch: number | null, 
/**
 * Bodies below this height leave the world (`?kill=`, default −50; none: `off`).
 */
killBelow: number | null, 
/**
 * What happens once the city settles (`?scenario=still|drive|war`).
 */
scenario: string, };

export type ImpactSceneDef = { 
/**
 * `impact` or `explosion` (`?mode=`).
 */
mode: string, 
/**
 * The impactor: radius (m), speed (km/s), density (kg/m³).
 */
radius: number, speed: number, density: number, 
/**
 * The charge: 10^charge kg TNT, `depth` m down.
 */
charge: number, depth: number, 
/**
 * The region's margin past the predicted crater (× its radius).
 */
margin: number, 
/**
 * `game` or `physical` (`engine_terrain::TerrainProfile`).
 */
profile: string, 
/**
 * Physics frames per second.
 */
hz: number, 
/**
 * Cells across the transient radius (none: the profile's).
 */
cpr: number | null, 
/**
 * The bake: `settle` (when the crater rests) or `time` (at `bake_at` s).
 */
bake: string, bakeAt: number, 
/**
 * The rest speed the settle bake waits for (m/s; none: 1% of √(gR)).
 */
settle: number | null, };

export type ShowcaseSceneDef = { 
/**
 * `drive`, `drive1`, `pass`, `bts` or `play`.
 */
scene: string, 
/**
 * `game` or `physical` (`engine_terrain::TerrainProfile`).
 */
profile: string, 
/**
 * `?rule=` (play: `strict`, else the local settle rule).
 */
rule: string | null, 
/**
 * The vehicle preset.
 */
car: string, 
/**
 * Play's random sequence (`?seed=`; the page picks one when absent).
 */
seed: number | null, 
/**
 * The meteor's radius (m) and speed (km/s).
 */
radius: number | null, speed: number | null, 
/**
 * Cells across the transient radius.
 */
cpr: number | null, 
/**
 * The region's margin past the crater (× its radius).
 */
margin: number | null, 
/**
 * The rest speed areas bake at (m/s; none: 1% of √(gR)).
 */
settle: number | null, 
/**
 * Areas open at once (drive and play).
 */
regions: number | null, 
/**
 * Play's basin half side (m).
 */
half: number | null, 
/**
 * The track: clearance from a region box, straights' half length, turn radius.
 */
clear: number | null, straight: number | null, turn: number | null, 
/**
 * The drive scene's flat ground outside the loop (m).
 */
flatout: number | null, 
/**
 * The surface patches' samples across.
 */
patch: number | null, 
/**
 * The autopilot's speed (km/h).
 */
kmh: number | null, 
/**
 * The run-up before the recording (s).
 */
lead: number | null, 
/**
 * drive1: rim to track (m).
 */
gap: number | null, 
/**
 * The fixed schedules' times (s) and the second strike's turn on the first's rim.
 */
t1: number | null, t2: number | null, t3: number | null, pass1: number | null, pass2: number | null, pass3: number | null, turn2: number | null, 
/**
 * Play: the director keeps out of areas older than this (s).
 */
settleAfter: number | null, };

export type RangeSceneDef = { 
/**
 * `broadside` or `timber`.
 */
scene: string, level: number, raze: boolean, };

export type Aim = { eye: [number, number, number], dir: [number, number, number], };

export type Action = { "action": "reset" } | { "action": "fire", aim?: Aim, } | { "action": "cannon", aim: Aim, } | { "action": "ram" } | { "action": "crater", at?: [number, number], } | { "action": "rain" } | { "action": "guide", on?: boolean, strength?: number, } | { "action": "start" } | { "action": "launch" } | { "action": "bake" } | { "action": "takeWheel" } | { "action": "shell", origin: [number, number, number], velocity: [number, number, number], radius: number, } | { "action": "strike", at?: [number, number], } | { "action": "drive", forward?: number, steer?: number, handbrake?: number, } | { "action": "targets", moving?: boolean, routes?: boolean, };

export type Event = { 
/**
 * The step it happened before.
 */
step: number, what: string, 
/**
 * What it was about (a touchdown's lane and score, a judged near miss…).
 */
data: JsonValue | null, };

export type PlayStats = { steps: number, bodies: number, awake: number, contacts: number, overflow: number, };

export type PresetInfo = { 
/**
 * Unique, URL-safe (`?preset=<name>`).
 */
name: string, title: string, 
/**
 * [`super::Family::name`]: the `family` tag of its [`SceneDef`].
 */
family: string, 
/**
 * The menu section it is listed under.
 */
hub: string, 
/**
 * Its own page (relative to `engine-webgpu/`) and that page's query.
 */
page: string, query: string, 
/**
 * Its scene is placed through [`SceneDef`] (any host can run it).
 */
placed: boolean, };

export type RenderManifest = { 
/**
 * [`World::render_version`] when this was made.
 */
version: number, groups: Array<RenderGroup>, planes: Array<RenderPlane>, 
/**
 * Colliders not drawn, by reason (a reason a renderer may want to know).
 */
skipped: { [key in string]: number }, };

export type RenderGroup = { "kind": "box", tag: string, colliders: ArrayLike<number>, } | { "kind": "sphere", tag: string, colliders: ArrayLike<number>, } | { "kind": "capsule", tag: string, colliders: ArrayLike<number>, } | { "kind": "mesh", tag: string, positions: ArrayLike<number>, normals: ArrayLike<number>, indices: ArrayLike<number>, vertexColliders: ArrayLike<number>, };

export type RenderPlane = { collider: number, normal: [number, number, number], offset: number, };
