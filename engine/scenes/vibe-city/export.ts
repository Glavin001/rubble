// Runs the vibe-city /destructible-stress structure presets (src/, copied from vibe-city)
// and writes one scene pack per preset to engine/crates/engine-webgpu/generated/vibe-city/ (git-ignored).
//
//   bun install && bun run export.ts            # every preset
//   bun run export.ts hut tower                 # a subset
//
// Each preset is built with the arguments the vibe-city page passes (SCENARIO_BUILDERS in
// src/app/destructible-stress/page.tsx), and with the page's collider and size rules
// (destructible-core.ts buildColliderDescForNode, page.tsx nodeSize). Math.random is seeded
// per preset so the Voronoi presets (three-pinata) are reproducible.

import "./shims/dom";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type * as THREE from "three";
import type { ScenarioDesc, Vec3 } from "./src/core/types";
import { ColliderDesc, type ShapeRecord } from "./shims/rapier";
import { cleanHull } from "./hulls";
import { buildBeamBridgeScenario } from "./src/scenarios/beamBridgeScenario";
import { buildBrickWallScenario } from "./src/scenarios/brickWallScenario";
import { buildFracturedBridgeScenario } from "./src/scenarios/fracturedBridgeScenario";
import { buildFracturedGlbScenario } from "./src/scenarios/fracturedGlbScenario";
import { buildFracturedTowerScenario } from "./src/scenarios/fracturedTowerScenario";
import { buildFracturedWallHutScenario } from "./src/scenarios/fracturedWallHutScenario";
import { buildFracturedWallScenario } from "./src/scenarios/fracturedWallScenario";
import {
  buildKinematicDumbbellScenario,
  buildReinforcedTowerScenario,
  buildTiltingGantryScenario,
} from "./src/scenarios/gantryPresets";
import {
  buildBridgeScenario,
  buildCourtyardHouseScenario,
  buildHutScenario,
  buildTowerScenario,
  buildTownhouseScenario,
  buildVaultedLoftScenario,
  STRESS_PRESET_METADATA,
} from "./src/scenarios/structurePresets";
import { buildWallScenario } from "./src/scenarios/wallScenario";

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, "../../crates/engine-webgpu/generated/vibe-city");

const APRIL = { ref: "feat/stress-destruction-april-6", commit: "bb16c52" };
const GANTRY = { ref: "feat/tilting-gantry", commit: "2665b34" };

// Page defaults (page.tsx useState): wall sliders and the projectile.
const WALL = { span: 6.0, height: 3.0, thickness: 0.32, spanSeg: 12, heightSeg: 6, layers: 1 };
const PROJECTILE = { radius: 0.5, mass: 15_000, speed: 36 };
// Blast ext-solver limits the page applies: base value x materialScale (page.tsx ~line 900).
const MATERIAL_SCALE = 100_000_000;
// Hull clean-up (hulls.ts): hull triangles within HULL_TOLERANCE of a larger face merge
// into it; the export fails if a rebuilt hull extends past its piece by more than
// HULL_MAX_OUTSIDE. Both are recorded in every pack.
const HULL_TOLERANCE = 1e-4;
const HULL_MAX_OUTSIDE = 1e-3;
const BASE_LIMITS = {
  compressionElastic: 0.0009,
  compressionFatal: 0.0027,
  tensionElastic: 0.0009,
  tensionFatal: 0.0027,
  shearElastic: 0.0012,
  shearFatal: 0.0036,
};

type Preset = {
  id: string;
  source: { ref: string; commit: string };
  label?: string;
  description?: string;
  build: () => ScenarioDesc | Promise<ScenarioDesc>;
  /** Skipped by a full export (still exported when named on the command line). */
  pending?: string;
};

const bonds = { bondsX: true, bondsY: true, bondsZ: true };

const PRESETS: Preset[] = [
  {
    id: "wall",
    source: APRIL,
    build: () =>
      buildWallScenario({
        span: WALL.span,
        height: WALL.height,
        thickness: WALL.thickness,
        spanSegments: WALL.spanSeg,
        heightSegments: WALL.heightSeg,
        layers: WALL.layers,
        ...bonds,
      }),
  },
  {
    id: "brickWall",
    source: APRIL,
    build: () =>
      buildBrickWallScenario({
        span: WALL.span,
        height: WALL.height,
        thickness: WALL.thickness,
        spanBricks: WALL.spanSeg,
        courses: WALL.heightSeg,
        layers: WALL.layers,
        ...bonds,
        includeHalfBricks: true,
        clumpCount: 7,
      }),
  },
  { id: "hut", source: APRIL, build: () => buildHutScenario(bonds) },
  { id: "bridge", source: APRIL, build: () => buildBridgeScenario(bonds) },
  { id: "beamBridge", source: APRIL, build: () => buildBeamBridgeScenario(bonds) },
  { id: "tower", source: APRIL, build: () => buildTowerScenario(bonds) },
  { id: "townhouse", source: APRIL, build: () => buildTownhouseScenario(bonds) },
  { id: "courtyardHouse", source: APRIL, build: () => buildCourtyardHouseScenario(bonds) },
  { id: "vaultedLoft", source: APRIL, build: () => buildVaultedLoftScenario(bonds) },
  {
    id: "fracturedWall",
    source: APRIL,
    build: () =>
      buildFracturedWallScenario({
        span: WALL.span,
        height: WALL.height,
        thickness: WALL.thickness,
        fragmentCount: 200,
      }),
  },
  {
    id: "fracturedWallHut",
    source: APRIL,
    build: () => buildFracturedWallHutScenario({ fragmentCountPerWall: 100 }),
  },
  {
    id: "fracturedTower",
    source: APRIL,
    build: () =>
      buildFracturedTowerScenario({
        width: 45,
        depth: 45,
        floorCount: 15,
        floorHeight: 4,
        columnSize: 1.5,
        columnSpacing: 9,
        columnInset: 0.12,
        fragmentCountPerWall: 15,
        fragmentCountPerFloor: 60,
        fragmentCountPerColumn: 5,
        useAutoBonding: true,
      }),
  },
  {
    id: "fracturedBridge",
    source: APRIL,
    build: () => buildFracturedBridgeScenario({ fragmentCountPerDeck: 200, fragmentCountPerPost: 10 }),
  },
  {
    id: "fracturedGlb",
    source: APRIL,
    build: () => buildFracturedGlbScenario({ url: new URL("./models/lion.glb", import.meta.url).href }),
    pending: "hull clean-up leaves ~10 lion pieces the engine's cooker rejects, and takes ~3 min",
  },
  {
    id: "reinforcedTower",
    source: GANTRY,
    label: "Reinforced tower",
    description: "Tower with a column grid, eight floor plates, outriggers and weaker facade bands.",
    build: () => buildReinforcedTowerScenario(bonds),
  },
  {
    id: "tiltingGantry",
    source: GANTRY,
    label: "Tilting gantry",
    description: "Ring of block columns on a kinematic platform that tilts and spins.",
    build: () => buildTiltingGantryScenario(),
  },
  {
    id: "kinematicDumbbell",
    source: GANTRY,
    label: "Kinematic dumbbell",
    description: "Two bonded blocks on a spinning kinematic pad.",
    build: () => buildKinematicDumbbellScenario(),
  },
];

/** mulberry32: deterministic Math.random for the Voronoi seeds. */
function seedRandom(seed: number) {
  let a = seed >>> 0;
  Math.random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const round = (v: number) => Math.round(v * 1e6) / 1e6;
const vec = (v: Vec3) => ({ x: round(v.x), y: round(v.y), z: round(v.z) });

/** page.tsx nodeSize: fragmentSizes, else spacing, else 0.5 x 0.5 x 0.32 (full extents). */
function nodeSize(s: ScenarioDesc, i: number): Vec3 {
  const sizes = (s.parameters as { fragmentSizes?: Vec3[] } | undefined)?.fragmentSizes;
  return sizes?.[i] ?? s.spacing ?? { x: 0.5, y: 0.5, z: 0.32 };
}

/** destructible-core.ts buildColliderDescForNode, with the Rapier shim recording the shape. */
function nodeCollider(s: ScenarioDesc, i: number): ShapeRecord {
  const builder = s.colliderDescForNode?.[i];
  const desc = typeof builder === "function" ? (builder() as unknown as ColliderDesc | null) : null;
  if (desc) return desc.shape;
  const size = nodeSize(s, i);
  const k = s.nodes[i].mass === 0 ? 0.999 : 1.0;
  const half = (v: number) => Math.max(0.05, v * 0.5) * k;
  return ColliderDesc.cuboid(half(size.x), half(size.y), half(size.z)).shape;
}

function meshOf(g: THREE.BufferGeometry) {
  const pos = g.getAttribute("position");
  const nor = g.getAttribute("normal");
  const idx = g.getIndex();
  return {
    positions: Array.from(pos.array as ArrayLike<number>, round),
    normals: nor ? Array.from(nor.array as ArrayLike<number>, round) : [],
    indices: idx ? Array.from(idx.array as ArrayLike<number>) : [],
  };
}

/** Plain JSON parameters (geometry and functions dropped). */
function jsonParameters(p: Record<string, unknown> | undefined) {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(p ?? {})) {
    if (k === "fragmentGeometries" || k === "fragmentSizes" || typeof v === "function") continue;
    try {
      out[k] = JSON.parse(JSON.stringify(v));
    } catch {}
  }
  return out;
}

async function exportPreset(p: Preset, seed: number) {
  seedRandom(seed);
  const hullStats = { cleaned: 0, split: 0, outside: 0 };
  const s = await p.build();
  const meta = STRESS_PRESET_METADATA.find((m) => m.id === p.id);
  const geoms = (s.parameters as { fragmentGeometries?: THREE.BufferGeometry[] } | undefined)
    ?.fragmentGeometries;
  const pack = {
    version: 1,
    key: p.id,
    title: p.label ?? meta?.label ?? p.id,
    description: p.description ?? meta?.description ?? "",
    source: {
      repo: "github.com/Glavin001/vibe-city",
      ref: p.source.ref,
      commit: p.source.commit,
      page: "/destructible-stress",
      seed,
    },
    defaults: {
      projectile: { ...PROJECTILE, ttlMs: 8000 },
      solver: {
        gravity: -9.81,
        materialScale: MATERIAL_SCALE,
        limits: Object.fromEntries(
          Object.entries(BASE_LIMITS).map(([k, v]) => [k, v * MATERIAL_SCALE]),
        ),
      },
    },
    parameters: jsonParameters(s.parameters),
    scenario: {
      nodes: s.nodes.map((n) => ({ centroid: vec(n.centroid), mass: round(n.mass), volume: round(n.volume) })),
      bonds: s.bonds.map((b) => ({
        node0: b.node0,
        node1: b.node1,
        centroid: vec(b.centroid),
        normal: vec(b.normal),
        area: round(b.area),
      })),
      nodeSizes: s.nodes.map((_, i) => vec(nodeSize(s, i))),
      nodeColliders: s.nodes.map((_, i) => {
        const c = nodeCollider(s, i);
        if (c.kind === "cuboid") return { kind: c.kind, halfExtents: c.halfExtents.map(round) };
        const pts: [number, number, number][] = [];
        for (let k = 0; k + 2 < c.points.length; k += 3) pts.push([c.points[k], c.points[k + 1], c.points[k + 2]]);
        const clean = cleanHull(pts, HULL_TOLERANCE, HULL_MAX_OUTSIDE);
        if (clean.outside > HULL_MAX_OUTSIDE)
          throw new Error(`node ${i}: cleaned hull extends ${clean.outside.toExponential(2)} m past the piece (max ${HULL_MAX_OUTSIDE})`);
        hullStats.cleaned += 1;
        hullStats.outside = Math.max(hullStats.outside, clean.outside);
        if (clean.parts.length > 1) hullStats.split += 1;
        const parts = clean.parts.map((part) => ({ points: part.points.flat(), faces: part.faces }));
        return parts.length === 1 ? { kind: "convex_hull", ...parts[0] } : { kind: "compound", parts };
      }),
    },
    ...(geoms?.length === s.nodes.length ? { nodeMeshes: geoms.map(meshOf) } : {}),
  };
  const hullCleanup = hullStats.cleaned
    ? { tolerance: HULL_TOLERANCE, maxOutside: HULL_MAX_OUTSIDE, hulls: hullStats.cleaned, split: hullStats.split, outside: hullStats.outside }
    : undefined;
  const file = join(OUT, `${p.id}.json`);
  if (hullCleanup) Object.assign(pack, { hullCleanup });
  writeFileSync(file, `${JSON.stringify(pack)}\n`);
  const supports = s.nodes.filter((n) => n.mass === 0).length;
  const mass = s.nodes.reduce((m, n) => m + n.mass, 0);
  return { id: p.id, nodes: s.nodes.length, supports, bonds: s.bonds.length, mass: Math.round(mass), ...(hullCleanup ? { hullCleanup } : {}) };
}

mkdirSync(OUT, { recursive: true });
const wanted = new Set(process.argv.slice(2));
const index: unknown[] = [];
for (const [i, p] of PRESETS.entries()) {
  if (wanted.size && !wanted.has(p.id)) continue;
  if (!wanted.size && p.pending) {
    console.log(`${p.id.padEnd(18)} pending: ${p.pending}`);
    index.push({ id: p.id, pending: p.pending });
    continue;
  }
  try {
    const r = await exportPreset(p, 1337 + i);
    index.push(r);
    console.log(`${r.id.padEnd(18)} nodes ${String(r.nodes).padStart(6)}  supports ${String(r.supports).padStart(5)}  bonds ${String(r.bonds).padStart(6)}  mass ${r.mass} kg${r.hullCleanup ? `  hulls cleaned ${r.hullCleanup.hulls}, split ${r.hullCleanup.split}, max outside ${(r.hullCleanup.outside * 1000).toFixed(3)} mm` : ""}`);
  } catch (e) {
    console.error(`${p.id}: FAILED ${(e as Error).message}`);
    index.push({ id: p.id, failed: (e as Error).message });
  }
}
if (!wanted.size) writeFileSync(join(OUT, "index.json"), `${JSON.stringify({ presets: index }, null, 2)}\n`);
