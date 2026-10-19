// Copied from vibe-city (github.com/Glavin001/vibe-city) branch feat/tilting-gantry,
// commit 2665b34, src/lib/stress/scenarios/structurePresets.ts: the three presets that
// exist only on that branch (reinforcedTower, tiltingGantry, kinematicDumbbell).
// Helpers are that file's own; buildRectilinearScenario comes from the april-6 copy.
import type { ScenarioDesc, Vec3 } from "@/lib/stress/core/types";
import { buildRectilinearScenario } from "./structurePresets";

const EPSILON = 1e-8;

function makeVec(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

function subVec(a: Vec3, b: Vec3): Vec3 {
  return makeVec(a.x - b.x, a.y - b.y, a.z - b.z);
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z);
  if (len <= EPSILON) return makeVec(0, 0, 0);
  return makeVec(v.x / len, v.y / len, v.z / len);
}


type PresetOptions = {
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
};


const TOWER_HEIGHT_CONFIG = {
  /** Number of vertical layers; tweak this to make the tower taller or shorter. */
  segmentCount: 58,
  /**
   * Maintain the original per-layer scale (9.2 m over 22 layers) so the new
   * height simply stretches the existing detailing proportionally.
   */
  metersPerSegment: 9.2 / 22,
} as const;


export function buildReinforcedTowerScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const segments = { x: 16, y: TOWER_HEIGHT_CONFIG.segmentCount, z: 16 };
  const towerHeightMeters = segments.y * TOWER_HEIGHT_CONFIG.metersPerSegment;
  const columnPositionsX = [
    1,
    segments.x - 2,
    Math.floor(segments.x * 0.25),
    Math.floor(segments.x * 0.75),
  ];
  const columnPositionsZ = [
    1,
    segments.z - 2,
    Math.floor(segments.z * 0.25),
    Math.floor(segments.z * 0.75),
  ];
  const columnSetX = new Set(columnPositionsX);
  const columnSetZ = new Set(columnPositionsZ);
  const floorHeights = [
    0,
    Math.floor(segments.y * 0.15),
    Math.floor(segments.y * 0.3),
    Math.floor(segments.y * 0.45),
    Math.floor(segments.y * 0.6),
    Math.floor(segments.y * 0.75),
    Math.floor(segments.y * 0.9),
    segments.y - 2,
  ];
  const floorSet = new Set(floorHeights);
  const coreBandMin = Math.floor(segments.x * 0.35);
  const coreBandMax = segments.x - 1 - coreBandMin;
  const windowBandStart = Math.floor(segments.y * 0.35);

  const scenario = buildRectilinearScenario({
    size: makeVec(6.8, towerHeightMeters, 6.8),
    segments,
    deckMass: 310_000,
    areaScale: 0.055,
    addDiagonals: true,
    diagScale: 0.65,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onShell = ix === 0 || ix === seg.x - 1 || iz === 0 || iz === seg.z - 1;
      const isRoof = iy === seg.y - 1;
      const inColumn = columnSetX.has(ix) && columnSetZ.has(iz);
      const inCore =
        ix >= coreBandMin && ix <= coreBandMax && iz >= coreBandMin && iz <= coreBandMax;
      const isFloor = floorSet.has(iy);

      if (iy === 0) {
        return inColumn || inCore;
      }
      if (isRoof) {
        return true;
      }
      if (inColumn || inCore) {
        return true;
      }
      if (isFloor) {
        return true;
      }
      if (onShell) {
        if (iy >= windowBandStart) {
          // Perforated curtain wall at upper levels
          return (ix + iz + iy) % 3 === 0;
        }
        return true;
      }
      return false;
    },
    supportPredicate: ({ iy }) => iy === 0,
  });

  type ReinforcedNodeRole = "core" | "column" | "slab" | "facade";
  const coords = scenario.gridCoordinates ?? [];
  const nodeRoles: ReinforcedNodeRole[] = new Array(scenario.nodes.length).fill("facade");
  const slabLevels = new Set<number>([
    ...floorSet,
    Math.max(1, segments.y - 4),
    Math.max(1, segments.y - 6),
  ]);

  coords.forEach((coord, index) => {
    if (!coord) return;
    const { ix, iy, iz } = coord;
    const inCore =
      ix >= coreBandMin && ix <= coreBandMax && iz >= coreBandMin && iz <= coreBandMax;
    const inColumn = columnSetX.has(ix) && columnSetZ.has(iz) && iy > 0;
    if (inCore) {
      nodeRoles[index] = "core";
    } else if (inColumn) {
      nodeRoles[index] = "column";
    } else if (slabLevels.has(iy)) {
      nodeRoles[index] = "slab";
    }
  });

  const roleStrength: Record<
    ReinforcedNodeRole,
    Partial<Record<ReinforcedNodeRole, number>>
  > = {
    core: { core: 5.0, column: 4.2, slab: 3.5, facade: 1.4 },
    column: { column: 3.2, slab: 2.6, facade: 1.25 },
    slab: { slab: 2.1, facade: 1.0 },
    facade: { facade: 0.65 },
  };
  const getRoleMultiplier = (a: ReinforcedNodeRole, b: ReinforcedNodeRole): number => {
    if (roleStrength[a]?.[b] != null) return roleStrength[a][b];
    if (roleStrength[b]?.[a] != null) return roleStrength[b][a];
    return 1;
  };
  const outriggerLevels = new Set<number>([
    Math.floor(segments.y * 0.33),
    Math.floor(segments.y * 0.66),
  ]);
  const isCoreOrColumn = (role: ReinforcedNodeRole) => role === "core" || role === "column";

  scenario.bonds.forEach((bond) => {
    const roleA = nodeRoles[bond.node0] ?? "facade";
    const roleB = nodeRoles[bond.node1] ?? "facade";
    const coordA = coords[bond.node0];
    const coordB = coords[bond.node1];
    let multiplier = getRoleMultiplier(roleA, roleB);
    if (coordA && coordB) {
      const sameLevel = coordA.iy === coordB.iy;
      const verticalStep = Math.abs(coordA.iy - coordB.iy) === 1;
      if (
        sameLevel &&
        outriggerLevels.has(coordA.iy) &&
        isCoreOrColumn(roleA) &&
        isCoreOrColumn(roleB)
      ) {
        multiplier = Math.max(multiplier, 6);
      }
      if (verticalStep && isCoreOrColumn(roleA) && isCoreOrColumn(roleB)) {
        multiplier = Math.max(multiplier, 4.5);
      }
      if (coordA.iy === 0 || coordB.iy === 0) {
        multiplier = Math.min(multiplier, 3.2);
      }
    }
    const clamped = Math.min(6.5, Math.max(0.45, multiplier));
    bond.area = Math.max(EPSILON, bond.area * clamped);
  });

  return scenario;
}

export const TILTING_GANTRY_DEFAULTS = {
  columnCount: 12,
  layers: 4,
  blockSize: makeVec(0.4, 0.55, 0.4),
  radius: 2.8,
  platformRadius: 3.2,
  platformThickness: 0.35,
  baseLift: 1.5,
  bondAreaScale: 0.14,
  density: 4_000,
  tiltAmplitudeDeg: 22,
  tiltFrequency: 0.38,
  spinSpeed: 0.4,
} as const;

type TiltingGantryOptions = Partial<typeof TILTING_GANTRY_DEFAULTS>;

export function buildTiltingGantryScenario(options: TiltingGantryOptions = {}): ScenarioDesc {
  const config = { ...TILTING_GANTRY_DEFAULTS, ...options };
  const {
    columnCount,
    layers,
    blockSize,
    radius,
    platformRadius,
    platformThickness,
    baseLift,
    bondAreaScale,
    density,
    tiltAmplitudeDeg,
    tiltFrequency,
    spinSpeed,
  } = config;
  const nodes: ScenarioDesc["nodes"] = [];
  const bonds: ScenarioDesc["bonds"] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];
  const blockVolume = blockSize.x * blockSize.y * blockSize.z;
  const blockMass = blockVolume * density;
  const verticalArea = blockSize.x * blockSize.z * bondAreaScale;
  const supportTopY = baseLift;
  const supportY = supportTopY - blockSize.y * 0.5;
  const columnBaseOffset = supportTopY;
  const supportNodes: number[] = [];
  const supportRingArea = blockSize.x * blockSize.x * bondAreaScale;

  const addBond = (nodeA: number, nodeB: number, area: number) => {
    if (nodeA < 0 || nodeB < 0) return;
    const na = nodes[nodeA];
    const nb = nodes[nodeB];
    const centroid = makeVec(
      (na.centroid.x + nb.centroid.x) * 0.5,
      (na.centroid.y + nb.centroid.y) * 0.5,
      (na.centroid.z + nb.centroid.z) * 0.5,
    );
    const normal = normalize(subVec(nb.centroid, na.centroid));
    bonds.push({
      node0: nodeA,
      node1: nodeB,
      centroid,
      normal,
      area: Math.max(area, EPSILON),
    });
  };

  for (let column = 0; column < columnCount; column += 1) {
    const angle = (column / columnCount) * Math.PI * 2;
    const cx = Math.cos(angle) * radius;
    const cz = Math.sin(angle) * radius;
    const supportIdx = nodes.length;
    nodes.push({
      centroid: makeVec(cx, supportY, cz),
      mass: 0,
      volume: 0,
    });
    supportNodes.push(supportIdx);
    gridCoordinates[supportIdx] = { ix: column, iy: -1, iz: 0 };
    let previous = supportIdx;
    for (let layer = 0; layer < layers; layer += 1) {
      const idx = nodes.length;
      nodes.push({
        centroid: makeVec(
          cx,
          columnBaseOffset + layer * blockSize.y + blockSize.y * 0.5,
          cz,
        ),
        mass: blockMass,
        volume: blockVolume,
      });
      gridCoordinates[idx] = { ix: column, iy: layer, iz: 0 };
      if (previous >= 0) addBond(previous, idx, verticalArea);
      previous = idx;
    }
  }

  if (supportNodes.length > 1) {
    for (let i = 0; i < supportNodes.length; i += 1) {
      const a = supportNodes[i];
      const b = supportNodes[(i + 1) % supportNodes.length];
      addBond(a, b, supportRingArea);
    }
  }

  const columnSpan = layers * blockSize.y;

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: makeVec(blockSize.x, blockSize.y, blockSize.z),
    parameters: {
      preset: "tiltingGantry",
      columnCount,
      layers,
      radius,
      platformRadius,
      platformThickness,
      columnHeight: columnSpan,
      platformTopHeight: supportTopY,
      baseLift,
      blockSize,
      tiltAmplitudeDeg,
      tiltFrequency,
      spinSpeed,
    },
  };
}

export const KINEMATIC_DUMBBELL_DEFAULTS = {
  radius: 1.5,
  blockSize: makeVec(0.4, 0.8, 0.4),
  platformRadius: 0.45,
  platformThickness: 0.1,
  spinSpeed: 0.65,
} as const;

export function buildKinematicDumbbellScenario(): ScenarioDesc {
  const config = KINEMATIC_DUMBBELL_DEFAULTS;
  const nodes: ScenarioDesc["nodes"] = [];
  const bonds: ScenarioDesc["bonds"] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];
  const { radius, blockSize } = config;
  const thickness = blockSize.x;
  const blockHeight = blockSize.y;

  const addNode = (centroid: Vec3, mass: number, volume: number, coord?: { ix: number; iy: number; iz: number }) => {
    const index = nodes.length;
    nodes.push({ centroid, mass, volume });
    if (coord) gridCoordinates[index] = coord;
    return index;
  };

  const padIdx = addNode({ x: 0, y: 0, z: 0 }, 0, 0, { ix: 0, iy: 0, iz: 0 });
  const leftIdx = addNode({ x: -radius, y: blockHeight * 0.5, z: 0 }, 20, thickness * blockHeight * thickness, { ix: -1, iy: 1, iz: 0 });
  const rightIdx = addNode({ x: radius, y: blockHeight * 0.5, z: 0 }, 20, thickness * blockHeight * thickness, { ix: 1, iy: 1, iz: 0 });

  const addBond = (node0: number, node1: number) => {
    const a = nodes[node0];
    const b = nodes[node1];
    const centroid = makeVec((a.centroid.x + b.centroid.x) * 0.5, (a.centroid.y + b.centroid.y) * 0.5, (a.centroid.z + b.centroid.z) * 0.5);
    const normal = normalize(subVec(b.centroid, a.centroid));
    bonds.push({ node0, node1, centroid, normal, area: blockHeight * thickness });
  };

  addBond(padIdx, leftIdx);
  addBond(leftIdx, rightIdx);

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: makeVec(thickness, blockHeight, thickness),
    parameters: {
      preset: "kinematicDumbbell",
      radius,
      blockSize,
      platformRadius: config.platformRadius,
      platformThickness: config.platformThickness,
      spinSpeed: config.spinSpeed,
    },
  };
}
