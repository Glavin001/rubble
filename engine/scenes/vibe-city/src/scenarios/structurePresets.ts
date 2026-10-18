import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import type {
  ColliderDescBuilder,
  ScenarioDesc,
  Vec3,
} from "@/lib/stress/core/types";

const EPSILON = 1e-8;

type Vec3i = { x: number; y: number; z: number };

type IncludeArgs = {
  ix: number;
  iy: number;
  iz: number;
  segments: Vec3i;
  position: Vec3;
};

type RectilinearOptions = {
  size: Vec3;
  segments: Vec3i;
  center?: Vec3;
  includeNode?: (args: IncludeArgs) => boolean;
  supportPredicate?: (args: IncludeArgs) => boolean;
  deckMass?: number;
  areaScale?: number;
  addDiagonals?: boolean;
  diagScale?: number;
  normalizeAreas?: boolean;
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
};

function clampSegments(value: number): number {
  return Math.max(1, Math.floor(value));
}

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

export function buildRectilinearScenario({
  size,
  segments,
  center,
  includeNode,
  supportPredicate,
  deckMass = 14_000,
  areaScale = 0.05,
  addDiagonals = false,
  diagScale = 0.75,
  normalizeAreas = true,
  bondsX = true,
  bondsY = true,
  bondsZ = true,
}: RectilinearOptions): ScenarioDesc {
  const segX = clampSegments(segments.x);
  const segY = clampSegments(segments.y);
  const segZ = clampSegments(segments.z);

  const cellX = size.x / segX;
  const cellY = size.y / segY;
  const cellZ = size.z / segZ;

  const origin = makeVec(
    (center?.x ?? 0) - size.x * 0.5 + cellX * 0.5,
    (center?.y ?? size.y * 0.5) - size.y * 0.5 + cellY * 0.5,
    (center?.z ?? 0) - size.z * 0.5 + cellZ * 0.5,
  );

  const grid: number[][][] = Array.from({ length: segX }, () =>
    Array.from({ length: segY }, () => Array.from({ length: segZ }, () => -1)),
  );

  const nodes: ScenarioDesc["nodes"] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];
  const colliderDescForNode: (ColliderDescBuilder | null)[] = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];

  const include = includeNode ?? (() => true);
  const support = supportPredicate ?? (({ iy }) => iy === 0);

  const cellVolume = cellX * cellY * cellZ;

  let totalVolume = 0;

  for (let ix = 0; ix < segX; ix += 1) {
    for (let iy = 0; iy < segY; iy += 1) {
      for (let iz = 0; iz < segZ; iz += 1) {
        const position = makeVec(origin.x + ix * cellX, origin.y + iy * cellY, origin.z + iz * cellZ);
        if (!include({ ix, iy, iz, segments: { x: segX, y: segY, z: segZ }, position })) continue;
        const isSupport = support({ ix, iy, iz, segments: { x: segX, y: segY, z: segZ }, position });
        const nodeIndex = nodes.length;
        const volume = isSupport ? 0 : cellVolume;
        if (!isSupport) totalVolume += volume;
        nodes.push({ centroid: position, mass: volume, volume });
        grid[ix][iy][iz] = nodeIndex;
        gridCoordinates[nodeIndex] = { ix, iy, iz };

        const hx = cellX * 0.5;
        const hy = cellY * 0.5;
        const hz = cellZ * 0.5;
        colliderDescForNode.push(() =>
          RAPIER.ColliderDesc.cuboid(
            hx * (isSupport ? 0.999 : 1),
            hy * (isSupport ? 0.999 : 1),
            hz * (isSupport ? 0.999 : 1),
          ),
        );
        // Create geometry for auto bonding support
        fragmentGeometries.push(new THREE.BoxGeometry(cellX, cellY, cellZ));
      }
    }
  }

  const massScale = totalVolume > 0 ? deckMass / totalVolume : 0;
  if (massScale > 0) {
    nodes.forEach((node) => {
      if (node.volume > 0) {
        node.mass = node.volume * massScale;
      } else {
        node.mass = 0;
      }
    });
  } else {
    nodes.forEach((node) => {
      node.mass = 0;
    });
  }

  const bonds: ScenarioDesc["bonds"] = [];

  const areaX = cellY * cellZ * areaScale;
  const areaY = cellX * cellZ * areaScale;
  const areaZ = cellX * cellY * areaScale;

  const addBond = (a: number, b: number, area: number) => {
    if (a < 0 || b < 0) return;
    const na = nodes[a];
    const nb = nodes[b];
    const centroid = makeVec(
      (na.centroid.x + nb.centroid.x) * 0.5,
      (na.centroid.y + nb.centroid.y) * 0.5,
      (na.centroid.z + nb.centroid.z) * 0.5,
    );
    const normal = normalize(subVec(nb.centroid, na.centroid));
    bonds.push({ node0: a, node1: b, centroid, normal, area: Math.max(area, EPSILON) });
  };

  for (let ix = 0; ix < segX; ix += 1) {
    for (let iy = 0; iy < segY; iy += 1) {
      for (let iz = 0; iz < segZ; iz += 1) {
        const current = grid[ix][iy][iz];
        if (current < 0) continue;
        if (bondsX && ix + 1 < segX) addBond(current, grid[ix + 1][iy][iz], areaX);
        if (bondsY && iy + 1 < segY) addBond(current, grid[ix][iy + 1][iz], areaY);
        if (bondsZ && iz + 1 < segZ) addBond(current, grid[ix][iy][iz + 1], areaZ);
        if (addDiagonals) {
          if (bondsX && bondsY && ix + 1 < segX && iy + 1 < segY) addBond(current, grid[ix + 1][iy + 1][iz], 0.5 * (areaX + areaY) * diagScale);
          if (bondsX && bondsZ && ix + 1 < segX && iz + 1 < segZ) addBond(current, grid[ix + 1][iy][iz + 1], 0.5 * (areaX + areaZ) * diagScale);
          if (bondsY && bondsZ && iy + 1 < segY && iz + 1 < segZ) addBond(current, grid[ix][iy + 1][iz + 1], 0.5 * (areaY + areaZ) * diagScale);
        }
      }
    }
  }

  if (normalizeAreas && bonds.length) {
    const target = { x: size.y * size.z, y: size.x * size.z, z: size.x * size.y };
    const sum = { x: 0, y: 0, z: 0 };
    const pick = (n: Vec3): "x" | "y" | "z" => {
      const ax = Math.abs(n.x);
      const ay = Math.abs(n.y);
      const az = Math.abs(n.z);
      if (ax >= ay && ax >= az) return "x";
      if (ay >= az) return "y";
      return "z";
    };
    bonds.forEach((bond) => {
      sum[pick(bond.normal)] += bond.area;
    });
    const scale = {
      x: sum.x > 0 ? target.x / sum.x : 1,
      y: sum.y > 0 ? target.y / sum.y : 1,
      z: sum.z > 0 ? target.z / sum.z : 1,
    } as const;
    bonds.forEach((bond) => {
      bond.area *= scale[pick(bond.normal)];
    });
  }

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: makeVec(cellX, cellY, cellZ),
    parameters: { size, segments, deckMass, areaScale, addDiagonals, fragmentGeometries },
    colliderDescForNode,
  } satisfies ScenarioDesc;
}

type PresetOptions = {
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
};

export function buildHutScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const segments = { x: 18, y: 9, z: 14 };
  const doorStart = Math.floor(segments.x * 0.38);
  const doorEnd = Math.ceil(segments.x * 0.62);
  const doorHeight = Math.floor(segments.y * 0.55);
  const windowRow = Math.floor(segments.y * 0.65);
  const windowStart = Math.floor(segments.z * 0.35);
  const windowEnd = Math.ceil(segments.z * 0.65);

  return buildRectilinearScenario({
    size: makeVec(6.5, 3.4, 5.2),
    segments,
    deckMass: 19_000,
    areaScale: 0.052,
    addDiagonals: true,
    diagScale: 0.6,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onX = ix === 0 || ix === seg.x - 1;
      const onZ = iz === 0 || iz === seg.z - 1;
      const onTop = iy === seg.y - 1;
      if (!onX && !onZ) return false;
      if (onTop) return false; // open roof
      // doorway carve-out on front wall (iz === 0)
      if (iz === 0 && ix >= doorStart && ix <= doorEnd && iy <= doorHeight) {
        return ix === doorStart || ix === doorEnd || iy === doorHeight;
      }
      // side window on the right wall
      if (ix === seg.x - 1 && iy === windowRow && iz >= windowStart && iz <= windowEnd) {
        return iy === windowRow && (iz === windowStart || iz === windowEnd);
      }
      return true;
    },
  });
}

export function buildBridgeScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const segments = { x: 40, y: 12, z: 12 };
  const deckLayers = 2;
  const towerHeight = segments.y - 1;
  const towerWidth = Math.floor(segments.x * 0.08);
  const towerOffset = Math.floor(segments.x * 0.18);
  const cablePeak = segments.y - 1;
  const cableSpanStart = towerOffset + towerWidth;
  const cableSpanEnd = segments.x - cableSpanStart - 1;
  const cableSpanLength = Math.max(1, cableSpanEnd - cableSpanStart);

  return buildRectilinearScenario({
    size: makeVec(18, 4.6, 4.8),
    segments,
    deckMass: 24_000,
    areaScale: 0.06,
    addDiagonals: true,
    diagScale: 0.55,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const isDeck = iy <= deckLayers;
      const isTower = (ix <= towerOffset || ix >= seg.x - towerOffset - 1) && iy <= towerHeight && (iz <= 2 || iz >= seg.z - 3);
      let isCable = false;
      if (ix >= cableSpanStart && ix <= cableSpanEnd && (iz === 0 || iz === seg.z - 1)) {
        const normalized = (ix - cableSpanStart) / cableSpanLength;
        const desiredHeight = deckLayers + Math.round(Math.sin(normalized * Math.PI) * (cablePeak - deckLayers));
        if (iy >= desiredHeight && iy <= cablePeak) {
          isCable = (iy - desiredHeight) % 2 === 0;
        }
      }
      return isDeck || isTower || isCable;
    },
    supportPredicate: ({ ix, iy, segments: seg }) => {
      if (iy > 0) return false;
      return ix <= towerOffset || ix >= seg.x - towerOffset - 1;
    },
  });
}

const TOWER_HEIGHT_CONFIG = {
  /** Number of vertical layers; tweak this to make the tower taller or shorter. */
  segmentCount: 58,
  /**
   * Maintain the original per-layer scale (9.2 m over 22 layers) so the new
   * height simply stretches the existing detailing proportionally.
   */
  metersPerSegment: 9.2 / 22,
} as const;

export function buildTowerScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const towerSegments = { x: 16, y: TOWER_HEIGHT_CONFIG.segmentCount, z: 16 };
  const towerHeightMeters = towerSegments.y * TOWER_HEIGHT_CONFIG.metersPerSegment;
  const floorHeights = [
    0,
    Math.floor(towerSegments.y * 0.33),
    Math.floor(towerSegments.y * 0.66),
    towerSegments.y - 2,
  ];
  const columnPositions = [Math.floor(towerSegments.x * 0.25), Math.floor(towerSegments.x * 0.75)];
  
  return buildRectilinearScenario({
    size: makeVec(6.8, towerHeightMeters, 6.8),
    segments: towerSegments,
    deckMass: 280_000,
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
      const isFloor = floorHeights.includes(iy);
      const inColumn = columnPositions.includes(ix) && columnPositions.includes(iz);
      const hasWindowBand = iy === Math.floor(seg.y * 0.5) && (ix + iz) % 2 === 0;
      if (isRoof) return true;
      if (onShell) {
        if (iy > Math.floor(seg.y * 0.4) && (ix === Math.floor(seg.x * 0.5) || iz === Math.floor(seg.z * 0.5))) {
          return (iy - Math.floor(seg.y * 0.4)) % 2 === 0; // vertical slit windows
        }
        return true;
      }
      if (isFloor) return true;
      if (inColumn) return true;
      if (hasWindowBand) return true;
      return false;
    },
  });
}

export function buildTownhouseScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const segments = { x: 26, y: 18, z: 16 };
  const midFloor = Math.floor(segments.y * 0.45);
  const roofStart = Math.floor(segments.y * 0.66);
  const doorCenter = Math.floor(segments.z * 0.5);
  const doorHalf = Math.floor(segments.z * 0.12);
  const doorHeight = Math.floor(segments.y * 0.35);
  const windowRow = Math.floor(segments.y * 0.58);
  const sideWindowRow = Math.floor(segments.y * 0.42);
  const windowWidth = Math.floor(segments.z * 0.16);
  const windowInset = Math.floor(segments.z * 0.15);
  const patioSpanStart = Math.floor(segments.z * 0.28);
  const patioSpanEnd = segments.z - patioSpanStart;
  const patioLintel = Math.floor(segments.y * 0.32);

  return buildRectilinearScenario({
    size: makeVec(9.0, 5.8, 5.5),
    segments,
    deckMass: 620_000,
    areaScale: 0.052,
    addDiagonals: true,
    diagScale: 0.6,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onFront = ix === 0;
      const onBack = ix === seg.x - 1;
      const onSide = iz === 0 || iz === seg.z - 1;
      const isFloor = iy === 0 || iy === midFloor;
      const interiorDivider =
        iz === Math.floor(seg.z * 0.5) &&
        ix > Math.floor(seg.x * 0.2) &&
        ix < Math.floor(seg.x * 0.8) &&
        iy <= midFloor;

      if (iy === 0) return true;
      if (isFloor) return true;
      if (interiorDivider) return true;

      if (iy >= roofStart) {
        const level = iy - roofStart;
        const inset = Math.min(level * 2 + 1, Math.floor(seg.x * 0.3));
        const minX = inset;
        const maxX = seg.x - 1 - inset;
        if (ix < minX || ix > maxX) return false;
        const roofEdge = ix === minX || ix === maxX;
        const ridge = iy === seg.y - 1;
        if (roofEdge && (iz === 0 || iz === seg.z - 1)) return true;
        if (ridge) {
          const ridgeBand = Math.floor(seg.z * 0.2);
          return iz >= ridgeBand && iz <= seg.z - 1 - ridgeBand;
        }
        return iz === 0 || iz === seg.z - 1;
      }

      if (onFront) {
        const inDoorway = iz >= doorCenter - doorHalf && iz <= doorCenter + doorHalf && iy <= doorHeight;
        if (inDoorway) {
          return iy === doorHeight || iz === doorCenter - doorHalf || iz === doorCenter + doorHalf;
        }
        const leftWindowStart = windowInset;
        const leftWindowEnd = windowInset + windowWidth;
        const rightWindowStart = seg.z - windowInset - windowWidth;
        const rightWindowEnd = seg.z - windowInset;
        if (iy === windowRow && iz >= leftWindowStart && iz <= leftWindowEnd) {
          return iz === leftWindowStart || iz === leftWindowEnd;
        }
        if (iy === windowRow + 1 && iz >= leftWindowStart && iz <= leftWindowEnd) {
          return true;
        }
        if (iy === windowRow && iz >= rightWindowStart && iz <= rightWindowEnd) {
          return iz === rightWindowStart || iz === rightWindowEnd;
        }
        if (iy === windowRow + 1 && iz >= rightWindowStart && iz <= rightWindowEnd) {
          return true;
        }
        return true;
      }

      if (onBack) {
        const backWindowStart = Math.floor(seg.z * 0.2);
        const backWindowEnd = seg.z - backWindowStart;
        const row = windowRow - 1;
        if (iy === row && iz >= backWindowStart && iz <= backWindowEnd) {
          return iz === backWindowStart || iz === backWindowEnd;
        }
        if (iy === row + 1 && iz >= backWindowStart && iz <= backWindowEnd) {
          return true;
        }
        const patioHeight = patioLintel;
        if (iy <= patioHeight && iz >= patioSpanStart && iz <= patioSpanEnd) {
          return iy === patioHeight || iz === patioSpanStart || iz === patioSpanEnd;
        }
        return true;
      }

      if (onSide) {
        const sideWindowStart = Math.floor(seg.x * 0.3);
        const sideWindowEnd = seg.x - 1 - sideWindowStart;
        if (iy === sideWindowRow && ix >= sideWindowStart && ix <= sideWindowEnd) {
          return ix === sideWindowStart || ix === sideWindowEnd;
        }
        if (iy === sideWindowRow + 1 && ix >= sideWindowStart && ix <= sideWindowEnd) {
          return true;
        }
        return true;
      }

      return false;
    },
  });
}

export function buildCourtyardHouseScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const segments = { x: 28, y: 14, z: 28 };
  const courtyardMinX = Math.floor(segments.x * 0.28);
  const courtyardMaxX = segments.x - courtyardMinX - 1;
  const courtyardMinZ = Math.floor(segments.z * 0.28);
  const courtyardMaxZ = segments.z - courtyardMinZ - 1;
  const skylightRow = Math.floor(segments.y * 0.85);
  const lintelRow = Math.floor(segments.y * 0.45);

  return buildRectilinearScenario({
    size: makeVec(12.5, 4.4, 12.5),
    segments,
    deckMass: 880_000,
    areaScale: 0.054,
    addDiagonals: true,
    diagScale: 0.58,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onOuterShell = ix === 0 || ix === seg.x - 1 || iz === 0 || iz === seg.z - 1;
      const onCourtyardRing =
        ix === courtyardMinX ||
        ix === courtyardMaxX ||
        iz === courtyardMinZ ||
        iz === courtyardMaxZ;
      const inCourtyardVoid = ix > courtyardMinX && ix < courtyardMaxX && iz > courtyardMinZ && iz < courtyardMaxZ;
      const galleryBand = iy === Math.floor(seg.y * 0.32);

      if (iy === 0) return true; // slab
      if (galleryBand && onCourtyardRing) return true;
      if (onOuterShell) {
        const doorSpanStart = Math.floor(seg.z * 0.4);
        const doorSpanEnd = Math.floor(seg.z * 0.6);
        const doorHeight = Math.floor(seg.y * 0.35);
        if (ix === 0 && iz >= doorSpanStart && iz <= doorSpanEnd && iy <= doorHeight) {
          return iy === doorHeight || iz === doorSpanStart || iz === doorSpanEnd;
        }
        if (iy === lintelRow && (ix === 0 || ix === seg.x - 1)) {
          return true;
        }
        return true;
      }

      if (onCourtyardRing) {
        if (iy >= Math.floor(seg.y * 0.6)) {
          return false; // open clerestory around courtyard
        }
        if (iy === Math.floor(seg.y * 0.25)) {
          return true; // waist-high garden wall
        }
        return iy <= Math.floor(seg.y * 0.5);
      }

      if (inCourtyardVoid) {
        return iy === skylightRow && (ix - courtyardMinX) % 3 === 0 && (iz - courtyardMinZ) % 3 === 0;
      }

      return false;
    },
  });
}

export function buildVaultedLoftScenario({ bondsX = true, bondsY = true, bondsZ = true }: PresetOptions = {}): ScenarioDesc {
  const segments = { x: 24, y: 18, z: 14 };
  const mezzanineRow = Math.floor(segments.y * 0.4);
  const roofStart = Math.floor(segments.y * 0.55);
  const centerZ = (segments.z - 1) / 2;

  return buildRectilinearScenario({
    size: makeVec(10.5, 6.2, 5.2),
    segments,
    deckMass: 810_000,
    areaScale: 0.053,
    addDiagonals: true,
    diagScale: 0.6,
    normalizeAreas: true,
    bondsX,
    bondsY,
    bondsZ,
    includeNode: ({ ix, iy, iz, segments: seg }) => {
      const onShell = ix === 0 || ix === seg.x - 1 || iz === 0 || iz === seg.z - 1;
      const isFloor = iy === 0;
      const isMezzanine = iy === mezzanineRow && ix >= Math.floor(seg.x * 0.45);

      if (isFloor) return true;
      if (isMezzanine) return true;

      if (iy >= roofStart) {
        const normalizedHeight = (iy - roofStart) / Math.max(1, seg.y - roofStart - 1);
        const halfSpan = Math.cos(normalizedHeight * Math.PI * 0.5) * (seg.z * 0.5 - 1);
        const distanceFromRidge = Math.abs(iz - centerZ);
        const shellThickness = 2;
        const shellLimit = Math.max(0, halfSpan - (shellThickness - 1));

        if (distanceFromRidge >= shellLimit) {
          return true;
        }

        const ridgeBand = 1;
        if (distanceFromRidge <= ridgeBand && normalizedHeight >= 0.4) {
          return true;
        }

        const purlinSpacing = 3;
        const tieLayer = iy % 2 === 0;
        if (tieLayer && ix % purlinSpacing === 0 && distanceFromRidge >= halfSpan * 0.5) {
          return true;
        }

        return false;
      }

      if (onShell) {
        const clerestoryRow = Math.floor(seg.y * 0.48);
        if ((ix === 0 || ix === seg.x - 1) && iy === clerestoryRow) {
          return (iz + clerestoryRow) % 3 === 0;
        }
        if (iz === 0 || iz === seg.z - 1) {
          const garageDoorStart = Math.floor(seg.x * 0.15);
          const garageDoorEnd = Math.floor(seg.x * 0.55);
          const doorHeight = Math.floor(seg.y * 0.35);
          if (iz === 0 && ix >= garageDoorStart && ix <= garageDoorEnd && iy <= doorHeight) {
            return iy === doorHeight || ix === garageDoorStart || ix === garageDoorEnd;
          }
        }
        return true;
      }

      const stairCore = ix === Math.floor(seg.x * 0.4) && iz >= Math.floor(seg.z * 0.3) && iz <= Math.floor(seg.z * 0.7);
      if (stairCore && iy <= mezzanineRow) return true;

      return false;
    },
  });
}

export type StressPresetId =
  | "wall"
  | "brickWall"
  | "hut"
  | "bridge"
  | "beamBridge"
  | "tower"
  | "fracturedGlb"
  | "fracturedWall"
  | "fracturedWallHut"
  | "fracturedTower"
  | "fracturedBridge"
  | "townhouse"
  | "courtyardHouse"
  | "vaultedLoft";

export const STRESS_PRESET_METADATA: Array<{
  id: StressPresetId;
  label: string;
  description: string;
}> = [
  {
    id: "wall",
    label: "Tunable wall panel",
    description: "Baseline single wall for dialing in solver parameters and stress thresholds.",
  },
  {
    id: "brickWall",
    label: "Brick wall (running bond)",
    description: "Per-brick wall with half-brick staggering and clumped mortar strengths.",
  },
  {
    id: "hut",
    label: "Mini concrete hut",
    description: "Hollow four-wall shelter with a doorway and side window to showcase shell fragmentation.",
  },
  {
    id: "bridge",
    label: "Suspension footbridge",
    description: "Slender deck with towers and cable arcs to demonstrate tension and support failures.",
  },
  {
    id: "beamBridge",
    label: "Beam bridge",
    description: "Deck on end posts with massless footings; rectilinear blocks bonded as a slab.",
  },
  {
    id: "tower",
    label: "Multi-storey frame tower",
    description: "Tall frame with interior columns and floor plates for progressive collapse testing.",
  },
  {
    id: "fracturedWall",
    label: "Fractured wall",
    description: "Wall built from irregular fracture pieces (three-pinata) instead of a uniform grid.",
  },
  {
    id: "fracturedWallHut",
    label: "Fractured wall hut",
    description: "Hollow hut built from 4 fractured walls with auto-bonded corners and foundation.",
  },
  {
    id: "fracturedGlb",
    label: "Fractured GLB",
    description: "Fractures a GLB and simulates destruction with a foundation plate.",
  },
  {
    id: "fracturedTower",
    label: "Fractured tower",
    description: "Multi-floor tower built from fractured walls and floor plates with auto-bonded joints.",
  },
  {
    id: "fracturedBridge",
    label: "Fractured bridge",
    description: "Beam bridge built from fractured deck and support posts with auto-bonded joints.",
  },
  {
    id: "townhouse",
    label: "Two-storey townhouse",
    description: "Residential shell with an interior divider, door and window cut-outs, and a stepped gable roof.",
  },
  {
    id: "courtyardHouse",
    label: "Courtyard bungalow",
    description: "Low-rise home wrapping a central garden with breezeways, lintels, and clerestory openings.",
  },
  {
    id: "vaultedLoft",
    label: "Vaulted loft",
    description: "Open-plan loft with barrel roof ribs, garage door opening, and mezzanine platform.",
  },
];

export function getPresetMetadata(id: StressPresetId) {
  return STRESS_PRESET_METADATA.find((preset) => preset.id === id);
}
