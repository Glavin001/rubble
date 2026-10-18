import RAPIER from "@dimforge/rapier3d-compat";
import { vec3 } from "blast-stress-solver";
import * as THREE from "three";
import type {
  ColliderDescBuilder,
  ScenarioBond,
  ScenarioDesc,
  ScenarioNode,
  Vec3,
} from "@/lib/stress/core/types";

type WallScenarioOptions = {
  span?: number; // X
  height?: number; // Y
  thickness?: number; // Z
  spanSegments?: number;
  heightSegments?: number;
  layers?: number; // Z layers
  deckMass?: number;
  areaScale?: number;
  addDiagonals?: boolean;
  diagScale?: number;
  normalizeAreas?: boolean;
  seed?: number; // optional seed to vary builds, currently unused but kept for API stability
  bondsX?: boolean; // enable side-to-side bonds
  bondsY?: boolean; // enable vertical bonds
  bondsZ?: boolean; // enable depth bonds (requires layers>1)
};

export function buildWallScenario({
  span = 6.0,
  height = 3.0,
  thickness = 0.32,
  spanSegments = 12,
  heightSegments = 6,
  layers = 1,
  // deckMass = 10_000.0,
  deckMass = 10_000.0,
  areaScale = 0.05,
  addDiagonals = false,
  diagScale = 0.75,
  normalizeAreas = true,
  bondsX = true,
  bondsY = true,
  bondsZ = true,
}: WallScenarioOptions = {}): ScenarioDesc {
  const nodes: ScenarioNode[] = [];
  const bonds: ScenarioBond[] = [];
  const colliderDescForNode: (ColliderDescBuilder | null)[] = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];

  const cellX = span / Math.max(spanSegments, 1);
  const cellY = height / Math.max(heightSegments, 1);
  const cellZ = thickness / Math.max(layers, 1);

  const originX = -span * 0.5 + 0.5 * cellX;
  const originY = 0 + 0.5 * cellY; // bottom sits on ground (y=0)
  const originZ = 0; // single layer centered at z=0 for MVP

  const totalNodes = spanSegments * heightSegments * layers;
  const massPerNode = deckMass / Math.max(totalNodes, 1);
  const volumePerNode = cellX * cellZ * cellY;

  const index3D: number[][][] = Array.from({ length: spanSegments }, () => Array.from({ length: heightSegments }, () => Array(layers)));
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];

  // Populate nodes (bottom row iy===0 are supports: mass=0)
  for (let ix = 0; ix < spanSegments; ix++) {
    for (let iy = 0; iy < heightSegments; iy++) {
      for (let iz = 0; iz < layers; iz++) {
        const centroid = vec3(
          originX + ix * cellX,
          originY + iy * cellY,
          originZ + (iz - (layers - 1) * 0.5) * cellZ,
        ) as unknown as Vec3;
        const isSupport = iy === 0;
        const node = {
          centroid,
          mass: isSupport ? 0 : massPerNode,
          volume: isSupport ? 0 : volumePerNode,
        };
        const index = nodes.length;
        nodes.push(node);
        index3D[ix][iy][iz] = index;
        gridCoordinates[index] = { ix, iy, iz };

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

  const areaX = cellY * cellZ * areaScale;
  const areaY = cellX * cellZ * areaScale;
  const areaZ = cellX * cellY * areaScale;

  const addBond = (a: number, b: number, area: number) => {
    const na = nodes[a];
    const nb = nodes[b];
    const centroid = vec3(
      (na.centroid.x + nb.centroid.x) * 0.5,
      (na.centroid.y + nb.centroid.y) * 0.5,
      (na.centroid.z + nb.centroid.z) * 0.5,
    ) as unknown as Vec3;
    const normal = normalize(subVec(nb.centroid, na.centroid));
    bonds.push({ node0: a, node1: b, centroid, normal, area: Math.max(area, 1e-8) });
  };

  // Face neighbors only (+X, +Y, +Z)
  for (let ix = 0; ix < spanSegments; ix++) {
    for (let iy = 0; iy < heightSegments; iy++) {
      for (let iz = 0; iz < layers; iz++) {
        const current = index3D[ix][iy][iz];
        if (bondsX && ix + 1 < spanSegments) addBond(current, index3D[ix + 1][iy][iz], areaX);
        if (bondsY && iy + 1 < heightSegments) addBond(current, index3D[ix][iy + 1][iz], areaY);
        if (bondsZ && iz + 1 < layers) addBond(current, index3D[ix][iy][iz + 1], areaZ);
        if (addDiagonals) {
          if (ix + 1 < spanSegments && iy + 1 < heightSegments) addBond(current, index3D[ix + 1][iy + 1][iz], 0.5 * (areaX + areaY) * diagScale);
          if (ix + 1 < spanSegments && iz + 1 < layers) addBond(current, index3D[ix + 1][iy][iz + 1], 0.5 * (areaX + areaZ) * diagScale);
          if (iy + 1 < heightSegments && iz + 1 < layers) addBond(current, index3D[ix][iy + 1][iz + 1], 0.5 * (areaY + areaZ) * diagScale);
        }
      }
    }
  }

  // Optional axis normalization
  if (normalizeAreas && bonds.length) {
    const target = { x: height * thickness, y: span * thickness, z: span * height };
    const sum = { x: 0, y: 0, z: 0 };
    const pick = (n: Vec3): 'x' | 'y' | 'z' => {
      const ax = Math.abs(n.x), ay = Math.abs(n.y), az = Math.abs(n.z);
      return ax >= ay && ax >= az ? 'x' : (ay >= az ? 'y' : 'z');
    };
    bonds.forEach((b) => { sum[pick(b.normal)] += b.area; });
    const scale = {
      x: sum.x > 0 ? target.x / sum.x : 1,
      y: sum.y > 0 ? target.y / sum.y : 1,
      z: sum.z > 0 ? target.z / sum.z : 1,
    } as const;
    bonds.forEach((b) => { b.area *= scale[pick(b.normal)]; });
  }

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: { x: cellX, y: cellY, z: cellZ },
    parameters: {
      span,
      height,
      thickness,
      spanSegments,
      heightSegments,
      layers,
      deckMass,
      areaScale,
      fragmentGeometries,
    },
    colliderDescForNode,
  } satisfies ScenarioDesc;
}

function subVec(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z);
  if (len === 0) return { x: 0, y: 0, z: 0 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}


