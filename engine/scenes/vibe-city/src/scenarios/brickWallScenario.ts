import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import type {
  ColliderDescBuilder,
  ScenarioDesc,
  Vec3,
} from "@/lib/stress/core/types";

const EPSILON = 1e-8;

type BrickWallOptions = {
  span?: number; // X extent (meters)
  height?: number; // Y extent (meters)
  thickness?: number; // Z extent (meters)
  spanBricks?: number; // bricks across even rows (full bricks)
  courses?: number; // number of brick courses (rows)
  layers?: number; // layers along Z (multi-wythe)
  includeHalfBricks?: boolean; // running bond with half bricks on odd rows
  areaScale?: number; // scales contact area → bond area
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
  deckMass?: number; // total mass of non-support bricks (kg)
  mortarGap?: number; // visual gap by shrinking colliders (meters)
  // Clumping controls
  clumpCount?: number;
  clumpRadius?: number; // world-space radius; if omitted uses span*0.28
  weakRange?: [number, number];
  strongRange?: [number, number];
  seed?: number;
};

type RowBrick = { index: number; x0: number; x1: number; len: number };

function vec(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v.x, v.y, v.z);
  if (l <= EPSILON) return vec(0, 0, 0);
  return vec(v.x / l, v.y / l, v.z / l);
}

function overlap1D(a0: number, a1: number, b0: number, b1: number): number {
  return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
}

function smoothstep01(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

// Deterministic RNG (Mulberry32)
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildBrickWallScenario({
  span = 6.0,
  height = 3.0,
  thickness = 0.32,
  spanBricks = 24,
  courses = 12,
  layers = 1,
  includeHalfBricks = true,
  areaScale = 0.05,
  bondsX = true,
  bondsY = true,
  bondsZ = true,
  deckMass = 40_000,
  mortarGap = 0.0,
  clumpCount = 7,
  clumpRadius,
  weakRange = [0.5, 0.9],
  strongRange = [1.1, 1.6],
  seed,
}: BrickWallOptions = {}): ScenarioDesc {
  const L = Math.max(EPSILON, span / Math.max(1, spanBricks)); // full brick length
  const H = Math.max(EPSILON, height / Math.max(1, courses));
  const D = Math.max(EPSILON, thickness / Math.max(1, layers));
  const half = L * 0.5;
  const leftEdge = -span * 0.5;
  const originY = 0 + H * 0.5; // sit bottom on ground
  const originZ = 0; // layers centered about z=0
  const localClumpRadius = clumpRadius ?? span * 0.28;

  // Seeded RNG, fallback to Math.random
  const rand = typeof seed === 'number' ? makeRng(seed) : Math.random;

  const nodes: ScenarioDesc["nodes"] = [];
  const bonds: ScenarioDesc["bonds"] = [];
  const colliderDescForNode: (ColliderDescBuilder | null)[] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];
  const fragmentSizes: Array<{ x: number; y: number; z: number }> = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];

  // Per-layer, per-course row bricks for connectivity
  const rows: Array<Array<RowBrick[]>> = Array.from({ length: layers }, () => Array.from({ length: courses }, () => []));

  let totalVolume = 0;

  for (let iz = 0; iz < layers; iz += 1) {
    const cz = originZ + (iz - (layers - 1) * 0.5) * D;
    for (let iy = 0; iy < courses; iy += 1) {
      const y = originY + iy * H;
      const odd = includeHalfBricks && (iy % 2 === 1);
      const xStart = leftEdge;
      let xCursor = xStart;

      const pushBrick = (len: number, ixInRow: number) => {
        const x0 = xCursor;
        const x1 = x0 + len;
        const cx = (x0 + x1) * 0.5;
        const isSupport = iy === 0;
        const volume = isSupport ? 0 : len * H * D;
        if (!isSupport) totalVolume += volume;
        const nodeIndex = nodes.length;
        nodes.push({ centroid: vec(cx, y, cz), mass: volume, volume });
        rows[iz][iy].push({ index: nodeIndex, x0, x1, len });
        gridCoordinates[nodeIndex] = { ix: ixInRow, iy, iz };
        fragmentSizes[nodeIndex] = { x: len, y: H, z: D };
        const hx = len * 0.5 - mortarGap * 0.5;
        const hy = H * 0.5 - mortarGap * 0.5;
        const hz = D * 0.5 - mortarGap * 0.5;
        colliderDescForNode.push(() =>
          RAPIER.ColliderDesc.cuboid(
            Math.max(hx, EPSILON) * (isSupport ? 0.999 : 1),
            Math.max(hy, EPSILON) * (isSupport ? 0.999 : 1),
            Math.max(hz, EPSILON) * (isSupport ? 0.999 : 1),
          ),
        );
        // Create geometry for autoBonding (geometry is centered at origin; world position handled via matrix)
        const geom = new THREE.BoxGeometry(len, H, D);
        fragmentGeometries[nodeIndex] = geom;
        xCursor = x1;
      };

      if (!odd) {
        // Even row: all full bricks
        for (let i = 0; i < spanBricks; i += 1) pushBrick(L, i);
      } else {
        // Odd row: half + full x (spanBricks - 1) + half
        pushBrick(half, 0);
        for (let i = 0; i < spanBricks - 1; i += 1) pushBrick(L, i + 1);
        pushBrick(half, spanBricks);
      }
    }
  }

  // Scale masses from volumes
  const massScale = totalVolume > 0 ? deckMass / totalVolume : 0;
  if (massScale > 0) {
    nodes.forEach((n) => { n.mass = n.volume > 0 ? n.volume * massScale : 0; });
  } else {
    nodes.forEach((n) => { n.mass = 0; });
  }

  const addBond = (a: number, b: number, area: number) => {
    if (a < 0 || b < 0) return;
    const na = nodes[a];
    const nb = nodes[b];
    const centroid = vec((na.centroid.x + nb.centroid.x) * 0.5, (na.centroid.y + nb.centroid.y) * 0.5, (na.centroid.z + nb.centroid.z) * 0.5);
    const normal = normalize(sub(nb.centroid, na.centroid));
    bonds.push({ node0: a, node1: b, centroid, normal, area: Math.max(area, EPSILON) });
  };

  // X bonds: neighbors within a row/layer
  if (bondsX) {
    for (let iz = 0; iz < layers; iz += 1) {
      for (let iy = 0; iy < courses; iy += 1) {
        const row = rows[iz][iy];
        for (let i = 0; i + 1 < row.length; i += 1) {
          const a = row[i];
          const b = row[i + 1];
          addBond(a.index, b.index, H * D * areaScale);
        }
      }
    }
  }

  // Y bonds: overlap between adjacent rows (running bond)
  if (bondsY) {
    for (let iz = 0; iz < layers; iz += 1) {
      for (let iy = 0; iy + 1 < courses; iy += 1) {
        const a = rows[iz][iy];
        const b = rows[iz][iy + 1];
        let i = 0, j = 0;
        while (i < a.length && j < b.length) {
          const ai = a[i];
          const bj = b[j];
          const ov = overlap1D(ai.x0, ai.x1, bj.x0, bj.x1);
          if (ov > EPSILON) addBond(ai.index, bj.index, ov * D * areaScale);
          if (ai.x1 < bj.x1 - EPSILON) i += 1; else if (bj.x1 < ai.x1 - EPSILON) j += 1; else { i += 1; j += 1; }
        }
      }
    }
  }

  // Z bonds: between layers (multi-wythe)
  if (bondsZ && layers > 1) {
    for (let iz = 0; iz + 1 < layers; iz += 1) {
      for (let iy = 0; iy < courses; iy += 1) {
        const a = rows[iz][iy];
        const b = rows[iz + 1][iy];
        let i = 0, j = 0;
        while (i < a.length && j < b.length) {
          const ai = a[i];
          const bj = b[j];
          const ov = overlap1D(ai.x0, ai.x1, bj.x0, bj.x1);
          if (ov > EPSILON) addBond(ai.index, bj.index, ov * H * areaScale);
          if (ai.x1 < bj.x1 - EPSILON) i += 1; else if (bj.x1 < ai.x1 - EPSILON) j += 1; else { i += 1; j += 1; }
        }
      }
    }
  }

  // Clumped strength multipliers over XY
  const centers: Array<{ x: number; y: number; scale: number }> = [];
  for (let i = 0; i < clumpCount; i += 1) {
    const cx = leftEdge + rand() * span;
    const cy = rand() * height;
    const strong = rand() < 0.5;
    const r = strong ? strongRange : weakRange;
    const scale = r[0] + rand() * Math.max(0, r[1] - r[0]);
    centers.push({ x: cx, y: cy, scale });
  }

  if (centers.length && bonds.length) {
    for (const b of bonds) {
      const px = b.centroid.x;
      const py = b.centroid.y;
      let bestD = Infinity;
      let s = 1.0;
      for (let i = 0; i < centers.length; i += 1) {
        const c = centers[i];
        const d = Math.hypot(px - c.x, py - c.y);
        if (d < bestD) {
          bestD = d;
          s = c.scale;
        }
      }
      const t = Math.max(0, 1 - bestD / localClumpRadius);
      const w = smoothstep01(t);
      const m = 1 + (s - 1) * w;
      b.area = Math.max(EPSILON, b.area * m);
    }
  }

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: vec(L, H, D),
    parameters: {
      span,
      height,
      thickness,
      spanBricks,
      courses,
      layers,
      areaScale,
      fragmentSizes,
      fragmentGeometries,
    },
    colliderDescForNode,
  } satisfies ScenarioDesc;
}

