import { DestructibleMesh, FractureOptions } from "@dgreenheck/three-pinata";
import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import type {
  ColliderDescBuilder,
  ScenarioDesc,
  Vec3,
} from "@/lib/stress/core/types";
import {
  generateAutoBondsFromChunks,
  type AutoBondChunkInput,
} from "@/lib/stress/core/autoBonding";

/**
 * Fragment types for bond strength calculation.
 */
export type FragmentType = "column" | "floor" | "wall" | "foundation";

/**
 * Represents a single fragment piece with its world position, extents, geometry, and support status.
 */
export type FragmentInfo = {
  worldPosition: THREE.Vector3;
  halfExtents: THREE.Vector3;
  geometry: THREE.BufferGeometry;
  isSupport: boolean;
  /** Fragment type for bond strength calculation */
  fragmentType?: FragmentType;
};

/**
 * Creates fractured wall fragments for a single wall panel, then transforms them
 * to the specified world position and rotation.
 */
export function buildWallFragments(
  span: number,
  height: number,
  thickness: number,
  fragmentCount: number,
  wallCenterX: number,
  wallCenterZ: number,
  rotationY: number,
  wallLiftY: number,
): FragmentInfo[] {
  // Create wall geometry aligned along X axis (span in X, height in Y, thickness in Z)
  const geom = new THREE.BoxGeometry(span, height, thickness, 2, 3, 1);
  const opts = new FractureOptions({
    fractureMethod: "voronoi",
    fragmentCount: fragmentCount,
    voronoiOptions: { mode: "3D" },
  });

  const destructibleMesh = new DestructibleMesh(geom);
  const pieceMeshes = destructibleMesh.fracture(opts);
  geom.dispose();

  // Wall center position (bottom of wall at wallLiftY)
  const center = new THREE.Vector3(wallCenterX, height * 0.5 + wallLiftY, wallCenterZ);

  // Rotation matrix for the wall
  const rotMatrix = new THREE.Matrix4().makeRotationY(rotationY);

  const fragments: FragmentInfo[] = pieceMeshes.map((m) => {
    const g = m.geometry;
    g.computeBoundingBox();
    const bbox = g.boundingBox as THREE.Box3;
    const localCenter = new THREE.Vector3();
    bbox.getCenter(localCenter);
    
    // Re-center geometry around its own COM
    g.translate(-localCenter.x, -localCenter.y, -localCenter.z);
    
    const size = new THREE.Vector3();
    bbox.getSize(size);

    // Apply rotation to local offset + voronoi center
    // The m.position puts the piece in the correct relative spot in the original wall box space.
    // localCenter is the offset from that piece's origin to its bbox center.
    // We combine them before rotation.
    const preRotationOffset = new THREE.Vector3(
      m.position.x + localCenter.x,
      m.position.y + localCenter.y,
      m.position.z + localCenter.z
    );

    const rotatedOffset = preRotationOffset.applyMatrix4(rotMatrix);
    
    const worldPosition = new THREE.Vector3(
      center.x + rotatedOffset.x,
      center.y + rotatedOffset.y,
      center.z + rotatedOffset.z,
    );

    // Rotate the geometry to match wall orientation
    g.applyMatrix4(rotMatrix);

    // Compute rotated half extents (for bounding box after rotation)
    const rotatedSize = new THREE.Vector3(size.x, size.y, size.z);
    if (Math.abs(rotationY) > 0.01) {
      // For 90 degree rotations, swap X and Z
      const cos = Math.abs(Math.cos(rotationY));
      const sin = Math.abs(Math.sin(rotationY));
      rotatedSize.x = size.x * cos + size.z * sin;
      rotatedSize.z = size.x * sin + size.z * cos;
    }

    return {
      worldPosition,
      halfExtents: new THREE.Vector3(
        Math.max(0.05, rotatedSize.x * 0.5),
        Math.max(0.05, size.y * 0.5),
        Math.max(0.05, rotatedSize.z * 0.5),
      ),
      geometry: g,
      isSupport: false,
      fragmentType: "wall" as const,
    };
  });

  return fragments;
}

/**
 * Creates fractured floor plate fragments for a horizontal slab.
 * The floor is a flat box (spanX x thickness x spanZ) positioned at centerY.
 */
export function buildFloorFragments(
  spanX: number,
  spanZ: number,
  thickness: number,
  fragmentCount: number,
  centerX: number,
  centerY: number,
  centerZ: number,
): FragmentInfo[] {
  // Create floor geometry: X is width, Y is thickness, Z is depth
  const geom = new THREE.BoxGeometry(spanX, thickness, spanZ, 3, 1, 3);
  const opts = new FractureOptions({
    fractureMethod: "voronoi",
    fragmentCount: fragmentCount,
    voronoiOptions: { mode: "3D" },
  });

  const destructibleMesh = new DestructibleMesh(geom);
  const pieceMeshes = destructibleMesh.fracture(opts);
  geom.dispose();

  // Floor center position
  const center = new THREE.Vector3(centerX, centerY, centerZ);

  const fragments: FragmentInfo[] = pieceMeshes.map((m) => {
    const g = m.geometry;
    g.computeBoundingBox();
    const bbox = g.boundingBox as THREE.Box3;
    const localCenter = new THREE.Vector3();
    bbox.getCenter(localCenter);
    
    // Re-center geometry around its own COM
    g.translate(-localCenter.x, -localCenter.y, -localCenter.z);
    
    const size = new THREE.Vector3();
    bbox.getSize(size);

    // Combine m.position + localCenter for the world offset
    const worldPosition = new THREE.Vector3(
      center.x + m.position.x + localCenter.x,
      center.y + m.position.y + localCenter.y,
      center.z + m.position.z + localCenter.z,
    );

    return {
      worldPosition,
      halfExtents: new THREE.Vector3(
        Math.max(0.05, size.x * 0.5),
        Math.max(0.05, size.y * 0.5),
        Math.max(0.05, size.z * 0.5),
      ),
      geometry: g,
      isSupport: false,
      fragmentType: "floor" as const,
    };
  });

  return fragments;
}

/**
 * Creates fractured column (vertical beam) fragments.
 * The column is a vertical box (sizeX x height x sizeZ) positioned at (centerX, baseY + height/2, centerZ).
 */
export function buildColumnFragments(
  sizeX: number,
  sizeZ: number,
  height: number,
  fragmentCount: number,
  centerX: number,
  baseY: number,
  centerZ: number,
): FragmentInfo[] {
  // Create column geometry: X is width, Y is height, Z is depth
  const geom = new THREE.BoxGeometry(sizeX, height, sizeZ, 1, 3, 1);
  const opts = new FractureOptions({
    fractureMethod: "voronoi",
    fragmentCount: fragmentCount,
    voronoiOptions: { mode: "3D" },
  });

  const destructibleMesh = new DestructibleMesh(geom);
  const pieceMeshes = destructibleMesh.fracture(opts);
  geom.dispose();

  // Column center position (baseY is the bottom of the column)
  const center = new THREE.Vector3(centerX, baseY + height * 0.5, centerZ);

  const fragments: FragmentInfo[] = pieceMeshes.map((m) => {
    const g = m.geometry;
    g.computeBoundingBox();
    const bbox = g.boundingBox as THREE.Box3;
    const localCenter = new THREE.Vector3();
    bbox.getCenter(localCenter);
    
    // Re-center geometry around its own COM
    g.translate(-localCenter.x, -localCenter.y, -localCenter.z);
    
    const size = new THREE.Vector3();
    bbox.getSize(size);

    // Combine m.position + localCenter for the world offset
    const worldPosition = new THREE.Vector3(
      center.x + m.position.x + localCenter.x,
      center.y + m.position.y + localCenter.y,
      center.z + m.position.z + localCenter.z,
    );

    return {
      worldPosition,
      halfExtents: new THREE.Vector3(
        Math.max(0.05, size.x * 0.5),
        Math.max(0.05, size.y * 0.5),
        Math.max(0.05, size.z * 0.5),
      ),
      geometry: g,
      isSupport: false,
      fragmentType: "column" as const,
    };
  });

  return fragments;
}

/**
 * Projects geometry extents onto an axis in world space.
 */
export function projectExtentsOnAxisWorld(
  geometry: THREE.BufferGeometry,
  worldPos: THREE.Vector3,
  axis: THREE.Vector3,
): { min: number; max: number } {
  const pos = geometry.getAttribute("position") as THREE.BufferAttribute;
  const ax = axis;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < pos.count; i += 1) {
    const x = (pos.getX(i) as number) + worldPos.x;
    const y = (pos.getY(i) as number) + worldPos.y;
    const z = (pos.getZ(i) as number) + worldPos.z;
    const p = x * ax.x + y * ax.y + z * ax.z;
    if (p < min) min = p;
    if (p > max) max = p;
  }
  return { min, max };
}

/**
 * Computes the 1D overlap between two intervals.
 */
export function overlap1D(
  a: { min: number; max: number },
  b: { min: number; max: number },
): number {
  return Math.max(0, Math.min(a.max, b.max) - Math.max(a.min, b.min));
}

/**
 * Computes bonds between fragment pairs that are nearly touching.
 */
export function computeBondsFromFragments(
  fragments: FragmentInfo[],
): Array<{ a: number; b: number; centroid: Vec3; normal: Vec3; area: number }> {
  const bonds: Array<{
    a: number;
    b: number;
    centroid: Vec3;
    normal: Vec3;
    area: number;
  }> = [];
  if (fragments.length === 0) return bonds;

  // Global tolerance baseline; refined per-pair below
  const globalTol =
    0.12 *
    Math.min(
      ...fragments.map((f) => Math.min(f.halfExtents.x, f.halfExtents.z)),
    );

  for (let i = 0; i < fragments.length; i += 1) {
    for (let j = i + 1; j < fragments.length; j += 1) {
      const A = fragments[i];
      const B = fragments[j];

      // Skip bonds between two supports; treat foundation as a single anchored slab
      if (A.isSupport && B.isSupport) continue;

      const n = B.worldPosition.clone().sub(A.worldPosition).normalize();
      if (
        !Number.isFinite(n.x) ||
        !Number.isFinite(n.y) ||
        !Number.isFinite(n.z)
      )
        continue;

      // Require small separation along the normal direction (nearly touching)
      const aN = projectExtentsOnAxisWorld(A.geometry, A.worldPosition, n);
      const bN = projectExtentsOnAxisWorld(B.geometry, B.worldPosition, n);
      const separation = bN.min - aN.max; // positive if B is in +n direction away from A

      // Use two tangents for overlap area test
      const up =
        Math.abs(n.y) < 0.9
          ? new THREE.Vector3(0, 1, 0)
          : new THREE.Vector3(1, 0, 0);
      const t1 = new THREE.Vector3().crossVectors(n, up).normalize();
      const t2 = new THREE.Vector3().crossVectors(n, t1).normalize();

      const a1 = projectExtentsOnAxisWorld(A.geometry, A.worldPosition, t1);
      const b1 = projectExtentsOnAxisWorld(B.geometry, B.worldPosition, t1);
      const a2 = projectExtentsOnAxisWorld(A.geometry, A.worldPosition, t2);
      const b2 = projectExtentsOnAxisWorld(B.geometry, B.worldPosition, t2);

      const o1 = overlap1D(a1, b1);
      const o2 = overlap1D(a2, b2);
      const size1 = Math.min(a1.max - a1.min, b1.max - b1.min);
      const size2 = Math.min(a2.max - a2.min, b2.max - b2.min);

      // Pair-relative gap threshold to avoid long-range bonds across thin gaps
      const pairMin = Math.max(1e-6, Math.min(size1, size2));
      const epsGap = Math.max(0.006, Math.min(globalTol, pairMin * 0.15));
      if (separation > epsGap) continue;

      if (o1 < size1 * 0.22 || o2 < size2 * 0.22) continue;

      const contactArea = o1 * o2;
      if (!(contactArea > 0)) continue;

      // Contact centroid: the center of the overlap rectangle in the (n, t1, t2) basis
      const cN = 0.5 * (Math.max(aN.min, bN.min) + Math.min(aN.max, bN.max));
      const c1 = 0.5 * (Math.max(a1.min, b1.min) + Math.min(a1.max, b1.max));
      const c2 = 0.5 * (Math.max(a2.min, b2.min) + Math.min(a2.max, b2.max));
      const mid = new THREE.Vector3()
        .addScaledVector(n, cN)
        .addScaledVector(t1, c1)
        .addScaledVector(t2, c2);

      bonds.push({
        a: i,
        b: j,
        centroid: { x: mid.x, y: mid.y, z: mid.z },
        normal: { x: n.x, y: n.y, z: n.z },
        area: contactArea,
      });
    }
  }
  return bonds;
}

/**
 * Normalizes bond areas by axis so material behavior is uniform across the structure.
 */
export function normalizeFractureAreasByAxis(
  list: Array<{
    a?: number;
    b?: number;
    centroid: Vec3;
    normal: Vec3;
    area: number;
  }>,
  dims: { width: number; depth: number; height: number },
) {
  // Target areas based on cross-sectional areas
  const target = {
    x: dims.height * dims.depth,
    y: dims.width * dims.depth,
    z: dims.width * dims.height,
  };
  const sum = { x: 0, y: 0, z: 0 };
  const pick = (n: Vec3): "x" | "y" | "z" => {
    const ax = Math.abs(n.x),
      ay = Math.abs(n.y),
      az = Math.abs(n.z);
    return ax >= ay && ax >= az ? "x" : ay >= az ? "y" : "z";
  };
  for (const b of list) sum[pick(b.normal)] += b.area;
  const scale = {
    x: sum.x > 0 ? target.x / sum.x : 1,
    y: sum.y > 0 ? target.y / sum.y : 1,
    z: sum.z > 0 ? target.z / sum.z : 1,
  } as const;
  return list.map((b) => {
    const axis = pick(b.normal);
    const area = Math.max(b.area * scale[axis], 1e-8);
    return { ...b, area };
  });
}

export type BuildScenarioOptions = {
  /**
   * Use auto bonding instead of legacy bond computation.
   * When true, uses the optimized WASM-based auto bonding for better accuracy and performance.
   */
  useAutoBonding?: boolean;
};

/**
 * Calculate bond strength multiplier based on the two fragment types.
 * Order of strength:
 * 1. Column-column: very strong (4x)
 * 2. Column-anything: strong (2.5x)
 * 3. Floor-floor: strong (2x)
 * 4. Floor-wall: medium (1.5x)
 * 5. Wall-wall: baseline (1x)
 */
function getBondStrengthMultiplier(
  type0: FragmentType | undefined,
  type1: FragmentType | undefined,
): number {
  // Column-column is strongest
  if (type0 === "column" && type1 === "column") {
    return 4.0;
  }
  // Column-anything is strong
  if (type0 === "column" || type1 === "column") {
    return 2.5;
  }
  // Floor-floor is strong
  if (type0 === "floor" && type1 === "floor") {
    return 2.0;
  }
  // Floor-wall is medium
  if (
    (type0 === "floor" && type1 === "wall") ||
    (type0 === "wall" && type1 === "floor")
  ) {
    return 1.5;
  }
  // Wall-wall is baseline
  return 1.0;
}

/**
 * Builds the scenario description (nodes, bonds, colliders) from a list of fragments.
 * 
 * @param allFragments - Array of fragment pieces with positions and geometries
 * @param dims - Structure dimensions for bond normalization
 * @param deckMass - Total mass to distribute across fragments
 * @param extraParams - Additional parameters to include in scenario
 * @param options - Build options (e.g., useAutoBonding for optimized bonding)
 */
export async function buildScenarioFromFragments(
  allFragments: FragmentInfo[],
  dims: { width: number; depth: number; height: number },
  deckMass: number,
  extraParams: Record<string, unknown> = {},
  options: BuildScenarioOptions = {},
): Promise<ScenarioDesc> {
  const { useAutoBonding = false } = options;

  const nodes: ScenarioDesc["nodes"] = [];
  const fragmentSizes: Vec3[] = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];
  const fragmentTypes: (FragmentType | undefined)[] = [];
  const colliderDescForNode: (ColliderDescBuilder | null)[] = [];
  let totalVolume = 0;

  allFragments.forEach((f, i) => {
    const hx = f.halfExtents.x;
    const hy = f.halfExtents.y;
    const hz = f.halfExtents.z;
    const size = { x: hx * 2, y: hy * 2, z: hz * 2 } satisfies Vec3;
    const volume = size.x * size.y * size.z;
    const isSupport = f.isSupport;
    const mass = isSupport ? 0 : volume;
    if (!isSupport) totalVolume += volume;

    nodes.push({
      centroid: {
        x: f.worldPosition.x,
        y: f.worldPosition.y,
        z: f.worldPosition.z,
      },
      mass,
      volume,
    });
    fragmentSizes.push(size);
    fragmentGeometries.push(f.geometry);
    fragmentTypes.push(f.fragmentType);

    if (isSupport) {
      colliderDescForNode[i] = () => RAPIER.ColliderDesc.cuboid(hx, hy, hz);
    } else {
      const pos = f.geometry.getAttribute("position") as THREE.BufferAttribute;
      const points =
        pos?.array instanceof Float32Array
          ? pos.array
          : new Float32Array((pos?.array ?? []) as ArrayLike<number>);
      colliderDescForNode[i] = () => RAPIER.ColliderDesc.convexHull(points);
    }
  });

  // Scale masses so total matches deckMass
  const scale = totalVolume > 0 ? deckMass / totalVolume : 0;
  if (scale > 0) {
    for (const n of nodes) {
      if (n.mass === 0) continue;
      n.mass = n.volume > 0 ? n.volume * scale : 0;
    }
  } else {
    for (const n of nodes) n.mass = 0;
  }

  // Compute bonds using either auto bonding (optimized) or legacy method
  let bonds: ScenarioDesc["bonds"] = [];
  if (useAutoBonding) {
    // Use optimized WASM-based auto bonding
    const chunks: AutoBondChunkInput[] = allFragments.map((f) => ({
      geometry: f.geometry,
      isSupport: f.isSupport,
      matrix: new THREE.Matrix4().makeTranslation(
        f.worldPosition.x,
        f.worldPosition.y,
        f.worldPosition.z,
      ),
    }));
    const autoBonds = await generateAutoBondsFromChunks(chunks, {
      label: "buildScenarioFromFragments",
    });
    if (autoBonds?.length) {
      bonds = autoBonds;
    }
  } else {
    // Use legacy bond computation
    const rawBonds = computeBondsFromFragments(allFragments);
    const normBonds = normalizeFractureAreasByAxis(rawBonds, dims);
    bonds = normBonds.map((b) => ({
      node0: (b as { a: number }).a,
      node1: (b as { b: number }).b,
      centroid: b.centroid,
      normal: b.normal,
      area: Math.max(b.area, 1e-8),
    }));
  }

  // Apply bond strength multipliers based on fragment types
  // Order of strength: column-column (4x) > column-anything (2x) > floor-floor (2x) > wall-wall (1x)
  bonds = bonds.map((bond) => {
    const type0 = fragmentTypes[bond.node0];
    const type1 = fragmentTypes[bond.node1];
    const multiplier = getBondStrengthMultiplier(type0, type1);
    return {
      ...bond,
      area: bond.area * multiplier,
    };
  });

  return {
    nodes,
    bonds,
    parameters: {
      fragmentSizes,
      fragmentGeometries,
      fragmentTypes,
      ...dims,
      ...extraParams,
    },
    colliderDescForNode,
  } satisfies ScenarioDesc;
}

/**
 * Adds rectangular foundation support nodes under a footprint.
 */
export function addFoundationFragments(
  allFragments: FragmentInfo[],
  width: number,
  depth: number,
  foundationHeight: number,
  groundClearance: number,
): void {
  const halfWidth = width * 0.5;
  const halfDepth = depth * 0.5;
  // Target tile size scales with structure size - aim for roughly 6-8 tiles per side
  const tileSize = Math.max(1.0, Math.min(width, depth) / 6);
  const foundationSegmentsX = Math.max(4, Math.round(width / tileSize));
  const foundationSegmentsZ = Math.max(4, Math.round(depth / tileSize));
  const cellW = width / foundationSegmentsX;
  const cellD = depth / foundationSegmentsZ;

  for (let ix = 0; ix < foundationSegmentsX; ix += 1) {
    for (let iz = 0; iz < foundationSegmentsZ; iz += 1) {
      const cx = -halfWidth + cellW * (ix + 0.5);
      const cz = -halfDepth + cellD * (iz + 0.5);
      const cy = groundClearance + foundationHeight * 0.5;
      const g = new THREE.BoxGeometry(cellW, foundationHeight, cellD);
      const worldPosition = new THREE.Vector3(cx, cy, cz);
      allFragments.push({
        worldPosition,
        halfExtents: new THREE.Vector3(
          cellW * 0.5,
          foundationHeight * 0.5,
          cellD * 0.5,
        ),
        geometry: g,
        isSupport: true,
        fragmentType: "foundation" as const,
      });
    }
  }
}
