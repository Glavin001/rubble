import { DestructibleMesh, FractureOptions } from "@dgreenheck/three-pinata";
import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import type {
  ColliderDescBuilder,
  ScenarioDesc,
  Vec3,
} from "@/lib/stress/core/types";

type FracturedWallOptions = {
  span?: number; // X
  height?: number; // Y
  thickness?: number; // Z
  fragmentCount?: number;
  deckMass?: number;
};

type FragmentInfo = {
  worldPosition: THREE.Vector3;
  halfExtents: THREE.Vector3;
  geometry: THREE.BufferGeometry;
  isSupport: boolean;
};

type FragmentBuildOptions = Required<
  Pick<
    FracturedWallOptions,
    "span" | "height" | "thickness" | "fragmentCount"
  >
>;

function buildFragments({
  span,
  height,
  thickness,
  fragmentCount,
}: FragmentBuildOptions): FragmentInfo[] {
  const geom = new THREE.BoxGeometry(span, height, thickness, 2, 3, 1);
  const opts = new FractureOptions({
    fractureMethod: "voronoi",
    fragmentCount: fragmentCount,
    voronoiOptions: { mode: "3D" },
  });

  const destructibleMesh = new DestructibleMesh(geom);
  const pieceMeshes = destructibleMesh.fracture(opts);
  geom.dispose();

  // Place foundation slightly above the ground and lift the wall so it sits above the foundation with a tiny gap.
  const foundationHeight = Math.min(0.08, height * 0.06);
  const groundClearance = Math.max(0.001, foundationHeight * 0.05);
  const foundationClearance = Math.max(0.001, foundationHeight * 0.05);
  const wallLiftY = groundClearance + foundationHeight + foundationClearance;

  const center = new THREE.Vector3(0, height * 0.5 + wallLiftY, 0); // bottom at y>0

  const fragments: FragmentInfo[] = pieceMeshes.map((m) => {
    const g = m.geometry;
    g.computeBoundingBox();
    const bbox = g.boundingBox as THREE.Box3;
    const localCenter = new THREE.Vector3();
    bbox.getCenter(localCenter);
    // Re-center geometry around its own COM so physics translation drives placement
    g.translate(-localCenter.x, -localCenter.y, -localCenter.z);
    const size = new THREE.Vector3();
    bbox.getSize(size);
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
    };
  });

  // Add a thin foundation slab along X under the wall (support-only nodes)
  const foundationSegmentsX = Math.max(6, Math.round(span / 0.5));
  const cellW = span / foundationSegmentsX;
  for (let ix = 0; ix < foundationSegmentsX; ix += 1) {
    const cx = -span * 0.5 + cellW * (ix + 0.5);
    const cy = groundClearance + foundationHeight * 0.5; // sits just above ground
    const cz = 0;
    const g = new THREE.BoxGeometry(cellW, foundationHeight, thickness);
    const worldPosition = new THREE.Vector3(cx, cy, cz);
    fragments.push({
      worldPosition,
      halfExtents: new THREE.Vector3(
        cellW * 0.5,
        foundationHeight * 0.5,
        thickness * 0.5,
      ),
      geometry: g,
      isSupport: true,
    });
  }
  return fragments;
}

function projectExtentsOnAxisWorld(
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

function overlap1D(
  a: { min: number; max: number },
  b: { min: number; max: number },
) {
  return Math.max(0, Math.min(a.max, b.max) - Math.max(a.min, b.min));
}

function computeBondsFromFragments(
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

function normalizeFractureAreasByAxis(
  list: Array<{
    a?: number;
    b?: number;
    centroid: Vec3;
    normal: Vec3;
    area: number;
  }>,
  dims: { span: number; height: number; thickness: number },
) {
  const target = {
    x: dims.height * dims.thickness,
    y: dims.span * dims.thickness,
    z: dims.span * dims.height,
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

export function buildFracturedWallScenario({
  span = 6.0,
  height = 3.0,
  thickness = 0.32,
  fragmentCount = 120,
  deckMass = 10_000,
}: FracturedWallOptions = {}): ScenarioDesc {
  const frags = buildFragments({ span, height, thickness, fragmentCount });

  // Approximate per-fragment volume via bbox; supports (mass=0) if bottom touches ground
  const nodes: ScenarioDesc["nodes"] = [];
  const fragmentSizes: Vec3[] = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];
  const colliderDescForNode: (ColliderDescBuilder | null)[] = [];
  let totalVolume = 0;

  const EPS = 1e-4;

  frags.forEach((f, i) => {
    const hx = f.halfExtents.x;
    const hy = f.halfExtents.y;
    const hz = f.halfExtents.z;
    const size = { x: hx * 2, y: hy * 2, z: hz * 2 } satisfies Vec3;
    const volume = size.x * size.y * size.z;
    const isSupport = f.isSupport || (f.worldPosition.y - hy <= EPS && false);
    const mass = isSupport ? 0 : volume; // scale later for non-supports
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
      if (n.mass === 0) {
        // Node is support, so leave mass=0
        continue;
      }
      n.mass = n.volume > 0 ? n.volume * scale : 0;
    }
  } else {
    for (const n of nodes) n.mass = 0;
  }

  const legacyRawBonds = computeBondsFromFragments(frags);
  const legacyNormBonds = normalizeFractureAreasByAxis(legacyRawBonds, {
    span,
    height,
    thickness,
  });
  const legacyBonds: ScenarioDesc["bonds"] = legacyNormBonds.map((b) => ({
    node0: (b as { a: number }).a,
    node1: (b as { b: number }).b,
    centroid: b.centroid,
    normal: b.normal,
    area: Math.max(b.area, 1e-8),
  }));

  return {
    nodes,
    bonds: legacyBonds,
    parameters: {
      fragmentSizes,
      fragmentGeometries,
      span,
      height,
      thickness,
      fragmentCount,
    },
    colliderDescForNode,
  } satisfies ScenarioDesc;
}
