import { DestructibleMesh, FractureOptions } from "@dgreenheck/three-pinata";
import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type {
  ColliderDescBuilder,
  ScenarioDesc,
  Vec3,
} from "@/lib/stress/core/types";

type FragmentInfo = {
  worldPosition: THREE.Vector3;
  halfExtents: THREE.Vector3;
  geometry: THREE.BufferGeometry;
  isSupport: boolean;
};

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

  const globalTol =
    0.12 *
    Math.min(
      ...fragments.map((f) => Math.min(f.halfExtents.x, f.halfExtents.z)),
    );

  for (let i = 0; i < fragments.length; i += 1) {
    for (let j = i + 1; j < fragments.length; j += 1) {
      const A = fragments[i];
      const B = fragments[j];

      // Skip bonds between two supports; allow support <-> fragment
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

function _normalizeFractureAreasByAxis(
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
  const sum = { x: 0, y: 0, z: 0 } as { x: number; y: number; z: number };
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

function uniformizeBondAreasByAxis(
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
  const pick = (n: Vec3): "x" | "y" | "z" => {
    const ax = Math.abs(n.x),
      ay = Math.abs(n.y),
      az = Math.abs(n.z);
    return ax >= ay && ax >= az ? "x" : ay >= az ? "y" : "z";
  };
  const counts = { x: 0, y: 0, z: 0 } as { x: number; y: number; z: number };
  for (const b of list) counts[pick(b.normal)] += 1;
  const areaPerAxis = {
    x: counts.x > 0 ? Math.max(1e-8, target.x / counts.x) : 0,
    y: counts.y > 0 ? Math.max(1e-8, target.y / counts.y) : 0,
    z: counts.z > 0 ? Math.max(1e-8, target.z / counts.z) : 0,
  } as const;
  return list.map((b) => ({
    ...b,
    area: areaPerAxis[pick(b.normal)] || Math.max(1e-8, b.area),
  }));
}

async function _loadMergedGeometryFromGlb(
  url: string,
): Promise<THREE.BufferGeometry | null> {
  const loader = new GLTFLoader();
  try {
    const draco = new DRACOLoader();
    // Use Google's hosted decoders by default; swap to '/draco/' if you host locally
    draco.setDecoderPath("https://www.gstatic.com/draco/v1/decoders/");
    loader.setDRACOLoader(draco);
  } catch {}
  const gltf = await loader.loadAsync(url);
  const geoms: THREE.BufferGeometry[] = [];
  try {
    gltf.scene.updateMatrixWorld(true);
  } catch {}
  gltf.scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    const geometry = mesh?.geometry as THREE.BufferGeometry | undefined;
    if (!geometry) return;
    const cloned = geometry.clone();
    const m = mesh.matrixWorld?.clone?.() ?? new THREE.Matrix4();
    try {
      cloned.applyMatrix4(m);
    } catch {}
    geoms.push(cloned);
  });
  if (!geoms.length) return null;
  const merged = mergeGeometries(geoms, false);
  return merged ?? null;
}

export async function buildFracturedGlbScenario({
  url = "/models/lion.glb",
  // url = '/models/building.glb',
  // url = '/models/atlanta_corperate_office_building.glb',
  // url = '/models/big_soviet_panel_house_lowpoly.glb',
  fragmentCount = 120,
  // fragmentCount = 300,
  objectMass = 10_000,
}: {
  url?: string;
  fragmentCount?: number;
  objectMass?: number;
} = {}): Promise<ScenarioDesc> {
  // 1) Load GLB and merge meshes to a single geometry in world space
  const merged = await (async () => {
    const loader = new GLTFLoader();
    try {
      const draco = new DRACOLoader();
      draco.setDecoderPath("https://www.gstatic.com/draco/v1/decoders/");
      loader.setDRACOLoader(draco);
    } catch {}
    const gltf = await loader.loadAsync(url);
    try {
      gltf.scene.updateMatrixWorld(true);
    } catch {}
    // Pick the largest Mesh geometry with a valid position attribute
    let bestGeom: THREE.BufferGeometry | null = null;
    let bestVolume = -Infinity;
    gltf.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      const geometry = mesh?.geometry as THREE.BufferGeometry | undefined;
      if (!geometry) return;
      const hasPos = !!geometry.getAttribute("position");
      if (!hasPos) return;
      const cloned = geometry.clone();
      const m = mesh.matrixWorld?.clone?.() ?? new THREE.Matrix4();
      try {
        cloned.applyMatrix4(m);
      } catch {}
      try {
        cloned.computeBoundingBox();
      } catch {}
      const bb = cloned.boundingBox as THREE.Box3 | null;
      if (!bb) return;
      const s = new THREE.Vector3();
      bb.getSize(s);
      const vol = Math.max(0, s.x) * Math.max(0, s.y) * Math.max(0, s.z);
      if (vol > bestVolume) {
        bestGeom = cloned;
        bestVolume = vol;
      }
    });
    if (bestGeom) return bestGeom;
    // Fallback: merge everything
    const geoms: THREE.BufferGeometry[] = [];
    gltf.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      const geometry = mesh?.geometry as THREE.BufferGeometry | undefined;
      if (!geometry) return;
      const cloned = geometry.clone();
      const m = mesh.matrixWorld?.clone?.() ?? new THREE.Matrix4();
      try {
        cloned.applyMatrix4(m);
      } catch {}
      geoms.push(cloned);
    });
    if (!geoms.length) return null;
    const mergedGeom = mergeGeometries(geoms, false);
    return mergedGeom ?? null;
  })();
  if (!merged) {
    // Fallback: tiny cube at origin to avoid crashing callers
    const g = new THREE.BoxGeometry(1, 1, 1);
    const center = new THREE.Vector3(0, 0.5, 0);
    return {
      nodes: [
        {
          centroid: { x: center.x, y: center.y, z: center.z },
          mass: objectMass,
          volume: 1,
        },
      ],
      bonds: [],
      parameters: {
        fragmentGeometries: [g],
        fragmentSizes: [{ x: 1, y: 1, z: 1 }],
        source: "fallback",
      },
      colliderDescForNode: [() => RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5)],
    } satisfies ScenarioDesc;
  }

  // Ensure geometry uses plain BufferAttributes (not interleaved) and has normals
  function ensurePlainAttributes(g: THREE.BufferGeometry) {
    const names: Array<keyof typeof g.attributes> = [
      "position",
      "normal",
      "uv",
    ] as unknown as Array<keyof typeof g.attributes>;
    for (const n of names) {
      const attr = g.getAttribute(n as unknown as string) as unknown as
        | {
            count?: number;
            itemSize?: number;
            getX?: (i: number) => number;
            getY?: (i: number) => number;
            getZ?: (i: number) => number;
            array?: unknown;
          }
        | undefined;
      if (!attr) continue;
      const hasArray = (attr as { array?: unknown }).array != null;
      if (!hasArray) {
        const count = (attr.count as number) ?? 0;
        const itemSize = (attr.itemSize as number) ?? 3;
        const data = new Float32Array(Math.max(0, count * itemSize));
        for (let i = 0; i < count; i += 1) {
          if (itemSize >= 1)
            data[i * itemSize + 0] = (attr.getX?.(i) as number) ?? 0;
          if (itemSize >= 2)
            data[i * itemSize + 1] = (attr.getY?.(i) as number) ?? 0;
          if (itemSize >= 3)
            data[i * itemSize + 2] = (attr.getZ?.(i) as number) ?? 0;
        }
        g.setAttribute(
          n as unknown as string,
          new THREE.BufferAttribute(data, itemSize),
        );
      }
    }
    // Recompute normals if absent
    if (!g.getAttribute("normal")) {
      try {
        g.computeVertexNormals();
      } catch {}
    }
  }
  // Convert to non-indexed to avoid interleaved quirks in downstream libs
  const prepared = merged.index ? merged.toNonIndexed() : merged;
  ensurePlainAttributes(prepared);

  // Measure model extents
  merged.computeBoundingBox();
  const bbox = merged.boundingBox as THREE.Box3;
  const size = new THREE.Vector3();
  bbox.getSize(size);

  // 2) Fracture the merged geometry
  const opts = new FractureOptions({
    fractureMethod: "voronoi",
    fragmentCount: fragmentCount,
    voronoiOptions: { mode: "3D" },
  });
  // three-pinata expects BufferGeometry with a non-null Float32Array position
  // Some pipelines produce interleaved attributes; ensure plain arrays on a clone
  const fractureInput = merged.clone();
  (function ensureAttributes(g: THREE.BufferGeometry) {
    const pos = g.getAttribute("position") as THREE.BufferAttribute | null;
    if (!pos) {
      // If no position, nothing to fracture
    }
    const names: string[] = ["position", "normal", "uv"];
    for (const n of names) {
      const attr = g.getAttribute(n) as THREE.BufferAttribute | null;
      if (!attr) continue;
      if (!(attr.array instanceof Float32Array)) {
        const count = attr.count ?? 0;
        const itemSize = attr.itemSize ?? 3;
        const data = new Float32Array(Math.max(0, count * itemSize));
        for (let i = 0; i < count; i += 1) {
          if (itemSize >= 1) data[i * itemSize + 0] = attr.getX(i) ?? 0;
          if (itemSize >= 2) data[i * itemSize + 1] = attr.getY(i) ?? 0;
          if (itemSize >= 3) data[i * itemSize + 2] = attr.getZ(i) ?? 0;
        }
        g.setAttribute(n, new THREE.BufferAttribute(data, itemSize));
      }
    }
    try {
      if (!g.getAttribute("normal")) g.computeVertexNormals();
    } catch {}
  })(fractureInput);
  const destructibleMesh = new DestructibleMesh(fractureInput);
  const pieceMeshes = destructibleMesh.fracture(opts);
  console.log("Fractured pieces", pieceMeshes.length);

  // Placement above foundation
  const foundationHeight = Math.min(0.08, size.y * 0.06);
  const groundClearance = Math.max(0.001, foundationHeight * 0.05);
  const foundationTop = groundClearance + foundationHeight;
  // Place fractured object so its lowest original vertex rests exactly on the foundation top
  const originalMinY = bbox.min.y;
  const contactEpsilon = 0.0005; // tiny lift to avoid z-fighting
  const modelCenter = new THREE.Vector3(
    0,
    foundationTop - originalMinY + contactEpsilon,
    0,
  );

  // 3) Build fragments list (recenter each geom to its own COM, compute world placement)
  const fragments: FragmentInfo[] = pieceMeshes.map((m) => {
    const g = m.geometry;
    // Ensure fractured piece has plain attributes for later convexHull
    (function ensureAttrs(geom: THREE.BufferGeometry) {
      const pos = geom.getAttribute("position") as THREE.BufferAttribute | null;
      if (!pos || !(pos.array instanceof Float32Array)) {
        const count = pos?.count ?? 0;
        const itemSize = pos?.itemSize ?? 3;
        const data = new Float32Array(Math.max(0, count * itemSize));
        for (let i = 0; i < count; i += 1) {
          if (itemSize >= 1) data[i * itemSize + 0] = pos?.getX(i) ?? 0;
          if (itemSize >= 2) data[i * itemSize + 1] = pos?.getY(i) ?? 0;
          if (itemSize >= 3) data[i * itemSize + 2] = pos?.getZ(i) ?? 0;
        }
        geom.setAttribute(
          "position",
          new THREE.BufferAttribute(data, itemSize),
        );
      }
      try {
        if (!geom.getAttribute("normal")) geom.computeVertexNormals();
      } catch {}
    })(g);
    g.computeBoundingBox();
    const gb = g.boundingBox as THREE.Box3;
    const localCenter = new THREE.Vector3();
    gb.getCenter(localCenter);
    g.translate(-localCenter.x, -localCenter.y, -localCenter.z);
    const gsize = new THREE.Vector3();
    gb.getSize(gsize);
    const worldPosition = new THREE.Vector3(
      modelCenter.x + m.position.x + localCenter.x,
      modelCenter.y + m.position.y + localCenter.y,
      modelCenter.z + m.position.z + localCenter.z,
    );
    return {
      worldPosition,
      halfExtents: new THREE.Vector3(
        Math.max(0.05, gsize.x * 0.5),
        Math.max(0.05, gsize.y * 0.5),
        Math.max(0.05, gsize.z * 0.5),
      ),
      geometry: g,
      isSupport: false,
    };
  });

  // 4) Add a single foundation support plate sized to the GLB footprint
  const foundationGeom = new THREE.BoxGeometry(
    size.x,
    foundationHeight,
    size.z,
  );
  const foundationWorldPos = new THREE.Vector3(
    0,
    groundClearance + foundationHeight * 0.5,
    0,
  );
  fragments.push({
    worldPosition: foundationWorldPos,
    halfExtents: new THREE.Vector3(
      size.x * 0.5,
      foundationHeight * 0.5,
      size.z * 0.5,
    ),
    geometry: foundationGeom,
    isSupport: true,
  });

  // 5) Compute bonds and normalize by axes
  const rawBonds = computeBondsFromFragments(fragments);
  // Ensure at least one (and preferably all touching) lowest fragments are bonded to the foundation
  const foundationIdx = fragments.length - 1;
  const lowestBottomY = Math.min(
    ...fragments
      .slice(0, foundationIdx)
      .map((f) => f.worldPosition.y - f.halfExtents.y),
  );
  const touchThreshold = Math.max(0.001, foundationHeight * 0.1);
  const desiredContactY = groundClearance + foundationHeight; // top of foundation
  const candidates: number[] = [];
  for (let i = 0; i < foundationIdx; i += 1) {
    const f = fragments[i];
    const bottomY = f.worldPosition.y - f.halfExtents.y;
    if (Math.abs(bottomY - desiredContactY) <= touchThreshold) {
      candidates.push(i);
    }
  }
  if (candidates.length === 0) {
    // Fallback: pick the single lowest fragment
    for (let i = 0; i < foundationIdx; i += 1) {
      const f = fragments[i];
      const bottomY = f.worldPosition.y - f.halfExtents.y;
      if (bottomY === lowestBottomY) {
        candidates.push(i);
        break;
      }
    }
  }
  for (const i of candidates) {
    const f = fragments[i];
    const footprintX = Math.max(1e-4, f.halfExtents.x * 2);
    const footprintZ = Math.max(1e-4, f.halfExtents.z * 2);
    const approxArea = Math.max(1e-8, footprintX * footprintZ * 0.25);
    rawBonds.push({
      a: foundationIdx,
      b: i,
      centroid: {
        x: f.worldPosition.x,
        y: desiredContactY,
        z: f.worldPosition.z,
      },
      normal: { x: 0, y: 1, z: 0 },
      area: approxArea,
    });
  }
  // Replace area magnitudes with uniform per-axis values so material behaves uniformly
  const normBonds = uniformizeBondAreasByAxis(rawBonds, {
    span: size.x,
    height: size.y,
    thickness: size.z,
  });

  // 6) Emit ScenarioDesc: nodes, mass scaling with objectMass, bonds, colliders, and parameters
  const nodes: ScenarioDesc["nodes"] = [];
  const fragmentSizes: Vec3[] = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];
  const colliderDescForNode: (ColliderDescBuilder | null)[] = [];
  let totalVolume = 0;

  fragments.forEach((f, i) => {
    const hx = f.halfExtents.x;
    const hy = f.halfExtents.y;
    const hz = f.halfExtents.z;
    const sizeVec = { x: hx * 2, y: hy * 2, z: hz * 2 } as Vec3;
    const volume = sizeVec.x * sizeVec.y * sizeVec.z;
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
    fragmentSizes.push(sizeVec);
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

  const massScale = totalVolume > 0 ? objectMass / totalVolume : 0;
  if (massScale > 0) {
    for (let i = 0; i < nodes.length; i += 1) {
      if (fragments[i]?.isSupport) {
        nodes[i].mass = 0;
        continue;
      }
      nodes[i].mass = nodes[i].volume > 0 ? nodes[i].volume * massScale : 0;
    }
  } else {
    for (let i = 0; i < nodes.length; i += 1)
      nodes[i].mass = fragments[i]?.isSupport ? 0 : 0;
  }

  const legacyBonds: ScenarioDesc["bonds"] = normBonds.map((b) => ({
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
      objectMass,
      sourceUrl: url,
    },
    colliderDescForNode,
  } satisfies ScenarioDesc;
}
