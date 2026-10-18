import * as THREE from "three";
import type { ScenarioDesc, Vec3 } from "@/lib/stress/core/types";

const EPS = 1e-8;

export type BeamBridgeOptions = {
  // Deck geometry
  span?: number; // X length of deck (m)
  deckWidth?: number; // Z width of deck (m)
  deckThickness?: number; // Y thickness of deck (m)
  spanSegments?: number; // number of blocks along X
  widthSegments?: number; // number of blocks along Z
  thicknessLayers?: number; // number of blocks through deck thickness (Y)
  deckMass?: number; // total mass distributed among deck blocks

  // Supports
  pierHeight?: number; // distance from ground to deck bottom (m)
  supportsPerSide?: number; // number of vertical posts at each end across Z
  supportWidthSegments?: number; // how many deck width segments a post occupies (>=1)
  supportDepthSegments?: number; // how many deck span columns deep each post extends (>=1)
  footingThickness?: number; // thin foundation plate thickness (m)

  // Bonding
  areaScale?: number;
  addDiagonals?: boolean;
  diagScale?: number;
  normalizeAreas?: boolean;
  bondsX?: boolean;
  bondsY?: boolean;
  bondsZ?: boolean;
};

function v(x: number, y: number, z: number): Vec3 {
  return { x, y, z };
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return v(a.x - b.x, a.y - b.y, a.z - b.z);
}
function nrm(p: Vec3): Vec3 {
  const L = Math.hypot(p.x, p.y, p.z);
  if (L <= EPS) return v(0, 0, 0);
  return v(p.x / L, p.y / L, p.z / L);
}

export function buildBeamBridgeScenario({
  // Deck
  span = 18.0,
  deckWidth = 5.0,
  deckThickness = 0.6,
  spanSegments = 30,
  widthSegments = 10,
  thicknessLayers = 2,
  deckMass = 60_000,
  // Supports
  pierHeight = 2.8,
  supportsPerSide = 4,
  supportWidthSegments = 2,
  supportDepthSegments = 2,
  footingThickness = 0.12,
  // Bonds
  areaScale = 0.05,
  addDiagonals = true,
  diagScale = 0.6,
  normalizeAreas = true,
  bondsX = true,
  bondsY = true,
  bondsZ = true,
}: BeamBridgeOptions = {}): ScenarioDesc {
  const segX = Math.max(1, Math.floor(spanSegments));
  const segY = Math.max(1, Math.floor(thicknessLayers));
  const segZ = Math.max(1, Math.floor(widthSegments));

  // Deck cell sizes
  const cellX = span / segX;
  const cellY = deckThickness / segY;
  const cellZ = deckWidth / segZ;

  // Choose an integer number of post layers and snap deck to that height to avoid initial overlaps
  const postLayers = Math.max(1, Math.ceil(pierHeight / cellY));
  const deckBottomY = postLayers * cellY; // snapped to whole cell layers above ground
  // World placement: footings rest on ground (top at y=0), deck sits above posts
  const deckOrigin = v(
    -span * 0.5 + 0.5 * cellX,
    deckBottomY + 0.5 * cellY,
    -deckWidth * 0.5 + 0.5 * cellZ,
  );

  // Index grids
  const gridDeck: number[][][] = Array.from({ length: segX }, () =>
    Array.from({ length: segY }, () => Array.from({ length: segZ }, () => -1)),
  );

  const nodes: ScenarioDesc["nodes"] = [];
  const fragmentSizes: Array<{ x: number; y: number; z: number }> = [];
  const fragmentGeometries: THREE.BufferGeometry[] = [];
  const gridCoordinates: Array<{ ix: number; iy: number; iz: number }> = [];

  const registerNode = (nodeIndex: number, size: Vec3) => {
    const geom = new THREE.BoxGeometry(
      Math.max(size.x, EPS),
      Math.max(size.y, EPS),
      Math.max(size.z, EPS),
    );
    fragmentGeometries[nodeIndex] = geom;
  };

  // Build deck nodes
  const deckCellVolume = cellX * cellY * cellZ;
  let deckTotalVolume = 0;
  for (let ix = 0; ix < segX; ix += 1) {
    for (let iy = 0; iy < segY; iy += 1) {
      for (let iz = 0; iz < segZ; iz += 1) {
        const p = v(
          deckOrigin.x + ix * cellX,
          deckOrigin.y + iy * cellY,
          deckOrigin.z + iz * cellZ,
        );
        const idx = nodes.length;
        nodes.push({
          centroid: p,
          mass: deckCellVolume,
          volume: deckCellVolume,
        });
        fragmentSizes.push({ x: cellX, y: cellY, z: cellZ });
        gridDeck[ix][iy][iz] = idx;
        gridCoordinates[idx] = { ix, iy, iz };
        deckTotalVolume += deckCellVolume;
        registerNode(idx, { x: cellX, y: cellY, z: cellZ });
      }
    }
  }

  // Mass scale so that sum(deck.mass) == deckMass
  const massScale = deckTotalVolume > 0 ? deckMass / deckTotalVolume : 0;
  if (massScale !== 1) {
    for (const n of nodes) if (n.volume > 0) n.mass = n.volume * massScale;
  }

  // Utility for bonds
  const bonds: ScenarioDesc["bonds"] = [];
  const areaX = cellY * cellZ * areaScale;
  const areaY = cellX * cellZ * areaScale;
  const areaZ = cellX * cellY * areaScale;
  const addBond = (a: number, b: number, area: number) => {
    if (a < 0 || b < 0) return;
    const na = nodes[a];
    const nb = nodes[b];
    const c = v(
      (na.centroid.x + nb.centroid.x) * 0.5,
      (na.centroid.y + nb.centroid.y) * 0.5,
      (na.centroid.z + nb.centroid.z) * 0.5,
    );
    const n = nrm(sub(nb.centroid, na.centroid));
    bonds.push({
      node0: a,
      node1: b,
      centroid: c,
      normal: n,
      area: Math.max(area, EPS),
    });
  };

  // Deck connectivity (face neighbors + optional plane diagonals)
  for (let ix = 0; ix < segX; ix += 1) {
    for (let iy = 0; iy < segY; iy += 1) {
      for (let iz = 0; iz < segZ; iz += 1) {
        const cur = gridDeck[ix][iy][iz];
        if (cur < 0) continue;
        if (bondsX && ix + 1 < segX)
          addBond(cur, gridDeck[ix + 1][iy][iz], areaX);
        if (bondsY && iy + 1 < segY)
          addBond(cur, gridDeck[ix][iy + 1][iz], areaY);
        if (bondsZ && iz + 1 < segZ)
          addBond(cur, gridDeck[ix][iy][iz + 1], areaZ);
        if (addDiagonals) {
          if (bondsX && bondsZ && ix + 1 < segX && iz + 1 < segZ)
            addBond(
              cur,
              gridDeck[ix + 1][iy][iz + 1],
              0.5 * (areaX + areaZ) * diagScale,
            );
          if (bondsX && bondsY && ix + 1 < segX && iy + 1 < segY)
            addBond(
              cur,
              gridDeck[ix + 1][iy + 1][iz],
              0.5 * (areaX + areaY) * diagScale,
            );
          if (bondsY && bondsZ && iy + 1 < segY && iz + 1 < segZ)
            addBond(
              cur,
              gridDeck[ix][iy + 1][iz + 1],
              0.5 * (areaY + areaZ) * diagScale,
            );
        }
      }
    }
  }

  // Build destructible posts under first and last span columns
  const postXCols = [0, segX - 1];
  const postTopYLayer = 0; // deck bottom layer index to connect to posts
  const postTopY = deckOrigin.y - 0.5 * cellY; // top of post touches deck bottom

  // Pick evenly spaced Z slots for posts
  const clamp = (v2: number, lo: number, hi: number) =>
    Math.min(hi, Math.max(lo, v2));
  const postSpan = Math.max(1, supportsPerSide);
  const slots: number[] = [];
  for (let i = 0; i < postSpan; i += 1) {
    // Distribute across [0, segZ)
    const t = postSpan === 1 ? 0.5 : i / (postSpan - 1);
    const zIndex = clamp(Math.round(t * (segZ - 1)), 0, segZ - 1);
    slots.push(zIndex);
  }

  // For each end and slot, create a vertical stack of post nodes expanded across Z and into the span (X)
  const uniq = (arr: number[]) => Array.from(new Set(arr));
  for (const ixEdge of postXCols) {
    const ixCover = uniq(
      Array.from({ length: supportDepthSegments }, (_, k) =>
        clamp(ixEdge + (ixEdge === 0 ? k : -k), 0, segX - 1),
      ),
    );
    const ixCoverSet = new Set(ixCover);
    for (const baseZ of slots) {
      const coverZ = uniq(
        Array.from({ length: supportWidthSegments }, (_, k) =>
          clamp(
            baseZ + k - Math.floor((supportWidthSegments - 1) / 2),
            0,
            segZ - 1,
          ),
        ),
      );
      const coverZSet = new Set(coverZ);
      const postMap = new Map<string, number>();
      const key = (ixp: number, py: number, iz: number) => `${ixp}|${py}|${iz}`;

      // Create stacks
      for (const iz of coverZ) {
        for (const ixp of ixCover) {
          for (let py = 0; py < postLayers; py += 1) {
            const yCenter = postTopY - py * cellY - 0.5 * cellY;
            const idx = nodes.length;
            const p = v(
              deckOrigin.x + ixp * cellX,
              yCenter,
              deckOrigin.z + iz * cellZ,
            );
            const volume = cellX * cellY * cellZ;
            nodes.push({
              centroid: p,
              mass: volume * massScale,
              volume: volume,
            });
            fragmentSizes.push({ x: cellX, y: cellY, z: cellZ });
            const gy = -1 - py;
            gridCoordinates[idx] = { ix: ixp, iy: gy, iz };
            postMap.set(key(ixp, py, iz), idx);
            registerNode(idx, { x: cellX, y: cellY, z: cellZ });

            if (py > 0) {
              const prevIdx = postMap.get(key(ixp, py - 1, iz));
              if (prevIdx != null) addBond(prevIdx, idx, areaY);
            } else {
              const deckIndex = gridDeck[ixp][postTopYLayer][iz];
              addBond(idx, deckIndex, areaY);
            }
          }

          // Footing under this column
          const footCenterY =
            postTopY - postLayers * cellY - 0.5 * footingThickness;
          const fIdx = nodes.length;
          const fPos = v(
            deckOrigin.x + ixp * cellX,
            footCenterY,
            deckOrigin.z + iz * cellZ,
          );
          nodes.push({ centroid: fPos, mass: 0, volume: 0 });
          fragmentSizes.push({
            x: cellX,
            y: Math.max(footingThickness, EPS),
            z: cellZ,
          });
          gridCoordinates[fIdx] = { ix: ixp, iy: -1 - postLayers, iz };
          const lowestPostIdx = postMap.get(key(ixp, postLayers - 1, iz));
          if (lowestPostIdx != null) addBond(fIdx, lowestPostIdx, areaY);
          registerNode(fIdx, {
            x: cellX,
            y: Math.max(footingThickness, EPS),
            z: cellZ,
          });
        }
      }

      // Lateral bonds within the post cluster (X/Z at same height)
      for (const iz of coverZ) {
        for (const ixp of ixCover) {
          for (let py = 0; py < postLayers; py += 1) {
            const cur = postMap.get(key(ixp, py, iz));
            if (cur == null) continue;
            const nx = ixEdge === 0 ? ixp + 1 : ixp - 1;
            if (ixCoverSet.has(nx)) {
              const nb = postMap.get(key(nx, py, iz));
              if (nb != null) addBond(cur, nb, areaX);
            }
            const nz = iz + 1;
            if (coverZSet.has(nz)) {
              const nbz = postMap.get(key(ixp, py, nz));
              if (nbz != null) addBond(cur, nbz, areaZ);
            }
          }
        }
      }
    }
  }

  // Optional per-axis area normalization
  if (normalizeAreas && bonds.length) {
    const size = {
      x: span,
      y: deckThickness + pierHeight + footingThickness,
      z: deckWidth,
    };
    const target = {
      x: size.y * size.z,
      y: size.x * size.z,
      z: size.x * size.y,
    };
    const sum = { x: 0, y: 0, z: 0 };
    const pick = (n: Vec3): "x" | "y" | "z" => {
      const ax = Math.abs(n.x),
        ay = Math.abs(n.y),
        az = Math.abs(n.z);
      return ax >= ay && ax >= az ? "x" : ay >= az ? "y" : "z";
    };
    for (const b of bonds) sum[pick(b.normal)] += b.area;
    const scale = {
      x: sum.x > 0 ? target.x / sum.x : 1,
      y: sum.y > 0 ? target.y / sum.y : 1,
      z: sum.z > 0 ? target.z / sum.z : 1,
    } as const;
    for (const b of bonds) b.area *= scale[pick(b.normal)];
  }

  return {
    nodes,
    bonds,
    gridCoordinates,
    spacing: v(cellX, cellY, cellZ),
    parameters: {
      span,
      deckWidth,
      deckThickness,
      deckMass,
      pierHeight,
      supportsPerSide,
      supportWidthSegments,
      supportDepthSegments,
      footingThickness,
      areaScale,
      addDiagonals,
      diagScale,
      fragmentSizes,
      fragmentGeometries,
    },
  } satisfies ScenarioDesc;
}
