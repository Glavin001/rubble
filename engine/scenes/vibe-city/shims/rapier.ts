// Stand-in for @dimforge/rapier3d-compat. The copied builders only call
// ColliderDesc.cuboid / ColliderDesc.convexHull to describe each node's
// collider; this records the shape so export.ts can write it to the pack.

export type ShapeRecord =
  | { kind: "cuboid"; halfExtents: [number, number, number] }
  | { kind: "convex_hull"; points: number[] };

export class ColliderDesc {
  constructor(readonly shape: ShapeRecord) {}
  static cuboid(hx: number, hy: number, hz: number): ColliderDesc {
    return new ColliderDesc({ kind: "cuboid", halfExtents: [hx, hy, hz] });
  }
  static convexHull(points: ArrayLike<number>): ColliderDesc {
    return new ColliderDesc({ kind: "convex_hull", points: Array.from(points) });
  }
}

// Type-only uses in core/types.ts.
export type World = unknown;
export type EventQueue = unknown;

export default { ColliderDesc };
