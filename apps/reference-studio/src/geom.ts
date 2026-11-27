// Display transforms only (image pixels <-> region coordinates <-> screen). Shape maths
// (normalising, measuring) is Rust's, through the server.

import type { ElementValue, Region, Xy } from '../shared/model.ts';

export type Crop = Region['crop'];

export function regionToImage(crop: Crop, w: number, h: number, p: Xy): Xy {
  return [(crop[0] + p[0] * (crop[2] - crop[0])) * w, (crop[1] + p[1] * (crop[3] - crop[1])) * h];
}

export function imageToRegion(crop: Crop, w: number, h: number, p: Xy): Xy {
  const r = (v: number) => Math.round(v * 1e5) / 1e5;
  return [r((p[0] / w - crop[0]) / (crop[2] - crop[0])), r((p[1] / h - crop[1]) / (crop[3] - crop[1]))];
}

export function hasData(v: ElementValue | undefined): boolean {
  return !!v && (!!v.p || (!!v.pts && v.pts.length > 0) || !!v.c);
}

/** Where a new point lengthens a polyline least (pixel distances), for shift-click insertion. */
export function insertIndex(pts: Xy[], p: Xy): number {
  if (pts.length < 2) return pts.length;
  const d = (a: Xy, b: Xy) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  let best = pts.length;
  let cost = Infinity;
  for (let i = 0; i <= pts.length; i++) {
    const c = i === 0 ? d(p, pts[0]) : i === pts.length ? d(pts[i - 1], p) : d(pts[i - 1], p) + d(p, pts[i]) - d(pts[i - 1], pts[i]);
    if (c < cost) {
      cost = c;
      best = i;
    }
  }
  return best;
}

export const SOURCE_VAR: Record<string, string> = { real: 'var(--real)', beamng: 'var(--beamng)', rubble: 'var(--rubble)', other: 'var(--other)' };
export const SOURCE_HEX: Record<string, string> = { real: '#ff5c5c', beamng: '#5b9bff', rubble: '#3ddc84', other: '#b48cff' };
