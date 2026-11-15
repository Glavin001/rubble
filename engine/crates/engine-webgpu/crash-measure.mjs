// The crush of a soft-body car: its crashed nodes against the same car at rest, in the
// rest car's own frame. Shared by crash-orbit.mjs (`measure`, from saved snapshots) and
// brick-ring-drive.html's inspect mode (live, from the node readback).
//
//   measureCrush(rest, crash) → { frontCrush, lengthLoss, bodyMaxDisp, … } (metres)
//
// `rest` and `crash` are the node positions ([x, y, z] world, the same node order). The
// rest car is taken to face +z (the drive's cars spawn heading +z). The crashed nodes are
// rigidly aligned (Kabsch, Horn's quaternion form) onto the rest nodes on the car's rear
// two fifths, iterated with the nodes that moved more than 3 cm left out; each node's
// displacement is then read in the rest car's frame: x right, y up, z forward.

// Rotation R and translation t with R·P + t ≈ Q (rows with w[i] set).
export function kabsch(P, Q, w) {
  let sw = 0; const cp = [0, 0, 0], cq = [0, 0, 0];
  for (let i = 0; i < P.length; i++) if (w[i]) { sw += 1; for (let j = 0; j < 3; j++) { cp[j] += P[i][j]; cq[j] += Q[i][j]; } }
  for (let j = 0; j < 3; j++) { cp[j] /= sw; cq[j] /= sw; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < P.length; i++) if (w[i]) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += (P[i][a] - cp[a]) * (Q[i][b] - cq[b]);
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz], [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]];
  // The largest eigenvector by power iteration on N + cI.
  let q = [1, 0, 0, 0];
  for (let it = 0; it < 500; it++) {
    const n = [0, 1, 2, 3].map((r) => N[r].reduce((s, v, c) => s + v * q[c], 0) + 1e3 * q[r]);
    const l = Math.hypot(...n); q = n.map((v) => v / l);
  }
  const [w0, x, y, z] = q;
  const R = [[w0 * w0 + x * x - y * y - z * z, 2 * (x * y - w0 * z), 2 * (x * z + w0 * y)], [2 * (x * y + w0 * z), w0 * w0 - x * x + y * y - z * z, 2 * (y * z - w0 * x)], [2 * (x * z - w0 * y), 2 * (y * z + w0 * x), w0 * w0 - x * x - y * y + z * z]];
  const t = [0, 1, 2].map((a) => cq[a] - (R[a][0] * cp[0] + R[a][1] * cp[1] + R[a][2] * cp[2]));
  return (p) => [0, 1, 2].map((a) => R[a][0] * p[0] + R[a][1] * p[1] + R[a][2] * p[2] + t[a]);
}

export function measureCrush(R, C) {
  const n = R.length;
  if (C.length !== n) throw new Error('positions of different cars');
  // The rest car's frame: z its +z heading (forward), centred on its bounding box.
  let zlo = Infinity, zhi = -Infinity, xlo = Infinity, xhi = -Infinity;
  for (const p of R) { zlo = Math.min(zlo, p[2]); zhi = Math.max(zhi, p[2]); xlo = Math.min(xlo, p[0]); xhi = Math.max(xhi, p[0]); }
  const len = zhi - zlo, cx = 0.5 * (xlo + xhi);
  let w = R.map((p) => (p[2] - zlo < 0.4 * len ? 1 : 0));
  let align = null;
  for (let it = 0; it < 6; it++) {
    align = kabsch(C, R, w);
    const res = C.map((p, i) => { const a = align(p); return Math.hypot(a[0] - R[i][0], a[1] - R[i][1], a[2] - R[i][2]); });
    const nw = R.map((p, i) => (p[2] - zlo < 0.4 * len && res[i] < 0.03 ? 1 : 0));
    if (nw.reduce((a, b) => a + b, 0) < 12) break;
    w = nw;
  }
  const A = C.map(align);
  const used = w.reduce((a, b) => a + b, 0);
  const rms = Math.sqrt(A.reduce((s, a, i) => s + (w[i] ? (a[0] - R[i][0]) ** 2 + (a[1] - R[i][1]) ** 2 + (a[2] - R[i][2]) ** 2 : 0), 0) / used);
  const disp = A.map((a, i) => [a[0] - R[i][0], a[1] - R[i][1], a[2] - R[i][2]]);
  const mag = disp.map((d) => Math.hypot(...d));
  // Front crush: the frontmost rest nodes (within 10 cm of the nose), how far back
  // they went (mean and the most), and the car's overall length lost at the front.
  const front = R.map((p, i) => i).filter((i) => zhi - R[i][2] < 0.1);
  const back = front.map((i) => -disp[i][2]);
  let mi = 0; for (let i = 1; i < n; i++) if (mag[i] > mag[mi]) mi = i;
  const where = (p) => {
    const f = (p[2] - zlo) / len;
    const along = f > 0.8 ? 'front' : f > 0.6 ? 'front quarter' : f > 0.4 ? 'middle' : f > 0.2 ? 'rear quarter' : 'rear';
    const side = Math.abs(p[0] - cx) < 0.25 ? 'centre' : p[0] > cx ? '+x side' : '−x side';
    return `${along}, ${side}`;
  };
  const over = (d) => mag.filter((m) => m > d).length;
  // Nodes more than 1.5 m from their place: torn off and left behind (in the wall or
  // the debris), not crush; the body's largest displacement leaves them out.
  const DETACHED = 1.5;
  const detached = mag.map((m, i) => i).filter((i) => mag[i] > DETACHED).map((i) => ({ node: i, disp: +mag[i].toFixed(3), dir: disp[i].map((x) => +x.toFixed(3)), where: where(R[i]) }));
  let bi = -1; for (let i = 0; i < n; i++) if (mag[i] <= DETACHED && (bi < 0 || mag[i] > mag[bi])) bi = i;
  // The length at the front: the body's frontmost node, torn-off nodes left out (a piece
  // thrown ahead of the car is not the car's length).
  let azhi = -Infinity; for (let i = 0; i < n; i++) if (mag[i] <= DETACHED) azhi = Math.max(azhi, A[i][2]);
  // Displacement by fifth of the car's length (rest position): mean and max.
  const bands = [0, 1, 2, 3, 4].map((b) => { const ids = R.map((p, i) => i).filter((i) => Math.min(4, Math.floor(5 * (R[i][2] - zlo) / len)) === b); return { from: +(b / 5).toFixed(1), nodes: ids.length, mean: +(ids.reduce((s, i) => s + mag[i], 0) / ids.length).toFixed(4), max: +Math.max(...ids.map((i) => mag[i])).toFixed(4) }; });
  return {
    restLength: +len.toFixed(4),
    alignNodes: used, alignRms: +rms.toFixed(4),
    frontNodes: front.length,
    frontCrush: +(back.reduce((a, b) => a + b, 0) / back.length).toFixed(4),
    frontCrushMax: +Math.max(...back).toFixed(4),
    lengthLoss: +(zhi - azhi).toFixed(4),
    maxDisp: +mag[mi].toFixed(4), maxNode: mi, maxAt: R[mi].map((x) => +x.toFixed(3)), maxDir: disp[mi].map((x) => +x.toFixed(3)), maxWhere: where(R[mi]),
    bodyMaxDisp: +mag[bi].toFixed(4), bodyMaxNode: bi, bodyMaxDir: disp[bi].map((x) => +x.toFixed(3)), bodyMaxWhere: where(R[bi]), detached,
    nodesMoved: { over5cm: over(0.05), over10cm: over(0.1), over20cm: over(0.2), of: n },
    bands,
  };
}
