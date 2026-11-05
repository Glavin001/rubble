// How much of a screenshot is not the empty studio: the fraction of pixels
// whose colour differs from their row's median by more than `tol` (summed over
// R, G, B; the lab's backdrop and floor vary mostly by row). An empty frame (no
// car, no scenery) is about 0; crash-lab-check fails a shot under its floor.
// Decodes the 8-bit, non-interlaced RGB/RGBA PNGs Chrome's captureScreenshot
// writes (node:zlib only).
import { inflateSync } from 'node:zlib';

export function decodePng(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let at = 8, w = 0, h = 0, ch = 0;
  const idat = [];
  while (at < buf.length) {
    const len = buf.readUInt32BE(at);
    const type = buf.toString('latin1', at + 4, at + 8);
    const data = buf.subarray(at + 8, at + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      if (data[8] !== 8 || data[12] !== 0) throw new Error('PNG: only 8-bit, non-interlaced');
      ch = { 2: 3, 6: 4 }[data[9]];
      if (!ch) throw new Error(`PNG colour type ${data[9]}`);
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    at += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * ch;
  const px = new Uint8Array(w * h * ch);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= ch ? px[y * stride + x - ch] : 0;
      const b = y > 0 ? px[(y - 1) * stride + x] : 0;
      const c = x >= ch && y > 0 ? px[(y - 1) * stride + x - ch] : 0;
      let v = src[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      px[y * stride + x] = v & 255;
    }
  }
  return { w, h, ch, px };
}

export function frameContent(buf, { tol = 45, skipTop = 0.12 } = {}) {
  const { w, h, ch, px } = decodePng(buf);
  let hit = 0, n = 0;
  const med = (a) => a.sort((p, q) => p - q)[a.length >> 1];
  for (let y = Math.floor(h * skipTop); y < h; y += 2) {
    const row = (k) => { const a = []; for (let x = 0; x < w; x += 4) a.push(px[(y * w + x) * ch + k]); return med(a); };
    const m = [row(0), row(1), row(2)];
    for (let x = 0; x < w; x += 2) {
      const o = (y * w + x) * ch;
      n++;
      if (Math.abs(px[o] - m[0]) + Math.abs(px[o + 1] - m[1]) + Math.abs(px[o + 2] - m[2]) > tol) hit++;
    }
  }
  return hit / n;
}
