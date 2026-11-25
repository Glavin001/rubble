// A small WebGL2 preview of a generated design: draws the render buffers the vehicle
// generator returns (vehicle-design-worker.mjs 'preview': per part positions, normals,
// 32-bit indices and [material, first, count] groups), lit by a sun and a sky, with the
// art spec's material colours. Drag to orbit, wheel to zoom. No libraries: the lab page
// has none (the designer page's three.js is its own).

// The art spec's ten materials (engine-vehiclegen mesh::MATERIALS order), linear RGB,
// metallic, alpha. Material 0 is the paint, which the design's parameters set.
const MATERIALS = [
  { n: 'paint' }, { n: 'chrome', c: [0.92, 0.92, 0.94], m: 1, r: 0.12 }, { n: 'trim_black', c: [0.018, 0.018, 0.02], r: 0.7 },
  { n: 'rubber', c: [0.016, 0.016, 0.017], r: 0.8 }, { n: 'glass', c: [0.06, 0.075, 0.08], a: 0.35, r: 0.05 },
  { n: 'lamp_lens', c: [0.5, 0.1, 0.08], a: 0.55, r: 0.05 }, { n: 'lamp_emissive', c: [0.9, 0.9, 0.88], m: 0.6, r: 0.2 },
  { n: 'interior_fabric', c: [0.035, 0.035, 0.037], r: 0.9 }, { n: 'interior_plastic', c: [0.03, 0.03, 0.032], r: 0.55 },
  { n: 'metal_bare', c: [0.3, 0.3, 0.31], m: 1, r: 0.55 },
];

const VS = `#version 300 es
in vec3 pos; in vec3 nrm;
uniform mat4 mvp;
out vec3 n; out vec3 p;
void main() { n = nrm; p = pos; gl_Position = mvp * vec4(pos, 1.0); }`;
const FS = `#version 300 es
precision highp float;
in vec3 n; in vec3 p;
uniform vec3 base; uniform float metal; uniform float rough; uniform float alpha; uniform vec3 eye;
out vec4 o;
void main() {
  vec3 N = normalize(n);
  if (!gl_FrontFacing) N = -N;
  vec3 V = normalize(eye - p);
  vec3 L = normalize(vec3(0.45, 0.8, 0.4));
  float d = max(dot(N, L), 0.0);
  float sky = 0.5 + 0.5 * N.y;
  vec3 H = normalize(L + V);
  float sp = pow(max(dot(N, H), 0.0), mix(8.0, 140.0, 1.0 - rough)) * mix(0.25, 1.0, 1.0 - rough);
  float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
  vec3 diffuse = base * (1.0 - metal) * (0.18 + 0.20 * sky + 0.85 * d);
  vec3 spec = mix(vec3(0.04), base, metal) * (sp * 1.6 + fres * 0.35 * sky);
  vec3 c = diffuse + spec;
  c = c / (1.0 + c);                       // soft tone map
  o = vec4(pow(c, vec3(1.0 / 2.2)), alpha); // premultiplied below
  o.rgb *= o.a;
}`;

const mul = (a, b) => { const o = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; } return o; };
const perspective = (fov, asp, n, f) => { const t = 1 / Math.tan(fov / 2); return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]); };
function lookAt(e, c, u) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], nrm = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = nrm(sub(e, c)), x = nrm(cross(u, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1]);
}

/** `createDesignPreview(canvas)`: `{ show(build), resize(), destroy() }` (build: the worker's preview message). */
export function createDesignPreview(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true });
  if (!gl) return null;
  const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const U = Object.fromEntries(['mvp', 'base', 'metal', 'rough', 'alpha', 'eye'].map((k) => [k, gl.getUniformLocation(prog, k)]));
  let parts = [], paint = [0.5, 0.05, 0.05, 0.5, 0.3], radius = 3, center = [0, 0.7, 0];
  let az = 0.75, el = 0.22, dist = 6.5, framed = false, raf = 0;
  const free = () => { for (const p of parts) { gl.deleteVertexArray(p.vao); for (const b of p.bufs) gl.deleteBuffer(b); } parts = []; };
  const draw = () => {
    raf = 0;
    const w = canvas.width, h = canvas.height;
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const eye = [center[0] + dist * Math.cos(el) * Math.sin(az), center[1] + dist * Math.sin(el), center[2] + dist * Math.cos(el) * Math.cos(az)];
    const mvp = mul(perspective(0.55, w / Math.max(h, 1), 0.1, 100), lookAt(eye, center, [0, 1, 0]));
    gl.useProgram(prog);
    gl.uniformMatrix4fv(U.mvp, false, mvp); gl.uniform3fv(U.eye, eye);
    for (const pass of [0, 1]) {
      // opaque first, then the glass and lenses blended over them
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(pass === 0);
      for (const p of parts) {
        gl.bindVertexArray(p.vao);
        for (const [mat, first, count] of p.groups) {
          const M = MATERIALS[mat] ?? MATERIALS[2];
          const a = M.a ?? 1;
          if ((a < 1) !== (pass === 1)) continue;
          const c = mat === 0 ? paint.slice(0, 3) : M.c, metal = mat === 0 ? paint[3] : M.m ?? 0, rough = mat === 0 ? paint[4] : M.r ?? 0.6;
          gl.uniform3fv(U.base, c); gl.uniform1f(U.metal, metal); gl.uniform1f(U.rough, rough); gl.uniform1f(U.alpha, a);
          gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, first * 4);
        }
      }
    }
    gl.depthMask(true);
  };
  const redraw = () => { if (!raf) raf = requestAnimationFrame(draw); };
  const resize = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr)); canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    redraw();
  };
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointermove', (e) => { if (!drag) return; az -= (e.clientX - drag[0]) * 0.008; el = Math.max(-0.1, Math.min(1.4, el + (e.clientY - drag[1]) * 0.006)); drag = [e.clientX, e.clientY]; redraw(); });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); dist = Math.max(2.5, Math.min(20, dist * Math.exp(e.deltaY * 0.001))); redraw(); }, { passive: false });
  return {
    resize,
    show(b) {
      free();
      const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
      for (const p of b.parts) {
        const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
        const bufs = [];
        const attr = (name, data) => { const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); const loc = gl.getAttribLocation(prog, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0); bufs.push(buf); };
        attr('pos', p.positions); attr('nrm', p.normals);
        const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, p.indices, gl.STATIC_DRAW); bufs.push(ib);
        const groups = []; for (let k = 0; k < p.groups.length; k += 3) groups.push([p.groups[k], p.groups[k + 1], p.groups[k + 2]]);
        parts.push({ vao, bufs, groups });
        // The body's extent frames the view (the wheels and mirrors count too).
        for (let i = 0; i < p.positions.length; i += 3) for (let a = 0; a < 3; a++) { const v = p.positions[i + a]; if (v < lo[a]) lo[a] = v; if (v > hi[a]) hi[a] = v; }
      }
      gl.bindVertexArray(null);
      paint = b.paint;
      center = lo.map((v, i) => (v + hi[i]) / 2);
      radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
      if (!framed) { dist = radius * 2.6; framed = true; }
      redraw();
    },
    destroy() { free(); },
  };
}
