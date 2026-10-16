//! Software rasterizer: flat-coloured triangles with a z-buffer, thick 3-D lines,
//! an object-id buffer for outline detection, and supersampled resolve.
//!
//! The drawing interface is the [`Target`] trait: its provided `polygon` and `line`
//! clip at the near plane, project and split into screen-space triangles, so the
//! software [`Raster`] and the GPU panel (`gpu.rs`) receive identical geometry.
//!
//! A [`Raster`] covers one panel at `ss x` the output resolution. The renderer draws
//! opaque geometry with [`Depth::Write`] (tagging every sample with an object/face id),
//! calls [`Raster::outline`] to darken samples where the id changes (box edges,
//! silhouettes, joints between chunks), then draws overlays (crack lines, bond decals)
//! with [`Depth::Test`] — visible only where they lie on the visible surface, within a
//! depth bias — and finally averages each `ss x ss` block into the output canvas.
//!
//! Depth is the view-space distance along the camera axis, interpolated
//! perspective-correctly (linear in `1/z` across the screen).

use crate::camera::{Projection, NEAR};
use crate::canvas::Canvas;
use crate::colormap::Rgb;

/// Relative depth difference below which a later opaque surface does not replace an
/// earlier one.
const COPLANAR: f32 = 1e-5;

/// Id of samples not covered by any opaque geometry.
pub const BACKGROUND: u32 = 0;

/// Packs an object number (1-based) and a face number (0..8) into a sample id.
pub fn object_id(object: u32, face: u32) -> u32 {
    (object << 3) | (face & 7)
}

/// How a primitive interacts with the depth buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Depth {
    /// Opaque: depth test and write, tagging covered samples with this id.
    Write { id: u32 },
    /// Overlay: drawn where its depth is within `bias` (m) of the stored depth or in
    /// front of it; nothing is written but colour.
    Test { bias: f32 },
    /// Always drawn on top.
    Always,
}

/// Darkening applied by [`Raster::outline`] (0 = none, 1 = black).
#[derive(Clone, Copy, Debug)]
pub struct OutlineStyle {
    /// Object against background.
    pub silhouette: f32,
    /// Two different objects (e.g. neighbouring chunks).
    pub object: f32,
    /// Two faces of the same object (box edges).
    pub crease: f32,
}

/// A panel the scene is drawn into: screen-space triangles in sample coordinates with
/// view depth, an outline pass, and the shared view-space front end.
pub trait Target {
    /// Projection of view space to this target's sample coordinates.
    fn proj(&self) -> &Projection;

    /// Clears to a vertical gradient from `top` to `bottom`, depth and ids reset.
    fn clear(&mut self, top: Rgb, bottom: Rgb);

    /// One screen-space triangle (`[x, y, view depth]` per vertex, either winding).
    fn triangle(&mut self, p: [[f32; 3]; 3], color: Rgb, mode: Depth);

    /// Darkens the samples on both sides of every id change.
    fn outline(&mut self, style: OutlineStyle);

    /// Fills a convex polygon given in view space (clipped at the near plane).
    fn polygon(&mut self, pts: &[[f32; 3]], color: Rgb, mode: Depth) {
        let clipped = clip_near(pts);
        if clipped.len() < 3 {
            return;
        }
        let s: Vec<[f32; 3]> = clipped
            .iter()
            .map(|v| {
                let p = self.proj().project(*v);
                [p[0], p[1], v[2]]
            })
            .collect();
        for i in 1..s.len() - 1 {
            self.triangle([s[0], s[i], s[i + 1]], color, mode);
        }
    }

    /// A line `width` samples wide between two view-space points.
    fn line(&mut self, a: [f32; 3], b: [f32; 3], width: f32, color: Rgb, mode: Depth) {
        let (a, b) = match (a[2] >= NEAR, b[2] >= NEAR) {
            (true, true) => (a, b),
            (false, false) => return,
            (true, false) => (a, lerp_to_near(a, b)),
            (false, true) => (lerp_to_near(b, a), b),
        };
        let (pa, pb) = (self.proj().project(a), self.proj().project(b));
        let d = [pb[0] - pa[0], pb[1] - pa[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let h = 0.5 * width;
        // Unit direction (any, for a degenerate line) and its normal, both half-width long.
        let u = if len > 1e-6 { [d[0] / len * h, d[1] / len * h] } else { [h, 0.0] };
        let n = [-u[1], u[0]];
        let q = [
            [pa[0] - u[0] + n[0], pa[1] - u[1] + n[1], a[2]],
            [pb[0] + u[0] + n[0], pb[1] + u[1] + n[1], b[2]],
            [pb[0] + u[0] - n[0], pb[1] + u[1] - n[1], b[2]],
            [pa[0] - u[0] - n[0], pa[1] - u[1] - n[1], a[2]],
        ];
        self.triangle([q[0], q[1], q[2]], color, mode);
        self.triangle([q[0], q[2], q[3]], color, mode);
    }
}

pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub proj: Projection,
    color: Vec<Rgb>,
    depth: Vec<f32>,
    id: Vec<u32>,
    dark: Vec<f32>,
}

impl Raster {
    pub fn new(width: usize, height: usize, proj: Projection) -> Raster {
        let n = width * height;
        Raster { width, height, proj, color: vec![[1.0; 3]; n], depth: vec![f32::INFINITY; n], id: vec![BACKGROUND; n], dark: vec![0.0; n] }
    }

    #[cfg(test)]
    pub fn color_at(&self, x: usize, y: usize) -> Rgb {
        self.color[y * self.width + x]
    }

    #[cfg(test)]
    pub fn id_at(&self, x: usize, y: usize) -> u32 {
        self.id[y * self.width + x]
    }

    /// Averages each `ss x ss` block into the canvas with its top-left at `(x0, y0)`.
    pub fn resolve(&self, canvas: &mut Canvas, x0: usize, y0: usize, ss: usize) {
        let k = 1.0 / (ss * ss) as f32;
        for oy in 0..self.height / ss {
            for ox in 0..self.width / ss {
                let mut acc = [0.0f32; 3];
                for sy in 0..ss {
                    let row = (oy * ss + sy) * self.width + ox * ss;
                    for c in &self.color[row..row + ss] {
                        acc[0] += c[0];
                        acc[1] += c[1];
                        acc[2] += c[2];
                    }
                }
                let (cx, cy) = (x0 + ox, y0 + oy);
                if cx < canvas.width && cy < canvas.height {
                    canvas.set(cx, cy, [acc[0] * k, acc[1] * k, acc[2] * k]);
                }
            }
        }
    }
}

impl Target for Raster {
    fn proj(&self) -> &Projection {
        &self.proj
    }

    fn clear(&mut self, top: Rgb, bottom: Rgb) {
        for y in 0..self.height {
            let t = y as f32 / (self.height.max(2) - 1) as f32;
            let c = crate::colormap::mix(top, bottom, t);
            self.color[y * self.width..(y + 1) * self.width].fill(c);
        }
        self.depth.fill(f32::INFINITY);
        self.id.fill(BACKGROUND);
    }

    /// A sample is covered when its centre is inside the triangle.
    fn triangle(&mut self, p: [[f32; 3]; 3], color: Rgb, mode: Depth) {
        let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
        if !area.is_finite() || area.abs() <= 1e-8 {
            return;
        }
        let inv = 1.0 / area;
        let min_x = p.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let max_x = p.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max).ceil().min(self.width as f32) as usize;
        let min_y = p.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as usize;
        let max_y = p.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min(self.height as f32) as usize;
        if min_x >= max_x || min_y >= max_y {
            return;
        }
        // Barycentric weight of vertex k as an affine function of the sample position.
        let weight = |k: usize| {
            let (a, b) = (p[(k + 1) % 3], p[(k + 2) % 3]);
            let dx = -(b[1] - a[1]) * inv;
            let dy = (b[0] - a[0]) * inv;
            (dx, dy, -(dx * a[0] + dy * a[1]))
        };
        let (w0, w1) = (weight(0), weight(1));
        let iz = [1.0 / p[0][2], 1.0 / p[1][2], 1.0 / p[2][2]];
        const EPS: f32 = -1e-5;
        for y in min_y..max_y {
            let py = y as f32 + 0.5;
            let row = y * self.width;
            let px0 = min_x as f32 + 0.5;
            let mut a = w0.0 * px0 + w0.1 * py + w0.2;
            let mut b = w1.0 * px0 + w1.1 * py + w1.2;
            for x in min_x..max_x {
                let c = 1.0 - a - b;
                if a >= EPS && b >= EPS && c >= EPS {
                    let z = 1.0 / (a * iz[0] + b * iz[1] + c * iz[2]);
                    let i = row + x;
                    match mode {
                        Depth::Write { id } => {
                            // Coplanar overlapping faces (touching or interpenetrating
                            // boxes): the first drawn wins, instead of float noise.
                            if z < self.depth[i] * (1.0 - COPLANAR) {
                                self.depth[i] = z;
                                self.id[i] = id;
                                self.color[i] = color;
                            }
                        }
                        Depth::Test { bias } => {
                            if z <= self.depth[i] + bias {
                                self.color[i] = color;
                            }
                        }
                        Depth::Always => self.color[i] = color,
                    }
                }
                a += w0.0;
                b += w1.0;
            }
        }
    }

    fn outline(&mut self, style: OutlineStyle) {
        let (w, h) = (self.width, self.height);
        self.dark.fill(0.0);
        let strength = |a: u32, b: u32| -> f32 {
            if a == b {
                0.0
            } else if a == BACKGROUND || b == BACKGROUND {
                style.silhouette
            } else if a >> 3 != b >> 3 {
                style.object
            } else {
                style.crease
            }
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                for j in [if x + 1 < w { i + 1 } else { i }, if y + 1 < h { i + w } else { i }] {
                    let s = strength(self.id[i], self.id[j]);
                    if s > 0.0 {
                        self.dark[i] = self.dark[i].max(s);
                        self.dark[j] = self.dark[j].max(s);
                    }
                }
            }
        }
        for (c, &d) in self.color.iter_mut().zip(&self.dark) {
            if d > 0.0 {
                *c = crate::colormap::scale(*c, 1.0 - d);
            }
        }
    }

}

/// Point on segment `inside -> outside` where it crosses the near plane.
fn lerp_to_near(inside: [f32; 3], outside: [f32; 3]) -> [f32; 3] {
    let t = (inside[2] - NEAR) / (inside[2] - outside[2]);
    [inside[0] + (outside[0] - inside[0]) * t, inside[1] + (outside[1] - inside[1]) * t, NEAR]
}

/// Sutherland-Hodgman clip of a polygon against `z >= NEAR`.
fn clip_near(pts: &[[f32; 3]]) -> Vec<[f32; 3]> {
    if pts.iter().all(|p| p[2] >= NEAR) {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(pts.len() + 2);
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        match (a[2] >= NEAR, b[2] >= NEAR) {
            (true, true) => out.push(b),
            (true, false) => out.push(lerp_to_near(a, b)),
            (false, true) => {
                out.push(lerp_to_near(b, a));
                out.push(b);
            }
            (false, false) => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster(w: usize, h: usize) -> Raster {
        // focal 1, centred: view (x, y, 1) maps to pixel (cx + x, cy - y).
        let mut r = Raster::new(w, h, Projection { focal: 1.0, cx: 0.0, cy: 0.0 });
        r.clear([1.0; 3], [1.0; 3]);
        r
    }

    const RED: Rgb = [1.0, 0.0, 0.0];
    const BLUE: Rgb = [0.0, 0.0, 1.0];

    #[test]
    fn triangle_covers_inside_samples_only() {
        let mut r = raster(10, 10);
        r.triangle([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [0.0, 10.0, 1.0]], RED, Depth::Write { id: 8 });
        assert_eq!(r.color_at(1, 1), RED);
        assert_eq!(r.id_at(1, 1), 8);
        assert_eq!(r.color_at(8, 8), [1.0; 3], "outside the hypotenuse");
        // Clockwise winding draws the same samples.
        let mut s = raster(10, 10);
        s.triangle([[0.0, 0.0, 1.0], [0.0, 10.0, 1.0], [10.0, 0.0, 1.0]], RED, Depth::Write { id: 8 });
        assert_eq!(r.color, s.color);
    }

    #[test]
    fn nearer_surface_wins_in_either_order() {
        let quad = |r: &mut Raster, z: f32, c: Rgb, id: u32| {
            r.triangle([[0.0, 0.0, z], [8.0, 0.0, z], [8.0, 8.0, z]], c, Depth::Write { id });
            r.triangle([[0.0, 0.0, z], [8.0, 8.0, z], [0.0, 8.0, z]], c, Depth::Write { id });
        };
        for order in [false, true] {
            let mut r = raster(8, 8);
            if order {
                quad(&mut r, 2.0, RED, 8);
                quad(&mut r, 1.0, BLUE, 16);
            } else {
                quad(&mut r, 1.0, BLUE, 16);
                quad(&mut r, 2.0, RED, 8);
            }
            assert_eq!(r.color_at(4, 4), BLUE);
        }
    }

    #[test]
    fn depth_is_perspective_correct() {
        // A plane tilted in depth: its screen-space midpoint is nearer than the mean depth.
        let mut r = raster(64, 1);
        r.triangle([[0.0, -1.0, 1.0], [64.0, -1.0, 3.0], [0.0, 3.0, 1.0]], RED, Depth::Write { id: 8 });
        r.triangle([[64.0, -1.0, 3.0], [64.0, 3.0, 3.0], [0.0, 3.0, 1.0]], RED, Depth::Write { id: 8 });
        let mid = r.depth[32];
        assert!((mid - 1.5).abs() < 0.03, "1/z is linear on screen: {mid}");
    }

    #[test]
    fn overlays_respect_depth_with_bias() {
        let mut r = raster(8, 8);
        r.triangle([[0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [0.0, 8.0, 1.0]], RED, Depth::Write { id: 8 });
        r.triangle([[0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [0.0, 8.0, 1.0]], BLUE, Depth::Test { bias: 0.01 });
        assert_eq!(r.color_at(1, 1), BLUE, "coplanar overlay shows");
        r.triangle([[0.0, 0.0, 1.5], [8.0, 0.0, 1.5], [0.0, 8.0, 1.5]], [0.0; 3], Depth::Test { bias: 0.01 });
        assert_eq!(r.color_at(1, 1), BLUE, "overlay behind the surface is hidden");
    }

    #[test]
    fn polygons_behind_the_eye_are_clipped() {
        let mut r = Raster::new(20, 20, Projection { focal: 10.0, cx: 10.0, cy: 10.0 });
        r.clear([1.0; 3], [1.0; 3]);
        // Spans from behind the eye to in front: must not panic, must draw something.
        r.polygon(&[[-1.0, -1.0, -1.0], [1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 1.0, -1.0]], RED, Depth::Write { id: 8 });
        assert!(r.color.contains(&RED));
        assert_eq!(clip_near(&[[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -2.0]]).len(), 0);
        r.line([0.0, 0.0, -1.0], [0.0, 0.0, -2.0], 2.0, RED, Depth::Always);
    }

    #[test]
    fn outline_darkens_id_changes_and_resolve_averages() {
        let mut r = raster(4, 2);
        r.triangle([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0]], RED, Depth::Write { id: object_id(1, 0) });
        r.triangle([[0.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 2.0, 1.0]], RED, Depth::Write { id: object_id(1, 0) });
        r.triangle([[2.0, 0.0, 1.0], [4.0, 0.0, 1.0], [4.0, 2.0, 1.0]], RED, Depth::Write { id: object_id(2, 0) });
        r.triangle([[2.0, 0.0, 1.0], [4.0, 2.0, 1.0], [2.0, 2.0, 1.0]], RED, Depth::Write { id: object_id(2, 0) });
        r.outline(OutlineStyle { silhouette: 1.0, object: 0.5, crease: 0.25 });
        assert_eq!(r.color_at(0, 0), RED, "inside one object");
        assert_eq!(r.color_at(1, 0), [0.5, 0.0, 0.0], "left of the joint");
        assert_eq!(r.color_at(2, 0), [0.5, 0.0, 0.0], "right of the joint");
        let mut c = Canvas::new(2, 1, [0.0; 3]);
        r.resolve(&mut c, 0, 0, 2);
        let px = c.get(0, 0);
        assert!((px[0] - 0.75).abs() < 0.01, "2x2 average: {px:?}");
    }
}
