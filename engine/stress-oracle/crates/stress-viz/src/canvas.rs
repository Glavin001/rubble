//! The output image (RGB24) and the 2-D drawing used for overlays: rectangles,
//! antialiased lines and dots, gradients and bitmap text.

use crate::colormap::Rgb;
use crate::font;

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB24.
    pub pixels: Vec<u8>,
}

fn to_u8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

impl Canvas {
    pub fn new(width: usize, height: usize, background: Rgb) -> Canvas {
        let px = [to_u8(background[0]), to_u8(background[1]), to_u8(background[2])];
        Canvas { width, height, pixels: px.repeat(width * height) }
    }

    pub fn get(&self, x: usize, y: usize) -> Rgb {
        let i = 3 * (y * self.width + x);
        [self.pixels[i] as f32 / 255.0, self.pixels[i + 1] as f32 / 255.0, self.pixels[i + 2] as f32 / 255.0]
    }

    pub fn set(&mut self, x: usize, y: usize, c: Rgb) {
        let i = 3 * (y * self.width + x);
        self.pixels[i] = to_u8(c[0]);
        self.pixels[i + 1] = to_u8(c[1]);
        self.pixels[i + 2] = to_u8(c[2]);
    }

    /// Copies a `w x h` RGBA8 image (rows `stride` bytes apart, alpha ignored) with its
    /// top-left at `(x0, y0)`, clipped to the canvas.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn blit_rgba(&mut self, x0: usize, y0: usize, w: usize, h: usize, data: &[u8], stride: usize) {
        let w = w.min(self.width.saturating_sub(x0));
        for y in 0..h.min(self.height.saturating_sub(y0)) {
            let src = &data[y * stride..y * stride + 4 * w];
            let dst = &mut self.pixels[3 * ((y0 + y) * self.width + x0)..][..3 * w];
            for (d, s) in dst.chunks_exact_mut(3).zip(src.chunks_exact(4)) {
                d.copy_from_slice(&s[..3]);
            }
        }
    }

    /// Blends `c` over pixel `(x, y)` with opacity `alpha`; out-of-bounds is ignored.
    pub fn blend(&mut self, x: i64, y: i64, c: Rgb, alpha: f32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 || alpha <= 0.0 {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        let a = alpha.min(1.0);
        let old = self.get(x, y);
        self.set(x, y, [old[0] + (c[0] - old[0]) * a, old[1] + (c[1] - old[1]) * a, old[2] + (c[2] - old[2]) * a]);
    }

    /// Fills an axis-aligned rectangle (clipped to the canvas).
    pub fn fill_rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: Rgb) {
        for yy in y.max(0)..(y + h).min(self.height as i64) {
            for xx in x.max(0)..(x + w).min(self.width as i64) {
                self.set(xx as usize, yy as usize, c);
            }
        }
    }

    /// One-pixel rectangle outline.
    pub fn stroke_rect(&mut self, x: i64, y: i64, w: i64, h: i64, c: Rgb) {
        self.fill_rect(x, y, w, 1, c);
        self.fill_rect(x, y + h - 1, w, 1, c);
        self.fill_rect(x, y, 1, h, c);
        self.fill_rect(x + w - 1, y, 1, h, c);
    }

    /// Vertical gradient filling a rectangle: `color(t)` with `t = 0` at the bottom row
    /// and `t = 1` at the top row.
    pub fn vertical_gradient(&mut self, x: i64, y: i64, w: i64, h: i64, color: impl Fn(f64) -> Rgb) {
        for row in 0..h {
            let t = if h > 1 { 1.0 - row as f64 / (h - 1) as f64 } else { 0.5 };
            self.fill_rect(x, y + row, w, 1, color(t));
        }
    }

    /// Antialiased line of `width` pixels (coverage from the distance to the segment).
    pub fn line(&mut self, a: [f32; 2], b: [f32; 2], width: f32, c: Rgb) {
        let r = 0.5 * width;
        let (x0, x1) = (a[0].min(b[0]) - r - 1.0, a[0].max(b[0]) + r + 1.0);
        let (y0, y1) = (a[1].min(b[1]) - r - 1.0, a[1].max(b[1]) + r + 1.0);
        let d = [b[0] - a[0], b[1] - a[1]];
        let len2 = (d[0] * d[0] + d[1] * d[1]).max(1e-12);
        for y in (y0.floor() as i64).max(0)..=(y1.ceil() as i64).min(self.height as i64 - 1) {
            for x in (x0.floor() as i64).max(0)..=(x1.ceil() as i64).min(self.width as i64 - 1) {
                let p = [x as f32 + 0.5 - a[0], y as f32 + 0.5 - a[1]];
                let t = ((p[0] * d[0] + p[1] * d[1]) / len2).clamp(0.0, 1.0);
                let q = [p[0] - t * d[0], p[1] - t * d[1]];
                let dist = (q[0] * q[0] + q[1] * q[1]).sqrt();
                self.blend(x, y, c, (r + 0.5 - dist).clamp(0.0, 1.0));
            }
        }
    }

    /// Polyline; with `dash = Some((on, off))` (pixels) only the "on" stretches are drawn.
    pub fn polyline(&mut self, pts: &[[f32; 2]], width: f32, c: Rgb, dash: Option<(f32, f32)>) {
        let mut along = 0.0f32;
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            match dash {
                None => self.line(a, b, width, c),
                Some((on, off)) => {
                    // Split the segment at dash boundaries.
                    let period = on + off;
                    let mut s = 0.0f32;
                    while s < len {
                        let phase = (along + s) % period;
                        let step = if phase < on { on - phase } else { period - phase };
                        let e = (s + step).min(len);
                        if phase < on && len > 0.0 {
                            let p = |u: f32| [a[0] + (b[0] - a[0]) * u / len, a[1] + (b[1] - a[1]) * u / len];
                            self.line(p(s), p(e), width, c);
                        }
                        s = e + 1e-4;
                    }
                }
            }
            along += len;
        }
    }

    /// Filled antialiased disc.
    pub fn dot(&mut self, center: [f32; 2], radius: f32, c: Rgb) {
        for y in (center[1] - radius - 1.0).floor() as i64..=(center[1] + radius + 1.0).ceil() as i64 {
            for x in (center[0] - radius - 1.0).floor() as i64..=(center[0] + radius + 1.0).ceil() as i64 {
                let d = ((x as f32 + 0.5 - center[0]).powi(2) + (y as f32 + 0.5 - center[1]).powi(2)).sqrt();
                self.blend(x, y, c, (radius + 0.5 - d).clamp(0.0, 1.0));
            }
        }
    }

    /// Draws `text` with its top-left corner at `(x, y)`; returns the width drawn.
    pub fn text(&mut self, x: i64, y: i64, text: &str, scale: usize, c: Rgb) -> usize {
        let s = scale as i64;
        font::for_each_pixel(text, |fx, fy| self.fill_rect(x + fx as i64 * s, y + fy as i64 * s, s, s, c));
        font::text_width(text, scale)
    }

    /// Text whose right edge is at `x_right`.
    pub fn text_right(&mut self, x_right: i64, y: i64, text: &str, scale: usize, c: Rgb) {
        let w = font::text_width(text, scale) as i64;
        self.text(x_right - w, y, text, scale, c);
    }

    /// Text on a translucent white box (labels over the 3-D view).
    pub fn label(&mut self, x: i64, y: i64, text: &str, scale: usize, fg: Rgb) {
        let pad = 2 * scale as i64;
        let w = font::text_width(text, scale) as i64 + 2 * pad;
        let h = font::text_height(scale) as i64 + 2 * pad;
        for yy in y..y + h {
            for xx in x..x + w {
                self.blend(xx, yy, [1.0; 3], 0.75);
            }
        }
        self.text(x + pad, y + pad, text, scale, fg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_covers_its_pixels_and_stays_thin() {
        let mut c = Canvas::new(20, 20, [1.0; 3]);
        c.line([2.0, 10.5], [18.0, 10.5], 1.0, [0.0; 3]);
        assert!(c.get(10, 10)[0] < 0.05, "on the line");
        assert!(c.get(10, 8)[0] > 0.99, "two rows away is untouched");
        assert!(c.get(10, 12)[0] > 0.99);
    }

    #[test]
    fn dashes_leave_gaps() {
        let mut c = Canvas::new(40, 5, [1.0; 3]);
        c.polyline(&[[0.0, 2.5], [40.0, 2.5]], 1.0, [0.0; 3], Some((4.0, 4.0)));
        assert!(c.get(1, 2)[0] < 0.1);
        assert!(c.get(6, 2)[0] > 0.9);
        assert!(c.get(9, 2)[0] < 0.1);
    }

    #[test]
    fn text_draws_scaled_pixels() {
        let mut c = Canvas::new(30, 30, [1.0; 3]);
        let w = c.text(1, 1, "|", 3, [0.0; 3]);
        assert_eq!(w, 15);
        // The bar is column 2 of the glyph: pixels 7..10 at scale 3.
        assert!(c.get(7, 5)[0] < 0.01 && c.get(9, 21)[0] < 0.01);
        assert!(c.get(6, 5)[0] > 0.99 && c.get(10, 5)[0] > 0.99);
    }
}
