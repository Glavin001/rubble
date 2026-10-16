//! Colour maps for the heat-map views, plus the categorical palette of the fragment view.
//!
//! Colours are display (sRGB-encoded) values in `0..=1`, the space the rasterizer
//! shades and averages in.

/// A display colour, `[r, g, b]` in `0..=1`.
pub type Rgb = [f32; 3];

/// Converts `#rrggbb`-style bytes to an [`Rgb`].
pub const fn rgb8(r: u8, g: u8, b: u8) -> Rgb {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

/// Linear blend `a + (b - a) t`.
pub fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// Scales a colour by a brightness factor (clamped to the displayable range).
pub fn scale(c: Rgb, k: f32) -> Rgb {
    [(c[0] * k).clamp(0.0, 1.0), (c[1] * k).clamp(0.0, 1.0), (c[2] * k).clamp(0.0, 1.0)]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    /// Google's Turbo (an improved jet): blue, cyan, green, yellow, red. The dark ends of
    /// the full map are trimmed so the lowest value still reads as blue.
    Turbo,
    /// ColorBrewer RdBu reversed: blue (negative), near white (zero), red (positive).
    Diverging,
    /// ColorBrewer YlOrRd: pale yellow to dark red (damage).
    Heat,
}

/// RdBu (11 classes), from negative (blue) to positive (red).
const DIVERGING: [Rgb; 11] = [
    rgb8(0x05, 0x30, 0x61),
    rgb8(0x21, 0x66, 0xac),
    rgb8(0x43, 0x93, 0xc3),
    rgb8(0x92, 0xc5, 0xde),
    rgb8(0xd1, 0xe5, 0xf0),
    rgb8(0xf4, 0xf4, 0xf4),
    rgb8(0xfd, 0xdb, 0xc7),
    rgb8(0xf4, 0xa5, 0x82),
    rgb8(0xd6, 0x60, 0x4d),
    rgb8(0xb2, 0x18, 0x2b),
    rgb8(0x67, 0x00, 0x1f),
];

/// YlOrRd from its yellow to its dark red.
const HEAT: [Rgb; 6] = [
    rgb8(0xfe, 0xe0, 0x6a),
    rgb8(0xfe, 0xb2, 0x4c),
    rgb8(0xfd, 0x8d, 0x3c),
    rgb8(0xf0, 0x3b, 0x20),
    rgb8(0xbd, 0x00, 0x26),
    rgb8(0x80, 0x00, 0x26),
];

impl Colormap {
    /// Colour at `t` in `0..=1` (clamped; NaN maps to the low end).
    pub fn sample(self, t: f64) -> Rgb {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) } as f32;
        match self {
            Colormap::Turbo => turbo(0.07 + 0.86 * t),
            Colormap::Diverging => table(&DIVERGING, t),
            Colormap::Heat => table(&HEAT, t),
        }
    }
}

/// Piecewise-linear interpolation of equally spaced colour stops.
fn table(stops: &[Rgb], t: f32) -> Rgb {
    let x = t * (stops.len() - 1) as f32;
    let i = (x.floor() as usize).min(stops.len() - 2);
    mix(stops[i], stops[i + 1], x - i as f32)
}

/// Polynomial fit of Turbo (Mikhailov 2019, Apache-2.0), `t` in `0..=1`.
fn turbo(t: f32) -> Rgb {
    let x = t.clamp(0.0, 1.0);
    let r = 0.135_721_38 + x * (4.615_392_6 + x * (-42.660_32 + x * (132.131_08 + x * (-152.942_4 + x * 59.286_38))));
    let g = 0.091_402_61 + x * (2.194_188_4 + x * (4.842_966_6 + x * (-14.185_033 + x * (4.277_298_5 + x * 2.829_566))));
    let b = 0.106_673_3 + x * (12.641_946 + x * (-60.582_05 + x * (110.362_77 + x * (-89.903_11 + x * 27.348_25))));
    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

/// A distinct, stable colour for a fragment (cluster) id: hues from the golden-ratio
/// sequence of a hashed id, moderately saturated so the edges and shading still read.
pub fn categorical(id: u64) -> Rgb {
    // splitmix64 finaliser: neighbouring ids get unrelated hues.
    let mut z = id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let hue = ((z & 0xffff) as f32 / 65536.0 * 0.618_034 * 7.0).fract();
    let sat = 0.55 + 0.2 * (((z >> 16) & 0xff) as f32 / 255.0);
    let val = 0.82 + 0.13 * (((z >> 24) & 0xff) as f32 / 255.0);
    hsv(hue, sat, val)
}

/// HSV (all components `0..=1`) to RGB.
pub fn hsv(h: f32, s: f32, v: f32) -> Rgb {
    let h6 = h.rem_euclid(1.0) * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Rgb, b: Rgb, tol: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn turbo_runs_blue_to_red() {
        let lo = Colormap::Turbo.sample(0.0);
        let mid = Colormap::Turbo.sample(0.5);
        let hi = Colormap::Turbo.sample(1.0);
        assert!(lo[2] > lo[0] && lo[2] > 0.6, "low end is blue: {lo:?}");
        assert!(mid[1] > 0.8, "middle is green-ish: {mid:?}");
        assert!(hi[0] > 0.6 && hi[2] < 0.1, "high end is red: {hi:?}");
    }

    #[test]
    fn samples_clamp_out_of_range_and_nan() {
        for map in [Colormap::Turbo, Colormap::Diverging, Colormap::Heat] {
            assert_eq!(map.sample(-3.0), map.sample(0.0));
            assert_eq!(map.sample(7.0), map.sample(1.0));
            assert_eq!(map.sample(f64::NAN), map.sample(0.0));
        }
    }

    #[test]
    fn diverging_is_symmetric_about_a_light_middle() {
        let mid = Colormap::Diverging.sample(0.5);
        assert!(mid.iter().all(|&c| c > 0.9), "{mid:?}");
        let (neg, pos) = (Colormap::Diverging.sample(0.0), Colormap::Diverging.sample(1.0));
        assert!(neg[2] > neg[0] && pos[0] > pos[2]);
        assert!(close(table(&[[0.0; 3], [1.0; 3]], 0.25), [0.25; 3], 1e-6));
    }

    #[test]
    fn categorical_is_stable_and_distinct() {
        assert_eq!(categorical(42), categorical(42));
        assert!(!close(categorical(1), categorical(2), 0.05));
        assert!(close(hsv(0.0, 1.0, 1.0), [1.0, 0.0, 0.0], 1e-6));
        assert!(close(hsv(1.0 / 3.0, 1.0, 1.0), [0.0, 1.0, 0.0], 1e-5));
    }
}
