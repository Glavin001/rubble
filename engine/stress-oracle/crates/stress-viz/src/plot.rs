//! The time-series strip under the panels: one or two y axes, full curves drawn
//! faintly with the elapsed part bold, and a cursor at the current time.

use crate::canvas::Canvas;
use crate::colormap::{mix, Rgb};
use crate::font;

#[derive(Clone, Debug)]
pub struct Series {
    pub label: String,
    pub color: Rgb,
    /// Plotted against the right-hand axis.
    pub right: bool,
    pub dashed: bool,
    pub t: Vec<f64>,
    pub v: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Plot {
    pub series: Vec<Series>,
    pub left_unit: String,
    pub right_unit: String,
    pub t_end: f64,
}

const TEXT: Rgb = [0.25, 0.27, 0.3];
const AXIS: Rgb = [0.72, 0.74, 0.78];
const GRIDLINE: Rgb = [0.92, 0.93, 0.95];

/// Formats a time for labels, in ms below 2 s of total duration.
pub fn fmt_time(t: f64, t_end: f64) -> String {
    if t_end < 0.01 {
        format!("{:.3} ms", t * 1e3)
    } else if t_end < 2.0 {
        format!("{:.1} ms", t * 1e3)
    } else {
        format!("{t:.2} s")
    }
}

/// Compact tick label with an SI suffix (`-1.5M`, `250k`, `0.02`).
pub fn fmt_si(v: f64) -> String {
    let a = v.abs();
    let (x, suffix) = if a >= 1e9 {
        (v / 1e9, "G")
    } else if a >= 1e6 {
        (v / 1e6, "M")
    } else if a >= 1e4 {
        (v / 1e3, "k")
    } else {
        (v, "")
    };
    format!("{}{suffix}", crate::views::fmt_number(x))
}

/// Tick positions: the sparsest round step (1, 2, 2.5 or 5 x 10^k) that still puts at
/// least `n` ticks inside `lo..=hi`.
pub fn ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if hi <= lo {
        return vec![lo];
    }
    let at = |step: f64| -> Vec<f64> { ((lo / step).ceil() as i64..=(hi / step).floor() as i64).map(|k| k as f64 * step).collect() };
    let decade = 10f64.powf(((hi - lo) / n as f64).log10().floor());
    let mut steps: Vec<f64> = (-1..=1).flat_map(|e| [1.0, 2.0, 2.5, 5.0].map(|m| m * decade * 10f64.powi(e))).collect();
    steps.sort_by(|a, b| b.total_cmp(a));
    steps.iter().map(|&s| at(s)).find(|t| t.len() >= n).unwrap_or_else(|| at(steps[steps.len() - 1]))
}

/// Range of the series on one side, padded, always including zero.
fn range(series: &[&Series]) -> Option<(f64, f64)> {
    let vals = series.iter().flat_map(|s| s.v.iter().copied()).filter(|v| v.is_finite());
    let (lo, mut hi) = vals.fold((0.0f64, 0.0f64), |(a, b), v| (a.min(v), b.max(v)));
    if series.is_empty() {
        return None;
    }
    // A flat (all-zero) series gets a unit range instead of rounding-noise ticks.
    if hi - lo <= 1e-9 * hi.abs().max(lo.abs()).max(1.0) {
        hi = lo + 1.0;
    }
    let pad = 0.08 * (hi - lo).max(1e-12);
    Some((if lo < 0.0 { lo - pad } else { lo }, hi + pad))
}

/// Draws `plot` into the rectangle `(x, y, w, h)` with the cursor at `t_now`.
pub fn draw(c: &mut Canvas, rect: (i64, i64, i64, i64), plot: &Plot, t_now: f64) {
    let (x, y, w, h) = rect;
    c.fill_rect(x, y, w, h, [1.0; 3]);
    c.fill_rect(x, y, w, 1, AXIS);
    let s = 2usize;
    let th = font::text_height(s) as i64;
    let left: Vec<&Series> = plot.series.iter().filter(|s| !s.right).collect();
    let right: Vec<&Series> = plot.series.iter().filter(|s| s.right).collect();
    let (lrange, rrange) = (range(&left), range(&right));
    // Plot area inside axis labels and the legend row.
    let ax0 = x + 110;
    let ax1 = x + w - if rrange.is_some() { 110 } else { 70 };
    let ay0 = y + th + 22;
    let ay1 = y + h - th - 16;
    if ax1 <= ax0 + 10 || ay1 <= ay0 + 10 {
        return;
    }
    let t_end = plot.t_end.max(1e-12);
    let px = |t: f64| ax0 as f32 + ((t / t_end).clamp(0.0, 1.0) * (ax1 - ax0) as f64) as f32;
    let py = |v: f64, (lo, hi): (f64, f64)| ay1 as f32 - (((v - lo) / (hi - lo)) * (ay1 - ay0) as f64) as f32;

    // Grid, ticks and axis labels.
    if let Some(lr) = lrange {
        for v in ticks(lr.0, lr.1, 3) {
            let yy = py(v, lr).round() as i64;
            c.fill_rect(ax0, yy, ax1 - ax0, 1, if v == 0.0 { AXIS } else { GRIDLINE });
            c.text_right(ax0 - 8, yy - th / 2, &fmt_si(v), s, TEXT);
        }
        c.text(x + 10, y + 10, &plot.left_unit, s, TEXT);
    }
    if let Some(rr) = rrange {
        for v in ticks(rr.0, rr.1, 3) {
            let yy = py(v, rr).round() as i64;
            c.fill_rect(ax1, yy, 5, 1, AXIS);
            c.text(ax1 + 9, yy - th / 2, &fmt_si(v), s, TEXT);
        }
        c.text_right(x + w - 10, y + 10, &plot.right_unit, s, TEXT);
    }
    c.fill_rect(ax0, ay1, ax1 - ax0, 1, AXIS);
    for t in ticks(0.0, t_end, 8) {
        let xx = px(t).round() as i64;
        c.fill_rect(xx, ay1, 1, 5, AXIS);
        let label = fmt_time(t, t_end);
        let tw = font::text_width(&label, s) as i64;
        c.text(xx - tw / 2, ay1 + 8, &label, s, TEXT);
    }

    // Curves: faint over the whole run, bold up to now.
    let mut legend_x = ax0.max(x + 10 + font::text_width(&plot.left_unit, s) as i64 + 40);
    for sr in &plot.series {
        let Some(r) = (if sr.right { rrange } else { lrange }) else { continue };
        let pts = decimate(&sr.t, &sr.v, (ax1 - ax0) as usize);
        let screen: Vec<[f32; 2]> = pts.iter().map(|&(t, v)| [px(t), py(v, r)]).collect();
        let dash = sr.dashed.then_some((7.0, 5.0));
        c.polyline(&screen, 1.5, mix(sr.color, [1.0; 3], 0.65), dash);
        let n_now = pts.partition_point(|&(t, _)| t <= t_now);
        if n_now > 0 {
            let mut now: Vec<[f32; 2]> = screen[..n_now].to_vec();
            if let Some(v) = value_at(&sr.t, &sr.v, t_now) {
                let p = [px(t_now.min(*sr.t.last().unwrap())), py(v, r)];
                now.push(p);
                c.polyline(&now, 2.5, sr.color, dash);
                c.dot(p, 4.0, sr.color);
            }
        }
        // Legend entry.
        let ly = (y + 10 + th / 2) as f32;
        c.polyline(&[[legend_x as f32, ly], [legend_x as f32 + 26.0, ly]], 3.0, sr.color, dash);
        legend_x += 34 + c.text(legend_x + 34, y + 10, &sr.label, s, TEXT) as i64 + 30;
    }
    let xx = px(t_now);
    c.line([xx, ay0 as f32], [xx, ay1 as f32], 1.0, [0.45, 0.47, 0.52]);
}

/// Linear interpolation of a series at `t` (None before its first sample).
fn value_at(ts: &[f64], vs: &[f64], t: f64) -> Option<f64> {
    let i = ts.partition_point(|&x| x <= t);
    if i == 0 {
        return None;
    }
    if i == ts.len() {
        return vs.last().copied();
    }
    let f = (t - ts[i - 1]) / (ts[i] - ts[i - 1]).max(1e-300);
    Some(vs[i - 1] + f * (vs[i] - vs[i - 1]))
}

/// At most two points (min and max) per pixel column, so long series keep their peaks.
fn decimate(ts: &[f64], vs: &[f64], columns: usize) -> Vec<(f64, f64)> {
    let pts: Vec<(f64, f64)> = ts.iter().copied().zip(vs.iter().copied()).filter(|p| p.1.is_finite()).collect();
    if pts.len() <= 2 * columns.max(1) || pts.is_empty() {
        return pts;
    }
    let (t0, t1) = (pts[0].0, pts[pts.len() - 1].0);
    let mut out = Vec::with_capacity(2 * columns);
    let mut i = 0;
    for col in 0..columns {
        let end_t = t0 + (t1 - t0) * (col + 1) as f64 / columns as f64;
        let start = i;
        while i < pts.len() && (pts[i].0 <= end_t || col + 1 == columns) {
            i += 1;
        }
        if i == start {
            continue;
        }
        let bucket = &pts[start..i];
        let lo = bucket.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        let hi = bucket.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        if lo.0 <= hi.0 {
            out.extend([*lo, *hi]);
        } else {
            out.extend([*hi, *lo]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_round_and_inside() {
        assert_eq!(ticks(0.0, 1.0, 4), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let t = ticks(-3.3, 7.9, 3);
        assert_eq!(t, vec![-2.5, 0.0, 2.5, 5.0, 7.5]);
        assert_eq!(ticks(-4.4e4, 4.6e4, 3), vec![-2.5e4, 0.0, 2.5e4]);
        assert_eq!(fmt_si(2_500_000.0), "2.5M");
        let flat = Series { label: String::new(), color: [0.0; 3], right: false, dashed: false, t: vec![0.0, 1.0], v: vec![0.0, 1e-14] };
        assert_eq!(range(&[&flat]), Some((0.0, 1.08)));
        assert_eq!(fmt_si(-15_000.0), "-15k");
        assert_eq!(fmt_time(0.0123, 0.04), "12.3 ms");
    }

    #[test]
    fn decimation_keeps_peaks_and_interpolation_works() {
        let ts: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let mut vs = vec![0.0; 10_000];
        vs[5_001] = 9.0;
        let d = decimate(&ts, &vs, 100);
        assert!(d.len() <= 200 && d.iter().any(|p| p.1 == 9.0));
        assert_eq!(value_at(&[0.0, 1.0], &[0.0, 10.0], 0.25), Some(2.5));
        assert_eq!(value_at(&[1.0, 2.0], &[0.0, 10.0], 0.5), None);
    }
}
