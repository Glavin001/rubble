//! Frame composition: a header (scene, time, slow-motion factor), a grid of 3-D panels
//! (one view mode of one recording each, with title and colour bar) and the
//! time-series strip.

use crate::camera::{Aabb, Camera, CameraSpec};
use crate::canvas::Canvas;
use crate::colormap::{mix, rgb8, Rgb};
use crate::font;
use crate::plot::{self, fmt_time, Plot, Series};
use crate::raster::Raster;
#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, GpuFrame};
use crate::record::{Frame, Recording};
use crate::scene3d::{clip_label, DrawOptions, PanelScene};
use crate::views::{fmt_number, Legend, Scales, ViewMode, ANCHORED};

const INK: Rgb = rgb8(0x1e, 0x22, 0x2a);
const MUTED: Rgb = rgb8(0x5a, 0x60, 0x6c);
const RULE: Rgb = rgb8(0xc8, 0xcc, 0xd4);
const PAGE: Rgb = rgb8(0xf4, 0xf5, 0xf7);
/// Series colours of the plot strip (variants in compare mode).
const PALETTE: [Rgb; 6] = [
    rgb8(0x1f, 0x6f, 0xc4),
    rgb8(0xd6, 0x3a, 0x2e),
    rgb8(0x2a, 0x9d, 0x55),
    rgb8(0x8e, 0x5c, 0xc2),
    rgb8(0xe8, 0x8a, 0x1a),
    rgb8(0x55, 0x58, 0x60),
];

const HEADER_H: usize = 62;
const PLOT_H: usize = 172;
const GUTTER: usize = 6;
/// Height kept free for the panel title when fitting the camera.
const TITLE_BAND: usize = 34;
/// Width of the colour-bar strip on the right of each panel.
const LEGEND_W: usize = 150;

/// One panel: a view mode of one recording.
#[derive(Clone, Debug)]
pub struct PanelSpec {
    pub title: String,
    pub mode: ViewMode,
    pub rec: usize,
}

pub struct Composer<'a> {
    pub recs: Vec<&'a Recording>,
    pub panels: Vec<PanelSpec>,
    pub cols: usize,
    pub width: usize,
    pub height: usize,
    pub scales: Scales,
    pub opts: DrawOptions,
    pub plot: Option<Plot>,
    pub title: String,
    pub subtitle: String,
    /// Simulated seconds per second of video.
    pub sim_per_video_second: f64,
    camera: Camera,
    cell: (usize, usize),
    bias: f32,
    bounds: Aabb,
}

impl<'a> Composer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        recs: Vec<&'a Recording>,
        panels: Vec<PanelSpec>,
        cols: usize,
        (width, height): (usize, usize),
        camera: &CameraSpec,
        scales: Scales,
        opts: DrawOptions,
        plot: Option<Plot>,
        title: String,
        subtitle: String,
        fps: f64,
    ) -> Composer<'a> {
        let rows = panels.len().div_ceil(cols);
        let plot_h = if plot.is_some() { PLOT_H } else { 0 };
        let area_h = height.saturating_sub(HEADER_H + plot_h);
        let cell_w = (width - GUTTER * (cols + 1)) / cols;
        let cell_h = (area_h.saturating_sub(GUTTER * (rows + 1)) / rows).max(16);
        let mut bounds = Aabb::EMPTY;
        for r in &recs {
            bounds.include(r.bounds.lo);
            bounds.include(r.bounds.hi);
        }
        let view_w = view_width(cell_w);
        let view_h = cell_h - title_band(cell_h);
        let cam = camera.build(&bounds, view_w as f64 / view_h as f64);
        let min_half = recs.iter().map(|r| r.min_half_extent).fold(f64::INFINITY, f64::min);
        let sim_per_video_second = recs[0].frame_interval() * fps;
        Composer {
            recs,
            panels,
            cols,
            width,
            height,
            scales,
            opts,
            plot,
            title,
            subtitle,
            sim_per_video_second,
            camera: cam,
            cell: (cell_w, cell_h),
            bias: (0.4 * min_half) as f32,
            bounds,
        }
    }

    /// Number of distinct frames (the longest recording).
    pub fn frame_count(&self) -> usize {
        self.recs.iter().map(|r| r.frames.len()).max().unwrap_or(0)
    }

    /// A raster sized for one panel (reused across panels and frames).
    pub fn new_raster(&self) -> Raster {
        let ss = self.opts.ss;
        let (w, h) = (self.cell.0 * ss, self.cell.1 * ss);
        Raster::new(w, h, self.panel_projection())
    }

    /// The panel projection at sample resolution: the 3-D view is fitted to the part of
    /// the cell left of the colour bar and below the title band; the raster still covers
    /// the whole cell.
    fn panel_projection(&self) -> crate::camera::Projection {
        let ss = self.opts.ss;
        let h = self.cell.1 * ss;
        let band = title_band(self.cell.1) * ss;
        let mut proj = self.camera.projection(view_width(self.cell.0) * ss, h - band);
        proj.cy += band as f32;
        proj
    }

    /// The rasterizer of one render thread: GPU panels for every panel of a frame when
    /// a GPU context is given, else the software raster.
    pub fn new_rasterizer(&self, gpu: Option<&GpuHandle>) -> Rasterizer {
        #[cfg(not(feature = "gpu"))]
        let _ = gpu;
        #[cfg(feature = "gpu")]
        if let Some(ctx) = gpu {
            let ss = self.opts.ss;
            let frame = GpuFrame::new(ctx, self.panels.len(), self.cell.0 * ss, self.cell.1 * ss, ss, self.panel_projection());
            return Rasterizer::Gpu(Box::new(frame));
        }
        Rasterizer::Cpu(self.new_raster())
    }

    fn cell_origin(&self, i: usize) -> (usize, usize) {
        let (c, r) = (i % self.cols, i / self.cols);
        (GUTTER + c * (self.cell.0 + GUTTER), HEADER_H + GUTTER + r * (self.cell.1 + GUTTER))
    }

    /// Renders output frame `index` (recordings shorter than it hold their last frame).
    pub fn render(&self, index: usize, rasterizer: &mut Rasterizer) -> Result<Canvas, String> {
        let mut canvas = Canvas::new(self.width, self.height, PAGE);
        let t_now = self.recs[0].frames[index.min(self.recs[0].frames.len() - 1)].time;
        self.draw_header(&mut canvas, t_now);
        let frame_of = |p: &PanelSpec| {
            let rec = self.recs[p.rec];
            (rec, &rec.frames[index.min(rec.frames.len() - 1)])
        };
        for (i, p) in self.panels.iter().enumerate() {
            let (rec, frame) = frame_of(p);
            let scene = PanelScene {
                camera: &self.camera,
                mode: p.mode,
                scales: &self.scales,
                opts: &self.opts,
                ground: rec.scene.ground.as_ref().map(|g| g.height),
                bounds: self.bounds,
                bias: self.bias,
            };
            match rasterizer {
                Rasterizer::Cpu(raster) => {
                    scene.draw(raster, frame);
                    let (x, y) = self.cell_origin(i);
                    raster.resolve(&mut canvas, x, y, self.opts.ss);
                }
                #[cfg(feature = "gpu")]
                Rasterizer::Gpu(gpu) => scene.draw(&mut gpu.panels[i], frame),
            }
        }
        #[cfg(feature = "gpu")]
        if let Rasterizer::Gpu(gpu) = rasterizer {
            gpu.finish(&mut canvas, |i| self.cell_origin(i))?;
        }
        for (i, p) in self.panels.iter().enumerate() {
            let (_, frame) = frame_of(p);
            let (x, y) = self.cell_origin(i);
            let mut title = p.title.clone();
            if let Some(c) = &self.opts.clip {
                title = format!("{title} · {}", clip_label(c));
            }
            let room = view_width(self.cell.0).saturating_sub(24);
            canvas.label(x as i64 + 8, y as i64 + 8, &fit(&title, 2, room), 2, INK);
            self.draw_legend(&mut canvas, i, p.mode.legend(&self.scales), frame);
        }
        if let Some(plot) = &self.plot {
            plot::draw(&mut canvas, (0, (self.height - PLOT_H) as i64, self.width as i64, PLOT_H as i64), plot, t_now);
        }
        Ok(canvas)
    }

    fn draw_header(&self, c: &mut Canvas, t: f64) {
        c.fill_rect(0, 0, self.width as i64, HEADER_H as i64, [1.0; 3]);
        c.fill_rect(0, HEADER_H as i64 - 1, self.width as i64, 1, RULE);
        let w = self.width as i64;
        let t_end = self.recs.iter().map(|r| r.end_time()).fold(0.0, f64::max);
        let time = format!("t = {}", fmt_time(t, t_end));
        let speed = speed_label(self.sim_per_video_second);
        let right_w = font::text_width(&time, 3).max(font::text_width(&speed, 2)) as i64;
        c.text_right(w - 16, 10, &time, 3, INK);
        c.text_right(w - 16, 38, &speed, 2, MUTED);
        let room = (w - 16 - right_w - 40).max(0) as usize;
        c.text(16, 10, &fit(&self.title, 3, room), 3, INK);
        c.text(16, 38, &fit(&self.subtitle, 2, room), 2, MUTED);
    }

    fn draw_legend(&self, c: &mut Canvas, panel: usize, legend: Legend, frame: &Frame) {
        let (cx, cy) = self.cell_origin(panel);
        let x = (cx + self.cell.0 - LEGEND_W + 14) as i64;
        let mut y = cy as i64 + 12;
        let bar_w = 18;
        match legend {
            Legend::Gradient { map, lo, hi, unit, top_swatch } => {
                c.text(x, y, &unit, 2, INK);
                y += 24;
                if let Some((color, label)) = top_swatch {
                    c.fill_rect(x, y, bar_w, 14, color);
                    c.stroke_rect(x, y, bar_w, 14, MUTED);
                    c.text(x + bar_w + 8, y, &label, 2, INK);
                    y += 32;
                }
                let bar_h = ((cy + self.cell.1) as i64 - y - 16).clamp(30, 280);
                c.vertical_gradient(x, y, bar_w, bar_h, |t| map.sample(t));
                c.stroke_rect(x - 1, y - 1, bar_w + 2, bar_h + 2, MUTED);
                let ticks = plot::ticks(lo, hi, 4);
                for v in ticks {
                    let yy = y + bar_h - 1 - (((v - lo) / (hi - lo)) * (bar_h - 1) as f64).round() as i64;
                    c.fill_rect(x + bar_w, yy, 5, 1, MUTED);
                    c.text(x + bar_w + 9, yy - 7, &fmt_number(v), 2, INK);
                }
            }
            Legend::Fragments => {
                let fragments = frame.chunks.iter().map(|c| c.cluster).collect::<std::collections::BTreeSet<_>>().len();
                c.text(x, y, &format!("{fragments} piece{}", if fragments == 1 { "" } else { "s" }), 2, INK);
                y += 26;
                c.fill_rect(x, y, bar_w, 14, ANCHORED);
                c.stroke_rect(x, y, bar_w, 14, MUTED);
                c.text(x + bar_w + 8, y, "anchored", 2, INK);
                y += 22;
                for (k, id) in [11u64, 23, 37].iter().enumerate() {
                    c.fill_rect(x + k as i64 * 8, y, 8, 14, crate::colormap::categorical(*id));
                }
                c.stroke_rect(x, y, bar_w + 6, 14, MUTED);
                c.text(x + bar_w + 14, y, "free", 2, INK);
            }
        }
    }
}

/// A GPU context to rasterize on (uninhabited without the `gpu` feature).
#[cfg(feature = "gpu")]
pub type GpuHandle = std::sync::Arc<GpuContext>;
#[cfg(not(feature = "gpu"))]
pub type GpuHandle = std::convert::Infallible;

/// Where a render thread rasterizes the 3-D panels.
pub enum Rasterizer {
    Cpu(Raster),
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuFrame>),
}

/// Columns of a cell left of the colour-bar strip (where the 3-D view is fitted).
fn view_width(cell_w: usize) -> usize {
    cell_w.saturating_sub(LEGEND_W).max(16)
}

/// Rows of a cell kept free for its title (at most a quarter of the cell).
fn title_band(cell_h: usize) -> usize {
    TITLE_BAND.min(cell_h / 4)
}

/// "×120 slow motion", "real time" or "×4 faster".
pub fn speed_label(sim_per_video_second: f64) -> String {
    let k = 1.0 / sim_per_video_second.max(1e-300);
    if (k - 1.0).abs() < 0.05 {
        "real time".into()
    } else if k > 1.0 {
        format!("×{} slow motion", fmt_number(round_sig(k)))
    } else {
        format!("×{} faster than real time", fmt_number(round_sig(1.0 / k)))
    }
}

/// Rounds to two significant digits.
fn round_sig(x: f64) -> f64 {
    let p = 10f64.powf(x.abs().log10().floor() - 1.0);
    (x / p).round() * p
}

/// Truncates `text` with "..." to fit `room` pixels at `scale`.
fn fit(text: &str, scale: usize, room: usize) -> String {
    if font::text_width(text, scale) <= room {
        return text.to_string();
    }
    let max_chars = (room / (font::ADVANCE * scale)).saturating_sub(3);
    format!("{}...", text.chars().take(max_chars).collect::<String>())
}

/// Grid columns for `n` panels: a row for up to three, else a near-square grid.
pub fn columns_for(n: usize) -> usize {
    if n <= 3 {
        n.max(1)
    } else {
        (n as f64).sqrt().ceil() as usize
    }
}

/// The plot strip for `names`: `energy` (default), `broken`, or scene probe names.
/// With several recordings (compare mode), each gets its own colour.
pub fn build_plot(recs: &[&Recording], labels: &[String], names: &[String]) -> Result<Option<Plot>, String> {
    if names.iter().any(|n| n == "none") {
        return Ok(None);
    }
    let t_end = recs.iter().map(|r| r.end_time()).fold(0.0, f64::max);
    let multi = recs.len() > 1;
    let mut series = Vec::new();
    let mut left_unit = String::new();
    let mut right_unit = String::new();
    let probes = names.iter().filter(|n| !matches!(n.as_str(), "energy" | "broken")).count();
    for (pi, name) in names.iter().enumerate() {
        for (k, rec) in recs.iter().enumerate() {
            let prefix = if multi { format!("{}: ", labels[k]) } else { String::new() };
            let color = |i: usize| if multi { PALETTE[k % PALETTE.len()] } else { PALETTE[i % PALETTE.len()] };
            let ts: Vec<f64> = rec.frames.iter().map(|f| f.time).collect();
            let broken = |dashed| Series {
                label: format!("{prefix}broken bonds"),
                color: if multi { color(0) } else { mix(INK, [1.0; 3], 0.25) },
                right: true,
                dashed,
                t: ts.clone(),
                v: rec.frames.iter().map(|f| f.broken_bonds as f64).collect(),
            };
            match name.as_str() {
                "energy" => {
                    let e0 = rec.frames[0].mechanical;
                    if !multi {
                        series.push(Series {
                            label: "Δ mechanical energy".into(),
                            color: color(0),
                            right: false,
                            dashed: false,
                            t: ts.clone(),
                            v: rec.frames.iter().map(|f| f.mechanical - e0).collect(),
                        });
                    }
                    series.push(Series {
                        label: format!("{prefix}dissipated"),
                        color: color(1),
                        right: false,
                        dashed: false,
                        t: ts.clone(),
                        v: rec.frames.iter().map(|f| f.dissipated).collect(),
                    });
                    series.push(broken(multi));
                    left_unit = "energy [J]".into();
                    right_unit = "broken bonds".into();
                }
                "broken" => {
                    series.push(broken(false));
                    right_unit = "broken bonds".into();
                }
                probe => {
                    let (s, unit) = rec.probes.get(probe).ok_or_else(|| {
                        let known: Vec<&str> = rec.probes.keys().map(String::as_str).collect();
                        format!("--plot {probe}: no such probe (energy, broken, none{}{})", if known.is_empty() { "" } else { ", " }, known.join(", "))
                    })?;
                    // Compare mode: colour per variant, dashes tell further probes apart.
                    series.push(Series {
                        label: format!("{prefix}{probe}"),
                        color: color(series.len()),
                        right: false,
                        dashed: multi && pi > 0,
                        t: s.t.clone(),
                        v: s.v.clone(),
                    });
                    left_unit = if probes > 1 { format!("[{unit}]") } else { format!("{probe} [{unit}]") };
                }
            }
        }
    }
    Ok(Some(Plot { series, left_unit, right_unit, t_end }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_and_grid() {
        assert_eq!(speed_label(1.0 / 120.0), "×120 slow motion");
        assert_eq!(speed_label(1.0), "real time");
        assert_eq!(speed_label(4.0), "×4 faster than real time");
        assert_eq!(speed_label(1.0 / 8333.3), "×8300 slow motion");
        assert_eq!(columns_for(1), 1);
        assert_eq!(columns_for(3), 3);
        assert_eq!(columns_for(4), 2);
        assert_eq!(columns_for(6), 3);
        assert_eq!(fit("abcdefghij", 1, 6 * 6), "abc...");
        assert_eq!(fit("abc", 1, 100), "abc");
    }
}
