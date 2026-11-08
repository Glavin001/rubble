//! Generic generators for the non-panel parts (port of `vehgen/parts.py`).
//! Each returns a Mesh in glTF coordinates; designs choose styles and
//! placements. Line-for-line with the Python, including its quirks (see the
//! notes on `rim_magnum`, `brake_disc` and `steering_wheel`), so the two
//! generators can be compared output for output.

use std::f64::consts::PI;

use glam::{DMat3, DVec2, DVec3};

use crate::curves::{resample_polyline2, sign};
use crate::mesh::{
    box_, catmull, catmull2, closed_lathe, gradient2, lathe, loft, orient_outward, rbox, rot_x, rot_y, rrect, sweep, zigzag_cap, Grid2,
    Mat, Mesh, SweepOpts,
};
use crate::tube::Tube;

/// Lathe axis +X -> +Z (forward).
pub fn ax_to_z() -> DMat3 {
    rot_y(-PI / 2.0)
}

/// Lathe axis +X -> -Z (rearward).
pub fn ax_to_mz() -> DMat3 {
    rot_y(PI / 2.0)
}

/// `np.linspace(a, b, n)` exactly as numpy computes it (`i * step + a`,
/// the last sample set to b), so samples match bit for bit.
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    let mut out: Vec<f64> = (0..n).map(|i| i as f64 * step + a).collect();
    out[n - 1] = b;
    out
}

/// `np.linspace(0, b, n, endpoint=False)`.
fn linspace_open(b: f64, n: usize) -> Vec<f64> {
    let step = b / n as f64;
    (0..n).map(|i| i as f64 * step).collect()
}

/// `mesh.circle(r, n)` centred at the origin, with numpy's sample angles
/// (`k * (2 pi / n)`); `mesh::circle` computes `2 pi k / n`, which differs in
/// the last bit and can flip ear-clipping ties in the sweep caps.
fn np_circle(r: f64, n: usize) -> Vec<DVec2> {
    linspace_open(2.0 * PI, n).into_iter().map(|a| DVec2::new(r * a.cos(), r * a.sin())).collect()
}

// -- wheels ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RimStyle {
    Magnum,
    Alloy,
    Steel,
}

/// The wheel fields of a design (Python `D.TRACK`, `D.WHEEL_Y`, ...).
#[derive(Clone, Copy, Debug)]
pub struct WheelSpec {
    pub track: f64,
    pub wheel_y: f64,
    pub tyre_d: f64,
    pub tyre_w: f64,
    pub rim_d: f64,
    pub rim_w: f64,
    pub rim_style: RimStyle,
    pub disc_d: f64,
}

/// Closed tyre: beads seated inside the rim flanges, sidewalls bulging to
/// the section width, rounded shoulders, circumferential grooves.
pub fn tyre(d_out: f64, w: f64, rim_d: f64, rim_w: f64, segs: usize, grooves: usize) -> Mesh {
    let big_r = d_out / 2.0;
    let r = rim_d / 2.0;
    let hw = w / 2.0;
    let hb = rim_w / 2.0 - 0.015; // bead axial position, inside the flanges
    let rb = r - 0.0015; // bead seat just over the barrel
    let rf = r + 0.026; // clears the flange top
    let mut prof = vec![(rb, -hb), (rf, -hb)];
    for &t in &linspace(0.0, 1.0, 9)[1..] {
        let rr = rf + (big_r - 0.030 - rf) * t;
        let a = -hb - (hw - hb) * (PI / 2.0 * (t * 1.6).min(1.0)).sin() + 0.010 * t * t;
        prof.push((rr, a));
    }
    for &k in &linspace(0.0, 1.0, 5)[1..] {
        let th = PI / 2.0 * k;
        prof.push((big_r - 0.030 + 0.030 * th.sin(), -hw + 0.010 + 0.022 - 0.022 * th.cos()));
    }
    let gw = 0.008;
    prof.push((big_r, -hw + 0.034));
    for g in 0..grooves {
        let c = -hw + 0.034 + (w - 0.068) * (g + 1) as f64 / (grooves + 1) as f64;
        prof.extend([(big_r, c - gw), (big_r - 0.008, c - gw * 0.6), (big_r - 0.008, c + gw * 0.6), (big_r, c + gw)]);
    }
    prof.push((big_r, hw - 0.034));
    let ks = linspace(0.0, 1.0, 5);
    for &k in &ks[..ks.len() - 1] {
        let th = PI / 2.0 * (1.0 - k);
        prof.push((big_r - 0.030 + 0.030 * th.sin(), hw - 0.010 - 0.022 + 0.022 * th.cos()));
    }
    for &t in &linspace(1.0, 0.0, 9)[1..] {
        let rr = rf + (big_r - 0.030 - rf) * t;
        let a = hb + (hw - hb) * (PI / 2.0 * (t * 1.6).min(1.0)).sin() - 0.010 * t * t;
        prof.push((rr, a));
    }
    prof.extend([(rf, hb), (rb, hb)]);
    closed_lathe(&prof, segs, Mat::Rubber)
}

/// `np.arctan2(*p[::-1])`: the angle of the 2D point p.
#[inline]
fn ang2(p: DVec2) -> f64 {
    p.y.atan2(p.x)
}

/// Late-60s polished five-spoke: a chrome dished face with five dark window
/// cut-outs (the spokes are the chrome left between them), a narrow polished
/// lip and a small domed centre cap.
pub fn rim_magnum(rim_d: f64, rim_w: f64, segs: usize, spokes: usize) -> Mesh {
    let r = rim_d / 2.0;
    let hw = rim_w / 2.0;
    let mut m = Mesh::new();
    let barrel = [
        (r + 0.012, -hw - 0.006),
        (r + 0.012, -hw + 0.004),
        (r - 0.004, -hw + 0.008),
        (r - 0.010, -hw + 0.030),
        (r - 0.010, hw - 0.020),
        (r - 0.004, hw - 0.006),
        (r - 0.008, hw - 0.006),
        (r - 0.014, hw - 0.020),
        (r - 0.014, -hw + 0.030),
        (r - 0.008, -hw + 0.006),
    ];
    m.add(&closed_lathe(&barrel, segs, Mat::MetalBare));
    let lip = [
        (r + 0.014, hw - 0.010),
        (r + 0.016, hw + 0.002),
        (r + 0.008, hw + 0.010),
        (r - 0.026, hw + 0.004),
        (r - 0.030, hw - 0.004),
        (r - 0.010, hw - 0.012),
    ];
    m.add(&closed_lathe(&lip, segs, Mat::Chrome));
    let (r0, r1) = (0.046, r - 0.033);
    let (a_in, a_out) = (hw - 0.024, hw - 0.004);
    let ax = |rr: f64| a_in + (a_out - a_in) * ((rr - r0) / (r1 - r0)).clamp(0.0, 1.0).powf(0.8);
    let dish = [(r0, a_in), (r1, a_out), (r1, a_out - 0.006), (r0, a_in - 0.006)];
    m.add(&closed_lathe(&dish, segs, Mat::Chrome));
    let cap = [(0.0, a_in + 0.020), (0.022, a_in + 0.018), (0.040, a_in + 0.008), (0.048, a_in - 0.002), (0.0, a_in - 0.002)];
    m.add(&closed_lathe(&cap, 48, Mat::Chrome));
    // windows: between spoke centre lines, spokes 30 mm wide at the cap and
    // 44 mm at the rim
    let (ri, ro) = (r0 + 0.016, r1 - 0.012);
    // left edge (offset from spoke at c0), outer arc, right edge, inner arc;
    // the point at radius rr along the line offset by wid from the spoke
    let edge = |c: f64, sgn: f64, rr: f64, wid: f64| -> DVec2 {
        let d = DVec2::new(c.cos(), c.sin());
        let n = DVec2::new(-c.sin(), c.cos()) * sgn;
        let t = (rr * rr - wid * wid).max(1e-9).sqrt();
        d * t + n * wid
    };
    for k in 0..spokes {
        let c0 = 2.0 * PI * k as f64 / spokes as f64;
        let c1 = 2.0 * PI * (k + 1) as f64 / spokes as f64;
        // (the Python has a dead `for rr, wid in ...: pass` loop here)
        let mut poly: Vec<DVec2> = Vec::new();
        for (rr, wid) in [(ri, 0.015), (ro, 0.022)] {
            poly.push(edge(c0, 1.0, rr, wid));
        }
        let a0_ = ang2(*poly.last().unwrap());
        let mut a1_ = ang2(edge(c1, -1.0, ro, 0.022));
        if a1_ < a0_ {
            a1_ += 2.0 * PI;
        }
        let la = linspace(a0_, a1_, 9);
        for &a_ in &la[1..la.len() - 1] {
            poly.push(DVec2::new(ro * a_.cos(), ro * a_.sin()));
        }
        poly.push(edge(c1, -1.0, ro, 0.022));
        poly.push(edge(c1, -1.0, ri, 0.015));
        let b0_ = ang2(*poly.last().unwrap());
        let mut b1_ = ang2(poly[0]);
        if b1_ > b0_ {
            b1_ -= 2.0 * PI;
        }
        let lb = linspace(b0_, b1_, 5);
        for &a_ in &lb[1..lb.len() - 1] {
            poly.push(DVec2::new(ri * a_.cos(), ri * a_.sin()));
        }
        // a single dark panel just proud of the dish (open surface)
        let front: Vec<DVec3> = poly.iter().map(|p| DVec3::new(ax(p.length()) + 0.0010, p.x, p.y)).collect();
        let ring: Vec<u32> = (0..front.len() as u32).collect();
        let tris = zigzag_cap(&ring, &front, DVec3::X);
        m.add(&Mesh::from_faces(front, vec![], tris, Mat::TrimBlack, None, -1));
    }
    m
}

/// Modern cast wheel: five split spokes in a concave face, polished lip.
pub fn rim_alloy(rim_d: f64, rim_w: f64, segs: usize, pairs: usize) -> Mesh {
    let r = rim_d / 2.0;
    let hw = rim_w / 2.0;
    let mut m = Mesh::new();
    let barrel = [
        (r + 0.012, -hw - 0.006),
        (r + 0.012, -hw + 0.004),
        (r - 0.004, -hw + 0.008),
        (r - 0.012, -hw + 0.040),
        (r - 0.012, hw - 0.020),
        (r - 0.004, hw - 0.006),
        (r - 0.009, hw - 0.006),
        (r - 0.017, hw - 0.020),
        (r - 0.017, -hw + 0.040),
        (r - 0.009, -hw + 0.006),
    ];
    m.add(&closed_lathe(&barrel, segs, Mat::MetalBare));
    let lip = [
        (r + 0.013, hw - 0.010),
        (r + 0.015, hw + 0.002),
        (r + 0.004, hw + 0.008),
        (r - 0.018, hw + 0.004),
        (r - 0.020, hw - 0.004),
        (r - 0.008, hw - 0.012),
    ];
    m.add(&closed_lathe(&lip, segs, Mat::Chrome));
    let hub = [(0.0, hw - 0.020), (0.030, hw - 0.021), (0.065, hw - 0.028), (0.075, hw - 0.040), (0.075, hw - 0.052), (0.0, hw - 0.052)];
    m.add(&closed_lathe(&hub, 40, Mat::Chrome));
    m.add(&closed_lathe(&[(0.0, hw - 0.018), (0.026, hw - 0.018), (0.028, hw - 0.022), (0.0, hw - 0.022)], 24, Mat::TrimBlack));
    let section = rrect(1.0, 0.016, 0.005, 2);
    for k in 0..pairs {
        for d in [-1.0f64, 1.0] {
            let a = 2.0 * PI * k as f64 / pairs as f64 + d * 0.11;
            let ts = linspace(0.0, 1.0, 8);
            let path: Vec<DVec3> = ts
                .iter()
                .map(|&t| {
                    let rr = 0.060 + (r - 0.020 - 0.060) * t;
                    let aa = a - d * 0.07 * t;
                    let axp = hw - 0.040 + 0.034 * t.powf(0.7);
                    DVec3::new(axp, rr * aa.cos(), rr * aa.sin())
                })
                .collect();
            let sx = linspace(0.026, 0.020, path.len());
            let sc: Vec<DVec2> = sx.iter().map(|&s| DVec2::new(s, 1.0)).collect();
            let o = SweepOpts { up: DVec3::X, scale: Some(sc), twist_up: Some(vec![DVec3::X; path.len()]), ..Default::default() };
            let sp = sweep(&path, &section, Mat::Chrome, &o);
            m.add(&orient_outward(sp));
        }
    }
    for k in 0..5 {
        let a = 2.0 * PI * (k as f64 + 0.5) / 5.0;
        let mut nut = closed_lathe(&[(0.0, 0.006), (0.008, 0.005), (0.009, -0.006), (0.0, -0.006)], 12, Mat::Chrome);
        nut.transform(None, DVec3::new(hw - 0.026, 0.045 * a.cos(), 0.045 * a.sin()));
        m.add(&nut);
    }
    m
}

/// Painted steel wheel with a small hub cap (work trucks).
pub fn rim_steel(rim_d: f64, rim_w: f64, segs: usize) -> Mesh {
    let r = rim_d / 2.0;
    let hw = rim_w / 2.0;
    let mut m = Mesh::new();
    let prof = [
        (r + 0.012, -hw - 0.006),
        (r + 0.012, -hw + 0.004),
        (r - 0.006, -hw + 0.008),
        (r - 0.010, hw - 0.030),
        (r - 0.002, hw - 0.012),
        (r + 0.012, hw - 0.006),
        (r + 0.012, hw + 0.004),
        (r - 0.012, hw - 0.002),
        (r - 0.016, hw - 0.030),
        (r - 0.040, hw - 0.034),
        (0.11, hw - 0.010),
        (0.10, hw - 0.020),
        (r - 0.044, hw - 0.046),
        (r - 0.020, hw - 0.044),
        (r - 0.014, -hw + 0.010),
        (r - 0.010, -hw + 0.006),
    ];
    m.add(&closed_lathe(&prof, segs, Mat::MetalBare));
    m.add(&closed_lathe(&[(0.0, hw + 0.006), (0.060, hw + 0.002), (0.085, hw - 0.012), (0.0, hw - 0.012)], 40, Mat::Chrome));
    m
}

pub fn brake_disc(d: f64, segs: usize) -> Mesh {
    let r = d / 2.0;
    let mut m = closed_lathe(&[(0.075, -0.052), (r, -0.052), (r, -0.028), (0.075, -0.028)], segs, Mat::MetalBare);
    m.add(&closed_lathe(&[(0.040, -0.030), (0.085, -0.030), (0.085, 0.004), (0.040, 0.004)], 32, Mat::MetalBare));
    // (the Python also builds a caliper `cal` here and never uses it; it is an
    // rbox, which takes no UV island, so skipping it changes nothing)
    // place on the disc's rim, wrapping it
    let mut cal2 = rbox(DVec3::new(-0.040, r - 0.022, 0.0), DVec3::new(0.024, 0.030, 0.060), 0.010, 2, Mat::TrimBlack);
    cal2.transform(Some(rot_x((-120.0f64).to_radians())), DVec3::ZERO);
    m.add(&cal2);
    m
}

/// (rim, tyre, disc, centre) for one corner. side +1 left, -1 right.
pub fn wheel(spec: &WheelSpec, side: f64, z_axle: f64) -> (Mesh, Mesh, Mesh, DVec3) {
    let x = side * spec.track / 2.0;
    let c = DVec3::new(x, spec.wheel_y, z_axle);
    let ty = tyre(spec.tyre_d, spec.tyre_w, spec.rim_d, spec.rim_w, 96, 3);
    let ri = match spec.rim_style {
        RimStyle::Magnum => rim_magnum(spec.rim_d, spec.rim_w, 72, 5),
        RimStyle::Alloy => rim_alloy(spec.rim_d, spec.rim_w, 72, 5),
        RimStyle::Steel => rim_steel(spec.rim_d, spec.rim_w, 64),
    };
    let di = brake_disc(spec.disc_d, 48);
    let mut out = [ri, ty, di];
    for mm in &mut out {
        if side < 0.0 {
            mm.mirror_x();
        }
        mm.transform(None, c);
    }
    let [ri, ty, di] = out;
    (ri, ty, di, c)
}

// -- lamps --------------------------------------------------------------------------

/// Rotation taking +X to `axis` (up kept as close to +Y as possible).
pub fn rot_to(axis: DVec3) -> DMat3 {
    let a = axis / axis.length();
    let up = DVec3::Y;
    let mut z = a.cross(up);
    if z.length() < 1e-6 {
        z = DVec3::Z;
    }
    z /= z.length();
    let y = z.cross(a);
    DMat3::from_cols(a, y, z)
}

/// Outward normal of the end face z = fz(x, y).
pub fn graph_normal(fz: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, facing: f64) -> DVec3 {
    let h = 1e-4;
    let fx = (fz(x + h, y) - fz(x - h, y)) / (2.0 * h);
    let fy = (fz(x, y + h) - fz(x, y - h)) / (2.0 * h);
    let n = DVec3::new(-fx, -fy, 1.0) * facing;
    n / n.length()
}

/// (lens, housing) of a round sealed-beam lamp facing +z (facing=1) or
/// along `axis`.
pub fn round_lamp(center: DVec3, r: f64, facing: f64, tint: [f64; 4], depth: f64, axis: Option<DVec3>) -> (Mesh, Mesh) {
    let depth = depth.max(0.014);
    let rot = match axis {
        Some(a) => rot_to(a),
        None => {
            if facing > 0.0 {
                ax_to_z()
            } else {
                ax_to_mz()
            }
        }
    };
    let rl = r - 0.005;
    let mut lens =
        closed_lathe(&[(0.0, 0.016), (rl * 0.5, 0.013), (rl * 0.85, 0.007), (rl, 0.0), (rl, -0.004), (0.0, -0.004)], 48, Mat::LampLens);
    lens.transform(Some(rot), center);
    lens.tint(tint);
    let mut hous = Mesh::new();
    let bezel = closed_lathe(
        &[(r - 0.002, 0.002), (r + 0.010, 0.006), (r + 0.016, 0.0), (r + 0.014, -0.010), (r - 0.002, -0.008)],
        48,
        Mat::Chrome,
    );
    let cup = closed_lathe(
        &[
            (r - 0.002, -0.008),
            (r * 0.7, -depth * 0.6),
            (0.02, -depth),
            (0.0, -depth),
            (0.0, -depth - 0.004),
            (0.02, -depth - 0.004),
            (r * 0.7 + 0.004, -depth * 0.6 - 0.004),
            (r + 0.002, -0.010),
        ],
        40,
        Mat::LampEmissive,
    );
    hous.add(&bezel).add(&cup);
    hous.transform(Some(rot), center);
    (lens, hous)
}

// -- helpers ----------------------------------------------------------------------

pub fn surface_point(tube: &Tube, z: f64, y: f64, side: f64, vrange: (f64, f64)) -> (DVec3, DVec3) {
    let mut v = tube.v_at_height(z, y, vrange.0, vrange.1);
    if side < 0.0 {
        v = tube.vm - v;
    }
    tube.eval_n(z, v)
}

/// `ys` may hold one value (broadcast over `zs`) or one per z.
pub fn path_on_surface(tube: &Tube, zs: &[f64], ys: &[f64], side: f64, off: f64, vrange: (f64, f64)) -> (Vec<DVec3>, Vec<DVec3>) {
    assert!(ys.len() == 1 || ys.len() == zs.len());
    let mut ps = Vec::with_capacity(zs.len());
    let mut ns = Vec::with_capacity(zs.len());
    for (i, &z) in zs.iter().enumerate() {
        let y = if ys.len() == 1 { ys[0] } else { ys[i] };
        let mut v = tube.v_at_height(z, y, vrange.0, vrange.1);
        if side < 0.0 {
            v = tube.vm - v;
        }
        let (p, n) = tube.eval_n(z, v);
        ps.push(p + n * off);
        ns.push(n);
    }
    (ps, ns)
}

pub fn door_handle(tube: &Tube, z: f64, y: f64, side: f64, length: f64) -> Mesh {
    let (p, n) = surface_point(tube, z, y, side, (1.5, 5.5));
    let n = n / n.length();
    let ry = rot_y((-n.z).atan2(n.x));
    let mut m = rbox(DVec3::ZERO, DVec3::new(0.009, 0.013, length / 2.0), 0.006, 2, Mat::Chrome);
    let mut plate = rbox(DVec3::ZERO, DVec3::new(0.0015, 0.020, length / 2.0 + 0.012), 0.0014, 1, Mat::TrimBlack);
    m.transform(Some(ry), p + n * 0.019);
    plate.transform(Some(ry), p + n * 0.0060);
    m.with(plate)
}

/// Bullet racing mirror: pad on the door top, short stem, rounded head.
pub fn sport_mirror(base: DVec3, side: f64, mat: Mat) -> Mesh {
    let b = base;
    let mut m = Mesh::new();
    let pad = rbox(b + DVec3::new(-side * 0.006, 0.017, 0.0), DVec3::new(0.018, 0.005, 0.038), 0.004, 2, mat);
    let stem = sweep(
        &[b + DVec3::new(0.0, 0.020, 0.0), b + DVec3::new(side * 0.025, 0.050, -0.005), b + DVec3::new(side * 0.050, 0.070, -0.010)],
        &np_circle(0.008, 12),
        Mat::Chrome,
        &SweepOpts::default(),
    );
    let head_c = b + DVec3::new(side * 0.080, 0.085, -0.015);
    let head = rbox(head_c, DVec3::new(0.050, 0.036, 0.042), 0.034, 4, mat);
    let glass = rbox(head_c + DVec3::new(0.0, 0.0, -0.043), DVec3::new(0.040, 0.027, 0.0015), 0.0014, 1, Mat::Chrome);
    m.add(&pad).add(&orient_outward(stem)).add(&head).add(&glass);
    m
}

/// Arm + blade along a polyline lying on the glass.
pub fn wiper(points: &[DVec3], mat: Mat) -> Mesh {
    let arm = sweep(points, &rrect(0.012, 0.006, 0.002, 1), mat, &SweepOpts::default());
    let p2: Vec<DVec3> = points.iter().map(|p| *p + DVec3::new(0.0, -0.004, 0.0)).collect();
    let blade = sweep(&p2, &rrect(0.008, 0.010, 0.002, 1), Mat::Rubber, &SweepOpts::default());
    orient_outward(arm).with(orient_outward(blade))
}

// -- interior -----------------------------------------------------------------------

/// Low-back bucket with an integrated head restraint (late-60s style).
pub fn bucket_seat(x: f64, z: f64, h_floor: f64, mat: Mat, high_back: bool) -> Mesh {
    let mut m = Mesh::new();
    let top = h_floor + 0.175;
    m.add(&rbox(DVec3::new(x, top - 0.065, z), DVec3::new(0.25, 0.065, 0.27), 0.05, 3, mat));
    let back_h = 0.27;
    let mut back = rbox(DVec3::ZERO, DVec3::new(0.25, back_h, 0.06), 0.05, 3, mat);
    let lean = (-14.0f64).to_radians();
    back.transform(Some(rot_x(lean)), DVec3::new(x, top + back_h - 0.03, z - 0.30));
    m.add(&back);
    if high_back {
        let mut hr = rbox(DVec3::ZERO, DVec3::new(0.13, 0.085, 0.05), 0.04, 2, mat);
        hr.transform(Some(rot_x(lean)), DVec3::new(x, top + 2.0 * back_h + 0.05, z - 0.30 - 0.15));
        m.add(&hr);
    }
    m.add(&box_(DVec3::new(x, h_floor + 0.035, z), DVec3::new(0.20, 0.03, 0.22), Mat::MetalBare, None));
    m
}

pub fn steering_wheel(center: DVec3, r: f64, tilt_deg: f64, mat: Mat, column: f64) -> Mesh {
    // (the Python first builds a diamond-section rim and immediately
    // overwrites it; that closed_lathe takes no UV island, so it is skipped)
    let rim_prof: Vec<(f64, f64)> = linspace_open(2.0 * PI, 12).into_iter().map(|a| (r + 0.014 * a.cos(), 0.014 * a.sin())).collect();
    let rim = closed_lathe(&rim_prof, 64, mat);
    let hub = closed_lathe(&[(0.0, 0.02), (0.05, 0.015), (0.055, -0.03), (0.0, -0.03)], 24, mat);
    let mut m = Mesh::new().with(rim).with(hub);
    let sec = rrect(0.022, 0.008, 0.003, 1);
    for a in [(-90.0f64).to_radians(), 30.0f64.to_radians(), 150.0f64.to_radians()] {
        let p0 = DVec3::new(0.0, 0.05 * a.cos(), 0.05 * a.sin());
        let p1 = DVec3::new(0.006, (r - 0.010) * a.cos(), (r - 0.010) * a.sin());
        // np.linspace(p0, p1, 4): i * step + p0, last sample exactly p1
        let step = (p1 - p0) / 3.0;
        let path: Vec<DVec3> = (0..4).map(|i| if i == 3 { p1 } else { step * i as f64 + p0 }).collect();
        let sp = sweep(&path, &sec, Mat::Chrome, &SweepOpts { up: DVec3::X, ..Default::default() });
        m.add(&orient_outward(sp));
    }
    let col = closed_lathe(&[(0.0, -0.02), (0.032, -0.02), (0.032, -column), (0.0, -column)], 20, mat);
    m.add(&col);
    // axis +X -> towards the driver (-Z), tilted up
    let r1 = ax_to_mz();
    let r2 = rot_x(tilt_deg.to_radians());
    m.transform(Some(r2 * r1), center);
    m
}

// -- underbody and bay ---------------------------------------------------------------

/// Half-drum liner over a wheel: roof (cylinder) + inner wall. Angles in
/// degrees.
pub fn wheel_well(center: DVec3, r: f64, x_in: f64, x_out: f64, side: f64, a_front: f64, a_rear: f64) -> Mesh {
    let prof = [(r, x_out), (r, x_in), (r * 0.55, x_in)];
    let mut m = lathe(&prof, 40, Mat::TrimBlack, (-a_rear).to_radians(), a_front.to_radians());
    // lathe: axis x, angle 0 -> +y, +90 -> +z. Profile above runs outward ->
    // inward; faces should look at the wheel (inwards).
    m.weld(1e-7);
    if side < 0.0 {
        m.mirror_x();
    }
    // make faces point towards the axle
    let tri = m.triangles();
    let mut s = 0.0;
    for f in &tri {
        let (a, b, cc) = (m.v[f[0] as usize], m.v[f[1] as usize], m.v[f[2] as usize]);
        let fnrm = (b - a).cross(cc - a);
        let cen = (a + b + cc) / 3.0;
        let to_axis = -cen * DVec3::new(0.0, 1.0, 1.0);
        s += fnrm.dot(to_axis);
    }
    // np.mean(...) < 0 (an empty mean is NaN: no flip)
    if s / (tri.len() as f64) < 0.0 {
        m.flip();
    }
    m.transform(None, center);
    m
}

pub fn tube_pipe(points: &[DVec3], r: f64, mat: Mat, n: usize, segs: usize) -> Mesh {
    let p = catmull(points, n, false);
    orient_outward(sweep(&p, &np_circle(r, segs), mat, &SweepOpts::default()))
}

/// Plan outline (x >= 0) of the body at height y round one end, as (x, z)
/// points: the end face z = fz(x, y) from the centre line to the corner, then
/// along the side (the tube at height y) away from the end. sign +1 front,
/// -1 rear.
#[allow(clippy::too_many_arguments)]
pub fn plan_outline(
    tube: &Tube,
    fz: &dyn Fn(f64, f64) -> f64,
    y: f64,
    sign: f64,
    z_guess: f64,
    side_len: f64,
    n_end: usize,
    n_side: usize,
) -> Vec<DVec2> {
    let xs = |z: f64| {
        let v = tube.v_at_height(z, y, 1.0, 5.0);
        tube.eval(z, v).x
    };
    let mut z = z_guess;
    for _ in 0..30 {
        let x = xs(z);
        z = fz(x, y);
    }
    let xc = xs(z);
    let mut out: Vec<DVec2> = linspace(0.0, xc, n_end).into_iter().map(|xe| DVec2::new(xe, fz(xe, y))).collect();
    for &s in &linspace(0.0, side_len, n_side + 1)[1..] {
        let zsd = z - sign * s;
        out.push(DVec2::new(xs(zsd), zsd));
    }
    out
}

/// Offset a centre->corner->side outline outward by d (2D), keeping the
/// distance at corners (miter, capped).
pub fn offset_outline(p: &[DVec2], d: f64, sign: f64) -> Vec<DVec2> {
    let n = p.len();
    let sn: Vec<DVec2> = p
        .windows(2)
        .map(|w| {
            let s = w[1] - w[0];
            let s = s / s.length();
            DVec2::new(-s.y, s.x) * sign
        })
        .collect();
    let mut vn = vec![DVec2::ZERO; n];
    for i in 0..n - 1 {
        vn[i] += sn[i];
    }
    for i in 1..n {
        vn[i] += sn[i - 1];
    }
    for v in &mut vn {
        *v /= v.length();
    }
    let mut sc = vec![1.0; n];
    for i in 1..n.saturating_sub(1) {
        sc[i] = 1.0 / vn[i].dot(sn[i]).clamp(0.6, 1.0);
    }
    (0..n).map(|i| p[i] + vn[i] * (d * sc[i])).collect()
}

pub fn chaikin(p: &[DVec2], it: usize) -> Vec<DVec2> {
    let mut p = p.to_vec();
    for _ in 0..it {
        let mut q = vec![p[0]];
        for w in p.windows(2) {
            let (a, b) = (w[0], w[1]);
            q.push(a * 0.75 + b * 0.25);
            q.push(a * 0.25 + b * 0.75);
        }
        q.push(*p.last().unwrap());
        p = q;
    }
    p
}

/// Distance from points q to the polyline segments a[i] -> b[i] (minimum).
fn seg_dist(q: &[DVec2], a: &[DVec2], b: &[DVec2]) -> Vec<f64> {
    q.iter()
        .map(|&qq| {
            let mut best = f64::INFINITY;
            for (&aa, &bb) in a.iter().zip(b) {
                let ab = bb - aa;
                let l2 = (ab.x * ab.x + ab.y * ab.y).max(1e-12);
                let t = ((qq - aa).dot(ab) / l2).clamp(0.0, 1.0);
                let c = aa + ab * t;
                best = best.min((qq - c).length());
            }
            best
        })
        .collect()
}

/// Closed 2D polygon with no edge longer than maxlen.
pub fn densify(poly: &[DVec2], maxlen: f64) -> Vec<DVec2> {
    let n = poly.len();
    let mut out = Vec::new();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let k = 1usize.max(((b - a).length() / maxlen).ceil() as usize);
        for j in 0..k {
            let t = j as f64 / k as f64;
            out.push(a + (b - a) * t);
        }
    }
    out
}

/// A smooth path at least d outside every outline (2D polylines, the body's
/// plan at several heights): miter offset of the middle one, rounded by
/// Chaikin, pushed out wherever it comes closer than d. `step` None (or 0,
/// Python-falsy) skips the final resampling.
pub fn fit_path(outlines: &[Vec<DVec2>], d: f64, sign: f64, it: usize, step: Option<f64>) -> Vec<DVec2> {
    let mut base = offset_outline(&outlines[outlines.len() / 2], d, sign);
    for _ in 0..it {
        base = chaikin(&base, 1);
        let t = gradient2(&base);
        let nrm: Vec<DVec2> = t
            .iter()
            .map(|tt| {
                let tt = *tt / tt.length();
                DVec2::new(-tt.y, tt.x) * sign
            })
            .collect();
        let mut dist = vec![f64::INFINITY; base.len()];
        for o in outlines {
            let dd = seg_dist(&base, &o[..o.len() - 1], &o[1..]);
            for (a, b) in dist.iter_mut().zip(dd) {
                *a = a.min(b);
            }
        }
        base = (0..base.len()).map(|i| base[i] + nrm[i] * (d - dist[i]).max(0.0)).collect();
    }
    if let Some(step) = step {
        if step != 0.0 {
            let l: f64 = base.windows(2).map(|w| (w[1] - w[0]).length()).sum();
            base = resample_polyline2(&base, 2usize.max((l / step).ceil() as usize));
        }
    }
    base
}

/// A chrome strip along tube parameters zv = (z, v), off the skin by `off`.
pub fn molding(tube: &Tube, zv: &[DVec2], off: f64, sec: (f64, f64), closed: bool, n: usize) -> Mesh {
    let pts = catmull2(zv, n, closed);
    let mut p = Vec::with_capacity(pts.len());
    let mut nn = Vec::with_capacity(pts.len());
    for q in &pts {
        let (pp, nrm) = tube.eval_n(q.x, q.y);
        p.push(pp + nrm * off);
        nn.push(nrm);
    }
    let section = rrect(sec.0, sec.1, sec.0.min(sec.1) * 0.45, 2);
    let m = sweep(&p, &section, Mat::Chrome, &SweepOpts { closed, twist_up: Some(nn), ..Default::default() });
    orient_outward(m)
}

/// A side scoop: a rounded wedge standing on the skin, deepest (h) at its
/// open front z0, tapering to 5 mm at z1.
#[allow(clippy::too_many_arguments)]
pub fn surface_bump(tube: &Tube, z0: f64, z1: f64, y0: f64, y1: f64, h: f64, side: f64, mat: Mat) -> Mesh {
    let ym = (y0 + y1) / 2.0;
    let nring = 22;
    let mut rings: Vec<DVec3> = Vec::new();
    let mut nsec = 0;
    for &t in &linspace(0.0, 1.0, nring) {
        let z = z0 + (z1 - z0) * t;
        let th = 0.010 + (h - 0.010) * (1.0 - t).powf(0.9);
        let hh = (y1 - y0) * (1.0 - 0.35 * t);
        let (p_, n_) = surface_point(tube, z, ym, side, (1.5, 5.5));
        let n_ = n_ / n_.length();
        let mut up = DVec3::Y;
        up -= n_ * up.dot(n_);
        up /= up.length();
        // superellipse section, sampled evenly by arc length
        // (e = 0.5: numpy evaluates `x ** 0.5` as sqrt)
        let mut raw: Vec<DVec2> = linspace_open(2.0 * PI, 400)
            .into_iter()
            .map(|a| {
                let (ca, sa) = (a.cos(), a.sin());
                DVec2::new(th / 2.0 * sign(ca) * ca.abs().sqrt(), hh / 2.0 * sign(sa) * sa.abs().sqrt())
            })
            .collect();
        raw.push(raw[0]);
        let mut sec = resample_polyline2(&raw, 16);
        sec.pop();
        nsec = sec.len();
        for s in &sec {
            rings.push(p_ + n_ * (s.x + th / 2.0 - 0.0012) + up * s.y);
        }
    }
    orient_outward(loft(&Grid2::new(nring, nsec, rings), mat, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::reset_islands;
    use crate::tube::{Handle, Key};

    /// [nv, nq, nt, sum v, sum v^2, order-weighted sum, quad index sum,
    /// tri index sum, material sum, island sum, uv sum]; the Python side is
    /// `sig()` in the generator script quoted in each test module comment.
    fn sig(m: &Mesh) -> [f64; 11] {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;
        for (i, p) in m.v.iter().enumerate() {
            s1 += p.x + p.y + p.z;
            s2 += p.x * p.x + p.y * p.y + p.z * p.z;
            s3 += ((i % 7) + 1) as f64 * (p.x + 2.0 * p.y + 3.0 * p.z);
        }
        let fq: f64 = m.q.iter().map(|f| f.iter().enumerate().map(|(k, &i)| ((k + 1) as u64 * i as u64) as f64).sum::<f64>()).sum();
        let ft: f64 = m.t.iter().map(|f| f.iter().enumerate().map(|(k, &i)| ((k + 1) as u64 * i as u64) as f64).sum::<f64>()).sum();
        let ms: f64 = m.qm.iter().chain(m.tm.iter()).map(|&x| x as u8 as f64).sum();
        let isl: f64 = m.qi.iter().chain(m.ti.iter()).map(|&x| x as f64).sum();
        let uv: f64 = m.quv.iter().flat_map(|u| u.iter()).chain(m.tuv.iter().flat_map(|u| u.iter())).map(|p| p.x + p.y).sum();
        [m.v.len() as f64, m.q.len() as f64, m.t.len() as f64, s1, s2, s3, fq, ft, ms, isl, uv]
    }

    fn check(name: &str, m: &Mesh, want: [f64; 11]) {
        let got = sig(m);
        for k in 0..11 {
            let tol = 1e-9 * want[k].abs().max(1.0);
            assert!((got[k] - want[k]).abs() <= tol, "{name}[{k}]: rust {} python {}\n rust {got:?}", got[k], want[k]);
        }
    }

    fn check2(name: &str, p: &[DVec2], want: [f64; 4]) {
        let n = p.len() as f64;
        let s1: f64 = p.iter().map(|q| q.x + q.y).sum();
        let s2: f64 = p.iter().map(|q| q.x * q.x + q.y * q.y).sum();
        let s3: f64 = p.iter().enumerate().map(|(i, q)| ((i % 7) + 1) as f64 * (q.x + 2.0 * q.y)).sum();
        let got = [n, s1, s2, s3];
        for k in 0..4 {
            let tol = 1e-9 * want[k].abs().max(1.0);
            assert!((got[k] - want[k]).abs() <= tol, "{name}[{k}]: rust {} python {}\n rust {got:?}", got[k], want[k]);
        }
    }

    fn check3(name: &str, p: &[DVec3], want: [f64; 4]) {
        let n = p.len() as f64;
        let s1: f64 = p.iter().map(|q| q.x + q.y + q.z).sum();
        let s2: f64 = p.iter().map(|q| q.length_squared()).sum();
        let s3: f64 = p.iter().enumerate().map(|(i, q)| ((i % 7) + 1) as f64 * (q.x + 2.0 * q.y + 3.0 * q.z)).sum();
        let got = [n, s1, s2, s3];
        for k in 0..4 {
            let tol = 1e-9 * want[k].abs().max(1.0);
            assert!((got[k] - want[k]).abs() <= tol, "{name}[{k}]: rust {} python {}\n rust {got:?}", got[k], want[k]);
        }
    }

    /// The test body: 5 key points, x and y tabled in z.
    fn test_tube() -> Tube {
        let keys = vec![
            Key::tables(&[(-2.0, 0.0), (2.0, 0.0)], &[(-2.0, 0.25), (0.0, 0.2), (2.0, 0.3)]),
            Key::tables(&[(-2.0, 0.7), (0.0, 0.8), (2.0, 0.75)], &[(-2.0, 0.3), (2.0, 0.25)]),
            Key::tables(&[(-2.0, 0.8), (0.0, 0.9), (2.0, 0.85)], &[(-2.0, 0.6), (0.0, 0.65), (2.0, 0.55)]),
            Key::tables(&[(-2.0, 0.6), (0.0, 0.7), (2.0, 0.5)], &[(-2.0, 1.1), (0.0, 1.25), (2.0, 0.9)]),
            Key::consts(0.0, 1.3),
        ];
        let h = vec![(Handle::C(1.0 / 3.0), Handle::C(1.0 / 3.0)); 5];
        Tube::new(keys, h, vec![])
    }

    fn fz_front(x: f64, y: f64) -> f64 {
        2.1 - 0.3 * x * x - 0.05 * y
    }

    fn outline() -> Vec<DVec2> {
        vec![
            DVec2::new(0.0, 2.0),
            DVec2::new(0.3, 1.98),
            DVec2::new(0.6, 1.9),
            DVec2::new(0.8, 1.7),
            DVec2::new(0.85, 1.4),
            DVec2::new(0.86, 1.0),
            DVec2::new(0.86, 0.6),
        ]
    }

    // Expected numbers below were produced by the Python generator with
    // Blender's python3.11 (numpy), run from engine/tools/vehicle-gen:
    // the same calls through vehgen.parts, with vehgen.mesh._ISLAND[0]
    // reset to 100000 before each mesh, and sig()/sig2()/sig3() computed
    // exactly as the Rust helpers above (vertex/quad/tri counts, sums,
    // order-weighted sums, face index sums, material, island and UV sums).
    fn pipe_pts() -> Vec<DVec3> {
        vec![DVec3::new(0.0, 0.2, -1.0), DVec3::new(0.2, 0.25, -1.5), DVec3::new(0.3, 0.22, -2.0), DVec3::new(0.35, 0.3, -2.3)]
    }
    #[test]
    fn py_tyre() {
        reset_islands();
        check(
            "tyre",
            &tyre(0.66, 0.235, 0.381, 0.20, 96, 3),
            [3936.0, 3936.0, 0.0, -10.32000000000005, 367.63247442837076, -35.96787054132176, 77440800.0, 0.0, 11808.0, -3936.0, 0.0],
        );
    }
    #[test]
    fn py_tyre_g2() {
        reset_islands();
        check(
            "tyre_g2",
            &tyre(0.60, 0.20, 0.36, 0.18, 64, 2),
            [2368.0, 2368.0, 0.0, -5.760000000000019, 179.04012165841263, -19.190449676472724, 28025280.0, 0.0, 7104.0, -2368.0, 0.0],
        );
    }
    #[test]
    fn py_rim_magnum() {
        reset_islands();
        check(
            "rim_magnum",
            &rim_magnum(0.381, 0.20, 72, 5),
            [1656.0, 1536.0, 156.0, 72.00742942618179, 58.54060106241174, 282.8124295775076, 11821944.0, 1457909.0, 7512.0, -1692.0, 0.0],
        );
    }
    #[test]
    fn py_rim_alloy() {
        reset_islands();
        check(
            "rim_alloy",
            &rim_alloy(0.43, 0.22, 72, 5),
            [
                2454.0,
                2196.0,
                448.0,
                147.0851863514619,
                91.59208075685652,
                581.8528589728161,
                25360590.0,
                4912338.0,
                8476.0,
                84002816.0,
                368.97663766567075,
            ],
        );
    }
    #[test]
    fn py_rim_steel() {
        reset_islands();
        check(
            "rim_steel",
            &rim_steel(0.40, 0.18, 64),
            [1106.0, 1064.0, 80.0, 28.221999999999994, 41.302092, 119.85114154096682, 5666700.0, 513300.0, 9336.0, -1144.0, 0.0],
        );
    }
    #[test]
    fn py_brake_disc() {
        reset_islands();
        check(
            "brake_disc",
            &brake_disc(0.28, 48),
            [538.0, 536.0, 0.0, -53.2036374869509, 7.435400899322474, -449.41088630828506, 1439638.0, 0.0, 3312.0, -536.0, 0.0],
        );
    }
    #[test]
    fn py_bucket_hi() {
        reset_islands();
        check(
            "bucket_hi",
            &bucket_seat(0.38, 0.2, 0.25, Mat::InteriorFabric, true),
            [998.0, 990.0, 0.0, 984.14, 692.3866647769136, 6284.180511151444, 4942688.0, 0.0, 6942.0, -990.0, 0.0],
        );
    }
    #[test]
    fn py_bucket_lo() {
        reset_islands();
        check(
            "bucket_lo",
            &bucket_seat(-0.38, 0.1, 0.22, Mat::Paint, false),
            [780.0, 774.0, 0.0, 40.33000000000001, 396.0505060508717, 1350.718504677533, 3024130.0, 0.0, 54.0, -774.0, 0.0],
        );
    }
    #[test]
    fn py_steering() {
        reset_islands();
        check(
            "steering",
            &steering_wheel(DVec3::new(0.38, 0.8, 0.6), 0.19, 22.0, Mat::InteriorPlastic, 0.30),
            [
                956.0,
                884.0,
                124.0,
                1705.4386305303483,
                1129.0190883503853,
                14493.244135941863,
                3947496.0,
                640296.0,
                7308.0,
                7199208.0,
                27.03315901997035,
            ],
        );
    }
    #[test]
    fn py_well_r() {
        reset_islands();
        check(
            "well_r",
            &wheel_well(DVec3::new(0.75, 0.33, 1.3), 0.37, 0.55, 0.85, -1.0, 100.0, 100.0),
            [123.0, 80.0, 0.0, 233.90756579712252, 251.81599567610084, 2432.8369398115533, 52682.0, 0.0, 160.0, -80.0, 0.0],
        );
    }
    #[test]
    fn py_well_l() {
        reset_islands();
        check(
            "well_l",
            &wheel_well(DVec3::new(0.75, 0.33, -1.3), 0.37, 0.55, 0.85, 1.0, 80.0, 120.0),
            [123.0, 80.0, 0.0, 65.51138786789585, 509.604301433636, -826.1581861344749, 41350.0, 0.0, 160.0, -80.0, 0.0],
        );
    }
    #[test]
    fn py_pipe() {
        reset_islands();
        check(
            "pipe",
            &tube_pipe(&pipe_pts(), 0.025, Mat::MetalBare, 6, 14),
            [
                266.0,
                252.0,
                24.0,
                -333.1144444444444,
                862.7980251371742,
                -4733.843333333334,
                333900.0,
                19080.0,
                2484.0,
                25200228.0,
                850.2692380024548,
            ],
        );
    }
    #[test]
    fn py_pipe2() {
        reset_islands();
        check(
            "pipe2",
            &tube_pipe(&pipe_pts(), 0.02, Mat::Chrome, 4, 10),
            [130.0, 120.0, 16.0, -162.7, 421.78462792968753, -2289.1959420329567, 77400.0, 6192.0, 136.0, 12000104.0, 397.0354984865622],
        );
    }
    #[test]
    fn py_wiper() {
        reset_islands();
        check(
            "wiper",
            &wiper(&[DVec3::new(0.1, 1.0, 0.9), DVec3::new(0.3, 1.05, 0.8), DVec3::new(0.5, 1.1, 0.7)], Mat::TrimBlack),
            [
                48.0,
                32.0,
                24.0,
                103.10400000000001,
                89.44017600000001,
                906.2577548642716,
                7520.0,
                3384.0,
                140.0,
                3200024.0,
                31.327071459644277,
            ],
        );
    }
    #[test]
    fn py_mirror_r() {
        reset_islands();
        check(
            "mirror_r",
            &sport_mirror(DVec3::new(0.8, 0.95, 0.6), -1.0, Mat::Paint),
            [
                954.0,
                936.0,
                20.0,
                2236.2999999999997,
                1843.456570699194,
                17263.988565078875,
                4507178.0,
                28260.0,
                140.0,
                2399092.0,
                5.997608902282728,
            ],
        );
    }
    #[test]
    fn py_mirror_l() {
        reset_islands();
        check(
            "mirror_l",
            &sport_mirror(DVec3::new(0.8, 0.95, 0.6), 1.0, Mat::Chrome),
            [
                954.0,
                936.0,
                20.0,
                2347.4839999999995,
                2021.350970699194,
                17708.550913913903,
                4507178.0,
                28260.0,
                956.0,
                2399092.0,
                5.9976089022827335,
            ],
        );
    }
    #[test]
    fn py_handle() {
        reset_islands();
        check(
            "handle",
            &door_handle(&test_tube(), 0.3, 0.6, 1.0, 0.13),
            [316.0, 312.0, 0.0, 574.9810153242865, 410.2153174548672, 3807.615921275169, 494964.0, 0.0, 408.0, -312.0, 0.0],
        );
    }
    #[test]
    fn py_handle_r() {
        reset_islands();
        check(
            "handle_r",
            &door_handle(&test_tube(), -0.3, 0.55, -1.0, 0.10),
            [316.0, 312.0, 0.0, -211.08483497953347, 391.10699758924443, -903.6643148962896, 494964.0, 0.0, 408.0, -312.0, 0.0],
        );
    }
    #[test]
    fn py_molding() {
        reset_islands();
        check(
            "molding",
            &molding(&test_tube(), &[DVec2::new(-1.0, 2.2), DVec2::new(0.0, 2.4), DVec2::new(1.0, 2.3)], 0.002, (0.010, 0.005), false, 4),
            [
                108.0,
                96.0,
                20.0,
                185.42388105676594,
                204.4421996093409,
                1101.0465061382959,
                51360.0,
                6420.0,
                116.0,
                9600076.0,
                393.2092776492537,
            ],
        );
    }
    #[test]
    fn py_molding_c() {
        reset_islands();
        check(
            "molding_c",
            &molding(
                &test_tube(),
                &[DVec2::new(-1.0, 2.2), DVec2::new(0.0, 2.4), DVec2::new(1.0, 2.3), DVec2::new(0.0, 2.0)],
                0.003,
                (0.012, 0.004),
                true,
                3,
            ),
            [
                144.0,
                144.0,
                0.0,
                238.87620896679732,
                263.6942809964214,
                1414.095911688313,
                102960.0,
                0.0,
                144.0,
                14400144.0,
                1175.5517729078947,
            ],
        );
    }
    #[test]
    fn py_bump() {
        reset_islands();
        check(
            "bump",
            &surface_bump(&test_tube(), 0.8, 0.2, 0.4, 0.6, 0.05, 1.0, Mat::Paint),
            [
                352.0,
                336.0,
                28.0,
                675.1545924014376,
                485.43772163364565,
                4791.886793838363,
                589680.0,
                29511.0,
                0.0,
                33600308.0,
                640.4445871655818,
            ],
        );
    }
    #[test]
    fn py_bump_r() {
        reset_islands();
        check(
            "bump_r",
            &surface_bump(&test_tube(), -0.5, 0.1, 0.45, 0.62, 0.04, -1.0, Mat::Chrome),
            [
                352.0,
                336.0,
                28.0,
                -204.3724351041113,
                422.13173248245687,
                -622.4992529880647,
                589680.0,
                29503.0,
                364.0,
                33600308.0,
                603.935221225805,
            ],
        );
    }
    #[test]
    fn py_wheel_magnum() {
        reset_islands();
        let spec = WheelSpec {
            track: 1.5,
            wheel_y: 0.33,
            tyre_d: 0.66,
            tyre_w: 0.235,
            rim_d: 0.381,
            rim_w: 0.20,
            rim_style: RimStyle::Magnum,
            disc_d: 0.30,
        };
        let (ri, ty, di, c) = wheel(&spec, 1.0, 1.3);
        check(
            "rim",
            &ri,
            [1656.0, 1536.0, 156.0, 4013.287429426182, 4077.0301452016847, 35424.39242957751, 11821944.0, 1457909.0, 7512.0, -1692.0, 0.0],
        );
        check("tyre", &ty, [3936.0, 3936.0, 0.0, 9357.36, 9646.622874428373, 83538.12212945867, 77440800.0, 0.0, 11808.0, -3936.0, 0.0]);
        check(
            "disc",
            &di,
            [538.0, 536.0, 0.0, 1224.2584271327992, 1179.5484714446002, 10929.119929753942, 1439638.0, 0.0, 3312.0, -536.0, 0.0],
        );
        let want = [0.75, 0.33, 1.3];
        assert!((c - DVec3::from_array(want)).length() < 1e-12);
    }
    #[test]
    fn py_wheel_alloy() {
        reset_islands();
        let spec = WheelSpec {
            track: 1.5,
            wheel_y: 0.33,
            tyre_d: 0.66,
            tyre_w: 0.235,
            rim_d: 0.381,
            rim_w: 0.20,
            rim_style: RimStyle::Alloy,
            disc_d: 0.30,
        };
        let (ri, ty, di, c) = wheel(&spec, -1.0, 1.3);
        check(
            "rim",
            &ri,
            [
                2454.0,
                2196.0,
                448.0,
                2028.3348136485383,
                6064.25615655702,
                36844.5350316804,
                25335910.0,
                4909918.0,
                8476.0,
                84002816.0,
                330.0045336543354,
            ],
        );
        check("tyre", &ty, [3936.0, 3936.0, 0.0, 3474.0, 9646.622874428373, 60011.26815880327, 77440800.0, 0.0, 11808.0, -3936.0, 0.0]);
        check(
            "disc",
            &di,
            [538.0, 536.0, 0.0, 453.3864271327991, 1179.5484714446002, 7849.741649264583, 1432282.0, 0.0, 3312.0, -536.0, 0.0],
        );
        let want = [-0.75, 0.33, 1.3];
        assert!((c - DVec3::from_array(want)).length() < 1e-12);
    }
    #[test]
    fn py_wheel_steel() {
        reset_islands();
        let spec = WheelSpec {
            track: 1.5,
            wheel_y: 0.33,
            tyre_d: 0.66,
            tyre_w: 0.235,
            rim_d: 0.381,
            rim_w: 0.20,
            rim_style: RimStyle::Steel,
            disc_d: 0.30,
        };
        let (ri, ty, di, c) = wheel(&spec, -1.0, 1.3);
        check("rim", &ri, [1106.0, 1064.0, 80.0, 940.398, 2700.933492, 16728.82120462065, 5660220.0, 511660.0, 9336.0, -1144.0, 0.0]);
        check("tyre", &ty, [3936.0, 3936.0, 0.0, 3474.0, 9646.622874428373, 60011.26815880327, 77440800.0, 0.0, 11808.0, -3936.0, 0.0]);
        check(
            "disc",
            &di,
            [538.0, 536.0, 0.0, 453.3864271327991, 1179.5484714446002, 7849.741649264583, 1432282.0, 0.0, 3312.0, -536.0, 0.0],
        );
        let want = [-0.75, 0.33, 1.3];
        assert!((c - DVec3::from_array(want)).length() < 1e-12);
    }
    #[test]
    fn py_rot_to() {
        // numpy row-major 3x3
        for (axis, want) in [
            (
                DVec3::new(0.3, 0.1, 1.0),
                [
                    0.28603877677367767,
                    -0.027397545898633385,
                    -0.9578262852211514,
                    0.09534625892455922,
                    0.9954441676503463,
                    0.0,
                    0.9534625892455922,
                    -0.09132515299544462,
                    0.28734788556634544,
                ],
            ),
            (DVec3::new(0.0, 2.0, 0.0), [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ] {
            let m = rot_to(axis);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((m.col(j)[i] - want[i * 3 + j]).abs() < 1e-12, "{i} {j}");
                }
            }
        }
    }
    #[test]
    fn py_graph_normal() {
        let fz = |x: f64, y: f64| 2.1 - 0.3 * x * x - 0.05 * y;
        assert!(
            (graph_normal(&fz, 0.4, 0.6, 1.0) - DVec3::from_array([0.23309761205488813, 0.0485620025117046, 0.9712400502277293])).length()
                < 1e-9
        );
        assert!(
            (graph_normal(&fz, 0.7, 0.2, -1.0) - DVec3::from_array([-0.3868216797960218, -0.04605019997603337, -0.9210039995146337]))
                .length()
                < 1e-9
        );
    }
    #[test]
    fn py_lamp1() {
        reset_islands();
        let (lens, hous) = round_lamp(DVec3::new(0.6, 0.7, 2.1), 0.09, 1.0, [1.0, 0.9, 0.8, 1.0], 0.07, None);
        check("lens", &lens, [194.0, 144.0, 96.0, 660.38, 1024.758367, 6410.088852686533, 152712.0, 58968.0, 1200.0, -240.0, 0.0]);
        check(
            "housing",
            &hous,
            [482.0, 440.0, 80.0, 1628.176, 2494.5967280000004, 15849.351404907451, 1044680.0, 134640.0, 1920.0, -520.0, 0.0],
        );
        let col: f64 = lens.col.as_ref().unwrap().iter().flatten().sum();
        assert!((col - 717.8000000000001).abs() < 1e-9);
        assert!(hous.col.is_none());
    }
    #[test]
    fn py_lamp2() {
        reset_islands();
        let (lens, hous) = round_lamp(DVec3::new(0.6, 0.7, -2.1), 0.06, -1.0, [1.0, 0.2, 0.1, 1.0], 0.01, Some(DVec3::new(0.3, 0.1, -1.0)));
        check(
            "lens",
            &lens,
            [194.0, 144.0, 96.0, -155.64622049176697, 1024.3785048521745, -3320.795698705493, 152712.0, 58968.0, 1200.0, -240.0, 0.0],
        );
        check(
            "housing",
            &hous,
            [482.0, 440.0, 80.0, -383.6869726609377, 2521.955023977345, -8246.046721907536, 1044680.0, 134640.0, 1920.0, -520.0, 0.0],
        );
        let col: f64 = lens.col.as_ref().unwrap().iter().flatten().sum();
        assert!((col - 446.20000000000005).abs() < 1e-9);
        assert!(hous.col.is_none());
    }
    #[test]
    fn py_outline_helpers() {
        let o = outline();
        check2("offset", &offset_outline(&o, 0.05, 1.0), [7.0, 15.257663117027235, 22.08006481531632, 94.34792246786203]);
        check2("offset_in", &offset_outline(&o, 0.03, -1.0), [7.0, 14.605402129783657, 20.420575655565205, 91.15124651928276]);
        check2("chaikin3", &chaikin(&o, 3), [56.0, 118.80000000000001, 166.9390787109375, 805.97484375]);
        check2("chaikin1", &chaikin(&o, 1), [14.0, 29.7, 41.809425000000005, 195.07499999999996]);
        check2("densify", &densify(&o, 0.1), [41.0, 81.445, 107.67625194444443, 548.5374019607843]);
        let outs = vec![
            o.iter().map(|p| *p * DVec2::new(0.95, 1.0)).collect::<Vec<_>>(),
            o.clone(),
            o.iter().map(|p| *p * DVec2::new(1.02, 1.01) + DVec2::new(0.0, 0.02)).collect::<Vec<_>>(),
        ];
        check2("fit", &fit_path(&outs, 0.04, 1.0, 3, None), [56.0, 123.48412615748498, 179.4500505221364, 833.4522414949093]);
        check2("fit_step", &fit_path(&outs, 0.04, 1.0, 2, Some(0.05)), [44.0, 98.28042751493012, 139.0860870321073, 645.1113131393108]);
    }
    #[test]
    fn py_tube_helpers() {
        let t = test_tube();
        let (p, n) = surface_point(&t, 0.5, 0.5, 1.0, (1.5, 5.5));
        assert!(
            (p - DVec3::from_array([0.9045999198593371, 0.5000000000000001, 0.5])).length() < 1e-9
                && (n - DVec3::from_array([0.9991665930309699, -0.040818125518992186, 0.0])).length() < 1e-9,
            "{p} {n}"
        );
        let (p, n) = surface_point(&t, 0.5, 0.5, -1.0, (1.5, 5.5));
        assert!(
            (p - DVec3::from_array([-0.9045999198593371, 0.5000000000000001, 0.5])).length() < 1e-9
                && (n - DVec3::from_array([-0.9991665930309699, -0.040818125518992186, 0.0])).length() < 1e-9,
            "{p} {n}"
        );
        let (p, n) = path_on_surface(&t, &linspace(-1.0, 1.0, 7), &[0.45], -1.0, 0.01, (1.5, 5.5));
        check3("path_p", &p, [7.0, -3.1776788861838936, 10.22299280882518, 27.741712393536528]);
        check3("path_n", &n, [7.0, -7.904833501208278, 6.999999999999999, -35.08055035348312]);
        let (p, n) = path_on_surface(&t, &linspace(-1.0, 1.0, 5), &linspace(0.4, 0.7, 5), 1.0, 0.0, (1.2, 5.0));
        check3("path2_p", &p, [5.0, 7.185032540653273, 8.004221855599198, 46.3469623609864]);
        check3("path2_n", &n, [5.0, 4.840531654094135, 5.0, 16.73177652717508]);
        check2(
            "plan_front",
            &plan_outline(&t, &fz_front, 0.5, 1.0, 2.3, 0.40, 28, 10),
            [38.0, 93.22225395332995, 153.47754259888953, 637.9218397987382],
        );
        let fz_rear = |x: f64, y: f64| -fz_front(x, y);
        check2(
            "plan_rear",
            &plan_outline(&t, &fz_rear, 0.7, -1.0, -2.3, 0.3, 12, 5),
            [17.0, -23.57416717947178, 68.33911929858945, -202.96535701327764],
        );
    }
}
