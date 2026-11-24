//! The Mach 1's parts: the wheels, lamps, bumpers, trim, interior and underbody
//! meshes and the markers, built on the car body (split from `mach1.rs`).

use super::*;

/// numpy.arange(a, b, step) for floats.
fn arange(a: f64, b: f64, step: f64) -> Vec<f64> {
    let n = ((b - a) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| a + i as f64 * step).collect()
}

/// A path round the whole nose or tail from the half outline (x >= 0).
fn full(pxz: &[DVec2]) -> Vec<DVec2> {
    let mut r: Vec<DVec2> = pxz.iter().rev().map(|q| DVec2::new(-q.x, q.y)).collect();
    r.pop();
    r.extend_from_slice(pxz);
    r
}

impl Mach1 {
    pub(super) fn make_parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        let mut out = Parts::default();
        let mut extras = Parts::default();
        let mut mk: Vec<Marker> = Vec::new();
        self.parts_wheels(&mut out, &mut mk);
        self.parts_headlamps(cb, &mut out);
        self.parts_front(cb, &mut out, &mut mk);
        self.parts_rear(cb, &mut out, &mut mk);
        self.parts_body_extras(cb, &mut out, &mut extras, &mut mk);
        self.parts_bonnet_wing(cb, &mut extras);
        self.parts_interior(cb, &mut out, &mut mk);
        self.parts_underbody(cb, &mut out);
        self.parts_markers(cb, &mut mk);
        (out, extras, mk)
    }

    fn parts_wheels(&self, out: &mut Parts, mk: &mut Vec<Marker>) {
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // wheels
        for (tag, w) in [("F", &self.run.front), ("R", &self.run.rear)] {
            let (spec, z) = (Self::wheel_spec(w), w.z);
            for (s, sg) in sides {
                let (rim, ty, di, c) = p::wheel(&spec, sg, z);
                out.insert(&format!("rim_{tag}{s}"), rim);
                out.insert(&format!("tyre_{tag}{s}"), ty);
                out.insert(&format!("brake_disc_{tag}{s}"), di);
                marker(format!("wheel_center_{tag}{s}"), c, DVec3::new(sg, 0.0, 0.0));
            }
        }
    }

    fn parts_headlamps(&self, cb: &CarBody, out: &mut Parts) {
        let tube = cb.tube();
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        // headlamps: outer in the fender ends, inner in the grille
        let zg = front_z(0.60, 0.65) - 0.045;
        for (s, sg) in sides {
            let hl = HEADLIGHT;
            let zs = front_z(hl.0, hl.1);
            let g = p::graph_normal(&front_z, hl.0, hl.1, 1.0);
            let nrm = DVec3::new(g.x * sg, g.y, g.z);
            // the lamp sits at the bottom of a scooped bucket under the fender brow
            const DEEP: f64 = 0.020;
            let cen = DVec3::new(sg * hl.0, hl.1, zs) - nrm * DEEP;
            let (mut l1, h1) = p::round_lamp(cen, hl.2 - 0.020, 1.0, CLEAR, 0.07, Some(nrm));
            let rt = p::rot_to(nrm);
            let rings: Vec<Vec<DVec3>> = [(0.0, hl.2 - 0.003, 0.003), (0.5, hl.2 - 0.006, 0.007), (1.0, hl.2 - 0.010, 0.011)]
                .iter()
                .map(|&(_t, rr, dd)| {
                    (0..48)
                        .map(|k| {
                            let a = std::f64::consts::TAU * k as f64 / 48.0;
                            let off = rt * DVec3::new(0.0, rr * a.cos(), rr * a.sin());
                            let mut q = DVec3::new(sg * hl.0, hl.1, 0.0) + off;
                            q.z = front_z(q.x.abs(), q.y);
                            q - nrm * dd
                        })
                        .collect()
                })
                .collect();
            let gr = Grid2::from_fn(rings.len(), 48, |i, j| rings[i][j]);
            let mut bucket = grid(&gr, None, Mat::Paint, false, true, -1);
            let cn = bucket.vertex_normals().iter().copied().sum::<DVec3>() / bucket.v.len() as f64;
            if cn.dot(nrm) < 0.0 {
                bucket.flip();
            }
            // the tunnel's inner wall, from its floor forward to behind the
            // grille surround
            let ys0 = linspace(0.497, 0.752, 60);
            let zb0: Vec<f64> = ys0.iter().map(|&y| front_z(0.665, y)).collect();
            let zfw = zg - 0.008;
            let ok: Vec<usize> = (0..60).filter(|&k| zfw - zb0[k] > 0.025).collect();
            let ys = linspace(ys0[ok[0]], ys0[*ok.last().unwrap()], 11);
            let tt = linspace(0.0, 1.0, 4);
            let pw = Grid2::from_fn(ys.len(), tt.len(), |i, j| {
                let zb = front_z(0.665, ys[i]);
                DVec3::new(sg * 0.654, ys[i], zb + (zfw - zb) * tt[j])
            });
            let mut wall = grid(&pw, None, Mat::Paint, false, false, -1);
            if sg < 0.0 {
                wall.flip();
            }
            let hi = HEADLIGHT_INNER;
            let (l2, h2) = p::round_lamp(DVec3::new(sg * hi.0, hi.1, zg + 0.020), hi.2, 1.0, CLEAR, 0.014, None);
            // side markers: amber at the front of the fender, red at the tail
            let (fm, fn_) = p::surface_point(tube, 1.97, 0.560, sg, (1.5, 5.5));
            let mut mk_ = rbox(DVec3::ZERO, DVec3::new(0.004, 0.016, 0.034), 0.003, 1, Mat::LampLens);
            mk_.transform(Some(p::rot_to(fn_)), fm + fn_ * 0.005);
            mk_.tint(AMBER);
            l1.add(&l2).add(&mk_);
            // (the bucket and the tunnel wall are sheets: they ride the lamp
            // housing, not the closed fender solid)
            let mut h1 = h1;
            h1.add(&h2).add(&bucket).add(&wall);
            out.insert(&format!("headlight_{s}_lens"), l1);
            out.insert(&format!("headlight_{s}_housing"), h1);
        }
    }

    fn parts_front(&self, cb: &CarBody, out: &mut Parts, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let zg = front_z(0.60, 0.65) - 0.045;
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // grille: black egg-crate in a thin bright surround, recessed between
        // the fender ends, closed behind by a back plate
        let mut g = Mesh::new();
        let (gx, gy0, gy1) = (0.652, 0.495, 0.735);
        let gm = (gy0 + gy1) / 2.0;
        let outline = rrect(2.0 * gx, gy1 - gy0, 0.035, 5);
        let path: Vec<DVec3> = outline.iter().map(|q| DVec3::new(q.x, q.y + gm, zg + 0.004)).collect();
        g.add(&orient_outward(sweep(&path, &rrect(0.010, 0.022, 0.004, 2), Mat::Chrome, &SweepOpts { closed: true, up: DVec3::Z, ..Default::default() })));
        g.add(&box_(DVec3::new(0.0, gm, zg - 0.045), DVec3::new(gx - 0.004, (gy1 - gy0) / 2.0 - 0.004, 0.003), Mat::TrimBlack, None));
        for x in arange(-0.420, 0.4201, 0.035) {
            g.add(&box_(DVec3::new(x, gm, zg - 0.018), DVec3::new(0.0022, (gy1 - gy0) / 2.0 - 0.012, 0.022), Mat::TrimBlack, None));
        }
        for y in linspace(gy0 + 0.04, gy1 - 0.04, 6) {
            g.add(&box_(DVec3::new(0.0, y, zg - 0.018), DVec3::new(0.43, 0.0022, 0.022), Mat::TrimBlack, None));
        }
        // solid black end panels round the inner lamps
        for sg in [1.0, -1.0] {
            g.add(&box_(DVec3::new(sg * 0.545, gm, zg - 0.006), DVec3::new(0.100, (gy1 - gy0) / 2.0 - 0.010, 0.004), Mat::TrimBlack, None));
        }
        out.insert("grille", g);

        // front bumper: chrome blade wrapping round to the wheel openings,
        // painted valance and a black chin spoiler below it
        let yb = 0.452;
        let ols: Vec<Vec<DVec2>> = [0.40, 0.452, 0.50].iter().map(|&y| full(&p::plan_outline(tube, &front_z, y, 1.0, 2.3, 0.14, 28, 10))).collect();
        let pp = p::fit_path(&ols, 0.022, 1.0, 3, Some(0.018));
        let path: Vec<DVec3> = pp.iter().map(|q| DVec3::new(q.x, yb, q.y)).collect();
        let mut bf = orient_outward(sweep(&path, &rrect(0.032, 0.084, 0.016, 4), Mat::Chrome, &SweepOpts::default()));
        let ols = vec![full(&p::plan_outline(tube, &front_z, 0.40, 1.0, 2.3, 0.06, 28, 10))];
        let pv = p::fit_path(&ols, -0.022, 1.0, 3, Some(0.018));
        let pv = &pv[4..pv.len() - 4];
        let vpath: Vec<DVec3> = pv.iter().map(|q| DVec3::new(q.x, 0.0, q.y)).collect();
        let vsec = [DVec2::new(0.004, 0.392), DVec2::new(-0.026, 0.392), DVec2::new(-0.046, 0.310), DVec2::new(-0.058, 0.282), DVec2::new(-0.040, 0.280), DVec2::new(-0.014, 0.310)];
        bf.add(&orient_outward(sweep(&vpath, &p::densify(&vsec, 0.02), Mat::Paint, &SweepOpts::default())));
        let csec = [DVec2::new(0.070, 0.272), DVec2::new(-0.050, 0.272), DVec2::new(-0.050, 0.260), DVec2::new(0.070, 0.262)];
        let k = vpath.len() / 8;
        let cpath: Vec<DVec3> = vpath[k..vpath.len() - k].iter().map(|q| DVec3::new(q.x * 0.97, q.y, q.z)).collect();
        bf.add(&orient_outward(sweep(&cpath, &csec, Mat::TrimBlack, &SweepOpts::default())));
        out.insert("bumper_front", bf);
        // parking lamps: wide amber lenses in the valance just under the blade
        for (s, sg) in sides {
            let kx = (0..vpath.len()).min_by(|&a, &b| (vpath[a].x - sg * 0.582).abs().total_cmp(&(vpath[b].x - sg * 0.582).abs())).unwrap();
            let zv = vpath[kx].z;
            let mut pl = rbox(DVec3::new(sg * 0.582, 0.366, zv + 0.017), DVec3::new(0.072, 0.022, 0.004), 0.007, 1, Mat::LampLens);
            pl.tint(AMBER);
            out.get_mut(&format!("headlight_{s}_lens")).unwrap().add(&pl);
            out.get_mut(&format!("headlight_{s}_housing")).unwrap().add(&rbox(DVec3::new(sg * 0.582, 0.366, zv + 0.009), DVec3::new(0.078, 0.027, 0.0035), 0.008, 1, Mat::TrimBlack));
        }
        for (k_, x) in [-0.62f64, -0.30, 0.30, 0.62].iter().enumerate() {
            let zz = front_z(x.abs(), yb);
            marker(format!("bolt_bumper_front_{}", k_ + 1), DVec3::new(*x, yb, zz - 0.004), DVec3::X);
        }
    }

    fn parts_rear(&self, cb: &CarBody, out: &mut Parts, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // rear bumper
        let yr = 0.600;
        let ols: Vec<Vec<DVec2>> = [0.56, 0.60, 0.64].iter().map(|&y| full(&p::plan_outline(tube, &rear_z_body, y, -1.0, -2.4, 0.15, 28, 10))).collect();
        let mut pr = p::fit_path(&ols, 0.023, -1.0, 3, Some(0.018));
        pr.reverse();
        let path: Vec<DVec3> = pr.iter().map(|q| DVec3::new(q.x, yr, q.y)).collect();
        out.insert("bumper_rear", orient_outward(sweep(&path, &rrect(0.040, 0.088, 0.018, 4), Mat::Chrome, &SweepOpts::default())));
        for (k_, x) in [-0.62f64, -0.30, 0.30, 0.62].iter().enumerate() {
            let zz = rear_z_body(x.abs(), yr);
            marker(format!("bolt_bumper_rear_{}", k_ + 1), DVec3::new(*x, yr, zz + 0.004), DVec3::X);
        }

        // taillamps: three lenses each side in a bright bezel, a reflector box behind
        let tl = TAILLIGHT;
        for (s, sg) in sides {
            let xm = (tl.0 + tl.1) / 2.0;
            let ym = (tl.2 + tl.3) / 2.0;
            let zt = rear_z(xm, ym);
            let w = tl.1 - tl.0;
            let hgt = tl.3 - tl.2;
            let mut lens = Mesh::new();
            let mut hous = Mesh::new();
            let gapx = 0.012;
            let lw = (w - 0.020 - 2.0 * gapx) / 3.0;
            hous.add(&box_(DVec3::new(sg * xm, ym, zt + 0.046), DVec3::new(w / 2.0 - 0.005, hgt / 2.0 - 0.005, 0.006), Mat::TrimBlack, None));
            for k in 0..3 {
                let x = tl.0 + 0.010 + lw / 2.0 + k as f64 * (lw + gapx);
                let zk = rear_z(x, ym);
                let lh = hgt - 0.016;
                lens.add(&rbox(DVec3::new(sg * x, ym, zk + 0.016), DVec3::new(lw / 2.0 - 0.007, lh / 2.0 - 0.008, 0.006), (lw / 2.0 - 0.008).min(0.022), 3, Mat::LampLens));
                let rr = rrect(lw - 0.002, lh + 0.002, (lw / 2.0 - 0.002).min(0.024), 4);
                let path: Vec<DVec3> = rr.iter().map(|q| DVec3::new(sg * x + q.x, ym + q.y, zk + 0.008)).collect();
                hous.add(&orient_outward(sweep(&path, &rrect(0.003, 0.010, 0.0013, 1), Mat::Chrome, &SweepOpts { closed: true, up: DVec3::Z, ..Default::default() })));
                hous.add(&box_(DVec3::new(sg * x, ym, zk + 0.032), DVec3::new(lw / 2.0 - 0.006, lh / 2.0 - 0.006, 0.003), Mat::LampEmissive, None));
            }
            lens.tint(RED);
            let (rm, rn) = p::surface_point(tube, -2.12, 0.605, sg, (1.5, 5.5));
            let mut mr = rbox(DVec3::ZERO, DVec3::new(0.004, 0.016, 0.040), 0.003, 1, Mat::LampLens);
            mr.transform(Some(p::rot_to(rn)), rm + rn * 0.007);
            mr.tint(RED);
            lens.add(&mr);
            out.insert(&format!("taillight_{s}_lens"), lens);
            out.insert(&format!("taillight_{s}_housing"), hous);
        }
    }

    fn parts_body_extras(&self, cb: &CarBody, out: &mut Parts, extras: &mut Parts, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let vm = tube.vm;
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // mirrors, handles, wipers
        for (s, sg) in sides {
            let v = if sg > 0.0 { 6.35 } else { vm - 6.35 };
            let base = tube.eval(0.50, v);
            out.insert(&format!("mirror_{s}"), p::sport_mirror(base, sg, Mat::Paint));
            marker(format!("mount_mirror_{s}"), base, DVec3::new(sg, 0.0, 0.0));
            out.insert(&format!("handle_F{s}"), p::door_handle(tube, -0.53, 0.826, sg, 0.13));
        }
        for (s, x0, x1) in [("L", 0.56, 0.10), ("R", 0.02, -0.44)] {
            let zs = 0.40;
            let pts: Vec<DVec3> = linspace(x0, x1, 8)
                .into_iter()
                .map(|x: f64| {
                    let mut v = tube.v_at_x(zs, x.abs(), 7.9, 9.0);
                    if x < 0.0 {
                        v = vm - v;
                    }
                    let (pp, nn) = tube.eval_n(zs, v);
                    pp + nn * 0.004
                })
                .collect();
            out.insert(&format!("wiper_{s}"), p::wiper(&pts, Mat::TrimBlack));
        }

        // body extras: quarter scoops, rocker mouldings, window reveals, fuel cap
        let mut bx = Mesh::new();
        for sg in [1.0, -1.0] {
            // (the right scoop is the left one mirrored: lofted on the right
            // its caps wind against its sides)
            let mut bump = p::surface_bump(tube, -0.735, -1.00, 0.775, 0.925, 0.030, 1.0, Mat::Paint);
            if sg < 0.0 {
                bump.mirror_x();
            }
            bx.add(&bump);
            // the scoop's dark mouth facing forward
            let (q, nq) = p::surface_point(tube, -0.742, 0.850, sg, (1.5, 5.5));
            let nq = nq / nq.length();
            let mut mouth = rbox(DVec3::ZERO, DVec3::new(0.011, 0.058, 0.0015), 0.0014, 1, Mat::TrimBlack);
            mouth.transform(None, q + nq * 0.016);
            bx.add(&mouth);
            let sv = linspace(0.52, -0.66, 30);
            let (pp, _) = p::path_on_surface(tube, &sv, &[0.285], sg, 0.004, (1.5, 5.5));
            bx.add(&orient_outward(sweep(&pp, &rrect(0.005, 0.016, 0.002, 1), Mat::Chrome, &SweepOpts::default())));
        }
        for side in [1.0, -1.0] {
            let f = |v: f64| if side > 0.0 { v } else { vm - v };
            let drip: Vec<DVec2> = [0.38, 0.066, -0.30, -0.62, -0.86].iter().map(|&z| DVec2::new(z, f(7.25))).collect();
            bx.add(&p::molding(tube, &drip, 0.003, (0.008, 0.005), false, 4));
        }
        let zv = |a: &[(f64, f64)]| a.iter().map(|&(z, v)| DVec2::new(z, v)).collect::<Vec<_>>();
        let ws = zv(&[(0.42, 8.0), (0.255, 8.0), (0.09, 8.0), (0.12, 9.0), (0.09, 10.0), (0.255, 10.0), (0.42, 10.0), (0.47, 9.0)]);
        bx.add(&p::molding(tube, &ws, 0.002, (0.010, 0.005), true, 6));
        let rw = zv(&[(-0.89, 8.0), (-1.33, 8.0), (-1.742, 8.0), (-1.742, 9.0), (-1.742, 10.0), (-1.33, 10.0), (-0.89, 10.0), (-0.90, 9.0)]);
        bx.add(&p::molding(tube, &rw, 0.002, (0.010, 0.005), true, 6));
        let zt = rear_z(0.0, 0.735);
        let mut cap = closed_lathe(&[(0.0, 0.012), (0.045, 0.010), (0.058, 0.0), (0.056, -0.004), (0.0, -0.004)], 40, Mat::Chrome);
        cap.transform(Some(rot_y(std::f64::consts::FRAC_PI_2)), DVec3::new(0.0, 0.735, zt - 0.004));
        bx.add(&cap);
        extras.extend("body_shell", &bx);

        // bright lip mouldings round the wheel openings (fender front, body rear)
        for (arch, owner) in [(ARCH_F, "fender_F"), (ARCH_R, "body_shell")] {
            let (zc, yc, r) = arch;
            let ra = r + 0.004;
            for (s, sg) in sides {
                let lo_y = if owner == "fender_F" { 0.30 } else { 0.26 };
                let a0 = ((lo_y - yc) / ra).clamp(-1.0, 1.0).asin();
                let angs = linspace(a0, std::f64::consts::PI - a0, 60);
                let zs_: Vec<f64> = angs.iter().map(|a| zc + ra * a.cos()).collect();
                let ys_: Vec<f64> = angs.iter().map(|a| yc + ra * a.sin()).collect();
                let (pp, nn) = p::path_on_surface(tube, &zs_, &ys_, sg, 0.004, (1.5, 5.5));
                let mold = orient_outward(sweep(&pp, &rrect(0.004, 0.012, 0.0018, 1), Mat::Chrome, &SweepOpts { twist_up: Some(nn), ..Default::default() }));
                let key = if owner == "fender_F" { format!("fender_F{s}") } else { owner.to_string() };
                extras.extend(&key, &mold);
            }
        }
    }

    fn parts_bonnet_wing(&self, cb: &CarBody, extras: &mut Parts) {
        let tube = cb.tube();
        let vm = tube.vm;
        // bonnet: scoop and pins
        let mut hx = Mesh::new();
        let nr = 12;
        let rings: Vec<Vec<DVec3>> = linspace(0.0, 1.0, nr)
            .into_iter()
            .map(|t: f64| {
                let z = 1.10 + 0.32 * t;
                let hgt = 0.010 + 0.042 * t.powf(1.2);
                let wdt = 0.15 + 0.025 * t;
                let yb_ = tube.eval(z, 9.0).y + 0.010;
                let mut ring: Vec<DVec3> = linspace(0.0, std::f64::consts::PI, 11).into_iter().map(|a: f64| DVec3::new(wdt * a.cos(), yb_ + hgt * a.sin().max(0.0).powf(0.6), z)).collect();
                ring.push(DVec3::new(-wdt * 0.98, yb_ - 0.006, z));
                ring.push(DVec3::new(wdt * 0.98, yb_ - 0.006, z));
                ring
            })
            .collect();
        let nrg = rings[0].len();
        let gr = Grid2::from_fn(rings.len(), nrg, |i, j| rings[i][nrg - 1 - j]);
        hx.add(&orient_outward(loft(&gr, Mat::TrimBlack, true)));
        hx.add(&box_(DVec3::new(0.0, tube.eval(1.42, 9.0).y + 0.034, 1.424), DVec3::new(0.15, 0.016, 0.002), Mat::InteriorPlastic, None));
        for sg in [1.0, -1.0] {
            let z = 2.08;
            let mut v = tube.v_at_x(z, 0.47, 6.0, 9.0);
            if sg < 0.0 {
                v = vm - v;
            }
            let pp = tube.eval(z, v);
            let mut pin = closed_lathe(&[(0.0, 0.016), (0.010, 0.016), (0.012, 0.010), (0.012, -0.004), (0.0, -0.004)], 16, Mat::Chrome);
            for q in &mut pin.v {
                *q = DVec3::new(q.y, q.x, q.z);
            }
            pin.flip();
            pin.transform(None, pp + DVec3::new(0.0, 0.001, 0.0));
            hx.add(&orient_outward(pin));
        }
        extras.extend("hood", &hx);

        // the rear wing: two black uprights on the lid, a cambered blade
        if self.p.wing {
            let mut wx = Mesh::new();
            let zw = -2.27;
            for sg in [1.0, -1.0] {
                let mut v = tube.v_at_x(zw, 0.72, 6.0, 9.0);
                if sg < 0.0 {
                    v = vm - v;
                }
                let q = tube.eval(zw, v);
                let mut upr = rbox(DVec3::ZERO, DVec3::new(0.010, 0.030, 0.045), 0.007, 2, Mat::TrimBlack);
                upr.transform(Some(rot_x(8.0f64.to_radians())), q + DVec3::new(0.0, 0.032, -0.005));
                wx.add(&upr);
            }
            let yb_ = tube.eval(zw, 9.0).y + 0.068;
            let blade: Vec<DVec3> = linspace(-0.80, 0.80, 33)
                .into_iter()
                .map(|x: f64| {
                    let droop = 0.030 * ((x.abs() - 0.62) / 0.18).max(0.0).powi(2);
                    DVec3::new(x, yb_ + 0.006 * (1.0 - (x / 0.8).powi(2)) - droop, zw - 0.005)
                })
                .collect();
            let sec = [DVec2::new(0.065, 0.004), DVec2::new(0.030, 0.012), DVec2::new(-0.040, 0.010), DVec2::new(-0.065, -0.004), DVec2::new(-0.040, -0.006), DVec2::new(0.030, -0.004)];
            wx.add(&orient_outward(sweep(&blade, &sec, Mat::TrimBlack, &SweepOpts::default())));
            extras.extend("trunk_lid", &wx);
        }
    }

    fn parts_interior(&self, cb: &CarBody, out: &mut Parts, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let vm = tube.vm;
        let solid = cb.extra.contains_key("solid");
        let thick = |m: Mesh, th: f64| if solid { crate::solid::slab(&m, th) } else { m };
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // interior
        let hf = 0.255;
        for (s, sg) in sides {
            // (solid build: the seats, the dash and the console are parts on
            // the structure: `mechanical`)
            if !solid {
                out.insert(&format!("seat_F{s}"), p::bucket_seat(sg * 0.37, -0.34, hf, Mat::InteriorFabric, true));
            }
            marker(format!("seat_H_point_F{s}"), DVec3::new(sg * 0.37, hf + 0.155, -0.47), DVec3::X);
        }
        if !solid {
            let mut rs = Mesh::new();
            rs.add(&rbox(DVec3::new(0.0, 0.40, -1.00), DVec3::new(0.50, 0.065, 0.20), 0.05, 3, Mat::InteriorFabric));
            let mut back = rbox(DVec3::ZERO, DVec3::new(0.50, 0.22, 0.05), 0.05, 3, Mat::InteriorFabric);
            back.transform(Some(rot_x((-38.0f64).to_radians())), DVec3::new(0.0, 0.67, -1.28));
            rs.add(&back);
            out.insert("seat_rear", rs);
            let mut dash = Mesh::new();
            dash.add(&rbox(DVec3::new(0.0, 0.735, 0.40), DVec3::new(0.68, 0.115, 0.105), 0.05, 3, Mat::InteriorPlastic));
            for sg in [1.0, -1.0] {
                dash.add(&rbox(DVec3::new(sg * 0.37, 0.86, 0.36), DVec3::new(0.20, 0.035, 0.07), 0.03, 2, Mat::InteriorPlastic));
            }
            for k in 0..2 {
                let mut g_ = closed_lathe(&[(0.0, 0.002), (0.040, 0.002), (0.040, -0.004), (0.0, -0.004)], 24, Mat::Chrome);
                g_.transform(Some(rot_y(-std::f64::consts::FRAC_PI_2)), DVec3::new(0.37 + (k as f64 - 0.5) * 0.10, 0.80, 0.302));
                dash.add(&g_);
            }
            out.insert("dash", dash);
            out.insert("console", rbox(DVec3::new(0.0, 0.405, -0.10), DVec3::new(0.09, 0.10, 0.40), 0.03, 2, Mat::InteriorPlastic));
        }
        out.insert("steering_wheel", p::steering_wheel(DVec3::new(0.37, 0.80, 0.11), 0.19, 24.0, Mat::InteriorPlastic, if solid { 0.19 } else { 0.14 }));
        for s in ["FL", "FR"] {
            let sk = cb.skins[&format!("door_{s}")].clone();
            let mut isb = vec![false; sk.v.len()];
            for lp in boundary_loops(&sk) {
                for i in lp {
                    isb[i as usize] = true;
                }
            }
            let trim = if solid { crate::solid::door_card(&sk, 0.085, 0.018, Mat::InteriorPlastic) } else { inner_panel(&sk, &isb, 0.085, -1_000_000, 2, Mat::InteriorPlastic) };
            out.insert(&format!("door_trim_{s}"), trim);
        }
        let zz = linspace(0.06, -0.84, 46);
        let vv = linspace(7.35, vm - 7.35, 60);
        let hp = Grid2::from_fn(zz.len(), vv.len(), |i, k| {
            let (pp, nn) = tube.eval_n(zz[i], vv[k]);
            pp - nn * 0.040
        });
        let mut hl = grid(&hp, None, Mat::InteriorFabric, false, false, -1);
        let my: f64 = hl.vertex_normals().iter().map(|n| n.y).sum::<f64>() / hl.v.len() as f64;
        if my > 0.0 {
            hl.flip();
        }
        out.insert("headliner", thick(hl, 0.008));
    }

    fn parts_underbody(&self, cb: &CarBody, out: &mut Parts) {
        let solid = cb.extra.contains_key("solid");
        let thick = |m: Mesh, th: f64| if solid { crate::solid::slab(&m, th) } else { m };
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        // engine bay and underbody
        if solid {
            // the body-in-white: every member a closed solid, and the
            // mechanical and interior parts on it
            let st = self.unibody();
            for (name, m) in st.parts() {
                out.insert(&name, m);
            }
            for part in self.mechanical(&st) {
                if !part.mesh.v.is_empty() {
                    out.insert(&part.name, part.mesh);
                }
            }
        } else {
            let mut eb = Mesh::new();
            eb.add(&rbox(DVec3::new(0.0, 0.43, 1.30), DVec3::new(0.21, 0.15, 0.30), 0.03, 2, Mat::MetalBare));
            for sg in [1.0f64, -1.0] {
                let rot = glam::DMat3::from_cols(DVec3::new((sg * 0.6).cos(), (sg * 0.6).sin(), 0.0), DVec3::new(-(sg * 0.6).sin(), (sg * 0.6).cos(), 0.0), DVec3::Z);
                let mut hd = rbox(DVec3::ZERO, DVec3::new(0.07, 0.06, 0.29), 0.02, 2, Mat::MetalBare);
                hd.transform(Some(rot), DVec3::new(sg * 0.22, 0.60, 1.30));
                eb.add(&hd);
                let mut vc = rbox(DVec3::ZERO, DVec3::new(0.05, 0.025, 0.27), 0.015, 2, Mat::Chrome);
                vc.transform(Some(rot), DVec3::new(sg * 0.26, 0.665, 1.30));
                eb.add(&vc);
            }
            eb.add(&rbox(DVec3::new(0.0, 0.61, 1.30), DVec3::new(0.14, 0.05, 0.24), 0.02, 2, Mat::MetalBare));
            let mut ac = closed_lathe(&[(0.0, 0.05), (0.18, 0.05), (0.19, 0.03), (0.19, -0.03), (0.18, -0.04), (0.0, -0.04)], 48, Mat::Chrome);
            for q in &mut ac.v {
                *q = DVec3::new(q.y, q.x, q.z);
            }
            ac.flip();
            ac.transform(None, DVec3::new(0.0, 0.715, 1.30));
            eb.add(&orient_outward(ac));
            eb.add(&rbox(DVec3::new(0.0, 0.25, 1.30), DVec3::new(0.17, 0.05, 0.27), 0.02, 2, Mat::MetalBare));
            let mut fan = closed_lathe(&[(0.0, 0.01), (0.17, 0.01), (0.17, -0.01), (0.0, -0.01)], 32, Mat::TrimBlack);
            fan.transform(Some(rot_y(-std::f64::consts::FRAC_PI_2)), DVec3::new(0.0, 0.48, 1.66));
            eb.add(&fan);
            out.insert("engine_block", eb);
            let mut rad = Mesh::new();
            rad.add(&box_(DVec3::new(0.0, 0.56, 2.06), DVec3::new(0.37, 0.17, 0.028), Mat::TrimBlack, None));
            rad.add(&box_(DVec3::new(0.0, 0.745, 2.06), DVec3::new(0.385, 0.016, 0.034), Mat::MetalBare, None));
            rad.add(&box_(DVec3::new(0.0, 0.375, 2.06), DVec3::new(0.385, 0.016, 0.034), Mat::MetalBare, None));
            out.insert("radiator", rad);
            out.insert("firewall", box_(DVec3::new(0.0, 0.57, 0.625), DVec3::new(0.60, 0.27, 0.003), Mat::TrimBlack, None));
            let mut ub = Mesh::new();
            ub.add(&box_(DVec3::new(0.0, 0.165, 1.56), DVec3::new(0.44, 0.003, 0.41), Mat::TrimBlack, None));
            ub.add(&box_(DVec3::new(0.0, 0.375, -2.06), DVec3::new(0.48, 0.004, 0.27), Mat::TrimBlack, None));
            ub.add(&rbox(DVec3::new(0.0, 0.43, -1.98), DVec3::new(0.42, 0.05, 0.20), 0.02, 2, Mat::MetalBare));
            let mut ax = closed_lathe(&[(0.0, 0.64), (0.045, 0.64), (0.045, -0.64), (0.0, -0.64)], 20, Mat::MetalBare);
            ax.transform(None, DVec3::new(0.0, WHEEL_Y, ZR_AX));
            ub.add(&ax);
            ub.add(&rbox(DVec3::new(0.0, WHEEL_Y, ZR_AX + 0.02), DVec3::new(0.11, 0.10, 0.08), 0.05, 3, Mat::MetalBare));
            ub.add(&p::tube_pipe(&[DVec3::new(0.0, 0.27, 0.90), DVec3::new(0.0, 0.255, -0.40), DVec3::new(0.0, 0.27, -1.24)], 0.033, Mat::MetalBare, 2, 14));
            out.insert("underbody", ub);
            let mut ex = Mesh::new();
            for sg in [1.0f64, -1.0] {
                let pts: Vec<DVec3> = [(0.30, 0.34, 1.32), (0.33, 0.25, 1.13), (0.31, 0.181, 0.98), (0.30, 0.178, 0.30), (0.30, 0.175, -0.60), (0.36, 0.185, -1.15), (0.40, 0.20, -1.45), (0.44, 0.22, -1.85), (0.50, 0.30, -2.10), (0.57, 0.415, -2.25), (0.58, 0.425, -2.355)]
                    .iter()
                    .map(|&(x, y, z)| DVec3::new(sg * x, y, z))
                    .collect();
                ex.add(&p::tube_pipe(&pts, 0.026, Mat::MetalBare, 5, 14));
                ex.add(&rbox(DVec3::new(sg * 0.30, 0.177, -0.85), DVec3::new(0.10, 0.026, 0.22), 0.02, 2, Mat::MetalBare));
                // the pipes end behind the rear valance; a bright bezel and a
                // dark opening on its face mark the cut-outs they leave through
                let ce = DVec3::new(sg * 0.58, 0.425, rear_z(0.58, 0.425));
                let g = p::graph_normal(&rear_z, 0.58, 0.425, -1.0);
                let ne = DVec3::new(g.x * sg, g.y, g.z);
                let mut ring = closed_lathe(&[(0.031, 0.0), (0.041, 0.0), (0.041, 0.003), (0.031, 0.003)], 32, Mat::Chrome);
                let mut hole = closed_lathe(&[(0.002, 0.0), (0.0305, 0.0), (0.0305, 0.0015), (0.002, 0.0015)], 32, Mat::TrimBlack);
                for m_ in [&mut ring, &mut hole] {
                    m_.transform(Some(p::rot_to(ne)), ce + ne * 0.0015);
                    ex.add(m_);
                }
            }
            out.insert("exhaust", ex);
        }
        for (tag, zc, r) in [("F", ZF_AX, 0.360), ("R", ZR_AX, 0.372)] {
            for (s, sg) in sides {
                // (solid build: the inner wall's opening lets the arms through)
                out.insert(&format!("wheel_well_{tag}{s}"), thick(p::wheel_well_with(DVec3::new(0.0, WHEEL_Y, zc), r, 0.535, 0.835, sg, 92.0, 92.0, if solid { 0.74 } else { 0.55 }), 0.004));
            }
        }
    }

    fn parts_markers(&self, cb: &CarBody, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // markers: closures
        for (s, sg) in sides {
            for (nm, y) in [("upper", 0.76), ("lower", 0.42)] {
                let (pp, nn) = p::surface_point(tube, 0.565, y, sg, (1.5, 5.5));
                marker(format!("hinge_door_F{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            }
            let (pp, nn) = p::surface_point(tube, -0.700, 0.62, sg, (1.5, 5.5));
            marker(format!("latch_door_F{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            marker(format!("hinge_hood_{s}"), DVec3::new(sg * 0.68, 0.860, 0.60), DVec3::X);
            marker(format!("hinge_trunk_{s}"), DVec3::new(sg * 0.55, 0.925, -1.72), DVec3::X);
            for (k, (z, y)) in [(2.10, 0.79), (1.70, 0.82), (1.10, 0.85), (0.70, 0.87)].iter().enumerate() {
                let x = if *z < 2.0 { 0.78 } else { 0.70 };
                marker(format!("bolt_fender_F{s}_{}", k + 1), DVec3::new(sg * x, y - 0.03, *z), DVec3::X);
            }
        }
        marker("latch_hood".into(), DVec3::new(0.0, 0.775, 2.25), DVec3::X);
        marker("latch_trunk".into(), DVec3::new(0.0, 0.905, -2.39), DVec3::X);
    }
}
