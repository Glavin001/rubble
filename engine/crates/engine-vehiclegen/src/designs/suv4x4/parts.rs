//! The 4x4's parts: wheels, lamps, grille, bumpers, trim, interior, underbody
//! and the markers, built on the car body (split from `suv4x4.rs`).

use super::*;

impl Suv4x4 {
    pub(super) fn make_parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        let mut out = Parts::default();
        let mut extras = Parts::default();
        let mut mk: Vec<Marker> = Vec::new();
        self.parts_wheels(&mut out, &mut mk);
        self.parts_lamps(cb, &mut out);
        self.parts_grille_bumpers(cb, &mut out, &mut extras);
        self.parts_mirrors(cb, &mut out, &mut mk);
        self.parts_extras(cb, &mut extras);
        self.parts_interior(cb, &mut out, &mut mk);
        self.parts_underbody(cb, &mut out);
        self.parts_markers(cb, &mut mk);
        (out, extras, mk)
    }

    fn parts_wheels(&self, out: &mut Parts, mk: &mut Vec<Marker>) {
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // wheels
        let spec = self.wheel_spec();
        for (tag, z) in [("F", ZF_AX), ("R", ZR_AX)] {
            for (s, sg) in sides {
                let (rim, ty, di, c) = p::wheel(&spec, sg, z);
                out.insert(&format!("rim_{tag}{s}"), rim);
                out.insert(&format!("tyre_{tag}{s}"), ty);
                out.insert(&format!("brake_disc_{tag}{s}"), di);
                marker(format!("wheel_center_{tag}{s}"), c, DVec3::new(sg, 0.0, 0.0));
            }
        }
    }

    fn parts_lamps(&self, cb: &CarBody, out: &mut Parts) {
        let solid = cb.extra.contains_key("solid");
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        // lamp housings: reflector shells behind the lenses, chrome cups in front
        for (s, _sg) in sides {
            for kind in ["headlight", "taillight"] {
                let sk = cb.skins[&format!("{kind}_{s}_lens")].clone();
                let mut isb = vec![false; sk.v.len()];
                for lp in boundary_loops(&sk) {
                    for i in lp {
                        isb[i as usize] = true;
                    }
                }
                let mut h = if solid {
                    crate::solid::reflector(&sk, 0.034, 0.003, Mat::LampEmissive)
                } else {
                    let mut h = inner_panel(&sk, &isb, 0.034, -1_000_000, 1, Mat::LampEmissive);
                    h.flip();
                    h
                };
                if kind == "headlight" {
                    let c0 = sk.v.iter().copied().sum::<DVec3>() / sk.v.len() as f64;
                    let vn = sk.vertex_normals();
                    let mut best = 0;
                    for (i, v) in sk.v.iter().enumerate() {
                        if (*v - c0).length() < (sk.v[best] - c0).length() {
                            best = i;
                        }
                    }
                    let n = vn[best];
                    let mut cup = closed_lathe(&[(0.0, 0.002), (0.045, 0.002), (0.050, -0.010), (0.016, -0.022), (0.0, -0.022)], 32, Mat::Chrome);
                    cup.transform(Some(p::rot_to(n)), sk.v[best] - n * 0.036);
                    h.add(&cup);
                }
                out.insert(&format!("{kind}_{s}_housing"), h);
            }
        }
    }

    fn parts_grille_bumpers(&self, cb: &CarBody, out: &mut Parts, extras: &mut Parts) {
        // grille: vertical slats in the opening, a dark back plate
        let mut g = Mesh::new();
        let gx = cb.extra["grille_cols"][0].abs() - 0.03;
        let (ya, yb) = (0.785, 1.080);
        let xs = linspace(-gx, gx, 30);
        let ys = linspace(ya, yb, 8);
        let pl = Grid2::from_fn(xs.len(), ys.len(), |i, j| DVec3::new(xs[i], ys[j], front_z(xs[i], ys[j]) - 0.050));
        g.add(&grid(&pl, None, Mat::TrimBlack, false, false, -1));
        let mut x = -gx + 0.02;
        while x < gx - 0.01 {
            let zc = front_z(x, (ya + yb) / 2.0) - 0.030;
            g.add(&box_(DVec3::new(x, (ya + yb) / 2.0, zc), DVec3::new(0.010, (yb - ya) / 2.0 - 0.010, 0.012), Mat::TrimBlack, None));
            x += 0.045;
        }
        out.insert("grille", g);

        // steel bumpers: a square tube bar on brackets, front and rear
        let bar = |zf: f64, y: f64, sgn: f64, half_w: f64| {
            let mut b = Mesh::new();
            b.add(&rbox(DVec3::new(0.0, y, zf + sgn * 0.095), DVec3::new(half_w, 0.065, 0.055), 0.012, 2, Mat::TrimBlack));
            for x in [-0.55, 0.55] {
                b.add(&rbox(DVec3::new(x, y - 0.02, zf + sgn * 0.030), DVec3::new(0.04, 0.04, 0.022), 0.006, 1, Mat::TrimBlack));
            }
            b
        };
        extras.insert("bumper_front", bar(front_z(0.5, 0.60), 0.600, 1.0, 0.92));
        extras.insert("bumper_rear", bar(rear_z(0.5, 0.60), 0.600, -1.0, 0.90));

        // spare wheel on the tailgate
        if self.p.spare_wheel {
            let zt = rear_z(0.0, 1.05);
            let mut spare = Mesh::new();
            spare.add(&p::rim_steel(RIM_D, RIM_W, 64));
            spare.add(&p::tyre(TYRE_D, TYRE_W, RIM_D, RIM_W, 96, 3));
            spare.transform(Some(rot_y(std::f64::consts::FRAC_PI_2)), DVec3::new(0.0, 1.09, zt - TYRE_W / 2.0 - 0.045));
            spare.add(&rbox(DVec3::new(0.0, 1.09, zt - 0.024), DVec3::new(0.06, 0.06, 0.018), 0.01, 1, Mat::MetalBare));
            out.insert("spare_wheel", spare);
        }
    }

    fn parts_mirrors(&self, cb: &CarBody, out: &mut Parts, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let vm = tube.vm;
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // mirrors, handles, wipers
        for (s, sg) in sides {
            let v = if sg > 0.0 { 5.8 } else { vm - 5.8 };
            let (base, bn) = tube.eval_n(0.62, v);
            let mut m = Mesh::new();
            m.add(&rbox(base + bn * 0.028, DVec3::new(0.012, 0.022, 0.035), 0.008, 2, Mat::TrimBlack));
            let hc = base + bn * 0.028 + DVec3::new(sg * 0.11, 0.07, -0.02);
            m.add(&rbox(hc, DVec3::new(0.10, 0.075, 0.035), 0.020, 3, Mat::TrimBlack));
            m.add(&rbox(hc + DVec3::new(0.0, 0.0, -0.036), DVec3::new(0.088, 0.064, 0.0015), 0.0014, 1, Mat::Chrome));
            m.add(&rbox(base + bn * 0.028 + DVec3::new(sg * 0.05, 0.045, -0.01), DVec3::new(0.045, 0.012, 0.016), 0.008, 2, Mat::TrimBlack));
            out.insert(&format!("mirror_{s}"), m);
            marker(format!("mount_mirror_{s}"), base, DVec3::new(sg, 0.0, 0.0));
            out.insert(&format!("handle_F{s}"), p::door_handle(tube, 0.04, 1.06, sg, 0.15));
            out.insert(&format!("handle_R{s}"), p::door_handle(tube, -0.72, 1.06, sg, 0.15));
        }
        for (s, x0, x1) in [("L", 0.62, 0.06), ("R", 0.02, -0.54)] {
            let zs = 0.66;
            let pts: Vec<DVec3> = linspace(x0, x1, 8)
                .into_iter()
                .map(|x: f64| {
                    let mut v = tube.v_at_x(zs, x.abs(), 8.0, 9.0);
                    if x < 0.0 {
                        v = vm - v;
                    }
                    let (pp, nn) = tube.eval_n(zs, v);
                    pp + nn * 0.005
                })
                .collect();
            out.insert(&format!("wiper_{s}"), p::wiper(&pts, Mat::TrimBlack));
        }
    }

    fn parts_extras(&self, cb: &CarBody, extras: &mut Parts) {
        let tube = cb.tube();
        let vm = tube.vm;
        let solid = cb.extra.contains_key("solid");
        // body extras: rock sliders, roof rack, windscreen reveal
        let mut bx = Mesh::new();
        for sg in [1.0f64, -1.0] {
            let zs_ = linspace(0.78, -0.82, 24);
            let (pp, _) = p::path_on_surface(tube, &zs_, &[0.50], sg, 0.060, (1.5, 5.5));
            bx.add(&p::tube_pipe(&pp, 0.030, Mat::TrimBlack, 1, 14));
            for z in [0.55, -0.10, -0.65] {
                let (q, nq) = p::surface_point(tube, z, 0.50, sg, (1.5, 5.5));
                bx.add(&p::tube_pipe(&[q + nq * 0.002, q + nq * 0.045], 0.016, Mat::TrimBlack, 1, 14));
            }
            if self.p.roof_rack {
                let v = if sg > 0.0 { 7.55 } else { vm - 7.55 };
                let rail: Vec<DVec3> = linspace(0.10, -2.12, 26).into_iter().map(|z| tube.eval(z, v) + DVec3::new(0.0, 0.055, 0.0)).collect();
                bx.add(&p::tube_pipe(&rail, 0.016, Mat::TrimBlack, 1, 14));
                for z in [0.0, -2.05] {
                    let q = tube.eval(z, v);
                    bx.add(&p::tube_pipe(&[q + DVec3::new(0.0, 0.004, 0.0), q + DVec3::new(0.0, 0.055, 0.0)], 0.014, Mat::TrimBlack, 1, 14));
                }
            }
        }
        if self.p.roof_rack {
            for z in [-0.35, -0.95, -1.55] {
                let a = tube.eval(z, 7.55) + DVec3::new(0.0, 0.055, 0.0);
                let b = tube.eval(z, vm - 7.55) + DVec3::new(0.0, 0.055, 0.0);
                bx.add(&p::tube_pipe(&[a, (a + b) / 2.0 + DVec3::new(0.0, 0.012, 0.0), b], 0.014, Mat::TrimBlack, 2, 14));
            }
        }
        bx.set_mat(Mat::TrimBlack);
        let ws: Vec<DVec2> = [(0.72, 8.0), (0.51, 8.0), (0.30, 8.0), (0.30, 9.0), (0.30, 10.0), (0.51, 10.0), (0.72, 10.0), (0.72, 9.0)].iter().map(|&(z, v)| DVec2::new(z, v)).collect();
        let mut rev = p::molding(tube, &ws, 0.0015, (0.010, 0.005), true, 6);
        rev.set_mat(Mat::TrimBlack);
        bx.add(&rev);
        // solid build: the B-pillar, behind the doors' shut line
        if solid {
            let z_b = ZLINES.iter().find(|l| l.0 == "BP").unwrap().1;
            for side in [1.0, -1.0] {
                bx.add(&super::sedan::b_pillar(tube, z_b, side));
            }
        }
        extras.insert("body_shell", bx);
    }

    fn parts_interior(&self, cb: &CarBody, out: &mut Parts, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let vm = tube.vm;
        let solid = cb.extra.contains_key("solid");
        let thick = |m: Mesh, th: f64| if solid { crate::solid::slab(&m, th) } else { m };
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // interior
        let hf = 0.430;
        for (s, sg) in sides {
            // (solid build: the seats, the bench, the dash and the console
            // are parts on the structure: `mechanical`)
            if !solid {
                out.insert(&format!("seat_F{s}"), p::bucket_seat(sg * 0.38, -0.30, hf, Mat::InteriorFabric, true));
            }
            marker(format!("seat_H_point_F{s}"), DVec3::new(sg * 0.38, hf + 0.18, -0.43), DVec3::X);
        }
        if !solid {
            let mut rs = Mesh::new();
            rs.add(&rbox(DVec3::new(0.0, hf + 0.20, -1.05), DVec3::new(0.52, 0.07, 0.24), 0.05, 3, Mat::InteriorFabric));
            let mut back = rbox(DVec3::ZERO, DVec3::new(0.52, 0.30, 0.06), 0.05, 3, Mat::InteriorFabric);
            back.transform(Some(rot_x((-12.0f64).to_radians())), DVec3::new(0.0, hf + 0.56, -1.32));
            rs.add(&back);
            out.insert("seat_rear", rs);
            let mut dash = Mesh::new();
            dash.add(&rbox(DVec3::new(0.0, 1.00, 0.52), DVec3::new(0.76, 0.13, 0.14), 0.04, 3, Mat::InteriorPlastic));
            dash.add(&rbox(DVec3::new(0.38, 1.15, 0.46), DVec3::new(0.17, 0.03, 0.07), 0.02, 2, Mat::InteriorPlastic));
            out.insert("dash", dash);
            out.insert("console", rbox(DVec3::new(0.0, 0.62, -0.05), DVec3::new(0.11, 0.10, 0.42), 0.03, 2, Mat::InteriorPlastic));
        }
        out.insert("steering_wheel", p::steering_wheel(DVec3::new(0.38, 1.04, 0.22), 0.19, 30.0, Mat::InteriorPlastic, if solid { 0.19 } else { 0.12 }));
        for s in ["FL", "FR", "RL", "RR"] {
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
        let zz = linspace(0.24, -2.15, 80);
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
            // inline six, radiator, firewall; ladder frame, solid axles,
            // transfer case and skid plates under it all
            let mut eb = Mesh::new();
            eb.add(&rbox(DVec3::new(0.0, 0.78, 1.38), DVec3::new(0.17, 0.20, 0.42), 0.03, 2, Mat::MetalBare));
            eb.add(&rbox(DVec3::new(0.0, 1.00, 1.38), DVec3::new(0.13, 0.03, 0.40), 0.015, 2, Mat::TrimBlack));
            eb.add(&rbox(DVec3::new(0.24, 0.85, 1.38), DVec3::new(0.06, 0.08, 0.32), 0.02, 2, Mat::MetalBare));
            eb.add(&rbox(DVec3::new(0.0, 0.70, 0.92), DVec3::new(0.17, 0.17, 0.11), 0.04, 2, Mat::MetalBare));
            out.insert("engine_block", eb);
            let mut rad = Mesh::new();
            rad.add(&box_(DVec3::new(0.0, 0.82, 1.92), DVec3::new(0.36, 0.18, 0.028), Mat::TrimBlack, None));
            rad.add(&box_(DVec3::new(0.0, 1.015, 1.92), DVec3::new(0.37, 0.015, 0.032), Mat::MetalBare, None));
            out.insert("radiator", rad);
            out.insert("firewall", box_(DVec3::new(0.0, 0.84, 0.775), DVec3::new(0.60, 0.28, 0.003), Mat::TrimBlack, None));
            let mut ub = Mesh::new();
            for sg in [1.0, -1.0] {
                ub.add(&box_(DVec3::new(sg * 0.42, 0.345, 0.0), DVec3::new(0.035, 0.045, 2.05), Mat::TrimBlack, None));
            }
            for z in [1.95, 0.70, -0.60, -2.10] {
                ub.add(&box_(DVec3::new(0.0, 0.345, z), DVec3::new(0.385, 0.035, 0.03), Mat::TrimBlack, None));
            }
            ub.add(&box_(DVec3::new(0.0, 0.300, 1.45), DVec3::new(0.32, 0.004, 0.40), Mat::MetalBare, None));
            for zc in [ZF_AX, ZR_AX] {
                let mut ax = closed_lathe(&[(0.0, 0.62), (0.050, 0.62), (0.050, -0.62), (0.0, -0.62)], 20, Mat::MetalBare);
                ax.transform(None, DVec3::new(0.0, WHEEL_Y, zc));
                ub.add(&ax);
                ub.add(&rbox(DVec3::new(0.08, WHEEL_Y, zc), DVec3::new(0.13, 0.12, 0.11), 0.06, 3, Mat::MetalBare));
            }
            ub.add(&rbox(DVec3::new(0.0, 0.315, 0.40), DVec3::new(0.12, 0.060, 0.16), 0.03, 2, Mat::MetalBare));
            ub.add(&p::tube_pipe(&[DVec3::new(0.03, 0.300, 0.20), DVec3::new(0.06, 0.290, -0.60), DVec3::new(0.08, 0.300, -1.24)], 0.030, Mat::MetalBare, 2, 14));
            ub.add(&p::tube_pipe(&[DVec3::new(0.03, 0.300, 0.56), DVec3::new(0.06, 0.300, 1.24)], 0.028, Mat::MetalBare, 1, 14));
            ub.add(&rbox(DVec3::new(-0.20, 0.30, -1.75), DVec3::new(0.30, 0.05, 0.25), 0.03, 2, Mat::MetalBare));
            out.insert("underbody", ub);
            let mut ex = Mesh::new();
            let pts: Vec<DVec3> = [(0.24, 0.66, 1.30), (0.28, 0.40, 1.08), (0.25, 0.282, 0.85), (0.25, 0.280, -0.70), (0.30, 0.282, -1.15), (0.34, 0.285, -1.55), (0.38, 0.290, -2.05), (0.50, 0.292, -2.25)]
                .iter()
                .map(|&(x, y, z)| DVec3::new(x, y, z))
                .collect();
            ex.add(&p::tube_pipe(&pts, 0.026, Mat::MetalBare, 5, 14));
            ex.add(&rbox(DVec3::new(0.25, 0.282, -0.10), DVec3::new(0.10, 0.026, 0.28), 0.02, 2, Mat::MetalBare));
            out.insert("exhaust", ex);
        }
        for (tag, zc) in [("F", ZF_AX), ("R", ZR_AX)] {
            for (s, sg) in sides {
                // (solid build: the inner wall's opening lets the arms through)
                out.insert(&format!("wheel_well_{tag}{s}"), thick(p::wheel_well_with(DVec3::new(0.0, WHEEL_Y, zc), WELL_R, WELL_X, 0.85, sg, 92.0, 92.0, if solid { 0.74 } else { 0.55 }), 0.004));
            }
        }
    }

    fn parts_markers(&self, cb: &CarBody, mk: &mut Vec<Marker>) {
        let tube = cb.tube();
        let sides = [("L", 1.0f64), ("R", -1.0f64)];
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        // markers
        for (s, sg) in sides {
            for (nm, y) in [("upper", 1.02), ("lower", 0.66)] {
                let (pp, nn) = p::surface_point(tube, 0.78, y, sg, (1.5, 5.5));
                marker(format!("hinge_door_F{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
                let (pp, nn) = p::surface_point(tube, -0.07, y, sg, (1.5, 5.5));
                marker(format!("hinge_door_R{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            }
            let (pp, nn) = p::surface_point(tube, -0.03, 0.86, sg, (1.5, 5.5));
            marker(format!("latch_door_F{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            let (pp, nn) = p::surface_point(tube, -0.86, 0.86, sg, (1.5, 5.5));
            marker(format!("latch_door_R{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            marker(format!("hinge_hood_{s}"), DVec3::new(sg * 0.68, 1.12, 0.84), DVec3::X);
            marker(format!("hinge_tailgate_{s}"), DVec3::new(sg * 0.45, 1.80, -2.24), DVec3::X);
            for (k, (z, y)) in [(2.00, 1.09), (1.60, 1.10), (1.20, 1.11), (0.88, 1.12)].iter().enumerate() {
                let x = tube.key(6, *z).x + 0.03;
                marker(format!("bolt_fender_F{s}_{}", k + 1), DVec3::new(sg * x, y - 0.03, *z), DVec3::X);
            }
        }
        for (k, x) in [-0.55f64, -0.25, 0.25, 0.55].into_iter().enumerate() {
            let zz = front_z(x.abs(), 0.60);
            marker(format!("bolt_bumper_front_{}", k + 1), DVec3::new(x, 0.60, zz - 0.03), DVec3::X);
            let zz = rear_z(x.abs(), 0.60);
            marker(format!("bolt_bumper_rear_{}", k + 1), DVec3::new(x, 0.60, zz + 0.03), DVec3::X);
        }
        marker("latch_hood".into(), DVec3::new(0.0, 1.10, 2.08), DVec3::X);
        marker("latch_tailgate".into(), DVec3::new(0.0, 0.70, -2.27), DVec3::X);
    }
}
