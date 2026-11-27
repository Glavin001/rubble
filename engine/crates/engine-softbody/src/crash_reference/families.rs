//! The scenario families ([`super::family`]): each kind of crash the suite stages, laid out
//! once. The reference scenarios are presets of these (S01, S02, S07, S10, S11, S12 of
//! [`BLOCK`]; S08, S09 of [`HEAD_ON`]; S03, S14 of [`SIDE_BARRIER`]; S13 of [`SIDE_POLE`];
//! S05 of [`TBONE`]; S04 of [`UNDERRIDE`]; S06 of [`SANDWICH`]); [`POLE`], [`LOW_BAR`],
//! [`DROP`] and [`COAST`] have no reference preset yet and are opened directly.
//!
//! Each layout keeps the arithmetic its scenarios stated (the order of every operation), so a
//! preset builds the fleet its scenario built word for word
//! (`engine-softbody-lab/tests/crash_fleet_digests.rs`). Where a new parameter generalises a
//! scenario (an angle, an aim point), its neutral value takes the stated arithmetic and any
//! other value the general one.
//!
//! Conventions (as [`Body`]): forward +Z, yaw from +Z toward +X, a car's left is +X; every
//! speed is multiplied by the run's jitter and the striking body moved across its path and
//! turned by it.

use super::family::{degrees, design, kmh, num, Family, Param, Values};
use super::*;
use crate::crash_rigs::MDB_FACE;
use std::f32::consts::PI;

/// Every family, in the picker's order.
pub fn all() -> Vec<&'static Family> {
    vec![&BLOCK, &POLE, &SIDE_POLE, &HEAD_ON, &TBONE, &SIDE_BARRIER, &UNDERRIDE, &SANDWICH, &LOW_BAR, &DROP, &COAST]
}

fn lead(default: f64) -> Param {
    num("lead", "lead", "s", default, 0.02, 2.0, 0.01).help("seconds of travel before first contact")
}
fn run_up(default: f64) -> Param {
    num("run_up", "run-up", "s", default, 0.05, 2.0, 0.05).help("seconds of travel before first contact (the struck body settles on its suspension meanwhile)")
}

// ------------------------------------------------------------------ a car into a block

/// A car into a rigid block: a full-width wall, or a block over a share of its front.
pub static BLOCK: Family = Family {
    id: "block",
    title: "Car into a rigid block or wall",
    about: "A car at a speed into a fixed block a lead time ahead: full width (overlap 1: a wall a metre past each side) or over a share of the car's width from its left (+X) side.",
    params: || {
        vec![
            design("car", "car", "sedan"),
            kmh("speed", "speed", 56.3),
            num("overlap", "overlap", "of width", 1.0, 0.05, 1.0, 0.05).help("the share of the car's width the block covers, from its left (+X) side; 1: a full-width wall a metre past each side"),
            num("block_width", "block width", "m", 0.0, 0.0, 5.0, 0.05).help("0: from the overlap's edge to a metre past the car's side; else the block's width from that edge"),
            num("height", "height", "m", 4.0, 0.2, 6.0, 0.1),
            num("depth", "depth", "m", 1.0, 0.2, 4.0, 0.1),
            degrees("angle", "block angle", 0.0, 60.0).help("the block turned about its vertical axis (an oblique wall); first contact is then computed"),
            lead(0.1),
        ]
    },
    layout: block,
    boxes: true,
    secs_after_impact: 0.5,
};

fn block(v: &Values, extent: &dyn Fn(&str) -> Extent, j: Jitter) -> Layout {
    let car = v.text("car");
    let e = extent(car);
    let speed = v.ms("speed") * j.speed;
    let gap = v.f32("lead") * speed;
    let face = e.hi.z + gap;
    let (h, d, overlap) = (v.f32("height"), v.f32("depth"), v.f32("overlap"));
    let b = if overlap >= 1.0 {
        (Vec3::new(0.0, 0.5 * h, face + 0.5 * d), Vec3::new(0.5 * e.width() + 1.0, 0.5 * h, 0.5 * d))
    } else if v.f32("block_width") > 0.0 {
        let (x0, w) = (e.hi.x - overlap * e.width(), v.f32("block_width"));
        (Vec3::new(x0 + 0.5 * w, 0.5 * h, face + 0.5 * d), Vec3::new(0.5 * w, 0.5 * h, 0.5 * d))
    } else {
        let (x0, x1) = (e.hi.x - overlap * e.width(), e.hi.x + 1.0);
        (Vec3::new(0.5 * (x0 + x1), 0.5 * h, face + 0.5 * d), Vec3::new(0.5 * (x1 - x0), 0.5 * h, 0.5 * d))
    };
    let mut l = Layout { box_yaw: Vec::new(), bodies: vec![Body::new("car", car, Vec3::new(j.lateral, 0.0, 0.0), j.yaw, speed)], boxes: vec![b], impact_s: v.f32("lead") };
    turned_boxes(&mut l, v.rad("angle"), extent);
    l
}

/// Turn every box of `l` by `yaw` about its centre and compute the first contact (nothing
/// changes at 0).
fn turned_boxes(l: &mut Layout, yaw: f32, extent: &dyn Fn(&str) -> Extent) {
    if yaw != 0.0 {
        l.box_yaw = vec![yaw; l.boxes.len()];
        if let Some(t) = desc::first_contact(l, extent) {
            l.impact_s = t;
        }
    }
}

// ------------------------------------------------------------------ a car into a pole

/// A car into a square pole on its path (no reference preset yet: the lab's `pole` scene's).
pub static POLE: Family = Family {
    id: "pole",
    title: "Car into a pole",
    about: "A car at a speed into a fixed square pole on its path, offset across it.",
    params: || {
        vec![
            design("car", "car", "sedan"),
            kmh("speed", "speed", 60.0),
            num("offset", "offset", "m", 0.0, -1.5, 1.5, 0.01).help("the pole's centre across the car's path, toward its left (+X)"),
            degrees("angle", "heading", 0.0, 60.0).help("the car's heading; the pole stays on its path"),
            num("pole_width", "pole width", "m", 0.2, 0.05, 1.0, 0.01),
            lead(0.1),
        ]
    },
    layout: |v, extent, j| {
        let car = v.text("car");
        let e = extent(car);
        let speed = v.ms("speed") * j.speed;
        let a = v.rad("angle");
        let half = 0.5 * v.f32("pole_width");
        let ahead = e.hi.z + v.f32("lead") * speed + half;
        let (s, c) = a.sin_cos();
        let off = v.f32("offset");
        let centre = Vec3::new(ahead * s + off * c, 2.0, ahead * c - off * s);
        Layout {
            box_yaw: vec![a],
            bodies: vec![Body::new("car", car, Vec3::new(j.lateral, 0.0, 0.0), a + j.yaw, speed)],
            boxes: vec![(centre, Vec3::new(half, 2.0, half))],
            impact_s: v.f32("lead"),
        }
    },
    boxes: true,
    secs_after_impact: 0.5,
};

// ------------------------------------------------------------------ side into a pole

/// A car carried sideways on a carrier into a pole aimed at a point of its side (S13).
pub static SIDE_POLE: Family = Family {
    id: "side_pole",
    title: "Car carried sideways into a pole",
    about: "A car on a side-impact carrier (its tyres on low-friction pads) travelling at an angle from its forward axis into a square pole whose centre lies on the travel line through an aim point of the car (the driver's head CG by default).",
    params: || {
        vec![
            design("car", "car", "sedan"),
            kmh("speed", "speed", 32.0),
            num("angle", "travel angle", "°", 75.0, 15.0, 165.0, 1.0).help("the travel's angle from the car's forward axis, toward its left (90: square to the side)"),
            num("pole_width", "pole width", "m", 0.254, 0.05, 1.0, 0.001),
            num("aim_left", "aim: left of centre", "m", 0.38, -1.0, 1.0, 0.01).help("the aimed point's distance left of the centre line"),
            num("aim_behind_axle", "aim: behind front axle", "m", 2.146, -1.5, 4.0, 0.01).help("the aimed point's distance behind the front axle"),
            lead(0.1),
        ]
    },
    layout: |v, extent, j| {
        let car = v.text("car");
        let e = extent(car);
        let speed = v.ms("speed") * j.speed;
        let a = v.rad("angle");
        // Travel: toward the car's left (+X at yaw 0) and a little forward (+Z).
        let u = Vec3::new(a.sin(), 0.0, a.cos());
        let cot = a.cos() / a.sin();
        let front = e.axles.map_or(e.hi.z, |x| x.0);
        let (hx, hz) = (v.f32("aim_left"), front - v.f32("aim_behind_axle"));
        let half = 0.5 * v.f32("pole_width");
        // The impact line on the side's outermost plane, and the pole's centre on the
        // travel line through the aim, a half width out from that plane.
        let side = e.hi.x;
        let z_irl = hz + (side - hx) * cot;
        let pole = (Vec3::new(side + half, 2.0, z_irl + half * cot), Vec3::new(half, 2.0, half));
        let impact_s = v.f32("lead");
        let across = Vec3::new(-u.z, 0.0, u.x);
        let at = -u * (speed * impact_s) + across * j.lateral;
        let mut body = Body::new("car", car, at, j.yaw, speed).on_carrier();
        body.vel = Some(u * speed);
        Layout { box_yaw: Vec::new(), bodies: vec![body], boxes: vec![pole], impact_s }
    },
    boxes: true,
    secs_after_impact: 0.3,
};

// ------------------------------------------------------------------ head-on

/// Two vehicles head-on, overlapping by a share of the first's width (S08, S09).
pub static HEAD_ON: Family = Family {
    id: "head_on",
    title: "Two vehicles head-on",
    about: "Two vehicles nose to nose, closing for a lead time; the second faces the first, its left side overlapping the first's left (+X) side by a share of the first's width.",
    params: || {
        vec![
            design("car_a", "vehicle A", "sedan"),
            design("car_b", "vehicle B", "sedan"),
            kmh("speed", "A's speed", 55.0),
            kmh("speed_b", "B's speed", 55.0),
            num("overlap", "overlap", "of A's width", 0.5, 0.05, 1.0, 0.05),
            degrees("angle", "B's angle", 0.0, 60.0).help("vehicle B turned from straight on (about its own origin); first contact is then computed"),
            lead(0.1),
        ]
    },
    layout: |v, extent, j| {
        let (a, b) = (v.text("car_a"), v.text("car_b"));
        let (ea, eb) = (extent(a), extent(b));
        let (va, vb) = (v.ms("speed") * j.speed, v.ms("speed_b") * j.speed);
        let gap = v.f32("lead") * (va + vb);
        let ov = v.f32("overlap");
        // B faces A (yaw π: its left is world −X). Its world x span is [xb − eb.hi.x,
        // xb − eb.lo.x]; its left edge stands `ov` of A's width in from A's left (+X) edge.
        let (xb, zb) = if a == b { (2.0 * ea.hi.x - ov * ea.width(), 2.0 * ea.hi.z + gap) } else { (ea.hi.x - ov * ea.width() + eb.hi.x, ea.hi.z + eb.hi.z + gap) };
        let angle = v.rad("angle");
        let mut l = Layout {
            box_yaw: Vec::new(),
            bodies: vec![Body::new("car_a", a, Vec3::new(j.lateral, 0.0, 0.0), j.yaw, va), Body::new("car_b", b, Vec3::new(xb, 0.0, zb), PI + angle, vb)],
            boxes: vec![],
            impact_s: v.f32("lead"),
        };
        if angle != 0.0 {
            if let Some(t) = desc::first_contact(&l, extent) {
                l.impact_s = t;
            }
        }
        l
    },
    boxes: false,
    secs_after_impact: 1.0,
};

// ------------------------------------------------------------------ T-bone

/// A striking vehicle into the side of a standing (or moving) one (S05).
pub static TBONE: Family = Family {
    id: "tbone",
    title: "Vehicle into the side of another (T-bone)",
    about: "A striking vehicle square to (or at an angle off) the struck vehicle's left (+X) side, aimed at a point along it from its wheelbase's middle; the struck vehicle stands or moves forward so it is there at first contact.",
    params: || {
        vec![
            design("striker", "striking", "truck_cab"),
            design("struck", "struck", "sedan"),
            kmh("speed", "striking speed", 70.0),
            kmh("struck_speed", "struck speed", 0.0),
            num("aim", "aim along the side", "m", 0.0, -3.0, 3.0, 0.05).help("forward of the struck vehicle's wheelbase middle"),
            degrees("angle", "angle off square", 0.0, 75.0).help("the striker's heading turned from square to the side"),
            run_up(0.4),
        ]
    },
    layout: |v, extent, j| {
        let (sd, cd) = (v.text("striker"), v.text("struck"));
        let (c, t) = (extent(cd), extent(sd));
        let speed = v.ms("speed") * j.speed;
        let ru = v.f32("run_up");
        let (front, rear) = c.axles.unwrap_or((c.lo.z + 0.2 * c.length(), c.lo.z + 0.8 * c.length()));
        let mid = 0.5 * (front + rear);
        let angle = v.rad("angle");
        let yaw = -0.5 * PI + angle;
        // The striker faces −X (yaw −90°), its bumper (`t.hi.z` ahead of its origin) a run-up
        // off the struck side; its centre line (its own x = 0) on the aimed point.
        let at = if angle == 0.0 {
            Vec3::new(c.hi.x + ru * speed + t.hi.z, 0.0, mid + v.f32("aim") + j.lateral)
        } else {
            let h = Vec3::new(yaw.sin(), 0.0, yaw.cos());
            Vec3::new(c.hi.x, 0.0, mid + v.f32("aim") + j.lateral) - h * (t.hi.z + ru * speed)
        };
        let struck_v = v.ms("struck_speed") * j.speed;
        Layout {
            box_yaw: Vec::new(),
            bodies: vec![Body::new("striker", sd, at, yaw + j.yaw, speed), Body::new("struck", cd, Vec3::new(0.0, 0.0, 0.0 - struck_v * ru), 0.0, struck_v)],
            boxes: vec![],
            impact_s: ru,
        }
    },
    boxes: false,
    secs_after_impact: 1.0,
};

// ------------------------------------------------------------------ moving barrier

/// A moving barrier into the left side of a standing car, square or crabbed (S03, S14).
pub static SIDE_BARRIER: Family = Family {
    id: "side_barrier",
    title: "Moving barrier into a car's side",
    about: "A moving barrier (FMVSS 214's by default) facing the standing car's left (+X) side, its face's forward edge on FMVSS 214's impact reference line (S8.10) moved by the aim; crabbed, it moves rearward along the side as it closes.",
    params: || {
        vec![
            design("car", "struck car", "sedan"),
            design("barrier", "barrier", "mdb214"),
            kmh("speed", "barrier speed", 50.0),
            num("crab", "crab", "°", 0.0, 0.0, 45.0, 1.0).help("the barrier's wheels turned toward the car's rear: it moves at this angle off square, its face still square to the side"),
            num("aim", "aim along the side", "m", 0.0, -2.0, 2.0, 0.05).help("the face moved forward of FMVSS 214's impact reference line"),
            run_up(0.4),
        ]
    },
    layout: |v, extent, j| {
        let (cd, bd) = (v.text("car"), v.text("barrier"));
        let (e, b) = (extent(cd), extent(bd));
        let speed = v.ms("speed") * j.speed;
        let ru = v.f32("run_up");
        let (front, rear) = e.axles.unwrap_or((e.lo.z + 0.2 * e.length(), e.lo.z + 0.8 * e.length()));
        // FMVSS 214 S8.10: the face's forward edge on the impact reference line.
        let irl = if front - rear <= 2.896 { 0.5 * (front + rear) + 0.94 } else { front - 0.508 };
        let face_z = irl - 0.5 * MDB_FACE[0] + v.f32("aim") + j.lateral;
        let crab = v.rad("crab");
        let car = Body::new("car", cd, Vec3::ZERO, 0.0, 0.0);
        let barrier = if crab == 0.0 {
            // Facing −X (yaw −90°), along its heading: its bumper's front (`b.hi.z` ahead of
            // its origin) a run-up short of the car's side.
            let x = e.hi.x + ru * speed + b.hi.z;
            Body::new("barrier", bd, Vec3::new(x, 0.0, face_z), -0.5 * PI + j.yaw, speed)
        } else {
            let dir = Vec3::new(-crab.cos(), 0.0, -crab.sin());
            // Its origin when its face reaches the side's outermost plane, and a run-up back
            // along its travel.
            let contact = Vec3::new(e.hi.x + b.hi.z, 0.0, face_z);
            let mut x = Body::new("barrier", bd, contact - dir * (speed * ru), -0.5 * PI + j.yaw, speed);
            x.vel = Some(dir * speed);
            x
        };
        Layout { box_yaw: Vec::new(), bodies: vec![car, barrier], boxes: Vec::new(), impact_s: ru }
    },
    boxes: false,
    secs_after_impact: 0.6,
};

// ------------------------------------------------------------------ underride

/// A car into the rear of a standing trailer (S04).
pub static UNDERRIDE: Family = Family {
    id: "underride",
    title: "Car under the rear of a trailer",
    about: "A car into the rear of a standing trailer (heading the same way), offset so a share of its width is behind it.",
    params: || {
        vec![
            design("car", "car", "sedan"),
            design("trailer", "trailer", "box_trailer"),
            kmh("speed", "speed", 56.0),
            num("overlap", "overlap", "of width", 1.0, 0.05, 1.0, 0.05).help("the share of the car's width behind the trailer: less moves the car toward its left (+X)"),
            run_up(0.5),
        ]
    },
    layout: |v, extent, j| {
        let (cd, td) = (v.text("car"), v.text("trailer"));
        let (c, t) = (extent(cd), extent(td));
        let speed = v.ms("speed") * j.speed;
        let ru = v.f32("run_up");
        // The trailer heads +Z, its rear (`t.lo.z` behind its origin) a run-up ahead of the
        // car's front.
        let z = c.hi.z + ru * speed - t.lo.z;
        let x = (1.0 - v.f32("overlap")) * c.width();
        Layout {
            box_yaw: Vec::new(),
            bodies: vec![Body::new("car", cd, Vec3::new(x + j.lateral, 0.0, 0.0), j.yaw, speed), Body::new("trailer", td, Vec3::new(0.0, 0.0, z), 0.0, 0.0)],
            boxes: vec![],
            impact_s: ru,
        }
    },
    boxes: false,
    secs_after_impact: 0.8,
};

// ------------------------------------------------------------------ sandwich

/// A striking vehicle shunting a standing car into a standing trailer ahead of it (S06).
pub static SANDWICH: Family = Family {
    id: "sandwich",
    title: "Car shunted into a trailer (sandwich)",
    about: "A striking vehicle into the rear of a standing car, which stands a gap behind a standing trailer.",
    params: || {
        vec![
            design("striker", "striking", "truck_cab"),
            design("car", "car", "sedan"),
            design("trailer", "trailer", "box_trailer"),
            kmh("speed", "striking speed", 45.0),
            num("gap", "gap to the trailer", "m", 1.0, 0.0, 10.0, 0.05).help("from the car's nose to the trailer's rear"),
            run_up(0.5),
        ]
    },
    layout: |v, extent, j| {
        let (sd, cd, td) = (v.text("striker"), v.text("car"), v.text("trailer"));
        let (c, t, tr) = (extent(cd), extent(sd), extent(td));
        let speed = v.ms("speed") * j.speed;
        let ru = v.f32("run_up");
        // the trailer's rear `gap` ahead of the car's nose; the striker's front a run-up behind
        // the car's tail
        let z_trailer = c.hi.z + v.f32("gap") - tr.lo.z;
        let z_truck = c.lo.z - ru * speed - t.hi.z;
        Layout {
            box_yaw: Vec::new(),
            bodies: vec![
                Body::new("striker", sd, Vec3::new(j.lateral, 0.0, z_truck), j.yaw, speed),
                Body::new("car", cd, Vec3::ZERO, 0.0, 0.0),
                Body::new("trailer", td, Vec3::new(0.0, 0.0, z_trailer), 0.0, 0.0),
            ],
            boxes: vec![],
            impact_s: ru,
        }
    },
    boxes: false,
    secs_after_impact: 1.4,
};

// ------------------------------------------------------------------ low bar

/// A car under a low rigid bar (the lab's `limbo` scene's).
pub static LOW_BAR: Family = Family {
    id: "low_bar",
    title: "Car under a low bar",
    about: "A car at a speed under a fixed bar 30 cm deep across its path, a metre past each side.",
    params: || vec![design("car", "car", "sedan"), kmh("speed", "speed", 60.0), num("bar", "bar underside", "m", 1.0, 0.3, 2.0, 0.05), lead(0.1)],
    layout: |v, extent, j| {
        let car = v.text("car");
        let e = extent(car);
        let speed = v.ms("speed") * j.speed;
        let face = e.hi.z + v.f32("lead") * speed;
        let bar = (Vec3::new(0.0, v.f32("bar") + 0.15, face + 0.15), Vec3::new(0.5 * e.width() + 1.0, 0.15, 0.15));
        Layout { box_yaw: Vec::new(), bodies: vec![Body::new("car", car, Vec3::new(j.lateral, 0.0, 0.0), j.yaw, speed)], boxes: vec![bar], impact_s: v.f32("lead") }
    },
    boxes: true,
    secs_after_impact: 0.6,
};

// ------------------------------------------------------------------ drop

/// A car dropped flat onto the ground (the lab's `drop` scene's).
pub static DROP: Family = Family {
    id: "drop",
    title: "Car dropped onto the ground",
    about: "A car dropped flat from a height (its wheels' lowest point that far up), moving forward at a speed.",
    params: || vec![design("car", "car", "sedan"), num("height", "height", "m", 8.0, 0.5, 40.0, 0.5), kmh("speed", "forward speed", 0.0), degrees("yaw", "heading", 0.0, 180.0)],
    layout: |v, _extent, j| {
        let car = v.text("car");
        let h = v.f32("height");
        Layout {
            box_yaw: Vec::new(),
            bodies: vec![Body::new("car", car, Vec3::new(j.lateral, h, 0.0), v.rad("yaw") + j.yaw, v.ms("speed") * j.speed)],
            boxes: vec![],
            // Free fall from h to the ground.
            impact_s: (2.0 * h / 9.81).sqrt(),
        }
    },
    boxes: false,
    secs_after_impact: 1.0,
};

// ------------------------------------------------------------------ coast / at rest

/// A car at rest (speed 0) or coasting on its wheels (the lab's `idle` and `coast` scenes').
pub static COAST: Family = Family {
    id: "coast",
    title: "Car at rest or coasting",
    about: "A car on the ground at rest (speed 0) or rolling forward at a speed for two seconds; nothing to hit.",
    params: || vec![design("car", "car", "sedan"), kmh("speed", "speed", 0.0)],
    layout: |v, _extent, j| Layout {
        box_yaw: Vec::new(),
        bodies: vec![Body::new("car", v.text("car"), Vec3::new(j.lateral, 0.0, 0.0), j.yaw, v.ms("speed") * j.speed)],
        boxes: vec![],
        // No contact: the run counts from its start.
        impact_s: 0.0,
    },
    boxes: false,
    secs_after_impact: 2.0,
};
