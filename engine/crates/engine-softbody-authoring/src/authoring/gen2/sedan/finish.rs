//! The last steps of the sedan's build (split from `sedan.rs`): the wheels and
//! reference nodes, the springs, and the yield by zone.

use super::*;

/// The wheels, the reference nodes, the springs and the yield by zone.
pub(super) fn wheels_springs_yield(c: &mut Car, body: Part, fw_y: f32) {
    // Wheels and reference nodes.
    for (name, w, s, y) in [
        ("FL", "F", 1.0f32, AXLE_F),
        ("FR", "F", -1.0, AXLE_F),
        ("RL", "R", 1.0, AXLE_R),
        ("RR", "R", -1.0, AXLE_R),
    ] {
        let x = lr(s);
        let _ = y;
        let (n1, n2) = (c.id(&format!("hub{w}i{x}")), c.id(&format!("hub{w}o{x}")));
        let arm = if w == "F" {
            c.id(&format!("twr{x}"))
        } else {
            c.id(&format!("wh{x}"))
        };
        c.wheels.push(Wheel {
            name,
            n1,
            n2,
            arm,
            dir: s,
            size: None,
        });
    }
    c.refs = [c.id("tn1"), c.id("tn2"), c.id("si1l"), c.id("rfC1")];
    let hub = hub_extra(c);
    c.assign_springs(&hub);
    // Yield by zone: a member's webs and ties are its section as much as its
    // chords, so the front crush zone's webs and cross ties take the crush
    // multiple and the cabin's (the safety cage: DP980/PHS) the cabin multiple.
    for b in &mut c.beams {
        if !matches!(b.part, Part::Body | Part::BumperF | Part::BumperR) || !b.deform.is_finite() {
            continue;
        }
        let mid = 0.5 * (c.nodes[b.a].p + c.nodes[b.b].p);
        // Crushed sheet steel stacks at about a third of its length (a folded
        // crush can's solid length): JBeam's deformLimit on the crush zones'
        // beams, so a crushed bay stacks instead of inverting.
        if (mid.y < fw_y - 0.05 || mid.y > 2.9)
            && !matches!(b.role, Role::Sheet | Role::Bend | Role::Tie | Role::Backing)
        {
            b.deform_limit = STACK_LIMIT;
        }
        if b.part != body {
            continue;
        }
        let web = matches!(
            b.role,
            Role::Web
                | Role::Cross
                | Role::Firewall
                | Role::Floor
                | Role::Tunnel
                | Role::RoofBow
                | Role::Sheet
        );
        if !web {
            continue;
        }
        if mid.y < fw_y - 0.05 {
            b.deform *= crush_mult();
        } else if mid.y < 2.9 {
            b.deform *= cabin_mult();
        } else {
            b.deform *= crush_mult();
        }
    }
}
