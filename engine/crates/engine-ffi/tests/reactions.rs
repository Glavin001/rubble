//! Every destruction reaction through the C ABI: a charge (`rbWorld_Detonate`) against the
//! reference building with blast loading, crushing and support loss on (the scene of
//! engine/docs/destruction-reactions.md, "The tall building partially destroyed") breaks
//! bonds with cause Blast, crushes the drywall infill it faces with cause Blast (its event
//! carrying mass, volume, pose, velocity and its fragments), and breaks the crushed chunk's
//! bonds with cause Crush.

use rubble::DestructibleAssetDesc;
use rubble_ffi::coords::RbCoordinates;
use rubble_ffi::*;
use std::ffi::{CStr, CString};
use std::ptr;

fn ok(r: RbResult) {
    if r != RbResult::RB_OK {
        let msg = unsafe { CStr::from_ptr(rbGetLastError()) };
        panic!("{r:?}: {}", msg.to_string_lossy());
    }
}

fn v3(v: [f32; 3]) -> RbVec3 {
    RbVec3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

const BUILDING: &[u8] =
    include_bytes!("../../engine-webgpu/fracture/packs/reference-building-crush.json");

/// A world (Rubble's coordinates) with the building placed in its pack's frame on its
/// ground, threshold model; `crush` and the reactions as given.
unsafe fn building(crush: RbCrushDef) -> (*mut RbWorld, RbDestructibleId, Result<(), String>) {
    let placement = DestructibleAssetDesc::from_pack(BUILDING)
        .unwrap()
        .stress
        .as_ref()
        .unwrap()
        .placement();
    let offset = placement.map(|x| -x as f32);
    let mut def = rbDefaultWorldDef();
    def.gravity = v3([0.0, -9.81, 0.0]);
    def.contacts = RbContacts::RB_CONTACTS_CANONICAL;
    def.enable_sleep = false;
    def.max_bodies = 4096;
    def.max_colliders = 8192;
    def.max_contacts = 65_536;
    def.max_candidates = 262_144;
    def.max_hull_vertices = 1 << 20;
    let mut w = ptr::null_mut();
    ok(rbCreateWorld(&def, &mut w));
    let mut asset = 0;
    ok(rbCreateDestructibleAssetFromPack(
        w,
        BUILDING.as_ptr(),
        BUILDING.len(),
        &mut asset,
    ));
    let mut dt = 0.0;
    ok(rbWorld_UseDestructibleAssetSolverFor(
        w,
        asset,
        RbStrengthModel::RB_STRENGTH_THRESHOLD,
        &mut dt,
    ));
    let (mut has_ground, mut point) = (false, v3([0.0; 3]));
    ok(rbWorld_GetDestructibleAssetGround(
        w,
        asset,
        &mut has_ground,
        &mut point,
    ));
    assert!(has_ground);
    let mut g = 0;
    ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
    let mut plane = 0;
    ok(rbCreatePlaneShape(
        w,
        g,
        &rbDefaultShapeDef(),
        v3([0.0, 1.0, 0.0]),
        point.y + offset[1],
        &mut plane,
    ));
    let mut dd = rbDefaultDestructibleDef();
    dd.asset = asset;
    dd.model = RbStrengthModel::RB_STRENGTH_THRESHOLD;
    dd.transform.p = v3(offset);
    dd.start_asleep = false;
    dd.blast_loading = true;
    dd.support_loss = true;
    dd.crush = crush;
    let mut id = 0;
    let r = rbCreateDestructible(w, &dd, &mut id);
    let r = if r == RbResult::RB_OK {
        Ok(())
    } else {
        Err(CStr::from_ptr(rbGetLastError())
            .to_string_lossy()
            .into_owned())
    };
    (w, id, r)
}

#[test]
fn a_charge_breaks_crushes_and_reports_every_cause_through_the_abi() {
    unsafe {
        let crush = RbCrushDef {
            enabled: true,
            ..rbDefaultDestructibleDef().crush
        };
        let (w, id, r) = building(crush);
        r.unwrap();
        // Settle 1 s, then 20 kg of TNT 0.30 m outside the ground-floor drywall infill.
        let mut blast = rbDefaultBlastDef(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        blast.position = v3([-0.9, 2.1, -2.36]);
        blast.tnt_kg = 20.0;
        let charge = glam::Vec3::new(-0.9, 2.1, -2.36);
        let (mut blast_bonds, mut crush_bonds, mut crushed) = (0u32, 0u32, 0u32);
        for n in 0..120 {
            if n == 60 {
                ok(rbWorld_Detonate(w, &blast));
            }
            ok(rbWorld_Step(w));
            let mut f: RbFrame = std::mem::zeroed();
            ok(rbWorld_Poll(w, true, true, &mut f));
            let events = std::slice::from_raw_parts(f.destruction, f.destruction_count as usize);
            for e in events {
                match e.type_ {
                    RbDestructionType::RB_BONDS_BROKEN => {
                        assert_eq!(e.destructible, id);
                        assert!(n >= 60, "nothing breaks before the charge");
                        match e.break_cause {
                            RbBreakCause::RB_BREAK_BLAST => blast_bonds += e.bond_count,
                            RbBreakCause::RB_BREAK_CRUSH => crush_bonds += e.bond_count,
                            _ => {}
                        }
                    }
                    RbDestructionType::RB_CHUNK_CRUSHED => {
                        assert_eq!(e.destructible, id);
                        assert_eq!(e.crush_cause, RbCrushCause::RB_CRUSH_BLAST);
                        assert!(
                            e.mass > 0.0 && e.volume > 0.0,
                            "{} kg, {} m^3",
                            e.mass,
                            e.volume
                        );
                        let at = glam::Vec3::new(e.transform.p.x, e.transform.p.y, e.transform.p.z);
                        assert!(
                            at.distance(charge) < 3.0,
                            "the crushed chunk at {at:?} faces the charge"
                        );
                        let v = [e.linear_velocity, e.angular_velocity];
                        assert!(v
                            .iter()
                            .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()));
                        assert!(e.peak_pressure > 0.0, "{}", e.peak_pressure);
                        // Conserve (the default): its fragments carry its mass.
                        let kids = std::slice::from_raw_parts(e.children, e.child_count as usize);
                        assert_eq!(kids.len(), 4);
                        let total: f32 = kids.iter().map(|k| k.mass).sum();
                        assert!((total - e.mass).abs() < 1e-3 * e.mass);
                        for k in kids {
                            let body = *k.bodies;
                            assert!(rbBody_IsValid(w, body));
                            let mut t = RbTransform::default();
                            ok(rbBody_GetTransform(w, body, &mut t));
                            // Live and moving with the blast's impulse (read after the
                            // step that made them, so not at the chunk's pose).
                            assert!(t.p.x.is_finite() && t.p.y.is_finite() && t.p.z.is_finite());
                        }
                        crushed += 1;
                    }
                    _ => {}
                }
            }
        }
        eprintln!(
            "bonds broken by the blast {blast_bonds}, by crushes {crush_bonds}; crushed {crushed}"
        );
        assert!(blast_bonds > 0, "the charge breaks bonds");
        assert!(crushed > 0, "the charge crushes the infill it faces");
        assert!(
            crush_bonds > 0,
            "a crushed chunk's bonds break with cause Crush"
        );
        rbDestroyWorld(w);
    }
}

#[test]
fn crush_definitions_are_checked_through_the_abi() {
    unsafe {
        // Materials by name replace the pack's; a missing name is an error.
        let name = CString::new("nonexistent-material").unwrap();
        let bad = [RbCrushMaterial {
            name: ptr::null(),
            cap_pressure: 1e7,
            cohesion: 1e6,
            friction_slope: 1.0,
            crush_energy: 1e6,
            crush_viscosity: 1e5,
        }];
        let crush = RbCrushDef {
            enabled: true,
            materials: bad.as_ptr(),
            material_count: 1,
            ..rbDefaultDestructibleDef().crush
        };
        let (w, _, r) = building(crush);
        assert!(r.is_err(), "a crush material without a name is an error");
        rbDestroyWorld(w);
        // A count with no list is an error.
        let crush = RbCrushDef {
            enabled: true,
            materials: ptr::null(),
            material_count: 2,
            ..rbDefaultDestructibleDef().crush
        };
        let (w, _, r) = building(crush);
        assert!(r.is_err(), "a null material list with a count is an error");
        rbDestroyWorld(w);
        // A material the stress scene does not have is an error, never ignored.
        let extra = [RbCrushMaterial {
            name: name.as_ptr(),
            cap_pressure: 1e7,
            cohesion: 1e6,
            friction_slope: 1.0,
            crush_energy: 1e6,
            crush_viscosity: 1e5,
        }];
        let crush = RbCrushDef {
            enabled: true,
            mass: RbCrushMass::RB_CRUSH_REMOVE,
            materials: extra.as_ptr(),
            material_count: 1,
            ..rbDefaultDestructibleDef().crush
        };
        let (w, _, r) = building(crush);
        let e = r.expect_err("an unknown crush material");
        assert!(e.contains("nonexistent-material"), "{e}");
        rbDestroyWorld(w);
        // Off by default, and a blast def must come from its default function.
        let d = rbDefaultDestructibleDef();
        assert!(!d.crush.enabled && !d.blast_loading);
        let (w, _, r) = building(d.crush);
        r.unwrap();
        let mut blast = rbDefaultBlastDef(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        blast.internal_value = 0;
        assert_ne!(rbWorld_Detonate(w, &blast), RbResult::RB_OK);
        let mut blast = rbDefaultBlastDef(RbCoordinates::RB_COORDINATES_Y_UP_METRES);
        blast.tnt_kg = -1.0;
        assert_ne!(
            rbWorld_Detonate(w, &blast),
            RbResult::RB_OK,
            "a negative charge"
        );
        rbDestroyWorld(w);
    }
}
