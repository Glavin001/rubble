//! A world on a host's own Metal device, queue and buffers (rbCreateWorldOnMetal,
//! rbWorld_ExportPoses): the poses exported into the host's buffers, at the host's offsets,
//! are the poses the world reports, and the previous range holds the poses before the latest
//! step, also when a frame takes several steps.
#![cfg(target_vendor = "apple")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use rubble_ffi::*;
use std::ptr;

fn ok(r: RbResult) {
    assert_eq!(r, RbResult::RB_OK, "{:?}", unsafe { std::ffi::CStr::from_ptr(rbGetLastError()) });
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Pose {
    p: [f32; 3],
    word: u32,
    q: [f32; 4],
}

fn poses(buf: &Retained<ProtocolObject<dyn MTLBuffer>>, offset: usize, slot: u32) -> Pose {
    unsafe { *(buf.contents().as_ptr().cast::<u8>().add(offset + slot as usize * 32).cast::<Pose>()) }
}

#[test]
fn poses_land_in_the_hosts_buffers() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
    let queue = device.newCommandQueue().expect("a queue");
    unsafe {
        let mut def = rbDefaultWorldDef();
        def.max_bodies = 4096;
        let mut w = ptr::null_mut();
        ok(rbCreateWorldOnMetal(
            &def,
            Retained::as_ptr(&device) as *mut _,
            Retained::as_ptr(&queue) as *mut _,
            &mut w,
        ));
        // Ground and a falling box.
        let mut g = 0;
        ok(rbCreateBody(w, &rbDefaultBodyDef(), &mut g));
        let mut s = 0;
        ok(rbCreatePlaneShape(w, g, &rbDefaultShapeDef(), RbVec3 { x: 0.0, y: 1.0, z: 0.0 }, 0.0, &mut s));
        let mut bd = rbDefaultBodyDef();
        bd.type_ = RbBodyType::RB_DYNAMIC_BODY;
        bd.transform.p = RbVec3 { x: 1.0, y: 3.0, z: -2.0 };
        bd.transform.q.w = 1.0;
        let mut boxy = 0;
        ok(rbCreateBody(w, &bd, &mut boxy));
        ok(rbCreateBoxShape(w, boxy, &rbDefaultShapeDef(), RbVec3 { x: 0.5, y: 0.5, z: 0.5 }, RbTransform { q: RbQuat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }, ..Default::default() }, &mut s));
        let mut slot = 0u32;
        ok(rbBody_GetPoseIndex(w, boxy, &mut slot));

        // The host's buffers: one shared allocation, the two ranges at offsets (as a renderer that
        // sub-allocates hands them out).
        let bytes = 4096u64 * 32;
        let host = device
            .newBufferWithLength_options((bytes * 2 + 512) as usize, MTLResourceOptions::StorageModeShared)
            .expect("a host buffer");
        let (cur_off, prev_off) = (256u64, 256 + bytes);
        // The box's height after each step, starting with its spawn height.
        let mut ys = vec![3.0f32];
        let mut last_ticket = 0;
        for frame in 0..20 {
            // One or two steps per frame, then one export.
            for _ in 0..1 + frame % 2 {
                let mut t = RbTransform::default();
                ok(rbWorld_Step(w));
                let mut f = std::mem::zeroed::<RbFrame>();
                ok(rbWorld_Poll(w, true, true, &mut f));
                ok(rbBody_GetTransform(w, boxy, &mut t));
                ys.push(t.p.y);
            }
            let before_latest = ys[ys.len() - 2];
            let mut ticket = 0;
            ok(rbWorld_ExportPoses(
                w,
                Retained::as_ptr(&host) as *mut _,
                cur_off,
                Retained::as_ptr(&host) as *mut _,
                prev_off,
                bytes,
                &mut ticket,
            ));
            assert!(ticket > last_ticket, "tickets increase");
            last_ticket = ticket;
            let mut done = 0;
            while done < ticket {
                ok(rbWorld_GetCompletedPoseExport(w, &mut done));
            }
            let mut t = RbTransform::default();
            ok(rbBody_GetTransform(w, boxy, &mut t));
            let cur = poses(&host, cur_off as usize, slot);
            let prev = poses(&host, prev_off as usize, slot);
            // The export holds the reported pose (body origin = centre of mass for a centred box).
            for k in 0..3 {
                let want = [t.p.x, t.p.y, t.p.z][k];
                assert!((cur.p[k] - want).abs() < 1e-5, "frame {frame}: exported {:?} vs reported {:?}", cur.p, t.p);
            }
            if frame > 0 {
                assert!(
                    (prev.p[1] - before_latest).abs() < 1e-5,
                    "frame {frame}: previous {} vs the pose before the latest step {before_latest}",
                    prev.p[1]
                );
            }
        }
        let last_y = *ys.last().unwrap();
        assert!(last_y < 3.0, "the box fell ({last_y})");
        rbDestroyWorld(w);
    }
}
