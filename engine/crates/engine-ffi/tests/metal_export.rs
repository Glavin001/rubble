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

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let r = qmul(qmul(q, [v[0], v[1], v[2], 0.0]), [-q[0], -q[1], -q[2], q[3]]);
    [r[0], r[1], r[2]]
}

/// a × b as poses: b applied in a's frame.
fn compose(ap: [f32; 3], aq: [f32; 4], bp: [f32; 3], bq: [f32; 4]) -> ([f32; 3], [f32; 4]) {
    let r = qrot(aq, bp);
    ([ap[0] + r[0], ap[1] + r[1], ap[2] + r[2]], qmul(aq, bq))
}

#[test]
fn poses_land_in_the_hosts_buffers() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
    let queue = device.newCommandQueueWithMaxCommandBufferCount(4096).expect("a queue");
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
        // A body whose centre of mass is off its origin: its record is not its origin.
        bd.transform.p = RbVec3 { x: -3.0, y: 2.0, z: 1.0 };
        let mut off = 0;
        ok(rbCreateBody(w, &bd, &mut off));
        let shifted = RbTransform { p: RbVec3 { x: 0.8, y: 0.0, z: 0.3 }, q: RbQuat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } };
        ok(rbCreateBoxShape(w, off, &rbDefaultShapeDef(), RbVec3 { x: 0.4, y: 0.4, z: 0.4 }, shifted, &mut s));
        let mut off_slot = 0u32;
        ok(rbBody_GetPoseIndex(w, off, &mut off_slot));

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
            // Record × offset is the reported origin of the off-centre body.
            let mut offset = RbTransform::default();
            ok(rbBody_GetPoseOffset(w, off, &mut offset));
            assert!((offset.p.x + 0.8).abs() < 1e-4 && (offset.p.z + 0.3).abs() < 1e-4, "the offset is the centre of mass's: {:?}", offset.p);
            let mut want = RbTransform::default();
            ok(rbBody_GetTransform(w, off, &mut want));
            let r = poses(&host, cur_off as usize, off_slot);
            let origin = compose(r.p, r.q, [offset.p.x, offset.p.y, offset.p.z], [offset.q.x, offset.q.y, offset.q.z, offset.q.w]);
            for k in 0..3 {
                let w_ = [want.p.x, want.p.y, want.p.z][k];
                assert!((origin.0[k] - w_).abs() < 1e-4, "frame {frame}: record x offset {:?} vs origin {:?}", origin.0, want.p);
            }
            let wq = [want.q.x, want.q.y, want.q.z, want.q.w];
            let dot: f32 = (0..4).map(|k| origin.1[k] * wq[k]).sum();
            assert!(dot.abs() > 1.0 - 1e-5, "frame {frame}: rotation {:?} vs {:?}", origin.1, wq);
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
