//! Contact and hit events (P4). Each step the GPU's `contact_events` pass
//! lists the touching manifolds of event-flagged shapes (deepest contact
//! `c0N <= 0`); the world resolves them to shape pairs and diffs the sorted
//! set against the previous step's. The diff never depends on how contacts
//! are matched across steps, so it is the same for every contact storage mode.

use crate::events::{ContactBeginEvent, ContactEndEvent, ContactEvents, ContactHitEvent};
use crate::ShapeId;

/// One step's touch of a shape pair (from one manifold record).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Touch {
    pub a: ShapeId,
    pub b: ShapeId,
    /// Body slots of `a` and `b`.
    pub bodies: [u32; 2],
    pub point: [f32; 3],
    /// From `a` to `b`.
    pub normal: [f32; 3],
    pub approach_speed: f32,
    pub c0n: f32,
}

impl Touch {
    /// The pair in canonical order (lower handle first; the normal follows).
    pub fn ordered(self) -> Self {
        if self.a.to_bits() <= self.b.to_bits() {
            return self;
        }
        let [x, y, z] = self.normal;
        Self {
            a: self.b,
            b: self.a,
            bodies: [self.bodies[1], self.bodies[0]],
            normal: [-x, -y, -z],
            ..self
        }
    }

    fn key(&self) -> (u64, u64) {
        (self.a.to_bits(), self.b.to_bits())
    }
}

/// What the diff needs to know about a live shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ShapeFlags {
    pub contact_events: bool,
    pub hit_events: bool,
    pub user_material_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    a: ShapeId,
    b: ShapeId,
    bodies: [u32; 2],
}

impl Pair {
    fn key(&self) -> (u64, u64) {
        (self.a.to_bits(), self.b.to_bits())
    }
}

/// The touching pairs as of the last diffed step, sorted.
#[derive(Clone, Debug, Default)]
pub(crate) struct ContactTracker {
    pairs: Vec<Pair>,
    /// Ends of pairs whose shape was destroyed, for the next delivered frame.
    pending_ends: Vec<ContactEndEvent>,
}

impl ContactTracker {
    /// A shape is being destroyed: its touching pairs end now (reported with
    /// the next frame). `flags` is read before the shape is released.
    pub fn shape_destroyed(&mut self, id: ShapeId, flags: impl Fn(ShapeId) -> Option<ShapeFlags>) {
        let mut ends = Vec::new();
        self.pairs.retain(|p| {
            if p.a != id && p.b != id {
                return true;
            }
            let contact = |s| flags(s).is_some_and(|f| f.contact_events);
            if contact(p.a) || contact(p.b) {
                ends.push(ContactEndEvent {
                    shape_a: p.a,
                    shape_b: p.b,
                });
            }
            false
        });
        self.pending_ends.extend(ends);
    }

    /// Ends queued by destroyed shapes (for frames that carry no diff).
    pub fn take_pending(&mut self) -> Vec<ContactEndEvent> {
        std::mem::take(&mut self.pending_ends)
    }

    /// Diff one step's touches against the last set.
    ///
    /// - `incomplete`: the step lost records to the capacity, so a pair it did
    ///   not report may still touch: no pair ends this step.
    /// - `resting(body)`: the body was not awake when the step began (static,
    ///   or asleep). A pair missing while both its bodies rest still touches.
    /// - `flags(shape)`: a live shape's event flags (`None`: not a user shape).
    pub fn diff(
        &mut self,
        mut touches: Vec<Touch>,
        incomplete: bool,
        resting: impl Fn(u32) -> bool,
        flags: impl Fn(ShapeId) -> Option<ShapeFlags>,
        hit_threshold: f32,
    ) -> ContactEvents {
        let mut out = ContactEvents {
            ends: self.take_pending(),
            ..ContactEvents::default()
        };
        // Deepest record first within a pair; the rest of the record breaks
        // ties, so the GPU's append order never shows.
        touches.sort_by(|x, y| {
            x.key()
                .cmp(&y.key())
                .then(x.c0n.total_cmp(&y.c0n))
                .then(x.approach_speed.total_cmp(&y.approach_speed))
                .then_with(|| bits(x.point).cmp(&bits(y.point)))
                .then_with(|| bits(x.normal).cmp(&bits(y.normal)))
        });
        touches.dedup_by_key(|t| t.key());
        let either = |p: (ShapeId, ShapeId), f: fn(&ShapeFlags) -> bool| {
            flags(p.0).is_some_and(|s| f(&s)) || flags(p.1).is_some_and(|s| f(&s))
        };
        let mut next = Vec::with_capacity(touches.len().max(self.pairs.len()));
        let (mut i, mut j) = (0, 0);
        while i < self.pairs.len() || j < touches.len() {
            let old = self.pairs.get(i).copied();
            let new = touches.get(j).copied();
            match (old, new) {
                (Some(p), Some(t)) if p.key() == t.key() => {
                    next.push(p);
                    i += 1;
                    j += 1;
                }
                (Some(p), t) if t.is_none_or(|t| p.key() < t.key()) => {
                    i += 1;
                    let keep = incomplete || (resting(p.bodies[0]) && resting(p.bodies[1]));
                    if keep {
                        next.push(p);
                    } else if either((p.a, p.b), |f| f.contact_events) {
                        out.ends.push(ContactEndEvent {
                            shape_a: p.a,
                            shape_b: p.b,
                        });
                    }
                }
                (_, Some(t)) => {
                    j += 1;
                    next.push(Pair {
                        a: t.a,
                        b: t.b,
                        bodies: t.bodies,
                    });
                    if either((t.a, t.b), |f| f.contact_events) {
                        out.begins.push(ContactBeginEvent {
                            shape_a: t.a,
                            shape_b: t.b,
                            point: t.point,
                            normal: t.normal,
                            approach_speed: t.approach_speed,
                        });
                    }
                    if either((t.a, t.b), |f| f.hit_events) && t.approach_speed >= hit_threshold {
                        let m = |s| flags(s).map_or(0, |f| f.user_material_id);
                        out.hits.push(ContactHitEvent {
                            shape_a: t.a,
                            shape_b: t.b,
                            point: t.point,
                            normal: t.normal,
                            approach_speed: t.approach_speed,
                            user_material_ids: [m(t.a), m(t.b)],
                        });
                    }
                }
                (None, None) => unreachable!(),
                (Some(_), None) => unreachable!(),
            }
        }
        self.pairs = next;
        out
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pairs.len()
    }
}

fn bits(v: [f32; 3]) -> [u32; 3] {
    v.map(f32::to_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(i: u32) -> ShapeId {
        ShapeId::new(i, 1)
    }

    fn touch(a: u32, b: u32, speed: f32, c0n: f32) -> Touch {
        Touch {
            a: shape(a),
            b: shape(b),
            bodies: [a, b],
            point: [0.0, c0n, 0.0],
            normal: [0.0, 1.0, 0.0],
            approach_speed: speed,
            c0n,
        }
        .ordered()
    }

    fn all(_: ShapeId) -> Option<ShapeFlags> {
        Some(ShapeFlags {
            contact_events: true,
            hit_events: true,
            user_material_id: 7,
        })
    }

    #[test]
    fn a_new_pair_begins_once_and_ends_when_it_is_gone() {
        let mut t = ContactTracker::default();
        let e = t.diff(vec![touch(1, 0, 3.0, -0.01)], false, |_| false, all, 1.0);
        assert_eq!(e.begins.len(), 1);
        assert_eq!(e.hits.len(), 1);
        // Canonical order: the lower handle first, the normal flipped with it.
        assert_eq!(e.begins[0].shape_a, shape(0));
        assert_eq!(e.begins[0].normal, [0.0, -1.0, 0.0]);
        assert_eq!(e.hits[0].user_material_ids, [7, 7]);
        for _ in 0..3 {
            let e = t.diff(vec![touch(0, 1, 0.0, -0.01)], false, |_| false, all, 1.0);
            assert!(e.is_empty(), "a resting pair does not begin again: {e:?}");
        }
        let e = t.diff(vec![], false, |_| false, all, 1.0);
        assert_eq!(
            e.ends,
            vec![ContactEndEvent {
                shape_a: shape(0),
                shape_b: shape(1)
            }]
        );
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn slow_begins_raise_no_hit() {
        let mut t = ContactTracker::default();
        let e = t.diff(vec![touch(0, 1, 0.5, -0.01)], false, |_| false, all, 1.0);
        assert_eq!((e.begins.len(), e.hits.len()), (1, 0));
    }

    #[test]
    fn sleep_and_lost_records_end_nothing() {
        let mut t = ContactTracker::default();
        t.diff(vec![touch(0, 1, 2.0, -0.01)], false, |_| false, all, 1.0);
        let e = t.diff(vec![], false, |_| true, all, 1.0);
        assert!(e.is_empty(), "both bodies asleep: still touching");
        let e = t.diff(vec![], true, |_| false, all, 1.0);
        assert!(e.is_empty(), "an incomplete step ends nothing");
        let e = t.diff(vec![], false, |b| b == 0, all, 1.0);
        assert_eq!(e.ends.len(), 1, "one body awake: the pair ended");
    }

    #[test]
    fn duplicate_manifolds_report_the_deepest() {
        let mut t = ContactTracker::default();
        let e = t.diff(
            vec![
                touch(0, 1, 1.0, -0.01),
                touch(1, 0, 2.0, -0.03),
                touch(0, 1, 3.0, -0.02),
            ],
            false,
            |_| false,
            all,
            0.0,
        );
        assert_eq!(e.begins.len(), 1);
        assert_eq!(e.begins[0].approach_speed, 2.0);
    }

    #[test]
    fn destroying_a_shape_ends_its_pairs() {
        let mut t = ContactTracker::default();
        t.diff(
            vec![touch(0, 1, 2.0, -0.01), touch(0, 2, 2.0, -0.01)],
            false,
            |_| false,
            all,
            1.0,
        );
        t.shape_destroyed(shape(1), all);
        let e = t.diff(vec![touch(0, 2, 0.0, -0.01)], false, |_| false, all, 1.0);
        assert_eq!(
            e.ends,
            vec![ContactEndEvent {
                shape_a: shape(0),
                shape_b: shape(1)
            }]
        );
        assert!(e.begins.is_empty());
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn flags_choose_what_is_reported() {
        let hits_only = |_: ShapeId| {
            Some(ShapeFlags {
                contact_events: false,
                hit_events: true,
                user_material_id: 0,
            })
        };
        let mut t = ContactTracker::default();
        let e = t.diff(
            vec![touch(0, 1, 5.0, -0.01)],
            false,
            |_| false,
            hits_only,
            1.0,
        );
        assert_eq!((e.begins.len(), e.hits.len()), (0, 1));
        let e = t.diff(vec![], false, |_| false, hits_only, 1.0);
        assert!(e.ends.is_empty());
        assert_eq!(t.len(), 0);
    }
}
