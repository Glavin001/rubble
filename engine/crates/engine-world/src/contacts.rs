//! Contact begin, end and hit events, and sensor overlaps (P4): the GPU lists
//! the touching pairs that involve an event-enabled shape or a sensor each
//! step (`contact_events` in world_step.slang, which also keeps sensor pairs
//! out of the solve); the host diffs that set against the previous one.

use crate::events::{ContactBeginEvent, ContactEndEvent, ContactEvents, SensorEvent, SensorEvents};
use crate::{ShapeDef, ShapeId, World};
use engine_schema::{
    COLLIDER_CONTACT_EVENTS, COLLIDER_HIT_EVENTS, COLLIDER_SENSOR, CONTACT_EVENT_CAP,
    CONTACT_EVENT_WORDS,
};
use std::collections::BTreeMap;

/// A touching pair, keyed by its two colliders (lower index first).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Touch {
    shape_a: ShapeId,
    shape_b: ShapeId,
    bodies: [u32; 2],
}

/// A sensor overlap, keyed by (sensor collider, visitor collider).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Overlap {
    event: SensorEvent,
    bodies: [u32; 2],
}

#[derive(Default)]
pub(crate) struct ContactTracker {
    touching: BTreeMap<(u32, u32), Touch>,
    overlapping: BTreeMap<(u32, u32), Overlap>,
    /// Ends of pairs whose shape was destroyed, for the next frame.
    pending_end: Vec<ContactEndEvent>,
    pending_sensor_end: Vec<SensorEvent>,
    /// Live shapes with contact or hit events: the pass runs while any do.
    pub(crate) enabled_shapes: u32,
}

/// The collider flag bits of a shape's events (`Collider.material` 12..15).
pub(crate) fn event_bits(def: &ShapeDef) -> u32 {
    let mut bits = 0;
    if def.enable_contact_events || def.enable_hit_events {
        bits |= COLLIDER_CONTACT_EVENTS;
    }
    if def.enable_hit_events {
        bits |= COLLIDER_HIT_EVENTS;
    }
    if def.is_sensor {
        bits |= COLLIDER_SENSOR;
    }
    bits
}

/// The pass must run while this shape exists (sensors need it to stay out of
/// the solve even without events).
fn wants_events(def: &ShapeDef) -> bool {
    def.enable_contact_events || def.enable_hit_events || def.is_sensor
}

impl World {
    pub(crate) fn contacts_shape_added(&mut self, def: &ShapeDef) {
        if wants_events(def) {
            self.contacts.enabled_shapes += 1;
            self.sim.set_contact_events(true);
        }
    }

    /// A shape is going away: end its pairs (next frame) and stop tracking it.
    pub(crate) fn contacts_shape_removed(&mut self, index: u32) {
        let slot = &self.shapes[index as usize];
        if !wants_events(&slot.def) {
            return;
        }
        let collider = slot.collider;
        let t = &mut self.contacts;
        let gone: Vec<(u32, u32)> = t
            .touching
            .keys()
            .filter(|(a, b)| *a == collider || *b == collider)
            .copied()
            .collect();
        for k in gone {
            if let Some(touch) = t.touching.remove(&k) {
                t.pending_end.push(ContactEndEvent {
                    shape_a: touch.shape_a,
                    shape_b: touch.shape_b,
                });
            }
        }
        let gone: Vec<(u32, u32)> = t
            .overlapping
            .keys()
            .filter(|(s, v)| *s == collider || *v == collider)
            .copied()
            .collect();
        for k in gone {
            if let Some(o) = t.overlapping.remove(&k) {
                t.pending_sensor_end.push(o.event);
            }
        }
        t.enabled_shapes -= 1;
        if t.enabled_shapes == 0 {
            self.sim.set_contact_events(false);
        }
    }

    /// Ends queued by destroyed shapes.
    pub(crate) fn take_pending_contact_ends(&mut self) -> (Vec<ContactEndEvent>, Vec<SensorEvent>) {
        (
            std::mem::take(&mut self.contacts.pending_end),
            std::mem::take(&mut self.contacts.pending_sensor_end),
        )
    }

    fn asleep(&self, bodies: [u32; 2]) -> bool {
        bodies
            .iter()
            .all(|&b| self.bodies.get(b as usize).is_none_or(|s| s.last_asleep))
    }

    fn shape_of_collider(&self, collider: u32) -> Option<(ShapeId, u32, bool, &ShapeDef)> {
        let &s = self.collider_shape.get(collider as usize)?;
        if s == u32::MAX {
            return None;
        }
        let slot = &self.shapes[s as usize];
        slot.alive.then(|| {
            (
                ShapeId::new(s, slot.generation),
                slot.body,
                slot.def.enable_hit_events,
                &slot.def,
            )
        })
    }

    /// Diff this step's touching pairs (`bytes`: the region's header and
    /// records) against the last set. Returns the events and the records the
    /// region could not hold.
    pub(crate) fn decode_contacts(&mut self, bytes: &[u8]) -> (ContactEvents, SensorEvents, u32) {
        let words: &[u32] = bytemuck::cast_slice(bytes);
        let count = words[0];
        let held = count.min(CONTACT_EVENT_CAP) as usize;
        let lost = count - held as u32;
        let stride = CONTACT_EVENT_WORDS as usize;
        let mut now: BTreeMap<(u32, u32), (Touch, ContactBeginEvent, bool)> = BTreeMap::new();
        let mut overlaps: BTreeMap<(u32, u32), Overlap> = BTreeMap::new();
        for k in 0..held {
            let r = &words[4 + k * stride..4 + (k + 1) * stride];
            let (ca, cb) = (r[0], r[1]);
            let (Some(a), Some(b)) = (self.shape_of_collider(ca), self.shape_of_collider(cb))
            else {
                continue;
            };
            if a.3.is_sensor || b.3.is_sensor {
                // Sensors do not detect sensors; only sensors with events report.
                let (sensor, visitor, key) = if a.3.is_sensor {
                    (a, b, (ca, cb))
                } else {
                    (b, a, (cb, ca))
                };
                if !visitor.3.is_sensor && sensor.3.enable_sensor_events {
                    overlaps.insert(
                        key,
                        Overlap {
                            event: SensorEvent {
                                sensor_shape: sensor.0,
                                visitor_shape: visitor.0,
                            },
                            bodies: [sensor.1, visitor.1],
                        },
                    );
                }
                continue;
            }
            let f = |i: usize| f32::from_bits(r[i]);
            let mut begin = ContactBeginEvent {
                shape_a: a.0,
                shape_b: b.0,
                point: [f(4), f(5), f(6)],
                normal: [f(8), f(9), f(10)],
                approach_speed: f(2),
            };
            let mut key = (ca, cb);
            let mut bodies = [a.1, b.1];
            if ca > cb {
                // One key per pair: lower collider first, normal still A to B.
                key = (cb, ca);
                std::mem::swap(&mut begin.shape_a, &mut begin.shape_b);
                begin.normal = begin.normal.map(|x| -x);
                bodies.swap(0, 1);
            }
            let touch = Touch {
                shape_a: begin.shape_a,
                shape_b: begin.shape_b,
                bodies,
            };
            now.insert(key, (touch, begin, a.2 || b.2));
        }
        let mut sensors = SensorEvents::default();
        let previous = std::mem::take(&mut self.contacts.overlapping);
        for (key, o) in &previous {
            if overlaps.contains_key(key) {
                continue;
            }
            // A step that lost records cannot tell a missing pair from one it
            // could not hold: it ends nothing, and the pair stays.
            if lost > 0 || self.asleep(o.bodies) {
                self.contacts.overlapping.insert(*key, *o);
            } else {
                sensors.end.push(o.event);
            }
        }
        for (key, o) in overlaps {
            if !previous.contains_key(&key) {
                sensors.begin.push(o.event);
            }
            self.contacts.overlapping.insert(key, o);
        }
        let mut events = ContactEvents::default();
        let threshold = self.hit_event_threshold;
        let previous = std::mem::take(&mut self.contacts.touching);
        for (key, touch) in &previous {
            if now.contains_key(key) {
                continue;
            }
            // Asleep pairs keep touching (their manifolds may not be rebuilt),
            // and a step that lost records ends nothing (the pair may be one
            // the region could not hold).
            if lost > 0 || self.asleep(touch.bodies) {
                self.contacts.touching.insert(*key, *touch);
            } else {
                events.end.push(ContactEndEvent {
                    shape_a: touch.shape_a,
                    shape_b: touch.shape_b,
                });
            }
        }
        for (key, (touch, begin, hits)) in now {
            if !previous.contains_key(&key) {
                events.begin.push(begin);
                if hits && begin.approach_speed >= threshold {
                    events.hits.push(begin);
                }
            }
            self.contacts.touching.insert(key, touch);
        }
        (events, sensors, lost)
    }
}
