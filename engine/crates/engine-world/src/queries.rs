//! Overlap queries (P5) and explosions: queued for the next step, run on the
//! GPU, overlap results delivered with its frame.

use crate::{ShapeId, World};
use engine_schema::{EXPLODE_CAP, QUERY_BASE, QUERY_CAP, QUERY_RESULT_BASE, QUERY_RESULT_CAP};
use std::collections::BTreeMap;

/// Which shapes a query sees (Box3D `b3QueryFilter`): a shape is reported
/// when its categories meet `mask` and its own mask meets `categories`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryFilter {
    pub categories: u16,
    pub mask: u16,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            categories: 0xFFFF,
            mask: 0xFFFF,
        }
    }
}

/// A queued overlap query ([`World::overlap_aabb`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueryId(pub u64);

/// A query's shapes, in shape order, each once.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapResult {
    pub query: QueryId,
    pub shapes: Vec<ShapeId>,
}

/// A radial impulse (Box3D `b3ExplosionDef`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplosionDef {
    pub position: [f32; 3],
    /// Full strength within this distance of a shape's surface (m).
    pub radius: f32,
    /// Then fading linearly to zero over this distance (m).
    pub falloff: f32,
    /// Impulse per unit of a shape's cross-section (N s / m²).
    pub impulse_per_area: f32,
    /// Shapes whose categories meet these bits are pushed.
    pub mask: u16,
}

impl Default for ExplosionDef {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            radius: 1.0,
            falloff: 0.0,
            impulse_per_area: 1.0,
            mask: 0xFFFF,
        }
    }
}

#[derive(Default)]
pub(crate) struct QueryQueue {
    pub(crate) boxes: Vec<([f32; 3], [f32; 3], u32, u32)>,
    pub(crate) ids: Vec<QueryId>,
    pub(crate) explosions: Vec<([f32; 3], f32, f32, f32, u32)>,
    next: u64,
}

impl World {
    /// The live shape a collider belongs to.
    pub(crate) fn shape_id_of_collider(&self, collider: u32) -> Option<ShapeId> {
        let &s = self.collider_shape.get(collider as usize)?;
        if s == u32::MAX {
            return None;
        }
        let slot = &self.shapes[s as usize];
        slot.alive.then(|| ShapeId::new(s, slot.generation))
    }

    /// Shapes whose bounds overlap the box `min..max`, for the next step's
    /// final poses (Box3D `b3World_OverlapAABB`: bounds, so a shape near
    /// the box may be included; sensors are not reported). The result
    /// arrives with that step's frame (`Frame::overlap_results`).
    pub fn overlap_aabb(
        &mut self,
        min: [f32; 3],
        max: [f32; 3],
        filter: QueryFilter,
    ) -> anyhow::Result<QueryId> {
        anyhow::ensure!(
            (self.queries.boxes.len() as u32) < QUERY_CAP,
            "more than {QUERY_CAP} overlap queries in one step"
        );
        anyhow::ensure!(
            (0..3).all(|k| min[k] <= max[k]),
            "overlap box min must not exceed max"
        );
        self.queries.next += 1;
        let id = QueryId(self.queries.next);
        self.queries.boxes.push((
            min,
            max,
            u32::from(filter.categories),
            u32::from(filter.mask),
        ));
        self.queries.ids.push(id);
        Ok(id)
    }

    /// A radial impulse at the start of the next step (Box3D
    /// `b3World_Explode`); bodies it reaches wake.
    pub fn explode(&mut self, def: &ExplosionDef) -> anyhow::Result<()> {
        anyhow::ensure!(
            (self.queries.explosions.len() as u32) < EXPLODE_CAP,
            "more than {EXPLODE_CAP} explosions in one step"
        );
        anyhow::ensure!(
            def.radius >= 0.0 && def.falloff >= 0.0,
            "negative explosion radius or falloff"
        );
        self.queries.explosions.push((
            def.position,
            def.radius,
            def.falloff,
            def.impulse_per_area,
            u32::from(def.mask),
        ));
        self.dirty = true;
        Ok(())
    }

    /// Group this step's overlap results (`bytes`: the query region) by
    /// query. Returns them and the results the region could not hold.
    pub(crate) fn decode_overlaps(
        &self,
        ids: &[QueryId],
        bytes: &[u8],
    ) -> (Vec<OverlapResult>, u32) {
        let words: &[u32] = bytemuck::cast_slice(bytes);
        let count = words[1];
        let held = count.min(QUERY_RESULT_CAP) as usize;
        let base = (QUERY_RESULT_BASE - QUERY_BASE) as usize;
        let mut by_query: BTreeMap<usize, Vec<ShapeId>> = BTreeMap::new();
        for k in 0..held {
            let (q, collider) = (words[base + 2 * k] as usize, words[base + 2 * k + 1]);
            if let Some(shape) = self.shape_id_of_collider(collider) {
                by_query.entry(q).or_default().push(shape);
            }
        }
        let results = ids
            .iter()
            .enumerate()
            .map(|(q, &id)| {
                let mut shapes = by_query.remove(&q).unwrap_or_default();
                shapes.sort_unstable();
                shapes.dedup();
                OverlapResult { query: id, shapes }
            })
            .collect();
        (results, count - held as u32)
    }
}
