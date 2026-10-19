//! Read-only physical contact memory. This is not a complete World checkpoint.
//! Cohort order is physical history: do not sort, pool or truncate the cohorts.
use super::{ContactKey, Stick, World};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "indices", rename_all = "snake_case")]
pub enum ContactIdentity {
    Ground(usize, usize),
    ImpactorGround(usize),
    Impactor(usize, usize, usize),
    Pair(usize, usize, usize, usize),
}

impl From<ContactKey> for ContactIdentity {
    fn from(key: ContactKey) -> Self {
        match key {
            ContactKey::Ground(s, c) => Self::Ground(s, c),
            ContactKey::ImpactorGround(i) => Self::ImpactorGround(i),
            ContactKey::Impactor(i, s, c) => Self::Impactor(i, s, c),
            ContactKey::Pair(sa, ca, sb, cb) => Self::Pair(sa, ca, sb, cb),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactLayerHistory {
    pub identity: ContactIdentity,
    pub normal: [f64; 3],
    pub indentation: f64,
    pub crushed_depth: f64,
    pub shear_displacement: [f64; 3],
    /// (stiffness, displacement on entry), oldest first; world vectors.
    pub shear_cohorts: Vec<(f64, [f64; 3])>,
    pub twist_displacement: f64,
    /// (rotational stiffness, angle on entry), oldest first.
    pub twist_cohorts: Vec<(f64, f64)>,
}

impl ContactLayerHistory {
    fn capture(identity: ContactKey, stick: &Stick) -> Result<Self, String> {
        let values = stick.normal.to_array().into_iter()
            .chain(stick.shear.d.to_array())
            .chain([stick.indent, stick.crushed, stick.torque.d])
            .chain(stick.shear.cohorts.iter().flat_map(|(k, d)| [*k, d.x, d.y, d.z]))
            .chain(stick.torque.cohorts.iter().flat_map(|(k, d)| [*k, *d]));
        if values.into_iter().any(|x| !x.is_finite()) { return Err("nonfinite contact history".into()); }
        Ok(Self { identity: identity.into(), normal: stick.normal.to_array(),
            indentation: stick.indent, crushed_depth: stick.crushed,
            shear_displacement: stick.shear.d.to_array(),
            shear_cohorts: stick.shear.cohorts.iter().map(|(k, d)| (*k, d.to_array())).collect(),
            twist_displacement: stick.torque.d, twist_cohorts: stick.torque.cohorts.clone() })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairOverlapHistory {
    /// (structure a, chunk a, structure b, chunk b).
    pub identity: (usize, usize, usize, usize),
    /// Native sample order. None is exactly the native NaN "not in contact" marker.
    pub samples: Vec<(Option<f64>, [f64; 3])>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactHistory {
    pub format: String,
    pub time: f64,
    /// Sorted by identity; each contact's cohort order stays unchanged.
    pub layers: Vec<ContactLayerHistory>,
    pub pair_overlaps: Vec<PairOverlapHistory>,
}

impl World {
    pub fn contact_history(&self) -> Result<ContactHistory, String> {
        if !self.solver.time.is_finite() { return Err("nonfinite contact time".into()); }
        let mut layers = self.sticks.iter().map(|(&key, stick)| ContactLayerHistory::capture(key, stick)).collect::<Result<Vec<_>, _>>()?;
        layers.sort_by(|a, b| a.identity.cmp(&b.identity));
        let mut pairs = Vec::new();
        for (&identity, samples) in &self.pair_offsets {
            let mut packed = Vec::with_capacity(samples.len());
            for &(depth, normal) in samples {
                if depth.is_infinite() || normal.to_array().iter().any(|x| !x.is_finite()) {
                    return Err("nonfinite pair overlap history".into());
                }
                packed.push((if depth.is_nan() { None } else { Some(depth) }, normal.to_array()));
            }
            pairs.push(PairOverlapHistory { identity, samples: packed });
        }
        pairs.sort_by_key(|pair| pair.identity);
        Ok(ContactHistory { format: "native-contact-history/1".into(), time: self.solver.time,
            layers, pair_overlaps: pairs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{builders, math::Vec3, world::Layer};

    #[test]
    fn aggregate_force_energy_and_stiffness_do_not_identify_contact_memory() {
        let a = Layer { d: 0., cohorts: vec![(2., -0.1), (2., 0.1)] };
        let b = Layer { d: 0., cohorts: vec![(2., 0.1), (2., -0.1)] };
        assert_eq!(a.load_energy(), b.load_energy());
        assert_eq!(a.stiffness_energy(), b.stiffness_energy());
        // Same present force/energy/stiffness; shrinking area removes the newest
        // cohort and produces opposite forces. A pooled state loses this history.
        let (a, _) = a.resize_from(4., 2.);
        let (b, _) = b.resize_from(4., 2.);
        assert_eq!(a.load_at(0.), 0.2);
        assert_eq!(b.load_at(0.), -0.2);
    }

    #[test]
    fn lossless_serialization_preserves_cohort_order_and_future_layer_response() {
        let mut world = World::new(&builders::bond_tension());
        let stick = Stick { shear: Layer { d: Vec3::new(0.02345678901234567, 0., 0.),
            cohorts: vec![(2., Vec3::new(-0.1, 0., 0.)), (3., Vec3::new(0.1, 0., 0.))] },
            torque: Layer { d: 0.12345678901234567, cohorts: vec![(1., -0.3), (2., 0.2)] },
            normal: Vec3::new(0., 1., 0.), indent: 0.01234567890123456, crushed: 0.001 };
        world.sticks.insert(ContactKey::Ground(0, 1), stick.clone());
        world.sticks.insert(ContactKey::Ground(0, 0), Stick::default());
        world.pair_offsets.insert((0, 0, 0, 1), vec![(f64::NAN, Vec3::ZERO), (0.12345678901234567, stick.normal)]);
        let original = world.contact_history().unwrap();
        let restored: ContactHistory = serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
        assert_eq!(restored, original);
        assert_eq!(world.contact_history().unwrap(), original);
        assert!(restored.pair_overlaps[0].samples[0].0.is_none());
        let row = &restored.layers[1];
        let decoded = Layer { d: Vec3::from_array(row.shear_displacement),
            cohorts: row.shear_cohorts.iter().map(|(k, d)| (*k, Vec3::from_array(*d))).collect() };
        let (next, _) = decoded.resize_from(5., 2.5);
        let (expected, _) = stick.shear.resize_from(5., 2.5);
        assert_eq!(next.strain(Vec3::new(0.01, 0., 0.)).load_energy(), expected.strain(Vec3::new(0.01, 0., 0.)).load_energy());
        world.sticks.get_mut(&ContactKey::Ground(0, 0)).unwrap().indent = f64::INFINITY;
        assert!(world.contact_history().is_err());
    }
}
