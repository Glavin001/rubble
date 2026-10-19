//! Full graph-state bridge checks. Oracle calls below are explicit test controls,
//! never evidence of learned inference.
use stress_ref::{
    builders::wall_impact,
    math::Vec3,
    scene::Support,
    solver::{ChunkLoads, ReferenceSolver},
    stress_provider::{BondJob, ChunkTransition, StressProvider, SuppliedStress},
    world::World,
};

#[derive(Default)]
struct Control {
    supplied: Option<Vec<ChunkTransition>>,
    observed: Vec<ChunkTransition>,
}
impl StressProvider for Control {
    fn responses(
        &mut self,
        s: &ReferenceSolver,
        _: &ChunkLoads,
        jobs: &[BondJob],
        dt: f64,
        fracture: bool,
    ) -> Result<Vec<SuppliedStress>, String> {
        Ok(jobs
            .iter()
            .map(|&job| s.oracle_stress_response(job, dt, fracture))
            .collect())
    }
    fn chunk_transitions(
        &mut self,
        _: &ReferenceSolver,
        _: &ChunkLoads,
        _: &[(usize, usize)],
        _: f64,
    ) -> Result<Option<Vec<ChunkTransition>>, String> {
        Ok(self.supplied.take())
    }
    fn records_chunk_transitions(&self) -> bool {
        true
    }
    fn observe_chunk_transitions(
        &mut self,
        states: &[ChunkTransition],
        _: f64,
    ) -> Result<(), String> {
        self.observed = states.to_vec();
        Ok(())
    }
}

fn fixture() -> (ReferenceSolver, ChunkLoads) {
    let mut solver = World::new(&wall_impact(12., "graph-transition")).solver;
    solver.config.fracture = false;
    let mut loads = ChunkLoads::new(&solver);
    let free = solver.structures[0]
        .chunks
        .iter()
        .position(|c| c.support == Support::None)
        .unwrap();
    loads.force[0][free] = Vec3::new(1000., 2000., 300.);
    loads.torque[0][free] = Vec3::new(1., -2., 3.);
    (solver, loads)
}

fn same_physical_state(a: &ReferenceSolver, b: &ReferenceSolver) {
    assert_eq!(format!("{:?}", a.chunks), format!("{:?}", b.chunks));
    assert_eq!(format!("{:?}", a.clusters), format!("{:?}", b.clusters));
    assert_eq!(format!("{:?}", a.energy), format!("{:?}", b.energy));
    assert_eq!(format!("{:?}", a.events), format!("{:?}", b.events));
    assert_eq!(a.time, b.time);
    assert_eq!(a.substeps, b.substeps);
}

#[test]
fn observed_native_targets_can_replace_internal_integration_exactly() {
    let (initial, loads) = fixture();
    let mut reference = initial.clone();
    reference.substep(1e-6, &loads);
    let mut recorded = initial.clone();
    let mut observer = Control::default();
    recorded
        .substep_with_provider(1e-6, &loads, &mut Some(&mut observer))
        .unwrap();
    same_physical_state(&reference, &recorded);
    assert!(!observer.observed.is_empty());
    assert_eq!(
        recorded.numerical_chunk_updates,
        observer.observed.len() as u64
    );
    assert_eq!(recorded.supplied_chunk_updates, 0);
    let n = observer.observed.len();
    let mut replay = Control {
        supplied: Some(observer.observed),
        observed: Vec::new(),
    };
    let mut supplied = initial;
    supplied
        .substep_with_provider(1e-6, &loads, &mut Some(&mut replay))
        .unwrap();
    same_physical_state(&reference, &supplied);
    assert_eq!(supplied.numerical_chunk_updates, 0);
    assert_eq!(supplied.supplied_chunk_updates, n as u64);
}

#[test]
fn supplied_state_is_applied_instead_of_a_hidden_numerical_update() {
    let (mut solver, loads) = fixture();
    let states: Vec<_> = solver
        .clusters
        .iter()
        .flat_map(|cl| {
            cl.chunks
                .iter()
                .map(|&c| ChunkTransition::capture(&solver, (cl.structure, c)))
        })
        .collect();
    let expected = states.clone();
    let mut numerical = solver.clone();
    let mut observer = Control::default();
    numerical
        .substep_with_provider(1e-6, &loads, &mut Some(&mut observer))
        .unwrap();
    assert!(expected
        .iter()
        .zip(&observer.observed)
        .any(|(a, b)| a.u != b.u || a.v != b.v || a.th != b.th || a.w != b.w));
    let mut provider = Control {
        supplied: Some(states),
        observed: Vec::new(),
    };
    solver
        .substep_with_provider(1e-6, &loads, &mut Some(&mut provider))
        .unwrap();
    assert!(solver.supplied_chunk_updates > 0);
    assert_eq!(solver.numerical_chunk_updates, 0);
    for (st, old) in provider.observed.iter().zip(expected) {
        assert_eq!(st.u, old.u);
        assert_eq!(st.v, old.v);
        assert_eq!(st.th, old.th);
        assert_eq!(st.w, old.w);
    }
}

#[test]
fn invalid_graph_transition_is_rejected_before_bond_or_motion_commit() {
    for invalid in ["count", "order", "nonfinite", "support"] {
        let (mut solver, loads) = fixture();
        let before = solver.clone();
        let mut states: Vec<_> = solver
            .clusters
            .iter()
            .flat_map(|cl| {
                cl.chunks
                    .iter()
                    .map(|&c| ChunkTransition::capture(&solver, (cl.structure, c)))
            })
            .collect();
        match invalid {
            "count" => {
                states.pop();
            }
            "order" => states.swap(0, 1),
            "nonfinite" => states[0].v.x = f64::NAN,
            "support" => {
                let st = states
                    .iter_mut()
                    .find(|s| solver.structures[s.key.0].chunks[s.key.1].support == Support::Fixed)
                    .unwrap();
                st.u.x = 1.;
            }
            _ => unreachable!(),
        }
        let mut provider = Control {
            supplied: Some(states),
            observed: Vec::new(),
        };
        assert!(solver
            .substep_with_provider(1e-6, &loads, &mut Some(&mut provider))
            .is_err());
        same_physical_state(&before, &solver);
        for (a, b) in before
            .bonds
            .iter()
            .flatten()
            .zip(solver.bonds.iter().flatten())
        {
            assert_eq!(format!("{:?}", a.joint), format!("{:?}", b.joint));
        }
    }
}
