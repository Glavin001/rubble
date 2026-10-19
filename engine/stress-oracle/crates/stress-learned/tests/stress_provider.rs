use stress_ref::{builders,material::Material,scene::{SolveMode,TimeFunction},world::World,
    stress_provider::{StressProvider,SuppliedStress,BondJob},solver::{ReferenceSolver,ChunkLoads}};
use stress_learned::runtime_snapshot::RuntimeGraph;
struct Oracle {calls:usize,rows:usize,invalid:Option<&'static str>}
impl StressProvider for Oracle {
    fn responses(&mut self,s:&ReferenceSolver,_:&ChunkLoads,jobs:&[BondJob],dt:f64,fracture:bool)->Result<Vec<SuppliedStress>,String> {
        self.calls+=1;self.rows+=jobs.len();
        let mut values:Vec<_>=jobs.iter().map(|&job|s.oracle_stress_response(job,dt,fracture)).collect();
        if let Some(first)=values.first_mut(){match self.invalid {
            Some("key")=>first.key.1=usize::MAX,
            Some("force")=>first.total_force.lin.x=f64::NAN,
            Some("work")=>first.damping_dissipated=-1.,
            Some("history")=>first.joint.state.dissipated+=1.,
            Some("connectivity")=>first.joint.disconnected=!first.joint.disconnected,
            Some("healing")=>first.joint.state.damage=0.,
            Some("fracture-disabled")=>first.joint.state.damage=0.25,
            _=>{}
        }}
        if self.invalid==Some("count"){values.clear()}
        Ok(values)
    }
}

#[test]
fn provider_cannot_heal_partial_damage_or_override_disabled_fracture() {
    for invalid in ["healing","fracture-disabled"] {
        let mut w=World::new(&scene());
        if invalid=="healing" {w.solver.bonds[0][0].joint.damage=0.25;}
        else {w.solver.config.fracture=false;}
        let before=w.solver.bonds[0][0].joint.clone();
        let loads=ChunkLoads::new(&w.solver);let mut provider=Oracle{calls:0,rows:0,invalid:Some(invalid)};
        assert!(w.solver.substep_with_provider(1e-6,&loads,&mut Some(&mut provider)).is_err(),"{invalid}");
        assert_eq!(w.solver.bonds[0][0].joint,before);
        assert_eq!(w.solver.bond_evaluations,0);
    }
}
fn scene()->stress_ref::scene::Scene {
    let m=Material::analytic_test();let force=4.*m.tensile_strength*0.01;
    let mut s=builders::bond_pull("provider-fracture",m,0.05,TimeFunction::Ramp{t0:0.,t1:0.005,value:force},0.02);
    s.sim.solve_mode=SolveMode::Explicit;s.sim.frame_dt=0.001;s
}
#[test]
fn supplied_oracle_matches_native_motion_history_energy_and_break_timing() {
    let mut a=World::new(&scene());let mut b=a.clone();let g=RuntimeGraph::new(&a).unwrap();let mut provider=Oracle{calls:0,rows:0,invalid:None};
    for _ in 0..20 {
        a.step_frame();b.step_frame_with_stress_provider(&mut provider).unwrap();
        assert_eq!(serde_json::to_value(g.capture(&a).unwrap()).unwrap(),serde_json::to_value(g.capture(&b).unwrap()).unwrap());
        assert_eq!(a.solver.events,b.solver.events);
        assert_eq!(serde_json::to_value(a.solver.energy).unwrap(),serde_json::to_value(b.solver.energy).unwrap());
    }
    assert!(!a.solver.bonds[0][0].connected());assert_eq!(b.solver.clusters.len(),2);
    assert!(provider.calls>0 && provider.rows>0);assert_eq!(provider.rows as u64,a.solver.bond_evaluations);
    assert_eq!(b.solver.bond_evaluations,0);
}
#[test]
fn invalid_provider_batch_does_not_commit_stress_or_fall_back() {
    for invalid in ["key","force","work","history","connectivity","count"] {
        let mut w=World::new(&scene());let g=RuntimeGraph::new(&w).unwrap();let before=serde_json::to_value(g.capture(&w).unwrap()).unwrap();
        let loads=ChunkLoads::new(&w.solver);let mut provider=Oracle{calls:0,rows:0,invalid:Some(invalid)};
        assert!(w.solver.substep_with_provider(1e-6,&loads,&mut Some(&mut provider)).is_err(),"{invalid}");
        assert_eq!(w.solver.bond_evaluations,0);
        assert_eq!(before,serde_json::to_value(g.capture(&w).unwrap()).unwrap());
    }
}
