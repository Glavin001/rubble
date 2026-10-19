use stress_learned::{runtime_snapshot::RuntimeGraph,runtime_endpoint::Prediction,joint_history,response_history};
use stress_ref::{builders,world::World,scene::SolveMode,math::Vec3};
fn setup()->World {let mut scene=builders::bond_tension();scene.sim.solve_mode=SolveMode::Explicit;scene.sim.duration=scene.sim.frame_dt*2.;World::new(&scene)}
fn unchanged(w:&World,g:&RuntimeGraph)->Prediction {
    let keys:Vec<_>=g.bond_keys().iter().enumerate().filter(|(_,&(s,b))|w.solver.bonds[s][b].alive).collect();
    Prediction{format:"learned-external-endpoint/1".into(),start_time:w.solver.time,dt:w.scene.sim.frame_dt,
        chunk_delta:vec![[0.;12];g.chunk_keys().len()],bond_ids:keys.iter().map(|(i,_)|*i).collect(),
        joint_values:keys.iter().map(|(_,&(s,b))|joint_history::pack(&w.solver.bonds[s][b].joint).map(|v|if v==f64::INFINITY {None}else {Some(v)})).collect(),
        rebar_broken:keys.iter().map(|(_,&(s,b))|w.solver.bonds[s][b].joint.rebar_broken).collect(),
        mode:keys.iter().map(|(_,&(s,b))|joint_history::mode_code(w.solver.bonds[s][b].joint.mode)).collect(),
        connected:keys.iter().map(|(_,&(s,b))|w.solver.bonds[s][b].connected()).collect(),
        response:keys.iter().map(|(_,&(s,b))|response_history::pack(&w.solver.bonds[s][b])).collect()}
}
#[test]
fn endpoint_break_really_splits_and_next_capture_uses_own_state() {
    let mut w=setup();let g=RuntimeGraph::new(&w).unwrap();let initial=g.capture(&w).unwrap();
    let mut p=unchanged(&w,&g);p.joint_values[0][0]=Some(1.);p.rebar_broken[0]=true;p.connected[0]=false;
    let e=p.decode(&g,&w).unwrap();let before=w.solver.bond_evaluations;
    let r=w.step_frame_external_endpoint(&e).unwrap();
    assert_eq!(r.new_breaks,1);assert_eq!(r.analytic_bond_evaluations,0);assert_eq!(before,w.solver.bond_evaluations);
    assert!(!w.solver.bonds[0][0].alive);assert_eq!(w.solver.clusters.len(),2);
    let next=g.capture(&w).unwrap();assert_ne!(initial.joint_flags,next.joint_flags);assert_eq!(next.joint_flags[0][2..],[0,0]);
    let e=unchanged(&w,&g).decode(&g,&w).unwrap();w.step_frame_external_endpoint(&e).unwrap();
    assert_eq!(g.capture(&w).unwrap().joint_flags[0][2..],[0,0]);
}
#[test]
fn malformed_endpoint_rejected_without_mutating_world() {
    let mut w=setup();let g=RuntimeGraph::new(&w).unwrap();let before=serde_json::to_value(g.capture(&w).unwrap()).unwrap();
    let mut p=unchanged(&w,&g);p.bond_ids.push(0);assert!(p.decode(&g,&w).is_err());
    let mut p=unchanged(&w,&g);p.connected[0]=false;assert!(p.decode(&g,&w).is_err());
    let mut p=unchanged(&w,&g);p.joint_values[0][0]=None;assert!(p.decode(&g,&w).is_err());
    let mut p=unchanged(&w,&g);p.response[0][12]=-1.;assert!(p.decode(&g,&w).is_err());
    let mut e=unchanged(&w,&g).decode(&g,&w).unwrap();e.dt*=2.;assert!(w.step_frame_external_endpoint(&e).is_err());
    assert_eq!(before,serde_json::to_value(g.capture(&w).unwrap()).unwrap());
}
#[test]
fn external_step_preserves_contact_substep_count_and_applies_hidden_state() {
    let mut w=setup();let mut oracle=w.clone();let g=RuntimeGraph::new(&w).unwrap();let mut p=unchanged(&w,&g);
    let free=g.chunk_keys().iter().position(|&(s,c)|w.solver.structures[s].chunks[c].support==stress_ref::scene::Support::None).unwrap();
    p.chunk_delta[free][0]=0.00001;let e=p.decode(&g,&w).unwrap();let before=w.solver.substeps;
    oracle.step_frame();let r=w.step_frame_external_endpoint(&e).unwrap();
    assert_eq!(r.contact_substeps,oracle.solver.substeps-before);
    let (s,c)=g.chunk_keys()[free];assert!((w.solver.chunks[s][c].u-Vec3::new(0.00001,0.,0.)).norm()<1e-12);
    assert_eq!(r.analytic_bond_evaluations,0);
}
#[test]
fn dense_observation_does_not_change_any_recorded_state() {
    let mut a=setup();let mut b=a.clone();let g=RuntimeGraph::new(&a).unwrap();let mut times=Vec::new();
    for _ in 0..3 {
        a.step_frame();b.step_frame_observed(|w|times.push(w.solver.time)).unwrap();
        assert_eq!(serde_json::to_value(g.capture(&a).unwrap()).unwrap(),serde_json::to_value(g.capture(&b).unwrap()).unwrap());
        assert_eq!(a.solver.bond_evaluations,b.solver.bond_evaluations);
    }
    assert_eq!(times.len() as u64,b.solver.substeps);assert!(times.windows(2).all(|t|t[1]>t[0]));
}
