use stress_learned::{runtime_snapshot::RuntimeGraph,runtime_trajectory::{Prediction,BondForecast,HistoryEncoding,FORMAT}};
use stress_ref::{builders,world::World,scene::SolveMode,solver::{ChunkLoads,SolverEvent},stress_provider::StressProvider};

fn fixture()->(World,RuntimeGraph,Prediction){
    let mut scene=builders::bond_tension();scene.sim.solve_mode=SolveMode::Explicit;scene.sim.frame_dt=0.001;
    let world=World::new(&scene);let graph=RuntimeGraph::new(&world).unwrap();let state=graph.capture(&world).unwrap();
    let bonds=state.joint_values.iter().enumerate().map(|(slot,h)|BondForecast{slot,history:vec![*h;3],
        response:vec![[0.;18];3],work_rate:vec![[0.;3];3],mode:vec![0;3],damage_complete_time:None,rebar_break_time:None,history_origin_encoded:None,ductility_set_time:[None;2],initial_work:None}).collect();
    let prediction=Prediction{format:FORMAT.into(),start_time:0.,dt:scene.sim.frame_dt,history_encoding:HistoryEncoding::Physical,bonds};(world,graph,prediction)
}

#[test]
fn complete_forecast_integrates_signed_work_and_breaks_at_a_native_phase(){
    let (mut w,g,mut p)=fixture();p.bonds[0].work_rate=vec![[2.,-1.,3.];3];p.bonds[0].damage_complete_time=Some(0.5);
    let mut provider=p.decode(&g,&w).unwrap();let dt=0.00025;
    for i in 0..4 {
        let loads=ChunkLoads::new(&w.solver);
        let r=provider.responses(&w.solver,&loads,&[(0,0,1.)],dt,true).unwrap();
        assert!((r[0].joint.dissipated-2.*dt).abs()<1e-18);
        assert!((r[0].joint.overshoot+dt).abs()<1e-18);
        assert!((r[0].damping_dissipated-3.*dt).abs()<1e-18);
        assert_eq!(r[0].joint.state.damage>=1.,i>=1);
        assert_eq!(r[0].joint.disconnected,i==1);
        w.solver.bonds[0][0].joint=r[0].joint.state.clone();w.solver.time+=dt;
    }
    assert!((w.solver.bonds[0][0].joint.dissipated-0.002).abs()<1e-18);
    assert_eq!(w.solver.bond_evaluations,0);
}

#[test]
fn linear_work_rate_integral_is_independent_of_phase_partition(){
    let (mut w,g,mut p)=fixture();p.bonds[0].work_rate=vec![[0.,-2.,0.],[2.,0.,4.],[4.,2.,8.]];
    let mut provider=p.clone().decode(&g,&w).unwrap();let loads=ChunkLoads::new(&w.solver);
    let full=provider.responses(&w.solver,&loads,&[(0,0,1.)],0.001,true).unwrap();
    assert!((full[0].joint.dissipated-0.002).abs()<1e-18);
    assert!((full[0].joint.overshoot).abs()<1e-18);
    let mut split=p.decode(&g,&w).unwrap();let mut total=0.;
    for dt in [0.00017,0.00023,0.0006]{
        total+=split.responses(&w.solver,&loads,&[(0,0,1.)],dt,true).unwrap()[0].joint.dissipated;w.solver.time+=dt;
    }
    assert!((total-full[0].joint.dissipated).abs()<1e-18);
}

#[test]
fn forecasts_drive_native_motion_without_analytical_fallback_or_endpoint_injection(){
    let (mut a,g,p)=fixture();let mut b=a.clone();let mut loaded=p.clone();
    for row in &mut loaded.bonds[0].response{row[2]=100.;row[8]=100.;}
    loaded.bonds[0].damage_complete_time=Some(0.5);
    let mut zero=p.decode(&g,&a).unwrap();let mut force=loaded.decode(&g,&b).unwrap();
    a.step_frame_with_stress_provider(&mut zero).unwrap();b.step_frame_with_stress_provider(&mut force).unwrap();
    assert_eq!(a.solver.bond_evaluations,0);assert_eq!(b.solver.bond_evaluations,0);
    assert!(force.calls>1);assert!(force.responses>0);
    assert_ne!(serde_json::to_value(g.capture(&a).unwrap()).unwrap(),serde_json::to_value(g.capture(&b).unwrap()).unwrap());
    let breaks:Vec<_>=b.solver.events.iter().filter_map(|e|if let SolverEvent::Broken{time,..}=e{Some(*time)}else{None}).collect();
    assert!(!breaks.is_empty());assert!(breaks[0]>=0.0005 && breaks[0]<0.001);
}

#[test]
fn invalid_trajectory_fails_before_world_advance_and_disabled_fracture_is_respected(){
    let (w,g,p)=fixture();
    for kind in ["missing","duplicate","clock","negative-work","null","nan"]{
        let mut bad=p.clone();match kind{
            "missing"=>bad.bonds.clear(),"duplicate"=>bad.bonds.push(bad.bonds[0].clone()),
            "clock"=>bad.start_time=1.,"negative-work"=>bad.bonds[0].work_rate[0][0]=-1.,
            "null"=>bad.bonds[0].history[0][0]=None,"nan"=>bad.bonds[0].response[0][0]=f64::NAN,_=>unreachable!(),
        }
        assert!(bad.decode(&g,&w).is_err(),"{kind}");assert_eq!(w.solver.bond_evaluations,0);
    }
    let mut proposed=p;proposed.bonds[0].damage_complete_time=Some(0.);proposed.bonds[0].rebar_break_time=Some(0.);
    let mut provider=proposed.decode(&g,&w).unwrap();let loads=ChunkLoads::new(&w.solver);
    let r=provider.responses(&w.solver,&loads,&[(0,0,1.)],0.001,false).unwrap();
    assert_eq!(r[0].joint.state.damage,w.solver.bonds[0][0].joint.damage);assert!(!r[0].joint.state.rebar_broken);
}

#[test]
fn signed_log_history_is_decoded_after_interpolation_with_continuous_initialization(){
    let (mut w,g,mut p)=fixture();p.history_encoding=HistoryEncoding::SignedLog;
    w.solver.bonds[0][0].joint.kappa=0.123456789012345;
    let mut origin=[0.;21];origin[2]=(w.solver.bonds[0][0].joint.kappa.ln_1p() as f32) as f64;
    for b in &mut p.bonds {
        b.history_origin_encoded=Some(origin);b.history=vec![origin.map(Some);3];
        b.history[0][7]=Some(0.);b.history[1][7]=Some(3_f64.ln());b.history[2][7]=Some(9_f64.ln());
        for row in &mut b.history{row[4]=Some(3_f64.ln());}
        b.ductility_set_time=[Some(0.25),None];
    }
    let mut provider=p.decode(&g,&w).unwrap();let loads=ChunkLoads::new(&w.solver);
    let before=provider.responses(&w.solver,&loads,&[(0,0,1.)],0.000125,true).unwrap();
    assert_eq!(before[0].joint.state.ductility,f64::INFINITY);
    let after=provider.responses(&w.solver,&loads,&[(0,0,1.)],0.00025,true).unwrap();
    assert!((after[0].joint.state.plastic.lin[0]-(3_f64.sqrt()-1.)*0.001).abs()<1e-16);
    assert!((after[0].joint.state.ductility-2.).abs()<1e-14);
    assert_eq!(after[0].joint.state.kappa.to_bits(),w.solver.bonds[0][0].joint.kappa.to_bits());
}

#[test]
fn initial_work_is_signed_once_and_replaces_first_rate_integral(){
    use stress_learned::runtime_trajectory::TRANSIENT_FORMAT;
    let (mut w,g,mut p)=fixture();p.format=TRANSIENT_FORMAT.into();
    for b in &mut p.bonds {b.initial_work=Some([7.,-11.,3.]);b.work_rate=vec![[2.,-4.,6.];3];}
    let mut provider=p.clone().decode(&g,&w).unwrap();let loads=ChunkLoads::new(&w.solver);
    // Empty jobs cannot consume a bond's initial pulse.
    assert!(provider.responses(&w.solver,&loads,&[],0.0001,true).unwrap().is_empty());
    let mut totals=[0.;3];
    for (i,dt) in [0.0001,0.00023,0.00067].into_iter().enumerate(){
        let r=provider.responses(&w.solver,&loads,&[(0,0,1.)],dt,true).unwrap();let q=&r[0];
        let work=[q.joint.dissipated,q.joint.overshoot,q.damping_dissipated];
        if i==0 {assert_eq!(work,[7.,-11.,3.]);}
        else {for j in 0..3 {assert!((work[j]-[2.,-4.,6.][j]*dt).abs()<1e-16);}}
        for j in 0..3 {totals[j]+=work[j];}w.solver.bonds[0][0].joint=q.joint.state.clone();w.solver.time+=dt;
    }
    for j in 0..3 {assert!((totals[j]-([7.,-11.,3.][j]+[2.,-4.,6.][j]*0.0009)).abs()<1e-12);}
    assert_eq!(w.solver.bonds[0][0].joint.dissipated,totals[0]);
    let (fresh,graph,_)=fixture();
    p.bonds[0].initial_work=None;assert!(p.clone().decode(&graph,&fresh).is_err());
    p.bonds[0].initial_work=Some([-1.,0.,0.]);assert!(p.clone().decode(&graph,&fresh).is_err());
    p.bonds[0].initial_work=Some([0.,f64::NAN,0.]);assert!(p.clone().decode(&graph,&fresh).is_err());
    p.format=FORMAT.into();p.bonds[0].initial_work=Some([0.;3]);assert!(p.decode(&graph,&fresh).is_err());
}
