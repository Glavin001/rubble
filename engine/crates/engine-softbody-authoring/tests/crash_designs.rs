//! The crash suite's design list (`crash_reference::family::DESIGNS`, what a scenario's
//! "car" parameter offers) is exactly the generator's registered designs plus the FMVSS 214
//! barrier, in the registry's order: a design added to the generator appears in the lab's and
//! the studio's pickers or this fails.

#[test]
fn the_suites_designs_are_the_generators_registry() {
    let mut want: Vec<&str> = engine_vehiclegen::designs::names();
    want.push("mdb214");
    assert_eq!(engine_softbody::crash_reference::family::DESIGNS.to_vec(), want);
}
