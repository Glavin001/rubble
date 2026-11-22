//! The soft-body crates' tools and probes (`examples/`: the vehicle generator's rigger,
//! films, comparisons, lint, probes), kept out of engine-softbody and
//! engine-softbody-authoring so that testing those crates does not build them (in a
//! release test build each is a thin-LTO link: 61 s for authoring's 35 after a library
//! edit). Run one: `cargo run --release -p engine-softbody-lab --example NAME`.
