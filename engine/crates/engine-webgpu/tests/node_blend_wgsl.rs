//! The vehicle lab's drawn nodes between World steps (`node-blend.mjs`): its kernel
//! parses and validates (naga, every check), and it reads the solver's node layout
//! (3 vec4 a node: position and mass, velocity, force) and misc layout (a vec4 a vehicle,
//! its origin's bits).

#[test]
fn node_blend_kernel_validates() {
    let src = include_str!("../node-blend.mjs");
    let start = src.find("const WGSL = /* wgsl */ `").expect("node-blend.mjs: no WGSL") + "const WGSL = /* wgsl */ `".len();
    let wgsl = &src[start..start + src[start..].find('`').expect("WGSL not closed")];
    let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| panic!("node-blend.mjs: {}", e.emit_to_string(wgsl)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("node-blend.mjs: {e:?}"));
    for (what, needle) in [
        ("the position", "cur[i * 3u]"),
        ("the previous position", "prev[i * 3u].xyz"),
        ("the velocity", "cur[i * 3u + 1u]"),
        ("the vehicle's origin", "misc[v].xyz"),
    ] {
        assert!(wgsl.contains(needle), "node-blend.mjs reads {what} as `{needle}` (the solver's layout)");
    }
    // The solver's layout as the car renderer reads it.
    let render = include_str!("../showcase-car-render.mjs");
    assert!(render.contains("let k = (params.nodeOffset + i) * 3u; return nodes[k].xyz"), "showcase-car-render.mjs: the node layout changed");
}
