// What an explicit module's WGSL gets after slangc, once, wherever it is compiled
// (`include!`d by build.rs for the shipped modules and by lib.rs for the test-only ones,
// `test_kernels_dir`): the barriers order storage only (Slang lowers
// DeviceMemoryBarrierWithGroupSync with a texture barrier naga rejects, and an empty
// statement Safari's parser rejects), and a module without the Slang loop caps (the
// frozen baseline and reference) gets every loop capped here. Needs wgsl_loop_cap.rs.

/// `text`, the WGSL of the explicit module `name` (its file name), post-processed.
#[allow(dead_code)]
fn explicit_post_text(name: &str, text: &str) -> Result<String, String> {
    let text = text
        .replace(
            "storageBarrier(); textureBarrier(); workgroupBarrier();;",
            "storageBarrier(); workgroupBarrier();",
        )
        .replace(
            "storageBarrier(); textureBarrier(); workgroupBarrier();",
            "storageBarrier(); workgroupBarrier();",
        );
    let uncapped = name.starts_with("explicit_ref_")
        && std::env::var("RUBBLE_EB_REF_UNCAPPED").is_ok_and(|v| v == "1");
    // (The Slang caps: their flag, ebLoopHit, set by the loops' ends or ebLoopCapHit.)
    if uncapped || text.contains("ebLoopCapHit") || text.contains("ebLoopHit") {
        Ok(text)
    } else {
        cap_explicit_loops(&text).map_err(|e| format!("{name}: {e}"))
    }
}
