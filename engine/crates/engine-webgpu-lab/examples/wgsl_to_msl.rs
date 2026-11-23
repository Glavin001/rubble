//! A WGSL module as Metal Shading Language through naga (wgpu's own translation on
//! Metal), for tools/build-time/kernel_profile.py: Apple's offline compiler then
//! inlines and optimizes it as the driver does, without a GPU or a browser.
//!
//!   cargo run --profile fast -p engine-webgpu-lab --example wgsl_to_msl -- IN.wgsl OUT.metal
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(a.len() == 2, "usage: wgsl_to_msl IN.wgsl OUT.metal");
    let src = std::fs::read_to_string(&a[0])?;
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| anyhow::anyhow!(e.emit_to_string(&src)))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)?;
    let opts = naga::back::msl::Options {
        lang_version: (2, 4),
        ..Default::default()
    };
    let (msl, _) = naga::back::msl::write_string(&module, &info, &opts, &Default::default())?;
    std::fs::write(&a[1], msl)?;
    Ok(())
}
