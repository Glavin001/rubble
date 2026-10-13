fn main() -> anyhow::Result<()> {
    env_logger::init();
    let out = engine_metal::run_abi_probe()?;
    println!(
        "abi_probe OK position={:?} counter={} chip={}",
        out.position(),
        out.counter,
        engine_metal::MetalDevice::new()?.chip
    );
    Ok(())
}
