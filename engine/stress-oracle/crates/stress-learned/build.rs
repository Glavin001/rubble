//! Embeds the source commit (`STRESS_LEARNED_GIT`: `git describe --always --dirty`) in the
//! binaries, for the recorder's provenance.
fn main() {
    let rev = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty", "--abbrev=12"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=STRESS_LEARNED_GIT={rev}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../stress-ref/src");
}
