//! Run exact PyTorch predictions alongside recorded oracle labels in the comparison UI.
//! Keep the checkpoint backend shared with training until Slang parity is established.
use std::{env, path::PathBuf, process::Command};

pub fn run(args: &[String]) -> Result<(), String> {
    let package = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../learned-impact/python");
    let python = env::var_os("STRESS_LEARNED_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| package.join(".venv/bin/python"));
    let mut paths = vec![package];
    if let Some(old) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&old));
    }
    let status = Command::new(&python)
        .args(["-m", "learned_impact.visualize"])
        .args(args)
        .env(
            "PYTHONPATH",
            env::join_paths(paths).map_err(|e| e.to_string())?,
        )
        .status()
        .map_err(|e| {
            format!(
                "{}: {e}. Set up learned-impact/python/.venv or set STRESS_LEARNED_PYTHON.",
                python.display()
            )
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("learned comparison exited {status}"))
    }
}
