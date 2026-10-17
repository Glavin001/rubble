//! What machine a result came from.

use serde_json::{json, Value};
use std::process::Command;

fn sh(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

pub fn cpu_model() -> String {
    if cfg!(target_os = "macos") {
        return sh("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default();
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()))
        .unwrap_or_default()
}

pub fn logical_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

pub fn describe() -> Value {
    let mut doc = json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu": cpu_model(),
        "logicalCores": logical_cores(),
        "hostname": sh("hostname", &[]).unwrap_or_default(),
    });
    if cfg!(target_os = "macos") {
        doc["osVersion"] = json!(sh("sw_vers", &["-productVersion"]).unwrap_or_default());
        doc["performanceCores"] = json!(sh("sysctl", &["-n", "hw.perflevel0.physicalcpu"]).and_then(|s| s.parse::<u32>().ok()));
        doc["memoryBytes"] = json!(sh("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse::<u64>().ok()));
    } else {
        doc["osVersion"] = json!(sh("uname", &["-r"]).unwrap_or_default());
        doc["gpu"] = json!(sh("nvidia-smi", &["--query-gpu=name,driver_version", "--format=csv,noheader"]));
    }
    doc
}

/// ISO-8601 UTC timestamp without pulling in a date crate.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}
