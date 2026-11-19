//! Names for GPU work, so captures read like the step plan.
//!
//! Passes are labelled after the plan's phase (`step:broadphase`) and, with debug labels on
//! ([`crate::WgpuSim::set_debug_labels`], `RUBBLE_GPU_LABELS=1`), every dispatch sits in a
//! debug group `phase/kernel`. Dawn forwards both to Metal (`MTLComputeCommandEncoder`
//! labels and `pushDebugGroup`), so a Chrome GPU trace, an Xcode capture of wgpu-on-Metal
//! and Instruments' Metal System Trace show the plan's names instead of `step`.
//! Labels change no work: the same dispatches, in the same passes.

use engine_exec::rigid_plan::Cmd;

/// `RUBBLE_GPU_LABELS=1` (natively; in the browser use the setter).
pub fn env_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("RUBBLE_GPU_LABELS").is_ok_and(|v| v == "1")
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// The phase each command of `cmds` belongs to: a phase marker follows its commands, so a
/// command's phase is the name of the next marker (`"tail"` after the last).
pub fn phase_of(cmds: &[Cmd]) -> Vec<&'static str> {
    let mut out = vec!["tail"; cmds.len()];
    let mut next = "tail";
    for (k, c) in cmds.iter().enumerate().rev() {
        if let Cmd::Phase { name } = *c {
            next = name;
        }
        out[k] = next;
    }
    out
}

/// A pass label: `step:<phase>` (static strings for the plan's phases, no allocation).
pub fn pass_label(phase: Option<&str>) -> String {
    match phase {
        Some(p) => format!("step:{p}"),
        None => "step".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_belongs_to_the_next_marker() {
        let cmds = [
            Cmd::Dispatch {
                kernel: "a",
                threads: 1,
                pass: 0,
            },
            Cmd::Phase { name: "one" },
            Cmd::Dispatch {
                kernel: "b",
                threads: 1,
                pass: 0,
            },
            Cmd::Dispatch {
                kernel: "c",
                threads: 1,
                pass: 0,
            },
            Cmd::Phase { name: "two" },
            Cmd::Dispatch {
                kernel: "d",
                threads: 1,
                pass: 0,
            },
        ];
        assert_eq!(phase_of(&cmds), ["one", "one", "two", "two", "two", "tail"]);
    }
}
