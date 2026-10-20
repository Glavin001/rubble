#!/usr/bin/env python3
"""The run specs of the P0 jobs (for `p0_runs.py --spec`), as they were generated.

    p0_specs.py ceiling|base_cpu|base_gpu|diag > SPEC.json
"""
import json
import sys

FH = {"STRESS_GPU_FIXED_HORIZON": "1"}


def ceiling():
    spec = []
    for s in ["free_wall", "wall_row", "free_wall_late", "wall_row_late"]:
        for fd in ["1/60", "1/120"]:
            f = fd.replace("/", "_")
            for k in range(16):
                m = ["--member", str(k), "--perturb", "kin", "--frame-dt", fd, "--seconds", "0.2"]
                spec.append({"name": f"{s}_oracle_{f}_{k:02d}", "args": ["oracle", "--scene", s] + m})
                vs = ["coupled", "window_full", "window_reduced", "pre_filtered", "pre_engine"] if fd == "1/60" else ["window_reduced", "pre_filtered"]
                for v in vs:
                    spec.append({"name": f"{s}_{v}_{f}_{k:02d}", "args": ["ceiling", "--scene", s, "--variant", v] + m})
    return spec


def base(kind):
    cpu, gpu = [], []
    for s in ["free_wall_late2", "wall_row_late2"]:
        for k in range(16):
            m = ["--scene", s, "--member", str(k), "--perturb", "kin", "--seconds", "0.25"]
            cpu.append({"name": f"{s}_oracle_1_60_{k:02d}", "args": ["oracle"] + m})
            cpu.append({"name": f"{s}_zerobreak_1_60_{k:02d}", "args": ["zero_break"] + m})
            cpu.append({"name": f"{s}_window_reduced_1_60_{k:02d}", "args": ["ceiling", "--variant", "window_reduced"] + m})
            for name, prof, env in [("gpu_realtime", "realtime", {}), ("gpu_blast_parity", "blast_parity", {}), ("gpu_qs", "qs", {}), ("gpu_routed", "blast_parity_routed", {}),
                                    ("gpu_punch", "blast_parity_routed", {"STRESS_IMPACT_PUNCH": "gpu"}), ("gpu_window4ms", "blast_parity_routed", {"STRESS_IMPACT_WINDOW": "0.004"})]:
                gpu.append({"name": f"{s}_{name}_1_60_{k:02d}", "args": ["gpu", "--profile", prof] + m, "env": dict(FH, **env)})
    return cpu if kind == "cpu" else gpu


def diag():
    return [{"name": f"{s}_{mode}_{k:02d}", "args": ["oracle", "--scene", s, "--member", str(k), "--perturb", mode]}
            for mode in ["speed", "point_um", "x_only", "z_only"] for k in range(6) for s in ["free_wall", "wall_row"]]


if __name__ == "__main__":
    which = sys.argv[1]
    spec = {"ceiling": ceiling, "base_cpu": lambda: base("cpu"), "base_gpu": lambda: base("gpu"), "diag": diag}[which]()
    json.dump(spec, sys.stdout)
