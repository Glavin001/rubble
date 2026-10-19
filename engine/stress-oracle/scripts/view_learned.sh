#!/usr/bin/env bash
# Recorded-frame comparison using the latest locally trained combined checkpoint.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
oracle_dir="$(cd -- "$script_dir/.." && pwd)"
run_dir="$oracle_dir/learned-impact/runs/combined-2026-10-19"
expanded="$oracle_dir/learned-impact/runs/expanded-2026-10-19"
physics="$oracle_dir/learned-impact/runs/physics-v2-2026-10-19"
python_bin="${STRESS_LEARNED_PYTHON:-$oracle_dir/learned-impact/python/.venv/bin/python}"
if [[ ! -f "$run_dir/model/state_dict.pt" ]]; then
  echo "Missing trained checkpoint: $run_dir/model/state_dict.pt" >&2
  echo "See learned-impact/COMPARISON.md to collect data and train." >&2
  exit 1
fi
preview="$oracle_dir/learned-impact/runs/causal-preview-2026-10-19"
preview_args=()
if [[ -f "$preview/matched-comparison.json" && -f "$preview/viewer-dataset/manifest.json" ]]; then
  preview_args=(--preview-dataset "$preview/viewer-dataset" --preview-weights "$preview/oracle-inputs" --preview-weights "$preview/preview-inputs")
  state_run="$oracle_dir/learned-impact/runs/state-v4-2026-10-19"
  if [[ -f "$state_run/final-comparison.json" ]]; then
    for variant in constrained frozen-heads; do
      if [[ -f "$state_run/$variant/metrics.json" ]]; then
        preview_args+=(--preview-weights "$state_run/$variant")
      fi
    done
  fi
fi
world_run="$oracle_dir/learned-impact/runs/world-v5-2026-10-19"
long_run="$oracle_dir/learned-impact/runs/long-fit-v5-2026-10-19"
motion_run="$oracle_dir/learned-impact/runs/world-motion-v6-2026-10-19"
forcing_run="$oracle_dir/learned-impact/runs/native-forcing-v7-2026-10-19"
long_native_ready=false
if "$python_bin" - "$forcing_run" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
try:
    digest=hashlib.sha256((r/'native-long/state_dict.pt').read_bytes()).hexdigest()
    plan=json.loads((r/'long-training-plan.json').read_text())
    comparison=json.loads((r/'long-comparison.json').read_text())
    parity=json.loads((r/'native-long-mps-parity.json').read_text())
    scale=json.loads((r/'native-long-scale.json').read_text())
    assert plan['status']=='completed'
    assert any(m['checkpoint']['sha256']==digest for m in comparison['models'])
    assert parity['checkpoint']['sha256']==scale['checkpoint']['sha256']==digest
    assert parity['frames']>0 and parity['break_decision_disagreements']==parity['fail_class_disagreements']==0
    assert parity['gpu_feature_construction'] and scale['chunks']==10400
except (OSError,KeyError,ValueError,AssertionError):
    sys.exit(1)
PY
then
  long_native_ready=true
fi
if [[ -f "$world_run/matched-comparison.json" && ${#preview_args[@]} -gt 0 ]]; then
  for variant in control world; do
    preview_args+=(--preview-weights "$world_run/$variant")
  done
fi
if [[ -f "$world_run/extended-comparison.json" && -f "$world_run/viewer-tails/manifest.json" ]]; then
  preview_args+=(--tail-dataset "$world_run/viewer-tails" --tail-weights "$world_run/extended-world")
  sampling_run="$oracle_dir/learned-impact/runs/sampling-v5-2026-10-19"
  if [[ -f "$sampling_run/comparison.json" ]]; then
    for variant in frame physical-group; do
      preview_args+=(--tail-weights "$sampling_run/$variant")
    done
  fi
  if [[ -f "$long_run/comparison.json" ]]; then
    preview_args+=(--tail-weights "$long_run/physical-group")
  fi
  if [[ -f "$motion_run/comparison.json" ]]; then
    for variant in control auxiliary; do
      preview_args+=(--tail-weights "$motion_run/$variant")
    done
  fi
  if [[ -f "$forcing_run/comparison.json" && -f "$forcing_run/native-mps-parity.json" ]]; then
    for variant in preview native; do
      preview_args+=(--tail-weights "$forcing_run/$variant")
    done
  fi
  if $long_native_ready; then
    preview_args+=(--tail-weights "$forcing_run/native-long")
  fi
fi
layout_run="$oracle_dir/learned-impact/runs/layout-v4-2026-10-19"
if [[ -f "$layout_run/combined-comparison.json" && -f "$layout_run/viewer-layouts/manifest.json" ]]; then
  preview_args+=(--layout-dataset "$layout_run/viewer-layouts" --layout-weights "$long_run/physical-group" --layout-weights "$layout_run/combined-world-v5")
  if [[ -f "$forcing_run/comparison.json" && -f "$forcing_run/native-mps-parity.json" ]]; then
    for variant in preview native; do
      preview_args+=(--layout-weights "$forcing_run/$variant")
    done
  fi
  if $long_native_ready; then
    preview_args+=(--layout-weights "$forcing_run/native-long")
  fi
  if [[ -f "$world_run/viewer-tails/manifest.json" ]]; then
    preview_args+=(--tail-weights "$layout_run/combined-world-v5")
  fi
fi
if [[ -f "$layout_run/scale-evaluation.json" && -f "$layout_run/viewer-scale/manifest.json" ]]; then
  preview_args+=(--scale-dataset "$layout_run/viewer-scale" --scale-weights "$world_run/world")
  if [[ -f "$long_run/scale-evaluation.json" ]]; then
    preview_args+=(--scale-weights "$long_run/physical-group")
  fi
  if [[ -f "$motion_run/scale-evaluation.json" ]]; then
    preview_args+=(--scale-weights "$motion_run/auxiliary")
  fi
  if [[ -f "$layout_run/combined-scale-evaluation.json" && -f "$layout_run/combined-comparison.json" ]]; then
    preview_args+=(--scale-weights "$layout_run/combined-world-v5")
  fi
  if [[ -f "$forcing_run/comparison.json" && -f "$forcing_run/native-mps-parity.json" && -f "$forcing_run/native-scale.json" && -f "$forcing_run/preview-scale.json" ]]; then
    for variant in preview native; do
      preview_args+=(--scale-weights "$forcing_run/$variant")
    done
  fi
  if $long_native_ready; then
    preview_args+=(--scale-weights "$forcing_run/native-long")
  fi
fi
response_run="$oracle_dir/learned-impact/runs/response-model-v18-2026-10-19"
response_scale_run="$oracle_dir/learned-impact/runs/response-scale-v19-2026-10-19"
response_view="$oracle_dir/learned-impact/runs/runtime-start-v20-2026-10-19/viewer-responses"
if [[ -f "$response_view/manifest.json" && -f "$response_run/exact-response-audit.json" && -f "$response_scale_run/exact-response-audit.json" ]]; then
  preview_args+=(--response-dataset "$response_view" --response-weights "$response_run/wall-model" --response-weights "$response_scale_run/wall-model")
  projectile_response="$oracle_dir/learned-impact/runs/projectile-response-v22-2026-10-19"
  if [[ -f "$projectile_response/exact-response.json" && -f "$projectile_response/topology.json" && -f "$projectile_response/model/metrics.json" ]]; then
    preview_args+=(--response-weights "$projectile_response/model")
  fi
fi
impact_response="$oracle_dir/learned-impact/runs/response-impact128-v21-2026-10-19"
if [[ -f "$impact_response/viewer-preparation.json" && -f "$impact_response/viewer-responses/manifest.json" && -f "$impact_response/exact-response.json" && -f "$impact_response/topology.json" ]]; then
  preview_args+=(--impact-response-dataset "$impact_response/viewer-responses" --impact-response-weights "$impact_response/model")
fi
rollout_catalog="$oracle_dir/learned-impact/runs/independent-endpoint-v25-2026-10-19/catalog.json"
if [[ -f "$rollout_catalog" ]]; then
  preview_args+=(--rollout-catalog "$rollout_catalog")
fi
export PYTHONPATH="$oracle_dir/learned-impact/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$physics/expanded-all/metrics.json" && -f "$physics/expanded-heldout/metrics.json" ]]; then
  exec "$python_bin" -m learned_impact.visualize \
    --dataset "$expanded/dataset" \
    --weights "$physics/expanded-all" --weights "$physics/expanded-heldout" \
    --weights "$physics/physical-features" --weights "$run_dir/model" \
    --geometry "$run_dir/geometry.json" "${preview_args[@]}" "$@"
fi
exec "$python_bin" -m learned_impact.visualize \
  --dataset "$run_dir/dataset" --weights "$run_dir/model" \
  --geometry "$run_dir/geometry.json" "${preview_args[@]}" "$@"
