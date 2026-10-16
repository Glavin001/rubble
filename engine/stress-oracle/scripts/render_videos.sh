#!/usr/bin/env bash
# Renders the debug-visualisation video set into videos/: every benchmark scene
# (builders::catalog) and every showcase (showcases::catalog), each with a view
# layout and camera suited to it, plus a few feature comparisons.
#
#   scripts/render_videos.sh                # everything
#   scripts/render_videos.sh b5 s_arch      # only names containing one of the patterns
#
# Environment: JOBS (render threads, default 2), CARGO_TARGET_DIR, VIDEOS (output dir,
# default videos), EXTRA (extra stress-viz options for every video, e.g. "--size 1280x720"),
# PARALLEL (videos rendered at once, default 1; with more, each video's log goes to
# $VIDEOS/logs/NAME.log). The solver scales little beyond ~4 threads, so on a many-core
# machine e.g. PARALLEL=4 RAYON_NUM_THREADS=4 finishes the set much sooner.
set -euo pipefail

cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}
VIDEOS=${VIDEOS:-videos}
PARALLEL=${PARALLEL:-1}
mkdir -p "$VIDEOS"

# RUBBLE=1 (default) also builds rubble's AVBD engine into the renderer and renders the
# oracle-vs-rubble comparisons at the end (they need a wgpu adapter); RUBBLE=0 skips them.
RUBBLE=${RUBBLE:-1}
if [ "$RUBBLE" = 1 ]; then
  cargo build --release --offline -p stress-viz --features rubble
else
  cargo build --release --offline -p stress-viz
fi
VIZ="${CARGO_TARGET_DIR:-target}/release/stress-viz"

PATTERNS=("$@")
# selected NAME: true when no patterns were given or NAME contains one of them.
selected() {
  [ ${#PATTERNS[@]} -eq 0 ] && return 0
  local pattern
  for pattern in "${PATTERNS[@]}"; do
    [[ "$1" == *"$pattern"* ]] && return 0
  done
  return 1
}

# launch OUTPUT stress-viz-arguments...: runs in the foreground, or with PARALLEL > 1 in
# the background once fewer than PARALLEL are running (bash 3.2 has no `wait -n`).
FAILED=0
launch() {
  local out=$1; shift
  echo "=== $out"
  if [ "$PARALLEL" -le 1 ]; then
    nice -n 10 "$VIZ" "$@" || FAILED=$((FAILED + 1))
    return 0
  fi
  mkdir -p "$VIDEOS/logs"
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 0.2; done
  (nice -n 10 "$VIZ" "$@" >"$VIDEOS/logs/$out.log" 2>&1 || echo "!!! $out failed, see $VIDEOS/logs/$out.log") &
}

# render NAME [stress-viz options...]
render() {
  local name=$1; shift
  selected "$name" || return 0
  launch "$name" render "$name" --jobs "$JOBS" -o "$VIDEOS/$name.mp4" "$@" ${EXTRA:-}
}

# compare OUTPUT NAME [stress-viz options...]
compare() {
  local out=$1 name=$2; shift 2
  selected "$out" || return 0
  launch "$out" compare "$name" --jobs "$JOBS" -o "$VIDEOS/$out.mp4" "$@" ${EXTRA:-}
}

# --- benchmarks (builders::catalog) -------------------------------------------------
for s in b1_bond_tension b1_bond_shear; do
  probe=$([ "$s" = b1_bond_tension ] && echo axial || echo shear)
  render "$s" --views utilization,stress,damage --camera iso --plot "$probe"
done
for s in b2_cantilever b2_cantilever_n10 b2_cantilever_n40; do
  render "$s" --views utilization,stress,deformation --cols 1 --camera front --pitch 10 --plot tip
done
for s in b3_bar_wave b3_bar_wave_scaled; do
  render "$s" --views stress,velocity --cols 1 --yaw -15 --pitch 15 --plot v20,v50,v80
done
for s in b4_support_loss_sudden b4_support_loss_gradual; do
  render "$s" --views utilization,stress,velocity --camera iso --plot reaction
done
for s in b5_wall_impact_v02 b5_wall_impact_v10 b5_wall_impact_v40; do
  render "$s" --views utilization,stress,damage,fragments --arrows
done
# Spall: section through the slab (the half x < 0 is cut away), wave and crack planes.
render b6_spall --views stress,velocity,damage --clip "x<0" --yaw -50 --pitch 20 \
  --plot back_face_velocity,front_face_velocity
for s in b7_masonry_v04 b7_masonry_v15; do
  render "$s" --views utilization,damage,fragments,velocity --arrows --plot ball_velocity
done
for s in b8_frame_sudden b8_frame_gradual b8_frame_sudden_elastic; do
  render "$s" --views utilization,stress,damage,fragments --camera front --plot axial_col0,axial_col2
done
for s in b9_panel_low b9_panel_high b9_panel_high_weibull; do
  render "$s" --views utilization,damage,fragments,velocity --yaw 35 --plot center_displacement
done

# --- showcases (showcases::catalog) -------------------------------------------------
for s in s_overhang_thin s_overhang_thick; do
  render "$s" --views utilization,stress,damage,fragments --yaw -20 --pitch 12 --arrows --plot slab_tip
done
render s_supports_one_by_one --views utilization,stress,damage,fragments --camera front --pitch 15
for s in s_car_brick s_car_ductile; do
  render "$s" --views utilization,damage,fragments,velocity --arrows --plot car_velocity
done
for s in s_floor_drop s_floor_static; do
  render "$s" --views utilization,damage,fragments,velocity --camera iso
done
for s in s_arch_keystone_removed s_arch; do
  render "$s" --views utilization,stress,damage,fragments --camera front --arrows
done
render s_blast_two_walls --views utilization,damage,fragments,velocity --camera iso --arrows
# Brick house: the self-weight stress field around the openings, then three failures.
render s_house --views utilization,stress,von_mises,deformation --camera iso
render s_house_settlement --views utilization,stress,damage,fragments --camera iso --plot broken
render s_house_car --views utilization,damage,fragments,velocity --camera iso --yaw 40 --arrows
render s_house_blast --views utilization,damage,fragments,velocity --camera iso --yaw -30 --arrows

# --- comparisons --------------------------------------------------------------------
compare s_overhang_thin_fatigue s_overhang_thin \
  --variant "static fatigue on:" --variant "static fatigue off:feature static_fatigue=off" \
  --views utilization,damage --yaw -20 --pitch 12 --plot slab_tip
compare b5_wall_impact_v10_rate b5_wall_impact_v10 \
  --variant "rate effects on:" --variant "rate effects off:feature rate_effects=off" \
  --views utilization,fragments
compare b7_masonry_v15_modes b7_masonry_v15 \
  --variant "explicit:" --variant "adaptive:sim.solve_mode=adaptive" \
  --views utilization,fragments --plot ball_velocity

# Lead demos: same ram fast vs slow (inertia decides hole vs push-over) and sudden vs
# gradual column loss (dynamic amplification).
compare lead_fast_vs_slow b5_wall_impact_v40 \
  --variant "40 m/s:" --variant "2 m/s:impactors.0.velocity=[0,2,0]" \
  --views utilization,fragments --duration 0.3
compare lead_sudden_vs_gradual b8_frame_sudden \
  --variant "sudden removal:" --variant "gradual removal (2 s):events.0.duration=2.0" \
  --views utilization,damage --camera front
# What each capability contributes: the same scene with it switched off.
compare feature_inertia b5_wall_impact_v40 \
  --variant "explicit dynamics:" --variant "quasi-static (no inertia):sim.solve_mode=quasi_static" \
  --views utilization,fragments
compare feature_stiffness_scaling b5_wall_impact_v40 \
  --variant "true stiffness:" --variant "E x 0.01:sim.stiffness_scale=0.01" --variant "mass scaling to 4 x dt:sim.mass_scaling_dt=1e-5" \
  --views utilization,fragments
compare feature_implicit b8_frame_sudden \
  --variant "explicit:" --variant "implicit at the frame step:sim.solve_mode=implicit" \
  --views utilization,damage --camera front
compare feature_shadowing s_blast_two_walls \
  --variant "shadowing on:" --variant "shadowing off:feature blast_shadowing=off" \
  --views damage,fragments --camera iso
compare feature_softening b5_wall_impact_v10 \
  --variant "fracture-energy softening:" --variant "threshold (no softening):feature softening=off" \
  --views damage,fragments
compare feature_crack_contact s_arch_keystone_removed \
  --variant "cracked joints carry compression:" --variant "cracked joints carry nothing:feature crack_contact=off" \
  --views utilization,fragments --camera front
compare feature_weibull b9_panel_high_weibull \
  --variant "Weibull strengths:" --variant "uniform strengths:feature weibull=off" \
  --views damage,fragments --yaw 35

# --- the oracle vs rubble as the rigid-body engine (RUBBLE=1) -----------------------
# Left: the standalone world (its own rigid bodies and contact, the oracle). Right: the
# same scene with rubble's AVBD GPU engine moving the bodies, the stress solve owning
# impact islands.
if [ "$RUBBLE" = 1 ]; then
  # rubble NAME [stress-viz options...]
  rubble() {
    local name=$1; shift
    compare "rubble_$name" "$name" --variant "oracle (standalone world):" --variant "rubble rigid bodies:engine rubble" "$@"
  }
  rubble s_arch_keystone_removed --views utilization,fragments --camera front --arrows
  rubble s_floor_drop --views utilization,fragments --camera iso
  rubble s_overhang_thin --views utilization,fragments --yaw -20 --pitch 12 --plot slab_tip
  rubble s_supports_one_by_one --views utilization,fragments --camera front --pitch 15
  rubble s_car_brick --views utilization,fragments --arrows --plot car_velocity
  rubble s_blast_two_walls --views damage,fragments --camera iso --arrows
  rubble s_house_car --views utilization,fragments --camera iso --yaw 40 --arrows
  rubble s_house_settlement --views utilization,fragments --camera iso --plot broken
  rubble b8_frame_sudden --views utilization,fragments --camera front --plot axial_col0,axial_col2
  rubble b7_masonry_v15 --views utilization,fragments --arrows --plot ball_velocity
  rubble b5_wall_impact_v40 --views utilization,fragments --arrows
fi

wait
echo "videos in $VIDEOS/"
[ "$FAILED" -eq 0 ]
