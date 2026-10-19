"""Independent execution and explicit small-fixture accuracy gate (not deployment)."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from .native_comparison import compare

def run(binary,scene,oracle,weights,out,frames):
    out=Path(out)
    subprocess.run([str(binary),"energy",str(scene),str(out),str(frames),str(weights)],check=True)
    execution=json.loads((out/"report.json").read_text())
    assert execution["oracle_queries"]==0 and execution["analytic_fallback_evaluations"]==0
    plan=json.loads((out/"plan.json").read_text());plan["checkpoint"]["sha256"]=hashlib.sha256((out/"weights.json").read_bytes()).hexdigest()
    plan["code_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    (out/"plan.json").write_text(json.dumps(plan,indent=2))
    report=compare(oracle,out);b=report["breaks"];tp=b["oracle"]-b["missed"]
    f1=2*tp/max(2*tp+b["extra"]+b["missed"],1) if b["oracle"] or b["model"] else 1.
    actors=report["interactions"]["actors"]
    report["eventual_break_f1"]=f1
    report["small_fixture_checks"]={
        "oracle_free":True,
        "eventual_break_f1_at_least_0_9":f1>=.9,
        "break_count_within_10_percent":abs(b["model"]-b["oracle"])<=.1*b["oracle"],
        "largest_fragment_within_5_percentage_points":abs(report["model_fragments"]["largest_mass_fraction"]-report["oracle_fragments"]["largest_mass_fraction"])<=.05,
        "chunk_final_rmse_below_2cm":report["final_position_rmse_m"]<.02,
        "no_mechanical_energy_above_initial_plus_1_percent":report["model_max_mechanical_energy_j"]<=1.01*report["oracle_initial_mechanical_energy_j"],
        "projectile_direction_matches":bool(actors) and all(a["final_direction_disagreement"] is False for a in actors),
        "projectile_velocity_rmse_below_5_percent_of_initial":bool(actors) and all(a["velocity_vector_rmse_mps"]<=.05*abs(a["oracle"]["forward_velocity_mps"][0]) for a in actors),
    }
    # Clearance uses the ORIGINAL far wall plane. Rotated box rear face must
    # cross it; forward direction alone does not prove perforation.
    graph=json.loads((Path(oracle)/"graph.json").read_text())
    scene_data=graph["scene"]
    # This gate's fixture is explicitly the y-normal wall spanning [0, .1] m.
    if scene_data.get("name")=="causal-small-wall" and report["duration_seconds"]>=.1-1e-10:
        clearance={}
        for side,root in [("oracle",Path(oracle)),("model",out/"native")]:
            actor=json.loads((root/f"snapshots/{frames:06}.json").read_text())["world"]["impactors"][0]
            half=np.asarray(actor["shape"]["half_extents"]);rotation=np.asarray(actor["rotation"])
            rear=float(actor["center"][1]-np.abs(rotation[1]).dot(half))
            clearance[side]={"rear_y_m":rear,"cleared_original_far_wall_plane":rear>.1}
        report["wall_clearance"]=clearance
        report["small_fixture_checks"]["clearance_matches"]=clearance["oracle"]["cleared_original_far_wall_plane"]==clearance["model"]["cleared_original_far_wall_plane"]
    report["small_fixture_pass"]=all(report["small_fixture_checks"].values())
    report["gate_scope"]="Research thresholds for these 64-chunk wall fixtures only. Not a production accuracy or runtime acceptance gate. Native timesteps retained."
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"out":str(out),"breaks":b,"eventual_break_f1":f1,"checks":report["small_fixture_checks"],"pass":report["small_fixture_pass"]}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ["binary","scene","oracle","weights","out"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--frames",type=int,default=100)
    a=p.parse_args();run(a.binary,a.scene,a.oracle,a.weights,a.out,a.frames)
if __name__=="__main__":main()
