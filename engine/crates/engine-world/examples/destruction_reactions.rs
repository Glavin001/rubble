//! Count every destruction reaction of a reference scene and shot
//! (`engine/docs/destruction-reactions.md`; the harness is `tests/support/reactions.rs`).
//!
//! ```sh
//! cd engine
//! cargo run --release -p engine-world --example destruction_reactions -- \
//!   crush-wall:1.4x reference-building-crush:ordinary reference-building-crush:gravity \
//!   [--seconds 6] [--json out.json] [--md out.md] \
//!   [--crush] [--support-loss] [--blast-loading] [--blast x,y,z,kg [--blast-at 2]]
//! ```
//!
//! Each argument is `scene:shot` (a label the pack authors in `defaults.shots`, or
//! `gravity` for none); `--list` prints the scenes and shots. `--crush` and
//! `--support-loss` turn on the destructible's `crush` (the pack's crush blocks,
//! `CrushMass::Conserve`) and `support_loss`; `--blast` detonates a charge at a point of the
//! pack's frame (m) of that TNT mass (kg) at `--blast-at` seconds (default 2, after the
//! settle) with the destructible's `blast_loading` on (`--blast-loading`: on without a
//! charge). Every run takes the same options.
//!
//! The tall building partially destroyed (`engine/docs/destruction-reactions.md`):
//!
//! ```sh
//! cargo run --release -p engine-world --example destruction_reactions -- \
//!   reference-building-crush:gravity --crush --support-loss --blast -0.9,2.1,-1.70,50 --seconds 6
//! ```

#[path = "../tests/support/reactions.rs"]
mod reactions;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut runs, mut seconds, mut json, mut md) = (Vec::new(), 6.0, None, None);
    let (mut crush, mut support_loss, mut blast, mut blast_at) = (false, false, None, 2.0);
    let mut blast_loading = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => seconds = args.next().unwrap_or_default().parse()?,
            "--crush" => crush = true,
            "--support-loss" => support_loss = true,
            "--blast-loading" => blast_loading = true,
            "--blast" => blast = Some(args.next().unwrap_or_default()),
            "--blast-at" => blast_at = args.next().unwrap_or_default().parse()?,
            "--json" => json = args.next(),
            "--md" => md = args.next(),
            "--list" => {
                for (key, bytes) in reactions::SCENES {
                    let shots = engine_fracture::pack::pack_shots(bytes)?;
                    println!(
                        "{key}: gravity {}",
                        shots
                            .iter()
                            .map(|s| format!(
                                "{} ({:.0} kg at {:.1} m/s; {})",
                                s.label,
                                s.mass,
                                glam::DVec3::from(s.velocity).length(),
                                s.note
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                return Ok(());
            }
            _ => runs.push(a),
        }
    }
    if runs.is_empty() {
        runs = vec![
            "crush-wall:1.4x".into(),
            "reference-building-crush:ordinary".into(),
        ];
    }
    let mut reports = Vec::new();
    for r in &runs {
        let (scene, shot) = r.split_once(':').unwrap_or((r, "gravity"));
        let spec = reactions::RunSpec {
            scene: scene.into(),
            shot: (shot != "gravity").then(|| shot.to_string()),
            seconds,
            blast: blast
                .as_deref()
                .map(|b| reactions::BlastSpec::parse(b, blast_at))
                .transpose()?,
            crush,
            support_loss,
            blast_loading,
        };
        let t = std::time::Instant::now();
        let report = reactions::run(&spec)?;
        eprintln!(
            "{r}: {} frames in {:.1} s",
            report.frames.len(),
            t.elapsed().as_secs_f64()
        );
        reports.push(report);
    }
    let table = reactions::to_markdown(&reports);
    println!("{table}");
    if let Some(p) = md {
        std::fs::write(p, &table)?;
    }
    if let Some(p) = json {
        let all: Vec<_> = reports.iter().map(reactions::to_json).collect();
        std::fs::write(p, serde_json::to_string_pretty(&all)?)?;
    }
    Ok(())
}
