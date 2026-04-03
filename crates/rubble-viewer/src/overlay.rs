use rubble_gpu::StepTimingsMs;

const STAGE_LABELS: &[(&str, &str)] = &[
    ("Upload", "(CPU)"),
    ("Predict", "(GPU)"),
    ("Broadphase", "(GPU+CPU)"),
    ("Narrowphase", "(GPU)"),
    ("Contacts", "(GPU>CPU)"),
    ("Solve", "(GPU)"),
    ("Extract", "(GPU)"),
];

pub fn draw_stats(
    ctx: &egui::Context,
    fps: f32,
    body_count: usize,
    timings: &StepTimingsMs,
    render_ms: f32,
) {
    let arr = timings.as_array();
    let total: f32 = arr.iter().sum();

    egui::Area::new(egui::Id::new("stats_overlay"))
        .fixed_pos(egui::pos2(12.0, 12.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(180))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.style_mut().visuals.override_text_color =
                        Some(egui::Color32::from_gray(200));
                    ui.spacing_mut().item_spacing.y = 2.0;

                    ui.monospace(format!("FPS: {fps:.0}"));
                    ui.monospace(format!("Bodies: {body_count}"));
                    ui.add_space(4.0);
                    ui.monospace(format!("Step: {total:.2} ms"));

                    for (i, &(name, kind)) in STAGE_LABELS.iter().enumerate() {
                        let ms = arr[i];
                        let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
                        ui.monospace(format!(
                            "  {:<12}{:<10}{:>6.2} ms {:>3.0}%",
                            name, kind, ms, pct
                        ));
                    }

                    ui.monospace(format!("Render      (GPU)    {:>6.2} ms", render_ms));
                    ui.add_space(6.0);
                    ui.monospace(
                        egui::RichText::new("[R] reset  [Esc] quit")
                            .color(egui::Color32::from_gray(120)),
                    );
                });
        });
}
