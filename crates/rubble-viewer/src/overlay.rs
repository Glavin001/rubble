use rubble_gpu::{StepTimingsMs, BROADPHASE_SUB_LABELS, STEP_TIMING_LABELS};

fn stat_card(ui: &mut egui::Ui, label: &str, value: String) {
    egui::Frame::new()
        .fill(egui::Color32::from_rgba_unmultiplied(255, 255, 255, 14))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 24),
        ))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .size(11.0)
                        .color(egui::Color32::from_gray(150)),
                );
                ui.label(
                    egui::RichText::new(value)
                        .size(22.0)
                        .strong()
                        .color(egui::Color32::from_gray(235)),
                );
            });
        });
}

#[allow(clippy::too_many_arguments)]
pub fn draw_panel(
    ctx: &egui::Context,
    title: &str,
    controls: &[&str],
    scene_names: &[String],
    selected_scene: &mut usize,
    reset_requested: &mut bool,
    fps: f32,
    body_count: usize,
    timings: &StepTimingsMs,
    render_ms: f32,
) {
    let arr = timings.as_array();
    let total: f32 = arr.iter().sum::<f32>() + timings.cpu_sync_ms;
    let panel_frame = egui::Frame::new()
        .fill(egui::Color32::from_rgba_unmultiplied(18, 20, 26, 232))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 24),
        ))
        .corner_radius(14.0)
        .inner_margin(egui::Margin::same(14));

    egui::Area::new(egui::Id::new("viewer_overlay_panel"))
        .fixed_pos(egui::pos2(16.0, 16.0))
        .show(ctx, |ui| {
            ui.set_width(320.0);
            panel_frame.show(ui, |ui| {
                ui.set_width(320.0);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);

                ui.label(
                    egui::RichText::new(title)
                        .size(22.0)
                        .strong()
                        .color(egui::Color32::from_gray(240)),
                );
                ui.label(
                    egui::RichText::new("Scenes, shortcuts, and live performance timing")
                        .size(12.0)
                        .color(egui::Color32::from_gray(150)),
                );

                ui.add_space(4.0);
                ui.separator();

                ui.label(
                    egui::RichText::new("Scene Controls")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_gray(225)),
                );
                egui::ComboBox::from_id_salt("scene_selector")
                    .width(ui.available_width() - 4.0)
                    .selected_text(&scene_names[*selected_scene])
                    .show_ui(ui, |ui| {
                        for (idx, name) in scene_names.iter().enumerate() {
                            ui.selectable_value(selected_scene, idx, name);
                        }
                    });
                if ui
                    .add_sized(
                        [ui.available_width(), 28.0],
                        egui::Button::new("Reset Scene"),
                    )
                    .clicked()
                {
                    *reset_requested = true;
                }

                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new("Controls")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_gray(225)),
                );
                egui::Frame::new()
                    .fill(egui::Color32::from_rgba_unmultiplied(255, 255, 255, 10))
                    .corner_radius(8.0)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        for line in controls {
                            ui.label(
                                egui::RichText::new(*line)
                                    .size(12.5)
                                    .color(egui::Color32::from_gray(190)),
                            );
                        }
                    });

                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new("Overview")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_gray(225)),
                );
                ui.columns(2, |columns| {
                    stat_card(&mut columns[0], "FPS", format!("{fps:.0}"));
                    stat_card(&mut columns[1], "Bodies", body_count.to_string());
                });
                ui.columns(2, |columns| {
                    stat_card(&mut columns[0], "Step", format!("{total:.2} ms"));
                    stat_card(&mut columns[1], "Render", format!("{render_ms:.2} ms"));
                });

                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new("Pipeline Breakdown")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_gray(225)),
                );
                egui::Grid::new("pipeline_grid")
                    .num_columns(4)
                    .spacing([8.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Stage").strong());
                        ui.label(egui::RichText::new("Lane").strong());
                        ui.label(egui::RichText::new("Time").strong());
                        ui.label(egui::RichText::new("%").strong());
                        ui.end_row();

                        for (i, &(name, kind)) in STEP_TIMING_LABELS.iter().enumerate() {
                            let ms = arr[i];
                            let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
                            ui.label(
                                egui::RichText::new(name).color(egui::Color32::from_gray(210)),
                            );
                            ui.label(
                                egui::RichText::new(kind).color(egui::Color32::from_gray(140)),
                            );
                            ui.label(
                                egui::RichText::new(format!("{ms:.2} ms"))
                                    .color(egui::Color32::from_gray(210)),
                            );
                            ui.label(
                                egui::RichText::new(format!("{pct:.1}%"))
                                    .color(egui::Color32::from_gray(170)),
                            );
                            ui.end_row();
                        }

                        if timings.cpu_sync_ms > 0.0 {
                            let pct = if total > 0.0 {
                                timings.cpu_sync_ms / total * 100.0
                            } else {
                                0.0
                            };
                            ui.label(
                                egui::RichText::new("StateSync")
                                    .color(egui::Color32::from_gray(210)),
                            );
                            ui.label(
                                egui::RichText::new("(CPU)").color(egui::Color32::from_gray(140)),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.2} ms", timings.cpu_sync_ms))
                                    .color(egui::Color32::from_gray(210)),
                            );
                            ui.label(
                                egui::RichText::new(format!("{pct:.1}%"))
                                    .color(egui::Color32::from_gray(170)),
                            );
                            ui.end_row();
                        }

                        ui.label(
                            egui::RichText::new("Render").color(egui::Color32::from_gray(210)),
                        );
                        ui.label(egui::RichText::new("(GPU)").color(egui::Color32::from_gray(140)));
                        ui.label(
                            egui::RichText::new(format!("{render_ms:.2} ms"))
                                .color(egui::Color32::from_gray(210)),
                        );
                        ui.label(egui::RichText::new("-").color(egui::Color32::from_gray(170)));
                        ui.end_row();
                    });

                {
                    ui.add_space(2.0);
                    egui::CollapsingHeader::new("Broadphase Details")
                        .default_open(false)
                        .show(ui, |ui| {
                            let bp_total = timings.broadphase_breakdown.total_ms();
                            let bp_arr = timings.broadphase_breakdown.as_array();
                            egui::Grid::new("broadphase_grid")
                                .num_columns(4)
                                .spacing([8.0, 4.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label(egui::RichText::new("Stage").strong());
                                    ui.label(egui::RichText::new("Lane").strong());
                                    ui.label(egui::RichText::new("Time").strong());
                                    ui.label(egui::RichText::new("% of BP").strong());
                                    ui.end_row();

                                    for (j, &(sub_name, sub_kind)) in
                                        BROADPHASE_SUB_LABELS.iter().enumerate()
                                    {
                                        let sub_ms = bp_arr[j];
                                        let sub_pct = if bp_total > 0.0 {
                                            sub_ms / bp_total * 100.0
                                        } else {
                                            0.0
                                        };
                                        ui.label(
                                            egui::RichText::new(sub_name)
                                                .color(egui::Color32::from_gray(195)),
                                        );
                                        ui.label(
                                            egui::RichText::new(sub_kind)
                                                .color(egui::Color32::from_gray(140)),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{sub_ms:.2} ms"))
                                                .color(egui::Color32::from_gray(195)),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{sub_pct:.1}%"))
                                                .color(egui::Color32::from_gray(170)),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                }

                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new("Shortcuts: R resets the scene, Esc quits")
                        .size(11.5)
                        .color(egui::Color32::from_gray(128)),
                );
            });
        });
}
