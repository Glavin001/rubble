//! Measurement only: `RUBBLE_PIPE_LOG=1` prints each World kernel pipeline the host creates
//! and how long the creation took (a pipeline created at its first use costs that frame
//! its compile).

/// Logs its label and the time since it was made when dropped (with the variable set).
pub(crate) struct Timed(Option<(String, std::time::Instant)>);

impl Timed {
    pub(crate) fn new(label: &str) -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("RUBBLE_PIPE_LOG").as_deref() == Ok("1"));
        Self(on.then(|| (label.to_string(), std::time::Instant::now())))
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        if let Some((label, t)) = self.0.take() {
            eprintln!("pipeline {label}: created in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
