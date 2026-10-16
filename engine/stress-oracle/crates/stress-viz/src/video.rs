//! Frame sinks: an H.264 MP4 through an `ffmpeg` child process (raw RGB24 on its
//! stdin), and numbered PNG files.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use crate::canvas::Canvas;

pub struct Ffmpeg {
    child: Child,
    stdin: Option<ChildStdin>,
    path: PathBuf,
}

impl Ffmpeg {
    /// Starts `ffmpeg` encoding `width x height` RGB24 frames at `fps` into `path`.
    pub fn start(path: &Path, width: usize, height: usize, fps: f64) -> Result<Ffmpeg, String> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
        let mut child = Command::new(&ffmpeg)
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}"), "-r", &format!("{fps}"), "-i", "-"])
            .args(["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start {ffmpeg}: {e}"))?;
        let stdin = child.stdin.take();
        Ok(Ffmpeg { child, stdin, path: path.to_path_buf() })
    }

    pub fn write(&mut self, frame: &Canvas) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("ffmpeg stdin closed")?;
        stdin.write_all(&frame.pixels).map_err(|e| format!("writing to ffmpeg: {e}"))
    }

    /// Closes the stream and waits for the encoder.
    pub fn finish(mut self) -> Result<PathBuf, String> {
        drop(self.stdin.take());
        let status = self.child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
        if status.success() {
            Ok(self.path)
        } else {
            Err(format!("ffmpeg failed ({status}) writing {}", self.path.display()))
        }
    }
}

/// Writes `frame` as a PNG file.
pub fn write_png(path: &Path, frame: &Canvas) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, crate::png::encode(frame.width, frame.height, &frame.pixels))
        .map_err(|e| format!("{}: {e}", path.display()))
}
