//! How many explicit substeps a frame needs (`engine/docs/deformable-terrain.md`,
//! "Time step"): each substep keeps the elastic wave speed plus the fastest
//! material point within `courant` cells.

use crate::config::{TerrainConfig, TerrainError};

/// `(substeps, dt)` covering `frame_dt` with dt ≤ courant·dx/(wave_speed + max_speed).
/// `max_speed` bounds the material's speed over the frame; the kernels check it
/// each substep and fail the step if a particle exceeds it.
pub fn substeps_for(
    config: &TerrainConfig,
    frame_dt: f32,
    dx: f32,
    wave_speed: f32,
    max_speed: f32,
) -> Result<(u32, f32), TerrainError> {
    if !(frame_dt > 0.0
        && frame_dt.is_finite()
        && dx > 0.0
        && wave_speed >= 0.0
        && max_speed >= 0.0)
    {
        return Err(TerrainError::InvalidRequest(
            "frame_dt and dx must be positive, speeds non-negative",
        ));
    }
    let speed = wave_speed + max_speed;
    let bound = if speed > 0.0 {
        config.courant * dx / speed
    } else {
        frame_dt
    };
    let n = (frame_dt / bound).ceil().max(1.0);
    TerrainConfig::check("max_substeps", n as u64, u64::from(config.max_substeps))?;
    let n = n as u32;
    Ok((n, frame_dt / n as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substeps_respect_the_courant_bound_and_cover_the_frame() {
        let c = TerrainConfig::default();
        // Sand-like P-wave speed 200 m/s, 10 cm cells, 5 m/s flow, a 60 Hz frame.
        let (n, dt) = substeps_for(&c, 1.0 / 60.0, 0.1, 200.0, 5.0).unwrap();
        assert!(dt * 205.0 <= c.courant * 0.1 * (1.0 + 1e-6));
        assert!((dt * n as f32 - 1.0 / 60.0).abs() < 1e-7);
        // One fewer substep would break the bound.
        assert!(1.0 / 60.0 / (n - 1) as f32 * 205.0 > c.courant * 0.1);
        assert_eq!(substeps_for(&c, 1.0 / 60.0, 0.1, 0.0, 0.0).unwrap().0, 1);
    }

    #[test]
    fn a_frame_needing_too_many_substeps_fails_by_name() {
        let c = TerrainConfig {
            max_substeps: 16,
            ..TerrainConfig::default()
        };
        let err = substeps_for(&c, 1.0 / 60.0, 0.01, 2000.0, 0.0).unwrap_err();
        assert!(
            matches!(
                err,
                TerrainError::LimitExceeded {
                    limit: "max_substeps",
                    ..
                }
            ),
            "{err}"
        );
    }
}
