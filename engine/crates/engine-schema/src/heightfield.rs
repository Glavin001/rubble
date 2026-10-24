//! Heightfield terrain (deformable terrain T1; `engine/docs/deformable-terrain.md`).
//!
//! A heightfield is a static grid of `nx × nz` height samples, centred on its
//! collider's origin: vertex `(i, j)` sits at `(x0 + i·dx, h, z0 + j·dz)` with
//! `x0 = -(nx-1)·dx/2`, `z0 = -(nz-1)·dz/2`. Each cell is two triangles split
//! along its `(i, j)–(i+1, j+1)` diagonal; the outward side is `+y`.
//!
//! The samples are packed into the hull arena (Slang `gHulls`), so the GPU reads
//! them through the convex stages without an extra binding. Records at `base`:
//!
//! | record   | contents                                     |
//! |----------|----------------------------------------------|
//! | `base+0` | `HEIGHT_FIELD_MAGIC`, `nx`, `nz`, flags (0)  |
//! | `base+1` | `dx`, `dz`, min height, max height (f32 bits)|
//! | `base+2..` | heights, four per record, row-major `j·nx + i` |
//!
//! Collider `params`: `x` = tagged base (as for hulls), `w` = bounding radius.
//! `engine/slang/physics/heightfield.slang` is the GPU side.

use glam::Vec3;

use crate::convex::{HullArena, HullError, HULL_MAX_BASE, HULL_PARAM_TAG};

pub const HEIGHT_FIELD_MAGIC: u32 = 0x4846_0001;
/// Fewest samples per side (one cell).
pub const HEIGHT_FIELD_MIN_SAMPLES: u32 = 2;
/// Most samples per side. The GPU's feature ids hold a 24-bit vertex index;
/// fields past it would still collide correctly but alias warm starts.
pub const HEIGHT_FIELD_MAX_SAMPLES: u32 = 4096;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum HeightFieldError {
    #[error("heightfield needs {min}..={max} samples per side (got {nx} x {nz})", min = HEIGHT_FIELD_MIN_SAMPLES, max = HEIGHT_FIELD_MAX_SAMPLES)]
    Samples { nx: u32, nz: u32 },
    #[error("heightfield has {got} heights; {nx} x {nz} needs {want}")]
    HeightCount {
        got: usize,
        nx: u32,
        nz: u32,
        want: usize,
    },
    #[error("heightfield cell size must be finite and positive (got {dx} x {dz})")]
    CellSize { dx: f32, dz: f32 },
    #[error("heightfield height {index} is not finite")]
    NonFinite { index: usize },
    #[error("heightfield edit {i0}..{i1} x {j0}..{j1} is outside the {nx} x {nz} field")]
    EditBounds {
        i0: u32,
        i1: u32,
        j0: u32,
        j1: u32,
        nx: u32,
        nz: u32,
    },
    #[error("no heightfield at arena base {0}")]
    NotAHeightField(u32),
    #[error(transparent)]
    Arena(#[from] HullError),
}

/// A grid of heights (metres) with its cell size.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightField {
    nx: u32,
    nz: u32,
    dx: f32,
    dz: f32,
    heights: Vec<f32>,
}

impl HeightField {
    /// `heights` is row-major: sample `(i, j)` at index `j·nx + i`.
    pub fn new(
        nx: u32,
        nz: u32,
        dx: f32,
        dz: f32,
        heights: Vec<f32>,
    ) -> Result<Self, HeightFieldError> {
        let range = HEIGHT_FIELD_MIN_SAMPLES..=HEIGHT_FIELD_MAX_SAMPLES;
        if !range.contains(&nx) || !range.contains(&nz) {
            return Err(HeightFieldError::Samples { nx, nz });
        }
        if !(dx.is_finite() && dz.is_finite() && dx > 0.0 && dz > 0.0) {
            return Err(HeightFieldError::CellSize { dx, dz });
        }
        let want = nx as usize * nz as usize;
        if heights.len() != want {
            return Err(HeightFieldError::HeightCount {
                got: heights.len(),
                nx,
                nz,
                want,
            });
        }
        if let Some(index) = heights.iter().position(|h| !h.is_finite()) {
            return Err(HeightFieldError::NonFinite { index });
        }
        Ok(Self {
            nx,
            nz,
            dx,
            dz,
            heights,
        })
    }

    /// A field sampled from `f(x, z)` at every vertex (local coordinates).
    pub fn from_fn(
        nx: u32,
        nz: u32,
        dx: f32,
        dz: f32,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, HeightFieldError> {
        let (x0, z0) = origin(nx, nz, dx, dz);
        let mut heights = Vec::with_capacity(nx as usize * nz as usize);
        for j in 0..nz {
            for i in 0..nx {
                heights.push(f(x0 + i as f32 * dx, z0 + j as f32 * dz));
            }
        }
        Self::new(nx, nz, dx, dz, heights)
    }

    pub fn samples(&self) -> (u32, u32) {
        (self.nx, self.nz)
    }

    pub fn cell_size(&self) -> (f32, f32) {
        (self.dx, self.dz)
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn height(&self, i: u32, j: u32) -> f32 {
        self.heights[(j * self.nx + i) as usize]
    }

    /// Local position of vertex `(i, j)`.
    pub fn vertex(&self, i: u32, j: u32) -> Vec3 {
        let (x0, z0) = origin(self.nx, self.nz, self.dx, self.dz);
        Vec3::new(
            x0 + i as f32 * self.dx,
            self.height(i, j),
            z0 + j as f32 * self.dz,
        )
    }

    /// Smallest and largest height.
    pub fn height_range(&self) -> (f32, f32) {
        self.heights
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &h| {
                (lo.min(h), hi.max(h))
            })
    }

    /// Local bounds `(min, max)`.
    pub fn local_bounds(&self) -> (Vec3, Vec3) {
        let (x0, z0) = origin(self.nx, self.nz, self.dx, self.dz);
        let (lo, hi) = self.height_range();
        (Vec3::new(x0, lo, z0), Vec3::new(-x0, hi, -z0))
    }

    /// Farthest point of the field's bounds from its origin.
    pub fn bounding_radius(&self) -> f32 {
        let (mn, mx) = self.local_bounds();
        mn.abs().max(mx.abs()).length()
    }

    /// Surface height and outward normal under local `(x, z)`, as the GPU
    /// evaluates it (the triangle containing the point). `None` off the field.
    pub fn surface(&self, x: f32, z: f32) -> Option<(f32, Vec3)> {
        let (x0, z0) = origin(self.nx, self.nz, self.dx, self.dz);
        let fx = (x - x0) / self.dx;
        let fz = (z - z0) / self.dz;
        let (last_x, last_z) = ((self.nx - 1) as f32, (self.nz - 1) as f32);
        if !(0.0..=last_x).contains(&fx) || !(0.0..=last_z).contains(&fz) {
            return None;
        }
        let i = (fx as u32).min(self.nx - 2);
        let j = (fz as u32).min(self.nz - 2);
        let (u, v) = (fx - i as f32, fz - j as f32);
        let h00 = self.height(i, j);
        let h10 = self.height(i + 1, j);
        let h01 = self.height(i, j + 1);
        let h11 = self.height(i + 1, j + 1);
        let (sx, sz) = if u >= v {
            (h10 - h00, h11 - h10)
        } else {
            (h11 - h01, h01 - h00)
        };
        let h = h00 + u * sx + v * sz;
        Some((h, Vec3::new(-sx / self.dx, 1.0, -sz / self.dz).normalize()))
    }

    /// Overwrite the samples in `[i0, i0+w) × [j0, j0+d)` (row-major `values`).
    pub fn set_region(
        &mut self,
        i0: u32,
        j0: u32,
        w: u32,
        d: u32,
        values: &[f32],
    ) -> Result<(), HeightFieldError> {
        let (i1, j1) = (i0.saturating_add(w), j0.saturating_add(d));
        if i1 > self.nx || j1 > self.nz {
            return Err(HeightFieldError::EditBounds {
                i0,
                i1,
                j0,
                j1,
                nx: self.nx,
                nz: self.nz,
            });
        }
        let want = w as usize * d as usize;
        if values.len() != want {
            return Err(HeightFieldError::HeightCount {
                got: values.len(),
                nx: w,
                nz: d,
                want,
            });
        }
        if let Some(index) = values.iter().position(|h| !h.is_finite()) {
            return Err(HeightFieldError::NonFinite { index });
        }
        for jj in 0..d {
            let row = ((j0 + jj) * self.nx + i0) as usize;
            let src = (jj * w) as usize;
            self.heights[row..row + w as usize].copy_from_slice(&values[src..src + w as usize]);
        }
        Ok(())
    }

    /// Records this field occupies in the arena.
    pub fn record_count(&self) -> u32 {
        2 + (self.heights.len() as u32).div_ceil(4)
    }

    /// Packed arena records (see the module table).
    pub fn pack(&self) -> Vec<[u32; 4]> {
        let (lo, hi) = self.height_range();
        let mut out = Vec::with_capacity(self.record_count() as usize);
        out.push([HEIGHT_FIELD_MAGIC, self.nx, self.nz, 0]);
        out.push([
            self.dx.to_bits(),
            self.dz.to_bits(),
            lo.to_bits(),
            hi.to_bits(),
        ]);
        for chunk in self.heights.chunks(4) {
            let mut r = [0u32; 4];
            for (k, h) in chunk.iter().enumerate() {
                r[k] = h.to_bits();
            }
            out.push(r);
        }
        out
    }
}

fn origin(nx: u32, nz: u32, dx: f32, dz: f32) -> (f32, f32) {
    (-0.5 * (nx - 1) as f32 * dx, -0.5 * (nz - 1) as f32 * dz)
}

/// Collider `params` for a heightfield at arena `base`.
pub fn height_field_collider_params(base: u32, field: &HeightField) -> [f32; 4] {
    assert!(
        base <= HULL_MAX_BASE,
        "heightfield base exceeds the tagged range"
    );
    [
        f32::from_bits(base | HULL_PARAM_TAG),
        0.0,
        0.0,
        field.bounding_radius(),
    ]
}

impl HullArena {
    /// Place a heightfield in the arena (never shared: each field is edited on
    /// its own). Returns its base; `release` frees it.
    pub fn insert_height_field(&mut self, field: &HeightField) -> Result<u32, HeightFieldError> {
        Ok(self.insert_unique(&field.pack())?)
    }

    /// Rewrite a placed heightfield after an edit (same sample counts). Only the
    /// header and the changed rows are marked for upload.
    pub fn update_height_field(
        &mut self,
        base: u32,
        field: &HeightField,
        rows: std::ops::Range<u32>,
    ) -> Result<(), HeightFieldError> {
        let head = self
            .records()
            .get(base as usize)
            .copied()
            .ok_or(HeightFieldError::NotAHeightField(base))?;
        if head[0] != HEIGHT_FIELD_MAGIC || head[1] != field.nx || head[2] != field.nz {
            return Err(HeightFieldError::NotAHeightField(base));
        }
        let packed = field.pack();
        self.write_records(base, &packed[..2]);
        let rows = rows.start.min(field.nz)..rows.end.min(field.nz);
        if !rows.is_empty() {
            let first = (rows.start * field.nx) / 4;
            let last = (rows.end * field.nx).div_ceil(4);
            self.write_records(
                base + 2 + first,
                &packed[2 + first as usize..2 + last as usize],
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> HeightField {
        HeightField::from_fn(5, 4, 0.5, 1.0, |x, z| 0.25 * x + 0.1 * z).unwrap()
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            HeightField::new(1, 4, 1.0, 1.0, vec![0.0; 4]),
            Err(HeightFieldError::Samples { nx: 1, nz: 4 })
        );
        assert_eq!(
            HeightField::new(2, 2, 0.0, 1.0, vec![0.0; 4]),
            Err(HeightFieldError::CellSize { dx: 0.0, dz: 1.0 })
        );
        assert!(matches!(
            HeightField::new(2, 2, 1.0, 1.0, vec![0.0; 3]),
            Err(HeightFieldError::HeightCount { .. })
        ));
        assert_eq!(
            HeightField::new(2, 2, 1.0, 1.0, vec![0.0, f32::NAN, 0.0, 0.0]),
            Err(HeightFieldError::NonFinite { index: 1 })
        );
    }

    #[test]
    fn surface_is_exact_on_a_plane() {
        // A plane is reproduced exactly by either triangle of every cell.
        let f = ramp();
        for &(x, z) in &[(-0.9, -1.4), (0.3, 0.2), (0.99, 1.49), (0.0, 0.0)] {
            let (h, n) = f.surface(x, z).unwrap();
            assert!((h - (0.25 * x + 0.1 * z)).abs() < 1e-5, "h({x},{z}) = {h}");
            let want = Vec3::new(-0.25, 1.0, -0.1).normalize();
            assert!((n - want).length() < 1e-5);
        }
        assert!(f.surface(1.01, 0.0).is_none());
        assert!(f.surface(0.0, -1.6).is_none());
    }

    #[test]
    fn surface_uses_the_diagonal_split() {
        // One raised corner (1, 1) of a single cell: the (0,0)-(1,1) diagonal
        // carries it, so the midpoint of the diagonal is at half its height.
        let f = HeightField::new(2, 2, 1.0, 1.0, vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        let (h, _) = f.surface(0.0, 0.0).unwrap();
        assert!((h - 0.5).abs() < 1e-6);
        // Below the diagonal (u > v) the triangle is (00, 10, 11).
        let (h, n) = f.surface(0.25, -0.25).unwrap();
        assert!((h - 0.25).abs() < 1e-6, "{h}");
        assert!((n - Vec3::new(0.0, 1.0, -1.0).normalize()).length() < 1e-6);
    }

    #[test]
    fn pack_round_trips_through_the_arena() {
        let f = ramp();
        let mut arena = HullArena::new();
        let base = arena.insert_height_field(&f).unwrap();
        let r = &arena.records()[base as usize..];
        assert_eq!(r[0], [HEIGHT_FIELD_MAGIC, 5, 4, 0]);
        assert_eq!(f32::from_bits(r[1][0]), 0.5);
        let (lo, hi) = f.height_range();
        assert_eq!((f32::from_bits(r[1][2]), f32::from_bits(r[1][3])), (lo, hi));
        for k in 0..20usize {
            assert_eq!(f32::from_bits(r[2 + k / 4][k % 4]), f.heights()[k]);
        }
        // Not shared: a second identical field gets its own records.
        let other = arena.insert_height_field(&f).unwrap();
        assert_ne!(base, other);
        assert!(arena.release(base));
    }

    #[test]
    fn edits_upload_only_changed_rows() {
        let mut f = HeightField::new(8, 8, 1.0, 1.0, vec![0.0; 64]).unwrap();
        let mut arena = HullArena::new();
        let base = arena.insert_height_field(&f).unwrap();
        arena.take_dirty();
        f.set_region(2, 5, 3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        arena.update_height_field(base, &f, 5..7).unwrap();
        let dirty = arena.take_dirty();
        // Header (2 records) and rows 5..7 = samples 40..56 = records 10..14.
        assert_eq!(dirty, vec![base..base + 2, base + 12..base + 16]);
        assert_eq!(f.height(4, 6), 6.0);
        assert_eq!(f32::from_bits(arena.records()[(base + 1) as usize][3]), 6.0);
        assert!(matches!(
            f.set_region(6, 0, 3, 1, &[0.0; 3]),
            Err(HeightFieldError::EditBounds { .. })
        ));
    }

    #[test]
    fn collider_params_tag_the_base() {
        let f = ramp();
        let p = height_field_collider_params(17, &f);
        assert_eq!(crate::convex::convex_params_base(p), 17);
        assert!(p[0].is_normal());
        assert!((p[3] - f.bounding_radius()).abs() < 1e-6);
    }
}
