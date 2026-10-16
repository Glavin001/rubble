//! Backend-neutral world contents: one scene built in Rust can be replayed on
//! Metal, wgpu, or the browser (raw little-endian GPU records, no re-encoding).

use crate::{AvbdParams, BodyState, Collider, Joint, Material};

#[derive(Clone, Debug, Default)]
pub struct WorldSnapshot {
    pub bodies: Vec<BodyState>,
    pub colliders: Vec<Collider>,
    /// The 64-entry material table.
    pub materials: Vec<Material>,
    pub joints: Vec<Joint>,
    pub params: Option<AvbdParams>,
}

const MAGIC: &[u8; 8] = b"RUBSNAP1";

impl WorldSnapshot {
    /// `MAGIC, [bodies, colliders, materials, joints] as u32 counts, then each
    /// array's raw records`. The browser reads it with typed-array views.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        for n in [
            self.bodies.len(),
            self.colliders.len(),
            self.materials.len(),
            self.joints.len(),
        ] {
            out.extend_from_slice(&(n as u32).to_le_bytes());
        }
        out.extend_from_slice(bytemuck::cast_slice(&self.bodies));
        out.extend_from_slice(bytemuck::cast_slice(&self.colliders));
        out.extend_from_slice(bytemuck::cast_slice(&self.materials));
        out.extend_from_slice(bytemuck::cast_slice(&self.joints));
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 || &bytes[..8] != MAGIC {
            return None;
        }
        let count = |i: usize| {
            u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize
        };
        let mut off = 24;
        fn take<T: bytemuck::Pod>(bytes: &[u8], off: &mut usize, n: usize) -> Option<Vec<T>> {
            let len = n * std::mem::size_of::<T>();
            let s = bytes.get(*off..*off + len)?;
            *off += len;
            let mut v = vec![T::zeroed(); n];
            bytemuck::cast_slice_mut::<T, u8>(&mut v).copy_from_slice(s);
            Some(v)
        }
        Some(Self {
            bodies: take(bytes, &mut off, count(0))?,
            colliders: take(bytes, &mut off, count(1))?,
            materials: take(bytes, &mut off, count(2))?,
            joints: take(bytes, &mut off, count(3))?,
            params: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn round_trip() {
        let s = WorldSnapshot {
            bodies: vec![BodyState::zeroed(); 3],
            colliders: vec![Collider::zeroed(); 2],
            materials: vec![Material::zeroed(); 64],
            joints: vec![Joint::zeroed(); 5],
            params: None,
        };
        let back = WorldSnapshot::from_bytes(&s.to_bytes()).unwrap();
        assert_eq!(back.bodies.len(), 3);
        assert_eq!(back.joints.len(), 5);
        assert_eq!(back.materials.len(), 64);
    }
}
