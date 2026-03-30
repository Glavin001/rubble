use rubble_math::Aabb3D;

pub fn overlap_pairs(aabbs: &[Aabb3D]) -> Vec<[u32; 2]> {
    let mut pairs = Vec::new();
    for i in 0..aabbs.len() {
        for j in (i + 1)..aabbs.len() {
            if intersects(&aabbs[i], &aabbs[j]) {
                pairs.push([i as u32, j as u32]);
            }
        }
    }
    pairs
}

pub fn intersects(a: &Aabb3D, b: &Aabb3D) -> bool {
    let amin = a.min.truncate();
    let amax = a.max.truncate();
    let bmin = b.min.truncate();
    let bmax = b.max.truncate();
    amin.x <= bmax.x
        && amax.x >= bmin.x
        && amin.y <= bmax.y
        && amax.y >= bmin.y
        && amin.z <= bmax.z
        && amax.z >= bmin.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn overlaps_expected_pairs() {
        let a = Aabb3D {
            min: Vec4::new(0.0, 0.0, 0.0, 0.0),
            max: Vec4::new(1.0, 1.0, 1.0, 0.0),
        };
        let b = Aabb3D {
            min: Vec4::new(0.5, 0.0, 0.0, 0.0),
            max: Vec4::new(1.5, 1.0, 1.0, 0.0),
        };
        let c = Aabb3D {
            min: Vec4::new(5.0, 5.0, 5.0, 0.0),
            max: Vec4::new(6.0, 6.0, 6.0, 0.0),
        };
        let pairs = overlap_pairs(&[a, b, c]);
        assert_eq!(pairs, vec![[0, 1]]);
    }
}
