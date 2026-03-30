use rubble_shapes2d::Aabb2D;

pub fn overlaps(aabbs: &[Aabb2D]) -> Vec<[u32; 2]> {
    let mut out = Vec::new();
    for i in 0..aabbs.len() {
        for j in (i + 1)..aabbs.len() {
            let a = aabbs[i];
            let b = aabbs[j];
            if a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
            {
                out.push([i as u32, j as u32]);
            }
        }
    }
    out
}
