//! Connected parts of a body: components of its graph of unbroken bonds.

/// Per node, the index of its component (components numbered from 0), and the count.
pub fn components(nodes: usize, links: impl Iterator<Item = (u32, u32)>) -> (Vec<usize>, usize) {
    let mut parent: Vec<usize> = (0..nodes).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for (a, b) in links {
        let (a, b) = (a as usize, b as usize);
        if a >= nodes || b >= nodes {
            continue;
        }
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let mut id = vec![usize::MAX; nodes];
    let mut label = vec![0usize; nodes];
    let mut count = 0;
    for i in 0..nodes {
        let r = root(&mut parent, i);
        if id[r] == usize::MAX {
            id[r] = count;
            count += 1;
        }
        label[i] = id[r];
    }
    (label, count)
}

/// Mass (kg) outside the heaviest component.
pub fn detached_mass(label: &[usize], count: usize, mass: &[f32]) -> f64 {
    let mut m = vec![0.0f64; count];
    for (l, w) in label.iter().zip(mass) {
        m[*l] += f64::from(*w);
    }
    let total: f64 = m.iter().sum();
    total - m.iter().copied().fold(0.0, f64::max)
}
