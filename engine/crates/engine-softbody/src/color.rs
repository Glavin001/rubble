//! Spawn-time graph colouring of a vehicle's beam network. A vehicle's beam graph
//! is static between breaks (breaks only remove edges, which keeps a colouring
//! valid), so it is coloured once instead of every step. Largest-degree-first
//! greedy keeps the colour count close to the max degree + 1.

/// Colour of each node so that no beam joins two nodes of one colour.
/// `edges` are node index pairs; returns (colours, colour count).
pub fn color_graph(nodes: usize, edges: &[(u32, u32)]) -> (Vec<u32>, u32) {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); nodes];
    for &(a, b) in edges {
        if a == b {
            continue;
        }
        adj[a as usize].push(b);
        adj[b as usize].push(a);
    }
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
    }
    let mut order: Vec<usize> = (0..nodes).collect();
    // Largest degree first, ties by index (deterministic).
    order.sort_by(|&x, &y| adj[y].len().cmp(&adj[x].len()).then(x.cmp(&y)));
    let mut color = vec![u32::MAX; nodes];
    let mut used = Vec::new();
    let mut count = 0u32;
    for &v in &order {
        used.clear();
        for &n in &adj[v] {
            let c = color[n as usize];
            if c != u32::MAX {
                used.push(c);
            }
        }
        used.sort_unstable();
        used.dedup();
        let mut c = 0u32;
        for &u in &used {
            if u == c {
                c += 1;
            } else if u > c {
                break;
            }
        }
        color[v] = c;
        count = count.max(c + 1);
    }
    (color, count)
}

/// True when no edge joins two nodes of one colour.
pub fn is_valid(colors: &[u32], edges: &[(u32, u32)]) -> bool {
    edges.iter().all(|&(a, b)| a == b || colors[a as usize] != colors[b as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_a_braced_lattice() {
        // 4×4×4 lattice with face diagonals (degree up to ~18).
        let n = 4usize;
        let id = |x: usize, y: usize, z: usize| (x + n * (y + n * z)) as u32;
        let mut edges = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)] {
                        if x + dx < n && y + dy < n && z + dz < n {
                            edges.push((id(x, y, z), id(x + dx, y + dy, z + dz)));
                        }
                    }
                }
            }
        }
        let (c, k) = color_graph(n * n * n, &edges);
        assert!(is_valid(&c, &edges));
        assert!(k <= 19, "{k} colours");
    }
}
