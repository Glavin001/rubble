//! Stable keys of chunks, bonds and impactors, shared by every learned-impact consumer
//! (P1's recorder and call builder, the DAgger restart, P0's closed-loop harness, P2's
//! calls): one definition, so they can never diverge.
//!
//! - chunk: `(structure << 32) | chunk index within the structure`;
//! - bond: `(structure << 32) | source`, the scene's own bond index (`RtBond::source`),
//!   which survives filtering and refinement;
//! - impactor: its index in the scene's impactors.

pub fn chunk_key(structure: usize, chunk: usize) -> i64 {
    ((structure as i64) << 32) | chunk as i64
}

pub fn bond_key(structure: usize, source: usize) -> i64 {
    ((structure as i64) << 32) | source as i64
}

pub fn impactor_key(index: usize) -> i64 {
    index as i64
}

/// `(structure, index)` of a chunk or bond key.
pub fn split_key(key: i64) -> (usize, usize) {
    ((key >> 32) as usize, (key & 0xffff_ffff) as usize)
}

/// The undirected key of a contact edge between two node keys (P2's calls schema):
/// `(a * 1_000_003) ^ b` with `a <= b`.
pub fn contact_key(a: i64, b: i64) -> i64 {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    a.wrapping_mul(1_000_003) ^ b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip() {
        assert_eq!(split_key(chunk_key(3, 17)), (3, 17));
        assert_eq!(split_key(bond_key(2, 4_000_000)), (2, 4_000_000));
        assert_eq!(contact_key(5, 9), contact_key(9, 5));
        assert_eq!(chunk_key(1, 0), 1 << 32);
    }
}
