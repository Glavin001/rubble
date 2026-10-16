//! Minimal PNG writer (8-bit RGB): per-row adaptive filtering and a small deflate
//! encoder (greedy LZ77 with hash chains, fixed Huffman codes). Rendered frames are
//! flat-coloured, so this compresses them well without an external crate.

/// Encodes an RGB24 image as PNG.
pub fn encode(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * 3, "RGB24 buffer size");
    let stride = width * 3;
    let mut raw = Vec::with_capacity((stride + 1) * height);
    let zero = vec![0u8; stride];
    let mut candidates = [vec![0u8; stride], vec![0u8; stride], vec![0u8; stride]];
    for y in 0..height {
        let row = &rgb[y * stride..(y + 1) * stride];
        let up = if y > 0 { &rgb[(y - 1) * stride..y * stride] } else { &zero[..] };
        // Filters 0 (none), 1 (sub), 2 (up); keep the one with the smallest signed sum.
        for i in 0..stride {
            let left = if i >= 3 { row[i - 3] } else { 0 };
            candidates[0][i] = row[i];
            candidates[1][i] = row[i].wrapping_sub(left);
            candidates[2][i] = row[i].wrapping_sub(up[i]);
        }
        let cost = |v: &[u8]| v.iter().map(|&b| (b as i8).unsigned_abs() as u64).sum::<u64>();
        let best = (0..3).min_by_key(|&f| cost(&candidates[f])).unwrap();
        raw.push(best as u8);
        raw.extend_from_slice(&candidates[best]);
    }
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, truecolour, deflate, adaptive, no interlace
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// CRC-32 (IEEE, reflected), as PNG chunks use.
pub fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, t) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    !data.iter().fold(!0u32, |c, &b| table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8))
}

/// Adler-32, the zlib stream checksum.
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for block in data.chunks(5552) {
        for &x in block {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// LSB-first bit writer.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn put(&mut self, value: u32, count: u32) {
        self.acc |= (value as u64) << self.n;
        self.n += count;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code, which deflate stores most-significant bit first.
    fn code(&mut self, code: u32, len: u32) {
        self.put(code.reverse_bits() >> (32 - len), len);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] =
    [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Literal/length symbol with the fixed Huffman code.
fn symbol(bits: &mut Bits, s: u32) {
    match s {
        0..=143 => bits.code(0x30 + s, 8),
        144..=255 => bits.code(0x190 + s - 144, 9),
        256..=279 => bits.code(s - 256, 7),
        _ => bits.code(0xc0 + s - 280, 8),
    }
}

fn emit_match(bits: &mut Bits, len: usize, dist: usize) {
    let li = LEN_BASE.partition_point(|&b| b as usize <= len) - 1;
    symbol(bits, 257 + li as u32);
    bits.put((len - LEN_BASE[li] as usize) as u32, LEN_EXTRA[li] as u32);
    let di = DIST_BASE.partition_point(|&b| b as usize <= dist) - 1;
    bits.code(di as u32, 5);
    bits.put((dist - DIST_BASE[di] as usize) as u32, DIST_EXTRA[di] as u32);
}

const WINDOW: usize = 32768;
const MAX_CHAIN: usize = 48;

/// zlib stream of one fixed-Huffman deflate block.
fn zlib(data: &[u8]) -> Vec<u8> {
    let mut bits = Bits { out: vec![0x78, 0x01], acc: 0, n: 0 };
    bits.put(1, 1); // final block
    bits.put(1, 2); // fixed Huffman
    let hash = |i: usize| ((data[i] as usize) << 10 ^ (data[i + 1] as usize) << 5 ^ data[i + 2] as usize) & 0x7fff;
    let mut head = vec![usize::MAX; 1 << 15];
    let mut prev = vec![usize::MAX; WINDOW];
    let insert = |head: &mut Vec<usize>, prev: &mut Vec<usize>, i: usize| {
        if i + 3 <= data.len() {
            let h = hash(i);
            prev[i % WINDOW] = head[h];
            head[h] = i;
        }
    };
    let mut i = 0;
    while i < data.len() {
        let (mut best_len, mut best_dist) = (0, 0);
        if i + 3 <= data.len() {
            let max_len = (data.len() - i).min(258);
            let mut cand = head[hash(i)];
            let mut chain = 0;
            while cand != usize::MAX && i - cand <= WINDOW && chain < MAX_CHAIN {
                let l = data[cand..cand + max_len].iter().zip(&data[i..i + max_len]).take_while(|(a, b)| a == b).count();
                if l > best_len {
                    best_len = l;
                    best_dist = i - cand;
                    if l == max_len {
                        break;
                    }
                }
                let next = prev[cand % WINDOW];
                if next == usize::MAX || next >= cand {
                    break;
                }
                cand = next;
                chain += 1;
            }
        }
        if best_len >= 3 {
            emit_match(&mut bits, best_len, best_dist);
            for k in i..i + best_len {
                insert(&mut head, &mut prev, k);
            }
            i += best_len;
        } else {
            symbol(&mut bits, data[i] as u32);
            insert(&mut head, &mut prev, i);
            i += 1;
        }
    }
    symbol(&mut bits, 256);
    let mut out = bits.finish();
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksums_match_reference_values() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    #[test]
    fn png_layout_and_compression() {
        let (w, h) = (64, 32);
        let img: Vec<u8> = (0..w * h).flat_map(|i| if (i % w) < 32 { [200, 30, 30] } else { [255, 255, 255] }).collect();
        let png = encode(w, h, &img);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), w as u32);
        assert!(png.ends_with(&[0xae, 0x42, 0x60, 0x82]), "IEND crc");
        assert!(png.len() < img.len() / 10, "flat image compresses: {} bytes", png.len());
    }

    #[test]
    fn length_and_distance_tables_cover_their_ranges() {
        // Every match length 3..=258 and distance 1..=32768 has a code.
        let mut bits = Bits { out: Vec::new(), acc: 0, n: 0 };
        for len in [3, 10, 11, 257, 258] {
            for dist in [1, 4, 5, 24577, 32768] {
                emit_match(&mut bits, len, dist);
            }
        }
        assert!(!bits.finish().is_empty());
    }
}
