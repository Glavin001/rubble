//! Textures of imported vehicles (a BeamNG `*.materials.json`'s colour maps):
//! decoded once here and handed to the renderer as one square RGBA8 layer
//! with its mip chain, so every texture of every imported car shares one
//! texture array.
//!
//! PNG (the `png` crate: 8 or 16 bits, grey, grey-alpha, RGB, RGBA, palette)
//! and DDS (DirectDraw Surface, Microsoft's documented container: BC1, BC2,
//! BC3 blocks per the block-compression formats' public definitions, and
//! uncompressed 32-bit RGBA/BGRA) are read. A layer is the image resampled to
//! `size` × `size` (box-filtered down, bilinear up), mips by 2 × 2 boxes.

/// A decoded image: width, height, RGBA8 rows top first.
#[derive(Clone, Debug)]
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

/// Decode a PNG or DDS file by its magic.
pub fn decode(bytes: &[u8]) -> Result<Image, String> {
    if bytes.starts_with(b"\x89PNG") {
        decode_png(bytes)
    } else if bytes.starts_with(b"DDS ") {
        decode_dds(bytes)
    } else {
        Err("not a PNG or DDS file".into())
    }
}

fn decode_png(bytes: &[u8]) -> Result<Image, String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; r.output_buffer_size().ok_or("PNG too large")?];
    let info = r.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width, info.height);
    let px = (w * h) as usize;
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf[..px * 4].to_vec(),
        png::ColorType::Rgb => buf[..px * 3].chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..px * 2].chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        png::ColorType::Grayscale => buf[..px].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return Err("PNG palette not expanded".into()),
    };
    Ok(Image { w, h, rgba })
}

fn rgb565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 31) as u32;
    let g = ((c >> 5) & 63) as u32;
    let b = (c & 31) as u32;
    [((r * 255 + 15) / 31) as u8, ((g * 255 + 31) / 63) as u8, ((b * 255 + 15) / 31) as u8]
}

/// A BC1 colour block's 16 texels (RGBA); `alpha_mode`: BC1's 3-colour
/// mode (c0 ≤ c1) has a transparent fourth colour, BC2/BC3 always use 4.
fn bc1_colors(b: &[u8], four_always: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([b[0], b[1]]);
    let c1 = u16::from_le_bytes([b[2], b[3]]);
    let (p0, p1) = (rgb565(c0), rgb565(c1));
    let mix = |a: u32, b: u32, wa: u32, wb: u32| ((a * wa + b * wb) / (wa + wb)) as u8;
    let mut pal = [[0u8; 4]; 4];
    pal[0] = [p0[0], p0[1], p0[2], 255];
    pal[1] = [p1[0], p1[1], p1[2], 255];
    if c0 > c1 || four_always {
        for k in 0..3 {
            pal[2][k] = mix(p0[k] as u32, p1[k] as u32, 2, 1);
            pal[3][k] = mix(p0[k] as u32, p1[k] as u32, 1, 2);
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for k in 0..3 {
            pal[2][k] = mix(p0[k] as u32, p1[k] as u32, 1, 1);
        }
        pal[2][3] = 255;
        pal[3] = [0, 0, 0, 0];
    }
    let bits = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let mut out = [[0u8; 4]; 16];
    for (i, o) in out.iter_mut().enumerate() {
        *o = pal[((bits >> (2 * i)) & 3) as usize];
    }
    out
}

/// BC3's interpolated alpha block (8 bytes).
fn bc3_alpha(b: &[u8]) -> [u8; 16] {
    let (a0, a1) = (b[0] as u32, b[1] as u32);
    let mut pal = [0u32; 8];
    pal[0] = a0;
    pal[1] = a1;
    if a0 > a1 {
        for k in 1..7 {
            pal[k + 1] = ((7 - k as u32) * a0 + k as u32 * a1) / 7;
        }
    } else {
        for k in 1..5 {
            pal[k + 1] = ((5 - k as u32) * a0 + k as u32 * a1) / 5;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut bits = 0u64;
    for (k, &x) in b[2..8].iter().enumerate() {
        bits |= (x as u64) << (8 * k);
    }
    let mut out = [0u8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        *o = pal[((bits >> (3 * i)) & 7) as usize] as u8;
    }
    out
}

fn decode_dds(bytes: &[u8]) -> Result<Image, String> {
    let u32_at = |o: usize| -> Result<u32, String> {
        bytes.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| "DDS truncated".to_string())
    };
    let h = u32_at(12)?;
    let w = u32_at(16)?;
    let pf_flags = u32_at(80)?;
    let fourcc = bytes.get(84..88).ok_or("DDS truncated")?;
    let data = bytes.get(128..).ok_or("DDS truncated")?;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let block = |kind: u8, data: &[u8], rgba: &mut [u8]| -> Result<(), String> {
        let bsize = if kind == 1 { 8 } else { 16 };
        let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
        if data.len() < (bw * bh) as usize * bsize {
            return Err("DDS data truncated".into());
        }
        for by in 0..bh {
            for bx in 0..bw {
                let b = &data[((by * bw + bx) as usize) * bsize..][..bsize];
                let (col, alpha): ([[u8; 4]; 16], Option<[u8; 16]>) = match kind {
                    1 => (bc1_colors(b, false), None),
                    2 => {
                        let mut a = [0u8; 16];
                        for (i, x) in a.iter_mut().enumerate() {
                            *x = ((b[i / 2] >> (4 * (i % 2))) & 15) * 17;
                        }
                        (bc1_colors(&b[8..], true), Some(a))
                    }
                    _ => (bc1_colors(&b[8..], true), Some(bc3_alpha(&b[..8]))),
                };
                for i in 0..16 {
                    let (x, y) = (bx * 4 + (i as u32 % 4), by * 4 + (i as u32 / 4));
                    if x < w && y < h {
                        let o = ((y * w + x) * 4) as usize;
                        rgba[o..o + 4].copy_from_slice(&col[i]);
                        if let Some(a) = alpha {
                            rgba[o + 3] = a[i];
                        }
                    }
                }
            }
        }
        Ok(())
    };
    match (pf_flags & 4 != 0, fourcc) {
        (true, b"DXT1") => block(1, data, &mut rgba)?,
        (true, b"DXT2") | (true, b"DXT3") => block(2, data, &mut rgba)?,
        (true, b"DXT4") | (true, b"DXT5") => block(3, data, &mut rgba)?,
        (false, _) if u32_at(88)? == 32 => {
            // Uncompressed 32-bit: the red mask says RGBA or BGRA.
            let bgra = u32_at(92)? == 0x00ff_0000;
            let n = (w * h * 4) as usize;
            let src = data.get(..n).ok_or("DDS data truncated")?;
            for (d, s) in rgba.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                d.copy_from_slice(&if bgra { [s[2], s[1], s[0], s[3]] } else { [s[0], s[1], s[2], s[3]] });
            }
        }
        (_, f) => return Err(format!("DDS format {:?} not read", String::from_utf8_lossy(f))),
    }
    Ok(Image { w, h, rgba })
}

/// The image as one `size` × `size` layer with its mips (size, size/2, … 1),
/// concatenated, RGBA8.
pub fn layer(img: &Image, size: u32) -> Vec<u8> {
    let base = resample(img, size);
    let mut out = base.clone();
    let mut cur = base;
    let mut s = size;
    while s > 1 {
        let n = s / 2;
        let mut next = vec![0u8; (n * n * 4) as usize];
        for y in 0..n {
            for x in 0..n {
                for k in 0..4 {
                    let at = |xx: u32, yy: u32| cur[((yy * s + xx) * 4 + k) as usize] as u32;
                    let v = at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1);
                    next[((y * n + x) * 4 + k) as usize] = ((v + 2) / 4) as u8;
                }
            }
        }
        out.extend_from_slice(&next);
        cur = next;
        s = n;
    }
    out
}

/// The image at `size` × `size`: each output texel the mean of the source
/// texels under it (shrinking) or bilinear (growing).
fn resample(img: &Image, size: u32) -> Vec<u8> {
    let mut out = vec![0u8; (size * size * 4) as usize];
    let (w, h) = (img.w.max(1), img.h.max(1));
    let src = |x: u32, y: u32, k: usize| img.rgba[((y.min(h - 1) * w + x.min(w - 1)) * 4) as usize + k] as f32;
    for y in 0..size {
        for x in 0..size {
            let (x0, x1) = (x as f32 * w as f32 / size as f32, (x + 1) as f32 * w as f32 / size as f32);
            let (y0, y1) = (y as f32 * h as f32 / size as f32, (y + 1) as f32 * h as f32 / size as f32);
            for k in 0..4 {
                let v = if x1 - x0 >= 1.0 && y1 - y0 >= 1.0 {
                    let (mut acc, mut n) = (0.0f32, 0.0f32);
                    for sy in y0 as u32..(y1.ceil() as u32).min(h) {
                        for sx in x0 as u32..(x1.ceil() as u32).min(w) {
                            acc += src(sx, sy, k);
                            n += 1.0;
                        }
                    }
                    acc / n.max(1.0)
                } else {
                    let (fx, fy) = (0.5 * (x0 + x1) - 0.5, 0.5 * (y0 + y1) - 0.5);
                    let (ix, iy) = (fx.max(0.0).floor() as u32, fy.max(0.0).floor() as u32);
                    let (tx, ty) = ((fx - ix as f32).clamp(0.0, 1.0), (fy - iy as f32).clamp(0.0, 1.0));
                    let a = src(ix, iy, k) * (1.0 - tx) + src(ix + 1, iy, k) * tx;
                    let b = src(ix, iy + 1, k) * (1.0 - tx) + src(ix + 1, iy + 1, k) * tx;
                    a * (1.0 - ty) + b * ty
                };
                out[((y * size + x) * 4) as usize + k] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BC1 by its definition: c0 = pure red (0xf800) > c1 = pure blue
    /// (0x001f), indices 0, 1, 2, 3 across the first row: red, blue, ⅔ red
    /// + ⅓ blue, ⅓ red + ⅔ blue.
    #[test]
    fn bc1_block_decodes_by_its_definition() {
        let mut dds = vec![0u8; 128];
        dds[..4].copy_from_slice(b"DDS ");
        dds[12..16].copy_from_slice(&4u32.to_le_bytes());
        dds[16..20].copy_from_slice(&4u32.to_le_bytes());
        dds[80..84].copy_from_slice(&4u32.to_le_bytes());
        dds[84..88].copy_from_slice(b"DXT1");
        dds.extend_from_slice(&[0x00, 0xf8, 0x1f, 0x00, 0b1110_0100, 0, 0, 0]);
        let img = decode(&dds).unwrap();
        let px = |x: usize| &img.rgba[x * 4..x * 4 + 4];
        assert_eq!(px(0), [255, 0, 0, 255]);
        assert_eq!(px(1), [0, 0, 255, 255]);
        assert_eq!(px(2), [170, 0, 85, 255]);
        assert_eq!(px(3), [85, 0, 170, 255]);
        // Index 0 elsewhere: red.
        assert_eq!(&img.rgba[16..20], [255, 0, 0, 255]);
    }

    /// A layer's mips: each level the mean of the one above; a uniform image
    /// stays uniform at every level, and the sizes add up.
    #[test]
    fn layers_box_filter_and_keep_the_mean() {
        let img = Image { w: 3, h: 2, rgba: [10u8, 20, 30, 255].repeat(6) };
        let l = layer(&img, 8);
        assert_eq!(l.len(), (64 + 16 + 4 + 1) * 4);
        assert!(l.chunks_exact(4).all(|c| c == [10, 20, 30, 255]));
        let checker = Image { w: 2, h: 2, rgba: vec![0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255] };
        let l = layer(&checker, 2);
        assert_eq!(&l[16..20], [128, 128, 128, 255]);
    }
}
