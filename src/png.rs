// 纯 std PNG 编码器 (灰度 8-bit, zlib "stored" 模式 = 不压缩)
//
// 实现:
// - PNG 签名 + IHDR + IDAT + IEND chunks
// - CRC32 (PNG polynomial 0xEDB88320)
// - ADLER32 (zlib checksum)
// - zlib "stored" 模式 (无压缩, 一次性最大 65535 字节/块)
//
// 优点: 简单可靠, 任何 PNG 解码器都能读
// 缺点: 输出体积 ≈ PPM (1.5MB/帧)
// 后续可用 deflate fixed Huffman 进一步压缩 (代码已留 LZ77 + Huffman 骨架, 但 fixed Huffman 编码复杂度高+体积大得不偿失)

use crate::image::Image;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// 写 8 字节 PNG 签名。
fn write_signature<W: Write>(w: &mut W) -> std::io::Result<()> {
    w.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
}

/// 写一个 PNG chunk = 4 byte length + 4 byte type + data + 4 byte CRC.
fn write_chunk<W: Write>(w: &mut W, chunk_type: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    let len = data.len() as u32;
    w.write_all(&len.to_be_bytes())?;
    w.write_all(chunk_type)?;
    w.write_all(data)?;
    let mut crc = Crc::new();
    crc.update(chunk_type);
    crc.update(data);
    w.write_all(&crc.finalize().to_be_bytes())?;
    Ok(())
}

/// CRC32 (PNG polynomial 0xEDB88320)
struct Crc {
    state: u32,
}

impl Crc {
    fn new() -> Self {
        Self { state: 0xFFFFFFFF }
    }
    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.state ^= b as u32;
            for _ in 0..8 {
                if self.state & 1 != 0 {
                    self.state = (self.state >> 1) ^ 0xEDB88320;
                } else {
                    self.state >>= 1;
                }
            }
        }
    }
    fn finalize(&self) -> u32 {
        !self.state
    }
}

/// ADLER32 (zlib checksum)
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Paeth 预测器 (PNG 滤镜)
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

/// 将 raw image 转为 filtered scanlines (filter type 0 = None, 简单不压缩)
fn filter_none(img: &Image) -> Vec<u8> {
    let w = img.width;
    let h = img.height;
    let stride = w + 1;
    let mut out = Vec::with_capacity(stride * h);
    for y in 0..h {
        out.push(0u8); // filter type: None
        let row_start = y * w;
        out.extend_from_slice(&img.data[row_start..row_start + w]);
    }
    out
}

/// 编码 PNG。Image 必须是灰度 8-bit (channels=1)。
pub fn write_png<P: AsRef<Path>>(path: P, img: &Image) -> Result<(), String> {
    let path = path.as_ref();
    let mut file = BufWriter::new(File::create(path).map_err(|e| format!("创建 {} 失败: {}", path.display(), e))?);
    write_png_to_writer(img, &mut file).map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// 把 PNG 写到任意 Write (用于内嵌到 HTML)
pub fn write_png_to_writer<W: Write>(img: &Image, file: &mut W) -> Result<(), String> {
    let w = img.width as u32;
    let h = img.height as u32;
    if img.channels != 1 {
        return Err(format!("PNG 编码仅支持灰度图, channels={}", img.channels));
    }

    write_signature(file).map_err(|e| e.to_string())?;

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.push(8);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    write_chunk(file, b"IHDR", &ihdr).map_err(|e| e.to_string())?;

    let filtered = filter_none(img);

    let mut zlib = Vec::with_capacity(filtered.len() + 8);
    zlib.push(0x78);
    zlib.push(0x01);
    let mut pos = 0;
    while pos < filtered.len() {
        let block_len = (filtered.len() - pos).min(65535);
        let bfinal = if pos + block_len == filtered.len() { 1 } else { 0 };
        zlib.push(bfinal);
        zlib.extend_from_slice(&(block_len as u16).to_le_bytes());
        zlib.extend_from_slice(&(!block_len as u16).to_le_bytes());
        zlib.extend_from_slice(&filtered[pos..pos + block_len]);
        pos += block_len;
    }
    zlib.extend_from_slice(&adler32(&filtered).to_be_bytes());

    write_chunk(file, b"IDAT", &zlib).map_err(|e| e.to_string())?;
    write_chunk(file, b"IEND", &[]).map_err(|e| e.to_string())?;
    let _ = paeth;
    Ok(())
}
