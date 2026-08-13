// PGM (P5) / PPM (P6) 图像格式 I/O, 纯 std, 无压缩, 零依赖。
//
// 格式 (二进制):
//   P5\n<width> <height>\n<maxval>\n<raw 字节>
//   P6\n<width> <height>\n<maxval>\n<RGBRGBRGB...>
//
// 注释以 '#' 开头, 通常在 header 中。

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// 8 位灰度图像。
#[derive(Debug, Clone)]
pub struct GrayImage {
    pub w: u32,
    pub h: u32,
    /// 行优先, 长度 = w * h
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            data: vec![0; (w as usize) * (h as usize)],
        }
    }

    pub fn pixel(&self, x: u32, y: u32) -> u8 {
        self.data[(y * self.w + x) as usize]
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, v: u8) {
        self.data[(y * self.w + x) as usize] = v;
    }

    /// 缩放 (双线性)。
    pub fn resize(&self, new_w: u32, new_h: u32) -> Self {
        if new_w == self.w && new_h == self.h {
            return self.clone();
        }
        let mut out = Self::new(new_w, new_h);
        let x_ratio = self.w as f32 / new_w as f32;
        let y_ratio = self.h as f32 / new_h as f32;
        for y in 0..new_h {
            let sy = (y as f32 * y_ratio).min(self.h as f32 - 1.0);
            let y0 = sy.floor() as u32;
            let y1 = (y0 + 1).min(self.h - 1);
            let dy = sy - y0 as f32;
            for x in 0..new_w {
                let sx = (x as f32 * x_ratio).min(self.w as f32 - 1.0);
                let x0 = sx.floor() as u32;
                let x1 = (x0 + 1).min(self.w - 1);
                let dx = sx - x0 as f32;
                let p00 = self.pixel(x0, y0) as f32;
                let p01 = self.pixel(x1, y0) as f32;
                let p10 = self.pixel(x0, y1) as f32;
                let p11 = self.pixel(x1, y1) as f32;
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = top * (1.0 - dy) + bot * dy;
                out.set_pixel(x, y, v.round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    /// 把矩形区域裁剪出来, 夹紧到边界。
    pub fn crop(&self, x: i32, y: i32, w: i32, h: i32) -> Self {
        let x0 = x.max(0) as u32;
        let y0 = y.max(0) as u32;
        let x1 = (x + w).max(0).min(self.w as i32) as u32;
        let y1 = (y + h).max(0).min(self.h as i32) as u32;
        let cw = x1.saturating_sub(x0);
        let ch = y1.saturating_sub(y0);
        let mut out = Self::new(cw, ch);
        for j in 0..ch {
            for i in 0..cw {
                out.set_pixel(i, j, self.pixel(x0 + i, y0 + j));
            }
        }
        out
    }

    /// 直方图均衡化(可选预处理)。
    pub fn equalize(&mut self) {
        let mut hist = [0u32; 256];
        for &v in &self.data {
            hist[v as usize] += 1;
        }
        let mut cdf = [0u32; 256];
        let mut acc = 0u32;
        for i in 0..256 {
            acc += hist[i];
            cdf[i] = acc;
        }
        let total = cdf[255].max(1);
        let mut lut = [0u8; 256];
        for i in 0..256 {
            lut[i] = ((cdf[i] as f32 / total as f32) * 255.0).round() as u8;
        }
        for v in self.data.iter_mut() {
            *v = lut[*v as usize];
        }
    }
}

/// 从 PGM 文件读取 (P5 magic), 支持带注释的 header。
pub fn read_pgm<P: AsRef<Path>>(path: P) -> Result<GrayImage, String> {
    let path = path.as_ref();
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("打开 {} 失败: {}", path.display(), e))?
        .read_to_end(&mut bytes)
        .map_err(|e| format!("读取 {} 失败: {}", path.display(), e))?;
    parse_pgm_bytes(&bytes)
}

/// 解析已读入内存的 PGM (P5) 字节。
pub fn parse_pgm_bytes(bytes: &[u8]) -> Result<GrayImage, String> {
    let mut i = 0usize;
    let skip_ws = |bytes: &[u8], i: &mut usize| {
        while *i < bytes.len() {
            let c = bytes[*i];
            if c == b'#' {
                // 跳到行尾
                while *i < bytes.len() && bytes[*i] != b'\n' { *i += 1; }
            } else if c.is_ascii_whitespace() {
                *i += 1;
            } else {
                break;
            }
        }
    };
    let read_token = |bytes: &[u8], i: &mut usize| -> Result<String, String> {
        skip_ws(bytes, i);
        if *i >= bytes.len() { return Err("PGM 提前结束".into()); }
        let start = *i;
        while *i < bytes.len() && !bytes[*i].is_ascii_whitespace() { *i += 1; }
        Ok(String::from_utf8(bytes[start..*i].to_vec()).map_err(|e| format!("非 UTF-8 token: {}", e))?)
    };
    let magic = read_token(bytes, &mut i)?;
    if magic != "P5" {
        return Err(format!("不是 PGM (P5), magic={}", magic));
    }
    let w: u32 = read_token(bytes, &mut i)?
        .parse()
        .map_err(|e| format!("PGM 宽度解析失败: {}", e))?;
    let h: u32 = read_token(bytes, &mut i)?
        .parse()
        .map_err(|e| format!("PGM 高度解析失败: {}", e))?;
    let maxv: u32 = read_token(bytes, &mut i)?
        .parse()
        .map_err(|e| format!("PGM maxval 解析失败: {}", e))?;
    if maxv > 255 {
        return Err(format!("仅支持 8 位 PGM, maxval={}", maxv));
    }
    // 跳过 maxval 后的一个空白字符
    if i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }
    let n = (w as usize) * (h as usize);
    if bytes.len() - i < n {
        return Err(format!(
            "PGM 数据不足: 需 {} 字节, 剩 {} 字节",
            n, bytes.len() - i
        ));
    }
    let data = bytes[i..i + n].to_vec();
    Ok(GrayImage { w, h, data })
}

/// 把 PGM 写到文件。
pub fn write_pgm<P: AsRef<Path>>(path: P, img: &GrayImage) -> Result<(), String> {
    let path = path.as_ref();
    let f = File::create(path).map_err(|e| format!("创建 {} 失败: {}", path.display(), e))?;
    let mut w = BufWriter::new(f);
    write!(w, "P5\n{} {}\n255\n", img.w, img.h).map_err(|e| e.to_string())?;
    w.write_all(&img.data).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// 从 PPM 文件读取 (P6 magic), 返回灰度 GrayImage。
pub fn read_ppm<P: AsRef<Path>>(path: P) -> Result<GrayImage, String> {
    let path = path.as_ref();
    let f = File::open(path).map_err(|e| format!("打开 {} 失败: {}", path.display(), e))?;
    let mut r = BufReader::new(f);
    let magic = read_token(&mut r)?;
    if magic != "P6" {
        return Err(format!("{} 不是 PPM (P6), magic={}", path.display(), magic));
    }
    let w: u32 = read_token(&mut r)?
        .parse()
        .map_err(|e| format!("PPM 宽度解析失败: {}", e))?;
    let h: u32 = read_token(&mut r)?
        .parse()
        .map_err(|e| format!("PPM 高度解析失败: {}", e))?;
    let _maxv: u32 = read_token(&mut r)?
        .parse()
        .map_err(|e| format!("PPM maxval 解析失败: {}", e))?;
    let mut b = [0u8; 1];
    r.read_exact(&mut b)
        .map_err(|e| format!("PPM header 后缺字节: {}", e))?;
    let n = (w as usize) * (h as usize) * 3;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf)
        .map_err(|e| format!("PPM 像素读取失败: {}", e))?;
    // 转灰度 (BT.601)
    let mut gray = vec![0u8; w as usize * h as usize];
    for i in 0..(w as usize * h as usize) {
        let r = buf[i * 3] as u32;
        let g = buf[i * 3 + 1] as u32;
        let b = buf[i * 3 + 2] as u32;
        let y = (299 * r + 587 * g + 114 * b) / 1000;
        gray[i] = y.min(255) as u8;
    }
    Ok(GrayImage { w, h, data: gray })
}

/// 把 3 通道 RGB 数据写成 PPM P6 (无压缩)。
pub fn write_ppm_rgb<P: AsRef<Path>>(
    path: P,
    w: u32,
    h: u32,
    rgb: &[u8],
) -> Result<(), String> {
    let path = path.as_ref();
    assert_eq!(rgb.len(), (w as usize) * (h as usize) * 3);
    let f = File::create(path).map_err(|e| format!("创建 {} 失败: {}", path.display(), e))?;
    let mut wri = BufWriter::new(f);
    write!(wri, "P6\n{} {}\n255\n", w, h).map_err(|e| e.to_string())?;
    wri.write_all(rgb).map_err(|e| e.to_string())?;
    wri.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// 读取一个空白分隔的 token, 跳过注释; 不消费尾部分隔符。
fn stream_pos<R: Seek>(r: &mut R) -> u64 {
    r.seek(SeekFrom::Current(0)).unwrap_or(0)
}

fn read_token<R: Read>(r: &mut R) -> Result<String, String> {
    let mut buf = Vec::with_capacity(32);
    let mut byte = [0u8; 1];
    // 先跳过前导空白 / 注释
    loop {
        if r.read(&mut byte).map_err(|e| e.to_string())? == 0 {
            return Err("PGM/PPM 提前结束".into());
        }
        let c = byte[0];
        if c == b'#' {
            // 跳过整行注释
            loop {
                if r.read(&mut byte).map_err(|e| e.to_string())? == 0 {
                    return Err("PGM/PPM 注释中提前结束".into());
                }
                if byte[0] == b'\n' {
                    break;
                }
            }
            continue;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        // 非空白, 开始累积 token
        buf.push(c);
        break;
    }
    // 累积直到下一个空白或 EOF
    while r.read(&mut byte).map_err(|e| e.to_string())? != 0 {
        let c = byte[0];
        if c == b'#' {
            loop {
                if r.read(&mut byte).map_err(|e| e.to_string())? == 0 {
                    break;
                }
                if byte[0] == b'\n' {
                    break;
                }
            }
            break;
        }
        if c.is_ascii_whitespace() {
            // 不消费这个分隔符 — 调用方用 read_exact(&mut [0u8;1]) 自己吃掉它
            break;
        }
        buf.push(c);
    }
    if buf.is_empty() {
        return Err("PGM/PPM 提前结束".into());
    }
    Ok(String::from_utf8(buf).map_err(|e| format!("非 UTF-8 token: {}", e))?)
}
