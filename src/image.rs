// 公共类型: 错误别名、矩形、图像。

use std::path::Path;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }
    pub fn right(&self) -> i32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> i32 {
        self.y + self.h
    }
    pub fn area(&self) -> i32 {
        self.w.max(0) * self.h.max(0)
    }
}

/// 简单图像容器: 行优先, 1 通道 (灰度) 或 3 通道 (RGB)。
#[derive(Debug, Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

impl Image {
    pub fn from_grayscale(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, channels: 1, data }
    }

    pub fn from_rgb(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 3);
        Self { width, height, channels: 3, data }
    }

    /// 从 PGM (P5) 加载, 转灰度 Image。
    pub fn load_pgm(path: &Path) -> Result<Self, BoxError> {
        let img = crate::ppm::read_pgm(path)?;
        let w = img.w as usize;
        let h = img.h as usize;
        Ok(Self::from_grayscale(w, h, img.data))
    }

    /// 从 PPM (P6) 加载, 保留 RGB 3 通道。
    pub fn load_ppm(path: &Path) -> Result<Self, BoxError> {
        let f = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::new(f);
        let mut header = String::new();
        use std::io::BufRead;
        r.read_line(&mut header)?;
        if header.trim() != "P6" {
            return Err("not a P6 PPM".into());
        }
        // 跳注释
        let mut line = String::new();
        loop {
            line.clear();
            let n = r.read_line(&mut line)?;
            if n == 0 {
                return Err("unexpected EOF in PPM header".into());
            }
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            break;
        }
        let mut dims = line.trim().split_whitespace();
        let w: usize = dims.next().ok_or("bad width")?.parse()?;
        let h: usize = dims.next().ok_or("bad height")?.parse()?;
        let mut line = String::new();
        loop {
            line.clear();
            let n = r.read_line(&mut line)?;
            if n == 0 {
                return Err("unexpected EOF in PPM maxval".into());
            }
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            break;
        }
        let maxv: u32 = line.trim().parse()?;
        if maxv > 255 {
            return Err("only 8-bit PPM supported".into());
        }
        // 吃掉一个空白字符
        let mut b = [0u8; 1];
        std::io::Read::read_exact(&mut r, &mut b)?;
        let mut buf = vec![0u8; w * h * 3];
        std::io::Read::read_exact(&mut r, &mut buf)?;
        Ok(Self::from_rgb(w, h, buf))
    }

    /// 转灰度 (复制为 1 通道)。
    pub fn to_grayscale(&self) -> Vec<u8> {
        if self.channels == 1 {
            return self.data.clone();
        }
        let mut out = vec![0u8; self.width * self.height];
        for i in 0..(self.width * self.height) {
            let r = self.data[i * 3] as u32;
            let g = self.data[i * 3 + 1] as u32;
            let b = self.data[i * 3 + 2] as u32;
            out[i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
        out
    }

    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self {
        let x0 = x.min(self.width);
        let y0 = y.min(self.height);
        let x1 = (x + w).min(self.width);
        let y1 = (y + h).min(self.height);
        let cw = x1 - x0;
        let ch = y1 - y0;
        let mut out = vec![0u8; cw * ch * self.channels];
        for j in 0..ch {
            for i in 0..cw {
                let src = ((y0 + j) * self.width + (x0 + i)) * self.channels;
                let dst = (j * cw + i) * self.channels;
                for c in 0..self.channels {
                    out[dst + c] = self.data[src + c];
                }
            }
        }
        Self {
            width: cw,
            height: ch,
            channels: self.channels,
            data: out,
        }
    }

    pub fn resize_bilinear(&self, new_w: usize, new_h: usize) -> Self {
        if new_w == self.width && new_h == self.height {
            return self.clone();
        }
        let mut out = vec![0u8; new_w * new_h * self.channels];
        let x_ratio = self.width as f64 / new_w as f64;
        let y_ratio = self.height as f64 / new_h as f64;
        for y in 0..new_h {
            let sy = (y as f64 * y_ratio).min((self.height as f64) - 1.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let dy = sy - y0 as f64;
            for x in 0..new_w {
                let sx = (x as f64 * x_ratio).min((self.width as f64) - 1.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let dx = sx - x0 as f64;
                for c in 0..self.channels {
                    let p00 = self.data[(y0 * self.width + x0) * self.channels + c] as f64;
                    let p01 = self.data[(y0 * self.width + x1) * self.channels + c] as f64;
                    let p10 = self.data[(y1 * self.width + x0) * self.channels + c] as f64;
                    let p11 = self.data[(y1 * self.width + x1) * self.channels + c] as f64;
                    let top = p00 * (1.0 - dx) + p01 * dx;
                    let bot = p10 * (1.0 - dx) + p11 * dx;
                    let v = top * (1.0 - dy) + bot * dy;
                    out[(y * new_w + x) * self.channels + c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Self {
            width: new_w,
            height: new_h,
            channels: self.channels,
            data: out,
        }
    }

    /// 保存为 PPM P6 (3 通道) 或 PGM P5 (1 通道)。
    pub fn save_ppm<P: AsRef<Path>>(&self, path: P) -> Result<(), BoxError> {
        let path = path.as_ref();
        if self.channels == 1 {
            let g = crate::ppm::GrayImage {
                w: self.width as u32,
                h: self.height as u32,
                data: self.data.clone(),
            };
            crate::ppm::write_pgm(path, &g)?;
        } else {
            let rgb = if self.channels == 3 {
                self.data.clone()
            } else {
                // channels >= 4 -> 取前 3
                let mut buf = Vec::with_capacity(self.width * self.height * 3);
                for px in 0..(self.width * self.height) {
                    buf.push(self.data[px * self.channels]);
                    buf.push(self.data[px * self.channels + 1]);
                    buf.push(self.data[px * self.channels + 2]);
                }
                buf
            };
            crate::ppm::write_ppm_rgb(path, self.width as u32, self.height as u32, &rgb)?;
        }
        Ok(())
    }

    /// 兼容 saver.rs 中调用的 save_png, 内部用纯 std PNG 编码器 (zlib store 模式)。
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> Result<(), BoxError> {
        let path = path.as_ref();
        crate::png::write_png(path, self)?;
        Ok(())
    }
}
