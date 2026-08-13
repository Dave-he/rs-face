// 积分图 (Integral Image / Summed Area Table) + 图像预处理 + NMS。

use crate::image::{Image, Rect};
use crate::ppm::GrayImage;

#[derive(Debug, Clone)]
pub struct IntegralImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u64>,
    /// 平方积分图, 与 data 同步大小; 用于精确 mean_stdev。
    pub sq_data: Vec<u64>,
}

impl IntegralImage {
    /// 兼容 Image 类型入口: 内部转灰度再算积分图。
    pub fn from_image(img: &Image) -> Self {
        let gray = img.to_grayscale();
        Self::from_gray_raw(&gray, img.width as u32, img.height as u32)
    }

    pub fn from_gray(img: &GrayImage) -> Self {
        Self::from_gray_raw(&img.data, img.w, img.h)
    }

    fn from_gray_raw(pixels: &[u8], w: u32, h: u32) -> Self {
        let width = w as usize;
        let height = h as usize;
        let stride = width + 1;
        let mut data = vec![0u64; (height + 1) * stride];
        let mut sq_data = vec![0u64; (height + 1) * stride];
        for y in 0..height {
            let mut row_sum: u64 = 0;
            let mut row_sq_sum: u64 = 0;
            for x in 0..width {
                let v = pixels[y * width + x] as u64;
                row_sum += v;
                row_sq_sum += v * v;
                data[(y + 1) * stride + (x + 1)] = data[y * stride + (x + 1)] + row_sum;
                sq_data[(y + 1) * stride + (x + 1)] = sq_data[y * stride + (x + 1)] + row_sq_sum;
            }
        }
        Self {
            width: w,
            height: h,
            data,
            sq_data,
        }
    }

    #[inline]
    pub fn sum(&self, x: i32, y: i32, w: i32, h: i32) -> u64 {
        if w <= 0 || h <= 0 {
            return 0;
        }
        let stride = (self.width as usize) + 1;
        let x0 = x.max(0) as usize;
        let y0 = y.max(0) as usize;
        let x2 = (x0 + w as usize).min(self.width as usize);
        let y2 = (y0 + h as usize).min(self.height as usize);
        if x2 <= x0 || y2 <= y0 {
            return 0;
        }
        let a = data_at(&self.data, stride, x0, y0);
        let b = data_at(&self.data, stride, x2, y0);
        let c = data_at(&self.data, stride, x0, y2);
        let d = data_at(&self.data, stride, x2, y2);
        a + d - b - c
    }

    #[inline]
    fn sum_sq(&self, x: i32, y: i32, w: i32, h: i32) -> u64 {
        if w <= 0 || h <= 0 {
            return 0;
        }
        let stride = (self.width as usize) + 1;
        let x0 = x.max(0) as usize;
        let y0 = y.max(0) as usize;
        let x2 = (x0 + w as usize).min(self.width as usize);
        let y2 = (y0 + h as usize).min(self.height as usize);
        if x2 <= x0 || y2 <= y0 {
            return 0;
        }
        let a = data_at(&self.sq_data, stride, x0, y0);
        let b = data_at(&self.sq_data, stride, x2, y0);
        let c = data_at(&self.sq_data, stride, x0, y2);
        let d = data_at(&self.sq_data, stride, x2, y2);
        a + d - b - c
    }

    pub fn mean_stdev(&self, x: i32, y: i32, w: i32, h: i32) -> (f64, f64) {
        let total = self.sum(x, y, w, h) as f64;
        let area = (w.max(0) as f64) * (h.max(0) as f64);
        if area <= 0.0 {
            return (0.0, 0.0);
        }
        let mean = total / area;
        let sum_sq = self.sum_sq(x, y, w, h) as f64;
        let mean_sq = sum_sq / area;
        let var = (mean_sq - mean * mean).max(0.0);
        (mean, var.sqrt())
    }
}

#[inline]
fn data_at(data: &[u64], stride: usize, x: usize, y: usize) -> u64 {
    data[y * stride + x]
}

// -------- align.rs / recognition 用的辅助函数 --------

/// 直方图均衡化 (针对灰度图)。
pub fn histogram_equalize(gray: &[u8], w: usize, h: usize) -> Vec<u8> {
    if gray.is_empty() {
        return vec![];
    }
    let mut hist = [0u32; 256];
    for &v in gray {
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
        lut[i] = ((cdf[i] as f64 / total as f64) * 255.0).round() as u8;
    }
    let mut out = vec![0u8; w * h];
    for (i, &v) in gray.iter().enumerate() {
        out[i] = lut[v as usize];
    }
    out
}

/// 归一化: 减均值, 除以标准差, 缩放到 [-1, 1]。给识别前处理用。
pub fn normalize_face(gray: &[u8], w: usize, h: usize) -> Vec<f64> {
    if gray.is_empty() {
        return vec![];
    }
    let n = (w * h) as f64;
    let sum: f64 = gray.iter().map(|&v| v as f64).sum();
    let mean = sum / n;
    let var: f64 = gray.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1.0);
    gray.iter().map(|&v| (v as f64 - mean) / std).collect()
}

// -------- NMS 矩形分组 --------

/// 计算两个矩形的 IoU (交并比)。
pub fn iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let iw = (x2 - x1).max(0) as i32;
    let ih = (y2 - y1).max(0) as i32;
    let inter = (iw * ih) as f32;
    let area_a = (a.w * a.h) as f32;
    let area_b = (b.w * b.h) as f32;
    let uni = area_a + area_b - inter;
    if uni <= 0.0 {
        0.0
    } else {
        inter / uni
    }
}

/// 把重叠的检测框按 IoU 合并, 邻居数少于 min_neighbors 的丢弃。
pub fn group_rectangles(rects: &[Rect], min_neighbors: u32) -> Vec<Rect> {
    if rects.is_empty() {
        return vec![];
    }
    if min_neighbors <= 1 {
        return rects.to_vec();
    }

    // 简单分组: 每个框自己先成组, 再和重叠>0.2的组合并
    let mut groups: Vec<Vec<Rect>> = Vec::new();
    for r in rects {
        let mut merged = false;
        for g in groups.iter_mut() {
            for gr in g.iter() {
                if iou(r, gr) > 0.2 {
                    g.push(*r);
                    merged = true;
                    break;
                }
            }
            if merged {
                break;
            }
        }
        if !merged {
            groups.push(vec![*r]);
        }
    }

    // 迭代: 把组之间重叠高的再合并
    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                let mut overlap = false;
                'check: for a in groups[i].iter() {
                    for b in groups[j].iter() {
                        if iou(a, b) > 0.1 {
                            overlap = true;
                            break 'check;
                        }
                    }
                }
                if overlap {
                    let mut other = groups.remove(j);
                    groups[i].append(&mut other);
                    changed = true;
                    break 'outer;
                }
            }
        }
    }

    let mut out = Vec::new();
    for g in groups {
        if g.len() as u32 >= min_neighbors {
            let n = g.len() as i32;
            let sum_x: i32 = g.iter().map(|r| r.x).sum();
            let sum_y: i32 = g.iter().map(|r| r.y).sum();
            let sum_w: i32 = g.iter().map(|r| r.w).sum();
            let sum_h: i32 = g.iter().map(|r| r.h).sum();
            out.push(Rect::new(sum_x / n, sum_y / n, sum_w / n, sum_h / n));
        }
    }
    out
}
