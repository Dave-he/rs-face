// HOG (Histogram of Oriented Gradients) + Linear SVM 人脸检测器。
//
// 算法来源: Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection",
//           CVPR 2005. 同年方法被 dlib `get_frontal_face_detector()` 采用。
//
// 流程:
//   1. 梯度计算 (中心差分)
//   2. cell 级梯度方向直方图 (默认 8x8 px/cell, 9 个方向 bin, 0°~180°)
//   3. block 级 L2-norm 归一化 (2x2 cell/block)
//   4. 滑动窗口 + 多尺度
//   5. 线性 SVM 打分 (超平面决策: w·x + b > 0 => face)
//
// 训练模式: 给定正/负样本, 在 HOG 特征上拟合线性 SVM (Hinge Loss + SGD),
//           保存权重向量到磁盘。预测模式加载权重, 做滑窗打分。
//
// 备注: 本模块遵循"零依赖"约束, 所有计算都在 std 中实现。

use crate::image::{BoxError, Image, Rect};
use crate::imgproc::iou;

pub const HOG_CELL: usize = 8;
pub const HOG_BLOCK: usize = 2;
pub const HOG_BINS: usize = 9;
pub const HOG_WINDOW: usize = 64;
/// HOG 描述子维度: (8-2+1)^2 * 2*2 * 9 = 49*4*9 = 1764
pub const HOG_DESC_LEN: usize = 1764;

/// 计算 HOG 描述子, 输入必须 == HOG_WINDOW x HOG_WINDOW 灰度图。
pub fn compute_hog(gray: &[u8], w: usize, h: usize) -> Vec<f64> {
    assert_eq!(w, HOG_WINDOW);
    assert_eq!(h, HOG_WINDOW);
    let n = w * h;
    let mut gx = vec![0f32; n];
    let mut gy = vec![0f32; n];
    for y in 0..h {
        for x in 0..w {
            let xm = if x == 0 { 0 } else { x - 1 };
            let xp = if x == w - 1 { w - 1 } else { x + 1 };
            let ym = if y == 0 { 0 } else { y - 1 };
            let yp = if y == h - 1 { h - 1 } else { y + 1 };
            gx[y * w + x] = gray[y * w + xp] as f32 - gray[y * w + xm] as f32;
            gy[y * w + x] = gray[yp * w + x] as f32 - gray[ym * w + x] as f32;
        }
    }
    let mut mag = vec![0f32; n];
    let mut ori = vec![0f32; n];
    for i in 0..n {
        let mx = gx[i];
        let my = gy[i];
        mag[i] = (mx * mx + my * my).sqrt();
        let mut a = my.atan2(mx);
        if a < 0.0 { a += std::f32::consts::PI; }
        ori[i] = a;
    }

    let ncells = w / HOG_CELL;
    let mut cells = vec![vec![0f32; HOG_BINS]; ncells * ncells];
    for cy in 0..ncells {
        for cx in 0..ncells {
            for dy in 0..HOG_CELL {
                for dx in 0..HOG_CELL {
                    let x = cx * HOG_CELL + dx;
                    let y = cy * HOG_CELL + dy;
                    let m = mag[y * w + x];
                    let o = ori[y * w + x];
                    let bin_f = o / std::f32::consts::PI * HOG_BINS as f32;
                    let bin0 = bin_f.floor() as usize % HOG_BINS;
                    let frac = bin_f - bin_f.floor();
                    let bin1 = (bin0 + 1) % HOG_BINS;
                    cells[cy * ncells + cx][bin0] += m * (1.0 - frac);
                    cells[cy * ncells + cx][bin1] += m * frac;
                }
            }
        }
    }
    let nblocks = ncells - HOG_BLOCK + 1;
    let mut desc = vec![0f64; HOG_DESC_LEN];
    let mut idx = 0usize;
    let eps = 1e-6f64;
    for by in 0..nblocks {
        for bx in 0..nblocks {
            let mut sum_sq = 0.0f64;
            for dy in 0..HOG_BLOCK {
                for dx in 0..HOG_BLOCK {
                    for b in 0..HOG_BINS {
                        let v = cells[(by + dy) * ncells + (bx + dx)][b] as f64;
                        sum_sq += v * v;
                    }
                }
            }
            let norm = (sum_sq + eps).sqrt();
            for dy in 0..HOG_BLOCK {
                for dx in 0..HOG_BLOCK {
                    for b in 0..HOG_BINS {
                        let v = cells[(by + dy) * ncells + (bx + dx)][b] as f64;
                        desc[idx] = v / norm;
                        idx += 1;
                    }
                }
            }
        }
    }
    desc
}

/// 双线性缩放灰度图到 new_w x new_h。
pub fn resize_gray_bilinear(gray: &[u8], w: usize, h: usize, new_w: usize, new_h: usize) -> Vec<u8> {
    if w == new_w && h == new_h { return gray.to_vec(); }
    let mut out = vec![0u8; new_w * new_h];
    let x_ratio = w as f64 / new_w as f64;
    let y_ratio = h as f64 / new_h as f64;
    for ny in 0..new_h {
        let sy = (ny as f64 * y_ratio).min((h as f64) - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let dy = sy - y0 as f64;
        for nx in 0..new_w {
            let sx = (nx as f64 * x_ratio).min((w as f64) - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let dx = sx - x0 as f64;
            let v00 = gray[y0 * w + x0] as f64;
            let v01 = gray[y0 * w + x1] as f64;
            let v10 = gray[y1 * w + x0] as f64;
            let v11 = gray[y1 * w + x1] as f64;
            let top = v00 * (1.0 - dx) + v01 * dx;
            let bot = v10 * (1.0 - dx) + v11 * dx;
            out[ny * new_w + nx] = (top * (1.0 - dy) + bot * dy).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ----------- 线性 SVM (Hinge Loss + SGD) -----------

#[derive(Debug, Clone)]
pub struct LinearSvm {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub n_dim: usize,
}

impl LinearSvm {
    pub fn new(n_dim: usize) -> Self {
        Self { weights: vec![0.0; n_dim], bias: 0.0, n_dim }
    }

    pub fn predict_raw(&self, x: &[f64]) -> f64 {
        let mut s = self.bias;
        let n = x.len().min(self.weights.len());
        for i in 0..n { s += self.weights[i] * x[i]; }
        s
    }

    pub fn predict_label(&self, x: &[f64]) -> i32 {
        if self.predict_raw(x) >= 0.0 { 1 } else { -1 }
    }

    /// Hinge Loss + SGD, labels 取 +1 (face) 或 -1 (non-face)。
    pub fn train_sgd(
        samples: &[Vec<f64>],
        labels: &[i32],
        iterations: usize,
        lr: f64,
        lambda: f64,
    ) -> Result<Self, BoxError> {
        if samples.is_empty() { return Err("LinearSvm: empty training set".into()); }
        let n = samples.len();
        let d = samples[0].len();
        for s in samples { if s.len() != d { return Err("LinearSvm: dim mismatch".into()); } }
        let mut w = vec![0.0f64; d];
        let mut b = 0.0f64;
        for _it in 0..iterations {
            for i in 0..n {
                let y = labels[i] as f64;
                let mut dot = b;
                for j in 0..d { dot += w[j] * samples[i][j]; }
                let margin = y * dot;
                if margin < 1.0 {
                    for j in 0..d { w[j] -= lr * (lambda * w[j] - y * samples[i][j]); }
                    b += lr * y;
                } else {
                    for j in 0..d { w[j] -= lr * lambda * w[j]; }
                }
            }
        }
        Ok(Self { weights: w, bias: b, n_dim: d })
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), BoxError> {
        let mut s = String::new();
        s.push_str("HOG_SVM_V1\n");
        s.push_str(&format!("dim {}\n", self.n_dim));
        s.push_str(&format!("bias {:.10}\n", self.bias));
        s.push_str("weights\n");
        for &w in &self.weights { s.push_str(&format!("{:.10}\n", w)); }
        std::fs::write(path, &s)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BoxError> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.lines();
        let header = lines.next().ok_or("bad svm file")?.trim();
        if header != "HOG_SVM_V1" { return Err("Not an HOG_SVM model".into()); }
        let dim_line = lines.next().ok_or("no dim")?.trim();
        let n_dim: usize = dim_line.strip_prefix("dim ").ok_or("bad dim")?.parse()?;
        let bias_line = lines.next().ok_or("no bias")?.trim();
        let bias: f64 = bias_line.strip_prefix("bias ").ok_or("bad bias")?.parse()?;
        let w_hdr = lines.next().ok_or("no weights hdr")?.trim();
        if w_hdr != "weights" { return Err("bad weights header".into()); }
        let mut weights = Vec::with_capacity(n_dim);
        for _ in 0..n_dim {
            let l = lines.next().ok_or("truncated weights")?.trim();
            weights.push(l.parse().unwrap_or(0.0));
        }
        Ok(Self { weights, bias, n_dim })
    }
}

// ----------- 多尺度滑窗 -----------

#[derive(Debug, Clone)]
pub struct HogSvmDetector {
    pub svm: LinearSvm,
    pub window: usize,
}

impl HogSvmDetector {
    pub fn new(svm: LinearSvm) -> Self { Self { svm, window: HOG_WINDOW } }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BoxError> {
        let svm = LinearSvm::load(path)?;
        Ok(Self { svm, window: HOG_WINDOW })
    }

    /// 多尺度滑窗检测。
    /// 算法: 对每个目标人脸尺寸 s, 把图像缩放到 (img_w * HOG_WINDOW/s, img_h * HOG_WINDOW/s),
    ///       然后用固定 HOG_WINDOW 大小窗口滑窗, 把窗口坐标反算回原图坐标。
    pub fn detect(
        &self,
        img: &Image,
        min_size: u32,
        max_size: u32,
        scale_factor: f32,
        step: u32,
        bias_threshold: f64,
    ) -> Vec<(Rect, f64)> {
        let mut out = Vec::new();
        if img.width == 0 || img.height == 0 { return out; }
        let gray = img.to_grayscale();
        let win_px = self.window as f64;
        let mut target_size = min_size.max(self.window as u32) as f64;
        loop {
            if target_size > max_size as f64 { break; }
            let scale = win_px / target_size;
            let new_w = ((img.width as f64) * scale).round().max(win_px) as usize;
            let new_h = ((img.height as f64) * scale).round().max(win_px) as usize;
            if new_w >= self.window && new_h >= self.window {
                let resized = resize_gray_bilinear(&gray, img.width, img.height, new_w, new_h);
                let step_x = step.max(1) as usize;
                let mut y = 0usize;
                while y + self.window <= new_h {
                    let mut x = 0usize;
                    while x + self.window <= new_w {
                        // 提取窗口
                        let mut win_gray: Vec<u8> = Vec::with_capacity(self.window * self.window);
                        for wy in 0..self.window {
                            let row_start = (y + wy) * new_w + x;
                            win_gray.extend_from_slice(&resized[row_start..row_start + self.window]);
                        }
                        let desc = compute_hog(&win_gray, self.window, self.window);
                        let s = self.svm.predict_raw(&desc);
                        if s >= bias_threshold {
                            let rx = (x as f64 / scale).round() as i32;
                            let ry = (y as f64 / scale).round() as i32;
                            let rw = target_size.round() as i32;
                            out.push((Rect::new(rx, ry, rw, rw), s));
                        }
                        x += step_x;
                    }
                    y += step_x;
                }
            }
            target_size *= scale_factor as f64;
            if target_size > (img.width.max(img.height) as f64) + 32.0 { break; }
        }
        out
    }
}

// ----------- NMS -----------

/// 按 score 降序贪心 NMS, IoU 阈值 iou_thresh。
pub fn nms_hog(rects: &[(Rect, f64)], iou_thresh: f32) -> Vec<(Rect, f64)> {
    if rects.is_empty() { return Vec::new(); }
    let mut sorted: Vec<(Rect, f64)> = rects.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<(Rect, f64)> = Vec::new();
    let mut suppressed = vec![false; sorted.len()];
    for i in 0..sorted.len() {
        if suppressed[i] { continue; }
        keep.push(sorted[i]);
        for j in (i + 1)..sorted.len() {
            if suppressed[j] { continue; }
            if iou(&sorted[i].0, &sorted[j].0) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// 合并两路检测结果 (Viola-Jones + HOG-SVM), 用 NMS 抑制重叠框。
pub fn merge_detections(
    a: &[Rect],
    b: &[Rect],
    iou_thresh: f32,
) -> Vec<Rect> {
    let mut all: Vec<(Rect, f64)> = Vec::new();
    for r in a { all.push((*r, 1.0)); }
    for r in b { all.push((*r, 0.8)); }
    nms_hog(&all, iou_thresh).into_iter().map(|(r, _)| r).collect()
}

/// 提取指定矩形区域的 HOG 描述子(直方图均衡化 + 缩放到 HOG_WINDOW)。
pub fn extract_hog_window(img: &Image, rect: Rect) -> Vec<f64> {
    let gray = img.to_grayscale();
    let x0 = rect.x.max(0) as usize;
    let y0 = rect.y.max(0) as usize;
    let x1 = (rect.x + rect.w).min(img.width as i32).max(0) as usize;
    let y1 = (rect.y + rect.h).min(img.height as i32).max(0) as usize;
    if x1 <= x0 || y1 <= y0 { return vec![0.0; HOG_DESC_LEN]; }
    let mut patch = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for y in y0..y1 { for x in x0..x1 { patch.push(gray[y * img.width + x]); } }
    let pw = x1 - x0;
    let ph = y1 - y0;
    let equalized = crate::imgproc::histogram_equalize(&patch, pw, ph);
    let resized = resize_gray_bilinear(&equalized, pw, ph, HOG_WINDOW, HOG_WINDOW);
    compute_hog(&resized, HOG_WINDOW, HOG_WINDOW)
}

/// 从图像列表 (正样本) 和 (负样本) 训练 HOG-SVM 检测器。
pub fn train_hog_svm(
    positives: &[Image],
    negatives: &[Image],
    iterations: usize,
    lr: f64,
    lambda: f64,
) -> Result<HogSvmDetector, BoxError> {
    let mut samples: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();
    for img in positives {
        let w = img.width as i32;
        let h = img.height as i32;
        samples.push(extract_hog_window(img, Rect::new(0, 0, w, h)));
        labels.push(1);
    }
    for img in negatives {
        let w = img.width as i32;
        let h = img.height as i32;
        samples.push(extract_hog_window(img, Rect::new(0, 0, w, h)));
        labels.push(-1);
    }
    let svm = LinearSvm::train_sgd(&samples, &labels, iterations, lr, lambda)?;
    Ok(HogSvmDetector::new(svm))
}