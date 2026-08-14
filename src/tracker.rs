// 在线人脸聚类跟踪器
//
// 用 LBPH (Local Binary Patterns Histograms) 做人脸聚类, 教学视频效果最佳:
// - 抗光照变化 (LBP 不变)
// - 简单, 计算快
// - 8x8 网格, 256 bin, 16384 维直方图
//
// 流程:
//   1. 人脸 → 92x112 灰度 → 直方图均衡化 → LBPH (8x8 网格, 256 bin)
//   2. 与画廊做余弦距离比对
//   3. 距离 < merge_threshold + 空间位置约束 (中心 < 100px) → 归并
//   4. 画廊在线均值更新 (0.8 老 + 0.2 新) 吸收漂移
//
// 输出: 手写 JSON 报告 (零依赖)

use crate::image::{Image, Rect};
use crate::recognition::LBPHModel;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FaceTrack {
    pub id: u32,
    pub first_ts: f64,
    pub last_ts: f64,
    pub frame_count: u32,
    pub sample_box: [i32; 4],
    pub frames: Vec<TrackFrame>,
}

#[derive(Debug, Clone)]
pub struct TrackFrame {
    pub file_index: u64,
    pub timestamp_secs: f64,
    pub box_: [i32; 4],
}

pub struct FaceTracker {
    model: LBPHModel,
    galleries: Vec<Vec<f64>>,
    grid_x: usize,
    grid_y: usize,
    face_w: usize,
    face_h: usize,
    merge_threshold: f64,
    tracks: Vec<FaceTrack>,
}

impl FaceTracker {
    pub fn new(merge_threshold: f64) -> Self {
        let grid_x = 8;
        let grid_y = 8;
        let face_w = 92;
        let face_h = 112;
        Self {
            model: LBPHModel::with_params(grid_x, grid_y, 1, 8),
            galleries: Vec::new(),
            grid_x,
            grid_y,
            face_w,
            face_h,
            merge_threshold,
            tracks: Vec::new(),
        }
    }

    pub fn register(&mut self, img: &Image, face: &Rect, file_index: u64, ts_secs: f64) -> u32 {
        let hist = self.histogram_of(img, face);
        let box_ = [face.x, face.y, face.w, face.h];
        let cur_cx = face.x + face.w / 2;
        let cur_cy = face.y + face.h / 2;
        // 候选过滤: 空间位置相近 (中心 < 100 px) 的轨道里找最近邻
        // 使用余弦距离 (1 - cosine_similarity), 比卡方距离更鲁棒
        let mut best_id: Option<u32> = None;
        let mut best_dist = f64::INFINITY;
        for (i, g) in self.galleries.iter().enumerate() {
            let track = &self.tracks[i];
            let last_box = &track.frames.last().unwrap().box_;
            let last_cx = last_box[0] + last_box[2] / 2;
            let last_cy = last_box[1] + last_box[3] / 2;
            let dx = (last_cx - cur_cx).abs();
            let dy = (last_cy - cur_cy).abs();
            if dx > 100 || dy > 100 {
                continue;
            }
            let d = cosine_distance(&hist, g);
            if d < best_dist {
                best_dist = d;
                best_id = Some(i as u32);
            }
        }
        if let Some(best_id) = best_id {
            if best_dist < self.merge_threshold {
                let track = &mut self.tracks[best_id as usize];
                track.last_ts = ts_secs;
                track.frame_count += 1;
                track.frames.push(TrackFrame { file_index, timestamp_secs: ts_secs, box_ });
                let g = &mut self.galleries[best_id as usize];
                for (gi, hi) in g.iter_mut().zip(hist.iter()) {
                    *gi = *gi * 0.8 + *hi * 0.2;
                }
                return best_id;
            }
        }
        let id = self.tracks.len() as u32;
        self.galleries.push(hist);
        self.tracks.push(FaceTrack {
            id,
            first_ts: ts_secs,
            last_ts: ts_secs,
            frame_count: 1,
            sample_box: box_,
            frames: vec![TrackFrame { file_index, timestamp_secs: ts_secs, box_ }],
        });
        id
    }

    fn histogram_of(&self, img: &Image, face: &Rect) -> Vec<f64> {
        let cw = face.w.max(1) as usize;
        let ch = face.h.max(1) as usize;
        let mut gray = vec![0u8; cw * ch];
        let chans = img.channels;
        for j in 0..ch {
            for i in 0..cw {
                let sx = (face.x + i as i32).max(0) as usize;
                let sy = (face.y + j as i32).max(0) as usize;
                if sx < img.width && sy < img.height {
                    let idx = (sy * img.width + sx) * chans;
                    gray[j * cw + i] = img.data[idx];
                }
            }
        }
        let side = cw.min(ch);
        let ox = (cw - side) / 2;
        let oy = (ch - side) / 2;
        let mut sq = vec![0u8; side * side];
        for j in 0..side {
            for i in 0..side {
                sq[j * side + i] = gray[(oy + j) * cw + (ox + i)];
            }
        }
        let mut resized = vec![0u8; self.face_w * self.face_h];
        for y in 0..self.face_h {
            let sy = (y * side / self.face_h).min(side - 1);
            for x in 0..self.face_w {
                let sx = (x * side / self.face_w).min(side - 1);
                resized[y * self.face_w + x] = sq[sy * side + sx];
            }
        }
        let mut hist = [0u32; 256];
        for &v in &resized { hist[v as usize] += 1; }
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
        for v in resized.iter_mut() { *v = lut[*v as usize]; }
        let normalized: Vec<f64> = resized.iter().map(|&v| v as f64 / 255.0).collect();
        self.model.compute_lbph(&normalized, self.face_w, self.face_h)
    }

    pub fn num_tracks(&self) -> usize { self.tracks.len() }
    pub fn tracks(&self) -> &[FaceTrack] { &self.tracks }

    /// 写出 JSON 报告 (手写, 零依赖)
    pub fn write_report(&self, path: &Path) -> Result<(), std::io::Error> {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"summary\": {{\n"));
        s.push_str(&format!("    \"total_unique_faces\": {},\n", self.tracks.len()));
        s.push_str(&format!("    \"merge_threshold\": {},\n", self.merge_threshold));
        s.push_str(&format!("    \"feature\": \"LBPH (8x8 grid, 256 bin)\",\n"));
        s.push_str(&format!("    \"face_size\": [{}, {}],\n", self.face_w, self.face_h));
        s.push_str(&format!("    \"grid\": [{}, {}]\n", self.grid_x, self.grid_y));
        s.push_str("  },\n");
        s.push_str("  \"tracks\": [\n");
        for (i, t) in self.tracks.iter().enumerate() {
            s.push_str("    {\n");
            let _ = writeln!(s, "      \"face_id\": {},", t.id);
            let _ = writeln!(s, "      \"first_ts\": {:.6},", t.first_ts);
            let _ = writeln!(s, "      \"last_ts\": {:.6},", t.last_ts);
            let _ = writeln!(s, "      \"duration_secs\": {:.6},", t.last_ts - t.first_ts);
            let _ = writeln!(s, "      \"frame_count\": {},", t.frame_count);
            let _ = writeln!(s, "      \"sample_box\": [{}, {}, {}, {}]", t.sample_box[0], t.sample_box[1], t.sample_box[2], t.sample_box[3]);
            s.push_str(",\n      \"frames\": [\n");
            for (j, f) in t.frames.iter().enumerate() {
                s.push_str(&format!("        {{\"file_index\": {}, \"timestamp_secs\": {:.6}, \"box\": [{}, {}, {}, {}]}}",
                    f.file_index, f.timestamp_secs, f.box_[0], f.box_[1], f.box_[2], f.box_[3]));
                if j + 1 < t.frames.len() { s.push(','); }
                s.push('\n');
            }
            s.push_str("      ]\n");
            s.push_str("    }");
            if i + 1 < self.tracks.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("  ]\n");
        s.push_str("}\n");
        std::fs::write(path, s)
    }
}

/// 余弦距离 = 1 - 余弦相似度。两个归一化直方图的余弦距离越小越相似。
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() { return f64::INFINITY; }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 { return 1.0; }
    let cos = dot / (na.sqrt() * nb.sqrt());
    1.0 - cos.clamp(-1.0, 1.0)
}
