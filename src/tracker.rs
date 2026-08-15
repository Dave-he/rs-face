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
    /// 代表帧的图像 (face 框最大的那一帧), 用于 HTML 报告缩略图
    pub cover: Option<crate::image::Image>,
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
            cover: Some(img.clone()),
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

    /// 后处理: 把位置相近 + 描述子相似 + 时间不重叠的轨道合并。
    /// 解决因为中空/转场导致的同一张脸被切成 N 段的问题。
    pub fn merge_similar_tracks(&mut self, merge_threshold: f64) {
        if self.tracks.len() < 2 { return; }
        let mut merged_into: std::collections::HashMap<usize, usize> = std::collections::HashMap::new(); // 旧 id -> 新 id
        let track_count = self.tracks.len();
        for i in 0..track_count {
            let mut cur = i;
            loop {
                if let Some(&next) = merged_into.get(&cur) {
                    cur = next;
                } else {
                    break;
                }
            }
            for j in (i + 1)..track_count {
                let mut cj = j;
                while let Some(&n) = merged_into.get(&cj) {
                    cj = n;
                }
                if cur == cj { continue; }
                // 位置相近? 同位置 + 描述子相似 = 合并 (允许时间重叠, 例如转场)
                let box_a = self.tracks[cur].sample_box;
                let box_b = self.tracks[cj].sample_box;
                let ax = box_a[0] + box_a[2] / 2;
                let ay = box_a[1] + box_a[3] / 2;
                let bx = box_b[0] + box_b[2] / 2;
                let by = box_b[1] + box_b[3] / 2;
                let dx = (ax - bx).abs();
                let dy = (ay - by).abs();
                if dx > 100 || dy > 100 { continue; }
                // 描述子相似? 放宽 0.05 容差 (历史画廊 EMA 漂移)
                let d = cosine_distance(&self.galleries[cur], &self.galleries[cj]);
                let eff_thr = merge_threshold + 0.05;
                if d < eff_thr {
                    merged_into.insert(cj, cur);
                }
            }
        }
        if merged_into.is_empty() { return; }
        // 用并查集重置 face_id: 旧 id -> root
        let mut id_map: Vec<usize> = (0..track_count).collect();
        for (&old, &new) in &merged_into {
            // 路径压缩
            let mut root_old = old;
            while id_map[root_old] != root_old { root_old = id_map[root_old]; }
            let mut root_new = new;
            while id_map[root_new] != root_new { root_new = id_map[root_new]; }
            // 合并: 小 id 作为 root
            let winner = root_old.min(root_new);
            let loser = root_old.max(root_new);
            id_map[loser] = winner;
        }
        // 压缩: 把同组的 tracks 合并
        let mut new_tracks: Vec<FaceTrack> = Vec::new();
        let mut new_galleries: Vec<Vec<f64>> = Vec::new();
        let mut root_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for (old, &mapped) in id_map.iter().enumerate() {
            let mut root = mapped;
            while id_map[root] != root {
                root = id_map[root];
            }
            if let Some(&new_id) = root_to_new.get(&root) {
                let new_track = &mut new_tracks[new_id];
                let old_track = if let Some(t) = self.tracks.get(old) {
                    t
                } else { continue; };
                new_track.first_ts = new_track.first_ts.min(old_track.first_ts);
                new_track.last_ts = new_track.last_ts.max(old_track.last_ts);
                new_track.frame_count += old_track.frame_count;
                new_track.frames.extend_from_slice(&old_track.frames);
            } else {
                let new_id = new_tracks.len();
                root_to_new.insert(root, new_id);
                let mut track = self.tracks[old].clone();
                track.id = new_id as u32;
                new_tracks.push(track);
                if old < self.galleries.len() {
                    new_galleries.push(self.galleries[old].clone());
                }
            }
        }
        // 按 timestamp_secs 重新排序帧 (用 total_cmp 避免 f64 Ord 限制)
        for t in &mut new_tracks {
            t.frames.sort_by(|a, b| a.timestamp_secs.partial_cmp(&b.timestamp_secs).unwrap_or(std::cmp::Ordering::Equal));
        }
        self.tracks = new_tracks;
        self.galleries = new_galleries;
    }

    /// 写出 JSON 报告 (手写, 零依赖)
    /// 从先前 run 的 tracks.json 加载画廊, 把当前帧合并到旧 face_id (跨视频合并)
    pub fn load_and_merge_from_json(&mut self, path: &Path, merge_threshold: f64) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        // 解析 tracks 数组: 找 "face_id": N, .* "sample_box": [x, y, w, h], .* "gallery": [...]
        let mut prior_ids: Vec<u32> = Vec::new();
        let mut prior_boxes: Vec<[i32; 4]> = Vec::new();
        let mut prior_galleries: Vec<Vec<f64>> = Vec::new();
        let bytes = content.as_bytes();
        let mut i = 0;
        while i + 30 < bytes.len() {
            if let Some(p) = content[i..].find("\"face_id\":") {
                let abs = i + p;
                let after = abs + 10;
                let mut j = after;
                while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                    j += 1;
                }
                let num_start = j;
                while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                    j += 1;
                }
                if let Ok(id) = content[num_start..j].parse::<u32>() {
                    prior_ids.push(id);
                    // 找 "sample_box":
                    let mut box_ = [0i32; 4];
                    if let Some(sb) = content[j..].find("\"sample_box\":") {
                        let sb_abs = j + sb + 13;
                        let brack_l = content[sb_abs..].find('[');
                        let brack_r = content[sb_abs..].find(']');
                        if let (Some(l), Some(r)) = (brack_l, brack_r) {
                            let inside = &content[sb_abs + l + 1..sb_abs + r];
                            let nums: Vec<i32> = inside.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                            if nums.len() == 4 {
                                box_ = [nums[0], nums[1], nums[2], nums[3]];
                            }
                        }
                    }
                    prior_boxes.push(box_);
                    // 找 "gallery": (在 tracks.json 高版本里有, 旧版没)
                    let rest = &content[j..];
                    if let Some(gp) = rest.find("\"gallery\":") {
                        let g_abs = j + gp + 10;
                        let brack_l = content[g_abs..].find('[');
                        let brack_r = content[g_abs..].find(']');
                        if let (Some(l), Some(r)) = (brack_l, brack_r) {
                            let inside = &content[g_abs + l + 1..g_abs + r];
                            let vals: Vec<f64> = inside.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                            if !vals.is_empty() {
                                prior_galleries.push(vals);
                            }
                        }
                    }
                }
                i = j;
            } else {
                break;
            }
        }
        if prior_ids.is_empty() {
            return Err("no tracks found in prior JSON".into());
        }
        // 合并策略:
        // 1) 若 prior_galleries 非空, 用 LBPH 余弦距离 (细筛)
        // 2) 否则, 用 box 位置 + 大小相似度 (粗筛)
        let use_gallery = !prior_galleries.is_empty() && prior_galleries.iter().all(|g| g.len() == self.galleries.first().map(|x| x.len()).unwrap_or(0));
        let mut id_map: Vec<u32> = (0..self.tracks.len() as u32).collect();
        for (i, cur) in self.tracks.iter().enumerate() {
            let cb = cur.sample_box;
            let mut best_id = cur.id;
            let mut best_dist = f64::INFINITY;
            for (j, &pb) in prior_boxes.iter().enumerate() {
                let dx = (cb[0] - pb[0]).abs();
                let dy = (cb[1] - pb[1]).abs();
                if dx > 150 || dy > 150 { continue; }
                let dist = if use_gallery {
                    let d = cosine_distance(&self.galleries[i], &prior_galleries[j]);
                    if d >= merge_threshold { continue; }
                    d
                } else {
                    let area_diff = ((cb[2] * cb[3] - pb[2] * pb[3]).abs()) as f64 /
                        (cb[2] * cb[3] + pb[2] * pb[3]) as f64;
                    let combined = (dx + dy) as f64 / 200.0 + area_diff;
                    combined
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_id = prior_ids[j];
                }
            }
            id_map[i] = best_id;
        }
        for (i, &new_id) in id_map.iter().enumerate() {
            self.tracks[i].id = new_id;
        }
        Ok(())
    }

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
            if let Some(g) = self.galleries.get(i) {
                s.push_str(",\n      \"gallery\": [");
                for (k, v) in g.iter().enumerate() {
                    if k > 0 { s.push(','); }
                    let _ = write!(s, "{:.6}", v);
                }
                s.push(']');
            }
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
