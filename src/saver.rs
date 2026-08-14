use crate::image::{BoxError, Image, Rect};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct FaceRecord {
    pub index: u64,
    pub timestamp_secs: f64,
    pub frame_index: u64,
    pub file_name: String,
    pub face_count: u32,
    pub boxes: Vec<[i32; 4]>,
    /// 跟踪 face_id (None 表示未启用 --track)
    pub face_ids: Vec<u32>,
    /// 是否为该 face_id 的代表帧 (在 --key-frames-only 模式下)
    pub is_keyframe: bool,
}

pub fn parse_frame_timestamp(path: &Path) -> (f64, u64) {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let mut parts: Vec<&str> = name.split('_').collect();
    let mut frame_idx = 0u64;
    let mut ms = 0u64;
    if !parts.is_empty() {
        if let Some(last) = parts.pop() {
            if last.ends_with("ms") {
                // 输出文件名格式: `0001_00h-00m-00s-001ms` — 末段是毫秒
                let s = last.trim_end_matches("ms");
                if s.chars().all(|c| c.is_ascii_digit()) {
                    ms = s.parse().unwrap_or(0);
                }
                if let Some(prev) = parts.pop() {
                    frame_idx = prev.parse().unwrap_or(0);
                }
            } else if last.chars().all(|c| c.is_ascii_digit()) {
                // ffmpeg 输出格式: `frame_000001` — 末段是帧编号
                frame_idx = last.parse().unwrap_or(0);
            }
        }
    }
    (ms as f64 / 1000.0, frame_idx)
}

pub fn format_timestamp(secs: f64) -> String {
    let total_ms = (secs * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{:02}h-{:02}m-{:02}s-{:03}ms", h, m, s, ms)
}

pub fn save_frame_with_faces(
    img: &Image,
    out_dir: &Path,
    index: u64,
    timestamp_secs: f64,
    frame_index: u64,
    faces: &[Rect],
    face_ids: &[u32],
    save_crops: bool,
    padding_ratio: f32,
) -> Result<FaceRecord, BoxError> {
    std::fs::create_dir_all(out_dir)?;
    let ts = format_timestamp(timestamp_secs);
    let file_name = format!("{:04}_{}.png", index, ts);
    let full_path = out_dir.join(&file_name);
    let mut annotated = img.clone();
    draw_rects(&mut annotated, faces);
    annotated.save_png(&full_path)?;
    let mut boxes = Vec::new();
    let mut crops_dir: Option<PathBuf> = None;
    if save_crops {
        crops_dir = Some(out_dir.join("crops"));
        std::fs::create_dir_all(crops_dir.as_ref().unwrap())?;
    }
    for (i, f) in faces.iter().enumerate() {
        boxes.push([f.x, f.y, f.w, f.h]);
        if let (Some(ref cd), true) = (crops_dir.as_ref(), save_crops) {
            let px = (f.w as f32 * padding_ratio) as i32;
            let py = (f.h as f32 * padding_ratio) as i32;
            let x = (f.x - px).max(0) as usize;
            let y = (f.y - py).max(0) as usize;
            let w = (f.w + 2 * px).min(img.width as i32 - x as i32) as usize;
            let h = (f.h + 2 * py).min(img.height as i32 - y as i32) as usize;
            let crop = img.crop(x, y, w, h);
            let crop_name = format!("{:04}_{}_face{:02}.png", index, ts, i + 1);
            crop.save_png(&cd.join(&crop_name))?;
        }
    }
    Ok(FaceRecord {
        index,
        timestamp_secs,
        frame_index,
        file_name,
        face_count: faces.len() as u32,
        boxes,
        face_ids: face_ids.to_vec(),
        is_keyframe: false,
    })
}

fn draw_rects(img: &mut Image, rects: &[Rect]) {
    for r in rects {
        let color = if img.channels >= 3 {
            vec![255u8, 0u8, 0u8]
        } else {
            vec![255u8]
        };
        let x0 = r.x.max(0) as usize;
        let y0 = r.y.max(0) as usize;
        let x1 = (r.x + r.w).min(img.width as i32 - 1) as usize;
        let y1 = (r.y + r.h).min(img.height as i32 - 1) as usize;
        for x in x0..=x1 {
            set_pixel(img, x, y0, &color);
            set_pixel(img, x, y1, &color);
        }
        for y in y0..=y1 {
            set_pixel(img, x0, y, &color);
            set_pixel(img, x1, y, &color);
        }
    }
}

fn set_pixel(img: &mut Image, x: usize, y: usize, color: &[u8]) {
    if x >= img.width || y >= img.height { return; }
    let idx = (y * img.width + x) * img.channels;
    for i in 0..img.channels.min(color.len()) {
        img.data[idx + i] = color[i];
    }
}

pub fn draw_label(img: &mut Image, x: i32, y: i32, label: &str) {
    let x = x.max(0) as usize;
    let y = y.max(0) as usize;
    let ch = img.channels;
    let color_r = if ch >= 3 { vec![0u8, 0u8, 200u8] } else { vec![200u8] };
    let color_fg = if ch >= 3 { vec![255u8, 255u8, 255u8] } else { vec![0u8] };
    let font_w = 5usize;
    let font_h = 8usize;
    let pad = 2;
    let text_w = label.len() * (font_w + 1) + pad * 2;
    let text_h = font_h + pad * 2;
    for yy in 0..text_h {
        for xx in 0..text_w {
            let px = (x + xx).min(img.width.saturating_sub(1));
            let py = (y as isize - text_h as isize + yy as isize).max(0) as usize;
            if py < img.height {
                let idx = (py * img.width + px) * ch;
                for i in 0..ch.min(color_r.len()) {
                    img.data[idx + i] = color_r[i];
                }
            }
        }
    }
    for (ci, ch_c) in label.chars().enumerate() {
        let glyph = char_to_glyph(ch_c);
        for gy in 0..font_h {
            for gx in 0..font_w {
                if glyph[gy][gx] == 1 {
                    let px = (x + pad + ci * (font_w + 1) + gx).min(img.width.saturating_sub(1));
                    let py = (y as isize - text_h as isize + pad as isize + gy as isize).max(0) as usize;
                    if py < img.height {
                        let idx = (py * img.width + px) * ch;
                        for i in 0..ch.min(color_fg.len()) {
                            img.data[idx + i] = color_fg[i];
                        }
                    }
                }
            }
        }
    }
}

fn char_to_glyph(c: char) -> [[u8; 5]; 8] {
    match c {
        'A' | 'a' => [
            [0,1,1,0,0],[1,0,0,1,0],[1,0,0,1,0],[1,1,1,1,0],
            [1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[0,0,0,0,0],
        ],
        'B' | 'b' => [
            [1,1,1,0,0],[1,0,0,1,0],[1,0,0,1,0],[1,1,1,0,0],
            [1,0,0,1,0],[1,0,0,1,0],[1,1,1,0,0],[0,0,0,0,0],
        ],
        'C' | 'c' => [
            [0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],
            [1,0,0,0,0],[1,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0],
        ],
        'D' | 'd' => [
            [1,1,1,0,0],[1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],
            [1,0,0,1,0],[1,0,0,1,0],[1,1,1,0,0],[0,0,0,0,0],
        ],
        'E' | 'e' => [
            [1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,0,0],
            [1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,0],
        ],
        'F' | 'f' => [
            [1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,0,0],
            [1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],
        ],
        'G' | 'g' => [
            [0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,1,1,0],
            [1,0,0,1,0],[1,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0],
        ],
        'H' | 'h' => [
            [1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[1,1,1,1,0],
            [1,0,0,1,0],[1,0,0,1,0],[1,0,0,1,0],[0,0,0,0,0],
        ],
        'I' | 'i' => [
            [0,1,1,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],
            [0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,0,0,0],
        ],
        'J' | 'j' => [
            [0,0,1,1,1],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,1,0],
            [0,0,0,1,0],[1,0,0,1,0],[0,1,1,0,0],[0,0,0,0,0],
        ],
        'K' | 'k' => [
            [1,0,0,1,0],[1,0,1,0,0],[1,1,0,0,0],[1,0,0,0,0],
            [1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[0,0,0,0,0],
        ],
        'L' | 'l' => [
            [1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],
            [1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,0],
        ],
        'M' | 'm' => [
            [1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],
            [1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,0,0,0,0],
        ],
        'N' | 'n' => [
            [1,0,0,0,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1],
            [1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,0,0,0,0],
        ],
        'O' | 'o' | '0' => [
            [0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],
            [1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[0,0,0,0,0],
        ],
        'P' | 'p' => [
            [1,1,1,0,0],[1,0,0,1,0],[1,0,0,1,0],[1,1,1,0,0],
            [1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0],
        ],
        'Q' | 'q' => [
            [0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],
            [1,0,1,0,1],[1,0,0,1,0],[0,1,1,0,1],[0,0,0,0,0],
        ],
        'R' | 'r' => [
            [1,1,1,0,0],[1,0,0,1,0],[1,0,0,1,0],[1,1,1,0,0],
            [1,0,1,0,0],[1,0,0,1,0],[1,0,0,1,0],[0,0,0,0,0],
        ],
        'S' | 's' | '5' => [
            [0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,1,0,0],
            [0,0,0,1,0],[0,0,0,1,0],[1,1,1,0,0],[0,0,0,0,0],
        ],
        'T' | 't' => [
            [1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],
            [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],
        ],
        'U' | 'u' => [
            [1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],
            [1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[0,0,0,0,0],
        ],
        'V' | 'v' => [
            [1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],
            [1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,0],
        ],
        'W' | 'w' => [
            [1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],
            [1,0,1,0,1],[1,0,1,0,1],[0,1,0,1,0],[0,0,0,0,0],
        ],
        'X' | 'x' => [
            [1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],
            [0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[0,0,0,0,0],
        ],
        'Y' | 'y' => [
            [1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],
            [0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],
        ],
        'Z' | 'z' | '2' => [
            [1,1,1,1,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],
            [0,1,0,0,0],[1,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],
        ],
        '1' => [
            [0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],
            [0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,0,0,0],
        ],
        '3' => [
            [1,1,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,1,1,0,0],
            [0,0,0,1,0],[0,0,0,1,0],[1,1,1,0,0],[0,0,0,0,0],
        ],
        '4' => [
            [0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],
            [1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0],
        ],
        '6' => [
            [0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,0,0],
            [1,0,0,1,0],[1,0,0,1,0],[0,1,1,0,0],[0,0,0,0,0],
        ],
        '7' => [
            [1,1,1,1,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0],
            [0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,0,0,0],
        ],
        '8' => [
            [0,1,1,1,0],[1,0,0,1,0],[1,0,0,1,0],[0,1,1,0,0],
            [1,0,0,1,0],[1,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0],
        ],
        '9' => [
            [0,1,1,1,0],[1,0,0,1,0],[1,0,0,1,0],[0,1,1,1,1],
            [0,0,0,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0],
        ],
        '-' | '_' => [
            [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
            [0,0,0,0,0],[1,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0],
        ],
        ':' => [
            [0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0],
            [0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0],
        ],
        '.' | ',' => [
            [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
            [0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],
        ],
        ' ' => [[0u8;5];8],
        _ => [[0u8;5];8],
    }
}
