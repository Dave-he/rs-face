// HTML 报告生成器 (单文件 HTML, 浏览器直接打开)
//
// 内容:
// - 每个 face_id 一张时间轴卡片 + 内嵌人脸缩略图
// - 视频元数据 (fps, 总帧数, 命中)
// - 颜色编码匹配 saver::draw_rects_palette (7 色循环)
// - 链接到 manifest.txt + tracks.json

use crate::image::Image;
use crate::saver::FaceRecord;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Default, Clone)]
pub struct Thumbnail {
    pub face_id: u32,
    pub png_base64: String,
}

pub struct HtmlReport<'a> {
    pub video_path: &'a Path,
    pub fps: f64,
    pub records: &'a [FaceRecord],
    pub tracks: Option<&'a [crate::tracker::FaceTrack]>,
    pub thumbnails: Vec<Thumbnail>,
}

const PALETTE: &[(u8, u8, u8); 7] = &[
    (255, 80, 80),   // 红
    (80, 220, 80),   // 绿
    (80, 150, 255),  // 蓝
    (255, 220, 80),  // 黄
    (220, 80, 255),  // 紫
    (80, 230, 230),  // 青
    (255, 160, 80),  // 橙
];

pub fn render(report: &HtmlReport) -> String {
    let mut s = String::new();
    s.push_str("<!DOCTYPE html>\n<html lang=\"zh\">\n<head>\n");
    s.push_str("<meta charset=\"UTF-8\">\n");
    s.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    let _ = writeln!(s, "<title>rs-face 报告: {}</title>", report.video_path.file_name().and_then(|s| s.to_str()).unwrap_or("?"));
    s.push_str("<style>\n");
    s.push_str("body{font-family:'SF Pro','Helvetica Neue',sans-serif;background:#f5f5f7;color:#1d1d1f;margin:0;padding:24px;max-width:1200px;margin:auto;}\n");
    s.push_str("h1{color:#1d1d1f;font-size:28px;margin:0 0 8px;}\n");
    s.push_str("h2{color:#1d1d1f;font-size:20px;margin:24px 0 12px;border-bottom:1px solid #d2d2d7;padding-bottom:8px;}\n");
    s.push_str(".meta{color:#6e6e73;font-size:14px;margin-bottom:24px;}\n");
    s.push_str(".stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:16px;margin:24px 0;}\n");
    s.push_str(".stat{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05);}\n");
    s.push_str(".stat-label{font-size:12px;color:#6e6e73;text-transform:uppercase;letter-spacing:0.5px;}\n");
    s.push_str(".stat-value{font-size:28px;font-weight:600;color:#1d1d1f;margin-top:4px;}\n");
    s.push_str(".track{background:#fff;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.05);display:flex;gap:16px;align-items:flex-start;}\n");
    s.push_str(".track-info{flex:1;}\n");
    s.push_str(".track-header{display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap;}\n");
    s.push_str(".track-pill{display:inline-block;color:#fff;border-radius:6px;padding:4px 10px;font-weight:600;font-size:13px;}\n");
    s.push_str(".track-time{color:#6e6e73;font-size:13px;}\n");
    s.push_str(".track-bar{height:6px;background:#e5e5ea;border-radius:3px;overflow:hidden;margin:8px 0;}\n");
    s.push_str(".track-bar-fill{height:100%;}\n");
    s.push_str(".track-frames{margin-top:8px;font-size:11px;color:#6e6e73;}\n");
    s.push_str(".track-frames code{background:#f5f5f7;padding:2px 6px;border-radius:3px;margin:0 3px;}\n");
    s.push_str(".thumb{width:180px;height:135px;background:#1d1d1f;border-radius:8px;flex-shrink:0;display:flex;align-items:center;justify-content:center;overflow:hidden;}\n");
    s.push_str(".thumb img{width:100%;height:100%;object-fit:cover;display:block;}\n");
    s.push_str(".thumb-empty{color:#6e6e73;font-size:11px;text-align:center;padding:8px;}\n");
    s.push_str("a{color:#0071e3;text-decoration:none;}\n");
    s.push_str("a:hover{text-decoration:underline;}\n");
    s.push_str("</style>\n</head>\n<body>\n");

    let _ = writeln!(s, "<h1>📹 {}</h1>", report.video_path.file_name().and_then(|s| s.to_str()).unwrap_or("video"));
    s.push_str("<div class=\"meta\">");
    let _ = writeln!(s, "路径: {}<br>", report.video_path.display());
    let _ = writeln!(s, "抽帧率: {} fps", report.fps);
    s.push_str("</div>\n");

    let total_frames = report.records.len();
    let total_faces: u32 = report.records.iter().map(|r| r.face_count).sum();
    let total_duration = report.records.iter().map(|r| r.timestamp_secs).fold(0.0_f64, |a, b| a.max(b));
    s.push_str("<div class=\"stats\">");
    let _ = writeln!(s, "<div class=\"stat\"><div class=\"stat-label\">命中帧</div><div class=\"stat-value\">{}</div></div>", total_frames);
    let _ = writeln!(s, "<div class=\"stat\"><div class=\"stat-label\">人脸总数</div><div class=\"stat-value\">{}</div></div>", total_faces);
    let _ = writeln!(s, "<div class=\"stat\"><div class=\"stat-label\">视频时长</div><div class=\"stat-value\">{:.0}s</div></div>", total_duration);
    if let Some(tracks) = report.tracks {
        let _ = writeln!(s, "<div class=\"stat\"><div class=\"stat-label\">独立人脸</div><div class=\"stat-value\">{}</div></div>", tracks.len());
    }
    s.push_str("</div>\n");

    if let Some(tracks) = report.tracks {
        s.push_str("<h2>👤 人脸时间轴</h2>");
        for t in tracks.iter() {
            let (pr, pg, pb) = PALETTE[(t.id as usize) % PALETTE.len()];
            let color = format!("rgb({},{},{})", pr, pg, pb);
            s.push_str("<div class=\"track\">");
            // 缩略图
            s.push_str("<div class=\"thumb\">");
            let mut found = false;
            for tn in &report.thumbnails {
                if tn.face_id == t.id {
                    s.push_str(&format!("<img src=\"data:image/png;base64,{}\" alt=\"face {}\">", tn.png_base64, t.id));
                    found = true;
                    break;
                }
            }
            if !found {
                s.push_str("<div class=\"thumb-empty\">无缩略图<br>face_id=");
                let _ = write!(s, "{}", t.id);
                s.push_str("</div>");
            }
            s.push_str("</div>");
            // 信息
            s.push_str("<div class=\"track-info\">");
            s.push_str("<div class=\"track-header\">");
            let _ = writeln!(s, "<span class=\"track-pill\" style=\"background:{}\">face_id = {}</span>", color, t.id);
            let _ = writeln!(s, "<span class=\"track-time\">{:.1}s - {:.1}s ({:.1}s)</span>", t.first_ts, t.last_ts, t.last_ts - t.first_ts);
            let _ = writeln!(s, "<span class=\"track-time\">{} 帧</span>", t.frame_count);
            s.push_str("</div>");
            if total_duration > 0.0 {
                let start_pct = (t.first_ts / total_duration) * 100.0;
                let width_pct = ((t.last_ts - t.first_ts) / total_duration) * 100.0;
                s.push_str("<div class=\"track-bar\">");
                let _ = writeln!(s, "<div class=\"track-bar-fill\" style=\"margin-left:{}%;width:{}%;background:{}\"></div>", start_pct, width_pct, color);
                s.push_str("</div>");
            }
            s.push_str("<div class=\"track-frames\">");
            let samples: Vec<_> = t.frames.iter().step_by((t.frames.len() / 5).max(1)).take(5).collect();
            for f in samples {
                let _ = writeln!(s, "<code>{:.1}s</code>", f.timestamp_secs);
            }
            s.push_str("</div></div></div>");
        }
    }

    s.push_str("<h2>📋 帧清单 (manifest.txt)</h2>");
    s.push_str("<div class=\"meta\">");
    let _ = writeln!(s, "{} 帧已保存.", report.records.len());
    s.push_str("</div>\n");

    s.push_str("<h2>🔗 关联文件</h2>");
    s.push_str("<ul>");
    s.push_str("<li><a href=\"manifest.txt\">manifest.txt</a> — 完整帧清单 (Tab 分隔)</li>");
    if report.tracks.is_some() {
        s.push_str("<li><a href=\"tracks.json\">tracks.json</a> — 人脸聚类 JSON</li>");
    }
    s.push_str("</ul>");

    let _ = writeln!(s, "<footer style=\"margin-top:48px;padding-top:24px;border-top:1px solid #d2d2d7;color:#6e6e73;font-size:12px\">");
    s.push_str("由 rs-face 生成 · 零依赖 Rust 人脸检测与跟踪");
    s.push_str("</footer>\n");

    s.push_str("</body>\n</html>\n");
    s
}

pub fn write(report: &HtmlReport, out_path: &Path) -> Result<(), std::io::Error> {
    std::fs::write(out_path, render(report))
}

/// 提取 track 代表帧的人脸缩略图 (从原始路径读图, 按 sample_box 抠)
pub fn extract_thumbnails(
    track: &crate::tracker::FaceTrack,
    source_image: &Image,
) -> String {
    // 抠脸 → resize 到 180x135 → PNG → base64
    let box_ = track.sample_box;
    let x0 = box_[0].max(0) as usize;
    let y0 = box_[1].max(0) as usize;
    let x1 = (box_[0] + box_[2]).min(source_image.width as i32) as usize;
    let y1 = (box_[1] + box_[3]).min(source_image.height as i32) as usize;
    if x1 <= x0 || y1 <= y0 {
        return String::new();
    }
    let crop = source_image.crop(x0, y0, x1 - x0, y1 - y0);
    let resized = crop.resize_bilinear(180, 135);
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    if crate::png::write_png_to_writer(&resized, &mut cursor).is_err() {
        return String::new();
    }
    // 手写 base64 编码 (零依赖)
    base64_encode(&buf)
}

/// 简化的 base64 编码 (RFC 4648, 标准表)
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let b0 = data[i];
        let b1 = data[i + 1];
        let b2 = data[i + 2];
        s.push(TABLE[(b0 >> 2) as usize] as char);
        s.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        s.push(TABLE[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char);
        s.push(TABLE[(b2 & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let b0 = data[i];
        s.push(TABLE[(b0 >> 2) as usize] as char);
        s.push(TABLE[((b0 & 0x03) << 4) as usize] as char);
        s.push('=');
        s.push('=');
    } else if rem == 2 {
        let b0 = data[i];
        let b1 = data[i + 1];
        s.push(TABLE[(b0 >> 2) as usize] as char);
        s.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        s.push(TABLE[((b1 & 0x0F) << 2) as usize] as char);
        s.push('=');
    }
    s
}