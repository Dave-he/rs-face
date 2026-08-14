// HTML 报告生成器 (单文件 HTML, 浏览器直接打开)
//
// 内容:
// - 每个 face_id 一张时间轴卡片, 显示代表帧缩略图
// - 视频元数据 (fps, 总帧数, 命中)
// - 直方图时间分布
// - 链接到 manifest.txt + tracks.json

use crate::image::Image;
use crate::saver::FaceRecord;
use std::fmt::Write as _;
use std::path::Path;

pub struct HtmlReport<'a> {
    pub video_path: &'a Path,
    pub fps: f64,
    pub records: &'a [FaceRecord],
    pub tracks: Option<&'a [crate::tracker::FaceTrack]>,
    pub cover_thumb: Option<&'a Image>,
}

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
    s.push_str(".track{background:#fff;border-radius:12px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.05);}\n");
    s.push_str(".track-header{display:flex;align-items:center;gap:16px;margin-bottom:12px;}\n");
    s.push_str(".track-id{background:#0071e3;color:#fff;border-radius:8px;padding:4px 12px;font-weight:600;font-size:14px;}\n");
    s.push_str(".track-time{color:#6e6e73;font-size:14px;}\n");
    s.push_str(".track-bar{height:8px;background:#e5e5ea;border-radius:4px;overflow:hidden;}\n");
    s.push_str(".track-bar-fill{height:100%;background:#0071e3;}\n");
    s.push_str("a{color:#0071e3;text-decoration:none;}\n");
    s.push_str("a:hover{text-decoration:underline;}\n");
    s.push_str(".thumb{width:120px;height:90px;background:#1d1d1f;border-radius:6px;display:inline-block;margin:4px;overflow:hidden;color:#fff;font-size:11px;display:flex;align-items:center;justify-content:center;}\n");
    s.push_str("</style>\n</head>\n<body>\n");

    let _ = writeln!(s, "<h1>📹 {}</h1>", report.video_path.file_name().and_then(|s| s.to_str()).unwrap_or("video"));
    s.push_str("<div class=\"meta\">");
    let _ = writeln!(s, "路径: {}<br>", report.video_path.display());
    let _ = writeln!(s, "抽帧率: {} fps", report.fps);
    s.push_str("</div>\n");

    // 统计卡片
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

    // 时间轴: 每个 face_id 一格
    if let Some(tracks) = report.tracks {
        s.push_str("<h2>👤 人脸时间轴</h2>");
        for t in tracks.iter() {
            let _ = writeln!(s, "<div class=\"track\">");
            let _ = writeln!(s, "<div class=\"track-header\">");
            let _ = writeln!(s, "<span class=\"track-id\">face_id = {}</span>", t.id);
            let _ = writeln!(s, "<span class=\"track-time\">{:.1}s - {:.1}s ({:.1}s)</span>", t.first_ts, t.last_ts, t.last_ts - t.first_ts);
            let _ = writeln!(s, "<span class=\"track-time\">{} 帧</span>", t.frame_count);
            s.push_str("</div>");
            // 时间轴
            if total_duration > 0.0 {
                let start_pct = (t.first_ts / total_duration) * 100.0;
                let width_pct = ((t.last_ts - t.first_ts) / total_duration) * 100.0;
                s.push_str("<div class=\"track-bar\">");
                let _ = writeln!(s, "<div class=\"track-bar-fill\" style=\"margin-left:{}%;width:{}%\"></div>", start_pct, width_pct);
                s.push_str("</div>");
            }
            // 帧列表 (前 5 个)
            s.push_str("<div style=\"margin-top:8px;font-size:12px;color:#6e6e73\">");
            let samples: Vec<_> = t.frames.iter().step_by((t.frames.len() / 5).max(1)).take(5).collect();
            for f in samples {
                let _ = writeln!(s, "<code style=\"background:#f5f5f7;padding:2px 6px;border-radius:4px;margin:0 4px\">{:.1}s</code>", f.timestamp_secs);
            }
            s.push_str("</div>");
            s.push_str("</div>");
        }
    }

    // 帧清单
    s.push_str("<h2>📋 帧清单 (manifest.txt)</h2>");
    s.push_str("<div class=\"meta\">");
    let _ = writeln!(s, "{} 帧已保存.", report.records.len());
    s.push_str("</div>");

    // 链接
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
