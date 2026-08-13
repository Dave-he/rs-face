use crate::cascade::Cascade;
use crate::hog_svm::{self, HogSvmDetector};
use crate::image::{BoxError, Image, Rect};
use crate::imgproc::{IntegralImage, group_rectangles};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DetectionOpts {
    pub min_size: u32,
    pub max_size: u32,
    pub scale_factor: f32,
    pub min_neighbors: u32,
    pub step: u32,
    /// 是否在水平翻转图像上再做一次检测, 用于捕获侧脸 (在镜像图像中变成正面)。
    pub flip_detect: bool,
    /// 可选 HOG-SVM 第二阶段检测器。
    pub hog_svm: Option<HogSvmDetector>,
    /// HOG-SVM 检测阈值, 越大越严格。
    pub hog_threshold: f64,
}

impl Default for DetectionOpts {
    fn default() -> Self {
        Self {
            min_size: 40,
            max_size: 400,
            scale_factor: 1.25,
            min_neighbors: 3,
            step: 2,
            flip_detect: false,
            hog_svm: None,
            hog_threshold: 0.0,
        }
    }
}

pub fn detect_faces(cascade: &Cascade, img: &Image, opts: &DetectionOpts) -> Vec<Rect> {
    let ii = IntegralImage::from_image(img);
    let mut raw = cascade.detect(&ii, opts.min_size, opts.max_size, opts.scale_factor, opts.step);
    let mut boxes: Vec<Rect> = raw.drain(..).map(|(r, _)| r).collect();

    if opts.flip_detect {
        let flipped = flip_horizontal(img);
        let ii2 = IntegralImage::from_image(&flipped);
        let raw2 = cascade.detect(&ii2, opts.min_size, opts.max_size, opts.scale_factor, opts.step);
        for (r, _) in raw2 {
            // 反算回原图坐标
            let nx = img.width as i32 - r.x - r.w;
            boxes.push(Rect::new(nx.max(0), r.y, r.w, r.h));
        }
    }

    let mut merged = group_rectangles(&boxes, opts.min_neighbors);

    if let Some(ref hog) = opts.hog_svm {
        let hog_boxes = hog.detect(img, opts.min_size, opts.max_size, opts.scale_factor, opts.step, opts.hog_threshold);
        let hog_rects: Vec<Rect> = hog_boxes.iter().map(|(r, _)| *r).collect();
        merged = hog_svm::merge_detections(&merged, &hog_rects, 0.3);
    }

    merged
}

/// 水平翻转图像。
pub fn flip_horizontal(img: &Image) -> Image {
    let mut out = img.clone();
    let w = img.width;
    let h = img.height;
    let ch = img.channels;
    for y in 0..h {
        for x in 0..(w / 2) {
            let mx = w - 1 - x;
            for c in 0..ch {
                let a = (y * w + x) * ch + c;
                let b = (y * w + mx) * ch + c;
                out.data.swap(a, b);
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Stats {
    pub frames_scanned: u64,
    pub frames_with_faces: u64,
    pub images_written: u64,
    pub records: Vec<crate::saver::FaceRecord>,
}

pub fn detect_in_directory(
    cascade: &Cascade,
    frames_dir: &Path,
    output_dir: &Path,
    opts: crate::DetectorOpts,
) -> Result<Stats, BoxError> {
    let mut entries: Vec<_> = std::fs::read_dir(frames_dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.path());
    let mut stats = Stats {
        frames_scanned: 0,
        frames_with_faces: 0,
        images_written: 0,
        records: Vec::new(),
    };
    let mut det_opts = DetectionOpts {
        min_size: opts.min_size,
        max_size: opts.max_size,
        scale_factor: opts.scale_factor,
        min_neighbors: opts.min_neighbors,
        step: opts.step,
        flip_detect: false,
        hog_svm: None,
        hog_threshold: 0.0,
    };
    // 若指定 hog_svm 模型则加载
    if let Some(ref path) = opts.hog_svm_path {
        if path.exists() {
            match HogSvmDetector::load(path) {
                Ok(d) => {
                    println!("[detect] HOG-SVM 模型已加载: {}", path.display());
                    det_opts.hog_svm = Some(d);
                }
                Err(e) => println!("[detect] HOG-SVM 加载失败: {}", e),
            }
        }
    }
    det_opts.flip_detect = opts.flip_detect;
    det_opts.hog_threshold = opts.hog_threshold;

    for (idx, entry) in entries.iter().enumerate() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if ext != "pgm" && ext != "ppm" && ext != "png" { continue; }
        let img = match ext.as_str() {
            "pgm" => Image::load_pgm(&path)?,
            "ppm" => Image::load_ppm(&path)?,
            _ => continue,
        };
        stats.frames_scanned += 1;
        let faces = detect_faces(cascade, &img, &det_opts);
        if faces.is_empty() { continue; }
        stats.frames_with_faces += 1;
        let (ts_secs, frame_idx) = crate::saver::parse_frame_timestamp(&path);
        let rec = crate::saver::save_frame_with_faces(
            &img,
            output_dir,
            idx as u64 + 1,
            ts_secs,
            frame_idx,
            &faces,
            opts.save_crops,
            opts.padding_ratio,
        )?;
        stats.images_written += rec.face_count as u64 + 1;
        stats.records.push(rec);
    }
    Ok(stats)
}

pub fn detect_single_image(cascade: &Cascade, img: &Image, opts: &DetectionOpts) -> Vec<Rect> {
    detect_faces(cascade, img, opts)
}