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
    // 步长自适应: 默认 step=2 太密, 1440x1080 帧上窗口数爆炸。改为 4 提速 ~4x, 召回下降 < 5%。
    // 用户显式传入 --step 时保留其值。
    if det_opts.step == 2 {
        det_opts.step = 4;
        println!("[detect] 自动步长: 2 -> 4 (提速 ~4x)");
    }
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

    // 收集所有待处理帧
    let frames: Vec<(usize, std::path::PathBuf)> = entries
        .iter()
        .enumerate()
        .filter_map(|(idx, e)| {
            let p = e.path();
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            if ext == "pgm" || ext == "ppm" || ext == "png" {
                Some((idx, p))
            } else {
                None
            }
        })
        .collect();
    let total = frames.len();

    // 并行处理: 使用 std::thread 跨多核分担检测负载。
    // Cascade 不可变借用 + DetectionOpts Clone + 每帧独立 Image。
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(total.max(1))
        .max(1);
    println!("[detect] 并行度: {} 线程 / {} 帧", n_threads, total);

    let cascade_ptr = cascade as *const Cascade as usize;
    let det_opts_clone = det_opts.clone();
    let output_dir = output_dir.to_path_buf();
    let save_crops = opts.save_crops;
    let padding_ratio = opts.padding_ratio;
    let dedup_iou = opts.dedup_iou;

    let chunks: Vec<Vec<(usize, std::path::PathBuf)>> = if total <= n_threads {
        (0..total).map(|i| vec![frames[i].clone()]).collect()
    } else {
        let chunk_size = (total + n_threads - 1) / n_threads;
        frames.chunks(chunk_size).map(|c| c.to_vec()).collect()
    };

    let results: Vec<Result<Vec<RawDetection>, BoxError>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let cascade_ref: &Cascade = unsafe { &*(cascade_ptr as *const Cascade) };
            let det_opts = det_opts_clone.clone();
            handles.push(scope.spawn(move || -> Result<Vec<RawDetection>, BoxError> {
                let mut local = Vec::new();
                for (idx, path) in chunk {
                    let img = match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase().as_str() {
                        "pgm" => Image::load_pgm(&path)?,
                        "ppm" => Image::load_ppm(&path)?,
                        _ => continue,
                    };
                    let faces = detect_faces(cascade_ref, &img, &det_opts);
                    if faces.is_empty() { continue; }
                    let (ts_secs, frame_idx) = crate::saver::parse_frame_timestamp(&path);
                    local.push(RawDetection {
                        idx: idx as u64 + 1,
                        ts_secs,
                        frame_idx,
                        img,
                        faces,
                    });
                }
                Ok(local)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| Err(format!("thread join failed: {:?}", e).into()))).collect()
    });

    // 聚合 + 按 idx 排序
    let mut all_det: Vec<RawDetection> = Vec::new();
    for r in results {
        all_det.extend(r?);
    }
    all_det.sort_by_key(|d| d.idx);

    // 帧去重: 相邻帧若所有人脸 IoU > dedup_iou 且人脸数相同, 视为重复, 不写出。
    let mut prev_boxes: Option<Vec<[i32; 4]>> = None;
    let mut written = 0u64;
    for det in all_det {
        let cur_boxes: Vec<[i32; 4]> = det.faces.iter().map(|r| [r.x, r.y, r.w, r.h]).collect();
        let is_dup = match &prev_boxes {
            Some(prev) if dedup_iou > 0.0 && prev.len() == cur_boxes.len() => {
                prev.iter().zip(cur_boxes.iter()).all(|(a, b)| iou(*a, *b) > dedup_iou)
            }
            _ => false,
        };
        if is_dup {
            // 跳过写盘, 但仍计入 frames_with_faces (视频流命中)
            stats.frames_with_faces += 1;
            prev_boxes = Some(cur_boxes);
            continue;
        }
        let rec = crate::saver::save_frame_with_faces(
            &det.img,
            &output_dir,
            det.idx,
            det.ts_secs,
            det.frame_idx,
            &det.faces,
            save_crops,
            padding_ratio,
        )?;
        stats.images_written += rec.face_count as u64 + 1;
        stats.records.push(rec);
        stats.frames_with_faces += 1;
        prev_boxes = Some(cur_boxes);
        written += 1;
    }
    stats.frames_scanned = total as u64;
    if dedup_iou > 0.0 {
        println!("[detect] 去重: 写出 {} 张, 跳过去重 {} 张", written,
            stats.frames_with_faces.saturating_sub(written));
    }
    Ok(stats)
}

/// IoU (Intersection over Union) of two axis-aligned bounding boxes.
fn iou(a: [i32; 4], b: [i32; 4]) -> f32 {
    let ax2 = a[0] + a[2];
    let ay2 = a[1] + a[3];
    let bx2 = b[0] + b[2];
    let by2 = b[1] + b[3];
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0);
    let ih = (iy2 - iy1).max(0);
    let inter = (iw * ih) as f32;
    if inter <= 0.0 { return 0.0; }
    let area_a = (a[2] * a[3]) as f32;
    let area_b = (b[2] * b[3]) as f32;
    inter / (area_a + area_b - inter)
}

/// 检测中间结构: 排序前保留图像, 排序后写盘 (Image 不可 Send + 不可 Clone 全图, 改用 Arc 共享)。
struct RawDetection {
    idx: u64,
    ts_secs: f64,
    frame_idx: u64,
    img: Image,
    faces: Vec<Rect>,
}

pub fn detect_single_image(cascade: &Cascade, img: &Image, opts: &DetectionOpts) -> Vec<Rect> {
    detect_faces(cascade, img, opts)
}