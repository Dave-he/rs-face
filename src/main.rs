mod align;
mod args;
mod benchmark;
mod cascade;
mod detector;
mod faces;
mod hog_svm;
mod http;
mod image;
mod imgproc;
mod linalg;
mod png;
mod ppm;
mod recognition;
mod report_html;
mod saver;
mod tracker;
mod video;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone)]
pub struct DetectorOpts {
    pub min_size: u32,
    pub max_size: u32,
    pub scale_factor: f32,
    pub min_neighbors: u32,
    pub step: u32,
    pub save_crops: bool,
    pub padding_ratio: f32,
    pub flip_detect: bool,
    pub hog_svm_path: Option<std::path::PathBuf>,
    pub hog_threshold: f64,
    /// 相邻帧人脸 IoU 高于此阈值视为重复, 不写出。0 表示不去重。
    pub dedup_iou: f32,
    /// 是否开启人脸跟踪 (LBPH 聚类, 写 tracks.json)。
    pub track: bool,
    /// 人脸聚类卡方距离阈值。
    pub track_threshold: f64,
    /// 仅输出每个 track 的代表帧 (需要 --track), 节省 90% 空间。
    pub key_frames_only: bool,
    /// 裁剪并对齐人脸后再保存 (--save-crops 时生效)。
    pub align_crops: bool,
    /// 清晰度过滤阈值 (Laplacian 方差, 0=不过滤)。
    pub quality_filter: f64,
    /// 跨视频人脸合并: 加载先前 tracks.json 作为画廊, 合并同人脸。
    pub prior_tracks: Option<std::path::PathBuf>,
}

fn main() -> ExitCode {
    let cli = match args::parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("参数错误: {}", e);
            eprintln!("使用 --help 查看用法");
            return ExitCode::from(2);
        }
    };
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[rs-face] 失败: {}", err_chain(&*e));
            ExitCode::from(1)
        }
    }
}

fn err_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(src) = cur {
        s.push_str(": ");
        s.push_str(&src.to_string());
        cur = src.source();
    }
    s
}

fn run(cmd: args::Command) -> Result<(), BoxError> {
    let started = Instant::now();
    match cmd {
        args::Command::Detect(o) => cmd_detect(o, started)?,
        args::Command::Train(o) => cmd_train(o, started)?,
        args::Command::Recognize(o) => cmd_recognize(o, started)?,
        args::Command::Benchmark(o) => cmd_benchmark(o, started)?,
        args::Command::Info => cmd_info(),
        args::Command::Help => {}
    }
    Ok(())
}

fn cmd_info() {
    println!("rs-face - 零依赖 Rust 人脸检测与识别系统");
    println!("============================================");
    println!();
    println!("【实现算法总览】");
    println!("  ├─ 人脸检测");
    println!("  │   └─ Viola-Jones (2001)");
    println!("  │       ├─ Haar-like 特征 (边缘/线/块，支持 tilted)");
    println!("  │       ├─ 积分图 (Integral Image) O(1) 矩形求和");
    println!("  │       ├─ 平方积分图 (方差归一化)");
    println!("  │       ├─ AdaBoost 弱分类器加权投票");
    println!("  │       └─ Cascade 级联拒绝 (多阶段快速过滤)");
    println!("  │");
    println!("  ├─ 人脸对齐");
    println!("  │   ├─ 几何关键点启发式估计 (双眼/鼻尖/嘴角)");
    println!("  │   ├─ 仿射变换 (旋转 + 缩放 + 平移)");
    println!("  │   └─ 中心裁剪 + 外扩 padding");
    println!("  │");
    println!("  ├─ 特征提取");
    println!("  │   ├─ Eigenfaces / PCA (主成分分析, Turk & Pentland 1991)");
    println!("  │   │   ├─ 协方差矩阵构建");
    println!("  │   │   ├─ Jacobi 特征值分解 (对称矩阵)");
    println!("  │   │   └─ 高维规避 (S^T S 技巧，样本数<<像素数)");
    println!("  │   ├─ Fisherfaces / LDA (线性判别分析, Belhumeur 1997)");
    println!("  │   │   ├─ 类间散度 SB / 类内散度 SW");
    println!("  │   │   ├─ 先 PCA 降维再 LDA 投影");
    println!("  │   │   └─ 高斯消元法矩阵求逆");
    println!("  │   └─ LBPH (局部二值模式直方图, Ahonen 2006)");
    println!("  │       ├─ 圆形 8 邻域 LBP 编码");
    println!("  │       ├─ 8x8 网格分块");
    println!("  │       └─ 归一化直方图拼接 (8x8x256 = 16384 维)");
    println!("  │");
    println!("  ├─ 匹配/识别");
    println!("  │   ├─ 距离度量: 欧氏 / 余弦 / 卡方 (Chi-Square)");
    println!("  │   ├─ KNN 最近邻分类器 (带置信度)");
    println!("  │   ├─ 线性 SVM (Hinge Loss + SGD)");
    println!("  │   └─ 多分类 One-vs-Rest");
    println!("  │");
    println!("  ├─ 图像处理 (零依赖纯 std 实现)");
    println!("  │   ├─ PPM/PGM 读写");
    println!("  │   ├─ PNG 写入 (zlib store + ADLER32 + CRC32)");
    println!("  │   ├─ 高斯模糊 (可分离卷积)");
    println!("  │   ├─ Sobel 边缘检测");
    println!("  │   ├─ 直方图均衡化");
    println!("  │   ├─ 双线性 / 最近邻缩放");
    println!("  │   ├─ Gamma 校正");
    println!("  │   ├─ 图像金字塔 (多尺度检测)");
    println!("  │   ├─ NMS 非极大值抑制");
    println!("  │   └─ 分组矩形 (检测框聚类)");
    println!("  │");
    println!("  ├─ 网络 (零依赖纯 std 实现)");
    println!("  │   ├─ HTTP/1.0 客户端 (TcpStream)");
    println!("  │   └─ 解析 OpenCV Haar Cascade XML");
    println!("  │       (手写非验证 SAX 风格 XML parser)");
    println!("  │");
    println!("  └─ 视频 (系统 ffmpeg 子进程, 非 Rust crate)");
    println!("      └─ 按 fps 抽帧 -> PGM");
    println!();
    println!("【零依赖说明】");
    println!("  Cargo.toml [dependencies] 为空。");
    println!("  所有图像/矩阵/算法/网络/XML/PNG 都用 std 手写。");
    println!("  仅视频抽帧调用系统 ffmpeg 命令行工具。");
    println!();
    println!("【参考论文】");
    println!("  1. Viola & Jones, CVPR 2001 - Rapid Object Detection using a Boosted Cascade");
    println!("  2. Turk & Pentland, JCN 1991 - Eigenfaces for Recognition");
    println!("  3. Belhumeur et al, PAMI 1997 - Eigenfaces vs. Fisherfaces");
    println!("  4. Ahonen et al, ECCV Workshop 2004 - Face Recognition with LBP");
    println!("  5. Lienhart & Maydt, ICIP 2002 - Extended Set of Haar-like Features");
    println!();
    println!("使用 `rs-face --help` 查看命令。");
}

fn cmd_detect(o: args::DetectOpts, started: Instant) -> Result<(), BoxError> {
    if let Some(img_path) = o.image.clone() {
        return cmd_detect_image(o, &img_path, started);
    }
    let video_path: PathBuf = match &o.input {
        Some(p) => p.clone(),
        None => {
            let url = o
                .url
                .as_ref()
                .ok_or("必须提供视频 URL 或 --input 本地文件或 --image 单图")?;
            let tmp_dir = o
                .tmp_dir
                .clone()
                .unwrap_or_else(|| std::env::temp_dir().join("rs-face"));
            std::fs::create_dir_all(&tmp_dir)?;
            println!("[detect] 下载视频: {}", url);
            http::download_to_file(url, &tmp_dir)?
        }
    };
    println!("[detect] 视频: {}", video_path.display());
    std::fs::create_dir_all(&o.output)?;
    println!("[detect] 输出目录: {}", o.output.display());
    let tmp_dir = o
        .tmp_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("rs-face"));
    let frames_dir = tmp_dir.join("frames");
    let is_dir_input = video_path.is_dir();
    if is_dir_input {
        // 输入已是帧目录, 直接复用
        println!("[detect] 输入是目录, 直接作为帧目录: {}", video_path.display());
        let frames_dir = video_path.clone();
        let cascade = load_cascade(o.cascade_path.as_deref())?;
        println!(
            "[detect] Cascade 就绪: {} stages, window={}x{}, features={}",
            cascade.stages.len(),
            cascade.window.0,
            cascade.window.1,
            cascade.features.len()
        );
        let stats = detector::detect_in_directory(
            &cascade,
            &frames_dir,
            &o.output,
            DetectorOpts {
                min_size: o.min_size,
                max_size: o.max_size,
                scale_factor: o.scale_factor,
                min_neighbors: o.min_neighbors,
                step: o.step,
                save_crops: o.save_crops,
                padding_ratio: o.padding_ratio,
                flip_detect: o.flip_detect,
                hog_svm_path: o.hog_svm_path.clone(),
                hog_threshold: o.hog_threshold,
                dedup_iou: o.dedup_iou,
                track: o.track,
                track_threshold: o.track_threshold,
                key_frames_only: o.key_frames_only,
                align_crops: o.align_crops,
                quality_filter: o.quality_filter,
                prior_tracks: o.prior_tracks.clone(),
            },
        )?;
        println!(
            "[detect] 完成: 扫描 {} 帧, 命中 {} 帧, 写入 {} 张图",
            stats.frames_scanned, stats.frames_with_faces, stats.images_written
        );
        let manifest_path = o.output.join("manifest.txt");
        write_manifest(&manifest_path, stats.records.iter())?;
        println!("[detect] 清单: {}", manifest_path.display());
        println!("[detect] 用时: {:.2}s", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if frames_dir.exists() {
        std::fs::remove_dir_all(&frames_dir)?;
    }
    std::fs::create_dir_all(&frames_dir)?;
    println!("[detect] ffmpeg 抽帧 fps={:.3}", o.fps);
    video::extract_frames_pgm(&video_path, &frames_dir, o.fps)?;
    let cascade = load_cascade(o.cascade_path.as_deref())?;
    println!(
        "[detect] Cascade 就绪: {} stages, window={}x{}, features={}",
        cascade.stages.len(),
        cascade.window.0,
        cascade.window.1,
        cascade.features.len()
    );
    let stats = detector::detect_in_directory(
        &cascade,
        &frames_dir,
        &o.output,
        DetectorOpts {
            min_size: o.min_size,
            max_size: o.max_size,
            scale_factor: o.scale_factor,
            min_neighbors: o.min_neighbors,
            step: o.step,
            save_crops: o.save_crops,
            padding_ratio: o.padding_ratio,
            flip_detect: o.flip_detect,
            hog_svm_path: o.hog_svm_path.clone(),
            hog_threshold: o.hog_threshold,
            dedup_iou: o.dedup_iou,
            track: o.track,
            track_threshold: o.track_threshold,
            key_frames_only: o.key_frames_only,
            align_crops: o.align_crops,
            quality_filter: o.quality_filter,
            prior_tracks: o.prior_tracks.clone(),
        },
    )?;
    println!(
        "[detect] 完成: 扫描 {} 帧, 命中 {} 帧, 写入 {} 张图",
        stats.frames_scanned, stats.frames_with_faces, stats.images_written
    );
    if !o.keep_frames {
        let _ = std::fs::remove_dir_all(&frames_dir);
    }
    let manifest_path = o.output.join("manifest.txt");
    write_manifest(&manifest_path, stats.records.iter())?;
    println!("[detect] 清单: {}", manifest_path.display());
    println!("[detect] 用时: {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_detect_image(o: args::DetectOpts, img_path: &Path, started: Instant) -> Result<(), BoxError> {
    let img = image::Image::load_pgm(img_path).or_else(|_| image::Image::load_ppm(img_path))?;
    println!("[detect] 图片: {} ({}x{})", img_path.display(), img.width, img.height);
    std::fs::create_dir_all(&o.output)?;
    let cascade = load_cascade(o.cascade_path.as_deref())?;
    println!(
        "[detect] Cascade: {} stages, {} features",
        cascade.stages.len(),
        cascade.features.len()
    );
    let det_opts = detector::DetectionOpts {
        min_size: o.min_size,
        max_size: o.max_size,
        scale_factor: o.scale_factor,
        min_neighbors: o.min_neighbors,
        step: o.step,
        flip_detect: o.flip_detect,
        hog_svm: None,
        hog_threshold: o.hog_threshold,
    };
    let faces = detector::detect_single_image(&cascade, &img, &det_opts);
    println!("[detect] 检测到 {} 张人脸", faces.len());
    let name = img_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let rec = saver::save_frame_with_faces(
        &img,
        &o.output,
        1,
        0.0,
        1,
        &faces,
        &[],
        o.save_crops,
        o.padding_ratio,
        o.align_crops,
    )?;
    println!("[detect] 输出: {}/{}", o.output.display(), rec.file_name);
    println!("[detect] 用时: {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn load_cascade(p: Option<&Path>) -> Result<cascade::Cascade, BoxError> {
    if let Some(p) = p {
        cascade::Cascade::load_from_xml(p)
    } else {
        cascade::Cascade::load_default().or_else(|_| {
            let alt = Path::new("data/haarcascade_frontalface_default.xml");
            if alt.exists() {
                cascade::Cascade::load_from_xml(alt)
            } else {
                Err(format!(
                    "找不到 Cascade XML: data/haarcascade_frontalface_alt2.xml。\n\
                     可从 OpenCV 仓库下载 data/haarcascades/*.xml 放到 data/ 目录,\n\
                     或用 --cascade <path> 指定。"
                )
                .into())
            }
        })
    }
}

fn cmd_train(o: args::TrainOpts, started: Instant) -> Result<(), BoxError> {
    println!("[train] 数据集: {}", o.dataset.display());
    println!("[train] 算法: {:?}  尺寸: {}x{}  成分数: {}",
        o.algorithm, o.size.0, o.size.1, o.num_components);
    let (data_mat, labels, names) = align::load_face_dataset(&o.dataset, o.size)?;
    println!(
        "[train] 加载样本: {} 张, {} 类 {:?}",
        data_mat.rows,
        names.len(),
        names
    );
    match o.algorithm {
        args::Algorithm::Eigenfaces => {
            let nc = o.num_components.min(data_mat.rows);
            let model = faces::EigenfacesModel::train(&data_mat, &labels, &names, nc)?;
            println!(
                "[train] Eigenfaces: 得到 {} 主成分 (前 5 特征值: {:.4}, {:.4}, {:.4}, {:.4}, {:.4})",
                model.eigenvectors.cols,
                model.eigenvalues.get(0).copied().unwrap_or(0.0),
                model.eigenvalues.get(1).copied().unwrap_or(0.0),
                model.eigenvalues.get(2).copied().unwrap_or(0.0),
                model.eigenvalues.get(3).copied().unwrap_or(0.0),
                model.eigenvalues.get(4).copied().unwrap_or(0.0),
            );
            model.save(&o.model_out)?;
        }
        args::Algorithm::Fisherfaces => {
            let classes = labels.iter().cloned().collect::<std::collections::BTreeSet<_>>();
            let nc = o.num_components.min(classes.len() - 1).max(1);
            let model = faces::FisherfacesModel::train(&data_mat, &labels, &names, nc)?;
            println!(
                "[train] Fisherfaces: 得到 {} 判别向量 (LDA 输出维度 <= C-1={})",
                model.eigenvectors.cols,
                classes.len() - 1
            );
            model.save(&o.model_out)?;
        }
        args::Algorithm::LBPH => {
            let n = data_mat.rows;
            let mut widths = Vec::with_capacity(n);
            let mut heights = Vec::with_capacity(n);
            let mut imgs: Vec<Vec<f64>> = Vec::with_capacity(n);
            for r in 0..n {
                let v: Vec<f64> = (0..data_mat.cols).map(|c| data_mat.get(r, c)).collect();
                imgs.push(v);
                widths.push(o.size.0);
                heights.push(o.size.1);
            }
            let model = recognition::LBPHModel::train(
                &imgs, &widths, &heights, &labels, &names, 8, 8)?;
            println!(
                "[train] LBPH: grid=8x8, 每张脸 {} 维直方图",
                model.histograms.get(0).map(|h| h.len()).unwrap_or(0)
            );
            model.save(&o.model_out)?;
        }
    }
    println!("[train] 模型已保存: {}", o.model_out.display());
    println!("[train] 用时: {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_recognize(o: args::RecognizeOpts, started: Instant) -> Result<(), BoxError> {
    println!("[recognize] 加载模型: {}", o.model.display());
    let model = recognition::FaceModel::load(&o.model)?;
    let raw_img = image::Image::load_pgm(&o.input).or_else(|_| image::Image::load_ppm(&o.input))?;
    println!("[recognize] 输入: {} ({}x{})", o.input.display(), raw_img.width, raw_img.height);
    let face_rects: Vec<image::Rect> = if let Some(ref cpath) = o.cascade_path {
        let cascade = cascade::Cascade::load_from_xml(cpath)?;
        let det_opts = detector::DetectionOpts::default();
        detector::detect_single_image(&cascade, &raw_img, &det_opts)
    } else {
        vec![]
    };
    println!("[recognize] 定位人脸: {}", if face_rects.is_empty() { "整图".to_string() } else { format!("{} 张", face_rects.len()) });
    let mut result_img = raw_img.clone();
    let faces_to_rec: Vec<Option<image::Rect>> = if face_rects.is_empty() {
        vec![None]
    } else {
        face_rects.iter().map(|r| Some(*r)).collect()
    };
    for (i, fopt) in faces_to_rec.iter().enumerate() {
        let vec = align::preprocess_for_recognition(&raw_img, fopt.as_ref(), o.size);
        let (lbl, conf, name) = model.predict_raw(&vec, o.size.0, o.size.1);
        let conf = if let Some(_thr) = o.threshold {
            conf
        } else {
            conf
        };
        let display_name = name.clone().unwrap_or_else(|| "<未知>".to_string());
        match fopt {
            Some(r) => {
                println!(
                    "[recognize] 人脸 #{}: bbox=({},{} {}x{})  类别={}({})  置信度={:.3}  匹配={}",
                    i + 1,
                    r.x, r.y, r.w, r.h,
                    lbl, display_name, conf,
                    if name.is_some() { "是" } else { "否 (低于阈值)" }
                );
                saver::draw_label(&mut result_img, r.x, r.y, &format!("{} {:.0}%", display_name, conf * 100.0));
            }
            None => {
                println!(
                    "[recognize] 整图: 类别={}({})  置信度={:.3}  匹配={}",
                    lbl, display_name, conf,
                    if name.is_some() { "是" } else { "否 (低于阈值)" }
                );
            }
        }
    }
    if let Some(out_dir) = o.output {
        std::fs::create_dir_all(&out_dir)?;
        let stem = o.input.file_stem().and_then(|s| s.to_str()).unwrap_or("rec");
        let out_path = out_dir.join(format!("{}_recognized.png", stem));
        result_img.save_png(&out_path)?;
        println!("[recognize] 标注图: {}", out_path.display());
    }
    println!("[recognize] 用时: {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_benchmark(o: args::BenchmarkOpts, started: Instant) -> Result<(), BoxError> {
    println!("[benchmark] 数据集: {}", o.dataset.display());
    let mode = benchmark::Mode::from_str(&o.mode)?;
    let dataset = benchmark::load_dataset(&o.dataset, o.size)?;
    let dataset_name = o.dataset.file_name()
        .and_then(|s| s.to_str()).unwrap_or("dataset").to_string();
    println!("[benchmark] 加载: {} 张 ({} 类), 平均每类 {:.1} 张",
        dataset.vectors.len(),
        dataset.names.len(),
        dataset.vectors.len() as f64 / dataset.names.len().max(1) as f64);

    let algorithms: Vec<benchmark::Algorithm> = if o.algorithm.eq_ignore_ascii_case("all") {
        vec![
            benchmark::Algorithm::Eigenfaces,
            benchmark::Algorithm::Fisherfaces,
            benchmark::Algorithm::LBPH,
        ]
    } else {
        vec![benchmark::Algorithm::from_str(&o.algorithm)?]
    };
    println!("[benchmark] 算法: {:?}  模式: {:?}  折数: {}  配对上限: {}",
        algorithms.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>(),
        mode, o.folds, o.max_pairs);

    let out_path = o.out.clone().unwrap_or_else(|| {
        std::path::PathBuf::from("./BENCH_RECOGNITION.local.md")
    });
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut combined_id: Vec<benchmark::IdentificationReport> = Vec::new();
    let mut combined_ver: Vec<benchmark::VerificationReport> = Vec::new();
    for alg in &algorithms {
        let opts = benchmark::Options {
            mode,
            algorithm: *alg,
            folds: o.folds,
            max_pairs: o.max_pairs,
            seed: o.seed,
            size: o.size,
            train_per_fold: 0,
        };
        let id_report = if matches!(mode, benchmark::Mode::Identification | benchmark::Mode::Both) {
            if dataset.names.len() < 2 {
                println!("[benchmark] 跳过 identification: 类别数 < 2");
                None
            } else {
                let mut r = benchmark::run_identification(&dataset, &opts)?;
                r.dataset = dataset_name.clone();
                Some(r)
            }
        } else { None };

        let ver_report = if matches!(mode, benchmark::Mode::Verification | benchmark::Mode::Both) {
            if dataset.vectors.len() < 2 {
                println!("[benchmark] 跳过 verification: 样本数 < 2");
                None
            } else {
                let mut r = benchmark::run_verification(&dataset, &opts)?;
                r.dataset = dataset_name.clone();
                Some(r)
            }
        } else { None };

        if let Some(id) = &id_report {
            println!(
                "[benchmark] identification ({:?}): top-1={:.2}%  top-5={:.2}%  ({} folds)",
                alg, id.top1 * 100.0, id.top5 * 100.0, id.folds
            );
            combined_id.push(id.clone());
        }
        if let Some(ver) = &ver_report {
            println!(
                "[benchmark] verification ({:?}): AUC={:.4}  EER={:.4}  best_acc={:.2}%  ({} pairs)",
                alg, ver.auc, ver.eer, ver.best_accuracy * 100.0, ver.n_pairs
            );
            combined_ver.push(ver.clone());
        }
    }

    benchmark::write_combined_markdown(&out_path, &combined_id, &combined_ver, &dataset_name, &o.dataset)?;
    println!("[benchmark] 报告: {}", out_path.display());
    println!("[benchmark] 用时: {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn write_manifest<'a, I: IntoIterator<Item = &'a saver::FaceRecord>>(
    path: &Path,
    records: I,
) -> Result<(), BoxError> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "# rs-face manifest")?;
    writeln!(
        f,
        "# format: index<TAB>timestamp_secs<TAB>frame_index<TAB>file_name<TAB>faces<TAB>face_ids<TAB>x,y,w,h;..."
    )?;
    for r in records {
        let boxes = r
            .boxes
            .iter()
            .map(|[x, y, w, h]| format!("{},{},{},{}", x, y, w, h))
            .collect::<Vec<_>>()
            .join(";");
        let face_ids = if r.face_ids.is_empty() {
            "-".to_string()
        } else {
            r.face_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",")
        };
        writeln!(
            f,
            "{}\t{:.3}\t{}\t{}\t{}\t{}\t{}",
            r.index, r.timestamp_secs, r.frame_index, r.file_name, r.face_count, face_ids, boxes
        )?;
    }
    Ok(())
}
