use crate::image::BoxError;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum Command {
    Detect(DetectOpts),
    Train(TrainOpts),
    Recognize(RecognizeOpts),
    Info,
    Help,
}

#[derive(Debug, Clone)]
pub struct DetectOpts {
    pub input: Option<PathBuf>,
    pub url: Option<String>,
    pub output: PathBuf,
    pub tmp_dir: Option<PathBuf>,
    pub fps: f64,
    pub min_size: u32,
    pub max_size: u32,
    pub scale_factor: f32,
    pub min_neighbors: u32,
    pub step: u32,
    pub save_crops: bool,
    pub keep_frames: bool,
    pub padding_ratio: f32,
    pub cascade_path: Option<PathBuf>,
    pub image: Option<PathBuf>,
    /// 在水平翻转图上再做一次检测, 用于捕获镜像后变成正脸的侧脸。
    pub flip_detect: bool,
    /// 可选 HOG+SVM 第二阶段检测器 (Dalal-Triggs 2005)。
    pub hog_svm_path: Option<PathBuf>,
    /// HOG-SVM 检测阈值 (越大越严格, 0 表示不设阈值)。
    pub hog_threshold: f64,
    /// 相邻帧人脸框 IoU 高于此阈值视为重复, 不写盘。0 表示不去重 (默认 0.85)。
    pub dedup_iou: f32,
    /// 是否开启人脸跟踪 (LBPH 聚类, 写 tracks.json)。默认关闭。
    pub track: bool,
    /// 人脸聚类阈值 (卡方距离, 越小越严格)。默认 0.5。
    pub track_threshold: f64,
    /// 仅输出每个 track 的代表帧 (1 张/人脸), 节省 90% 空间。需要 --track。
    pub key_frames_only: bool,
    /// 裁剪并对齐人脸 (固定 92x112, 双眼水平) 后再保存。需要 --save-crops。
    pub align_crops: bool,
}

#[derive(Debug, Clone)]
pub struct TrainOpts {
    pub dataset: PathBuf,
    pub model_out: PathBuf,
    pub algorithm: Algorithm,
    pub num_components: usize,
    pub size: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Algorithm {
    Eigenfaces,
    Fisherfaces,
    LBPH,
}

impl Algorithm {
    pub fn from_str(s: &str) -> Result<Self, BoxError> {
        match s.to_lowercase().as_str() {
            "eigenfaces" | "eigen" | "pca" => Ok(Algorithm::Eigenfaces),
            "fisherfaces" | "fisher" | "lda" => Ok(Algorithm::Fisherfaces),
            "lbph" => Ok(Algorithm::LBPH),
            _ => Err(format!("未知算法: {} (可选: eigenfaces, fisherfaces, lbph)", s).into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecognizeOpts {
    pub model: PathBuf,
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub threshold: Option<f64>,
    pub size: (usize, usize),
    pub cascade_path: Option<PathBuf>,
}

impl Default for DetectOpts {
    fn default() -> Self {
        Self {
            input: None,
            url: None,
            output: PathBuf::from("./output"),
            tmp_dir: None,
            fps: 1.0,
            min_size: 40,
            max_size: 400,
            scale_factor: 1.25,
            min_neighbors: 3,
            step: 2,
            save_crops: false,
            keep_frames: false,
            padding_ratio: 0.25,
            cascade_path: None,
            image: None,
            flip_detect: false,
            hog_svm_path: None,
            hog_threshold: 0.0,
            dedup_iou: 0.0,
            track: false,
            track_threshold: 0.3,
            key_frames_only: false,
            align_crops: false,
        }
    }
}

pub fn parse() -> Result<Command, BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return Ok(Command::Help);
    }
    match args[0].as_str() {
        "detect" => parse_detect(&args[1..]),
        "train" => parse_train(&args[1..]),
        "recognize" | "rec" => parse_recognize(&args[1..]),
        "info" => Ok(Command::Info),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(Command::Help)
        }
        s if s.starts_with('-') => parse_detect(&args[..]),
        s => Err(format!("未知子命令: {} (可用: detect, train, recognize, info)", s).into()),
    }
}

fn parse_detect(args: &[String]) -> Result<Command, BoxError> {
    let mut opts = DetectOpts::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => { opts.input = Some(take_path(args, &mut i)?); }
            "--url" | "-u" => { opts.url = Some(take_str(args, &mut i)?); }
            "--output" | "-o" => { opts.output = take_path(args, &mut i)?; }
            "--image" => { opts.image = Some(take_path(args, &mut i)?); }
            "--tmp-dir" => { opts.tmp_dir = Some(take_path(args, &mut i)?); }
            "--fps" => { opts.fps = take_str(args, &mut i)?.parse()?; }
            "--min-size" => { opts.min_size = take_str(args, &mut i)?.parse()?; }
            "--max-size" => { opts.max_size = take_str(args, &mut i)?.parse()?; }
            "--scale" => { opts.scale_factor = take_str(args, &mut i)?.parse()?; }
            "--min-neighbors" => { opts.min_neighbors = take_str(args, &mut i)?.parse()?; }
            "--step" => { opts.step = take_str(args, &mut i)?.parse()?; }
            "--save-crops" => { opts.save_crops = true; }
            "--keep-frames" => { opts.keep_frames = true; }
            "--padding" => { opts.padding_ratio = take_str(args, &mut i)?.parse()?; }
            "--cascade" => { opts.cascade_path = Some(take_path(args, &mut i)?); }
            "--flip-detect" => { opts.flip_detect = true; }
            "--hog-svm" => { opts.hog_svm_path = Some(take_path(args, &mut i)?); }
            "--hog-threshold" => { opts.hog_threshold = take_str(args, &mut i)?.parse()?; }
            "--dedup-iou" => { opts.dedup_iou = take_str(args, &mut i)?.parse()?; }
            "--track" => { opts.track = true; }
            "--track-threshold" => { opts.track_threshold = take_str(args, &mut i)?.parse()?; }
            "--key-frames-only" => { opts.key_frames_only = true; }
            "--align-crops" => { opts.align_crops = true; }
            "--help" | "-h" => { print_help(); return Ok(Command::Help); }
            other => return Err(format!("未知 detect 参数: {}", other).into()),
        }
        i += 1;
    }
    Ok(Command::Detect(opts))
}

fn parse_train(args: &[String]) -> Result<Command, BoxError> {
    let mut dataset = PathBuf::new();
    let mut model_out = PathBuf::from("face_model.bin");
    let mut algorithm = Algorithm::Eigenfaces;
    let mut num_components = 50;
    let mut size_w = 92;
    let mut size_h = 112;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" | "-d" => { dataset = take_path(args, &mut i)?; }
            "--out" | "-m" | "-o" => { model_out = take_path(args, &mut i)?; }
            "--algorithm" | "-a" => { algorithm = Algorithm::from_str(&take_str(args, &mut i)?)?; }
            "--components" | "-k" => { num_components = take_str(args, &mut i)?.parse()?; }
            "--size" => {
                let s = take_str(args, &mut i)?;
                let parts: Vec<&str> = s.split('x').collect();
                if parts.len() == 2 {
                    size_w = parts[0].parse()?;
                    size_h = parts[1].parse()?;
                }
            }
            "--help" | "-h" => { print_help(); return Ok(Command::Help); }
            other => return Err(format!("未知 train 参数: {}", other).into()),
        }
        i += 1;
    }
    if dataset.as_os_str().is_empty() {
        return Err("train 必须指定 --dataset".into());
    }
    Ok(Command::Train(TrainOpts {
        dataset,
        model_out,
        algorithm,
        num_components,
        size: (size_w, size_h),
    }))
}

fn parse_recognize(args: &[String]) -> Result<Command, BoxError> {
    let mut model = PathBuf::new();
    let mut input = PathBuf::new();
    let mut output: Option<PathBuf> = None;
    let mut threshold: Option<f64> = None;
    let mut size_w = 92;
    let mut size_h = 112;
    let mut cascade_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => { model = take_path(args, &mut i)?; }
            "--input" | "-i" => { input = take_path(args, &mut i)?; }
            "--output" | "-o" => { output = Some(take_path(args, &mut i)?); }
            "--threshold" | "-t" => { threshold = Some(take_str(args, &mut i)?.parse()?); }
            "--size" => {
                let s = take_str(args, &mut i)?;
                let parts: Vec<&str> = s.split('x').collect();
                if parts.len() == 2 {
                    size_w = parts[0].parse()?;
                    size_h = parts[1].parse()?;
                }
            }
            "--cascade" => { cascade_path = Some(take_path(args, &mut i)?); }
            "--help" | "-h" => { print_help(); return Ok(Command::Help); }
            other => return Err(format!("未知 recognize 参数: {}", other).into()),
        }
        i += 1;
    }
    if model.as_os_str().is_empty() { return Err("recognize 必须指定 --model".into()); }
    if input.as_os_str().is_empty() { return Err("recognize 必须指定 --input".into()); }
    Ok(Command::Recognize(RecognizeOpts {
        model,
        input,
        output,
        threshold,
        size: (size_w, size_h),
        cascade_path,
    }))
}

fn take_path(args: &[String], i: &mut usize) -> Result<PathBuf, BoxError> {
    if *i + 1 >= args.len() { return Err(format!("参数 {} 需要路径值", args[*i]).into()); }
    *i += 1;
    Ok(PathBuf::from(&args[*i]))
}

fn take_str(args: &[String], i: &mut usize) -> Result<String, BoxError> {
    if *i + 1 >= args.len() { return Err(format!("参数 {} 需要值", args[*i]).into()); }
    *i += 1;
    Ok(args[*i].clone())
}

pub fn print_help() {
    println!("rs-face - 零依赖 Rust 人脸检测与识别系统");
    println!();
    println!("算法体系（全部纯 Rust std 实现）:");
    println!("  【人脸检测】 Viola-Jones (Haar-like 特征 + 积分图 + AdaBoost + Cascade)");
    println!("  【人脸对齐】 基于几何中心的仿射变换对齐 / 中心裁剪对齐");
    println!("  【特征提取】 Eigenfaces (PCA)  Fisherfaces (PCA+LDA)  LBPH (局部二值模式)");
    println!("  【匹配识别】 KNN(k近邻)  线性SVM  余弦/欧氏/卡方距离");
    println!();
    println!("子命令:");
    println!("  detect     对视频/图片做人脸检测，输出人脸帧和裁剪图");
    println!("  train      用数据集训练人脸识别模型");
    println!("  recognize  用已有模型识别图片中的人脸");
    println!("  info       显示系统信息");
    println!();
    println!("detect 参数:");
    println!("  --input <path>      本地视频文件路径");
    println!("  --url <url>         视频 URL（下载后用 ffmpeg 抽帧）");
    println!("  --image <path>      单张图片直接检测（跳过视频流程）");
    println!("  --output <dir>      输出目录 [默认: ./output]");
    println!("  --tmp-dir <dir>     临时帧目录 [默认: $TMP/rs-face]");
    println!("  --fps <n>           抽帧帧率 [默认: 1.0]");
    println!("  --min-size <px>     最小检测人脸 [默认: 40]");
    println!("  --max-size <px>     最大检测人脸 [默认: 400]");
    println!("  --scale <f>         图像金字塔缩放系数 [默认: 1.25]");
    println!("  --min-neighbors <n> 分组最小邻居 [默认: 3]");
    println!("  --step <px>         滑动窗口步长 [默认: 2]");
    println!("  --save-crops        同时保存人脸裁剪图");
    println!("  --keep-frames       保留中间抽帧");
    println!("  --padding <f>       裁剪外扩比例 [默认: 0.25]");
    println!("  --cascade <path>    指定 Haar Cascade XML [默认: data/haarcascade_frontalface_alt2.xml]");
    println!("  --flip-detect       在水平翻转图上再做一次检测 (捕获镜像侧脸)");
    println!("  --hog-svm <path>    加载 HOG+SVM 第二阶段检测器 (Dalal-Triggs 2005)");
    println!("  --hog-threshold <f> HOG+SVM 决策阈值 [默认: 0.0]");
    println!("  --dedup-iou <f>     相邻帧人脸 IoU > f 视为重复, 不写盘 [默认: 0.0 = 不去重]");
    println!();
    println!("train 参数:");
    println!("  --dataset <dir>     数据集目录（每个子目录=一个人,或按 filename_label.ext 命名）");
    println!("  --out <file>        模型输出路径 [默认: face_model.bin]");
    println!("  --algorithm <a>     eigenfaces / fisherfaces / lbph [默认: eigenfaces]");
    println!("  --components <k>    PCA/LDA 主成分数 [默认: 50]");
    println!("  --size WxH          训练/识别图像尺寸 [默认: 92x112]");
    println!();
    println!("recognize 参数:");
    println!("  --model <file>      模型文件");
    println!("  --input <path>      待识别图片");
    println!("  --output <dir>      可选：保存带识别结果的标注图");
    println!("  --threshold <f>     覆盖模型内置距离阈值");
    println!("  --size WxH          模型对应尺寸 [默认: 92x112]");
    println!("  --cascade <path>    先做人脸检测再识别");
    println!();
    println!("示例:");
    println!("  检测视频:  rs-face detect --url <URL> --output ./out --fps 1 --save-crops");
    println!("  检测图片:  rs-face detect --image face.jpg --output ./out");
    println!("  训练模型:  rs-face train --dataset ./dataset --out eigen.bin --algorithm eigenfaces --size 92x112");
    println!("  人脸识别:  rs-face recognize --model eigen.bin --input test.pgm");
}
