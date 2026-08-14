// 人脸识别 / 验证基准。
// 支持:
//   - K 折交叉验证 (identification): top-1/top-5 准确率 + 混淆矩阵
//   - 同/异人对验证 (verification): AUC / EER / 最佳阈值 / ROC 采样
// 数据集格式: <dataset>/<class>/<file>.pgm 或 .ppm 或 .png
// 零第三方依赖: 仅使用 std + 本项目内其他模块。

use crate::align::load_face_dataset;
use crate::faces::{cosine_distance, euclidean_distance};
use crate::image::BoxError;
use crate::linalg::Matrix;
use crate::recognition::LBPHModel;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Identification,
    Verification,
    Both,
}

impl Mode {
    pub fn from_str(s: &str) -> Result<Self, BoxError> {
        match s.to_lowercase().as_str() {
            "identification" | "id" | "identify" => Ok(Mode::Identification),
            "verification" | "ver" | "verify" => Ok(Mode::Verification),
            "both" | "all" => Ok(Mode::Both),
            _ => Err(format!("未知 mode: {} (identification|verification|both)", s).into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            _ => Err(format!("未知算法: {} (eigenfaces|fisherfaces|lbph)", s).into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub mode: Mode,
    pub algorithm: Algorithm,
    pub folds: usize,         // K 折交叉验证 (默认 5)
    pub max_pairs: usize,     // 验证任务最大配对数 (默认 2000)
    pub seed: u64,            // RNG 种子 (默认 42)
    pub size: (usize, usize), // 训练/识别图像尺寸 (默认 92x112)
    pub train_per_fold: usize,// 每类训练样本数 (0 = 自动 = max(1, samples_per_class - test_per_fold))
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mode: Mode::Both,
            algorithm: Algorithm::Eigenfaces,
            folds: 5,
            max_pairs: 2000,
            seed: 42,
            size: (92, 112),
            train_per_fold: 0,
        }
    }
}

// ---------- 加载 ----------

#[derive(Debug, Clone)]
pub struct LoadedDataset {
    pub paths: Vec<std::path::PathBuf>,
    pub vectors: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub names: Vec<String>,
    pub samples_per_class: Vec<usize>,
}

pub fn load_dataset(dir: &Path, size: (usize, usize)) -> Result<LoadedDataset, BoxError> {
    // 复用 align::load_face_dataset, 同时保留原始路径用于按文件配对。
    let (mat, labels, names) = load_face_dataset(dir, size)?;
    // 重新扫描路径, 按 labels 顺序对齐
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.path());
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let mut vectors: Vec<Vec<f64>> = Vec::new();
    let mut out_labels: Vec<usize> = Vec::new();
    let mut class_count: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let idx = names.iter().position(|n| n == &name).unwrap_or(0);
            let mut files: Vec<_> = std::fs::read_dir(&path)?.filter_map(|e| e.ok()).collect();
            files.sort_by_key(|e| e.path());
            for f in files {
                let p = f.path();
                let ext = p.extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if ext != "pgm" && ext != "ppm" && ext != "png" { continue; }
                if let Ok(img) = crate::image::Image::load_pgm(&p)
                    .or_else(|_| crate::image::Image::load_ppm(&p))
                {
                    let vec = crate::align::preprocess_for_recognition(&img, None, size);
                    paths.push(p);
                    vectors.push(vec);
                    out_labels.push(idx);
                    *class_count.entry(idx).or_insert(0) += 1;
                }
            }
        }
    }
    let samples_per_class = (0..names.len()).map(|c| *class_count.get(&c).unwrap_or(&0)).collect();
    let _ = mat; // 已用 vectors, 释放 mat
    if vectors.is_empty() {
        return Err("数据集为空或无可识别图像".into());
    }
    Ok(LoadedDataset {
        paths,
        vectors,
        labels: out_labels,
        names,
        samples_per_class,
    })
}

// ---------- 公共 RNG (xorshift64, 零依赖) ----------

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n.max(1)
    }
    pub fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = self.gen_range(i + 1);
            v.swap(i, j);
        }
    }
}

// ---------- K 折分层划分 ----------

pub fn stratified_kfold(labels: &[usize], k: usize, seed: u64) -> Vec<Vec<usize>> {
    // 每类的索引独立洗牌后, 切分成 k 份
    let mut by_class: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_class.entry(l).or_default().push(i);
    }
    let mut rng = Rng::new(seed);
    for v in by_class.values_mut() { rng.shuffle(v); }
    let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for class_indices in by_class.values() {
        for (i, &idx) in class_indices.iter().enumerate() {
            folds[i % k].push(idx);
        }
    }
    folds
}

// ---------- 通用识别预测 (用于识别 + 验证都需的模型训练) ----------

fn train_model(
    algorithm: Algorithm,
    train_vecs: &[Vec<f64>],
    train_labels: &[usize],
    train_names: &[String],
    size: (usize, usize),
) -> Result<TrainedModel, BoxError> {
    match algorithm {
        Algorithm::Eigenfaces => {
            let mat = matrix_from(train_vecs);
            let model = crate::faces::EigenfacesModel::train(&mat, train_labels, train_names, 50.min(train_vecs.len()))?;
            Ok(TrainedModel::Eigen(model))
        }
        Algorithm::Fisherfaces => {
            let mat = matrix_from(train_vecs);
            let classes: std::collections::BTreeSet<usize> = train_labels.iter().cloned().collect();
            let nc = (classes.len().saturating_sub(1)).max(1);
            let model = crate::faces::FisherfacesModel::train(&mat, train_labels, train_names, nc)?;
            Ok(TrainedModel::Fisher(model))
        }
        Algorithm::LBPH => {
            let n = train_vecs.len();
            let mut widths = Vec::with_capacity(n);
            let mut heights = Vec::with_capacity(n);
            for _ in 0..n {
                widths.push(size.0);
                heights.push(size.1);
            }
            let model = LBPHModel::train(train_vecs, &widths, &heights, train_labels, train_names, 8, 8)?;
            Ok(TrainedModel::Lbph(model))
        }
    }
}

fn matrix_from(vecs: &[Vec<f64>]) -> Matrix {
    if vecs.is_empty() { return Matrix::new(0, 0); }
    let rows = vecs.len();
    let cols = vecs[0].len();
    let mut m = Matrix::new(rows, cols);
    for (r, v) in vecs.iter().enumerate() {
        for (c, &x) in v.iter().enumerate() {
            m.set(r, c, x);
        }
    }
    m
}

enum TrainedModel {
    Eigen(crate::faces::EigenfacesModel),
    Fisher(crate::faces::FisherfacesModel),
    Lbph(LBPHModel),
}

impl TrainedModel {
    fn predict(&self, vec: &[f64], size: (usize, usize)) -> (usize, f64, f64) {
        // 返回 (label, distance_to_best, distance_to_second_best)
        match self {
            TrainedModel::Eigen(m) => {
                let proj = m.project_new(vec);
                let mut best = f64::INFINITY;
                let mut second = f64::INFINITY;
                let mut best_lbl = 0usize;
                for (i, p) in m.projections.iter().enumerate() {
                    let d = cosine_distance(&proj, p);
                    if d < best {
                        second = best;
                        best = d;
                        best_lbl = m.labels[i];
                    } else if d < second {
                        second = d;
                    }
                }
                (best_lbl, best, second)
            }
            TrainedModel::Fisher(m) => {
                let proj = crate::linalg::project(vec, &m.eigenvectors, &m.mean);
                let mut best = f64::INFINITY;
                let mut second = f64::INFINITY;
                let mut best_lbl = 0usize;
                for (i, p) in m.projections.iter().enumerate() {
                    let d = euclidean_distance(&proj, p);
                    if d < best {
                        second = best;
                        best = d;
                        best_lbl = m.labels[i];
                    } else if d < second {
                        second = d;
                    }
                }
                (best_lbl, best, second)
            }
            TrainedModel::Lbph(m) => {
                let hist = m.compute_lbph(vec, size.0, size.1);
                let mut best = f64::INFINITY;
                let mut second = f64::INFINITY;
                let mut best_lbl = 0usize;
                for (i, h) in m.histograms.iter().enumerate() {
                    let d = crate::faces::chi_square_distance(&hist, h);
                    if d < best {
                        second = best;
                        best = d;
                        best_lbl = m.labels[i];
                    } else if d < second {
                        second = d;
                    }
                }
                (best_lbl, best, second)
            }
        }
    }

    /// 提取人脸表征向量 (用于验证任务: 计算两张人脸的距离)
    fn embed(&self, vec: &[f64], size: (usize, usize)) -> Vec<f64> {
        match self {
            TrainedModel::Eigen(m) => m.project_new(vec),
            TrainedModel::Fisher(m) => crate::linalg::project(vec, &m.eigenvectors, &m.mean),
            TrainedModel::Lbph(m) => m.compute_lbph(vec, size.0, size.1),
        }
    }
}

fn distance(a: &[f64], b: &[f64], algorithm: Algorithm) -> f64 {
    match algorithm {
        Algorithm::Eigenfaces => cosine_distance(a, b),
        Algorithm::Fisherfaces => euclidean_distance(a, b),
        Algorithm::LBPH => crate::faces::chi_square_distance(a, b),
    }
}

// ---------- Identification 报告 ----------

#[derive(Debug, Clone)]
pub struct IdentificationReport {
    pub dataset: String,
    pub algorithm: String,
    pub n_samples: usize,
    pub n_classes: usize,
    pub folds: usize,
    pub train_secs: f64,
    pub test_secs: f64,
    pub top1: f64,
    pub top5: f64,
    pub per_class_top1: Vec<f64>,
    pub confusion: Vec<Vec<u32>>, // [n_classes][n_classes]
    pub failures: Vec<Failure>,
    pub train_ms_per_fold: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct Failure {
    pub true_name: String,
    pub pred_name: String,
    pub path: std::path::PathBuf,
    pub distance: f64,
    pub second: f64,
}

pub fn run_identification(dataset: &LoadedDataset, opts: &Options) -> Result<IdentificationReport, BoxError> {
    if dataset.vectors.is_empty() { return Err("空数据集".into()); }
    let k = opts.folds.max(2).min(dataset.vectors.len());
    let folds = stratified_kfold(&dataset.labels, k, opts.seed);
    let n = dataset.vectors.len();
    let c = dataset.names.len();
    let mut confusion = vec![vec![0u32; c]; c];
    let mut failures: Vec<Failure> = Vec::new();
    let mut per_class_correct: Vec<u32> = vec![0; c];
    let mut per_class_total: Vec<u32> = vec![0; c];
    let mut top5_hits = 0usize;
    let mut total = 0usize;
    let mut train_ms: Vec<u64> = Vec::with_capacity(k);
    let mut fold_test_secs = 0.0f64;
    for (fold_idx, test_idx) in folds.iter().enumerate() {
        let test_set: std::collections::BTreeSet<usize> = test_idx.iter().copied().collect();
        let mut train_vecs: Vec<Vec<f64>> = Vec::new();
        let mut train_labels: Vec<usize> = Vec::new();
        for i in 0..n {
            if test_set.contains(&i) { continue; }
            train_vecs.push(dataset.vectors[i].clone());
            train_labels.push(dataset.labels[i]);
        }
        let train_t = std::time::Instant::now();
        let model = train_model(opts.algorithm, &train_vecs, &train_labels, &dataset.names, opts.size)?;
        train_ms.push(train_t.elapsed().as_millis() as u64);

        let fold_test_t = std::time::Instant::now();
        for &i in test_idx {
            let (lbl, best, second) = model.predict(&dataset.vectors[i], opts.size);
            let true_lbl = dataset.labels[i];
            confusion[true_lbl][lbl] += 1;
            per_class_total[true_lbl] += 1;
            if lbl == true_lbl {
                per_class_correct[true_lbl] += 1;
            } else {
                failures.push(Failure {
                    true_name: dataset.names[true_lbl].clone(),
                    pred_name: dataset.names.get(lbl).cloned().unwrap_or_default(),
                    path: dataset.paths[i].clone(),
                    distance: best,
                    second,
                });
            }
            // top-5: 真实标签出现在"前 5 最小距离"候选中
            if lbl_eq_or_within_top_k(&model, &dataset.vectors[i], true_lbl, opts.size, 5) {
                top5_hits += 1;
            }
            total += 1;
        }
        fold_test_secs += fold_test_t.elapsed().as_secs_f64();
        println!("[bench] identification fold {}/{}: train={}ms, test={}ms",
            fold_idx + 1, k, train_ms[fold_idx], (fold_test_t.elapsed().as_secs_f64() * 1000.0) as u64);
    }
    let top1_correct: u32 = per_class_correct.iter().sum();
    let top1 = top1_correct as f64 / total as f64;
    let top5 = top5_hits as f64 / total as f64;
    let per_class_top1: Vec<f64> = (0..c).map(|i| {
        if per_class_total[i] == 0 { 0.0 } else { per_class_correct[i] as f64 / per_class_total[i] as f64 }
    }).collect();
    Ok(IdentificationReport {
        dataset: String::new(),
        algorithm: format!("{:?}", opts.algorithm).to_lowercase(),
        n_samples: n,
        n_classes: c,
        folds: k,
        train_secs: train_ms.iter().sum::<u64>() as f64 / 1000.0,
        test_secs: fold_test_secs,
        top1,
        top5,
        per_class_top1,
        confusion,
        failures,
        train_ms_per_fold: train_ms,
    })
}

/// 判断真实标签是否出现在"前 k 最小距离"的候选中。
fn lbl_eq_or_within_top_k(
    model: &TrainedModel,
    vec: &[f64],
    true_lbl: usize,
    size: (usize, usize),
    k: usize,
) -> bool {
    match model {
        TrainedModel::Eigen(m) => {
            let proj = m.project_new(vec);
            let mut dists: Vec<(f64, usize)> = m.projections.iter().enumerate()
                .map(|(i, p)| (cosine_distance(&proj, p), m.labels[i])).collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.iter().take(k).any(|(_, l)| *l == true_lbl)
        }
        TrainedModel::Fisher(m) => {
            let proj = crate::linalg::project(vec, &m.eigenvectors, &m.mean);
            let mut dists: Vec<(f64, usize)> = m.projections.iter().enumerate()
                .map(|(i, p)| (euclidean_distance(&proj, p), m.labels[i])).collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.iter().take(k).any(|(_, l)| *l == true_lbl)
        }
        TrainedModel::Lbph(m) => {
            let hist = m.compute_lbph(vec, size.0, size.1);
            let mut dists: Vec<(f64, usize)> = m.histograms.iter().enumerate()
                .map(|(i, h)| (crate::faces::chi_square_distance(&hist, h), m.labels[i])).collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.iter().take(k).any(|(_, l)| *l == true_lbl)
        }
    }
}

// ---------- Verification 报告 ----------

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub dataset: String,
    pub algorithm: String,
    pub n_pairs: usize,
    pub n_pos: usize,
    pub n_neg: usize,
    pub auc: f64,
    pub eer: f64,
    pub best_threshold: f64,
    pub best_accuracy: f64,
    pub train_secs: f64,
    pub test_secs: f64,
    pub mean_pos: f64,
    pub mean_neg: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub i: usize,
    pub j: usize,
    pub same: bool,
}

/// 生成同/异人对。
/// 同人: 每类内部组合; 异人: 跨类组合 (随机抽样, 平衡 1:1)
pub fn generate_pairs(labels: &[usize], max_pairs: usize, seed: u64) -> Vec<Pair> {
    let mut by_class: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by_class.entry(l).or_default().push(i);
    }
    let mut pos: Vec<Pair> = Vec::new();
    for v in by_class.values() {
        for w in v.windows(2) {
            pos.push(Pair { i: w[0], j: w[1], same: true });
        }
    }
    if pos.len() > max_pairs / 2 {
        // 太多就下采样
        let mut rng = Rng::new(seed);
        let mut keep: Vec<bool> = vec![false; pos.len()];
        for _ in 0..(max_pairs / 2) {
            keep[rng.gen_range(pos.len())] = true;
        }
        let mut new_pos = Vec::new();
        for (i, p) in pos.into_iter().enumerate() {
            if keep[i] { new_pos.push(p); }
        }
        pos = new_pos;
    }
    let target_neg = pos.len();
    let mut rng = Rng::new(seed.wrapping_mul(0x9E3779B97F4A7C15));
    let all_idx: Vec<usize> = (0..labels.len()).collect();
    let mut neg: Vec<Pair> = Vec::new();
    let mut attempts = 0usize;
    while neg.len() < target_neg && attempts < target_neg * 20 {
        attempts += 1;
        let a = all_idx[rng.gen_range(all_idx.len())];
        let b = all_idx[rng.gen_range(all_idx.len())];
        if a == b { continue; }
        if labels[a] == labels[b] { continue; }
        // 去重 (无序对)
        let key = if a < b { (a, b) } else { (b, a) };
        let already = neg.iter().any(|p| {
            let k = if p.i < p.j { (p.i, p.j) } else { (p.j, p.i) };
            k == key
        });
        if already { continue; }
        neg.push(Pair { i: a, j: b, same: false });
    }
    let mut all_pairs = pos;
    all_pairs.extend(neg);
    let mut rng = Rng::new(seed);
    rng.shuffle(&mut all_pairs.iter_mut().enumerate().map(|(i, _)| i).collect::<Vec<_>>());
    // 上面 shuffle 不对 (借用问题), 改为 Fisher-Yates on indices:
    let mut indices: Vec<usize> = (0..all_pairs.len()).collect();
    rng.shuffle(&mut indices);
    indices.into_iter().map(|i| all_pairs[i].clone()).collect()
}

pub fn run_verification(dataset: &LoadedDataset, opts: &Options) -> Result<VerificationReport, BoxError> {
    if dataset.vectors.is_empty() { return Err("空数据集".into()); }
    let train_start = std::time::Instant::now();
    let model = train_model(opts.algorithm, &dataset.vectors, &dataset.labels, &dataset.names, opts.size)?;
    let train_secs = train_start.elapsed().as_secs_f64();
    let pairs = generate_pairs(&dataset.labels, opts.max_pairs, opts.seed);
    let n_pos = pairs.iter().filter(|p| p.same).count();
    let n_neg = pairs.len() - n_pos;
    let test_start = std::time::Instant::now();
    let mut scores: Vec<(f64, bool)> = Vec::with_capacity(pairs.len());
    let mut sum_pos = 0.0f64;
    let mut sum_neg = 0.0f64;
    for p in &pairs {
        let ea = model.embed(&dataset.vectors[p.i], opts.size);
        let eb = model.embed(&dataset.vectors[p.j], opts.size);
        let d = distance(&ea, &eb, opts.algorithm);
        if p.same { sum_pos += d; } else { sum_neg += d; }
        scores.push((d, p.same));
    }
    let test_secs = test_start.elapsed().as_secs_f64();
    let mean_pos = if n_pos > 0 { sum_pos / n_pos as f64 } else { 0.0 };
    let mean_neg = if n_neg > 0 { sum_neg / n_neg as f64 } else { 0.0 };
    let (auc, eer, best_thr, best_acc) = compute_roc(&scores);
    Ok(VerificationReport {
        dataset: String::new(),
        algorithm: format!("{:?}", opts.algorithm).to_lowercase(),
        n_pairs: pairs.len(),
        n_pos,
        n_neg,
        auc,
        eer,
        best_threshold: best_thr,
        best_accuracy: best_acc,
        train_secs,
        test_secs,
        mean_pos,
        mean_neg,
    })
}

// ROC / AUC / EER (纯 std 实现)
// 输入: (score, is_same), 距离越小越相似 → 把距离当作 "score"
// 取阈值 t, predicted_same = score <= t; 计算 FAR(t) = FPR / FRR(t) = FNR
fn compute_roc(scores: &[(f64, bool)]) -> (f64, f64, f64, f64) {
    if scores.is_empty() { return (0.0, 1.0, 0.0, 0.0); }
    let n_pos = scores.iter().filter(|(_, s)| *s).count() as f64;
    let n_neg = scores.iter().filter(|(_, s)| !*s).count() as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return (0.5, 1.0, 0.0, 0.0);
    }
    // 距离从小到大排序 (越小越相似 → 在 ROC 阈值下方是 accepted)
    let mut sorted: Vec<(f64, bool)> = scores.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // 扫一遍: 在每个唯一阈值处计算 TPR/FPR
    let mut auc = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_tpr = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut last_d = f64::NEG_INFINITY;
    let mut eer = 1.0;
    let mut best_acc = 0.0;
    let mut best_thr = 0.0;
    let n = sorted.len();
    let mut i = 0usize;
    while i < n {
        let d = sorted[i].0;
        while i < n && sorted[i].0 == d {
            if sorted[i].1 { tp += 1.0; } else { fp += 1.0; }
            i += 1;
        }
        let tpr = tp / n_pos;
        let fpr = fp / n_neg;
        // 梯形 AUC
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_fpr = fpr;
        prev_tpr = tpr;
        // EER: FPR + FNR = 1 处; FNR = 1 - TPR
        let frr = 1.0 - tpr;
        let e = (fpr + frr).abs();
        if e < eer {
            eer = e;
        }
        // 最佳 acc: 1 - min(FPR, FNR)
        let acc = 1.0 - (fpr.max(frr));
        if acc > best_acc {
            best_acc = acc;
            best_thr = d;
        }
        last_d = d;
    }
    (auc.clamp(0.0, 1.0), eer, best_thr, best_acc)
}

// ---------- Markdown 报告 ----------

pub fn write_markdown(
    out_path: &Path,
    id: Option<&IdentificationReport>,
    ver: Option<&VerificationReport>,
    dataset_name: &str,
    dataset_dir: &Path,
) -> Result<(), BoxError> {
    let mut f = std::fs::File::create(out_path)?;
    writeln!(f, "# 人脸识别 / 验证基准 (rs-face)")?;
    writeln!(f)?;
    writeln!(f, "- 数据集: `{}` ({})", dataset_name, dataset_dir.display())?;
    writeln!(f, "- 时间: {}", chrono_like_now())?;
    writeln!(f)?;
    if let Some(id) = id {
        write_identification_md(&mut f, id)?;
    }
    if let Some(ver) = ver {
        writeln!(f)?;
        write_verification_md(&mut f, ver)?;
    }
    Ok(())
}

fn chrono_like_now() -> String {
    // 零依赖: 使用 SystemTime 算 unix epoch, 然后手动格式化
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0);
    let secs_per_day = 86400u64;
    let days = now / secs_per_day;
    let mut year = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let dy = if leap { 366 } else { 365 };
        if remaining_days < dy { break; }
        remaining_days -= dy;
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_days = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &dm in &month_days {
        if remaining_days < dm { break; }
        remaining_days -= dm;
        month += 1;
    }
    let day = remaining_days + 1;
    let secs_today = now % secs_per_day;
    let hour = secs_today / 3600;
    let minute = (secs_today % 3600) / 60;
    let second = secs_today % 60;
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", year, month, day, hour, minute, second)
}

fn write_identification_md<W: Write>(f: &mut W, id: &IdentificationReport) -> std::io::Result<()> {
    writeln!(f, "## 🎯 识别 (K 折交叉验证)")?;
    writeln!(f)?;
    writeln!(f, "| 指标 | 数值 |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| 算法 | `{}` |", id.algorithm)?;
    writeln!(f, "| 样本数 | {} |", id.n_samples)?;
    writeln!(f, "| 类别数 | {} |", id.n_classes)?;
    writeln!(f, "| 折数 (K) | {} |", id.folds)?;
    writeln!(f, "| 总训练耗时 | {:.2}s (平均 {:.0}ms/折) |", id.train_secs,
        id.train_ms_per_fold.iter().sum::<u64>() as f64 / id.folds.max(1) as f64)?;
    writeln!(f, "| 总测试耗时 | {:.2}s |", id.test_secs)?;
    writeln!(f, "| **Top-1 准确率** | **{:.2}%** |", id.top1 * 100.0)?;
    writeln!(f, "| **Top-5 准确率** | **{:.2}%** |", id.top5 * 100.0)?;
    writeln!(f)?;
    // 类别级准确率 (有意义的统计)
    if id.n_classes <= 30 {
        writeln!(f, "### 类别级 Top-1 准确率")?;
        writeln!(f)?;
        let mut per_class = id.per_class_top1.iter().enumerate()
            .map(|(i, &acc)| (i, acc)).collect::<Vec<_>>();
        per_class.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_worst = id.n_classes.min(10);
        writeln!(f, "最差 {} 类:", n_worst)?;
        for (i, acc) in per_class.iter().take(n_worst) {
            writeln!(f, "- 类别 {}: {:.2}%", i, acc * 100.0)?;
        }
        writeln!(f)?;
    }
    if !id.failures.is_empty() {
        let nf = id.failures.len().min(10);
        writeln!(f, "### 误分类样本 (Top {})", nf)?;
        writeln!(f)?;
        writeln!(f, "| 真实 | 预测 | 距离 | 次佳距离 |")?;
        writeln!(f, "|---|---|---|---|")?;
        for fail in id.failures.iter().take(nf) {
            writeln!(f, "| {} | {} | {:.4} | {:.4} |",
                fail.true_name, fail.pred_name, fail.distance, fail.second)?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn write_verification_md<W: Write>(f: &mut W, ver: &VerificationReport) -> std::io::Result<()> {
    writeln!(f, "## 🔍 验证 (LFW 风格同/异人对)")?;
    writeln!(f)?;
    writeln!(f, "| 指标 | 数值 |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| 算法 | `{}` |", ver.algorithm)?;
    writeln!(f, "| 总配对数 | {} (同 {} / 异 {}) |", ver.n_pairs, ver.n_pos, ver.n_neg)?;
    writeln!(f, "| 训练耗时 | {:.2}s |", ver.train_secs)?;
    writeln!(f, "| 测试耗时 | {:.2}s |", ver.test_secs)?;
    writeln!(f, "| **AUC** | **{:.4}** |", ver.auc)?;
    writeln!(f, "| **EER** | **{:.4}** (越小越好) |", ver.eer)?;
    writeln!(f, "| 最佳阈值 | {:.4} |", ver.best_threshold)?;
    writeln!(f, "| 最佳准确率 | {:.2}% |", ver.best_accuracy * 100.0)?;
    writeln!(f, "| 同人对平均距离 | {:.4} |", ver.mean_pos)?;
    writeln!(f, "| 异人对平均距离 | {:.4} |", ver.mean_neg)?;
    writeln!(f)?;
    writeln!(f, "> **AUC (Area Under ROC Curve)**: 1.0 = 完美区分, 0.5 = 随机。")?;
    writeln!(f, "> **EER (Equal Error Rate)**: FAR = FRR 时的错误率, 越小越好。")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_kfold_balanced() {
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let folds = stratified_kfold(&labels, 4, 42);
        assert_eq!(folds.len(), 4);
        for f in &folds {
            // 每个 fold 应该有 3 个样本 (每类 1 个)
            assert_eq!(f.len(), 3);
        }
    }

    #[test]
    fn rng_shuffle_full_coverage() {
        let mut rng = Rng::new(1);
        let mut v: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut v);
        v.sort();
        assert_eq!(v, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn roc_auc_perfect() {
        // 同人对距离都很小, 异人对距离都大
        let scores = vec![
            (0.1, true), (0.2, true), (0.3, true),
            (0.9, false), (1.0, false), (1.1, false),
        ];
        let (auc, _eer, _thr, acc) = compute_roc(&scores);
        assert!(auc > 0.99, "perfect AUC, got {}", auc);
        assert!(acc > 0.99, "perfect acc, got {}", acc);
    }

    #[test]
    fn roc_auc_random() {
        let scores = vec![
            (0.5, true), (0.5, false), (0.5, true), (0.5, false),
        ];
        let (auc, _, _, _) = compute_roc(&scores);
        // 全相同距离 → 不能区分 → AUC 0.5
        assert!((auc - 0.5).abs() < 0.01, "expected ~0.5, got {}", auc);
    }
}