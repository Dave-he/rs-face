use crate::image::BoxError;
use crate::faces::{parse_list_str, parse_list_usize, parse_matrix_vec, chi_square_distance};

#[derive(Debug, Clone)]
pub struct LBPHModel {
    pub grid_x: usize,
    pub grid_y: usize,
    pub radius: i32,
    pub neighbors: i32,
    pub histograms: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub names: Vec<String>,
    pub threshold: f64,
}

impl LBPHModel {
    pub fn new() -> Self {
        Self {
            grid_x: 8,
            grid_y: 8,
            radius: 1,
            neighbors: 8,
            histograms: Vec::new(),
            labels: Vec::new(),
            names: Vec::new(),
            threshold: 50.0,
        }
    }

    pub fn with_params(grid_x: usize, grid_y: usize, radius: i32, neighbors: i32) -> Self {
        Self {
            grid_x,
            grid_y,
            radius,
            neighbors,
            histograms: Vec::new(),
            labels: Vec::new(),
            names: Vec::new(),
            threshold: 50.0,
        }
    }

    pub fn train(
        images: &[Vec<f64>],
        widths: &[usize],
        heights: &[usize],
        labels: &[usize],
        names: &[String],
        grid_x: usize,
        grid_y: usize,
    ) -> Result<Self, BoxError> {
        if images.is_empty() {
            return Err("LBPH: empty training set".into());
        }
        let mut model = Self::with_params(grid_x, grid_y, 1, 8);
        for i in 0..images.len() {
            let hist = model.compute_lbph(&images[i], widths[i], heights[i]);
            model.histograms.push(hist);
            model.labels.push(labels[i]);
        }
        model.names = names.to_vec();
        Ok(model)
    }

    pub fn compute_lbp(&self, gray: &[u8], w: usize, h: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        let r = self.radius;
        let n = self.neighbors;
        for y in r..(h as i32 - r) {
            for x in r..(w as i32 - r) {
                let mut code = 0u8;
                let center = gray[(y as usize) * w + (x as usize)];
                for k in 0..n {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                    let rx = (x as f64 - r as f64 * angle.cos()).round() as i32;
                    let ry = (y as f64 - r as f64 * angle.sin()).round() as i32;
                    let rx = rx.clamp(0, w as i32 - 1) as usize;
                    let ry = ry.clamp(0, h as i32 - 1) as usize;
                    let v = if gray[ry * w + rx] >= center { 1 } else { 0 };
                    code |= (v << k) as u8;
                }
                out[(y as usize) * w + (x as usize)] = code;
            }
        }
        out
    }

    pub fn compute_lbph(&self, normalized: &[f64], w: usize, h: usize) -> Vec<f64> {
        let gray: Vec<u8> = normalized.iter().map(|&v| {
            let x = v * 64.0 + 128.0;
            x.clamp(0.0, 255.0) as u8
        }).collect();
        let lbp = self.compute_lbp(&gray, w, h);
        let bins = 256;
        let cell_w = w / self.grid_x;
        let cell_h = h / self.grid_y;
        let mut hist = vec![0.0f64; self.grid_x * self.grid_y * bins];
        for gy in 0..self.grid_y {
            for gx in 0..self.grid_x {
                let base = (gy * self.grid_x + gx) * bins;
                let x0 = gx * cell_w;
                let y0 = gy * cell_h;
                let x1 = (x0 + cell_w).min(w);
                let y1 = (y0 + cell_h).min(h);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let b = lbp[y * w + x] as usize;
                        hist[base + b] += 1.0;
                    }
                }
                let mut sum = 0.0;
                for i in 0..bins { sum += hist[base + i]; }
                if sum > 0.0 {
                    for i in 0..bins { hist[base + i] /= sum; }
                }
            }
        }
        hist
    }

    pub fn predict(&self, face_vec: &[f64], w: usize, h: usize) -> (usize, f64, Option<String>) {
        let hist = self.compute_lbph(face_vec, w, h);
        let mut best_label = 0usize;
        let mut best_dist = f64::INFINITY;
        let mut second = f64::INFINITY;
        for (i, h) in self.histograms.iter().enumerate() {
            let d = chi_square_distance(&hist, h);
            if d < best_dist {
                second = best_dist;
                best_dist = d;
                best_label = self.labels[i];
            } else if d < second {
                second = d;
            }
        }
        let confidence = if second.is_finite() && second > 0.0 {
            (1.0 - best_dist / second).clamp(0.0, 1.0)
        } else {
            1.0 / (1.0 + best_dist / 10.0)
        };
        let name = if best_dist < self.threshold {
            self.names.get(best_label).cloned()
        } else {
            None
        };
        (best_label, confidence, name)
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), BoxError> {
        let mut s = String::new();
        s.push_str("LBPH_V1\n");
        s.push_str(&format!("params {} {} {} {}\n", self.grid_x, self.grid_y, self.radius, self.neighbors));
        s.push_str(&format!("histograms {} {}\n", self.histograms.len(), if self.histograms.is_empty() { 0 } else { self.histograms[0].len() }));
        for h in &self.histograms {
            for (i, v) in h.iter().enumerate() {
                if i > 0 { s.push(' '); }
                s.push_str(&format!("{:.10}", v));
            }
            s.push('\n');
        }
        s.push_str(&format!("labels {}\n", self.labels.len()));
        for l in &self.labels { s.push_str(&format!("{}\n", l)); }
        s.push_str(&format!("names {}\n", self.names.len()));
        for n in &self.names { s.push_str(n); s.push('\n'); }
        s.push_str(&format!("threshold {:.10}\n", self.threshold));
        std::fs::write(path, &s)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BoxError> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.lines();
        let header = lines.next().ok_or("bad model")?.trim();
        if header != "LBPH_V1" { return Err("Not an LBPH model".into()); }
        let params_line = lines.next().ok_or("no params")?.trim();
        let rest = params_line.strip_prefix("params ").ok_or("bad params")?;
        let parts: Vec<&str> = rest.split_whitespace().collect();
        let grid_x: usize = parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(8);
        let grid_y: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
        let radius: i32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
        let neighbors: i32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
        let histograms = parse_matrix_vec(&mut lines, "histograms")?;
        let labels = parse_list_usize(&mut lines, "labels")?;
        let names = parse_list_str(&mut lines, "names")?;
        let threshold_line = lines.next().ok_or("no threshold")?.trim();
        let threshold: f64 = threshold_line.strip_prefix("threshold ").ok_or("bad threshold")?.parse()?;
        Ok(Self {
            grid_x,
            grid_y,
            radius,
            neighbors,
            histograms,
            labels,
            names,
            threshold,
        })
    }
}

impl Default for LBPHModel {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub enum FaceModel {
    Eigenfaces(crate::faces::EigenfacesModel),
    Fisherfaces(crate::faces::FisherfacesModel),
    LBPH(LBPHModel),
}

impl FaceModel {
    pub fn predict_raw(&self, face_vec: &[f64], w: usize, h: usize) -> (usize, f64, Option<String>) {
        match self {
            FaceModel::Eigenfaces(m) => m.predict(face_vec),
            FaceModel::Fisherfaces(m) => m.predict(face_vec),
            FaceModel::LBPH(m) => m.predict(face_vec, w, h),
        }
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), BoxError> {
        match self {
            FaceModel::Eigenfaces(m) => m.save(path),
            FaceModel::Fisherfaces(m) => m.save(path),
            FaceModel::LBPH(m) => m.save(path),
        }
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BoxError> {
        let content = std::fs::read_to_string(&path)?;
        let header = content.lines().next().unwrap_or("").trim();
        match header {
            "EIGENFACES_V1" => Ok(FaceModel::Eigenfaces(crate::faces::EigenfacesModel::load(path)?)),
            "FISHERFACES_V1" => Ok(FaceModel::Fisherfaces(crate::faces::FisherfacesModel::load(path)?)),
            "LBPH_V1" => Ok(FaceModel::LBPH(LBPHModel::load(path)?)),
            _ => Err("Unknown face model format".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnnClassifier {
    pub train_vectors: Vec<Vec<f64>>,
    pub train_labels: Vec<usize>,
    pub names: Vec<String>,
    pub k: usize,
}

impl KnnClassifier {
    pub fn new(k: usize) -> Self {
        Self {
            train_vectors: Vec::new(),
            train_labels: Vec::new(),
            names: Vec::new(),
            k,
        }
    }

    pub fn train(&mut self, vectors: &[Vec<f64>], labels: &[usize], names: &[String]) {
        self.train_vectors = vectors.to_vec();
        self.train_labels = labels.to_vec();
        self.names = names.to_vec();
    }

    pub fn predict(&self, v: &[f64], metric: fn(&[f64], &[f64]) -> f64) -> (usize, f64, Option<String>) {
        let mut scores: Vec<(f64, usize)> = self.train_vectors.iter().enumerate().map(|(i, tv)| {
            (metric(v, tv), self.train_labels[i])
        }).collect();
        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.k.min(scores.len()).max(1);
        let mut class_votes = std::collections::BTreeMap::new();
        let mut sum_dist = 0.0f64;
        for (d, lbl) in scores.iter().take(k) {
            let entry = class_votes.entry(*lbl).or_insert((0usize, 0.0f64));
            entry.0 += 1;
            entry.1 += *d;
            sum_dist += *d;
        }
        let best = class_votes.iter().max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal)));
        match best {
            Some((&lbl, (votes, avg_dist))) => {
                let confidence = (*votes as f64 / k as f64).min(1.0) * (1.0 / (1.0 + avg_dist / 100.0));
                let name = self.names.get(lbl).cloned();
                (lbl, confidence, name)
            }
            None => (0, 0.0, None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SvmLinear {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub labels: Vec<usize>,
    pub names: Vec<String>,
}

impl SvmLinear {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            biases: Vec::new(),
            labels: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn train(&mut self, vectors: &[Vec<f64>], labels: &[usize], names: &[String], iterations: usize, lr: f64, lambda: f64) {
        let n = vectors.len();
        if n == 0 { return; }
        let d = vectors[0].len();
        let class_set: std::collections::BTreeSet<usize> = labels.iter().cloned().collect();
        let classes: Vec<usize> = class_set.into_iter().collect();
        let nc = classes.len();
        let mut all_w = Vec::with_capacity(nc);
        let mut all_b = Vec::with_capacity(nc);
        for (ci, &class_id) in classes.iter().enumerate() {
            let _ = ci;
            let mut w = vec![0.0f64; d];
            let mut b = 0.0f64;
            for _iter in 0..iterations {
                for i in 0..n {
                    let y = if labels[i] == class_id { 1.0 } else { -1.0 };
                    let mut score = b;
                    for j in 0..d { score += w[j] * vectors[i][j]; }
                    let margin = y * score;
                    if margin < 1.0 {
                        for j in 0..d {
                            w[j] -= lr * (lambda * w[j] - y * vectors[i][j]);
                        }
                        b += lr * y;
                    } else {
                        for j in 0..d {
                            w[j] -= lr * lambda * w[j];
                        }
                    }
                }
            }
            all_w.push(w);
            all_b.push(b);
            self.labels.push(class_id);
        }
        self.weights = all_w;
        self.biases = all_b;
        self.names = names.to_vec();
    }

    pub fn predict(&self, v: &[f64]) -> (usize, f64, Option<String>) {
        let mut best_label = *self.labels.first().unwrap_or(&0);
        let mut best_score = f64::NEG_INFINITY;
        let mut second = f64::NEG_INFINITY;
        for (i, lbl) in self.labels.iter().enumerate() {
            let mut score = self.biases[i];
            let w = &self.weights[i];
            for j in 0..v.len().min(w.len()) {
                score += w[j] * v[j];
            }
            if score > best_score {
                second = best_score;
                best_score = score;
                best_label = *lbl;
            } else if score > second {
                second = score;
            }
        }
        let total = best_score.exp() + (if second.is_finite() { second.exp() } else { 0.0 }) + 1e-9;
        let confidence = (best_score.exp() / total).clamp(0.0, 1.0);
        let name = self.names.get(best_label).cloned();
        (best_label, confidence, name)
    }
}

impl Default for SvmLinear { fn default() -> Self { Self::new() } }

