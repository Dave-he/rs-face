use crate::linalg::{Matrix, project, pca, lda};
use crate::image::BoxError;

#[derive(Debug, Clone)]
pub struct EigenfacesModel {
    pub eigenvectors: Matrix,
    pub eigenvalues: Vec<f64>,
    pub mean: Vec<f64>,
    pub projections: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub names: Vec<String>,
    pub threshold: f64,
}

impl EigenfacesModel {
    pub fn train(
        data: &Matrix,
        labels: &[usize],
        names: &[String],
        num_components: usize,
    ) -> Result<Self, BoxError> {
        let (eigenvalues, eigenvectors, mean) = pca(data, num_components)?;
        let mut projections = Vec::with_capacity(data.rows);
        for r in 0..data.rows {
            let row: Vec<f64> = (0..data.cols).map(|c| data.get(r, c)).collect();
            projections.push(project(&row, &eigenvectors, &mean));
        }
        let mut threshold = 0.0;
        for i in 0..projections.len() {
            for j in (i + 1)..projections.len() {
                let d = cosine_distance(&projections[i], &projections[j]);
                if d > threshold {
                    threshold = d;
                }
            }
        }
        threshold = (threshold * 0.8).max(0.5);
        Ok(Self {
            eigenvectors,
            eigenvalues,
            mean,
            projections,
            labels: labels.to_vec(),
            names: names.to_vec(),
            threshold,
        })
    }

    pub fn project_new(&self, face_vec: &[f64]) -> Vec<f64> {
        project(face_vec, &self.eigenvectors, &self.mean)
    }

    pub fn predict(&self, face_vec: &[f64]) -> (usize, f64, Option<String>) {
        let proj = self.project_new(face_vec);
        let mut best_label = 0usize;
        let mut best_dist = f64::INFINITY;
        let mut second = f64::INFINITY;
        for (i, p) in self.projections.iter().enumerate() {
            let d = cosine_distance(&proj, p);
            if d < best_dist {
                second = best_dist;
                best_dist = d;
                best_label = self.labels[i];
            } else if d < second {
                second = d;
            }
        }
        let confidence = if second.is_finite() && second > 0.0 {
            1.0 - best_dist / second
        } else {
            1.0 - best_dist
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
        s.push_str("EIGENFACES_V1\n");
        s.push_str(&format!("components {} {}\n", self.eigenvectors.rows, self.eigenvectors.cols));
        s.push_str(&format!("eigenvalues {}\n", self.eigenvalues.len()));
        for &v in &self.eigenvalues {
            s.push_str(&format!("{:.10}\n", v));
        }
        s.push_str(&format!("mean {}\n", self.mean.len()));
        for &v in &self.mean {
            s.push_str(&format!("{:.10}\n", v));
        }
        s.push_str(&format!("eigenvectors {} {}\n", self.eigenvectors.rows, self.eigenvectors.cols));
        for r in 0..self.eigenvectors.rows {
            for c in 0..self.eigenvectors.cols {
                if c > 0 { s.push(' '); }
                s.push_str(&format!("{:.10}", self.eigenvectors.get(r, c)));
            }
            s.push('\n');
        }
        s.push_str(&format!("projections {} {}\n", self.projections.len(), if self.projections.is_empty() { 0 } else { self.projections[0].len() }));
        for p in &self.projections {
            for (i, v) in p.iter().enumerate() {
                if i > 0 { s.push(' '); }
                s.push_str(&format!("{:.10}", v));
            }
            s.push('\n');
        }
        s.push_str(&format!("labels {}\n", self.labels.len()));
        for l in &self.labels {
            s.push_str(&format!("{}\n", l));
        }
        s.push_str(&format!("names {}\n", self.names.len()));
        for n in &self.names {
            s.push_str(n);
            s.push('\n');
        }
        s.push_str(&format!("threshold {:.10}\n", self.threshold));
        std::fs::write(path, &s)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BoxError> {
        let content = std::fs::read_to_string(path)?;
        let mut lines = content.lines();
        let header = lines.next().ok_or("bad model")?.trim();
        if header != "EIGENFACES_V1" { return Err("Not an Eigenfaces model".into()); }
        let eigenvectors = parse_matrix(&mut lines, "components")?;
        let eigenvalues = parse_list_f64(&mut lines, "eigenvalues")?;
        let mean = parse_list_f64(&mut lines, "mean")?;
        let _ev_hdr = lines.next();
        let ev_data = eigenvectors.clone();
        let projections = parse_matrix_vec(&mut lines, "projections")?;
        let labels = parse_list_usize(&mut lines, "labels")?;
        let names = parse_list_str(&mut lines, "names")?;
        let threshold_line = lines.next().ok_or("no threshold")?.trim();
        let threshold: f64 = threshold_line.strip_prefix("threshold ").ok_or("bad threshold")?.parse()?;
        Ok(Self {
            eigenvectors: ev_data,
            eigenvalues,
            mean,
            projections,
            labels,
            names,
            threshold,
        })
    }
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let na = na.sqrt();
    let nb = nb.sqrt();
    if na < 1e-12 || nb < 1e-12 { return 1.0; }
    let sim = dot / (na * nb);
    (1.0 - sim.clamp(-1.0, 1.0)) / 2.0
}

pub fn chi_square_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len().min(b.len()) {
        let denom = a[i] + b[i];
        if denom.abs() > 1e-12 {
            let d = a[i] - b[i];
            sum += d * d / denom;
        }
    }
    sum * 0.5
}

pub fn parse_header<'a, I: Iterator<Item = &'a str>>(lines: &mut I, key: &str) -> (usize, usize) {
    let line = lines.next().unwrap_or("").trim();
    let rest = line.strip_prefix(key).unwrap_or("").trim();
    let parts: Vec<&str> = rest.split_whitespace().collect();
    let x: usize = parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(0);
    let y: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    (x, y)
}

pub fn parse_matrix<'a, I: Iterator<Item = &'a str>>(lines: &mut I, key: &str) -> Result<Matrix, BoxError> {
    let (rows, cols) = parse_header(lines, key);
    let mut m = Matrix::new(rows, cols);
    for r in 0..rows {
        let line = lines.next().ok_or("truncated matrix")?.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        for c in 0..cols.min(parts.len()) {
            m.set(r, c, parts[c].parse().unwrap_or(0.0));
        }
    }
    Ok(m)
}

pub fn parse_matrix_vec<'a, I: Iterator<Item = &'a str>>(lines: &mut I, key: &str) -> Result<Vec<Vec<f64>>, BoxError> {
    let (rows, cols) = parse_header(lines, key);
    let mut out = Vec::with_capacity(rows);
    for _ in 0..rows {
        let line = lines.next().ok_or("truncated matrixvec")?.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols.min(parts.len()) {
            row.push(parts[c].parse().unwrap_or(0.0));
        }
        while row.len() < cols { row.push(0.0); }
        out.push(row);
    }
    Ok(out)
}

pub fn parse_list_f64<'a, I: Iterator<Item = &'a str>>(lines: &mut I, key: &str) -> Result<Vec<f64>, BoxError> {
    let line = lines.next().ok_or("bad list header")?.trim();
    let n: usize = line.strip_prefix(key).unwrap_or("").trim().parse().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let l = lines.next().ok_or("truncated list")?.trim();
        out.push(l.parse().unwrap_or(0.0));
    }
    Ok(out)
}

pub fn parse_list_usize<'a, I: Iterator<Item = &'a str>>(lines: &mut I, key: &str) -> Result<Vec<usize>, BoxError> {
    let line = lines.next().ok_or("bad header")?.trim();
    let n: usize = line.strip_prefix(key).unwrap_or("").trim().parse().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let l = lines.next().ok_or("truncated")?.trim();
        out.push(l.parse().unwrap_or(0));
    }
    Ok(out)
}

pub fn parse_list_str<'a, I: Iterator<Item = &'a str>>(lines: &mut I, key: &str) -> Result<Vec<String>, BoxError> {
    let line = lines.next().ok_or("bad str header")?.trim();
    let n: usize = line.strip_prefix(key).unwrap_or("").trim().parse().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(lines.next().unwrap_or("").to_string());
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct FisherfacesModel {
    pub eigenvectors: Matrix,
    pub eigenvalues: Vec<f64>,
    pub mean: Vec<f64>,
    pub projections: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub names: Vec<String>,
    pub threshold: f64,
}

impl FisherfacesModel {
    pub fn train(
        data: &Matrix,
        labels: &[usize],
        names: &[String],
        num_components: usize,
    ) -> Result<Self, BoxError> {
        let classes = labels.iter().cloned().collect::<std::collections::BTreeSet<_>>();
        let c = classes.len();
        let max_comp = (c - 1).max(1);
        let pca_comp = data.rows.min(data.cols).min(if data.rows > c { data.rows - c } else { data.rows });
        let (_pcav, pca_vec, pca_mean) = pca(data, pca_comp)?;
        let mut pca_proj = Matrix::new(data.rows, pca_vec.cols);
        for r in 0..data.rows {
            let row: Vec<f64> = (0..data.cols).map(|c| data.get(r, c)).collect();
            let p = project(&row, &pca_vec, &pca_mean);
            for (c, v) in p.iter().enumerate() {
                pca_proj.set(r, c, *v);
            }
        }
        let (eig_vals, eig_vecs_w, lda_mean) = lda(&pca_proj, labels, num_components.min(max_comp))?;
        let combined_eig = pca_vec.mul(&eig_vecs_w);
        let total_mean = pca_mean.clone();
        let overall_mean = if !lda_mean.is_empty() { lda_mean } else { pca_mean.clone() };
        let _ = overall_mean;
        let mut projections = Vec::with_capacity(data.rows);
        for r in 0..data.rows {
            let row: Vec<f64> = (0..data.cols).map(|c| data.get(r, c)).collect();
            projections.push(project(&row, &combined_eig, &total_mean));
        }
        let threshold = 0.7f64;
        Ok(Self {
            eigenvectors: combined_eig,
            eigenvalues: eig_vals,
            mean: total_mean,
            projections,
            labels: labels.to_vec(),
            names: names.to_vec(),
            threshold,
        })
    }

    pub fn predict(&self, face_vec: &[f64]) -> (usize, f64, Option<String>) {
        let proj = project(face_vec, &self.eigenvectors, &self.mean);
        let mut best_label = 0usize;
        let mut best_dist = f64::INFINITY;
        let mut second = f64::INFINITY;
        for (i, p) in self.projections.iter().enumerate() {
            let d = euclidean_distance(&proj, p);
            if d < best_dist {
                second = best_dist;
                best_dist = d;
                best_label = self.labels[i];
            } else if d < second {
                second = d;
            }
        }
        let avg_dist: f64 = self.projections.iter().enumerate().map(|(i, p)| {
            if self.labels[i] == best_label { euclidean_distance(&proj, p) } else { 0.0 }
        }).sum::<f64>() / self.labels.iter().filter(|&&l| l == best_label).count().max(1) as f64;
        let confidence = (1.0 / (1.0 + avg_dist / 100.0)).clamp(0.0, 1.0);
        let name = if avg_dist < 5000.0 {
            self.names.get(best_label).cloned()
        } else {
            None
        };
        (best_label, confidence, name)
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), BoxError> {
        let mut s = String::new();
        s.push_str("FISHERFACES_V1\n");
        s.push_str(&format!("components {} {}\n", self.eigenvectors.rows, self.eigenvectors.cols));
        s.push_str(&format!("eigenvalues {}\n", self.eigenvalues.len()));
        for &v in &self.eigenvalues { s.push_str(&format!("{:.10}\n", v)); }
        s.push_str(&format!("mean {}\n", self.mean.len()));
        for &v in &self.mean { s.push_str(&format!("{:.10}\n", v)); }
        s.push_str(&format!("eigenvectors {} {}\n", self.eigenvectors.rows, self.eigenvectors.cols));
        for r in 0..self.eigenvectors.rows {
            for c in 0..self.eigenvectors.cols {
                if c > 0 { s.push(' '); }
                s.push_str(&format!("{:.10}", self.eigenvectors.get(r, c)));
            }
            s.push('\n');
        }
        s.push_str(&format!("projections {} {}\n", self.projections.len(), if self.projections.is_empty() { 0 } else { self.projections[0].len() }));
        for p in &self.projections {
            for (i, v) in p.iter().enumerate() {
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
        if header != "FISHERFACES_V1" { return Err("Not a Fisherfaces model".into()); }
        let eigenvectors = parse_matrix(&mut lines, "components")?;
        let eigenvalues = parse_list_f64(&mut lines, "eigenvalues")?;
        let mean = parse_list_f64(&mut lines, "mean")?;
        let _ = parse_matrix(&mut lines, "eigenvectors");
        let projections = parse_matrix_vec(&mut lines, "projections")?;
        let labels = parse_list_usize(&mut lines, "labels")?;
        let names = parse_list_str(&mut lines, "names")?;
        let threshold_line = lines.next().ok_or("no threshold")?.trim();
        let threshold: f64 = threshold_line.strip_prefix("threshold ").ok_or("bad threshold")?.parse()?;
        Ok(Self {
            eigenvectors,
            eigenvalues,
            mean,
            projections,
            labels,
            names,
            threshold,
        })
    }
}
