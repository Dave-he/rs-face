use crate::image::BoxError;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![1.0; rows * cols],
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] + other.data[i];
        }
        out
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] - other.data[i];
        }
        out
    }

    pub fn mul_scalar(&self, s: f64) -> Matrix {
        let mut out = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            out.data[i] = self.data[i] * s;
        }
        out
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut out = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                out.set(i, j, sum);
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    pub fn row(&self, r: usize) -> Vec<f64> {
        self.data[r * self.cols..(r + 1) * self.cols].to_vec()
    }

    pub fn col(&self, c: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            out.push(self.get(r, c));
        }
        out
    }

    pub fn set_row(&mut self, r: usize, row: &[f64]) {
        for (c, &v) in row.iter().enumerate() {
            self.set(r, c, v);
        }
    }

    pub fn set_col(&mut self, c: usize, col: &[f64]) {
        for (r, &v) in col.iter().enumerate() {
            self.set(r, c, v);
        }
    }

    pub fn mean_row(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.cols];
        for c in 0..self.cols {
            let mut sum = 0.0;
            for r in 0..self.rows {
                sum += self.get(r, c);
            }
            out[c] = sum / self.rows as f64;
        }
        out
    }

    pub fn mean_col(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c);
            }
            out[r] = sum / self.cols as f64;
        }
        out
    }

    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for &v in &self.data {
            sum += v * v;
        }
        sum.sqrt()
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        let mut t = 0.0;
        for i in 0..self.rows {
            t += self.get(i, i);
        }
        t
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn vec_norm(a: &[f64]) -> f64 {
    let mut sum = 0.0;
    for &v in a {
        sum += v * v;
    }
    sum.sqrt()
}

pub fn vec_normalize(a: &[f64]) -> Vec<f64> {
    let n = vec_norm(a);
    if n < 1e-12 {
        a.to_vec()
    } else {
        a.iter().map(|&v| v / n).collect()
    }
}

pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

pub fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|&v| v * s).collect()
}

pub fn solve_symmetric_eigen(matrix: &Matrix, max_iter: usize) -> (Vec<f64>, Matrix) {
    let n = matrix.rows;
    assert_eq!(n, matrix.cols);
    let mut a = matrix.clone();
    let mut v = Matrix::identity(n);
    for _ in 0..max_iter {
        let (p, q, max_off) = find_max_off_diagonal(&a);
        if max_off < 1e-10 {
            break;
        }
        let theta = (a.get(q, q) - a.get(p, p)) / (2.0 * a.get(p, q));
        let t = if theta.abs() < 1e-12 {
            1.0
        } else if theta > 0.0 {
            1.0 / (theta + (theta * theta + 1.0).sqrt())
        } else {
            1.0 / (theta - (theta * theta + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);
        let app = a.get(p, p);
        let aqq = a.get(q, q);
        let apq = a.get(p, q);
        a.set(p, p, app - t * apq);
        a.set(q, q, aqq + t * apq);
        a.set(p, q, 0.0);
        a.set(q, p, 0.0);
        for i in 0..n {
            if i != p && i != q {
                let aip = a.get(i, p);
                let aiq = a.get(i, q);
                a.set(i, p, aip - s * (aiq + tau * aip));
                a.set(p, i, a.get(i, p));
                a.set(i, q, aiq + s * (aip - tau * aiq));
                a.set(q, i, a.get(i, q));
            }
        }
        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, vip - s * (viq + tau * vip));
            v.set(i, q, viq + s * (vip - tau * viq));
        }
    }
    let mut eigenvalues: Vec<(f64, usize)> = (0..n).map(|i| (a.get(i, i), i)).collect();
    eigenvalues.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let eigen_values: Vec<f64> = eigenvalues.iter().map(|(v, _)| *v).collect();
    let mut eigen_vectors = Matrix::new(n, n);
    for (new_col, &(_, old_col)) in eigenvalues.iter().enumerate() {
        for r in 0..n {
            eigen_vectors.set(r, new_col, v.get(r, old_col));
        }
    }
    (eigen_values, eigen_vectors)
}

fn find_max_off_diagonal(a: &Matrix) -> (usize, usize, f64) {
    let mut max = 0.0;
    let mut p = 0;
    let mut q = 1;
    for i in 0..a.rows {
        for j in (i + 1)..a.cols {
            let v = a.get(i, j).abs();
            if v > max {
                max = v;
                p = i;
                q = j;
            }
        }
    }
    (p, q, max)
}

pub fn pca(
    data: &Matrix,
    num_components: usize,
) -> Result<(Vec<f64>, Matrix, Vec<f64>), BoxError> {
    let n = data.rows;
    let d = data.cols;
    if n == 0 {
        return Err("PCA: empty data".into());
    }
    let mean = data.mean_row();
    let mut centered = Matrix::new(n, d);
    for r in 0..n {
        for c in 0..d {
            centered.set(r, c, data.get(r, c) - mean[c]);
        }
    }
    let cov = if n < d {
        let ct = centered.transpose();
        let small = centered.mul(&ct).mul_scalar(1.0 / (n - 1) as f64);
        let (eig_vals, eig_vecs_small) = solve_symmetric_eigen(&small, 1000);
        let mut eig_vecs_large = ct.mul(&eig_vecs_small);
        for c in 0..eig_vecs_large.cols {
            let mut norm_sq = 0.0;
            for r in 0..eig_vecs_large.rows {
                norm_sq += eig_vecs_large.get(r, c) * eig_vecs_large.get(r, c);
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-12 {
                for r in 0..eig_vecs_large.rows {
                    eig_vecs_large.set(r, c, eig_vecs_large.get(r, c) / norm);
                }
            }
        }
        (eig_vals, eig_vecs_large)
    } else {
        let cov = centered.transpose().mul(&centered).mul_scalar(1.0 / (n - 1) as f64);
        solve_symmetric_eigen(&cov, 5000)
    };
    let k = num_components.min(cov.0.len()).min(n);
    let eigen_values: Vec<f64> = cov.0.into_iter().take(k).collect();
    let mut eigen_vectors = Matrix::new(d, k);
    for c in 0..k {
        for r in 0..d {
            eigen_vectors.set(r, c, cov.1.get(r, c));
        }
    }
    Ok((eigen_values, eigen_vectors, mean))
}

pub fn project(data: &[f64], eigenvectors: &Matrix, mean: &[f64]) -> Vec<f64> {
    let d = data.len();
    let k = eigenvectors.cols;
    assert_eq!(d, mean.len());
    assert_eq!(d, eigenvectors.rows);
    let mut out = vec![0.0; k];
    for j in 0..k {
        let mut sum = 0.0;
        for i in 0..d {
            sum += (data[i] - mean[i]) * eigenvectors.get(i, j);
        }
        out[j] = sum;
    }
    out
}

pub fn reconstruct(projection: &[f64], eigenvectors: &Matrix, mean: &[f64]) -> Vec<f64> {
    let d = eigenvectors.rows;
    let k = projection.len();
    assert_eq!(k, eigenvectors.cols);
    let mut out = mean.to_vec();
    for i in 0..d {
        for j in 0..k {
            out[i] += projection[j] * eigenvectors.get(i, j);
        }
    }
    out
}

pub fn lda(
    data: &Matrix,
    labels: &[usize],
    num_components: usize,
) -> Result<(Vec<f64>, Matrix, Vec<f64>), BoxError> {
    let n = data.rows;
    let d = data.cols;
    if n == 0 {
        return Err("LDA: empty data".into());
    }
    assert_eq!(labels.len(), n);
    let classes: std::collections::BTreeSet<usize> = labels.iter().cloned().collect();
    let c = classes.len();
    if c < 2 {
        return Err("LDA: need at least 2 classes".into());
    }
    let overall_mean = data.mean_row();
    let mut class_mean = std::collections::BTreeMap::new();
    let mut class_count = std::collections::BTreeMap::new();
    for (&lbl, row) in labels.iter().zip(0..n) {
        class_count.entry(lbl).or_insert(0);
        *class_count.get_mut(&lbl).unwrap() += 1;
        let e = class_mean.entry(lbl).or_insert_with(|| vec![0.0; d]);
        for col in 0..d {
            e[col] += data.get(row, col);
        }
    }
    for (&lbl, mean_vec) in class_mean.iter_mut() {
        let cnt = class_count[&lbl] as f64;
        for v in mean_vec.iter_mut() {
            *v /= cnt;
        }
    }
    let mut sb = Matrix::new(d, d);
    for (&lbl, mean_vec) in &class_mean {
        let cnt = class_count[&lbl] as f64;
        let mut diff = vec![0.0; d];
        for i in 0..d {
            diff[i] = mean_vec[i] - overall_mean[i];
        }
        for i in 0..d {
            for j in 0..d {
                sb.set(i, j, sb.get(i, j) + cnt * diff[i] * diff[j]);
            }
        }
    }
    let mut sw = Matrix::new(d, d);
    for (row, &lbl) in labels.iter().enumerate() {
        let mean_vec = &class_mean[&lbl];
        for i in 0..d {
            let xi = data.get(row, i) - mean_vec[i];
            for j in 0..d {
                let xj = data.get(row, j) - mean_vec[j];
                sw.set(i, j, sw.get(i, j) + xi * xj);
            }
        }
    }
    let max_eig = 1e-8;
    for i in 0..d {
        sw.set(i, i, sw.get(i, i) + max_eig);
    }
    let sw_inv = invert_matrix_gauss(&sw).ok_or("LDA: failed to invert SW")?;
    let mat = sw_inv.mul(&sb);
    let symmetrize = mat.transpose().mul(&mat);
    let (eig_vals, eig_vecs) = solve_symmetric_eigen(&symmetrize, 3000);
    let k = num_components.min(c - 1).min(d);
    let eigenvalues: Vec<f64> = eig_vals.into_iter().take(k).collect();
    let mut eigenvectors = Matrix::new(d, k);
    for col in 0..k {
        for r in 0..d {
            eigenvectors.set(r, col, eig_vecs.get(r, col));
        }
    }
    Ok((eigenvalues, eigenvectors, overall_mean))
}

fn invert_matrix_gauss(a: &Matrix) -> Option<Matrix> {
    let n = a.rows;
    assert_eq!(n, a.cols);
    let m = 2 * n;
    let mut aug = Matrix::new(n, m);
    for i in 0..n {
        for j in 0..n {
            aug.set(i, j, a.get(i, j));
        }
        aug.set(i, n + i, 1.0);
    }
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = aug.get(col, col).abs();
        for r in (col + 1)..n {
            let v = aug.get(r, col).abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < 1e-12 {
            return None;
        }
        if pivot_row != col {
            for j in 0..m {
                let tmp = aug.get(col, j);
                aug.set(col, j, aug.get(pivot_row, j));
                aug.set(pivot_row, j, tmp);
            }
        }
        let inv_p = 1.0 / aug.get(col, col);
        for j in 0..m {
            aug.set(col, j, aug.get(col, j) * inv_p);
        }
        for r in 0..n {
            if r != col {
                let factor = aug.get(r, col);
                if factor.abs() > 1e-12 {
                    for j in 0..m {
                        aug.set(r, j, aug.get(r, j) - factor * aug.get(col, j));
                    }
                }
            }
        }
    }
    let mut inv = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            inv.set(i, j, aug.get(i, n + j));
        }
    }
    Some(inv)
}
