use crate::image::{Image, Rect, BoxError};
use crate::linalg::Matrix;

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
}

pub fn detect_face_center_projection(gray: &[u8], w: usize, h: usize, face: &Rect) -> [Point; 5] {
    let fx = face.x as isize;
    let fy = face.y as isize;
    let fw = face.w as isize;
    let fh = face.h as isize;
    let cx = fx + fw / 2;
    let cy = fy + fh / 3;
    let eye_y = fy + fh / 3;
    let le = Point::new((fx + fw / 4) as f64, eye_y as f64);
    let re = Point::new((fx + 3 * fw / 4) as f64, eye_y as f64);
    let nose = Point::new(cx as f64, (fy + fh / 2) as f64);
    let ml = Point::new((fx + fw / 4) as f64, (fy + 5 * fh / 6) as f64);
    let mr = Point::new((fx + 3 * fw / 4) as f64, (fy + 5 * fh / 6) as f64);
    let _ = gray;
    let _ = w;
    let _ = h;
    [le, re, nose, ml, mr]
}

pub fn align_face(img: &Image, face: &Rect, points: &[Point; 5], out_w: usize, out_h: usize) -> Image {
    let (w, h) = (img.width, img.height);
    let channels = img.channels;
    let gray = img.to_grayscale();
    let le = points[0];
    let re = points[1];
    let dx = re.x - le.x;
    let dy = re.y - le.y;
    let angle = dy.atan2(dx);
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let cx = (le.x + re.x) / 2.0;
    let cy = (le.y + re.y) / 2.0;
    let eye_dist = (dx * dx + dy * dy).sqrt();
    let scale = if eye_dist > 1.0 {
        (out_w as f64 * 0.35) / eye_dist
    } else {
        out_w as f64 / face.w as f64
    };
    let out_cx = out_w as f64 * 0.5;
    let out_cy = out_h as f64 * 0.4;
    let mut out_data = vec![0u8; out_w * out_h];
    for y in 0..out_h {
        for x in 0..out_w {
            let rx = (x as f64 - out_cx) / scale;
            let ry = (y as f64 - out_cy) / scale;
            let src_x = cos_a * rx + sin_a * ry + cx;
            let src_y = -sin_a * rx + cos_a * ry + cy;
            let sx = src_x.round() as isize;
            let sy = src_y.round() as isize;
            if sx >= 0 && sx < w as isize && sy >= 0 && sy < h as isize {
                let g = if channels == 1 {
                    gray[(sy as usize) * w + (sx as usize)]
                } else {
                    let idx = ((sy as usize) * w + (sx as usize)) * channels;
                    let r = img.data[idx] as u32;
                    let g = img.data[idx + 1] as u32;
                    let b = img.data[idx + 2] as u32;
                    ((r * 77 + g * 150 + b * 29) / 256) as u8
                };
                out_data[y * out_w + x] = g;
            }
        }
    }
    Image::from_grayscale(out_w, out_h, out_data)
}

pub fn simple_crop_align(img: &Image, face: &Rect, out_w: usize, out_h: usize, pad: f32) -> Image {
    let pad_x = (face.w as f32 * pad) as i32;
    let pad_y = (face.h as f32 * pad) as i32;
    let x = (face.x - pad_x).max(0) as usize;
    let y = (face.y - pad_y).max(0) as usize;
    let w = (face.w + 2 * pad_x).min(img.width as i32 - x as i32) as usize;
    let h = (face.h + 2 * pad_y).min(img.height as i32 - y as i32) as usize;
    let cropped = img.crop(x, y, w, h);
    cropped.resize_bilinear(out_w, out_h)
}

pub fn preprocess_for_recognition(
    img: &Image,
    face: Option<&Rect>,
    size: (usize, usize),
) -> Vec<f64> {
    let cropped = match face {
        Some(r) => simple_crop_align(img, r, size.0, size.1, 0.15),
        None => img.resize_bilinear(size.0, size.1),
    };
    let gray = cropped.to_grayscale();
    let eq = crate::imgproc::histogram_equalize(&gray, size.0, size.1);
    crate::imgproc::normalize_face(&eq, size.0, size.1)
}

pub fn load_face_dataset(dir: &std::path::Path, size: (usize, usize)) -> Result<(Matrix, Vec<usize>, Vec<String>), BoxError> {
    let mut data = Vec::new();
    let mut labels = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut label_id = 0usize;
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let idx = names.iter().position(|n| n == &name).unwrap_or_else(|| {
                names.push(name);
                let l = label_id;
                label_id += 1;
                l
            });
            let mut files: Vec<_> = std::fs::read_dir(&path)?.filter_map(|e| e.ok()).collect();
            files.sort_by_key(|e| e.path());
            for f in files {
                let p = f.path();
                if let Some(ext) = p.extension() {
                    let e = ext.to_string_lossy().to_lowercase();
                    if e == "pgm" || e == "ppm" || e == "png" {
                        if let Ok(img) = Image::load_pgm(&p).or_else(|_| Image::load_ppm(&p)) {
                            let vec = preprocess_for_recognition(&img, None, size);
                            data.push(vec);
                            labels.push(idx);
                        }
                    }
                }
            }
        } else if let Some(ext) = path.extension() {
            let e = ext.to_string_lossy().to_lowercase();
            if e == "pgm" || e == "ppm" || e == "png" {
                if let Ok(img) = Image::load_pgm(&path).or_else(|_| Image::load_ppm(&path)) {
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    let base = name.split('_').next().unwrap_or(&name).to_string();
                    let idx = names.iter().position(|n| n == &base).unwrap_or_else(|| {
                        names.push(base);
                        let l = label_id;
                        label_id += 1;
                        l
                    });
                    let vec = preprocess_for_recognition(&img, None, size);
                    data.push(vec);
                    labels.push(idx);
                }
            }
        }
    }
    if data.is_empty() {
        return Err("No training images found".into());
    }
    let rows = data.len();
    let cols = data[0].len();
    let mut mat = Matrix::new(rows, cols);
    for (r, row_vec) in data.iter().enumerate() {
        for (c, &v) in row_vec.iter().enumerate() {
            mat.set(r, c, v);
        }
    }
    Ok((mat, labels, names))
}
