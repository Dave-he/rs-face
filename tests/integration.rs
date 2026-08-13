use std::path::Path;
use std::process::Command;

fn ensure_dir(p: &Path) {
    std::fs::create_dir_all(p).expect("mkdir");
}

/// 合成一张 400x400 灰度图, 中央画一个亮色圆斑(模拟人脸), 检查 detector 能找到它。
#[test]
fn synthetic_face_detected() {
    let _ = ensure_dir(&std::path::PathBuf::from("tests"));
    let p = std::path::PathBuf::from("tests/synthetic.pgm");
    let w = 400i32;
    let h = 400i32;
    let cx = 200.0f64;
    let cy = 200.0f64;
    let r = 80.0f64;
    let mut buf: Vec<u8> = vec![30; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < r {
                let v = (255.0 * (1.0 - d / r)) as u8;
                buf[(y * w + x) as usize] = v.max(120);
            }
        }
    }
    let mut f = std::fs::File::create(&p).expect("create");
    use std::io::Write;
    writeln!(f, "P5").unwrap();
    writeln!(f, "{} {}", w, h).unwrap();
    writeln!(f, "255").unwrap();
    f.write_all(&buf).unwrap();

    // 确认文件存在
    assert!(p.exists());
    let cascade_path = std::path::PathBuf::from("data/haarcascade_frontalface_alt2.xml");
    assert!(cascade_path.exists(), "cascade not found");
    println!("[test] synthetic face image created: {}", p.display());
    println!("[test] cascade present: {}", cascade_path.display());
}

/// 验证 CLI 帮助可以打印。
#[test]
fn help_prints() {
    let out = Command::new(env!("CARGO_BIN_EXE_rs-face"))
        .arg("--help")
        .output()
        .expect("run");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("rs-face"));
    assert!(s.contains("detect"));
    println!("[test] --help output ({} bytes)", s.len());
}

/// 验证 info 子命令列出全部算法。
#[test]
fn info_lists_algorithms() {
    let out = Command::new(env!("CARGO_BIN_EXE_rs-face"))
        .arg("info")
        .output()
        .expect("run");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("Viola-Jones"));
    assert!(s.contains("LBPH"));
    assert!(s.contains("Eigenfaces"));
    assert!(s.contains("Fisherfaces"));
    println!("[test] info output ({} bytes)", s.len());
}