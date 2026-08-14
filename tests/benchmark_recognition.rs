// 集成测试: 跑 `rs-face benchmark` 子命令在公开人脸库上的精度。
// 测试默认标记 `#[ignore]`, 需要本地已有数据集或启用下载。
// CI 通过 `cargo test --release -- --ignored` 显式触发。

use std::process::Command;
use std::time::Duration;

const DATASETS_DIR: &str = "./datasets";

fn bench_exists() -> bool {
    let p = format!("{}/s1/1.pgm", DATASETS_DIR);
    std::path::Path::new(&p).exists()
}

fn run_bench(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rs-face"))
        .args(args)
        .output()
        .expect("run rs-face benchmark")
}

#[test]
fn help_includes_benchmark() {
    let out = Command::new(env!("CARGO_BIN_EXE_rs-face"))
        .args(&["benchmark", "--help"])
        .output()
        .expect("run");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("benchmark"), "--help should mention benchmark subcommand");
    assert!(s.contains("--dataset"), "--help should document --dataset");
}

#[test]
fn smoke_benchmark_runs() {
    if !bench_exists() {
        eprintln!(
            "[skip] ORL 数据集不存在: 跑 ./scripts/download_datasets.sh orl 后重试"
        );
        return;
    }
    // 2 折 + 小配对上限, 期望 < 60s 完成
    let out = run_bench(&[
        "benchmark",
        "--dataset", DATASETS_DIR,
        "--algorithm", "eigenfaces",
        "--folds", "2",
        "--max-pairs", "200",
        "--out", "/tmp/rs_face_bench_smoke.md",
    ]);
    assert!(out.status.success(), "benchmark must succeed: {}",
        String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("top-1="), "must print top-1: {}", stdout);
    assert!(stdout.contains("AUC="), "must print AUC: {}", stdout);
}

#[test]
#[ignore = "完整基准, 默认不跑; 启用: cargo test --release -- --ignored full_bench"]
fn full_bench_orl_eigenfaces() {
    if !bench_exists() {
        panic!(
            "ORL 数据集不存在。先跑: ./scripts/download_datasets.sh orl"
        );
    }
    let t = std::time::Instant::now();
    let out = run_bench(&[
        "benchmark",
        "--dataset", DATASETS_DIR,
        "--algorithm", "eigenfaces",
        "--folds", "5",
        "--max-pairs", "2000",
        "--out", "/tmp/rs_face_bench_eigen.md",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{}", stdout);
    assert!(stdout.contains("top-1="));
    // Top-1 ≥ 88% (ORL 40 人 PCA, 经典基线 ~94%, 留出余量)
    let top1 = extract_metric(&stdout, "top-1=");
    assert!(top1 >= 88.0, "Eigenfaces top-1={:.2}% 应 ≥ 88%", top1);
    // 5 折训练 + 测试 ≤ 120s (M2 / 类似 x86)
    assert!(t.elapsed() < Duration::from_secs(180), "full bench too slow: {:?}", t.elapsed());
}

#[test]
#[ignore = "完整基准"]
fn full_bench_orl_lbph() {
    if !bench_exists() { panic!("先跑 ./scripts/download_datasets.sh orl"); }
    let out = run_bench(&[
        "benchmark",
        "--dataset", DATASETS_DIR,
        "--algorithm", "lbph",
        "--folds", "5",
        "--out", "/tmp/rs_face_bench_lbph.md",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{}", stdout);
    let top1 = extract_metric(&stdout, "top-1=");
    // LBPH 在 ORL 应 ≥ 95%
    assert!(top1 >= 95.0, "LBPH top-1={:.2}% 应 ≥ 95%", top1);
}

fn extract_metric(stdout: &str, key: &str) -> f64 {
    let after = stdout.split(key).nth(1).unwrap_or("0");
    let num: String = after.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    num.parse().unwrap_or(0.0)
}