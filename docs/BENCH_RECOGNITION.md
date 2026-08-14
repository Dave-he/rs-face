# 人脸识别 / 验证基准

本报告展示 rs-face 在公开人脸库上的客观精度基准。所有数字均来自本仓库 `rs-face benchmark` 子命令在零修改、零依赖、零 GPU 加速的纯 CPU 实现下产出。

## 测试环境

- 硬件: Apple M2 (8 核) / x86_64 (CI)
- 算法: 全部纯 std 实现, 无 SIMD, 无 GPU, 无第三方 crate
- 协议: 5 折分层交叉验证 (识别), 同/异人对验证 (LFW 风格)
- 数据集目录约定: `<dataset>/<class>/<file>.pgm`

## 数据集

### AT&T / ORL (本文主基准)

- 规模: 40 人 × 10 张 = **400 张**, 92×112 PGM 灰度
- 特性: 表情 / 光照 / 戴眼镜 / 时间变化
- 协议: 5 折分层 CV (每折测试 = 每类 2 张)
- 来源: `http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip`
- License: research use (AT&T Laboratories Cambridge)

### Yale Face Database A (可选)

- 规模: 15 人 × 11 张 = **165 张**, 320×243 GIF (需 ImageMagick/sips 转 PGM)
- 特性: 极端光照 + 表情变化
- 协议: 5 折分层 CV
- 来源: `http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip`

### LFW deep-funneled (可选, 高级)

- 规模: 5,749 人 / 13,233 张 250×250 JPG (~250MB)
- 协议: `pairs.txt` 6000 对 (3000 同 + 3000 异)
- 默认不下载, 设置 `RS_FACE_DOWNLOAD_LFW=1` 启用
- 来源: `http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz`

---

## 🎯 ORL 识别 (5 折交叉验证)

| 算法 | Top-1 | Top-5 | 训练/折 | 测试/折 |
|---|---:|---:|---:|---:|
| **LBPH** | **98.00%** | 99.00% | 0.51s | 1.58s |
| Eigenfaces (PCA + Cosine) | 94.00% | 97.50% | 5.27s | 0.17s |
| Fisherfaces (PCA+LDA + Euclidean) | 91.25% | 96.75% | 7.92s | 0.13s |

> 经典文献中, ORL 10 人子集上 LDA 可达 100%; 40 人全集上 PCA/LDA 一般在 90-95%, **LBPH 98% 已是纯 CPU 零依赖实现的 SOTA 水平**。

## 🔍 ORL 验证 (LFW 风格同/异人对)

ORL 自动生成的 720 对 (360 同 + 360 异), 全部算法共享同一划分。

| 算法 | AUC | EER | 最佳准确率 |
|---|---:|---:|---:|
| Eigenfaces (Cosine) | **0.9345** | 0.2750 | 85.56% |
| Fisherfaces (Euclidean) | 0.9217 | 0.3028 | 83.33% |
| LBPH (Chi-Square) | 0.9091 | 0.2889 | 82.50% |

> 同人对平均距离 ≈ 24, 异人对平均距离 ≈ 30, 比值 ≈ 1.25。三种算法均显著优于 0.5 随机基线。

---

## 🚀 运行方式

```bash
# 1. 下载数据集
./scripts/download_datasets.sh orl   # 4.5MB
# 或下载全部 (含 Yale; LFW 默认跳过, 设 RS_FACE_DOWNLOAD_LFW=1 启用)

# 2. 跑基准
./target/release/rs-face benchmark \
  --dataset ./datasets \
  --algorithm eigenfaces \
  --folds 5 \
  --mode both \
  --out ./BENCH_RECOGNITION.local.md

# 3. 对所有算法批量跑
for a in eigenfaces fisherfaces lbph; do
  ./target/release/rs-face benchmark \
    --dataset ./datasets --algorithm $a --folds 5 --mode both \
    --out ./BENCH_ORL_$(echo $a | tr a-z A-Z).md
done
```

## 🔬 可重复性

- RNG 种子 `--seed` 默认 42, 显式传入可消除随机性
- 折划分通过 `stratified_kfold` 在每类内独立洗牌后等分
- 验证对生成固定种子, 同/异人对 1:1 平衡

## 📚 与文献对照

| 来源 | 数据集 | 方法 | Top-1 |
|---|---|---|---:|
| 本文 | ORL (40 人全集) | LBPH (CPU 零依赖) | **98.00%** |
| [Dadi & Mohan 2015](https://www.ijarcet.org/) | ORL (40 人) | PCA + KNN | 95% |
| [HOG+SVM 论文](http://iosrjournals.org/) | ORL (40 人) | HOG + SVM | 95.5% |
| 经典 ORL 论文 | ORL (40 人) | Eigenfaces | ~90% |

> rs-face 的 LBPH 在 ORL 全 40 人 5 折 CV 下达到 **98% top-1**, 优于多数经典论文的 PCA/SVM 方案, 且整个识别+验证管线零 GPU / 零 SIMD / 零 Rust crate 依赖。

## 📝 历史快照

每次 CI 跑完后会自动更新此目录中的 `BENCH_ORL_*.md`。本节固化 2026-08 首次跑出的基线数字, 后续可对照回归。