# 数据集目录

此目录**默认 gitignore**, 用于存放公开人脸库, 供 `rs-face benchmark` 子命令评估精度。

## 预期目录结构

```
datasets/
├── s1/        # 类别 1 (ORL 命名风格: subject 1)
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ...    # 每类所有样本文件, PGM / PPM / PNG 均可
├── s2/
├── ...
└── s40/       # ORL 共 40 类, 每类 10 张 = 400 张
```

`rs-face benchmark` 通过以下规则发现类别:
1. 每个直接子目录 = 一个类别 (目录名即类别标签)
2. 子目录内的 `.pgm` / `.ppm` / `.png` 文件 = 该类别的样本

## 下载

```bash
# AT&T / ORL (主推, 4.5MB)
./scripts/download_datasets.sh orl

# LFW deep-funneled (250MB, 验证任务)
RS_FACE_DOWNLOAD_LFW=1 ./scripts/download_datasets.sh lfw

# Yale Face Database A (官方 URL 已 404, 需手动获取 yalefaces.zip
#                       并解压到 datasets/yalefaces/)
```

## 跑基准

```bash
rs-face benchmark \
  --dataset ./datasets \
  --algorithm eigenfaces|fisherfaces|lbph \
  --folds 5 \
  --mode both \
  --out ./BENCH_RECOGNITION.local.md
```

## 详细结果

每次 CI 跑完会把报告上传为 artifact, 最新基线见
[`docs/BENCH_RECOGNITION.md`](../docs/BENCH_RECOGNITION.md)。