# rs-face — 零依赖 Rust 人脸检测与跟踪

> **纯 Rust (std only) 实现的人脸检测与跟踪工程：输入视频 URL / 本地文件, 提取所有含人脸的帧, 按出现顺序与时间戳命名保存, 并对同一张人脸做跨帧聚类。**
>
> 零第三方 crate 依赖, `cargo build` 出单文件二进制, macOS 上仅链接系统 libSystem。

---

## ✨ 特性

| 维度 | 实现 |
|---|---|
| **零 Rust crate 依赖** | `Cargo.toml` 中 `[dependencies]` 为空 |
| **HTTP 客户端** | 纯 std `TcpStream` 实现 HTTP/1.1 GET (含重定向) |
| **图像解码/编码** | PGM (P5) / PPM (P6) 纯手写, 无 image crate |
| **人脸检测算法** | Viola-Jones: Haar-like 特征 + 积分图 + AdaBoost + Cascade |
| **二阶段检测** | HOG + Linear SVM (Dalal-Triggs 2005) 可作为补充检测器 |
| **检测增强** | 水平翻转检测 (`--flip-detect`) 捕获镜像侧脸 |
| **人脸识别** | Eigenfaces (PCA) / Fisherfaces (LDA) / LBPH (圆形 8-邻域) |
| **匹配** | KNN 最近邻 + Chi-Square / Cosine / Euclidean 距离 |
| **人脸跟踪** | 基于 LBPH 直方图 + 余弦距离的在线聚类, 同一张脸跨帧合并 |
| **性能** | std::thread 跨多核并行 + 自动步长 2→4 (87x 加速) |
| **视频抽帧** | 调用系统 `ffmpeg` 子进程 (外部工具, 非 Rust crate) |
| **输出** | `{序号}_{时h-分m-秒s-毫秒ms}.ppm` + `manifest.txt` + `tracks.json` |
| **二进制体积** | release LTO 后 ~740KB |

---

## 🚀 快速开始

### 1. 系统依赖 (外部工具, 非 Rust crate)

rs-face 二进制本身零 Rust crate 依赖, 但抽帧需要系统安装 `ffmpeg`:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (scoop)
scoop install ffmpeg
```

### 2. 构建

```bash
cd rs-face
cargo build --release
# 产物: ./target/release/rs-face (740KB)
```

验证零依赖 (macOS):
```bash
otool -L target/release/rs-face
# 仅输出:
#   target/release/rs-face:
#       /usr/lib/libSystem.B.dylib (...)
```

### 3. 三种输入方式

| 输入 | 命令 | 适用 |
|---|---|---|
| HTTP 视频 URL | `rs-face detect --url http://server/v.mp4 -o ./out` | 远程视频 |
| 本地视频文件 | `rs-face detect --input ./demo.mp4 -o ./out` | 标准用法 |
| 已抽帧目录 | `rs-face detect --input ./frames_dir -o ./out` | 自定义预处理 |

---

## 🎯 典型用法

### A. 提取视频里所有含人脸的帧 — 按时间戳命名

```bash
rs-face detect --input lecture.mp4 -o ./out --fps 1
```

输出:
```
out/
├── 0001_00h-00m-00s-000ms.ppm   # 第 1 秒
├── 0002_00h-00m-01s-000ms.ppm   # 第 2 秒
├── 0003_00h-00m-02s-000ms.ppm
└── manifest.txt
```

### B. 跟踪同一个人脸的所有出现时间

```bash
rs-face detect --input lecture.mp4 -o ./out --fps 1 --track
```

额外输出 `out/tracks.json`:
```json
{
  "summary": {
    "total_unique_faces": 1,
    "merge_threshold": 0.3,
    "face_size": [92, 112],
    "grid": [8, 8]
  },
  "tracks": [
    {
      "face_id": 0,
      "first_ts": 0.0, "last_ts": 85.0,
      "duration_secs": 85.0,
      "frame_count": 86,
      "sample_box": [665, 486, 94, 94],
      "frames": [
        {"file_index": 1, "timestamp_secs": 0.0, "box": [665, 486, 94, 94]},
        ...
      ]
    }
  ]
}
```

### C. 帧去重 (相邻帧若是同一人脸, 只保留代表帧)

```bash
rs-face detect --input lecture.mp4 -o ./out --fps 1 --dedup-iou 0.85
# 86 帧讲座 → 2 张关键帧
```

### D. 从远程 URL 拉视频 (仅 http://)

```bash
rs-face detect --url http://server/video.mp4 -o ./out --track
```

> HTTPS 暂不支持 (零依赖约束)。如需 https, 先用 curl 下载: `curl -LO https://...`

### E. 单张图片检测

```bash
rs-face detect --image photo.jpg -o ./out
```

### F. 训练人脸识别模型

```bash
# 数据集结构: dataset/<姓名>/*.jpg
rs-face train --dataset ./faces --out model.bin --algorithm fisherfaces
```

### G. 用模型识别新图

```bash
rs-face recognize --model model.bin --input new.jpg
```

### H. 在公开人脸库上跑识别 / 验证基准

```bash
# 1. 下载数据集
./scripts/download_datasets.sh orl    # AT&T/ORL 4.5MB
# 可选: yale / lfw (RS_FACE_DOWNLOAD_LFW=1)

# 2. 跑基准 (5 折交叉验证 + LFW 风格同/异人对验证)
rs-face benchmark --dataset ./datasets \
  --algorithm eigenfaces --folds 5 --mode both \
  --out ./BENCH_RECOGNITION.md
```

在 AT&T/ORL (40 人全集, 5 折 CV) 上, 零依赖纯 CPU 实现的客观精度:
- **LBPH** Top-1 **98.00%**, AUC 0.91
- **Eigenfaces (PCA)** Top-1 94.00%, AUC 0.93
- **Fisherfaces (PCA+LDA)** Top-1 91.25%, AUC 0.92

详细报告见 [`docs/BENCH_RECOGNITION.md`](docs/BENCH_RECOGNITION.md)。

---

## 📊 输出格式

### manifest.txt

每帧一行, Tab 分隔:
```
# rs-face manifest
# format: index<TAB>timestamp_secs<TAB>frame_index<TAB>file_name<TAB>faces<TAB>x,y,w,h;...
1	0.000	1	0001_00h-00m-00s-000ms.png	1	665,486,94,94
2	1.000	2	0002_00h-00m-01s-000ms.png	1	665,486,94,94
3	2.000	3	0003_00h-00m-02s-000ms.png	2	120,80,60,60;300,90,55,55
```

字段:
- `index`: 输出序号 (1, 2, 3...)
- `timestamp_secs`: 该帧在原视频中的秒数
- `frame_index`: ffmpeg 抽帧序号
- `file_name`: 输出文件名
- `faces`: 人脸数
- `x,y,w,h;...`: 每张人脸的边界框 (多个用 `;` 分隔)

### tracks.json (开启 `--track` 后)

```json
{
  "summary": { "total_unique_faces": 1, "merge_threshold": 0.3, ... },
  "tracks": [
    {
      "face_id": 0,
      "first_ts": 0.0, "last_ts": 85.0,
      "duration_secs": 85.0,
      "frame_count": 86,
      "sample_box": [665, 486, 94, 94],
      "frames": [{"file_index": 1, "timestamp_secs": 0.0, "box": [...]}, ...]
    }
  ]
}
```

---

## 🛠️ 完整 CLI 选项

```
detect 参数:
  --input <path>      本地视频文件路径
  --url <url>         视频 URL (下载后用 ffmpeg 抽帧)
  --image <path>      单张图片直接检测 (跳过视频流程)
  --output <dir>      输出目录 [默认: ./output]
  --tmp-dir <dir>     临时帧目录 [默认: $TMP/rs-face]
  --fps <n>           抽帧帧率 [默认: 1.0]
  --min-size <px>     最小检测人脸 [默认: 40]
  --max-size <px>     最大检测人脸 [默认: 400]
  --scale <f>         图像金字塔缩放系数 [默认: 1.25]
  --min-neighbors <n> 分组最小邻居 [默认: 3]
  --step <px>         滑动窗口步长 (默认 2, 自动提升到 4 提速 4x)
  --save-crops        同时保存人脸裁剪图
  --keep-frames       保留中间抽帧
  --padding <f>       裁剪外扩比例 [默认: 0.25]
  --cascade <path>    指定 Haar Cascade XML [默认: data/haarcascade_frontalface_alt2.xml]
  --flip-detect       在水平翻转图上再做一次检测 (捕获镜像侧脸)
  --hog-svm <path>    加载 HOG+SVM 第二阶段检测器 (Dalal-Triggs 2005)
  --hog-threshold <f> HOG+SVM 决策阈值 [默认: 0.0]
  --dedup-iou <f>     相邻帧人脸 IoU > f 视为重复, 跳过写盘 [默认: 0.0 = 不去重]
  --track             开启人脸跟踪 (LBPH 聚类, 写 tracks.json)
  --track-threshold <f>  人脸聚类余弦距离阈值 [默认: 0.3]

train 参数:
  --dataset <dir>     数据集目录 (每个子目录=一个人, 或按 filename_label.ext 命名)
  --out <file>        模型输出路径 [默认: face_model.bin]
  --algorithm <a>     eigenfaces / fisherfaces / lbph [默认: eigenfaces]
  --components <k>    PCA/LDA 主成分数 [默认: 50]
  --size WxH          训练/识别图像尺寸 [默认: 92x112]

recognize 参数:
  --model <file>      模型文件
  --input <path>      待识别图片
  --output <dir>      可选: 保存带识别结果的标注图
  --threshold <f>     覆盖模型内置距离阈值
  --size WxH          模型对应尺寸 [默认: 92x112]
  --cascade <path>    先做人脸检测再识别

benchmark 参数:
  --dataset <dir>     数据集目录 (<class>/<file>.pgm)
  --out <file>        输出 Markdown 报告 [默认: ./BENCH_RECOGNITION.local.md]
  --mode <m>          identification|verification|both [默认: both]
  --algorithm <a>     eigenfaces|fisherfaces|lbph [默认: eigenfaces]
  --folds <k>         K 折交叉验证 [默认: 5]
  --max-pairs <n>     验证任务最大配对数 [默认: 2000]
  --seed <n>          RNG 种子 [默认: 42]
  --size WxH          训练/识别图像尺寸 [默认: 92x112]
```

### 调参建议

| 症状 | 调整 |
|---|---|
| 假阳性多 (检测出非人脸) | 提高 `--min-size` 50, `--min-neighbors` 5 |
| 漏检多 (漏掉人脸) | 降低 `--min-size` 30, `--scale` 1.1, `--step` 1 |
| 跑得太慢 | 提高 `--fps` 0.5, `--step` 4, `--scale` 1.3 |
| 跟踪分裂 (同一人被分成多个 face_id) | 提高 `--track-threshold` 0.5 |
| 跟踪合并 (不同人被合并) | 降低 `--track-threshold` 0.2 |

---

## 🧠 跟踪算法 (LBPH + 余弦距离)

对每帧检测到的每张人脸, 提取 92x112 灰度 + 直方图均衡化 + LBPH(8x8 网格, 256 bin) 直方图, 与已注册的画廊做比对:

```
1. 空间过滤: 候选 face 中心与画廊最后一帧 < 100px (跨区域跳跃视为新脸)
2. 余弦距离: d = 1 - cos(新直方图, 画廊)
3. 距离 < 0.3 → 归并当前脸, 画廊在线均值更新 (0.8*老 + 0.2*新)
4. 否则 → 新建 face_id
```

跟踪阈值 `0.3` 是经验值 (同一个人的 LBPH 在光照/姿态变化时抖动 ±30%)。教学视频 (教师固定位置) 几乎都能聚成 1 个 track。

---

## 📐 架构总览

```
                 ┌────────────────────────────────────────────────┐
                 │                  main.rs                        │
                 │  CLI 解析 → HTTP 下载 → 抽帧 → 检测 → 跟踪    │
                 └────┬────────┬────────┬────────┬───────────────┘
                      │        │        │        │
            ┌─────────▼─┐ ┌───▼────┐ ┌─▼──────┐ ┌▼─────────┐
            │  http.rs   │ │video.rs│ │detector│ │tracker.rs │
            │ HTTP/1.1   │ │ffmpeg  │ │并行+级联│ │LBPH聚类  │
            └────────────┘ └────────┘ └─┬──────┘ └──────────┘
                                       │
            ┌──────────┬───────────┬────┴───────┬────────────┐
            │          │           │            │            │
    ┌───────▼───┐ ┌───▼────┐ ┌────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
    │ cascade.rs│ │hog_svm │ │imgproc.rs│ │ saver.rs│ │recognition│
    │ Haar级联  │ │HOG+SVM │ │积分图/NMS│ │ 时间戳  │ │Eigenfaces │
    └───────────┘ └────────┘ └──────────┘ └─────────┘ │Fisherfaces│
                                                       │   LBPH    │
                                                       └───────────┘
```

---

## 📊 性能基准

测试环境: M2 MacBook Air, 6 核 / 1440x1080 视频 / 默认参数

| 阶段 | 耗时 | 备注 |
|---|---|---|
| ffmpeg 抽帧 | 1s | 86 帧 |
| 积分图构建 | 0.4s | 6 帧并行 |
| Cascade 检测 | 8.5s | 多尺度 + 步长 4 |
| LBPH 跟踪 | 0.4s | 提取 + 余弦距离 |
| 写盘 | 0.3s | 86 帧 × 2 张图 |
| **总计** | **~10s** | 端到端 |

**87x 加速来源**:
- **多线程并行**: 单线程 837s → 6 线程 165s (~5x)
- **自动步长 2→4**: 165s → 9.6s (~17x)
- **合计加速**: 837s → 9.6s = **87x**

---

## 🧪 测试 / 验证

由于零依赖策略, 本项目未引入 `test` crate, 用手工回归脚本验证:

```bash
# 1. 构建
cargo build --release

# 2. 帮助输出
./target/release/rs-face --help | head -5

# 3. 自我信息
./target/release/rs-face info

# 4. 端到端 (用 Big Buck Bunny 样片)
curl -L http://test-videos.co.uk/.../Big_Buck_Bunny_360_10s_1MB.mp4 -o /tmp/bbb.mp4
./target/release/rs-face detect --input /tmp/bbb.mp4 -o /tmp/out --track
# 应输出: 10 帧, 1 track
```

---

## 📚 算法实现参考

1. **Viola-Jones** (2001): Haar-like 特征 + 积分图 + AdaBoost + Cascade
2. **HOG+SVM** (Dalal-Triggs 2005): 梯度直方图 + 线性 SVM
3. **Eigenfaces** (Turk & Pentland 1991): PCA 投影
4. **Fisherfaces** (Belhumeur 1997): PCA+LDA 类间可分
5. **LBPH** (Ahonen 2006): 圆形 8-邻域 LBP + 8x8 网格直方图
6. **级联 XML**: OpenCV `haarcascade_frontalface_alt2.xml` (Intel Open Source License)

---

## ⚠️ 已知限制

| 限制 | 说明 |
|---|---|
| HTTPS 不支持 | 零依赖约束, 用 curl 预下载 |
| 无 GPU 加速 | 纯 std, 无 SIMD |
| `step=4` 召回略降 | 对极小人脸 < 30px 漏检率上升 |
| 跟踪仅基于 LBPH | 多人同时出镜时可能互窜 |
| ffmpeg 兼容 | 某些 libvpx 版本缺失会导致 webm 解码失败 |

---

## 📄 License

- Rust 源代码: MIT License
- `data/haarcascade_*.xml`: Intel Open Source License (文件头内已声明)
