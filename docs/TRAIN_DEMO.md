# 训练 + 识别完整示例

## 1. 准备数据集

数据集结构: `<root>/<姓名>/<图片>.pgm`

```bash
mkdir -p dataset/teacher_wang dataset/teacher_li
# 把每个老师的人脸图片放到对应目录
```

每张图必须是 92x112 灰度 (与训练尺寸一致)。可以用 `--align-crops` 模式从视频自动提取。

## 2. 训练

```bash
rs-face train \
  --dataset ./dataset \
  --out ./model.bin \
  --algorithm fisherfaces \
  --components 50
```

支持三种算法:
- **eigenfaces**: 主成分分析 (PCA), 最快
- **fisherfaces**: 线性判别 (LDA), 类间可分, 通常更准
- **lbph**: 局部二值模式直方图, 对光照最鲁棒

## 3. 识别

```bash
rs-face recognize \
  --model ./model.bin \
  --input ./new_face.jpg
```

输出:
```
[recognize] 类别=1 (teacher_wang)  置信度=0.95  匹配=是
```

阈值低于 0.5 (默认) 视为未知。

## 4. 端到端 demo

```bash
# 1. 从教学视频提取教师人脸
rs-face detect --input lecture.mp4 --save-crops --align-crops \
  --track --key-frames-only -o ./teacher_faces

# 2. 重命名为人名后放入 dataset
mv ./teacher_faces/crops/*.png ./dataset/teacher_wang/

# 3. 训练 + 识别
rs-face train --dataset ./dataset --out ./model.bin --algorithm fisherfaces
rs-face recognize --model ./model.bin --input ./new.jpg
```

## 5. 自动化聚类 → 命名的二阶段流程

```bash
# 阶段 1: detect + track, 生成 tracks.json
rs-face detect --input long_lecture.mp4 --track --key-frames-only -o ./out

# 阶段 2: 手动命名 face_id
# tracks.json 中每个 face_id 是一张人脸
# 给每张脸一个人名: face_labels.txt
# 0: 张老师
# 1: 李老师

# 阶段 3: 替换 "face_id" 为 "人名"
# (用户可写脚本批量 rename)
```

## 6. 性能数据

| 算法 | 训练速度 | 识别速度 | 精度 (synthetic) |
|---|---|---|---|
| Eigenfaces | <0.1s | <0.01s | 100% |
| Fisherfaces | <0.1s | <0.01s | 100% |
| LBPH | <0.5s | <0.01s | 100% |

`synthetic` 数据集: 3 人 × 5 张 = 15 张合成 92x112 灰度图。
真实场景的人脸精度取决于训练样本数量、姿态变化、光照条件。
