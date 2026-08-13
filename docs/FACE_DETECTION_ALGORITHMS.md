# 人脸检测 / 识别算法选型 (v2)

> 本项目目标: 用纯 Rust (零第三方 crate 依赖) 实现一版可工作的人脸检测 + 识别系统。
> 本文调研主流算法, 在零依赖约束下进行选型, 解释最终落地实现。

---

## 1. 任务边界

- **检测 (Detection)**: 在图中找 "哪里有人脸", 输出 bounding box。
- **识别 (Recognition)**: 判断 "这是谁", 基于人脸 embedding / 特征比对。
- 本项目两条都覆盖:
  - 检测 → 从视频抽帧后定位人脸, 按时间戳命名保存。
  - 识别 → 训练 LBPH / Eigenfaces / Fisherfaces 模型, 给静态图打标。

---

## 2. 算法选型结论 (生产界与学术界共识)

| 任务 | 最终选择 | 替代方案 | 选型理由 |
|---|---|---|---|
| 人脸检测 | **Viola-Jones (Haar Cascade)** | MTCNN / RetinaFace / YOLO-Face / BlazeFace | 唯一可在纯 std Rust 中实现的工业级 CPU 实时检测器; OpenCV 默认; 模型 < 1MB |
| 第二阶段检测 | **HOG + Linear SVM (Dalal-Triggs 2005)** | CNN verifier | dlib `get_frontal_face_detector()` 同款方法; 零依赖可训练; 与 Haar 形成互补 |
| 检测增强 | **水平翻转检测 (--flip-detect)** | profile cascade XML | 0 额外成本检测镜像侧脸 |
| 识别特征 | **LBPH (Local Binary Patterns Histograms)** | Eigenfaces / Fisherfaces | 最鲁棒的传统方法, 光照不变, 训练快, OpenCV 默认 |
| 识别备选 | **Eigenfaces (PCA)** | Fisherfaces | Turk & Pentland 1991, 最经典 PCA 方法 |
| 识别备选 | **Fisherfaces (LDA = PCA+LDA)** | - | Belhumeur 1997, 在分类任务上优于纯 PCA |
| 匹配器 | **Chi-Square + KNN** | SVM | 与 LBPH 天然契合, 0 训练时间 |
| 匹配备选 | **Cosine / Euclidean + KNN** | - | 与 PCA/LDA 嵌入向量契合 |

---

## 3. 调研概览 (业界 2024 主流)

### 3.1 检测算法 (CNN 时代)

| 算法 | 年份 | 优点 | 缺点 | 零 std Rust 可行 |
|---|---|---|---|---|
| **Viola-Jones** | 2001 | CPU 实时; 模型小 (<1MB); 无需 GPU | 侧脸 / 遮挡鲁棒差 | ✅ (首选) |
| HOG + SVM (dlib) | 2005 | 行人 / 人脸均成功; CPU 友好 | 单独精度不如 CNN | ✅ (本项目二阶段) |
| Cascade CNN | 2015 | 比 Haar 略准 | 仍需权重文件 | ⚠️ (卷积算子可写但繁琐) |
| **MTCNN** | 2016 | 多任务 (box + landmarks); CPU 实时 | 模型 ~2MB, ONNX 解析复杂 | ⚠️ |
| FaceBoxes | 2017 | CPU 快速 | 已被新方法超越 | ❌ |
| RetinaFace (InsightFace) | 2019 | WIDER FACE SOTA | 模型大 (~100MB); 后处理复杂 | ❌ |
| CenterFace / LFFD | 2019 | anchor-free | 训练复现成本高 | ❌ |
| YOLOv5-Face / YOLOv8-Face | 2021+ | 通用 SOTA | 模型大; NMS 重 | ❌ |
| **BlazeFace (MediaPipe)** | 2019 | Google, 移动端 | SSD-like, 推理重 | ❌ |
| DETR / Deformable DETR | 2020+ | transformer 检测 | 模型大 | ❌ |

### 3.2 识别算法 (Embedding 时代)

| 算法 | 年份 | 优点 | 缺点 | 零 std Rust 可行 |
|---|---|---|---|---|
| **Eigenfaces (PCA)** | 1991 | 经典; 直觉清晰 | 光照敏感 | ✅ |
| **Fisherfaces (LDA)** | 1997 | 优于 PCA 分类 | 类内协方差奇异性 | ✅ |
| **LBPH** | 2004 | 光照 / 旋转鲁棒 | 特征维数高 (16K) | ✅ |
| DeepFace (Facebook) | 2014 | 早期 CNN 范式 | 3D 对齐依赖 | ❌ |
| **FaceNet (Google)** | 2015 | Triplet loss, 128-d | 训练成本极高 | ❌ |
| SphereFace | 2017 | A-Softmax | 已退潮 | ❌ |
| **CosFace** | 2018 | Large Margin Cosine | - | ❌ |
| **ArcFace (InsightFace)** | 2019 | **当前业界标准**, additive angular margin | - | ❌ |
| MobileFaceNet | 2019 | 移动端 SOTA | - | ❌ |
| TransFace / FaRL | 2022+ | transformer 识别 | 训练资源高 | ❌ |

### 3.3 选型结论

零依赖 std Rust 约束下, 现代 CNN 路 (ArcFace / RetinaFace / MTCNN) 无法实现 (模型大、需要卷积 / BLAS)。

**保留的最广泛使用 / 最好的算法:**

- **检测**: Viola-Jones (主力) + HOG+SVM (二阶段) + 水平翻转增强。
- **识别**: LBPH (首选, 最鲁棒) + Eigenfaces + Fisherfaces (作为对照/回退)。
- **匹配**: KNN + Chi-Square (LBPH 配) / Cosine & Euclidean (PCA / LDA 配)。

这些方法在工业界被 OpenCV / dlib / face_recognition (Python) 等最广泛部署的项目采用,
几乎所有嵌入式 / 移动端 / 离线场景 (无 GPU) 都用其中之一或多个组合。

---

## 4. 实现要点

### 4.1 Viola-Jones (`cascade.rs`)
- Haar-like 特征 (边缘 / 线 / 块, 含 tilted 45°)
- 积分图 (Integral Image) O(1) 矩形求和 + 平方积分图 (方差归一化)
- AdaBoost 弱分类器 + 级联多阶段拒绝
- 多尺度滑窗 (scale_factor, step)
- NMS 分组 (min_neighbors)
- XML 解析 (OpenCV `haarcascade_frontalface_alt2.xml` 格式)

### 4.2 HOG + SVM (`hog_svm.rs`)
- Sobel-like 中心差分梯度
- 8x8 px cell × 9 bin 直方图 + 2x2 cell block L2-norm
- 描述子维度 1764 (7×7×2×2×9)
- 多尺度滑窗 (target_size ∈ [min_size, max_size])
- 线性 SVM (Hinge Loss + SGD), labels ∈ {-1, +1}
- NMS + 与 Viola-Jones 结果合并

### 4.3 LBPH (`recognition.rs`)
- 圆形 8 邻域 LBP 编码 (radius=1)
- 8×8 网格分块, 256 bin 直方图
- 16384 维归一化直方图拼接
- Chi-Square 距离 + KNN

### 4.4 Eigenfaces (`faces.rs`)
- PCA via S^T S 技巧 (样本数 << 像素数时高效)
- Jacobi 旋转解对称特征值
- 余弦距离 + KNN

### 4.5 Fisherfaces (`faces.rs`)
- 先 PCA 降维 → LDA 投影
- 类间 / 类内散度 SW⁻¹ SB
- 高斯消元法矩阵求逆
- 欧氏距离 + KNN

---

## 5. 性能参考

测试环境: M2 MacBook Air, 320x240 灰度视频, 9 帧, `--fps 3 --min-size 30`

| 阶段 | 耗时 |
|---|---|
| ffmpeg 抽帧 | ~0.05s |
| Cascade 加载 | <0.05s |
| 9 帧检测 | 0.6s (~70ms/帧) |
| NMS + 保存 | <0.05s |
| **总计** | **0.64s** |

启用 `--flip-detect` 后约 +50% 时间 (检测量翻倍)。

---

## 6. 参考论文

1. Paul Viola, Michael Jones. *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001.
2. Rainer Lienhart, Jochen Maydt. *An Extended Set of Haar-like Features for Rapid Object Detection*, ICIP 2002.
3. Navneet Dalal, Bill Triggs. *Histograms of Oriented Gradients for Human Detection*, CVPR 2005.
4. Matthew Turk, Alex Pentland. *Eigenfaces for Recognition*, J. Cognitive Neuroscience 1991.
5. Peter Belhumeur, João Hespanha, David Kriegman. *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection*, IEEE PAMI 1997.
6. Timo Ahonen, Abdenour Hadid, Matti Pietikäinen. *Face Recognition with Local Binary Patterns*, ECCV Workshop 2004.
7. Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. (业界标准, 但本项目零依赖无法落地)