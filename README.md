
# Gemini335 Camera AI Project  
**Person Detection & Face Recognition Based on Orbbec Gemini335**

本项目基于 **Orbbec Gemini335 RGB-D 相机**，使用 **Orbbec Python SDK（pyorbbecsdk）** 获取 RGB 图像数据，结合 **YOLOv8** 实现人体检测，并使用 **InsightFace** 实现基于照片注册的人脸识别。

项目整体遵循 **“相机采集 + AI 推理 + 离线配置”** 的工程化设计思路，支持 GPU / CPU 自动切换，适合门禁、考勤、智能感知等应用场景。

---

## ✨ Features

- ✅ 支持 Orbbec Gemini335 RGB 相机
- ✅ YOLOv8 人体检测（CUDA / CPU）
- ✅ InsightFace 人脸检测 + 人脸识别（ArcFace）
- ✅ **基于照片的人脸注册（离线生成特征库）**
- ✅ JSON 人脸库（持久化，不依赖运行时注册）
- ✅ CUDA 不可用时自动回退 CPU
- ✅ 清晰模块化结构，便于扩展

---

## 📁 Project Structure

```text
Gemini335/
├── main.py                     # 程序入口
│
├── camera/
│   └── gemini335_camera.py     # Gemini335 相机封装（Orbbec SDK）
│
├── detectors/
│   └── yolo_detector.py        # YOLOv8 人体检测模块
│
├── recognizers/
│   └── person_recognizer.py    # 人脸检测 + 识别（InsightFace）
│
├── face_db/
│   ├── images/                 # ✅ 人脸注册图片目录
│   │   ├── Alice/
│   │   │   ├── 1.jpg
│   │   │   └── 2.jpg
│   │   └── Bob/
│   │       └── 1.jpg
│   │
│   └── face_db.json            # ✅ 自动生成的人脸特征库
│
├── tools/
│   └── register_faces_from_images.py  # 照片注册脚本
│
├── requirements.txt
└── README.md

---

## 🧩 Core Modules

### 1️⃣ Camera Module（Gemini335）

- 基于 **Orbbec Python SDK**
- 负责 RGB 图像采集
- 与 AI 模块完全解耦，仅输出图像数据流

设计思路与 Orbbec 官方 Python SDK Quick Start 示例一致 [[4]][doc_4][[5]][doc_5]。

---

### 2️⃣ Person Detection（YOLOv8）

- 使用 Ultralytics YOLOv8
- 基于 PyTorch，优先使用 CUDA
- 与人脸识别模块解耦

示例日志：

```text
[INFO] Person detection ENABLED
[YOLO] Loaded YOLOv8 model on cuda
[INFO] Gemini335 started in RGB mode
[INFO] Application running... (Ctrl+C to exit)
```

---

### 3️⃣ Face Recognition（InsightFace）

- 使用 InsightFace `buffalo_l` 模型集
- 包含：
  - 人脸检测
  - 关键点对齐
  - ArcFace 特征提取（512 维 embedding）
- 基于 **ONNX Runtime**
- 支持 CUDA / CPU 自动回退

---

## 👤 Face Registration（基于照片）

本项目**不使用运行时录入**，而是采用 **照片注册（工业级标准做法）**。

### ✅ 注册规则

```text
face_db/images/<PersonName>/*.jpg
```

示例：

```text
face_db/images/Alice/1.jpg
face_db/images/Alice/2.jpg
```

### ✅ 注册步骤

1. 将照片按上述规则放入 `face_db/images`
2. 运行注册脚本（只需一次）：

```bash
python tools/register_faces_from_images.py
```

3. 自动生成：

```text
face_db/face_db.json
```

`face_db.json` 中保存的是 **人脸特征向量（embedding）**，而非图片本身。

---

## 🚀 Deployment Guide

### ✅ 1. Clone Orbbec Python SDK

```bash
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
git checkout v2-main
```

参考 Orbbec 官方 Python SDK 下载与分支说明 [[6]][doc_6]。

---

### ✅ 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

主要依赖：

- pyorbbecsdk
- ultralytics
- insightface
- onnxruntime / onnxruntime-gpu
- opencv-python
- numpy

---

### ✅ 3. Windows / Linux 环境说明

- Windows：直接使用 Python SDK 编译或已编译版本  
- Linux：首次使用需注册 udev 规则

```bash
cd pyorbbecsdk/scripts
sudo chmod +x install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload-rules
```

该流程与 Orbbec 官方文档一致 [[2]][doc_2][[3]][doc_3]。

---

### ✅ 4. CUDA Support（可选）

- YOLOv8：基于 PyTorch CUDA
- InsightFace：基于 ONNX Runtime CUDA

若 CUDA 环境不完整，InsightFace 会自动回退 CPU，不影响程序运行。

---

## ▶️ Run Application

```bash
python main.py --launch rgb --detect 1
```

运行后：

- Gemini335 启动 RGB 流
- YOLOv8 检测人体
- InsightFace 对人脸进行识别
- 已注册人脸显示姓名，未匹配显示 `Unknown`

---

## 📌 Notes

- 建议每人注册 **3~5 张不同姿态照片**
- 人脸过小或遮挡会降低识别准确率
- JSON 人脸库支持版本管理与离线部署
- 推荐使用 Conda 管理 CUDA / ONNX Runtime 环境

---

## 📚 References

- Orbbec Python SDK Quick Start [[4]][doc_4][[5]][doc_5]
- Orbbec Python SDK Windows / Linux Configuration [[2]][doc_2][[3]][doc_3]
- Orbbec Python SDK Installation & Build [[6]][doc_6]
- Orbbec Official Examples [[1]][doc_1]

---

## ✅ Project Status

- ✅ Gemini335 相机稳定运行
- ✅ 人体检测支持 GPU
- ✅ 照片注册人脸识别已完成
- ✅ 具备工程化部署基础

---

## 📈 Future Work

- 深度信息融合（Gemini335 Depth）
- 人脸跟踪与 ID 稳定
- SQLite / Server 人脸库
- 多相机支持

---

**Maintained by:**  
SoulNail