# 乐器音色识别 · Instrument Timbre Recognition

[English](#english) | [中文](#中文)

A lightweight machine learning demo for musical instrument classification with timeline visualization.  
一个轻量级乐器分类机器学习 Demo，支持音频上传、实时推理和时间轴可视化。

---

## Screenshots · 截图

| | | |
|---|---|---|
| **Home Page · 主页** | **Recognition Result · 识别结果** | **Instrument Timeline · 乐器时间轴** |
| ![home](https://github.com/user-attachments/assets/08dd3a04-ac7e-49e9-8076-d38b3bed9c88) | ![result](https://github.com/user-attachments/assets/3feb5182-85c9-4ec8-a1a4-15e23e139c7d) | ![timeline](https://github.com/user-attachments/assets/9d9aeb18-339a-4a5d-bbcf-4f4a491b569b) |

---

<h2 id="english">English</h2>

### Overview

This project demonstrates a complete machine learning workflow:

- **Feature extraction** with `librosa` (MFCC, chroma, spectral features)
- **Classification** with `scikit-learn` Random Forest
- **Web API** via Flask
- **Interactive UI** with WaveSurfer waveform + instrument timeline

It is designed as a compact, explainable ML demo rather than a production-grade music analysis system.

### Features

- Upload audio: `WAV`, `MP3`, `FLAC`, `OGG`, `M4A`
- Waveform playback
- Full-track instrument probability distribution
- Top prediction with confidence bars
- Sliding-window timeline analysis — one row per instrument

### Project Structure

```text
audio-app/
  index.html          # Frontend page
  script.js           # Frontend logic & API calls
  style.css           # Styles
  features.py         # Shared feature extraction & constants
  timbre_api.py       # Flask app + inference endpoints
  train_model.py      # Model training script
  timbre_model.pkl    # Pre-trained model (~84MB, included)
  run.bat             # Windows one-click launcher
  dataset/            # Training data (download separately)
  features_cache/     # Feature cache (auto-generated)
  README.md
```

### Quick Start

```bash
# 1. Install dependencies
pip install flask flask-cors joblib librosa numpy scikit-learn

# 2. Start the server
python timbre_api.py
# Or double-click: run.bat

# 3. Open browser → http://127.0.0.1:5000
```

Upload an audio file and click "开始识别" (Start Recognition).

### API Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/` | GET | Frontend page |
| `/health` | GET | Model status & config |
| `/analyze` | POST | Full-track prediction |
| `/analyze_timeline` | POST | Sliding-window timeline |

### Training

The pre-trained model (`timbre_model.pkl`) uses 6 instrument classes (Clarinet, Flute, Acoustic Guitar, Piano, Trumpet, Violin) with 76-dim features, achieving ~70% test accuracy.

```bash
# Quick training (~1-3 min)
python train_model.py --quick

# Full training (~10-20 min)
python train_model.py
```

### Dataset

This project uses the [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset. Download `IRMAS-TrainingData.zip` and place it in `dataset/`:

```
dataset/
  cel/  cla/  flu/  gac/  gel/  org/
  pia/  sax/  tru/  vio/  voi/
```

---

<h2 id="中文">中文</h2>

### 项目概述

上传音频后，系统会使用训练好的 `scikit-learn` 模型进行推理，并展示：

- 整段音频的类别概率分布
- Top 预测乐器
- 基于滑动时间窗的乐器活动时间轴

本项目将 `librosa` 特征提取、`scikit-learn` 随机森林分类、`Flask` 后端和浏览器前端组合在一起，更适合作为紧凑、可解释的 ML Demo，而不是工业级音乐分析系统。

### 功能特性

- 支持上传：`WAV`、`MP3`、`FLAC`、`OGG`、`M4A`
- WaveSurfer 波形播放
- 整段音频乐器分类概率
- Top 预测和各类别置信度条
- 滑动时间窗分段分析 + 逐乐器时间轴

### 项目结构

```text
audio-app/
  index.html          # 前端页面结构
  script.js           # 前端交互逻辑与 API 调用
  style.css           # 前端样式
  features.py         # 共享特征提取逻辑与常量配置
  timbre_api.py       # Flask 应用、静态文件服务、推理接口
  train_model.py      # 模型训练脚本
  timbre_model.pkl    # 预训练模型（~84MB，仓库已包含）
  run.bat             # Windows 一键启动脚本
  dataset/            # 训练数据（需手动下载）
  features_cache/     # 特征缓存（训练时自动生成）
```

### 快速开始

```bash
# 1. 安装依赖
pip install flask flask-cors joblib librosa numpy scikit-learn

# 2. 启动服务
python timbre_api.py
# 或双击 run.bat

# 3. 浏览器访问 http://127.0.0.1:5000
```

上传音频文件，点击「开始识别」即可。

### API 接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/health` | GET | 模型与配置状态 |
| `/analyze` | POST | 整段音频预测 |
| `/analyze_timeline` | POST | 时间轴预测 |

### 代码架构

**前端** — `index.html` + `script.js` + `style.css`
- 文件上传与拖拽、WaveSurfer 波形播放、API 调用、结果渲染、时间轴渲染

**后端** — `timbre_api.py`
- 加载模型、校验文件、整段推理、滑动时间窗推理、返回 JSON

**特征提取** — `features.py`
- 提供 40/76/77 维三种提取器，后端根据模型维度自动匹配
- 维护共享常量：显示名称、颜色、窗口参数

**训练** — `train_model.py`
- 自动发现类别、缓存特征、交叉验证、训练 Random Forest
- 快速模式：`--quick`（减少树数量和 CV 折数）

### 数据集

使用 [IRMAS](https://www.upf.edu/web/mtg/irmas) 数据集，下载 `IRMAS-TrainingData.zip` 解压到 `dataset/`：

```
dataset/
  cel/   大提琴      cla/   单簧管
  flu/   长笛        gac/   原声吉他
  gel/   电吉他      org/   管风琴
  pia/   钢琴        sax/   萨克斯
  tru/   小号        vio/   小提琴
  voi/   人声
```

```bash
# 仓库已包含预训练模型，可直接使用
# 如需自行训练：
python train_model.py --quick
```

### 技术栈

| 层 | 技术 |
|----|------|
| 音频特征 | Python, librosa |
| 机器学习 | scikit-learn (Random Forest + StandardScaler) |
| 后端 | Flask |
| 前端 | HTML5, CSS3, JavaScript, WaveSurfer |

### 说明

- 当前系统更适合理解为"主导乐器识别 + 近似多候选时间轴展示"
- 时间轴基于滑动时间窗推理，不是真正的逐帧多标签源分离
- 同一窗口内多种乐器同时发声时，模型只能以概率分布近似表达
- 最终效果受数据集覆盖范围、标签质量和录音条件影响

---

<p align="center">
  <sub>Built with Python · librosa · scikit-learn · Flask · WaveSurfer</sub>
</p>
