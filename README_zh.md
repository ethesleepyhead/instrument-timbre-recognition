# 乐器音色识别 Web Demo

一个轻量级音频机器学习项目，用于乐器分类与时间轴可视化。

上传音频后，系统会使用训练好的 `scikit-learn` 模型进行推理，并展示：

- 整段音频的类别概率分布
- Top 预测乐器
- 基于滑动时间窗的乐器活动时间轴

## 项目功能

本项目将以下几个部分组合在一起：

- 使用 `librosa` 提取音频特征
- 使用 `scikit-learn` 进行传统机器学习分类
- 使用 `Flask` 提供本地推理接口
- 使用浏览器页面展示波形、概率结果和时间轴

它更适合作为一个紧凑、可解释的机器学习 Demo，而不是一个工业级音乐分析系统。

## 功能特性

- 支持上传音频文件：`WAV`、`MP3`、`FLAC`、`OGG`、`M4A`
- 支持波形播放
- 对整段音频输出乐器分类概率
- 显示 Top 预测和各类别置信度条
- 对音频按滑动时间窗进行分段分析
- 在波形下方绘制逐乐器时间轴

## 项目结构

```text
audio-app/
  index.html          # 前端页面结构
  script.js           # 前端交互逻辑与 API 调用
  style.css           # 前端样式
  features.py         # 共享特征提取逻辑与常量配置
  timbre_api.py       # Flask 应用、静态文件服务、推理接口
  train_model.py      # 模型训练脚本
  timbre_model.pkl    # 训练好的模型文件
  run.bat             # Windows 启动脚本
  dataset/            # 按类别组织的训练音频
  features_cache/     # 训练阶段缓存的特征文件
  README.md
```

## 代码架构

### 1. 前端

文件：

- [index.html](C:/Users/30328/Desktop/audio-app/index.html)
- [script.js](C:/Users/30328/Desktop/audio-app/script.js)
- [style.css](C:/Users/30328/Desktop/audio-app/style.css)

职责：

- 文件上传与拖拽
- WaveSurfer 波形播放
- 调用后端接口
- 渲染整段音频的分类结果
- 渲染乐器时间轴

主流程：

1. 用户上传音频文件
2. 前端将文件发送到 `/analyze` 和 `/analyze_timeline`
3. 前端渲染：
   - 整段预测摘要
   - 概率条
   - 时间轴行与片段块

### 2. 后端 API

文件：

- [timbre_api.py](C:/Users/30328/Desktop/audio-app/timbre_api.py)

职责：

- 在 `http://127.0.0.1:5000` 提供前端页面
- 从 `timbre_model.pkl` 加载训练好的模型
- 校验上传文件
- 执行整段音频推理
- 执行滑动时间窗分段推理
- 返回前端可直接使用的 JSON 结构

路由：

- `GET /`：前端页面
- `GET /health`：模型与配置状态
- `POST /analyze`：整段音频预测
- `POST /analyze_timeline`：时间轴预测

### 3. 特征提取模块

文件：

- [features.py](C:/Users/30328/Desktop/audio-app/features.py)

职责：

- 提供可复用的特征提取函数
- 保持训练阶段与推理阶段的特征一致
- 维护共享常量，例如：
  - 乐器显示名称
  - 颜色配置
  - 时间轴窗口参数

当前代码支持的特征版本：

- `40` 维：MFCC 均值 + 标准差
- `76` 维：MFCC + Chroma + 频谱特征
- `77` 维：包含 tempo / HPSS 的旧版本特征

后端会根据模型实际输入维度自动选择对应的提取器。

### 4. 训练流程

文件：

- [train_model.py](C:/Users/30328/Desktop/audio-app/train_model.py)

职责：

- 从 `dataset/` 自动发现类别
- 提取并缓存特征
- 划分训练集与测试集
- 执行交叉验证
- 训练 `StandardScaler + RandomForestClassifier`
- 将最终模型保存为 `timbre_model.pkl`

训练模式：

- 快速模式：`python train_model.py --quick`
- 完整模式：`python train_model.py`
- 强制重新提取特征：`python train_model.py --force-extract`

## 推理流程

### 整段音频推理

接口：`POST /analyze`

流程：

1. 临时保存上传文件
2. 提取整段音频特征
3. 调用模型预测
4. 返回按置信度排序的类别概率

示例返回：

```json
{
  "results": [
    {"label": "pia", "confidence": 0.82},
    {"label": "vio", "confidence": 0.11}
  ],
  "top_prediction": {"label": "pia", "confidence": 0.82},
  "labels": ["cla", "flu", "gac", "pia", "tru", "vio"],
  "model_info": {
    "n_features": 76,
    "test_accuracy": 0.70
  }
}
```

### 时间轴推理

接口：`POST /analyze_timeline`

流程：

1. 读取整段音频
2. 按滑动时间窗切分
3. 对每个时间窗提取特征
4. 对每个时间窗预测 top-k 候选类别
5. 合并相邻的兼容片段
6. 返回前端可直接渲染的时间轴 JSON

默认时间轴参数定义在 `features.py` 中。

示例返回：

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "predictions": [
        {"label": "pia", "confidence": 0.81},
        {"label": "vio", "confidence": 0.22}
      ]
    }
  ],
  "labels": ["cla", "flu", "gac", "pia", "tru", "vio"],
  "duration": 3.0,
  "window_size": 1.0,
  "hop_size": 0.5
}
```

## 本地运行

### 方式一

```bash
python timbre_api.py
```

### 方式二

双击：

```text
run.bat
```

然后打开：

```text
http://127.0.0.1:5000
```

## 依赖安装

```bash
pip install flask flask-cors joblib librosa numpy scikit-learn
```

## 截图位置

建议将截图放在：

```text
github/screenshots/
```

预留路径：

- `github/screenshots/home.png`
- `github/screenshots/result.png`
- `github/screenshots/timeline.png`
- `github/screenshots/training.png`

## 说明

- 当前系统更适合理解为“主导乐器识别 + 近似多候选时间轴展示”
- 时间轴结果基于滑动时间窗推理，不是真正的逐帧多标签源分析
- 最终效果会受到数据集覆盖范围、标签质量和录音条件影响

