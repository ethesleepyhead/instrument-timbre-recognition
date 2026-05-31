# CLAUDE.md — 乐器音色识别项目上下文

## 项目路径

`C:\Users\30328\Desktop\audio-app`

## 项目定位

这是一个面向**学习与展示**的机器学习 Demo，不是工业级高精度系统。核心价值在于展示 ML 完整流程（数据→特征→模型→Web 部署→可视化），而非追求最高准确率。

## 文件结构与职责

```
audio-app/
  index.html          # 前端页面（Hero + 上传 + 播放 + 整段结果 + 乐器时间轴）
  script.js           # 前端逻辑（WaveSurfer 播放器、并行请求、时间轴渲染、配置加载）
  style.css           # 全部样式（含时间轴卡片）
  features.py         # 共享模块：特征提取函数 + 常量（display names / colors / window params）
  timbre_api.py       # Flask 后端 + 启动入口（一键演示）
  train_model.py      # 独立训练脚本（不绑定启动流程）
  timbre_model.pkl    # 已训练模型（~31MB）
  run.bat             # Windows 一键启动
  dataset/            # 训练数据（按类别子目录存放 .wav 文件）
  features_cache/     # 特征缓存（训练时自动生成）
  README.md           # 完整项目文档
```

## 关键设计决策

### 1. 训练与演示分离
- `timbre_api.py` 是演示入口，启动即用，不触发训练
- `train_model.py` 是独立工具，仅在需要更新模型时手动运行
- 两者通过 `timbre_model.pkl` 文件耦合，共享 `features.py` 中的提取函数和常量

### 2. 特征提取模块（`features.py`）
所有特征提取函数集中在 `features.py`，训练和 API 共用同一份代码，不再各自维护。注册了三个提取器，每个都有文件版和内存数组版（`_from_audio`）：

| 维度 | 文件版函数 | 内容 |
|------|-----------|------|
| 40 | `extract_features_40()` | 仅 MFCC 均值+标准差 |
| 76 | `extract_features_76()` | 完整特征，单次 STFT 复用，无 hpss/tempo（当前默认） |
| 77 | `extract_features_77()` | 原始版本，含 hpss + beat_track |

文件版均委托到对应的 `_extract_features_*_from_audio(y, sr)`，训练用文件版，时间轴滑窗用内存版避免落盘。API 根据模型文件中的实际 `n_features_in_` 值自动选择提取函数，不依赖 `feature_version` 字段。`_get_model_input_dim()` 同时处理 Pipeline 和原始 RandomForestClassifier。

### 3. 统一常量管理
`features.py` 是常量的唯一来源，前后端通过 API 同步：

- `LABEL_DISPLAY_NAMES` — 11 类 instrument code → 展示名映射
- `INSTRUMENT_COLORS` — 11 色调色板
- `WINDOW_SEC` / `HOP_SEC` / `TOP_K` / `MIN_CONFIDENCE` — 时间轴窗口参数
- `SAMPLE_RATE = 22050`

前端启动时调用 `GET /health` 获取 `label_display_names`、`instrument_colors`、`window_params`，不再硬编码。`index.html` 中时间轴描述文字也由 JS 动态填充。

### 4. 当前模型状态
- 6 类（cla/flu/gac/pia/tru/vio），76 维特征（MFCC+chroma+频谱特征），测试准确率 ~70%
- 训练数据来自 IRMAS 数据集，每类 450-720 条 3s 片段
- cel 曾作为第 6 类但 F1 过低（0.56），替换为 cla（F1=0.63）

### 5. 模型元信息
模型文件（`timbre_model.pkl`）artifact 包含：
- `model` — Pipeline（StandardScaler + RandomForestClassifier）
- `labels` — 训练时的类别列表
- `feature_version` — 特征版本号（当前为 3）
- `feature_dim` — 实际特征维度
- `label_display_names` — 本次训练所用类别的展示名
- `metrics` — test_accuracy、cv_accuracy_mean/std、sample_counts、classification_report 等

`load_model_artifact()` 返回 `(model, labels, metrics, n_features, meta)`，后向兼容旧模型（无 meta 字段时返回空 dict）。

### 6. 双接口设计
- `POST /analyze` — 整段音频 → 全部类别概率分布
- `POST /analyze_timeline` — 滑动窗口分段预测，1s 窗口/0.5s 步长/top-2/阈值 0.10/相邻合并
- `GET /health` — 模型状态 + 配置常量（labels / display_names / colors / window_params）

### 7. 时间轴实现
- 后端：`_extract_segment_features()` 直接对内存音频数组提取特征（调用 `get_extractor_from_audio()`），不再写临时 WAV
- 合并：`_merge_segments()` 两遍扫描：先合并相邻同标签窗口（加权平均置信度），再桥接间隔 ≤1 步长的同标签段
- 前端：并行请求两个接口，`showTimeline()` 渲染每乐器一行的时间条
- 颜色：由服务端 `/health` 下发，按 label 映射（fallback 为本地 11 色调色板）
- tooltip：动态创建/销毁 DOM 元素

### 8. 训练脚本
- `--labels` 参数可指定训练哪些类（逗号分隔），不传则使用 dataset/ 下全部目录
- 特征缓存按数据集 hash 自动管理，换类组合会自动重建缓存
- 进度条使用 ASCII 字符（兼容 Windows GBK 编码）
- 训练完成后保存 `feature_dim` 和 `label_display_names` 到 artifact

### 9. 用户偏好
- 优先展示效果和易用性，而非模型精度
- 不要重构为深度学习项目
- 保留项目结构可理解
- 直接改代码，不先写长文档再改
- 修改完代码后给出简洁总结

## 后端 API 细节

### `load_model_artifact()`
返回 `(model, labels, metrics, n_features, meta)`。meta 含 `feature_version` 和 `feature_dim`（旧模型为空 dict）。

### `GET /health` 返回格式
```json
{
  "status": "ok",
  "labels": ["cla", "flu", "gac", "pia", "tru", "vio"],
  "n_features": 76,
  "test_accuracy": 0.70,
  "label_display_names": {"cla": "Clarinet", "flu": "Flute", ...},
  "instrument_colors": {"cla": "#4f46e5", "flu": "#0d9488", ...},
  "window_params": {"window_sec": 1.0, "hop_sec": 0.5, "top_k": 2, "min_confidence": 0.10}
}
```

### `/analyze` 返回格式
```json
{
  "results": [{"label": "pia", "confidence": 0.85}, ...],
  "top_prediction": {"label": "pia", "confidence": 0.85},
  "labels": [...],
  "label_display_names": {"pia": "Piano", ...},
  "model_info": {"n_features": 76, "test_accuracy": 0.70}
}
```

### `/analyze_timeline` 返回格式
```json
{
  "segments": [{"start": 0.0, "end": 1.5, "predictions": [...]}],
  "labels": [...],
  "label_display_names": {...},
  "duration": 3.0, "window_size": 1.0, "hop_size": 0.5,
  "model_info": {"n_features": 76, "test_accuracy": 0.70}
}
```

## 启动方式
```bash
python timbre_api.py          # 或双击 run.bat
python train_model.py --quick  # 快速训练
python train_model.py          # 完整训练
```

启动后自动打开浏览器到 `http://127.0.0.1:5000`。

## 常见问题
- 修改代码后报错 → 删 `__pycache__/`，重启服务器
- "X has 77 features, expecting 40" → 服务器在跑旧代码，需要重启
- 时间轴不可用 → 同上，旧代码没有 `/analyze_timeline` 路由
- 前端显示 code 而非中文名 → `/health` 加载失败，检查后端是否启动
