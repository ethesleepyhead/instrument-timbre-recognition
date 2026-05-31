const API_BASE_URL = window.location.origin;

// Config fetched from server — populated on load, kept as fallback until then
let appConfig = {
  labelDisplayNames: {},
  instrumentColors: {},
  windowParams: { window_sec: 1.0, hop_sec: 0.5, top_k: 2, min_confidence: 0.10 },
  labels: [],
  n_features: 0,
  test_accuracy: null,
};

(async function loadConfig() {
  try {
    const resp = await fetch(`${API_BASE_URL}/health`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.status !== "ok") return;

    appConfig.labelDisplayNames = data.label_display_names || {};
    appConfig.instrumentColors = data.instrument_colors || {};
    appConfig.windowParams = data.window_params || appConfig.windowParams;
    appConfig.labels = data.labels || [];
    appConfig.n_features = data.n_features || 0;
    appConfig.test_accuracy = data.test_accuracy;

    // Update timeline description with actual window params
    const descEl = document.getElementById("timelineDesc");
    if (descEl) {
      const wp = appConfig.windowParams;
      descEl.textContent =
        `将音频按 ${wp.window_sec} 秒窗口 / ${wp.hop_sec} 秒步长 逐段分析，` +
        `展示不同时间段各乐器的预测概率。颜色越深表示置信度越高。`;
    }
  } catch (_) { /* use fallbacks */ }
})();

// ---- DOM refs ----
const fileInput    = document.getElementById("fileInput");
const dropZone     = document.getElementById("dropZone");
const controls     = document.getElementById("controls");
const playBtn      = document.getElementById("playBtn");
const analyzeBtn   = document.getElementById("analyzeBtn");
const currentTimeEl = document.getElementById("currentTime");
const durationEl    = document.getElementById("duration");
const fileName      = document.getElementById("fileName");
const statusText    = document.getElementById("statusText");
const resultDiv     = document.getElementById("result");
const loading       = document.getElementById("loading");
const errorBox      = document.getElementById("errorBox");
const summaryCard   = document.getElementById("summaryCard");
const topLabel      = document.getElementById("topLabel");
const topConfidence = document.getElementById("topConfidence");
const modelHint     = document.getElementById("modelHint");

// Timeline refs
const timelineCard    = document.getElementById("timelineCard");
const timelineBody    = document.getElementById("timelineBody");
const timelineTicks   = document.getElementById("timelineTicks");
const timelineHint    = document.getElementById("timelineHint");
const timelineError   = document.getElementById("timelineError");
const timelineLoading = document.getElementById("timelineLoading");

let currentFile = null;

// ---- WaveSurfer ----
const wavesurfer = WaveSurfer.create({
  container: "#waveform",
  waveColor: "#a5b4fc",
  progressColor: "#4f46e5",
  cursorColor: "#1e293b",
  barWidth: 2,
  barGap: 1,
  height: 88,
  responsive: true
});

// ---- Upload: drag & drop + file input ----
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => loadAudio(e.target.files[0]));

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  loadAudio(e.dataTransfer.files[0]);
});

// ---- Playback ----
playBtn.addEventListener("click", () => {
  if (!currentFile) return;
  wavesurfer.playPause();
  playBtn.innerHTML = wavesurfer.isPlaying() ? "&#x23F8; 暂停" : "&#x25B6; 播放";
});

// ---- Analyze (overall + timeline in parallel) ----
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) {
    showError("请先上传一个音频文件。");
    return;
  }

  setBusy(true);
  clearError();
  clearTimeline();

  const formData = new FormData();
  formData.append("file", currentFile);

  // Fire both requests in parallel
  const [overallResp, timelineResp] = await Promise.allSettled([
    fetch(`${API_BASE_URL}/analyze`, { method: "POST", body: formData }),
    fetch(`${API_BASE_URL}/analyze_timeline`, { method: "POST", body: cloneFormData(formData) })
  ]);

  // Process overall result
  if (overallResp.status === "fulfilled" && overallResp.value.ok) {
    const data = await overallResp.value.json();
    showResult(data);
    statusText.textContent = "识别完成。你可以上传其他音频继续比较。";
  } else {
    const errData = overallResp.status === "fulfilled"
      ? await overallResp.value.json().catch(() => ({}))
      : {};
    showError(errData.error || "整段识别失败，请检查后端服务。");
    statusText.textContent = "暂时无法完成识别，请确认模型已训练且后端正常运行。";
  }

  // Process timeline result
  if (timelineResp.status === "fulfilled" && timelineResp.value.ok) {
    const tlData = await timelineResp.value.json();
    showTimeline(tlData);
  } else {
    const errData = timelineResp.status === "fulfilled"
      ? await timelineResp.value.json().catch(() => ({}))
      : {};
    timelineError.hidden = false;
    timelineError.textContent = errData.error
      ? `时间轴分析失败：${errData.error}`
      : "时间轴分析暂时不可用。";
  }

  timelineLoading.style.display = "none";
  setBusy(false);
});

// ---- WaveSurfer events ----
wavesurfer.on("audioprocess", () => {
  currentTimeEl.textContent = formatTime(wavesurfer.getCurrentTime());
});
wavesurfer.on("ready", () => {
  durationEl.textContent = formatTime(wavesurfer.getDuration());
  currentTimeEl.textContent = "0:00";
  playBtn.innerHTML = "&#x25B6; 播放";
});
wavesurfer.on("finish", () => {
  playBtn.innerHTML = "&#x25B6; 播放";
});

// ---- Timeline rendering ----

function showTimeline(data) {
  const { segments, labels, duration } = data;
  timelineCard.style.display = "block";

  if (!segments || segments.length === 0) {
    timelineBody.innerHTML =
      '<div class="timeline-empty">未检测到明显的乐器片段（所有片段置信度均低于阈值）。</div>';
    timelineTicks.innerHTML = "";
    timelineHint.style.display = "none";
    return;
  }

  // Build label → color map from server config (fallback: use own palette)
  const colorMap = {};
  const fallbackColors = ["#4f46e5", "#0d9488", "#d97706", "#dc2626", "#7c3aed", "#059669", "#db2777", "#2563eb", "#ea580c", "#65a30d", "#0891b2"];
  labels.forEach((lbl, i) => {
    colorMap[lbl] = appConfig.instrumentColors[lbl] || fallbackColors[i % fallbackColors.length];
  });

  // Time axis ticks (~6 evenly spaced)
  const tickCount = Math.min(6, Math.ceil(duration));
  const tickStep = duration / (tickCount - 1 || 1);
  let ticksHTML = "";
  for (let i = 0; i < tickCount; i++) {
    ticksHTML += `<span>${formatTime(tickStep * i)}</span>`;
  }
  timelineTicks.innerHTML = ticksHTML;

  // One row per label
  let rowsHTML = "";
  labels.forEach(lbl => {
    const color = colorMap[lbl];
    // Collect segments where this label appears (any position in predictions)
    const segs = [];
    segments.forEach(seg => {
      seg.predictions.forEach(p => {
        if (p.label === lbl) {
          segs.push({ start: seg.start, end: seg.end, confidence: p.confidence });
        }
      });
    });

    if (segs.length === 0) {
      // Empty row — still show the label
      rowsHTML += `
        <div class="tl-row">
          <span class="tl-label">${toDisplayLabel(lbl)}</span>
          <div class="tl-track"></div>
        </div>`;
      return;
    }

    let barsHTML = "";
    segs.forEach(s => {
      const leftPct = (s.start / duration) * 100;
      const widthPct = ((s.end - s.start) / duration) * 100;
      const opacity = Math.max(0.15, s.confidence);
      const title = `${s.start.toFixed(1)}s-${s.end.toFixed(1)}s ${toDisplayLabel(lbl)} ${(s.confidence*100).toFixed(0)}%`;
      barsHTML += `<div class="tl-segment"
        style="left:${leftPct}%;width:${widthPct}%;background:${color};opacity:${opacity}"
        data-tooltip="${title}"></div>`;
    });

    rowsHTML += `
      <div class="tl-row">
        <span class="tl-label">${toDisplayLabel(lbl)}</span>
        <div class="tl-track">${barsHTML}</div>
      </div>`;
  });

  timelineBody.innerHTML = rowsHTML;
  timelineHint.style.display = "block";

  // Tooltip behavior
  attachTooltips();
}

function attachTooltips() {
  let tooltipEl = null;

  const onEnter = (e) => {
    const text = e.target.dataset.tooltip;
    if (!text) return;
    tooltipEl = document.createElement("div");
    tooltipEl.className = "tl-tooltip";
    tooltipEl.textContent = text;
    document.body.appendChild(tooltipEl);
    moveTooltip(e);
  };

  const onMove = (e) => {
    if (!tooltipEl) return;
    moveTooltip(e);
  };

  const onLeave = () => {
    if (tooltipEl) {
      tooltipEl.remove();
      tooltipEl = null;
    }
  };

  const moveTooltip = (e) => {
    if (!tooltipEl) return;
    const x = e.clientX + 12;
    const y = e.clientY - 32;
    tooltipEl.style.left = x + "px";
    tooltipEl.style.top = y + "px";
  };

  document.querySelectorAll(".tl-segment").forEach(el => {
    el.addEventListener("mouseenter", onEnter);
    el.addEventListener("mousemove", onMove);
    el.addEventListener("mouseleave", onLeave);
  });
}

// ---- Helpers ----
function loadAudio(file) {
  if (!file || !file.type.startsWith("audio/")) {
    showError("请选择有效的音频文件。");
    return;
  }

  currentFile = file;
  clearError();
  clearTimeline();
  resultDiv.innerHTML = "";
  summaryCard.hidden = true;
  fileName.textContent = file.name;
  statusText.textContent = "文件已加载，可以试听或开始识别。";
  controls.style.display = "block";

  wavesurfer.load(URL.createObjectURL(file));
}

function showResult(data) {
  const results = Array.isArray(data.results) ? [...data.results] : [];
  results.sort((a, b) => b.confidence - a.confidence);
  if (results.length === 0) {
    showError("模型没有返回任何类别结果。");
    return;
  }

  const best = data.top_prediction || results[0];
  summaryCard.hidden = false;
  topLabel.textContent = toDisplayLabel(best.label);
  topConfidence.textContent = `${(best.confidence * 100).toFixed(1)}%`;

  const acc = data.model_info && data.model_info.test_accuracy;
  modelHint.textContent = acc != null
    ? `当前模型测试集准确率约 ${(acc * 100).toFixed(1)}%（仅供参考，实际效果受数据集影响）。`
    : "";

  resultDiv.innerHTML = "";
  results.forEach((item, i) => {
    const pct = (item.confidence * 100).toFixed(1);
    const el = document.createElement("article");
    el.className = "result-item" + (i === 0 ? " top-result" : "");
    el.innerHTML = `
      <div class="result-label">
        <span>${toDisplayLabel(item.label)}</span>
        <span>${pct}%</span>
      </div>
      <div class="result-bar"><div class="result-fill" style="width:${pct}%"></div></div>
    `;
    resultDiv.appendChild(el);
  });
}

function cloneFormData(formData) {
  const fd = new FormData();
  for (const [key, value] of formData.entries()) {
    fd.append(key, value);
  }
  return fd;
}

function clearTimeline() {
  timelineCard.style.display = "none";
  timelineBody.innerHTML = "";
  timelineTicks.innerHTML = "";
  timelineHint.style.display = "block";
  timelineError.hidden = true;
  timelineLoading.style.display = "block";
}

function toDisplayLabel(code) {
  return appConfig.labelDisplayNames[code] || code;
}

function setBusy(on) {
  loading.style.display = on ? "block" : "none";
  timelineLoading.style.display = on ? "block" : "none";
  analyzeBtn.disabled = on;
  analyzeBtn.textContent = on ? "分析中..." : "开始识别";
}

function showError(msg) {
  errorBox.hidden = false;
  errorBox.textContent = msg;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
