/**
 * Coderstation — Kokoro Text-to-Voice Studio Frontend
 * All settings are organized into an interactive Modal Dialog (#settingsDialog) with 4 tabs:
 *  1. Voice & Feelings in Speech (dlgTabVoice)
 *  2. Punctuation Pause Calibration (dlgTabPauses)
 *  3. Voice Cloning Studio (dlgTabCloning)
 *  4. Output Format (WAV / MP3) & Saved History (dlgTabOutput)
 */

const PRESET_SCRIPTS = {
  dramatic_story:
    "[dramatic] Listen closely, my friend. In the quiet of the night... something extraordinary is about to happen — can you feel it? [whisper] Every shadow holds a secret, every breath tells a story... [excited] And now, the voice comes alive with real emotion!",
  multi_emotion:
    "[happy] Today is a wonderful day to create something new, bright, and inspiring. [sad] Sometimes the road feels long... and memories linger in the silence — [angry] but we refuse to give up! [calm] Take a deep breath, relax, and listen to the gentle cadence.",
  pause_showcase:
    "First, a comma creates a short pause, keeping the clause connected. Second, a period creates a moderate pause. Third... an ellipsis creates a longer, dramatic pause... Fourth — an em dash creates a natural break in thought — before concluding the sentence.",
};

const CLONE_READING_CONTEXTS = {
  rainbow:
    '"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The division of white light into many beautiful colors — with deep resonance, clear vowels... and natural rhythm."',
  expressive:
    '"Listen closely to the sound of my voice. From a calm, steady whisper... to a bright, confident exclamation — every word carries warmth, clarity, and natural human expression."',
  harvard:
    '"The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background... It is easy to tell the depth of a well — kick the ball straight and follow through."',
  technical:
    '"Hello, this is my voice calibration sample for neural voice cloning. I am speaking clearly at a natural pace, pausing at commas, periods... and em dashes — so the system captures my true vocal timbre."',
};

const state = {
  voices: { builtin: [], cloned: [], provider: "CPU" },
  emotions: [],
  selectedEmotion: "neutral",
  emotionMap: {},
  timeline: [],
  recordedAudioBase64: null,
  mediaRecorder: null,
  micChunks: [],
  // Live Microphone PCM WAV Recorder & Teleprompter state
  isMicRecording: false,
  micStream: null,
  micAudioCtx: null,
  micProcessor: null,
  micAnalyser: null,
  micPcmChunks: [],
  micSampleRate: 24000,
  micStartTime: 0,
  micMeterRaf: null,
  // Web Audio API nodes
  audioCtx: null,
  analyser: null,
  sourceNode: null,
  freqData: null,
  timeData: null,
  // 60FPS 3D Canvas Animation state
  phase: 0,
  smoothRms: 0,
  smoothBass: 0,
  smoothMid: 0,
  smoothHigh: 0,
  avatarDots: [],
  bodyCircuits: [],
  envMotes3D: [],
  activeChunk: null,
  activePause: null,
  // Synthesis Progress inside Avatar Animation
  isSynthesizing: false,
  synthProgress: 0,
  synthDisplayProgress: 0,
  synthFadeAlpha: 0,
  synthStageLabel: "",
  synthStartMs: 0,
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  initAnimatedAvatarDots();
  initBodyCircuits();
  init3DEnvironmentMotes();
  bindUIEvents();
  bindDialogEvents();
  startCanvasLoop();
  startSystemMonitor();
  await fetchStudioInit();
  updateSettingsSummaryStrip();
});

// ---------------------------------------------------------------------------
// Live GPU, RAM, and CPU Separate Accurate Sparkline Charts (Transparent)
// ---------------------------------------------------------------------------
function startSystemMonitor() {
  const gpuCanvas = document.getElementById("sysGpuChart");
  const ramCanvas = document.getElementById("sysRamChart");
  const cpuCanvas = document.getElementById("sysCpuChart");
  if (!gpuCanvas && !ramCanvas && !cpuCanvas) return;

  const maxSamples = 32;
  const hist = {
    gpu: [],
    ram: [],
    cpu: [],
  };

  function drawSingleChart(canvas, data, stroke, fill) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (data.length === 0) return;

    // Plot only actual recorded samples (right-aligned so real data scrolls in cleanly)
    const stepX = W / (maxSamples - 1);
    const startIdx = maxSamples - data.length;

    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (startIdx + i) * stepX;
      const val = Math.max(0, Math.min(100, data[i]));
      const y = H - 2.5 - (val / 100) * (H - 5.5);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.7;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Soft translucent area fill strictly under actual samples
    const firstX = startIdx * stepX;
    const lastX = (startIdx + data.length - 1) * stepX;
    ctx.lineTo(lastX, H);
    ctx.lineTo(firstX, H);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();

    // Live glowing endpoint dot at exact current value
    const lastVal = Math.max(0, Math.min(100, data[data.length - 1]));
    const lastY = H - 2.5 - (lastVal / 100) * (H - 5.5);
    ctx.beginPath();
    ctx.arc(Math.min(W - 2.2, lastX), lastY, 2.3, 0, Math.PI * 2);
    ctx.fillStyle = stroke;
    ctx.shadowColor = stroke;
    ctx.shadowBlur = 6;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  function drawAllCharts() {
    drawSingleChart(gpuCanvas, hist.gpu, "#FF5C0A", "rgba(255, 92, 10, 0.22)");
    drawSingleChart(ramCanvas, hist.ram, "#38bdf8", "rgba(56, 189, 248, 0.20)");
    drawSingleChart(cpuCanvas, hist.cpu, "#10b981", "rgba(16, 185, 129, 0.20)");
  }

  async function pollStats() {
    try {
      const res = await fetch("/api/system-stats", { cache: "no-store" });
      if (!res.ok) return;
      const d = await res.json();
      const gpu = Number(d.gpu);
      const ram = Number(d.ram);
      const cpu = Number(d.cpu);
      if (!Number.isFinite(gpu) || !Number.isFinite(ram) || !Number.isFinite(cpu)) return;

      hist.gpu.push(gpu);
      hist.ram.push(ram);
      hist.cpu.push(cpu);
      if (hist.gpu.length > maxSamples) hist.gpu.shift();
      if (hist.ram.length > maxSamples) hist.ram.shift();
      if (hist.cpu.length > maxSamples) hist.cpu.shift();

      const gEl = document.getElementById("sysGpuVal");
      const rEl = document.getElementById("sysRamVal");
      const cEl = document.getElementById("sysCpuVal");
      if (gEl) {
        gEl.textContent = d.gpu_mem_total_gb > 0
          ? `${gpu.toFixed(0)}% • ${d.gpu_mem_used_gb.toFixed(1)}/${d.gpu_mem_total_gb.toFixed(0)}G`
          : `${gpu.toFixed(0)}%`;
      }
      if (rEl) {
        rEl.textContent = d.ram_total_gb > 0
          ? `${ram.toFixed(0)}% • ${d.ram_used_gb.toFixed(1)}/${d.ram_total_gb.toFixed(0)}G`
          : `${ram.toFixed(0)}%`;
      }
      if (cEl) {
        cEl.textContent = d.cpu_freq_ghz > 0
          ? `${cpu.toFixed(0)}% • ${d.cpu_freq_ghz.toFixed(1)}GHz`
          : `${cpu.toFixed(0)}%`;
      }

      drawAllCharts();
    } catch (_) {
      // Ignore transient poll errors
    }
  }

  pollStats();
  setInterval(pollStats, 1000);
}

async function fetchStudioInit() {
  try {
    const res = await fetch("/api/init");
    const data = await res.json();
    state.voices = data.voices;
    state.emotions = data.emotions;
    state.emotionMap = {};
    data.emotions.forEach((e) => {
      state.emotionMap[e.id] = e;
    });

    populateVoiceSelectors();
    renderEmotionGrid();
    renderSavedClonesList();
    updateSettingsSummaryStrip();
  } catch (err) {
    console.error("Failed to initialize studio:", err);
  }
}

// ---------------------------------------------------------------------------
// Settings Panel Tab Controller (Docked Left Column)
// ---------------------------------------------------------------------------
function openSettingsDialog(tabId = "dlgTabPauses") {
  const dlg = document.getElementById("settingsDialog");
  const targetTab = document.getElementById(tabId) ? tabId : "dlgTabPauses";
  switchDialogTab(targetTab);
  if (dlg && typeof dlg.showModal === "function" && !dlg.open) {
    dlg.showModal();
  } else if (dlg) {
    dlg.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function closeSettingsDialog() {
  const dlg = document.getElementById("settingsDialog");
  if (dlg && typeof dlg.close === "function" && dlg.open) {
    dlg.close();
  }
  updateSettingsSummaryStrip();
}

function switchDialogTab(tabId) {
  const targetTab = document.getElementById(tabId) ? tabId : "dlgTabPauses";
  document.querySelectorAll(".modal-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.dialogTab === targetTab);
  });
  document.querySelectorAll(".modal-tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === targetTab);
  });
}

function bindDialogEvents() {
  const dlg = document.getElementById("settingsDialog");

  // Jump to settings tab triggers (summary pills)
  document.querySelectorAll("[data-open-dialog]").forEach((btn) => {
    btn.addEventListener("click", () => {
      openSettingsDialog(btn.dataset.openDialog);
    });
  });

  // Switch tabs inside settings panel
  document.querySelectorAll("[data-dialog-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchDialogTab(btn.dataset.dialogTab);
    });
  });

  // Optional modal buttons (when rendered as <dialog>)
  document.getElementById("closeDialogBtn")?.addEventListener("click", closeSettingsDialog);
  document.getElementById("doneDialogBtn")?.addEventListener("click", closeSettingsDialog);
  document.getElementById("applyAndGenerateBtn")?.addEventListener("click", () => {
    closeSettingsDialog();
    handleSynthesize();
  });

  if (dlg && typeof dlg.close === "function") {
    dlg.addEventListener("click", (e) => {
      if (e.target === dlg) {
        closeSettingsDialog();
      }
    });
  }
}

function updateSettingsSummaryStrip() {
  const voiceSelect = document.getElementById("voiceSelect");
  const selectedOpt = voiceSelect.options[voiceSelect.selectedIndex];
  const voiceLabel = selectedOpt ? selectedOpt.textContent.split("[")[0].split("•")[0].trim() : "Heart (AF)";
  const elVoice = document.getElementById("summaryVoice");
  if (elVoice) elVoice.textContent = `Voice: ${voiceLabel}`;

  const emo = state.emotionMap[state.selectedEmotion] || { icon: "😐", label: "Neutral" };
  const intensity = Math.round(parseFloat(document.getElementById("emotionIntensity").value) * 100);
  const speed = parseFloat(document.getElementById("speedInput").value).toFixed(2);
  const elEmotion = document.getElementById("summaryEmotion");
  if (elEmotion) elEmotion.textContent = `Feeling: ${emo.icon} ${emo.label} (${intensity}% • ${speed}x)`;

  const c = parseFloat(document.getElementById("pauseComma").value).toFixed(2);
  const p = parseFloat(document.getElementById("pausePeriod").value).toFixed(2);
  const el = parseFloat(document.getElementById("pauseEllipsis").value).toFixed(2);
  const em = parseFloat(document.getElementById("pauseEmdash").value).toFixed(2);
  const elPauses = document.getElementById("summaryPauses");
  if (elPauses) elPauses.textContent = `Pauses: , ${c}s | . ${p}s | ... ${el}s | — ${em}s`;

  const fmt = document.getElementById("preferredFormat").value.toUpperCase();
  const customName = document.getElementById("customFilename").value.trim();
  const elFormat = document.getElementById("summaryFormat");
  if (elFormat) {
    elFormat.textContent = customName ? `Export: Manual ${fmt} (${customName})` : `Export: Manual Save (${fmt})`;
  }
}

// ---------------------------------------------------------------------------
// Voice & Emotion Selectors
// ---------------------------------------------------------------------------
function populateVoiceSelectors(selectVoiceId = null) {
  const voiceSelect = document.getElementById("voiceSelect");
  const prevVal = selectVoiceId || voiceSelect.value || "af_heart";
  voiceSelect.innerHTML = "";

  // Group 1: Cloned Voices
  if (state.voices.cloned && state.voices.cloned.length > 0) {
    const grpClone = document.createElement("optgroup");
    grpClone.label = "🧬 Cloned Voices (Custom)";
    state.voices.cloned.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = `${v.name} [${v.gender} • ${v.mode === "reference_audio" ? "Ref Audio Clone" : "Neural Blend"}]`;
      grpClone.appendChild(opt);
    });
    voiceSelect.appendChild(grpClone);
  }

  // Group 2: Built-in Kokoro Voices grouped by Region/Gender
  const groups = {};
  state.voices.builtin.forEach((v) => {
    const key = `${v.region} — ${v.gender}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(v);
  });

  Object.entries(groups).forEach(([label, list]) => {
    const grp = document.createElement("optgroup");
    grp.label = label;
    list.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.id;
      const f0Tag = v.f0_hz ? ` • ~${Math.round(v.f0_hz)}Hz` : "";
      opt.textContent = `${v.name}${f0Tag}`;
      grp.appendChild(opt);
    });
    voiceSelect.appendChild(grp);
  });

  if ([...voiceSelect.options].some((o) => o.value === prevVal)) {
    voiceSelect.value = prevVal;
  } else if (voiceSelect.options.length > 0) {
    voiceSelect.selectedIndex = 0;
  }

  // Populate Neural Blender dropdowns
  ["blendVoice1", "blendVoice2", "blendVoice3"].forEach((id, idx) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = "";
    state.voices.builtin.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = `${v.name} (${v.region} ${v.gender})`;
      sel.appendChild(opt);
    });
    const defaults = ["af_heart", "bf_emma", "af_bella"];
    sel.value = current || defaults[idx] || state.voices.builtin[0]?.id;
  });

  updateSettingsSummaryStrip();
}

function renderEmotionGrid() {
  const select = document.getElementById("emotionSelect");
  if (select) {
    select.innerHTML = "";
    state.emotions.forEach((emo) => {
      const opt = document.createElement("option");
      opt.value = emo.id;
      opt.textContent = `${emo.icon} ${emo.label} — ${emo.description}`;
      select.appendChild(opt);
    });
    select.value = state.selectedEmotion;
    const active = state.emotionMap[state.selectedEmotion];
    const descEl = document.getElementById("activeEmotionDesc");
    if (descEl && active) descEl.textContent = active.description;
  }

  const grid = document.getElementById("emotionGrid");
  if (!grid) return;
  grid.innerHTML = "";
  state.emotions.forEach((emo) => {
    const card = document.createElement("div");
    card.className = `emo-card ${emo.id === state.selectedEmotion ? "active" : ""}`;
    card.style.setProperty("--emo-color", emo.color);
    card.innerHTML = `
      <span class="emo-icon">${emo.icon}</span>
      <div>
        <div class="emo-name">${emo.label}</div>
      </div>
    `;
    card.addEventListener("click", () => {
      state.selectedEmotion = emo.id;
      const descEl = document.getElementById("activeEmotionDesc");
      if (descEl) descEl.textContent = `${emo.icon} ${emo.description}`;
      const hudEmo = document.getElementById("hudEmotionBadge");
      if (hudEmo) hudEmo.textContent = `${emo.icon} ${emo.label}`;
      renderEmotionGrid();
      updateSettingsSummaryStrip();
    });
    grid.appendChild(card);
  });
}

function renderSavedClonesList() {
  const container = document.getElementById("savedClonesList");
  if (!state.voices.cloned || state.voices.cloned.length === 0) {
    container.innerHTML = `<div class="empty-timeline-msg">No cloned voices saved yet. Use the Reference Audio Cloner or Neural Style Blender tab to create one!</div>`;
    return;
  }

  container.innerHTML = "";
  state.voices.cloned.forEach((c) => {
    const recipeStr = Object.entries(c.recipe || {})
      .map(([k, w]) => `${k} (${Math.round(w * 100)}%)`)
      .join(" + ");
    const item = document.createElement("div");
    item.className = "clone-item";
    item.innerHTML = `
      <div>
        <strong>${c.name}</strong>
        <div style="font-size:0.72rem;color:#94a3b8;">
          ${c.gender} • F0: ~${Math.round(c.f0_hz || 165)}Hz • Pitch Δ: ${c.pitch_shift_semitones || 0}st<br/>
          Manifold: ${recipeStr}
        </div>
      </div>
      <div style="display:flex;gap:6px;">
        <button type="button" class="btn-ghost btn-xs use-clone-btn" data-id="${c.id}">Select Voice</button>
        <button type="button" class="btn-ghost btn-xs del-clone-btn" data-id="${c.id}" style="color:#fda4af;">Delete</button>
      </div>
    `;
    container.appendChild(item);
  });

  container.querySelectorAll(".use-clone-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("voiceSelect").value = btn.dataset.id;
      applyClonedVoiceSettingsToUI(btn.dataset.id);
      closeSettingsDialog();
    });
  });

  container.querySelectorAll(".del-clone-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.id;
      const res = await fetch(`/api/clone/${id}`, { method: "DELETE" });
      if (res.ok) {
        const data = await res.json();
        state.voices = data.voices;
        populateVoiceSelectors();
        renderSavedClonesList();
      }
    });
  });
}

// ---------------------------------------------------------------------------
// UI Event Bindings & Live Punctuation Counter
// ---------------------------------------------------------------------------
function applyClonedVoiceSettingsToUI(voiceId) {
  const clonedList = state.voices?.cloned || [];
  const cv = clonedList.find((v) => v.id === voiceId);
  if (!cv) return;

  const intensityEl = document.getElementById("emotionIntensity");
  const valIntensityEl = document.getElementById("valIntensity");
  const speedEl = document.getElementById("speedInput");
  const valSpeedEl = document.getElementById("valSpeed");
  const pitchEl = document.getElementById("pitchInput");
  const valPitchEl = document.getElementById("valPitch");

  if (cv.detected_emotion_intensity !== undefined && intensityEl) {
    const val = Math.max(0.0, Math.min(1.5, Number(cv.detected_emotion_intensity)));
    intensityEl.value = val.toFixed(2);
    if (valIntensityEl) valIntensityEl.textContent = `${Math.round(val * 100)}%`;
  }
  if (cv.detected_speed !== undefined && speedEl) {
    const val = Math.max(0.6, Math.min(1.6, Number(cv.detected_speed)));
    speedEl.value = val.toFixed(2);
    if (valSpeedEl) valSpeedEl.textContent = `${val.toFixed(2)}x`;
  }
  if (cv.detected_pitch_offset !== undefined && pitchEl) {
    const val = Math.max(-4.0, Math.min(4.0, Number(cv.detected_pitch_offset)));
    pitchEl.value = val.toFixed(1);
    if (valPitchEl) valPitchEl.textContent = `${val > 0 ? "+" : ""}${val.toFixed(1)} st`;
  }
  updateSettingsSummaryStrip();
}

function bindUIEvents() {
  const scriptInput = document.getElementById("scriptInput");

  const updateScriptMeta = () => {
    const meta = document.getElementById("scriptMetaLabel");
    if (!meta || !scriptInput) return;
    const raw = scriptInput.value.trim();
    const words = raw ? raw.split(/\s+/).length : 0;
    meta.textContent = `${words} words • ${raw.length} chars`;
  };
  scriptInput.addEventListener("input", updateScriptMeta);
  updateScriptMeta();

  document.getElementById("voiceSelect").addEventListener("change", (e) => {
    applyClonedVoiceSettingsToUI(e.target.value);
    updateSettingsSummaryStrip();
  });
  document.getElementById("emotionSelect")?.addEventListener("change", (e) => {
    const emoId = e.target.value;
    state.selectedEmotion = emoId;
    const emo = state.emotionMap[emoId];
    const descEl = document.getElementById("activeEmotionDesc");
    if (descEl && emo) descEl.textContent = emo.description;
    updateSettingsSummaryStrip();
  });
  document.getElementById("preferredFormat").addEventListener("change", updateSettingsSummaryStrip);
  document.getElementById("customFilename").addEventListener("input", updateSettingsSummaryStrip);

  // Studio Settings Summary Pulldown -> opens Settings Dialog directly to chosen tab
  const settingsSummarySelect = document.getElementById("settingsSummarySelect");
  if (settingsSummarySelect) {
    settingsSummarySelect.addEventListener("change", () => {
      const tabId = settingsSummarySelect.value;
      if (tabId) {
        openSettingsDialog(tabId);
        settingsSummarySelect.value = "";
      }
    });
  }

  // Preset Script Pulldown
  const presetSelect = document.getElementById("presetSelect");
  if (presetSelect) {
    presetSelect.addEventListener("change", () => {
      const key = presetSelect.value;
      if (key && PRESET_SCRIPTS[key]) {
        scriptInput.value = PRESET_SCRIPTS[key];
        scriptInput.dispatchEvent(new Event("input"));
      }
      presetSelect.value = "";
    });
  }

  // Legacy preset buttons (if any)
  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.preset;
      if (PRESET_SCRIPTS[key]) {
        scriptInput.value = PRESET_SCRIPTS[key];
      }
    });
  });

  // Pulldown selectors for Insert Pause & Inline Feeling tags
  const insertAtCursor = (snippet) => {
    if (!snippet) return;
    const start = scriptInput.selectionStart ?? scriptInput.value.length;
    const end = scriptInput.selectionEnd ?? scriptInput.value.length;
    const text = scriptInput.value;
    scriptInput.value = text.slice(0, start) + snippet + text.slice(end);
    scriptInput.focus();
    scriptInput.selectionStart = scriptInput.selectionEnd = start + snippet.length;
  };

  const pauseSelect = document.getElementById("insertPauseSelect");
  pauseSelect.addEventListener("change", () => {
    insertAtCursor(pauseSelect.value);
    pauseSelect.value = "";
  });

  const feelingSelect = document.getElementById("insertFeelingSelect");
  feelingSelect.addEventListener("change", () => {
    insertAtCursor(feelingSelect.value);
    feelingSelect.value = "";
  });

  // Pause & Prosody sliders
  const bindSlider = (id, labelId, fmt) => {
    const el = document.getElementById(id);
    const lbl = document.getElementById(labelId);
    if (!el || !lbl) return;
    el.addEventListener("input", () => {
      lbl.textContent = fmt(parseFloat(el.value));
      updateSettingsSummaryStrip();
    });
  };

  bindSlider("pauseComma", "valComma", (v) => `${v.toFixed(2)}s`);
  bindSlider("pausePeriod", "valPeriod", (v) => `${v.toFixed(2)}s`);
  bindSlider("pauseEllipsis", "valEllipsis", (v) => `${v.toFixed(2)}s`);
  bindSlider("pauseEmdash", "valEmdash", (v) => `${v.toFixed(2)}s`);

  document.getElementById("resetPausesBtn").addEventListener("click", () => {
    const defaults = {
      pauseComma: [0.22, "valComma"],
      pausePeriod: [0.50, "valPeriod"],
      pauseEllipsis: [0.95, "valEllipsis"],
      pauseEmdash: [0.40, "valEmdash"],
    };
    Object.entries(defaults).forEach(([id, [val, lblId]]) => {
      document.getElementById(id).value = val;
      document.getElementById(lblId).textContent = `${val.toFixed(2)}s`;
    });
    updateSettingsSummaryStrip();
  });

  bindSlider("emotionIntensity", "valIntensity", (v) => `${Math.round(v * 100)}%`);
  bindSlider("speedInput", "valSpeed", (v) => `${v.toFixed(2)}x`);
  bindSlider("pitchInput", "valPitch", (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} st`);

  // Cloning sliders
  bindSlider("cloneRefPitch", "valClonePitch", (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} st`);
  bindSlider("cloneRefWarmth", "valCloneWarmth", (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`);
  bindSlider("blendPitch", "valBlendPitch", (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} st`);
  bindSlider("blendWarmth", "valBlendWarmth", (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`);
  bindSlider("blendWeight1", "blendVal1", (v) => `${Math.round(v)}%`);
  bindSlider("blendWeight2", "blendVal2", (v) => `${Math.round(v)}%`);
  bindSlider("blendWeight3", "blendVal3", (v) => `${Math.round(v)}%`);

  // Voice Cloning Sub-Tabs inside Dialog
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // Reference Audio File Upload handler
  document.getElementById("cloneRefFile").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      state.recordedAudioBase64 = reader.result;
      const preview = document.getElementById("refPreviewAudio");
      preview.src = reader.result;
      preview.style.display = "inline-block";
      document.getElementById("micStatus").textContent = `Loaded file: ${file.name}`;
    };
    reader.readAsDataURL(file);
  });

  // Live Microphone Recording for Voice Cloning
  document.getElementById("micRecordBtn").addEventListener("click", toggleMicRecording);

  // Reading Context selector & Next Script button for Microphone Voice Cloning
  const promptSel = document.getElementById("readingPromptSelect");
  const teleprompter = document.getElementById("readingTeleprompter");
  promptSel.addEventListener("change", () => {
    const key = promptSel.value;
    if (CLONE_READING_CONTEXTS[key]) {
      teleprompter.textContent = CLONE_READING_CONTEXTS[key];
    }
  });

  document.getElementById("nextPromptBtn").addEventListener("click", () => {
    const keys = Object.keys(CLONE_READING_CONTEXTS);
    const curIdx = keys.indexOf(promptSel.value);
    const nextKey = keys[(curIdx + 1) % keys.length];
    promptSel.value = nextKey;
    teleprompter.textContent = CLONE_READING_CONTEXTS[nextKey];
  });

  // Create Reference Audio Clone button
  document.getElementById("createRefCloneBtn").addEventListener("click", handleCreateReferenceClone);

  // Optional Neural Style Blend Clone button (removed when pure input cloning is active)
  document.getElementById("createBlendCloneBtn")?.addEventListener("click", handleCreateBlendClone);

  // Synthesize Speech button
  document.getElementById("synthesizeBtn").addEventListener("click", handleSynthesize);

  // Manual Save WAV & Save MP3 buttons (only saves when explicitly clicked by user)
  // Manual Save WAV / MP3 Pulldown
  const manualSaveSelect = document.getElementById("manualSaveSelect");
  if (manualSaveSelect) {
    manualSaveSelect.addEventListener("change", () => {
      const fmt = manualSaveSelect.value;
      if (fmt === "wav" || fmt === "mp3") {
        handleManualSave(fmt);
      }
      manualSaveSelect.value = "";
    });
  }
  document.getElementById("downloadWavBtn")?.addEventListener("click", () => handleManualSave("wav"));
  document.getElementById("downloadMp3Btn")?.addEventListener("click", () => handleManualSave("mp3"));
}

// ---------------------------------------------------------------------------
// Voice Cloning Handlers (Reference Audio / Live Mic WAV & Neural Style Blender)
// ---------------------------------------------------------------------------
function encodeFloat32ToWavBase64(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true);  // PCM format
  view.setUint16(22, 1, true);  // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); // 16-bit
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return "data:audio/wav;base64," + btoa(binary);
}

async function toggleMicRecording() {
  const btn = document.getElementById("micRecordBtn");
  const status = document.getElementById("micStatus");
  const card = document.getElementById("readingContextCard");
  const paceFill = document.getElementById("readingPaceFill");
  const levelFill = document.getElementById("micLevelFill");
  const dbLabel = document.getElementById("micDbLabel");
  const timerBadge = document.getElementById("micTimerBadge");

  // Stop active recording and finalize lossless WAV buffer
  if (state.isMicRecording) {
    state.isMicRecording = false;
    if (state.micMeterRaf) cancelAnimationFrame(state.micMeterRaf);
    if (state.micProcessor) {
      state.micProcessor.disconnect();
      state.micProcessor = null;
    }
    if (state.micStream) {
      state.micStream.getTracks().forEach((t) => t.stop());
      state.micStream = null;
    }
    if (state.micAudioCtx) {
      await state.micAudioCtx.close();
      state.micAudioCtx = null;
    }

    btn.classList.remove("recording");
    card.classList.remove("recording-active");
    btn.textContent = "🎙️ Re-Record Microphone";
    levelFill.style.width = "0%";

    // Merge recorded PCM chunks into single Float32Array
    const totalLen = state.micPcmChunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLen < state.micSampleRate * 0.5) {
      status.textContent = "⚠️ Recording was too short (< 0.5s). Please click Record and read the sentence above.";
      return;
    }

    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of state.micPcmChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    const wavDataUrl = encodeFloat32ToWavBase64(merged, state.micSampleRate);
    state.recordedAudioBase64 = wavDataUrl;

    const preview = document.getElementById("refPreviewAudio");
    preview.src = wavDataUrl;
    preview.style.display = "inline-block";
    const durSec = (totalLen / state.micSampleRate).toFixed(1);
    status.textContent = `✅ Captured ${durSec}s lossless voice sample! Click "🧬 Analyze & Clone Voice" below.`;
    return;
  }

  // Start new microphone capture
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    state.micAudioCtx = audioCtx;
    state.micSampleRate = audioCtx.sampleRate;
    state.micStream = stream;
    state.micPcmChunks = [];
    state.isMicRecording = true;
    state.micStartTime = performance.now();

    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    state.micAnalyser = analyser;

    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    state.micProcessor = processor;

    processor.onaudioprocess = (e) => {
      if (!state.isMicRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      state.micPcmChunks.push(new Float32Array(input));
    };

    source.connect(analyser);
    analyser.connect(processor);
    processor.connect(audioCtx.destination);

    btn.classList.add("recording");
    card.classList.add("recording-active");
    btn.textContent = "⏹️ Stop Recording (When Finished Reading)";
    status.textContent = "🔴 Recording... Read the highlighted passage above aloud in your natural voice.";

    const timeBuf = new Uint8Array(analyser.fftSize);
    const updateMicHUD = () => {
      if (!state.isMicRecording) return;
      analyser.getByteTimeDomainData(timeBuf);
      let sumSq = 0;
      for (let i = 0; i < timeBuf.length; i++) {
        const v = (timeBuf[i] - 128) / 128.0;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / timeBuf.length);
      const levelPct = Math.min(100, Math.round(rms * 320));
      levelFill.style.width = `${levelPct}%`;
      dbLabel.textContent = `Level: ${levelPct}%`;

      const elapsedSec = (performance.now() - state.micStartTime) / 1000.0;
      timerBadge.textContent = `${elapsedSec.toFixed(1)}s`;
      // Recommended reading pace bar (~9 seconds for the passage)
      const pacePct = Math.min(100, (elapsedSec / 9.0) * 100);
      paceFill.style.width = `${pacePct}%`;

      state.micMeterRaf = requestAnimationFrame(updateMicHUD);
    };
    state.micMeterRaf = requestAnimationFrame(updateMicHUD);
  } catch (err) {
    status.textContent = `❌ Microphone access error: ${err.message}`;
  }
}

async function handleCreateReferenceClone() {
  const btn = document.getElementById("createRefCloneBtn");
  const resBox = document.getElementById("cloneAnalysisResult");
  const name = document.getElementById("cloneRefName").value.trim() || "Cloned Voice";

  if (!state.recordedAudioBase64) {
    resBox.classList.remove("hidden");
    resBox.textContent = "⚠️ Please upload a reference .wav/.mp3 file or record from your microphone first.";
    return;
  }

  btn.disabled = true;
  btn.textContent = "🧬 Extracting Vocal Formants & Cloning...";

  try {
    const genderSel = document.getElementById("cloneRefGender");
    const chosenGender = genderSel ? genderSel.value : "female";
    const payload = {
      name,
      audio_base64: state.recordedAudioBase64,
      gender: chosenGender,
      manual_pitch_offset: 0.0,
      warmth_bias_db: 0.0,
    };
    const res = await fetch("/api/clone/reference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Cloning failed");

    state.voices = data.voices;
    const cv = data.cloned_voice;
    populateVoiceSelectors(cv.id);
    applyClonedVoiceSettingsToUI(cv.id);
    renderSavedClonesList();

    const recipeText = Object.entries(cv.recipe)
      .map(([v, w]) => `${v} (${Math.round(w * 100)}%)`)
      .join(" + ");
    const fp = cv.target_fingerprint || {};
    const formantsStr = fp.f1_hz
      ? `F1: ${fp.f1_hz}Hz • F2: ${fp.f2_hz}Hz • F3: ${fp.f3_hz}Hz`
      : `Centroid: ${fp.centroid_hz}Hz`;
    const detIntPct = Math.round((cv.detected_emotion_intensity ?? 1.0) * 100);
    const detSpd = Number(cv.detected_speed ?? 1.0).toFixed(2);
    const detPitch = Number(cv.detected_pitch_offset ?? 0.0);
    const detPitchStr = `${detPitch > 0 ? "+" : ""}${detPitch.toFixed(1)} st`;

    resBox.classList.remove("hidden");
    resBox.innerHTML = `
      <strong>✅ Exact Voice Clone Created &amp; Applied to Text-to-Voice: ${cv.name}</strong><br/>
      • Gender: <strong>${cv.gender}</strong> • Input F0: <strong>${fp.f0_hz} Hz</strong> • Formants: <strong>${formantsStr}</strong><br/>
      • Auto-Applied from Input → Intensity: <strong>${detIntPct}%</strong> • Base Speed: <strong>${detSpd}x</strong> • Pitch Offset: <strong>${detPitchStr}</strong>
    `;
  } catch (err) {
    resBox.classList.remove("hidden");
    resBox.textContent = `❌ Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🧬 Clone Exact Voice from Input (No Modification)";
  }
}

async function handleCreateBlendClone() {
  const btn = document.getElementById("createBlendCloneBtn");
  const name = document.getElementById("cloneBlendName").value.trim() || "Neural Blend";
  const v1 = document.getElementById("blendVoice1").value;
  const v2 = document.getElementById("blendVoice2").value;
  const v3 = document.getElementById("blendVoice3").value;
  const w1 = parseFloat(document.getElementById("blendWeight1").value);
  const w2 = parseFloat(document.getElementById("blendWeight2").value);
  const w3 = parseFloat(document.getElementById("blendWeight3").value);

  const weights = {};
  if (w1 > 0) weights[v1] = (weights[v1] || 0) + w1;
  if (w2 > 0) weights[v2] = (weights[v2] || 0) + w2;
  if (w3 > 0) weights[v3] = (weights[v3] || 0) + w3;

  btn.disabled = true;
  try {
    const res = await fetch("/api/clone/blend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        voice_weights: weights,
        pitch_shift_semitones: parseFloat(document.getElementById("blendPitch").value),
        warmth_bias_db: parseFloat(document.getElementById("blendWarmth").value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Blend failed");

    state.voices = data.voices;
    populateVoiceSelectors(data.cloned_voice.id);
    renderSavedClonesList();
    switchDialogTab("dlgTabVoice");
  } catch (err) {
    alert(`Failed to create blended voice: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Synthesize Speech & Connect Web Audio API Visualizer
// ---------------------------------------------------------------------------
function setEngineStatus(mode, title, sub) {
  const dot = document.getElementById("engineStatusDot");
  const titleEl = document.getElementById("engineStatusTitle");
  const subEl = document.getElementById("engineStatusSub");
  if (dot) {
    dot.classList.remove("synthesizing", "speaking");
    if (mode === "synthesizing" || mode === "speaking") {
      dot.classList.add(mode);
    }
  }
  if (titleEl) titleEl.textContent = title;
  if (subEl) subEl.textContent = sub;
}

async function handleSynthesize() {
  const btn = document.getElementById("synthesizeBtn");
  const text = document.getElementById("scriptInput").value.trim();
  if (!text) return;

  btn.disabled = true;
  btn.innerHTML = "<span>⚡ Synthesizing Speech...</span>";

  // Activate in-avatar holographic synthesis progress animation
  state.isSynthesizing = true;
  state.synthProgress = 5;
  state.synthDisplayProgress = 0;
  state.synthStartMs = performance.now();
  state.synthStageLabel = "Parsing punctuation & feelings...";
  setEngineStatus("synthesizing", "Synthesizing Speech", state.synthStageLabel);

  try {
    const payload = {
      text,
      voice_id: document.getElementById("voiceSelect").value,
      emotion: state.selectedEmotion,
      emotion_intensity: parseFloat(document.getElementById("emotionIntensity").value),
      speed: parseFloat(document.getElementById("speedInput").value),
      pitch_semitones: parseFloat(document.getElementById("pitchInput").value),
      lang: document.getElementById("langSelect").value,
      pause_comma: parseFloat(document.getElementById("pauseComma").value),
      pause_period: parseFloat(document.getElementById("pausePeriod").value),
      pause_ellipsis: parseFloat(document.getElementById("pauseEllipsis").value),
      pause_em_dash: parseFloat(document.getElementById("pauseEmdash").value),
      output_format: document.getElementById("preferredFormat").value,
      save_filename: document.getElementById("customFilename").value.trim() || null,
    };

    const res = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Synthesis failed");

    // Complete progress smoothly to 100% before launching audio
    state.synthProgress = 100;
    state.synthStageLabel = "Voice synthesis complete — Launching audio...";
    setEngineStatus("synthesizing", "Synthesis Complete • 100%", "Launching 24 kHz audio...");
    await new Promise((r) => setTimeout(r, 240));
    state.isSynthesizing = false;

    state.timeline = data.timeline || [];
    state.lastDefaultStem = data.default_stem || "kokoro_speech";
    renderKaraokeTimeline(state.timeline);
    enableManualSaveButtons();

    document.getElementById("synthStatsLabel").textContent =
      `${data.duration_sec}s audio ready in memory (${data.generation_time_sec}s, RTF ${data.rtf}x) — Click Save WAV or Save MP3 to save`;

    // Play in-memory audio in the Web Audio connected player (no automatic disk file saved)
    const player = document.getElementById("mainAudioPlayer");
    ensureWebAudioGraph(player);
    player.src = data.audio_base64;
    await player.play();
  } catch (err) {
    state.isSynthesizing = false;
    state.synthProgress = 0;
    setEngineStatus("ready", "CODERSTATION Studio", "Neural Voice Technology");
    alert(`Synthesis Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>🔊 Generate &amp; Animate Speech</span>";
  }
}

function enableManualSaveButtons() {
  const saveSelect = document.getElementById("manualSaveSelect");
  if (saveSelect) {
    saveSelect.disabled = false;
  }

  const wavBtn = document.getElementById("downloadWavBtn");
  const mp3Btn = document.getElementById("downloadMp3Btn");
  if (wavBtn) {
    wavBtn.disabled = false;
    wavBtn.classList.remove("disabled");
  }
  if (mp3Btn) {
    mp3Btn.disabled = false;
    mp3Btn.classList.remove("disabled");
  }
}

async function handleManualSave(format) {
  const saveSelect = document.getElementById("manualSaveSelect");
  const firstOpt = saveSelect ? saveSelect.options[0] : null;
  const origOptText = firstOpt ? firstOpt.textContent : "";

  const customInput = document.getElementById("customFilename").value.trim();
  const defaultName = customInput || state.lastDefaultStem || "kokoro_speech";

  const chosenName = window.prompt(
    `Save ${format.toUpperCase()} output as filename:`,
    defaultName
  );
  if (chosenName === null) {
    return; // User cancelled manual save dialog
  }

  if (saveSelect) {
    saveSelect.disabled = true;
    if (firstOpt) firstOpt.textContent = `⏳ Saving ${format.toUpperCase()}...`;
  }

  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        format,
        filename: chosenName.trim() || defaultName,
      }),
    });
    const saved = await res.json();
    if (!res.ok) throw new Error(saved.detail || "Save failed");

    // Trigger browser download for the manually saved file
    const link = document.createElement("a");
    link.href = saved.download_url;
    link.download = saved.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();

    prependHistoryItem(saved);
  } catch (err) {
    alert(`Manual Save Error: ${err.message}`);
  } finally {
    if (saveSelect) {
      saveSelect.disabled = false;
      saveSelect.value = "";
      if (firstOpt) firstOpt.textContent = origOptText;
    }
  }
}

function prependHistoryItem(saved) {
  const list = document.getElementById("historyList");
  if (list.querySelector(".empty-timeline-msg")) {
    list.innerHTML = "";
  }
  const item = document.createElement("div");
  item.className = "history-item";
  const cleanText = (saved.text || "").replace(/\[.*?\]/g, "").trim() || saved.filename;
  const fmtBadge = saved.format === "mp3" ? "btn-mp3" : "btn-wav";
  item.innerHTML = `
    <div class="history-meta">
      <strong>${saved.filename}</strong>
      <small>${cleanText.slice(0, 55)}${cleanText.length > 55 ? "..." : ""} (${saved.duration_sec}s)</small>
    </div>
    <div class="history-actions">
      <a class="btn-export ${fmtBadge}" href="${saved.download_url}" download="${saved.filename}">Download ${saved.format.toUpperCase()}</a>
    </div>
  `;
  list.prepend(item);
}

function renderKaraokeTimeline(timeline) {
  const container = document.getElementById("karaokeTimeline");
  container.innerHTML = "";
  timeline.forEach((seg, idx) => {
    const chunkEl = document.createElement("span");
    chunkEl.className = "k-chunk";
    chunkEl.id = `k-chunk-${idx}`;
    chunkEl.style.borderColor = `${seg.emotion_color}55`;
    chunkEl.innerHTML = `<span>${seg.emotion_icon}</span> <span>${seg.text}${seg.punctuation}</span>`;
    chunkEl.addEventListener("click", () => {
      const player = document.getElementById("mainAudioPlayer");
      if (player.src) {
        player.currentTime = seg.speech_start;
        player.play();
      }
    });
    container.appendChild(chunkEl);

    if (seg.pause_type && seg.pause_type !== "none" && seg.pause_duration > 0) {
      const pauseEl = document.createElement("span");
      pauseEl.className = "k-pause-tag";
      pauseEl.id = `k-pause-${idx}`;
      const symMap = { comma: ",", period: ".", ellipsis: "...", em_dash: "—" };
      pauseEl.textContent = `⏸ ${symMap[seg.pause_type] || ""} ${seg.pause_duration.toFixed(2)}s`;
      container.appendChild(pauseEl);
    }
  });
}

// ---------------------------------------------------------------------------
// 5. Web Audio API + 60FPS Canvas Animated Graphical Effects in Speech
// ---------------------------------------------------------------------------
function ensureWebAudioGraph(audioElement) {
  if (!state.audioCtx) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    state.audioCtx = new AudioCtx();
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = 1024;
    state.analyser.smoothingTimeConstant = 0.78;

    state.sourceNode = state.audioCtx.createMediaElementSource(audioElement);
    state.sourceNode.connect(state.analyser);
    state.analyser.connect(state.audioCtx.destination);

    state.freqData = new Uint8Array(state.analyser.frequencyBinCount);
    state.timeData = new Uint8Array(state.analyser.fftSize);
  }
  if (state.audioCtx.state === "suspended") {
    state.audioCtx.resume();
  }
}

/**
 * Anatomical half-width of the sculpted human head & ears matching the reference silhouette
 * (dy in [-124, +92]: sleek rounded cranium, distinct ear silhouette, tapered cheeks, soft rounded chin).
 */
function getAnatomicalHeadHalfW(dy) {
  const topY = -124;
  const botY = 92;
  if (dy <= topY || dy >= botY) return 0;
  const midY = -22;
  let hw = 0;
  if (dy <= midY) {
    // Sleek, smooth rounded human cranial dome (max half-width ~64px)
    const u = (dy - midY) / (midY - topY); // -1 at crown, 0 at temples
    hw = 64 * Math.pow(Math.max(0, 1 - u * u), 0.48);
  } else {
    // Graceful cheek taper into a softly rounded human chin
    const u = (dy - midY) / (botY - midY); // 0 at temples, +1 at chin apex
    const oval = Math.sqrt(Math.max(0, 1 - u * u));
    hw = 64 * oval * (1 - 0.10 * Math.pow(u, 1.3) - 0.14 * Math.pow(u, 2.3));
  }
  // Distinct, sleek human ear silhouette at dy in [-20, +28]
  if (dy >= -20 && dy <= 28) {
    const earT = (dy + 20) / 48; // 0 top of ear, 1 earlobe
    const earCurve = Math.sin(earT * Math.PI) * (0.85 + 0.25 * Math.sin(earT * Math.PI * 0.7));
    hw += earCurve * 7.2;
  }
  return hw;
}

/**
 * Graceful human neck, sloping trapezius, and wide smoothly-rounded shoulders
 * matching the reference silhouette (dy in [64, 265] relative to head center).
 */
function getAnatomicalTorsoHalfW(dy) {
  if (dy < 64) return 0;

  // 1. Tall, Slender Concave Human Neck (dy = 64 .. 122):
  //    Tucks neatly under the jaw, curves inward to ~31px waist, then begins flaring at neck base
  if (dy <= 122) {
    const t = (dy - 64) / 58; // 0 under jaw, 1 at base of neck
    const inward = Math.sin(t * Math.PI) * 5.2;
    const baseFlare = Math.pow(Math.max(0, (t - 0.62) / 0.38), 2.0) * 14.0;
    return 36.5 - inward + baseFlare; // 36.5 -> ~31.3 -> 50.5
  }

  // 2. Smooth Diagonal Trapezius Slope into Broad Shoulders (dy = 122 .. 176):
  //    Continuous graceful S-curve sweeping outward & downward from neck base (50.5px -> 148px)
  if (dy <= 176) {
    const t = (dy - 122) / 54;
    const smoothS = t * t * (3 - 2 * t);
    return 50.5 + smoothS * 97.5; // 50.5 -> 148
  }

  // 3. Wide, Smoothly Rounded Deltoid Shoulder Dome (dy = 176 .. 226):
  //    Rounds smoothly over the outer shoulder cap (148px -> 178px)
  if (dy <= 226) {
    const t = (dy - 176) / 50;
    const dome = Math.sin(t * Math.PI * 0.5);
    return 148 + dome * 30; // 148 -> 178
  }

  // 4. Outer Upper Arm Contour (dy = 226 .. 265):
  const t = Math.min(1.0, (dy - 226) / 39);
  return 178 + Math.sin(t * Math.PI * 0.5) * 4; // 178 -> 182
}

/**
 * 3D Human Facial Relief offset for a dot at normalized face coordinates (nx, ny)
 */
function getDotFacialReliefY(nx, ny) {
  const r2 = Math.min(1, nx * nx + ny * ny);
  const zSphere = Math.sqrt(Math.max(0, 1 - r2));
  let archY = ny * (1 - zSphere) * 9.0;

  // Smooth 3D cranial/facial curvature without mouth or eye figures
  const nose =
    Math.exp(-Math.pow(nx / 0.14, 2)) * Math.exp(-Math.pow((ny - 0.04) / 0.22, 2));
  archY -= nose * 3.8;

  return archY;
}

/**
 * Builds the larger Human-Form Avatar using a clean constellation of animated dots
 * with realistic human shoulders, deltoid caps, collarbones, neck, and 3D human face.
 * Every dot moves in real time. No yellow colors — strictly #062E52 blue & #FF5C0A orange.
 */
function initAnimatedAvatarDots() {
  const dots = [];

  // 1. High-Resolution 3D Head Outer Silhouette & Inner Micro-Bevel Rim (Tiny Dots)
  const rimSteps = 44;
  for (let i = 0; i <= rimSteps; i++) {
    const t = i / rimSteps;
    const dy = -122 + t * 218;
    const hw = getAnatomicalHeadHalfW(dy);
    if (hw < 3) {
      dots.push({
        x: 0,
        y: dy,
        role: "rim",
        group: "head_rim_center",
        isSecondary: false,
        size: 1.05,
        row: i,
        u: 0,
        phaseOff: i * 0.22,
      });
    } else {
      [-1, 1].forEach((side) => {
        dots.push({
          x: side * hw,
          y: dy,
          role: "rim",
          group: `head_rim_${side}`,
          isSecondary: false,
          size: 1.0,
          row: i,
          u: side,
          phaseOff: i * 0.2 + side,
        });
        // Inner micro-bevel rim for crisp 3D cranial edge definition
        if (i % 2 === 0 && hw > 14) {
          dots.push({
            x: side * (hw - 4.5),
            y: dy,
            role: "face_contour",
            group: `head_bevel_${side}`,
            isSecondary: false,
            size: 0.72,
            row: i,
            u: side * 0.92,
            phaseOff: i * 0.24 + side * 1.3,
          });
        }
      });
    }
  }

  // 2. High-Density 3D Sculpted Head Surface Point Cloud (27 rows of tiny glowing dots, zero orange in head)
  const numFaceRows = 27;
  for (let r = 0; r < numFaceRows; r++) {
    const tRow = (r + 0.6) / (numFaceRows + 0.2);
    const dy = -117 + tRow * 208;
    const hw = getAnatomicalHeadHalfW(dy);
    if (hw < 10) continue;

    const ny = (dy + 13) / 110; // -1 at crown, +1 at chin
    const numCols = Math.max(8, Math.round((hw * 2) / 8.2));
    const normCoreY = (dy - 14) / 56;

    for (let c = 1; c < numCols; c++) {
      const u = (c / numCols) * 2 - 1; // -1 to +1
      const archY = getDotFacialReliefY(u, ny);

      const radialCoreSq = (u * u) / 0.44 + normCoreY * normCoreY;
      const inCore = radialCoreSq <= 1.0;
      const coreWeight = Math.max(0, 1 - radialCoreSq);

      dots.push({
        x: u * hw,
        y: dy + archY,
        role: inCore ? "face_core" : "face_contour",
        group: `face_row_${r}`,
        isSecondary: false,
        coreWeight,
        size: inCore ? 0.78 + coreWeight * 0.34 : 0.68,
        row: r,
        u,
        phaseOff: r * 0.34 + u * 2.6,
      });
    }
  }

  // 2B. Cylindrical 3D Neck Micro-Dot Rings (tall slender concave neck matching reference)
  for (let nr = 0; nr < 7; nr++) {
    const dy = 86 + nr * 6.2;
    const hw = getAnatomicalTorsoHalfW(dy);
    const cols = 9;
    for (let c = 1; c < cols; c++) {
      const u = (c / cols) * 2 - 1;
      const px = u * hw * 0.90;
      if (dy < 94 && Math.abs(px) < getAnatomicalHeadHalfW(dy) * 0.72) continue;
      dots.push({
        x: px,
        y: dy,
        role: "shoulder_contour",
        group: `neck_ring_${nr}`,
        isSecondary: false,
        size: 0.66,
        row: nr,
        u,
        phaseOff: nr * 0.45 + u * 2.1,
      });
    }
  }

  // 3. High-Definition Neck, Sloping Trapezius & Wide Rounded Shoulders Double-Bevel Tiny-Dot Rim
  const torsoRimSteps = 50;
  for (let i = 0; i <= torsoRimSteps; i++) {
    const t = i / torsoRimSteps;
    const dy = 66 + t * 194;
    const hw = getAnatomicalTorsoHalfW(dy);
    [-1, 1].forEach((side) => {
      if (dy < 94 && hw < getAnatomicalHeadHalfW(dy) + 1) return;
      // Outer silhouette tiny dot
      dots.push({
        x: side * hw,
        y: dy,
        role: "shoulder_rim",
        group: `torso_rim_${side}`,
        isSecondary: false,
        size: 1.0,
        row: i,
        u: side,
        phaseOff: i * 0.18 + side * 0.5,
      });
      // Inner 3D sculpted shoulder bevel row
      if (dy >= 118 && i % 2 === 0) {
        dots.push({
          x: side * (hw - 5.5),
          y: dy + 1.5,
          role: "shoulder_contour",
          group: `torso_bevel_${side}`,
          isSecondary: false,
          size: 0.72,
          row: i,
          u: side * 0.92,
          phaseOff: i * 0.22 + side,
        });
      }
    });
  }

  // 3B. Sculpted 3D Sloping Clavicle, Rounded Deltoid Shoulder Dome & Transparent Upper-Torso Micro-Dot Mesh
  [-1, 1].forEach((side) => {
    // High-density Clavicle ridge following the graceful diagonal trapezius slope
    for (let c = 0; c <= 18; c++) {
      const f = c / 18;
      const cx = side * (22 + f * 124);
      const cy = 136 + Math.sin(f * Math.PI * 0.65) * 10 + f * 32;
      if (Math.abs(cx) > getAnatomicalTorsoHalfW(cy) - 4) continue;
      dots.push({
        x: cx,
        y: cy,
        role: "shoulder_rim",
        isSecondary: false,
        size: c % 4 === 0 ? 0.96 : 0.74,
        row: c,
        u: side * f,
        phaseOff: c * 0.28 + side * 0.4,
      });
    }

    // Layered 3D Rounded Deltoid Shoulder Cap Micro-Dot Mesh (6 smooth anatomical layers)
    for (let layer = 0; layer < 6; layer++) {
      const baseY = 166 + layer * 16;
      const steps = 11;
      for (let s = 0; s <= steps; s++) {
        const f = s / steps;
        const px = side * (108 + f * 66);
        const py = baseY + Math.sin(f * Math.PI * 0.55) * 12;
        if (py > 256 || Math.abs(px) > getAnatomicalTorsoHalfW(py) - 5) continue;
        dots.push({
          x: px,
          y: py,
          role: "shoulder_contour",
          isSecondary: false,
          size: 0.70,
          row: layer,
          u: side * (0.66 + f * 0.34),
          phaseOff: layer * 0.42 + s * 0.25,
        });
      }
    }

    // Airy Transparent Upper-Torso / Pectoral Contour Micro-Dots (lets 3D circuits shine through!)
    for (let pr = 0; pr < 7; pr++) {
      const py = 144 + pr * 16;
      const maxW = getAnatomicalTorsoHalfW(py) - 10;
      for (let pc = 0; pc < 6; pc++) {
        const px = side * (26 + pc * 17);
        if (Math.abs(px) > maxW) continue;
        // Keep central Arc Reactor window clear at (0, 204)
        if (Math.hypot(px, py - 204) < 30) continue;
        dots.push({
          x: px,
          y: py,
          role: "shoulder_contour",
          isSecondary: false,
          size: 0.64,
          row: pr,
          u: side * (px / Math.max(1, maxW)),
          phaseOff: pr * 0.38 + pc * 0.31 + side,
        });
      }
    }
  });

  // 4. Body Circuit-Board Microchip Arrays (Symmetric 3D IC Chip Tiny-Dot Matrices on Chest & Shoulders)
  [-1, 1].forEach((side) => {
    const chips = [
      { cx: side * 78, cy: 176, cols: 5, rows: 3, step: 5.2, isAccent: false }, // Pectoral IC Processor
      { cx: side * 124, cy: 184, cols: 3, rows: 3, step: 5.2, isAccent: true },  // Deltoid Power Module
      { cx: side * 66, cy: 232, cols: 5, rows: 2, step: 5.2, isAccent: false },  // Lower Chest Bus Chip
    ];
    chips.forEach((chip, cIdx) => {
      for (let r = 0; r < chip.rows; r++) {
        for (let c = 0; c < chip.cols; c++) {
          const px = chip.cx + (c - (chip.cols - 1) * 0.5) * chip.step;
          const py = chip.cy + (r - (chip.rows - 1) * 0.5) * chip.step;
          if (Math.abs(px) > getAnatomicalTorsoHalfW(py) - 6) continue;
          const isEdge = r === 0 || r === chip.rows - 1 || c === 0 || c === chip.cols - 1;
          dots.push({
            x: px,
            y: py,
            role: chip.isAccent && !isEdge ? "throat_core" : "shoulder_contour",
            isSecondary: chip.isAccent && !isEdge,
            size: isEdge ? 0.85 : 1.08,
            row: r,
            u: side * 0.5,
            phaseOff: cIdx * 0.9 + r * 0.4 + c * 0.3,
          });
        }
      }
    });
  });

  // 5. Orthogonal & 45-Degree Neck-to-Collarbone Circuit Bus Tiny-Dot Matrix
  [-1, 1].forEach((side) => {
    const neckTracks = [
      [[14, 96], [14, 134], [36, 156], [62, 156]],
      [[24, 88], [24, 120], [48, 144], [92, 144], [116, 168]],
      [[32, 98], [32, 114], [58, 130], [110, 130], [136, 156]],
    ];
    neckTracks.forEach((track, tIdx) => {
      for (let s = 0; s < track.length - 1; s++) {
        const [x0, y0] = track[s];
        const [x1, y1] = track[s + 1];
        const segLen = Math.hypot(x1 - x0, y1 - y0);
        const count = Math.max(3, Math.round(segLen / 5.5));
        for (let k = 0; k <= count; k++) {
          const frac = k / count;
          const px = side * (x0 + (x1 - x0) * frac);
          const py = y0 + (y1 - y0) * frac;
          if (py < 100 && Math.abs(px) < getAnatomicalHeadHalfW(py) + 2) continue;
          if (Math.abs(px) > getAnatomicalTorsoHalfW(py) - 5) continue;
          const isBendNode = k === 0 || k === count;
          dots.push({
            x: px,
            y: py,
            role: "shoulder_contour",
            isSecondary: false,
            size: isBendNode ? 1.05 : 0.72,
            row: tIdx,
            u: side * frac,
            phaseOff: tIdx * 0.7 + k * 0.35,
          });
        }
      }
    });
  });

  // 6. Throat Vertical Circuit Conduit Dots in pure #FF5C0A (orthogonal circuit bus up the neck)
  const throatSteps = 12;
  for (let s = 0; s <= throatSteps; s++) {
    const t = s / throatSteps; // 0 at Arc Reactor top (dy=184), 1 under chin (dy=100)
    const dy = 184 - t * 84;
    const stepX = t > 0.55 ? 11.0 : 5.5;
    [-1, 1].forEach((side) => {
      dots.push({
        x: side * stepX,
        y: dy,
        role: "throat_core",
        group: `throat_${side}`,
        isSecondary: true,
        throatT: t,
        size: 1.35 + (s % 3 === 0 ? 0.4 : 0),
        row: s,
        u: side * 0.2,
        phaseOff: s * 0.48 + side * 0.35,
      });
    });
  }

  state.avatarDots = dots;
}

/**
 * Defines symmetric 3D PCB / Circuit-Board pathways (strictly 0°, 90°, and 45° angles)
 * across the chest, collarbones, neck, and shoulders radiating from the central Arc Reactor (0, 204).
 */
function initBodyCircuits() {
  const halfTraces = [
    // Upper Collarbone 45°/90° Bus -> Outer Deltoid Shoulder
    { pts: [[16, 188], [16, 162], [38, 140], [108, 140], [138, 170], [138, 220]], isAccent: false, speed: 0.44 },
    // Mid Pectoral Power Bus -> Deltoid IC Module
    { pts: [[24, 200], [52, 200], [72, 180], [116, 180], [144, 208], [144, 246]], isAccent: true, speed: 0.56 },
    // Lower Chest Orthogonal Rail -> Outer Bicep Bus
    { pts: [[22, 214], [48, 214], [64, 230], [110, 230], [126, 246]], isAccent: false, speed: 0.38 },
    // Sternum -> Vertical Neck Data Bus -> Jaw Base
    { pts: [[8, 182], [8, 144], [20, 132], [20, 96]], isAccent: true, speed: 0.62 },
    // Upper Shoulder Trace -> Clavicle Node
    { pts: [[22, 192], [44, 170], [74, 170], [96, 148], [124, 148]], isAccent: false, speed: 0.48 },
    // Lower Torso Vertical Bus
    { pts: [[14, 226], [34, 246], [88, 246], [104, 230]], isAccent: false, speed: 0.42 },
  ];

  const circuits = [];
  [-1, 1].forEach((side) => {
    halfTraces.forEach((tr, idx) => {
      const mirroredPts = tr.pts.map(([x, y]) => [side * x, y]);
      const segLens = [];
      let totalLen = 0;
      for (let i = 0; i < mirroredPts.length - 1; i++) {
        const dx = mirroredPts[i + 1][0] - mirroredPts[i][0];
        const dy = mirroredPts[i + 1][1] - mirroredPts[i][1];
        const len = Math.hypot(dx, dy);
        segLens.push(len);
        totalLen += len;
      }

      // Pre-sample dense pinpoint dots along every circuit path (ZERO lines!)
      const traceDots = [];
      for (let i = 0; i < mirroredPts.length - 1; i++) {
        const [x0, y0] = mirroredPts[i];
        const [x1, y1] = mirroredPts[i + 1];
        const count = Math.max(2, Math.round(segLens[i] / 6.5));
        for (let k = 0; k <= count; k++) {
          const f = k / count;
          traceDots.push({
            x: x0 + (x1 - x0) * f,
            y: y0 + (y1 - y0) * f,
            isNode: k === 0 || k === count,
          });
        }
      }

      circuits.push({
        pts: mirroredPts,
        traceDots,
        segLens,
        totalLen,
        isAccent: tr.isAccent,
        speed: tr.speed,
        phaseOffset: (idx * 0.19 + (side > 0 ? 0.08 : 0)) % 1.0,
      });
    });
  });

  state.bodyCircuits = circuits;
}

/**
 * Renders the 3D Pure-Dot Circuit Body & Pure-Dot Rotating Arc Reactor (ZERO lines/strokes!).
 */
function drawIronManBodyCircuitsAndArcReactor(ctx, rotY, rotX, rms, bass, mid, high) {
  ctx.save();

  const projBody = (x, y, zOffset = 0) => {
    const zSurf = computeAvatarSurfaceZ(x, y, false) + zOffset;
    return project3DPoint(x, y, zSurf, rotY, rotX);
  };

  // --- 1. 3D PURE-DOT CIRCUIT TRACES, VIA NODES & MOVING SIGNAL PULSE DOTS ---
  for (let c = 0; c < state.bodyCircuits.length; c++) {
    const tr = state.bodyCircuits[c];
    const pts = tr.pts;

    // A. Static dotted circuit pathway & junction via pads (ZERO stroke lines!)
    for (let dIdx = 0; dIdx < tr.traceDots.length; dIdx++) {
      const td = tr.traceDots[dIdx];
      const pp = projBody(td.x, td.y, 2);
      const r = (td.isNode ? 1.7 : 0.95) * pp.scale;
      ctx.beginPath();
      ctx.arc(pp.sx, pp.sy, r, 0, Math.PI * 2);
      ctx.fillStyle = tr.isAccent
        ? `rgba(255, 92, 10, ${(td.isNode ? 0.62 : 0.34) + rms * 0.30})`
        : `rgba(42, 172, 255, ${(td.isNode ? 0.62 : 0.34) + rms * 0.28})`;
      ctx.fill();
    }

    // B. Two moving multi-dot signal pulse trains racing along each 3D circuit pathway
    const sampleAtDist = (dist) => {
      let acc = 0;
      for (let s = 0; s < tr.segLens.length; s++) {
        const sl = tr.segLens[s];
        if (dist <= acc + sl || s === tr.segLens.length - 1) {
          const frac = sl > 0 ? Math.max(0, Math.min(1, (dist - acc) / sl)) : 0;
          return [
            pts[s][0] + (pts[s + 1][0] - pts[s][0]) * frac,
            pts[s][1] + (pts[s + 1][1] - pts[s][1]) * frac,
          ];
        }
        acc += sl;
      }
      return pts[0];
    };

    for (let p = 0; p < 2; p++) {
      const progress =
        (state.phase * 0.088 * tr.speed * (1 + rms * 2.5) + tr.phaseOffset + p * 0.5) % 1.0;
      const headDist = progress * tr.totalLen;

      // Render a 4-dot cascading pulse train (no lines!)
      for (let trail = 0; trail < 4; trail++) {
        const dPos = Math.max(0, headDist - trail * 5.0);
        const [px, py] = sampleAtDist(dPos);
        const pp = projBody(px, py, 4);
        const fade = 1.0 - trail * 0.22;
        const r = (1.75 - trail * 0.28 + rms * 0.95) * pp.scale;

        ctx.beginPath();
        ctx.arc(pp.sx, pp.sy, Math.max(0.6, r), 0, Math.PI * 2);
        ctx.fillStyle = tr.isAccent
          ? `rgba(255, ${100 + (trail === 0 ? 70 : 0)}, ${18 + (trail === 0 ? 50 : 0)}, ${fade * (0.85 + rms * 0.15)})`
          : `rgba(${130 + (trail === 0 ? 110 : 0)}, 238, 255, ${fade * (0.85 + rms * 0.15)})`;
        ctx.shadowColor = tr.isAccent ? "#FF5C0A" : "#38d6ff";
        ctx.shadowBlur = (trail === 0 ? 9 : 4) + rms * 6;
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
  }

  // --- 2. 3D PURE-DOT ROTATING ARC REACTOR CORE AT (0, 204, z=+54) ---
  const reactorPulse = 1.0 + rms * 0.14 + bass * 0.08;
  const arcCenter = projBody(0, 204, 6);
  const rx = arcCenter.sx;
  const ry = arcCenter.sy;
  const rScale = arcCenter.scale * reactorPulse;

  // Soft radial cyan reactor bloom
  const arcGlow = ctx.createRadialGradient(rx, ry, 2, rx, ry, 32 * rScale);
  arcGlow.addColorStop(0, `rgba(200, 248, 255, ${0.58 + rms * 0.35})`);
  arcGlow.addColorStop(0.45, `rgba(38, 185, 255, ${0.32 + rms * 0.32})`);
  arcGlow.addColorStop(1, "rgba(6, 46, 82, 0)");
  ctx.fillStyle = arcGlow;
  ctx.beginPath();
  ctx.arc(rx, ry, 34 * rScale, 0, Math.PI * 2);
  ctx.fill();

  // Concentric rotating 3D dot rings (ZERO stroked circles!)
  const cosYaw = Math.max(0.55, Math.cos(rotY));
  const rotAngle = state.phase * 0.42;

  // Outer ring: 20 rotating coil dots
  for (let k = 0; k < 20; k++) {
    const a = rotAngle + (k / 20) * Math.PI * 2;
    const dx = Math.cos(a) * 18.5 * rScale * cosYaw;
    const dy = Math.sin(a) * 18.5 * rScale;
    const isMajor = k % 2 === 0;
    ctx.beginPath();
    ctx.arc(rx + dx, ry + dy, (isMajor ? 1.8 : 1.15) * rScale, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(135, 235, 255, ${(isMajor ? 0.92 : 0.65) + rms * 0.15})`;
    ctx.shadowColor = "#38d6ff";
    ctx.shadowBlur = 7 + rms * 6;
    ctx.fill();
  }

  // Inner counter-rotating ring: 10 precision dots
  for (let k = 0; k < 10; k++) {
    const a = -rotAngle * 1.4 + (k / 10) * Math.PI * 2;
    const dx = Math.cos(a) * 10.5 * rScale * cosYaw;
    const dy = Math.sin(a) * 10.5 * rScale;
    ctx.beginPath();
    ctx.arc(rx + dx, ry + dy, 1.35 * rScale, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(220, 252, 255, ${0.88 + rms * 0.12})`;
    ctx.fill();
  }

  // Center core dot
  ctx.beginPath();
  ctx.arc(rx, ry, 4.2 * rScale, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(240, 253, 255, ${0.92 + rms * 0.08})`;
  ctx.shadowColor = "#7ce8ff";
  ctx.shadowBlur = 12 + rms * 8;
  ctx.fill();

  ctx.shadowBlur = 0;
  ctx.restore();
}

function updateTimelineSync(currentTime, isPlaying) {
  let activeSeg = null;
  let pausingSeg = null;

  state.timeline.forEach((seg, idx) => {
    const chunkEl = document.getElementById(`k-chunk-${idx}`);
    const pauseEl = document.getElementById(`k-pause-${idx}`);

    const inSpeech = isPlaying && currentTime >= seg.speech_start && currentTime <= seg.speech_end;
    const inPause =
      isPlaying &&
      seg.pause_duration > 0 &&
      currentTime > seg.speech_end &&
      currentTime <= seg.pause_end;

    if (chunkEl) chunkEl.classList.toggle("speaking", inSpeech);
    if (pauseEl) pauseEl.classList.toggle("pausing", inPause);

    if (inSpeech) activeSeg = seg;
    if (inPause) pausingSeg = seg;
  });

  state.activeChunk = activeSeg;
  state.activePause = pausingSeg;

  const hudEmo = document.getElementById("hudEmotionBadge");
  if (hudEmo) {
    if (activeSeg) {
      hudEmo.textContent = `${activeSeg.emotion_icon} ${activeSeg.emotion_label}`;
    } else if (pausingSeg) {
      hudEmo.textContent = `⏸️ ${pausingSeg.pause_label}`;
    }
  }
}

function startCanvasLoop() {
  const canvas = document.getElementById("speechCanvas");
  const ctx = canvas.getContext("2d");
  const player = document.getElementById("mainAudioPlayer");
  let lastNow = performance.now();

  function renderFrame(now) {
    requestAnimationFrame(renderFrame);

    // Frame-rate independent delta time (seconds) for ultra-smooth, exact timing
    const dt = Math.min(0.05, Math.max(0.004, (now - lastNow) / 1000.0));
    lastNow = now;

    const W = canvas.width;
    const H = canvas.height;
    const isPlaying = player && !player.paused && !player.ended;

    // 1. Update timeline & punctuation pause state
    updateTimelineSync(player ? player.currentTime : 0, isPlaying);
    const inPunctuationPause = Boolean(state.activePause);

    // 1b. Update in-avatar synthesis progress state & smooth interpolation
    if (state.isSynthesizing) {
      const elapsedSec = (performance.now() - state.synthStartMs) / 1000;
      if (state.synthProgress < 95) {
        const target = Math.min(94, 6 + 88 * (1 - Math.exp(-elapsedSec / 2.4)));
        state.synthProgress = Math.max(state.synthProgress, target);
      }
      if (state.synthProgress < 25) {
        state.synthStageLabel = "Parsing punctuation & feelings...";
      } else if (state.synthProgress < 60) {
        state.synthStageLabel = "Synthesizing Kokoro neural voice...";
      } else if (state.synthProgress < 88) {
        state.synthStageLabel = "Applying prosody & speaker identity lock...";
      } else if (state.synthProgress < 100) {
        state.synthStageLabel = "Finalizing 24 kHz studio waveform...";
      }
      const pct = Math.round(state.synthDisplayProgress);
      setEngineStatus("synthesizing", `Synthesizing • ${pct}%`, state.synthStageLabel);
      const metricsBadge = document.getElementById("hudMetricsBadge");
      if (metricsBadge) {
        metricsBadge.textContent = `Synthesizing: ${pct}% • ${state.synthStageLabel}`;
      }
    } else if (isPlaying) {
      setEngineStatus("speaking", "Speaking Live", "Real-time 60FPS holographic audio sync");
    } else if (state.synthFadeAlpha < 0.02) {
      setEngineStatus("ready", "CODERSTATION Studio", "Neural Voice Technology");
    }

    const fadeTarget = state.isSynthesizing ? 1.0 : 0.0;
    state.synthFadeAlpha += (fadeTarget - state.synthFadeAlpha) * Math.min(1, dt * 7.5);
    const progSpeed = state.synthProgress >= 100 ? 14.0 : 6.0;
    state.synthDisplayProgress += (state.synthProgress - state.synthDisplayProgress) * Math.min(1, dt * progSpeed);

    // 2. Read real-time vocal formant & amplitude energy
    let rms = 0;
    let bass = 0;
    let mid = 0;
    let high = 0;

    if (state.isSynthesizing) {
      // Gentle energetic pulse inside the avatar while synthesizing
      rms = 0.14 + 0.06 * Math.sin(state.phase * 2.6);
      bass = 0.18 + 0.08 * Math.sin(state.phase * 1.9);
      mid = 0.22 + 0.08 * Math.cos(state.phase * 2.3);
      high = 0.20 + 0.10 * Math.sin(state.phase * 3.1);
    } else if (state.analyser && isPlaying && !inPunctuationPause) {
      state.analyser.getByteFrequencyData(state.freqData);
      state.analyser.getByteTimeDomainData(state.timeData);

      let sumSq = 0;
      for (let i = 0; i < state.timeData.length; i++) {
        const norm = (state.timeData[i] - 128) / 128.0;
        sumSq += norm * norm;
      }
      rms = Math.sqrt(sumSq / state.timeData.length);
      // Clean noise gate so silent gaps between syllables stay completely calm
      if (rms < 0.012) rms = 0;

      const bins = state.freqData;
      for (let i = 2; i < 16; i++) bass += bins[i];
      for (let i = 16; i < 75; i++) mid += bins[i];
      for (let i = 75; i < 190; i++) high += bins[i];

      bass /= 14 * 255;
      mid /= 59 * 255;
      high /= 115 * 255;
    }

    // Asymmetric Attack / Release Envelope Follower (frame-rate independent):
    // Fast attack (~45ms) locks right onto syllable onset; smooth release (~160ms) glides gently into pauses
    const smoothFollow = (cur, target, atkTau, relTau) => {
      const tau = target > cur ? atkTau : relTau;
      const alpha = 1.0 - Math.exp(-dt / tau);
      return cur + (target - cur) * alpha;
    };

    state.smoothRms = smoothFollow(state.smoothRms, rms, 0.045, 0.16);
    state.smoothBass = smoothFollow(state.smoothBass, bass, 0.055, 0.18);
    state.smoothMid = smoothFollow(state.smoothMid, mid, 0.045, 0.15);
    state.smoothHigh = smoothFollow(state.smoothHigh, high, 0.050, 0.16);

    const sRms = state.smoothRms;
    const sBass = state.smoothBass;
    const sMid = state.smoothMid;
    const sHigh = state.smoothHigh;

    if (isPlaying) {
      const metricsBadge = document.getElementById("hudMetricsBadge");
      if (metricsBadge) {
        const estFreq = Math.round(115 + sMid * 245 + sHigh * 180);
        metricsBadge.textContent =
          `Vocal Energy: ${Math.min(100, Math.round(sRms * 265))}% • Formant Activity: ~${estFreq} Hz`;
      }
    }

    // Smooth, steady time-continuous phase clock (no abrupt stuttering or speed jumps)
    state.phase += dt * (1.75 + sRms * 2.8);

    // 3. Render 3D Environment Background, 3D Pure-Dot Circuit Avatar & Synthesis Progress Overlay
    drawBalancedSignalWaveBackground(ctx, W, H, sRms, sBass, sMid, sHigh, isPlaying);
    drawAnimatedDotAvatar(ctx, W, H, sRms, sBass, sMid, sHigh, isPlaying);
    if (state.synthFadeAlpha > 0.01) {
      drawAvatarSynthProgressOverlay(ctx, W, H);
    }
  }

  requestAnimationFrame(renderFrame);
}

/**
 * Renders a sleek glassmorphic pure-dot holographic synthesis progress HUD
 * directly inside the 3D avatar animation stage (zero gradients).
 */
function drawAvatarSynthProgressOverlay(ctx, W, H) {
  const alpha = state.synthFadeAlpha;
  const pct = Math.max(0, Math.min(100, state.synthDisplayProgress));
  const frac = pct / 100;

  ctx.save();
  ctx.globalAlpha = alpha;

  // 1. Subtle holographic vertical scan dot-ring sweeping across the avatar
  const scanY = H * (0.14 + 0.62 * (0.5 + 0.5 * Math.sin(state.phase * 1.85)));
  const scanSpan = W * 0.26;
  const numScanDots = 28;
  for (let i = 0; i < numScanDots; i++) {
    const t = i / (numScanDots - 1);
    const sx = W * 0.5 + (t - 0.5) * scanSpan * 2;
    const edgeFade = Math.sin(t * Math.PI);
    ctx.beginPath();
    ctx.arc(sx, scanY, 1.3 + edgeFade * 1.2, 0, Math.PI * 2);
    ctx.fillStyle = i % 4 === 0
      ? `rgba(255, 92, 10, ${0.75 * edgeFade})`
      : `rgba(110, 232, 255, ${0.65 * edgeFade})`;
    ctx.fill();
  }

  // 2. Frosted Glass Progress Bar HUD at bottom-center of the avatar stage
  const boxW = Math.min(440, W * 0.78);
  const boxH = 54;
  const boxX = (W - boxW) * 0.5;
  const boxY = H - boxH - 16;

  // Rounded glass pill container
  const rad = 13;
  ctx.beginPath();
  ctx.moveTo(boxX + rad, boxY);
  ctx.lineTo(boxX + boxW - rad, boxY);
  ctx.quadraticCurveTo(boxX + boxW, boxY, boxX + boxW, boxY + rad);
  ctx.lineTo(boxX + boxW, boxY + boxH - rad);
  ctx.quadraticCurveTo(boxX + boxW, boxY + boxH, boxX + boxW - rad, boxY + boxH);
  ctx.lineTo(boxX + rad, boxY + boxH);
  ctx.quadraticCurveTo(boxX, boxY + boxH, boxX, boxY + boxH - rad);
  ctx.lineTo(boxX, boxY + rad);
  ctx.quadraticCurveTo(boxX, boxY, boxX + rad, boxY);
  ctx.closePath();

  ctx.fillStyle = "rgba(4, 24, 46, 0.76)";
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
  ctx.stroke();

  // Top label row: stage description (left) + percentage (right)
  ctx.font = "600 11.5px 'Segoe UI', Inter, system-ui, sans-serif";
  ctx.fillStyle = "#e2e8f0";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(state.synthStageLabel || "Synthesizing speech...", boxX + 16, boxY + 18);

  ctx.font = "700 12px 'Segoe UI', Inter, system-ui, sans-serif";
  ctx.fillStyle = "#FF5C0A";
  ctx.textAlign = "right";
  ctx.fillText(`${Math.round(pct)}%`, boxX + boxW - 16, boxY + 18);

  // Pure-dot holographic LED progress bar (36 glowing dots)
  const dotCount = 36;
  const barLeft = boxX + 18;
  const barRight = boxX + boxW - 18;
  const barY = boxY + 38;
  const filledCount = Math.round(frac * dotCount);

  for (let i = 0; i < dotCount; i++) {
    const dx = barLeft + (i / (dotCount - 1)) * (barRight - barLeft);
    const isFilled = i < filledCount;
    const isHead = i === filledCount - 1 && pct > 0 && pct < 100;

    ctx.beginPath();
    ctx.arc(dx, barY, isHead ? 3.2 : isFilled ? 2.2 : 1.5, 0, Math.PI * 2);

    if (isHead) {
      ctx.fillStyle = "#FF5C0A";
      ctx.shadowColor = "#FF5C0A";
      ctx.shadowBlur = 10;
    } else if (isFilled) {
      ctx.fillStyle = i > dotCount * 0.72 ? "rgba(255, 92, 10, 0.92)" : "rgba(56, 210, 255, 0.92)";
      ctx.shadowColor = i > dotCount * 0.72 ? "#FF5C0A" : "#38bdf8";
      ctx.shadowBlur = 6;
    } else {
      ctx.fillStyle = "rgba(148, 184, 215, 0.24)";
      ctx.shadowBlur = 0;
    }
    ctx.fill();
  }

  ctx.shadowBlur = 0;
  ctx.restore();
}

/**
 * Initializes 3D ambient volumetric depth motes in the holographic chamber environment.
 */
function init3DEnvironmentMotes() {
  const motes = [];
  for (let i = 0; i < 72; i++) {
    motes.push({
      x: (Math.random() - 0.5) * 980,
      y: (Math.random() - 0.5) * 520,
      z: -80 + Math.random() * 680,
      speedZ: 0.45 + Math.random() * 1.15,
      size: 0.8 + Math.random() * 1.6,
      isAccent: Math.random() < 0.22,
      phaseOff: Math.random() * Math.PI * 2,
    });
  }
  state.envMotes3D = motes;
}

/**
 * 3D Rotation (Yaw around Y, Pitch around X) + Perspective Projection helper
 * Camera sits at z = -FOV looking toward +Z (positive z = closer to viewer).
 */
function project3DPoint(x, y, z, rotY, rotX, fov = 560) {
  // Rotate around Y axis (Yaw: left/right 3D turn)
  const cosY = Math.cos(rotY);
  const sinY = Math.sin(rotY);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;

  // Rotate around X axis (Pitch: gentle up/down 3D nod)
  const cosX = Math.cos(rotX);
  const sinX = Math.sin(rotX);
  const y2 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;

  const persp = fov / Math.max(120, fov - z2);
  return {
    sx: x1 * persp,
    sy: y2 * persp,
    z: z2,
    scale: persp,
  };
}

/**
 * Computes true 3D surface depth `z` (front = positive, sides = 0) for any point (x, y)
 * on the Iron Man head/helmet or upper torso/shoulders.
 */
function computeAvatarSurfaceZ(x, y, isHead) {
  if (isHead) {
    const hw = Math.max(12, getAnatomicalHeadHalfW(y));
    const nx = Math.max(-1, Math.min(1, x / hw));
    const ny = Math.max(-1, Math.min(1, (y + 13) / 112));
    const zBase = Math.sqrt(Math.max(0, 1 - nx * nx * 0.90 - ny * ny * 0.30)) * 78;
    // Sculpt smooth 3D cranial dome, cheekbones, nose bridge & chin plane (no eyes/mouth/V-jaw)
    const noseBump = Math.exp(-Math.pow(nx / 0.16, 2)) * Math.exp(-Math.pow((ny - 0.05) / 0.24, 2)) * 15;
    const cheekBump = Math.exp(-Math.pow((Math.abs(nx) - 0.50) / 0.26, 2)) * Math.exp(-Math.pow((ny - 0.14) / 0.28, 2)) * 9;
    const chinBump = Math.exp(-Math.pow(nx / 0.30, 2)) * Math.exp(-Math.pow((ny - 0.72) / 0.22, 2)) * 11;
    return zBase + noseBump + cheekBump + chinBump;
  } else {
    const hw = Math.max(20, getAnatomicalTorsoHalfW(y));
    const nx = Math.max(-1, Math.min(1, x / hw));
    // Sculpted 3D torso curvature: cylindrical neck, prominent clavicles, rounded pectorals & deltoid caps
    const chestCurve = Math.sqrt(Math.max(0, 1 - nx * nx * 0.86)) * 60;
    const clavicleBump = Math.exp(-Math.pow((y - 126) / 14, 2)) * Math.exp(-Math.pow((Math.abs(nx) - 0.42) / 0.35, 2)) * 9;
    const pectoralBump = Math.exp(-Math.pow((y - 184) / 34, 2)) * Math.exp(-Math.pow((Math.abs(nx) - 0.36) / 0.32, 2)) * 11;
    const deltoidBump = Math.exp(-Math.pow((y - 168) / 36, 2)) * Math.exp(-Math.pow((Math.abs(nx) - 0.84) / 0.18, 2)) * 10;
    return chestCurve + clavicleBump + pectoralBump + deltoidBump - 6;
  }
}

/**
 * Renders the 3D Holographic Chamber Environment Background:
 * 1. Deep #062E52 volumetric studio backdrop
 * 2. Receding 3D perspective holographic floor grid & horizon rails
 * 3. 3D tilted holographic orbital HUD rings around the avatar
 * 4. 3D volumetric depth motes & perspective moving wave-like signal ribbons
 */
function drawBalancedSignalWaveBackground(ctx, W, H, rms, bass, mid, high, isPlaying) {
  ctx.save();

  const vpX = W * 0.5;
  const vpY = H * 0.44;
  const fov = 440;

  // 1. Deep Primary Color (#062E52) 3D Volumetric Chamber Backdrop
  const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
  bgGrad.addColorStop(0, "#020c18");
  bgGrad.addColorStop(0.45, "#062E52");
  bgGrad.addColorStop(0.82, "#041c34");
  bgGrad.addColorStop(1, "#020b16");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  // Volumetric Horizon Spotlight behind Avatar
  const horizonGlow = ctx.createRadialGradient(vpX, vpY, 16, vpX, vpY, W * 0.52);
  horizonGlow.addColorStop(0, `rgba(28, 145, 245, ${0.24 + rms * 0.18})`);
  horizonGlow.addColorStop(0.45, `rgba(6, 46, 82, ${0.38 + bass * 0.12})`);
  horizonGlow.addColorStop(1, "rgba(2, 11, 22, 0)");
  ctx.fillStyle = horizonGlow;
  ctx.fillRect(0, 0, W, H);

  // 2. 3D Perspective Holographic Floor Grid (receding from foreground to vanishing horizon)
  const floorY = 175;
  const minZ = 40;
  const maxZ = 680;
  const zScroll = (state.phase * 18) % 55;

  // Perspective longitudinal rails (converging toward vanishing point)
  for (let gx = -9; gx <= 9; gx++) {
    const worldX = gx * 78;
    const pNear = fov / (fov + minZ);
    const pFar = fov / (fov + maxZ);
    const x1 = vpX + worldX * pNear;
    const y1 = vpY + floorY * pNear;
    const x2 = vpX + worldX * pFar;
    const y2 = vpY + floorY * pFar;

    const isCenterTrack = Math.abs(gx) === 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = isCenterTrack
      ? `rgba(255, 92, 10, ${0.18 + rms * 0.18})`
      : `rgba(36, 155, 245, ${0.14 + rms * 0.10})`;
    ctx.lineWidth = isCenterTrack ? 1.3 : 0.9;
    ctx.stroke();
  }

  // Perspective transverse depth lines scrolling smoothly toward camera
  for (let gz = 0; gz < 12; gz++) {
    const worldZ = minZ + gz * 55 - zScroll;
    if (worldZ < minZ * 0.5) continue;
    const depthFade = Math.max(0, 1 - (worldZ - minZ) / (maxZ - minZ));
    const p = fov / (fov + worldZ);
    const y = vpY + floorY * p;
    const halfSpan = 720 * p;

    ctx.beginPath();
    ctx.moveTo(vpX - halfSpan, y);
    ctx.lineTo(vpX + halfSpan, y);
    ctx.strokeStyle = `rgba(36, 165, 255, ${depthFade * (0.24 + rms * 0.16)})`;
    ctx.lineWidth = 0.7 + depthFade * 0.7;
    ctx.stroke();
  }

  // 3. 3D Volumetric Floating Depth Motes (parallax depth drift through the 3D chamber)
  for (let i = 0; i < state.envMotes3D.length; i++) {
    const m = state.envMotes3D[i];
    m.z -= m.speedZ * (1 + rms * 2.2);
    if (m.z < -60) {
      m.z = 620;
      m.x = (Math.random() - 0.5) * 980;
      m.y = (Math.random() - 0.5) * 520;
    }
    const p = fov / Math.max(80, fov + m.z);
    const sx = vpX + m.x * p;
    const sy = vpY + m.y * p;
    if (sx < 0 || sx > W || sy < 0 || sy > H) continue;

    const depthAlpha = Math.max(0.05, Math.min(0.85, (1 - m.z / 650) * (0.55 + rms * 0.4)));
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(0.6, m.size * p * 1.4), 0, Math.PI * 2);
    ctx.fillStyle = m.isAccent
      ? `rgba(255, 92, 10, ${depthAlpha})`
      : `rgba(80, 200, 255, ${depthAlpha})`;
    ctx.fill();
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// Larger Human-Form Dot-Constellation Avatar with Realistic Shoulders
// (All Dots Animated, Subtle Constellation Filaments, Zero Yellow!)
// ---------------------------------------------------------------------------
function drawAnimatedDotAvatar(ctx, W, H, rms, bass, mid, high, isPlaying) {
  ctx.save();

  // Position head center and apply live 3D Yaw (left/right turn) & Pitch (speaking nod)
  const avatarScale = 1.12 + Math.sin(state.phase * 0.45) * 0.006 + rms * 0.022;
  const headX = W * 0.5;
  const headY = H * 0.355 + Math.sin(state.phase * 0.45) * 1.6;

  // Lifelike 3D holographic rotation (gentle 3D yaw +-15 deg & pitch +-5 deg)
  const rotY = Math.sin(state.phase * 0.28) * (0.24 + rms * 0.08);
  const rotX = Math.sin(state.phase * 0.55) * 0.06 + (isPlaying ? Math.sin(state.phase * 1.6) * rms * 0.09 : 0);

  ctx.translate(headX, headY);
  ctx.scale(avatarScale, avatarScale);

  // 1. 3D Perspective-Projected Iron Man Body Circuits & Rotating Arc Reactor
  drawIronManBodyCircuitsAndArcReactor(ctx, rotY * 0.65, rotX * 0.5, rms, bass, mid, high);

  // Soft 3D Electric-Blue Vocal Resonance Glow inside the Iron Man Helmet during speech
  const headGlowCenter = project3DPoint(0, 6, 40, rotY, rotX);
  const headGlowAlpha = 0.08 + rms * 0.42 + mid * 0.22;
  const headGlow = ctx.createRadialGradient(
    headGlowCenter.sx,
    headGlowCenter.sy,
    4,
    headGlowCenter.sx,
    headGlowCenter.sy,
    76 + rms * 28
  );
  headGlow.addColorStop(0, `rgba(110, 220, 255, ${headGlowAlpha})`);
  headGlow.addColorStop(0.5, `rgba(24, 145, 245, ${headGlowAlpha * 0.48})`);
  headGlow.addColorStop(1, "rgba(6, 46, 82, 0)");
  ctx.fillStyle = headGlow;
  ctx.beginPath();
  ctx.arc(headGlowCenter.sx, headGlowCenter.sy, 98 + rms * 22, 0, Math.PI * 2);
  ctx.fill();

  // 2. Transform All Dots in True 3D (x, y, z), Depth-Sort Back-to-Front, and Render!
  //    Uses smooth syllable-timed harmonic waves (zero raw PCM jitter!)
  const projectedDots = [];

  for (let i = 0; i < state.avatarDots.length; i++) {
    const d = state.avatarDots[i];

    const isHeadRole =
      d.role === "face_core" ||
      d.role === "face_contour" ||
      d.role === "rim";

    let dx = Math.sin(state.phase * 1.15 + d.phaseOff) * (0.75 + rms * 1.8);
    let dy = Math.cos(state.phase * 0.95 + d.phaseOff * 0.9) * (0.75 + rms * 1.6);
    let dz = 0;
    let dotRadius = d.size;
    let speechFlash = 0;

    if (d.role === "face_core" || d.role === "face_contour") {
      const distFromCore = Math.hypot(d.x, d.y - 12);
      // Smooth radial sonic ripple propagating outward in sync with each spoken syllable
      const radialWave = Math.sin(distFromCore * 0.078 - state.phase * 3.4);
      const wavePush = radialWave * (rms * 5.2 + mid * 3.2);

      const normAngle = Math.atan2(d.y - 12, d.x);
      dx += Math.cos(normAngle) * wavePush;
      dy += Math.sin(normAngle) * wavePush;
      dz += radialWave * (rms * 12 + mid * 7);

      if (d.role === "face_core") {
        const w = d.coreWeight || 0.5;
        dx += Math.sin(state.phase * 2.8 + d.row * 0.65) * (rms * 5.5 + mid * 3.5) * w;
        dy += Math.sin(d.u * 6.5 - state.phase * 3.2 + d.row * 0.6) * (0.9 + rms * 7.2 + mid * 4.5) * w;
        speechFlash = Math.min(1.0, (rms * 2.6 + mid * 1.3 + Math.max(0, radialWave) * rms * 1.5) * (0.48 + 0.52 * w));
        dotRadius = d.size * (1.0 + (0.15 + rms * 0.75 + mid * 0.38) * w);
      } else {
        dy += Math.sin(d.u * 5.5 - state.phase * 2.0 + d.row * 0.45) * (0.8 + rms * 3.2);
        speechFlash = Math.min(1.0, Math.max(0, radialWave) * (rms * 2.1 + high * 0.75));
        dotRadius = d.size * (1.0 + rms * 0.38 + speechFlash * 0.28);
      }
    } else if (d.role === "rim") {
      const rimWave = Math.sin(d.row * 0.5 - state.phase * 3.0);
      const pulseOut = 1.0 + Math.sin(state.phase * 1.2 + d.phaseOff) * 0.011 + rms * 0.052 + rimWave * rms * 0.022;
      dx += d.x * (pulseOut - 1.0);
      dy += d.y * (pulseOut - 1.0) * 0.42;
      speechFlash = Math.min(1.0, rms * 1.85 + high * 0.8 + Math.max(0, rimWave) * rms * 1.1);
      dotRadius = d.size * (1.0 + rms * 0.48 + high * 0.22);
    } else if (d.role === "throat_core") {
      const travelWave = Math.sin(d.throatT * 9 - state.phase * 2.8);
      dx += travelWave * (1.2 + rms * 4.6 + bass * 2.8);
      dy -= ((state.phase * 9 * (1 + rms * 1.6) + d.row * 10) % 12) - 6;
      dz = 6;
      dotRadius = d.size * (1.0 + Math.max(0, travelWave) * (0.20 + rms * 0.60));
    } else if (d.role === "shoulder_rim") {
      const pulseOut = 1.0 + Math.sin(state.phase * 1.2 + d.phaseOff) * 0.010 + rms * 0.030;
      dx += d.x * (pulseOut - 1.0);
      dy += d.y * (pulseOut - 1.0) * 0.32;
      dotRadius = d.size * (1.0 + rms * 0.28 + high * 0.14);
    } else {
      dy += Math.sin(d.u * 5.2 - state.phase * 1.8 + d.row * 0.45) * (0.9 + rms * 3.0);
      dotRadius = d.size * (1.0 + rms * 0.24);
    }

    // Compute true 3D surface Z coordinate and project with perspective
    const baseZ = computeAvatarSurfaceZ(d.x, d.y, isHeadRole) + dz;
    // Head rotates slightly more freely than shoulders for lifelike 3D posture
    const p3 = project3DPoint(
      d.x + dx,
      d.y + dy,
      baseZ,
      isHeadRole ? rotY : rotY * 0.65,
      isHeadRole ? rotX : rotX * 0.5
    );

    projectedDots.push({
      px: p3.sx,
      py: p3.sy,
      z: p3.z,
      r: dotRadius * p3.scale,
      depthShade: Math.max(0.45, Math.min(1.15, (p3.z + 65) / 120)),
      speechFlash,
      d,
    });
  }

  // Z-buffer sort back-to-front for authentic 3D depth occlusion
  projectedDots.sort((a, b) => a.z - b.z);

  for (let i = 0; i < projectedDots.length; i++) {
    const { px, py, r, depthShade, speechFlash, d } = projectedDots[i];
    // Subtle per-dot micro-twinkle for delicate starlight / LED hologram sparkle
    const twinkle = 0.5 + 0.5 * Math.sin(state.phase * 2.4 + d.phaseOff * 3.1);
    const glowBoost = twinkle * (0.28 + rms * 0.72);

    // Layer 1: Tight delicate additive micro-glow halo around each tiny dot
    const haloR = r * (1.85 + glowBoost * 0.7 + speechFlash * 0.65);
    ctx.beginPath();
    ctx.arc(px, py, haloR, 0, Math.PI * 2);

    if (d.role === "throat_core") {
      ctx.fillStyle = `rgba(255, 92, 10, ${(0.14 + glowBoost * 0.22 + rms * 0.18) * depthShade})`;
      ctx.fill();

      // Main dot body (#FF5C0A)
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 92, 10, ${Math.min(1.0, (0.82 + rms * 0.18) * depthShade)})`;
      ctx.shadowColor = "#FF5C0A";
      ctx.shadowBlur = 5 + rms * 6 + glowBoost * 4;
      ctx.fill();

      // Tiny bright pinpoint center sparkle
      if (twinkle > 0.55 || rms > 0.08) {
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(px, py, r * 0.45, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 195, 145, ${(0.55 + glowBoost * 0.45) * depthShade})`;
        ctx.fill();
      }
    } else if (
      d.role === "face_core" ||
      d.role === "face_contour" ||
      d.role === "rim"
    ) {
      // Tight cool electric-blue / ice-cyan micro-halo
      ctx.fillStyle = `rgba(${Math.round(70 + speechFlash * 150)}, ${Math.round(
        195 + speechFlash * 60
      )}, 255, ${(0.12 + glowBoost * 0.18 + speechFlash * 0.24) * depthShade})`;
      ctx.fill();

      // Main dot body
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      const rCol = Math.min(255, Math.round((45 + speechFlash * 190 + glowBoost * 35) * depthShade));
      const gCol = Math.min(255, Math.round((170 + speechFlash * 85 + glowBoost * 30) * depthShade));
      const bCol = Math.min(255, Math.round(255 * Math.min(1, depthShade + 0.12)));
      const alpha = Math.min(1.0, ((d.role === "rim" ? 0.90 : 0.78) + speechFlash * 0.22) * depthShade);
      ctx.fillStyle = `rgba(${rCol}, ${gCol}, ${bCol}, ${alpha})`;
      ctx.shadowColor = speechFlash > 0.30 || twinkle > 0.75 ? "#a5ecff" : "#1e96ff";
      ctx.shadowBlur = ((d.role === "rim" ? 6 : 4) + speechFlash * 9 + glowBoost * 4) * depthShade;
      ctx.fill();

      // Tiny luminous pinpoint core highlight on sparkling/speaking head dots
      if (twinkle > 0.62 || speechFlash > 0.28) {
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(px, py, r * 0.48, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(235, 252, 255, ${Math.min(1.0, (0.45 + speechFlash * 0.55 + glowBoost * 0.35) * depthShade)})`;
        ctx.fill();
      }
    } else if (d.role === "shoulder_rim") {
      ctx.fillStyle = `rgba(75, 195, 255, ${(0.13 + glowBoost * 0.18) * depthShade})`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(75, 195, 255, ${Math.min(1.0, (0.88 + rms * 0.12) * depthShade)})`;
      ctx.shadowColor = "#1e96ff";
      ctx.shadowBlur = 6 + rms * 5 + glowBoost * 3;
      ctx.fill();

      if (twinkle > 0.68) {
        ctx.shadowBlur = 0;
        ctx.beginPath();
        ctx.arc(px, py, r * 0.45, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(220, 248, 255, ${0.65 * depthShade})`;
        ctx.fill();
      }
    } else {
      ctx.fillStyle = `rgba(42, 168, 255, ${(0.10 + glowBoost * 0.15) * depthShade})`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(38, 158, 252, ${Math.min(1.0, (0.76 + rms * 0.22) * depthShade)})`;
      ctx.shadowColor = "#0b5da8";
      ctx.shadowBlur = 3 + glowBoost * 3;
      ctx.fill();
    }
  }

  ctx.shadowBlur = 0;
  ctx.restore();
}
