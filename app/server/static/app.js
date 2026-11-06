/* HandsFree Control dashboard */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------- WebSocket ---------------- */
let ws = null;
let wsReady = false;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsReady = true; setChip("chip-conn", "connected", true); };
  ws.onclose = () => {
    wsReady = false; setChip("chip-conn", "reconnecting…", false);
    setTimeout(connect, 1500);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "state") onState(msg.data);
    else if (msg.type === "event") onEvent(msg.topic, msg.data);
  };
}
function send(obj) { if (wsReady) ws.send(JSON.stringify(obj)); }

/* ---------------- helpers ---------------- */
function setChip(id, text, ok) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("on", ok === true);
  el.classList.toggle("off", ok === false);
}
function toast(text) {
  const div = document.createElement("div");
  div.className = "toast";
  div.textContent = text;
  $("toasts").appendChild(div);
  setTimeout(() => div.remove(), 4200);
}
function logEvent(html) {
  const li = document.createElement("li");
  li.innerHTML = `<b>${new Date().toLocaleTimeString()}</b> ${html}`;
  const ul = $("event-log");
  ul.prepend(li);
  while (ul.children.length > 80) ul.lastChild.remove();
}

/* ---------------- charts ---------------- */
class Spark {
  constructor(canvas, color, max = null) {
    this.ctx = canvas.getContext("2d");
    this.w = canvas.width; this.h = canvas.height;
    this.color = color; this.max = max; this.data = [];
  }
  push(v) {
    if (v == null || isNaN(v)) return;
    this.data.push(v);
    if (this.data.length > 120) this.data.shift();
    this.draw();
  }
  draw() {
    const { ctx, w, h, data } = this;
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;
    const max = this.max || Math.max(...data, 1e-6) * 1.15;
    const pts = data.map((v, i) => [
      (i / 119) * w,
      h - Math.min(v / max, 1) * (h - 6) - 3,
    ]);
    // area fill under the trace
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.lineTo(pts[pts.length - 1][0], h); ctx.lineTo(pts[0][0], h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, this.color + "44"); grad.addColorStop(1, this.color + "00");
    ctx.fillStyle = grad; ctx.fill();
    // glowing trace line
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.shadowColor = this.color; ctx.shadowBlur = 6;
    ctx.strokeStyle = this.color; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.shadowBlur = 0;
    // live value readout
    ctx.fillStyle = this.color; ctx.font = "600 11px 'IBM Plex Mono', monospace";
    ctx.fillText(data[data.length - 1].toFixed(1), 5, 12);
  }
}
const chFps = new Spark($("ch-fps"), "#55e07f");
const chLat = new Spark($("ch-lat"), "#ffb347");
const chJit = new Spark($("ch-jit"), "#ff5d4d");
const chCpu = new Spark($("ch-cpu"), "#59d6e3", 100);

/* ---------------- selects & toggles ---------------- */
let populated = false;
function populateSelects(available, providers) {
  for (const [kind, sel] of [["gaze", "sel-gaze"], ["blink", "sel-blink"],
    ["headpose", "sel-headpose"], ["voice", "sel-voice"]]) {
    const el = $(sel);
    el.innerHTML = "";
    for (const name of available[kind]) {
      const o = document.createElement("option");
      o.value = name; o.textContent = name;
      el.appendChild(o);
    }
    el.value = providers[kind] === "off" ? el.value : providers[kind];
    el.onchange = () => {
      send({ type: "switch", kind, name: el.value });
      if (kind === "voice") setupWebSpeech(el.value === "webspeech");
    };
  }
  populated = true;
}

for (const [id, modality] of [["tgl-gaze", "gaze"], ["tgl-blink", "blink"],
  ["tgl-headpose", "headpose"], ["tgl-voice", "voice"]]) {
  $(id).onchange = (e) => send({ type: "toggle", modality, on: e.target.checked });
}

/* ---------------- sliders (config-bound) ---------------- */
const sliderIds = ["sl-mincutoff", "sl-beta", "sl-deadzone", "sl-expand",
  "sl-gainx", "sl-gainy", "sl-poseassist", "sl-holdclick", "sl-adratio",
  "sl-yawon", "sl-pitchon", "sl-dwell", "sl-cooldown"];
const valueSpan = {
  "sl-mincutoff": "v-mincutoff", "sl-beta": "v-beta", "sl-deadzone": "v-deadzone",
  "sl-expand": "v-expand",
  "sl-gainx": "v-gainx", "sl-gainy": "v-gainy", "sl-poseassist": "v-poseassist",
  "sl-holdclick": "v-holdclick", "sl-adratio": "v-adratio", "sl-yawon": "v-yawon",
  "sl-pitchon": "v-pitchon", "sl-dwell": "v-dwell", "sl-cooldown": "v-cooldown",
};
function getPath(obj, dotted) {
  return dotted.split(".").reduce((o, k) => (o ? o[k] : undefined), obj);
}
async function initSliders() {
  const cfg = await (await fetch("/api/config")).json();
  for (const id of sliderIds) {
    const el = $(id);
    const v = getPath(cfg, el.dataset.path);
    if (v !== undefined) { el.value = v; $(valueSpan[id]).textContent = v; }
    el.oninput = () => { $(valueSpan[id]).textContent = el.value; };
    el.onchange = () =>
      send({ type: "set", path: el.dataset.path, value: parseFloat(el.value) });
  }
  $("tgl-tts").checked = !!(cfg.voice && cfg.voice.tts_feedback);
  speechLang = (cfg.voice && cfg.voice.webspeech_lang) || speechLang;
  if (recognition) recognition.lang = speechLang;
  const m = cfg.headpose && cfg.headpose.mapping;
  if (m && m.head_left === "tab_prev") $("sel-mapping").value = "tabs_lr";
}
$("sel-mapping").onchange = () => {
  const apps = $("sel-mapping").value === "apps_lr";
  const map = apps
    ? { head_left: "app_prev", head_right: "app_next",
        head_up: "tab_next", head_down: "tab_prev" }
    : { head_left: "tab_prev", head_right: "tab_next",
        head_up: "app_next", head_down: "app_prev" };
  for (const [k, v] of Object.entries(map))
    send({ type: "set", path: `headpose.mapping.${k}`, value: v });
  toast("gesture mapping updated");
};
$("tgl-tts").onchange = (e) =>
  send({ type: "set", path: "voice.tts_feedback", value: e.target.checked });

/* ---------------- buttons ---------------- */
let paused = false;
$("btn-pause").onclick = () => {
  send({ type: "action", name: paused ? "resume" : "pause" });
};
$("btn-calibrate").onclick = () => {
  toast("Quick 9-point calibration — follow the dots on screen");
  send({ type: "action", name: "calibrate", points: 9 });
};
$("btn-calibrate-hd").onclick = () => {
  const points = parseInt($("sel-calib-points").value, 10);
  toast(`Precision calibration (${points} points) — follow the dots, then the quality check`);
  send({ type: "action", name: "calibrate", points });
};
$("btn-center").onclick = () => send({ type: "action", name: "set_center" });
$("btn-neutral").onclick = () => send({ type: "action", name: "set_neutral" });
$("btn-report").onclick = () => {
  toast("Generating report…");
  send({ type: "action", name: "report" });
};
$("btn-session").onclick = () => {
  send({ type: "action", name: "new_session" });
  toast("New metrics session started");
};

/* ---------------- Web Speech API engine ---------------- */
let recognition = null;
let speechLang = "en-IN";     // overridden by voice.webspeech_lang from config
let ttsSpeaking = false;      // gate: never transcribe our own TTS voice
let srBlocked = false;        // browser unsupported or mic denied — stop retrying

function setupWebSpeech(enable) {
  if (!enable) {
    if (recognition) { recognition.onend = null; recognition.stop(); recognition = null; }
    return;
  }
  if (recognition || srBlocked) return;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    srBlocked = true;
    toast("Web Speech API not supported in this browser — use Chrome/Edge");
    return;
  }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = speechLang;
  recognition.onresult = (e) => {
    if (ttsSpeaking) return;              // our own voice — ignore
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      const text = r[0].transcript.trim();
      if (!text) continue;
      if (r.isFinal) send({ type: "voice_transcript", text });
      else $("voice-partial").textContent = text;
    }
  };
  recognition.onend = () => {              // keep alive
    if (recognition) try { recognition.start(); } catch (_) {}
  };
  recognition.onerror = (e) => {
    if (e.error === "not-allowed") {
      srBlocked = true;
      if (recognition) { recognition.onend = null; recognition = null; }
      toast("Microphone permission denied — allow the mic and reload");
    }
  };
  recognition.start();
  toast(`Web Speech listening (${speechLang})`);
}

function speak(text) {
  if (!$("tgl-tts").checked || !text || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.15; u.volume = 0.9;
  u.onstart = () => { ttsSpeaking = true; };
  // small grace period: recognition finals can arrive just after speech ends
  u.onend = () => setTimeout(() => { ttsSpeaking = false; }, 300);
  u.onerror = () => { ttsSpeaking = false; };
  speechSynthesis.speak(u);
}

/* "HELLO ONE" opens this page with ?greet=1 — speak immediately rather than
   waiting for the voice.command round-trip, so the greeting is never lost
   to a tab that just opened and hasn't connected its websocket yet. */
if (new URLSearchParams(location.search).get("greet") === "1") {
  window.addEventListener("load", () =>
    setTimeout(() => speak("Hello! HandsFree is ready to help."), 400));
}

/* ---------------- state rendering ---------------- */
function onState(s) {
  if (!populated && s.available) populateSelects(s.available, s.providers);

  // The webspeech engine lives in THIS tab: if the server expects it and we
  // aren't listening yet (fresh page load), start it now.
  if (s.providers.voice === "webspeech" && s.enabled.voice) setupWebSpeech(true);

  // ambient UI: eye emblem opens green when a face is tracked, and its iris
  // mirrors the live gaze estimate
  document.body.classList.toggle("face-on", !!s.face);
  const logo = $("eye-logo");
  if (logo && s.gaze_sample) {
    const gx = Math.min(Math.max(s.gaze_sample.nx, 0), 1);
    const gy = Math.min(Math.max(s.gaze_sample.ny, 0), 1);
    logo.style.setProperty("--iris-x", `${((gx - 0.5) * 14).toFixed(1)}px`);
    logo.style.setProperty("--iris-y", `${((gy - 0.5) * 8).toFixed(1)}px`);
  }

  const camText = !s.camera_ok ? "no camera"
    : (s.camera_healthy === false ? "camera recovering…" : "camera ok");
  setChip("chip-camera", camText, s.camera_ok && s.camera_healthy !== false);
  setChip("chip-face", s.face ? "face ✓" : "no face — aim camera at you", s.face);
  setChip("chip-fps", `${s.metrics.fps} fps`, s.metrics.fps > 15);
  setChip("chip-cpu", `${s.metrics.cpu_percent}% cpu`, s.metrics.cpu_percent < 60);
  $("no-cam").classList.toggle("hidden", s.camera_ok);

  paused = s.cursor.paused;
  $("btn-pause").textContent = paused ? "Resume control" : "Pause control";
  $("btn-pause").classList.toggle("active", paused);

  // toggles reflect server state (avoid clobbering user mid-click: only set if different)
  for (const [id, key] of [["tgl-gaze", "gaze"], ["tgl-blink", "blink"],
    ["tgl-headpose", "headpose"], ["tgl-voice", "voice"]]) {
    const el = $(id);
    if (el !== document.activeElement && el.checked !== s.enabled[key])
      el.checked = s.enabled[key];
  }

  // gaze
  if (s.gaze_sample) {
    $("ro-gaze").textContent =
      `cursor (${s.cursor.x}, ${s.cursor.y})  q=${s.gaze_sample.quality}` +
      (s.cursor.override ? "  · mouse override" : "") +
      (s.cursor.frozen ? "  · frozen" : "");
  }
  if (s.calibrating) {
    $("calib-info").textContent = `calibrating… ${s.calib_status}`;
  } else if (s.calibrated) {
    let txt = "calibrated ✓ (polynomial";
    if (s.calib_accuracy_px != null) txt += ` · ±${s.calib_accuracy_px}px error`;
    if (s.calib_validation_px != null) txt += ` · ±${s.calib_validation_px}px held-out`;
    $("calib-info").textContent = txt + ")";
  } else {
    $("calib-info").textContent = s.calib_last
      ? `${s.calib_last} — geometric mapping in use`
      : "not calibrated — geometric mapping in use";
  }

  // blink
  if (s.blink) {
    const b = s.blink;
    // custom_cnn reports open-probabilities in 0..1; ear ~0.1-0.4; angle ~8-40°
    const scale = b.measure === "custom_cnn" ? 1.0
      : b.measure === "ear" ? 0.45 : 45;
    $("bar-left").style.width = `${Math.min(b.left / scale, 1) * 100}%`;
    $("bar-right").style.width = `${Math.min(b.right / scale, 1) * 100}%`;
    $("thr-left").style.left = `${Math.min(b.thr_left / scale, 1) * 100}%`;
    $("thr-right").style.left = `${Math.min(b.thr_right / scale, 1) * 100}%`;
    const p = b.progress;
    if (p.closed) {
      const pct = Math.min(p.duration_ms / 1700, 1) * 100;
      $("hold-fill").style.width = `${pct}%`;
      $("hold-label").textContent = p.stage || "hold…";
    } else {
      $("hold-fill").style.width = "0";
      $("hold-label").textContent = "open";
    }
    $("ro-blink").textContent =
      `clicks ${b.counts.click || 0} · right ${b.counts.right_click || 0} · ` +
      `double ${b.counts.double_click || 0} · natural ${b.counts.natural || 0}`;
  }

  // head pose
  if (s.pose) {
    const clamp = (v, r) => Math.max(-r, Math.min(r, v));
    $("g-yaw").style.left = `${50 + clamp(s.pose.yaw, 40) / 40 * 48}%`;
    $("g-pitch").style.left = `${50 + clamp(s.pose.pitch, 40) / 40 * 48}%`;
    $("t-yaw").textContent = `${s.pose.yaw}°`;
    $("t-pitch").textContent = `${s.pose.pitch}°`;
    const c = s.pose.counts;
    $("ro-head").textContent = s.pose.valid
      ? `← ${c.head_left} → ${c.head_right} ↑ ${c.head_up} ↓ ${c.head_down}`
      : "calibrating neutral…";
  }

  // voice
  $("ro-voice").textContent = s.providers.voice === "off"
    ? "engine off" : `engine: ${s.providers.voice}` +
      (s.voice.dictation ? " · DICTATION ON" : "");

  // metrics
  chFps.push(s.metrics.fps);
  const total = s.metrics.stages_ms.total;
  chLat.push(total ? total.mean : 0);
  const jit = s.cursor.jitter_px;
  if (jit >= 0) chJit.push(jit);
  chCpu.push(s.metrics.cpu_percent);

  const st = s.metrics.stages_ms;
  $("stage-table").innerHTML = Object.entries(st)
    .map(([k, v]) => `<div>${k}: <b>${v.mean}ms</b> <span>p95 ${v.p95}</span></div>`)
    .join("");
}

/* ---------------- events ---------------- */
function onEvent(topic, d) {
  if (topic === "system.toast") { toast(d.text); logEvent(d.text); }
  else if (topic === "blink.event") {
    if (d.kind !== "natural")
      logEvent(`😉 blink <b>${d.kind}</b> (${Math.round(d.duration * 1000)}ms)`);
  }
  else if (topic === "head.gesture") logEvent(`↔ gesture <b>${d.name}</b>`);
  else if (topic === "voice.partial") $("voice-partial").textContent = d.text;
  else if (topic === "voice.command") {
    $("voice-partial").textContent = "";
    $("voice-final").textContent = `“${d.text}”`;
    $("voice-reply").textContent = d.reply || "";
    if (d.intent && d.intent !== "dictation")
      logEvent(`🎙 “${d.text}” → <b>${d.intent}</b> ${d.handled ? "✓" : "✗"}`);
    // replies are spoken by the backend (Windows SAPI) — speaking them here
    // too would double-voice every response
  }
  else if (topic === "control.paused") { paused = true; toast("Control paused"); }
  else if (topic === "control.resumed") { paused = false; toast("Control resumed"); }
  else if (topic === "action.executed" && d.source === "gesture")
    logEvent(`↔ ${d.gesture} → <b>${d.action}</b>`);
}

/* ---------------- mission clock ---------------- */
const bootT = Date.now();
setInterval(() => {
  const el = $("mission-clock");
  if (!el) return;
  const s = Math.floor((Date.now() - bootT) / 1000);
  const mm = String(Math.floor(s / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  el.textContent = `T+${mm}:${ss}`;
}, 1000);

/* ---------------- boot ---------------- */
connect();
initSliders();
