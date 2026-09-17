"use strict";
const $ = id => document.getElementById(id);
let result = null;
let activeWindow = 0;
let running = false;
let chartGeometry = null;
const sampleNames = {tone: "Steady tone", sweep: "Frequency sweep", pulses: "Rhythmic pulses"};
const stops = [[13,18,45],[65,37,111],[126,49,137],[191,70,115],[236,118,81],[252,190,86],[249,244,174]];
function color(value) {
  const x = Math.max(0, Math.min(1, value)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * (x - i)));
}
function setupCanvas(canvas) {
  const width = Math.max(240, canvas.getBoundingClientRect().width);
  const height = canvas.getBoundingClientRect().height;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  ctx.fillStyle = "#131d2c"; ctx.fillRect(0, 0, width, height);
  ctx.font = "12px Segoe UI, sans-serif";
  return {ctx, width, height};
}
function drawWaveform() {
  const {ctx, width, height} = setupCanvas($("waveform"));
  if (!result) return;
  const left = 43, right = width - 14, center = (height - 22) / 2;
  const envelope = result.prepared.envelope;
  const peak = Math.max(.05, ...envelope.flat().map(Math.abs));
  const scale = (center - 10) / peak;
  ctx.strokeStyle = "#304257"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(left, center); ctx.lineTo(right, center); ctx.stroke();
  ctx.fillStyle = "#a2b3c8"; ctx.textAlign = "right";
  ctx.fillText(peak.toFixed(2), left - 8, 14); ctx.fillText("0", left - 8, center + 4);
  ctx.fillText((-peak).toFixed(2), left - 8, center * 2 - 6);
  ctx.strokeStyle = "#65e0e0"; ctx.lineWidth = Math.max(1, (right - left) / envelope.length);
  ctx.beginPath(); envelope.forEach(([low, high], i) => {
    const x = left + (i + .5) / envelope.length * (right - left);
    ctx.moveTo(x, center - high * scale); ctx.lineTo(x, center - low * scale);
  }); ctx.stroke();
  const window = result.windows[activeWindow];
  ctx.fillStyle = "rgba(195,238,131,.09)";
  ctx.fillRect(left + window.start_seconds / result.prepared.duration_seconds * (right - left), 3,
    (window.end_seconds - window.start_seconds) / result.prepared.duration_seconds * (right - left), center * 2 - 3);
  ctx.fillStyle = "#a2b3c8";
  for (let i = 0; i <= 4; i++) {
    ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
    ctx.fillText(`${(i * result.prepared.duration_seconds / 4).toFixed(1)} s`, left + i / 4 * (right - left), height - 2);
  }
}
function drawSpectrogram() {
  const {ctx, width, height} = setupCanvas($("spectrogram"));
  if (!result) return;
  const window = result.windows[activeWindow];
  const [rows, columns] = window.shape;
  const left = 47, top = 18, plotWidth = width - 131, plotHeight = height - 64;
  const minimum = Math.floor(Math.min(...result.windows.map(w => w.min_db)) / 10) * 10;
  const maximum = Math.max(minimum + 10, Math.ceil(Math.max(...result.windows.map(w => w.max_db)) / 10) * 10);
  const bitmap = document.createElement("canvas"); bitmap.width = columns; bitmap.height = rows;
  const bitmapContext = bitmap.getContext("2d");
  const pixels = bitmapContext.createImageData(columns, rows);
  for (let band = 0; band < rows; band++) for (let frame = 0; frame < columns; frame++) {
    const offset = ((rows - 1 - band) * columns + frame) * 4;
    pixels.data.set([...color((window.values[band][frame] - minimum) / (maximum - minimum)), 255], offset);
  }
  bitmapContext.putImageData(pixels, 0, 0);
  ctx.fillStyle = "#0d122d"; ctx.fillRect(left, top, plotWidth, plotHeight);
  // Each column is centered at its FFT-frame time, with a one-hop display width.
  const hopSeconds = result.recipe.hop_length / result.prepared.sample_rate_hz;
  const firstEdge = window.frame_times_seconds[0] - hopSeconds / 2;
  const imageLeft = left + firstEdge / window.duration_seconds * plotWidth;
  const imageWidth = columns * hopSeconds / window.duration_seconds * plotWidth;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bitmap, imageLeft, top, imageWidth, plotHeight);
  ctx.strokeStyle = "#3e5269"; ctx.lineWidth = 1; ctx.strokeRect(left, top, plotWidth, plotHeight);
  ctx.fillStyle = "#a2b3c8"; ctx.textAlign = "right";
  for (const band of [1,16,32,48,64]) {
    const y = top + (rows - band + .5) / rows * plotHeight;
    ctx.fillText(String(band), left - 9, y + 4);
  }
  ctx.save(); ctx.translate(12, top + plotHeight / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
  ctx.fillText("Mel band", 0, 0); ctx.restore();
  for (let i = 0; i <= 4; i++) {
    const x = left + i / 4 * plotWidth;
    ctx.textAlign = "center"; ctx.fillText((i * window.duration_seconds / 4).toFixed(2), x, top + plotHeight + 19);
  }
  ctx.fillText("Time within selected window (seconds)", left + plotWidth / 2, height - 3);
  if (window.padding_seconds > 0) {
    const actualDuration = window.end_seconds - window.start_seconds;
    const x = left + actualDuration / window.duration_seconds * plotWidth;
    ctx.setLineDash([4,4]); ctx.strokeStyle = "#edf4ff"; ctx.beginPath();
    ctx.moveTo(x, top); ctx.lineTo(x, top + plotHeight); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#edf4ff"; ctx.textAlign = "left"; ctx.fillText("Zero-padded tail", x + 8, top + 16);
  }
  const barX = left + plotWidth + 16;
  for (let y = 0; y < plotHeight; y++) {
    ctx.fillStyle = `rgb(${color(1 - y / plotHeight).join(",")})`; ctx.fillRect(barX, top + y, 11, 1.5);
  }
  ctx.fillStyle = "#a2b3c8"; ctx.textAlign = "left";
  for (let i = 0; i <= 4; i++) ctx.fillText(String(Math.round(maximum - i / 4 * (maximum - minimum))),
    barX + 16, top + i / 4 * (plotHeight - 8) + 8);
  ctx.fillText("dB", barX - 1, top - 5);
  chartGeometry = {left, top, plotWidth, plotHeight, imageLeft, imageWidth};
  $("spectrogram").setAttribute("aria-label", `${sampleNames[result.sample]} Log-Mel spectrogram, ${rows} Mel bands and ${columns} time frames; ${window.padding_seconds.toFixed(2)} seconds of padding. ${result.description}`);
}
function selectWindow(index) {
  activeWindow = index;
  const window = result.windows[index];
  $("shape-label").textContent = `${window.shape[0]} bands × ${window.shape[1]} frames`;
  $("plot-detail").textContent = `Source ${window.start_seconds.toFixed(2)}–${window.end_seconds.toFixed(2)} s · padding ${window.padding_seconds.toFixed(2)} s · hover to inspect`;
  drawWaveform(); drawSpectrogram();
}
async function runDemo(sample) {
  if (running) return;
  running = true;
  $("error").hidden = true;
  $("run-button").disabled = true; $("export-button").disabled = true; $("window-select").disabled = true;
  document.querySelectorAll('input[name="sample"]').forEach(input => {input.disabled = true; input.checked = input.value === sample;});
  $("run-state").textContent = "Processing"; $("run-state").classList.add("busy");
  $("status").textContent = `Processing ${sampleNames[sample].toLowerCase()} through the Python preparation and feature pipeline…${result ? " Previous result remains below." : " First run may take a moment."}`;
  $("spectrogram").setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/demo/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sample})});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The demo request could not be processed.");
    result = data;
    $("rate-value").textContent = `${data.source.sample_rate_hz / 1000} → ${data.prepared.sample_rate_hz / 1000} kHz`;
    $("channel-value").textContent = "Stereo → mono";
    $("window-value").textContent = String(data.windows.length);
    $("band-value").textContent = String(data.recipe.n_mels);
    $("wave-meta").textContent = `${data.prepared.duration_seconds.toFixed(1)} s · ${data.prepared.rms_dbfs.toFixed(1)} dBFS RMS · amplitude envelope`;
    $("sample-description").textContent = data.description;
    $("source-audio").pause(); $("prepared-audio").pause();
    $("source-audio").src = data.source.audio_url; $("prepared-audio").src = data.prepared.audio_url;
    $("source-audio").volume = .4; $("prepared-audio").volume = .4;
    $("fft-value").textContent = `${data.recipe.n_fft} samples`;
    $("hop-value").textContent = `${data.recipe.hop_length / data.prepared.sample_rate_hz * 1000} ms`;
    $("window-select").replaceChildren(...data.windows.map((window, index) => {
      const option = document.createElement("option"); option.value = String(index);
      option.textContent = `${window.duration_seconds} s window · source ${window.start_seconds.toFixed(2)}–${window.end_seconds.toFixed(2)} s${window.padding_seconds ? " · padded" : ""}`;
      return option;
    }));
    $("checks").replaceChildren();
    for (const [verified, label] of [[data.checks.source_unchanged, "Source unchanged"], [data.checks.repeat_reused, "Repeat outputs reused"],
      [data.checks.feature_bundles_verified === data.windows.length, "Saved features verified"], [data.checks.temporary_files_removed, "Temporary files cleaned"]]) {
      const item = document.createElement("span"); item.textContent = `${verified ? "✓" : "!"} ${label}`; $("checks").append(item);
    }
    selectWindow(0);
    $("run-state").textContent = "Complete";
    $("status").textContent = `${sampleNames[sample]} · ${data.windows.length} verified feature bundles · computed in ${data.elapsed_seconds.toFixed(2)} s`;
    return {sample, feature_count: data.windows.length, shapes: data.windows.map(w => w.shape)};
  } catch (error) {
    $("error").textContent = `${error.message} Confirm the local server is running and select Generate spectrogram to retry.`;
    $("error").hidden = false; $("run-state").textContent = "Could not complete";
    $("status").textContent = result ? `Showing the previous successful result: ${sampleNames[result.sample]}.` : "No processing result is available yet.";
  } finally {
    running = false; $("run-button").disabled = false;
    document.querySelectorAll('input[name="sample"]').forEach(input => input.disabled = false);
    $("export-button").disabled = !result; $("window-select").disabled = !result;
    $("run-state").classList.remove("busy"); $("spectrogram").setAttribute("aria-busy", "false");
  }
}
$("demo-form").addEventListener("submit", event => {event.preventDefault(); runDemo(new FormData(event.currentTarget).get("sample"));});
$("window-select").addEventListener("change", event => selectWindow(Number(event.target.value)));
$("source-audio").addEventListener("play", () => $("prepared-audio").pause());
$("prepared-audio").addEventListener("play", () => $("source-audio").pause());
$("export-button").addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = `audio-sentinel-${result.sample}-window-${activeWindow + 1}.png`;
  link.href = $("spectrogram").toDataURL("image/png"); link.click();
});
$("spectrogram").addEventListener("pointermove", event => {
  if (!result || !chartGeometry) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  const {imageLeft, imageWidth, top, plotHeight} = chartGeometry;
  if (x < imageLeft || x >= imageLeft + imageWidth || y < top || y >= top + plotHeight) return;
  const window = result.windows[activeWindow];
  const frame = Math.min(window.shape[1] - 1, Math.floor((x - imageLeft) / imageWidth * window.shape[1]));
  const band = window.shape[0] - 1 - Math.floor((y - top) / plotHeight * window.shape[0]);
  $("plot-detail").textContent = `Window time ${window.frame_times_seconds[frame].toFixed(3)} s · Mel band ${band + 1} · frame ${frame} · ${window.values[band][frame].toFixed(2)} dB`;
});
const resizeObserver = new ResizeObserver(() => {drawWaveform(); drawSpectrogram();});
resizeObserver.observe($("waveform")); resizeObserver.observe($("spectrogram"));
// Optional page-scoped access to the same generated-signal action. Unsupported
// browsers continue to use the form; no external script or polyfill is needed.
if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const tool = {
    name: "generate_audio_spectrogram",
    title: "Generate an Audio Sentinel spectrogram",
    description: "Run the local Python pipeline on a generated signal and display its waveform and Log-Mel features. No model inference is performed.",
    inputSchema: {type: "object", properties: {sample: {type: "string", enum: ["tone", "sweep", "pulses"]}}, required: ["sample"], additionalProperties: false},
    annotations: {readOnlyHint: false, untrustedContentHint: false},
    async execute(input) {
      if (!input || Object.keys(input).length !== 1 || !Object.hasOwn(sampleNames, input.sample)) throw new Error("Choose tone, sweep, or pulses.");
      if (running) throw new Error("A demonstration is already running.");
      const outcome = await runDemo(input.sample);
      if (!outcome) throw new Error("The pipeline could not complete; see the visible error message.");
      return outcome;
    }
  };
  try { Promise.resolve(document.modelContext.registerTool(tool, {signal: lifecycle.signal})).catch(() => {}); }
  catch { /* Optional browser capability; the visible form remains available. */ }
  window.addEventListener("pagehide", () => lifecycle.abort(), {once: true});
}
runDemo("sweep");
