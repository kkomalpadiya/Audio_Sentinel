"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("upload-form"), file: $("file-input"), details: $("file-details"),
    name: $("file-name"), size: $("file-size"), duration: $("file-duration"),
    preview: $("preview"), scope: $("scope"), consent: $("consent"),
    analyze: $("analyze-button"), reset: $("reset-button"), state: $("run-state"),
    status: $("status"), error: $("error"), loading: $("loading"), results: $("results"),
    formatInfo: $("format-info"), ask: $("ask-button"), answer: $("answer"),
  };
  let limits = { max_bytes: 10 * 1024 * 1024, max_duration_seconds: 30, extensions: [".wav", ".flac"] };
  let previewUrl = null;
  let current = null; // last successful analysis response

  const fmtBytes = (n) => (n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`);
  const text = (node, value) => { node.textContent = value; };
  const pretty = (s) => String(s).replace(/_/g, " ");

  function setState(label, busy = false) { text(el.state, label); el.state.classList.toggle("busy", busy); }
  function showError(message) { el.error.hidden = !message; text(el.error, message || ""); }

  fetch("/api/v1/uploads/limits").then((r) => (r.ok ? r.json() : null)).then((data) => {
    if (!data) return;
    limits = data;
    text(el.formatInfo, `${data.formats.join(" or ")} · up to ${fmtBytes(data.max_bytes)} · up to ${data.max_duration_seconds} s`);
  }).catch(() => {});

  function clientCheck(file) {
    if (!file) return "Choose a recording first.";
    if (file.size === 0) return "The selected file is empty.";
    if (file.size > limits.max_bytes) return `The file is ${fmtBytes(file.size)}; the limit is ${fmtBytes(limits.max_bytes)}.`;
    const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
    if (!limits.extensions.includes(ext)) return "Unsupported file type. Choose a .wav or .flac recording.";
    return null;
  }

  function updateButton() {
    el.analyze.disabled = !(el.file.files[0] && el.consent.checked && !clientCheck(el.file.files[0]));
  }

  el.file.addEventListener("change", () => {
    const file = el.file.files[0];
    showError("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    if (!file) { el.details.hidden = true; updateButton(); return; }
    el.details.hidden = false;
    text(el.name, file.name);
    text(el.size, fmtBytes(file.size));
    text(el.duration, "");
    previewUrl = URL.createObjectURL(file);
    el.preview.src = previewUrl;
    const problem = clientCheck(file);
    if (problem) { showError(problem); setState("Invalid file"); }
    else { setState("Ready"); text(el.status, "File selected. Confirm consent and click Analyze Audio."); }
    updateButton();
  });
  el.preview.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(el.preview.duration)) text(el.duration, `${el.preview.duration.toFixed(2)} s`);
  });
  el.consent.addEventListener("change", updateButton);

  async function readJson(response) {
    try { return await response.json(); } catch { return null; }
  }

  el.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = el.file.files[0];
    const problem = clientCheck(file);
    if (problem) { showError(problem); return; }
    if (!el.consent.checked) { showError("Please confirm you are authorized to analyse this recording."); return; }
    const body = new FormData();
    body.append("file", file, file.name);
    body.append("processing_scope", el.scope.value);
    body.append("consent_confirmed", "true");

    showError(""); el.results.hidden = true; el.answer.hidden = true;
    el.loading.hidden = false; el.analyze.disabled = true; el.file.disabled = true;
    setState("Analyzing", true); text(el.status, `Analyzing ${file.name}…`);
    try {
      const response = await fetch("/api/v1/uploads/analyze", { method: "POST", body });
      const data = await readJson(response);
      if (!response.ok) {
        const message = data && data.error ? data.error : `The server returned ${response.status}.`;
        throw new Error(message);
      }
      current = data;
      render(data);
      setState("Complete");
      text(el.status, `Analysis complete for ${data.input.display_name} (${data.input.duration_seconds} s, ${data.input.sample_rate_hz} Hz, ${data.input.channels} ch).`);
    } catch (error) {
      setState("Failed");
      showError(error.message || "The analysis could not be completed.");
      text(el.status, "Fix the problem above or choose another file.");
    } finally {
      el.loading.hidden = true; el.file.disabled = false; el.reset.hidden = false; updateButton();
    }
  });

  function cell(row, value, cls) { const td = document.createElement("td"); if (cls) td.className = cls; td.textContent = value; row.appendChild(td); }
  function list(node, items) {
    node.replaceChildren(...items.map((item) => { const li = document.createElement("li"); li.textContent = item; return li; }));
  }

  function render(data) {
    const ex = data.explanation;
    el.results.hidden = false;
    text($("score"), `${Number(ex.risk_score).toFixed(0)} / ${ex.score_range[1]}`);
    text($("score-range"), `range ${ex.score_range[0]}–${ex.score_range[1]}`);
    $("meter-fill").style.width = `${Math.max(0, Math.min(100, ex.risk_score))}%`;
    el.results.className = `lvl-${ex.risk_level}`;
    text($("level"), ex.risk_level);
    text($("outcome"), pretty(ex.outcome));
    text($("review-flag"), ex.human_review_recommended ? "Human review recommended" : "No review required by policy");
    text($("scope-label"), data.evaluation.processing_scope === "acoustic_only" ? "Acoustic-only scope" : "Acoustic + speech scope");

    const events = $("events-body");
    events.replaceChildren();
    const rows = [...ex.acoustic_events.map((e) => [e.display_name, `${e.start_seconds.toFixed(2)}–${e.end_seconds.toFixed(2)} s`, e.peak_score.toFixed(2) + (e.high_confidence ? " (high)" : ""), `+${e.rule_points}`]),
      ...ex.language_findings.map((f) => [f.display_name, `${f.start_seconds.toFixed(2)}–${f.end_seconds.toFixed(2)} s`, "rule match", `+${f.rule_points}`])];
    if (!rows.length) rows.push(["No acoustic events or language findings above the threshold", "–", "–", "0"]);
    rows.forEach((values) => { const tr = document.createElement("tr"); values.forEach((v, i) => cell(tr, v, i > 1 ? "num" : "")); events.appendChild(tr); });
    text($("speech-summary"), ex.speech_summary);

    const contrib = $("contrib-body");
    contrib.replaceChildren();
    ex.contributions.forEach((c) => {
      const tr = document.createElement("tr");
      cell(tr, pretty(c.branch)); cell(tr, `${pretty(c.input_status)} · ${pretty(c.agreement)}`);
      cell(tr, c.points === null ? "n/a" : `${c.points} / ${c.max_points}`, "num"); cell(tr, c.summary);
      contrib.appendChild(tr);
    });
    const b = ex.breakdown;
    text($("rule-set"), `${b.rule_set_id} v${b.rule_set_version}` + (b.reproduced && b.total_before_cap !== null && b.total_before_cap > b.total_max_score ? ` · ${b.total_before_cap} capped at ${b.total_max_score}` : ""));
    list($("reasons"), ex.risk_reasons);
  }

  function renderAnswer(ex) {
    el.answer.hidden = false;
    const verdict = $("verdict");
    verdict.className = `verdict v-${ex.verdict}`;
    text(verdict, pretty(ex.verdict));
    text($("answer-text"), ex.answer);
    list($("decision-reasons"), ex.decision_reasons);
    list($("limitations"), ex.limitations);
  }

  el.ask.addEventListener("click", async () => {
    if (!current) return;
    el.ask.disabled = true; showError("");
    try {
      const response = await fetch(`/api/v1/uploads/${encodeURIComponent(current.upload_id)}/explanation`);
      const data = await readJson(response);
      if (!response.ok) throw new Error(data && data.error ? data.error : "The explanation is unavailable; treat the result as inconclusive.");
      renderAnswer(data);
    } catch (error) {
      showError(error.message);
    } finally { el.ask.disabled = false; }
  });

  el.reset.addEventListener("click", () => {
    el.form.reset(); current = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; el.preview.removeAttribute("src");
    el.details.hidden = true; el.results.hidden = true; el.answer.hidden = true; el.reset.hidden = true;
    showError(""); setState("Waiting for file"); text(el.status, "Select a recording to begin."); updateButton();
  });
})();
