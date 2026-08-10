"use strict";

let pollTimer = null;
let logOffset = 0;
let currentCorpus = null;

function $(sel, root) { return (root || document).querySelector(sel); }
function $all(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

function showScreen(id) {
  $all(".screen").forEach((el) => el.classList.add("hidden"));
  $(`#${id}`).classList.remove("hidden");
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function appendLog(text) {
  if (!text) return;
  const box = $("#log-box");
  box.textContent += text;
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ------------------------------------------------------------
// Polling / status machine
// ------------------------------------------------------------

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(poll, 1500);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

async function poll() {
  let data;
  try {
    data = await api(`/api/status?since=${logOffset}`);
  } catch (e) {
    appendLog(`\n[status check failed: ${e.message}]\n`);
    return;
  }
  handleStatus(data, { fromPoll: true });
}

function handleStatus(data, opts) {
  opts = opts || {};
  currentCorpus = data.corpus_name || currentCorpus;

  if (data.log) {
    appendLog(data.log);
    logOffset = data.log_length;
  }
  if (data.current_step) {
    $("#step-label").textContent = data.current_step;
  }

  if (data.status === "running") {
    showScreen("screen-progress");
    $("#error-box").classList.add("hidden");
    startPolling();
  } else if (data.status === "awaiting_input") {
    stopPolling();
    renderDecision(data.decision);
  } else if (data.status === "error") {
    stopPolling();
    showScreen("screen-progress");
    $("#error-box").textContent = data.error || "Unknown error.";
    $("#error-box").classList.remove("hidden");
  } else if (data.status === "complete") {
    stopPolling();
    renderComplete(data.manifest, currentCorpus);
  } else if (data.status === "idle") {
    stopPolling();
    if (!opts.fromPoll) showScreen("screen-setup");
  }
}

function renderDecision(decision) {
  switch (decision.type) {
    case "flagged_term_review": return renderFlaggedTermReview(decision.payload);
    case "cluster_review": return renderClusterReview(decision.payload);
    case "voyant_settings": return renderVoyantSettings(decision.payload);
    case "phase2_setup": return renderPhase2Setup(decision.payload);
    case "escalation": return renderEscalation(decision.payload);
    case "injection_review": return renderInjectionReview(decision.payload);
    default:
      console.warn("Unknown decision type", decision);
  }
}

// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------

$("#setup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errBox = $("#setup-error");
  errBox.classList.add("hidden");

  const form = e.target;
  const fd = new FormData(form);
  fd.set("clean", $("#clean").checked ? "true" : "false");

  logOffset = 0;
  $("#log-box").textContent = "";

  try {
    await api("/api/start", { method: "POST", body: fd });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
});

// ------------------------------------------------------------
// FLAGGED TERM REVIEW
// ------------------------------------------------------------

function renderFlaggedTermReview(payload) {
  const container = $("#flagged-terms-list");
  container.innerHTML = "";

  payload.items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.id = item.id;

    const candidatesOptions = item.top_candidates.map((c, i) =>
      `<option value="${i}">[${i + 1}] ${escapeHtml(c.sense)} -- ${escapeHtml(c.definition)} (score ${c.score})</option>`
    ).join("");

    card.innerHTML = `
      <h3>${escapeHtml(item.word)} <span class="meta">(stem: ${escapeHtml(item.stem)}, ${escapeHtml(item.pos)}, count ${item.count})</span></h3>
      <p class="meta">"${escapeHtml(item.sentence)}"</p>
      <div class="choice-row">
        <label><input type="radio" name="choice-${item.id}" value="predicted" checked>
          Use predicted: <em>${escapeHtml(item.predicted_definition)}</em></label><br>
        <label><input type="radio" name="choice-${item.id}" value="default">
          Keep default: <em>${escapeHtml(item.default_definition)}</em></label><br>
        <label><input type="radio" name="choice-${item.id}" value="candidate">
          Pick a candidate:
          <select class="candidate-select" data-id="${item.id}">${candidatesOptions}</select></label><br>
        <label><input type="radio" name="choice-${item.id}" value="manual">
          Manual definition: <input type="text" class="manual-input" data-id="${item.id}" placeholder="type a definition"></label>
      </div>
    `;
    container.appendChild(card);
  });

  showScreen("screen-flagged-terms");
}

$("#accept-all-predicted").addEventListener("click", () => {
  $all("#flagged-terms-list .card").forEach((card) => {
    const id = card.dataset.id;
    const radio = card.querySelector(`input[name="choice-${id}"][value="predicted"]`);
    if (radio) radio.checked = true;
  });
});

async function submitFlaggedTerms() {
  const decisions = $all("#flagged-terms-list .card").map((card) => {
    const id = parseInt(card.dataset.id, 10);
    const choice = card.querySelector(`input[name="choice-${id}"]:checked`).value;
    const entry = { id, choice };
    if (choice === "candidate") {
      entry.candidate_index = parseInt(card.querySelector(".candidate-select").value, 10);
    } else if (choice === "manual") {
      entry.manual_definition = card.querySelector(".manual-input").value;
    }
    return entry;
  });

  try {
    await api("/api/decisions/flagged-terms", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$all(".submit-flagged-terms").forEach((btn) => btn.addEventListener("click", submitFlaggedTerms));

// ------------------------------------------------------------
// CLUSTER REVIEW
// ------------------------------------------------------------

function renderClusterReview(payload) {
  const container = $("#cluster-review-list");
  container.innerHTML = "";

  payload.clusters.forEach((cluster, idx) => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.originalName = cluster.name;
    card.dataset.index = idx;

    const stemChips = cluster.stems.map((s) => `
      <label class="chip">
        <input type="checkbox" class="stem-keep" value="${escapeHtml(s)}" checked> ${escapeHtml(s)}
        <span class="exclude-wrap hidden">
          &middot; <label><input type="checkbox" class="stem-exclude" value="${escapeHtml(s)}"> also exclude globally</label>
        </span>
      </label>`).join("");

    const ngramChips = cluster.ngrams.map((g) => `
      <label class="chip">
        <input type="checkbox" class="ngram-keep" value="${escapeHtml(g)}" checked> ${escapeHtml(g)}
      </label>`).join("") || '<span class="hint">none</span>';

    card.innerHTML = `
      <label>Cluster name
        <input type="text" class="cluster-name" value="${escapeHtml(cluster.name)}">
      </label>
      <div class="bulk-actions">
        <button type="button" class="secondary remove-all-stems">Remove all stems</button>
        <button type="button" class="secondary remove-all-ngrams">Remove all n-grams</button>
        <button type="button" class="danger remove-cluster">Remove entire cluster</button>
      </div>
      <p class="meta">Stems (uncheck to remove):</p>
      <div class="chip-list stems">${stemChips}</div>
      <p class="meta">N-grams (uncheck to remove):</p>
      <div class="chip-list ngrams">${ngramChips}</div>
    `;
    card.dataset.removed = "false";
    container.appendChild(card);
  });

  function setStemKept(checkbox, keep) {
    checkbox.checked = keep;
    const chip = checkbox.closest(".chip");
    const wrap = chip.querySelector(".exclude-wrap");
    wrap.classList.toggle("hidden", keep);
    chip.classList.toggle("removed", !keep);
    if (keep) wrap.querySelector(".stem-exclude").checked = false;
  }

  function setNgramKept(checkbox, keep) {
    checkbox.checked = keep;
    checkbox.closest(".chip").classList.toggle("removed", !keep);
  }

  // Toggle the "also exclude globally" sub-checkbox's visibility based on keep state
  container.addEventListener("change", (e) => {
    if (e.target.classList.contains("stem-keep")) setStemKept(e.target, e.target.checked);
    if (e.target.classList.contains("ngram-keep")) setNgramKept(e.target, e.target.checked);
  });

  container.addEventListener("click", (e) => {
    if (e.target.classList.contains("remove-all-stems")) {
      const card = e.target.closest(".card");
      $all(".stem-keep", card).forEach((cb) => setStemKept(cb, false));
    }
    if (e.target.classList.contains("remove-all-ngrams")) {
      const card = e.target.closest(".card");
      $all(".ngram-keep", card).forEach((cb) => setNgramKept(cb, false));
    }
    if (e.target.classList.contains("remove-cluster")) {
      const card = e.target.closest(".card");
      const removed = card.dataset.removed !== "true";
      card.dataset.removed = String(removed);
      card.classList.toggle("cluster-removed", removed);
      e.target.textContent = removed ? "Undo remove" : "Remove entire cluster";
      if (removed) {
        $all(".stem-keep", card).forEach((cb) => setStemKept(cb, false));
        $all(".ngram-keep", card).forEach((cb) => setNgramKept(cb, false));
      }
    }
  });

  showScreen("screen-cluster-review");
}

async function submitClusterReview() {
  const clusters = $all("#cluster-review-list .card").map((card) => {
    const keptStems = [], removedStems = [], globallyExcluded = [];
    $all(".stem-keep", card).forEach((cb) => {
      if (cb.checked) keptStems.push(cb.value);
      else removedStems.push(cb.value);
    });
    $all(".stem-exclude", card).forEach((cb) => {
      if (cb.checked) globallyExcluded.push(cb.value);
    });

    const keptNgrams = [], removedNgrams = [];
    $all(".ngram-keep", card).forEach((cb) => {
      if (cb.checked) keptNgrams.push(cb.value);
      else removedNgrams.push(cb.value);
    });

    return {
      original_name: card.dataset.originalName,
      removed: card.dataset.removed === "true",
      new_name: $(".cluster-name", card).value,
      kept_stems: keptStems, removed_stems: removedStems,
      kept_ngrams: keptNgrams, removed_ngrams: removedNgrams,
      globally_excluded_stems: globallyExcluded,
    };
  });

  try {
    await api("/api/decisions/cluster-review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clusters }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$all(".submit-cluster-review").forEach((btn) => btn.addEventListener("click", submitClusterReview));

// ------------------------------------------------------------
// VOYANT SETTINGS
// ------------------------------------------------------------

function renderVoyantSettings(payload) {
  $("#voyant-summary").textContent =
    `Phase 1 produced ${payload.cluster_count} cluster(s) covering ${payload.stem_count} stem(s).`;
  showScreen("screen-voyant-settings");
}

$all("#screen-voyant-settings .yesno-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $all("#screen-voyant-settings .yesno-btn").forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    $("#use_smart_stopwords").value = btn.dataset.value;
  });
});

$("#voyant-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/decisions/voyant-settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corpus_id: $("#corpus_id").value,
        use_smart_stopwords: $("#use_smart_stopwords").value === "true",
      }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
});

// ------------------------------------------------------------
// PHASE 2 SETUP
// ------------------------------------------------------------

function renderPhase2Setup(payload) {
  $("#phase2-summary").textContent =
    `Phase 1 finished with ${payload.cluster_count} cluster(s). State saved to ${payload.phase1_state_path}.`;
  refreshOllamaModels();
  showScreen("screen-phase2-setup");
}

$all("#screen-phase2-setup .yesno-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const group = $all(`.yesno-btn[data-target="${btn.dataset.target}"]`);
    group.forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    $(`#${btn.dataset.target}`).value = btn.dataset.value;
    $("#condensation-fields").classList.toggle("hidden", btn.dataset.value === "false");
  });
});

$("#auto_detect_metadata").addEventListener("change", (e) => {
  $("#metadata-fields").classList.toggle("hidden", e.target.checked);
});

async function refreshOllamaModels() {
  const statusEl = $("#ollama-status");
  const select = $("#ollama_model");
  statusEl.textContent = "Checking Ollama...";
  select.innerHTML = "";
  try {
    const data = await api("/api/ollama/models");
    if (!data.available) {
      statusEl.textContent = "Ollama does not appear to be running. Install it, pull a model, and start `ollama serve`.";
      return;
    }
    if (data.models.length === 0) {
      statusEl.textContent = "Ollama is running but no models are installed. Run `ollama pull <model>`.";
      return;
    }
    data.models.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      select.appendChild(opt);
    });
    statusEl.textContent = `Ollama is available -- ${data.models.length} model(s) found.`;
  } catch (err) {
    statusEl.textContent = `Could not check Ollama: ${err.message}`;
  }
}

$("#refresh-models").addEventListener("click", refreshOllamaModels);

$("#phase2-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const runCondensation = $("#run_condensation").value === "true";
  try {
    await api("/api/decisions/phase2-setup", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        top_n: parseInt($("#top_n").value, 10),
        density_percentile: parseFloat($("#density_percentile").value),
        run_condensation: runCondensation,
        rates: $("#rates").value,
        ollama_model: $("#ollama_model").value,
        max_trials: parseInt($("#max_trials").value, 10),
        title: $("#source_title").value,
        authors: $("#source_authors").value,
        date: $("#source_date").value,
        auto_detect_metadata: $("#auto_detect_metadata").checked,
      }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
});

// ------------------------------------------------------------
// ESCALATION
// ------------------------------------------------------------

function renderEscalation(payload) {
  const container = $("#escalation-list");
  container.innerHTML = "";

  payload.rates.forEach((r) => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.rate = r.rate;

    const trialsHtml = r.trials.map((t) =>
      `<li>trial ${t.trial}: ${t.word_count} words (target ${t.target_words})</li>`
    ).join("");

    card.innerHTML = `
      <h3>${r.rate}% condensation</h3>
      <ul class="meta">${trialsHtml}</ul>
      <p>Retry with the full source text included in the prompt?</p>
      <div class="yesno">
        <button type="button" class="yesno-btn escalate-btn selected" data-value="true">Yes</button>
        <button type="button" class="yesno-btn escalate-btn" data-value="false">No</button>
      </div>
      <input type="hidden" class="escalate-value" value="true">
    `;
    container.appendChild(card);
  });

  container.addEventListener("click", (e) => {
    if (!e.target.classList.contains("escalate-btn")) return;
    const card = e.target.closest(".card");
    $all(".escalate-btn", card).forEach((b) => b.classList.remove("selected"));
    e.target.classList.add("selected");
    $(".escalate-value", card).value = e.target.dataset.value;
  });

  showScreen("screen-escalation");
}

async function submitEscalation() {
  const decisions = {};
  $all("#escalation-list .card").forEach((card) => {
    decisions[card.dataset.rate] = $(".escalate-value", card).value === "true";
  });

  try {
    await api("/api/decisions/escalation", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$all(".submit-escalation").forEach((btn) => btn.addEventListener("click", submitEscalation));

// ------------------------------------------------------------
// INJECTION REVIEW
// ------------------------------------------------------------

const SPAN_TYPES = ["F", "T", "R", "C"];

function renderInjectionReview(payload) {
  const container = $("#injection-review-list");
  container.innerHTML = "";

  payload.rates.forEach((r) => {
    const rateHeader = document.createElement("h3");
    rateHeader.textContent = `${r.rate}% condensation`;
    container.appendChild(rateHeader);

    r.flags.forEach((flag) => {
      const card = document.createElement("div");
      card.className = "card";
      card.dataset.rate = r.rate;
      card.dataset.spanId = flag.span_id;

      const typeButtons = ["keep", ...SPAN_TYPES].map((t) => {
        const label = t === "keep" ? `Keep (${flag.type})` : t;
        const selected = t === "keep" ? "selected" : "";
        return `<button type="button" class="span-type-btn ${selected}" data-value="${t}">${label}</button>`;
      }).join("");

      card.innerHTML = `
        <p class="meta">${escapeHtml(flag.span_id)} (${escapeHtml(flag.type)}): ${escapeHtml(flag.reason)}</p>
        <p>"${escapeHtml(flag.text)}"</p>
        <div class="span-type-btns">${typeButtons}</div>
        <input type="hidden" class="span-choice" value="keep">
      `;
      container.appendChild(card);
    });
  });

  container.addEventListener("click", (e) => {
    if (!e.target.classList.contains("span-type-btn")) return;
    const card = e.target.closest(".card");
    $all(".span-type-btn", card).forEach((b) => b.classList.remove("selected"));
    e.target.classList.add("selected");
    $(".span-choice", card).value = e.target.dataset.value;
  });

  showScreen("screen-injection-review");
}

async function submitInjectionReview() {
  const decisions = {};
  $all("#injection-review-list .card").forEach((card) => {
    const rate = card.dataset.rate;
    decisions[rate] = decisions[rate] || {};
    decisions[rate][card.dataset.spanId] = $(".span-choice", card).value;
  });

  try {
    await api("/api/decisions/injection-review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$all(".submit-injection-review").forEach((btn) => btn.addEventListener("click", submitInjectionReview));

// ------------------------------------------------------------
// COMPLETE
// ------------------------------------------------------------

const FILE_LABELS = {
  condensed_text: "Condensed text",
  injection_report: "Injection report (JSON)",
  html_fragment: "HTML fragment (Spyral-paste-ready)",
  html_preview: "Standalone preview",
  human_report: "Human-readable report",
  plain_summary: "Plain-text summary",
  voyant_notebook: "Filled Voyant notebook",
  standalone_report: "Standalone report (non-Voyant)",
};

function fileLink(corpus, relativePath) {
  // relativePath is already relative to data/<corpus>/ (see
  // PipelineSession._rel in webapp/pipeline_session.py), so this is a
  // direct map onto the file-serving route -- no path parsing needed.
  return `/api/files/${encodeURIComponent(corpus)}/${relativePath.split("/").map(encodeURIComponent).join("/")}`;
}

let currentManifestRates = [];

function renderComplete(manifest, corpus) {
  if (!manifest) return;
  $("#complete-summary").textContent = `Corpus "${corpus}" -- all outputs saved under data/${corpus}/.`;

  const container = $("#complete-files");
  container.innerHTML = "";

  const topLevel = document.createElement("div");
  topLevel.className = "card file-links";
  topLevel.innerHTML = "<h3>Phase 1 / decision logs</h3>";
  [
    ["phase1_state", "Phase 1 state JSON"],
    ["phase1_html", "Phase 1 cluster HTML"],
    ["phase2_output_json", "Informative sentences JSON"],
    ["phase1_decisions_log", "Phase 1 decision log"],
    ["phase2_decisions_log", "Phase 2 decision log"],
  ].forEach(([key, label]) => {
    const path = manifest[key];
    if (!path) return;
    const a = document.createElement("a");
    a.textContent = label;
    a.href = fileLink(corpus, path);
    a.target = "_blank";
    topLevel.appendChild(a);
  });
  container.appendChild(topLevel);

  Object.entries(manifest.rates || {}).forEach(([rate, files]) => {
    const card = document.createElement("div");
    card.className = "card file-links";
    card.innerHTML = `<h3>${rate}% condensation</h3>`;
    Object.entries(files).forEach(([key, path]) => {
      const label = FILE_LABELS[key] || key;
      const a = document.createElement("a");
      a.textContent = label;
      a.href = fileLink(corpus, path);
      a.target = "_blank";
      card.appendChild(a);
    });
    container.appendChild(card);
  });

  currentManifestRates = Object.keys(manifest.rates || {}).map(Number).sort((a, b) => a - b);
  $("#regenerate-panel").classList.toggle("hidden", currentManifestRates.length === 0);
  $("#regen-existing-rates").textContent = currentManifestRates.length
    ? `Existing rates: ${currentManifestRates.map((r) => r + "%").join(", ")}. Enter one of these to redo it, or a new percentage to add another.`
    : "";
  if (currentManifestRates.length) {
    $("#regen-rate").value = currentManifestRates[currentManifestRates.length - 1];
  }
  $("#regen-prompt-field").classList.add("hidden");
  $("#regen-prompt").value = "";

  showScreen("screen-complete");
}

$("#start-new-run").addEventListener("click", async () => {
  try {
    await api("/api/reset", { method: "POST" });
  } catch (err) {
    // Reload anyway -- worst case init() re-syncs to whatever the server
    // still reports, same as before this fix existed.
  }
  window.location.reload();
});

// ------------------------------------------------------------
// REGENERATE A CONDENSATION (from the complete screen)
// ------------------------------------------------------------

async function previewRegenerationPrompt() {
  const rate = parseInt($("#regen-rate").value, 10);
  if (!rate) {
    alert("Enter a rate first.");
    return;
  }
  try {
    const data = await api(`/api/regenerate/prompt-preview?rate=${rate}`);
    $("#regen-prompt").value = data.prompt;
    $("#regen-prompt-field").classList.remove("hidden");
  } catch (err) {
    alert(err.message);
  }
}

async function submitRegeneration() {
  const rate = parseInt($("#regen-rate").value, 10);
  if (!rate) {
    alert("Enter a rate first.");
    return;
  }
  // If the prompt field was never revealed, send no override -- the
  // backend renders the same default prompt itself. If it was revealed,
  // send its current contents verbatim (edited or not).
  const promptRevealed = !$("#regen-prompt-field").classList.contains("hidden");
  const prompt = promptRevealed ? $("#regen-prompt").value : null;

  try {
    await api("/api/regenerate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rate, prompt }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$("#regen-preview-prompt").addEventListener("click", previewRegenerationPrompt);
$("#regen-submit").addEventListener("click", submitRegeneration);

// ------------------------------------------------------------
// INIT
// ------------------------------------------------------------

(async function init() {
  try {
    const data = await api("/api/status");
    handleStatus(data);
  } catch (err) {
    showScreen("screen-setup");
  }
})();
