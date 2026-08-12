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
    case "phase2_setup": return renderPhase2Setup(decision.payload);
    case "sanity_review": return renderSanityReview(decision.payload);
    case "escalation": return renderEscalation(decision.payload);
    case "injection_review": return renderInjectionReview(decision.payload);
    case "collocation_review": return renderCollocationReview(decision.payload);
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
    checkbox.closest(".chip").classList.toggle("removed", !keep);
  }

  function setNgramKept(checkbox, keep) {
    checkbox.checked = keep;
    checkbox.closest(".chip").classList.toggle("removed", !keep);
  }

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
    const keptStems = [], removedStems = [];
    $all(".stem-keep", card).forEach((cb) => {
      if (cb.checked) keptStems.push(cb.value);
      else removedStems.push(cb.value);
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

async function detectMetadata(ollamaModel) {
  const statusEl = $("#metadata-status");
  statusEl.classList.remove("error-text");
  if (!ollamaModel) {
    statusEl.textContent = "Pick an Ollama model above, then AI detection will run automatically.";
    return;
  }
  statusEl.textContent = "Detecting the source's title/author(s)/date with AI...";
  try {
    const data = await api(`/api/detect-metadata?ollama_model=${encodeURIComponent(ollamaModel)}`);
    $("#source_title").value = data.title === "Unclear" ? "" : data.title;
    $("#source_authors").value = data.authors === "Unclear" ? "" : data.authors;
    $("#source_date").value = data.date === "Unclear" ? "" : data.date;
    statusEl.textContent =
      "The workflow already used AI to try to identify this information (shown below, or left " +
      "blank where it wasn't confident). Review it and correct anything that's wrong before continuing.";
  } catch (err) {
    // Loud on purpose: a quiet hint here is exactly how this failure mode
    // goes unnoticed -- fields stay blank, get submitted as "Unclear",
    // and there's no other visible sign anything went wrong.
    console.error("detectMetadata failed:", err);
    statusEl.classList.add("error-text");
    statusEl.textContent =
      `Could not auto-detect title/author/date (${err.message}). This is worth checking the server ` +
      "console for -- enter the fields manually below in the meantime.";
  }
}

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
    if (data.models.length > 0) detectMetadata(select.value);
  } catch (err) {
    statusEl.textContent = `Could not check Ollama: ${err.message}`;
  }
}

$("#refresh-models").addEventListener("click", refreshOllamaModels);
$("#ollama_model").addEventListener("change", (e) => detectMetadata(e.target.value));

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
      }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
});

// ------------------------------------------------------------
// SANITY-CHECK REVIEW
// ------------------------------------------------------------

function renderSanityReview(payload) {
  const container = $("#sanity-review-list");
  container.innerHTML = "";

  payload.rates.forEach((r) => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.rate = r.rate;

    const candidatesHtml = r.candidates.map((c) => {
      const issuesHtml = (c.sanity_issues || []).map((i) => `<li>${escapeHtml(i)}</li>`).join("");
      return `
        <label class="sanity-candidate">
          <input type="radio" name="sanity-choice-${r.rate}" value="${c.trial}">
          Trial ${c.trial}: ${c.word_count} words (target ${c.target_words})
          <ul class="meta">${issuesHtml}</ul>
        </label>`;
    }).join("");

    card.innerHTML = `
      <h3>${r.rate}% condensation</h3>
      <p class="meta">${r.candidates.length} trial(s) hit the target word count but failed a sanity check:</p>
      ${candidatesHtml}
      <label class="sanity-candidate">
        <input type="radio" name="sanity-choice-${r.rate}" value="" checked>
        Reject all (fall back to escalation)
      </label>
    `;
    container.appendChild(card);
  });

  showScreen("screen-sanity-review");
}

async function submitSanityReview() {
  const decisions = {};
  $all("#sanity-review-list .card").forEach((card) => {
    const rate = card.dataset.rate;
    const checked = card.querySelector(`input[name="sanity-choice-${rate}"]:checked`);
    decisions[rate] = checked && checked.value ? parseInt(checked.value, 10) : null;
  });

  try {
    await api("/api/decisions/sanity-review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$all(".submit-sanity-review").forEach((btn) => btn.addEventListener("click", submitSanityReview));

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

    const trialsHtml = r.trials.map((t) => {
      const status = t.within_tolerance ? "within tolerance" : "outside tolerance";
      const issues = (t.sanity_issues || []).length
        ? ` -- failed sanity check: ${t.sanity_issues.map(escapeHtml).join("; ")}`
        : "";
      return `<li>trial ${t.trial}: ${t.word_count} words (target ${t.target_words}, ${status})${issues}</li>`;
    }).join("");

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

      const sourceLabel = flag.source_is_candidate_only
        ? "Closest candidate source sentence (not an official match -- this span's classification skipped source attribution):"
        : "Matched source sentence(s):";
      const sourceHtml = (flag.source_texts || []).length
        ? `<p class="meta">${sourceLabel}</p>` +
          flag.source_texts.map((s) => `<p class="source-context">"${escapeHtml(s)}"</p>`).join("")
        : `<p class="meta">No matching source sentence found for this span.</p>`;

      card.innerHTML = `
        <p class="meta">${escapeHtml(flag.span_id)} (${escapeHtml(flag.type)}): ${escapeHtml(flag.reason)}</p>
        <p>Condensed: "${escapeHtml(flag.text)}"</p>
        ${sourceHtml}
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
// COLLOCATION REVIEW (Phase 3)
// ------------------------------------------------------------

function renderCollocationReview(payload) {
  const container = $("#collocation-review-list");
  container.innerHTML = "";

  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr>
      <th></th><th>Term A</th><th>Cluster A</th><th>Term B</th><th>Cluster B</th><th>Co-occurrences</th>
    </tr></thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");

  payload.candidates.forEach((c, i) => {
    const tr = document.createElement("tr");
    if (c.confounded) tr.classList.add("confounded");
    tr.innerHTML = `
      <td><input type="checkbox" class="colloc-pick" value="${i}" ${i === 0 ? "checked" : ""}></td>
      <td>${escapeHtml(c.term_a)}</td>
      <td class="meta">${escapeHtml(c.cluster_a)}</td>
      <td>${escapeHtml(c.term_b)}</td>
      <td class="meta">${escapeHtml(c.cluster_b)}</td>
      <td>${c.hits}${c.confounded ? " &middot; possibly confounded" : ""}</td>
    `;
    tbody.appendChild(tr);
  });

  container.appendChild(table);
  showScreen("screen-collocation-review");
}

async function submitCollocationReview() {
  const selected_indices = $all(".colloc-pick:checked").map((cb) => parseInt(cb.value, 10));

  try {
    await api("/api/decisions/collocation-review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_indices }),
    });
    showScreen("screen-progress");
    startPolling();
  } catch (err) {
    alert(err.message);
  }
}

$all(".submit-collocation-review").forEach((btn) => btn.addEventListener("click", submitCollocationReview));

// ------------------------------------------------------------
// COMPLETE
// ------------------------------------------------------------

const FILE_LABELS = {
  condensed_text: "Condensed text",
  injection_report: "Injection report (JSON)",
  html_fragment: "HTML fragment",
  html_preview: "Standalone preview",
  human_report: "Human-readable report",
  plain_summary: "Plain-text summary",
  standalone_report: "Standalone report",
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

  // Phase 2 (condensation rates) before Phase 3 (distant reading), matching
  // pipeline order -- Phase 3's distant-reading report is built from the
  // Phase 2 condensations, so it belongs after them here too.
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

  if (manifest.distant_reading_report || manifest.distant_reading_note) {
    const phase3Card = document.createElement("div");
    phase3Card.className = "card file-links";
    phase3Card.innerHTML = "<h3>Phase 3 / distant reading</h3>";
    if (manifest.distant_reading_report) {
      const a = document.createElement("a");
      a.textContent = "Distant reading report";
      a.href = fileLink(corpus, manifest.distant_reading_report);
      a.target = "_blank";
      phase3Card.appendChild(a);
    }
    if (manifest.distant_reading_note) {
      const note = document.createElement("p");
      note.className = "hint";
      note.textContent = `Contexts/Collocates comparison skipped -- ${manifest.distant_reading_note}.`;
      phase3Card.appendChild(note);
    }
    container.appendChild(phase3Card);
  }

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
