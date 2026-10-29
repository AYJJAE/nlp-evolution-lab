/**
 * NLP Evolution Lab - Frontend Interactive Controller
 * Generative AI Module 1 Project
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    activeTab: "playground",
    currentSentence: "This movie was not bad at all, in fact I quite liked it.",
    attnMode: "real",
    attnLayer: 0,
    attnHead: 0,
    benchmarkData: null,
    scenarios: []
  };

  // DOM Elements
  const tabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");
  
  // Playground Elements
  const sentenceInput = document.getElementById("sentence-input");
  const charCounter = document.getElementById("char-counter");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnClear = document.getElementById("btn-clear");
  const playgroundLoading = document.getElementById("playground-loading");
  const playgroundAlert = document.getElementById("playground-alert");
  const presetChips = document.querySelectorAll(".preset-chips .chip");

  // Similarity Elements
  const word1Input = document.getElementById("word1-input");
  const word2Input = document.getElementById("word2-input");
  const btnCalcSim = document.getElementById("btn-calc-similarity");
  const simScoreVal = document.getElementById("sim-score-val");
  const simInterpretationTitle = document.getElementById("sim-interpretation-title");
  const simInterpretationDesc = document.getElementById("sim-interpretation-desc");
  const formulaSubs = document.getElementById("formula-subs");
  const oovWarning = document.getElementById("oov-warning");
  const vecW1Name = document.getElementById("vec-w1-name");
  const vecW2Name = document.getElementById("vec-w2-name");
  const vecW1Code = document.getElementById("vec-w1-code");
  const vecW2Code = document.getElementById("vec-w2-code");
  const simChips = document.querySelectorAll(".chip-sim");

  // Attention Elements
  const attnModeSelect = document.getElementById("attn-mode-select");
  const attnLayerSelect = document.getElementById("attn-layer-select");
  const attnHeadSelect = document.getElementById("attn-head-select");
  const btnRefreshAttn = document.getElementById("btn-refresh-attn");
  const attnActiveSentence = document.getElementById("attn-active-sentence");
  const attnModeBadge = document.getElementById("attn-mode-badge");
  const heatmapContainer = document.getElementById("heatmap-container");
  const heatmapHeadTitle = document.getElementById("heatmap-head-title");
  const heatmapHeadDesc = document.getElementById("heatmap-head-desc");
  const tokenInspectionPanel = document.getElementById("token-inspection-panel");
  const btnJumpAttention = document.getElementById("btn-jump-attention");

  // Results Elements
  const resTfidfAcc = document.getElementById("res-tfidf-acc");
  const resTfidfCounts = document.getElementById("res-tfidf-counts");
  const resTfidfTime = document.getElementById("res-tfidf-time");
  const resEmbAcc = document.getElementById("res-emb-acc");
  const resEmbCounts = document.getElementById("res-emb-counts");
  const resEmbTime = document.getElementById("res-emb-time");
  const resTrAcc = document.getElementById("res-tr-acc");
  const resTrCounts = document.getElementById("res-tr-counts");
  const resTrTime = document.getElementById("res-tr-time");
  const benchmarkTbody = document.getElementById("benchmark-tbody");
  const filterCategory = document.getElementById("filter-category");

  // About Elements
  const scenariosGrid = document.getElementById("scenarios-grid");

  // --------------------------------------------------------------------------
  // Tab Navigation Handling
  // --------------------------------------------------------------------------
  function switchTab(tabId) {
    state.activeTab = tabId;
    tabs.forEach(t => t.classList.toggle("active", t.dataset.tab === tabId));
    tabPanes.forEach(pane => pane.classList.toggle("active", pane.id === `tab-${tabId}`));
    window.scrollTo({ top: 0, behavior: "smooth" });

    if (tabId === "attention") {
      fetchAttentionData();
    } else if (tabId === "results" && !state.benchmarkData) {
      fetchBenchmarkData();
    }
  }

  tabs.forEach(tab => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  if (btnJumpAttention) {
    btnJumpAttention.addEventListener("click", () => switchTab("attention"));
  }

  // --------------------------------------------------------------------------
  // Feature 1: Text Playground Logic
  // --------------------------------------------------------------------------
  function updateCharCount() {
    const len = sentenceInput.value.length;
    charCounter.textContent = `${len} / 600 characters`;
    if (len > 600) {
      charCounter.style.color = "var(--rose-600)";
    } else {
      charCounter.style.color = "var(--text-muted)";
    }
  }

  sentenceInput.addEventListener("input", updateCharCount);

  btnClear.addEventListener("click", () => {
    sentenceInput.value = "";
    updateCharCount();
    hidePlaygroundAlert();
    sentenceInput.focus();
  });

  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const text = chip.dataset.text;
      sentenceInput.value = text;
      updateCharCount();
      runSentimentAnalysis(text);
    });
  });

  btnAnalyze.addEventListener("click", () => {
    const text = sentenceInput.value.trim();
    runSentimentAnalysis(text);
  });

  function showPlaygroundAlert(msg, type = "error") {
    playgroundAlert.textContent = msg;
    playgroundAlert.className = `alert-banner ${type}`;
    playgroundAlert.classList.remove("hidden");
  }

  function hidePlaygroundAlert() {
    playgroundAlert.classList.add("hidden");
  }

  async function runSentimentAnalysis(text) {
    if (!text || !text.trim()) {
      showPlaygroundAlert("Input sentence is empty. Please enter an English sentence to analyze.", "error");
      sentenceInput.focus();
      return;
    }

    if (text.length > 600) {
      showPlaygroundAlert("Input exceeds 600 characters. Please provide a more concise sentence for laboratory evaluation.", "warning");
      return;
    }

    hidePlaygroundAlert();
    playgroundLoading.classList.remove("hidden");
    btnAnalyze.disabled = true;

    try {
      const response = await fetch("/api/sentiment/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });

      const res = await response.json();
      if (!res.success) {
        showPlaygroundAlert(res.error || "Analysis failed. Please check server status.", "error");
        return;
      }

      state.currentSentence = text;
      renderPlaygroundResults(res.results);

      // Synchronize sentence in Attention Explorer bar
      if (attnActiveSentence) {
        attnActiveSentence.textContent = `"${text}"`;
      }

    } catch (err) {
      showPlaygroundAlert("Network error contacting NLP backend server. Please verify Flask is running on port 5000.", "error");
    } finally {
      playgroundLoading.classList.add("hidden");
      btnAnalyze.disabled = false;
    }
  }

  function renderPlaygroundResults(results) {
    // 1. TF-IDF Card
    const tfidf = results.tfidf;
    document.getElementById("tfidf-latency").textContent = `${tfidf.inference_time_ms} ms`;
    const tfidfBox = document.getElementById("tfidf-pred-box");
    tfidfBox.className = `prediction-box ${tfidf.prediction.toLowerCase()}`;
    document.getElementById("tfidf-prediction").textContent = tfidf.prediction;
    document.getElementById("tfidf-confidence").textContent = `Score: ${(tfidf.confidence * 100).toFixed(1)}%`;
    const tfidfPct = tfidf.prediction === "Positive" ? (tfidf.confidence * 100) : ((1 - tfidf.confidence) * 100);
    document.getElementById("tfidf-progress").style.width = `${Math.max(5, Math.min(95, tfidfPct))}%`;
    document.getElementById("tfidf-pedagogy").textContent = tfidf.pedagogical_note;

    // Tokens cloud
    const tfidfCloud = document.getElementById("tfidf-tokens-cloud");
    tfidfCloud.innerHTML = "";
    if (tfidf.token_contributions && tfidf.token_contributions.length > 0) {
      tfidf.token_contributions.forEach(t => {
        const span = document.createElement("span");
        span.className = `token-pill ${t.impact === "positive" ? "positive-weight" : "negative-weight"}`;
        span.textContent = `${t.token} (${t.contribution > 0 ? "+" : ""}${t.contribution})`;
        tfidfCloud.appendChild(span);
      });
    } else {
      tfidfCloud.innerHTML = '<span class="empty-hint">No active vocabulary features detected</span>';
    }

    // 2. Word Embedding Card
    const emb = results.embedding;
    document.getElementById("emb-latency").textContent = `${emb.inference_time_ms} ms`;
    const embBox = document.getElementById("emb-pred-box");
    embBox.className = `prediction-box ${emb.prediction.toLowerCase()}`;
    document.getElementById("emb-prediction").textContent = emb.prediction;
    document.getElementById("emb-confidence").textContent = `Score: ${(emb.confidence * 100).toFixed(1)}%`;
    const embPct = emb.prediction === "Positive" ? (emb.confidence * 100) : ((1 - emb.confidence) * 100);
    document.getElementById("emb-progress").style.width = `${Math.max(5, Math.min(95, embPct))}%`;
    document.getElementById("emb-pedagogy").textContent = emb.pedagogical_note;
    document.getElementById("emb-coverage-text").textContent = `In-Vocab Coverage: ${emb.coverage.coverage_pct}% (${emb.coverage.in_vocab_count}/${emb.coverage.total_tokens} words)`;
    document.getElementById("emb-vector-preview").textContent = `Centroid Sample: [${emb.vector_preview.join(", ")}...]`;

    // 3. Transformer Card
    const tr = results.transformer;
    document.getElementById("tr-latency").textContent = `${tr.inference_time_ms} ms`;
    const trBox = document.getElementById("tr-pred-box");
    trBox.className = `prediction-box ${tr.prediction.toLowerCase()}`;
    document.getElementById("tr-prediction").textContent = tr.prediction;
    document.getElementById("tr-confidence").textContent = `Softmax: ${(tr.confidence * 100).toFixed(1)}%`;
    const trPct = tr.prediction === "Positive" ? (tr.confidence * 100) : ((1 - tr.confidence) * 100);
    document.getElementById("tr-progress").style.width = `${Math.max(5, Math.min(95, trPct))}%`;
    document.getElementById("tr-pedagogy").textContent = tr.pedagogical_note;

    // Transformer tokens
    const trCloud = document.getElementById("tr-tokens-cloud");
    trCloud.innerHTML = "";
    tr.tokens.forEach(t => {
      const span = document.createElement("span");
      span.className = "token-pill";
      span.textContent = t;
      trCloud.appendChild(span);
    });
  }

  // --------------------------------------------------------------------------
  // Feature 2: Word Similarity Logic
  // --------------------------------------------------------------------------
  btnCalcSim.addEventListener("click", () => {
    const w1 = word1Input.value.trim();
    const w2 = word2Input.value.trim();
    calculateSimilarity(w1, w2);
  });

  simChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const w1 = chip.dataset.w1;
      const w2 = chip.dataset.w2;
      word1Input.value = w1;
      word2Input.value = w2;
      calculateSimilarity(w1, w2);
    });
  });

  async function calculateSimilarity(w1, w2) {
    if (!w1 || !w2) {
      oovWarning.textContent = "Please enter both words to compute cosine distance.";
      oovWarning.classList.remove("hidden");
      return;
    }

    try {
      const response = await fetch("/api/similarity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word1: w1, word2: w2 })
      });

      const res = await response.json();
      if (!res.success) {
        oovWarning.textContent = res.error || "Similarity calculation failed.";
        oovWarning.classList.remove("hidden");
        return;
      }

      const data = res.data;
      if (!data.in_vocab) {
        oovWarning.textContent = data.error_message;
        oovWarning.classList.remove("hidden");
        simScoreVal.textContent = "OOV";
        simInterpretationTitle.textContent = "Out-Of-Vocabulary Token";
        simInterpretationDesc.textContent = data.explanation;
        formulaSubs.textContent = "Cannot compute dot product with unrepresented vocabulary token.";
        vecW1Code.textContent = "No vector assigned (OOV)";
        vecW2Code.textContent = "No vector assigned (OOV)";
        return;
      }

      oovWarning.classList.add("hidden");
      simScoreVal.textContent = data.similarity.toFixed(2);
      simInterpretationTitle.textContent = `${data.word1} ↔ ${data.word2}: ${data.similarity >= 0.5 ? 'High Correlation' : data.similarity >= 0.2 ? 'Moderate Correlation' : 'Low/Orthogonal'}`;
      simInterpretationDesc.textContent = data.explanation;
      formulaSubs.innerHTML = `$\\mathbf{u} \\cdot \\mathbf{v} = ${data.dot_product}, \\quad \\|\\mathbf{u}\\| = ${data.norm1}, \\quad \\|\\mathbf{v}\\| = ${data.norm2}$`;

      vecW1Name.textContent = data.word1;
      vecW2Name.textContent = data.word2;
      vecW1Code.textContent = `[${data.vector_sample_1.join(", ")}...] (50D)`;
      vecW2Code.textContent = `[${data.vector_sample_2.join(", ")}...] (50D)`;

    } catch (err) {
      oovWarning.textContent = "Error communicating with embedding model server.";
      oovWarning.classList.remove("hidden");
    }
  }

  // --------------------------------------------------------------------------
  // Feature 3: Attention Explorer Logic
  // --------------------------------------------------------------------------
  attnModeSelect.addEventListener("change", (e) => {
    state.attnMode = e.target.value;
    attnModeBadge.textContent = state.attnMode === "real" ? "Live PyTorch Output" : "Illustrative Demo (Curated)";
    attnModeBadge.className = `badge-pill ${state.attnMode === "real" ? "badge-teal" : "badge-outline"}`;
    fetchAttentionData();
  });

  attnLayerSelect.addEventListener("change", (e) => {
    state.attnLayer = parseInt(e.target.value, 10);
    fetchAttentionData();
  });

  attnHeadSelect.addEventListener("change", (e) => {
    state.attnHead = parseInt(e.target.value, 10);
    fetchAttentionData();
  });

  btnRefreshAttn.addEventListener("click", () => fetchAttentionData());

  async function fetchAttentionData() {
    heatmapContainer.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Calculating self-attention matrix...</p></div>';

    try {
      const response = await fetch("/api/attention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: state.currentSentence,
          mode: state.attnMode,
          layer: state.attnLayer,
          head: state.attnHead
        })
      });

      const res = await response.json();
      if (!res.success) {
        heatmapContainer.innerHTML = `<div class="alert-banner error">${res.error || "Failed to load attention matrix."}</div>`;
        return;
      }

      renderHeatmap(res);

    } catch (err) {
      heatmapContainer.innerHTML = '<div class="alert-banner error">Error loading attention data from server.</div>';
    }
  }

  function renderHeatmap(data) {
    heatmapHeadTitle.textContent = `${data.layer_name} • ${data.head_name} [${data.label}]`;
    heatmapHeadDesc.textContent = data.description;

    const tokens = data.tokens;
    const matrix = data.attention_matrix;
    const n = tokens.length;

    let html = '<table class="heatmap-table"><thead><tr><th>Query \\ Key</th>';
    for (let j = 0; j < n; j++) {
      html += `<th>${tokens[j]}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (let i = 0; i < n; i++) {
      html += `<tr><th>${tokens[i]}</th>`;
      for (let j = 0; j < n; j++) {
        const val = matrix[i][j];
        // Interpolate teal color according to attention weight
        const r = Math.round(240 - val * (240 - 15));
        const g = Math.round(253 - val * (253 - 118));
        const b = Math.round(250 - val * (250 - 110));
        const textColor = val > 0.4 ? "#ffffff" : "#0f172a";

        html += `<td class="matrix-cell" data-row="${i}" data-col="${j}" data-val="${val}" style="background-color: rgb(${r}, ${g}, ${b}); color: ${textColor};" title="Query '${tokens[i]}' -> Key '${tokens[j]}': ${val.toFixed(3)}">${val.toFixed(2)}</td>`;
      }
      html += '</tr>';
    }
    html += '</tbody></table>';

    heatmapContainer.innerHTML = html;

    // Attach interactive hover & click listeners to cells
    const cells = heatmapContainer.querySelectorAll(".matrix-cell");
    cells.forEach(cell => {
      cell.addEventListener("mouseenter", () => {
        const r = parseInt(cell.dataset.row, 10);
        const c = parseInt(cell.dataset.col, 10);
        const val = parseFloat(cell.dataset.val);
        tokenInspectionPanel.innerHTML = `
          <strong>Focused Query Token:</strong> <code>"${tokens[r]}"</code> attending to Key <code>"${tokens[c]}"</code> with weight <strong>${(val * 100).toFixed(1)}%</strong>.
          <div class="mt-2 text-muted" style="font-size: 0.8rem;">
            Outgoing attention distribution from "${tokens[r]}": ${tokens.map((t, idx) => `${t}: ${(matrix[r][idx] * 100).toFixed(0)}%`).join(" | ")}
          </div>
        `;
      });
    });
  }

  // --------------------------------------------------------------------------
  // Feature 4: Results & Benchmark Logic
  // --------------------------------------------------------------------------
  async function fetchBenchmarkData() {
    try {
      const response = await fetch("/api/benchmark");
      const res = await response.json();
      if (!res.success) return;

      state.benchmarkData = res.benchmark;
      renderBenchmarkSummary(state.benchmarkData);
      renderBenchmarkTable(state.benchmarkData.detailed_rows);

    } catch (err) {
      console.error("Failed to load benchmark:", err);
    }
  }

  function renderBenchmarkSummary(bench) {
    const models = bench.models;
    // TF-IDF
    resTfidfAcc.textContent = `${models.tfidf.accuracy}%`;
    resTfidfCounts.textContent = `${models.tfidf.correct_count} / ${bench.total_samples} correct`;
    resTfidfTime.textContent = `${models.tfidf.avg_latency_ms} ms`;

    // Embedding
    resEmbAcc.textContent = `${models.embedding.accuracy}%`;
    resEmbCounts.textContent = `${models.embedding.correct_count} / ${bench.total_samples} correct`;
    resEmbTime.textContent = `${models.embedding.avg_latency_ms} ms`;

    // Transformer
    resTrAcc.textContent = `${models.transformer.accuracy}%`;
    resTrCounts.textContent = `${models.transformer.correct_count} / ${bench.total_samples} correct`;
    resTrTime.textContent = `${models.transformer.avg_latency_ms} ms`;
  }

  function renderBenchmarkTable(rows) {
    const filter = filterCategory.value;
    let filtered = rows;

    if (filter === "errors-only") {
      filtered = rows.filter(r => !r.tfidf.correct || !r.embedding.correct || !r.transformer.correct);
    } else if (filter !== "all") {
      filtered = rows.filter(r => r.category === filter);
    }

    if (filtered.length === 0) {
      benchmarkTbody.innerHTML = '<tr><td colspan="8" class="text-center p-4">No test cases match the selected filter.</td></tr>';
      return;
    }

    let html = "";
    filtered.forEach(row => {
      html += `
        <tr>
          <td><strong>#${row.id}</strong></td>
          <td style="max-width: 280px; font-weight: 500;">"${row.text}"</td>
          <td><span class="badge-pill">${row.category}</span></td>
          <td><strong class="${row.ground_truth === 'Positive' ? 'text-teal' : ''}">${row.ground_truth}</strong></td>
          <td>
            <span class="badge-status ${row.tfidf.correct ? 'pass' : 'fail'}">
              ${row.tfidf.prediction} (${(row.tfidf.confidence * 100).toFixed(0)}%)
            </span>
          </td>
          <td>
            <span class="badge-status ${row.embedding.correct ? 'pass' : 'fail'}">
              ${row.embedding.prediction} (${(row.embedding.confidence * 100).toFixed(0)}%)
            </span>
          </td>
          <td>
            <span class="badge-status ${row.transformer.correct ? 'pass' : 'fail'}">
              ${row.transformer.prediction} (${(row.transformer.confidence * 100).toFixed(0)}%)
            </span>
          </td>
          <td style="font-size: 0.8rem; color: var(--text-secondary); max-width: 220px;">${row.rationale}</td>
        </tr>
      `;
    });

    benchmarkTbody.innerHTML = html;
  }

  filterCategory.addEventListener("change", () => {
    if (state.benchmarkData) {
      renderBenchmarkTable(state.benchmarkData.detailed_rows);
    }
  });

  // --------------------------------------------------------------------------
  // Feature 5: Presentation Demo Scenarios Logic
  // --------------------------------------------------------------------------
  async function fetchDemoScenarios() {
    try {
      const response = await fetch("/api/demo-scenarios");
      const res = await response.json();
      if (!res.success) return;

      state.scenarios = res.scenarios;
      renderScenarios(state.scenarios);
    } catch (err) {
      console.error("Failed to load demo scenarios:", err);
    }
  }

  function renderScenarios(scenarios) {
    let html = "";
    scenarios.forEach(sc => {
      const isWordPair = Boolean(sc.word_pair);
      html += `
        <div class="scenario-card">
          <h4 class="scenario-title">${sc.title}</h4>
          <p class="text-muted" style="font-size: 0.82rem;"><strong>Objective:</strong> ${sc.learning_objective}</p>
          <div class="scenario-sentence-box">
            ${isWordPair ? `Pair: <code>"${sc.word_pair[0]}"</code> ↔ <code>"${sc.word_pair[1]}"</code>` : `Sentence: "${sc.sentence}"`}
          </div>
          <ul class="scenario-steps">
            ${sc.walkthrough.map(step => `<li>${step}</li>`).join("")}
          </ul>
          <button class="btn btn-sm btn-outline-teal mt-auto" onclick="window.launchScenario('${sc.id}')">
            ${isWordPair ? "Test in Word Similarity →" : "Run in Playground →"}
          </button>
        </div>
      `;
    });
    scenariosGrid.innerHTML = html;
  }

  // Global helper for launching scenarios
  window.launchScenario = function(scenarioId) {
    const sc = state.scenarios.find(s => s.id === scenarioId);
    if (!sc) return;

    if (sc.word_pair) {
      switchTab("similarity");
      word1Input.value = sc.word_pair[0];
      word2Input.value = sc.word_pair[1];
      calculateSimilarity(sc.word_pair[0], sc.word_pair[1]);
    } else {
      switchTab("playground");
      sentenceInput.value = sc.sentence;
      updateCharCount();
      runSentimentAnalysis(sc.sentence);
    }
  };

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------
  updateCharCount();
  fetchDemoScenarios();
  fetchBenchmarkData();
  // Auto-run initial representative sentence in playground
  runSentimentAnalysis(state.currentSentence);
});

