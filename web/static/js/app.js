const pages = {
  home: {
    kicker: "Platform",
    title: "Enterprise Retention Intelligence Workspace",
    html: `
      <div class="grid">
        <section class="card hero">
          <p class="eyebrow" style="color:#c7d7ff">RetainIQ Enterprise Operating System</p>
          <h2>Predict Churn. Uncover Drivers. Protect Revenue.</h2>
          <p>Upload your customer book, validate for data leakage, compare production ML algorithms using Stratified Cross-Validation, explain individual account churn drivers, and deploy targeted retention playbooks.</p>
          <div class="pills">
            <span class="pill">CSV &amp; Excel Intake</span>
            <span class="pill">Data Leakage Prevention</span>
            <span class="pill">Stratified 5-Fold CV</span>
            <span class="pill">PR-AUC Champion Selection</span>
            <span class="pill">Model Explainability</span>
            <span class="pill">Configurable Risk Thresholds</span>
            <span class="pill">Executive Excel &amp; PDF Reports</span>
          </div>
          <div class="top-actions">
            <button class="btn btn-primary" data-go="upload" type="button">Upload Customer Extract</button>
            <a class="btn btn-secondary" href="/api/sample">Load Sample Telco Extract</a>
          </div>
        </section>
        <section class="grid grid-3">
          <article class="card">
            <h3>1. Ingest &amp; Validate</h3>
            <p class="muted">Automatic schema detection, duplicate checking, target normalization, and post-churn data leakage prevention.</p>
          </article>
          <article class="card">
            <h3>2. Model Lab &amp; Cross-Val</h3>
            <p class="muted">Compare Logistic Regression, Random Forest, Gradient Boosting, and XGBoost on Stratified 5-Fold Cross-Validation.</p>
          </article>
          <article class="card">
            <h3>3. Explain &amp; Intervene</h3>
            <p class="muted">Identify top risk elevators and retention anchors for every customer, prescribe actions, and export executive reports.</p>
          </article>
        </section>
      </div>`
  },
  upload: {
    kicker: "Data Intake & Governance",
    title: "Upload & Validate Customer Extract",
    html: `
      <div class="grid grid-2">
        <section class="card">
          <h2>Upload Customer Book</h2>
          <p class="muted">Each upload operates in an isolated workspace. Preprocessing pipelines are fit strictly on this extract to ensure zero cross-client leakage.</p>
          <div class="dropzone" id="dropzone" style="margin-top:18px">
            <h3>Drag &amp; drop CSV or Excel file</h3>
            <p class="muted" style="margin:8px 0 16px">Required: Binary churn target (e.g. Churn, Exited, Attrition) with Yes/No or 1/0 values.</p>
            <button class="btn btn-primary" id="pick-file" type="button">Select File</button>
            <input class="file-input" id="file" type="file" accept=".csv,.xlsx,.xls" />
          </div>
          <p class="muted" id="upload-name" style="margin-top:12px; font-weight:600;"></p>
        </section>
        <section class="card">
          <h2>Data Quality &amp; Validation Report</h2>
          <p class="muted">Automated pre-flight checks run on upload before model training is permitted.</p>
          <div id="validation-box" style="margin-top:14px">
            <p class="muted">No dataset uploaded in this session yet.</p>
          </div>
          <div id="schema-box" style="margin-top:16px; padding-top:14px; border-top:1px solid var(--line)"></div>
        </section>
      </div>`
  },
  dashboard: {
    kicker: "Executive Overview",
    title: "Portfolio Health & Risk Exposure",
    html: `
      <div class="grid grid-4" id="kpis"></div>
      <div class="grid grid-2" style="margin-top:16px">
        <section class="card">
          <h3>Portfolio Composition (Active vs Churned)</h3>
          <div class="chart-wrap"><canvas id="mixChart"></canvas></div>
        </section>
        <section class="card">
          <h3>Churn Rate by Tenure Cohort</h3>
          <div class="chart-wrap"><canvas id="tenureChart"></canvas></div>
        </section>
      </div>`
  },
  segments: {
    kicker: "Diagnostics",
    title: "Segment Risk & Churn Lift Analysis",
    html: `
      <section class="card" style="margin-bottom:16px">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px">
          <div>
            <h2>Categorical Leak Points</h2>
            <p class="muted">Empirical churn rates and lift vs baseline portfolio. Highlights cohorts with statistically meaningful elevated risk.</p>
          </div>
          <div style="display:flex; gap:8px; align-items:center;">
            <span class="muted" style="font-size:12px; font-weight:700">Sort by:</span>
            <select id="seg-sort" style="min-height:36px; font-size:13px">
              <option value="lift">Highest Churn Lift</option>
              <option value="rate">Highest Churn Rate</option>
              <option value="population">Largest Population</option>
            </select>
          </div>
        </div>
      </section>
      <div id="segment-grid" class="grid grid-2"></div>
      <section class="card" style="margin-top:16px">
        <h3>Customer Extract Preview</h3>
        <p class="muted" style="margin-bottom:12px">First 15 records from the sanitized active dataset.</p>
        <div id="preview" style="overflow:auto"></div>
      </section>`
  },
  model: {
    kicker: "Model Lab & Governance",
    title: "Multi-Algorithm Benchmark & Cross-Validation",
    html: `
      <div class="grid grid-2">
        <section class="card">
          <h2>Train Retention Models</h2>
          <p class="muted">Fits Logistic Regression, Random Forest, Gradient Boosting, and XGBoost using Stratified 5-Fold Cross-Validation. Champion model is selected based on PR-AUC &amp; ROC-AUC composite.</p>
          <div style="margin-top:18px; display:flex; gap:10px; flex-wrap:wrap">
            <button class="btn btn-primary" id="train-btn" type="button">Train &amp; Benchmark Models</button>
            <button class="btn btn-secondary" data-go="insights" type="button">View Model Insights</button>
          </div>
          <p class="muted" id="train-status" style="margin-top:14px; font-weight:600;"></p>
        </section>
        <section class="card" id="model-summary">
          <h3>Champion Model</h3>
          <div class="empty-state">
            <div class="empty-icon">⚙️</div>
            <p class="muted">No model trained in this workspace yet.<br>Click <strong>Train &amp; Benchmark Models</strong> to begin evaluation.</p>
          </div>
        </section>
      </div>
      <section class="card" style="margin-top:16px" id="model-table">
        <h3>Comparative Performance Matrix</h3>
        <p class="muted">Comprehensive evaluation metrics on holdout test data and Stratified Cross-Validation.</p>
        <div id="model-table-content" style="margin-top:12px; overflow:auto"></div>
      </section>
      <div class="grid grid-2" style="margin-top:16px" id="model-deepdive"></div>`
  },
  insights: {
    kicker: "Model Explainability",
    title: "Global Predictive Drivers & Feature Attribution",
    html: `
      <div id="insights-container"></div>`
  },
  score: {
    kicker: "Activation & Scoring",
    title: "Customer Risk Scoring & Attribution Desk",
    html: `
      <div id="score-container"></div>`
  },
  playbook: {
    kicker: "Commercial Execution",
    title: "Evidence-Grounded Retention Playbook",
    html: `
      <section class="card" style="margin-bottom:16px">
        <h2>Strategic Retention Action Plan</h2>
        <p class="muted">Dynamic 4-horizon playbook derived from empirical high-risk cohorts and predictive model drivers.</p>
      </section>
      <div class="grid grid-2">
        <section class="card" id="plays-pillars"></section>
        <section class="card" id="plays-cadence"></section>
      </div>`
  },
  exports: {
    kicker: "Delivery & Reporting",
    title: "Executive Reports & CRM Exports",
    html: `
      <div class="grid grid-3">
        <section class="card">
          <h3>📊 Executive Excel Workbook</h3>
          <p class="muted">Multi-tab formatted spreadsheet (.xlsx) with Executive Summary, Model Governance, Segment Matrix, Drivers, Playbook, and Scored Accounts.</p>
          <a class="btn btn-primary" href="/api/export/excel" style="margin-top:16px">Download Excel (.xlsx)</a>
        </section>
        <section class="card">
          <h3>🖨️ Printable Executive Report</h3>
          <p class="muted">Standalone executive briefing document formatted for stakeholder sign-off and instant browser Print-to-PDF export.</p>
          <a class="btn btn-teal" href="/api/export/report" target="_blank" style="margin-top:16px">View / Print PDF Report</a>
        </section>
        <section class="card">
          <h3>📁 Scored Customer Book</h3>
          <p class="muted">Full customer dataset enriched with model churn probability, risk band tier, priority SLA, and recommended actions.</p>
          <a class="btn btn-secondary" href="/api/export/book" style="margin-top:16px">Download Scored CSV</a>
        </section>
      </div>`
  }
};

const state = { status: null, charts: {}, currentSegSort: "lift" };
const $ = (sel) => document.querySelector(sel);

function formatINR(amount) {
  if (amount === null || amount === undefined || isNaN(amount)) return "—";
  return "₹" + Number(amount).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function showAlert(message, type = "ok") {
  const el = $("#alert");
  if (!el) return;
  el.hidden = !message;
  el.className = `alert ${type}`;
  el.innerHTML = message ? `<span>${message}</span><button style="background:none;border:none;cursor:pointer;font-weight:bold" onclick="showAlert('')">✕</button>` : "";
}

async function api(url, options = {}) {
  const res = await fetch(url, { credentials: "include", ...options });
  if (!res.ok) {
    let detail = "Request failed";
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return res.json();
  return res;
}

function setStatus(status) {
  state.status = status;
  const el = $("#sidebar-status");
  if (!el) return;
  if (!status?.has_data) {
    el.innerHTML = `<span class="dot"></span><span>No dataset loaded</span>`;
    return;
  }
  const rate = ((status.kpis?.churn_rate || 0) * 100).toFixed(1);
  const rows = status.schema?.rows || status.kpis?.customers || 0;
  el.innerHTML = `<span class="dot ok"></span><span>${rows.toLocaleString()} customers · ${rate}% churn${status.has_model ? " · Model Ready" : ""}</span>`;
}

function renderPage(name) {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.page === name);
  });
  const spec = pages[name] || pages["home"];
  $("#page-kicker").textContent = spec.kicker;
  $("#page-title").textContent = spec.title;
  $("#pages").innerHTML = `<div class="page is-visible">${spec.html}</div>`;
  bindPage(name);
}

function bindGlobal() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      location.hash = btn.dataset.page;
    });
  });
  document.body.addEventListener("click", (event) => {
    const go = event.target.closest("[data-go]");
    if (go) location.hash = go.dataset.go;
  });
  $("#btn-reset").addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear this workspace? All uploaded data and trained models in this session will be wiped.")) return;
    try {
      await api("/api/reset", { method: "POST" });
      showAlert("Workspace successfully cleared. Upload a customer extract to start.", "ok");
      await refreshStatus();
      location.hash = "upload";
    } catch (err) {
      showAlert(err.message, "err");
    }
  });
}

function bindPage(name) {
  if (name === "upload") bindUpload();
  if (name === "dashboard") loadDashboard();
  if (name === "segments") loadSegments();
  if (name === "model") bindModel();
  if (name === "insights") loadInsights();
  if (name === "score") bindScore();
  if (name === "playbook") loadPlaybook();
}

function bindUpload() {
  const input = $("#file");
  const zone = $("#dropzone");
  $("#pick-file").addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("is-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("is-over");
    if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) uploadFile(input.files[0]);
  });
  paintValidationAndSchema(state.status);
}

async function uploadFile(file) {
  showAlert(`Uploading and validating ${file.name}…`, "ok");
  const body = new FormData();
  body.append("file", file);
  try {
    const result = await api("/api/upload", { method: "POST", body });
    await refreshStatus();
    paintValidationAndSchema(state.status);
    showAlert(`✓ ${file.name} ingested (${result.schema.rows.toLocaleString()} accounts). Data quality validation passed.`, "ok");
    $("#upload-name").textContent = `Active File: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  } catch (err) {
    showAlert(err.message, "err");
  }
}

function paintValidationAndSchema(status) {
  const vBox = $("#validation-box");
  const sBox = $("#schema-box");
  if (!vBox || !sBox) return;

  if (!status?.has_data) {
    vBox.innerHTML = `<p class="muted">No dataset uploaded in this session yet.</p>`;
    sBox.innerHTML = "";
    return;
  }

  const v = status.validation;
  if (v && v.checks) {
    vBox.innerHTML = `
      <div style="margin-bottom:12px; display:flex; justify-content:space-between; align-items:center">
        <strong>Validation Status: <span class="badge ${v.status === 'valid' ? 'success' : (v.status === 'warning' ? 'warning' : 'danger')}">${v.status.toUpperCase()}</span></strong>
        <span class="muted" style="font-size:12px">${v.checks.length} automated checks</span>
      </div>
      <div>
        ${v.checks.map(c => `
          <div class="check-item">
            <div class="check-icon ${c.status}">${c.status === 'pass' ? '✓' : (c.status === 'warning' ? '!' : '✕')}</div>
            <div>
              <strong>${c.name}</strong>
              <div class="muted" style="font-size:12px">${c.message}</div>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  }

  const s = status.schema;
  if (s) {
    sBox.innerHTML = `
      <h3>Detected Column Architecture</h3>
      <p class="muted" style="margin-top:4px">Target: <strong>${s.target || 'None'}</strong> · Identifier: <strong>${s.id_column || 'Row Index'}</strong></p>
      <div style="margin-top:10px; font-size:12.5px; line-height:1.6">
        <div><strong>Numeric Predictors (${s.numeric.length}):</strong> <span class="muted">${s.numeric.join(", ") || "None"}</span></div>
        <div style="margin-top:4px"><strong>Categorical Predictors (${s.categorical.length}):</strong> <span class="muted">${s.categorical.join(", ") || "None"}</span></div>
        ${s.leakage_columns && s.leakage_columns.length ? `<div style="margin-top:4px; color:#b91c1c"><strong>Leakage Excluded (${s.leakage_columns.length}):</strong> ${s.leakage_columns.join(", ")}</div>` : ''}
      </div>
    `;
  }
}

function destroyCharts() {
  Object.values(state.charts).forEach((chart) => {
    try { chart.destroy(); } catch (_) {}
  });
  state.charts = {};
}

async function loadDashboard() {
  destroyCharts();
  if (!state.status?.has_data) {
    $("#kpis").innerHTML = `<div class="card empty-state" style="grid-column:1/-1"><div class="empty-icon">📁</div><h3>No Dataset Loaded</h3><p class="muted">Upload a customer CSV or Excel file to view portfolio health KPIs.</p><button class="btn btn-primary" style="margin-top:14px" data-go="upload">Go to Upload</button></div>`;
    return;
  }

  try {
    const data = await api("/api/analytics");
    const status = await api("/api/status");
    const k = status.kpis;

    $("#kpis").innerHTML = [
      ["Portfolio Accounts", k.customers.toLocaleString("en-IN"), `${k.retained.toLocaleString("en-IN")} active / ${k.churners.toLocaleString("en-IN")} churned`],
      ["Baseline Churn Rate", `${(k.churn_rate * 100).toFixed(1)}%`, "Historical portfolio benchmark"],
      ["Monthly Revenue at Risk", k.monthly_revenue_at_risk ? formatINR(k.monthly_revenue_at_risk) : "—", "Direct recurring monthly exposure (INR)"],
      ["Annualized Revenue at Risk", k.annual_revenue_at_risk ? formatINR(k.annual_revenue_at_risk) : "—", "12-month run-rate revenue exposure (INR)"]
    ].map(([label, value, sub]) => `
      <article class="card">
        <div class="kpi-label">${label}</div>
        <div class="kpi-value">${value}</div>
        <div class="kpi-sub">${sub}</div>
      </article>`).join("");

    state.charts.mix = new Chart($("#mixChart"), {
      type: "doughnut",
      data: {
        labels: data.churn_mix.map((d) => d.label),
        datasets: [{ data: data.churn_mix.map((d) => d.value), backgroundColor: ["#12b76a", "#d92d20"] }]
      },
      options: { plugins: { legend: { position: "bottom" } }, maintainAspectRatio: false }
    });

    if (data.tenure_bins.length) {
      state.charts.tenure = new Chart($("#tenureChart"), {
        type: "bar",
        data: {
          labels: data.tenure_bins.map((d) => d.bucket),
          datasets: [{ label: "Churn Rate (%)", data: data.tenure_bins.map((d) => d.churn_rate * 100), backgroundColor: "#1d4ed8", borderRadius: 6 }]
        },
        options: {
          scales: { y: { ticks: { callback: (v) => v + "%" }, beginAtZero: true } },
          maintainAspectRatio: false
        }
      });
    }
  } catch (err) {
    showAlert(err.message, "err");
  }
}

async function loadSegments() {
  if (!state.status?.has_data) {
    $("#segment-grid").innerHTML = `<div class="card empty-state" style="grid-column:1/-1"><div class="empty-icon">📁</div><h3>No Dataset Loaded</h3><p class="muted">Upload customer data to diagnose categorical risk concentrations.</p></div>`;
    return;
  }

  try {
    const data = await api("/api/analytics");
    const renderSegs = () => {
      const sortMode = $("#seg-sort") ? $("#seg-sort").value : "lift";
      let segList = [...data.segments];

      $("#segment-grid").innerHTML = segList.slice(0, 6).map((seg) => {
        let rows = [...seg.rows];
        if (sortMode === "rate") rows.sort((a, b) => b.churn_rate - a.churn_rate);
        else if (sortMode === "population") rows.sort((a, b) => b.customers - a.customers);
        else rows.sort((a, b) => b.lift - a.lift);

        return `
          <section class="card">
            <h3>${seg.column}</h3>
            <table>
              <thead><tr><th>Cohort</th><th>Accounts</th><th>Churn Rate</th><th>Lift vs Baseline</th></tr></thead>
              <tbody>
                ${rows.map((row) => `
                  <tr>
                    <td><strong>${row.segment}</strong> ${row.is_high_risk ? '<span class="badge danger">High Risk</span>' : ''}</td>
                    <td>${row.customers.toLocaleString()}</td>
                    <td><strong>${(row.churn_rate * 100).toFixed(1)}%</strong></td>
                    <td style="color:${row.lift > 0.05 ? '#b91c1c' : (row.lift < -0.05 ? '#15803d' : '#475569')}; font-weight:700">
                      ${row.lift > 0 ? '+' : ''}${(row.lift * 100).toFixed(1)}%
                    </td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </section>
        `;
      }).join("");
    };

    renderSegs();
    if ($("#seg-sort")) $("#seg-sort").addEventListener("change", renderSegs);

    const cols = data.preview_columns;
    $("#preview").innerHTML = `
      <table>
        <thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
        <tbody>${data.preview.map((row) => `<tr>${cols.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>`;
  } catch (err) {
    showAlert(err.message, "err");
  }
}

function bindModel() {
  paintModel(state.status?.metrics);
  $("#train-btn").addEventListener("click", async () => {
    $("#train-btn").disabled = true;
    $("#train-status").innerHTML = `<span>⏳ Running Stratified 5-Fold Cross-Validation across 4 algorithms…</span>`;
    try {
      const metrics = await api("/api/train", { method: "POST" });
      await refreshStatus();
      paintModel(metrics);
      showAlert(`✓ Champion Selected: ${metrics.best_model_display}. Cross-validation & holdout evaluation complete.`, "ok");
    } catch (err) {
      showAlert(err.message, "err");
    } finally {
      $("#train-btn").disabled = false;
      $("#train-status").textContent = "";
    }
  });
}

function paintModel(metrics) {
  if (!metrics) {
    $("#model-summary").innerHTML = `
      <h3>Champion Model</h3>
      <div class="empty-state">
        <div class="empty-icon">⚙️</div>
        <p class="muted">No model trained in this workspace yet.<br>Click <strong>Train &amp; Benchmark Models</strong> to begin evaluation.</p>
      </div>`;
    $("#model-table-content").innerHTML = `<p class="muted">Metrics will appear here after model training.</p>`;
    $("#model-deepdive").innerHTML = "";
    return;
  }

  const champion = metrics.models[metrics.best_model];

  $("#model-summary").innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:flex-start">
      <div>
        <span class="badge champion">Selected Champion</span>
        <div class="kpi-value" style="margin-top:6px">${metrics.best_model_display}</div>
        <p class="muted" style="margin-top:4px">Version ${metrics.model_version} · Trained on ${metrics.rows_train.toLocaleString()} records</p>
      </div>
      <div style="text-align:right">
        <div class="kpi-label">Holdout PR-AUC</div>
        <div class="kpi-value" style="color:var(--blue)">${champion.pr_auc.toFixed(4)}</div>
      </div>
    </div>
    <div style="margin-top:16px; padding-top:14px; border-top:1px solid var(--line); font-size:12.5px" class="muted">
      <div><strong>Selection Metric:</strong> ${metrics.selection_metric}</div>
      <div><strong>Trained At:</strong> ${metrics.trained_at_formatted}</div>
      <div><strong>Cross-Validation:</strong> Stratified 5-Fold (${champion.cv_pr_auc_mean.toFixed(4)} ± ${champion.cv_pr_auc_std.toFixed(4)} PR-AUC)</div>
    </div>
  `;

  const rows = Object.entries(metrics.models).map(([name, m]) => {
    const isChamp = name === metrics.best_model;
    return `
      <tr style="${isChamp ? 'background:#f0fdf4; font-weight:700' : ''}">
        <td><strong>${name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong> ${isChamp ? '<span class="badge champion">Champion</span>' : ''}</td>
        <td><span class="badge ${m.training_status === 'Completed' ? 'success' : 'warning'}">${m.training_status}</span></td>
        <td><strong>${m.roc_auc.toFixed(4)}</strong></td>
        <td><strong style="color:var(--blue)">${m.pr_auc.toFixed(4)}</strong></td>
        <td>${(m.accuracy * 100).toFixed(1)}%</td>
        <td>${m.precision.toFixed(4)}</td>
        <td>${m.recall.toFixed(4)}</td>
        <td>${m.f1.toFixed(4)}</td>
        <td>${m.cv_pr_auc_mean.toFixed(4)} ± ${m.cv_pr_auc_std.toFixed(4)}</td>
      </tr>`;
  }).join("");

  $("#model-table-content").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Algorithm</th>
          <th>Status</th>
          <th>ROC-AUC</th>
          <th>PR-AUC</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1 Score</th>
          <th>5-Fold CV PR-AUC</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;

  const cm = champion.confusion_matrix;
  $("#model-deepdive").innerHTML = `
    <section class="card">
      <h3>Champion Confusion Matrix (Test Split: ${metrics.rows_test.toLocaleString()} accounts)</h3>
      <p class="muted">Holdout test split breakdown comparing predicted vs actual churn.</p>
      <div class="cm-grid">
        <div class="cm-cell positive">
          <div class="cm-label">True Negatives (Retained)</div>
          <div class="cm-val">${cm.tn.toLocaleString()}</div>
          <div class="cm-sub">${(cm.tn_rate * 100).toFixed(1)}% of actual retained</div>
        </div>
        <div class="cm-cell negative">
          <div class="cm-label">False Positives (False Alarm)</div>
          <div class="cm-val">${cm.fp.toLocaleString()}</div>
          <div class="cm-sub">${(cm.fp_rate * 100).toFixed(1)}% false positive rate</div>
        </div>
        <div class="cm-cell negative">
          <div class="cm-label">False Negatives (Missed Churn)</div>
          <div class="cm-val">${cm.fn.toLocaleString()}</div>
          <div class="cm-sub">${(cm.fn_rate * 100).toFixed(1)}% missed churn rate</div>
        </div>
        <div class="cm-cell positive">
          <div class="cm-label">True Positives (Saved Churn)</div>
          <div class="cm-val">${cm.tp.toLocaleString()}</div>
          <div class="cm-sub">${(cm.tp_rate * 100).toFixed(1)}% of actual churners caught</div>
        </div>
      </div>
    </section>
    <section class="card">
      <h3>Model Governance &amp; Metadata</h3>
      <div style="font-size:13px; line-height:1.7; margin-top:8px">
        <div><strong>Active Model:</strong> ${metrics.best_model_display} (Pipeline v${metrics.model_version})</div>
        <div><strong>Dataset:</strong> ${metrics.dataset_filename} (${metrics.rows_total.toLocaleString()} total rows)</div>
        <div><strong>Train / Test Split:</strong> ${metrics.rows_train.toLocaleString()} train (${(100 - metrics.rows_test / metrics.rows_total * 100).toFixed(0)}%) / ${metrics.rows_test.toLocaleString()} test (${(metrics.rows_test / metrics.rows_total * 100).toFixed(0)}%)</div>
        <div><strong>Test Positive Churners:</strong> ${metrics.test_churn_count.toLocaleString()} (${(metrics.test_churn_count / metrics.rows_test * 100).toFixed(1)}%)</div>
        <div><strong>Cross-Validation Folds:</strong> 5-Fold Stratified K-Fold</div>
      </div>
      <div style="margin-top:16px">
        <a class="btn btn-teal" href="/api/export/excel">Download Model Validation Workbook</a>
      </div>
    </section>
  `;
}

async function loadInsights() {
  const container = $("#insights-container");
  if (!container) return;

  if (!state.status?.has_data) {
    container.innerHTML = `<div class="card empty-state"><div class="empty-icon">📁</div><h3>No Dataset Loaded</h3><p class="muted">Upload customer data first to view model insights.</p></div>`;
    return;
  }
  if (!state.status?.has_model) {
    container.innerHTML = `
      <div class="card empty-state">
        <div class="empty-icon">⚙️</div>
        <h3>Model Not Trained Yet</h3>
        <p class="muted">Train a machine learning model to compute global feature importance and attribution drivers.</p>
        <button class="btn btn-primary" style="margin-top:14px" data-go="model">Go to Model Lab</button>
      </div>`;
    return;
  }

  try {
    const data = await api("/api/insights");
    const drivers = data.global_drivers || [];

    container.innerHTML = `
      <div class="grid grid-2">
        <section class="card">
          <h2>Top Global Churn Drivers</h2>
          <p class="muted">Ranked feature importance indicating overall influence on the model's churn risk scores.</p>
          <div style="margin-top:16px">
            ${drivers.slice(0, 8).map(d => `
              <div style="margin-bottom:14px">
                <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px">
                  <strong>#${d.rank} ${d.feature} <span class="badge">${d.feature_type}</span></strong>
                  <span style="font-weight:700; color:var(--blue)">${d.importance_pct}%</span>
                </div>
                <div class="meter"><span style="width:${d.importance_pct * 2.5}%"></span></div>
                <p class="muted" style="font-size:11.5px; margin-top:3px">${d.description}</p>
              </div>
            `).join("")}
          </div>
        </section>
        <section class="card">
          <h2>Feature Attribution Guide</h2>
          <p class="muted">How RetainIQ computes account-level risk contributors.</p>
          <div class="steps" style="margin-top:16px">
            <div class="step"><b>1</b><div><strong>Risk Elevators (+)</strong><p class="muted">Account characteristics that increase predicted churn probability (e.g. Month-to-month contracts, electronic check payments, high monthly charge ratio).</p></div></div>
            <div class="step"><b>2</b><div><strong>Retention Anchors (−)</strong><p class="muted">Protective factors that reduce churn probability (e.g. Long-tenure maturity, annual/multi-year commitments, automated billing, active tech support).</p></div></div>
            <div class="step"><b>3</b><div><strong>Action Prescriptions</strong><p class="muted">Direct operational interventions matched against the dominant risk elevators to maximize save efficiency.</p></div></div>
          </div>
          <div style="margin-top:24px; padding-top:16px; border-top:1px solid var(--line)">
            <button class="btn btn-primary" data-go="score" type="button">Score Individual Customers</button>
          </div>
        </section>
      </div>
    `;
  } catch (err) {
    showAlert(err.message, "err");
  }
}

async function bindScore() {
  const container = $("#score-container");
  if (!container) return;

  if (!state.status?.has_data) {
    container.innerHTML = `<div class="card empty-state"><div class="empty-icon">📁</div><h3>No Dataset Loaded</h3><p class="muted">Upload customer data before accessing risk scoring.</p><button class="btn btn-primary" style="margin-top:14px" data-go="upload">Go to Upload</button></div>`;
    return;
  }

  if (!state.status?.has_model) {
    container.innerHTML = `
      <div class="card empty-state">
        <div class="empty-icon">🔒</div>
        <h2>Model Training Required</h2>
        <p class="muted" style="max-width:480px; margin:8px auto 16px;">Train a model before generating churn predictions. Single customer and batch scoring require an active trained champion model.</p>
        <button class="btn btn-primary" data-go="model" type="button">Train a Model Now</button>
      </div>`;
    return;
  }

  const th = state.status.thresholds || { high: 0.60, medium: 0.35 };

  container.innerHTML = `
    <section class="card" style="margin-bottom:16px">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px">
        <div>
          <h2>Configurable Risk Thresholds</h2>
          <p class="muted">Tailor risk classification tiers to match your intervention economics and retention capacity.</p>
        </div>
        <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap">
          <label style="display:flex; align-items:center; gap:6px; font-size:12px">
            <span>High Risk ≥</span>
            <input type="number" id="th-high" min="0.40" max="0.95" step="0.05" value="${th.high}" style="width:75px; min-height:36px" />
          </label>
          <label style="display:flex; align-items:center; gap:6px; font-size:12px">
            <span>Medium Risk ≥</span>
            <input type="number" id="th-med" min="0.10" max="0.60" step="0.05" value="${th.medium}" style="width:75px; min-height:36px" />
          </label>
          <button class="btn btn-secondary" id="save-thresholds" type="button" style="min-height:36px">Apply Thresholds</button>
        </div>
      </div>
    </section>

    <div class="grid grid-2">
      <section class="card">
        <h2>Single-Account Scoring Desk</h2>
        <p class="muted">Live interactive form dynamically mapped to the active dataset schema.</p>
        <form id="score-form" class="form-grid" style="margin-top:16px"></form>
        <div style="margin-top:18px">
          <button class="btn btn-primary" id="score-one" type="button">Generate Churn Risk Score</button>
        </div>
      </section>

      <section class="card" id="score-result-card">
        <h2>Risk Assessment &amp; Action Plan</h2>
        <div id="score-result">
          <div class="empty-state">
            <div class="empty-icon">🎯</div>
            <p class="muted">Fill in account details and click <strong>Generate Churn Risk Score</strong> to see probability, key drivers, and retention recommendations.</p>
          </div>
        </div>
        <div style="margin-top:24px; padding-top:18px; border-top:1px solid var(--line)">
          <h3>Batch Customer Scoring</h3>
          <p class="muted">Upload prospective customer file (CSV or Excel). <strong>Does NOT require a Churn column.</strong></p>
          <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap">
            <button class="btn btn-teal" id="batch-btn" type="button">Upload Batch File to Score</button>
            <input class="file-input" id="batch-file" type="file" accept=".csv,.xlsx,.xls" />
          </div>
        </div>
      </section>
    </div>
  `;

  // Threshold update binding
  $("#save-thresholds").addEventListener("click", async () => {
    const high = parseFloat($("#th-high").value);
    const med = parseFloat($("#th-med").value);
    try {
      await api("/api/thresholds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ high, medium: med })
      });
      await refreshStatus();
      showAlert(`✓ Risk thresholds updated: High ≥ ${(high*100).toFixed(0)}%, Medium ≥ ${(med*100).toFixed(0)}%`, "ok");
    } catch (err) {
      showAlert(err.message, "err");
    }
  });

  // Populate dynamic form
  try {
    const fields = await api("/api/fields");
    $("#score-form").innerHTML = fields.fields.map((field) => {
      if (field.type === "number") {
        const isMoney = /charge|monthly|total|spend|revenue|price|cost|fee/i.test(field.name);
        const labelText = isMoney ? `${field.name} (₹ / Rs.)` : field.name;
        return `<label>${labelText}<input name="${field.name}" type="number" step="any" value="${field.default}" /></label>`;
      }
      const opts = field.options.map((opt) => `<option ${opt === field.default ? 'selected' : ''}>${opt}</option>`).join("");
      return `<label>${field.name}<select name="${field.name}">${opts}</select></label>`;
    }).join("");
  } catch (err) {
    showAlert(err.message, "err");
  }

  // Score single customer
  $("#score-one").addEventListener("click", async () => {
    const payload = {};
    $("#score-form").querySelectorAll("input, select").forEach((el) => {
      payload[el.name] = el.type === "number" ? Number(el.value) : el.value;
    });
    try {
      const result = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const exp = result.local_explanation || {};
      const elevators = exp.top_risk_elevators || [];
      const anchors = exp.top_retention_anchors || [];

      $("#score-result").innerHTML = `
        <div style="margin-top:10px">
          <div style="display:flex; justify-content:space-between; align-items:flex-start">
            <div>
              <div class="kpi-label">Predicted Churn Probability</div>
              <div class="kpi-value" style="color:${result.risk_band === 'High' ? 'var(--danger)' : (result.risk_band === 'Medium' ? 'var(--gold)' : 'var(--ok)')}">
                ${result.churn_probability_pct}%
              </div>
            </div>
            <div style="text-align:right">
              <span class="risk ${result.risk_band}">${result.risk_band} Risk</span>
              <div style="font-size:12px; font-weight:700; margin-top:6px; color:var(--muted)">${result.predicted_status}</div>
            </div>
          </div>
          <div class="meter" style="margin:12px 0"><span style="width:${result.churn_probability_pct}%"></span></div>

          <div class="grid grid-2" style="margin:14px 0">
            <div style="background:#f8fafc; padding:10px; border-radius:8px; font-size:12px">
              <strong>Priority SLA:</strong><br>${result.priority}
            </div>
            <div style="background:#f8fafc; padding:10px; border-radius:8px; font-size:12px">
              <strong>Annual Rev at Risk:</strong><br>${result.annual_revenue_at_risk ? `${formatINR(result.annual_revenue_at_risk)}/yr` : '—'}
            </div>
          </div>

          <div style="margin-top:14px">
            <strong style="font-size:12px; text-transform:uppercase; color:var(--muted)">Top Risk Elevators (Pushing to Churn)</strong>
            <div style="margin-top:6px">
              ${elevators.map(e => `
                <div class="factor-pill elevator">
                  <span>🔺 ${e.label}</span>
                  <span style="font-weight:700">+${e.impact_pct}%</span>
                </div>
              `).join("") || "<p class='muted' style='font-size:12px'>None identified.</p>"}
            </div>
          </div>

          ${anchors.length ? `
            <div style="margin-top:12px">
              <strong style="font-size:12px; text-transform:uppercase; color:var(--muted)">Retention Anchors (Reducing Churn Risk)</strong>
              <div style="margin-top:6px">
                ${anchors.map(a => `
                  <div class="factor-pill anchor">
                    <span>🛡️ ${a.label}</span>
                    <span style="font-weight:700">-${a.impact_pct}%</span>
                  </div>
                `).join("")}
              </div>
            </div>
          ` : ''}

          <div style="margin-top:16px; background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:12px">
            <div style="font-size:11px; text-transform:uppercase; font-weight:800; color:#1e40af">🎯 Recommended Retention Action</div>
            <div style="font-size:13px; font-weight:700; color:#1e3a8a; margin-top:4px">${result.recommended_action}</div>
          </div>
        </div>`;
    } catch (err) {
      showAlert(err.message, "err");
    }
  });

  // Batch scoring upload
  $("#batch-btn").addEventListener("click", () => $("#batch-file").click());
  $("#batch-file").addEventListener("change", async () => {
    const file = $("#batch-file").files[0];
    if (!file) return;
    showAlert(`Scoring ${file.name}…`, "ok");
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await api("/api/predict-batch", { method: "POST", body });
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "retainiq_scored_customers.csv";
      a.click();
      showAlert(`✓ Successfully batch-scored ${file.name}. Scored CSV downloaded.`, "ok");
    } catch (err) {
      showAlert(err.message, "err");
    }
  });
}

async function loadPlaybook() {
  if (!state.status?.has_data) {
    $("#plays-pillars").innerHTML = `<div class="empty-state"><p class="muted">Upload data to generate playbook.</p></div>`;
    return;
  }

  try {
    const data = await api("/api/insights");
    const pb = data.playbook || {};

    $("#plays-pillars").innerHTML = `
      <h2>Strategic Retention Pillars</h2>
      <p class="muted" style="margin-bottom:14px">Prioritized intervention tracks grounded in portfolio risk data.</p>
      ${(pb.strategic_pillars || []).map(p => `
        <div style="margin-bottom:18px; padding-bottom:14px; border-bottom:1px solid var(--line)">
          <div style="display:flex; justify-content:space-between; align-items:center">
            <span class="badge ${p.badge_color}">${p.priority}</span>
            <span style="font-size:12px; font-weight:700; color:var(--muted)">${p.annual_revenue_at_risk}</span>
          </div>
          <h3 style="margin-top:6px">${p.pillar_name}</h3>
          <p style="font-size:12.5px"><strong>Target Cohort:</strong> ${p.target_segment} (${p.affected_accounts}) · <em>${p.observed_metric}</em></p>
          <p class="muted" style="font-size:12px; margin:6px 0">${p.strategic_rationale}</p>
          <ul style="padding-left:18px; font-size:12.5px; margin-top:6px; color:#334155">
            ${p.recommended_actions.map(act => `<li style="margin-bottom:3px">${act}</li>`).join("")}
          </ul>
        </div>
      `).join("")}
    `;

    $("#plays-cadence").innerHTML = `
      <h2>Operating Cadence</h2>
      <p class="muted" style="margin-bottom:14px">Recommended execution schedule for retention teams.</p>
      ${(pb.operational_cadence || []).map(c => `
        <div style="margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid var(--line)">
          <span class="badge info">${c.frequency}</span>
          <h4 style="margin-top:6px">${c.title}</h4>
          <p class="muted" style="font-size:12.5px; margin-top:2px">${c.action}</p>
        </div>
      `).join("")}
    `;
  } catch (err) {
    showAlert(err.message, "err");
  }
}

async function refreshStatus() {
  try {
    state.status = await api("/api/status");
    setStatus(state.status);
  } catch (_) {}
}

function route() {
  showAlert("");
  const name = (location.hash || "#home").replace("#", "");
  renderPage(pages[name] ? name : "home");
}

window.addEventListener("hashchange", route);
bindGlobal();
refreshStatus().then(route);
