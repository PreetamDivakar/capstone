// benchmark.js
// - Fetches /benchmark-data
// - Renders a Chart.js chart (graph) and also renders confusion matrix and precision/recall metrics
// - Provides three toggle buttons to display the desired content

// Helpers
function el(id) { return document.getElementById(id); }
function hide(...els) { els.forEach(e => { if (e) e.style.display = 'none'; }); }
function show(e, display = 'block') { if (e) e.style.display = display; }
function setHidden(node, isHidden) { if (node) node.hidden = !!isHidden; }
function toastError(msg) {
  const t = el('errorToast');
  if (!t) return;
  t.textContent = msg || 'Something went wrong.';
  setHidden(t, false);
  setTimeout(() => setHidden(t, true), 3500);
}

// Render helpers
function computeConfusionMatrix(actual, predicted) {
  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0;i<actual.length;i++){
    if (actual[i] === 1 && predicted[i] === 1) tp++;
    if (actual[i] === 0 && predicted[i] === 0) tn++;
    if (actual[i] === 0 && predicted[i] === 1) fp++;
    if (actual[i] === 1 && predicted[i] === 0) fn++;
  }
  return { tp, tn, fp, fn };
}

function computeMetrics(cm) {
  const {tp,tn,fp,fn} = cm;
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  return { precision, recall, f1, accuracy };
}

function percentile(values, p) {
  if (!values || values.length === 0) return 0;
  const arr = [...values].sort((a,b)=>a-b);
  const idx = Math.min(arr.length-1, Math.max(0, Math.round((p/100) * (arr.length-1))));
  return arr[idx];
}

let chartInstance = null;
function renderSimpleChart(ctx, labels, values, metric) {
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  const conf = {
    Efficiency: {
      color: 'rgba(34,197,94,0.9)',
      title: 'Efficiency (%)',
      format: v => `${v} %`
    },
    Cost: {
      color: 'rgba(96,165,250,0.9)',
      title: 'Cost ($)',
      format: v => `$${v}`
    },
    Latency: {
      color: 'rgba(249,115,22,0.9)',
      title: 'Latency (ms)',
      format: v => `${v} ms`
    }
  };
  const cfg = conf[metric] || conf.Efficiency;

  chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: cfg.title,
        data: values,
        backgroundColor: [cfg.color, cfg.color, cfg.color],
        borderRadius: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#7dd3fc' }, grid: { display: false } },
        y: {
          beginAtZero: true,
          ticks: { color: '#9aa4b2' },
          title: { display: true, text: cfg.title, color: '#7dd3fc' }
        }
      },
      plugins: {
        legend: { labels: { color: '#7dd3fc' } },
        tooltip: {
          callbacks: {
            label: (c) => `${cfg.title}: ${cfg.format(c.parsed.y)}`
          }
        }
      }
    }
  });
}

function renderConfusion(container, cm) {
  container.innerHTML = '';
  const table = document.createElement('table');
  table.className = 'confusion-table';
  table.innerHTML = `
    <thead>
      <tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>
    </thead>
    <tbody>
      <tr><th>Actual 0</th><td>${cm.tn}</td><td>${cm.fp}</td></tr>
      <tr><th>Actual 1</th><td>${cm.fn}</td><td>${cm.tp}</td></tr>
    </tbody>
  `;
  container.appendChild(table);
}

function renderMetrics(container, metrics) {
  container.innerHTML = '';
  const wrap = document.createElement('div');
  wrap.className = 'metrics-list';
  wrap.innerHTML = `
    <div class="metrics-item">Precision: ${metrics.precision.toFixed(3)}</div>
    <div class="metrics-item">Recall: ${metrics.recall.toFixed(3)}</div>
    <div class="metrics-item">F1 score: ${metrics.f1.toFixed(3)}</div>
    <div class="metrics-item">Accuracy: ${metrics.accuracy.toFixed(3)}</div>
  `;
  container.appendChild(wrap);
}

function renderSummary(panel, data) {
  if (!panel) return;
  const costs = data.cost || [];
  const lat = data.latency_ms || [];
  const avg = (arr)=> (arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length) : 0);
  const costAvg = avg(costs);
  const latP50 = percentile(lat, 50);
  const latP25 = percentile(lat, 25);
  panel.innerHTML = `
    <div class="summary-grid">
      <div class="badge" role="status" aria-label="Average cost">
        <div class="badge-title">Avg Cost</div>
        <div class="badge-value">$${costAvg.toFixed(2)}</div>
      </div>
      <div class="badge" role="status" aria-label="p50 latency">
        <div class="badge-title">p50 Latency</div>
        <div class="badge-value">${latP50.toFixed(0)} ms</div>
      </div>
      <div class="badge" role="status" aria-label="p25 latency">
        <div class="badge-title">p25 Latency</div>
        <div class="badge-value">${latP25.toFixed(0)} ms</div>
      </div>
    </div>
  `;
  setHidden(panel, false);
}

// Toggle handlers
function setActive(btn) {
  document.querySelectorAll('.bbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.bbtn[role="tab"]').forEach(b => b.setAttribute('aria-selected', String(b === btn)));
}

window.addEventListener('DOMContentLoaded', async () => {
  const showGraphBtn = el('showGraphBtn');
  const showConfusionBtn = el('showConfusionBtn');
  const showMetricsBtn = el('showMetricsBtn');
  const metricCostBtn = el('metricCostBtn');
  const metricLatencyBtn = el('metricLatencyBtn');
  const metricEfficiencyBtn = el('metricEfficiencyBtn');

  const chartCanvas = el('benchmarkChart');
  const confusionContainer = el('confusionContainer');
  const metricsContainer = el('metricsContainer');
  const chartPanel = el('chartPanel');
  const summaryPanel = el('summaryPanel');

  // Start by fetching benchmark data
  let data;
  try {
    const res = await fetch('/benchmark-data');
    if (!res.ok) throw new Error('Failed to load benchmark data');
    data = await res.json();
  } catch (err) {
    console.error('Could not load benchmark data', err);
    toastError('Failed to load benchmark data.');
    return;
  }

  // Pre-compute confusion matrix & metrics from arrays
  const actual = data.actual || [];
  const predicted = data.predicted || [];
  const cm = computeConfusionMatrix(actual, predicted);
  const metrics = computeMetrics(cm);

  // Determine labels – force to our three comparison categories
  let labels = ['Serverless', 'Traditional', 'Hybrid'];

  // Base arrays from backend
  let costArr = Array.isArray(data.cost) ? [...data.cost] : [];
  let latArr = Array.isArray(data.latency_ms) ? [...data.latency_ms] : [];
  let effArr = Array.isArray(data.efficiency_percent) ? [...data.efficiency_percent] : [];

  // If backend sends hybrid metrics separately, append them and label
  const hasHybridObject = data.hybrid && (typeof data.hybrid === 'object');
  const hasHybridScalars = (typeof data.hybrid_cost === 'number') || (typeof data.hybrid_latency_ms === 'number') || (typeof data.hybrid_efficiency_percent === 'number');
  if (hasHybridObject || hasHybridScalars) {
    const hCost = hasHybridObject ? data.hybrid.cost : data.hybrid_cost;
    const hLat = hasHybridObject ? data.hybrid.latency_ms : data.hybrid_latency_ms;
    const hEff = hasHybridObject ? data.hybrid.efficiency_percent : data.hybrid_efficiency_percent;
    if (typeof hCost === 'number') costArr.push(hCost);
    if (typeof hLat === 'number') latArr.push(hLat);
    if (typeof hEff === 'number') effArr.push(hEff);
  }

  // Ensure we always have three values, pad with zeros if missing
  if (costArr.length < labels.length || latArr.length < labels.length || effArr.length < labels.length) {
    while (costArr.length < labels.length) costArr.push(0);
    while (latArr.length < labels.length) latArr.push(0);
    while (effArr.length < labels.length) effArr.push(0);
  }

  // Render simple graph initially (Efficiency by default)
  let currentMetric = 'Efficiency';
  const ctx = chartCanvas.getContext('2d');
  const metricToSeries = () => {
    if (currentMetric === 'Cost') return costArr;
    if (currentMetric === 'Latency') return latArr;
    return effArr;
  };
  const renderCurrent = () => renderSimpleChart(ctx, labels, metricToSeries(), currentMetric);
  renderCurrent();

  renderSummary(summaryPanel, {
    cost: costArr,
    latency_ms: latArr
  });

  // Default visibility
  show(chartPanel, 'block'); // keep canvas visible initially
  setHidden(summaryPanel, false);
  hide(confusionContainer, metricsContainer);

  // Wire buttons
  if (showGraphBtn) {
    showGraphBtn.addEventListener('click', () => {
      setActive(showGraphBtn);
      show(chartPanel, 'block');
      setHidden(summaryPanel, false);
      hide(confusionContainer, metricsContainer);
    });
  }
  if (showConfusionBtn) {
    showConfusionBtn.addEventListener('click', () => {
      setActive(showConfusionBtn);
      hide(chartPanel, metricsContainer);
      setHidden(summaryPanel, true);
      renderConfusion(confusionContainer, cm);
      show(confusionContainer, 'block');
    });
  }
  if (showMetricsBtn) {
    showMetricsBtn.addEventListener('click', () => {
      setActive(showMetricsBtn);
      hide(chartPanel, confusionContainer);
      setHidden(summaryPanel, true);
      renderMetrics(metricsContainer, metrics);
      show(metricsContainer, 'block');
    });
  }

  // Wire metric toggle buttons
  function setMetricActive(btn) {
    [metricEfficiencyBtn, metricCostBtn, metricLatencyBtn].forEach(b => {
      if (!b) return;
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-pressed', String(b === btn));
    });
  }
  if (metricEfficiencyBtn) {
    metricEfficiencyBtn.addEventListener('click', () => {
      currentMetric = 'Efficiency';
      setMetricActive(metricEfficiencyBtn);
      renderCurrent();
    });
  }
  if (metricCostBtn) {
    metricCostBtn.addEventListener('click', () => {
      currentMetric = 'Cost';
      setMetricActive(metricCostBtn);
      renderCurrent();
    });
  }
  if (metricLatencyBtn) {
    metricLatencyBtn.addEventListener('click', () => {
      currentMetric = 'Latency';
      setMetricActive(metricLatencyBtn);
      renderCurrent();
    });
  }

  // Keyboard navigation for tabs (Left/Right arrows)
  const tabs = [showGraphBtn, showConfusionBtn, showMetricsBtn].filter(Boolean);
  tabs.forEach((tab, idx) => {
    tab.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        const dir = e.key === 'ArrowRight' ? 1 : -1;
        const next = tabs[(idx + dir + tabs.length) % tabs.length];
        next.focus();
        next.click();
      }
    });
  });
});
