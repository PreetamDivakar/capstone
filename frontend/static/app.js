// app.js — frontend main logic
// - Shows processing popup for a minimum of 2 seconds when Predict is pressed
// - Disables buttons while running
// - Sends /predict request and displays results

const processingPopup = document.getElementById("processing-popup");
const resultBox = document.getElementById("result");
const loader = document.getElementById("loader"); // kept for compatibility (hidden)
const allButtons = Array.from(document.querySelectorAll("button"));
const predictBtn = document.getElementById("btn-predict");

// Show processing popup (accessibility)
function showProcessingPopup() {
  if (!processingPopup) return;
  processingPopup.classList.remove("hidden");
  processingPopup.setAttribute("aria-hidden", "false");
}

// Hide processing popup
function hideProcessingPopup() {
  if (!processingPopup) return;
  processingPopup.classList.add("hidden");
  processingPopup.setAttribute("aria-hidden", "true");
}

// Disable / enable all buttons
function setButtonsDisabled(val) {
  allButtons.forEach(b => b.disabled = val);
}

// Format numeric output
function formatNum(n, dp=2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
  return Number(n).toFixed(dp);
}

// Utility: wait ms
function wait(ms) { return new Promise(res => setTimeout(res, ms)); }

// Main: send prediction but ensure popup stays visible at least 2000ms
async function sendPrediction() {
  // Input fields
  const cpuRaw = document.getElementById("cpu_cores").value;
  const memRaw = document.getElementById("memory_mb").value;
  const latencyRaw = document.getElementById("latency_sensitive").value;
  const timeRaw = document.getElementById("execution_time").value;
  const sizeRaw = document.getElementById("data_size_mb").value;

  // Validate presence
  if ([cpuRaw, memRaw, latencyRaw, timeRaw, sizeRaw].some(v => v === "")) {
    alert("⚠️ Please fill all required fields before predicting!");
    return;
  }

  // Parse values
  const cpu = parseFloat(cpuRaw);
  const mem = parseFloat(memRaw);
  const latency = parseInt(latencyRaw);
  const time = parseFloat(timeRaw);
  const size = parseFloat(sizeRaw);

  if (Number.isNaN(cpu) || Number.isNaN(mem) || Number.isNaN(latency) || Number.isNaN(time) || Number.isNaN(size)) {
    alert("⚠️ Please enter valid numeric values.");
    return;
  }

  if (cpu < 0 || mem < 0 || latency < 0 || time < 0 || size < 0) {
    alert("⚠️ Values cannot be negative.");
    return;
  }

  if (cpu === 0 && mem === 0 && latency === 0 && time === 0 && size === 0) {
    alert("⚠️ All values are zero — please provide realistic workload parameters.");
    return;
  }

  // Clear previous
  resultBox.innerHTML = "";

  // Start the visible processing and start timer for minimum display time
  const MIN_MS = 1000;
  showProcessingPopup();
  setButtonsDisabled(true);
  const t0 = Date.now();

  try {
    const payload = {
      cpu_cores: cpu,
      memory_mb: mem,
      latency_sensitive: latency,
      execution_time: time,
      data_size_mb: size,
    };

    // Fire fetch in parallel with the minimum wait
    const fetchPromise = fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    // Wait for both: server response and minimum time
    const [res] = await Promise.all([fetchPromise, wait(MIN_MS)]);

    // Got response (or error status)
    const textBody = await res.text();
    let data;
    try { data = JSON.parse(textBody); } catch (e) { data = null; }

    // Ensure at least MIN_MS elapsed (we already waited), but keep this guard
    const elapsed = Date.now() - t0;
    if (elapsed < MIN_MS) {
      await wait(MIN_MS - elapsed);
    }

    // Hide popup + enable UI
    hideProcessingPopup();
    setButtonsDisabled(false);

    if (!res.ok) {
      const msg = (data && data.error) ? data.error : textBody || res.statusText;
      alert("⚠️ " + (msg || `Server returned ${res.status}`));
      return;
    }

    if (data && data.error) {
      alert("⚠️ " + data.error);
      return;
    }

    // Format and display results
    const cpuOut = data.cpu_cores ?? cpu;
    const memOut = data.memory_mb ?? mem;
    const costTrad = (data.cost_traditional !== undefined) ? formatNum(data.cost_traditional, 6) : "-";
    const costSrv = (data.cost_serverless !== undefined) ? formatNum(data.cost_serverless, 6) : "-";
    const costRatio = (data.cost_ratio !== undefined && data.cost_ratio !== null) ? formatNum(data.cost_ratio, 3) : "-";
    const platform = data.ideal_platform ?? "Unknown";

    resultBox.innerHTML = `
      <p><b>CPU:</b> ${cpuOut}</p>
      <p><b>Memory:</b> ${memOut} MB</p>
      <hr>
      <p><b>💰 Cost (Traditional):</b> $${costTrad}</p>
      <p><b>⚡ Cost (Serverless):</b> $${costSrv}</p>
      <p><b>📊 Cost Ratio:</b> ${costRatio}</p>
      <hr>
      <p><b>🏁 Ideal Deployment:</b> 
         <span class="${platform === 'Serverless' ? 'serverless' : 'traditional'}">
         ${platform}</span>
      </p>`;
  } catch (error) {
    // Ensure popup hidden and UI re-enabled on errors
    hideProcessingPopup();
    setButtonsDisabled(false);
    console.error(error);
    alert("❌ Error processing request. " + (error.message || ""));
  }
}

// Presets / reset
function presetServerless() {
  // NOTE: latency_sensitive should be 0 for serverless-friendly preset (not latency-sensitive)
  document.getElementById("cpu_cores").value = 1.5;
  document.getElementById("memory_mb").value = 1024;
  document.getElementById("latency_sensitive").value = 0;   // <-- changed to 0
  document.getElementById("execution_time").value = 120;
  document.getElementById("data_size_mb").value = 300;
  resultBox.innerHTML = "<p>Results will appear here after prediction.</p>";
}

function presetTraditional() {
  document.getElementById("cpu_cores").value = 8;
  document.getElementById("memory_mb").value = 16384;
  document.getElementById("latency_sensitive").value = 1; // latency-sensitive for traditional workload
  document.getElementById("execution_time").value = 1800;
  document.getElementById("data_size_mb").value = 20000;
  resultBox.innerHTML = "<p>Results will appear here after prediction.</p>";
}

function resetFields() {
  document.querySelectorAll("input").forEach(i => i.value = "");
  resultBox.innerHTML = "<p>Results will appear here after prediction.</p>";
  hideProcessingPopup();
  setButtonsDisabled(false);
}

// Wire events (if buttons exist in DOM)
window.addEventListener('DOMContentLoaded', () => {
  const pb = document.getElementById("btn-predict");
  if (pb) pb.addEventListener('click', sendPrediction);

  const presetSrv = document.getElementById("btn-preset-serverless");
  if (presetSrv) presetSrv.addEventListener('click', presetServerless);

  const presetTrad = document.getElementById("btn-preset-traditional");
  if (presetTrad) presetTrad.addEventListener('click', presetTraditional);

  const resetBtn = document.getElementById("btn-reset");
  if (resetBtn) resetBtn.addEventListener('click', resetFields);
});
