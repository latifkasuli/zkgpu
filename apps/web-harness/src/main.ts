/**
 * Main-thread UI controller for the zkGPU web harness.
 *
 * Spawns a dedicated Web Worker (worker.ts) that owns the GPU device
 * and wasm module. Communicates via postMessage with JSON payloads.
 */

import type {
  WorkerRequest,
  WorkerResponse,
  HarnessRequest,
  HarnessResponse,
  SuiteReport,
  CaseReport,
  DeviceReport,
  HashCaseReport,
  HashInputPattern,
  HashSuiteReport,
} from "./types";

// ---------- DOM refs ----------

const devicePanel = document.getElementById("device-panel")!;
const suiteSelect = document.getElementById("suite-select") as HTMLSelectElement;
const familySelect = document.getElementById("family-select") as HTMLSelectElement;
const runBtn = document.getElementById("run-btn") as HTMLButtonElement;
const exportBtn = document.getElementById("export-btn") as HTMLButtonElement;
const copyBtn = document.getElementById("copy-btn") as HTMLButtonElement;
const downloadBtn = document.getElementById("download-btn") as HTMLButtonElement;
const statusDiv = document.getElementById("status")!;
const summaryDiv = document.getElementById("summary")!;
const resultsBody = document.getElementById("results-body") as HTMLTableSectionElement;
const jsonOutput = document.getElementById("json-output")!;
const jsonText = document.getElementById("json-text") as HTMLTextAreaElement;

// ---------- State ----------

let lastReport: SuiteReport | null = null;
let lastHashReport: HashSuiteReport | null = null;
let lastResponseJson: string | null = null;
let running = false;

// ---------- Worker setup ----------

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});

// Expose on window so Playwright (G.0.2 tail-A/B test suite) can post
// arbitrary HarnessRequest payloads that aren't in the UI's preset list
// — notably `stockham_tail_override: Local|Global` for forced-A/B.
// Harmless for production users; scoped to the browser harness.
(globalThis as unknown as { __zkgpuWorker: Worker }).__zkgpuWorker = worker;

worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
  handleWorkerMessage(event.data);
};

worker.onerror = (event) => {
  setStatus("error", `Worker error: ${event.message}`);
  setRunning(false);
};

// ---------- Worker message dispatch ----------

function handleWorkerMessage(msg: WorkerResponse) {
  switch (msg.type) {
    case "init_ok":
      onInitOk(msg.device);
      break;
    case "init_error":
      onInitError(msg.error);
      break;
    case "suite_result":
      onSuiteResult(msg.response);
      break;
    case "suite_error":
      setStatus("error", msg.error);
      setRunning(false);
      break;
    case "hash_result":
      onHashResult(msg.report);
      break;
    case "hash_error":
      setStatus("error", msg.error);
      setRunning(false);
      break;
    case "version_result":
      console.log("zkgpu-web version:", msg.version);
      break;
    case "log":
      handleLog(msg.level, msg.message);
      break;
  }
}

// ---------- Init flow ----------

function onInitOk(device: DeviceReport) {
  devicePanel.textContent = [
    `Device:   ${device.name}`,
    `Backend:  ${device.backend}`,
    `Tier:     ${device.tier}`,
    `Family:   ${device.gpu_family}`,
    `Platform: ${device.platform_class}`,
    `Memory:   ${device.memory_model}`,
    `Buffer:   ${formatBytes(device.max_buffer_size_bytes)}`,
    `WG size:  ${device.max_workgroup_size_x}  (invocations: ${device.max_compute_invocations})`,
    `Features: ${device.feature_flags.join(", ") || "none"}`,
  ].join("\n");

  runBtn.disabled = false;
  setStatus("success", "GPU device ready");
}

function onInitError(error: string) {
  devicePanel.textContent = `Failed to initialize GPU: ${error}`;
  setStatus("error", error);
}

// ---------- Suite execution ----------

function startSuite() {
  if (running) return;
  setRunning(true);
  clearResults();

  const suite = suiteSelect.value as "Smoke" | "Validation" | "Benchmark";
  const family = familySelect.value as "Auto" | "Stockham" | "FourStep";

  const request: HarnessRequest = { suite };
  if (family !== "Auto") {
    request.family_override = family;
  }

  setStatus("info", `Running ${suite} suite...`);
  sendToWorker({ type: "run_suite", request });
}

function onSuiteResult(response: HarnessResponse) {
  setRunning(false);

  if (!response.ok || !response.report) {
    setStatus("error", response.error ?? "Unknown error");
    return;
  }

  lastReport = response.report;
  // Clear hash-run state so a suite result following a hash result
  // doesn't inherit the `zkgpu_hash_` download prefix or other
  // hash-side metadata. Mirrors the `lastReport = null` in
  // onHashResult() for full state symmetry on the __zkgpuWorker path.
  lastHashReport = null;
  // Enrich with browser timing metadata before serializing
  const enriched = enrichResponse(response);
  lastResponseJson = JSON.stringify(enriched, null, 2);

  renderReport(response.report);
  exportBtn.disabled = false;
  copyBtn.disabled = false;
  downloadBtn.disabled = false;

  const s = response.report.summary;
  if (s.failed_cases === 0) {
    setStatus("success", `All ${s.total_cases} cases passed`);
  } else {
    setStatus("error", `${s.failed_cases}/${s.total_cases} cases failed`);
  }
}

// ---------- Hash result handling (Phase F.3.e) ----------
//
// The shipped UI has no "Run Hash" button today; this path exists so
// consumers driving the worker through `__zkgpuWorker` (Playwright,
// future UI hooks) get a uniform main-thread response instead of a
// silent drop. Mirrors `onSuiteResult` for JSON-export symmetry.

function onHashResult(report: HashSuiteReport) {
  setRunning(false);

  lastReport = null;
  lastHashReport = report;

  // Reuse the enrichment shape so exported JSON carries the same
  // browser metadata (clock source, UA, worker flag) as NTT runs.
  const enriched = enrichHashReport(report);
  lastResponseJson = JSON.stringify(enriched, null, 2);

  // Re-render summary + rows so any stale NTT case table from a
  // previous `run_suite` is replaced — otherwise the DOM would show
  // NTT kernel labels while `lastResponseJson` / download now point
  // at the hash report.
  renderHashReport(report);

  exportBtn.disabled = false;
  copyBtn.disabled = false;
  downloadBtn.disabled = false;

  const s = report.summary;
  const line = `hash suite ${report.suite}: ${s.passed_cases}/${s.total_cases} passed`;
  console.log(`[hash] ${line}`);
  if (s.failed_cases === 0) {
    setStatus("success", `All ${s.total_cases} hash cases passed`);
  } else {
    setStatus("error", `${s.failed_cases}/${s.total_cases} hash cases failed`);
  }
}

function enrichHashReport(report: HashSuiteReport): Record<string, unknown> {
  const obj = JSON.parse(JSON.stringify(report)) as Record<string, unknown>;
  obj.timing_metadata = collectTimingMetadata();
  obj.collected_at = new Date().toISOString();
  obj.harness = "web";
  return obj;
}

// ---------- Rendering ----------

function renderReport(report: SuiteReport) {
  // Summary bar
  const s = report.summary;
  summaryDiv.innerHTML = [
    `<span class="summary-item"><span class="summary-label">Total:</span> ${s.total_cases}</span>`,
    `<span class="summary-item pass"><span class="summary-label">Passed:</span> ${s.passed_cases}</span>`,
    `<span class="summary-item fail"><span class="summary-label">Failed:</span> ${s.failed_cases}</span>`,
    `<span class="summary-item"><span class="summary-label">Kernel:</span> ${report.kernel.ntt_variant}</span>`,
  ].join("");
  summaryDiv.classList.add("visible");

  // Case rows
  resultsBody.innerHTML = "";
  for (const c of report.cases) {
    renderCaseRow(c, resultsBody);
  }
}

// Hash-specific rendering. The table headers in index.html are fixed
// for the NTT schema (log_n / Direction), so hash rows overload those
// two columns with the closest analogues: permutation count and input
// pattern. Kernel / timings / mismatches map 1:1. Keeps the UI state
// self-consistent for __zkgpuWorker consumers without needing a second
// table element.
function renderHashReport(report: HashSuiteReport) {
  const s = report.summary;
  summaryDiv.innerHTML = [
    `<span class="summary-item"><span class="summary-label">Total:</span> ${s.total_cases}</span>`,
    `<span class="summary-item pass"><span class="summary-label">Passed:</span> ${s.passed_cases}</span>`,
    `<span class="summary-item fail"><span class="summary-label">Failed:</span> ${s.failed_cases}</span>`,
    `<span class="summary-item"><span class="summary-label">Kernel:</span> ${escapeHtml(report.kernel.field)} poseidon2</span>`,
  ].join("");
  summaryDiv.classList.add("visible");

  resultsBody.innerHTML = "";
  for (const c of report.cases) {
    renderHashCaseRow(c, resultsBody);
  }
}

function formatHashInput(p: HashInputPattern): string {
  if (typeof p === "string") return p;
  if ("SplitMix64" in p) return `SplitMix64(${p.SplitMix64.seed})`;
  return "?";
}

function renderHashCaseRow(c: HashCaseReport, tbody: HTMLTableSectionElement) {
  const tr = document.createElement("tr");

  const statusClass = c.passed ? "pass" : "fail";
  const statusText = c.passed ? "PASS" : "FAIL";
  const wallMs = c.timings.wall_time_ns != null ? (c.timings.wall_time_ns / 1e6).toFixed(2) : "-";
  const gpuMs = c.timings.gpu_total_ns != null ? (c.timings.gpu_total_ns / 1e6).toFixed(2) : "-";
  const mismatches = c.mismatch_count > 0
    ? `${c.mismatch_count} (first @ [${c.first_mismatch_index?.join(",") ?? "?"}])`
    : "0";

  tr.innerHTML = [
    `<td class="${statusClass}">${statusText}</td>`,
    `<td>${escapeHtml(c.name)}</td>`,
    `<td>${c.num_permutations}</td>`,
    `<td>${escapeHtml(formatHashInput(c.input))}</td>`,
    `<td>${escapeHtml(c.kernel_family ?? "-")}</td>`,
    `<td class="timing">${wallMs}</td>`,
    `<td class="timing">${gpuMs}</td>`,
    `<td>${mismatches}</td>`,
  ].join("");

  tbody.appendChild(tr);

  if (c.error) {
    const errTr = document.createElement("tr");
    errTr.innerHTML = `<td></td><td colspan="7" class="fail" style="font-size:0.75rem">${escapeHtml(c.error)}</td>`;
    tbody.appendChild(errTr);
  }
}

function renderCaseRow(c: CaseReport, tbody: HTMLTableSectionElement) {
  const tr = document.createElement("tr");

  const statusClass = c.passed ? "pass" : "fail";
  const statusText = c.passed ? "PASS" : "FAIL";
  const wallMs = c.timings.wall_time_ns != null ? (c.timings.wall_time_ns / 1e6).toFixed(2) : "-";
  const gpuMs = c.timings.gpu_total_ns != null ? (c.timings.gpu_total_ns / 1e6).toFixed(2) : "-";
  const mismatches = c.mismatch_count > 0
    ? `${c.mismatch_count} (first @ ${c.first_mismatch_index})`
    : "0";

  tr.innerHTML = [
    `<td class="${statusClass}">${statusText}</td>`,
    `<td>${escapeHtml(c.name)}</td>`,
    `<td>${c.log_n}</td>`,
    `<td>${c.direction}</td>`,
    `<td>${c.kernel_family ?? "-"}</td>`,
    `<td class="timing">${wallMs}</td>`,
    `<td class="timing">${gpuMs}</td>`,
    `<td>${mismatches}</td>`,
  ].join("");

  tbody.appendChild(tr);

  // Show error as a collapsible second row
  if (c.error) {
    const errTr = document.createElement("tr");
    errTr.innerHTML = `<td></td><td colspan="7" class="fail" style="font-size:0.75rem">${escapeHtml(c.error)}</td>`;
    tbody.appendChild(errTr);
  }
}

function clearResults() {
  resultsBody.innerHTML = "";
  summaryDiv.classList.remove("visible");
  summaryDiv.innerHTML = "";
  jsonOutput.style.display = "none";
  lastReport = null;
  lastHashReport = null;
  lastResponseJson = null;
  exportBtn.disabled = true;
  copyBtn.disabled = true;
  downloadBtn.disabled = true;
}

// ---------- JSON export ----------

function exportJson() {
  if (!lastResponseJson) return;
  jsonText.value = lastResponseJson;
  jsonOutput.style.display = "block";
}

async function copyJson() {
  if (!lastResponseJson) return;
  try {
    await navigator.clipboard.writeText(lastResponseJson);
    setStatus("success", "JSON copied to clipboard");
  } catch {
    // Fallback: show the textarea
    exportJson();
    jsonText.select();
  }
}

function downloadJson() {
  if (!lastResponseJson) return;
  const source = lastReport ?? lastHashReport;
  if (!source) return;
  const blob = new Blob([lastResponseJson], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const deviceName = source.device.name.replace(/[^a-zA-Z0-9]/g, "_");
  const suite = source.suite;
  const prefix = lastHashReport ? "zkgpu_hash" : "zkgpu";
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  a.href = url;
  a.download = `${prefix}_${suite}_${deviceName}_${ts}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- Browser metadata enrichment ----------

interface TimingMetadata {
  clock_source: string;
  timestamp_quantized: boolean;
  worker: boolean;
  secure_context: boolean;
  user_agent: string;
  adapter_info: string;
}

function collectTimingMetadata(): TimingMetadata {
  return {
    clock_source: "browser-wall",
    timestamp_quantized: true, // browsers quantize performance.now()
    worker: true, // our execution runs in a Web Worker
    secure_context: globalThis.isSecureContext ?? false,
    user_agent: navigator.userAgent,
    adapter_info: lastReport?.device.name ?? lastHashReport?.device.name ?? "",
  };
}

function enrichResponse(response: HarnessResponse): Record<string, unknown> {
  const obj = JSON.parse(JSON.stringify(response)) as Record<string, unknown>;
  if (obj.report && typeof obj.report === "object") {
    (obj.report as Record<string, unknown>).timing_metadata = collectTimingMetadata();
    (obj.report as Record<string, unknown>).collected_at = new Date().toISOString();
    (obj.report as Record<string, unknown>).harness = "web";
  }
  return obj;
}

// ---------- Helpers ----------

function sendToWorker(msg: WorkerRequest) {
  worker.postMessage(msg);
}

function setRunning(state: boolean) {
  running = state;
  runBtn.disabled = state;
  runBtn.textContent = state ? "Running..." : "Run Suite";
}

function setStatus(level: "info" | "success" | "error", text: string) {
  statusDiv.className = level;
  statusDiv.textContent = text;
}

function handleLog(level: string, message: string) {
  const method = level === "error" ? "error" : level === "warn" ? "warn" : "log";
  console[method](`[worker] ${message}`);

  // Update status for info-level logs during init/run
  if (running || level === "info") {
    setStatus("info", message);
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------- Event bindings ----------

runBtn.addEventListener("click", startSuite);
exportBtn.addEventListener("click", exportJson);
copyBtn.addEventListener("click", copyJson);
downloadBtn.addEventListener("click", downloadJson);

// ---------- Boot ----------

setStatus("info", "Initializing GPU device...");
sendToWorker({ type: "init" });
sendToWorker({ type: "version" });
