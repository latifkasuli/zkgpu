/**
 * Web Worker that owns the WebGPU device and runs NTT test suites.
 *
 * Communication:
 *   main thread  --postMessage-->  worker  (WorkerRequest)
 *   worker       --postMessage-->  main    (WorkerResponse)
 *
 * Request serialization: incoming messages are queued and processed
 * one at a time. This prevents a second `init` or `run_suite` from
 * interleaving with an in-flight async operation, which would corrupt
 * the wasm-side global device state.
 */

import type { WorkerRequest, WorkerResponse, HarnessResponse, DeviceReport } from "./types";

// ---- State ----

let wasmReady = false;
let wasmModule: typeof import("../pkg/zkgpu_web") | null = null;

// ---- Request queue (serializes async work) ----

type QueueEntry = () => Promise<void>;
const requestQueue: QueueEntry[] = [];
let processing = false;

/**
 * Enqueue an async handler and drain the queue serially.
 * Each handler runs to completion before the next one starts.
 */
function enqueue(handler: QueueEntry) {
  requestQueue.push(handler);
  if (!processing) {
    drainQueue();
  }
}

async function drainQueue() {
  processing = true;
  while (requestQueue.length > 0) {
    const next = requestQueue.shift()!;
    try {
      await next();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      log("error", `Unhandled queue error: ${msg}`);
    }
  }
  processing = false;
}

// ---- Helpers ----

function post(msg: WorkerResponse) {
  self.postMessage(msg);
}

function log(level: string, message: string) {
  post({ type: "log", level, message });
}

// ---- Message handler ----

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;

  switch (msg.type) {
    case "init":
      enqueue(() => handleInit());
      break;

    case "run_suite":
      enqueue(() => handleRunSuite(msg.request));
      break;

    case "version":
      // version is synchronous, but still serialize it to avoid ordering issues
      enqueue(async () => handleVersion());
      break;

    default:
      log("warn", `unknown message type: ${(msg as unknown as Record<string, string>).type}`);
  }
};

// ---- Init ----

async function handleInit() {
  try {
    log("info", "Loading wasm module...");

    // Dynamic import of the wasm-pack output.
    const wasm = await import("../pkg/zkgpu_web");
    await wasm.default();
    wasmModule = wasm;

    log("info", "Wasm loaded. Initializing GPU device...");

    const deviceJson = await wasm.gpu_init();
    const device: DeviceReport = JSON.parse(deviceJson);

    wasmReady = true;
    log("info", `GPU ready: ${device.name} (${device.backend})`);

    post({ type: "init_ok", device });
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e);
    log("error", `Init failed: ${error}`);
    post({ type: "init_error", error });
  }
}

// ---- Run suite ----

async function handleRunSuite(request: import("./types").HarnessRequest) {
  if (!wasmReady || !wasmModule) {
    post({
      type: "suite_error",
      error: "Worker not initialized — send 'init' first",
    });
    return;
  }

  try {
    const requestJson = JSON.stringify(request);
    log("info", `Running suite: ${request.suite ?? "custom"}...`);

    const resultJson = await wasmModule.run_suite(requestJson);
    const response: HarnessResponse = JSON.parse(resultJson);

    post({ type: "suite_result", response });
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e);
    log("error", `Suite failed: ${error}`);
    post({ type: "suite_error", error });
  }
}

// ---- Version ----

function handleVersion() {
  if (!wasmModule) {
    log("warn", "version requested before init");
    return;
  }
  try {
    const versionJson = wasmModule.version();
    const version = JSON.parse(versionJson);
    post({ type: "version_result", version });
  } catch (e) {
    log("error", `Version failed: ${e}`);
  }
}
