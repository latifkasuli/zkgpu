/// <reference types="vite/client" />

// Declare the wasm-pack generated module so TypeScript doesn't complain.
declare module "../pkg/zkgpu_web" {
  /** Initialize the wasm module. Must be called before other exports. */
  export default function init(): Promise<void>;

  /** Initialize the GPU device. Returns JSON DeviceReport. */
  export function gpu_init(): Promise<string>;

  /** Return cached device info as JSON. */
  export function device_info(): string;

  /** Run a test suite from JSON HarnessRequest. Returns JSON HarnessResponse. */
  export function run_suite(request_json: string): Promise<string>;

  /**
   * Run a single test case. Input: JSON `SingleCaseRequest` envelope
   * (or legacy bare `CaseSpec`, BabyBear-only).
   *
   * Output: JSON-encoded `HarnessResponse` on both paths —
   *   - success → `{ ok: true, case_report: ... }`
   *   - error   → `{ ok: false, error: ... }`
   * Shape mirrors `run_suite` and `run_hash` so callers parse all
   * three wasm entry points uniformly.
   */
  export function run_case(case_json: string): Promise<string>;

  /**
   * Phase F.3.d wasm entry — run a Poseidon2 hash suite.
   *
   * Input: JSON-encoded `HashSpec` (no envelope; bare spec).
   *
   * Output: JSON-encoded `HarnessResponse` on both paths —
   *   - success → `{ ok: true, hash_report: ... }`
   *   - error   → `{ ok: false, error: ... }`
   * Shape mirrors `run_suite` so callers parse uniformly.
   */
  export function run_hash(spec_json: string): Promise<string>;

  /** Return version info as JSON. */
  export function version(): string;
}
