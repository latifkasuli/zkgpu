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

  /** Run a single test case from JSON CaseSpec. Returns JSON CaseReport. */
  export function run_case(case_json: string): Promise<string>;

  /** Return version info as JSON. */
  export function version(): string;
}
