/**
 * TypeScript types mirroring zkgpu-report's serde JSON schema.
 * Field names use snake_case to match Rust serialization.
 */

// ---------- Request ----------

export interface HarnessRequest {
  suite?: "Smoke" | "Validation" | "Benchmark";
  spec?: SuiteSpec;
  family_override?: "Auto" | "Stockham" | "FourStep";
}

export interface SuiteSpec {
  kind: string;
  family_override: string;
  fail_fast: boolean;
  cases: CaseSpec[];
}

export interface CaseSpec {
  name: string;
  log_n: number;
  direction: "Forward" | "Inverse" | "Roundtrip";
  input: InputPattern;
  profile_gpu_timestamps: boolean;
}

export type InputPattern =
  | "Sequential"
  | "AllZeros"
  | "AllOnes"
  | "LargeValuesDescending"
  | { PseudoRandomDeterministic: { seed: number } };

// ---------- Response ----------

export interface HarnessResponse {
  ok: boolean;
  report?: SuiteReport;
  error?: string;
}

export interface SuiteReport {
  schema_version: number;
  suite: string;
  device: DeviceReport;
  kernel: KernelReport;
  cases: CaseReport[];
  summary: SuiteSummary;
}

export interface DeviceReport {
  name: string;
  backend: string;
  tier: string;
  gpu_family: string;
  platform_class: string;
  memory_model: string;
  max_buffer_size_bytes: number;
  max_workgroup_size_x: number;
  max_compute_invocations: number;
  feature_flags: string[];
}

export interface KernelReport {
  field: string;
  ntt_variant: string;
}

export interface CaseReport {
  name: string;
  log_n: number;
  direction: string;
  input: InputPattern;
  kernel_family?: string;
  passed: boolean;
  mismatch_count: number;
  first_mismatch_index?: number;
  first_mismatch_gpu?: string;
  first_mismatch_cpu?: string;
  timings: TimingReport;
  error?: string;
}

export interface TimingReport {
  wall_time_ns?: number;
  gpu_total_ns?: number;
  gpu_stage_ns: StageTimingReport[];
}

export interface StageTimingReport {
  label: string;
  duration_ns: number;
}

export interface SuiteSummary {
  total_cases: number;
  passed_cases: number;
  failed_cases: number;
}

// ---------- Worker Messages ----------

export type WorkerRequest =
  | { type: "init" }
  | { type: "run_suite"; request: HarnessRequest }
  | { type: "version" };

export type WorkerResponse =
  | { type: "init_ok"; device: DeviceReport }
  | { type: "init_error"; error: string }
  | { type: "suite_result"; response: HarnessResponse }
  | { type: "suite_error"; error: string }
  | { type: "version_result"; version: unknown }
  | { type: "log"; level: string; message: string };
