/**
 * TypeScript types mirroring zkgpu-report's serde JSON schema.
 * Field names use snake_case to match Rust serialization.
 */

// ---------- Request ----------

/**
 * Stockham tail-phase override. `Auto` lets the planner heuristic pick
 * between `LocalFusedR4` (workgroup-local fused tail) and `GlobalOnlyR4`
 * (extend the global R4 chain to the end). `Local` and `Global` force the
 * corresponding strategy. Wire format mirrors `zkgpu_report::StockhamTailOverride`.
 */
export type StockhamTailOverride = "Auto" | "Local" | "Global";

export interface HarnessRequest {
  suite?: "Smoke" | "Validation" | "Benchmark";
  spec?: SuiteSpec;
  family_override?: "Auto" | "Stockham" | "FourStep";
  /** Top-level override; if present, overrides any value on `spec`. */
  stockham_tail_override?: StockhamTailOverride;
}

export interface SuiteSpec {
  kind: string;
  family_override: string;
  fail_fast: boolean;
  cases: CaseSpec[];
  /**
   * Per-spec Stockham tail override. Defaults to `Auto` (heuristic) on the
   * Rust side via `#[serde(default)]`, so this field is optional in JSON.
   */
  stockham_tail_override?: StockhamTailOverride;
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
  max_compute_workgroup_storage_size_bytes?: number;
  feature_flags: string[];
}

export interface KernelReport {
  field: string;
  ntt_variant: string;
  /**
   * Suite-level summary of Stockham tail strategy across executed cases.
   * One of `"LocalFusedR4"`, `"GlobalOnlyR4"`, `"mixed"`, or omitted when
   * no case in the suite recorded a tail decision (all four-step or all
   * `log_n < LOG_BLOCK`).
   */
  stockham_tail_strategy?: string;
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
  /**
   * Actual Stockham tail strategy for this case (`"LocalFusedR4"` or
   * `"GlobalOnlyR4"`). Omitted for non-Stockham families and for Stockham
   * cases below `LOG_BLOCK` that have no tail phase.
   */
  stockham_tail_strategy?: string;
  /**
   * Why the planner chose that strategy (e.g. `"HeuristicXclipseLargeN"`,
   * `"HeuristicDefaultLocal"`, `"ForcedLocal"`). Omitted alongside
   * `stockham_tail_strategy` when no tail decision applied.
   */
  stockham_tail_reason?: string;
  /**
   * Per-thread global-memory gather/scatter stride in bytes for the
   * `LocalFusedR4` tail. Omitted for `GlobalOnlyR4` (no strided gather)
   * and for plans without a tail phase.
   */
  tail_stride_bytes?: number;
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
