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
  /**
   * Phase F.3.e hash dispatch. When `hash_spec` is set, the FFI
   * router runs a hash suite and returns `hash_report` on the
   * response instead of `report`. Exactly one of
   * `{suite, spec, hash_spec}` should be populated; the router
   * rejects ambiguous requests.
   */
  hash_spec?: HashSpec;
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
  /**
   * Phase F.3.e hash-path output. Populated when the caller set
   * `hash_spec` on the request. `report` and `hash_report` are
   * mutually exclusive — a single response only ever carries one.
   */
  hash_report?: HashSuiteReport;
  /**
   * Single-case output from the wasm `run_case` entry point. `None`
   * for all FFI `run_request` paths (which only dispatch whole
   * suites) and for the wasm `run_suite` / `run_hash` paths. Present
   * only when a caller invokes `run_case` directly.
   */
  case_report?: CaseReport;
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
   * Why the planner chose that strategy (e.g. `"HeuristicDefaultLocal"`,
   * `"HeuristicBrowserConservative"`, `"ForcedLocal"`). Omitted alongside
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

// ---------- Hash surface (Phase F.3.e) ----------

/**
 * Which hash primitive a suite targets. Only Poseidon2 today;
 * future algorithms extend this enum on the Rust side and here.
 * Mirrors `zkgpu_report::HashAlgorithm`.
 */
export type HashAlgorithm = "Poseidon2";

/**
 * Which prime field a Poseidon2 suite targets. Mirrors
 * `zkgpu_report::Field`.
 */
export type Field = "BabyBear" | "Goldilocks";

/**
 * Hash-case input pattern. Unit variants (`AllZeros`, `AllOnes`,
 * `Sequential`) serialise as bare strings; `SplitMix64` carries a
 * seed in a tagged envelope. Mirrors
 * `zkgpu_report::HashInputPattern`.
 */
export type HashInputPattern =
  | "AllZeros"
  | "AllOnes"
  | "Sequential"
  | { SplitMix64: { seed: number } };

export interface HashCaseSpec {
  name: string;
  /** Batch size — number of independent permutation instances. */
  num_permutations: number;
  input: HashInputPattern;
  profile_gpu_timestamps: boolean;
  iterations: number;
  warmup_iterations: number;
}

export interface HashSpec {
  kind: "Smoke" | "Validation" | "Benchmark" | "Soak";
  cases: HashCaseSpec[];
  fail_fast: boolean;
  algorithm: HashAlgorithm;
  field: Field;
}

export interface HashCaseReport {
  name: string;
  num_permutations: number;
  input: HashInputPattern;
  /**
   * `"babybear-poseidon2"` / `"goldilocks-poseidon2-portable"` etc.
   * Lets mixed-field aggregators distinguish rows without sniffing
   * the spec.
   */
  kernel_family?: string;
  passed: boolean;
  mismatch_count: number;
  /**
   * `(permutation_index, slot_index)` tuple of the first mismatch.
   * Serialised as a two-element array by serde. `undefined` when
   * the case passed.
   */
  first_mismatch_index?: [number, number];
  first_mismatch_gpu?: string;
  first_mismatch_cpu?: string;
  timings: TimingReport;
  error?: string;
}

export interface HashSuiteReport {
  schema_version: number;
  suite: string;
  device: DeviceReport;
  kernel: KernelReport;
  cases: HashCaseReport[];
  summary: SuiteSummary;
}

// ---------- Worker Messages ----------

export type WorkerRequest =
  | { type: "init" }
  | { type: "run_suite"; request: HarnessRequest }
  /**
   * Phase F.3.e hash-run worker message. Corresponds to the wasm
   * `run_hash(spec_json)` entry point — worker receives a
   * `HashSpec` directly (not the envelope carried by `run_suite`).
   */
  | { type: "run_hash"; spec: HashSpec }
  | { type: "version" };

export type WorkerResponse =
  | { type: "init_ok"; device: DeviceReport }
  | { type: "init_error"; error: string }
  | { type: "suite_result"; response: HarnessResponse }
  | { type: "suite_error"; error: string }
  /**
   * Hash result carries a `HashSuiteReport` unwrapped from the
   * `HarnessResponse` envelope that wasm `run_hash` returns. The
   * worker parses the envelope and forwards the inner report so the
   * main-thread consumer sees a flat shape symmetric with
   * `suite_result`.
   */
  | { type: "hash_result"; report: HashSuiteReport }
  | { type: "hash_error"; error: string }
  | { type: "version_result"; version: unknown }
  | { type: "log"; level: string; message: string };
