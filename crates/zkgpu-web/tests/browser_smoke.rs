//! Browser-based smoke tests using `wasm-bindgen-test`.
//!
//! Run with:
//!   wasm-pack test --headless --chrome crates/zkgpu-web
//!   wasm-pack test --headless --firefox crates/zkgpu-web
//!
//! These tests execute in a real browser environment with WebGPU.
//! They will be skipped (or fail gracefully) if the browser does not
//! support WebGPU (e.g., older Firefox, headless without GPU).

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a JSON response string into the ok/error structure.
fn parse_response(json: &str) -> (bool, Option<String>) {
    let v: serde_json::Value = serde_json::from_str(json).expect("valid JSON");
    let ok = v["ok"].as_bool().unwrap_or(false);
    let error = v["error"].as_str().map(|s| s.to_string());
    (ok, error)
}

/// Try to initialize the GPU. Returns true if successful.
/// Logs a warning and returns false if WebGPU is unavailable.
async fn try_init_gpu() -> bool {
    match zkgpu_web::gpu_init().await {
        Ok(_) => true,
        Err(e) => {
            let msg = e.as_string().unwrap_or_else(|| format!("{:?}", e));
            web_sys::console::warn_1(&JsValue::from_str(&format!(
                "gpu_init failed (WebGPU may not be available): {msg}"
            )));
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
async fn test_gpu_init() {
    let result = zkgpu_web::gpu_init().await;
    match result {
        Ok(json) => {
            let v: serde_json::Value =
                serde_json::from_str(&json).expect("gpu_init should return valid JSON");
            assert!(v["name"].is_string(), "device report should have a name");
            assert!(v["backend"].is_string(), "device report should have a backend");
        }
        Err(e) => {
            let msg = e.as_string().unwrap_or_else(|| format!("{:?}", e));
            web_sys::console::warn_1(&JsValue::from_str(&format!(
                "gpu_init failed (WebGPU may not be available): {msg}"
            )));
        }
    }
}

#[wasm_bindgen_test]
async fn test_version() {
    let json = zkgpu_web::version();
    let v: serde_json::Value =
        serde_json::from_str(&json).expect("version() should return valid JSON");
    assert_eq!(v["crate_name"].as_str(), Some("zkgpu-web"));
    assert!(v["version"].is_string());
    assert_eq!(v["target"].as_str(), Some("wasm32"));
}

#[wasm_bindgen_test]
async fn test_run_smoke_suite() {
    if !try_init_gpu().await {
        return;
    }

    let request = r#"{"suite":"Smoke"}"#;
    let response_json = zkgpu_web::run_suite(request).await;

    let (ok, error) = parse_response(&response_json);
    assert!(ok, "smoke suite should pass: {:?}", error);

    let v: serde_json::Value = serde_json::from_str(&response_json).unwrap();
    let report = &v["report"];
    assert!(report.is_object(), "response should contain a report");
    assert_eq!(report["schema_version"].as_u64(), Some(1));

    let summary = &report["summary"];
    let total = summary["total_cases"].as_u64().unwrap_or(0);
    let passed = summary["passed_cases"].as_u64().unwrap_or(0);
    assert!(total >= 2, "smoke suite should have at least 2 cases");
    assert_eq!(total, passed, "all smoke cases should pass");

    // Verify cases have timing data
    if let Some(cases) = report["cases"].as_array() {
        for case in cases {
            assert!(case["passed"].as_bool().unwrap_or(false), "case should pass");
            assert!(
                case["timings"]["wall_time_ns"].is_number(),
                "case should have wall timing"
            );
        }
    }
}

#[wasm_bindgen_test]
async fn test_run_single_case() {
    if !try_init_gpu().await {
        return;
    }

    // Include all required CaseSpec fields: iterations, warmup_iterations
    let case_json = r#"{
        "name": "browser_test_fwd_4",
        "log_n": 4,
        "direction": "Forward",
        "input": "Sequential",
        "profile_gpu_timestamps": false,
        "iterations": 1,
        "warmup_iterations": 0
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value =
        serde_json::from_str(&result_json).expect("run_case should return valid JSON");

    // run_case returns a bare CaseReport (not wrapped in HarnessResponse)
    assert!(
        v["passed"].as_bool().unwrap_or(false),
        "forward NTT on sequential input should pass: {}",
        result_json
    );
    assert_eq!(v["log_n"].as_u64(), Some(4));
    assert_eq!(v["mismatch_count"].as_u64(), Some(0));
}

#[wasm_bindgen_test]
async fn test_run_inverse_case() {
    if !try_init_gpu().await {
        return;
    }

    let case_json = r#"{
        "name": "browser_test_inv_8",
        "log_n": 8,
        "direction": "Inverse",
        "input": "Sequential",
        "profile_gpu_timestamps": false,
        "iterations": 1,
        "warmup_iterations": 0
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(
        v["passed"].as_bool().unwrap_or(false),
        "inverse NTT should pass: {}",
        result_json
    );
}

#[wasm_bindgen_test]
async fn test_run_roundtrip_case() {
    if !try_init_gpu().await {
        return;
    }

    let case_json = r#"{
        "name": "browser_roundtrip_10",
        "log_n": 10,
        "direction": "Roundtrip",
        "input": {"PseudoRandomDeterministic": {"seed": 42}},
        "profile_gpu_timestamps": false,
        "iterations": 1,
        "warmup_iterations": 0
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(
        v["passed"].as_bool().unwrap_or(false),
        "roundtrip NTT should pass: {}",
        result_json
    );
}

#[wasm_bindgen_test]
async fn test_run_benchmark_case_with_iterations() {
    if !try_init_gpu().await {
        return;
    }

    // Verify warmup + multi-iteration works
    let case_json = r#"{
        "name": "browser_bench_log10",
        "log_n": 10,
        "direction": "Forward",
        "input": "Sequential",
        "profile_gpu_timestamps": false,
        "iterations": 3,
        "warmup_iterations": 1
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert!(
        v["passed"].as_bool().unwrap_or(false),
        "benchmark case with iterations should pass: {}",
        result_json
    );
    assert!(
        v["timings"]["wall_time_ns"].is_number(),
        "should have averaged wall timing"
    );
}

#[wasm_bindgen_test]
async fn test_device_info_after_init() {
    if !try_init_gpu().await {
        return;
    }

    let info_result = zkgpu_web::device_info();
    match info_result {
        Ok(json) => {
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(v["name"].is_string());
            assert!(v["backend"].is_string());
            assert_eq!(
                v["platform_class"].as_str(),
                Some("Browser"),
                "platform_class should be Browser"
            );
        }
        Err(e) => {
            panic!("device_info() failed after successful init: {:?}", e);
        }
    }
}

#[wasm_bindgen_test]
async fn test_invalid_request_returns_error() {
    let response_json = zkgpu_web::run_suite("not valid json").await;
    let (ok, error) = parse_response(&response_json);
    assert!(!ok, "invalid JSON should fail");
    assert!(
        error.is_some(),
        "error message should be present"
    );
}

#[wasm_bindgen_test]
async fn test_missing_suite_returns_error() {
    let response_json = zkgpu_web::run_suite("{}").await;
    let (ok, error) = parse_response(&response_json);
    assert!(!ok, "empty request should fail");
    assert!(error.is_some(), "should have an error message");
}

#[wasm_bindgen_test]
async fn test_run_case_invalid_json_returns_error() {
    let result_json = zkgpu_web::run_case("not json").await;
    // run_case returns error JSON when parsing fails
    let v: serde_json::Value = serde_json::from_str(&result_json).expect("should be valid JSON");
    assert_eq!(v["ok"].as_bool(), Some(false), "should indicate failure");
    assert!(v["error"].is_string(), "should have error message");
}

#[wasm_bindgen_test]
async fn test_case_report_includes_stockham_tail_metadata() {
    // Codex P3 regression coverage: PR 1 added three tail-observability
    // fields to `CaseReport` (`stockham_tail_strategy`, `stockham_tail_reason`,
    // `tail_stride_bytes`). Verify they actually surface through the wasm
    // boundary on a successful case at a `log_n` large enough that the
    // tail phase fires (LOG_BLOCK = 10, so we need log_n > LOG_BLOCK).
    //
    // BrowserWebGpu's planner policy is `stockham_only`, so a forward
    // case at log_n = 12 will go through the Stockham family and pick
    // `LocalFusedR4` from the heuristic (the new Browser>=20 hostile
    // band only fires at log_n >= 20). This is the success-path mirror
    // of the failure-path tests in `runner::tests`.
    if !try_init_gpu().await {
        return;
    }

    let case_json = r#"{
        "name": "browser_tail_metadata",
        "log_n": 12,
        "direction": "Forward",
        "input": "Sequential",
        "profile_gpu_timestamps": false,
        "iterations": 1,
        "warmup_iterations": 0
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value =
        serde_json::from_str(&result_json).expect("run_case should return valid JSON");

    assert!(
        v["passed"].as_bool().unwrap_or(false),
        "case should pass: {}",
        result_json
    );

    // The three tail fields must be present on the JSON object; Browser
    // policy guarantees a Stockham plan, and log_n > LOG_BLOCK guarantees
    // a tail phase, so `stockham_tail_strategy` must be a string.
    let strategy = v["stockham_tail_strategy"].as_str();
    assert!(
        strategy.is_some(),
        "stockham_tail_strategy must be populated for log_n > LOG_BLOCK \
         on Browser policy: {}",
        result_json
    );
    assert!(
        matches!(strategy, Some("LocalFusedR4") | Some("GlobalOnlyR4")),
        "tail strategy must be one of the two known variants, got {:?}",
        strategy
    );
    assert!(
        v["stockham_tail_reason"].as_str().is_some(),
        "stockham_tail_reason must be populated alongside the strategy: {}",
        result_json
    );

    // tail_stride_bytes is set only for LocalFusedR4 (the strided gather
    // is what makes it interesting); GlobalOnlyR4 has no per-thread stride
    // to report. Both shapes are valid.
    match strategy {
        Some("LocalFusedR4") => assert!(
            v["tail_stride_bytes"].is_number(),
            "LocalFusedR4 must report tail_stride_bytes: {}",
            result_json
        ),
        Some("GlobalOnlyR4") => assert!(
            v["tail_stride_bytes"].is_null(),
            "GlobalOnlyR4 must leave tail_stride_bytes null: {}",
            result_json
        ),
        _ => unreachable!("guarded by earlier assert"),
    }
}

#[wasm_bindgen_test]
async fn test_run_case_invalid_log_n_returns_well_formed_error() {
    // Companion to the success-path tail-metadata test above. PR 1
    // promised that *failed* cases also surface a structured CaseReport
    // (with `error` populated and `passed: false`) rather than throwing.
    // log_n = 0 is rejected by the planner, so this hits the
    // `make_plan` failure branch in `run_direction`. The pre-plan
    // failure path correctly leaves the tail fields null — matches the
    // `case_error_family_keeps_tail_fields_none` native test.
    if !try_init_gpu().await {
        return;
    }

    let case_json = r#"{
        "name": "browser_invalid_log_n",
        "log_n": 0,
        "direction": "Forward",
        "input": "Sequential",
        "profile_gpu_timestamps": false,
        "iterations": 1,
        "warmup_iterations": 0
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value = serde_json::from_str(&result_json).unwrap();

    assert_eq!(
        v["passed"].as_bool(),
        Some(false),
        "log_n=0 must fail: {}",
        result_json
    );
    assert!(
        v["error"].is_string(),
        "failed case must have an error message: {}",
        result_json
    );
    // Pre-plan failures have nothing to report yet; the new tail fields
    // remain null. (The interesting tail-preserving failure paths —
    // measurement crash on Xclipse/Mali — can't be reliably triggered
    // from a browser test, but the native unit tests in `runner::tests`
    // cover them via direct helper calls.)
    assert!(
        v["stockham_tail_strategy"].is_null(),
        "pre-plan failure should leave stockham_tail_strategy null: {}",
        result_json
    );
    assert!(
        v["stockham_tail_reason"].is_null(),
        "pre-plan failure should leave stockham_tail_reason null: {}",
        result_json
    );
    assert!(
        v["tail_stride_bytes"].is_null(),
        "pre-plan failure should leave tail_stride_bytes null: {}",
        result_json
    );
}

#[wasm_bindgen_test]
async fn test_run_case_omitted_iterations_uses_defaults() {
    if !try_init_gpu().await {
        return;
    }

    // iterations and warmup_iterations are now optional with serde defaults
    // (iterations=1, warmup_iterations=0). This should succeed.
    let case_json = r#"{
        "name": "browser_defaults",
        "log_n": 4,
        "direction": "Forward",
        "input": "Sequential"
    }"#;

    let result_json = zkgpu_web::run_case(case_json).await;
    let v: serde_json::Value = serde_json::from_str(&result_json).expect("should be valid JSON");

    // With serde defaults, this should parse as a valid CaseSpec and succeed.
    assert!(
        v["passed"].as_bool().unwrap_or(false),
        "case with default iterations should pass: {}",
        result_json
    );
}
