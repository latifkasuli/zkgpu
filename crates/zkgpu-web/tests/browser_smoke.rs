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
