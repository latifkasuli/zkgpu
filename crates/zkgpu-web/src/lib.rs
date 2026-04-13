//! WebAssembly bindings for zkgpu browser harness.
//!
//! Provides async entry points for initializing a GPU device, running
//! NTT validation suites, and collecting reports — all in a browser
//! WebGPU context. Designed to run inside a dedicated Web Worker.
//!
//! # API
//!
//! All public functions are `#[wasm_bindgen]` and return JSON strings
//! via `JsValue`. The caller (JavaScript worker or main thread) parses
//! the JSON and relays results.
//!
//! ```js
//! import init, { gpu_init, device_info, run_suite, run_case } from './zkgpu_web.js';
//! await init();
//! const info = await gpu_init();
//! const report = await run_suite('{"suite":"Smoke"}');
//! ```

mod device;
mod inputs;
mod runner;
mod validation;

use wasm_bindgen::prelude::*;
use zkgpu_report::{HarnessRequest, HarnessResponse, SuiteKind};

/// Initialize the GPU device. Must be called before any other function.
///
/// Returns a JSON-encoded `DeviceReport` on success.
#[wasm_bindgen]
pub async fn gpu_init() -> Result<String, JsValue> {
    // Install panic hook on first call — redirects Rust panics to
    // console.error so wasm "unreachable" traps show the real message.
    console_error_panic_hook::set_once();

    device::init_device()
        .await
        .map_err(|e| JsValue::from_str(&e))
}

/// Return cached device info as JSON. Panics if `gpu_init` was not called.
#[wasm_bindgen]
pub fn device_info() -> Result<String, JsValue> {
    device::get_device_info_json().map_err(|e| JsValue::from_str(&e))
}

/// Run a test suite from a JSON-encoded `HarnessRequest`.
///
/// Returns a JSON-encoded `HarnessResponse`.
#[wasm_bindgen]
pub async fn run_suite(request_json: &str) -> String {
    let request: HarnessRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return to_error_json(&format!("invalid harness request JSON: {e}"));
        }
    };

    let spec = if let Some(spec) = request.spec {
        spec
    } else {
        let mut suite = match request.suite {
            Some(SuiteKind::Smoke) => zkgpu_report::smoke_suite(),
            Some(SuiteKind::Validation) => zkgpu_report::validation_suite(),
            Some(SuiteKind::Benchmark) => zkgpu_report::benchmark_suite(),
            Some(SuiteKind::Soak) => {
                return to_error_json(
                    "soak benchmark is not supported in the browser harness; \
                     use the native testkit runner instead",
                );
            }
            None => {
                return to_error_json(
                    "harness request must include either `spec` or `suite`",
                );
            }
        };
        if let Some(family) = request.family_override {
            suite.family_override = family;
        }
        suite
    };

    match runner::run_suite_async(&spec).await {
        Ok(report) => {
            let response = HarnessResponse {
                ok: true,
                report: Some(report),
                error: None,
            };
            serde_json::to_string(&response).unwrap_or_else(|e| {
                to_error_json(&format!("failed to serialize response: {e}"))
            })
        }
        Err(e) => to_error_json(&e),
    }
}

/// Run a single test case by JSON spec. Returns JSON `CaseReport`.
#[wasm_bindgen]
pub async fn run_case(case_json: &str) -> String {
    let case: zkgpu_report::CaseSpec = match serde_json::from_str(case_json) {
        Ok(c) => c,
        Err(e) => {
            return to_error_json(&format!("invalid case spec JSON: {e}"));
        }
    };

    match runner::run_single_case_async(&case).await {
        Ok(report) => serde_json::to_string(&report).unwrap_or_else(|e| {
            to_error_json(&format!("failed to serialize case report: {e}"))
        }),
        Err(e) => to_error_json(&e),
    }
}

/// Return version info as JSON.
#[wasm_bindgen]
pub fn version() -> String {
    serde_json::to_string(&serde_json::json!({
        "crate_name": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "target": "wasm32",
    }))
    .unwrap_or_else(|_| r#"{"error":"failed to serialize version"}"#.to_string())
}

fn to_error_json(msg: &str) -> String {
    let response = HarnessResponse {
        ok: false,
        report: None,
        error: Some(msg.to_string()),
    };
    serde_json::to_string(&response)
        .unwrap_or_else(|_| format!(r#"{{"ok":false,"error":"{}"}}"#, msg.replace('"', "\\\""),))
}
