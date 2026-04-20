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
mod hash_runner;
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
                hash_report: None,
                case_report: None,
                error: None,
            };
            serde_json::to_string(&response).unwrap_or_else(|e| {
                to_error_json(&format!("failed to serialize response: {e}"))
            })
        }
        Err(e) => to_error_json(&e),
    }
}

/// Run a single test case by JSON spec. Returns a JSON-encoded
/// [`HarnessResponse`]: success carries the [`CaseReport`] on
/// `case_report`, failure carries `ok: false` + `error`. Shape mirrors
/// `run_suite` and `run_hash` so browser callers parse all three wasm
/// entry points uniformly.
///
/// Accepts two JSON shapes on input, tried in this order:
///   1. `SingleCaseRequest` envelope: `{"case": {...}, "field": "goldilocks"}`
///      (Phase E.2.b+) — lets a browser caller pick the target field
///      for a one-off case, matching `run_suite`'s `spec.field`.
///   2. Legacy bare `CaseSpec`: `{"name": ..., "log_n": ..., ...}`
///      (pre-E.2) — assumed `Field::BabyBear` for backward
///      compatibility with existing JS consumers.
///
/// Ambiguity is impossible: `SingleCaseRequest` has a top-level `case`
/// key that bare `CaseSpec` lacks.
#[wasm_bindgen]
pub async fn run_case(case_json: &str) -> String {
    // Try envelope first — the envelope's required `case` key makes this
    // non-ambiguous against a bare CaseSpec. On failure, fall back to
    // parsing the legacy bare shape and assume BabyBear.
    let (case, field) = if let Ok(req) =
        serde_json::from_str::<zkgpu_report::SingleCaseRequest>(case_json)
    {
        (req.case, req.field)
    } else {
        match serde_json::from_str::<zkgpu_report::CaseSpec>(case_json) {
            Ok(c) => (c, zkgpu_report::Field::BabyBear),
            Err(e) => {
                return to_error_json(&format!(
                    "invalid case spec JSON (tried SingleCaseRequest envelope \
                     and bare CaseSpec): {e}"
                ));
            }
        }
    };

    match runner::run_single_case_async(&case, field).await {
        Ok(report) => {
            let response = HarnessResponse {
                ok: true,
                report: None,
                hash_report: None,
                case_report: Some(report),
                error: None,
            };
            serde_json::to_string(&response).unwrap_or_else(|e| {
                to_error_json(&format!("failed to serialize case response: {e}"))
            })
        }
        Err(e) => to_error_json(&e),
    }
}

/// Run a hash suite by JSON-encoded [`HashSpec`].
///
/// Phase F.3.d — parallels `run_suite` but for the Poseidon2 surface.
/// Takes a raw `HashSpec` (no envelope) and returns a JSON-encoded
/// [`HarnessResponse`]. Success carries `ok: true` with the
/// [`HashSuiteReport`] on `hash_report`; failure carries `ok: false`
/// with `error`. Shape mirrors `run_suite`, so browser workers can
/// parse both entry points uniformly.
///
/// Wall-only timings until a future F.3.* sub-phase adds
/// `execute_profiled_async` to the Poseidon2 plans.
#[wasm_bindgen]
pub async fn run_hash(spec_json: &str) -> String {
    let spec: zkgpu_report::HashSpec = match serde_json::from_str(spec_json) {
        Ok(s) => s,
        Err(e) => {
            return to_error_json(&format!("invalid HashSpec JSON: {e}"));
        }
    };

    match hash_runner::run_hash_suite_async(&spec).await {
        Ok(report) => {
            let response = HarnessResponse {
                ok: true,
                report: None,
                hash_report: Some(report),
                case_report: None,
                error: None,
            };
            serde_json::to_string(&response).unwrap_or_else(|e| {
                to_error_json(&format!("failed to serialize hash response: {e}"))
            })
        }
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
        hash_report: None,
        case_report: None,
        error: Some(msg.to_string()),
    };
    serde_json::to_string(&response)
        .unwrap_or_else(|_| format!(r#"{{"ok":false,"error":"{}"}}"#, msg.replace('"', "\\\""),))
}
