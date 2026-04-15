use serde_json::json;
use zkgpu_testkit::run_suite;

// Re-export for backward compatibility (ffi/lib.rs re-exports these).
pub use zkgpu_report::{HarnessRequest, HarnessResponse, VersionResponse};
use zkgpu_report::SuiteKind;

pub const FFI_API_VERSION: u32 = 1;

pub fn run_request(request: HarnessRequest) -> HarnessResponse {
    let result = if let Some(mut spec) = request.spec {
        // A top-level `stockham_tail_override` on the request always wins
        // over whatever the spec itself carries — it lets harness callers
        // flip the knob without rebuilding the spec. `None` leaves the
        // spec's value (often `Auto` via `#[serde(default)]`) untouched.
        if let Some(tail) = request.stockham_tail_override {
            spec.stockham_tail_override = tail;
        }
        run_suite(&spec)
    } else {
        let mut suite = match request.suite {
            Some(SuiteKind::Smoke) => zkgpu_report::smoke_suite(),
            Some(SuiteKind::Validation) => zkgpu_report::validation_suite(),
            Some(SuiteKind::Benchmark) => zkgpu_report::benchmark_suite(),
            Some(SuiteKind::Soak) => {
                return error_response(
                    "soak benchmark requires the SoakSpec path; use `run_soak_suite()` \
                     from zkgpu-testkit instead of the HarnessRequest preset"
                        .to_string(),
                );
            }
            None => {
                return error_response(
                    "harness request must include either `spec` or `suite`".to_string(),
                );
            }
        };
        if let Some(family) = request.family_override {
            suite.family_override = family;
        }
        if let Some(tail) = request.stockham_tail_override {
            suite.stockham_tail_override = tail;
        }
        run_suite(&suite)
    };

    match result {
        Ok(report) => HarnessResponse {
            ok: true,
            report: Some(report),
            error: None,
        },
        Err(err) => error_response(err.to_string()),
    }
}

pub fn run_request_json(request_json: &str) -> HarnessResponse {
    match serde_json::from_str::<HarnessRequest>(request_json) {
        Ok(request) => run_request(request),
        Err(err) => error_response(format!("invalid harness request JSON: {err}")),
    }
}

pub fn serialize_response(response: &HarnessResponse) -> String {
    match serde_json::to_string(response) {
        Ok(json) => json,
        Err(err) => fallback_error_json(&format!("failed to serialize harness response: {err}")),
    }
}

pub fn version_response() -> VersionResponse {
    VersionResponse {
        crate_name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        ffi_api_version: FFI_API_VERSION,
    }
}

pub fn serialize_version_response() -> String {
    match serde_json::to_string(&version_response()) {
        Ok(json) => json,
        Err(err) => fallback_version_json(&format!("failed to serialize version response: {err}")),
    }
}

pub fn error_response(error: String) -> HarnessResponse {
    HarnessResponse {
        ok: false,
        report: None,
        error: Some(error),
    }
}

pub fn fallback_error_json(error: &str) -> String {
    format!(
        "{{\"ok\":false,\"report\":null,\"error\":{}}}",
        serde_json::to_string(error).unwrap_or_else(|_| "\"unknown ffi error\"".to_string())
    )
}

pub fn fallback_version_json(error: &str) -> String {
    serde_json::to_string(&json!({
        "crate_name": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "ffi_api_version": FFI_API_VERSION,
        "error": error,
    }))
    .unwrap_or_else(|_| fallback_error_json("failed to serialize fallback version response"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkgpu_report::{FamilyOverride, SuiteSpec};

    #[test]
    fn run_request_json_rejects_invalid_json() {
        let response = run_request_json("{");
        assert!(!response.ok);
        assert!(response.report.is_none());
        assert!(response
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("invalid harness request JSON"));
    }

    #[test]
    fn run_request_requires_suite_or_spec() {
        let response = run_request(HarnessRequest {
            suite: None,
            spec: None,
            family_override: None,
            stockham_tail_override: None,
        });
        assert!(!response.ok);
        assert!(response.report.is_none());
        assert_eq!(
            response.error.as_deref(),
            Some("harness request must include either `spec` or `suite`")
        );
    }

    #[test]
    fn spec_takes_precedence_over_suite() {
        let response = run_request(HarnessRequest {
            suite: Some(SuiteKind::Smoke),
            spec: Some(SuiteSpec {
                kind: SuiteKind::Validation,
                cases: Vec::new(),
                fail_fast: true,
                family_override: FamilyOverride::Auto,
                stockham_tail_override: zkgpu_report::StockhamTailOverride::Auto,
            }),
            family_override: None,
            stockham_tail_override: None,
        });
        assert!(!response.ok);
        assert_eq!(
            response.error.as_deref(),
            Some("suite must contain at least one case")
        );
    }

    #[test]
    fn request_level_tail_override_propagates_to_spec() {
        // When the HarnessRequest carries a tail override AND a spec, the
        // request-level value should overwrite spec.stockham_tail_override.
        // We can't check runtime effect here without a device, but we can
        // verify the public field exists and round-trips through JSON.
        let req = HarnessRequest {
            suite: None,
            spec: Some(SuiteSpec {
                kind: SuiteKind::Validation,
                cases: Vec::new(),
                fail_fast: true,
                family_override: FamilyOverride::Auto,
                stockham_tail_override: zkgpu_report::StockhamTailOverride::Auto,
            }),
            family_override: None,
            stockham_tail_override: Some(zkgpu_report::StockhamTailOverride::Global),
        };
        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: HarnessRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.stockham_tail_override,
            Some(zkgpu_report::StockhamTailOverride::Global)
        );
    }

    #[test]
    fn version_response_serializes() {
        let json = serialize_version_response();
        let parsed: VersionResponse = serde_json::from_str(&json).expect("valid version JSON");
        assert_eq!(parsed.crate_name, env!("CARGO_PKG_NAME"));
        assert_eq!(parsed.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(parsed.ffi_api_version, FFI_API_VERSION);
    }

    #[test]
    fn fallback_version_json_is_valid_json() {
        let json = fallback_version_json("quote: \" and slash: \\");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("valid fallback version JSON");
        assert_eq!(parsed["crate_name"], env!("CARGO_PKG_NAME"));
        assert_eq!(parsed["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(parsed["ffi_api_version"], FFI_API_VERSION);
        assert_eq!(parsed["error"], "quote: \" and slash: \\");
    }
}
