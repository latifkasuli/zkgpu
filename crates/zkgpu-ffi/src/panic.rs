use std::panic::{catch_unwind, AssertUnwindSafe};

use zkgpu_report::HarnessResponse;

use crate::json::fallback_error_json;

pub fn catch_response_json<F>(f: F) -> String
where
    F: FnOnce() -> HarnessResponse,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(response) => crate::json::serialize_response(&response),
        Err(payload) => fallback_error_json(&format!(
            "panic across FFI boundary: {}",
            panic_message(&payload)
        )),
    }
}

pub fn catch_json<F>(f: F, fallback: impl FnOnce(String) -> String) -> String
where
    F: FnOnce() -> String,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(json) => json,
        Err(payload) => fallback(format!(
            "panic across FFI boundary: {}",
            panic_message(&payload)
        )),
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::json::error_response;

    #[test]
    fn catch_response_json_converts_panics_to_error_json() {
        let json = catch_response_json(|| panic!("boom"));
        let parsed: Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["ok"], Value::Bool(false));
        assert!(parsed["error"]
            .as_str()
            .unwrap_or_default()
            .contains("panic across FFI boundary"));
    }

    #[test]
    fn catch_json_uses_fallback_on_panic() {
        let json = catch_json(
            || panic!("boom"),
            |msg| crate::json::serialize_response(&error_response(msg)),
        );
        let parsed: Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["ok"], Value::Bool(false));
    }
}
