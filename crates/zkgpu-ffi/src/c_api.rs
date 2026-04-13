use std::ffi::c_char;

use crate::json::{
    error_response, fallback_version_json, run_request_json, serialize_version_response,
};
use crate::memory::{c_string_to_json, free_c_string, json_to_c_string};
use crate::panic::{catch_json, catch_response_json};

#[no_mangle]
/// Run a harness request encoded as UTF-8 JSON and return a newly allocated
/// UTF-8 JSON response string.
///
/// # Safety
///
/// `request_json` must be either null or a valid, NUL-terminated C string
/// pointer that remains alive for the duration of the call. The returned
/// pointer must be released exactly once with [`zkgpu_free_string`].
pub unsafe extern "C" fn zkgpu_run_request_json(request_json: *const c_char) -> *mut c_char {
    let response_json = catch_response_json(|| {
        let request_json = match unsafe { c_string_to_json(request_json) } {
            Ok(json) => json,
            Err(err) => return error_response(err),
        };
        run_request_json(&request_json)
    });
    json_to_c_string(response_json)
}

#[no_mangle]
pub extern "C" fn zkgpu_get_version_json() -> *mut c_char {
    let version_json = catch_json(serialize_version_response, |err| {
        fallback_version_json(&err)
    });
    json_to_c_string(version_json)
}

#[no_mangle]
/// Free a string previously returned by [`zkgpu_run_request_json`] or
/// [`zkgpu_get_version_json`].
///
/// # Safety
///
/// `ptr` must be either null or a pointer returned by this library via
/// `CString::into_raw`. Passing any other pointer, or freeing the same pointer
/// more than once, is undefined behavior.
pub unsafe extern "C" fn zkgpu_free_string(ptr: *mut c_char) {
    unsafe { free_c_string(ptr) };
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use serde_json::Value;

    use super::*;

    fn read_and_free(ptr: *mut c_char) -> String {
        let text = unsafe {
            CStr::from_ptr(ptr.cast_const())
                .to_str()
                .expect("valid UTF-8")
                .to_owned()
        };
        unsafe { zkgpu_free_string(ptr) };
        text
    }

    #[test]
    fn ffi_rejects_null_request_pointer() {
        let ptr = unsafe { zkgpu_run_request_json(std::ptr::null()) };
        let json = read_and_free(ptr);
        let parsed: Value = serde_json::from_str(&json).expect("valid error JSON");
        assert_eq!(parsed["ok"], Value::Bool(false));
        assert!(parsed["error"]
            .as_str()
            .unwrap_or_default()
            .contains("must not be null"));
    }

    #[test]
    fn ffi_rejects_invalid_request_json() {
        let input = CString::new("{").expect("no NUL");
        let ptr = unsafe { zkgpu_run_request_json(input.as_ptr()) };
        let json = read_and_free(ptr);
        let parsed: Value = serde_json::from_str(&json).expect("valid error JSON");
        assert_eq!(parsed["ok"], Value::Bool(false));
        assert!(parsed["error"]
            .as_str()
            .unwrap_or_default()
            .contains("invalid harness request JSON"));
    }

    #[test]
    fn ffi_version_json_is_parseable() {
        let ptr = zkgpu_get_version_json();
        let json = read_and_free(ptr);
        let parsed: Value = serde_json::from_str(&json).expect("valid version JSON");
        assert_eq!(
            parsed["crate_name"],
            Value::String(env!("CARGO_PKG_NAME").to_string())
        );
        assert_eq!(
            parsed["version"],
            Value::String(env!("CARGO_PKG_VERSION").to_string())
        );
    }
}
