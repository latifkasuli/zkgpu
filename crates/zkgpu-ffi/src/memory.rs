use std::ffi::{c_char, CStr, CString};

pub fn json_to_c_string(json: String) -> *mut c_char {
    CString::new(json)
        .expect("serialized JSON must not contain interior NUL bytes")
        .into_raw()
}

pub unsafe fn c_string_to_json(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("request_json pointer must not be null".to_string());
    }

    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes();
    let text = std::str::from_utf8(bytes)
        .map_err(|err| format!("request_json must be valid UTF-8: {err}"))?;
    Ok(text.to_owned())
}

pub unsafe fn free_c_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = unsafe { CString::from_raw(ptr) };
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;

    #[test]
    fn roundtrip_json_string() {
        let ptr = json_to_c_string("{\"ok\":true}".to_string());
        let text = unsafe { c_string_to_json(ptr.cast_const()) }.expect("valid C string");
        assert_eq!(text, "{\"ok\":true}");
        unsafe { free_c_string(ptr) };
    }

    #[test]
    fn rejects_null_pointer() {
        let err = unsafe { c_string_to_json(std::ptr::null()) }.expect_err("null should fail");
        assert!(err.contains("must not be null"));
    }

    #[test]
    fn rejects_invalid_utf8() {
        let raw = CString::new(vec![0xff]).expect("single non-NUL byte");
        let err = unsafe { c_string_to_json(raw.as_ptr()) }.expect_err("invalid UTF-8 should fail");
        assert!(err.contains("valid UTF-8"));
    }
}
