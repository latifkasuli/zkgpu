# Rust Bridge For `ios-harness`

This directory contains the C header and Rust build helper for the iOS harness.

## Exposed ABI

The iOS app depends on exactly three symbols from `zkgpu-ffi`:

- `zkgpu_run_request_json`
- `zkgpu_get_version_json`
- `zkgpu_free_string`

The Swift bridge owns:

- JSON encoding/decoding
- C string conversion
- memory cleanup via `zkgpu_free_string`

## Build Flow

From the repo root:

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
apps/ios-harness/RustBridge/build_ios_staticlibs.sh
```

The helper script writes static libraries to:

```text
apps/ios-harness/RustBridge/build/
├── device/libzkgpu_ffi.a
├── simulator-aarch64/libzkgpu_ffi.a
└── simulator-x86_64/libzkgpu_ffi.a
```

## Xcode Integration

In Xcode:

1. Add `include/zkgpu_ffi.h` to the project or header search path.
2. Add `libzkgpu_ffi.a` for the active target architecture to the app target.
3. Set the bridging header to:
   `ZkgpuHarness/ZkgpuHarness-Bridging-Header.h`

If full Xcode is available, the preferred next step is to wrap the three static
libs into an `.xcframework`. This scaffold stops short of that because
`xcodebuild -create-xcframework` is unavailable in the current environment.
