# `ios-harness`

iOS host shell for `zkgpu` benchmarking and validation on physical Apple devices.

What is included here:

- a `cbindgen`-generated C header for the stable `zkgpu-ffi` surface
- a Swift bridge (`ZkgpuBridge`) that owns pointer lifecycle and JSON handling
- a SwiftUI app shell with smoke, validation, benchmark, and crossover presets
- XCTest coverage for the bridge path
- a Rust build helper for iOS static libraries

What is intentionally not checked in:

- `ZkgpuHarness.xcodeproj` (generated locally per machine)

## Layout

```text
apps/ios-harness/
├── README.md
├── ZkgpuHarness/
│   ├── AppDelegate.swift
│   ├── ContentView.swift
│   ├── ZkgpuBridge.swift
│   ├── ZkgpuHarness-Bridging-Header.h
│   ├── ZkgpuHarnessApp.swift
│   └── Resources/
│       └── Info.plist
├── ZkgpuHarnessTests/
│   └── ZkgpuHarnessTests.swift
└── RustBridge/
    ├── build_ios_staticlibs.sh
    └── include/
        └── zkgpu_ffi.h          (cbindgen-generated)
```

## Prerequisites

- macOS with full Xcode installed (not just Command Line Tools)
- `xcode-select -s /Applications/Xcode.app/Contents/Developer`
- Rust iOS targets: `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`
- `cbindgen` (for header regeneration): `cargo install cbindgen`

## Bootstrap

1. Build the Rust static libraries:
   ```sh
   ./RustBridge/build_ios_staticlibs.sh
   ```
2. Create a new iOS App project named `ZkgpuHarness` in Xcode under this directory.
3. Add the files from `ZkgpuHarness/` and `ZkgpuHarnessTests/`.
4. Set the Objective-C bridging header to:
   `ZkgpuHarness/ZkgpuHarness-Bridging-Header.h`
5. Add a header search path:
   `$(SRCROOT)/RustBridge/include`
6. Add the generated `libzkgpu_ffi.a` for the target architecture to the app target.
7. Run the app on a physical iPhone.

## Regenerating the C header

```sh
cd ../../crates/zkgpu-ffi
cbindgen --config cbindgen.toml --output include/zkgpu_ffi.h
cp include/zkgpu_ffi.h ../../apps/ios-harness/RustBridge/include/zkgpu_ffi.h
```

## Notes

- The bridge uses the JSON FFI boundary from `crates/zkgpu-ffi`.
- The Swift side is intentionally thin. Validation logic stays in Rust.
- The crossover benchmark runs `log_n` 18-22 (forward + inverse), matching the Android harness.
- `build_ios_staticlibs.sh` respects `CARGO_TARGET_DIR` if set.
