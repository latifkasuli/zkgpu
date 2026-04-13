# Rust Build Notes For `apps/android-harness`

This app expects the Rust FFI library from [`crates/zkgpu-ffi`](/Users/latifkasuli/web3/zkgpu/crates/zkgpu-ffi) to be packaged as `libzkgpu_ffi.so` under `app/src/main/jniLibs/<abi>/`.

## Targets

Recommended Android Rust targets:

- `aarch64-linux-android` for physical phones
- `x86_64-linux-android` for emulator/debug if needed

## Tooling

Install:

```bash
rustup target add aarch64-linux-android x86_64-linux-android
cargo install cargo-ndk
```

## Build From Repo Root

From [`/Users/latifkasuli/web3/zkgpu`](/Users/latifkasuli/web3/zkgpu):

```bash
cargo ndk -t arm64-v8a -t x86_64 -p 29 -o apps/android-harness/app/src/main/jniLibs build -p zkgpu-ffi --release
```

That should produce:

```text
apps/android-harness/app/src/main/jniLibs/arm64-v8a/libzkgpu_ffi.so
apps/android-harness/app/src/main/jniLibs/x86_64/libzkgpu_ffi.so
```

## First Bring-Up Flow

1. Build the Rust `.so` files with `cargo-ndk`.
2. Open `apps/android-harness` in Android Studio.
3. Install on a USB-connected Android phone first.
4. Run the smoke suite.
5. Pull logs with `adb logcat | rg ZkgpuHarness`.
6. Pull the saved report from the app sandbox if needed.

The JNI shim library is built by the Android app itself. The Rust library is loaded at runtime via `dlopen("libzkgpu_ffi.so")`, so Android Studio sync can succeed even before the Rust artifacts are present.
