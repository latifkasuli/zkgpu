Place Rust-built `libzkgpu_ffi.so` artifacts here under ABI-specific subdirectories.

Expected layout:

```text
app/src/main/jniLibs/
├── arm64-v8a/libzkgpu_ffi.so
└── x86_64/libzkgpu_ffi.so
```

Use the commands in [`rust/README.md`](../../../../rust/README.md) to produce these files.
