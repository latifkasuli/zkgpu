#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/build"
TARGET_BASE="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"

build_target() {
  target="$1"
  out_subdir="$2"

  echo "Building zkgpu-ffi for ${target}..."
  cargo build \
    --manifest-path "${REPO_ROOT}/Cargo.toml" \
    -p zkgpu-ffi \
    --release \
    --target "${target}"

  mkdir -p "${OUT_DIR}/${out_subdir}"
  cp \
    "${TARGET_BASE}/${target}/release/libzkgpu_ffi.a" \
    "${OUT_DIR}/${out_subdir}/libzkgpu_ffi.a"
}

build_target aarch64-apple-ios device
build_target aarch64-apple-ios-sim simulator-aarch64

echo
echo "Static libraries written to ${OUT_DIR}"
echo "If full Xcode is installed, package them into an XCFramework next."
