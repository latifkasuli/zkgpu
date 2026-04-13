#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../../.." && pwd)"
HARNESS_DIR="${SCRIPT_DIR}/.."
PROJECT="${HARNESS_DIR}/ZkgpuHarness.xcodeproj"
RESULTS_DIR="${REPO_ROOT}/test-results/ios"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run zkgpu iOS harness tests on a connected physical device.

Options:
  --device ID        CoreDevice UUID (auto-detected if one device connected)
  --suite NAME       Run a single test: smoke, validation, benchmark, crossover,
                     stockham, fourstep, version, or all (default: all)
  --skip-build-rust  Skip Rust static library rebuild
  --skip-build-app   Skip Xcode build (use existing build products)
  --results-dir DIR  Override results output directory
  -h, --help         Show this help

Examples:
  $(basename "$0")                          # build + run all tests
  $(basename "$0") --suite smoke            # quick smoke test
  $(basename "$0") --suite crossover        # crossover benchmark only
  $(basename "$0") --skip-build-rust        # skip Rust rebuild
EOF
  exit 0
}

DEVICE_ID=""
SUITE="all"
SKIP_RUST=false
SKIP_APP=false

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE_ID="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --skip-build-rust) SKIP_RUST=true; shift ;;
    --skip-build-app) SKIP_APP=true; shift ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

auto_detect_device() {
  local devices
  devices=$(xcrun xctrace list devices 2>&1 \
    | sed -n '/^== Devices ==$/,/^== /p' \
    | grep -v '^==' \
    | grep -v 'MacBook\|Mac Pro\|Mac mini\|iMac\|Mac Studio' \
    | grep -oE '\([A-F0-9-]+\)' \
    | tr -d '()' || true)

  local count
  count=$(echo "$devices" | grep -c . || true)

  if [ "$count" -eq 0 ]; then
    echo "ERROR: No iOS device found. Connect an iPhone and enable Developer Mode."
    exit 1
  elif [ "$count" -gt 1 ]; then
    echo "ERROR: Multiple iOS devices found. Specify one with --device ID:"
    echo "$devices"
    exit 1
  fi

  DEVICE_ID="$devices"
}

if [ -z "$DEVICE_ID" ]; then
  auto_detect_device
fi

echo "=== zkgpu iOS Test Runner ==="
echo "Device:  $DEVICE_ID"
echo "Suite:   $SUITE"
echo "Results: $RESULTS_DIR"
echo ""

# Step 1: Build Rust static libraries
if [ "$SKIP_RUST" = false ]; then
  echo "--- Building Rust static libraries ---"
  "${SCRIPT_DIR}/build_ios_staticlibs.sh"
  echo ""
fi

# Step 2: Build the iOS app + tests
if [ "$SKIP_APP" = false ]; then
  echo "--- Building ZkgpuHarness + Tests ---"
  xcodebuild \
    -project "$PROJECT" \
    -scheme ZkgpuHarness \
    -destination "id=$DEVICE_ID" \
    -allowProvisioningUpdates \
    build-for-testing 2>&1 | \
    grep -E '(BUILD|error:|warning:.*ZkgpuHarness|Compile|Link|Sign|Test)' || true

  if [ "${PIPESTATUS[0]:-0}" -ne 0 ]; then
    echo "ERROR: build-for-testing failed. Running full build to show errors..."
    xcodebuild \
      -project "$PROJECT" \
      -scheme ZkgpuHarness \
      -destination "id=$DEVICE_ID" \
      -allowProvisioningUpdates \
      build-for-testing 2>&1 | tail -30
    exit 1
  fi
  echo ""
fi

# Map suite name to xcodebuild test filter
resolve_filter() {
  case "$1" in
    all)        echo "" ;;
    version)    echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testVersionHandshakeParses" ;;
    smoke)      echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testSmokeSuiteSucceeds -only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testSmokeSuiteIncludesDeviceMetadata" ;;
    validation) echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testValidationSuiteSucceeds" ;;
    benchmark)  echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testBenchmarkSuiteSucceeds" ;;
    crossover)  echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testCrossoverBenchmarkSucceeds" ;;
    stockham)   echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testSmokeSuiteStockhamFamily" ;;
    fourstep)   echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testSmokeSuiteFourStepFamily" ;;
    crossover-stockham) echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testCrossoverStockhamFamily" ;;
    crossover-fourstep) echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testCrossoverFourStepFamily" ;;
    crossover-both) echo "-only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testCrossoverStockhamFamily -only-testing:ZkgpuHarnessTests/ZkgpuHarnessTests/testCrossoverFourStepFamily" ;;
    *) echo "Unknown suite: $1" >&2; exit 1 ;;
  esac
}

FILTER=$(resolve_filter "$SUITE")

# Step 3: Run tests on device
echo "--- Running tests on device ---"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${RESULTS_DIR}/ios-test-${SUITE}-${TIMESTAMP}.log"

# shellcheck disable=SC2086
xcodebuild \
  -project "$PROJECT" \
  -scheme ZkgpuHarness \
  -destination "id=$DEVICE_ID" \
  -allowProvisioningUpdates \
  -resultBundlePath "${RESULTS_DIR}/ios-${SUITE}-${TIMESTAMP}.xcresult" \
  test-without-building \
  $FILTER 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "--- Summary ---"
echo "Log:    $LOG_FILE"
echo "Result: ${RESULTS_DIR}/ios-${SUITE}-${TIMESTAMP}.xcresult"

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "Status: ALL TESTS PASSED"
else
  echo "Status: SOME TESTS FAILED (exit $EXIT_CODE)"
fi

exit "$EXIT_CODE"
