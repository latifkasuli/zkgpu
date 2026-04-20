import XCTest

/// Phase F.3.f — Poseidon2 XCUITest harness for BrowserStack.
///
/// BrowserStack App Automate only accepts UI-test (`XCUITest`)
/// targets; unit tests wrapped inside the host app's PlugIns/ folder
/// can't be uploaded as a Runner.ipa. This target compiles
/// `ZkgpuBridge.swift` into the UI-test process directly (see
/// `project.yml`) so Poseidon2 suites run without needing any UI
/// interaction — the tests call the Rust FFI from the UITests binary.
///
/// Local `xcodebuild test` keeps using the shipped unit-test target
/// (`ZkgpuHarnessTests`) for the NTT surface. This UI target only
/// hosts the Poseidon2 cases that need BrowserStack validation.
final class ZkgpuHarnessPoseidon2UITests: XCTestCase {

    // MARK: - Poseidon2 smoke

    func testPoseidon2BabyBearSmokeSuitePasses() throws {
        let request = ZkgpuBridge.poseidon2SmokeRequest(field: .babyBear)
        let response = try ZkgpuBridge.run(request)
        XCTAssertTrue(response.ok,
            response.error ?? "poseidon2 babybear smoke failed")
        let report = try XCTUnwrap(response.hashReport,
            "expected hash_report on hash path")
        XCTAssertEqual(report.summary.failedCases, 0,
            "failed=\(report.summary.failedCases)/\(report.summary.totalCases)")
        XCTAssertEqual(report.summary.totalCases, 5)
        XCTAssertEqual(report.kernel.field, "BabyBear")
        XCTAssertEqual(report.kernel.nttVariant, "babybear-poseidon2")
    }

    func testPoseidon2GoldilocksSmokeSuitePasses() throws {
        let request = ZkgpuBridge.poseidon2SmokeRequest(field: .goldilocks)
        let response = try ZkgpuBridge.run(request)
        XCTAssertTrue(response.ok,
            response.error ?? "poseidon2 goldilocks smoke failed")
        let report = try XCTUnwrap(response.hashReport)
        XCTAssertEqual(report.summary.failedCases, 0,
            "failed=\(report.summary.failedCases)/\(report.summary.totalCases)")
        XCTAssertEqual(report.kernel.field, "Goldilocks")
        XCTAssertEqual(report.kernel.nttVariant, "goldilocks-poseidon2-portable")
    }

    // MARK: - Poseidon2 benchmark

    /// Benchmark ladder — both fields × (1024 / 16384 / 65536)
    /// permutations. Logs `POSEIDON2_BENCH field=... n=... wall=...
    /// M perms/s` per row so BrowserStack device-logs can be scraped
    /// for throughput without re-running.
    func testPoseidon2BenchmarkLadderBothFields() throws {
        for field in [HashField.babyBear, HashField.goldilocks] {
            let request = ZkgpuBridge.poseidon2BenchmarkRequest(field: field)
            let response = try ZkgpuBridge.run(request)
            XCTAssertTrue(response.ok,
                "benchmark \(field.rawValue) failed: \(response.error ?? "<no err>")")
            let report = try XCTUnwrap(response.hashReport)
            XCTAssertEqual(report.summary.failedCases, 0,
                "\(field.rawValue) benchmark failed=\(report.summary.failedCases)")

            print("--- Poseidon2 benchmark \(field.rawValue) ---")
            print("Device: \(report.device.name) [\(report.device.backend)]")
            for c in report.cases {
                let wallUs = c.timings.wallTimeNs.map { Double($0) / 1_000.0 } ?? 0
                let permsPerSec = wallUs > 0
                    ? Double(c.numPermutations) * 1_000_000.0 / wallUs
                    : 0
                let status = c.passed ? "PASS" : "FAIL(\(c.mismatchCount))"
                let wallStr = String(format: "%.0fus", wallUs)
                let mppsStr = String(format: "%.3fM perms/s", permsPerSec / 1e6)
                print("POSEIDON2_BENCH field=\(field.rawValue) \(c.name) n=\(c.numPermutations) wall=\(wallStr) \(mppsStr) \(status)")
            }
        }
    }
}
