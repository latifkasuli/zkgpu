import XCTest
@testable import ZkgpuHarness

final class ZkgpuHarnessTests: XCTestCase {

    // MARK: - Version handshake

    func testVersionHandshakeParses() throws {
        let version = try ZkgpuBridge.getVersion()
        XCTAssertFalse(version.crateName.isEmpty)
        XCTAssertFalse(version.version.isEmpty)
        XCTAssertGreaterThanOrEqual(version.ffiApiVersion, 1)
        print("FFI: \(version.crateName) \(version.version) (api \(version.ffiApiVersion))")
    }

    // MARK: - Smoke suite

    func testSmokeSuiteSucceeds() throws {
        let response = try runSuite(.smoke)
        XCTAssertTrue(response.ok, response.error ?? "smoke suite failed")
        let report = try XCTUnwrap(response.report)
        XCTAssertEqual(report.summary.failedCases, 0)
        printReport("Smoke", report)
    }

    func testSmokeSuiteIncludesDeviceMetadata() throws {
        let response = try runSuite(.smoke)
        let report = try XCTUnwrap(response.report)
        XCTAssertFalse(report.device.name.isEmpty)
        XCTAssertFalse(report.device.backend.isEmpty)
        XCTAssertFalse(report.device.gpuFamily.isEmpty)
        print("Device: \(report.device.name) [\(report.device.backend)] family=\(report.device.gpuFamily) tier=\(report.device.tier) memory=\(report.device.memoryModel)")
    }

    // MARK: - Validation suite

    func testValidationSuiteSucceeds() throws {
        let response = try runSuite(.validation)
        XCTAssertTrue(response.ok, response.error ?? "validation suite failed")
        let report = try XCTUnwrap(response.report)
        XCTAssertEqual(report.summary.failedCases, 0,
            "validation: \(report.summary.failedCases)/\(report.summary.totalCases) cases failed")
        printReport("Validation", report)
        try persistReport(response, suite: .validation)
    }

    // MARK: - Benchmark suite

    func testBenchmarkSuiteSucceeds() throws {
        let response = try runSuite(.benchmark)
        XCTAssertTrue(response.ok, response.error ?? "benchmark suite failed")
        let report = try XCTUnwrap(response.report)
        XCTAssertEqual(report.summary.failedCases, 0)
        printReport("Benchmark", report)
        try persistReport(response, suite: .benchmark)
    }

    // MARK: - Crossover benchmark

    func testCrossoverBenchmarkSucceeds() throws {
        let request = ZkgpuBridge.crossoverRequest(family: .auto)
        let requestJson = try ZkgpuBridge.encodeRequest(request)
        let json = try ZkgpuBridge.runRequestJson(requestJson)
        let response = try ZkgpuBridge.decodeResponse(json)

        XCTAssertTrue(response.ok, response.error ?? "crossover benchmark failed")
        let report = try XCTUnwrap(response.report)
        XCTAssertEqual(report.summary.failedCases, 0)

        print("--- Crossover Results ---")
        print("Device: \(report.device.name) [\(report.device.backend)]")
        for c in report.cases {
            let wallMs = c.timings.wallTimeNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            let gpuMs = c.timings.gpuTotalNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            let status = c.passed ? "PASS" : "FAIL"
            print("  \(c.name): \(status)  wall=\(wallMs)  gpu=\(gpuMs)")
        }
        print("Passed: \(report.summary.passedCases)/\(report.summary.totalCases)")

        let url = try ZkgpuBridge.persistResponseJson(json, suite: .benchmark)
        print("Report saved: \(url.path)")
    }

    // MARK: - Kernel family overrides

    func testSmokeSuiteStockhamFamily() throws {
        let response = try runSuiteWithFamily(.smoke, family: .stockham)
        XCTAssertTrue(response.ok, response.error ?? "smoke stockham failed")
    }

    func testSmokeSuiteFourStepFamily() throws {
        let response = try runSuiteWithFamily(.smoke, family: .fourStep)
        XCTAssertTrue(response.ok, response.error ?? "smoke four-step failed")
    }

    func testCrossoverStockhamFamily() throws {
        let request = ZkgpuBridge.crossoverRequest(family: .stockham)
        let requestJson = try ZkgpuBridge.encodeRequest(request)
        let json = try ZkgpuBridge.runRequestJson(requestJson)
        let response = try ZkgpuBridge.decodeResponse(json)

        XCTAssertTrue(response.ok, response.error ?? "crossover stockham failed")
        let report = try XCTUnwrap(response.report)
        XCTAssertEqual(report.summary.failedCases, 0)

        print("--- Crossover STOCKHAM ---")
        print("Device: \(report.device.name) [\(report.device.backend)]")
        for c in report.cases {
            let wallMs = c.timings.wallTimeNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            let gpuMs = c.timings.gpuTotalNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            print("  \(c.name): \(c.passed ? "PASS" : "FAIL")  wall=\(wallMs)  gpu=\(gpuMs)")
        }
        print("Passed: \(report.summary.passedCases)/\(report.summary.totalCases)")

        let url = try ZkgpuBridge.persistResponseJson(json, suite: .benchmark)
        print("Report saved: \(url.path)")
    }

    func testCrossoverFourStepFamily() throws {
        let request = ZkgpuBridge.crossoverRequest(family: .fourStep)
        let requestJson = try ZkgpuBridge.encodeRequest(request)
        let json = try ZkgpuBridge.runRequestJson(requestJson)
        let response = try ZkgpuBridge.decodeResponse(json)

        XCTAssertTrue(response.ok, response.error ?? "crossover four-step failed")
        let report = try XCTUnwrap(response.report)
        XCTAssertEqual(report.summary.failedCases, 0)

        print("--- Crossover FOUR-STEP ---")
        print("Device: \(report.device.name) [\(report.device.backend)]")
        for c in report.cases {
            let wallMs = c.timings.wallTimeNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            let gpuMs = c.timings.gpuTotalNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            print("  \(c.name): \(c.passed ? "PASS" : "FAIL")  wall=\(wallMs)  gpu=\(gpuMs)")
        }
        print("Passed: \(report.summary.passedCases)/\(report.summary.totalCases)")

        let url = try ZkgpuBridge.persistResponseJson(json, suite: .benchmark)
        print("Report saved: \(url.path)")
    }

    // MARK: - Helpers

    private func runSuite(_ suite: HarnessSuite) throws -> HarnessResponse {
        try ZkgpuBridge.run(HarnessRequest(suite: suite, spec: nil, familyOverride: nil))
    }

    private func runSuiteWithFamily(_ suite: HarnessSuite, family: HarnessFamilyOverride) throws -> HarnessResponse {
        try ZkgpuBridge.run(HarnessRequest(suite: suite, spec: nil, familyOverride: family))
    }

    private func printReport(_ label: String, _ report: SuiteReport) {
        print("--- \(label) Results ---")
        print("Device: \(report.device.name) [\(report.device.backend)] tier=\(report.device.tier)")
        print("Kernel: \(report.kernel.field) / \(report.kernel.nttVariant)")
        for c in report.cases {
            let wallMs = c.timings.wallTimeNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            let gpuMs = c.timings.gpuTotalNs.map { String(format: "%.2fms", Double($0) / 1_000_000) } ?? "n/a"
            let status = c.passed ? "PASS" : "FAIL"
            print("  \(c.name): \(status)  wall=\(wallMs)  gpu=\(gpuMs)")
        }
        print("Passed: \(report.summary.passedCases)/\(report.summary.totalCases)")
    }

    private func persistReport(_ response: HarnessResponse, suite: HarnessSuite) throws {
        let json = try ZkgpuBridge.encodeRequest(
            HarnessRequest(suite: suite, spec: nil, familyOverride: nil)
        )
        _ = json // request json not needed for persistence
        let responseJson = try ZkgpuBridge.runRequestJson(
            ZkgpuBridge.encodeRequest(HarnessRequest(suite: suite, spec: nil, familyOverride: nil))
        )
        let url = try ZkgpuBridge.persistResponseJson(responseJson, suite: suite)
        print("Report saved: \(url.path)")
    }
}
