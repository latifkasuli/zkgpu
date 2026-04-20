import SwiftUI

struct ContentView: View {
    @State private var selectedSuite: HarnessSuite = .smoke
    @State private var selectedFamily: HarnessFamilyOverride = .auto
    @State private var isRunning = false
    @State private var status = "Idle"
    @State private var responseJson = ""
    @State private var reportPath = ""
    @State private var versionText = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    GroupBox("Suite") {
                        Picker("Suite", selection: $selectedSuite) {
                            ForEach(HarnessSuite.allCases) { suite in
                                Text(suite.rawValue).tag(suite)
                            }
                        }
                        .pickerStyle(.segmented)
                    }

                    GroupBox("Kernel Family") {
                        Picker("Kernel Family", selection: $selectedFamily) {
                            ForEach(HarnessFamilyOverride.allCases) { family in
                                Text(family.displayName).tag(family)
                            }
                        }
                        .pickerStyle(.segmented)
                    }

                    HStack(spacing: 12) {
                        Button(action: runSelectedSuite) {
                            Text(isRunning ? "Running…" : "Run Suite")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isRunning)

                        Button(action: runCrossoverBenchmark) {
                            Text("Crossover")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                        .disabled(isRunning)

                        Button(action: loadVersion) {
                            Text("Version")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(isRunning)
                    }

                    GroupBox("Status") {
                        Text(status)
                            .font(.body.monospaced())
                            .textSelection(.enabled)
                    }

                    if !versionText.isEmpty {
                        GroupBox("FFI Version") {
                            Text(versionText)
                                .font(.body.monospaced())
                                .textSelection(.enabled)
                        }
                    }

                    if !reportPath.isEmpty {
                        GroupBox("Saved Report") {
                            Text(reportPath)
                                .font(.footnote.monospaced())
                                .textSelection(.enabled)
                        }
                    }

                    if !responseJson.isEmpty {
                        GroupBox("Last Response JSON") {
                            Text(responseJson)
                                .font(.footnote.monospaced())
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("zkgpu iOS Harness")
            .onAppear { loadVersion() }
        }
    }

    private func runSelectedSuite() {
        isRunning = true
        status = "Running \(selectedSuite.rawValue)…"
        responseJson = ""
        reportPath = ""

        let suite = selectedSuite
        let family = selectedFamily

        Task.detached(priority: .userInitiated) {
            do {
                let request = HarnessRequest(
                    suite: suite,
                    spec: nil,
                    hashSpec: nil,
                    familyOverride: family == .auto ? nil : family
                )
                let requestJson = try ZkgpuBridge.encodeRequest(request)
                let json = try ZkgpuBridge.runRequestJson(requestJson)
                let response = try ZkgpuBridge.decodeResponse(json)

                let path: String
                if response.ok {
                    let url = try ZkgpuBridge.persistResponseJson(json, suite: suite)
                    path = url.path
                } else {
                    path = ""
                }

                await MainActor.run {
                    responseJson = json
                    reportPath = path
                    status = summarize(response)
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    status = "Run failed: \(error.localizedDescription)"
                    isRunning = false
                }
            }
        }
    }

    private func runCrossoverBenchmark() {
        isRunning = true
        status = "Running crossover benchmark (2^18–2^22)…"
        responseJson = ""
        reportPath = ""

        let family = selectedFamily

        Task.detached(priority: .userInitiated) {
            do {
                let request = ZkgpuBridge.crossoverRequest(family: family)
                let requestJson = try ZkgpuBridge.encodeRequest(request)
                let json = try ZkgpuBridge.runRequestJson(requestJson)
                let response = try ZkgpuBridge.decodeResponse(json)

                let path: String
                if response.ok {
                    let url = try ZkgpuBridge.persistResponseJson(json, suite: .benchmark)
                    path = url.path
                } else {
                    path = ""
                }

                await MainActor.run {
                    responseJson = json
                    reportPath = path
                    status = summarizeCrossover(response)
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    status = "Crossover failed: \(error.localizedDescription)"
                    isRunning = false
                }
            }
        }
    }

    private func loadVersion() {
        Task.detached(priority: .utility) {
            do {
                let version = try ZkgpuBridge.getVersion()
                let text = "\(version.crateName) \(version.version) (ffi api \(version.ffiApiVersion))"
                await MainActor.run {
                    versionText = text
                }
            } catch {
                await MainActor.run {
                    versionText = "Version query failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func summarize(_ response: HarnessResponse) -> String {
        if let error = response.error {
            return "Rust reported error: \(error)"
        }
        guard let report = response.report else {
            return "Missing report in successful response"
        }
        return "\(report.summary.passedCases)/\(report.summary.totalCases) cases passed on \(report.device.name) [\(report.device.backend)]"
    }

    private func summarizeCrossover(_ response: HarnessResponse) -> String {
        if let error = response.error {
            return "Crossover error: \(error)"
        }
        guard let report = response.report else {
            return "Missing report in successful response"
        }
        var lines: [String] = [
            "Crossover: \(report.summary.passedCases)/\(report.summary.totalCases) passed",
            "Device: \(report.device.name) [\(report.device.backend)]",
            ""
        ]
        for c in report.cases {
            let wallMs = c.timings.wallTimeNs.map { String(format: "%.1fms", Double($0) / 1_000_000) } ?? "n/a"
            let gpuMs = c.timings.gpuTotalNs.map { String(format: "%.1fms", Double($0) / 1_000_000) } ?? "n/a"
            let status = c.passed ? "PASS" : "FAIL"
            lines.append("\(c.name): \(status)  wall=\(wallMs)  gpu=\(gpuMs)")
        }
        return lines.joined(separator: "\n")
    }
}

#Preview {
    ContentView()
}
