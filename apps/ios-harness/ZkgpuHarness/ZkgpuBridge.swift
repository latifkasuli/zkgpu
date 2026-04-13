import Foundation

enum HarnessSuite: String, CaseIterable, Codable, Identifiable {
    case smoke = "Smoke"
    case validation = "Validation"
    case benchmark = "Benchmark"

    var id: String { rawValue }
}

enum HarnessFamilyOverride: String, CaseIterable, Codable, Identifiable {
    case auto = "Auto"
    case stockham = "Stockham"
    case fourStep = "FourStep"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .auto:
            return "Auto"
        case .stockham:
            return "Stockham"
        case .fourStep:
            return "Four-Step"
        }
    }
}

struct HarnessRequest: Codable {
    var suite: HarnessSuite?
    var spec: SuiteSpec?
    var familyOverride: HarnessFamilyOverride?
}

struct HarnessResponse: Codable {
    var ok: Bool
    var report: SuiteReport?
    var error: String?
}

struct VersionResponse: Codable {
    var crateName: String
    var version: String
    var ffiApiVersion: UInt32
}

struct SuiteReport: Codable {
    var schemaVersion: UInt32
    var suite: HarnessSuite
    var device: DeviceReport
    var kernel: KernelReport
    var cases: [CaseReport]
    var summary: SuiteSummary
}

struct DeviceReport: Codable {
    var name: String
    var backend: String
    var tier: String
    var gpuFamily: String
    var platformClass: String
    var memoryModel: String
    var maxBufferSizeBytes: UInt64
    var maxWorkgroupSizeX: UInt32
    var maxComputeInvocations: UInt32
    var featureFlags: [String]
}

struct KernelReport: Codable {
    var field: String
    var nttVariant: String
}

struct SuiteSummary: Codable {
    var totalCases: UInt32
    var passedCases: UInt32
    var failedCases: UInt32
}

struct CaseReport: Codable {
    var name: String
    var logN: UInt32
    var direction: String
    var input: InputPattern
    var kernelFamily: String?
    var passed: Bool
    var mismatchCount: UInt32
    var firstMismatchIndex: UInt32?
    var firstMismatchGpu: String?
    var firstMismatchCpu: String?
    var timings: TimingReport
    var error: String?
}

struct TimingReport: Codable {
    var wallTimeNs: UInt64?
    var gpuTotalNs: UInt64?
    var gpuStageNs: [StageTimingReport]
}

struct StageTimingReport: Codable {
    var label: String
    var durationNs: UInt64
}

struct SuiteSpec: Codable {
    var kind: HarnessSuite
    var cases: [CaseSpec]
    var failFast: Bool
    var familyOverride: HarnessFamilyOverride
}

struct CaseSpec: Codable {
    var name: String
    var logN: UInt32
    var direction: String
    var input: InputPattern
    var profileGpuTimestamps: Bool
    var iterations: UInt32
    var warmupIterations: UInt32
}

enum InputPattern: Codable {
    case sequential
    case allZeros
    case allOnes
    case largeValuesDescending
    case pseudoRandomDeterministic(seed: UInt64)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            switch value {
            case "Sequential":
                self = .sequential
            case "AllZeros":
                self = .allZeros
            case "AllOnes":
                self = .allOnes
            case "LargeValuesDescending":
                self = .largeValuesDescending
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Unknown input pattern \(value)"
                )
            }
            return
        }

        let keyed = try decoder.container(keyedBy: DynamicCodingKey.self)
        if keyed.allKeys.count == 1, let key = keyed.allKeys.first, key.stringValue == "PseudoRandomDeterministic" {
            let payload = try keyed.decode(PseudoRandomPayload.self, forKey: key)
            self = .pseudoRandomDeterministic(seed: payload.seed)
            return
        }

        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Unsupported input pattern encoding"
        )
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .sequential:
            var container = encoder.singleValueContainer()
            try container.encode("Sequential")
        case .allZeros:
            var container = encoder.singleValueContainer()
            try container.encode("AllZeros")
        case .allOnes:
            var container = encoder.singleValueContainer()
            try container.encode("AllOnes")
        case .largeValuesDescending:
            var container = encoder.singleValueContainer()
            try container.encode("LargeValuesDescending")
        case .pseudoRandomDeterministic(let seed):
            var container = encoder.container(keyedBy: DynamicCodingKey.self)
            try container.encode(PseudoRandomPayload(seed: seed), forKey: DynamicCodingKey("PseudoRandomDeterministic"))
        }
    }
}

private struct PseudoRandomPayload: Codable {
    var seed: UInt64
}

private struct DynamicCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }

    init(_ stringValue: String) {
        self.stringValue = stringValue
    }

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) {
        return nil
    }
}

enum ZkgpuBridgeError: LocalizedError {
    case ffiReturnedNull
    case requestEncodingFailed(String)
    case invalidUtf8Response
    case responseDecodingFailed(String)
    case documentsDirectoryUnavailable

    var errorDescription: String? {
        switch self {
        case .ffiReturnedNull:
            return "zkgpu-ffi returned a null pointer"
        case .requestEncodingFailed(let message):
            return "failed to encode request JSON: \(message)"
        case .invalidUtf8Response:
            return "zkgpu-ffi returned invalid UTF-8"
        case .responseDecodingFailed(let message):
            return "failed to decode response JSON: \(message)"
        case .documentsDirectoryUnavailable:
            return "documents directory unavailable"
        }
    }
}

enum ZkgpuBridge {
    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    static func run(_ request: HarnessRequest) throws -> HarnessResponse {
        let json = try encodeRequest(request)
        let responseJson = try runRequestJson(json)
        return try decodeResponse(responseJson)
    }

    static func encodeRequest(_ request: HarnessRequest) throws -> String {
        let data: Data
        do {
            data = try encoder.encode(request)
        } catch {
            throw ZkgpuBridgeError.requestEncodingFailed(error.localizedDescription)
        }

        guard let json = String(data: data, encoding: .utf8) else {
            throw ZkgpuBridgeError.requestEncodingFailed("request JSON was not UTF-8")
        }
        return json
    }

    static func runRequestJson(_ request: String) throws -> String {
        try request.withCString { cString in
            guard let raw = zkgpu_run_request_json(cString) else {
                throw ZkgpuBridgeError.ffiReturnedNull
            }
            defer { zkgpu_free_string(raw) }
            guard let response = String(validatingUTF8: raw) else {
                throw ZkgpuBridgeError.invalidUtf8Response
            }
            return response
        }
    }

    static func getVersionJson() throws -> String {
        guard let raw = zkgpu_get_version_json() else {
            throw ZkgpuBridgeError.ffiReturnedNull
        }
        defer { zkgpu_free_string(raw) }
        guard let response = String(validatingUTF8: raw) else {
            throw ZkgpuBridgeError.invalidUtf8Response
        }
        return response
    }

    static func getVersion() throws -> VersionResponse {
        let json = try getVersionJson()
        do {
            return try decoder.decode(VersionResponse.self, from: Data(json.utf8))
        } catch {
            throw ZkgpuBridgeError.responseDecodingFailed(error.localizedDescription)
        }
    }

    static func decodeResponse(_ json: String) throws -> HarnessResponse {
        do {
            return try decoder.decode(HarnessResponse.self, from: Data(json.utf8))
        } catch {
            throw ZkgpuBridgeError.responseDecodingFailed(error.localizedDescription)
        }
    }

    @discardableResult
    static func persistResponseJson(_ json: String, suite: HarnessSuite) throws -> URL {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw ZkgpuBridgeError.documentsDirectoryUnavailable
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withDashSeparatorInDate, .withColonSeparatorInTime]
        let timestamp = formatter.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = docs.appendingPathComponent("zkgpu-\(suite.rawValue.lowercased())-\(timestamp).json")
        try json.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    static func crossoverRequest(family: HarnessFamilyOverride) -> HarnessRequest {
        let logNs: [UInt32] = [18, 19, 20, 21, 22]
        let directions = ["Forward", "Inverse"]

        var cases: [CaseSpec] = []
        for logN in logNs {
            for dir in directions {
                cases.append(CaseSpec(
                    name: "\(dir.lowercased())_log\(logN)",
                    logN: logN,
                    direction: dir,
                    input: .sequential,
                    profileGpuTimestamps: true,
                    iterations: 5,
                    warmupIterations: 2
                ))
            }
        }

        let spec = SuiteSpec(
            kind: .benchmark,
            cases: cases,
            failFast: false,
            familyOverride: family
        )

        return HarnessRequest(suite: nil, spec: spec, familyOverride: nil)
    }
}
