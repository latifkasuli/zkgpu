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
    /// Phase F.3.f — iOS twin of the F.3.e.1 FFI dispatch. When
    /// `hashSpec` is set, the Rust router runs `run_hash_suite` and
    /// returns `hashReport` instead of `report`. NTT fields stay nil.
    var hashSpec: HashSpec?
    var familyOverride: HarnessFamilyOverride?
}

struct HarnessResponse: Codable {
    var ok: Bool
    var report: SuiteReport?
    var hashReport: HashSuiteReport?
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

// MARK: - Hash surface (Phase F.3.f)

/// Swift twin of `zkgpu_report::HashAlgorithm`. Only Poseidon2 today.
enum HashAlgorithm: String, Codable {
    case poseidon2 = "Poseidon2"
}

/// Swift twin of `zkgpu_report::Field`. Matches the Android harness
/// `HashField` enum but shipped separately because Swift and Kotlin
/// don't share source.
enum HashField: String, Codable {
    case babyBear = "BabyBear"
    case goldilocks = "Goldilocks"
}

/// Swift twin of `zkgpu_report::HashInputPattern`. Custom `Codable`
/// because the Rust variants mix bare strings (unit variants) and
/// keyed objects (`SplitMix64 { seed }`) — same shape trick as
/// [`InputPattern`] above.
enum HashInputPattern: Codable {
    case allZeros
    case allOnes
    case sequential
    case splitMix64(seed: UInt64)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            switch value {
            case "AllZeros":
                self = .allZeros
            case "AllOnes":
                self = .allOnes
            case "Sequential":
                self = .sequential
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Unknown hash input pattern \(value)"
                )
            }
            return
        }

        let keyed = try decoder.container(keyedBy: DynamicCodingKey.self)
        if keyed.allKeys.count == 1,
           let key = keyed.allKeys.first,
           key.stringValue == "SplitMix64" {
            let payload = try keyed.decode(SplitMix64Payload.self, forKey: key)
            self = .splitMix64(seed: payload.seed)
            return
        }

        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Unsupported hash input pattern encoding"
        )
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .allZeros:
            var container = encoder.singleValueContainer()
            try container.encode("AllZeros")
        case .allOnes:
            var container = encoder.singleValueContainer()
            try container.encode("AllOnes")
        case .sequential:
            var container = encoder.singleValueContainer()
            try container.encode("Sequential")
        case .splitMix64(let seed):
            var container = encoder.container(keyedBy: DynamicCodingKey.self)
            try container.encode(
                SplitMix64Payload(seed: seed),
                forKey: DynamicCodingKey("SplitMix64")
            )
        }
    }
}

private struct SplitMix64Payload: Codable {
    var seed: UInt64
}

/// Swift twin of `zkgpu_report::HashCaseSpec`. Field names are
/// camelCase; the encoder's `.convertToSnakeCase` strategy remaps
/// them to `num_permutations` / `profile_gpu_timestamps` /
/// `warmup_iterations` on the wire.
struct HashCaseSpec: Codable {
    var name: String
    var numPermutations: UInt32
    var input: HashInputPattern
    var profileGpuTimestamps: Bool
    var iterations: UInt32
    var warmupIterations: UInt32
}

/// Swift twin of `zkgpu_report::HashSpec`.
struct HashSpec: Codable {
    var kind: HarnessSuite
    var cases: [HashCaseSpec]
    var failFast: Bool
    var algorithm: HashAlgorithm
    var field: HashField
}

/// Swift twin of `zkgpu_report::HashCaseReport`.
/// `firstMismatchIndex` is a `(perm, slot)` tuple on the Rust side,
/// serialised as a two-element JSON array. Decode it as a pair.
struct HashCaseReport: Codable {
    var name: String
    var numPermutations: UInt32
    var input: HashInputPattern
    var kernelFamily: String?
    var passed: Bool
    var mismatchCount: UInt32
    var firstMismatchIndex: [UInt32]?
    var firstMismatchGpu: String?
    var firstMismatchCpu: String?
    var timings: TimingReport
    var error: String?
}

/// Swift twin of `zkgpu_report::HashSuiteReport`. `suite` is
/// `SuiteKind` on the Rust side; reuses `HarnessSuite` here.
struct HashSuiteReport: Codable {
    var schemaVersion: UInt32
    var suite: HarnessSuite
    var device: DeviceReport
    var kernel: KernelReport
    var cases: [HashCaseReport]
    var summary: SuiteSummary
}

/// Plain-string `CodingKey` used by the custom key-encoding
/// strategy above. Lets us build a snake-cased key from a converted
/// string at runtime without needing a second enum.
private struct SnakeCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
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
    /// Phase F.3.f key-encoding strategy.
    ///
    /// Swift's built-in `.convertToSnakeCase` is applied to **every**
    /// coding key during encoding — including the dynamic keys that
    /// `HashInputPattern.splitMix64` uses for the `{"SplitMix64":
    /// {...}}` variant envelope. That mangles the Rust-side variant
    /// tag into `split_mix64`, which serde rejects with
    /// `unknown variant`.
    ///
    /// Custom rule: snake-case only keys that start with a lowercase
    /// letter (i.e. Swift struct / enum *field* names). Keys that
    /// start with an uppercase letter are enum *variant tags*
    /// (`SplitMix64`, `PseudoRandomDeterministic`, `AllZeros`
    /// serialised as-bare-strings doesn't hit this path) and must
    /// pass through verbatim.
    private static func keyToSnakeCase(_ keys: [CodingKey]) -> CodingKey {
        // `keys` is the full path; the last element is the one being
        // encoded. Preserves container depth semantics.
        let last = keys.last!
        let name = last.stringValue
        guard let first = name.first else { return last }
        if first.isUppercase {
            return last
        }
        // Standard Swift camelCase → snake_case conversion.
        var out = ""
        var previousIsLower = false
        for ch in name {
            if ch.isUppercase, previousIsLower {
                out.append("_")
            }
            out.append(ch.lowercased())
            previousIsLower = ch.isLowercase
        }
        return SnakeCodingKey(stringValue: out)
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .custom(keyToSnakeCase)
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

    // MARK: - Hash builders (Phase F.3.f)

    /// Default Poseidon2 smoke spec — mirrors the shipped
    /// `zkgpu_report::poseidon2_smoke_suite()` for the requested
    /// field. Matches the Android harness's
    /// `HarnessJson.poseidon2SmokeRequestJson` shape.
    static func poseidon2SmokeRequest(field: HashField) -> HarnessRequest {
        let cases: [HashCaseSpec] = [
            HashCaseSpec(
                name: "poseidon2_smoke_single",
                numPermutations: 1,
                input: .sequential,
                profileGpuTimestamps: false,
                iterations: 1,
                warmupIterations: 0
            ),
            HashCaseSpec(
                name: "poseidon2_smoke_batch17",
                numPermutations: 17,
                input: .sequential,
                profileGpuTimestamps: false,
                iterations: 1,
                warmupIterations: 0
            ),
            HashCaseSpec(
                name: "poseidon2_smoke_zeros",
                numPermutations: 8,
                input: .allZeros,
                profileGpuTimestamps: false,
                iterations: 1,
                warmupIterations: 0
            ),
            HashCaseSpec(
                name: "poseidon2_smoke_ones",
                numPermutations: 8,
                input: .allOnes,
                profileGpuTimestamps: false,
                iterations: 1,
                warmupIterations: 0
            ),
            HashCaseSpec(
                name: "poseidon2_smoke_rng",
                numPermutations: 32,
                // Swift `UInt64` handles the full [0, 2^64) range at
                // the type level; serde's u64 deserializer is happy
                // as long as the JSON number stays non-negative.
                input: .splitMix64(seed: 0xCAFE_BABE),
                profileGpuTimestamps: false,
                iterations: 1,
                warmupIterations: 0
            ),
        ]
        let spec = HashSpec(
            kind: .smoke,
            cases: cases,
            failFast: true,
            algorithm: .poseidon2,
            field: field
        )
        return HarnessRequest(
            suite: nil,
            spec: nil,
            hashSpec: spec,
            familyOverride: nil
        )
    }

    /// Poseidon2 benchmark ladder. Mirrors the Android harness's
    /// `poseidon2BenchmarkRequestJson` (1024 / 16384 / 65536 default
    /// ladder, SplitMix64 seed 1, 1 warmup + 5 measured).
    static func poseidon2BenchmarkRequest(
        field: HashField,
        permutationCounts: [UInt32] = [1_024, 16_384, 65_536]
    ) -> HarnessRequest {
        let cases = permutationCounts.map { num in
            HashCaseSpec(
                name: "ios_poseidon2_n\(num)",
                numPermutations: num,
                input: .splitMix64(seed: 1),
                // profileGpuTimestamps stays false until
                // execute_profiled lands on the Poseidon2 plans.
                // Setting true would hit the F.3.d post-review
                // rejection in measure_*_poseidon2_plan.
                profileGpuTimestamps: false,
                iterations: 5,
                warmupIterations: 1
            )
        }
        let spec = HashSpec(
            kind: .benchmark,
            cases: cases,
            failFast: false,
            algorithm: .poseidon2,
            field: field
        )
        return HarnessRequest(
            suite: nil,
            spec: nil,
            hashSpec: spec,
            familyOverride: nil
        )
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

        return HarnessRequest(
            suite: nil,
            spec: spec,
            hashSpec: nil,
            familyOverride: nil
        )
    }
}
