import Foundation
import MLX

/// Thin wrapper around `MLX.loadArrays(url:)` for the bundled nvdiffrast
/// reference fixtures. The actual safetensors parsing is built into MLX —
/// see [[feedback-mlx-builtins]] for the reminder.
enum Fixtures {

    /// Lazily load and cache the fixture dict once per test run.
    private static var _cache: [String: MLXArray]?

    static var bundle: [String: MLXArray] {
        if let c = _cache { return c }
        guard let url = Bundle.module.url(
            forResource: "diffrast_fixtures", withExtension: "safetensors"
        ) else {
            fatalError("Could not find diffrast_fixtures.safetensors in test bundle. " +
                       "Listed as a resource in Package.swift?")
        }
        do {
            let dict = try MLX.loadArrays(url: url)
            _cache = dict
            return dict
        } catch {
            fatalError("MLX.loadArrays failed at \(url.path): \(error)")
        }
    }

    /// Fetch a fixture tensor by name. Crashes if missing — these are bundled
    /// fixtures we ship; a missing key means the export script and the test
    /// have drifted.
    static func get(_ name: String) -> MLXArray {
        guard let t = bundle[name] else {
            fatalError("Fixture '\(name)' not in bundle; have: \(bundle.keys.sorted())")
        }
        return t
    }
}
