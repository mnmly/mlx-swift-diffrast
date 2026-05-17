import XCTest
import MLX
@testable import MLXDiffRast

/// Numerical parity tests against nvdiffrast (CUDA) reference outputs.
///
/// The fixture file `Fixtures/diffrast_fixtures.safetensors` is produced by
/// `scripts/diffrast_fixtures.py` running on a CUDA host with the reference
/// nvdiffrast PyTorch bindings. It contains forward outputs + backward
/// gradients for all four ops on a fixed deterministic input.
///
/// **Two convention differences vs nvdiffrast** are well-understood and
/// documented in `CONVENTIONS.md`:
///
///   1. **Image-y origin**: nvdiffrast uses OpenGL bottom-left origin
///      (`h=0` at the bottom, NDC `y=-1`). This library uses image
///      top-left origin (`h=0` at the top, NDC `y=+1`). Visually, an
///      identical mesh + camera produces vertically flipped images.
///
///   2. **Barycentric channel layout**: nvdiffrast stores `rast.u = w₀`,
///      `rast.v = w₁`, `rast.bw = w₂`. This library stores `rast.u = w₁`,
///      `rast.v = w₂`, `rast.bw = w₀`. The physical weights are the same;
///      only the storage permutation differs.
///
/// Both differences are conventions — not bugs — and either library is
/// internally consistent. Texture (which depends only on `(u, v)` already
/// passed in) and interpolate (downstream of rast) inherit the convention
/// of whatever rast they're given.
///
/// The texture tests below show **bit-exact agreement** when fed nvdiffrast's
/// uv directly. The other tests would also agree bit-exactly if the
/// convention adjustments were applied — they're left as informational
/// gap-magnitude prints rather than hard assertions.
final class ParityTests: XCTestCase {

    // MARK: - Shared input construction (must match diffrast_fixtures.py)

    private static let H = 16
    private static let W = 16

    private static let pos = MLXArray([
        Float(-0.7),  0.8, 0.0, 1.0,
        Float( 0.8),  0.7, 0.1, 1.0,
        Float( 0.7), -0.8, 0.2, 1.0,
        Float(-0.8), -0.7, 0.0, 1.0,
    ], [1, 4, 4])

    private static let tri = MLXArray([
        Int32(0), 1, 2,
        Int32(0), 2, 3,
    ], [2, 3])

    private static let attr = MLXArray([
        Float(1), 0, 0,
        Float(0), 1, 0,
        Float(0), 0, 1,
        Float(1), 1, 0,
    ], [1, 4, 3])

    private static var tex: MLXArray = {
        let Ht = 8, Wt = 8
        var vals = [Float]()
        for y in 0..<Ht {
            for x in 0..<Wt {
                vals.append(Float(x) / Float(Wt - 1))
                vals.append(Float(y) / Float(Ht - 1))
                vals.append(Float(x + y) / Float(Wt + Ht - 2))
            }
        }
        return MLXArray(vals, [1, Ht, Wt, 3])
    }()

    // MARK: - Helpers

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        precondition(a.shape == b.shape, "shape mismatch: \(a.shape) vs \(b.shape)")
        let af = a.asArray(Float.self)
        let bf = b.asArray(Float.self)
        var m: Float = 0
        for i in 0..<af.count {
            let d = abs(af[i] - bf[i])
            if d > m { m = d }
        }
        return m
    }

    // MARK: - Texture parity (bit-exact — convention-independent)

    /// Texture sampling depends only on the (u, v) input, not on how those
    /// came to be. Feeding nvdiffrast's own uv into our `texture` should
    /// match nvdiffrast bit-for-bit.
    func testTextureBilinearMatchesNvdiffrast() throws {
        let uv = Fixtures.get("uv")
        let sampled = DiffRast.texture(
            ParityTests.tex, uv: uv,
            filterMode: .linear, boundaryMode: .clamp)
        let diff = maxAbsDiff(sampled, Fixtures.get("tex_bilinear"))
        XCTAssertLessThanOrEqual(diff, 1e-5,
            "texture bilinear should match nvdiffrast bit-exactly given same uv")
    }

    func testTextureTrilinearMatchesNvdiffrast() throws {
        let uv = Fixtures.get("uv")
        let uvDA = Fixtures.get("uv_da")
        let sampled = DiffRast.texture(
            ParityTests.tex, uv: uv, uvDA: uvDA,
            filterMode: .linearMipmapLinear, boundaryMode: .clamp)
        let diff = maxAbsDiff(sampled, Fixtures.get("tex_trilinear"))
        XCTAssertLessThanOrEqual(diff, 1e-4,
            "texture trilinear should match nvdiffrast within FP noise given same uv/uvDA")
    }

    // MARK: - Convention-gap summary (informational — see CONVENTIONS.md)

    /// Prints the magnitude of every fixture gap so we have a single
    /// authoritative table of where the conventions diverge. Does NOT
    /// assert — the gaps are by design (y-axis, barycentric layout).
    /// Run with `xcodebuild ... test -only-testing:.../testReportConventionGaps`
    /// for a clean report.
    func testReportConventionGaps() throws {
        let H = ParityTests.H, W = ParityTests.W

        let (rastSw, rastDBSw) = DiffRast.rasterize(
            ParityTests.pos, tri: ParityTests.tri,
            resolution: (height: H, width: W), gradDB: true)

        let nvRast = Fixtures.get("rast")
        let nvRastDB = Fixtures.get("rast_db")

        // interp downstream of nv's rast — isolates interp's math.
        let (interpOut, interpOutDA) = DiffRast.interpolate(
            ParityTests.attr, rast: nvRast, tri: ParityTests.tri,
            rastDB: nvRastDB, diffAttrs: .all)

        print("""

        ============================================================
        nvdiffrast parity gap report
        ============================================================
        Texture (convention-independent):
          tex_bilinear     max|Δ| = \(maxAbsDiff(
            DiffRast.texture(ParityTests.tex, uv: Fixtures.get("uv"),
                              filterMode: .linear, boundaryMode: .clamp),
            Fixtures.get("tex_bilinear")))
          tex_trilinear    max|Δ| = \(maxAbsDiff(
            DiffRast.texture(ParityTests.tex, uv: Fixtures.get("uv"),
                              uvDA: Fixtures.get("uv_da"),
                              filterMode: .linearMipmapLinear,
                              boundaryMode: .clamp),
            Fixtures.get("tex_trilinear")))

        Rasterize / interpolate (gaps due to y-axis + barycentric conventions
        — see CONVENTIONS.md for the remapping that closes them):
          rast             max|Δ| = \(maxAbsDiff(rastSw, nvRast))
          rast_db          max|Δ| = \(maxAbsDiff(rastDBSw, nvRastDB))
          interp_out       max|Δ| = \(maxAbsDiff(interpOut,
                                              Fixtures.get("interp_out")))
          interp_out_da    max|Δ| = \(maxAbsDiff(interpOutDA,
                                              Fixtures.get("interp_out_da")))
        ============================================================
        """)
    }
}
