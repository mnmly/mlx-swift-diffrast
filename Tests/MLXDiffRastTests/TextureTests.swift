import XCTest
import MLX
import MLXRandom
@testable import MLXDiffRast

final class TextureTests: XCTestCase {

    // MARK: - Forward

    /// Sampling at exact texel centers should return that texel verbatim.
    func testForwardTexelCenters() {
        // 2×2 texture, 1 channel — values [10, 20; 30, 40].
        let tex = MLXArray([
            Float(10), 20,
            Float(30), 40,
        ], [1, 2, 2, 1])
        // Texel-center UV coords for a 2×2 texture: (0.25, 0.25), (0.75, 0.25), etc.
        let uv = MLXArray([
            Float(0.25), 0.25,   // top-left texel  → 10
            Float(0.75), 0.25,   // top-right texel → 20
            Float(0.25), 0.75,   // bot-left texel  → 30
            Float(0.75), 0.75,   // bot-right texel → 40
        ], [1, 2, 2, 2])

        let out = DiffRast.texture(tex, uv: uv)
        XCTAssertEqual(out.asArray(Float.self), [10, 20, 30, 40])
    }

    /// Center of a 2×2 texture lies between all four texels — bilinear gives the mean.
    func testForwardCenterIsMean() {
        let tex = MLXArray([
            Float(10), 20,
            Float(30), 40,
        ], [1, 2, 2, 1])
        let uv = MLXArray([Float(0.5), 0.5], [1, 1, 1, 2])
        let out = DiffRast.texture(tex, uv: uv)
        XCTAssertEqual(out.item(Float.self), 25, accuracy: 1e-5)  // mean of [10,20,30,40]
    }

    /// Zero boundary: sample well outside [0,1] should return all zeros.
    func testBoundaryZero() {
        let tex = MLXArray(Array(repeating: Float(1), count: 4), [1, 2, 2, 1])
        let uv = MLXArray([Float(5.0), 5.0], [1, 1, 1, 2])
        let out = DiffRast.texture(tex, uv: uv, boundaryMode: .zero)
        XCTAssertEqual(out.item(Float.self), 0, accuracy: 1e-6)
    }

    /// Clamp boundary: a UV way past 1 should snap to the corner texel.
    func testBoundaryClamp() {
        let tex = MLXArray([
            Float(10), 20,
            Float(30), 40,
        ], [1, 2, 2, 1])
        // u=v=10 → tx, ty are huge positive → both indices clamp to W_tex-1 = 1 → texel 40.
        let uv = MLXArray([Float(10), 10], [1, 1, 1, 2])
        let out = DiffRast.texture(tex, uv: uv, boundaryMode: .clamp)
        XCTAssertEqual(out.item(Float.self), 40, accuracy: 1e-6)
    }

    /// Wrap boundary: u advanced by exactly 1.0 should give the same sample.
    func testBoundaryWrapPeriodic() {
        let tex = MLXArray([
            Float(10), 20,
            Float(30), 40,
        ], [1, 2, 2, 1])
        let uv0 = MLXArray([Float(0.4), 0.6], [1, 1, 1, 2])
        let uv1 = MLXArray([Float(1.4), 0.6], [1, 1, 1, 2])
        let out0 = DiffRast.texture(tex, uv: uv0, boundaryMode: .wrap)
        let out1 = DiffRast.texture(tex, uv: uv1, boundaryMode: .wrap)
        XCTAssertEqual(out0.item(Float.self), out1.item(Float.self), accuracy: 1e-5)
    }

    // MARK: - Gradcheck

    func testGradcheckTex() throws {
        try runTextureGradcheck(perturb: .tex, boundary: .wrap)
    }

    func testGradcheckUV() throws {
        try runTextureGradcheck(perturb: .uv, boundary: .wrap)
    }

    func testGradcheckTexClamp() throws {
        try runTextureGradcheck(perturb: .tex, boundary: .clamp)
    }

    func testGradcheckUVZero() throws {
        try runTextureGradcheck(perturb: .uv, boundary: .zero)
    }

    // MARK: - Mip pyramid construction

    /// Building a mip pyramid on a 4×4 single-channel texture should yield
    /// levels [4×4, 2×2, 1×1], with each smaller level being the 2×2 mean.
    func testTextureConstructMipShapes() {
        let tex = MLXArray((0..<16).map { Float($0) }, [1, 4, 4, 1])
        let levels = DiffRast.textureConstructMip(tex)
        XCTAssertEqual(levels.count, 3)
        XCTAssertEqual(levels[0].shape, [1, 4, 4, 1])
        XCTAssertEqual(levels[1].shape, [1, 2, 2, 1])
        XCTAssertEqual(levels[2].shape, [1, 1, 1, 1])

        // Level 1 top-left: mean of texels (0,1,4,5) = 2.5.
        let level1 = levels[1].asArray(Float.self)
        XCTAssertEqual(level1[0], 2.5, accuracy: 1e-6)

        // Level 2: mean of all 16 = (0+1+...+15)/16 = 7.5.
        XCTAssertEqual(levels[2].item(Float.self), 7.5, accuracy: 1e-6)
    }

    func testTextureConstructMipMaxLevel() {
        let tex = MLXArray.zeros([1, 8, 8, 1])
        let levels = DiffRast.textureConstructMip(tex, maxLevel: 2)
        // maxLevel=2 ⇒ levels 0, 1, 2 (3 entries).
        XCTAssertEqual(levels.count, 3)
        XCTAssertEqual(levels[0].shape, [1, 8, 8, 1])
        XCTAssertEqual(levels[1].shape, [1, 4, 4, 1])
        XCTAssertEqual(levels[2].shape, [1, 2, 2, 1])
    }

    // MARK: - Trilinear forward

    /// At very small `uvDA` (LOD ≈ 0), trilinear should match bilinear at the
    /// source resolution to floating-point precision.
    func testTrilinearMatchesBilinearAtLOD0() {
        let tex = MLXArray((0..<32).map { Float($0) * 0.05 }, [1, 4, 4, 2])
        let uv = MLXArray([Float(0.42), 0.31], [1, 1, 1, 2])
        // Tiny derivatives → LOD ≈ 0.
        let uvDA = MLXArray([Float(1e-6), 1e-6, 1e-6, 1e-6], [1, 1, 1, 4])

        let bilinear = DiffRast.texture(tex, uv: uv, filterMode: .linear)
        let trilinear = DiffRast.texture(tex, uv: uv, uvDA: uvDA,
                                         filterMode: .linearMipmapLinear)
        let b = bilinear.asArray(Float.self)
        let t = trilinear.asArray(Float.self)
        for i in 0..<b.count {
            XCTAssertEqual(b[i], t[i], accuracy: 1e-4,
                           "channel \(i): bilinear \(b[i]) vs trilinear@LOD0 \(t[i])")
        }
    }

    /// At a large `uvDA` (LOD pinned at top of pyramid), trilinear converges
    /// to the global mean of the texture (the highest mip level for a
    /// uniformly built pyramid).
    func testTrilinearAtMaxLODYieldsGlobalMean() {
        // 4×4 texture, value 1 everywhere. Mean is 1; should hold at any LOD.
        let tex = MLXArray(Array(repeating: Float(1), count: 16), [1, 4, 4, 1])
        let uv = MLXArray([Float(0.5), 0.5], [1, 1, 1, 2])
        // Huge derivatives → LOD saturates at top of pyramid.
        let uvDA = MLXArray([Float(100), 100, 100, 100], [1, 1, 1, 4])
        let out = DiffRast.texture(tex, uv: uv, uvDA: uvDA,
                                   filterMode: .linearMipmapLinear)
        XCTAssertEqual(out.item(Float.self), 1, accuracy: 1e-6)
    }

    // MARK: - Trilinear gradcheck

    func testGradcheckTrilinearTex() throws {
        try runTrilinearGradcheck(perturb: .tex)
    }

    func testGradcheckTrilinearUV() throws {
        try runTrilinearGradcheck(perturb: .uv)
    }

    private func runTrilinearGradcheck(perturb: PerturbTarget) throws {
        let N = 1, Himg = 2, Wimg = 2, Htex = 4, Wtex = 4, C = 2

        MLXRandom.seed(0x71117EAB)
        let tex = MLXRandom.normal([N, Htex, Wtex, C])

        // UVs avoid pixel-grid alignments to keep floor() steady under eps.
        var uvVals = [Float]()
        for ph in 0..<Himg {
            for pw in 0..<Wimg {
                uvVals.append(0.32 + 0.27 * Float(pw))
                uvVals.append(0.41 + 0.23 * Float(ph))
            }
        }
        let uv = MLXArray(uvVals, [N, Himg, Wimg, 2])
        // Moderate uvDA → LOD ≈ 1 (interior, non-saturated).
        let uvDA = MLXArray(
            (0..<(N * Himg * Wimg * 4)).map { Float(0.4 + 0.05 * Float($0 % 3)) },
            [N, Himg, Wimg, 4]
        )

        var wOutVals = [Float]()
        for i in 0..<(N * Himg * Wimg * C) {
            wOutVals.append(Float(1 + i % 5) * 0.1)
        }
        let wOut = MLXArray(wOutVals, [N, Himg, Wimg, C])

        let loss: (MLXArray) -> MLXArray = { x in
            let out: MLXArray
            switch perturb {
            case .tex: out = DiffRast.texture(x, uv: uv, uvDA: uvDA,
                                              filterMode: .linearMipmapLinear)
            case .uv:  out = DiffRast.texture(tex, uv: x, uvDA: uvDA,
                                              filterMode: .linearMipmapLinear)
            }
            return (out * wOut).sum()
        }
        let input: MLXArray = (perturb == .tex) ? tex : uv

        let analytic = MLX.grad(loss)(input)
        analytic.eval()
        let analyticFlat = analytic.asArray(Float.self)

        let eps: Float = 1e-3
        let flat = input.asArray(Float.self)
        for i in 0..<flat.count {
            var plus = flat; plus[i] += eps
            var minus = flat; minus[i] -= eps
            let lp = loss(MLXArray(plus, input.shape)).item(Float.self)
            let lm = loss(MLXArray(minus, input.shape)).item(Float.self)
            let fd = (lp - lm) / (2 * eps)
            XCTAssertEqual(analyticFlat[i], fd, accuracy: 8e-3,
                           "[\(perturb) trilinear] elem \(i): \(analyticFlat[i]) vs \(fd)")
        }
    }

    private enum PerturbTarget { case tex, uv }

    /// Keeps UVs well inside (0, 1) so `floor(tx)` is stable across the eps
    /// perturbation — analogous to the silhouette-edge problem in rasterize.
    private func runTextureGradcheck(perturb: PerturbTarget, boundary: DiffRast.BoundaryMode) throws {
        let N = 1, Himg = 2, Wimg = 2, Htex = 3, Wtex = 3, C = 2

        MLXRandom.seed(0x7E27)
        let tex = MLXRandom.normal([N, Htex, Wtex, C])
        // UV in [0.2, 0.8] — keeps `floor` step away from texel boundaries.
        // Build a deterministic grid plus tiny offsets.
        var uvVals = [Float]()
        for ph in 0..<Himg {
            for pw in 0..<Wimg {
                let u = 0.25 + 0.3 * Float(pw)        // 0.25, 0.55
                let v = 0.30 + 0.25 * Float(ph)       // 0.30, 0.55
                uvVals.append(u); uvVals.append(v)
            }
        }
        let uv = MLXArray(uvVals, [N, Himg, Wimg, 2])

        // Deterministic, distinct cotangent weights.
        var wOutVals = [Float]()
        for i in 0..<(N * Himg * Wimg * C) {
            wOutVals.append(Float(1 + i % 5) * 0.1)
        }
        let wOut = MLXArray(wOutVals, [N, Himg, Wimg, C])

        let loss: (MLXArray) -> MLXArray = { x in
            let out: MLXArray
            switch perturb {
            case .tex: out = DiffRast.texture(x, uv: uv, boundaryMode: boundary)
            case .uv:  out = DiffRast.texture(tex, uv: x, boundaryMode: boundary)
            }
            return (out * wOut).sum()
        }
        let input: MLXArray = (perturb == .tex) ? tex : uv

        let analytic = MLX.grad(loss)(input)
        analytic.eval()
        let analyticFlat = analytic.asArray(Float.self)

        let eps: Float = 1e-3
        let flat = input.asArray(Float.self)
        var numeric = [Float](repeating: 0, count: flat.count)
        for i in 0..<flat.count {
            var plus = flat; plus[i] += eps
            var minus = flat; minus[i] -= eps
            let lp = loss(MLXArray(plus, input.shape)).item(Float.self)
            let lm = loss(MLXArray(minus, input.shape)).item(Float.self)
            numeric[i] = (lp - lm) / (2 * eps)
        }
        for i in 0..<flat.count {
            XCTAssertEqual(analyticFlat[i], numeric[i], accuracy: 5e-3,
                           "[\(perturb) \(boundary)] elem \(i): \(analyticFlat[i]) vs \(numeric[i])")
        }
    }
}
