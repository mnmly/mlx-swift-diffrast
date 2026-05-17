import XCTest
import MLX
import MLXRandom
@testable import MLXDiffRast

final class InterpolateTests: XCTestCase {

    // MARK: - Forward (basic)

    /// Constant barycentrics across all pixels; hand-computed expected values.
    func testForwardConstantBarys() {
        let N = 1, H = 2, W = 2, A = 3, V = 3

        var attrVals: [Float] = []
        for i in 0..<V {
            attrVals += [Float(i), Float(i + 1), Float(i + 2)]
        }
        let attr = MLXArray(attrVals, [N, V, A])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])

        // (u, v) = (0.25, 0.5) ⇒ bw = 0.25; triangle id+1 = 1 everywhere.
        let rast = MLXArray(
            Array(repeating: [Float(0.25), 0.5, 0.0, 1.0], count: N * H * W).flatMap { $0 },
            [N, H, W, 4])

        let (out, outDA) = DiffRast.interpolate(attr, rast: rast, tri: tri)
        XCTAssertEqual(out.shape, [N, H, W, A])
        XCTAssertEqual(outDA.shape, [N, H, W, 0])

        // 0.25*a0 + 0.25*a1 + 0.5*a2  for each attr.
        let expected: [Float] = [1.25, 2.25, 3.25]
        let flat = out.asArray(Float.self)
        for pix in 0..<(N * H * W) {
            for a in 0..<A {
                XCTAssertEqual(flat[pix * A + a], expected[a], accuracy: 1e-5)
            }
        }
    }

    func testEmptyPixelsAreZero() {
        let N = 1, H = 1, W = 2, A = 2, V = 3
        let attr = MLXArray(Array(repeating: Float(1), count: N * V * A), [N, V, A])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let rast = MLXArray([
            Float(0), 0, 0, 0,   // empty pixel
            0,        0, 0, 1,   // tri 0, u=v=0 ⇒ all weight on vertex 0
        ], [N, H, W, 4])

        let (out, _) = DiffRast.interpolate(attr, rast: rast, tri: tri)
        let flat = out.asArray(Float.self)
        XCTAssertEqual(flat, [0, 0, 1, 1])
    }

    // MARK: - Range mode

    /// `attr` shape `[V, A]` should broadcast across batch.
    func testRangeModeBroadcastsAcrossBatch() {
        let N = 2, H = 1, W = 1, A = 2, V = 3
        let attr2D = MLXArray([Float(10), 20,  30, 40,  50, 60], [V, A])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let rast = MLXArray([
            // batch 0: bw=1 → vertex 0
            Float(0), 0, 0, 1,
            // batch 1: u=1 → vertex 1
            1, 0, 0, 1,
        ], [N, H, W, 4])

        let (out, _) = DiffRast.interpolate(attr2D, rast: rast, tri: tri)
        XCTAssertEqual(out.shape, [N, H, W, A])
        XCTAssertEqual(out.asArray(Float.self), [10, 20, 30, 40])
    }

    // MARK: - Forward (DA)

    /// Verify out_da matches the analytic formula
    ///   out_da[2a]   = (a1-a0)*udx + (a2-a0)*vdx
    ///   out_da[2a+1] = (a1-a0)*udy + (a2-a0)*vdy
    func testForwardDAMatchesAnalytic() {
        let N = 1, H = 1, W = 1, A = 2, V = 3
        // a0 = (1,2), a1 = (4,5), a2 = (7,8)  ⇒ d01 = (3,3), d02 = (6,6)
        let attr = MLXArray([Float(1), 2,  4, 5,  7, 8], [N, V, A])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let rast = MLXArray([Float(0.1), 0.2, 0.0, 1.0], [N, H, W, 4])
        // (udx, udy, vdx, vdy) = (0.5, -0.25, 0.125, 1.0)
        let rastDB = MLXArray([Float(0.5), -0.25, 0.125, 1.0], [N, H, W, 4])

        let (_, outDA) = DiffRast.interpolate(
            attr, rast: rast, tri: tri, rastDB: rastDB, diffAttrs: .all)
        XCTAssertEqual(outDA.shape, [N, H, W, 2 * A])

        // d01 = (3,3), d02 = (6,6); same per channel.
        //   2a   = 3*0.5   + 6*0.125 = 2.25
        //   2a+1 = 3*-0.25 + 6*1.0   = 5.25
        let expected: [Float] = [2.25, 5.25, 2.25, 5.25]
        let got = outDA.asArray(Float.self)
        for i in 0..<expected.count {
            XCTAssertEqual(got[i], expected[i], accuracy: 1e-5)
        }
    }

    // MARK: - diff_attrs subset

    /// `diffAttrs: .indices(...)` should produce an `outDA` of shape
    /// `[N, H, W, 2·K]` where K is the index count, with values matching the
    /// corresponding channels from `.all`.
    func testDiffAttrsIndicesMatchesAllSubset() {
        let N = 1, H = 1, W = 1, A = 3, V = 3
        let attr = MLXArray([
            Float(1), 2, 3,
            Float(4), 5, 6,
            Float(7), 8, 9,
        ], [N, V, A])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let rast = MLXArray([Float(0.2), 0.3, 0.0, 1.0], [N, H, W, 4])
        let rastDB = MLXArray([Float(0.5), 0.1, -0.2, 0.3], [N, H, W, 4])

        let (_, daAll) = DiffRast.interpolate(
            attr, rast: rast, tri: tri, rastDB: rastDB, diffAttrs: .all)
        let (_, daIdx) = DiffRast.interpolate(
            attr, rast: rast, tri: tri, rastDB: rastDB,
            diffAttrs: .indices([0, 2]))
        XCTAssertEqual(daAll.shape, [N, H, W, 2 * A])
        XCTAssertEqual(daIdx.shape, [N, H, W, 4])  // 2*K, K=2

        let all = daAll.asArray(Float.self)
        let idx = daIdx.asArray(Float.self)
        XCTAssertEqual(idx[0], all[0], accuracy: 1e-6, "attr 0 du")
        XCTAssertEqual(idx[1], all[1], accuracy: 1e-6, "attr 0 dv")
        XCTAssertEqual(idx[2], all[4], accuracy: 1e-6, "attr 2 du (from .all index 4)")
        XCTAssertEqual(idx[3], all[5], accuracy: 1e-6, "attr 2 dv (from .all index 5)")
    }

    func testDiffAttrsIndicesGradcheckAttr() throws {
        try runDiffAttrsIndicesGradcheck(perturb: .attr)
    }

    func testDiffAttrsIndicesGradcheckRastDB() throws {
        try runDiffAttrsIndicesGradcheck(perturb: .rastDB)
    }

    private func runDiffAttrsIndicesGradcheck(perturb: PerturbTarget) throws {
        let N = 1, H = 2, W = 2, A = 4, V = 3

        MLXRandom.seed(0x1D7E50A2)

        var us: [Float] = [], vs: [Float] = []
        let uniform: () -> Float = { Float.random(in: 0.05...0.45, using: &Self.rng) }
        for _ in 0..<(H * W) { us.append(uniform()); vs.append(uniform()) }
        var rastVals: [Float] = []
        for i in 0..<(H * W) { rastVals += [us[i], vs[i], 0.0, 1.0] }
        let attr = MLXRandom.normal([N, V, A])
        let rast = MLXArray(rastVals, [N, H, W, 4])
        let rastDB = MLXRandom.normal([N, H, W, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])

        let indices: [Int32] = [1, 3]
        let K = indices.count
        let wOut  = MLXRandom.normal([N, H, W, A])
        let wOutDA = MLXRandom.normal([N, H, W, 2 * K])

        let loss: (MLXArray) -> MLXArray = { x in
            let (o, oda): (MLXArray, MLXArray)
            switch perturb {
            case .attr:
                (o, oda) = DiffRast.interpolate(
                    x, rast: rast, tri: tri, rastDB: rastDB,
                    diffAttrs: .indices(indices))
            case .rast: fatalError("not used here")
            case .rastDB:
                (o, oda) = DiffRast.interpolate(
                    attr, rast: rast, tri: tri, rastDB: x,
                    diffAttrs: .indices(indices))
            }
            return (o * wOut).sum() + (oda * wOutDA).sum()
        }
        let input: MLXArray = (perturb == .attr) ? attr : rastDB

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
            XCTAssertEqual(analyticFlat[i], fd, accuracy: 5e-3,
                           "[\(perturb) .indices] elem \(i): \(analyticFlat[i]) vs \(fd)")
        }
    }

    // MARK: - Gradcheck

    /// Finite-difference check of d_out/d_attr against MLX's autograd.
    func testGradcheckAttr() throws {
        try runGradcheck(perturb: .attr, withDA: false)
    }

    func testGradcheckRast() throws {
        try runGradcheck(perturb: .rast, withDA: false)
    }

    func testGradcheckAttrDA() throws {
        try runGradcheck(perturb: .attr, withDA: true)
    }

    func testGradcheckRastDA() throws {
        try runGradcheck(perturb: .rast, withDA: true)
    }

    func testGradcheckRastDB() throws {
        try runGradcheck(perturb: .rastDB, withDA: true)
    }

    // MARK: - Gradcheck helper

    private enum PerturbTarget { case attr, rast, rastDB }

    /// Build a small fixture where every pixel hits the same triangle with random
    /// barycentrics; compare MLX `grad` against central finite differences on a
    /// scalar loss `sum(out * w_out) + sum(out_da * w_da)`.
    private func runGradcheck(perturb: PerturbTarget, withDA: Bool) throws {
        let N = 1, H = 2, W = 2, A = 2, V = 3
        MLXRandom.seed(0xD1FFA570BEEF)

        // Random barycentrics in (0, 1) with u+v < 1.
        var us: [Float] = []
        var vs: [Float] = []
        let uniform: () -> Float = { Float.random(in: 0.05...0.45, using: &Self.rng) }
        for _ in 0..<(H * W) {
            let u = uniform(); let v = uniform()
            us.append(u); vs.append(v)
        }
        var rastVals: [Float] = []
        for i in 0..<(H * W) {
            rastVals += [us[i], vs[i], 0.0, 1.0]   // tri 0 everywhere
        }

        let attr = MLXRandom.normal([N, V, A])
        let rast = MLXArray(rastVals, [N, H, W, 4])
        let rastDB: MLXArray = withDA ? MLXRandom.normal([N, H, W, 4]) : MLXArray.zeros([N, H, W, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])

        // Random cotangent weights.
        let wOut  = MLXRandom.normal([N, H, W, A])
        let wOutDA = withDA ? MLXRandom.normal([N, H, W, 2 * A]) : MLXArray.zeros([N, H, W, 0])

        // Scalar loss in terms of one differentiable input at a time.
        let loss: (MLXArray) -> MLXArray = { x in
            let (o, oda): (MLXArray, MLXArray)
            switch perturb {
            case .attr:
                (o, oda) = DiffRast.interpolate(
                    x, rast: rast, tri: tri,
                    rastDB: withDA ? rastDB : nil, diffAttrs: withDA ? .all : nil)
            case .rast:
                (o, oda) = DiffRast.interpolate(
                    attr, rast: x, tri: tri,
                    rastDB: withDA ? rastDB : nil, diffAttrs: withDA ? .all : nil)
            case .rastDB:
                (o, oda) = DiffRast.interpolate(
                    attr, rast: rast, tri: tri,
                    rastDB: x, diffAttrs: .all)
            }
            let primary = (o * wOut).sum()
            if withDA {
                return primary + (oda * wOutDA).sum()
            }
            return primary
        }

        let input: MLXArray
        switch perturb {
        case .attr:   input = attr
        case .rast:   input = rast
        case .rastDB: input = rastDB
        }

        let analytic = MLX.grad(loss)(input)
        analytic.eval()
        let analyticFlat = analytic.asArray(Float.self)

        // Central finite differences.
        let eps: Float = 1e-3
        let flat = input.asArray(Float.self)
        var numeric = [Float](repeating: 0, count: flat.count)
        for i in 0..<flat.count {
            // Skip the tri-id channel of rast — non-differentiable and would change
            // which triangle the pixel hits (discontinuous loss).
            if perturb == .rast && (i % 4 == 3 || i % 4 == 2) { continue }

            var plus = flat; plus[i] += eps
            var minus = flat; minus[i] -= eps
            let lp = loss(MLXArray(plus, input.shape)).item(Float.self)
            let lm = loss(MLXArray(minus, input.shape)).item(Float.self)
            numeric[i] = (lp - lm) / (2 * eps)
        }

        // Compare element-wise on differentiable entries.
        for i in 0..<flat.count {
            if perturb == .rast && (i % 4 == 3 || i % 4 == 2) {
                // d_rast should be exactly zero on these channels per the kernel.
                XCTAssertEqual(analyticFlat[i], 0, accuracy: 1e-5, "rast non-diff channel \(i)")
                continue
            }
            XCTAssertEqual(analyticFlat[i], numeric[i], accuracy: 5e-3,
                           "[\(perturb) withDA=\(withDA)] elem \(i)")
        }
    }

    // Deterministic RNG for the gradcheck fixture (Swift's stdlib RNG is non-seedable
    // by default; the small wrapper below mirrors `SystemRandomNumberGenerator` shape).
    private static var rng = SeededRNG(seed: 0xD1FFA570BEEF)
}

private struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
