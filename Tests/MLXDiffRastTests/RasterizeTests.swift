import XCTest
import MLX
import MLXRandom
@testable import MLXDiffRast

final class RasterizeTests: XCTestCase {

    /// Single triangle covering the entire image. With vertices at the screen
    /// corners (top-left, bottom-left, top-right in pixel space), the barycentric
    /// at each pixel can be hand-computed from the pixel's NDC coords.
    func testSingleTriangleCoversImage() {
        // V0 = top-left (-1, +1), V1 = bottom-left (-1, -1), V2 = top-right (+1, +1)
        // All at z=0, w=1 → flat plane at NDC z=0.
        //
        // Edge functions in this configuration:
        //   area = (sx1-sx0)*(sy2-sy0) - (sx2-sx0)*(sy1-sy0)
        //        = (0)*(0) - (2)*(-2) = 4
        //   e0(p) = (sx1-px)*(sy2-py) - (sx2-px)*(sy1-py)
        //        = (-1-px)*(1-py)    - (1-px)*(-1-py)
        //   bw = e0 / 4
        //
        // For the center pixel (W/2, H/2) with W=H=2, ndc = (0, 0):
        //   e0 = (-1)*(1) - (1)*(-1) = -1 + 1 = 0 → bw = 0   (on edge v1v2)
        // For pixel (0, 0) (top-left), ndc = (-0.5, +0.5):
        //   e0 = (-0.5)*(0.5) - (1.5)*(-1.5) = -0.25 + 2.25 = 2.0 → bw = 0.5
        //   e1 = (1.5)*(0.5)  - (-0.5)*(0.5) = 0.75 + 0.25 = 1.0  → u = 0.25
        //   e2 = (-0.5)*(-1.5) - (-0.5)*(0.5) = 0.75 + 0.25 = 1.0 → v = 0.25
        let pos = MLXArray([
            Float(-1),  1, 0, 1,   // V0 top-left
            Float(-1), -1, 0, 1,   // V1 bottom-left
            Float( 1),  1, 0, 1,   // V2 top-right
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])

        let (rast, _) = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2), gradDB: false)
        XCTAssertEqual(rast.shape, [1, 2, 2, 4])
        let flat = rast.asArray(Float.self)

        // Top-left pixel: u = 0.25, v = 0.25, z = 0, tri_id+1 = 1
        XCTAssertEqual(flat[0], 0.25, accuracy: 1e-5)
        XCTAssertEqual(flat[1], 0.25, accuracy: 1e-5)
        XCTAssertEqual(flat[2], 0.0,  accuracy: 1e-5)
        XCTAssertEqual(flat[3], 1.0,  accuracy: 1e-5)

        // This corner-triangle only covers the upper-left half of the image.
        // Bottom-right pixel (ndc 0.5, -0.5) lies outside the line y=x and is empty.
        XCTAssertEqual(flat[(0 * 2 + 0) * 4 + 3], 1.0, accuracy: 1e-5, "(0,0) inside")
        XCTAssertEqual(flat[(0 * 2 + 1) * 4 + 3], 1.0, accuracy: 1e-5, "(0,1) on edge → inside")
        XCTAssertEqual(flat[(1 * 2 + 0) * 4 + 3], 1.0, accuracy: 1e-5, "(1,0) inside")
        XCTAssertEqual(flat[(1 * 2 + 1) * 4 + 3], 0.0, accuracy: 1e-5, "(1,1) outside → empty")
    }

    /// Two coplanar triangles at different depths; closer one wins.
    func testDepthTestPicksFrontTriangle() {
        // Triangle 0: at z=0.5 (farther), covers full screen via the same corner setup.
        // Triangle 1: at z=-0.5 (closer), covers full screen.
        let pos = MLXArray([
            // tri 0 vertices: z = 0.5
            Float(-1),  1, 0.5, 1,
            Float(-1), -1, 0.5, 1,
            Float( 1),  1, 0.5, 1,
            // tri 1 vertices: z = -0.5
            Float(-1),  1, -0.5, 1,
            Float(-1), -1, -0.5, 1,
            Float( 1),  1, -0.5, 1,
        ], [1, 6, 4])
        let tri = MLXArray([
            Int32(0), 1, 2,
            Int32(3), 4, 5,
        ], [2, 3])

        let (rast, _) = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2), gradDB: false)
        let flat = rast.asArray(Float.self)

        // Every pixel that's inside both should report tri 1 (tri_id+1 = 2) and z=-0.5.
        for h in 0..<2 {
            for w in 0..<2 {
                let base = (h * 2 + w) * 4
                if flat[base + 3] == 0 { continue }
                XCTAssertEqual(flat[base + 2], -0.5, accuracy: 1e-5,
                               "pixel (h=\(h), w=\(w)) depth")
                XCTAssertEqual(flat[base + 3], 2.0, accuracy: 1e-5,
                               "pixel (h=\(h), w=\(w)) should pick front triangle")
            }
        }
    }

    /// Pixels outside the triangle should be exactly zero across all channels.
    func testPixelsOutsideTriangleAreZero() {
        // A small triangle near the top-left corner of the image.
        // V0 = (-1, 1), V1 = (-0.5, 1), V2 = (-1, 0.5). Image is 4×4.
        let pos = MLXArray([
            Float(-1.0),  1.0,  0, 1,
            Float(-0.5),  1.0,  0, 1,
            Float(-1.0),  0.5,  0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, _) = DiffRast.rasterize(pos, tri: tri, resolution: (height: 4, width: 4), gradDB: false)
        let flat = rast.asArray(Float.self)

        // Bottom-right pixel (h=3, w=3) ndc ≈ (0.75, -0.75) is far outside.
        let base = (3 * 4 + 3) * 4
        XCTAssertEqual(flat[base + 0], 0)
        XCTAssertEqual(flat[base + 1], 0)
        XCTAssertEqual(flat[base + 2], 0)
        XCTAssertEqual(flat[base + 3], 0)
    }

    /// Pixel-derivative `rast_db` matches the analytic formula on a known triangle.
    ///
    /// Triangle: V0=(-1, 1), V1=(-1, -1), V2=(1, 1) at z=0, w=1. Image 2×2.
    ///   sx0=-1, sy0= 1;  sx1=-1, sy1=-1;  sx2= 1, sy2= 1
    ///   area = (sx1-sx0)*(sy2-sy0) - (sx2-sx0)*(sy1-sy0)
    ///        = 0*0 - 2*(-2) = 4
    ///   du/dx = (sy2-sy0)/area * (2/W)  = 0/4    * 1  = 0
    ///   du/dy = (sx0-sx2)/area * (-2/H) = -2/4   * -1 = 0.5
    ///   dv/dx = (sy0-sy1)/area * (2/W)  = 2/4    * 1  = 0.5
    ///   dv/dy = (sx1-sx0)/area * (-2/H) = 0/4    * -1 = 0
    func testRastDBAnalytic() {
        let pos = MLXArray([
            Float(-1),  1, 0, 1,
            Float(-1), -1, 0, 1,
            Float( 1),  1, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])

        let (_, rastDB) = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2))
        XCTAssertEqual(rastDB.shape, [1, 2, 2, 4])
        let flat = rastDB.asArray(Float.self)

        // Check the three covered pixels (top-left, top-right, bottom-left).
        let expected: [Float] = [0.0, 0.5, 0.5, 0.0]
        for pixIdx in [0, 1, 2] {
            for ch in 0..<4 {
                XCTAssertEqual(flat[pixIdx * 4 + ch], expected[ch], accuracy: 1e-5,
                               "pix \(pixIdx) channel \(ch)")
            }
        }
        // Empty pixel (bottom-right) → all zeros.
        for ch in 0..<4 {
            XCTAssertEqual(flat[3 * 4 + ch], 0, accuracy: 1e-5)
        }
    }

    /// End-to-end DA path: rasterize(gradDB=true) feeds rast_db into interpolate
    /// with diffAttrs=.all, and the resulting out_da should match a hand-computed
    /// value using the rast_db channels.
    func testRasterizeIntoInterpolateDA() {
        let pos = MLXArray([
            Float(-1),  1, 0, 1,
            Float(-1), -1, 0, 1,
            Float( 1),  1, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, rastDB) = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2))

        // One scalar attr per vertex: a0=0, a1=2, a2=4 ⇒ d01=2, d02=4.
        let attr = MLXArray([Float(0), 2, 4], [1, 3, 1])
        let (_, outDA) = DiffRast.interpolate(
            attr, rast: rast, tri: tri, rastDB: rastDB, diffAttrs: .all)
        XCTAssertEqual(outDA.shape, [1, 2, 2, 2])

        // rast_db at every covered pixel = (0, 0.5, 0.5, 0). With d01=2, d02=4:
        //   dA/dx = d01 * du/dx + d02 * dv/dx = 2*0   + 4*0.5 = 2.0
        //   dA/dy = d01 * du/dy + d02 * dv/dy = 2*0.5 + 4*0   = 1.0
        let flat = outDA.asArray(Float.self)
        XCTAssertEqual(flat[0], 2.0, accuracy: 1e-5)
        XCTAssertEqual(flat[1], 1.0, accuracy: 1e-5)
    }

    // MARK: - Gradcheck (M2.3 backward)

    /// Triangle whose interior strictly contains all pixel centers AND whose
    /// edges stay >0.5 away from any of them, so 1e-3 perturbations on `pos`
    /// can't flip coverage. Crucial for FD: edge-crossing discontinuities
    /// would make finite differences disagree with the analytic gradient.
    ///
    /// Vertices: V0=(-3, 3), V1=(-3, -3), V2=(3, 0) — none of the three edges
    /// pass near pixel centers in our 2×2 image (NDC ±0.5). Distinct z per
    /// vertex so the z-gradient path is exercised.
    private func fullCoverPos() -> MLXArray {
        MLXArray([
            Float(-3),  3, 0.0, 1,
            Float(-3), -3, 0.1, 1,
            Float( 3),  0, -0.1, 1,
        ], [1, 3, 4])
    }

    func testGradcheckRastWithoutDB() throws {
        try runRasterizeGradcheck(useDB: false)
    }

    func testGradcheckRastAndDB() throws {
        try runRasterizeGradcheck(useDB: true)
    }

    private func runRasterizeGradcheck(useDB: Bool) throws {
        let N = 1, H = 2, W = 2
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let pos0 = fullCoverPos()

        // Deterministic, distinct cotangent weights — avoids RNG-state surprises and
        // makes per-channel contributions easy to reason about. Tri-id channel (3) is
        // zeroed since it's non-differentiable.
        var wRastVals = [Float]()
        for i in 0..<(N * H * W * 4) {
            wRastVals.append(i % 4 == 3 ? 0.0 : Float(1 + i % 7) * 0.1)
        }
        let wRast = MLXArray(wRastVals, [N, H, W, 4])

        var wRastDBVals = [Float]()
        for i in 0..<(N * H * W * 4) {
            wRastDBVals.append(useDB ? Float(2 + i % 5) * 0.05 : 0.0)
        }
        let wRastDB = MLXArray(wRastDBVals, [N, H, W, 4])

        let loss: (MLXArray) -> MLXArray = { pos in
            let (r, db) = DiffRast.rasterize(
                pos, tri: tri, resolution: (height: H, width: W), gradDB: useDB)
            if useDB {
                return (r * wRast).sum() + (db * wRastDB).sum()
            } else {
                return (r * wRast).sum()
            }
        }

        let analytic = MLX.grad(loss)(pos0)
        analytic.eval()
        let analyticFlat = analytic.asArray(Float.self)

        let eps: Float = 1e-3
        let flat = pos0.asArray(Float.self)
        var numeric = [Float](repeating: 0, count: flat.count)
        for i in 0..<flat.count {
            var plus = flat; plus[i] += eps
            var minus = flat; minus[i] -= eps
            let lp = loss(MLXArray(plus, pos0.shape)).item(Float.self)
            let lm = loss(MLXArray(minus, pos0.shape)).item(Float.self)
            numeric[i] = (lp - lm) / (2 * eps)
        }

        for i in 0..<flat.count {
            XCTAssertEqual(analyticFlat[i], numeric[i], accuracy: 5e-3,
                           "[useDB=\(useDB)] pos elem \(i): analytic \(analyticFlat[i]) vs fd \(numeric[i])")
        }
    }

    // MARK: - Range mode

    /// Two triangles in the same `tri` buffer, different batches each render
    /// a different subrange. Verifies range-mode picks the right triangles.
    func testRangeModeSelectsPerBatchTriangles() {
        // Shared vertex buffer with 6 vertices arranged as two non-overlapping
        // triangles. tri lists triangle 0 (vertices 0-2) and triangle 1
        // (vertices 3-5). Batch 0 sees only triangle 0; batch 1 only triangle 1.
        let pos = MLXArray([
            // triangle 0: upper-left filled
            Float(-1.0),  1.0, 0, 1,
            Float(-1.0), -1.0, 0, 1,
            Float( 1.0),  1.0, 0, 1,
            // triangle 1: lower-right filled
            Float( 1.0),  1.0, 0.5, 1,
            Float(-1.0), -1.0, 0.5, 1,
            Float( 1.0), -1.0, 0.5, 1,
        ], [6, 4])
        let tri = MLXArray([
            Int32(0), 1, 2,
            Int32(3), 4, 5,
        ], [2, 3])
        let ranges = MLXArray([
            Int32(0), 1,    // batch 0: triangles [0, 1) → just tri 0
            Int32(1), 1,    // batch 1: triangles [1, 2) → just tri 1
        ], [2, 2])

        let (rast, _) = DiffRast.rasterize(
            pos, tri: tri, resolution: (height: 2, width: 2), gradDB: false, ranges: ranges)
        XCTAssertEqual(rast.shape, [2, 2, 2, 4])

        let flat = rast.asArray(Float.self)
        // Batch 0: only tri 0 (z=0) is visible → tri_id+1 must be 1 where covered.
        // Batch 1: only tri 1 (z=0.5) is visible → tri_id+1 must be 2 where covered.
        for h in 0..<2 {
            for w in 0..<2 {
                let i0 = ((0 * 2 + h) * 2 + w) * 4 + 3
                let i1 = ((1 * 2 + h) * 2 + w) * 4 + 3
                let id0 = flat[i0]
                let id1 = flat[i1]
                XCTAssertTrue(id0 == 0 || id0 == 1,
                              "batch 0 (h=\(h),w=\(w)) tri_id+1 = \(id0); only tri 0 allowed")
                XCTAssertTrue(id1 == 0 || id1 == 2,
                              "batch 1 (h=\(h),w=\(w)) tri_id+1 = \(id1); only tri 1 allowed")
            }
        }
        // At least one pixel in each batch should be covered by its assigned
        // triangle (otherwise the ranges aren't doing anything visible).
        let anyB0 = (0..<4).contains { flat[(0 * 4 + $0) * 4 + 3] == 1 }
        let anyB1 = (0..<4).contains { flat[(1 * 4 + $0) * 4 + 3] == 2 }
        XCTAssertTrue(anyB0, "batch 0 should have at least one pixel covered by tri 0")
        XCTAssertTrue(anyB1, "batch 1 should have at least one pixel covered by tri 1")
    }

    /// Sanity: feeding the rast output into `interpolate` produces non-zero attrs
    /// on covered pixels. This is the first end-to-end check across the two ops.
    func testRasterizeIntoInterpolate() {
        // Same full-screen triangle as testSingleTriangleCoversImage.
        let pos = MLXArray([
            Float(-1),  1, 0, 1,
            Float(-1), -1, 0, 1,
            Float( 1),  1, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, _) = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2), gradDB: false)

        // Per-vertex color: v0 = red, v1 = green, v2 = blue.
        let attr = MLXArray([
            Float(1), 0, 0,
            Float(0), 1, 0,
            Float(0), 0, 1,
        ], [1, 3, 3])

        let (out, _) = DiffRast.interpolate(attr, rast: rast, tri: tri)
        XCTAssertEqual(out.shape, [1, 2, 2, 3])

        // On covered pixels each RGB sums to 1 (one-hot per vertex partitions unity).
        // Empty pixels stay zero (interpolate writes 0 when tri_id+1 == 0).
        let flat = out.asArray(Float.self)
        let rastFlat = rast.asArray(Float.self)
        for pix in 0..<4 {
            let sum = flat[pix * 3] + flat[pix * 3 + 1] + flat[pix * 3 + 2]
            let covered = rastFlat[pix * 4 + 3] > 0
            XCTAssertEqual(sum, covered ? 1.0 : 0.0, accuracy: 1e-5,
                           "pixel \(pix) bary sum (covered=\(covered))")
        }
    }
}
